#!/usr/bin/env python3
"""
ChangeAdviceEngine — 変化HOWアドバイスを生成するエンジン（LLM使用）

GapAnalysisEngine の構造分析結果、古典テキストの智慧、日記メタデータを統合し、
「今日から実行できる具体的な変化アドバイス」をLLMで生成する。

アーキテクチャ:
  - prompts/change_advice.md のテンプレートにデータを注入
  - LLM（claude-sonnet-4-20250514）でアドバイスJSON生成
  - 品質ゲート（禁止ワード、必須キー、動詞チェック）で検証
  - LLM不使用時はデモアドバイスを返すフォールバック

Usage:
    from change_advice_engine import ChangeAdviceEngine

    engine = ChangeAdviceEngine()
    if engine.is_available():
        advice = engine.generate_advice(
            hexagram_a=12,
            hexagram_g=11,
            gap_analysis=gap_result,
            diary_meta=diary_meta_dict,
        )
    else:
        advice = engine._generate_demo_advice(gap_result, diary_meta_dict)
"""

import json
import os
import re
import sys
import warnings
from typing import Dict, List, Optional, Tuple

# --- パス設定 ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from llm_dialogue import extract_json_from_response

__all__ = ["ChangeAdviceEngine"]


# ============================================================
# デモアドバイステンプレート
# ============================================================

DEMO_ADVICE = {
    "easy": {
        "core_tension": "現在地と目的地は近い距離にある。小さな意識の転換で到達できる",
        "change_approach": (
            "大きな変化は必要ない。今ある流れの中で、一つだけ視点を変えてみる。"
            "その小さな違いが、全体の景色を変えていく"
        ),
        "immediate_focus": "今日一つだけ、いつもと違う選択を意識的にする",
        "what_to_release": "「もっと大きく変わらなければ」という焦り",
        "what_to_embrace": "すでに動き始めている小さな変化に気づくこと",
        "timing_quality": (
            "微調整に適した時期。大きく動く必要はなく、繊細な感度を持つことが大切"
        ),
        "practice": (
            "今日の出来事の中で「小さいけれど良かったこと」を3つ書き出す（3分）"
        ),
    },
    "moderate": {
        "core_tension": "現在の状態と望む状態の間に、意識的に越えるべきギャップがある",
        "change_approach": (
            "一気に飛ぶのではなく、段階的に移行する。まず内側の準備を整え、"
            "次に外側の行動を変える。焦りは最大の敵になる"
        ),
        "immediate_focus": (
            "「望む状態の自分なら、今日の小さな場面でどう振る舞うか」を一つ試す"
        ),
        "what_to_release": "現状に留まるための「もっともらしい理由」",
        "what_to_embrace": "不完全でも動き出すことの価値",
        "timing_quality": (
            "準備と行動のバランスが問われる時期。動きすぎず、止まりすぎず"
        ),
        "practice": "就寝前に「理想の1日の朝」を30秒間だけ具体的に想像する",
    },
    "hard": {
        "core_tension": (
            "現在と理想の間に大きな構造的ギャップがある。根本的な転換が求められている"
        ),
        "change_approach": (
            "一度に全てを変えようとしない。最も重要な一点を見定め、"
            "そこから変化を波及させる。古い構造を壊す勇気と、"
            "新しい構造を育てる忍耐の両方が必要"
        ),
        "immediate_focus": "変化の出発点として「最も抵抗が少ない一歩」を特定する",
        "what_to_release": "「今のやり方でもなんとかなる」という幻想",
        "what_to_embrace": "不確実性の中に可能性があること",
        "timing_quality": (
            "大きな転換期。痛みを伴うが、先延ばしにするほど難しくなる"
        ),
        "practice": (
            "「もし何の制約もなかったら、明日何をする？」と自分に問いかけ、"
            "答えを1文で書く（2分）"
        ),
    },
}


# ============================================================
# 必須出力キー
# ============================================================

_REQUIRED_KEYS = [
    "core_tension",
    "change_approach",
    "immediate_focus",
    "what_to_release",
    "what_to_embrace",
    "timing_quality",
    "practice",
]


# ============================================================
# ChangeAdviceEngine クラス
# ============================================================

class ChangeAdviceEngine:
    """変化HOWアドバイスを生成するエンジン（LLM使用）"""

    # 品質ゲート禁止ワード
    PROHIBITED_WORDS = [
        "頑張って", "頑張り", "ファイト", "応援して",
        "必ず", "絶対に", "間違いなく",     # 予測的表現
        "卦", "爻", "八卦", "易経",          # 専門用語
        "陰陽", "運気", "運勢", "占い",      # 占い用語
    ]

    def __init__(self, texts_path: Optional[str] = None):
        """初期化。古典テキストとプロンプトテンプレートをロードする。

        Args:
            texts_path: iching_texts_ctext_legge_ja.json のパス。
                        None の場合はデフォルトパスを使用。
        """
        self.client = None  # Lazy init
        self._prompt_template: Optional[str] = None
        self._texts: dict = {}

        # --- 古典テキストの読み込み ---
        default_texts_path = os.path.join(
            _PROJECT_ROOT, "data", "reference", "iching_texts_ctext_legge_ja.json"
        )
        path = texts_path or default_texts_path
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._texts = data.get("hexagrams", {})
            except (json.JSONDecodeError, KeyError) as e:
                warnings.warn(f"古典テキストの読み込みに失敗: {e}")
        else:
            warnings.warn(f"古典テキストファイルが見つかりません: {path}")

        # --- プロンプトテンプレートの読み込み ---
        prompt_path = os.path.join(_PROJECT_ROOT, "prompts", "change_advice.md")
        if os.path.isfile(prompt_path):
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    self._prompt_template = f.read()
            except IOError as e:
                warnings.warn(f"プロンプトテンプレートの読み込みに失敗: {e}")
        else:
            warnings.warn(f"プロンプトファイルが見つかりません: {prompt_path}")

    # ----------------------------------------------------------
    # 公開API
    # ----------------------------------------------------------

    def is_available(self) -> bool:
        """ANTHROPIC_API_KEYが設定されているか確認する。"""
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def generate_advice(
        self,
        hexagram_a: int,
        hexagram_g: int,
        gap_analysis: dict,
        diary_meta: dict,
        yao_position: int = 3,
    ) -> Optional[dict]:
        """変化アドバイスを生成する。

        LLMが利用可能な場合はLLMで生成し、品質ゲートを通す。
        LLMが利用不可、またはLLM生成が失敗した場合はデモアドバイスを返す。

        Args:
            hexagram_a: 本卦の番号 (1-64)
            hexagram_g: 目標卦の番号 (1-64)
            gap_analysis: GapAnalysisEngine.analyze() の出力辞書
            diary_meta: diary_meta辞書。キー:
                        emotional_tone, gap_clarity, key_tension, domain
            yao_position: アクティブな爻の位置 (1-6)。
                          タイミング文脈に使用。デフォルトは3（中爻）

        Returns:
            アドバイス辞書（7キー）。LLM失敗時はデモアドバイス。
            致命的エラー時のみNone。
        """
        # 入力バリデーション
        if not isinstance(hexagram_a, int) or not (1 <= hexagram_a <= 64):
            warnings.warn(f"hexagram_a が不正: {hexagram_a}")
            return None
        if not isinstance(hexagram_g, int) or not (1 <= hexagram_g <= 64):
            warnings.warn(f"hexagram_g が不正: {hexagram_g}")
            return None
        if not isinstance(yao_position, int) or not (1 <= yao_position <= 6):
            yao_position = 3  # デフォルトにフォールバック

        # LLM利用不可 → デモアドバイス
        if not self.is_available() or self._prompt_template is None:
            return self._generate_demo_advice(gap_analysis, diary_meta)

        # LLMクライアント初期化
        if not self._ensure_client():
            return self._generate_demo_advice(gap_analysis, diary_meta)

        # プロンプト構築
        prompt = self._build_prompt(
            hexagram_a, hexagram_g, gap_analysis, diary_meta, yao_position
        )
        if prompt is None:
            return self._generate_demo_advice(gap_analysis, diary_meta)

        # LLM呼び出し
        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                temperature=0.4,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            response_text = message.content[0].text
        except Exception as e:
            warnings.warn(f"LLM呼び出しエラー: {e}")
            return self._generate_demo_advice(gap_analysis, diary_meta)

        # JSON抽出
        advice = extract_json_from_response(response_text)
        if advice is None:
            warnings.warn("LLMレスポンスからJSONを抽出できませんでした。")
            return self._generate_demo_advice(gap_analysis, diary_meta)

        # 品質ゲート
        is_valid, issues = self._validate_advice(advice)
        if not is_valid:
            warnings.warn(
                f"LLM生成アドバイスが品質ゲートを通過しませんでした: "
                f"{'; '.join(issues)}"
            )
            # 部分的に修正可能か試行
            advice = self._attempt_repair(advice, issues)
            is_valid_after, remaining_issues = self._validate_advice(advice)
            if not is_valid_after:
                warnings.warn(
                    f"修復後も品質ゲート不通過。デモアドバイスにフォールバック: "
                    f"{'; '.join(remaining_issues)}"
                )
                return self._generate_demo_advice(gap_analysis, diary_meta)

        return advice

    # ----------------------------------------------------------
    # テキスト取得
    # ----------------------------------------------------------

    def _get_hexagram_text(
        self, hexagram_number: int, yao_position: Optional[int] = None
    ) -> str:
        """古典テキストから日本語要約を取得する。

        卦の judgment.modern_ja と tuan.modern_ja を結合し、
        yao_position が指定されていれば対応する爻のテキストも追加する。

        Args:
            hexagram_number: 卦番号 (1-64)
            yao_position: 爻の位置 (1-6)。Noneなら卦辞のみ

        Returns:
            フォーマットされた日本語テキスト。データ未取得時は空文字列。
        """
        hex_key = str(hexagram_number)
        hex_data = self._texts.get(hex_key)
        if hex_data is None:
            return ""

        parts: List[str] = []

        # 卦名
        local_name = hex_data.get("local_name", "")
        if local_name:
            parts.append(f"【{local_name}】")

        # 卦辞 (judgment)
        judgment = hex_data.get("judgment", {})
        judgment_ja = judgment.get("modern_ja", "")
        if judgment_ja:
            parts.append(f"卦辞: {judgment_ja}")

        # 彖伝 (tuan)
        tuan = hex_data.get("tuan", {})
        tuan_ja = tuan.get("modern_ja", "")
        if tuan_ja:
            parts.append(f"彖伝: {tuan_ja}")

        # 大象 (xiang)
        xiang = hex_data.get("xiang", {})
        xiang_ja = xiang.get("modern_ja", "")
        if xiang_ja:
            parts.append(f"大象: {xiang_ja}")

        # 爻辞
        if yao_position is not None:
            lines = hex_data.get("lines", {})
            yao_key = str(yao_position)
            yao_data = lines.get(yao_key)
            if yao_data:
                yao_ja = yao_data.get("modern_ja", "")
                if yao_ja:
                    parts.append(f"第{yao_position}爻: {yao_ja}")
                # 爻の小象
                yao_xiang = yao_data.get("xiang", {})
                yao_xiang_ja = yao_xiang.get("modern_ja", "")
                if yao_xiang_ja:
                    parts.append(f"第{yao_position}爻の象: {yao_xiang_ja}")

        return "\n".join(parts)

    # ----------------------------------------------------------
    # ギャップ分析フォーマット
    # ----------------------------------------------------------

    def _format_gap_summary(self, gap_analysis: dict) -> str:
        """GapAnalysisEngine.analyze() の出力をプロンプト注入用にフォーマットする。

        Args:
            gap_analysis: GapAnalysisEngine.analyze() の戻り値

        Returns:
            人間が読めるフォーマットされた文字列。
        """
        lines: List[str] = []

        # 難易度
        difficulty = gap_analysis.get("difficulty", "moderate")
        score = gap_analysis.get("difficulty_score", 0.5)
        lines.append(f"難易度: {difficulty} (スコア {score:.2f})")

        # ハミング距離
        hamming = gap_analysis.get("hamming_distance", 0)
        lines.append(f"変化の大きさ: {hamming}/6 爻が異なる")

        # 変爻位置
        changing = gap_analysis.get("changing_lines", [])
        if changing:
            changing_str = ", ".join(str(c) for c in changing)
            lines.append(f"変爻位置: {changing_str}")
        else:
            lines.append("変爻位置: なし（同一卦）")

        # 構造的関係
        relationship = gap_analysis.get("structural_relationship", "none")
        rel_labels = {
            "identical": "同一（変化なし）",
            "cuo_gua": "錯卦（全爻反転 — 対極の関係）",
            "zong_gua": "綜卦（上下反転 — 視点の転換）",
            "hu_gua": "互卦（核卦 — 内なる構造）",
            "none": "特殊な構造的関係なし",
        }
        if relationship.startswith("zhi_gua"):
            rel_display = f"之卦（1爻変 — 直接的な変化）: {relationship}"
        else:
            rel_display = rel_labels.get(relationship, relationship)
        lines.append(f"構造的関係: {rel_display}")

        # 相性
        compat = gap_analysis.get("compatibility")
        if compat:
            compat_type = compat.get("type", "")
            compat_score = compat.get("score", "")
            compat_summary = compat.get("summary", "")
            lines.append(f"相性: {compat_type} (スコア {compat_score})")
            if compat_summary:
                lines.append(f"相性概要: {compat_summary}")

        # 中間経路
        paths = gap_analysis.get("intermediate_paths", [])
        if paths:
            lines.append("中間経路候補:")
            for i, p in enumerate(paths, 1):
                name = p.get("name", "?")
                role = p.get("role", "")
                lines.append(f"  {i}. {name} — {role}")

        # 八卦変化
        tri = gap_analysis.get("trigram_changes", {})
        lower = tri.get("lower", {})
        upper = tri.get("upper", {})
        if lower.get("changed"):
            lines.append(
                f"下卦（内面）: {lower.get('from', '?')} → {lower.get('to', '?')}（変化あり）"
            )
        if upper.get("changed"):
            lines.append(
                f"上卦（外面）: {upper.get('from', '?')} → {upper.get('to', '?')}（変化あり）"
            )

        return "\n".join(lines)

    # ----------------------------------------------------------
    # 品質ゲート
    # ----------------------------------------------------------

    def _validate_advice(self, advice: dict) -> Tuple[bool, List[str]]:
        """品質ゲートによるアドバイスの検証。

        チェック項目:
          1. 7つの必須キーがすべて存在する
          2. どのフィールドにも禁止ワードが含まれない
          3. immediate_focus が動詞的表現で終わる
          4. practice が具体性を持つ（時間参照またはアクション動詞を含む）
          5. どのフィールドも空でない

        Args:
            advice: LLMが生成したアドバイス辞書

        Returns:
            (is_valid, issues) — issues は不合格理由のリスト
        """
        issues: List[str] = []

        if not isinstance(advice, dict):
            return False, ["アドバイスがdict型ではありません"]

        # 1. 必須キーチェック
        for key in _REQUIRED_KEYS:
            if key not in advice:
                issues.append(f"必須キー '{key}' が欠落")

        # 以降のチェックは存在するキーのみ
        # 5. 空フィールドチェック
        for key in _REQUIRED_KEYS:
            value = advice.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                issues.append(f"'{key}' が空です")

        # 2. 禁止ワードチェック
        for key in _REQUIRED_KEYS:
            value = advice.get(key)
            if not isinstance(value, str):
                continue
            for word in self.PROHIBITED_WORDS:
                if word in value:
                    issues.append(f"'{key}' に禁止ワード「{word}」が含まれています")

        # 3. immediate_focus の動詞チェック
        focus = advice.get("immediate_focus", "")
        if isinstance(focus, str) and focus.strip():
            # 日本語の動詞的語尾パターン
            verb_endings = (
                "する", "してみる", "してみよう", "しよう",
                "く", "む", "る", "す", "つ", "ぬ", "ぶ",
                "げる", "ける", "せる", "てる", "ねる", "べる", "める", "れる",
                "に気づく", "を見つける", "を試す", "を意識する",
            )
            if not any(focus.rstrip("。、！!？?）)」』】") .endswith(end) for end in verb_endings):
                issues.append(
                    "immediate_focus が動詞的表現で終わっていません"
                )

        # 4. practice の具体性チェック
        practice = advice.get("practice", "")
        if isinstance(practice, str) and practice.strip():
            # 時間参照の存在チェック
            time_patterns = [
                r"\d+分", r"\d+秒",
                r"（\d", r"\(\d",
                "1分", "2分", "3分", "5分",
                "30秒", "10秒",
            ]
            has_time = any(
                re.search(p, practice) for p in time_patterns
            )
            # アクション動詞の存在チェック
            action_words = [
                "書く", "書き出す", "メモする", "想像する", "思い浮かべる",
                "問いかけ", "深呼吸", "目を閉じ", "声に出", "数える",
                "見つけ", "気づ", "振り返", "思い出",
            ]
            has_action = any(word in practice for word in action_words)

            if not has_time and not has_action:
                issues.append(
                    "practice に時間の目安またはアクション動詞が含まれていません"
                )

        is_valid = len(issues) == 0
        return is_valid, issues

    # ----------------------------------------------------------
    # 部分修復
    # ----------------------------------------------------------

    def _attempt_repair(self, advice: dict, issues: List[str]) -> dict:
        """品質ゲート不合格のアドバイスを部分的に修復する。

        修復可能なケース:
          - 欠落キー → デモアドバイスの値で補完
          - 空フィールド → デモアドバイスの値で補完
          - 禁止ワード → 除去を試行

        修復不可能なケース:
          - immediate_focus の動詞チェック（LLMの出力を書き換えるのは危険）

        Args:
            advice: LLM生成のアドバイス辞書
            issues: _validate_advice で検出された問題リスト

        Returns:
            修復済みのアドバイス辞書（元のオブジェクトを変更する場合あり）
        """
        # デモアドバイスをフォールバックソースとして取得
        demo = DEMO_ADVICE.get("moderate", {})

        # 欠落キー / 空フィールド の補完
        for key in _REQUIRED_KEYS:
            if key not in advice or not advice.get(key, "").strip():
                advice[key] = demo.get(key, "")

        # 禁止ワードの除去
        for key in _REQUIRED_KEYS:
            value = advice.get(key, "")
            if not isinstance(value, str):
                continue
            for word in self.PROHIBITED_WORDS:
                if word in value:
                    value = value.replace(word, "")
            advice[key] = value.strip()

        return advice

    # ----------------------------------------------------------
    # デモアドバイス
    # ----------------------------------------------------------

    def _generate_demo_advice(
        self, gap_analysis: dict, diary_meta: dict
    ) -> dict:
        """LLM不使用時のデモアドバイスを生成する。

        gap_analysisのdifficulty値に基づいてテンプレートを選択し、
        diary_metaのkey_tensionがあればcore_tensionに反映する。

        Args:
            gap_analysis: GapAnalysisEngine.analyze() の出力辞書
            diary_meta: diary_meta辞書

        Returns:
            アドバイス辞書（7キー）
        """
        difficulty = gap_analysis.get("difficulty", "moderate")
        if difficulty not in DEMO_ADVICE:
            difficulty = "moderate"

        # テンプレートのコピーを返す（元データを汚さない）
        advice = dict(DEMO_ADVICE[difficulty])

        # diary_metaのkey_tensionが具体的なら反映
        key_tension = ""
        if isinstance(diary_meta, dict):
            key_tension = diary_meta.get("key_tension", "")
        if key_tension and len(key_tension) > 5:
            advice["core_tension"] = key_tension

        return advice

    # ----------------------------------------------------------
    # プロンプト構築
    # ----------------------------------------------------------

    def _build_prompt(
        self,
        hexagram_a: int,
        hexagram_g: int,
        gap_analysis: dict,
        diary_meta: dict,
        yao_position: int,
    ) -> Optional[str]:
        """プロンプトテンプレートにデータを注入する。

        Args:
            hexagram_a: 本卦の番号
            hexagram_g: 目標卦の番号
            gap_analysis: ギャップ分析結果
            diary_meta: 日記メタデータ
            yao_position: 爻の位置

        Returns:
            完成したプロンプト文字列。テンプレート未読込時はNone。
        """
        if self._prompt_template is None:
            return None

        # 卦名の取得
        hex_a_info = gap_analysis.get("hexagram_a", {})
        hex_g_info = gap_analysis.get("hexagram_g", {})
        name_a = hex_a_info.get("name", f"第{hexagram_a}卦")
        name_g = hex_g_info.get("name", f"第{hexagram_g}卦")

        # テキスト取得
        text_a = self._get_hexagram_text(hexagram_a, yao_position)
        text_g = self._get_hexagram_text(hexagram_g)

        # ギャップ分析要約
        gap_summary = self._format_gap_summary(gap_analysis)

        # diary_meta JSON
        diary_meta_json = json.dumps(
            diary_meta, ensure_ascii=False, indent=2
        )

        # key_tension
        key_tension = ""
        if isinstance(diary_meta, dict):
            key_tension = diary_meta.get("key_tension", "")

        # プレースホルダー置換
        prompt = self._prompt_template
        prompt = prompt.replace("{hexagram_a_name}", name_a)
        prompt = prompt.replace("{hexagram_a_number}", str(hexagram_a))
        prompt = prompt.replace("{hexagram_g_name}", name_g)
        prompt = prompt.replace("{hexagram_g_number}", str(hexagram_g))
        prompt = prompt.replace("{gap_analysis_summary}", gap_summary)
        prompt = prompt.replace("{hexagram_a_text}", text_a or "（テキストなし）")
        prompt = prompt.replace("{hexagram_g_text}", text_g or "（テキストなし）")
        prompt = prompt.replace("{diary_meta}", diary_meta_json)
        prompt = prompt.replace("{key_tension}", key_tension or "（不明）")

        return prompt

    # ----------------------------------------------------------
    # LLMクライアント管理
    # ----------------------------------------------------------

    def _ensure_client(self) -> bool:
        """anthropic.Anthropicクライアントを初期化する。

        Returns:
            True: 初期化成功
            False: APIキー未設定またはSDK未インストール
        """
        if self.client is not None:
            return True

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return False

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            return True
        except ImportError:
            warnings.warn(
                "anthropic SDK がインストールされていません。"
                "pip install anthropic を実行してください。"
            )
            return False
        except Exception as e:
            warnings.warn(f"Anthropicクライアント初期化エラー: {e}")
            return False


# ============================================================
# CLI（スタンドアロン実行用）
# ============================================================

def main():
    """デモ実行: GapAnalysisEngine と組み合わせたアドバイス生成"""
    import argparse

    parser = argparse.ArgumentParser(
        description="ChangeAdviceEngine — 変化HOWアドバイス生成"
    )
    parser.add_argument(
        "--from", dest="hex_a", type=int, default=12,
        help="本卦の番号 (1-64)。デフォルト: 12（天地否）"
    )
    parser.add_argument(
        "--to", dest="hex_g", type=int, default=11,
        help="目標卦の番号 (1-64)。デフォルト: 11（地天泰）"
    )
    parser.add_argument(
        "--yao", type=int, default=3,
        help="爻の位置 (1-6)。デフォルト: 3"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="LLMを使わずデモアドバイスを出力"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="JSON形式で出力"
    )
    args = parser.parse_args()

    # GapAnalysisEngine のインポートと実行
    try:
        from gap_analysis_engine import GapAnalysisEngine
    except ImportError:
        print("エラー: gap_analysis_engine.py が見つかりません。")
        print("scripts/ ディレクトリにファイルが存在するか確認してください。")
        sys.exit(1)

    gap_engine = GapAnalysisEngine()
    gap_result = gap_engine.analyze(args.hex_a, args.hex_g)

    # デモ用diary_meta
    demo_diary_meta = {
        "emotional_tone": "mixed",
        "gap_clarity": 0.65,
        "key_tension": "現状の安定を保ちたい気持ちと、変化への渇望が拮抗している",
        "domain": "個人",
    }

    # アドバイス生成
    advice_engine = ChangeAdviceEngine()

    if args.demo or not advice_engine.is_available():
        if not args.demo:
            print("  (!) ANTHROPIC_API_KEY 未設定。デモモードで実行します。")
            print()
        advice = advice_engine._generate_demo_advice(gap_result, demo_diary_meta)
    else:
        advice = advice_engine.generate_advice(
            hexagram_a=args.hex_a,
            hexagram_g=args.hex_g,
            gap_analysis=gap_result,
            diary_meta=demo_diary_meta,
            yao_position=args.yao,
        )

    if advice is None:
        print("  エラー: アドバイスの生成に失敗しました。")
        sys.exit(1)

    if args.json:
        print(json.dumps(advice, ensure_ascii=False, indent=2))
    else:
        _print_advice(advice, args.hex_a, args.hex_g, gap_result)


def _print_advice(advice: dict, hex_a: int, hex_g: int, gap_result: dict) -> None:
    """アドバイスを人間が読める形式で出力する。"""
    a_name = gap_result.get("hexagram_a", {}).get("name", f"第{hex_a}卦")
    g_name = gap_result.get("hexagram_g", {}).get("name", f"第{hex_g}卦")
    difficulty = gap_result.get("difficulty", "?")

    print("=" * 64)
    print(f"  変化アドバイス: {a_name} → {g_name}")
    print(f"  難易度: {difficulty}")
    print("=" * 64)

    labels = {
        "core_tension": "核心的テンション",
        "change_approach": "変化のアプローチ",
        "immediate_focus": "今日の焦点",
        "what_to_release": "手放すもの",
        "what_to_embrace": "受け入れるもの",
        "timing_quality": "今の時期の質",
        "practice": "今夜のプラクティス",
    }

    for key in _REQUIRED_KEYS:
        label = labels.get(key, key)
        value = advice.get(key, "")
        print()
        print(f"  [{label}]")
        print(f"  {value}")

    print()
    print("=" * 64)

    # 品質ゲート結果
    engine = ChangeAdviceEngine()
    is_valid, issues = engine._validate_advice(advice)
    if is_valid:
        print("  品質ゲート: PASS")
    else:
        print(f"  品質ゲート: FAIL ({len(issues)} 件)")
        for issue in issues:
            print(f"    - {issue}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  中断しました。")
        sys.exit(0)
