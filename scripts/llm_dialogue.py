#!/usr/bin/env python3
"""
LLM対話による状況抽出エンジン

LLMを使って自然な対話から5軸カテゴリを抽出し、
ProbabilityMapper → FeedbackEngine に渡す入力層モジュール。

アーキテクチャ: 2段階LLM呼び出し
  Stage 2（抽出LLM）: prompts/structured_extraction.md + few_shot_examples.json
    → ユーザーテキストから5軸+confidence付きJSONを抽出（temperature=0）
  Stage 1（対話LLM）: prompts/dialogue_followup.md
    → 低確信度の軸に対する自然なフォローアップ質問を生成

Usage:
    from llm_dialogue import LLMDialogueEngine

    engine = LLMDialogueEngine()
    if engine.is_available():
        result = engine.run_dialogue()
        if result:
            # result を ProbabilityMapper に渡す
            pass
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

from iching_cli import (
    STATE_TO_DB, ACTION_TO_DB, TRIGGER_TO_DB, PHASE_TO_DB, ENERGY_TO_DB,
    CURRENT_STATES, ENERGY_DIRECTIONS, INTENDED_ACTIONS, TRIGGER_NATURES, PHASE_STAGES,
)

__all__ = ["LLMDialogueEngine"]


# ============================================================
# 自然言語フレーズ辞書（summarize_for_user 用）
# ============================================================

STATE_PHRASES = {
    "どん底・危機": "厳しい危機的状況の中にいる",
    "停滞・閉塞": "動きが止まって先が見えない状態にいる",
    "不安定・混乱": "不安定で方向が定まらない状況にいる",
    "安定・順調": "安定した状態にある",
    "成長・発展": "成長し前に進んでいる最中にいる",
    "頂点・過剰": "ピークに達している",
    "衰退・下降": "力が落ちてきている",
    "転換期": "大きな変化の渦中にいる",
    "萌芽・準備": "新しいことが芽生え始めている段階にいる",
    "回復途上": "少しずつ回復に向かっている",
    "対立・分裂": "内部の対立に直面している",
    "依存・束縛": "何かに縛られていると感じている",
    "喪失・空虚": "大切なものを失った後の空白の中にいる",
    "未知・探索": "未知の領域を手探りで進んでいる",
    "忍耐・持久": "苦しいが踏みとどまっている",
}

ACTION_PHRASES = {
    "攻める・前進": "積極的に攻めたい",
    "守る・維持": "今あるものを守りたい",
    "待つ・忍耐": "じっと耐えて待ちたい",
    "撤退・手放す": "手放して撤退したい",
    "刷新・破壊": "一度壊して新しくしたい",
    "協調・融和": "協力して融和を図りたい",
    "分離・独立": "独立して自分の道を行きたい",
    "育成・教育": "育てて伸ばしたい",
    "蓄積・準備": "力を蓄えて準備したい",
    "表現・発信": "自分を表現して発信したい",
    "交渉・取引": "交渉で道を開きたい",
    "決断・断行": "思い切って決断したい",
    "探索・調査": "まず調べて可能性を探りたい",
    "慎重・観察": "慎重に観察したい",
    "受容・適応": "受け入れて適応したい",
    "統合・まとめ": "バラバラなものを統合したい",
    "改善・修正": "今のやり方を改善したい",
    "挑戦・冒険": "新しいことに挑戦したい",
    "奉仕・献身": "人のために尽くしたい",
    "楽しむ・喜び": "楽しみたい・喜びを味わいたい",
    "選択・判断": "重要な選択をしなければならない",
    "整理・清算": "整理して清算したい",
}

TRIGGER_PHRASES = {
    "突発的外圧": "予期せぬ外からの衝撃がきっかけ",
    "漸進的変化": "少しずつ進んできた変化の積み重ね",
    "内発的衝動": "自分の内側から湧いてきた衝動",
    "周期的転換": "自然な流れの中での転換期",
    "関係性変化": "人間関係の変化がきっかけ",
    "環境変化": "周囲の環境の変化",
    "制度・構造変化": "制度や仕組みの変更",
    "偶発的機会": "偶然訪れた機会",
}

PHASE_PHRASES = {
    "始まり": "変化はまだ始まったばかり",
    "展開初期": "動き出してまだ間もない",
    "展開中期": "変化の真っ只中にいる",
    "展開後期": "変化が大きく進み、決断の時期に近づいている",
    "頂点・転換": "変化がピークを迎えている",
    "終結": "変化の終わりが近づいている",
}

ENERGY_PHRASES = {
    "内向・収束": "エネルギーは内に向かっている",
    "外向・拡散": "エネルギーは外に向かっている",
    "上昇": "上に向かうエネルギーがある",
    "下降": "足元を固め直す方向に向かっている",
    "循環・往復": "揺れ動いている",
    "停止": "エネルギーが止まっている",
}


# ============================================================
# JSON抽出ユーティリティ
# ============================================================

def extract_json_from_response(response: str) -> Optional[dict]:
    """
    LLMレスポンスからJSON辞書を抽出する。

    複数のパース戦略を試行:
    1. レスポンス全体をそのままパース
    2. ```json ... ``` コードブロックからの抽出
    3. { ... } で囲まれた最大ブロックの抽出

    Args:
        response: LLMからのレスポンステキスト

    Returns:
        パース済みのdict、失敗時はNone
    """
    # 戦略1: そのままパース
    try:
        return json.loads(response)
    except (json.JSONDecodeError, TypeError):
        pass

    if not response:
        return None

    # 戦略2: ```json ... ``` コードブロック
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # 戦略3: { ... } ブロック
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


# ============================================================
# 類似度マッチング（フォールバック用）
# ============================================================

def _find_closest_key(label: str, mapping: dict) -> Tuple[str, bool]:
    """
    ラベルがマッピング辞書に見つからない場合、最も近いキーを返す。

    Args:
        label: 検索するラベル
        mapping: UI→DBマッピング辞書

    Returns:
        (マッチしたキー, 完全一致かどうか)
    """
    if label in mapping:
        return label, True

    # 部分一致を試行
    for key in mapping:
        if label in key or key in label:
            return key, False

    # difflib で最も近いものを選ぶ
    import difflib
    matches = difflib.get_close_matches(label, mapping.keys(), n=1, cutoff=0.4)
    if matches:
        return matches[0], False

    # 最終フォールバック: 辞書の最初のキーを返す
    first_key = next(iter(mapping))
    return first_key, False


# ============================================================
# LLMDialogueEngine クラス
# ============================================================

class LLMDialogueEngine:
    """LLM対話による状況抽出エンジン"""

    MAX_FOLLOWUP_TURNS = 2   # 最大フォローアップ回数
    CONFIDENCE_THRESHOLD = 0.60  # この閾値未満ならフォローアップ

    # 5軸のキー名一覧
    _AXIS_KEYS = [
        "current_state", "energy_direction", "intended_action",
        "trigger_nature", "phase_stage",
    ]

    # 軸名の日本語表示
    _AXIS_LABELS = {
        "current_state": "現在の状況の核心的な感覚",
        "energy_direction": "エネルギーの方向",
        "intended_action": "実際にやっていること・やりたいこと",
        "trigger_nature": "変化のきっかけ",
        "phase_stage": "変化の進行度合い",
    }

    def __init__(self):
        self.client = None  # Lazy init
        self._extraction_prompt: Optional[str] = None
        self._few_shot_examples: Optional[list] = None
        self._followup_prompt: Optional[str] = None

    # ----------------------------------------------------------
    # 初期化・チェック
    # ----------------------------------------------------------

    def is_available(self) -> bool:
        """ANTHROPIC_API_KEYが設定されているか確認する。"""
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _ensure_client(self) -> bool:
        """
        anthropic.Anthropicクライアントを初期化する。

        Returns:
            True: クライアントの初期化に成功
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

    def _load_prompts(self) -> bool:
        """
        プロンプトファイルを読み込む。

        読み込むファイル:
          - prompts/structured_extraction.md
          - prompts/few_shot_examples.json
          - prompts/dialogue_followup.md

        Returns:
            True: 全ファイルの読み込みに成功
            False: 1つ以上のファイルが見つからない
        """
        if (self._extraction_prompt is not None
                and self._few_shot_examples is not None
                and self._followup_prompt is not None):
            return True

        prompts_dir = os.path.join(_PROJECT_ROOT, "prompts")
        success = True

        # structured_extraction.md
        extraction_path = os.path.join(prompts_dir, "structured_extraction.md")
        try:
            with open(extraction_path, "r", encoding="utf-8") as f:
                self._extraction_prompt = f.read()
        except FileNotFoundError:
            warnings.warn(f"プロンプトファイルが見つかりません: {extraction_path}")
            success = False

        # few_shot_examples.json
        examples_path = os.path.join(prompts_dir, "few_shot_examples.json")
        try:
            with open(examples_path, "r", encoding="utf-8") as f:
                self._few_shot_examples = json.load(f)
        except FileNotFoundError:
            warnings.warn(f"プロンプトファイルが見つかりません: {examples_path}")
            success = False
        except json.JSONDecodeError as e:
            warnings.warn(f"few_shot_examples.json のパースに失敗: {e}")
            success = False

        # dialogue_followup.md
        followup_path = os.path.join(prompts_dir, "dialogue_followup.md")
        try:
            with open(followup_path, "r", encoding="utf-8") as f:
                self._followup_prompt = f.read()
        except FileNotFoundError:
            warnings.warn(f"プロンプトファイルが見つかりません: {followup_path}")
            success = False

        return success

    # ----------------------------------------------------------
    # Stage 2: 抽出LLM
    # ----------------------------------------------------------

    def extract_axes(self, user_text: str) -> Optional[dict]:
        """
        抽出LLM (Stage 2) を呼び出し、ユーザーテキストから5軸+confidenceを抽出する。

        structured_extraction.md のシステムプロンプトを使い、
        few_shot_examples.json の例を含めたメッセージを送信する。

        Args:
            user_text: ユーザーの状況説明テキスト（複数ターンの場合は連結済み）

        Returns:
            structured_extraction.md で定義されたJSON形式の辞書:
            {
                "current_state": {"primary": "...", "confidence": 0.0-1.0, "reasoning": "..."},
                "energy_direction": {...},
                "intended_action": {...},
                "trigger_nature": {...},
                "phase_stage": {...},
                "domain": "企業|個人|家族|国家"
            }
            失敗時はNone。
        """
        if not self._ensure_client():
            return None
        if not self._load_prompts():
            return None

        # structured_extraction.md 内の {user_text} プレースホルダーを置換
        system_with_input = self._extraction_prompt.replace("{user_text}", user_text)

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": system_with_input},
                ],
            )
            response_text = message.content[0].text
        except Exception as e:
            warnings.warn(f"抽出LLM呼び出しエラー: {e}")
            return None

        extraction = extract_json_from_response(response_text)
        if extraction is None:
            warnings.warn("抽出LLMのレスポンスからJSONを抽出できませんでした。")
            return None

        # 必須キーの存在確認
        for key in self._AXIS_KEYS:
            if key not in extraction:
                warnings.warn(f"抽出結果に '{key}' が含まれていません。")
                return None

        return extraction

    # ----------------------------------------------------------
    # 確信度評価
    # ----------------------------------------------------------

    def assess_confidence(self, extraction: dict) -> dict:
        """
        確信度を評価し、フォローアップが必要かどうかを判定する。

        Args:
            extraction: extract_axes() の戻り値

        Returns:
            {
                "action": "proceed" | "follow_up" | "broad_follow_up",
                "low_axes": [(axis_name, confidence, reasoning), ...],
                "overall_confidence": float  # 全軸の平均確信度
            }
        """
        low_axes: List[Tuple[str, float, str]] = []
        confidences: List[float] = []

        for key in self._AXIS_KEYS:
            axis_data = extraction.get(key, {})
            if isinstance(axis_data, dict):
                conf = axis_data.get("confidence", 0.0)
                reasoning = axis_data.get("reasoning", "")
            else:
                conf = 0.0
                reasoning = ""

            confidences.append(conf)

            if conf < self.CONFIDENCE_THRESHOLD:
                label = self._AXIS_LABELS.get(key, key)
                low_axes.append((label, conf, reasoning))

        overall = sum(confidences) / len(confidences) if confidences else 0.0

        if len(low_axes) == 0:
            action = "proceed"
        elif len(low_axes) >= 3:
            action = "broad_follow_up"
        else:
            action = "follow_up"

        return {
            "action": action,
            "low_axes": low_axes,
            "overall_confidence": round(overall, 3),
        }

    # ----------------------------------------------------------
    # Stage 1: 対話LLM（フォローアップ質問生成）
    # ----------------------------------------------------------

    def generate_followup_question(
        self, extraction: dict, low_axes: List[Tuple[str, float, str]], user_text: str
    ) -> Optional[str]:
        """
        対話LLM (Stage 1) を呼び出し、フォローアップ質問を生成する。

        dialogue_followup.md の {low_confidence_details} と {user_text} を置換して呼び出す。

        Args:
            extraction: extract_axes() の戻り値
            low_axes: assess_confidence() で得られた低確信度軸のリスト
                [(axis_label, confidence, reasoning), ...]
            user_text: ユーザーの元の入力テキスト

        Returns:
            フォローアップ質問文字列。失敗時はNone。
        """
        if not self._ensure_client():
            return None
        if not self._load_prompts():
            return None

        # low_confidence_details を構築
        details_lines = []
        for axis_label, conf, reasoning in low_axes:
            reason_part = f" — {reasoning}" if reasoning else ""
            details_lines.append(
                f"- 「{axis_label}」: 確信度{conf:.2f}{reason_part}"
            )
        low_confidence_details = "\n".join(details_lines)

        # プロンプトのプレースホルダー置換
        prompt = self._followup_prompt
        prompt = prompt.replace("{low_confidence_details}", low_confidence_details)
        prompt = prompt.replace("{user_text}", user_text)

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=256,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            question = message.content[0].text.strip()
            return question
        except Exception as e:
            warnings.warn(f"フォローアップLLM呼び出しエラー: {e}")
            return None

    # ----------------------------------------------------------
    # 抽出結果マージ
    # ----------------------------------------------------------

    def merge_extractions(self, original: dict, new_extraction: dict) -> dict:
        """
        元の抽出結果と追加情報からの再抽出結果をマージする。

        各軸について、confidenceが高い方を採用する。
        domainは新しい方にデータがあればそちらを優先。

        Args:
            original: 最初の extract_axes() の結果
            new_extraction: フォローアップ後の extract_axes() の結果

        Returns:
            マージされた抽出結果
        """
        merged = {}

        for key in self._AXIS_KEYS:
            orig_data = original.get(key, {})
            new_data = new_extraction.get(key, {})

            orig_conf = orig_data.get("confidence", 0.0) if isinstance(orig_data, dict) else 0.0
            new_conf = new_data.get("confidence", 0.0) if isinstance(new_data, dict) else 0.0

            if new_conf > orig_conf:
                merged[key] = new_data
            else:
                merged[key] = orig_data

        # domain: 新しい方があればそちら、なければ元のまま
        merged["domain"] = new_extraction.get("domain") or original.get("domain", "個人")

        return merged

    # ----------------------------------------------------------
    # DB用ラベル変換
    # ----------------------------------------------------------

    def extraction_to_db_labels(self, extraction: dict) -> dict:
        """
        抽出結果のprimaryラベルをDB用ラベルに変換する。

        iching_cli.py の STATE_TO_DB, ACTION_TO_DB, TRIGGER_TO_DB,
        PHASE_TO_DB, ENERGY_TO_DB を使用してマッピングする。
        primaryラベルが辞書にない場合は最も近いキーにフォールバックし、
        confidenceを0.1下げる。

        Args:
            extraction: extract_axes() またはマージ済みの抽出結果

        Returns:
            {
                "db_state": str,
                "db_energy": str or None,
                "db_action": str,
                "db_trigger": str,
                "db_phase": str,
                "ui_state": str,
                "ui_action": str,
                "overall_confidence": float,
                "domain": str,
            }
        """
        # confidence 値を集計
        confidences = []
        penalty = 0.0

        # current_state
        state_primary = extraction.get("current_state", {}).get("primary", "停滞・閉塞")
        state_key, state_exact = _find_closest_key(state_primary, STATE_TO_DB)
        db_state = STATE_TO_DB[state_key]
        if not state_exact:
            penalty += 0.1
        confidences.append(extraction.get("current_state", {}).get("confidence", 0.5))

        # energy_direction
        energy_primary = extraction.get("energy_direction", {}).get("primary", "循環・往復")
        energy_key, energy_exact = _find_closest_key(energy_primary, ENERGY_TO_DB)
        db_energy = ENERGY_TO_DB[energy_key]
        if not energy_exact:
            penalty += 0.1
        confidences.append(extraction.get("energy_direction", {}).get("confidence", 0.5))

        # intended_action
        action_primary = extraction.get("intended_action", {}).get("primary", "慎重・観察")
        action_key, action_exact = _find_closest_key(action_primary, ACTION_TO_DB)
        db_action = ACTION_TO_DB[action_key]
        if not action_exact:
            penalty += 0.1
        confidences.append(extraction.get("intended_action", {}).get("confidence", 0.5))

        # trigger_nature
        trigger_primary = extraction.get("trigger_nature", {}).get("primary", "漸進的変化")
        trigger_key, trigger_exact = _find_closest_key(trigger_primary, TRIGGER_TO_DB)
        db_trigger = TRIGGER_TO_DB[trigger_key]
        if not trigger_exact:
            penalty += 0.1
        confidences.append(extraction.get("trigger_nature", {}).get("confidence", 0.5))

        # phase_stage
        phase_primary = extraction.get("phase_stage", {}).get("primary", "展開中期")
        phase_key, phase_exact = _find_closest_key(phase_primary, PHASE_TO_DB)
        db_phase = PHASE_TO_DB[phase_key]
        if not phase_exact:
            penalty += 0.1
        confidences.append(extraction.get("phase_stage", {}).get("confidence", 0.5))

        # 全体の確信度（平均 - ペナルティ）
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.5
        overall = max(0.0, round(avg_conf - penalty, 3))

        return {
            "db_state": db_state,
            "db_energy": db_energy,
            "db_action": db_action,
            "db_trigger": db_trigger,
            "db_phase": db_phase,
            "ui_state": state_key,
            "ui_action": action_key,
            "overall_confidence": overall,
            "domain": extraction.get("domain", "個人"),
        }

    # ----------------------------------------------------------
    # ユーザー向け要約
    # ----------------------------------------------------------

    def summarize_for_user(self, extraction: dict) -> str:
        """
        ユーザー確認用の自然言語要約を生成する。

        カテゴリラベルを表示せず、状況の理解を平易な日本語で伝える。
        各軸のprimaryラベルに対応する自然言語フレーズの辞書を使ってテンプレートで構築する。

        Args:
            extraction: extract_axes() またはマージ済みの抽出結果

        Returns:
            要約テキスト（複数行）
        """
        state_label = extraction.get("current_state", {}).get("primary", "")
        action_label = extraction.get("intended_action", {}).get("primary", "")
        trigger_label = extraction.get("trigger_nature", {}).get("primary", "")
        phase_label = extraction.get("phase_stage", {}).get("primary", "")
        energy_label = extraction.get("energy_direction", {}).get("primary", "")

        state_p = STATE_PHRASES.get(state_label, "ある状況の中にいる")
        action_p = ACTION_PHRASES.get(action_label, "何かをしたいと感じている")
        trigger_p = TRIGGER_PHRASES.get(trigger_label, "何かがきっかけ")
        phase_p = PHASE_PHRASES.get(phase_label, "変化の途中にいる")
        energy_p = ENERGY_PHRASES.get(energy_label, "エネルギーが動いている")

        return (
            f"  あなたは今、{state_p}ようです。\n"
            f"  {action_p}という気持ちがあり、{energy_p}ようです。\n"
            f"  {trigger_p}で、{phase_p}のようですね。"
        )

    # ----------------------------------------------------------
    # メイン対話ループ
    # ----------------------------------------------------------

    def run_dialogue(self) -> Optional[dict]:
        """
        メイン対話ループ。

        Flow:
        1. ユーザーに状況入力を促す (input())
        2. extract_axes() で5軸抽出
        3. assess_confidence() で確信度評価
        4. 低確信度 → generate_followup_question() → input() → 再抽出
        5. merge_extractions() で結果をマージ
        6. summarize_for_user() で要約を表示
        7. ユーザーに確認 (Y/n)
        8. extraction_to_db_labels() でDB用ラベルに変換して返す

        Returns:
            extraction_to_db_labels() の戻り値。キャンセル時はNone。
        """
        if not self.is_available():
            print("  ! ANTHROPIC_API_KEY が設定されていません。")
            print("  ! LLM対話モードを使用するには環境変数を設定してください。")
            return None

        if not self._ensure_client():
            print("  ! Anthropic クライアントの初期化に失敗しました。")
            return None

        if not self._load_prompts():
            print("  ! プロンプトファイルの読み込みに失敗しました。")
            return None

        # --- Step 1: ユーザー入力 ---
        print()
        print("  あなたの状況を自由に教えてください。")
        print("  （企業経営、個人の人生、家族の問題、国家の課題 など）")
        print("  入力が終わったらEnterを2回押してください。")
        print()

        lines = []
        empty_count = 0
        try:
            while True:
                line = input()
                if line.strip() == "":
                    empty_count += 1
                    if empty_count >= 1 and lines:
                        break
                else:
                    empty_count = 0
                    lines.append(line)
        except (EOFError, KeyboardInterrupt):
            print("\n  中断しました。")
            return None

        user_text = "\n".join(lines).strip()
        if not user_text:
            print("  入力がありません。")
            return None

        # --- Step 2: 5軸抽出 ---
        print()
        print("  分析中...")

        extraction = self.extract_axes(user_text)
        if extraction is None:
            print("  ! 状況の分析に失敗しました。")
            return None

        # --- Step 3-5: 確信度評価 & フォローアップ ---
        accumulated_text = user_text
        current_extraction = extraction
        followup_count = 0

        while followup_count < self.MAX_FOLLOWUP_TURNS:
            assessment = self.assess_confidence(current_extraction)

            if assessment["action"] == "proceed":
                break

            # フォローアップ質問を生成
            question = self.generate_followup_question(
                current_extraction,
                assessment["low_axes"],
                accumulated_text,
            )

            if question is None:
                # LLM呼び出し失敗 → 現在の結果で進む
                break

            print()
            print(f"  {question}")
            print()

            try:
                followup_answer = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  中断しました。")
                return None

            if not followup_answer:
                # 空回答 → 現在の結果で進む
                break

            # 追加テキストを連結して再抽出
            accumulated_text = f"{accumulated_text}\n\n追加情報:\n{followup_answer}"

            print()
            print("  追加情報を分析中...")

            new_extraction = self.extract_axes(accumulated_text)
            if new_extraction is not None:
                current_extraction = self.merge_extractions(
                    current_extraction, new_extraction
                )

            followup_count += 1

        # --- Step 6: 要約表示 ---
        print()
        print("  ─── 状況の理解 ───")
        print()
        summary = self.summarize_for_user(current_extraction)
        print(summary)
        print()

        # 確信度情報を表示
        final_assessment = self.assess_confidence(current_extraction)
        conf_pct = final_assessment["overall_confidence"] * 100
        print(f"  （理解の確度: {conf_pct:.0f}%）")
        print()

        # --- Step 7: ユーザー確認 ---
        try:
            confirm = input("  この理解で進めてよいですか？ [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  中断しました。")
            return None

        if confirm in ("n", "no"):
            print("  キャンセルしました。")
            return None

        # --- Step 8: DB用ラベルに変換 ---
        db_labels = self.extraction_to_db_labels(current_extraction)

        return db_labels


# ============================================================
# スタンドアロン実行用
# ============================================================

def main():
    """スタンドアロン実行時のエントリポイント"""
    engine = LLMDialogueEngine()

    if not engine.is_available():
        print("ANTHROPIC_API_KEY が設定されていません。")
        print("export ANTHROPIC_API_KEY='your-api-key' を実行してください。")
        sys.exit(1)

    result = engine.run_dialogue()

    if result:
        print()
        print("  ─── DB用ラベル ───")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print()
        print("  結果なし（キャンセルまたはエラー）")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  中断しました。")
        sys.exit(0)
