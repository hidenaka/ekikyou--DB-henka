#!/usr/bin/env python3
"""
フィードバック生成エンジン

易経変化理解支援システムのMVP。入力（卦番号 + 動爻 + before_state + action_type）から
5レイヤー構造のフィードバックを生成する。

Usage:
    from feedback_engine import FeedbackEngine
    engine = FeedbackEngine()
    result = engine.generate(12, 3, "停滞・閉塞", "刷新・破壊", 0.72)
    text = engine.generate_text(12, 3, "停滞・閉塞", "刷新・破壊", 0.72, show_extended=True)

specs/feedback_output_layer.md に準拠。
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- パス設定 ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from hexagram_transformations import (
    get_cuo_gua,
    get_hexagram_name,
    get_hu_gua,
    get_trigrams,
    get_zhi_gua,
    get_zong_gua,
    hexagram_to_lines,
)
from case_search import CaseSearchEngine


# ============================================================
# 定数
# ============================================================

PHASE_NAMES = ["潜伏", "出現", "転換", "選択", "最盛", "極限"]

PHASE_DESCRIPTIONS = {
    1: "まだ表に出る時ではない。力を蓄えている段階です。",
    2: "内側で力をつけ、基礎を固めている段階です。",
    3: "内側から外側へ出る境目にあり、不安定ですが、ここを越えれば次のステージに入ります。",
    4: "新しいステージに入ったが、まだ足場が固まっていない時期です。",
    5: "最も力を発揮できる位置にいます。",
    6: "極まった状態です。これ以上は進めず、次のサイクルへの転換が近づいています。",
}

CHANGE_NATURE = {
    1: "始まりの性質が変わる。根本的な方向転換",
    2: "基盤の在り方が変わる。内部構造の転換",
    3: "転換点での判断が変わる。外への出方が変わる",
    4: "選択の質が変わる。新環境での立ち位置が変わる",
    5: "中心の在り方が変わる。リーダーシップの転換",
    6: "極限の反転。サイクルの終わりと始まり",
}

# LAYER 5: 動爻位置に基づく問い
QUESTIONS_BY_YAO = {
    1: "今、表に出ないことで蓄えているものは何ですか？",
    2: "今固めている基盤の中で、最も確かなものは何ですか？",
    3: "内側に留まるリスクと、外に出るリスク、どちらが大きいですか？",
    4: "新しいステージで、最初に手放すべきものは何ですか？",
    5: "あなたが今、最も影響を与えている相手は誰ですか？",
    6: "ここまで来て、まだ握りしめているものは何ですか？",
}

# フェーズ別デフォルト行動指針（ルールテーブルのフォールバック）
PHASE_DEFAULTS_DO_NOT = {
    1: "拙速に動くこと",
    2: "独断で進めること",
    3: "油断すること",
    4: "過去のやり方に固執すること",
    5: "傲慢になること",
    6: "しがみつくこと",
}

PHASE_DEFAULTS_DO = {
    1: "準備と観察に専念する",
    2: "信頼を築き、基盤を固める",
    3: "慎重に判断し、決断する",
    4: "新しい環境に適応する",
    5: "中庸を保ち、影響力を適切に使う",
    6: "手放し、次の段階に備える",
}

# 品質ゲート Q1: 禁止語リスト
FORBIDDEN_WORDS = ["予測", "予言", "運命", "占い", "になるでしょう", "必ず", "確実に"]

# domain (日本語) → scale (DB内部値) マッピング
DOMAIN_TO_SCALE = {
    "個人": "individual",
    "企業": "company",
    "家族": "family",
    "国家": "country",
}


# ============================================================
# FeedbackEngine
# ============================================================

def _display_width(text: str) -> int:
    """文字列の端末表示幅を返す。CJK文字は2カラム、ASCII文字は1カラム。"""
    import unicodedata
    width = 0
    for ch in text:
        if unicodedata.east_asian_width(ch) in ('F', 'W'):
            width += 2
        else:
            width += 1
    return width


def _pad_to_width(text: str, target_width: int) -> str:
    """文字列を target_width カラム幅になるように半角スペースで右パディングする。"""
    current = _display_width(text)
    padding = max(0, target_width - current)
    return text + " " * padding


class FeedbackEngine:
    """5レイヤー構造のフィードバックを生成するエンジン"""

    def __init__(self):
        """データファイルをロードし、CaseSearchEngineをインスタンス化する。"""
        # iching_texts
        iching_path = os.path.join(_PROJECT_ROOT, "data", "reference",
                                   "iching_texts_ctext_legge_ja.json")
        with open(iching_path, "r", encoding="utf-8") as f:
            self._iching = json.load(f)

        # line_positions
        lp_path = os.path.join(_PROJECT_ROOT, "data", "diagnostic",
                               "line_positions.json")
        with open(lp_path, "r", encoding="utf-8") as f:
            self._line_positions = json.load(f)

        # yao_384 (フォールバック)
        yao384_path = os.path.join(_PROJECT_ROOT, "data", "diagnostic",
                                   "yao_384.json")
        if os.path.exists(yao384_path):
            with open(yao384_path, "r", encoding="utf-8") as f:
                self._yao384 = json.load(f)
        else:
            self._yao384 = None

        # hexagram_64 (archetype / modern_interpretation)
        hex64_path = os.path.join(_PROJECT_ROOT, "data", "diagnostic",
                                  "hexagram_64.json")
        with open(hex64_path, "r", encoding="utf-8") as f:
            _hex64_raw = json.load(f)
        self._hex64 = {}
        for _name, _info in _hex64_raw["hexagrams"].items():
            self._hex64[_info["number"]] = _info

        # テンプレート
        tpl_path = os.path.join(_PROJECT_ROOT, "templates",
                                "feedback_template.txt")
        with open(tpl_path, "r", encoding="utf-8") as f:
            self._template = f.read()

        # 384爻アクションルール
        yao_rules_path = os.path.join(_PROJECT_ROOT, "data", "diagnostic",
                                      "yao_action_rules.json")
        if os.path.exists(yao_rules_path):
            with open(yao_rules_path, "r", encoding="utf-8") as f:
                self._yao_action_rules = json.load(f).get("rules", {})
        else:
            self._yao_action_rules = {}

        # CaseSearchEngine
        self._case_search = CaseSearchEngine()

    # ------------------------------------------------------------------
    # 公開API
    # ------------------------------------------------------------------

    def generate(
        self,
        hexagram_number: int,
        yao_position: int,
        before_state: str,
        action_type: str,
        mapping_confidence: float = 0.7,
        domain: str = None,
    ) -> dict:
        """5レイヤー構造のフィードバックを dict 形式で返す。

        Args:
            domain: ユーザーのドメイン (例: "個人", "企業", "家族", "国家")。
                    指定時、事例検索をそのスケールでフィルタリングする。
        """
        # domain → scale 変換
        scale = DOMAIN_TO_SCALE.get(domain) if domain else None

        layer1 = self._build_layer1(hexagram_number, yao_position)
        layer2 = self._build_layer2(hexagram_number, yao_position)
        layer3 = self._build_layer3(hexagram_number)
        layer4 = self._build_layer4(hexagram_number, yao_position,
                                    before_state, action_type, scale=scale)
        layer5 = self._build_layer5(yao_position,
                                    layer2["resulting_hexagram"]["id"])

        result = {
            "version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mapping_confidence": mapping_confidence,
            "layer1_current": layer1,
            "layer2_direction": layer2,
            "layer3_hidden": layer3,
            "layer4_reference": layer4,
            "layer5_question": layer5,
        }

        # 品質ゲート
        self._run_quality_gates(result)

        return result

    def generate_text(
        self,
        hexagram_number: int,
        yao_position: int,
        before_state: str,
        action_type: str,
        mapping_confidence: float = 0.7,
        show_extended: bool = False,
        domain: str = None,
    ) -> str:
        """CLI 表示用のテキストを返す。"""
        data = self.generate(hexagram_number, yao_position,
                             before_state, action_type, mapping_confidence,
                             domain=domain)
        return self._render_text(data, show_extended)

    # ------------------------------------------------------------------
    # 5点固定出力ビューモデル
    # ------------------------------------------------------------------

    def build_5point_view(self, base_feedback: dict,
                          hexagram_number: int, yao_position: int) -> dict:
        """
        5レイヤーフィードバックから5点固定出力ビューモデルを構築する。

        5点: 現在地 / 今やるな / 今やれ / 反対視点 / 類似事例

        Args:
            base_feedback: generate() の返り値
            hexagram_number: 卦番号
            yao_position: 爻位

        Returns:
            5点固定出力のdict
        """
        l1 = base_feedback["layer1_current"]
        l2 = base_feedback["layer2_direction"]
        l3 = base_feedback["layer3_hidden"]
        l4 = base_feedback["layer4_reference"]

        # --- 1. 現在地（局面の圧縮）---
        hex_info = l1["hexagram"]
        cl = l1["changing_line"]
        hex64_info = self._hex64.get(hexagram_number, {})

        current_position = {
            "title": f"{hex_info['name']}・第{cl['position']}爻「{cl['phase']}」",
            "summary": l1.get("archetype", hex64_info.get("archetype", "")),
            "phase_description": cl["phase_description"],
            "yao_message": cl["yao_text_modern_ja"],
            "hexagram_id": hexagram_number,
            "hexagram_name": hex_info["name"],
            "yao_position": yao_position,
            "phase": cl["phase"],
            "confidence": base_feedback.get("mapping_confidence", 0.0),
        }

        # --- 2. 今やるな / 3. 今やれ ---
        rule_key = f"{hexagram_number}_{yao_position}"
        rule = self._yao_action_rules.get(rule_key, {})

        do_not = {
            "action": rule.get("do_not", PHASE_DEFAULTS_DO_NOT.get(yao_position, "")),
            "reason": rule.get("condition", cl["phase_description"]),
            "strength": rule.get("strength", "moderate"),
        }

        do_action = {
            "action": rule.get("do", PHASE_DEFAULTS_DO.get(yao_position, "")),
            "reason": rule.get("condition", cl["phase_description"]),
            "strength": rule.get("strength", "moderate"),
            "direction_hint": self._first_sentence(
                l2.get("resulting_judgment_modern_ja", "")
            ),
        }

        # --- 4. 反対視点（綜卦を主、錯卦を補助）---
        zong = l3["inverted"]
        cuo = l3["complementary"]
        opposite_view = {
            "primary": {
                "type": "綜卦",
                "label": "立場を逆にすると",
                "hexagram_name": zong["hexagram_name"],
                "hexagram_id": zong["hexagram_id"],
                "insight": self._first_sentence(zong["judgment_modern_ja"]),
            },
            "secondary": {
                "type": "錯卦",
                "label": "全てを反転すると",
                "hexagram_name": cuo["hexagram_name"],
                "hexagram_id": cuo["hexagram_id"],
                "insight": self._first_sentence(cuo["judgment_modern_ja"]),
            },
        }

        # --- 5. 類似事例 ---
        dist = l4["conditional_distribution"]
        similar = l4["similar_cases"]
        reference_cases = {
            "corpus_n": len(self._case_search.cases),
            "matched_n": dist["total_n"],
            "scale": dist["condition"].get("scale"),
            "confidence_note": dist["confidence_note"],
            "top_outcome": (
                dist["distribution"][0] if dist["distribution"] else None
            ),
            "cases": [
                {
                    "name": c["target_name"],
                    "period": c["period"],
                    "flow": f"{c['before_state']} → {c['action_type']} → {c['after_state']}",
                    "summary": c.get("story_summary", ""),
                    "basis": c["similarity_basis"],
                }
                for c in similar
            ],
        }

        # --- 品質ゲート ---
        warnings = []
        # QG1: 禁止と推奨が同一の行動を指していないか
        if do_not["action"] and do_action["action"]:
            if do_not["action"] == do_action["action"]:
                warnings.append("do_not と do が同一の行動を指しています")
        # QG2: 各ポイントに内容があるか
        if not current_position.get("summary"):
            warnings.append("現在地のサマリーが空です")
        if not do_not.get("action"):
            warnings.append("今やるなが空です")
        if not do_action.get("action"):
            warnings.append("今やれが空です")
        # QG3: 類似事例の母集団情報が正しいか
        if reference_cases["corpus_n"] <= 0:
            warnings.append("事例母集団が0件です")

        return {
            "format": "5point",
            "version": "1.0",
            "generated_at": base_feedback.get("generated_at", ""),
            "point1_current_position": current_position,
            "point2_do_not": do_not,
            "point3_do": do_action,
            "point4_opposite_view": opposite_view,
            "point5_reference_cases": reference_cases,
            "quality_warnings": warnings,
        }

    def generate_5point(
        self,
        hexagram_number: int,
        yao_position: int,
        before_state: str,
        action_type: str,
        mapping_confidence: float = 0.7,
        domain: str = None,
    ) -> dict:
        """5点固定出力を直接生成する。"""
        base = self.generate(
            hexagram_number, yao_position, before_state, action_type,
            mapping_confidence, domain=domain,
        )
        return self.build_5point_view(base, hexagram_number, yao_position)

    # ------------------------------------------------------------------
    # LAYER 1: 現在地の読み解き
    # ------------------------------------------------------------------

    def _build_layer1(self, hex_id: int, yao: int) -> dict:
        hex_name = get_hexagram_name(hex_id)
        lower, upper = get_trigrams(hex_id)
        lines = hexagram_to_lines(hex_id)

        hex_data = self._iching["hexagrams"].get(str(hex_id), {})
        judgment_ja = hex_data.get("judgment", {}).get("modern_ja", "")

        line_data = hex_data.get("lines", {}).get(str(yao), {})
        yao_text = line_data.get("modern_ja", "")
        xiang_text = line_data.get("xiang", {}).get("modern_ja", "")

        # フォールバック: yao_384
        if not yao_text and self._yao384:
            yao_text = self._get_yao384_text(hex_id, yao)

        phase_name = PHASE_NAMES[yao - 1]
        phase_desc = PHASE_DESCRIPTIONS[yao]

        hex64_info = self._hex64.get(hex_id, {})

        return {
            "hexagram": {
                "id": hex_id,
                "name": hex_name,
                "upper_trigram": upper,
                "lower_trigram": lower,
                "lines": lines,
                "visual": self._format_visual(lines, highlight_yao=yao),
            },
            "judgment_modern_ja": judgment_ja,
            "archetype": hex64_info.get("archetype", ""),
            "modern_interpretation": hex64_info.get("modern_interpretation", ""),
            "changing_line": {
                "position": yao,
                "phase": phase_name,
                "phase_description": phase_desc,
                "yao_text_modern_ja": yao_text,
                "xiang_modern_ja": xiang_text,
            },
            "phase_model": {
                "current_position": yao,
                "phases": PHASE_NAMES[:],
            },
        }

    # ------------------------------------------------------------------
    # LAYER 2: 変化が向かう構造（之卦）
    # ------------------------------------------------------------------

    def _build_layer2(self, hex_id: int, yao: int) -> dict:
        zhi_id = get_zhi_gua(hex_id, yao)
        zhi_name = get_hexagram_name(zhi_id)
        zhi_lower, zhi_upper = get_trigrams(zhi_id)
        zhi_lines = hexagram_to_lines(zhi_id)

        hex_name = get_hexagram_name(hex_id)
        original_lines = hexagram_to_lines(hex_id)

        # 変爻の陰陽
        original_val = original_lines[yao - 1]
        from_yin_yang = "陽" if original_val == 1 else "陰"
        to_yin_yang = "陰" if original_val == 1 else "陽"

        # 卦辞テキスト
        hex_data = self._iching["hexagrams"].get(str(hex_id), {})
        hex_judgment = hex_data.get("judgment", {}).get("modern_ja", "")

        zhi_data = self._iching["hexagrams"].get(str(zhi_id), {})
        zhi_judgment = zhi_data.get("judgment", {}).get("modern_ja", "")

        # 構造的読み解き生成
        structural_reading = self._build_structural_reading(
            hex_name, hex_judgment, zhi_name, zhi_judgment,
            yao, from_yin_yang, to_yin_yang
        )

        change_meaning = CHANGE_NATURE.get(yao, "")

        zhi_hex64_info = self._hex64.get(zhi_id, {})

        return {
            "resulting_hexagram": {
                "id": zhi_id,
                "name": zhi_name,
                "upper_trigram": zhi_upper,
                "lower_trigram": zhi_lower,
                "lines": zhi_lines,
                "archetype": zhi_hex64_info.get("archetype", ""),
            },
            "transformation": {
                "changed_line": yao,
                "from": from_yin_yang,
                "to": to_yin_yang,
                "meaning": change_meaning,
            },
            "structural_reading": structural_reading,
            "resulting_judgment_modern_ja": zhi_judgment,
        }

    def _build_structural_reading(
        self,
        hex_name: str,
        hex_judgment: str,
        zhi_name: str,
        zhi_judgment: str,
        yao: int,
        from_yy: str,
        to_yy: str,
    ) -> str:
        """之卦の構造的読み解きテンプレートを生成する。"""
        # 卦辞を短く要約（最初の一文を使用）
        hex_summary = self._first_sentence(hex_judgment)
        zhi_summary = self._first_sentence(zhi_judgment)

        text = (
            f"{hex_name}の第{yao}爻が変化すると、{zhi_name}の構造が現れます。\n\n"
            f"{hex_name}は「{hex_summary}」という状態ですが、\n"
            f"第{yao}爻の{from_yy}が極まって{to_yy}に転じることで、\n"
            f"{zhi_name}の「{zhi_summary}」という方向に構造が移ります。\n\n"
            f"これは「窮まれば変ず」（繋辞伝）の原理です。\n"
            f"今の状態が極まったとき、変化は"
            f"「{zhi_summary}」方向に向かう力学を持っています。"
        )
        return text

    # ------------------------------------------------------------------
    # LAYER 3: 見えていない力学（互卦・錯卦・綜卦）
    # ------------------------------------------------------------------

    def _build_layer3(self, hex_id: int) -> dict:
        hex_name = get_hexagram_name(hex_id)

        hu_id = get_hu_gua(hex_id)
        hu_name = get_hexagram_name(hu_id)
        hu_judgment = self._get_judgment(hu_id)

        cuo_id = get_cuo_gua(hex_id)
        cuo_name = get_hexagram_name(cuo_id)
        cuo_judgment = self._get_judgment(cuo_id)

        zong_id = get_zong_gua(hex_id)
        zong_name = get_hexagram_name(zong_id)
        zong_judgment = self._get_judgment(zong_id)

        # 互卦テンプレート
        nuclear_reading = (
            f"表面に見えている{hex_name}の内側には、{hu_name}の力学が働いています。\n"
            f"{self._first_sentence(hu_judgment)}\n"
            f"この内的構造が、今の状況を根底で形作っています。"
        )

        # 錯卦テンプレート
        comp_reading = (
            f"{hex_name}の全ての要素を反転させると、{cuo_name}が現れます。\n"
            f"これは今の状況の「影」、意識から外れている側面です。\n"
            f"{self._first_sentence(cuo_judgment)}\n"
            f"この対極を認識することで、見えていなかったものが見えるかもしれません。"
        )

        # 綜卦テンプレート
        inv_reading = (
            f"{hex_name}を上下逆さまにすると、{zong_name}になります。\n"
            f"同じ状況を、相手の立場・逆の視点から見た姿です。\n"
            f"{self._first_sentence(zong_judgment)}\n"
            f"視点を180度変えたとき、この構造が浮かび上がります。"
        )

        return {
            "depth": "extended",
            "nuclear": {
                "hexagram_id": hu_id,
                "hexagram_name": hu_name,
                "reading": nuclear_reading,
                "judgment_modern_ja": hu_judgment,
            },
            "complementary": {
                "hexagram_id": cuo_id,
                "hexagram_name": cuo_name,
                "reading": comp_reading,
                "judgment_modern_ja": cuo_judgment,
            },
            "inverted": {
                "hexagram_id": zong_id,
                "hexagram_name": zong_name,
                "reading": inv_reading,
                "judgment_modern_ja": zong_judgment,
            },
        }

    # ------------------------------------------------------------------
    # LAYER 4: 過去事例から見える分布
    # ------------------------------------------------------------------

    def _build_layer4(
        self,
        hex_id: int,
        yao: int,
        before_state: str,
        action_type: str,
        scale: str = None,
    ) -> dict:
        dist = self._case_search.get_conditional_distribution(
            before_state, action_type, scale=scale
        )

        # 八卦タグを取得（before_hex / action_hex）
        lower, upper = get_trigrams(hex_id)
        similar = self._case_search.search_similar_cases(
            hexagram_number=hex_id,
            yao_position=yao,
            before_state=before_state,
            action_type=action_type,
            before_hex=lower,
            action_hex=upper,
            limit=3,
            scale=scale,
        )

        return {
            "conditional_distribution": dist,
            "similar_cases": similar,
        }

    # ------------------------------------------------------------------
    # LAYER 5: 問いかけ
    # ------------------------------------------------------------------

    def _build_layer5(self, yao: int, zhi_gua_id: int) -> dict:
        zhi_name = get_hexagram_name(zhi_gua_id)
        question = self._generate_question(yao, zhi_name)
        phase_name = PHASE_NAMES[yao - 1]

        return {
            "question": question,
            "generation_basis": (
                f"動爻={yao}爻（{phase_name}フェーズ）、"
                f"之卦={zhi_name}（{self._first_sentence(self._get_judgment(zhi_gua_id))}）"
            ),
        }

    def _generate_question(self, yao: int, zhi_gua_name: str) -> str:
        """動爻 + 之卦に基づく問いを生成する。"""
        return QUESTIONS_BY_YAO.get(yao,
            "今の状況で、最も大切にすべきことは何ですか？")

    # ------------------------------------------------------------------
    # 品質ゲート
    # ------------------------------------------------------------------

    def _run_quality_gates(self, result: dict) -> None:
        """品質ゲート Q1-Q6 を検証する。違反時は warnings に記録。"""
        warnings = []

        # Q1: 禁止語チェック
        # caveat テキスト（「予測するものではありません」）は除外対象
        allowed_phrases = [
            "予測するものではありません",
            "「必ずこうなる」という意味ではなく",
        ]
        text_blob = json.dumps(result, ensure_ascii=False)
        for word in FORBIDDEN_WORDS:
            if word in text_blob:
                # 許容フレーズ内での出現かチェック
                is_allowed = any(
                    phrase in text_blob and word in phrase
                    for phrase in allowed_phrases
                )
                if not is_allowed:
                    warnings.append(f"Q1違反: 禁止語「{word}」が検出されました")

        # Q2: LAYER 4 の n > 0 チェック
        total_n = result.get("layer4_reference", {}).get(
            "conditional_distribution", {}).get("total_n", 0)
        if total_n == 0:
            warnings.append("Q2注意: 該当事例が0件です")

        # Q3: LAYER 2 の構造的読み解きが断定形でないこと
        reading = result.get("layer2_direction", {}).get(
            "structural_reading", "")
        for phrase in ["になります", "になるでしょう"]:
            if phrase in reading:
                warnings.append(
                    f"Q3違反: 構造的読み解きに断定形「{phrase}」が含まれています")

        # Q4: LAYER 5 に問いが存在すること
        question = result.get("layer5_question", {}).get("question", "")
        if not question:
            warnings.append("Q4違反: LAYER 5 に問いがありません")

        # Q5: 卦辞/爻辞が iching_texts から取得されていること
        judgment = result.get("layer1_current", {}).get(
            "judgment_modern_ja", "")
        if not judgment:
            warnings.append("Q5注意: 卦辞が空です")

        yao_text = result.get("layer1_current", {}).get(
            "changing_line", {}).get("yao_text_modern_ja", "")
        if not yao_text:
            warnings.append("Q5注意: 爻辞が空です")

        # Q6: mapping_confidence が表示されていること
        if "mapping_confidence" not in result:
            warnings.append("Q6違反: mapping_confidence が未設定です")

        if warnings:
            result["quality_warnings"] = warnings

    # ------------------------------------------------------------------
    # テキストレンダリング
    # ------------------------------------------------------------------

    def _render_text(self, data: dict, show_extended: bool) -> str:
        """dict データを CLI 用テキストに変換する。"""
        l1 = data["layer1_current"]
        l2 = data["layer2_direction"]
        l3 = data["layer3_hidden"]
        l4 = data["layer4_reference"]
        l5 = data["layer5_question"]

        hex_info = l1["hexagram"]
        cl = l1["changing_line"]

        lines = []

        # === ヘッダー ===
        conf_pct = int(data["mapping_confidence"] * 100)
        lines.append("━" * 61)
        lines.append(f"  変化の読み解き                          確信度: {conf_pct}%")
        lines.append("━" * 61)
        lines.append("")
        lines.append("")

        # === LAYER 1: 現在地 ===
        BOX_INNER = 61  # ボックス内部の表示幅
        lines.append("┌" + "─" * BOX_INNER + "┐")
        lines.append("│" + _pad_to_width("  現在地", BOX_INNER) + "│")
        hex_title = f"  第{hex_info['id']}卦【{hex_info['name']}】"
        lines.append("│" + _pad_to_width(hex_title, BOX_INNER) + "│")
        trigram_text = f"  上卦: {hex_info['upper_trigram']}  下卦: {hex_info['lower_trigram']}"
        lines.append("│" + _pad_to_width(trigram_text, BOX_INNER) + "│")
        lines.append("└" + "─" * BOX_INNER + "┘")
        lines.append("")

        # 爻図（動爻ハイライト付き）
        lines.append(hex_info["visual"])
        lines.append("")

        # 卦辞
        lines.append(f"  {l1['judgment_modern_ja']}")
        lines.append("")

        # フェーズバー
        yao_pos = cl["position"]
        lines.append(self._format_phase_bar(yao_pos))
        lines.append("")
        lines.append(f"  あなたは今、6段階のうち第{yao_pos}段階「{cl['phase']}」にいます。")
        lines.append(f"  {cl['phase_description']}")
        lines.append("")

        # 爻辞
        lines.append(f"  【第{yao_pos}爻のメッセージ】")
        lines.append(f"  {cl['yao_text_modern_ja']}")
        lines.append("")

        # 象伝
        if cl.get("xiang_modern_ja"):
            lines.append(f"  【第{yao_pos}爻の象伝】")
            lines.append(f"  {cl['xiang_modern_ja']}")
            lines.append("")

        lines.append("")

        # === LAYER 2: 変化が向かう構造 ===
        rh = l2["resulting_hexagram"]
        tf = l2["transformation"]
        lines.append("┌" + "─" * BOX_INNER + "┐")
        lines.append("│" + _pad_to_width("  変化が向かう構造", BOX_INNER) + "│")
        direction_text = f"  {hex_info['name']}（{hex_info['id']}）→ {rh['name']}（{rh['id']}）"
        lines.append("│" + _pad_to_width(direction_text, BOX_INNER) + "│")
        lines.append("└" + "─" * BOX_INNER + "┘")
        lines.append("")

        # 構造的読み解き
        for line in l2["structural_reading"].split("\n"):
            lines.append(f"  {line}" if line else "")
        lines.append("")

        lines.append(f"  ※ これは「必ずこうなる」という意味ではなく、")
        lines.append(f"    第{yao_pos}爻が変化した場合に現れる構造です。")
        lines.append("")

        # === LAYER 3 & 4: 折りたたみ or 展開 ===
        if show_extended:
            lines.append("")
            lines.extend(self._render_layer3_extended(l3))
            lines.append("")
            lines.extend(self._render_layer4_extended(l4))
        else:
            lines.append("")
            lines.append("  ──────────── 詳細を見る ────────────")
            lines.append("")
            nuc = l3["nuclear"]
            comp = l3["complementary"]
            inv = l3["inverted"]
            n_total = l4["conditional_distribution"]["total_n"]
            n_cases = len(l4["similar_cases"])
            lines.append(f"  ▶ 内的本質（互卦: {nuc['hexagram_name']}）")
            lines.append(f"  ▶ 対極の構造（錯卦: {comp['hexagram_name']}）")
            lines.append(f"  ▶ 逆の視点（綜卦: {inv['hexagram_name']}）")
            lines.append(f"  ▶ 過去事例の分布（{n_total}件）")
            lines.append(f"  ▶ 類似事例（{n_cases}件）")

        lines.append("")
        lines.append("")

        # === LAYER 5: 問いかけ ===
        lines.append("┌" + "─" * BOX_INNER + "┐")
        lines.append("│" + _pad_to_width("  問いかけ", BOX_INNER) + "│")
        lines.append("└" + "─" * BOX_INNER + "┘")
        lines.append("")
        lines.append(f"  {l5['question']}")
        lines.append("")
        lines.append("━" * 61)

        return "\n".join(lines)

    def _render_layer3_extended(self, l3: dict) -> List[str]:
        """LAYER 3 の展開表示"""
        lines = []

        # 互卦
        nuc = l3["nuclear"]
        lines.append(f"  ──────────── 内的本質 ────────────")
        lines.append("")
        lines.append(f"  【互卦: {nuc['hexagram_name']}（第{nuc['hexagram_id']}卦）】")
        for line in nuc["reading"].split("\n"):
            lines.append(f"  {line}" if line else "")
        lines.append("")

        # 錯卦
        comp = l3["complementary"]
        lines.append(f"  ──────────── 対極の構造 ────────────")
        lines.append("")
        lines.append(f"  【錯卦: {comp['hexagram_name']}（第{comp['hexagram_id']}卦）】")
        for line in comp["reading"].split("\n"):
            lines.append(f"  {line}" if line else "")
        lines.append("")

        # 綜卦
        inv = l3["inverted"]
        lines.append(f"  ──────────── 逆の視点 ────────────")
        lines.append("")
        lines.append(f"  【綜卦: {inv['hexagram_name']}（第{inv['hexagram_id']}卦）】")
        for line in inv["reading"].split("\n"):
            lines.append(f"  {line}" if line else "")

        return lines

    def _render_layer4_extended(self, l4: dict) -> List[str]:
        """LAYER 4 の展開表示"""
        lines = []
        dist_data = l4["conditional_distribution"]
        total_n = dist_data["total_n"]
        cond = dist_data["condition"]
        distribution = dist_data["distribution"]

        # 分布テーブル
        lines.append(f"  ──────────── 過去事例の分布 ────────────")
        lines.append("")
        lines.append(f"  ┌──────────────────────────────────────────┐")
        lines.append(f"  │  同様の状況 × 行動 の過去事例（{total_n}件）"
                     + " " * max(0, 5 - len(str(total_n))) + "│")
        lines.append(f"  │  条件: {cond['before_state']} × {cond['action_type']}"
                     + " " * max(0, 30 - len(f"{cond['before_state']} × {cond['action_type']}"))
                     + "│")
        lines.append(f"  ├──────────────────────────────────────────┤")

        # 上位5件 + その他
        shown = distribution[:5]
        shown_total = sum(d["percentage"] for d in shown)

        for d in shown:
            bar = self._format_distribution_bar_inline(d["percentage"])
            label = d["state"]
            # ラベルを最大14文字に揃える
            if len(label) > 14:
                label = label[:13] + "…"
            pct_str = f"{d['percentage']:5.1f}%"
            lines.append(f"  │  {label:<14s} {bar} {pct_str}   │")

        if len(distribution) > 5:
            other_pct = round(100.0 - shown_total, 1)
            bar = self._format_distribution_bar_inline(other_pct)
            lines.append(f"  │  {'その他':<14s} {bar} {other_pct:5.1f}%   │")

        lines.append(f"  ├──────────────────────────────────────────┤")
        lines.append(f"  │  ※ 過去事例の分布であり、                  │")
        lines.append(f"  │    あなたの結果を予測するものではありません │")
        lines.append(f"  └──────────────────────────────────────────┘")
        lines.append("")

        # 類似事例
        similar = l4["similar_cases"]
        if similar:
            lines.append(f"  ──────────── 類似事例 ────────────")
            lines.append("")
            for case in similar:
                lines.append(
                    f"  ■ {case['target_name']}（{case['period']}）")
                lines.append(
                    f"    状況: {case['before_state']} → "
                    f"行動: {case['action_type']} → "
                    f"結果: {case['after_state']}")
                if case.get("story_summary"):
                    lines.append(
                        f"    概要: {case['story_summary']}")
                lines.append(
                    f"    類似の根拠: {case['similarity_basis']}")
                lines.append("")

        return lines

    # ------------------------------------------------------------------
    # ヘルパーメソッド
    # ------------------------------------------------------------------

    def _get_judgment(self, hex_id: int) -> str:
        """卦辞 modern_ja を取得する。"""
        return (self._iching.get("hexagrams", {})
                .get(str(hex_id), {})
                .get("judgment", {})
                .get("modern_ja", ""))

    def _get_yao384_text(self, hex_id: int, yao: int) -> str:
        """yao_384.json からフォールバックテキストを取得する。"""
        if not self._yao384:
            return ""
        key = f"{hex_id}_{yao}"
        entry = self._yao384.get(key, {})
        return entry.get("interpretation", entry.get("modern_ja", ""))

    def _first_sentence(self, text: str) -> str:
        """テキストの最初の一文を返す。句点で分割。"""
        if not text:
            return ""
        # 句点で区切って最初の文を返す
        for sep in ["。", "．", ". "]:
            if sep in text:
                return text.split(sep)[0]
        return text

    def _format_visual(self, lines: List[int], highlight_yao: int = None) -> str:
        """卦の爻図を視覚化する。動爻をハイライトし、フェーズラベルを付ける。"""
        result = []
        result.append("    ┌─────────┐")

        for i in range(5, -1, -1):  # 上爻(index 5)から初爻(index 0)へ
            yao_num = i + 1
            line_val = lines[i]
            symbol = "━━━━━━━" if line_val == 1 else "━━ ━━━"

            phase_label = f"  {yao_num} {PHASE_NAMES[i]}"
            if highlight_yao is not None and yao_num == highlight_yao:
                marker = " ← 変爻"
            else:
                marker = ""

            result.append(f"    │ {symbol} │{phase_label}{marker}")

        result.append("    └─────────┘")
        return "\n".join(result)

    def _format_phase_bar(self, yao_position: int) -> str:
        """フェーズバーの視覚化"""
        parts = []
        for i, name in enumerate(PHASE_NAMES, 1):
            if i == yao_position:
                parts.append(f"[★{name}]")
            else:
                parts.append(f"[{name}]")

        bar_content = " → ".join(parts)
        separator = "━" * (len(bar_content) + 4)

        return (
            f"  {separator}\n"
            f"  {bar_content}\n"
            f"  {separator}"
        )

    def _format_distribution_bar_inline(self, percentage: float) -> str:
        """分布の棒グラフ (インライン版)。最大幅16文字。"""
        max_blocks = 16
        filled = int(round(percentage / 100.0 * max_blocks))
        filled = min(filled, max_blocks)
        empty = max_blocks - filled
        return "█" * filled + "░" * empty

    @staticmethod
    def _format_distribution_bar(distribution: list, total_n: int) -> str:
        """分布の棒グラフ (スタンドアロン版)。"""
        if total_n == 0:
            return "  データなし"
        lines = []
        for d in distribution[:5]:
            bar_len = 16
            filled = int(round(d["percentage"] / 100.0 * bar_len))
            bar = "█" * filled + "░" * (bar_len - filled)
            lines.append(f"  {d['state']:<14s} {bar} {d['percentage']:5.1f}%")
        return "\n".join(lines)


# ============================================================
# テスト実行
# ============================================================

if __name__ == "__main__":
    engine = FeedbackEngine()

    # テストケース: 天地否(12) 第3爻 停滞・閉塞 × 刷新・破壊
    print("=" * 61)
    print("  テスト: 天地否(12) 第3爻 / 停滞・閉塞 × 刷新・破壊")
    print("=" * 61)
    print()

    result = engine.generate(12, 3, "停滞・閉塞", "刷新・破壊", 0.72)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n" + "=" * 61 + "\n")

    text = engine.generate_text(
        12, 3, "停滞・閉塞", "刷新・破壊", 0.72, show_extended=True
    )
    print(text)

    # 品質ゲート結果
    if "quality_warnings" in result:
        print("\n--- 品質ゲート警告 ---")
        for w in result["quality_warnings"]:
            print(f"  ⚠ {w}")
    else:
        print("\n--- 品質ゲート: 全て通過 ---")
