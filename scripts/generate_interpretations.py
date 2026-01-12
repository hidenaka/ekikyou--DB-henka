#!/usr/bin/env python3
"""
多視点卦解釈生成スクリプト v2

各事例に対して、本体データの卦(yao_analysis.before_hexagram_id)を使用し、
Primary解釈として採用。互卦・覆卦・錯卦を追加解釈として生成する。

新設計:
- Primary: yao_analysis.before_hexagram_idをそのまま使用
- Nuclear（互卦）: 内なる構造を表す卦
- Inverted（覆卦）: 上下反転した視点
- Complementary（錯卦）: 陰陽反転した対極的視点

使用法:
    python3 scripts/generate_interpretations.py              # 通常実行
    python3 scripts/generate_interpretations.py --dry-run    # ドライラン（保存しない）
    python3 scripts/generate_interpretations.py --force      # 既存の解釈を上書き
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# スクリプトのディレクトリを取得
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# データパス
CASES_PATH = PROJECT_ROOT / "data" / "raw" / "cases.jsonl"
HEXAGRAM_MASTER_PATH = PROJECT_ROOT / "data" / "hexagrams" / "hexagram_master.json"

# hexagram_transformationsモジュールをインポート
from hexagram_transformations import (
    get_nuclear_hexagram,
    get_inverted_hexagram,
    get_complementary_hexagram,
    HEXAGRAM_BY_ID,
    TRIGRAM_LINES,
    hexagram_to_lines,
)

# 八卦の定義
TRIGRAMS = ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]

# 64卦テーブル (下卦, 上卦) -> (卦番号, 卦名)
HEXAGRAM_TABLE = {
    ("乾", "乾"): (1, "乾為天"),
    ("乾", "坤"): (12, "天地否"),
    ("乾", "震"): (25, "天雷无妄"),
    ("乾", "巽"): (44, "天風姤"),
    ("乾", "坎"): (6, "天水訟"),
    ("乾", "離"): (13, "天火同人"),
    ("乾", "艮"): (33, "天山遯"),
    ("乾", "兌"): (10, "天沢履"),

    ("坤", "乾"): (11, "地天泰"),
    ("坤", "坤"): (2, "坤為地"),
    ("坤", "震"): (24, "地雷復"),
    ("坤", "巽"): (46, "地風升"),
    ("坤", "坎"): (7, "地水師"),
    ("坤", "離"): (36, "地火明夷"),
    ("坤", "艮"): (15, "地山謙"),
    ("坤", "兌"): (19, "地沢臨"),

    ("震", "乾"): (34, "雷天大壮"),
    ("震", "坤"): (16, "雷地豫"),
    ("震", "震"): (51, "震為雷"),
    ("震", "巽"): (32, "雷風恒"),
    ("震", "坎"): (40, "雷水解"),
    ("震", "離"): (55, "雷火豊"),
    ("震", "艮"): (62, "雷山小過"),
    ("震", "兌"): (54, "雷沢帰妹"),

    ("巽", "乾"): (9, "風天小畜"),
    ("巽", "坤"): (20, "風地観"),
    ("巽", "震"): (42, "風雷益"),
    ("巽", "巽"): (57, "巽為風"),
    ("巽", "坎"): (59, "風水渙"),
    ("巽", "離"): (37, "風火家人"),
    ("巽", "艮"): (53, "風山漸"),
    ("巽", "兌"): (61, "風沢中孚"),

    ("坎", "乾"): (5, "水天需"),
    ("坎", "坤"): (8, "水地比"),
    ("坎", "震"): (3, "水雷屯"),
    ("坎", "巽"): (48, "水風井"),
    ("坎", "坎"): (29, "坎為水"),
    ("坎", "離"): (63, "水火既済"),
    ("坎", "艮"): (39, "水山蹇"),
    ("坎", "兌"): (60, "水沢節"),

    ("離", "乾"): (14, "火天大有"),
    ("離", "坤"): (35, "火地晋"),
    ("離", "震"): (21, "火雷噬嗑"),
    ("離", "巽"): (50, "火風鼎"),
    ("離", "坎"): (64, "火水未済"),
    ("離", "離"): (30, "離為火"),
    ("離", "艮"): (56, "火山旅"),
    ("離", "兌"): (38, "火沢睽"),

    ("艮", "乾"): (26, "山天大畜"),
    ("艮", "坤"): (23, "山地剥"),
    ("艮", "震"): (27, "山雷頤"),
    ("艮", "巽"): (18, "山風蠱"),
    ("艮", "坎"): (4, "山水蒙"),
    ("艮", "離"): (22, "山火賁"),
    ("艮", "艮"): (52, "艮為山"),
    ("艮", "兌"): (41, "山沢損"),

    ("兌", "乾"): (43, "沢天夬"),
    ("兌", "坤"): (45, "沢地萃"),
    ("兌", "震"): (17, "沢雷随"),
    ("兌", "巽"): (28, "沢風大過"),
    ("兌", "坎"): (47, "沢水困"),
    ("兌", "離"): (49, "沢火革"),
    ("兌", "艮"): (31, "沢山咸"),
    ("兌", "兌"): (58, "兌為沢"),
}

# 逆引きテーブル (爻から八卦名)
LINES_TO_TRIGRAM = {
    tuple(v): k for k, v in TRIGRAM_LINES.items()
}

# 視点タイプの定義
@dataclass
class PerspectiveType:
    name: str
    name_ja: str
    primary_trigrams: List[str]
    description: str

PERSPECTIVES = {
    "shock_response": PerspectiveType(
        name="shock_response",
        name_ja="衝撃対応",
        primary_trigrams=["震", "坎"],
        description="突発的変化・危機への対応を重視"
    ),
    "endurance": PerspectiveType(
        name="endurance",
        name_ja="忍耐持久",
        primary_trigrams=["艮", "坤"],
        description="忍耐・安定・持久戦を重視"
    ),
    "strategic_action": PerspectiveType(
        name="strategic_action",
        name_ja="戦略行動",
        primary_trigrams=["乾", "離"],
        description="リーダーシップ・明晰な判断を重視"
    ),
    "adaptation": PerspectiveType(
        name="adaptation",
        name_ja="適応融和",
        primary_trigrams=["巽", "兌"],
        description="柔軟性・調和・融合を重視"
    ),
    "transformation": PerspectiveType(
        name="transformation",
        name_ja="変革革新",
        primary_trigrams=["離", "震"],
        description="変革・刷新・イノベーションを重視"
    ),
    "crisis_survival": PerspectiveType(
        name="crisis_survival",
        name_ja="危機生存",
        primary_trigrams=["坎", "艮"],
        description="危機回避・生存戦略を重視"
    ),
}

# 解釈タイプごとの視点マッピング（多様性を確保するため、typeごとに異なる視点を推奨）
TYPE_PERSPECTIVE_PRIORITY = {
    "primary": ["shock_response", "strategic_action", "transformation"],
    "nuclear": ["endurance", "crisis_survival", "adaptation"],  # 内なる構造→忍耐・危機生存
    "inverted": ["adaptation", "transformation", "endurance"],  # 上下反転→視点の転換
    "complementary": ["crisis_survival", "endurance", "shock_response"],  # 対極→危機・忍耐
}

# patternと八卦の相性
PATTERN_TRIGRAM_AFFINITY = {
    "Shock_Recovery": {"震": 0.9, "坎": 0.8, "離": 0.7, "乾": 0.6},
    "Hubris_Collapse": {"乾": 0.9, "離": 0.8, "坎": 0.5, "艮": 0.4},
    "Pivot_Success": {"離": 0.9, "震": 0.8, "巽": 0.7, "乾": 0.6},
    "Endurance": {"艮": 0.9, "坤": 0.8, "坎": 0.6, "巽": 0.5},
    "Slow_Decline": {"坤": 0.8, "艮": 0.7, "坎": 0.6, "兌": 0.4},
    "Steady_Growth": {"坤": 0.9, "艮": 0.8, "乾": 0.7, "離": 0.6},
    "Crisis_Pivot": {"坎": 0.9, "震": 0.8, "離": 0.7, "巽": 0.6},
    "Breakthrough": {"乾": 0.9, "震": 0.8, "離": 0.7, "兌": 0.5},
    "Exploration": {"巽": 0.9, "兌": 0.8, "坎": 0.6, "震": 0.5},
    "Managed_Decline": {"艮": 0.9, "坤": 0.8, "巽": 0.6, "坎": 0.5},
    "Decline": {"坤": 0.8, "坎": 0.7, "艮": 0.6, "震": 0.3},
    "Quiet_Fade": {"坤": 0.9, "巽": 0.7, "艮": 0.6, "坎": 0.5},
    "Stagnation": {"艮": 0.9, "坤": 0.8, "坎": 0.5, "兌": 0.3},
    "Failed_Attempt": {"震": 0.7, "坎": 0.7, "離": 0.5, "乾": 0.4},
}

# outcomeと八卦の相性
OUTCOME_TRIGRAM_AFFINITY = {
    "Success": {"乾": 0.9, "離": 0.8, "兌": 0.7, "震": 0.6},
    "PartialSuccess": {"巽": 0.8, "艮": 0.7, "離": 0.6, "坤": 0.5},
    "Failure": {"坎": 0.8, "坤": 0.6, "艮": 0.5, "震": 0.4},
    "Mixed": {"巽": 0.8, "兌": 0.7, "坎": 0.5, "離": 0.5},
}

# 爻位置の段階
YAO_STAGES = {
    1: "発芽期・始動期",
    2: "成長期・発展期",
    3: "転換期・岐路",
    4: "進展期・前進期",
    5: "全盛期・成熟期",
    6: "終末期・過熱期",
}

# 解釈タイプの説明
INTERPRETATION_TYPE_DESC = {
    "primary": "本卦（主要な象意）",
    "nuclear": "互卦（内なる構造・潜在的傾向）",
    "inverted": "覆卦（視点を反転した解釈）",
    "complementary": "錯卦（対極的・補完的視点）",
}


@dataclass
class Interpretation:
    """解釈データクラス（拡張版）"""
    rank: int
    type: str  # "primary", "nuclear", "inverted", "complementary"
    confidence: float
    perspective: str
    hexagram_id: int
    hexagram_name: str
    yao_position: int
    lower_trigram: str
    upper_trigram: str
    rationale: str


def load_hexagram_master() -> Dict:
    """hexagram_master.jsonを読み込む"""
    if HEXAGRAM_MASTER_PATH.exists():
        with open(HEXAGRAM_MASTER_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def get_hexagram_info(hexagram_id: int) -> Optional[Tuple[str, str, str]]:
    """卦IDから(下卦, 上卦, 卦名)を取得"""
    if hexagram_id not in HEXAGRAM_BY_ID:
        return None
    lower, upper, name = HEXAGRAM_BY_ID[hexagram_id]
    return (lower, upper, name)


def get_hexagram_by_trigrams(lower: str, upper: str) -> Optional[Tuple[int, str]]:
    """八卦の組み合わせから卦を取得"""
    return HEXAGRAM_TABLE.get((lower, upper))


def calculate_semantic_match(case: Dict, perspective: PerspectiveType,
                             lower: str, upper: str) -> float:
    """
    セマンティックマッチスコアを計算 (max 0.30)
    事例の内容が視点の八卦とどれだけ合致するか
    """
    score = 0.0
    before_hex = case.get("before_hex", "")
    trigger_hex = case.get("trigger_hex", "")
    action_hex = case.get("action_hex", "")
    after_hex = case.get("after_hex", "")

    # 視点の主要八卦との一致を確認
    for trigram in perspective.primary_trigrams:
        if trigram == before_hex:
            score += 0.05
        if trigram == trigger_hex:
            score += 0.05
        if trigram == action_hex:
            score += 0.05
        if trigram == after_hex:
            score += 0.05
        # 解釈の卦との一致
        if trigram == lower:
            score += 0.05
        if trigram == upper:
            score += 0.05

    return min(score, 0.30)


def calculate_pattern_fit(case: Dict, lower: str, upper: str) -> float:
    """
    パターン適合スコアを計算 (max 0.25)
    pattern_typeと八卦の相性
    """
    pattern = case.get("pattern_type", "")
    if pattern not in PATTERN_TRIGRAM_AFFINITY:
        return 0.10  # デフォルトスコア

    affinity = PATTERN_TRIGRAM_AFFINITY[pattern]
    lower_score = affinity.get(lower, 0.3) * 0.125
    upper_score = affinity.get(upper, 0.3) * 0.125

    return min(lower_score + upper_score, 0.25)


def calculate_yao_fit(case: Dict, yao_position: int) -> float:
    """
    爻位置適合スコアを計算 (max 0.20)
    事例の段階と爻位置の一致度
    """
    before_state = case.get("before_state", "")

    # 状態から推測される爻位置との比較
    state_yao_map = {
        "絶頂・慢心": 6,
        "成長痛": 3,
        "どん底・危機": 1,
        "安定・平和": 4,
        "停滞・閉塞": 2,
        "混乱・カオス": 3,
        "V字回復・大成功": 5,
        "縮小安定・生存": 2,
    }

    expected_yao = state_yao_map.get(before_state, 3)
    diff = abs(yao_position - expected_yao)

    if diff == 0:
        return 0.20
    elif diff == 1:
        return 0.15
    elif diff == 2:
        return 0.08
    else:
        return 0.03


def calculate_outcome_fit(case: Dict, lower: str, upper: str) -> float:
    """
    結果適合スコアを計算 (max 0.15)
    outcomeと八卦の相性
    """
    outcome = case.get("outcome", "")
    if outcome not in OUTCOME_TRIGRAM_AFFINITY:
        return 0.05  # デフォルトスコア

    affinity = OUTCOME_TRIGRAM_AFFINITY[outcome]
    lower_score = affinity.get(lower, 0.3) * 0.075
    upper_score = affinity.get(upper, 0.3) * 0.075

    return min(lower_score + upper_score, 0.15)


def calculate_type_bonus(interp_type: str) -> float:
    """
    解釈タイプに応じたボーナス/ペナルティ (max 0.10)
    - primary: 最も信頼度が高い
    - nuclear: やや低い（内なる傾向なので）
    - inverted: 中程度
    - complementary: やや低い（対極的なので）
    """
    bonuses = {
        "primary": 0.10,
        "nuclear": 0.04,
        "inverted": 0.06,
        "complementary": 0.02,
    }
    return bonuses.get(interp_type, 0.0)


def calculate_confidence(case: Dict, perspective: PerspectiveType,
                         lower: str, upper: str, yao_position: int,
                         interp_type: str) -> float:
    """
    総合信頼度を計算 (max 1.0)
    """
    semantic = calculate_semantic_match(case, perspective, lower, upper)
    pattern = calculate_pattern_fit(case, lower, upper)
    yao = calculate_yao_fit(case, yao_position)
    outcome = calculate_outcome_fit(case, lower, upper)
    type_bonus = calculate_type_bonus(interp_type)

    return round(semantic + pattern + yao + outcome + type_bonus, 3)


def select_best_perspective(case: Dict, lower: str, upper: str,
                           yao_position: int, interp_type: str,
                           used_perspectives: set) -> Tuple[PerspectiveType, float]:
    """
    最適な視点を選択し、信頼度を返す
    """
    priority_perspectives = TYPE_PERSPECTIVE_PRIORITY.get(interp_type, list(PERSPECTIVES.keys()))

    best_perspective = None
    best_confidence = 0.0

    # 優先視点から試す
    for pname in priority_perspectives:
        if pname in used_perspectives:
            continue
        perspective = PERSPECTIVES[pname]
        conf = calculate_confidence(case, perspective, lower, upper, yao_position, interp_type)
        if conf > best_confidence:
            best_confidence = conf
            best_perspective = perspective

    # 優先視点で見つからない場合、全視点から探す
    if best_perspective is None:
        for pname, perspective in PERSPECTIVES.items():
            if pname in used_perspectives:
                continue
            conf = calculate_confidence(case, perspective, lower, upper, yao_position, interp_type)
            if conf > best_confidence:
                best_confidence = conf
                best_perspective = perspective

    # 使用済み視点しかない場合、最高スコアの視点を再利用
    if best_perspective is None:
        for pname, perspective in PERSPECTIVES.items():
            conf = calculate_confidence(case, perspective, lower, upper, yao_position, interp_type)
            if conf > best_confidence:
                best_confidence = conf
                best_perspective = perspective

    return best_perspective, best_confidence


def generate_rationale(case: Dict, perspective: PerspectiveType,
                       hexagram_name: str, yao_position: int,
                       interp_type: str) -> str:
    """解釈の理由を生成"""
    perspective_desc = {
        "shock_response": "危機対応時の衝撃と反射的行動を重視",
        "endurance": "忍耐と持久力による安定維持を重視",
        "strategic_action": "リーダーシップと明確なビジョンによる前進を重視",
        "adaptation": "柔軟な適応と調和的融合を重視",
        "transformation": "変革と革新による新生を重視",
        "crisis_survival": "危機回避と生存戦略を重視",
    }

    type_desc = INTERPRETATION_TYPE_DESC.get(interp_type, "")
    base_rationale = perspective_desc.get(perspective.name, "多角的視点からの解釈")

    action = case.get("action_type", "")
    pattern = case.get("pattern_type", "")

    rationale = f"【{type_desc}】{base_rationale}。"
    rationale += f"{hexagram_name}の象意が{action}の行動と呼応し、"
    rationale += f"{pattern}パターンを示唆する。"
    rationale += f"第{yao_position}爻（{YAO_STAGES.get(yao_position, '不明')}）の段階。"

    return rationale


def determine_yao_position(case: Dict) -> int:
    """事例から爻位置を決定"""
    # 既存のyao_analysisから取得
    yao_analysis = case.get("yao_analysis", {})
    if yao_analysis and "before_yao_position" in yao_analysis:
        return yao_analysis["before_yao_position"]

    # changing_linesから推測
    changing_lines = case.get("changing_lines_1", [])
    if changing_lines:
        return changing_lines[0]

    # 状態から推測
    before_state = case.get("before_state", "")
    state_yao_map = {
        "絶頂・慢心": 6,
        "成長痛": 3,
        "どん底・危機": 1,
        "安定・平和": 4,
        "停滞・閉塞": 2,
        "混乱・カオス": 3,
        "V字回復・大成功": 5,
        "縮小安定・生存": 2,
    }
    return state_yao_map.get(before_state, 3)


def generate_interpretations_for_case(case: Dict) -> List[Dict]:
    """
    事例に対して複数の解釈を生成

    新ロジック:
    1. Primary: yao_analysis.before_hexagram_id をそのまま使用
    2. Nuclear（互卦）: get_nuclear_hexagram() で計算
    3. Inverted（覆卦）: get_inverted_hexagram() で計算
    4. Complementary（錯卦）: 信頼度0.50以上なら追加
    """
    interpretations = []
    used_perspectives = set()
    seen_hexagrams = set()

    yao_position = determine_yao_position(case)

    # 1. Primary解釈: yao_analysis.before_hexagram_id を使用
    yao_analysis = case.get("yao_analysis") or {}
    primary_hex_id = yao_analysis.get("before_hexagram_id") if yao_analysis else None

    if primary_hex_id and primary_hex_id in HEXAGRAM_BY_ID:
        hex_info = get_hexagram_info(primary_hex_id)
        if hex_info:
            lower, upper, hex_name = hex_info
            seen_hexagrams.add(primary_hex_id)

            perspective, confidence = select_best_perspective(
                case, lower, upper, yao_position, "primary", used_perspectives
            )

            # Primary解釈は本体データの卦なので、閾値に関係なく常に生成
            # (信頼度が低くても本卦として採用する)
            if perspective:
                # 信頼度が低い場合でも最低0.50を保証
                confidence = max(confidence, 0.50)
                used_perspectives.add(perspective.name)
                interp = Interpretation(
                    rank=1,
                    type="primary",
                    confidence=confidence,
                    perspective=perspective.name,
                    hexagram_id=primary_hex_id,
                    hexagram_name=hex_name,
                    yao_position=yao_position,
                    lower_trigram=lower,
                    upper_trigram=upper,
                    rationale=generate_rationale(case, perspective, hex_name, yao_position, "primary")
                )
                interpretations.append(asdict(interp))

    # Primary解釈がない場合、before_hex + trigger_hex にフォールバック
    if not interpretations:
        before_hex = case.get("before_hex", "")
        trigger_hex = case.get("trigger_hex", "")
        fallback_hex = get_hexagram_by_trigrams(before_hex, trigger_hex)

        if fallback_hex:
            hex_id, hex_name = fallback_hex
            seen_hexagrams.add(hex_id)

            perspective, confidence = select_best_perspective(
                case, before_hex, trigger_hex, yao_position, "primary", used_perspectives
            )

            if perspective and confidence >= 0.50:
                used_perspectives.add(perspective.name)
                interp = Interpretation(
                    rank=1,
                    type="primary",
                    confidence=confidence,
                    perspective=perspective.name,
                    hexagram_id=hex_id,
                    hexagram_name=hex_name,
                    yao_position=yao_position,
                    lower_trigram=before_hex,
                    upper_trigram=trigger_hex,
                    rationale=generate_rationale(case, perspective, hex_name, yao_position, "primary")
                )
                interpretations.append(asdict(interp))
                primary_hex_id = hex_id  # 互卦等の計算用

    # Primary卦がない場合は終了
    if not primary_hex_id:
        return interpretations

    # 2. Nuclear（互卦）解釈
    try:
        nuclear_id, nuclear_name = get_nuclear_hexagram(primary_hex_id)
        if nuclear_id not in seen_hexagrams:
            hex_info = get_hexagram_info(nuclear_id)
            if hex_info:
                lower, upper, _ = hex_info

                perspective, confidence = select_best_perspective(
                    case, lower, upper, yao_position, "nuclear", used_perspectives
                )

                if perspective and confidence >= 0.50:
                    seen_hexagrams.add(nuclear_id)
                    used_perspectives.add(perspective.name)
                    rank = len(interpretations) + 1

                    interp = Interpretation(
                        rank=rank,
                        type="nuclear",
                        confidence=confidence,
                        perspective=perspective.name,
                        hexagram_id=nuclear_id,
                        hexagram_name=nuclear_name,
                        yao_position=yao_position,
                        lower_trigram=lower,
                        upper_trigram=upper,
                        rationale=generate_rationale(case, perspective, nuclear_name, yao_position, "nuclear")
                    )
                    interpretations.append(asdict(interp))
    except (ValueError, KeyError):
        pass  # 互卦計算エラーは無視

    # 3. Inverted（覆卦）解釈
    try:
        inverted_id, inverted_name = get_inverted_hexagram(primary_hex_id)
        if inverted_id not in seen_hexagrams:
            hex_info = get_hexagram_info(inverted_id)
            if hex_info:
                lower, upper, _ = hex_info

                perspective, confidence = select_best_perspective(
                    case, lower, upper, yao_position, "inverted", used_perspectives
                )

                if perspective and confidence >= 0.50:
                    seen_hexagrams.add(inverted_id)
                    used_perspectives.add(perspective.name)
                    rank = len(interpretations) + 1

                    interp = Interpretation(
                        rank=rank,
                        type="inverted",
                        confidence=confidence,
                        perspective=perspective.name,
                        hexagram_id=inverted_id,
                        hexagram_name=inverted_name,
                        yao_position=yao_position,
                        lower_trigram=lower,
                        upper_trigram=upper,
                        rationale=generate_rationale(case, perspective, inverted_name, yao_position, "inverted")
                    )
                    interpretations.append(asdict(interp))
    except (ValueError, KeyError):
        pass  # 覆卦計算エラーは無視

    # 4. Complementary（錯卦）解釈
    try:
        comp_id, comp_name = get_complementary_hexagram(primary_hex_id)
        if comp_id not in seen_hexagrams:
            hex_info = get_hexagram_info(comp_id)
            if hex_info:
                lower, upper, _ = hex_info

                perspective, confidence = select_best_perspective(
                    case, lower, upper, yao_position, "complementary", used_perspectives
                )

                if perspective and confidence >= 0.50:
                    seen_hexagrams.add(comp_id)
                    used_perspectives.add(perspective.name)
                    rank = len(interpretations) + 1

                    interp = Interpretation(
                        rank=rank,
                        type="complementary",
                        confidence=confidence,
                        perspective=perspective.name,
                        hexagram_id=comp_id,
                        hexagram_name=comp_name,
                        yao_position=yao_position,
                        lower_trigram=lower,
                        upper_trigram=upper,
                        rationale=generate_rationale(case, perspective, comp_name, yao_position, "complementary")
                    )
                    interpretations.append(asdict(interp))
    except (ValueError, KeyError):
        pass  # 錯卦計算エラーは無視

    # ソート: primaryを最優先、その後は信頼度順
    # type優先度: primary > nuclear > inverted > complementary
    type_priority = {"primary": 0, "nuclear": 1, "inverted": 2, "complementary": 3}
    interpretations.sort(key=lambda x: (type_priority.get(x["type"], 9), -x["confidence"]))

    for i, interp in enumerate(interpretations):
        interp["rank"] = i + 1

    return interpretations[:4]  # 最大4解釈


def process_cases(dry_run: bool = False, force: bool = False) -> None:
    """全事例を処理"""
    print(f"事例ファイル: {CASES_PATH}")
    print(f"卦マスターファイル: {HEXAGRAM_MASTER_PATH}")
    print(f"ドライラン: {'はい' if dry_run else 'いいえ'}")
    print(f"強制上書き: {'はい' if force else 'いいえ'}")
    print()

    # hexagram_master読み込み
    hexagram_master = load_hexagram_master()
    print(f"卦マスター: {len(hexagram_master)}件ロード済み")

    # 事例読み込み
    with open(CASES_PATH, 'r', encoding='utf-8') as f:
        cases = [json.loads(line) for line in f if line.strip()]

    print(f"事例数: {len(cases)}件")
    print()
    print("解釈生成を開始...")

    processed = 0
    skipped = 0
    generated = 0
    total_interpretations = 0

    # 統計用
    type_counts = {"primary": 0, "nuclear": 0, "inverted": 0, "complementary": 0}
    perspective_counts = {p: 0 for p in PERSPECTIVES}

    for i, case in enumerate(cases):
        # 進捗表示
        if (i + 1) % 500 == 0:
            print(f"  進捗: {i + 1}/{len(cases)} 件処理完了")

        # 既存の解釈をチェック
        if "interpretations" in case and case["interpretations"] and not force:
            skipped += 1
            continue

        # 解釈を生成
        interpretations = generate_interpretations_for_case(case)

        if interpretations:
            case["interpretations"] = interpretations
            generated += 1
            total_interpretations += len(interpretations)

            # 統計更新
            for interp in interpretations:
                type_counts[interp["type"]] = type_counts.get(interp["type"], 0) + 1
                perspective_counts[interp["perspective"]] = perspective_counts.get(interp["perspective"], 0) + 1

        processed += 1

    print()
    print("=== 処理結果 ===")
    print(f"処理件数: {processed}")
    print(f"スキップ: {skipped}")
    print(f"生成件数: {generated}")
    print(f"総解釈数: {total_interpretations}")
    if generated > 0:
        print(f"平均解釈数: {total_interpretations / generated:.2f}")

    print()
    print("=== 解釈タイプ別 ===")
    for t, count in type_counts.items():
        print(f"  {t}: {count}")

    print()
    print("=== 視点別 ===")
    for p, count in sorted(perspective_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {p}: {count}")

    # 保存
    if not dry_run:
        with open(CASES_PATH, 'w', encoding='utf-8') as f:
            for case in cases:
                f.write(json.dumps(case, ensure_ascii=False) + '\n')
        print()
        print(f"保存完了: {CASES_PATH}")
    else:
        print()
        print("ドライランのため保存をスキップしました。")

        # サンプル出力
        print()
        print("=== サンプル出力（最初の3件） ===")
        sample_count = 0
        for case in cases:
            if "interpretations" in case and case["interpretations"]:
                print(f"\n--- {case.get('target_name', 'Unknown')} ---")
                print(json.dumps(case["interpretations"], ensure_ascii=False, indent=2))
                sample_count += 1
                if sample_count >= 3:
                    break


def main():
    parser = argparse.ArgumentParser(
        description="多視点卦解釈生成スクリプト v2 (Primary + Nuclear/Inverted/Complementary)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ドライラン（保存しない）"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存の解釈を上書き"
    )

    args = parser.parse_args()
    process_cases(dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
