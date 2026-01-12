#!/usr/bin/env python3
"""
多視点卦解釈生成スクリプト

各事例に対して、異なる視点から複数の卦解釈を生成し、
信頼度0.50以上のものをinterpretations配列に追加する。

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

# 八卦の定義
TRIGRAMS = ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]

# 64卦テーブル (下卦, 上卦) -> (卦番号, 卦名)
# 行: 下卦 (乾坤震巽坎離艮兌), 列: 上卦 (乾坤震巽坎離艮兌)
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

# before_stateから推奨八卦へのマッピング
BEFORE_STATE_TRIGRAM_MAP = {
    "絶頂・慢心": {"primary": "乾", "alt": "離"},
    "停滞・閉塞": {"primary": "艮", "alt": "坤"},
    "混乱・カオス": {"primary": "坎", "alt": "震"},
    "成長痛": {"primary": "震", "alt": "巽"},
    "どん底・危機": {"primary": "坎", "alt": "艮"},
    "安定・平和": {"primary": "坤", "alt": "兌"},
    "V字回復・大成功": {"primary": "乾", "alt": "離"},
    "縮小安定・生存": {"primary": "艮", "alt": "坤"},
}

# trigger_typeから推奨八卦へのマッピング
TRIGGER_TYPE_TRIGRAM_MAP = {
    "外部ショック": {"primary": "震", "alt": "坎"},
    "内部崩壊": {"primary": "坎", "alt": "離"},
    "意図的決断": {"primary": "乾", "alt": "離"},
    "偶発・出会い": {"primary": "兌", "alt": "巽"},
}

# action_typeから推奨八卦へのマッピング
ACTION_TYPE_TRIGRAM_MAP = {
    "攻める・挑戦": "乾",
    "守る・維持": "艮",
    "捨てる・撤退": "巽",
    "耐える・潜伏": "坤",
    "対話・融合": "兌",
    "刷新・破壊": "離",
    "逃げる・放置": "坎",
    "分散・スピンオフ": "巽",
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

@dataclass
class Interpretation:
    """解釈データクラス"""
    rank: int
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


def get_hexagram_by_trigrams(lower: str, upper: str) -> Optional[Tuple[int, str]]:
    """八卦の組み合わせから卦を取得"""
    return HEXAGRAM_TABLE.get((lower, upper))


def calculate_semantic_match(case: Dict, perspective: PerspectiveType) -> float:
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
            score += 0.075
        if trigram == trigger_hex:
            score += 0.075
        if trigram == action_hex:
            score += 0.075
        if trigram == after_hex:
            score += 0.075

    return min(score, 0.30)


def calculate_trigram_flow(case: Dict, lower: str, upper: str) -> float:
    """
    八卦フロー整合性スコアを計算 (max 0.25)
    before→trigger→action→afterの流れと卦の一貫性
    """
    score = 0.0
    before_hex = case.get("before_hex", "")
    trigger_hex = case.get("trigger_hex", "")
    action_hex = case.get("action_hex", "")
    after_hex = case.get("after_hex", "")

    # 下卦が内面（before/action）と一致
    if lower == before_hex:
        score += 0.08
    if lower == action_hex:
        score += 0.05

    # 上卦が外面（trigger/after）と一致
    if upper == trigger_hex:
        score += 0.08
    if upper == after_hex:
        score += 0.04

    return min(score, 0.25)


def calculate_pattern_fit(case: Dict, lower: str, upper: str) -> float:
    """
    パターン適合スコアを計算 (max 0.20)
    pattern_typeと八卦の相性
    """
    pattern = case.get("pattern_type", "")
    if pattern not in PATTERN_TRIGRAM_AFFINITY:
        return 0.10  # デフォルトスコア

    affinity = PATTERN_TRIGRAM_AFFINITY[pattern]
    lower_score = affinity.get(lower, 0.3) * 0.10
    upper_score = affinity.get(upper, 0.3) * 0.10

    return min(lower_score + upper_score, 0.20)


def calculate_yao_fit(case: Dict, yao_position: int) -> float:
    """
    爻位置適合スコアを計算 (max 0.15)
    事例の段階と爻位置の一致度
    """
    before_state = case.get("before_state", "")
    after_state = case.get("after_state", "")

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
        return 0.15
    elif diff == 1:
        return 0.10
    elif diff == 2:
        return 0.05
    else:
        return 0.02


def calculate_outcome_fit(case: Dict, lower: str, upper: str) -> float:
    """
    結果適合スコアを計算 (max 0.10)
    outcomeと八卦の相性
    """
    outcome = case.get("outcome", "")
    if outcome not in OUTCOME_TRIGRAM_AFFINITY:
        return 0.05  # デフォルトスコア

    affinity = OUTCOME_TRIGRAM_AFFINITY[outcome]
    lower_score = affinity.get(lower, 0.3) * 0.05
    upper_score = affinity.get(upper, 0.3) * 0.05

    return min(lower_score + upper_score, 0.10)


def calculate_confidence(case: Dict, perspective: PerspectiveType,
                         lower: str, upper: str, yao_position: int) -> float:
    """
    総合信頼度を計算 (max 1.0)
    """
    semantic = calculate_semantic_match(case, perspective)
    flow = calculate_trigram_flow(case, lower, upper)
    pattern = calculate_pattern_fit(case, lower, upper)
    yao = calculate_yao_fit(case, yao_position)
    outcome = calculate_outcome_fit(case, lower, upper)

    return round(semantic + flow + pattern + yao + outcome, 3)


def generate_rationale(case: Dict, perspective: PerspectiveType,
                       hexagram_name: str, yao_position: int) -> str:
    """解釈の理由を生成"""
    perspective_desc = {
        "shock_response": "危機対応時の衝撃と反射的行動を重視",
        "endurance": "忍耐と持久力による安定維持を重視",
        "strategic_action": "リーダーシップと明確なビジョンによる前進を重視",
        "adaptation": "柔軟な適応と調和的融合を重視",
        "transformation": "変革と革新による新生を重視",
        "crisis_survival": "危機回避と生存戦略を重視",
    }

    base_rationale = perspective_desc.get(perspective.name, "多角的視点からの解釈")

    action = case.get("action_type", "")
    pattern = case.get("pattern_type", "")

    rationale = f"{base_rationale}。"
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


def get_alternative_trigrams(case: Dict) -> List[Tuple[str, str]]:
    """事例から代替の八卦ペアを生成"""
    alternatives = []

    before_state = case.get("before_state", "")
    trigger_type = case.get("trigger_type", "")
    action_type = case.get("action_type", "")

    # before_stateから下卦候補を取得
    before_map = BEFORE_STATE_TRIGRAM_MAP.get(before_state, {})
    lower_candidates = [before_map.get("primary"), before_map.get("alt")]
    lower_candidates = [t for t in lower_candidates if t]

    # trigger_typeから上卦候補を取得
    trigger_map = TRIGGER_TYPE_TRIGRAM_MAP.get(trigger_type, {})
    upper_candidates = [trigger_map.get("primary"), trigger_map.get("alt")]
    upper_candidates = [t for t in upper_candidates if t]

    # action_typeからも候補を追加
    action_trigram = ACTION_TYPE_TRIGRAM_MAP.get(action_type)
    if action_trigram:
        lower_candidates.append(action_trigram)

    # 組み合わせを生成
    for lower in lower_candidates[:2]:
        for upper in upper_candidates[:2]:
            if lower and upper:
                alternatives.append((lower, upper))

    return list(set(alternatives))


def generate_interpretations_for_case(case: Dict) -> List[Dict]:
    """事例に対して複数の解釈を生成"""
    interpretations = []
    seen_hexagrams = set()

    before_hex = case.get("before_hex", "")
    trigger_hex = case.get("trigger_hex", "")
    yao_position = determine_yao_position(case)

    # 1. プライマリ解釈（既存の八卦を使用）
    primary_hex = get_hexagram_by_trigrams(before_hex, trigger_hex)
    if primary_hex:
        hex_id, hex_name = primary_hex
        seen_hexagrams.add(hex_id)

        # 最適な視点を選択
        best_perspective = None
        best_confidence = 0.0

        for perspective in PERSPECTIVES.values():
            conf = calculate_confidence(case, perspective, before_hex, trigger_hex, yao_position)
            if conf > best_confidence:
                best_confidence = conf
                best_perspective = perspective

        if best_perspective and best_confidence >= 0.50:
            interp = Interpretation(
                rank=1,
                confidence=best_confidence,
                perspective=best_perspective.name,
                hexagram_id=hex_id,
                hexagram_name=hex_name,
                yao_position=yao_position,
                lower_trigram=before_hex,
                upper_trigram=trigger_hex,
                rationale=generate_rationale(case, best_perspective, hex_name, yao_position)
            )
            interpretations.append(asdict(interp))

    # 2. 代替解釈（異なる視点から）
    alt_trigrams = get_alternative_trigrams(case)

    for lower, upper in alt_trigrams:
        hex_info = get_hexagram_by_trigrams(lower, upper)
        if not hex_info:
            continue

        hex_id, hex_name = hex_info
        if hex_id in seen_hexagrams:
            continue

        # 各視点で評価
        for perspective in PERSPECTIVES.values():
            # この視点の主要八卦と一致するか確認
            if lower not in perspective.primary_trigrams and upper not in perspective.primary_trigrams:
                continue

            conf = calculate_confidence(case, perspective, lower, upper, yao_position)
            if conf >= 0.50:
                seen_hexagrams.add(hex_id)
                rank = len(interpretations) + 1

                interp = Interpretation(
                    rank=rank,
                    confidence=conf,
                    perspective=perspective.name,
                    hexagram_id=hex_id,
                    hexagram_name=hex_name,
                    yao_position=yao_position,
                    lower_trigram=lower,
                    upper_trigram=upper,
                    rationale=generate_rationale(case, perspective, hex_name, yao_position)
                )
                interpretations.append(asdict(interp))

                if len(interpretations) >= 3:
                    break

        if len(interpretations) >= 3:
            break

    # 信頼度でソートし、ランクを更新
    interpretations.sort(key=lambda x: x["confidence"], reverse=True)
    for i, interp in enumerate(interpretations):
        interp["rank"] = i + 1

    return interpretations[:3]


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

        processed += 1

    print()
    print("=== 処理結果 ===")
    print(f"処理件数: {processed}")
    print(f"スキップ: {skipped}")
    print(f"生成件数: {generated}")
    print(f"総解釈数: {total_interpretations}")
    if generated > 0:
        print(f"平均解釈数: {total_interpretations / generated:.2f}")

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
        description="多視点卦解釈生成スクリプト"
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
