#!/usr/bin/env python3
"""
既存のケースデータに爻レベル情報を付与するスクリプト

以下の情報を追加:
- before_yao_position: 開始時の爻位（1-6）
- yao_action_compatibility: 行動と爻位の適合性スコア
- predicted_outcome: ロジックに基づく予測結果
- prediction_analysis: 予測と実際の比較分析
"""

import json
from pathlib import Path
from typing import Optional
import re

# 設定
BASE_DIR = Path(__file__).parent.parent
CASES_FILE = BASE_DIR / "data" / "raw" / "cases.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "raw" / "cases_enriched.jsonl"
MAPPINGS_DIR = BASE_DIR / "data" / "mappings"

# マッピングデータ読み込み
def load_mappings():
    with open(MAPPINGS_DIR / "yao_recommendations.json", "r", encoding="utf-8") as f:
        recommendations = json.load(f)
    with open(MAPPINGS_DIR / "action_yao_compatibility.json", "r", encoding="utf-8") as f:
        compatibility = json.load(f)
    with open(MAPPINGS_DIR / "yao_transitions.json", "r", encoding="utf-8") as f:
        transitions = json.load(f)
    return recommendations, compatibility, transitions


# 卦名から卦番号への変換
HEXAGRAM_NAME_TO_ID = {
    "乾為天": 1, "坤為地": 2, "水雷屯": 3, "山水蒙": 4, "水天需": 5,
    "天水訟": 6, "地水師": 7, "水地比": 8, "風天小畜": 9, "天沢履": 10,
    "地天泰": 11, "天地否": 12, "天火同人": 13, "火天大有": 14, "地山謙": 15,
    "雷地予": 16, "沢雷随": 17, "山風蠱": 18, "地沢臨": 19, "風地観": 20,
    "火雷噬嗑": 21, "山火賁": 22, "山地剥": 23, "地雷復": 24, "天雷无妄": 25,
    "山天大畜": 26, "山雷頤": 27, "沢風大過": 28, "坎為水": 29, "離為火": 30,
    "沢山咸": 31, "雷風恒": 32, "天山遯": 33, "雷天大壮": 34, "火地晋": 35,
    "地火明夷": 36, "風火家人": 37, "火沢睽": 38, "水山蹇": 39, "雷水解": 40,
    "山沢損": 41, "風雷益": 42, "沢天夬": 43, "天風姤": 44, "沢地萃": 45,
    "地風升": 46, "沢水困": 47, "水風井": 48, "沢火革": 49, "火風鼎": 50,
    "震為雷": 51, "艮為山": 52, "風山漸": 53, "雷沢帰妹": 54, "雷火豊": 55,
    "火山旅": 56, "巽為風": 57, "兌為沢": 58, "風水渙": 59, "水沢節": 60,
    "風沢中孚": 61, "雷山小過": 62, "水火既済": 63, "火水未済": 64,
}


def parse_hexagram_name(name_str: str) -> Optional[int]:
    """
    卦名文字列から卦番号を抽出
    例: "52_艮", "艮為山", "52_艮為山" → 52
    """
    if not name_str:
        return None

    # 数字プレフィックスがある場合
    match = re.match(r'^(\d+)', name_str)
    if match:
        return int(match.group(1))

    # 卦名での検索
    for name, hex_id in HEXAGRAM_NAME_TO_ID.items():
        if name in name_str:
            return hex_id

    return None


def diagnose_yao_position(case: dict) -> int:
    """
    ケースの状態から爻位を診断

    診断ロジック:
    - before_state と pattern_type を組み合わせて判定
    - 企業の場合、市場ポジションも考慮
    """
    before_state = case.get("before_state", "")
    pattern_type = case.get("pattern_type", "")
    outcome = case.get("outcome", "")
    scale = case.get("scale", "")

    # before_state に基づく基本判定
    state_to_yao = {
        "絶頂・慢心": 6,      # 上爻: 極みにいる、行き過ぎのリスク
        "安定・平和": 5,      # 五爻: リーダーポジション、安定
        "成長痛": 3,          # 三爻: 岐路、転換点
        "停滞・閉塞": 4,      # 四爻: 次のステップへの準備期
        "混乱・カオス": 3,    # 三爻: 分岐点、不安定
        "どん底・危機": 1,    # 初爻: 底からの出発
    }

    base_yao = state_to_yao.get(before_state, 3)  # デフォルトは三爻（岐路）

    # pattern_type による調整
    if pattern_type == "Hubris_Collapse":
        # 慢心からの崩壊は上爻が多い
        if before_state == "絶頂・慢心":
            base_yao = 6
    elif pattern_type == "Endurance":
        # 耐え忍びは初爻〜二爻が多い
        if before_state in ["どん底・危機", "混乱・カオス"]:
            base_yao = 1
    elif pattern_type == "Pivot_Success":
        # ピボット成功は三爻〜四爻が多い
        if before_state in ["停滞・閉塞", "どん底・危機"]:
            base_yao = 3
    elif pattern_type == "Shock_Recovery":
        # ショック回復は二爻〜三爻が多い
        base_yao = min(base_yao, 3)

    return base_yao


def calculate_compatibility(action_type: str, yao_position: int, compatibility_matrix: dict) -> dict:
    """
    行動タイプと爻位の適合性を計算
    """
    if action_type in compatibility_matrix:
        yao_data = compatibility_matrix[action_type].get(str(yao_position), {})
        return {
            "score": yao_data.get("score", 3),
            "reason": yao_data.get("reason", "")
        }
    return {"score": 3, "reason": "該当なし"}


def predict_outcome(yao_position: int, action_type: str, compatibility_matrix: dict) -> str:
    """
    爻位と行動から結果を予測

    score 1 = Success
    score 2 = PartialSuccess
    score 3 = Mixed
    score 4 = Failure
    """
    compat = calculate_compatibility(action_type, yao_position, compatibility_matrix)
    score = compat.get("score", 3)

    score_to_outcome = {
        1: "Success",
        2: "PartialSuccess",
        3: "Mixed",
        4: "Failure"
    }

    return score_to_outcome.get(score, "Mixed")


def analyze_prediction(predicted: str, actual: str) -> dict:
    """
    予測と実際の結果を比較分析
    """
    # 完全一致
    if predicted == actual:
        return {
            "match": True,
            "accuracy": "exact",
            "note": "予測と実際が一致"
        }

    # 近い結果
    close_pairs = [
        ("Success", "PartialSuccess"),
        ("PartialSuccess", "Mixed"),
        ("Mixed", "Failure"),
    ]
    for pair in close_pairs:
        if (predicted in pair and actual in pair):
            return {
                "match": False,
                "accuracy": "close",
                "note": f"予測({predicted})と実際({actual})は近い"
            }

    # 大きく外れ
    return {
        "match": False,
        "accuracy": "miss",
        "note": f"予測({predicted})と実際({actual})は乖離"
    }


def get_yao_phrase(hexagram_id: int, yao_position: int, yao_phrases: dict) -> dict:
    """
    卦番号と爻位から爻辞を取得
    """
    key = f"{hexagram_id}-{yao_position}"
    return yao_phrases.get(key, {"classic": "", "modern": ""})


def enrich_case(case: dict, recommendations: dict, compatibility: dict, transitions: dict, yao_phrases: dict) -> dict:
    """
    1件のケースに爻情報を付与
    """
    enriched = case.copy()

    # 1. 開始時の卦番号を取得
    before_hex_name = case.get("classical_before_hexagram", "")
    before_hex_id = parse_hexagram_name(before_hex_name)

    # 2. 爻位を診断
    yao_position = diagnose_yao_position(case)

    # 3. 行動との適合性を計算
    action_type = case.get("action_type", "")
    compat = calculate_compatibility(action_type, yao_position, compatibility)

    # 4. 結果を予測
    predicted = predict_outcome(yao_position, action_type, compatibility)
    actual = case.get("outcome", "Mixed")

    # 5. 予測分析
    analysis = analyze_prediction(predicted, actual)

    # 6. 爻辞を取得
    if before_hex_id:
        yao_phrase = get_yao_phrase(before_hex_id, yao_position, yao_phrases)
    else:
        yao_phrase = {"classic": "", "modern": ""}

    # 7. 推奨行動を取得
    yao_rec = recommendations.get(str(yao_position), {})

    # 新しいフィールドを追加
    enriched["yao_analysis"] = {
        "before_hexagram_id": before_hex_id,
        "before_yao_position": yao_position,
        "yao_phrase_classic": yao_phrase.get("classic", ""),
        "yao_phrase_modern": yao_phrase.get("modern", ""),
        "yao_stage": yao_rec.get("stage", ""),
        "yao_basic_stance": yao_rec.get("basic_stance", ""),
        "action_compatibility": {
            "score": compat.get("score"),
            "reason": compat.get("reason"),
            "is_optimal": compat.get("score") == 1
        },
        "predicted_outcome": predicted,
        "actual_outcome": actual,
        "prediction_analysis": analysis,
        "recommended_actions": yao_rec.get("recommended_actions", []),
        "avoid_actions": yao_rec.get("avoid_actions", []),
    }

    return enriched


def load_yao_phrases() -> dict:
    """
    384爻の爻辞データを読み込み
    """
    phrases_file = BASE_DIR / "data" / "yao_phrases_384.json"
    if phrases_file.exists():
        with open(phrases_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    print("=== ケースデータへの爻情報付与 ===\n")

    # マッピングデータ読み込み
    print("1. マッピングデータ読み込み...")
    recommendations, compatibility, transitions = load_mappings()
    yao_phrases = load_yao_phrases()
    print(f"   - 爻推奨: {len(recommendations)} 件")
    print(f"   - 適合性マトリクス: {len(compatibility)} × 6 = {len(compatibility) * 6} 件")
    print(f"   - 爻辞: {len(yao_phrases)} 件")

    # ケースデータ読み込み
    print("\n2. ケースデータ読み込み...")
    cases = []
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    print(f"   - {len(cases)} 件のケースを読み込み")

    # 各ケースに爻情報を付与
    print("\n3. 爻情報付与...")
    enriched_cases = []
    stats = {"exact": 0, "close": 0, "miss": 0}

    for case in cases:
        enriched = enrich_case(case, recommendations, compatibility, transitions, yao_phrases)
        enriched_cases.append(enriched)

        # 統計更新
        accuracy = enriched.get("yao_analysis", {}).get("prediction_analysis", {}).get("accuracy", "miss")
        stats[accuracy] = stats.get(accuracy, 0) + 1

    # 結果を保存
    print("\n4. 結果保存...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for case in enriched_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"   - {OUTPUT_FILE} に保存完了")

    # 統計出力
    total = len(cases)
    print("\n=== 予測精度統計 ===")
    print(f"完全一致: {stats['exact']} 件 ({stats['exact']/total*100:.1f}%)")
    print(f"近い結果: {stats['close']} 件 ({stats['close']/total*100:.1f}%)")
    print(f"乖離: {stats['miss']} 件 ({stats['miss']/total*100:.1f}%)")
    print(f"合計: {total} 件")

    # サンプル出力
    print("\n=== サンプル出力（最初の3件）===")
    for i, case in enumerate(enriched_cases[:3]):
        print(f"\n--- {i+1}. {case.get('target_name')} ---")
        yao = case.get("yao_analysis", {})
        print(f"爻位: {yao.get('before_yao_position')}爻 ({yao.get('yao_stage')})")
        print(f"爻辞: {yao.get('yao_phrase_classic')} - {yao.get('yao_phrase_modern')}")
        print(f"行動: {case.get('action_type')}")
        print(f"適合性: スコア{yao.get('action_compatibility', {}).get('score')} - {yao.get('action_compatibility', {}).get('reason')}")
        print(f"予測: {yao.get('predicted_outcome')} → 実際: {yao.get('actual_outcome')}")
        print(f"分析: {yao.get('prediction_analysis', {}).get('note')}")


if __name__ == "__main__":
    main()
