#!/usr/bin/env python3
"""
改善版：既存のケースデータに爻レベル情報を付与するスクリプト

v2 改善点:
- 実データに存在するすべてのbefore_state値に対応
- 実データに存在するすべてのaction_type値に対応
- pattern_typeとoutcomeを考慮した爻位診断
- 類似action_typeのマッピング追加
"""

import json
from pathlib import Path
from typing import Optional
import re

# 設定
BASE_DIR = Path(__file__).parent.parent
CASES_FILE = BASE_DIR / "data" / "raw" / "cases.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "raw" / "cases_enriched_v2.jsonl"
MAPPINGS_DIR = BASE_DIR / "data" / "mappings"

# マッピングデータ読み込み
def load_mappings():
    with open(MAPPINGS_DIR / "yao_recommendations.json", "r", encoding="utf-8") as f:
        recommendations = json.load(f)
    # v2の調整版マトリクスを使用
    compat_file = MAPPINGS_DIR / "action_yao_compatibility_v2.json"
    if not compat_file.exists():
        compat_file = MAPPINGS_DIR / "action_yao_compatibility.json"
    with open(compat_file, "r", encoding="utf-8") as f:
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


# 拡張されたbefore_state → 爻位マッピング
STATE_TO_YAO = {
    # 既存（定義済み）
    "絶頂・慢心": 6,      # 上爻: 極みにいる、行き過ぎのリスク
    "安定・平和": 5,      # 五爻: リーダーポジション、安定
    "成長痛": 3,          # 三爻: 岐路、転換点
    "停滞・閉塞": 4,      # 四爻: 次のステップへの準備期
    "混乱・カオス": 3,    # 三爻: 分岐点、不安定
    "どん底・危機": 1,    # 初爻: 底からの出発

    # 追加マッピング（実データに存在する値）
    "混乱・衰退": 3,      # 三爻: 混乱・カオスと同様
    "安定・停止": 4,      # 四爻: 停滞に近い
    "安定成長・成功": 5,  # 五爻: 安定・平和と同様
    "拡大・繁栄": 5,      # 五爻: 成長のピーク
    "成長・拡大": 2,      # 二爻: 成長期
    "縮小安定・生存": 6,  # 上爻: 縮小後の安定、終末期
    "調和・繁栄": 5,      # 五爻: 安定・平和と同様
    "V字回復・大成功": 5, # 五爻: 成功状態
    "急成長・拡大": 2,    # 二爻: 成長期
}


# action_typeの正規化マッピング（類似した行動を統合）
ACTION_NORMALIZE = {
    # 攻める・挑戦 系
    "攻める・挑戦": "攻める・挑戦",
    "拡大・攻め": "攻める・挑戦",
    "集中・拡大": "攻める・挑戦",
    "輝く・表現": "攻める・挑戦",  # 表に出る行動

    # 守る・維持 系
    "守る・維持": "守る・維持",
    "逃げる・守る": "守る・維持",

    # 捨てる・撤退 系
    "捨てる・撤退": "捨てる・撤退",
    "捨てる・転換": "捨てる・撤退",
    "撤退・収縮": "捨てる・撤退",
    "撤退・縮小": "捨てる・撤退",

    # 耐える・潜伏 系
    "耐える・潜伏": "耐える・潜伏",

    # 対話・融合 系
    "対話・融合": "対話・融合",
    "交流・発表": "対話・融合",  # 他者との交流

    # 刷新・破壊 系
    "刷新・破壊": "刷新・破壊",

    # 逃げる・放置 系
    "逃げる・放置": "逃げる・放置",
    "撤退・逃げる": "逃げる・放置",
    "逃げる・分散": "逃げる・放置",

    # 分散・スピンオフ 系
    "分散・スピンオフ": "分散・スピンオフ",
    "分散・探索": "分散・スピンオフ",
    "分散・多角化": "分散・スピンオフ",
    "分散・独立": "分散・スピンオフ",
    "分散する・独立する": "分散・スピンオフ",
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
    ケースの状態から爻位を診断（改善版）

    診断ロジック:
    - before_state をベースに
    - pattern_type で調整
    - outcome も参考にする
    """
    before_state = case.get("before_state", "")
    pattern_type = case.get("pattern_type", "")
    outcome = case.get("outcome", "")

    # before_state に基づく基本判定
    base_yao = STATE_TO_YAO.get(before_state, 3)  # デフォルトは三爻（岐路）

    # pattern_type による調整
    if pattern_type == "Hubris_Collapse":
        # 慢心からの崩壊は上爻が多い
        if before_state in ["絶頂・慢心", "拡大・繁栄", "安定成長・成功"]:
            base_yao = 6

    elif pattern_type == "Endurance":
        # 耐え忍びは初爻〜二爻が多い
        if before_state in ["どん底・危機", "混乱・カオス", "混乱・衰退"]:
            base_yao = 1
        elif before_state in ["停滞・閉塞", "安定・停止"]:
            base_yao = 2

    elif pattern_type == "Pivot_Success":
        # ピボット成功は三爻〜四爻が多い（転換点での判断）
        if before_state in ["停滞・閉塞", "どん底・危機", "混乱・カオス"]:
            base_yao = 3

    elif pattern_type == "Shock_Recovery":
        # ショック回復は様々だが、危機からの回復は初爻〜二爻
        if before_state in ["どん底・危機", "混乱・カオス", "混乱・衰退"]:
            base_yao = 1
        elif before_state in ["成長痛", "停滞・閉塞"]:
            base_yao = 3

    elif pattern_type == "Slow_Decline":
        # じわじわ衰退は四爻〜六爻
        if before_state in ["絶頂・慢心", "安定・平和"]:
            base_yao = 6
        elif before_state in ["停滞・閉塞", "安定・停止"]:
            base_yao = 4

    elif pattern_type == "Steady_Growth":
        # 安定成長は二爻〜三爻
        base_yao = 2

    elif pattern_type == "Breakthrough":
        # ブレークスルーは三爻（岐路での決断）
        base_yao = 3

    elif pattern_type == "Crisis_Pivot":
        # 危機でのピボット
        if before_state in ["どん底・危機", "混乱・カオス"]:
            base_yao = 1

    elif pattern_type == "Managed_Decline":
        # 管理された衰退は上爻
        base_yao = 6

    elif pattern_type == "Exploration":
        # 探索は二爻〜三爻
        base_yao = 2

    elif pattern_type == "Stagnation":
        # 停滞は四爻
        base_yao = 4

    elif pattern_type == "Failed_Attempt":
        # 失敗した試みは様々
        if before_state in ["成長痛", "停滞・閉塞"]:
            base_yao = 3

    elif pattern_type == "Quiet_Fade":
        # 静かな衰退は上爻
        base_yao = 6

    return base_yao


def normalize_action(action_type: str) -> str:
    """行動タイプを正規化"""
    return ACTION_NORMALIZE.get(action_type, action_type)


def calculate_compatibility(action_type: str, yao_position: int, compatibility_matrix: dict) -> dict:
    """
    行動タイプと爻位の適合性を計算（改善版）
    """
    # 行動タイプを正規化
    normalized_action = normalize_action(action_type)

    if normalized_action in compatibility_matrix:
        yao_data = compatibility_matrix[normalized_action].get(str(yao_position), {})
        return {
            "score": yao_data.get("score", 3),
            "reason": yao_data.get("reason", ""),
            "original_action": action_type,
            "normalized_action": normalized_action
        }

    # マッピングがない場合はデフォルト値
    return {
        "score": 3,
        "reason": "該当なし",
        "original_action": action_type,
        "normalized_action": normalized_action
    }


def predict_outcome(yao_position: int, action_type: str, compatibility_matrix: dict,
                    pattern_type: str = "", before_state: str = "") -> str:
    """
    爻位と行動から結果を予測（改善版）

    score 1 = Success
    score 2 = PartialSuccess
    score 3 = Mixed
    score 4 = Failure

    pattern_typeとbefore_stateも考慮
    """
    compat = calculate_compatibility(action_type, yao_position, compatibility_matrix)
    score = compat.get("score", 3)

    # pattern_typeによる補正
    if pattern_type == "Hubris_Collapse":
        # 慢心パターンは基本的にFailure傾向
        if score <= 2:
            score = min(score + 1, 4)

    elif pattern_type == "Endurance":
        # 耐え忍びパターンでは潜伏系アクションが有効
        normalized = normalize_action(action_type)
        if normalized == "耐える・潜伏" and yao_position <= 2:
            score = max(score - 1, 1)

    elif pattern_type in ["Pivot_Success", "Shock_Recovery"]:
        # 転換成功/ショック回復は刷新・対話が有効
        normalized = normalize_action(action_type)
        if normalized in ["刷新・破壊", "対話・融合", "捨てる・撤退"]:
            score = max(score - 1, 1)

    elif pattern_type == "Slow_Decline":
        # じわじわ衰退は守りが悪化を招く
        normalized = normalize_action(action_type)
        if normalized in ["守る・維持", "耐える・潜伏"]:
            score = min(score + 1, 4)

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
    1件のケースに爻情報を付与（改善版）
    """
    enriched = case.copy()

    # 1. 開始時の卦番号を取得
    before_hex_name = case.get("classical_before_hexagram", "")
    before_hex_id = parse_hexagram_name(before_hex_name)

    # 2. 爻位を診断（改善版）
    yao_position = diagnose_yao_position(case)

    # 3. 行動との適合性を計算（改善版）
    action_type = case.get("action_type", "")
    pattern_type = case.get("pattern_type", "")
    before_state = case.get("before_state", "")
    compat = calculate_compatibility(action_type, yao_position, compatibility)

    # 4. 結果を予測（改善版）
    predicted = predict_outcome(yao_position, action_type, compatibility, pattern_type, before_state)
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
            "is_optimal": compat.get("score") == 1,
            "original_action": compat.get("original_action", action_type),
            "normalized_action": compat.get("normalized_action", action_type)
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
    print("=== ケースデータへの爻情報付与（v2 改善版）===\n")

    # マッピングデータ読み込み
    print("1. マッピングデータ読み込み...")
    recommendations, compatibility, transitions = load_mappings()
    yao_phrases = load_yao_phrases()
    print(f"   - 爻推奨: {len(recommendations)} 件")
    print(f"   - 適合性マトリクス: {len(compatibility)} × 6 = {len(compatibility) * 6} 件")
    print(f"   - 爻辞: {len(yao_phrases)} 件")
    print(f"   - 拡張状態マッピング: {len(STATE_TO_YAO)} 件")
    print(f"   - 行動正規化マッピング: {len(ACTION_NORMALIZE)} 件")

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
    print("\n3. 爻情報付与（改善版ロジック）...")
    enriched_cases = []
    stats = {"exact": 0, "close": 0, "miss": 0}
    yao_stats = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

    for case in cases:
        enriched = enrich_case(case, recommendations, compatibility, transitions, yao_phrases)
        enriched_cases.append(enriched)

        # 統計更新
        yao_analysis = enriched.get("yao_analysis", {})
        accuracy = yao_analysis.get("prediction_analysis", {}).get("accuracy", "miss")
        stats[accuracy] = stats.get(accuracy, 0) + 1

        yao_pos = yao_analysis.get("before_yao_position", 3)
        yao_stats[yao_pos] = yao_stats.get(yao_pos, 0) + 1

    # 結果を保存
    print("\n4. 結果保存...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for case in enriched_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"   - {OUTPUT_FILE} に保存完了")

    # 統計出力
    total = len(cases)
    print("\n=== 予測精度統計（改善版）===")
    print(f"完全一致: {stats['exact']} 件 ({stats['exact']/total*100:.1f}%)")
    print(f"近い結果: {stats['close']} 件 ({stats['close']/total*100:.1f}%)")
    print(f"乖離: {stats['miss']} 件 ({stats['miss']/total*100:.1f}%)")
    print(f"合計: {total} 件")
    print(f"\n精度スコア（一致+近い）: {(stats['exact']+stats['close'])/total*100:.1f}%")

    print("\n=== 爻位分布 ===")
    for yao in range(1, 7):
        count = yao_stats.get(yao, 0)
        print(f"{yao}爻: {count} 件 ({count/total*100:.1f}%)")

    # サンプル出力
    print("\n=== サンプル出力（最初の3件）===")
    for i, case in enumerate(enriched_cases[:3]):
        print(f"\n--- {i+1}. {case.get('target_name')} ---")
        yao = case.get("yao_analysis", {})
        print(f"爻位: {yao.get('before_yao_position')}爻 ({yao.get('yao_stage')})")
        print(f"行動: {case.get('action_type')} → 正規化: {yao.get('action_compatibility', {}).get('normalized_action')}")
        print(f"適合性: スコア{yao.get('action_compatibility', {}).get('score')} - {yao.get('action_compatibility', {}).get('reason')}")
        print(f"予測: {yao.get('predicted_outcome')} → 実際: {yao.get('actual_outcome')}")
        print(f"分析: {yao.get('prediction_analysis', {}).get('note')}")


if __name__ == "__main__":
    main()
