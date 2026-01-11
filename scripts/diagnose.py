#!/usr/bin/env python3
"""
実例ベース診断ツール - ユーザーの状況に類似する事例を検索してアドバイスを提供
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
from schema_v3 import Case, Scale, BeforeState, TriggerType, Hex

# 成功パターンの卦ペア（分析結果より抽出）
SUCCESS_PATTERNS = {
    # トランジション1（初期→トリガー）: 震→乾, 坎→乾, 艮→乾など
    "transition_1": {
        "震→乾": 15, "坎→乾": 15, "艮→乾": 12, "艮→巽": 10, "坤→乾": 10,
    },
    # トランジション2（トリガー→行動）: 乾→離が最強パターン
    "transition_2": {
        "乾→離": 20, "離→艮": 12, "乾→巽": 10, "巽→乾": 8, "乾→艮": 8,
    },
    # トランジション3（行動→結果）: 離→乾, 巽→乾など
    "transition_3": {
        "離→乾": 15, "巽→乾": 12, "離→離": 12, "艮→乾": 12, "乾→兌": 10,
    }
}

# 失敗パターンの卦ペア
FAILURE_PATTERNS = {
    # トランジション1: 乾→震が最大の失敗パターン
    "transition_1": {
        "乾→震": -20, "乾→離": -5, "艮→坎": -3,
    },
    # トランジション2
    "transition_2": {
        "震→離": -8, "震→坎": -6, "震→乾": -5,
    },
    # トランジション3
    "transition_3": {
        "離→坎": -10, "坎→坤": -6, "坎→坎": -6, "乾→坎": -5,
    }
}

def calculate_similarity(user_case: Dict, db_case: Case) -> Tuple[float, Dict]:
    """
    ユーザーの状況とデータベース事例の類似度を計算

    スコアリング要素（優先度順）：
    1. スケールの一致（必須条件）- 100点
    2. 初期状態の一致 - 50点
    3. トリガータイプの一致 - 30点
    4. 初期の卦の一致 - 20点
    5. 成功/失敗パターンによる調整 - ±30点（NEW!）
    6. 変爻パターンの一致 - 15点（NEW!）
    7. 信頼性ランクボーナス - 10点

    Returns:
        (similarity_score, match_details)
    """
    score = 0.0
    max_score = 0.0
    details = {}

    # 1. スケールの完全一致（必須条件） - 重み: 100点
    max_score += 100
    if user_case.get("scale") == db_case.scale.value:
        score += 100
        details["scale_match"] = "完全一致"
    else:
        details["scale_match"] = "不一致"
        return (0.0, details)  # スケールが違う場合は候補から除外

    # 2. 初期状態（before_state）の一致 - 重み: 50点
    max_score += 50
    if user_case.get("before_state") == db_case.before_state.value:
        score += 50
        details["before_state_match"] = "完全一致"
    else:
        details["before_state_match"] = "不一致"

    # 3. トリガータイプの一致 - 重み: 30点
    max_score += 30
    if user_case.get("trigger_type"):
        if user_case.get("trigger_type") == db_case.trigger_type.value:
            score += 30
            details["trigger_type_match"] = "完全一致"
        else:
            details["trigger_type_match"] = "不一致"
    else:
        # ユーザーが指定しない場合はスコアに含めない
        max_score -= 30
        details["trigger_type_match"] = "未指定"

    # 4. before_hex（初期状態の卦）の一致 - 重み: 20点
    max_score += 20
    if user_case.get("before_hex"):
        if user_case.get("before_hex") == db_case.before_hex.value:
            score += 20
            details["before_hex_match"] = "完全一致"
        else:
            details["before_hex_match"] = "不一致"
    else:
        max_score -= 20
        details["before_hex_match"] = "未指定"

    # 5. 【NEW!】成功/失敗パターンによる調整 - 重み: 30点
    # データ分析により判明した成功・失敗に関連する卦ペアで加減点
    max_score += 30
    pattern_bonus = 0
    pattern_details = []

    # トランジション1の評価
    pair_1 = f"{db_case.before_hex.value}→{db_case.trigger_hex.value}"
    if pair_1 in SUCCESS_PATTERNS["transition_1"]:
        bonus = SUCCESS_PATTERNS["transition_1"][pair_1]
        pattern_bonus += bonus
        pattern_details.append(f"成功パターン1:{pair_1}(+{bonus})")
    elif pair_1 in FAILURE_PATTERNS["transition_1"]:
        penalty = FAILURE_PATTERNS["transition_1"][pair_1]
        pattern_bonus += penalty
        pattern_details.append(f"失敗パターン1:{pair_1}({penalty})")

    # トランジション2の評価
    pair_2 = f"{db_case.trigger_hex.value}→{db_case.action_hex.value}"
    if pair_2 in SUCCESS_PATTERNS["transition_2"]:
        bonus = SUCCESS_PATTERNS["transition_2"][pair_2]
        pattern_bonus += bonus
        pattern_details.append(f"成功パターン2:{pair_2}(+{bonus})")
    elif pair_2 in FAILURE_PATTERNS["transition_2"]:
        penalty = FAILURE_PATTERNS["transition_2"][pair_2]
        pattern_bonus += penalty
        pattern_details.append(f"失敗パターン2:{pair_2}({penalty})")

    # トランジション3の評価
    pair_3 = f"{db_case.action_hex.value}→{db_case.after_hex.value}"
    if pair_3 in SUCCESS_PATTERNS["transition_3"]:
        bonus = SUCCESS_PATTERNS["transition_3"][pair_3]
        pattern_bonus += bonus
        pattern_details.append(f"成功パターン3:{pair_3}(+{bonus})")
    elif pair_3 in FAILURE_PATTERNS["transition_3"]:
        penalty = FAILURE_PATTERNS["transition_3"][pair_3]
        pattern_bonus += penalty
        pattern_details.append(f"失敗パターン3:{pair_3}({penalty})")

    # 30点満点に正規化（最大+60点、最小-60点の範囲を0-30に）
    normalized_pattern = max(0, min(30, (pattern_bonus + 60) / 4))
    score += normalized_pattern
    details["pattern_bonus"] = f"{pattern_bonus:+d}点 -> {normalized_pattern:.1f}点"
    details["pattern_details"] = pattern_details if pattern_details else ["なし"]

    # 6. 【NEW!】変爻パターンの一致 - 重み: 15点
    # ユーザーが変爻情報を提供した場合、一致度で加点
    max_score += 15
    if user_case.get("changing_lines_1") is not None:
        user_lines = set(user_case.get("changing_lines_1", []))
        case_lines = set(db_case.changing_lines_1 or [])

        if user_lines and case_lines:
            # Jaccard係数で類似度を計算
            intersection = len(user_lines & case_lines)
            union = len(user_lines | case_lines)
            similarity = intersection / union if union > 0 else 0
            line_score = similarity * 15
            score += line_score
            details["changing_lines_match"] = f"類似度{similarity:.2f} ({intersection}/{union})"
        else:
            details["changing_lines_match"] = "変爻なし"
    else:
        # ユーザーが指定しない場合はスコアに含めない
        max_score -= 15
        details["changing_lines_match"] = "未指定"

    # 7. 信頼性ランクによるボーナス - 重み: 10点
    max_score += 10
    credibility_bonus = {
        "S": 10,
        "A": 7,
        "B": 4,
        "C": 2
    }
    score += credibility_bonus.get(db_case.credibility_rank.value, 0)
    details["credibility_rank"] = db_case.credibility_rank.value

    # 正規化（0-100のスコアに）
    normalized_score = (score / max_score) * 100 if max_score > 0 else 0

    return (normalized_score, details)

def diagnose(user_input: Dict, db_path: Path, top_n: int = 10) -> List[Tuple[Case, float, Dict]]:
    """
    ユーザーの状況に基づいて類似事例を検索

    Args:
        user_input: ユーザーの状況 {"scale": "individual", "before_state": "どん底・危機", ...}
        db_path: データベースのパス
        top_n: 返す事例の数

    Returns:
        [(事例, 類似度スコア, マッチ詳細), ...]
    """
    matches = []

    with open(db_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)
            case = Case(**data)

            similarity_score, match_details = calculate_similarity(user_input, case)

            if similarity_score > 0:  # スケールが一致する事例のみ
                matches.append((case, similarity_score, match_details))

    # 類似度スコアでソート
    matches.sort(key=lambda x: x[1], reverse=True)

    return matches[:top_n]

def format_diagnosis_result(matches: List[Tuple[Case, float, Dict]]) -> str:
    """診断結果を読みやすい形式にフォーマット"""

    if not matches:
        return "該当する事例が見つかりませんでした。"

    output = []
    output.append("=" * 80)
    output.append("易経変化ロジックDB - 診断結果")
    output.append("=" * 80)
    output.append(f"\n{len(matches)}件の類似事例が見つかりました\n")

    for i, (case, score, details) in enumerate(matches, 1):
        output.append("-" * 80)
        output.append(f"\n【第{i}位】類似度: {score:.1f}%")
        output.append(f"対象: {case.target_name}")
        output.append(f"スケール: {case.scale.value}")
        output.append(f"期間: {case.period}")
        output.append(f"\n■ ストーリー:")
        output.append(f"  {case.story_summary}")
        output.append(f"\n■ 変化の流れ:")
        output.append(f"  初期状態: {case.before_state.value} ({case.before_hex.value})")
        output.append(f"  トリガー: {case.trigger_type.value} ({case.trigger_hex.value})")
        output.append(f"  行動: {case.action_type.value} ({case.action_hex.value})")
        output.append(f"  結果: {case.after_state.value} ({case.after_hex.value})")
        output.append(f"\n■ 結果: {case.outcome.value}")
        output.append(f"  パターン: {case.pattern_type.value}")
        output.append(f"\n■ 易経ロジック:")
        output.append(f"  {case.logic_memo}")
        output.append(f"\n■ マッチ詳細:")
        output.append(f"  スケール: {details.get('scale_match', 'N/A')}")
        output.append(f"  初期状態: {details.get('before_state_match', 'N/A')}")
        output.append(f"  トリガー: {details.get('trigger_type_match', 'N/A')}")
        output.append(f"  初期の卦: {details.get('before_hex_match', 'N/A')}")
        output.append(f"  パターン評価: {details.get('pattern_bonus', 'N/A')}")
        pattern_details_list = details.get('pattern_details', [])
        if pattern_details_list and pattern_details_list != ["なし"]:
            for pd in pattern_details_list:
                output.append(f"    - {pd}")
        output.append(f"  変爻一致: {details.get('changing_lines_match', 'N/A')}")
        output.append(f"  信頼性: {details.get('credibility_rank', 'N/A')}")
        output.append(f"\nタグ: {', '.join(case.free_tags)}")
        output.append("")

    return "\n".join(output)

def main():
    """対話型の診断ツール"""
    db_path = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"

    print("=" * 80)
    print("易経変化ロジックDB - 実例ベース診断ツール")
    print("=" * 80)
    print("\nあなたの現在の状況を教えてください。\n")

    # スケールの選択
    print("1. スケール（規模）を選んでください:")
    scales = ["individual", "company", "family", "country", "other"]
    for i, s in enumerate(scales, 1):
        print(f"  {i}. {s}")

    while True:
        try:
            scale_choice = int(input("\n選択 (1-5): "))
            if 1 <= scale_choice <= 5:
                scale = scales[scale_choice - 1]
                break
        except ValueError:
            pass
        print("1-5の数字を入力してください")

    # 初期状態の選択
    print("\n2. 現在の状態を選んでください:")
    before_states = [
        "絶頂・慢心",
        "停滞・閉塞",
        "混乱・カオス",
        "成長痛",
        "どん底・危機",
        "安定・平和"
    ]
    for i, s in enumerate(before_states, 1):
        print(f"  {i}. {s}")

    while True:
        try:
            state_choice = int(input("\n選択 (1-6): "))
            if 1 <= state_choice <= 6:
                before_state = before_states[state_choice - 1]
                break
        except ValueError:
            pass
        print("1-6の数字を入力してください")

    # トリガータイプの選択（オプション）
    print("\n3. きっかけのタイプ（わかれば）:")
    trigger_types = [
        "外部ショック",
        "内部崩壊",
        "意図的決断",
        "偶発・出会い",
        "スキップ（指定しない）"
    ]
    for i, t in enumerate(trigger_types, 1):
        print(f"  {i}. {t}")

    trigger_type = None
    while True:
        try:
            trigger_choice = int(input("\n選択 (1-5): "))
            if 1 <= trigger_choice <= 5:
                if trigger_choice < 5:
                    trigger_type = trigger_types[trigger_choice - 1]
                break
        except ValueError:
            pass
        print("1-5の数字を入力してください")

    # ユーザー入力を構築
    user_input = {
        "scale": scale,
        "before_state": before_state
    }
    if trigger_type:
        user_input["trigger_type"] = trigger_type

    print("\n診断中...\n")

    # 診断実行
    matches = diagnose(user_input, db_path, top_n=10)

    # 結果表示
    result = format_diagnosis_result(matches)
    print(result)

    # 統計サマリー
    if matches:
        print("\n" + "=" * 80)
        print("診断サマリー")
        print("=" * 80)

        # 結果の分布
        outcomes = defaultdict(int)
        patterns = defaultdict(int)
        for case, _, _ in matches:
            outcomes[case.outcome.value] += 1
            patterns[case.pattern_type.value] += 1

        print("\n■ 結果の傾向:")
        for outcome, count in sorted(outcomes.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(matches) * 100
            print(f"  {outcome}: {count}件 ({pct:.0f}%)")

        print("\n■ パターンの傾向:")
        for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(matches) * 100
            print(f"  {pattern}: {count}件 ({pct:.0f}%)")

if __name__ == "__main__":
    main()
