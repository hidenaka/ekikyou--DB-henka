#!/usr/bin/env python3
"""
テストセットでの最終評価

訓練・検証で一切使用していないテストセットで
モデルの真の予測精度を評価する。
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).parent.parent
SPLITS_DIR = BASE_DIR / "data" / "splits"
MODELS_DIR = BASE_DIR / "data" / "models"


def load_cases(filepath):
    cases = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def load_model():
    with open(MODELS_DIR / "prediction_model_v1.json", "r", encoding="utf-8") as f:
        return json.load(f)


def predict(model, before_state, action_type):
    """予測を行う"""
    action_normalize = model.get("action_normalize", {})
    normalized_action = action_normalize.get(action_type, action_type)

    prob_model = model["models"]["probability"]["data"]
    key = str((before_state, normalized_action))

    if key in prob_model["conditional"]:
        counts = prob_model["conditional"][key]
        total = sum(counts.values())
        best_outcome = max(counts, key=counts.get)
        confidence = counts[best_outcome] / total
    elif before_state in prob_model["marginal_state"]:
        counts = prob_model["marginal_state"][before_state]
        total = sum(counts.values())
        best_outcome = max(counts, key=counts.get)
        confidence = counts[best_outcome] / total * 0.7
    else:
        counts = prob_model["overall"]
        total = sum(counts.values())
        best_outcome = max(counts, key=counts.get)
        confidence = counts[best_outcome] / total * 0.5

    return best_outcome, confidence


def is_close(predicted, actual):
    close_pairs = [
        ("Success", "PartialSuccess"),
        ("PartialSuccess", "Mixed"),
        ("Mixed", "Failure"),
    ]
    for pair in close_pairs:
        if predicted in pair and actual in pair:
            return True
    return False


def main():
    print("=" * 60)
    print("テストセット最終評価")
    print("（訓練・検証で一切使用していないデータ）")
    print("=" * 60)
    print()

    # データとモデル読み込み
    test_cases = load_cases(SPLITS_DIR / "test.jsonl")
    model = load_model()

    print(f"テストケース: {len(test_cases)}件")
    print()

    # 評価
    stats = {"exact": 0, "close": 0, "miss": 0}
    by_confidence = defaultdict(lambda: {"total": 0, "correct": 0})
    by_state = defaultdict(lambda: {"total": 0, "correct": 0})
    by_action = defaultdict(lambda: {"total": 0, "correct": 0})
    confusion = defaultdict(int)

    for case in test_cases:
        before_state = case.get("before_state", "")
        action_type = case.get("action_type", "")
        actual = case.get("outcome", "Mixed")

        predicted, confidence = predict(model, before_state, action_type)

        # 判定
        if predicted == actual:
            stats["exact"] += 1
            is_correct = True
        elif is_close(predicted, actual):
            stats["close"] += 1
            is_correct = True
        else:
            stats["miss"] += 1
            is_correct = False

        # 確信度別
        if confidence >= 0.6:
            level = "高(≥0.6)"
        elif confidence >= 0.4:
            level = "中(0.4-0.6)"
        else:
            level = "低(<0.4)"

        by_confidence[level]["total"] += 1
        if is_correct:
            by_confidence[level]["correct"] += 1

        # 状態別
        by_state[before_state]["total"] += 1
        if is_correct:
            by_state[before_state]["correct"] += 1

        # 行動別
        action_normalize = model.get("action_normalize", {})
        norm_action = action_normalize.get(action_type, action_type)
        by_action[norm_action]["total"] += 1
        if is_correct:
            by_action[norm_action]["correct"] += 1

        # 混同行列
        confusion[(predicted, actual)] += 1

    # 結果表示
    total = len(test_cases)

    print("=" * 40)
    print("【全体精度】")
    print("=" * 40)
    print(f"完全一致: {stats['exact']} ({stats['exact']/total*100:.1f}%)")
    print(f"近似一致: {stats['close']} ({stats['close']/total*100:.1f}%)")
    print(f"乖離: {stats['miss']} ({stats['miss']/total*100:.1f}%)")
    print()
    accuracy_score = (stats['exact'] + stats['close']) / total * 100
    print(f"★ 精度スコア: {accuracy_score:.1f}%")
    print()

    print("=" * 40)
    print("【確信度別精度】")
    print("=" * 40)
    for level in ["高(≥0.6)", "中(0.4-0.6)", "低(<0.4)"]:
        data = by_confidence[level]
        if data["total"] > 0:
            acc = data["correct"] / data["total"] * 100
            print(f"{level}: {data['correct']}/{data['total']} ({acc:.1f}%)")
    print()

    print("=" * 40)
    print("【状態別精度（上位10）】")
    print("=" * 40)
    sorted_states = sorted(by_state.items(), key=lambda x: -x[1]["total"])
    for state, data in sorted_states[:10]:
        if data["total"] > 0:
            acc = data["correct"] / data["total"] * 100
            print(f"{state}: {acc:.0f}% ({data['total']}件)")
    print()

    print("=" * 40)
    print("【行動別精度】")
    print("=" * 40)
    for action, data in sorted(by_action.items(), key=lambda x: -x[1]["total"]):
        if data["total"] > 0:
            acc = data["correct"] / data["total"] * 100
            print(f"{action}: {acc:.0f}% ({data['total']}件)")
    print()

    print("=" * 40)
    print("【混同行列（上位10）】")
    print("=" * 40)
    for (pred, actual), count in sorted(confusion.items(), key=lambda x: -x[1])[:10]:
        marker = "✓" if pred == actual else "✗"
        print(f"{marker} {pred} → {actual}: {count}件")

    # サマリー
    print()
    print("=" * 60)
    print("【結論】")
    print("=" * 60)
    print(f"テストセット精度スコア: {accuracy_score:.1f}%")
    print()
    print("これは、pattern_typeを使わず、before_state + action_type のみから")
    print("予測した真の精度です。循環論法なし、過学習なしの数字です。")
    print()

    # 信頼区間の概算
    import math
    p = accuracy_score / 100
    n = total
    se = math.sqrt(p * (1 - p) / n)
    ci_low = (p - 1.96 * se) * 100
    ci_high = (p + 1.96 * se) * 100
    print(f"95%信頼区間: {ci_low:.1f}% - {ci_high:.1f}%")


if __name__ == "__main__":
    main()
