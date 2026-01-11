#!/usr/bin/env python3
"""
データ分割スクリプト

訓練/検証/テストセットへの分割を行う。
層化抽出により、各セットでの分布を維持する。
"""

import json
import random
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
CASES_FILE = BASE_DIR / "data" / "raw" / "cases.jsonl"
SPLITS_DIR = BASE_DIR / "data" / "splits"

# 分割比率
TRAIN_RATIO = 0.6
VALID_RATIO = 0.2
TEST_RATIO = 0.2

# 乱数シード（再現性のため）
RANDOM_SEED = 42


def load_cases():
    """ケースデータを読み込み"""
    cases = []
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def stratified_split(cases, train_ratio, valid_ratio, test_ratio, seed=42):
    """
    層化抽出による分割

    outcome（結果）の分布を各セットで維持する
    """
    random.seed(seed)

    # outcomeでグループ化
    by_outcome = defaultdict(list)
    for case in cases:
        outcome = case.get("outcome", "Mixed")
        by_outcome[outcome].append(case)

    train, valid, test = [], [], []

    for outcome, outcome_cases in by_outcome.items():
        # シャッフル
        shuffled = outcome_cases.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)

        train.extend(shuffled[:n_train])
        valid.extend(shuffled[n_train:n_train + n_valid])
        test.extend(shuffled[n_train + n_valid:])

    # 最終シャッフル
    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)

    return train, valid, test


def save_split(cases, filepath):
    """分割データを保存"""
    with open(filepath, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")


def print_distribution(cases, name):
    """分布を表示"""
    print(f"\n=== {name} ({len(cases)}件) ===")

    # outcome分布
    outcomes = Counter(c.get("outcome", "") for c in cases)
    print("outcome分布:")
    for outcome, count in outcomes.most_common():
        print(f"  {outcome}: {count} ({count/len(cases)*100:.1f}%)")

    # before_state分布（上位5）
    states = Counter(c.get("before_state", "") for c in cases)
    print("before_state分布（上位5）:")
    for state, count in states.most_common(5):
        print(f"  {state}: {count}")

    # action_type分布（上位5）
    actions = Counter(c.get("action_type", "") for c in cases)
    print("action_type分布（上位5）:")
    for action, count in actions.most_common(5):
        print(f"  {action}: {count}")


def main():
    print("=== データ分割 ===")
    print(f"分割比率: 訓練{TRAIN_RATIO*100:.0f}% / 検証{VALID_RATIO*100:.0f}% / テスト{TEST_RATIO*100:.0f}%")
    print(f"乱数シード: {RANDOM_SEED}")
    print()

    # データ読み込み
    cases = load_cases()
    print(f"全データ: {len(cases)}件")

    # 分割
    train, valid, test = stratified_split(
        cases, TRAIN_RATIO, VALID_RATIO, TEST_RATIO, RANDOM_SEED
    )

    # 分布確認
    print_distribution(train, "訓練セット")
    print_distribution(valid, "検証セット")
    print_distribution(test, "テストセット")

    # 保存
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    save_split(train, SPLITS_DIR / "train.jsonl")
    save_split(valid, SPLITS_DIR / "validation.jsonl")
    save_split(test, SPLITS_DIR / "test.jsonl")

    # メタ情報保存
    meta = {
        "created_at": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "train_ratio": TRAIN_RATIO,
        "valid_ratio": VALID_RATIO,
        "test_ratio": TEST_RATIO,
        "total_cases": len(cases),
        "train_cases": len(train),
        "valid_cases": len(valid),
        "test_cases": len(test),
    }
    with open(SPLITS_DIR / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n=== 保存完了 ===")
    print(f"訓練: {SPLITS_DIR / 'train.jsonl'}")
    print(f"検証: {SPLITS_DIR / 'validation.jsonl'}")
    print(f"テスト: {SPLITS_DIR / 'test.jsonl'}")


if __name__ == "__main__":
    main()
