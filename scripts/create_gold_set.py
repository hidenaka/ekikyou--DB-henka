#!/usr/bin/env python3
"""
層化抽出によるゴールドセット作成スクリプト

cases.jsonlから500件を以下の条件で層化抽出:
- source_type: news 250, book 80, official 80, blog 60, sns 30
- country: JP 300, International 200
- state_label(before/after): 各カテゴリ均等に近づける
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── 設定 ──────────────────────────────────────────────
SEED = 42
TOTAL = 500

# source_type別の目標件数
SOURCE_QUOTA = {
    "news": 250,
    "book": 80,
    "official": 80,
    "blog": 60,
    "sns": 30,
}

# country別の目標件数
COUNTRY_QUOTA = {
    "JP": 300,
    "International": 200,
}

# ── パス ──────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
CASES_PATH = BASE / "data" / "raw" / "cases.jsonl"
OUTPUT_DIR = BASE / "analysis" / "gold_set"
OUTPUT_PATH = OUTPUT_DIR / "gold_set_sample.json"


def classify_country(case: dict) -> str:
    """transition_idからJP/Internationalを判定"""
    tid = case.get("transition_id", "")
    return "JP" if "_JP_" in tid else "International"


def load_cases() -> list[dict]:
    """cases.jsonlを全件読み込み"""
    cases = []
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def stratified_sample(cases: list[dict]) -> list[dict]:
    """
    層化抽出を実行。

    戦略:
    1. source_type × country の2次元で層を作る
    2. 各層の目標件数を按分計算
    3. before_state/after_stateの均等化のため、各層内でstate分布を考慮して抽出
    """
    random.seed(SEED)

    # ── Step 1: 全事例をsource_type × countryで分類 ──
    buckets = defaultdict(list)  # (source_type, country) -> [case, ...]
    for case in cases:
        st = case.get("source_type", "unknown")
        country = classify_country(case)
        buckets[(st, country)].append(case)

    # ── Step 2: 各(source_type, country)の目標件数を按分 ──
    # source_type quotaとcountry quotaを同時に満たすよう按分
    total_cases = len(cases)
    country_counts = Counter(classify_country(c) for c in cases)
    source_counts = Counter(c.get("source_type", "unknown") for c in cases)

    targets = {}  # (source_type, country) -> target_count
    for st, st_quota in SOURCE_QUOTA.items():
        for country, co_quota in COUNTRY_QUOTA.items():
            bucket_size = len(buckets.get((st, country), []))
            if bucket_size == 0:
                targets[(st, country)] = 0
                continue
            # 按分: source_typeの割り当て × (このcountryの割合)
            # country比率をcountry quotaから算出
            country_ratio = co_quota / TOTAL
            target = round(st_quota * country_ratio)
            # バケットサイズを超えないようにキャップ
            targets[(st, country)] = min(target, bucket_size)

    # ── Step 3: before_state均等化のため層内でstate-aware抽出 ──
    # before_stateの全カテゴリを取得
    all_before_states = sorted(set(c.get("before_state", "unknown") for c in cases))
    all_after_states = sorted(set(c.get("after_state", "unknown") for c in cases))

    selected = []

    for (st, country), target in sorted(targets.items()):
        if target == 0:
            continue
        pool = buckets[(st, country)]

        # before_stateでサブ分類
        by_before = defaultdict(list)
        for c in pool:
            by_before[c.get("before_state", "unknown")].append(c)

        # 各before_stateから均等に抽出
        n_states = len(by_before)
        if n_states == 0:
            continue

        per_state = max(1, target // n_states)
        remainder = target - per_state * n_states

        state_keys = sorted(by_before.keys())
        random.shuffle(state_keys)

        picked_from_bucket = []
        for i, bs in enumerate(state_keys):
            n = per_state + (1 if i < remainder else 0)
            available = by_before[bs]
            # after_state多様性のためシャッフル
            random.shuffle(available)
            picked_from_bucket.extend(available[:n])

        # 目標数に調整
        if len(picked_from_bucket) > target:
            picked_from_bucket = picked_from_bucket[:target]
        elif len(picked_from_bucket) < target:
            # 不足分は残りのプールから補充
            picked_ids = {c.get("transition_id", id(c)) for c in picked_from_bucket}
            remaining = [c for c in pool if c.get("transition_id", id(c)) not in picked_ids]
            random.shuffle(remaining)
            need = target - len(picked_from_bucket)
            picked_from_bucket.extend(remaining[:need])

        selected.extend(picked_from_bucket)

    # ── Step 4: 総数調整 ──
    if len(selected) > TOTAL:
        random.shuffle(selected)
        selected = selected[:TOTAL]
    elif len(selected) < TOTAL:
        # 不足分をnews/JPから補充
        selected_ids = {c.get("transition_id", id(c)) for c in selected}
        remaining = [c for c in cases if c.get("transition_id", id(c)) not in selected_ids]
        random.shuffle(remaining)
        need = TOTAL - len(selected)
        selected.extend(remaining[:need])

    return selected


def build_output(selected: list[dict]) -> list[dict]:
    """ゴールドセット用の出力フォーマットを構築"""
    output = []
    for case in selected:
        entry = {
            "transition_id": case.get("transition_id"),
            "target_name": case.get("target_name"),
            "scale": case.get("scale"),
            "period": case.get("period"),
            "story_summary": case.get("story_summary"),
            "before_state": case.get("before_state"),
            "before_summary": case.get("before_state"),  # before_stateをsummaryとして利用
            "trigger_type": case.get("trigger_type"),
            "action_type": case.get("action_type"),
            "after_state": case.get("after_state"),
            "after_summary": case.get("after_state"),    # after_stateをsummaryとして利用
            "source_type": case.get("source_type"),
            "country": classify_country(case),
            # 既存のhexagramラベル（比較用に保持）
            "original_before_hex": case.get("before_hex"),
            "original_after_hex": case.get("after_hex"),
            "original_classical_before": case.get("classical_before_hexagram"),
            "original_classical_after": case.get("classical_after_hexagram"),
            # アノテーション対象フィールド（空欄）
            "gold_before_lower": None,
            "gold_before_upper": None,
            "gold_before_hexagram": None,
            "gold_before_reasoning": None,
            "gold_after_lower": None,
            "gold_after_upper": None,
            "gold_after_hexagram": None,
            "gold_after_reasoning": None,
        }
        output.append(entry)
    return output


def print_stats(selected: list[dict]):
    """抽出結果の統計を出力"""
    print(f"\n{'='*60}")
    print(f"ゴールドセット抽出結果: {len(selected)}件")
    print(f"{'='*60}")

    # source_type別
    st_counts = Counter(c.get("source_type") for c in selected)
    print(f"\n--- source_type別 (目標: news=250, book=80, official=80, blog=60, sns=30) ---")
    for st in ["news", "book", "official", "blog", "sns"]:
        actual = st_counts.get(st, 0)
        target = SOURCE_QUOTA.get(st, 0)
        print(f"  {st:10s}: {actual:4d} / {target:4d} (差: {actual - target:+d})")

    # country別
    co_counts = Counter(classify_country(c) for c in selected)
    print(f"\n--- country別 (目標: JP=300, International=200) ---")
    for co in ["JP", "International"]:
        actual = co_counts.get(co, 0)
        target = COUNTRY_QUOTA.get(co, 0)
        print(f"  {co:15s}: {actual:4d} / {target:4d} (差: {actual - target:+d})")

    # before_state別
    bs_counts = Counter(c.get("before_state") for c in selected)
    print(f"\n--- before_state別 ---")
    for bs, cnt in bs_counts.most_common():
        print(f"  {bs:20s}: {cnt:4d} ({cnt/len(selected)*100:.1f}%)")

    # after_state別
    as_counts = Counter(c.get("after_state") for c in selected)
    print(f"\n--- after_state別 ---")
    for astate, cnt in as_counts.most_common():
        print(f"  {astate:20s}: {cnt:4d} ({cnt/len(selected)*100:.1f}%)")

    # scale別
    sc_counts = Counter(c.get("scale") for c in selected)
    print(f"\n--- scale別 ---")
    for sc, cnt in sc_counts.most_common():
        print(f"  {sc:15s}: {cnt:4d} ({cnt/len(selected)*100:.1f}%)")

    # source_type × country クロス集計
    cross = Counter((c.get("source_type"), classify_country(c)) for c in selected)
    print(f"\n--- source_type × country クロス集計 ---")
    print(f"  {'':10s} {'JP':>6s} {'Intl':>6s} {'Total':>6s}")
    for st in ["news", "book", "official", "blog", "sns"]:
        jp = cross.get((st, "JP"), 0)
        intl = cross.get((st, "International"), 0)
        print(f"  {st:10s} {jp:6d} {intl:6d} {jp+intl:6d}")


def main():
    print("cases.jsonlを読み込み中...")
    cases = load_cases()
    print(f"  全件数: {len(cases)}")

    print("層化抽出を実行中...")
    selected = stratified_sample(cases)

    print("出力ファイルを作成中...")
    output = build_output(selected)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  出力: {OUTPUT_PATH}")

    print_stats(selected)

    return 0


if __name__ == "__main__":
    sys.exit(main())
