#!/usr/bin/env python3
"""
爻（こう）付与スクリプト

全事例に爻（1-6）を付与し、384パターンの一意IDを生成する。

付与ルール:
- pattern_typeがあれば優先して判定
- なければoutcomeで判定
- どちらもなければデフォルト二爻

追加フィールド:
- yao: 爻番号（1-6）
- yao_name: 爻名（初爻、二爻、三爻、四爻、五爻、上爻）
- hexagram_yao_id: 384パターンの一意ID（例: "01-5" = 乾為天の五爻）
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

# 爻番号 → 爻名のマッピング
YAO_NAMES = {
    1: "初爻",
    2: "二爻",
    3: "三爻",
    4: "四爻",
    5: "五爻",
    6: "上爻"
}

# pattern_type → 爻のマッピング（優先判定）
PATTERN_TO_YAO = {
    # 成功・成長 → 五爻（最良、成功）
    'Steady_Growth': 5,
    'Breakthrough': 5,
    'Pivot_Success': 5,

    # 持久・探索 → 二爻（準備、中庸）
    'Endurance': 2,
    'Exploration': 2,

    # 危機からの回復 → 四爻（不安定、変化中）
    'Crisis_Pivot': 4,
    'Shock_Recovery': 4,

    # 下降・困難 → 三爻（転換点、危険）
    'Slow_Decline': 3,
    'Managed_Decline': 3,
    'Decline': 3,
    'Stagnation': 3,

    # 失敗の始まり → 初爻（始まり、潜在的）
    'Failed_Attempt': 1,

    # 極限・終焉 → 上爻（極限、終わり）
    'Hubris_Collapse': 6,
    'Quiet_Fade': 6,
}

# outcome → 爻のマッピング（pattern_typeがない場合のフォールバック）
OUTCOME_TO_YAO = {
    'Success': 5,       # 成功 → 五爻
    'Failure': 3,       # 失敗 → 三爻
    'Mixed': 4,         # 混合 → 四爻
    'PartialSuccess': 2,  # 部分的成功 → 二爻
}

def determine_yao(case: dict) -> int:
    """事例から爻番号（1-6）を決定する"""
    pattern_type = case.get('pattern_type', '')
    outcome = case.get('outcome', '')

    # pattern_typeがあれば優先
    if pattern_type in PATTERN_TO_YAO:
        return PATTERN_TO_YAO[pattern_type]

    # outcomeで判定
    if outcome in OUTCOME_TO_YAO:
        return OUTCOME_TO_YAO[outcome]

    # デフォルトは二爻（準備、中庸）
    return 2

def format_hexagram_yao_id(hexagram_number: int, yao: int) -> str:
    """384パターンの一意IDを生成

    例: hexagram_number=1, yao=5 → "01-5"
    """
    return f"{hexagram_number:02d}-{yao}"

def main():
    cases_path = Path(__file__).parent.parent.parent / "data" / "raw" / "cases.jsonl"

    if not cases_path.exists():
        print(f"Error: {cases_path} not found")
        sys.exit(1)

    # 統計用カウンター
    yao_counter = Counter()
    pattern_yao_counter = defaultdict(Counter)
    outcome_yao_counter = defaultdict(Counter)
    hexagram_yao_counter = defaultdict(Counter)  # 卦番号×爻の分布
    unmatched_patterns = Counter()
    unmatched_outcomes = Counter()

    # 事例の読み込みと爻付与
    updated_cases = []
    total = 0

    print("Loading cases and assigning yao...")

    with open(cases_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            case = json.loads(line)
            total += 1

            # 爻の決定
            yao = determine_yao(case)
            yao_name = YAO_NAMES[yao]

            # hexagram_numberの取得（なければhexagram_idを使用）
            hexagram_number = case.get('hexagram_number')
            if hexagram_number is None:
                hexagram_number = case.get('hexagram_id', 0)

            # hexagram_yao_idの生成
            if hexagram_number and hexagram_number > 0:
                hexagram_yao_id = format_hexagram_yao_id(hexagram_number, yao)
            else:
                hexagram_yao_id = f"00-{yao}"  # 卦番号がない場合

            # フィールドの追加
            case['yao'] = yao
            case['yao_name'] = yao_name
            case['hexagram_yao_id'] = hexagram_yao_id

            updated_cases.append(case)

            # 統計の収集
            yao_counter[yao] += 1

            pattern_type = case.get('pattern_type', 'NONE')
            outcome = case.get('outcome', 'NONE')

            pattern_yao_counter[pattern_type][yao] += 1
            outcome_yao_counter[outcome][yao] += 1

            if hexagram_number and hexagram_number > 0:
                hexagram_yao_counter[hexagram_number][yao] += 1

            # マッチしなかったパターン/アウトカムを記録
            if pattern_type not in PATTERN_TO_YAO and pattern_type != 'NONE':
                unmatched_patterns[pattern_type] += 1
            if outcome not in OUTCOME_TO_YAO and outcome != 'NONE':
                unmatched_outcomes[outcome] += 1

    print(f"Total cases processed: {total}")

    # 結果の書き込み
    print(f"\nWriting updated cases to {cases_path}...")

    with open(cases_path, 'w', encoding='utf-8') as f:
        for case in updated_cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')

    print(f"Successfully updated {len(updated_cases)} cases")

    # 統計レポート
    print("\n" + "=" * 60)
    print("爻付与統計レポート")
    print("=" * 60)

    print("\n【爻別分布】")
    print("-" * 40)
    for yao in sorted(yao_counter.keys()):
        count = yao_counter[yao]
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {YAO_NAMES[yao]} (爻{yao}): {count:,}件 ({pct:.1f}%) {bar}")

    print("\n【pattern_type別の爻分布】")
    print("-" * 40)
    for pattern in sorted(pattern_yao_counter.keys()):
        if pattern == 'NONE':
            continue
        yao_dist = pattern_yao_counter[pattern]
        total_pattern = sum(yao_dist.values())
        yao_list = ', '.join([f"爻{y}:{c}件" for y, c in sorted(yao_dist.items())])
        print(f"  {pattern}: {total_pattern}件 → {yao_list}")

    print("\n【outcome別の爻分布】")
    print("-" * 40)
    for outcome in sorted(outcome_yao_counter.keys()):
        if outcome == 'NONE':
            continue
        yao_dist = outcome_yao_counter[outcome]
        total_outcome = sum(yao_dist.values())
        yao_list = ', '.join([f"爻{y}:{c}件" for y, c in sorted(yao_dist.items())])
        print(f"  {outcome}: {total_outcome}件 → {yao_list}")

    print("\n【384パターン（64卦×6爻）の分布】")
    print("-" * 40)

    # 384パターンのカバレッジ
    covered_patterns = set()
    for hex_num in hexagram_yao_counter:
        for yao in hexagram_yao_counter[hex_num]:
            covered_patterns.add((hex_num, yao))

    total_patterns = 64 * 6  # 384
    coverage = len(covered_patterns) / total_patterns * 100
    print(f"  カバレッジ: {len(covered_patterns)}/{total_patterns} ({coverage:.1f}%)")

    # 卦ごとの爻分布サマリー
    print("\n【卦×爻の詳細分布（上位10卦）】")
    print("-" * 40)

    hex_totals = {h: sum(yao_dist.values()) for h, yao_dist in hexagram_yao_counter.items()}
    top_hexagrams = sorted(hex_totals.items(), key=lambda x: x[1], reverse=True)[:10]

    for hex_num, total_hex in top_hexagrams:
        yao_dist = hexagram_yao_counter[hex_num]
        yao_str = ' '.join([f"{yao}爻:{yao_dist.get(yao, 0)}" for yao in range(1, 7)])
        print(f"  卦{hex_num:02d}: {total_hex:>4}件 | {yao_str}")

    # 空セル（0件のパターン）
    empty_patterns = []
    for hex_num in range(1, 65):
        for yao in range(1, 7):
            if (hex_num, yao) not in covered_patterns:
                empty_patterns.append((hex_num, yao))

    if empty_patterns:
        print(f"\n【空セル（0件のパターン）: {len(empty_patterns)}件】")
        print("-" * 40)
        # 最初の20件を表示
        for hex_num, yao in empty_patterns[:20]:
            print(f"  {hex_num:02d}-{yao} ({YAO_NAMES[yao]})")
        if len(empty_patterns) > 20:
            print(f"  ... 他 {len(empty_patterns) - 20} 件")

    # マッチしなかったパターン
    if unmatched_patterns:
        print("\n【未マッチのpattern_type（デフォルト適用）】")
        print("-" * 40)
        for pattern, count in unmatched_patterns.most_common():
            print(f"  {pattern}: {count}件")

    if unmatched_outcomes:
        print("\n【未マッチのoutcome（デフォルト適用）】")
        print("-" * 40)
        for outcome, count in unmatched_outcomes.most_common():
            print(f"  {outcome}: {count}件")

    print("\n" + "=" * 60)
    print("完了")
    print("=" * 60)

if __name__ == "__main__":
    main()
