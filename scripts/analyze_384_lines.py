#!/usr/bin/env python3
"""
384爻の網羅性分析ツール

データベース内で使用されている変爻パターンを分析し、
どの八卦ペアのどの爻が使われているかを可視化します。
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from schema_v3 import Case

def analyze_384_lines(db_path: Path):
    """
    変爻の網羅性を分析

    Returns:
        統計情報の辞書
    """
    # (from_hex, to_hex, line_number) の組み合わせを記録
    line_usage: Set[Tuple[str, str, int]] = set()

    # 各八卦ペアでの変爻の使用回数
    hex_pair_usage: Dict[Tuple[str, str], Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    # 変爻なし（同じ卦）の回数
    no_change_count = 0

    # 各トランジション位置での統計
    transition_stats = {
        "transition_1": defaultdict(int),  # before → trigger
        "transition_2": defaultdict(int),  # trigger → action
        "transition_3": defaultdict(int),  # action → after
    }

    total_cases = 0
    cases_with_changing_lines = 0

    with open(db_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)
            case = Case(**data)
            total_cases += 1

            has_any = False

            # 3つのトランジションを分析
            transitions = [
                ("transition_1", case.before_hex.value, case.trigger_hex.value, case.changing_lines_1),
                ("transition_2", case.trigger_hex.value, case.action_hex.value, case.changing_lines_2),
                ("transition_3", case.action_hex.value, case.after_hex.value, case.changing_lines_3),
            ]

            for trans_name, from_hex, to_hex, changing_lines in transitions:
                if changing_lines is None:
                    continue

                has_any = True

                if len(changing_lines) == 0:
                    # 変爻なし（同じ卦）
                    no_change_count += 1
                    transition_stats[trans_name]["no_change"] += 1
                else:
                    # 変爻あり
                    for line_num in changing_lines:
                        line_usage.add((from_hex, to_hex, line_num))
                        hex_pair_usage[(from_hex, to_hex)][line_num] += 1
                        transition_stats[trans_name][f"line_{line_num}"] += 1

            if has_any:
                cases_with_changing_lines += 1

    return {
        "total_cases": total_cases,
        "cases_with_changing_lines": cases_with_changing_lines,
        "unique_line_combinations": len(line_usage),
        "line_usage": line_usage,
        "hex_pair_usage": hex_pair_usage,
        "no_change_count": no_change_count,
        "transition_stats": transition_stats,
    }

def print_analysis(stats: dict):
    """分析結果を見やすく表示"""
    print("=" * 80)
    print("384爻 網羅性分析")
    print("=" * 80)

    print(f"\n【基本統計】")
    print(f"  総事例数: {stats['total_cases']}")
    print(f"  変爻情報あり: {stats['cases_with_changing_lines']}")
    print(f"  変爻情報なし: {stats['total_cases'] - stats['cases_with_changing_lines']}")

    print(f"\n【変爻の使用状況】")
    print(f"  ユニークな(卦ペア, 爻番号)の組み合わせ: {stats['unique_line_combinations']}")
    print(f"  変爻なし（同じ卦）の回数: {stats['no_change_count']}")

    # トランジション別の統計
    print(f"\n【トランジション別の変爻使用】")
    for trans_name, trans_stats in stats['transition_stats'].items():
        trans_label = {
            "transition_1": "初期状態 → トリガー",
            "transition_2": "トリガー → 行動",
            "transition_3": "行動 → 結果"
        }[trans_name]

        print(f"\n  {trans_label}:")
        print(f"    変爻なし: {trans_stats.get('no_change', 0)} 回")
        for line_num in [1, 2, 3]:
            count = trans_stats.get(f'line_{line_num}', 0)
            line_name = {1: "初爻", 2: "二爻", 3: "三爻"}[line_num]
            print(f"    {line_name}が変化: {count} 回")

    # 最も使われている卦ペアと爻の組み合わせ
    print(f"\n【最も使われている卦ペア×爻の組み合わせ】")
    all_combinations = []
    for (from_hex, to_hex), lines_dict in stats['hex_pair_usage'].items():
        for line_num, count in lines_dict.items():
            all_combinations.append((count, from_hex, to_hex, line_num))

    all_combinations.sort(reverse=True)

    for i, (count, from_hex, to_hex, line_num) in enumerate(all_combinations[:15], 1):
        line_name = {1: "初爻", 2: "二爻", 3: "三爻"}[line_num]
        print(f"  {i:2d}. {from_hex} → {to_hex} の {line_name}: {count} 回")

    # 八卦ペアごとの変爻パターン
    print(f"\n【八卦ペアごとの変爻パターン】")

    # 全ての八卦ペアを集計
    hex_pairs = sorted(set(
        (from_hex, to_hex)
        for from_hex, to_hex, _ in stats['line_usage']
    ))

    print(f"  使用されている八卦ペア数: {len(hex_pairs)}")
    print(f"  最大可能な八卦ペア数: {8 * 8} (8卦 × 8卦)")
    print(f"  網羅率: {len(hex_pairs) / 64 * 100:.1f}%")

    # 各八卦がどの位置で何回使われているか
    print(f"\n【八卦別の使用状況】")
    hex_usage = defaultdict(lambda: {"from": 0, "to": 0, "lines": defaultdict(int)})

    for from_hex, to_hex, line_num in stats['line_usage']:
        hex_usage[from_hex]["from"] += 1
        hex_usage[to_hex]["to"] += 1
        hex_usage[from_hex]["lines"][line_num] += 1

    for hex_name in ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]:
        usage = hex_usage[hex_name]
        lines_used = sorted(usage["lines"].keys())
        lines_str = ", ".join([
            f"{line}爻({usage['lines'][line]}回)"
            for line in lines_used
        ]) if lines_used else "なし"

        print(f"\n  {hex_name}:")
        print(f"    変化元として: {usage['from']} 回")
        print(f"    変化先として: {usage['to']} 回")
        print(f"    使用された爻: {lines_str}")

def main():
    db_path = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"

    print("データベースを分析中...\n")

    stats = analyze_384_lines(db_path)
    print_analysis(stats)

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
