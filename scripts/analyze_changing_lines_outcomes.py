#!/usr/bin/env python3
"""
変爻パターンと結果の関係性分析

変爻（changing lines）のパターンと outcome の関係を分析し、
診断アルゴリズムの強化に活用できる知見を抽出します。
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
from schema_v3 import Case

def analyze_changing_lines_by_outcome(db_path: Path):
    """変爻パターンと結果の相関を分析"""

    # データ構造
    outcome_stats = defaultdict(lambda: {
        'count': 0,
        'changing_lines_1': defaultdict(int),  # before→trigger
        'changing_lines_2': defaultdict(int),  # trigger→action
        'changing_lines_3': defaultdict(int),  # action→after
        'total_changes': [],  # 総変爻数
        'hex_pairs_1': defaultdict(int),  # 卦ペア
        'hex_pairs_2': defaultdict(int),
        'hex_pairs_3': defaultdict(int),
    })

    # データ読み込み
    with open(db_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            case = Case(**data)

            outcome = case.outcome.value
            outcome_stats[outcome]['count'] += 1

            # 各トランジションの変爻を記録
            if case.changing_lines_1 is not None:
                for line_num in case.changing_lines_1:
                    outcome_stats[outcome]['changing_lines_1'][line_num] += 1

                # 卦ペアを記録
                pair = f"{case.before_hex.value}→{case.trigger_hex.value}"
                outcome_stats[outcome]['hex_pairs_1'][pair] += 1

            if case.changing_lines_2 is not None:
                for line_num in case.changing_lines_2:
                    outcome_stats[outcome]['changing_lines_2'][line_num] += 1

                pair = f"{case.trigger_hex.value}→{case.action_hex.value}"
                outcome_stats[outcome]['hex_pairs_2'][pair] += 1

            if case.changing_lines_3 is not None:
                for line_num in case.changing_lines_3:
                    outcome_stats[outcome]['changing_lines_3'][line_num] += 1

                pair = f"{case.action_hex.value}→{case.after_hex.value}"
                outcome_stats[outcome]['hex_pairs_3'][pair] += 1

            # 総変爻数を計算
            total = 0
            if case.changing_lines_1:
                total += len(case.changing_lines_1)
            if case.changing_lines_2:
                total += len(case.changing_lines_2)
            if case.changing_lines_3:
                total += len(case.changing_lines_3)
            outcome_stats[outcome]['total_changes'].append(total)

    return outcome_stats

def print_analysis_report(stats: Dict):
    """分析結果をレポート形式で出力"""

    print("=" * 80)
    print("変爻パターンと結果の相関分析レポート")
    print("=" * 80)

    outcomes = ['Success', 'PartialSuccess', 'Mixed', 'Failure']

    for outcome in outcomes:
        if outcome not in stats:
            continue

        data = stats[outcome]
        count = data['count']

        print(f"\n{'=' * 80}")
        print(f"【{outcome}】 - {count}件")
        print(f"{'=' * 80}")

        # 総変爻数の統計
        if data['total_changes']:
            total_changes = data['total_changes']
            avg_changes = sum(total_changes) / len(total_changes)
            print(f"\n■ 平均変爻数: {avg_changes:.2f}爻/事例")
            print(f"   最小: {min(total_changes)}爻, 最大: {max(total_changes)}爻")

        # トランジション1: before→trigger
        print(f"\n■ トランジション1（初期状態→トリガー）の変爻:")
        if data['changing_lines_1']:
            total_1 = sum(data['changing_lines_1'].values())
            for line_num in [1, 2, 3]:
                count_line = data['changing_lines_1'].get(line_num, 0)
                pct = count_line / total_1 * 100 if total_1 > 0 else 0
                line_name = {1: '初爻', 2: '二爻', 3: '三爻'}[line_num]
                print(f"   {line_name}: {count_line}回 ({pct:.1f}%)")

        # トランジション2: trigger→action
        print(f"\n■ トランジション2（トリガー→行動）の変爻:")
        if data['changing_lines_2']:
            total_2 = sum(data['changing_lines_2'].values())
            for line_num in [1, 2, 3]:
                count_line = data['changing_lines_2'].get(line_num, 0)
                pct = count_line / total_2 * 100 if total_2 > 0 else 0
                line_name = {1: '初爻', 2: '二爻', 3: '三爻'}[line_num]
                print(f"   {line_name}: {count_line}回 ({pct:.1f}%)")

        # トランジション3: action→after
        print(f"\n■ トランジション3（行動→結果）の変爻:")
        if data['changing_lines_3']:
            total_3 = sum(data['changing_lines_3'].values())
            for line_num in [1, 2, 3]:
                count_line = data['changing_lines_3'].get(line_num, 0)
                pct = count_line / total_3 * 100 if total_3 > 0 else 0
                line_name = {1: '初爻', 2: '二爻', 3: '三爻'}[line_num]
                print(f"   {line_name}: {count_line}回 ({pct:.1f}%)")

        # 最頻出の卦ペア（各トランジションで上位5位）
        print(f"\n■ 最頻出の卦ペア:")

        print(f"   トランジション1（初期→トリガー）:")
        top_pairs_1 = sorted(data['hex_pairs_1'].items(), key=lambda x: x[1], reverse=True)[:5]
        for pair, cnt in top_pairs_1:
            print(f"      {pair}: {cnt}回")

        print(f"   トランジション2（トリガー→行動）:")
        top_pairs_2 = sorted(data['hex_pairs_2'].items(), key=lambda x: x[1], reverse=True)[:5]
        for pair, cnt in top_pairs_2:
            print(f"      {pair}: {cnt}回")

        print(f"   トランジション3（行動→結果）:")
        top_pairs_3 = sorted(data['hex_pairs_3'].items(), key=lambda x: x[1], reverse=True)[:5]
        for pair, cnt in top_pairs_3:
            print(f"      {pair}: {cnt}回")

def find_success_patterns(stats: Dict) -> Dict:
    """成功パターンの特徴を抽出"""

    print("\n" + "=" * 80)
    print("【成功パターンの特徴抽出】")
    print("=" * 80)

    success_data = stats.get('Success', {})
    failure_data = stats.get('Failure', {})

    if not success_data or not failure_data:
        print("Success または Failure のデータが不足しています")
        return {}

    # 成功と失敗で差が大きい卦ペアを見つける
    print("\n■ 成功に特徴的な卦ペア（失敗との差が大きい）:")

    for transition_name, success_pairs, failure_pairs in [
        ("トランジション1", success_data['hex_pairs_1'], failure_data['hex_pairs_1']),
        ("トランジション2", success_data['hex_pairs_2'], failure_data['hex_pairs_2']),
        ("トランジション3", success_data['hex_pairs_3'], failure_data['hex_pairs_3']),
    ]:
        print(f"\n  {transition_name}:")

        # 正規化して比較
        success_total = sum(success_pairs.values()) or 1
        failure_total = sum(failure_pairs.values()) or 1

        differences = {}
        for pair in set(list(success_pairs.keys()) + list(failure_pairs.keys())):
            success_rate = success_pairs.get(pair, 0) / success_total
            failure_rate = failure_pairs.get(pair, 0) / failure_total
            diff = success_rate - failure_rate

            if success_pairs.get(pair, 0) >= 3:  # 最低3件以上ある場合のみ
                differences[pair] = diff

        # 差が大きい順に表示
        top_diffs = sorted(differences.items(), key=lambda x: x[1], reverse=True)[:5]
        for pair, diff in top_diffs:
            success_cnt = success_pairs.get(pair, 0)
            failure_cnt = failure_pairs.get(pair, 0)
            print(f"    {pair}: 成功{success_cnt}回 vs 失敗{failure_cnt}回 (差: {diff:+.3f})")

    return {}

def main():
    db_path = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"

    print("データベースを分析中...\n")

    # 分析実行
    stats = analyze_changing_lines_by_outcome(db_path)

    # レポート出力
    print_analysis_report(stats)

    # 成功パターンの抽出
    find_success_patterns(stats)

    print("\n" + "=" * 80)
    print("分析完了")
    print("=" * 80)

if __name__ == "__main__":
    main()
