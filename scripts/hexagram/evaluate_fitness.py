#!/usr/bin/env python3
"""Phase 5: 適合度評価

460件のレビューサンプルについて、「この事例にこの卦は妥当か」を評価する。
各卦には固有の意味があり、trigger/actionの組み合わせと整合しているかを確認。
"""

import json
from collections import defaultdict
from pathlib import Path

# 八卦とtriggerの対応（下卦がtriggerを表す）
TRIGRAM_TO_TRIGGER = {
    '乾': 'T_GROWTH_MOMENTUM',  # 天: 積極的成長
    '坤': 'T_STAGNATION',       # 地: 停滞・受容
    '震': 'T_EXTERNAL_FORCE',   # 雷: 外的衝撃
    '巽': 'T_ENVIRONMENT',      # 風: 環境変化
    '坎': 'T_INTERNAL_CRISIS',  # 水: 内部危機
    '離': 'T_OPPORTUNITY',      # 火: 機会・可視化
    '艮': 'T_LEADERSHIP',       # 山: リーダーシップ変化
    '兌': 'T_INTERACTION'       # 沢: 交流・対話
}

# 八卦とactionの対応（上卦がactionを表す）
TRIGRAM_TO_ACTION = {
    '乾': 'A_EXPAND',     # 天: 拡大
    '坤': 'A_MAINTAIN',   # 地: 維持
    '震': 'A_TRANSFORM',  # 雷: 変革
    '巽': 'A_ADAPT',      # 風: 適応
    '坎': 'A_PAUSE',      # 水: 一時停止・慎重
    '離': 'A_FOCUS',      # 火: 集中・明確化
    '艮': 'A_RETREAT',    # 山: 撤退・縮小
    '兌': 'A_CONNECT'     # 沢: 連携・協力
}

# 六十四卦名（参考用）
HEXAGRAM_NAMES = {
    (1, '乾', '乾'): '乾為天',
    (2, '坤', '坤'): '坤為地',
    # 他は必要に応じて追加
}


def evaluate_fitness(input_path: str, verbose: bool = False):
    """適合度を評価する"""

    with open(input_path, 'r', encoding='utf-8') as f:
        cases = [json.loads(line) for line in f if line.strip()]

    results = {
        'total': len(cases),
        'trigger_match': 0,
        'action_match': 0,
        'both_match': 0,
        'neither_match': 0,
        'mismatches': [],
        'by_hexagram': defaultdict(lambda: {'total': 0, 'match': 0, 'trigger_match': 0, 'action_match': 0}),
        'by_trigram_pair': defaultdict(lambda: {'total': 0, 'trigger_match': 0, 'action_match': 0})
    }

    for case in cases:
        lower = case.get('lower_trigram', '')
        upper = case.get('upper_trigram', '')
        trigger = case.get('trigger', '')
        action = case.get('action', '')
        hexagram_name = case.get('hexagram_name', '不明')
        hexagram_number = case.get('hexagram_number', 0)
        entity_name = case.get('target_name', case.get('entity_name', '不明'))

        expected_trigger = TRIGRAM_TO_TRIGGER.get(lower, '')
        expected_action = TRIGRAM_TO_ACTION.get(upper, '')

        trigger_ok = (trigger == expected_trigger)
        action_ok = (action == expected_action)

        # カウント
        if trigger_ok:
            results['trigger_match'] += 1
        if action_ok:
            results['action_match'] += 1
        if trigger_ok and action_ok:
            results['both_match'] += 1
        if not trigger_ok and not action_ok:
            results['neither_match'] += 1

        # 卦別統計
        hex_key = f"{hexagram_number}_{hexagram_name}"
        results['by_hexagram'][hex_key]['total'] += 1
        if trigger_ok:
            results['by_hexagram'][hex_key]['trigger_match'] += 1
        if action_ok:
            results['by_hexagram'][hex_key]['action_match'] += 1
        if trigger_ok and action_ok:
            results['by_hexagram'][hex_key]['match'] += 1

        # 八卦ペア別統計
        pair_key = f"{lower}→{upper}"
        results['by_trigram_pair'][pair_key]['total'] += 1
        if trigger_ok:
            results['by_trigram_pair'][pair_key]['trigger_match'] += 1
        if action_ok:
            results['by_trigram_pair'][pair_key]['action_match'] += 1

        # 不一致の記録
        if not (trigger_ok and action_ok):
            mismatch_info = {
                'entity': entity_name,
                'hexagram_number': hexagram_number,
                'hexagram_name': hexagram_name,
                'lower_trigram': lower,
                'upper_trigram': upper,
                'trigger': trigger,
                'expected_trigger': expected_trigger,
                'trigger_match': trigger_ok,
                'action': action,
                'expected_action': expected_action,
                'action_match': action_ok
            }
            results['mismatches'].append(mismatch_info)

    return results


def print_results(results: dict):
    """結果を表示する"""

    total = results['total']

    print("=" * 60)
    print("Phase 5: 適合度評価結果")
    print("=" * 60)

    print(f"\n【総合統計】")
    print(f"  総サンプル数: {total}")
    print(f"  trigger一致: {results['trigger_match']} ({results['trigger_match']/total*100:.1f}%)")
    print(f"  action一致:  {results['action_match']} ({results['action_match']/total*100:.1f}%)")
    print(f"  両方一致:    {results['both_match']} ({results['both_match']/total*100:.1f}%)")
    print(f"  両方不一致:  {results['neither_match']} ({results['neither_match']/total*100:.1f}%)")

    # 卦別の適合度ランキング（件数の多い順）
    print(f"\n【卦別適合度（件数上位10）】")
    hex_stats = sorted(results['by_hexagram'].items(),
                      key=lambda x: x[1]['total'], reverse=True)[:10]

    print(f"  {'卦名':<20} {'件数':>5} {'両方一致':>8} {'trigger':>8} {'action':>8}")
    print("-" * 55)
    for hex_key, stats in hex_stats:
        total_h = stats['total']
        match_rate = stats['match'] / total_h * 100 if total_h > 0 else 0
        trigger_rate = stats['trigger_match'] / total_h * 100 if total_h > 0 else 0
        action_rate = stats['action_match'] / total_h * 100 if total_h > 0 else 0
        print(f"  {hex_key:<20} {total_h:>5} {match_rate:>7.1f}% {trigger_rate:>7.1f}% {action_rate:>7.1f}%")

    # 適合度の低い卦
    print(f"\n【適合度の低い卦（両方一致0%のもの）】")
    low_fitness = [(k, v) for k, v in results['by_hexagram'].items()
                   if v['match'] == 0 and v['total'] >= 3]
    for hex_key, stats in sorted(low_fitness, key=lambda x: x[1]['total'], reverse=True)[:10]:
        total_h = stats['total']
        trigger_rate = stats['trigger_match'] / total_h * 100 if total_h > 0 else 0
        action_rate = stats['action_match'] / total_h * 100 if total_h > 0 else 0
        print(f"  {hex_key}: {total_h}件 (trigger: {trigger_rate:.0f}%, action: {action_rate:.0f}%)")

    # 八卦ペア別統計
    print(f"\n【八卦ペア別統計（上位10）】")
    pair_stats = sorted(results['by_trigram_pair'].items(),
                       key=lambda x: x[1]['total'], reverse=True)[:10]
    print(f"  {'ペア(下→上)':<12} {'件数':>5} {'trigger一致':>10} {'action一致':>10}")
    print("-" * 45)
    for pair, stats in pair_stats:
        total_p = stats['total']
        trigger_rate = stats['trigger_match'] / total_p * 100 if total_p > 0 else 0
        action_rate = stats['action_match'] / total_p * 100 if total_p > 0 else 0
        print(f"  {pair:<12} {total_p:>5} {trigger_rate:>9.1f}% {action_rate:>9.1f}%")

    # 不一致パターン分析
    print(f"\n【不一致パターン分析】")
    mismatch_patterns = defaultdict(int)
    for m in results['mismatches']:
        pattern = f"{m['trigger']}→{m['expected_trigger']}" if not m['trigger_match'] else "OK"
        pattern += " | "
        pattern += f"{m['action']}→{m['expected_action']}" if not m['action_match'] else "OK"
        mismatch_patterns[pattern] += 1

    top_patterns = sorted(mismatch_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
    print("  頻出不一致パターン:")
    for pattern, count in top_patterns:
        print(f"    {pattern}: {count}件")

    # サンプル不一致事例
    print(f"\n【不一致事例サンプル（最初の5件）】")
    for m in results['mismatches'][:5]:
        print(f"\n  [{m['hexagram_number']}] {m['hexagram_name']} - {m['entity']}")
        print(f"    下卦: {m['lower_trigram']} | 上卦: {m['upper_trigram']}")
        if not m['trigger_match']:
            print(f"    trigger: {m['trigger']} (期待: {m['expected_trigger']})")
        if not m['action_match']:
            print(f"    action:  {m['action']} (期待: {m['expected_action']})")


def save_results(results: dict, output_path: str):
    """結果をJSONで保存"""

    # defaultdictを通常のdictに変換
    output = {
        'summary': {
            'total': results['total'],
            'trigger_match': results['trigger_match'],
            'trigger_match_rate': results['trigger_match'] / results['total'] * 100,
            'action_match': results['action_match'],
            'action_match_rate': results['action_match'] / results['total'] * 100,
            'both_match': results['both_match'],
            'both_match_rate': results['both_match'] / results['total'] * 100,
            'neither_match': results['neither_match'],
            'neither_match_rate': results['neither_match'] / results['total'] * 100
        },
        'by_hexagram': dict(results['by_hexagram']),
        'by_trigram_pair': dict(results['by_trigram_pair']),
        'mismatches': results['mismatches']
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n結果を保存: {output_path}")


if __name__ == '__main__':
    base_path = Path(__file__).parent.parent.parent
    input_path = base_path / 'data/hexagram/review_samples.jsonl'
    output_path = base_path / 'data/hexagram/fitness_evaluation_results.json'

    print(f"入力ファイル: {input_path}")

    results = evaluate_fitness(str(input_path))
    print_results(results)
    save_results(results, str(output_path))
