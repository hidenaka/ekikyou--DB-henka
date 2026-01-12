#!/usr/bin/env python3
"""
trigger_hex再割り当てスクリプト

論理的矛盾のある割り当てを修正する。
事実は変えず、解釈の精度を上げる。

ルール:
1. 意図的決断 + 震 → 乾（決断）または 離（明確化）
2. Endurance + 震 + 非ショック系trigger → 艮（忍耐）
3. Steady_Growth + 震 + 意図的/偶発trigger → 巽（浸透）
4. 偶発・出会い + 震 + 非ショックpattern → 兌（出会い）
"""

import json
import random
from pathlib import Path
from collections import Counter

CASES_FILE = Path("data/raw/cases.jsonl")
BACKUP_FILE = Path("data/archive/cases_backup_before_reassign.jsonl")


def load_cases():
    cases = []
    with open(CASES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def save_cases(cases):
    with open(CASES_FILE, 'w', encoding='utf-8') as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')


def backup_cases(cases):
    with open(BACKUP_FILE, 'w', encoding='utf-8') as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')


def should_reassign_rule1(case):
    """ルール1: 意図的決断 + 震 → 乾 or 離"""
    return (case.get('trigger_type') == '意図的決断' and
            case.get('trigger_hex') == '震')


def should_reassign_rule2(case):
    """ルール2: Endurance + 震 + 非ショック系 → 艮"""
    return (case.get('pattern_type') == 'Endurance' and
            case.get('trigger_hex') == '震' and
            case.get('trigger_type') not in ['外部ショック', '内部崩壊'])


def should_reassign_rule3(case):
    """ルール3: Steady_Growth + 震 + 意図的/偶発 → 巽"""
    return (case.get('pattern_type') == 'Steady_Growth' and
            case.get('trigger_hex') == '震' and
            case.get('trigger_type') in ['意図的決断', '偶発・出会い'])


def should_reassign_rule4(case):
    """ルール4: 偶発・出会い + 震 + 非ショックpattern → 兌"""
    shock_patterns = ['Shock_Recovery', 'Crisis_Pivot', 'Hubris_Collapse']
    return (case.get('trigger_type') == '偶発・出会い' and
            case.get('trigger_hex') == '震' and
            case.get('pattern_type') not in shock_patterns)


def reassign(case):
    """ルールに基づいて再割り当て"""
    original = case.get('trigger_hex')
    new_hex = original
    rule_applied = None

    if should_reassign_rule1(case):
        # 意図的決断 → 乾(決断/リーダーシップ) or 離(明確化)
        outcome = case.get('outcome', '')
        if outcome == 'Success':
            new_hex = '乾'  # 成功した決断 = 乾
        else:
            new_hex = '離'  # 明確化・可視化
        rule_applied = 1

    elif should_reassign_rule2(case):
        # Endurance + 非ショック → 艮(忍耐) or 坤(受容)
        new_hex = '艮'  # 忍耐・停止
        rule_applied = 2

    elif should_reassign_rule3(case):
        # Steady_Growth + 意図的/偶発 → 巽(浸透)
        new_hex = '巽'  # 緩やかな浸透
        rule_applied = 3

    elif should_reassign_rule4(case):
        # 偶発・出会い + 非ショック → 兌(出会い)
        new_hex = '兌'  # 喜び・出会い
        rule_applied = 4

    return new_hex, rule_applied


def main(dry_run=True):
    cases = load_cases()

    print(f"=== trigger_hex再割り当て ===")
    print(f"総事例数: {len(cases)}")
    print(f"ドライラン: {'はい' if dry_run else 'いいえ'}")

    # バックアップ
    if not dry_run:
        backup_cases(cases)
        print(f"バックアップ: {BACKUP_FILE}")

    # 統計
    stats = {
        'rule1': 0,  # 意図的決断
        'rule2': 0,  # Endurance
        'rule3': 0,  # Steady_Growth
        'rule4': 0,  # 偶発・出会い
    }

    changes = []

    for case in cases:
        original = case.get('trigger_hex')
        new_hex, rule = reassign(case)

        if new_hex != original:
            stats[f'rule{rule}'] += 1
            changes.append({
                'target': case.get('target_name'),
                'original': original,
                'new': new_hex,
                'rule': rule
            })

            if not dry_run:
                case['trigger_hex'] = new_hex
                # 再割り当ての記録
                case['trigger_hex_reassigned'] = True
                case['trigger_hex_original'] = original
                case['trigger_hex_rule'] = rule

    # 結果表示
    print(f"\n=== 再割り当て結果 ===")
    print(f"ルール1 (意図的決断→乾/離): {stats['rule1']}件")
    print(f"ルール2 (Endurance→艮): {stats['rule2']}件")
    print(f"ルール3 (Steady_Growth→巽): {stats['rule3']}件")
    print(f"ルール4 (偶発・出会い→兌): {stats['rule4']}件")
    print(f"合計: {sum(stats.values())}件")

    # サンプル表示
    print(f"\n=== サンプル ===")
    for change in changes[:10]:
        print(f"  {change['target']}: {change['original']} → {change['new']} (ルール{change['rule']})")

    if not dry_run:
        save_cases(cases)
        print(f"\n保存完了: {CASES_FILE}")

    return stats


if __name__ == '__main__':
    import sys
    dry_run = '--execute' not in sys.argv
    main(dry_run=dry_run)
