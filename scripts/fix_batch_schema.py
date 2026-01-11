#!/usr/bin/env python3
"""バッチファイルのスキーマを現行版に変換"""
import json
import sys

# 変換マッピング
BEFORE_STATE_MAP = {
    '成長・拡大': '絶頂・慢心',
    '安定・停止': '安定・平和',
    '混乱・衰退': '混乱・カオス',
}

TRIGGER_TYPE_MAP = {
    '自然推移': '意図的決断',
    '外部ショック': '外部ショック',
}

ACTION_TYPE_MAP = {
    '拡大・攻め': '攻める・挑戦',
    '耐える・潜伏': '耐える・潜伏',
    '刷新・破壊': '刷新・破壊',
    '分散・探索': '分散・スピンオフ',
}

AFTER_STATE_MAP = {
    '成長・拡大': '持続成長・大成功',
    '混乱・衰退': 'どん底・危機',
    '分岐・様子見': '変質・新生',
    '安定・停止': '安定・平和',
}

PATTERN_TYPE_MAP = {
    'Breakthrough': 'Steady_Growth',
    'Crisis_Pivot': 'Pivot_Success',
    'Exploration': 'Pivot_Success',
    'Managed_Decline': 'Slow_Decline',
    'Shock_Recovery': 'Shock_Recovery',
    'Hubris_Collapse': 'Hubris_Collapse',
    'Pivot_Success': 'Pivot_Success',
    'Endurance': 'Endurance',
    'Slow_Decline': 'Slow_Decline',
    'Steady_Growth': 'Steady_Growth',
}

SOURCE_TYPE_MAP = {
    'academic': 'article',
    'official': 'official',
    'news': 'news',
    'book': 'book',
    'blog': 'blog',
    'sns': 'sns',
    'article': 'article',
}

def fix_case(case):
    """1件の事例を修正"""
    if case.get('before_state') in BEFORE_STATE_MAP:
        case['before_state'] = BEFORE_STATE_MAP[case['before_state']]

    if case.get('trigger_type') in TRIGGER_TYPE_MAP:
        case['trigger_type'] = TRIGGER_TYPE_MAP[case['trigger_type']]

    if case.get('action_type') in ACTION_TYPE_MAP:
        case['action_type'] = ACTION_TYPE_MAP[case['action_type']]

    if case.get('after_state') in AFTER_STATE_MAP:
        case['after_state'] = AFTER_STATE_MAP[case['after_state']]

    if case.get('pattern_type') in PATTERN_TYPE_MAP:
        case['pattern_type'] = PATTERN_TYPE_MAP[case['pattern_type']]

    if case.get('source_type') in SOURCE_TYPE_MAP:
        case['source_type'] = SOURCE_TYPE_MAP[case['source_type']]

    # 不要なフィールドを削除
    for field in ['country', 'main_domain', 'sources', 'confidence_percent', 'evidence_notes', 'life_domain', 'tech_layer']:
        case.pop(field, None)

    return case

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 fix_batch_schema.py <batch_file.json>")
        sys.exit(1)

    filepath = sys.argv[1]

    with open(filepath, 'r', encoding='utf-8') as f:
        cases = json.load(f)

    fixed_cases = [fix_case(c) for c in cases]

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(fixed_cases, f, ensure_ascii=False, indent=2)

    print(f"Fixed {len(fixed_cases)} cases in {filepath}")

if __name__ == '__main__':
    main()
