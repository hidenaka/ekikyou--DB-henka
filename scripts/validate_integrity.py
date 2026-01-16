#!/usr/bin/env python3
"""
整合性検証スクリプト

参照整合性、ユニーク制約、NOT NULL制約、Enum値制約を検証
"""

import json
from pathlib import Path
from collections import Counter

# Enum定義
VALID_ENTITY_TYPES = ['company', 'individual', 'government', 'organization']
VALID_EVENT_DRIVER_TYPES = ['internal', 'market', 'policy', 'disaster', 'pandemic', 'technology', 'competition']
VALID_EVENT_PHASES = ['announcement', 'execution', 'completion', 'outcome', None]
VALID_ANNOTATION_STATUS = ['single', 'double', 'verified']

REQUIRED_FIELDS = [
    'target_name', 'scale', 'period', 'story_summary',
    'before_state', 'trigger_type', 'action_type', 'after_state',
    'before_hex', 'trigger_hex', 'action_hex', 'after_hex',
    'pattern_type', 'outcome', 'source_type', 'credibility_rank'
]

def validate_integrity():
    """整合性を検証"""
    cases_path = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"
    entity_table_path = Path(__file__).parent.parent / "data" / "master" / "entity_table.json"
    alias_table_path = Path(__file__).parent.parent / "data" / "master" / "alias_table.json"

    # データ読み込み
    cases = []
    with open(cases_path, 'r', encoding='utf-8') as f:
        for line in f:
            cases.append(json.loads(line.strip()))

    entity_table = {}
    if entity_table_path.exists():
        with open(entity_table_path, 'r', encoding='utf-8') as f:
            entity_table = json.load(f)

    alias_table = {}
    if alias_table_path.exists():
        with open(alias_table_path, 'r', encoding='utf-8') as f:
            alias_table = json.load(f)

    print(f"\n{'='*60}")
    print(f"整合性検証")
    print(f"{'='*60}")
    print(f"事例数: {len(cases):,}件")
    print(f"エンティティ数: {len(entity_table):,}件")
    print(f"alias数: {len(alias_table):,}件")

    violations = []

    # 1. 参照整合性チェック
    print(f"\n--- 参照整合性チェック ---")
    entity_ids = set(entity_table.keys())
    ref_violations = 0
    for i, case in enumerate(cases):
        psid = case.get('primary_subject_id')
        if psid and psid not in entity_ids:
            ref_violations += 1
            if ref_violations <= 5:
                violations.append({
                    'type': 'ref_integrity',
                    'case_index': i,
                    'field': 'primary_subject_id',
                    'value': psid,
                    'message': 'entity_tableに存在しない'
                })
    print(f"  違反: {ref_violations}件")

    # 2. ユニーク制約チェック
    print(f"\n--- ユニーク制約チェック ---")
    # (event_id, event_phase, primary_subject_id)
    composite_keys = Counter()
    for case in cases:
        key = (case.get('event_id'), case.get('event_phase'), case.get('primary_subject_id'))
        composite_keys[key] += 1

    duplicate_keys = {k: v for k, v in composite_keys.items() if v > 1}
    unique_violations = sum(v - 1 for v in duplicate_keys.values())
    print(f"  複合キー重複: {len(duplicate_keys)}組 ({unique_violations}件)")
    if duplicate_keys:
        for k, v in list(duplicate_keys.items())[:3]:
            print(f"    {k}: {v}件")

    # 3. NOT NULL制約チェック
    print(f"\n--- NOT NULL制約チェック ---")
    null_violations = Counter()
    for case in cases:
        for field in REQUIRED_FIELDS:
            if not case.get(field):
                null_violations[field] += 1

    total_null = sum(null_violations.values())
    print(f"  NULL違反: {total_null}件")
    if null_violations:
        for field, count in null_violations.most_common(5):
            print(f"    {field}: {count}件")

    # 4. Enum値チェック
    print(f"\n--- Enum値チェック ---")
    enum_violations = Counter()

    for case in cases:
        et = case.get('entity_type')
        if et and et not in VALID_ENTITY_TYPES:
            enum_violations['entity_type'] += 1

        edt = case.get('event_driver_type')
        if edt and edt not in VALID_EVENT_DRIVER_TYPES:
            enum_violations['event_driver_type'] += 1

        ep = case.get('event_phase')
        if ep and ep not in VALID_EVENT_PHASES:
            enum_violations['event_phase'] += 1

        ans = case.get('annotation_status')
        if ans and ans not in VALID_ANNOTATION_STATUS:
            enum_violations['annotation_status'] += 1

    total_enum = sum(enum_violations.values())
    print(f"  Enum違反: {total_enum}件")
    if enum_violations:
        for field, count in enum_violations.most_common():
            print(f"    {field}: {count}件")

    # 5. 組み合わせ制約チェック
    print(f"\n--- 組み合わせ制約チェック ---")
    combo_violations = 0
    entity_scale_map = {
        'company': ['company', 'other'],
        'individual': ['individual', 'family'],
        'government': ['country'],
        'organization': ['other'],
    }

    for case in cases:
        et = case.get('entity_type')
        scale = case.get('scale')
        if et and scale:
            allowed_scales = entity_scale_map.get(et, [])
            if scale not in allowed_scales:
                combo_violations += 1

    print(f"  組み合わせ違反: {combo_violations}件")

    # サマリー
    print(f"\n{'='*60}")
    print(f"検証結果サマリー")
    print(f"{'='*60}")
    total_violations = ref_violations + unique_violations + total_null + total_enum + combo_violations
    print(f"総違反数: {total_violations}件")
    print(f"  参照整合性: {ref_violations}件")
    print(f"  ユニーク制約: {unique_violations}件")
    print(f"  NOT NULL: {total_null}件")
    print(f"  Enum値: {total_enum}件")
    print(f"  組み合わせ: {combo_violations}件")

    if total_violations == 0:
        print(f"\n✅ 全ての整合性チェックをパス")
    else:
        print(f"\n⚠ 整合性違反が検出されました")

    # 結果保存
    result = {
        'total_cases': len(cases),
        'total_violations': total_violations,
        'violations': {
            'ref_integrity': ref_violations,
            'unique': unique_violations,
            'not_null': total_null,
            'enum': total_enum,
            'combination': combo_violations
        },
        'details': violations[:10]
    }

    result_path = Path(__file__).parent.parent / "data" / "quality" / "integrity_check.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n[結果保存] {result_path}")

    return result

if __name__ == "__main__":
    validate_integrity()
