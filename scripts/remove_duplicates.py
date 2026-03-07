#!/usr/bin/env python3
"""Remove duplicate records from cases.jsonl.

Strategy:
- Group records by (scale, target_name, before_state, after_state, period)
- In each group with >1 record, keep the record with transition_id (prefer CASE-* or other valid id)
- Remove records where transition_id is None if a valid-id version exists
- Create backup before modification
"""

import json
import shutil
import os
from collections import defaultdict
from datetime import datetime

DATA_PATH = "/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/data/raw/cases.jsonl"

def load_records(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)
    return records

def find_duplicates(records):
    """Find duplicate groups by (scale, target_name, before_state, after_state, period).
    Only consider groups where at least one record has transition_id=None
    and at least one has a valid transition_id."""

    groups = defaultdict(list)
    for idx, rec in enumerate(records):
        key = (
            rec.get('scale', ''),
            rec.get('target_name', ''),
            rec.get('before_state', ''),
            rec.get('after_state', ''),
            rec.get('period', ''),
        )
        groups[key].append(idx)

    # Filter: groups with >1 record where there's a None-tid and a valid-tid
    duplicate_groups = {}
    for key, indices in groups.items():
        if len(indices) < 2:
            continue

        none_indices = [i for i in indices if records[i].get('transition_id') is None]
        valid_indices = [i for i in indices if records[i].get('transition_id') is not None]

        if none_indices and valid_indices:
            duplicate_groups[key] = {
                'none': none_indices,
                'valid': valid_indices,
                'all': indices,
            }

    return duplicate_groups

def main():
    print("=" * 60)
    print("Duplicate Removal Script")
    print("=" * 60)

    # Load
    records = load_records(DATA_PATH)
    print(f"\nTotal records loaded: {len(records)}")

    # Safety check: target_name must not be empty for safe dedup
    empty_target = sum(1 for r in records if not r.get('target_name'))
    if empty_target > 0:
        print(f"\nERROR: {empty_target} records have empty target_name. Aborting.")
        print("Safe dedup requires target_name for all records.")
        return

    # Find duplicates
    dup_groups = find_duplicates(records)

    to_remove = set()
    for key, info in dup_groups.items():
        for idx in info['none']:
            to_remove.add(idx)

    print(f"\nDuplicate groups (None+valid tid): {len(dup_groups)}")
    print(f"Records to remove (None tid): {len(to_remove)}")

    # Show sample
    print(f"\n--- Sample groups (first 10) ---")
    for i, (key, info) in enumerate(list(dup_groups.items())[:10]):
        target_name = f"{key[0]}/{key[1]}" if key[1] else "(empty)"
        print(f"\nGroup {i+1}: {target_name}")
        print(f"  Period: {key[4]}")
        print(f"  None-tid records: {len(info['none'])} (indices: {info['none']})")
        print(f"  Valid-tid records: {len(info['valid'])} (indices: {info['valid'][:3]}...)")
        for idx in info['none'][:2]:
            r = records[idx]
            print(f"    None rec keys: {sorted(r.keys())}")
        for idx in info['valid'][:1]:
            r = records[idx]
            print(f"    Valid rec tid: {r.get('transition_id')}")

    if not to_remove:
        print("\nNo duplicates to remove.")
        return

    # Backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = DATA_PATH + f".backup_{timestamp}"
    shutil.copy2(DATA_PATH, backup_path)
    print(f"\nBackup created: {backup_path}")

    # Remove
    new_records = [rec for idx, rec in enumerate(records) if idx not in to_remove]
    print(f"\nRecords after removal: {len(new_records)}")
    print(f"Removed: {len(records) - len(new_records)}")

    # Write
    with open(DATA_PATH, 'w', encoding='utf-8') as f:
        for rec in new_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"\nFile written: {DATA_PATH}")

    # Write audit log
    log_path = os.path.join(os.path.dirname(DATA_PATH), '..', 'diagnostic', 'deleted_duplicates_log.txt')
    log_path = os.path.normpath(log_path)
    with open(log_path, 'a', encoding='utf-8') as logf:
        logf.write(f"\n=== 重複削除ログ ===\n")
        logf.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        logf.write(f"削除キー: scale + target_name + before_state + after_state + period\n")
        logf.write(f"削除前: {len(records)}件\n")
        logf.write(f"削除後: {len(new_records)}件\n")
        logf.write(f"削除数: {len(to_remove)}件\n")
        logf.write(f"削除グループ数: {len(dup_groups)}\n\n")
        logf.write("削除されたレコード:\n")
        for idx in sorted(to_remove):
            r = records[idx]
            logf.write(f"  scale={r.get('scale')}, target={r.get('target_name')}, "
                       f"tid={r.get('transition_id')}, cid={r.get('canonical_id')}, "
                       f"period={r.get('period')}, before={r.get('before_state')}, "
                       f"after={r.get('after_state')}\n")
    print(f"Audit log appended: {log_path}")

    # Verify
    verify = load_records(DATA_PATH)
    print(f"Verification: {len(verify)} records loaded successfully")

    # Check no remaining None+valid duplicates
    remaining_dups = find_duplicates(verify)
    print(f"Remaining None+valid duplicate groups: {len(remaining_dups)}")

if __name__ == '__main__':
    main()
