#!/usr/bin/env python3
"""
重複事例のcanonical化スクリプト

Phase 1: 完全一致重複の自動クラスタ化
- target_name + period が完全一致するグループを検出
- 各グループで canonical を選定
- canonical_id を付与
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import uuid

def load_cases():
    """DBから全事例をロード"""
    cases_path = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"
    cases = []
    with open(cases_path, 'r', encoding='utf-8') as f:
        for line in f:
            cases.append(json.loads(line.strip()))
    return cases

def save_cases(cases, output_path):
    """事例を保存"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')

def get_credibility_score(rank):
    """credibility_rank をスコアに変換"""
    scores = {'S': 4, 'A': 3, 'B': 2, 'C': 1}
    return scores.get(rank, 0)

def select_canonical(group):
    """
    グループ内で canonical を選定

    優先順位:
    1. credibility_rank が最高
    2. source_url が存在
    3. logic_memo が最も詳細
    4. transition_id が最小（最古）
    """
    def score(case):
        cred = get_credibility_score(case.get('credibility_rank', 'C'))
        has_url = 1 if case.get('source_url') or case.get('sources') else 0
        memo_len = len(case.get('logic_memo', '') or '')
        # transition_id から数値部分を抽出（小さいほど古い）
        tid = case.get('transition_id', 'ZZZZ_999999')
        try:
            tid_num = int(''.join(filter(str.isdigit, tid)) or '999999')
        except:
            tid_num = 999999
        return (cred, has_url, memo_len, -tid_num)

    return max(group, key=score)

def find_duplicate_groups(cases):
    """完全一致重複グループを検出"""
    groups = defaultdict(list)

    for i, case in enumerate(cases):
        key = (case.get('target_name', ''), case.get('period', ''))
        groups[key].append((i, case))

    # 2件以上のグループのみ抽出
    duplicate_groups = {k: v for k, v in groups.items() if len(v) > 1}
    return duplicate_groups

def generate_canonical_id():
    """ユニークな canonical_id を生成"""
    return f"CAN_{uuid.uuid4().hex[:8].upper()}"

def canonicalize(cases, dry_run=False):
    """
    重複事例をcanonical化

    Args:
        cases: 全事例リスト
        dry_run: True の場合、変更を適用せずレポートのみ
    """
    print(f"\n{'='*60}")
    print(f"Canonical化処理")
    print(f"{'='*60}")

    duplicate_groups = find_duplicate_groups(cases)

    print(f"\n総事例数: {len(cases):,}件")
    print(f"重複グループ数: {len(duplicate_groups):,}グループ")

    total_duplicates = sum(len(g) - 1 for g in duplicate_groups.values())
    print(f"重複事例数（削減可能）: {total_duplicates:,}件")

    if dry_run:
        print(f"\n[DRY RUN] 変更は適用されません")
        print(f"\n--- サンプルグループ ---")
        for i, ((name, period), group) in enumerate(list(duplicate_groups.items())[:5]):
            print(f"\nグループ {i+1}: {name} ({period}) - {len(group)}件")
            canonical = select_canonical([c for _, c in group])
            print(f"  → Canonical候補: {canonical.get('transition_id')}")
            print(f"     credibility: {canonical.get('credibility_rank')}")
            print(f"     source_url: {'あり' if canonical.get('source_url') else 'なし'}")
        return cases

    # canonical_id を付与
    canonical_count = 0
    alias_count = 0

    # 全事例に canonical_id を初期化（単独事例は自身が canonical）
    processed_indices = set()

    for (name, period), group in duplicate_groups.items():
        # canonical 選定
        canonical_case = select_canonical([c for _, c in group])
        canonical_id = generate_canonical_id()

        alias_ids = []

        for idx, case in group:
            processed_indices.add(idx)

            if case.get('transition_id') == canonical_case.get('transition_id'):
                # canonical
                cases[idx]['canonical_id'] = canonical_id
                cases[idx]['is_canonical'] = True
                canonical_count += 1
            else:
                # alias
                cases[idx]['canonical_id'] = canonical_id
                cases[idx]['is_canonical'] = False
                alias_ids.append(case.get('transition_id'))
                alias_count += 1

        # canonical に alias_ids を追加
        for idx, case in group:
            if cases[idx].get('is_canonical'):
                cases[idx]['alias_ids'] = alias_ids

    # 単独事例（重複なし）にも canonical_id を付与
    single_count = 0
    for i, case in enumerate(cases):
        if i not in processed_indices:
            cases[i]['canonical_id'] = generate_canonical_id()
            cases[i]['is_canonical'] = True
            cases[i]['alias_ids'] = []
            single_count += 1

    print(f"\n--- 処理結果 ---")
    print(f"Canonical事例: {canonical_count + single_count:,}件")
    print(f"  - 重複グループの代表: {canonical_count:,}件")
    print(f"  - 単独事例: {single_count:,}件")
    print(f"Alias事例: {alias_count:,}件")

    return cases

def main():
    dry_run = '--dry-run' in sys.argv or '-n' in sys.argv

    cases = load_cases()

    updated_cases = canonicalize(cases, dry_run=dry_run)

    if not dry_run:
        # バックアップ
        backup_path = Path(__file__).parent.parent / "data" / "raw" / f"cases_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        original_path = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"

        print(f"\n[バックアップ] {backup_path}")
        save_cases(load_cases(), backup_path)

        print(f"[保存] {original_path}")
        save_cases(updated_cases, original_path)

        print(f"\n✅ Canonical化完了")
    else:
        print(f"\n[DRY RUN完了] 実行するには --dry-run を外してください")

if __name__ == "__main__":
    main()
