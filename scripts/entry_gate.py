#!/usr/bin/env python3
"""
入口ゲート（Entry Gate）

新規事例追加前に重複・類似事例をチェックし、
重複の場合は alias として登録することを促す
"""

import json
import sys
from pathlib import Path
from difflib import SequenceMatcher

def load_canonical_cases():
    """canonical 事例のみをロード"""
    cases_path = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"
    cases = []
    with open(cases_path, 'r', encoding='utf-8') as f:
        for line in f:
            case = json.loads(line.strip())
            if case.get('is_canonical', True):  # canonical のみ
                cases.append(case)
    return cases

def similarity(s1, s2):
    """文字列の類似度"""
    if not s1 or not s2:
        return 0.0
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

def check_exact_duplicate(new_case, existing_cases):
    """完全一致重複チェック"""
    new_key = (new_case.get('target_name', ''), new_case.get('period', ''))

    for case in existing_cases:
        existing_key = (case.get('target_name', ''), case.get('period', ''))
        if new_key == existing_key:
            return case
    return None

def check_near_duplicates(new_case, existing_cases, name_threshold=0.8, summary_threshold=0.7):
    """高類似事例チェック"""
    new_name = new_case.get('target_name', '')
    new_summary = new_case.get('story_summary', '')

    near_duplicates = []

    for case in existing_cases:
        name_sim = similarity(new_name, case.get('target_name', ''))
        summary_sim = similarity(new_summary, case.get('story_summary', ''))

        if name_sim >= name_threshold:
            near_duplicates.append({
                'case': case,
                'reason': 'name_similar',
                'name_similarity': name_sim,
                'summary_similarity': summary_sim
            })
        elif name_sim >= 0.6 and summary_sim >= summary_threshold:
            near_duplicates.append({
                'case': case,
                'reason': 'summary_similar',
                'name_similarity': name_sim,
                'summary_similarity': summary_sim
            })

    return sorted(near_duplicates, key=lambda x: -x['name_similarity'])[:5]

def check_same_entity(new_case, existing_cases):
    """同一企業・異フェーズチェック"""
    new_name = new_case.get('target_name', '')

    same_entity = []
    for case in existing_cases:
        if case.get('target_name') == new_name:
            same_entity.append(case)

    return same_entity

def gate_check(new_case_or_path):
    """
    入口ゲートチェック実行

    Args:
        new_case_or_path: 新規事例（dict）またはJSONファイルパス
    """
    # 入力処理
    if isinstance(new_case_or_path, str):
        with open(new_case_or_path, 'r', encoding='utf-8') as f:
            new_cases = json.load(f)
            if isinstance(new_cases, dict):
                new_cases = [new_cases]
    else:
        new_cases = [new_case_or_path] if isinstance(new_case_or_path, dict) else new_case_or_path

    existing = load_canonical_cases()

    print(f"\n{'='*60}")
    print(f"入口ゲートチェック")
    print(f"{'='*60}")
    print(f"既存canonical事例: {len(existing):,}件")
    print(f"チェック対象: {len(new_cases)}件")

    results = {
        'exact_duplicates': [],
        'near_duplicates': [],
        'same_entity': [],
        'new_canonical': []
    }

    for i, new_case in enumerate(new_cases):
        name = new_case.get('target_name', 'N/A')
        period = new_case.get('period', 'N/A')
        print(f"\n--- [{i+1}] {name} ({period}) ---")

        # 1. 完全一致チェック
        exact = check_exact_duplicate(new_case, existing)
        if exact:
            print(f"❌ 完全一致重複: {exact.get('transition_id')}")
            print(f"   → alias として追加するか、追加を中止してください")
            results['exact_duplicates'].append({
                'new': new_case,
                'existing': exact
            })
            continue

        # 2. 高類似チェック
        near = check_near_duplicates(new_case, existing)
        if near:
            print(f"⚠️ 類似事例あり:")
            for n in near[:3]:
                print(f"   - {n['case'].get('target_name')} ({n['case'].get('period')})")
                print(f"     名前類似度: {n['name_similarity']:.1%}, 内容類似度: {n['summary_similarity']:.1%}")
            results['near_duplicates'].append({
                'new': new_case,
                'similar': near
            })

        # 3. 同一企業チェック
        same = check_same_entity(new_case, existing)
        if same:
            print(f"ℹ️ 同一企業の既存事例: {len(same)}件")
            for s in same[:3]:
                print(f"   - {s.get('period')}: {s.get('story_summary', '')[:50]}...")
            results['same_entity'].append({
                'new': new_case,
                'existing': same
            })

        # 4. 新規OK
        if not exact and not near:
            print(f"✅ 新規canonical として追加可能")
            results['new_canonical'].append(new_case)

    # サマリー
    print(f"\n{'='*60}")
    print(f"ゲートチェック結果サマリー")
    print(f"{'='*60}")
    print(f"完全一致重複: {len(results['exact_duplicates'])}件 → 追加不可")
    print(f"高類似（要確認）: {len(results['near_duplicates'])}件")
    print(f"同一企業・異フェーズ: {len(results['same_entity'])}件 → 差別化確認推奨")
    print(f"新規canonical可: {len(results['new_canonical'])}件")

    return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 entry_gate.py <new_case.json>")
        print("\n新規事例のJSONファイルを指定してください")
        sys.exit(1)

    path = sys.argv[1]
    results = gate_check(path)

    # 全件新規OKならexit 0
    if results['exact_duplicates'] or results['near_duplicates']:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
