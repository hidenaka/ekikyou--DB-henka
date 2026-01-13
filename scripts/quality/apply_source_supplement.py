"""
ソース補完適用スクリプト
rejected URL事例に実ソースを追加
"""

import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'

def load_supplements():
    """補完データ読み込み"""
    supplements = {}
    import_dir = DATA_DIR / 'import'

    for path in import_dir.glob('source_補完*.json'):
        with open(path, 'r') as f:
            data = json.load(f)
            for item in data:
                name = item['target_name']
                supplements[name] = item['sources_to_add']

    return supplements

def main():
    print("=" * 60)
    print("ソース補完適用")
    print("=" * 60)

    # 補完データ読み込み
    supplements = load_supplements()
    print(f"補完対象: {len(supplements)}件")

    # cases.jsonl読み込み
    cases = []
    with open(DATA_DIR / 'raw' / 'cases.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    print(f"全事例数: {len(cases)}件")

    # 補完適用
    updated_count = 0
    updated_names = []

    for case in cases:
        name = case.get('target_name', '')

        # 完全一致チェック
        if name in supplements:
            new_sources = supplements[name]
            existing = case.get('sources', [])
            if not existing:
                existing = [case.get('source')] if case.get('source') else []

            # 新ソース追加
            case['sources'] = list(set(existing + new_sources))
            updated_count += 1
            updated_names.append(name)

    print(f"更新件数: {updated_count}件")

    # 書き戻し
    with open(DATA_DIR / 'raw' / 'cases.jsonl', 'w') as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')

    print(f"cases.jsonl更新完了")

    # 更新一覧
    print()
    print("【更新事例】")
    for name in updated_names[:20]:
        print(f"  - {name}")
    if len(updated_names) > 20:
        print(f"  ... 他{len(updated_names) - 20}件")

    return updated_count

if __name__ == '__main__':
    main()
