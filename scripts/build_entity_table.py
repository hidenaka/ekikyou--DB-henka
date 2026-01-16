#!/usr/bin/env python3
"""
主体正規化テーブル生成

Codex批判対応:
- ハッシュではなく、canonical主体テーブル＋alias解決を中核に据える
- 表記ゆれ（株式会社/略称/旧社名/英名）を統合
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def normalize_company_name(name: str) -> str:
    """企業名を正規化"""
    # 括弧内の説明を除去
    name = re.sub(r'[（\(].+?[）\)]', '', name)
    # 法人形態を除去
    name = re.sub(r'(株式会社|株|有限会社|合同会社|Inc\.|Corp\.|Ltd\.|LLC|Co\.)', '', name)
    # 前後の空白・記号を除去
    name = name.strip(' 　・')
    return name

def extract_aliases(cases: list) -> dict:
    """target_nameから表記ゆれを抽出"""
    # 正規化名 -> [元の名前のリスト]
    normalized_to_originals = defaultdict(set)

    for case in cases:
        original = case.get('target_name', '')
        normalized = normalize_company_name(original)
        if normalized:
            normalized_to_originals[normalized].add(original)

    return normalized_to_originals

def generate_entity_id(entity_type: str, country: str, index: int) -> str:
    """エンティティIDを生成

    フォーマット: {TYPE}_{COUNTRY}_{INDEX:06d}
    """
    type_prefix = {
        'company': 'CORP',
        'individual': 'INDV',
        'government': 'GOVT',
        'organization': 'ORG',
    }
    prefix = type_prefix.get(entity_type, 'UNKN')
    return f"{prefix}_{country}_{index:06d}"

def build_entity_table(dry_run: bool = True):
    """主体正規化テーブルを構築"""
    cases_path = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"
    entity_table_path = Path(__file__).parent.parent / "data" / "master" / "entity_table.json"
    alias_table_path = Path(__file__).parent.parent / "data" / "master" / "alias_table.json"

    cases = []
    with open(cases_path, 'r', encoding='utf-8') as f:
        for line in f:
            cases.append(json.loads(line.strip()))

    print(f"\n{'='*60}")
    print(f"主体正規化テーブル構築")
    print(f"{'='*60}")
    print(f"総事例数: {len(cases):,}件")

    # 表記ゆれを抽出
    normalized_to_originals = extract_aliases(cases)
    print(f"ユニーク正規化名: {len(normalized_to_originals):,}")

    # 表記ゆれがある主体を特定
    aliases_count = sum(1 for v in normalized_to_originals.values() if len(v) > 1)
    print(f"表記ゆれのある主体: {aliases_count:,}")

    # エンティティテーブルを構築
    entity_table = {}  # entity_id -> {canonical_name, entity_type, country, aliases}
    alias_table = {}   # original_name -> entity_id

    # 国別・タイプ別のインデックス
    counters = defaultdict(lambda: defaultdict(int))

    for normalized, originals in normalized_to_originals.items():
        # 代表的な事例を取得（最初に見つかったもの）
        representative_case = None
        for case in cases:
            if normalize_company_name(case.get('target_name', '')) == normalized:
                representative_case = case
                break

        if not representative_case:
            continue

        entity_type = representative_case.get('entity_type', 'company')
        country = representative_case.get('country', 'JP')
        if country is None:
            country = 'JP'

        # インデックスを取得・更新
        counters[entity_type][country] += 1
        index = counters[entity_type][country]

        # エンティティIDを生成
        entity_id = generate_entity_id(entity_type, country, index)

        # 正準名（最も短い名前を選択）
        canonical_name = min(originals, key=len)

        entity_table[entity_id] = {
            'canonical_name': canonical_name,
            'entity_type': entity_type,
            'country': country,
            'aliases': list(originals),
            'case_count': sum(1 for c in cases
                            if normalize_company_name(c.get('target_name', '')) == normalized)
        }

        # aliasテーブルを構築
        for original in originals:
            alias_table[original] = entity_id

    print(f"\n--- 結果 ---")
    print(f"エンティティ数: {len(entity_table):,}")
    print(f"alias数: {len(alias_table):,}")

    # 統計
    type_counts = defaultdict(int)
    for entity in entity_table.values():
        type_counts[entity['entity_type']] += 1

    print(f"\n--- エンティティタイプ別 ---")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c:,}件")

    # 表記ゆれサンプル
    multi_alias = [(eid, e) for eid, e in entity_table.items() if len(e['aliases']) > 1]
    print(f"\n--- 表記ゆれサンプル（{len(multi_alias)}件中上位5件）---")
    for eid, e in sorted(multi_alias, key=lambda x: -len(x[1]['aliases']))[:5]:
        print(f"  {eid}: {e['canonical_name']}")
        print(f"    aliases: {', '.join(e['aliases'][:5])}")

    if dry_run:
        print(f"\n[DRY RUN] 変更は適用されません")
    else:
        # 保存
        entity_table_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n[保存] {entity_table_path}")
        with open(entity_table_path, 'w', encoding='utf-8') as f:
            json.dump(entity_table, f, ensure_ascii=False, indent=2)

        print(f"[保存] {alias_table_path}")
        with open(alias_table_path, 'w', encoding='utf-8') as f:
            json.dump(alias_table, f, ensure_ascii=False, indent=2)

        # cases.jsonlを更新（primary_subject_idを新IDに更新）
        print(f"\n[更新] cases.jsonl - primary_subject_idを正規化IDに更新")
        updated_count = 0
        for case in cases:
            target_name = case.get('target_name', '')
            if target_name in alias_table:
                new_id = alias_table[target_name]
                if case.get('primary_subject_id') != new_id:
                    case['primary_subject_id'] = new_id
                    updated_count += 1

        # バックアップ
        backup_path = cases_path.parent / f"cases_backup_entity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(backup_path, 'w', encoding='utf-8') as f:
            with open(cases_path, 'r', encoding='utf-8') as src:
                f.write(src.read())

        # 保存
        with open(cases_path, 'w', encoding='utf-8') as f:
            for case in cases:
                f.write(json.dumps(case, ensure_ascii=False) + '\n')

        print(f"  更新件数: {updated_count:,}件")
        print(f"\n✅ 主体正規化完了")

    return entity_table, alias_table

if __name__ == "__main__":
    import sys
    dry_run = '--dry-run' in sys.argv or '-n' in sys.argv
    build_entity_table(dry_run=dry_run)
