#!/usr/bin/env python3
"""
信頼度タグ付けスクリプト
cases.jsonlの全事例にtrust_levelを付与
"""

import json
import re
import urllib.request
import urllib.parse
import time
import sys
from pathlib import Path

# Wikipedia検索キャッシュ
wiki_cache = {}

def search_wikipedia_ja(query):
    """Wikipedia日本語版でタイトル検索（キャッシュ付き）"""
    if query in wiki_cache:
        return wiki_cache[query]

    # クエリのクリーニング
    clean_query = re.sub(r'[（(].*?[）)]', '', query)
    clean_query = re.sub(r'_.*', '', clean_query)
    clean_query = re.sub(r'\s*(の|から|への|による).*', '', clean_query)
    clean_query = clean_query.strip()

    # 短すぎるクエリはスキップ
    if len(clean_query) < 3:
        wiki_cache[query] = {'found': False, 'reason': 'too_short'}
        return wiki_cache[query]

    url = 'https://ja.wikipedia.org/w/api.php'
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': clean_query,
        'format': 'json',
        'srlimit': 3
    }
    full_url = url + '?' + urllib.parse.urlencode(params)

    try:
        req = urllib.request.Request(full_url, headers={'User-Agent': 'HaQeiBot/1.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            results = data.get('query', {}).get('search', [])
            if results:
                result = {
                    'found': True,
                    'title': results[0]['title'],
                    'clean_query': clean_query
                }
                wiki_cache[query] = result
                return result
    except Exception as e:
        wiki_cache[query] = {'found': False, 'error': str(e)}
        return wiki_cache[query]

    wiki_cache[query] = {'found': False}
    return wiki_cache[query]


def is_strict_match(target_name, wiki_title):
    """厳格な一致判定"""
    # クリーニング
    clean_name = re.sub(r'[（(].*?[）)]', '', target_name)
    clean_name = re.sub(r'_.*', '', clean_name)
    clean_name = re.sub(r'\s*(の|から|への|による).*', '', clean_name)
    clean_name = clean_name.strip()

    clean_title = wiki_title.strip()

    # 完全一致
    if clean_name == clean_title:
        return True

    # 一方が他方を含む（かつ長さの差が小さい）
    if clean_name in clean_title and len(clean_name) >= len(clean_title) * 0.5:
        return True
    if clean_title in clean_name and len(clean_title) >= len(clean_name) * 0.5:
        return True

    return False


def classify_trust(target_name, scale=None):
    """信頼度を分類"""
    if not target_name:
        return 'unverified', 'empty_name'

    # 1. 自動unverified: 一般名詞パターン
    generic_patterns = [
        r'^(個人|人気|一般|サンプル|架空|匿名)',
        r'^[A-Z]さん',
        r'^[あ-ん]さん',
        r'事例\d+-\d+$',
        r'^(震|巽|坎|離|艮|兌|乾|坤)(前|後|行動|結果|輝き).*事例',
        r'^近世商人\d+',
        r'^\d+代.*男性',
        r'^\d+代.*女性',
    ]

    for pattern in generic_patterns:
        if re.search(pattern, target_name):
            return 'unverified', 'generic_pattern'

    # 2. Wikipedia検索
    result = search_wikipedia_ja(target_name)

    if result.get('found'):
        wiki_title = result['title']
        if is_strict_match(target_name, wiki_title):
            return 'verified', f'wiki:{wiki_title}'
        else:
            # 関連があっても厳格一致でなければplausible
            return 'plausible', f'wiki_related:{wiki_title}'

    # 3. 未発見だが、企業名・国名パターンなら plausible
    if scale == 'company':
        if re.search(r'(株式会社|Inc\.|Corp\.|Ltd\.|HD|ホールディングス)', target_name):
            return 'plausible', 'company_pattern'

    if scale == 'country':
        return 'plausible', 'country_scale'

    # 4. 未確認
    return 'unverified', 'not_found'


def main():
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / 'data' / 'raw' / 'cases.jsonl'
    output_file = base_dir / 'data' / 'raw' / 'cases_with_trust.jsonl'

    # 全件読み込み
    cases = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            cases.append(json.loads(line))

    total = len(cases)
    print(f'Total cases: {total}')
    print('Processing...')

    stats = {'verified': 0, 'plausible': 0, 'unverified': 0}
    verified_samples = []

    for i, case in enumerate(cases):
        target_name = case.get('target_name', '')
        scale = case.get('scale', '')

        trust_level, reason = classify_trust(target_name, scale)
        case['trust_level'] = trust_level
        case['trust_reason'] = reason

        stats[trust_level] += 1

        if trust_level == 'verified' and len(verified_samples) < 50:
            verified_samples.append(target_name)

        # Progress
        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{total} processed...')
            time.sleep(0.1)  # Rate limit buffer

        # API rate limit (only for Wikipedia calls)
        if 'wiki' in reason and i < total - 1:
            time.sleep(0.15)

    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')

    print()
    print('=== 統計レポート ===')
    for level, count in stats.items():
        pct = count * 100 / total
        print(f'{level}: {count}件 ({pct:.1f}%)')

    print()
    print('=== verified事例サンプル (上位20) ===')
    for name in verified_samples[:20]:
        print(f'  - {name}')

    print()
    print(f'Output: {output_file}')

    # 統計をJSONでも保存
    stats_file = base_dir / 'data' / 'diagnostic' / 'trust_level_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total': total,
            'stats': stats,
            'verified_samples': verified_samples
        }, f, ensure_ascii=False, indent=2)

    print(f'Stats: {stats_file}')


if __name__ == '__main__':
    main()
