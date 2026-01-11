#!/usr/bin/env python3
"""
信頼度タグ付けロジックのテストスクリプト
"""

import json
import re
import urllib.request
import urllib.parse
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR / "data/diagnostic/sample_100_for_verification.json"
OUTPUT_FILE = BASE_DIR / "data/diagnostic/trust_level_sample_test.json"


def search_wikipedia_ja(query):
    """Wikipedia日本語版でクエリを検索"""
    clean_query = re.sub(r'[（(].*?[）)]', '', query)
    clean_query = re.sub(r'_.*', '', clean_query).strip()
    if len(clean_query) < 2:
        return {'found': False, 'reason': 'query_too_short'}
    
    url = 'https://ja.wikipedia.org/w/api.php'
    params = {
        'action': 'query', 'list': 'search',
        'srsearch': clean_query, 'format': 'json', 'srlimit': 3
    }
    full_url = url + '?' + urllib.parse.urlencode(params)
    
    try:
        req = urllib.request.Request(full_url, headers={'User-Agent': 'HaQeiBot/1.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            results = data.get('query', {}).get('search', [])
            if results:
                return {'found': True, 'title': results[0]['title']}
    except Exception as e:
        return {'found': False, 'reason': f'api_error:{str(e)[:50]}'}
    
    return {'found': False, 'reason': 'no_results'}


def classify_trust(target_name, scale):
    """信頼度を分類"""
    # 1. 自動unverified: 一般名詞パターン
    if re.match(r'^(個人|人気|一般|サンプル|架空)', target_name):
        return 'unverified', 'generic_pattern'
    
    # 2. 事例ID系のパターン（〜事例NNN-N）
    if re.search(r'事例\d+-\d+$', target_name):
        return 'unverified', 'id_pattern'
    
    # 3. Wikipedia API検索
    result = search_wikipedia_ja(target_name)
    if result and result.get('found'):
        wiki_title = result['title']
        # target_nameとWikiタイトルの類似度をチェック
        clean_name = re.sub(r'[（(].*?[）)]', '', target_name).strip()
        if clean_name in wiki_title or wiki_title in clean_name:
            return 'verified', f'wiki_match:{wiki_title}'
        else:
            return 'plausible', f'wiki_related:{wiki_title}'
    
    # 4. Wikipedia未発見
    reason = result.get('reason', 'not_found') if result else 'not_found'
    return 'unverified', reason


def main():
    print("=" * 60)
    print("信頼度タグ付けテスト")
    print("=" * 60)
    
    # サンプルデータ読み込み
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    print(f"読み込み件数: {len(samples)}件")
    print()
    
    # 分類結果格納
    results = {
        'verified': [],
        'plausible': [],
        'unverified': []
    }
    
    # API呼び出しカウンタ
    api_calls = 0
    
    for i, case in enumerate(samples):
        target_name = case.get('target_name', '')
        scale = case.get('scale', '')
        case_id = case.get('id', f'unknown_{i}')
        
        # 分類実行
        trust_level, reason = classify_trust(target_name, scale)
        
        # API呼び出しがあった場合（generic/id_pattern以外）
        if reason not in ('generic_pattern', 'id_pattern'):
            api_calls += 1
            if api_calls % 10 == 0:
                print(f"  処理中... {i+1}/{len(samples)} (API calls: {api_calls})")
            time.sleep(0.3)  # レートリミット対策
        
        # 結果格納
        results[trust_level].append({
            'id': case_id,
            'target_name': target_name,
            'scale': scale,
            'reason': reason
        })
    
    print()
    print("=" * 60)
    print("結果サマリー")
    print("=" * 60)
    
    # 件数出力
    print(f"\n【分類件数】")
    print(f"  verified:   {len(results['verified']):3d}件 (Wikipedia一致)")
    print(f"  plausible:  {len(results['plausible']):3d}件 (Wikipedia関連)")
    print(f"  unverified: {len(results['unverified']):3d}件 (未確認)")
    print(f"  合計:       {sum(len(v) for v in results.values()):3d}件")
    
    # 各カテゴリのサンプル5件
    print(f"\n【verified サンプル (最大5件)】")
    for item in results['verified'][:5]:
        print(f"  - {item['target_name']} [{item['scale']}]")
        print(f"    reason: {item['reason']}")
    
    print(f"\n【plausible サンプル (最大5件)】")
    for item in results['plausible'][:5]:
        print(f"  - {item['target_name']} [{item['scale']}]")
        print(f"    reason: {item['reason']}")
    
    print(f"\n【unverified サンプル (最大5件)】")
    for item in results['unverified'][:5]:
        print(f"  - {item['target_name']} [{item['scale']}]")
        print(f"    reason: {item['reason']}")
    
    # 結果をJSONファイルに保存
    output_data = {
        'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_cases': len(samples),
        'api_calls': api_calls,
        'summary': {
            'verified': len(results['verified']),
            'plausible': len(results['plausible']),
            'unverified': len(results['unverified'])
        },
        'details': results
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n結果を保存しました: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
