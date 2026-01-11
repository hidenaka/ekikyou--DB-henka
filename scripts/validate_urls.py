#!/usr/bin/env python3
"""
URL品質検証スクリプト
cases.jsonlからランダムサンプリングしてURLの有効性を検証
"""

import json
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import sys
from pathlib import Path

# 設定
SAMPLE_SIZE = 100
TIMEOUT = 5
MAX_WORKERS = 10

def load_cases_with_urls(jsonl_path: Path) -> list:
    """URLを持つ事例を読み込む"""
    cases_with_urls = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                case = json.loads(line)
                sources = case.get('sources', [])
                if sources and len(sources) > 0:
                    first_url = sources[0] if isinstance(sources[0], str) else sources[0].get('url', '')
                    if first_url and first_url.startswith('http'):
                        cases_with_urls.append({
                            'id': case.get('id', 'unknown'),
                            'title': case.get('title', 'unknown'),
                            'url': first_url
                        })
    return cases_with_urls

def check_url(case_info: dict) -> dict:
    """URLをチェックして結果を返す"""
    url = case_info['url']
    result = {
        'id': case_info['id'],
        'title': case_info['title'],
        'url': url,
        'status': None,
        'category': None,
        'error': None
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    try:
        # まずHEADを試す
        response = requests.head(url, timeout=TIMEOUT, headers=headers, allow_redirects=True)
        result['status'] = response.status_code
    except requests.exceptions.Timeout:
        result['category'] = 'timeout'
        result['error'] = 'Timeout'
        return result
    except requests.exceptions.ConnectionError as e:
        result['category'] = 'connection_error'
        result['error'] = str(e)[:100]
        return result
    except requests.exceptions.SSLError as e:
        result['category'] = 'ssl_error'
        result['error'] = str(e)[:100]
        return result
    except Exception as e:
        result['category'] = 'other_error'
        result['error'] = str(e)[:100]
        return result
    
    # ステータスコードで分類
    status = result['status']
    if 200 <= status < 300:
        result['category'] = 'success'
    elif 300 <= status < 400:
        result['category'] = 'redirect'
    elif 400 <= status < 500:
        result['category'] = 'client_error'
    elif 500 <= status < 600:
        result['category'] = 'server_error'
    else:
        result['category'] = 'unknown'
    
    return result

def main():
    jsonl_path = Path("/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/data/raw/cases.jsonl")
    
    print("=" * 60)
    print("URL品質検証スクリプト")
    print("=" * 60)
    
    # URLを持つ事例を読み込み
    print("\n[1] cases.jsonl読み込み中...")
    cases_with_urls = load_cases_with_urls(jsonl_path)
    print(f"    URLを持つ事例数: {len(cases_with_urls)}件")
    
    # ランダムサンプリング
    sample_size = min(SAMPLE_SIZE, len(cases_with_urls))
    print(f"\n[2] {sample_size}件をランダムサンプリング...")
    sampled = random.sample(cases_with_urls, sample_size)
    
    # 並列でURL検証
    print(f"\n[3] URL検証実行中（並列数: {MAX_WORKERS}）...")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(check_url, case): case for case in sampled}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 20 == 0:
                print(f"    進捗: {completed}/{sample_size}")
            results.append(future.result())
    
    # 結果集計
    print("\n[4] 結果集計...")
    categories = defaultdict(list)
    for r in results:
        categories[r['category']].append(r)
    
    # 結果出力
    print("\n" + "=" * 60)
    print("検証結果サマリー")
    print("=" * 60)
    
    category_labels = {
        'success': '成功 (HTTP 200-299)',
        'redirect': 'リダイレクト (HTTP 300-399)',
        'client_error': 'クライアントエラー (HTTP 400-499)',
        'server_error': 'サーバーエラー (HTTP 500-599)',
        'timeout': 'タイムアウト',
        'connection_error': '接続エラー',
        'ssl_error': 'SSLエラー',
        'other_error': 'その他エラー',
        'unknown': '不明'
    }
    
    total = len(results)
    for cat, label in category_labels.items():
        count = len(categories[cat])
        if count > 0:
            pct = count / total * 100
            print(f"  {label}: {count}件 ({pct:.1f}%)")
    
    # 問題のあるURL例
    problem_categories = ['client_error', 'server_error', 'timeout', 'connection_error', 'ssl_error', 'other_error']
    problem_urls = []
    for cat in problem_categories:
        problem_urls.extend(categories[cat])
    
    if problem_urls:
        print("\n" + "-" * 60)
        print("問題のあるURL例（最大10件）")
        print("-" * 60)
        for r in problem_urls[:10]:
            print(f"\n  ID: {r['id']}")
            print(f"  タイトル: {r['title'][:50]}...")
            print(f"  URL: {r['url'][:80]}...")
            print(f"  問題: {r['category']} (status={r['status']}, error={r.get('error', '-')})")
    
    # 成功率
    success_count = len(categories['success']) + len(categories['redirect'])
    success_rate = success_count / total * 100
    print("\n" + "=" * 60)
    print(f"URL有効率: {success_rate:.1f}% ({success_count}/{total}件)")
    print("=" * 60)

if __name__ == '__main__':
    main()
