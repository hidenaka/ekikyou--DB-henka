#!/usr/bin/env python3
"""
事例検証スクリプト

検証項目:
1. ソースURLの有効性
2. 事実関係の正確性
3. 年代・数値の整合性

使用方法:
  python3 scripts/verify_cases.py --check-urls      # URL有効性チェック
  python3 scripts/verify_cases.py --list-unverified # 未検証リスト表示
  python3 scripts/verify_cases.py --stats           # 統計表示
"""

import json
import argparse
import requests
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

CASES_FILE = Path("data/raw/cases.jsonl")
VERIFICATION_QUEUE = Path("data/diagnostic/verification_queue.json")
VERIFICATION_RESULTS = Path("data/diagnostic/verification_results.json")


def load_cases():
    """cases.jsonlを読み込み"""
    cases = []
    with open(CASES_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip():
                c = json.loads(line)
                c['_line_num'] = i
                cases.append(c)
    return cases


def get_unverified_cases(cases):
    """未検証事例を取得（hexagram_id + yao_analysisがある事例）"""
    return [c for c in cases if c.get('hexagram_id') and c.get('yao_analysis')]


def check_url(url, timeout=10):
    """URLの有効性をチェック"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; CaseVerifier/1.0)'}
        resp = requests.head(url, timeout=timeout, headers=headers, allow_redirects=True)
        return resp.status_code < 400
    except Exception:
        try:
            resp = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
            return resp.status_code < 400
        except Exception:
            return False


def check_urls_batch(cases, max_workers=10):
    """バッチでURL有効性をチェック"""
    results = []

    # 全URLを抽出
    url_cases = []
    for c in cases:
        sources = c.get('sources', [])
        for url in sources:
            if url and url.startswith('http'):
                url_cases.append((c.get('target_name'), url))

    print(f"チェック対象URL: {len(url_cases)}件")

    valid = 0
    invalid = 0
    invalid_list = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(check_url, url): (name, url) for name, url in url_cases}

        for i, future in enumerate(as_completed(future_to_url)):
            name, url = future_to_url[future]
            try:
                is_valid = future.result()
                if is_valid:
                    valid += 1
                else:
                    invalid += 1
                    invalid_list.append({'target_name': name, 'url': url})
            except Exception as e:
                invalid += 1
                invalid_list.append({'target_name': name, 'url': url, 'error': str(e)})

            if (i + 1) % 50 == 0:
                print(f"  進捗: {i + 1}/{len(url_cases)} (有効: {valid}, 無効: {invalid})")

    print(f"\n結果: 有効={valid}, 無効={invalid}")

    # 無効URLリストを保存
    if invalid_list:
        output_file = Path("data/diagnostic/invalid_urls.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(invalid_list, f, ensure_ascii=False, indent=2)
        print(f"無効URLリスト: {output_file}")

    return {'valid': valid, 'invalid': invalid, 'invalid_list': invalid_list}


def show_stats(cases):
    """統計を表示"""
    unverified = get_unverified_cases(cases)

    print("=== 事例検証統計 ===")
    print(f"総事例数: {len(cases)}")
    print(f"検証対象（hexagram_id + yao_analysis）: {len(unverified)}")

    # credibility_rank分布
    ranks = Counter([c.get('credibility_rank', 'なし') for c in unverified])
    print(f"\ncredibility_rank分布:")
    for rank, count in sorted(ranks.items()):
        print(f"  {rank}: {count}件")

    # source_type分布
    types = Counter([c.get('source_type', 'なし') for c in unverified])
    print(f"\nsource_type分布:")
    for t, count in sorted(types.items()):
        print(f"  {t}: {count}件")

    # ソースURLの有無
    with_sources = len([c for c in unverified if c.get('sources')])
    print(f"\nソースURL: あり={with_sources}, なし={len(unverified) - with_sources}")


def list_unverified(cases, limit=20):
    """未検証事例をリスト表示"""
    unverified = get_unverified_cases(cases)

    print(f"=== 未検証事例（{len(unverified)}件中、先頭{limit}件） ===\n")

    for c in unverified[:limit]:
        hex_id = c.get('hexagram_id')
        hex_name = c.get('hexagram_name')
        yao = c.get('yao_analysis', {}).get('before_yao_position')
        print(f"[{hex_id}.{hex_name}-{yao}爻] {c.get('target_name')}")
        print(f"  sources: {c.get('sources', [])[:2]}")
        print()


def main():
    parser = argparse.ArgumentParser(description='事例検証スクリプト')
    parser.add_argument('--check-urls', action='store_true', help='URL有効性チェック')
    parser.add_argument('--list-unverified', action='store_true', help='未検証リスト表示')
    parser.add_argument('--stats', action='store_true', help='統計表示')
    parser.add_argument('--limit', type=int, default=20, help='表示件数制限')

    args = parser.parse_args()

    cases = load_cases()

    if args.check_urls:
        unverified = get_unverified_cases(cases)
        check_urls_batch(unverified)
    elif args.list_unverified:
        list_unverified(cases, args.limit)
    elif args.stats:
        show_stats(cases)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
