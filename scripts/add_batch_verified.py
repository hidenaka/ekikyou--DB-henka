#!/usr/bin/env python3
"""
検証付き事例追加スクリプト

通常のadd_batch.pyの代わりに使用。
事例追加前にURL検証を実施し、無効URLがある場合は警告。

使用方法:
  python3 scripts/add_batch_verified.py data/import/xxx.json
"""

import json
import sys
import requests
from pathlib import Path

# 既存のadd_batch機能をインポート
sys.path.insert(0, str(Path(__file__).parent))
from add_batch import main as add_batch_main


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


def verify_batch(filepath):
    """バッチファイルのURL検証"""
    with open(filepath, 'r', encoding='utf-8') as f:
        cases = json.load(f)

    print(f"=== URL検証: {filepath} ===")
    print(f"事例数: {len(cases)}")

    invalid_cases = []
    for i, case in enumerate(cases):
        sources = case.get('sources', [])
        for url in sources:
            if url and url.startswith('http'):
                if not check_url(url):
                    invalid_cases.append({
                        'index': i,
                        'target_name': case.get('target_name'),
                        'url': url
                    })

    if invalid_cases:
        print(f"\n⚠️ 無効URL検出: {len(invalid_cases)}件")
        for item in invalid_cases[:5]:
            print(f"  [{item['index']}] {item['target_name']}")
            print(f"      {item['url'][:60]}...")

        print(f"\n続行しますか？ (y/n): ", end='')
        response = input().strip().lower()
        if response != 'y':
            print("中止しました。URLを修正してから再実行してください。")
            return False
    else:
        print("✓ 全URLが有効です")

    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/add_batch_verified.py <batch_file.json>")
        sys.exit(1)

    filepath = sys.argv[1]

    # URL検証
    if not verify_batch(filepath):
        sys.exit(1)

    # 通常のadd_batch実行
    print("\n事例を追加中...")
    add_batch_main()


if __name__ == '__main__':
    main()
