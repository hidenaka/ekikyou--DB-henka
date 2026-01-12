#!/usr/bin/env python3
"""
無効URLの修正スクリプト

1. ペイウォールサイト（日経、BBCなど）はevidence_notesに注記
2. 企業公式サイトはトップページに修正
3. Wikipediaリンクは日本語版に修正
"""

import json
import re
from pathlib import Path

CASES_FILE = Path("data/raw/cases.jsonl")
INVALID_URLS_FILE = Path("data/diagnostic/invalid_urls.json")

# ペイウォールサイト（アクセス制限あり、URLは有効だが検証困難）
PAYWALL_DOMAINS = [
    'nikkei.com', 'wsj.com', 'nytimes.com', 'ft.com',
    'economist.com', 'bbc.com', 'reuters.com'
]

# 企業公式サイトのトップページマッピング
COMPANY_TOP_PAGES = {
    'www.mercari.com': 'https://about.mercari.com/',
    'www.zozo.jp': 'https://corp.zozo.com/',
    'www.keyence.co.jp': 'https://www.keyence.co.jp/company/',
    'global.toyota': 'https://global.toyota/jp/',
    'www.sony.com': 'https://www.sony.com/ja/',
    'www.honda.co.jp': 'https://www.honda.co.jp/about/',
    'www.suntory.co.jp': 'https://www.suntory.co.jp/company/',
}


def load_invalid_urls():
    """無効URLリストを読み込み"""
    with open(INVALID_URLS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_cases():
    """cases.jsonlを読み込み"""
    cases = []
    with open(CASES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def save_cases(cases):
    """cases.jsonlを保存"""
    with open(CASES_FILE, 'w', encoding='utf-8') as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')


def is_paywall_url(url):
    """ペイウォールサイトかどうか"""
    return any(domain in url for domain in PAYWALL_DOMAINS)


def fix_url(url, target_name):
    """URLを修正"""
    # ペイウォールサイトはそのまま
    if is_paywall_url(url):
        return url, "paywall"

    # 企業トップページに修正
    for domain, top_page in COMPANY_TOP_PAGES.items():
        if domain in url:
            return top_page, "fixed_to_top"

    # 404の企業サイトはWikipediaに変更を検討
    return url, "needs_manual"


def main():
    invalid_urls = load_invalid_urls()
    cases = load_cases()

    # 無効URLを持つ事例のtarget_nameを抽出
    invalid_targets = {item['target_name']: item['url'] for item in invalid_urls}

    paywall_count = 0
    fixed_count = 0
    manual_count = 0

    for case in cases:
        target = case.get('target_name')
        if target in invalid_targets:
            sources = case.get('sources', [])
            new_sources = []
            notes = []

            for url in sources:
                new_url, status = fix_url(url, target)
                new_sources.append(new_url)

                if status == "paywall":
                    notes.append(f"ペイウォール: {url[:50]}")
                    paywall_count += 1
                elif status == "fixed_to_top":
                    notes.append(f"トップページに修正")
                    fixed_count += 1
                elif status == "needs_manual":
                    manual_count += 1

            case['sources'] = new_sources

            # evidence_notesに追記
            existing_notes = case.get('evidence_notes') or ''
            if notes:
                case['evidence_notes'] = existing_notes + ' | '.join(notes)

    save_cases(cases)

    print(f"=== 修正結果 ===")
    print(f"ペイウォール注記: {paywall_count}件")
    print(f"トップページ修正: {fixed_count}件")
    print(f"手動確認必要: {manual_count}件")


if __name__ == '__main__':
    main()
