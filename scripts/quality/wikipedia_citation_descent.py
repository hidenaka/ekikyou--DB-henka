#!/usr/bin/env python3
"""
Wikipedia引用降下スクリプト

Wikipedia記事から外部リンク（引用）を抽出し、
一次/二次ソースを取得してverification_confidenceを昇格する。

使用方法:
    python3 scripts/quality/wikipedia_citation_descent.py [--max N] [--dry-run]

オプション:
    --max N      最大処理件数（デフォルト: 100）
    --dry-run    実際の更新を行わずに結果を表示
"""

import json
import re
import time
import urllib.parse
import urllib.request
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

# パス設定
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CASES_FILE = BASE_DIR / "data" / "raw" / "cases.jsonl"
OUTPUT_DIR = BASE_DIR / "data" / "enriched"
OUTPUT_FILE = OUTPUT_DIR / "wikipedia_descended.jsonl"

# MediaWiki API設定
API_ENDPOINT = "https://ja.wikipedia.org/w/api.php"
REQUEST_DELAY = 1.0  # 秒
REQUEST_TIMEOUT = 5  # 秒

# ソース分類ルール
PRIMARY_SOURCE_PATTERNS = [
    r'\.go\.jp',           # 政府機関
    r'\.ac\.jp',           # 学術機関
    r'\.edu',              # 教育機関
    r'/ir/',               # IR情報
    r'/investor',          # 投資家向け情報
    r'/press/',            # プレスリリース
    r'/newsroom/',         # ニュースルーム
    r'\.co\.jp/[^/]*news', # 企業ニュースページ
    r'\.com/[^/]*press',   # プレスリリース
]

SECONDARY_SOURCE_PATTERNS = [
    r'nikkei\.com',        # 日経
    r'reuters\.com',       # ロイター
    r'bloomberg\.com',     # ブルームバーグ
    r'wsj\.com',           # WSJ
    r'ft\.com',            # FT
    r'economist\.com',     # エコノミスト
    r'asahi\.com',         # 朝日新聞
    r'yomiuri\.co\.jp',    # 読売新聞
    r'mainichi\.jp',       # 毎日新聞
    r'sankei\.com',        # 産経新聞
    r'nhk\.or\.jp',        # NHK
    r'bbc\.(com|co\.uk)',  # BBC
    r'cnn\.com',           # CNN
    r'nytimes\.com',       # NYT
    r'forbes\.com',        # Forbes
    r'techcrunch\.com',    # TechCrunch
    r'wired\.(com|jp)',    # Wired
    r'itmedia\.co\.jp',    # ITmedia
    r'impress\.co\.jp',    # Impress
    r'toyo ?keizai',       # 東洋経済
    r'diamond\.jp',        # ダイヤモンド
]


def extract_page_title_from_url(url: str) -> Optional[str]:
    """WikipediaのURLから記事タイトルを抽出"""
    # パターン: https://ja.wikipedia.org/wiki/記事タイトル
    match = re.search(r'wikipedia\.org/wiki/([^#?]+)', url)
    if match:
        title = urllib.parse.unquote(match.group(1))
        return title.replace('_', ' ')
    return None


def fetch_external_links(page_title: str) -> list[str]:
    """MediaWiki APIを使用して外部リンクを取得"""
    params = {
        "action": "parse",
        "page": page_title,
        "prop": "externallinks",
        "format": "json"
    }

    url = f"{API_ENDPOINT}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "YijingDB/1.0 (Citation Descent)"}
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
            data = json.loads(response.read().decode('utf-8'))

            if "error" in data:
                print(f"  API error: {data['error'].get('info', 'Unknown error')}")
                return []

            if "parse" in data and "externallinks" in data["parse"]:
                return data["parse"]["externallinks"]

    except urllib.error.URLError as e:
        print(f"  URL error: {e}")
    except json.JSONDecodeError as e:
        print(f"  JSON decode error: {e}")
    except Exception as e:
        print(f"  Unexpected error: {e}")

    return []


def classify_source(url: str) -> str:
    """URLをソースタイプに分類"""
    url_lower = url.lower()

    # Wikipedia自体は除外
    if 'wikipedia.org' in url_lower:
        return 'wikipedia'

    # 一次ソース判定
    for pattern in PRIMARY_SOURCE_PATTERNS:
        if re.search(pattern, url_lower):
            return 'primary'

    # 二次ソース判定
    for pattern in SECONDARY_SOURCE_PATTERNS:
        if re.search(pattern, url_lower):
            return 'secondary'

    return 'other'


def determine_new_confidence(primary_count: int, secondary_count: int) -> tuple[str, str]:
    """新しいverification_confidenceを決定"""
    if primary_count >= 1:
        return 'high', f'primary_source_found({primary_count})'
    elif secondary_count >= 2:
        return 'high', f'multiple_secondary_sources({secondary_count})'
    elif secondary_count >= 1:
        return 'medium', f'secondary_source_found({secondary_count})'
    else:
        return 'low', 'no_quality_source_found'


def process_case(case: dict) -> Optional[dict]:
    """1件の事例を処理"""
    sources = case.get('sources') or []
    current_conf = case.get('verification_confidence', 'low')

    # Wikipedia URLを探す
    wiki_urls = [s for s in sources if s and 'wikipedia' in s.lower()]

    if not wiki_urls:
        return None

    all_external_links = []
    processed_titles = set()

    for wiki_url in wiki_urls:
        title = extract_page_title_from_url(wiki_url)
        if not title or title in processed_titles:
            continue

        processed_titles.add(title)
        print(f"  Fetching: {title}")

        links = fetch_external_links(title)
        all_external_links.extend(links)

        time.sleep(REQUEST_DELAY)

    if not all_external_links:
        return None

    # リンクを分類
    classified = {
        'primary': [],
        'secondary': [],
        'other': [],
        'wikipedia': []
    }

    for link in all_external_links:
        source_type = classify_source(link)
        classified[source_type].append(link)

    primary_count = len(classified['primary'])
    secondary_count = len(classified['secondary'])

    # 新しい信頼度を決定
    new_conf, reason = determine_new_confidence(primary_count, secondary_count)

    # 結果を構築
    result = {
        'transition_id': case.get('transition_id'),
        'target_name': case.get('target_name'),
        'original_confidence': current_conf,
        'new_confidence': new_conf,
        'confidence_reason': reason,
        'wikipedia_pages': list(processed_titles),
        'external_links_found': len(all_external_links),
        'primary_sources': classified['primary'][:10],  # 最大10件
        'secondary_sources': classified['secondary'][:10],
        'other_sources_count': len(classified['other']),
        'upgraded': new_conf != current_conf and (new_conf == 'high' or (new_conf == 'medium' and current_conf == 'low'))
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='Wikipedia引用降下スクリプト')
    parser.add_argument('--max', type=int, default=100, help='最大処理件数')
    parser.add_argument('--dry-run', action='store_true', help='実際の更新を行わない')
    args = parser.parse_args()

    print("=" * 60)
    print("Wikipedia Citation Descent")
    print("=" * 60)
    print(f"開始時刻: {datetime.now().isoformat()}")
    print(f"最大処理件数: {args.max}")
    print(f"Dry-run: {args.dry_run}")
    print()

    # 対象事例を抽出
    candidates = []
    with open(CASES_FILE, 'r') as f:
        for line in f:
            case = json.loads(line)
            sources = case.get('sources') or []
            conf = case.get('verification_confidence', '')

            # low confidence かつ Wikipedia ソースを持つ事例
            has_wiki = any('wikipedia' in s.lower() for s in sources if s)
            if conf == 'low' and has_wiki:
                candidates.append(case)

    print(f"対象候補: {len(candidates)} 件")
    print(f"処理対象: {min(len(candidates), args.max)} 件")
    print()

    # 出力ディレクトリ作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 処理実行
    results = []
    processed = 0
    upgraded = 0
    errors = 0

    for i, case in enumerate(candidates[:args.max]):
        case_id = case.get('transition_id', 'unknown')
        target = case.get('target_name', 'unknown')

        print(f"[{i+1}/{args.max}] {case_id}: {target}")

        try:
            result = process_case(case)
            if result:
                results.append(result)
                if result.get('upgraded'):
                    upgraded += 1
                    print(f"  -> Upgraded: {result['original_confidence']} -> {result['new_confidence']}")
                else:
                    print(f"  -> No upgrade: {result['confidence_reason']}")
                processed += 1
            else:
                print("  -> No external links found")
        except Exception as e:
            print(f"  -> Error: {e}")
            errors += 1

    # 結果をファイルに書き出し
    with open(OUTPUT_FILE, 'w') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # 統計レポート
    print()
    print("=" * 60)
    print("統計レポート")
    print("=" * 60)
    print(f"処理件数: {processed}")
    print(f"エラー件数: {errors}")
    print(f"昇格成功: {upgraded} ({100*upgraded/max(processed,1):.1f}%)")
    print()

    # ソース内訳
    total_primary = sum(len(r.get('primary_sources', [])) for r in results)
    total_secondary = sum(len(r.get('secondary_sources', [])) for r in results)
    total_other = sum(r.get('other_sources_count', 0) for r in results)

    print("発見されたソースの内訳:")
    print(f"  一次ソース: {total_primary} 件")
    print(f"  二次ソース: {total_secondary} 件")
    print(f"  その他: {total_other} 件")
    print()

    # 信頼度変更の内訳
    conf_changes = {}
    for r in results:
        key = f"{r['original_confidence']} -> {r['new_confidence']}"
        conf_changes[key] = conf_changes.get(key, 0) + 1

    print("信頼度変更の内訳:")
    for change, count in sorted(conf_changes.items()):
        print(f"  {change}: {count} 件")
    print()

    print(f"出力ファイル: {OUTPUT_FILE}")
    print(f"完了時刻: {datetime.now().isoformat()}")

    return results


if __name__ == '__main__':
    main()
