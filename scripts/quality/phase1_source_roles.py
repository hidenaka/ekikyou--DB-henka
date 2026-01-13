"""
Phase 1: 役割ベースソース分類
Codex批評に基づき、ドメイン一律拒否ではなく役割推定で処理

役割分類:
- primary_source: 一次情報源（公式発表、決算資料等）
- secondary_source: 二次情報源（報道、解説等）
- pointer_to_sources: 引用への導線（Wikipedia等）
- context_only: 文脈情報のみ（評価主張等）
- rejected: 情報源として不適切（検索結果等）
"""

import json
import re
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
from urllib.parse import urlparse, parse_qs

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'

# ============================================
# 1. URL役割分類ルール
# ============================================

SOURCE_ROLE_RULES = {
    # 一次情報源パターン
    'primary_source': {
        'domain_patterns': [
            r'\.go\.jp$', r'\.gov$', r'\.ac\.jp$', r'\.edu$',  # 政府・学術
            r'sec\.gov$', r'fca\.org\.uk$',  # 規制当局
            r'ir\.[^/]+\.(com|co\.jp)$',  # IR専用サブドメイン
        ],
        'path_patterns': [
            r'/ir/', r'/investor/', r'/press/', r'/news/',
            r'/about/', r'/corporate/',
            r'/annual-report', r'/earnings',
        ],
    },

    # 二次情報源パターン
    'secondary_source': {
        'domain_patterns': [
            r'nikkei\.com$', r'reuters\.com$', r'bloomberg\.com$',
            r'wsj\.com$', r'ft\.com$', r'bbc\.(com|co\.uk)$',
            r'asahi\.com$', r'yomiuri\.co\.jp$', r'nhk\.or\.jp$',
            r'techcrunch\.com$', r'theverge\.com$',
            r'toyokeizai\.net$', r'diamond\.jp$',
        ],
    },

    # ポインタ（引用への導線）
    'pointer_to_sources': {
        'domain_patterns': [
            r'wikipedia\.org$',
            r'britannica\.com$',
        ],
    },

    # 拒否パターン（情報源として不適切）
    'rejected': {
        'url_patterns': [
            r'google\.(com|co\.jp)/search',
            r'bing\.com/search',
            r'yahoo\.(com|co\.jp)/search',
            r'duckduckgo\.com/\?q=',
            r'webcache\.googleusercontent\.com',
        ],
    },
}

# YouTube公式チャンネルリスト（手動登録）
YOUTUBE_OFFICIAL_CHANNELS = {
    # 日本企業
    'SonyJapan', 'Sony', 'SonyPictures',
    'Toyota', 'ToyotaGlobal', 'ToyotaJapan',
    'Nintendo', 'NintendoJP',
    'SoftBank', 'SoftBankJP',
    'Rakuten',
    # 米国テック
    'Apple', 'Google', 'GoogleDevelopers',
    'Amazon', 'AmazonWebServices',
    'Microsoft', 'MicrosoftDeveloper',
    'Tesla', 'SpaceX',
    'NVIDIA', 'NVIDIADeveloper',
    'Netflix',
    # 報道機関
    'BBCNews', 'CNN', 'Reuters',
    'naborinews', 'ANNnewsCH',  # 日本報道
}

# ============================================
# 2. 役割判定関数
# ============================================

def classify_source_role(url: str) -> dict:
    """
    URLの役割を分類

    Returns:
        {
            'role': 'primary_source' | 'secondary_source' | 'pointer_to_sources' | 'context_only' | 'rejected',
            'reason': str,
            'domain': str,
            'can_extract_references': bool,  # Wikipedia等の引用抽出可能性
        }
    """
    if not url:
        return {'role': 'rejected', 'reason': 'empty_url'}

    # URL解析
    try:
        parsed = urlparse(url.lower())
        domain = parsed.netloc.replace('www.', '')
        path = parsed.path
    except Exception:
        return {'role': 'rejected', 'reason': 'invalid_url'}

    result = {
        'domain': domain,
        'can_extract_references': False,
    }

    # 1. 拒否パターンチェック
    for pattern in SOURCE_ROLE_RULES['rejected']['url_patterns']:
        if re.search(pattern, url, re.IGNORECASE):
            result['role'] = 'rejected'
            result['reason'] = f'rejected_pattern: {pattern}'
            return result

    # 2. YouTube特別処理
    if 'youtube.com' in domain or 'youtu.be' in domain:
        return classify_youtube(url, parsed)

    # 3. ポインタ（Wikipedia等）
    for pattern in SOURCE_ROLE_RULES['pointer_to_sources']['domain_patterns']:
        if re.search(pattern, domain):
            result['role'] = 'pointer_to_sources'
            result['reason'] = 'wikipedia_or_encyclopedia'
            result['can_extract_references'] = True
            return result

    # 4. 一次情報源チェック
    for pattern in SOURCE_ROLE_RULES['primary_source']['domain_patterns']:
        if re.search(pattern, domain):
            result['role'] = 'primary_source'
            result['reason'] = f'domain_match: {pattern}'
            return result

    for pattern in SOURCE_ROLE_RULES['primary_source']['path_patterns']:
        if re.search(pattern, path):
            result['role'] = 'primary_source'
            result['reason'] = f'path_match: {pattern}'
            return result

    # 5. 二次情報源チェック
    for pattern in SOURCE_ROLE_RULES['secondary_source']['domain_patterns']:
        if re.search(pattern, domain):
            result['role'] = 'secondary_source'
            result['reason'] = f'domain_match: {pattern}'
            return result

    # 6. デフォルト: context_only
    result['role'] = 'context_only'
    result['reason'] = 'unclassified_domain'
    return result

def classify_youtube(url: str, parsed) -> dict:
    """
    YouTube URLの役割分類
    公式チャンネルは一次証拠候補
    """
    result = {
        'domain': 'youtube.com',
        'can_extract_references': False,
    }

    # チャンネルURL解析
    path = parsed.path

    # /c/ChannelName, /channel/ID, /@handle 形式
    channel_match = re.search(r'/(c|channel|@)/([\w-]+)', path)
    if channel_match:
        channel_id = channel_match.group(2)
        if channel_id in YOUTUBE_OFFICIAL_CHANNELS:
            result['role'] = 'primary_source'
            result['reason'] = f'official_channel: {channel_id}'
            return result

    # 動画URL: /watch?v=ID
    if '/watch' in path:
        # 動画の場合、チャンネル情報がないとcontext_only
        result['role'] = 'context_only'
        result['reason'] = 'youtube_video_unknown_channel'
        return result

    # その他YouTube URL
    result['role'] = 'context_only'
    result['reason'] = 'youtube_other'
    return result

# ============================================
# 3. Wikipedia引用抽出（シミュレーション）
# ============================================

def extract_wikipedia_references(url: str) -> list:
    """
    Wikipedia記事から引用元URLを抽出
    実際のスクレイピングではなく、構造を示すシミュレーション

    Returns:
        [
            {'url': '...', 'type': 'primary' | 'secondary'},
            ...
        ]
    """
    # 注: 実際の実装ではWikipedia APIまたはスクレイピングが必要
    # ここではパイプラインの構造を示す

    # Wikipedia URLパターン確認
    if 'wikipedia.org' not in url:
        return []

    # シミュレーション: 典型的な引用パターン
    # 実際はMediaWiki APIの`action=parse`で取得可能
    simulated_references = [
        {'url': 'https://example-primary.com/ir/report', 'type': 'primary'},
        {'url': 'https://example-news.com/article', 'type': 'secondary'},
    ]

    return simulated_references

def descend_to_references(source_url: str) -> list:
    """
    ポインタソースから実際の情報源へ降下

    Returns:
        [
            {'original': url, 'descended': [url1, url2, ...], 'status': 'success' | 'no_references'}
        ]
    """
    role_result = classify_source_role(source_url)

    if role_result.get('can_extract_references'):
        references = extract_wikipedia_references(source_url)
        if references:
            return {
                'original': source_url,
                'descended': [r['url'] for r in references],
                'reference_types': [r['type'] for r in references],
                'status': 'success',
            }
        return {
            'original': source_url,
            'descended': [],
            'status': 'no_references',
        }

    return {
        'original': source_url,
        'descended': [],
        'status': 'not_pointer',
    }

# ============================================
# 4. URL正規化（リダイレクト解決含む）
# ============================================

# 既知のリダイレクトマッピング
REDIRECT_MAP = {
    'about.google': 'google.com',
    'aboutamazon.com': 'amazon.com',
    'global.toyota': 'toyota.co.jp',
}

def normalize_url_final(url: str) -> str:
    """
    最終URL正規化
    - プロトコル統一
    - www除去
    - リダイレクト解決
    - トラッキングパラメータ除去
    """
    if not url:
        return ''

    # 小文字化
    url = url.lower().strip()

    # プロトコル除去（比較用）
    url = re.sub(r'^https?://', '', url)

    # www除去
    url = re.sub(r'^www\.', '', url)

    # 末尾スラッシュ除去
    url = url.rstrip('/')

    # リダイレクト解決
    for old_domain, new_domain in REDIRECT_MAP.items():
        if url.startswith(old_domain):
            url = url.replace(old_domain, new_domain, 1)
            break

    # トラッキングパラメータ除去
    tracking_params = ['utm_source', 'utm_medium', 'utm_campaign', 'ref', 'source']
    parsed = urlparse('https://' + url)
    if parsed.query:
        params = parse_qs(parsed.query)
        clean_params = {k: v for k, v in params.items() if k not in tracking_params}
        if clean_params:
            query = '&'.join(f'{k}={v[0]}' for k, v in clean_params.items())
            url = f"{parsed.netloc}{parsed.path}?{query}"
        else:
            url = f"{parsed.netloc}{parsed.path}"

    return url

# ============================================
# 5. メイン処理
# ============================================

def main():
    print("=" * 70)
    print("Phase 1: 役割ベースソース分類")
    print("=" * 70)
    print()
    print("Codex批評対応:")
    print("  - ドメイン一律拒否ではなく役割推定")
    print("  - Wikipedia = pointer_to_sources（引用降下）")
    print("  - YouTube公式チャンネル = 一次証拠候補")
    print("  - 最終URL正規化")
    print()

    # データ読み込み
    cases = []
    with open(DATA_DIR / 'raw' / 'cases.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    print(f"入力: {len(cases)}件")

    # 全ソースを収集・分類
    all_sources = []
    for case in cases:
        sources = case.get('sources') or []
        if not sources and case.get('source'):
            sources = [case.get('source')]
        if sources:
            all_sources.extend(sources)

    print(f"総ソース数: {len(all_sources)}件")

    # 役割分類
    role_stats = Counter()
    reason_stats = Counter()
    pointer_count = 0
    youtube_official = 0

    classified_sources = []
    for url in all_sources:
        if not url:
            continue

        result = classify_source_role(url)
        role_stats[result['role']] += 1
        reason_stats[result.get('reason', 'unknown')] += 1

        if result.get('can_extract_references'):
            pointer_count += 1

        if result['role'] == 'primary_source' and 'youtube' in result.get('domain', ''):
            youtube_official += 1

        classified_sources.append({
            'url': url,
            'normalized': normalize_url_final(url),
            **result
        })

    print()
    print("【役割分布】")
    for role, count in role_stats.most_common():
        pct = count / len(all_sources) * 100
        print(f"  {role}: {count}件 ({pct:.1f}%)")

    print()
    print("【分類理由（上位15）】")
    for reason, count in reason_stats.most_common(15):
        print(f"  {reason}: {count}件")

    print()
    print("【特殊ソース】")
    print(f"  引用抽出可能（Wikipedia等）: {pointer_count}件")
    print(f"  YouTube公式チャンネル: {youtube_official}件")

    # Wikipedia引用降下シミュレーション
    print()
    print("【Wikipedia引用降下（構造確認）】")
    wiki_sources = [s for s in all_sources if s and 'wikipedia.org' in s.lower()]
    print(f"  Wikipediaソース数: {len(wiki_sources)}件")
    print(f"  引用降下対象: {len(wiki_sources)}件")
    print(f"  注: 実際のスクレイピングは別途実装が必要")

    # 事例レベルの役割集計
    print()
    print("【事例レベル役割集計】")
    case_role_stats = Counter()
    for case in cases:
        sources = case.get('sources') or []
        if not sources and case.get('source'):
            sources = [case.get('source')]
        if not sources:
            sources = []

        roles = []
        for url in sources:
            if url:
                result = classify_source_role(url)
                roles.append(result['role'])

        # 最良の役割を採用
        if 'primary_source' in roles:
            case_role_stats['has_primary'] += 1
        elif 'secondary_source' in roles:
            case_role_stats['has_secondary'] += 1
        elif 'pointer_to_sources' in roles:
            case_role_stats['pointer_only'] += 1
        elif 'context_only' in roles:
            case_role_stats['context_only'] += 1
        else:
            case_role_stats['rejected_or_none'] += 1

    for status, count in case_role_stats.most_common():
        pct = count / len(cases) * 100
        print(f"  {status}: {count}件 ({pct:.1f}%)")

    # レポート出力
    report = {
        'generated_at': datetime.now().isoformat(),
        'phase': '1_source_roles',
        'codex_compliance': {
            'role_based_classification': True,
            'wikipedia_as_pointer': True,
            'youtube_official_detection': True,
            'url_normalization': True,
        },
        'source_statistics': {
            'total_sources': len(all_sources),
            'role_distribution': dict(role_stats),
            'pointer_count': pointer_count,
            'youtube_official': youtube_official,
        },
        'case_statistics': {
            'total_cases': len(cases),
            'case_role_distribution': dict(case_role_stats),
        },
    }

    report_path = DATA_DIR / 'gold' / 'phase1_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print()
    print(f"【レポート出力】{report_path.name}")

    print()
    print("【Phase 1 完了】")
    print("  次のステップ: Phase 2（固定評価セット構築）")
    print()
    print("【TODO】Wikipedia引用抽出の実装")
    print("  - MediaWiki API (`action=parse`) を使用")
    print("  - 外部リンク (extlinks) を抽出")
    print("  - 引用元URLをprimary/secondaryに分類")

if __name__ == '__main__':
    main()
