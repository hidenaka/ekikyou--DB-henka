#!/usr/bin/env python3
"""
apply_schema_v2.py

全事例（12,871件）に新スキーマ（outcome_status, verification_confidence, coi_status）を適用する。

Usage:
    python3 scripts/quality/apply_schema_v2.py

新スキーマフィールド:
1. outcome_status: verified_correct / verified_incorrect / unverified
2. verification_confidence: high / medium / low / none
3. coi_status: self / none / unknown
"""

import json
import re
import sys
import os
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional, Tuple

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "cases.jsonl"
BACKUP_DIR = PROJECT_ROOT / "data" / "backup"

# ============================================
# ドメイン分類定義
# ============================================

# 一次ソース（政府・学術・公式IR）
PRIMARY_SOURCE_PATTERNS = [
    # 政府系
    r'\.go\.jp', r'\.gov$', r'\.gov\.', r'gouv\.fr',
    r'bundesregierung\.de', r'ec\.europa\.eu',
    # 学術系
    r'\.ac\.jp', r'\.edu$', r'\.ac\.uk',
    # 国際機関
    r'imf\.org', r'worldbank\.org', r'un\.org', r'who\.int', r'oecd\.org',
    # 中央銀行
    r'boj\.or\.jp', r'federalreserve\.gov', r'ecb\.europa\.eu',
    # 規制当局
    r'sec\.gov', r'fsa\.go\.jp', r'fca\.org\.uk',
    # 企業IR (明示的なIRパス)
    r'/ir/', r'/investor',
]

# 二次ソース（報道機関）
SECONDARY_SOURCE_PATTERNS = [
    # 日本メディア
    r'nikkei\.com', r'asahi\.com', r'yomiuri\.co\.jp', r'mainichi\.jp',
    r'sankei\.com', r'nhk\.or\.jp', r'jiji\.com', r'kyodo\.co\.jp',
    # 海外メディア
    r'reuters\.com', r'bloomberg\.com', r'wsj\.com', r'nytimes\.com',
    r'ft\.com', r'economist\.com', r'bbc\.(com|co\.uk)', r'cnn\.com',
    r'apnews\.com', r'theguardian\.com',
    # ビジネス・テック専門メディア
    r'toyokeizai\.net', r'diamond\.jp', r'president\.jp', r'forbesjapan\.com',
    r'techcrunch\.com', r'theverge\.com', r'wired\.(com|jp)',
    r'itmedia\.co\.jp', r'impress\.co\.jp', r'businessinsider',
]

# pointer_only（Wikipedia等）
POINTER_ONLY_PATTERNS = [
    r'wikipedia\.org', r'britannica\.com',
]

# 検索URL・無効URL
REJECTED_PATTERNS = [
    r'google\.(com|co\.jp)/search', r'bing\.com/search',
    r'yahoo\.(com|co\.jp)/search', r'duckduckgo\.com/\?q=',
    r'webcache\.googleusercontent\.com',
]

# 第三者（政府・報道機関） → coi_status=none
THIRD_PARTY_PATTERNS = [
    # 政府系
    r'\.go\.jp', r'\.gov$', r'\.gov\.', r'gouv\.fr',
    r'bundesregierung\.de', r'ec\.europa\.eu',
    # 国際機関
    r'imf\.org', r'worldbank\.org', r'un\.org', r'who\.int', r'oecd\.org',
    # 報道機関（日本）
    r'nikkei\.com', r'asahi\.com', r'yomiuri\.co\.jp', r'mainichi\.jp',
    r'sankei\.com', r'nhk\.or\.jp', r'jiji\.com', r'kyodo\.co\.jp',
    # 報道機関（海外）
    r'reuters\.com', r'bloomberg\.com', r'wsj\.com', r'nytimes\.com',
    r'ft\.com', r'economist\.com', r'bbc\.(com|co\.uk)', r'cnn\.com',
    r'apnews\.com', r'theguardian\.com',
]

# 企業名パターン（動的生成用）
COMPANY_DOMAIN_KEYWORDS = [
    'toyota', 'honda', 'nissan', 'sony', 'panasonic', 'hitachi', 'toshiba',
    'sharp', 'nec', 'fujitsu', 'canon', 'nikon', 'nintendo', 'softbank',
    'rakuten', 'mercari', 'line', 'apple', 'google', 'amazon', 'meta',
    'microsoft', 'nvidia', 'tesla', 'netflix', 'openai', 'spacex',
    'airbnb', 'uber', 'zoom', 'samsung', 'alibaba', 'tencent', 'huawei',
    'siemens', 'volkswagen', 'bmw', 'daimler', 'nestle', 'lvmh',
    'mcdonalds', 'starbucks', 'walmart', 'costco', 'jal', 'ana',
    'aeon', 'seven', 'uniqlo', 'fastretailing',
]

# ============================================
# 判定関数
# ============================================

def is_rejected_url(url: str) -> bool:
    """無効URLかどうか判定"""
    if not url:
        return True
    url_lower = url.lower()
    return any(re.search(p, url_lower) for p in REJECTED_PATTERNS)


def is_primary_source(url: str) -> bool:
    """一次ソースかどうか判定"""
    if not url:
        return False
    url_lower = url.lower()
    return any(re.search(p, url_lower) for p in PRIMARY_SOURCE_PATTERNS)


def is_secondary_source(url: str) -> bool:
    """二次ソースかどうか判定"""
    if not url:
        return False
    url_lower = url.lower()
    return any(re.search(p, url_lower) for p in SECONDARY_SOURCE_PATTERNS)


def is_pointer_only(url: str) -> bool:
    """pointer_only（Wikipedia）かどうか判定"""
    if not url:
        return False
    url_lower = url.lower()
    return any(re.search(p, url_lower) for p in POINTER_ONLY_PATTERNS)


def is_third_party(url: str) -> bool:
    """第三者（政府・報道）かどうか判定"""
    if not url:
        return False
    url_lower = url.lower()
    return any(re.search(p, url_lower) for p in THIRD_PARTY_PATTERNS)


def extract_domain(url: str) -> str:
    """URLからドメインを抽出"""
    if not url:
        return ''
    url = url.lower().replace('https://', '').replace('http://', '')
    url = url.replace('www.', '')
    return url.split('/')[0].split(':')[0]


def is_self_reported(url: str, target_name: str) -> bool:
    """
    自己報告（企業公式サイト）かどうか判定
    ドメインが対象企業名を含むかどうかで判定
    """
    if not url or not target_name:
        return False

    domain = extract_domain(url)
    target_lower = target_name.lower()

    # 企業名の主要部分を抽出（括弧内を除去）
    target_clean = re.sub(r'[（(].+[)）]', '', target_lower).strip()

    # 主要な単語を抽出
    target_words = re.findall(r'[a-zA-Z]{3,}', target_clean)

    # ドメインに企業名のキーワードが含まれるか
    for word in target_words:
        if word in domain:
            return True

    # 既知の企業ドメインキーワードとマッチするか
    for keyword in COMPANY_DOMAIN_KEYWORDS:
        if keyword in domain and keyword in target_lower:
            return True

    return False


def get_sources_from_case(case: dict) -> List[str]:
    """事例からソースURLリストを取得"""
    sources = case.get('sources', [])
    if sources is None:
        return []
    if isinstance(sources, str):
        return [sources] if sources else []
    return [s for s in sources if s and isinstance(s, str)]


def determine_verification_confidence(sources: List[str]) -> str:
    """
    verification_confidence を判定

    - 一次ソース（.go.jp, .ac.jp, 公式IR）あり → high
    - 二次ソース（nikkei.com, reuters等）あり → medium
    - pointer_only（Wikipedia）のみ → low
    - ソースなし/不明 → none
    """
    if not sources:
        return 'none'

    valid_sources = [s for s in sources if not is_rejected_url(s)]
    if not valid_sources:
        return 'none'

    # 一次ソースがあれば high
    if any(is_primary_source(s) for s in valid_sources):
        return 'high'

    # 二次ソースがあれば medium
    if any(is_secondary_source(s) for s in valid_sources):
        return 'medium'

    # pointer_onlyのみなら low
    if any(is_pointer_only(s) for s in valid_sources):
        return 'low'

    # それ以外のソースがあっても low
    if valid_sources:
        return 'low'

    return 'none'


def determine_outcome_status(case: dict, verification_confidence: str) -> str:
    """
    outcome_status を判定

    - trust_level='verified' かつ outcome が明確 → verified_correct
    - trust_level='verified' でない → unverified
    """
    trust_level = case.get('trust_level', '')
    outcome = case.get('outcome', '')

    # verified + 明確なoutcome → verified_correct
    if trust_level == 'verified' and outcome in ['Success', 'Failure', 'Mixed', 'PartialSuccess']:
        # verification_confidence が high/medium であることも確認
        if verification_confidence in ['high', 'medium']:
            return 'verified_correct'

    # それ以外は unverified
    return 'unverified'


def determine_coi_status(sources: List[str], target_name: str) -> str:
    """
    coi_status を判定

    - ソースなし → unknown
    - ドメインが対象企業名を含む → self
    - 政府/報道機関 → none
    - その他 → unknown
    """
    if not sources:
        return 'unknown'

    valid_sources = [s for s in sources if not is_rejected_url(s)]
    if not valid_sources:
        return 'unknown'

    # 全ソースを確認して最も厳しい（self優先）ものを返す
    has_self = False
    has_third_party = False

    for url in valid_sources:
        if is_self_reported(url, target_name):
            has_self = True
        elif is_third_party(url):
            has_third_party = True

    # selfが1つでもあれば self（利益相反あり）
    if has_self:
        return 'self'

    # 第三者のみであれば none（利益相反なし）
    if has_third_party:
        return 'none'

    # それ以外は unknown
    return 'unknown'


def apply_new_schema(case: dict) -> dict:
    """事例に新スキーマを適用"""
    sources = get_sources_from_case(case)
    target_name = case.get('target_name', '')

    # verification_confidence を先に計算（outcome_status判定に使用）
    verification_confidence = determine_verification_confidence(sources)

    # 各フィールドを設定
    case['verification_confidence'] = verification_confidence
    case['outcome_status'] = determine_outcome_status(case, verification_confidence)
    case['coi_status'] = determine_coi_status(sources, target_name)

    return case


# ============================================
# メイン処理
# ============================================

def main():
    print("=" * 60)
    print("新スキーマ適用スクリプト v2")
    print(f"実行日時: {datetime.now().isoformat()}")
    print("=" * 60)

    # ファイル存在確認
    if not DATA_FILE.exists():
        print(f"Error: データファイルが見つかりません: {DATA_FILE}")
        sys.exit(1)

    # バックアップ作成
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_file = BACKUP_DIR / f"cases_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    print(f"\n[1] バックアップ作成: {backup_file}")

    # データ読み込み
    print(f"\n[2] データ読み込み: {DATA_FILE}")
    cases = []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  Warning: JSON解析エラー - {e}")

    total_count = len(cases)
    print(f"  読み込み件数: {total_count:,}")

    # バックアップ書き込み
    with open(backup_file, 'w', encoding='utf-8') as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')
    print(f"  バックアップ完了: {backup_file.name}")

    # 新スキーマ適用
    print(f"\n[3] 新スキーマ適用中...")

    # 統計カウンター
    outcome_status_counter = Counter()
    verification_confidence_counter = Counter()
    coi_status_counter = Counter()

    updated_cases = []
    for i, case in enumerate(cases, 1):
        updated_case = apply_new_schema(case)
        updated_cases.append(updated_case)

        # 統計収集
        outcome_status_counter[updated_case.get('outcome_status', 'unknown')] += 1
        verification_confidence_counter[updated_case.get('verification_confidence', 'unknown')] += 1
        coi_status_counter[updated_case.get('coi_status', 'unknown')] += 1

        if i % 2000 == 0:
            print(f"  処理中... {i:,} / {total_count:,}")

    print(f"  処理完了: {total_count:,} 件")

    # 更新データ書き込み
    print(f"\n[4] 更新データ書き込み: {DATA_FILE}")
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        for case in updated_cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')
    print(f"  書き込み完了")

    # 統計レポート
    print("\n" + "=" * 60)
    print("統計レポート")
    print("=" * 60)

    print(f"\n[outcome_status 分布]")
    for status, count in sorted(outcome_status_counter.items(), key=lambda x: -x[1]):
        pct = count / total_count * 100
        print(f"  {status}: {count:,} ({pct:.1f}%)")

    print(f"\n[verification_confidence 分布]")
    for conf, count in sorted(verification_confidence_counter.items(), key=lambda x: -x[1]):
        pct = count / total_count * 100
        print(f"  {conf}: {count:,} ({pct:.1f}%)")

    print(f"\n[coi_status 分布]")
    for status, count in sorted(coi_status_counter.items(), key=lambda x: -x[1]):
        pct = count / total_count * 100
        print(f"  {status}: {count:,} ({pct:.1f}%)")

    # Unknown率
    unverified_count = outcome_status_counter.get('unverified', 0)
    unknown_rate = unverified_count / total_count * 100
    print(f"\n[Unknown率（= unverified / 全件）]")
    print(f"  {unverified_count:,} / {total_count:,} = {unknown_rate:.1f}%")

    # 品質サマリー
    verified_correct = outcome_status_counter.get('verified_correct', 0)
    high_conf = verification_confidence_counter.get('high', 0)
    medium_conf = verification_confidence_counter.get('medium', 0)

    print(f"\n[品質サマリー]")
    print(f"  検証済み正確（verified_correct）: {verified_correct:,} ({verified_correct/total_count*100:.1f}%)")
    print(f"  高確信度（high）: {high_conf:,} ({high_conf/total_count*100:.1f}%)")
    print(f"  中確信度（medium）: {medium_conf:,} ({medium_conf/total_count*100:.1f}%)")
    print(f"  利益相反なし（coi=none）: {coi_status_counter.get('none', 0):,}")
    print(f"  自己報告（coi=self）: {coi_status_counter.get('self', 0):,}")

    print("\n" + "=" * 60)
    print("処理完了")
    print("=" * 60)

    return {
        'total': total_count,
        'outcome_status': dict(outcome_status_counter),
        'verification_confidence': dict(verification_confidence_counter),
        'coi_status': dict(coi_status_counter),
        'unknown_rate': unknown_rate,
    }


if __name__ == '__main__':
    main()
