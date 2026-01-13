"""
Phase 1 & 2: ゴールドセット抽出 + 汚染データ隔離
実行: python3 scripts/quality/phase1_2_extract.py
"""

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from collections import Counter

# 設定読み込み
from quality_config import (
    ANONYMOUS_PATTERNS, TRUSTED_DOMAINS, REJECTED_URL_PATTERNS,
    COUNTRY_NORMALIZATION, QUALITY_FLAGS
)

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'

def is_anonymous(target_name: str) -> bool:
    """匿名・半架空判定"""
    if not target_name:
        return True
    for pattern in ANONYMOUS_PATTERNS:
        if re.search(pattern, target_name):
            return True
    return False

def get_source_domain(url: str) -> str:
    """URLからドメイン抽出"""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except:
        return ''

def is_rejected_url(url: str) -> bool:
    """rejected URL判定（Google検索等）"""
    for pattern in REJECTED_URL_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return True
    return False

def is_trusted_domain(domain: str) -> tuple:
    """信頼ドメイン判定 → (is_trusted, category)"""
    for category, domains in TRUSTED_DOMAINS.items():
        for trusted in domains:
            if trusted in domain:
                return True, category
    return False, None

def get_sources(case: dict) -> list:
    """ソースリスト取得（sources / source両対応）"""
    sources = case.get('sources', [])
    if not sources:
        single = case.get('source')
        if single:
            sources = [single]
    return sources if sources else []

def evaluate_case(case: dict) -> dict:
    """事例の品質評価"""
    result = {
        'tier': None,  # gold / quarantine / standard
        'reason': [],
        'quality_flags': [],
        'source_analysis': {},
    }

    target_name = case.get('target_name') or case.get('entity_name', '')
    sources = get_sources(case)
    trust_level = case.get('trust_level')

    # 1. 匿名チェック
    if is_anonymous(target_name):
        result['tier'] = 'quarantine'
        result['reason'].append('anonymous_name')
        return result

    # 2. ソースチェック
    if not sources:
        result['tier'] = 'quarantine'
        result['reason'].append('no_source')
        return result

    # 3. rejected URLチェック
    rejected_sources = [s for s in sources if is_rejected_url(s)]
    if rejected_sources:
        if len(rejected_sources) == len(sources):
            result['tier'] = 'quarantine'
            result['reason'].append('google_url_only')
            return result
        else:
            result['quality_flags'].append('has_rejected_url')

    # 4. ドメイン分析
    valid_sources = [s for s in sources if not is_rejected_url(s)]
    trusted_count = 0
    domain_categories = []

    for src in valid_sources:
        domain = get_source_domain(src)
        is_trusted, category = is_trusted_domain(domain)
        if is_trusted:
            trusted_count += 1
            domain_categories.append(category)

    result['source_analysis'] = {
        'total': len(sources),
        'valid': len(valid_sources),
        'trusted': trusted_count,
        'categories': domain_categories,
    }

    # 5. 品質フラグ追加
    if len(valid_sources) == 1:
        result['quality_flags'].append('single_source')

    if domain_categories and all(c.startswith('news') for c in domain_categories):
        result['quality_flags'].append('news_only')

    # 6. tier判定
    if trust_level == 'verified' and trusted_count > 0:
        result['tier'] = 'gold'
        result['reason'].append('verified_with_trusted_source')
    elif trust_level in ['verified', 'plausible'] and len(valid_sources) > 0:
        result['tier'] = 'standard'
        result['reason'].append('plausible_with_source')
    elif len(valid_sources) > 0:
        result['tier'] = 'standard'
        result['reason'].append('has_valid_source')
    else:
        result['tier'] = 'quarantine'
        result['reason'].append('no_valid_source')

    return result

def ensure_transition_id(case: dict, generation_date: str) -> dict:
    """transition_id補完（生成メタ情報付き）"""
    if not case.get('transition_id'):
        new_id = str(uuid.uuid4())
        case['transition_id'] = new_id
        case['_transition_id_generated'] = {
            'date': generation_date,
            'reason': 'auto_generated_phase1_2',
        }
    return case

def main():
    print("=" * 60)
    print("Phase 1 & 2: ゴールドセット抽出 + 汚染データ隔離")
    print("=" * 60)

    generation_date = datetime.now().isoformat()

    # データ読み込み
    cases = []
    with open(DATA_DIR / 'raw' / 'cases.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    print(f"入力件数: {len(cases)}")

    # 分類
    gold_cases = []
    quarantine_cases = {
        'anonymous': [],
        'no_source': [],
        'google_url': [],
        'other': [],
    }
    standard_cases = []

    stats = Counter()

    for case in cases:
        # transition_id補完
        case = ensure_transition_id(case, generation_date)

        # 評価
        eval_result = evaluate_case(case)
        case['_quality_evaluation'] = eval_result

        tier = eval_result['tier']
        reasons = eval_result['reason']

        stats[tier] += 1

        if tier == 'gold':
            gold_cases.append(case)
        elif tier == 'quarantine':
            if 'anonymous_name' in reasons:
                quarantine_cases['anonymous'].append(case)
            elif 'no_source' in reasons:
                quarantine_cases['no_source'].append(case)
            elif 'google_url_only' in reasons:
                quarantine_cases['google_url'].append(case)
            else:
                quarantine_cases['other'].append(case)
        else:
            standard_cases.append(case)

    # 出力
    print()
    print("【分類結果】")
    print(f"  Gold (verified+信頼ソース): {len(gold_cases)}")
    print(f"  Standard (通常): {len(standard_cases)}")
    print(f"  Quarantine合計: {sum(len(v) for v in quarantine_cases.values())}")
    print(f"    - 匿名: {len(quarantine_cases['anonymous'])}")
    print(f"    - ソースなし: {len(quarantine_cases['no_source'])}")
    print(f"    - Google検索: {len(quarantine_cases['google_url'])}")
    print(f"    - その他: {len(quarantine_cases['other'])}")

    # ファイル出力
    def save_jsonl(data, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"  保存: {path} ({len(data)}件)")

    print()
    print("【ファイル出力】")
    save_jsonl(gold_cases, DATA_DIR / 'gold' / 'verified_cases.jsonl')
    save_jsonl(standard_cases, DATA_DIR / 'raw' / 'standard_cases.jsonl')
    save_jsonl(quarantine_cases['anonymous'], DATA_DIR / 'quarantine' / 'anonymous.jsonl')
    save_jsonl(quarantine_cases['no_source'], DATA_DIR / 'quarantine' / 'no_source.jsonl')
    save_jsonl(quarantine_cases['google_url'], DATA_DIR / 'quarantine' / 'google_url.jsonl')
    if quarantine_cases['other']:
        save_jsonl(quarantine_cases['other'], DATA_DIR / 'quarantine' / 'other.jsonl')

    # サマリーレポート
    report = {
        'generated_at': generation_date,
        'input_count': len(cases),
        'gold_count': len(gold_cases),
        'standard_count': len(standard_cases),
        'quarantine': {
            'anonymous': len(quarantine_cases['anonymous']),
            'no_source': len(quarantine_cases['no_source']),
            'google_url': len(quarantine_cases['google_url']),
            'other': len(quarantine_cases['other']),
        },
        'transition_id_generated': sum(1 for c in cases if c.get('_transition_id_generated')),
    }

    with open(DATA_DIR / 'gold' / 'extraction_report.json', 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print("完了！")
    print(f"  Gold比率: {len(gold_cases)/len(cases)*100:.1f}%")
    print(f"  Quarantine比率: {sum(len(v) for v in quarantine_cases.values())/len(cases)*100:.1f}%")

if __name__ == '__main__':
    main()
