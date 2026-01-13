"""
Phase 1 & 2 v2: 三層分類（Gold/Silver/Bronze）+ 詳細診断
"""

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from collections import Counter, defaultdict

from quality_config_v2 import (
    TRUSTED_DOMAINS, GOLD_ELIGIBLE_TIERS, PLAUSIBLE_ELIGIBLE_TIERS,
    ANONYMOUS_PATTERNS, REJECTED_URL_PATTERNS, SUCCESS_LEVEL_RULES,
)

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'

def is_anonymous(name: str) -> bool:
    if not name:
        return True
    for pattern in ANONYMOUS_PATTERNS:
        if re.search(pattern, str(name)):
            return True
    return False

def is_rejected_url(url: str) -> bool:
    for pattern in REJECTED_URL_PATTERNS:
        if re.search(pattern, str(url), re.IGNORECASE):
            return True
    return False

def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except:
        return ''

def classify_domain(domain: str) -> tuple:
    """ドメインをTier分類 → (tier_name, is_trusted)"""
    for tier, domains in TRUSTED_DOMAINS.items():
        for d in domains:
            if d in domain:
                return tier, True
    return None, False

def get_sources(case: dict) -> list:
    sources = case.get('sources', [])
    if not sources:
        s = case.get('source')
        if s:
            sources = [s]
    return sources or []

def evaluate_case_v2(case: dict) -> dict:
    """詳細評価"""
    name = case.get('target_name') or case.get('entity_name', '')
    sources = get_sources(case)
    trust_level = case.get('trust_level')

    result = {
        'tier': None,
        'drop_reasons': [],
        'source_tiers': [],
        'quality_flags': [],
    }

    # 1. 匿名チェック
    if is_anonymous(name):
        result['tier'] = 'quarantine'
        result['drop_reasons'].append('anonymous')
        return result

    # 2. ソースなしチェック
    if not sources:
        result['tier'] = 'quarantine'
        result['drop_reasons'].append('no_source')
        return result

    # 3. ソース分析
    valid_sources = []
    for src in sources:
        if is_rejected_url(src):
            result['quality_flags'].append(f'rejected_url:{src[:50]}')
            continue
        domain = get_domain(src)
        tier, is_trusted = classify_domain(domain)
        valid_sources.append({'url': src, 'domain': domain, 'tier': tier})
        if tier:
            result['source_tiers'].append(tier)

    if not valid_sources:
        result['tier'] = 'quarantine'
        result['drop_reasons'].append('only_rejected_urls')
        return result

    # 4. Tier判定
    has_gold_source = any(t in GOLD_ELIGIBLE_TIERS for t in result['source_tiers'])
    has_plausible_source = any(t in PLAUSIBLE_ELIGIBLE_TIERS for t in result['source_tiers'])

    if trust_level == 'verified' and has_gold_source:
        result['tier'] = 'gold'
    elif trust_level in ['verified', 'plausible'] and has_plausible_source:
        result['tier'] = 'silver'
    elif valid_sources:
        result['tier'] = 'bronze'
    else:
        result['tier'] = 'quarantine'
        result['drop_reasons'].append('unknown')

    # 5. 品質フラグ
    if len(valid_sources) == 1:
        result['quality_flags'].append('single_source')

    return result

def main():
    print("=" * 70)
    print("Phase 1 & 2 v2: 三層分類 + 詳細診断")
    print("=" * 70)

    cases = []
    with open(DATA_DIR / 'raw' / 'cases.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    print(f"入力: {len(cases)}件")

    # 分類
    tiers = defaultdict(list)
    drop_stats = Counter()
    tier_outcomes = defaultdict(Counter)

    for case in cases:
        eval_result = evaluate_case_v2(case)
        tier = eval_result['tier']
        tiers[tier].append(case)

        for reason in eval_result['drop_reasons']:
            drop_stats[reason] += 1

        outcome = case.get('outcome')
        if outcome:
            tier_outcomes[tier][outcome] += 1

    # 結果出力
    print()
    print("【Tier分布】")
    for tier in ['gold', 'silver', 'bronze', 'quarantine']:
        count = len(tiers[tier])
        pct = count / len(cases) * 100
        print(f"  {tier:12}: {count:6}件 ({pct:5.1f}%)")

    print()
    print("【Quarantine理由】")
    for reason, count in drop_stats.most_common():
        print(f"  {reason}: {count}件")

    print()
    print("【Tier別Outcome分布】")
    for tier in ['gold', 'silver', 'bronze']:
        outcomes = tier_outcomes[tier]
        total = sum(outcomes.values())
        if total == 0:
            continue
        print(f"  {tier}:")
        for outcome in ['Success', 'Mixed', 'Failure', 'PartialSuccess']:
            count = outcomes.get(outcome, 0)
            pct = count / total * 100 if total > 0 else 0
            print(f"    {outcome}: {count} ({pct:.1f}%)")

    # verified事例の詳細分析
    print()
    print("【verified事例の分類先】")
    verified_cases = [c for c in cases if c.get('trust_level') == 'verified']
    verified_dest = Counter()
    for case in verified_cases:
        eval_result = evaluate_case_v2(case)
        verified_dest[eval_result['tier']] += 1
    for tier, count in verified_dest.most_common():
        print(f"  → {tier}: {count}件")

    # ファイル出力
    print()
    print("【ファイル出力】")

    def save_jsonl(data, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"  {path.name}: {len(data)}件")

    save_jsonl(tiers['gold'], DATA_DIR / 'gold' / 'gold_cases.jsonl')
    save_jsonl(tiers['silver'], DATA_DIR / 'gold' / 'silver_cases.jsonl')
    save_jsonl(tiers['bronze'], DATA_DIR / 'raw' / 'bronze_cases.jsonl')
    save_jsonl(tiers['quarantine'], DATA_DIR / 'quarantine' / 'all_quarantine.jsonl')

    # サマリー
    print()
    print("【推奨アクション】")
    print(f"  1. Gold {len(tiers['gold'])}件を成功率算出のコアに使用")
    print(f"  2. Silver {len(tiers['silver'])}件を追加検証後Goldに昇格検討")
    print(f"  3. Quarantine {len(tiers['quarantine'])}件のうち実ソース化可能なものを選別")

if __name__ == '__main__':
    main()
