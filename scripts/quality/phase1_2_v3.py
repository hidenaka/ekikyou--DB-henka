"""
Phase 1 & 2 v3: 正規化ルール適用 + 企業公式追加
"""

import json
import re
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict

from domain_rules import (
    normalize_url, extract_domain, is_rejected, classify_domain,
    GOLD_TIERS, SILVER_TIERS
)
from quality_config_v2 import ANONYMOUS_PATTERNS

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'

def is_anonymous(name: str) -> bool:
    if not name:
        return True
    for pattern in ANONYMOUS_PATTERNS:
        if re.search(pattern, str(name)):
            return True
    return False

def get_sources(case: dict) -> list:
    sources = case.get('sources', [])
    if not sources:
        s = case.get('source')
        if s:
            sources = [s]
    return sources or []

def evaluate_case_v3(case: dict) -> dict:
    """v3評価（正規化ルール適用）"""
    name = case.get('target_name') or case.get('entity_name', '')
    sources = get_sources(case)
    trust_level = case.get('trust_level')

    result = {
        'tier': None,
        'drop_reasons': [],
        'source_analysis': [],
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

    # 3. 各ソースを分類
    valid_sources = []
    best_tier = None
    best_tier_priority = 999

    TIER_PRIORITY = {
        'tier1_official': 1,
        'tier2_major_media': 2,
        'tier4_corporate': 3,
        'tier3_specialist': 4,
        'tier5_pr': 5,
        'unclassified': 10,
    }

    for src in sources:
        classification = classify_domain(src)
        result['source_analysis'].append(classification)

        if classification['tier'] == 'rejected':
            result['quality_flags'].append('has_rejected_url')
            continue

        valid_sources.append(src)
        tier = classification['tier']
        priority = TIER_PRIORITY.get(tier, 99)

        if priority < best_tier_priority:
            best_tier_priority = priority
            best_tier = tier

    if not valid_sources:
        result['tier'] = 'quarantine'
        result['drop_reasons'].append('only_rejected_urls')
        return result

    # 4. Tier判定
    if trust_level == 'verified' and best_tier in GOLD_TIERS:
        result['tier'] = 'gold'
    elif trust_level in ['verified', 'plausible'] and best_tier in SILVER_TIERS:
        result['tier'] = 'silver'
    elif valid_sources:
        result['tier'] = 'bronze'
    else:
        result['tier'] = 'quarantine'

    # 5. 品質フラグ
    if len(valid_sources) == 1:
        result['quality_flags'].append('single_source')

    return result

def main():
    print("=" * 70)
    print("Phase 1 & 2 v3: 正規化ルール + 企業公式追加")
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
    domain_coverage = Counter()

    for case in cases:
        eval_result = evaluate_case_v3(case)
        tier = eval_result['tier']
        tiers[tier].append(case)

        for reason in eval_result['drop_reasons']:
            drop_stats[reason] += 1

        outcome = case.get('outcome')
        if outcome:
            tier_outcomes[tier][outcome] += 1

        # ドメインカバレッジ
        for sa in eval_result['source_analysis']:
            if sa['tier'] not in ['rejected', 'unknown']:
                domain_coverage[sa['tier']] += 1

    # 結果出力
    print()
    print("【Tier分布（v3）】")
    for tier in ['gold', 'silver', 'bronze', 'quarantine']:
        count = len(tiers[tier])
        pct = count / len(cases) * 100
        print(f"  {tier:12}: {count:6}件 ({pct:5.1f}%)")

    print()
    print("【Quarantine理由】")
    for reason, count in drop_stats.most_common():
        print(f"  {reason}: {count}件")

    print()
    print("【ドメインTier分布（重複あり）】")
    for tier, count in domain_coverage.most_common():
        print(f"  {tier}: {count}件")

    print()
    print("【Tier別Outcome分布】")
    for tier in ['gold', 'silver', 'bronze']:
        outcomes = tier_outcomes[tier]
        total = sum(outcomes.values())
        if total == 0:
            continue
        success = outcomes.get('Success', 0)
        failure = outcomes.get('Failure', 0)
        ratio = success / failure if failure > 0 else float('inf')
        print(f"  {tier}: Success {success} / Failure {failure} (比 {ratio:.1f}:1)")

    # verified事例の分類先
    print()
    print("【verified事例の分類先】")
    verified_cases = [c for c in cases if c.get('trust_level') == 'verified']
    verified_dest = Counter()
    for case in verified_cases:
        eval_result = evaluate_case_v3(case)
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

    save_jsonl(tiers['gold'], DATA_DIR / 'gold' / 'gold_cases_v3.jsonl')
    save_jsonl(tiers['silver'], DATA_DIR / 'gold' / 'silver_cases_v3.jsonl')
    save_jsonl(tiers['bronze'], DATA_DIR / 'raw' / 'bronze_cases_v3.jsonl')
    save_jsonl(tiers['quarantine'], DATA_DIR / 'quarantine' / 'quarantine_v3.jsonl')

    # サマリーレポート
    report = {
        'generated_at': datetime.now().isoformat(),
        'version': 'v3',
        'input_count': len(cases),
        'tiers': {
            'gold': len(tiers['gold']),
            'silver': len(tiers['silver']),
            'bronze': len(tiers['bronze']),
            'quarantine': len(tiers['quarantine']),
        },
        'verified_distribution': dict(verified_dest),
        'outcome_by_tier': {tier: dict(outcomes) for tier, outcomes in tier_outcomes.items()},
    }

    report_path = DATA_DIR / 'gold' / 'extraction_report_v3.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print("【サマリー】")
    gold_silver = len(tiers['gold']) + len(tiers['silver'])
    print(f"  Gold+Silver (統計算出母数): {gold_silver}件 ({gold_silver/len(cases)*100:.1f}%)")
    print(f"  verified→Gold回収率: {verified_dest.get('gold',0)}/{len(verified_cases)} ({verified_dest.get('gold',0)/len(verified_cases)*100:.1f}%)")

if __name__ == '__main__':
    main()
