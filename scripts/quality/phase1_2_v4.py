"""
Phase 1 & 2 v4: Codex批評5点対応版
- Gold二階建て（Verified/Verifiable）
- COI手動レジストリ統合
- 証拠独立性チェック
"""

import json
import re
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
from urllib.parse import urlparse

from domain_rules import (
    normalize_url, extract_domain, is_rejected, classify_domain,
    GOLD_TIERS, SILVER_TIERS
)
from quality_config_v2 import ANONYMOUS_PATTERNS
from coi_registry import evaluate_coi, get_domain_owner

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'

# ============================================
# 1. 証拠独立性チェック
# ============================================

# 同一ソース（提携配信・シンジケーション）パターン
SYNDICATION_GROUPS = {
    # 通信社→各紙
    'kyodo': ['kyodo.co.jp', '47news.jp'],
    'jiji': ['jiji.com'],
    'reuters': ['reuters.com', 'jp.reuters.com'],
    'ap': ['apnews.com'],
    # PRワイヤー
    'pr_wire': ['prtimes.jp', 'businesswire.com', 'prnewswire.com', 'globenewswire.com'],
}

def get_syndication_group(domain: str) -> str | None:
    """ドメインが属するシンジケーショングループを取得"""
    for group, domains in SYNDICATION_GROUPS.items():
        for d in domains:
            if d in domain:
                return group
    return None

def check_evidence_independence(sources: list[str]) -> dict:
    """
    証拠独立性チェック
    同一ソース（提携配信等）からの重複を検出

    Returns:
        {
            'is_independent': bool,
            'unique_sources': int,
            'syndication_overlap': list,
            'same_domain_count': int,
        }
    """
    if not sources:
        return {
            'is_independent': False,
            'unique_sources': 0,
            'syndication_overlap': [],
            'same_domain_count': 0,
        }

    domains = [extract_domain(s) for s in sources]
    unique_domains = set(domains)

    # シンジケーショングループ重複チェック
    groups_seen = {}
    syndication_overlap = []

    for domain in domains:
        group = get_syndication_group(domain)
        if group:
            if group in groups_seen:
                syndication_overlap.append({
                    'group': group,
                    'domains': [groups_seen[group], domain]
                })
            else:
                groups_seen[group] = domain

    # 同一ドメインカウント
    domain_counts = Counter(domains)
    same_domain_count = sum(1 for c in domain_counts.values() if c > 1)

    # 独立性判定: 重複なし かつ ユニークドメインが2以上
    is_independent = (
        len(syndication_overlap) == 0 and
        same_domain_count == 0 and
        len(unique_domains) >= 2
    )

    return {
        'is_independent': is_independent,
        'unique_sources': len(unique_domains),
        'syndication_overlap': syndication_overlap,
        'same_domain_count': same_domain_count,
    }

# ============================================
# 2. 検証状態判定
# ============================================

def determine_verification_status(case: dict, sources: list[str]) -> str:
    """
    検証状態を判定
    - verified: 検証済み（trust_level=verified かつ 有効ソースあり）
    - verifiable: 検証可能だが未検証
    - unverifiable: 検証不能（paywall等）
    """
    trust_level = case.get('trust_level')

    # trust_level=verifiedは検証済み
    if trust_level == 'verified':
        return 'verified'

    # ソースがあれば検証可能
    if sources:
        # paywall/アーカイブ判定（簡易）
        paywall_patterns = [
            r'wsj\.com',  # Wall Street Journal
            r'ft\.com',   # Financial Times
            r'nikkei\.com/.*/DGXZ',  # 日経有料記事
        ]
        all_paywall = all(
            any(re.search(p, s) for p in paywall_patterns)
            for s in sources
        )
        if all_paywall:
            return 'unverifiable'

        return 'verifiable'

    return 'unverifiable'

# ============================================
# 3. 品質分類関数
# ============================================

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

def evaluate_case_v4(case: dict) -> dict:
    """
    v4評価（Codex批評5点対応）

    新分類:
    - gold_verified: 検証済み + 高信頼ソース + COI=none
    - gold_verifiable: 検証可能 + 高信頼ソース + COI=none
    - silver: 中信頼ソース or COI=self(事実のみ)
    - bronze: 低信頼ソース
    - quarantine: 匿名/ソースなし/rejected
    """
    name = case.get('target_name') or case.get('entity_name', '')
    sources = get_sources(case)

    result = {
        'tier': None,
        'sub_tier': None,  # gold_verified / gold_verifiable
        'drop_reasons': [],
        'source_analysis': [],
        'coi_analysis': [],
        'evidence_independence': None,
        'verification_status': None,
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

    # 3. 各ソースを分類 + COI評価
    valid_sources = []
    best_tier = None
    best_tier_priority = 999
    has_coi_self = False
    coi_self_count = 0

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

        # COI評価（tier4_corporateの場合）
        if tier == 'tier4_corporate':
            coi_result = evaluate_coi(src, name)
            result['coi_analysis'].append(coi_result)
            if coi_result['coi_type'] == 'self':
                has_coi_self = True
                coi_self_count += 1

    if not valid_sources:
        result['tier'] = 'quarantine'
        result['drop_reasons'].append('only_rejected_urls')
        return result

    # 4. 証拠独立性チェック
    evidence_result = check_evidence_independence(valid_sources)
    result['evidence_independence'] = evidence_result

    if evidence_result['syndication_overlap']:
        result['quality_flags'].append('syndication_overlap')

    # 5. 検証状態
    verification_status = determine_verification_status(case, valid_sources)
    result['verification_status'] = verification_status

    # 6. Tier判定（Codex批評対応）
    trust_level = case.get('trust_level')

    # Gold判定条件（厳格化）
    # - Verified: trust_level=verified + 高信頼ソース + COI=none
    # - Verifiable: 検証可能 + 高信頼ソース + COI=none（ただしGoldではない）
    if best_tier in GOLD_TIERS:
        # COI=selfの場合、tier4_corporateのみSilverに降格
        if has_coi_self and best_tier == 'tier4_corporate':
            # 全ソースがCOI=selfならSilverに降格
            non_coi_sources = len(valid_sources) - coi_self_count
            if non_coi_sources == 0:
                result['tier'] = 'silver'
                result['drop_reasons'].append('coi_self_only')
            elif trust_level == 'verified':
                result['tier'] = 'gold'
                result['sub_tier'] = 'gold_verified'
                result['quality_flags'].append('partial_coi_self')
            else:
                result['tier'] = 'silver'
        elif trust_level == 'verified':
            result['tier'] = 'gold'
            result['sub_tier'] = 'gold_verified'
        elif verification_status == 'verifiable':
            # Codex指摘: verifiable（未検証）はGoldではなくSilver
            result['tier'] = 'silver'
            result['sub_tier'] = 'gold_verifiable'  # 記録用
        else:
            result['tier'] = 'silver'

    elif best_tier in SILVER_TIERS:
        if trust_level in ['verified', 'plausible']:
            result['tier'] = 'silver'
        else:
            result['tier'] = 'bronze'

    elif valid_sources:
        result['tier'] = 'bronze'

    else:
        result['tier'] = 'quarantine'

    # 7. 追加品質フラグ
    if len(valid_sources) == 1:
        result['quality_flags'].append('single_source')

    if not evidence_result['is_independent'] and len(valid_sources) > 1:
        result['quality_flags'].append('non_independent_sources')

    return result

# ============================================
# 4. メイン処理
# ============================================

def main():
    print("=" * 70)
    print("Phase 1 & 2 v4: Codex批評5点対応版")
    print("=" * 70)
    print()
    print("変更点:")
    print("  1. Gold二階建て（Verified/Verifiable分離）")
    print("  2. COI手動レジストリ統合")
    print("  3. 証拠独立性チェック")
    print("  4. Verifiable（未検証）はGoldからSilverへ降格")
    print()

    cases = []
    with open(DATA_DIR / 'raw' / 'cases.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    print(f"入力: {len(cases)}件")

    # 分類
    tiers = defaultdict(list)
    sub_tiers = defaultdict(list)
    drop_stats = Counter()
    tier_outcomes = defaultdict(Counter)
    coi_stats = Counter()
    evidence_stats = Counter()

    for case in cases:
        eval_result = evaluate_case_v4(case)
        tier = eval_result['tier']
        sub_tier = eval_result.get('sub_tier')

        tiers[tier].append(case)
        if sub_tier:
            sub_tiers[sub_tier].append(case)

        for reason in eval_result['drop_reasons']:
            drop_stats[reason] += 1

        outcome = case.get('outcome')
        if outcome:
            tier_outcomes[tier][outcome] += 1

        # COI統計
        for coi in eval_result.get('coi_analysis', []):
            coi_stats[coi['coi_type']] += 1

        # 証拠独立性統計
        ev = eval_result.get('evidence_independence')
        if ev:
            if ev['is_independent']:
                evidence_stats['independent'] += 1
            elif ev['syndication_overlap']:
                evidence_stats['syndication_overlap'] += 1
            elif ev['same_domain_count'] > 0:
                evidence_stats['same_domain'] += 1
            else:
                evidence_stats['single_source'] += 1

    # 結果出力
    print()
    print("【Tier分布（v4）】")
    for tier in ['gold', 'silver', 'bronze', 'quarantine']:
        count = len(tiers[tier])
        pct = count / len(cases) * 100
        print(f"  {tier:12}: {count:6}件 ({pct:5.1f}%)")

    print()
    print("【Gold内訳（二階建て）】")
    print(f"  gold_verified:   {len(sub_tiers.get('gold_verified', []))}件")
    print(f"  gold_verifiable: {len(sub_tiers.get('gold_verifiable', []))}件 (→Silver扱い)")

    print()
    print("【COI分析結果】")
    for coi_type, count in coi_stats.most_common():
        print(f"  {coi_type}: {count}件")

    print()
    print("【証拠独立性】")
    for status, count in evidence_stats.most_common():
        print(f"  {status}: {count}件")

    print()
    print("【Quarantine/降格理由】")
    for reason, count in drop_stats.most_common():
        print(f"  {reason}: {count}件")

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
        eval_result = evaluate_case_v4(case)
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

    save_jsonl(tiers['gold'], DATA_DIR / 'gold' / 'gold_cases_v4.jsonl')
    save_jsonl(tiers['silver'], DATA_DIR / 'gold' / 'silver_cases_v4.jsonl')
    save_jsonl(tiers['bronze'], DATA_DIR / 'raw' / 'bronze_cases_v4.jsonl')
    save_jsonl(tiers['quarantine'], DATA_DIR / 'quarantine' / 'quarantine_v4.jsonl')

    # サマリーレポート
    report = {
        'generated_at': datetime.now().isoformat(),
        'version': 'v4',
        'codex_compliance': {
            'gold_two_tier': True,
            'coi_registry': True,
            'evidence_independence': True,
            'verifiable_to_silver': True,
        },
        'input_count': len(cases),
        'tiers': {
            'gold': len(tiers['gold']),
            'silver': len(tiers['silver']),
            'bronze': len(tiers['bronze']),
            'quarantine': len(tiers['quarantine']),
        },
        'sub_tiers': {
            'gold_verified': len(sub_tiers.get('gold_verified', [])),
            'gold_verifiable': len(sub_tiers.get('gold_verifiable', [])),
        },
        'coi_stats': dict(coi_stats),
        'evidence_stats': dict(evidence_stats),
        'verified_distribution': dict(verified_dest),
        'outcome_by_tier': {tier: dict(outcomes) for tier, outcomes in tier_outcomes.items()},
    }

    report_path = DATA_DIR / 'gold' / 'extraction_report_v4.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print("【v4サマリー】")
    print(f"  Gold (verified only): {len(tiers['gold'])}件")
    print(f"  Silver (含verifiable): {len(tiers['silver'])}件")
    gold_silver = len(tiers['gold']) + len(tiers['silver'])
    print(f"  Gold+Silver (統計算出母数): {gold_silver}件 ({gold_silver/len(cases)*100:.1f}%)")
    print(f"  verified→Gold回収率: {verified_dest.get('gold',0)}/{len(verified_cases)} ({verified_dest.get('gold',0)/len(verified_cases)*100:.1f}%)")

    print()
    print("【Codex批評対応状況】")
    print("  [x] Gold二階建て化 - Verifiedのみgold、VerifiableはSilver")
    print("  [x] COI手動レジストリ - tier4_corporateの自社報道検出")
    print("  [x] 証拠独立性チェック - シンジケーション重複検出")
    print("  [ ] 統計KPI変更 - 校正・スコア評価は別途実装")
    print("  [ ] 主張単位評価 - 現行は事例単位を維持")

if __name__ == '__main__':
    main()
