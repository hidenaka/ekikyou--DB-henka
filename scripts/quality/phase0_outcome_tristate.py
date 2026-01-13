"""
Phase 0: Outcome三値化
Codex批評に基づき、unverifiedをSuccess扱いせず、Unknownとして分離

設計原則:
- outcome: Success/Failure/Mixed/PartialSuccess（事象の結果）
- outcome_status: verified/unverified（結果の検証状態）
- 統計計算: Verified subsetのみで算出
- Unknown率: 新KPIとして追加
"""

import json
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'

def determine_outcome_status(case: dict) -> str:
    """
    outcome_status判定
    - verified: trust_level=verifiedの場合
    - unverified: それ以外
    """
    trust_level = case.get('trust_level')
    if trust_level == 'verified':
        return 'verified'
    return 'unverified'

def add_outcome_status(cases: list) -> list:
    """全事例にoutcome_statusを付与"""
    for case in cases:
        case['outcome_status'] = determine_outcome_status(case)
    return cases

def calculate_verified_statistics(cases: list) -> dict:
    """
    Verified subsetのみで統計を計算
    Unknown率を新KPIとして追加
    """
    total = len(cases)
    verified_cases = [c for c in cases if c.get('outcome_status') == 'verified']
    unverified_cases = [c for c in cases if c.get('outcome_status') == 'unverified']

    # Unknown率（新KPI）
    unknown_rate = len(unverified_cases) / total if total > 0 else 0

    # Verified subsetのoutcome分布
    verified_outcomes = Counter(c.get('outcome') for c in verified_cases)

    # 成功率（Verified subsetのみ）
    verified_success = verified_outcomes.get('Success', 0)
    verified_failure = verified_outcomes.get('Failure', 0)
    verified_total_sf = verified_success + verified_failure

    success_rate_verified = (
        verified_success / verified_total_sf
        if verified_total_sf > 0 else None
    )

    return {
        'total_cases': total,
        'verified_count': len(verified_cases),
        'unverified_count': len(unverified_cases),
        'unknown_rate': unknown_rate,
        'verified_outcomes': dict(verified_outcomes),
        'success_rate_verified': success_rate_verified,
        'success_failure_ratio_verified': (
            verified_success / verified_failure
            if verified_failure > 0 else float('inf')
        ),
    }

def calculate_statistics_by_tier(cases: list) -> dict:
    """Tier別のVerified統計"""
    # v4分類を再利用
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from phase1_2_v4 import evaluate_case_v4

    tier_stats = defaultdict(lambda: {
        'total': 0,
        'verified': 0,
        'unverified': 0,
        'verified_success': 0,
        'verified_failure': 0,
    })

    for case in cases:
        eval_result = evaluate_case_v4(case)
        tier = eval_result['tier']
        outcome_status = case.get('outcome_status', 'unverified')
        outcome = case.get('outcome')

        tier_stats[tier]['total'] += 1

        if outcome_status == 'verified':
            tier_stats[tier]['verified'] += 1
            if outcome == 'Success':
                tier_stats[tier]['verified_success'] += 1
            elif outcome == 'Failure':
                tier_stats[tier]['verified_failure'] += 1
        else:
            tier_stats[tier]['unverified'] += 1

    # 計算済み統計を追加
    for tier, stats in tier_stats.items():
        stats['unknown_rate'] = (
            stats['unverified'] / stats['total']
            if stats['total'] > 0 else 0
        )
        total_sf = stats['verified_success'] + stats['verified_failure']
        stats['success_rate_verified'] = (
            stats['verified_success'] / total_sf
            if total_sf > 0 else None
        )

    return dict(tier_stats)

def calculate_statistics_by_hexagram(cases: list) -> dict:
    """卦別のVerified統計"""
    hex_stats = defaultdict(lambda: {
        'total': 0,
        'verified': 0,
        'unverified': 0,
        'verified_success': 0,
        'verified_failure': 0,
    })

    for case in cases:
        # 卦を取得（複数フィールド対応）
        hexagram = (
            case.get('hexagram_number') or
            case.get('primary_hexagram', {}).get('number') or
            case.get('hexagram')
        )
        if not hexagram:
            continue

        outcome_status = case.get('outcome_status', 'unverified')
        outcome = case.get('outcome')

        hex_stats[hexagram]['total'] += 1

        if outcome_status == 'verified':
            hex_stats[hexagram]['verified'] += 1
            if outcome == 'Success':
                hex_stats[hexagram]['verified_success'] += 1
            elif outcome == 'Failure':
                hex_stats[hexagram]['verified_failure'] += 1
        else:
            hex_stats[hexagram]['unverified'] += 1

    # 計算済み統計を追加
    for hexagram, stats in hex_stats.items():
        stats['unknown_rate'] = (
            stats['unverified'] / stats['total']
            if stats['total'] > 0 else 0
        )
        total_sf = stats['verified_success'] + stats['verified_failure']
        stats['success_rate_verified'] = (
            stats['verified_success'] / total_sf
            if total_sf > 0 else None
        )

    return dict(hex_stats)

def main():
    print("=" * 70)
    print("Phase 0: Outcome三値化")
    print("=" * 70)
    print()
    print("Codex批評対応:")
    print("  - unverifiedをSuccess扱いしない")
    print("  - 統計はVerified subsetのみで算出")
    print("  - Unknown率を新KPIとして追加")
    print()

    # データ読み込み
    cases = []
    with open(DATA_DIR / 'raw' / 'cases.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    print(f"入力: {len(cases)}件")

    # outcome_status付与
    cases = add_outcome_status(cases)

    # 全体統計
    overall_stats = calculate_verified_statistics(cases)

    print()
    print("【全体統計（Verified subset）】")
    print(f"  総事例数: {overall_stats['total_cases']}件")
    print(f"  Verified: {overall_stats['verified_count']}件 ({overall_stats['verified_count']/overall_stats['total_cases']*100:.1f}%)")
    print(f"  Unverified: {overall_stats['unverified_count']}件")
    print(f"  Unknown率: {overall_stats['unknown_rate']*100:.1f}% ← 新KPI")
    print()
    print(f"  Verified内Outcome: {overall_stats['verified_outcomes']}")
    if overall_stats['success_rate_verified'] is not None:
        print(f"  成功率（Verified only）: {overall_stats['success_rate_verified']*100:.1f}%")
        print(f"  Success/Failure比: {overall_stats['success_failure_ratio_verified']:.2f}:1")

    # Tier別統計
    print()
    print("【Tier別統計（Verified subset）】")
    tier_stats = calculate_statistics_by_tier(cases)
    for tier in ['gold', 'silver', 'bronze', 'quarantine']:
        if tier not in tier_stats:
            continue
        stats = tier_stats[tier]
        print(f"  {tier}:")
        print(f"    総数: {stats['total']}件, Verified: {stats['verified']}件, Unknown率: {stats['unknown_rate']*100:.1f}%")
        if stats['success_rate_verified'] is not None:
            print(f"    成功率（Verified）: {stats['success_rate_verified']*100:.1f}%")
        else:
            print(f"    成功率（Verified）: N/A（S+F=0）")

    # 旧方式との比較
    print()
    print("【旧方式との比較】")

    # 旧方式: 全件でのSuccess率
    all_outcomes = Counter(c.get('outcome') for c in cases)
    old_success = all_outcomes.get('Success', 0)
    old_failure = all_outcomes.get('Failure', 0)
    old_rate = old_success / (old_success + old_failure) if (old_success + old_failure) > 0 else 0

    print(f"  旧方式（全件）: Success {old_success} / Failure {old_failure} = {old_rate*100:.1f}%")
    print(f"  新方式（Verified）: Success {overall_stats['verified_outcomes'].get('Success',0)} / Failure {overall_stats['verified_outcomes'].get('Failure',0)} = {overall_stats['success_rate_verified']*100:.1f}%")
    print()
    print(f"  差分: {(old_rate - overall_stats['success_rate_verified'])*100:.1f}ポイント")
    print(f"  → 旧方式はunverifiedのSuccess偏りで{(old_rate - overall_stats['success_rate_verified'])*100:.1f}%過大評価されていた")

    # データ出力（outcome_status付き）
    print()
    print("【データ出力】")

    output_path = DATA_DIR / 'raw' / 'cases_with_outcome_status.jsonl'
    with open(output_path, 'w') as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')
    print(f"  {output_path.name}: {len(cases)}件")

    # レポート出力
    report = {
        'generated_at': datetime.now().isoformat(),
        'phase': '0_outcome_tristate',
        'codex_compliance': {
            'outcome_status_added': True,
            'verified_subset_stats': True,
            'unknown_rate_kpi': True,
        },
        'overall_statistics': overall_stats,
        'tier_statistics': tier_stats,
        'comparison': {
            'old_success_rate': old_rate,
            'new_success_rate': overall_stats['success_rate_verified'],
            'overestimation': old_rate - overall_stats['success_rate_verified'],
        },
        'new_kpis': {
            'unknown_rate': overall_stats['unknown_rate'],
            'unknown_rate_target': 0.5,  # 目標: 50%以下
            'success_rate_verified': overall_stats['success_rate_verified'],
        },
    }

    report_path = DATA_DIR / 'gold' / 'phase0_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  {report_path.name}")

    print()
    print("【新KPI定義】")
    print(f"  1. Unknown率: {overall_stats['unknown_rate']*100:.1f}% (目標: ≤50%)")
    print(f"  2. 成功率（Verified）: {overall_stats['success_rate_verified']*100:.1f}%")
    print(f"  3. Verified件数: {overall_stats['verified_count']}件")

    print()
    print("【Phase 0 完了】")
    print("  次のステップ: Phase 1（役割ベースソース分類）")

if __name__ == '__main__':
    main()
