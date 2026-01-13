"""
Phase 3: 主張単位スキーマ
Codex批評に基づき、逆インセンティブを排除する設計

Core Claims概念:
- 品質判定に必須の主張のみを検証対象とする
- 主張を増やしても不利にならない集約則
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from enum import Enum

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'

# ============================================
# 1. Enum定義
# ============================================

class ClaimType(str, Enum):
    VERIFIABLE_FACT = 'verifiable_fact'
    INFERENCE = 'inference'
    EVALUATION = 'evaluation'
    OPINION = 'opinion'

class EvidenceType(str, Enum):
    PRIMARY_SOURCE = 'primary_source'
    SECONDARY_SOURCE = 'secondary_source'
    TERTIARY_SOURCE = 'tertiary_source'
    NO_EVIDENCE = 'no_evidence'

class COIType(str, Enum):
    NONE = 'none'
    SELF = 'self'
    AFFILIATED = 'affiliated'
    SPONSORED = 'sponsored'
    UNKNOWN = 'unknown'

# ============================================
# 2. スキーマ定義
# ============================================

CLAIM_SCHEMA = {
    'claim_id': str,  # 一意識別子
    'claim_text': str,  # 主張の文言
    'claim_type': ClaimType,  # 主張タイプ
    'is_core': bool,  # Core Claimかどうか

    # 証拠リンク
    'evidence': [{
        'source_index': int,  # sources配列のインデックス
        'evidence_type': EvidenceType,
        'excerpt': str,  # 該当箇所の引用（optional）
        'verified': bool,
    }],

    # 矛盾検出
    'contradictions': [str],  # 矛盾する主張のIDリスト

    # COI
    'coi': COIType,

    # 検証状態
    'verified': bool,
}

CASE_WITH_CLAIMS_SCHEMA = {
    # 既存フィールド
    'id': str,
    'target_name': str,
    'outcome': str,
    'sources': [str],
    'trust_level': str,

    # 新規: outcome_status (Phase 0)
    'outcome_status': str,  # 'verified' | 'unverified'

    # 新規: 主張リスト (Phase 3)
    'claims': [CLAIM_SCHEMA],

    # 新規: Core Claims サマリー
    'core_claims_summary': {
        'outcome_claim_id': str,
        'target_claim_id': str,
        'timeline_claim_id': Optional[str],
    },
}

# ============================================
# 3. 集約則（逆インセンティブ排除）
# ============================================

def is_gold_claim(claim: dict) -> Tuple[bool, str]:
    """
    主張レベルのGold判定
    """
    # 1. 検証可能な事実のみがGold候補
    if claim.get('claim_type') != ClaimType.VERIFIABLE_FACT.value:
        return False, f'claim_type_not_fact: {claim.get("claim_type")}'

    # 2. 証拠があること
    evidence = claim.get('evidence', [])
    valid_evidence = [e for e in evidence
                      if e.get('evidence_type') in [EvidenceType.PRIMARY_SOURCE.value,
                                                    EvidenceType.SECONDARY_SOURCE.value]]
    if not valid_evidence:
        return False, 'no_valid_evidence'

    # 3. 検証済みであること
    if not claim.get('verified'):
        return False, 'not_verified'

    # 4. COI=selfでないこと
    if claim.get('coi') == COIType.SELF.value:
        return False, 'coi_self'

    return True, 'gold_claim'

def is_gold_case(case: dict) -> Tuple[bool, str]:
    """
    ケースレベルのGold判定
    逆インセンティブを排除する集約則
    """
    claims = case.get('claims', [])

    # claimsがない場合は旧ロジックにフォールバック
    if not claims:
        return None, 'no_claims_fallback_to_legacy'

    # Core Claimsを抽出
    core_claims = [c for c in claims if c.get('is_core')]

    # 条件1: Core Claimsが存在すること
    if not core_claims:
        return False, 'no_core_claims'

    # 条件2: 全Core Claimsがverified
    unverified_core = [c for c in core_claims if not c.get('verified')]
    if unverified_core:
        return False, f'unverified_core_claims: {len(unverified_core)}'

    # 条件3: 全Core ClaimsがGold基準を満たす
    for claim in core_claims:
        is_gold, reason = is_gold_claim(claim)
        if not is_gold:
            return False, f'core_claim_not_gold: {reason}'

    # 条件4: 重大な反証主張がないこと
    contradictions = [c for c in claims if c.get('contradicts_core')]
    if contradictions:
        return False, f'contradicting_claims: {len(contradictions)}'

    return True, 'all_conditions_met'

# ============================================
# 4. サンプル生成
# ============================================

def generate_sample_case_with_claims() -> dict:
    """
    主張付きケースのサンプル生成
    """
    return {
        'id': 'sample_001',
        'target_name': 'サンプル株式会社',
        'outcome': 'Success',
        'sources': [
            'https://example.go.jp/ir/report.pdf',
            'https://nikkei.com/article/xxx',
        ],
        'trust_level': 'verified',
        'outcome_status': 'verified',

        'claims': [
            {
                'claim_id': 'c001',
                'claim_text': '2023年3月期の売上高は1,000億円に達した',
                'claim_type': 'verifiable_fact',
                'is_core': True,  # Core Claim
                'evidence': [
                    {
                        'source_index': 0,
                        'evidence_type': 'primary_source',
                        'excerpt': '売上高 100,000百万円',
                        'verified': True,
                    },
                ],
                'contradictions': [],
                'coi': 'none',
                'verified': True,
            },
            {
                'claim_id': 'c002',
                'claim_text': '3年連続の増収増益を達成',
                'claim_type': 'verifiable_fact',
                'is_core': True,  # Core Claim (outcome根拠)
                'evidence': [
                    {
                        'source_index': 1,
                        'evidence_type': 'secondary_source',
                        'excerpt': '同社は3年連続で増収増益',
                        'verified': True,
                    },
                ],
                'contradictions': [],
                'coi': 'none',
                'verified': True,
            },
            {
                'claim_id': 'c003',
                'claim_text': '業界で最も革新的な企業と評価されている',
                'claim_type': 'evaluation',  # 評価→Core Claimではない
                'is_core': False,
                'evidence': [],
                'contradictions': [],
                'coi': 'none',
                'verified': False,  # 未検証でもCore Claimでないので影響なし
            },
        ],

        'core_claims_summary': {
            'outcome_claim_id': 'c002',
            'target_claim_id': 'c001',
            'timeline_claim_id': None,
        },
    }

# ============================================
# 5. メイン処理
# ============================================

def main():
    print("=" * 70)
    print("Phase 3: 主張単位スキーマ")
    print("=" * 70)
    print()
    print("Codex批評対応:")
    print("  - Core Claims概念で逆インセンティブ排除")
    print("  - 主張を増やしても不利にならない集約則")
    print("  - claim_type定義とevidence linking")
    print()

    # サンプル生成
    sample = generate_sample_case_with_claims()

    print("【サンプルケース】")
    print(f"  target: {sample['target_name']}")
    print(f"  outcome: {sample['outcome']}")
    print(f"  claims: {len(sample['claims'])}件")

    # Core Claims
    core_claims = [c for c in sample['claims'] if c.get('is_core')]
    print(f"  core_claims: {len(core_claims)}件")

    # Gold判定
    is_gold, reason = is_gold_case(sample)
    print(f"  gold_case: {is_gold} ({reason})")

    print()
    print("【主張詳細】")
    for claim in sample['claims']:
        is_gold_c, reason_c = is_gold_claim(claim)
        core_mark = "[CORE]" if claim['is_core'] else ""
        gold_mark = "[GOLD]" if is_gold_c else ""
        print(f"  {claim['claim_id']} {core_mark} {gold_mark}")
        print(f"    type: {claim['claim_type']}")
        print(f"    text: {claim['claim_text'][:50]}...")
        print(f"    verified: {claim['verified']}")
        if not is_gold_c:
            print(f"    reason: {reason_c}")

    print()
    print("【集約則の検証】")
    print("  1. Core Claims存在: ", "OK" if core_claims else "NG")
    print("  2. 全Core Claims検証済み: ", "OK" if all(c['verified'] for c in core_claims) else "NG")
    print("  3. 重大な反証なし: ", "OK" if not any(c.get('contradicts_core') for c in sample['claims']) else "NG")

    print()
    print("【逆インセンティブ排除の確認】")
    print("  - c003(evaluation)は未検証だがCore Claimではないため判定に影響しない")
    print("  - 主張を追加してもCore Claims条件が満たされていればGold維持")

    # スキーマ出力
    schema_path = DATA_DIR / 'schema' / 'claim_schema_v1.json'
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    with open(schema_path, 'w') as f:
        json.dump({
            'version': 'v1',
            'generated_at': datetime.now().isoformat(),
            'claim_types': [e.value for e in ClaimType],
            'evidence_types': [e.value for e in EvidenceType],
            'coi_types': [e.value for e in COIType],
            'sample_case': sample,
        }, f, ensure_ascii=False, indent=2)
    print()
    print(f"【スキーマ出力】{schema_path}")

    # レポート
    report = {
        'generated_at': datetime.now().isoformat(),
        'phase': '3_claim_schema',
        'codex_compliance': {
            'core_claims_concept': True,
            'no_reverse_incentive': True,
            'claim_type_definitions': True,
            'evidence_linking': True,
            'aggregation_rules': True,
        },
        'schema_version': 'v1',
        'sample_validation': {
            'is_gold': is_gold,
            'reason': reason,
        },
    }

    report_path = DATA_DIR / 'gold' / 'phase3_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"【レポート出力】{report_path}")

    print()
    print("【Phase 3 完了】")
    print()
    print("【次のステップ】")
    print("  1. evaluation_set_500から100件を選定してパイロット実施")
    print("  2. 手動で主張タグ付け（Core Claims特定含む）")
    print("  3. 集約則の妥当性検証")
    print("  4. 自動化可能性の評価")

if __name__ == '__main__':
    main()
