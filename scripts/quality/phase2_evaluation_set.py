"""
Phase 2: 固定評価セット構築
Codex批評に基づき、校正KPIの前提となる検証済みデータセットを構築

設計:
- 層化抽出で500件
- 人手検証（二重ラベリング）用のプロトコル
- Proper Scoring Rule計算の基盤
"""

import json
import random
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'

# ============================================
# 1. 層化抽出設計
# ============================================

STRATIFICATION_SCHEMA = {
    # 品質Tier別
    'tier': {
        'gold': 0.25,      # 25%
        'silver': 0.35,    # 35%
        'bronze': 0.20,    # 20%
        'quarantine': 0.20,  # 20%
    },

    # Outcome別（各Tier内での配分）
    'outcome': {
        'Success': 0.45,
        'Failure': 0.35,
        'Mixed': 0.15,
        'PartialSuccess': 0.05,
    },

    # 時期別（optional、データがある場合）
    'period': {
        'recent': 0.40,    # 最近3年
        'mid': 0.35,       # 3-10年前
        'old': 0.25,       # 10年以上前
    },
}

TARGET_SIZE = 500

def stratified_sample(cases: list, tier_assignments: dict) -> list:
    """
    層化抽出

    Args:
        cases: 全事例リスト
        tier_assignments: case_id -> tier のマッピング

    Returns:
        抽出された事例リスト
    """
    # Tier別にグループ化
    tier_groups = defaultdict(list)
    for case in cases:
        case_id = case.get('id') or id(case)
        tier = tier_assignments.get(case_id, 'unknown')
        tier_groups[tier].append(case)

    sampled = []
    tier_quotas = {tier: int(TARGET_SIZE * ratio) for tier, ratio in STRATIFICATION_SCHEMA['tier'].items()}

    for tier, quota in tier_quotas.items():
        pool = tier_groups.get(tier, [])
        if not pool:
            print(f"  警告: {tier}のプールが空")
            continue

        # Outcome別にさらに層化
        outcome_groups = defaultdict(list)
        for case in pool:
            outcome = case.get('outcome', 'Unknown')
            outcome_groups[outcome].append(case)

        tier_sampled = []
        for outcome, outcome_ratio in STRATIFICATION_SCHEMA['outcome'].items():
            outcome_quota = int(quota * outcome_ratio)
            outcome_pool = outcome_groups.get(outcome, [])

            if len(outcome_pool) >= outcome_quota:
                tier_sampled.extend(random.sample(outcome_pool, outcome_quota))
            else:
                tier_sampled.extend(outcome_pool)
                print(f"  警告: {tier}/{outcome} プール不足 ({len(outcome_pool)}/{outcome_quota})")

        # 不足分を補充
        remaining = quota - len(tier_sampled)
        if remaining > 0:
            already_sampled_ids = {id(c) for c in tier_sampled}
            available = [c for c in pool if id(c) not in already_sampled_ids]
            if available:
                additional = random.sample(available, min(remaining, len(available)))
                tier_sampled.extend(additional)

        sampled.extend(tier_sampled)

    return sampled

# ============================================
# 2. 検証プロトコル
# ============================================

VERIFICATION_PROTOCOL = {
    'fields_to_verify': [
        {
            'name': 'outcome_verified',
            'description': '結果（Success/Failure/Mixed）が事実として正しいか',
            'options': ['correct', 'incorrect', 'unverifiable'],
        },
        {
            'name': 'source_quality',
            'description': '情報源の品質',
            'options': ['primary', 'secondary', 'tertiary', 'unreliable'],
        },
        {
            'name': 'factual_accuracy',
            'description': '記載内容の事実正確性',
            'options': ['accurate', 'minor_error', 'major_error', 'unverifiable'],
        },
        {
            'name': 'coi_assessment',
            'description': '利益相反の有無',
            'options': ['none', 'potential', 'clear', 'unknown'],
        },
    ],

    'adjudication_rules': {
        'agreement_threshold': 0.8,  # 80%一致で確定
        'disagreement_resolution': 'third_reviewer',  # 不一致時は第三者
    },

    'documentation': {
        'evidence_url_required': True,  # 根拠URLの記録必須
        'notes_required_for_error': True,  # エラー判定時はメモ必須
    },
}

def generate_verification_template(case: dict) -> dict:
    """
    検証テンプレート生成
    """
    return {
        'case_id': case.get('id'),
        'target_name': case.get('target_name') or case.get('entity_name'),
        'original_outcome': case.get('outcome'),
        'original_trust_level': case.get('trust_level'),
        'sources': case.get('sources') or [case.get('source')],

        # 検証フィールド（空欄）
        'verification': {
            'reviewer_1': {
                'outcome_verified': None,
                'source_quality': None,
                'factual_accuracy': None,
                'coi_assessment': None,
                'evidence_urls': [],
                'notes': '',
                'verified_at': None,
            },
            'reviewer_2': {
                'outcome_verified': None,
                'source_quality': None,
                'factual_accuracy': None,
                'coi_assessment': None,
                'evidence_urls': [],
                'notes': '',
                'verified_at': None,
            },
            'adjudication': {
                'final_outcome_verified': None,
                'final_source_quality': None,
                'final_factual_accuracy': None,
                'final_coi_assessment': None,
                'adjudicator_notes': '',
                'adjudicated_at': None,
            },
        },

        'status': 'pending',  # pending, in_review, completed, disputed
    }

# ============================================
# 3. Proper Scoring Rule計算基盤
# ============================================

def calculate_calibration_metrics(verified_set: list) -> dict:
    """
    校正指標を計算（検証済みセットが揃った後に使用）

    Args:
        verified_set: 検証済み事例リスト（verification.final_* が埋まっている）

    Returns:
        校正指標
    """
    # 注: 実際の計算は検証完了後
    # ここでは構造を示す

    return {
        'ece': None,  # Expected Calibration Error
        'brier_score': None,  # Brier Score
        'log_loss': None,  # Log Loss
        'calibration_slope': None,  # 校正傾き
        'calibration_intercept': None,  # 校正切片
        'auc_roc': None,  # AUC-ROC

        'by_tier': {
            'gold': {'ece': None, 'brier': None},
            'silver': {'ece': None, 'brier': None},
            'bronze': {'ece': None, 'brier': None},
        },

        'unknown_rate': None,  # Unverifiable率
    }

# ============================================
# 4. メイン処理
# ============================================

def main():
    print("=" * 70)
    print("Phase 2: 固定評価セット構築")
    print("=" * 70)
    print()
    print("Codex批評対応:")
    print("  - 層化抽出で代表性確保")
    print("  - 二重ラベリングで再現性確保")
    print("  - Proper Scoring Rule計算の基盤")
    print()

    # データ読み込み
    cases = []
    with open(DATA_DIR / 'raw' / 'cases.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    print(f"入力: {len(cases)}件")

    # v4分類を使用してTier割り当て
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from phase1_2_v4 import evaluate_case_v4

    tier_assignments = {}
    tier_counts = Counter()
    for i, case in enumerate(cases):
        case['id'] = case.get('id') or f"case_{i}"
        eval_result = evaluate_case_v4(case)
        tier_assignments[case['id']] = eval_result['tier']
        tier_counts[eval_result['tier']] += 1

    print()
    print("【Tier分布】")
    for tier, count in tier_counts.most_common():
        print(f"  {tier}: {count}件")

    # 層化抽出
    print()
    print(f"【層化抽出: {TARGET_SIZE}件】")
    random.seed(42)  # 再現性のため
    sampled = stratified_sample(cases, tier_assignments)

    print(f"  抽出結果: {len(sampled)}件")

    # 抽出結果の分布確認
    sampled_tier_counts = Counter()
    sampled_outcome_counts = Counter()
    for case in sampled:
        tier = tier_assignments.get(case['id'], 'unknown')
        sampled_tier_counts[tier] += 1
        sampled_outcome_counts[case.get('outcome', 'Unknown')] += 1

    print()
    print("【抽出セットTier分布】")
    for tier, count in sampled_tier_counts.most_common():
        target = int(TARGET_SIZE * STRATIFICATION_SCHEMA['tier'].get(tier, 0))
        print(f"  {tier}: {count}件 (目標: {target}件)")

    print()
    print("【抽出セットOutcome分布】")
    for outcome, count in sampled_outcome_counts.most_common():
        print(f"  {outcome}: {count}件")

    # 検証テンプレート生成
    print()
    print("【検証テンプレート生成】")
    evaluation_set = []
    for case in sampled:
        template = generate_verification_template(case)
        template['assigned_tier'] = tier_assignments.get(case['id'], 'unknown')
        evaluation_set.append(template)

    # 出力
    output_path = DATA_DIR / 'audit' / 'evaluation_set_500.jsonl'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for item in evaluation_set:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  {output_path}: {len(evaluation_set)}件")

    # 検証プロトコル出力
    protocol_path = DATA_DIR / 'audit' / 'verification_protocol.json'
    with open(protocol_path, 'w') as f:
        json.dump(VERIFICATION_PROTOCOL, f, ensure_ascii=False, indent=2)
    print(f"  {protocol_path}")

    # レポート
    report = {
        'generated_at': datetime.now().isoformat(),
        'phase': '2_evaluation_set',
        'codex_compliance': {
            'stratified_sampling': True,
            'double_labeling_protocol': True,
            'proper_scoring_foundation': True,
        },
        'sampling': {
            'target_size': TARGET_SIZE,
            'actual_size': len(sampled),
            'tier_distribution': dict(sampled_tier_counts),
            'outcome_distribution': dict(sampled_outcome_counts),
        },
        'verification_status': {
            'pending': len(evaluation_set),
            'in_review': 0,
            'completed': 0,
            'disputed': 0,
        },
    }

    report_path = DATA_DIR / 'gold' / 'phase2_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  {report_path}")

    print()
    print("【Phase 2 完了】")
    print()
    print("【次のステップ】")
    print("  1. evaluation_set_500.jsonl を検証者に配布")
    print("  2. 二重ラベリング実施（reviewer_1, reviewer_2）")
    print("  3. 不一致箇所のadjudication")
    print("  4. 完了後にProper Scoring Rule計算")
    print()
    print("【検証フィールド】")
    for field in VERIFICATION_PROTOCOL['fields_to_verify']:
        print(f"  - {field['name']}: {field['options']}")

if __name__ == '__main__':
    main()
