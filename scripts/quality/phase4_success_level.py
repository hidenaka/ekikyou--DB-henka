"""
Phase 4: success_level実測化（ベイズ平滑化 + 信頼区間付き）
実行: python3 scripts/quality/phase4_success_level.py
"""

import json
import math
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# 設定
SMOOTHING_ALPHA = 1.0  # ラプラス補正のα（事前観測数）
PRIOR_SUCCESS = 0.5    # 事前成功率
MIN_SAMPLE_RELIABLE = 10  # 信頼できる最小サンプル数

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'

def outcome_to_score(outcome: str) -> float:
    """outcome を数値スコアに変換"""
    mapping = {
        'Success': 1.0,
        'PartialSuccess': 0.7,
        'Mixed': 0.5,
        'Failure': 0.0,
    }
    return mapping.get(outcome, 0.5)

def calculate_bayesian_rate(successes: float, total: int, alpha: float = SMOOTHING_ALPHA, prior: float = PRIOR_SUCCESS) -> float:
    """ベイズ推定による平滑化成功率"""
    # Beta分布の事後期待値: (successes + α*prior) / (total + α)
    return (successes + alpha * prior) / (total + alpha)

def calculate_confidence_interval(successes: float, total: int, confidence: float = 0.95) -> tuple:
    """95%信頼区間（Wilson score interval）"""
    if total == 0:
        return (0, 1)

    p = successes / total if total > 0 else 0.5
    z = 1.96  # 95% confidence

    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

    lower = max(0, centre - margin)
    upper = min(1, centre + margin)

    return (round(lower, 3), round(upper, 3))

def main():
    print("=" * 60)
    print("Phase 4: success_level実測化")
    print("=" * 60)

    # Goldセットを使用（最も信頼できるデータ）
    gold_path = DATA_DIR / 'gold' / 'verified_cases.jsonl'
    if not gold_path.exists():
        print("エラー: Goldセットがありません。Phase 1-2を先に実行してください。")
        return

    cases = []
    with open(gold_path, 'r') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    print(f"Gold事例数: {len(cases)}")

    # 統計収集: hexagram × yao × pattern_type
    stats = defaultdict(lambda: {'total': 0, 'success_sum': 0.0, 'outcomes': []})

    for case in cases:
        yao = case.get('yao_analysis', {})
        hex_id = yao.get('before_hexagram_id')
        yao_pos = yao.get('before_yao_position')
        pattern = case.get('pattern_type')
        outcome = case.get('outcome')

        if hex_id and yao_pos and outcome:
            # 複数粒度でカウント
            keys = [
                f"hex:{hex_id}",                          # 卦のみ
                f"hex:{hex_id}:yao:{yao_pos}",            # 卦×爻
                f"pattern:{pattern}",                     # パターンのみ
                f"hex:{hex_id}:pattern:{pattern}",        # 卦×パターン
            ]

            score = outcome_to_score(outcome)
            for key in keys:
                stats[key]['total'] += 1
                stats[key]['success_sum'] += score
                stats[key]['outcomes'].append(outcome)

    # 統計テーブル作成
    success_table = {}
    for key, data in stats.items():
        total = data['total']
        success_sum = data['success_sum']

        raw_rate = success_sum / total if total > 0 else 0.5
        smoothed_rate = calculate_bayesian_rate(success_sum, total)
        ci_low, ci_high = calculate_confidence_interval(success_sum, total)

        success_table[key] = {
            'sample_count': total,
            'raw_success_rate': round(raw_rate, 3),
            'smoothed_success_rate': round(smoothed_rate, 3),
            'confidence_interval': [ci_low, ci_high],
            'is_reliable': total >= MIN_SAMPLE_RELIABLE,
            'outcome_distribution': dict(zip(*map(list, zip(*[(o, data['outcomes'].count(o)) for o in set(data['outcomes'])]))) if data['outcomes'] else {}),
        }

    # 結果出力
    output_path = DATA_DIR / 'analysis' / 'success_rate_table.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(success_table, f, ensure_ascii=False, indent=2)

    print(f"\n統計テーブル保存: {output_path}")
    print(f"  キー数: {len(success_table)}")

    # サンプル出力
    print("\n【サンプル（卦別）】")
    hex_stats = {k: v for k, v in success_table.items() if k.startswith('hex:') and ':' not in k[4:]}
    for key in sorted(hex_stats.keys(), key=lambda x: int(x.split(':')[1]))[:10]:
        s = hex_stats[key]
        print(f"  {key}: {s['smoothed_success_rate']*100:.0f}% (n={s['sample_count']}, CI={s['confidence_interval']})")

    print("\n【サンプル（パターン別）】")
    pattern_stats = {k: v for k, v in success_table.items() if k.startswith('pattern:')}
    for key, s in sorted(pattern_stats.items(), key=lambda x: -x[1]['sample_count'])[:10]:
        print(f"  {key}: {s['smoothed_success_rate']*100:.0f}% (n={s['sample_count']})")

    # 旧データへの適用シミュレーション
    print("\n【success_level更新シミュレーション】")
    print("  旧: 固定値 85/65/50/15")
    print("  新: outcome × 卦 × パターンの実測統計ベース")
    print("  平滑化: ラプラス補正 (α=1.0, prior=0.5)")
    print("  メタ情報: sample_count, confidence_interval, is_reliable")

    # レポート
    report = {
        'generated_at': datetime.now().isoformat(),
        'gold_cases_used': len(cases),
        'unique_keys': len(success_table),
        'smoothing_alpha': SMOOTHING_ALPHA,
        'prior_success_rate': PRIOR_SUCCESS,
        'min_sample_reliable': MIN_SAMPLE_RELIABLE,
    }

    report_path = DATA_DIR / 'analysis' / 'success_rate_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nレポート保存: {report_path}")
    print("\n完了！")

if __name__ == '__main__':
    main()
