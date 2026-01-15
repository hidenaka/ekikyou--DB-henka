#!/usr/bin/env python3
"""
一致度計測ツール（Codex推奨#4対応）

複数レビュワーの卦・爻判定の一致率をCohen's Kappaで計算
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def cohens_kappa(judgments1, judgments2):
    """
    Cohen's Kappa係数を計算

    κ = (Po - Pe) / (1 - Pe)
    Po: 観測一致率
    Pe: 偶然一致率
    """
    if len(judgments1) != len(judgments2):
        raise ValueError("判定数が一致しません")

    n = len(judgments1)
    if n == 0:
        return 0.0

    # 一致数
    agreements = sum(1 for j1, j2 in zip(judgments1, judgments2) if j1 == j2)
    po = agreements / n

    # カテゴリ分布
    categories = set(judgments1) | set(judgments2)
    pe = 0.0

    for cat in categories:
        p1 = sum(1 for j in judgments1 if j == cat) / n
        p2 = sum(1 for j in judgments2 if j == cat) / n
        pe += p1 * p2

    # Kappa計算
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0

    kappa = (po - pe) / (1 - pe)
    return kappa

def interpret_kappa(kappa):
    """Kappa値の解釈"""
    if kappa < 0:
        return "偶然以下（問題あり）"
    elif kappa < 0.2:
        return "slight（わずか）"
    elif kappa < 0.4:
        return "fair（まあまあ）"
    elif kappa < 0.6:
        return "moderate（中程度）"
    elif kappa < 0.8:
        return "substantial（実質的）"
    else:
        return "almost perfect（ほぼ完全）"

def analyze_review_file(review_file_path):
    """
    レビューファイルを分析

    期待フォーマット:
    [
      {
        "case_id": "...",
        "reviewer_a": {"hexagram": 1, "yao": 6},
        "reviewer_b": {"hexagram": 1, "yao": 6}
      },
      ...
    ]
    """
    with open(review_file_path, 'r', encoding='utf-8') as f:
        reviews = json.load(f)

    hex_a = []
    hex_b = []
    yao_a = []
    yao_b = []
    combined_a = []
    combined_b = []

    disagreements = []

    for r in reviews:
        ra = r.get('reviewer_a', {})
        rb = r.get('reviewer_b', {})

        ha, ya = ra.get('hexagram'), ra.get('yao')
        hb, yb = rb.get('hexagram'), rb.get('yao')

        if ha and hb:
            hex_a.append(ha)
            hex_b.append(hb)

        if ya and yb:
            yao_a.append(ya)
            yao_b.append(yb)

        if ha and hb and ya and yb:
            combined_a.append(f"{ha}_{ya}")
            combined_b.append(f"{hb}_{yb}")

            if ha != hb or ya != yb:
                disagreements.append({
                    'case_id': r.get('case_id'),
                    'a': f"第{ha}卦{ya}爻",
                    'b': f"第{hb}卦{yb}爻"
                })

    return {
        'hex': (hex_a, hex_b),
        'yao': (yao_a, yao_b),
        'combined': (combined_a, combined_b),
        'disagreements': disagreements,
        'total': len(reviews)
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 calculate_agreement.py <review_json_path>")
        print("\nレビューファイルを作成してから実行してください。")
        print("フォーマット例:")
        print("""
[
  {
    "case_id": "case_001",
    "target_name": "WeWork",
    "reviewer_a": {"hexagram": 1, "yao": 6},
    "reviewer_b": {"hexagram": 1, "yao": 6}
  }
]
""")
        sys.exit(1)

    review_path = sys.argv[1]
    data = analyze_review_file(review_path)

    print(f"\n{'='*50}")
    print(f"一致度計測レポート")
    print(f"{'='*50}")
    print(f"\n対象件数: {data['total']}件")

    # 卦レベル
    hex_a, hex_b = data['hex']
    if hex_a:
        kappa_hex = cohens_kappa(hex_a, hex_b)
        agreement_hex = sum(1 for a, b in zip(hex_a, hex_b) if a == b) / len(hex_a)
        print(f"\n--- 卦レベル一致 ---")
        print(f"  観測一致率: {agreement_hex:.1%}")
        print(f"  Cohen's κ: {kappa_hex:.3f} ({interpret_kappa(kappa_hex)})")
        print(f"  目標: κ ≥ 0.6")
        if kappa_hex >= 0.6:
            print(f"  → ✅ 合格")
        else:
            print(f"  → ❌ 不合格（レビュープロセス改善が必要）")

    # 爻レベル
    yao_a, yao_b = data['yao']
    if yao_a:
        kappa_yao = cohens_kappa(yao_a, yao_b)
        agreement_yao = sum(1 for a, b in zip(yao_a, yao_b) if a == b) / len(yao_a)
        print(f"\n--- 爻レベル一致 ---")
        print(f"  観測一致率: {agreement_yao:.1%}")
        print(f"  Cohen's κ: {kappa_yao:.3f} ({interpret_kappa(kappa_yao)})")
        print(f"  目標: κ ≥ 0.4")
        if kappa_yao >= 0.4:
            print(f"  → ✅ 合格")
        else:
            print(f"  → ❌ 不合格（爻の判定基準を明確化する必要）")

    # 不一致事例
    if data['disagreements']:
        print(f"\n--- 不一致事例 ---")
        for d in data['disagreements'][:5]:
            print(f"  {d['case_id']}: {d['a']} vs {d['b']}")
        if len(data['disagreements']) > 5:
            print(f"  ...他{len(data['disagreements'])-5}件")

    print(f"\n{'='*50}")

if __name__ == "__main__":
    main()
