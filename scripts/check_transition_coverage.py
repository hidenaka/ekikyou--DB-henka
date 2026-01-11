#!/usr/bin/env python3
"""
4段階遷移パターンのカバレッジ確認スクリプト

使用方法:
  python3 scripts/check_transition_coverage.py [--detail] [--missing N]

オプション:
  --detail    : 詳細な分析を表示
  --missing N : 未カバーパターンをN件表示
"""

import json
import sys
from collections import Counter
from itertools import product
from pathlib import Path

def main():
    detail = '--detail' in sys.argv
    missing_count = 20
    for i, arg in enumerate(sys.argv):
        if arg == '--missing' and i + 1 < len(sys.argv):
            missing_count = int(sys.argv[i + 1])

    db_path = Path('data/raw/cases.jsonl')
    cases = [json.loads(line) for line in open(db_path)]

    # 現在のパターンをカウント
    existing = Counter(
        (c.get('before_hex',''), c.get('trigger_hex',''), c.get('action_hex',''), c.get('after_hex',''))
        for c in cases
    )

    # 全パターン
    all_hexes = ['乾', '坤', '震', '巽', '坎', '離', '艮', '兌']
    all_patterns = list(product(all_hexes, repeat=4))

    missing = [p for p in all_patterns if existing.get(p, 0) == 0]
    low_coverage = [p for p in all_patterns if 0 < existing.get(p, 0) < 3]
    good_coverage = [p for p in all_patterns if existing.get(p, 0) >= 3]

    print("=" * 60)
    print("    4段階遷移パターン カバレッジレポート")
    print("=" * 60)
    print(f"\n総ケース数: {len(cases):,}件")
    print(f"\n■ パターンカバレッジ")
    print(f"  理論上: 4,096通り")
    print(f"  カバー済み(1件以上): {len(existing):,}通り ({len(existing)/4096*100:.1f}%)")
    print(f"  十分(3件以上): {len(good_coverage):,}通り ({len(good_coverage)/4096*100:.1f}%)")
    print(f"  低カバー(1-2件): {len(low_coverage):,}通り")
    print(f"  未カバー: {len(missing):,}通り ({len(missing)/4096*100:.1f}%)")

    # 目標との差分
    target_coverage = 0.70
    current_coverage = len(existing) / 4096
    patterns_needed = int(4096 * target_coverage) - len(existing)
    print(f"\n■ 目標達成まで")
    print(f"  現在: {current_coverage*100:.1f}%")
    print(f"  目標: {target_coverage*100:.1f}%")
    print(f"  必要パターン数: {max(0, patterns_needed)}通り")

    if detail:
        print(f"\n■ before_hex別カバレッジ")
        for h in all_hexes:
            total_h = 512  # 8^3
            covered_h = sum(1 for p in existing if p[0] == h)
            print(f"  {h}: {covered_h}/512 ({covered_h/512*100:.1f}%)")

    # 意味のある未カバーパターン
    print(f"\n■ 優先対応すべき未カバーパターン (上位{missing_count}件)")
    priority = []
    for p in missing:
        before, trigger, action, after = p
        score = 0
        if trigger in ['震', '坎']: score += 2
        if action in ['震', '巽', '艮']: score += 1
        if (before == '乾' and after == '坤') or (before == '坤' and after == '乾'): score += 2
        if (before == '坎' and after in ['乾', '離']): score += 1
        if (before == '艮' and after in ['震', '乾']): score += 1
        priority.append((score, p))

    priority.sort(reverse=True)
    for i, (score, p) in enumerate(priority[:missing_count]):
        print(f"  {i+1}. {p[0]}→{p[1]}→{p[2]}→{p[3]} (スコア:{score})")

    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
