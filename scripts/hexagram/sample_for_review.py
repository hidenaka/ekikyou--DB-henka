#!/usr/bin/env python3
"""Phase 5: 適合度レビュー用サンプル抽出

サンプリング方針（v3.1計画に基づく）:
- 希少卦（10件以下）: 全件
- 中頻度卦（11-100件）: 10件
- 頻出卦（101件以上）: 5件
"""

import json
import random
from collections import defaultdict
from pathlib import Path

def sample_for_review(input_path: str, output_path: str):
    """卦ごとにサンプリングしてレビュー用データを生成"""

    # 事例を読み込み
    cases = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    print(f"読み込み: {len(cases)}件")

    # 卦ごとにグループ化
    by_hexagram = defaultdict(list)
    for case in cases:
        hexagram = case.get('hexagram_name', '不明')
        by_hexagram[hexagram].append(case)

    # サンプリング
    samples = []
    stats = {
        'rare': {'count': 0, 'hexagrams': [], 'samples': 0},      # 10件以下
        'medium': {'count': 0, 'hexagrams': [], 'samples': 0},    # 11-100件
        'frequent': {'count': 0, 'hexagrams': [], 'samples': 0},  # 101件以上
    }

    # 再現性のためシードを固定
    random.seed(42)

    for hexagram, hex_cases in sorted(by_hexagram.items()):
        count = len(hex_cases)
        if count == 0:
            continue
        elif count <= 10:
            # 希少卦は全件
            samples.extend(hex_cases)
            stats['rare']['count'] += 1
            stats['rare']['hexagrams'].append(f"{hexagram}({count}件)")
            stats['rare']['samples'] += count
        elif count <= 100:
            # 中頻度卦は10件
            sampled = random.sample(hex_cases, 10)
            samples.extend(sampled)
            stats['medium']['count'] += 1
            stats['medium']['hexagrams'].append(f"{hexagram}({count}件)")
            stats['medium']['samples'] += 10
        else:
            # 頻出卦は5件
            sampled = random.sample(hex_cases, 5)
            samples.extend(sampled)
            stats['frequent']['count'] += 1
            stats['frequent']['hexagrams'].append(f"{hexagram}({count}件)")
            stats['frequent']['samples'] += 5

    # 出力ディレクトリ作成
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 出力
    with open(output_path, 'w', encoding='utf-8') as f:
        for case in samples:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')

    # 結果サマリー
    print("\n" + "="*60)
    print("適合度レビュー用サンプル抽出完了")
    print("="*60)
    print(f"\n総サンプル数: {len(samples)}件")
    print(f"カバー卦数: {len(by_hexagram)}卦")

    print(f"\n【希少卦（10件以下）】{stats['rare']['count']}卦 → 全件抽出: {stats['rare']['samples']}件")
    if stats['rare']['hexagrams']:
        for h in stats['rare']['hexagrams']:
            print(f"  - {h}")

    print(f"\n【中頻度卦（11-100件）】{stats['medium']['count']}卦 → 各10件抽出: {stats['medium']['samples']}件")
    if stats['medium']['hexagrams']:
        for h in stats['medium']['hexagrams'][:10]:
            print(f"  - {h}")
        if len(stats['medium']['hexagrams']) > 10:
            print(f"  ... 他{len(stats['medium']['hexagrams'])-10}卦")

    print(f"\n【頻出卦（101件以上）】{stats['frequent']['count']}卦 → 各5件抽出: {stats['frequent']['samples']}件")
    if stats['frequent']['hexagrams']:
        for h in stats['frequent']['hexagrams']:
            print(f"  - {h}")

    print(f"\n出力: {output_path}")

    return samples, stats

if __name__ == '__main__':
    sample_for_review(
        'data/raw/cases_with_hexagram.jsonl',
        'data/hexagram/review_samples.jsonl'
    )
