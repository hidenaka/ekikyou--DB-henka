#!/usr/bin/env python3
"""
64卦×6爻 網羅性チェック（Canonical版）

canonical 事例のみをカウントし、正確な薄いセルを特定
"""

import json
from pathlib import Path
from collections import defaultdict

def load_canonical_cases():
    """canonical 事例のみをロード"""
    cases_path = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"
    cases = []
    with open(cases_path, 'r', encoding='utf-8') as f:
        for line in f:
            case = json.loads(line.strip())
            if case.get('is_canonical', True):  # canonical のみ
                cases.append(case)
    return cases

def analyze_coverage():
    """64卦×6爻のカバレッジを分析"""
    cases = load_canonical_cases()

    print(f"\n{'='*60}")
    print(f"64卦×6爻 網羅性チェック（Canonical版）")
    print(f"{'='*60}")
    print(f"\nCanonical事例数: {len(cases):,}件")

    # セルごとにカウント
    cells = defaultdict(int)
    for case in cases:
        yao_info = case.get('yao_analysis') or {}
        hex_id = yao_info.get('before_hexagram_id')
        yao = yao_info.get('before_yao_position')
        if hex_id and yao:
            cells[(hex_id, yao)] += 1

    # 統計
    total_cells = 64 * 6
    covered = len([v for v in cells.values() if v > 0])
    thin_1 = len([v for v in cells.values() if v == 1])
    thin_2 = len([v for v in cells.values() if v == 2])
    thin_3_5 = len([v for v in cells.values() if 3 <= v <= 5])
    good = len([v for v in cells.values() if v > 5])

    print(f"\n--- 統計 ---")
    print(f"総セル数: {total_cells}")
    print(f"カバー済み: {covered} ({covered/total_cells:.1%})")
    print(f"")
    print(f"分布:")
    print(f"  1件: {thin_1}セル")
    print(f"  2件: {thin_2}セル")
    print(f"  3-5件: {thin_3_5}セル")
    print(f"  6件以上: {good}セル")
    print(f"")
    print(f"薄いセル合計（5件以下）: {thin_1 + thin_2 + thin_3_5}セル")

    # 最も薄いセル一覧
    thin_cells = [(k, v) for k, v in cells.items() if v <= 5]
    thin_cells.sort(key=lambda x: (x[1], x[0][0], x[0][1]))

    print(f"\n--- 最も薄いセル（補強優先）---")
    for (hex_id, yao), count in thin_cells[:30]:
        print(f"  第{hex_id}卦 {yao}爻: {count}件")

    # 補強計画
    print(f"\n--- 補強計画 ---")
    needed = sum(max(0, 6 - v) for v in cells.values())
    print(f"全セル6件以上にするために必要な追加: {needed}件")

    # JSONで保存
    report = {
        'total_canonical': len(cases),
        'coverage': {
            'total_cells': total_cells,
            'covered': covered,
            'thin_1': thin_1,
            'thin_2': thin_2,
            'thin_3_5': thin_3_5,
            'good': good
        },
        'thin_cells': [
            {'hexagram_id': k[0], 'yao': k[1], 'count': v}
            for k, v in thin_cells
        ],
        'needed_for_6': needed
    }

    output_path = Path(__file__).parent.parent / "data" / "diagnostic" / "yao_coverage_canonical.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[保存] {output_path}")

    return report

if __name__ == "__main__":
    analyze_coverage()
