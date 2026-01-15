#!/usr/bin/env python3
"""
意味的重複検出スクリプト（Codex推奨#5対応）

完全一致だけでなく、類似度ベースで重複・近接事例を検出
"""

import json
import sys
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict

def similarity(s1, s2):
    """文字列の類似度（0-1）"""
    if not s1 or not s2:
        return 0.0
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

def load_cases():
    """DBから全事例をロード"""
    cases_path = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"
    cases = []
    if cases_path.exists():
        with open(cases_path, 'r', encoding='utf-8') as f:
            for line in f:
                cases.append(json.loads(line.strip()))
    return cases

def find_duplicates(cases, name_threshold=0.8, summary_threshold=0.7):
    """
    重複・類似事例を検出

    Returns:
        list: 重複グループのリスト
    """
    duplicates = []
    checked = set()

    for i, case1 in enumerate(cases):
        if i in checked:
            continue

        group = [case1]
        name1 = case1.get('target_name', '')
        summary1 = case1.get('story_summary', '')
        period1 = case1.get('period', '')

        for j, case2 in enumerate(cases[i+1:], start=i+1):
            if j in checked:
                continue

            name2 = case2.get('target_name', '')
            summary2 = case2.get('story_summary', '')
            period2 = case2.get('period', '')

            # 完全一致
            if name1 == name2 and period1 == period2:
                group.append(case2)
                checked.add(j)
                continue

            # 名前の高類似
            name_sim = similarity(name1, name2)
            if name_sim >= name_threshold:
                # 期間が重複しているか
                period_overlap = similarity(period1, period2) > 0.5
                if period_overlap:
                    group.append(case2)
                    checked.add(j)
                    continue

            # サマリーの高類似（同一企業・異なる表現）
            if name_sim >= 0.6:
                summary_sim = similarity(summary1, summary2)
                if summary_sim >= summary_threshold:
                    group.append(case2)
                    checked.add(j)

        if len(group) > 1:
            duplicates.append(group)
            checked.add(i)

    return duplicates

def find_pattern_clusters(cases):
    """
    同一パターン（卦・爻・pattern_type）の事例クラスタを検出
    """
    clusters = defaultdict(list)

    for case in cases:
        yao_info = case.get('yao_analysis') or {}
        hex_id = yao_info.get('before_hexagram_id')
        yao = yao_info.get('before_yao_position')
        pattern = case.get('pattern_type')
        outcome = case.get('outcome')

        if hex_id and yao:
            key = (hex_id, yao, pattern, outcome)
            clusters[key].append(case)

    # 5件以上のクラスタを抽出
    large_clusters = {k: v for k, v in clusters.items() if len(v) >= 5}
    return large_clusters

def main():
    print(f"\n{'='*60}")
    print(f"意味的重複検出レポート")
    print(f"{'='*60}")

    cases = load_cases()
    print(f"\n総事例数: {len(cases):,}件")

    # 重複検出
    print(f"\n--- 1. 重複・高類似事例 ---")
    duplicates = find_duplicates(cases)

    if duplicates:
        print(f"⚠️ 重複グループ: {len(duplicates)}件")
        for i, group in enumerate(duplicates[:10], 1):
            names = [c.get('target_name', 'N/A')[:30] for c in group]
            periods = [c.get('period', 'N/A') for c in group]
            print(f"\n  グループ{i} ({len(group)}件):")
            for name, period in zip(names, periods):
                print(f"    - {name} ({period})")
        if len(duplicates) > 10:
            print(f"\n  ...他{len(duplicates)-10}グループ")
    else:
        print("✅ 重複なし")

    # パターンクラスタ
    print(f"\n--- 2. 同一パターン集中 ---")
    clusters = find_pattern_clusters(cases)

    if clusters:
        sorted_clusters = sorted(clusters.items(), key=lambda x: -len(x[1]))
        print(f"大規模クラスタ（5件以上）: {len(clusters)}件")
        for (hex_id, yao, pattern, outcome), group in sorted_clusters[:5]:
            print(f"\n  第{hex_id}卦{yao}爻 / {pattern} / {outcome}: {len(group)}件")
            for c in group[:3]:
                print(f"    - {c.get('target_name', 'N/A')[:40]}")
            if len(group) > 3:
                print(f"    ...他{len(group)-3}件")
    else:
        print("✅ 過度な集中なし")

    # 統計
    print(f"\n--- 3. 推奨アクション ---")
    if duplicates:
        total_dups = sum(len(g) - 1 for g in duplicates)
        print(f"  - 重複解消により最大 {total_dups} 件削減可能")
    if clusters:
        print(f"  - 大規模クラスタの事例は多様性確認を推奨")

    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
