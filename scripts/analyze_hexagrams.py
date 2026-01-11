#!/usr/bin/env python3
"""
八卦の網羅性を分析するスクリプト
"""
import json
from pathlib import Path
from collections import defaultdict
from schema_v3 import Case

def main():
    db_path = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"

    # 八卦の出現回数を記録
    before_hex_count = defaultdict(int)
    trigger_hex_count = defaultdict(int)
    action_hex_count = defaultdict(int)
    after_hex_count = defaultdict(int)

    # 八卦の組み合わせパターン（変化の流れ）
    transition_patterns = defaultdict(int)  # (before, trigger, action, after)

    total = 0

    with open(db_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)
            case = Case(**data)
            total += 1

            # 各ポジションでの八卦をカウント
            before_hex_count[case.before_hex] += 1
            trigger_hex_count[case.trigger_hex] += 1
            action_hex_count[case.action_hex] += 1
            after_hex_count[case.after_hex] += 1

            # 変化パターンをカウント
            pattern = (case.before_hex, case.trigger_hex, case.action_hex, case.after_hex)
            transition_patterns[pattern] += 1

    print("=== 八卦の網羅性分析 ===\n")
    print(f"総事例数: {total}\n")

    # 八卦の説明
    hex_description = {
        "乾": "天・創造・剛健",
        "坤": "地・受容・柔順",
        "震": "雷・動き・奮起",
        "巽": "風・浸透・柔軟",
        "坎": "水・危険・困難",
        "離": "火・明知・分離",
        "艮": "山・止まる・待機",
        "兌": "沢・喜び・和悦"
    }

    # 各ポジションでの八卦分布
    print("【before_hex（初期状態）の分布】")
    for hex_name in ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]:
        count = before_hex_count.get(hex_name, 0)
        pct = (count / total * 100) if total > 0 else 0
        desc = hex_description[hex_name]
        print(f"  {hex_name}（{desc}）: {count:3d}件 ({pct:5.1f}%)")

    print("\n【trigger_hex（トリガー）の分布】")
    for hex_name in ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]:
        count = trigger_hex_count.get(hex_name, 0)
        pct = (count / total * 100) if total > 0 else 0
        desc = hex_description[hex_name]
        print(f"  {hex_name}（{desc}）: {count:3d}件 ({pct:5.1f}%)")

    print("\n【action_hex（行動）の分布】")
    for hex_name in ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]:
        count = action_hex_count.get(hex_name, 0)
        pct = (count / total * 100) if total > 0 else 0
        desc = hex_description[hex_name]
        print(f"  {hex_name}（{desc}）: {count:3d}件 ({pct:5.1f}%)")

    print("\n【after_hex（結果）の分布】")
    for hex_name in ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]:
        count = after_hex_count.get(hex_name, 0)
        pct = (count / total * 100) if total > 0 else 0
        desc = hex_description[hex_name]
        print(f"  {hex_name}（{desc}）: {count:3d}件 ({pct:5.1f}%)")

    # 各八卦がどこかで使われているかチェック
    print("\n【八卦の使用状況サマリー】")
    all_hexagrams = ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]
    for hex_name in all_hexagrams:
        before = before_hex_count.get(hex_name, 0)
        trigger = trigger_hex_count.get(hex_name, 0)
        action = action_hex_count.get(hex_name, 0)
        after = after_hex_count.get(hex_name, 0)
        total_usage = before + trigger + action + after
        desc = hex_description[hex_name]
        print(f"  {hex_name}（{desc}）: 合計{total_usage:4d}回使用")
        print(f"    Before: {before:3d}, Trigger: {trigger:3d}, Action: {action:3d}, After: {after:3d}")

    # よく見られる変化パターン（上位20件）
    print("\n【頻出の変化パターン（上位20件）】")
    sorted_patterns = sorted(transition_patterns.items(), key=lambda x: x[1], reverse=True)
    for i, (pattern, count) in enumerate(sorted_patterns[:20], 1):
        before, trigger, action, after = pattern
        print(f"{i:2d}. {before} → {trigger} → {action} → {after}: {count}件")

    # 使用されていない組み合わせを確認
    print("\n【網羅性チェック】")

    # 各ポジションで使われていない八卦があるかチェック
    unused_before = [h for h in all_hexagrams if before_hex_count.get(h, 0) == 0]
    unused_trigger = [h for h in all_hexagrams if trigger_hex_count.get(h, 0) == 0]
    unused_action = [h for h in all_hexagrams if action_hex_count.get(h, 0) == 0]
    unused_after = [h for h in all_hexagrams if after_hex_count.get(h, 0) == 0]

    if unused_before:
        print(f"  ⚠️  before_hexで未使用: {', '.join(unused_before)}")
    else:
        print(f"  ✅ before_hexは全八卦を網羅")

    if unused_trigger:
        print(f"  ⚠️  trigger_hexで未使用: {', '.join(unused_trigger)}")
    else:
        print(f"  ✅ trigger_hexは全八卦を網羅")

    if unused_action:
        print(f"  ⚠️  action_hexで未使用: {', '.join(unused_action)}")
    else:
        print(f"  ✅ action_hexは全八卦を網羅")

    if unused_after:
        print(f"  ⚠️  after_hexで未使用: {', '.join(unused_after)}")
    else:
        print(f"  ✅ after_hexは全八卦を網羅")

    # バランスチェック（極端に少ない八卦）
    print("\n【バランスチェック（使用回数が少ない八卦）】")
    threshold = total * 0.05  # 全体の5%未満

    for position, counter in [
        ("before_hex", before_hex_count),
        ("trigger_hex", trigger_hex_count),
        ("action_hex", action_hex_count),
        ("after_hex", after_hex_count)
    ]:
        underused = [(h, counter.get(h, 0)) for h in all_hexagrams if counter.get(h, 0) < threshold]
        if underused:
            print(f"\n  {position}で使用が少ない八卦（全体の5%未満）:")
            for hex_name, count in underused:
                pct = (count / total * 100) if total > 0 else 0
                desc = hex_description[hex_name]
                print(f"    {hex_name}（{desc}）: {count}件 ({pct:.1f}%)")

if __name__ == "__main__":
    main()
