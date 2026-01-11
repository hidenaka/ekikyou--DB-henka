#!/usr/bin/env python3
"""
予測乖離ケースの分析スクリプト

乖離パターンを特定し、診断ロジックの改善点を明らかにする
"""

import json
from pathlib import Path
from collections import defaultdict, Counter

BASE_DIR = Path(__file__).parent.parent
CASES_FILE = BASE_DIR / "data" / "raw" / "cases.jsonl"


def load_cases():
    """ケースデータを読み込み"""
    cases = []
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def analyze_errors(cases):
    """乖離ケースを分析"""

    # 分類用の辞書
    miss_cases = []  # 乖離ケース
    exact_cases = []  # 完全一致ケース

    # パターン別の統計
    stats = {
        "by_before_state": defaultdict(lambda: {"total": 0, "miss": 0, "exact": 0}),
        "by_action_type": defaultdict(lambda: {"total": 0, "miss": 0, "exact": 0}),
        "by_yao_position": defaultdict(lambda: {"total": 0, "miss": 0, "exact": 0}),
        "by_pattern_type": defaultdict(lambda: {"total": 0, "miss": 0, "exact": 0}),
        "by_outcome": defaultdict(lambda: {"total": 0, "miss": 0, "exact": 0}),
        "by_predicted": defaultdict(lambda: {"total": 0, "miss": 0, "exact": 0}),
        "by_state_action": defaultdict(lambda: {"total": 0, "miss": 0, "exact": 0}),
        "by_yao_action": defaultdict(lambda: {"total": 0, "miss": 0, "exact": 0}),
    }

    # 乖離パターンの詳細
    miss_patterns = defaultdict(list)

    for case in cases:
        yao = case.get("yao_analysis", {})
        accuracy = yao.get("prediction_analysis", {}).get("accuracy", "miss")

        before_state = case.get("before_state", "")
        action_type = case.get("action_type", "")
        yao_position = yao.get("before_yao_position", 0)
        pattern_type = case.get("pattern_type", "")
        actual = yao.get("actual_outcome", "")
        predicted = yao.get("predicted_outcome", "")

        # 基本統計
        stats["by_before_state"][before_state]["total"] += 1
        stats["by_action_type"][action_type]["total"] += 1
        stats["by_yao_position"][yao_position]["total"] += 1
        stats["by_pattern_type"][pattern_type]["total"] += 1
        stats["by_outcome"][actual]["total"] += 1
        stats["by_predicted"][predicted]["total"] += 1

        state_action = f"{before_state} + {action_type}"
        stats["by_state_action"][state_action]["total"] += 1

        yao_action = f"{yao_position}爻 + {action_type}"
        stats["by_yao_action"][yao_action]["total"] += 1

        if accuracy == "miss":
            miss_cases.append(case)
            stats["by_before_state"][before_state]["miss"] += 1
            stats["by_action_type"][action_type]["miss"] += 1
            stats["by_yao_position"][yao_position]["miss"] += 1
            stats["by_pattern_type"][pattern_type]["miss"] += 1
            stats["by_outcome"][actual]["miss"] += 1
            stats["by_predicted"][predicted]["miss"] += 1
            stats["by_state_action"][state_action]["miss"] += 1
            stats["by_yao_action"][yao_action]["miss"] += 1

            # 乖離パターンの記録
            miss_key = f"{predicted} → {actual}"
            miss_patterns[miss_key].append({
                "name": case.get("target_name"),
                "before_state": before_state,
                "action_type": action_type,
                "yao_position": yao_position,
                "pattern_type": pattern_type,
            })

        elif accuracy == "exact":
            exact_cases.append(case)
            stats["by_before_state"][before_state]["exact"] += 1
            stats["by_action_type"][action_type]["exact"] += 1
            stats["by_yao_position"][yao_position]["exact"] += 1
            stats["by_pattern_type"][pattern_type]["exact"] += 1
            stats["by_outcome"][actual]["exact"] += 1
            stats["by_predicted"][predicted]["exact"] += 1
            stats["by_state_action"][state_action]["exact"] += 1
            stats["by_yao_action"][yao_action]["exact"] += 1

    return miss_cases, exact_cases, stats, miss_patterns


def print_analysis(miss_cases, exact_cases, stats, miss_patterns):
    """分析結果を出力"""

    total = len(miss_cases) + len(exact_cases)

    print("=" * 60)
    print("予測乖離分析レポート")
    print("=" * 60)

    print(f"\n総ケース数: {total}")
    print(f"乖離ケース: {len(miss_cases)} ({len(miss_cases)/total*100:.1f}%)")

    # 乖離パターンの分析
    print("\n" + "-" * 40)
    print("【乖離パターン】予測 → 実際")
    print("-" * 40)
    for pattern, cases_list in sorted(miss_patterns.items(), key=lambda x: -len(x[1])):
        print(f"\n{pattern}: {len(cases_list)}件")
        # 詳細を3件まで表示
        for c in cases_list[:3]:
            print(f"  - {c['name']}: {c['before_state']} / {c['action_type']} / {c['yao_position']}爻")

    # before_state別の分析
    print("\n" + "-" * 40)
    print("【before_state別】乖離率")
    print("-" * 40)
    for state, data in sorted(stats["by_before_state"].items(), key=lambda x: -x[1]["miss"]/max(x[1]["total"],1)):
        if data["total"] > 0:
            miss_rate = data["miss"] / data["total"] * 100
            print(f"{state}: {data['miss']}/{data['total']} ({miss_rate:.1f}%)")

    # action_type別の分析
    print("\n" + "-" * 40)
    print("【action_type別】乖離率")
    print("-" * 40)
    for action, data in sorted(stats["by_action_type"].items(), key=lambda x: -x[1]["miss"]/max(x[1]["total"],1)):
        if data["total"] > 0:
            miss_rate = data["miss"] / data["total"] * 100
            print(f"{action}: {data['miss']}/{data['total']} ({miss_rate:.1f}%)")

    # yao_position別の分析
    print("\n" + "-" * 40)
    print("【爻位別】乖離率")
    print("-" * 40)
    for yao, data in sorted(stats["by_yao_position"].items()):
        if data["total"] > 0:
            miss_rate = data["miss"] / data["total"] * 100
            exact_rate = data["exact"] / data["total"] * 100
            print(f"{yao}爻: 乖離 {data['miss']}/{data['total']} ({miss_rate:.1f}%) / 一致 {exact_rate:.1f}%")

    # pattern_type別の分析
    print("\n" + "-" * 40)
    print("【pattern_type別】乖離率")
    print("-" * 40)
    for pattern, data in sorted(stats["by_pattern_type"].items(), key=lambda x: -x[1]["miss"]/max(x[1]["total"],1)):
        if data["total"] > 0:
            miss_rate = data["miss"] / data["total"] * 100
            print(f"{pattern}: {data['miss']}/{data['total']} ({miss_rate:.1f}%)")

    # 予測結果別の分析
    print("\n" + "-" * 40)
    print("【予測結果別】乖離率")
    print("-" * 40)
    for pred, data in sorted(stats["by_predicted"].items(), key=lambda x: -x[1]["miss"]/max(x[1]["total"],1)):
        if data["total"] > 0:
            miss_rate = data["miss"] / data["total"] * 100
            print(f"予測{pred}: {data['miss']}/{data['total']} ({miss_rate:.1f}%)")

    # 実際の結果別の分析
    print("\n" + "-" * 40)
    print("【実際の結果別】乖離率")
    print("-" * 40)
    for outcome, data in sorted(stats["by_outcome"].items(), key=lambda x: -x[1]["miss"]/max(x[1]["total"],1)):
        if data["total"] > 0:
            miss_rate = data["miss"] / data["total"] * 100
            print(f"実際{outcome}: {data['miss']}/{data['total']} ({miss_rate:.1f}%)")

    # 爻位×行動の組み合わせで乖離率が高いもの
    print("\n" + "-" * 40)
    print("【爻位×行動】乖離率が高い組み合わせ (TOP 15)")
    print("-" * 40)
    yao_action_sorted = sorted(
        stats["by_yao_action"].items(),
        key=lambda x: (-x[1]["miss"]/max(x[1]["total"],1), -x[1]["total"])
    )
    for combo, data in yao_action_sorted[:15]:
        if data["total"] >= 10:  # 10件以上のもののみ
            miss_rate = data["miss"] / data["total"] * 100
            print(f"{combo}: {data['miss']}/{data['total']} ({miss_rate:.1f}%)")

    # 状態×行動の組み合わせで乖離率が高いもの
    print("\n" + "-" * 40)
    print("【状態×行動】乖離率が高い組み合わせ (TOP 15)")
    print("-" * 40)
    state_action_sorted = sorted(
        stats["by_state_action"].items(),
        key=lambda x: (-x[1]["miss"]/max(x[1]["total"],1), -x[1]["total"])
    )
    for combo, data in state_action_sorted[:15]:
        if data["total"] >= 10:
            miss_rate = data["miss"] / data["total"] * 100
            print(f"{combo}: {data['miss']}/{data['total']} ({miss_rate:.1f}%)")


def identify_improvement_areas(stats, miss_patterns):
    """改善すべき領域を特定"""

    print("\n" + "=" * 60)
    print("改善提案")
    print("=" * 60)

    # 1. 爻位診断の問題
    print("\n【1. 爻位診断の問題】")
    for yao, data in stats["by_yao_position"].items():
        if data["total"] > 0:
            miss_rate = data["miss"] / data["total"] * 100
            if miss_rate > 50:
                print(f"  - {yao}爻: 乖離率{miss_rate:.1f}% → 診断ロジックの見直しが必要")

    # 2. 適合性マトリクスの問題
    print("\n【2. 適合性マトリクスの問題】")
    # 予測がFailureだが実際はSuccessのパターン
    for pattern, cases_list in miss_patterns.items():
        if "Failure → Success" in pattern and len(cases_list) > 50:
            print(f"  - {pattern}: {len(cases_list)}件 → スコアが厳しすぎる可能性")
        if "Success → Failure" in pattern and len(cases_list) > 50:
            print(f"  - {pattern}: {len(cases_list)}件 → スコアが甘すぎる可能性")

    # 3. 特定の行動タイプの問題
    print("\n【3. 行動タイプ別の問題】")
    for action, data in stats["by_action_type"].items():
        if data["total"] > 0:
            miss_rate = data["miss"] / data["total"] * 100
            if miss_rate > 50:
                print(f"  - {action}: 乖離率{miss_rate:.1f}% → 適合性スコアの調整が必要")

    # 4. before_state と爻位のマッピング問題
    print("\n【4. 状態→爻位マッピングの問題】")
    for state, data in stats["by_before_state"].items():
        if data["total"] > 0:
            miss_rate = data["miss"] / data["total"] * 100
            if miss_rate > 50:
                print(f"  - {state}: 乖離率{miss_rate:.1f}% → 爻位マッピングの見直しが必要")


def generate_improvement_config(stats, miss_patterns):
    """改善用の設定を生成"""

    improvements = {
        "yao_position_adjustments": {},
        "compatibility_adjustments": {},
        "state_to_yao_mapping": {}
    }

    # 1. 爻位診断の調整提案
    # before_stateごとの最適な爻位を、成功ケースから逆算
    for state, data in stats["by_before_state"].items():
        if data["exact"] > 0:
            # この状態で成功しているケースの特徴を分析
            pass

    # 2. 適合性スコアの調整提案
    # 乖離パターンから調整
    for pattern, cases_list in miss_patterns.items():
        if len(cases_list) > 100:
            # 頻出する乖離パターンに基づいて調整
            if "Failure → Success" in pattern:
                # スコアを下げる（緩くする）必要がある組み合わせ
                for c in cases_list:
                    key = f"{c['yao_position']}_{c['action_type']}"
                    if key not in improvements["compatibility_adjustments"]:
                        improvements["compatibility_adjustments"][key] = {
                            "current_issue": "too_strict",
                            "count": 0
                        }
                    improvements["compatibility_adjustments"][key]["count"] += 1

            elif "Success → Failure" in pattern:
                # スコアを上げる（厳しくする）必要がある組み合わせ
                for c in cases_list:
                    key = f"{c['yao_position']}_{c['action_type']}"
                    if key not in improvements["compatibility_adjustments"]:
                        improvements["compatibility_adjustments"][key] = {
                            "current_issue": "too_lenient",
                            "count": 0
                        }
                    improvements["compatibility_adjustments"][key]["count"] += 1

    return improvements


def main():
    print("ケースデータ読み込み中...")
    cases = load_cases()
    print(f"{len(cases)}件のケースを読み込みました")

    print("\n乖離分析実行中...")
    miss_cases, exact_cases, stats, miss_patterns = analyze_errors(cases)

    print_analysis(miss_cases, exact_cases, stats, miss_patterns)

    identify_improvement_areas(stats, miss_patterns)

    improvements = generate_improvement_config(stats, miss_patterns)

    # 改善設定を保存
    output_file = BASE_DIR / "data" / "mappings" / "improvement_suggestions.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(improvements, f, ensure_ascii=False, indent=2)
    print(f"\n改善提案を {output_file} に保存しました")


if __name__ == "__main__":
    main()
