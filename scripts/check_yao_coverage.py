#!/usr/bin/env python3
"""
64卦×6爻 網羅性チェックスクリプト

MCPメモリ「ワークフロー_64卦爻補充_2026-01」と連携
- 384マス（64卦×6爻）のカバレッジを計算
- 欠損マス（0件）と少量マス（1-5件）をリストアップ
- Phase別の進捗を表示
- JSON形式で結果を出力
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# 64卦マスター情報
HEXAGRAM_NAMES = {
    1: "乾為天", 2: "坤為地", 3: "水雷屯", 4: "山水蒙", 5: "水天需",
    6: "天水訟", 7: "地水師", 8: "水地比", 9: "風天小畜", 10: "天沢履",
    11: "地天泰", 12: "天地否", 13: "天火同人", 14: "火天大有", 15: "地山謙",
    16: "雷地豫", 17: "沢雷随", 18: "山風蠱", 19: "地沢臨", 20: "風地観",
    21: "火雷噬嗑", 22: "山火賁", 23: "山地剥", 24: "地雷復", 25: "天雷无妄",
    26: "山天大畜", 27: "山雷頤", 28: "沢風大過", 29: "坎為水", 30: "離為火",
    31: "沢山咸", 32: "雷風恒", 33: "天山遯", 34: "雷天大壮", 35: "火地晋",
    36: "地火明夷", 37: "風火家人", 38: "火沢睽", 39: "水山蹇", 40: "雷水解",
    41: "山沢損", 42: "風雷益", 43: "沢天夬", 44: "天風姤", 45: "沢地萃",
    46: "地風升", 47: "沢水困", 48: "水風井", 49: "沢火革", 50: "火風鼎",
    51: "震為雷", 52: "艮為山", 53: "風山漸", 54: "雷沢帰妹", 55: "雷火豊",
    56: "火山旅", 57: "巽為風", 58: "兌為沢", 59: "風水渙", 60: "水沢節",
    61: "風沢中孚", 62: "雷山小過", 63: "水火既済", 64: "火水未済"
}

# Phase定義（MCPメモリ「ワークフロー_64卦爻補充_2026-01」と連携）
PHASE_DEFINITIONS = {
    "Phase1": {
        "name": "最優先(5爻欠損卦4卦)",
        "description": "5爻が欠損している4卦",
        "hexagrams": [27, 35, 42, 50],
        "target_yao": [1, 2, 4, 5, 6],  # 3爻以外
    },
    "Phase2": {
        "name": "高優先(4爻欠損卦10卦)",
        "description": "4爻が欠損している10卦",
        "hexagrams": [14, 20, 21, 24, 25, 38, 44, 45, 56, 64],
        "target_yao": None,  # 卦ごとに異なる
    },
    "Phase3": {
        "name": "中優先(第4爻全般)",
        "description": "第4爻の欠損補充",
        "target_yao": [4],
    },
    "Phase4": {
        "name": "中優先(第1爻全般)",
        "description": "第1爻の欠損補充",
        "target_yao": [1],
    },
    "Phase5": {
        "name": "少量マス補強",
        "description": "1-5件のマスを補強",
        "threshold": 5,
    },
}


def load_cases(db_path: Path) -> List[dict]:
    """cases.jsonlを読み込み"""
    cases = []
    with open(db_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def analyze_yao_coverage(cases: List[dict]) -> Dict[int, Dict[int, int]]:
    """
    64卦×6爻のカバレッジを計算

    Returns:
        {hexagram_id: {yao_position: count}}
    """
    coverage = defaultdict(lambda: defaultdict(int))

    for case in cases:
        yao_analysis = case.get("yao_analysis")

        if yao_analysis:
            # yao_analysis.before_hexagram_id を優先使用
            hex_id = yao_analysis.get("before_hexagram_id")
            if hex_id is None:
                # フォールバック: case.hexagram_id
                hex_id = case.get("hexagram_id")

            yao_pos = yao_analysis.get("before_yao_position")

            if hex_id and yao_pos and 1 <= hex_id <= 64 and 1 <= yao_pos <= 6:
                coverage[hex_id][yao_pos] += 1

    return coverage


def calculate_statistics(coverage: Dict[int, Dict[int, int]]) -> dict:
    """
    統計情報を計算

    Returns:
        総マス数、カバー済み、欠損、少量マス等の統計
    """
    total_cells = 64 * 6  # 384
    covered = 0
    missing = 0
    low_count = 0  # 1-5件

    missing_cells = []  # [(hex_id, yao)]
    low_count_cells = []  # [(hex_id, yao, count)]

    for hex_id in range(1, 65):
        for yao in range(1, 7):
            count = coverage.get(hex_id, {}).get(yao, 0)

            if count == 0:
                missing += 1
                missing_cells.append((hex_id, yao))
            else:
                covered += 1
                if count <= 5:
                    low_count += 1
                    low_count_cells.append((hex_id, yao, count))

    return {
        "total_cells": total_cells,
        "covered": covered,
        "covered_percent": round(covered / total_cells * 100, 1),
        "missing": missing,
        "missing_percent": round(missing / total_cells * 100, 1),
        "low_count": low_count,
        "missing_cells": missing_cells,
        "low_count_cells": low_count_cells,
    }


def calculate_phase_progress(coverage: Dict[int, Dict[int, int]], stats: dict) -> dict:
    """
    Phase別の進捗を計算
    """
    progress = {}

    # Phase1: 5爻欠損卦4卦
    phase1_hexs = PHASE_DEFINITIONS["Phase1"]["hexagrams"]
    phase1_yaos = PHASE_DEFINITIONS["Phase1"]["target_yao"]
    phase1_total = len(phase1_hexs) * len(phase1_yaos)
    phase1_done = 0
    phase1_missing = []

    for hex_id in phase1_hexs:
        for yao in phase1_yaos:
            if coverage.get(hex_id, {}).get(yao, 0) > 0:
                phase1_done += 1
            else:
                phase1_missing.append((hex_id, yao))

    progress["Phase1"] = {
        "name": PHASE_DEFINITIONS["Phase1"]["name"],
        "total": phase1_total,
        "done": phase1_done,
        "percent": round(phase1_done / phase1_total * 100, 1) if phase1_total > 0 else 0,
        "missing": phase1_missing,
    }

    # Phase2: 4爻欠損卦10卦（各卦で欠損パターンが異なる）
    phase2_details = {
        14: [1, 2, 4, 5],
        20: [1, 4, 5, 6],
        21: [1, 4, 5, 6],
        24: [1, 4, 5, 6],
        25: [1, 4, 5, 6],
        38: [2, 4, 5, 6],
        44: [1, 2, 4, 5],
        45: [1, 4, 5, 6],
        56: [1, 2, 5, 6],
        64: [2, 4, 5, 6],
    }
    phase2_total = sum(len(yaos) for yaos in phase2_details.values())
    phase2_done = 0
    phase2_missing = []

    for hex_id, yaos in phase2_details.items():
        for yao in yaos:
            if coverage.get(hex_id, {}).get(yao, 0) > 0:
                phase2_done += 1
            else:
                phase2_missing.append((hex_id, yao))

    progress["Phase2"] = {
        "name": PHASE_DEFINITIONS["Phase2"]["name"],
        "total": phase2_total,
        "done": phase2_done,
        "percent": round(phase2_done / phase2_total * 100, 1) if phase2_total > 0 else 0,
        "missing": phase2_missing,
    }

    # Phase3: 第4爻全般
    phase3_total = 64
    phase3_done = 0
    phase3_missing = []

    for hex_id in range(1, 65):
        if coverage.get(hex_id, {}).get(4, 0) > 0:
            phase3_done += 1
        else:
            phase3_missing.append((hex_id, 4))

    progress["Phase3"] = {
        "name": PHASE_DEFINITIONS["Phase3"]["name"],
        "total": phase3_total,
        "done": phase3_done,
        "percent": round(phase3_done / phase3_total * 100, 1),
        "missing": phase3_missing,
    }

    # Phase4: 第1爻全般
    phase4_total = 64
    phase4_done = 0
    phase4_missing = []

    for hex_id in range(1, 65):
        if coverage.get(hex_id, {}).get(1, 0) > 0:
            phase4_done += 1
        else:
            phase4_missing.append((hex_id, 1))

    progress["Phase4"] = {
        "name": PHASE_DEFINITIONS["Phase4"]["name"],
        "total": phase4_total,
        "done": phase4_done,
        "percent": round(phase4_done / phase4_total * 100, 1),
        "missing": phase4_missing,
    }

    # Phase5: 少量マス補強（1-5件）
    progress["Phase5"] = {
        "name": PHASE_DEFINITIONS["Phase5"]["name"],
        "total": stats["low_count"],
        "done": 0,  # 補強完了かどうかは6件以上で判定
        "percent": 0,
        "cells": stats["low_count_cells"],
    }

    return progress


def get_hexagram_summary(coverage: Dict[int, Dict[int, int]]) -> List[dict]:
    """
    卦ごとの欠損サマリを作成
    """
    summaries = []

    for hex_id in range(1, 65):
        hex_name = HEXAGRAM_NAMES.get(hex_id, f"卦{hex_id}")
        hex_coverage = coverage.get(hex_id, {})

        missing_yaos = []
        present_yaos = []

        for yao in range(1, 7):
            count = hex_coverage.get(yao, 0)
            if count == 0:
                missing_yaos.append(yao)
            else:
                present_yaos.append((yao, count))

        if missing_yaos:
            summaries.append({
                "hexagram_id": hex_id,
                "hexagram_name": hex_name,
                "missing_count": len(missing_yaos),
                "missing_yaos": missing_yaos,
                "present_yaos": present_yaos,
            })

    # 欠損数が多い順にソート
    summaries.sort(key=lambda x: x["missing_count"], reverse=True)

    return summaries


def print_report(stats: dict, phase_progress: dict, hex_summaries: List[dict]):
    """
    コンソールにレポートを出力
    """
    print("=" * 60)
    print("=== 64卦×6爻 網羅性レポート ===")
    print("=" * 60)

    print(f"\n【総合統計】")
    print(f"総マス: {stats['total_cells']}")
    print(f"カバー済み: {stats['covered']} ({stats['covered_percent']}%)")
    print(f"欠損: {stats['missing']} ({stats['missing_percent']}%)")
    print(f"少量(1-5件): {stats['low_count']}マス")

    print("\n" + "=" * 60)
    print("=== Phase別進捗 ===")
    print("=" * 60)

    for phase_name, progress in phase_progress.items():
        if phase_name == "Phase5":
            print(f"\n{phase_name} ({progress['name']}): {progress['total']}マス要補強")
        else:
            print(f"\n{phase_name} ({progress['name']}): {progress['done']}/{progress['total']}マス完了 ({progress['percent']}%)")
            if progress['missing'] and len(progress['missing']) <= 10:
                missing_str = ", ".join([f"{HEXAGRAM_NAMES.get(h, h)}-{y}爻" for h, y in progress['missing']])
                print(f"  欠損: {missing_str}")

    print("\n" + "=" * 60)
    print("=== 欠損マス一覧 (欠損数順) ===")
    print("=" * 60)

    # 欠損が多い卦のみ表示
    for summary in hex_summaries[:20]:  # 上位20件
        if summary["missing_count"] == 0:
            continue

        hex_id = summary["hexagram_id"]
        hex_name = summary["hexagram_name"]
        missing = summary["missing_yaos"]
        present = summary["present_yaos"]

        missing_str = ",".join([f"{y}爻" for y in missing])
        present_str = ", ".join([f"{y}爻:{c}件" for y, c in present]) if present else "なし"

        print(f"\n{hex_id}. {hex_name}: 欠損[{missing_str}]")
        print(f"   現状: [{present_str}]")

    if len(hex_summaries) > 20:
        remaining = len([s for s in hex_summaries[20:] if s["missing_count"] > 0])
        if remaining > 0:
            print(f"\n... 他 {remaining}卦に欠損あり")


def save_report(stats: dict, phase_progress: dict, hex_summaries: List[dict],
                coverage: Dict[int, Dict[int, int]], output_path: Path):
    """
    JSON形式でレポートを保存
    """
    # coverage を JSON serializable な形式に変換
    coverage_serializable = {}
    for hex_id, yao_counts in coverage.items():
        coverage_serializable[str(hex_id)] = dict(yao_counts)

    # phase_progress を JSON serializable な形式に変換
    phase_progress_serializable = {}
    for phase_name, progress in phase_progress.items():
        progress_copy = progress.copy()
        if "missing" in progress_copy:
            progress_copy["missing"] = [
                {"hexagram_id": h, "yao": y}
                for h, y in progress_copy["missing"]
            ]
        if "cells" in progress_copy:
            progress_copy["cells"] = [
                {"hexagram_id": h, "yao": y, "count": c}
                for h, y, c in progress_copy["cells"]
            ]
        phase_progress_serializable[phase_name] = progress_copy

    report = {
        "generated_at": datetime.now().isoformat(),
        "workflow": "ワークフロー_64卦爻補充_2026-01",
        "statistics": {
            "total_cells": stats["total_cells"],
            "covered": stats["covered"],
            "covered_percent": stats["covered_percent"],
            "missing": stats["missing"],
            "missing_percent": stats["missing_percent"],
            "low_count_cells": stats["low_count"],
        },
        "phase_progress": phase_progress_serializable,
        "hexagram_summaries": hex_summaries,
        "coverage_matrix": coverage_serializable,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] レポートを保存しました: {output_path}")


def main():
    base_dir = Path(__file__).parent.parent
    db_path = base_dir / "data" / "raw" / "cases.jsonl"
    output_path = base_dir / "data" / "diagnostic" / "yao_coverage_report.json"

    print("64卦×6爻 網羅性チェックを実行中...\n")

    # データ読み込み
    cases = load_cases(db_path)
    print(f"読み込んだ事例数: {len(cases)}")

    # カバレッジ分析
    coverage = analyze_yao_coverage(cases)

    # 統計計算
    stats = calculate_statistics(coverage)

    # Phase別進捗
    phase_progress = calculate_phase_progress(coverage, stats)

    # 卦ごとのサマリ
    hex_summaries = get_hexagram_summary(coverage)

    # レポート出力
    print_report(stats, phase_progress, hex_summaries)

    # JSON保存
    save_report(stats, phase_progress, hex_summaries, coverage, output_path)

    print("\n" + "=" * 60)
    print("完了")


if __name__ == "__main__":
    main()
