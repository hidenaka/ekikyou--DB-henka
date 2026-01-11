#!/usr/bin/env python3
"""
予測エンジン用統計データ生成スクリプト
cases.jsonl を分析し、64卦別成功率、卦×アクションマトリクス、爻×結果相関を生成
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Any

# パス設定
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ANALYSIS_DIR = DATA_DIR / "analysis"
HEXAGRAM_DIR = DATA_DIR / "hexagrams"

# 分析ディレクトリ作成
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def load_cases() -> list[dict]:
    """cases.jsonl を読み込む"""
    cases = []
    with open(RAW_DIR / "cases.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def load_hexagram_master() -> dict:
    """64卦マスターを読み込む"""
    with open(HEXAGRAM_DIR / "hexagram_master.json", "r", encoding="utf-8") as f:
        return json.load(f)


def is_success(outcome: str) -> bool:
    """成功判定"""
    success_outcomes = {"Success", "PartialSuccess", "V字回復・大成功"}
    return outcome in success_outcomes


def is_failure(outcome: str) -> bool:
    """失敗判定"""
    failure_outcomes = {"Failure", "Mixed", "Unknown"}
    return outcome in failure_outcomes


def extract_hexagram_id(classical_hexagram: str) -> int | None:
    """古典卦名から卦番号を抽出 (例: "52_艮" -> 52, "地雷復" -> 24)"""
    if not classical_hexagram:
        return None

    # "52_艮" 形式
    if "_" in classical_hexagram:
        try:
            return int(classical_hexagram.split("_")[0])
        except ValueError:
            pass

    # 数字のみの場合
    try:
        return int(classical_hexagram)
    except (ValueError, TypeError):
        pass

    return None


def analyze_hexagram_success_rate(cases: list[dict], hexagram_master: dict) -> dict:
    """64卦別の成功率を分析"""
    stats = defaultdict(lambda: {"total": 0, "success": 0, "failure": 0, "other": 0})

    for case in cases:
        # yao_analysis から before_hexagram_id を取得
        yao = case.get("yao_analysis", {})
        hex_id = yao.get("before_hexagram_id")

        if not hex_id:
            # classical_before_hexagram からも試行
            hex_id = extract_hexagram_id(case.get("classical_before_hexagram"))

        if hex_id and 1 <= hex_id <= 64:
            hex_key = str(hex_id)
            stats[hex_key]["total"] += 1

            outcome = case.get("outcome", "")
            if is_success(outcome):
                stats[hex_key]["success"] += 1
            elif is_failure(outcome):
                stats[hex_key]["failure"] += 1
            else:
                stats[hex_key]["other"] += 1

    # 成功率を計算し、卦名を追加
    result = {}
    for hex_id in range(1, 65):
        hex_key = str(hex_id)
        s = stats.get(hex_key, {"total": 0, "success": 0, "failure": 0, "other": 0})

        hex_info = hexagram_master.get(hex_key, {})
        name = hex_info.get("name", f"卦{hex_id}")

        total = s["total"]
        success = s["success"]
        failure = s["failure"]

        rate = round(success / total, 4) if total > 0 else 0.0

        result[hex_key] = {
            "id": hex_id,
            "name": name,
            "keyword": hex_info.get("keyword", ""),
            "total": total,
            "success": success,
            "failure": failure,
            "other": s["other"],
            "success_rate": rate
        }

    return result


def analyze_hexagram_action_matrix(cases: list[dict], hexagram_master: dict) -> dict:
    """卦×action_type別の成功率を分析"""
    matrix = defaultdict(lambda: {"total": 0, "success": 0, "failure": 0})

    for case in cases:
        yao = case.get("yao_analysis", {})
        hex_id = yao.get("before_hexagram_id")

        if not hex_id:
            hex_id = extract_hexagram_id(case.get("classical_before_hexagram"))

        action_type = case.get("action_type", "")

        if hex_id and action_type:
            key = f"{hex_id}_{action_type}"
            matrix[key]["total"] += 1

            outcome = case.get("outcome", "")
            if is_success(outcome):
                matrix[key]["success"] += 1
            elif is_failure(outcome):
                matrix[key]["failure"] += 1

    # 成功率を計算
    result = {}
    for key, data in matrix.items():
        total = data["total"]
        success = data["success"]
        rate = round(success / total, 4) if total > 0 else 0.0

        parts = key.split("_", 1)
        hex_id = int(parts[0])
        action = parts[1] if len(parts) > 1 else ""

        hex_info = hexagram_master.get(str(hex_id), {})

        result[key] = {
            "hexagram_id": hex_id,
            "hexagram_name": hex_info.get("name", f"卦{hex_id}"),
            "action_type": action,
            "total": total,
            "success": success,
            "failure": data["failure"],
            "success_rate": rate
        }

    return result


def analyze_yao_outcome_correlation(cases: list[dict]) -> dict:
    """爻位置(1-6)×結果の相関を分析"""
    yao_stats = defaultdict(lambda: {"total": 0, "success": 0, "failure": 0})
    yao_stage_stats = defaultdict(lambda: {"total": 0, "success": 0, "failure": 0})
    yao_stance_stats = defaultdict(lambda: {"total": 0, "success": 0, "failure": 0})

    for case in cases:
        yao = case.get("yao_analysis", {})
        yao_position = yao.get("before_yao_position")
        yao_stage = yao.get("yao_stage", "")
        yao_stance = yao.get("yao_basic_stance", "")

        outcome = case.get("outcome", "")

        if yao_position and 1 <= yao_position <= 6:
            yao_stats[yao_position]["total"] += 1
            if is_success(outcome):
                yao_stats[yao_position]["success"] += 1
            elif is_failure(outcome):
                yao_stats[yao_position]["failure"] += 1

        if yao_stage:
            yao_stage_stats[yao_stage]["total"] += 1
            if is_success(outcome):
                yao_stage_stats[yao_stage]["success"] += 1
            elif is_failure(outcome):
                yao_stage_stats[yao_stage]["failure"] += 1

        if yao_stance:
            yao_stance_stats[yao_stance]["total"] += 1
            if is_success(outcome):
                yao_stance_stats[yao_stance]["success"] += 1
            elif is_failure(outcome):
                yao_stance_stats[yao_stance]["failure"] += 1

    # 爻位置別成功率
    position_result = {}
    position_descriptions = {
        1: "発芽期・始動期",
        2: "成長期・準備期",
        3: "転換期・岐路",
        4: "拡大期・飛躍期",
        5: "成熟期・完成期",
        6: "終末期・変革期"
    }

    for pos in range(1, 7):
        data = yao_stats.get(pos, {"total": 0, "success": 0, "failure": 0})
        total = data["total"]
        success = data["success"]
        rate = round(success / total, 4) if total > 0 else 0.0

        position_result[str(pos)] = {
            "position": pos,
            "description": position_descriptions.get(pos, ""),
            "total": total,
            "success": success,
            "failure": data["failure"],
            "success_rate": rate
        }

    # ステージ別成功率
    stage_result = {}
    for stage, data in yao_stage_stats.items():
        total = data["total"]
        success = data["success"]
        rate = round(success / total, 4) if total > 0 else 0.0

        stage_result[stage] = {
            "stage": stage,
            "total": total,
            "success": success,
            "failure": data["failure"],
            "success_rate": rate
        }

    # スタンス別成功率
    stance_result = {}
    for stance, data in yao_stance_stats.items():
        total = data["total"]
        success = data["success"]
        rate = round(success / total, 4) if total > 0 else 0.0

        stance_result[stance] = {
            "stance": stance,
            "total": total,
            "success": success,
            "failure": data["failure"],
            "success_rate": rate
        }

    return {
        "by_position": position_result,
        "by_stage": stage_result,
        "by_stance": stance_result
    }


def analyze_pattern_success_rate(cases: list[dict]) -> dict:
    """pattern_type別の成功率を分析"""
    pattern_stats = defaultdict(lambda: {"total": 0, "success": 0, "failure": 0})

    for case in cases:
        pattern = case.get("pattern_type", "")
        if pattern:
            pattern_stats[pattern]["total"] += 1

            outcome = case.get("outcome", "")
            if is_success(outcome):
                pattern_stats[pattern]["success"] += 1
            elif is_failure(outcome):
                pattern_stats[pattern]["failure"] += 1

    result = {}
    for pattern, data in pattern_stats.items():
        total = data["total"]
        success = data["success"]
        rate = round(success / total, 4) if total > 0 else 0.0

        result[pattern] = {
            "pattern": pattern,
            "total": total,
            "success": success,
            "failure": data["failure"],
            "success_rate": rate
        }

    return result


def generate_prediction_base_stats(
    hexagram_stats: dict,
    action_matrix: dict,
    yao_correlation: dict,
    pattern_stats: dict,
    cases: list[dict]
) -> dict:
    """予測エンジン用の統合統計データを生成"""

    # 全体統計
    total_cases = len(cases)
    success_count = sum(1 for c in cases if is_success(c.get("outcome", "")))
    failure_count = sum(1 for c in cases if is_failure(c.get("outcome", "")))

    # 最も成功率の高い卦 TOP10
    hex_ranking = sorted(
        [(k, v) for k, v in hexagram_stats.items() if v["total"] >= 10],
        key=lambda x: x[1]["success_rate"],
        reverse=True
    )[:10]

    # 最も成功率の低い卦 TOP10
    hex_ranking_low = sorted(
        [(k, v) for k, v in hexagram_stats.items() if v["total"] >= 10],
        key=lambda x: x[1]["success_rate"]
    )[:10]

    # 最も成功率の高い卦×アクション組み合わせ TOP20
    action_ranking = sorted(
        [(k, v) for k, v in action_matrix.items() if v["total"] >= 5],
        key=lambda x: x[1]["success_rate"],
        reverse=True
    )[:20]

    return {
        "meta": {
            "generated_at": str(Path(__file__).stat().st_mtime),
            "total_cases": total_cases,
            "cases_with_yao": sum(1 for c in cases if c.get("yao_analysis")),
        },
        "overall": {
            "total": total_cases,
            "success": success_count,
            "failure": failure_count,
            "success_rate": round(success_count / total_cases, 4) if total_cases > 0 else 0.0
        },
        "top_hexagrams": [
            {
                "rank": i + 1,
                "hexagram_id": int(k),
                "name": v["name"],
                "keyword": v["keyword"],
                "total": v["total"],
                "success_rate": v["success_rate"]
            }
            for i, (k, v) in enumerate(hex_ranking)
        ],
        "low_hexagrams": [
            {
                "rank": i + 1,
                "hexagram_id": int(k),
                "name": v["name"],
                "keyword": v["keyword"],
                "total": v["total"],
                "success_rate": v["success_rate"]
            }
            for i, (k, v) in enumerate(hex_ranking_low)
        ],
        "top_hexagram_actions": [
            {
                "rank": i + 1,
                "hexagram_id": v["hexagram_id"],
                "hexagram_name": v["hexagram_name"],
                "action_type": v["action_type"],
                "total": v["total"],
                "success_rate": v["success_rate"]
            }
            for i, (k, v) in enumerate(action_ranking)
        ],
        "yao_position_summary": yao_correlation["by_position"],
        "pattern_summary": pattern_stats
    }


def main():
    print("=== 予測エンジン用統計データ生成 ===\n")

    # データ読み込み
    print("1. データ読み込み中...")
    cases = load_cases()
    hexagram_master = load_hexagram_master()
    print(f"   - 総事例数: {len(cases)}")
    print(f"   - 64卦マスター: {len(hexagram_master)}件")

    # 64卦別成功率
    print("\n2. 64卦別成功率を分析中...")
    hexagram_stats = analyze_hexagram_success_rate(cases, hexagram_master)
    with open(ANALYSIS_DIR / "hexagram_success_rate.json", "w", encoding="utf-8") as f:
        json.dump(hexagram_stats, f, ensure_ascii=False, indent=2)
    print(f"   - 出力: {ANALYSIS_DIR / 'hexagram_success_rate.json'}")

    # 卦×アクションマトリクス
    print("\n3. 卦×アクションマトリクスを分析中...")
    action_matrix = analyze_hexagram_action_matrix(cases, hexagram_master)
    with open(ANALYSIS_DIR / "hexagram_action_matrix.json", "w", encoding="utf-8") as f:
        json.dump(action_matrix, f, ensure_ascii=False, indent=2)
    print(f"   - 出力: {ANALYSIS_DIR / 'hexagram_action_matrix.json'}")
    print(f"   - 組み合わせ数: {len(action_matrix)}")

    # 爻×結果相関
    print("\n4. 爻×結果相関を分析中...")
    yao_correlation = analyze_yao_outcome_correlation(cases)
    with open(ANALYSIS_DIR / "yao_outcome_correlation.json", "w", encoding="utf-8") as f:
        json.dump(yao_correlation, f, ensure_ascii=False, indent=2)
    print(f"   - 出力: {ANALYSIS_DIR / 'yao_outcome_correlation.json'}")

    # pattern_type別成功率
    print("\n5. pattern_type別成功率を分析中...")
    pattern_stats = analyze_pattern_success_rate(cases)

    # 予測エンジン用統計データ
    print("\n6. 予測エンジン用統計データを生成中...")
    base_stats = generate_prediction_base_stats(
        hexagram_stats, action_matrix, yao_correlation, pattern_stats, cases
    )
    with open(ANALYSIS_DIR / "prediction_base_stats.json", "w", encoding="utf-8") as f:
        json.dump(base_stats, f, ensure_ascii=False, indent=2)
    print(f"   - 出力: {ANALYSIS_DIR / 'prediction_base_stats.json'}")

    # サマリー表示
    print("\n=== 分析サマリー ===")
    print(f"総事例数: {base_stats['overall']['total']}")
    print(f"全体成功率: {base_stats['overall']['success_rate']:.1%}")

    print("\n【成功率TOP5卦】")
    for item in base_stats["top_hexagrams"][:5]:
        print(f"  {item['rank']}. {item['name']} ({item['keyword']}) - {item['success_rate']:.1%} ({item['total']}件)")

    print("\n【爻位置別成功率】")
    for pos in range(1, 7):
        data = yao_correlation["by_position"][str(pos)]
        print(f"  第{pos}爻 ({data['description']}): {data['success_rate']:.1%} ({data['total']}件)")

    print("\n【パターン別成功率】")
    for pattern, data in sorted(pattern_stats.items(), key=lambda x: x[1]["success_rate"], reverse=True):
        print(f"  {pattern}: {data['success_rate']:.1%} ({data['total']}件)")

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
