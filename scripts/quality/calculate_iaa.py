#!/usr/bin/env python3
"""
IAA（Inter-Annotator Agreement）計算スクリプト

2名の検証者の結果を比較してCohen's Kappaを算出し、
一致率と不一致事例を出力する。

使用方法:
    python3 scripts/quality/calculate_iaa.py \
        data/pilot/reviewer_1_results.jsonl \
        data/pilot/reviewer_2_results.jsonl \
        --output data/pilot/iaa_report.json

入力形式 (JSONL):
    {"pilot_id": "P001", "outcome_status": "verified_correct", "verification_confidence": "high", "coi_status": "none"}

出力形式 (JSON):
    {
        "overall_agreement": 0.85,
        "kappa_by_field": {
            "outcome_status": 0.78,
            "verification_confidence": 0.65,
            "coi_status": 0.82
        },
        "disagreements": [...],
        "recommendation": "..."
    }
"""

import json
import argparse
from pathlib import Path
from collections import Counter
from typing import Optional
from datetime import datetime


# IAA対象フィールド
IAA_FIELDS = ["outcome_status", "verification_confidence", "coi_status"]

# 各フィールドの有効な値
VALID_VALUES = {
    "outcome_status": ["verified_correct", "verified_incorrect", "unverified"],
    "verification_confidence": ["high", "medium", "low", "none"],
    "coi_status": ["none", "self", "affiliated", "unknown"]
}

# Kappa閾値
KAPPA_EXCELLENT = 0.8
KAPPA_ACCEPTABLE = 0.7


def load_results(filepath: Path) -> dict[str, dict]:
    """
    検証結果ファイルを読み込む

    Returns:
        {pilot_id: {field: value, ...}, ...}
    """
    results = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                pilot_id = record.get('pilot_id')
                if not pilot_id:
                    print(f"Warning: Line {line_num} has no pilot_id, skipping")
                    continue
                results[pilot_id] = {
                    field: record.get(field) for field in IAA_FIELDS
                }
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} is not valid JSON: {e}")
    return results


def validate_results(results: dict[str, dict], reviewer_name: str) -> list[str]:
    """
    検証結果の妥当性をチェック

    Returns:
        エラーメッセージのリスト
    """
    errors = []
    for pilot_id, fields in results.items():
        for field, value in fields.items():
            if value is None:
                errors.append(f"{reviewer_name}/{pilot_id}: {field} is missing")
            elif value not in VALID_VALUES.get(field, []):
                errors.append(f"{reviewer_name}/{pilot_id}: {field}='{value}' is invalid (valid: {VALID_VALUES[field]})")
    return errors


def calculate_agreement_rate(r1: dict[str, dict], r2: dict[str, dict], field: str) -> tuple[float, int, int]:
    """
    単純一致率を計算

    Returns:
        (一致率, 一致数, 比較可能な総数)
    """
    common_ids = set(r1.keys()) & set(r2.keys())
    if not common_ids:
        return 0.0, 0, 0

    agree = 0
    total = 0
    for pilot_id in common_ids:
        v1 = r1[pilot_id].get(field)
        v2 = r2[pilot_id].get(field)
        if v1 is not None and v2 is not None:
            total += 1
            if v1 == v2:
                agree += 1

    rate = agree / total if total > 0 else 0.0
    return rate, agree, total


def calculate_cohens_kappa(r1: dict[str, dict], r2: dict[str, dict], field: str) -> Optional[float]:
    """
    Cohen's Kappaを計算

    κ = (Po - Pe) / (1 - Pe)

    Po: 観測一致率
    Pe: 偶然一致率（各ラベルの周辺確率の積の和）

    Returns:
        Kappa値（計算不能な場合はNone）
    """
    common_ids = set(r1.keys()) & set(r2.keys())
    if not common_ids:
        return None

    # 各検証者のラベル分布を集計
    labels_r1 = []
    labels_r2 = []

    for pilot_id in common_ids:
        v1 = r1[pilot_id].get(field)
        v2 = r2[pilot_id].get(field)
        if v1 is not None and v2 is not None:
            labels_r1.append(v1)
            labels_r2.append(v2)

    n = len(labels_r1)
    if n == 0:
        return None

    # 観測一致率 Po
    agree = sum(1 for a, b in zip(labels_r1, labels_r2) if a == b)
    po = agree / n

    # 偶然一致率 Pe
    # 各ラベルの周辺確率を計算
    all_labels = set(labels_r1) | set(labels_r2)
    counter_r1 = Counter(labels_r1)
    counter_r2 = Counter(labels_r2)

    pe = 0.0
    for label in all_labels:
        p1 = counter_r1.get(label, 0) / n
        p2 = counter_r2.get(label, 0) / n
        pe += p1 * p2

    # Kappa計算
    if pe == 1.0:
        # 完全一致または全て同じラベルの場合
        return 1.0 if po == 1.0 else None

    kappa = (po - pe) / (1 - pe)
    return kappa


def find_disagreements(r1: dict[str, dict], r2: dict[str, dict]) -> list[dict]:
    """
    不一致事例を抽出

    Returns:
        [{"pilot_id": "P005", "field": "verification_confidence", "r1": "high", "r2": "medium"}, ...]
    """
    disagreements = []
    common_ids = sorted(set(r1.keys()) & set(r2.keys()))

    for pilot_id in common_ids:
        for field in IAA_FIELDS:
            v1 = r1[pilot_id].get(field)
            v2 = r2[pilot_id].get(field)
            if v1 is not None and v2 is not None and v1 != v2:
                disagreements.append({
                    "pilot_id": pilot_id,
                    "field": field,
                    "r1": v1,
                    "r2": v2
                })

    return disagreements


def generate_recommendation(kappa_by_field: dict[str, Optional[float]]) -> str:
    """
    Kappa値に基づく推奨事項を生成
    """
    excellent = []
    acceptable = []
    needs_improvement = []

    for field, kappa in kappa_by_field.items():
        if kappa is None:
            needs_improvement.append(f"{field}(計算不能)")
        elif kappa >= KAPPA_EXCELLENT:
            excellent.append(f"{field}")
        elif kappa >= KAPPA_ACCEPTABLE:
            acceptable.append(f"{field}")
        else:
            needs_improvement.append(f"{field}")

    parts = []
    if excellent:
        parts.append(f"優良(k>=0.8): {', '.join(excellent)}")
    if acceptable:
        parts.append(f"合格(k>=0.7): {', '.join(acceptable)}")
    if needs_improvement:
        parts.append(f"要改善(k<0.7): {', '.join(needs_improvement)}")

    return "; ".join(parts)


def calculate_iaa(r1_path: Path, r2_path: Path) -> dict:
    """
    IAA計算のメイン処理
    """
    # データ読み込み
    r1 = load_results(r1_path)
    r2 = load_results(r2_path)

    print(f"\n=== IAA計算 ===")
    print(f"Reviewer 1: {r1_path.name} ({len(r1)}件)")
    print(f"Reviewer 2: {r2_path.name} ({len(r2)}件)")

    # バリデーション
    errors = validate_results(r1, "R1") + validate_results(r2, "R2")
    if errors:
        print(f"\n[Warning] バリデーションエラー ({len(errors)}件):")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... (+{len(errors) - 10}件)")

    # 共通IDの確認
    common_ids = set(r1.keys()) & set(r2.keys())
    only_r1 = set(r1.keys()) - set(r2.keys())
    only_r2 = set(r2.keys()) - set(r1.keys())

    print(f"\n共通: {len(common_ids)}件, R1のみ: {len(only_r1)}件, R2のみ: {len(only_r2)}件")

    if not common_ids:
        print("[Error] 共通のpilot_idがありません")
        return {"error": "No common pilot_ids"}

    # フィールド別Kappa計算
    kappa_by_field = {}
    agreement_by_field = {}

    print(f"\n=== フィールド別スコア ===")
    for field in IAA_FIELDS:
        rate, agree, total = calculate_agreement_rate(r1, r2, field)
        kappa = calculate_cohens_kappa(r1, r2, field)
        kappa_by_field[field] = round(kappa, 3) if kappa is not None else None
        agreement_by_field[field] = {
            "rate": round(rate, 3),
            "agree": agree,
            "total": total
        }

        kappa_str = f"{kappa:.3f}" if kappa is not None else "N/A"
        status = ""
        if kappa is not None:
            if kappa >= KAPPA_EXCELLENT:
                status = "[優良]"
            elif kappa >= KAPPA_ACCEPTABLE:
                status = "[合格]"
            else:
                status = "[要改善]"

        print(f"  {field}: kappa={kappa_str} {status}, 一致率={rate:.1%} ({agree}/{total})")

    # 全体一致率（全フィールド一致した事例の割合）
    full_agree = 0
    for pilot_id in common_ids:
        if all(r1[pilot_id].get(f) == r2[pilot_id].get(f) for f in IAA_FIELDS):
            full_agree += 1
    overall_agreement = full_agree / len(common_ids) if common_ids else 0.0

    print(f"\n全体一致率（全フィールド一致）: {overall_agreement:.1%} ({full_agree}/{len(common_ids)})")

    # 不一致事例
    disagreements = find_disagreements(r1, r2)
    print(f"\n不一致事例: {len(disagreements)}件")

    # 推奨事項
    recommendation = generate_recommendation(kappa_by_field)
    print(f"\n推奨: {recommendation}")

    # 結果オブジェクト
    result = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "reviewer_1_file": str(r1_path),
            "reviewer_2_file": str(r2_path),
            "common_cases": len(common_ids),
            "r1_only_cases": len(only_r1),
            "r2_only_cases": len(only_r2)
        },
        "overall_agreement": round(overall_agreement, 3),
        "kappa_by_field": kappa_by_field,
        "agreement_by_field": agreement_by_field,
        "disagreements": disagreements,
        "recommendation": recommendation,
        "validation_errors": errors if errors else None
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="IAA（Inter-Annotator Agreement）計算スクリプト"
    )
    parser.add_argument(
        "reviewer1",
        type=Path,
        help="検証者1の結果ファイル (JSONL)"
    )
    parser.add_argument(
        "reviewer2",
        type=Path,
        help="検証者2の結果ファイル (JSONL)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="出力ファイルパス (JSON)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="詳細出力"
    )

    args = parser.parse_args()

    # ファイル存在チェック
    if not args.reviewer1.exists():
        print(f"Error: {args.reviewer1} が見つかりません")
        return 1
    if not args.reviewer2.exists():
        print(f"Error: {args.reviewer2} が見つかりません")
        return 1

    # IAA計算
    result = calculate_iaa(args.reviewer1, args.reviewer2)

    # 出力
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n結果を出力しました: {args.output}")
    else:
        print(f"\n=== 出力 (JSON) ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    # 不一致詳細（verboseモード）
    if args.verbose and result.get("disagreements"):
        print(f"\n=== 不一致詳細 ===")
        for d in result["disagreements"]:
            print(f"  {d['pilot_id']}/{d['field']}: R1={d['r1']} vs R2={d['r2']}")

    return 0


if __name__ == "__main__":
    exit(main())
