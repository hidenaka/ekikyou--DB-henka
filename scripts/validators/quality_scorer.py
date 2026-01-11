#!/usr/bin/env python3
"""
品質スコアリング・検証スクリプト

抽出されたCase形式のデータを検証し、品質スコアを付与する。
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from schema_v3 import Case, Scale, Hex, PatternType, Outcome


class QualityLevel(Enum):
    """品質レベル"""
    HIGH = "high"        # 80-100点: 自動承認
    MEDIUM = "medium"    # 50-79点: サンプリングレビュー
    LOW = "low"          # 30-49点: 要レビュー
    REJECT = "reject"    # 0-29点: 破棄


@dataclass
class QualityReport:
    """品質レポート"""
    total_score: float
    level: QualityLevel
    details: Dict[str, float]
    issues: List[str]
    suggestions: List[str]


def calculate_schema_score(data: Dict) -> Tuple[float, List[str]]:
    """
    スキーマ準拠スコア（30点満点）

    - 必須フィールドの存在: 15点
    - enum値の妥当性: 15点
    """
    score = 0.0
    issues = []

    # 必須フィールド（15点）
    required_fields = [
        "target_name", "scale", "period", "story_summary",
        "before_state", "trigger_type", "action_type", "after_state",
        "before_hex", "trigger_hex", "action_hex", "after_hex",
        "pattern_type", "outcome"
    ]

    missing = [f for f in required_fields if not data.get(f)]
    if not missing:
        score += 15
    else:
        score += 15 * (len(required_fields) - len(missing)) / len(required_fields)
        issues.append(f"必須フィールド不足: {', '.join(missing)}")

    # enum値の妥当性（15点）
    enum_checks = [
        ("scale", ["company", "individual", "family", "country", "other"]),
        ("before_hex", ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]),
        ("trigger_hex", ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]),
        ("action_hex", ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]),
        ("after_hex", ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]),
        ("outcome", ["Success", "Failure", "Mixed", "PartialSuccess"]),
    ]

    valid_enums = 0
    for field, valid_values in enum_checks:
        if data.get(field) in valid_values:
            valid_enums += 1
        else:
            issues.append(f"無効な値: {field}={data.get(field)}")

    score += 15 * valid_enums / len(enum_checks)

    return score, issues


def calculate_content_score(data: Dict) -> Tuple[float, List[str]]:
    """
    コンテンツ品質スコア（40点満点）

    - story_summaryの長さと質: 15点
    - periodの具体性: 10点
    - logic_memoの存在と質: 10点
    - free_tagsの存在: 5点
    """
    score = 0.0
    issues = []

    # story_summary（15点）
    summary = data.get("story_summary", "")
    if len(summary) >= 150:
        score += 15
    elif len(summary) >= 100:
        score += 12
    elif len(summary) >= 50:
        score += 8
    elif len(summary) >= 20:
        score += 4
    else:
        issues.append("story_summaryが短すぎる")

    # 具体的な内容があるか
    if re.search(r'\d{4}年', summary):
        score += 0  # すでにカウント済みなので加点なし
    else:
        issues.append("story_summaryに具体的な年号がない")

    # period（10点）
    period = data.get("period", "")
    if re.match(r'\d{4}[-〜]\d{4}', period):  # 範囲指定
        score += 10
    elif re.match(r'\d{4}年', period):  # 単一年
        score += 7
    elif re.match(r'\d{4}', period):  # 年のみ
        score += 5
    elif period:
        score += 2
    else:
        issues.append("periodが不明確")

    # logic_memo（10点）
    logic = data.get("logic_memo", "")
    if len(logic) >= 100:
        score += 10
    elif len(logic) >= 50:
        score += 7
    elif len(logic) >= 20:
        score += 4
    elif logic:
        score += 2

    # 八卦の言及があるか
    hex_chars = ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]
    if any(h in logic for h in hex_chars):
        pass  # 良い
    elif logic:
        issues.append("logic_memoに八卦の言及がない")

    # free_tags（5点）
    tags = data.get("free_tags", [])
    if isinstance(tags, list) and len(tags) >= 3:
        score += 5
    elif isinstance(tags, list) and len(tags) >= 1:
        score += 3

    return score, issues


def calculate_consistency_score(data: Dict) -> Tuple[float, List[str]]:
    """
    整合性スコア（30点満点）

    - 八卦の変化の整合性: 15点
    - パターンと結果の整合性: 15点
    """
    score = 0.0
    issues = []

    # 八卦の変化（15点）
    before_hex = data.get("before_hex", "")
    after_hex = data.get("after_hex", "")

    # 変化があるか
    if before_hex and after_hex:
        if before_hex != after_hex:
            score += 10  # 変化がある
        else:
            score += 5   # 変化がない（ありうるが稀）
            issues.append("before_hexとafter_hexが同じ")
    else:
        issues.append("八卦が不完全")

    # 変化の妥当性チェック（既知のパターンと照合）
    known_success_patterns = [
        ("坎", "乾"), ("震", "乾"), ("艮", "乾"),
        ("巽", "乾"), ("離", "乾"), ("艮", "巽"),
    ]
    known_failure_patterns = [
        ("乾", "震"), ("乾", "坎"), ("艮", "坎"),
    ]

    transition = (before_hex, after_hex)
    outcome = data.get("outcome", "")

    if outcome == "Success" and transition in known_success_patterns:
        score += 5  # 既知の成功パターンと一致
    elif outcome == "Failure" and transition in known_failure_patterns:
        score += 5  # 既知の失敗パターンと一致
    elif outcome in ["Success", "Failure"]:
        score += 3  # パターンは不明だが結果はある

    # パターンと結果の整合性（15点）
    pattern = data.get("pattern_type", "")

    pattern_outcome_map = {
        "Shock_Recovery": ["Success", "PartialSuccess"],
        "Hubris_Collapse": ["Failure", "Mixed"],
        "Pivot_Success": ["Success", "PartialSuccess"],
        "Endurance": ["Success", "PartialSuccess", "Mixed"],
        "Slow_Decline": ["Failure", "Mixed"],
        "Steady_Growth": ["Success"],
    }

    if pattern in pattern_outcome_map:
        if outcome in pattern_outcome_map[pattern]:
            score += 15
        else:
            score += 5
            issues.append(f"パターン({pattern})と結果({outcome})の不整合")
    else:
        score += 5

    return score, issues


def calculate_quality_score(data: Dict) -> QualityReport:
    """
    総合品質スコアを計算

    Args:
        data: Case形式のデータ

    Returns:
        QualityReport
    """
    issues = []
    suggestions = []
    details = {}

    # 各カテゴリのスコア計算
    schema_score, schema_issues = calculate_schema_score(data)
    details["schema"] = schema_score
    issues.extend(schema_issues)

    content_score, content_issues = calculate_content_score(data)
    details["content"] = content_score
    issues.extend(content_issues)

    consistency_score, consistency_issues = calculate_consistency_score(data)
    details["consistency"] = consistency_score
    issues.extend(consistency_issues)

    # 総合スコア
    total_score = schema_score + content_score + consistency_score

    # レベル判定
    if total_score >= 80:
        level = QualityLevel.HIGH
    elif total_score >= 50:
        level = QualityLevel.MEDIUM
    elif total_score >= 30:
        level = QualityLevel.LOW
    else:
        level = QualityLevel.REJECT

    # 改善提案
    if schema_score < 25:
        suggestions.append("必須フィールドを埋めてください")
    if content_score < 30:
        suggestions.append("story_summaryをより詳細に記述してください")
    if consistency_score < 20:
        suggestions.append("八卦の変化とパターン/結果の整合性を確認してください")

    return QualityReport(
        total_score=total_score,
        level=level,
        details=details,
        issues=issues,
        suggestions=suggestions
    )


def validate_case(data: Dict) -> Tuple[bool, QualityReport]:
    """
    Caseデータを検証

    Returns:
        (is_valid, report)
    """
    report = calculate_quality_score(data)

    # Pydanticモデルでの検証も試行
    try:
        Case(**data)
        is_valid = True
    except Exception as e:
        is_valid = False
        report.issues.append(f"スキーマ検証エラー: {str(e)[:100]}")

    return is_valid, report


def validate_jsonl_file(filepath: Path) -> Dict:
    """
    JSONLファイル全体を検証

    Returns:
        検証結果のサマリー
    """
    results = {
        "total": 0,
        "valid": 0,
        "high_quality": 0,
        "medium_quality": 0,
        "low_quality": 0,
        "rejected": 0,
        "errors": [],
        "cases": []
    }

    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue

            results["total"] += 1

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                results["errors"].append(f"行{i}: JSON解析エラー - {e}")
                continue

            is_valid, report = validate_case(data)

            case_result = {
                "line": i,
                "target_name": data.get("target_name", "不明"),
                "is_valid": is_valid,
                "score": report.total_score,
                "level": report.level.value,
                "issues": report.issues[:3]  # 最初の3件のみ
            }
            results["cases"].append(case_result)

            if is_valid:
                results["valid"] += 1

            if report.level == QualityLevel.HIGH:
                results["high_quality"] += 1
            elif report.level == QualityLevel.MEDIUM:
                results["medium_quality"] += 1
            elif report.level == QualityLevel.LOW:
                results["low_quality"] += 1
            else:
                results["rejected"] += 1

    return results


def main():
    """メイン実行"""
    import argparse

    parser = argparse.ArgumentParser(description="品質スコアリング・検証")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="入力ファイル（JSONLまたはJSON）"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="検証結果の出力先（JSON）"
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="詳細な検証結果を表示"
    )

    args = parser.parse_args()

    if args.input.suffix == ".jsonl":
        # JSONL検証
        results = validate_jsonl_file(args.input)

        print(f"\n=== 検証結果 ===")
        print(f"総件数: {results['total']}")
        print(f"有効: {results['valid']} ({results['valid']/max(results['total'],1)*100:.1f}%)")
        print(f"  高品質(80+): {results['high_quality']}")
        print(f"  中品質(50-79): {results['medium_quality']}")
        print(f"  低品質(30-49): {results['low_quality']}")
        print(f"  破棄(0-29): {results['rejected']}")

        if results["errors"]:
            print(f"\nエラー: {len(results['errors'])}件")
            for err in results["errors"][:5]:
                print(f"  - {err}")

        if args.detail:
            print(f"\n=== 詳細 ===")
            for case in results["cases"]:
                status = "✓" if case["is_valid"] else "✗"
                print(f"{status} [{case['score']:.0f}点] {case['target_name']}")
                if case["issues"]:
                    for issue in case["issues"]:
                        print(f"    - {issue}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n結果を {args.output} に保存しました")

    else:
        # 単一JSON検証
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            for i, item in enumerate(data):
                is_valid, report = validate_case(item)
                print(f"\n=== ケース {i+1}: {item.get('target_name', '不明')} ===")
                print(f"スコア: {report.total_score:.1f} ({report.level.value})")
                print(f"詳細: {report.details}")
                if report.issues:
                    print("問題点:")
                    for issue in report.issues:
                        print(f"  - {issue}")
        else:
            is_valid, report = validate_case(data)
            print(f"スコア: {report.total_score:.1f} ({report.level.value})")
            print(f"詳細: {report.details}")
            if report.issues:
                print("問題点:")
                for issue in report.issues:
                    print(f"  - {issue}")


if __name__ == "__main__":
    main()
