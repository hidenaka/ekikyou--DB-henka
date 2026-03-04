#!/usr/bin/env python3
"""
build_polysemy_metadata.py — 多義性メタデータインデックス構築

Phase 3 分析で判明したaction_typeラベルのスケール間多義性(JSD)を
BacktraceEngine が警告として使用できるメタデータJSONに整形する。

入力:
  - analysis/phase3/label_polysemy_stats.json  (JSD分析結果)
  - data/raw/cases.jsonl                       (成功率集計用)

出力:
  - data/reverse/polysemy_metadata.json

Usage:
    python3 scripts/build_polysemy_metadata.py
"""

import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# パス設定
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

_POLYSEMY_STATS_PATH = os.path.join(
    _PROJECT_ROOT, "analysis", "phase3", "label_polysemy_stats.json"
)
_CASES_PATH = os.path.join(_PROJECT_ROOT, "data", "raw", "cases.jsonl")
_OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "data", "reverse", "polysemy_metadata.json")

# ---------------------------------------------------------------------------
# 閾値
# ---------------------------------------------------------------------------
_JSD_HIGH = 0.1
_JSD_MEDIUM = 0.05


def _load_polysemy_stats() -> Dict[str, Any]:
    """label_polysemy_stats.json を読み込む。"""
    with open(_POLYSEMY_STATS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _compute_scale_success_rates(cases_path: str) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    cases.jsonl から action_type x scale x outcome の集計を行う。

    Returns:
        {
            action_type: {
                scale: {"success": N, "total": N}
            }
        }
    """
    counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"success": 0, "total": 0})
    )

    with open(cases_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError:
                continue

            action_type = case.get("action_type", "")
            scale = case.get("scale", "other")
            outcome = case.get("outcome", "")

            if not action_type:
                continue

            counts[action_type][scale]["total"] += 1
            if outcome == "Success":
                counts[action_type][scale]["success"] += 1

    return dict(counts)


def _classify_polysemy_level(jsd_combined: float) -> str:
    """JSD値から多義性レベルを分類する。"""
    if jsd_combined > _JSD_HIGH:
        return "high"
    elif jsd_combined > _JSD_MEDIUM:
        return "medium"
    else:
        return "low"


def _build_warning_message(
    label: str,
    scale_success_rates: Dict[str, float],
    polysemy_level: str,
) -> Optional[str]:
    """多義性警告メッセージを生成する。"""
    if not scale_success_rates or polysemy_level == "low":
        return None

    rates = list(scale_success_rates.values())
    if len(rates) < 2:
        return None

    max_rate = max(rates)
    min_rate = min(rates)
    divergence_pct = round((max_rate - min_rate) * 100, 1)

    if divergence_pct < 10:
        return None

    max_scale = max(scale_success_rates, key=scale_success_rates.get)
    min_scale = min(scale_success_rates, key=scale_success_rates.get)

    return (
        f"このラベルはスケールにより成功率が大きく異なります"
        f"（最大{divergence_pct}%の乖離: "
        f"{max_scale}={round(max_rate*100, 1)}% vs "
        f"{min_scale}={round(min_rate*100, 1)}%）"
    )


def build_polysemy_metadata() -> Dict[str, Any]:
    """多義性メタデータインデックスを構築する。"""
    stats = _load_polysemy_stats()
    jsd_scores = stats.get("jsd_scores", {})
    crosstab = stats.get("action_type_x_scale_crosstab", {})

    # cases.jsonl から成功率を集計
    success_data = _compute_scale_success_rates(_CASES_PATH)

    labels_meta: Dict[str, Any] = {}
    high_count = 0
    medium_count = 0
    low_count = 0

    for label, jsd_info in jsd_scores.items():
        jsd_combined = jsd_info.get("jsd_combined", 0.0)
        jsd_before = jsd_info.get("jsd_before_avg", 0.0)
        jsd_after = jsd_info.get("jsd_after_avg", 0.0)
        total_cases = jsd_info.get("total_cases", 0)

        polysemy_level = _classify_polysemy_level(jsd_combined)
        if polysemy_level == "high":
            high_count += 1
        elif polysemy_level == "medium":
            medium_count += 1
        else:
            low_count += 1

        # scale分布
        scale_distribution = crosstab.get(label, {})

        # scale別成功率
        scale_success_rates: Dict[str, float] = {}
        label_success_data = success_data.get(label, {})
        for scale_name, scale_counts in label_success_data.items():
            total = scale_counts["total"]
            if total > 0:
                scale_success_rates[scale_name] = round(
                    scale_counts["success"] / total, 4
                )

        # 警告メッセージ
        warning_message = _build_warning_message(
            label, scale_success_rates, polysemy_level
        )

        labels_meta[label] = {
            "jsd_combined": round(jsd_combined, 4),
            "jsd_before": round(jsd_before, 4),
            "jsd_after": round(jsd_after, 4),
            "polysemy_level": polysemy_level,
            "total_cases": total_cases,
            "scale_distribution": scale_distribution,
            "scale_success_rates": scale_success_rates,
            "warning_message": warning_message,
        }

    metadata = {
        "metadata": {
            "analysis_date": stats.get("metadata", {}).get("analysis_date", ""),
            "total_labels": len(jsd_scores),
            "high_polysemy_count": high_count,
            "medium_polysemy_count": medium_count,
            "low_polysemy_count": low_count,
            "thresholds": {
                "high": f"JSD > {_JSD_HIGH}",
                "medium": f"{_JSD_MEDIUM} < JSD <= {_JSD_HIGH}",
                "low": f"JSD <= {_JSD_MEDIUM}",
            },
        },
        "labels": labels_meta,
    }

    return metadata


def main() -> None:
    """メイン: メタデータを構築してJSONに出力する。"""
    print("=== 多義性メタデータインデックス構築 ===")

    # 入力ファイル存在確認
    if not os.path.isfile(_POLYSEMY_STATS_PATH):
        print(f"ERROR: {_POLYSEMY_STATS_PATH} が見つかりません")
        sys.exit(1)
    if not os.path.isfile(_CASES_PATH):
        print(f"ERROR: {_CASES_PATH} が見つかりません")
        sys.exit(1)

    metadata = build_polysemy_metadata()

    # 出力ディレクトリ確認
    os.makedirs(os.path.dirname(_OUTPUT_PATH), exist_ok=True)

    with open(_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # サマリー出力
    meta = metadata["metadata"]
    print(f"  総ラベル数: {meta['total_labels']}")
    print(f"  高多義性 (JSD > {_JSD_HIGH}): {meta['high_polysemy_count']}")
    print(f"  中多義性: {meta['medium_polysemy_count']}")
    print(f"  低多義性: {meta['low_polysemy_count']}")
    print()

    # 高多義性ラベルの警告サンプル
    print("--- 高多義性ラベル(警告あり) ---")
    for label, info in metadata["labels"].items():
        if info["polysemy_level"] == "high" and info.get("warning_message"):
            print(f"  [{label}] JSD={info['jsd_combined']:.4f}")
            print(f"    {info['warning_message']}")
    print()

    print(f"出力: {_OUTPUT_PATH}")
    print("完了")


if __name__ == "__main__":
    main()
