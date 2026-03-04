#!/usr/bin/env python3
"""
特徴量エンリッチメント: reversibility_score + duration_years

reversibility_score (1-5): after_state × outcome のルールテーブルで100%カバー
duration_years (int|None): periodフィールドから算出（95%+カバー）

使用方法:
  python3 scripts/enrich_features.py --dry-run   # 統計確認のみ
  python3 scripts/enrich_features.py --apply      # cases.jsonl を更新
"""

import json
import re
import sys
import shutil
from datetime import datetime
from collections import Counter
from pathlib import Path

# ============================================================
# reversibility_score ルールテーブル
# score: 1=irreversible, 2=mostly_irreversible,
#        3=partially_reversible, 4=mostly_reversible,
#        5=highly_reversible
# ============================================================

REVERSIBILITY_RULES = {
    # --- 崩壊系 ---
    ("崩壊・消滅", "Failure"): 1,
    ("崩壊・消滅", "Mixed"): 1,
    ("消滅・破綻", "Failure"): 1,
    ("消滅・破綻", "Mixed"): 1,
    # --- どん底系 ---
    ("どん底・危機", "Failure"): 2,
    ("どん底・危機", "Mixed"): 2,
    ("どん底・危機", "Success"): 3,
    # --- 混乱系 ---
    ("混乱・カオス", "Failure"): 2,
    ("混乱・カオス", "Mixed"): 2,
    ("混乱・カオス", "Success"): 3,
    ("混乱・衰退", "Failure"): 2,
    ("混乱・衰退", "Mixed"): 2,
    ("迷走・混乱", "Failure"): 2,
    ("迷走・混乱", "Mixed"): 2,
    # --- 縮小系 ---
    ("縮小安定・生存", "Failure"): 2,
    ("縮小安定・生存", "Mixed"): 3,
    ("縮小安定・生存", "Success"): 3,
    # --- 停滞系 ---
    ("停滞・閉塞", "Failure"): 3,
    ("停滞・閉塞", "Mixed"): 3,
    ("停滞・閉塞", "Success"): 4,
    # --- 現状維持系 ---
    ("現状維持・延命", "Failure"): 3,
    ("現状維持・延命", "Mixed"): 3,
    ("現状維持・延命", "Success"): 4,
    # --- 分岐系 ---
    ("分岐・様子見", "Failure"): 3,
    ("分岐・様子見", "Mixed"): 3,
    ("分岐・様子見", "Success"): 4,
    # --- 安定系 (ワイルドカード) ---
    ("安定・平和", "*"): 4,
    ("安定・停止", "*"): 4,
    ("安定成長・成功", "*"): 4,
    # --- 成長系 ---
    ("成長・拡大", "*"): 4,
    ("成長痛", "*"): 3,
    # --- 回復・成功系 ---
    ("V字回復・大成功", "*"): 5,
    ("拡大・繁栄", "*"): 4,
    ("持続成長・大成功", "*"): 5,
    # --- 変質系 ---
    ("変質・新生", "*"): 4,
    # --- 交流系 ---
    ("喜び・交流", "*"): 5,
}

# outcome単体のデフォルト（ルールテーブルにマッチしない場合のフォールバック）
OUTCOME_DEFAULTS = {
    "Success": 4,
    "Mixed": 3,
    "Failure": 2,
    "PartialSuccess": 3,
}

# 中間値フォールバック
FALLBACK_SCORE = 3


def calc_reversibility_score(after_state: str, outcome: str) -> int:
    """
    reversibility_score を算出する。
    ルックアップ順序:
      1. 完全一致 (after_state, outcome)
      2. ワイルドカード (after_state, "*")
      3. outcome 単体デフォルト
      4. フォールバック = 3
    """
    # 1. 完全一致
    key = (after_state, outcome)
    if key in REVERSIBILITY_RULES:
        return REVERSIBILITY_RULES[key]

    # 2. ワイルドカード
    wildcard_key = (after_state, "*")
    if wildcard_key in REVERSIBILITY_RULES:
        return REVERSIBILITY_RULES[wildcard_key]

    # 3. outcome 単体デフォルト
    if outcome in OUTCOME_DEFAULTS:
        return OUTCOME_DEFAULTS[outcome]

    # 4. フォールバック
    return FALLBACK_SCORE


# ============================================================
# duration_years 算出
# ============================================================

def parse_duration_years(period: str) -> int | None:
    """
    period文字列からduration_yearsを算出。
    パース不能 → None
    """
    if not period or not period.strip():
        return None

    p = period.strip()

    # "Present" / "現在" を 2026 に置換
    p_normalized = p.replace("Present", "2026").replace("present", "2026").replace("現在", "2026")

    # パターン1: YYYY-YYYY (最も一般的: 89.6%)
    m = re.match(r'^(\d{3,4})-(\d{3,4})$', p_normalized)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        return max(0, end - start)

    # パターン2: YYYY (単年: 5.6%)
    m = re.match(r'^(\d{4})$', p)
    if m:
        return 0

    # パターン3: YYYYs (年代: 0.1%)
    m = re.match(r'^(\d{4})s$', p)
    if m:
        return 10

    # パターン4: YYYYs-YYYYs (0.5%)
    m = re.match(r'^(\d{4})s-(\d{4})s$', p_normalized)
    if m:
        start_decade = int(m.group(1))
        end_decade = int(m.group(2))
        return max(0, end_decade - start_decade + 10)

    # パターン5: YYYY-YYYYs (0.1%)
    m = re.match(r'^(\d{4})-(\d{4})s$', p_normalized)
    if m:
        start = int(m.group(1))
        end_decade = int(m.group(2))
        return max(0, end_decade + 5 - start)  # 年代の中間値

    # パターン6: YYYYs-YYYY (0.1%)
    m = re.match(r'^(\d{4})s-(\d{4})$', p_normalized)
    if m:
        start_decade = int(m.group(1))
        end = int(m.group(2))
        return max(0, end - start_decade - 5)  # 年代の中間値

    # パターン7: BC付き — BC3000-BC500, BC500-1800, BC500-AD500 等
    m = re.match(r'^BC(\d+)-BC(\d+)$', p_normalized)
    if m:
        start_bc = int(m.group(1))
        end_bc = int(m.group(2))
        return max(0, start_bc - end_bc)

    m = re.match(r'^BC(\d+)-(?:AD)?(\d+)$', p_normalized)
    if m:
        start_bc = int(m.group(1))
        end_ad = int(m.group(2))
        return start_bc + end_ad

    # パターン8: "YYYY-" (継続中)
    m = re.match(r'^(\d{4})-$', p)
    if m:
        start = int(m.group(1))
        return max(0, 2026 - start)

    # パターン9: "〜YYYY" (開始不明)
    m = re.match(r'^[〜~](\d{4})$', p)
    if m:
        return None  # 開始年不明

    # パターン10: 日本語の年代表現（パース不能）
    if any(kw in p for kw in ['代', '期', '年', '世紀', '通時']):
        return None

    # パターン11: 3桁年 (578-2024 等) — p_normalizedで処理済み
    m = re.match(r'^(\d{3})-(\d{3,4})$', p_normalized)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        return max(0, end - start)

    return None


# ============================================================
# メイン処理
# ============================================================

def load_cases(filepath: Path) -> list[dict]:
    """cases.jsonl を読み込み"""
    cases = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def save_cases(cases: list[dict], filepath: Path) -> None:
    """cases.jsonl を書き出し"""
    with open(filepath, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")


def enrich(cases: list[dict]) -> dict:
    """
    全事例に reversibility_score と duration_years を追加。
    統計情報を返す。
    """
    rev_dist = Counter()
    dur_values = []
    dur_none_count = 0
    rev_method = Counter()  # どのルックアップ段階でマッチしたか

    for case in cases:
        after_state = case.get("after_state", "")
        outcome = case.get("outcome", "")
        period = case.get("period", "")

        # --- reversibility_score ---
        score = calc_reversibility_score(after_state, outcome)
        case["reversibility_score"] = score
        rev_dist[score] += 1

        # マッチ方法追跡
        key = (after_state, outcome)
        if key in REVERSIBILITY_RULES:
            rev_method["exact_match"] += 1
        elif (after_state, "*") in REVERSIBILITY_RULES:
            rev_method["wildcard_match"] += 1
        elif outcome in OUTCOME_DEFAULTS:
            rev_method["outcome_default"] += 1
        else:
            rev_method["fallback"] += 1

        # --- duration_years ---
        dur = parse_duration_years(period)
        case["duration_years"] = dur
        if dur is not None:
            dur_values.append(dur)
        else:
            dur_none_count += 1

    total = len(cases)
    dur_filled = len(dur_values)

    stats = {
        "total_cases": total,
        "reversibility": {
            "distribution": {f"score_{k}": v for k, v in sorted(rev_dist.items())},
            "coverage": f"{total}/{total} (100.0%)",
            "method": dict(rev_method.most_common()),
        },
        "duration_years": {
            "filled": dur_filled,
            "null": dur_none_count,
            "coverage": f"{dur_filled}/{total} ({dur_filled/total*100:.1f}%)",
            "mean": round(sum(dur_values) / len(dur_values), 1) if dur_values else None,
            "median": sorted(dur_values)[len(dur_values) // 2] if dur_values else None,
            "min": min(dur_values) if dur_values else None,
            "max": max(dur_values) if dur_values else None,
        },
    }

    return stats


def print_stats(stats: dict) -> None:
    """統計情報を表示"""
    print("=" * 60)
    print(f"  特徴量エンリッチメント統計  ({stats['total_cases']} 件)")
    print("=" * 60)

    print("\n[reversibility_score]")
    print(f"  カバレッジ: {stats['reversibility']['coverage']}")
    print("  分布:")
    for k, v in sorted(stats["reversibility"]["distribution"].items()):
        pct = v / stats["total_cases"] * 100
        bar = "█" * int(pct / 2)
        print(f"    {k}: {v:>5} ({pct:5.1f}%) {bar}")
    print("  マッチ方法:")
    for method, cnt in stats["reversibility"]["method"].items():
        print(f"    {method}: {cnt}")

    print("\n[duration_years]")
    dur = stats["duration_years"]
    print(f"  カバレッジ: {dur['coverage']}")
    print(f"  Null件数:  {dur['null']}")
    if dur["mean"] is not None:
        print(f"  平均:      {dur['mean']} 年")
        print(f"  中央値:    {dur['median']} 年")
        print(f"  最小:      {dur['min']} 年")
        print(f"  最大:      {dur['max']} 年")

    print("\n" + "=" * 60)


def main():
    project_root = Path(__file__).resolve().parent.parent
    cases_path = project_root / "data" / "raw" / "cases.jsonl"

    if not cases_path.exists():
        print(f"ERROR: {cases_path} が見つかりません", file=sys.stderr)
        sys.exit(1)

    # 引数解析
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    apply = "--apply" in args

    if not dry_run and not apply:
        print("使用方法:")
        print("  python3 scripts/enrich_features.py --dry-run   # 統計確認のみ")
        print("  python3 scripts/enrich_features.py --apply      # cases.jsonl を更新")
        sys.exit(1)

    # 読み込み
    print(f"Loading {cases_path} ...")
    cases = load_cases(cases_path)
    print(f"  {len(cases)} 件読み込み完了")

    # エンリッチ
    stats = enrich(cases)
    print_stats(stats)

    if dry_run:
        print("[DRY-RUN] cases.jsonl は更新されません")
        return

    if apply:
        # バックアップ
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = cases_path.parent / f"cases_backup_{timestamp}.jsonl"
        print(f"バックアップ作成: {backup_path}")
        shutil.copy2(cases_path, backup_path)

        # 書き出し
        save_cases(cases, cases_path)
        print(f"cases.jsonl 更新完了 ({len(cases)} 件)")

        # 検証: 再読み込みして確認
        verify = load_cases(cases_path)
        rev_count = sum(1 for c in verify if "reversibility_score" in c)
        dur_count = sum(1 for c in verify if "duration_years" in c)
        print(f"\n[検証] reversibility_score 付与率: {rev_count}/{len(verify)} ({rev_count/len(verify)*100:.1f}%)")
        print(f"[検証] duration_years 付与率: {dur_count}/{len(verify)} ({dur_count/len(verify)*100:.1f}%)")


if __name__ == "__main__":
    main()
