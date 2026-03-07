#!/usr/bin/env python3
"""
条件分岐正規化スクリプト

前回の一括正規化で一律マッピングされた値を、
バックアップのraw値を参照して条件分岐で正規化し直す。

実行順序:
  1. before_state 上書き
  2. action_type 上書き
  3. pattern_type 条件分岐（上書き後のbefore_state/action_typeを使用）

Usage:
  python3 scripts/normalize_conditional.py --dry-run   # レポートのみ
  python3 scripts/normalize_conditional.py              # 実行（自動バックアップあり）
"""

import json
import sys
import os
import shutil
from datetime import datetime
from collections import Counter
from pathlib import Path

# --- パス設定 ---
BASE_DIR = Path(__file__).resolve().parent.parent
CASES_FILE = BASE_DIR / "data" / "raw" / "cases.jsonl"
BACKUP_RAW = BASE_DIR / "data" / "raw" / "cases_backup_20260307_075625.jsonl"

# --- schema_v3 正規値 ---
VALID_BEFORE_STATE = {
    "絶頂・慢心", "停滞・閉塞", "混乱・カオス", "成長痛",
    "どん底・危機", "安定・平和"
}
VALID_TRIGGER_TYPE = {
    "外部ショック", "内部崩壊", "意図的決断", "偶発・出会い"
}
VALID_ACTION_TYPE = {
    "攻める・挑戦", "守る・維持", "捨てる・撤退", "耐える・潜伏",
    "対話・融合", "刷新・破壊", "逃げる・放置", "分散・スピンオフ"
}
VALID_AFTER_STATE = {
    "V字回復・大成功", "縮小安定・生存", "変質・新生",
    "現状維持・延命", "迷走・混乱", "崩壊・消滅"
}
VALID_PATTERN_TYPE = {
    "Shock_Recovery", "Hubris_Collapse", "Pivot_Success",
    "Endurance", "Slow_Decline"
}


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def validate_enum(records):
    """全件schema_v3準拠チェック"""
    errors = []
    for i, rec in enumerate(records):
        tid = rec.get("transition_id", rec.get("canonical_id", f"index_{i}"))
        if rec.get("before_state") not in VALID_BEFORE_STATE:
            errors.append(f"  {tid}: before_state='{rec.get('before_state')}'")
        if rec.get("action_type") not in VALID_ACTION_TYPE:
            errors.append(f"  {tid}: action_type='{rec.get('action_type')}'")
        if rec.get("after_state") not in VALID_AFTER_STATE:
            errors.append(f"  {tid}: after_state='{rec.get('after_state')}'")
        if rec.get("pattern_type") not in VALID_PATTERN_TYPE:
            errors.append(f"  {tid}: pattern_type='{rec.get('pattern_type')}'")
        if rec.get("trigger_type") and rec["trigger_type"] not in VALID_TRIGGER_TYPE:
            errors.append(f"  {tid}: trigger_type='{rec.get('trigger_type')}'")
    return errors


def main():
    dry_run = "--dry-run" in sys.argv

    print("=" * 70)
    print("条件分岐正規化スクリプト")
    print(f"モード: {'DRY-RUN（変更なし）' if dry_run else '実行モード'}")
    print("=" * 70)

    # --- ファイル読み込み ---
    if not CASES_FILE.exists():
        print(f"ERROR: {CASES_FILE} が見つかりません")
        sys.exit(1)
    if not BACKUP_RAW.exists():
        print(f"ERROR: {BACKUP_RAW} が見つかりません")
        sys.exit(1)

    current = load_jsonl(CASES_FILE)
    backup = load_jsonl(BACKUP_RAW)

    if len(current) != len(backup):
        print(f"ERROR: レコード数不一致 current={len(current)} backup={len(backup)}")
        sys.exit(1)

    print(f"\nレコード数: {len(current)}")

    # --- 変換前 pattern_type 分布 ---
    before_pt = Counter(c.get("pattern_type") for c in current)

    # --- 変換カウンター ---
    counters = {
        "before_state: 安定成長・成功→安定・平和": 0,
        "action_type: 逃げる・守る→耐える・潜伏": 0,
        "Failed_Attempt→Slow_Decline": 0,
        "Failed_Attempt→Hubris_Collapse(据置)": 0,
        "Stagnation→Endurance": 0,
        "Stagnation→Slow_Decline(据置)": 0,
        "Exploration→Pivot_Success": 0,
        "Exploration→Endurance(据置)": 0,
        "Steady_Growth→Pivot_Success": 0,
        "Steady_Growth→Slow_Decline": 0,
        "Steady_Growth→Endurance(据置)": 0,
    }

    # =============================================
    # Step 1: before_state 上書き
    # =============================================
    print("\n--- Step 1: before_state 上書き ---")
    for i, (cur, bak) in enumerate(zip(current, backup)):
        raw_bs = bak.get("before_state")
        if raw_bs == "安定成長・成功" and current[i]["before_state"] != "安定・平和":
            current[i]["before_state"] = "安定・平和"
            counters["before_state: 安定成長・成功→安定・平和"] += 1

    print(f"  安定成長・成功 → 安定・平和: {counters['before_state: 安定成長・成功→安定・平和']}件")

    # =============================================
    # Step 2: action_type 上書き
    # =============================================
    print("\n--- Step 2: action_type 上書き ---")
    for i, (cur, bak) in enumerate(zip(current, backup)):
        raw_at = bak.get("action_type")
        if raw_at == "逃げる・守る" and current[i]["action_type"] != "耐える・潜伏":
            current[i]["action_type"] = "耐える・潜伏"
            counters["action_type: 逃げる・守る→耐える・潜伏"] += 1

    print(f"  逃げる・守る → 耐える・潜伏: {counters['action_type: 逃げる・守る→耐える・潜伏']}件")

    # =============================================
    # Step 3: pattern_type 条件分岐
    # =============================================
    print("\n--- Step 3: pattern_type 条件分岐 ---")

    for i, (cur, bak) in enumerate(zip(current, backup)):
        raw_pt = bak.get("pattern_type")
        # 正規化済みの値を参照（Step 1/2の上書き後）
        norm_before = current[i].get("before_state")
        norm_after = current[i].get("after_state")
        norm_action = current[i].get("action_type")

        # --- Rule 1: Failed_Attempt ---
        if raw_pt == "Failed_Attempt":
            if norm_before == "絶頂・慢心" and norm_after == "崩壊・消滅":
                # 据え置き Hubris_Collapse
                counters["Failed_Attempt→Hubris_Collapse(据置)"] += 1
            else:
                target = "Slow_Decline"
                if current[i]["pattern_type"] != target:
                    current[i]["pattern_type"] = target
                    counters["Failed_Attempt→Slow_Decline"] += 1

        # --- Rule 2: Stagnation ---
        elif raw_pt == "Stagnation":
            if norm_after in {"迷走・混乱", "崩壊・消滅"}:
                # 据え置き Slow_Decline
                counters["Stagnation→Slow_Decline(据置)"] += 1
            else:
                target = "Endurance"
                if current[i]["pattern_type"] != target:
                    current[i]["pattern_type"] = target
                    counters["Stagnation→Endurance"] += 1

        # --- Rule 3: Exploration ---
        elif raw_pt == "Exploration":
            if norm_after in {"V字回復・大成功", "変質・新生"}:
                target = "Pivot_Success"
                if current[i]["pattern_type"] != target:
                    current[i]["pattern_type"] = target
                    counters["Exploration→Pivot_Success"] += 1
            else:
                # 据え置き Endurance
                counters["Exploration→Endurance(据置)"] += 1

        # --- Rule 4: Steady_Growth ---
        elif raw_pt == "Steady_Growth":
            if norm_action in {"守る・維持", "耐える・潜伏"} and \
               norm_after in {"現状維持・延命", "縮小安定・生存"}:
                # 据え置き Endurance
                counters["Steady_Growth→Endurance(据置)"] += 1
            elif norm_after in {"崩壊・消滅", "迷走・混乱"}:
                # 安定成長からの崩壊/混乱 = 緩慢な衰退
                target = "Slow_Decline"
                if current[i]["pattern_type"] != target:
                    current[i]["pattern_type"] = target
                    counters["Steady_Growth→Slow_Decline"] += 1
            else:
                target = "Pivot_Success"
                if current[i]["pattern_type"] != target:
                    current[i]["pattern_type"] = target
                    counters["Steady_Growth→Pivot_Success"] += 1

    # --- 変換サマリー ---
    print("\n  [Failed_Attempt (818件)]")
    print(f"    → Hubris_Collapse(据置): {counters['Failed_Attempt→Hubris_Collapse(据置)']}")
    print(f"    → Slow_Decline:          {counters['Failed_Attempt→Slow_Decline']}")

    print(f"\n  [Stagnation (691件)]")
    print(f"    → Slow_Decline(据置): {counters['Stagnation→Slow_Decline(据置)']}")
    print(f"    → Endurance:          {counters['Stagnation→Endurance']}")

    print(f"\n  [Exploration (407件)]")
    print(f"    → Pivot_Success: {counters['Exploration→Pivot_Success']}")
    print(f"    → Endurance(据置): {counters['Exploration→Endurance(据置)']}")

    print(f"\n  [Steady_Growth (1,975件)]")
    print(f"    → Pivot_Success:    {counters['Steady_Growth→Pivot_Success']}")
    print(f"    → Slow_Decline:     {counters['Steady_Growth→Slow_Decline']}")
    print(f"    → Endurance(据置): {counters['Steady_Growth→Endurance(据置)']}")

    # --- 変換後 pattern_type 分布 ---
    after_pt = Counter(c.get("pattern_type") for c in current)

    print("\n" + "=" * 70)
    print("pattern_type 分布表")
    print("=" * 70)
    print(f"{'pattern_type':<20} {'変換前':>8} {'変換後':>8} {'差分':>8}")
    print("-" * 50)
    for pt in sorted(VALID_PATTERN_TYPE):
        b = before_pt.get(pt, 0)
        a = after_pt.get(pt, 0)
        d = a - b
        sign = "+" if d > 0 else ""
        print(f"{pt:<20} {b:>8} {a:>8} {sign}{d:>7}")
    print("-" * 50)
    print(f"{'合計':<20} {sum(before_pt.values()):>8} {sum(after_pt.values()):>8}")

    # --- 差分検証式（GPT-5.4指定） ---
    print("\n" + "=" * 70)
    print("差分検証式（GPT-5.4指定）")
    print("=" * 70)

    delta_hc = -counters["Failed_Attempt→Slow_Decline"]
    delta_sd = counters["Failed_Attempt→Slow_Decline"] - counters["Stagnation→Endurance"] + counters["Steady_Growth→Slow_Decline"]
    delta_end = counters["Stagnation→Endurance"] - counters["Exploration→Pivot_Success"] - counters["Steady_Growth→Pivot_Success"]
    delta_ps = counters["Exploration→Pivot_Success"] + counters["Steady_Growth→Pivot_Success"] - counters["Steady_Growth→Slow_Decline"]
    delta_sr = 0

    actual_delta_hc = after_pt.get("Hubris_Collapse", 0) - before_pt.get("Hubris_Collapse", 0)
    actual_delta_sd = after_pt.get("Slow_Decline", 0) - before_pt.get("Slow_Decline", 0)
    actual_delta_end = after_pt.get("Endurance", 0) - before_pt.get("Endurance", 0)
    actual_delta_ps = after_pt.get("Pivot_Success", 0) - before_pt.get("Pivot_Success", 0)
    actual_delta_sr = after_pt.get("Shock_Recovery", 0) - before_pt.get("Shock_Recovery", 0)

    checks = [
        ("Hubris_Collapse", delta_hc, actual_delta_hc),
        ("Slow_Decline", delta_sd, actual_delta_sd),
        ("Endurance", delta_end, actual_delta_end),
        ("Pivot_Success", delta_ps, actual_delta_ps),
        ("Shock_Recovery", delta_sr, actual_delta_sr),
    ]

    all_ok = True
    for name, expected, actual in checks:
        ok = expected == actual
        status = "OK" if ok else "NG"
        if not ok:
            all_ok = False
        print(f"  Delta_{name}: 計算値={expected:+d}, 実測値={actual:+d} [{status}]")

    if all_ok:
        print("\n  >>> 全検証式 OK <<<")
    else:
        print("\n  >>> 検証式に不一致あり！ <<<")

    # --- enum検証 ---
    print("\n" + "=" * 70)
    print("enum検証（schema_v3準拠チェック）")
    print("=" * 70)

    errors = validate_enum(current)
    if errors:
        print(f"  {len(errors)}件のenum違反を検出:")
        for e in errors[:20]:
            print(e)
        if len(errors) > 20:
            print(f"  ... 他 {len(errors) - 20}件")
    else:
        print("  全件OK: 全enumフィールドがschema_v3準拠")

    # --- 保存 ---
    if dry_run:
        print("\n" + "=" * 70)
        print("DRY-RUN 完了: ファイルは変更されていません")
        print("=" * 70)
    else:
        if not all_ok:
            print("\n検証式に不一致があるため、保存を中止します。")
            sys.exit(1)

        # 自動バックアップ
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = CASES_FILE.parent / f"cases_backup_{timestamp}.jsonl"
        shutil.copy2(CASES_FILE, backup_path)
        print(f"\n自動バックアップ: {backup_path}")

        # 保存
        save_jsonl(current, CASES_FILE)
        print(f"保存完了: {CASES_FILE}")

        # 保存後検証
        reloaded = load_jsonl(CASES_FILE)
        if len(reloaded) != len(current):
            print(f"ERROR: 保存後レコード数不一致 {len(reloaded)} != {len(current)}")
            sys.exit(1)

        reload_pt = Counter(r.get("pattern_type") for r in reloaded)
        if reload_pt != after_pt:
            print("ERROR: 保存後pattern_type分布が不一致")
            sys.exit(1)

        print("保存後検証: OK")

    # --- 総変換件数 ---
    total_changes = (
        counters["before_state: 安定成長・成功→安定・平和"]
        + counters["action_type: 逃げる・守る→耐える・潜伏"]
        + counters["Failed_Attempt→Slow_Decline"]
        + counters["Stagnation→Endurance"]
        + counters["Exploration→Pivot_Success"]
        + counters["Steady_Growth→Pivot_Success"]
        + counters["Steady_Growth→Slow_Decline"]
    )
    print(f"\n総変換件数: {total_changes}件（据え置き除く）")


if __name__ == "__main__":
    main()
