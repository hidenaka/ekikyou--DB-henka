#!/usr/bin/env python3
"""
Shadow Run 20: BacktraceEngineの品質を20の層化サンプリングシナリオで検証する。

層化軸:
  - scale: company(5), individual(5), country(4), family(3), other(3)
  - pattern_type: 全5パターンをカバー
  - 異なる from_hex -> to_hex の組み合わせ

各シナリオで記録:
  - シナリオ名（scale + from_hex + to_hex）
  - routes数
  - actions数
  - polysemy件数
  - quality_weight
  - PASS/FAIL/WARN
"""

import json
import os
import sys
import traceback
from datetime import datetime

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TEST_DIR)
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")

if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backtrace_engine import BacktraceEngine

# ============================================================
# 20 Stratified Scenarios
# ============================================================
# scale distribution: company(5), individual(5), country(4), family(3), other(3)
# pattern_type coverage: each of 5 types appears 4 times
# hex pairs: all unique from_hex -> to_hex combinations

SCENARIOS = [
    # --- company (5) ---
    {
        "id": "C01",
        "name": "company: 坎為水(29)->火天大有(14) Shock_Recovery",
        "params": {
            "current_hex": 29, "current_state": "どん底・危機",
            "goal_hex": 14, "goal_state": "持続成長・大成功",
            "scale": "company",
        },
        "expected_pattern": "Shock_Recovery",
    },
    {
        "id": "C02",
        "name": "company: 乾為天(1)->天地否(12) Hubris_Collapse",
        "params": {
            "current_hex": 1, "current_state": "成長・拡大",
            "goal_hex": 12, "goal_state": "停滞・閉塞",
            "scale": "company",
        },
        "expected_pattern": "Hubris_Collapse",
    },
    {
        "id": "C03",
        "name": "company: 山水蒙(4)->風火家人(37) Pivot_Success",
        "params": {
            "current_hex": 4, "current_state": "混乱・カオス",
            "goal_hex": 37, "goal_state": "安定・平和",
            "scale": "company",
        },
        "expected_pattern": "Pivot_Success",
    },
    {
        "id": "C04",
        "name": "company: 地天泰(11)->地雷復(24) Endurance",
        "params": {
            "current_hex": 11, "current_state": "安定・平和",
            "goal_hex": 24, "goal_state": "V字回復・大成功",
            "scale": "company",
        },
        "expected_pattern": "Endurance",
    },
    {
        "id": "C05",
        "name": "company: 火天大有(14)->山地剥(23) Slow_Decline",
        "params": {
            "current_hex": 14, "current_state": "持続成長・大成功",
            "goal_hex": 23, "goal_state": "崩壊・消滅",
            "scale": "company",
        },
        "expected_pattern": "Slow_Decline",
    },
    # --- individual (5) ---
    {
        "id": "I01",
        "name": "individual: 天地否(12)->地天泰(11) Shock_Recovery",
        "params": {
            "current_hex": 12, "current_state": "停滞・閉塞",
            "goal_hex": 11, "goal_state": "安定・平和",
            "scale": "individual",
        },
        "expected_pattern": "Shock_Recovery",
    },
    {
        "id": "I02",
        "name": "individual: 離為火(30)->坎為水(29) Hubris_Collapse",
        "params": {
            "current_hex": 30, "current_state": "拡大・繁栄",
            "goal_hex": 29, "goal_state": "どん底・危機",
            "scale": "individual",
        },
        "expected_pattern": "Hubris_Collapse",
    },
    {
        "id": "I03",
        "name": "individual: 水雷屯(3)->風天小畜(9) Pivot_Success",
        "params": {
            "current_hex": 3, "current_state": "混乱・カオス",
            "goal_hex": 9, "goal_state": "安定成長・成功",
            "scale": "individual",
        },
        "expected_pattern": "Pivot_Success",
    },
    {
        "id": "I04",
        "name": "individual: 艮為山(52)->雷地豫(16) Endurance",
        "params": {
            "current_hex": 52, "current_state": "安定・停止",
            "goal_hex": 16, "goal_state": "成長・拡大",
            "scale": "individual",
        },
        "expected_pattern": "Endurance",
    },
    {
        "id": "I05",
        "name": "individual: 兌為沢(58)->山風蠱(18) Slow_Decline",
        "params": {
            "current_hex": 58, "current_state": "安定成長・成功",
            "goal_hex": 18, "goal_state": "混乱・衰退",
            "scale": "individual",
        },
        "expected_pattern": "Slow_Decline",
    },
    # --- country (4) ---
    {
        "id": "N01",
        "name": "country: 山地剥(23)->地雷復(24) Shock_Recovery",
        "params": {
            "current_hex": 23, "current_state": "崩壊・消滅",
            "goal_hex": 24, "goal_state": "V字回復・大成功",
            "scale": "country",
        },
        "expected_pattern": "Shock_Recovery",
    },
    {
        "id": "N02",
        "name": "country: 雷火豊(55)->火山旅(56) Hubris_Collapse",
        "params": {
            "current_hex": 55, "current_state": "拡大・繁栄",
            "goal_hex": 56, "goal_state": "縮小安定・生存",
            "scale": "country",
        },
        "expected_pattern": "Hubris_Collapse",
    },
    {
        "id": "N03",
        "name": "country: 水地比(8)->天火同人(13) Pivot_Success",
        "params": {
            "current_hex": 8, "current_state": "安定・平和",
            "goal_hex": 13, "goal_state": "成長・拡大",
            "scale": "country",
        },
        "expected_pattern": "Pivot_Success",
    },
    {
        "id": "N04",
        "name": "country: 風地観(20)->雷天大壮(34) Slow_Decline",
        "params": {
            "current_hex": 20, "current_state": "停滞・閉塞",
            "goal_hex": 34, "goal_state": "成長・拡大",
            "scale": "country",
        },
        "expected_pattern": "Slow_Decline",
    },
    # --- family (3) ---
    {
        "id": "F01",
        "name": "family: 火水未済(64)->水火既済(63) Endurance",
        "params": {
            "current_hex": 64, "current_state": "混乱・カオス",
            "goal_hex": 63, "goal_state": "安定・平和",
            "scale": "family",
        },
        "expected_pattern": "Endurance",
    },
    {
        "id": "F02",
        "name": "family: 風火家人(37)->火雷噬嗑(21) Pivot_Success",
        "params": {
            "current_hex": 37, "current_state": "安定・平和",
            "goal_hex": 21, "goal_state": "変質・新生",
            "scale": "family",
        },
        "expected_pattern": "Pivot_Success",
    },
    {
        "id": "F03",
        "name": "family: 沢山咸(31)->雷風恒(32) Slow_Decline",
        "params": {
            "current_hex": 31, "current_state": "成長・拡大",
            "goal_hex": 32, "goal_state": "現状維持・延命",
            "scale": "family",
        },
        "expected_pattern": "Slow_Decline",
    },
    # --- other (3) ---
    {
        "id": "O01",
        "name": "other: 震為雷(51)->巽為風(57) Shock_Recovery",
        "params": {
            "current_hex": 51, "current_state": "混乱・カオス",
            "goal_hex": 57, "goal_state": "安定成長・成功",
            "scale": "other",
        },
        "expected_pattern": "Shock_Recovery",
    },
    {
        "id": "O02",
        "name": "other: 天山遯(33)->地風升(46) Endurance",
        "params": {
            "current_hex": 33, "current_state": "縮小安定・生存",
            "goal_hex": 46, "goal_state": "成長・拡大",
            "scale": "other",
        },
        "expected_pattern": "Endurance",
    },
    {
        "id": "O03",
        "name": "other: 坤為地(2)->乾為天(1) Pivot_Success",
        "params": {
            "current_hex": 2, "current_state": "停滞・閉塞",
            "goal_hex": 1, "goal_state": "持続成長・大成功",
            "scale": "other",
        },
        "expected_pattern": "Pivot_Success",
    },
]


def check_actions_japanese(actions):
    """recommended_actionsが八卦名ではなく日本語action_typeかチェック"""
    trigram_names = {"乾", "坤", "震", "巽", "坎", "離", "艮", "兌"}
    for a in actions:
        at = a.get("action_type", "")
        if at in trigram_names:
            return False, f"八卦名がaction_typeに含まれている: {at}"
    return True, "OK"


def run_scenario(engine, scenario):
    """1シナリオを実行して結果辞書を返す"""
    sid = scenario["id"]
    name = scenario["name"]
    params = scenario["params"]
    record = {
        "id": sid,
        "name": name,
        "scale": params["scale"],
        "from_hex": params["current_hex"],
        "to_hex": params["goal_hex"],
        "current_state": params["current_state"],
        "goal_state": params["goal_state"],
        "expected_pattern": scenario.get("expected_pattern", ""),
        "status": "PASS",
        "routes": 0,
        "actions": 0,
        "polysemy": 0,
        "quality_weight": None,
        "warnings": [],
        "details": {},
        "error": None,
    }

    # 1. Execute full_backtrace
    try:
        result = engine.full_backtrace(**params)
    except Exception as e:
        record["status"] = "FAIL"
        record["error"] = str(e)
        record["details"]["execution"] = f"FAIL: {e}"
        return record

    if "error" in result:
        record["status"] = "FAIL"
        record["error"] = result["error"]
        record["details"]["execution"] = f"FAIL: API error: {result['error']}"
        return record

    record["details"]["execution"] = "PASS"

    # 2. Routes
    routes = result.get("recommended_routes", [])
    record["routes"] = len(routes)
    if len(routes) == 0:
        record["status"] = "WARN"
        record["warnings"].append("0 routes")
        record["details"]["routes"] = "WARN: 0 routes"
    else:
        record["details"]["routes"] = f"PASS: {len(routes)} routes"

    # 3. Actions
    l2 = result.get("l2_state", {})
    actions = l2.get("recommended_actions", [])
    record["actions"] = len(actions)
    if len(actions) == 0:
        record["warnings"].append("0 actions")
        record["details"]["actions"] = "WARN: 0 actions"
    else:
        is_jp, msg = check_actions_japanese(actions)
        if is_jp:
            action_types = [a.get("action_type", "?") for a in actions[:3]]
            record["details"]["actions"] = f"PASS: {len(actions)} actions, top3={action_types}"
        else:
            record["status"] = "FAIL"
            record["details"]["actions"] = f"FAIL: {msg}"

    # 4. Polysemy warnings
    pw = result.get("polysemy_warnings", [])
    record["polysemy"] = len(pw)
    if pw:
        pw_types = [w.get("action_type", "?") for w in pw[:3]]
        record["details"]["polysemy"] = f"{len(pw)} warnings, types={pw_types}"
    else:
        record["details"]["polysemy"] = "0 (none)"

    # 5. Quality info
    qi = result.get("quality_info", {})
    qw = qi.get("quality_weight", None)
    total = qi.get("total", None)
    record["quality_weight"] = qw
    record["details"]["quality_weight"] = f"quality_weight={qw}, total={total}"

    # 6. Low quality evidence flags
    low_q_routes = [r.get("title", "?") for r in routes if "low_quality_evidence" in r.get("labels", [])]
    if low_q_routes:
        record["details"]["low_quality_evidence"] = f"FLAGGED: {low_q_routes[:3]}"
    else:
        record["details"]["low_quality_evidence"] = "OK"

    # 7. Cross-scale patterns (check if present)
    csp = result.get("cross_scale_patterns", None)
    if csp is not None:
        if isinstance(csp, list):
            record["details"]["cross_scale"] = f"{len(csp)} patterns"
        elif isinstance(csp, dict):
            record["details"]["cross_scale"] = f"dict with {len(csp)} keys"

    return record


def main():
    print("=" * 70)
    print("Shadow Run 20: BacktraceEngine Stratified Scenario Test")
    print(f"Date: {datetime.now().isoformat()}")
    print("=" * 70)

    print("\nInitializing BacktraceEngine...")
    engine = BacktraceEngine()
    print("Engine initialized.\n")

    all_results = []
    pass_count = 0
    fail_count = 0
    warn_count = 0

    for scenario in SCENARIOS:
        print(f"\n{'─' * 60}")
        print(f"[{scenario['id']}] {scenario['name']}")
        print(f"{'─' * 60}")

        record = run_scenario(engine, scenario)
        all_results.append(record)

        status = record["status"]
        if status == "PASS":
            pass_count += 1
        elif status == "FAIL":
            fail_count += 1
        else:
            warn_count += 1

        print(f"  Status: {status}")
        print(f"  Routes: {record['routes']}, Actions: {record['actions']}, "
              f"Polysemy: {record['polysemy']}, QW: {record['quality_weight']}")
        if record["error"]:
            print(f"  Error: {record['error']}")
        for key, val in record["details"].items():
            print(f"    {key}: {val}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'ID':<5} {'Scenario':<55} {'Routes':>6} {'Acts':>5} {'Poly':>5} {'QW':>6} {'Status':>6}")
    print(f"{'─' * 5} {'─' * 55} {'─' * 6} {'─' * 5} {'─' * 5} {'─' * 6} {'─' * 6}")
    for r in all_results:
        qw_str = f"{r['quality_weight']:.2f}" if r['quality_weight'] is not None else "N/A"
        print(f"{r['id']:<5} {r['name']:<55} {r['routes']:>6} {r['actions']:>5} "
              f"{r['polysemy']:>5} {qw_str:>6} {r['status']:>6}")

    print(f"{'─' * 5} {'─' * 55} {'─' * 6} {'─' * 5} {'─' * 5} {'─' * 6} {'─' * 6}")
    print(f"Total: PASS={pass_count}, WARN={warn_count}, FAIL={fail_count} / {len(all_results)}")
    print()

    # Scale breakdown
    print("By Scale:")
    for scale in ["company", "individual", "country", "family", "other"]:
        subset = [r for r in all_results if r["scale"] == scale]
        sp = sum(1 for r in subset if r["status"] == "PASS")
        sw = sum(1 for r in subset if r["status"] == "WARN")
        sf = sum(1 for r in subset if r["status"] == "FAIL")
        print(f"  {scale:>12}: PASS={sp}, WARN={sw}, FAIL={sf} / {len(subset)}")

    # Pattern breakdown
    print("\nBy Expected Pattern:")
    for pt in ["Shock_Recovery", "Hubris_Collapse", "Pivot_Success", "Endurance", "Slow_Decline"]:
        subset = [r for r in all_results if r["expected_pattern"] == pt]
        sp = sum(1 for r in subset if r["status"] == "PASS")
        sw = sum(1 for r in subset if r["status"] == "WARN")
        sf = sum(1 for r in subset if r["status"] == "FAIL")
        print(f"  {pt:>20}: PASS={sp}, WARN={sw}, FAIL={sf} / {len(subset)}")

    # ============================================================
    # Scale Quality Summary (family scale monitoring)
    # ============================================================
    print(f"\n{'=' * 70}")
    print("Scale Quality Summary")
    print(f"{'=' * 70}")
    for scale in ["company", "individual", "country", "family", "other"]:
        subset = [r for r in all_results if r["scale"] == scale]
        if not subset:
            continue
        qw_vals = [r["quality_weight"] for r in subset if r["quality_weight"] is not None]
        route_vals = [r["routes"] for r in subset]
        action_vals = [r["actions"] for r in subset]
        sp = sum(1 for r in subset if r["status"] == "PASS")
        sw = sum(1 for r in subset if r["status"] == "WARN")
        sf = sum(1 for r in subset if r["status"] == "FAIL")
        avg_qw = sum(qw_vals) / len(qw_vals) if qw_vals else 0.0
        avg_routes = sum(route_vals) / len(route_vals) if route_vals else 0.0
        avg_actions = sum(action_vals) / len(action_vals) if action_vals else 0.0
        warn_flag = " << LOW QW WARNING" if avg_qw < 0.70 else ""
        print(f"  {scale:12s}: scenarios={len(subset)}, "
              f"avg_routes={avg_routes:.1f}, avg_actions={avg_actions:.1f}, "
              f"avg_qw={avg_qw:.4f}, "
              f"PASS={sp}/WARN={sw}/FAIL={sf}"
              f"{warn_flag}")
    # family-specific alert
    family_subset = [r for r in all_results if r["scale"] == "family"]
    family_qw_vals = [r["quality_weight"] for r in family_subset if r["quality_weight"] is not None]
    if family_qw_vals:
        family_avg_qw = sum(family_qw_vals) / len(family_qw_vals)
        if family_avg_qw < 0.70:
            print(f"\n  WARNING: family scale avg_qw={family_avg_qw:.4f} < 0.70 threshold")
            print(f"  GPT-5.4 flagged quality_weight=0.68 as relatively low.")
            print(f"  Review family-scale evidence quality in data/raw/cases.jsonl")

    # ============================================================
    # Save results to JSON
    # ============================================================
    output_path = "/tmp/schema_normalization/shadow_run_20_results.json"
    output = {
        "run_date": datetime.now().isoformat(),
        "total_scenarios": len(all_results),
        "pass": pass_count,
        "warn": warn_count,
        "fail": fail_count,
        "cases_count": len(engine.cases) if hasattr(engine, 'cases') else sum(1 for _ in open(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'cases.jsonl'), encoding='utf-8') if _.strip()),
        "scenarios": all_results,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
