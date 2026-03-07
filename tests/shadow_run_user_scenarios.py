#!/usr/bin/env python3
"""
Shadow Run: 5つのユーザーシナリオでBacktraceEngineの品質を検証する。
"""

import json
import os
import sys
import traceback

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TEST_DIR)
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")

if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backtrace_engine import BacktraceEngine

# ============================================================
# シナリオ定義
# ============================================================
SCENARIOS = [
    {
        "name": "Scenario 1: Startup Founder (坎為水 -> 火天大有)",
        "params": {
            "current_hex": 29,
            "current_state": "どん底・危機",
            "goal_hex": 14,
            "goal_state": "持続成長・大成功",
            "scale": "company",
        },
        "checks": {
            "routes_exist": True,
            "actions_japanese": True,
            "cross_scale": False,
        },
    },
    {
        "name": "Scenario 2: Career Change (天地否 -> 地天泰)",
        "params": {
            "current_hex": 12,
            "current_state": "停滞",       # ファジーマッチングテスト（短縮形）
            "goal_hex": 11,
            "goal_state": "安定",           # ファジーマッチングテスト（短縮形）
            "scale": "individual",
        },
        "checks": {
            "routes_exist": True,
            "actions_japanese": True,
            "fuzzy_match": True,
            "cross_scale": False,
        },
    },
    {
        "name": "Scenario 3: National Crisis (山地剥 -> 地雷復)",
        "params": {
            "current_hex": 23,
            "current_state": "崩壊・消滅",
            "goal_hex": 24,
            "goal_state": "V字回復・大成功",
            "scale": "country",
        },
        "checks": {
            "routes_exist": True,
            "actions_japanese": True,
            "scale_specific": True,
            "cross_scale": False,
        },
    },
    {
        "name": "Scenario 4: Family Issue (火水未済 -> 水火既済)",
        "params": {
            "current_hex": 64,
            "current_state": "混乱・カオス",
            "goal_hex": 63,
            "goal_state": "安定・平和",
            "scale": "family",
        },
        "checks": {
            "routes_exist": True,
            "actions_japanese": True,
            "bayes_smoothing": True,
            "cross_scale": False,
        },
    },
    {
        "name": "Scenario 5: Cross-Scale (天地否 -> 乾為天)",
        "params": {
            "current_hex": 12,
            "current_state": "停滞・閉塞",
            "goal_hex": 1,
            "goal_state": "成長・拡大",
            "scale": "individual",
            "include_cross_scale": True,
        },
        "checks": {
            "routes_exist": True,
            "actions_japanese": True,
            "cross_scale": True,
        },
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
    """1シナリオを実行して結果を返す"""
    name = scenario["name"]
    params = scenario["params"]
    checks = scenario["checks"]
    results = {"name": name, "pass": True, "details": {}}

    # 1. full_backtrace がエラーなく完了するか
    try:
        result = engine.full_backtrace(**params)
        results["details"]["execution"] = "PASS"
    except Exception as e:
        results["details"]["execution"] = f"FAIL: {e}\n{traceback.format_exc()}"
        results["pass"] = False
        return results

    # エラーレスポンスチェック
    if "error" in result:
        results["details"]["execution"] = f"FAIL: API returned error: {result['error']}"
        results["pass"] = False
        return results

    # 2. routes数
    routes = result.get("recommended_routes", [])
    route_count = len(routes)
    if route_count == 0 and checks.get("routes_exist"):
        results["details"]["routes"] = f"FAIL: 0 routes"
        results["pass"] = False
    else:
        results["details"]["routes"] = f"PASS: {route_count} routes"

    # 3. recommended_actions の内容
    l2 = result.get("l2_state", {})
    actions = l2.get("recommended_actions", [])
    action_count = len(actions)
    if action_count == 0:
        results["details"]["actions"] = "WARN: 0 recommended_actions"
    else:
        is_jp, msg = check_actions_japanese(actions)
        if is_jp:
            action_types = [a.get("action_type", "?") for a in actions[:3]]
            results["details"]["actions"] = f"PASS: {action_count} actions, top3={action_types}"
        else:
            results["details"]["actions"] = f"FAIL: {msg}"
            results["pass"] = False

    # 4. polysemy_warnings の有無と内容
    pw = result.get("polysemy_warnings", [])
    if pw:
        pw_summary = [w.get("action_type", "?") for w in pw[:3]]
        results["details"]["polysemy"] = f"PRESENT: {len(pw)} warnings, types={pw_summary}"
    else:
        results["details"]["polysemy"] = "NONE (no polysemy warnings)"

    # 5. quality_info の quality_weight と low_quality_evidence フラグ
    qi = result.get("quality_info", {})
    qw = qi.get("quality_weight", None)
    total = qi.get("total", None)
    results["details"]["quality_weight"] = f"quality_weight={qw}, total={total}"

    # low_quality_evidence フラグはルートのlabelsに含まれる
    low_q_routes = [r.get("title", "?") for r in routes if "low_quality_evidence" in r.get("labels", [])]
    if low_q_routes:
        results["details"]["low_quality_evidence"] = f"FLAGGED on: {low_q_routes[:3]}"
    else:
        results["details"]["low_quality_evidence"] = "NOT flagged (quality OK)"

    # Scenario-specific checks

    # ファジーマッチング (Scenario 2)
    if checks.get("fuzzy_match"):
        # 短縮形の「停滞」「安定」が正しく処理されたか
        # エラーなく結果が返ってきていれば成功
        results["details"]["fuzzy_match"] = "PASS: short-form state labels accepted"

    # scale固有事例 (Scenario 3)
    if checks.get("scale_specific"):
        scale_note = l2.get("scale_fallback_note", "")
        results["details"]["scale_specific"] = f"scale_fallback_note='{scale_note}'"

    # ベイズ平滑化 (Scenario 4)
    if checks.get("bayes_smoothing"):
        # family (少数データ) でquality_weightが0.5-1.0の間にあるか
        if qw is not None and 0.3 <= qw <= 1.0:
            results["details"]["bayes_smoothing"] = f"PASS: quality_weight={qw} (in range)"
        else:
            results["details"]["bayes_smoothing"] = f"WARN: quality_weight={qw} (unexpected range)"

    # クロススケール (Scenario 5)
    if checks.get("cross_scale"):
        csp = result.get("cross_scale_patterns", None)
        if csp is not None:
            if isinstance(csp, list):
                results["details"]["cross_scale"] = f"PASS: {len(csp)} cross-scale patterns"
            elif isinstance(csp, dict):
                results["details"]["cross_scale"] = f"PASS: cross_scale_patterns returned (dict with {len(csp)} keys)"
            else:
                results["details"]["cross_scale"] = f"PASS: cross_scale_patterns type={type(csp).__name__}"
        else:
            results["details"]["cross_scale"] = "FAIL: cross_scale_patterns is None"
            results["pass"] = False

    return results


def main():
    print("=" * 70)
    print("Shadow Run: BacktraceEngine User Scenario Test")
    print("=" * 70)

    print("\nInitializing BacktraceEngine...")
    engine = BacktraceEngine()
    print("Engine initialized.\n")

    all_results = []
    for scenario in SCENARIOS:
        print(f"\n{'─' * 60}")
        print(f"Running: {scenario['name']}")
        print(f"{'─' * 60}")

        result = run_scenario(engine, scenario)
        all_results.append(result)

        status = "PASS" if result["pass"] else "FAIL"
        print(f"  Status: {status}")
        for key, val in result["details"].items():
            print(f"    {key}: {val}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Scenario':<55} {'Result':>6}")
    print(f"{'─' * 55} {'─' * 6}")
    pass_count = 0
    for r in all_results:
        status = "PASS" if r["pass"] else "FAIL"
        if r["pass"]:
            pass_count += 1
        print(f"{r['name']:<55} {status:>6}")
    print(f"{'─' * 55} {'─' * 6}")
    print(f"{'Total':.<55} {pass_count}/{len(all_results)}")
    print()

    if pass_count == len(all_results):
        print("All scenarios passed.")
    else:
        print(f"WARNING: {len(all_results) - pass_count} scenario(s) failed.")

    return 0 if pass_count == len(all_results) else 1


if __name__ == "__main__":
    sys.exit(main())
