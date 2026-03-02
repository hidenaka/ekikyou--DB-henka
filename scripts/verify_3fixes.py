#!/usr/bin/env python3
"""
BacktraceEngine 3つの構造的修正（Fix1/Fix2/Fix3）の実動作検証スクリプト

Fix1: ファジーマッチング — 状態ラベルのエイリアス解決
Fix2: CI計算 — n<10でCI幅>0.15
Fix3: 行動語彙統一 — 八卦名が混入しない

実際のfull_backtrace返り値のキー構造:
  - l1_yao (小文字) — hamming_distance, changing_lines, ...
  - l2_state (小文字) — case_count, matched_goal_state, ...
  - l3_action (小文字) — routes, action_recommendations, ...
  - recommended_routes (トップレベル) — title, route, confidence_interval, action_recommendations, ...
  - quality_gates (dict, warningsなし)
"""
import sys
import json

sys.path.insert(0, "/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB")
from scripts.backtrace_engine import BacktraceEngine

engine = BacktraceEngine()

scenarios = [
    {"name": "シナリオ1: 坎為水→乾為天（どん底→大成功）", "current_hex": 29, "current_state": "どん底・危機", "goal_hex": 1, "goal_state": "持続成長・大成功"},
    {"name": "シナリオ2: 天地否→地天泰（停滞→回復）", "current_hex": 12, "current_state": "停滞・閉塞", "goal_hex": 11, "goal_state": "安定成長・順調"},
    {"name": "シナリオ3: 水雷屯→火水未済（混乱→新生）", "current_hex": 3, "current_state": "不安定・混乱", "goal_hex": 63, "goal_state": "変質・新生"},
    {"name": "シナリオ4: 山地剥→地雷復（衰退→復活）", "current_hex": 23, "current_state": "衰退・下降", "goal_hex": 24, "goal_state": "V字回復・大成功"},
    {"name": "シナリオ5: 地風升→火天大有（成長→繁栄）", "current_hex": 46, "current_state": "安定成長・成功", "goal_hex": 14, "goal_state": "持続成長・大成功"},
]

RAW_TRIGRAMS = {"乾", "坤", "震", "巽", "坎", "離", "艮", "兌"}

results = []
for s in scenarios:
    print(f"\n{'='*70}")
    print(f"■ {s['name']}")
    print(f"  入力: hex {s['current_hex']}→{s['goal_hex']}, state: {s['current_state']}→{s['goal_state']}")

    result = engine.full_backtrace(s["current_hex"], s["current_state"], s["goal_hex"], s["goal_state"])

    # ----- 構造抽出 -----
    l1 = result.get("l1_yao", {})
    l2 = result.get("l2_state", {})
    l3 = result.get("l3_action", {})
    top_routes = result.get("recommended_routes", [])
    qg = result.get("quality_gates", {})

    # ----- Fix1: ファジーマッチ効果 -----
    l2_cases = l2.get("case_count", 0)
    matched_state = l2.get("matched_goal_state", "N/A")
    fuzzy_ok = l2_cases > 0

    # ----- Fix2: CI正直さ -----
    # recommended_routes内のconfidence_intervalとrouteのsteps内のcountを使う
    ci_honest = True
    ci_details = []
    for i, rt in enumerate(top_routes):
        ci = rt.get("confidence_interval", {})
        ci_lower = ci.get("lower", 0)
        ci_upper = ci.get("upper", 1)
        ci_width = ci_upper - ci_lower

        # nの取得: route.steps内のcount合計 or direct_pair_stats.total_count
        route_data = rt.get("route", {})
        steps = route_data.get("steps", [])
        n = sum(step.get("count", 0) for step in steps)
        # nが0の場合、他の場所から推定
        if n == 0:
            # direct_pair_statsから
            dps = l3.get("direct_pair_stats", {})
            n = dps.get("total_count", 0)

        ci_details.append(
            f"Route{i+1} '{rt.get('title','?')}': "
            f"CI=[{ci_lower:.3f},{ci_upper:.3f}] width={ci_width:.3f} n={n} "
            f"score={rt.get('score', 'N/A')}"
        )
        if n < 10 and ci_width < 0.15:
            ci_honest = False

    # ----- Fix3: 八卦名排除 -----
    # 3箇所をチェック:
    #   (a) top_routes[*].action_recommendations
    #   (b) l3_action.action_recommendations
    #   (c) top_routes[*].route.steps[*].action
    trigram_leak = False
    trigram_leak_details = []

    # (a) トップレベルルートのaction_recommendations
    for i, rt in enumerate(top_routes):
        for a in rt.get("action_recommendations", []):
            name = a.get("action_type", "") if isinstance(a, dict) else str(a)
            if name in RAW_TRIGRAMS:
                trigram_leak = True
                trigram_leak_details.append(f"  routes[{i}].action_recs: '{name}'")

    # (b) l3_action.action_recommendations
    for a in l3.get("action_recommendations", []):
        name = a.get("action_type", "") if isinstance(a, dict) else str(a)
        if name in RAW_TRIGRAMS:
            trigram_leak = True
            trigram_leak_details.append(f"  l3.action_recs: '{name}'")

    # (c) ルートのsteps内のaction
    for i, rt in enumerate(top_routes):
        route_data = rt.get("route", {})
        for j, step in enumerate(route_data.get("steps", [])):
            action = step.get("action", "")
            if action in RAW_TRIGRAMS:
                trigram_leak = True
                trigram_leak_details.append(f"  routes[{i}].steps[{j}].action: '{action}'")

    # (d) l3.routes内のsteps.action
    for i, rt in enumerate(l3.get("routes", [])):
        route_data = rt.get("route", {})
        for j, step in enumerate(route_data.get("steps", [])):
            action = step.get("action", "")
            if action in RAW_TRIGRAMS:
                trigram_leak = True
                trigram_leak_details.append(f"  l3.routes[{i}].steps[{j}].action: '{action}'")

    # (e) pattern_suggestionsのbefore_hex（八卦名はここでは意味的に正しい場合もある）
    # ただし action_type のみチェック
    for ps in l3.get("pattern_suggestions", []):
        for entry in ps.get("top_entries", []):
            act = entry.get("action_type", "")
            if act in RAW_TRIGRAMS:
                trigram_leak = True
                trigram_leak_details.append(f"  pattern_suggestions.action_type: '{act}'")

    # ----- action一覧（表示用） -----
    action_names = []
    for a in l3.get("action_recommendations", []):
        action_names.append(a.get("action_type", "?") if isinstance(a, dict) else str(a))

    # ----- 判定 -----
    status = "PASS" if (fuzzy_ok and ci_honest and not trigram_leak) else "PARTIAL"
    if not fuzzy_ok:
        status = "FAIL"

    # ----- 出力 -----
    print(f"  ★判定: {status}")
    print(f"")
    print(f"  [Fix1] ファジーマッチ: {'OK' if fuzzy_ok else 'NG'}")
    print(f"    L2事例数      = {l2_cases}")
    print(f"    入力goal_state = {s['goal_state']}")
    print(f"    matched_state  = {matched_state}")
    print(f"    goal_reachability = {l2.get('goal_reachability', 'N/A')}")
    print(f"")
    print(f"  [Fix2] CI正直さ: {'OK' if ci_honest else 'NG'}")
    for cd in ci_details:
        print(f"    {cd}")
    if not ci_details:
        print(f"    (ルートなし)")
    print(f"")
    print(f"  [Fix3] 八卦名排除: {'OK' if not trigram_leak else 'NG'}")
    print(f"    action_recommendations = {action_names[:5]}")
    if trigram_leak_details:
        print(f"    !! 八卦名リーク発見:")
        for d in trigram_leak_details:
            print(f"      {d}")
    print(f"")
    print(f"  [L1] hamming_distance = {l1.get('hamming_distance', 'N/A')}, changing_lines = {l1.get('changing_lines', 'N/A')}")
    print(f"  [L1] difficulty = {l1.get('difficulty', 'N/A')}")
    print(f"  [QG] {qg}")

    results.append({
        "name": s["name"],
        "status": status,
        "fuzzy": fuzzy_ok,
        "ci": ci_honest,
        "trigram": not trigram_leak,
        "l2_cases": l2_cases,
        "matched_state": matched_state,
        "trigram_leak_details": trigram_leak_details,
    })

print(f"\n\n{'='*70}")
print("■ 総合結果")
print(f"{'─'*70}")
for r in results:
    fix1 = "OK" if r["fuzzy"] else "NG"
    fix2 = "OK" if r["ci"] else "NG"
    fix3 = "OK" if r["trigram"] else "NG"
    print(f"  {r['status']:7s} | {r['name']}")
    print(f"          Fix1={fix1} (n={r['l2_cases']}, matched='{r['matched_state']}'), Fix2={fix2}, Fix3={fix3}")
    if r["trigram_leak_details"]:
        for d in r["trigram_leak_details"]:
            print(f"          !! {d.strip()}")

pass_count = sum(1 for r in results if r["status"] == "PASS")
partial_count = sum(1 for r in results if r["status"] == "PARTIAL")
fail_count = sum(1 for r in results if r["status"] == "FAIL")
print(f"\n  PASS: {pass_count}/5, PARTIAL: {partial_count}/5, FAIL: {fail_count}/5")
