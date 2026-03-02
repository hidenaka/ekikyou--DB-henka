#!/usr/bin/env python3
"""
backtrace_cli.py — 易経逆算エンジン CLI

対話型ウィザードで「なりたい姿」から逆算ルートを表示する。

Usage:
    # 対話モード
    python3 scripts/backtrace_cli.py

    # クイックモード
    python3 scripts/backtrace_cli.py --quick --goal 1 --current 2
    python3 scripts/backtrace_cli.py --quick --goal 11 --current 12 --goal-state 安定・平和 --current-state 停滞・閉塞
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# パス設定
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from backtrace_engine import BacktraceEngine

# ---------------------------------------------------------------------------
# 卦データ読み込み
# ---------------------------------------------------------------------------

_HEXAGRAM_64_PATH = os.path.join(
    _PROJECT_ROOT, "data", "diagnostic", "hexagram_64.json"
)


def _load_hexagram_data() -> Dict:
    """hexagram_64.json を読み込み、番号→情報の辞書を返す。"""
    if not os.path.isfile(_HEXAGRAM_64_PATH):
        return {}
    with open(_HEXAGRAM_64_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # hexagrams セクションを番号引きに変換
    result = {}
    for name, info in data.get("hexagrams", {}).items():
        num = info.get("number")
        if num:
            result[num] = {**info, "full_name": name}
    return result


_HEX_DATA = _load_hexagram_data()

# ---------------------------------------------------------------------------
# 状態ラベル一覧（rev_after_state.json のキー）
# ---------------------------------------------------------------------------

_VALID_STATES = [
    "V字回復・大成功",
    "どん底・危機",
    "停滞・閉塞",
    "分岐・様子見",
    "喜び・交流",
    "変質・新生",
    "安定・停止",
    "安定・平和",
    "安定成長・成功",
    "崩壊・消滅",
    "成長・拡大",
    "成長痛",
    "拡大・繁栄",
    "持続成長・大成功",
    "消滅・破綻",
    "混乱・カオス",
    "混乱・衰退",
    "現状維持・延命",
    "縮小安定・生存",
    "迷走・混乱",
]

# テーマキーワード→目標卦のマッピング
_THEME_TO_GOAL: Dict[str, Dict] = {
    "リーダーシップ": {"hex": 1, "state": "持続成長・大成功"},
    "成長": {"hex": 11, "state": "安定成長・成功"},
    "安定": {"hex": 11, "state": "安定・平和"},
    "変革": {"hex": 49, "state": "変質・新生"},
    "回復": {"hex": 24, "state": "V字回復・大成功"},
    "協力": {"hex": 13, "state": "安定成長・成功"},
    "繁栄": {"hex": 14, "state": "拡大・繁栄"},
    "学び": {"hex": 4, "state": "成長・拡大"},
    "忍耐": {"hex": 5, "state": "安定・平和"},
    "決断": {"hex": 43, "state": "V字回復・大成功"},
    "撤退": {"hex": 33, "state": "縮小安定・生存"},
    "改革": {"hex": 18, "state": "変質・新生"},
}


# ---------------------------------------------------------------------------
# 表示ヘルパー
# ---------------------------------------------------------------------------


def _hex_label(num: int) -> str:
    """卦番号からラベル文字列を生成する。例: '乾為天 (#1) -- 創造・剛健'"""
    info = _HEX_DATA.get(num)
    if info:
        return f"{info['full_name']} (#{num}) -- {info.get('meaning', '')}"
    return f"卦#{num}"


def _hex_short(num: int) -> str:
    """卦番号から短いラベルを生成する。例: '乾為天(#1)'"""
    info = _HEX_DATA.get(num)
    if info:
        return f"{info['full_name']}(#{num})"
    return f"卦#{num}"


def _yao_name(position: int) -> str:
    """爻位置を日本語名に変換する。"""
    names = {1: "初爻", 2: "二爻", 3: "三爻", 4: "四爻", 5: "五爻", 6: "上爻"}
    return names.get(position, f"第{position}爻")


def _confidence_label(level: str) -> str:
    """信頼度レベルを日本語に変換する。"""
    labels = {
        "high": "高 (事例豊富)",
        "medium": "中 (ある程度の事例あり)",
        "low": "低 (事例少数)",
        "very_low": "極低 (参考情報程度)",
    }
    return labels.get(level, level)


def _difficulty_label(diff: str) -> str:
    """難易度を日本語に変換する。"""
    labels = {
        "trivial": "極めて容易 (同一卦)",
        "easy": "容易 (1爻変化)",
        "moderate": "中程度 (2-3爻変化)",
        "hard": "困難 (4-6爻変化)",
    }
    return labels.get(diff, diff)


# ---------------------------------------------------------------------------
# 対話型ウィザード
# ---------------------------------------------------------------------------


def _prompt_goal() -> Dict:
    """目標を対話で取得する。"""
    print()
    print("  目標の設定方法を選んでください:")
    print("  [1] テーマキーワードで指定")
    print("  [2] 卦番号(1-64)で直接指定")
    print("  [3] デフォルト (地天泰 #11 = 安定・調和)")
    print()

    choice = input("  選択 (1/2/3): ").strip()

    if choice == "1":
        print()
        print("  利用可能なテーマ:")
        themes = list(_THEME_TO_GOAL.keys())
        for i, theme in enumerate(themes, 1):
            t = _THEME_TO_GOAL[theme]
            print(f"    {i:2d}. {theme} -> {_hex_short(t['hex'])}")
        print()
        kw = input("  テーマ名を入力 (例: リーダーシップ): ").strip()
        if kw in _THEME_TO_GOAL:
            goal = _THEME_TO_GOAL[kw]
            return {"hex": goal["hex"], "state": goal["state"]}
        # テーマが見つからない場合、部分一致を試みる
        for k, v in _THEME_TO_GOAL.items():
            if kw in k or k in kw:
                print(f"  -> テーマ「{k}」にマッチしました。")
                return {"hex": v["hex"], "state": v["state"]}
        print("  テーマが見つかりません。デフォルトを使用します。")
        return {"hex": 11, "state": "安定・平和"}

    elif choice == "2":
        print()
        num_str = input("  目標卦の番号 (1-64): ").strip()
        try:
            num = int(num_str)
            if not 1 <= num <= 64:
                raise ValueError
        except ValueError:
            print("  無効な番号です。デフォルト(#11)を使用します。")
            num = 11
        print(f"  -> {_hex_label(num)}")
        print()
        print("  目標状態を選んでください:")
        for i, state in enumerate(_VALID_STATES, 1):
            print(f"    {i:2d}. {state}")
        print()
        state_input = input("  番号またはテキスト入力 (空欄=安定・平和): ").strip()
        if not state_input:
            state = "安定・平和"
        else:
            try:
                idx = int(state_input) - 1
                if 0 <= idx < len(_VALID_STATES):
                    state = _VALID_STATES[idx]
                else:
                    state = "安定・平和"
            except ValueError:
                state = state_input
        return {"hex": num, "state": state}

    else:
        return {"hex": 11, "state": "安定・平和"}


def _prompt_current() -> Dict:
    """現在地を対話で取得する。"""
    print()
    print("  現在の状態を入力してください。")
    print()

    num_str = input("  現在の卦番号 (1-64, 空欄=12[天地否]): ").strip()
    if not num_str:
        num = 12
    else:
        try:
            num = int(num_str)
            if not 1 <= num <= 64:
                raise ValueError
        except ValueError:
            print("  無効な番号です。デフォルト(#12)を使用します。")
            num = 12
    print(f"  -> {_hex_label(num)}")

    print()
    print("  現在の状態を選んでください:")
    for i, state in enumerate(_VALID_STATES, 1):
        print(f"    {i:2d}. {state}")
    print()
    state_input = input("  番号またはテキスト入力 (空欄=停滞・閉塞): ").strip()
    if not state_input:
        state = "停滞・閉塞"
    else:
        try:
            idx = int(state_input) - 1
            if 0 <= idx < len(_VALID_STATES):
                state = _VALID_STATES[idx]
            else:
                state = state_input
        except ValueError:
            state = state_input

    return {"hex": num, "state": state}


# ---------------------------------------------------------------------------
# 結果表示
# ---------------------------------------------------------------------------


def _display_results(result: Dict) -> None:
    """full_backtrace() の結果を読みやすい日本語で表示する。"""
    summary = result["summary"]
    l1 = result["l1_yao"]
    l2 = result["l2_state"]
    l3 = result["l3_action"]
    routes = result["recommended_routes"]
    qg = result["quality_gates"]

    W = 50  # 区切り線の幅

    # ===== ヘッダー =====
    print()
    print("=" * W)
    print("  易経逆算エンジン -- なりたい姿から逆算")
    print("=" * W)
    print()
    print(f"  目標: {_hex_label(summary['goal_hex'])}")
    print(f"        状態: {summary['goal_state']}")
    print()
    print(f"  現在: {_hex_label(summary['current_hex'])}")
    print(f"        状態: {summary['current_state']}")
    print()
    print(f"  信頼度: {_confidence_label(summary['confidence_level'])}")
    print(f"  総合スコア: {summary['primary_route_score']:.2f}")
    print(f"  代替ルート数: {summary['alternative_count']}")

    # ===== R1: 目標地点 =====
    print()
    print("-" * W)
    print("  R1: 目標地点")
    print("-" * W)
    goal_info = _HEX_DATA.get(summary["goal_hex"], {})
    if goal_info:
        print(f"  卦名: {goal_info.get('full_name', '')}")
        print(f"  象意: {goal_info.get('meaning', '')}")
        print(f"  上卦: {goal_info.get('upper', '')}  下卦: {goal_info.get('lower', '')}")
        print(f"  象辞: {goal_info.get('image', '')}")
        if goal_info.get("modern_interpretation"):
            interp = goal_info["modern_interpretation"]
            # 長い場合は折り返し
            if len(interp) > 60:
                print(f"  解釈: {interp[:60]}")
                print(f"        {interp[60:]}")
            else:
                print(f"  解釈: {interp}")

    # ===== R2: ギャップ構造 =====
    print()
    print("-" * W)
    print("  R2: ギャップ構造")
    print("-" * W)
    print(f"  ハミング距離: {l1['hamming_distance']}")
    print(f"  難易度: {_difficulty_label(l1['difficulty'])}")

    if l1["changing_lines"]:
        yao_names = [_yao_name(y) for y in l1["changing_lines"]]
        print(f"  変爻: {', '.join(yao_names)}")

    if l1["direct_yao_path"]:
        print(f"  -> 1爻変化で直接到達可能! (変爻: {_yao_name(l1['direct_yao_position'])})")

    if l1["structural_relationship"]:
        print(f"  構造的関係: {l1['structural_relationship']}")

    if l1["current_is_source"]:
        print(f"  -> 現在の卦は過去事例でも目標卦への遷移が確認されています。")

    # 中間卦の提案
    intermediates = l1.get("intermediate_suggestions", [])
    if intermediates:
        print()
        print(f"  中間卦候補:")
        for inter in intermediates[:3]:
            if isinstance(inter, dict):
                name = inter.get("name", "")
                number = inter.get("number", "")
                role = inter.get("role", "")
                if name and number:
                    label = f"{name}(#{number})"
                elif name:
                    label = name
                else:
                    label = f"卦#{number}"
                if role:
                    label += f" -- {role}"
                print(f"    -> {label}")
            else:
                print(f"    -> {inter}")

    # ===== R2.5: 状態レベル分析 =====
    print()
    print("-" * W)
    print("  R2.5: 状態レベル分析")
    print("-" * W)
    print(f"  「{l2['goal_state']}」に至った過去事例: {l2['case_count']}件")
    print(f"  現在地からの到達可能性: {l2['goal_reachability']:.1%}")
    print(f"  {l2['confidence_note']}")

    if l2.get("before_state_distribution"):
        print()
        print(f"  前状態の分布 (上位5件):")
        for entry in l2["before_state_distribution"][:5]:
            bar_len = int(entry.get("pct", 0) / 5)
            bar = "#" * bar_len
            print(f"    {entry['state']:16s} {entry.get('count', 0):4d}件 ({entry.get('pct', 0):5.1f}%) {bar}")

    if l2.get("recommended_actions"):
        print()
        print(f"  推奨行動タイプ (状態分析):")
        for action in l2["recommended_actions"][:3]:
            print(
                f"    {action['action_type']:16s} "
                f"頻度={action['frequency_pct']:.1f}% "
                f"成功事例={action['success_case_count']}件 "
                f"スコア={action['composite_score']:.3f}"
            )

    # ===== R3: 推奨ルート =====
    print()
    print("-" * W)
    print("  R3: 推奨ルート")
    print("-" * W)

    if not routes:
        print("  ルートが見つかりませんでした。")
    else:
        for i, r in enumerate(routes, 1):
            route_data = r.get("route", {})
            ci = r.get("confidence_interval", {})
            labels = r.get("labels", [])
            label_str = ""
            if labels:
                label_tags = []
                if "reference_only" in labels:
                    label_tags.append("参考情報")
                if "fallback" in labels:
                    label_tags.append("近傍探索")
                if "alternative_derived" in labels:
                    label_tags.append("代替推定")
                if label_tags:
                    label_str = f" [{', '.join(label_tags)}]"

            sr = route_data.get("total_success_rate", 0)
            steps = route_data.get("steps", [])
            step_count = route_data.get("step_count", len(steps))

            print()
            print(f"  Route {i}: {r.get('title', 'ルート')}{label_str}")
            print(f"    スコア: {r['score']:.3f}  成功率: {sr:.1%}  ステップ数: {step_count}")
            if ci:
                print(
                    f"    95%信頼区間: [{ci.get('lower', 0):.3f} - {ci.get('upper', 1):.3f}] "
                    f"({ci.get('note', '')})"
                )

            if steps:
                print(f"    遷移:")
                for step in steps:
                    from_h = step.get("from_hex", "?")
                    to_h = step.get("to_hex", "?")
                    action = step.get("action", "?")
                    s_rate = step.get("success_rate", 0)
                    s_count = step.get("count", 0)
                    print(
                        f"      {from_h} --[{action}]--> {to_h}"
                        f"  (成功率: {s_rate:.1%}, 事例: {s_count}件)"
                    )

    # ===== R4: 行動パターン =====
    print()
    print("-" * W)
    print("  R4: 行動パターン")
    print("-" * W)

    action_recs = l3.get("action_recommendations", [])
    if action_recs:
        print(f"  統合行動推奨 (ルート+統計+パターン分析):")
        for rec in action_recs:
            score_bar = "#" * int(rec["score"] * 20)
            print(f"    {rec['action_type']:16s} スコア: {rec['score']:.3f} {score_bar}")
    else:
        print("  行動推奨データなし。")

    # ペア統計
    pair = l3.get("direct_pair_stats", {})
    if pair.get("total_count", 0) > 0:
        print()
        print(f"  直接ペア統計 ({pair.get('pair_key', '')}):")
        print(f"    事例数: {pair['total_count']}件  成功率: {pair.get('success_rate', 0):.1%}")
        for ta in pair.get("top_actions", [])[:3]:
            print(f"    行動: {ta.get('action_type', '?')} ({ta.get('count', 0)}件)")

    # パターン提案
    patterns = l3.get("pattern_suggestions", [])
    if patterns:
        print()
        print(f"  パターン分析:")
        for ps in patterns[:3]:
            print(
                f"    パターン: {ps['pattern_type']} -> {ps['after_state']} "
                f"({ps['total_count']}件)"
            )

    # ===== R5: 注意点 =====
    print()
    print("-" * W)
    print("  R5: 品質ゲート・注意点")
    print("-" * W)

    _positive_keys = {
        "rq3_no_deterministic_words",
        "rq4_has_alternative_route",
        "rq5_confidence_interval_computed",
        "rq7_contradictory_routes_excluded",
    }
    _warn_if_true = {"rq1_reference_only", "rq2_low_success_rate"}

    _gate_names = {
        "rq1_reference_only": "RQ1: 事例数チェック",
        "rq2_low_success_rate": "RQ2: 成功率チェック",
        "rq3_no_deterministic_words": "RQ3: 表現品質",
        "rq4_has_alternative_route": "RQ4: 代替ルート確保",
        "rq5_confidence_interval_computed": "RQ5: 信頼区間計算",
        "rq6_fallback_used": "RQ6: フォールバック使用",
        "rq7_contradictory_routes_excluded": "RQ7: 矛盾ルート除去",
    }

    warnings = []
    for key, val in qg.items():
        name = _gate_names.get(key, key)
        if key in _positive_keys:
            if val:
                print(f"  [OK]   {name}")
            else:
                print(f"  [WARN] {name}")
                warnings.append(name)
        elif key in _warn_if_true:
            if val:
                print(f"  [WARN] {name}")
                warnings.append(name)
            else:
                print(f"  [OK]   {name}")
        else:
            flag = "INFO" if val else "OK"
            print(f"  [{flag}] {name}")

    if warnings:
        print()
        print("  注意事項:")
        if qg.get("rq1_reference_only"):
            print("  * 事例数が少ないため、統計的信頼性が限定的です。参考情報としてご利用ください。")
        if qg.get("rq2_low_success_rate"):
            print("  * 主要ルートの成功率が低めです。代替ルートも合わせて検討してください。")
        if not qg.get("rq4_has_alternative_route"):
            print("  * 代替ルートが見つかりませんでした。慎重にご判断ください。")

    print()
    print("=" * W)
    print()


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="易経逆算エンジン CLI -- なりたい姿から逆算",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # 対話モード
  python3 scripts/backtrace_cli.py

  # クイックモード
  python3 scripts/backtrace_cli.py --quick --goal 1 --current 2
  python3 scripts/backtrace_cli.py --quick --goal 11 --current 12 \\
      --goal-state 安定・平和 --current-state 停滞・閉塞
        """,
    )
    parser.add_argument(
        "--quick", action="store_true", help="非対話モード (--goal, --current必須)"
    )
    parser.add_argument("--goal", type=int, help="目標卦の番号 (1-64)")
    parser.add_argument("--current", type=int, help="現在卦の番号 (1-64)")
    parser.add_argument("--goal-state", type=str, default=None, help="目標状態 (例: 安定・平和)")
    parser.add_argument(
        "--current-state", type=str, default=None, help="現在の状態 (例: 停滞・閉塞)"
    )

    args = parser.parse_args()

    if args.quick:
        # クイックモード
        if not args.goal or not args.current:
            parser.error("--quick モードでは --goal と --current が必須です。")

        goal_hex = args.goal
        current_hex = args.current

        if not 1 <= goal_hex <= 64:
            parser.error(f"--goal は 1-64 の範囲で指定してください: {goal_hex}")
        if not 1 <= current_hex <= 64:
            parser.error(f"--current は 1-64 の範囲で指定してください: {current_hex}")

        # 状態のデフォルト値を設定
        goal_state = args.goal_state if args.goal_state else "安定・平和"
        current_state = args.current_state if args.current_state else "停滞・閉塞"

        print()
        print(f"  目標:  {_hex_label(goal_hex)} / {goal_state}")
        print(f"  現在:  {_hex_label(current_hex)} / {current_state}")
        print(f"  逆算実行中...")

    else:
        # 対話モード
        print()
        print("=" * 50)
        print("  易経逆算エンジン -- なりたい姿から逆算")
        print("=" * 50)

        goal = _prompt_goal()
        goal_hex = goal["hex"]
        goal_state = goal["state"]

        current = _prompt_current()
        current_hex = current["hex"]
        current_state = current["state"]

        print()
        print(f"  目標:  {_hex_label(goal_hex)} / {goal_state}")
        print(f"  現在:  {_hex_label(current_hex)} / {current_state}")
        print()
        print(f"  逆算実行中...")

    # エンジン実行
    try:
        engine = BacktraceEngine()
        result = engine.full_backtrace(
            current_hex=current_hex,
            current_state=current_state,
            goal_hex=goal_hex,
            goal_state=goal_state,
        )
    except Exception as e:
        print(f"\n  エラー: エンジン実行に失敗しました。")
        print(f"  詳細: {e}")
        sys.exit(1)

    # 結果表示
    _display_results(result)


if __name__ == "__main__":
    main()
