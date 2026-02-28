#!/usr/bin/env python3
"""
逆引きインデックス構築スクリプト

cases.jsonl + yao_transitions.json から6つの逆引きインデックスを生成する。
出力先: data/reverse/

Usage:
    python3 scripts/build_reverse_indices.py
    python3 scripts/build_reverse_indices.py --validate
"""

import argparse
import json
import os
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# パス定義
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CASES_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "cases.jsonl")
YAO_TRANSITIONS_PATH = os.path.join(PROJECT_ROOT, "data", "mappings", "yao_transitions.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "reverse")

# すべての8卦（trigram名）
ALL_TRIGRAMS = {"乾", "坤", "震", "巽", "坎", "離", "艮", "兌"}


# ---------------------------------------------------------------------------
# ローダー
# ---------------------------------------------------------------------------

def load_cases(path: str) -> list:
    """cases.jsonlを1行ずつ読み込んでリストとして返す。"""
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [警告] {path}:{lineno} — JSON解析エラー: {e}", file=sys.stderr)
    return cases


def load_yao_transitions(path: str) -> dict:
    """yao_transitions.jsonを読み込んで返す。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# インデックス構築関数
# ---------------------------------------------------------------------------

def build_rev_yao(yao_transitions: dict) -> dict:
    """
    Index 1: rev_yao.json
    target_hexagram_id (str) → [{source_hex_id, source_hex_name, yao_pos}]

    yao_transitions.jsonの全ソース卦・全爻位置を走査し、
    変化先（next_hexagram_id）をキーとして逆引きマップを作る。
    """
    index = defaultdict(list)

    for source_id, hex_data in yao_transitions.items():
        source_name = hex_data.get("name", "")
        transitions = hex_data.get("transitions", {})
        for yao_pos_str, trans_info in transitions.items():
            target_id = trans_info.get("next_hexagram_id")
            if target_id is None:
                continue
            target_key = str(target_id)
            index[target_key].append({
                "source_hex_id": int(source_id),
                "source_hex_name": source_name,
                "yao_pos": int(yao_pos_str),
            })

    # ソート: source_hex_id → yao_pos の順で安定化
    for key in index:
        index[key].sort(key=lambda x: (x["source_hex_id"], x["yao_pos"]))

    return dict(index)


def build_rev_after_hex(cases: list, max_per_key: int = 50) -> dict:
    """
    Index 2: rev_after_hex.json
    after_hex (trigram名) → [{transition_id, target_name, before_hex, action_type,
                               before_state, after_state, period}]

    1キーあたり最大 max_per_key 件に制限（ファイルサイズ管理）。
    """
    index = defaultdict(list)

    for case in cases:
        after_hex = case.get("after_hex")
        if not after_hex:
            continue
        entry = {
            "transition_id": case.get("transition_id", ""),
            "target_name":   case.get("target_name", ""),
            "before_hex":    case.get("before_hex", ""),
            "action_type":   case.get("action_type", ""),
            "before_state":  case.get("before_state", ""),
            "after_state":   case.get("after_state", ""),
            "period":        case.get("period", ""),
        }
        index[after_hex].append(entry)

    # 各キーを max_per_key 件に切り詰め
    return {k: v[:max_per_key] for k, v in index.items()}


def build_rev_after_state(cases: list) -> dict:
    """
    Index 3: rev_after_state.json
    after_state → {total_count, before_state_distribution: [{state, count, pct}],
                   top_actions: [{action_type, count, pct}]}

    cases.jsonlを集計して構築。
    """
    # after_state → before_state のカウント
    before_counter: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # after_state → action_type のカウント
    action_counter: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for case in cases:
        after_state = case.get("after_state")
        if not after_state:
            continue
        before_state = case.get("before_state", "不明")
        action_type = case.get("action_type", "不明")
        before_counter[after_state][before_state] += 1
        action_counter[after_state][action_type] += 1

    index = {}
    for after_state, b_counts in before_counter.items():
        total = sum(b_counts.values())
        before_dist = sorted(
            [
                {"state": state, "count": cnt, "pct": round(cnt / total * 100, 1)}
                for state, cnt in b_counts.items()
            ],
            key=lambda x: -x["count"],
        )

        a_counts = action_counter[after_state]
        a_total = sum(a_counts.values())
        top_actions = sorted(
            [
                {"action_type": at, "count": cnt, "pct": round(cnt / a_total * 100, 1)}
                for at, cnt in a_counts.items()
            ],
            key=lambda x: -x["count"],
        )

        index[after_state] = {
            "total_count": total,
            "before_state_distribution": before_dist,
            "top_actions": top_actions,
        }

    return index


def build_rev_outcome_action(cases: list) -> dict:
    """
    Index 4: rev_outcome_action.json
    "outcome|action_type" → {total_count, before_state_distribution: [{state, count, pct}]}

    cases.jsonlを集計して構築。
    """
    # (outcome, action_type) → before_state のカウント
    counter: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for case in cases:
        outcome = case.get("outcome")
        action_type = case.get("action_type")
        if not outcome or not action_type:
            continue
        before_state = case.get("before_state", "不明")
        counter[(outcome, action_type)][before_state] += 1

    index = {}
    for (outcome, action_type), b_counts in counter.items():
        total = sum(b_counts.values())
        before_dist = sorted(
            [
                {"state": state, "count": cnt, "pct": round(cnt / total * 100, 1)}
                for state, cnt in b_counts.items()
            ],
            key=lambda x: -x["count"],
        )
        key = f"{outcome}|{action_type}"
        index[key] = {
            "total_count": total,
            "before_state_distribution": before_dist,
        }

    return index


def build_rev_pattern_after(cases: list, top_n: int = 20) -> dict:
    """
    Index 5: rev_pattern_after.json
    "pattern_type|after_state" → {total_count, entries: [{before_hex, action_type, count}]}

    1キーあたり上位 top_n エントリのみ保持。
    """
    # (pattern_type, after_state) → (before_hex, action_type) のカウント
    counter: dict[tuple, dict[tuple, int]] = defaultdict(lambda: defaultdict(int))

    for case in cases:
        pattern_type = case.get("pattern_type")
        after_state = case.get("after_state")
        if not pattern_type or not after_state:
            continue
        before_hex = case.get("before_hex", "不明")
        action_type = case.get("action_type", "不明")
        counter[(pattern_type, after_state)][(before_hex, action_type)] += 1

    index = {}
    for (pattern_type, after_state), pair_counts in counter.items():
        total = sum(pair_counts.values())
        entries = sorted(
            [
                {"before_hex": bh, "action_type": at, "count": cnt}
                for (bh, at), cnt in pair_counts.items()
            ],
            key=lambda x: -x["count"],
        )[:top_n]

        key = f"{pattern_type}|{after_state}"
        index[key] = {
            "total_count": total,
            "entries": entries,
        }

    return index


def build_rev_hex_pair_stats(cases: list) -> dict:
    """
    Index 6: rev_hex_pair_stats.json
    "before_hex|after_hex" → {total_count, success_count, success_rate,
                               outcomes: {outcome: count},
                               top_actions: [{action_type, count}],
                               avg_duration_hint}

    success_rate = (outcome=="Success"件数) / total_count
    avg_duration_hint はperiodフィールドから年数を推定（推定不可の場合はnull）。
    """
    # (before_hex, after_hex) → 集計用コンテナ
    accum: dict[tuple, dict] = defaultdict(lambda: {
        "outcomes": defaultdict(int),
        "actions": defaultdict(int),
        "durations": [],
    })

    for case in cases:
        before_hex = case.get("before_hex")
        after_hex = case.get("after_hex")
        if not before_hex or not after_hex:
            continue

        key = (before_hex, after_hex)
        outcome = case.get("outcome", "不明")
        action_type = case.get("action_type", "不明")

        accum[key]["outcomes"][outcome] += 1
        accum[key]["actions"][action_type] += 1

        # period から継続年数を推定（例: "2015-2020" → 5年）
        duration = _estimate_duration(case.get("period", ""))
        if duration is not None:
            accum[key]["durations"].append(duration)

    index = {}
    for (before_hex, after_hex), data in accum.items():
        outcomes_dict = dict(data["outcomes"])
        total = sum(outcomes_dict.values())
        success_count = outcomes_dict.get("Success", 0)
        success_rate = round(success_count / total, 4) if total > 0 else 0.0

        top_actions = sorted(
            [{"action_type": at, "count": cnt} for at, cnt in data["actions"].items()],
            key=lambda x: -x["count"],
        )

        durations = data["durations"]
        avg_duration_hint = round(sum(durations) / len(durations), 1) if durations else None

        pair_key = f"{before_hex}|{after_hex}"
        index[pair_key] = {
            "total_count":       total,
            "success_count":     success_count,
            "success_rate":      success_rate,
            "outcomes":          outcomes_dict,
            "top_actions":       top_actions,
            "avg_duration_hint": avg_duration_hint,
        }

    return index


def _estimate_duration(period: str) -> float | None:
    """
    period 文字列から継続年数（float）を推定する。

    対応形式:
      "2015-2020"   → 5.0
      "2015-2021"   → 6.0
      "2020"        → None（単年は継続期間不明）
      "令和前期"     → None（和暦テキストは推定不可）
    """
    if not period:
        return None
    parts = period.split("-")
    if len(parts) == 2:
        try:
            start = int(parts[0].strip())
            end = int(parts[1].strip())
            if 1800 <= start <= 2100 and 1800 <= end <= 2100 and end >= start:
                return float(end - start)
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# 保存ヘルパー
# ---------------------------------------------------------------------------

def save_index(data: dict, filename: str) -> str:
    """インデックスをJSON形式でOUTPUT_DIRに保存し、出力パスを返す。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# バリデーション
# ---------------------------------------------------------------------------

def validate_indices():
    """
    生成された6インデックスを検証し、カバレッジを報告する。

    チェック内容:
      - rev_yao.json: 全64卦がターゲットとして存在するか
      - rev_after_hex.json: 全8卦がキーとして存在するか
    """
    print("\n=== バリデーション ===")
    all_pass = True

    # --- rev_yao.json ---
    rev_yao_path = os.path.join(OUTPUT_DIR, "rev_yao.json")
    print(f"\n[1] rev_yao.json: {rev_yao_path}")
    if not os.path.exists(rev_yao_path):
        print("  FAIL: ファイルが存在しません")
        all_pass = False
    else:
        with open(rev_yao_path, "r", encoding="utf-8") as f:
            rev_yao = json.load(f)
        total_keys = len(rev_yao)
        total_entries = sum(len(v) for v in rev_yao.values())
        all_hex_ids = set(str(i) for i in range(1, 65))
        missing_hex = sorted(all_hex_ids - set(rev_yao.keys()), key=int)
        print(f"  合計キー数: {total_keys} / 64")
        print(f"  合計エントリ数: {total_entries}")
        if missing_hex:
            print(f"  WARN: 以下の卦がターゲットに含まれていません: {missing_hex}")
            all_pass = False
        else:
            print("  PASS: 全64卦がターゲットとして存在します")

    # --- rev_after_hex.json ---
    rev_after_hex_path = os.path.join(OUTPUT_DIR, "rev_after_hex.json")
    print(f"\n[2] rev_after_hex.json: {rev_after_hex_path}")
    if not os.path.exists(rev_after_hex_path):
        print("  FAIL: ファイルが存在しません")
        all_pass = False
    else:
        with open(rev_after_hex_path, "r", encoding="utf-8") as f:
            rev_after_hex = json.load(f)
        total_keys = len(rev_after_hex)
        total_entries = sum(len(v) for v in rev_after_hex.values())
        missing_trigrams = sorted(ALL_TRIGRAMS - set(rev_after_hex.keys()))
        print(f"  合計キー数: {total_keys} / 8")
        print(f"  合計エントリ数: {total_entries}")
        if missing_trigrams:
            print(f"  WARN: 以下の卦が欠けています: {missing_trigrams}")
            all_pass = False
        else:
            print("  PASS: 全8卦がキーとして存在します")

    # --- 残り4インデックスのサマリー ---
    for fname, label in [
        ("rev_after_state.json",    "[3] rev_after_state"),
        ("rev_outcome_action.json", "[4] rev_outcome_action"),
        ("rev_pattern_after.json",  "[5] rev_pattern_after"),
        ("rev_hex_pair_stats.json", "[6] rev_hex_pair_stats"),
    ]:
        path = os.path.join(OUTPUT_DIR, fname)
        print(f"\n{label}.json: {path}")
        if not os.path.exists(path):
            print("  FAIL: ファイルが存在しません")
            all_pass = False
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            total_count = sum(
                v.get("total_count", 0) if isinstance(v, dict) else 0
                for v in data.values()
            )
            print(f"  合計キー数: {len(data)}")
            print(f"  合計件数 (total_count合計): {total_count}")
            print("  PASS: ファイル存在・JSON読み込み成功")

    print("\n" + ("=== バリデーション完了: 全PASS ===" if all_pass else "=== バリデーション完了: 警告あり ==="))
    return all_pass


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def build_all():
    """全6インデックスを構築して保存する。"""
    print("=== 逆引きインデックス構築開始 ===")
    print(f"  cases.jsonl:        {CASES_PATH}")
    print(f"  yao_transitions.json: {YAO_TRANSITIONS_PATH}")
    print(f"  出力先:               {OUTPUT_DIR}")

    # ---- データ読み込み ----
    print("\n[読み込み]")
    cases = load_cases(CASES_PATH)
    print(f"  cases.jsonl: {len(cases):,} 件")

    yao_transitions = load_yao_transitions(YAO_TRANSITIONS_PATH)
    print(f"  yao_transitions: {len(yao_transitions)} 卦")

    # ---- インデックス構築 & 保存 ----
    print("\n[構築・保存]")

    print("  [1/6] rev_yao.json ...")
    idx1 = build_rev_yao(yao_transitions)
    path1 = save_index(idx1, "rev_yao.json")
    print(f"       → {len(idx1)} ターゲット卦, 保存: {path1}")

    print("  [2/6] rev_after_hex.json ...")
    idx2 = build_rev_after_hex(cases)
    path2 = save_index(idx2, "rev_after_hex.json")
    total2 = sum(len(v) for v in idx2.values())
    print(f"       → {len(idx2)} 卦, {total2} エントリ, 保存: {path2}")

    print("  [3/6] rev_after_state.json ...")
    idx3 = build_rev_after_state(cases)
    path3 = save_index(idx3, "rev_after_state.json")
    print(f"       → {len(idx3)} after_state, 保存: {path3}")

    print("  [4/6] rev_outcome_action.json ...")
    idx4 = build_rev_outcome_action(cases)
    path4 = save_index(idx4, "rev_outcome_action.json")
    print(f"       → {len(idx4)} キー, 保存: {path4}")

    print("  [5/6] rev_pattern_after.json ...")
    idx5 = build_rev_pattern_after(cases)
    path5 = save_index(idx5, "rev_pattern_after.json")
    print(f"       → {len(idx5)} キー, 保存: {path5}")

    print("  [6/6] rev_hex_pair_stats.json ...")
    idx6 = build_rev_hex_pair_stats(cases)
    path6 = save_index(idx6, "rev_hex_pair_stats.json")
    print(f"       → {len(idx6)} ペア, 保存: {path6}")

    print("\n=== 構築完了 ===")


def main():
    parser = argparse.ArgumentParser(
        description="cases.jsonl + yao_transitions.json から6つの逆引きインデックスを生成する。"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="構築後にインデックスのカバレッジを検証する",
    )
    args = parser.parse_args()

    build_all()

    if args.validate:
        success = validate_indices()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
