#!/usr/bin/env python3
"""
逆引きインデックス構築スクリプト（scale別分離対応 v2）

cases.jsonl + yao_transitions.json から6つの逆引きインデックスを生成する。
各インデックスをscale別（company/individual/family/country/other）に分離生成し、
全カテゴリ合算版（all）も維持する。

出力先: data/reverse/
出力ファイル例:
  rev_after_hex.json          (後方互換: allと同一内容)
  rev_after_hex_all.json      (全カテゴリ合算)
  rev_after_hex_company.json  (企業のみ)
  rev_after_hex_individual.json (個人のみ)
  rev_after_hex_family.json   (家族のみ)
  rev_after_hex_country.json  (国家のみ)
  rev_after_hex_other.json    (その他のみ)

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

# scale定義
ALL_SCALES = ["company", "individual", "family", "country", "other"]


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


def split_cases_by_scale(cases: list) -> dict:
    """
    casesリストをscale別に分類する。

    Returns:
        {"all": [...], "company": [...], "individual": [...],
         "family": [...], "country": [...], "other": [...]}
    """
    by_scale = {"all": cases}
    for scale in ALL_SCALES:
        by_scale[scale] = []

    for case in cases:
        scale = case.get("scale", "other")
        if scale not in ALL_SCALES:
            scale = "other"
        by_scale[scale].append(case)

    return by_scale


# ---------------------------------------------------------------------------
# インデックス構築関数
# ---------------------------------------------------------------------------

def build_rev_yao(yao_transitions: dict) -> dict:
    """
    Index 1: rev_yao.json
    target_hexagram_id (str) -> [{source_hex_id, source_hex_name, yao_pos}]

    yao_transitions.jsonの全ソース卦・全爻位置を走査し、
    変化先（next_hexagram_id）をキーとして逆引きマップを作る。

    注: このインデックスは事例データに依存しないため、scale別分離は
    同一内容のコピーとなる。
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

    # ソート: source_hex_id -> yao_pos の順で安定化
    for key in index:
        index[key].sort(key=lambda x: (x["source_hex_id"], x["yao_pos"]))

    return dict(index)


def build_rev_after_hex(cases: list, max_per_key: int = 50) -> dict:
    """
    Index 2: rev_after_hex.json
    after_hex (trigram名) -> [{transition_id, target_name, before_hex, action_type,
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
    after_state -> {total_count, before_state_distribution: [{state, count, pct}],
                   top_actions: [{action_type, count, pct}]}

    cases.jsonlを集計して構築。
    """
    # after_state -> before_state のカウント
    before_counter: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # after_state -> action_type のカウント
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
    "outcome|action_type" -> {total_count, before_state_distribution: [{state, count, pct}]}

    cases.jsonlを集計して構築。
    """
    # (outcome, action_type) -> before_state のカウント
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
    "pattern_type|after_state" -> {total_count, entries: [{before_hex, action_type, count}]}

    1キーあたり上位 top_n エントリのみ保持。
    """
    # (pattern_type, after_state) -> (before_hex, action_type) のカウント
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
    "before_hex|after_hex" -> {total_count, success_count, success_rate,
                               outcomes: {outcome: count},
                               top_actions: [{action_type, count}],
                               avg_duration_hint}

    success_rate = (outcome=="Success"件数) / total_count
    avg_duration_hint はperiodフィールドから年数を推定（推定不可の場合はnull）。
    """
    # (before_hex, after_hex) -> 集計用コンテナ
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

        # period から継続年数を推定（例: "2015-2020" -> 5年）
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


def build_quality_stats(cases_by_scale: dict) -> dict:
    """
    品質統計を構築する。

    scale別に以下を計算:
      - total: 事例数
      - sa_rate: credibility_rank が S or A の割合
      - verified_rate: outcome_status が verified_correct の割合
      - high_confidence_rate: verification_confidence が high の割合
      - trust_verified_rate: trust_level が verified の割合
      - quality_weight: 上記4指標の加重平均 (0.5〜1.0 にクリップ)

    quality_weight の計算式:
      raw = 0.35 * sa_rate + 0.30 * verified_rate + 0.20 * high_confidence_rate + 0.15 * trust_verified_rate
      quality_weight = max(0.5, min(1.0, 0.5 + raw * 0.5))
      → rawが1.0の時 quality_weight=1.0, rawが0.0の時 quality_weight=0.5

    Returns:
        {"company": {"total": N, "sa_rate": 0.XX, ...}, ...}
    """
    stats = {}
    for scale in ["all"] + ALL_SCALES:
        cases = cases_by_scale[scale]
        total = len(cases)
        if total == 0:
            stats[scale] = {
                "total": 0,
                "sa_rate": 0.0,
                "verified_rate": 0.0,
                "high_confidence_rate": 0.0,
                "trust_verified_rate": 0.0,
                "quality_weight": 0.7,  # デフォルト
            }
            continue

        sa_count = sum(
            1 for c in cases
            if c.get("credibility_rank", "") in ("S", "A")
        )
        verified_count = sum(
            1 for c in cases
            if c.get("outcome_status", "") == "verified_correct"
        )
        high_conf_count = sum(
            1 for c in cases
            if c.get("verification_confidence", "") == "high"
        )
        trust_ver_count = sum(
            1 for c in cases
            if c.get("trust_level", "") == "verified"
        )

        sa_rate = round(sa_count / total, 4)
        verified_rate = round(verified_count / total, 4)
        high_confidence_rate = round(high_conf_count / total, 4)
        trust_verified_rate = round(trust_ver_count / total, 4)

        # 加重平均 → 0.5〜1.0にマップ
        raw = (
            0.35 * sa_rate
            + 0.30 * verified_rate
            + 0.20 * high_confidence_rate
            + 0.15 * trust_verified_rate
        )
        quality_weight = round(max(0.5, min(1.0, 0.5 + raw * 0.5)), 4)

        stats[scale] = {
            "total": total,
            "sa_rate": sa_rate,
            "verified_rate": verified_rate,
            "high_confidence_rate": high_confidence_rate,
            "trust_verified_rate": trust_verified_rate,
            "quality_weight": quality_weight,
        }

    return stats


def _estimate_duration(period: str) -> float | None:
    """
    period 文字列から継続年数（float）を推定する。

    対応形式:
      "2015-2020"   -> 5.0
      "2015-2021"   -> 6.0
      "2020"        -> None（単年は継続期間不明）
      "令和前期"     -> None（和暦テキストは推定不可）
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
# scale別インデックス構築・保存ヘルパー
# ---------------------------------------------------------------------------

def _build_and_save_for_all_scales(
    index_name: str,
    build_func,
    cases_by_scale: dict,
    label: str,
    summary_func=None,
) -> dict:
    """
    指定のインデックス構築関数を全scale + all で実行し、保存する。

    Args:
        index_name: ベースファイル名（拡張子なし） 例: "rev_after_hex"
        build_func: cases -> dict を返す構築関数
        cases_by_scale: split_cases_by_scaleの戻り値
        label: ログ表示用ラベル
        summary_func: インデックスのサマリー文字列を返す関数 (dict -> str)

    Returns:
        {"all": idx_all, "company": idx_company, ...} の辞書
    """
    results = {}
    scale_keys = ["all"] + ALL_SCALES

    for scale in scale_keys:
        cases_subset = cases_by_scale[scale]
        idx = build_func(cases_subset)
        results[scale] = idx

        # ファイル名の決定
        if scale == "all":
            # all版 + 後方互換の無印版
            path_all = save_index(idx, f"{index_name}_all.json")
            path_compat = save_index(idx, f"{index_name}.json")
            summary = summary_func(idx) if summary_func else f"{len(idx)} keys"
            print(f"    {scale:>12}: {summary} -> {os.path.basename(path_all)} + {os.path.basename(path_compat)}")
        else:
            path = save_index(idx, f"{index_name}_{scale}.json")
            summary = summary_func(idx) if summary_func else f"{len(idx)} keys"
            print(f"    {scale:>12}: {summary} -> {os.path.basename(path)}")

    return results


# ---------------------------------------------------------------------------
# バリデーション
# ---------------------------------------------------------------------------

def validate_indices():
    """
    生成された逆引きインデックスを検証し、カバレッジとscale別統計を報告する。

    チェック内容:
      - rev_yao.json: 全64卦がターゲットとして存在するか
      - rev_after_hex.json: 全8卦がキーとして存在するか
      - scale別ファイルの存在確認と件数整合性チェック
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

    # --- rev_yao scale別ファイル確認 ---
    for suffix in ["all"] + ALL_SCALES:
        fname = f"rev_yao_{suffix}.json"
        fpath = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  FAIL: {fname} が存在しません")
            all_pass = False

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

    # --- scale別ファイルの存在確認と統計 ---
    print("\n=== scale別ファイル検証 ===")
    index_names = [
        "rev_yao", "rev_after_hex", "rev_after_state",
        "rev_outcome_action", "rev_pattern_after", "rev_hex_pair_stats",
    ]
    for idx_name in index_names:
        print(f"\n  {idx_name}:")
        for suffix in ["all"] + ALL_SCALES:
            fname = f"{idx_name}_{suffix}.json"
            fpath = os.path.join(OUTPUT_DIR, fname)
            if not os.path.exists(fpath):
                print(f"    {suffix:>12}: FAIL - ファイルなし")
                all_pass = False
            else:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                keys_count = len(data)
                # total_countの合計を計算（辞書型の場合）
                total_count = sum(
                    v.get("total_count", 0) if isinstance(v, dict) else 0
                    for v in data.values()
                )
                # リスト型（rev_yao, rev_after_hex）の場合はエントリ合計
                entry_count = sum(
                    len(v) if isinstance(v, list) else 0
                    for v in data.values()
                )
                if entry_count > 0:
                    print(f"    {suffix:>12}: PASS - {keys_count} keys, {entry_count} entries")
                elif total_count > 0:
                    print(f"    {suffix:>12}: PASS - {keys_count} keys, total_count={total_count:,}")
                else:
                    print(f"    {suffix:>12}: PASS - {keys_count} keys")

    # --- scale別 allとの整合性チェック ---
    print("\n=== scale合算整合性チェック ===")
    for idx_name in ["rev_after_state", "rev_outcome_action", "rev_hex_pair_stats"]:
        all_path = os.path.join(OUTPUT_DIR, f"{idx_name}_all.json")
        if not os.path.exists(all_path):
            continue
        with open(all_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        all_total = sum(
            v.get("total_count", 0) if isinstance(v, dict) else 0
            for v in all_data.values()
        )

        scale_total = 0
        for scale in ALL_SCALES:
            scale_path = os.path.join(OUTPUT_DIR, f"{idx_name}_{scale}.json")
            if not os.path.exists(scale_path):
                continue
            with open(scale_path, "r", encoding="utf-8") as f:
                scale_data = json.load(f)
            scale_total += sum(
                v.get("total_count", 0) if isinstance(v, dict) else 0
                for v in scale_data.values()
            )

        if all_total == scale_total:
            print(f"  {idx_name}: PASS - all({all_total:,}) == scale合計({scale_total:,})")
        else:
            print(f"  {idx_name}: FAIL - all({all_total:,}) != scale合計({scale_total:,})")
            all_pass = False

    # --- scale別の事例件数サマリー ---
    print("\n=== scale別事例件数 ===")
    cases = load_cases(CASES_PATH)
    cases_by_scale = split_cases_by_scale(cases)
    for scale in ["all"] + ALL_SCALES:
        count = len(cases_by_scale[scale])
        pct = round(count / len(cases) * 100, 1) if cases else 0
        print(f"  {scale:>12}: {count:>6,} 件 ({pct}%)")

    print("\n" + ("=== バリデーション完了: 全PASS ===" if all_pass else "=== バリデーション完了: 警告あり ==="))
    return all_pass


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def build_all():
    """全6インデックスをscale別に構築して保存する。"""
    print("=== 逆引きインデックス構築開始（scale別分離対応 v2） ===")
    print(f"  cases.jsonl:          {CASES_PATH}")
    print(f"  yao_transitions.json: {YAO_TRANSITIONS_PATH}")
    print(f"  出力先:               {OUTPUT_DIR}")
    print(f"  scale:                {', '.join(ALL_SCALES)} + all")

    # ---- データ読み込み ----
    print("\n[読み込み]")
    cases = load_cases(CASES_PATH)
    print(f"  cases.jsonl: {len(cases):,} 件")

    yao_transitions = load_yao_transitions(YAO_TRANSITIONS_PATH)
    print(f"  yao_transitions: {len(yao_transitions)} 卦")

    # ---- scale別に事例を分類 ----
    cases_by_scale = split_cases_by_scale(cases)
    print("\n[scale別事例数]")
    for scale in ["all"] + ALL_SCALES:
        count = len(cases_by_scale[scale])
        pct = round(count / len(cases) * 100, 1) if cases else 0
        print(f"  {scale:>12}: {count:>6,} 件 ({pct}%)")

    # ---- インデックス構築 & 保存 ----
    print("\n[構築・保存]")

    # -- [1/6] rev_yao: yao_transitionsから構築（事例非依存） --
    print("\n  [1/6] rev_yao (事例非依存: 全scaleに同一内容をコピー)")
    idx_yao = build_rev_yao(yao_transitions)
    summary_yao = f"{len(idx_yao)} target hexagrams, {sum(len(v) for v in idx_yao.values())} entries"
    # all + 後方互換
    save_index(idx_yao, "rev_yao_all.json")
    save_index(idx_yao, "rev_yao.json")
    print(f"    {'all':>12}: {summary_yao} -> rev_yao_all.json + rev_yao.json")
    # scale別（同一内容）
    for scale in ALL_SCALES:
        save_index(idx_yao, f"rev_yao_{scale}.json")
        print(f"    {scale:>12}: {summary_yao} -> rev_yao_{scale}.json")

    # -- [2/6] rev_after_hex --
    print("\n  [2/6] rev_after_hex")
    _build_and_save_for_all_scales(
        "rev_after_hex",
        build_rev_after_hex,
        cases_by_scale,
        "[2/6]",
        summary_func=lambda idx: f"{len(idx)} trigrams, {sum(len(v) for v in idx.values())} entries",
    )

    # -- [3/6] rev_after_state --
    print("\n  [3/6] rev_after_state")
    _build_and_save_for_all_scales(
        "rev_after_state",
        build_rev_after_state,
        cases_by_scale,
        "[3/6]",
        summary_func=lambda idx: f"{len(idx)} states, total_count={sum(v.get('total_count', 0) for v in idx.values()):,}",
    )

    # -- [4/6] rev_outcome_action --
    print("\n  [4/6] rev_outcome_action")
    _build_and_save_for_all_scales(
        "rev_outcome_action",
        build_rev_outcome_action,
        cases_by_scale,
        "[4/6]",
        summary_func=lambda idx: f"{len(idx)} keys, total_count={sum(v.get('total_count', 0) for v in idx.values()):,}",
    )

    # -- [5/6] rev_pattern_after --
    print("\n  [5/6] rev_pattern_after")
    _build_and_save_for_all_scales(
        "rev_pattern_after",
        build_rev_pattern_after,
        cases_by_scale,
        "[5/6]",
        summary_func=lambda idx: f"{len(idx)} keys, total_count={sum(v.get('total_count', 0) for v in idx.values()):,}",
    )

    # -- [6/6] rev_hex_pair_stats --
    print("\n  [6/6] rev_hex_pair_stats")
    _build_and_save_for_all_scales(
        "rev_hex_pair_stats",
        build_rev_hex_pair_stats,
        cases_by_scale,
        "[6/6]",
        summary_func=lambda idx: f"{len(idx)} pairs, total_count={sum(v.get('total_count', 0) for v in idx.values()):,}",
    )

    # -- [7/7] quality_stats --
    print("\n  [7/7] quality_stats (品質統計)")
    quality_stats = build_quality_stats(cases_by_scale)
    path_qs = save_index(quality_stats, "quality_stats.json")
    for scale in ["all"] + ALL_SCALES:
        qs = quality_stats[scale]
        print(
            f"    {scale:>12}: total={qs['total']:>6,}, "
            f"sa_rate={qs['sa_rate']:.3f}, "
            f"verified_rate={qs['verified_rate']:.3f}, "
            f"quality_weight={qs['quality_weight']:.4f}"
        )
    print(f"    -> {os.path.basename(path_qs)}")

    print("\n=== 構築完了 ===")


def main():
    parser = argparse.ArgumentParser(
        description="cases.jsonl + yao_transitions.json から6つの逆引きインデックスをscale別に生成する。"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="構築後にインデックスのカバレッジとscale別統計を検証する",
    )
    args = parser.parse_args()

    build_all()

    if args.validate:
        success = validate_indices()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
