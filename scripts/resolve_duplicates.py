#!/usr/bin/env python3
"""
resolve_duplicates.py — 重複83ペアの対処
完全一致26ペア + 高リスクニアミス57ペア（Jaccard >= 0.3）

処理:
- 完全一致: 情報量が多い方を残し、もう一方に duplicate_of フラグ
- ニアミス Jaccard >= 0.5: probable_duplicate_of フラグ
- ニアミス 0.3 <= Jaccard < 0.5: possible_duplicate_of フラグ
  ただし before_state × after_state の組み合わせが異なれば独立事例として除外

事例の削除は一切しない。フラグ付与のみ。
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CANDIDATES = BASE / "analysis" / "quality" / "duplicate_candidates.json"
CASES_FILE = BASE / "data" / "raw" / "cases.jsonl"
REPORT_FILE = BASE / "analysis" / "quality" / "duplicate_resolution.json"


def load_cases(path):
    """cases.jsonl → {transition_id: case_dict}"""
    cases = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            tid = case.get("transition_id", "")
            if tid:
                cases[tid] = case
    return cases


def info_score(case):
    """情報量スコア: summary長 + source_url有無 + sources数"""
    score = 0
    story = case.get("story_summary", "")
    score += len(story) if story else 0
    # source_url (legacy field)
    if case.get("source_url"):
        score += 100
    # sources list
    sources = case.get("sources", [])
    if isinstance(sources, list):
        score += len(sources) * 50
    elif isinstance(sources, str) and sources:
        score += 50
    return score


def resolve_exact_matches(exact_matches, cases):
    """完全一致ペア: 情報量が多い方を残し、もう一方に duplicate_of を付与"""
    flagged = []
    for pair in exact_matches:
        tid_a = pair["transition_id_a"]
        tid_b = pair["transition_id_b"]
        case_a = cases.get(tid_a)
        case_b = cases.get(tid_b)
        if not case_a or not case_b:
            continue

        score_a = info_score(case_a)
        score_b = info_score(case_b)

        if score_a >= score_b:
            keep, remove = tid_a, tid_b
        else:
            keep, remove = tid_b, tid_a

        cases[remove]["duplicate_of"] = keep
        flagged.append({
            "flagged_id": remove,
            "keep_id": keep,
            "flag": "duplicate_of",
            "entity_name": pair.get("entity_name", ""),
            "period": pair.get("period", ""),
            "info_score_keep": max(score_a, score_b),
            "info_score_flagged": min(score_a, score_b),
        })

    return flagged


def resolve_near_misses(near_misses, cases):
    """高リスクニアミス（Jaccard >= 0.3）の処理"""
    probable = []  # >= 0.5
    possible = []  # 0.3 - 0.5
    independent = []  # 独立事例

    high_risk = [nm for nm in near_misses if nm.get("jaccard", 0) >= 0.3]

    for pair in high_risk:
        tid_a = pair["transition_id_a"]
        tid_b = pair["transition_id_b"]
        jaccard = pair["jaccard"]
        case_a = cases.get(tid_a)
        case_b = cases.get(tid_b)
        if not case_a or not case_b:
            continue

        # 独立事例判定: before_state × after_state の組み合わせが異なれば独立
        bs_a = case_a.get("before_state", "")
        as_a = case_a.get("after_state", "")
        bs_b = case_b.get("before_state", "")
        as_b = case_b.get("after_state", "")

        state_combo_a = (bs_a, as_a)
        state_combo_b = (bs_b, as_b)

        if state_combo_a != state_combo_b:
            # 同一entityの異なるフェーズ → 独立事例
            independent.append({
                "id_a": tid_a,
                "id_b": tid_b,
                "jaccard": jaccard,
                "entity_a": pair.get("entity_name_a", ""),
                "entity_b": pair.get("entity_name_b", ""),
                "reason": "different_state_combination",
                "state_a": f"{bs_a} → {as_a}",
                "state_b": f"{bs_b} → {as_b}",
            })
            continue

        # 情報量比較で keep/flag を決定
        score_a = info_score(case_a)
        score_b = info_score(case_b)
        if score_a >= score_b:
            keep, flagged_id = tid_a, tid_b
        else:
            keep, flagged_id = tid_b, tid_a

        if jaccard >= 0.5:
            flag_name = "probable_duplicate_of"
            cases[flagged_id]["probable_duplicate_of"] = keep
            probable.append({
                "flagged_id": flagged_id,
                "keep_id": keep,
                "flag": flag_name,
                "jaccard": jaccard,
                "entity_a": pair.get("entity_name_a", ""),
                "entity_b": pair.get("entity_name_b", ""),
                "info_score_keep": max(score_a, score_b),
                "info_score_flagged": min(score_a, score_b),
            })
        else:
            flag_name = "possible_duplicate_of"
            cases[flagged_id]["possible_duplicate_of"] = keep
            possible.append({
                "flagged_id": flagged_id,
                "keep_id": keep,
                "flag": flag_name,
                "jaccard": jaccard,
                "entity_a": pair.get("entity_name_a", ""),
                "entity_b": pair.get("entity_name_b", ""),
                "info_score_keep": max(score_a, score_b),
                "info_score_flagged": min(score_a, score_b),
            })

    return probable, possible, independent


def save_cases(cases, path):
    """cases辞書をcases.jsonlとして保存（元の順序を保持するため再読み込み）"""
    # 元のファイルの順序を維持
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            tid = case.get("transition_id", "")
            if tid and tid in cases:
                lines.append(json.dumps(cases[tid], ensure_ascii=False))
            else:
                lines.append(json.dumps(case, ensure_ascii=False))

    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main():
    print("=== 重複解消スクリプト ===")

    # 1. バックアップ
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = CASES_FILE.parent / f"cases_backup_{ts}.jsonl"
    shutil.copy2(CASES_FILE, backup_path)
    print(f"バックアップ: {backup_path}")

    # 2. データ読み込み
    cases = load_cases(CASES_FILE)
    print(f"事例数: {len(cases)}")

    with open(CANDIDATES, encoding="utf-8") as f:
        candidates = json.load(f)

    exact_matches = candidates.get("exact_matches", [])
    near_misses = candidates.get("near_misses", [])
    print(f"完全一致ペア: {len(exact_matches)}")
    print(f"ニアミス（全体）: {len(near_misses)}")
    high_risk = [nm for nm in near_misses if nm.get("jaccard", 0) >= 0.3]
    print(f"高リスクニアミス（Jaccard >= 0.3）: {len(high_risk)}")

    # 3. 完全一致の解消
    exact_flagged = resolve_exact_matches(exact_matches, cases)
    print(f"\n--- 完全一致 ---")
    print(f"  duplicate_of フラグ付与: {len(exact_flagged)} 件")

    # 4. ニアミスの解消
    probable, possible, independent = resolve_near_misses(near_misses, cases)
    print(f"\n--- 高リスクニアミス ---")
    print(f"  probable_duplicate_of (Jaccard >= 0.5): {len(probable)} 件")
    print(f"  possible_duplicate_of (0.3 <= Jaccard < 0.5): {len(possible)} 件")
    print(f"  独立事例（フラグなし）: {len(independent)} 件")

    # 5. cases.jsonl 更新
    save_cases(cases, CASES_FILE)
    print(f"\ncases.jsonl 更新完了")

    # 6. レポート保存
    report = {
        "timestamp": datetime.now().isoformat(),
        "backup": str(backup_path),
        "total_cases": len(cases),
        "exact_match_pairs": len(exact_matches),
        "high_risk_near_miss_pairs": len(high_risk),
        "results": {
            "duplicate_of": len(exact_flagged),
            "probable_duplicate_of": len(probable),
            "possible_duplicate_of": len(possible),
            "independent": len(independent),
            "total_flagged": len(exact_flagged) + len(probable) + len(possible),
        },
        "details": {
            "exact_flagged": exact_flagged,
            "probable_flagged": probable,
            "possible_flagged": possible,
            "independent_pairs": independent,
        },
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"レポート保存: {REPORT_FILE}")

    # 7. サマリー表示
    print(f"\n=== 結果サマリー ===")
    print(f"  duplicate_of:          {len(exact_flagged)} 件")
    print(f"  probable_duplicate_of: {len(probable)} 件")
    print(f"  possible_duplicate_of: {len(possible)} 件")
    print(f"  独立事例:              {len(independent)} 件")
    print(f"  フラグ付与合計:        {len(exact_flagged) + len(probable) + len(possible)} 件")


if __name__ == "__main__":
    main()
