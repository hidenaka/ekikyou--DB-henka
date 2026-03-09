#!/usr/bin/env python3
"""
Gold Set Case Selection Script (Step 1-A)
==========================================
Selects 800 cases from cases.jsonl for LLM-based reannotation.

Selection strategy:
  1. Filter to cases with story_summary >= 50 characters
  2. Ensure state coverage (before_state and after_state)
  3. Prioritize text containing 離/兌 trigram keywords (underrepresented)
  4. Ensure scale diversity (company/individual/country/family)
  5. Fill remaining with random diverse sampling across domains

Output: analysis/gold_set/selected_800_cases.json
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

# Reproducibility
random.seed(42)

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent.parent
CASES_PATH = BASE_DIR / "data" / "raw" / "cases.jsonl"
OUTPUT_DIR = BASE_DIR / "analysis" / "gold_set"
OUTPUT_PATH = OUTPUT_DIR / "selected_800_cases.json"

TARGET_COUNT = 800

# ── Trigram keyword lists ──
RI_KEYWORDS = [
    "ビジョン", "可視化", "情熱", "明確", "公開", "透明",
    "メディア", "注目", "発見", "革新", "見える化", "発信",
    "ブランド", "露出", "啓発", "照らす", "光", "炎上",
    "映像", "放送", "公表", "開示", "表明", "表現",
]

DA_KEYWORDS = [
    "対話", "交流", "喜び", "協力", "共有", "コミュニケーション",
    "パートナー", "提携", "楽しみ", "合意", "和解", "連携",
    "共創", "共感", "笑顔", "ファン", "コラボ", "親しみ",
    "顧客満足", "エンゲージメント", "参加", "祝", "歓迎",
]


def load_cases():
    """Load all cases from JSONL."""
    cases = []
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def has_keywords(text, keywords):
    """Check if text contains any of the given keywords."""
    if not text:
        return False
    return any(kw in text for kw in keywords)


def case_text(case):
    """Get combined text fields for keyword matching."""
    parts = []
    for field in ["story_summary", "logic_memo", "pre_outcome_text"]:
        val = case.get(field) or ""
        if val:
            parts.append(val)
    return " ".join(parts)


def extract_meta(case):
    """Extract metadata dict for output."""
    ss = case.get("story_summary") or ""
    return {
        "transition_id": case.get("transition_id", ""),
        "story_summary": ss,
        "before_state": case.get("before_state", ""),
        "after_state": case.get("after_state", ""),
        "main_domain": case.get("main_domain", ""),
        "scale": case.get("scale", ""),
        "story_length": len(ss),
    }


def select_cases():
    """Main selection algorithm."""
    print("=" * 60)
    print("Gold Set Case Selection (Step 1-A)")
    print("=" * 60)

    # ── Load ──
    all_cases = load_cases()
    print(f"\nTotal cases loaded: {len(all_cases)}")

    # ── Step 1: Filter by story_summary length ──
    candidates = [
        c for c in all_cases
        if len(c.get("story_summary") or "") >= 50
    ]
    print(f"Candidates (story_summary >= 50 chars): {len(candidates)}")

    # Index by transition_id for deduplication
    cand_by_id = {c["transition_id"]: c for c in candidates}

    selected_ids = set()

    def add_cases(case_list, label, max_count=None):
        """Add cases to selected set, return count added."""
        added = 0
        for c in case_list:
            tid = c["transition_id"]
            if tid not in selected_ids and tid in cand_by_id:
                selected_ids.add(tid)
                added += 1
                if max_count and added >= max_count:
                    break
        return added

    # ── Step 2: Ensure before_state coverage (20 per category) ──
    before_state_groups = defaultdict(list)
    for c in candidates:
        bs = c.get("before_state", "")
        if bs:
            before_state_groups[bs].append(c)

    print(f"\n--- Before State Coverage ---")
    for state, group in sorted(before_state_groups.items()):
        random.shuffle(group)
        n = add_cases(group, f"before:{state}", max_count=20)
        print(f"  {state}: selected {n} (pool: {len(group)})")

    print(f"  After before_state sampling: {len(selected_ids)} selected")

    # ── Step 3: Ensure after_state coverage (20 per category) ──
    after_state_groups = defaultdict(list)
    for c in candidates:
        afs = c.get("after_state", "")
        if afs:
            after_state_groups[afs].append(c)

    print(f"\n--- After State Coverage ---")
    for state, group in sorted(after_state_groups.items()):
        random.shuffle(group)
        n = add_cases(group, f"after:{state}", max_count=20)
        print(f"  {state}: selected {n} (pool: {len(group)})")

    print(f"  After after_state sampling: {len(selected_ids)} selected")

    # ── Step 4: 離-related keyword cases (target 100) ──
    ri_cases = [
        c for c in candidates
        if has_keywords(case_text(c), RI_KEYWORDS)
    ]
    random.shuffle(ri_cases)
    n_ri = add_cases(ri_cases, "離-keywords", max_count=100)
    print(f"\n--- 離 (Ri) Keyword Cases ---")
    print(f"  Pool with 離 keywords: {len(ri_cases)}")
    print(f"  Added: {n_ri}")
    print(f"  After 離 sampling: {len(selected_ids)} selected")

    # ── Step 5: 兌-related keyword cases (target 100) ──
    da_cases = [
        c for c in candidates
        if has_keywords(case_text(c), DA_KEYWORDS)
    ]
    random.shuffle(da_cases)
    n_da = add_cases(da_cases, "兌-keywords", max_count=100)
    print(f"\n--- 兌 (Da) Keyword Cases ---")
    print(f"  Pool with 兌 keywords: {len(da_cases)}")
    print(f"  Added: {n_da}")
    print(f"  After 兌 sampling: {len(selected_ids)} selected")

    # ── Step 6: Scale diversity (50 per scale category) ──
    scale_groups = defaultdict(list)
    for c in candidates:
        s = c.get("scale", "") or "other"
        scale_groups[s].append(c)

    print(f"\n--- Scale Coverage ---")
    target_scales = ["company", "individual", "country", "family"]
    for scale in target_scales:
        group = scale_groups.get(scale, [])
        random.shuffle(group)
        n = add_cases(group, f"scale:{scale}", max_count=50)
        print(f"  {scale}: selected {n} (pool: {len(group)})")

    # Also add from 'other' scale
    other_group = scale_groups.get("other", [])
    random.shuffle(other_group)
    n_other = add_cases(other_group, "scale:other", max_count=30)
    print(f"  other: selected {n_other} (pool: {len(other_group)})")

    print(f"  After scale sampling: {len(selected_ids)} selected")

    # ── Step 7: Domain diversity fill ──
    domain_groups = defaultdict(list)
    for c in candidates:
        d = c.get("main_domain", "") or "NONE"
        domain_groups[d].append(c)

    # Sort domains by size (smallest first) to boost underrepresented domains
    sorted_domains = sorted(domain_groups.keys(), key=lambda d: len(domain_groups[d]))

    remaining_slots = TARGET_COUNT - len(selected_ids)
    print(f"\n--- Domain Diversity Fill ---")
    print(f"  Remaining slots: {remaining_slots}")

    if remaining_slots > 0:
        # Round-robin across domains
        per_domain = max(1, remaining_slots // len(sorted_domains))
        filled = 0
        for domain in sorted_domains:
            group = domain_groups[domain]
            random.shuffle(group)
            n = add_cases(group, f"domain:{domain}", max_count=per_domain)
            filled += n
            if len(selected_ids) >= TARGET_COUNT:
                break

        print(f"  Added via domain round-robin: {filled}")

    # ── Step 8: Final random fill to reach 800 ──
    remaining_slots = TARGET_COUNT - len(selected_ids)
    if remaining_slots > 0:
        remaining_pool = [c for c in candidates if c["transition_id"] not in selected_ids]
        random.shuffle(remaining_pool)
        n_final = add_cases(remaining_pool, "random_fill", max_count=remaining_slots)
        print(f"\n--- Random Fill ---")
        print(f"  Added: {n_final}")

    # ── Build output ──
    selected_cases = [cand_by_id[tid] for tid in selected_ids if tid in cand_by_id]
    output = [extract_meta(c) for c in selected_cases]

    # Sort by transition_id for deterministic output
    output.sort(key=lambda x: x["transition_id"])

    # ── Save ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"FINAL SELECTION: {len(output)} cases")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"{'=' * 60}")

    # ── Statistics ──
    print_statistics(output, candidates)

    return output


def print_statistics(output, candidates):
    """Print detailed selection statistics."""
    print(f"\n{'=' * 60}")
    print("SELECTION STATISTICS")
    print(f"{'=' * 60}")

    # Before state distribution
    bs_dist = defaultdict(int)
    for c in output:
        bs_dist[c["before_state"]] += 1
    print(f"\n--- Before State Distribution ---")
    for k, v in sorted(bs_dist.items(), key=lambda x: -x[1]):
        pct = 100 * v / len(output)
        print(f"  {k}: {v} ({pct:.1f}%)")

    # After state distribution
    as_dist = defaultdict(int)
    for c in output:
        as_dist[c["after_state"]] += 1
    print(f"\n--- After State Distribution ---")
    for k, v in sorted(as_dist.items(), key=lambda x: -x[1]):
        pct = 100 * v / len(output)
        print(f"  {k}: {v} ({pct:.1f}%)")

    # Scale distribution
    sc_dist = defaultdict(int)
    for c in output:
        sc_dist[c["scale"]] += 1
    print(f"\n--- Scale Distribution ---")
    for k, v in sorted(sc_dist.items(), key=lambda x: -x[1]):
        pct = 100 * v / len(output)
        print(f"  {k}: {v} ({pct:.1f}%)")

    # Domain distribution (top 20)
    dom_dist = defaultdict(int)
    for c in output:
        dom_dist[c["main_domain"] or "NONE"] += 1
    print(f"\n--- Domain Distribution (top 20 of {len(dom_dist)} domains) ---")
    for k, v in sorted(dom_dist.items(), key=lambda x: -x[1])[:20]:
        pct = 100 * v / len(output)
        print(f"  {k}: {v} ({pct:.1f}%)")

    # Story length stats
    lengths = [c["story_length"] for c in output]
    print(f"\n--- Story Length ---")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {sum(lengths)/len(lengths):.1f}")
    print(f"  Median: {sorted(lengths)[len(lengths)//2]}")

    # Keyword coverage check
    selected_ids = {c["transition_id"] for c in output}
    ri_count = 0
    da_count = 0
    for c in candidates:
        if c["transition_id"] in selected_ids:
            text = case_text(c)
            if has_keywords(text, RI_KEYWORDS):
                ri_count += 1
            if has_keywords(text, DA_KEYWORDS):
                da_count += 1
    print(f"\n--- Trigram Keyword Coverage in Selection ---")
    print(f"  Cases with 離 keywords: {ri_count} ({100*ri_count/len(output):.1f}%)")
    print(f"  Cases with 兌 keywords: {da_count} ({100*da_count/len(output):.1f}%)")


if __name__ == "__main__":
    select_cases()
