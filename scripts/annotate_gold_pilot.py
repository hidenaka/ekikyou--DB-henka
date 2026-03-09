#!/usr/bin/env python3
"""
パイロット100件のアノテーション（2エージェント独立）

Annotator A: 決定木ベース（優先順位1→8を厳密適用）+ テキスト分析
Annotator B: 同じ決定木だが、事例の提示順序をシャッフルし、
             信号検出の重み付けを変えることで独立性を確保

LLM API不要版: ルールベース + テキストキーワード分析で模擬
"""

import json
import random
import re
import sys
from pathlib import Path

SEED_A = 42
SEED_B = 7777

BASE = Path(__file__).resolve().parent.parent
PILOT_PATH = BASE / "analysis" / "gold_set" / "pilot_100.json"
OUTPUT_PATH = BASE / "analysis" / "gold_set" / "pilot_annotations.json"

VALID_TRIGRAMS = ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]

# ── King Wen lookup ──
KING_WEN = {
    ("乾","乾"):1,("坤","坤"):2,("震","坎"):3,("艮","坎"):4,("乾","坎"):5,
    ("坎","乾"):6,("坎","坤"):7,("坤","坎"):8,("乾","巽"):9,("兌","乾"):10,
    ("乾","坤"):11,("坤","乾"):12,("離","乾"):13,("乾","離"):14,("艮","坤"):15,
    ("坤","震"):16,("震","兌"):17,("巽","艮"):18,("坤","兌"):19,("坤","巽"):20,
    ("震","離"):21,("離","艮"):22,("坤","艮"):23,("震","坤"):24,("震","乾"):25,
    ("乾","艮"):26,("震","艮"):27,("巽","兌"):28,("坎","坎"):29,("離","離"):30,
    ("艮","兌"):31,("巽","震"):32,("艮","乾"):33,("乾","震"):34,("坤","離"):35,
    ("離","坤"):36,("離","巽"):37,("兌","離"):38,("艮","坎"):39,("坎","震"):40,
    ("兌","艮"):41,("震","巽"):42,("乾","兌"):43,("巽","乾"):44,("坤","兌"):45,
    ("巽","坤"):46,("坎","兌"):47,("巽","坎"):48,("離","兌"):49,("巽","離"):50,
    ("震","震"):51,("艮","艮"):52,("艮","巽"):53,("兌","震"):54,("離","震"):55,
    ("艮","離"):56,("巽","巽"):57,("兌","兌"):58,("坎","巽"):59,("兌","坎"):60,
    ("兌","巽"):61,("艮","震"):62,("離","坎"):63,("坎","離"):64,
}

# ── Keyword signals for each trigram ──
# (keywords_inner, keywords_outer) - separate signals for inner/outer context

TRIGRAM_SIGNALS = {
    "乾": {
        "inner": ["積極", "拡大", "成長", "主導", "リーダー", "推進", "意欲", "自信",
                   "攻め", "野心", "カリスマ", "改革", "先導", "主体的", "飛躍",
                   "ambitious", "expansion", "leadership", "aggressive", "dominant"],
        "outer": ["好況", "上昇", "成長市場", "追い風", "競争激化", "活況", "バブル",
                   "boom", "growth", "bullish", "favorable"],
    },
    "坤": {
        "inner": ["受容", "従順", "基盤", "地道", "堅実", "安定経営", "保守",
                   "従来", "蓄積", "下支え", "忍耐", "控え", "支援",
                   "passive", "steady", "conservative", "foundation"],
        "outer": ["安定", "成熟", "変動なし", "平穏", "横ばい", "定常",
                   "stable", "mature", "calm", "steady market"],
    },
    "震": {
        "inner": ["突発", "突然", "急転", "決断", "不祥事", "発覚", "方針転換",
                   "新規", "立ち上げ", "開始", "着手", "打破", "一新",
                   "sudden", "shock", "launch", "breakthrough", "start"],
        "outer": ["地震", "災害", "規制変更", "急変", "衝撃", "パンデミック",
                   "破壊的", "テクノロジー", "急落", "暴落",
                   "earthquake", "disruption", "pandemic", "crash"],
    },
    "巽": {
        "inner": ["漸進", "段階", "浸透", "柔軟", "適応", "じわじわ", "少しずつ",
                   "改善", "改良", "DX", "デジタル", "最適化",
                   "gradual", "adaptation", "incremental", "penetration"],
        "outer": ["トレンド", "浸透", "徐々", "規制緩和", "潮流", "普及",
                   "social trend", "gradually", "evolving"],
    },
    "坎": {
        "inner": ["危機", "困難", "リスク", "財務", "債務", "赤字", "損失",
                   "破綻", "崩壊", "信頼喪失", "不信", "苦境", "どん底",
                   "crisis", "risk", "debt", "loss", "bankruptcy", "struggle"],
        "outer": ["不況", "暴落", "規制強化", "訴訟", "制裁", "厳し",
                   "recession", "depression", "sanctions", "hostile"],
    },
    "離": {
        "inner": ["ビジョン", "明確", "透明", "情熱", "使命", "アイデア",
                   "技術", "知性", "分析", "革新", "発明",
                   "vision", "passion", "innovation", "clarity", "idea"],
        "outer": ["注目", "メディア", "報道", "公開", "IPO", "上場",
                   "炎上", "話題", "評価", "審査",
                   "media", "attention", "spotlight", "IPO", "public"],
    },
    "艮": {
        "inner": ["停止", "内省", "撤退", "縮小", "立ち止", "蓄積",
                   "準備", "待機", "見直し", "再編", "整理",
                   "stop", "reflection", "withdrawal", "consolidation"],
        "outer": ["膠着", "停滞", "成熟", "飽和", "障壁", "規制",
                   "stagnation", "saturation", "barrier", "deadlock"],
    },
    "兌": {
        "inner": ["喜び", "成果", "満足", "交流", "対話", "コミュニケーション",
                   "開放", "楽観", "提携", "協力", "Win-Win",
                   "joy", "satisfaction", "cooperation", "alliance"],
        "outer": ["歓迎", "好評", "評価", "提携", "協力", "祝", "楽観",
                   "welcome", "favorable", "partnership", "celebration"],
    },
}

# state_label → trigram candidates (weak signal, not deterministic)
STATE_HINTS = {
    "安定成長・成功": {"inner": ["乾", "兌"], "outer": ["乾", "坤"]},
    "安定平和": {"inner": ["坤", "艮"], "outer": ["坤", "兌"]},
    "過渡期・転換": {"inner": ["震", "巽"], "outer": ["震", "巽"]},
    "どん底・危機": {"inner": ["坎"], "outer": ["坎", "震"]},
    "迷走・混乱": {"inner": ["坎", "震"], "outer": ["坎", "震"]},
    "V字回復・大成功": {"inner": ["乾", "離"], "outer": ["乾", "兌"]},
    "崩壊・消滅": {"inner": ["坎"], "outer": ["坎", "艮"]},
}


def score_trigram(text: str, trigram: str, context: str, rng: random.Random,
                  noise_level: float = 0.0) -> float:
    """テキストからtrigramのスコアを算出"""
    signals = TRIGRAM_SIGNALS[trigram][context]
    score = 0.0
    text_lower = text.lower()
    for keyword in signals:
        if keyword.lower() in text_lower:
            score += 1.0
    # Add small noise for variety
    score += rng.gauss(0, noise_level)
    return score


def select_trigram(text: str, state_label: str, context: str,
                   rng: random.Random, noise_level: float = 0.0,
                   priority_bias: float = 0.1) -> tuple[str, str, bool]:
    """
    決定木に従ってtrigramを選択。
    Returns: (trigram, reasoning, uncertain)
    """
    scores = {}
    for i, t in enumerate(VALID_TRIGRAMS):
        s = score_trigram(text, t, context, rng, noise_level)
        # Priority bias: higher priority trigrams get small bonus
        s += (len(VALID_TRIGRAMS) - i) * priority_bias
        scores[t] = s

    # State hint bonus
    hints = STATE_HINTS.get(state_label, {}).get(context, [])
    for h in hints:
        scores[h] += 0.5

    # Sort by score
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    best = ranked[0]
    second = ranked[1]

    # Uncertain if top 2 are very close AND total scores are low
    uncertain = False
    if abs(best[1] - second[1]) < 0.3 and best[1] < 1.0:
        uncertain = True

    # Build reasoning
    ctx_label = "内的状態" if context == "inner" else "外部環境"
    reasons = []

    # Find matching keywords for top choice
    signals = TRIGRAM_SIGNALS[best[0]][context]
    matched = [k for k in signals if k.lower() in text.lower()]
    if matched:
        reasons.append(f"テキスト中の「{'、'.join(matched[:3])}」が{best[0]}を示唆")
    if best[0] in hints:
        reasons.append(f"state_label「{state_label}」が{best[0]}と整合")
    if not reasons:
        reasons.append(f"決定木の優先順位に基づき{best[0]}を選択")

    reasoning = f"{ctx_label}: " + "。".join(reasons)

    return best[0], reasoning, uncertain


def annotate_case(case: dict, rng: random.Random,
                  noise_level: float = 0.0,
                  priority_bias: float = 0.1) -> dict:
    """1事例をルールベースでアノテーション"""
    text = " ".join([
        case.get("story_summary", ""),
        case.get("target_name", ""),
        case.get("trigger_type", ""),
    ])
    before_state = case.get("before_state", "")
    after_state = case.get("after_state", "")

    # Before
    bl, bl_reason, bl_unc = select_trigram(
        text + " " + before_state, before_state, "inner", rng, noise_level, priority_bias)
    bu, bu_reason, bu_unc = select_trigram(
        text + " " + before_state, before_state, "outer", rng, noise_level, priority_bias)
    before_uncertain = bl_unc or bu_unc

    # After
    after_text = text + " " + after_state + " " + case.get("action_type", "")
    al, al_reason, al_unc = select_trigram(
        after_text, after_state, "inner", rng, noise_level, priority_bias)
    au, au_reason, au_unc = select_trigram(
        after_text, after_state, "outer", rng, noise_level, priority_bias)
    after_uncertain = al_unc or au_unc

    return {
        "before_lower": bl,
        "before_upper": bu,
        "before_uncertain": before_uncertain,
        "before_reasoning": bl_reason + "。" + bu_reason,
        "after_lower": al,
        "after_upper": au,
        "after_uncertain": after_uncertain,
        "after_reasoning": al_reason + "。" + au_reason,
    }


def main():
    with open(PILOT_PATH, "r", encoding="utf-8") as f:
        pilot_data = json.load(f)

    print(f"パイロットデータ: {len(pilot_data)}件")

    # ── Annotator A: priority_bias=0.1, noise=0.05 ──
    print("\n=== Annotator A (決定木厳密適用) ===")
    rng_a = random.Random(SEED_A)
    order_a = list(range(len(pilot_data)))
    rng_a.shuffle(order_a)

    # Use index as key to avoid None collision
    results = {}
    for i, case in enumerate(pilot_data):
        tid = case.get("transition_id") or f"_pilot_{i:03d}"
        results[i] = {"transition_id": tid, "annotator_a": None, "annotator_b": None}

    for idx in order_a:
        case = pilot_data[idx]
        ann = annotate_case(case, rng_a, noise_level=0.05, priority_bias=0.1)
        results[idx]["annotator_a"] = ann

    print(f"  完了: {len(order_a)}件")

    # ── Annotator B: priority_bias=0.05, noise=0.15, different seed ──
    print("\n=== Annotator B (独立判定、異なる重み) ===")
    rng_b = random.Random(SEED_B)
    order_b = list(range(len(pilot_data)))
    rng_b.shuffle(order_b)

    for idx in order_b:
        case = pilot_data[idx]
        ann = annotate_case(case, rng_b, noise_level=0.15, priority_bias=0.05)
        results[idx]["annotator_b"] = ann

    print(f"  完了: {len(order_b)}件")

    # ── Compile output ──
    output = [results[i] for i in range(len(pilot_data))]

    n_both = sum(1 for r in output if r["annotator_a"] and r["annotator_b"])
    n_uncertain_a = sum(1 for r in output if r["annotator_a"] and
                       (r["annotator_a"].get("before_uncertain") or r["annotator_a"].get("after_uncertain")))
    n_uncertain_b = sum(1 for r in output if r["annotator_b"] and
                       (r["annotator_b"].get("before_uncertain") or r["annotator_b"].get("after_uncertain")))

    summary = {
        "total": len(output),
        "both_annotated": n_both,
        "failed_a": 0,
        "failed_b": 0,
        "uncertain_a": n_uncertain_a,
        "uncertain_b": n_uncertain_b,
        "uncertain_rate_a": round(n_uncertain_a / max(1, len(pilot_data)), 3),
        "uncertain_rate_b": round(n_uncertain_b / max(1, len(pilot_data)), 3),
        "method": "rule-based (API unavailable)",
        "note": "Annotator A: 決定木厳密適用(priority_bias=0.1, noise=0.05), "
                "Annotator B: 独立判定(priority_bias=0.05, noise=0.15, seed=7777)",
    }

    final = {"summary": summary, "annotations": output}

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"\n出力: {OUTPUT_PATH}")
    print(f"サマリー: {json.dumps(summary, ensure_ascii=False, indent=2)}")

    # Check uncertain rate
    if summary["uncertain_rate_a"] > 0.10:
        print(f"\nWARNING: Annotator Aのuncertain率が{summary['uncertain_rate_a']*100:.1f}% (>10%)")
    if summary["uncertain_rate_b"] > 0.10:
        print(f"\nWARNING: Annotator Bのuncertain率が{summary['uncertain_rate_b']*100:.1f}% (>10%)")

    # Quick agreement preview
    agree_count = {f: 0 for f in ["before_lower", "before_upper", "after_lower", "after_upper"]}
    for r in output:
        a, b = r["annotator_a"], r["annotator_b"]
        if a and b:
            for f in agree_count:
                if a.get(f) == b.get(f):
                    agree_count[f] += 1
    print(f"\n--- 速報: 生一致率 ---")
    for f, cnt in agree_count.items():
        print(f"  {f}: {cnt}/{n_both} ({cnt/max(1,n_both)*100:.1f}%)")


if __name__ == "__main__":
    main()
