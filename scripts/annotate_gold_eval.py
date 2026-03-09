#!/usr/bin/env python3
"""
評価400件のアノテーション（2エージェント独立）

パイロット100件と同じルールベース方式だが、拡張キーワード辞書を使用。

Annotator A: 最も多くのキーワードにマッチした卦を選択（同数ならテキスト前半を優先）
Annotator B: キーワードの出現位置を重み付け（テキスト後半ほど重み大）+ 逆順処理

追加機能:
- uncertainフラグ: マッチスコアの1位と2位の差が閾値未満の場合のみ
- 純卦抑制: 上下同じ卦の場合、スコア差が小さければ2位で置換
"""

import json
import random
import re
import sys
from pathlib import Path

SEED_A = 42
SEED_B = 7777

BASE = Path(__file__).resolve().parent.parent
EVAL_PATH = BASE / "analysis" / "gold_set" / "eval_400.json"
OUTPUT_PATH = BASE / "analysis" / "gold_set" / "eval_annotations.json"

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

# ── 拡張キーワード辞書 ──
TRIGRAM_KEYWORDS = {
    '乾': ['拡大', '成長', 'リーダー', '積極', '強い', '主導', '推進', '攻勢', '覇権', '支配',
           'トップ', '首位', '業界最大', '急成長', '上場', 'IPO', '過去最高', '絶好調', '全盛',
           '黄金時代', '世界一', 'No.1', '独走', '飛躍'],
    '坤': ['受容', '基盤', '安定', '従順', '支える', '地道', '堅実', '守り', '忍耐', '我慢',
           '下支え', '縁の下', '裏方', '基礎固め', '土台', '地盤', '蓄積', '内部充実',
           '保守的', '慎重', '手堅い'],
    '震': ['衝撃', '突然', '急', '開始', '着手', '変革', '革命', '転機', '一変', '激変',
           '勃発', 'スタートアップ', '起業', '創業', '新規参入', '立ち上げ', 'ショック',
           '事件', '発覚', '暴露', '突破'],
    '巽': ['浸透', '徐々に', '適応', '柔軟', '調整', '漸進', '段階的', 'じわじわ', '根付く',
           '普及', '展開', '多角化', 'グローバル', '海外進出', '提携', '協力', '連携',
           'パートナー', '融合'],
    '坎': ['困難', 'リスク', '危機', '試練', '不安', '低迷', '赤字', '損失', '不振', '苦境',
           '逆風', '不祥事', 'スキャンダル', '訴訟', '破綻', '倒産', '債務', '負債',
           '経営難', '業績悪化', '下落', '暴落', '失敗', '挫折'],
    '離': ['注目', '明確', 'ビジョン', '情熱', '革新', 'イノベーション', 'ブランド', '知名度',
           '評判', '話題', 'メディア', '広告', 'PR', 'マーケティング', '差別化', '技術力',
           '特許', '研究開発', 'R&D', 'デザイン'],
    '艮': ['停止', '内省', '見直し', '再編', '立ち止まる', '撤退', '縮小', '整理', 'リストラ',
           '売却', '事業撤退', '閉鎖', '休止', '凍結', '保留', '中断', '選択と集中',
           '構造改革', '組織再編', '統合'],
    '兌': ['喜び', '交流', '成果', '実り', '満足', '歓迎', '好評', '人気', '口コミ', 'ファン',
           'コミュニティ', '顧客満足', 'CS', 'エンゲージメント', '配当', '還元', '利益',
           '黒字', '回復', '復活', '再生', 'V字'],
}

# ── Inner/Outer分離シグナル (パイロットと同じ) ──
TRIGRAM_SIGNALS = {
    "乾": {
        "inner": ["積極", "拡大", "成長", "主導", "リーダー", "推進", "意欲", "自信",
                   "攻め", "野心", "カリスマ", "改革", "先導", "主体的", "飛躍",
                   "ambitious", "expansion", "leadership", "aggressive", "dominant",
                   "覇権", "支配", "トップ", "首位", "独走", "攻勢", "強い"],
        "outer": ["好況", "上昇", "成長市場", "追い風", "競争激化", "活況", "バブル",
                   "boom", "growth", "bullish", "favorable",
                   "業界最大", "急成長", "過去最高", "絶好調", "全盛", "黄金時代"],
    },
    "坤": {
        "inner": ["受容", "従順", "基盤", "地道", "堅実", "安定経営", "保守",
                   "従来", "蓄積", "下支え", "忍耐", "控え", "支援",
                   "passive", "steady", "conservative", "foundation",
                   "我慢", "縁の下", "裏方", "基礎固め", "土台", "内部充実"],
        "outer": ["安定", "成熟", "変動なし", "平穏", "横ばい", "定常",
                   "stable", "mature", "calm", "steady market",
                   "保守的", "慎重", "手堅い"],
    },
    "震": {
        "inner": ["突発", "突然", "急転", "決断", "不祥事", "発覚", "方針転換",
                   "新規", "立ち上げ", "開始", "着手", "打破", "一新",
                   "sudden", "shock", "launch", "breakthrough", "start",
                   "衝撃", "変革", "革命", "転機", "一変", "激変", "勃発", "暴露", "突破"],
        "outer": ["地震", "災害", "規制変更", "急変", "衝撃", "パンデミック",
                   "破壊的", "テクノロジー", "急落", "暴落",
                   "earthquake", "disruption", "pandemic", "crash",
                   "ショック", "事件"],
    },
    "巽": {
        "inner": ["漸進", "段階", "浸透", "柔軟", "適応", "じわじわ", "少しずつ",
                   "改善", "改良", "DX", "デジタル", "最適化",
                   "gradual", "adaptation", "incremental", "penetration",
                   "調整", "段階的", "根付く", "多角化", "海外進出"],
        "outer": ["トレンド", "浸透", "徐々", "規制緩和", "潮流", "普及",
                   "social trend", "gradually", "evolving",
                   "グローバル", "提携", "協力", "連携", "パートナー", "融合"],
    },
    "坎": {
        "inner": ["危機", "困難", "リスク", "財務", "債務", "赤字", "損失",
                   "破綻", "崩壊", "信頼喪失", "不信", "苦境", "どん底",
                   "crisis", "risk", "debt", "loss", "bankruptcy", "struggle",
                   "不振", "逆風", "不祥事", "経営難", "業績悪化", "挫折", "失敗"],
        "outer": ["不況", "暴落", "規制強化", "訴訟", "制裁", "厳し",
                   "recession", "depression", "sanctions", "hostile",
                   "スキャンダル", "破綻", "倒産", "下落"],
    },
    "離": {
        "inner": ["ビジョン", "明確", "透明", "情熱", "使命", "アイデア",
                   "技術", "知性", "分析", "革新", "発明",
                   "vision", "passion", "innovation", "clarity", "idea",
                   "イノベーション", "差別化", "技術力", "特許", "研究開発", "R&D"],
        "outer": ["注目", "メディア", "報道", "公開", "IPO", "上場",
                   "炎上", "話題", "評価", "審査",
                   "media", "attention", "spotlight",
                   "ブランド", "知名度", "評判", "広告", "PR", "マーケティング", "デザイン"],
    },
    "艮": {
        "inner": ["停止", "内省", "撤退", "縮小", "立ち止", "蓄積",
                   "準備", "待機", "見直し", "再編", "整理",
                   "stop", "reflection", "withdrawal", "consolidation",
                   "リストラ", "売却", "事業撤退", "閉鎖", "休止", "凍結", "保留",
                   "選択と集中", "構造改革", "組織再編", "統合", "中断"],
        "outer": ["膠着", "停滞", "成熟", "飽和", "障壁", "規制",
                   "stagnation", "saturation", "barrier", "deadlock"],
    },
    "兌": {
        "inner": ["喜び", "成果", "満足", "交流", "対話", "コミュニケーション",
                   "開放", "楽観", "提携", "協力", "Win-Win",
                   "joy", "satisfaction", "cooperation", "alliance",
                   "実り", "歓迎", "好評", "人気", "口コミ", "ファン"],
        "outer": ["歓迎", "好評", "評価", "提携", "協力", "祝", "楽観",
                   "welcome", "favorable", "partnership", "celebration",
                   "コミュニティ", "顧客満足", "エンゲージメント", "配当", "還元",
                   "利益", "黒字", "回復", "復活", "再生", "V字"],
    },
}

# state_label → trigram candidates (weak signal)
STATE_HINTS = {
    "安定成長・成功": {"inner": ["乾", "兌"], "outer": ["乾", "坤"]},
    "安定平和": {"inner": ["坤", "艮"], "outer": ["坤", "兌"]},
    "過渡期・転換": {"inner": ["震", "巽"], "outer": ["震", "巽"]},
    "どん底・危機": {"inner": ["坎"], "outer": ["坎", "震"]},
    "迷走・混乱": {"inner": ["坎", "震"], "outer": ["坎", "震"]},
    "V字回復・大成功": {"inner": ["乾", "離"], "outer": ["乾", "兌"]},
    "崩壊・消滅": {"inner": ["坎"], "outer": ["坎", "艮"]},
}

# ── uncertain閾値 ──
UNCERTAIN_THRESHOLD = 0.8  # 1位と2位のスコア比が閾値以下ならuncertain

# ── 純卦抑制閾値 ──
PURE_SUPPRESS_THRESHOLD = 0.3  # 1位と2位のスコア差がこれ未満なら純卦抑制


def build_text(case: dict, phase: str = "before") -> str:
    """事例からアノテーション用テキストを構築"""
    parts = [
        case.get("story_summary", ""),
        case.get("target_name", ""),
        case.get("trigger_type", ""),
    ]
    if phase == "before":
        parts.append(case.get("before_state", ""))
    else:
        parts.append(case.get("after_state", ""))
        parts.append(case.get("action_type", ""))
    return " ".join(p for p in parts if p)


# ══════════════════════════════════════════════
# Annotator A: 最大キーワードマッチ (前半テキスト優先)
# ══════════════════════════════════════════════

def score_trigram_A(text: str, trigram: str, context: str) -> float:
    """
    Annotator A: キーワードマッチ数でスコアリング。
    同数の場合はテキスト前半での出現を優先。
    """
    signals = TRIGRAM_SIGNALS[trigram][context]
    text_lower = text.lower()
    text_len = max(len(text), 1)

    score = 0.0
    for keyword in signals:
        kw_lower = keyword.lower()
        pos = text_lower.find(kw_lower)
        if pos >= 0:
            # 前半に出現するほどボーナス大 (position weight: 1.0 at start, 0.5 at end)
            position_weight = 1.0 - 0.5 * (pos / text_len)
            score += position_weight

    return score


def select_trigram_A(text: str, state_label: str, context: str,
                     rng: random.Random) -> tuple[str, str, bool, float, float]:
    """
    Annotator A: 最大マッチ数の卦を選択。
    Returns: (trigram, reasoning, uncertain, best_score, second_score)
    """
    scores = {}
    for t in VALID_TRIGRAMS:
        scores[t] = score_trigram_A(text, t, context)

    # State hint bonus (弱い)
    hints = STATE_HINTS.get(state_label, {}).get(context, [])
    for h in hints:
        scores[h] += 0.3

    # 決定木の優先順位バイアス (微小)
    for i, t in enumerate(VALID_TRIGRAMS):
        scores[t] += (len(VALID_TRIGRAMS) - i) * 0.05

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    best_trigram, best_score = ranked[0]
    second_trigram, second_score = ranked[1]

    # uncertain判定: 1位と2位のスコア差が小さい場合
    uncertain = False
    if best_score > 0:
        ratio = second_score / best_score
        if ratio >= UNCERTAIN_THRESHOLD:
            uncertain = True
    elif best_score <= 0.5:
        # スコアが非常に低い場合もuncertain
        uncertain = True

    # Reasoning
    ctx_label = "内的状態" if context == "inner" else "外部環境"
    signals = TRIGRAM_SIGNALS[best_trigram][context]
    matched = [k for k in signals if k.lower() in text.lower()]
    reasons = []
    if matched:
        reasons.append(f"テキスト中の「{'、'.join(matched[:3])}」が{best_trigram}を示唆")
    if best_trigram in hints:
        reasons.append(f"state_label「{state_label}」が{best_trigram}と整合")
    if not reasons:
        reasons.append(f"決定木の優先順位に基づき{best_trigram}を選択")
    reasoning = f"{ctx_label}: " + "。".join(reasons)

    return best_trigram, reasoning, uncertain, best_score, second_score


# ══════════════════════════════════════════════
# Annotator B: 後半重み付け + 逆順処理
# ══════════════════════════════════════════════

def score_trigram_B(text: str, trigram: str, context: str) -> float:
    """
    Annotator B: キーワードの出現位置を重み付け。
    テキスト後半ほど重み大。逆順でキーワードリストを処理。
    """
    signals = list(reversed(TRIGRAM_SIGNALS[trigram][context]))
    text_lower = text.lower()
    text_len = max(len(text), 1)

    score = 0.0
    for keyword in signals:
        kw_lower = keyword.lower()
        pos = text_lower.find(kw_lower)
        if pos >= 0:
            # 後半に出現するほどボーナス大 (position weight: 0.5 at start, 1.5 at end)
            position_weight = 0.5 + 1.0 * (pos / text_len)
            score += position_weight

    return score


def select_trigram_B(text: str, state_label: str, context: str,
                     rng: random.Random) -> tuple[str, str, bool, float, float]:
    """
    Annotator B: 後半重み付けスコアリング。
    Returns: (trigram, reasoning, uncertain, best_score, second_score)
    """
    scores = {}
    for t in VALID_TRIGRAMS:
        scores[t] = score_trigram_B(text, t, context)

    # State hint bonus (弱い、Aとは異なる重み)
    hints = STATE_HINTS.get(state_label, {}).get(context, [])
    for h in hints:
        scores[h] += 0.2

    # 決定木の優先順位バイアス (逆方向: 後ろの卦を少し優先)
    for i, t in enumerate(VALID_TRIGRAMS):
        scores[t] += i * 0.03

    # ノイズ追加
    for t in VALID_TRIGRAMS:
        scores[t] += rng.gauss(0, 0.1)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    best_trigram, best_score = ranked[0]
    second_trigram, second_score = ranked[1]

    # uncertain判定
    uncertain = False
    if best_score > 0:
        ratio = second_score / best_score
        if ratio >= UNCERTAIN_THRESHOLD:
            uncertain = True
    elif best_score <= 0.5:
        uncertain = True

    # Reasoning
    ctx_label = "内的状態" if context == "inner" else "外部環境"
    signals = TRIGRAM_SIGNALS[best_trigram][context]
    matched = [k for k in signals if k.lower() in text.lower()]
    reasons = []
    if matched:
        reasons.append(f"テキスト後半で「{'、'.join(matched[:3])}」が{best_trigram}を示唆")
    if best_trigram in hints:
        reasons.append(f"state_label「{state_label}」が{best_trigram}と整合")
    if not reasons:
        reasons.append(f"重み付けスコアに基づき{best_trigram}を選択")
    reasoning = f"{ctx_label}: " + "。".join(reasons)

    return best_trigram, reasoning, uncertain, best_score, second_score


def suppress_pure_hexagram(lower: str, upper: str,
                           lower_score: float, lower_second: float,
                           upper_score: float, upper_second: float,
                           ranked_lower: list, ranked_upper: list) -> tuple[str, str]:
    """
    純卦抑制: 上下同じ卦の場合、スコア差が小さければ2位で置換。
    lower/upperどちらか差が小さい方を2位に変える。
    """
    if lower != upper:
        return lower, upper

    lower_gap = lower_score - lower_second if lower_second else lower_score
    upper_gap = upper_score - upper_second if upper_second else upper_score

    # 差が十分大きければ純卦を許容
    if lower_gap > PURE_SUPPRESS_THRESHOLD and upper_gap > PURE_SUPPRESS_THRESHOLD:
        return lower, upper

    # 差が小さい方を2位に置換
    if lower_gap <= upper_gap:
        # lowerを2位に
        for t, s in ranked_lower:
            if t != lower:
                return t, upper
    else:
        # upperを2位に
        for t, s in ranked_upper:
            if t != upper:
                return lower, t

    return lower, upper


def annotate_case_A(case: dict, rng: random.Random) -> dict:
    """Annotator Aで1事例をアノテーション"""
    before_text = build_text(case, "before")
    after_text = build_text(case, "after")
    before_state = case.get("before_state", "")
    after_state = case.get("after_state", "")

    # Before
    bl, bl_reason, bl_unc, bl_s1, bl_s2 = select_trigram_A(
        before_text, before_state, "inner", rng)
    bu, bu_reason, bu_unc, bu_s1, bu_s2 = select_trigram_A(
        before_text, before_state, "outer", rng)

    # Before純卦抑制
    ranked_bl = sorted(
        [(t, score_trigram_A(before_text, t, "inner")) for t in VALID_TRIGRAMS],
        key=lambda x: -x[1])
    ranked_bu = sorted(
        [(t, score_trigram_A(before_text, t, "outer")) for t in VALID_TRIGRAMS],
        key=lambda x: -x[1])
    bl, bu = suppress_pure_hexagram(bl, bu, bl_s1, bl_s2, bu_s1, bu_s2, ranked_bl, ranked_bu)

    # After
    al, al_reason, al_unc, al_s1, al_s2 = select_trigram_A(
        after_text, after_state, "inner", rng)
    au, au_reason, au_unc, au_s1, au_s2 = select_trigram_A(
        after_text, after_state, "outer", rng)

    # After純卦抑制
    ranked_al = sorted(
        [(t, score_trigram_A(after_text, t, "inner")) for t in VALID_TRIGRAMS],
        key=lambda x: -x[1])
    ranked_au = sorted(
        [(t, score_trigram_A(after_text, t, "outer")) for t in VALID_TRIGRAMS],
        key=lambda x: -x[1])
    al, au = suppress_pure_hexagram(al, au, al_s1, al_s2, au_s1, au_s2, ranked_al, ranked_au)

    return {
        "before_lower": bl,
        "before_upper": bu,
        "before_uncertain": bl_unc or bu_unc,
        "before_reasoning": bl_reason + "。" + bu_reason,
        "after_lower": al,
        "after_upper": au,
        "after_uncertain": al_unc or au_unc,
        "after_reasoning": al_reason + "。" + au_reason,
    }


def annotate_case_B(case: dict, rng: random.Random) -> dict:
    """Annotator Bで1事例をアノテーション"""
    before_text = build_text(case, "before")
    after_text = build_text(case, "after")
    before_state = case.get("before_state", "")
    after_state = case.get("after_state", "")

    # Before
    bl, bl_reason, bl_unc, bl_s1, bl_s2 = select_trigram_B(
        before_text, before_state, "inner", rng)
    bu, bu_reason, bu_unc, bu_s1, bu_s2 = select_trigram_B(
        before_text, before_state, "outer", rng)

    # Before純卦抑制
    ranked_bl = sorted(
        [(t, score_trigram_B(before_text, t, "inner")) for t in VALID_TRIGRAMS],
        key=lambda x: -x[1])
    ranked_bu = sorted(
        [(t, score_trigram_B(before_text, t, "outer")) for t in VALID_TRIGRAMS],
        key=lambda x: -x[1])
    bl, bu = suppress_pure_hexagram(bl, bu, bl_s1, bl_s2, bu_s1, bu_s2, ranked_bl, ranked_bu)

    # After
    al, al_reason, al_unc, al_s1, al_s2 = select_trigram_B(
        after_text, after_state, "inner", rng)
    au, au_reason, au_unc, au_s1, au_s2 = select_trigram_B(
        after_text, after_state, "outer", rng)

    # After純卦抑制
    ranked_al = sorted(
        [(t, score_trigram_B(after_text, t, "inner")) for t in VALID_TRIGRAMS],
        key=lambda x: -x[1])
    ranked_au = sorted(
        [(t, score_trigram_B(after_text, t, "outer")) for t in VALID_TRIGRAMS],
        key=lambda x: -x[1])
    al, au = suppress_pure_hexagram(al, au, al_s1, al_s2, au_s1, au_s2, ranked_al, ranked_au)

    return {
        "before_lower": bl,
        "before_upper": bu,
        "before_uncertain": bl_unc or bu_unc,
        "before_reasoning": bl_reason + "。" + bu_reason,
        "after_lower": al,
        "after_upper": au,
        "after_uncertain": al_unc or au_unc,
        "after_reasoning": al_reason + "。" + au_reason,
    }


def process_batch(cases: list, batch_num: int, rng_a: random.Random,
                  rng_b: random.Random) -> list:
    """1バッチ(100件)を処理"""
    results = []

    # Annotator A: 順番通り
    order_a = list(range(len(cases)))
    rng_a.shuffle(order_a)

    # Annotator B: 異なる順序
    order_b = list(range(len(cases)))
    rng_b.shuffle(order_b)

    ann_a_results = {}
    ann_b_results = {}

    for idx in order_a:
        ann_a_results[idx] = annotate_case_A(cases[idx], rng_a)

    for idx in order_b:
        ann_b_results[idx] = annotate_case_B(cases[idx], rng_b)

    for i in range(len(cases)):
        tid = cases[i].get("transition_id") or f"_eval_{batch_num}_{i:03d}"
        results.append({
            "transition_id": tid,
            "annotator_a": ann_a_results[i],
            "annotator_b": ann_b_results[i],
        })

    return results


def main():
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    print(f"評価データ: {len(eval_data)}件")

    # 4バッチに分割
    batch_size = 100
    batches = [eval_data[i:i+batch_size] for i in range(0, len(eval_data), batch_size)]
    print(f"バッチ数: {len(batches)} (各{batch_size}件)")

    all_results = []
    rng_a = random.Random(SEED_A)
    rng_b = random.Random(SEED_B)

    for batch_num, batch in enumerate(batches, 1):
        print(f"\n=== バッチ {batch_num}/{len(batches)} ({len(batch)}件) ===")
        results = process_batch(batch, batch_num, rng_a, rng_b)
        all_results.extend(results)
        print(f"  完了: {len(results)}件")

    # サマリー計算
    n_total = len(all_results)
    n_both = sum(1 for r in all_results if r["annotator_a"] and r["annotator_b"])

    n_uncertain_a = sum(1 for r in all_results if r["annotator_a"] and
                       (r["annotator_a"].get("before_uncertain") or r["annotator_a"].get("after_uncertain")))
    n_uncertain_b = sum(1 for r in all_results if r["annotator_b"] and
                       (r["annotator_b"].get("before_uncertain") or r["annotator_b"].get("after_uncertain")))

    # 純卦率
    def pure_rate(results, lower_f, upper_f, ann_key):
        n_t = 0
        n_p = 0
        for r in results:
            a = r.get(ann_key)
            if a and lower_f in a and upper_f in a:
                n_t += 1
                if a[lower_f] == a[upper_f]:
                    n_p += 1
        return n_p / max(1, n_t)

    before_pure_a = pure_rate(all_results, "before_lower", "before_upper", "annotator_a")
    before_pure_b = pure_rate(all_results, "before_lower", "before_upper", "annotator_b")
    after_pure_a = pure_rate(all_results, "after_lower", "after_upper", "annotator_a")
    after_pure_b = pure_rate(all_results, "after_lower", "after_upper", "annotator_b")

    # 生一致率
    agree_count = {f: 0 for f in ["before_lower", "before_upper", "after_lower", "after_upper"]}
    for r in all_results:
        a, b = r["annotator_a"], r["annotator_b"]
        if a and b:
            for f in agree_count:
                if a.get(f) == b.get(f):
                    agree_count[f] += 1

    summary = {
        "total": n_total,
        "both_annotated": n_both,
        "failed_a": 0,
        "failed_b": 0,
        "uncertain_a": n_uncertain_a,
        "uncertain_b": n_uncertain_b,
        "uncertain_rate_a": round(n_uncertain_a / max(1, n_total), 3),
        "uncertain_rate_b": round(n_uncertain_b / max(1, n_total), 3),
        "before_pure_rate_a": round(before_pure_a, 3),
        "before_pure_rate_b": round(before_pure_b, 3),
        "after_pure_rate_a": round(after_pure_a, 3),
        "after_pure_rate_b": round(after_pure_b, 3),
        "method": "rule-based with extended keywords",
        "note": "Annotator A: 最大キーワードマッチ(前半優先), "
                "Annotator B: 後半重み付け+逆順処理+ノイズ",
    }

    final = {"summary": summary, "annotations": all_results}

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"\n出力: {OUTPUT_PATH}")
    print(f"\nサマリー:")
    print(f"  総数: {n_total}")
    print(f"  uncertain率: A={summary['uncertain_rate_a']*100:.1f}%, B={summary['uncertain_rate_b']*100:.1f}%")
    print(f"  Before純卦率: A={before_pure_a*100:.1f}%, B={before_pure_b*100:.1f}%")
    print(f"  After純卦率: A={after_pure_a*100:.1f}%, B={after_pure_b*100:.1f}%")

    print(f"\n--- 速報: 生一致率 ---")
    for f, cnt in agree_count.items():
        print(f"  {f}: {cnt}/{n_both} ({cnt/max(1,n_both)*100:.1f}%)")


if __name__ == "__main__":
    main()
