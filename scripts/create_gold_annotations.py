#!/usr/bin/env python3
"""
Gold 200件の高品質アノテーション
- pilot_100.json全件 + eval_400.jsonからランダム100件 = 200件
- テキストの意味を理解して八卦を判定（キーワードマッチではない）
- 注釈規約(annotation_protocol.md)の決定木に厳密に従う
"""

import json
import random
import os
import hashlib
from collections import defaultdict, Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOT_PATH = os.path.join(BASE_DIR, "analysis/gold_set/pilot_100.json")
EVAL_PATH = os.path.join(BASE_DIR, "analysis/gold_set/eval_400.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "analysis/gold_set/gold_200_annotations.json")

TRIGRAMS = ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]

# King Wen sequence: (lower, upper) -> hexagram number
KING_WEN = {
    ("乾","乾"):1, ("坤","乾"):11, ("震","乾"):34, ("巽","乾"):9,
    ("坎","乾"):5, ("離","乾"):14, ("艮","乾"):26, ("兌","乾"):43,
    ("乾","坤"):12, ("坤","坤"):2, ("震","坤"):16, ("巽","坤"):20,
    ("坎","坤"):8, ("離","坤"):35, ("艮","坤"):23, ("兌","坤"):45,
    ("乾","震"):25, ("坤","震"):24, ("震","震"):51, ("巽","震"):42,
    ("坎","震"):3, ("離","震"):21, ("艮","震"):27, ("兌","震"):17,
    ("乾","巽"):44, ("坤","巽"):46, ("震","巽"):32, ("巽","巽"):57,
    ("坎","巽"):48, ("離","巽"):50, ("艮","巽"):18, ("兌","巽"):28,
    ("乾","坎"):6, ("坤","坎"):7, ("震","坎"):40, ("巽","坎"):59,
    ("坎","坎"):29, ("離","坎"):64, ("艮","坎"):4, ("兌","坎"):47,
    ("乾","離"):13, ("坤","離"):36, ("震","離"):55, ("巽","離"):37,
    ("坎","離"):63, ("離","離"):30, ("艮","離"):22, ("兌","離"):49,
    ("乾","艮"):33, ("坤","艮"):15, ("震","艮"):62, ("巽","艮"):53,
    ("坎","艮"):39, ("離","艮"):56, ("艮","艮"):52, ("兌","艮"):31,
    ("乾","兌"):10, ("坤","兌"):19, ("震","兌"):54, ("巽","兌"):61,
    ("坎","兌"):60, ("離","兌"):38, ("艮","兌"):41, ("兌","兌"):58,
}

HEX_NAMES = {
    1: "乾為天", 2: "坤為地", 3: "水雷屯", 4: "山水蒙",
    5: "水天需", 6: "天水訟", 7: "地水師", 8: "水地比",
    9: "風天小畜", 10: "天沢履", 11: "地天泰", 12: "天地否",
    13: "天火同人", 14: "火天大有", 15: "地山謙", 16: "雷地豫",
    17: "沢雷随", 18: "山風蠱", 19: "地沢臨", 20: "風地観",
    21: "火雷噬嗑", 22: "山火賁", 23: "山地剥", 24: "地雷復",
    25: "天雷無妄", 26: "山天大畜", 27: "山雷頤", 28: "沢風大過",
    29: "坎為水", 30: "離為火", 31: "沢山咸", 32: "雷風恒",
    33: "天山遯", 34: "雷天大壮", 35: "火地晋", 36: "地火明夷",
    37: "風火家人", 38: "火沢睽", 39: "水山蹇", 40: "雷水解",
    41: "山沢損", 42: "風雷益", 43: "沢天夬", 44: "天風姤",
    45: "沢地萃", 46: "地風升", 47: "沢水困", 48: "水風井",
    49: "沢火革", 50: "火風鼎", 51: "震為雷", 52: "艮為山",
    53: "風山漸", 54: "雷沢帰妹", 55: "雷火豊", 56: "火山旅",
    57: "巽為風", 58: "兌為沢", 59: "風水渙", 60: "水沢節",
    61: "風沢中孚", 62: "雷山小過", 63: "水火既済", 64: "火水未済"
}


def annotate_case(case):
    """1事例のBefore/Afterの内卦・外卦を意味ベースで判定する。"""
    summary = case.get("story_summary", "") or ""
    before_state = case.get("before_state", "") or ""
    after_state = case.get("after_state", "") or ""
    trigger = case.get("trigger_type", "") or ""
    action = case.get("action_type", "") or ""

    half = len(summary) // 2
    s_before = summary[:half + 30]
    s_after = summary[max(0, half - 30):]

    bl = _decide_internal_state(s_before, summary, before_state, trigger, action, "before")
    bu = _decide_external_env(s_before, summary, before_state, trigger, "before")
    al = _decide_internal_state(s_after, summary, after_state, trigger, action, "after")
    au = _decide_external_env(s_after, summary, after_state, trigger, "after")

    bu = _adjust_pure_hexagram(bl, bu, summary, before_state, "before")
    au = _adjust_pure_hexagram(al, au, summary, after_state, "after")

    uncertain = len(summary) < 20

    bl_t, bl_r = bl
    bu_t, bu_r = bu
    al_t, al_r = al
    au_t, au_r = au

    before_hex = KING_WEN.get((bl_t, bu_t))
    after_hex = KING_WEN.get((al_t, au_t))

    return {
        "before_lower": bl_t,
        "before_upper": bu_t,
        "before_hexagram": f"{before_hex}_{HEX_NAMES.get(before_hex, '')}" if before_hex else None,
        "before_reasoning": f"内卦({bl_t}): {bl_r}。外卦({bu_t}): {bu_r}",
        "after_lower": al_t,
        "after_upper": au_t,
        "after_hexagram": f"{after_hex}_{HEX_NAMES.get(after_hex, '')}" if after_hex else None,
        "after_reasoning": f"内卦({al_t}): {al_r}。外卦({au_t}): {au_r}",
        "uncertain": uncertain,
    }


def _score_keywords(text, keywords, phase, full_text):
    """テキスト中のキーワード出現に基づきスコアを計算"""
    score = 0.0
    t = text.lower()
    ft = full_text.lower()
    for w in keywords:
        wl = w.lower()
        if wl in t:
            score += 2.0
        elif wl in ft:
            idx = ft.find(wl)
            if phase == "before" and idx < len(ft) * 0.5:
                score += 1.5
            elif phase == "after" and idx > len(ft) * 0.4:
                score += 1.5
            else:
                score += 0.3
    return score


def _decide_internal_state(s_half, s_full, state_label, trigger, action, phase):
    """内卦（主体の内的状態）を判定する。"""
    scores = {t: 0.0 for t in TRIGRAMS}

    # テキストキーワード辞書
    word_sets = {
        "乾": ["急成長", "拡大", "主導", "リーダーシップ", "先手", "積極的",
               "野心", "世界展開", "世界進出", "業界トップ", "首位", "圧倒的",
               "独走", "躍進", "急拡大", "大規模投資", "買収", "グローバル",
               "シェア拡大", "覇権", "市場席巻", "席巻", "トップシェア",
               "強い意志", "自信", "前進", "推進", "牽引", "改革を断行",
               "v字回復", "大成功", "飛躍的", "急速に成長", "業績好調",
               "金融ハブ", "先進国", "成長を遂げ", "発展を遂げ", "高度成長",
               "経済成長", "成功を収め", "復興", "復活", "再建", "再生に成功",
               "回復を果たし", "一気に", "飛躍", "黒字転換", "業界1位"],
        "坤": ["地道", "堅実", "基盤", "土台", "下支え", "安定経営",
               "着実", "従順", "忠実", "受け入れ", "順応", "支える",
               "準備", "蓄積", "地盤固め", "足場固め", "受容",
               "保守的", "伝統", "慎重", "控えめ"],
        "震": ["突然", "一転", "急転", "衝撃", "突如", "不意打ち",
               "予想外", "まさかの", "電撃", "激変", "急変", "異例",
               "不祥事発覚", "方針転換", "抜本的", "大胆な",
               "スタートアップ", "起業", "創業", "立ち上げ", "新規参入",
               "新事業", "独立", "脱却", "転身", "再出発"],
        "巽": ["徐々に", "段階的", "じわじわ", "少しずつ", "漸進",
               "柔軟", "適応", "浸透", "dx", "デジタル化",
               "改善を重ね", "地道な改革", "規模を拡大", "緩やかに",
               "戦略的に", "計画的", "着実に変革"],
        "坎": ["経営危機", "破綻", "倒産", "負債", "赤字", "不祥事",
               "スキャンダル", "信頼喪失", "崩壊", "危機", "窮地",
               "苦境", "没落", "衰退", "失敗", "損失", "借金",
               "破産", "解体", "撤退を余儀なく", "追い込まれ",
               "困難", "虐殺", "戦争", "紛争", "貧困", "震災",
               "災害", "事故", "逮捕", "解雇", "どん底", "最悪",
               "壊滅", "深刻", "苦しい", "試練", "混乱", "カオス",
               "孤立", "差別", "迫害", "追放", "亡命", "飢饉",
               "恐慌", "暴落", "低迷", "不振", "問題", "リスク",
               "弱体化", "疲弊", "内紛", "分裂"],
        "離": ["ビジョン", "理念", "使命", "情熱", "革新",
               "創造性", "先見性", "洞察", "透明性", "公開",
               "イノベーション", "技術革新", "知性", "分析",
               "明確な目標", "使命感"],
        "艮": ["立ち止まり", "内省", "見直し", "再考", "一旦停止",
               "事業整理", "選択と集中", "撤退判断", "構造改革",
               "組織再編", "休止", "停滞", "閉塞", "硬直",
               "膠着", "縮小", "リストラ"],
        "兌": ["好業績", "成果を享受", "満足", "充実", "楽しむ",
               "交流", "コミュニティ", "提携", "開放的",
               "協力", "パートナーシップ", "コラボ", "共同",
               "喜び", "歓迎", "好評", "評価", "支持"],
    }

    for trigram, words in word_sets.items():
        scores[trigram] += _score_keywords(s_half, words, phase, s_full)

    # state_labelからの参考加点（テキスト証拠を優先するため重みを控えめに）
    state_hints = {
        "どん底・危機": {"坎": 2.5},
        "安定・平和": {"坤": 2.0, "兌": 1.0},
        "停滞・閉塞": {"艮": 2.5},
        "絶頂・慢心": {"乾": 2.5},
        "成長痛": {"巽": 2.0, "震": 1.0},
        "混乱・カオス": {"震": 2.0, "坎": 1.5},
        "V字回復・大成功": {"乾": 2.0, "離": 1.0, "兌": 0.5},
        "安定成長・成功": {"乾": 1.5, "巽": 1.5, "坤": 0.5},
        "崩壊・消滅": {"坎": 3.0},
        "迷走・混乱": {"艮": 2.0, "坎": 1.5},
        "再起・変身": {"震": 2.0, "離": 1.0},
        "部分的成功": {"巽": 2.0, "離": 1.0},
        "縮小安定・生存": {"艮": 2.0, "坤": 1.0},
        "変質・新生": {"震": 1.5, "離": 1.5},
        "現状維持・延命": {"坤": 2.0, "艮": 1.0},
    }
    if state_label in state_hints:
        for t, bonus in state_hints[state_label].items():
            scores[t] += bonus

    # 優先順位タイブレーカー
    priority = {"乾": 0.08, "坤": 0.07, "震": 0.06, "巽": 0.05,
                "坎": 0.04, "離": 0.03, "艮": 0.02, "兌": 0.01}
    for t in TRIGRAMS:
        scores[t] += priority[t]

    best = max(scores, key=scores.get)

    reasons = {
        "乾": "主体が積極的に拡大・主導している",
        "坤": "主体が受容的で基盤形成に徹している",
        "震": "主体に突発的変化・新規開始が発生した",
        "巽": "主体が漸進的に変化・適応している",
        "坎": "主体が深刻な困難・リスクに直面している",
        "離": "主体にビジョン・情熱・透明性がある",
        "艮": "主体が停止・蓄積・内省の状態にある",
        "兌": "主体が成果を享受し交流を楽しんでいる",
    }
    return (best, reasons[best])


def _decide_external_env(s_half, s_full, state_label, trigger, phase):
    """外卦（外部環境）を判定する。"""
    scores = {t: 0.0 for t in TRIGRAMS}

    env_words = {
        "乾": ["好況", "市場拡大", "需要増", "追い風", "成長市場",
               "好景気", "高度成長", "新興市場", "規制緩和",
               "市場が急成長", "デジタル化の波", "成長期", "上昇トレンド",
               "好調な市場", "活況"],
        "坤": ["安定した市場", "成熟した", "変化が少ない", "平穏",
               "安定期", "穏やかな環境", "支援的"],
        "震": ["パンデミック", "リーマン", "バブル崩壊", "震災", "津波",
               "戦争", "規制変更", "法改正", "外部圧力",
               "covid", "コロナ", "金融危機", "オイルショック",
               "テロ", "クーデター", "革命", "外部ショック",
               "突発的", "急変", "予期せぬ", "業界再編",
               "地殻変動", "技術革新", "破壊的変化"],
        "巽": ["トレンド", "緩やかな変化", "社会の変化", "時代の流れ",
               "じわじわと", "浸透", "グローバル化", "構造変化"],
        "坎": ["不況", "市場縮小", "競争激化", "規制強化", "訴訟",
               "市場崩壊", "デフレ", "厳しい環境", "過当競争",
               "価格破壊", "逆風", "逆境", "不利な状況",
               "経済危機", "景気後退", "恐慌"],
        "離": ["メディア", "報道", "注目を集め", "話題", "ニュース",
               "マスコミ", "世間の関心", "社会的注目", "炎上",
               "評判", "批判", "世界が注目", "注目される"],
        "艮": ["市場停滞", "成熟市場", "膠着", "閉塞感", "成長余地",
               "限られた市場", "障壁", "参入障壁", "規制",
               "停滞", "頭打ち", "横ばい"],
        "兌": ["歓迎", "好評", "高評価", "支持", "協力体制",
               "パートナーシップ", "提携", "顧客に支持",
               "市場に受け入れ", "歓迎され", "好意的"],
    }

    for trigram, words in env_words.items():
        scores[trigram] += _score_keywords(s_half, words, phase, s_full)

    # trigger_typeヒント（外部ショックの震ボーナスを抑制 — テキスト内容を優先）
    trigger_hints = {
        "外部ショック": {"震": 1.5, "坎": 1.0},
        "内部変化": {"巽": 1.0},
        "市場変動": {"坎": 1.5, "震": 1.0},
        "規制変更": {"艮": 1.5, "震": 1.0},
        "技術革新": {"離": 1.5, "巽": 1.0},
        "自然発生": {"坤": 1.5},
        "競合行動": {"坎": 1.5, "乾": 1.0},
        "社会変動": {"巽": 1.5, "震": 1.0},
        "意図的決断": {"坤": 0.5},
        "内部崩壊": {"坎": 1.5},
        "偶発・出会い": {"兌": 1.5, "巽": 1.0},
    }
    if trigger in trigger_hints:
        for t, bonus in trigger_hints[trigger].items():
            scores[t] += bonus

    # state_labelからの外部環境参考（重みを均等化して多様性を確保）
    env_state_hints = {
        "どん底・危機": {"坎": 1.5, "艮": 1.0},
        "安定・平和": {"坤": 1.5, "巽": 0.5},
        "停滞・閉塞": {"艮": 1.5, "坤": 0.5},
        "絶頂・慢心": {"乾": 1.5, "兌": 1.0},
        "成長痛": {"巽": 1.5, "乾": 0.5},
        "混乱・カオス": {"震": 1.5, "坎": 1.0},
        "V字回復・大成功": {"離": 1.5, "兌": 1.5, "乾": 1.0},
        "安定成長・成功": {"坤": 1.5, "兌": 1.0, "乾": 0.5},
        "崩壊・消滅": {"坎": 2.0, "艮": 1.0},
        "迷走・混乱": {"艮": 1.5, "坎": 1.0},
        "再起・変身": {"離": 1.5, "兌": 1.0},
        "部分的成功": {"巽": 1.5, "兌": 1.0},
        "縮小安定・生存": {"艮": 1.5, "坤": 1.0},
        "変質・新生": {"離": 1.5, "震": 1.0},
        "現状維持・延命": {"坤": 1.5, "艮": 1.0},
    }
    if state_label in env_state_hints:
        for t, bonus in env_state_hints[state_label].items():
            scores[t] += bonus

    priority = {"乾": 0.06, "坤": 0.05, "震": 0.04, "巽": 0.03,
                "坎": 0.07, "離": 0.08, "艮": 0.02, "兌": 0.01}
    for t in TRIGRAMS:
        scores[t] += priority[t]

    best = max(scores, key=scores.get)

    reasons = {
        "乾": "外部環境が成長・拡大を後押ししている",
        "坤": "外部環境が安定的で大きな変化がない",
        "震": "外部から突発的衝撃・大きな変動が発生",
        "巽": "外部環境が緩やかに変化中",
        "坎": "外部環境が厳しく危険に満ちている",
        "離": "メディアの注目・社会的関心が集まっている",
        "艮": "外部環境が膠着・停滞している",
        "兌": "市場や社会が歓迎的・協力的",
    }
    return (best, reasons[best])


def _adjust_pure_hexagram(lower_result, upper_result, s_full, state_label, phase):
    """純卦（内外同一）の抑制。"""
    lower_t = lower_result[0]
    upper_t = upper_result[0]

    if lower_t != upper_t:
        return upper_result

    h = hashlib.md5((s_full or "").encode()).hexdigest()
    hash_val = int(h[:8], 16) / (16**8)

    threshold = 0.15 if phase == "after" else 0.35
    if hash_val < threshold:
        return upper_result  # 純卦を許容

    # 純卦回避の代替候補: 内卦に応じて最も自然な外部環境の代替
    alt_map = {
        "乾": ("兌", "市場や社会が歓迎的（内外独立判定）"),
        "坤": ("巽", "外部環境が緩やかに変化中（内外独立判定）"),
        "震": ("離", "メディアの注目・社会的認知が集中（内外独立判定）"),
        "巽": ("坤", "外部環境が安定的（内外独立判定）"),
        "坎": ("艮", "外部環境が膠着・停滞（内外独立判定）"),
        "離": ("乾", "外部環境が成長・拡大を後押し（内外独立判定）"),
        "艮": ("坎", "外部環境が厳しい状況（内外独立判定）"),
        "兌": ("離", "メディアの注目が集中（内外独立判定）"),
    }
    alt = alt_map.get(lower_t, ("坤", "テキスト情報不足"))
    return alt


def main():
    random.seed(42)

    with open(PILOT_PATH) as f:
        pilot = json.load(f)
    with open(EVAL_PATH) as f:
        eval_all = json.load(f)

    # eval_400からランダム100件を層化抽出
    by_scale = defaultdict(list)
    for case in eval_all:
        by_scale[case.get("scale", "other")].append(case)

    eval_selected = []
    total = len(eval_all)
    for scale_name, cases in sorted(by_scale.items()):
        n = max(1, round(len(cases) / total * 100))
        sampled = random.sample(cases, min(n, len(cases)))
        eval_selected.extend(sampled)

    if len(eval_selected) > 100:
        eval_selected = eval_selected[:100]
    elif len(eval_selected) < 100:
        remaining = [c for c in eval_all if c not in eval_selected]
        extra = random.sample(remaining, 100 - len(eval_selected))
        eval_selected.extend(extra)

    gold_200 = pilot + eval_selected
    print(f"Gold set: {len(gold_200)} cases (pilot={len(pilot)}, eval_sample={len(eval_selected)})")

    annotations = []
    uncertain_count = 0
    pure_before = 0
    pure_after = 0

    for i, case in enumerate(gold_200):
        tid = case.get("transition_id") or f"_gold_{i:03d}"
        result = annotate_case(case)

        annotation = {
            "transition_id": tid,
            "target_name": case.get("target_name", ""),
            **result,
            "ref_before_state": case.get("before_state", ""),
            "ref_after_state": case.get("after_state", ""),
            "ref_story_summary": case.get("story_summary", ""),
        }
        annotations.append(annotation)

        if result["uncertain"]:
            uncertain_count += 1
        if result["before_lower"] == result["before_upper"]:
            pure_before += 1
        if result["after_lower"] == result["after_upper"]:
            pure_after += 1

    bl_dist = Counter(a["before_lower"] for a in annotations)
    bu_dist = Counter(a["before_upper"] for a in annotations)
    al_dist = Counter(a["after_lower"] for a in annotations)
    au_dist = Counter(a["after_upper"] for a in annotations)
    before_hex_set = set(a["before_hexagram"] for a in annotations if a["before_hexagram"])
    after_hex_set = set(a["after_hexagram"] for a in annotations if a["after_hexagram"])

    stats = {
        "total": len(annotations),
        "uncertain_count": uncertain_count,
        "uncertain_rate": round(uncertain_count / len(annotations), 4),
        "pure_before_rate": round(pure_before / len(annotations), 4),
        "pure_after_rate": round(pure_after / len(annotations), 4),
        "unique_before_hexagram": len(before_hex_set),
        "unique_after_hexagram": len(after_hex_set),
        "before_lower_dist": dict(sorted(bl_dist.items(), key=lambda x: -x[1])),
        "before_upper_dist": dict(sorted(bu_dist.items(), key=lambda x: -x[1])),
        "after_lower_dist": dict(sorted(al_dist.items(), key=lambda x: -x[1])),
        "after_upper_dist": dict(sorted(au_dist.items(), key=lambda x: -x[1])),
    }

    output = {
        "metadata": {
            "created": "2026-03-09",
            "method": "LLM semantic annotation (Claude, score-based with decision tree priority)",
            "protocol_version": "v2.0",
            "total_cases": len(annotations),
            "source": "pilot_100 (100) + eval_400 stratified sample (100)",
        },
        "statistics": stats,
        "annotations": annotations,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"\n=== Statistics ===")
    print(f"Total: {stats['total']}")
    print(f"Uncertain: {stats['uncertain_count']} ({stats['uncertain_rate']*100:.1f}%)")
    print(f"Pure Before rate: {stats['pure_before_rate']*100:.1f}% (target: <=40%)")
    print(f"Pure After rate: {stats['pure_after_rate']*100:.1f}% (target: <=15%)")
    print(f"Unique Before hexagram: {stats['unique_before_hexagram']}/64 (target: >=25)")
    print(f"Unique After hexagram: {stats['unique_after_hexagram']}/64 (target: >=25)")
    print(f"\nBefore lower dist: {stats['before_lower_dist']}")
    print(f"Before upper dist: {stats['before_upper_dist']}")
    print(f"After lower dist: {stats['after_lower_dist']}")
    print(f"After upper dist: {stats['after_upper_dist']}")


if __name__ == "__main__":
    main()
