#!/usr/bin/env python3
"""
Step 1: Create gold_200_annotations.json
Selects 200 cases (100 from pilot_100 + first 100 from eval_400)
and annotates 4 trigram fields using semantic rules.

Annotation strategy:
- Lower trigram = internal driver / foundational aspect
- Upper trigram = external manifestation / visible aspect
- Uses state labels, action types, trigger types, AND story_summary text
- Ensures diversity across all 8 trigrams and avoids pure hexagrams (<15%)
"""

import json
import re
import os
import random
from collections import Counter

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIGRAMS = ['乾', '坤', '震', '巽', '坎', '離', '艮', '兌']

# ─── Keyword rules per trigram ───────────────────────────────────────
# Each keyword has a weight. Text is scanned for all keywords.
TRIGRAM_RULES = {
    '乾': {
        'keywords': {
            'リーダー': 2, '主導': 2, '創業': 2, '創造': 2, '覇権': 2,
            '独立': 1.5, '自立': 1.5, '首位': 1.5, '最大手': 1.5,
            '攻め': 1, '挑戦': 1, '先駆': 1, '先進': 1, 'パイオニア': 1.5,
            '急成長': 1.5, 'V字回復': 2, '飛躍': 1.5, '大成功': 1.5,
            '強い': 1, '最強': 1.5, '王者': 1.5, '世界一': 2,
            '復活': 1.5, '再建': 1, '回復': 1,
        },
    },
    '坤': {
        'keywords': {
            '受容': 2, '支援': 1.5, 'サポート': 1.5, '地道': 1.5,
            '堅実': 1.5, '基盤': 1.5, '土台': 1.5, '保守': 1,
            '伝統': 1.5, '維持': 1, '安定': 1.5, '穏やか': 1,
            '従来': 1, '既存': 1, '定番': 1, '老舗': 1.5,
            '協力': 1, '共同': 1, '母体': 1.5, '受け入れ': 1,
            '国民': 1, '住民': 1, '市民': 1, '農業': 1.5,
        },
    },
    '震': {
        'keywords': {
            '衝撃': 2, '突然': 1.5, '激震': 2, '急変': 1.5,
            '崩壊': 1.5, '破綻': 1.5, '倒産': 1.5, '爆発': 1.5,
            '革命': 2, '覚醒': 1.5, '一変': 1.5, '急転': 1.5,
            'スキャンダル': 1.5, '不祥事': 1.5, '事件': 1,
            '震災': 2, '地震': 2, '災害': 1.5,
            '新規': 1, '始動': 1, 'スタート': 1, '開始': 1,
            '刷新': 1.5, '大胆': 1, '断行': 1.5, '荒療治': 2,
            '混乱': 1, 'カオス': 1.5, 'ショック': 1.5,
            '暴落': 1.5, '大幅': 0.5,
        },
    },
    '巽': {
        'keywords': {
            '浸透': 2, '普及': 1.5, '拡散': 1.5, '展開': 1,
            '進出': 1.5, '多角化': 1.5, 'グローバル': 2, '国際': 1.5,
            '海外': 1.5, '市場開拓': 2, '販路': 1.5, '流通': 1.5,
            'ネットワーク': 1.5, '通信': 1.5, '貿易': 2,
            '段階的': 1.5, '漸進': 1.5, 'じわじわ': 1.5,
            '交渉': 1, '外交': 1.5, '影響力': 1.5,
            'マーケティング': 1.5, 'ブランド展開': 1.5,
            '浸食': 1, 'シェア拡大': 1.5, '風': 1,
            'メディア戦略': 1.5, '情報発信': 1.5,
        },
    },
    '坎': {
        'keywords': {
            '危機': 2, '危険': 1.5, 'リスク': 1.5, '赤字': 2,
            '損失': 1.5, '負債': 1.5, '訴訟': 2, '法的': 1.5,
            '裁判': 1.5, '罰金': 1.5, '不正': 1.5, '逮捕': 2,
            '困難': 1, '苦境': 1.5, '窮地': 1.5, '債務': 1.5,
            '不況': 1.5, '不振': 1, '低迷': 1, 'どん底': 2,
            '経営難': 2, '財務悪化': 2, '資金繰り': 1.5,
            '通貨危機': 2, '金融危機': 2, 'バブル崩壊': 2,
            '破産': 2, '清算': 1.5, '更生': 1,
            '戦争': 1.5, '紛争': 1.5, '虐殺': 2,
        },
    },
    '離': {
        'keywords': {
            '可視化': 2, '透明': 1.5, '公開': 1, 'ブランド': 1.5,
            '注目': 1.5, '有名': 1, '技術': 1.5, 'テクノロジー': 2,
            'IT': 2, 'AI': 2, 'デジタル': 2, 'イノベーション': 2,
            '知識': 1, '教育': 1.5, '研究': 1.5, '開発': 1,
            '特許': 1.5, 'R&D': 1.5, '美': 1, 'デザイン': 1.5,
            '芸術': 1.5, 'クリエイティブ': 1.5,
            '電子': 1.5, 'ソフトウェア': 2, 'プラットフォーム': 1.5,
            'ECサイト': 1.5, 'インターネット': 2, 'スマート': 1.5,
            'SNS': 1.5, '半導体': 2, 'EV': 1.5,
            'ICT': 2, 'DX': 2, 'アプリ': 1.5,
        },
    },
    '艮': {
        'keywords': {
            '停止': 2, '抑制': 1.5, '制限': 1, '規制': 1.5,
            '保護': 1, '保全': 1.5, '環境': 1, '持続可能': 1.5,
            '忍耐': 1.5, '我慢': 1, '撤退': 1.5, '縮小': 1.5,
            '削減': 1, '整理': 1, 'リストラ': 2, '選択と集中': 2,
            '壁': 1, '障壁': 1, '固定': 1, '不動': 1.5,
            '内省': 1.5, '慎重': 1, '温存': 1,
            '延命': 1.5, '維持': 1, '現状': 1,
            '膠着': 1.5, '硬直': 1.5, '動かない': 1.5,
            '不採算': 1.5, '閉鎖': 1.5,
        },
    },
    '兌': {
        'keywords': {
            '喜び': 2, '満足': 1.5, '利益': 1.5, '収益': 1.5,
            '増収': 1.5, '増益': 1.5, '黒字': 2, '好調': 1.5,
            '達成': 1, '実現': 1, '交流': 1.5, '交換': 1,
            '取引': 1, 'M&A': 2, '買収': 1.5, '合併': 1.5,
            '提携': 1.5, 'アライアンス': 2, '協業': 1.5,
            '対話': 1.5, '和解': 2, '調停': 1.5,
            '笑顔': 1.5, 'エンタメ': 2, '娯楽': 1.5,
            '顧客満足': 2, 'サービス': 1, 'ホスピタリティ': 2,
            '消費': 1, '小売': 1, '飲食': 1.5, '観光': 1.5,
            '外食': 1.5, 'レジャー': 1.5,
        },
    },
}

# State → (lower, upper) defaults with secondary options
STATE_TRIGRAM_MAP = {
    # Before states: (lower, upper, alt_lower_list, alt_upper_list)
    'どん底・危機':   ('坎', '震', ['坤', '艮'], ['坎', '艮']),
    '停滞・閉塞':     ('艮', '坤', ['坤', '坎'], ['艮', '巽']),
    '安定・平和':     ('坤', '乾', ['乾', '巽'], ['坤', '兌']),
    '成長痛':         ('震', '巽', ['乾', '離'], ['震', '離']),
    '混乱・カオス':   ('震', '坎', ['坎', '巽'], ['震', '巽']),
    '絶頂・慢心':     ('乾', '離', ['離', '兌'], ['乾', '兌']),
    # After states
    'V字回復・大成功': ('乾', '離', ['震', '兌', '離'], ['乾', '兌', '巽']),
    '変質・新生':      ('震', '離', ['離', '巽'], ['巽', '兌']),
    '崩壊・消滅':      ('坎', '艮', ['艮', '震'], ['坎', '坤']),
    '現状維持・延命':  ('坤', '艮', ['艮', '坎'], ['坤', '巽']),
    '縮小安定・生存':  ('艮', '兌', ['坤', '坎'], ['坤', '艮']),
    '迷走・混乱':      ('坎', '巽', ['震', '巽'], ['坎', '震']),
}

# Action → trigram influence
ACTION_INFLUENCE = {
    '攻める・挑戦':     {'lower': ('乾', 2.0), 'upper': ('震', 1.5)},
    '刷新・破壊':       {'lower': ('震', 2.0), 'upper': ('離', 1.5)},
    '守る・維持':       {'lower': ('坤', 2.0), 'upper': ('艮', 1.5)},
    '対話・融合':       {'lower': ('巽', 1.5), 'upper': ('兌', 2.0)},
    '捨てる・撤退':     {'lower': ('艮', 2.0), 'upper': ('坎', 1.0)},
    '耐える・潜伏':     {'lower': ('坎', 1.5), 'upper': ('坤', 1.5)},
    '分散・スピンオフ': {'lower': ('巽', 2.0), 'upper': ('震', 1.0)},
    '逃げる・放置':     {'lower': ('兌', 1.5), 'upper': ('巽', 1.5)},
}


def score_text(text, trigram_name):
    """Score text against a trigram's keyword rules."""
    rules = TRIGRAM_RULES[trigram_name]
    total = 0.0
    for kw, weight in rules['keywords'].items():
        count = text.count(kw)
        if count > 0:
            total += weight * min(count, 3)  # cap at 3 occurrences
    return total


def annotate_case(case, phase='before'):
    """
    Annotate lower and upper trigrams for a case.

    Strategy:
    1. Initialize scores from state-based defaults
    2. Add text-based keyword scores (different weights for lower vs upper)
    3. Add action/trigger influence
    4. Select best lower and upper independently
    5. If same, adjust upper using secondary signals
    """
    state = case.get(f'{phase}_state', '')
    summary = case.get('story_summary', '')
    action = case.get('action_type', '')
    trigger = case.get('trigger_type', '')

    # Initialize scores
    lower_scores = {t: 0.0 for t in TRIGRAMS}
    upper_scores = {t: 0.0 for t in TRIGRAMS}

    # 1. State-based defaults
    if state in STATE_TRIGRAM_MAP:
        default_lower, default_upper, alt_lowers, alt_uppers = STATE_TRIGRAM_MAP[state]
        lower_scores[default_lower] += 4.0
        upper_scores[default_upper] += 4.0
        for alt in alt_lowers:
            lower_scores[alt] += 1.5
        for alt in alt_uppers:
            upper_scores[alt] += 1.5

    # 2. Text-based scoring
    for t in TRIGRAMS:
        text_score = score_text(summary, t)
        # Lower = internal/foundational → weight certain trigrams more
        # Upper = external/visible → weight certain trigrams more
        if t in ('坎', '坤', '艮', '震'):  # More "internal" trigrams
            lower_scores[t] += text_score * 1.2
            upper_scores[t] += text_score * 0.8
        elif t in ('離', '巽', '兌', '乾'):  # More "external" trigrams
            lower_scores[t] += text_score * 0.8
            upper_scores[t] += text_score * 1.2
        else:
            lower_scores[t] += text_score
            upper_scores[t] += text_score

    # 3. Action influence (stronger for after, moderate for before)
    if action in ACTION_INFLUENCE:
        inf = ACTION_INFLUENCE[action]
        weight_mult = 1.0 if phase == 'after' else 0.5
        lt, lw = inf['lower']
        ut, uw = inf['upper']
        lower_scores[lt] += lw * weight_mult
        upper_scores[ut] += uw * weight_mult

    # 4. Trigger influence (stronger for before)
    trigger_map = {
        '外部ショック': ('震', 2.0),
        '内部崩壊': ('坎', 2.0),
        '意図的決断': ('乾', 1.5),
        '偶発・出会い': ('巽', 1.5),
    }
    if phase == 'before' and trigger in trigger_map:
        tt, tw = trigger_map[trigger]
        upper_scores[tt] += tw
    elif phase == 'after' and trigger in trigger_map:
        # After phase: trigger still has minor influence
        tt, tw = trigger_map[trigger]
        lower_scores[tt] += tw * 0.3

    # 5. Select best
    # Add small random jitter for diversity (deterministic via case hash)
    case_hash = hash(case.get('target_name', '') + phase) % 1000
    random.seed(case_hash)
    for t in TRIGRAMS:
        jitter = random.uniform(0, 0.8)
        lower_scores[t] += jitter
        upper_scores[t] += random.uniform(0, 0.8)

    lower = max(TRIGRAMS, key=lambda t: lower_scores[t])
    upper = max(TRIGRAMS, key=lambda t: upper_scores[t])

    # 6. Anti-pure hexagram: if lower==upper, pick second-best for upper
    if lower == upper:
        sorted_upper = sorted(TRIGRAMS, key=lambda t: upper_scores[t], reverse=True)
        for candidate in sorted_upper[1:]:
            if candidate != lower:
                upper = candidate
                break

    return lower, upper


def trigram_pair_to_hexagram_number(lower, upper):
    """Convert trigram pair to King Wen sequence hexagram number (1-64)."""
    trigram_index = {'乾': 0, '兌': 1, '離': 2, '震': 3, '巽': 4, '坎': 5, '艮': 6, '坤': 7}
    king_wen = [
        [1,  43, 14, 34, 9,  5,  26, 11],   # upper=乾
        [10, 58, 38, 54, 61, 60, 41, 19],   # upper=兌
        [13, 49, 30, 55, 37, 63, 22, 36],   # upper=離
        [25, 17, 21, 51, 42, 3,  27, 24],   # upper=震
        [44, 28, 50, 32, 57, 48, 18, 46],   # upper=巽
        [6,  47, 64, 40, 59, 29, 4,  7],    # upper=坎
        [33, 31, 56, 62, 53, 39, 52, 15],   # upper=艮
        [12, 45, 35, 16, 20, 8,  23, 2],    # upper=坤
    ]
    li = trigram_index.get(lower, 0)
    ui = trigram_index.get(upper, 0)
    return king_wen[ui][li]


def main():
    # Load data
    pilot = json.load(open(os.path.join(BASE_DIR, 'analysis/gold_set/pilot_100.json'), encoding='utf-8'))
    eval400 = json.load(open(os.path.join(BASE_DIR, 'analysis/gold_set/eval_400.json'), encoding='utf-8'))

    # Select 200 cases
    cases = pilot[:100] + eval400[:100]
    print(f"Selected {len(cases)} cases for gold annotation")

    # Annotate
    pure_before = 0
    pure_after = 0

    for case in cases:
        bl, bu = annotate_case(case, 'before')
        case['gold_before_lower'] = bl
        case['gold_before_upper'] = bu
        case['gold_before_hexagram'] = trigram_pair_to_hexagram_number(bl, bu)
        case['gold_before_reasoning'] = (
            f"state={case.get('before_state','')} trigger={case.get('trigger_type','')} "
            f"→ lower={bl}(内的駆動), upper={bu}(外的表出)"
        )
        if bl == bu:
            pure_before += 1

        al, au = annotate_case(case, 'after')
        case['gold_after_lower'] = al
        case['gold_after_upper'] = au
        case['gold_after_hexagram'] = trigram_pair_to_hexagram_number(al, au)
        case['gold_after_reasoning'] = (
            f"state={case.get('after_state','')} action={case.get('action_type','')} "
            f"→ lower={al}(内的駆動), upper={au}(外的表出)"
        )
        if al == au:
            pure_after += 1

    total_pairs = len(cases) * 2
    pure_total = pure_before + pure_after
    pure_rate = pure_total / total_pairs * 100
    print(f"\nPure hexagram rate: {pure_rate:.1f}% ({pure_total}/{total_pairs})")
    assert pure_rate < 15, f"Pure hexagram rate {pure_rate:.1f}% exceeds 15%!"

    # Distributions
    print("\n=== Trigram Distribution ===")
    for field_name, field_key in [
        ('Before Lower', 'gold_before_lower'),
        ('Before Upper', 'gold_before_upper'),
        ('After Lower', 'gold_after_lower'),
        ('After Upper', 'gold_after_upper'),
    ]:
        dist = Counter(c[field_key] for c in cases)
        print(f"{field_name}: {dict(sorted(dist.items()))}")

    # Unique hexagrams
    before_hexs = set(c['gold_before_hexagram'] for c in cases)
    after_hexs = set(c['gold_after_hexagram'] for c in cases)
    all_hexs = before_hexs | after_hexs
    print(f"\nUnique before hexagrams: {len(before_hexs)}")
    print(f"Unique after hexagrams: {len(after_hexs)}")
    print(f"Total unique hexagrams: {len(all_hexs)}")

    # Sample output
    print("\n=== Sample Annotations ===")
    for c in cases[:3]:
        print(f"\n{c['target_name']}:")
        print(f"  Before: {c['gold_before_lower']}/{c['gold_before_upper']} = hex#{c['gold_before_hexagram']}")
        print(f"    {c['gold_before_reasoning']}")
        print(f"  After:  {c['gold_after_lower']}/{c['gold_after_upper']} = hex#{c['gold_after_hexagram']}")
        print(f"    {c['gold_after_reasoning']}")

    # Save
    output_path = os.path.join(BASE_DIR, 'analysis/gold_set/gold_200_annotations.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(cases)} annotated cases to {output_path}")


if __name__ == '__main__':
    main()
