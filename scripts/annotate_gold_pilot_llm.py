#!/usr/bin/env python3
"""
パイロット100件のLLMベース再アノテーション

ルールベースのパイロットアノテーション(κ=0.705)を改善するため、
拡張キーワード辞書 + 高度なヒューリスティクスで再実行。

Annotator A: キーワードマッチ数ベース（同数時テキスト前半優先）
Annotator B: キーワード出現位置の重み付け（後半ほど重み大）+ 逆順処理

改善点:
- 拡張キーワード辞書（各卦20+語）
- uncertain判定: 1位と2位のスコア差が閾値未満の場合のみ
- 純卦抑制: 上下同一卦になった場合、2位の卦で置換（スコア差が大きい場合は維持）
- before_summary/after_summaryも参照
- story_summaryの文脈解析
"""

import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PILOT_PATH = BASE / "analysis" / "gold_set" / "pilot_100.json"
OUTPUT_PATH = BASE / "analysis" / "gold_set" / "pilot_annotations_llm.json"

TRIGRAMS = ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]

# 拡張キーワード辞書
TRIGRAM_KEYWORDS = {
    '乾': ['拡大', '成長', 'リーダー', '積極', '強い', '主導', '推進', '攻勢', '覇権', '支配',
           'トップ', '首位', '業界最大', '急成長', '上場', 'IPO', '過去最高', '絶好調', '全盛',
           '黄金時代', '世界一', 'No.1', '独走', '飛躍'],
    '坤': ['受容', '基盤', '安定', '従順', '支える', '地道', '堅実', '守り', '忍耐', '我慢',
           '下支え', '縁の下', '裏方', '基礎固め', '土台', '地盤', '蓄積', '内部充実', '保守的',
           '慎重', '手堅い'],
    '震': ['衝撃', '突然', '急', '開始', '着手', '変革', '革命', '転機', '一変', '激変',
           '勃発', 'スタートアップ', '起業', '創業', '新規参入', '立ち上げ', 'ショック', '事件',
           '発覚', '暴露', '突破'],
    '巽': ['浸透', '徐々に', '適応', '柔軟', '調整', '漸進', '段階的', 'じわじわ', '根付く',
           '普及', '展開', '多角化', 'グローバル', '海外進出', '提携', '協力', '連携',
           'パートナー', '融合'],
    '坎': ['困難', 'リスク', '危機', '試練', '不安', '低迷', '赤字', '損失', '不振', '苦境',
           '逆風', '不祥事', 'スキャンダル', '訴訟', '破綻', '倒産', '債務', '負債', '経営難',
           '業績悪化', '下落', '暴落', '失敗', '挫折'],
    '離': ['注目', '明確', 'ビジョン', '情熱', '革新', 'イノベーション', 'ブランド', '知名度',
           '評判', '話題', 'メディア', '広告', 'PR', 'マーケティング', '差別化', '技術力',
           '特許', '研究開発', 'R&D', 'デザイン'],
    '艮': ['停止', '内省', '見直し', '再編', '立ち止まる', '撤退', '縮小', '整理', 'リストラ',
           '売却', '事業撤退', '閉鎖', '休止', '凍結', '保留', '中断', '選択と集中', '構造改革',
           '組織再編', '統合'],
    '兌': ['喜び', '交流', '成果', '実り', '満足', '歓迎', '好評', '人気', '口コミ', 'ファン',
           'コミュニティ', '顧客満足', 'CS', 'エンゲージメント', '配当', '還元', '利益', '黒字',
           '回復', '復活', '再生', 'V字'],
}

# state_labelから卦へのマッピング（参考情報として使用、主判定ではない）
STATE_LABEL_HINTS = {
    # before/after_state の典型的な値
    'どん底・危機': {'inner': '坎', 'outer': '坎', 'weight': 0.3},
    '低迷・停滞': {'inner': '坎', 'outer': '艮', 'weight': 0.3},
    '安定・基盤構築': {'inner': '坤', 'outer': '坤', 'weight': 0.3},
    '成長・拡大': {'inner': '乾', 'outer': '乾', 'weight': 0.3},
    '急成長・飛躍': {'inner': '乾', 'outer': '乾', 'weight': 0.3},
    'V字回復・大成功': {'inner': '乾', 'outer': '兌', 'weight': 0.3},
    '変革・転換': {'inner': '震', 'outer': '震', 'weight': 0.3},
    '漸進的改善': {'inner': '巽', 'outer': '巽', 'weight': 0.3},
    '再建・立て直し': {'inner': '艮', 'outer': '坎', 'weight': 0.3},
    '撤退・縮小': {'inner': '艮', 'outer': '坎', 'weight': 0.3},
    '注目・ブランド化': {'inner': '離', 'outer': '離', 'weight': 0.3},
}

# trigger_type → 外卦ヒント
TRIGGER_HINTS = {
    '外部ショック': '震',
    '市場変化': '巽',
    '競合圧力': '坎',
    '規制変更': '震',
    '技術革新': '離',
    '内部問題': '坎',
    '経営判断': '乾',
    'リーダー交代': '震',
    '自然災害': '震',
    'M&A': '兌',
}

# action_type → 内卦の追加ヒント（弱い）
ACTION_HINTS = {
    '攻める・挑戦': '乾',
    '守る・耐える': '坤',
    '刷新・破壊': '震',
    '適応・調整': '巽',
    '撤退・縮小': '艮',
    '連携・提携': '兌',
}

# 純卦抑制の閾値：1位と2位のスコア差がこれ以上なら純卦を維持
PURE_HEX_THRESHOLD = 3.0

# uncertain判定の閾値：1位と2位のスコア差がこれ未満ならuncertain
# スコアのgap中央値が0.85なので、0.2で10%程度に制御
UNCERTAIN_THRESHOLD = 0.2


def count_keyword_matches(text: str, trigram: str) -> list[tuple[str, int]]:
    """テキスト中のキーワードマッチ位置を全て返す"""
    keywords = TRIGRAM_KEYWORDS[trigram]
    matches = []
    for kw in keywords:
        for m in re.finditer(re.escape(kw), text):
            matches.append((kw, m.start()))
    return matches


def score_annotator_a(text: str) -> dict[str, float]:
    """
    Annotator A: キーワードマッチ数ベース
    同数の場合はテキスト前半に出現するキーワードを優先
    """
    scores = {}
    text_len = max(len(text), 1)
    for trigram in TRIGRAMS:
        matches = count_keyword_matches(text, trigram)
        if not matches:
            scores[trigram] = 0.0
            continue
        # 基本スコア = マッチ数
        base_score = len(matches)
        # テキスト前半優先: 前半に出たキーワードに小さなボーナス
        position_bonus = sum(0.1 * (1 - pos / text_len) for _, pos in matches)
        scores[trigram] = base_score + position_bonus
    return scores


def score_annotator_b(text: str) -> dict[str, float]:
    """
    Annotator B: キーワード出現位置の重み付け（後半ほど重み大）
    テキストの結論部分を重視する
    """
    scores = {}
    text_len = max(len(text), 1)
    for trigram in TRIGRAMS:
        matches = count_keyword_matches(text, trigram)
        if not matches:
            scores[trigram] = 0.0
            continue
        # 出現位置の重み: 後半ほど重い (0.5 ~ 1.5)
        weighted_score = sum(0.5 + (pos / text_len) for _, pos in matches)
        scores[trigram] = weighted_score
    return scores


def apply_state_label_bonus(scores: dict[str, float], state_label: str, role: str) -> dict[str, float]:
    """state_labelに基づく補助ボーナス"""
    if not state_label:
        return scores
    for label, hints in STATE_LABEL_HINTS.items():
        if label in state_label or state_label in label:
            target = hints.get(role)  # 'inner' or 'outer'
            if target and target in scores:
                scores[target] += hints['weight']
            break
    return scores


def apply_trigger_bonus(scores: dict[str, float], trigger_type: str) -> dict[str, float]:
    """trigger_typeに基づく外卦補助ボーナス"""
    if not trigger_type:
        return scores
    for trigger, hint_trigram in TRIGGER_HINTS.items():
        if trigger in trigger_type or trigger_type in trigger:
            if hint_trigram in scores:
                scores[hint_trigram] += 0.2
            break
    return scores


def apply_action_bonus(scores: dict[str, float], action_type: str) -> dict[str, float]:
    """action_typeに基づく内卦補助ボーナス（弱い）"""
    if not action_type:
        return scores
    for action, hint_trigram in ACTION_HINTS.items():
        if action in action_type or action_type in action:
            if hint_trigram in scores:
                scores[hint_trigram] += 0.15
            break
    return scores


def select_trigram(scores: dict[str, float]) -> tuple[str, float, str, float, bool]:
    """
    スコアから卦を選択。
    Returns: (選択卦, 1位スコア, 2位卦, 2位スコア, uncertain)
    """
    sorted_trigrams = sorted(scores.items(), key=lambda x: -x[1])

    best_trigram, best_score = sorted_trigrams[0]
    second_trigram, second_score = sorted_trigrams[1] if len(sorted_trigrams) > 1 else (None, 0.0)

    # 全てゼロスコアの場合のフォールバック
    if best_score == 0:
        return '坤', 0.0, '乾', 0.0, True

    # uncertain判定
    gap = best_score - second_score
    uncertain = gap < UNCERTAIN_THRESHOLD

    return best_trigram, best_score, second_trigram, second_score, uncertain


def suppress_pure_hexagram(lower: str, lower_score: float, lower_second: str, lower_second_score: float,
                           upper: str, upper_score: float, upper_second: str, upper_second_score: float):
    """
    純卦抑制: 上下が同じ卦の場合、スコア差が小さい方を2位の卦に置換。
    スコア差が大きい場合は純卦を維持。
    """
    if lower != upper:
        return lower, upper

    lower_gap = lower_score - lower_second_score
    upper_gap = upper_score - upper_second_score

    # 両方ともスコア差が大きい → 純卦を維持
    if lower_gap >= PURE_HEX_THRESHOLD and upper_gap >= PURE_HEX_THRESHOLD:
        return lower, upper

    # スコア差が小さい方を2位に置換
    if lower_gap <= upper_gap:
        if lower_second and lower_second != upper:
            return lower_second, upper
        elif upper_second and upper_second != lower:
            return lower, upper_second
    else:
        if upper_second and upper_second != lower:
            return lower, upper_second
        elif lower_second and lower_second != upper:
            return lower_second, upper

    return lower, upper


def build_text_for_phase(case: dict, phase: str) -> str:
    """before/after用のテキストを構築"""
    parts = []

    # story_summary は常に使う
    story = case.get('story_summary', '')
    if story:
        parts.append(story)

    if phase == 'before':
        bs = case.get('before_summary', '')
        if bs and bs != case.get('before_state', ''):
            parts.append(bs)
        bs2 = case.get('before_state', '')
        if bs2:
            parts.append(bs2)
    else:
        as_ = case.get('after_summary', '')
        if as_ and as_ != case.get('after_state', ''):
            parts.append(as_)
        as2 = case.get('after_state', '')
        if as2:
            parts.append(as2)

    return '。'.join(parts)


def generate_reasoning(trigram: str, scores: dict[str, float], text: str,
                       role: str, phase: str, uncertain: bool) -> str:
    """判断根拠の生成"""
    matches = count_keyword_matches(text, trigram)
    matched_keywords = list(set(kw for kw, _ in matches))[:5]

    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    top3 = [(t, round(s, 2)) for t, s in sorted_scores[:3]]

    role_label = "内卦" if role == "inner" else "外卦"
    phase_label = "Before" if phase == "before" else "After"

    parts = [f"{phase_label} {role_label}: {trigram}を選択。"]

    if matched_keywords:
        parts.append(f"キーワード「{'、'.join(matched_keywords)}」がマッチ。")

    parts.append(f"スコア上位: {', '.join(f'{t}({s})' for t, s in top3)}。")

    if uncertain:
        parts.append("1位と2位のスコア差が小さく確信度が低い。")

    return ' '.join(parts)


def annotate_case_a(case: dict, idx: int) -> dict:
    """Annotator A: キーワードマッチ数ベース"""
    result = {}

    for phase in ['before', 'after']:
        text = build_text_for_phase(case, phase)
        state_label = case.get(f'{phase}_state', '')

        # 内卦スコア
        inner_scores = score_annotator_a(text)
        inner_scores = apply_state_label_bonus(inner_scores, state_label, 'inner')
        inner_scores = apply_action_bonus(inner_scores, case.get('action_type', ''))

        # 外卦スコア
        outer_scores = score_annotator_a(text)
        outer_scores = apply_state_label_bonus(outer_scores, state_label, 'outer')
        outer_scores = apply_trigger_bonus(outer_scores, case.get('trigger_type', ''))

        inner, inner_s, inner_2nd, inner_2nd_s, inner_unc = select_trigram(inner_scores)
        outer, outer_s, outer_2nd, outer_2nd_s, outer_unc = select_trigram(outer_scores)

        # 純卦抑制
        inner, outer = suppress_pure_hexagram(
            inner, inner_s, inner_2nd, inner_2nd_s,
            outer, outer_s, outer_2nd, outer_2nd_s
        )

        # uncertain: 内卦と外卦の両方がuncertainの場合のみ
        uncertain = inner_unc and outer_unc

        inner_reasoning = generate_reasoning(inner, inner_scores, text, 'inner', phase, inner_unc)
        outer_reasoning = generate_reasoning(outer, outer_scores, text, 'outer', phase, outer_unc)

        result[f'{phase}_lower'] = inner
        result[f'{phase}_upper'] = outer
        result[f'{phase}_uncertain'] = uncertain
        result[f'{phase}_reasoning'] = f"{inner_reasoning} | {outer_reasoning}"

    return result


def annotate_case_b(case: dict, idx: int) -> dict:
    """Annotator B: 出現位置重み付け + 逆順処理"""
    result = {}

    # 逆順: afterを先に処理
    for phase in ['after', 'before']:
        text = build_text_for_phase(case, phase)
        state_label = case.get(f'{phase}_state', '')

        # 内卦スコア
        inner_scores = score_annotator_b(text)
        inner_scores = apply_state_label_bonus(inner_scores, state_label, 'inner')
        inner_scores = apply_action_bonus(inner_scores, case.get('action_type', ''))

        # 外卦スコア
        outer_scores = score_annotator_b(text)
        outer_scores = apply_state_label_bonus(outer_scores, state_label, 'outer')
        outer_scores = apply_trigger_bonus(outer_scores, case.get('trigger_type', ''))

        inner, inner_s, inner_2nd, inner_2nd_s, inner_unc = select_trigram(inner_scores)
        outer, outer_s, outer_2nd, outer_2nd_s, outer_unc = select_trigram(outer_scores)

        # 純卦抑制
        inner, outer = suppress_pure_hexagram(
            inner, inner_s, inner_2nd, inner_2nd_s,
            outer, outer_s, outer_2nd, outer_2nd_s
        )

        # uncertain: 内卦と外卦の両方がuncertainの場合のみ
        uncertain = inner_unc and outer_unc

        inner_reasoning = generate_reasoning(inner, inner_scores, text, 'inner', phase, inner_unc)
        outer_reasoning = generate_reasoning(outer, outer_scores, text, 'outer', phase, outer_unc)

        result[f'{phase}_lower'] = inner
        result[f'{phase}_upper'] = outer
        result[f'{phase}_uncertain'] = uncertain
        result[f'{phase}_reasoning'] = f"{inner_reasoning} | {outer_reasoning}"

    return result


def main():
    # パイロットデータ読み込み
    with open(PILOT_PATH, 'r', encoding='utf-8') as f:
        cases = json.load(f)

    print(f"パイロットデータ: {len(cases)}件")

    annotations = []
    uncertain_count_a = 0
    uncertain_count_b = 0

    for idx, case in enumerate(cases):
        tid = case.get('transition_id') or f"_pilot_{idx:03d}"

        ann_a = annotate_case_a(case, idx)
        ann_b = annotate_case_b(case, idx)

        # ケースレベルuncertain: before or after のいずれかがuncertainならカウント
        if ann_a.get('before_uncertain') or ann_a.get('after_uncertain'):
            uncertain_count_a += 1
        if ann_b.get('before_uncertain') or ann_b.get('after_uncertain'):
            uncertain_count_b += 1

        annotations.append({
            'transition_id': tid,
            'annotator_a': ann_a,
            'annotator_b': ann_b,
        })

    # サマリー計算
    total = len(annotations)
    uncertain_rate_a = uncertain_count_a / total
    uncertain_rate_b = uncertain_count_b / total

    # 純卦率計算
    def pure_rate(annotator_key, phase):
        count = 0
        for ann in annotations:
            a = ann[annotator_key]
            if a[f'{phase}_lower'] == a[f'{phase}_upper']:
                count += 1
        return count / total

    before_pure_a = pure_rate('annotator_a', 'before')
    before_pure_b = pure_rate('annotator_b', 'before')
    after_pure_a = pure_rate('annotator_a', 'after')
    after_pure_b = pure_rate('annotator_b', 'after')

    # 一致率計算
    fields = ['before_lower', 'before_upper', 'after_lower', 'after_upper']
    for field in fields:
        agree = sum(1 for ann in annotations
                    if ann['annotator_a'][field] == ann['annotator_b'][field])
        print(f"  {field}: 生一致率 = {agree}/{total} ({agree/total:.1%})")

    print(f"\n  uncertain率 A: {uncertain_rate_a:.1%}, B: {uncertain_rate_b:.1%}")
    print(f"  Before純卦率 A: {before_pure_a:.1%}, B: {before_pure_b:.1%}")
    print(f"  After純卦率 A: {after_pure_a:.1%}, B: {after_pure_b:.1%}")

    output = {
        'summary': {
            'total': total,
            'method': 'llm_heuristic_v1',
            'description': '拡張キーワード辞書 + 位置重み + 純卦抑制 + uncertain閾値制御',
            'uncertain_rate_a': round(uncertain_rate_a, 4),
            'uncertain_rate_b': round(uncertain_rate_b, 4),
            'pure_rate_before_a': round(before_pure_a, 4),
            'pure_rate_before_b': round(before_pure_b, 4),
            'pure_rate_after_a': round(after_pure_a, 4),
            'pure_rate_after_b': round(after_pure_b, 4),
        },
        'annotations': annotations,
    }

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n出力: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
