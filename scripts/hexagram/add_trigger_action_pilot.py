#!/usr/bin/env python3
"""
Phase 3-4: パイロット64件にtrigger/actionを付与する

trigger類型（判定優先順位順）:
1. T_EXTERNAL_FORCE: 外部から強制的に変化を迫られた（法規制、経済危機、災害等）
2. T_ENVIRONMENT: 外部環境が変化した（市場変化、競合、技術革新等）
3. T_INTERNAL_CRISIS: 内部で問題が発生した（不祥事、経営失敗等）
4. T_LEADERSHIP: 経営者・リーダーの決断や交代
5. T_OPPORTUNITY: 新たな機会を発見・認識した
6. T_STAGNATION: 停滞・行き詰まりから変化が必要になった
7. T_GROWTH_MOMENTUM: 成長・拡大の流れの中での施策
8. T_INTERACTION: 他者との協調・競争が契機

action類型（判定優先順位順）:
1. A_RETREAT: 撤退・縮小
2. A_FOCUS: 選択と集中
3. A_TRANSFORM: 変革・転換
4. A_EXPAND: 拡大・投資
5. A_CONNECT: 協調・提携
6. A_ADAPT: 適応・調整
7. A_MAINTAIN: 維持・継続
8. A_PAUSE: 静観・様子見
"""

import json
import re
from pathlib import Path

# trigger/action判定のためのキーワード定義
TRIGGER_KEYWORDS = {
    'T_EXTERNAL_FORCE': [
        'リーマン', '金融危機', '経済危機', 'バブル崩壊', 'オイルショック', '震災', '地震',
        '台風', '災害', 'コロナ', 'パンデミック', '法規制', '規制強化', '入管法',
        '円高', '円安', '為替', '関税', '貿易摩擦', '制裁', '戦争', '紛争',
        '破綻', '危機', '崩壊', '暴落', 'サブプライム', '通貨危機', '財政危機',
        '外圧', '強制', '迫られ', '直撃', '被災'
    ],
    'T_ENVIRONMENT': [
        '市場縮小', '市場変化', 'EC台頭', 'デジタル化', '技術革新', 'スマホ',
        'Amazon', '競合', 'ライバル', '価格競争', '新興国', '人口減少',
        '少子高齢', '環境変化', '業界再編', 'パラダイム', '時代の変化',
        '成熟', '縮小', '低迷', 'シェア低下', '陳腐化', '先行を許す',
        'デジタルカメラ', 'フィルム', '技術進歩'
    ],
    'T_INTERNAL_CRISIS': [
        '粉飾', '不正', '偽装', '隠蔽', '不祥事', 'リコール', '品質問題',
        '過剰投資', '債務超過', '赤字', '欠陥', '内紛', '対立',
        '失敗', '急拡大', '自社競合', '経営破綻', 'データ偽装',
        '背任', '横領', '問題発覚', 'スキャンダル'
    ],
    'T_LEADERSHIP': [
        '社長', '会長', 'CEO', '経営者', '後継者', '創業者', '承継',
        '就任', '退任', '引退', '交代', '引責', 'リーダー', '委譲',
        '創業', '決断', '発明', '起業', 'トップ', '辞任'
    ],
    'T_OPPORTUNITY': [
        '発見', '発明', '機会', 'チャンス', '可能性', 'ビジネスモデル',
        '新市場', '新規', 'イノベーション', '先駆', 'パイオニア',
        '開拓', '創出', '着眼', '見抜', '潜在', 'ダイヤモンド',
        '特許', '鉱脈', 'ブレークスルー'
    ],
    'T_STAGNATION': [
        '停滞', '行き詰まり', '低迷', '伸び悩み', '足踏み', '膠着',
        '赤字が常態化', '方向性が定まらない', '迷走', '孤立',
        '変わらず', '長期', '大企業病', '硬直', '閉塞', '限界'
    ],
    'T_GROWTH_MOMENTUM': [
        '急成長', '拡大', '成長', '躍進', '好調', '増加',
        '右肩上がり', 'ブーム', 'ヒット', '成功', '急拡大',
        '上昇', '好況', '絶好調', '飛躍', '増収増益'
    ],
    'T_INTERACTION': [
        '提携', 'M&A', '合併', '買収', '連携', '協業', 'アライアンス',
        '統合', 'パートナー', '協力', '協調', '連合', '合従',
        '共同', 'コンソーシアム', 'ジョイント', '資本業務提携'
    ]
}

ACTION_KEYWORDS = {
    'A_RETREAT': [
        '撤退', '縮小', '売却', '閉鎖', '解散', '清算', '廃業',
        '閉店', '退出', '手放す', '引き揚げ', 'リストラ', '削減',
        '整理', '店舗整理', '事業売却', '撤収'
    ],
    'A_FOCUS': [
        '集中', '選択と集中', '特化', '絞り込み', '専念', 'コア',
        '強み', '重点', '注力', '一本足', 'ニッチ', '専門',
        'リソース集中', '経営資源', 'スリム化'
    ],
    'A_TRANSFORM': [
        '変革', '転換', '改革', '刷新', '再編', '革新', '変貌',
        'ピボット', '転身', '脱却', '変身', '再生', '再建',
        'V字回復', '構造改革', '抜本的', '体質改善'
    ],
    'A_EXPAND': [
        '拡大', '投資', '進出', '展開', '多角化', '成長',
        '買収', 'M&A', '新規参入', '出店', '海外展開',
        '増資', '設備投資', '事業拡大', '規模拡大'
    ],
    'A_CONNECT': [
        '提携', '協力', '連携', 'アライアンス', 'パートナーシップ',
        '協業', '協調', '統合', '連合', '共同', '合弁',
        'コンソーシアム', '友好', '同盟', '絆'
    ],
    'A_ADAPT': [
        '適応', '調整', '対応', '順応', '柔軟', '変更',
        '修正', '調和', '融合', 'バランス', '最適化',
        '環境適応', '市場適応', '時代に合わせ'
    ],
    'A_MAINTAIN': [
        '維持', '継続', '持続', '安定', '守り', '堅持',
        '温存', '保守', '現状維持', '堅実', '着実',
        '長期', '一貫', 'ぶれない', '不変'
    ],
    'A_PAUSE': [
        '静観', '様子見', '待機', '見極め', '見守り', '観察',
        '一時停止', '保留', '判断保留', '慎重', '検討',
        '時期を待つ', '機会をうかがう'
    ]
}

# triggerから下卦へのマッピング
TRIGGER_TO_LOWER = {
    'T_GROWTH_MOMENTUM': '乾',
    'T_STAGNATION': '坤',
    'T_EXTERNAL_FORCE': '震',
    'T_ENVIRONMENT': '巽',
    'T_INTERNAL_CRISIS': '坎',
    'T_OPPORTUNITY': '離',
    'T_LEADERSHIP': '艮',
    'T_INTERACTION': '兌'
}

# actionから上卦へのマッピング
ACTION_TO_UPPER = {
    'A_EXPAND': '乾',
    'A_MAINTAIN': '坤',
    'A_TRANSFORM': '震',
    'A_ADAPT': '巽',
    'A_PAUSE': '坎',
    'A_FOCUS': '離',
    'A_RETREAT': '艮',
    'A_CONNECT': '兌'
}

# 八卦の組み合わせから六十四卦番号へのマッピング（伏羲先天八卦序）
# [上卦][下卦] -> 卦番号
TRIGRAM_TO_HEXAGRAM = {
    ('乾', '乾'): 1,  # 乾為天
    ('坤', '乾'): 11, # 地天泰
    ('震', '乾'): 34, # 雷天大壮
    ('巽', '乾'): 9,  # 風天小畜
    ('坎', '乾'): 5,  # 水天需
    ('離', '乾'): 14, # 火天大有
    ('艮', '乾'): 26, # 山天大畜
    ('兌', '乾'): 43, # 沢天夬

    ('乾', '坤'): 12, # 天地否
    ('坤', '坤'): 2,  # 坤為地
    ('震', '坤'): 16, # 雷地豫
    ('巽', '坤'): 20, # 風地観
    ('坎', '坤'): 8,  # 水地比
    ('離', '坤'): 35, # 火地晋
    ('艮', '坤'): 23, # 山地剥
    ('兌', '坤'): 45, # 沢地萃

    ('乾', '震'): 25, # 天雷无妄
    ('坤', '震'): 24, # 地雷復
    ('震', '震'): 51, # 震為雷
    ('巽', '震'): 42, # 風雷益
    ('坎', '震'): 3,  # 水雷屯
    ('離', '震'): 21, # 火雷噬嗑
    ('艮', '震'): 27, # 山雷頤
    ('兌', '震'): 17, # 沢雷随

    ('乾', '巽'): 44, # 天風姤
    ('坤', '巽'): 46, # 地風升
    ('震', '巽'): 32, # 雷風恒
    ('巽', '巽'): 57, # 巽為風
    ('坎', '巽'): 48, # 水風井
    ('離', '巽'): 50, # 火風鼎
    ('艮', '巽'): 18, # 山風蠱
    ('兌', '巽'): 28, # 沢風大過

    ('乾', '坎'): 6,  # 天水訟
    ('坤', '坎'): 7,  # 地水師
    ('震', '坎'): 40, # 雷水解
    ('巽', '坎'): 59, # 風水渙
    ('坎', '坎'): 29, # 坎為水
    ('離', '坎'): 64, # 火水未済
    ('艮', '坎'): 4,  # 山水蒙
    ('兌', '坎'): 47, # 沢水困

    ('乾', '離'): 13, # 天火同人
    ('坤', '離'): 36, # 地火明夷
    ('震', '離'): 55, # 雷火豊
    ('巽', '離'): 37, # 風火家人
    ('坎', '離'): 63, # 水火既済
    ('離', '離'): 30, # 離為火
    ('艮', '離'): 22, # 山火賁
    ('兌', '離'): 49, # 沢火革

    ('乾', '艮'): 33, # 天山遯
    ('坤', '艮'): 15, # 地山謙
    ('震', '艮'): 62, # 雷山小過
    ('巽', '艮'): 53, # 風山漸
    ('坎', '艮'): 39, # 水山蹇
    ('離', '艮'): 56, # 火山旅
    ('艮', '艮'): 52, # 艮為山
    ('兌', '艮'): 31, # 沢山咸

    ('乾', '兌'): 10, # 天沢履
    ('坤', '兌'): 19, # 地沢臨
    ('震', '兌'): 54, # 雷沢帰妹
    ('巽', '兌'): 61, # 風沢中孚
    ('坎', '兌'): 60, # 水沢節
    ('離', '兌'): 38, # 火沢睽
    ('艮', '兌'): 41, # 山沢損
    ('兌', '兌'): 58, # 兌為沢
}

# 六十四卦番号から名前へのマッピング
HEXAGRAM_NAMES = {
    1: '乾為天', 2: '坤為地', 3: '水雷屯', 4: '山水蒙', 5: '水天需',
    6: '天水訟', 7: '地水師', 8: '水地比', 9: '風天小畜', 10: '天沢履',
    11: '地天泰', 12: '天地否', 13: '天火同人', 14: '火天大有', 15: '地山謙',
    16: '雷地豫', 17: '沢雷随', 18: '山風蠱', 19: '地沢臨', 20: '風地観',
    21: '火雷噬嗑', 22: '山火賁', 23: '山地剥', 24: '地雷復', 25: '天雷无妄',
    26: '山天大畜', 27: '山雷頤', 28: '沢風大過', 29: '坎為水', 30: '離為火',
    31: '沢山咸', 32: '雷風恒', 33: '天山遯', 34: '雷天大壮', 35: '火地晋',
    36: '地火明夷', 37: '風火家人', 38: '火沢睽', 39: '水山蹇', 40: '雷水解',
    41: '山沢損', 42: '風雷益', 43: '沢天夬', 44: '天風姤', 45: '沢地萃',
    46: '地風升', 47: '沢水困', 48: '水風井', 49: '沢火革', 50: '火風鼎',
    51: '震為雷', 52: '艮為山', 53: '風山漸', 54: '雷沢帰妹', 55: '雷火豊',
    56: '火山旅', 57: '巽為風', 58: '兌為沢', 59: '風水渙', 60: '水沢節',
    61: '風沢中孚', 62: '雷山小過', 63: '水火既済', 64: '火水未済'
}


def detect_trigger(text: str, estimated_trigger: str = None) -> str:
    """テキストからtriggerを判定する"""
    # estimated_triggerがあれば優先的に使用
    if estimated_trigger and estimated_trigger.startswith('T_'):
        return estimated_trigger

    # キーワードマッチでスコアリング
    scores = {}
    for trigger_type, keywords in TRIGGER_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[trigger_type] = score

    if not scores:
        return 'T_ENVIRONMENT'  # デフォルト

    # 優先順位を考慮してソート
    priority = ['T_EXTERNAL_FORCE', 'T_ENVIRONMENT', 'T_INTERNAL_CRISIS',
                'T_LEADERSHIP', 'T_OPPORTUNITY', 'T_STAGNATION',
                'T_GROWTH_MOMENTUM', 'T_INTERACTION']

    # スコアが同じ場合は優先順位の高いものを選ぶ
    max_score = max(scores.values())
    candidates = [t for t, s in scores.items() if s == max_score]

    for p in priority:
        if p in candidates:
            return p

    return candidates[0]


def detect_action(text: str) -> str:
    """テキストからactionを判定する"""
    # キーワードマッチでスコアリング
    scores = {}
    for action_type, keywords in ACTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[action_type] = score

    if not scores:
        return 'A_ADAPT'  # デフォルト

    # 優先順位を考慮してソート
    priority = ['A_RETREAT', 'A_FOCUS', 'A_TRANSFORM', 'A_EXPAND',
                'A_CONNECT', 'A_ADAPT', 'A_MAINTAIN', 'A_PAUSE']

    max_score = max(scores.values())
    candidates = [a for a, s in scores.items() if s == max_score]

    for p in priority:
        if p in candidates:
            return p

    return candidates[0]


def get_hexagram(trigger: str, action: str) -> tuple:
    """trigger/actionから六十四卦を計算する"""
    lower = TRIGGER_TO_LOWER.get(trigger, '巽')
    upper = ACTION_TO_UPPER.get(action, '巽')

    hexagram_num = TRIGRAM_TO_HEXAGRAM.get((upper, lower), 57)
    hexagram_name = HEXAGRAM_NAMES.get(hexagram_num, '巽為風')

    return lower, upper, hexagram_num, hexagram_name


def process_pilot_file(input_path: str, output_path: str):
    """パイロットファイルを処理する"""
    results = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            case = json.loads(line)
            text = case.get('pre_outcome_text', '')
            estimated_trigger = case.get('estimated_trigger', '')

            # trigger/action判定
            trigger = detect_trigger(text, estimated_trigger)
            action = detect_action(text)

            # 六十四卦を計算
            lower, upper, hex_num, hex_name = get_hexagram(trigger, action)

            # 結果を追加
            case['trigger'] = trigger
            case['action'] = action
            case['lower_trigram'] = lower
            case['upper_trigram'] = upper
            case['hexagram_number'] = hex_num
            case['hexagram_name'] = hex_name

            results.append(case)

    # 出力
    with open(output_path, 'w', encoding='utf-8') as f:
        for case in results:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')

    return results


def print_statistics(results: list):
    """統計を出力する"""
    # trigger統計
    trigger_counts = {}
    for r in results:
        t = r['trigger']
        trigger_counts[t] = trigger_counts.get(t, 0) + 1

    print("\n=== Trigger統計 ===")
    for t, c in sorted(trigger_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}件")

    # action統計
    action_counts = {}
    for r in results:
        a = r['action']
        action_counts[a] = action_counts.get(a, 0) + 1

    print("\n=== Action統計 ===")
    for a, c in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {a}: {c}件")

    # 六十四卦統計
    hex_counts = {}
    for r in results:
        h = r['hexagram_name']
        hex_counts[h] = hex_counts.get(h, 0) + 1

    print("\n=== 六十四卦統計 ===")
    for h, c in sorted(hex_counts.items(), key=lambda x: -x[1]):
        print(f"  {h}: {c}件")


if __name__ == '__main__':
    base_path = Path('/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB')
    input_file = base_path / 'data/hexagram/pilot_64.jsonl'
    output_file = base_path / 'data/hexagram/pilot_64_with_hexagram.jsonl'

    print("パイロット64件にtrigger/actionを付与中...")
    results = process_pilot_file(str(input_file), str(output_file))

    print(f"\n処理完了: {len(results)}件")
    print(f"出力先: {output_file}")

    print_statistics(results)
