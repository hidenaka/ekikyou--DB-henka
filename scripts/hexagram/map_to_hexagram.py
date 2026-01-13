#!/usr/bin/env python3
"""
Phase 3-4: 全件処理スクリプト - cases_with_pre_outcome.jsonl から六十四卦をマッピング

処理フロー:
1. data/raw/cases_with_pre_outcome.jsonl を読み込む
2. 各事例のpre_outcome_textからtrigger/actionを抽出
3. trigger→下卦、action→上卦の対応で六十四卦を決定
4. 結果を data/raw/cases_with_hexagram.jsonl に出力
"""

import json
import re
from pathlib import Path
from datetime import datetime
import sys

# ===== trigger/action判定のためのキーワード定義 =====

TRIGGER_KEYWORDS = {
    'T_EXTERNAL_FORCE': [
        'リーマン', '金融危機', '経済危機', 'バブル崩壊', 'オイルショック', '震災', '地震',
        '台風', '災害', 'コロナ', 'パンデミック', '法規制', '規制強化', '入管法',
        '円高', '円安', '為替', '関税', '貿易摩擦', '制裁', '戦争', '紛争',
        '破綻', '危機', '崩壊', '暴落', 'サブプライム', '通貨危機', '財政危機',
        '外圧', '強制', '迫られ', '直撃', '被災', '規制', '法改正', 'ロックダウン',
        'ショック', 'クラッシュ', '暴落', '急落', '急騰', 'インフレ', 'デフレ',
        '恐慌', '不況', 'リセッション', '制裁', '経済制裁', '封鎖', '禁輸',
        '洪水', '津波', '原発', '福島', 'チェルノブイリ'
    ],
    'T_ENVIRONMENT': [
        '市場縮小', '市場変化', 'EC台頭', 'デジタル化', '技術革新', 'スマホ',
        'Amazon', '競合', 'ライバル', '価格競争', '新興国', '人口減少',
        '少子高齢', '環境変化', '業界再編', 'パラダイム', '時代の変化',
        '成熟', '縮小', '低迷', 'シェア低下', '陳腐化', '先行を許す',
        'デジタルカメラ', 'フィルム', '技術進歩', 'テクノロジー', 'AI',
        '人工知能', 'DX', 'IoT', 'クラウド', '5G', '6G', '自動運転',
        'EV', '電気自動車', '再エネ', '脱炭素', 'カーボンニュートラル',
        'グローバル化', '新興市場', 'BRICS', '中国', 'インド', 'ASEAN',
        '競争激化', 'コモディティ化', '差別化', 'ディスラプション',
        '構造変化', '産業構造', '業界地図', 'ゲームチェンジャー'
    ],
    'T_INTERNAL_CRISIS': [
        '粉飾', '不正', '偽装', '隠蔽', '不祥事', 'リコール', '品質問題',
        '過剰投資', '債務超過', '赤字', '欠陥', '内紛', '対立',
        '失敗', '急拡大', '自社競合', '経営破綻', 'データ偽装',
        '背任', '横領', '問題発覚', 'スキャンダル', '訴訟', '告発',
        '内部告発', 'パワハラ', 'セクハラ', 'コンプライアンス違反',
        '品質不正', '検査不正', '談合', '贈賄', '収賄', '脱税',
        '資金流出', '資金ショート', '倒産', '破産', '民事再生',
        '会社更生', '債務不履行', 'デフォルト', '信用不安'
    ],
    'T_LEADERSHIP': [
        '社長', '会長', 'CEO', '経営者', '後継者', '創業者', '承継',
        '就任', '退任', '引退', '交代', '引責', 'リーダー', '委譲',
        '創業', '決断', '発明', '起業', 'トップ', '辞任', '解任',
        'COO', 'CFO', 'CTO', '取締役', '執行役員', '代表', '理事長',
        '院長', '総裁', '頭取', '監督', '指揮者', '指導者', 'カリスマ',
        '世襲', '同族', 'ファミリー', 'オーナー', '大株主'
    ],
    'T_OPPORTUNITY': [
        '発見', '発明', '機会', 'チャンス', '可能性', 'ビジネスモデル',
        '新市場', '新規', 'イノベーション', '先駆', 'パイオニア',
        '開拓', '創出', '着眼', '見抜', '潜在', 'ダイヤモンド',
        '特許', '鉱脈', 'ブレークスルー', '革新的', '画期的',
        'ゲームチェンジ', 'ディスラプト', 'ブルーオーシャン',
        '未開拓', 'アンタップド', 'ポテンシャル', '成長市場',
        'フロンティア', '新領域', '新分野', '新技術', '新素材'
    ],
    'T_STAGNATION': [
        '停滞', '行き詰まり', '低迷', '伸び悩み', '足踏み', '膠着',
        '赤字が常態化', '方向性が定まらない', '迷走', '孤立',
        '変わらず', '長期', '大企業病', '硬直', '閉塞', '限界',
        'マンネリ', 'ジレンマ', '袋小路', 'デッドロック',
        '前進できない', '打開策がない', '手詰まり', '八方塞がり',
        '成長が止まった', '頭打ち', '天井', '上限', '限界点'
    ],
    'T_GROWTH_MOMENTUM': [
        '急成長', '拡大', '成長', '躍進', '好調', '増加',
        '右肩上がり', 'ブーム', 'ヒット', '成功', '急拡大',
        '上昇', '好況', '絶好調', '飛躍', '増収増益', '過去最高',
        '記録更新', 'V字回復', '黒字転換', '最高益', '売上増',
        '利益増', 'シェア拡大', '市場制覇', '首位', 'トップ',
        'リーディング', 'No.1', '業界首位', 'グローバル展開'
    ],
    'T_INTERACTION': [
        '提携', 'M&A', '合併', '買収', '連携', '協業', 'アライアンス',
        '統合', 'パートナー', '協力', '協調', '連合', '合従',
        '共同', 'コンソーシアム', 'ジョイント', '資本業務提携',
        '子会社化', '完全子会社', '経営統合', '持株会社', '吸収',
        '合弁', 'JV', 'OEM', 'ライセンス', 'フランチャイズ',
        '業務委託', 'アウトソーシング', 'BPO', 'オフショア'
    ]
}

ACTION_KEYWORDS = {
    'A_RETREAT': [
        '撤退', '縮小', '売却', '閉鎖', '解散', '清算', '廃業',
        '閉店', '退出', '手放す', '引き揚げ', 'リストラ', '削減',
        '整理', '店舗整理', '事業売却', '撤収', '撤去', '廃止',
        '終了', '中止', '打ち切り', '断念', '見送り', '凍結',
        'スピンオフ', 'カーブアウト', 'ダイベスト', '資産圧縮',
        '人員削減', '希望退職', '早期退職', '整理解雇', '工場閉鎖'
    ],
    'A_FOCUS': [
        '集中', '選択と集中', '特化', '絞り込み', '専念', 'コア',
        '強み', '重点', '注力', '一本足', 'ニッチ', '専門',
        'リソース集中', '経営資源', 'スリム化', 'フォーカス',
        'コアコンピタンス', '得意分野', '本業回帰', '原点回帰',
        '事業整理', 'ポートフォリオ', '選別', '峻別', '取捨選択',
        '主力事業', '成長事業', '収益源', '柱'
    ],
    'A_TRANSFORM': [
        '変革', '転換', '改革', '刷新', '再編', '革新', '変貌',
        'ピボット', '転身', '脱却', '変身', '再生', '再建',
        'V字回復', '構造改革', '抜本的', '体質改善', 'トランスフォーム',
        'リストラクチャリング', 'リエンジニアリング', 'DX',
        'デジタルトランスフォーメーション', 'ビジネスモデル変革',
        '組織改革', '意識改革', '風土改革', '働き方改革'
    ],
    'A_EXPAND': [
        '拡大', '投資', '進出', '展開', '多角化', '成長',
        '買収', 'M&A', '新規参入', '出店', '海外展開',
        '増資', '設備投資', '事業拡大', '規模拡大', '積極投資',
        '大型投資', '戦略投資', '成長投資', '研究開発',
        'グローバル展開', '海外進出', '新市場開拓', '新事業',
        '新製品', '新サービス', 'ラインナップ拡充'
    ],
    'A_CONNECT': [
        '提携', '協力', '連携', 'アライアンス', 'パートナーシップ',
        '協業', '協調', '統合', '連合', '共同', '合弁',
        'コンソーシアム', '友好', '同盟', '絆', 'コラボ',
        'コラボレーション', 'オープンイノベーション', '共創',
        'エコシステム', 'プラットフォーム', 'ネットワーク',
        'コミュニティ', '組合', '組織化', '団体', '協会'
    ],
    'A_ADAPT': [
        '適応', '調整', '対応', '順応', '柔軟', '変更',
        '修正', '調和', '融合', 'バランス', '最適化',
        '環境適応', '市場適応', '時代に合わせ', 'アジャスト',
        'カスタマイズ', 'ローカライズ', 'パーソナライズ',
        '微調整', 'チューニング', '改善', '改良', '改修',
        'アップデート', 'バージョンアップ', 'リニューアル'
    ],
    'A_MAINTAIN': [
        '維持', '継続', '持続', '安定', '守り', '堅持',
        '温存', '保守', '現状維持', '堅実', '着実',
        '長期', '一貫', 'ぶれない', '不変', 'サステナブル',
        '永続', '恒常', '定常', '安定経営', '堅実経営',
        '保全', '保護', '確保', 'キープ', '定着'
    ],
    'A_PAUSE': [
        '静観', '様子見', '待機', '見極め', '見守り', '観察',
        '一時停止', '保留', '判断保留', '慎重', '検討',
        '時期を待つ', '機会をうかがう', 'ホールド', 'ペンディング',
        '先送り', '延期', '繰り延べ', '見合わせ', '自粛',
        '控える', '抑制', '我慢', '忍耐', '辛抱'
    ]
}

# ===== 八卦マッピング =====

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

# 八卦の組み合わせから六十四卦番号へのマッピング
TRIGRAM_TO_HEXAGRAM = {
    ('乾', '乾'): 1, ('坤', '乾'): 11, ('震', '乾'): 34, ('巽', '乾'): 9,
    ('坎', '乾'): 5, ('離', '乾'): 14, ('艮', '乾'): 26, ('兌', '乾'): 43,
    ('乾', '坤'): 12, ('坤', '坤'): 2, ('震', '坤'): 16, ('巽', '坤'): 20,
    ('坎', '坤'): 8, ('離', '坤'): 35, ('艮', '坤'): 23, ('兌', '坤'): 45,
    ('乾', '震'): 25, ('坤', '震'): 24, ('震', '震'): 51, ('巽', '震'): 42,
    ('坎', '震'): 3, ('離', '震'): 21, ('艮', '震'): 27, ('兌', '震'): 17,
    ('乾', '巽'): 44, ('坤', '巽'): 46, ('震', '巽'): 32, ('巽', '巽'): 57,
    ('坎', '巽'): 48, ('離', '巽'): 50, ('艮', '巽'): 18, ('兌', '巽'): 28,
    ('乾', '坎'): 6, ('坤', '坎'): 7, ('震', '坎'): 40, ('巽', '坎'): 59,
    ('坎', '坎'): 29, ('離', '坎'): 64, ('艮', '坎'): 4, ('兌', '坎'): 47,
    ('乾', '離'): 13, ('坤', '離'): 36, ('震', '離'): 55, ('巽', '離'): 37,
    ('坎', '離'): 63, ('離', '離'): 30, ('艮', '離'): 22, ('兌', '離'): 49,
    ('乾', '艮'): 33, ('坤', '艮'): 15, ('震', '艮'): 62, ('巽', '艮'): 53,
    ('坎', '艮'): 39, ('離', '艮'): 56, ('艮', '艮'): 52, ('兌', '艮'): 31,
    ('乾', '兌'): 10, ('坤', '兌'): 19, ('震', '兌'): 54, ('巽', '兌'): 61,
    ('坎', '兌'): 60, ('離', '兌'): 38, ('艮', '兌'): 41, ('兌', '兌'): 58,
}

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


def detect_trigger(text: str, existing_trigger_hex: str = None) -> str:
    """テキストからtriggerを判定する

    既存のtrigger_hexがあればそれを考慮してマッピング
    """
    # 既存のtrigger_hexから逆算
    if existing_trigger_hex:
        hex_to_trigger = {
            '乾': 'T_GROWTH_MOMENTUM',
            '坤': 'T_STAGNATION',
            '震': 'T_EXTERNAL_FORCE',
            '巽': 'T_ENVIRONMENT',
            '坎': 'T_INTERNAL_CRISIS',
            '離': 'T_OPPORTUNITY',
            '艮': 'T_LEADERSHIP',
            '兌': 'T_INTERACTION'
        }
        if existing_trigger_hex in hex_to_trigger:
            return hex_to_trigger[existing_trigger_hex]

    # キーワードマッチでスコアリング
    scores = {}
    for trigger_type, keywords in TRIGGER_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[trigger_type] = score

    if not scores:
        return 'T_ENVIRONMENT'  # デフォルト

    # 優先順位
    priority = ['T_EXTERNAL_FORCE', 'T_ENVIRONMENT', 'T_INTERNAL_CRISIS',
                'T_LEADERSHIP', 'T_OPPORTUNITY', 'T_STAGNATION',
                'T_GROWTH_MOMENTUM', 'T_INTERACTION']

    max_score = max(scores.values())
    candidates = [t for t, s in scores.items() if s == max_score]

    for p in priority:
        if p in candidates:
            return p

    return candidates[0]


def detect_action(text: str, existing_action_hex: str = None) -> str:
    """テキストからactionを判定する"""
    # 既存のaction_hexから逆算
    if existing_action_hex:
        hex_to_action = {
            '乾': 'A_EXPAND',
            '坤': 'A_MAINTAIN',
            '震': 'A_TRANSFORM',
            '巽': 'A_ADAPT',
            '坎': 'A_PAUSE',
            '離': 'A_FOCUS',
            '艮': 'A_RETREAT',
            '兌': 'A_CONNECT'
        }
        if existing_action_hex in hex_to_action:
            return hex_to_action[existing_action_hex]

    # キーワードマッチでスコアリング
    scores = {}
    for action_type, keywords in ACTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[action_type] = score

    if not scores:
        return 'A_ADAPT'  # デフォルト

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


def process_all_cases(input_path: str, output_path: str) -> dict:
    """全件を処理して六十四卦をマッピングする"""
    stats = {
        'total': 0,
        'processed': 0,
        'errors': 0,
        'triggers': {},
        'actions': {},
        'hexagrams': {},
        'lower_trigrams': {},
        'upper_trigrams': {}
    }

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            stats['total'] += 1

            if not line.strip():
                continue

            try:
                case = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num}: JSON parse error - {e}")
                stats['errors'] += 1
                continue

            # テキストを取得
            text = case.get('pre_outcome_text', '') or ''
            text += ' ' + (case.get('description', '') or '')
            text += ' ' + (case.get('entity_name', '') or '')

            # 既存のtrigger_hex/action_hexがあれば参照
            existing_trigger_hex = case.get('trigger_hex', '')
            existing_action_hex = case.get('action_hex', '')

            # trigger/action判定
            trigger = detect_trigger(text, existing_trigger_hex)
            action = detect_action(text, existing_action_hex)

            # 六十四卦を計算
            lower, upper, hex_num, hex_name = get_hexagram(trigger, action)

            # 結果を追加
            case['trigger'] = trigger
            case['action'] = action
            case['lower_trigram'] = lower
            case['upper_trigram'] = upper
            case['hexagram_number'] = hex_num
            case['hexagram_name'] = hex_name

            # 統計を更新
            stats['triggers'][trigger] = stats['triggers'].get(trigger, 0) + 1
            stats['actions'][action] = stats['actions'].get(action, 0) + 1
            stats['hexagrams'][hex_name] = stats['hexagrams'].get(hex_name, 0) + 1
            stats['lower_trigrams'][lower] = stats['lower_trigrams'].get(lower, 0) + 1
            stats['upper_trigrams'][upper] = stats['upper_trigrams'].get(upper, 0) + 1

            # 出力
            outfile.write(json.dumps(case, ensure_ascii=False) + '\n')
            stats['processed'] += 1

            # 進捗表示
            if stats['processed'] % 1000 == 0:
                print(f"  処理中: {stats['processed']:,}件...")

    return stats


def print_statistics(stats: dict):
    """統計を出力する"""
    print("\n" + "=" * 60)
    print("六十四卦マッピング完了 - 統計レポート")
    print("=" * 60)

    print(f"\n【処理結果】")
    print(f"  総件数: {stats['total']:,}件")
    print(f"  処理済: {stats['processed']:,}件")
    print(f"  エラー: {stats['errors']:,}件")

    print(f"\n【Trigger分布】（下卦の契機）")
    for t, c in sorted(stats['triggers'].items(), key=lambda x: -x[1]):
        pct = c / stats['processed'] * 100
        bar = '#' * int(pct / 2)
        print(f"  {t:22s}: {c:5,}件 ({pct:5.1f}%) {bar}")

    print(f"\n【Action分布】（上卦の対応）")
    for a, c in sorted(stats['actions'].items(), key=lambda x: -x[1]):
        pct = c / stats['processed'] * 100
        bar = '#' * int(pct / 2)
        print(f"  {a:22s}: {c:5,}件 ({pct:5.1f}%) {bar}")

    print(f"\n【下卦（内卦）分布】")
    for t, c in sorted(stats['lower_trigrams'].items(), key=lambda x: -x[1]):
        pct = c / stats['processed'] * 100
        print(f"  {t}: {c:5,}件 ({pct:5.1f}%)")

    print(f"\n【上卦（外卦）分布】")
    for t, c in sorted(stats['upper_trigrams'].items(), key=lambda x: -x[1]):
        pct = c / stats['processed'] * 100
        print(f"  {t}: {c:5,}件 ({pct:5.1f}%)")

    print(f"\n【六十四卦分布 Top20】")
    sorted_hex = sorted(stats['hexagrams'].items(), key=lambda x: -x[1])[:20]
    for h, c in sorted_hex:
        pct = c / stats['processed'] * 100
        bar = '#' * int(pct)
        print(f"  {h:12s}: {c:5,}件 ({pct:5.1f}%) {bar}")

    # カバレッジ
    covered = len([h for h, c in stats['hexagrams'].items() if c > 0])
    print(f"\n【六十四卦カバレッジ】")
    print(f"  カバー数: {covered}/64卦 ({covered/64*100:.1f}%)")

    # 0件の卦
    all_hex = set(HEXAGRAM_NAMES.values())
    covered_hex = set(stats['hexagrams'].keys())
    missing = all_hex - covered_hex
    if missing:
        print(f"  未カバー: {', '.join(sorted(missing))}")


def main():
    base_path = Path('/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB')
    input_file = base_path / 'data/raw/cases_with_pre_outcome.jsonl'
    output_file = base_path / 'data/raw/cases_with_hexagram.jsonl'

    print("=" * 60)
    print("六十四卦マッピング処理開始")
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print(f"\n入力: {input_file}")
    print(f"出力: {output_file}")

    if not input_file.exists():
        print(f"\n[ERROR] 入力ファイルが存在しません: {input_file}")
        sys.exit(1)

    print("\n処理中...")
    stats = process_all_cases(str(input_file), str(output_file))

    print_statistics(stats)

    print(f"\n完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"出力先: {output_file}")


if __name__ == '__main__':
    main()
