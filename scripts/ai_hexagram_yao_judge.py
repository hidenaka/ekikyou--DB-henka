#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI判定スクリプト: 64卦 + 爻位の同時判定

事例の内容（story_summary, before_state, after_state, pattern_type, outcome等）
から適切な64卦と爻位（1-6）を判定する。

判定ロジック:
1. 64卦判定: パターンタイプ、結果、状態変化の組み合わせから卦を選択
2. 爻位判定: 物語のフェーズ（開始・成長・過渡・転換・頂点・終結）を判定
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import re
import random

# ==============================================================================
# パスと設定
# ==============================================================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

CASES_PATH = PROJECT_ROOT / "data/raw/cases.jsonl"
HEXAGRAM_MASTER_PATH = PROJECT_ROOT / "data/hexagrams/hexagram_master.json"
YAO_MASTER_PATH = PROJECT_ROOT / "data/hexagrams/yao_master.json"

# ==============================================================================
# 改善版スコアリング定数（v2）
# ==============================================================================

# 基本スコア定数
PATTERN_BASE_SCORE = 5.0       # pattern_typeの最大スコア
PATTERN_DECAY = 0.5            # 順位ごとの減衰
STATE_SCORE = 2.0              # before/after_stateのスコア
KEYWORD_BASE_SCORE = 4.0       # キーワードマッチ基本スコア（改善: 1.5→4.0）
MEANING_PHRASE_SCORE = 1.0     # meaning部分一致スコア（改善: 0.5→1.0）

# 新規追加スコア
HEXAGRAM_NAME_BONUS = 10.0     # 卦名完全一致ボーナス（新規）
SPECIFIC_KEYWORD_SCORE = 3.0   # 卦固有キーワードボーナス（新規）

# ==============================================================================
# 卦名エイリアス（短縮名→卦ID）- 直接マッチ用
# ==============================================================================
HEXAGRAM_NAME_ALIASES: Dict[int, List[str]] = {
    1: ["乾", "乾為天"],
    2: ["坤", "坤為地"],
    3: ["屯", "水雷屯", "スタートアップ苦難", "創業苦難"],
    4: ["蒙", "山水蒙"],
    5: ["需", "水天需"],
    6: ["訟", "天水訟"],
    7: ["師", "地水師", "大規模動員", "組織力結集"],
    8: ["比", "水地比"],
    9: ["小畜", "風天小畜"],
    10: ["履", "天沢履", "虎の尾", "コンプライアンス重視"],
    11: ["泰", "地天泰"],
    12: ["否", "天地否"],
    13: ["同人", "天火同人", "業界連合", "オープンイノベーション"],
    14: ["大有", "火天大有"],
    15: ["謙", "地山謙"],
    16: ["豫", "雷地豫"],
    17: ["随", "沢雷随"],
    18: ["蠱", "山風蠱"],
    19: ["臨", "地沢臨"],
    20: ["観", "風地観"],
    21: ["噬嗑", "火雷噬嗑"],
    22: ["賁", "山火賁"],
    23: ["剥", "山地剥"],
    24: ["復", "地雷復"],
    25: ["无妄", "天雷无妄", "誠実経営", "天災", "不可抗力"],
    26: ["大畜", "山天大畜", "内部留保", "人材蓄積", "技術蓄積"],
    27: ["頤", "山雷頤"],
    28: ["大過", "沢風大過", "過剰投資", "バブル", "オーバーストレッチ"],
    29: ["坎", "坎為水"],
    30: ["離", "離為火"],
    31: ["咸", "沢山咸", "顧客共感", "市場感応"],
    32: ["恒", "雷風恒"],
    33: ["遯", "天山遯"],
    34: ["大壮", "雷天大壮", "積極攻勢", "勢いに乗る"],
    35: ["晋", "火地晋"],
    36: ["明夷", "地火明夷"],
    37: ["家人", "風火家人"],
    38: ["睽", "火沢睽"],
    39: ["蹇", "水山蹇"],
    40: ["解", "雷水解"],
    41: ["損", "山沢損"],
    42: ["益", "風雷益"],
    43: ["夬", "沢天夬"],
    44: ["姤", "天風姤"],
    45: ["萃", "沢地萃", "人材集結", "チームビルディング"],
    46: ["升", "地風升"],
    47: ["困", "沢水困"],
    48: ["井", "水風井"],
    49: ["革", "沢火革"],
    50: ["鼎", "火風鼎"],
    51: ["震", "震為雷"],
    52: ["艮", "艮為山"],
    53: ["漸", "風山漸"],
    54: ["帰妹", "雷沢帰妹", "被買収", "子会社化"],
    55: ["豊", "雷火豊"],
    56: ["旅", "火山旅", "海外展開", "一時プロジェクト"],
    57: ["巽", "巽為風"],
    58: ["兌", "兌為沢"],
    59: ["渙", "風水渙", "分社化", "スピンオフ", "組織分散"],
    60: ["節", "水沢節"],
    61: ["中孚", "風沢中孚"],
    62: ["小過", "雷山小過"],
    63: ["既済", "水火既済"],
    64: ["未済", "火水未済"],
}

# ==============================================================================
# 卦固有キーワード辞書（希少卦の識別強化）
# ==============================================================================
HEXAGRAM_SPECIFIC_KEYWORDS: Dict[int, List[str]] = {
    1: ["乾為天", "純粋な陽", "創造力", "天の力", "剛健", "自強不息", "龍の象"],  # 新規追加
    2: ["坤為地", "純粋な陰", "受容", "柔順", "包容力", "地の徳", "牝馬の象"],  # 新規追加
    3: ["産みの苦しみ", "困難な始まり", "創業の苦労", "立ち上げ期", "新規参入の壁", "スタートアップ初期", "創業期混乱"],
    4: ["山水蒙", "蒙", "啓蒙", "教育", "指導", "未熟", "学び"],  # 新規追加
    18: ["山風蠱", "蠱", "腐敗", "改革", "立て直し", "刷新", "弊害除去"],  # 新規追加
    33: ["天山遯", "遯", "隠遁", "退避", "戦略的撤退", "身を引く"],  # 新規追加
    43: ["沢天夬", "夬", "決断", "決裂", "断行", "決然と断つ"],  # 新規追加
    48: ["水風井", "井", "井戸", "不変の価値", "安定供給", "源泉"],  # 新規追加
    58: ["兌為沢", "兌", "喜び", "悦楽", "交流", "沢の象"],  # 新規追加
    7: ["大規模動員", "組織力結集", "統率強化", "全社プロジェクト", "総力戦", "軍事的", "組織的対応"],
    9: ["風天小畜", "小畜", "小さな蓄え", "少しずつ蓄える", "待機", "準備期間", "力不足"],  # 新規追加
    10: ["虎の尾を踏む", "慎重に", "礼儀正しく", "コンプライアンス", "危険を避け", "リスク管理", "法務重視", "ガバナンス"],
    13: ["同志が集まり", "業界団体", "コンソーシアム", "標準化活動", "オープンソース", "同業協調", "志の共有"],
    14: ["大いなる所有", "火天大有", "繁栄を謳歌", "豊かさの中", "大成功の頂点", "富の蓄積", "所有の極み"],  # 新規追加
    16: ["雷地豫", "豫", "悦楽", "楽観", "準備万端", "歓喜", "娯楽"],  # 新規追加
    21: ["火雷噬嗑", "噬嗑", "噛み砕", "問題解決", "障害除去", "法的措置", "断罪", "粛清"],  # 新規追加
    22: ["山火賁", "賁", "装飾", "美化", "文飾", "ブランド構築", "見せかけ", "外観"],  # 新規追加
    25: ["誠実", "天命", "無妄", "偽りなき", "正道経営", "予期せぬ事態", "自然災害", "不可抗力"],
    26: ["蓄積", "蓄える", "内部留保", "人材育成", "力を溜める", "抑制", "養成", "貯蓄", "内に力", "R&D投資", "技術蓄積", "設備投資準備"],
    28: ["過剰", "無理", "行き過ぎ", "オーバー", "バブル", "負債過多", "棟木がたわむ", "極端"],
    31: ["沢山咸", "咸", "顧客共感", "市場感応", "感情的つながり", "共鳴", "相互作用", "心を通わせ"],
    32: ["雷風恒", "恒", "恒常", "持続", "永続", "一貫性", "不変"],  # 新規追加
    38: ["火沢睽", "睽", "背反", "対立", "不和", "異見", "競合"],  # 新規追加
    34: ["勢いに乗り", "積極的に拡大", "大いなる力", "攻勢に出る", "勢い", "積極拡大"],
    44: ["偶発的機会", "予期せぬ出会い", "リスク含む出会い", "突然の商談"],
    45: ["人材集結", "チームビルディング", "組織拡大", "求心力", "採用強化"],
    50: ["火風鼎", "鼎の象", "新しい秩序", "新体制構築", "調和の器", "三本足", "安定した基盤づくり"],  # 新規追加
    53: ["風山漸", "漸", "漸進", "徐々に", "ゆっくり進む", "着実な歩み", "段階的成長"],  # 新規追加
    54: ["従属", "副次的", "買収される", "子会社化", "連合参加", "統合される側"],
    56: ["旅人", "海外赴任", "プロジェクト型", "一時的拠点", "定まらぬ"],
    57: ["巽為風", "巽", "従順", "柔軟", "浸透", "風の象", "謙虚な姿勢"],  # 新規追加
    60: ["水沢節", "節", "節度", "制約", "節制", "適度", "限界設定"],  # 新規追加
    64: ["火水未済", "未済", "未完成", "発展途上", "まだ終わらない", "可能性"],  # 新規追加
    59: ["離散", "分社化", "スピンオフ", "事業分離", "分権化", "組織分散", "散り散り"],
    61: ["風沢中孚", "内に誠", "真心", "信頼関係", "誠実な交流", "心からの信頼", "互いの誠"],  # 新規追加
}

# ==============================================================================
# 希少卦ブースト係数（カバレッジ均等化用）
# ==============================================================================
RARE_HEXAGRAM_BOOST: Dict[int, float] = {
    1: 1.5,   # 乾 - 新規追加（空セル対策）
    2: 1.5,   # 坤 - 新規追加（空セル対策）
    26: 2.0,  # 大畜 - 最優先
    25: 2.0,  # 无妄
    13: 2.0,  # 同人
    54: 2.0,  # 帰妹
    14: 2.0,  # 大有 - 新規追加（空セル対策）
    50: 2.0,  # 鼎 - 新規追加（空セル対策）
    61: 2.0,  # 中孚 - 新規追加（空セル対策）
    21: 2.0,  # 噬嗑 - 新規追加（空セル対策）
    22: 2.0,  # 賁 - 新規追加（空セル対策）
    9: 1.8,   # 小畜 - 新規追加（空セル対策）
    16: 1.8,  # 豫 - 新規追加（空セル対策）
    53: 1.8,  # 漸 - 新規追加（空セル対策）
    32: 1.8,  # 恒 - 新規追加（空セル対策）
    38: 1.8,  # 睽 - 新規追加（空セル対策）
    57: 1.8,  # 巽 - 新規追加（空セル対策）
    60: 1.8,  # 節 - 新規追加（空セル対策）
    64: 1.8,  # 未済 - 新規追加（空セル対策）
    10: 1.8,  # 履
    45: 1.8,  # 萃
    28: 1.8,  # 大過
    59: 1.8,  # 渙
    3: 1.5,   # 屯
    7: 1.5,   # 師
    31: 1.5,  # 咸
    34: 1.5,  # 大壮
    44: 1.5,  # 姤
    56: 1.5,  # 旅
    6: 1.3,   # 訟
}

# ==============================================================================
# 64卦のカテゴリ分類（内容ベース判定用）
# ==============================================================================

# パターンタイプ → 適合する卦のリスト（優先度順、拡張版：全64卦カバー）
PATTERN_TO_HEXAGRAMS: Dict[str, List[int]] = {
    # 成功系パターン（安定成長・好調）
    "Steady_Growth": [11, 53, 46, 32, 57, 16, 25, 58, 61, 9, 48, 21],  # 泰、漸、升、恒、巽、豫、无妄、兌、中孚、小畜、井、噬嗑
    "Strategic_Patience": [5, 20, 33, 4, 60, 15, 22, 26, 52, 62, 9, 48],  # 需、観、遯、蒙、節、謙、賁、大畜、艮、小過、小畜、井
    "Opportunity_Seized": [42, 35, 14, 55, 34, 1, 17, 30, 56, 10],  # 益、晋、大有、豊、大壮、乾、随、離、旅、履
    "Collaborative_Rise": [8, 13, 37, 45, 31, 19, 58, 41, 61, 28],  # 比、同人、家人、萃、咸、臨、兌、損、中孚、大過
    "Bold_Leap": [1, 34, 43, 49, 17, 25, 14, 55, 30, 56],  # 乾、大壮、夬、革、随、无妄、大有、豊、離、旅

    # 危機対応系パターン
    "Shock_Recovery": [51, 40, 24, 21, 3, 16, 55, 63, 27, 42],  # 震、解、復、噬嗑、屯、豫、豊、既済、頤、益
    "Crisis_Pivot": [49, 18, 64, 59, 47, 38, 50, 3, 48, 5],  # 革、蠱、未済、渙、困、睽、鼎、屯、井、需
    "Adaptive_Survival": [29, 39, 47, 36, 6, 7, 60, 4, 48, 63],  # 坎、蹇、困、明夷、訟、師、節、蒙、井、既済

    # 衰退系パターン
    "Hubris_Collapse": [12, 23, 36, 44, 33, 28, 54, 38, 10, 6],  # 否、剥、明夷、姤、遯、大過、帰妹、睽、履、訟
    "Slow_Decline": [23, 12, 33, 52, 62, 36, 4, 20, 15, 2],  # 剥、否、遯、艮、小過、明夷、蒙、観、謙、坤
    "Quiet_Fade": [52, 62, 15, 20, 4, 2, 22, 26, 41, 9],  # 艮、小過、謙、観、蒙、坤、賁、大畜、損、小畜
    "Managed_Decline": [19, 41, 15, 2, 7, 8, 46, 53, 60, 48],  # 臨、損、謙、坤、師、比、升、漸、節、井
    "Failed_Attempt": [47, 29, 39, 4, 3, 6, 36, 64, 59, 54],  # 困、坎、蹇、蒙、屯、訟、明夷、未済、渙、帰妹

    # 変革系パターン
    "Legacy_Burden": [18, 23, 36, 12, 44, 27, 50, 28, 53, 32],  # 蠱、剥、明夷、否、姤、頤、鼎、大過、漸、恒
    "Tech_Disruption": [49, 50, 64, 63, 21, 30, 56, 38, 17, 35],  # 革、鼎、未済、既済、噬嗑、離、旅、睽、随、晋
    "Market_Shift": [54, 38, 10, 6, 47, 44, 31, 28, 43, 58],  # 帰妹、睽、履、訟、困、姤、咸、大過、夬、兌
    "Generational_Change": [18, 50, 27, 44, 32, 53, 37, 13, 19, 45],  # 蠱、鼎、頤、姤、恒、漸、家人、同人、臨、萃

    # その他
    "Internal_Conflict": [6, 38, 47, 10, 21, 44, 54, 43, 28, 12, 9],  # 訟、睽、困、履、噬嗑、姤、帰妹、夬、大過、否、小畜
    "External_Threat": [7, 13, 6, 33, 36, 29, 39, 5, 40, 51, 21, 48],  # 師、同人、訟、遯、明夷、坎、蹇、需、解、震、噬嗑、井
}

# 結果 → 卦の重み付け調整
OUTCOME_HEXAGRAM_WEIGHTS: Dict[str, Dict[int, float]] = {
    "Success": {
        1: 1.5, 11: 1.5, 14: 1.5, 35: 1.5, 42: 1.5, 55: 1.5,  # 成功に適合
        47: 0.3, 29: 0.3, 39: 0.3, 36: 0.3,  # 失敗卦は下げる
    },
    "Failure": {
        47: 1.5, 29: 1.5, 39: 1.5, 36: 1.5, 12: 1.5, 23: 1.5,  # 失敗に適合
        1: 0.3, 11: 0.3, 14: 0.3, 55: 0.3,  # 成功卦は下げる
    },
    "Mixed": {
        63: 1.3, 64: 1.3, 62: 1.3, 9: 1.3, 10: 1.3,  # 両義的な卦
    },
}

# before_state → 卦の関連付け（拡張版、全64卦カバー）
BEFORE_STATE_HINTS: Dict[str, List[int]] = {
    "安定・平和": [11, 32, 2, 8, 15, 58, 45, 37, 31, 61, 9, 48],  # 泰、恒、坤、比、謙、兌、萃、家人、咸、中孚、小畜、井
    "停滞・閉塞": [12, 52, 23, 47, 33, 62, 4, 20, 22, 26, 9],  # 否、艮、剥、困、遯、小過、蒙、観、賁、大畜、小畜
    "危機・困難": [29, 39, 47, 3, 6, 36, 64, 59, 7, 48, 21],  # 坎、蹇、困、屯、訟、明夷、未済、渙、師、井、噬嗑
    "成長期・好調": [1, 14, 35, 34, 55, 42, 17, 30, 25, 10, 9, 21],  # 乾、大有、晋、大壮、豊、益、随、離、无妄、履、小畜、噬嗑
    "混乱・不安定": [59, 64, 4, 6, 38, 54, 28, 21, 43, 44, 48],  # 渙、未済、蒙、訟、睽、帰妹、大過、噬嗑、夬、姤、井
    "衰退期・下降": [23, 36, 33, 12, 20, 4, 52, 62, 41, 9, 48],  # 剥、明夷、遯、否、観、蒙、艮、小過、損、小畜、井
    "変革期": [49, 50, 18, 24, 51, 40, 63, 64, 27, 56, 21],  # 革、鼎、蠱、復、震、解、既済、未済、頤、旅、噬嗑
}

# after_state → 卦の関連付け（拡張版、全64卦カバー）
AFTER_STATE_HINTS: Dict[str, List[int]] = {
    "安定・平和": [11, 32, 63, 2, 15, 37, 8, 53, 60, 48, 9],  # 泰、恒、既済、坤、謙、家人、比、漸、節、井、小畜
    "V字回復・大成功": [1, 14, 55, 35, 34, 42, 17, 25, 30, 43, 21],  # 乾、大有、豊、晋、大壮、益、随、无妄、離、夬、噬嗑
    "緩やかな成長": [53, 46, 42, 11, 32, 57, 16, 58, 61, 19, 9, 48],  # 漸、升、益、泰、恒、巽、豫、兌、中孚、臨、小畜、井
    "衰退・失敗": [23, 36, 12, 47, 29, 39, 6, 38, 54, 44, 21],  # 剥、明夷、否、困、坎、蹇、訟、睽、帰妹、姤、噬嗑
    "変質・新生": [49, 24, 50, 64, 51, 40, 18, 3, 21, 27, 48],  # 革、復、鼎、未済、震、解、蠱、屯、噬嗑、頤、井
    "完全崩壊": [23, 2, 36, 29, 47, 39, 4, 12, 33, 28, 9],  # 剥、坤、明夷、坎、困、蹇、蒙、否、遯、大過、小畜
    "維持・現状維持": [52, 62, 15, 32, 20, 22, 26, 57, 9, 41, 48, 21],  # 艮、小過、謙、恒、観、賁、大畜、巽、小畜、損、井、噬嗑
}

# ==============================================================================
# 爻位判定ルール（改善版v2）
# ==============================================================================

# 爻位判定スコア定数
YAO_PATTERN_SCORE = 2.0           # pattern_typeからのスコア
YAO_KEYWORD_BASE_SCORE = 3.5      # キーワードマッチスコア（改善: 1.5→3.5）
YAO_SPECIFIC_KEYWORD_SCORE = 5.0  # 爻固有キーワードボーナス（新規）
YAO_MASTER_MATCH_SCORE = 1.5      # yao_masterマッチスコア（改善: 1.0→1.5）
YAO_STATE_HINT_SCORE = 2.0        # before/after_stateからのヒント（新規）

# 希少爻ブースト係数（バランス調整版v2）
RARE_YAO_BOOST: Dict[int, float] = {
    1: 1.6,  # 緩和: 2.8→1.6（目標15%程度）
    2: 1.4,  # 緩和: 2.0→1.4（目標15%程度）
    4: 1.5,  # 新規: 4爻が空セル多いため追加
    6: 1.3,  # 新規: 6爻が空セル多いため追加
}

# パターンタイプ → 爻位の傾向（改善版：1爻・2爻を強化）
PATTERN_YAO_TENDENCY: Dict[str, List[int]] = {
    # 成功パターン：初期段階も重視
    "Steady_Growth": [1, 2, 3, 4, 5],  # 初期〜頂点まで
    "Strategic_Patience": [1, 2, 3, 4],  # 待機〜転換まで
    "Opportunity_Seized": [1, 2, 3, 4, 5],  # 機会は初期から（改善）
    "Collaborative_Rise": [1, 2, 3, 4, 5],  # 協調は初期から（改善）
    "Bold_Leap": [1, 2, 4, 5],  # 大胆な飛躍は準備から（改善）

    # 危機対応：再出発は初期
    "Shock_Recovery": [1, 2, 3, 4, 5],  # 初期〜回復
    "Crisis_Pivot": [1, 2, 3, 4, 5, 6],  # 転換は全段階（改善）
    "Adaptive_Survival": [1, 2, 3, 4, 5, 6],  # 全段階

    # 衰退パターン
    "Hubris_Collapse": [4, 5, 6, 3],  # 転換〜終結
    "Slow_Decline": [3, 4, 5, 6],  # 過渡〜終結
    "Quiet_Fade": [2, 4, 5, 6],  # 成長期も含む（改善）
    "Managed_Decline": [2, 3, 4, 5, 6],  # 成長期も含む（改善）
    "Failed_Attempt": [1, 2, 3, 4, 5, 6],  # 全段階（改善）

    # 変革系
    "Legacy_Burden": [1, 2, 3, 4, 5],  # 継承は初期から（改善）
    "Tech_Disruption": [1, 2, 3, 4, 5, 6],  # 全段階（改善）
    "Market_Shift": [1, 2, 3, 4, 5],  # 初期から（改善）
    "Generational_Change": [1, 2, 3, 4, 5],  # 世代交代は初期から（改善）

    # その他
    "Internal_Conflict": [1, 2, 3, 4, 5],  # 初期から（改善）
    "External_Threat": [1, 2, 3, 4, 5, 6],  # 全段階

    # 追加パターン（データベースにある可能性）
    "Endurance": [1, 2, 3, 4, 5],  # 耐久は初期から
    "Pivot_Success": [1, 2, 3, 4, 5],  # ピボットは初期から
    "Breakthrough": [1, 2, 3, 4, 5],  # ブレイクスルーは初期から
}

# outcome → 爻位の調整（改善版）
OUTCOME_YAO_ADJUSTMENT: Dict[str, Dict[int, float]] = {
    "Success": {
        1: 1.3,  # 成功の始まりも評価（新規）
        2: 1.3,  # 成長初期の成功も評価（新規）
        4: 1.2,  # 転換成功
        5: 1.4,  # 成功は5爻（最盛期）が多い（調整）
        6: 0.6,  # 行き過ぎは少ない
    },
    "Failure": {
        1: 1.2,  # 初期段階での失敗も（新規）
        3: 1.2,  # 分岐点での失敗
        6: 1.5,  # 失敗は6爻（行き過ぎ・転落）
        5: 0.6,  # 最盛期は少ない
    },
    "Mixed": {
        1: 1.2,  # 初期の混合結果（新規）
        2: 1.2,  # 成長期の混合結果（新規）
        3: 1.3,  # 過渡期
        4: 1.3,  # 転換点
    },
}

# 爻位固有キーワード（強化版）- ユーザー相談マッチング用
YAO_SPECIFIC_KEYWORDS: Dict[int, List[str]] = {
    1: [
        # 創業・開始系
        "創業", "起業", "設立", "発足", "開業", "新設", "新規参入", "スタートアップ",
        "立ち上げ", "着手", "開始", "始動", "発起", "起ち上げ",
        # 準備・計画系
        "準備中", "計画中", "構想", "企画段階", "検討中", "模索", "アイデア段階",
        "プレシード", "シード期", "0→1",
        # 時間軸
        "創業時", "設立当初", "初年度", "1年目", "草創期", "黎明期", "揺籃期",
        # 状態
        "まだ〜ない", "これから", "将来", "潜む", "萌芽", "胎動", "兆し",
        "第一歩", "最初の", "初めて",
    ],
    2: [
        # 成長初期系
        "成長", "拡大", "発展", "進展", "前進", "上昇", "向上",
        "軌道に乗り", "定着", "確立", "基盤構築", "基盤固め",
        # 認知系
        "認められ", "評価され", "認知", "注目", "実績を積み", "信頼を得",
        # 資金調達系
        "シリーズA", "シリーズB", "PMF", "プロダクトマーケットフィット",
        # 時間軸
        "2年目", "3年目", "成長期", "発展期", "拡大初期",
        # 組織
        "チーム拡大", "採用", "組織化", "体制整備",
    ],
    3: [
        # 困難・危機系
        "困難", "試練", "危機", "苦境", "苦難", "逆境", "障害", "壁",
        # 分岐・選択系
        "分岐", "岐路", "選択", "決断を迫られ", "転換期", "過渡期",
        # 変化系
        "変化", "転機", "節目", "曲がり角", "ターニングポイント",
        # 挑戦系
        "挑戦", "チャレンジ", "試み", "模索",
    ],
    4: [
        # 飛躍系
        "飛躍", "躍進", "ブレイク", "ブレイクスルー", "急成長", "急拡大",
        # 転換系
        "転換", "変革", "改革", "刷新", "大転換", "方向転換", "ピボット",
        # 決断系
        "決断", "勝負", "賭け", "大勝負", "勝負に出",
        # M&A系
        "買収", "合併", "統合", "M&A", "IPO", "上場",
    ],
    5: [
        # 頂点系
        "頂点", "頂上", "ピーク", "最盛期", "全盛期", "黄金期", "絶頂",
        # 成功系
        "成功", "達成", "実現", "完成", "結実", "大成功", "快挙",
        # リーダーシップ系
        "リーダー", "トップ", "首位", "1位", "シェア1位", "業界最大",
        "支配的", "覇権", "制覇", "独占",
        # 評価系
        "高評価", "受賞", "表彰", "栄誉",
    ],
    6: [
        # 終結系
        "終結", "終焉", "終了", "完了", "閉幕", "幕引き",
        # 衰退系
        "衰退", "没落", "凋落", "下降", "低迷", "斜陽",
        # 崩壊系
        "崩壊", "破綻", "倒産", "破産", "清算", "解散",
        # 過剰系
        "過度", "行き過ぎ", "過剰", "やりすぎ", "暴走",
        # 引退系
        "引退", "退任", "辞任", "撤退", "売却", "事業譲渡",
        # 時間軸
        "末期", "晩年", "終盤", "最終局面", "最後の",
    ],
}

# before_state → 爻位ヒント（新規追加）
BEFORE_STATE_YAO_HINTS: Dict[str, List[int]] = {
    "どん底・危機": [1, 2, 3],      # 危機からの再出発は初期〜中盤
    "停滞・閉塞": [1, 2, 3],        # 停滞打破は初期〜中盤
    "混乱・カオス": [1, 2, 3, 4],   # 混乱期は広範囲
    "成長痛": [2, 3, 4],            # 成長痛は成長〜転換期
    "安定・平和": [2, 3, 4, 5],     # 安定期は中盤〜頂点
    "絶頂・慢心": [5, 6],           # 絶頂からは後半
}

# after_state → 爻位ヒント（新規追加）
AFTER_STATE_YAO_HINTS: Dict[str, List[int]] = {
    "安定・平和": [2, 4, 5],               # 安定への到達
    "安定成長・成功": [2, 4, 5],           # 成功フェーズ
    "持続成長・大成功": [4, 5],            # 大成功は4-5爻
    "V字回復・大成功": [4, 5],             # V字回復
    "停滞・閉塞": [3, 6],                  # 停滞は分岐or終結
    "混乱・カオス": [3, 6],                # 混乱は分岐or終結
    "崩壊・消滅": [6],                     # 崩壊は6爻
    "どん底・危機": [3, 6],                # 危機は分岐or終結
    "縮小安定・生存": [2, 3, 6],           # 縮小は複数
    "変質・新生": [1, 4],                  # 新生は1爻or転換
    "現状維持・延命": [2, 3],              # 維持は2-3爻
}

# 従来のキーワード（互換性のため維持、YAO_SPECIFIC_KEYWORDSを優先）
YAO_KEYWORDS: Dict[int, List[str]] = {
    1: ["始まり", "準備", "潜む", "萌芽", "初期", "着手", "開始", "立ち上げ", "創業", "着想"],
    2: ["成長", "認められ", "進展", "実績", "発展", "拡大", "認知", "定着", "基盤"],
    3: ["困難", "分岐", "過渡", "試練", "危機", "転機", "岐路", "苦難", "挑戦", "転換期"],
    4: ["飛躍", "転換", "決断", "転機", "躍進", "ブレイク", "変革", "大転換", "勝負"],
    5: ["頂点", "成功", "最盛", "リーダー", "達成", "絶頂", "全盛", "頂上", "支配"],
    6: ["終結", "過度", "転落", "衰退", "没落", "崩壊", "終焉", "行き過ぎ", "引退", "撤退"],
}


def load_reference_data() -> Tuple[Dict, Dict]:
    """リファレンスデータを読み込む"""
    with open(HEXAGRAM_MASTER_PATH, "r", encoding="utf-8") as f:
        hexagram_master = json.load(f)

    with open(YAO_MASTER_PATH, "r", encoding="utf-8") as f:
        yao_master = json.load(f)

    return hexagram_master, yao_master


def judge_hexagram(case: Dict, hexagram_master: Dict) -> int:
    """
    事例の内容から64卦を判定する（改善版v2）

    判定優先度（新規追加あり）:
    0. 卦名完全一致（最優先、+10.0）
    1. 卦固有キーワードマッチ（+3.0/回）
    2. pattern_type からの候補（+5.0〜0.5）
    3. before_state / after_state からの候補（各+2.0）
    4. outcome による重み付け（乗算）
    5. story_summary のキーワードマッチ（+4.0）
    6. 希少卦ブースト（乗算）
    """
    pattern_type = case.get("pattern_type", "")
    outcome = case.get("outcome", "Mixed")
    before_state = case.get("before_state", "")
    after_state = case.get("after_state", "")
    story = case.get("story_summary", "")

    # 候補卦をスコア付きで収集
    candidates: Dict[int, float] = {}

    # 0. 卦名完全一致ボーナス（最優先）- 新規追加
    for hex_id, aliases in HEXAGRAM_NAME_ALIASES.items():
        for alias in aliases:
            if alias in story:
                candidates[hex_id] = candidates.get(hex_id, 0) + HEXAGRAM_NAME_BONUS
                break  # 同一卦で複数マッチしても1回分

    # 1. 卦固有キーワードマッチ - 新規追加
    for hex_id, keywords in HEXAGRAM_SPECIFIC_KEYWORDS.items():
        match_count = sum(1 for kw in keywords if kw in story)
        if match_count > 0:
            candidates[hex_id] = candidates.get(hex_id, 0) + (SPECIFIC_KEYWORD_SCORE * min(match_count, 3))

    # 2. pattern_type からの候補
    if pattern_type in PATTERN_TO_HEXAGRAMS:
        for i, hex_id in enumerate(PATTERN_TO_HEXAGRAMS[pattern_type]):
            # 優先度に応じてスコア（先頭ほど高い）
            score = max(PATTERN_BASE_SCORE - i * PATTERN_DECAY, 0.5)
            candidates[hex_id] = candidates.get(hex_id, 0) + score

    # 3. before_state からの候補
    if before_state in BEFORE_STATE_HINTS:
        for hex_id in BEFORE_STATE_HINTS[before_state]:
            candidates[hex_id] = candidates.get(hex_id, 0) + STATE_SCORE

    # 4. after_state からの候補
    if after_state in AFTER_STATE_HINTS:
        for hex_id in AFTER_STATE_HINTS[after_state]:
            candidates[hex_id] = candidates.get(hex_id, 0) + STATE_SCORE

    # 5. outcome による重み調整
    if outcome in OUTCOME_HEXAGRAM_WEIGHTS:
        for hex_id, weight in OUTCOME_HEXAGRAM_WEIGHTS[outcome].items():
            if hex_id in candidates:
                candidates[hex_id] *= weight

    # 6. story_summary からキーワードマッチ（重み増加版）
    for hex_id_str, hex_data in hexagram_master.items():
        hex_id = int(hex_id_str)
        keyword = hex_data.get("keyword", "")
        meaning = hex_data.get("meaning", "")

        # キーワードがstoryに含まれるか（重み増加: 1.5→4.0）
        keywords = keyword.split("・")
        for kw in keywords:
            if kw and kw in story:
                candidates[hex_id] = candidates.get(hex_id, 0) + KEYWORD_BASE_SCORE

        # meaningの一部がstoryに含まれるか（重み増加: 0.5→1.0）
        if len(meaning) > 4:
            for phrase in [meaning[i:i+4] for i in range(0, len(meaning)-3, 2)]:
                if phrase in story:
                    candidates[hex_id] = candidates.get(hex_id, 0) + MEANING_PHRASE_SCORE

    # 7. 希少卦ブースト適用 - 新規追加
    for hex_id in list(candidates.keys()):
        if hex_id in RARE_HEXAGRAM_BOOST:
            candidates[hex_id] *= RARE_HEXAGRAM_BOOST[hex_id]

    # スコアが高い順にソート
    if candidates:
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        # 上位3つの中からランダムに選択（多様性確保）
        top_candidates = sorted_candidates[:min(3, len(sorted_candidates))]
        weights = [c[1] for c in top_candidates]
        total = sum(weights)
        weights = [w/total for w in weights]

        # 重み付きランダム選択
        r = random.random()
        cumulative = 0
        for (hex_id, score), weight in zip(top_candidates, weights):
            cumulative += weight
            if r < cumulative:
                return hex_id
        return top_candidates[0][0]

    # フォールバック: outcomeベースでデフォルト卦を返す
    default_hexagrams = {
        "Success": 11,  # 地天泰
        "Failure": 12,  # 天地否
        "Mixed": 63,    # 水火既済
    }
    return default_hexagrams.get(outcome, 11)


def judge_yao_position(case: Dict, hexagram_id: int, yao_master: Dict) -> int:
    """
    事例の内容から爻位（1-6）を判定する（強化版v2）

    判定優先度:
    0. 爻固有キーワードマッチ（最優先、+5.0/回）
    1. pattern_type からの傾向（+2.0）
    2. before_state / after_state からのヒント（各+2.0）
    3. outcome による調整（乗算）
    4. 従来キーワードマッチ（+3.5）
    5. 卦固有の爻位意味からマッチ（+1.5）
    6. 希少爻ブースト（1爻: 2.8x, 2爻: 2.0x）
    """
    pattern_type = case.get("pattern_type", "")
    outcome = case.get("outcome", "Mixed")
    before_state = case.get("before_state", "")
    after_state = case.get("after_state", "")
    story = case.get("story_summary", "")
    target_name = case.get("target_name", "")
    combined_text = story + " " + target_name

    # 爻位のスコア（初期値を低めに設定して差をつけやすくする）
    yao_scores: Dict[int, float] = {i: 0.5 for i in range(1, 7)}

    # 0. 爻固有キーワードマッチ（最優先）- 新規
    for yao, keywords in YAO_SPECIFIC_KEYWORDS.items():
        match_count = sum(1 for kw in keywords if kw in combined_text)
        if match_count > 0:
            # 最大3回分までカウント（過剰なマッチを防ぐ）
            yao_scores[yao] += YAO_SPECIFIC_KEYWORD_SCORE * min(match_count, 3)

    # 1. pattern_type からの傾向
    if pattern_type in PATTERN_YAO_TENDENCY:
        for yao in PATTERN_YAO_TENDENCY[pattern_type]:
            yao_scores[yao] += YAO_PATTERN_SCORE

    # 2. before_state からのヒント - 新規
    if before_state in BEFORE_STATE_YAO_HINTS:
        for yao in BEFORE_STATE_YAO_HINTS[before_state]:
            yao_scores[yao] += YAO_STATE_HINT_SCORE

    # 3. after_state からのヒント - 新規
    if after_state in AFTER_STATE_YAO_HINTS:
        for yao in AFTER_STATE_YAO_HINTS[after_state]:
            yao_scores[yao] += YAO_STATE_HINT_SCORE

    # 4. outcome による調整（乗算）
    if outcome in OUTCOME_YAO_ADJUSTMENT:
        for yao, weight in OUTCOME_YAO_ADJUSTMENT[outcome].items():
            yao_scores[yao] *= weight

    # 5. 従来のキーワードマッチ（互換性維持、重み増加）
    for yao, keywords in YAO_KEYWORDS.items():
        for kw in keywords:
            if kw in combined_text:
                yao_scores[yao] += YAO_KEYWORD_BASE_SCORE

    # 6. 卦固有の爻位意味からマッチ（yao_master参照）
    hex_str = str(hexagram_id)
    if hex_str in yao_master:
        yao_data = yao_master[hex_str].get("yao", {})
        for yao_str, yao_info in yao_data.items():
            yao_num = int(yao_str)
            modern = yao_info.get("modern", "")
            sns = yao_info.get("sns_style", "")

            # modernの一部がstoryに含まれるか
            for phrase in modern.split():
                if len(phrase) >= 2 and phrase in story:
                    yao_scores[yao_num] += YAO_MASTER_MATCH_SCORE

            # sns_styleもチェック
            if sns and len(sns) > 3:
                for phrase in sns.split():
                    if len(phrase) >= 2 and phrase in story:
                        yao_scores[yao_num] += YAO_MASTER_MATCH_SCORE * 0.5

    # 7. 希少爻ブースト適用（1爻・2爻の過少対策）- 新規
    for yao in list(yao_scores.keys()):
        if yao in RARE_YAO_BOOST:
            yao_scores[yao] *= RARE_YAO_BOOST[yao]

    # スコアが高い順にソート
    sorted_yao = sorted(yao_scores.items(), key=lambda x: x[1], reverse=True)

    # 上位3つの中から重み付きランダム選択（多様性確保）
    top_yao = sorted_yao[:3]
    weights = [max(y[1], 0.1) for y in top_yao]  # 最低値を保証
    total = sum(weights)
    weights = [w/total for w in weights]

    r = random.random()
    cumulative = 0
    for (yao, score), weight in zip(top_yao, weights):
        cumulative += weight
        if r < cumulative:
            return yao
    return top_yao[0][0]


def process_case(case: Dict, hexagram_master: Dict, yao_master: Dict) -> Dict:
    """
    1件の事例を処理し、hexagram_idとyao_positionを判定・更新する
    """
    # 64卦を判定
    hexagram_id = judge_hexagram(case, hexagram_master)

    # 爻位を判定
    yao_position = judge_yao_position(case, hexagram_id, yao_master)

    # 卦名を取得
    hex_str = str(hexagram_id)
    hexagram_name = hexagram_master.get(hex_str, {}).get("name", f"卦{hexagram_id}")

    # 更新
    case["hexagram_id"] = hexagram_id
    case["hexagram_name"] = hexagram_name
    case["changing_lines_2"] = [yao_position]

    return case


def process_all_cases(
    input_path: Path,
    output_path: Path,
    hexagram_master: Dict,
    yao_master: Dict,
    backup: bool = True,
    dry_run: bool = False
) -> Dict:
    """
    全事例を処理してhexagram_idとyao_positionを再判定する
    """
    stats = {
        "total": 0,
        "processed": 0,
        "hexagram_changed": 0,
        "yao_changed": 0,
        "errors": 0,
        "hexagram_distribution": {},
        "yao_distribution": {},
    }

    # バックアップ
    if backup and input_path.exists() and not dry_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = input_path.with_suffix(f".jsonl.bak_aijudge_{timestamp}")
        shutil.copy2(input_path, backup_path)
        print(f"バックアップ作成: {backup_path}")

    cases = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                case = json.loads(line)
                stats["total"] += 1

                old_hexagram = case.get("hexagram_id")
                old_yao = case.get("changing_lines_2", [None])[0] if case.get("changing_lines_2") else None

                # AI判定
                case = process_case(case, hexagram_master, yao_master)
                stats["processed"] += 1

                new_hexagram = case["hexagram_id"]
                new_yao = case["changing_lines_2"][0]

                # 変更カウント
                if old_hexagram != new_hexagram:
                    stats["hexagram_changed"] += 1
                if old_yao != new_yao:
                    stats["yao_changed"] += 1

                # 分布カウント
                stats["hexagram_distribution"][new_hexagram] = \
                    stats["hexagram_distribution"].get(new_hexagram, 0) + 1
                stats["yao_distribution"][new_yao] = \
                    stats["yao_distribution"].get(new_yao, 0) + 1

                cases.append(case)

            except json.JSONDecodeError as e:
                stats["errors"] += 1
                print(f"行 {line_num}: JSON解析エラー: {e}")
            except Exception as e:
                stats["errors"] += 1
                print(f"行 {line_num}: 処理エラー: {e}")

    # 出力
    if not dry_run:
        with open(output_path, "w", encoding="utf-8") as f:
            for case in cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")

    return stats


def print_distribution_report(stats: Dict):
    """分布レポートを表示"""
    print("\n" + "=" * 70)
    print("AI判定結果レポート")
    print("=" * 70)

    print(f"\n処理件数: {stats['total']:,}")
    print(f"成功: {stats['processed']:,}")
    print(f"hexagram_id変更: {stats['hexagram_changed']:,}")
    print(f"yao_position変更: {stats['yao_changed']:,}")
    print(f"エラー: {stats['errors']:,}")

    # 64卦分布
    print("\n--- 64卦分布（上位20件）---")
    hex_dist = sorted(stats["hexagram_distribution"].items(), key=lambda x: x[1], reverse=True)
    for hex_id, count in hex_dist[:20]:
        pct = count / stats["processed"] * 100 if stats["processed"] > 0 else 0
        print(f"  卦{hex_id:02d}: {count:5,} ({pct:5.2f}%)")

    # 384セル（64卦×6爻）カバレッジ
    covered_cells = set()
    # TODO: 実際の分布からカバレッジを計算

    # 爻位分布
    print("\n--- 爻位分布 ---")
    for yao in range(1, 7):
        count = stats["yao_distribution"].get(yao, 0)
        pct = count / stats["processed"] * 100 if stats["processed"] > 0 else 0
        print(f"  {yao}爻: {count:5,} ({pct:5.2f}%)")

    print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AI判定による64卦+爻位の再計算"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/raw/cases.jsonl",
        help="入力ファイルパス"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="出力ファイルパス（デフォルト: 入力と同じ）"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="バックアップを作成しない"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実際には書き込まず、統計のみ表示"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="乱数シード（再現性のため）"
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / (args.output or args.input)

    print("=" * 70)
    print("AI判定スクリプト: 64卦 + 爻位の同時判定")
    print("=" * 70)
    print(f"入力: {input_path}")
    print(f"出力: {output_path}")
    print(f"バックアップ: {'なし' if args.no_backup else 'あり'}")
    print(f"モード: {'ドライラン' if args.dry_run else '実行'}")
    if args.seed is not None:
        print(f"乱数シード: {args.seed}")
    print()

    if not input_path.exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_path}")
        return

    # リファレンスデータ読み込み
    print("リファレンスデータ読み込み中...")
    hexagram_master, yao_master = load_reference_data()
    print(f"  64卦マスタ: {len(hexagram_master)}件")
    print(f"  爻マスタ: {len(yao_master)}件")

    # 処理実行
    print("\n処理中...")
    stats = process_all_cases(
        input_path,
        output_path,
        hexagram_master,
        yao_master,
        backup=not args.no_backup,
        dry_run=args.dry_run
    )

    # レポート表示
    print_distribution_report(stats)

    if not args.dry_run:
        print(f"\n完了: {output_path} に保存しました")


if __name__ == "__main__":
    main()
