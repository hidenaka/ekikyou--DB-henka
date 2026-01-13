#!/usr/bin/env python3
"""
ルーブリックv1生成スクリプト

384クラス（64卦×6爻）の参照分布を生成する。
既存のhexagram_master.jsonとdiagnose-v2.tsのデータを活用。
"""

import json
from pathlib import Path
from typing import Dict, List, Any

# ============================================================
# 軸の定義（Axis Rules）
# ============================================================

AXIS_RULES = {
    "changeNature": {
        "description": "変化の性質を表す軸。拡大/収縮/維持/転換の4カテゴリ",
        "categories": {
            "拡大": {
                "definition": "新しいことを始める、規模を大きくする、可能性を広げる変化",
                "iChingKeywords": ["創造", "発展", "進む", "増やす", "始まり", "繁栄", "上昇"],
                "exampleHexagrams": [1, 11, 14, 35, 42, 55]
            },
            "収縮": {
                "definition": "手放す、減らす、集中する、終わらせる変化",
                "iChingKeywords": ["退く", "損", "減らす", "剥落", "困窮", "節度"],
                "exampleHexagrams": [9, 12, 23, 33, 41, 47]
            },
            "維持": {
                "definition": "今の状態を保つ、安定させる、守る変化",
                "iChingKeywords": ["安定", "持続", "守る", "恒常", "静止", "養う"],
                "exampleHexagrams": [2, 5, 15, 32, 48, 52]
            },
            "転換": {
                "definition": "根本的に変える、別の道を選ぶ、リセットする変化",
                "iChingKeywords": ["革命", "改革", "変革", "解放", "震動", "背反"],
                "exampleHexagrams": [6, 18, 28, 40, 49, 51]
            }
        }
    },
    "agency": {
        "description": "主体性を表す軸。自ら動く/受け止める/待つの3カテゴリ",
        "categories": {
            "自ら動く": {
                "definition": "主導権を握り、積極的に行動する姿勢",
                "iChingKeywords": ["剛健", "積極", "前進", "決断", "主導"],
                "exampleHexagrams": [1, 7, 14, 34, 43]
            },
            "受け止める": {
                "definition": "流れに従い、柔軟に対応する姿勢",
                "iChingKeywords": ["柔順", "受容", "従う", "適応", "随う"],
                "exampleHexagrams": [2, 8, 15, 17, 22]
            },
            "待つ": {
                "definition": "時機を見計らい、様子を見る姿勢",
                "iChingKeywords": ["待機", "忍耐", "静止", "観察", "蓄える"],
                "exampleHexagrams": [5, 20, 23, 26, 52]
            }
        }
    },
    "timeframe": {
        "description": "時間軸を表す軸。即時/短期/中期/長期の4カテゴリ",
        "categories": {
            "即時": {
                "definition": "今すぐ、緊急性が高い、すぐに結果が必要",
                "iChingKeywords": ["緊急", "今", "速やか", "即座"],
                "exampleHexagrams": [1, 6, 40, 43, 49]
            },
            "短期": {
                "definition": "数週間〜3ヶ月程度の短期視点",
                "iChingKeywords": ["近い", "間もなく", "接近"],
                "exampleHexagrams": [3, 19, 31, 44]
            },
            "中期": {
                "definition": "数ヶ月〜1年程度の中期視点",
                "iChingKeywords": ["段階的", "着実", "漸進"],
                "exampleHexagrams": [5, 8, 35, 46, 53]
            },
            "長期": {
                "definition": "1年以上の長期視点",
                "iChingKeywords": ["長期", "恒久", "持続", "忍耐"],
                "exampleHexagrams": [2, 26, 32, 48, 52]
            }
        }
    },
    "relationship": {
        "description": "関係性の範囲を表す軸。個人/組織内/対外の3カテゴリ",
        "categories": {
            "個人": {
                "definition": "自分自身や家族など、個人的な範囲",
                "iChingKeywords": ["自己", "内省", "個人", "家庭"],
                "exampleHexagrams": [4, 27, 37, 52]
            },
            "組織内": {
                "definition": "同僚・チーム・組織全体など、組織内部の範囲",
                "iChingKeywords": ["組織", "統率", "団結", "協力"],
                "exampleHexagrams": [7, 8, 13, 45]
            },
            "対外": {
                "definition": "顧客・取引先・業界・社会など、外部との関係",
                "iChingKeywords": ["外部", "交流", "国際", "社会"],
                "exampleHexagrams": [10, 19, 30, 31, 57]
            }
        }
    },
    "emotionalTone": {
        "description": "感情基調を表す軸。前向き/慎重/不安/楽観の4カテゴリ",
        "categories": {
            "前向き": {
                "definition": "ワクワク・期待感・積極的な感情",
                "iChingKeywords": ["喜び", "明るい", "積極", "希望"],
                "exampleHexagrams": [1, 11, 31, 58]
            },
            "慎重": {
                "definition": "用心深さ・注意深さ・控えめな感情",
                "iChingKeywords": ["慎重", "注意", "謙虚", "控えめ"],
                "exampleHexagrams": [5, 10, 15, 33, 52]
            },
            "不安": {
                "definition": "心配・恐れ・危機感を伴う感情",
                "iChingKeywords": ["困難", "危険", "苦難", "試練"],
                "exampleHexagrams": [3, 6, 29, 39, 47]
            },
            "楽観": {
                "definition": "なんとかなる感・信頼・希望に満ちた感情",
                "iChingKeywords": ["楽観", "信頼", "繁栄", "成功"],
                "exampleHexagrams": [14, 16, 24, 35, 55]
            }
        }
    }
}

# ============================================================
# 64卦のベースデータ（diagnose-v2.tsから転記）
# ============================================================

HEXAGRAM_BASE = {
    1: {"name": "乾為天", "changeType": "expansion", "agencyType": "active", "timeHorizon": "immediate", "relationScope": "organizational", "emotionalQuality": "positive"},
    2: {"name": "坤為地", "changeType": "stability", "agencyType": "receptive", "timeHorizon": "long", "relationScope": "personal", "emotionalQuality": "cautious"},
    3: {"name": "水雷屯", "changeType": "expansion", "agencyType": "waiting", "timeHorizon": "medium", "relationScope": "personal", "emotionalQuality": "anxious"},
    4: {"name": "山水蒙", "changeType": "expansion", "agencyType": "receptive", "timeHorizon": "long", "relationScope": "personal", "emotionalQuality": "cautious"},
    5: {"name": "水天需", "changeType": "stability", "agencyType": "waiting", "timeHorizon": "medium", "relationScope": "personal", "emotionalQuality": "cautious"},
    6: {"name": "天水訟", "changeType": "transformation", "agencyType": "active", "timeHorizon": "immediate", "relationScope": "external", "emotionalQuality": "anxious"},
    7: {"name": "地水師", "changeType": "expansion", "agencyType": "active", "timeHorizon": "medium", "relationScope": "organizational", "emotionalQuality": "positive"},
    8: {"name": "水地比", "changeType": "stability", "agencyType": "receptive", "timeHorizon": "medium", "relationScope": "organizational", "emotionalQuality": "positive"},
    9: {"name": "風天小畜", "changeType": "contraction", "agencyType": "waiting", "timeHorizon": "medium", "relationScope": "personal", "emotionalQuality": "cautious"},
    10: {"name": "天沢履", "changeType": "stability", "agencyType": "active", "timeHorizon": "immediate", "relationScope": "external", "emotionalQuality": "cautious"},
    11: {"name": "地天泰", "changeType": "expansion", "agencyType": "active", "timeHorizon": "medium", "relationScope": "organizational", "emotionalQuality": "optimistic"},
    12: {"name": "天地否", "changeType": "contraction", "agencyType": "waiting", "timeHorizon": "long", "relationScope": "organizational", "emotionalQuality": "anxious"},
    13: {"name": "天火同人", "changeType": "expansion", "agencyType": "active", "timeHorizon": "medium", "relationScope": "organizational", "emotionalQuality": "positive"},
    14: {"name": "火天大有", "changeType": "expansion", "agencyType": "active", "timeHorizon": "immediate", "relationScope": "organizational", "emotionalQuality": "optimistic"},
    15: {"name": "地山謙", "changeType": "stability", "agencyType": "receptive", "timeHorizon": "long", "relationScope": "personal", "emotionalQuality": "cautious"},
    16: {"name": "雷地予", "changeType": "expansion", "agencyType": "active", "timeHorizon": "immediate", "relationScope": "organizational", "emotionalQuality": "optimistic"},
    17: {"name": "沢雷随", "changeType": "stability", "agencyType": "receptive", "timeHorizon": "medium", "relationScope": "personal", "emotionalQuality": "positive"},
    18: {"name": "山風蠱", "changeType": "transformation", "agencyType": "active", "timeHorizon": "medium", "relationScope": "organizational", "emotionalQuality": "cautious"},
    19: {"name": "地沢臨", "changeType": "expansion", "agencyType": "active", "timeHorizon": "immediate", "relationScope": "external", "emotionalQuality": "positive"},
    20: {"name": "風地観", "changeType": "stability", "agencyType": "waiting", "timeHorizon": "long", "relationScope": "external", "emotionalQuality": "cautious"},
    21: {"name": "火雷噬嗑", "changeType": "transformation", "agencyType": "active", "timeHorizon": "immediate", "relationScope": "organizational", "emotionalQuality": "positive"},
    22: {"name": "山火賁", "changeType": "stability", "agencyType": "receptive", "timeHorizon": "medium", "relationScope": "personal", "emotionalQuality": "positive"},
    23: {"name": "山地剥", "changeType": "contraction", "agencyType": "waiting", "timeHorizon": "long", "relationScope": "personal", "emotionalQuality": "anxious"},
    24: {"name": "地雷復", "changeType": "expansion", "agencyType": "waiting", "timeHorizon": "medium", "relationScope": "personal", "emotionalQuality": "optimistic"},
    25: {"name": "天雷无妄", "changeType": "stability", "agencyType": "receptive", "timeHorizon": "immediate", "relationScope": "personal", "emotionalQuality": "positive"},
    26: {"name": "山天大畜", "changeType": "expansion", "agencyType": "waiting", "timeHorizon": "long", "relationScope": "organizational", "emotionalQuality": "cautious"},
    27: {"name": "山雷頤", "changeType": "stability", "agencyType": "receptive", "timeHorizon": "long", "relationScope": "personal", "emotionalQuality": "cautious"},
    28: {"name": "沢風大過", "changeType": "transformation", "agencyType": "active", "timeHorizon": "immediate", "relationScope": "organizational", "emotionalQuality": "anxious"},
    29: {"name": "坎為水", "changeType": "contraction", "agencyType": "waiting", "timeHorizon": "medium", "relationScope": "personal", "emotionalQuality": "anxious"},
    30: {"name": "離為火", "changeType": "expansion", "agencyType": "active", "timeHorizon": "immediate", "relationScope": "external", "emotionalQuality": "positive"},
    31: {"name": "沢山咸", "changeType": "expansion", "agencyType": "receptive", "timeHorizon": "immediate", "relationScope": "external", "emotionalQuality": "positive"},
    32: {"name": "雷風恒", "changeType": "stability", "agencyType": "active", "timeHorizon": "long", "relationScope": "personal", "emotionalQuality": "cautious"},
    33: {"name": "天山遯", "changeType": "contraction", "agencyType": "waiting", "timeHorizon": "immediate", "relationScope": "personal", "emotionalQuality": "cautious"},
    34: {"name": "雷天大壮", "changeType": "expansion", "agencyType": "active", "timeHorizon": "immediate", "relationScope": "organizational", "emotionalQuality": "optimistic"},
    35: {"name": "火地晋", "changeType": "expansion", "agencyType": "active", "timeHorizon": "medium", "relationScope": "organizational", "emotionalQuality": "optimistic"},
    36: {"name": "地火明夷", "changeType": "contraction", "agencyType": "waiting", "timeHorizon": "long", "relationScope": "personal", "emotionalQuality": "anxious"},
    37: {"name": "風火家人", "changeType": "stability", "agencyType": "receptive", "timeHorizon": "long", "relationScope": "personal", "emotionalQuality": "positive"},
    38: {"name": "火沢睽", "changeType": "transformation", "agencyType": "waiting", "timeHorizon": "medium", "relationScope": "external", "emotionalQuality": "cautious"},
    39: {"name": "水山蹇", "changeType": "contraction", "agencyType": "waiting", "timeHorizon": "medium", "relationScope": "personal", "emotionalQuality": "anxious"},
    40: {"name": "雷水解", "changeType": "transformation", "agencyType": "active", "timeHorizon": "immediate", "relationScope": "organizational", "emotionalQuality": "optimistic"},
    41: {"name": "山沢損", "changeType": "contraction", "agencyType": "receptive", "timeHorizon": "medium", "relationScope": "personal", "emotionalQuality": "cautious"},
    42: {"name": "風雷益", "changeType": "expansion", "agencyType": "active", "timeHorizon": "immediate", "relationScope": "organizational", "emotionalQuality": "optimistic"},
    43: {"name": "沢天夬", "changeType": "transformation", "agencyType": "active", "timeHorizon": "immediate", "relationScope": "organizational", "emotionalQuality": "positive"},
    44: {"name": "天風姤", "changeType": "transformation", "agencyType": "receptive", "timeHorizon": "immediate", "relationScope": "external", "emotionalQuality": "cautious"},
    45: {"name": "沢地萃", "changeType": "expansion", "agencyType": "active", "timeHorizon": "medium", "relationScope": "organizational", "emotionalQuality": "positive"},
    46: {"name": "地風升", "changeType": "expansion", "agencyType": "active", "timeHorizon": "medium", "relationScope": "organizational", "emotionalQuality": "optimistic"},
    47: {"name": "沢水困", "changeType": "contraction", "agencyType": "waiting", "timeHorizon": "medium", "relationScope": "personal", "emotionalQuality": "anxious"},
    48: {"name": "水風井", "changeType": "stability", "agencyType": "receptive", "timeHorizon": "long", "relationScope": "organizational", "emotionalQuality": "cautious"},
    49: {"name": "沢火革", "changeType": "transformation", "agencyType": "active", "timeHorizon": "immediate", "relationScope": "organizational", "emotionalQuality": "positive"},
    50: {"name": "火風鼎", "changeType": "stability", "agencyType": "active", "timeHorizon": "medium", "relationScope": "organizational", "emotionalQuality": "positive"},
    51: {"name": "震為雷", "changeType": "transformation", "agencyType": "active", "timeHorizon": "immediate", "relationScope": "personal", "emotionalQuality": "anxious"},
    52: {"name": "艮為山", "changeType": "stability", "agencyType": "waiting", "timeHorizon": "long", "relationScope": "personal", "emotionalQuality": "cautious"},
    53: {"name": "風山漸", "changeType": "expansion", "agencyType": "active", "timeHorizon": "long", "relationScope": "personal", "emotionalQuality": "cautious"},
    54: {"name": "雷沢帰妹", "changeType": "stability", "agencyType": "receptive", "timeHorizon": "medium", "relationScope": "external", "emotionalQuality": "cautious"},
    55: {"name": "雷火豊", "changeType": "expansion", "agencyType": "active", "timeHorizon": "immediate", "relationScope": "organizational", "emotionalQuality": "optimistic"},
    56: {"name": "火山旅", "changeType": "transformation", "agencyType": "active", "timeHorizon": "medium", "relationScope": "external", "emotionalQuality": "cautious"},
    57: {"name": "巽為風", "changeType": "expansion", "agencyType": "receptive", "timeHorizon": "long", "relationScope": "external", "emotionalQuality": "positive"},
    58: {"name": "兌為沢", "changeType": "expansion", "agencyType": "active", "timeHorizon": "immediate", "relationScope": "external", "emotionalQuality": "optimistic"},
    59: {"name": "風水渙", "changeType": "transformation", "agencyType": "active", "timeHorizon": "medium", "relationScope": "organizational", "emotionalQuality": "positive"},
    60: {"name": "水沢節", "changeType": "contraction", "agencyType": "receptive", "timeHorizon": "medium", "relationScope": "personal", "emotionalQuality": "cautious"},
    61: {"name": "風沢中孚", "changeType": "stability", "agencyType": "active", "timeHorizon": "long", "relationScope": "external", "emotionalQuality": "positive"},
    62: {"name": "雷山小過", "changeType": "contraction", "agencyType": "receptive", "timeHorizon": "immediate", "relationScope": "personal", "emotionalQuality": "cautious"},
    63: {"name": "水火既済", "changeType": "stability", "agencyType": "waiting", "timeHorizon": "medium", "relationScope": "organizational", "emotionalQuality": "positive"},
    64: {"name": "火水未済", "changeType": "expansion", "agencyType": "active", "timeHorizon": "long", "relationScope": "organizational", "emotionalQuality": "cautious"},
}

# ============================================================
# 爻の修正係数
# ============================================================

# 爻による時間軸の修正
YAO_TIMEFRAME_MODIFIER = {
    1: {"即時": -0.1, "短期": +0.1, "中期": +0.05, "長期": +0.05},  # 初爻: まだ動くな、準備段階
    2: {"即時": +0.0, "短期": +0.1, "中期": +0.05, "長期": -0.05},  # 二爻: 発展初期
    3: {"即時": +0.1, "短期": +0.05, "中期": -0.05, "長期": -0.1},  # 三爻: 危険、転換点
    4: {"即時": +0.05, "短期": +0.05, "中期": +0.0, "長期": -0.1},   # 四爻: 上位への移行
    5: {"即時": +0.1, "短期": +0.0, "中期": -0.05, "長期": -0.05},   # 五爻: 頂点、決断
    6: {"即時": -0.1, "短期": -0.05, "中期": +0.05, "長期": +0.1},   # 上爻: 終末、次へ
}

# 爻による主体性の修正
YAO_AGENCY_MODIFIER = {
    1: {"自ら動く": -0.15, "受け止める": +0.05, "待つ": +0.1},  # 初爻: まだ動くな
    2: {"自ら動く": +0.0, "受け止める": +0.1, "待つ": -0.1},     # 二爻: 柔軟に対応
    3: {"自ら動く": +0.1, "受け止める": -0.05, "待つ": -0.05},   # 三爻: 積極行動か危険
    4: {"自ら動く": -0.05, "受け止める": +0.1, "待つ": -0.05},   # 四爻: 慎重に
    5: {"自ら動く": +0.15, "受け止める": -0.05, "待つ": -0.1},   # 五爻: リーダーシップ
    6: {"自ら動く": -0.1, "受け止める": +0.05, "待つ": +0.05},   # 上爻: 引退、手放す
}

# 爻による感情の修正
YAO_EMOTION_MODIFIER = {
    1: {"前向き": -0.05, "慎重": +0.1, "不安": +0.05, "楽観": -0.1},   # 初爻: 慎重に
    2: {"前向き": +0.05, "慎重": +0.0, "不安": -0.05, "楽観": +0.0},   # 二爻: やや前向き
    3: {"前向き": -0.05, "慎重": +0.0, "不安": +0.1, "楽観": -0.05},   # 三爻: 危険、不安
    4: {"前向き": +0.0, "慎重": +0.05, "不安": +0.0, "楽観": -0.05},   # 四爻: 慎重に
    5: {"前向き": +0.1, "慎重": -0.05, "不安": -0.1, "楽観": +0.05},   # 五爻: 自信
    6: {"前向き": -0.05, "慎重": +0.05, "不安": +0.05, "楽観": -0.05}, # 上爻: 複雑
}

# 爻の名前
YAO_NAMES = {
    1: "初",
    2: "二",
    3: "三",
    4: "四",
    5: "五",
    6: "上"
}

# 爻の段階説明
YAO_STAGES = {
    1: "潜伏期・準備段階 - まだ表に出る時ではない",
    2: "発現期・成長段階 - 徐々に形が見えてくる",
    3: "転換期・危険段階 - 行き過ぎると危険、判断が必要",
    4: "展開期・移行段階 - 新しい局面への移行",
    5: "成就期・頂点段階 - 最も力を発揮できる時",
    6: "終末期・完成段階 - 終わりと次への準備"
}


def type_to_distribution(type_value: str, type_category: str) -> Dict[str, float]:
    """型の値を確率分布に変換"""

    if type_category == "changeNature":
        mapping = {
            "expansion": {"拡大": 0.6, "収縮": 0.05, "維持": 0.15, "転換": 0.2},
            "contraction": {"拡大": 0.05, "収縮": 0.6, "維持": 0.2, "転換": 0.15},
            "stability": {"拡大": 0.15, "収縮": 0.1, "維持": 0.6, "転換": 0.15},
            "transformation": {"拡大": 0.15, "収縮": 0.1, "維持": 0.1, "転換": 0.65}
        }
        return mapping.get(type_value, {"拡大": 0.25, "収縮": 0.25, "維持": 0.25, "転換": 0.25})

    elif type_category == "agency":
        mapping = {
            "active": {"自ら動く": 0.65, "受け止める": 0.2, "待つ": 0.15},
            "receptive": {"自ら動く": 0.2, "受け止める": 0.6, "待つ": 0.2},
            "waiting": {"自ら動く": 0.15, "受け止める": 0.25, "待つ": 0.6}
        }
        return mapping.get(type_value, {"自ら動く": 0.33, "受け止める": 0.34, "待つ": 0.33})

    elif type_category == "timeframe":
        mapping = {
            "immediate": {"即時": 0.55, "短期": 0.25, "中期": 0.15, "長期": 0.05},
            "medium": {"即時": 0.15, "短期": 0.35, "中期": 0.35, "長期": 0.15},
            "long": {"即時": 0.05, "短期": 0.15, "中期": 0.3, "長期": 0.5}
        }
        return mapping.get(type_value, {"即時": 0.25, "短期": 0.25, "中期": 0.25, "長期": 0.25})

    elif type_category == "relationship":
        mapping = {
            "personal": {"個人": 0.65, "組織内": 0.2, "対外": 0.15},
            "organizational": {"個人": 0.15, "組織内": 0.65, "対外": 0.2},
            "external": {"個人": 0.15, "組織内": 0.25, "対外": 0.6}
        }
        return mapping.get(type_value, {"個人": 0.33, "組織内": 0.34, "対外": 0.33})

    elif type_category == "emotionalTone":
        mapping = {
            "positive": {"前向き": 0.5, "慎重": 0.2, "不安": 0.1, "楽観": 0.2},
            "cautious": {"前向き": 0.15, "慎重": 0.55, "不安": 0.15, "楽観": 0.15},
            "anxious": {"前向き": 0.1, "慎重": 0.25, "不安": 0.5, "楽観": 0.15},
            "optimistic": {"前向き": 0.3, "慎重": 0.1, "不安": 0.1, "楽観": 0.5}
        }
        return mapping.get(type_value, {"前向き": 0.25, "慎重": 0.25, "不安": 0.25, "楽観": 0.25})

    return {}


def apply_yao_modifier(base_dist: Dict[str, float], yao: int, axis: str) -> Dict[str, float]:
    """爻の修正係数を適用"""

    if axis == "timeframe":
        modifier = YAO_TIMEFRAME_MODIFIER.get(yao, {})
    elif axis == "agency":
        modifier = YAO_AGENCY_MODIFIER.get(yao, {})
    elif axis == "emotionalTone":
        modifier = YAO_EMOTION_MODIFIER.get(yao, {})
    else:
        return base_dist  # changeNature, relationshipは爻で修正しない

    # 修正を適用
    result = {}
    for key, value in base_dist.items():
        mod = modifier.get(key, 0)
        result[key] = max(0.01, min(0.95, value + mod))  # 0.01〜0.95にクリップ

    # 正規化
    total = sum(result.values())
    for key in result:
        result[key] = round(result[key] / total, 3)

    return result


def generate_class_profile(hexagram: int, yao: int) -> Dict[str, Any]:
    """1つのクラスプロファイルを生成"""

    base = HEXAGRAM_BASE[hexagram]
    class_id = (hexagram - 1) * 6 + yao

    # 陰陽の判定（簡易版: 卦の性質から推定）
    yang_hexagrams = {1, 14, 34, 43}  # 陽が強い卦
    yin_hexagrams = {2, 23, 36, 47}   # 陰が強い卦

    if hexagram in yang_hexagrams:
        yao_type = "九" if yao % 2 == 1 else "六"
    elif hexagram in yin_hexagrams:
        yao_type = "六" if yao % 2 == 0 else "九"
    else:
        yao_type = "九" if yao in [1, 3, 5] else "六"  # デフォルト

    yao_name = YAO_NAMES[yao]
    full_yao_name = f"{yao_name}{yao_type}" if yao < 6 else f"{yao_name}{yao_type}"

    # ベース分布を生成
    change_dist = type_to_distribution(base["changeType"], "changeNature")
    agency_dist = type_to_distribution(base["agencyType"], "agency")
    time_dist = type_to_distribution(base["timeHorizon"], "timeframe")
    relation_dist = type_to_distribution(base["relationScope"], "relationship")
    emotion_dist = type_to_distribution(base["emotionalQuality"], "emotionalTone")

    # 爻による修正を適用
    agency_dist = apply_yao_modifier(agency_dist, yao, "agency")
    time_dist = apply_yao_modifier(time_dist, yao, "timeframe")
    emotion_dist = apply_yao_modifier(emotion_dist, yao, "emotionalTone")

    return {
        "classId": class_id,
        "hexagram": hexagram,
        "yao": yao,
        "name": f"{base['name']} {full_yao_name}",
        "hexagramName": base["name"],
        "yaoName": full_yao_name,
        "yaoStage": YAO_STAGES[yao],
        "distributions": {
            "changeNature": change_dist,
            "agency": agency_dist,
            "timeframe": time_dist,
            "relationship": relation_dist,
            "emotionalTone": emotion_dist
        },
        "rubricVersion": "v1",
        "rubricSource": f"{base['name']}の{yao_name}爻の伝統的解釈に基づく"
    }


def generate_rubric() -> Dict[str, Any]:
    """ルーブリック全体を生成"""

    class_profiles = []

    for hexagram in range(1, 65):
        for yao in range(1, 7):
            profile = generate_class_profile(hexagram, yao)
            class_profiles.append(profile)

    rubric = {
        "version": "v1",
        "createdAt": "2026-01-13",
        "description": "HaQei診断v5用ルーブリック。64卦×6爻=384クラスの参照分布",
        "axisRules": AXIS_RULES,
        "classProfiles": class_profiles,
        "metadata": {
            "totalClasses": 384,
            "hexagramCount": 64,
            "yaoPerHexagram": 6,
            "generationMethod": "既存64卦データ + 爻修正係数による自動生成",
            "validationStatus": "初期版（パイロットテスト前）"
        }
    }

    return rubric


def main():
    """メイン処理"""

    print("ルーブリックv1を生成中...")

    rubric = generate_rubric()

    # 出力先
    output_path = Path(__file__).parent.parent / "data" / "rubric_v1.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(rubric, f, ensure_ascii=False, indent=2)

    print(f"完了: {output_path}")
    print(f"  - 総クラス数: {len(rubric['classProfiles'])}")
    print(f"  - バージョン: {rubric['version']}")

    # サンプル表示
    print("\nサンプル（乾為天 初九）:")
    sample = rubric['classProfiles'][0]
    print(f"  - classId: {sample['classId']}")
    print(f"  - name: {sample['name']}")
    print(f"  - changeNature: {sample['distributions']['changeNature']}")
    print(f"  - agency: {sample['distributions']['agency']}")


if __name__ == "__main__":
    main()
