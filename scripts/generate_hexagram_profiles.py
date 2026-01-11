#!/usr/bin/env python3
"""
64卦特性データベース生成スクリプト
既存の1-16卦に17-64卦を追加
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
PROFILES_FILE = BASE_DIR / "data" / "hexagrams" / "hexagram_profiles.json"

# 17-64卦の定義
HEXAGRAMS_17_64 = {
    "17": {
        "id": 17, "name": "沢雷随", "chinese": "随",
        "keyword": "随順・追従・適応",
        "nature": {"element": "沢雷", "movement": "追従・適応", "timing": "変化への適応期", "warning": "主体性喪失"},
        "business_context": {
            "favorable_actions": ["対話・融合", "守る・維持"],
            "unfavorable_actions": ["攻める・挑戦", "刷新・破壊"],
            "optimal_timing": "トレンドに乗る時期、市場変化への適応",
            "risk_factors": ["盲目的追従", "主体性喪失", "判断力低下"]
        },
        "yao": {
            "1": {"phrase": "官有渝", "meaning": "官が変わる。変化を受け入れよ", "action_modifier": {"対話・融合": 2, "守る・維持": 1}},
            "2": {"phrase": "係小子", "meaning": "小人に係る。選択を誤るな", "action_modifier": {"対話・融合": -1, "守る・維持": 1}},
            "3": {"phrase": "係丈夫", "meaning": "大人に係る。正しい選択", "action_modifier": {"対話・融合": 2, "攻める・挑戦": 0}},
            "4": {"phrase": "随有獲", "meaning": "随って獲るあり。成果", "action_modifier": {"対話・融合": 1, "攻める・挑戦": 1}},
            "5": {"phrase": "孚于嘉", "meaning": "嘉に孚あり。誠意が通じる", "action_modifier": {"対話・融合": 2, "守る・維持": 1}},
            "6": {"phrase": "拘係之", "meaning": "拘束される。行き詰まり", "action_modifier": {"刷新・破壊": 1, "捨てる・撤退": 1}}
        }
    },
    "18": {
        "id": 18, "name": "山風蠱", "chinese": "蠱",
        "keyword": "腐敗・改革・立て直し",
        "nature": {"element": "山風", "movement": "腐敗の除去", "timing": "改革期", "warning": "放置・先送り"},
        "business_context": {
            "favorable_actions": ["刷新・破壊", "攻める・挑戦"],
            "unfavorable_actions": ["逃げる・放置", "守る・維持"],
            "optimal_timing": "組織改革、レガシー問題解決",
            "risk_factors": ["問題の放置", "先送り", "表面的対処"]
        },
        "yao": {
            "1": {"phrase": "幹父之蠱", "meaning": "父の蠱を幹す。先人の問題を正す", "action_modifier": {"刷新・破壊": 2, "攻める・挑戦": 1}},
            "2": {"phrase": "幹母之蠱", "meaning": "母の蠱を幹す。柔軟に対処", "action_modifier": {"対話・融合": 1, "刷新・破壊": 1}},
            "3": {"phrase": "幹父之蠱小有悔", "meaning": "小さな後悔あり。慎重に", "action_modifier": {"刷新・破壊": 1, "守る・維持": 0}},
            "4": {"phrase": "裕父之蠱", "meaning": "父の蠱を裕にす。余裕を持って", "action_modifier": {"守る・維持": 1, "耐える・潜伏": 1}},
            "5": {"phrase": "幹父之蠱用誉", "meaning": "名誉をもって正す。評価される", "action_modifier": {"刷新・破壊": 2, "対話・融合": 1}},
            "6": {"phrase": "不事王侯", "meaning": "王侯に仕えず。独立独歩", "action_modifier": {"分散・スピンオフ": 2, "守る・維持": 0}}
        }
    },
    "19": {
        "id": 19, "name": "地沢臨", "chinese": "臨",
        "keyword": "接近・監督・臨む",
        "nature": {"element": "地沢", "movement": "接近・監督", "timing": "成長期", "warning": "過干渉"},
        "business_context": {
            "favorable_actions": ["攻める・挑戦", "対話・融合"],
            "unfavorable_actions": ["逃げる・放置", "捨てる・撤退"],
            "optimal_timing": "新規参入、市場開拓期",
            "risk_factors": ["過干渉", "焦り", "過度な期待"]
        },
        "yao": {
            "1": {"phrase": "咸臨", "meaning": "感じて臨む。共感をもって", "action_modifier": {"対話・融合": 2, "攻める・挑戦": 1}},
            "2": {"phrase": "咸臨吉无不利", "meaning": "感じて臨む。吉で不利なし", "action_modifier": {"対話・融合": 2, "攻める・挑戦": 1}},
            "3": {"phrase": "甘臨", "meaning": "甘く臨む。甘すぎる対応", "action_modifier": {"守る・維持": 0, "対話・融合": -1}},
            "4": {"phrase": "至臨", "meaning": "至って臨む。全力で", "action_modifier": {"攻める・挑戦": 2, "対話・融合": 1}},
            "5": {"phrase": "知臨", "meaning": "知をもって臨む。賢明な対応", "action_modifier": {"対話・融合": 2, "守る・維持": 1}},
            "6": {"phrase": "敦臨", "meaning": "敦く臨む。誠実に", "action_modifier": {"対話・融合": 2, "守る・維持": 1}}
        }
    },
    "20": {
        "id": 20, "name": "風地観", "chinese": "観",
        "keyword": "観察・洞察・示す",
        "nature": {"element": "風地", "movement": "観察・示範", "timing": "観察期", "warning": "行動不足"},
        "business_context": {
            "favorable_actions": ["耐える・潜伏", "対話・融合"],
            "unfavorable_actions": ["攻める・挑戦", "刷新・破壊"],
            "optimal_timing": "市場調査、状況把握期",
            "risk_factors": ["観察だけで行動しない", "分析麻痺", "優柔不断"]
        },
        "yao": {
            "1": {"phrase": "童観", "meaning": "童子の観。浅い観察", "action_modifier": {"耐える・潜伏": 1, "攻める・挑戦": -2}},
            "2": {"phrase": "闘観", "meaning": "覗き見る観。狭い視野", "action_modifier": {"耐える・潜伏": 1, "守る・維持": 1}},
            "3": {"phrase": "観我生", "meaning": "我が生を観る。自己反省", "action_modifier": {"守る・維持": 1, "耐える・潜伏": 1}},
            "4": {"phrase": "観国之光", "meaning": "国の光を観る。大局を見る", "action_modifier": {"対話・融合": 2, "攻める・挑戦": 0}},
            "5": {"phrase": "観我生", "meaning": "我が生を観る。深い内省", "action_modifier": {"守る・維持": 2, "対話・融合": 1}},
            "6": {"phrase": "観其生", "meaning": "その生を観る。他者を観察", "action_modifier": {"対話・融合": 1, "守る・維持": 1}}
        }
    },
    "21": {
        "id": 21, "name": "火雷噬嗑", "chinese": "噬嗑",
        "keyword": "断罪・決断・噛み砕く",
        "nature": {"element": "火雷", "movement": "断罪・決断", "timing": "決断期", "warning": "過酷・厳格すぎ"},
        "business_context": {
            "favorable_actions": ["刷新・破壊", "攻める・挑戦"],
            "unfavorable_actions": ["逃げる・放置", "耐える・潜伏"],
            "optimal_timing": "問題解決、障害除去の時期",
            "risk_factors": ["過酷すぎる対応", "独断", "報復"]
        },
        "yao": {
            "1": {"phrase": "屨校滅趾", "meaning": "足かせをはめる。軽い罰", "action_modifier": {"守る・維持": 1, "刷新・破壊": 0}},
            "2": {"phrase": "噬膚滅鼻", "meaning": "皮を噛む。適度な対処", "action_modifier": {"刷新・破壊": 1, "守る・維持": 1}},
            "3": {"phrase": "噬腊肉", "meaning": "乾肉を噛む。困難な対処", "action_modifier": {"耐える・潜伏": 1, "刷新・破壊": 0}},
            "4": {"phrase": "噬乾胏", "meaning": "乾いた骨を噛む。非常に困難", "action_modifier": {"刷新・破壊": 1, "攻める・挑戦": 1}},
            "5": {"phrase": "噬乾肉得黄金", "meaning": "乾肉を噛んで黄金を得る", "action_modifier": {"刷新・破壊": 2, "攻める・挑戦": 1}},
            "6": {"phrase": "何校滅耳", "meaning": "首かせで耳が隠れる。重罰", "action_modifier": {"刷新・破壊": -1, "守る・維持": -1}}
        }
    },
    "22": {
        "id": 22, "name": "山火賁", "chinese": "賁",
        "keyword": "飾り・文化・外観",
        "nature": {"element": "山火", "movement": "装飾・文化", "timing": "仕上げ期", "warning": "虚飾"},
        "business_context": {
            "favorable_actions": ["守る・維持", "対話・融合"],
            "unfavorable_actions": ["攻める・挑戦", "刷新・破壊"],
            "optimal_timing": "ブランディング、仕上げの時期",
            "risk_factors": ["外見だけ", "実質を伴わない", "虚飾"]
        },
        "yao": {
            "1": {"phrase": "賁其趾", "meaning": "足を飾る。基礎を整える", "action_modifier": {"守る・維持": 2, "耐える・潜伏": 1}},
            "2": {"phrase": "賁其須", "meaning": "髭を飾る。上に従う", "action_modifier": {"対話・融合": 1, "守る・維持": 1}},
            "3": {"phrase": "賁如濡如", "meaning": "潤いある飾り。永く貞", "action_modifier": {"守る・維持": 2, "対話・融合": 1}},
            "4": {"phrase": "賁如皤如", "meaning": "白い飾り。純粋", "action_modifier": {"対話・融合": 1, "守る・維持": 1}},
            "5": {"phrase": "賁于丘園", "meaning": "丘園を飾る。質素ながら吉", "action_modifier": {"守る・維持": 2, "耐える・潜伏": 1}},
            "6": {"phrase": "白賁", "meaning": "白い飾り。純粋で咎なし", "action_modifier": {"守る・維持": 2, "対話・融合": 1}}
        }
    },
    "23": {
        "id": 23, "name": "山地剥", "chinese": "剥",
        "keyword": "剥落・衰退・崩壊",
        "nature": {"element": "山地", "movement": "剥落・衰退", "timing": "衰退期", "warning": "無理な抵抗"},
        "business_context": {
            "favorable_actions": ["耐える・潜伏", "捨てる・撤退"],
            "unfavorable_actions": ["攻める・挑戦", "分散・スピンオフ"],
            "optimal_timing": "撤退・縮小の時期",
            "risk_factors": ["無理な抵抗", "現実逃避", "遅すぎる撤退"]
        },
        "yao": {
            "1": {"phrase": "剥牀以足", "meaning": "床の足が剥げる。始まり", "action_modifier": {"耐える・潜伏": 1, "守る・維持": 0}},
            "2": {"phrase": "剥牀以辨", "meaning": "床の枠が剥げる。進行", "action_modifier": {"耐える・潜伏": 1, "捨てる・撤退": 1}},
            "3": {"phrase": "剥之", "meaning": "剥がれる。咎なし", "action_modifier": {"捨てる・撤退": 2, "耐える・潜伏": 0}},
            "4": {"phrase": "剥牀以膚", "meaning": "床の皮が剥げる。危機迫る", "action_modifier": {"捨てる・撤退": 2, "耐える・潜伏": -1}},
            "5": {"phrase": "貫魚以宮人寵", "meaning": "魚を貫く。秩序回復", "action_modifier": {"対話・融合": 2, "守る・維持": 1}},
            "6": {"phrase": "碩果不食", "meaning": "大きな実を食べず。再生の種", "action_modifier": {"耐える・潜伏": 2, "守る・維持": 1}}
        }
    },
    "24": {
        "id": 24, "name": "地雷復", "chinese": "復",
        "keyword": "復活・回帰・再生",
        "nature": {"element": "地雷", "movement": "復活・回帰", "timing": "再起期", "warning": "焦り"},
        "business_context": {
            "favorable_actions": ["攻める・挑戦", "刷新・破壊"],
            "unfavorable_actions": ["逃げる・放置"],
            "optimal_timing": "再起動、復活の時期",
            "risk_factors": ["焦り", "過去への固執", "同じ過ちの繰り返し"]
        },
        "yao": {
            "1": {"phrase": "不遠復", "meaning": "遠からず復る。早い回復", "action_modifier": {"攻める・挑戦": 2, "刷新・破壊": 1}},
            "2": {"phrase": "休復", "meaning": "休んで復る。良い回復", "action_modifier": {"守る・維持": 1, "対話・融合": 1}},
            "3": {"phrase": "頻復", "meaning": "しばしば復る。不安定", "action_modifier": {"耐える・潜伏": 1, "守る・維持": 0}},
            "4": {"phrase": "中行独復", "meaning": "中道を行き独り復る", "action_modifier": {"守る・維持": 1, "耐える・潜伏": 1}},
            "5": {"phrase": "敦復", "meaning": "敦く復る。誠実な回復", "action_modifier": {"守る・維持": 2, "対話・融合": 1}},
            "6": {"phrase": "迷復", "meaning": "迷って復る。凶", "action_modifier": {"攻める・挑戦": -3, "守る・維持": -1}}
        }
    },
    "25": {
        "id": 25, "name": "天雷无妄", "chinese": "无妄",
        "keyword": "無妄・誠実・天真",
        "nature": {"element": "天雷", "movement": "誠実・自然", "timing": "正道期", "warning": "偽り"},
        "business_context": {
            "favorable_actions": ["守る・維持", "対話・融合"],
            "unfavorable_actions": ["逃げる・放置", "刷新・破壊"],
            "optimal_timing": "正道を歩む時期、誠実な経営",
            "risk_factors": ["策略", "不正", "天の理に反する行動"]
        },
        "yao": {
            "1": {"phrase": "无妄往吉", "meaning": "妄なく往けば吉", "action_modifier": {"攻める・挑戦": 2, "守る・維持": 1}},
            "2": {"phrase": "不耕獲", "meaning": "耕さずして獲る。自然の恵み", "action_modifier": {"守る・維持": 2, "耐える・潜伏": 1}},
            "3": {"phrase": "无妄之災", "meaning": "妄なきの災い。不運", "action_modifier": {"耐える・潜伏": 2, "守る・維持": 0}},
            "4": {"phrase": "可貞无咎", "meaning": "貞にすべし、咎なし", "action_modifier": {"守る・維持": 2, "耐える・潜伏": 1}},
            "5": {"phrase": "无妄之疾", "meaning": "妄なきの疾。薬を用いるな", "action_modifier": {"耐える・潜伏": 2, "守る・維持": 1}},
            "6": {"phrase": "无妄行有眚", "meaning": "妄なく行くも災いあり", "action_modifier": {"攻める・挑戦": -2, "耐える・潜伏": 1}}
        }
    },
    "26": {
        "id": 26, "name": "山天大畜", "chinese": "大畜",
        "keyword": "大いに蓄える・止める・育成",
        "nature": {"element": "山天", "movement": "蓄積・止める", "timing": "蓄積期", "warning": "停滞"},
        "business_context": {
            "favorable_actions": ["守る・維持", "耐える・潜伏"],
            "unfavorable_actions": ["分散・スピンオフ", "捨てる・撤退"],
            "optimal_timing": "人材育成、資源蓄積期",
            "risk_factors": ["蓄えすぎ", "動かない", "機会損失"]
        },
        "yao": {
            "1": {"phrase": "有厲利已", "meaning": "危険あり、止まるが利", "action_modifier": {"耐える・潜伏": 2, "守る・維持": 1}},
            "2": {"phrase": "輿説輹", "meaning": "車の軸を外す。止まる", "action_modifier": {"耐える・潜伏": 2, "守る・維持": 1}},
            "3": {"phrase": "良馬逐", "meaning": "良馬が追う。進んでよし", "action_modifier": {"攻める・挑戦": 2, "対話・融合": 1}},
            "4": {"phrase": "童牛之牿", "meaning": "若い牛の角を塞ぐ。予防", "action_modifier": {"守る・維持": 2, "耐える・潜伏": 1}},
            "5": {"phrase": "豶豕之牙", "meaning": "去勢した豚の牙。制御", "action_modifier": {"守る・維持": 2, "対話・融合": 1}},
            "6": {"phrase": "何天之衢", "meaning": "天の衢を何す。大いに通ず", "action_modifier": {"攻める・挑戦": 2, "刷新・破壊": 1}}
        }
    },
    "27": {
        "id": 27, "name": "山雷頤", "chinese": "頤",
        "keyword": "養い・口・慎み",
        "nature": {"element": "山雷", "movement": "養生・慎み", "timing": "養生期", "warning": "貪欲"},
        "business_context": {
            "favorable_actions": ["守る・維持", "耐える・潜伏"],
            "unfavorable_actions": ["攻める・挑戦", "分散・スピンオフ"],
            "optimal_timing": "人材育成、組織の養生期",
            "risk_factors": ["貪欲", "消費過多", "言葉の過ち"]
        },
        "yao": {
            "1": {"phrase": "舎爾霊亀", "meaning": "霊亀を捨てる。自らを養え", "action_modifier": {"守る・維持": 1, "耐える・潜伏": 0}},
            "2": {"phrase": "顛頤", "meaning": "顎を逆さにする。依存", "action_modifier": {"対話・融合": -1, "守る・維持": 0}},
            "3": {"phrase": "拂頤", "meaning": "頤に拂う。凶", "action_modifier": {"攻める・挑戦": -3, "守る・維持": -1}},
            "4": {"phrase": "顛頤吉", "meaning": "顎を逆さにするも吉。助けを求めよ", "action_modifier": {"対話・融合": 2, "守る・維持": 1}},
            "5": {"phrase": "拂経", "meaning": "経を拂う。正道に反す", "action_modifier": {"耐える・潜伏": 2, "守る・維持": 1}},
            "6": {"phrase": "由頤", "meaning": "頤に由る。大吉", "action_modifier": {"攻める・挑戦": 2, "対話・融合": 1}}
        }
    },
    "28": {
        "id": 28, "name": "沢風大過", "chinese": "大過",
        "keyword": "大過・過剰・非常時",
        "nature": {"element": "沢風", "movement": "過剰・極端", "timing": "非常時", "warning": "崩壊"},
        "business_context": {
            "favorable_actions": ["刷新・破壊", "捨てる・撤退"],
            "unfavorable_actions": ["守る・維持", "耐える・潜伏"],
            "optimal_timing": "非常事態、大胆な決断が必要な時",
            "risk_factors": ["崩壊", "過剰負荷", "無謀"]
        },
        "yao": {
            "1": {"phrase": "藉用白茅", "meaning": "白茅を藉く。慎重に", "action_modifier": {"守る・維持": 2, "耐える・潜伏": 1}},
            "2": {"phrase": "枯楊生稊", "meaning": "枯れた柳に芽が出る。再生", "action_modifier": {"刷新・破壊": 1, "対話・融合": 1}},
            "3": {"phrase": "棟橈", "meaning": "棟木がたわむ。凶", "action_modifier": {"捨てる・撤退": 2, "刷新・破壊": 0}},
            "4": {"phrase": "棟隆", "meaning": "棟木が隆起。吉", "action_modifier": {"攻める・挑戦": 1, "守る・維持": 1}},
            "5": {"phrase": "枯楊生華", "meaning": "枯れた柳に花。老女の結婚", "action_modifier": {"対話・融合": 0, "守る・維持": 0}},
            "6": {"phrase": "過渉滅頂", "meaning": "渡って頭が沈む。凶だが咎なし", "action_modifier": {"攻める・挑戦": -2, "捨てる・撤退": 1}}
        }
    },
    "29": {
        "id": 29, "name": "坎為水", "chinese": "坎",
        "keyword": "険難・陥穽・水",
        "nature": {"element": "水", "movement": "陥穽・困難", "timing": "危機期", "warning": "絶望"},
        "business_context": {
            "favorable_actions": ["耐える・潜伏", "対話・融合"],
            "unfavorable_actions": ["攻める・挑戦", "分散・スピンオフ"],
            "optimal_timing": "危機管理、困難を乗り越える時期",
            "risk_factors": ["絶望", "無謀な行動", "孤立"]
        },
        "yao": {
            "1": {"phrase": "習坎入于坎窞", "meaning": "坎に習い穴に入る。凶", "action_modifier": {"耐える・潜伏": 1, "攻める・挑戦": -3}},
            "2": {"phrase": "坎有険", "meaning": "坎に険あり。小さく得る", "action_modifier": {"耐える・潜伏": 1, "守る・維持": 1}},
            "3": {"phrase": "来之坎坎", "meaning": "来るも坎、行くも坎。動くな", "action_modifier": {"耐える・潜伏": 2, "攻める・挑戦": -2}},
            "4": {"phrase": "樽酒簋貳", "meaning": "酒と食を捧げる。誠意", "action_modifier": {"対話・融合": 2, "守る・維持": 1}},
            "5": {"phrase": "坎不盈", "meaning": "坎満たず。まだ足りない", "action_modifier": {"耐える・潜伏": 1, "守る・維持": 1}},
            "6": {"phrase": "係用徽纆", "meaning": "縄で縛られる。凶", "action_modifier": {"捨てる・撤退": 1, "耐える・潜伏": 0}}
        }
    },
    "30": {
        "id": 30, "name": "離為火", "chinese": "離",
        "keyword": "明晰・付着・火",
        "nature": {"element": "火", "movement": "明晰・付着", "timing": "発展期", "warning": "燃え尽き"},
        "business_context": {
            "favorable_actions": ["攻める・挑戦", "対話・融合"],
            "unfavorable_actions": ["逃げる・放置"],
            "optimal_timing": "ビジョン発信、明確な方向性を示す時",
            "risk_factors": ["燃え尽き", "過度な露出", "軽率"]
        },
        "yao": {
            "1": {"phrase": "履錯然", "meaning": "足跡が乱れる。慎重に", "action_modifier": {"守る・維持": 1, "耐える・潜伏": 1}},
            "2": {"phrase": "黄離元吉", "meaning": "黄色い離。大吉", "action_modifier": {"攻める・挑戦": 2, "対話・融合": 2}},
            "3": {"phrase": "日昃之離", "meaning": "夕陽の離。終わりの兆し", "action_modifier": {"守る・維持": 0, "捨てる・撤退": 1}},
            "4": {"phrase": "突如其来如", "meaning": "突然やってくる。焼け死ぬ", "action_modifier": {"攻める・挑戦": -2, "耐える・潜伏": 1}},
            "5": {"phrase": "出涕沱若", "meaning": "涙を流す。しかし吉", "action_modifier": {"対話・融合": 1, "守る・維持": 1}},
            "6": {"phrase": "王用出征", "meaning": "王が征伐に出る。首魁を斬る", "action_modifier": {"攻める・挑戦": 2, "刷新・破壊": 2}}
        }
    },
    "31": {
        "id": 31, "name": "沢山咸", "chinese": "咸",
        "keyword": "感応・交流・恋愛",
        "nature": {"element": "沢山", "movement": "感応・交流", "timing": "交流期", "warning": "軽率"},
        "business_context": {
            "favorable_actions": ["対話・融合", "攻める・挑戦"],
            "unfavorable_actions": ["逃げる・放置", "耐える・潜伏"],
            "optimal_timing": "パートナーシップ形成、交渉時",
            "risk_factors": ["軽率な決定", "感情的判断", "表面的関係"]
        },
        "yao": {
            "1": {"phrase": "咸其拇", "meaning": "親指で感じる。始まり", "action_modifier": {"対話・融合": 1, "耐える・潜伏": 1}},
            "2": {"phrase": "咸其腓", "meaning": "ふくらはぎで感じる。動くな", "action_modifier": {"耐える・潜伏": 2, "守る・維持": 1}},
            "3": {"phrase": "咸其股", "meaning": "腿で感じる。追随するな", "action_modifier": {"守る・維持": 1, "対話・融合": 0}},
            "4": {"phrase": "貞吉悔亡", "meaning": "貞しければ吉。朋従う", "action_modifier": {"対話・融合": 2, "攻める・挑戦": 1}},
            "5": {"phrase": "咸其脢", "meaning": "背中で感じる。後悔なし", "action_modifier": {"守る・維持": 2, "対話・融合": 1}},
            "6": {"phrase": "咸其輔頬舌", "meaning": "顎と頬と舌で感じる。言葉", "action_modifier": {"対話・融合": 1, "攻める・挑戦": 0}}
        }
    },
    "32": {
        "id": 32, "name": "雷風恒", "chinese": "恒",
        "keyword": "恒久・持続・継続",
        "nature": {"element": "雷風", "movement": "持続・継続", "timing": "安定期", "warning": "硬直"},
        "business_context": {
            "favorable_actions": ["守る・維持", "耐える・潜伏"],
            "unfavorable_actions": ["刷新・破壊", "分散・スピンオフ"],
            "optimal_timing": "長期戦略実行、継続的改善期",
            "risk_factors": ["硬直化", "変化への対応遅れ", "マンネリ"]
        },
        "yao": {
            "1": {"phrase": "浚恒", "meaning": "深く恒する。凶", "action_modifier": {"攻める・挑戦": -2, "守る・維持": 0}},
            "2": {"phrase": "悔亡", "meaning": "後悔消える。正しく持続", "action_modifier": {"守る・維持": 2, "耐える・潜伏": 1}},
            "3": {"phrase": "不恒其徳", "meaning": "徳を恒にせず。恥", "action_modifier": {"守る・維持": -1, "刷新・破壊": 0}},
            "4": {"phrase": "田無禽", "meaning": "狩りに獲物なし。方向転換を", "action_modifier": {"刷新・破壊": 1, "捨てる・撤退": 1}},
            "5": {"phrase": "恒其徳貞", "meaning": "徳を恒にして貞。婦人は吉", "action_modifier": {"守る・維持": 2, "対話・融合": 1}},
            "6": {"phrase": "振恒", "meaning": "恒を振る。凶", "action_modifier": {"攻める・挑戦": -2, "刷新・破壊": -1}}
        }
    }
}

# 33-64卦の定義（簡略化版）
HEXAGRAMS_33_64 = {
    "33": {"id": 33, "name": "天山遯", "chinese": "遯", "keyword": "退避・隠遁・撤退",
           "nature": {"element": "天山", "movement": "退避", "timing": "撤退期", "warning": "遅すぎる撤退"},
           "business_context": {"favorable_actions": ["捨てる・撤退", "耐える・潜伏"], "unfavorable_actions": ["攻める・挑戦"], "optimal_timing": "戦略的撤退", "risk_factors": ["撤退の遅れ"]}},
    "34": {"id": 34, "name": "雷天大壮", "chinese": "大壮", "keyword": "大いに壮ん・勢い・力",
           "nature": {"element": "雷天", "movement": "勢い", "timing": "勢力期", "warning": "傲慢"},
           "business_context": {"favorable_actions": ["攻める・挑戦", "刷新・破壊"], "unfavorable_actions": ["逃げる・放置"], "optimal_timing": "攻勢の時期", "risk_factors": ["傲慢", "暴走"]}},
    "35": {"id": 35, "name": "火地晋", "chinese": "晋", "keyword": "晋む・昇進・発展",
           "nature": {"element": "火地", "movement": "昇進", "timing": "上昇期", "warning": "焦り"},
           "business_context": {"favorable_actions": ["攻める・挑戦", "対話・融合"], "unfavorable_actions": ["捨てる・撤退"], "optimal_timing": "昇進・発展期", "risk_factors": ["焦り"]}},
    "36": {"id": 36, "name": "地火明夷", "chinese": "明夷", "keyword": "明が傷つく・暗黒・隠忍",
           "nature": {"element": "地火", "movement": "隠忍", "timing": "暗黒期", "warning": "絶望"},
           "business_context": {"favorable_actions": ["耐える・潜伏", "守る・維持"], "unfavorable_actions": ["攻める・挑戦"], "optimal_timing": "困難期を耐える", "risk_factors": ["絶望"]}},
    "37": {"id": 37, "name": "風火家人", "chinese": "家人", "keyword": "家庭・組織・内部",
           "nature": {"element": "風火", "movement": "内部調和", "timing": "組織強化期", "warning": "閉鎖的"},
           "business_context": {"favorable_actions": ["守る・維持", "対話・融合"], "unfavorable_actions": ["分散・スピンオフ"], "optimal_timing": "組織固め", "risk_factors": ["閉鎖的"]}},
    "38": {"id": 38, "name": "火沢睽", "chinese": "睽", "keyword": "背反・対立・相違",
           "nature": {"element": "火沢", "movement": "対立", "timing": "対立期", "warning": "決裂"},
           "business_context": {"favorable_actions": ["対話・融合", "守る・維持"], "unfavorable_actions": ["攻める・挑戦"], "optimal_timing": "対立解消", "risk_factors": ["決裂"]}},
    "39": {"id": 39, "name": "水山蹇", "chinese": "蹇", "keyword": "蹇難・困難・足踏み",
           "nature": {"element": "水山", "movement": "困難", "timing": "難局期", "warning": "無謀"},
           "business_context": {"favorable_actions": ["耐える・潜伏", "対話・融合"], "unfavorable_actions": ["攻める・挑戦"], "optimal_timing": "困難を待つ", "risk_factors": ["無謀"]}},
    "40": {"id": 40, "name": "雷水解", "chinese": "解", "keyword": "解放・解決・緩和",
           "nature": {"element": "雷水", "movement": "解放", "timing": "解決期", "warning": "放任"},
           "business_context": {"favorable_actions": ["刷新・破壊", "対話・融合"], "unfavorable_actions": ["逃げる・放置"], "optimal_timing": "問題解決", "risk_factors": ["放任"]}},
    "41": {"id": 41, "name": "山沢損", "chinese": "損", "keyword": "損失・減少・献上",
           "nature": {"element": "山沢", "movement": "減少", "timing": "縮小期", "warning": "過度な削減"},
           "business_context": {"favorable_actions": ["捨てる・撤退", "守る・維持"], "unfavorable_actions": ["攻める・挑戦"], "optimal_timing": "コスト削減", "risk_factors": ["過度な削減"]}},
    "42": {"id": 42, "name": "風雷益", "chinese": "益", "keyword": "増益・利益・恩恵",
           "nature": {"element": "風雷", "movement": "増加", "timing": "成長期", "warning": "貪欲"},
           "business_context": {"favorable_actions": ["攻める・挑戦", "対話・融合"], "unfavorable_actions": ["捨てる・撤退"], "optimal_timing": "投資・拡大", "risk_factors": ["貪欲"]}},
    "43": {"id": 43, "name": "沢天夬", "chinese": "夬", "keyword": "決断・決裂・排除",
           "nature": {"element": "沢天", "movement": "決断", "timing": "決断期", "warning": "拙速"},
           "business_context": {"favorable_actions": ["刷新・破壊", "攻める・挑戦"], "unfavorable_actions": ["逃げる・放置"], "optimal_timing": "決断の時", "risk_factors": ["拙速"]}},
    "44": {"id": 44, "name": "天風姤", "chinese": "姤", "keyword": "遭遇・出会い・偶然",
           "nature": {"element": "天風", "movement": "遭遇", "timing": "出会い期", "warning": "誘惑"},
           "business_context": {"favorable_actions": ["対話・融合", "守る・維持"], "unfavorable_actions": ["攻める・挑戦"], "optimal_timing": "新たな出会い", "risk_factors": ["誘惑"]}},
    "45": {"id": 45, "name": "沢地萃", "chinese": "萃", "keyword": "集合・集結・結集",
           "nature": {"element": "沢地", "movement": "集合", "timing": "結集期", "warning": "烏合の衆"},
           "business_context": {"favorable_actions": ["対話・融合", "攻める・挑戦"], "unfavorable_actions": ["分散・スピンオフ"], "optimal_timing": "人材結集", "risk_factors": ["烏合の衆"]}},
    "46": {"id": 46, "name": "地風升", "chinese": "升", "keyword": "上昇・昇進・成長",
           "nature": {"element": "地風", "movement": "上昇", "timing": "成長期", "warning": "傲慢"},
           "business_context": {"favorable_actions": ["攻める・挑戦", "対話・融合"], "unfavorable_actions": ["捨てる・撤退"], "optimal_timing": "昇進期", "risk_factors": ["傲慢"]}},
    "47": {"id": 47, "name": "沢水困", "chinese": "困", "keyword": "困窮・行き詰まり・窮乏",
           "nature": {"element": "沢水", "movement": "困窮", "timing": "困難期", "warning": "絶望"},
           "business_context": {"favorable_actions": ["耐える・潜伏", "対話・融合"], "unfavorable_actions": ["攻める・挑戦"], "optimal_timing": "困難を耐える", "risk_factors": ["絶望"]}},
    "48": {"id": 48, "name": "水風井", "chinese": "井", "keyword": "井戸・資源・供給",
           "nature": {"element": "水風", "movement": "供給", "timing": "安定供給期", "warning": "枯渇"},
           "business_context": {"favorable_actions": ["守る・維持", "対話・融合"], "unfavorable_actions": ["捨てる・撤退"], "optimal_timing": "インフラ整備", "risk_factors": ["枯渇"]}},
    "49": {"id": 49, "name": "沢火革", "chinese": "革", "keyword": "革命・変革・脱皮",
           "nature": {"element": "沢火", "movement": "変革", "timing": "変革期", "warning": "急進"},
           "business_context": {"favorable_actions": ["刷新・破壊", "攻める・挑戦"], "unfavorable_actions": ["守る・維持"], "optimal_timing": "大改革", "risk_factors": ["急進的すぎる"]}},
    "50": {"id": 50, "name": "火風鼎", "chinese": "鼎", "keyword": "鼎・新生・革新",
           "nature": {"element": "火風", "movement": "革新", "timing": "新体制期", "warning": "不安定"},
           "business_context": {"favorable_actions": ["刷新・破壊", "対話・融合"], "unfavorable_actions": ["逃げる・放置"], "optimal_timing": "新体制構築", "risk_factors": ["不安定"]}},
    "51": {"id": 51, "name": "震為雷", "chinese": "震", "keyword": "震動・衝撃・始動",
           "nature": {"element": "雷", "movement": "衝撃", "timing": "衝撃期", "warning": "動揺"},
           "business_context": {"favorable_actions": ["攻める・挑戦", "刷新・破壊"], "unfavorable_actions": ["逃げる・放置"], "optimal_timing": "始動・衝撃", "risk_factors": ["動揺"]}},
    "52": {"id": 52, "name": "艮為山", "chinese": "艮", "keyword": "止まる・静止・安定",
           "nature": {"element": "山", "movement": "静止", "timing": "静止期", "warning": "停滞"},
           "business_context": {"favorable_actions": ["耐える・潜伏", "守る・維持"], "unfavorable_actions": ["攻める・挑戦"], "optimal_timing": "静観・待機", "risk_factors": ["停滞"]}},
    "53": {"id": 53, "name": "風山漸", "chinese": "漸", "keyword": "漸進・徐々に・段階",
           "nature": {"element": "風山", "movement": "漸進", "timing": "漸進期", "warning": "焦り"},
           "business_context": {"favorable_actions": ["守る・維持", "対話・融合"], "unfavorable_actions": ["刷新・破壊"], "optimal_timing": "段階的発展", "risk_factors": ["焦り"]}},
    "54": {"id": 54, "name": "雷沢帰妹", "chinese": "帰妹", "keyword": "帰嫁・結合・従属",
           "nature": {"element": "雷沢", "movement": "結合", "timing": "結合期", "warning": "従属"},
           "business_context": {"favorable_actions": ["対話・融合", "守る・維持"], "unfavorable_actions": ["攻める・挑戦"], "optimal_timing": "提携・M&A", "risk_factors": ["従属"]}},
    "55": {"id": 55, "name": "雷火豊", "chinese": "豊", "keyword": "豊か・繁栄・絶頂",
           "nature": {"element": "雷火", "movement": "繁栄", "timing": "絶頂期", "warning": "衰退の兆し"},
           "business_context": {"favorable_actions": ["攻める・挑戦", "分散・スピンオフ"], "unfavorable_actions": ["逃げる・放置"], "optimal_timing": "絶頂期の活用", "risk_factors": ["衰退"]}},
    "56": {"id": 56, "name": "火山旅", "chinese": "旅", "keyword": "旅・移動・異境",
           "nature": {"element": "火山", "movement": "移動", "timing": "移動期", "warning": "不安定"},
           "business_context": {"favorable_actions": ["攻める・挑戦", "対話・融合"], "unfavorable_actions": ["守る・維持"], "optimal_timing": "海外進出", "risk_factors": ["不安定"]}},
    "57": {"id": 57, "name": "巽為風", "chinese": "巽", "keyword": "従順・浸透・風",
           "nature": {"element": "風", "movement": "浸透", "timing": "浸透期", "warning": "優柔不断"},
           "business_context": {"favorable_actions": ["対話・融合", "守る・維持"], "unfavorable_actions": ["刷新・破壊"], "optimal_timing": "浸透戦略", "risk_factors": ["優柔不断"]}},
    "58": {"id": 58, "name": "兌為沢", "chinese": "兌", "keyword": "喜び・交流・沢",
           "nature": {"element": "沢", "movement": "喜悦", "timing": "交流期", "warning": "軽率"},
           "business_context": {"favorable_actions": ["対話・融合", "攻める・挑戦"], "unfavorable_actions": ["耐える・潜伏"], "optimal_timing": "交流・営業", "risk_factors": ["軽率"]}},
    "59": {"id": 59, "name": "風水渙", "chinese": "渙", "keyword": "渙散・拡散・解散",
           "nature": {"element": "風水", "movement": "拡散", "timing": "拡散期", "warning": "分裂"},
           "business_context": {"favorable_actions": ["分散・スピンオフ", "対話・融合"], "unfavorable_actions": ["守る・維持"], "optimal_timing": "分散・展開", "risk_factors": ["分裂"]}},
    "60": {"id": 60, "name": "水沢節", "chinese": "節", "keyword": "節度・制限・節約",
           "nature": {"element": "水沢", "movement": "制限", "timing": "節制期", "warning": "過度な制限"},
           "business_context": {"favorable_actions": ["守る・維持", "捨てる・撤退"], "unfavorable_actions": ["攻める・挑戦"], "optimal_timing": "コスト管理", "risk_factors": ["過度な制限"]}},
    "61": {"id": 61, "name": "風沢中孚", "chinese": "中孚", "keyword": "誠信・信頼・真心",
           "nature": {"element": "風沢", "movement": "信頼", "timing": "信頼構築期", "warning": "盲信"},
           "business_context": {"favorable_actions": ["対話・融合", "攻める・挑戦"], "unfavorable_actions": ["逃げる・放置"], "optimal_timing": "信頼構築", "risk_factors": ["盲信"]}},
    "62": {"id": 62, "name": "雷山小過", "chinese": "小過", "keyword": "小さな過ち・謙虚・控えめ",
           "nature": {"element": "雷山", "movement": "控えめ", "timing": "慎重期", "warning": "消極的"},
           "business_context": {"favorable_actions": ["守る・維持", "耐える・潜伏"], "unfavorable_actions": ["攻める・挑戦"], "optimal_timing": "小事に集中", "risk_factors": ["消極的"]}},
    "63": {"id": 63, "name": "水火既済", "chinese": "既済", "keyword": "完成・既に済む・達成",
           "nature": {"element": "水火", "movement": "完成", "timing": "完成期", "warning": "油断"},
           "business_context": {"favorable_actions": ["守る・維持", "耐える・潜伏"], "unfavorable_actions": ["攻める・挑戦"], "optimal_timing": "完成後の維持", "risk_factors": ["油断"]}},
    "64": {"id": 64, "name": "火水未済", "chinese": "未済", "keyword": "未完・未だ済まず・新たな始まり",
           "nature": {"element": "火水", "movement": "未完", "timing": "新たな始まり", "warning": "焦り"},
           "business_context": {"favorable_actions": ["攻める・挑戦", "対話・融合"], "unfavorable_actions": ["逃げる・放置"], "optimal_timing": "新たな挑戦", "risk_factors": ["焦り"]}}
}

# 簡略版爻の生成
def generate_simple_yao(hexagram):
    """簡略版の爻データを生成"""
    return {
        "1": {"phrase": "初爻", "meaning": "始まりの段階", "action_modifier": {"耐える・潜伏": 1, "攻める・挑戦": -1}},
        "2": {"phrase": "二爻", "meaning": "成長の段階", "action_modifier": {"対話・融合": 1, "守る・維持": 1}},
        "3": {"phrase": "三爻", "meaning": "転換の段階", "action_modifier": {"攻める・挑戦": 0, "守る・維持": 0}},
        "4": {"phrase": "四爻", "meaning": "接近の段階", "action_modifier": {"対話・融合": 1, "耐える・潜伏": 0}},
        "5": {"phrase": "五爻", "meaning": "全盛の段階", "action_modifier": {"攻める・挑戦": 1, "対話・融合": 1}},
        "6": {"phrase": "上爻", "meaning": "終わりの段階", "action_modifier": {"捨てる・撤退": 1, "守る・維持": 0}}
    }

def main():
    # 既存のプロファイルを読み込み
    with open(PROFILES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 17-32卦を追加
    for hex_id, hex_data in HEXAGRAMS_17_64.items():
        if "yao" not in hex_data:
            hex_data["yao"] = generate_simple_yao(hex_data)
        data["hexagrams"][hex_id] = hex_data

    # 33-64卦を追加（簡略版）
    for hex_id, hex_data in HEXAGRAMS_33_64.items():
        hex_data["yao"] = generate_simple_yao(hex_data)
        data["hexagrams"][hex_id] = hex_data

    # 保存
    with open(PROFILES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"64卦特性データベースを更新しました: {PROFILES_FILE}")
    print(f"合計 {len(data['hexagrams'])} 卦")

if __name__ == "__main__":
    main()
