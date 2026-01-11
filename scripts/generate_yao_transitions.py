#!/usr/bin/env python3
"""
64卦×6爻の遷移テーブルを生成するスクリプト

易経の卦は6本の爻で構成される。
各爻は陽（━━）または陰（━ ━）。
爻が変化すると、別の卦に遷移する。

爻の順序：
- 初爻（第1爻）= 一番下
- 上爻（第6爻）= 一番上

二進数表現は「上から下」の順で記述（伝統的な易経の記法に合わせる）
例：乾為天 = (1,1,1,1,1,1) = 上爻,五爻,四爻,三爻,二爻,初爻
"""

import json
from pathlib import Path

# 64卦の二進数表現（上から下へ: 上爻→五爻→四爻→三爻→二爻→初爻）
# 陽=1, 陰=0
HEXAGRAM_BINARY = {
    1: (1,1,1,1,1,1),   # 乾為天 ䷀
    2: (0,0,0,0,0,0),   # 坤為地 ䷁
    3: (0,1,0,0,0,1),   # 水雷屯 ䷂
    4: (1,0,0,0,1,0),   # 山水蒙 ䷃
    5: (0,1,0,1,1,1),   # 水天需 ䷄
    6: (1,1,1,0,1,0),   # 天水訟 ䷅
    7: (0,0,0,0,1,0),   # 地水師 ䷆
    8: (0,1,0,0,0,0),   # 水地比 ䷇
    9: (1,1,0,1,1,1),   # 風天小畜 ䷈
    10: (1,1,1,0,1,1),  # 天沢履 ䷉
    11: (0,0,0,1,1,1),  # 地天泰 ䷊
    12: (1,1,1,0,0,0),  # 天地否 ䷋
    13: (1,1,1,1,0,1),  # 天火同人 ䷌
    14: (1,0,1,1,1,1),  # 火天大有 ䷍
    15: (0,0,0,1,0,0),  # 地山謙 ䷎
    16: (0,0,1,0,0,0),  # 雷地予 ䷏
    17: (0,1,1,0,0,1),  # 沢雷随 ䷐
    18: (1,0,0,1,1,0),  # 山風蠱 ䷑
    19: (0,0,0,0,1,1),  # 地沢臨 ䷒
    20: (1,1,0,0,0,0),  # 風地観 ䷓
    21: (1,0,1,0,0,1),  # 火雷噬嗑 ䷔
    22: (1,0,0,1,0,1),  # 山火賁 ䷕
    23: (1,0,0,0,0,0),  # 山地剥 ䷖
    24: (0,0,0,0,0,1),  # 地雷復 ䷗
    25: (1,1,1,0,0,1),  # 天雷无妄 ䷘
    26: (1,0,0,1,1,1),  # 山天大畜 ䷙
    27: (1,0,0,0,0,1),  # 山雷頤 ䷚
    28: (0,1,1,1,1,0),  # 沢風大過 ䷛
    29: (0,1,0,0,1,0),  # 坎為水 ䷜
    30: (1,0,1,1,0,1),  # 離為火 ䷝
    31: (0,1,1,1,0,0),  # 沢山咸 ䷞
    32: (0,0,1,1,1,0),  # 雷風恒 ䷟
    33: (1,1,1,1,0,0),  # 天山遯 ䷠
    34: (0,0,1,1,1,1),  # 雷天大壮 ䷡
    35: (1,0,1,0,0,0),  # 火地晋 ䷢
    36: (0,0,0,1,0,1),  # 地火明夷 ䷣
    37: (1,1,0,1,0,1),  # 風火家人 ䷤
    38: (1,0,1,0,1,1),  # 火沢睽 ䷥
    39: (0,1,0,1,0,0),  # 水山蹇 ䷦
    40: (0,0,1,0,1,0),  # 雷水解 ䷧
    41: (1,0,0,0,1,1),  # 山沢損 ䷨
    42: (1,1,0,0,0,1),  # 風雷益 ䷩
    43: (0,1,1,1,1,1),  # 沢天夬 ䷪
    44: (1,1,1,1,1,0),  # 天風姤 ䷫
    45: (0,1,1,0,0,0),  # 沢地萃 ䷬
    46: (0,0,0,1,1,0),  # 地風升 ䷭
    47: (0,1,1,0,1,0),  # 沢水困 ䷮
    48: (0,1,0,1,1,0),  # 水風井 ䷯
    49: (0,1,1,1,0,1),  # 沢火革 ䷰
    50: (1,0,1,1,1,0),  # 火風鼎 ䷱
    51: (0,0,1,0,0,1),  # 震為雷 ䷲
    52: (1,0,0,1,0,0),  # 艮為山 ䷳
    53: (1,1,0,1,0,0),  # 風山漸 ䷴
    54: (0,0,1,0,1,1),  # 雷沢帰妹 ䷵
    55: (0,0,1,1,0,1),  # 雷火豊 ䷶
    56: (1,0,1,1,0,0),  # 火山旅 ䷷
    57: (1,1,0,1,1,0),  # 巽為風 ䷸
    58: (0,1,1,0,1,1),  # 兌為沢 ䷹
    59: (1,1,0,0,1,0),  # 風水渙 ䷺
    60: (0,1,0,0,1,1),  # 水沢節 ䷻
    61: (1,1,0,0,1,1),  # 風沢中孚 ䷼
    62: (0,0,1,1,0,0),  # 雷山小過 ䷽
    63: (0,1,0,1,0,1),  # 水火既済 ䷾
    64: (1,0,1,0,1,0),  # 火水未済 ䷿
}

# 卦名
HEXAGRAM_NAMES = {
    1: "乾為天", 2: "坤為地", 3: "水雷屯", 4: "山水蒙", 5: "水天需",
    6: "天水訟", 7: "地水師", 8: "水地比", 9: "風天小畜", 10: "天沢履",
    11: "地天泰", 12: "天地否", 13: "天火同人", 14: "火天大有", 15: "地山謙",
    16: "雷地予", 17: "沢雷随", 18: "山風蠱", 19: "地沢臨", 20: "風地観",
    21: "火雷噬嗑", 22: "山火賁", 23: "山地剥", 24: "地雷復", 25: "天雷无妄",
    26: "山天大畜", 27: "山雷頤", 28: "沢風大過", 29: "坎為水", 30: "離為火",
    31: "沢山咸", 32: "雷風恒", 33: "天山遯", 34: "雷天大壮", 35: "火地晋",
    36: "地火明夷", 37: "風火家人", 38: "火沢睽", 39: "水山蹇", 40: "雷水解",
    41: "山沢損", 42: "風雷益", 43: "沢天夬", 44: "天風姤", 45: "沢地萃",
    46: "地風升", 47: "沢水困", 48: "水風井", 49: "沢火革", 50: "火風鼎",
    51: "震為雷", 52: "艮為山", 53: "風山漸", 54: "雷沢帰妹", 55: "雷火豊",
    56: "火山旅", 57: "巽為風", 58: "兌為沢", 59: "風水渙", 60: "水沢節",
    61: "風沢中孚", 62: "雷山小過", 63: "水火既済", 64: "火水未済",
}

# 384爻の爻辞（簡略版、yao_phrases_384.jsonから参照可能）
YAO_PHRASES = {
    "1-1": {"classic": "潜龍勿用", "modern": "力を蓄えて好機を待つ"},
    "1-2": {"classic": "見龍在田", "modern": "実力が認められ始める"},
    "1-3": {"classic": "君子終日乾乾", "modern": "日々努力し警戒を怠らない"},
    "1-4": {"classic": "或躍在淵", "modern": "飛躍のチャンスをうかがう"},
    "1-5": {"classic": "飛龍在天", "modern": "リーダーとして手腕を振るう"},
    "1-6": {"classic": "亢龍有悔", "modern": "進みすぎて孤立する"},
    # ... 以下384爻分を定義（別ファイルから読み込む）
}

# 二進数から卦番号への逆引き
BINARY_TO_HEXAGRAM = {v: k for k, v in HEXAGRAM_BINARY.items()}


def flip_yao(binary: tuple, position: int) -> tuple:
    """
    指定位置の爻を反転させる
    position: 1-6（初爻から上爻）

    二進数は(上爻,五爻,四爻,三爻,二爻,初爻)の順なので
    position=1（初爻）はindex=5
    position=6（上爻）はindex=0
    """
    lst = list(binary)
    idx = 6 - position  # 初爻(1)→idx=5, 上爻(6)→idx=0
    lst[idx] = 1 - lst[idx]  # 0→1, 1→0
    return tuple(lst)


def get_transition(hexagram_id: int, changing_yao: int) -> int:
    """
    卦IDと変爻位置から、遷移先の卦IDを取得
    """
    original = HEXAGRAM_BINARY[hexagram_id]
    changed = flip_yao(original, changing_yao)
    return BINARY_TO_HEXAGRAM.get(changed)


def generate_transition_table():
    """
    64卦×6爻の遷移テーブルを生成
    """
    transitions = {}

    for hex_id in range(1, 65):
        hex_name = HEXAGRAM_NAMES[hex_id]
        transitions[hex_id] = {
            "name": hex_name,
            "transitions": {}
        }

        for yao in range(1, 7):
            next_hex = get_transition(hex_id, yao)
            if next_hex:
                next_name = HEXAGRAM_NAMES[next_hex]
            else:
                next_name = "ERROR"
                next_hex = 0

            transitions[hex_id]["transitions"][yao] = {
                "next_hexagram_id": next_hex,
                "next_hexagram_name": next_name,
            }

    return transitions


def generate_yao_recommendations():
    """
    爻位別の推奨行動テーブル
    """
    recommendations = {
        1: {
            "name": "初爻",
            "stage": "発芽期・始動期",
            "basic_stance": "待機・潜伏",
            "recommended_actions": ["学習", "準備", "観察", "基盤作り"],
            "avoid_actions": ["拙速な行動", "表に出る", "大きな投資"],
            "success_condition": "時機を待ち、力を蓄えること",
            "failure_pattern": "焦って動き、準備不足で失敗",
            "lifecycle_corporate": "スタートアップ・構想段階",
            "lifecycle_individual": "新人・見習い・修業中"
        },
        2: {
            "name": "二爻",
            "stage": "成長期・基盤確立期",
            "basic_stance": "着実・中庸",
            "recommended_actions": ["実務遂行", "基盤強化", "協力関係構築", "実績作り"],
            "avoid_actions": ["冒険", "孤立", "上を急ぐ"],
            "success_condition": "地道な努力で実績を積むこと",
            "failure_pattern": "基盤が固まる前に拡大を急ぐ",
            "lifecycle_corporate": "成長軌道・第二創業期",
            "lifecycle_individual": "中堅・実務者・専門性確立"
        },
        3: {
            "name": "三爻",
            "stage": "転換期・岐路",
            "basic_stance": "慎重・熟考",
            "recommended_actions": ["状況分析", "相談", "柔軟な対応", "リスク評価"],
            "avoid_actions": ["強行突破", "固執", "独断"],
            "success_condition": "分岐点を正しく認識し、適切な選択をすること",
            "failure_pattern": "岐路での判断を誤る、または決断を先送り",
            "lifecycle_corporate": "事業転換点・M&A検討期",
            "lifecycle_individual": "管理職候補・キャリア転換点"
        },
        4: {
            "name": "四爻",
            "stage": "成熟期・接近期",
            "basic_stance": "謙虚・調整",
            "recommended_actions": ["上位者との調整", "準備の仕上げ", "根回し", "謙虚な姿勢"],
            "avoid_actions": ["独断専行", "越権行為", "傲慢"],
            "success_condition": "適切な関係者と連携し、機を待つこと",
            "failure_pattern": "独走して孤立、または過度な慎重さで機を逃す",
            "lifecycle_corporate": "拡大期・業界上位進出",
            "lifecycle_individual": "管理職・専門家・次期リーダー"
        },
        5: {
            "name": "五爻",
            "stage": "全盛期・リーダー期",
            "basic_stance": "決断・実行",
            "recommended_actions": ["リーダーシップ発揮", "大きな決断", "実行", "人材登用"],
            "avoid_actions": ["優柔不断", "過度な委任", "現状維持"],
            "success_condition": "最適なタイミングで決断し、リーダーシップを発揮すること",
            "failure_pattern": "機会を逃す、または決断を下せない",
            "lifecycle_corporate": "業界リーダー・最盛期",
            "lifecycle_individual": "経営層・第一人者・トップ"
        },
        6: {
            "name": "上爻",
            "stage": "衰退期・転換期・極み",
            "basic_stance": "撤退・譲渡",
            "recommended_actions": ["引き際を見極める", "後進育成", "事業承継", "縮小・撤退"],
            "avoid_actions": ["執着", "強行突破", "過度な拡大"],
            "success_condition": "適切なタイミングで引く、または次世代に託すこと",
            "failure_pattern": "行き過ぎて孤立、または引き際を誤る",
            "lifecycle_corporate": "成熟産業・次の一手・事業承継",
            "lifecycle_individual": "引退期・次世代育成・セカンドキャリア"
        }
    }
    return recommendations


def generate_action_yao_compatibility():
    """
    行動タイプと爻位の適合性マトリクス
    score: 1=最適, 2=適切, 3=注意, 4=不適
    """
    compatibility = {
        "攻める・挑戦": {
            1: {"score": 4, "reason": "まだ早い。準備不足で失敗する可能性が高い"},
            2: {"score": 3, "reason": "基盤固めが優先。小さな挑戦は可"},
            3: {"score": 3, "reason": "慎重に。分岐点での攻めはリスクを伴う"},
            4: {"score": 2, "reason": "準備が整っていれば可。調整を忘れずに"},
            5: {"score": 1, "reason": "最適なタイミング。リーダーシップを発揮せよ"},
            6: {"score": 4, "reason": "行き過ぎ。後悔する結果になりやすい"},
        },
        "守る・維持": {
            1: {"score": 2, "reason": "妥当。まずは基盤を固めよ"},
            2: {"score": 1, "reason": "最適。着実に実績を積め"},
            3: {"score": 3, "reason": "変化が必要な時に守りは危険"},
            4: {"score": 2, "reason": "適切。準備を怠るな"},
            5: {"score": 3, "reason": "停滞は機会損失。決断の時"},
            6: {"score": 3, "reason": "守っても衰退は止まらない"},
        },
        "捨てる・撤退": {
            1: {"score": 4, "reason": "まだ始まってもいない。撤退は時期尚早"},
            2: {"score": 4, "reason": "成長中の撤退は勿体ない"},
            3: {"score": 2, "reason": "分岐点での戦略的撤退は有効"},
            4: {"score": 3, "reason": "調整中の撤退は要検討"},
            5: {"score": 3, "reason": "全盛期の撤退は判断を要する"},
            6: {"score": 1, "reason": "最適。引き際を見極めた賢明な判断"},
        },
        "耐える・潜伏": {
            1: {"score": 1, "reason": "最適。潜龍勿用。時を待て"},
            2: {"score": 2, "reason": "適切。地道に実績を積め"},
            3: {"score": 3, "reason": "分岐点で耐えるだけでは不十分"},
            4: {"score": 2, "reason": "準備期間として妥当"},
            5: {"score": 4, "reason": "全盛期に潜伏は機会損失"},
            6: {"score": 2, "reason": "適切。静かに引く準備を"},
        },
        "対話・融合": {
            1: {"score": 2, "reason": "協力者を探すのは良い"},
            2: {"score": 1, "reason": "最適。仲間と共に基盤を作れ"},
            3: {"score": 2, "reason": "分岐点での対話は有効"},
            4: {"score": 1, "reason": "最適。上位者との調整を"},
            5: {"score": 2, "reason": "リーダーとして統合を図れ"},
            6: {"score": 2, "reason": "後継者との対話を"},
        },
        "刷新・破壊": {
            1: {"score": 4, "reason": "まだ何もない。壊すものがない"},
            2: {"score": 4, "reason": "基盤を壊すな"},
            3: {"score": 2, "reason": "分岐点での刷新は有効"},
            4: {"score": 3, "reason": "上との調整なしの刷新は危険"},
            5: {"score": 2, "reason": "トップダウンでの改革は可能"},
            6: {"score": 2, "reason": "次世代のための刷新は有効"},
        },
        "逃げる・放置": {
            1: {"score": 3, "reason": "静観は良いが、逃げは問題先送り"},
            2: {"score": 4, "reason": "成長中の放置は衰退を招く"},
            3: {"score": 4, "reason": "分岐点での放置は最悪。決断せよ"},
            4: {"score": 4, "reason": "準備中の放置は機会損失"},
            5: {"score": 4, "reason": "リーダーの責任放棄"},
            6: {"score": 4, "reason": "引き際で逃げるのは無責任"},
        },
        "分散・スピンオフ": {
            1: {"score": 4, "reason": "分散するものがない"},
            2: {"score": 3, "reason": "基盤が固まってから"},
            3: {"score": 2, "reason": "リスク分散として有効"},
            4: {"score": 2, "reason": "拡大期の分散は妥当"},
            5: {"score": 2, "reason": "戦略的な事業分離は可"},
            6: {"score": 1, "reason": "最適。事業承継・分社化の時"},
        },
    }

    return compatibility


def verify_transitions():
    """
    乾為天の遷移を検証
    正しい遷移:
    - 初爻変 → 天風姤(44)
    - 二爻変 → 天火同人(13)
    - 三爻変 → 天沢履(10)
    - 四爻変 → 風天小畜(9)
    - 五爻変 → 火天大有(14)
    - 上爻変 → 沢天夬(43)
    """
    expected = {
        1: 44,  # 天風姤
        2: 13,  # 天火同人
        3: 10,  # 天沢履
        4: 9,   # 風天小畜
        5: 14,  # 火天大有
        6: 43,  # 沢天夬
    }

    print("乾為天の遷移検証:")
    all_correct = True
    for yao, expected_id in expected.items():
        actual_id = get_transition(1, yao)
        status = "✓" if actual_id == expected_id else "✗"
        if actual_id != expected_id:
            all_correct = False
        print(f"  {yao}爻変 → 期待: {expected_id}({HEXAGRAM_NAMES.get(expected_id)}), "
              f"実際: {actual_id}({HEXAGRAM_NAMES.get(actual_id)}) {status}")

    return all_correct


def main():
    # 検証
    print("=== 遷移ロジック検証 ===")
    if not verify_transitions():
        print("\n警告: 遷移ロジックに問題があります。二進数定義を確認してください。")
        return

    print("\n遷移ロジック検証: OK\n")

    output_dir = Path(__file__).parent.parent / "data" / "mappings"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 遷移テーブル
    transitions = generate_transition_table()
    with open(output_dir / "yao_transitions.json", "w", encoding="utf-8") as f:
        json.dump(transitions, f, ensure_ascii=False, indent=2)
    print(f"Generated: {output_dir / 'yao_transitions.json'}")

    # 2. 爻位別推奨行動
    recommendations = generate_yao_recommendations()
    with open(output_dir / "yao_recommendations.json", "w", encoding="utf-8") as f:
        json.dump(recommendations, f, ensure_ascii=False, indent=2)
    print(f"Generated: {output_dir / 'yao_recommendations.json'}")

    # 3. 行動-爻位適合性マトリクス
    compatibility = generate_action_yao_compatibility()
    with open(output_dir / "action_yao_compatibility.json", "w", encoding="utf-8") as f:
        json.dump(compatibility, f, ensure_ascii=False, indent=2)
    print(f"Generated: {output_dir / 'action_yao_compatibility.json'}")

    # サマリー出力
    print("\n=== 生成完了 ===")
    print(f"遷移パターン数: {64 * 6} = 384")
    print(f"爻位推奨: 6")
    print(f"行動-爻位組み合わせ: {len(compatibility)} × 6 = {len(compatibility) * 6}")


if __name__ == "__main__":
    main()
