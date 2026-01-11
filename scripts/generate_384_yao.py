#!/usr/bin/env python3
"""
384爻のデータを生成するスクリプト
易経の伝統的な解釈をベースに、現代的な解説を追加
"""

import json
from pathlib import Path

# 64卦の基本情報
HEXAGRAMS = {
    1: {"name": "乾為天", "theme": "創造・剛健", "nature": "陽の極み、積極的に進む"},
    2: {"name": "坤為地", "theme": "受容・柔順", "nature": "陰の極み、受け入れ従う"},
    3: {"name": "水雷屯", "theme": "困難の始まり", "nature": "産みの苦しみ、忍耐が必要"},
    4: {"name": "山水蒙", "theme": "未熟・啓蒙", "nature": "学びの時、師を求める"},
    5: {"name": "水天需", "theme": "待つ", "nature": "時機を待ち、焦らない"},
    6: {"name": "天水訟", "theme": "争い", "nature": "対立を避け、妥協点を探る"},
    7: {"name": "地水師", "theme": "統率", "nature": "組織を率い、規律を守る"},
    8: {"name": "水地比", "theme": "親しむ", "nature": "協力し、仲間と共に"},
    9: {"name": "風天小畜", "theme": "小さく蓄える", "nature": "少しずつ積み上げる"},
    10: {"name": "天沢履", "theme": "慎重に歩む", "nature": "礼を守り、危険を避ける"},
    11: {"name": "地天泰", "theme": "通じる", "nature": "順調だが永続しない"},
    12: {"name": "天地否", "theme": "塞がる", "nature": "閉塞期、内に籠もる"},
    13: {"name": "天火同人", "theme": "協調", "nature": "志を同じくする者と"},
    14: {"name": "火天大有", "theme": "豊か", "nature": "豊かさ、謙虚さを忘れるな"},
    15: {"name": "地山謙", "theme": "謙遜", "nature": "へりくだることで吉"},
    16: {"name": "雷地予", "theme": "喜び・準備", "nature": "楽観しつつも備える"},
    17: {"name": "沢雷随", "theme": "従う", "nature": "時勢に従い、柔軟に"},
    18: {"name": "山風蠱", "theme": "立て直し", "nature": "腐敗を改め、刷新する"},
    19: {"name": "地沢臨", "theme": "臨む", "nature": "責任を持って対処"},
    20: {"name": "風地観", "theme": "観る", "nature": "よく観察し、状況把握"},
    21: {"name": "火雷噬嗑", "theme": "障害除去", "nature": "断固として悪を除く"},
    22: {"name": "山火賁", "theme": "飾る", "nature": "外見を整えつつ本質も"},
    23: {"name": "山地剥", "theme": "崩壊", "nature": "衰退期、静観する"},
    24: {"name": "地雷復", "theme": "回復", "nature": "再生、焦らず自然に"},
    25: {"name": "天雷无妄", "theme": "誠実", "nature": "虚偽なく、正直に"},
    26: {"name": "山天大畜", "theme": "大いに蓄える", "nature": "力を蓄え、学ぶ"},
    27: {"name": "山雷頤", "theme": "養う", "nature": "養生、言動に注意"},
    28: {"name": "沢風大過", "theme": "過剰", "nature": "非常事態、決断を"},
    29: {"name": "坎為水", "theme": "険難", "nature": "困難重なる、誠実に"},
    30: {"name": "離為火", "theme": "明らか", "nature": "明晰、正しきに付く"},
    31: {"name": "沢山咸", "theme": "感応", "nature": "交流、心を開く"},
    32: {"name": "雷風恒", "theme": "持続", "nature": "一貫性を保つ"},
    33: {"name": "天山遯", "theme": "退く", "nature": "撤退、無理に戦わない"},
    34: {"name": "雷天大壮", "theme": "壮大", "nature": "力あり、礼節を守れ"},
    35: {"name": "火地晋", "theme": "昇進", "nature": "前進、才能を発揮"},
    36: {"name": "地火明夷", "theme": "明が傷つく", "nature": "才能を隠し、耐える"},
    37: {"name": "風火家人", "theme": "家庭", "nature": "内部を整える"},
    38: {"name": "火沢睽", "theme": "背く", "nature": "対立、小事から"},
    39: {"name": "水山蹇", "theme": "困難", "nature": "進めない、自省"},
    40: {"name": "雷水解", "theme": "解ける", "nature": "解放、許す"},
    41: {"name": "山沢損", "theme": "減らす", "nature": "過剰を抑制"},
    42: {"name": "風雷益", "theme": "増やす", "nature": "発展、良きを増やす"},
    43: {"name": "沢天夬", "theme": "決する", "nature": "断固、悪を除く"},
    44: {"name": "天風姤", "theme": "出会い", "nature": "偶然、注意深く"},
    45: {"name": "沢地萃", "theme": "集まる", "nature": "結集、組織化"},
    46: {"name": "地風升", "theme": "昇る", "nature": "着実に上昇"},
    47: {"name": "沢水困", "theme": "困窮", "nature": "苦難、志を貫く"},
    48: {"name": "水風井", "theme": "源泉", "nature": "基本に立ち返る"},
    49: {"name": "沢火革", "theme": "変革", "nature": "古いものを改める"},
    50: {"name": "火風鼎", "theme": "刷新", "nature": "新体制を確立"},
    51: {"name": "震為雷", "theme": "震動", "nature": "衝撃、恐れつつ省みる"},
    52: {"name": "艮為山", "theme": "止まる", "nature": "静止、動かない"},
    53: {"name": "風山漸", "theme": "漸進", "nature": "徐々に、着実に"},
    54: {"name": "雷沢帰妹", "theme": "従属", "nature": "分を守る"},
    55: {"name": "雷火豊", "theme": "豊穣", "nature": "繁栄、永続しない"},
    56: {"name": "火山旅", "theme": "旅", "nature": "不安定、寄寓"},
    57: {"name": "巽為風", "theme": "浸透", "nature": "柔軟に入り込む"},
    58: {"name": "兌為沢", "theme": "喜び", "nature": "和やかに交流"},
    59: {"name": "風水渙", "theme": "散らす", "nature": "分散、まとめ直す"},
    60: {"name": "水沢節", "theme": "節度", "nature": "制限、度を過ごさない"},
    61: {"name": "風沢中孚", "theme": "誠信", "nature": "内なる誠で通じる"},
    62: {"name": "雷山小過", "theme": "小さく過ぎる", "nature": "大事は控える"},
    63: {"name": "水火既済", "theme": "完成", "nature": "達成、油断禁物"},
    64: {"name": "火水未済", "theme": "未完成", "nature": "まだ終わらない"},
}

# 爻位の基本的な意味
LINE_MEANINGS = {
    1: {
        "stage": "始まりの段階",
        "base": "まだ力がなく、表に出る時ではない",
        "advice_positive": "力を蓄えながら、機会を待ちましょう",
        "advice_negative": "軽はずみに動くと失敗します",
    },
    2: {
        "stage": "発展初期",
        "base": "基礎を固め、中庸を得ている段階",
        "advice_positive": "着実に実力をつけていきましょう",
        "advice_negative": "焦って表に出ようとしないこと",
    },
    3: {
        "stage": "転換点",
        "base": "内から外へ出る境目、危険も多い",
        "advice_positive": "慎重に、しかし勇気を持って進みましょう",
        "advice_negative": "この段階での失敗は痛手になります",
    },
    4: {
        "stage": "新段階の入口",
        "base": "上の世界に入ったが、まだ不安定",
        "advice_positive": "謙虚に、上位者との関係を大切に",
        "advice_negative": "出過ぎると反発を受けます",
    },
    5: {
        "stage": "中心・リーダー",
        "base": "全体を統べる最良の位置",
        "advice_positive": "徳をもってリードしましょう",
        "advice_negative": "傲慢になると足元をすくわれます",
    },
    6: {
        "stage": "終わり・極み",
        "base": "極まった状態、これ以上は進めない",
        "advice_positive": "次の段階への移行を考えましょう",
        "advice_negative": "執着すると衰退を招きます",
    },
}

def generate_yao_interpretation(hex_num: int, line_num: int) -> dict:
    """一つの爻の解説を生成"""
    hex_info = HEXAGRAMS[hex_num]
    line_info = LINE_MEANINGS[line_num]

    # 爻のIDを生成（例：01-1 = 乾為天の初爻）
    yao_id = f"{hex_num:02d}-{line_num}"

    # 基本的な解説を組み合わせる
    situation = f"{hex_info['theme']}の卦において、{line_info['stage']}にいます。"

    # 爻位と卦の性質を組み合わせたアドバイス
    if hex_num in [1, 14, 34, 43]:  # 陽の強い卦
        if line_num in [1, 2]:
            advice = "まだ力を発揮する時ではありません。準備に徹しましょう。"
        elif line_num == 3:
            advice = "踏み出す時が近づいています。しかし慎重に。"
        elif line_num == 4:
            advice = "勢いがありますが、謙虚さを忘れずに。"
        elif line_num == 5:
            advice = "力を発揮できる時です。しかし傲慢にならないように。"
        else:
            advice = "極まれば転ずる。次への準備を始めましょう。"
    elif hex_num in [2, 8, 15, 16]:  # 陰の強い卦
        if line_num in [1, 2]:
            advice = "柔順に従い、力を蓄える時期です。"
        elif line_num == 3:
            advice = "動き出す兆しがありますが、まだ控えめに。"
        elif line_num in [4, 5]:
            advice = "中庸を守り、穏やかに進みましょう。"
        else:
            advice = "穏やかに次の段階へ移行しましょう。"
    elif hex_num in [3, 29, 39, 47]:  # 困難の卦
        if line_num in [1, 2]:
            advice = "困難の中にいますが、焦らず基盤を固めましょう。"
        elif line_num == 3:
            advice = "最も厳しい時期かもしれません。耐えてください。"
        elif line_num == 4:
            advice = "少し光が見えてきました。慎重に進みましょう。"
        elif line_num == 5:
            advice = "困難を乗り越える力があります。誠実に対処を。"
        else:
            advice = "困難の出口が見えています。もう少しです。"
    else:  # その他
        advice = f"{hex_info['nature']}。{line_info['advice_positive']}"

    # 警告メッセージ
    warning = line_info['advice_negative']

    return {
        "id": yao_id,
        "hexagram_number": hex_num,
        "hexagram_name": hex_info["name"],
        "line_number": line_num,
        "line_name": ["", "初爻", "二爻", "三爻", "四爻", "五爻", "上爻"][line_num],
        "situation": situation,
        "interpretation": f"{hex_info['theme']}の時、{line_info['stage']}。{line_info['base']}。",
        "advice": advice,
        "warning": warning,
        "keywords": [hex_info['theme'], line_info['stage']],
    }


def generate_all_384():
    """384爻すべてを生成"""
    all_yao = {}

    for hex_num in range(1, 65):
        for line_num in range(1, 7):
            yao = generate_yao_interpretation(hex_num, line_num)
            yao_id = yao["id"]
            all_yao[yao_id] = yao

    return all_yao


def main():
    output_path = Path(__file__).parent.parent / "data" / "diagnostic" / "yao_384.json"

    all_yao = generate_all_384()

    data = {
        "version": "1.0",
        "description": "384爻の解説データ",
        "total_count": len(all_yao),
        "yao": all_yao
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(all_yao)} yao interpretations")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
