#!/usr/bin/env python3
"""
六十四卦算出モジュール - 構造化質問方式

構造化された質問への回答から、決定論的に卦を算出する。
乱数を使用せず、同じ回答は常に同じ卦を返す。
"""

import uuid
from datetime import datetime
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json


class Trigram(Enum):
    """八卦"""
    QIAN = ("乾", "天", "☰", 1)    # 乾 - 天
    DUI = ("兌", "沢", "☱", 2)     # 兌 - 沢
    LI = ("離", "火", "☲", 3)      # 離 - 火
    ZHEN = ("震", "雷", "☳", 4)    # 震 - 雷
    XUN = ("巽", "風", "☴", 5)     # 巽 - 風
    KAN = ("坎", "水", "☵", 6)     # 坎 - 水
    GEN = ("艮", "山", "☶", 7)     # 艮 - 山
    KUN = ("坤", "地", "☷", 8)     # 坤 - 地

    def __init__(self, name_jp: str, symbol: str, unicode_symbol: str, number: int):
        self.name_jp = name_jp
        self.symbol = symbol
        self.unicode_symbol = unicode_symbol
        self.number = number


# 八卦の配列（文王後天図の順序）
TRIGRAM_ORDER = [
    Trigram.QIAN,  # 1
    Trigram.DUI,   # 2
    Trigram.LI,    # 3
    Trigram.ZHEN,  # 4
    Trigram.XUN,   # 5
    Trigram.KAN,   # 6
    Trigram.GEN,   # 7
    Trigram.KUN,   # 8
]

# 六十四卦名（上卦×下卦の順序で配列）
HEXAGRAM_NAMES = {
    (1, 1): (1, "乾", "けん", "乾為天"),
    (1, 2): (10, "履", "り", "天沢履"),
    (1, 3): (13, "同人", "どうじん", "天火同人"),
    (1, 4): (25, "无妄", "むぼう", "天雷无妄"),
    (1, 5): (44, "姤", "こう", "天風姤"),
    (1, 6): (6, "訟", "しょう", "天水訟"),
    (1, 7): (33, "遯", "とん", "天山遯"),
    (1, 8): (12, "否", "ひ", "天地否"),
    (2, 1): (43, "夬", "かい", "沢天夬"),
    (2, 2): (58, "兌", "だ", "兌為沢"),
    (2, 3): (49, "革", "かく", "沢火革"),
    (2, 4): (17, "隨", "ずい", "沢雷随"),
    (2, 5): (28, "大過", "たいか", "沢風大過"),
    (2, 6): (47, "困", "こん", "沢水困"),
    (2, 7): (31, "咸", "かん", "沢山咸"),
    (2, 8): (45, "萃", "すい", "沢地萃"),
    (3, 1): (14, "大有", "たいゆう", "火天大有"),
    (3, 2): (38, "睽", "けい", "火沢睽"),
    (3, 3): (30, "離", "り", "離為火"),
    (3, 4): (21, "噬嗑", "ぜいこう", "火雷噬嗑"),
    (3, 5): (50, "鼎", "てい", "火風鼎"),
    (3, 6): (64, "未済", "びせい", "火水未済"),
    (3, 7): (56, "旅", "りょ", "火山旅"),
    (3, 8): (35, "晋", "しん", "火地晋"),
    (4, 1): (34, "大壮", "たいそう", "雷天大壮"),
    (4, 2): (54, "帰妹", "きまい", "雷沢帰妹"),
    (4, 3): (55, "豊", "ほう", "雷火豊"),
    (4, 4): (51, "震", "しん", "震為雷"),
    (4, 5): (32, "恒", "こう", "雷風恒"),
    (4, 6): (40, "解", "かい", "雷水解"),
    (4, 7): (62, "小過", "しょうか", "雷山小過"),
    (4, 8): (16, "豫", "よ", "雷地豫"),
    (5, 1): (9, "小畜", "しょうちく", "風天小畜"),
    (5, 2): (61, "中孚", "ちゅうふ", "風沢中孚"),
    (5, 3): (37, "家人", "かじん", "風火家人"),
    (5, 4): (42, "益", "えき", "風雷益"),
    (5, 5): (57, "巽", "そん", "巽為風"),
    (5, 6): (59, "渙", "かん", "風水渙"),
    (5, 7): (53, "漸", "ぜん", "風山漸"),
    (5, 8): (20, "観", "かん", "風地観"),
    (6, 1): (5, "需", "じゅ", "水天需"),
    (6, 2): (60, "節", "せつ", "水沢節"),
    (6, 3): (63, "既済", "きせい", "水火既済"),
    (6, 4): (3, "屯", "ちゅん", "水雷屯"),
    (6, 5): (48, "井", "せい", "水風井"),
    (6, 6): (29, "坎", "かん", "坎為水"),
    (6, 7): (39, "蹇", "けん", "水山蹇"),
    (6, 8): (8, "比", "ひ", "水地比"),
    (7, 1): (26, "大畜", "たいちく", "山天大畜"),
    (7, 2): (41, "損", "そん", "山沢損"),
    (7, 3): (22, "賁", "ひ", "山火賁"),
    (7, 4): (27, "頤", "い", "山雷頤"),
    (7, 5): (18, "蠱", "こ", "山風蠱"),
    (7, 6): (4, "蒙", "もう", "山水蒙"),
    (7, 7): (52, "艮", "ごん", "艮為山"),
    (7, 8): (23, "剥", "はく", "山地剥"),
    (8, 1): (11, "泰", "たい", "地天泰"),
    (8, 2): (19, "臨", "りん", "地沢臨"),
    (8, 3): (36, "明夷", "めいい", "地火明夷"),
    (8, 4): (24, "復", "ふく", "地雷復"),
    (8, 5): (46, "升", "しょう", "地風升"),
    (8, 6): (7, "師", "し", "地水師"),
    (8, 7): (15, "謙", "けん", "地山謙"),
    (8, 8): (2, "坤", "こん", "坤為地"),
}


@dataclass
class DiagnosisAnswers:
    """診断質問への回答"""
    perspective: str  # Q1: self, org, family, observer
    theme: str        # Q2: career, relationship, business, etc.
    q3_energy: str    # Q3: expand or contract (外部)
    q4_motion: str    # Q4: moving or still (外部)
    q5_symbol: str    # Q5: A-H (外部象徴)
    q6_energy: str    # Q6: active or passive (内部)
    q7_motion: str    # Q7: moving or still (内部)
    q8_symbol: str    # Q8: A-H (内部象徴)
    q9_stage: str     # Q9: A-F (進展段階)


@dataclass
class DiagnosisResult:
    """診断結果"""
    reading_id: str
    created_at: str
    interpretation_version: str
    inputs: Dict
    upper_trigram: str
    lower_trigram: str
    hexagram_number: int
    hexagram_name: str
    hexagram_reading: str
    hexagram_full_name: str
    active_line: int

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class HexagramCalculator:
    """六十四卦算出器"""

    # Q5/Q8の選択肢から八卦へのマッピング
    SYMBOL_TO_TRIGRAM = {
        'A': Trigram.ZHEN,  # 雷
        'B': Trigram.LI,    # 火
        'C': Trigram.QIAN,  # 天
        'D': Trigram.DUI,   # 沢
        'E': Trigram.XUN,   # 風
        'F': Trigram.KAN,   # 水
        'G': Trigram.KUN,   # 地
        'H': Trigram.GEN,   # 山
    }

    # Q9の選択肢から爻位へのマッピング
    STAGE_TO_LINE = {
        'A': 1,  # 初爻 - 始まったばかり
        'B': 2,  # 二爻 - 初期段階
        'C': 3,  # 三爻 - 転換点
        'D': 4,  # 四爻 - 後半
        'E': 5,  # 五爻 - 終盤
        'F': 6,  # 上爻 - 完了間近
    }

    def __init__(self, version: str = "1.0"):
        self.version = version

    def determine_upper_trigram(self, q3: str, q4: str, q5: str) -> Trigram:
        """
        上卦（外的状況）を決定

        Args:
            q3: エネルギー方向 'A'(拡大) or 'B'(収縮)
            q4: 動静 'A'(動) or 'B'(静)
            q5: 象徴 'A'-'H'

        Returns:
            Trigram: 決定された上卦
        """
        # Q5の選択が最終決定
        # Q3, Q4は質問フローのガイドとして使用（検証用に保持）
        return self.SYMBOL_TO_TRIGRAM[q5.upper()]

    def determine_lower_trigram(self, q6: str, q7: str, q8: str) -> Trigram:
        """
        下卦（内的状況）を決定

        Args:
            q6: エネルギー方向 'A'(積極) or 'B'(慎重)
            q7: 動静 'A'(動) or 'B'(静)
            q8: 象徴 'A'-'H'

        Returns:
            Trigram: 決定された下卦
        """
        return self.SYMBOL_TO_TRIGRAM[q8.upper()]

    def determine_active_line(self, q9: str) -> int:
        """
        爻位（進展段階）を決定

        Args:
            q9: 進展段階 'A'-'F'

        Returns:
            int: 爻位 (1-6)
        """
        return self.STAGE_TO_LINE[q9.upper()]

    def calculate_hexagram(self, upper: Trigram, lower: Trigram) -> Tuple[int, str, str, str]:
        """
        上卦と下卦から六十四卦を算出

        Args:
            upper: 上卦
            lower: 下卦

        Returns:
            Tuple[int, str, str, str]: (卦番号, 卦名, 読み, 正式名)
        """
        key = (upper.number, lower.number)
        return HEXAGRAM_NAMES[key]

    def diagnose(self, answers: DiagnosisAnswers) -> DiagnosisResult:
        """
        診断を実行

        Args:
            answers: 質問への回答

        Returns:
            DiagnosisResult: 診断結果
        """
        # 上卦・下卦を決定
        upper = self.determine_upper_trigram(
            answers.q3_energy, answers.q4_motion, answers.q5_symbol
        )
        lower = self.determine_lower_trigram(
            answers.q6_energy, answers.q7_motion, answers.q8_symbol
        )

        # 爻位を決定
        active_line = self.determine_active_line(answers.q9_stage)

        # 六十四卦を算出
        hex_num, hex_name, hex_reading, hex_full = self.calculate_hexagram(upper, lower)

        # 結果を生成
        return DiagnosisResult(
            reading_id=str(uuid.uuid4()),
            created_at=datetime.now().isoformat(),
            interpretation_version=self.version,
            inputs={
                "perspective": answers.perspective,
                "theme": answers.theme,
                "answers": {
                    "Q3": answers.q3_energy,
                    "Q4": answers.q4_motion,
                    "Q5": answers.q5_symbol,
                    "Q6": answers.q6_energy,
                    "Q7": answers.q7_motion,
                    "Q8": answers.q8_symbol,
                    "Q9": answers.q9_stage,
                }
            },
            upper_trigram=upper.name_jp,
            lower_trigram=lower.name_jp,
            hexagram_number=hex_num,
            hexagram_name=hex_name,
            hexagram_reading=hex_reading,
            hexagram_full_name=hex_full,
            active_line=active_line,
        )


def demo():
    """デモ実行"""
    calculator = HexagramCalculator()

    # サンプル回答
    answers = DiagnosisAnswers(
        perspective="self",
        theme="career",
        q3_energy="A",   # 拡大
        q4_motion="A",   # 動
        q5_symbol="A",   # 雷（震）
        q6_energy="A",   # 積極
        q7_motion="B",   # 静
        q8_symbol="C",   # 天（乾）
        q9_stage="C",    # 転換点（三爻）
    )

    result = calculator.diagnose(answers)

    print("=== 診断結果 ===")
    print(f"Reading ID: {result.reading_id}")
    print(f"上卦: {result.upper_trigram}")
    print(f"下卦: {result.lower_trigram}")
    print(f"卦: 第{result.hexagram_number}卦 {result.hexagram_name}（{result.hexagram_reading}）")
    print(f"正式名: {result.hexagram_full_name}")
    print(f"爻位: {result.active_line}爻")
    print()
    print("=== JSON出力 ===")
    print(result.to_json())


if __name__ == "__main__":
    demo()
