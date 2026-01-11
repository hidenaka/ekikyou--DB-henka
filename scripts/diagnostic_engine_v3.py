#!/usr/bin/env python3
"""
易経変化診断エンジン v3.0

64卦384爻に基づく多面的診断システム
- 本卦（現在の状態）
- 錯卦（隠れた対極・見えていない側面）
- 綜卦（相手から見た姿・逆の視点）
- 互卦（内なる本質・根底にあるもの）
- 之卦（変化の先・これからの方向）
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# 八卦の爻パターン（下から順に line1, line2, line3）
# 1 = 陽（━━━）, 0 = 陰（━ ━）
TRIGRAM_LINES = {
    "乾": [1, 1, 1],  # ☰ 天
    "兌": [1, 1, 0],  # ☱ 沢
    "離": [1, 0, 1],  # ☲ 火
    "震": [1, 0, 0],  # ☳ 雷
    "巽": [0, 1, 1],  # ☴ 風
    "坎": [0, 1, 0],  # ☵ 水
    "艮": [0, 0, 1],  # ☶ 山
    "坤": [0, 0, 0],  # ☷ 地
}

# 爻パターンから八卦への逆引き
LINES_TO_TRIGRAM = {tuple(v): k for k, v in TRIGRAM_LINES.items()}


@dataclass
class HexagramInfo:
    """卦の情報"""
    number: int
    name: str
    upper: str
    lower: str
    meaning: str
    image: str
    situation: str
    lines: List[int]  # 6爻のパターン [line1, ..., line6]


@dataclass
class RelatedHexagrams:
    """関連する卦群"""
    main: HexagramInfo        # 本卦（現在の状態）
    opposite: HexagramInfo    # 錯卦（隠れた対極）
    inverted: HexagramInfo    # 綜卦（逆さまの視点）
    nuclear: HexagramInfo     # 互卦（内なる本質）
    resulting: HexagramInfo   # 之卦（変化の先）
    changing_line: int        # 変爻の位置


@dataclass
class DiagnosticResultV3:
    """多面的診断結果"""
    # 本卦の情報
    hexagram: HexagramInfo
    line_number: int
    line_name: str
    line_interpretation: str
    line_advice: str
    line_warning: str

    # 関連卦
    related: RelatedHexagrams

    # スコア情報
    upper_scores: Dict[str, int]
    lower_scores: Dict[str, int]
    line_scores: List[int]

    # 類似事例
    similar_cases: List[Dict] = field(default_factory=list)


class DiagnosticEngineV3:
    """64卦384爻 多面的診断エンジン"""

    TRIGRAM_ORDER = ["乾", "兌", "離", "震", "巽", "坎", "艮", "坤"]

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data" / "diagnostic"
        self.data_dir = Path(data_dir)

        # データを読み込み
        self.hexagram_data = self._load_json("hexagram_64.json")
        self.yao_data = self._load_json("yao_384.json")
        self.questions = self._load_json("questions_64hexagram.json")
        self.line_positions = self._load_json("line_positions.json")

        # 回答を保持
        self.answers: Dict[str, Dict] = {}

    def _load_json(self, filename: str) -> Dict:
        """JSONファイルを読み込み"""
        filepath = self.data_dir / filename
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_hexagram_lines(self, upper: str, lower: str) -> List[int]:
        """上卦・下卦から6爻のパターンを取得"""
        lower_lines = TRIGRAM_LINES[lower]
        upper_lines = TRIGRAM_LINES[upper]
        return lower_lines + upper_lines

    def lines_to_hexagram(self, lines: List[int]) -> HexagramInfo:
        """6爻のパターンから卦を取得"""
        lower_lines = tuple(lines[:3])
        upper_lines = tuple(lines[3:])

        lower = LINES_TO_TRIGRAM.get(lower_lines, "坤")
        upper = LINES_TO_TRIGRAM.get(upper_lines, "坤")

        return self.get_hexagram_info(upper, lower, lines)

    def get_hexagram_info(self, upper: str, lower: str, lines: Optional[List[int]] = None) -> HexagramInfo:
        """上卦・下卦から卦情報を取得"""
        matrix = self.hexagram_data["hexagram_matrix"]
        upper_idx = self.TRIGRAM_ORDER.index(upper)
        lower_idx = self.TRIGRAM_ORDER.index(lower)

        hexagram_name = matrix[upper_idx][lower_idx]
        hexagram_info = self.hexagram_data["hexagrams"].get(hexagram_name, {})

        if lines is None:
            lines = self.get_hexagram_lines(upper, lower)

        return HexagramInfo(
            number=hexagram_info.get("number", 0),
            name=hexagram_name,
            upper=upper,
            lower=lower,
            meaning=hexagram_info.get("meaning", ""),
            image=hexagram_info.get("image", ""),
            situation=hexagram_info.get("situation", ""),
            lines=lines,
        )

    def calculate_opposite(self, lines: List[int]) -> HexagramInfo:
        """錯卦を計算（全ての爻を反転）"""
        opposite_lines = [1 - x for x in lines]
        return self.lines_to_hexagram(opposite_lines)

    def calculate_inverted(self, lines: List[int]) -> HexagramInfo:
        """綜卦を計算（上下を逆転）"""
        inverted_lines = lines[::-1]
        return self.lines_to_hexagram(inverted_lines)

    def calculate_nuclear(self, lines: List[int]) -> HexagramInfo:
        """互卦を計算（2,3,4爻→下卦、3,4,5爻→上卦）"""
        # lines[1], lines[2], lines[3] が下卦
        # lines[2], lines[3], lines[4] が上卦
        nuclear_lines = [
            lines[1], lines[2], lines[3],  # 下卦
            lines[2], lines[3], lines[4],  # 上卦
        ]
        return self.lines_to_hexagram(nuclear_lines)

    def calculate_resulting(self, lines: List[int], changing_line: int) -> HexagramInfo:
        """之卦を計算（変爻を反転）"""
        result_lines = lines.copy()
        idx = changing_line - 1  # 1-indexed → 0-indexed
        result_lines[idx] = 1 - result_lines[idx]
        return self.lines_to_hexagram(result_lines)

    def get_all_questions(self) -> Dict[str, List[Dict]]:
        """全質問を取得"""
        return self.questions["questions"]

    def record_answer(self, question_id: str, answer_value: str) -> None:
        """回答を記録"""
        all_q = self.questions["questions"]
        for category in ["inner", "outer", "line"]:
            for q in all_q.get(category, []):
                if q["id"] == question_id:
                    for opt in q["options"]:
                        if opt["value"] == answer_value:
                            self.answers[question_id] = {
                                "category": category,
                                "value": answer_value,
                                "option": opt
                            }
                            return
        raise ValueError(f"Unknown question or answer: {question_id}, {answer_value}")

    def calculate_trigram(self, category: str) -> Tuple[str, Dict[str, int]]:
        """八卦を計算（upper/lower）"""
        from collections import defaultdict
        scores = defaultdict(int)

        for qid, answer in self.answers.items():
            if answer["category"] == ("inner" if category == "lower" else "outer"):
                weights = answer["option"].get("weights", {})
                for trigram, weight in weights.items():
                    scores[trigram] += weight

        if not scores:
            return "坤", dict(scores)

        max_trigram = max(scores, key=scores.get)
        return max_trigram, dict(scores)

    def calculate_line(self) -> Tuple[int, List[int]]:
        """爻位を計算"""
        line_answers = []

        for qid, answer in self.answers.items():
            if answer["category"] == "line":
                line_num = answer["option"].get("line")
                if line_num:
                    line_answers.append(line_num)

        if not line_answers:
            return 3, []

        avg = sum(line_answers) / len(line_answers)
        line = max(1, min(6, round(avg)))

        return line, line_answers

    def get_yao(self, hexagram_number: int, line_number: int) -> Dict:
        """特定の爻の解説を取得"""
        yao_id = f"{hexagram_number:02d}-{line_number}"
        return self.yao_data["yao"].get(yao_id, {})

    def get_line_position_info(self, line_number: int) -> Dict:
        """爻位の基本情報を取得"""
        return self.line_positions["positions"].get(str(line_number), {})

    def load_similar_cases(self, hexagram_name: str, line_number: int, limit: int = 3) -> List[Dict]:
        """類似事例を読み込み（爻位一致を優先）"""
        cases_path = self.data_dir.parent / "raw" / "cases.jsonl"

        if not cases_path.exists():
            return []

        exact_matches = []
        hexagram_matches = []

        with open(cases_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    case = json.loads(line)
                    if case.get("classical_before_hexagram") == hexagram_name:
                        changing_lines = case.get("changing_lines_1", [])
                        if line_number in changing_lines:
                            exact_matches.append(case)
                        else:
                            hexagram_matches.append(case)
                except json.JSONDecodeError:
                    continue

        result = exact_matches[:limit]
        if len(result) < limit:
            result.extend(hexagram_matches[:limit - len(result)])

        return result

    def diagnose(self) -> DiagnosticResultV3:
        """診断を実行"""
        # 八卦を計算
        lower_trigram, lower_scores = self.calculate_trigram("lower")
        upper_trigram, upper_scores = self.calculate_trigram("upper")

        # 本卦を取得
        lines = self.get_hexagram_lines(upper_trigram, lower_trigram)
        main_hexagram = self.get_hexagram_info(upper_trigram, lower_trigram, lines)

        # 爻位を計算
        line_number, line_scores = self.calculate_line()

        # 関連卦を計算
        opposite = self.calculate_opposite(lines)
        inverted = self.calculate_inverted(lines)
        nuclear = self.calculate_nuclear(lines)
        resulting = self.calculate_resulting(lines, line_number)

        related = RelatedHexagrams(
            main=main_hexagram,
            opposite=opposite,
            inverted=inverted,
            nuclear=nuclear,
            resulting=resulting,
            changing_line=line_number,
        )

        # 爻の解説を取得
        yao = self.get_yao(main_hexagram.number, line_number)
        line_pos = self.get_line_position_info(line_number)

        # 類似事例を取得
        similar_cases = self.load_similar_cases(main_hexagram.name, line_number)

        return DiagnosticResultV3(
            hexagram=main_hexagram,
            line_number=line_number,
            line_name=line_pos.get("name", f"第{line_number}爻"),
            line_interpretation=yao.get("interpretation", ""),
            line_advice=yao.get("advice", ""),
            line_warning=yao.get("warning", ""),
            related=related,
            upper_scores=upper_scores,
            lower_scores=lower_scores,
            line_scores=line_scores,
            similar_cases=similar_cases,
        )

    def reset(self):
        """回答をリセット"""
        self.answers.clear()


def draw_hexagram_lines(lines: List[int], highlight_line: Optional[int] = None) -> List[str]:
    """卦の爻を描画"""
    output = []
    output.append("  ┌─────────┐")
    for i in range(6, 0, -1):
        line_val = lines[i - 1]
        if line_val == 1:
            symbol = "━━━━━"  # 陽爻
        else:
            symbol = "━━ ━━"  # 陰爻

        if i == highlight_line:
            mark = f" ◉ {symbol}"
            label = f" ← 第{i}爻（変爻）"
        else:
            mark = f"   {symbol}"
            label = ""
        output.append(f"  │{mark}│{label}")
    output.append("  └─────────┘")
    return output


def format_result_v3(result: DiagnosticResultV3) -> str:
    """診断結果をフォーマット（多面的表示）"""
    lines = []

    lines.append("")
    lines.append("━" * 65)
    lines.append("  易経六十四卦 多面的診断")
    lines.append("━" * 65)
    lines.append("")

    # ═══════════════════════════════════════════════════════════════
    # 本卦（現在の状態）
    # ═══════════════════════════════════════════════════════════════
    lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    lines.append("┃  【本卦】あなたの現在の状態                                  ┃")
    lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    lines.append("")

    main = result.hexagram
    lines.append(f"  第{main.number}卦【{main.name}】 ─ {main.meaning}")
    lines.append("")
    lines.extend(draw_hexagram_lines(main.lines, result.line_number))
    lines.append("")
    lines.append(f"  上卦（外）: {main.upper}  ─ 外部環境・周囲の状況")
    lines.append(f"  下卦（内）: {main.lower}  ─ 内面・自分の本質")
    lines.append("")
    lines.append(f"  【象】{main.image}")
    lines.append(f"  【状況】{main.situation}")
    lines.append("")
    lines.append(f"  【{result.line_name}】{result.line_interpretation}")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════
    # 之卦（変化の先）
    # ═══════════════════════════════════════════════════════════════
    lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    lines.append("┃  【之卦】変化の先・これからの方向                            ┃")
    lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    lines.append("")

    resulting = result.related.resulting
    lines.append(f"  第{resulting.number}卦【{resulting.name}】 ─ {resulting.meaning}")
    lines.append("")
    lines.extend(draw_hexagram_lines(resulting.lines))
    lines.append("")

    # 変爻の詳細を表示
    main_line = main.lines[result.line_number - 1]
    from_type = "陽（━━━）" if main_line == 1 else "陰（━ ━）"
    to_type = "陰（━ ━）" if main_line == 1 else "陽（━━━）"
    lines.append(f"  → 第{result.line_number}爻が {from_type} から {to_type} に変化")
    lines.append(f"  → これにより【{main.name}】から【{resulting.name}】へ移行")
    lines.append(f"  → {resulting.situation}")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════
    # 互卦（内なる本質）
    # ═══════════════════════════════════════════════════════════════
    lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    lines.append("┃  【互卦】内なる本質・根底にあるもの                          ┃")
    lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    lines.append("")

    nuclear = result.related.nuclear
    lines.append(f"  第{nuclear.number}卦【{nuclear.name}】 ─ {nuclear.meaning}")
    lines.append("")
    lines.extend(draw_hexagram_lines(nuclear.lines))
    lines.append("")
    lines.append(f"  → 表面には見えないが、状況の核心にあるテーマ")
    lines.append(f"  → {nuclear.situation}")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════
    # 錯卦（隠れた対極）
    # ═══════════════════════════════════════════════════════════════
    lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    lines.append("┃  【錯卦】隠れた対極・見えていない側面                        ┃")
    lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    lines.append("")

    opposite = result.related.opposite
    lines.append(f"  第{opposite.number}卦【{opposite.name}】 ─ {opposite.meaning}")
    lines.append("")
    lines.extend(draw_hexagram_lines(opposite.lines))
    lines.append("")
    lines.append(f"  → あなたが意識していない裏側、影の部分")
    lines.append(f"  → {opposite.situation}")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════
    # 綜卦（逆の視点）
    # ═══════════════════════════════════════════════════════════════
    lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    lines.append("┃  【綜卦】相手から見た姿・逆の視点                            ┃")
    lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    lines.append("")

    inverted = result.related.inverted
    lines.append(f"  第{inverted.number}卦【{inverted.name}】 ─ {inverted.meaning}")
    lines.append("")
    lines.extend(draw_hexagram_lines(inverted.lines))
    lines.append("")
    lines.append(f"  → 立場を入れ替えたとき、相手にはこう見えている")
    lines.append(f"  → {inverted.situation}")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════
    # アドバイス
    # ═══════════════════════════════════════════════════════════════
    lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    lines.append("┃  【総合アドバイス】                                          ┃")
    lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    lines.append("")
    lines.append(f"  ✅ {result.line_advice}")
    lines.append("")
    lines.append(f"  ⚠️ 注意: {result.line_warning}")
    lines.append("")

    # 変化の方向性
    lines.append("  【変化の方向性】")
    lines.append(f"  現在: {main.name}（{main.meaning}）")
    lines.append(f"     ↓  第{result.line_number}爻の変化")
    lines.append(f"  未来: {resulting.name}（{resulting.meaning}）")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════
    # 類似事例
    # ═══════════════════════════════════════════════════════════════
    if result.similar_cases:
        lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
        lines.append("┃  【同じ卦を持つ歴史的事例】                                  ┃")
        lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
        lines.append("")
        for case in result.similar_cases[:2]:
            lines.append(f"  ■ {case.get('target_name', '不明')}")
            summary = case.get('story_summary', '')
            if len(summary) > 70:
                summary = summary[:70] + "..."
            lines.append(f"    {summary}")
            lines.append(f"    結果: {case.get('outcome', '不明')}")
            lines.append("")

    lines.append("━" * 65)
    lines.append("")

    return "\n".join(lines)


def run_interactive_v3():
    """対話形式で診断を実行（v3）"""
    engine = DiagnosticEngineV3()

    print("\n" + "=" * 65)
    print("  易経六十四卦 多面的診断システム v3.0")
    print("=" * 65 + "\n")

    questions = engine.get_all_questions()

    # 内卦の質問
    print("\n【内面について】\n")
    for q in questions["inner"]:
        print(f"  {q['text']}")
        print("-" * 55)
        for i, opt in enumerate(q["options"], 1):
            print(f"  {i}. {opt['label']}")

        while True:
            try:
                choice = input("\n  選択してください (番号): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(q["options"]):
                    engine.record_answer(q["id"], q["options"][idx]["value"])
                    print(f"  → 「{q['options'][idx]['label']}」\n")
                    break
                else:
                    print("  無効な番号です。")
            except ValueError:
                print("  数字を入力してください。")
            except KeyboardInterrupt:
                print("\n\n診断を中断しました。")
                return

    # 外卦の質問
    print("\n【外部環境について】\n")
    for q in questions["outer"]:
        print(f"  {q['text']}")
        print("-" * 55)
        for i, opt in enumerate(q["options"], 1):
            print(f"  {i}. {opt['label']}")

        while True:
            try:
                choice = input("\n  選択してください (番号): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(q["options"]):
                    engine.record_answer(q["id"], q["options"][idx]["value"])
                    print(f"  → 「{q['options'][idx]['label']}」\n")
                    break
                else:
                    print("  無効な番号です。")
            except ValueError:
                print("  数字を入力してください。")
            except KeyboardInterrupt:
                print("\n\n診断を中断しました。")
                return

    # 爻位の質問
    print("\n【段階について】\n")
    for q in questions["line"]:
        print(f"  {q['text']}")
        print("-" * 55)
        for i, opt in enumerate(q["options"], 1):
            desc = f" - {opt.get('description', '')}" if opt.get('description') else ""
            print(f"  {i}. {opt['label']}{desc}")

        while True:
            try:
                choice = input("\n  選択してください (番号): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(q["options"]):
                    engine.record_answer(q["id"], q["options"][idx]["value"])
                    print(f"  → 「{q['options'][idx]['label']}」\n")
                    break
                else:
                    print("  無効な番号です。")
            except ValueError:
                print("  数字を入力してください。")
            except KeyboardInterrupt:
                print("\n\n診断を中断しました。")
                return

    # 診断実行
    print("\n\n診断中...")
    result = engine.diagnose()

    # 結果表示
    print(format_result_v3(result))


if __name__ == "__main__":
    run_interactive_v3()
