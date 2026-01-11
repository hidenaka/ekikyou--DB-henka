#!/usr/bin/env python3
"""
易経変化診断エンジン v2.0

64卦384爻に基づく厳密な診断システム
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


@dataclass
class DiagnosticResultV2:
    """64卦384爻の診断結果"""
    # 卦の情報
    hexagram_number: int
    hexagram_name: str
    upper_trigram: str
    lower_trigram: str
    hexagram_meaning: str
    hexagram_image: str
    hexagram_situation: str

    # 爻の情報
    line_number: int
    line_name: str
    line_interpretation: str
    line_advice: str
    line_warning: str

    # スコア情報
    upper_scores: Dict[str, int]
    lower_scores: Dict[str, int]
    line_scores: List[int]

    # 類似事例（オプション）
    similar_cases: List[Dict] = field(default_factory=list)


class DiagnosticEngineV2:
    """64卦384爻診断エンジン"""

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

    def get_all_questions(self) -> Dict[str, List[Dict]]:
        """全質問を取得"""
        return self.questions["questions"]

    def record_answer(self, question_id: str, answer_value: str) -> None:
        """回答を記録"""
        # 質問を検索
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
        scores = defaultdict(int)

        for qid, answer in self.answers.items():
            if answer["category"] == ("inner" if category == "lower" else "outer"):
                weights = answer["option"].get("weights", {})
                for trigram, weight in weights.items():
                    scores[trigram] += weight

        if not scores:
            return "坤", dict(scores)  # デフォルト

        # 最大スコアの八卦を選択
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
            return 3, []  # デフォルトは三爻

        # 平均を計算し、最も近い整数に丸める
        avg = sum(line_answers) / len(line_answers)
        line = max(1, min(6, round(avg)))

        return line, line_answers

    def get_hexagram(self, upper: str, lower: str) -> Dict:
        """上卦・下卦から64卦を取得"""
        matrix = self.hexagram_data["hexagram_matrix"]
        upper_idx = self.TRIGRAM_ORDER.index(upper)
        lower_idx = self.TRIGRAM_ORDER.index(lower)

        hexagram_name = matrix[upper_idx][lower_idx]
        hexagram_info = self.hexagram_data["hexagrams"].get(hexagram_name, {})

        return {
            "name": hexagram_name,
            "number": hexagram_info.get("number", 0),
            "meaning": hexagram_info.get("meaning", ""),
            "image": hexagram_info.get("image", ""),
            "situation": hexagram_info.get("situation", ""),
            "keywords": hexagram_info.get("keywords", []),
        }

    def get_yao(self, hexagram_number: int, line_number: int) -> Dict:
        """特定の爻の解説を取得"""
        yao_id = f"{hexagram_number:02d}-{line_number}"
        yao = self.yao_data["yao"].get(yao_id, {})
        return yao

    def get_line_position_info(self, line_number: int) -> Dict:
        """爻位の基本情報を取得"""
        return self.line_positions["positions"].get(str(line_number), {})

    def load_similar_cases(self, hexagram_name: str, line_number: int, limit: int = 3) -> List[Dict]:
        """類似事例を読み込み（爻位一致を優先）"""
        cases_path = self.data_dir.parent / "raw" / "cases.jsonl"

        if not cases_path.exists():
            return []

        # 卦一致 + 爻位一致の事例と、卦のみ一致の事例を分けて収集
        exact_matches = []  # 卦 + 爻位が一致
        hexagram_matches = []  # 卦のみ一致

        with open(cases_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    case = json.loads(line)
                    if case.get("classical_before_hexagram") == hexagram_name:
                        # changing_lines_1 に爻位が含まれているかチェック
                        changing_lines = case.get("changing_lines_1", [])
                        if line_number in changing_lines:
                            exact_matches.append(case)
                        else:
                            hexagram_matches.append(case)
                except json.JSONDecodeError:
                    continue

        # 爻位一致を優先して返す
        result = exact_matches[:limit]
        if len(result) < limit:
            result.extend(hexagram_matches[:limit - len(result)])

        return result

    def diagnose(self) -> DiagnosticResultV2:
        """診断を実行"""
        # 八卦を計算
        lower_trigram, lower_scores = self.calculate_trigram("lower")
        upper_trigram, upper_scores = self.calculate_trigram("upper")

        # 卦を取得
        hexagram = self.get_hexagram(upper_trigram, lower_trigram)

        # 爻位を計算
        line_number, line_scores = self.calculate_line()

        # 爻の解説を取得
        yao = self.get_yao(hexagram["number"], line_number)
        line_pos = self.get_line_position_info(line_number)

        # 類似事例を取得
        similar_cases = self.load_similar_cases(hexagram["name"], line_number)

        return DiagnosticResultV2(
            hexagram_number=hexagram["number"],
            hexagram_name=hexagram["name"],
            upper_trigram=upper_trigram,
            lower_trigram=lower_trigram,
            hexagram_meaning=hexagram["meaning"],
            hexagram_image=hexagram["image"],
            hexagram_situation=hexagram["situation"],
            line_number=line_number,
            line_name=line_pos.get("name", f"第{line_number}爻"),
            line_interpretation=yao.get("interpretation", ""),
            line_advice=yao.get("advice", ""),
            line_warning=yao.get("warning", ""),
            upper_scores=upper_scores,
            lower_scores=lower_scores,
            line_scores=line_scores,
            similar_cases=similar_cases,
        )

    def reset(self):
        """回答をリセット"""
        self.answers.clear()


def format_result_v2(result: DiagnosticResultV2) -> str:
    """診断結果をフォーマット"""
    lines = []

    lines.append("")
    lines.append("━" * 55)
    lines.append("  易経六十四卦 診断結果")
    lines.append("━" * 55)
    lines.append("")

    # 卦の情報
    lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    lines.append("┃  あなたの卦                                       ┃")
    lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    lines.append("")
    lines.append(f"  第{result.hexagram_number}卦 【{result.hexagram_name}】")
    lines.append(f"")
    lines.append(f"    {result.hexagram_meaning}")
    lines.append(f"")
    lines.append(f"  上卦（外）: {result.upper_trigram}  ─ 外部環境・周囲の状況")
    lines.append(f"  下卦（内）: {result.lower_trigram}  ─ 内面・自分の本質")
    lines.append(f"")
    lines.append(f"  【象】{result.hexagram_image}")
    lines.append(f"")
    lines.append(f"  【状況】{result.hexagram_situation}")
    lines.append("")

    # 爻の情報
    lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    lines.append("┃  あなたの爻（段階）                                ┃")
    lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    lines.append("")
    lines.append(f"  【{result.line_name}】（第{result.line_number}爻）")
    lines.append(f"")

    # 爻位の視覚的表現
    lines.append("  ┌───────┐")
    for i in range(6, 0, -1):
        mark = " ● " if i == result.line_number else "───"
        label = f" ← {result.line_name}" if i == result.line_number else ""
        lines.append(f"  │ {mark} │{label}")
    lines.append("  └───────┘")
    lines.append("")

    lines.append(f"  {result.line_interpretation}")
    lines.append("")

    # アドバイス
    lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    lines.append("┃  この時期のアドバイス                              ┃")
    lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    lines.append("")
    lines.append(f"  ✅ {result.line_advice}")
    lines.append("")
    lines.append(f"  ⚠️ 注意: {result.line_warning}")
    lines.append("")

    # 類似事例
    if result.similar_cases:
        lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
        lines.append("┃  同じ卦を持つ事例                                ┃")
        lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
        lines.append("")
        for case in result.similar_cases[:2]:
            lines.append(f"  ■ {case.get('target_name', '不明')}")
            summary = case.get('story_summary', '')
            if len(summary) > 80:
                summary = summary[:80] + "..."
            lines.append(f"    {summary}")
            lines.append(f"    結果: {case.get('outcome', '不明')}")
            lines.append("")

    lines.append("━" * 55)
    lines.append("")

    return "\n".join(lines)


def run_interactive_v2():
    """対話形式で診断を実行（v2）"""
    engine = DiagnosticEngineV2()

    print("\n" + "=" * 55)
    print("  易経六十四卦 診断システム v2.0")
    print("=" * 55 + "\n")

    questions = engine.get_all_questions()

    # 内卦の質問
    print("\n【内面について】\n")
    for q in questions["inner"]:
        print(f"  {q['text']}")
        print("-" * 50)
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
        print("-" * 50)
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
        print("-" * 50)
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
    print(format_result_v2(result))


if __name__ == "__main__":
    run_interactive_v2()
