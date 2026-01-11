#!/usr/bin/env python3
"""
易経変化診断エンジン v1.0

位相（八卦）× 勢（momentum）× 時（timing）に基づく
実例ベースの行動推奨システム
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


@dataclass
class DiagnosticAnswer:
    """ユーザーの回答を保持"""
    question_id: str
    value: str
    score: Optional[float] = None
    weights: Dict[str, int] = field(default_factory=dict)
    trigger_type: Optional[str] = None
    avoid_pattern: Optional[str] = None
    preferred_action: Optional[str] = None


@dataclass
class DiagnosticResult:
    """診断結果"""
    primary_hex: str
    hex_scores: Dict[str, int]
    momentum: str
    momentum_score: float
    timing: str
    timing_score: float
    recommended_actions: List[Tuple[str, float, str]]  # (action, score, reason)
    warnings: List[str]
    judgment: str
    detail: str
    before_state: Optional[str] = None
    trigger_type: Optional[str] = None
    avoid_pattern: Optional[str] = None
    preferred_action: Optional[str] = None


class DiagnosticEngine:
    """易経変化診断エンジン"""

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data" / "diagnostic"
        self.data_dir = Path(data_dir)

        # 設定ファイルを読み込み
        self.question_mapping = self._load_json("question_mapping.json")
        self.judgment_rules = self._load_json("judgment_rules.json")
        self.statistics_table = self._load_json("statistics_table.json")
        self.failure_avoidance = self._load_json("failure_avoidance.json")

        # マスターデータを取得
        self.action_types = self.question_mapping["action_type_master"]
        self.pattern_types = self.question_mapping["pattern_type_master"]
        self.questions = self.question_mapping["questions"]
        self.phase_definitions = self.judgment_rules["phase_definitions"]

        # 回答を保持
        self.answers: Dict[str, DiagnosticAnswer] = {}

    def _load_json(self, filename: str) -> Dict:
        """JSONファイルを読み込み"""
        filepath = self.data_dir / filename
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_questions(self) -> List[Dict]:
        """質問リストを取得"""
        return self.questions

    def get_question_by_id(self, question_id: str) -> Optional[Dict]:
        """質問IDで質問を取得"""
        for q in self.questions:
            if q["id"] == question_id:
                return q
        return None

    def record_answer(self, question_id: str, option_value: str) -> DiagnosticAnswer:
        """回答を記録"""
        question = self.get_question_by_id(question_id)
        if not question:
            raise ValueError(f"Unknown question: {question_id}")

        # 選択肢を検索
        selected_option = None
        for opt in question["options"]:
            if opt["value"] == option_value:
                selected_option = opt
                break

        if not selected_option:
            raise ValueError(f"Unknown option: {option_value} for {question_id}")

        # 回答オブジェクトを作成
        answer = DiagnosticAnswer(
            question_id=question_id,
            value=option_value,
            score=selected_option.get("score"),
            weights=selected_option.get("weights", {}),
            trigger_type=selected_option.get("trigger_type"),
            avoid_pattern=selected_option.get("avoid_pattern"),
            preferred_action=selected_option.get("preferred_action")
        )

        self.answers[question_id] = answer
        return answer

    def calculate_hex_scores(self) -> Dict[str, int]:
        """八卦スコアを計算"""
        hex_scores = defaultdict(int)

        for answer in self.answers.values():
            for hex_name, weight in answer.weights.items():
                hex_scores[hex_name] += weight

        return dict(hex_scores)

    def get_primary_hex(self) -> Tuple[str, Dict[str, int]]:
        """主要八卦を決定"""
        hex_scores = self.calculate_hex_scores()
        if not hex_scores:
            return "坤", {}  # デフォルト

        primary = max(hex_scores, key=hex_scores.get)
        return primary, hex_scores

    def _parse_threshold_condition(self, condition: str, score: float) -> bool:
        """閾値条件を解析して判定"""
        # 例: "score >= 1.0", "-0.5 <= score < 1.0", "score < -2.0"
        condition = condition.replace("score", str(score))
        try:
            return eval(condition)
        except:
            return False

    def calculate_momentum(self) -> Tuple[str, float]:
        """勢（momentum）を計算"""
        # (Q1.score + Q2.score + Q5.score + Q6.score) / 4
        q_ids = ["Q1", "Q2", "Q5", "Q6"]
        scores = []

        for qid in q_ids:
            if qid in self.answers and self.answers[qid].score is not None:
                scores.append(self.answers[qid].score)

        if not scores:
            return "stable", 0.0

        avg_score = sum(scores) / len(scores)

        # 閾値判定（JSONから読み込み）
        thresholds = self.judgment_rules["momentum_rules"]["calculation"]["thresholds"]
        for th in thresholds:
            if self._parse_threshold_condition(th["condition"], avg_score):
                return th["id"], avg_score

        return "stable", avg_score

    def calculate_timing(self) -> Tuple[str, float]:
        """時（timing）を計算"""
        # (Q3.score + Q5.score - Q6.score) / 3
        q3_score = self.answers.get("Q3", DiagnosticAnswer("Q3", "")).score or 0
        q5_score = self.answers.get("Q5", DiagnosticAnswer("Q5", "")).score or 0
        q6_score = self.answers.get("Q6", DiagnosticAnswer("Q6", "")).score or 0

        # Q6はマイナス値が多いので、引くことでプラスに転じる
        timing_score = (q3_score + q5_score - q6_score) / 3

        # 閾値判定（JSONから読み込み）
        thresholds = self.judgment_rules["timing_rules"]["calculation"]["thresholds"]
        for th in thresholds:
            if self._parse_threshold_condition(th["condition"], timing_score):
                return th["id"], timing_score

        return "adapt", timing_score

    def get_before_state_from_answers(self) -> Optional[str]:
        """回答から before_state を推定"""
        # Q1, Q2, Q3の回答から状態を推定
        momentum, m_score = self.calculate_momentum()
        timing, t_score = self.calculate_timing()

        # 簡易マッピング
        if momentum == "ascending" and timing == "act_now":
            return "成長痛"
        elif momentum == "ascending":
            return "安定・平和"
        elif momentum == "descending" and timing == "wait":
            return "どん底・危機"
        elif momentum == "descending":
            return "停滞・閉塞"
        elif momentum == "chaotic":
            return "混乱・カオス"
        else:
            return "安定・平和"

    def lookup_statistics(self, before_state: Optional[str],
                          trigger_type: Optional[str],
                          primary_hex: str) -> Dict[str, Dict]:
        """統計テーブルから成功率を取得（優先順位に従う）"""
        stats = {}

        # 優先順位1: by_state_trigger_action
        if before_state and trigger_type:
            state_trigger = self.statistics_table.get("by_state_trigger_action", {})
            if before_state in state_trigger:
                if trigger_type in state_trigger[before_state]:
                    stats = state_trigger[before_state][trigger_type]
                    return stats

        # 優先順位2: by_before_state_action
        if before_state:
            state_action = self.statistics_table.get("by_before_state_action", {})
            if before_state in state_action:
                stats = state_action[before_state]
                return stats

        # 優先順位3: by_trigger_action
        if trigger_type:
            trigger_action = self.statistics_table.get("by_trigger_action", {})
            if trigger_type in trigger_action:
                stats = trigger_action[trigger_type]
                return stats

        # 優先順位4: by_before_hex_action
        hex_action = self.statistics_table.get("by_before_hex_action", {})
        if primary_hex in hex_action:
            stats = hex_action[primary_hex]

        return stats

    def calculate_action_scores(self) -> List[Tuple[str, float, str]]:
        """行動推奨スコアを計算"""
        primary_hex, hex_scores = self.get_primary_hex()
        momentum, m_score = self.calculate_momentum()
        timing, t_score = self.calculate_timing()

        # Q4からtrigger_typeを取得
        trigger_type = None
        if "Q4" in self.answers:
            trigger_type = self.answers["Q4"].trigger_type

        # before_stateを推定
        before_state = self.get_before_state_from_answers()

        # Q7からavoid_patternを取得
        avoid_pattern = None
        if "Q7" in self.answers:
            avoid_pattern = self.answers["Q7"].avoid_pattern

        # Q8からpreferred_actionを取得
        preferred_action = None
        if "Q8" in self.answers:
            preferred_action = self.answers["Q8"].preferred_action

        # 統計テーブルから基本スコアを取得
        stats = self.lookup_statistics(before_state, trigger_type, primary_hex)

        # 各行動タイプのスコアを計算
        action_scores = {}
        for action in self.action_types:
            # 基本スコア（成功率をベースに）
            if action in stats:
                base_score = stats[action].get("success_rate", 50)
            else:
                base_score = 50  # デフォルト

            score = base_score
            reasons = []

            # primary_hexのbase_action/risk_actionによる補正
            if primary_hex in self.phase_definitions:
                phase_def = self.phase_definitions[primary_hex]
                if action == phase_def.get("base_action"):
                    score += 10
                    reasons.append(f"{primary_hex}の基本行動")
                elif action == phase_def.get("risk_action"):
                    score -= 10
                    reasons.append(f"{primary_hex}ではリスク")

            # momentum modifierを適用
            m_modifiers = self.judgment_rules["momentum_rules"]["action_modifiers"].get(momentum, {})
            modifier = m_modifiers.get("modifier", 1.0)
            if action in m_modifiers.get("boost", []):
                score *= modifier
                reasons.append(f"勢い({momentum})に適合")
            elif action in m_modifiers.get("penalty", []):
                score /= modifier
                reasons.append(f"勢い({momentum})に不適")

            # timing modifierを適用
            t_modifiers = self.judgment_rules["timing_rules"]["action_modifiers"].get(timing, {})
            t_modifier = t_modifiers.get("modifier", 1.0)
            if action in t_modifiers.get("boost", []):
                score *= t_modifier
                reasons.append(f"時({timing})に適合")
            elif action in t_modifiers.get("penalty", []):
                score /= t_modifier
                reasons.append(f"時({timing})に不適")

            # preferred_actionボーナス
            if preferred_action and action == preferred_action:
                score += 15
                reasons.append("優先価値に合致")

            # avoid_patternペナルティ（2段階チェック）
            if avoid_pattern:
                # 1. pattern_warnings の high_risk_actions をチェック（優先）
                pattern_warnings = self.failure_avoidance.get("pattern_warnings", {})
                if avoid_pattern in pattern_warnings:
                    high_risk_actions = pattern_warnings[avoid_pattern].get("high_risk_actions", [])
                    if action in high_risk_actions:
                        score -= 20
                        reasons.append(f"回避パターン({avoid_pattern})で高リスク")

                # 2. pattern_action_risk の failure_rate をチェック（追加ペナルティ）
                pattern_risk = self.failure_avoidance.get("pattern_action_risk", {})
                if avoid_pattern in pattern_risk:
                    if action in pattern_risk[avoid_pattern]:
                        failure_rate = pattern_risk[avoid_pattern][action].get("failure_rate", 0)
                        if failure_rate >= 30:  # 閾値を下げて感度を上げる
                            penalty = min(failure_rate / 5, 15)  # 最大15点のペナルティ
                            score -= penalty
                            if f"回避パターン({avoid_pattern})" not in "、".join(reasons):
                                reasons.append(f"失敗率{failure_rate:.0f}%")

            reason_text = "、".join(reasons) if reasons else "標準"
            action_scores[action] = (score, reason_text)

        # スコア順にソート
        sorted_actions = sorted(
            action_scores.items(),
            key=lambda x: x[1][0],
            reverse=True
        )

        return [(action, score, reason) for action, (score, reason) in sorted_actions]

    def get_warnings(self) -> List[str]:
        """警告メッセージを生成"""
        warnings = []

        before_state = self.get_before_state_from_answers()
        avoid_pattern = None
        if "Q7" in self.answers:
            avoid_pattern = self.answers["Q7"].avoid_pattern

        # パターン警告
        if avoid_pattern:
            pattern_warnings = self.failure_avoidance.get("pattern_warnings", {})
            if avoid_pattern in pattern_warnings:
                pw = pattern_warnings[avoid_pattern]
                warnings.append(f"【{pw['description']}】{pw['warning']}")
                if pw.get("high_risk_actions"):
                    warnings.append(f"  避けるべき行動: {', '.join(pw['high_risk_actions'])}")

        # 高リスク状態×行動の警告
        high_risk = self.failure_avoidance.get("high_risk_state_action", [])
        for risk in high_risk:
            if risk.get("before_state") == before_state:
                if risk.get("failure_rate", 0) >= 50:
                    warnings.append(risk.get("warning", ""))

        return warnings

    def get_judgment_text(self) -> Tuple[str, str]:
        """判定テキストを取得"""
        momentum, _ = self.calculate_momentum()
        timing, _ = self.calculate_timing()

        # テンプレートキーを生成
        template_key = f"{timing}_{momentum}"

        templates = self.judgment_rules.get("output_templates", {})
        if template_key in templates:
            t = templates[template_key]
            return t.get("judgment", ""), t.get("detail", "")

        # デフォルト
        return "状況を見極めて行動を。", "慎重に判断する時期です。"

    def diagnose(self) -> DiagnosticResult:
        """診断を実行"""
        primary_hex, hex_scores = self.get_primary_hex()
        momentum, m_score = self.calculate_momentum()
        timing, t_score = self.calculate_timing()
        recommended_actions = self.calculate_action_scores()
        warnings = self.get_warnings()
        judgment, detail = self.get_judgment_text()

        # 追加情報
        before_state = self.get_before_state_from_answers()
        trigger_type = self.answers.get("Q4", DiagnosticAnswer("Q4", "")).trigger_type
        avoid_pattern = self.answers.get("Q7", DiagnosticAnswer("Q7", "")).avoid_pattern
        preferred_action = self.answers.get("Q8", DiagnosticAnswer("Q8", "")).preferred_action

        return DiagnosticResult(
            primary_hex=primary_hex,
            hex_scores=hex_scores,
            momentum=momentum,
            momentum_score=m_score,
            timing=timing,
            timing_score=t_score,
            recommended_actions=recommended_actions[:3],  # 上位3つ
            warnings=warnings,
            judgment=judgment,
            detail=detail,
            before_state=before_state,
            trigger_type=trigger_type,
            avoid_pattern=avoid_pattern,
            preferred_action=preferred_action
        )

    def reset(self):
        """回答をリセット"""
        self.answers.clear()


def format_result(result: DiagnosticResult) -> str:
    """診断結果をフォーマット"""
    lines = []
    lines.append("=" * 50)
    lines.append("【易経変化診断結果】")
    lines.append("=" * 50)
    lines.append("")

    # 位相
    lines.append(f"◆ あなたの位相（八卦）: {result.primary_hex}")
    if result.primary_hex in ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]:
        hex_names = {
            "乾": "創造・主導", "坤": "受容・育成", "震": "覚醒・始動",
            "巽": "浸透・適応", "坎": "試練・深化", "離": "明晰・発現",
            "艮": "停止・内省", "兌": "交流・喜悦"
        }
        lines.append(f"  ({hex_names.get(result.primary_hex, '')})")

    # 勢と時
    momentum_labels = {
        "ascending": "上昇の勢い", "stable": "安定",
        "descending": "下降の勢い", "chaotic": "混乱"
    }
    timing_labels = {
        "act_now": "動くべき時", "adapt": "適応すべき時", "wait": "待つべき時"
    }

    lines.append("")
    lines.append(f"◆ 勢: {momentum_labels.get(result.momentum, result.momentum)} (スコア: {result.momentum_score:.1f})")
    lines.append(f"◆ 時: {timing_labels.get(result.timing, result.timing)} (スコア: {result.timing_score:.1f})")

    # 判定
    lines.append("")
    lines.append("-" * 50)
    lines.append(f"【判定】{result.judgment}")
    lines.append(f"  {result.detail}")
    lines.append("-" * 50)

    # 推奨行動
    lines.append("")
    lines.append("◆ 推奨される行動（上位3つ）:")
    for i, (action, score, reason) in enumerate(result.recommended_actions, 1):
        lines.append(f"  {i}. {action} (スコア: {score:.1f})")
        lines.append(f"     理由: {reason}")

    # 警告
    if result.warnings:
        lines.append("")
        lines.append("◆ 注意事項:")
        for warning in result.warnings:
            lines.append(f"  ⚠ {warning}")

    lines.append("")
    lines.append("=" * 50)

    return "\n".join(lines)


def run_interactive_diagnosis():
    """対話形式で診断を実行"""
    engine = DiagnosticEngine()

    print("\n" + "=" * 50)
    print("易経変化診断システム v1.0")
    print("位相 × 勢 × 時 に基づく行動推奨")
    print("=" * 50 + "\n")

    for question in engine.get_questions():
        print(f"\n【{question['id']}】{question['text']}")
        print("-" * 40)

        for i, opt in enumerate(question["options"], 1):
            print(f"  {i}. {opt['label']}")

        while True:
            try:
                choice = input("\n選択してください (番号): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(question["options"]):
                    selected = question["options"][idx]
                    engine.record_answer(question["id"], selected["value"])
                    print(f"  → 「{selected['label']}」を選択しました")
                    break
                else:
                    print("  無効な番号です。もう一度入力してください。")
            except ValueError:
                print("  数字を入力してください。")
            except KeyboardInterrupt:
                print("\n\n診断を中断しました。")
                return

    # 診断実行
    print("\n\n診断中...")
    result = engine.diagnose()

    # 結果表示
    print(format_result(result))


if __name__ == "__main__":
    run_interactive_diagnosis()
