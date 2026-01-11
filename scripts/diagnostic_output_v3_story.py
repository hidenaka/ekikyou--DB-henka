#!/usr/bin/env python3
"""
診断結果の出力フォーマット v3 - ストーリー型
実例を中心に据えて説得力を持たせる
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


def load_matching_cases(
    before_state: Optional[str],
    trigger_type: Optional[str],
    action_type: Optional[str],
    outcome: Optional[str] = None,
    limit: int = 5
) -> List[Dict]:
    """条件に合う事例をデータベースから取得"""
    cases_path = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"

    if not cases_path.exists():
        return []

    matching_cases = []

    with open(cases_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                case = json.loads(line)

                # スコアリング
                score = 0
                if before_state and case.get("before_state") == before_state:
                    score += 3
                if trigger_type and case.get("trigger_type") == trigger_type:
                    score += 2
                if action_type and case.get("action_type") == action_type:
                    score += 2
                if outcome and case.get("outcome") == outcome:
                    score += 1

                # 最低2つ以上の条件がマッチ
                if score >= 3:
                    case["_match_score"] = score
                    matching_cases.append(case)

            except json.JSONDecodeError:
                continue

    # スコア順にソート
    matching_cases.sort(key=lambda x: x.get("_match_score", 0), reverse=True)

    return matching_cases[:limit]


def analyze_outcomes(cases: List[Dict], action_type: str) -> Dict:
    """同じ行動を取った事例の結果を分析"""
    same_action = [c for c in cases if c.get("action_type") == action_type]

    if not same_action:
        return {"total": 0}

    outcomes = defaultdict(int)
    for c in same_action:
        outcomes[c.get("outcome", "Unknown")] += 1

    total = len(same_action)
    success_rate = outcomes.get("Success", 0) / total * 100 if total > 0 else 0

    return {
        "total": total,
        "success": outcomes.get("Success", 0),
        "failure": outcomes.get("Failure", 0),
        "mixed": outcomes.get("Mixed", 0),
        "success_rate": success_rate
    }


def get_contrast_case(
    before_state: Optional[str],
    trigger_type: Optional[str],
    recommended_action: str
) -> Optional[Dict]:
    """推奨行動と違う行動を取って失敗した事例を取得"""
    cases_path = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"

    if not cases_path.exists():
        return None

    with open(cases_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                case = json.loads(line)

                # 同じ状況だが、違う行動を取って失敗した事例
                if (case.get("before_state") == before_state and
                    case.get("action_type") != recommended_action and
                    case.get("outcome") == "Failure"):
                    return case

            except json.JSONDecodeError:
                continue

    return None


def format_case_story(case: Dict, is_success: bool = True) -> List[str]:
    """事例をストーリー形式でフォーマット"""
    lines = []

    name = case.get("target_name", "ある組織")
    period = case.get("period", "")
    before = case.get("before_state", "")
    trigger = case.get("trigger_type", "")
    action = case.get("action_type", "")
    after = case.get("after_state", "")
    summary = case.get("story_summary", "")

    icon = "✅" if is_success else "❌"

    lines.append(f"  {icon} {name}")
    if period:
        lines.append(f"     時期: {period}")

    # ストーリーを組み立て
    if summary:
        # 80文字で折り返し
        if len(summary) > 100:
            summary = summary[:100] + "..."
        lines.append(f"")
        lines.append(f"     {summary}")

    lines.append(f"")
    lines.append(f"     状況: {before} → きっかけ: {trigger}")
    lines.append(f"     行動: 「{action}」")
    lines.append(f"     結果: {after}")

    return lines


def format_result_story(result, engine) -> str:
    """ストーリー型の診断結果フォーマット"""
    lines = []

    # 基本情報を取得
    before_state = result.before_state
    trigger_type = result.trigger_type
    top_action = result.recommended_actions[0][0] if result.recommended_actions else None

    # ヘッダー
    lines.append("")
    lines.append("━" * 50)
    lines.append("📖 あなたと似た人の物語")
    lines.append("━" * 50)
    lines.append("")

    # 1. あなたの状況を要約
    lines.append("【あなたの状況】")
    lines.append("")

    situation_parts = []
    if before_state:
        situation_parts.append(f"「{before_state}」の状態")
    if trigger_type:
        situation_parts.append(f"「{trigger_type}」がきっかけ")

    if situation_parts:
        lines.append(f"  {' で '.join(situation_parts)} にいます。")
    else:
        lines.append(f"  変化の渦中にいます。")
    lines.append("")

    # 2. 成功事例を取得
    success_cases = load_matching_cases(
        before_state=before_state,
        trigger_type=trigger_type,
        action_type=top_action,
        outcome="Success",
        limit=2
    )

    # 3. 成功事例を表示
    if success_cases:
        lines.append("─" * 50)
        lines.append("")
        lines.append("【似た状況で成功した人】")
        lines.append("")

        for case in success_cases[:2]:
            lines.extend(format_case_story(case, is_success=True))
            lines.append("")

        # 共通点を抽出
        lines.append("  📍 共通点:")
        lines.append(f"     ・「{before_state}」という状況")
        if trigger_type:
            lines.append(f"     ・「{trigger_type}」というきっかけ")
        lines.append(f"     ・「{top_action}」という行動を選択")
        lines.append("")

    # 4. 失敗事例（対照）
    contrast_case = get_contrast_case(before_state, trigger_type, top_action)
    if contrast_case:
        lines.append("─" * 50)
        lines.append("")
        lines.append("【同じ状況で失敗した人】")
        lines.append("")
        lines.extend(format_case_story(contrast_case, is_success=False))
        lines.append("")
        lines.append(f"  💡 この人は「{contrast_case.get('action_type')}」を選んで失敗しました。")
        lines.append("")

    # 5. 統計的な裏付け
    all_similar = load_matching_cases(
        before_state=before_state,
        trigger_type=trigger_type,
        action_type=None,
        limit=100
    )

    if all_similar and top_action:
        stats = analyze_outcomes(all_similar, top_action)
        if stats["total"] >= 3:
            lines.append("─" * 50)
            lines.append("")
            lines.append("【データが示すこと】")
            lines.append("")
            lines.append(f"  あなたと似た状況で「{top_action}」を選んだ人は")
            lines.append(f"  {stats['total']}件中 {stats['success']}件が成功しています。")
            lines.append(f"  （成功率: {stats['success_rate']:.0f}%）")
            lines.append("")

    # 6. あなたへの提案
    lines.append("─" * 50)
    lines.append("")
    lines.append("【だから、あなたには】")
    lines.append("")
    lines.append(f"  ➡️ 「{top_action}」をおすすめします。")
    lines.append("")

    # 行動の具体例
    ACTION_EXAMPLES = {
        "攻める・挑戦": ["新しいことを始める", "営業先を増やす", "投資する"],
        "守る・維持": ["既存顧客を大切にする", "品質を磨く", "足場を固める"],
        "耐える・潜伏": ["派手な動きを控える", "情報収集に徹する", "力を蓄える"],
        "刷新・破壊": ["組織を見直す", "不採算を切る", "ゼロから考え直す"],
        "対話・融合": ["協力者を探す", "人に相談する", "チームで考える"],
        "捨てる・撤退": ["やめる決断をする", "損切りする", "執着を手放す"],
        "逃げる・放置": ["距離を置く", "休息を取る", "別の場所で再起"],
        "分散・スピンオフ": ["リスクを分散する", "複数の選択肢を持つ"],
    }

    if top_action in ACTION_EXAMPLES:
        lines.append("  具体的には...")
        for ex in ACTION_EXAMPLES[top_action][:3]:
            lines.append(f"    • {ex}")
        lines.append("")

    # 7. 今週のチェックリスト
    lines.append("─" * 50)
    lines.append("")
    lines.append("【今週のチェックリスト】")
    lines.append("")
    lines.append("  □ 成功事例を1つ詳しく調べてみる")

    if top_action in ACTION_EXAMPLES:
        lines.append(f"  □ 「{ACTION_EXAMPLES[top_action][0]}」について考える")
        lines.append(f"  □ 最初の一歩を決める（小さくてOK）")

    lines.append("  □ 1週間後に振り返る")
    lines.append("")

    # 8. 警告（あれば）
    if result.avoid_pattern:
        avoid_names = {
            "Hubris_Collapse": "調子に乗って失敗",
            "Slow_Decline": "じわじわ衰退",
            "Shock_Recovery": "突然のショック",
            "Endurance": "耐えきれずに崩壊",
            "Pivot_Success": "方向転換の失敗",
            "Steady_Growth": "成長の停滞",
        }
        avoid_name = avoid_names.get(result.avoid_pattern, result.avoid_pattern)

        lines.append("─" * 50)
        lines.append("")
        lines.append("【気をつけること】")
        lines.append("")
        lines.append(f"  ⚠️ 「{avoid_name}」を避けたいとのこと。")
        lines.append("")
        lines.append(f"  成功事例の人たちも、この落とし穴には注意していました。")
        lines.append(f"  うまくいっている時ほど、慎重に。")
        lines.append("")

    lines.append("━" * 50)
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    from diagnostic_engine import DiagnosticEngine

    engine = DiagnosticEngine()

    answers = [
        ('Q1', 'static_stuck'),
        ('Q2', 'inward_protect'),
        ('Q3', 'unclear_danger'),
        ('Q4', 'external_shock'),
        ('Q5', 'pressure'),
        ('Q6', 'resources'),
        ('Q7', 'slow_decline'),
        ('Q8', 'renewal'),
    ]

    for qid, value in answers:
        engine.record_answer(qid, value)

    result = engine.diagnose()
    print(format_result_story(result, engine))
