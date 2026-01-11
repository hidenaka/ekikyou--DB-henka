#!/usr/bin/env python3
"""
診断結果の出力フォーマット v2
一般ユーザー向けにわかりやすく変換
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# 行動タイプの具体例マッピング
ACTION_EXAMPLES = {
    "攻める・挑戦": {
        "summary": "積極的に動く",
        "examples": [
            "新しいプロジェクトを始める",
            "営業先を増やす・新規開拓する",
            "転職活動を開始する",
            "新商品・新サービスを投入する",
            "投資や設備拡大を行う",
        ],
        "one_liner": "今は攻めどき。新しいことを始めるチャンスです。"
    },
    "守る・維持": {
        "summary": "今あるものを大切にする",
        "examples": [
            "既存顧客との関係を深める",
            "品質管理を徹底する",
            "無駄な出費を見直す",
            "チームの結束を固める",
            "基本に立ち返って足場を固める",
        ],
        "one_liner": "今は守りの時期。焦らず足元を固めましょう。"
    },
    "耐える・潜伏": {
        "summary": "じっと待つ",
        "examples": [
            "派手な動きを控える",
            "情報収集に徹する",
            "スキルアップの時間に充てる",
            "体力・資金を温存する",
            "嵐が過ぎるのを待つ",
        ],
        "one_liner": "今は動かない方が得策。力を蓄える時期です。"
    },
    "刷新・破壊": {
        "summary": "古いものを捨てて新しくする",
        "examples": [
            "組織体制を抜本的に見直す",
            "不採算事業から撤退する",
            "これまでのやり方を全て変える",
            "人間関係をリセットする",
            "ゼロベースで考え直す",
        ],
        "one_liner": "思い切った変化が必要。過去を手放す覚悟を。"
    },
    "対話・融合": {
        "summary": "人と協力する",
        "examples": [
            "パートナーや協力者を探す",
            "異業種の人と話してみる",
            "チームでブレストする",
            "メンターや相談相手を見つける",
            "競合と協業を検討する",
        ],
        "one_liner": "一人で抱えず、人の力を借りる時期です。"
    },
    "捨てる・撤退": {
        "summary": "損切りする",
        "examples": [
            "うまくいかないプロジェクトを止める",
            "赤字部門を閉鎖する",
            "続けても意味のない関係を整理する",
            "執着を手放す",
            "早めに見切りをつける",
        ],
        "one_liner": "続けるより、やめる勇気が必要な時期です。"
    },
    "逃げる・放置": {
        "summary": "距離を置く",
        "examples": [
            "問題から一度離れて冷静になる",
            "休息を取る",
            "別の場所で再起を図る",
        ],
        "one_liner": "今は関わらない方がいいかもしれません。"
    },
    "分散・スピンオフ": {
        "summary": "分けて独立させる",
        "examples": [
            "事業を分社化する",
            "リスクを分散させる",
            "複数の選択肢を並行で進める",
            "一つに賭けず複線化する",
        ],
        "one_liner": "一極集中より、分散がリスクヘッジになります。"
    },
}

# 状況の説明テンプレート
SITUATION_TEMPLATES = {
    ("ascending", "act_now"): {
        "title": "追い風が吹いています",
        "description": "状況は上向きで、見通しも立っています。今動けば成果が出やすい時期です。",
        "advice": "このチャンスを逃さず、積極的に行動しましょう。ただし調子に乗りすぎないよう注意。"
    },
    ("ascending", "adapt"): {
        "title": "良い流れの中にいます",
        "description": "状況は良い方向に動いていますが、まだタイミングを見計らう必要があります。",
        "advice": "焦らず、周囲と歩調を合わせながら進みましょう。"
    },
    ("ascending", "wait"): {
        "title": "上り調子ですが、まだ早い",
        "description": "状況は改善していますが、行動に移すにはもう少し準備が必要です。",
        "advice": "もう少し状況を見極めてから動きましょう。"
    },
    ("stable", "act_now"): {
        "title": "安定した中でのチャンス",
        "description": "大きな変化はありませんが、動くには良いタイミングです。",
        "advice": "現状維持に甘んじず、次のステップを踏み出しましょう。"
    },
    ("stable", "adapt"): {
        "title": "穏やかな時期",
        "description": "特に急ぐ必要はありません。じっくり考える余裕があります。",
        "advice": "この時間を使って、次に向けた準備を進めましょう。"
    },
    ("stable", "wait"): {
        "title": "静かに待つ時期",
        "description": "今は大きく動くより、現状を維持する方が賢明です。",
        "advice": "無理に変化を起こさず、機が熟すのを待ちましょう。"
    },
    ("descending", "act_now"): {
        "title": "厳しい中でも動くべき時",
        "description": "状況は厳しいですが、今動かないともっと悪くなる可能性があります。",
        "advice": "痛みを伴っても、必要な手を打つ時期です。"
    },
    ("descending", "adapt"): {
        "title": "下り坂を歩いています",
        "description": "状況は少しずつ厳しくなっています。柔軟に対応する必要があります。",
        "advice": "現実を直視し、早めに軌道修正を考えましょう。"
    },
    ("descending", "wait"): {
        "title": "試練の時期",
        "description": "今は耐える時期です。無理に動くと傷が深くなります。",
        "advice": "守りを固め、嵐が過ぎるのを待ちましょう。"
    },
    ("chaotic", "act_now"): {
        "title": "混乱の中でも決断を",
        "description": "状況は混沌としていますが、何かを変えないと抜け出せません。",
        "advice": "完璧を求めず、まず一歩を踏み出すことが大切です。"
    },
    ("chaotic", "adapt"): {
        "title": "混乱期",
        "description": "先が見えにくい状況です。状況に合わせて柔軟に動く必要があります。",
        "advice": "大きな決断は避け、小さく試しながら方向を探りましょう。"
    },
    ("chaotic", "wait"): {
        "title": "嵐の中にいます",
        "description": "今は何をしても難しい時期です。動くより耐える方が賢明です。",
        "advice": "まず生き延びることを優先してください。"
    },
}

# 回避パターンの説明
AVOID_PATTERN_ADVICE = {
    "Hubris_Collapse": {
        "name": "調子に乗って失敗",
        "description": "成功が続くと油断が生まれ、致命的なミスを犯しやすくなります。",
        "warning": "「自分は大丈夫」と思った時が一番危ない。謙虚さを忘れずに。",
        "historical": "過去の事例では、絶頂期に攻めすぎて崩壊したケースが多くあります。"
    },
    "Slow_Decline": {
        "name": "じわじわ衰退",
        "description": "小さな問題を放置し続けると、気づいた時には手遅れになります。",
        "warning": "「まだ大丈夫」は危険なサイン。早めの対策が命運を分けます。",
        "historical": "衰退に気づいても現状維持を選んだ組織は、ほとんどが回復できませんでした。"
    },
    "Shock_Recovery": {
        "name": "突然のショック",
        "description": "予期せぬ出来事に見舞われ、回復に時間がかかるパターンです。",
        "warning": "備えがないと、ショックからの回復が非常に困難になります。",
        "historical": "事前にリスク分散していた組織は、ショックからの回復が早い傾向があります。"
    },
    "Endurance": {
        "name": "耐えきれずに崩壊",
        "description": "長期戦を強いられ、体力・気力・資金が尽きてしまうパターンです。",
        "warning": "「もう少しだけ」の繰り返しが致命傷になることがあります。",
        "historical": "撤退のタイミングを見誤ると、全てを失うことがあります。"
    },
    "Pivot_Success": {
        "name": "方向転換の失敗",
        "description": "新しい方向に舵を切ったものの、うまくいかないパターンです。",
        "warning": "変化そのものは正しくても、タイミングと準備が重要です。",
        "historical": "成功した方向転換には、十分な準備期間がありました。"
    },
    "Steady_Growth": {
        "name": "成長の停滞",
        "description": "安定に甘んじて成長が止まり、徐々に取り残されるパターンです。",
        "warning": "現状維持は後退と同じ。常に次の一手を考える必要があります。",
        "historical": "成長を続けた組織は、安定期にも小さな挑戦を続けていました。"
    },
}


def load_similar_cases(before_state: str, action_type: str, limit: int = 3) -> List[Dict]:
    """類似事例をデータベースから取得"""
    cases_path = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"

    if not cases_path.exists():
        return []

    similar_cases = []

    with open(cases_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                case = json.loads(line)
                # 条件マッチング
                if (case.get("before_state") == before_state and
                    case.get("action_type") == action_type and
                    case.get("outcome") == "Success"):
                    similar_cases.append({
                        "name": case.get("target_name", "不明"),
                        "summary": case.get("story_summary", ""),
                        "period": case.get("period", ""),
                        "action": case.get("action_type", ""),
                    })
                    if len(similar_cases) >= limit:
                        break
            except json.JSONDecodeError:
                continue

    return similar_cases


def format_result_v2(result, engine) -> str:
    """診断結果をわかりやすい形式でフォーマット"""
    lines = []

    # ヘッダー
    lines.append("")
    lines.append("━" * 50)
    lines.append("📊 あなたの診断結果")
    lines.append("━" * 50)
    lines.append("")

    # 1. 状況の要約（最も重要）
    situation_key = (result.momentum, result.timing)
    situation = SITUATION_TEMPLATES.get(situation_key, {
        "title": "状況を見極める時期",
        "description": "慎重に判断する必要があります。",
        "advice": "焦らず、状況を見ながら進みましょう。"
    })

    lines.append(f"【今のあなたの状況】")
    lines.append(f"")
    lines.append(f"  🔹 {situation['title']}")
    lines.append(f"")
    lines.append(f"  {situation['description']}")
    lines.append(f"")
    lines.append(f"  💡 {situation['advice']}")
    lines.append("")

    # 2. 推奨される行動（具体例付き）
    lines.append("─" * 50)
    lines.append("")
    lines.append("【おすすめの行動】")
    lines.append("")

    top_action = result.recommended_actions[0][0] if result.recommended_actions else None

    if top_action and top_action in ACTION_EXAMPLES:
        action_info = ACTION_EXAMPLES[top_action]
        lines.append(f"  ✅ {action_info['summary']}")
        lines.append(f"")
        lines.append(f"  {action_info['one_liner']}")
        lines.append(f"")
        lines.append(f"  例えば...")
        for ex in action_info["examples"][:3]:
            lines.append(f"    • {ex}")

    lines.append("")

    # 3. 次点の選択肢
    if len(result.recommended_actions) >= 2:
        second_action = result.recommended_actions[1][0]
        if second_action in ACTION_EXAMPLES:
            lines.append(f"  📌 もう一つの選択肢: {ACTION_EXAMPLES[second_action]['summary']}")

    lines.append("")

    # 4. 避けるべきこと（警告）
    if result.avoid_pattern and result.avoid_pattern in AVOID_PATTERN_ADVICE:
        avoid_info = AVOID_PATTERN_ADVICE[result.avoid_pattern]
        lines.append("─" * 50)
        lines.append("")
        lines.append("【注意してください】")
        lines.append("")
        lines.append(f"  ⚠️ 「{avoid_info['name']}」を避けたいとのこと")
        lines.append(f"")
        lines.append(f"  {avoid_info['description']}")
        lines.append(f"")
        lines.append(f"  📍 {avoid_info['warning']}")
        lines.append("")

    # 5. 類似事例（あれば）
    if result.before_state and top_action:
        similar_cases = load_similar_cases(result.before_state, top_action, limit=2)
        if similar_cases:
            lines.append("─" * 50)
            lines.append("")
            lines.append("【似た状況で成功した事例】")
            lines.append("")
            for case in similar_cases:
                lines.append(f"  📖 {case['name']}")
                if case['summary']:
                    # 要約が長すぎる場合は切り詰める
                    summary = case['summary'][:80] + "..." if len(case['summary']) > 80 else case['summary']
                    lines.append(f"     {summary}")
                lines.append("")

    # 6. 今週やること
    lines.append("─" * 50)
    lines.append("")
    lines.append("【今週やること】")
    lines.append("")

    if top_action and top_action in ACTION_EXAMPLES:
        examples = ACTION_EXAMPLES[top_action]["examples"]
        lines.append(f"  1. まず「{examples[0]}」から始めてみる")
        if len(examples) > 1:
            lines.append(f"  2. できれば「{examples[1]}」も検討する")
        lines.append(f"  3. 1週間後に状況を振り返る")

    lines.append("")
    lines.append("━" * 50)
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    # テスト実行
    from diagnostic_engine import DiagnosticEngine

    engine = DiagnosticEngine()

    # サンプル回答
    answers = [
        ('Q1', 'active_mild'),
        ('Q2', 'outward_expand'),
        ('Q3', 'clear_certain'),
        ('Q4', 'intentional'),
        ('Q5', 'power_influence'),
        ('Q6', 'nothing'),
        ('Q7', 'hubris_collapse'),
        ('Q8', 'growth'),
    ]

    for qid, value in answers:
        engine.record_answer(qid, value)

    result = engine.diagnose()
    print(format_result_v2(result, engine))
