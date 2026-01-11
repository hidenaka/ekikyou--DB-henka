#!/usr/bin/env python3
"""
診断結果の出力フォーマット v4 - ストーリー重視版
統計を排し、事例の物語性と具体性を強化
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


# 行動タイプの詳細説明
ACTION_DETAILS = {
    "攻める・挑戦": {
        "meaning": "リスクを取って新しい領域に踏み出すこと",
        "typical_actions": [
            "新規事業・新市場への参入",
            "大型投資・設備拡大",
            "積極的な採用・組織拡大",
            "競合への直接対決",
        ],
        "requires": "資金・体力・時間の余裕があること",
        "risk": "失敗すると大きな損失を被る可能性",
    },
    "守る・維持": {
        "meaning": "今あるものを大切にし、基盤を固めること",
        "typical_actions": [
            "既存顧客との関係強化",
            "品質・サービスの改善",
            "コスト管理の徹底",
            "組織の結束強化",
        ],
        "requires": "守るべき価値があること",
        "risk": "変化に取り残される可能性",
    },
    "耐える・潜伏": {
        "meaning": "嵐が過ぎるまで動かず、力を蓄えること",
        "typical_actions": [
            "派手な動きを控える",
            "支出を最小限に抑える",
            "情報収集・学習に徹する",
            "体力・資金を温存する",
        ],
        "requires": "耐えられるだけの体力があること",
        "risk": "耐えている間に状況がさらに悪化する可能性",
    },
    "刷新・破壊": {
        "meaning": "古いものを壊して、根本から作り直すこと",
        "typical_actions": [
            "組織構造の抜本的な変更",
            "不採算事業からの撤退",
            "経営陣・キーパーソンの刷新",
            "ビジネスモデルの転換",
        ],
        "requires": "変える覚悟と、変えた後のビジョン",
        "risk": "混乱が長引く可能性、人心の離反",
    },
    "対話・融合": {
        "meaning": "他者と手を組み、力を合わせること",
        "typical_actions": [
            "パートナーシップの構築",
            "M&A・業務提携",
            "社外の知恵を借りる",
            "チームでの合意形成",
        ],
        "requires": "信頼できる相手がいること",
        "risk": "相手に依存しすぎる、主導権を失う可能性",
    },
    "捨てる・撤退": {
        "meaning": "損失を最小限に抑えて、手を引くこと",
        "typical_actions": [
            "赤字事業の売却・閉鎖",
            "不良資産の処分",
            "関係の清算",
            "早めの見切り",
        ],
        "requires": "執着を捨てる勇気",
        "risk": "タイミングを誤ると何も残らない",
    },
    "逃げる・放置": {
        "meaning": "問題から距離を置くこと",
        "typical_actions": [
            "一時的に離れて冷静になる",
            "別の場所で再起を図る",
            "関わらないことを選ぶ",
        ],
        "requires": "逃げる先があること",
        "risk": "問題が肥大化する可能性",
    },
    "分散・スピンオフ": {
        "meaning": "一つに賭けず、リスクを分けること",
        "typical_actions": [
            "事業の分社化",
            "投資先の分散",
            "複数の選択肢を並行で進める",
        ],
        "requires": "分散できるだけのリソース",
        "risk": "どれも中途半端になる可能性",
    },
}

# 状況の詳細説明
STATE_DETAILS = {
    "混乱・カオス": {
        "description": "何が起きているのか把握しきれない状態",
        "feeling": "先が見えない不安、何をしても裏目に出そうな恐怖",
        "typical_causes": ["突然の危機", "複数の問題の同時発生", "前提が崩れた"],
    },
    "どん底・危機": {
        "description": "最悪の状態、これ以上悪くなりようがない",
        "feeling": "絶望感、しかし同時に「もう失うものがない」という開き直り",
        "typical_causes": ["長期的な衰退の行き着く先", "致命的な失敗", "外部からの壊滅的打撃"],
    },
    "停滞・閉塞": {
        "description": "動きがなく、じわじわと悪化している状態",
        "feeling": "焦り、でも何をすればいいかわからない",
        "typical_causes": ["成功体験への固執", "変化への恐れ", "リーダーシップの不在"],
    },
    "成長痛": {
        "description": "成長しているが、その歪みが出ている状態",
        "feeling": "忙しさ、人手不足、仕組みが追いついていない感覚",
        "typical_causes": ["急成長", "想定以上の成功", "組織の肥大化"],
    },
    "安定・平和": {
        "description": "大きな問題がなく、穏やかな状態",
        "feeling": "安心感、しかし「このままでいいのか」という漠然とした不安",
        "typical_causes": ["過去の努力の成果", "環境が味方している"],
    },
    "絶頂・慢心": {
        "description": "最高の状態、うまくいきすぎている",
        "feeling": "自信、しかし客観視できなくなっている危険",
        "typical_causes": ["連続した成功", "競争優位の確立", "周囲からの称賛"],
    },
}

# 易経の知恵（パターン別）
ICHING_WISDOM = {
    "混乱・カオス": {
        "hexagram": "坎（水）",
        "teaching": "険難の中にあっても、誠実さを失わなければ道は開ける",
        "advice": "今は見通しが立たなくても、一つ一つ誠実に対応していくしかありません。焦って大きな賭けに出るより、まず足元を固めること。",
    },
    "どん底・危機": {
        "hexagram": "復（地雷復）",
        "teaching": "どん底まで落ちたものは、そこから再び上昇を始める",
        "advice": "これ以上悪くならないなら、あとは上がるだけ。この時期は「何を残し、何を捨てるか」を見極める貴重な機会です。",
    },
    "停滞・閉塞": {
        "hexagram": "否（天地否）",
        "teaching": "天地が交わらず、万物が通じない。しかし永遠には続かない",
        "advice": "停滞は必ず終わります。ただし、待っているだけでは終わりません。小さくても「動く」ことが、流れを変えるきっかけになります。",
    },
    "成長痛": {
        "hexagram": "大畜（山天大畜）",
        "teaching": "大きな力を蓄えている時、それを正しく使う知恵が要る",
        "advice": "成長の勢いがあるときこそ、立ち止まって仕組みを整える時間が必要です。走りながら考えるには限界があります。",
    },
    "安定・平和": {
        "hexagram": "泰（地天泰）",
        "teaching": "天地が交わり万物が通じる。しかし泰は否に転じやすい",
        "advice": "安定しているときこそ、次の変化の準備を。「平和なときに乱を忘れず」が古来の教えです。",
    },
    "絶頂・慢心": {
        "hexagram": "夬（沢天夬）",
        "teaching": "決壊は突然やってくる。満ちたものは欠け始める",
        "advice": "うまくいっているときほど、謙虚さが必要です。成功の理由を運ではなく実力だと思い始めたら、危険信号です。",
    },
}


def load_cases_for_story(
    before_state: Optional[str],
    trigger_type: Optional[str],
    limit: int = 10
) -> Dict[str, List[Dict]]:
    """成功事例と失敗事例を分けて取得"""
    cases_path = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"

    if not cases_path.exists():
        return {"success": [], "failure": [], "mixed": []}

    all_cases = []

    with open(cases_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                case = json.loads(line)

                # マッチング条件
                score = 0
                if before_state and case.get("before_state") == before_state:
                    score += 3
                if trigger_type and case.get("trigger_type") == trigger_type:
                    score += 2

                if score >= 3:  # 最低でもbefore_stateが一致
                    case["_match_score"] = score
                    all_cases.append(case)

            except json.JSONDecodeError:
                continue

    # 結果別に分類
    result = {"success": [], "failure": [], "mixed": []}

    for case in sorted(all_cases, key=lambda x: x.get("_match_score", 0), reverse=True):
        outcome = case.get("outcome", "")
        if outcome == "Success" and len(result["success"]) < limit:
            result["success"].append(case)
        elif outcome == "Failure" and len(result["failure"]) < limit:
            result["failure"].append(case)
        elif outcome in ["Mixed", "PartialSuccess"] and len(result["mixed"]) < limit:
            result["mixed"].append(case)

    return result


def format_case_detail(case: Dict, show_result: bool = True) -> List[str]:
    """事例を詳細なストーリー形式でフォーマット"""
    lines = []

    name = case.get("target_name", "ある組織")
    period = case.get("period", "")
    before = case.get("before_state", "")
    trigger = case.get("trigger_type", "")
    action = case.get("action_type", "")
    after = case.get("after_state", "")
    outcome = case.get("outcome", "")
    summary = case.get("story_summary", "")

    # ヘッダー
    lines.append(f"  ┌─────────────────────────────────")
    lines.append(f"  │ {name}")
    if period:
        lines.append(f"  │ （{period}）")
    lines.append(f"  └─────────────────────────────────")
    lines.append("")

    # ストーリー本文
    if summary:
        # 文を適切に分割して読みやすく
        lines.append(f"  {summary}")
        lines.append("")

    # 構造化された情報
    lines.append(f"  【直面した状況】{before}")
    if trigger:
        lines.append(f"  【きっかけ】{trigger}")
    lines.append(f"  【選んだ行動】{action}")

    # 行動の詳細説明
    if action in ACTION_DETAILS:
        detail = ACTION_DETAILS[action]
        lines.append(f"")
        lines.append(f"    → {detail['meaning']}")

    if show_result:
        lines.append(f"  【結果】{after}")
        if outcome == "Success":
            lines.append(f"    → 成功")
        elif outcome == "Failure":
            lines.append(f"    → 失敗")
        elif outcome in ["Mixed", "PartialSuccess"]:
            lines.append(f"    → 一部成功・課題も残る")

    lines.append("")
    return lines


def format_action_explanation(action_type: str) -> List[str]:
    """行動タイプの詳細な説明"""
    lines = []

    if action_type not in ACTION_DETAILS:
        return lines

    detail = ACTION_DETAILS[action_type]

    lines.append(f"  「{action_type}」とは")
    lines.append(f"")
    lines.append(f"    {detail['meaning']}")
    lines.append(f"")
    lines.append(f"    具体的には:")
    for act in detail["typical_actions"]:
        lines.append(f"      • {act}")
    lines.append(f"")
    lines.append(f"    必要なこと: {detail['requires']}")
    lines.append(f"    リスク: {detail['risk']}")
    lines.append("")

    return lines


def format_result_v4(result, engine) -> str:
    """ストーリー重視の診断結果フォーマット v4"""
    lines = []

    # 基本情報
    before_state = result.before_state
    trigger_type = result.trigger_type
    top_action = result.recommended_actions[0][0] if result.recommended_actions else None
    second_action = result.recommended_actions[1][0] if len(result.recommended_actions) > 1 else None

    # 事例を取得
    cases = load_cases_for_story(before_state, trigger_type)

    # ===== ヘッダー =====
    lines.append("")
    lines.append("━" * 55)
    lines.append("  あなたと似た状況を経験した人たちの物語")
    lines.append("━" * 55)
    lines.append("")

    # ===== あなたの状況 =====
    lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    lines.append("┃  あなたの今の状況                                ┃")
    lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    lines.append("")

    if before_state and before_state in STATE_DETAILS:
        state_info = STATE_DETAILS[before_state]
        lines.append(f"  ■ {before_state}")
        lines.append(f"")
        lines.append(f"    {state_info['description']}")
        lines.append(f"")
        lines.append(f"    こんな気持ちではありませんか？")
        lines.append(f"    「{state_info['feeling']}」")
        lines.append("")
    else:
        lines.append(f"  変化の渦中にいます。")
        lines.append("")

    if trigger_type:
        lines.append(f"  きっかけ: {trigger_type}")
        lines.append("")

    # ===== 成功した人の事例 =====
    if cases["success"]:
        lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
        lines.append("┃  この状況を乗り越えた人                          ┃")
        lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
        lines.append("")

        # 推奨行動と同じ行動を取った成功事例を優先
        success_with_recommended = [c for c in cases["success"] if c.get("action_type") == top_action]
        other_success = [c for c in cases["success"] if c.get("action_type") != top_action]

        shown = 0
        for case in (success_with_recommended + other_success)[:2]:
            lines.extend(format_case_detail(case))
            shown += 1

        if shown == 0:
            lines.append("  （類似事例を検索中...）")
            lines.append("")

    # ===== 失敗した人の事例 =====
    if cases["failure"]:
        lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
        lines.append("┃  同じ状況で失敗した人                            ┃")
        lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
        lines.append("")

        # 1件だけ表示
        for case in cases["failure"][:1]:
            lines.extend(format_case_detail(case))

            # なぜ失敗したかの考察
            failed_action = case.get("action_type", "")
            if failed_action and failed_action in ACTION_DETAILS:
                lines.append(f"    なぜ「{failed_action}」で失敗したのか？")
                lines.append(f"    → この状況で{ACTION_DETAILS[failed_action]['risk']}")
                lines.append("")

    # ===== 事例から見える共通点 =====
    if cases["success"]:
        lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
        lines.append("┃  成功した人に共通していたこと                    ┃")
        lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
        lines.append("")

        # 成功事例の行動を集計
        success_actions = {}
        for c in cases["success"]:
            act = c.get("action_type", "")
            if act:
                success_actions[act] = success_actions.get(act, 0) + 1

        if success_actions:
            most_common = max(success_actions, key=success_actions.get)
            lines.append(f"  多くの人が「{most_common}」を選んでいます。")
            lines.append("")

            if most_common in ACTION_DETAILS:
                lines.extend(format_action_explanation(most_common))

    # ===== 易経の知恵 =====
    if before_state and before_state in ICHING_WISDOM:
        wisdom = ICHING_WISDOM[before_state]
        lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
        lines.append("┃  古典の知恵                                       ┃")
        lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
        lines.append("")
        lines.append(f"  易経「{wisdom['hexagram']}」より")
        lines.append(f"")
        lines.append(f"    「{wisdom['teaching']}」")
        lines.append(f"")
        lines.append(f"  解釈:")
        lines.append(f"    {wisdom['advice']}")
        lines.append("")

    # ===== あなたへの問いかけ =====
    lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    lines.append("┃  あなたはどうしますか？                            ┃")
    lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    lines.append("")

    if top_action:
        lines.append(f"  事例を見る限り、「{top_action}」が")
        lines.append(f"  この状況を打開した人に多い選択でした。")
        lines.append("")

        if top_action in ACTION_DETAILS:
            detail = ACTION_DETAILS[top_action]
            lines.append(f"  もしこの道を選ぶなら:")
            for i, act in enumerate(detail["typical_actions"][:3], 1):
                lines.append(f"    {i}. {act}")
            lines.append("")
            lines.append(f"  ただし、{detail['risk']}に注意が必要です。")
            lines.append("")

    if second_action and second_action != top_action:
        lines.append(f"  別の選択肢として「{second_action}」もあります。")
        if second_action in ACTION_DETAILS:
            lines.append(f"  → {ACTION_DETAILS[second_action]['meaning']}")
        lines.append("")

    # ===== 警告 =====
    if result.avoid_pattern:
        avoid_names = {
            "Hubris_Collapse": "調子に乗りすぎて転落すること",
            "Slow_Decline": "気づかないうちにじわじわ衰退すること",
            "Shock_Recovery": "突然の打撃を受けて回復に苦しむこと",
            "Endurance": "耐えきれずに力尽きること",
            "Pivot_Success": "方向転換に失敗すること",
            "Steady_Growth": "成長が止まって取り残されること",
        }
        avoid_desc = avoid_names.get(result.avoid_pattern, result.avoid_pattern)

        lines.append("  ⚠ あなたが避けたいこと")
        lines.append(f"    「{avoid_desc}」")
        lines.append("")
        lines.append("    成功した人たちも、この落とし穴を意識していました。")
        lines.append("    うまくいき始めても、警戒を緩めないことが大切です。")
        lines.append("")

    # ===== 今週のチェックリスト =====
    lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    lines.append("┃  今週やってみること                                ┃")
    lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    lines.append("")
    lines.append("  □ 成功事例の人物・組織について、もう少し調べてみる")
    lines.append("  □ 自分の状況と、事例の状況の「違い」を考える")
    lines.append("  □ 「もし自分が同じ選択をするなら」を具体的に想像する")
    lines.append("  □ 信頼できる人に、この診断結果を見せて意見をもらう")
    lines.append("  □ 小さくてもいいので、一つだけ行動を起こす")
    lines.append("")

    lines.append("━" * 55)
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
    print(format_result_v4(result, engine))
