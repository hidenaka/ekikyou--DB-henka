#!/usr/bin/env python3
"""
変化のロジック：予測API

シンプルなAPIとして使用可能。
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "data" / "models"

_model = None


def _load_model():
    global _model
    if _model is None:
        with open(MODELS_DIR / "prediction_model_v1.json", "r", encoding="utf-8") as f:
            _model = json.load(f)
    return _model


def predict_outcome(before_state: str, action_type: str) -> dict:
    """
    状態と行動から結果を予測

    Args:
        before_state: 現在の状態
            - "絶頂・慢心", "安定・平和", "成長痛", "停滞・閉塞"
            - "混乱・カオス", "どん底・危機", "安定成長・成功", "成長・拡大"
        action_type: 取ろうとしている行動
            - "攻める・挑戦", "守る・維持", "捨てる・撤退", "耐える・潜伏"
            - "対話・融合", "刷新・破壊", "逃げる・放置", "分散・スピンオフ"

    Returns:
        {
            "prediction": "Success" | "PartialSuccess" | "Mixed" | "Failure",
            "confidence": 0.0-1.0,
            "confidence_level": "高" | "中" | "低",
            "yao_position": 1-6,
            "yao_stage": "発芽期・始動期" など,
            "distribution": {"Success": 0.x, "Failure": 0.x, ...}
        }
    """
    model = _load_model()

    # 行動を正規化
    action_normalize = model.get("action_normalize", {})
    normalized_action = action_normalize.get(action_type, action_type)

    # 爻位を診断
    state_to_yao = model.get("state_to_yao", {})
    yao_position = state_to_yao.get(before_state, 3)

    yao_stages = {
        1: "発芽期・始動期",
        2: "成長期・基盤確立期",
        3: "転換期・岐路",
        4: "成熟期・接近期",
        5: "全盛期・リーダー期",
        6: "衰退期・転換期・極み",
    }

    # 確率モデルで予測
    prob_model = model["models"]["probability"]["data"]
    key = str((before_state, normalized_action))

    if key in prob_model["conditional"]:
        counts = prob_model["conditional"][key]
        total = sum(counts.values())
        best_outcome = max(counts, key=counts.get)
        confidence = counts[best_outcome] / total
        distribution = {k: v/total for k, v in counts.items()}
    elif before_state in prob_model["marginal_state"]:
        counts = prob_model["marginal_state"][before_state]
        total = sum(counts.values())
        best_outcome = max(counts, key=counts.get)
        confidence = counts[best_outcome] / total * 0.7
        distribution = {k: v/total for k, v in counts.items()}
    else:
        counts = prob_model["overall"]
        total = sum(counts.values())
        best_outcome = max(counts, key=counts.get)
        confidence = counts[best_outcome] / total * 0.5
        distribution = {k: v/total for k, v in counts.items()}

    # 確信度レベル
    if confidence >= 0.6:
        confidence_level = "高"
    elif confidence >= 0.4:
        confidence_level = "中"
    else:
        confidence_level = "低"

    return {
        "prediction": best_outcome,
        "confidence": round(confidence, 3),
        "confidence_level": confidence_level,
        "yao_position": yao_position,
        "yao_stage": yao_stages.get(yao_position, "不明"),
        "distribution": {k: round(v, 3) for k, v in distribution.items()},
    }


def get_recommendation(before_state: str) -> dict:
    """
    現在の状態に対する推奨行動を取得

    Returns:
        {
            "yao_position": 1-6,
            "yao_stage": "発芽期・始動期" など,
            "recommended_actions": ["行動1", "行動2", ...],
            "avoid_actions": ["行動1", "行動2", ...],
            "best_action": "最も成功確率が高い行動",
            "best_action_success_rate": 0.0-1.0
        }
    """
    model = _load_model()

    state_to_yao = model.get("state_to_yao", {})
    yao_position = state_to_yao.get(before_state, 3)

    yao_stages = {
        1: "発芽期・始動期",
        2: "成長期・基盤確立期",
        3: "転換期・岐路",
        4: "成熟期・接近期",
        5: "全盛期・リーダー期",
        6: "衰退期・転換期・極み",
    }

    # 各行動の成功確率を計算
    actions = [
        "攻める・挑戦", "守る・維持", "捨てる・撤退", "耐える・潜伏",
        "対話・融合", "刷新・破壊", "逃げる・放置", "分散・スピンオフ"
    ]

    action_scores = []
    for action in actions:
        result = predict_outcome(before_state, action)
        success_prob = result["distribution"].get("Success", 0) + \
                       result["distribution"].get("PartialSuccess", 0) * 0.5
        action_scores.append((action, success_prob, result["confidence"]))

    # ソート
    action_scores.sort(key=lambda x: -x[1])

    # 推奨と回避
    recommended = [a[0] for a in action_scores[:3]]
    avoid = [a[0] for a in action_scores[-2:]]

    return {
        "yao_position": yao_position,
        "yao_stage": yao_stages.get(yao_position, "不明"),
        "recommended_actions": recommended,
        "avoid_actions": avoid,
        "best_action": action_scores[0][0],
        "best_action_success_rate": round(action_scores[0][1], 3),
    }


# テスト
if __name__ == "__main__":
    print("=== 予測テスト ===")
    result = predict_outcome("停滞・閉塞", "刷新・破壊")
    print(f"状態: 停滞・閉塞 → 行動: 刷新・破壊")
    print(f"予測: {result['prediction']} (確信度{result['confidence']*100:.0f}%)")
    print(f"爻位: {result['yao_position']}爻 ({result['yao_stage']})")
    print()

    print("=== 推奨行動テスト ===")
    rec = get_recommendation("どん底・危機")
    print(f"状態: どん底・危機 ({rec['yao_position']}爻)")
    print(f"推奨: {rec['recommended_actions']}")
    print(f"回避: {rec['avoid_actions']}")
    print(f"最良: {rec['best_action']} (成功率{rec['best_action_success_rate']*100:.0f}%)")
