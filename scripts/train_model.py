#!/usr/bin/env python3
"""
純粋予測モデルの訓練スクリプト

pattern_typeを使わず、before_state + action_type のみから
outcomeを予測するモデルを構築する。
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
SPLITS_DIR = BASE_DIR / "data" / "splits"
MODELS_DIR = BASE_DIR / "data" / "models"


def load_cases(filepath):
    """ケースデータを読み込み"""
    cases = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


# 状態→爻位マッピング（pattern_typeなし版）
STATE_TO_YAO_PURE = {
    "絶頂・慢心": 6,
    "安定・平和": 5,
    "成長痛": 3,
    "停滞・閉塞": 4,
    "混乱・カオス": 3,
    "どん底・危機": 1,
    "混乱・衰退": 3,
    "安定・停止": 4,
    "安定成長・成功": 2,
    "拡大・繁栄": 5,
    "成長・拡大": 2,
    "縮小安定・生存": 6,
    "調和・繁栄": 5,
    "V字回復・大成功": 5,
    "急成長・拡大": 2,
}

# 行動正規化
ACTION_NORMALIZE = {
    "攻める・挑戦": "攻める・挑戦",
    "拡大・攻め": "攻める・挑戦",
    "集中・拡大": "攻める・挑戦",
    "輝く・表現": "攻める・挑戦",
    "守る・維持": "守る・維持",
    "逃げる・守る": "守る・維持",
    "捨てる・撤退": "捨てる・撤退",
    "捨てる・転換": "捨てる・撤退",
    "撤退・収縮": "捨てる・撤退",
    "撤退・縮小": "捨てる・撤退",
    "耐える・潜伏": "耐える・潜伏",
    "対話・融合": "対話・融合",
    "交流・発表": "対話・融合",
    "刷新・破壊": "刷新・破壊",
    "逃げる・放置": "逃げる・放置",
    "撤退・逃げる": "逃げる・放置",
    "逃げる・分散": "逃げる・放置",
    "分散・スピンオフ": "分散・スピンオフ",
    "分散・探索": "分散・スピンオフ",
    "分散・多角化": "分散・スピンオフ",
    "分散・独立": "分散・スピンオフ",
    "分散する・独立する": "分散・スピンオフ",
}


def normalize_action(action_type):
    """行動タイプを正規化"""
    return ACTION_NORMALIZE.get(action_type, action_type)


def diagnose_yao_position_pure(before_state):
    """
    純粋な爻位診断（pattern_typeを使わない）
    """
    return STATE_TO_YAO_PURE.get(before_state, 3)


def build_probability_model(train_cases):
    """
    訓練データから確率モデルを構築

    P(outcome | before_state, action_type) を推定
    """
    # 条件付き頻度カウント
    # key: (before_state, normalized_action)
    # value: Counter of outcomes
    conditional_counts = defaultdict(Counter)

    # 周辺分布
    marginal_by_state = defaultdict(Counter)
    marginal_by_action = defaultdict(Counter)
    overall_counts = Counter()

    for case in train_cases:
        before_state = case.get("before_state", "")
        action_type = case.get("action_type", "")
        outcome = case.get("outcome", "Mixed")

        normalized_action = normalize_action(action_type)

        # 条件付き
        key = (before_state, normalized_action)
        conditional_counts[key][outcome] += 1

        # 周辺
        marginal_by_state[before_state][outcome] += 1
        marginal_by_action[normalized_action][outcome] += 1
        overall_counts[outcome] += 1

    return {
        "conditional": {str(k): dict(v) for k, v in conditional_counts.items()},
        "marginal_state": {k: dict(v) for k, v in marginal_by_state.items()},
        "marginal_action": {k: dict(v) for k, v in marginal_by_action.items()},
        "overall": dict(overall_counts),
    }


def build_yao_action_model(train_cases):
    """
    爻位×行動モデルを構築

    訓練データから爻位×行動の組み合わせごとの
    outcome分布を学習
    """
    yao_action_counts = defaultdict(Counter)

    for case in train_cases:
        before_state = case.get("before_state", "")
        action_type = case.get("action_type", "")
        outcome = case.get("outcome", "Mixed")

        yao_pos = diagnose_yao_position_pure(before_state)
        normalized_action = normalize_action(action_type)

        key = (yao_pos, normalized_action)
        yao_action_counts[key][outcome] += 1

    # 各組み合わせで最頻出outcomeと確信度を計算
    yao_action_model = {}
    for key, counts in yao_action_counts.items():
        total = sum(counts.values())
        most_common = counts.most_common(1)[0]

        yao_action_model[str(key)] = {
            "predicted_outcome": most_common[0],
            "confidence": most_common[1] / total,
            "distribution": dict(counts),
            "total_samples": total,
        }

    return yao_action_model


def evaluate_model(model, cases, model_type="probability"):
    """
    モデルを評価
    """
    stats = {"exact": 0, "close": 0, "miss": 0}
    predictions = []

    for case in cases:
        before_state = case.get("before_state", "")
        action_type = case.get("action_type", "")
        actual = case.get("outcome", "Mixed")

        # 予測
        if model_type == "probability":
            predicted, confidence = predict_probability(model, before_state, action_type)
        else:
            predicted, confidence = predict_yao_action(model, before_state, action_type)

        # 評価
        if predicted == actual:
            accuracy = "exact"
            stats["exact"] += 1
        elif is_close(predicted, actual):
            accuracy = "close"
            stats["close"] += 1
        else:
            accuracy = "miss"
            stats["miss"] += 1

        predictions.append({
            "before_state": before_state,
            "action_type": action_type,
            "predicted": predicted,
            "actual": actual,
            "confidence": confidence,
            "accuracy": accuracy,
        })

    total = len(cases)
    return {
        "total": total,
        "exact": stats["exact"],
        "close": stats["close"],
        "miss": stats["miss"],
        "exact_rate": stats["exact"] / total * 100 if total > 0 else 0,
        "accuracy_score": (stats["exact"] + stats["close"]) / total * 100 if total > 0 else 0,
        "predictions": predictions,
    }


def predict_probability(model, before_state, action_type):
    """
    確率モデルで予測
    """
    normalized_action = normalize_action(action_type)
    key = str((before_state, normalized_action))

    # 条件付き分布を確認
    if key in model["conditional"]:
        counts = model["conditional"][key]
        total = sum(counts.values())
        best_outcome = max(counts, key=counts.get)
        confidence = counts[best_outcome] / total
        return best_outcome, confidence

    # 周辺分布にフォールバック
    if before_state in model["marginal_state"]:
        counts = model["marginal_state"][before_state]
        total = sum(counts.values())
        best_outcome = max(counts, key=counts.get)
        confidence = counts[best_outcome] / total * 0.7  # 確信度を下げる
        return best_outcome, confidence

    # 全体分布にフォールバック
    counts = model["overall"]
    total = sum(counts.values())
    best_outcome = max(counts, key=counts.get)
    confidence = counts[best_outcome] / total * 0.5
    return best_outcome, confidence


def predict_yao_action(model, before_state, action_type):
    """
    爻位×行動モデルで予測
    """
    yao_pos = diagnose_yao_position_pure(before_state)
    normalized_action = normalize_action(action_type)
    key = str((yao_pos, normalized_action))

    if key in model:
        return model[key]["predicted_outcome"], model[key]["confidence"]

    # デフォルト
    return "Mixed", 0.3


def is_close(predicted, actual):
    """近い結果かどうか"""
    close_pairs = [
        ("Success", "PartialSuccess"),
        ("PartialSuccess", "Mixed"),
        ("Mixed", "Failure"),
    ]
    for pair in close_pairs:
        if predicted in pair and actual in pair:
            return True
    return False


def main():
    print("=== 純粋予測モデル訓練 ===")
    print("（pattern_typeを使用しない）")
    print()

    # データ読み込み
    train_cases = load_cases(SPLITS_DIR / "train.jsonl")
    valid_cases = load_cases(SPLITS_DIR / "validation.jsonl")

    print(f"訓練データ: {len(train_cases)}件")
    print(f"検証データ: {len(valid_cases)}件")
    print()

    # モデル1: 確率モデル
    print("=== モデル1: 確率モデル（条件付き確率）===")
    prob_model = build_probability_model(train_cases)

    train_eval = evaluate_model(prob_model, train_cases, "probability")
    valid_eval = evaluate_model(prob_model, valid_cases, "probability")

    print(f"訓練セット: 一致{train_eval['exact_rate']:.1f}% 精度{train_eval['accuracy_score']:.1f}%")
    print(f"検証セット: 一致{valid_eval['exact_rate']:.1f}% 精度{valid_eval['accuracy_score']:.1f}%")
    print()

    # モデル2: 爻位×行動モデル
    print("=== モデル2: 爻位×行動モデル ===")
    yao_model = build_yao_action_model(train_cases)

    train_eval2 = evaluate_model(yao_model, train_cases, "yao_action")
    valid_eval2 = evaluate_model(yao_model, valid_cases, "yao_action")

    print(f"訓練セット: 一致{train_eval2['exact_rate']:.1f}% 精度{train_eval2['accuracy_score']:.1f}%")
    print(f"検証セット: 一致{valid_eval2['exact_rate']:.1f}% 精度{valid_eval2['accuracy_score']:.1f}%")
    print()

    # モデル保存
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_package = {
        "created_at": datetime.now().isoformat(),
        "train_samples": len(train_cases),
        "valid_samples": len(valid_cases),
        "models": {
            "probability": {
                "type": "conditional_probability",
                "train_accuracy": train_eval["accuracy_score"],
                "valid_accuracy": valid_eval["accuracy_score"],
                "data": prob_model,
            },
            "yao_action": {
                "type": "yao_position_action",
                "train_accuracy": train_eval2["accuracy_score"],
                "valid_accuracy": valid_eval2["accuracy_score"],
                "data": yao_model,
            },
        },
        "state_to_yao": STATE_TO_YAO_PURE,
        "action_normalize": ACTION_NORMALIZE,
    }

    with open(MODELS_DIR / "prediction_model_v1.json", "w", encoding="utf-8") as f:
        json.dump(model_package, f, ensure_ascii=False, indent=2)

    print(f"モデル保存: {MODELS_DIR / 'prediction_model_v1.json'}")

    # 検証セットでの詳細分析
    print("\n=== 検証セット詳細分析（確率モデル）===")

    # 確信度別精度
    high_conf = [p for p in valid_eval["predictions"] if p["confidence"] >= 0.6]
    mid_conf = [p for p in valid_eval["predictions"] if 0.4 <= p["confidence"] < 0.6]
    low_conf = [p for p in valid_eval["predictions"] if p["confidence"] < 0.4]

    def calc_acc(preds):
        if not preds:
            return 0
        return sum(1 for p in preds if p["accuracy"] in ["exact", "close"]) / len(preds) * 100

    print(f"高確信度(≥0.6): {len(high_conf)}件 精度{calc_acc(high_conf):.1f}%")
    print(f"中確信度(0.4-0.6): {len(mid_conf)}件 精度{calc_acc(mid_conf):.1f}%")
    print(f"低確信度(<0.4): {len(low_conf)}件 精度{calc_acc(low_conf):.1f}%")


if __name__ == "__main__":
    main()
