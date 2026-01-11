#!/usr/bin/env python3
"""
変化のロジック：予測API v2

卦と爻の特性を反映した改善版予測エンジン
"""

import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "data" / "models"
HEXAGRAMS_DIR = BASE_DIR / "data" / "hexagrams"

_model = None
_hexagram_profiles = None


def _load_model():
    global _model
    if _model is None:
        with open(MODELS_DIR / "prediction_model_v1.json", "r", encoding="utf-8") as f:
            _model = json.load(f)
    return _model


def _load_hexagram_profiles():
    global _hexagram_profiles
    if _hexagram_profiles is None:
        profiles_file = HEXAGRAMS_DIR / "hexagram_profiles.json"
        if profiles_file.exists():
            with open(profiles_file, "r", encoding="utf-8") as f:
                _hexagram_profiles = json.load(f)
        else:
            _hexagram_profiles = {"hexagrams": {}}
    return _hexagram_profiles


def get_hexagram_profile(hexagram_id: int) -> Optional[Dict]:
    """卦のプロファイルを取得"""
    profiles = _load_hexagram_profiles()
    return profiles["hexagrams"].get(str(hexagram_id))


def diagnose_yao_from_state(before_state: str) -> int:
    """状態から爻位を診断"""
    model = _load_model()
    state_to_yao = model.get("state_to_yao", {})
    return state_to_yao.get(before_state, 3)


def calculate_action_modifier(hexagram_id: int, yao_position: int, action_type: str) -> float:
    """卦×爻×行動の調整値を計算"""
    profile = get_hexagram_profile(hexagram_id)
    if not profile:
        return 0.0

    modifier = 0.0

    # 卦レベルの行動適合性
    business = profile.get("business_context", {})
    if action_type in business.get("favorable_actions", []):
        modifier += 0.5
    if action_type in business.get("unfavorable_actions", []):
        modifier -= 0.5

    # 爻レベルの調整
    yao_data = profile.get("yao", {}).get(str(yao_position), {})
    action_modifiers = yao_data.get("action_modifier", {})
    if action_type in action_modifiers:
        # 調整値を-1〜+1の範囲に正規化
        raw_mod = action_modifiers[action_type]
        modifier += raw_mod * 0.15  # スケール調整

    return modifier


def predict_outcome_v2(
    before_state: str,
    action_type: str,
    hexagram_id: Optional[int] = None,
    yao_position: Optional[int] = None
) -> Dict:
    """
    状態と行動から結果を予測（v2: 卦×爻特性反映版）

    Args:
        before_state: 現在の状態
        action_type: 取ろうとしている行動
        hexagram_id: 卦番号（1-64）。指定すると卦の特性を反映
        yao_position: 爻位（1-6）。指定しない場合は状態から推定

    Returns:
        {
            "prediction": "Success" | "PartialSuccess" | "Mixed" | "Failure",
            "confidence": 0.0-1.0,
            "confidence_level": "高" | "中" | "低",
            "yao_position": 1-6,
            "yao_stage": "発芽期・始動期" など,
            "distribution": {"Success": 0.x, ...},
            "hexagram_info": {卦の情報} (hexagram_id指定時),
            "yao_advice": "爻辞に基づくアドバイス" (hexagram_id指定時)
        }
    """
    model = _load_model()

    # 行動を正規化
    action_normalize = model.get("action_normalize", {})
    normalized_action = action_normalize.get(action_type, action_type)

    # 爻位を決定
    if yao_position is None:
        yao_position = diagnose_yao_from_state(before_state)

    yao_stages = {
        1: "発芽期・始動期",
        2: "成長期・基盤確立期",
        3: "転換期・岐路",
        4: "成熟期・接近期",
        5: "全盛期・リーダー期",
        6: "衰退期・転換期・極み",
    }

    # 基本予測（確率モデル）
    prob_model = model["models"]["probability"]["data"]
    key = str((before_state, normalized_action))

    if key in prob_model["conditional"]:
        counts = prob_model["conditional"][key]
        total = sum(counts.values())
        distribution = {k: v/total for k, v in counts.items()}
    elif before_state in prob_model["marginal_state"]:
        counts = prob_model["marginal_state"][before_state]
        total = sum(counts.values())
        distribution = {k: v/total * 0.7 for k, v in counts.items()}
    else:
        counts = prob_model["overall"]
        total = sum(counts.values())
        distribution = {k: v/total * 0.5 for k, v in counts.items()}

    # 卦×爻の調整を適用
    hexagram_info = None
    yao_advice = None

    if hexagram_id:
        modifier = calculate_action_modifier(hexagram_id, yao_position, normalized_action)

        # 調整値を確率分布に反映
        if modifier > 0:
            # ポジティブ: Success/PartialSuccessを上げる
            boost = modifier * 0.1
            if "Success" in distribution:
                distribution["Success"] = min(0.95, distribution["Success"] + boost)
            if "Failure" in distribution:
                distribution["Failure"] = max(0.02, distribution["Failure"] - boost * 0.5)
        elif modifier < 0:
            # ネガティブ: Failure/Mixedを上げる
            penalty = abs(modifier) * 0.1
            if "Failure" in distribution:
                distribution["Failure"] = min(0.95, distribution["Failure"] + penalty)
            if "Success" in distribution:
                distribution["Success"] = max(0.02, distribution["Success"] - penalty * 0.5)

        # 正規化
        total = sum(distribution.values())
        distribution = {k: v/total for k, v in distribution.items()}

        # 卦情報を追加
        profile = get_hexagram_profile(hexagram_id)
        if profile:
            hexagram_info = {
                "id": hexagram_id,
                "name": profile.get("name", ""),
                "keyword": profile.get("keyword", ""),
                "warning": profile.get("nature", {}).get("warning", ""),
                "optimal_timing": profile.get("business_context", {}).get("optimal_timing", ""),
                "risk_factors": profile.get("business_context", {}).get("risk_factors", [])
            }

            # 爻辞アドバイス
            yao_data = profile.get("yao", {}).get(str(yao_position), {})
            if yao_data:
                yao_advice = {
                    "phrase": yao_data.get("phrase", ""),
                    "meaning": yao_data.get("meaning", ""),
                    "position": yao_position,
                    "stage": yao_stages.get(yao_position, "不明")
                }

    # 最終予測
    best_outcome = max(distribution, key=distribution.get)
    confidence = distribution[best_outcome]

    # 確信度レベル
    if confidence >= 0.6:
        confidence_level = "高"
    elif confidence >= 0.4:
        confidence_level = "中"
    else:
        confidence_level = "低"

    result = {
        "prediction": best_outcome,
        "confidence": round(confidence, 3),
        "confidence_level": confidence_level,
        "yao_position": yao_position,
        "yao_stage": yao_stages.get(yao_position, "不明"),
        "distribution": {k: round(v, 3) for k, v in distribution.items()},
    }

    if hexagram_info:
        result["hexagram_info"] = hexagram_info
    if yao_advice:
        result["yao_advice"] = yao_advice

    return result


def get_recommendation_v2(
    before_state: str,
    hexagram_id: Optional[int] = None
) -> Dict:
    """
    現在の状態に対する推奨行動を取得（v2版）

    Returns:
        {
            "yao_position": 1-6,
            "yao_stage": "発芽期・始動期" など,
            "recommended_actions": [(行動, スコア, 理由), ...],
            "avoid_actions": [(行動, スコア, 理由), ...],
            "best_action": "最良の行動",
            "best_action_success_rate": 0.0-1.0,
            "hexagram_advice": "卦に基づくアドバイス" (hexagram_id指定時)
        }
    """
    yao_position = diagnose_yao_from_state(before_state)

    yao_stages = {
        1: "発芽期・始動期",
        2: "成長期・基盤確立期",
        3: "転換期・岐路",
        4: "成熟期・接近期",
        5: "全盛期・リーダー期",
        6: "衰退期・転換期・極み",
    }

    # 各行動の評価
    actions = [
        "攻める・挑戦", "守る・維持", "捨てる・撤退", "耐える・潜伏",
        "対話・融合", "刷新・破壊", "逃げる・放置", "分散・スピンオフ"
    ]

    action_evaluations = []
    for action in actions:
        result = predict_outcome_v2(before_state, action, hexagram_id, yao_position)
        success_prob = result["distribution"].get("Success", 0) + \
                       result["distribution"].get("PartialSuccess", 0) * 0.5
        failure_prob = result["distribution"].get("Failure", 0)

        # 理由を生成
        reason = ""
        if hexagram_id:
            profile = get_hexagram_profile(hexagram_id)
            if profile:
                if action in profile.get("business_context", {}).get("favorable_actions", []):
                    reason = f"{profile['name']}に適合"
                elif action in profile.get("business_context", {}).get("unfavorable_actions", []):
                    reason = f"{profile['name']}に不適合"

        action_evaluations.append({
            "action": action,
            "success_score": success_prob,
            "failure_score": failure_prob,
            "confidence": result["confidence"],
            "reason": reason
        })

    # ソート
    action_evaluations.sort(key=lambda x: -x["success_score"])

    # 推奨と回避
    recommended = [
        (e["action"], round(e["success_score"], 3), e["reason"])
        for e in action_evaluations[:3]
    ]
    avoid = [
        (e["action"], round(e["failure_score"], 3), e["reason"])
        for e in sorted(action_evaluations, key=lambda x: -x["failure_score"])[:2]
    ]

    result = {
        "yao_position": yao_position,
        "yao_stage": yao_stages.get(yao_position, "不明"),
        "recommended_actions": recommended,
        "avoid_actions": avoid,
        "best_action": action_evaluations[0]["action"],
        "best_action_success_rate": round(action_evaluations[0]["success_score"], 3),
    }

    # 卦に基づくアドバイス
    if hexagram_id:
        profile = get_hexagram_profile(hexagram_id)
        if profile:
            result["hexagram_advice"] = {
                "name": profile.get("name", ""),
                "keyword": profile.get("keyword", ""),
                "optimal_timing": profile.get("business_context", {}).get("optimal_timing", ""),
                "warning": profile.get("nature", {}).get("warning", ""),
                "risk_factors": profile.get("business_context", {}).get("risk_factors", [])
            }

    return result


def identify_hexagram(before_state: str, context_keywords: List[str] = None) -> List[Tuple[int, str, float]]:
    """
    状態とコンテキストから該当しそうな卦を推定

    Returns:
        [(hexagram_id, name, score), ...] 上位5卦
    """
    profiles = _load_hexagram_profiles()
    scores = []

    state_keyword_map = {
        "絶頂・慢心": ["繁栄", "絶頂", "傲慢", "天"],
        "安定・平和": ["安泰", "平和", "安定", "調和"],
        "成長痛": ["成長", "困難", "始動", "発展"],
        "停滞・閉塞": ["閉塞", "停滞", "困難", "蹇難"],
        "混乱・カオス": ["混乱", "対立", "変革", "衝撃"],
        "どん底・危機": ["険難", "危機", "困窮", "剥落"],
    }

    target_keywords = state_keyword_map.get(before_state, [])
    if context_keywords:
        target_keywords.extend(context_keywords)

    for hex_id, hex_data in profiles.get("hexagrams", {}).items():
        score = 0.0
        keyword = hex_data.get("keyword", "")
        nature = hex_data.get("nature", {})

        for kw in target_keywords:
            if kw in keyword:
                score += 1.0
            if kw in nature.get("timing", ""):
                score += 0.5
            if kw in nature.get("movement", ""):
                score += 0.5

        if score > 0:
            scores.append((int(hex_id), hex_data.get("name", ""), score))

    scores.sort(key=lambda x: -x[2])
    return scores[:5]


# テスト
if __name__ == "__main__":
    print("=== 予測テスト v2 ===\n")

    # 基本予測（卦なし）
    print("【基本予測（卦指定なし）】")
    result = predict_outcome_v2("停滞・閉塞", "刷新・破壊")
    print(f"状態: 停滞・閉塞 → 行動: 刷新・破壊")
    print(f"予測: {result['prediction']} (確信度{result['confidence']*100:.0f}%)")
    print(f"爻位: {result['yao_position']}爻 ({result['yao_stage']})")
    print()

    # 卦を指定した予測
    print("【卦指定あり予測】")
    result = predict_outcome_v2("どん底・危機", "耐える・潜伏", hexagram_id=29)  # 坎為水
    print(f"状態: どん底・危機 → 行動: 耐える・潜伏")
    print(f"卦: {result.get('hexagram_info', {}).get('name', 'N/A')}")
    print(f"予測: {result['prediction']} (確信度{result['confidence']*100:.0f}%)")
    if result.get('yao_advice'):
        print(f"爻辞: {result['yao_advice']['phrase']} - {result['yao_advice']['meaning']}")
    print()

    # 推奨行動
    print("【推奨行動 v2】")
    rec = get_recommendation_v2("混乱・カオス", hexagram_id=49)  # 沢火革
    print(f"状態: 混乱・カオス ({rec['yao_position']}爻)")
    if rec.get('hexagram_advice'):
        print(f"卦: {rec['hexagram_advice']['name']} - {rec['hexagram_advice']['keyword']}")
    print(f"推奨:")
    for action, score, reason in rec['recommended_actions']:
        print(f"  - {action} (成功率{score*100:.0f}%) {reason}")
    print(f"回避:")
    for action, score, reason in rec['avoid_actions']:
        print(f"  - {action} (失敗率{score*100:.0f}%) {reason}")
    print()

    # 卦の推定
    print("【卦の推定】")
    candidates = identify_hexagram("どん底・危機", ["危機", "困難"])
    print("該当しそうな卦:")
    for hex_id, name, score in candidates:
        print(f"  {hex_id}. {name} (スコア: {score})")
