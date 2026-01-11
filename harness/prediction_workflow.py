#!/usr/bin/env python3
"""
変化のロジック：予測ワークフロー

新規ケースの入力、予測、結果記録、精度追跡を行う。
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "data" / "models"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"

# 保存ファイル
PENDING_FILE = PREDICTIONS_DIR / "pending_predictions.jsonl"
CONFIRMED_FILE = PREDICTIONS_DIR / "confirmed_predictions.jsonl"
STATS_FILE = PREDICTIONS_DIR / "prediction_stats.json"


class PredictionWorkflow:
    """予測ワークフロー管理クラス"""

    def __init__(self):
        self.model = self._load_model()
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    def _load_model(self):
        """モデルを読み込み"""
        model_file = MODELS_DIR / "prediction_model_v1.json"
        if not model_file.exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_file}")
        with open(model_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def predict(self, before_state: str, action_type: str,
                target_name: str = "", trigger_type: str = "",
                scale: str = "", notes: str = "") -> Dict[str, Any]:
        """
        新規ケースの予測を行う

        Args:
            before_state: 現在の状態
            action_type: 取ろうとしている行動
            target_name: 対象名（オプション）
            trigger_type: きっかけ（オプション）
            scale: 規模（オプション）
            notes: メモ（オプション）

        Returns:
            予測結果
        """
        # 正規化
        action_normalize = self.model.get("action_normalize", {})
        normalized_action = action_normalize.get(action_type, action_type)

        # 爻位診断
        state_to_yao = self.model.get("state_to_yao", {})
        yao_position = state_to_yao.get(before_state, 3)

        # 確率モデルで予測
        prob_model = self.model["models"]["probability"]["data"]
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

        # 予測結果を構築
        prediction_id = str(uuid.uuid4())[:8]
        prediction = {
            "prediction_id": prediction_id,
            "created_at": datetime.now().isoformat(),
            "status": "pending",

            # 入力情報
            "input": {
                "target_name": target_name,
                "before_state": before_state,
                "action_type": action_type,
                "normalized_action": normalized_action,
                "trigger_type": trigger_type,
                "scale": scale,
                "notes": notes,
            },

            # 診断結果
            "diagnosis": {
                "yao_position": yao_position,
                "yao_stage": self._get_yao_stage(yao_position),
            },

            # 予測結果
            "prediction": {
                "outcome": best_outcome,
                "confidence": round(confidence, 3),
                "confidence_level": self._get_confidence_level(confidence),
                "distribution": {k: round(v, 3) for k, v in distribution.items()},
            },

            # 結果（後で入力）
            "actual": {
                "outcome": None,
                "confirmed_at": None,
            },
        }

        # 保存
        self._save_pending(prediction)

        return prediction

    def _get_yao_stage(self, yao_position: int) -> str:
        """爻位の段階名を取得"""
        stages = {
            1: "発芽期・始動期",
            2: "成長期・基盤確立期",
            3: "転換期・岐路",
            4: "成熟期・接近期",
            5: "全盛期・リーダー期",
            6: "衰退期・転換期・極み",
        }
        return stages.get(yao_position, "不明")

    def _get_confidence_level(self, confidence: float) -> str:
        """確信度レベルを取得"""
        if confidence >= 0.7:
            return "高"
        elif confidence >= 0.5:
            return "中"
        else:
            return "低"

    def _save_pending(self, prediction: Dict):
        """未確定予測を保存"""
        with open(PENDING_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    def list_pending(self) -> list:
        """未確定の予測一覧を取得"""
        if not PENDING_FILE.exists():
            return []

        pending = []
        with open(PENDING_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pred = json.loads(line)
                    if pred.get("status") == "pending":
                        pending.append(pred)
        return pending

    def confirm_outcome(self, prediction_id: str, actual_outcome: str) -> Dict:
        """
        予測の結果を確定する

        Args:
            prediction_id: 予測ID
            actual_outcome: 実際の結果 (Success/PartialSuccess/Mixed/Failure)

        Returns:
            更新された予測
        """
        # 未確定予測を読み込み
        pending = []
        target = None
        if PENDING_FILE.exists():
            with open(PENDING_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        pred = json.loads(line)
                        if pred.get("prediction_id") == prediction_id:
                            target = pred
                        else:
                            pending.append(pred)

        if not target:
            raise ValueError(f"予測ID {prediction_id} が見つかりません")

        # 結果を設定
        target["status"] = "confirmed"
        target["actual"]["outcome"] = actual_outcome
        target["actual"]["confirmed_at"] = datetime.now().isoformat()

        # 精度を計算
        predicted = target["prediction"]["outcome"]
        if predicted == actual_outcome:
            target["accuracy"] = "exact"
        elif self._is_close(predicted, actual_outcome):
            target["accuracy"] = "close"
        else:
            target["accuracy"] = "miss"

        # 確定済みに保存
        with open(CONFIRMED_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(target, ensure_ascii=False) + "\n")

        # 未確定を更新
        with open(PENDING_FILE, "w", encoding="utf-8") as f:
            for pred in pending:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")

        # 統計を更新
        self._update_stats(target)

        return target

    def _is_close(self, predicted: str, actual: str) -> bool:
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

    def _update_stats(self, confirmed: Dict):
        """統計を更新"""
        stats = self.get_stats()

        stats["total_confirmed"] += 1
        stats["by_accuracy"][confirmed["accuracy"]] = \
            stats["by_accuracy"].get(confirmed["accuracy"], 0) + 1

        # 確信度別
        conf_level = confirmed["prediction"]["confidence_level"]
        if conf_level not in stats["by_confidence"]:
            stats["by_confidence"][conf_level] = {"total": 0, "correct": 0}
        stats["by_confidence"][conf_level]["total"] += 1
        if confirmed["accuracy"] in ["exact", "close"]:
            stats["by_confidence"][conf_level]["correct"] += 1

        stats["last_updated"] = datetime.now().isoformat()

        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    def get_stats(self) -> Dict:
        """統計を取得"""
        if STATS_FILE.exists():
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "total_confirmed": 0,
            "by_accuracy": {},
            "by_confidence": {},
            "last_updated": None,
        }

    def print_stats(self):
        """統計を表示"""
        stats = self.get_stats()

        print("=== 予測精度統計 ===")
        print(f"確定済み予測: {stats['total_confirmed']}件")
        print()

        if stats["total_confirmed"] > 0:
            print("【精度】")
            for acc, count in stats["by_accuracy"].items():
                pct = count / stats["total_confirmed"] * 100
                print(f"  {acc}: {count}件 ({pct:.1f}%)")

            total_correct = stats["by_accuracy"].get("exact", 0) + \
                           stats["by_accuracy"].get("close", 0)
            accuracy_score = total_correct / stats["total_confirmed"] * 100
            print(f"\n精度スコア: {accuracy_score:.1f}%")
            print()

            print("【確信度別精度】")
            for level, data in stats["by_confidence"].items():
                if data["total"] > 0:
                    acc = data["correct"] / data["total"] * 100
                    print(f"  {level}確信度: {data['correct']}/{data['total']} ({acc:.1f}%)")


def interactive_predict():
    """対話的に予測を行う"""
    workflow = PredictionWorkflow()

    print("=== 変化のロジック：予測システム ===")
    print()

    # 状態の選択肢
    states = [
        "絶頂・慢心", "安定・平和", "成長痛", "停滞・閉塞",
        "混乱・カオス", "どん底・危機", "安定成長・成功", "成長・拡大"
    ]
    print("【before_state の選択肢】")
    for i, s in enumerate(states, 1):
        print(f"  {i}. {s}")
    print()

    before_state = input("現在の状態を入力: ").strip()

    # 行動の選択肢
    actions = [
        "攻める・挑戦", "守る・維持", "捨てる・撤退", "耐える・潜伏",
        "対話・融合", "刷新・破壊", "逃げる・放置", "分散・スピンオフ"
    ]
    print("\n【action_type の選択肢】")
    for i, a in enumerate(actions, 1):
        print(f"  {i}. {a}")
    print()

    action_type = input("取ろうとしている行動を入力: ").strip()

    target_name = input("対象名（オプション）: ").strip()

    # 予測実行
    result = workflow.predict(
        before_state=before_state,
        action_type=action_type,
        target_name=target_name,
    )

    print("\n" + "=" * 50)
    print("【予測結果】")
    print("=" * 50)
    print(f"予測ID: {result['prediction_id']}")
    print(f"対象: {result['input']['target_name'] or '(未指定)'}")
    print(f"状態: {result['input']['before_state']}")
    print(f"行動: {result['input']['action_type']}")
    print(f"爻位: {result['diagnosis']['yao_position']}爻 ({result['diagnosis']['yao_stage']})")
    print()
    print(f"予測結果: {result['prediction']['outcome']}")
    print(f"確信度: {result['prediction']['confidence']*100:.0f}% ({result['prediction']['confidence_level']})")
    print()
    print("確率分布:")
    for outcome, prob in sorted(result['prediction']['distribution'].items(),
                                 key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"  {outcome}: {prob*100:.0f}% {bar}")
    print()
    print(f"予測を保存しました。結果が判明したら以下のコマンドで確定してください:")
    print(f"  python harness/prediction_workflow.py confirm {result['prediction_id']} <結果>")


def confirm_outcome(prediction_id: str, outcome: str):
    """結果を確定する"""
    workflow = PredictionWorkflow()
    result = workflow.confirm_outcome(prediction_id, outcome)

    print("=== 予測結果確定 ===")
    print(f"予測ID: {result['prediction_id']}")
    print(f"予測: {result['prediction']['outcome']}")
    print(f"実際: {result['actual']['outcome']}")
    print(f"判定: {result['accuracy']}")


def show_pending():
    """未確定予測を表示"""
    workflow = PredictionWorkflow()
    pending = workflow.list_pending()

    print("=== 未確定予測一覧 ===")
    print(f"件数: {len(pending)}件")
    print()

    for pred in pending:
        print(f"ID: {pred['prediction_id']}")
        print(f"  対象: {pred['input']['target_name'] or '(未指定)'}")
        print(f"  状態: {pred['input']['before_state']} → {pred['input']['action_type']}")
        print(f"  予測: {pred['prediction']['outcome']} (確信度{pred['prediction']['confidence']*100:.0f}%)")
        print(f"  作成: {pred['created_at']}")
        print()


def show_stats():
    """統計を表示"""
    workflow = PredictionWorkflow()
    workflow.print_stats()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python prediction_workflow.py predict    # 新規予測")
        print("  python prediction_workflow.py pending    # 未確定一覧")
        print("  python prediction_workflow.py confirm <ID> <結果>  # 結果確定")
        print("  python prediction_workflow.py stats      # 統計表示")
        sys.exit(1)

    command = sys.argv[1]

    if command == "predict":
        interactive_predict()
    elif command == "pending":
        show_pending()
    elif command == "confirm":
        if len(sys.argv) < 4:
            print("使用方法: python prediction_workflow.py confirm <ID> <結果>")
            print("結果: Success / PartialSuccess / Mixed / Failure")
            sys.exit(1)
        confirm_outcome(sys.argv[2], sys.argv[3])
    elif command == "stats":
        show_stats()
    else:
        print(f"不明なコマンド: {command}")
