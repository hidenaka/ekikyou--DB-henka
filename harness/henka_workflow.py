#!/usr/bin/env python3
"""
変化のロジック：統合ワークフロー

すべてのコンポーネントを統合したメインハーネス
- 予測（v1/v2）
- ギャップ分析
- リサーチワークフロー
- データ管理
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# パス設定
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "harness"))
sys.path.insert(0, str(BASE_DIR / "scripts"))

from predict_v2 import (
    predict_outcome_v2,
    get_recommendation_v2,
    identify_hexagram,
    get_hexagram_profile
)
from predict_api import predict_outcome, get_recommendation
from research_workflow import ResearchWorkflow


class HenkaWorkflow:
    """変化のロジック統合ワークフロー"""

    def __init__(self):
        self.research = ResearchWorkflow()
        self.prediction_log_file = BASE_DIR / "data" / "analysis" / "prediction_log.jsonl"
        self.prediction_log_file.parent.mkdir(parents=True, exist_ok=True)

    def predict(self, before_state: str, action_type: str,
                hexagram_id: int = None, use_v2: bool = True) -> dict:
        """
        予測を実行

        Args:
            before_state: 現在の状態
            action_type: 行動タイプ
            hexagram_id: 卦番号（オプション）
            use_v2: v2エンジンを使用するか

        Returns:
            予測結果
        """
        if use_v2:
            result = predict_outcome_v2(before_state, action_type, hexagram_id)
        else:
            result = predict_outcome(before_state, action_type)

        result["engine"] = "v2" if use_v2 else "v1"
        result["timestamp"] = datetime.now().isoformat()

        # ログに記録
        self._log_prediction(before_state, action_type, hexagram_id, result)

        return result

    def recommend(self, before_state: str, hexagram_id: int = None,
                  use_v2: bool = True) -> dict:
        """推奨行動を取得"""
        if use_v2:
            return get_recommendation_v2(before_state, hexagram_id)
        else:
            return get_recommendation(before_state)

    def diagnose(self, before_state: str, context: list = None) -> dict:
        """
        状態を診断し、該当する卦を推定

        Args:
            before_state: 現在の状態
            context: 追加のコンテキストキーワード

        Returns:
            診断結果
        """
        candidates = identify_hexagram(before_state, context)

        result = {
            "before_state": before_state,
            "context": context or [],
            "candidates": [],
            "recommendation": None
        }

        for hex_id, name, score in candidates:
            profile = get_hexagram_profile(hex_id)
            result["candidates"].append({
                "hexagram_id": hex_id,
                "name": name,
                "score": score,
                "keyword": profile.get("keyword", "") if profile else "",
                "warning": profile.get("nature", {}).get("warning", "") if profile else ""
            })

        if candidates:
            best_hex = candidates[0][0]
            rec = self.recommend(before_state, best_hex)
            result["recommendation"] = {
                "hexagram": candidates[0][1],
                "best_action": rec["best_action"],
                "success_rate": rec["best_action_success_rate"],
                "avoid": [a[0] for a in rec["avoid_actions"]]
            }

        return result

    def get_gap_analysis(self) -> dict:
        """ギャップ分析結果を取得"""
        gap_file = BASE_DIR / "data" / "analysis" / "gap_analysis.json"
        if gap_file.exists():
            with open(gap_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"error": "Gap analysis not found. Run analyze_gaps.py first."}

    def get_research_tasks(self) -> list:
        """リサーチタスクを取得"""
        gap = self.get_gap_analysis()
        return gap.get("research_tasks", [])[:10]

    def create_research_task(self, hexagram_id: int) -> dict:
        """リサーチタスクを作成"""
        profile = get_hexagram_profile(hexagram_id)
        if not profile:
            return {"error": f"Hexagram {hexagram_id} not found"}

        return self.research.create_research_task(
            hexagram_id,
            profile["name"],
            profile.get("keyword", "")
        )

    def _log_prediction(self, before_state, action_type, hexagram_id, result):
        """予測をログに記録"""
        log_entry = {
            "timestamp": result["timestamp"],
            "before_state": before_state,
            "action_type": action_type,
            "hexagram_id": hexagram_id,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "engine": result["engine"]
        }
        with open(self.prediction_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def get_prediction_stats(self) -> dict:
        """予測統計を取得"""
        if not self.prediction_log_file.exists():
            return {"total": 0, "by_prediction": {}, "by_state": {}}

        stats = {
            "total": 0,
            "by_prediction": {},
            "by_state": {},
            "by_engine": {"v1": 0, "v2": 0}
        }

        with open(self.prediction_log_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                stats["total"] += 1

                pred = entry["prediction"]
                stats["by_prediction"][pred] = stats["by_prediction"].get(pred, 0) + 1

                state = entry["before_state"]
                stats["by_state"][state] = stats["by_state"].get(state, 0) + 1

                engine = entry.get("engine", "v1")
                stats["by_engine"][engine] = stats["by_engine"].get(engine, 0) + 1

        return stats


def interactive_mode():
    """対話モード"""
    workflow = HenkaWorkflow()

    print("=" * 50)
    print("変化のロジック - 対話型予測システム")
    print("=" * 50)
    print()

    states = [
        "絶頂・慢心", "安定・平和", "成長痛", "停滞・閉塞",
        "混乱・カオス", "どん底・危機"
    ]

    actions = [
        "攻める・挑戦", "守る・維持", "捨てる・撤退", "耐える・潜伏",
        "対話・融合", "刷新・破壊", "逃げる・放置", "分散・スピンオフ"
    ]

    # 状態選択
    print("【現在の状態を選択】")
    for i, s in enumerate(states, 1):
        print(f"  {i}. {s}")
    state_idx = input("番号を入力 (1-6): ").strip()
    if not state_idx or not state_idx.isdigit():
        print("キャンセルしました")
        return
    before_state = states[int(state_idx) - 1]

    # 卦の診断
    print(f"\n【{before_state}の診断】")
    diagnosis = workflow.diagnose(before_state)
    if diagnosis["candidates"]:
        print("該当しそうな卦:")
        for c in diagnosis["candidates"][:3]:
            print(f"  {c['hexagram_id']}. {c['name']} - {c['keyword']}")

    # 卦を選択するか
    use_hexagram = input("\n卦を指定しますか? (y/n): ").strip().lower()
    hexagram_id = None
    if use_hexagram == 'y':
        hex_input = input("卦番号 (1-64): ").strip()
        if hex_input.isdigit():
            hexagram_id = int(hex_input)
            profile = get_hexagram_profile(hexagram_id)
            if profile:
                print(f"選択: {profile['name']} - {profile['keyword']}")

    # 推奨行動を表示
    print("\n【推奨行動】")
    rec = workflow.recommend(before_state, hexagram_id)
    print(f"  爻位: {rec['yao_position']}爻 ({rec['yao_stage']})")
    print(f"  推奨: {rec['best_action']} (成功率{rec['best_action_success_rate']*100:.0f}%)")
    print(f"  回避: {[a[0] for a in rec['avoid_actions']]}")

    # 行動選択
    print("\n【行動を選択】")
    for i, a in enumerate(actions, 1):
        print(f"  {i}. {a}")
    action_idx = input("番号を入力 (1-8): ").strip()
    if not action_idx or not action_idx.isdigit():
        print("キャンセルしました")
        return
    action_type = actions[int(action_idx) - 1]

    # 予測実行
    print(f"\n【予測結果】")
    result = workflow.predict(before_state, action_type, hexagram_id)
    print(f"  状態: {before_state}")
    print(f"  行動: {action_type}")
    if hexagram_id:
        print(f"  卦: {result.get('hexagram_info', {}).get('name', 'N/A')}")
    print(f"  予測: {result['prediction']} (確信度{result['confidence']*100:.0f}%)")
    print(f"  爻位: {result['yao_position']}爻 ({result['yao_stage']})")

    if result.get('yao_advice'):
        print(f"\n【爻辞】")
        print(f"  {result['yao_advice']['phrase']}")
        print(f"  意味: {result['yao_advice']['meaning']}")

    if result.get('hexagram_info', {}).get('warning'):
        print(f"\n【警告】{result['hexagram_info']['warning']}")


def show_gaps():
    """ギャップ分析を表示"""
    workflow = HenkaWorkflow()
    gap = workflow.get_gap_analysis()

    if "error" in gap:
        print(gap["error"])
        return

    print("=== データギャップ分析 ===\n")

    coverage = gap.get("coverage", {})
    print(f"卦カバレッジ: {coverage.get('hexagram_coverage', 'N/A')}")
    print(f"卦×爻カバレッジ: {coverage.get('hex_yao_coverage', 'N/A')}")
    print(f"卦ID付与率: {coverage.get('hexagram_id_ratio', 'N/A')}")

    print("\n【優先リサーチタスク】")
    for task in workflow.get_research_tasks()[:5]:
        print(f"  - {task['hexagram_name']} ({task['keyword']})")


def show_stats():
    """統計を表示"""
    workflow = HenkaWorkflow()
    stats = workflow.get_prediction_stats()

    print("=== 予測統計 ===\n")
    print(f"総予測数: {stats['total']}")

    if stats['total'] > 0:
        print("\n予測結果の分布:")
        for pred, count in stats['by_prediction'].items():
            pct = count / stats['total'] * 100
            print(f"  {pred}: {count} ({pct:.1f}%)")

        print("\nエンジン別:")
        for engine, count in stats['by_engine'].items():
            print(f"  {engine}: {count}")


def main():
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python henka_workflow.py predict   # 対話型予測")
        print("  python henka_workflow.py gaps      # ギャップ分析")
        print("  python henka_workflow.py stats     # 統計表示")
        print("  python henka_workflow.py hexagram <id>  # 卦情報")
        sys.exit(1)

    command = sys.argv[1]

    if command == "predict":
        interactive_mode()
    elif command == "gaps":
        show_gaps()
    elif command == "stats":
        show_stats()
    elif command == "hexagram" and len(sys.argv) >= 3:
        hex_id = int(sys.argv[2])
        profile = get_hexagram_profile(hex_id)
        if profile:
            print(f"=== {profile['name']} ===")
            print(f"キーワード: {profile['keyword']}")
            print(f"性質: {profile['nature']}")
            print(f"ビジネス文脈: {json.dumps(profile['business_context'], ensure_ascii=False, indent=2)}")
        else:
            print(f"卦 {hex_id} が見つかりません")
    else:
        print(f"不明なコマンド: {command}")


if __name__ == "__main__":
    main()
