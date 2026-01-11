#!/usr/bin/env python3
"""
リサーチワークフロー

不足データを収集するための調査・事例収集ワークフロー
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"
CASES_FILE = BASE_DIR / "data" / "raw" / "cases.jsonl"

# 状態とキーワードの対応
STATE_KEYWORDS = {
    "絶頂・慢心": ["トップ企業", "業界リーダー", "過去最高益", "独占", "シェア1位"],
    "安定・平和": ["安定成長", "堅実経営", "優良企業", "持続的成長"],
    "成長痛": ["急成長", "組織問題", "スケーリング", "成長の壁"],
    "停滞・閉塞": ["業績横ばい", "市場飽和", "成長鈍化", "停滞"],
    "混乱・カオス": ["経営混乱", "内紛", "ガバナンス問題", "迷走"],
    "どん底・危機": ["経営危機", "倒産危機", "赤字", "リストラ"],
}

# 行動とキーワードの対応
ACTION_KEYWORDS = {
    "攻める・挑戦": ["新規事業", "海外進出", "M&A", "投資拡大"],
    "守る・維持": ["コスト削減", "効率化", "現状維持", "守りの経営"],
    "捨てる・撤退": ["事業売却", "撤退", "選択と集中", "整理"],
    "耐える・潜伏": ["低姿勢", "準備期間", "蓄積", "我慢"],
    "対話・融合": ["提携", "協業", "合併", "アライアンス"],
    "刷新・破壊": ["改革", "変革", "DX", "構造改革"],
    "逃げる・放置": ["先送り", "対応遅れ", "問題放置"],
    "分散・スピンオフ": ["分社化", "スピンオフ", "多角化"],
}

# 結果とキーワードの対応
OUTCOME_KEYWORDS = {
    "Success": ["V字回復", "成功", "復活", "達成", "成長"],
    "PartialSuccess": ["一部成功", "改善", "持ち直し"],
    "Mixed": ["まだら模様", "評価分かれる", "途上"],
    "Failure": ["失敗", "破綻", "倒産", "撤退", "衰退"],
}


class ResearchWorkflow:
    """リサーチワークフロー管理"""

    def __init__(self):
        self.tasks_file = ANALYSIS_DIR / "research_tasks.json"
        self.collected_file = ANALYSIS_DIR / "collected_cases.json"
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    def load_gap_analysis(self) -> Dict:
        """ギャップ分析結果を読み込み"""
        gap_file = ANALYSIS_DIR / "gap_analysis.json"
        if gap_file.exists():
            with open(gap_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def get_pending_tasks(self) -> List[Dict]:
        """未完了のタスクを取得"""
        if self.tasks_file.exists():
            with open(self.tasks_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [t for t in data.get("tasks", []) if t["status"] == "pending"]
        return []

    def generate_search_query(self, hexagram_name: str, keyword: str,
                              target_state: str = None,
                              target_action: str = None,
                              target_outcome: str = None) -> str:
        """検索クエリを生成"""
        query_parts = ["企業 事例"]

        # 卦のキーワード
        query_parts.append(keyword)

        # 状態
        if target_state and target_state in STATE_KEYWORDS:
            query_parts.extend(STATE_KEYWORDS[target_state][:2])

        # 行動
        if target_action and target_action in ACTION_KEYWORDS:
            query_parts.extend(ACTION_KEYWORDS[target_action][:2])

        # 結果
        if target_outcome and target_outcome in OUTCOME_KEYWORDS:
            query_parts.extend(OUTCOME_KEYWORDS[target_outcome][:1])

        return " ".join(query_parts)

    def create_research_task(self, hexagram_id: int, hexagram_name: str,
                             keyword: str, target_cases: int = 10) -> Dict:
        """リサーチタスクを作成"""
        task = {
            "task_id": str(uuid.uuid4())[:8],
            "created_at": datetime.now().isoformat(),
            "status": "pending",
            "hexagram_id": hexagram_id,
            "hexagram_name": hexagram_name,
            "keyword": keyword,
            "target_cases": target_cases,
            "collected_cases": 0,
            "search_queries": [],
            "notes": "",
        }

        # 各状態×結果の組み合わせで検索クエリを生成
        for state in ["どん底・危機", "停滞・閉塞", "成長痛"]:
            for outcome in ["Success", "Failure"]:
                query = self.generate_search_query(
                    hexagram_name, keyword,
                    target_state=state, target_outcome=outcome
                )
                task["search_queries"].append({
                    "query": query,
                    "target_state": state,
                    "target_outcome": outcome,
                })

        return task

    def save_collected_case(self, case_data: Dict) -> str:
        """収集したケースを保存"""
        collected = self._load_collected()
        case_id = str(uuid.uuid4())[:8]
        case_data["case_id"] = case_id
        case_data["collected_at"] = datetime.now().isoformat()
        case_data["status"] = "draft"  # draft -> reviewed -> approved
        collected.append(case_data)

        with open(self.collected_file, "w", encoding="utf-8") as f:
            json.dump(collected, f, ensure_ascii=False, indent=2)

        return case_id

    def _load_collected(self) -> List:
        if self.collected_file.exists():
            with open(self.collected_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def convert_to_case(self, collected_case: Dict) -> Dict:
        """収集したケースをcases.jsonl形式に変換"""
        return {
            "transition_id": f"RESEARCH_{collected_case['case_id']}",
            "target_name": collected_case.get("target_name", ""),
            "scale": collected_case.get("scale", "company"),
            "period": collected_case.get("period", ""),
            "story_summary": collected_case.get("story_summary", ""),
            "before_state": collected_case.get("before_state", ""),
            "trigger_type": collected_case.get("trigger_type", ""),
            "action_type": collected_case.get("action_type", ""),
            "after_state": collected_case.get("after_state", ""),
            "pattern_type": collected_case.get("pattern_type", ""),
            "outcome": collected_case.get("outcome", ""),
            "source_type": "research",
            "credibility_rank": collected_case.get("credibility_rank", "B"),
            "classical_before_hexagram": collected_case.get("hexagram_name", ""),
            "yao_analysis": {
                "before_hexagram_id": collected_case.get("hexagram_id"),
                "before_yao_position": collected_case.get("yao_position"),
            },
        }

    def approve_and_add_case(self, case_id: str) -> bool:
        """ケースを承認してメインDBに追加"""
        collected = self._load_collected()

        target = None
        for case in collected:
            if case.get("case_id") == case_id:
                target = case
                break

        if not target:
            return False

        # 変換
        case_data = self.convert_to_case(target)

        # メインDBに追加
        with open(CASES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(case_data, ensure_ascii=False) + "\n")

        # ステータス更新
        target["status"] = "approved"
        with open(self.collected_file, "w", encoding="utf-8") as f:
            json.dump(collected, f, ensure_ascii=False, indent=2)

        return True


def interactive_collect():
    """対話的にケースを収集"""
    workflow = ResearchWorkflow()

    print("=== ケース収集ワークフロー ===")
    print()

    # ギャップ分析から優先タスクを表示
    gap = workflow.load_gap_analysis()
    tasks = gap.get("research_tasks", [])[:5]

    if tasks:
        print("【優先収集対象】")
        for i, task in enumerate(tasks, 1):
            print(f"{i}. {task['hexagram_name']} ({task['keyword']})")
        print()

    print("収集するケースの情報を入力してください。")
    print()

    # 卦の選択
    hexagram_id = input("卦番号 (1-64): ").strip()
    if not hexagram_id:
        print("キャンセルしました")
        return

    hexagram_id = int(hexagram_id)

    # 爻位
    yao_position = input("爻位 (1-6): ").strip()
    yao_position = int(yao_position) if yao_position else None

    # ケース情報
    target_name = input("対象名（企業名など）: ").strip()
    period = input("時期（例: 2020-2023）: ").strip()
    story_summary = input("ストーリー要約（50-200字）: ").strip()

    # 状態
    print("\n【before_state の選択肢】")
    states = list(STATE_KEYWORDS.keys())
    for i, s in enumerate(states, 1):
        print(f"  {i}. {s}")
    before_state_idx = input("番号を入力: ").strip()
    before_state = states[int(before_state_idx) - 1] if before_state_idx else ""

    # 行動
    print("\n【action_type の選択肢】")
    actions = list(ACTION_KEYWORDS.keys())
    for i, a in enumerate(actions, 1):
        print(f"  {i}. {a}")
    action_idx = input("番号を入力: ").strip()
    action_type = actions[int(action_idx) - 1] if action_idx else ""

    # 結果
    print("\n【outcome の選択肢】")
    outcomes = ["Success", "PartialSuccess", "Mixed", "Failure"]
    for i, o in enumerate(outcomes, 1):
        print(f"  {i}. {o}")
    outcome_idx = input("番号を入力: ").strip()
    outcome = outcomes[int(outcome_idx) - 1] if outcome_idx else ""

    # 保存
    case_data = {
        "hexagram_id": hexagram_id,
        "yao_position": yao_position,
        "target_name": target_name,
        "period": period,
        "story_summary": story_summary,
        "before_state": before_state,
        "action_type": action_type,
        "outcome": outcome,
    }

    case_id = workflow.save_collected_case(case_data)
    print(f"\nケース保存完了: {case_id}")
    print("承認してDBに追加するには:")
    print(f"  python harness/research_workflow.py approve {case_id}")


def approve_case(case_id: str):
    """ケースを承認"""
    workflow = ResearchWorkflow()
    if workflow.approve_and_add_case(case_id):
        print(f"ケース {case_id} を承認し、DBに追加しました")
    else:
        print(f"ケース {case_id} が見つかりません")


def show_pending():
    """未承認ケースを表示"""
    workflow = ResearchWorkflow()
    collected = workflow._load_collected()
    pending = [c for c in collected if c.get("status") != "approved"]

    print("=== 未承認ケース ===")
    for case in pending:
        print(f"\nID: {case['case_id']}")
        print(f"  対象: {case.get('target_name', '')}")
        print(f"  卦: {case.get('hexagram_id', '')} / {case.get('yao_position', '')}爻")
        print(f"  状態: {case.get('before_state', '')} → {case.get('action_type', '')}")
        print(f"  結果: {case.get('outcome', '')}")


def show_search_queries():
    """検索クエリ例を表示"""
    workflow = ResearchWorkflow()
    gap = workflow.load_gap_analysis()
    tasks = gap.get("research_tasks", [])[:5]

    print("=== 検索クエリ例 ===")
    for task in tasks:
        print(f"\n【{task['hexagram_name']}】")
        research_task = workflow.create_research_task(
            task["hexagram_id"],
            task["hexagram_name"],
            task["keyword"]
        )
        for q in research_task["search_queries"][:3]:
            print(f"  {q['target_state']} → {q['target_outcome']}:")
            print(f"    \"{q['query']}\"")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python research_workflow.py collect    # ケース収集")
        print("  python research_workflow.py pending    # 未承認一覧")
        print("  python research_workflow.py approve <ID>  # 承認")
        print("  python research_workflow.py queries    # 検索クエリ例")
        sys.exit(1)

    command = sys.argv[1]

    if command == "collect":
        interactive_collect()
    elif command == "pending":
        show_pending()
    elif command == "approve":
        if len(sys.argv) < 3:
            print("IDを指定してください")
        else:
            approve_case(sys.argv[2])
    elif command == "queries":
        show_search_queries()
    else:
        print(f"不明なコマンド: {command}")
