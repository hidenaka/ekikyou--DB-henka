#!/usr/bin/env python3
"""データベースの現状を分析するスクリプト"""

import json
from pathlib import Path
from collections import defaultdict
from schema_v3 import Case, Scale, BeforeState

def root_dir() -> Path:
    return Path(__file__).resolve().parents[1]

def cases_path() -> Path:
    return root_dir() / "data" / "raw" / "cases.jsonl"

def main():
    path = cases_path()
    if not path.exists():
        print("cases.jsonl が見つかりません")
        return

    # 統計情報
    total = 0
    scale_dist = defaultdict(int)
    before_state_dist = defaultdict(int)
    combo_dist = defaultdict(int)  # (scale, before_state) の組み合わせ
    outcome_dist = defaultdict(int)
    credibility_dist = defaultdict(int)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                case = Case(**obj)
                total += 1
                scale_dist[case.scale] += 1
                before_state_dist[case.before_state] += 1
                combo_dist[(case.scale, case.before_state)] += 1
                outcome_dist[case.outcome] += 1
                credibility_dist[case.credibility_rank] += 1
            except Exception:
                continue

    print(f"=== データベース分析結果 ===")
    print(f"\n総事例数: {total}")

    print(f"\n【スケール別分布】")
    for scale in Scale:
        count = scale_dist[scale]
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {scale.value:12s}: {count:3d} 件 ({pct:5.1f}%)")

    print(f"\n【初期状態別分布】")
    for state in BeforeState:
        count = before_state_dist[state]
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {state.value:20s}: {count:3d} 件 ({pct:5.1f}%)")

    print(f"\n【結果別分布】")
    for outcome, count in sorted(outcome_dist.items(), key=lambda x: -x[1]):
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {outcome:20s}: {count:3d} 件 ({pct:5.1f}%)")

    print(f"\n【信頼性別分布】")
    for cred, count in sorted(credibility_dist.items(), key=lambda x: -x[1]):
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {cred}: {count:3d} 件 ({pct:5.1f}%)")

    print(f"\n【組み合わせ分布】(scale × before_state)")
    print(f"※ 不足している組み合わせ（5件未満）を優先的に収集すべき")
    print()

    # 全組み合わせをチェック
    lacking = []
    for scale in Scale:
        for state in BeforeState:
            count = combo_dist.get((scale, state), 0)
            if count < 5:
                lacking.append((scale, state, count))

    # 不足している組み合わせを表示（individual優先）
    lacking.sort(key=lambda x: (x[0] != Scale.individual, x[2], x[0].value))

    print("【優先収集対象】")
    for scale, state, count in lacking[:20]:
        print(f"  {scale.value:12s} × {state.value:20s}: {count} 件")

if __name__ == "__main__":
    main()
