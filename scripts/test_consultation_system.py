#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
易経相談システム テストスイート（20件）

カテゴリ:
- 正常系: 各状況カテゴリのテスト (8件)
- 複合状況: 複数のキーワードが混在 (5件)
- エッジケース: 空入力、極端に長い入力等 (4件)
- 目標なし: 目標が曖昧または未指定 (3件)
"""

import sys
from pathlib import Path

# 同一ディレクトリのモジュールをインポート
sys.path.insert(0, str(Path(__file__).parent))

from consultation_system import (
    parse_situation,
    parse_goal,
    judge_hexagram_deterministic,
    judge_goal_hexagram,
    load_hexagram_master,
    run_consultation
)

# テストケース定義
TEST_CASES = [
    # ==== 正常系 (8件) ====
    {
        "id": "normal_01",
        "category": "正常系",
        "situation": "上司と対立している",
        "goal": "円満に解決したい",
        "expected_hex_id": 6,  # 天水訟
        "description": "対立・争いのケース"
    },
    {
        "id": "normal_02",
        "category": "正常系",
        "situation": "新規事業を立ち上げたが軌道に乗らない",
        "goal": "成功したい",
        "expected_hex_id": 3,  # 水雷屯
        "description": "創業苦難のケース"
    },
    {
        "id": "normal_03",
        "category": "正常系",
        "situation": "停滞していて何も進まない",
        "goal": "安定したい",
        "expected_hex_id": 12,  # 天地否
        "description": "停滞・閉塞のケース"
    },
    {
        "id": "normal_04",
        "category": "正常系",
        "situation": "会社が倒産しそうで危機的状況",
        "goal": "生き残りたい",
        "expected_hex_id": 29,  # 坎為水
        "description": "危機・困難のケース"
    },
    {
        "id": "normal_05",
        "category": "正常系",
        "situation": "事業が急成長している",
        "goal": "さらに発展したい",
        "expected_hex_id": 11,  # 地天泰（成長期）
        "description": "成長期・好調のケース"
    },
    {
        "id": "normal_06",
        "category": "正常系",
        "situation": "転職を考えている",
        "goal": "新しいキャリアを築きたい",
        "expected_hex_id": 49,  # 沢火革
        "description": "変革期のケース"
    },
    {
        "id": "normal_07",
        "category": "正常系",
        "situation": "人生の岐路に立っていて迷っている",
        "goal": "正しい選択をしたい",
        "expected_hex_id": 6,  # 天水訟（岐路→混乱・不安定→6番が優先）
        "description": "迷い・選択のケース"
    },
    {
        "id": "normal_08",
        "category": "正常系",
        "situation": "チームの人間関係がうまくいかない",
        "goal": "協力できるようになりたい",
        "expected_hex_id": 45,  # 沢地萃（チーム→協力の卦）
        "description": "人間関係のケース"
    },

    # ==== 複合状況 (5件) ====
    {
        "id": "complex_01",
        "category": "複合状況",
        "situation": "事業は成長しているが競争も激しく、チーム内で対立が起きている",
        "goal": "安定して成長を続けたい",
        "expected_hex_id": None,  # 複合のため特定しない
        "description": "成長+対立の複合ケース"
    },
    {
        "id": "complex_02",
        "category": "複合状況",
        "situation": "転職を考えているが、今の仕事も停滞気味で迷っている",
        "goal": "キャリアを前進させたい",
        "expected_hex_id": None,
        "description": "変化+停滞+迷いの複合ケース"
    },
    {
        "id": "complex_03",
        "category": "複合状況",
        "situation": "スタートアップで資金繰りに苦しみつつ、競合との争いもある",
        "goal": "軌道に乗せたい",
        "expected_hex_id": None,
        "description": "創業苦難+対立の複合ケース"
    },
    {
        "id": "complex_04",
        "category": "複合状況",
        "situation": "上司との関係が悪化し、仕事全体が停滞している",
        "goal": "状況を改善したい",
        "expected_hex_id": None,
        "description": "対立+停滞の複合ケース"
    },
    {
        "id": "complex_05",
        "category": "複合状況",
        "situation": "成長していたが最近危機的な状況になってきた",
        "goal": "再び成長軌道に戻したい",
        "expected_hex_id": None,
        "description": "成長→危機の複合ケース"
    },

    # ==== エッジケース (4件) ====
    {
        "id": "edge_01",
        "category": "エッジケース",
        "situation": "",
        "goal": "成功したい",
        "expected_hex_id": 6,  # 空入力→デフォルト「混乱・不安定」→天水訟
        "description": "空の状況入力"
    },
    {
        "id": "edge_02",
        "category": "エッジケース",
        "situation": "あああああ",
        "goal": "いいいいい",
        "expected_hex_id": 6,  # マッチなし→デフォルト「混乱・不安定」→天水訟
        "description": "意味のない入力"
    },
    {
        "id": "edge_03",
        "category": "エッジケース",
        "situation": "私は今、非常に困っています。会社の業績が悪化し、上司からのプレッシャーも強く、家庭の問題も抱えています。毎日がストレスで、何をどうすればいいのかわかりません。このままでは心身ともに限界を迎えそうです。助けてください。",
        "goal": "すべてが良くなってほしい",
        "expected_hex_id": None,  # 長文
        "description": "極端に長い入力"
    },
    {
        "id": "edge_04",
        "category": "エッジケース",
        "situation": "ABC123!@#危機的状況",
        "goal": "安定",
        "expected_hex_id": 29,  # 危機キーワードがマッチ
        "description": "特殊文字を含む入力"
    },

    # ==== 目標なし/曖昧 (3件) ====
    {
        "id": "nogoal_01",
        "category": "目標なし",
        "situation": "今の状況に不満がある",
        "goal": "",
        "expected_hex_id": None,
        "description": "空の目標"
    },
    {
        "id": "nogoal_02",
        "category": "目標なし",
        "situation": "仕事で悩んでいる",
        "goal": "なんとかしたい",
        "expected_hex_id": None,
        "description": "曖昧な目標"
    },
    {
        "id": "nogoal_03",
        "category": "目標なし",
        "situation": "何か変化が必要だと感じている",
        "goal": "とにかく現状を変えたい",
        "expected_hex_id": 49,  # 変化キーワード
        "description": "変化を求める曖昧な目標"
    },
]


def run_test(test_case: dict, hexagram_master: dict) -> dict:
    """単一テストケースを実行"""
    result = {
        "id": test_case["id"],
        "category": test_case["category"],
        "description": test_case["description"],
        "passed": False,
        "error": None,
        "actual_hex_id": None,
        "actual_hex_name": None,
    }

    try:
        # パース
        parsed_situation = parse_situation(test_case["situation"])

        # 卦判定
        judgment = judge_hexagram_deterministic(parsed_situation, hexagram_master)
        result["actual_hex_id"] = judgment.hexagram_id
        result["actual_hex_name"] = judgment.hexagram_name

        # 期待値との比較
        if test_case["expected_hex_id"] is None:
            # 複合/エッジケースは実行できれば成功
            result["passed"] = True
        else:
            result["passed"] = (judgment.hexagram_id == test_case["expected_hex_id"])

        # 完全実行テスト（run_consultation）
        if test_case["situation"]:  # 空でない場合のみ
            goal = test_case["goal"] or "より良い状態になりたい"
            output = run_consultation(test_case["situation"], goal)
            result["full_execution"] = True
        else:
            result["full_execution"] = False

    except Exception as e:
        result["error"] = str(e)
        result["passed"] = False
        result["full_execution"] = False

    return result


def main():
    """テストスイートを実行"""
    print("=" * 70)
    print("易経相談システム テストスイート")
    print("=" * 70)

    # マスターデータ読み込み
    hexagram_master = load_hexagram_master()

    # 結果集計
    results = []
    categories = {}

    for test_case in TEST_CASES:
        result = run_test(test_case, hexagram_master)
        results.append(result)

        # カテゴリ別集計
        cat = test_case["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0}
        categories[cat]["total"] += 1
        if result["passed"]:
            categories[cat]["passed"] += 1

    # 結果出力
    print(f"\n{'─' * 70}")
    print("【テスト結果一覧】")
    print(f"{'─' * 70}")

    for result in results:
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"\n{result['id']}: {status}")
        print(f"  カテゴリ: {result['category']}")
        print(f"  説明: {result['description']}")
        if result["actual_hex_id"]:
            print(f"  判定結果: {result['actual_hex_name']}（{result['actual_hex_id']}番）")
        if result["error"]:
            print(f"  エラー: {result['error']}")

    # サマリー
    print(f"\n{'=' * 70}")
    print("【サマリー】")
    print(f"{'=' * 70}")

    total_passed = sum(1 for r in results if r["passed"])
    total_tests = len(results)

    print(f"\n総合: {total_passed}/{total_tests} ({total_passed/total_tests*100:.0f}%)")

    for cat, stats in categories.items():
        print(f"  {cat}: {stats['passed']}/{stats['total']}")

    print(f"\n{'=' * 70}")

    # 終了コード
    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    exit(main())
