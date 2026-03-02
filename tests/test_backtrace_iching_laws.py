#!/usr/bin/env python3
"""
BacktraceEngine L1-L3逆算ロジック × 易経変化法則 整合性検証

検証A: L1（爻レベル逆算）の正確性
検証B: ルート探索（L3）が爻変法則に基づいているか
検証C: 易経の伝統的な変化パターンの整合性
検証D: 変化の連鎖（multi-step transition）が正しいか

Usage:
    python3 tests/test_backtrace_iching_laws.py
"""

import json
import os
import random
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# パス設定
# ---------------------------------------------------------------------------
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TEST_DIR)
_SCRIPT_DIR = os.path.join(_PROJECT_ROOT, "scripts")

if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from hexagram_transformations import (
    hexagram_to_lines,
    lines_to_hexagram,
    get_hexagram_name,
    get_trigrams,
    get_zhi_gua,
    get_hu_gua,
    get_cuo_gua,
    get_zong_gua,
    get_nuclear_hexagram,
    get_complementary_hexagram,
    get_inverted_hexagram,
    HEXAGRAM_BY_ID,
)
from gap_analysis_engine import GapAnalysisEngine
from backtrace_engine import BacktraceEngine

# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

class TestResult:
    """テスト結果の収集・出力"""
    def __init__(self):
        self.results = []
        self.pass_count = 0
        self.fail_count = 0
        self.current_section = ""

    def section(self, name):
        self.current_section = name
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")

    def check(self, description, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        if condition:
            self.pass_count += 1
        else:
            self.fail_count += 1
        self.results.append({
            "section": self.current_section,
            "description": description,
            "status": status,
            "detail": detail,
        })
        mark = "  [PASS]" if condition else "  [FAIL]"
        print(f"{mark} {description}")
        if detail and not condition:
            print(f"         {detail}")

    def summary(self):
        total = self.pass_count + self.fail_count
        print(f"\n{'='*70}")
        print(f"  総合結果")
        print(f"{'='*70}")
        print(f"  PASS: {self.pass_count} / {total}")
        print(f"  FAIL: {self.fail_count} / {total}")
        if self.fail_count == 0:
            print(f"  ==> 全検証 PASS")
        else:
            print(f"  ==> {self.fail_count}件の不一致あり")
            print(f"\n  FAIL一覧:")
            for r in self.results:
                if r["status"] == "FAIL":
                    print(f"    [{r['section']}] {r['description']}")
                    if r["detail"]:
                        print(f"      -> {r['detail']}")
        print(f"{'='*70}")
        return self.fail_count == 0


def hamming_distance(hex_a, hex_b):
    """2卦間のハミング距離を計算"""
    lines_a = hexagram_to_lines(hex_a)
    lines_b = hexagram_to_lines(hex_b)
    return sum(1 for a, b in zip(lines_a, lines_b) if a != b)


def changing_lines_between(hex_a, hex_b):
    """2卦間の変爻位置を返す(1-indexed)"""
    lines_a = hexagram_to_lines(hex_a)
    lines_b = hexagram_to_lines(hex_b)
    return [i + 1 for i in range(6) if lines_a[i] != lines_b[i]]


# ---------------------------------------------------------------------------
# 検証A: L1（爻レベル逆算）の正確性
# ---------------------------------------------------------------------------

def verify_A(t: TestResult, engine: BacktraceEngine):
    t.section("検証A: L1（爻レベル逆算）の正確性")

    # A-1: ランダム20組のペアで reverse_yao を検証
    random.seed(42)  # 再現性のため固定
    pairs = []
    while len(pairs) < 20:
        a = random.randint(1, 64)
        b = random.randint(1, 64)
        if (a, b) not in pairs:
            pairs.append((a, b))

    for current, goal in pairs:
        result = engine.reverse_yao(current, goal)
        expected_hamming = hamming_distance(current, goal)
        expected_changing = changing_lines_between(current, goal)

        # ハミング距離の一致
        t.check(
            f"(#{current}->{goal}) ハミング距離: {result['hamming_distance']} == {expected_hamming}",
            result['hamming_distance'] == expected_hamming,
            f"BacktraceEngine={result['hamming_distance']}, 直接計算={expected_hamming}"
        )

        # 変爻位置の一致
        t.check(
            f"(#{current}->{goal}) changing_lines: {result['changing_lines']} == {expected_changing}",
            sorted(result['changing_lines']) == sorted(expected_changing),
            f"BacktraceEngine={sorted(result['changing_lines'])}, 直接計算={sorted(expected_changing)}"
        )

    # A-2: ハミング距離=1のペアでget_zhi_guaで確認
    print("\n  --- ハミング距離=1のペアの之卦確認 ---")
    hd1_pairs = []
    for a in range(1, 65):
        for yao in range(1, 7):
            b = get_zhi_gua(a, yao)
            if a != b:
                hd1_pairs.append((a, b, yao))

    # ランダム10組テスト
    random.shuffle(hd1_pairs)
    for current, goal, expected_yao in hd1_pairs[:10]:
        result = engine.reverse_yao(current, goal)
        t.check(
            f"HD=1: #{current}->{goal} (第{expected_yao}爻変) direct_yao_path=True",
            result['direct_yao_path'] is True,
            f"direct_yao_path={result['direct_yao_path']}"
        )
        t.check(
            f"HD=1: #{current}->{goal} direct_yao_position={expected_yao}",
            result['direct_yao_position'] == expected_yao,
            f"direct_yao_position={result['direct_yao_position']}, expected={expected_yao}"
        )
        # 実際に爻を反転して目標卦に到達するか
        actual_zhi = get_zhi_gua(current, expected_yao)
        t.check(
            f"HD=1: 第{expected_yao}爻変で #{current} -> #{actual_zhi} == #{goal}",
            actual_zhi == goal,
            f"zhi_gua(#{current}, {expected_yao}) = #{actual_zhi}, goal=#{goal}"
        )

    # A-3: ハミング距離=0（同一卦）の場合の挙動
    print("\n  --- ハミング距離=0（同一卦）の挙動 ---")
    for hex_id in [1, 2, 11, 29, 63]:
        result = engine.reverse_yao(hex_id, hex_id)
        t.check(
            f"同一卦 #{hex_id}: ハミング距離=0",
            result['hamming_distance'] == 0,
            f"hamming_distance={result['hamming_distance']}"
        )
        t.check(
            f"同一卦 #{hex_id}: changing_lines=[]",
            result['changing_lines'] == [],
            f"changing_lines={result['changing_lines']}"
        )
        t.check(
            f"同一卦 #{hex_id}: direct_yao_path=False",
            result['direct_yao_path'] is False,
            f"direct_yao_path={result['direct_yao_path']}"
        )


# ---------------------------------------------------------------------------
# 検証B: ルート探索（L3）が爻変法則に基づいているか
# ---------------------------------------------------------------------------

def verify_B(t: TestResult, engine: BacktraceEngine):
    t.section("検証B: ルート探索（L3）が爻変法則に基づいているか")

    # transition_map.json を読み込んで有効遷移テーブルを構築
    transition_map_path = os.path.join(
        _PROJECT_ROOT, "data", "hexagrams", "transition_map.json"
    )
    with open(transition_map_path, "r", encoding="utf-8") as f:
        tm_data = json.load(f)
    transitions = tm_data.get("transitions", {})

    # 有効遷移ペアを集める
    valid_transitions = set()
    for from_hex, targets in transitions.items():
        for to_hex in targets.keys():
            valid_transitions.add((from_hex, to_hex))

    scenarios = [
        (1, 2, "乾為天->坤為地（最大距離=6）"),
        (11, 12, "地天泰->天地否（錯卦）"),
        (1, 44, "乾為天->天風姤（距離=1）"),
        (63, 64, "水火既済->火水未済（錯卦）"),
        (29, 30, "坎為水->離為火（錯卦）"),
    ]

    for current, goal, desc in scenarios:
        print(f"\n  --- シナリオ: {desc} ---")
        try:
            result = engine.full_backtrace(
                current_hex=current,
                current_state="停滞・閉塞",
                goal_hex=goal,
                goal_state="安定・平和",
            )
        except Exception as e:
            t.check(f"full_backtrace({current}->{goal}): 実行可能", False, str(e))
            continue

        t.check(
            f"full_backtrace({current}->{goal}): 実行成功",
            result is not None,
        )

        # L1結果確認
        l1 = result.get("l1_yao", {})
        expected_hd = hamming_distance(current, goal)
        t.check(
            f"L1 #{current}->{goal}: ハミング距離={expected_hd}",
            l1.get("hamming_distance") == expected_hd,
            f"L1 hamming={l1.get('hamming_distance')}"
        )

        # L3ルートの確認
        recommended = result.get("recommended_routes", [])
        l3 = result.get("l3_action", {})
        all_routes = []
        for r in recommended:
            all_routes.append(("recommended", r))
        for r in l3.get("routes", []):
            all_routes.append(("l3", r))

        if not all_routes:
            # ルートがゼロでも、代替ルートがあるかチェック
            t.check(
                f"L3 #{current}->{goal}: ルートまたは代替が存在",
                len(recommended) > 0 or len(l3.get("action_recommendations", [])) > 0,
                "ルートも行動推奨もなし"
            )
            continue

        for source_label, route_info in all_routes[:3]:  # 上位3件のみチェック
            route_data = route_info.get("route", {})
            steps = route_data.get("steps", [])
            title = route_info.get("title", "?")

            if not steps:
                continue

            # 各ステップが transition_map に存在する有効遷移か確認
            for i, step in enumerate(steps):
                from_h = step.get("from_hex", "")
                to_h = step.get("to_hex", "")
                if from_h and to_h:
                    is_valid = (from_h, to_h) in valid_transitions
                    t.check(
                        f"[{source_label}] '{title}' step{i+1}: {from_h}->{to_h} はtransition_mapに存在",
                        is_valid,
                        f"from={from_h}, to={to_h}"
                    )

            # ルートの始点→終点が入力と一致するか（名前ベース）
            if steps:
                first_from = steps[0].get("from_hex", "")
                last_to = steps[-1].get("to_hex", "")
                current_name = get_hexagram_name(current)
                goal_name = get_hexagram_name(goal)

                # transition_mapのキーは "番号_八卦短名" 形式
                # 完全一致チェックではなく、卦名を含むかチェック
                from_match = (current_name in first_from or
                              str(current) in first_from or
                              first_from in current_name)
                to_match = (goal_name in last_to or
                            str(goal) in last_to or
                            last_to in goal_name)

                t.check(
                    f"[{source_label}] '{title}': 始点が#{current}({current_name})を含む",
                    from_match,
                    f"first_from='{first_from}', expected='{current_name}'({current})"
                )
                t.check(
                    f"[{source_label}] '{title}': 終点が#{goal}({goal_name})を含む",
                    to_match,
                    f"last_to='{last_to}', expected='{goal_name}'({goal})"
                )


# ---------------------------------------------------------------------------
# 検証C: 易経の伝統的な変化パターンの整合性
# ---------------------------------------------------------------------------

def verify_C(t: TestResult, engine: BacktraceEngine):
    t.section("検証C: 易経の伝統的な変化パターンの整合性")

    # C-1: 互卦（互体）: 2-5爻で構成される上下卦が正しいか
    print("\n  --- C-1: 互卦の計算精度 ---")
    for hex_id in range(1, 65):
        lines = hexagram_to_lines(hex_id)
        # 互卦: 新下卦=2,3,4爻, 新上卦=3,4,5爻
        expected_lower = lines[1:4]  # index 1,2,3 = 第2,3,4爻
        expected_upper = lines[2:5]  # index 2,3,4 = 第3,4,5爻
        expected_lines = expected_lower + expected_upper
        expected_id, expected_name = lines_to_hexagram(expected_lines)

        hu_id = get_hu_gua(hex_id)
        if hex_id <= 5 or hex_id == 63 or hex_id == 64:
            t.check(
                f"互卦 #{hex_id}({get_hexagram_name(hex_id)}) = #{hu_id}({get_hexagram_name(hu_id)})",
                hu_id == expected_id,
                f"期待=#{expected_id}, 実際=#{hu_id}"
            )

    # 全64卦の互卦チェック（サイレント、エラーのみ報告）
    hu_errors = []
    for hex_id in range(1, 65):
        lines = hexagram_to_lines(hex_id)
        expected_lower = lines[1:4]
        expected_upper = lines[2:5]
        expected_lines = expected_lower + expected_upper
        expected_id, _ = lines_to_hexagram(expected_lines)
        hu_id = get_hu_gua(hex_id)
        if hu_id != expected_id:
            hu_errors.append(f"#{hex_id}: expected={expected_id}, got={hu_id}")
    t.check(
        f"全64卦の互卦計算: エラー{len(hu_errors)}件",
        len(hu_errors) == 0,
        "; ".join(hu_errors) if hu_errors else ""
    )

    # C-2: 錯卦: 全爻反転=ハミング距離6の卦ペアを全列挙（理論上32組）
    print("\n  --- C-2: 錯卦（全爻反転）ペアの列挙 ---")
    cuo_pairs = set()
    cuo_errors = []
    for hex_id in range(1, 65):
        cuo_id = get_cuo_gua(hex_id)

        # 全爻反転のチェック
        lines = hexagram_to_lines(hex_id)
        complementary_lines = [1 - l for l in lines]
        expected_id, _ = lines_to_hexagram(complementary_lines)

        if cuo_id != expected_id:
            cuo_errors.append(f"#{hex_id}: cuo={cuo_id}, expected={expected_id}")

        # ハミング距離が6であることを確認
        hd = hamming_distance(hex_id, cuo_id)
        if hd != 6 and hex_id != cuo_id:
            cuo_errors.append(f"#{hex_id}<->#{cuo_id}: HD={hd}, expected=6")

        pair = tuple(sorted([hex_id, cuo_id]))
        cuo_pairs.add(pair)

    t.check(
        f"錯卦の計算: エラー{len(cuo_errors)}件",
        len(cuo_errors) == 0,
        "; ".join(cuo_errors[:5]) if cuo_errors else ""
    )

    # 自己錯卦（自身の錯卦が自分自身）のチェック
    self_cuo = [hex_id for hex_id in range(1, 65) if get_cuo_gua(hex_id) == hex_id]
    # 理論上、全爻反転で自分に戻る卦は存在しない（0,0,0,0,0,0の反転は1,1,1,1,1,1）
    t.check(
        f"自己錯卦: {len(self_cuo)}件（理論値=0）",
        len(self_cuo) == 0,
        f"self_cuo={self_cuo}" if self_cuo else ""
    )

    # 錯卦ペア数のチェック: 64卦 / 2 = 32ペア
    non_self_pairs = [p for p in cuo_pairs if p[0] != p[1]]
    t.check(
        f"錯卦ペア数: {len(non_self_pairs)}組（理論値=32）",
        len(non_self_pairs) == 32,
        f"実際={len(non_self_pairs)}"
    )

    # C-3: 綜卦: 上下反転
    print("\n  --- C-3: 綜卦（上下反転）ペアの確認 ---")
    zong_errors = []
    self_zong = []
    for hex_id in range(1, 65):
        zong_id = get_zong_gua(hex_id)

        # 爻の順序反転チェック
        lines = hexagram_to_lines(hex_id)
        inverted_lines = lines[::-1]
        expected_id, _ = lines_to_hexagram(inverted_lines)

        if zong_id != expected_id:
            zong_errors.append(f"#{hex_id}: zong={zong_id}, expected={expected_id}")

        # 綜卦の綜卦 = 元の卦
        double_zong = get_zong_gua(zong_id)
        if double_zong != hex_id:
            zong_errors.append(f"#{hex_id}: double_zong={double_zong}, expected={hex_id}")

        if zong_id == hex_id:
            self_zong.append(hex_id)

    t.check(
        f"綜卦の計算: エラー{len(zong_errors)}件",
        len(zong_errors) == 0,
        "; ".join(zong_errors[:5]) if zong_errors else ""
    )

    # 自己綜卦（対称卦）の確認
    # 上下対称の卦（乾、坤、頤(27)、大過(28)、坎(29)、離(30)、中孚(61)、小過(62)）
    known_symmetric = {1, 2, 27, 28, 29, 30, 61, 62}
    actual_symmetric = set(self_zong)
    t.check(
        f"自己綜卦（対称卦）: {len(self_zong)}卦",
        actual_symmetric == known_symmetric,
        f"実際={sorted(actual_symmetric)}, 期待={sorted(known_symmetric)}"
    )

    # C-4: BacktraceEngine の structural_relationship が正しく認識しているか
    print("\n  --- C-4: BacktraceEngine structural_relationship ---")

    # 錯卦ペアの確認
    cuo_test_pairs = [(1, 2), (11, 12), (29, 30), (63, 64)]
    for a, b in cuo_test_pairs:
        result = engine.reverse_yao(a, b)
        # (11,12) = 地天泰→天地否は錯卦かつ綜卦
        if a == 11 and b == 12:
            # 11の錯卦 = ?, 綜卦 = ?
            cuo_11 = get_cuo_gua(11)
            zong_11 = get_zong_gua(11)
            is_cuo = (cuo_11 == 12)
            is_zong = (zong_11 == 12)
            rel = result['structural_relationship']
            # detect_relationship の順序: identical -> cuo_gua -> zong_gua -> hu_gua -> zhi_gua -> none
            if is_cuo:
                expected_rel = "cuo_gua"
            elif is_zong:
                expected_rel = "zong_gua"
            else:
                expected_rel = "none"
            t.check(
                f"#{a}->#{b} structural_relationship: '{rel}' (期待: '{expected_rel}')",
                rel == expected_rel,
                f"cuo_gua({a})={cuo_11}, zong_gua({a})={zong_11}"
            )
        else:
            cuo_a = get_cuo_gua(a)
            if cuo_a == b:
                t.check(
                    f"#{a}->#{b} structural_relationship: 'cuo_gua'",
                    result['structural_relationship'] == "cuo_gua",
                    f"actual={result['structural_relationship']}"
                )

    # 之卦ペアの確認
    zhi_test_pairs = [(1, 44), (1, 14)]  # 乾→天風姤(第1爻変), 乾→火天大有(第5爻変)
    for a, b in zhi_test_pairs:
        result = engine.reverse_yao(a, b)
        rel = result['structural_relationship']
        t.check(
            f"#{a}->#{b} structural_relationship: starts with 'zhi_gua'",
            rel.startswith("zhi_gua"),
            f"actual='{rel}'"
        )


# ---------------------------------------------------------------------------
# 検証D: 変化の連鎖（multi-step transition）が正しいか
# ---------------------------------------------------------------------------

def verify_D(t: TestResult, engine: BacktraceEngine):
    t.section("検証D: 変化の連鎖（multi-step transition）")

    # D-1: ハミング距離=n の場合、少なくともnステップの爻変が必要
    print("\n  --- D-1: ハミング距離 と 最小ステップ数の関係 ---")

    test_pairs_d = [
        (1, 2, 6, "乾→坤（全爻反転、HD=6）"),
        (1, 44, 1, "乾→天風姤（HD=1）"),
        (11, 12, 6, "泰→否（HD=6）"),
        (63, 64, 6, "既済→未済（HD=6）"),
        (29, 30, 6, "坎→離（HD=6）"),
    ]

    for current, goal, expected_hd, desc in test_pairs_d:
        hd = hamming_distance(current, goal)
        t.check(
            f"{desc}: HD={hd} (期待={expected_hd})",
            hd == expected_hd,
            f"実際のHD={hd}"
        )

    # D-2: 乾(1)→坤(2) のステップバイステップ爻変検証
    print("\n  --- D-2: 乾(1)->坤(2) の6ステップ爻変シミュレーション ---")

    # 乾: [1,1,1,1,1,1] → 坤: [0,0,0,0,0,0]
    # 下から1爻ずつ変えていく
    current_lines = hexagram_to_lines(1)  # [1,1,1,1,1,1]
    goal_lines = hexagram_to_lines(2)     # [0,0,0,0,0,0]
    changing = changing_lines_between(1, 2)

    t.check(
        f"乾->坤 changing_lines = {changing} (全6爻)",
        len(changing) == 6 and sorted(changing) == [1, 2, 3, 4, 5, 6],
        f"changing={changing}"
    )

    # 順番に1爻ずつ変えて中間卦を確認
    current_id = 1
    step_path = [1]
    for yao_pos in [1, 2, 3, 4, 5, 6]:
        next_id = get_zhi_gua(current_id, yao_pos)
        step_path.append(next_id)
        remaining_hd = hamming_distance(next_id, 2)
        t.check(
            f"Step {yao_pos}: #{current_id} --[第{yao_pos}爻変]--> #{next_id}({get_hexagram_name(next_id)}), 残HD={remaining_hd}",
            remaining_hd == 6 - yao_pos,
            f"残HD={remaining_hd}, 期待={6 - yao_pos}"
        )
        current_id = next_id

    t.check(
        f"6ステップ後に坤(#2)に到達",
        current_id == 2,
        f"最終到達=#{current_id}({get_hexagram_name(current_id)})"
    )

    # D-3: 中間卦を経由する2ステップルートの存在確認
    print("\n  --- D-3: 距離2以上のペアで中間卦経由ルートの存在 ---")

    # 距離2のペアを作成
    hd2_pairs = []
    for a in range(1, 65):
        for b in range(a + 1, 65):
            if hamming_distance(a, b) == 2:
                hd2_pairs.append((a, b))
                if len(hd2_pairs) >= 5:
                    break
        if len(hd2_pairs) >= 5:
            break

    for a, b in hd2_pairs:
        changing = changing_lines_between(a, b)
        # 2爻変のうち1つを先に変えれば中間卦を経由できる
        found_intermediate = False
        for yao in changing:
            mid = get_zhi_gua(a, yao)
            mid_to_b = hamming_distance(mid, b)
            if mid_to_b == 1:
                found_intermediate = True
                # 中間卦からbへの最後の1爻変がget_zhi_guaで到達できるか
                remaining_cl = changing_lines_between(mid, b)
                if len(remaining_cl) == 1:
                    actual_b = get_zhi_gua(mid, remaining_cl[0])
                    t.check(
                        f"HD=2: #{a}->[#{mid}]->{b}: 中間卦経由で到達可能",
                        actual_b == b,
                        f"mid=#{mid}, actual_b=#{actual_b}"
                    )
                break

        if not found_intermediate:
            t.check(
                f"HD=2: #{a}->{b}: 中間卦が存在する",
                False,
                "1爻変の中間卦が見つからない"
            )

    # D-4: 各ステップが有効な爻変であることの数学的証明
    print("\n  --- D-4: 爻変の数学的整合性（全64卦 × 6爻 之卦テーブル） ---")

    zhi_errors = []
    zhi_table = {}  # (hex_id, yao) -> zhi_id
    for hex_id in range(1, 65):
        lines = hexagram_to_lines(hex_id)
        for yao in range(1, 7):
            # 手動計算
            flipped = lines.copy()
            flipped[yao - 1] = 1 - flipped[yao - 1]
            expected_id, _ = lines_to_hexagram(flipped)

            # API計算
            actual_id = get_zhi_gua(hex_id, yao)

            if actual_id != expected_id:
                zhi_errors.append(f"#{hex_id}[{yao}]: expected={expected_id}, got={actual_id}")

            zhi_table[(hex_id, yao)] = actual_id

    t.check(
        f"全384(64x6) 之卦計算: エラー{len(zhi_errors)}件",
        len(zhi_errors) == 0,
        "; ".join(zhi_errors[:5]) if zhi_errors else ""
    )

    # D-5: 之卦の対称性: A --yao変--> B ならば B --yao変--> A
    print("\n  --- D-5: 之卦の対称性（往復一致） ---")
    symmetry_errors = []
    for hex_id in range(1, 65):
        for yao in range(1, 7):
            zhi = get_zhi_gua(hex_id, yao)
            back = get_zhi_gua(zhi, yao)
            if back != hex_id:
                symmetry_errors.append(
                    f"#{hex_id}--[{yao}]-->#{zhi}--[{yao}]-->#{back} != #{hex_id}"
                )

    t.check(
        f"全384 之卦の対称性（往復一致）: エラー{len(symmetry_errors)}件",
        len(symmetry_errors) == 0,
        "; ".join(symmetry_errors[:5]) if symmetry_errors else ""
    )

    # D-6: 錯卦（全爻反転）が6ステップの爻変で到達可能であることの確認
    print("\n  --- D-6: 錯卦=6ステップ爻変で到達可能 ---")
    for hex_id in [1, 11, 29, 51, 63]:
        cuo = get_cuo_gua(hex_id)
        hd = hamming_distance(hex_id, cuo)
        t.check(
            f"#{hex_id}->#{cuo}(錯卦): HD={hd}=6",
            hd == 6,
            f"HD={hd}"
        )

        # 順番に全爻を反転して到達可能か
        curr = hex_id
        for yao in range(1, 7):
            # 現在の卦の第yao爻が目標と異なるかチェック
            curr_lines = hexagram_to_lines(curr)
            cuo_lines = hexagram_to_lines(cuo)
            # 最初の異なる爻を見つけて変える
            cl = changing_lines_between(curr, cuo)
            if cl:
                curr = get_zhi_gua(curr, cl[0])

        t.check(
            f"#{hex_id}->#{cuo}(錯卦): 全爻順次反転で到達",
            curr == cuo,
            f"最終到達=#{curr}, 期待=#{cuo}"
        )


# ---------------------------------------------------------------------------
# メイン実行
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  BacktraceEngine × 易経変化法則 整合性検証")
    print("  " + "=" * 66)
    print()

    t = TestResult()

    print("エンジン初期化中...")
    try:
        engine = BacktraceEngine()
        print("BacktraceEngine 初期化成功")
    except Exception as e:
        print(f"BacktraceEngine 初期化失敗: {e}")
        return 1

    verify_A(t, engine)
    verify_B(t, engine)
    verify_C(t, engine)
    verify_D(t, engine)

    all_pass = t.summary()
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
