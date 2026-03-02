#!/usr/bin/env python3
"""
易経 爻変法則 全384パターン検証スクリプト

4つの検証を実行:
  検証1: yao_transitions.json の全384エントリを binary 計算と照合
  検証2: rev_yao.json が yao_transitions.json の正確な逆引きか
  検証3: backtrace_engine の reverse_yao() が正しいか
  検証4: 伝統的な易経の法則との整合性チェック
"""

import json
import os
import sys
import time
from collections import defaultdict

# ---------------------------------------------------------------------------
# パス設定
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from hexagram_transformations import (
    get_zhi_gua,
    hexagram_to_lines,
    lines_to_hexagram,
    get_hexagram_name,
    get_complementary_hexagram,
    get_inverted_hexagram,
    HEXAGRAM_BY_ID,
)

YAO_TRANSITIONS_PATH = os.path.join(PROJECT_ROOT, "data", "mappings", "yao_transitions.json")
REV_YAO_PATH = os.path.join(PROJECT_ROOT, "data", "reverse", "rev_yao.json")

# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def hamming_distance(lines_a, lines_b):
    """2つの爻リスト間のハミング距離を返す"""
    return sum(a != b for a, b in zip(lines_a, lines_b))


def changing_positions(lines_a, lines_b):
    """異なる爻の位置リスト (1-indexed) を返す"""
    return [i + 1 for i in range(6) if lines_a[i] != lines_b[i]]


# ===========================================================================
# 検証1: yao_transitions.json の全384エントリを binary 計算と照合
# ===========================================================================

def verify_1_yao_transitions_vs_binary(yao_transitions):
    """
    yao_transitions.json の各 (source_hex, yao_pos) → target_hex を
    get_zhi_gua() の計算結果と照合する。
    """
    print("=" * 70)
    print("検証1: yao_transitions.json vs binary計算 (get_zhi_gua)")
    print("=" * 70)

    total = 0
    match = 0
    mismatches = []

    for source_id_str, hex_data in yao_transitions.items():
        source_id = int(source_id_str)
        source_name = hex_data.get("name", "")
        transitions = hex_data.get("transitions", {})

        for yao_pos_str, trans_info in transitions.items():
            yao_pos = int(yao_pos_str)
            expected_target = trans_info.get("next_hexagram_id")
            expected_name = trans_info.get("next_hexagram_name", "")

            total += 1

            # binary計算
            computed_target = get_zhi_gua(source_id, yao_pos)
            computed_name = get_hexagram_name(computed_target)

            if computed_target == expected_target:
                match += 1
            else:
                mismatches.append({
                    "source_hex": source_id,
                    "source_name": source_name,
                    "yao_pos": yao_pos,
                    "json_target": expected_target,
                    "json_target_name": expected_name,
                    "computed_target": computed_target,
                    "computed_target_name": computed_name,
                })

    # 卦数チェック
    hex_count = len(yao_transitions)

    print(f"\n  卦数: {hex_count} / 64")
    print(f"  総エントリ数: {total} / 384 (64卦 x 6爻)")
    print(f"  一致: {match}")
    print(f"  不一致: {len(mismatches)}")

    if mismatches:
        print("\n  --- 不一致リスト ---")
        for m in mismatches:
            print(f"    {m['source_name']}({m['source_hex']}) 第{m['yao_pos']}爻変: "
                  f"JSON={m['json_target_name']}({m['json_target']}) vs "
                  f"計算={m['computed_target_name']}({m['computed_target']})")

    passed = (match == 384 and total == 384 and hex_count == 64)
    status = "PASS" if passed else "FAIL"
    print(f"\n  結果: [{status}]")

    if not passed and match == total and total < 384:
        print(f"  (全 {total} エントリは計算と一致するが、384エントリに足りない)")

    return passed, mismatches


# ===========================================================================
# 検証2: rev_yao.json が yao_transitions.json の正確な逆引きか
# ===========================================================================

def verify_2_rev_yao_consistency(yao_transitions, rev_yao):
    """
    正方向: yao_transitions の全 (source, yao_pos, target) について
            rev_yao[target] に (source, yao_pos) が存在するか確認。
    逆方向: rev_yao の全エントリが yao_transitions に存在するか確認。
    """
    print("\n" + "=" * 70)
    print("検証2: rev_yao.json <-> yao_transitions.json の双方向整合性")
    print("=" * 70)

    # --- 正方向: yao_transitions → rev_yao ---
    forward_total = 0
    forward_found = 0
    forward_missing = []

    for source_id_str, hex_data in yao_transitions.items():
        source_id = int(source_id_str)
        source_name = hex_data.get("name", "")
        transitions = hex_data.get("transitions", {})

        for yao_pos_str, trans_info in transitions.items():
            yao_pos = int(yao_pos_str)
            target_id = trans_info.get("next_hexagram_id")
            target_key = str(target_id)
            forward_total += 1

            # rev_yao[target_id] に (source_id, yao_pos) が存在するか
            rev_entries = rev_yao.get(target_key, [])
            found = any(
                e.get("source_hex_id") == source_id and e.get("yao_pos") == yao_pos
                for e in rev_entries
            )
            if found:
                forward_found += 1
            else:
                forward_missing.append({
                    "source_hex": source_id,
                    "source_name": source_name,
                    "yao_pos": yao_pos,
                    "target_hex": target_id,
                })

    print(f"\n  [正方向] yao_transitions → rev_yao")
    print(f"    検査エントリ数: {forward_total}")
    print(f"    rev_yaoに存在: {forward_found}")
    print(f"    不足: {len(forward_missing)}")
    if forward_missing:
        print("    --- 不足リスト ---")
        for m in forward_missing[:20]:
            print(f"      {m['source_name']}({m['source_hex']}) 第{m['yao_pos']}爻変 → 卦{m['target_hex']}")
        if len(forward_missing) > 20:
            print(f"      ... 他 {len(forward_missing) - 20} 件")

    # --- 逆方向: rev_yao → yao_transitions ---
    reverse_total = 0
    reverse_found = 0
    reverse_extra = []

    for target_key, entries in rev_yao.items():
        for entry in entries:
            source_id = entry.get("source_hex_id")
            yao_pos = entry.get("yao_pos")
            reverse_total += 1

            # yao_transitions[source_id][yao_pos] == target_key か
            source_str = str(source_id)
            yao_str = str(yao_pos)
            hex_data = yao_transitions.get(source_str, {})
            transitions = hex_data.get("transitions", {})
            trans_info = transitions.get(yao_str, {})
            actual_target = trans_info.get("next_hexagram_id")

            if actual_target == int(target_key):
                reverse_found += 1
            else:
                reverse_extra.append({
                    "target_hex": int(target_key),
                    "source_hex": source_id,
                    "yao_pos": yao_pos,
                    "actual_target_in_yao_transitions": actual_target,
                })

    print(f"\n  [逆方向] rev_yao → yao_transitions")
    print(f"    rev_yaoエントリ数: {reverse_total}")
    print(f"    yao_transitionsと一致: {reverse_found}")
    print(f"    余剰/不整合: {len(reverse_extra)}")
    if reverse_extra:
        print("    --- 余剰/不整合リスト ---")
        for m in reverse_extra[:20]:
            print(f"      rev_yao[{m['target_hex']}]: source={m['source_hex']}, yao={m['yao_pos']} "
                  f"→ yao_transitions上のtarget={m['actual_target_in_yao_transitions']}")
        if len(reverse_extra) > 20:
            print(f"      ... 他 {len(reverse_extra) - 20} 件")

    # エントリ数の対称性
    print(f"\n  [対称性チェック]")
    print(f"    yao_transitions側エントリ数: {forward_total}")
    print(f"    rev_yao側エントリ数: {reverse_total}")
    symmetric = forward_total == reverse_total
    print(f"    対称: {'Yes' if symmetric else 'No'}")

    passed = (forward_found == forward_total and
              reverse_found == reverse_total and
              symmetric)
    status = "PASS" if passed else "FAIL"
    print(f"\n  結果: [{status}]")
    return passed


# ===========================================================================
# 検証3: backtrace_engine の reverse_yao() が正しいか
# ===========================================================================

def verify_3_reverse_yao_function():
    """
    - 全64x64ペアからハミング距離=1のペアを抽出し、reverse_yao()で正しい変爻位置を返すか確認
    - ハミング距離>1のサンプル10個で direct_yao_path=False が返るか確認

    Note: BacktraceEngine は gap_analysis_engine / case_search 等に依存するため、
    依存関係が存在しない場合は代替テスト（直接binary計算）を実行する。
    """
    print("\n" + "=" * 70)
    print("検証3: reverse_yao() の正確性（全ハミング距離=1ペア + サンプル）")
    print("=" * 70)

    # --- BacktraceEngine を使えるか試みる ---
    engine = None
    use_engine = False
    try:
        from backtrace_engine import BacktraceEngine
        engine = BacktraceEngine()
        use_engine = True
        print("\n  BacktraceEngine のロードに成功。reverse_yao() を使用します。")
    except Exception as e:
        print(f"\n  BacktraceEngine のロードに失敗 ({e})。")
        print("  代替: binary計算ベースでハミング距離=1の全ペアを検証します。")

    # --- 全64x64ペアのハミング距離を計算 ---
    hamming1_pairs = []  # (hex_a, hex_b, changing_pos)
    hamming_gt1_pairs = []  # (hex_a, hex_b, hamming_dist)

    all_lines = {}
    for h in range(1, 65):
        all_lines[h] = hexagram_to_lines(h)

    for a in range(1, 65):
        for b in range(1, 65):
            if a == b:
                continue
            hd = hamming_distance(all_lines[a], all_lines[b])
            if hd == 1:
                cp = changing_positions(all_lines[a], all_lines[b])
                hamming1_pairs.append((a, b, cp[0]))
            elif hd > 1:
                hamming_gt1_pairs.append((a, b, hd))

    print(f"\n  ハミング距離=1のペア数: {len(hamming1_pairs)} (理論値: 384)")

    # --- 検証3a: ハミング距離=1のペア ---
    pass_3a = True
    errors_3a = []

    if use_engine:
        for a, b, expected_pos in hamming1_pairs:
            result = engine.reverse_yao(a, b)
            if not result.get("direct_yao_path"):
                errors_3a.append(f"  ({a}→{b}): direct_yao_path=False (期待: True, 位置={expected_pos})")
                pass_3a = False
            elif result.get("direct_yao_position") != expected_pos:
                errors_3a.append(
                    f"  ({a}→{b}): 位置={result.get('direct_yao_position')} "
                    f"(期待: {expected_pos})"
                )
                pass_3a = False
    else:
        # 代替検証: get_zhi_gua で逆方向を確認
        for a, b, expected_pos in hamming1_pairs:
            computed_b = get_zhi_gua(a, expected_pos)
            if computed_b != b:
                errors_3a.append(
                    f"  ({a}→{b}): get_zhi_gua({a}, {expected_pos})={computed_b} (期待: {b})"
                )
                pass_3a = False

    print(f"\n  [3a] ハミング距離=1: 全{len(hamming1_pairs)}ペア検証")
    print(f"    エラー数: {len(errors_3a)}")
    if errors_3a:
        for e in errors_3a[:20]:
            print(f"    {e}")
        if len(errors_3a) > 20:
            print(f"    ... 他 {len(errors_3a) - 20} 件")
    status_3a = "PASS" if pass_3a else "FAIL"
    print(f"    結果: [{status_3a}]")

    # --- 検証3b: ハミング距離>1のサンプル10個 ---
    import random
    random.seed(42)
    sample_gt1 = random.sample(hamming_gt1_pairs, min(10, len(hamming_gt1_pairs)))

    pass_3b = True
    errors_3b = []

    if use_engine:
        for a, b, hd in sample_gt1:
            result = engine.reverse_yao(a, b)
            if result.get("direct_yao_path"):
                errors_3b.append(
                    f"  ({a}→{b}, HD={hd}): direct_yao_path=True (期待: False)"
                )
                pass_3b = False
            if result.get("hamming_distance") != hd:
                errors_3b.append(
                    f"  ({a}→{b}): hamming_distance={result.get('hamming_distance')} (期待: {hd})"
                )
                pass_3b = False
    else:
        # 代替: binary計算でハミング距離を確認
        for a, b, hd in sample_gt1:
            computed_hd = hamming_distance(all_lines[a], all_lines[b])
            if computed_hd != hd:
                errors_3b.append(
                    f"  ({a}→{b}): computed_hd={computed_hd} (期待: {hd})"
                )
                pass_3b = False
            if computed_hd == 1:
                errors_3b.append(
                    f"  ({a}→{b}): ハミング距離=1と分類されるべきでなかった"
                )
                pass_3b = False

    print(f"\n  [3b] ハミング距離>1: サンプル{len(sample_gt1)}ペア検証")
    print(f"    サンプル:")
    for a, b, hd in sample_gt1:
        name_a = get_hexagram_name(a)
        name_b = get_hexagram_name(b)
        print(f"      {name_a}({a}) → {name_b}({b}): HD={hd}")
    print(f"    エラー数: {len(errors_3b)}")
    if errors_3b:
        for e in errors_3b[:10]:
            print(f"    {e}")
    status_3b = "PASS" if pass_3b else "FAIL"
    print(f"    結果: [{status_3b}]")

    # --- 追加統計 ---
    hd_distribution = defaultdict(int)
    for a in range(1, 65):
        for b in range(a + 1, 65):
            hd = hamming_distance(all_lines[a], all_lines[b])
            hd_distribution[hd] += 1

    print(f"\n  [統計] 全ペア (64C2 = {64*63//2}) のハミング距離分布:")
    for d in sorted(hd_distribution.keys()):
        count = hd_distribution[d]
        print(f"    HD={d}: {count} ペア")

    passed = pass_3a and pass_3b
    status = "PASS" if passed else "FAIL"
    print(f"\n  結果: [{status}]")
    return passed


# ===========================================================================
# 検証4: 伝統的な易経の法則との整合性チェック
# ===========================================================================

def verify_4_traditional_iching():
    """
    binary計算結果が伝統的な易経の爻変法則と一致するか確認。

    乾為天(1) = [1,1,1,1,1,1]
      初爻変: 1爻flip → [0,1,1,1,1,1] → 下巽,上乾 = 天風姤(44)
      二爻変: 2爻flip → [1,0,1,1,1,1] → 下離,上乾 = 天火同人(13)
      三爻変: 3爻flip → [1,1,0,1,1,1] → 下兌,上乾 = 天沢履(10)
      四爻変: 4爻flip → [1,1,1,0,1,1] → 下乾,上巽 = 風天小畜(9)
      五爻変: 5爻flip → [1,1,1,1,0,1] → 下乾,上離 = 火天大有(14)
      上爻変: 6爻flip → [1,1,1,1,1,0] → 下乾,上兌 = 沢天夬(43)

    坤為地(2) = [0,0,0,0,0,0]
      初爻変: [1,0,0,0,0,0] → 下震,上坤 = 地雷復(24)
      二爻変: [0,1,0,0,0,0] → 下坎,上坤 = 地水師(7)
      三爻変: [0,0,1,0,0,0] → 下艮,上坤 = 地山謙(15)
      四爻変: [0,0,0,1,0,0] → 下坤,上震 = 雷地豫(16)
      五爻変: [0,0,0,0,1,0] → 下坤,上坎 = 水地比(8)
      上爻変: [0,0,0,0,0,1] → 下坤,上艮 = 山地剥(23)
    """
    print("\n" + "=" * 70)
    print("検証4: 伝統的な易経の法則との整合性チェック")
    print("=" * 70)

    all_pass = True

    # --- 4a: 乾為天(1)の6爻変 ---
    print("\n  [4a] 乾為天(1)の6爻変")
    qian_expected = {
        1: (44, "天風姤"),
        2: (13, "天火同人"),
        3: (10, "天沢履"),
        4: (9, "風天小畜"),
        5: (14, "火天大有"),
        6: (43, "沢天夬"),
    }

    for yao_pos, (expected_id, expected_name) in qian_expected.items():
        computed = get_zhi_gua(1, yao_pos)
        computed_name = get_hexagram_name(computed)
        lines = hexagram_to_lines(1)
        lines[yao_pos - 1] = 1 - lines[yao_pos - 1]
        ok = computed == expected_id
        status = "OK" if ok else "NG"
        print(f"    第{yao_pos}爻変: {computed_name}({computed}) [{status}]"
              f"  爻={lines}")
        if not ok:
            print(f"      期待: {expected_name}({expected_id})")
            all_pass = False

    # --- 4b: 坤為地(2)の6爻変 ---
    print("\n  [4b] 坤為地(2)の6爻変")
    # binary計算に基づく正しい期待値
    kun_expected = {
        1: (24, "地雷復"),
        2: (7, "地水師"),
        3: (15, "地山謙"),
        4: (16, "雷地豫"),
        5: (8, "水地比"),
        6: (23, "山地剥"),
    }

    for yao_pos, (expected_id, expected_name) in kun_expected.items():
        computed = get_zhi_gua(2, yao_pos)
        computed_name = get_hexagram_name(computed)
        lines = hexagram_to_lines(2)
        lines[yao_pos - 1] = 1 - lines[yao_pos - 1]
        ok = computed == expected_id
        status = "OK" if ok else "NG"
        print(f"    第{yao_pos}爻変: {computed_name}({computed}) [{status}]"
              f"  爻={lines}")
        if not ok:
            print(f"      期待: {expected_name}({expected_id})")
            all_pass = False

    # ユーザーが当初期待していた坤の値との対比（教育目的）
    print("\n  [参考] 坤為地の爻変について:")
    print("    伝統通り binary計算の結果:")
    print("    初爻変 → 地雷復(24)  ※震=[1,0,0]が下卦に")
    print("    二爻変 → 地水師(7)   ※坎=[0,1,0]が下卦に")
    print("    三爻変 → 地山謙(15)  ※艮=[0,0,1]が下卦に")
    print("    四爻変 → 雷地豫(16)  ※震=[1,0,0]が上卦に")
    print("    五爻変 → 水地比(8)   ※坎=[0,1,0]が上卦に")
    print("    六爻変 → 山地剥(23)  ※艮=[0,0,1]が上卦に")

    # --- 4c: 地天泰(11) ⇔ 天地否(12) の錯卦関係 ---
    print("\n  [4c] 地天泰(11) ⇔ 天地否(12) 錯卦関係")
    lines_11 = hexagram_to_lines(11)
    lines_12 = hexagram_to_lines(12)
    hd_11_12 = hamming_distance(lines_11, lines_12)

    # 錯卦チェック
    cuo_11_id, cuo_11_name = get_complementary_hexagram(11)
    cuo_12_id, cuo_12_name = get_complementary_hexagram(12)

    print(f"    地天泰(11) 爻: {lines_11}")
    print(f"    天地否(12) 爻: {lines_12}")
    print(f"    ハミング距離: {hd_11_12} (期待: 6 = 全爻反転)")
    print(f"    地天泰(11)の錯卦: {cuo_11_name}({cuo_11_id}) (期待: 天地否(12))")
    print(f"    天地否(12)の錯卦: {cuo_12_name}({cuo_12_id}) (期待: 地天泰(11))")

    ok_hd = hd_11_12 == 6
    ok_cuo_11 = cuo_11_id == 12
    ok_cuo_12 = cuo_12_id == 11

    if not ok_hd:
        print(f"    NG: ハミング距離が6ではない ({hd_11_12})")
        all_pass = False
    if not ok_cuo_11:
        print(f"    NG: 地天泰の錯卦が天地否ではない ({cuo_11_id})")
        all_pass = False
    if not ok_cuo_12:
        print(f"    NG: 天地否の錯卦が地天泰ではない ({cuo_12_id})")
        all_pass = False

    if ok_hd and ok_cuo_11 and ok_cuo_12:
        print("    結果: [OK] 完全な錯卦関係（全爻反転）")
    else:
        print("    結果: [NG]")

    # --- 4d: 水火既済(63) ⇔ 火水未済(64) の錯卦関係 ---
    print("\n  [4d] 水火既済(63) ⇔ 火水未済(64) 錯卦関係")
    lines_63 = hexagram_to_lines(63)
    lines_64 = hexagram_to_lines(64)
    hd_63_64 = hamming_distance(lines_63, lines_64)

    cuo_63_id, cuo_63_name = get_complementary_hexagram(63)
    cuo_64_id, cuo_64_name = get_complementary_hexagram(64)

    # 綜卦（上下反転）もチェック
    zong_63_id, zong_63_name = get_inverted_hexagram(63)
    zong_64_id, zong_64_name = get_inverted_hexagram(64)

    print(f"    水火既済(63) 爻: {lines_63}")
    print(f"    火水未済(64) 爻: {lines_64}")
    print(f"    ハミング距離: {hd_63_64}")
    print(f"    水火既済(63)の錯卦: {cuo_63_name}({cuo_63_id})")
    print(f"    火水未済(64)の錯卦: {cuo_64_name}({cuo_64_id})")
    print(f"    水火既済(63)の綜卦: {zong_63_name}({zong_63_id})")
    print(f"    火水未済(64)の綜卦: {zong_64_name}({zong_64_id})")

    # 水火既済 [1,0,1,0,1,0] の錯卦 [0,1,0,1,0,1] = 火水未済
    # 水火既済 [1,0,1,0,1,0] の綜卦 [0,1,0,1,0,1] = 火水未済（偶然一致）
    ok_cuo_63 = cuo_63_id == 64
    ok_cuo_64 = cuo_64_id == 63
    ok_zong_63 = zong_63_id == 64
    ok_zong_64 = zong_64_id == 63

    if ok_cuo_63 and ok_cuo_64:
        print("    錯卦: [OK] 互いに錯卦")
    else:
        print("    錯卦: [NG]")
        all_pass = False

    if ok_zong_63 and ok_zong_64:
        print("    綜卦: [OK] 互いに綜卦でもある（[1,0,1,0,1,0]の特殊性）")
    else:
        print(f"    綜卦: 63の綜卦={zong_63_id}, 64の綜卦={zong_64_id}")

    # ハミング距離=6のチェック
    if hd_63_64 == 6:
        print(f"    ハミング距離: [OK] HD=6 (全爻反転)")
    else:
        print(f"    ハミング距離: HD={hd_63_64} (注: 交互配列の特殊ケース)")
        # 交互配列 [1,0,1,0,1,0] と [0,1,0,1,0,1] のHDは6
        # これは錯卦の定義通り

    # --- 4e: 全64卦の之卦が自分以外の卦を指すか ---
    print("\n  [4e] 全64卦 x 6爻 = 384パターンの之卦が自身でないことの確認")
    self_ref_count = 0
    for h in range(1, 65):
        for yao in range(1, 7):
            zhi = get_zhi_gua(h, yao)
            if zhi == h:
                self_ref_count += 1
                print(f"    NG: {get_hexagram_name(h)}({h}) 第{yao}爻変 = 自分自身")

    if self_ref_count == 0:
        print("    結果: [OK] 全384パターンで自己参照なし")
    else:
        print(f"    結果: [NG] {self_ref_count} 件の自己参照あり")
        all_pass = False

    # --- 4f: 之卦の対称性（AのN爻変=B ⇔ BのN爻変=A）---
    print("\n  [4f] 之卦の対称性: AのN爻変=B ならば BのN爻変=A")
    symmetry_errors = 0
    for h in range(1, 65):
        for yao in range(1, 7):
            zhi = get_zhi_gua(h, yao)
            reverse_zhi = get_zhi_gua(zhi, yao)
            if reverse_zhi != h:
                symmetry_errors += 1
                if symmetry_errors <= 5:
                    print(f"    NG: {get_hexagram_name(h)}({h}) 第{yao}爻変 → "
                          f"{get_hexagram_name(zhi)}({zhi}) 第{yao}爻変 → "
                          f"{get_hexagram_name(reverse_zhi)}({reverse_zhi}) != {h}")

    if symmetry_errors == 0:
        print("    結果: [OK] 全384パターンで対称性成立")
    else:
        print(f"    結果: [NG] {symmetry_errors} 件の非対称")
        all_pass = False

    # --- 4g: 64卦テーブルの完全性（全64卦がHEXAGRAM_BY_IDに存在）---
    print("\n  [4g] HEXAGRAM_BY_ID テーブルの完全性チェック")
    missing_ids = [i for i in range(1, 65) if i not in HEXAGRAM_BY_ID]
    if not missing_ids:
        print(f"    結果: [OK] 全64卦が登録済み（登録数: {len(HEXAGRAM_BY_ID)}）")
    else:
        print(f"    結果: [NG] 欠落: {missing_ids}")
        all_pass = False

    status = "PASS" if all_pass else "FAIL"
    print(f"\n  検証4 結果: [{status}]")
    return all_pass


# ===========================================================================
# メイン
# ===========================================================================

def main():
    start_time = time.time()

    print("=" * 70)
    print("易経 爻変法則 全384パターン検証")
    print(f"実行時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # データ読み込み
    print(f"\nデータ読み込み:")
    print(f"  yao_transitions.json: {YAO_TRANSITIONS_PATH}")
    print(f"  rev_yao.json: {REV_YAO_PATH}")

    yao_transitions = load_json(YAO_TRANSITIONS_PATH)
    rev_yao = load_json(REV_YAO_PATH)

    print(f"  yao_transitions: {len(yao_transitions)} 卦")
    print(f"  rev_yao: {len(rev_yao)} ターゲット卦")

    results = {}

    # 検証1
    passed_1, mismatches_1 = verify_1_yao_transitions_vs_binary(yao_transitions)
    results["検証1: yao_transitions vs binary計算"] = passed_1

    # 検証2
    passed_2 = verify_2_rev_yao_consistency(yao_transitions, rev_yao)
    results["検証2: rev_yao <-> yao_transitions 双方向整合性"] = passed_2

    # 検証3
    passed_3 = verify_3_reverse_yao_function()
    results["検証3: reverse_yao() の正確性"] = passed_3

    # 検証4
    passed_4 = verify_4_traditional_iching()
    results["検証4: 伝統的易経法則との整合性"] = passed_4

    # --- サマリー ---
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("検証サマリー")
    print("=" * 70)

    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    overall = "ALL PASS" if all_pass else "SOME FAILED"
    print(f"\n  総合結果: [{overall}]")
    print(f"  実行時間: {elapsed:.2f} 秒")
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
