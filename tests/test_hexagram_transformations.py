#!/usr/bin/env python3
"""
hexagram_transformations.py の包括的ユニットテスト

テスト対象:
- 基本変換（hexagram_to_lines, lines_to_hexagram）
- 三卦分解（get_trigrams）
- 之卦（get_zhi_gua）
- 互卦（get_hu_gua / get_nuclear_hexagram）
- 綜卦（get_zong_gua / get_inverted_hexagram）
- 錯卦（get_cuo_gua / get_complementary_hexagram）
- エラーハンドリング
- 対称性（involution性質）
- get_hexagram_by_trigrams の整合性
"""

import sys
import os
import pytest

# scripts/ ディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from hexagram_transformations import (
    hexagram_to_lines,
    lines_to_hexagram,
    get_trigrams,
    get_hexagram_name,
    get_hexagram_by_trigrams,
    get_zhi_gua,
    get_hu_gua,
    get_cuo_gua,
    get_zong_gua,
    get_nuclear_hexagram,
    get_inverted_hexagram,
    get_complementary_hexagram,
    get_all_transformations,
    get_lines,
    get_hexagram_number,
    HEXAGRAM_BY_ID,
    HEXAGRAM_TABLE,
    TRIGRAM_LINES,
)


# ============================================================
# 1. 基本変換テスト: hexagram_to_lines / lines_to_hexagram
# ============================================================

class TestBasicConversion:
    """hexagram_to_lines と lines_to_hexagram のテスト"""

    def test_round_trip_all_64(self):
        """全64卦のラウンドトリップ: id -> lines -> id"""
        for hex_id in range(1, 65):
            lines = hexagram_to_lines(hex_id)
            result_id, result_name = lines_to_hexagram(lines)
            assert result_id == hex_id, (
                f"卦{hex_id}のラウンドトリップが失敗: "
                f"lines={lines} -> id={result_id}"
            )

    def test_lines_length_always_6(self):
        """hexagram_to_lines は常に長さ6のリストを返す"""
        for hex_id in range(1, 65):
            lines = hexagram_to_lines(hex_id)
            assert len(lines) == 6, f"卦{hex_id}: 長さが{len(lines)}（期待値6）"

    def test_lines_values_binary(self):
        """hexagram_to_lines は 0 と 1 のみを含む"""
        for hex_id in range(1, 65):
            lines = hexagram_to_lines(hex_id)
            assert all(v in (0, 1) for v in lines), (
                f"卦{hex_id}: 不正な値を含む: {lines}"
            )

    def test_known_hexagrams(self):
        """既知の卦の爻構成を検証"""
        # 乾為天: 全陽
        assert hexagram_to_lines(1) == [1, 1, 1, 1, 1, 1]
        # 坤為地: 全陰
        assert hexagram_to_lines(2) == [0, 0, 0, 0, 0, 0]
        # 震為雷: 上=震[1,0,0], 下=震[1,0,0]
        assert hexagram_to_lines(51) == [1, 0, 0, 1, 0, 0]
        # 天地否: 上=乾[1,1,1], 下=坤[0,0,0]
        assert hexagram_to_lines(12) == [0, 0, 0, 1, 1, 1]
        # 地天泰: 上=坤[0,0,0], 下=乾[1,1,1]
        assert hexagram_to_lines(11) == [1, 1, 1, 0, 0, 0]
        # 水火既済: 上=坎[0,1,0], 下=離[1,0,1]
        assert hexagram_to_lines(63) == [1, 0, 1, 0, 1, 0]
        # 火水未済: 上=離[1,0,1], 下=坎[0,1,0]
        assert hexagram_to_lines(64) == [0, 1, 0, 1, 0, 1]

    def test_lines_to_hexagram_known(self):
        """既知の爻構成から卦を特定"""
        assert lines_to_hexagram([1, 1, 1, 1, 1, 1]) == (1, "乾為天")
        assert lines_to_hexagram([0, 0, 0, 0, 0, 0]) == (2, "坤為地")
        assert lines_to_hexagram([1, 0, 0, 1, 0, 0]) == (51, "震為雷")

    def test_all_64_unique_lines(self):
        """全64卦が異なる爻構成を持つ"""
        all_lines = set()
        for hex_id in range(1, 65):
            lines = tuple(hexagram_to_lines(hex_id))
            all_lines.add(lines)
        assert len(all_lines) == 64, "64卦が全てユニークではない"

    def test_get_lines_alias(self):
        """get_lines は hexagram_to_lines のエイリアス"""
        for hex_id in range(1, 65):
            assert get_lines(hex_id) == hexagram_to_lines(hex_id)

    def test_get_hexagram_number_alias(self):
        """get_hexagram_number は lines_to_hexagram の番号部分を返す"""
        for hex_id in range(1, 65):
            lines = hexagram_to_lines(hex_id)
            assert get_hexagram_number(lines) == hex_id


# ============================================================
# 2. 三卦分解テスト: get_trigrams
# ============================================================

class TestTrigrams:
    """get_trigrams のテスト"""

    def test_qian_wei_tian(self):
        """乾為天(1): 下卦=乾, 上卦=乾"""
        lower, upper = get_trigrams(1)
        assert lower == "乾"
        assert upper == "乾"

    def test_kun_wei_di(self):
        """坤為地(2): 下卦=坤, 上卦=坤"""
        lower, upper = get_trigrams(2)
        assert lower == "坤"
        assert upper == "坤"

    def test_tian_di_pi(self):
        """天地否(12): 下卦=坤, 上卦=乾"""
        lower, upper = get_trigrams(12)
        assert lower == "坤"
        assert upper == "乾"

    def test_di_tian_tai(self):
        """地天泰(11): 下卦=乾, 上卦=坤"""
        lower, upper = get_trigrams(11)
        assert lower == "乾"
        assert upper == "坤"

    def test_shui_huo_jiji(self):
        """水火既済(63): 下卦=離, 上卦=坎"""
        lower, upper = get_trigrams(63)
        assert lower == "離"
        assert upper == "坎"

    def test_zhen_wei_lei(self):
        """震為雷(51): 下卦=震, 上卦=震"""
        lower, upper = get_trigrams(51)
        assert lower == "震"
        assert upper == "震"

    def test_all_trigrams_valid(self):
        """全64卦の上卦・下卦が有効な八卦名"""
        valid_trigrams = set(TRIGRAM_LINES.keys())
        for hex_id in range(1, 65):
            lower, upper = get_trigrams(hex_id)
            assert lower in valid_trigrams, f"卦{hex_id}: 無効な下卦 '{lower}'"
            assert upper in valid_trigrams, f"卦{hex_id}: 無効な上卦 '{upper}'"

    def test_trigrams_consistent_with_lines(self):
        """get_trigrams の結果が hexagram_to_lines と整合する"""
        for hex_id in range(1, 65):
            lower, upper = get_trigrams(hex_id)
            lines = hexagram_to_lines(hex_id)
            lower_lines = TRIGRAM_LINES[lower]
            upper_lines = TRIGRAM_LINES[upper]
            assert lines[:3] == lower_lines, (
                f"卦{hex_id}: 下卦不一致 lines[:3]={lines[:3]} vs {lower}={lower_lines}"
            )
            assert lines[3:] == upper_lines, (
                f"卦{hex_id}: 上卦不一致 lines[3:]={lines[3:]} vs {upper}={upper_lines}"
            )


# ============================================================
# 3. get_hexagram_name テスト
# ============================================================

class TestGetHexagramName:
    """get_hexagram_name のテスト"""

    def test_known_names(self):
        assert get_hexagram_name(1) == "乾為天"
        assert get_hexagram_name(2) == "坤為地"
        assert get_hexagram_name(51) == "震為雷"
        assert get_hexagram_name(63) == "水火既済"
        assert get_hexagram_name(64) == "火水未済"

    def test_all_64_have_names(self):
        """全64卦に名前がある"""
        for hex_id in range(1, 65):
            name = get_hexagram_name(hex_id)
            assert isinstance(name, str)
            assert len(name) > 0


# ============================================================
# 4. 之卦テスト: get_zhi_gua
# ============================================================

class TestZhiGua:
    """之卦（動爻変化）のテスト"""

    def test_qian_yao1(self):
        """乾為天(1) 初爻変 → 天風姤(44)"""
        # [1,1,1,1,1,1] -> [0,1,1,1,1,1]
        # 下卦=[0,1,1]=巽, 上卦=[1,1,1]=乾 → 天風姤(44)
        assert get_zhi_gua(1, 1) == 44

    def test_qian_yao2(self):
        """乾為天(1) 二爻変 → 天火同人(13)"""
        # [1,1,1,1,1,1] -> [1,0,1,1,1,1]
        # 下卦=[1,0,1]=離, 上卦=[1,1,1]=乾 → 天火同人(13)
        assert get_zhi_gua(1, 2) == 13

    def test_qian_yao3(self):
        """乾為天(1) 三爻変 → 天沢履(10)"""
        # [1,1,1,1,1,1] -> [1,1,0,1,1,1]
        # 下卦=[1,1,0]=兌, 上卦=[1,1,1]=乾 → 天沢履(10)
        assert get_zhi_gua(1, 3) == 10

    def test_qian_yao4(self):
        """乾為天(1) 四爻変 → 風天小畜(9)"""
        # [1,1,1,1,1,1] -> [1,1,1,0,1,1]
        # 下卦=[1,1,1]=乾, 上卦=[0,1,1]=巽 → 風天小畜(9)
        assert get_zhi_gua(1, 4) == 9

    def test_qian_yao5(self):
        """乾為天(1) 五爻変 → 火天大有(14)"""
        # [1,1,1,1,1,1] -> [1,1,1,1,0,1]
        # 下卦=[1,1,1]=乾, 上卦=[1,0,1]=離 → 火天大有(14)
        assert get_zhi_gua(1, 5) == 14

    def test_qian_yao6(self):
        """乾為天(1) 上爻変 → 沢天夬(43)"""
        # [1,1,1,1,1,1] -> [1,1,1,1,1,0]
        # 下卦=[1,1,1]=乾, 上卦=[1,1,0]=兌 → 沢天夬(43)
        assert get_zhi_gua(1, 6) == 43

    def test_kun_all_yao(self):
        """坤為地(2) の各爻変"""
        # 坤為地: [0,0,0,0,0,0]
        # 初爻変: [1,0,0,0,0,0] → 下卦=[1,0,0]=震, 上卦=[0,0,0]=坤 → 地雷復(24)
        assert get_zhi_gua(2, 1) == 24
        # 二爻変: [0,1,0,0,0,0] → 下卦=[0,1,0]=坎, 上卦=[0,0,0]=坤 → 地水師(7)
        assert get_zhi_gua(2, 2) == 7
        # 三爻変: [0,0,1,0,0,0] → 下卦=[0,0,1]=艮, 上卦=[0,0,0]=坤 → 地山謙(15)
        assert get_zhi_gua(2, 3) == 15
        # 四爻変: [0,0,0,1,0,0] → 下卦=[0,0,0]=坤, 上卦=[1,0,0]=震 → 雷地豫(16)
        assert get_zhi_gua(2, 4) == 16
        # 五爻変: [0,0,0,0,1,0] → 下卦=[0,0,0]=坤, 上卦=[0,1,0]=坎 → 水地比(8)
        assert get_zhi_gua(2, 5) == 8
        # 上爻変: [0,0,0,0,0,1] → 下卦=[0,0,0]=坤, 上卦=[0,0,1]=艮 → 山地剥(23)
        assert get_zhi_gua(2, 6) == 23

    def test_zhi_gua_double_flip_returns_original(self):
        """同じ爻を2回反転すると元に戻る"""
        for hex_id in range(1, 65):
            for yao in range(1, 7):
                zhi = get_zhi_gua(hex_id, yao)
                back = get_zhi_gua(zhi, yao)
                assert back == hex_id, (
                    f"卦{hex_id}の{yao}爻変の二重反転が元に戻らない: "
                    f"{hex_id} -> {zhi} -> {back}"
                )


# ============================================================
# 5. 互卦テスト: get_hu_gua / get_nuclear_hexagram
# ============================================================

class TestHuGua:
    """互卦（Nuclear Hexagram）のテスト"""

    def test_zhen_wei_lei(self):
        """震為雷(51) → 互卦: 水山蹇(39)"""
        assert get_hu_gua(51) == 39
        nuclear_id, nuclear_name = get_nuclear_hexagram(51)
        assert nuclear_id == 39
        assert nuclear_name == "水山蹇"

    def test_qian_wei_tian(self):
        """乾為天(1) → 互卦: 乾為天(1)（全陽なので互卦も乾）"""
        assert get_hu_gua(1) == 1
        nuclear_id, nuclear_name = get_nuclear_hexagram(1)
        assert nuclear_id == 1
        assert nuclear_name == "乾為天"

    def test_kun_wei_di(self):
        """坤為地(2) → 互卦: 坤為地(2)（全陰なので互卦も坤）"""
        assert get_hu_gua(2) == 2
        nuclear_id, nuclear_name = get_nuclear_hexagram(2)
        assert nuclear_id == 2
        assert nuclear_name == "坤為地"

    def test_shui_huo_jiji(self):
        """水火既済(63) → 互卦: 火水未済(64)"""
        # 水火既済: [1,0,1,0,1,0]
        # 互卦: 下卦=2,3,4爻=[0,1,0]=坎, 上卦=3,4,5爻=[1,0,1]=離
        # → 火水未済(64)
        assert get_hu_gua(63) == 64
        nuclear_id, nuclear_name = get_nuclear_hexagram(63)
        assert nuclear_id == 64
        assert nuclear_name == "火水未済"

    def test_huo_shui_weiji(self):
        """火水未済(64) → 互卦: 水火既済(63)"""
        # 火水未済: [0,1,0,1,0,1]
        # 互卦: 下卦=2,3,4爻=[1,0,1]=離, 上卦=3,4,5爻=[0,1,0]=坎
        # → 水火既済(63)
        assert get_hu_gua(64) == 63

    def test_hu_gua_api_consistency(self):
        """get_hu_gua と get_nuclear_hexagram の整合性"""
        for hex_id in range(1, 65):
            hu = get_hu_gua(hex_id)
            nuclear_id, _ = get_nuclear_hexagram(hex_id)
            assert hu == nuclear_id, (
                f"卦{hex_id}: get_hu_gua={hu} != get_nuclear_hexagram={nuclear_id}"
            )


# ============================================================
# 6. 綜卦テスト: get_zong_gua / get_inverted_hexagram
# ============================================================

class TestZongGua:
    """綜卦（Inverted Hexagram）のテスト"""

    def test_zhen_wei_lei(self):
        """震為雷(51) → 綜卦: 艮為山(52)"""
        assert get_zong_gua(51) == 52
        inv_id, inv_name = get_inverted_hexagram(51)
        assert inv_id == 52
        assert inv_name == "艮為山"

    def test_qian_wei_tian(self):
        """乾為天(1) → 綜卦: 乾為天(1)（回文対称）"""
        assert get_zong_gua(1) == 1

    def test_kun_wei_di(self):
        """坤為地(2) → 綜卦: 坤為地(2)（回文対称）"""
        assert get_zong_gua(2) == 2

    def test_shui_huo_jiji(self):
        """水火既済(63) → 綜卦: 火水未済(64)"""
        assert get_zong_gua(63) == 64
        inv_id, inv_name = get_inverted_hexagram(63)
        assert inv_id == 64
        assert inv_name == "火水未済"

    def test_huo_shui_weiji(self):
        """火水未済(64) → 綜卦: 水火既済(63)"""
        assert get_zong_gua(64) == 63

    def test_tian_di_pi(self):
        """天地否(12) → 綜卦: 地天泰(11)"""
        # 天地否: [0,0,0,1,1,1] → 反転: [1,1,1,0,0,0] = 地天泰(11)
        assert get_zong_gua(12) == 11

    def test_di_tian_tai(self):
        """地天泰(11) → 綜卦: 天地否(12)"""
        assert get_zong_gua(11) == 12

    def test_zong_gua_api_consistency(self):
        """get_zong_gua と get_inverted_hexagram の整合性"""
        for hex_id in range(1, 65):
            zong = get_zong_gua(hex_id)
            inv_id, _ = get_inverted_hexagram(hex_id)
            assert zong == inv_id, (
                f"卦{hex_id}: get_zong_gua={zong} != get_inverted_hexagram={inv_id}"
            )


# ============================================================
# 7. 錯卦テスト: get_cuo_gua / get_complementary_hexagram
# ============================================================

class TestCuoGua:
    """錯卦（Complementary Hexagram）のテスト"""

    def test_zhen_wei_lei(self):
        """震為雷(51) → 錯卦: 巽為風(57)"""
        assert get_cuo_gua(51) == 57
        comp_id, comp_name = get_complementary_hexagram(51)
        assert comp_id == 57
        assert comp_name == "巽為風"

    def test_qian_wei_tian(self):
        """乾為天(1) → 錯卦: 坤為地(2)"""
        assert get_cuo_gua(1) == 2
        comp_id, comp_name = get_complementary_hexagram(1)
        assert comp_id == 2
        assert comp_name == "坤為地"

    def test_kun_wei_di(self):
        """坤為地(2) → 錯卦: 乾為天(1)"""
        assert get_cuo_gua(2) == 1

    def test_shui_huo_jiji(self):
        """水火既済(63) → 錯卦: 火水未済(64)"""
        # [1,0,1,0,1,0] → [0,1,0,1,0,1] = 火水未済(64)
        assert get_cuo_gua(63) == 64
        comp_id, comp_name = get_complementary_hexagram(63)
        assert comp_id == 64
        assert comp_name == "火水未済"

    def test_huo_shui_weiji(self):
        """火水未済(64) → 錯卦: 水火既済(63)"""
        assert get_cuo_gua(64) == 63

    def test_cuo_gua_api_consistency(self):
        """get_cuo_gua と get_complementary_hexagram の整合性"""
        for hex_id in range(1, 65):
            cuo = get_cuo_gua(hex_id)
            comp_id, _ = get_complementary_hexagram(hex_id)
            assert cuo == comp_id, (
                f"卦{hex_id}: get_cuo_gua={cuo} != get_complementary_hexagram={comp_id}"
            )


# ============================================================
# 8. 対称性テスト（Involution / 自己逆元性質）
# ============================================================

class TestSymmetry:
    """変換の数学的性質テスト"""

    def test_cuo_of_cuo_is_original(self):
        """錯卦の錯卦 = 元卦（全64卦）"""
        for hex_id in range(1, 65):
            cuo = get_cuo_gua(hex_id)
            cuo_of_cuo = get_cuo_gua(cuo)
            assert cuo_of_cuo == hex_id, (
                f"卦{hex_id}: 錯卦の錯卦={cuo_of_cuo}（期待値: {hex_id}）"
            )

    def test_zong_of_zong_is_original(self):
        """綜卦の綜卦 = 元卦（全64卦）"""
        for hex_id in range(1, 65):
            zong = get_zong_gua(hex_id)
            zong_of_zong = get_zong_gua(zong)
            assert zong_of_zong == hex_id, (
                f"卦{hex_id}: 綜卦の綜卦={zong_of_zong}（期待値: {hex_id}）"
            )

    def test_cuo_produces_valid_hexagram(self):
        """錯卦は常に有効な卦番号(1-64)を返す"""
        for hex_id in range(1, 65):
            cuo = get_cuo_gua(hex_id)
            assert 1 <= cuo <= 64, f"卦{hex_id}の錯卦が範囲外: {cuo}"

    def test_zong_produces_valid_hexagram(self):
        """綜卦は常に有効な卦番号(1-64)を返す"""
        for hex_id in range(1, 65):
            zong = get_zong_gua(hex_id)
            assert 1 <= zong <= 64, f"卦{hex_id}の綜卦が範囲外: {zong}"

    def test_hu_gua_produces_valid_hexagram(self):
        """互卦は常に有効な卦番号(1-64)を返す"""
        for hex_id in range(1, 65):
            hu = get_hu_gua(hex_id)
            assert 1 <= hu <= 64, f"卦{hex_id}の互卦が範囲外: {hu}"

    def test_zhi_gua_produces_valid_hexagram(self):
        """之卦は常に有効な卦番号(1-64)を返す"""
        for hex_id in range(1, 65):
            for yao in range(1, 7):
                zhi = get_zhi_gua(hex_id, yao)
                assert 1 <= zhi <= 64, (
                    f"卦{hex_id}の{yao}爻変の之卦が範囲外: {zhi}"
                )

    def test_cuo_qian_kun_pair(self):
        """乾(1)と坤(2)は互いの錯卦"""
        assert get_cuo_gua(1) == 2
        assert get_cuo_gua(2) == 1

    def test_zong_pi_tai_pair(self):
        """天地否(12)と地天泰(11)は互いの綜卦"""
        assert get_zong_gua(12) == 11
        assert get_zong_gua(11) == 12

    def test_self_symmetric_hexagrams_zong(self):
        """綜卦が自分自身になる卦が存在する（乾・坤など回文構造の卦）"""
        self_symmetric = []
        for hex_id in range(1, 65):
            if get_zong_gua(hex_id) == hex_id:
                self_symmetric.append(hex_id)
        # 乾為天(1)と坤為地(2)は確実に自己対称
        assert 1 in self_symmetric
        assert 2 in self_symmetric
        # 自己対称な卦が存在することを確認（回文構造の爻を持つ卦）
        assert len(self_symmetric) > 0

    def test_self_symmetric_hexagrams_cuo(self):
        """錯卦が自分自身になる卦が存在する"""
        self_symmetric = []
        for hex_id in range(1, 65):
            if get_cuo_gua(hex_id) == hex_id:
                self_symmetric.append(hex_id)
        # 水火既済(63)と火水未済(64)は錯卦が相手だが自己対称ではない
        # 自己対称な錯卦は爻が[1,0,1,0,1,0]の逆=[0,1,0,1,0,1]が同じ卦になる場合のみ
        # これは存在しないはず（全爻反転で同一になるには全爻が0.5=不可能）
        # しかしテスト自体は破綻しない（空リストでもOK）
        # 錯卦の自己対称は数学的に不可能（全爻反転して同じ→全爻0.5→ありえない）
        assert len(self_symmetric) == 0, (
            f"錯卦の自己対称卦が存在する: {self_symmetric}（数学的に不可能のはず）"
        )


# ============================================================
# 9. get_hexagram_by_trigrams テスト
# ============================================================

class TestGetHexagramByTrigrams:
    """get_hexagram_by_trigrams の整合性テスト"""

    def test_known_combinations(self):
        """既知の組み合わせ"""
        assert get_hexagram_by_trigrams("乾", "乾") == 1  # 乾為天
        assert get_hexagram_by_trigrams("坤", "坤") == 2  # 坤為地
        assert get_hexagram_by_trigrams("坤", "乾") == 12  # 天地否
        assert get_hexagram_by_trigrams("乾", "坤") == 11  # 地天泰
        assert get_hexagram_by_trigrams("震", "震") == 51  # 震為雷
        assert get_hexagram_by_trigrams("離", "坎") == 63  # 水火既済

    def test_round_trip_with_get_trigrams(self):
        """get_trigrams → get_hexagram_by_trigrams のラウンドトリップ（全64卦）"""
        for hex_id in range(1, 65):
            lower, upper = get_trigrams(hex_id)
            result = get_hexagram_by_trigrams(lower, upper)
            assert result == hex_id, (
                f"卦{hex_id}: get_trigrams=({lower},{upper}) → "
                f"get_hexagram_by_trigrams={result}"
            )

    def test_all_8x8_combinations_exist(self):
        """8つの三卦の全64組み合わせが有効"""
        trigram_names = list(TRIGRAM_LINES.keys())
        for lower in trigram_names:
            for upper in trigram_names:
                hex_id = get_hexagram_by_trigrams(lower, upper)
                assert 1 <= hex_id <= 64, (
                    f"({lower},{upper})の卦番号が範囲外: {hex_id}"
                )

    def test_all_64_covered(self):
        """8x8の組み合わせが64卦全てを網羅"""
        trigram_names = list(TRIGRAM_LINES.keys())
        all_ids = set()
        for lower in trigram_names:
            for upper in trigram_names:
                all_ids.add(get_hexagram_by_trigrams(lower, upper))
        assert all_ids == set(range(1, 65))


# ============================================================
# 10. get_all_transformations テスト
# ============================================================

class TestGetAllTransformations:
    """get_all_transformations の統合テスト"""

    def test_structure(self):
        """返り値の構造が正しい"""
        result = get_all_transformations(1)
        assert 'original' in result
        assert 'nuclear' in result
        assert 'inverted' in result
        assert 'complementary' in result

    def test_original_fields(self):
        """original の各フィールド"""
        result = get_all_transformations(51)
        orig = result['original']
        assert orig['id'] == 51
        assert orig['name'] == "震為雷"
        assert orig['lines'] == [1, 0, 0, 1, 0, 0]

    def test_nuclear_fields(self):
        """nuclear の各フィールド"""
        result = get_all_transformations(51)
        nuclear = result['nuclear']
        assert nuclear['id'] == 39
        assert nuclear['name'] == "水山蹇"
        assert 'description' in nuclear

    def test_inverted_fields(self):
        """inverted の各フィールド"""
        result = get_all_transformations(51)
        inverted = result['inverted']
        assert inverted['id'] == 52
        assert inverted['name'] == "艮為山"
        assert 'description' in inverted

    def test_complementary_fields(self):
        """complementary の各フィールド"""
        result = get_all_transformations(51)
        comp = result['complementary']
        assert comp['id'] == 57
        assert comp['name'] == "巽為風"
        assert 'description' in comp

    def test_lines_consistent(self):
        """各変換結果のlinesが卦番号と整合する"""
        for hex_id in range(1, 65):
            result = get_all_transformations(hex_id)
            for key in ['original', 'nuclear', 'inverted', 'complementary']:
                entry = result[key]
                expected_lines = hexagram_to_lines(entry['id'])
                assert entry['lines'] == expected_lines, (
                    f"卦{hex_id}の{key}: lines不整合"
                )


# ============================================================
# 11. エラーハンドリングテスト
# ============================================================

class TestErrorHandling:
    """無効な入力に対するエラーハンドリング"""

    # hexagram_to_lines のエラー
    def test_hexagram_to_lines_id_0(self):
        """卦番号0は無効"""
        with pytest.raises(ValueError, match="無効な卦番号"):
            hexagram_to_lines(0)

    def test_hexagram_to_lines_id_65(self):
        """卦番号65は無効"""
        with pytest.raises(ValueError, match="無効な卦番号"):
            hexagram_to_lines(65)

    def test_hexagram_to_lines_negative(self):
        """卦番号-1は無効"""
        with pytest.raises(ValueError, match="無効な卦番号"):
            hexagram_to_lines(-1)

    # lines_to_hexagram のエラー
    def test_lines_to_hexagram_too_short(self):
        """5爻は無効"""
        with pytest.raises(ValueError, match="6爻が必要"):
            lines_to_hexagram([1, 0, 0, 1, 0])

    def test_lines_to_hexagram_too_long(self):
        """7爻は無効"""
        with pytest.raises(ValueError, match="6爻が必要"):
            lines_to_hexagram([1, 0, 0, 1, 0, 0, 1])

    def test_lines_to_hexagram_empty(self):
        """空リストは無効"""
        with pytest.raises(ValueError, match="6爻が必要"):
            lines_to_hexagram([])

    def test_lines_to_hexagram_invalid_value(self):
        """0/1以外の値は無効"""
        with pytest.raises(ValueError, match="0または1のみ"):
            lines_to_hexagram([1, 0, 2, 1, 0, 0])

    def test_lines_to_hexagram_invalid_value_negative(self):
        """負の値は無効"""
        with pytest.raises(ValueError, match="0または1のみ"):
            lines_to_hexagram([1, 0, -1, 1, 0, 0])

    # get_zhi_gua のエラー
    def test_zhi_gua_invalid_hexagram(self):
        """無効な卦番号"""
        with pytest.raises(ValueError):
            get_zhi_gua(0, 1)

    def test_zhi_gua_yao_0(self):
        """爻位置0は無効"""
        with pytest.raises(ValueError, match="無効な爻位置"):
            get_zhi_gua(1, 0)

    def test_zhi_gua_yao_7(self):
        """爻位置7は無効"""
        with pytest.raises(ValueError, match="無効な爻位置"):
            get_zhi_gua(1, 7)

    def test_zhi_gua_yao_negative(self):
        """負の爻位置は無効"""
        with pytest.raises(ValueError, match="無効な爻位置"):
            get_zhi_gua(1, -1)

    # get_trigrams のエラー
    def test_get_trigrams_invalid_id(self):
        """無効な卦番号"""
        with pytest.raises(ValueError, match="無効な卦番号"):
            get_trigrams(0)

    def test_get_trigrams_invalid_id_65(self):
        with pytest.raises(ValueError, match="無効な卦番号"):
            get_trigrams(65)

    # get_hexagram_name のエラー
    def test_get_hexagram_name_invalid(self):
        """無効な卦番号"""
        with pytest.raises(ValueError, match="無効な卦番号"):
            get_hexagram_name(0)

    # get_hexagram_by_trigrams のエラー
    def test_get_hexagram_by_trigrams_invalid_trigram(self):
        """無効な三卦名"""
        with pytest.raises(ValueError, match="無効な卦の組み合わせ"):
            get_hexagram_by_trigrams("天", "地")

    # get_nuclear_hexagram のエラー
    def test_nuclear_invalid_id(self):
        with pytest.raises(ValueError):
            get_nuclear_hexagram(0)

    # get_inverted_hexagram のエラー
    def test_inverted_invalid_id(self):
        with pytest.raises(ValueError):
            get_inverted_hexagram(0)

    # get_complementary_hexagram のエラー
    def test_complementary_invalid_id(self):
        with pytest.raises(ValueError):
            get_complementary_hexagram(0)


# ============================================================
# 12. HEXAGRAM_TABLE 整合性テスト
# ============================================================

class TestDataIntegrity:
    """データテーブルの整合性"""

    def test_hexagram_table_has_64_entries(self):
        """HEXAGRAM_TABLE に64エントリある"""
        assert len(HEXAGRAM_TABLE) == 64

    def test_hexagram_by_id_has_64_entries(self):
        """HEXAGRAM_BY_ID に64エントリある"""
        assert len(HEXAGRAM_BY_ID) == 64

    def test_hexagram_ids_are_1_to_64(self):
        """卦番号が1-64を網羅"""
        ids = set(HEXAGRAM_BY_ID.keys())
        assert ids == set(range(1, 65))

    def test_trigram_lines_has_8_entries(self):
        """TRIGRAM_LINES に8つの三卦"""
        assert len(TRIGRAM_LINES) == 8

    def test_trigram_names(self):
        """8つの三卦名が正しい"""
        expected = {"乾", "兌", "離", "震", "巽", "坎", "艮", "坤"}
        assert set(TRIGRAM_LINES.keys()) == expected

    def test_all_hexagram_names_unique(self):
        """全64卦の名前がユニーク"""
        names = [get_hexagram_name(i) for i in range(1, 65)]
        assert len(set(names)) == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
