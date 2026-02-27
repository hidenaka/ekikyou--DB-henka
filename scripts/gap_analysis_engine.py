#!/usr/bin/env python3
"""
GapAnalysisEngine — 二卦間の変化経路を分析するエンジン

本卦(A)と目標卦(G)の間の構造的差異を純粋な計算で解析する。
LLM呼び出しは一切行わない。

主な出力:
  - ハミング距離（爻の差異数）
  - 変爻の位置
  - 難易度スコア
  - 構造的関係（錯卦・綜卦・互卦・之卦）
  - 中間経路の提案（最大3卦）
  - 八卦（上卦・下卦）の変化
  - 相性データ（hexagram_compatibility_lookup.json参照）

使用法:
    python3 scripts/gap_analysis_engine.py
    python3 scripts/gap_analysis_engine.py --from 1 --to 2
    python3 scripts/gap_analysis_engine.py --from 11 --to 12
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple

# パス設定: 他スクリプトと同一パターン
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from hexagram_transformations import (
    hexagram_to_lines,
    lines_to_hexagram,
    get_hexagram_name,
    get_trigrams,
    get_hu_gua,
    get_cuo_gua,
    get_zong_gua,
    get_zhi_gua,
)

# デフォルトの相性データパス
_DEFAULT_COMPAT_PATH = os.path.join(
    _PROJECT_ROOT, "data", "reference", "hexagram_compatibility_lookup.json"
)


class GapAnalysisEngine:
    """二卦間のギャップ（変化経路）を分析するエンジン。

    全計算はローカルデータのみで完結し、LLM呼び出しは行わない。
    """

    def __init__(self, compat_path: Optional[str] = None):
        """初期化。相性データをロードする。

        Args:
            compat_path: hexagram_compatibility_lookup.json のパス。
                         None の場合はデフォルトパスを使用。
                         ファイルが存在しなければ相性データなしで動作する。
        """
        self._compat: Dict[str, dict] = {}
        path = compat_path or _DEFAULT_COMPAT_PATH
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                self._compat = json.load(f)

    # ------------------------------------------------------------------
    # 公開API
    # ------------------------------------------------------------------

    def analyze(self, hexagram_a: int, hexagram_g: int) -> dict:
        """本卦(A)と目標卦(G)の間の変化経路を分析する。

        Args:
            hexagram_a: 本卦の番号 (1-64)
            hexagram_g: 目標卦の番号 (1-64)

        Returns:
            分析結果の辞書。
        """
        self._validate(hexagram_a, "hexagram_a")
        self._validate(hexagram_g, "hexagram_g")

        lines_a = hexagram_to_lines(hexagram_a)
        lines_g = hexagram_to_lines(hexagram_g)

        hamming, changing = self._compute_hamming(lines_a, lines_g)

        lower_a, upper_a = get_trigrams(hexagram_a)
        lower_g, upper_g = get_trigrams(hexagram_g)

        relationship = self._detect_relationship(hexagram_a, hexagram_g)

        if hexagram_a == hexagram_g:
            intermediate = []
        else:
            intermediate = self._find_intermediate_paths(
                hexagram_a, hexagram_g, lines_a, lines_g, hamming, changing
            )

        return {
            "hexagram_a": {
                "number": hexagram_a,
                "name": get_hexagram_name(hexagram_a),
                "lines": lines_a,
            },
            "hexagram_g": {
                "number": hexagram_g,
                "name": get_hexagram_name(hexagram_g),
                "lines": lines_g,
            },
            "hamming_distance": hamming,
            "changing_lines": changing,
            "difficulty": self._difficulty_label(hamming),
            "difficulty_score": round(hamming / 6.0, 4),
            "compatibility": self._lookup_compat(hexagram_a, hexagram_g),
            "structural_relationship": relationship,
            "intermediate_paths": intermediate,
            "trigram_changes": {
                "lower": {
                    "from": lower_a,
                    "to": lower_g,
                    "changed": lower_a != lower_g,
                },
                "upper": {
                    "from": upper_a,
                    "to": upper_g,
                    "changed": upper_a != upper_g,
                },
            },
        }

    # ------------------------------------------------------------------
    # 内部メソッド
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(hexagram_number: int, param_name: str) -> None:
        """卦番号のバリデーション。"""
        if not isinstance(hexagram_number, int) or not (1 <= hexagram_number <= 64):
            raise ValueError(
                f"{param_name} は 1-64 の整数で指定してください（受け取った値: {hexagram_number}）"
            )

    @staticmethod
    def _compute_hamming(
        lines_a: List[int], lines_g: List[int]
    ) -> Tuple[int, List[int]]:
        """ハミング距離と変爻位置を計算する。

        Returns:
            (distance, changing_positions)
            changing_positions は 1-indexed の爻位置リスト。
        """
        changing: List[int] = []
        for i in range(6):
            if lines_a[i] != lines_g[i]:
                changing.append(i + 1)  # 1-indexed
        return len(changing), changing

    @staticmethod
    def _difficulty_label(hamming: int) -> str:
        """ハミング距離から難易度ラベルを返す。"""
        if hamming <= 1:
            return "easy"
        elif hamming <= 3:
            return "moderate"
        else:
            return "hard"

    def _lookup_compat(
        self, hex_a: int, hex_g: int
    ) -> Optional[dict]:
        """相性データを検索する。見つからなければ None。"""
        key = f"{hex_a}-{hex_g}"
        entry = self._compat.get(key)
        if entry is None:
            return None
        return {
            "type": entry.get("type"),
            "score": entry.get("score"),
            "summary": entry.get("summary"),
        }

    @staticmethod
    def _detect_relationship(hex_a: int, hex_g: int) -> str:
        """A と G の構造的関係を検出する。

        チェック順序:
          1. identical  — A == G
          2. cuo_gua    — 錯卦（全爻反転）
          3. zong_gua   — 綜卦（上下反転）
          4. hu_gua     — 互卦（核卦）
          5. zhi_gua    — 之卦（1爻変）
          6. none
        """
        if hex_a == hex_g:
            return "identical"
        if get_cuo_gua(hex_a) == hex_g:
            return "cuo_gua"
        if get_zong_gua(hex_a) == hex_g:
            return "zong_gua"
        if get_hu_gua(hex_a) == hex_g:
            return "hu_gua"
        for yao in range(1, 7):
            if get_zhi_gua(hex_a, yao) == hex_g:
                return f"zhi_gua(第{yao}爻変)"
        return "none"

    def _find_intermediate_paths(
        self,
        hex_a: int,
        hex_g: int,
        lines_a: List[int],
        lines_g: List[int],
        hamming: int,
        changing: List[int],
    ) -> List[dict]:
        """中間経路（ウェイポイント）を最大3つ提案する。

        優先順位:
          1. 互卦(A) — 常に候補（A,Gと異なる場合）
          2. 錯卦(A) — hamming >= 4 の場合
          3. 綜卦(A) — hamming >= 3 の場合
          4. 之卦(A, yao) — G に最も近い1爻変
        上限3つまで。
        """
        candidates: List[dict] = []
        seen: set = set()

        # ---- 互卦 ----
        hu = get_hu_gua(hex_a)
        if hu != hex_a and hu != hex_g and hu not in seen:
            candidates.append({
                "number": hu,
                "name": get_hexagram_name(hu),
                "role": "互卦（内なる構造）",
            })
            seen.add(hu)

        # ---- 錯卦（対極）: hamming >= 4 のとき ----
        if hamming >= 4:
            cuo = get_cuo_gua(hex_a)
            if cuo != hex_a and cuo != hex_g and cuo not in seen:
                candidates.append({
                    "number": cuo,
                    "name": get_hexagram_name(cuo),
                    "role": "錯卦（対極）",
                })
                seen.add(cuo)

        # ---- 綜卦（視点の転換）: hamming >= 3 のとき ----
        if hamming >= 3:
            zong = get_zong_gua(hex_a)
            if zong != hex_a and zong != hex_g and zong not in seen:
                candidates.append({
                    "number": zong,
                    "name": get_hexagram_name(zong),
                    "role": "綜卦（視点の転換）",
                })
                seen.add(zong)

        # ---- 之卦: G に最も近い1爻変を探す ----
        if len(candidates) < 3 and changing:
            best_zhi: Optional[dict] = None
            best_zhi_hamming = hamming  # 現状のhamming以上なら意味がない

            for yao in changing:
                zhi = get_zhi_gua(hex_a, yao)
                if zhi == hex_a or zhi == hex_g or zhi in seen:
                    continue
                zhi_lines = hexagram_to_lines(zhi)
                zhi_dist, _ = self._compute_hamming(zhi_lines, lines_g)
                if zhi_dist < best_zhi_hamming:
                    best_zhi_hamming = zhi_dist
                    best_zhi = {
                        "number": zhi,
                        "name": get_hexagram_name(zhi),
                        "role": f"之卦（第{yao}爻変）",
                    }

            if best_zhi is not None:
                candidates.append(best_zhi)

        return candidates[:3]


# ======================================================================
# CLI
# ======================================================================

def _print_result(result: dict) -> None:
    """分析結果を人間が読める形式で出力する。"""
    a = result["hexagram_a"]
    g = result["hexagram_g"]

    print("=" * 64)
    print(f"  ギャップ分析: {a['name']}（第{a['number']}卦）"
          f" → {g['name']}（第{g['number']}卦）")
    print("=" * 64)

    print(f"\n  爻 A: {a['lines']}")
    print(f"  爻 G: {g['lines']}")

    tri = result["trigram_changes"]
    print(f"\n  下卦: {tri['lower']['from']} → {tri['lower']['to']}"
          f"  {'【変化】' if tri['lower']['changed'] else '（同一）'}")
    print(f"  上卦: {tri['upper']['from']} → {tri['upper']['to']}"
          f"  {'【変化】' if tri['upper']['changed'] else '（同一）'}")

    print(f"\n  ハミング距離: {result['hamming_distance']} / 6")
    print(f"  変爻位置:     {result['changing_lines'] or 'なし'}")
    print(f"  難易度:       {result['difficulty']} "
          f"(スコア {result['difficulty_score']:.2f})")

    print(f"\n  構造的関係:   {result['structural_relationship']}")

    compat = result["compatibility"]
    if compat:
        print(f"\n  相性タイプ:   {compat['type']} (スコア {compat['score']})")
        print(f"  相性概要:     {compat['summary']}")
    else:
        print("\n  相性データ:   なし")

    paths = result["intermediate_paths"]
    if paths:
        print("\n  中間経路候補:")
        for i, p in enumerate(paths, 1):
            print(f"    {i}. {p['name']}（第{p['number']}卦）— {p['role']}")
    else:
        print("\n  中間経路候補: なし（同一卦）")

    print("\n" + "=" * 64)


def main():
    parser = argparse.ArgumentParser(
        description="GapAnalysisEngine — 二卦間の変化経路を分析"
    )
    parser.add_argument(
        "--from", dest="hex_a", type=int, default=None,
        help="本卦の番号 (1-64)"
    )
    parser.add_argument(
        "--to", dest="hex_g", type=int, default=None,
        help="目標卦の番号 (1-64)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="結果をJSON形式で出力"
    )
    args = parser.parse_args()

    engine = GapAnalysisEngine()

    if args.hex_a is not None and args.hex_g is not None:
        # 指定ペアの分析
        pairs = [(args.hex_a, args.hex_g)]
    else:
        # デモ: いくつかのペアを分析
        pairs = [
            (1, 2),    # 乾為天 → 坤為地（全爻反転 = 錯卦）
            (11, 12),  # 地天泰 → 天地否（上下反転 = 綜卦）
            (63, 64),  # 水火既済 → 火水未済（互卦・綜卦・錯卦の特殊関係）
            (1, 1),    # 同一卦
            (1, 14),   # 乾為天 → 火天大有（之卦: 第5爻変）
        ]

    for hex_a, hex_g in pairs:
        result = engine.analyze(hex_a, hex_g)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _print_result(result)
        print()


if __name__ == "__main__":
    main()
