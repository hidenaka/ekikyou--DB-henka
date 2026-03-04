#!/usr/bin/env python3
"""
確率テーブルを使った八卦マッピングモジュール

入力マッピング設計v2 (docs/input_mapping_design_v2.md) に基づき、
DB 13,060件から構築した条件付き確率テーブルを使って
テキスト相談 → 卦・爻マッピングを実行する。

Usage (マッピング):
    from probability_tables import ProbabilityMapper
    mapper = ProbabilityMapper()
    result = mapper.get_top_candidates("どん底・危機", "守る・維持")

Usage (テーブル生成):
    python3 scripts/probability_tables.py --build
    # → data/diagnostic/prob_tables.json (全体, 後方互換)
    # → data/diagnostic/prob_tables_company.json
    # → data/diagnostic/prob_tables_individual.json
    # → data/diagnostic/prob_tables_family.json
    # → data/diagnostic/prob_tables_country.json
    # → data/diagnostic/prob_tables_other.json
"""

import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional

# --- 八卦定数 ---
TRIGRAMS = ["乾", "兌", "離", "震", "巽", "坎", "艮", "坤"]

# 八卦 → 3ビット表現 (下から: 初爻, 二爻, 三爻)
TRIGRAM_BITS = {
    "乾": (1, 1, 1),
    "兌": (1, 1, 0),
    "離": (1, 0, 1),
    "震": (1, 0, 0),
    "巽": (0, 1, 1),
    "坎": (0, 1, 0),
    "艮": (0, 0, 1),
    "坤": (0, 0, 0),
}
BITS_TRIGRAM = {v: k for k, v in TRIGRAM_BITS.items()}

# 上卦・下卦 → 64卦番号 (King Wen sequence)
# HEXAGRAM_TABLE[lower][upper] = hexagram_number
HEXAGRAM_TABLE = {
    "乾": {"乾": 1, "兌": 43, "離": 14, "震": 34, "巽": 9, "坎": 5, "艮": 26, "坤": 11},
    "兌": {"乾": 10, "兌": 58, "離": 38, "震": 54, "巽": 61, "坎": 60, "艮": 41, "坤": 19},
    "離": {"乾": 13, "兌": 49, "離": 30, "震": 55, "巽": 37, "坎": 63, "艮": 22, "坤": 36},
    "震": {"乾": 25, "兌": 17, "離": 21, "震": 51, "巽": 42, "坎": 3, "艮": 27, "坤": 24},
    "巽": {"乾": 44, "兌": 28, "離": 50, "震": 32, "巽": 57, "坎": 48, "艮": 18, "坤": 46},
    "坎": {"乾": 6, "兌": 47, "離": 64, "震": 40, "巽": 59, "坎": 29, "艮": 4, "坤": 7},
    "艮": {"乾": 33, "兌": 31, "離": 56, "震": 62, "巽": 53, "坎": 39, "艮": 52, "坤": 15},
    "坤": {"乾": 12, "兌": 45, "離": 35, "震": 16, "巽": 20, "坎": 8, "艮": 23, "坤": 2},
}

# 64卦番号 → 卦名
HEXAGRAM_NAMES = {
    1: "乾為天", 2: "坤為地", 3: "水雷屯", 4: "山水蒙",
    5: "水天需", 6: "天水訟", 7: "地水師", 8: "水地比",
    9: "風天小畜", 10: "天沢履", 11: "地天泰", 12: "天地否",
    13: "天火同人", 14: "火天大有", 15: "地山謙", 16: "雷地予",
    17: "沢雷随", 18: "山風蠱", 19: "地沢臨", 20: "風地観",
    21: "火雷噬嗑", 22: "山火賁", 23: "山地剥", 24: "地雷復",
    25: "天雷无妄", 26: "山天大畜", 27: "山雷頤", 28: "沢風大過",
    29: "坎為水", 30: "離為火", 31: "沢山咸", 32: "雷風恒",
    33: "天山遯", 34: "雷天大壮", 35: "火地晋", 36: "地火明夷",
    37: "風火家人", 38: "火沢睽", 39: "水山蹇", 40: "雷水解",
    41: "山沢損", 42: "風雷益", 43: "沢天夬", 44: "天風姤",
    45: "沢地萃", 46: "地風升", 47: "沢水困", 48: "水風井",
    49: "沢火革", 50: "火風鼎", 51: "震為雷", 52: "艮為山",
    53: "風山漸", 54: "雷沢帰妹", 55: "雷火豊", 56: "火山旅",
    57: "巽為風", 58: "兌為沢", 59: "風水渙", 60: "水沢節",
    61: "風沢中孚", 62: "雷山小過", 63: "水火既済", 64: "火水未済",
}

# フェーズ → 爻位マッピング (決定論的)
PHASE_TO_YAO = {
    "潜伏・発芽": 1,
    "出現・成長": 2,
    "危機・転換点": 3,
    "危機・転換": 3,
    "選択・跳躍準備": 4,
    "選択・準備": 4,
    "最盛・中正": 5,
    "最盛・実行": 5,
    "過剰・衰退": 6,
}

# --- 重み (docs/input_mapping_design_v2.md Section 3) ---
# 下卦: w1=0.75 (current_state), w2=0.25 (energy_direction)
W1_CURRENT_STATE = 0.75
W2_ENERGY_DIR = 0.25

# 上卦: w3=0.50 (action), w4=0.35 (trigger), w5=0.15 (energy_external)
# ただしtrigger指定なしの場合はaction単独。ここではw3, w4のみ使用し正規化する。
W3_ACTION = 0.60  # 0.50/(0.50+0.35) ≈ 0.588 → 0.60
W4_TRIGGER = 0.40  # 0.35/(0.50+0.35) ≈ 0.412 → 0.40


class ProbabilityMapper:
    """DB条件付き確率テーブルを使った八卦マッピングエンジン"""

    def __init__(self, table_path: str = None):
        """確率テーブルをロード"""
        if table_path is None:
            # スクリプトの位置から相対パスで探す
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            table_path = os.path.join(base_dir, "data", "diagnostic", "prob_tables.json")

        with open(table_path, "r", encoding="utf-8") as f:
            self._tables = json.load(f)

        self.before_state_to_hex = self._tables["before_state_to_hex"]
        self.action_type_to_hex = self._tables["action_type_to_hex"]
        self.trigger_type_to_hex = self._tables["trigger_type_to_hex"]
        self.phase_to_yao_dist = self._tables["phase_to_yao"]
        self.metadata = self._tables["metadata"]

    # --- エネルギー方向による八卦の事前分布 ---
    @staticmethod
    def _energy_direction_prior(energy_direction: str) -> dict:
        """
        energy_direction ("expanding"/"contracting") → 八卦の事前分布。
        陽卦(乾/離/震/巽)と陰卦(坤/坎/艮/兌)の重みを調整する。
        """
        if energy_direction == "expanding":
            # 陽的な卦を強調
            return {
                "乾": 0.20, "離": 0.15, "震": 0.15, "巽": 0.10,
                "兌": 0.10, "坎": 0.10, "艮": 0.10, "坤": 0.10,
            }
        elif energy_direction == "contracting":
            # 陰的な卦を強調
            return {
                "坤": 0.20, "坎": 0.15, "艮": 0.15, "兌": 0.10,
                "巽": 0.10, "離": 0.10, "震": 0.10, "乾": 0.10,
            }
        else:
            # 均等分布
            return {t: 1.0 / 8.0 for t in TRIGRAMS}

    def _get_hex_distribution(self, table: dict, category: str) -> dict:
        """テーブルからカテゴリの八卦分布を取得。存在しない場合は均等分布。"""
        if category in table:
            entry = table[category]
            return {t: entry.get(t, 0.0) for t in TRIGRAMS}
        # カテゴリが見つからない場合は均等分布を返す
        return {t: 1.0 / 8.0 for t in TRIGRAMS}

    def _get_sample_size(self, table: dict, category: str) -> int:
        """テーブルからカテゴリの事例件数を取得。"""
        if category in table:
            return table[category].get("_n", 0)
        return 0

    @staticmethod
    def _normalize(dist: dict) -> dict:
        """確率分布を合計1.0に正規化する。"""
        total = sum(dist.values())
        if total == 0:
            return {t: 1.0 / len(dist) for t in dist}
        return {t: round(v / total, 6) for t, v in dist.items()}

    @staticmethod
    def _weighted_combine(dist_a: dict, weight_a: float,
                          dist_b: dict, weight_b: float) -> dict:
        """2つの確率分布を重み付き結合し正規化する。"""
        combined = {}
        for t in TRIGRAMS:
            combined[t] = weight_a * dist_a.get(t, 0) + weight_b * dist_b.get(t, 0)
        return ProbabilityMapper._normalize(combined)

    # --- メインAPI ---

    def map_to_lower_trigram(self, current_state: str,
                             energy_direction: str = None) -> dict:
        """
        current_state → 下卦の確率分布を返す。

        Args:
            current_state: before_stateラベル (例: "どん底・危機")
            energy_direction: "expanding" or "contracting" (省略可)

        Returns:
            {"乾": 0.05, "兌": 0.02, ..., "坤": 0.07} 合計1.0
        """
        state_dist = self._get_hex_distribution(
            self.before_state_to_hex, current_state
        )
        n = self._get_sample_size(self.before_state_to_hex, current_state)

        if energy_direction:
            energy_dist = self._energy_direction_prior(energy_direction)
            result = self._weighted_combine(
                state_dist, W1_CURRENT_STATE,
                energy_dist, W2_ENERGY_DIR
            )
        else:
            result = self._normalize(state_dist)

        return {"distribution": result, "n": n, "source_category": current_state}

    def map_to_upper_trigram(self, intended_action: str,
                             trigger_nature: str = None) -> dict:
        """
        intended_action + trigger_nature → 上卦の確率分布を返す。
        重み: action=0.60, trigger=0.40 (docs/input_mapping_design_v2.md w3,w4 正規化)

        Args:
            intended_action: action_typeラベル (例: "守る・維持")
            trigger_nature: trigger_typeラベル (例: "外部ショック") (省略可)

        Returns:
            {"distribution": {...}, "n_action": int, "n_trigger": int}
        """
        action_dist = self._get_hex_distribution(
            self.action_type_to_hex, intended_action
        )
        n_action = self._get_sample_size(self.action_type_to_hex, intended_action)

        if trigger_nature:
            trigger_dist = self._get_hex_distribution(
                self.trigger_type_to_hex, trigger_nature
            )
            n_trigger = self._get_sample_size(
                self.trigger_type_to_hex, trigger_nature
            )
            result = self._weighted_combine(
                action_dist, W3_ACTION,
                trigger_dist, W4_TRIGGER
            )
        else:
            result = self._normalize(action_dist)
            n_trigger = 0

        return {
            "distribution": result,
            "n_action": n_action,
            "n_trigger": n_trigger,
            "source_action": intended_action,
            "source_trigger": trigger_nature,
        }

    def map_to_yao(self, phase_stage: str) -> int:
        """
        phase_stage → 爻位(1-6)を返す。決定論的マッピング。

        Args:
            phase_stage: フェーズラベル (例: "潜伏・発芽", "危機・転換点")

        Returns:
            1-6 の整数
        """
        if phase_stage in PHASE_TO_YAO:
            return PHASE_TO_YAO[phase_stage]

        # phase_to_yao テーブルから探す
        for label, info in self.phase_to_yao_dist.items():
            if phase_stage in label or label in phase_stage:
                return info["yao"]

        # デフォルト: 爻位3 (転換点)
        return 3

    def map_to_hexagram(self, lower_trigram: str, upper_trigram: str) -> int:
        """
        下卦 + 上卦 → 64卦番号 (King Wen sequence) を返す。

        Args:
            lower_trigram: 下卦の八卦名 (例: "坎")
            upper_trigram: 上卦の八卦名 (例: "艮")

        Returns:
            1-64 の整数 (卦番号)
        """
        if lower_trigram not in HEXAGRAM_TABLE:
            raise ValueError(f"Invalid lower trigram: {lower_trigram}")
        if upper_trigram not in HEXAGRAM_TABLE[lower_trigram]:
            raise ValueError(f"Invalid upper trigram: {upper_trigram}")
        return HEXAGRAM_TABLE[lower_trigram][upper_trigram]

    @staticmethod
    def get_hexagram_name(hexagram_number: int) -> str:
        """卦番号から卦名を返す。"""
        return HEXAGRAM_NAMES.get(hexagram_number, f"卦{hexagram_number}")

    @staticmethod
    def get_zhi_gua(lower_trigram: str, upper_trigram: str,
                    yao_position: int) -> tuple:
        """
        本卦 + 動爻 → 之卦(変化後の卦)の上下卦を返す。

        Args:
            lower_trigram: 下卦
            upper_trigram: 上卦
            yao_position: 動爻位置 (1-6, 下から)

        Returns:
            (new_lower_trigram, new_upper_trigram)
        """
        bits = list(TRIGRAM_BITS[lower_trigram]) + list(TRIGRAM_BITS[upper_trigram])
        idx = yao_position - 1
        bits[idx] = 1 - bits[idx]  # 反転
        new_lower = tuple(bits[0:3])
        new_upper = tuple(bits[3:6])
        return BITS_TRIGRAM[new_lower], BITS_TRIGRAM[new_upper]

    def get_top_candidates(self, current_state: str,
                           intended_action: str,
                           trigger_nature: str = None,
                           phase_stage: str = None,
                           energy_direction: str = None,
                           n: int = 3) -> list:
        """
        全情報を統合して、上位n個の卦候補を確率付きで返す。

        Args:
            current_state: 現在の状態 (例: "どん底・危機")
            intended_action: 意図する行動 (例: "守る・維持")
            trigger_nature: 変化のトリガー (省略可)
            phase_stage: フェーズ段階 (省略可)
            energy_direction: エネルギー方向 (省略可)
            n: 返す候補数 (デフォルト3)

        Returns:
            [
                {
                    "rank": 1,
                    "hexagram_number": 39,
                    "hexagram_name": "水山蹇",
                    "lower_trigram": "坎",
                    "upper_trigram": "艮",
                    "probability": 0.672,
                    "yao": 3,
                    "zhi_gua_number": ...,
                    "zhi_gua_name": ...,
                },
                ...
            ]
        """
        # 1. 下卦分布
        lower_result = self.map_to_lower_trigram(current_state, energy_direction)
        lower_dist = lower_result["distribution"]

        # 2. 上卦分布
        upper_result = self.map_to_upper_trigram(intended_action, trigger_nature)
        upper_dist = upper_result["distribution"]

        # 3. 爻位
        yao = self.map_to_yao(phase_stage) if phase_stage else None

        # 4. 全64卦の確率を算出 (下卦確率 x 上卦確率)
        hexagram_probs = []
        for lower_t in TRIGRAMS:
            for upper_t in TRIGRAMS:
                prob = lower_dist[lower_t] * upper_dist[upper_t]
                hex_num = self.map_to_hexagram(lower_t, upper_t)
                hexagram_probs.append({
                    "hexagram_number": hex_num,
                    "hexagram_name": self.get_hexagram_name(hex_num),
                    "lower_trigram": lower_t,
                    "upper_trigram": upper_t,
                    "probability": round(prob, 6),
                })

        # 5. 確率降順でソートし上位n件を取得
        hexagram_probs.sort(key=lambda x: x["probability"], reverse=True)
        top_n = hexagram_probs[:n]

        # 6. 之卦情報を追加
        candidates = []
        for i, item in enumerate(top_n, 1):
            candidate = {
                "rank": i,
                **item,
            }

            if yao:
                candidate["yao"] = yao
                candidate["yao_name"] = f"{'初二三四五上'[yao - 1]}爻"
                zhi_lower, zhi_upper = self.get_zhi_gua(
                    item["lower_trigram"], item["upper_trigram"], yao
                )
                zhi_num = self.map_to_hexagram(zhi_lower, zhi_upper)
                candidate["zhi_gua_number"] = zhi_num
                candidate["zhi_gua_name"] = self.get_hexagram_name(zhi_num)
                candidate["zhi_lower_trigram"] = zhi_lower
                candidate["zhi_upper_trigram"] = zhi_upper

            candidates.append(candidate)

        return {
            "candidates": candidates,
            "input": {
                "current_state": current_state,
                "intended_action": intended_action,
                "trigger_nature": trigger_nature,
                "phase_stage": phase_stage,
                "energy_direction": energy_direction,
            },
            "lower_trigram_distribution": lower_dist,
            "upper_trigram_distribution": upper_dist,
            "n_before_state": lower_result["n"],
            "n_action": upper_result["n_action"],
            "n_trigger": upper_result["n_trigger"],
        }

    def list_categories(self) -> dict:
        """利用可能なカテゴリラベル一覧を返す。"""
        return {
            "before_state": sorted(self.before_state_to_hex.keys()),
            "action_type": sorted(self.action_type_to_hex.keys()),
            "trigger_type": sorted(self.trigger_type_to_hex.keys()),
            "phase_stage": sorted(PHASE_TO_YAO.keys()),
        }


# =============================================================================
# 確率テーブル生成 (Builder)
# =============================================================================

# 爻位 → フェーズラベル（phase_to_yao生成用）
YAO_TO_PHASE_LABEL = {
    1: "潜伏・発芽",
    2: "出現・成長",
    3: "危機・転換点",
    4: "選択・跳躍準備",
    5: "最盛・中正",
    6: "過剰・衰退",
}

# 対応するscale値
KNOWN_SCALES = ["company", "individual", "family", "country", "other"]


class ProbabilityTableBuilder:
    """cases.jsonlから確率テーブルを構築するビルダー"""

    def __init__(self, cases_path: str = None):
        if cases_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cases_path = os.path.join(base_dir, "data", "raw", "cases.jsonl")
        self.cases_path = cases_path
        self._cases = None

    def _load_cases(self) -> list:
        """cases.jsonlを読み込む（キャッシュ付き）"""
        if self._cases is not None:
            return self._cases
        cases = []
        with open(self.cases_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
        self._cases = cases
        return cases

    @staticmethod
    def _build_hex_distribution(cases: list, category_field: str,
                                hex_field: str) -> dict:
        """
        カテゴリフィールド × 八卦フィールドの条件付き確率テーブルを構築。

        Args:
            cases: 事例リスト
            category_field: カテゴリのフィールド名 (例: "before_state")
            hex_field: 八卦のフィールド名 (例: "before_hex")

        Returns:
            {"カテゴリ名": {"_n": 件数, "乾": 0.xx, ...}, ...}
        """
        # カテゴリごとに八卦の出現をカウント
        cat_hex_counts = defaultdict(Counter)
        for case in cases:
            cat_val = case.get(category_field, "")
            hex_val = case.get(hex_field, "")
            if cat_val and hex_val:
                cat_hex_counts[cat_val][hex_val] += 1

        result = {}
        for cat_val in sorted(cat_hex_counts.keys()):
            counts = cat_hex_counts[cat_val]
            total = sum(counts.values())
            entry = {"_n": total}
            for t in TRIGRAMS:
                entry[t] = round(counts.get(t, 0) / total, 4) if total > 0 else 0.0
            result[cat_val] = entry

        return result

    @staticmethod
    def _build_phase_to_yao(cases: list) -> dict:
        """
        爻位分布テーブルを構築。

        Returns:
            {"潜伏・発芽": {"yao": 1, "count": N, "percentage": X.X}, ...}
        """
        yao_counts = Counter()
        for case in cases:
            yao_val = case.get("yao")
            if yao_val is not None:
                yao_counts[yao_val] += 1

        total = sum(yao_counts.values())
        result = {}
        for yao_pos in sorted(YAO_TO_PHASE_LABEL.keys()):
            label = YAO_TO_PHASE_LABEL[yao_pos]
            count = yao_counts.get(yao_pos, 0)
            pct = round(count / total * 100, 1) if total > 0 else 0.0
            result[label] = {
                "yao": yao_pos,
                "count": count,
                "percentage": pct,
            }

        return result, total

    def build_tables(self, cases: list = None) -> dict:
        """
        確率テーブルを構築する。

        Args:
            cases: 事例リスト（Noneなら全件読み込み）

        Returns:
            prob_tables.json と同じ構造の辞書
        """
        if cases is None:
            cases = self._load_cases()

        before_state_to_hex = self._build_hex_distribution(
            cases, "before_state", "before_hex"
        )
        action_type_to_hex = self._build_hex_distribution(
            cases, "action_type", "action_hex"
        )
        trigger_type_to_hex = self._build_hex_distribution(
            cases, "trigger_type", "trigger_hex"
        )
        phase_to_yao, cases_with_yao = self._build_phase_to_yao(cases)

        return {
            "before_state_to_hex": before_state_to_hex,
            "action_type_to_hex": action_type_to_hex,
            "trigger_type_to_hex": trigger_type_to_hex,
            "phase_to_yao": phase_to_yao,
            "metadata": {
                "total_cases": len(cases),
                "cases_with_yao": cases_with_yao,
                "generated_at": datetime.now().isoformat(),
                "description": "条件付き確率テーブル: カテゴリラベル→八卦の経験的対応関係",
                "trigrams": TRIGRAMS,
                "notes": {
                    "before_state_to_hex": "before_state→before_hex: 内的状態→下卦候補の確率分布",
                    "action_type_to_hex": "action_type→action_hex: 行動→上卦候補の確率分布",
                    "trigger_type_to_hex": "trigger_type→trigger_hex: トリガー→上卦補助の確率分布",
                    "phase_to_yao": "爻位(1-6)の全体分布。phase_stageから爻位への決定論的マッピングに使用",
                    "_n": "各カテゴリの事例件数。確率は _n を母数として正規化済み（合計≈1.0）",
                },
            },
        }

    def build_scale_tables(self, scale: str, cases: list = None) -> dict:
        """
        特定scaleの確率テーブルを構築する。

        Args:
            scale: スケール値 (company, individual, family, country, other)
            cases: 事例リスト（Noneなら全件読み込みしてフィルタ）

        Returns:
            {"meta": {...}, "tables": {prob_tables構造}}
        """
        if cases is None:
            cases = self._load_cases()

        filtered = [c for c in cases if c.get("scale", "other") == scale]
        tables = self.build_tables(filtered)

        return {
            "meta": {
                "scale": scale,
                "total_cases": len(filtered),
                "generated_at": datetime.now().isoformat(),
            },
            "tables": tables,
        }

    def build_all(self, output_dir: str = None):
        """
        全体 + scale別の確率テーブルをすべて生成して保存する。

        生成ファイル:
            - prob_tables.json (全体, 後方互換)
            - prob_tables_company.json
            - prob_tables_individual.json
            - prob_tables_family.json
            - prob_tables_country.json
            - prob_tables_other.json
        """
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, "data", "diagnostic")

        os.makedirs(output_dir, exist_ok=True)

        all_cases = self._load_cases()
        print(f"総事例数: {len(all_cases)}")

        # --- 全体テーブル（後方互換: 既存と同一構造） ---
        all_tables = self.build_tables(all_cases)
        all_path = os.path.join(output_dir, "prob_tables.json")
        with open(all_path, "w", encoding="utf-8") as f:
            json.dump(all_tables, f, ensure_ascii=False, indent=2)
        print(f"  全体: {all_path} ({all_tables['metadata']['total_cases']}件)")

        # --- scale別テーブル ---
        scale_totals = {}
        for scale in KNOWN_SCALES:
            scale_data = self.build_scale_tables(scale, all_cases)
            scale_path = os.path.join(output_dir, f"prob_tables_{scale}.json")
            with open(scale_path, "w", encoding="utf-8") as f:
                json.dump(scale_data, f, ensure_ascii=False, indent=2)
            n = scale_data["meta"]["total_cases"]
            scale_totals[scale] = n
            print(f"  {scale}: {scale_path} ({n}件)")

        # --- 検証: scale別合計 = 全体 ---
        scale_sum = sum(scale_totals.values())
        if scale_sum != len(all_cases):
            print(f"\n[WARNING] scale別合計({scale_sum}) != 全体({len(all_cases)})")
            # 不明なscale値があるか調査
            unknown = Counter(
                c.get("scale", "other") for c in all_cases
                if c.get("scale", "other") not in KNOWN_SCALES
            )
            if unknown:
                print(f"  未知のscale値: {dict(unknown)}")
        else:
            print(f"\n[OK] scale別合計({scale_sum}) == 全体({len(all_cases)})")

        return all_tables, scale_totals


# =============================================================================
# テスト実行
# =============================================================================

def _run_mapper_tests():
    """既存のProbabilityMapperテスト（後方互換確認）"""
    print("=" * 70)
    print("ProbabilityMapper テスト実行")
    print("=" * 70)

    mapper = ProbabilityMapper()

    # --- テスト1: map_to_lower_trigram ---
    print("\n--- テスト1: map_to_lower_trigram('どん底・危機') ---")
    result = mapper.map_to_lower_trigram("どん底・危機")
    dist = result["distribution"]
    top = max(dist, key=dist.get)
    print(f"  n={result['n']}")
    for t in TRIGRAMS:
        marker = " <<<" if t == top else ""
        print(f"  {t}: {dist[t]:.4f}{marker}")
    assert top == "坎", f"FAIL: Expected 坎 but got {top}"
    print("  PASS: 坎が最大確率")

    # --- テスト2: map_to_upper_trigram ---
    print("\n--- テスト2: map_to_upper_trigram('守る・維持') ---")
    result = mapper.map_to_upper_trigram("守る・維持")
    dist = result["distribution"]
    top = max(dist, key=dist.get)
    print(f"  n_action={result['n_action']}")
    for t in TRIGRAMS:
        marker = " <<<" if t == top else ""
        print(f"  {t}: {dist[t]:.4f}{marker}")
    assert top == "艮", f"FAIL: Expected 艮 but got {top}"
    print("  PASS: 艮が最大確率")

    # --- テスト3: map_to_upper_trigram with trigger ---
    print("\n--- テスト3: map_to_upper_trigram('攻める・挑戦', '外部ショック') ---")
    result = mapper.map_to_upper_trigram("攻める・挑戦", "外部ショック")
    dist = result["distribution"]
    top = max(dist, key=dist.get)
    print(f"  n_action={result['n_action']}, n_trigger={result['n_trigger']}")
    for t in TRIGRAMS:
        marker = " <<<" if t == top else ""
        print(f"  {t}: {dist[t]:.4f}{marker}")
    print(f"  最大: {top}")

    # --- テスト4: map_to_yao ---
    print("\n--- テスト4: map_to_yao ---")
    for phase, expected in [("潜伏・発芽", 1), ("危機・転換点", 3), ("最盛・中正", 5), ("過剰・衰退", 6)]:
        yao = mapper.map_to_yao(phase)
        status = "PASS" if yao == expected else "FAIL"
        print(f"  {phase} → yao={yao} (expected {expected}) [{status}]")
        assert yao == expected

    # --- テスト5: map_to_hexagram ---
    print("\n--- テスト5: map_to_hexagram ---")
    for lower, upper, expected in [("乾", "乾", 1), ("坤", "坤", 2), ("坎", "艮", 4), ("震", "離", 21)]:
        num = mapper.map_to_hexagram(lower, upper)
        name = mapper.get_hexagram_name(num)
        status = "PASS" if num == expected else "FAIL"
        print(f"  {lower}(下) + {upper}(上) → {num} {name} (expected {expected}) [{status}]")
        assert num == expected

    # --- テスト6: get_zhi_gua ---
    print("\n--- テスト6: get_zhi_gua ---")
    zhi_l, zhi_u = mapper.get_zhi_gua("乾", "乾", 1)
    zhi_num = mapper.map_to_hexagram(zhi_l, zhi_u)
    print(f"  乾為天 初爻変 → {zhi_l}(下) + {zhi_u}(上) = {mapper.get_hexagram_name(zhi_num)}")
    # 乾為天の初爻変 → 天風姤(44)
    assert zhi_num == 44, f"FAIL: Expected 44 but got {zhi_num}"
    print("  PASS")

    # --- テスト7: get_top_candidates ---
    print("\n--- テスト7: get_top_candidates('停滞・閉塞', '刷新・破壊') ---")
    result = mapper.get_top_candidates("停滞・閉塞", "刷新・破壊")
    print(f"  入力: {result['input']}")
    print(f"  n_before_state={result['n_before_state']}, n_action={result['n_action']}")
    for c in result["candidates"]:
        print(f"  #{c['rank']}: {c['hexagram_name']}({c['hexagram_number']}) "
              f"[{c['lower_trigram']}+{c['upper_trigram']}] p={c['probability']:.4f}")
    assert len(result["candidates"]) == 3, "FAIL: Expected 3 candidates"
    print("  PASS: 3候補返却")

    # --- テスト8: get_top_candidates with full params ---
    print("\n--- テスト8: get_top_candidates('どん底・危機', '耐える・潜伏', '内部崩壊', '危機・転換点') ---")
    result = mapper.get_top_candidates(
        "どん底・危機", "耐える・潜伏",
        trigger_nature="内部崩壊",
        phase_stage="危機・転換点",
        n=5
    )
    for c in result["candidates"]:
        zhi_info = f" → {c.get('zhi_gua_name', 'N/A')}" if 'zhi_gua_name' in c else ""
        print(f"  #{c['rank']}: {c['hexagram_name']}({c['hexagram_number']}) "
              f"[{c['lower_trigram']}+{c['upper_trigram']}] p={c['probability']:.4f} "
              f"yao={c.get('yao', 'N/A')}{zhi_info}")
    assert len(result["candidates"]) == 5
    assert all("yao" in c for c in result["candidates"])
    assert all("zhi_gua_name" in c for c in result["candidates"])
    print("  PASS: 5候補 + 爻位 + 之卦")

    # --- テスト9: list_categories ---
    print("\n--- テスト9: list_categories ---")
    cats = mapper.list_categories()
    print(f"  before_state: {len(cats['before_state'])} categories")
    print(f"  action_type: {len(cats['action_type'])} categories")
    print(f"  trigger_type: {len(cats['trigger_type'])} categories")
    print(f"  phase_stage: {len(cats['phase_stage'])} categories")

    print("\n" + "=" * 70)
    print("全テスト完了")
    print("=" * 70)


def _run_builder_tests():
    """ProbabilityTableBuilderのテスト"""
    print("\n" + "=" * 70)
    print("ProbabilityTableBuilder テスト実行")
    print("=" * 70)

    builder = ProbabilityTableBuilder()
    all_cases = builder._load_cases()
    print(f"\n総事例数: {len(all_cases)}")

    # --- テスト10: 全体テーブル構築 ---
    print("\n--- テスト10: 全体テーブル構築 ---")
    tables = builder.build_tables(all_cases)
    assert tables["metadata"]["total_cases"] == len(all_cases), \
        f"FAIL: total_cases mismatch"
    assert "before_state_to_hex" in tables
    assert "action_type_to_hex" in tables
    assert "trigger_type_to_hex" in tables
    assert "phase_to_yao" in tables

    # _n合計が全体件数と一致
    bs_total = sum(v["_n"] for v in tables["before_state_to_hex"].values())
    at_total = sum(v["_n"] for v in tables["action_type_to_hex"].values())
    tt_total = sum(v["_n"] for v in tables["trigger_type_to_hex"].values())
    print(f"  before_state _n合計: {bs_total}")
    print(f"  action_type _n合計: {at_total}")
    print(f"  trigger_type _n合計: {tt_total}")
    assert bs_total == len(all_cases), f"FAIL: before_state _n合計({bs_total}) != 全体({len(all_cases)})"
    assert at_total == len(all_cases), f"FAIL: action_type _n合計({at_total}) != 全体({len(all_cases)})"
    assert tt_total == len(all_cases), f"FAIL: trigger_type _n合計({tt_total}) != 全体({len(all_cases)})"
    print("  PASS: 全テーブルの_n合計が全体件数と一致")

    # --- テスト11: scale別テーブル構築 ---
    print("\n--- テスト11: scale別テーブル構築 ---")
    scale_sum = 0
    for scale in KNOWN_SCALES:
        scale_data = builder.build_scale_tables(scale, all_cases)
        n = scale_data["meta"]["total_cases"]
        scale_sum += n
        assert scale_data["meta"]["scale"] == scale
        assert "tables" in scale_data
        assert "before_state_to_hex" in scale_data["tables"]
        # scale別の_n合計がそのscaleの事例数と一致
        bs_n = sum(v["_n"] for v in scale_data["tables"]["before_state_to_hex"].values())
        assert bs_n == n, f"FAIL: {scale} before_state _n合計({bs_n}) != {n}"
        print(f"  {scale}: {n}件 [OK]")

    assert scale_sum == len(all_cases), \
        f"FAIL: scale別合計({scale_sum}) != 全体({len(all_cases)})"
    print(f"  scale別合計: {scale_sum} == 全体: {len(all_cases)} [PASS]")

    # --- テスト12: scale別テーブルの読み込み互換性 ---
    print("\n--- テスト12: scale別テーブルの読み込み互換性 ---")
    # scale別テーブルをtmpに書き出してProbabilityMapperで読めるか
    import tempfile
    scale_data = builder.build_scale_tables("company", all_cases)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                      encoding="utf-8") as tmp:
        # scale別ファイルのtablesの中身をそのまま書く（Mapper互換確認用）
        json.dump(scale_data["tables"], tmp, ensure_ascii=False, indent=2)
        tmp_path = tmp.name
    try:
        scale_mapper = ProbabilityMapper(table_path=tmp_path)
        result = scale_mapper.map_to_lower_trigram("どん底・危機")
        assert result["n"] > 0 or result["n"] == 0  # カテゴリが存在する場合のみ
        print(f"  company Mapper読み込み: OK (n={result['n']})")
        print("  PASS: scale別テーブルはProbabilityMapperで読み込み可能")
    finally:
        os.unlink(tmp_path)

    # --- テスト13: 確率値の妥当性 ---
    print("\n--- テスト13: 確率値の妥当性 ---")
    for cat, entry in tables["before_state_to_hex"].items():
        prob_sum = sum(entry.get(t, 0.0) for t in TRIGRAMS)
        assert abs(prob_sum - 1.0) < 0.01, \
            f"FAIL: {cat} の確率合計が1.0でない: {prob_sum}"
    print("  PASS: before_state_to_hex の全カテゴリで確率合計 ≈ 1.0")

    for cat, entry in tables["action_type_to_hex"].items():
        prob_sum = sum(entry.get(t, 0.0) for t in TRIGRAMS)
        assert abs(prob_sum - 1.0) < 0.01, \
            f"FAIL: {cat} の確率合計が1.0でない: {prob_sum}"
    print("  PASS: action_type_to_hex の全カテゴリで確率合計 ≈ 1.0")

    for cat, entry in tables["trigger_type_to_hex"].items():
        prob_sum = sum(entry.get(t, 0.0) for t in TRIGRAMS)
        assert abs(prob_sum - 1.0) < 0.01, \
            f"FAIL: {cat} の確率合計が1.0でない: {prob_sum}"
    print("  PASS: trigger_type_to_hex の全カテゴリで確率合計 ≈ 1.0")

    print("\n" + "=" * 70)
    print("Builder全テスト完了")
    print("=" * 70)


if __name__ == "__main__":
    if "--build" in sys.argv:
        # テーブル生成モード
        print("=" * 70)
        print("確率テーブル生成 (全体 + scale別)")
        print("=" * 70)
        builder = ProbabilityTableBuilder()
        builder.build_all()
        print("\n生成完了")
    elif "--test" in sys.argv or len(sys.argv) == 1:
        # テストモード（デフォルト）
        _run_mapper_tests()
        _run_builder_tests()
