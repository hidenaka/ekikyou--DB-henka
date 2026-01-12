#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI判定スクリプト: 64卦 + 爻位の同時判定

事例の内容（story_summary, before_state, after_state, pattern_type, outcome等）
から適切な64卦と爻位（1-6）を判定する。

判定ロジック:
1. 64卦判定: パターンタイプ、結果、状態変化の組み合わせから卦を選択
2. 爻位判定: 物語のフェーズ（開始・成長・過渡・転換・頂点・終結）を判定
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import re
import random

# ==============================================================================
# パスと設定
# ==============================================================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

CASES_PATH = PROJECT_ROOT / "data/raw/cases.jsonl"
HEXAGRAM_MASTER_PATH = PROJECT_ROOT / "data/hexagrams/hexagram_master.json"
YAO_MASTER_PATH = PROJECT_ROOT / "data/hexagrams/yao_master.json"

# ==============================================================================
# 64卦のカテゴリ分類（内容ベース判定用）
# ==============================================================================

# パターンタイプ → 適合する卦のリスト（優先度順、拡張版：全64卦カバー）
PATTERN_TO_HEXAGRAMS: Dict[str, List[int]] = {
    # 成功系パターン（安定成長・好調）
    "Steady_Growth": [11, 53, 46, 32, 57, 16, 25, 58, 61, 9, 48, 21],  # 泰、漸、升、恒、巽、豫、无妄、兌、中孚、小畜、井、噬嗑
    "Strategic_Patience": [5, 20, 33, 4, 60, 15, 22, 26, 52, 62, 9, 48],  # 需、観、遯、蒙、節、謙、賁、大畜、艮、小過、小畜、井
    "Opportunity_Seized": [42, 35, 14, 55, 34, 1, 17, 30, 56, 10],  # 益、晋、大有、豊、大壮、乾、随、離、旅、履
    "Collaborative_Rise": [8, 13, 37, 45, 31, 19, 58, 41, 61, 28],  # 比、同人、家人、萃、咸、臨、兌、損、中孚、大過
    "Bold_Leap": [1, 34, 43, 49, 17, 25, 14, 55, 30, 56],  # 乾、大壮、夬、革、随、无妄、大有、豊、離、旅

    # 危機対応系パターン
    "Shock_Recovery": [51, 40, 24, 21, 3, 16, 55, 63, 27, 42],  # 震、解、復、噬嗑、屯、豫、豊、既済、頤、益
    "Crisis_Pivot": [49, 18, 64, 59, 47, 38, 50, 3, 48, 5],  # 革、蠱、未済、渙、困、睽、鼎、屯、井、需
    "Adaptive_Survival": [29, 39, 47, 36, 6, 7, 60, 4, 48, 63],  # 坎、蹇、困、明夷、訟、師、節、蒙、井、既済

    # 衰退系パターン
    "Hubris_Collapse": [12, 23, 36, 44, 33, 28, 54, 38, 10, 6],  # 否、剥、明夷、姤、遯、大過、帰妹、睽、履、訟
    "Slow_Decline": [23, 12, 33, 52, 62, 36, 4, 20, 15, 2],  # 剥、否、遯、艮、小過、明夷、蒙、観、謙、坤
    "Quiet_Fade": [52, 62, 15, 20, 4, 2, 22, 26, 41, 9],  # 艮、小過、謙、観、蒙、坤、賁、大畜、損、小畜
    "Managed_Decline": [19, 41, 15, 2, 7, 8, 46, 53, 60, 48],  # 臨、損、謙、坤、師、比、升、漸、節、井
    "Failed_Attempt": [47, 29, 39, 4, 3, 6, 36, 64, 59, 54],  # 困、坎、蹇、蒙、屯、訟、明夷、未済、渙、帰妹

    # 変革系パターン
    "Legacy_Burden": [18, 23, 36, 12, 44, 27, 50, 28, 53, 32],  # 蠱、剥、明夷、否、姤、頤、鼎、大過、漸、恒
    "Tech_Disruption": [49, 50, 64, 63, 21, 30, 56, 38, 17, 35],  # 革、鼎、未済、既済、噬嗑、離、旅、睽、随、晋
    "Market_Shift": [54, 38, 10, 6, 47, 44, 31, 28, 43, 58],  # 帰妹、睽、履、訟、困、姤、咸、大過、夬、兌
    "Generational_Change": [18, 50, 27, 44, 32, 53, 37, 13, 19, 45],  # 蠱、鼎、頤、姤、恒、漸、家人、同人、臨、萃

    # その他
    "Internal_Conflict": [6, 38, 47, 10, 21, 44, 54, 43, 28, 12, 9],  # 訟、睽、困、履、噬嗑、姤、帰妹、夬、大過、否、小畜
    "External_Threat": [7, 13, 6, 33, 36, 29, 39, 5, 40, 51, 21, 48],  # 師、同人、訟、遯、明夷、坎、蹇、需、解、震、噬嗑、井
}

# 結果 → 卦の重み付け調整
OUTCOME_HEXAGRAM_WEIGHTS: Dict[str, Dict[int, float]] = {
    "Success": {
        1: 1.5, 11: 1.5, 14: 1.5, 35: 1.5, 42: 1.5, 55: 1.5,  # 成功に適合
        47: 0.3, 29: 0.3, 39: 0.3, 36: 0.3,  # 失敗卦は下げる
    },
    "Failure": {
        47: 1.5, 29: 1.5, 39: 1.5, 36: 1.5, 12: 1.5, 23: 1.5,  # 失敗に適合
        1: 0.3, 11: 0.3, 14: 0.3, 55: 0.3,  # 成功卦は下げる
    },
    "Mixed": {
        63: 1.3, 64: 1.3, 62: 1.3, 9: 1.3, 10: 1.3,  # 両義的な卦
    },
}

# before_state → 卦の関連付け（拡張版、全64卦カバー）
BEFORE_STATE_HINTS: Dict[str, List[int]] = {
    "安定・平和": [11, 32, 2, 8, 15, 58, 45, 37, 31, 61, 9, 48],  # 泰、恒、坤、比、謙、兌、萃、家人、咸、中孚、小畜、井
    "停滞・閉塞": [12, 52, 23, 47, 33, 62, 4, 20, 22, 26, 9],  # 否、艮、剥、困、遯、小過、蒙、観、賁、大畜、小畜
    "危機・困難": [29, 39, 47, 3, 6, 36, 64, 59, 7, 48, 21],  # 坎、蹇、困、屯、訟、明夷、未済、渙、師、井、噬嗑
    "成長期・好調": [1, 14, 35, 34, 55, 42, 17, 30, 25, 10, 9, 21],  # 乾、大有、晋、大壮、豊、益、随、離、无妄、履、小畜、噬嗑
    "混乱・不安定": [59, 64, 4, 6, 38, 54, 28, 21, 43, 44, 48],  # 渙、未済、蒙、訟、睽、帰妹、大過、噬嗑、夬、姤、井
    "衰退期・下降": [23, 36, 33, 12, 20, 4, 52, 62, 41, 9, 48],  # 剥、明夷、遯、否、観、蒙、艮、小過、損、小畜、井
    "変革期": [49, 50, 18, 24, 51, 40, 63, 64, 27, 56, 21],  # 革、鼎、蠱、復、震、解、既済、未済、頤、旅、噬嗑
}

# after_state → 卦の関連付け（拡張版、全64卦カバー）
AFTER_STATE_HINTS: Dict[str, List[int]] = {
    "安定・平和": [11, 32, 63, 2, 15, 37, 8, 53, 60, 48, 9],  # 泰、恒、既済、坤、謙、家人、比、漸、節、井、小畜
    "V字回復・大成功": [1, 14, 55, 35, 34, 42, 17, 25, 30, 43, 21],  # 乾、大有、豊、晋、大壮、益、随、无妄、離、夬、噬嗑
    "緩やかな成長": [53, 46, 42, 11, 32, 57, 16, 58, 61, 19, 9, 48],  # 漸、升、益、泰、恒、巽、豫、兌、中孚、臨、小畜、井
    "衰退・失敗": [23, 36, 12, 47, 29, 39, 6, 38, 54, 44, 21],  # 剥、明夷、否、困、坎、蹇、訟、睽、帰妹、姤、噬嗑
    "変質・新生": [49, 24, 50, 64, 51, 40, 18, 3, 21, 27, 48],  # 革、復、鼎、未済、震、解、蠱、屯、噬嗑、頤、井
    "完全崩壊": [23, 2, 36, 29, 47, 39, 4, 12, 33, 28, 9],  # 剥、坤、明夷、坎、困、蹇、蒙、否、遯、大過、小畜
    "維持・現状維持": [52, 62, 15, 32, 20, 22, 26, 57, 9, 41, 48, 21],  # 艮、小過、謙、恒、観、賁、大畜、巽、小畜、損、井、噬嗑
}

# ==============================================================================
# 爻位判定ルール
# ==============================================================================

# パターンタイプ → 爻位の傾向（拡張版：全6爻に分散）
PATTERN_YAO_TENDENCY: Dict[str, List[int]] = {
    # 成功パターン：全段階に分散
    "Steady_Growth": [1, 2, 3, 4, 5],  # 初期〜頂点まで
    "Strategic_Patience": [1, 2, 3, 4],  # 待機〜転換まで
    "Opportunity_Seized": [3, 4, 5, 6],  # 過渡〜終結
    "Collaborative_Rise": [2, 3, 4, 5],  # 成長〜頂点
    "Bold_Leap": [4, 5, 1, 2],  # 転換、頂点、新規開始

    # 危機対応：広範囲
    "Shock_Recovery": [1, 2, 3, 4, 5],  # 初期〜回復
    "Crisis_Pivot": [3, 4, 5, 6],  # 過渡〜転落
    "Adaptive_Survival": [1, 2, 3, 4, 5, 6],  # 全段階

    # 衰退パターン：後半多め
    "Hubris_Collapse": [4, 5, 6, 3],  # 転換〜終結
    "Slow_Decline": [3, 4, 5, 6],  # 過渡〜終結
    "Quiet_Fade": [4, 5, 6, 2],  # 転換〜終結、一部成長
    "Managed_Decline": [3, 4, 5, 6],  # 過渡〜終結
    "Failed_Attempt": [2, 3, 4, 5, 6],  # 成長〜終結

    # 変革系：中盤中心
    "Legacy_Burden": [2, 3, 4, 5],  # 成長〜頂点
    "Tech_Disruption": [3, 4, 5, 6],  # 過渡〜終結
    "Market_Shift": [2, 3, 4, 5],  # 成長〜頂点
    "Generational_Change": [3, 4, 5, 1],  # 過渡、転換、頂点、新開始

    # その他
    "Internal_Conflict": [2, 3, 4, 5],  # 成長〜頂点
    "External_Threat": [1, 2, 3, 4, 5, 6],  # 全段階
}

# outcome → 爻位の調整
OUTCOME_YAO_ADJUSTMENT: Dict[str, Dict[int, float]] = {
    "Success": {
        5: 1.5,  # 成功は5爻（最盛期）が多い
        4: 1.2,  # 転換成功
        6: 0.5,  # 行き過ぎは少ない
    },
    "Failure": {
        6: 1.5,  # 失敗は6爻（行き過ぎ・転落）
        3: 1.2,  # 分岐点での失敗
        5: 0.5,  # 最盛期は少ない
    },
    "Mixed": {
        3: 1.3,  # 過渡期
        4: 1.3,  # 転換点
    },
}

# キーワードから爻位を推測
YAO_KEYWORDS: Dict[int, List[str]] = {
    1: ["始まり", "準備", "潜む", "萌芽", "初期", "着手", "開始", "立ち上げ", "創業", "着想"],
    2: ["成長", "認められ", "進展", "実績", "発展", "拡大", "認知", "定着", "基盤"],
    3: ["困難", "分岐", "過渡", "試練", "危機", "転機", "岐路", "苦難", "挑戦", "転換期"],
    4: ["飛躍", "転換", "決断", "転機", "躍進", "ブレイク", "変革", "大転換", "勝負"],
    5: ["頂点", "成功", "最盛", "リーダー", "達成", "絶頂", "全盛", "頂上", "支配"],
    6: ["終結", "過度", "転落", "衰退", "没落", "崩壊", "終焉", "行き過ぎ", "引退", "撤退"],
}


def load_reference_data() -> Tuple[Dict, Dict]:
    """リファレンスデータを読み込む"""
    with open(HEXAGRAM_MASTER_PATH, "r", encoding="utf-8") as f:
        hexagram_master = json.load(f)

    with open(YAO_MASTER_PATH, "r", encoding="utf-8") as f:
        yao_master = json.load(f)

    return hexagram_master, yao_master


def judge_hexagram(case: Dict, hexagram_master: Dict) -> int:
    """
    事例の内容から64卦を判定する

    判定優先度:
    1. pattern_type からの候補
    2. before_state / after_state からの候補
    3. outcome による重み付け
    4. story_summary のキーワードマッチ
    """
    pattern_type = case.get("pattern_type", "")
    outcome = case.get("outcome", "Mixed")
    before_state = case.get("before_state", "")
    after_state = case.get("after_state", "")
    story = case.get("story_summary", "")

    # 候補卦をスコア付きで収集
    candidates: Dict[int, float] = {}

    # 1. pattern_type からの候補（最優先）
    if pattern_type in PATTERN_TO_HEXAGRAMS:
        for i, hex_id in enumerate(PATTERN_TO_HEXAGRAMS[pattern_type]):
            # 優先度に応じてスコア（先頭ほど高い）
            candidates[hex_id] = candidates.get(hex_id, 0) + (5 - i * 0.5)

    # 2. before_state からの候補
    if before_state in BEFORE_STATE_HINTS:
        for hex_id in BEFORE_STATE_HINTS[before_state]:
            candidates[hex_id] = candidates.get(hex_id, 0) + 2

    # 3. after_state からの候補
    if after_state in AFTER_STATE_HINTS:
        for hex_id in AFTER_STATE_HINTS[after_state]:
            candidates[hex_id] = candidates.get(hex_id, 0) + 2

    # 4. outcome による重み調整
    if outcome in OUTCOME_HEXAGRAM_WEIGHTS:
        for hex_id, weight in OUTCOME_HEXAGRAM_WEIGHTS[outcome].items():
            if hex_id in candidates:
                candidates[hex_id] *= weight

    # 5. story_summary からキーワードマッチ
    for hex_id_str, hex_data in hexagram_master.items():
        hex_id = int(hex_id_str)
        keyword = hex_data.get("keyword", "")
        meaning = hex_data.get("meaning", "")

        # キーワードがstoryに含まれるか
        keywords = keyword.split("・")
        for kw in keywords:
            if kw and kw in story:
                candidates[hex_id] = candidates.get(hex_id, 0) + 1.5

        # meaningの一部がstoryに含まれるか
        if len(meaning) > 4:
            for phrase in [meaning[i:i+4] for i in range(0, len(meaning)-3, 2)]:
                if phrase in story:
                    candidates[hex_id] = candidates.get(hex_id, 0) + 0.5

    # スコアが高い順にソート
    if candidates:
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        # 上位3つの中からランダムに選択（多様性確保）
        top_candidates = sorted_candidates[:min(3, len(sorted_candidates))]
        weights = [c[1] for c in top_candidates]
        total = sum(weights)
        weights = [w/total for w in weights]

        # 重み付きランダム選択
        r = random.random()
        cumulative = 0
        for (hex_id, score), weight in zip(top_candidates, weights):
            cumulative += weight
            if r < cumulative:
                return hex_id
        return top_candidates[0][0]

    # フォールバック: outcomeベースでデフォルト卦を返す
    default_hexagrams = {
        "Success": 11,  # 地天泰
        "Failure": 12,  # 天地否
        "Mixed": 63,    # 水火既済
    }
    return default_hexagrams.get(outcome, 11)


def judge_yao_position(case: Dict, hexagram_id: int, yao_master: Dict) -> int:
    """
    事例の内容から爻位（1-6）を判定する

    判定優先度:
    1. pattern_type からの傾向
    2. outcome による調整
    3. story_summary のキーワードマッチ
    """
    pattern_type = case.get("pattern_type", "")
    outcome = case.get("outcome", "Mixed")
    story = case.get("story_summary", "")

    # 爻位のスコア
    yao_scores: Dict[int, float] = {i: 1.0 for i in range(1, 7)}

    # 1. pattern_type からの傾向
    if pattern_type in PATTERN_YAO_TENDENCY:
        for yao in PATTERN_YAO_TENDENCY[pattern_type]:
            yao_scores[yao] += 2.0

    # 2. outcome による調整
    if outcome in OUTCOME_YAO_ADJUSTMENT:
        for yao, weight in OUTCOME_YAO_ADJUSTMENT[outcome].items():
            yao_scores[yao] *= weight

    # 3. story_summary のキーワードマッチ
    for yao, keywords in YAO_KEYWORDS.items():
        for kw in keywords:
            if kw in story:
                yao_scores[yao] += 1.5

    # 4. 卦固有の爻位意味からマッチ（yao_master参照）
    hex_str = str(hexagram_id)
    if hex_str in yao_master:
        yao_data = yao_master[hex_str].get("yao", {})
        for yao_str, yao_info in yao_data.items():
            yao_num = int(yao_str)
            modern = yao_info.get("modern", "")
            sns = yao_info.get("sns_style", "")

            # modernの一部がstoryに含まれるか
            for phrase in modern.split():
                if len(phrase) >= 2 and phrase in story:
                    yao_scores[yao_num] += 1.0

    # スコアが高い順にソート
    sorted_yao = sorted(yao_scores.items(), key=lambda x: x[1], reverse=True)

    # 上位2つの中から重み付きランダム選択
    top_yao = sorted_yao[:2]
    weights = [y[1] for y in top_yao]
    total = sum(weights)
    weights = [w/total for w in weights]

    r = random.random()
    cumulative = 0
    for (yao, score), weight in zip(top_yao, weights):
        cumulative += weight
        if r < cumulative:
            return yao
    return top_yao[0][0]


def process_case(case: Dict, hexagram_master: Dict, yao_master: Dict) -> Dict:
    """
    1件の事例を処理し、hexagram_idとyao_positionを判定・更新する
    """
    # 64卦を判定
    hexagram_id = judge_hexagram(case, hexagram_master)

    # 爻位を判定
    yao_position = judge_yao_position(case, hexagram_id, yao_master)

    # 卦名を取得
    hex_str = str(hexagram_id)
    hexagram_name = hexagram_master.get(hex_str, {}).get("name", f"卦{hexagram_id}")

    # 更新
    case["hexagram_id"] = hexagram_id
    case["hexagram_name"] = hexagram_name
    case["changing_lines_2"] = [yao_position]

    return case


def process_all_cases(
    input_path: Path,
    output_path: Path,
    hexagram_master: Dict,
    yao_master: Dict,
    backup: bool = True,
    dry_run: bool = False
) -> Dict:
    """
    全事例を処理してhexagram_idとyao_positionを再判定する
    """
    stats = {
        "total": 0,
        "processed": 0,
        "hexagram_changed": 0,
        "yao_changed": 0,
        "errors": 0,
        "hexagram_distribution": {},
        "yao_distribution": {},
    }

    # バックアップ
    if backup and input_path.exists() and not dry_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = input_path.with_suffix(f".jsonl.bak_aijudge_{timestamp}")
        shutil.copy2(input_path, backup_path)
        print(f"バックアップ作成: {backup_path}")

    cases = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                case = json.loads(line)
                stats["total"] += 1

                old_hexagram = case.get("hexagram_id")
                old_yao = case.get("changing_lines_2", [None])[0] if case.get("changing_lines_2") else None

                # AI判定
                case = process_case(case, hexagram_master, yao_master)
                stats["processed"] += 1

                new_hexagram = case["hexagram_id"]
                new_yao = case["changing_lines_2"][0]

                # 変更カウント
                if old_hexagram != new_hexagram:
                    stats["hexagram_changed"] += 1
                if old_yao != new_yao:
                    stats["yao_changed"] += 1

                # 分布カウント
                stats["hexagram_distribution"][new_hexagram] = \
                    stats["hexagram_distribution"].get(new_hexagram, 0) + 1
                stats["yao_distribution"][new_yao] = \
                    stats["yao_distribution"].get(new_yao, 0) + 1

                cases.append(case)

            except json.JSONDecodeError as e:
                stats["errors"] += 1
                print(f"行 {line_num}: JSON解析エラー: {e}")
            except Exception as e:
                stats["errors"] += 1
                print(f"行 {line_num}: 処理エラー: {e}")

    # 出力
    if not dry_run:
        with open(output_path, "w", encoding="utf-8") as f:
            for case in cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")

    return stats


def print_distribution_report(stats: Dict):
    """分布レポートを表示"""
    print("\n" + "=" * 70)
    print("AI判定結果レポート")
    print("=" * 70)

    print(f"\n処理件数: {stats['total']:,}")
    print(f"成功: {stats['processed']:,}")
    print(f"hexagram_id変更: {stats['hexagram_changed']:,}")
    print(f"yao_position変更: {stats['yao_changed']:,}")
    print(f"エラー: {stats['errors']:,}")

    # 64卦分布
    print("\n--- 64卦分布（上位20件）---")
    hex_dist = sorted(stats["hexagram_distribution"].items(), key=lambda x: x[1], reverse=True)
    for hex_id, count in hex_dist[:20]:
        pct = count / stats["processed"] * 100 if stats["processed"] > 0 else 0
        print(f"  卦{hex_id:02d}: {count:5,} ({pct:5.2f}%)")

    # 384セル（64卦×6爻）カバレッジ
    covered_cells = set()
    # TODO: 実際の分布からカバレッジを計算

    # 爻位分布
    print("\n--- 爻位分布 ---")
    for yao in range(1, 7):
        count = stats["yao_distribution"].get(yao, 0)
        pct = count / stats["processed"] * 100 if stats["processed"] > 0 else 0
        print(f"  {yao}爻: {count:5,} ({pct:5.2f}%)")

    print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AI判定による64卦+爻位の再計算"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/raw/cases.jsonl",
        help="入力ファイルパス"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="出力ファイルパス（デフォルト: 入力と同じ）"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="バックアップを作成しない"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実際には書き込まず、統計のみ表示"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="乱数シード（再現性のため）"
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / (args.output or args.input)

    print("=" * 70)
    print("AI判定スクリプト: 64卦 + 爻位の同時判定")
    print("=" * 70)
    print(f"入力: {input_path}")
    print(f"出力: {output_path}")
    print(f"バックアップ: {'なし' if args.no_backup else 'あり'}")
    print(f"モード: {'ドライラン' if args.dry_run else '実行'}")
    if args.seed is not None:
        print(f"乱数シード: {args.seed}")
    print()

    if not input_path.exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_path}")
        return

    # リファレンスデータ読み込み
    print("リファレンスデータ読み込み中...")
    hexagram_master, yao_master = load_reference_data()
    print(f"  64卦マスタ: {len(hexagram_master)}件")
    print(f"  爻マスタ: {len(yao_master)}件")

    # 処理実行
    print("\n処理中...")
    stats = process_all_cases(
        input_path,
        output_path,
        hexagram_master,
        yao_master,
        backup=not args.no_backup,
        dry_run=args.dry_run
    )

    # レポート表示
    print_distribution_report(stats)

    if not args.dry_run:
        print(f"\n完了: {output_path} に保存しました")


if __name__ == "__main__":
    main()
