#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
易経相談システム（完全版）

ユーザーの悩みから「現在の卦」を診断し、
目標状態から「目標卦」を判定し、
変化ルートとアクションを提示する統合システム。

機能:
1. 自然言語パーサー（現在状況 + 目標状態）
2. 決定論的卦判定（再現性確保）
3. 類似事例検索（信頼性向上）
4. ルートナビ連携（未来からの逆算）
5. アクション提示（各ステップの行動指針）
6. フィードバック生成（構造化レスポンス）

使用例:
    python3 scripts/consultation_system.py --situation "上司と対立している" --goal "円満に解決したい"
    python3 scripts/consultation_system.py --interactive
"""

import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field, asdict
import re
import sys

# 同一ディレクトリのモジュールをインポート
sys.path.insert(0, str(Path(__file__).parent))
from route_navigator import RouteNavigator, ACTION_MEANINGS as ROUTE_ACTION_MEANINGS
from prediction_engine import PredictionEngine

# ==============================================================================
# パス設定
# ==============================================================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

CASES_PATH = PROJECT_ROOT / "data/raw/cases.jsonl"
HEXAGRAM_MASTER_PATH = PROJECT_ROOT / "data/hexagrams/hexagram_master.json"
YAO_MASTER_PATH = PROJECT_ROOT / "data/hexagrams/yao_master.json"
TRANSITION_MAP_PATH = PROJECT_ROOT / "data/hexagrams/transition_map.json"

# ==============================================================================
# データクラス定義
# ==============================================================================

@dataclass
class ParsedSituation:
    """パース済み状況データ"""
    raw_text: str
    keywords: List[str]
    inferred_state: str  # 安定・危機・混乱・成長・衰退・変革
    inferred_pattern: str  # pattern_type推定
    confidence: float
    # 複合状況対応
    secondary_state: Optional[str] = None  # 2番目に強い状態
    all_matched_categories: List[str] = field(default_factory=list)  # 全マッチカテゴリ

@dataclass
class ParsedGoal:
    """パース済み目標データ"""
    raw_text: str
    keywords: List[str]
    inferred_state: str
    confidence: float

@dataclass
class HexagramJudgment:
    """卦判定結果"""
    hexagram_id: int
    hexagram_name: str
    yao_position: int
    confidence: float
    reasoning: List[str]

@dataclass
class SimilarCase:
    """類似事例"""
    case_id: str
    target_name: str
    story_summary: str
    outcome: str
    similarity_score: float

@dataclass
class RouteStep:
    """変化ルートのステップ"""
    from_hexagram: str
    to_hexagram: str
    action: str
    action_meaning: str
    success_rate: float
    recommended_actions: List[str]

@dataclass
class AlternativeRoute:
    """代替ルート"""
    title: str  # "最短ルート", "高成功率ルート (閾値70%)" など
    steps: List[RouteStep]
    total_success_rate: float


@dataclass
class ConsultationResult:
    """相談結果（完全版）"""
    # 入力
    situation_text: str
    goal_text: str

    # 現在の診断
    current_hexagram: HexagramJudgment
    current_interpretation: Dict[str, str]
    current_yao_advice: Dict[str, str]

    # 目標
    goal_hexagram: HexagramJudgment
    goal_interpretation: Dict[str, str]

    # 変化ルート（メイン）
    route: List[RouteStep]
    total_steps: int
    total_success_rate: float

    # 類似事例
    similar_cases: List[SimilarCase]

    # アクション提示
    immediate_action: str
    step_by_step_actions: List[str]
    warnings: List[str]

    # 代替ルート（オプション）
    alternative_routes: List[AlternativeRoute] = field(default_factory=list)

# ==============================================================================
# 自然言語パーサー
# ==============================================================================

# 状況キーワード辞書
SITUATION_KEYWORDS = {
    "創業苦難": {
        "keywords": ["立ち上げ", "新規事業", "スタートアップ", "創業", "起業", "軌道に乗らない", "苦戦", "赤字", "資金繰り"],
        "state": "創業苦難",
        "patterns": ["Steady_Growth", "Crisis_Pivot", "Shock_Recovery"]
    },
    "危機": {
        "keywords": ["危機", "危険", "ピンチ", "追い込まれ", "崩壊", "倒産", "破綻", "失敗", "大損", "暴落"],
        "state": "危機・困難",
        "patterns": ["Adaptive_Survival", "Crisis_Pivot", "Shock_Recovery"]
    },
    "対立": {
        "keywords": ["対立", "争い", "衝突", "喧嘩", "訴訟", "紛争", "敵対", "競争", "ライバル", "反発"],
        "state": "混乱・不安定",
        "patterns": ["Internal_Conflict", "External_Threat"]
    },
    "停滞": {
        "keywords": ["停滞", "行き詰まり", "閉塞", "動けない", "進まない", "膠着", "沈滞", "低迷", "マンネリ"],
        "state": "停滞・閉塞",
        "patterns": ["Slow_Decline", "Quiet_Fade", "Legacy_Burden"]
    },
    "成長": {
        "keywords": ["成長", "発展", "拡大", "上昇", "好調", "成功", "躍進", "ブレイク", "急成長"],
        "state": "成長期・好調",
        "patterns": ["Steady_Growth", "Opportunity_Seized", "Bold_Leap"]
    },
    "変化": {
        "keywords": ["変化", "転職", "転機", "転換", "変革", "改革", "刷新", "ピボット", "方向転換"],
        "state": "変革期",
        "patterns": ["Crisis_Pivot", "Tech_Disruption", "Generational_Change"]
    },
    "迷い": {
        "keywords": ["迷い", "悩み", "迷っている", "どうすれば", "選択", "決断", "分岐", "岐路"],
        "state": "混乱・不安定",
        "patterns": ["Strategic_Patience", "Crisis_Pivot"]
    },
    "人間関係": {
        "keywords": ["人間関係", "上司", "部下", "同僚", "友人", "恋人", "家族", "パートナー", "チーム"],
        "state": "混乱・不安定",
        "patterns": ["Internal_Conflict", "Collaborative_Rise"]
    },
    "仕事": {
        "keywords": ["仕事", "キャリア", "転職", "昇進", "降格", "リストラ", "副業", "独立"],
        "state": "変革期",
        "patterns": ["Bold_Leap", "Crisis_Pivot", "Steady_Growth"]
    },
    "待機": {
        "keywords": ["待つ", "様子見", "時期を見", "タイミング", "準備中", "検討中"],
        "state": "待機・準備",
        "patterns": ["Strategic_Patience"]
    }
}

# 目標キーワード辞書
GOAL_KEYWORDS = {
    "安定": {
        "keywords": ["安定", "平和", "穏やか", "落ち着", "バランス", "調和", "円満", "安心"],
        "state": "安定・平和",
        "target_hexagrams": [11, 32, 2, 15, 53]  # 泰、恒、坤、謙、漸
    },
    "成功": {
        "keywords": ["成功", "勝利", "達成", "実現", "繁栄", "豊か", "大成功", "飛躍", "トップ"],
        "state": "V字回復・大成功",
        "target_hexagrams": [14, 55, 1, 35, 42]  # 大有、豊、乾、晋、益
    },
    "解決": {
        "keywords": ["解決", "解消", "解放", "脱出", "改善", "修復", "回復", "克服"],
        "state": "変質・新生",
        "target_hexagrams": [40, 24, 49, 50, 63]  # 解、復、革、鼎、既済
    },
    "成長": {
        "keywords": ["成長", "発展", "向上", "進歩", "スキルアップ", "レベルアップ", "能力向上"],
        "state": "緩やかな成長",
        "target_hexagrams": [53, 46, 42, 35, 57]  # 漸、升、益、晋、巽
    },
    "変化": {
        "keywords": ["変わりたい", "変化", "新しい", "転換", "チャレンジ", "挑戦", "リセット"],
        "state": "変質・新生",
        "target_hexagrams": [49, 24, 50, 51, 64]  # 革、復、鼎、震、未済
    }
}


# 同義語辞書（キーワード展開用）
SYNONYMS = {
    "競争": ["ライバル", "競合", "シェア争い"],
    "対立": ["衝突", "摩擦", "不和", "確執"],
    "危機": ["ピンチ", "窮地", "瀬戸際", "崖っぷち"],
    "成長": ["伸び", "上昇", "拡大", "躍進"],
    "停滞": ["低迷", "足踏み", "頭打ち", "伸び悩み"],
    "変化": ["転換", "シフト", "移行", "変革"],
    "迷い": ["悩み", "葛藤", "ジレンマ", "板挟み"],
}


def expand_synonyms(text: str) -> str:
    """テキスト内の同義語を展開（検出力向上）"""
    expanded = text
    for base_word, synonyms in SYNONYMS.items():
        for syn in synonyms:
            if syn in text and base_word not in text:
                expanded += f" {base_word}"
                break
    return expanded


def parse_situation(text: str) -> ParsedSituation:
    """
    ユーザーの状況テキストをパースして構造化データに変換

    改善点:
    - 同義語展開による検出力向上
    - カテゴリ別スコアリングで複合状況に対応
    - 2番目に強い状態も返す（secondary_state）
    """
    # 同義語展開
    expanded_text = expand_synonyms(text)

    matched_keywords = []
    category_scores: Dict[str, float] = {}

    # カテゴリ別にスコア計算
    for category, data in SITUATION_KEYWORDS.items():
        score = 0.0
        matched_in_category = []

        for kw in data["keywords"]:
            if kw in expanded_text:
                matched_in_category.append(kw)
                # キーワードの長さに応じた重み付け（長いキーワードはより具体的）
                weight = 1.0 + len(kw) * 0.1
                score += weight

        if matched_in_category:
            category_scores[category] = score
            matched_keywords.extend(matched_in_category)

    # 重複除去
    matched_keywords = list(dict.fromkeys(matched_keywords))

    # スコア順にソート
    sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    all_matched_categories = [cat for cat, _ in sorted_categories]

    # 最も優先度の高いカテゴリを選択
    if sorted_categories:
        primary_category = sorted_categories[0][0]
        data = SITUATION_KEYWORDS[primary_category]
        inferred_state = data["state"]
        inferred_pattern = data["patterns"][0] if data["patterns"] else "Strategic_Patience"

        # 信頼度：スコアとマッチ数に基づく
        primary_score = sorted_categories[0][1]
        confidence = min(0.4 + primary_score * 0.1 + len(matched_keywords) * 0.05, 0.95)

        # 2番目に強い状態
        secondary_state = None
        if len(sorted_categories) >= 2:
            secondary_category = sorted_categories[1][0]
            secondary_data = SITUATION_KEYWORDS[secondary_category]
            # 2番目のスコアが1番目の50%以上なら有意
            if sorted_categories[1][1] >= sorted_categories[0][1] * 0.5:
                secondary_state = secondary_data["state"]
    else:
        inferred_state = "混乱・不安定"
        inferred_pattern = "Strategic_Patience"
        confidence = 0.3
        secondary_state = None

    return ParsedSituation(
        raw_text=text,
        keywords=matched_keywords,
        inferred_state=inferred_state,
        inferred_pattern=inferred_pattern,
        confidence=confidence,
        secondary_state=secondary_state,
        all_matched_categories=all_matched_categories
    )


def parse_goal(text: str) -> ParsedGoal:
    """
    ユーザーの目標テキストをパースして構造化データに変換
    """
    matched_keywords = []
    matched_categories = []

    for category, data in GOAL_KEYWORDS.items():
        for kw in data["keywords"]:
            if kw in text:
                matched_keywords.append(kw)
                if category not in matched_categories:
                    matched_categories.append(category)

    if matched_categories:
        primary_category = matched_categories[0]
        data = GOAL_KEYWORDS[primary_category]
        inferred_state = data["state"]
        confidence = min(0.5 + len(matched_keywords) * 0.1, 0.9)
    else:
        inferred_state = "安定・平和"
        confidence = 0.3

    return ParsedGoal(
        raw_text=text,
        keywords=matched_keywords,
        inferred_state=inferred_state,
        confidence=confidence
    )


# ==============================================================================
# 決定論的卦判定
# ==============================================================================

# 状態→卦のマッピング（決定論的）
STATE_TO_HEXAGRAM: Dict[str, List[Tuple[int, float]]] = {
    "創業苦難": [(3, 0.95), (4, 0.85), (39, 0.75), (47, 0.7), (29, 0.65)],  # 水雷屯が最優先
    "危機・困難": [(29, 0.9), (47, 0.85), (39, 0.8), (3, 0.75), (36, 0.7)],
    "混乱・不安定": [(6, 0.85), (38, 0.8), (64, 0.75), (59, 0.7), (4, 0.65)],
    "停滞・閉塞": [(12, 0.9), (52, 0.85), (23, 0.8), (33, 0.75), (62, 0.7)],
    "成長期・好調": [(11, 0.9), (14, 0.85), (35, 0.8), (42, 0.75), (55, 0.7)],
    "変革期": [(49, 0.9), (51, 0.85), (18, 0.8), (50, 0.75), (24, 0.7)],
    "安定・平和": [(11, 0.9), (32, 0.85), (53, 0.8), (15, 0.75), (2, 0.7)],
    "V字回復・大成功": [(14, 0.9), (55, 0.85), (1, 0.8), (35, 0.75), (42, 0.7)],
    "変質・新生": [(49, 0.9), (24, 0.85), (50, 0.8), (40, 0.75), (64, 0.7)],
    "緩やかな成長": [(53, 0.9), (46, 0.85), (57, 0.8), (32, 0.75), (11, 0.7)],
    "待機・準備": [(5, 0.9), (20, 0.85), (33, 0.8), (52, 0.75), (4, 0.7)],  # 水天需が最優先
}

# パターン→卦の微調整
PATTERN_HEXAGRAM_ADJUSTMENT: Dict[str, Dict[int, float]] = {
    "Internal_Conflict": {6: 1.3, 38: 1.2, 47: 1.1},
    "External_Threat": {7: 1.3, 6: 1.2, 33: 1.1},
    "Adaptive_Survival": {29: 1.3, 39: 1.2, 47: 1.1},
    "Crisis_Pivot": {49: 1.3, 18: 1.2, 64: 1.1},
    "Shock_Recovery": {51: 1.3, 40: 1.2, 24: 1.1},
    "Steady_Growth": {11: 1.3, 53: 1.2, 46: 1.1},
    "Strategic_Patience": {5: 1.3, 20: 1.2, 33: 1.1},
    "Opportunity_Seized": {42: 1.3, 35: 1.2, 14: 1.1},
    "Collaborative_Rise": {8: 1.3, 13: 1.2, 45: 1.1},
    "Bold_Leap": {1: 1.3, 34: 1.2, 43: 1.1},
    "Slow_Decline": {23: 1.3, 12: 1.2, 33: 1.1},
    "Quiet_Fade": {52: 1.3, 62: 1.2, 15: 1.1},
    "Legacy_Burden": {18: 1.3, 23: 1.2, 36: 1.1},
    "Tech_Disruption": {49: 1.3, 50: 1.2, 64: 1.1},
    "Generational_Change": {18: 1.3, 50: 1.2, 27: 1.1},
}

# キーワード→卦の直接マッピング
KEYWORD_TO_HEXAGRAM: Dict[str, int] = {
    "対立": 6, "争い": 6, "訴訟": 6,
    "危機": 29, "危険": 29, "困難": 29,
    "停滞": 12, "閉塞": 12, "行き詰まり": 47,
    "変化": 49, "変革": 49, "転換": 49,
    "成長": 11, "発展": 35, "上昇": 46,
    "衰退": 23, "没落": 23, "崩壊": 23,
    "迷い": 4, "未熟": 4, "学び": 4,
    "待機": 5, "準備": 5, "タイミング": 5,
    "協力": 8, "連携": 8, "チーム": 45,
    "決断": 43, "断行": 43, "踏み切る": 43,
    "謙虚": 15, "控えめ": 15, "低姿勢": 15,
    "喜び": 58, "楽しい": 58, "交流": 58,
    "信頼": 61, "誠実": 61, "真心": 61,
}

# 爻位判定用キーワード（決定論的）
YAO_KEYWORDS_DETERMINISTIC: Dict[int, List[str]] = {
    1: ["始まり", "準備", "これから", "初期", "着手", "開始", "立ち上げ", "創業", "新規", "スタート"],
    2: ["成長", "認められ", "進展", "発展", "拡大", "認知", "定着", "軌道に乗"],
    3: ["困難", "分岐", "過渡", "試練", "危機", "転機", "岐路", "迷い", "選択", "決断を迫られ"],
    4: ["飛躍", "転換", "決断", "躍進", "ブレイク", "変革", "大転換", "勝負", "M&A", "買収"],
    5: ["頂点", "成功", "最盛", "リーダー", "達成", "絶頂", "トップ", "支配", "完成"],
    6: ["終結", "過度", "転落", "衰退", "終わり", "引退", "撤退", "行き過ぎ", "崩壊", "解散"],
}


def judge_hexagram_deterministic(
    parsed: ParsedSituation,
    hexagram_master: Dict
) -> HexagramJudgment:
    """
    決定論的に卦を判定する（再現性確保）

    判定優先度:
    1. キーワード直接マッチ（最優先）
    2. 状態ベースの候補選択
    3. 複合状況（secondary_state）の考慮
    4. パターンによる微調整
    """
    reasoning = []
    scores: Dict[int, float] = {}

    # 1. キーワード直接マッチ
    for kw in parsed.keywords:
        if kw in KEYWORD_TO_HEXAGRAM:
            hex_id = KEYWORD_TO_HEXAGRAM[kw]
            scores[hex_id] = scores.get(hex_id, 0) + 10.0
            reasoning.append(f"キーワード「{kw}」→ 卦{hex_id}")

    # 2. 状態ベースの候補
    if parsed.inferred_state in STATE_TO_HEXAGRAM:
        for hex_id, base_score in STATE_TO_HEXAGRAM[parsed.inferred_state]:
            scores[hex_id] = scores.get(hex_id, 0) + base_score * 5
        reasoning.append(f"状態「{parsed.inferred_state}」から候補選択")

    # 3. 複合状況：secondary_stateがある場合は追加スコア
    if parsed.secondary_state and parsed.secondary_state in STATE_TO_HEXAGRAM:
        for hex_id, base_score in STATE_TO_HEXAGRAM[parsed.secondary_state]:
            # 副次的状態は主状態の30%の重みで加算
            scores[hex_id] = scores.get(hex_id, 0) + base_score * 1.5
        reasoning.append(f"複合状況「{parsed.secondary_state}」も考慮")

    # 4. パターンによる微調整
    if parsed.inferred_pattern in PATTERN_HEXAGRAM_ADJUSTMENT:
        for hex_id, multiplier in PATTERN_HEXAGRAM_ADJUSTMENT[parsed.inferred_pattern].items():
            if hex_id in scores:
                scores[hex_id] *= multiplier
        reasoning.append(f"パターン「{parsed.inferred_pattern}」で調整")

    # 最高スコアの卦を選択（決定論的）
    if scores:
        best_hex_id = max(scores.keys(), key=lambda k: scores[k])
        confidence = min(scores[best_hex_id] / 20.0, 0.95)
    else:
        # フォールバック
        best_hex_id = 64  # 未済（まだ終わらない）
        confidence = 0.3
        reasoning.append("マッチなし → 未済（64番）をデフォルト選択")

    # 卦名を取得
    hex_name = hexagram_master.get(str(best_hex_id), {}).get("name", f"卦{best_hex_id}")

    # 爻位を判定
    yao = judge_yao_deterministic(parsed, best_hex_id)

    return HexagramJudgment(
        hexagram_id=best_hex_id,
        hexagram_name=hex_name,
        yao_position=yao,
        confidence=confidence,
        reasoning=reasoning
    )


def judge_yao_deterministic(parsed: ParsedSituation, hexagram_id: int) -> int:
    """
    決定論的に爻位を判定
    """
    text = parsed.raw_text
    scores = {i: 0.0 for i in range(1, 7)}

    # キーワードマッチ
    for yao, keywords in YAO_KEYWORDS_DETERMINISTIC.items():
        for kw in keywords:
            if kw in text:
                scores[yao] += 3.0

    # 状態による傾向
    state_yao_tendency = {
        "危機・困難": [3, 4, 6],
        "混乱・不安定": [3, 4],
        "停滞・閉塞": [2, 3, 6],
        "成長期・好調": [2, 4, 5],
        "変革期": [3, 4, 5],
        "安定・平和": [2, 5],
    }

    if parsed.inferred_state in state_yao_tendency:
        for yao in state_yao_tendency[parsed.inferred_state]:
            scores[yao] += 2.0

    # 最高スコアの爻を選択
    best_yao = max(scores.keys(), key=lambda k: scores[k])

    # 同点の場合は3爻（分岐点）をデフォルト
    if scores[best_yao] == 0:
        best_yao = 3

    return best_yao


def judge_goal_hexagram(parsed_goal: ParsedGoal, hexagram_master: Dict) -> HexagramJudgment:
    """
    目標状態から目標卦を判定
    """
    reasoning = []
    scores: Dict[int, float] = {}

    # キーワードから候補を取得
    for category, data in GOAL_KEYWORDS.items():
        for kw in data["keywords"]:
            if kw in parsed_goal.raw_text:
                for hex_id in data["target_hexagrams"]:
                    scores[hex_id] = scores.get(hex_id, 0) + 5.0
                reasoning.append(f"目標キーワード「{kw}」→ {category}")
                break

    # 状態からの候補
    if parsed_goal.inferred_state in STATE_TO_HEXAGRAM:
        for hex_id, base_score in STATE_TO_HEXAGRAM[parsed_goal.inferred_state]:
            scores[hex_id] = scores.get(hex_id, 0) + base_score * 3
        reasoning.append(f"目標状態「{parsed_goal.inferred_state}」")

    # 最高スコアを選択
    if scores:
        best_hex_id = max(scores.keys(), key=lambda k: scores[k])
        confidence = min(scores[best_hex_id] / 15.0, 0.95)
    else:
        best_hex_id = 11  # 地天泰（安定・繁栄）
        confidence = 0.5
        reasoning.append("デフォルト → 地天泰（11番）")

    hex_name = hexagram_master.get(str(best_hex_id), {}).get("name", f"卦{best_hex_id}")

    return HexagramJudgment(
        hexagram_id=best_hex_id,
        hexagram_name=hex_name,
        yao_position=5,  # 目標は5爻（最盛期）
        confidence=confidence,
        reasoning=reasoning
    )


# ==============================================================================
# 類似事例検索（PredictionEngineのインデックス活用）
# ==============================================================================

# PredictionEngineインスタンス（キャッシュ）
_prediction_engine: Optional[PredictionEngine] = None


def get_prediction_engine() -> PredictionEngine:
    """PredictionEngineのシングルトンインスタンスを取得"""
    global _prediction_engine
    if _prediction_engine is None:
        _prediction_engine = PredictionEngine()
    return _prediction_engine


# 64卦ID→八卦（before_hex）のマッピング（hexagram_masterから自動生成）
_hexagram_to_trigram_cache: Optional[Dict[int, str]] = None


def get_hexagram_to_trigram() -> Dict[int, str]:
    """
    hexagram_master.jsonから64卦ID→上卦（八卦）のマッピングを自動生成

    上卦（upper_trigram）を使用する理由:
    - 上卦は外側・現れ・状況を表す
    - PredictionEngineのbefore_hexは状況の八卦を表す
    """
    global _hexagram_to_trigram_cache
    if _hexagram_to_trigram_cache is not None:
        return _hexagram_to_trigram_cache

    hexagram_master = load_hexagram_master()
    mapping = {}

    for hex_id_str, data in hexagram_master.items():
        try:
            hex_id = int(hex_id_str)
            # upper_trigramを使用（外側・状況を表す）
            upper = data.get("upper_trigram", "")
            if upper:
                mapping[hex_id] = upper
        except (ValueError, KeyError):
            continue

    # フォールバック：マッピングが空の場合はハードコード値を使用
    if not mapping:
        mapping = {
            1: "乾", 2: "坤", 3: "坎", 4: "艮", 5: "坎", 6: "乾", 7: "坤", 8: "坎",
            9: "巽", 10: "乾", 11: "坤", 12: "乾", 13: "乾", 14: "離", 15: "坤", 16: "震",
            17: "兌", 18: "艮", 19: "坤", 20: "巽", 21: "離", 22: "艮", 23: "艮", 24: "坤",
            25: "乾", 26: "艮", 27: "艮", 28: "兌", 29: "坎", 30: "離", 31: "兌", 32: "震",
            33: "乾", 34: "震", 35: "離", 36: "坤", 37: "巽", 38: "離", 39: "坎", 40: "震",
            41: "艮", 42: "巽", 43: "兌", 44: "乾", 45: "兌", 46: "坤", 47: "兌", 48: "坎",
            49: "兌", 50: "離", 51: "震", 52: "艮", 53: "巽", 54: "震", 55: "震", 56: "離",
            57: "巽", 58: "兌", 59: "巽", 60: "坎", 61: "巽", 62: "震", 63: "坎", 64: "離",
        }

    _hexagram_to_trigram_cache = mapping
    return mapping


def search_similar_cases(
    hexagram_id: int,
    yao_position: int,
    keywords: List[str],
    cases_path: Path = CASES_PATH,  # 後方互換性のため残す（使用しない）
    limit: int = 3
) -> List[SimilarCase]:
    """
    類似事例を検索（PredictionEngineのインデックス活用でO(1)検索）

    改善点:
    - 全件スキャン(O(n))からインデックス検索(O(1))に変更
    - PredictionEngineの構築済みインデックスを再利用
    """
    engine = get_prediction_engine()

    # hexagram_id → before_hex（八卦）に変換
    hexagram_to_trigram = get_hexagram_to_trigram()
    before_hex = hexagram_to_trigram.get(hexagram_id, "坎")

    # インデックスを使用して候補を絞り込み
    candidates = []

    # 1. before_hex一致の事例を取得（インデックス検索 O(1)）
    for action_type in ["攻める・挑戦", "守る・維持", "捨てる・撤退", "耐える・潜伏",
                        "対話・融合", "刷新・破壊", "逃げる・放置", "分散・スピンオフ"]:
        key = (before_hex, action_type)
        for case in engine.hex_action_index.get(key, []):
            # 重複チェック
            case_id = case.get("id", case.get("transition_id", "unknown"))
            if not any(c[0] == case_id for c in candidates):
                candidates.append((case_id, case))

    # 候補がなければ全事例から探す（フォールバック）
    if not candidates:
        for case in engine.cases[:100]:  # 最大100件に制限
            case_id = case.get("id", case.get("transition_id", "unknown"))
            if case.get("hexagram_id") == hexagram_id:
                candidates.append((case_id, case))

    # スコア計算とSimilarCase変換
    similar_cases = []
    for case_id, case in candidates[:50]:  # 最大50件でスコア計算
        score = 0.3  # 基本スコア

        # hexagram_idが一致
        if case.get("hexagram_id") == hexagram_id:
            score += 0.4

        # 爻が一致
        case_yao = case.get("changing_lines_2", [None])
        if case_yao and case_yao[0] == yao_position:
            score += 0.2

        # キーワードマッチ
        story = case.get("story_summary", "")
        for kw in keywords:
            if kw in story:
                score += 0.05

        # 信頼度が高いものを優先
        trust = case.get("trust_level", "unverified")
        if trust == "verified":
            score += 0.15
        elif trust == "plausible":
            score += 0.08

        similar_cases.append(SimilarCase(
            case_id=case_id,
            target_name=case.get("target_name", "不明"),
            story_summary=story[:200] + "..." if len(story) > 200 else story,
            outcome=case.get("outcome", "不明"),
            similarity_score=min(score, 1.0)
        ))

    # スコア順にソートして上位を返す
    similar_cases.sort(key=lambda x: x.similarity_score, reverse=True)
    return similar_cases[:limit]


# ==============================================================================
# ルートナビ連携
# ==============================================================================

def load_transition_map(path: Path = TRANSITION_MAP_PATH) -> Dict:
    """遷移マップを読み込む"""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("transitions", {})


def load_hexagram_master(path: Path = HEXAGRAM_MASTER_PATH) -> Dict:
    """卦マスターを読み込む"""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yao_master(path: Path = YAO_MASTER_PATH) -> Dict:
    """爻マスターを読み込む"""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# 八卦アクションの意味と具体的行動（route_navigator.pyから拡張）
ACTION_MEANINGS = {
    "乾": {
        "meaning": ROUTE_ACTION_MEANINGS.get("乾", "積極的にリードする"),
        "actions": [
            "自ら先頭に立って行動を起こす",
            "強い意志で決断を下す",
            "周囲を引っ張るリーダーシップを発揮"
        ]
    },
    "坤": {
        "meaning": ROUTE_ACTION_MEANINGS.get("坤", "受容し支える"),
        "actions": [
            "相手の意見をまず受け入れる",
            "サポート役に徹する",
            "柔軟に状況に適応する"
        ]
    },
    "震": {
        "meaning": ROUTE_ACTION_MEANINGS.get("震", "積極的に動く・衝撃を与える"),
        "actions": [
            "今すぐ行動を起こす",
            "変化のきっかけを作る",
            "勇気を持って一歩踏み出す"
        ]
    },
    "巽": {
        "meaning": ROUTE_ACTION_MEANINGS.get("巽", "柔軟に浸透する・交渉する"),
        "actions": [
            "時間をかけて少しずつ働きかける",
            "対話と交渉で解決を図る",
            "風のように柔らかく、しかし確実に影響を与える"
        ]
    },
    "坎": {
        "meaning": ROUTE_ACTION_MEANINGS.get("坎", "困難に耐える・深く掘り下げる"),
        "actions": [
            "困難から逃げずに向き合う",
            "問題の本質を深く分析する",
            "誠実さで信頼を勝ち取る"
        ]
    },
    "離": {
        "meaning": ROUTE_ACTION_MEANINGS.get("離", "明確化する・可視化する"),
        "actions": [
            "状況を明確に言語化する",
            "情報を可視化して共有する",
            "光を当てて問題を明らかにする"
        ]
    },
    "艮": {
        "meaning": ROUTE_ACTION_MEANINGS.get("艮", "止まる・守る・蓄積する"),
        "actions": [
            "一度立ち止まって考える",
            "無理に動かず現状を維持する",
            "内省と準備に時間を使う"
        ]
    },
    "兌": {
        "meaning": ROUTE_ACTION_MEANINGS.get("兌", "喜びを与える・対話する"),
        "actions": [
            "笑顔とポジティブな態度で接する",
            "相手を喜ばせることを考える",
            "楽しい交流の場を作る"
        ]
    },
}


# RouteNavigatorインスタンス（キャッシュ）
_route_navigator: Optional[RouteNavigator] = None


def get_route_navigator() -> RouteNavigator:
    """RouteNavigatorのシングルトンインスタンスを取得"""
    global _route_navigator
    if _route_navigator is None:
        _route_navigator = RouteNavigator()
    return _route_navigator


def find_route(
    from_hexagram: int,
    to_hexagram: int,
    transitions: Dict,  # 後方互換性のため残す（使用しない）
    hexagram_master: Dict,
    max_steps: int = 5,
    use_high_success: bool = True
) -> List[RouteStep]:
    """
    現在卦から目標卦への変化ルートを探索

    route_navigator.pyのRouteNavigatorクラスを活用:
    - 最短ルート（BFS）
    - 高成功率ルート（Dijkstra）
    """
    # 卦名の取得ヘルパー
    def get_hex_name(hex_id: int) -> str:
        return hexagram_master.get(str(hex_id), {}).get("name", f"卦{hex_id}")

    from_name = get_hex_name(from_hexagram)
    to_name = get_hex_name(to_hexagram)

    # 同じ卦の場合
    if from_hexagram == to_hexagram:
        return []

    # RouteNavigatorを使用
    navigator = get_route_navigator()

    # 高成功率ルートを優先、なければ最短ルート
    route = None
    if use_high_success:
        route = navigator.find_high_success_route(from_name, to_name, max_steps=max_steps)

    if route is None or not route.steps:
        route = navigator.find_shortest_route(from_name, to_name, max_steps=max_steps)

    if route is None or not route.steps:
        return []  # ルートが見つからない

    # RouteNavigatorのTransitionStepをRouteStepに変換
    result = []
    for step in route.steps:
        action = step.action
        action_data = ACTION_MEANINGS.get(action, {"meaning": "行動する", "actions": []})

        result.append(RouteStep(
            from_hexagram=navigator.normalizer.get_display_name(step.from_hex),
            to_hexagram=navigator.normalizer.get_display_name(step.to_hex),
            action=action,
            action_meaning=action_data["meaning"],
            success_rate=step.success_rate,
            recommended_actions=action_data.get("actions", [])
        ))

    return result


def find_alternative_routes(
    from_hexagram: int,
    to_hexagram: int,
    hexagram_master: Dict,
    max_routes: int = 3
) -> List[Tuple[str, List[RouteStep], float]]:
    """
    複数の代替ルートを探索

    Returns:
        List of (ルート名, ステップリスト, 総合成功率)
    """
    def get_hex_name(hex_id: int) -> str:
        return hexagram_master.get(str(hex_id), {}).get("name", f"卦{hex_id}")

    from_name = get_hex_name(from_hexagram)
    to_name = get_hex_name(to_hexagram)

    if from_hexagram == to_hexagram:
        return []

    navigator = get_route_navigator()
    routes = navigator.find_alternative_routes(from_name, to_name, max_routes=max_routes)

    result = []
    for title, route in routes:
        steps = []
        for step in route.steps:
            action = step.action
            action_data = ACTION_MEANINGS.get(action, {"meaning": "行動する", "actions": []})

            steps.append(RouteStep(
                from_hexagram=navigator.normalizer.get_display_name(step.from_hex),
                to_hexagram=navigator.normalizer.get_display_name(step.to_hex),
                action=action,
                action_meaning=action_data["meaning"],
                success_rate=step.success_rate,
                recommended_actions=action_data.get("actions", [])
            ))

        result.append((title, steps, route.total_success_rate))

    return result


# ==============================================================================
# フィードバック生成
# ==============================================================================

def generate_feedback(
    situation_text: str,
    goal_text: str,
    current_hex: HexagramJudgment,
    goal_hex: HexagramJudgment,
    route: List[RouteStep],
    similar_cases: List[SimilarCase],
    hexagram_master: Dict,
    yao_master: Dict,
    alternative_routes: Optional[List[Tuple[str, List[RouteStep], float]]] = None
) -> ConsultationResult:
    """
    統合的なフィードバックを生成
    """
    # 現在の卦の解釈
    current_hex_data = hexagram_master.get(str(current_hex.hexagram_id), {})
    current_interpretation = current_hex_data.get("interpretations", {})

    # 現在の爻のアドバイス
    current_yao_data = yao_master.get(str(current_hex.hexagram_id), {}).get("yao", {})
    current_yao_info = current_yao_data.get(str(current_hex.yao_position), {})
    current_yao_advice = {
        "classic": current_yao_info.get("classic", ""),
        "modern": current_yao_info.get("modern", ""),
        "sns_style": current_yao_info.get("sns_style", "")
    }

    # 目標の卦の解釈
    goal_hex_data = hexagram_master.get(str(goal_hex.hexagram_id), {})
    goal_interpretation = goal_hex_data.get("interpretations", {})

    # アクション生成
    if route:
        immediate_action = f"まず「{route[0].action}」の姿勢で。{route[0].action_meaning}"
        step_by_step_actions = []
        for i, step in enumerate(route):
            actions_str = "、".join(step.recommended_actions[:2]) if step.recommended_actions else step.action_meaning
            step_by_step_actions.append(
                f"ステップ{i+1}: {step.from_hexagram} → {step.to_hexagram} | {step.action}（{actions_str}）| 成功率{step.success_rate:.0%}"
            )
    else:
        immediate_action = "現状を維持しながら、機会を待つ"
        step_by_step_actions = ["直接のルートが見つかりませんでした。別の目標設定を検討してください。"]

    # 警告生成
    warnings = []
    if current_hex.hexagram_id in [29, 47, 39]:  # 危険卦
        warnings.append("現在、危険な状況にあります。慎重な行動を心がけてください。")
    if route and any(s.success_rate < 0.3 for s in route):
        warnings.append("ルート上に成功率の低いステップがあります。代替案も検討してください。")

    # 総合成功率
    total_success_rate = 1.0
    for step in route:
        total_success_rate *= step.success_rate

    # 代替ルートをAlternativeRouteオブジェクトに変換
    alt_route_objects = []
    if alternative_routes:
        for title, steps, success_rate in alternative_routes:
            # メインルートと同じルートはスキップ
            if steps and route and len(steps) == len(route):
                if all(s1.to_hexagram == s2.to_hexagram for s1, s2 in zip(steps, route)):
                    continue
            alt_route_objects.append(AlternativeRoute(
                title=title,
                steps=steps,
                total_success_rate=success_rate
            ))

    return ConsultationResult(
        situation_text=situation_text,
        goal_text=goal_text,
        current_hexagram=current_hex,
        current_interpretation=current_interpretation,
        current_yao_advice=current_yao_advice,
        goal_hexagram=goal_hex,
        goal_interpretation=goal_interpretation,
        route=route,
        total_steps=len(route),
        total_success_rate=total_success_rate,
        alternative_routes=alt_route_objects,
        similar_cases=similar_cases,
        immediate_action=immediate_action,
        step_by_step_actions=step_by_step_actions,
        warnings=warnings
    )


# ==============================================================================
# 出力フォーマット
# ==============================================================================

def format_result_text(result: ConsultationResult) -> str:
    """結果をテキスト形式でフォーマット"""
    lines = []

    lines.append("=" * 70)
    lines.append("易経相談システム - 診断結果")
    lines.append("=" * 70)

    lines.append(f"\n【あなたの状況】")
    lines.append(f"  {result.situation_text}")

    lines.append(f"\n【あなたの目標】")
    lines.append(f"  {result.goal_text}")

    lines.append(f"\n{'─' * 70}")
    lines.append("【現在の診断】")
    lines.append(f"  卦: {result.current_hexagram.hexagram_name}（{result.current_hexagram.hexagram_id}番）")
    lines.append(f"  爻: {result.current_hexagram.yao_position}爻")
    lines.append(f"  確信度: {result.current_hexagram.confidence:.0%}")

    lines.append(f"\n  判定理由:")
    for reason in result.current_hexagram.reasoning:
        lines.append(f"    - {reason}")

    if result.current_yao_advice.get("sns_style"):
        lines.append(f"\n  【爻辞からのメッセージ】")
        lines.append(f"    {result.current_yao_advice['sns_style']}")

    if result.current_interpretation:
        lines.append(f"\n  【卦の解釈】")
        for perspective, interp in list(result.current_interpretation.items())[:2]:
            lines.append(f"    {perspective}: {interp}")

    lines.append(f"\n{'─' * 70}")
    lines.append("【目標の卦】")
    lines.append(f"  卦: {result.goal_hexagram.hexagram_name}（{result.goal_hexagram.hexagram_id}番）")
    lines.append(f"  確信度: {result.goal_hexagram.confidence:.0%}")

    if result.goal_interpretation:
        lines.append(f"\n  【目標状態の解釈】")
        for perspective, interp in list(result.goal_interpretation.items())[:2]:
            lines.append(f"    {perspective}: {interp}")

    lines.append(f"\n{'─' * 70}")
    lines.append("【変化ルート】")
    lines.append(f"  {result.current_hexagram.hexagram_name} → {result.goal_hexagram.hexagram_name}")
    lines.append(f"  ステップ数: {result.total_steps}")
    lines.append(f"  総合成功率: {result.total_success_rate:.0%}")

    if result.step_by_step_actions:
        lines.append(f"\n  【ステップ詳細】")
        for action in result.step_by_step_actions:
            lines.append(f"    {action}")

    # 代替ルートの表示
    if result.alternative_routes:
        lines.append(f"\n  【代替ルート】")
        for alt_route in result.alternative_routes:
            lines.append(f"\n  ◆ {alt_route.title} - 総合成功率: {alt_route.total_success_rate:.0%}")
            if alt_route.steps:
                for i, step in enumerate(alt_route.steps):
                    lines.append(f"      {i+1}. {step.from_hexagram} → {step.to_hexagram} | {step.action}（{step.action_meaning}）| {step.success_rate:.0%}")
            else:
                lines.append(f"      （ルートなし）")

    lines.append(f"\n{'─' * 70}")
    lines.append("【今すぐやるべきこと】")
    lines.append(f"  {result.immediate_action}")

    if result.warnings:
        lines.append(f"\n【注意事項】")
        for warning in result.warnings:
            lines.append(f"  ⚠️ {warning}")

    if result.similar_cases:
        lines.append(f"\n{'─' * 70}")
        lines.append("【類似の過去事例】")
        for case in result.similar_cases:
            lines.append(f"\n  {case.target_name}")
            lines.append(f"    {case.story_summary}")
            lines.append(f"    結果: {case.outcome} (類似度: {case.similarity_score:.0%})")

    lines.append(f"\n{'=' * 70}")

    return "\n".join(lines)


def format_result_json(result: ConsultationResult) -> str:
    """結果をJSON形式でフォーマット"""
    def to_dict(obj):
        if hasattr(obj, '__dataclass_fields__'):
            return asdict(obj)
        return obj

    return json.dumps(asdict(result), ensure_ascii=False, indent=2, default=to_dict)


# ==============================================================================
# メイン処理
# ==============================================================================

def run_consultation(
    situation: str,
    goal: str,
    output_format: str = "text"
) -> str:
    """
    相談を実行
    """
    # データ読み込み
    hexagram_master = load_hexagram_master()
    yao_master = load_yao_master()
    transitions = load_transition_map()

    # 状況をパース
    parsed_situation = parse_situation(situation)
    parsed_goal = parse_goal(goal)

    # 卦を判定
    current_hex = judge_hexagram_deterministic(parsed_situation, hexagram_master)
    goal_hex = judge_goal_hexagram(parsed_goal, hexagram_master)

    # ルートを探索（メイン）
    route = find_route(
        current_hex.hexagram_id,
        goal_hex.hexagram_id,
        transitions,
        hexagram_master
    )

    # 代替ルートを探索
    alternative_routes = find_alternative_routes(
        current_hex.hexagram_id,
        goal_hex.hexagram_id,
        hexagram_master,
        max_routes=3
    )

    # 類似事例を検索
    similar_cases = search_similar_cases(
        current_hex.hexagram_id,
        current_hex.yao_position,
        parsed_situation.keywords
    )

    # フィードバック生成
    result = generate_feedback(
        situation,
        goal,
        current_hex,
        goal_hex,
        route,
        similar_cases,
        hexagram_master,
        yao_master,
        alternative_routes=alternative_routes
    )

    # 出力
    if output_format == "json":
        return format_result_json(result)
    else:
        return format_result_text(result)


def interactive_mode():
    """対話モード"""
    print("=" * 70)
    print("易経相談システム（対話モード）")
    print("=" * 70)
    print("あなたの状況と目標を教えてください。")
    print("終了するには 'quit' または 'exit' と入力してください。")
    print("=" * 70)

    while True:
        print("\n")
        situation = input("【あなたの状況】> ").strip()
        if situation.lower() in ["quit", "exit", "q"]:
            print("ご利用ありがとうございました。")
            break

        if not situation:
            print("状況を入力してください。")
            continue

        goal = input("【あなたの目標】> ").strip()
        if goal.lower() in ["quit", "exit", "q"]:
            print("ご利用ありがとうございました。")
            break

        if not goal:
            goal = "より良い状態になりたい"

        print("\n診断中...")
        result = run_consultation(situation, goal)
        print(result)


def main():
    parser = argparse.ArgumentParser(
        description="易経相談システム - 未来から逆算する変化ルートナビ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python3 scripts/consultation_system.py --situation "上司と対立している" --goal "円満に解決したい"
  python3 scripts/consultation_system.py --interactive
  python3 scripts/consultation_system.py -s "仕事で行き詰まっている" -g "成功したい" --json
        """
    )

    parser.add_argument(
        "--situation", "-s",
        help="現在の状況・悩み"
    )
    parser.add_argument(
        "--goal", "-g",
        help="目標・なりたい状態"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="対話モードで起動"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON形式で出力"
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.situation:
        goal = args.goal or "より良い状態になりたい"
        output_format = "json" if args.json else "text"
        result = run_consultation(args.situation, goal, output_format)
        print(result)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
