#!/usr/bin/env python3
"""
易経変化ロジックDB 予測エンジン v1.0

入力された状況から、予測される結果・類似事例・推奨アクションを出力する。

Usage:
    python3 scripts/prediction_engine.py --before-hex 坎 --action 攻める・挑戦 --scale company
    python3 scripts/prediction_engine.py --before-hex 離 --before-state 絶頂・慢心 --action 守る・維持 --scale individual
"""

import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict


# 八卦のマッピング（上卦×下卦 → 卦番号）
TRIGRAM_TO_HEXAGRAM = {
    ("乾", "乾"): 1, ("坤", "乾"): 11, ("震", "乾"): 34, ("巽", "乾"): 9,
    ("坎", "乾"): 5, ("離", "乾"): 14, ("艮", "乾"): 26, ("兌", "乾"): 43,
    ("乾", "坤"): 12, ("坤", "坤"): 2, ("震", "坤"): 16, ("巽", "坤"): 20,
    ("坎", "坤"): 8, ("離", "坤"): 35, ("艮", "坤"): 23, ("兌", "坤"): 45,
    ("乾", "震"): 25, ("坤", "震"): 24, ("震", "震"): 51, ("巽", "震"): 42,
    ("坎", "震"): 3, ("離", "震"): 21, ("艮", "震"): 27, ("兌", "震"): 17,
    ("乾", "巽"): 44, ("坤", "巽"): 46, ("震", "巽"): 32, ("巽", "巽"): 57,
    ("坎", "巽"): 48, ("離", "巽"): 50, ("艮", "巽"): 18, ("兌", "巽"): 28,
    ("乾", "坎"): 6, ("坤", "坎"): 7, ("震", "坎"): 40, ("巽", "坎"): 59,
    ("坎", "坎"): 29, ("離", "坎"): 64, ("艮", "坎"): 4, ("兌", "坎"): 47,
    ("乾", "離"): 13, ("坤", "離"): 36, ("震", "離"): 55, ("巽", "離"): 37,
    ("坎", "離"): 63, ("離", "離"): 30, ("艮", "離"): 22, ("兌", "離"): 49,
    ("乾", "艮"): 33, ("坤", "艮"): 15, ("震", "艮"): 62, ("巽", "艮"): 53,
    ("坎", "艮"): 39, ("離", "艮"): 56, ("艮", "艮"): 52, ("兌", "艮"): 31,
    ("乾", "兌"): 10, ("坤", "兌"): 19, ("震", "兌"): 54, ("巽", "兌"): 61,
    ("坎", "兌"): 60, ("離", "兌"): 38, ("艮", "兌"): 41, ("兌", "兌"): 58,
}

# 八卦リスト
TRIGRAMS = ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]

# before_state リスト
BEFORE_STATES = [
    "絶頂・慢心", "停滞・閉塞", "混乱・カオス", "成長痛",
    "どん底・危機", "安定・平和", "V字回復・大成功", "縮小安定・生存"
]

# action_type リスト
ACTION_TYPES = [
    "攻める・挑戦", "守る・維持", "捨てる・撤退", "耐える・潜伏",
    "対話・融合", "刷新・破壊", "逃げる・放置", "分散・スピンオフ"
]

# scale リスト
SCALES = ["company", "individual", "family", "country", "other"]


@dataclass
class SimilarCase:
    """類似事例"""
    transition_id: str
    target_name: str
    scale: str
    period: str
    story_summary: str
    outcome: str
    similarity_score: float
    match_factors: List[str] = field(default_factory=list)


@dataclass
class PredictionResult:
    """予測結果"""
    predicted_outcome: str  # Success, Failure, PartialSuccess, Mixed
    confidence: float  # 0.0-1.0
    success_rate: float
    failure_rate: float
    partial_rate: float
    mixed_rate: float
    sample_size: int
    similar_cases: List[SimilarCase]
    recommended_actions: List[Tuple[str, float, str]]  # (action, score, reason)
    warnings: List[str]
    hexagram_advice: Optional[str] = None
    yao_advice: Optional[str] = None
    input_summary: Dict[str, Any] = field(default_factory=dict)


class PredictionEngine:
    """易経変化予測エンジン"""

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        self.data_dir = Path(data_dir)

        # データ読み込み
        self.cases = self._load_cases()
        self.statistics = self._load_json("diagnostic/statistics_table.json")
        self.hexagram_master = self._load_json("hexagrams/hexagram_master.json")
        self.yao_384 = self._load_json("diagnostic/yao_384.json")
        self.failure_avoidance = self._load_json("diagnostic/failure_avoidance.json")

        # インデックス構築
        self._build_indices()

    def _load_json(self, relative_path: str) -> Dict:
        """JSONファイルを読み込み"""
        filepath = self.data_dir / relative_path
        if not filepath.exists():
            return {}
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_cases(self) -> List[Dict]:
        """cases.jsonlを読み込み"""
        cases = []
        cases_file = self.data_dir / "raw" / "cases.jsonl"
        if cases_file.exists():
            with open(cases_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        cases.append(json.loads(line))
        return cases

    def _build_indices(self):
        """検索用インデックスを構築"""
        # before_hex × action_type のインデックス
        self.hex_action_index = defaultdict(list)
        # before_hex × before_state × action_type のインデックス
        self.hex_state_action_index = defaultdict(list)
        # scale別インデックス
        self.scale_index = defaultdict(list)

        for case in self.cases:
            before_hex = case.get("before_hex", "")
            before_state = case.get("before_state", "")
            action_type = case.get("action_type", "")
            scale = case.get("scale", "")

            key1 = (before_hex, action_type)
            self.hex_action_index[key1].append(case)

            key2 = (before_hex, before_state, action_type)
            self.hex_state_action_index[key2].append(case)

            self.scale_index[scale].append(case)

    def get_hexagram_id(self, upper: str, lower: str) -> Optional[int]:
        """上卦・下卦から64卦IDを取得"""
        return TRIGRAM_TO_HEXAGRAM.get((upper, lower))

    def get_hexagram_from_single_trigram(self, trigram: str) -> int:
        """単一の八卦から重卦（同じ卦を重ねた）のIDを取得"""
        return TRIGRAM_TO_HEXAGRAM.get((trigram, trigram), 1)

    def get_statistics(self, before_hex: str, action_type: str,
                       before_state: Optional[str] = None) -> Dict[str, float]:
        """統計データを取得"""
        result = {
            "success_rate": 50.0,
            "failure_rate": 25.0,
            "partial_rate": 15.0,
            "mixed_rate": 10.0,
            "total": 0
        }

        # 優先順位1: by_before_state_action
        if before_state:
            state_action = self.statistics.get("by_before_state_action", {})
            if before_state in state_action:
                if action_type in state_action[before_state]:
                    stats = state_action[before_state][action_type]
                    return {
                        "success_rate": stats.get("success_rate", 50.0),
                        "failure_rate": stats.get("failure_rate", 25.0),
                        "partial_rate": stats.get("partial_rate", 15.0),
                        "mixed_rate": stats.get("mixed_rate", 10.0),
                        "total": stats.get("total", 0)
                    }

        # 優先順位2: by_before_hex_action
        hex_action = self.statistics.get("by_before_hex_action", {})
        if before_hex in hex_action:
            if action_type in hex_action[before_hex]:
                stats = hex_action[before_hex][action_type]
                return {
                    "success_rate": stats.get("success_rate", 50.0),
                    "failure_rate": stats.get("failure_rate", 25.0),
                    "partial_rate": stats.get("partial_rate", 15.0),
                    "mixed_rate": stats.get("mixed_rate", 10.0),
                    "total": stats.get("total", 0)
                }

        return result

    def find_similar_cases(self, before_hex: str, action_type: str,
                           scale: str, before_state: Optional[str] = None,
                           limit: int = 3) -> List[SimilarCase]:
        """類似事例を検索"""
        candidates = []

        # 完全一致（before_hex + before_state + action_type + scale）
        if before_state:
            key = (before_hex, before_state, action_type)
            for case in self.hex_state_action_index.get(key, []):
                if case.get("scale") == scale:
                    candidates.append((case, 1.0, ["before_hex", "before_state", "action_type", "scale"]))
                else:
                    candidates.append((case, 0.8, ["before_hex", "before_state", "action_type"]))

        # before_hex + action_type + scale
        key = (before_hex, action_type)
        for case in self.hex_action_index.get(key, []):
            if case.get("scale") == scale:
                already = any(c[0].get("transition_id") == case.get("transition_id") for c in candidates)
                if not already:
                    candidates.append((case, 0.7, ["before_hex", "action_type", "scale"]))
            else:
                already = any(c[0].get("transition_id") == case.get("transition_id") for c in candidates)
                if not already:
                    candidates.append((case, 0.5, ["before_hex", "action_type"]))

        # scaleのみ一致
        for case in self.scale_index.get(scale, [])[:50]:  # 最大50件まで
            if case.get("action_type") == action_type:
                already = any(c[0].get("transition_id") == case.get("transition_id") for c in candidates)
                if not already:
                    candidates.append((case, 0.3, ["action_type", "scale"]))

        # スコア順にソート
        candidates.sort(key=lambda x: x[1], reverse=True)

        # SimilarCaseオブジェクトに変換
        similar_cases = []
        for case, score, factors in candidates[:limit]:
            similar_cases.append(SimilarCase(
                transition_id=case.get("transition_id", ""),
                target_name=case.get("target_name", ""),
                scale=case.get("scale", ""),
                period=case.get("period", ""),
                story_summary=case.get("story_summary", "")[:100] + "...",
                outcome=case.get("outcome", ""),
                similarity_score=score,
                match_factors=factors
            ))

        return similar_cases

    def get_hexagram_advice(self, before_hex: str, scale: str) -> Optional[str]:
        """卦のアドバイスを取得"""
        hex_id = self.get_hexagram_from_single_trigram(before_hex)
        hex_data = self.hexagram_master.get(str(hex_id), {})

        if hex_data:
            interpretations = hex_data.get("interpretations", {})
            return interpretations.get(scale, hex_data.get("meaning", ""))

        return None

    def get_yao_advice(self, hexagram_id: int, yao_position: int = 3) -> Optional[str]:
        """爻のアドバイスを取得（デフォルトは三爻=転換点）"""
        if yao_position < 1 or yao_position > 6:
            yao_position = 3

        yao_key = f"{hexagram_id:02d}-{yao_position}"
        yao_data = self.yao_384.get("yao", {}).get(yao_key, {})

        if yao_data:
            return yao_data.get("advice", "")

        return None

    def get_recommended_actions(self, before_hex: str, before_state: Optional[str] = None,
                                 current_action: Optional[str] = None) -> List[Tuple[str, float, str]]:
        """推奨アクションを取得"""
        recommendations = []

        for action in ACTION_TYPES:
            stats = self.get_statistics(before_hex, action, before_state)
            success_rate = stats.get("success_rate", 50.0)
            failure_rate = stats.get("failure_rate", 25.0)
            total = stats.get("total", 0)

            # スコア計算: 成功率 - (失敗率 * 0.5) + サンプルサイズボーナス
            score = success_rate - (failure_rate * 0.5)
            if total > 10:
                score += 5  # サンプルが十分ある場合ボーナス

            # 理由の生成
            reasons = []
            if success_rate >= 70:
                reasons.append(f"成功率{success_rate:.0f}%")
            elif success_rate >= 50:
                reasons.append(f"成功率{success_rate:.0f}%（中程度）")
            else:
                reasons.append(f"成功率{success_rate:.0f}%（低め）")

            if failure_rate >= 30:
                reasons.append(f"失敗率{failure_rate:.0f}%に注意")

            if total > 0:
                reasons.append(f"事例{total}件")

            reason = "、".join(reasons) if reasons else "標準的"
            recommendations.append((action, score, reason))

        # スコア順にソート
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations

    def get_warnings(self, before_hex: str, action_type: str,
                     before_state: Optional[str] = None) -> List[str]:
        """警告メッセージを生成"""
        warnings = []

        stats = self.get_statistics(before_hex, action_type, before_state)

        # 失敗率が高い場合の警告
        if stats.get("failure_rate", 0) >= 40:
            warnings.append(f"この組み合わせは失敗率{stats['failure_rate']:.0f}%と高めです。慎重に検討してください。")

        # 特定の危険パターンの警告
        high_risk = self.failure_avoidance.get("high_risk_state_action", [])
        for risk in high_risk:
            if risk.get("before_state") == before_state and risk.get("action_type") == action_type:
                warnings.append(risk.get("warning", ""))

        # 八卦別の警告
        hex_warnings = {
            "乾": "絶頂時の攻めは慢心を招きやすい",
            "坤": "受け身すぎると機会を逃す",
            "震": "衝動的な行動は失敗のもと",
            "巽": "優柔不断は状況を悪化させる",
            "坎": "試練の中で無理は禁物",
            "離": "派手な動きは反動を招く",
            "艮": "止まりすぎると時機を逃す",
            "兌": "楽観しすぎると足元をすくわれる"
        }

        if before_hex in hex_warnings:
            # 該当する行動タイプと組み合わせて警告
            if before_hex == "乾" and action_type == "攻める・挑戦":
                warnings.append(hex_warnings["乾"])
            elif before_hex == "坤" and action_type in ["耐える・潜伏", "守る・維持"]:
                warnings.append(hex_warnings["坤"])
            elif before_hex == "震" and action_type == "攻める・挑戦":
                warnings.append(hex_warnings["震"])
            elif before_hex == "巽" and action_type in ["守る・維持", "逃げる・放置"]:
                warnings.append(hex_warnings["巽"])
            elif before_hex == "坎" and action_type == "攻める・挑戦":
                if stats.get("total", 0) > 0 and stats.get("success_rate", 0) < 80:
                    warnings.append("試練の中での挑戦は成功率が変動しやすい。状況をよく見極めて。")

        return warnings

    def predict(self, before_hex: str, action_type: str, scale: str,
                before_state: Optional[str] = None) -> PredictionResult:
        """予測を実行"""
        # 統計データ取得
        stats = self.get_statistics(before_hex, action_type, before_state)

        # 予測結果の決定
        success_rate = stats.get("success_rate", 50.0)
        failure_rate = stats.get("failure_rate", 25.0)
        partial_rate = stats.get("partial_rate", 15.0)
        mixed_rate = stats.get("mixed_rate", 10.0)

        # 最も確率の高い結果を予測
        rates = {
            "Success": success_rate,
            "Failure": failure_rate,
            "PartialSuccess": partial_rate,
            "Mixed": mixed_rate
        }
        predicted_outcome = max(rates, key=rates.get)

        # 信頼度計算
        max_rate = max(rates.values())
        total_samples = stats.get("total", 0)

        # 信頼度 = (最大確率 / 100) * サンプルサイズ補正
        if total_samples >= 50:
            sample_factor = 1.0
        elif total_samples >= 20:
            sample_factor = 0.8
        elif total_samples >= 10:
            sample_factor = 0.6
        elif total_samples >= 5:
            sample_factor = 0.4
        else:
            sample_factor = 0.2

        confidence = (max_rate / 100) * sample_factor
        confidence = min(confidence, 0.95)  # 最大95%

        # 類似事例検索
        similar_cases = self.find_similar_cases(before_hex, action_type, scale, before_state)

        # 推奨アクション
        recommended_actions = self.get_recommended_actions(before_hex, before_state, action_type)

        # 警告
        warnings = self.get_warnings(before_hex, action_type, before_state)

        # 卦・爻のアドバイス
        hexagram_advice = self.get_hexagram_advice(before_hex, scale)
        hex_id = self.get_hexagram_from_single_trigram(before_hex)
        yao_advice = self.get_yao_advice(hex_id, 3)  # 三爻（転換点）

        return PredictionResult(
            predicted_outcome=predicted_outcome,
            confidence=confidence,
            success_rate=success_rate,
            failure_rate=failure_rate,
            partial_rate=partial_rate,
            mixed_rate=mixed_rate,
            sample_size=total_samples,
            similar_cases=similar_cases,
            recommended_actions=recommended_actions[:5],  # 上位5つ
            warnings=warnings,
            hexagram_advice=hexagram_advice,
            yao_advice=yao_advice,
            input_summary={
                "before_hex": before_hex,
                "action_type": action_type,
                "scale": scale,
                "before_state": before_state
            }
        )


def format_prediction_result(result: PredictionResult) -> str:
    """予測結果をフォーマット"""
    lines = []
    lines.append("=" * 60)
    lines.append("【易経変化予測エンジン - 予測結果】")
    lines.append("=" * 60)
    lines.append("")

    # 入力サマリー
    lines.append("[入力条件]")
    lines.append(f"  before_hex（八卦）: {result.input_summary.get('before_hex', '-')}")
    if result.input_summary.get('before_state'):
        lines.append(f"  before_state（状態）: {result.input_summary.get('before_state')}")
    lines.append(f"  action_type（アクション）: {result.input_summary.get('action_type', '-')}")
    lines.append(f"  scale（スケール）: {result.input_summary.get('scale', '-')}")
    lines.append("")

    # 予測結果
    lines.append("-" * 60)
    outcome_ja = {
        "Success": "成功",
        "Failure": "失敗",
        "PartialSuccess": "部分的成功",
        "Mixed": "混合結果"
    }
    lines.append(f"[予測結果] {outcome_ja.get(result.predicted_outcome, result.predicted_outcome)}")
    lines.append(f"  信頼度: {result.confidence:.1%}")
    lines.append(f"  サンプルサイズ: {result.sample_size}件")
    lines.append("")

    # 確率分布
    lines.append("[確率分布]")
    lines.append(f"  成功率: {result.success_rate:.1f}%")
    lines.append(f"  失敗率: {result.failure_rate:.1f}%")
    lines.append(f"  部分成功率: {result.partial_rate:.1f}%")
    lines.append(f"  混合結果率: {result.mixed_rate:.1f}%")
    lines.append("")

    # 類似事例
    if result.similar_cases:
        lines.append("-" * 60)
        lines.append("[類似事例（上位3件）]")
        for i, case in enumerate(result.similar_cases, 1):
            lines.append(f"  {i}. {case.target_name} ({case.scale}, {case.period})")
            lines.append(f"     結果: {case.outcome}, 類似度: {case.similarity_score:.0%}")
            lines.append(f"     一致項目: {', '.join(case.match_factors)}")
            lines.append(f"     概要: {case.story_summary}")
        lines.append("")

    # 推奨アクション
    lines.append("-" * 60)
    lines.append("[推奨アクション（上位5つ）]")
    for i, (action, score, reason) in enumerate(result.recommended_actions, 1):
        lines.append(f"  {i}. {action} (スコア: {score:.1f})")
        lines.append(f"     {reason}")
    lines.append("")

    # 卦のアドバイス
    if result.hexagram_advice:
        lines.append("-" * 60)
        lines.append("[卦のアドバイス]")
        lines.append(f"  {result.hexagram_advice}")
        lines.append("")

    # 爻のアドバイス
    if result.yao_advice:
        lines.append("[爻のアドバイス（転換点）]")
        lines.append(f"  {result.yao_advice}")
        lines.append("")

    # 警告
    if result.warnings:
        lines.append("-" * 60)
        lines.append("[注意事項]")
        for warning in result.warnings:
            lines.append(f"  ! {warning}")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    """CLI メイン関数"""
    parser = argparse.ArgumentParser(
        description="易経変化予測エンジン - 状況から結果を予測",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python3 scripts/prediction_engine.py --before-hex 坎 --action 攻める・挑戦 --scale company
  python3 scripts/prediction_engine.py --before-hex 離 --before-state 絶頂・慢心 --action 守る・維持 --scale individual
  python3 scripts/prediction_engine.py --before-hex 震 --action 刷新・破壊 --scale country --json

八卦（before-hex）:
  乾（創造）、坤（受容）、震（動）、巽（浸透）、坎（試練）、離（明晰）、艮（停止）、兌（喜悦）

アクション（action）:
  攻める・挑戦、守る・維持、捨てる・撤退、耐える・潜伏、
  対話・融合、刷新・破壊、逃げる・放置、分散・スピンオフ

スケール（scale）:
  company, individual, family, country, other
        """
    )

    parser.add_argument(
        "--before-hex", "-b",
        required=True,
        choices=TRIGRAMS,
        help="現在の八卦（位相）"
    )

    parser.add_argument(
        "--before-state", "-s",
        choices=BEFORE_STATES,
        help="現在の状態（オプション）"
    )

    parser.add_argument(
        "--action", "-a",
        required=True,
        choices=ACTION_TYPES,
        help="取ろうとしているアクション"
    )

    parser.add_argument(
        "--scale", "-c",
        required=True,
        choices=SCALES,
        help="対象のスケール"
    )

    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="結果をJSON形式で出力"
    )

    args = parser.parse_args()

    # エンジン初期化
    engine = PredictionEngine()

    # 予測実行
    result = engine.predict(
        before_hex=args.before_hex,
        action_type=args.action,
        scale=args.scale,
        before_state=args.before_state
    )

    # 出力
    if args.json:
        output = {
            "predicted_outcome": result.predicted_outcome,
            "confidence": result.confidence,
            "success_rate": result.success_rate,
            "failure_rate": result.failure_rate,
            "partial_rate": result.partial_rate,
            "mixed_rate": result.mixed_rate,
            "sample_size": result.sample_size,
            "similar_cases": [
                {
                    "transition_id": c.transition_id,
                    "target_name": c.target_name,
                    "scale": c.scale,
                    "period": c.period,
                    "story_summary": c.story_summary,
                    "outcome": c.outcome,
                    "similarity_score": c.similarity_score,
                    "match_factors": c.match_factors
                }
                for c in result.similar_cases
            ],
            "recommended_actions": [
                {"action": a, "score": s, "reason": r}
                for a, s, r in result.recommended_actions
            ],
            "warnings": result.warnings,
            "hexagram_advice": result.hexagram_advice,
            "yao_advice": result.yao_advice,
            "input_summary": result.input_summary
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(format_prediction_result(result))


if __name__ == "__main__":
    main()
