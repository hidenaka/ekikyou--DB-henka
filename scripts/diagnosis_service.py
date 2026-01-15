#!/usr/bin/env python3
"""
六十四卦診断サービス - 統合モジュール

構造化質問方式による診断と類似事例検索を統合。
無料版/有料版の出力差別化をサポート。
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path

from hexagram_calculator import HexagramCalculator, DiagnosisAnswers, DiagnosisResult
from similar_case_search import SimilarCaseSearcher, SimilarCase


# 卦の基本解説データ
HEXAGRAM_BASIC_INFO = {
    1: {"keyword": "創造・始まり", "one_line": "天の力。創造と始まりの時。大いなる成功の可能性。"},
    2: {"keyword": "受容・柔順", "one_line": "地の力。受容と従順。寛容さが成功を導く。"},
    3: {"keyword": "困難・萌芽", "one_line": "産みの苦しみ。困難の中に新しい芽生えがある。"},
    4: {"keyword": "蒙昧・学び", "one_line": "無知の状態。師に学び、成長する時。"},
    5: {"keyword": "待機・忍耐", "one_line": "待つことが大切。焦らず機を待て。"},
    6: {"keyword": "争い・訴訟", "one_line": "争いの兆し。和解を模索すべし。"},
    7: {"keyword": "統率・軍", "one_line": "組織の力。規律正しく進め。"},
    8: {"keyword": "親和・協力", "one_line": "協力関係を築く時。仲間と共に。"},
    9: {"keyword": "抑制・蓄積", "one_line": "小さな力で大きなものを留める。忍耐の時。"},
    10: {"keyword": "慎重・礼節", "one_line": "虎の尾を踏むような危険。礼を尽くせ。"},
    11: {"keyword": "平和・通泰", "one_line": "天地交わり万物通ず。最良の時。"},
    12: {"keyword": "閉塞・停滞", "one_line": "天地交わらず。閉塞の時。静かに待て。"},
    13: {"keyword": "同志・協調", "one_line": "志を同じくする者と共に。協調の力。"},
    14: {"keyword": "大いなる所有", "one_line": "大きな成功。謙虚さを忘れずに。"},
    15: {"keyword": "謙遜・控えめ", "one_line": "謙虚さが吉を招く。控えめに行動せよ。"},
    16: {"keyword": "喜び・準備", "one_line": "楽しみの前触れ。準備を整えよ。"},
    17: {"keyword": "従う・随順", "one_line": "時に従い、柔軟に対応せよ。"},
    18: {"keyword": "腐敗・改革", "one_line": "古いものを改める時。改革の好機。"},
    19: {"keyword": "接近・臨む", "one_line": "良いものが近づく。謙虚に迎えよ。"},
    20: {"keyword": "観察・省察", "one_line": "よく観察し、自己を省みよ。"},
    21: {"keyword": "決断・明確化", "one_line": "噛み砕いて物事を明らかにする。"},
    22: {"keyword": "装飾・美", "one_line": "外見を整える。本質も大切に。"},
    23: {"keyword": "剥落・衰退", "one_line": "衰えの時。静かに耐えよ。"},
    24: {"keyword": "復活・回帰", "one_line": "一陽来復。新しい始まりの兆し。"},
    25: {"keyword": "無妄・純真", "one_line": "作為なき純粋さ。自然体で進め。"},
    26: {"keyword": "大いなる蓄え", "one_line": "力を蓄える時。大器晩成。"},
    27: {"keyword": "養い・滋養", "one_line": "養うこと。心身の滋養を。"},
    28: {"keyword": "過大・重荷", "one_line": "荷が重すぎる。支えを求めよ。"},
    29: {"keyword": "困難・険しさ", "one_line": "重なる困難。誠実に乗り越えよ。"},
    30: {"keyword": "明晰・付着", "one_line": "火のように輝く。明晰さを保て。"},
    31: {"keyword": "感応・交流", "one_line": "心の交流。相互理解の時。"},
    32: {"keyword": "恒常・持続", "one_line": "一貫性を保て。継続は力なり。"},
    33: {"keyword": "退却・遁れ", "one_line": "戦略的撤退。時を待て。"},
    34: {"keyword": "大いなる力", "one_line": "勢いがある。正しく使え。"},
    35: {"keyword": "進歩・昇進", "one_line": "陽が昇る。進展の時。"},
    36: {"keyword": "暗黒・隠遁", "one_line": "明を傷つける時。才を隠せ。"},
    37: {"keyword": "家庭・内部", "one_line": "家を整える。内部の調和。"},
    38: {"keyword": "背反・対立", "one_line": "対立の時。小事に吉。"},
    39: {"keyword": "困難・蹇", "one_line": "足が進まない。助けを求めよ。"},
    40: {"keyword": "解放・解決", "one_line": "困難から解放される。迅速に動け。"},
    41: {"keyword": "損失・減少", "one_line": "減らすことで得るものがある。"},
    42: {"keyword": "増加・利益", "one_line": "増益の時。大きな事業に吉。"},
    43: {"keyword": "決断・決壊", "one_line": "決断の時。断固として進め。"},
    44: {"keyword": "出会い・遭遇", "one_line": "思わぬ出会い。警戒も必要。"},
    45: {"keyword": "集合・結集", "one_line": "人が集まる。団結の力。"},
    46: {"keyword": "上昇・成長", "one_line": "昇進・成長の時。努力が実る。"},
    47: {"keyword": "困窮・窮乏", "one_line": "困窮の時。言葉より行動で。"},
    48: {"keyword": "井戸・源泉", "one_line": "変わらぬ源。本質を守れ。"},
    49: {"keyword": "変革・革命", "one_line": "変革の時。古いものを改めよ。"},
    50: {"keyword": "鼎・新秩序", "one_line": "新しい秩序の確立。安定の象。"},
    51: {"keyword": "震動・雷", "one_line": "突然の衝撃。恐れず対処せよ。"},
    52: {"keyword": "静止・止まる", "one_line": "止まるべき時。動くな。"},
    53: {"keyword": "漸進・徐々に", "one_line": "ゆっくり着実に進め。"},
    54: {"keyword": "軽率・嫁ぐ", "one_line": "軽率な動きに注意。慎重に。"},
    55: {"keyword": "豊穣・絶頂", "one_line": "豊かさの極み。驕るな。"},
    56: {"keyword": "旅・流浪", "one_line": "旅人の心得。慎ましく進め。"},
    57: {"keyword": "浸透・従順", "one_line": "風のように柔軟に浸透せよ。"},
    58: {"keyword": "喜悦・歓喜", "one_line": "喜びの時。心を通わせよ。"},
    59: {"keyword": "散開・離散", "one_line": "凝り固まりを解く。柔軟に。"},
    60: {"keyword": "節制・制限", "one_line": "節度を守れ。過ぎたるは及ばず。"},
    61: {"keyword": "誠実・中心", "one_line": "まことの心。誠実さが通じる。"},
    62: {"keyword": "小過・謙虚", "one_line": "小さく超える。控えめに行動。"},
    63: {"keyword": "完成・既済", "one_line": "既に成る。油断するな。"},
    64: {"keyword": "未完成・未済", "one_line": "まだ成らず。最後まで慎重に。"},
}

# 爻位の基本解説
YAO_BASIC_INFO = {
    1: {"name": "初爻", "meaning": "始まりの段階。まだ動くべきでない。準備の時。"},
    2: {"name": "二爻", "meaning": "現れ始める段階。徐々に形になる。内に留まれ。"},
    3: {"name": "三爻", "meaning": "転換点。危険を伴う。慎重な判断が必要。"},
    4: {"name": "四爻", "meaning": "上への接近。跳躍の準備。機を見て動け。"},
    5: {"name": "五爻", "meaning": "中正の位。最も良い位置。リーダーシップを発揮。"},
    6: {"name": "上爻", "meaning": "極点。行き過ぎに注意。次への移行を考えよ。"},
}


@dataclass
class FreeDiagnosisOutput:
    """無料版診断出力"""
    reading_id: str
    hexagram_name: str
    hexagram_number: int
    hexagram_full_name: str
    keyword: str
    one_line_summary: str
    yao_position: int
    yao_name: str
    yao_meaning: str
    sample_case: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_text(self) -> str:
        """テキスト形式で出力"""
        lines = [
            f"【診断結果】第{self.hexagram_number}卦 {self.hexagram_name}（{self.hexagram_full_name}）",
            f"",
            f"キーワード: {self.keyword}",
            f"",
            f"{self.one_line_summary}",
            f"",
            f"現在の段階: {self.yao_name}（{self.yao_position}爻）",
            f"{self.yao_meaning}",
        ]
        if self.sample_case:
            lines.extend([
                f"",
                f"【参考事例】{self.sample_case.get('target_name', 'N/A')}",
                f"結果: {self.sample_case.get('outcome', 'N/A')}",
            ])
        lines.extend([
            f"",
            f"─────────────────",
            f"詳細な解説・アドバイスは有料版で",
            f"Reading ID: {self.reading_id}",
        ])
        return "\n".join(lines)


@dataclass
class PaidDiagnosisOutput:
    """有料版診断出力"""
    reading_id: str
    hexagram_name: str
    hexagram_number: int
    hexagram_full_name: str
    upper_trigram: str
    lower_trigram: str
    keyword: str
    one_line_summary: str
    yao_position: int
    yao_name: str
    yao_meaning: str
    detailed_analysis: str
    similar_cases: List[Dict] = field(default_factory=list)
    action_recommendations: List[str] = field(default_factory=list)
    caution_points: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_text(self) -> str:
        """テキスト形式で出力"""
        lines = [
            f"═══════════════════════════════════════",
            f"【診断結果】第{self.hexagram_number}卦 {self.hexagram_name}",
            f"正式名: {self.hexagram_full_name}",
            f"上卦: {self.upper_trigram} / 下卦: {self.lower_trigram}",
            f"═══════════════════════════════════════",
            f"",
            f"■ キーワード: {self.keyword}",
            f"",
            f"■ 概要",
            f"{self.one_line_summary}",
            f"",
            f"■ 現在の段階: {self.yao_name}（{self.yao_position}爻）",
            f"{self.yao_meaning}",
            f"",
            f"■ 詳細分析",
            f"{self.detailed_analysis}",
            f"",
        ]

        if self.action_recommendations:
            lines.append("■ 推奨アクション")
            for i, rec in enumerate(self.action_recommendations, 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")

        if self.caution_points:
            lines.append("■ 注意点")
            for i, caution in enumerate(self.caution_points, 1):
                lines.append(f"  {i}. {caution}")
            lines.append("")

        if self.similar_cases:
            lines.append("■ 類似事例（根拠）")
            for i, case in enumerate(self.similar_cases, 1):
                lines.extend([
                    f"",
                    f"  [{i}] {case.get('target_name', 'N/A')}",
                    f"      卦: {case.get('hexagram_name', 'N/A')} / 結果: {case.get('outcome', 'N/A')}",
                    f"      {case.get('story_summary', 'N/A')[:100]}...",
                ])

        lines.extend([
            f"",
            f"─────────────────────────────────────",
            f"Reading ID: {self.reading_id}",
        ])
        return "\n".join(lines)


class DiagnosisService:
    """六十四卦診断サービス"""

    def __init__(self, cases_path: str = None):
        self.calculator = HexagramCalculator()
        self.searcher = SimilarCaseSearcher(cases_path)

    def diagnose(self, answers: DiagnosisAnswers, tier: str = "free") -> Dict:
        """
        診断を実行

        Args:
            answers: 質問への回答
            tier: "free" または "paid"

        Returns:
            Dict: 診断結果（tierに応じた出力）
        """
        # 卦を算出
        result = self.calculator.diagnose(answers)

        # 類似事例を検索
        similar_cases = self.searcher.search(
            upper_trigram=result.upper_trigram,
            lower_trigram=result.lower_trigram,
            hexagram_number=result.hexagram_number,
            active_line=result.active_line,
            theme=answers.theme,
            max_results=5 if tier == "paid" else 1
        )

        # 基本情報を取得
        hex_info = HEXAGRAM_BASIC_INFO.get(result.hexagram_number, {})
        yao_info = YAO_BASIC_INFO.get(result.active_line, {})

        if tier == "free":
            return self._create_free_output(result, hex_info, yao_info, similar_cases)
        else:
            return self._create_paid_output(result, hex_info, yao_info, similar_cases)

    def _create_free_output(
        self,
        result: DiagnosisResult,
        hex_info: Dict,
        yao_info: Dict,
        similar_cases: List[SimilarCase]
    ) -> FreeDiagnosisOutput:
        """無料版出力を生成"""
        sample_case = None
        if similar_cases:
            sc = similar_cases[0]
            sample_case = {
                "target_name": sc.target_name,
                "outcome": sc.outcome,
            }

        return FreeDiagnosisOutput(
            reading_id=result.reading_id,
            hexagram_name=result.hexagram_name,
            hexagram_number=result.hexagram_number,
            hexagram_full_name=result.hexagram_full_name,
            keyword=hex_info.get("keyword", ""),
            one_line_summary=hex_info.get("one_line", ""),
            yao_position=result.active_line,
            yao_name=yao_info.get("name", ""),
            yao_meaning=yao_info.get("meaning", ""),
            sample_case=sample_case,
        )

    def _create_paid_output(
        self,
        result: DiagnosisResult,
        hex_info: Dict,
        yao_info: Dict,
        similar_cases: List[SimilarCase]
    ) -> PaidDiagnosisOutput:
        """有料版出力を生成"""
        # 詳細分析（実際のサービスではLLMで生成）
        detailed_analysis = self._generate_detailed_analysis(result, hex_info, yao_info)

        # 推奨アクション（実際のサービスではLLMで生成）
        actions = self._generate_action_recommendations(result, hex_info, yao_info)

        # 注意点（実際のサービスではLLMで生成）
        cautions = self._generate_caution_points(result, hex_info, yao_info)

        # 類似事例を辞書形式に変換
        case_dicts = [
            {
                "target_name": sc.target_name,
                "hexagram_name": sc.hexagram_name,
                "outcome": sc.outcome,
                "story_summary": sc.story_summary,
                "main_domain": sc.main_domain,
            }
            for sc in similar_cases
        ]

        return PaidDiagnosisOutput(
            reading_id=result.reading_id,
            hexagram_name=result.hexagram_name,
            hexagram_number=result.hexagram_number,
            hexagram_full_name=result.hexagram_full_name,
            upper_trigram=result.upper_trigram,
            lower_trigram=result.lower_trigram,
            keyword=hex_info.get("keyword", ""),
            one_line_summary=hex_info.get("one_line", ""),
            yao_position=result.active_line,
            yao_name=yao_info.get("name", ""),
            yao_meaning=yao_info.get("meaning", ""),
            detailed_analysis=detailed_analysis,
            similar_cases=case_dicts,
            action_recommendations=actions,
            caution_points=cautions,
        )

    def _generate_detailed_analysis(
        self,
        result: DiagnosisResult,
        hex_info: Dict,
        yao_info: Dict
    ) -> str:
        """詳細分析を生成（テンプレート版。実サービスではLLMを使用）"""
        return (
            f"{result.hexagram_full_name}は、外部環境（{result.upper_trigram}）と"
            f"内部状態（{result.lower_trigram}）の組み合わせを示しています。\n"
            f"現在の段階は{yao_info.get('name', '')}であり、"
            f"{yao_info.get('meaning', '')}\n"
            f"この卦は「{hex_info.get('keyword', '')}」を表し、"
            f"全体として{hex_info.get('one_line', '')}という状況です。"
        )

    def _generate_action_recommendations(
        self,
        result: DiagnosisResult,
        hex_info: Dict,
        yao_info: Dict
    ) -> List[str]:
        """推奨アクションを生成（テンプレート版）"""
        actions = []

        # 爻位に基づく基本アクション
        yao_actions = {
            1: "まだ動くべきではありません。準備と情報収集に集中してください。",
            2: "少しずつ自分の存在を示しつつ、まだ内に留まりましょう。",
            3: "転換点にいます。慎重に判断し、リスクを見極めてください。",
            4: "上位者や決定権者との関係構築を意識してください。",
            5: "リーダーシップを発揮し、決断を下す時です。",
            6: "過度な執着を避け、次のステップへの移行を考えてください。",
        }
        actions.append(yao_actions.get(result.active_line, ""))

        # 卦の特性に基づく追加アクション（簡易版）
        if result.upper_trigram in ["震", "離"]:
            actions.append("外部環境は活発です。積極的に機会を捉えましょう。")
        elif result.upper_trigram in ["艮", "坤"]:
            actions.append("外部環境は静かです。焦らず待つ姿勢が有効です。")

        return actions

    def _generate_caution_points(
        self,
        result: DiagnosisResult,
        hex_info: Dict,
        yao_info: Dict
    ) -> List[str]:
        """注意点を生成（テンプレート版）"""
        cautions = []

        # 爻位に基づく注意点
        yao_cautions = {
            1: "早まった行動は禁物です。",
            2: "自己主張しすぎないように。",
            3: "この時期の判断ミスは後に影響します。",
            4: "焦って飛び出さないように。",
            5: "驕りに注意。謙虚さを忘れずに。",
            6: "執着しすぎると逆効果になります。",
        }
        cautions.append(yao_cautions.get(result.active_line, ""))

        # 卦の組み合わせによる注意点
        if result.upper_trigram == result.lower_trigram:
            cautions.append("内外が同じ卦です。偏りに注意してください。")

        return cautions


def demo():
    """デモ実行"""
    service = DiagnosisService()

    # サンプル回答
    answers = DiagnosisAnswers(
        perspective="self",
        theme="career",
        q3_energy="A",   # 拡大
        q4_motion="A",   # 動
        q5_symbol="A",   # 雷（震）
        q6_energy="A",   # 積極
        q7_motion="B",   # 静
        q8_symbol="C",   # 天（乾）
        q9_stage="C",    # 転換点（三爻）
    )

    print("=" * 60)
    print("【無料版診断結果】")
    print("=" * 60)
    free_result = service.diagnose(answers, tier="free")
    print(free_result.to_text())

    print("\n\n")
    print("=" * 60)
    print("【有料版診断結果】")
    print("=" * 60)
    paid_result = service.diagnose(answers, tier="paid")
    print(paid_result.to_text())


if __name__ == "__main__":
    demo()
