#!/usr/bin/env python3
"""
類似事例検索モジュール

診断結果（卦・爻）に基づいて、既存の12,000件以上の事例から
類似事例を検索し、根拠として提示する。
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class SimilarCase:
    """類似事例"""
    transition_id: str
    target_name: str
    story_summary: str
    hexagram_name: str
    hexagram_number: int
    upper_trigram: str
    lower_trigram: str
    yao: Optional[int]
    main_domain: Optional[str]
    outcome: str
    similarity_score: float
    match_reasons: List[str]


class SimilarCaseSearcher:
    """類似事例検索器"""

    # 八卦名の正規化マッピング
    TRIGRAM_NORMALIZE = {
        "乾": "乾", "天": "乾", "☰": "乾",
        "兌": "兌", "沢": "兌", "☱": "兌",
        "離": "離", "火": "離", "☲": "離",
        "震": "震", "雷": "震", "☳": "震",
        "巽": "巽", "風": "巽", "☴": "巽",
        "坎": "坎", "水": "坎", "☵": "坎",
        "艮": "艮", "山": "艮", "☶": "艮",
        "坤": "坤", "地": "坤", "☷": "坤",
    }

    # main_domainの正規化マッピング
    DOMAIN_NORMALIZE = {
        "career": ["キャリア", "ビジネス", "Business", "business", "経営"],
        "relationship": ["生活・暮らし", "家族", "社会・コミュニティ"],
        "business": ["ビジネス", "Business", "business", "製造", "小売・サービス", "金融"],
        "health": ["医療・製薬", "Healthcare", "医療"],
        "education": ["教育", "education", "学術研究"],
        "finance": ["金融", "金融・投資", "finance", "fintech"],
        "technology": ["テクノロジー", "Technology", "technology", "IT", "IT・テクノロジー"],
    }

    def __init__(self, cases_path: str = None):
        """
        初期化

        Args:
            cases_path: cases.jsonlのパス。Noneの場合はデフォルトパスを使用
        """
        if cases_path is None:
            cases_path = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"
        else:
            cases_path = Path(cases_path)

        self.cases = self._load_cases(cases_path)
        self._build_index()

    def _load_cases(self, path: Path) -> List[Dict]:
        """事例を読み込む"""
        cases = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    case = json.loads(line.strip())
                    cases.append(case)
                except json.JSONDecodeError:
                    continue
        return cases

    def _build_index(self):
        """検索用インデックスを構築"""
        # 卦番号でインデックス
        self.by_hexagram = {}
        # 上卦でインデックス
        self.by_upper = {}
        # 下卦でインデックス
        self.by_lower = {}

        for i, case in enumerate(self.cases):
            # 卦番号
            hex_num = case.get('hexagram_number')
            if hex_num:
                if hex_num not in self.by_hexagram:
                    self.by_hexagram[hex_num] = []
                self.by_hexagram[hex_num].append(i)

            # 上卦
            upper = case.get('upper_trigram')
            if upper:
                upper_norm = self.TRIGRAM_NORMALIZE.get(upper, upper)
                if upper_norm not in self.by_upper:
                    self.by_upper[upper_norm] = []
                self.by_upper[upper_norm].append(i)

            # 下卦
            lower = case.get('lower_trigram')
            if lower:
                lower_norm = self.TRIGRAM_NORMALIZE.get(lower, lower)
                if lower_norm not in self.by_lower:
                    self.by_lower[lower_norm] = []
                self.by_lower[lower_norm].append(i)

    def _normalize_trigram(self, trigram: str) -> str:
        """八卦名を正規化"""
        return self.TRIGRAM_NORMALIZE.get(trigram, trigram)

    def _match_domain(self, case_domain: Optional[str], target_theme: str) -> bool:
        """ドメインのマッチングを判定"""
        if not case_domain or not target_theme:
            return False

        target_domains = self.DOMAIN_NORMALIZE.get(target_theme, [])
        return case_domain in target_domains

    def search(
        self,
        upper_trigram: str,
        lower_trigram: str,
        hexagram_number: int,
        active_line: Optional[int] = None,
        theme: Optional[str] = None,
        max_results: int = 5
    ) -> List[SimilarCase]:
        """
        類似事例を検索

        Args:
            upper_trigram: 上卦（例: "震"）
            lower_trigram: 下卦（例: "乾"）
            hexagram_number: 卦番号（1-64）
            active_line: 爻位（1-6、Noneの場合は爻でフィルタしない）
            theme: テーマ（career, relationship, etc.）
            max_results: 最大結果数

        Returns:
            List[SimilarCase]: 類似事例のリスト（スコア順）
        """
        upper_norm = self._normalize_trigram(upper_trigram)
        lower_norm = self._normalize_trigram(lower_trigram)

        # 候補を収集
        candidates = []
        seen = set()

        # 1. 完全一致（卦番号）を優先
        if hexagram_number in self.by_hexagram:
            for idx in self.by_hexagram[hexagram_number]:
                if idx not in seen:
                    seen.add(idx)
                    candidates.append((idx, 1.0))  # 完全一致スコア

        # 2. 上卦のみ一致
        if upper_norm in self.by_upper:
            for idx in self.by_upper[upper_norm]:
                if idx not in seen:
                    seen.add(idx)
                    candidates.append((idx, 0.5))  # 部分一致スコア

        # 3. 下卦のみ一致
        if lower_norm in self.by_lower:
            for idx in self.by_lower[lower_norm]:
                if idx not in seen:
                    seen.add(idx)
                    candidates.append((idx, 0.5))  # 部分一致スコア

        # スコアリングと結果生成
        results = []
        for idx, base_score in candidates:
            case = self.cases[idx]
            score = base_score
            reasons = []

            # 卦の一致を確認
            case_hex_num = case.get('hexagram_number')
            if case_hex_num == hexagram_number:
                reasons.append("卦が完全一致")
            else:
                case_upper = self._normalize_trigram(case.get('upper_trigram', ''))
                case_lower = self._normalize_trigram(case.get('lower_trigram', ''))
                if case_upper == upper_norm:
                    reasons.append("上卦が一致")
                if case_lower == lower_norm:
                    reasons.append("下卦が一致")

            # 爻の一致を確認
            case_yao = case.get('yao')
            if active_line and case_yao:
                if case_yao == active_line:
                    score += 0.3
                    reasons.append(f"爻位が一致({active_line}爻)")

            # ドメインの一致を確認
            case_domain = case.get('main_domain')
            if theme and self._match_domain(case_domain, theme):
                score += 0.2
                reasons.append(f"分野が類似({case_domain})")

            # 結果を生成
            if reasons:  # 何らかのマッチがある場合のみ追加
                results.append(SimilarCase(
                    transition_id=case.get('transition_id', 'N/A'),
                    target_name=case.get('target_name', 'N/A'),
                    story_summary=case.get('story_summary', 'N/A')[:200],
                    hexagram_name=case.get('hexagram_name', 'N/A'),
                    hexagram_number=case.get('hexagram_number', 0),
                    upper_trigram=case.get('upper_trigram', 'N/A'),
                    lower_trigram=case.get('lower_trigram', 'N/A'),
                    yao=case.get('yao'),
                    main_domain=case_domain,
                    outcome=case.get('outcome', 'N/A'),
                    similarity_score=min(score, 1.0),
                    match_reasons=reasons
                ))

        # スコア順でソート
        results.sort(key=lambda x: x.similarity_score, reverse=True)

        return results[:max_results]

    def search_by_diagnosis(self, diagnosis_result: Dict, max_results: int = 5) -> List[SimilarCase]:
        """
        診断結果から類似事例を検索

        Args:
            diagnosis_result: HexagramCalculator.diagnose()の結果
            max_results: 最大結果数

        Returns:
            List[SimilarCase]: 類似事例のリスト
        """
        return self.search(
            upper_trigram=diagnosis_result.get('upper_trigram', ''),
            lower_trigram=diagnosis_result.get('lower_trigram', ''),
            hexagram_number=diagnosis_result.get('hexagram_number', 0),
            active_line=diagnosis_result.get('active_line'),
            theme=diagnosis_result.get('inputs', {}).get('theme'),
            max_results=max_results
        )


def demo():
    """デモ実行"""
    searcher = SimilarCaseSearcher()

    print(f"=== 事例データベース ===")
    print(f"総事例数: {len(searcher.cases)}件")
    print()

    # 雷天大壮（第34卦）、3爻、キャリア関連で検索
    print("=== 検索条件 ===")
    print("卦: 雷天大壮（第34卦）")
    print("上卦: 震（雷）")
    print("下卦: 乾（天）")
    print("爻位: 3爻")
    print("テーマ: career")
    print()

    results = searcher.search(
        upper_trigram="震",
        lower_trigram="乾",
        hexagram_number=34,
        active_line=3,
        theme="career",
        max_results=5
    )

    print(f"=== 検索結果 ({len(results)}件) ===")
    for i, r in enumerate(results, 1):
        print(f"\n--- {i}. {r.target_name} ---")
        print(f"卦: {r.hexagram_name} (第{r.hexagram_number}卦)")
        print(f"上卦: {r.upper_trigram} / 下卦: {r.lower_trigram}")
        print(f"爻: {r.yao}爻" if r.yao else "爻: N/A")
        print(f"分野: {r.main_domain}")
        print(f"結果: {r.outcome}")
        print(f"類似度: {r.similarity_score:.2f}")
        print(f"一致理由: {', '.join(r.match_reasons)}")
        print(f"概要: {r.story_summary}...")


if __name__ == "__main__":
    demo()
