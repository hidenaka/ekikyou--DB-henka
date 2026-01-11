#!/usr/bin/env python3
"""
reverse_lookup.py - 動画テーマから易経の卦を逆引きするスクリプト

使い方:
    python scripts/reverse_lookup.py "孤独"
    python scripts/reverse_lookup.py --theme "趣味がない"
    python scripts/reverse_lookup.py --phrase "なんかモヤモヤする"
    python scripts/reverse_lookup.py --interactive
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


# パス設定
PROJECT_ROOT = Path(__file__).parent.parent
MAPPINGS_DIR = PROJECT_ROOT / "data" / "mappings"
CASES_FILE = PROJECT_ROOT / "data" / "raw" / "cases.jsonl"


@dataclass
class LookupResult:
    """逆引き結果"""
    theme_id: str
    theme_name: str
    primary_trigram: str
    suggested_hexagrams: List[str]
    before_state: str
    typical_pattern: str
    video_angle: str
    outro_message: str
    matched_keywords: List[str]
    confidence: float  # 0.0 - 1.0


@dataclass
class CaseMatch:
    """マッチした事例"""
    transition_id: str
    target_name: str
    story_summary: str
    before_state: str
    after_state: str
    before_hex: str
    after_hex: str
    classical_before_hexagram: str
    pattern_type: str
    outcome: str
    relevance_score: float


class ReverseLookup:
    """動画テーマから易経を逆引きするクラス"""

    def __init__(self):
        self.theme_mapping = self._load_json("theme_to_hexagram.json")
        self.era_mapping = self._load_json("era_classification.json")
        self.daily_mapping = self._load_json("daily_to_yijing.json")
        self.cases = self._load_cases()

    def _load_json(self, filename: str) -> dict:
        """JSONファイルを読み込む"""
        filepath = MAPPINGS_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} が見つかりません")
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_cases(self) -> List[dict]:
        """事例データを読み込む"""
        cases = []
        if CASES_FILE.exists():
            with open(CASES_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        cases.append(json.loads(line))
        return cases

    def lookup_by_theme(self, query: str) -> Optional[LookupResult]:
        """テーマキーワードから卦を逆引き"""
        query_lower = query.lower()
        best_match = None
        best_score = 0
        matched_keywords = []

        for theme in self.theme_mapping.get("themes", []):
            keywords = theme.get("keywords", [])
            score = 0
            current_matches = []

            for kw in keywords:
                if kw in query or query in kw:
                    score += 1
                    current_matches.append(kw)

            # テーマ名自体もチェック
            if query in theme.get("theme_name", ""):
                score += 2
                current_matches.append(theme["theme_name"])

            if score > best_score:
                best_score = score
                best_match = theme
                matched_keywords = current_matches

        if best_match:
            confidence = min(best_score / 3.0, 1.0)  # 3つ以上マッチで100%
            return LookupResult(
                theme_id=best_match["theme_id"],
                theme_name=best_match["theme_name"],
                primary_trigram=best_match["primary_trigram"],
                suggested_hexagrams=best_match["suggested_hexagrams"],
                before_state=best_match["before_state"],
                typical_pattern=best_match["typical_pattern"],
                video_angle=best_match["video_angle"],
                outro_message=best_match["outro_message"],
                matched_keywords=matched_keywords,
                confidence=confidence
            )
        return None

    def lookup_by_phrase(self, phrase: str) -> Tuple[str, str, List[str]]:
        """日常語フレーズから状態と卦を逆引き"""
        # まずクイックルックアップをチェック
        quick = self.daily_mapping.get("quick_lookup", {})
        for key, value in quick.items():
            if key in phrase:
                return value["state"], value["trigram"], [key]

        # 詳細マッピングをチェック
        for category, items in self.daily_mapping.get("feelings_to_state", {}).items():
            for item in items:
                for daily_phrase in item.get("daily_phrases", []):
                    if daily_phrase in phrase or phrase in daily_phrase:
                        return (
                            item["yijing_state"],
                            item["trigram"],
                            [daily_phrase]
                        )

        return "不明", "不明", []

    def find_similar_cases(
        self,
        trigram: str = None,
        state: str = None,
        pattern: str = None,
        limit: int = 5
    ) -> List[CaseMatch]:
        """条件に合う事例を検索"""
        scored_cases = []

        for case in self.cases:
            score = 0

            # 八卦マッチ
            if trigram and case.get("before_hex") == trigram:
                score += 30

            # 状態マッチ
            if state and case.get("before_state") == state:
                score += 50

            # パターンマッチ
            if pattern and case.get("pattern_type") == pattern:
                score += 20

            # 成功事例を優先
            if case.get("outcome") == "Success":
                score += 10

            # 信頼性でボーナス
            cred = case.get("credibility_rank", "C")
            cred_bonus = {"S": 15, "A": 10, "B": 5, "C": 0}.get(cred, 0)
            score += cred_bonus

            if score > 0:
                scored_cases.append((score, case))

        # スコア順にソート
        scored_cases.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, case in scored_cases[:limit]:
            results.append(CaseMatch(
                transition_id=case.get("transition_id", ""),
                target_name=case.get("target_name", ""),
                story_summary=case.get("story_summary", "")[:100] + "...",
                before_state=case.get("before_state", ""),
                after_state=case.get("after_state", ""),
                before_hex=case.get("before_hex", ""),
                after_hex=case.get("after_hex", ""),
                classical_before_hexagram=case.get("classical_before_hexagram", ""),
                pattern_type=case.get("pattern_type", ""),
                outcome=case.get("outcome", ""),
                relevance_score=score / 100.0
            ))

        return results

    def get_era_info(self, era_name: str) -> Optional[dict]:
        """時代情報を取得"""
        eras = self.era_mapping.get("eras", {})
        return eras.get(era_name)

    def generate_video_brief(self, theme_query: str) -> dict:
        """動画企画用のブリーフを生成"""
        # テーマから卦を逆引き
        theme_result = self.lookup_by_theme(theme_query)

        if not theme_result:
            return {"error": f"テーマ '{theme_query}' に対応するマッピングが見つかりません"}

        # 類似事例を検索
        similar_cases = self.find_similar_cases(
            trigram=theme_result.primary_trigram,
            state=theme_result.before_state,
            pattern=theme_result.typical_pattern,
            limit=5
        )

        # 時代別の切り口を取得
        era_angles = {}
        for era_name in ["昭和", "平成", "令和", "未来A", "未来B"]:
            era_info = self.get_era_info(era_name)
            if era_info:
                era_angles[era_name] = {
                    "characteristics": era_info.get("characteristics", {}),
                    "symbols": era_info.get("video_template", {}).get("symbols", []),
                    "mood": era_info.get("video_template", {}).get("mood", "")
                }

        return {
            "theme": {
                "query": theme_query,
                "matched_theme": theme_result.theme_name,
                "confidence": theme_result.confidence,
                "matched_keywords": theme_result.matched_keywords
            },
            "yijing": {
                "primary_trigram": theme_result.primary_trigram,
                "suggested_hexagrams": theme_result.suggested_hexagrams,
                "before_state": theme_result.before_state,
                "typical_pattern": theme_result.typical_pattern
            },
            "video_content": {
                "angle": theme_result.video_angle,
                "outro_message": theme_result.outro_message
            },
            "era_angles": era_angles,
            "similar_cases": [
                {
                    "name": c.target_name,
                    "summary": c.story_summary,
                    "pattern": c.pattern_type,
                    "outcome": c.outcome,
                    "relevance": c.relevance_score
                }
                for c in similar_cases
            ]
        }


def main():
    parser = argparse.ArgumentParser(
        description="動画テーマから易経の卦を逆引き"
    )
    parser.add_argument("query", nargs="?", help="検索クエリ（テーマまたはフレーズ）")
    parser.add_argument("--theme", "-t", help="テーマキーワードで検索")
    parser.add_argument("--phrase", "-p", help="日常語フレーズで検索")
    parser.add_argument("--brief", "-b", action="store_true", help="動画ブリーフを生成")
    parser.add_argument("--interactive", "-i", action="store_true", help="対話モード")
    parser.add_argument("--json", "-j", action="store_true", help="JSON形式で出力")

    args = parser.parse_args()

    lookup = ReverseLookup()

    if args.interactive:
        print("=" * 60)
        print("易経逆引きツール - 対話モード")
        print("=" * 60)
        print("動画テーマまたは日常の悩みを入力してください。")
        print("終了するには 'exit' または 'quit' と入力。")
        print()

        while True:
            query = input("入力> ").strip()
            if query.lower() in ["exit", "quit", "q"]:
                print("終了します。")
                break

            if not query:
                continue

            # テーマ検索
            result = lookup.lookup_by_theme(query)
            if result:
                print(f"\n【テーマ】{result.theme_name}")
                print(f"【卦】{result.primary_trigram}（{', '.join(result.suggested_hexagrams[:2])}）")
                print(f"【状態】{result.before_state}")
                print(f"【パターン】{result.typical_pattern}")
                print(f"【動画切り口】{result.video_angle}")
                print(f"【Outro】{result.outro_message}")
                print(f"【信頼度】{result.confidence:.0%}")
            else:
                # 日常語検索
                state, trigram, matches = lookup.lookup_by_phrase(query)
                print(f"\n【日常語→易経】")
                print(f"  状態: {state}")
                print(f"  卦: {trigram}")
                if matches:
                    print(f"  マッチ: {matches}")

            # 類似事例
            if result:
                cases = lookup.find_similar_cases(
                    trigram=result.primary_trigram,
                    state=result.before_state,
                    limit=3
                )
                if cases:
                    print(f"\n【類似事例】")
                    for c in cases:
                        print(f"  - {c.target_name}: {c.story_summary[:50]}...")

            print()

    elif args.brief:
        query = args.theme or args.query
        if not query:
            print("エラー: --brief には検索クエリが必要です")
            return

        brief = lookup.generate_video_brief(query)
        print(json.dumps(brief, ensure_ascii=False, indent=2))

    elif args.theme or args.query:
        query = args.theme or args.query
        result = lookup.lookup_by_theme(query)

        if args.json:
            if result:
                print(json.dumps({
                    "theme_id": result.theme_id,
                    "theme_name": result.theme_name,
                    "primary_trigram": result.primary_trigram,
                    "suggested_hexagrams": result.suggested_hexagrams,
                    "before_state": result.before_state,
                    "video_angle": result.video_angle,
                    "outro_message": result.outro_message,
                    "confidence": result.confidence
                }, ensure_ascii=False, indent=2))
            else:
                print(json.dumps({"error": "not found"}, ensure_ascii=False))
        else:
            if result:
                print(f"テーマ: {result.theme_name}")
                print(f"卦: {result.primary_trigram}")
                print(f"推奨卦: {', '.join(result.suggested_hexagrams)}")
                print(f"状態: {result.before_state}")
                print(f"パターン: {result.typical_pattern}")
                print(f"動画切り口: {result.video_angle}")
                print(f"Outro: {result.outro_message}")
                print(f"信頼度: {result.confidence:.0%}")
            else:
                print(f"'{query}' に対応するテーマが見つかりません")

    elif args.phrase:
        state, trigram, matches = lookup.lookup_by_phrase(args.phrase)
        if args.json:
            print(json.dumps({
                "state": state,
                "trigram": trigram,
                "matches": matches
            }, ensure_ascii=False, indent=2))
        else:
            print(f"日常語: {args.phrase}")
            print(f"状態: {state}")
            print(f"卦: {trigram}")
            if matches:
                print(f"マッチ: {', '.join(matches)}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
