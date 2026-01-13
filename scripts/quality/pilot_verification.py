#!/usr/bin/env python3
"""
パイロット検証スクリプト

30件のパイロットセットを2名の検証者視点で検証し、
IAA測定用の結果ファイルを出力する。

使用方法:
    python3 scripts/quality/pilot_verification.py

出力:
    data/pilot/reviewer_1_results.jsonl (厳格)
    data/pilot/reviewer_2_results.jsonl (中庸)
"""

import json
import re
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime
from typing import Optional


# ベースパス
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PILOT_FILE = BASE_DIR / "data" / "pilot" / "pilot_verification_30.jsonl"
OUTPUT_DIR = BASE_DIR / "data" / "pilot"


# ソースドメイン分類
PRIMARY_SOURCE_DOMAINS = {
    # 企業公式サイト
    "volkswagen.com", "spacex.com", "aboutamazon.com", "corp.rakuten.co.jp",
    "nissan-global.com", "toyota.co.jp", "who.int",
    # 政府機関
    "bundesregierung.de", "mlit.go.jp", "mext.go.jp", "mhlw.go.jp",
    "meti.go.jp", "soumu.go.jp", "npa.go.jp",
}

SECONDARY_SOURCE_DOMAINS = {
    # ニュースメディア
    "nikkei.com", "asahi.com", "yomiuri.co.jp", "mainichi.jp",
    "reuters.com", "bloomberg.com", "wsj.com", "nytimes.com",
    # 業界団体
    "jada.or.jp", "scaj.org",
}

POINTER_ONLY_DOMAINS = {
    "wikipedia.org", "ja.wikipedia.org", "en.wikipedia.org",
}


def parse_source_url(url: str) -> dict:
    """
    ソースURLを解析し、役割とCOIを判定

    Returns:
        {
            "domain": "example.com",
            "source_role": "primary_source|secondary_source|pointer_only|context_only|rejected",
            "is_search_url": bool,
            "is_accessible": bool (仮定)
        }
    """
    if not url:
        return {
            "domain": None,
            "source_role": "rejected",
            "is_search_url": False,
            "is_accessible": False
        }

    # 検索URLの判定
    is_search_url = (
        "google.com/search" in url or
        "search?q=" in url or
        "bing.com/search" in url
    )

    if is_search_url:
        return {
            "domain": "search",
            "source_role": "rejected",
            "is_search_url": True,
            "is_accessible": False
        }

    # ドメイン抽出
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
    except Exception:
        return {
            "domain": None,
            "source_role": "rejected",
            "is_search_url": False,
            "is_accessible": False
        }

    # ソース役割判定
    if any(d in domain for d in PRIMARY_SOURCE_DOMAINS):
        source_role = "primary_source"
    elif any(d in domain for d in SECONDARY_SOURCE_DOMAINS):
        source_role = "secondary_source"
    elif any(d in domain for d in POINTER_ONLY_DOMAINS):
        source_role = "pointer_only"
    else:
        # 不明なドメインはcontext_onlyとして扱う
        source_role = "context_only"

    return {
        "domain": domain,
        "source_role": source_role,
        "is_search_url": False,
        "is_accessible": True  # 仮定
    }


def check_coi(target_name: str, sources: list) -> str:
    """
    COI（利益相反）ステータスを判定

    Args:
        target_name: 事例の対象名
        sources: ソースURLリスト

    Returns:
        "none" | "self" | "affiliated" | "unknown"
    """
    if not sources:
        return "unknown"

    target_lower = target_name.lower() if target_name else ""

    # 企業名のキーワードを抽出
    company_keywords = []
    # 日本企業
    jp_companies = ["トヨタ", "toyota", "日産", "nissan", "ホンダ", "honda",
                    "ソフトバンク", "softbank", "楽天", "rakuten", "任天堂", "nintendo",
                    "三井", "mitsui", "三菱", "mitsubishi", "volkswagen", "vw",
                    "amazon", "spacex", "openai", "who", "横浜"]

    for keyword in jp_companies:
        if keyword in target_lower:
            company_keywords.append(keyword)

    # 各ソースをチェック
    for source in sources:
        if not source:
            continue
        source_lower = source.lower()

        # 企業名がドメインに含まれているかチェック
        for keyword in company_keywords:
            if keyword in source_lower:
                return "self"

        # 政府機関のソースはCOI noneとして扱う
        if any(gov in source_lower for gov in [".go.jp", ".gov", "bundesregierung"]):
            return "none"

    return "none"


def determine_source_quality(sources: list) -> tuple[str, str]:
    """
    ソースの品質を総合判定

    Returns:
        (best_source_role, verification_confidence)
    """
    if not sources or all(s is None for s in sources):
        return ("rejected", "none")

    roles = []
    for source in sources:
        if source:
            parsed = parse_source_url(source)
            roles.append(parsed["source_role"])

    if not roles:
        return ("rejected", "none")

    # 最良の役割を選択
    role_priority = ["primary_source", "secondary_source", "pointer_only", "context_only", "rejected"]
    best_role = min(roles, key=lambda r: role_priority.index(r) if r in role_priority else 999)

    # 確信度判定
    if best_role == "primary_source":
        confidence = "high"
    elif best_role == "secondary_source":
        confidence = "medium"
    elif best_role == "pointer_only":
        confidence = "low"
    else:
        confidence = "none"

    return (best_role, confidence)


class Reviewer1:
    """
    厳格な検証者

    - 一次ソースがないとverifiedにしない
    - COIに厳しい（企業名がドメインに含まれればself）
    """

    @staticmethod
    def verify(case: dict) -> dict:
        sources = case.get("sources") or []
        target_name = case.get("target_name", "")

        best_role, confidence = determine_source_quality(sources)

        # outcome_status判定（厳格）
        if best_role == "primary_source":
            outcome_status = "verified_correct"
            verification_confidence = "high"
        elif best_role == "secondary_source":
            # 厳格: 二次ソースはmedium確信度でunverified
            outcome_status = "unverified"
            verification_confidence = "medium"
        else:
            outcome_status = "unverified"
            verification_confidence = confidence

        # COI判定（厳格）
        coi_status = "unknown"
        for source in sources:
            if source:
                parsed = parse_source_url(source)
                domain = parsed.get("domain", "")

                # 企業名がドメインに含まれるかを厳密にチェック
                target_words = re.findall(r'[a-zA-Z]+', target_name.lower())
                for word in target_words:
                    if len(word) >= 4 and word in domain:
                        coi_status = "self"
                        break

                # 政府機関はnone
                if any(gov in domain for gov in [".go.jp", ".gov", "who.int", "bundesregierung"]):
                    coi_status = "none"

        if coi_status == "unknown" and sources:
            # デフォルトはnoneに
            coi_status = "none"

        return {
            "pilot_id": case.get("pilot_id"),
            "outcome_status": outcome_status,
            "verification_confidence": verification_confidence,
            "coi_status": coi_status,
            "reviewer_notes": "厳格判定: 一次ソース必須"
        }


class Reviewer2:
    """
    中庸な検証者

    - 二次ソース（新聞等）でもverified_correctとする場合あり
    - COIは明確な場合のみself
    """

    @staticmethod
    def verify(case: dict) -> dict:
        sources = case.get("sources") or []
        target_name = case.get("target_name", "")

        best_role, confidence = determine_source_quality(sources)

        # outcome_status判定（中庸）
        if best_role == "primary_source":
            outcome_status = "verified_correct"
            verification_confidence = "high"
        elif best_role == "secondary_source":
            # 中庸: 信頼性の高い二次ソースならverified_correct
            outcome_status = "verified_correct"
            verification_confidence = "medium"
        elif best_role == "pointer_only":
            outcome_status = "unverified"
            verification_confidence = "low"
        else:
            outcome_status = "unverified"
            verification_confidence = "none"

        # COI判定（中庸）
        coi_status = "none"
        for source in sources:
            if source:
                parsed = parse_source_url(source)
                domain = parsed.get("domain", "")

                # 明確に企業公式サイトの場合のみself
                if any(corp in domain for corp in ["corp.", "ir.", "investor", "aboutamazon", "volkswagen", "spacex", "nissan-global", "rakuten"]):
                    coi_status = "self"
                    break

        return {
            "pilot_id": case.get("pilot_id"),
            "outcome_status": outcome_status,
            "verification_confidence": verification_confidence,
            "coi_status": coi_status,
            "reviewer_notes": "中庸判定: 二次ソースも信頼"
        }


def load_pilot_cases(filepath: Path) -> list[dict]:
    """パイロット事例を読み込む"""
    cases = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def save_results(results: list[dict], filepath: Path):
    """結果をJSONL形式で保存"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def main():
    print("=== パイロット検証 ===")
    print(f"入力: {PILOT_FILE}")

    # パイロット事例を読み込む
    cases = load_pilot_cases(PILOT_FILE)
    print(f"事例数: {len(cases)}件")

    # Reviewer 1 (厳格) による検証
    print("\n--- Reviewer 1 (厳格) ---")
    r1_results = []
    for case in cases:
        result = Reviewer1.verify(case)
        r1_results.append(result)
        print(f"  {result['pilot_id']}: {result['outcome_status']} / {result['verification_confidence']} / {result['coi_status']}")

    # Reviewer 2 (中庸) による検証
    print("\n--- Reviewer 2 (中庸) ---")
    r2_results = []
    for case in cases:
        result = Reviewer2.verify(case)
        r2_results.append(result)
        print(f"  {result['pilot_id']}: {result['outcome_status']} / {result['verification_confidence']} / {result['coi_status']}")

    # 結果を保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    r1_path = OUTPUT_DIR / "reviewer_1_results.jsonl"
    r2_path = OUTPUT_DIR / "reviewer_2_results.jsonl"

    save_results(r1_results, r1_path)
    save_results(r2_results, r2_path)

    print(f"\n=== 出力完了 ===")
    print(f"  Reviewer 1: {r1_path}")
    print(f"  Reviewer 2: {r2_path}")

    # サマリー
    r1_verified = sum(1 for r in r1_results if r['outcome_status'] == 'verified_correct')
    r2_verified = sum(1 for r in r2_results if r['outcome_status'] == 'verified_correct')
    print(f"\n=== サマリー ===")
    print(f"  Reviewer 1 verified_correct: {r1_verified}/{len(cases)} ({r1_verified/len(cases)*100:.1f}%)")
    print(f"  Reviewer 2 verified_correct: {r2_verified}/{len(cases)} ({r2_verified/len(cases)*100:.1f}%)")


if __name__ == "__main__":
    main()
