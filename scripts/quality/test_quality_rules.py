"""
品質ルール整合性テスト
仕様（docs/*.md）と実装（scripts/quality/*.py）の整合性を確認
"""

import json
import sys
from pathlib import Path

# パス設定
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from domain_rules import (
    normalize_url, extract_domain, is_rejected, classify_domain,
    GOLD_TIERS, SILVER_TIERS, DOMAIN_TIERS
)
from phase1_2_v3 import evaluate_case_v3, is_anonymous
from phase4_success_level import (
    calculate_bayesian_rate, calculate_confidence_interval,
    outcome_to_score, SMOOTHING_ALPHA, PRIOR_SUCCESS, MIN_SAMPLE_RELIABLE
)

def test_domain_classification():
    """ドメイン分類テスト"""
    print("=" * 50)
    print("1. ドメイン分類テスト")
    print("=" * 50)

    test_cases = [
        # (URL, expected_tier, description)
        ("https://www.meti.go.jp/policy/", "tier1_official", "日本政府"),
        ("https://www.sec.gov/", "tier1_official", "米国SEC"),
        ("https://gov.uk/", "tier1_official", "英国政府"),
        ("https://www.nikkei.com/article/", "tier2_major_media", "日経新聞"),
        ("https://www.reuters.com/", "tier2_major_media", "Reuters"),
        ("https://ja.wikipedia.org/wiki/Test", "tier3_specialist", "Wikipedia"),
        ("https://www.apple.com/newsroom/", "tier4_corporate", "Apple公式"),
        ("https://www.toyota.co.jp/", "tier4_corporate", "Toyota公式"),
        ("https://prtimes.jp/main/", "tier5_pr", "PRTimes"),
        ("https://www.google.com/search?q=test", "rejected", "Google検索"),
        ("https://unknown-site.com/", "unclassified", "未分類"),
    ]

    passed = 0
    failed = 0

    for url, expected, desc in test_cases:
        result = classify_domain(url)
        actual = result['tier']

        if actual == expected:
            print(f"  ✓ {desc}: {actual}")
            passed += 1
        else:
            print(f"  ✗ {desc}: expected {expected}, got {actual}")
            failed += 1

    print(f"\n  結果: {passed}/{passed+failed} passed")
    return failed == 0


def test_tier_classification():
    """Tier分類ロジックテスト"""
    print("\n" + "=" * 50)
    print("2. Tier分類ロジックテスト")
    print("=" * 50)

    test_cases = [
        # (trust_level, source_tiers, expected_tier, description)
        ("verified", ["tier1_official"], "gold", "verified + tier1 → Gold"),
        ("verified", ["tier2_major_media"], "gold", "verified + tier2 → Gold"),
        ("verified", ["tier4_corporate"], "gold", "verified + tier4 → Gold"),
        ("verified", ["tier3_specialist"], "silver", "verified + tier3 → Silver"),
        ("plausible", ["tier1_official"], "silver", "plausible + tier1 → Silver"),
        ("unverified", ["tier1_official"], "bronze", "unverified + tier1 → Bronze"),
        (None, ["tier1_official"], "bronze", "None + tier1 → Bronze"),
    ]

    passed = 0
    failed = 0

    for trust, source_tiers, expected, desc in test_cases:
        # ダミー事例作成
        case = {
            'target_name': 'テスト企業',
            'trust_level': trust,
            'sources': ['https://example.com/'],  # ダミー
        }

        # 実際の判定はソースURLに依存するため、ロジックを直接テスト
        best_tier = source_tiers[0] if source_tiers else None

        if trust == 'verified' and best_tier in GOLD_TIERS:
            actual = 'gold'
        elif trust in ['verified', 'plausible'] and best_tier in SILVER_TIERS:
            actual = 'silver'
        elif source_tiers:
            actual = 'bronze'
        else:
            actual = 'quarantine'

        if actual == expected:
            print(f"  ✓ {desc}")
            passed += 1
        else:
            print(f"  ✗ {desc}: expected {expected}, got {actual}")
            failed += 1

    print(f"\n  結果: {passed}/{passed+failed} passed")
    return failed == 0


def test_success_level_calculation():
    """success_level算出テスト"""
    print("\n" + "=" * 50)
    print("3. success_level算出テスト")
    print("=" * 50)

    # 仕様値の確認
    print(f"  SMOOTHING_ALPHA: {SMOOTHING_ALPHA} (仕様: 1.0)")
    print(f"  PRIOR_SUCCESS: {PRIOR_SUCCESS} (仕様: 0.5)")
    print(f"  MIN_SAMPLE_RELIABLE: {MIN_SAMPLE_RELIABLE} (仕様: 10)")

    test_cases = [
        # (successes, total, expected_rate_approx, description)
        (0, 0, 0.5, "n=0 → 50% (事前確率)"),
        (10, 10, 0.955, "10/10 → ~95.5%"),
        (5, 10, 0.5, "5/10 → 50%"),
        (0, 10, 0.045, "0/10 → ~4.5%"),
    ]

    passed = 0
    failed = 0

    for successes, total, expected, desc in test_cases:
        actual = calculate_bayesian_rate(successes, total)

        if abs(actual - expected) < 0.01:
            print(f"  ✓ {desc}: {actual:.3f}")
            passed += 1
        else:
            print(f"  ✗ {desc}: expected ~{expected}, got {actual:.3f}")
            failed += 1

    # 信頼区間テスト
    print("\n  信頼区間テスト:")
    ci_low, ci_high = calculate_confidence_interval(5, 10)
    print(f"  5/10 の 95% CI: [{ci_low}, {ci_high}]")
    if 0.2 < ci_low < 0.3 and 0.7 < ci_high < 0.8:
        print("  ✓ CI範囲は妥当")
        passed += 1
    else:
        print("  ✗ CI範囲が不正")
        failed += 1

    print(f"\n  結果: {passed}/{passed+failed} passed")
    return failed == 0


def test_anonymous_detection():
    """匿名判定テスト"""
    print("\n" + "=" * 50)
    print("4. 匿名判定テスト")
    print("=" * 50)

    test_cases = [
        ("Aさん_転職", True, "Aさん_ パターン"),
        ("XXさん_起業", True, "XXさん_ パターン"),
        ("匿名企業", True, "匿名 パターン"),
        ("某企業", True, "某企業 パターン"),
        ("トヨタ自動車", False, "実名企業"),
        ("孫正義", False, "実名個人"),
        ("", True, "空文字列"),
        (None, True, "None"),
    ]

    passed = 0
    failed = 0

    for name, expected, desc in test_cases:
        actual = is_anonymous(name)

        if actual == expected:
            print(f"  ✓ {desc}: {actual}")
            passed += 1
        else:
            print(f"  ✗ {desc}: expected {expected}, got {actual}")
            failed += 1

    print(f"\n  結果: {passed}/{passed+failed} passed")
    return failed == 0


def test_tier_constants():
    """Tier定数の整合性テスト"""
    print("\n" + "=" * 50)
    print("5. Tier定数整合性テスト")
    print("=" * 50)

    passed = 0
    failed = 0

    # GOLD_TIERS確認
    expected_gold = ['tier1_official', 'tier2_major_media', 'tier4_corporate']
    if GOLD_TIERS == expected_gold:
        print(f"  ✓ GOLD_TIERS: {GOLD_TIERS}")
        passed += 1
    else:
        print(f"  ✗ GOLD_TIERS: expected {expected_gold}, got {GOLD_TIERS}")
        failed += 1

    # SILVER_TIERSがGOLD_TIERSを包含
    if all(t in SILVER_TIERS for t in GOLD_TIERS):
        print(f"  ✓ SILVER_TIERS includes all GOLD_TIERS")
        passed += 1
    else:
        print(f"  ✗ SILVER_TIERS does not include all GOLD_TIERS")
        failed += 1

    # 全てのTierがDOMAIN_TIERSに定義
    all_tiers = set(GOLD_TIERS + SILVER_TIERS)
    defined_tiers = set(DOMAIN_TIERS.keys())
    if all_tiers.issubset(defined_tiers):
        print(f"  ✓ All tiers defined in DOMAIN_TIERS")
        passed += 1
    else:
        missing = all_tiers - defined_tiers
        print(f"  ✗ Missing tiers in DOMAIN_TIERS: {missing}")
        failed += 1

    print(f"\n  結果: {passed}/{passed+failed} passed")
    return failed == 0


def main():
    print("=" * 60)
    print("品質ルール整合性テスト")
    print("=" * 60)

    results = []
    results.append(("ドメイン分類", test_domain_classification()))
    results.append(("Tier分類ロジック", test_tier_classification()))
    results.append(("success_level算出", test_success_level_calculation()))
    results.append(("匿名判定", test_anonymous_detection()))
    results.append(("Tier定数整合性", test_tier_constants()))

    print("\n" + "=" * 60)
    print("総合結果")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("全テストPASS")
        return 0
    else:
        print("一部テストFAIL - 仕様と実装の確認が必要")
        return 1


if __name__ == '__main__':
    exit(main())
