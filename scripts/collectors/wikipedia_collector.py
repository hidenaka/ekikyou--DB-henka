#!/usr/bin/env python3
"""
Wikipedia人物記事収集スクリプト（改良版）

Wikipedia APIを使用して人物記事を収集し、
経歴セクションを抽出する。

改良点:
- 人物記事のみをフィルタリング
- 生年・没年の存在チェック
- 記事の品質スコアリング
"""

import json
import time
import re
from pathlib import Path
from typing import Generator, Optional, Dict, List, Tuple
from urllib.parse import quote
import urllib.request
from dataclasses import dataclass, asdict

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "sources" / "wikipedia"


@dataclass
class ArticleInfo:
    """記事情報"""
    title: str
    url: str
    category: str
    career_text: str
    full_text: str
    full_text_length: int
    is_person: bool
    person_type: Optional[str]  # individual, group, concept
    birth_year: Optional[int]
    death_year: Optional[int]
    quality_score: float
    collected_at: str


def make_api_request(url: str) -> Optional[Dict]:
    """API リクエストを実行"""
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'EkikyoDB/1.0 (https://github.com/ekikyo-db; contact@example.com)'
        })
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"API Error: {e}")
        return None


def get_category_members(category: str, limit: int = 500) -> Generator[str, None, None]:
    """
    カテゴリに属するページ一覧を取得（サブカテゴリも展開）
    """
    base_url = "https://ja.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": "50",
        "cmtype": "page|subcat",  # ページとサブカテゴリ両方
        "format": "json"
    }

    count = 0
    continue_token = None
    subcategories = []

    while count < limit:
        if continue_token:
            params["cmcontinue"] = continue_token

        url = base_url + "?" + "&".join(f"{k}={quote(str(v))}" for k, v in params.items())
        data = make_api_request(url)

        if not data:
            break

        for member in data.get("query", {}).get("categorymembers", []):
            if count >= limit:
                break

            if member.get("ns") == 14:  # サブカテゴリ
                subcat_name = member["title"].replace("Category:", "")
                subcategories.append(subcat_name)
            else:  # 通常のページ
                yield member["title"]
                count += 1

        if "continue" in data:
            continue_token = data["continue"].get("cmcontinue")
        else:
            break

        time.sleep(0.3)

    # サブカテゴリも処理（人物カテゴリを優先）
    person_subcats = [s for s in subcategories if any(
        kw in s for kw in ["人物", "実業家", "経営者", "創業者", "起業家"]
    )]

    for subcat in person_subcats[:5]:  # 最大5サブカテゴリ
        if count >= limit:
            break
        print(f"  サブカテゴリ '{subcat}' を展開中...")
        for title in get_category_members(subcat, min(50, limit - count)):
            if count >= limit:
                break
            yield title
            count += 1


def get_page_content(title: str) -> Optional[Dict]:
    """ページの本文とプロパティを取得"""
    base_url = "https://ja.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts|info|categories|pageprops",
        "explaintext": "true",
        "exsectionformat": "plain",
        "inprop": "url",
        "cllimit": "20",
        "format": "json"
    }

    url = base_url + "?" + "&".join(f"{k}={quote(str(v))}" for k, v in params.items())
    data = make_api_request(url)

    if not data:
        return None

    pages = data.get("query", {}).get("pages", {})
    for page_id, page_data in pages.items():
        if page_id == "-1":
            return None

        categories = [c.get("title", "") for c in page_data.get("categories", [])]

        return {
            "title": page_data.get("title", title),
            "url": page_data.get("fullurl", f"https://ja.wikipedia.org/wiki/{quote(title)}"),
            "extract": page_data.get("extract", ""),
            "categories": categories,
            "pageprops": page_data.get("pageprops", {})
        }

    return None


def is_person_article(page_data: Dict) -> Tuple[bool, str]:
    """
    人物記事かどうかを判定

    Returns:
        (is_person, person_type)
        person_type: "individual", "group", "concept", "organization"
    """
    text = page_data.get("extract", "")
    categories = page_data.get("categories", [])
    title = page_data.get("title", "")

    # カテゴリベースの判定
    person_keywords = ["人物", "実業家", "経営者", "政治家", "選手", "俳優", "歌手", "作家"]
    non_person_keywords = ["一覧", "歴史", "会社", "企業", "団体", "概念", "用語"]

    has_person_category = any(
        any(kw in cat for kw in person_keywords) for cat in categories
    )
    has_non_person_category = any(
        any(kw in cat for kw in non_person_keywords) for cat in categories
    )

    # テキストベースの判定
    # 生年パターン
    birth_patterns = [
        r'(\d{4})年\d{1,2}月\d{1,2}日生まれ',
        r'(\d{4})年\d{1,2}月\d{1,2}日\s*[-–−]\s*',
        r'（(\d{4})年\d{1,2}月\d{1,2}日',
        r'(\d{4})年生まれ',
        r'生年月日.*?(\d{4})年',
    ]

    has_birth_year = any(re.search(p, text) for p in birth_patterns)

    # 人物らしい記述パターン
    person_text_patterns = [
        r'は、.*?の(実業家|経営者|政治家|起業家|創業者)',
        r'は、日本の',
        r'である。.*?として',
        r'生まれ。.*?卒業',
    ]

    has_person_text = any(re.search(p, text[:500]) for p in person_text_patterns)

    # 組織記事の判定
    org_patterns = [
        r'は、.*?(株式会社|有限会社|法人|団体|協会|機構)',
        r'本社.*?所在地',
        r'設立.*?\d{4}年',
    ]
    is_org = any(re.search(p, text[:500]) for p in org_patterns) and not has_birth_year

    # 概念記事の判定
    concept_patterns = [
        r'とは、.*?のこと',
        r'とは、.*?を指す',
        r'とは、.*?概念',
    ]
    is_concept = any(re.search(p, text[:300]) for p in concept_patterns)

    # 最終判定
    if is_concept:
        return (False, "concept")
    if is_org:
        return (False, "organization")
    if has_non_person_category and not has_birth_year:
        return (False, "other")
    if has_person_category or has_birth_year or has_person_text:
        return (True, "individual")

    return (False, "unknown")


def extract_birth_death_years(text: str) -> Tuple[Optional[int], Optional[int]]:
    """生年・没年を抽出"""
    birth_year = None
    death_year = None

    # 生年パターン
    birth_patterns = [
        r'（(\d{4})年\d{1,2}月\d{1,2}日\s*[-–−]',
        r'(\d{4})年\d{1,2}月\d{1,2}日生まれ',
        r'(\d{4})年生まれ',
        r'生年月日.*?(\d{4})年',
    ]

    for pattern in birth_patterns:
        match = re.search(pattern, text[:1000])
        if match:
            birth_year = int(match.group(1))
            break

    # 没年パターン
    death_patterns = [
        r'[-–−]\s*(\d{4})年\d{1,2}月\d{1,2}日\s*[）\)]',
        r'(\d{4})年\d{1,2}月\d{1,2}日没',
        r'(\d{4})年没',
    ]

    for pattern in death_patterns:
        match = re.search(pattern, text[:1000])
        if match:
            death_year = int(match.group(1))
            break

    return birth_year, death_year


def extract_career_section(text: str) -> str:
    """テキストから経歴・来歴セクションを抽出"""
    patterns = [
        r"== 経歴 ==\n(.*?)(?=\n== |\Z)",
        r"== 来歴 ==\n(.*?)(?=\n== |\Z)",
        r"== 人物 ==\n(.*?)(?=\n== |\Z)",
        r"== 略歴 ==\n(.*?)(?=\n== |\Z)",
        r"== 生涯 ==\n(.*?)(?=\n== |\Z)",
        r"=== 経歴 ===\n(.*?)(?=\n=== |\n== |\Z)",
        r"=== 来歴 ===\n(.*?)(?=\n=== |\n== |\Z)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # セクションが見つからない場合は最初の部分を返す
    # ただし、目次や脚注を除去
    cleaned = re.sub(r'\n== 脚注 ==.*', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'\n== 関連項目 ==.*', '', cleaned, flags=re.DOTALL)
    return cleaned[:3000]


def calculate_quality_score(article: Dict, career_text: str) -> float:
    """
    記事の品質スコアを計算（0-100）

    評価基準:
    - 経歴テキストの長さ（30点）
    - 生年の存在（20点）
    - 人物カテゴリの存在（20点）
    - 具体的な年号の存在（15点）
    - 会社名・役職の存在（15点）
    """
    score = 0.0

    # 経歴テキストの長さ（30点）
    text_len = len(career_text)
    if text_len >= 2000:
        score += 30
    elif text_len >= 1000:
        score += 20
    elif text_len >= 500:
        score += 10
    elif text_len >= 200:
        score += 5

    # 生年の存在（20点）
    birth_year, death_year = extract_birth_death_years(article.get("extract", ""))
    if birth_year:
        score += 20

    # 人物カテゴリの存在（20点）
    categories = article.get("categories", [])
    person_keywords = ["人物", "実業家", "経営者", "政治家"]
    if any(any(kw in cat for kw in person_keywords) for cat in categories):
        score += 20

    # 具体的な年号の存在（15点）
    year_count = len(re.findall(r'\d{4}年', career_text))
    if year_count >= 10:
        score += 15
    elif year_count >= 5:
        score += 10
    elif year_count >= 2:
        score += 5

    # 会社名・役職の存在（15点）
    role_patterns = [
        r'(社長|会長|取締役|CEO|創業者|代表)',
        r'(株式会社|有限会社)',
        r'(就任|退任|設立|創業)',
    ]
    role_count = sum(len(re.findall(p, career_text)) for p in role_patterns)
    if role_count >= 5:
        score += 15
    elif role_count >= 3:
        score += 10
    elif role_count >= 1:
        score += 5

    return score


def collect_person_articles(
    category: str,
    limit: int = 100,
    min_quality_score: float = 30.0,
    output_dir: Optional[Path] = None
) -> List[ArticleInfo]:
    """
    カテゴリから人物記事のみを収集

    Args:
        category: カテゴリ名
        limit: 取得件数上限（人物記事のみカウント）
        min_quality_score: 最低品質スコア
        output_dir: 出力ディレクトリ

    Returns:
        収集したArticleInfoのリスト
    """
    if output_dir is None:
        output_dir = DATA_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    processed = 0
    skipped_not_person = 0
    skipped_low_quality = 0

    print(f"カテゴリ '{category}' から人物記事を収集中...")
    print(f"目標: {limit}件, 最低品質スコア: {min_quality_score}")
    print("-" * 50)

    for title in get_category_members(category, limit * 3):  # 余裕を持って取得
        if len(results) >= limit:
            break

        processed += 1
        print(f"[{processed}] {title}...", end=" ")

        # ページ取得
        page = get_page_content(title)
        if not page:
            print("取得失敗")
            continue

        # 人物記事判定
        is_person, person_type = is_person_article(page)
        if not is_person:
            print(f"スキップ（{person_type}）")
            skipped_not_person += 1
            continue

        # 経歴抽出
        career_text = extract_career_section(page["extract"])

        # 品質スコア計算
        quality_score = calculate_quality_score(page, career_text)
        if quality_score < min_quality_score:
            print(f"品質不足（{quality_score:.1f}点）")
            skipped_low_quality += 1
            continue

        # 生年・没年抽出
        birth_year, death_year = extract_birth_death_years(page["extract"])

        # ArticleInfo作成
        article_info = ArticleInfo(
            title=page["title"],
            url=page["url"],
            category=category,
            career_text=career_text,
            full_text=page["extract"],
            full_text_length=len(page["extract"]),
            is_person=True,
            person_type=person_type,
            birth_year=birth_year,
            death_year=death_year,
            quality_score=quality_score,
            collected_at=time.strftime("%Y-%m-%d %H:%M:%S")
        )

        results.append(article_info)

        # ファイル保存
        safe_title = re.sub(r'[\\/*?:"<>|]', "_", title)
        output_file = output_dir / f"{safe_title}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(asdict(article_info), f, ensure_ascii=False, indent=2)

        print(f"✓ 品質{quality_score:.1f}点")

        time.sleep(0.5)

    # サマリー
    print("-" * 50)
    print(f"収集完了: {len(results)}件")
    print(f"処理総数: {processed}件")
    print(f"スキップ（非人物）: {skipped_not_person}件")
    print(f"スキップ（品質不足）: {skipped_low_quality}件")

    # インデックスファイル保存
    index_file = output_dir / f"_index_{category.replace(' ', '_')}.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump({
            "category": category,
            "count": len(results),
            "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "min_quality_score": min_quality_score,
            "stats": {
                "processed": processed,
                "skipped_not_person": skipped_not_person,
                "skipped_low_quality": skipped_low_quality
            },
            "articles": [
                {"title": r.title, "quality_score": r.quality_score, "birth_year": r.birth_year}
                for r in results
            ]
        }, f, ensure_ascii=False, indent=2)

    return results


# 収集対象のカテゴリリスト（人物中心）
TARGET_CATEGORIES = [
    "日本の実業家",
    "日本の企業創立者",
    "20世紀日本の実業家",
    "21世紀日本の実業家",
    "アメリカ合衆国の実業家",
    "日本の政治家",
]


def main():
    """メイン実行"""
    import argparse

    parser = argparse.ArgumentParser(description="Wikipedia人物記事収集（改良版）")
    parser.add_argument(
        "--category", "-c",
        default="日本の実業家",
        help="収集するカテゴリ名"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=10,
        help="取得件数上限（人物記事のみカウント）"
    )
    parser.add_argument(
        "--min-quality", "-q",
        type=float,
        default=30.0,
        help="最低品質スコア（0-100）"
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="対象カテゴリ一覧を表示"
    )

    args = parser.parse_args()

    if args.list_categories:
        print("対象カテゴリ一覧:")
        for cat in TARGET_CATEGORIES:
            print(f"  - {cat}")
        return

    collect_person_articles(
        args.category,
        args.limit,
        args.min_quality
    )


if __name__ == "__main__":
    main()
