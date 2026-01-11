#!/usr/bin/env python3
"""
テキストから易経変化を抽出・構造化するスクリプト

Claude APIを使用して、生テキストから構造化されたCase形式のJSONを生成する。
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional
import subprocess

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
DATA_DIR = PROJECT_ROOT / "data"

# スキーマのインポート
import sys
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from schema_v3 import Case


def load_extraction_prompt() -> str:
    """抽出プロンプトを読み込む"""
    prompt_file = PROMPTS_DIR / "extraction_prompt.md"
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()


def call_claude_api(prompt: str, text: str) -> Optional[str]:
    """
    Claude APIを呼び出して構造化を実行

    注意: 実際のAPI呼び出しはanthropic SDKを使用するか、
    claude CLIを使用する。ここではサンプル実装。

    Args:
        prompt: システムプロンプト
        text: 入力テキスト

    Returns:
        APIからのレスポンス（JSON文字列）
    """
    # 環境変数からAPIキーを取得
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        print("警告: ANTHROPIC_API_KEY が設定されていません")
        print("手動モードで実行します（プロンプトを表示）")
        return None

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        full_prompt = prompt.replace("{text}", text)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )

        return message.content[0].text

    except ImportError:
        print("anthropic SDK がインストールされていません")
        print("pip install anthropic を実行してください")
        return None
    except Exception as e:
        print(f"API Error: {e}")
        return None


def extract_json_from_response(response: str) -> List[Dict]:
    """
    APIレスポンスからJSONを抽出

    Args:
        response: APIからのレスポンス

    Returns:
        抽出されたJSONデータのリスト
    """
    # JSON配列を探す
    json_match = re.search(r'\[[\s\S]*\]', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # JSONブロックを探す
    code_block_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    print("警告: JSONの抽出に失敗しました")
    return []


def validate_and_fix_case(data: Dict) -> Optional[Dict]:
    """
    抽出されたデータを検証・修正

    Args:
        data: 抽出されたデータ

    Returns:
        検証済みデータ（無効な場合はNone）
    """
    # 必須フィールドのチェック
    required_fields = [
        "target_name", "scale", "period", "story_summary",
        "before_state", "trigger_type", "action_type", "after_state",
        "before_hex", "trigger_hex", "action_hex", "after_hex",
        "pattern_type", "outcome", "source_type", "credibility_rank"
    ]

    for field in required_fields:
        if field not in data or not data[field]:
            print(f"  警告: 必須フィールド '{field}' が不足")
            return None

    # Pydanticモデルで検証
    try:
        case = Case(**data)
        return case.model_dump()
    except Exception as e:
        print(f"  検証エラー: {e}")
        return None


def process_wikipedia_article(article_path: Path) -> List[Dict]:
    """
    Wikipedia記事ファイルを処理

    Args:
        article_path: 記事JSONファイルのパス

    Returns:
        抽出されたCaseのリスト
    """
    with open(article_path, "r", encoding="utf-8") as f:
        article = json.load(f)

    title = article.get("title", "不明")
    text = article.get("career_text", "")

    if not text or len(text) < 100:
        print(f"スキップ: {title}（テキストが短すぎる）")
        return []

    print(f"処理中: {title}")

    # プロンプトを読み込み
    prompt = load_extraction_prompt()

    # AI APIを呼び出し
    response = call_claude_api(prompt, text)

    if not response:
        # 手動モード: プロンプトを表示
        print("\n=== 手動モード ===")
        print("以下のテキストをClaude等に貼り付けて、JSONを取得してください：")
        print("-" * 50)
        print(prompt.replace("{text}", text[:1000] + "...（省略）"))
        print("-" * 50)
        return []

    # レスポンスからJSONを抽出
    cases_data = extract_json_from_response(response)

    # 各ケースを検証
    valid_cases = []
    for i, case_data in enumerate(cases_data):
        print(f"  ケース {i+1} を検証中...")

        # ソース情報を追加
        case_data["source_type"] = "article"  # Wikipediaはarticle
        if "credibility_rank" not in case_data:
            case_data["credibility_rank"] = "A"  # Wikipediaは信頼性A

        validated = validate_and_fix_case(case_data)
        if validated:
            valid_cases.append(validated)
            print(f"  ケース {i+1}: 有効")
        else:
            print(f"  ケース {i+1}: 無効（スキップ）")

    return valid_cases


def process_all_articles(source_dir: Path, output_file: Path) -> int:
    """
    ディレクトリ内の全記事を処理

    Args:
        source_dir: 記事ファイルのディレクトリ
        output_file: 出力先ファイル

    Returns:
        処理した件数
    """
    all_cases = []

    for article_path in source_dir.glob("*.json"):
        if article_path.name.startswith("_"):  # インデックスファイルはスキップ
            continue

        cases = process_wikipedia_article(article_path)
        all_cases.extend(cases)

    # 出力
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for case in all_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"\n完了: {len(all_cases)}件を {output_file} に出力")
    return len(all_cases)


def main():
    """メイン実行"""
    import argparse

    parser = argparse.ArgumentParser(description="テキストから易経変化を抽出")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="入力ファイル（Wikipedia記事JSON）またはディレクトリ"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DATA_DIR / "pending" / "extracted.jsonl",
        help="出力ファイル"
    )
    parser.add_argument(
        "--text", "-t",
        help="直接テキストを指定（テスト用）"
    )

    args = parser.parse_args()

    if args.text:
        # 直接テキストを処理
        prompt = load_extraction_prompt()
        response = call_claude_api(prompt, args.text)
        if response:
            print(response)
        return

    if args.input:
        if args.input.is_dir():
            process_all_articles(args.input, args.output)
        else:
            cases = process_wikipedia_article(args.input)
            for case in cases:
                print(json.dumps(case, ensure_ascii=False, indent=2))
    else:
        print("使用方法:")
        print("  # 単一ファイルを処理")
        print("  python text_to_transitions.py -i article.json")
        print("")
        print("  # ディレクトリ内の全ファイルを処理")
        print("  python text_to_transitions.py -i data/sources/wikipedia/ -o data/pending/extracted.jsonl")
        print("")
        print("  # テキストを直接処理")
        print('  python text_to_transitions.py -t "経歴テキスト..."')


if __name__ == "__main__":
    main()
