#!/usr/bin/env python3
"""
既存の全事例に変爻情報を追加するスクリプト

全てのケースについて、八卦のペアから変爻を推定し、
changing_lines_1, changing_lines_2, changing_lines_3 フィールドを更新します。
"""
import json
from pathlib import Path
from schema_v3 import Case
from infer_changing_lines import infer_changing_lines

def add_changing_lines_to_cases(input_path: Path, output_path: Path, dry_run: bool = False):
    """
    全事例に変爻情報を追加

    Args:
        input_path: 入力JSONLファイル
        output_path: 出力JSONLファイル
        dry_run: Trueの場合、ファイルを書き込まずに統計のみ表示
    """
    cases_processed = 0
    cases_updated = 0
    errors = []

    updated_cases = []

    print("=" * 80)
    print("変爻情報追加スクリプト")
    print("=" * 80)
    print(f"\n入力ファイル: {input_path}")
    print(f"出力ファイル: {output_path}")
    print(f"ドライラン: {dry_run}\n")

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                case = Case(**data)
                cases_processed += 1

                # 各変化の変爻を推定
                changing_lines_1 = infer_changing_lines(
                    case.before_hex.value,
                    case.trigger_hex.value
                )
                changing_lines_2 = infer_changing_lines(
                    case.trigger_hex.value,
                    case.action_hex.value
                )
                changing_lines_3 = infer_changing_lines(
                    case.action_hex.value,
                    case.after_hex.value
                )

                # 既に変爻情報がある場合はスキップするか、上書きするか判定
                has_existing = (
                    case.changing_lines_1 is not None or
                    case.changing_lines_2 is not None or
                    case.changing_lines_3 is not None
                )

                if has_existing:
                    # 既存の情報がある場合はスキップ
                    if not dry_run:
                        updated_cases.append(case.model_dump())
                    continue

                # 新しい変爻情報を追加
                data["changing_lines_1"] = changing_lines_1
                data["changing_lines_2"] = changing_lines_2
                data["changing_lines_3"] = changing_lines_3

                # 更新されたケースを検証
                updated_case = Case(**data)
                cases_updated += 1

                if not dry_run:
                    updated_cases.append(updated_case.model_dump())

                # サンプル表示（最初の5件のみ）
                if cases_updated <= 5:
                    print(f"\n【事例 {cases_processed}】 {case.target_name}")
                    print(f"  {case.before_hex.value} → {case.trigger_hex.value}: {changing_lines_1}")
                    print(f"  {case.trigger_hex.value} → {case.action_hex.value}: {changing_lines_2}")
                    print(f"  {case.action_hex.value} → {case.after_hex.value}: {changing_lines_3}")

            except Exception as e:
                errors.append(f"行 {line_num}: {str(e)}")
                # エラーがあっても既存のデータはそのまま保持
                if not dry_run:
                    updated_cases.append(data)

    # 結果を出力ファイルに書き込み
    if not dry_run:
        with open(output_path, "w", encoding="utf-8") as f:
            for case_data in updated_cases:
                json.dump(case_data, f, ensure_ascii=False)
                f.write("\n")

    # 統計表示
    print("\n" + "=" * 80)
    print("処理結果")
    print("=" * 80)
    print(f"処理した事例数: {cases_processed}")
    print(f"更新した事例数: {cases_updated}")
    print(f"エラー数: {len(errors)}")

    if errors:
        print("\n【エラー一覧】")
        for error in errors[:10]:  # 最大10件まで表示
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... 他 {len(errors) - 10} 件")

    if dry_run:
        print("\n⚠️ ドライランモードのため、ファイルは更新されていません")
        print(f"実際に更新するには、--apply オプションを指定してください")
    else:
        print(f"\n✅ {output_path} に更新されたデータを書き込みました")

    return cases_processed, cases_updated, len(errors)

def main():
    import sys

    db_path = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"

    # デフォルトはドライラン
    dry_run = "--apply" not in sys.argv

    if dry_run:
        print("\n⚠️ ドライランモード：ファイルは更新されません")
        print("実際に更新するには --apply オプションを指定してください\n")

    # 出力先は同じファイル（上書き）
    output_path = db_path

    # バックアップを作成
    if not dry_run:
        backup_path = db_path.parent / f"cases_backup_{Path(__file__).stem}.jsonl"
        import shutil
        shutil.copy(db_path, backup_path)
        print(f"✅ バックアップ作成: {backup_path}\n")

    # 変爻情報を追加
    processed, updated, errors = add_changing_lines_to_cases(
        db_path,
        output_path,
        dry_run=dry_run
    )

    # 検証
    if not dry_run and errors == 0:
        print("\n" + "=" * 80)
        print("検証中...")
        print("=" * 80)

        from validate_cases import validate_cases
        valid, total, error_list = validate_cases(output_path)

        if valid == total:
            print(f"✅ 全 {total} 件の事例が有効です")
        else:
            print(f"❌ {total} 件中 {valid} 件が有効です")
            print(f"エラー数: {len(error_list)}")

if __name__ == "__main__":
    main()
