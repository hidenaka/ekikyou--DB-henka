import sqlite3
import json
import os

# --- パス設定 ---
# このスクリプトがあるディレクトリ
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 読み込むデータベースのパス
DB_FILE = os.path.join(SCRIPT_DIR, "../databasemake/haqei_database.db")
# 出力するJSONファイルのパス
JSON_OUTPUT_FILE = os.path.join(SCRIPT_DIR, "haqei_database.json")

def convert_db_to_json():
    """SQLiteデータベースを読み込み、全テーブルのデータを単一のJSONファイルに出力する"""
    if not os.path.exists(DB_FILE):
        print(f"[エラー] データベースファイルが見つかりません: {DB_FILE}")
        return

    print(f"データベース '{DB_FILE}' を読み込んでいます...")
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    database_data = {}
    
    tables = [
        "trigrams_master",
        "element_relationships",
        "hexagrams_master",
        "mbti_map",
        "enneagram_map",
        "strengthsfinder_map"
    ]

    print("テーブルデータを変換中...")
    for table_name in tables:
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        # Rowオブジェクトを辞書のリストに変換
        database_data[table_name] = [dict(row) for row in rows]
        print(f" - '{table_name}' テーブル完了 ({len(database_data[table_name])}件)")

    conn.close()

    print(f"\nJSONファイル '{JSON_OUTPUT_FILE}' にデータを書き出しています...")
    # JSONファイルに書き出す
    with open(JSON_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # JavaScriptで読み込みやすいように変数定義を追加
        f.write("const HDB = ")
        # JSONデータを書き込み
        json.dump(database_data, f, ensure_ascii=False, indent=2)
        f.write(";")

    print("\n変換が完了しました。")
    print(f"'{JSON_OUTPUT_FILE}' が作成されました。このファイルをHTMLと同じ場所に置いてください。")


if __name__ == '__main__':
    convert_db_to_json()