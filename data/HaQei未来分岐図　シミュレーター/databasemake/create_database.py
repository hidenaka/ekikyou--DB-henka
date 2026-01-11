import sqlite3
import csv
import os

# --- スクリプトとデータベースファイルのパス設定 ---
# スクリプトが置かれているディレクトリを取得
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# データベースを出力するフォルダを指定
OUTPUT_DIR = "/Volumes/SSD-PGCU3-A/易経/HaQei未来分岐図　シミュレーター/databasemake"
# データベースファイルのフルパス
DB_FILE = os.path.join(OUTPUT_DIR, "haqei_database.db")

# --- CSVファイル名の定義 ---
CSV_FILES = {
    'master_info': '1.csv', # 卦のID, 正式名称, キーワード等を含むファイル
    'features':    '2.csv', # 卦の性格等の詳細情報ファイル
    'mbti':        '3.csv', # MBTI対応表
    'enneagram':   '4.csv', # エニアグラム対応表
    'sf':          '5.csv', # ストレングスファインダー対応表
}

# ★★★★★【定義1】六十四卦の構成（上卦・下卦） ★★★★★
# 易経の普遍的な定義であり、CSVファイルに依存しないようにするため、コード内に直接定義します。
# 形式 -> key: 卦番号, value: (上卦のID, 下卦のID)
HEXAGRAM_COMPOSITION = {
    1: (1, 1), 2: (8, 8), 3: (6, 4), 4: (7, 6), 5: (6, 1), 6: (1, 6), 7: (8, 6), 8: (6, 8),
    9: (5, 1), 10: (1, 2), 11: (8, 1), 12: (1, 8), 13: (1, 3), 14: (3, 1), 15: (8, 7), 16: (4, 8),
    17: (2, 4), 18: (7, 5), 19: (8, 2), 20: (5, 8), 21: (3, 4), 22: (7, 3), 23: (7, 8), 24: (8, 4),
    25: (1, 4), 26: (7, 1), 27: (7, 4), 28: (2, 5), 29: (6, 6), 30: (3, 3), 31: (2, 7), 32: (4, 5),
    33: (1, 7), 34: (4, 1), 35: (3, 8), 36: (8, 3), 37: (5, 3), 38: (3, 2), 39: (6, 7), 40: (4, 6),
    41: (7, 2), 42: (5, 4), 43: (2, 1), 44: (1, 5), 45: (2, 8), 46: (8, 5), 47: (2, 6), 48: (6, 5),
    49: (2, 3), 50: (3, 5), 51: (4, 4), 52: (7, 7), 53: (5, 7), 54: (4, 2), 55: (4, 3), 56: (3, 7),
    57: (5, 5), 58: (2, 2), 59: (5, 6), 60: (6, 2), 61: (5, 2), 62: (4, 7), 63: (6, 3), 64: (3, 6)
}


def create_database():
    """
    各種CSVファイルを読み込み、SQLiteデータベースを構築する。
    """
    print(f"データベースの出力先フォルダ: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print("テーブルを作成中...")
    # ★★★★★【定義2】八卦マスターテーブル (trigrams_master) ★★★★★
    # ご指摘の通り「乾→天」「兌→沢」などの対応関係をここで定義します。
    # 形式 -> (ID, 八卦の文字, 自然の象徴, 五行)
    trigrams_data = [
        (1, '乾', '天', '金'), (2, '兌', '沢', '金'), (3, '離', '火', '火'), (4, '震', '雷', '木'),
        (5, '巽', '風', '木'), (6, '坎', '水', '水'), (7, '艮', '山', '土'), (8, '坤', '地', '土')
    ]
    cursor.execute('CREATE TABLE trigrams_master (trigram_id INTEGER PRIMARY KEY, name_jp TEXT, name_en TEXT, element TEXT)')
    cursor.executemany('INSERT INTO trigrams_master VALUES (?, ?, ?, ?)', trigrams_data)

    # --- 五行の関係性テーブル (element_relationships) ---
    metaphors = [
        ('木', '火', '相生', '{source}が持つ木のエネルギーは、{target}が持つ火の炎を燃え上がらせます。'), ('火', '土', '相生', '{source}が持つ火のエネルギーは、{target}が持つ土の大地を温め、育みます。'),
        ('土', '金', '相生', '{source}が持つ土のエネルギーは、{target}から金という貴い鉱石を産み出します。'), ('金', '水', '相生', '{source}が持つ金のエネルギーの表面には、{target}という清らかな水滴が生まれます。'),
        ('水', '木', '相生', '{source}が持つ水のエネルギーは、{target}という若木を育む、生命の源です。'), ('木', '土', '相剋', '{source}が持つ木のエネルギーは、{target}が持つ土の養分を吸い上げ、その力を抑制します。'),
        ('土', '水', '相剋', '{source}が持つ土のエネルギーは、{target}が持つ水の流れを堰き止め、その自由を奪います。'), ('水', '火', '相剋', '{source}が持つ水のエネルギーは、{target}が持つ火の炎を消し止め、その輝きを失わせます。'),
        ('火', '金', '相剋', '{source}が持つ火のエネルギーは、{target}が持つ金の金属を溶かし、その形を変えてしまいます。'), ('金', '木', '相剋', '{source}が持つ金のエネルギーは、{target}が持つ木の若木を切り倒す斧にもなります。')
    ]
    cursor.execute('CREATE TABLE element_relationships (relationship_id INTEGER PRIMARY KEY, source_element TEXT, target_element TEXT, relationship_type TEXT, metaphor_text TEXT)')
    cursor.executemany('INSERT INTO element_relationships VALUES (NULL, ?, ?, ?, ?)', metaphors)

    # --- 空テーブルを作成 ---
    cursor.execute('CREATE TABLE hexagrams_master (hexagram_id INTEGER PRIMARY KEY, name_jp TEXT, upper_trigram_id INTEGER, lower_trigram_id INTEGER, description TEXT, keywords TEXT)')
    cursor.execute('CREATE TABLE mbti_map (mbti_type TEXT PRIMARY KEY, hexagram_id INTEGER)')
    cursor.execute('CREATE TABLE enneagram_map (enneagram_type INTEGER PRIMARY KEY, hexagram_id INTEGER)')
    cursor.execute('CREATE TABLE strengthsfinder_map (strength_name TEXT PRIMARY KEY, base_trigram_id INTEGER)')

    print("CSVデータをインポート中...")

    try:
        # 八卦の名前（'乾'や'天'など）とIDを対応付ける辞書を作成
        trigram_name_map = {name.strip(): id for id, name, en, ele in trigrams_data}
        trigram_name_map.update({en.strip(): id for id, name, en, ele in trigrams_data})

        # --- CSVからのデータ読み込みとテーブル構築ロジック ---
        with open(os.path.join(SCRIPT_DIR, CSV_FILES['features']), 'r', encoding='utf-8') as f_feat, \
             open(os.path.join(SCRIPT_DIR, CSV_FILES['master_info']), 'r', encoding='utf-8') as f_master:

            features_reader = csv.reader(f_feat); next(features_reader, None)
            master_reader = csv.reader(f_master); next(master_reader, None)

            # 2.csv から、卦IDをキーに「性格」の情報を取得 (説明文として使用)
            # CSVの列: [0:卦番号, 5:性格]
            features_data = {row[0].strip(): row[5].strip() for row in features_reader if row and len(row) > 5 and row[0].isdigit()}

            # 1.csv から、卦IDをキーに「卦名」と「キーワード」を取得
            # CSVの列: [0:卦番号, 1:名前, 6:卦のキーワード]
            master_info_map = {row[0].strip(): {'name': row[1].strip(), 'keywords': row[6].strip()} for row in master_reader if row and len(row) > 6 and row[0].isdigit()}

        print("hexagrams_master テーブルを構築中...")
        inserted_count = 0
        
        # 1.csvから取得した master_info_map を元にループ
        for hex_id_str, info in master_info_map.items():
            hex_id = int(hex_id_str)
            
            # 1.csv から卦名とキーワードを取得
            hex_name = info.get('name', '')
            keywords = info.get('keywords', '')
            
            # 2.csvから説明(性格)を取得
            description = features_data.get(hex_id_str, '')

            # 【定義1】で作成した辞書から上卦・下卦IDを直接取得
            composition = HEXAGRAM_COMPOSITION.get(hex_id)
            if composition is None:
                print(f"  [診断エラー] 卦ID {hex_id} の構成情報が定義されていません。スキップします。")
                continue
            
            upper_id, lower_id = composition

            cursor.execute("INSERT INTO hexagrams_master VALUES (?, ?, ?, ?, ?, ?)", (hex_id, hex_name, upper_id, lower_id, description, keywords))
            inserted_count += 1

        print(f"  - hexagrams_master に {inserted_count} 件のデータを挿入しました。")
        conn.commit()
        
        # --- MBTI, エニアグラム, ストレングスファインダーのデータをインポート (ここは変更なし) ---
        hex_name_to_id = {row['name_jp'].strip(): row['hexagram_id'] for row in cursor.execute('SELECT name_jp, hexagram_id FROM hexagrams_master').fetchall()}
        
        for key, filename in {'mbti': CSV_FILES['mbti'], 'enneagram': CSV_FILES['enneagram']}.items():
            data_to_insert = []
            with open(os.path.join(SCRIPT_DIR, filename), 'r', encoding='utf-8') as f:
                reader = csv.reader(f); next(reader, None)
                for row in reader:
                    if not row or len(row) < 3 or not row[0]: continue
                    id_key, value_str = row[0].strip(), row[2].strip()
                    hex_id = hex_name_to_id.get(value_str)
                    if hex_id:
                        data_to_insert.append((id_key, hex_id))
            if data_to_insert:
                cursor.executemany(f'INSERT INTO {key}_map VALUES (?,?)', data_to_insert)
            print(f"  - {filename} -> {key}_map ... 完了")

        with open(os.path.join(SCRIPT_DIR, CSV_FILES['sf']), 'r', encoding='utf-8') as f:
            reader = csv.reader(f); next(reader, None)
            data = [(row[0].strip(), trigram_name_map.get(row[1].strip())) for row in reader if row and len(row) > 1 and trigram_name_map.get(row[1].strip())]
            if data:
                cursor.executemany('INSERT INTO strengthsfinder_map VALUES (?,?)', data)
            print(f"  - {CSV_FILES['sf']} -> strengthsfinder_map ... 完了")

    except FileNotFoundError as e:
        print(f"[致命的エラー] ファイルが見つかりません。ファイル名が正しいか、スクリプトと同じ場所にあるか確認してください: {e.filename}")
    except Exception as e:
        print(f"[致命的エラー] データベース構築中に予期せぬエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

    finally:
        conn.commit()
        conn.close()
        print(f"\nデータベース '{DB_FILE}' の構築が完了しました。")


if __name__ == '__main__':
    create_database()