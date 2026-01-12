import sqlite3
import os
import itertools

# データベースファイルの絶対パスを指定
DB_FILE = "/Volumes/SSD-PGCU3-A/易経/HaQei未来分岐図　シミュレーター/databasemake/haqei_database.db"

def get_db_connection():
    """データベースへの接続を取得する"""
    if not os.path.exists(DB_FILE):
        print(f"[エラー] データベースファイルが見つかりません: {DB_FILE}")
        print("まず create_database.py を実行して、データベースを構築してください。")
        return None
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def calculate_mbti_points(cursor, mbti_type, scores, helpers):
    """MBTIのポイントを計算"""
    print(f"  - MBTI ({mbti_type}) のポイントを計算中...")
    cursor.execute("SELECT hexagram_id FROM mbti_map WHERE mbti_type = ?", (mbti_type,))
    result = cursor.fetchone()
    if result and result['hexagram_id']:
        cursor.execute("SELECT upper_trigram_id, lower_trigram_id FROM hexagrams_master WHERE hexagram_id = ?", (result['hexagram_id'],))
        hex_data = cursor.fetchone()
        if hex_data and hex_data['upper_trigram_id'] and hex_data['lower_trigram_id']:
            scores[helpers['id_to_trigram'][hex_data['upper_trigram_id']]] += 10
            scores[helpers['id_to_trigram'][hex_data['lower_trigram_id']]] += 10
    return scores

def calculate_enneagram_points(cursor, enneagram_data, scores, helpers):
    """エニアグラムのポイントを計算"""
    print(f"  - エニアグラム のポイントを計算中...")
    for ennea_type, points in enneagram_data.items():
        cursor.execute("SELECT hexagram_id FROM enneagram_map WHERE enneagram_type = ?", (ennea_type,))
        result = cursor.fetchone()
        if result and result['hexagram_id']:
            cursor.execute("SELECT upper_trigram_id, lower_trigram_id FROM hexagrams_master WHERE hexagram_id = ?", (result['hexagram_id'],))
            hex_data = cursor.fetchone()
            if hex_data and hex_data['upper_trigram_id'] and hex_data['lower_trigram_id']:
                upper_name = helpers['id_to_trigram'][hex_data['upper_trigram_id']]
                lower_name = helpers['id_to_trigram'][hex_data['lower_trigram_id']]
                if upper_name == lower_name:
                    scores[upper_name] += points * 2
                else:
                    scores[upper_name] += points
                    scores[lower_name] += points
    return scores

def calculate_sf_points(cursor, sf_data, mbti_type, scores, helpers):
    """SFのポイントを計算（文脈補正ロジック入り）"""
    print(f"  - ストレングスファインダー のポイントを計算中...")
    sf_points_by_rank = [15, 12, 10, 8, 5]
    for i, strength in enumerate(sf_data):
        points = sf_points_by_rank[i]
        
        # INFP向けの文脈補正ロジック
        if mbti_type == 'INFP' and strength == '戦略性':
            scores['巽'] += round(points * 0.75)
            scores['乾'] += round(points * 0.25)
            print(f"    - '{strength}'(INFP) -> 巽, 乾")
            continue
        elif mbti_type == 'INFP' and strength == '最上志向':
            scores['兌'] += round(points * 0.7)
            scores['乾'] += round(points * 0.3)
            print(f"    - '{strength}'(INFP) -> 兌, 乾")
            continue
        elif mbti_type == 'INFP' and strength == '適応性':
            scores['巽'] += points
            print(f"    - '{strength}'(INFP) -> 巽")
            continue
        
        # デフォルトのマッピング
        cursor.execute("SELECT base_trigram_id FROM strengthsfinder_map WHERE strength_name = ?", (strength,))
        result = cursor.fetchone()
        if result and result['base_trigram_id'] in helpers['id_to_trigram']:
            trigram_name = helpers['id_to_trigram'][result['base_trigram_id']]
            scores[trigram_name] += points
            print(f"    - '{strength}' -> {trigram_name}")
    return scores

def generate_bunjin_analysis_text(cursor, final_scores, helpers):
    """分人分析レポートを生成"""
    sorted_scores = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
    top3_trigrams = [item[0] for item in sorted_scores[:3] if item[1] > 0]
    if len(top3_trigrams) < 2: return "（分析に必要なエネルギーが不足しています）"
    
    bunjin_intro = f"あなたの個性は、主に{len(top3_trigrams)}つの「分人」が動かしています。\n\n"
    for i, name in enumerate(top3_trigrams):
        bunjin_intro += f"{i+1}. 【{name}の分人】\n"

    synergy_texts, conflict_texts = [], []
    for source_name, target_name in itertools.permutations(top3_trigrams, 2):
        source_element = helpers['trigram_to_element'][source_name]
        target_element = helpers['trigram_to_element'][target_name]
        cursor.execute("SELECT relationship_type, metaphor_text FROM element_relationships WHERE source_element = ? AND target_element = ?", (source_element, target_element))
        rel = cursor.fetchone()
        if rel:
            text = rel['metaphor_text'].format(source=f"【{source_name}】", target=f"【{target_name}】")
            if rel['relationship_type'] == '相生':
                synergy_texts.append("・ " + text)
            else:
                conflict_texts.append("・ " + text)

    final_text = bunjin_intro
    if synergy_texts:
        final_text += "\n✅ **シナジー（あなたの強みが生まれる流れ）**\n" + "\n".join(synergy_texts)
    if conflict_texts:
        final_text += "\n\n⚠️ **葛藤（あなたが成長するための課題）**\n" + "\n".join(conflict_texts)
    return final_text

def generate_hexagram_profile_text(cursor, final_scores, helpers):
    """64卦プロファイルレポートを生成"""
    sorted_scores = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
    if len(sorted_scores) < 2 or sorted_scores[1][1] == 0: return "あなたを象徴する卦を決定できませんでした。"
    
    upper_trigram_name, lower_trigram_name = sorted_scores[0][0], sorted_scores[1][0]
    upper_trigram_id, lower_trigram_id = helpers['name_to_trigram_id'][upper_trigram_name], helpers['name_to_trigram_id'][lower_trigram_name]
    
    cursor.execute("SELECT * FROM hexagrams_master WHERE upper_trigram_id = ? AND lower_trigram_id = ?", (upper_trigram_id, lower_trigram_id))
    profile_hex = cursor.fetchone()
    
    if profile_hex:
        text = f"あなたの統合的な人格は、64卦の中で**「{profile_hex['hexagram_id']}. {profile_hex['name_jp']}」**として象徴されます。\n\n"
        text += f"これは、あなたのエネルギー1位である**【{upper_trigram_name}】が「外卦（社会に見せる顔）」**、2位の**【{lower_trigram_name}】が「内卦（内面的な本質）」**となる組み合わせです。\n\n"
        text += f"**【{profile_hex['name_jp']}の解説】**\n"
        text += profile_hex['description'] if profile_hex['description'] else "（詳細な解説はデータベースにありません）"
        return text
    else:
        return f"あなたを象徴する卦（上卦：{upper_trigram_name}, 下卦：{lower_trigram_name}）は見つかりませんでした。"

def generate_report_text(cursor, final_scores, helpers):
    """レポート全体のテキストを組み立てる"""
    print("\nレポート生成中...")
    report = {
        'bunjin_analysis': generate_bunjin_analysis_text(cursor, final_scores, helpers),
        'hexagram_profile': generate_hexagram_profile_text(cursor, final_scores, helpers)
    }
    return report

def analyze_user_profile(user_inputs):
    """分析の全プロセスを実行するメイン関数"""
    print("分析を開始します...")
    conn = get_db_connection()
    if conn is None: return None
    cursor = conn.cursor()

    helpers = {
        'id_to_trigram': {row['trigram_id']: row['name_jp'] for row in cursor.execute("SELECT * FROM trigrams_master").fetchall()},
        'trigram_to_element': {row['name_jp']: row['element'] for row in cursor.execute("SELECT * FROM trigrams_master").fetchall()},
        'name_to_trigram_id': {row['name_jp']: row['trigram_id'] for row in cursor.execute("SELECT * FROM trigrams_master").fetchall()}
    }
    
    scores = {'乾': 0, '兌': 0, '離': 0, '震': 0, '巽': 0, '坎': 0, '艮': 0, '坤': 0}
    
    if 'mbti' in user_inputs: scores = calculate_mbti_points(cursor, user_inputs['mbti'], scores, helpers)
    if 'enneagram' in user_inputs: scores = calculate_enneagram_points(cursor, user_inputs['enneagram'], scores, helpers)
    if 'strengths' in user_inputs: scores = calculate_sf_points(cursor, user_inputs['strengths'], user_inputs.get('mbti'), scores, helpers)
    
    final_scores = {k: round(v) for k, v in scores.items()}
    final_report = generate_report_text(cursor, final_scores, helpers)
    
    conn.close()
    print("\n分析が完了しました。")
    return {'scores': final_scores, 'report': final_report}

# --- 実行 ---
if __name__ == '__main__':
    your_profile = {
        "mbti": "INFP",
        "enneagram": {3: 5, 7: 5, 8: 4, 9: 2, 6: 1},
        "strengths": ['未来志向', '戦略性', '最上志向', '原点思考', '適応性']
    }
    
    analysis_result = analyze_user_profile(your_profile)
    
    if analysis_result:
        print("\n\n--- 最終分析レポート ---")
        print("\n【最終スコア】")
        sorted_scores = sorted(analysis_result['scores'].items(), key=lambda item: item[1], reverse=True)
        for name, score in sorted_scores:
            print(f"{name}: {score}")
        print("\n\n【Part A: あなたの人格OSの内部構造（分人分析）】")
        print(analysis_result['report']['bunjin_analysis'])
        print("\n\n【Part B: あなたを象徴する64卦プロファイル】")
        print(analysis_result['report']['hexagram_profile'])