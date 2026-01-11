#!/usr/bin/env python3
"""
事例と64卦・爻の関連付けを強化するスクリプト

各事例のstory_summaryと64卦・爻の意味を関連付け、
yao_analysisに以下のフィールドを追加:
- hexagram_relation: 卦と事例の関連性解説
- yao_relation: 爻と事例の関連性解説
- transformation_meaning: 変化（before→after）の易経的解釈

スケール対応版:
hexagram_master.jsonを読み込み、事例のscaleに応じた解釈を使用
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

# スクリプトのディレクトリを取得
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# hexagram_master.jsonのパス
HEXAGRAM_MASTER_PATH = PROJECT_ROOT / "data" / "hexagrams" / "hexagram_master.json"

# 有効なscale値
VALID_SCALES = {"company", "individual", "family", "country", "other"}

# hexagram_master.jsonを読み込み
def load_hexagram_master() -> Dict:
    """hexagram_master.jsonを読み込む"""
    if HEXAGRAM_MASTER_PATH.exists():
        with open(HEXAGRAM_MASTER_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# グローバル変数としてロード
HEXAGRAM_MASTER = load_hexagram_master()

# 64卦の基本情報（フォールバック用、hexagram_master.jsonがない場合に使用）
HEXAGRAM_INFO_FALLBACK = {
    1: {"name": "乾為天", "keyword": "創造・剛健", "meaning": "積極的前進、リーダーシップ"},
    2: {"name": "坤為地", "keyword": "受容・柔順", "meaning": "着実な成長、サポート役"},
    3: {"name": "水雷屯", "keyword": "産みの苦しみ", "meaning": "困難な始まり、忍耐"},
    4: {"name": "山水蒙", "keyword": "蒙昧・教育", "meaning": "学び、指導を受ける"},
    5: {"name": "水天需", "keyword": "待機・準備", "meaning": "時機を待つ、準備期間"},
    6: {"name": "天水訟", "keyword": "争い・訴訟", "meaning": "対立、紛争の解決"},
    7: {"name": "地水師", "keyword": "軍隊・組織", "meaning": "統率、組織力"},
    8: {"name": "水地比", "keyword": "親和・協力", "meaning": "連携、仲間との協力"},
    9: {"name": "風天小畜", "keyword": "小さな蓄積", "meaning": "少しずつ貯める"},
    10: {"name": "天沢履", "keyword": "礼・実践", "meaning": "慎重に進む、礼節"},
    11: {"name": "地天泰", "keyword": "平和・繁栄", "meaning": "調和、安定期"},
    12: {"name": "天地否", "keyword": "閉塞・停滞", "meaning": "行き詰まり、沈滞"},
    13: {"name": "天火同人", "keyword": "協調・同志", "meaning": "志を同じくする仲間"},
    14: {"name": "火天大有", "keyword": "大いなる所有", "meaning": "繁栄、成功"},
    15: {"name": "地山謙", "keyword": "謙虚・謙遜", "meaning": "控えめな態度"},
    16: {"name": "雷地豫", "keyword": "喜び・準備", "meaning": "楽しみ、事前準備"},
    17: {"name": "沢雷随", "keyword": "従う・適応", "meaning": "流れに従う"},
    18: {"name": "山風蠱", "keyword": "腐敗・改革", "meaning": "古いものの刷新"},
    19: {"name": "地沢臨", "keyword": "臨む・監督", "meaning": "指導的立場"},
    20: {"name": "風地観", "keyword": "観察・洞察", "meaning": "じっくり見る"},
    21: {"name": "火雷噬嗑", "keyword": "噛み砕く", "meaning": "障害の除去"},
    22: {"name": "山火賁", "keyword": "飾り・文化", "meaning": "外見を整える"},
    23: {"name": "山地剥", "keyword": "剥落・衰退", "meaning": "崩れ落ちる"},
    24: {"name": "地雷復", "keyword": "復帰・回復", "meaning": "原点回帰"},
    25: {"name": "天雷无妄", "keyword": "無妄・誠実", "meaning": "偽りのない態度"},
    26: {"name": "山天大畜", "keyword": "大きな蓄積", "meaning": "力を蓄える"},
    27: {"name": "山雷頤", "keyword": "養い・養生", "meaning": "育てる、養う"},
    28: {"name": "沢風大過", "keyword": "過剰・無理", "meaning": "行き過ぎ、負担"},
    29: {"name": "坎為水", "keyword": "危険・試練", "meaning": "困難の連続"},
    30: {"name": "離為火", "keyword": "明るさ・依存", "meaning": "輝き、付着"},
    31: {"name": "沢山咸", "keyword": "感応・交流", "meaning": "相互作用"},
    32: {"name": "雷風恒", "keyword": "恒常・持続", "meaning": "変わらないこと"},
    33: {"name": "天山遯", "keyword": "退却・隠遁", "meaning": "退く、避ける"},
    34: {"name": "雷天大壮", "keyword": "大いなる力", "meaning": "勢いがある"},
    35: {"name": "火地晋", "keyword": "進む・昇進", "meaning": "前進、出世"},
    36: {"name": "地火明夷", "keyword": "明るさを傷つける", "meaning": "才能を隠す"},
    37: {"name": "風火家人", "keyword": "家庭・組織", "meaning": "内部の調和"},
    38: {"name": "火沢睽", "keyword": "背反・対立", "meaning": "意見の相違"},
    39: {"name": "水山蹇", "keyword": "困難・険阻", "meaning": "前進が困難"},
    40: {"name": "雷水解", "keyword": "解放・解決", "meaning": "困難からの脱出"},
    41: {"name": "山沢損", "keyword": "損・減らす", "meaning": "損して得取れ"},
    42: {"name": "風雷益", "keyword": "益・増やす", "meaning": "利益、増加"},
    43: {"name": "沢天夬", "keyword": "決断・決裂", "meaning": "決然と断つ"},
    44: {"name": "天風姤", "keyword": "出会い・遭遇", "meaning": "思わぬ出会い"},
    45: {"name": "沢地萃", "keyword": "集合・集結", "meaning": "人が集まる"},
    46: {"name": "地風升", "keyword": "上昇・昇進", "meaning": "着実な上昇"},
    47: {"name": "沢水困", "keyword": "困窮・苦境", "meaning": "行き詰まり"},
    48: {"name": "水風井", "keyword": "井戸・源泉", "meaning": "変わらぬ価値"},
    49: {"name": "沢火革", "keyword": "革命・変革", "meaning": "抜本的変化"},
    50: {"name": "火風鼎", "keyword": "鼎・新体制", "meaning": "新しい秩序"},
    51: {"name": "震為雷", "keyword": "衝撃・覚醒", "meaning": "突然の変化"},
    52: {"name": "艮為山", "keyword": "停止・静止", "meaning": "止まること"},
    53: {"name": "風山漸", "keyword": "漸進・段階", "meaning": "徐々に進む"},
    54: {"name": "雷沢帰妹", "keyword": "従属・副次", "meaning": "副次的立場"},
    55: {"name": "雷火豊", "keyword": "豊か・繁栄", "meaning": "最盛期"},
    56: {"name": "火山旅", "keyword": "旅・流浪", "meaning": "旅人の心得"},
    57: {"name": "巽為風", "keyword": "従順・浸透", "meaning": "風のように浸透"},
    58: {"name": "兌為沢", "keyword": "喜び・悦楽", "meaning": "喜びの表現"},
    59: {"name": "風水渙", "keyword": "散らす・解散", "meaning": "散開、拡散"},
    60: {"name": "水沢節", "keyword": "節度・制限", "meaning": "適度な制約"},
    61: {"name": "風沢中孚", "keyword": "誠・信頼", "meaning": "真心、信頼"},
    62: {"name": "雷山小過", "keyword": "小さな過ち", "meaning": "控えめに"},
    63: {"name": "水火既済", "keyword": "完成・達成", "meaning": "一応の完成"},
    64: {"name": "火水未済", "keyword": "未完成", "meaning": "まだ終わらない"},
}


def get_hexagram_info(hex_id: int) -> Dict:
    """卦情報を取得（hexagram_master.jsonから取得、なければフォールバック）"""
    hex_key = str(hex_id)
    if hex_key in HEXAGRAM_MASTER:
        master_info = HEXAGRAM_MASTER[hex_key]
        return {
            "name": master_info.get("name", ""),
            "keyword": master_info.get("keyword", ""),
            "meaning": master_info.get("meaning", ""),
            "interpretations": master_info.get("interpretations", {})
        }
    return HEXAGRAM_INFO_FALLBACK.get(hex_id, {})


def get_scale_interpretation(hex_id: int, scale: str) -> str:
    """scaleに応じた卦の解釈を取得"""
    hex_key = str(hex_id)

    # scaleの正規化（無効な値はotherにフォールバック）
    if scale not in VALID_SCALES:
        scale = "other"

    # hexagram_master.jsonから取得
    if hex_key in HEXAGRAM_MASTER:
        interpretations = HEXAGRAM_MASTER[hex_key].get("interpretations", {})
        if scale in interpretations:
            return interpretations[scale]
        # scaleに対応する解釈がない場合はotherにフォールバック
        return interpretations.get("other", "")

    # フォールバック（旧来のmeaning）
    fallback_info = HEXAGRAM_INFO_FALLBACK.get(hex_id, {})
    return fallback_info.get("meaning", "")

# 爻位置の意味
YAO_STAGES = {
    1: {"stage": "発芽期・始動期", "meaning": "始まり、潜伏、準備段階"},
    2: {"stage": "成長期・発展期", "meaning": "成長、発展、力を蓄える"},
    3: {"stage": "転換期・岐路", "meaning": "危機、選択、分岐点"},
    4: {"stage": "進展期・前進期", "meaning": "進展、新局面、転機"},
    5: {"stage": "全盛期・成熟期", "meaning": "最盛、達成、リーダーの位置"},
    6: {"stage": "終末期・過熱期", "meaning": "終わり、過剰、次への移行"},
}


def generate_hexagram_relation(case: Dict, hex_id: int) -> str:
    """事例と卦の関連性を生成（スケール対応版）"""
    hex_info = get_hexagram_info(hex_id)
    if not hex_info:
        return ""

    target = case.get('target_name', '')
    pattern = case.get('pattern_type', '')
    outcome = case.get('outcome', '')
    scale = case.get('scale', 'other')

    keyword = hex_info.get('keyword', '')

    # スケールに応じた解釈を取得
    scale_interpretation = get_scale_interpretation(hex_id, scale)

    # スケールに応じた解釈がある場合はそれを使用
    if scale_interpretation:
        relation = f"「{keyword}」の象意：{scale_interpretation}"
    else:
        # フォールバック
        meaning = hex_info.get('meaning', '')
        relation = f"「{keyword}」の象意が示す通り、{meaning}の状況。"

    # パターンに応じた解釈を追加
    pattern_suffix = ""
    if pattern == 'Shock_Recovery':
        pattern_suffix = "衝撃からの回復過程を象徴。"
    elif pattern == 'Hubris_Collapse':
        pattern_suffix = "過信による崩壊の教訓を示す。"
    elif pattern == 'Steady_Growth':
        pattern_suffix = "着実な成長の道筋を表す。"
    elif pattern == 'Crisis_Pivot':
        pattern_suffix = "危機を転機に変える力を示唆。"
    elif pattern == 'Breakthrough':
        pattern_suffix = "突破口を開く可能性を示す。"
    elif pattern == 'Gradual_Decline':
        pattern_suffix = "徐々に衰退する過程を示す。"
    elif pattern == 'Stagnation':
        pattern_suffix = "停滞状態からの脱却が課題。"
    elif pattern == 'Quiet_Fade':
        pattern_suffix = "静かに姿を消す過程を示す。"

    if pattern_suffix:
        relation += f" {pattern_suffix}"

    return relation


def generate_yao_relation(case: Dict, hex_id: int, yao_pos: int) -> str:
    """事例と爻の関連性を生成"""
    yao_info = YAO_STAGES.get(yao_pos, {})
    if not yao_info:
        return ""

    stage = yao_info.get('stage', '')
    meaning = yao_info.get('meaning', '')
    action = case.get('action_type', '')

    relation = f"第{yao_pos}爻（{stage}）の位置。{meaning}の段階にあり、"

    # 行動との関連
    if action == '攻める・挑戦':
        relation += "積極的な行動が求められた時期。"
    elif action == '守る・維持':
        relation += "守りを固めることが重要だった局面。"
    elif action == '耐える・潜伏':
        relation += "忍耐と待機が必要な状況。"
    elif action == '刷新・破壊':
        relation += "抜本的な変革を迫られた転換点。"
    elif action == '捨てる・撤退':
        relation += "撤退の判断が求められた局面。"
    else:
        relation += "適切な対応が求められた局面。"

    return relation


def generate_transformation_meaning(case: Dict) -> str:
    """変化の易経的解釈を生成"""
    before = case.get('before_hex', '')
    trigger = case.get('trigger_hex', '')
    action = case.get('action_hex', '')
    after = case.get('after_hex', '')

    # 八卦の意味
    trigram_meanings = {
        '乾': '創造・天・父',
        '坤': '受容・地・母',
        '震': '動き・雷・長男',
        '巽': '浸透・風・長女',
        '坎': '危険・水・中男',
        '離': '明晰・火・中女',
        '艮': '停止・山・少男',
        '兌': '喜悦・沢・少女',
    }

    before_meaning = trigram_meanings.get(before, before)
    trigger_meaning = trigram_meanings.get(trigger, trigger)
    after_meaning = trigram_meanings.get(after, after)

    interpretation = f"内なる状態（{before}={before_meaning}）に、"
    interpretation += f"外からの影響（{trigger}={trigger_meaning}）が作用し、"
    interpretation += f"結果として（{after}={after_meaning}）へと変化した。"

    return interpretation


def enrich_case(case: Dict) -> Dict:
    """事例を64卦・爻情報で強化（スケール対応版）"""
    yao = case.get('yao_analysis', {})
    if not yao or not isinstance(yao, dict):
        return case

    hex_id = yao.get('before_hexagram_id')
    yao_pos = yao.get('before_yao_position')
    scale = case.get('scale', 'other')

    if hex_id:
        # 卦名を追加（hexagram_master.jsonから取得）
        hex_info = get_hexagram_info(hex_id)
        if hex_info:
            yao['hexagram_name'] = hex_info.get('name', '')
            yao['hexagram_keyword'] = hex_info.get('keyword', '')
            # スケール別の解釈も追加
            yao['scale_interpretation'] = get_scale_interpretation(hex_id, scale)

        # 卦との関連性を追加（スケールを考慮）
        yao['hexagram_relation'] = generate_hexagram_relation(case, hex_id)

    if hex_id and yao_pos:
        # 爻との関連性を追加
        yao['yao_relation'] = generate_yao_relation(case, hex_id, yao_pos)

    # 変化の解釈を追加
    yao['transformation_meaning'] = generate_transformation_meaning(case)

    case['yao_analysis'] = yao
    return case


def process_all_cases(input_file: str, output_file: str):
    """全事例を処理（スケール対応版）"""
    # hexagram_master.jsonのロード状況を確認
    if HEXAGRAM_MASTER:
        print(f"hexagram_master.json: {len(HEXAGRAM_MASTER)}卦のデータをロード済み")
    else:
        print("警告: hexagram_master.jsonがロードされていません。フォールバックを使用します。")

    with open(input_file, 'r', encoding='utf-8') as f:
        cases = [json.loads(line) for line in f if line.strip()]

    enriched = 0
    scale_counts = {scale: 0 for scale in VALID_SCALES}

    for i, case in enumerate(cases):
        scale = case.get('scale', 'other')
        if scale not in VALID_SCALES:
            scale = 'other'
        scale_counts[scale] += 1

        case = enrich_case(case)
        if case.get('yao_analysis', {}).get('hexagram_relation'):
            enriched += 1

    with open(output_file, 'w', encoding='utf-8') as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')

    print(f"\n処理完了: {len(cases)}件中 {enriched}件を強化")
    print("\nスケール別事例数:")
    for scale, count in sorted(scale_counts.items(), key=lambda x: -x[1]):
        print(f"  {scale}: {count}件")

    return enriched


if __name__ == '__main__':
    # プロジェクトルートからの相対パスを使用
    input_file = PROJECT_ROOT / 'data' / 'raw' / 'cases.jsonl'
    output_file = PROJECT_ROOT / 'data' / 'raw' / 'cases.jsonl'

    print(f"入力: {input_file}")
    print(f"出力: {output_file}")
    print(f"hexagram_master.json: {HEXAGRAM_MASTER_PATH}")
    print()

    process_all_cases(str(input_file), str(output_file))
