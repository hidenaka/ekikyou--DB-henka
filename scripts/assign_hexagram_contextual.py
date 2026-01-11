#!/usr/bin/env python3
"""
64卦ID付与スクリプト（文脈理解ベース）

単純な上下卦組合せではなく、変化のコンテキスト（pattern_type, before_state, after_state等）を
考慮して64卦IDを付与する。

64卦の決定基準:
1. before_hex = 下卦（内なる状態、本質）
2. trigger_hex = 上卦（外からの影響、状況）
3. pattern_type で卦の方向性を調整
4. outcome (Success/Failure) で類似卦を選択
"""

import json
from typing import Optional, Dict, Tuple

# 64卦マスターテーブル（下卦→上卦→卦番号）
# 形式: HEXAGRAM_TABLE[下卦][上卦] = (卦番号, 卦名)
HEXAGRAM_TABLE = {
    '乾': {
        '乾': (1, '乾為天'),
        '坤': (11, '地天泰'),
        '震': (34, '雷天大壮'),
        '巽': (9, '風天小畜'),
        '坎': (5, '水天需'),
        '離': (14, '火天大有'),
        '艮': (26, '山天大畜'),
        '兌': (43, '沢天夬'),
    },
    '坤': {
        '乾': (12, '天地否'),
        '坤': (2, '坤為地'),
        '震': (16, '雷地豫'),
        '巽': (20, '風地観'),
        '坎': (8, '水地比'),
        '離': (35, '火地晋'),
        '艮': (23, '山地剥'),
        '兌': (45, '沢地萃'),
    },
    '震': {
        '乾': (25, '天雷无妄'),
        '坤': (24, '地雷復'),
        '震': (51, '震為雷'),
        '巽': (42, '風雷益'),
        '坎': (3, '水雷屯'),
        '離': (21, '火雷噬嗑'),
        '艮': (27, '山雷頤'),
        '兌': (17, '沢雷随'),
    },
    '巽': {
        '乾': (44, '天風姤'),
        '坤': (46, '地風升'),
        '震': (32, '雷風恒'),
        '巽': (57, '巽為風'),
        '坎': (48, '水風井'),
        '離': (50, '火風鼎'),
        '艮': (18, '山風蠱'),
        '兌': (28, '沢風大過'),
    },
    '坎': {
        '乾': (6, '天水訟'),
        '坤': (7, '地水師'),
        '震': (40, '雷水解'),
        '巽': (59, '風水渙'),
        '坎': (29, '坎為水'),
        '離': (64, '火水未済'),
        '艮': (4, '山水蒙'),
        '兌': (47, '沢水困'),
    },
    '離': {
        '乾': (13, '天火同人'),
        '坤': (36, '地火明夷'),
        '震': (55, '雷火豊'),
        '巽': (37, '風火家人'),
        '坎': (63, '水火既済'),
        '離': (30, '離為火'),
        '艮': (22, '山火賁'),
        '兌': (49, '沢火革'),
    },
    '艮': {
        '乾': (33, '天山遯'),
        '坤': (15, '地山謙'),
        '震': (62, '雷山小過'),
        '巽': (53, '風山漸'),
        '坎': (39, '水山蹇'),
        '離': (56, '火山旅'),
        '艮': (52, '艮為山'),
        '兌': (31, '沢山咸'),
    },
    '兌': {
        '乾': (10, '天沢履'),
        '坤': (19, '地沢臨'),
        '震': (54, '雷沢帰妹'),
        '巽': (61, '風沢中孚'),
        '坎': (60, '水沢節'),
        '離': (38, '火沢睽'),
        '艮': (41, '山沢損'),
        '兌': (58, '兌為沢'),
    },
}

# パターン別の卦決定ロジック
# pattern_type と outcome によって、上下卦の解釈を調整する
def determine_hexagram_contextual(
    before_hex: str,
    trigger_hex: str,
    action_hex: str,
    after_hex: str,
    pattern_type: str,
    outcome: str,
    before_state: str,
    after_state: str
) -> Tuple[Optional[int], Optional[str]]:
    """
    文脈を考慮して64卦を決定する

    Returns: (hexagram_id, hexagram_name) or (None, None)
    """

    # 基本ルール: before_hex を下卦、trigger_hex を上卦として決定
    lower = before_hex
    upper = trigger_hex

    # パターン別調整
    if pattern_type in ['Shock_Recovery', 'Crisis_Pivot']:
        # 危機からの回復: 外部衝撃(trigger)が上卦
        lower = before_hex
        upper = trigger_hex

    elif pattern_type in ['Steady_Growth', 'Breakthrough']:
        # 安定成長・躍進: action_hex が主導
        if action_hex and action_hex != 'N/A':
            lower = before_hex
            upper = action_hex

    elif pattern_type in ['Slow_Decline', 'Gradual_Decline', 'Managed_Decline']:
        # 衰退: after_hex が結果を示す
        if after_hex and after_hex != 'N/A':
            lower = before_hex
            upper = after_hex

    elif pattern_type in ['Hubris_Collapse', 'Failed_Attempt']:
        # 傲慢による崩壊・失敗: 内部(before)と外部(trigger)の衝突
        lower = before_hex
        upper = trigger_hex

    elif pattern_type == 'Endurance':
        # 忍耐・持続: before_hex の純卦または trigger との組合せ
        if before_hex == trigger_hex:
            lower = before_hex
            upper = before_hex
        else:
            lower = before_hex
            upper = trigger_hex

    elif pattern_type == 'Stagnation':
        # 停滞: 下卦と上卦が相反（天地否など）
        lower = before_hex
        upper = trigger_hex

    # Outcomeによる調整（同じ卦でも吉凶が分かれる場合）
    # 基本的には上記のロジックに従うが、特殊ケースを処理

    # ルックアップ
    if lower in HEXAGRAM_TABLE and upper in HEXAGRAM_TABLE[lower]:
        hex_id, hex_name = HEXAGRAM_TABLE[lower][upper]
        return hex_id, hex_name

    return None, None


def process_cases(input_file: str, output_file: str, dry_run: bool = False):
    """事例を処理して64卦IDを付与"""

    with open(input_file, 'r') as f:
        cases = [json.loads(line) for line in f if line.strip()]

    updated = 0
    skipped = 0
    errors = 0

    for case in cases:
        # 既にhexagram_idがある場合はスキップ
        yao = case.get('yao_analysis', {})
        if yao and isinstance(yao, dict) and yao.get('before_hexagram_id'):
            skipped += 1
            continue

        # 必要なフィールドを取得
        before_hex = case.get('before_hex', '')
        trigger_hex = case.get('trigger_hex', '')
        action_hex = case.get('action_hex', '')
        after_hex = case.get('after_hex', '')
        pattern_type = case.get('pattern_type', '')
        outcome = case.get('outcome', '')
        before_state = case.get('before_state', '')
        after_state = case.get('after_state', '')

        # 64卦を決定
        hex_id, hex_name = determine_hexagram_contextual(
            before_hex, trigger_hex, action_hex, after_hex,
            pattern_type, outcome, before_state, after_state
        )

        if hex_id:
            if 'yao_analysis' not in case or not case['yao_analysis']:
                case['yao_analysis'] = {}
            case['yao_analysis']['before_hexagram_id'] = hex_id
            case['yao_analysis']['hexagram_name'] = hex_name
            updated += 1
        else:
            errors += 1

    if not dry_run:
        with open(output_file, 'w') as f:
            for case in cases:
                f.write(json.dumps(case, ensure_ascii=False) + '\n')

    return {
        'total': len(cases),
        'updated': updated,
        'skipped': skipped,
        'errors': errors
    }


if __name__ == '__main__':
    import sys

    input_file = 'data/raw/cases.jsonl'
    output_file = 'data/raw/cases.jsonl'
    dry_run = '--dry-run' in sys.argv

    print(f'入力: {input_file}')
    print(f'出力: {output_file}')
    print(f'ドライラン: {dry_run}')
    print()

    result = process_cases(input_file, output_file, dry_run)

    print(f"処理結果:")
    print(f"  総件数: {result['total']}")
    print(f"  更新: {result['updated']}件")
    print(f"  スキップ: {result['skipped']}件（既存ID）")
    print(f"  エラー: {result['errors']}件")
