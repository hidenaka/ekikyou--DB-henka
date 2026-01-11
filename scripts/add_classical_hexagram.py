#!/usr/bin/env python3
"""
既存事例に古典64卦情報を付与するスクリプト
八卦ペア(before_hex, after_hex)から64卦を計算して付与する
"""

import json
from pathlib import Path

# 八卦から三爻への変換（下から上へ）
TRIGRAM_TO_LINES = {
    '乾': (1, 1, 1),  # 天
    '坤': (0, 0, 0),  # 地
    '震': (1, 0, 0),  # 雷
    '巽': (0, 1, 1),  # 風
    '坎': (0, 1, 0),  # 水
    '離': (1, 0, 1),  # 火
    '艮': (0, 0, 1),  # 山
    '兌': (1, 1, 0),  # 沢
}

# 64卦の名前（上卦×下卦の順序で配列）
# インデックス = (上卦番号 * 8) + 下卦番号
# 卦番号: 乾=0, 坤=1, 震=2, 巽=3, 坎=4, 離=5, 艮=6, 兌=7
TRIGRAM_ORDER = ['乾', '坤', '震', '巽', '坎', '離', '艮', '兌']

# 64卦マッピング (上卦, 下卦) -> (卦番号, 卦名)
HEXAGRAM_MAP = {
    # 乾が上卦
    ('乾', '乾'): (1, '乾'),
    ('乾', '坤'): (11, '泰'),
    ('乾', '震'): (34, '大壮'),
    ('乾', '巽'): (9, '小畜'),
    ('乾', '坎'): (5, '需'),
    ('乾', '離'): (14, '大有'),
    ('乾', '艮'): (26, '大畜'),
    ('乾', '兌'): (43, '夬'),
    # 坤が上卦
    ('坤', '乾'): (12, '否'),
    ('坤', '坤'): (2, '坤'),
    ('坤', '震'): (16, '豫'),
    ('坤', '巽'): (20, '観'),
    ('坤', '坎'): (8, '比'),
    ('坤', '離'): (35, '晋'),
    ('坤', '艮'): (23, '剥'),
    ('坤', '兌'): (45, '萃'),
    # 震が上卦
    ('震', '乾'): (25, '无妄'),
    ('震', '坤'): (24, '復'),
    ('震', '震'): (51, '震'),
    ('震', '巽'): (42, '益'),
    ('震', '坎'): (3, '屯'),
    ('震', '離'): (21, '噬嗑'),
    ('震', '艮'): (27, '頤'),
    ('震', '兌'): (17, '随'),
    # 巽が上卦
    ('巽', '乾'): (44, '姤'),
    ('巽', '坤'): (46, '升'),
    ('巽', '震'): (32, '恒'),
    ('巽', '巽'): (57, '巽'),
    ('巽', '坎'): (48, '井'),
    ('巽', '離'): (50, '鼎'),
    ('巽', '艮'): (18, '蠱'),
    ('巽', '兌'): (28, '大過'),
    # 坎が上卦
    ('坎', '乾'): (6, '訟'),
    ('坎', '坤'): (7, '師'),
    ('坎', '震'): (40, '解'),
    ('坎', '巽'): (59, '渙'),
    ('坎', '坎'): (29, '坎'),
    ('坎', '離'): (64, '未済'),
    ('坎', '艮'): (4, '蒙'),
    ('坎', '兌'): (47, '困'),
    # 離が上卦
    ('離', '乾'): (13, '同人'),
    ('離', '坤'): (36, '明夷'),
    ('離', '震'): (55, '豊'),
    ('離', '巽'): (37, '家人'),
    ('離', '坎'): (63, '既済'),
    ('離', '離'): (30, '離'),
    ('離', '艮'): (22, '賁'),
    ('離', '兌'): (49, '革'),
    # 艮が上卦
    ('艮', '乾'): (33, '遯'),
    ('艮', '坤'): (15, '謙'),
    ('艮', '震'): (62, '小過'),
    ('艮', '巽'): (53, '漸'),
    ('艮', '坎'): (39, '蹇'),
    ('艮', '離'): (56, '旅'),
    ('艮', '艮'): (52, '艮'),
    ('艮', '兌'): (31, '咸'),
    # 兌が上卦
    ('兌', '乾'): (10, '履'),
    ('兌', '坤'): (19, '臨'),
    ('兌', '震'): (54, '帰妹'),
    ('兌', '巽'): (61, '中孚'),
    ('兌', '坎'): (60, '節'),
    ('兌', '離'): (38, '睽'),
    ('兌', '艮'): (41, '損'),
    ('兌', '兌'): (58, '兌'),
}


def get_classical_hexagram(upper_trigram: str, lower_trigram: str) -> tuple:
    """
    上卦と下卦から64卦を取得
    before_hexを下卦、after_hexを上卦として扱う（変化の流れ）
    """
    key = (upper_trigram, lower_trigram)
    if key in HEXAGRAM_MAP:
        return HEXAGRAM_MAP[key]
    return None, None


def main():
    import argparse
    parser = argparse.ArgumentParser(description='古典64卦情報を付与')
    parser.add_argument('--apply', action='store_true', help='実際に更新を適用')
    parser.add_argument('--dry-run', action='store_true', help='変更内容を表示のみ')
    args = parser.parse_args()

    data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'cases.jsonl'

    cases = []
    updated_count = 0

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            case = json.loads(line)
            cases.append(case)

    for case in cases:
        before_hex = case.get('before_hex')
        after_hex = case.get('after_hex')

        if before_hex and after_hex:
            # before → after の変化を表す卦
            # before_hexを下卦、after_hexを上卦として64卦を構成
            hex_num, hex_name = get_classical_hexagram(after_hex, before_hex)

            if hex_name:
                old_before = case.get('classical_before_hexagram')
                old_after = case.get('classical_after_hexagram')

                # 変化前の状態を表す卦（before_hex単独の重卦）
                before_num, before_name = get_classical_hexagram(before_hex, before_hex)
                # 変化後の状態を表す卦（after_hex単独の重卦）
                after_num, after_name = get_classical_hexagram(after_hex, after_hex)

                if not old_before and before_name:
                    case['classical_before_hexagram'] = f'{before_num}_{before_name}'
                    updated_count += 1

                if not old_after and after_name:
                    case['classical_after_hexagram'] = f'{after_num}_{after_name}'

    print(f'処理した事例数: {len(cases)}')
    print(f'更新した事例数: {updated_count}')

    if args.apply:
        with open(data_path, 'w', encoding='utf-8') as f:
            for case in cases:
                f.write(json.dumps(case, ensure_ascii=False) + '\n')
        print(f'✅ {data_path} に更新されたデータを書き込みました')
    elif args.dry_run:
        print('ドライラン: 変更は適用されていません')
    else:
        print('--apply または --dry-run を指定してください')


if __name__ == '__main__':
    main()
