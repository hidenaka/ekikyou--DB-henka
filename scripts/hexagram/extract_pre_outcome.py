#!/usr/bin/env python3
"""
Phase 1: 結果前テキストの切り出し
story_summaryから結果記述より前の部分を抽出してpre_outcome_textを生成
"""

import json
import re
from pathlib import Path

# 結果を示す接続詞・表現
RESULT_MARKERS = [
    'の結果', 'その結果', 'により', 'を経て', 'を果たした',
    '成功', '失敗', '回復', '崩壊', '破綻', '達成', 'V字',
    '黒字化', '上場廃止', '買収成立', '和解', '勝訴', '敗訴',
    '復活', '再生', '倒産', '上場', '完了', '実現'
]

def extract_pre_outcome(story_summary: str) -> str:
    """結果記述より前の部分を抽出"""
    if not story_summary:
        return ""

    earliest_pos = len(story_summary)
    for marker in RESULT_MARKERS:
        pos = story_summary.find(marker)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos

    # 結果マーカーより前の部分を返す（最低100文字は確保）
    if earliest_pos < 100:
        return story_summary[:200] if len(story_summary) > 200 else story_summary
    return story_summary[:earliest_pos]

def process_all_cases(input_path: str, output_path: str):
    """全事例にpre_outcome_textを追加"""
    cases = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    processed = 0
    for case in cases:
        story = case.get('story_summary', '')
        case['pre_outcome_text'] = extract_pre_outcome(story)
        processed += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')

    print(f"処理完了: {processed}件")
    return processed

if __name__ == '__main__':
    input_file = 'data/raw/cases.jsonl'
    output_file = 'data/raw/cases_with_pre_outcome.jsonl'
    process_all_cases(input_file, output_file)
