#!/usr/bin/env python3
"""
バッチファイル一括インポートスクリプト
- {"cases": [...]} 形式と [...] 形式の両方に対応
- 重複チェック付き
- フィールド正規化対応
"""

import sys
import json
import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from schema_v3 import Case
from id_utils import cases_path, load_existing_ids, generate_next_id

# 正規化マッピング
BEFORE_STATE_MAP = {
    "成長期": "絶頂・慢心",
    "衰退期": "停滞・閉塞",
    "成熟・安定": "安定・平和",
    "安定期": "安定・平和",
    "危機": "どん底・危機",
    "拡大期": "絶頂・慢心",
    "混乱期": "混乱・カオス",
    "再建期": "成長痛",
    "成長": "成長痛",
    "低迷": "停滞・閉塞",
    "安定": "安定・平和",
    "黎明期": "成長痛",
    "創業期": "成長痛",
    "発展期": "成長痛",
    "全盛期": "絶頂・慢心",
    "危機・混乱": "混乱・カオス",
    "低迷期": "停滞・閉塞",
    "変革期": "成長痛",
    "急成長": "絶頂・慢心",
    "破綻寸前": "どん底・危機",
}

AFTER_STATE_MAP = {
    "成長期": "V字回復・大成功",
    "V字回復": "V字回復・大成功",
    "衰退期": "停滞・閉塞",
    "成熟・安定": "安定・平和",
    "安定期": "安定・平和",
    "危機": "どん底・危機",
    "拡大期": "V字回復・大成功",
    "混乱期": "混乱・カオス",
    "持続成長": "持続成長・大成功",
    "安定成長": "安定成長・成功",
    "再建成功": "V字回復・大成功",
    "縮小均衡": "縮小安定・生存",
    "低迷継続": "停滞・閉塞",
    "継続成長": "持続成長・大成功",
    "安定": "安定・平和",
    "成長": "V字回復・大成功",
    "危機・混乱": "混乱・カオス",
    "縮小安定": "縮小安定・生存",
    "消滅": "崩壊・消滅",
    "崩壊": "崩壊・消滅",
    "衰退": "停滞・閉塞",
    "低迷": "停滞・閉塞",
    "安定・復活": "V字回復・大成功",
    "再編完了": "変質・新生",
    "変革完了": "変質・新生",
    "回復途上": "現状維持・延命",
    "回復": "V字回復・大成功",
    "復活": "V字回復・大成功",
    "成長継続": "持続成長・大成功",
    "再建中": "現状維持・延命",
    "転換成功": "変質・新生",
    "継続": "現状維持・延命",
    "破綻・消滅": "崩壊・消滅",
    "急成長": "V字回復・大成功",
}

TRIGGER_TYPE_MAP = {
    "内部革新": "意図的決断",
    "内部決断": "意図的決断",
    "技術革新": "外部ショック",
    "市場変化": "外部ショック",
    "経済危機": "外部ショック",
    "市場拡大": "外部ショック",
    "競争激化": "外部ショック",
    "規制変更": "外部ショック",
    "経営危機": "内部崩壊",
    "内部改革": "意図的決断",
    "戦略転換": "意図的決断",
    "リーダー交代": "意図的決断",
    "合併・買収": "意図的決断",
    "人材確保": "偶発・出会い",
    "パンデミック": "外部ショック",
    "コロナ禍": "外部ショック",
    "不正発覚": "内部崩壊",
    "スキャンダル": "内部崩壊",
    "規制強化": "外部ショック",
    "政策転換": "外部ショック",
    "金融危機": "外部ショック",
    "需要変化": "外部ショック",
    "戦争": "外部ショック",
    "自然災害": "外部ショック",
    "創業者退任": "意図的決断",
    "株主圧力": "外部ショック",
    "買収提案": "外部ショック",
    "技術変化": "外部ショック",
    "デジタル化": "外部ショック",
    "環境規制": "外部ショック",
    "貿易摩擦": "外部ショック",
    "経営陣交代": "意図的決断",
    "後継者問題": "内部崩壊",
    "新規参入": "外部ショック",
    "市場崩壊": "外部ショック",
    "資金枯渇": "内部崩壊",
    "地政学リスク": "外部ショック",
    "組織改革": "意図的決断",
    "事故": "外部ショック",
    "品質問題": "内部崩壊",
    "投資家圧力": "外部ショック",
    "金利上昇": "外部ショック",
    "製品ヒット": "偶発・出会い",
    "M&A": "意図的決断",
    "パートナー離脱": "内部崩壊",
    "デザイン刷新": "意図的決断",
    "内部抗争": "内部崩壊",
    "制度的制約": "外部ショック",
    "市場変動": "外部ショック",
    "規制・スキャンダル": "外部ショック",
    "買収戦争": "外部ショック",
    "資本調達": "意図的決断",
    "通貨危機": "外部ショック",
}

ACTION_TYPE_MAP = {
    "攻め・拡張": "攻める・挑戦",
    "攻め": "攻める・挑戦",
    "拡大戦略": "攻める・挑戦",
    "守り・縮小": "守る・維持",
    "守り": "守る・維持",
    "防衛・維持": "守る・維持",
    "撤退・縮小": "捨てる・撤退",
    "縮小・撤退": "捨てる・撤退",
    "革新・変革": "刷新・破壊",
    "構造改革": "刷新・破壊",
    "M&A・統合": "対話・融合",
    "提携・協力": "対話・融合",
    "事業転換": "刷新・破壊",
    "耐える": "耐える・潜伏",
    "忍耐": "耐える・潜伏",
    "様子見・維持": "守る・維持",
    "様子見": "守る・維持",
    "積極投資": "攻める・挑戦",
    "攻め・拡大": "攻める・挑戦",
    "拡大": "攻める・挑戦",
    "統合・再編": "対話・融合",
    "多角化": "分散・スピンオフ",
    "集中": "捨てる・撤退",
    "選択と集中": "捨てる・撤退",
    "リストラ": "捨てる・撤退",
    "売却": "捨てる・撤退",
    "撤退": "捨てる・撤退",
    "変革": "刷新・破壊",
    "改革": "刷新・破壊",
    "転換": "刷新・破壊",
    "協業": "対話・融合",
    "提携": "対話・融合",
    "買収": "対話・融合",
    "合併": "対話・融合",
    "維持": "守る・維持",
    "防衛": "守る・維持",
    "挑戦": "攻める・挑戦",
    "新規事業": "攻める・挑戦",
    "分社化": "分散・スピンオフ",
    "スピンオフ": "分散・スピンオフ",
    "潜伏": "耐える・潜伏",
    "放置": "逃げる・放置",
    "無策": "逃げる・放置",
    "攻め・張": "攻める・挑戦",
    "攻め・張・拡大": "攻める・挑戦",
    "拡張・成長": "攻める・挑戦",
    "成長": "攻める・挑戦",
    "拡張": "攻める・挑戦",
}


def infer_before_state(text: str) -> str:
    """説明テキストから before_state を推論"""
    text = text.lower() if text else ''
    if any(k in text for k in ['危機', '破綻', 'どん底', '赤字', '倒産', '崩壊']):
        return 'どん底・危機'
    if any(k in text for k in ['混乱', 'カオス', '問題', 'トラブル']):
        return '混乱・カオス'
    if any(k in text for k in ['停滞', '閉塞', '低迷', '衰退', '苦戦']):
        return '停滞・閉塞'
    if any(k in text for k in ['成長', '発展', '拡大', '上昇']):
        return '成長痛'
    if any(k in text for k in ['絶頂', '全盛', 'トップ', '最大', '1位']):
        return '絶頂・慢心'
    # デフォルト：成長前の安定状態として扱う
    return '安定・平和'


def infer_after_state(text: str) -> str:
    """説明テキストから after_state を推論"""
    text = text.lower() if text else ''
    if any(k in text for k in ['世界', 'グローバル', 'トップ', '最大', '1位', '殿堂', '達成', '完成']):
        return 'V字回復・大成功'
    if any(k in text for k in ['成長', '発展', '拡大', '成功']):
        return '持続成長・大成功'
    if any(k in text for k in ['安定', '継続', '維持']):
        return '安定・平和'
    if any(k in text for k in ['縮小', '撤退', '集中']):
        return '縮小安定・生存'
    if any(k in text for k in ['転換', '変革', '再編']):
        return '変質・新生'
    if any(k in text for k in ['破綻', '消滅', '倒産']):
        return '崩壊・消滅'
    return 'V字回復・大成功'


PATTERN_TYPE_MAP = {
    'Gradual_Ascent': 'Steady_Growth',
    'Innovation_Leap': 'Breakthrough',
    'gradual_ascent': 'Steady_Growth',
    'innovation_leap': 'Breakthrough',
    'Community_Building': 'Steady_Growth',
    'Global_Expansion': 'Breakthrough',
    'Goal_Achievement': 'Pivot_Success',
    'Heritage_Innovation': 'Steady_Growth',
    'Heritage_Preservation': 'Endurance',
    'Revolution': 'Crisis_Pivot',
    'Serial_Transformation': 'Pivot_Success',
    'Small_Failure_Learning': 'Failed_Attempt',
    'Strategic_Dispersion': 'Managed_Decline',
    'Sustainable_Growth': 'Steady_Growth',
    'Talent_Magnet': 'Steady_Growth',
    'Technology_Accumulation': 'Endurance',
    'Trust_Building': 'Steady_Growth',
}

MAIN_DOMAIN_MAP = {
    'Retail': '小売・サービス',
    'Technology': 'テクノロジー',
    'Fashion': '小売・サービス',
    'Food': '食品・農業',
    'Finance': '金融',
    'Manufacturing': '製造',
    'Healthcare': 'ヘルスケア',
    'Entertainment': 'エンタメ',
    'Construction': '建設・不動産',
    'Transportation': '物流・運輸',
    'Energy': 'エネルギー',
}


def normalize_case(item: dict) -> dict:
    """ケースデータを正規化"""
    result = item.copy()

    # before_state正規化（マップになければ推論）
    bs = result.get('before_state', '')
    if bs in BEFORE_STATE_MAP:
        result['before_state'] = BEFORE_STATE_MAP[bs]
    elif bs and bs not in ['絶頂・慢心', '停滞・閉塞', '混乱・カオス', '成長痛', 'どん底・危機', '安定・平和', 'V字回復・大成功', '縮小安定・生存']:
        result['before_state'] = infer_before_state(bs)

    # after_state正規化（マップになければ推論）
    afs = result.get('after_state', '')
    if afs in AFTER_STATE_MAP:
        result['after_state'] = AFTER_STATE_MAP[afs]
    elif afs and afs not in ['V字回復・大成功', '縮小安定・生存', '変質・新生', '現状維持・延命', '迷走・混乱', '混乱・カオス', '崩壊・消滅', '停滞・閉塞', 'どん底・危機', '持続成長・大成功', '安定成長・成功', '安定・平和']:
        result['after_state'] = infer_after_state(afs)

    # trigger_type正規化
    tt = result.get('trigger_type', '')
    if tt in TRIGGER_TYPE_MAP:
        result['trigger_type'] = TRIGGER_TYPE_MAP[tt]

    # action_type正規化
    at = result.get('action_type', '')
    if at in ACTION_TYPE_MAP:
        result['action_type'] = ACTION_TYPE_MAP[at]

    # pattern_type正規化
    pt = result.get('pattern_type', '')
    if pt in PATTERN_TYPE_MAP:
        result['pattern_type'] = PATTERN_TYPE_MAP[pt]

    # main_domain正規化
    md = result.get('main_domain', '')
    if md in MAIN_DOMAIN_MAP:
        result['main_domain'] = MAIN_DOMAIN_MAP[md]

    # 必須フィールドのデフォルト値
    if 'source_type' not in result:
        result['source_type'] = 'news'

    if 'credibility_rank' not in result:
        result['credibility_rank'] = 'B'

    return result


def load_batch_cases(filepath: str) -> list:
    """バッチファイルからケースを読み込む"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'cases' in data:
        return data['cases']
    else:
        return []


def get_existing_targets() -> set:
    """既存のtarget_nameとperiodの組み合わせを取得"""
    targets = set()
    path = cases_path()
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    c = json.loads(line)
                    key = (c.get('target_name', ''), c.get('period', ''))
                    targets.add(key)
                except:
                    pass
    return targets


def main():
    base_dir = Path(__file__).parent.parent
    import_dir = base_dir / 'data' / 'import'

    # パターン指定
    patterns = sys.argv[1:] if len(sys.argv) > 1 else ['batch_international_*.json', 'batch_hex_*.json']

    all_files = []
    for pattern in patterns:
        all_files.extend(sorted(glob.glob(str(import_dir / pattern))))

    if not all_files:
        print("インポートするファイルがありません")
        return 1

    print(f"対象ファイル: {len(all_files)}件")

    # 既存データのチェック
    existing_targets = get_existing_targets()
    used_ids = load_existing_ids()

    path = cases_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    total_added = 0
    total_skipped = 0
    errors = []

    with open(path, 'a', encoding='utf-8') as out:
        for filepath in all_files:
            filename = Path(filepath).name
            try:
                cases = load_batch_cases(filepath)
                added = 0
                skipped = 0

                for idx, item in enumerate(cases, 1):
                    try:
                        # 重複チェック
                        key = (item.get('target_name', ''), item.get('period', ''))
                        if key in existing_targets:
                            skipped += 1
                            continue

                        # 正規化してからスキーマ検証
                        normalized = normalize_case(item)

                        # 追加フィールドを保存
                        extra_fields = {}
                        for key in ['country', 'main_domain', 'sources', 'yao_analysis', 'hexagram_id', 'hexagram_name', 'yao_context']:
                            if key in normalized:
                                extra_fields[key] = normalized[key]

                        # hexagram_idがある場合、yao_analysisに反映
                        if 'hexagram_id' in extra_fields and extra_fields['hexagram_id']:
                            if 'yao_analysis' not in extra_fields:
                                extra_fields['yao_analysis'] = {}
                            if isinstance(extra_fields['yao_analysis'], dict):
                                extra_fields['yao_analysis']['before_hexagram_id'] = extra_fields['hexagram_id']

                        case = Case(**normalized)

                        # ID生成
                        tid = case.transition_id
                        if not tid or (isinstance(tid, str) and not tid.strip()):
                            new_tid = generate_next_id(case.scale, used_ids)
                            case.transition_id = new_tid
                            used_ids.add(new_tid)
                        elif tid in used_ids:
                            new_tid = generate_next_id(case.scale, used_ids)
                            case.transition_id = new_tid
                            used_ids.add(new_tid)
                        else:
                            used_ids.add(tid)

                        # 書き込み（追加フィールドをマージ）
                        case_dict = case.model_dump()
                        case_dict.update(extra_fields)
                        out.write(json.dumps(case_dict, ensure_ascii=False) + '\n')
                        existing_targets.add(key)
                        added += 1

                    except Exception as e:
                        errors.append(f"{filename}[{idx}]: {str(e)[:100]}")

                if added > 0:
                    print(f"  {filename}: +{added}件 (skip: {skipped})")
                total_added += added
                total_skipped += skipped

            except Exception as e:
                errors.append(f"{filename}: {str(e)[:100]}")

    print(f"\n完了: 追加 {total_added}件, スキップ {total_skipped}件")

    if errors:
        print(f"\nエラー ({len(errors)}件):")
        for e in errors[:20]:
            print(f"  {e}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
