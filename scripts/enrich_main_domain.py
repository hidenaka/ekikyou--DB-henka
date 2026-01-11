#!/usr/bin/env python3
"""
Phase A: main_domain自動補完スクリプト
目標: 4,131件のmain_domain未設定ケースを自動分類し、15種類に正規化

使用方法:
  python3 scripts/enrich_main_domain.py [--dry-run] [--report-only]

オプション:
  --dry-run     : DBを更新せずに結果をプレビュー
  --report-only : 現状レポートのみ表示
"""

import json
import re
import sys
from datetime import datetime
from collections import Counter
from pathlib import Path

# ドメイン正規化マッピング（70種類→15種類）
DOMAIN_NORMALIZE = {
    # テクノロジー
    'IT・通信': 'テクノロジー',
    'IT・情報通信': 'テクノロジー',
    '通信・IT': 'テクノロジー',
    'IT': 'テクノロジー',
    # 製造
    '製造業': '製造',
    # 金融
    '金融・銀行': '金融',
    '金融・保険': '金融',
    '経済・金融': '金融',
    # 小売・サービス
    '小売業': '小売・サービス',
    '小売・流通': '小売・サービス',
    'サービス業': '小売・サービス',
    '飲食': '小売・サービス',
    '飲食業': '小売・サービス',
    # 医療・製薬
    '医療': '医療・製薬',
    '医療・福祉': '医療・製薬',
    # エンタメ
    'エンターテインメント': 'エンタメ',
    'エンタメ・ゲーム': 'エンタメ',
    '文化・芸術': 'エンタメ',
    'メディア': 'エンタメ',
    'メディア・広告': 'エンタメ',
    # 物流・交通
    '交通': '物流・交通',
    '交通・運輸': '物流・交通',
    '物流': '物流・交通',
    '物流・運輸': '物流・交通',
    '物流・輸送': '物流・交通',
    '運輸・物流': '物流・交通',
    'モビリティ': '物流・交通',
    'インフラ・交通': '物流・交通',
    # 不動産・建設
    '建設': '不動産・建設',
    '不動産': '不動産・建設',
    '建設・不動産': '不動産・建設',
    # エネルギー・環境
    'エネルギー': 'エネルギー・環境',
    '環境・エネルギー': 'エネルギー・環境',
    # 農林水産
    '農業': '農林水産',
    '農業・食品': '農林水産',
    '食品・農業': '農林水産',
    # 観光・旅行
    '観光・レジャー': '観光・旅行',
    '観光・サービス': '観光・旅行',
    'レジャー': '観光・旅行',
    # 社会・コミュニティ
    '地域・自治体': '社会・コミュニティ',
    '社会・制度': '社会・コミュニティ',
    '地域・コミュニティ': '社会・コミュニティ',
    '地域経済': '社会・コミュニティ',
    'NPO・ボランティア': '社会・コミュニティ',
    'NPO・市民活動': '社会・コミュニティ',
    '公共サービス': '社会・コミュニティ',
    '防災・災害': '社会・コミュニティ',
    '防災・危機管理': '社会・コミュニティ',
    '宗教': '社会・コミュニティ',
    # 国家・政治
    '政策・行政': '国家・政治',
    # 生活・暮らし
    '趣味・レクリエーション': '生活・暮らし',
    '家族・相続': '生活・暮らし',
    '人事・HR': '生活・暮らし',
    '人材・組織': '生活・暮らし',
    # その他正規化
    '経営・戦略': 'ビジネス',
    '商社・貿易': 'ビジネス',
    'ビジネス': 'ビジネス',
    '法律・専門サービス': 'ビジネス',
    '宇宙・航空': 'テクノロジー',
    '自動車': '製造',
}

# 15種類の正規ドメイン
VALID_DOMAINS = [
    'テクノロジー', '製造', '金融', '小売・サービス', '医療・製薬',
    'エンタメ', '物流・交通', '不動産・建設', 'エネルギー・環境',
    '農林水産', '教育', '観光・旅行', '社会・コミュニティ',
    '国家・政治', 'スポーツ', '生活・暮らし', 'ビジネス'
]

# キーワード→ドメイン推定ルール
KEYWORD_RULES = [
    # テクノロジー
    (r'(AI|IT|DX|デジタル|半導体|ソフトウェア|アプリ|プログラム|システム|クラウド|量子|ロボット|IoT)', 'テクノロジー'),
    (r'(スタートアップ|ベンチャー|テック)', 'テクノロジー'),
    # 製造
    (r'(製造|工場|自動車|電機|機械|素材|鉄鋼|化学|繊維)', '製造'),
    (r'(メーカー|ものづくり)', '製造'),
    # 金融
    (r'(銀行|証券|保険|金融|投資|ファンド|融資|為替|株|債券)', '金融'),
    (r'(決済|キャッシュレス|FinTech)', '金融'),
    # 小売・サービス
    (r'(小売|スーパー|コンビニ|百貨店|デパート|ドラッグストア|ホームセンター)', '小売・サービス'),
    (r'(飲食|レストラン|カフェ|外食|ラーメン|寿司)', '小売・サービス'),
    (r'(EC|通販|ネットショップ)', '小売・サービス'),
    (r'(サービス|ホテル|旅館|清掃|警備)', '小売・サービス'),
    # 医療・製薬
    (r'(病院|医療|医師|看護|介護|福祉|製薬|薬|iPS|再生医療|健康)', '医療・製薬'),
    # エンタメ
    (r'(映画|音楽|アニメ|漫画|ゲーム|エンタメ|芸能|アイドル|アーティスト)', 'エンタメ'),
    (r'(テレビ|放送|メディア|出版|新聞|雑誌)', 'エンタメ'),
    (r'(美術|芸術|文化|博物館|美術館)', 'エンタメ'),
    # 物流・交通
    (r'(物流|運送|配送|宅配|トラック|鉄道|航空|空港|港|船)', '物流・交通'),
    (r'(交通|タクシー|バス|電車|新幹線|リニア)', '物流・交通'),
    # 不動産・建設
    (r'(不動産|建設|住宅|マンション|ビル|土地|開発|再開発)', '不動産・建設'),
    # エネルギー・環境
    (r'(エネルギー|電力|ガス|石油|原子力|再エネ|太陽光|風力)', 'エネルギー・環境'),
    (r'(環境|脱炭素|CO2|温暖化|リサイクル|廃棄物)', 'エネルギー・環境'),
    # 農林水産
    (r'(農業|農家|農協|漁業|林業|食品|食料|米|野菜|果物|酪農|畜産)', '農林水産'),
    (r'(日本酒|ワイン|ウイスキー|茶|和牛)', '農林水産'),
    # 教育
    (r'(教育|学校|大学|高校|中学|小学|塾|予備校|学習|研究|ノーベル)', '教育'),
    # 観光・旅行
    (r'(観光|旅行|ツアー|インバウンド|リゾート|温泉)', '観光・旅行'),
    # 社会・コミュニティ
    (r'(地域|自治体|市町村|NPO|ボランティア|コミュニティ|住民)', '社会・コミュニティ'),
    (r'(災害|防災|復興|震災|津波|台風)', '社会・コミュニティ'),
    # 国家・政治
    (r'(政治|政府|国会|選挙|政策|外交|安全保障|軍|防衛)', '国家・政治'),
    (r'(大統領|首相|総理|内閣|与党|野党)', '国家・政治'),
    # スポーツ
    (r'(スポーツ|野球|サッカー|バスケ|テニス|ゴルフ|オリンピック|選手|監督)', 'スポーツ'),
    (r'(MLB|NBA|プレミア|ワールドカップ|甲子園)', 'スポーツ'),
    # 生活・暮らし
    (r'(生活|暮らし|家族|結婚|離婚|子育て|育児|介護|老後|年金)', '生活・暮らし'),
    (r'(趣味|ライフスタイル|副業|転職|キャリア)', '生活・暮らし'),
]

# scale→ドメイン推定（フォールバック）
SCALE_DOMAIN_FALLBACK = {
    'country': '国家・政治',
    'family': '生活・暮らし',
}


def infer_domain(case):
    """ケースからmain_domainを推定"""
    target = case.get('target_name', '')
    summary = case.get('story_summary', '')
    tags = ' '.join(case.get('free_tags', []))
    scale = case.get('scale', '')

    text = f"{target} {summary} {tags}"

    # キーワードルールで判定
    for pattern, domain in KEYWORD_RULES:
        if re.search(pattern, text):
            return domain

    # scaleによるフォールバック
    if scale in SCALE_DOMAIN_FALLBACK:
        return SCALE_DOMAIN_FALLBACK[scale]

    return None


def normalize_domain(domain):
    """ドメイン名を正規化"""
    if not domain or domain == 'unknown':
        return None
    if domain in VALID_DOMAINS:
        return domain
    return DOMAIN_NORMALIZE.get(domain, domain)


def main():
    dry_run = '--dry-run' in sys.argv
    report_only = '--report-only' in sys.argv

    db_path = Path('data/raw/cases.jsonl')
    checkpoint_path = Path('data/diagnostic/phase_a_progress.json')

    # データ読み込み
    with open(db_path, 'r') as f:
        cases = [json.loads(line.strip()) for line in f]

    print(f"総ケース数: {len(cases)}")

    # 現状分析
    no_domain = [c for c in cases if not c.get('main_domain') or c.get('main_domain') == 'unknown']
    has_domain = [c for c in cases if c.get('main_domain') and c.get('main_domain') != 'unknown']

    print(f"main_domain未設定: {len(no_domain)}件 ({len(no_domain)/len(cases)*100:.1f}%)")
    print(f"main_domain設定済: {len(has_domain)}件 ({len(has_domain)/len(cases)*100:.1f}%)")

    if report_only:
        # 現状レポートのみ
        domain_counts = Counter(c.get('main_domain', 'unknown') for c in cases)
        print("\n現在のドメイン分布:")
        for domain, count in domain_counts.most_common(20):
            print(f"  {domain}: {count}件")
        return

    # 推定実行
    updated_count = 0
    normalized_count = 0
    inferred_domains = Counter()

    for case in cases:
        current_domain = case.get('main_domain', '')

        # 既存ドメインの正規化
        if current_domain and current_domain != 'unknown':
            normalized = normalize_domain(current_domain)
            if normalized and normalized != current_domain:
                case['main_domain'] = normalized
                normalized_count += 1

        # 未設定の場合は推定
        if not case.get('main_domain') or case.get('main_domain') == 'unknown':
            inferred = infer_domain(case)
            if inferred:
                case['main_domain'] = inferred
                inferred_domains[inferred] += 1
                updated_count += 1

    print(f"\n推定結果:")
    print(f"  新規推定: {updated_count}件")
    print(f"  正規化: {normalized_count}件")

    print(f"\n推定されたドメイン分布:")
    for domain, count in inferred_domains.most_common():
        print(f"  {domain}: {count}件")

    # 残り未設定
    still_no_domain = sum(1 for c in cases if not c.get('main_domain') or c.get('main_domain') == 'unknown')
    print(f"\n残り未設定: {still_no_domain}件 ({still_no_domain/len(cases)*100:.1f}%)")

    if dry_run:
        print("\n[DRY RUN] DBは更新されていません")
        return

    # DB更新
    print("\nDBを更新中...")
    with open(db_path, 'w') as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')

    # チェックポイント保存
    checkpoint = {
        'timestamp': datetime.now().isoformat(),
        'total_cases': len(cases),
        'inferred_count': updated_count,
        'normalized_count': normalized_count,
        'remaining_unknown': still_no_domain,
        'status': 'COMPLETED' if still_no_domain / len(cases) < 0.1 else 'IN_PROGRESS'
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    print(f"完了。チェックポイント保存: {checkpoint_path}")


if __name__ == '__main__':
    main()
