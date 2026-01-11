#!/usr/bin/env python3
"""
generate_batch_cases.py - 事例データのバッチ生成スクリプト

使い方:
    python scripts/generate_batch_cases.py --phase 1 --output data/import/phase1.json
    python scripts/generate_batch_cases.py --phase all --output data/import/all_phases.json
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict
from itertools import product


# 八卦の爻構成（変爻計算用）
TRIGRAM_LINES = {
    "乾": [1, 1, 1], "兌": [1, 1, 0], "離": [1, 0, 1], "震": [1, 0, 0],
    "巽": [0, 1, 1], "坎": [0, 1, 0], "艮": [0, 0, 1], "坤": [0, 0, 0],
}

def infer_changing_lines(from_hex: str, to_hex: str) -> List[int]:
    """2つの八卦間の変爻を推定"""
    from_lines = TRIGRAM_LINES.get(from_hex, [0, 0, 0])
    to_lines = TRIGRAM_LINES.get(to_hex, [0, 0, 0])
    changes = []
    for i in range(3):
        if from_lines[i] != to_lines[i]:
            changes.append(i + 1)
    return changes if changes else None


# ===== Phase 1: 現代テーマ =====

PHASE1_TEMPLATES = {
    "SNS・承認欲求": [
        {
            "target_name": "インフルエンサー志望_{n}",
            "scale": "individual",
            "period": "2020-2024",
            "before_state": "成長痛",
            "trigger_type": "意図的決断",
            "action_type": "攻める・挑戦",
            "before_hex": "兌",
            "trigger_hex": "離",
            "action_hex": "震",
            "pattern_type": "Pivot_Success",
            "source_type": "sns",
            "credibility_rank": "B",
            "free_tags": ["#インフルエンサー", "#SNS", "#承認欲求"],
            "stories": [
                "フォロワー1万人を目指してSNS発信を始めた{職業}。毎日投稿を続けるも、いいね数に一喜一憂する日々。{結末}",
                "会社員をしながらYouTubeを始めた{年代}。チャンネル登録者1000人で壁にぶつかり、{結末}",
                "インスタ映えを追求するあまり、私生活が破綻寸前に。{結末}",
            ]
        },
        {
            "target_name": "SNS疲れ_{n}",
            "scale": "individual",
            "period": "2021-2024",
            "before_state": "成長痛",
            "trigger_type": "内部崩壊",
            "action_type": "捨てる・撤退",
            "before_hex": "兌",
            "trigger_hex": "坎",
            "action_hex": "艮",
            "pattern_type": "Endurance",
            "source_type": "blog",
            "credibility_rank": "B",
            "free_tags": ["#SNS疲れ", "#デジタルデトックス", "#承認欲求"],
            "stories": [
                "毎日のいいね数チェックが習慣化し、自己肯定感がSNSの評価に依存。精神的に追い詰められ{結末}",
                "フォロワー数が全てだと思っていた{年代}。炎上をきっかけにSNSを休止、{結末}",
                "インスタの更新が義務になり、楽しさを見失った。思い切ってアカウントを削除し{結末}",
            ]
        },
        {
            "target_name": "炎上経験者_{n}",
            "scale": "individual",
            "period": "2020-2024",
            "before_state": "絶頂・慢心",
            "trigger_type": "外部ショック",
            "action_type": "耐える・潜伏",
            "before_hex": "乾",
            "trigger_hex": "震",
            "action_hex": "坎",
            "pattern_type": "Hubris_Collapse",
            "source_type": "news",
            "credibility_rank": "A",
            "free_tags": ["#炎上", "#SNS", "#承認欲求", "#批判"],
            "stories": [
                "人気インフルエンサーだった{職業}が不用意な発言で大炎上。フォロワーが激減し{結末}",
                "バズった投稿の反動で誹謗中傷の嵐。精神的に追い詰められ{結末}",
                "調子に乗った発言が切り取られて拡散。謝罪するも火に油を注ぎ{結末}",
            ]
        },
    ],
    "リモートワーク・オンライン": [
        {
            "target_name": "リモートワーカー_{n}",
            "scale": "individual",
            "period": "2020-2024",
            "before_state": "安定・平和",
            "trigger_type": "外部ショック",
            "action_type": "守る・維持",
            "before_hex": "坤",
            "trigger_hex": "震",
            "action_hex": "艮",
            "pattern_type": "Endurance",
            "source_type": "article",
            "credibility_rank": "B",
            "free_tags": ["#リモートワーク", "#在宅勤務", "#コロナ"],
            "stories": [
                "コロナ禍で突然の在宅勤務に。最初は快適だったが、次第に孤独感が増し{結末}",
                "フルリモートで地方移住を決意した{年代}。理想と現実のギャップに苦しみ{結末}",
                "オンライン会議漬けの毎日に疲弊。コミュニケーションの質が低下し{結末}",
            ]
        },
        {
            "target_name": "オンライン孤立_{n}",
            "scale": "individual",
            "period": "2021-2024",
            "before_state": "停滞・閉塞",
            "trigger_type": "内部崩壊",
            "action_type": "対話・融合",
            "before_hex": "艮",
            "trigger_hex": "坎",
            "action_hex": "兌",
            "pattern_type": "Slow_Decline",
            "source_type": "blog",
            "credibility_rank": "C",
            "free_tags": ["#オンライン孤立", "#リモート", "#コミュニケーション不足"],
            "stories": [
                "リモートワークで同僚との雑談がゼロに。会社への帰属意識が薄れ{結末}",
                "オンラインだけの人間関係に限界を感じた{年代}。対面の場を求めて{結末}",
                "Slackのメッセージだけでは伝わらない微妙なニュアンス。誤解が積み重なり{結末}",
            ]
        },
    ],
    "AI・テクノロジー不安": [
        {
            "target_name": "AI脅威論者_{n}",
            "scale": "individual",
            "period": "2023-2024",
            "before_state": "混乱・カオス",
            "trigger_type": "外部ショック",
            "action_type": "攻める・挑戦",
            "before_hex": "震",
            "trigger_hex": "離",
            "action_hex": "乾",
            "pattern_type": "Shock_Recovery",
            "source_type": "news",
            "credibility_rank": "A",
            "free_tags": ["#AI", "#ChatGPT", "#失業不安", "#リスキリング"],
            "stories": [
                "ChatGPTの登場で自分の仕事が奪われると感じた{職業}。焦ってリスキリングを始め{結末}",
                "AIに代替されない仕事とは何かを真剣に考え始めた{年代}。{結末}",
                "会社がAI導入を発表。自分の部署が縮小されると知り{結末}",
            ]
        },
        {
            "target_name": "AI活用成功_{n}",
            "scale": "individual",
            "period": "2023-2024",
            "before_state": "成長痛",
            "trigger_type": "意図的決断",
            "action_type": "刷新・破壊",
            "before_hex": "震",
            "trigger_hex": "離",
            "action_hex": "乾",
            "pattern_type": "Pivot_Success",
            "source_type": "article",
            "credibility_rank": "B",
            "free_tags": ["#AI活用", "#生産性向上", "#DX"],
            "stories": [
                "AIツールを積極的に取り入れた{職業}。業務効率が3倍になり{結末}",
                "ChatGPTを使いこなして副業収入を得るようになった{年代}。{結末}",
                "AI嫌いだった上司を説得し、部署全体でAI導入を推進。{結末}",
            ]
        },
        {
            "target_name": "DX失敗_{n}",
            "scale": "company",
            "period": "2020-2024",
            "before_state": "停滞・閉塞",
            "trigger_type": "意図的決断",
            "action_type": "刷新・破壊",
            "before_hex": "坤",
            "trigger_hex": "震",
            "action_hex": "離",
            "pattern_type": "Slow_Decline",
            "source_type": "news",
            "credibility_rank": "A",
            "free_tags": ["#DX失敗", "#デジタル化", "#レガシー"],
            "stories": [
                "鳴り物入りで始めたDXプロジェクトが頓挫。現場の抵抗に遭い{結末}",
                "高額なシステム投資をしたが、社員が使いこなせず{結末}",
                "外注に丸投げしたDXが大失敗。自社のノウハウが蓄積されず{結末}",
            ]
        },
    ],
    "Z世代・令和の価値観": [
        {
            "target_name": "Z世代社員_{n}",
            "scale": "individual",
            "period": "2022-2024",
            "before_state": "成長痛",
            "trigger_type": "意図的決断",
            "action_type": "捨てる・撤退",
            "before_hex": "震",
            "trigger_hex": "巽",
            "action_hex": "艮",
            "pattern_type": "Pivot_Success",
            "source_type": "article",
            "credibility_rank": "B",
            "free_tags": ["#Z世代", "#早期退職", "#ホワイト企業", "#タイパ"],
            "stories": [
                "入社1年で「成長できない」と感じ転職を決意した{年代}。{結末}",
                "残業のない会社を選んだが、物足りなさを感じ{結末}",
                "「ゆるブラック」に耐えられず、ベンチャーに飛び込み{結末}",
            ]
        },
        {
            "target_name": "ワークライフバランス追求_{n}",
            "scale": "individual",
            "period": "2020-2024",
            "before_state": "成長痛",
            "trigger_type": "意図的決断",
            "action_type": "守る・維持",
            "before_hex": "震",
            "trigger_hex": "坤",
            "action_hex": "艮",
            "pattern_type": "Steady_Growth",
            "source_type": "blog",
            "credibility_rank": "C",
            "free_tags": ["#ワークライフバランス", "#副業", "#複業"],
            "stories": [
                "本業と副業のバランスを取りながら収入を増やした{年代}。{結末}",
                "定時退社を徹底し、趣味の時間を確保。人生の満足度が上がり{結末}",
                "「静かな退職」を実践。最低限の仕事だけして{結末}",
            ]
        },
        {
            "target_name": "FIRE目指し_{n}",
            "scale": "individual",
            "period": "2020-2024",
            "before_state": "安定・平和",
            "trigger_type": "意図的決断",
            "action_type": "攻める・挑戦",
            "before_hex": "坤",
            "trigger_hex": "震",
            "action_hex": "乾",
            "pattern_type": "Pivot_Success",
            "source_type": "blog",
            "credibility_rank": "C",
            "free_tags": ["#FIRE", "#投資", "#経済的自立", "#早期リタイア"],
            "stories": [
                "30代でFIREを達成した{職業}。しかし達成後に虚無感に襲われ{結末}",
                "インデックス投資を続けて資産1億円を達成。会社を辞めて{結末}",
                "FIRE目指して節約生活。しかし途中で心が折れ{結末}",
            ]
        },
    ],
}

# ===== Phase 2: 八卦バランス =====

PHASE2_TEMPLATES = {
    "兌（交流・喜び）": [
        {
            "target_name": "コミュニティ運営_{n}",
            "scale": "other",
            "period": "2018-2024",
            "before_state": "成長痛",
            "trigger_type": "意図的決断",
            "action_type": "対話・融合",
            "before_hex": "兌",
            "trigger_hex": "離",
            "action_hex": "巽",
            "pattern_type": "Steady_Growth",
            "source_type": "blog",
            "credibility_rank": "B",
            "free_tags": ["#コミュニティ", "#オンラインサロン", "#つながり"],
            "stories": [
                "趣味のオンラインコミュニティを立ち上げた{年代}。会員が増えて運営が大変になり{結末}",
                "地域のボランティア団体を活性化させた{職業}。{結末}",
                "SNSで出会った仲間とオフ会を開催。そこから新しいプロジェクトが生まれ{結末}",
            ]
        },
        {
            "target_name": "人気商売_{n}",
            "scale": "individual",
            "period": "2015-2024",
            "before_state": "成長痛",
            "trigger_type": "偶発・出会い",
            "action_type": "攻める・挑戦",
            "before_hex": "兌",
            "trigger_hex": "震",
            "action_hex": "離",
            "pattern_type": "Pivot_Success",
            "source_type": "news",
            "credibility_rank": "A",
            "free_tags": ["#芸能", "#エンタメ", "#人気", "#ファン"],
            "stories": [
                "下積み時代を経てブレイクした{職業}。しかし人気の維持に苦しみ{結末}",
                "一発屋で終わると思われたが、地道な活動で復活した{年代}。{結末}",
                "ファンとの交流を大切にし、長く愛されるタレントに。{結末}",
            ]
        },
    ],
    "巽（適応・浸透）": [
        {
            "target_name": "営業マン_{n}",
            "scale": "individual",
            "period": "2010-2024",
            "before_state": "成長痛",
            "trigger_type": "意図的決断",
            "action_type": "対話・融合",
            "before_hex": "巽",
            "trigger_hex": "兌",
            "action_hex": "離",
            "pattern_type": "Steady_Growth",
            "source_type": "book",
            "credibility_rank": "A",
            "free_tags": ["#営業", "#交渉", "#人脈", "#コミュニケーション"],
            "stories": [
                "断られても諦めない営業スタイルで成績トップに。しかし燃え尽き{結末}",
                "顧客の話を聞くことに徹した{年代}。信頼を勝ち取り{結末}",
                "押し売りではなく、提案型営業に切り替えて成功した{職業}。{結末}",
            ]
        },
        {
            "target_name": "適応上手_{n}",
            "scale": "individual",
            "period": "2015-2024",
            "before_state": "混乱・カオス",
            "trigger_type": "外部ショック",
            "action_type": "守る・維持",
            "before_hex": "巽",
            "trigger_hex": "坎",
            "action_hex": "艮",
            "pattern_type": "Endurance",
            "source_type": "article",
            "credibility_rank": "B",
            "free_tags": ["#適応力", "#柔軟性", "#変化対応"],
            "stories": [
                "何度も部署異動を経験した{職業}。その都度適応し{結末}",
                "業界の急激な変化に対応し続けた{年代}。{結末}",
                "変化を楽しめる性格が功を奏し、キャリアを築いた。{結末}",
            ]
        },
    ],
    "離（表現・情熱）": [
        {
            "target_name": "クリエイター_{n}",
            "scale": "individual",
            "period": "2015-2024",
            "before_state": "成長痛",
            "trigger_type": "意図的決断",
            "action_type": "攻める・挑戦",
            "before_hex": "離",
            "trigger_hex": "乾",
            "action_hex": "震",
            "pattern_type": "Pivot_Success",
            "source_type": "article",
            "credibility_rank": "B",
            "free_tags": ["#クリエイター", "#表現", "#アート", "#創作"],
            "stories": [
                "会社員を辞めてフリーのクリエイターに。不安定な収入に苦しみながらも{結末}",
                "SNSで作品を発信し続けた{年代}。ある投稿がバズって{結末}",
                "自分の表現を追求し続けた{職業}。商業的成功より{結末}",
            ]
        },
        {
            "target_name": "燃え尽き症候群_{n}",
            "scale": "individual",
            "period": "2018-2024",
            "before_state": "成長痛",
            "trigger_type": "内部崩壊",
            "action_type": "耐える・潜伏",
            "before_hex": "離",
            "trigger_hex": "坎",
            "action_hex": "艮",
            "pattern_type": "Slow_Decline",
            "source_type": "blog",
            "credibility_rank": "C",
            "free_tags": ["#燃え尽き", "#バーンアウト", "#過労", "#休職"],
            "stories": [
                "情熱を持って働き続けた{職業}。ある日突然、体が動かなくなり{結末}",
                "成果を出し続けるプレッシャーに押しつぶされた{年代}。{結末}",
                "好きなことを仕事にしたはずが、嫌いになってしまい{結末}",
            ]
        },
    ],
}

# ===== Phase 3: スケール拡充 =====

PHASE3_TEMPLATES = {
    "individual追加": [
        {
            "target_name": "キャリアチェンジ_{n}",
            "scale": "individual",
            "period": "2015-2024",
            "before_state": "停滞・閉塞",
            "trigger_type": "意図的決断",
            "action_type": "刷新・破壊",
            "before_hex": "坤",
            "trigger_hex": "震",
            "action_hex": "離",
            "pattern_type": "Pivot_Success",
            "source_type": "article",
            "credibility_rank": "B",
            "free_tags": ["#転職", "#キャリアチェンジ", "#未経験"],
            "stories": [
                "30代で未経験からエンジニアに転身した{年代}。{結末}",
                "安定した仕事を捨てて、夢だった{職業}に挑戦。{結末}",
                "リストラをきっかけに全く違う業界へ。{結末}",
            ]
        },
        {
            "target_name": "メンタルヘルス_{n}",
            "scale": "individual",
            "period": "2018-2024",
            "before_state": "どん底・危機",
            "trigger_type": "内部崩壊",
            "action_type": "耐える・潜伏",
            "before_hex": "坎",
            "trigger_hex": "艮",
            "action_hex": "坤",
            "pattern_type": "Endurance",
            "source_type": "blog",
            "credibility_rank": "C",
            "free_tags": ["#うつ", "#メンタルヘルス", "#休職", "#復職"],
            "stories": [
                "うつ病で1年間休職した{職業}。回復までの道のりは長く{結末}",
                "パニック障害を抱えながら働き続けた{年代}。{結末}",
                "心療内科に通いながら、少しずつ日常を取り戻し{結末}",
            ]
        },
    ],
    "company追加": [
        {
            "target_name": "スタートアップ_{n}",
            "scale": "company",
            "period": "2018-2024",
            "before_state": "成長痛",
            "trigger_type": "意図的決断",
            "action_type": "攻める・挑戦",
            "before_hex": "震",
            "trigger_hex": "乾",
            "action_hex": "離",
            "pattern_type": "Pivot_Success",
            "source_type": "news",
            "credibility_rank": "A",
            "free_tags": ["#スタートアップ", "#起業", "#VC", "#資金調達"],
            "stories": [
                "シード期の資金調達に成功したスタートアップ。しかしPMFに苦戦し{結末}",
                "ピボットを繰り返した末に、ようやく成長軌道に乗り{結末}",
                "創業メンバーの対立で空中分解しかけたが{結末}",
            ]
        },
        {
            "target_name": "老舗企業_{n}",
            "scale": "company",
            "period": "2010-2024",
            "before_state": "停滞・閉塞",
            "trigger_type": "外部ショック",
            "action_type": "刷新・破壊",
            "before_hex": "坤",
            "trigger_hex": "震",
            "action_hex": "離",
            "pattern_type": "Shock_Recovery",
            "source_type": "news",
            "credibility_rank": "A",
            "free_tags": ["#老舗", "#事業転換", "#伝統"],
            "stories": [
                "創業100年の老舗がDXに挑戦。伝統と革新の両立に苦しみ{結末}",
                "コロナ禍で主力事業が壊滅。新規事業への転換を迫られ{結末}",
                "後継者不在で廃業寸前だったが、M&Aで存続し{結末}",
            ]
        },
    ],
    "family追加": [
        {
            "target_name": "介護家族_{n}",
            "scale": "family",
            "period": "2015-2024",
            "before_state": "安定・平和",
            "trigger_type": "外部ショック",
            "action_type": "耐える・潜伏",
            "before_hex": "坤",
            "trigger_hex": "坎",
            "action_hex": "艮",
            "pattern_type": "Endurance",
            "source_type": "blog",
            "credibility_rank": "C",
            "free_tags": ["#介護", "#親の介護", "#介護離職", "#ヤングケアラー"],
            "stories": [
                "親の突然の認知症発症で、介護生活が始まった。仕事との両立に苦しみ{結末}",
                "遠距離介護を続けた{年代}。心身ともに疲弊し{結末}",
                "介護のために仕事を辞めた。経済的な不安と孤独に苛まれ{結末}",
            ]
        },
        {
            "target_name": "相続問題_{n}",
            "scale": "family",
            "period": "2010-2024",
            "before_state": "安定・平和",
            "trigger_type": "外部ショック",
            "action_type": "対話・融合",
            "before_hex": "坤",
            "trigger_hex": "震",
            "action_hex": "兌",
            "pattern_type": "Slow_Decline",
            "source_type": "article",
            "credibility_rank": "B",
            "free_tags": ["#相続", "#遺産争い", "#家族崩壊"],
            "stories": [
                "親の死後、兄弟間で遺産争いが勃発。絶縁状態に{結末}",
                "実家の土地をめぐって親戚一同が対立。調停を経て{結末}",
                "相続税対策を怠ったために、家を売却せざるを得なくなり{結末}",
            ]
        },
    ],
    "country追加": [
        {
            "target_name": "政策転換_{n}",
            "scale": "country",
            "period": "2010-2024",
            "before_state": "停滞・閉塞",
            "trigger_type": "意図的決断",
            "action_type": "刷新・破壊",
            "before_hex": "坤",
            "trigger_hex": "乾",
            "action_hex": "震",
            "pattern_type": "Pivot_Success",
            "source_type": "news",
            "credibility_rank": "S",
            "free_tags": ["#政策", "#改革", "#規制緩和"],
            "stories": [
                "長年の規制を撤廃し、新産業が勃興した。{結末}",
                "少子化対策として大胆な政策を打ち出したが{結末}",
                "財政再建と経済成長の両立を目指したが{結末}",
            ]
        },
    ],
    "other追加": [
        {
            "target_name": "業界団体_{n}",
            "scale": "other",
            "period": "2015-2024",
            "before_state": "停滞・閉塞",
            "trigger_type": "外部ショック",
            "action_type": "刷新・破壊",
            "before_hex": "艮",
            "trigger_hex": "震",
            "action_hex": "離",
            "pattern_type": "Shock_Recovery",
            "source_type": "news",
            "credibility_rank": "A",
            "free_tags": ["#業界", "#団体", "#変革"],
            "stories": [
                "旧態依然とした業界に新規参入者が現れ、既存企業は{結末}",
                "規制緩和で業界地図が一変。生き残りをかけて{結末}",
                "業界全体でDXを推進したが、足並みが揃わず{結末}",
            ]
        },
    ],
}

# ===== Phase 4: 時代バランス =====

PHASE4_TEMPLATES = {
    "昭和事例": [
        {
            "target_name": "高度成長期_{n}",
            "scale": "company",
            "period": "1960-1975",
            "before_state": "成長痛",
            "trigger_type": "意図的決断",
            "action_type": "攻める・挑戦",
            "before_hex": "震",
            "trigger_hex": "乾",
            "action_hex": "離",
            "pattern_type": "Steady_Growth",
            "source_type": "book",
            "credibility_rank": "S",
            "free_tags": ["#高度成長", "#昭和", "#モーレツ社員"],
            "stories": [
                "戦後復興から高度成長へ。猛烈に働いた社員たちが日本を支え{結末}",
                "町工場から始まった企業が、輸出で急成長。{結末}",
                "終身雇用と年功序列の中で、安定したキャリアを築き{結末}",
            ]
        },
        {
            "target_name": "バブル時代_{n}",
            "scale": "individual",
            "period": "1985-1991",
            "before_state": "絶頂・慢心",
            "trigger_type": "外部ショック",
            "action_type": "耐える・潜伏",
            "before_hex": "乾",
            "trigger_hex": "坎",
            "action_hex": "艮",
            "pattern_type": "Hubris_Collapse",
            "source_type": "book",
            "credibility_rank": "A",
            "free_tags": ["#バブル", "#昭和", "#崩壊", "#不動産"],
            "stories": [
                "バブル期に不動産投資で財を成したが、崩壊後に{結末}",
                "派手な生活を送っていたが、バブル崩壊で一転{結末}",
                "「土地は必ず上がる」と信じて借金をしたが{結末}",
            ]
        },
    ],
    "平成事例": [
        {
            "target_name": "就職氷河期_{n}",
            "scale": "individual",
            "period": "1993-2005",
            "before_state": "どん底・危機",
            "trigger_type": "外部ショック",
            "action_type": "耐える・潜伏",
            "before_hex": "坎",
            "trigger_hex": "艮",
            "action_hex": "坤",
            "pattern_type": "Endurance",
            "source_type": "article",
            "credibility_rank": "A",
            "free_tags": ["#氷河期", "#平成", "#ロスジェネ", "#非正規"],
            "stories": [
                "100社受けても内定がもらえなかった{年代}。非正規から這い上がり{結末}",
                "正社員になれず、派遣で食いつないだ氷河期世代。{結末}",
                "夢を諦めて手に職をつけることを選び{結末}",
            ]
        },
        {
            "target_name": "ITバブル_{n}",
            "scale": "company",
            "period": "1999-2001",
            "before_state": "絶頂・慢心",
            "trigger_type": "外部ショック",
            "action_type": "捨てる・撤退",
            "before_hex": "乾",
            "trigger_hex": "坎",
            "action_hex": "艮",
            "pattern_type": "Hubris_Collapse",
            "source_type": "news",
            "credibility_rank": "A",
            "free_tags": ["#ITバブル", "#平成", "#ドットコム", "#崩壊"],
            "stories": [
                "上場してCEOになったが、ITバブル崩壊で株価暴落。{結末}",
                "ネット企業に転職したが、半年で会社が消滅{結末}",
                "時価総額100億と言われたベンチャーが{結末}",
            ]
        },
    ],
    "令和事例": [
        {
            "target_name": "コロナ禍_{n}",
            "scale": "company",
            "period": "2020-2023",
            "before_state": "安定・平和",
            "trigger_type": "外部ショック",
            "action_type": "刷新・破壊",
            "before_hex": "坤",
            "trigger_hex": "坎",
            "action_hex": "震",
            "pattern_type": "Shock_Recovery",
            "source_type": "news",
            "credibility_rank": "S",
            "free_tags": ["#コロナ", "#令和", "#パンデミック", "#ピボット"],
            "stories": [
                "コロナで売上が9割減。ピボットして新事業を立ち上げ{結末}",
                "飲食店がテイクアウト専門に切り替えて生き残り{結末}",
                "イベント業が壊滅。オンライン化に活路を見出し{結末}",
            ]
        },
        {
            "target_name": "令和の働き方_{n}",
            "scale": "individual",
            "period": "2020-2024",
            "before_state": "成長痛",
            "trigger_type": "意図的決断",
            "action_type": "刷新・破壊",
            "before_hex": "震",
            "trigger_hex": "巽",
            "action_hex": "離",
            "pattern_type": "Pivot_Success",
            "source_type": "article",
            "credibility_rank": "B",
            "free_tags": ["#令和", "#新しい働き方", "#副業", "#フリーランス"],
            "stories": [
                "会社員を辞めてフリーランスに。収入は不安定だが{結末}",
                "副業を始めて本業の収入を超えた{年代}。{結末}",
                "複数の会社と業務委託契約を結ぶ新しい働き方{結末}",
            ]
        },
    ],
}


# ===== 生成用ユーティリティ =====

OCCUPATIONS = [
    "会社員", "エンジニア", "営業職", "事務職", "管理職", "経営者",
    "フリーランス", "クリエイター", "デザイナー", "ライター", "コンサルタント",
    "教師", "医療従事者", "公務員", "研究者", "職人", "販売員"
]

AGES = [
    "20代前半", "20代後半", "30代前半", "30代後半", "40代前半",
    "40代後半", "50代前半", "50代後半", "60代"
]

OUTCOMES_SUCCESS = [
    "成功を収めた。",
    "見事に復活した。",
    "新たな道を切り開いた。",
    "人生が好転した。",
    "目標を達成した。",
    "安定を取り戻した。",
]

OUTCOMES_PARTIAL = [
    "なんとか持ちこたえた。",
    "完全ではないが回復した。",
    "一定の成果を得た。",
    "まだ道半ばだが前進した。",
]

OUTCOMES_FAILURE = [
    "挫折した。",
    "失敗に終わった。",
    "状況は悪化した。",
    "諦めざるを得なかった。",
    "崩壊した。",
]

OUTCOMES_MIXED = [
    "結果は複雑だった。",
    "良い面と悪い面があった。",
    "予想とは違う結末になった。",
    "今も模索が続いている。",
]


def generate_story(template: str, outcome_type: str) -> str:
    """ストーリーを生成"""
    occupation = random.choice(OCCUPATIONS)
    age = random.choice(AGES)

    if outcome_type == "Success":
        ending = random.choice(OUTCOMES_SUCCESS)
    elif outcome_type == "PartialSuccess":
        ending = random.choice(OUTCOMES_PARTIAL)
    elif outcome_type == "Failure":
        ending = random.choice(OUTCOMES_FAILURE)
    else:
        ending = random.choice(OUTCOMES_MIXED)

    story = template.format(職業=occupation, 年代=age, 結末=ending)
    return story


def determine_outcome(pattern_type: str) -> str:
    """パターンタイプから結果を決定"""
    if pattern_type in ["Pivot_Success", "Steady_Growth", "Shock_Recovery"]:
        return random.choices(
            ["Success", "PartialSuccess", "Mixed"],
            weights=[0.6, 0.25, 0.15]
        )[0]
    elif pattern_type == "Hubris_Collapse":
        return random.choices(
            ["Failure", "Mixed", "PartialSuccess"],
            weights=[0.5, 0.3, 0.2]
        )[0]
    elif pattern_type == "Slow_Decline":
        return random.choices(
            ["Failure", "Mixed", "PartialSuccess"],
            weights=[0.4, 0.35, 0.25]
        )[0]
    else:  # Endurance
        return random.choices(
            ["PartialSuccess", "Success", "Mixed"],
            weights=[0.4, 0.35, 0.25]
        )[0]


def determine_after_state(outcome: str, pattern_type: str) -> str:
    """結果とパターンから最終状態を決定"""
    if outcome == "Success":
        return random.choice([
            "V字回復・大成功", "持続成長・大成功", "安定成長・成功"
        ])
    elif outcome == "PartialSuccess":
        return random.choice([
            "縮小安定・生存", "安定・平和", "変質・新生"
        ])
    elif outcome == "Failure":
        return random.choice([
            "崩壊・消滅", "停滞・閉塞", "どん底・危機"
        ])
    else:  # Mixed
        return random.choice([
            "変質・新生", "現状維持・延命", "迷走・混乱"
        ])


def determine_after_hex(outcome: str, action_hex: str) -> str:
    """結果とアクション八卦から最終八卦を決定"""
    if outcome == "Success":
        return random.choice(["乾", "離", "震"])
    elif outcome == "PartialSuccess":
        return random.choice(["坤", "艮", "巽"])
    elif outcome == "Failure":
        return random.choice(["坎", "坤", "艮"])
    else:
        return random.choice(["巽", "兌", "坤"])


def generate_cases_from_template(template: Dict, count: int, start_n: int = 1) -> List[Dict]:
    """テンプレートから事例を生成"""
    cases = []
    stories = template.get("stories", [])

    for i in range(count):
        n = start_n + i
        story_template = random.choice(stories)
        outcome = determine_outcome(template["pattern_type"])
        story = generate_story(story_template, outcome)

        after_state = determine_after_state(outcome, template["pattern_type"])
        after_hex = determine_after_hex(outcome, template["action_hex"])

        case = {
            "target_name": template["target_name"].format(n=n),
            "scale": template["scale"],
            "period": template["period"],
            "story_summary": story,
            "before_state": template["before_state"],
            "trigger_type": template["trigger_type"],
            "action_type": template["action_type"],
            "after_state": after_state,
            "before_hex": template["before_hex"],
            "trigger_hex": template["trigger_hex"],
            "action_hex": template["action_hex"],
            "after_hex": after_hex,
            "pattern_type": template["pattern_type"],
            "outcome": outcome,
            "free_tags": template["free_tags"],
            "source_type": template["source_type"],
            "credibility_rank": template["credibility_rank"],
        }

        # 変爻を計算
        cl1 = infer_changing_lines(case["before_hex"], case["trigger_hex"])
        cl2 = infer_changing_lines(case["trigger_hex"], case["action_hex"])
        cl3 = infer_changing_lines(case["action_hex"], case["after_hex"])

        if cl1:
            case["changing_lines_1"] = cl1
        if cl2:
            case["changing_lines_2"] = cl2
        if cl3:
            case["changing_lines_3"] = cl3

        cases.append(case)

    return cases


def generate_phase(phase_num: int, templates: Dict, cases_per_template: int = 50) -> List[Dict]:
    """フェーズ全体の事例を生成"""
    all_cases = []
    global_n = 1

    for category, template_list in templates.items():
        for template in template_list:
            cases = generate_cases_from_template(template, cases_per_template, global_n)
            all_cases.extend(cases)
            global_n += cases_per_template

    return all_cases


def main():
    parser = argparse.ArgumentParser(description="事例データのバッチ生成")
    parser.add_argument("--phase", type=str, default="all",
                       help="生成するフェーズ (1, 2, 3, 4, all)")
    parser.add_argument("--output", type=str, default="data/import/generated.json",
                       help="出力ファイルパス")
    parser.add_argument("--count", type=int, default=50,
                       help="テンプレートあたりの生成数")

    args = parser.parse_args()

    all_cases = []

    if args.phase in ["1", "all"]:
        print("Phase 1: 現代テーマ生成中...")
        cases = generate_phase(1, PHASE1_TEMPLATES, args.count)
        all_cases.extend(cases)
        print(f"  {len(cases)}件生成")

    if args.phase in ["2", "all"]:
        print("Phase 2: 八卦バランス生成中...")
        cases = generate_phase(2, PHASE2_TEMPLATES, args.count)
        all_cases.extend(cases)
        print(f"  {len(cases)}件生成")

    if args.phase in ["3", "all"]:
        print("Phase 3: スケール拡充生成中...")
        cases = generate_phase(3, PHASE3_TEMPLATES, args.count)
        all_cases.extend(cases)
        print(f"  {len(cases)}件生成")

    if args.phase in ["4", "all"]:
        print("Phase 4: 時代バランス生成中...")
        cases = generate_phase(4, PHASE4_TEMPLATES, args.count)
        all_cases.extend(cases)
        print(f"  {len(cases)}件生成")

    # 出力
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)

    print(f"\n合計 {len(all_cases)}件 を {output_path} に出力しました")


if __name__ == "__main__":
    main()
