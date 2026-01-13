"""
品質判定ルール設定ファイル v2
修正: ドメインリスト拡張、success_level算出ルール明確化
"""

# ============================================
# 1. 信頼ドメインリスト（拡張版）
# ============================================
TRUSTED_DOMAINS = {
    # Tier 1: 公式・政府・学術（最高信頼）
    'tier1_official': [
        'go.jp', 'gov', 'ac.jp', 'edu', '.org',
        'meti.go.jp', 'mof.go.jp', 'cao.go.jp', 'jftc.go.jp',
    ],

    # Tier 2: 主要メディア（高信頼）
    'tier2_major_media_jp': [
        'nikkei.com', 'asahi.com', 'yomiuri.co.jp', 'mainichi.jp',
        'sankei.com', 'nhk.or.jp', 'jiji.com', 'kyodo.co.jp',
    ],
    'tier2_major_media_intl': [
        'reuters.com', 'bloomberg.com', 'wsj.com', 'nytimes.com',
        'ft.com', 'economist.com', 'bbc.com', 'bbc.co.uk',
        'cnn.com', 'apnews.com', 'afp.com',
    ],

    # Tier 3: ビジネス・テック専門メディア（中〜高信頼）
    'tier3_business_jp': [
        'toyokeizai.net', 'diamond.jp', 'sbbit.jp', 'newspicks.com',
        'president.jp', 'businessinsider.jp', 'forbesjapan.com',
    ],
    'tier3_tech': [
        'techcrunch.com', 'theverge.com', 'wired.com', 'wired.jp',
        'itmedia.co.jp', 'impress.co.jp', 'cnet.com', 'zdnet.com',
        'engadget.com', 'gizmodo.com', 'arstechnica.com',
    ],

    # Tier 4: プレスリリース・企業公式（中信頼）
    'tier4_pr': [
        'prtimes.jp', 'pr-times.com', 'prwire.com', 'businesswire.com',
        'globenewswire.com', 'prnewswire.com',
    ],
    'tier4_corporate': [
        # 企業ドメインは個別判定が必要なため空
    ],

    # Tier 5: その他ニュース（要検証）
    'tier5_other_news': [
        'huffingtonpost.jp', 'buzzfeed.com', 'vice.com',
        'j-cast.com', 'iza.ne.jp', 'zakzak.co.jp',
    ],
}

# Gold判定に使用するTier
GOLD_ELIGIBLE_TIERS = ['tier1_official', 'tier2_major_media_jp', 'tier2_major_media_intl']
PLAUSIBLE_ELIGIBLE_TIERS = GOLD_ELIGIBLE_TIERS + ['tier3_business_jp', 'tier3_tech', 'tier4_pr']

# ============================================
# 2. 匿名判定パターン（明文化）
# ============================================
ANONYMOUS_PATTERNS = [
    r'[A-Z]さん[_＿]',      # "Pさん_"
    r'[あ-んア-ン]さん[_＿]', # "たろうさん_"
    r'^匿名',              # "匿名企業"
    r'^Anonymous',
    r'^Unknown',
    r'某[企業会社組織団体]', # "某企業"
    r'架空',
    r'仮名',
    r'サンプル',
    r'^XX?$',              # "X", "XX"
]

# ============================================
# 3. rejected URL パターン（明文化）
# ============================================
REJECTED_URL_PATTERNS = [
    r'google\.(com|co\.jp|[a-z]{2})/search',
    r'bing\.com/search',
    r'yahoo\.(com|co\.jp)/search',
    r'duckduckgo\.com/\?q=',
    r'search\.',           # search.yahoo.com等
]

# ============================================
# 4. success_level算出ルール（明文化）
# ============================================
SUCCESS_LEVEL_RULES = {
    # 最小サンプル数ルール
    'min_n_for_display': 5,        # 表示に必要な最小n
    'min_n_for_reliable': 20,      # 信頼できる最小n
    'min_n_for_production': 50,    # プロダクション使用の最小n

    # ベイズ平滑化パラメータ
    'smoothing_alpha': 2.0,        # α=2でより保守的に
    'prior_success_rate': 0.5,     # 事前確率50%

    # 出力ルール
    'always_show_n': True,         # 常にnを表示
    'always_show_ci': True,        # 常に信頼区間を表示
    'flag_low_n': True,            # n<20は警告フラグ

    # パターン名バイアス対策
    # パターン名に結果が含まれるため、パターン別成功率は参考値として扱う
    'pattern_success_is_reference_only': True,
}

# ============================================
# 5. 品質Tier定義
# ============================================
QUALITY_TIERS = {
    'gold': {
        'description': 'Tier1-2ソース + verified + 実名',
        'min_sources': 1,
        'eligible_tiers': GOLD_ELIGIBLE_TIERS,
        'require_verified': True,
    },
    'silver': {
        'description': 'Tier3-4ソース + plausible以上 + 実名',
        'min_sources': 1,
        'eligible_tiers': PLAUSIBLE_ELIGIBLE_TIERS,
        'require_verified': False,
    },
    'bronze': {
        'description': '有効ソースあり + 実名',
        'min_sources': 1,
        'eligible_tiers': None,  # Any valid source
        'require_verified': False,
    },
    'quarantine': {
        'description': 'ソースなし / rejected URL / 匿名',
    },
}

# ============================================
# 6. 国コード正規化（ISO 3166-1 alpha-2）
# ============================================
COUNTRY_NORMALIZATION = {
    '日本': 'JP', 'Japan': 'JP', 'japan': 'JP',
    'アメリカ': 'US', 'USA': 'US', 'United States': 'US', 'america': 'US',
    '中国': 'CN', 'China': 'CN', 'china': 'CN',
    '韓国': 'KR', 'Korea': 'KR', 'South Korea': 'KR',
    'イギリス': 'GB', 'UK': 'GB', 'United Kingdom': 'GB', 'Britain': 'GB',
    'ドイツ': 'DE', 'Germany': 'DE',
    'フランス': 'FR', 'France': 'FR',
    'ヨーロッパ': 'EU', 'Europe': 'EU',  # 特殊コード
    'グローバル': 'XX', 'Global': 'XX', '世界': 'XX',  # 特殊コード
}
