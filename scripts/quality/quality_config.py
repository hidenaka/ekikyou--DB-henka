"""
品質判定ルール設定ファイル
再現性のため、全判定基準をここに明文化
"""

# ============================================
# 1. 匿名・半架空パターン（rejected判定用）
# ============================================
ANONYMOUS_PATTERNS = [
    r'[A-Z]さん[_＿]',      # "Pさん_", "Tさん＿"
    r'[あ-ん]さん[_＿]',    # "たろうさん_"
    r'^匿名',              # "匿名企業"
    r'^Anonymous',
    r'^Unknown',
    r'某[企業会社]',        # "某企業"
    r'架空',
    r'仮名',
    r'サンプル[企業会社]',
]

# ============================================
# 2. 有効ソースドメインホワイトリスト
# ============================================
TRUSTED_DOMAINS = {
    # 公式・政府
    'official': [
        'go.jp', 'gov', 'ac.jp', 'edu', 'org',
        'meti.go.jp', 'mof.go.jp', 'cao.go.jp',
    ],
    # 主要メディア（日本）
    'news_jp': [
        'nikkei.com', 'asahi.com', 'yomiuri.co.jp', 'mainichi.jp',
        'sankei.com', 'nhk.or.jp', 'toyokeizai.net', 'diamond.jp',
        'newspicks.com', 'itmedia.co.jp', 'impress.co.jp',
    ],
    # 主要メディア（海外）
    'news_intl': [
        'reuters.com', 'bloomberg.com', 'wsj.com', 'nytimes.com',
        'ft.com', 'economist.com', 'bbc.com', 'cnn.com',
        'techcrunch.com', 'theverge.com', 'wired.com',
    ],
    # 企業公式
    'corporate': [
        'prtimes.jp', 'pr-times.com',  # プレスリリース
    ],
    # 学術
    'academic': [
        'arxiv.org', 'doi.org', 'jstor.org', 'springer.com',
        'nature.com', 'science.org',
    ],
}

# Google検索は常にrejected
REJECTED_URL_PATTERNS = [
    r'google\.(com|co\.jp)/search',
    r'bing\.com/search',
    r'yahoo\.(com|co\.jp)/search',
    r'duckduckgo\.com',
]

# ============================================
# 3. trust_level判定基準
# ============================================
TRUST_LEVEL_CRITERIA = {
    'verified': {
        'description': '公式ソース + 著名事例 + 複数ソース',
        'rules': [
            'source_domain in official OR academic',
            'OR (source_domain in news AND source_count >= 2)',
            'AND NOT anonymous_pattern',
            'AND transition_id EXISTS',
        ]
    },
    'plausible': {
        'description': 'ニュースソース + 検証可能な固有名詞',
        'rules': [
            'source_domain in news_jp OR news_intl OR corporate',
            'AND NOT anonymous_pattern',
            'AND source_count >= 1',
        ]
    },
    'unverified': {
        'description': 'ソースあり但し検証困難',
        'rules': [
            'source EXISTS',
            'AND source_domain NOT in trusted',
            'AND NOT rejected_url_pattern',
        ]
    },
    'rejected': {
        'description': 'ソースなし / Google検索のみ / 匿名',
        'rules': [
            'source NOT EXISTS',
            'OR rejected_url_pattern',
            'OR anonymous_pattern',
        ]
    },
}

# ============================================
# 4. success_level計算設定
# ============================================
SUCCESS_LEVEL_CONFIG = {
    # ベイズ平滑化パラメータ（ラプラス補正）
    'smoothing_alpha': 1.0,  # 事前分布の強さ
    'prior_success_rate': 0.5,  # 事前確率（50%）

    # 最小サンプル数（これ以下は信頼区間を広げる）
    'min_sample_for_reliable': 10,

    # メタ情報フィールド
    'metadata_fields': [
        'sample_count',        # 該当組み合わせの総件数
        'confidence_interval', # 95%信頼区間
        'smoothed',           # 平滑化適用有無
    ],
}

# ============================================
# 5. 国コード正規化マッピング
# ============================================
COUNTRY_NORMALIZATION = {
    '日本': 'JP',
    'Japan': 'JP',
    'アメリカ': 'US',
    'USA': 'US',
    'United States': 'US',
    '中国': 'CN',
    'China': 'CN',
    'ヨーロッパ': 'EU',
    'Europe': 'EU',
    '韓国': 'KR',
    'Korea': 'KR',
    'イギリス': 'GB',
    'UK': 'GB',
    'ドイツ': 'DE',
    'Germany': 'DE',
    'フランス': 'FR',
    'France': 'FR',
}

# ============================================
# 6. main_domain推定キーワード（自動分類用）
# ============================================
DOMAIN_KEYWORDS = {
    'technology': ['IT', 'テック', 'AI', 'ソフトウェア', 'アプリ', 'プラットフォーム', 'SaaS'],
    'finance': ['銀行', '金融', '投資', 'ファンド', '証券', '保険', 'VC'],
    'retail': ['小売', '店舗', 'EC', 'eコマース', '百貨店', 'コンビニ'],
    'manufacturing': ['製造', '工場', '自動車', '電機', '機械', '素材'],
    'healthcare': ['医療', '製薬', 'バイオ', '病院', 'ヘルスケア'],
    'entertainment': ['エンタメ', '映画', '音楽', 'ゲーム', 'メディア'],
    'politics': ['政府', '政治', '政策', '選挙', '外交', '国際'],
}

# ============================================
# 7. 品質フラグ定義
# ============================================
QUALITY_FLAGS = {
    'single_source': 'ソースが1件のみ',
    'news_only': 'ニュースソースのみ（公式なし）',
    'old_source': 'ソースが5年以上前',
    'broken_link_risk': 'リンク切れリスクあり',
    'auto_classified': '自動分類による値',
    'smoothed_success': '平滑化適用済み',
    'low_sample': 'サンプル数10件未満',
}
