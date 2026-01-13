"""
ドメイン判定ルール v3
正規化・サブドメイン・リダイレクト対応
"""

import re
from urllib.parse import urlparse

# ============================================
# 1. URL正規化ルール
# ============================================
NORMALIZATION_RULES = {
    'remove_www': True,           # www. を除去
    'remove_protocol': True,      # http(s):// を除去して比較
    'lowercase': True,            # 小文字化
    'remove_trailing_slash': True,
    'remove_query_params': False, # クエリパラメータは保持（識別用）
}

# ============================================
# 2. サブドメイン許可ルール
# ============================================
SUBDOMAIN_POLICY = {
    # 完全一致が必要（サブドメイン不可）
    'exact_match_only': [],

    # 任意のサブドメインを許可（*.example.com）
    'allow_any_subdomain': [
        # 政府系
        'go.jp', 'go.kr', 'gov', 'gov.uk', 'gouv.fr',
        'bundesregierung.de', 'gouvernement.fr',
        # 学術系
        'ac.jp', 'edu', 'ac.uk',
        # 企業系（global.*, about.*, ir.* など多様）
        'sony.com', 'toyota', 'toshiba', 'softbank',
        'panasonic.com', 'mitsubishi.com', 'sharp',
        'apple.com', 'google', 'amazon', 'meta.com',
        'nvidia.com', 'tesla.com', 'netflix.com',
        'openai.com', 'microsoft.com',
        'fastretailing.com', 'uniqlo.com',
    ],

    # 特定サブドメインのみ許可
    'allowed_subdomains': {
        # 例: ir.company.com, news.company.com のみ許可
        'example.com': ['ir', 'news', 'press', 'about'],
    },
}

# ============================================
# 3. リダイレクト正規化マッピング
# ============================================
REDIRECT_NORMALIZATION = {
    # about.google → google.com として扱う
    'about.google': 'google.com',
    'aboutamazon.com': 'amazon.com',
    'aboutamazon.jp': 'amazon.co.jp',
    'global.toyota': 'toyota.co.jp',
    'global.toshiba': 'toshiba.co.jp',
    'group.softbank': 'softbank.co.jp',
    'jp.sharp': 'sharp.co.jp',
    'mitsuipr.com': 'mitsui.com',
}

# ============================================
# 4. Tier定義（拡張版）
# ============================================
DOMAIN_TIERS = {
    # Tier 1: 公式・政府・学術・国際機関（最高信頼）
    'tier1_official': {
        'patterns': [
            # 日本政府・学術
            r'\.go\.jp$', r'\.ac\.jp$',
            # 米国
            r'\.gov$', r'\.edu$',
            # 欧州（サブドメインありとトップレベルドメイン直接の両方に対応）
            r'gov\.uk$', r'\.ac\.uk$', r'\.gouv\.fr$',
            r'bundesregierung\.de', r'gouvernement\.fr',
            r'ec\.europa\.eu', r'europa\.eu$',
            # アジア
            r'\.go\.kr$', r'gov\.cn$', r'india\.gov\.in$',
            r'indonesia\.go\.id$',
            # その他
            r'gov\.br$', r'government\.ru$',
            # 国際機関
            r'imf\.org$', r'worldbank\.org$', r'un\.org$',
            r'oecd\.org$', r'who\.int$', r'wto\.org$',
            # 中央銀行
            r'boj\.or\.jp$', r'federalreserve\.gov$',
            # 業界団体・協会
            r'riaj\.or\.jp$', r'eiren\.org$', r'joc\.or\.jp$',
            r'jata-net\.or\.jp$', r'cesa\.or\.jp$',
            r'whc\.unesco\.org$',
            # 海外規制当局
            r'sec\.gov$', r'fca\.org\.uk$', r'ecb\.europa\.eu$',
        ],
        'description': '政府・学術・国際機関',
        'source_type_override': 'official',
    },

    # Tier 2: 主要メディア（高信頼）
    'tier2_major_media': {
        'patterns': [
            # 日本
            r'nikkei\.com$', r'asahi\.com$', r'yomiuri\.co\.jp$',
            r'mainichi\.jp$', r'sankei\.com$', r'nhk\.or\.jp$',
            r'jiji\.com$', r'kyodo\.co\.jp$',
            # 海外
            r'reuters\.com$', r'bloomberg\.com$', r'wsj\.com$',
            r'nytimes\.com$', r'ft\.com$', r'economist\.com$',
            r'bbc\.(com|co\.uk)$', r'cnn\.com$', r'apnews\.com$',
        ],
        'description': '主要報道機関',
        'source_type_override': 'news',
    },

    # Tier 3: ビジネス・テック専門メディア + 参考情報源
    'tier3_specialist': {
        'patterns': [
            # ビジネス誌
            r'toyokeizai\.net$', r'diamond\.jp$', r'newspicks\.com$',
            r'president\.jp$', r'businessinsider\.jp$', r'forbesjapan\.com$',
            # テックメディア
            r'techcrunch\.com$', r'theverge\.com$', r'wired\.(com|jp)$',
            r'itmedia\.co\.jp$', r'impress\.co\.jp$', r'cnet\.com$',
            r'zdnet\.com$', r'arstechnica\.com$',
            # 百科事典（二次情報源、検証用参考）
            r'wikipedia\.org$', r'britannica\.com$',
        ],
        'description': '専門メディア・参考情報源',
        'source_type_override': 'news',
    },

    # Tier 4: 企業公式（IR/PR/About限定）
    'tier4_corporate': {
        'patterns': [
            # 日本企業（製造業）
            r'sony\.(com|co\.jp)', r'toyota', r'toshiba',
            r'panasonic\.com', r'mitsubishi\.com', r'sharp',
            r'hitachi\.(com|co\.jp)', r'honda', r'nissan',
            r'nintendo\.(com|co\.jp)', r'canon\.(com|co\.jp)',
            r'nikon\.(com|co\.jp)', r'fujifilm', r'boeing\.com',
            # 日本企業（サービス・小売・金融）
            r'softbank', r'rakuten\.co\.jp', r'linecorp\.com',
            r'fastretailing\.com', r'uniqlo\.com',
            r'aeon\.(info|co\.jp)', r'7andi\.com', r'seven-i\.com',
            r'mcdonalds\.(com|co\.jp)', r'mitsui\.com',
            r'sumitomo', r'marubeni\.com',
            # 海外テック
            r'apple\.com', r'google', r'amazon', r'meta\.com',
            r'nvidia\.com', r'tesla\.com', r'netflix\.com',
            r'openai\.com', r'microsoft\.com', r'spacex\.com',
            r'airbnb\.com', r'zoom\.us', r'uber\.com',
            r'tiktok\.com', r'bytedance\.com',
            # 海外その他
            r'gm\.com', r'ford\.com', r'disney', r'walmart\.com',
            # スポーツ・エンタメ
            r'mlb\.com', r'nba\.com', r'nfl\.com', r'jfa\.jp',
            r'npb\.jp$',
            # アジア大手
            r'samsung\.com', r'tsmc\.com', r'alibaba\.com',
            r'tencent\.com', r'huawei\.com',
            # 欧州大手
            r'siemens\.com', r'volkswagen', r'lvmh\.com',
            r'nestle\.com', r'unilever\.com',
            # 日本追加
            r'kyocera\.co\.jp', r'nec\.com', r'fujitsu\.com',
            r'keyence\.co\.jp', r'jal\.com', r'ana\.co\.jp',
            # 米国追加
            r'berkshirehathaway\.com', r'jpmorgan\.com',
            r'goldmansachs\.com', r'blackrock\.com',
        ],
        'description': '企業公式サイト',
        'source_type_override': 'official',
        # 企業公式の場合、パスに以下が含まれることを推奨（強制ではない）
        'recommended_paths': ['/ir/', '/news/', '/press/', '/about/', '/corporate/'],
    },

    # Tier 5: プレスリリース配信
    'tier5_pr': {
        'patterns': [
            r'prtimes\.jp$', r'businesswire\.com$',
            r'prnewswire\.com$', r'globenewswire\.com$',
        ],
        'description': 'プレスリリース配信',
        'source_type_override': 'official',
    },
}

# ============================================
# 5. rejected パターン（明文化）
# ============================================
REJECTED_PATTERNS = [
    r'google\.(com|co\.jp|[a-z]{2})/search',
    r'bing\.com/search',
    r'yahoo\.(com|co\.jp)/search',
    r'duckduckgo\.com/\?q=',
    r'^search\.',  # search.yahoo.com等
    r'webcache\.googleusercontent\.com',  # Googleキャッシュ
]

# ============================================
# 6. 判定関数
# ============================================

def normalize_url(url: str) -> str:
    """URL正規化"""
    if not url:
        return ''

    url = url.strip()

    # 小文字化
    if NORMALIZATION_RULES['lowercase']:
        url = url.lower()

    # プロトコル除去（内部比較用）
    url = re.sub(r'^https?://', '', url)

    # www除去
    if NORMALIZATION_RULES['remove_www']:
        url = re.sub(r'^www\.', '', url)

    # 末尾スラッシュ除去
    if NORMALIZATION_RULES['remove_trailing_slash']:
        url = url.rstrip('/')

    return url

def extract_domain(url: str) -> str:
    """ドメイン抽出（正規化済み）"""
    normalized = normalize_url(url)
    # パス除去
    domain = normalized.split('/')[0]
    # ポート除去
    domain = domain.split(':')[0]

    # リダイレクト正規化
    for redirect_from, redirect_to in REDIRECT_NORMALIZATION.items():
        if redirect_from in domain:
            domain = redirect_to
            break

    return domain

def is_rejected(url: str) -> bool:
    """rejected URL判定"""
    for pattern in REJECTED_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return True
    return False

def classify_domain(url: str) -> dict:
    """ドメインをTier分類"""
    if is_rejected(url):
        return {'tier': 'rejected', 'reason': 'rejected_pattern'}

    domain = extract_domain(url)
    if not domain:
        return {'tier': 'unknown', 'reason': 'empty_domain'}

    for tier_name, tier_config in DOMAIN_TIERS.items():
        for pattern in tier_config['patterns']:
            if re.search(pattern, domain, re.IGNORECASE):
                return {
                    'tier': tier_name,
                    'domain': domain,
                    'source_type_override': tier_config.get('source_type_override'),
                }

    return {'tier': 'unclassified', 'domain': domain}

# ============================================
# 7. Gold/Silver判定用Tier集合
# ============================================
GOLD_TIERS = ['tier1_official', 'tier2_major_media', 'tier4_corporate']
SILVER_TIERS = GOLD_TIERS + ['tier3_specialist', 'tier5_pr']
