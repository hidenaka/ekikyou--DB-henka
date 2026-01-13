"""
COI (利益相反) 手動レジストリ v1
Codex批評に基づき、自動化は確実な範囲に限定し、手動レジストリを主とする

COI判定:
- self: 自社/グループ会社について報じている
- affiliated: 業務提携先/投資先について報じている
- sponsored: 広告/PR記事
- none: 利益相反なし
"""

import re
from typing import Optional, Tuple

# ============================================
# 1. ドメイン→所有企業マッピング（手動）
# ============================================
DOMAIN_OWNER_REGISTRY = {
    # 日本企業 - テック/製造
    'sony.com': 'Sony',
    'sony.co.jp': 'Sony',
    'sony.net': 'Sony',
    'toyota.com': 'Toyota',
    'toyota.co.jp': 'Toyota',
    'global.toyota': 'Toyota',
    'toshiba.co.jp': 'Toshiba',
    'global.toshiba': 'Toshiba',
    'panasonic.com': 'Panasonic',
    'panasonic.jp': 'Panasonic',
    'sharp.co.jp': 'Sharp',
    'jp.sharp': 'Sharp',
    'hitachi.com': 'Hitachi',
    'hitachi.co.jp': 'Hitachi',
    'honda.com': 'Honda',
    'honda.co.jp': 'Honda',
    'nissan.com': 'Nissan',
    'nissan-global.com': 'Nissan',
    'nintendo.com': 'Nintendo',
    'nintendo.co.jp': 'Nintendo',
    'canon.com': 'Canon',
    'canon.co.jp': 'Canon',
    'nikon.com': 'Nikon',
    'nikon.co.jp': 'Nikon',
    'fujifilm.com': 'Fujifilm',
    'kyocera.co.jp': 'Kyocera',
    'nec.com': 'NEC',
    'fujitsu.com': 'Fujitsu',
    'keyence.co.jp': 'Keyence',

    # 日本企業 - サービス/小売
    'softbank.co.jp': 'SoftBank',
    'softbank.jp': 'SoftBank',
    'group.softbank': 'SoftBank',
    'rakuten.co.jp': 'Rakuten',
    'linecorp.com': 'LINE',
    'fastretailing.com': 'FastRetailing',
    'uniqlo.com': 'FastRetailing',
    'aeon.co.jp': 'AEON',
    'aeon.info': 'AEON',
    '7andi.com': 'SevenAndI',
    'seven-i.com': 'SevenAndI',
    'mcdonalds.co.jp': 'McDonalds_Japan',
    'jal.com': 'JAL',
    'ana.co.jp': 'ANA',

    # 日本企業 - 商社/金融
    'mitsui.com': 'Mitsui',
    'mitsuipr.com': 'Mitsui',
    'sumitomocorp.com': 'Sumitomo',
    'marubeni.com': 'Marubeni',

    # 米国テック
    'apple.com': 'Apple',
    'google.com': 'Google',
    'about.google': 'Google',
    'alphabet.com': 'Google',
    'amazon.com': 'Amazon',
    'aboutamazon.com': 'Amazon',
    'amazon.co.jp': 'Amazon',
    'aboutamazon.jp': 'Amazon',
    'meta.com': 'Meta',
    'facebook.com': 'Meta',
    'instagram.com': 'Meta',
    'nvidia.com': 'NVIDIA',
    'tesla.com': 'Tesla',
    'netflix.com': 'Netflix',
    'openai.com': 'OpenAI',
    'microsoft.com': 'Microsoft',
    'spacex.com': 'SpaceX',
    'airbnb.com': 'Airbnb',
    'zoom.us': 'Zoom',
    'uber.com': 'Uber',

    # 米国その他
    'gm.com': 'GM',
    'ford.com': 'Ford',
    'disney.com': 'Disney',
    'walmart.com': 'Walmart',
    'boeing.com': 'Boeing',
    'mcdonalds.com': 'McDonalds',
    'berkshirehathaway.com': 'BerkshireHathaway',
    'jpmorgan.com': 'JPMorgan',
    'goldmansachs.com': 'GoldmanSachs',
    'blackrock.com': 'BlackRock',

    # 中国/韓国/台湾
    'samsung.com': 'Samsung',
    'tsmc.com': 'TSMC',
    'alibaba.com': 'Alibaba',
    'tencent.com': 'Tencent',
    'huawei.com': 'Huawei',
    'tiktok.com': 'ByteDance',
    'bytedance.com': 'ByteDance',

    # 欧州
    'siemens.com': 'Siemens',
    'volkswagen.de': 'Volkswagen',
    'vw.com': 'Volkswagen',
    'lvmh.com': 'LVMH',
    'nestle.com': 'Nestle',
    'unilever.com': 'Unilever',
}

# ============================================
# 2. 企業グループマッピング（子会社・関連会社）
# ============================================
CORPORATE_GROUPS = {
    'SoftBank': ['Yahoo_Japan', 'LINE', 'PayPay', 'SoftBankVision'],
    'Google': ['YouTube', 'DeepMind', 'Waymo', 'Fitbit'],
    'Meta': ['WhatsApp', 'Instagram', 'Oculus'],
    'Amazon': ['AWS', 'Twitch', 'Whole_Foods', 'MGM'],
    'Microsoft': ['LinkedIn', 'GitHub', 'Activision_Blizzard', 'OpenAI_Partner'],
    'SevenAndI': ['SevenEleven', 'Ito_Yokado', 'Sogo_Seibu'],
    'FastRetailing': ['UNIQLO', 'GU', 'Theory'],
    'Rakuten': ['Rakuten_Mobile', 'Rakuten_Bank', 'Rakuten_Travel'],
    'Sony': ['PlayStation', 'Sony_Music', 'Sony_Pictures', 'Crunchyroll'],
    'Toyota': ['Lexus', 'Daihatsu', 'Hino'],
    'LVMH': ['Louis_Vuitton', 'Dior', 'Tiffany', 'Moet_Hennessy'],
}

# ============================================
# 3. target_name正規化（企業名の揺らぎ対応）
# ============================================
TARGET_NAME_NORMALIZATION = {
    # 日本企業
    r'ソニー|sony': 'Sony',
    r'トヨタ|toyota|トヨタ自動車': 'Toyota',
    r'東芝|toshiba': 'Toshiba',
    r'パナソニック|panasonic|松下電器': 'Panasonic',
    r'シャープ|sharp': 'Sharp',
    r'日立|hitachi|日立製作所': 'Hitachi',
    r'ホンダ|honda|本田技研': 'Honda',
    r'日産|nissan': 'Nissan',
    r'任天堂|nintendo': 'Nintendo',
    r'キヤノン|canon': 'Canon',
    r'ニコン|nikon': 'Nikon',
    r'ソフトバンク|softbank': 'SoftBank',
    r'楽天|rakuten': 'Rakuten',
    r'ユニクロ|uniqlo|ファーストリテイリング': 'FastRetailing',
    r'イオン|aeon': 'AEON',
    r'セブン.*イレブン|7-eleven|セブンアンドアイ': 'SevenAndI',
    r'三井物産': 'Mitsui',
    r'住友商事': 'Sumitomo',
    r'丸紅': 'Marubeni',
    r'JAL|日本航空': 'JAL',
    r'ANA|全日空|全日本空輸': 'ANA',

    # 米国テック
    r'アップル|apple': 'Apple',
    r'グーグル|google|アルファベット|alphabet': 'Google',
    r'アマゾン|amazon': 'Amazon',
    r'メタ|meta|フェイスブック|facebook': 'Meta',
    r'エヌビディア|nvidia': 'NVIDIA',
    r'テスラ|tesla': 'Tesla',
    r'ネットフリックス|netflix': 'Netflix',
    r'マイクロソフト|microsoft': 'Microsoft',
    r'スペースX|spacex': 'SpaceX',

    # その他
    r'サムスン|samsung': 'Samsung',
    r'TSMC|台湾積体電路': 'TSMC',
    r'アリババ|alibaba': 'Alibaba',
    r'テンセント|tencent|騰訊': 'Tencent',
    r'ファーウェイ|huawei|華為': 'Huawei',
    r'ディズニー|disney': 'Disney',
    r'ウォルマート|walmart': 'Walmart',
}

# ============================================
# 4. COI判定関数
# ============================================

def normalize_target_name(name: str) -> Optional[str]:
    """target_nameを正規化して企業IDに変換"""
    if not name:
        return None

    name_lower = name.lower()
    for pattern, normalized in TARGET_NAME_NORMALIZATION.items():
        if re.search(pattern, name_lower, re.IGNORECASE):
            return normalized

    return None

def get_domain_owner(domain: str) -> Optional[str]:
    """ドメインから所有企業を取得"""
    if not domain:
        return None

    domain = domain.lower().strip()

    # 直接マッチ
    if domain in DOMAIN_OWNER_REGISTRY:
        return DOMAIN_OWNER_REGISTRY[domain]

    # サブドメイン考慮（ir.sony.com → sony.com）
    parts = domain.split('.')
    for i in range(len(parts)):
        candidate = '.'.join(parts[i:])
        if candidate in DOMAIN_OWNER_REGISTRY:
            return DOMAIN_OWNER_REGISTRY[candidate]

    return None

def is_same_group(company1: str, company2: str) -> bool:
    """同一企業グループか判定"""
    if not company1 or not company2:
        return False

    if company1 == company2:
        return True

    # グループ内子会社チェック
    for parent, subsidiaries in CORPORATE_GROUPS.items():
        members = [parent] + subsidiaries
        if company1 in members and company2 in members:
            return True

    return False

def determine_coi(source_domain: str, target_name: str) -> Tuple[str, Optional[str]]:
    """
    COI判定（手動レジストリベース）

    Returns:
        Tuple[coi_type, reason]
        coi_type: 'self' | 'affiliated' | 'none' | 'unknown'
    """
    source_owner = get_domain_owner(source_domain)
    target_normalized = normalize_target_name(target_name)

    # ソース所有者が不明
    if not source_owner:
        return ('unknown', 'source_owner_not_in_registry')

    # ターゲット企業が不明
    if not target_normalized:
        return ('unknown', 'target_not_normalized')

    # 同一企業/グループ
    if is_same_group(source_owner, target_normalized):
        return ('self', f'{source_owner} reporting on {target_normalized}')

    # TODO: affiliated判定（業務提携データが必要）
    # 現時点では手動レジストリに提携関係がないため、判定不能

    return ('none', None)

# ============================================
# 5. 確実な自動検出（補助）
# ============================================

def detect_sponsored_content(url: str, content_hint: str = '') -> bool:
    """スポンサードコンテンツの自動検出（確実なパターンのみ）"""
    patterns = [
        r'/sponsored/',
        r'/advertorial/',
        r'/pr-release/',
        r'/ad/',
        r'[?&]sponsored=',
        r'prtimes\.jp',
        r'businesswire\.com',
        r'prnewswire\.com',
    ]

    combined = url.lower() + ' ' + content_hint.lower()
    for pattern in patterns:
        if re.search(pattern, combined):
            return True

    return False

def detect_corporate_blog(url: str) -> bool:
    """企業ブログの自動検出"""
    patterns = [
        r'/blog/',
        r'/blogs/',
        r'blog\.',
        r'developer\.',
        r'/developers/',
    ]

    for pattern in patterns:
        if re.search(pattern, url.lower()):
            return True

    return False

# ============================================
# 6. 統合COI評価
# ============================================

def evaluate_coi(source_url: str, target_name: str) -> dict:
    """
    COI総合評価

    Returns:
        {
            'coi_type': 'self' | 'affiliated' | 'sponsored' | 'none' | 'unknown',
            'reason': str,
            'confidence': 'high' | 'medium' | 'low',
            'source_owner': str or None,
            'target_normalized': str or None,
        }
    """
    from domain_rules import extract_domain

    domain = extract_domain(source_url)
    source_owner = get_domain_owner(domain)
    target_normalized = normalize_target_name(target_name)

    result = {
        'source_owner': source_owner,
        'target_normalized': target_normalized,
        'confidence': 'low',
    }

    # 1. スポンサードコンテンツ（確実）
    if detect_sponsored_content(source_url):
        result['coi_type'] = 'sponsored'
        result['reason'] = 'sponsored_content_detected'
        result['confidence'] = 'high'
        return result

    # 2. 手動レジストリによるCOI判定
    coi_type, reason = determine_coi(domain, target_name)
    result['coi_type'] = coi_type
    result['reason'] = reason

    if coi_type == 'self':
        result['confidence'] = 'high'
    elif coi_type == 'none' and source_owner and target_normalized:
        result['confidence'] = 'medium'

    # 3. 企業ブログは信頼度を下げる
    if detect_corporate_blog(source_url):
        result['quality_flag'] = 'corporate_blog'
        if result['confidence'] == 'high':
            result['confidence'] = 'medium'

    return result
