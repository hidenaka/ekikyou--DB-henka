#!/usr/bin/env python3
"""
generate_yao_action_rules.py

384爻の行動ルール（do / do_not / condition / strength）を
data/reference/iching_texts_ctext_legge_ja.json の modern_ja テキストから
ルールベースで抽出し、data/diagnostic/yao_action_rules.json を生成する。

追加ソース:
  - data/diagnostic/yao_384.json  (advice / warning)
  - data/diagnostic/hexagram_64.json  (archetype / modern_interpretation)
"""

import json
import re
import os
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ICHING_TEXTS = os.path.join(BASE_DIR, "data", "reference", "iching_texts_ctext_legge_ja.json")
YAO_384 = os.path.join(BASE_DIR, "data", "diagnostic", "yao_384.json")
HEXAGRAM_64 = os.path.join(BASE_DIR, "data", "diagnostic", "hexagram_64.json")
OUTPUT = os.path.join(BASE_DIR, "data", "diagnostic", "yao_action_rules.json")

# ---------------------------------------------------------------------------
# Position defaults (used when text-based extraction fails)
# ---------------------------------------------------------------------------
POSITION_DEFAULTS = {
    1: {
        "do_not": "拙速に動くこと",
        "do": "準備と観察に専念する",
        "condition": "始まりの段階。まだ動くべき時ではない",
    },
    2: {
        "do_not": "独断で進めること",
        "do": "信頼を築き、基盤を固める",
        "condition": "発展初期。基盤を固め、協力関係を築く段階",
    },
    3: {
        "do_not": "油断すること",
        "do": "慎重に判断し、決断する",
        "condition": "転換点。内から外へ出る境目で、危険も多い",
    },
    4: {
        "do_not": "過去のやり方に固執すること",
        "do": "新しい環境に適応する",
        "condition": "新段階の入口。上の世界に入ったが、まだ不安定",
    },
    5: {
        "do_not": "傲慢になること",
        "do": "中庸を保ち、影響力を適切に使う",
        "condition": "中心的立場。力を発揮できる最良の位置",
    },
    6: {
        "do_not": "しがみつくこと",
        "do": "手放し、次の段階に備える",
        "condition": "終わりの段階。やりすぎに注意し、次に備える",
    },
}

# ---------------------------------------------------------------------------
# Keyword pattern tables for extraction
# ---------------------------------------------------------------------------

# do_not extraction patterns: (regex, extracted do_not text)
# Ordered by specificity — first match wins
DO_NOT_PATTERNS = [
    # Explicit prohibition patterns
    (r"焦[らりるれっ]", "焦ること"),
    (r"急[いぎぐげご]", "急いで行動すること"),
    (r"無理に(?:自分を)?押し通[すそさし]", "無理に自分を押し通すこと"),
    (r"無理に(?:動|追|進)", "無理に動くこと"),
    (r"やりすぎ", "やりすぎること"),
    (r"頑張りすぎ", "頑張りすぎること"),
    (r"油断は?禁物", "油断すること"),
    (r"油断", "油断すること"),
    (r"過信", "自分の能力を過信すること"),
    (r"傲慢", "傲慢になること"),
    (r"独断", "独断で進めること"),
    (r"動く時ではありません", "焦って動くこと"),
    (r"動かず", "焦って動くこと"),
    (r"まだ表に出る(?:段階|時期)ではありません", "焦って表に出ること"),
    (r"前面に出す(?:より|のでは)", "自分を前面に出すこと"),
    (r"控えめに", "自分を前面に出すこと"),
    (r"主張しなくても", "成果を声高に主張すること"),
    (r"凶", "現状に固執すること"),
    (r"危険", "軽率に行動すること"),
    (r"厲|危う", "軽率に行動すること"),
    (r"衝突", "対立を招くような行動"),
    (r"後悔を避け", "極端に走ること"),
    (r"引き返す勇気", "無理に追いかけること"),
    (r"迷って", "方向を見失うこと"),
    (r"慎重さ(?:も|を|は)(?:必要|忘れず)", "慎重さを失うこと"),
    (r"正しい(?:もの|こと)が損なわれ", "基盤を軽視すること"),
    (r"崩れ", "基盤を軽視すること"),
    (r"進みづらさ", "焦って進もうとすること"),
    (r"実力以上", "実力以上のことに手を出すこと"),
    (r"小人は用いるべきではありません", "信頼できない人に任せること"),
    (r"待つ(?:時期|段階|べき)", "焦って動くこと"),
    (r"受け入れ(?:ながら|て)", "現状に抗うこと"),
    (r"謙虚さ", "傲慢になること"),
]

# do extraction patterns: (regex, extracted do text)
DO_PATTERNS = [
    # Explicit recommendation patterns
    (r"力を蓄え", "力を蓄え、静かに準備する"),
    (r"準備を(?:整え|してお)", "準備を整える"),
    (r"土台を固め", "土台をしっかり固める"),
    (r"基盤を固め", "基盤を固める"),
    (r"基礎から", "基礎をしっかり固める"),
    (r"信頼できる人", "信頼できる人との関係を大切にする"),
    (r"出会い(?:が|も)", "良い出会いを活かす"),
    (r"全力で取り組[むみ]", "全力で取り組みながらも振り返りを忘れない"),
    (r"振り返り", "これまでの歩みを振り返る"),
    (r"自信を持って進[むめん]", "自信を持って前に進む"),
    (r"今こそ行動", "今こそ行動を起こす"),
    (r"行動を起こす", "行動を起こして前に進む"),
    (r"コツコツ", "小さなことからコツコツ始める"),
    (r"謙虚さに謙虚さ", "徹底的に謙虚であること"),
    (r"謙虚(?:さ|に)", "謙虚さを保つ"),
    (r"堅実(?:さ|に)", "堅実に進める"),
    (r"慎重に", "慎重に進める"),
    (r"控えめ(?:に|で)", "控えめに力を蓄える"),
    (r"受け入れ(?:ながら|て|る)", "現状を受け入れて次を考える"),
    (r"待つ(?:時期|段階|べき)", "時機を待つ"),
    (r"落ち着くのを待つ", "状況が落ち着くのを待つ"),
    (r"周囲の協力", "周囲の協力を得る"),
    (r"影響力(?:のある|が)", "影響力を適切に活かす"),
    (r"存在感", "確かな存在感を示す"),
    (r"飛躍のチャンス", "飛躍のチャンスに備える"),
    (r"タイミング(?:を計|を見極)", "タイミングを慎重に計る"),
    (r"決断力", "決断力を持って進む"),
    (r"様子を見", "様子を見ながらタイミングを計る"),
    (r"乗り越え", "困難を乗り越える"),
    (r"正しく進め", "正しい道を歩む"),
    (r"努力を続け", "努力を継続する"),
    (r"一歩ずつ進む", "着実に一歩ずつ進む"),
    (r"柔軟に対応", "柔軟に対応する"),
    (r"次の準備", "次の段階への準備をする"),
    (r"恩恵を与え", "周囲に恩恵を与える"),
    (r"祭祀", "誠意をもって行動する"),
    (r"称賛", "前に進んで称賛を得る"),
    (r"簡素な", "質素で誠実な姿勢を保つ"),
    (r"功績(?:が|を)", "功績を上げつつも謙虚さを保つ"),
    (r"問題を解決", "身近な問題から解決する"),
    (r"自分を高め", "自分を高め続ける"),
    (r"徳を養", "徳を養う"),
]

# condition extraction patterns
CONDITION_PATTERNS = [
    (r"最も力を発揮できる時期", "最も力を発揮できる好機"),
    (r"飛躍のチャンス(?:が近づ)", "飛躍のチャンスが近づいている"),
    (r"忙しい時期", "忙しく緊張感の高い時期"),
    (r"前に進みづらさ", "前に進みづらい困難な時期"),
    (r"辛い状況", "進退窮まった辛い状況"),
    (r"頂点に達し", "頂点に達した後の下降局面"),
    (r"表舞台に出始める", "徐々に表舞台に出始める時期"),
    (r"小さな変化の兆し", "変化の兆しが見え始めている"),
    (r"控えめ(?:に|で)", "自分を控えめに保つべき時期"),
    (r"大きな幸運", "大きな幸運が期待できる時期"),
    (r"衝突(?:が起こ)", "衝突が起こりやすい時期"),
    (r"凶", "困難・障害の多い時期"),
    (r"吉", "好転の兆しがある時期"),
    (r"危険", "危険を伴う時期"),
    (r"問題な(?:く|い)", "問題なく進められる時期"),
    (r"大きな可能性", "大きな可能性が開かれている時期"),
    (r"力(?:が|を)発揮", "力を発揮できる時期"),
    (r"困難", "困難に直面している時期"),
    (r"待つ(?:時期|段階)", "待つべき時期"),
    (r"準備", "準備の段階"),
    (r"転機", "転機を迎えている"),
    (r"崩れ", "基盤が崩れ始めている"),
    (r"恵み", "恵みを分かち合う時期"),
]

# strength extraction patterns (from classic text)
STRONG_POSITIVE = re.compile(r"大吉|元亨|元吉|大いに亨る")
STRONG_NEGATIVE = re.compile(r"大凶|凶(?!。)|凶です")
MODERATE_POSITIVE = re.compile(r"(?<!大)吉|小吉|利[するある]|利貞|无咎|問題(?:なく|ない|ありません)")
MODERATE_NEGATIVE = re.compile(r"咎|悔|厲|吝|危[うい険]")


def determine_strength(classic_text: str, modern_ja: str) -> str:
    """Determine strength based on classical 吉凶 markers."""
    combined = classic_text + modern_ja

    # Check strong markers first
    if STRONG_POSITIVE.search(combined):
        return "strong"
    if STRONG_NEGATIVE.search(combined):
        return "strong"

    # Check moderate markers
    if MODERATE_POSITIVE.search(combined):
        return "moderate"
    if MODERATE_NEGATIVE.search(combined):
        return "moderate"

    return "neutral"


def extract_do_not(modern_ja: str, yao_extra: dict | None, position: int) -> tuple[str, bool]:
    """
    Extract the do_not (prohibition) from modern_ja text.
    Priority: modern_ja specific patterns > yao_384 warning > position default.
    Returns (do_not_text, was_extracted_from_text).
    """
    # 1. Try modern_ja patterns first (more specific per hexagram)
    for pattern, result in DO_NOT_PATTERNS:
        if re.search(pattern, modern_ja):
            return (result, True)

    # 2. Try yao_384 warning as secondary source
    if yao_extra and yao_extra.get("warning"):
        warning = yao_extra["warning"]
        # Direct mapping for known warning patterns
        WARNING_TO_DO_NOT = {
            "軽はずみに動くと失敗します": "軽はずみに動くこと",
            "傲慢になると足元をすくわれます": "傲慢になること",
            "出過ぎると反発を受けます": "出過ぎること",
            "執着すると衰退を招きます": "執着すること",
            "焦って表に出ようとしないこと": "焦って表に出ること",
            "この段階での失敗は痛手になります": "この段階で油断すること",
        }
        if warning in WARNING_TO_DO_NOT:
            return (WARNING_TO_DO_NOT[warning], True)
        # Fallback for unknown warnings: extract cause before "と"
        m = re.match(r"(.+?)と(?:失敗|衰退|問題|足元|痛手|反発|孤立|混乱)", warning)
        if m:
            return (m.group(1) + "こと", True)
        # Last resort: strip suffixes
        cleaned = re.sub(r"(?:です|ます|ません)$", "", warning)
        if cleaned and not cleaned.endswith("こと"):
            cleaned = cleaned + "こと"
        if cleaned and len(cleaned) > 3:
            return (cleaned, True)

    # 3. Fall back to position default
    return (POSITION_DEFAULTS[position]["do_not"], False)


def extract_do(modern_ja: str, yao_extra: dict | None, position: int) -> tuple[str, bool]:
    """
    Extract the do (recommendation) from modern_ja text.
    Returns (do_text, was_extracted_from_text).
    """
    # Try yao_384 advice first
    if yao_extra and yao_extra.get("advice"):
        advice = yao_extra["advice"]
        # Clean up and use as do
        cleaned = re.sub(r"(?:ましょう|ください|です|ます)(?:[。])?$", "", advice)
        if cleaned and len(cleaned) > 3:
            return (advice, True)

    # Try modern_ja patterns
    for pattern, result in DO_PATTERNS:
        if re.search(pattern, modern_ja):
            return (result, True)

    # Fall back to position default
    return (POSITION_DEFAULTS[position]["do"], False)


def extract_condition(modern_ja: str, yao_extra: dict | None, position: int) -> tuple[str, bool]:
    """
    Extract the condition (situational context) from modern_ja text.
    Returns (condition_text, was_extracted_from_text).
    """
    # Try yao_384 situation first
    if yao_extra and yao_extra.get("situation"):
        return (yao_extra["situation"], True)

    # Try modern_ja patterns
    for pattern, result in CONDITION_PATTERNS:
        if re.search(pattern, modern_ja):
            return (result, True)

    # Fall back to position default
    return (POSITION_DEFAULTS[position]["condition"], False)


def main():
    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    with open(ICHING_TEXTS, "r", encoding="utf-8") as f:
        iching_data = json.load(f)

    yao_384_data = {}
    if os.path.exists(YAO_384):
        with open(YAO_384, "r", encoding="utf-8") as f:
            yao_384_raw = json.load(f)
        yao_384_data = yao_384_raw.get("yao", {})

    hexagram_64_data = {}
    if os.path.exists(HEXAGRAM_64):
        with open(HEXAGRAM_64, "r", encoding="utf-8") as f:
            hex64_raw = json.load(f)
        hexagram_64_data = hex64_raw.get("hexagrams", {})

    # Build a number-to-name lookup from hexagram_64
    num_to_name = {}
    for name, info in hexagram_64_data.items():
        num_to_name[info["number"]] = name

    # -----------------------------------------------------------------------
    # Generate rules
    # -----------------------------------------------------------------------
    rules = {}
    stats = {
        "do_not_extracted": 0,
        "do_not_default": 0,
        "do_extracted": 0,
        "do_default": 0,
        "condition_extracted": 0,
        "condition_default": 0,
        "strength_strong": 0,
        "strength_moderate": 0,
        "strength_neutral": 0,
    }

    for hex_num_str in sorted(iching_data["hexagrams"].keys(), key=int):
        hex_num = int(hex_num_str)
        hex_data = iching_data["hexagrams"][hex_num_str]
        hex_name = hex_data.get("local_name", f"卦{hex_num}")

        for line_num in range(1, 7):
            line_str = str(line_num)
            line_data = hex_data.get("lines", {}).get(line_str, {})
            modern_ja = line_data.get("modern_ja", "")
            classic = line_data.get("classic", "")

            # Look up yao_384 extra data
            yao_key_padded = f"{hex_num:02d}-{line_num}"
            yao_extra = yao_384_data.get(yao_key_padded)

            # Extract fields
            do_not, do_not_extracted = extract_do_not(modern_ja, yao_extra, line_num)
            do, do_extracted = extract_do(modern_ja, yao_extra, line_num)
            condition, cond_extracted = extract_condition(modern_ja, yao_extra, line_num)
            strength = determine_strength(classic, modern_ja)

            # Update stats
            if do_not_extracted:
                stats["do_not_extracted"] += 1
            else:
                stats["do_not_default"] += 1

            if do_extracted:
                stats["do_extracted"] += 1
            else:
                stats["do_default"] += 1

            if cond_extracted:
                stats["condition_extracted"] += 1
            else:
                stats["condition_default"] += 1

            stats[f"strength_{strength}"] += 1

            # Build rule key: "{hex_num}_{line_num}"
            rule_key = f"{hex_num}_{line_num}"
            rules[rule_key] = {
                "hexagram_number": hex_num,
                "yao_position": line_num,
                "hexagram_name": hex_name,
                "do_not": do_not,
                "do": do,
                "condition": condition,
                "strength": strength,
                "source_text": modern_ja,
            }

    # -----------------------------------------------------------------------
    # Build output
    # -----------------------------------------------------------------------
    output = {
        "version": "1.0",
        "description": "384爻の行動ルールテーブル。5点固定出力の「今やるな」「今やれ」の根拠。",
        "rules": rules,
        "metadata": {
            "total_rules": len(rules),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generation_method": "pattern_extraction_from_modern_ja",
            "sources": [
                "data/reference/iching_texts_ctext_legge_ja.json",
                "data/diagnostic/yao_384.json",
                "data/diagnostic/hexagram_64.json",
            ],
            "statistics": stats,
        },
    }

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # -----------------------------------------------------------------------
    # Print statistics
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("  yao_action_rules.json 生成完了")
    print("=" * 60)
    print(f"  総ルール数: {len(rules)}")
    print()
    print("  --- do_not (禁止行動) ---")
    print(f"    テキストから抽出: {stats['do_not_extracted']}")
    print(f"    デフォルト使用:   {stats['do_not_default']}")
    print(f"    抽出率:           {stats['do_not_extracted']/len(rules)*100:.1f}%")
    print()
    print("  --- do (推奨行動) ---")
    print(f"    テキストから抽出: {stats['do_extracted']}")
    print(f"    デフォルト使用:   {stats['do_default']}")
    print(f"    抽出率:           {stats['do_extracted']/len(rules)*100:.1f}%")
    print()
    print("  --- condition (状況) ---")
    print(f"    テキストから抽出: {stats['condition_extracted']}")
    print(f"    デフォルト使用:   {stats['condition_default']}")
    print(f"    抽出率:           {stats['condition_extracted']/len(rules)*100:.1f}%")
    print()
    print("  --- strength (強度) ---")
    print(f"    strong:   {stats['strength_strong']}")
    print(f"    moderate: {stats['strength_moderate']}")
    print(f"    neutral:  {stats['strength_neutral']}")
    print()
    print(f"  出力ファイル: {OUTPUT}")
    print("=" * 60)

    # Verify exactly 384
    assert len(rules) == 384, f"Expected 384 rules, got {len(rules)}"
    print("\n  [OK] 384エントリの検証完了")


if __name__ == "__main__":
    main()
