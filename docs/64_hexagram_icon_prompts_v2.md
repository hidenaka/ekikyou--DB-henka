# 64卦アイコン画像生成プロンプト v2

## 重要：スタイル統一のための厳格なルール

### 必ず守るべきレイアウト構成

```
┌─────────────────────────────────┐
│                                 │
│     【シンボル・イラスト】        │
│      （アイコン上部60%）          │
│                                 │
├─────────────────────────────────┤
│                                 │
│     ━━━━━━━━━━  ← 上爻（6番目）  │
│     ━━━  ━━━   ← 五爻（5番目）  │
│     ━━━━━━━━━━  ← 四爻（4番目）  │
│     ━━━  ━━━   ← 三爻（3番目）  │
│     ━━━━━━━━━━  ← 二爻（2番目）  │
│     ━━━  ━━━   ← 初爻（1番目）  │
│      （六本の爻線 下部30%）       │
│                                 │
│     【卦名】                     │
│     （最下部10%）                │
│                                 │
└─────────────────────────────────┘
```

### 爻線の描き方（最重要）

```
【陽爻（Yang Line）】
━━━━━━━━━━━━━━━━
→ 途切れのない1本の実線
→ 太さ: 線幅8-10px程度
→ 色: 卦のメインカラーまたは白/金

【陰爻（Yin Line）】
━━━━━   ━━━━━
→ 中央に明確な隙間（線幅の1.5倍程度）
→ 左右対称の2つの短線
→ 同じ太さ・色で統一
```

---

## マスタープロンプト（全64卦共通）

```
You are creating a series of 64 I Ching hexagram icons.
ALL icons MUST follow this EXACT specification to maintain visual consistency:

████████████████████████████████████████████████████████
██  YOU MUST DRAW EXACTLY 6 HORIZONTAL LINES.         ██
██  NOT 5, NOT 7, NOT 8. EXACTLY 6.                   ██
██  COUNT THEM BEFORE FINISHING: 1, 2, 3, 4, 5, 6     ██
████████████████████████████████████████████████████████

【CRITICAL LAYOUT REQUIREMENTS】

1. CANVAS: 512x512px, circular icon with solid color background

2. STRUCTURE (top to bottom):
   - Top 60%: Symbolic illustration representing the hexagram meaning
   - Bottom 30%: Six horizontal lines (爻 yao lines) stacked vertically
   - Bottom 10%: Hexagram name in Japanese kanji

3. YAO LINES (六爻) - THIS IS CRITICAL:
   - Yang line (陽爻): One solid unbroken horizontal line ━━━━━━━━
   - Yin line (陰爻): Two short lines with a clear gap in center ━━━ ━━━
   - Lines are stacked from bottom (line 1/初爻) to top (line 6/上爻)
   - All lines same width, evenly spaced
   - Line color: white, gold, or matching the icon's accent color
   - Lines should be clearly visible against background

4. HEXAGRAM NAME:
   - Display at bottom center
   - Format: "卦名" (Japanese kanji only)
   - Example: "乾為天", "水雷屯", "地天泰"
   - Font: Clean, readable, slightly stylized
   - Color: White or gold for contrast

5. STYLE:
   - Modern minimalist flat design
   - Subtle gradients for depth
   - Eastern aesthetic meets contemporary design
   - Consistent with the 8 trigram icons already created
   - Professional, app-icon quality

6. COLORS:
   - Background: Solid color based on the dominant trigram
   - Blend upper and lower trigram colors harmoniously
   - High contrast between background and yao lines
```

---

## 64卦個別プロンプト

### 1. 乾為天

```
Create icon for Hexagram 1: 乾為天 (乾 - 創造・剛健)

SYMBOL: Heaven over Heaven. 創造・剛健. sky, sun, metal and sky, sun, metal imagery.
Visual focus: 天の動きは剛健である。君子はこれに則り、自ら努めて息まない

BINARY: 111111
TRIGRAMS: Upper=Heaven(乾) + Lower=Heaven(乾)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 乾為天

BACKGROUND: Deep Royal Purple (#2D1B4E) to Navy Gradient
ACCENT: Gold (#D4AF37)
SYMBOL STYLE: Modern, minimal, conveying "創造・剛健"
Keywords: 創造, リーダーシップ, 剛健, 前進, 天
```

---

### 2. 坤為地

```
Create icon for Hexagram 2: 坤為地 (坤 - 受容・柔順)

SYMBOL: Earth over Earth. 受容・柔順. earth, soil, field and earth, soil, field imagery.
Visual focus: 地の勢いは順である。君子はこれに則り、厚い徳で万物を載せる

BINARY: 000000
TRIGRAMS: Upper=Earth(坤) + Lower=Earth(坤)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 坤為地

BACKGROUND: Warm Terracotta (#CD853F) Gradient
ACCENT: Wheat/Beige
SYMBOL STYLE: Modern, minimal, conveying "受容・柔順"
Keywords: 受容, 柔順, 育成, 従う, 地
```

---

### 3. 水雷屯

```
Create icon for Hexagram 3: 水雷屯 (屯 - 困難の始まり・産みの苦しみ)

SYMBOL: Water over Thunder. 困難の始まり・産みの苦しみ. water, rain, moon and thunder, sprout imagery.
Visual focus: 雲雷が立ち込める。君子はこれに則り、事業を整える

BINARY: 100010
TRIGRAMS: Upper=Water(坎) + Lower=Thunder(震)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 水雷屯

BACKGROUND: Gradient blending Water (Deep Navy (#000080)) and Thunder (Electric Yellow (#FFD700))
ACCENT: Silver
SYMBOL STYLE: Modern, minimal, conveying "困難の始まり・産みの苦しみ"
Keywords: 困難, 始まり, 忍耐, 基盤づくり
```

---

### 4. 山水蒙

```
Create icon for Hexagram 4: 山水蒙 (蒙 - 未熟・啓蒙)

SYMBOL: Mountain over Water. 未熟・啓蒙. mountain, stone and water, rain, moon imagery.
Visual focus: 山の下に水が湧く。君子はこれに則り、徳を養う

BINARY: 010001
TRIGRAMS: Upper=Mountain(艮) + Lower=Water(坎)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 山水蒙

BACKGROUND: Gradient blending Mountain (Earth Brown (#8B4513)) and Water (Deep Navy (#000080))
ACCENT: Forest Green
SYMBOL STYLE: Modern, minimal, conveying "未熟・啓蒙"
Keywords: 未熟, 学び, 教育, 指導を受ける
```

---

### 5. 水天需

```
Create icon for Hexagram 5: 水天需 (需 - 待つ・養う)

SYMBOL: Water over Heaven. 待つ・養う. water, rain, moon and sky, sun, metal imagery.
Visual focus: 雲が天に上る。君子はこれに則り、飲食宴楽する

BINARY: 111010
TRIGRAMS: Upper=Water(坎) + Lower=Heaven(乾)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 水天需

BACKGROUND: Gradient blending Water (Deep Navy (#000080)) and Heaven (Deep Royal Purple (#2D1B4E) to Navy)
ACCENT: Silver
SYMBOL STYLE: Modern, minimal, conveying "待つ・養う"
Keywords: 待機, 忍耐, 準備, 時機を待つ
```

---

### 6. 天水訟

```
Create icon for Hexagram 6: 天水訟 (訟 - 争い・訴訟)

SYMBOL: Heaven over Water. 争い・訴訟. sky, sun, metal and water, rain, moon imagery.
Visual focus: 天と水が背き合う。君子はこれに則り、事の始めを謀る

BINARY: 010111
TRIGRAMS: Upper=Heaven(乾) + Lower=Water(坎)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 天水訟

BACKGROUND: Gradient blending Heaven (Deep Royal Purple (#2D1B4E) to Navy) and Water (Deep Navy (#000080))
ACCENT: Gold (#D4AF37)
SYMBOL STYLE: Modern, minimal, conveying "争い・訴訟"
Keywords: 争い, 対立, 訴訟, 仲裁
```

---

### 7. 地水師

```
Create icon for Hexagram 7: 地水師 (師 - 軍隊・統率)

SYMBOL: Earth over Water. 軍隊・統率. earth, soil, field and water, rain, moon imagery.
Visual focus: 地中に水がある。君子はこれに則り、民を養い衆を蓄える

BINARY: 010000
TRIGRAMS: Upper=Earth(坤) + Lower=Water(坎)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 地水師

BACKGROUND: Gradient blending Earth (Warm Terracotta (#CD853F)) and Water (Deep Navy (#000080))
ACCENT: Wheat/Beige
SYMBOL STYLE: Modern, minimal, conveying "軍隊・統率"
Keywords: 統率, 組織, 規律, リーダーシップ
```

---

### 8. 水地比

```
Create icon for Hexagram 8: 水地比 (比 - 親しむ・団結)

SYMBOL: Water over Earth. 親しむ・団結. water, rain, moon and earth, soil, field imagery.
Visual focus: 地上に水がある。先王はこれに則り、万国を建て諸侯を親しむ

BINARY: 000010
TRIGRAMS: Upper=Water(坎) + Lower=Earth(坤)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 水地比

BACKGROUND: Gradient blending Water (Deep Navy (#000080)) and Earth (Warm Terracotta (#CD853F))
ACCENT: Silver
SYMBOL STYLE: Modern, minimal, conveying "親しむ・団結"
Keywords: 団結, 協力, 親睦, 連携
```

---

### 9. 風天小畜

```
Create icon for Hexagram 9: 風天小畜 (小畜 - 小さく蓄える)

SYMBOL: Wind over Heaven. 小さく蓄える. wind, wood and sky, sun, metal imagery.
Visual focus: 風が天を行く。君子はこれに則り、文徳を美しくする

BINARY: 111011
TRIGRAMS: Upper=Wind(巽) + Lower=Heaven(乾)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 風天小畜

BACKGROUND: Gradient blending Wind (Emerald Green (#50C878)) and Heaven (Deep Royal Purple (#2D1B4E) to Navy)
ACCENT: White
SYMBOL STYLE: Modern, minimal, conveying "小さく蓄える"
Keywords: 小さな蓄積, 準備, 徐々に, 忍耐
```

---

### 10. 天沢履

```
Create icon for Hexagram 10: 天沢履 (履 - 礼を踏む・慎重に歩む)

SYMBOL: Heaven over Lake. 礼を踏む・慎重に歩む. sky, sun, metal and marsh, lake, reflection imagery.
Visual focus: 天の下に沢がある。君子はこれに則り、上下を辨え民の志を定める

BINARY: 110111
TRIGRAMS: Upper=Heaven(乾) + Lower=Lake(兌)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 天沢履

BACKGROUND: Gradient blending Heaven (Deep Royal Purple (#2D1B4E) to Navy) and Lake (Sky Blue (#87CEEB))
ACCENT: Gold (#D4AF37)
SYMBOL STYLE: Modern, minimal, conveying "礼を踏む・慎重に歩む"
Keywords: 礼儀, 慎重, 危険回避, 正しい行動
```

---

### 11. 地天泰

```
Create icon for Hexagram 11: 地天泰 (泰 - 通じる・安泰)

SYMBOL: Earth over Heaven. 通じる・安泰. earth, soil, field and sky, sun, metal imagery.
Visual focus: 天地が交わる。君子はこれに則り、天地の道を裁成し輔相する

BINARY: 111000
TRIGRAMS: Upper=Earth(坤) + Lower=Heaven(乾)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 地天泰

BACKGROUND: Gradient blending Earth (Warm Terracotta (#CD853F)) and Heaven (Deep Royal Purple (#2D1B4E) to Navy)
ACCENT: Wheat/Beige
SYMBOL STYLE: Modern, minimal, conveying "通じる・安泰"
Keywords: 安泰, 順調, 調和, 通じる
```

---

### 12. 天地否

```
Create icon for Hexagram 12: 天地否 (否 - 塞がる・閉塞)

SYMBOL: Heaven over Earth. 塞がる・閉塞. sky, sun, metal and earth, soil, field imagery.
Visual focus: 天地が交わらない。君子はこれに則り、徳を倹約して難を避ける

BINARY: 000111
TRIGRAMS: Upper=Heaven(乾) + Lower=Earth(坤)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 天地否

BACKGROUND: Gradient blending Heaven (Deep Royal Purple (#2D1B4E) to Navy) and Earth (Warm Terracotta (#CD853F))
ACCENT: Gold (#D4AF37)
SYMBOL STYLE: Modern, minimal, conveying "塞がる・閉塞"
Keywords: 閉塞, 停滞, 忍耐, 内省
```

---

### 13. 天火同人

```
Create icon for Hexagram 13: 天火同人 (同人 - 人と同じくする・協調)

SYMBOL: Heaven over Fire. 人と同じくする・協調. sky, sun, metal and fire, sun, lightning imagery.
Visual focus: 天と火が同じ。君子はこれに則り、族を類し物を辨つ

BINARY: 101111
TRIGRAMS: Upper=Heaven(乾) + Lower=Fire(離)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 天火同人

BACKGROUND: Gradient blending Heaven (Deep Royal Purple (#2D1B4E) to Navy) and Fire (Vermillion Red (#E34234) to Orange)
ACCENT: Gold (#D4AF37)
SYMBOL STYLE: Modern, minimal, conveying "人と同じくする・協調"
Keywords: 協調, 同志, 団結, 共通目的
```

---

### 14. 火天大有

```
Create icon for Hexagram 14: 火天大有 (大有 - 大いに持つ・豊か)

SYMBOL: Fire over Heaven. 大いに持つ・豊か. fire, sun, lightning and sky, sun, metal imagery.
Visual focus: 火が天上にある。君子はこれに則り、悪を遏め善を揚げ天命に順う

BINARY: 111101
TRIGRAMS: Upper=Fire(離) + Lower=Heaven(乾)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 火天大有

BACKGROUND: Gradient blending Fire (Vermillion Red (#E34234) to Orange) and Heaven (Deep Royal Purple (#2D1B4E) to Navy)
ACCENT: Bright Yellow
SYMBOL STYLE: Modern, minimal, conveying "大いに持つ・豊か"
Keywords: 豊穣, 成功, 富, 謙虚さ
```

---

### 15. 地山謙

```
Create icon for Hexagram 15: 地山謙 (謙 - 謙遜・へりくだる)

SYMBOL: Earth over Mountain. 謙遜・へりくだる. earth, soil, field and mountain, stone imagery.
Visual focus: 地中に山がある。君子はこれに則り、多きを減らし少なきに益す

BINARY: 001000
TRIGRAMS: Upper=Earth(坤) + Lower=Mountain(艮)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 地山謙

BACKGROUND: Gradient blending Earth (Warm Terracotta (#CD853F)) and Mountain (Earth Brown (#8B4513))
ACCENT: Wheat/Beige
SYMBOL STYLE: Modern, minimal, conveying "謙遜・へりくだる"
Keywords: 謙虚, 控えめ, バランス, 徳
```

---

### 16. 雷地予

```
Create icon for Hexagram 16: 雷地予 (予 - 喜び・準備)

SYMBOL: Thunder over Earth. 喜び・準備. thunder, sprout and earth, soil, field imagery.
Visual focus: 雷が地を出づる。先王はこれに則り、楽を作り徳を崇め、殷に上帝に薦め祖考を配す

BINARY: 000100
TRIGRAMS: Upper=Thunder(震) + Lower=Earth(坤)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 雷地予

BACKGROUND: Gradient blending Thunder (Electric Yellow (#FFD700)) and Earth (Warm Terracotta (#CD853F))
ACCENT: Purple
SYMBOL STYLE: Modern, minimal, conveying "喜び・準備"
Keywords: 喜び, 準備, 楽観, 用意
```

---

### 17. 沢雷随

```
Create icon for Hexagram 17: 沢雷随 (随 - 従う・随う)

SYMBOL: Lake over Thunder. 従う・随う. marsh, lake, reflection and thunder, sprout imagery.
Visual focus: 沢中に雷がある。君子はこれに則り、晦きに向かいて入り宴息す

BINARY: 100110
TRIGRAMS: Upper=Lake(兌) + Lower=Thunder(震)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 沢雷随

BACKGROUND: Gradient blending Lake (Sky Blue (#87CEEB)) and Thunder (Electric Yellow (#FFD700))
ACCENT: Soft Pink
SYMBOL STYLE: Modern, minimal, conveying "従う・随う"
Keywords: 随順, 柔軟, 適応, 従う
```

---

### 18. 山風蠱

```
Create icon for Hexagram 18: 山風蠱 (蠱 - 腐敗・立て直し)

SYMBOL: Mountain over Wind. 腐敗・立て直し. mountain, stone and wind, wood imagery.
Visual focus: 山の下に風がある。君子はこれに則り、民を振るい徳を育む

BINARY: 011001
TRIGRAMS: Upper=Mountain(艮) + Lower=Wind(巽)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 山風蠱

BACKGROUND: Gradient blending Mountain (Earth Brown (#8B4513)) and Wind (Emerald Green (#50C878))
ACCENT: Forest Green
SYMBOL STYLE: Modern, minimal, conveying "腐敗・立て直し"
Keywords: 改革, 立て直し, 腐敗除去, 刷新
```

---

### 19. 地沢臨

```
Create icon for Hexagram 19: 地沢臨 (臨 - 臨む・監督)

SYMBOL: Earth over Lake. 臨む・監督. earth, soil, field and marsh, lake, reflection imagery.
Visual focus: 地上に沢がある。君子はこれに則り、教え思うこと窮まりなく、民を容れ保つこと限りなし

BINARY: 110000
TRIGRAMS: Upper=Earth(坤) + Lower=Lake(兌)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 地沢臨

BACKGROUND: Gradient blending Earth (Warm Terracotta (#CD853F)) and Lake (Sky Blue (#87CEEB))
ACCENT: Wheat/Beige
SYMBOL STYLE: Modern, minimal, conveying "臨む・監督"
Keywords: 監督, 臨む, 責任, 指導
```

---

### 20. 風地観

```
Create icon for Hexagram 20: 風地観 (観 - 観る・観察)

SYMBOL: Wind over Earth. 観る・観察. wind, wood and earth, soil, field imagery.
Visual focus: 風が地上を行く。先王はこれに則り、方を省み民を観て教えを設く

BINARY: 000011
TRIGRAMS: Upper=Wind(巽) + Lower=Earth(坤)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 風地観

BACKGROUND: Gradient blending Wind (Emerald Green (#50C878)) and Earth (Warm Terracotta (#CD853F))
ACCENT: White
SYMBOL STYLE: Modern, minimal, conveying "観る・観察"
Keywords: 観察, 洞察, 視察, 理解
```

---

### 21. 火雷噬嗑

```
Create icon for Hexagram 21: 火雷噬嗑 (噬嗑 - 噛み砕く・障害除去)

SYMBOL: Fire over Thunder. 噛み砕く・障害除去. fire, sun, lightning and thunder, sprout imagery.
Visual focus: 雷電が合する。先王はこれに則り、罰を明らかにして法を勅む

BINARY: 100101
TRIGRAMS: Upper=Fire(離) + Lower=Thunder(震)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 火雷噬嗑

BACKGROUND: Gradient blending Fire (Vermillion Red (#E34234) to Orange) and Thunder (Electric Yellow (#FFD700))
ACCENT: Bright Yellow
SYMBOL STYLE: Modern, minimal, conveying "噛み砕く・障害除去"
Keywords: 障害除去, 決断, 法の執行, 断固
```

---

### 22. 山火賁

```
Create icon for Hexagram 22: 山火賁 (賁 - 飾る・文飾)

SYMBOL: Mountain over Fire. 飾る・文飾. mountain, stone and fire, sun, lightning imagery.
Visual focus: 山の下に火がある。君子はこれに則り、庶政を明らかにし、折獄を敢えてせず

BINARY: 101001
TRIGRAMS: Upper=Mountain(艮) + Lower=Fire(離)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 山火賁

BACKGROUND: Gradient blending Mountain (Earth Brown (#8B4513)) and Fire (Vermillion Red (#E34234) to Orange)
ACCENT: Forest Green
SYMBOL STYLE: Modern, minimal, conveying "飾る・文飾"
Keywords: 装飾, 文化, 外見, 形式
```

---

### 23. 山地剥

```
Create icon for Hexagram 23: 山地剥 (剥 - 剥がれる・崩壊)

SYMBOL: Mountain over Earth. 剥がれる・崩壊. mountain, stone and earth, soil, field imagery.
Visual focus: 山が地に附く。上は以て下を厚くし宅を安んず

BINARY: 000001
TRIGRAMS: Upper=Mountain(艮) + Lower=Earth(坤)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 山地剥

BACKGROUND: Gradient blending Mountain (Earth Brown (#8B4513)) and Earth (Warm Terracotta (#CD853F))
ACCENT: Forest Green
SYMBOL STYLE: Modern, minimal, conveying "剥がれる・崩壊"
Keywords: 衰退, 崩壊, 静観, 忍耐
```

---

### 24. 地雷復

```
Create icon for Hexagram 24: 地雷復 (復 - 復る・回復)

SYMBOL: Earth over Thunder. 復る・回復. earth, soil, field and thunder, sprout imagery.
Visual focus: 雷が地中に在る。先王はこれに則り、至日に関を閉じ商旅行かず

BINARY: 100000
TRIGRAMS: Upper=Earth(坤) + Lower=Thunder(震)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 地雷復

BACKGROUND: Gradient blending Earth (Warm Terracotta (#CD853F)) and Thunder (Electric Yellow (#FFD700))
ACCENT: Wheat/Beige
SYMBOL STYLE: Modern, minimal, conveying "復る・回復"
Keywords: 回復, 再生, 復活, 始まり
```

---

### 25. 天雷无妄

```
Create icon for Hexagram 25: 天雷无妄 (无妄 - 無妄・誠実)

SYMBOL: Heaven over Thunder. 無妄・誠実. sky, sun, metal and thunder, sprout imagery.
Visual focus: 天の下に雷行る。先王はこれに則り、時に茂り万物を育む

BINARY: 100111
TRIGRAMS: Upper=Heaven(乾) + Lower=Thunder(震)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 天雷无妄

BACKGROUND: Gradient blending Heaven (Deep Royal Purple (#2D1B4E) to Navy) and Thunder (Electric Yellow (#FFD700))
ACCENT: Gold (#D4AF37)
SYMBOL STYLE: Modern, minimal, conveying "無妄・誠実"
Keywords: 誠実, 正直, 自然, 無作為
```

---

### 26. 山天大畜

```
Create icon for Hexagram 26: 山天大畜 (大畜 - 大いに蓄える)

SYMBOL: Mountain over Heaven. 大いに蓄える. mountain, stone and sky, sun, metal imagery.
Visual focus: 天が山中にある。君子はこれに則り、多く前言往行を識し以て其の徳を畜う

BINARY: 111001
TRIGRAMS: Upper=Mountain(艮) + Lower=Heaven(乾)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 山天大畜

BACKGROUND: Gradient blending Mountain (Earth Brown (#8B4513)) and Heaven (Deep Royal Purple (#2D1B4E) to Navy)
ACCENT: Forest Green
SYMBOL STYLE: Modern, minimal, conveying "大いに蓄える"
Keywords: 蓄積, 学習, 準備, 力を貯める
```

---

### 27. 山雷頤

```
Create icon for Hexagram 27: 山雷頤 (頤 - 養う・口)

SYMBOL: Mountain over Thunder. 養う・口. mountain, stone and thunder, sprout imagery.
Visual focus: 山の下に雷がある。君子はこれに則り、言を慎み飲食を節す

BINARY: 100001
TRIGRAMS: Upper=Mountain(艮) + Lower=Thunder(震)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 山雷頤

BACKGROUND: Gradient blending Mountain (Earth Brown (#8B4513)) and Thunder (Electric Yellow (#FFD700))
ACCENT: Forest Green
SYMBOL STYLE: Modern, minimal, conveying "養う・口"
Keywords: 養生, 節制, 慎み, 育てる
```

---

### 28. 沢風大過

```
Create icon for Hexagram 28: 沢風大過 (大過 - 大いに過ぎる)

SYMBOL: Lake over Wind. 大いに過ぎる. marsh, lake, reflection and wind, wood imagery.
Visual focus: 沢が木を滅する。君子はこれに則り、独り立ちて懼れず、世を遁れて悶えず

BINARY: 011110
TRIGRAMS: Upper=Lake(兌) + Lower=Wind(巽)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 沢風大過

BACKGROUND: Gradient blending Lake (Sky Blue (#87CEEB)) and Wind (Emerald Green (#50C878))
ACCENT: Soft Pink
SYMBOL STYLE: Modern, minimal, conveying "大いに過ぎる"
Keywords: 過剰, 危機, 非常事態, 独立
```

---

### 29. 坎為水

```
Create icon for Hexagram 29: 坎為水 (坎 - 険難・水)

SYMBOL: Water over Water. 険難・水. water, rain, moon and water, rain, moon imagery.
Visual focus: 水が洊り至る。君子はこれに則り、常に徳行を習い教事を習う

BINARY: 010010
TRIGRAMS: Upper=Water(坎) + Lower=Water(坎)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 坎為水

BACKGROUND: Deep Navy (#000080) Gradient
ACCENT: Silver
SYMBOL STYLE: Modern, minimal, conveying "険難・水"
Keywords: 険難, 困難, 誠実, 忍耐
```

---

### 30. 離為火

```
Create icon for Hexagram 30: 離為火 (離 - 付く・明らか)

SYMBOL: Fire over Fire. 付く・明らか. fire, sun, lightning and fire, sun, lightning imagery.
Visual focus: 明が両たび作る。大人はこれに則り、継いで明を照らし四方に及ぶ

BINARY: 101101
TRIGRAMS: Upper=Fire(離) + Lower=Fire(離)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 離為火

BACKGROUND: Vermillion Red (#E34234) to Orange Gradient
ACCENT: Bright Yellow
SYMBOL STYLE: Modern, minimal, conveying "付く・明らか"
Keywords: 明晰, 付着, 輝き, 知性
```

---

### 31. 沢山咸

```
Create icon for Hexagram 31: 沢山咸 (咸 - 感応・交わり)

SYMBOL: Lake over Mountain. 感応・交わり. marsh, lake, reflection and mountain, stone imagery.
Visual focus: 山上に沢がある。君子はこれに則り、虚を以て人を受く

BINARY: 001110
TRIGRAMS: Upper=Lake(兌) + Lower=Mountain(艮)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 沢山咸

BACKGROUND: Gradient blending Lake (Sky Blue (#87CEEB)) and Mountain (Earth Brown (#8B4513))
ACCENT: Soft Pink
SYMBOL STYLE: Modern, minimal, conveying "感応・交わり"
Keywords: 感応, 交流, 共感, 結婚
```

---

### 32. 雷風恒

```
Create icon for Hexagram 32: 雷風恒 (恒 - 恒久・持続)

SYMBOL: Thunder over Wind. 恒久・持続. thunder, sprout and wind, wood imagery.
Visual focus: 雷風相い与す。君子はこれに則り、立ちて方を易えず

BINARY: 011100
TRIGRAMS: Upper=Thunder(震) + Lower=Wind(巽)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 雷風恒

BACKGROUND: Gradient blending Thunder (Electric Yellow (#FFD700)) and Wind (Emerald Green (#50C878))
ACCENT: Purple
SYMBOL STYLE: Modern, minimal, conveying "恒久・持続"
Keywords: 持続, 恒久, 一貫性, 継続
```

---

### 33. 天山遯

```
Create icon for Hexagram 33: 天山遯 (遯 - 退く・逃れる)

SYMBOL: Heaven over Mountain. 退く・逃れる. sky, sun, metal and mountain, stone imagery.
Visual focus: 天の下に山がある。君子はこれに則り、小人を遠ざけ悪まずして厳なり

BINARY: 001111
TRIGRAMS: Upper=Heaven(乾) + Lower=Mountain(艮)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 天山遯

BACKGROUND: Gradient blending Heaven (Deep Royal Purple (#2D1B4E) to Navy) and Mountain (Earth Brown (#8B4513))
ACCENT: Gold (#D4AF37)
SYMBOL STYLE: Modern, minimal, conveying "退く・逃れる"
Keywords: 退却, 隠遁, 逃避, 撤退
```

---

### 34. 雷天大壮

```
Create icon for Hexagram 34: 雷天大壮 (大壮 - 大いに壮ん)

SYMBOL: Thunder over Heaven. 大いに壮ん. thunder, sprout and sky, sun, metal imagery.
Visual focus: 雷が天上にある。君子はこれに則り、礼に非ざれば履まず

BINARY: 111100
TRIGRAMS: Upper=Thunder(震) + Lower=Heaven(乾)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 雷天大壮

BACKGROUND: Gradient blending Thunder (Electric Yellow (#FFD700)) and Heaven (Deep Royal Purple (#2D1B4E) to Navy)
ACCENT: Purple
SYMBOL STYLE: Modern, minimal, conveying "大いに壮ん"
Keywords: 壮大, 力, 勢い, 礼節
```

---

### 35. 火地晋

```
Create icon for Hexagram 35: 火地晋 (晋 - 進む・昇進)

SYMBOL: Fire over Earth. 進む・昇進. fire, sun, lightning and earth, soil, field imagery.
Visual focus: 明が地上に出づる。君子はこれに則り、自ら明徳を昭らかにす

BINARY: 000101
TRIGRAMS: Upper=Fire(離) + Lower=Earth(坤)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 火地晋

BACKGROUND: Gradient blending Fire (Vermillion Red (#E34234) to Orange) and Earth (Warm Terracotta (#CD853F))
ACCENT: Bright Yellow
SYMBOL STYLE: Modern, minimal, conveying "進む・昇進"
Keywords: 昇進, 前進, 発展, 明るい
```

---

### 36. 地火明夷

```
Create icon for Hexagram 36: 地火明夷 (明夷 - 明が傷つく)

SYMBOL: Earth over Fire. 明が傷つく. earth, soil, field and fire, sun, lightning imagery.
Visual focus: 明が地中に入る。君子はこれに則り、衆に莅みて用て晦きを以てし明を顕す

BINARY: 101000
TRIGRAMS: Upper=Earth(坤) + Lower=Fire(離)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 地火明夷

BACKGROUND: Gradient blending Earth (Warm Terracotta (#CD853F)) and Fire (Vermillion Red (#E34234) to Orange)
ACCENT: Wheat/Beige
SYMBOL STYLE: Modern, minimal, conveying "明が傷つく"
Keywords: 隠蔽, 忍耐, 暗い時期, 内に秘める
```

---

### 37. 風火家人

```
Create icon for Hexagram 37: 風火家人 (家人 - 家庭・家族)

SYMBOL: Wind over Fire. 家庭・家族. wind, wood and fire, sun, lightning imagery.
Visual focus: 風が火より出づる。君子はこれに則り、言に物有り行いに恒有り

BINARY: 101011
TRIGRAMS: Upper=Wind(巽) + Lower=Fire(離)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 風火家人

BACKGROUND: Gradient blending Wind (Emerald Green (#50C878)) and Fire (Vermillion Red (#E34234) to Orange)
ACCENT: White
SYMBOL STYLE: Modern, minimal, conveying "家庭・家族"
Keywords: 家庭, 家族, 内部, 秩序
```

---

### 38. 火沢睽

```
Create icon for Hexagram 38: 火沢睽 (睽 - 背く・そむく)

SYMBOL: Fire over Lake. 背く・そむく. fire, sun, lightning and marsh, lake, reflection imagery.
Visual focus: 上に火あり下に沢あり。君子はこれに則り、同にして異なる

BINARY: 110101
TRIGRAMS: Upper=Fire(離) + Lower=Lake(兌)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 火沢睽

BACKGROUND: Gradient blending Fire (Vermillion Red (#E34234) to Orange) and Lake (Sky Blue (#87CEEB))
ACCENT: Bright Yellow
SYMBOL STYLE: Modern, minimal, conveying "背く・そむく"
Keywords: 対立, 不和, 異なる, 小事
```

---

### 39. 水山蹇

```
Create icon for Hexagram 39: 水山蹇 (蹇 - 足を引く・困難)

SYMBOL: Water over Mountain. 足を引く・困難. water, rain, moon and mountain, stone imagery.
Visual focus: 山上に水がある。君子はこれに則り、身を反りて徳を修む

BINARY: 001010
TRIGRAMS: Upper=Water(坎) + Lower=Mountain(艮)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 水山蹇

BACKGROUND: Gradient blending Water (Deep Navy (#000080)) and Mountain (Earth Brown (#8B4513))
ACCENT: Silver
SYMBOL STYLE: Modern, minimal, conveying "足を引く・困難"
Keywords: 困難, 障害, 自省, 修養
```

---

### 40. 雷水解

```
Create icon for Hexagram 40: 雷水解 (解 - 解ける・解放)

SYMBOL: Thunder over Water. 解ける・解放. thunder, sprout and water, rain, moon imagery.
Visual focus: 雷雨作る。君子はこれに則り、過ちを赦し罪を宥む

BINARY: 010100
TRIGRAMS: Upper=Thunder(震) + Lower=Water(坎)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 雷水解

BACKGROUND: Gradient blending Thunder (Electric Yellow (#FFD700)) and Water (Deep Navy (#000080))
ACCENT: Purple
SYMBOL STYLE: Modern, minimal, conveying "解ける・解放"
Keywords: 解放, 解決, 許し, 緩和
```

---

### 41. 山沢損

```
Create icon for Hexagram 41: 山沢損 (損 - 減らす・損)

SYMBOL: Mountain over Lake. 減らす・損. mountain, stone and marsh, lake, reflection imagery.
Visual focus: 山の下に沢がある。君子はこれに則り、忿りを懲らし欲を窒ぐ

BINARY: 110001
TRIGRAMS: Upper=Mountain(艮) + Lower=Lake(兌)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 山沢損

BACKGROUND: Gradient blending Mountain (Earth Brown (#8B4513)) and Lake (Sky Blue (#87CEEB))
ACCENT: Forest Green
SYMBOL STYLE: Modern, minimal, conveying "減らす・損"
Keywords: 減少, 抑制, 節制, 献身
```

---

### 42. 風雷益

```
Create icon for Hexagram 42: 風雷益 (益 - 増やす・益)

SYMBOL: Wind over Thunder. 増やす・益. wind, wood and thunder, sprout imagery.
Visual focus: 風雷、益。君子はこれに則り、善を見ては遷り過ちあれば改む

BINARY: 100011
TRIGRAMS: Upper=Wind(巽) + Lower=Thunder(震)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 風雷益

BACKGROUND: Gradient blending Wind (Emerald Green (#50C878)) and Thunder (Electric Yellow (#FFD700))
ACCENT: White
SYMBOL STYLE: Modern, minimal, conveying "増やす・益"
Keywords: 増加, 利益, 発展, 改善
```

---

### 43. 沢天夬

```
Create icon for Hexagram 43: 沢天夬 (夬 - 決する・決壊)

SYMBOL: Lake over Heaven. 決する・決壊. marsh, lake, reflection and sky, sun, metal imagery.
Visual focus: 沢が天上に上る。君子はこれに則り、禄を施して下に居り、徳を居ることを忌む

BINARY: 111110
TRIGRAMS: Upper=Lake(兌) + Lower=Heaven(乾)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 沢天夬

BACKGROUND: Gradient blending Lake (Sky Blue (#87CEEB)) and Heaven (Deep Royal Purple (#2D1B4E) to Navy)
ACCENT: Soft Pink
SYMBOL STYLE: Modern, minimal, conveying "決する・決壊"
Keywords: 決断, 決壊, 除去, 断固
```

---

### 44. 天風姤

```
Create icon for Hexagram 44: 天風姤 (姤 - 遇う・出会い)

SYMBOL: Heaven over Wind. 遇う・出会い. sky, sun, metal and wind, wood imagery.
Visual focus: 天の下に風がある。后はこれに則り、命を施して四方に誥く

BINARY: 011111
TRIGRAMS: Upper=Heaven(乾) + Lower=Wind(巽)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 天風姤

BACKGROUND: Gradient blending Heaven (Deep Royal Purple (#2D1B4E) to Navy) and Wind (Emerald Green (#50C878))
ACCENT: Gold (#D4AF37)
SYMBOL STYLE: Modern, minimal, conveying "遇う・出会い"
Keywords: 出会い, 偶然, 注意, 機会
```

---

### 45. 沢地萃

```
Create icon for Hexagram 45: 沢地萃 (萃 - 集まる・萃)

SYMBOL: Lake over Earth. 集まる・萃. marsh, lake, reflection and earth, soil, field imagery.
Visual focus: 沢が地上にある。君子はこれに則り、戎器を除き不虞に戒む

BINARY: 000110
TRIGRAMS: Upper=Lake(兌) + Lower=Earth(坤)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 沢地萃

BACKGROUND: Gradient blending Lake (Sky Blue (#87CEEB)) and Earth (Warm Terracotta (#CD853F))
ACCENT: Soft Pink
SYMBOL STYLE: Modern, minimal, conveying "集まる・萃"
Keywords: 集合, 結集, 組織, 準備
```

---

### 46. 地風升

```
Create icon for Hexagram 46: 地風升 (升 - 昇る・上昇)

SYMBOL: Earth over Wind. 昇る・上昇. earth, soil, field and wind, wood imagery.
Visual focus: 地中に木が生ずる。君子はこれに則り、徳に順い積み、小を以て高大に至る

BINARY: 011000
TRIGRAMS: Upper=Earth(坤) + Lower=Wind(巽)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 地風升

BACKGROUND: Gradient blending Earth (Warm Terracotta (#CD853F)) and Wind (Emerald Green (#50C878))
ACCENT: Wheat/Beige
SYMBOL STYLE: Modern, minimal, conveying "昇る・上昇"
Keywords: 上昇, 成長, 漸進, 発展
```

---

### 47. 沢水困

```
Create icon for Hexagram 47: 沢水困 (困 - 困窮・苦しむ)

SYMBOL: Lake over Water. 困窮・苦しむ. marsh, lake, reflection and water, rain, moon imagery.
Visual focus: 沢に水がない。君子はこれに則り、命を致して志を遂ぐ

BINARY: 010110
TRIGRAMS: Upper=Lake(兌) + Lower=Water(坎)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 沢水困

BACKGROUND: Gradient blending Lake (Sky Blue (#87CEEB)) and Water (Deep Navy (#000080))
ACCENT: Soft Pink
SYMBOL STYLE: Modern, minimal, conveying "困窮・苦しむ"
Keywords: 困窮, 苦難, 忍耐, 志
```

---

### 48. 水風井

```
Create icon for Hexagram 48: 水風井 (井 - 井戸・源泉)

SYMBOL: Water over Wind. 井戸・源泉. water, rain, moon and wind, wood imagery.
Visual focus: 木上に水がある。君子はこれに則り、民を労い相い勧むることを助く

BINARY: 011010
TRIGRAMS: Upper=Water(坎) + Lower=Wind(巽)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 水風井

BACKGROUND: Gradient blending Water (Deep Navy (#000080)) and Wind (Emerald Green (#50C878))
ACCENT: Silver
SYMBOL STYLE: Modern, minimal, conveying "井戸・源泉"
Keywords: 源泉, 基本, 養う, 変わらぬもの
```

---

### 49. 沢火革

```
Create icon for Hexagram 49: 沢火革 (革 - 革める・変革)

SYMBOL: Lake over Fire. 革める・変革. marsh, lake, reflection and fire, sun, lightning imagery.
Visual focus: 沢中に火がある。君子はこれに則り、暦を治めて時を明らかにす

BINARY: 101110
TRIGRAMS: Upper=Lake(兌) + Lower=Fire(離)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 沢火革

BACKGROUND: Gradient blending Lake (Sky Blue (#87CEEB)) and Fire (Vermillion Red (#E34234) to Orange)
ACCENT: Soft Pink
SYMBOL STYLE: Modern, minimal, conveying "革める・変革"
Keywords: 変革, 革命, 改革, 刷新
```

---

### 50. 火風鼎

```
Create icon for Hexagram 50: 火風鼎 (鼎 - 鼎・刷新)

SYMBOL: Fire over Wind. 鼎・刷新. fire, sun, lightning and wind, wood imagery.
Visual focus: 木の上に火がある。君子はこれに則り、位を正し命を凝らす

BINARY: 011101
TRIGRAMS: Upper=Fire(離) + Lower=Wind(巽)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 火風鼎

BACKGROUND: Gradient blending Fire (Vermillion Red (#E34234) to Orange) and Wind (Emerald Green (#50C878))
ACCENT: Bright Yellow
SYMBOL STYLE: Modern, minimal, conveying "鼎・刷新"
Keywords: 刷新, 鼎立, 秩序, 新体制
```

---

### 51. 震為雷

```
Create icon for Hexagram 51: 震為雷 (震 - 震える・雷)

SYMBOL: Thunder over Thunder. 震える・雷. thunder, sprout and thunder, sprout imagery.
Visual focus: 洊雷、震。君子はこれに則り、恐懼して修省す

BINARY: 100100
TRIGRAMS: Upper=Thunder(震) + Lower=Thunder(震)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 震為雷

BACKGROUND: Electric Yellow (#FFD700) Gradient
ACCENT: Purple
SYMBOL STYLE: Modern, minimal, conveying "震える・雷"
Keywords: 衝撃, 震動, 覚醒, 恐れ
```

---

### 52. 艮為山

```
Create icon for Hexagram 52: 艮為山 (艮 - 止まる・山)

SYMBOL: Mountain over Mountain. 止まる・山. mountain, stone and mountain, stone imagery.
Visual focus: 兼山、艮。君子はこれに則り、思うこと其の位を出でず

BINARY: 001001
TRIGRAMS: Upper=Mountain(艮) + Lower=Mountain(艮)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 艮為山

BACKGROUND: Earth Brown (#8B4513) Gradient
ACCENT: Forest Green
SYMBOL STYLE: Modern, minimal, conveying "止まる・山"
Keywords: 静止, 止まる, 内省, 不動
```

---

### 53. 風山漸

```
Create icon for Hexagram 53: 風山漸 (漸 - 漸進・徐々に)

SYMBOL: Wind over Mountain. 漸進・徐々に. wind, wood and mountain, stone imagery.
Visual focus: 山上に木がある。君子はこれに則り、賢徳に居り風俗を善くす

BINARY: 001011
TRIGRAMS: Upper=Wind(巽) + Lower=Mountain(艮)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 風山漸

BACKGROUND: Gradient blending Wind (Emerald Green (#50C878)) and Mountain (Earth Brown (#8B4513))
ACCENT: White
SYMBOL STYLE: Modern, minimal, conveying "漸進・徐々に"
Keywords: 漸進, 徐々に, 着実, 順序
```

---

### 54. 雷沢帰妹

```
Create icon for Hexagram 54: 雷沢帰妹 (帰妹 - 嫁ぐ妹)

SYMBOL: Thunder over Lake. 嫁ぐ妹. thunder, sprout and marsh, lake, reflection imagery.
Visual focus: 沢上に雷がある。君子はこれに則り、終わりを以て永く弊を知る

BINARY: 110100
TRIGRAMS: Upper=Thunder(震) + Lower=Lake(兌)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 雷沢帰妹

BACKGROUND: Gradient blending Thunder (Electric Yellow (#FFD700)) and Lake (Sky Blue (#87CEEB))
ACCENT: Purple
SYMBOL STYLE: Modern, minimal, conveying "嫁ぐ妹"
Keywords: 従属, 嫁入り, 副次的, 慎み
```

---

### 55. 雷火豊

```
Create icon for Hexagram 55: 雷火豊 (豊 - 豊か・盛大)

SYMBOL: Thunder over Fire. 豊か・盛大. thunder, sprout and fire, sun, lightning imagery.
Visual focus: 雷電皆至る。君子はこれに則り、獄を折し刑を致す

BINARY: 101100
TRIGRAMS: Upper=Thunder(震) + Lower=Fire(離)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 雷火豊

BACKGROUND: Gradient blending Thunder (Electric Yellow (#FFD700)) and Fire (Vermillion Red (#E34234) to Orange)
ACCENT: Purple
SYMBOL STYLE: Modern, minimal, conveying "豊か・盛大"
Keywords: 繁栄, 盛大, 豊穣, 頂点
```

---

### 56. 火山旅

```
Create icon for Hexagram 56: 火山旅 (旅 - 旅・寄寓)

SYMBOL: Fire over Mountain. 旅・寄寓. fire, sun, lightning and mountain, stone imagery.
Visual focus: 山上に火がある。君子はこれに則り、刑を明らかにし慎み獄を留めず

BINARY: 001101
TRIGRAMS: Upper=Fire(離) + Lower=Mountain(艮)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 火山旅

BACKGROUND: Gradient blending Fire (Vermillion Red (#E34234) to Orange) and Mountain (Earth Brown (#8B4513))
ACCENT: Bright Yellow
SYMBOL STYLE: Modern, minimal, conveying "旅・寄寓"
Keywords: 旅, 移動, 不安定, 寄寓
```

---

### 57. 巽為風

```
Create icon for Hexagram 57: 巽為風 (巽 - 入る・風)

SYMBOL: Wind over Wind. 入る・風. wind, wood and wind, wood imagery.
Visual focus: 随風、巽。君子はこれに則り、命を申ね行い事を施す

BINARY: 011011
TRIGRAMS: Upper=Wind(巽) + Lower=Wind(巽)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 巽為風

BACKGROUND: Emerald Green (#50C878) Gradient
ACCENT: White
SYMBOL STYLE: Modern, minimal, conveying "入る・風"
Keywords: 浸透, 適応, 柔順, 従う
```

---

### 58. 兌為沢

```
Create icon for Hexagram 58: 兌為沢 (兌 - 悦ぶ・沢)

SYMBOL: Lake over Lake. 悦ぶ・沢. marsh, lake, reflection and marsh, lake, reflection imagery.
Visual focus: 麗沢、兌。君子はこれに則り、朋友と講習す

BINARY: 110110
TRIGRAMS: Upper=Lake(兌) + Lower=Lake(兌)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 兌為沢

BACKGROUND: Sky Blue (#87CEEB) Gradient
ACCENT: Soft Pink
SYMBOL STYLE: Modern, minimal, conveying "悦ぶ・沢"
Keywords: 喜び, 交流, 和悦, 言葉
```

---

### 59. 風水渙

```
Create icon for Hexagram 59: 風水渙 (渙 - 散らす・渙散)

SYMBOL: Wind over Water. 散らす・渙散. wind, wood and water, rain, moon imagery.
Visual focus: 風が水上を行く。先王はこれに則り、帝を享り廟を立つ

BINARY: 010011
TRIGRAMS: Upper=Wind(巽) + Lower=Water(坎)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 風水渙

BACKGROUND: Gradient blending Wind (Emerald Green (#50C878)) and Water (Deep Navy (#000080))
ACCENT: White
SYMBOL STYLE: Modern, minimal, conveying "散らす・渙散"
Keywords: 渙散, 分散, 解散, 再統合
```

---

### 60. 水沢節

```
Create icon for Hexagram 60: 水沢節 (節 - 節度・制限)

SYMBOL: Water over Lake. 節度・制限. water, rain, moon and marsh, lake, reflection imagery.
Visual focus: 沢上に水がある。君子はこれに則り、数度を制し徳行を議す

BINARY: 110010
TRIGRAMS: Upper=Water(坎) + Lower=Lake(兌)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 水沢節

BACKGROUND: Gradient blending Water (Deep Navy (#000080)) and Lake (Sky Blue (#87CEEB))
ACCENT: Silver
SYMBOL STYLE: Modern, minimal, conveying "節度・制限"
Keywords: 節度, 制限, 節制, 適度
```

---

### 61. 風沢中孚

```
Create icon for Hexagram 61: 風沢中孚 (中孚 - まこと・誠信)

SYMBOL: Wind over Lake. まこと・誠信. wind, wood and marsh, lake, reflection imagery.
Visual focus: 沢上に風がある。君子はこれに則り、獄を議して死を緩くす

BINARY: 110011
TRIGRAMS: Upper=Wind(巽) + Lower=Lake(兌)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 風沢中孚

BACKGROUND: Gradient blending Wind (Emerald Green (#50C878)) and Lake (Sky Blue (#87CEEB))
ACCENT: White
SYMBOL STYLE: Modern, minimal, conveying "まこと・誠信"
Keywords: 誠信, 信頼, まこと, 内実
```

---

### 62. 雷山小過

```
Create icon for Hexagram 62: 雷山小過 (小過 - 小さく過ぎる)

SYMBOL: Thunder over Mountain. 小さく過ぎる. thunder, sprout and mountain, stone imagery.
Visual focus: 山上に雷がある。君子はこれに則り、行いは恭に過ぎ、喪は哀に過ぎ、用は倹に過ぐ

BINARY: 001100
TRIGRAMS: Upper=Thunder(震) + Lower=Mountain(艮)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 雷山小過

BACKGROUND: Gradient blending Thunder (Electric Yellow (#FFD700)) and Mountain (Earth Brown (#8B4513))
ACCENT: Purple
SYMBOL STYLE: Modern, minimal, conveying "小さく過ぎる"
Keywords: 小過, 控えめ, 小事, 謙虚
```

---

### 63. 水火既済

```
Create icon for Hexagram 63: 水火既済 (既済 - すでに済む・完成)

SYMBOL: Water over Fire. すでに済む・完成. water, rain, moon and fire, sun, lightning imagery.
Visual focus: 水が火の上にある。君子はこれに則り、患を思いて豫め之を防ぐ

BINARY: 101010
TRIGRAMS: Upper=Water(坎) + Lower=Fire(離)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━━━━━━ (SOLID/yang)
    2: ━━━ ━━━ (BROKEN/yin)
    3: ━━━━━━━━ (SOLID/yang)
    4: ━━━ ━━━ (BROKEN/yin)
    5: ━━━━━━━━ (SOLID/yang)
    6: ━━━ ━━━ (BROKEN/yin)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━━━━ (yang)
Line 2 (二爻): ━━━ ━━━ (yin)
Line 3 (三爻): ━━━━━━ (yang)
Line 4 (四爻): ━━━ ━━━ (yin)
Line 5 (五爻): ━━━━━━ (yang)
Line 6 (上爻): ━━━ ━━━ (yin)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 水火既済

BACKGROUND: Gradient blending Water (Deep Navy (#000080)) and Fire (Vermillion Red (#E34234) to Orange)
ACCENT: Silver
SYMBOL STYLE: Modern, minimal, conveying "すでに済む・完成"
Keywords: 完成, 成就, 達成, 警戒
```

---

### 64. 火水未済

```
Create icon for Hexagram 64: 火水未済 (未済 - まだ済まず・未完成)

SYMBOL: Fire over Water. まだ済まず・未完成. fire, sun, lightning and water, rain, moon imagery.
Visual focus: 火が水の上にある。君子はこれに則り、慎みて物を辨じ方に居らしむ

BINARY: 010101
TRIGRAMS: Upper=Fire(離) + Lower=Water(坎)

VISUAL VERIFICATION (must match this exactly):
    1: ━━━ ━━━ (BROKEN/yin)
    2: ━━━━━━━━ (SOLID/yang)
    3: ━━━ ━━━ (BROKEN/yin)
    4: ━━━━━━━━ (SOLID/yang)
    5: ━━━ ━━━ (BROKEN/yin)
    6: ━━━━━━━━ (SOLID/yang)

YAO LINES (6 lines, from bottom to top):
Line 1 (初爻): ━━━ ━━━ (yin)
Line 2 (二爻): ━━━━━━ (yang)
Line 3 (三爻): ━━━ ━━━ (yin)
Line 4 (四爻): ━━━━━━ (yang)
Line 5 (五爻): ━━━ ━━━ (yin)
Line 6 (上爻): ━━━━━━ (yang)

⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)

DISPLAY NAME: 火水未済

BACKGROUND: Gradient blending Fire (Vermillion Red (#E34234) to Orange) and Water (Deep Navy (#000080))
ACCENT: Bright Yellow
SYMBOL STYLE: Modern, minimal, conveying "まだ済まず・未完成"
Keywords: 未完成, 移行, 発展途上, 慎重
```

---
