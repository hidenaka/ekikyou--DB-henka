# 64卦アイコン画像生成プロンプト

## 共通指示（全アイコン共通）

```
あなたは易経（I Ching）の六十四卦をモダンなアイコンとしてデザインするデザイナーです。
八卦アイコンと同じテイストで、64卦それぞれの個性を表現してください。

【デザインスタイル】
- モダンでミニマルなフラットデザイン
- 円形のアイコンベース（app icon style）
- 微細なグラデーションで立体感を演出
- 背景は透過
- 512x512px

【必須要素】
1. その卦を象徴する自然物・状況・シンボル（メイン）
2. 伝統的な六本の爻線（━━━ or ━ ━）をアイコン下部または背景に小さく配置
3. 上卦・下卦の色を融合したカラーパレット

【デザイン原則】
- 易経に馴染みのない人でも直感的に意味が伝わる
- 日本・中国の伝統要素と現代的デザインの融合
- シンプルだが印象的
- 八卦アイコンとの統一感を保つ
- 動画素材としても映える
```

---

## 八卦カラーパレット参照

| 八卦 | 主色 | HEX |
|-----|------|-----|
| 乾（天） | ゴールド/パープル | #D4AF37 / #663399 |
| 兌（沢） | スカイブルー | #87CEEB |
| 離（火） | 朱赤/オレンジ | #E34234 / #FF6B35 |
| 震（雷） | イエロー | #FFD700 |
| 巽（風） | エメラルドグリーン | #50C878 |
| 坎（水） | ディープブルー | #000080 / #1E3A5F |
| 艮（山） | アースブラウン | #8B4513 |
| 坤（地） | テラコッタ | #CD853F / #DAA520 |

---

## 爻線の表記法

```
陽爻（実線）: ━━━
陰爻（断線）: ━ ━

六爻の並び（下から上へ）:
上爻 ━━━ or ━ ━
五爻 ━━━ or ━ ━
四爻 ━━━ or ━ ━
三爻 ━━━ or ━ ━
二爻 ━━━ or ━ ━
初爻 ━━━ or ━ ━
```

---

## 64卦プロンプト一覧

### 1. 乾為天（けんいてん）Qian - The Creative

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 1 "Qian" (乾為天), The Creative / Heaven over Heaven.

Visual elements:
- Radiant golden sun at zenith, emanating powerful light rays
- Six dragons ascending in a spiral pattern (optional: simplified to one majestic dragon)
- Sense of ultimate creative power and celestial authority
- Six solid lines (━━━━━━) subtly in background

Trigram composition:
- Upper: Qian (Heaven) ☰
- Lower: Qian (Heaven) ☰

Color palette:
- Primary: Royal gold (#D4AF37) to deep purple (#663399) gradient
- Accent: White radiance, silver highlights
- Background: Transparent

Mood: Supreme power, pure creativity, ultimate yang, unstoppable force
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Creation, strength, leadership, heaven, dragon, primal power
```

---

### 2. 坤為地（こんいち）Kun - The Receptive

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 2 "Kun" (坤為地), The Receptive / Earth over Earth.

Visual elements:
- Vast nurturing earth landscape extending to horizon
- Gentle rolling hills or fertile fields
- Mare (female horse) silhouette representing devoted strength
- Six broken lines (━ ━ ━ ━ ━ ━) subtly in background

Trigram composition:
- Upper: Kun (Earth) ☷
- Lower: Kun (Earth) ☷

Color palette:
- Primary: Terracotta (#CD853F) to golden wheat (#DAA520)
- Accent: Soft green growth, brown earth tones
- Background: Transparent

Mood: Ultimate receptivity, nurturing, devotion, mother earth, pure yin
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Receptivity, support, devotion, earth, mare, yielding strength
```

---

### 3. 水雷屯（すいらいちゅん）Zhun - Difficulty at the Beginning

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 3 "Zhun" (水雷屯), Difficulty at the Beginning / Water over Thunder.

Visual elements:
- A seedling pushing through dark soil into stormy rain
- Thunder clouds above with rain falling
- Sense of difficult birth, struggle, but potential
- Lightning flash illuminating the sprouting seed

Trigram composition:
- Upper: Kan (Water) ☵ - rain, difficulty
- Lower: Zhen (Thunder) ☳ - movement, sprouting

Color palette:
- Primary: Deep navy (#1E3A5F) blending into electric yellow (#FFD700)
- Accent: Green seedling, white lightning
- Background: Transparent

Mood: Birth pains, initial chaos, gathering resources, hopeful struggle
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Beginning difficulties, birth, chaos, potential, sprouting, gathering
```

---

### 4. 山水蒙（さんすいもう）Meng - Youthful Folly

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 4 "Meng" (山水蒙), Youthful Folly / Mountain over Water.

Visual elements:
- A spring emerging from the base of a mountain
- Young student or child figure seeking guidance
- Mist or fog representing unclear vision
- Mountain blocking the path, requiring guidance to navigate

Trigram composition:
- Upper: Gen (Mountain) ☶ - stillness, obstacle
- Lower: Kan (Water) ☵ - hidden depth, danger

Color palette:
- Primary: Earth brown (#8B4513) fading into deep blue (#1E3A5F)
- Accent: White mist, silver spring water
- Background: Transparent

Mood: Inexperience, seeking wisdom, student mindset, covered spring
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Youth, learning, folly, guidance needed, inexperience, spring
```

---

### 5. 水天需（すいてんじゅ）Xu - Waiting

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 5 "Xu" (水天需), Waiting / Water over Heaven.

Visual elements:
- Rain clouds gathering above a bright sky
- Figure waiting patiently under shelter
- Food and drink suggesting nourishment during waiting
- Sense of confident patience, not anxious waiting

Trigram composition:
- Upper: Kan (Water) ☵ - rain, clouds, danger ahead
- Lower: Qian (Heaven) ☰ - strength, confidence

Color palette:
- Primary: Deep blue (#1E3A5F) above golden (#D4AF37)
- Accent: Grey clouds, white highlights
- Background: Transparent

Mood: Patient waiting, nourishment, confident anticipation, timing
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Waiting, patience, nourishment, timing, clouds before sun
```

---

### 6. 天水訟（てんすいしょう）Song - Conflict

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 6 "Song" (天水訟), Conflict / Heaven over Water.

Visual elements:
- Two forces moving in opposite directions (heaven rising, water falling)
- Legal scales or courtroom symbolism
- Tension between two parties
- Warning sign suggesting conflict should be avoided

Trigram composition:
- Upper: Qian (Heaven) ☰ - strong, ascending
- Lower: Kan (Water) ☵ - cunning, descending

Color palette:
- Primary: Gold (#D4AF37) clashing with deep blue (#1E3A5F)
- Accent: Red warning, grey tension
- Background: Transparent

Mood: Dispute, legal conflict, opposing forces, need for compromise
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Conflict, lawsuit, opposition, dispute, incompatibility
```

---

### 7. 地水師（ちすいし）Shi - The Army

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 7 "Shi" (地水師), The Army / Earth over Water.

Visual elements:
- Organized troops or disciplined formation
- Underground water representing hidden strength
- General or leader figure commanding respect
- Sense of organized power, discipline, collective action

Trigram composition:
- Upper: Kun (Earth) ☷ - masses, obedience
- Lower: Kan (Water) ☵ - hidden danger, discipline

Color palette:
- Primary: Terracotta (#CD853F) over deep navy (#1E3A5F)
- Accent: Silver armor, red banners
- Background: Transparent

Mood: Military discipline, organized strength, leadership of masses
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Army, discipline, leadership, organized force, hidden strength
```

---

### 8. 水地比（すいちひ）Bi - Holding Together

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 8 "Bi" (水地比), Holding Together / Water over Earth.

Visual elements:
- Water flowing over and uniting with earth
- Multiple streams joining into one river
- Hands clasped together or people in circle
- Sense of alliance, union, seeking a leader

Trigram composition:
- Upper: Kan (Water) ☵ - flowing, uniting
- Lower: Kun (Earth) ☷ - receptive, supporting

Color palette:
- Primary: Deep blue (#1E3A5F) merging with terracotta (#CD853F)
- Accent: Green growth from union
- Background: Transparent

Mood: Union, alliance, seeking leadership, holding together, support
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Unity, alliance, support, following, holding together
```

---

### 9. 風天小畜（ふうてんしょうちく）Xiao Xu - Small Accumulating

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 9 "Xiao Xu" (風天小畜), Small Accumulating / Wind over Heaven.

Visual elements:
- Gentle clouds gathering but not yet raining
- Wind gently restraining powerful force
- Small savings or collection growing slowly
- Sense of gentle restraint, not yet time for action

Trigram composition:
- Upper: Xun (Wind) ☴ - gentle, penetrating
- Lower: Qian (Heaven) ☰ - strong, creative

Color palette:
- Primary: Emerald green (#50C878) softening gold (#D4AF37)
- Accent: White clouds, soft grey
- Background: Transparent

Mood: Gentle restraint, small accumulation, patience, clouds gathering
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Small accumulation, gentle restraint, patience, gathering clouds
```

---

### 10. 天沢履（てんたくり）Lu - Treading

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 10 "Lu" (天沢履), Treading / Heaven over Lake.

Visual elements:
- Person carefully walking, treading on tiger's tail
- Tiger and human in delicate balance
- Footsteps on a precarious path
- Sense of careful conduct, proper behavior prevents danger

Trigram composition:
- Upper: Qian (Heaven) ☰ - strength, tiger
- Lower: Dui (Lake) ☱ - joy, youngest daughter

Color palette:
- Primary: Gold (#D4AF37) above sky blue (#87CEEB)
- Accent: Tiger stripes (orange/black), careful footprints
- Background: Transparent

Mood: Careful conduct, treading carefully, proper behavior, danger awareness
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Treading, conduct, carefulness, tiger's tail, propriety
```

---

### 11. 地天泰（ちてんたい）Tai - Peace

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 11 "Tai" (地天泰), Peace / Earth over Heaven.

Visual elements:
- Heaven and Earth in harmonious exchange
- Spring season with growth and prosperity
- Yin and yang in perfect balance, intermingling
- Flourishing garden or abundant harvest

Trigram composition:
- Upper: Kun (Earth) ☷ - earth descending
- Lower: Qian (Heaven) ☰ - heaven ascending
- They meet and mingle!

Color palette:
- Primary: Terracotta (#CD853F) harmonizing with gold (#D4AF37)
- Accent: Vibrant green growth, blue sky
- Background: Transparent

Mood: Perfect peace, prosperity, harmony, spring, flourishing
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Peace, prosperity, harmony, spring, flourishing, balance
```

---

### 12. 天地否（てんちひ）Pi - Standstill

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 12 "Pi" (天地否), Standstill / Heaven over Earth.

Visual elements:
- Heaven and Earth separated, not communicating
- Autumn/winter scene of decline
- Closed gate or blocked path
- Sense of stagnation, things not flowing

Trigram composition:
- Upper: Qian (Heaven) ☰ - heaven rising away
- Lower: Kun (Earth) ☷ - earth sinking down
- They move apart!

Color palette:
- Primary: Gold (#D4AF37) separated from brown (#CD853F)
- Accent: Grey, muted tones, withered plants
- Background: Transparent

Mood: Stagnation, decline, separation, autumn, blocked communication
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Stagnation, standstill, decline, separation, blocked
```

---

### 13. 天火同人（てんかどうじん）Tong Ren - Fellowship

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 13 "Tong Ren" (天火同人), Fellowship / Heaven over Fire.

Visual elements:
- People gathered around a fire under open sky
- Torch or beacon uniting people
- Circle of fellowship, shared purpose
- Sense of community, common goals

Trigram composition:
- Upper: Qian (Heaven) ☰ - clarity, openness
- Lower: Li (Fire) ☲ - illumination, warmth

Color palette:
- Primary: Gold (#D4AF37) with orange-red (#FF6B35)
- Accent: Warm firelight, silhouettes of people
- Background: Transparent

Mood: Fellowship, community, shared purpose, openness, gathering
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Fellowship, community, unity, shared fire, common purpose
```

---

### 14. 火天大有（かてんたいゆう）Da You - Great Possession

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 14 "Da You" (火天大有), Great Possession / Fire over Heaven.

Visual elements:
- Sun at its zenith, illuminating everything
- Abundance, treasure, great harvest
- Fire burning bright in the heavens
- Sense of supreme success, great wealth

Trigram composition:
- Upper: Li (Fire) ☲ - sun, clarity
- Lower: Qian (Heaven) ☰ - strength, power

Color palette:
- Primary: Bright orange-red (#E34234) blazing over gold (#D4AF37)
- Accent: Yellow sunlight, rich golden tones
- Background: Transparent

Mood: Great abundance, supreme success, wealth, midday sun
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Great possession, abundance, success, wealth, midday sun
```

---

### 15. 地山謙（ちざんけん）Qian - Modesty

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 15 "Qian" (地山謙), Modesty / Earth over Mountain.

Visual elements:
- Mountain hidden beneath the earth (humility)
- Bowing figure or lowered head
- Balanced scales representing fairness
- Sense of humble strength, hidden greatness

Trigram composition:
- Upper: Kun (Earth) ☷ - lowly, receptive
- Lower: Gen (Mountain) ☶ - inner strength hidden

Color palette:
- Primary: Terracotta (#CD853F) over muted brown (#8B4513)
- Accent: Soft, understated tones
- Background: Transparent

Mood: Modesty, humility, hidden strength, balanced, understated power
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Modesty, humility, mountain beneath earth, hidden greatness
```

---

### 16. 雷地予（らいちよ）Yu - Enthusiasm

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 16 "Yu" (雷地予), Enthusiasm / Thunder over Earth.

Visual elements:
- Thunder breaking forth from the earth
- Music, dancing, celebration
- Drums or musical instruments
- Sense of joyful movement, inspiring others

Trigram composition:
- Upper: Zhen (Thunder) ☳ - movement, excitement
- Lower: Kun (Earth) ☷ - support, following

Color palette:
- Primary: Electric yellow (#FFD700) bursting from brown (#CD853F)
- Accent: Festive colors, rhythmic patterns
- Background: Transparent

Mood: Enthusiasm, music, celebration, inspiring movement, joy
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Enthusiasm, music, celebration, thunder from earth, inspiration
```

---

### 17. 沢雷随（たくらいずい）Sui - Following

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 17 "Sui" (沢雷随), Following / Lake over Thunder.

Visual elements:
- Thunder resting beneath the lake (evening rest)
- Leader followed by loyal companions
- Adaptable movement, going with the flow
- Sense of joyful following, adapting to circumstances

Trigram composition:
- Upper: Dui (Lake) ☱ - joy, youngest
- Lower: Zhen (Thunder) ☳ - movement, eldest son

Color palette:
- Primary: Sky blue (#87CEEB) over electric yellow (#FFD700)
- Accent: Soft evening tones
- Background: Transparent

Mood: Following, adaptation, evening rest, joyful compliance
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Following, adaptation, rest, compliance, going with flow
```

---

### 18. 山風蠱（さんぷうこ）Gu - Work on the Decayed

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 18 "Gu" (山風蠱), Work on the Decayed / Mountain over Wind.

Visual elements:
- Bowl with decay or corruption that needs cleaning
- Repairing broken vessel or fixing damage
- Wind trapped under mountain, stagnation
- Sense of necessary repair, correcting past mistakes

Trigram composition:
- Upper: Gen (Mountain) ☶ - stopping, stillness
- Lower: Xun (Wind) ☴ - penetrating, but blocked

Color palette:
- Primary: Brown (#8B4513) over muted green (#50C878)
- Accent: Signs of repair, renewal emerging
- Background: Transparent

Mood: Decay requiring repair, fixing mistakes, renovation, correction
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Decay, repair, corruption, renovation, fixing mistakes
```

---

### 19. 地沢臨（ちたくりん）Lin - Approach

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 19 "Lin" (地沢臨), Approach / Earth over Lake.

Visual elements:
- Earth approaching and overlooking a lake
- Superior approaching inferior with care
- Spring approaching, growth beginning
- Sense of benevolent oversight, teaching moment

Trigram composition:
- Upper: Kun (Earth) ☷ - receptive, nurturing
- Lower: Dui (Lake) ☱ - joy, openness

Color palette:
- Primary: Terracotta (#CD853F) gently over sky blue (#87CEEB)
- Accent: Spring green, hopeful tones
- Background: Transparent

Mood: Approach, oversight, teaching, spring coming, benevolent authority
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Approach, oversight, teaching, spring, benevolent authority
```

---

### 20. 風地観（ふうちかん）Guan - Contemplation

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 20 "Guan" (風地観), Contemplation / Wind over Earth.

Visual elements:
- Watchtower or high vantage point overlooking land
- Wind blowing across the earth, observing all
- Contemplative figure in meditation
- Sense of deep observation, being observed, example-setting

Trigram composition:
- Upper: Xun (Wind) ☴ - penetrating view
- Lower: Kun (Earth) ☷ - vast land below

Color palette:
- Primary: Emerald green (#50C878) over terracotta (#CD853F)
- Accent: Clear sky, far-seeing tones
- Background: Transparent

Mood: Contemplation, observation, example-setting, high view
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Contemplation, observation, watchtower, example, viewing
```

---

### 21. 火雷噬嗑（からいぜいごう）Shi He - Biting Through

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 21 "Shi He" (火雷噬嗑), Biting Through / Fire over Thunder.

Visual elements:
- Open mouth with teeth biting through obstacle
- Lightning strike breaking through barrier
- Justice, law enforcement, removing obstruction
- Sense of decisive action, punishment, clearing blockage

Trigram composition:
- Upper: Li (Fire) ☲ - clarity, justice
- Lower: Zhen (Thunder) ☳ - decisive movement

Color palette:
- Primary: Orange-red (#E34234) with electric yellow (#FFD700)
- Accent: Sharp white teeth, decisive lines
- Background: Transparent

Mood: Biting through, justice, decisive action, punishment, clearing
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Biting through, justice, decisive, punishment, obstruction removed
```

---

### 22. 山火賁（さんかひ）Bi - Grace

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 22 "Bi" (山火賁), Grace / Mountain over Fire.

Visual elements:
- Fire illuminating a mountain at sunset
- Beautiful decoration, ornament, elegance
- Artistic patterns, refined aesthetics
- Sense of beauty, but beauty as secondary to substance

Trigram composition:
- Upper: Gen (Mountain) ☶ - stillness, form
- Lower: Li (Fire) ☲ - illumination, beauty

Color palette:
- Primary: Brown (#8B4513) with warm orange (#FF6B35)
- Accent: Sunset colors, elegant patterns
- Background: Transparent

Mood: Grace, beauty, adornment, elegance, refined form
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Grace, beauty, adornment, sunset mountain, elegance
```

---

### 23. 山地剥（さんちはく）Bo - Splitting Apart

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 23 "Bo" (山地剥), Splitting Apart / Mountain over Earth.

Visual elements:
- Mountain eroding, crumbling at its base
- Structure collapsing from below
- Bed with legs being cut away
- Sense of decay, deterioration, things falling apart

Trigram composition:
- Upper: Gen (Mountain) ☶ - about to fall
- Lower: Kun (Earth) ☷ - undermining base

Color palette:
- Primary: Brown (#8B4513) crumbling into dark terracotta (#CD853F)
- Accent: Grey decay, warning red accents
- Background: Transparent

Mood: Splitting apart, decay, collapse, undermining, deterioration
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Splitting apart, collapse, decay, erosion, undermining
```

---

### 24. 地雷復（ちらいふく）Fu - Return

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 24 "Fu" (地雷復), Return / Earth over Thunder.

Visual elements:
- Thunder returning beneath the earth (winter solstice)
- Single yang line emerging from all yin
- Sunrise or new beginning
- Sense of return, renewal, turning point

Trigram composition:
- Upper: Kun (Earth) ☷ - dormant surface
- Lower: Zhen (Thunder) ☳ - first movement returning

Color palette:
- Primary: Terracotta (#CD853F) with golden dawn (#FFD700)
- Accent: First light breaking through, hopeful yellow
- Background: Transparent

Mood: Return, renewal, winter solstice, turning point, first yang
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Return, renewal, solstice, turning point, first light
```

---

### 25. 天雷无妄（てんらいむぼう）Wu Wang - Innocence

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 25 "Wu Wang" (天雷无妄), Innocence / Heaven over Thunder.

Visual elements:
- Thunder moving naturally under heaven's guidance
- Pure heart, uncorrupted intention
- Natural movement without ulterior motive
- Sense of innocence, sincerity, unexpected fortune

Trigram composition:
- Upper: Qian (Heaven) ☰ - natural law
- Lower: Zhen (Thunder) ☳ - natural movement

Color palette:
- Primary: Gold (#D4AF37) with electric yellow (#FFD700)
- Accent: Pure white, clear intentions
- Background: Transparent

Mood: Innocence, sincerity, natural action, no ulterior motive
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Innocence, sincerity, unexpected, natural, pure intention
```

---

### 26. 山天大畜（さんてんたいちく）Da Xu - Great Accumulating

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 26 "Da Xu" (山天大畜), Great Accumulating / Mountain over Heaven.

Visual elements:
- Mountain containing heaven's power within
- Treasury or storehouse of great value
- Tamed horse or disciplined power
- Sense of great accumulation, restrained power, cultivation

Trigram composition:
- Upper: Gen (Mountain) ☶ - holding, containing
- Lower: Qian (Heaven) ☰ - great power restrained

Color palette:
- Primary: Brown (#8B4513) containing gold (#D4AF37)
- Accent: Rich treasure tones, disciplined strength
- Background: Transparent

Mood: Great accumulation, restrained power, cultivation, treasury
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Great accumulation, restraint, cultivation, stored power
```

---

### 27. 山雷頤（さんらいい）Yi - Nourishment

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 27 "Yi" (山雷頤), Nourishment / Mountain over Thunder.

Visual elements:
- Open mouth receiving nourishment (jaws)
- Food, feeding, sustenance
- Mountain above thunder forming mouth shape
- Sense of nourishment, what we consume and express

Trigram composition:
- Upper: Gen (Mountain) ☶ - upper jaw
- Lower: Zhen (Thunder) ☳ - lower jaw moving

Color palette:
- Primary: Brown (#8B4513) with yellow (#FFD700)
- Accent: Green nourishing food, life-giving tones
- Background: Transparent

Mood: Nourishment, feeding, consumption, words spoken, sustenance
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Nourishment, jaws, feeding, what we consume and express
```

---

### 28. 沢風大過（たくふうたいか）Da Guo - Great Exceeding

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 28 "Da Guo" (沢風大過), Great Exceeding / Lake over Wind.

Visual elements:
- Beam bending under excessive weight
- Lake flooding over trees (wind/wood)
- Structure pushed to breaking point
- Sense of excess, critical pressure, extraordinary times

Trigram composition:
- Upper: Dui (Lake) ☱ - weight, pressure
- Lower: Xun (Wind/Wood) ☴ - supporting beam bending

Color palette:
- Primary: Sky blue (#87CEEB) pressing on green (#50C878)
- Accent: Warning red, stressed structural lines
- Background: Transparent

Mood: Excess, extraordinary pressure, critical point, bending beam
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Great exceeding, excess, pressure, breaking point, extraordinary
```

---

### 29. 坎為水（かんいすい）Kan - The Abysmal

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 29 "Kan" (坎為水), The Abysmal / Water over Water.

Visual elements:
- Double abyss, water flowing into water
- Deep gorge or dangerous pit
- Moon reflected in dark water
- Sense of repeated danger, depth, but also sincerity that carries through

Trigram composition:
- Upper: Kan (Water) ☵ - danger
- Lower: Kan (Water) ☵ - danger again

Color palette:
- Primary: Deep navy (#000080) to midnight blue (#1E3A5F)
- Accent: Silver moonlight, white foam on dark water
- Background: Transparent

Mood: Double danger, abyss, depth, sincerity through peril, persistence
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Abyss, danger, depth, water, persistence through danger
```

---

### 30. 離為火（りいか）Li - The Clinging

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 30 "Li" (離為火), The Clinging / Fire over Fire.

Visual elements:
- Double flame, fire clinging to fuel
- Sun and its reflection, double brightness
- Eyes seeing clearly (fire = clarity)
- Sense of dependence, clarity, brilliance

Trigram composition:
- Upper: Li (Fire) ☲ - fire
- Lower: Li (Fire) ☲ - fire again

Color palette:
- Primary: Vermillion (#E34234) to bright orange (#FF6B35)
- Accent: Yellow flame core, radiant light
- Background: Transparent

Mood: Clinging fire, double brightness, clarity, dependence on fuel
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Clinging, fire, brightness, clarity, dependence, illumination
```

---

### 31. 沢山咸（たくざんかん）Xian - Influence

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 31 "Xian" (沢山咸), Influence / Lake over Mountain.

Visual elements:
- Lake resting on mountain (mutual attraction)
- Young couple, courtship, marriage
- Heart connection, mutual influence
- Sense of attraction, wooing, receptive influence

Trigram composition:
- Upper: Dui (Lake/Youngest Daughter) ☱
- Lower: Gen (Mountain/Youngest Son) ☶
- Young couple!

Color palette:
- Primary: Sky blue (#87CEEB) over brown (#8B4513)
- Accent: Romantic pink, connection lines
- Background: Transparent

Mood: Mutual attraction, influence, courtship, marriage, connection
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Influence, attraction, courtship, marriage, mutual feeling
```

---

### 32. 雷風恒（らいふうこう）Heng - Duration

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 32 "Heng" (雷風恒), Duration / Thunder over Wind.

Visual elements:
- Thunder and wind working together endlessly
- Married couple in lasting union
- Eternal cycle, seasons continuing
- Sense of perseverance, constancy, lasting commitment

Trigram composition:
- Upper: Zhen (Thunder/Eldest Son) ☳
- Lower: Xun (Wind/Eldest Daughter) ☴
- Married couple!

Color palette:
- Primary: Electric yellow (#FFD700) with emerald green (#50C878)
- Accent: Eternal circle, steady rhythms
- Background: Transparent

Mood: Duration, constancy, marriage, perseverance, eternal cycle
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Duration, constancy, perseverance, marriage, lasting
```

---

### 33. 天山遯（てんざんとん）Dun - Retreat

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 33 "Dun" (天山遯), Retreat / Heaven over Mountain.

Visual elements:
- Figure retreating to mountain sanctuary
- Strategic withdrawal, not defeat
- Pig (symbol of retreat) heading to hills
- Sense of timely retreat, strength in withdrawal

Trigram composition:
- Upper: Qian (Heaven) ☰ - strength
- Lower: Gen (Mountain) ☶ - stopping, boundary

Color palette:
- Primary: Gold (#D4AF37) withdrawing to brown (#8B4513)
- Accent: Misty distance, strategic grey
- Background: Transparent

Mood: Strategic retreat, withdrawal, sanctuary, timely escape
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Retreat, withdrawal, sanctuary, strategic, timely escape
```

---

### 34. 雷天大壮（らいてんたいそう）Da Zhuang - Great Power

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 34 "Da Zhuang" (雷天大壮), Great Power / Thunder over Heaven.

Visual elements:
- Ram with powerful horns, charging forward
- Thunder roaring across heaven
- Explosive strength, great vigor
- Sense of powerful action, but warning against excess

Trigram composition:
- Upper: Zhen (Thunder) ☳ - movement, power
- Lower: Qian (Heaven) ☰ - strength doubled

Color palette:
- Primary: Electric yellow (#FFD700) blazing over gold (#D4AF37)
- Accent: Ram's horns, powerful burst
- Background: Transparent

Mood: Great power, vigor, strength in action, ram charging
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Great power, strength, ram, vigor, thunder over heaven
```

---

### 35. 火地晋（かちしん）Jin - Progress

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 35 "Jin" (火地晋), Progress / Fire over Earth.

Visual elements:
- Sun rising brilliantly over the earth
- Rapid advancement, promotion
- Prince receiving horses, recognition
- Sense of easy progress, recognition, sunrise success

Trigram composition:
- Upper: Li (Fire/Sun) ☲ - brilliance rising
- Lower: Kun (Earth) ☷ - supportive base

Color palette:
- Primary: Orange-red (#E34234) rising over terracotta (#CD853F)
- Accent: Golden sunrise, advancement arrows
- Background: Transparent

Mood: Progress, advancement, sunrise, recognition, easy success
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Progress, advancement, sunrise, recognition, promotion
```

---

### 36. 地火明夷（ちかめいい）Ming Yi - Darkening of the Light

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 36 "Ming Yi" (地火明夷), Darkening of the Light / Earth over Fire.

Visual elements:
- Sun setting beneath the earth, light hidden
- Wounded phoenix or injured bird
- Intelligence hidden in darkness, persecution
- Sense of light obscured, wisdom concealed for survival

Trigram composition:
- Upper: Kun (Earth) ☷ - darkness covering
- Lower: Li (Fire) ☲ - light wounded

Color palette:
- Primary: Dark terracotta (#CD853F) covering muted orange (#FF6B35)
- Accent: Dim red glow, hidden ember
- Background: Transparent

Mood: Darkening, hidden light, persecution, survival, wounded brilliance
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Darkening light, hidden, persecution, sunset, wounded bird
```

---

### 37. 風火家人（ふうかかじん）Jia Ren - The Family

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 37 "Jia Ren" (風火家人), The Family / Wind over Fire.

Visual elements:
- Warm hearth with family gathered around
- Wind (words) spreading from fire (heart of home)
- House with warm glow within
- Sense of family harmony, proper roles, domestic order

Trigram composition:
- Upper: Xun (Wind/Eldest Daughter) ☴
- Lower: Li (Fire/Middle Daughter) ☲
- Daughters maintaining home!

Color palette:
- Primary: Emerald green (#50C878) with warm orange (#FF6B35)
- Accent: Warm hearth glow, family silhouettes
- Background: Transparent

Mood: Family, home, warmth, proper roles, domestic harmony
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Family, home, hearth, domestic order, proper roles
```

---

### 38. 火沢睽（かたくけい）Kui - Opposition

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 38 "Kui" (火沢睽), Opposition / Fire over Lake.

Visual elements:
- Fire and water facing each other, opposing
- Two sisters looking away from each other
- Things that should unite but are separate
- Sense of estrangement, opposition, but potential for small matters

Trigram composition:
- Upper: Li (Fire/Middle Daughter) ☲
- Lower: Dui (Lake/Youngest Daughter) ☱
- Sisters opposed!

Color palette:
- Primary: Orange-red (#E34234) opposed to sky blue (#87CEEB)
- Accent: Tension line, divided space
- Background: Transparent

Mood: Opposition, estrangement, contrast, small successes despite division
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Opposition, estrangement, contrast, divided, sisters opposed
```

---

### 39. 水山蹇（すいざんけん）Jian - Obstruction

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 39 "Jian" (水山蹇), Obstruction / Water over Mountain.

Visual elements:
- Dangerous water blocking mountain path
- Limping person facing obstacle
- Steep cliff with water hazard
- Sense of obstruction, need to stop and reassess

Trigram composition:
- Upper: Kan (Water) ☵ - danger ahead
- Lower: Gen (Mountain) ☶ - stopped

Color palette:
- Primary: Deep blue (#1E3A5F) blocking brown (#8B4513)
- Accent: Warning red, difficult path
- Background: Transparent

Mood: Obstruction, difficulty, limping, need to pause, reassess
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Obstruction, difficulty, limping, blocked path, reassess
```

---

### 40. 雷水解（らいすいかい）Xie - Deliverance

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 40 "Xie" (雷水解), Deliverance / Thunder over Water.

Visual elements:
- Thunder breaking through rain clouds
- Knot being untied, tension released
- Spring rain bringing relief
- Sense of release, liberation, problems dissolving

Trigram composition:
- Upper: Zhen (Thunder) ☳ - breaking free
- Lower: Kan (Water) ☵ - difficulty dissolving

Color palette:
- Primary: Electric yellow (#FFD700) breaking through deep blue (#1E3A5F)
- Accent: Fresh rain, spring green
- Background: Transparent

Mood: Deliverance, release, liberation, knot untied, spring relief
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Deliverance, release, liberation, untying, spring thunder
```

---

### 41. 山沢損（さんたくそん）Sun - Decrease

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 41 "Sun" (山沢損), Decrease / Mountain over Lake.

Visual elements:
- Lake at mountain's base diminishing
- Offering being made upward (sacrifice)
- Less is more, voluntary reduction
- Sense of decrease that leads to increase, sincere sacrifice

Trigram composition:
- Upper: Gen (Mountain) ☶ - receiving above
- Lower: Dui (Lake) ☱ - giving from below

Color palette:
- Primary: Brown (#8B4513) receiving from sky blue (#87CEEB)
- Accent: Humble offering, upward movement
- Background: Transparent

Mood: Decrease, sacrifice, giving, sincerity, less leads to more
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Decrease, sacrifice, offering, reduction, sincere giving
```

---

### 42. 風雷益（ふうらいえき）Yi - Increase

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 42 "Yi" (風雷益), Increase / Wind over Thunder.

Visual elements:
- Wind and thunder working together, multiplying effect
- Growth, abundance, benefits flowing down
- Leader sharing with people below
- Sense of increase, expansion, generosity from above

Trigram composition:
- Upper: Xun (Wind) ☴ - benefits spreading
- Lower: Zhen (Thunder) ☳ - receiving below

Color palette:
- Primary: Emerald green (#50C878) flowing to yellow (#FFD700)
- Accent: Abundant growth, downward blessing
- Background: Transparent

Mood: Increase, abundance, generosity, benefits flowing, expansion
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Increase, abundance, generosity, growth, benefits flowing
```

---

### 43. 沢天夬（たくてんかい）Guai - Breakthrough

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 43 "Guai" (沢天夬), Breakthrough / Lake over Heaven.

Visual elements:
- Water breaking through dam decisively
- Five yang lines pushing out one yin
- Proclamation in royal court, exposure of evil
- Sense of decisive breakthrough, resolution, determination

Trigram composition:
- Upper: Dui (Lake) ☱ - decisive speech
- Lower: Qian (Heaven) ☰ - strength pushing

Color palette:
- Primary: Sky blue (#87CEEB) breaking through gold (#D4AF37)
- Accent: Decisive white, breakthrough moment
- Background: Transparent

Mood: Breakthrough, resolution, decisive action, exposing, determination
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Breakthrough, resolution, decisive, dam breaking, proclamation
```

---

### 44. 天風姤（てんぷうこう）Gou - Coming to Meet

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 44 "Gou" (天風姤), Coming to Meet / Heaven over Wind.

Visual elements:
- Wind rising to meet heaven unexpectedly
- One yin line entering from below (unexpected encounter)
- Bold woman approaching, sudden meeting
- Sense of unexpected encounter, temptation, caution needed

Trigram composition:
- Upper: Qian (Heaven) ☰ - five yang above
- Lower: Xun (Wind) ☴ - one yin sneaking in

Color palette:
- Primary: Gold (#D4AF37) with subtle green (#50C878) entering
- Accent: Surprise element, caution yellow
- Background: Transparent

Mood: Unexpected meeting, encounter, temptation, caution, sudden
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Coming to meet, unexpected, encounter, temptation, caution
```

---

### 45. 沢地萃（たくちすい）Cui - Gathering Together

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 45 "Cui" (沢地萃), Gathering Together / Lake over Earth.

Visual elements:
- Lake gathering water from earth
- People assembling at temple or gathering place
- Congregation, assembly, offerings
- Sense of gathering, unity, collective purpose

Trigram composition:
- Upper: Dui (Lake) ☱ - joy of gathering
- Lower: Kun (Earth) ☷ - masses coming together

Color palette:
- Primary: Sky blue (#87CEEB) over terracotta (#CD853F)
- Accent: Gathering crowd, ceremonial tones
- Background: Transparent

Mood: Gathering, assembly, congregation, collective, offerings
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Gathering, assembly, congregation, unity, collective purpose
```

---

### 46. 地風升（ちふうしょう）Sheng - Pushing Upward

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 46 "Sheng" (地風升), Pushing Upward / Earth over Wind.

Visual elements:
- Tree/plant pushing up through earth
- Gradual ascent, step by step climbing
- Seeds germinating, growth from below
- Sense of upward movement, advancement, growth

Trigram composition:
- Upper: Kun (Earth) ☷ - yielding above
- Lower: Xun (Wind/Wood) ☴ - growing upward

Color palette:
- Primary: Terracotta (#CD853F) with green (#50C878) rising
- Accent: Upward arrows, growth patterns
- Background: Transparent

Mood: Pushing upward, growth, gradual ascent, advancement
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Pushing upward, growth, ascent, advancement, tree growing
```

---

### 47. 沢水困（たくすいこん）Kun - Oppression

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 47 "Kun" (沢水困), Oppression / Lake over Water.

Visual elements:
- Lake with water drained away beneath
- Tree without water, exhaustion
- Confined space, oppressive situation
- Sense of exhaustion, oppression, but inner strength

Trigram composition:
- Upper: Dui (Lake) ☱ - empty above
- Lower: Kan (Water) ☵ - drained below

Color palette:
- Primary: Faded blue (#87CEEB) over dark navy (#1E3A5F)
- Accent: Exhausted grey, but inner glow
- Background: Transparent

Mood: Oppression, exhaustion, confined, but maintaining inner worth
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Oppression, exhaustion, confined, drained, inner strength
```

---

### 48. 水風井（すいふうせい）Jing - The Well

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 48 "Jing" (水風井), The Well / Water over Wind.

Visual elements:
- Traditional well with bucket and rope
- Wood/wind structure holding up water
- Unchanging source, nourishing community
- Sense of reliable resource, inexhaustible wisdom

Trigram composition:
- Upper: Kan (Water) ☵ - water drawn up
- Lower: Xun (Wind/Wood) ☴ - well structure

Color palette:
- Primary: Deep blue (#1E3A5F) over green (#50C878)
- Accent: Stone well rim, clear water
- Background: Transparent

Mood: Well, reliable source, community resource, unchanging wisdom
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Well, inexhaustible, community, reliable source, wisdom
```

---

### 49. 沢火革（たくかかく）Ge - Revolution

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 49 "Ge" (沢火革), Revolution / Lake over Fire.

Visual elements:
- Fire and water creating transformation
- Animal skin being changed/molted
- Complete transformation, radical change
- Sense of revolution, renewal, shedding old form

Trigram composition:
- Upper: Dui (Lake) ☱ - water
- Lower: Li (Fire) ☲ - fire
- Opposites creating change!

Color palette:
- Primary: Sky blue (#87CEEB) transforming with orange (#FF6B35)
- Accent: Transformation patterns, molting imagery
- Background: Transparent

Mood: Revolution, transformation, molting, radical change, renewal
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Revolution, transformation, change, molting, renewal
```

---

### 50. 火風鼎（かふうてい）Ding - The Cauldron

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 50 "Ding" (火風鼎), The Cauldron / Fire over Wind.

Visual elements:
- Ancient bronze cauldron with three legs
- Fire beneath, wood feeding flames
- Sacred vessel for offerings and transformation
- Sense of nourishment, civilization, refinement

Trigram composition:
- Upper: Li (Fire) ☲ - fire cooking
- Lower: Xun (Wind/Wood) ☴ - wood fuel

Color palette:
- Primary: Orange-red (#E34234) over green (#50C878)
- Accent: Bronze/copper tones, ceremonial dignity
- Background: Transparent

Mood: Cauldron, nourishment, civilization, sacred cooking, refinement
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Cauldron, nourishment, civilization, transformation, sacred
```

---

### 51. 震為雷（しんいらい）Zhen - The Arousing

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 51 "Zhen" (震為雷), The Arousing / Thunder over Thunder.

Visual elements:
- Double thunder, repeated shock
- Lightning striking twice
- Startled figure then composed
- Sense of shock leading to awakening, fear then composure

Trigram composition:
- Upper: Zhen (Thunder) ☳ - shock
- Lower: Zhen (Thunder) ☳ - shock again

Color palette:
- Primary: Electric yellow (#FFD700) intensified
- Accent: White lightning, purple storm clouds
- Background: Transparent

Mood: Double shock, arousing, awakening, fear then composure
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Arousing, thunder, shock, awakening, double thunder
```

---

### 52. 艮為山（ごんいさん）Gen - Keeping Still

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 52 "Gen" (艮為山), Keeping Still / Mountain over Mountain.

Visual elements:
- Double mountain, complete stillness
- Meditation pose, perfect stillness
- Back turned, not seeing temptation
- Sense of profound stillness, meditation, stopping

Trigram composition:
- Upper: Gen (Mountain) ☶ - stillness
- Lower: Gen (Mountain) ☶ - stillness again

Color palette:
- Primary: Earth brown (#8B4513) deepened
- Accent: Stone grey, meditative calm
- Background: Transparent

Mood: Complete stillness, meditation, stopping, inner peace
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Stillness, meditation, mountain, stopping, keeping still
```

---

### 53. 風山漸（ふうざんぜん）Jian - Development

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 53 "Jian" (風山漸), Development / Wind over Mountain.

Visual elements:
- Wild goose gradually ascending mountain
- Tree growing slowly on mountainside
- Marriage procession, proper development
- Sense of gradual progress, proper sequence, patience

Trigram composition:
- Upper: Xun (Wind/Wood) ☴ - gradual growth
- Lower: Gen (Mountain) ☶ - stable base

Color palette:
- Primary: Emerald green (#50C878) growing on brown (#8B4513)
- Accent: Goose silhouette, gradual steps
- Background: Transparent

Mood: Gradual development, patience, proper sequence, steady growth
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Development, gradual, patience, growth, wild goose
```

---

### 54. 雷沢帰妹（らいたくきまい）Gui Mei - The Marrying Maiden

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 54 "Gui Mei" (雷沢帰妹), The Marrying Maiden / Thunder over Lake.

Visual elements:
- Young woman in marriage procession
- Thunder above lake, improper order
- Secondary wife or concubine imagery
- Sense of subordinate position, making best of situation

Trigram composition:
- Upper: Zhen (Thunder/Eldest Son) ☳
- Lower: Dui (Lake/Youngest Daughter) ☱
- Improper marriage order!

Color palette:
- Primary: Electric yellow (#FFD700) over sky blue (#87CEEB)
- Accent: Wedding red, subordinate position
- Background: Transparent

Mood: Subordinate marriage, making do, improper but managing
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Marrying maiden, subordinate, improper order, acceptance
```

---

### 55. 雷火豊（らいかほう）Feng - Abundance

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 55 "Feng" (雷火豊), Abundance / Thunder over Fire.

Visual elements:
- Midday sun at its brightest
- Thunder and lightning at peak power
- Abundant harvest, fullness
- Sense of peak abundance, but awareness of decline to come

Trigram composition:
- Upper: Zhen (Thunder) ☳ - movement at peak
- Lower: Li (Fire) ☲ - brightness at zenith

Color palette:
- Primary: Electric yellow (#FFD700) with bright orange (#FF6B35)
- Accent: Abundant gold, peak radiance
- Background: Transparent

Mood: Abundance, zenith, fullness, peak power, awareness of change
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Abundance, peak, fullness, zenith, thunder and fire
```

---

### 56. 火山旅（かざんりょ）Lu - The Wanderer

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 56 "Lu" (火山旅), The Wanderer / Fire over Mountain.

Visual elements:
- Lone traveler with staff on mountain path
- Fire on mountain (brief, passing)
- Bird without nest, temporary lodging
- Sense of journey, impermanence, stranger in strange land

Trigram composition:
- Upper: Li (Fire) ☲ - bright but brief
- Lower: Gen (Mountain) ☶ - foreign territory

Color palette:
- Primary: Orange (#FF6B35) flickering on brown (#8B4513)
- Accent: Traveler silhouette, path winding
- Background: Transparent

Mood: Wandering, travel, impermanence, stranger, brief stay
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Wanderer, travel, impermanence, stranger, journey
```

---

### 57. 巽為風（そんいふう）Xun - The Gentle

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 57 "Xun" (巽為風), The Gentle / Wind over Wind.

Visual elements:
- Double wind, gentle but penetrating
- Grass bending in continuous breeze
- Subtle influence spreading everywhere
- Sense of gentle penetration, persistent influence

Trigram composition:
- Upper: Xun (Wind) ☴ - gentle
- Lower: Xun (Wind) ☴ - gentle again

Color palette:
- Primary: Emerald green (#50C878) flowing
- Accent: Soft white breeze lines, subtle movement
- Background: Transparent

Mood: Gentle penetration, persistent influence, subtle, pervasive
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Gentle, wind, penetrating, subtle, persistent influence
```

---

### 58. 兌為沢（だいたく）Dui - The Joyous

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 58 "Dui" (兌為沢), The Joyous / Lake over Lake.

Visual elements:
- Double lake, connected joyful waters
- Smiling face, open expression
- Friends in joyful conversation
- Sense of pure joy, communication, shared happiness

Trigram composition:
- Upper: Dui (Lake) ☱ - joy
- Lower: Dui (Lake) ☱ - joy again

Color palette:
- Primary: Sky blue (#87CEEB) radiant
- Accent: Sparkling water, joyful white
- Background: Transparent

Mood: Pure joy, communication, friendship, shared happiness
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Joy, lake, communication, friendship, happiness
```

---

### 59. 風水渙（ふうすいかん）Huan - Dispersion

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 59 "Huan" (風水渙), Dispersion / Wind over Water.

Visual elements:
- Wind scattering water, breaking ice
- Spring thaw dissolving winter's hardness
- Dispersing gathered negativity
- Sense of dissolution, scattering, breaking up stagnation

Trigram composition:
- Upper: Xun (Wind) ☴ - dispersing force
- Lower: Kan (Water) ☵ - frozen/gathered

Color palette:
- Primary: Emerald green (#50C878) dispersing deep blue (#1E3A5F)
- Accent: Scattering patterns, spring freshness
- Background: Transparent

Mood: Dispersion, dissolution, scattering, spring thaw, breaking ice
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Dispersion, dissolution, scattering, thaw, breaking up
```

---

### 60. 水沢節（すいたくせつ）Jie - Limitation

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 60 "Jie" (水沢節), Limitation / Water over Lake.

Visual elements:
- Lake with defined banks, proper limits
- Bamboo joints (節) showing natural articulation
- Balanced scales, proper measure
- Sense of healthy limitation, proper boundaries

Trigram composition:
- Upper: Kan (Water) ☵ - regulated flow
- Lower: Dui (Lake) ☱ - contained joy

Color palette:
- Primary: Deep blue (#1E3A5F) over sky blue (#87CEEB)
- Accent: Bamboo joints, measured lines
- Background: Transparent

Mood: Limitation, articulation, proper measure, healthy boundaries
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Limitation, articulation, boundaries, measure, bamboo
```

---

### 61. 風沢中孚（ふうたくちゅうふ）Zhong Fu - Inner Truth

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 61 "Zhong Fu" (風沢中孚), Inner Truth / Wind over Lake.

Visual elements:
- Bird sitting on eggs, faithful nurturing
- Hollow center with solid sides (卦の形)
- Crane calling, truthful expression
- Sense of inner sincerity, trustworthiness, faithful heart

Trigram composition:
- Upper: Xun (Wind) ☴ - spreading truth
- Lower: Dui (Lake) ☱ - open reception

Color palette:
- Primary: Emerald green (#50C878) over sky blue (#87CEEB)
- Accent: White crane, pure heart symbol
- Background: Transparent

Mood: Inner truth, sincerity, faithfulness, trustworthiness
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Inner truth, sincerity, faithfulness, crane, trustworthy
```

---

### 62. 雷山小過（らいざんしょうか）Xiao Guo - Small Exceeding

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 62 "Xiao Guo" (雷山小過), Small Exceeding / Thunder over Mountain.

Visual elements:
- Bird flying too high, small transgression
- Thunder trapped between mountains
- Slight excess, minor overstepping
- Sense of small matters only, don't aim too high

Trigram composition:
- Upper: Zhen (Thunder) ☳ - movement limited
- Lower: Gen (Mountain) ☶ - stopping

Color palette:
- Primary: Electric yellow (#FFD700) muted by brown (#8B4513)
- Accent: Small bird, limited flight
- Background: Transparent

Mood: Small excess, minor transgression, limited scope, stay low
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Small exceeding, minor, limited, bird, small matters only
```

---

### 63. 水火既済（すいかきせい）Ji Ji - After Completion

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 63 "Ji Ji" (水火既済), After Completion / Water over Fire.

Visual elements:
- Water and fire in perfect balance
- All lines in proper places, completion
- Sunset, task completed
- Sense of completion, but vigilance needed for new challenges

Trigram composition:
- Upper: Kan (Water) ☵ - over fire
- Lower: Li (Fire) ☲ - under water
- Perfect balance!

Color palette:
- Primary: Deep blue (#1E3A5F) over orange-red (#E34234)
- Accent: Balanced harmony, completion symbol
- Background: Transparent

Mood: Completion, balance, success achieved, but stay vigilant
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: After completion, balance, success, vigilance, harmony
```

---

### 64. 火水未済（かすいびせい）Wei Ji - Before Completion

```
【生成プロンプト】

Create a modern minimalist icon for Hexagram 64 "Wei Ji" (火水未済), Before Completion / Fire over Water.

Visual elements:
- Fire and water not yet in harmony
- Fox crossing river, almost across
- Dawn before sunrise, potential
- Sense of not yet complete, potential, careful progress needed

Trigram composition:
- Upper: Li (Fire) ☲ - over water
- Lower: Kan (Water) ☵ - under fire
- Not yet balanced!

Color palette:
- Primary: Orange-red (#E34234) over deep blue (#1E3A5F)
- Accent: Fox crossing, almost-there moment
- Background: Transparent

Mood: Not yet complete, potential, almost there, careful final steps
Style: Flat design with subtle gradients, circular icon format, 512x512px

Keywords: Before completion, potential, almost, fox crossing, not yet
```

---

## ファイル命名規則

```
hexagram_{番号:02d}_{ローマ字名}_{サイズ}.png

例:
hexagram_01_qian_512.png      (乾為天)
hexagram_03_zhun_512.png      (水雷屯)
hexagram_11_tai_512.png       (地天泰)
hexagram_63_jiji_512.png      (水火既済)
hexagram_64_weiji_512.png     (火水未済)
```

---

## 一括生成用プロンプト（グループ別）

### 上経（1-30卦）まとめ生成

```
易経（I Ching）の上経（Hexagram 1-30）をモダンなアプリアイコンスタイルでデザインしてください。

【共通スタイル】
- 円形アイコン（512x512px）
- フラットデザイン＋微細グラデーション
- 各アイコンの背景に六本の爻線を小さく配置
- 背景透過
- 八卦アイコンと同じモダン×東洋的テイスト

【上経のテーマ】
上経は宇宙論的・自然的な卦が中心。天・地・水・火などの根本原理を扱う。

【カラー原則】
各卦は上卦・下卦の色を融合。
例: 水雷屯（3番）= 深青（坎）×電光黄（震）
```

### 下経（31-64卦）まとめ生成

```
易経（I Ching）の下経（Hexagram 31-64）をモダンなアプリアイコンスタイルでデザインしてください。

【共通スタイル】
- 円形アイコン（512x512px）
- フラットデザイン＋微細グラデーション
- 各アイコンの背景に六本の爻線を小さく配置
- 背景透過
- 八卦アイコンと同じモダン×東洋的テイスト

【下経のテーマ】
下経は人間関係・社会的な卦が中心。結婚、家族、社会など人の営みを扱う。

【カラー原則】
各卦は上卦・下卦の色を融合。
例: 沢山咸（31番）= 水色（兌）×茶（艮）= 結婚・影響
```

---

## 参照ドキュメント

- `docs/image_generation_prompts.md` - 八卦アイコンプロンプト
- `docs/visual_design_guide.md` - カラーパレット、デザイン原則
- `assets/icons/` - 生成済み八卦アイコン（参考用）
