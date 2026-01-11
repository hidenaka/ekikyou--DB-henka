import json
import os

# Paths
JSON_PATH = 'data/diagnostic/hexagram_64.json'
OUTPUT_PATH = 'docs/64_hexagram_icon_prompts_v2.md'

# Trigram Data (Bottom to Top binary)
# 1 = Yang (Solid), 0 = Yin (Broken)
TRIGRAMS = {
    "乾": {"bin": [1, 1, 1], "name": "Heaven", "color": "Deep Royal Purple (#2D1B4E) to Navy", "accent": "Gold (#D4AF37)", "nature": "sky, sun, metal", "keywords": ["creative", "leadership", "father"]},
    "兌": {"bin": [1, 1, 0], "name": "Lake", "color": "Sky Blue (#87CEEB)", "accent": "Soft Pink", "nature": "marsh, lake, reflection", "keywords": ["joy", "openness", "youngest daughter"]},
    "離": {"bin": [1, 0, 1], "name": "Fire", "color": "Vermillion Red (#E34234) to Orange", "accent": "Bright Yellow", "nature": "fire, sun, lightning", "keywords": ["clarity", "passion", "middle daughter"]},
    "震": {"bin": [1, 0, 0], "name": "Thunder", "color": "Deep Forest Green (#004D40)", "accent": "Purple", "nature": "thunder, sprout", "keywords": ["arousing", "movement", "eldest son"]},
    "巽": {"bin": [0, 1, 1], "name": "Wind", "color": "Emerald Green (#50C878)", "accent": "White", "nature": "wind, wood", "keywords": ["gentle", "penetrating", "eldest daughter"]},
    "坎": {"bin": [0, 1, 0], "name": "Water", "color": "Deep Navy (#000080)", "accent": "Silver", "nature": "water, rain, moon", "keywords": ["abysmal", "danger", "middle son"]},
    "艮": {"bin": [0, 0, 1], "name": "Mountain", "color": "Earth Brown (#8B4513)", "accent": "Forest Green", "nature": "mountain, stone", "keywords": ["stillness", "stopping", "youngest son"]},
    "坤": {"bin": [0, 0, 0], "name": "Earth", "color": "Warm Terracotta (#CD853F)", "accent": "Wheat/Beige", "nature": "earth, soil, field", "keywords": ["receptive", "nurturing", "mother"]}
}

# Line Graphics
YANG_LINE = "━━━━━━ (yang/solid)"
YIN_LINE  = "━━━ ━━━ (yin/broken)"

HEADER = """# 64卦アイコン画像生成プロンプト v2

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
"""

def generate_lines_display(hex_data):
    upper_name = hex_data['upper']
    lower_name = hex_data['lower']
    
    upper_bins = TRIGRAMS[upper_name]['bin'] # [bottom, mid, top] of trigram
    lower_bins = TRIGRAMS[lower_name]['bin'] # [bottom, mid, top] of trigram
    
    # Combined hexagram lines from bottom (Line 1) to top (Line 6)
    # Lower trigram is lines 1,2,3. Upper trigram is lines 4,5,6.
    all_lines = lower_bins + upper_bins
    
    lines_str = "YAO LINES (6 lines, from bottom to top):\n"
    line_names = ["初爻", "二爻", "三爻", "四爻", "五爻", "上爻"]
    
    for i, val in enumerate(all_lines):
        line_num = i + 1
        # V3 Format: Line X (Name): XXXXXX (yang/yin)
        # 1=Yang, 0=Yin
        graphic = "━━━━━━" if val == 1 else "━━━ ━━━"
        label = "(yang)" if val == 1 else "(yin)"
        
        lines_str += f"Line {line_num} ({line_names[i]}): {graphic} {label}\n"
    
    return lines_str

def generate_visual_verification(all_lines):
    verif_str = "VISUAL VERIFICATION (must match this exactly):\n"
    for i, val in enumerate(all_lines):
        line_num = i + 1
        graphic = "━━━━━━━━" if val == 1 else "━━━ ━━━"
        label = "(SOLID/yang)" if val == 1 else "(BROKEN/yin)"
        verif_str += f"    {line_num}: {graphic} {label}\n"
    return verif_str

def main():
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    hexagrams = data['hexagrams']
    # Sort by number to ensure 1-64 order
    sorted_hex_keys = sorted(hexagrams.keys(), key=lambda k: hexagrams[k]['number'])
    
    output_md = HEADER
    
    for key in sorted_hex_keys:
        h = hexagrams[key]
        num = h['number']
        name = key
        upper = h['upper']
        lower = h['lower']
        meaning = h['meaning']
        image_text = h['image']
        
        # Get Trigram info
        t_upper = TRIGRAMS[upper]
        t_lower = TRIGRAMS[lower]
        
        # Color strategy
        if upper == lower:
            bg_color = f"{t_upper['color']} Gradient"
        else:
            bg_color = f"Gradient blending {t_upper['name']} ({t_upper['color']}) and {t_lower['name']} ({t_lower['color']})"
        
        # Symbol Description
        symbol_desc = f"{t_upper['name']} over {t_lower['name']}. {meaning}. {t_upper['nature']} and {t_lower['nature']} imagery."
        
        # V4 Redundancy Data
        upper_bins = t_upper['bin']
        lower_bins = t_lower['bin']
        all_lines = lower_bins + upper_bins
        binary_str = "".join(map(str, all_lines))
        
        trigrams_desc = f"Upper={t_upper['name']}({upper}) + Lower={t_lower['name']}({lower})"
        visual_verification = generate_visual_verification(all_lines)
        
        common_mistakes = """⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)"""

        prompt_block = f"""
### {num}. {name}

```
Create icon for Hexagram {num}: {name} ({h['chinese_name']} - {meaning})

SYMBOL: {symbol_desc}
Visual focus: {image_text}

BINARY: {binary_str}
TRIGRAMS: {trigrams_desc}

{visual_verification}
{generate_lines_display(h)}
{common_mistakes}

DISPLAY NAME: {name}

BACKGROUND: {bg_color}
ACCENT: {t_upper['accent']}
SYMBOL STYLE: Modern, minimal, conveying "{meaning}"
Keywords: {', '.join(h['keywords'])}
```

---
"""
        output_md += prompt_block
        
    # Write to file
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(output_md)
    
    print(f"Generated {OUTPUT_PATH} with {len(sorted_hex_keys)} hexagrams.")

if __name__ == "__main__":
    main()
