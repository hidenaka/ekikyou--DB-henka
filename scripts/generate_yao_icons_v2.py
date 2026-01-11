import os
import sys
import json
import base64
import urllib.request
import urllib.error
import subprocess
from datetime import datetime

# You can set your API key here or passed as an env var
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Try reading from .openai_key file if env var is not set
if not OPENAI_API_KEY:
    key_file = ".openai_key"
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            OPENAI_API_KEY = f.read().strip()

# Trigram Data (Bottom to Top binary)
TRIGRAMS = {
    "乾": {"bin": [1, 1, 1], "name": "Heaven", "color": "Deep Royal Purple (#2D1B4E) to Navy", "accent": "Gold (#D4AF37)", "nature": "sky, sun, metal"},
    "兌": {"bin": [1, 1, 0], "name": "Lake", "color": "Sky Blue (#87CEEB)", "accent": "Soft Pink", "nature": "marsh, lake, reflection"},
    "離": {"bin": [1, 0, 1], "name": "Fire", "color": "Vermillion Red (#E34234) to Orange", "accent": "Bright Yellow", "nature": "fire, sun, lightning"},
    "震": {"bin": [1, 0, 0], "name": "Thunder", "color": "Deep Forest Green (#004D40)", "accent": "Purple", "nature": "thunder, sprout"},
    "巽": {"bin": [0, 1, 1], "name": "Wind", "color": "Emerald Green (#50C878)", "accent": "White", "nature": "wind, wood"},
    "坎": {"bin": [0, 1, 0], "name": "Water", "color": "Deep Navy (#000080)", "accent": "Silver", "nature": "water, rain, moon"},
    "艮": {"bin": [0, 0, 1], "name": "Mountain", "color": "Earth Brown (#8B4513)", "accent": "Forest Green", "nature": "mountain, stone"},
    "坤": {"bin": [0, 0, 0], "name": "Earth", "color": "Warm Terracotta (#CD853F)", "accent": "Wheat/Beige", "nature": "earth, soil, field"}
}

def generate_lines_display(hex_data):
    upper_name = hex_data['upper']
    lower_name = hex_data['lower']
    
    upper_bins = TRIGRAMS[upper_name]['bin']
    lower_bins = TRIGRAMS[lower_name]['bin']
    
    all_lines = lower_bins + upper_bins
    
    lines_str = "YAO LINES (6 lines, from bottom to top):\n"
    line_names = ["初爻", "二爻", "三爻", "四爻", "五爻", "上爻"]
    
    for i, val in enumerate(all_lines):
        line_num = i + 1
        graphic = "━━━━━━" if val == 1 else "━━━ ━━━"
        label = "(yang)" if val == 1 else "(yin)"
        lines_str += f"Line {line_num} ({line_names[i]}): {graphic} {label}\n"
    
    return all_lines, lines_str

def generate_visual_verification(all_lines):
    verif_str = "VISUAL VERIFICATION (must match this exactly):\n"
    for i, val in enumerate(all_lines):
        line_num = i + 1
        graphic = "━━━━━━━━" if val == 1 else "━━━ ━━━"
        label = "(SOLID/yang)" if val == 1 else "(BROKEN/yin)"
        verif_str += f"    {line_num}: {graphic} {label}\n"
    return verif_str

def generate_image_dalle3(prompt, output_path):
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set and .openai_key not found.")
        return False

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
        "model": "dall-e-3",
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
        "response_format": "b64_json",
        "style": "vivid",
        "quality": "standard"
    }

    print(f"Generating image for: {os.path.basename(output_path)}...")
    try:
        req = urllib.request.Request(
            "https://api.openai.com/v1/images/generations",
            data=json.dumps(payload).encode('utf-8'),
            headers=headers,
            method='POST'
        )
        
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            
        b64_data = data['data'][0]['b64_json']
        
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(b64_data))
            
        print(f"Success: {output_path}")
        return True
        
    except urllib.error.HTTPError as e:
        print(f"Failed to generate: HTTP {e.code}")
        try:
            print(e.read().decode())
        except:
            pass
        return False
    except Exception as e:
        print(f"Failed to generate: {e}")
        return False

def generate_layout(target_hex, output_dir):
    """Generates the layout images using scripts/compose_yao_icons.py"""
    print(f"Generating Layouts for Hexagram {target_hex}...")
    try:
        # Using subprocess to call the existing script to reuse its logic + font handling
        cmd = [
            sys.executable, 
            "scripts/compose_yao_icons.py",
            "data/diagnostic/hexagram_64.json",
            "data/yao_phrases_384.json", # CSV argument in original script, but we don't have CSV easily. Passing json might fail parsing but let's try or skip.
            # Actually compose_yao_icons.py takes <hex_json> <yao_json/csv> <output_dir>
            # And it tries to read CSV lines.
            # Step 56 shows it reads sys.argv[2] as csv_path and tries to split by comma.
            # If we pass a dummy or valid file it might work for layout generation if it falls back to modern json.
            # It loads "data/yao_phrases_384.json" internally at line 197.
            # So the second arg is strictly for "Classic Text from CSV", which we might not need if phrases json works.
            # Let's pass a dummy path or existing file.
            output_dir,
            "--hex", str(target_hex)
        ]
        # We need to point to a valid file for arg 2 so it doesn't crash on open
        cmd[3] = "data/yao_phrases_384.json" 
        
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to generate layout: {e}")
    except Exception as e:
        print(f"Error calling layout script: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 generate_yao_icons_v2.py <hex_number> [output_dir] [start_line] [line_limit]")
        sys.exit(1)
        
    target_hex = int(sys.argv[1])
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "images/yao_icons"
    start_line = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    # Adding limit capability for testing
    limit_count = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    
    # Load Data
    hex_json_path = "data/diagnostic/hexagram_64.json"
    yao_phrases_path = "data/yao_phrases_384.json"
    
    with open(hex_json_path, 'r') as f:
        hex_data = json.load(f)['hexagrams']
        
    with open(yao_phrases_path, 'r') as f:
        yao_phrases = json.load(f)
        
    target_info = None
    for key, val in hex_data.items():
        if val['number'] == target_hex:
            target_info = val
            target_info['name'] = key
            break
            
    if not target_info:
        print(f"Hexagram {target_hex} not found.")
        sys.exit(1)

    print(f"Starting V2 Generation for Hexagram {target_hex}: {target_info['name']}")
    
    # 1. Generate Layout first (Reference)
    generate_layout(target_hex, output_dir)
    
    # Pre-calculate hexagram structure info
    upper = target_info['upper']
    lower = target_info['lower']
    t_upper = TRIGRAMS[upper]
    t_lower = TRIGRAMS[lower]
    
    if upper == lower:
        bg_color = f"{t_upper['color']} Gradient"
    else:
        bg_color = f"Gradient blending {t_upper['name']} ({t_upper['color']}) and {t_lower['name']} ({t_lower['color']})"
        
    all_lines, lines_display = generate_lines_display(target_info)
    visual_verif = generate_visual_verification(all_lines)
    
    binary_str = "".join(map(str, all_lines))
    trigrams_desc = f"Upper={t_upper['name']}({upper}) + Lower={t_lower['name']}({lower})"
    symbol_desc = f"{t_upper['name']} over {t_lower['name']}. {target_info['meaning']}. {t_upper['nature']} and {t_lower['nature']} imagery."

    common_mistakes = """⛔ COMMON MISTAKES TO AVOID:
- Drawing only 5 lines (forgetting one)
- Drawing 7 or 8 lines (adding extras)
- Confusing solid (yang) and broken (yin)
- Drawing yin as a dotted line (WRONG - it's TWO SEGMENTS with a GAP)"""

    count = 0
    # Generate for lines 1 to 6
    for line_num in range(start_line, 7):
        if count >= limit_count:
            print(f"Reached limit of {limit_count} images. Stopping.")
            break
            
        output_filename = f"hex_{target_hex:02d}_line_{line_num}_v2.png"
        hex_dir = os.path.join(output_dir, f"hex_{target_hex:02d}")
        os.makedirs(hex_dir, exist_ok=True)
        output_path = os.path.join(hex_dir, output_filename)
        
        if os.path.exists(output_path):
            print(f"Skipping {output_filename}, already exists.")
            continue

        yao_key = f"{target_hex}-{line_num}"
        phrase_info = yao_phrases.get(yao_key, {})
        modern_text = phrase_info.get('modern', '')
        classic_text = phrase_info.get('classic', '')

        prompt = f"""
You are creating a series of 64 I Ching hexagram icons.
ALL icons MUST follow this EXACT specification:

████████████████████████████████████████████████████████
██  YOU MUST DRAW EXACTLY 6 HORIZONTAL LINES.         ██
██  NOT 5, NOT 7, NOT 8. EXACTLY 6.                   ██
██  COUNT THEM BEFORE FINISHING: 1, 2, 3, 4, 5, 6     ██
████████████████████████████████████████████████████████

【CRITICAL LAYOUT REQUIREMENTS】
1. CANVAS: 512x512px (or square), circular icon with solid color background
2. STRUCTURE (top to bottom):
   - Top 60%: Symbolic illustration representing the hexagram meaning AND separate specific meaning for Line {line_num}.
   - Bottom 30%: Six horizontal lines (爻 yao lines) stacked vertically.
   - Bottom 10%: Hexagram name "{target_info['name']}" (Japanese kanji only).
3. YAO LINES (六爻):
   - Yang line (陽爻): One solid unbroken horizontal line ━━━━━━━━
   - Yin line (陰爻): Two short lines with a clear gap in center ━━━ ━━━
   - Lines are stacked from bottom (line 1) to top (line 6).
   - Line {line_num} (the active line) should be GLOWING GOLD/WHITE to highlight it.
   - Other lines should be visible but dimmer.

### Hexagram {target_hex}: {target_info['name']} ({target_info['chinese_name']} - {target_info['meaning']})

SYMBOL: {symbol_desc}
Visual focus: {target_info['image']}
Specific Line {line_num} Meaning: "{classic_text}" -> "{modern_text}".
The illustration should reflect this specific line meaning within the context of the hexagram.

BINARY: {binary_str}
TRIGRAMS: {trigrams_desc}

{visual_verif}
{lines_display}
{common_mistakes}

DISPLAY NAME: {target_info['name']}
BACKGROUND: {bg_color}
ACCENT: {t_upper['accent']}
STYLE: Modern minimalist flat design, subtle gradients, Eastern aesthetic meets contemporary app design.
Keywords: {', '.join(target_info['keywords'])}
"""
        generate_image_dalle3(prompt, output_path)
        count += 1

if __name__ == "__main__":
    main()
