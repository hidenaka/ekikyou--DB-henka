import json
import os

# Configuration
OUTPUT_BASE_DIR = "images/yao_icons"
HEX_DATA_PATH = "data/diagnostic/hexagram_64.json"
YAO_PHRASES_PATH = "data/yao_phrases_384.json"

# Color Map (Upper Trigram determines Background)
# Based on scripts/compose_yao_icons.py
COLOR_MAP = {
    "乾": {"name": "Midnight Blue", "code": "#2D1B4E"},
    "兌": {"name": "Sky Blue", "code": "#87CEEB"},
    "離": {"name": "Vermilion", "code": "#E34234"},
    "震": {"name": "Deep Forest Green", "code": "#004D40"},
    "巽": {"name": "Emerald Green", "code": "#50C878"},
    "坎": {"name": "Navy Blue", "code": "#000080"},
    "艮": {"name": "Saddle Brown", "code": "#8B4513"},
    "坤": {"name": "Earth Brown", "code": "#CD853F"}
}

PROMPT_TEMPLATE = """Generate a flat design square icon (1:1 aspect ratio) for Hexagram {hex_num} Line {line_num}.

REFERENCE IMAGES:
1. First image is the LAYOUT. Use this EXACTLY for the text and hexagram lines. Do NOT change the text "{hex_name} {line_kanji_pos}" or "{modern_text}".

STYLE: Minimalist vector icon, NOT realistic. Flat design.
BACKGROUND: Solid {bg_color_name} ({bg_color_code}).

SUBJECT: "{modern_text}" ({classic_text}).
- A simple illustration representing "{modern_text}".
- Colors: {bg_color_name}, White, Cream.

LAYOUT:
- Keep the layout from Image 1.
- Draw the illustration in the top square area.
- Hexagram lines: {line_pos_english} line from bottom is YELLOW (Active), others White.

IMPORTANT: Strictly enforce the {bg_color_name} background.
"""

POS_NAMES = {1: "初", 2: "二", 3: "三", 4: "四", 5: "五", 6: "上"}
VAL_NAMES = {1: "九", 0: "六"}
LINE_POS_ENGLISH = {1: "Bottom", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "Top"}

def get_line_kanji_pos(line_num, line_val):
    p_name = POS_NAMES[line_num]
    v_name = VAL_NAMES[line_val]
    if line_num == 1:
        return f"{p_name}{v_name}"
    elif line_num == 6:
        return f"{p_name}{v_name}"
    else:
        return f"{v_name}{p_name}"

def get_hexagram_lines(hex_data):
    # Trigram logic from compose_yao_icons.py
    TRIGRAMS = {
        "乾": [1, 1, 1], "兌": [1, 1, 0], "離": [1, 0, 1], "震": [1, 0, 0],
        "巽": [0, 1, 1], "坎": [0, 1, 0], "艮": [0, 0, 1], "坤": [0, 0, 0]
    }
    upper = TRIGRAMS[hex_data['upper']]
    lower = TRIGRAMS[hex_data['lower']]
    return lower + upper

def main():
    # Load Data
    with open(HEX_DATA_PATH, 'r', encoding='utf-8') as f:
        hex_data_full = json.load(f)
    
    with open(YAO_PHRASES_PATH, 'r', encoding='utf-8') as f:
        yao_phrases = json.load(f)

    # Process Hexagrams 10 to 64
    for key, hex_info in hex_data_full['hexagrams'].items():
        hex_num = hex_info['number']
        
        if hex_num < 10 or hex_num > 64:
            continue
            
        print(f"Processing Hexagram {hex_num}: {key}")
        
        # Directory
        hex_dir = os.path.join(OUTPUT_BASE_DIR, f"hex_{hex_num:02d}")
        os.makedirs(hex_dir, exist_ok=True)
        
        # Background Color
        upper_trigram = hex_info['upper']
        bg_info = COLOR_MAP.get(upper_trigram, {"name": "Dark Blue", "code": "#2D1B4E"})
        
        # Lines
        lines = get_hexagram_lines(hex_info)
        
        for i in range(6):
            line_num = i + 1
            line_val = lines[i]
            
            # Phrase Data
            phrase_key = f"{hex_num}-{line_num}"
            phrase_data = yao_phrases.get(phrase_key, {})
            classic_text = phrase_data.get('classic', "")
            modern_text = phrase_data.get('modern', "")
            
            if not modern_text:
                modern_text = classic_text # Fallback
            
            # Formatting
            line_kanji_pos = get_line_kanji_pos(line_num, line_val)
            
            prompt_text = PROMPT_TEMPLATE.format(
                hex_num=hex_num,
                line_num=line_num,
                hex_name=key,
                line_kanji_pos=line_kanji_pos,
                modern_text=modern_text,
                classic_text=classic_text,
                bg_color_name=bg_info['name'],
                bg_color_code=bg_info['code'],
                line_pos_english=LINE_POS_ENGLISH[line_num]
            )
            
            # Save File
            filename = f"hex_{hex_num:02d}_line_{line_num}_prompt.txt"
            filepath = os.path.join(hex_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as out:
                out.write(prompt_text)
                
    print("Done generating prompts for Hexagrams 10-64.")

if __name__ == "__main__":
    main()
