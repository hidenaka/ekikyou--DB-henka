import json
import os
import sys
from PIL import Image, ImageDraw, ImageFont

# Correct Japanese Font on macOS
FONT_PATH = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc" 
if not os.path.exists(FONT_PATH):
    FONT_PATH = "/System/Library/Fonts/Hiragino Sans GB.ttc"

# Dimensions
IMG_SIZE = 512
CENTER_X = IMG_SIZE // 2

# Layout Config - Lines
LINES_START_Y = 280  # Moved up slightly to make room for 2 lines of text
LINE_HEIGHT = 12
LINE_GAP = 12
LINE_WIDTH = 260
YIN_GAP = 40

# Layout Config - Text
TEXT_MAIN_Y = 430
TEXT_SUB_Y = 475
TEXT_MAIN_SIZE = 42
TEXT_SUB_SIZE = 28

# Trigram Binary Data (Bottom to Top)
TRIGRAMS = {
    "乾": [1, 1, 1], "兌": [1, 1, 0], "離": [1, 0, 1], "震": [1, 0, 0],
    "巽": [0, 1, 1], "坎": [0, 1, 0], "艮": [0, 0, 1], "坤": [0, 0, 0]
}

def get_hexagram_lines(hex_data):
    upper = TRIGRAMS[hex_data['upper']]
    lower = TRIGRAMS[hex_data['lower']]
    return lower + upper

def create_yao_layout(output_path, hex_data, active_line, subtitle_text):
    """
    Creates a layout highlighting a specific line (1-6).
    active_line: 1-based index (1=Bottom, 6=Top)
    subtitle_text: The classic text (e.g. "潜龍勿用")
    """
    upper_name = hex_data['upper']
    
    # Standard Colors
    COLOR_MAP = {
        "乾": "#2D1B4E", "兌": "#87CEEB", "離": "#E34234", "震": "#004D40",
        "巽": "#50C878", "坎": "#000080", "艮": "#8B4513", "坤": "#CD853F"
    }
    bg_color = COLOR_MAP.get(upper_name, "#2D1B4E")
    
    img = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), bg_color)
    draw = ImageDraw.Draw(img)
    
    lines = get_hexagram_lines(hex_data) # [L1, L2, L3, L4, L5, L6]
    
    # Draw Lines (Top to Bottom visual)
    start_y_for_line_6 = LINES_START_Y
    
    HIGHLIGHT_COLOR = (255, 215, 0, 255) # Gold
    DIM_COLOR = (255, 255, 255, 60)     # Dimmed White
    
    for i in range(6):
        # i=0 -> Top Line (Line 6)
        current_line_num = 6 - i 
        line_idx = current_line_num - 1
        
        if current_line_num == active_line:
            fill_color = HIGHLIGHT_COLOR
        else:
            fill_color = DIM_COLOR
            
        is_yang = lines[line_idx] == 1
        current_y = start_y_for_line_6 + (i * (LINE_HEIGHT + LINE_GAP))
        
        left_x = CENTER_X - (LINE_WIDTH // 2)
        right_x = CENTER_X + (LINE_WIDTH // 2)
        
        if is_yang:
            draw.rectangle([left_x, current_y, right_x, current_y + LINE_HEIGHT], fill=fill_color)
        else:
            mid_left = CENTER_X - (YIN_GAP // 2)
            mid_right = CENTER_X + (YIN_GAP // 2)
            draw.rectangle([left_x, current_y, mid_left, current_y + LINE_HEIGHT], fill=fill_color)
            draw.rectangle([mid_right, current_y, right_x, current_y + LINE_HEIGHT], fill=fill_color)

    # Draw Text
    try:
        font_main = ImageFont.truetype(FONT_PATH, TEXT_MAIN_SIZE)
        font_sub  = ImageFont.truetype(FONT_PATH, TEXT_SUB_SIZE)
        
        # 1. Main Title: "乾為天 初九"
        line_val = lines[active_line - 1]
        pos_names = {1: "初", 2: "二", 3: "三", 4: "四", 5: "五", 6: "上"}
        val_names = {1: "九", 0: "六"}
        
        p_name = pos_names[active_line]
        v_name = val_names[line_val]
        
        if active_line == 1:
            line_text = f"{p_name}{v_name}"
        elif active_line == 6:
            line_text = f"{p_name}{v_name}"
        else:
            line_text = f"{v_name}{p_name}"
            
        main_text = f"{hex_data['name']} {line_text}"
        
        bbox_main = draw.textbbox((0, 0), main_text, font=font_main)
        w_main = bbox_main[2] - bbox_main[0]
        x_main = CENTER_X - (w_main // 2)
        draw.text((x_main, TEXT_MAIN_Y), main_text, font=font_main, fill=(255, 255, 255, 255))
        
        # 2. Subtitle: Classic Text from CSV / Manual Map
        if subtitle_text:
            # Auto-scale text to fit
            max_w = IMG_SIZE - 40 # 20px padding on each side
            current_font_size = TEXT_SUB_SIZE
            font_sub = ImageFont.truetype(FONT_PATH, current_font_size)
            
            bbox_sub = draw.textbbox((0, 0), subtitle_text, font=font_sub)
            w_sub = bbox_sub[2] - bbox_sub[0]
            
            while w_sub > max_w and current_font_size > 10:
                current_font_size -= 2
                font_sub = ImageFont.truetype(FONT_PATH, current_font_size)
                bbox_sub = draw.textbbox((0, 0), subtitle_text, font=font_sub)
                w_sub = bbox_sub[2] - bbox_sub[0]
            
            x_sub = CENTER_X - (w_sub // 2)
            # Use Gold/Yellow for the description to make it pop? Or White?
            # Let's use a soft Yellow to link with the highlighted line.
            draw.text((x_sub, TEXT_SUB_Y), subtitle_text, font=font_sub, fill=(255, 223, 128, 255))
            
    except Exception as e:
        print(f"Font Error: {e}")
        
    img.save(output_path)
    print(f"Created layout: {output_path}")

def main():
    if len(sys.argv) < 4:
        print("Usage: python compose_yao_icons.py <hex_json> <yao_json> <output_dir> --hex <num>")
        return

    hex_json_path = sys.argv[1]
    csv_path = sys.argv[2]
    output_dir = sys.argv[3]
    
    target_hex = None
    if "--hex" in sys.argv:
        idx = sys.argv.index("--hex")
        target_hex = int(sys.argv[idx+1])
    
    with open(hex_json_path, 'r') as f:
        hex_data = json.load(f)
        
    # Read CSV for Classic Text
    yao_texts = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 4: continue
                
                row_id = parts[0]
                row_type = parts[1]
                item_name = parts[3]
                
                if row_type == "六爻" and row_id.startswith("line_"):
                    try:
                        id_parts = row_id.split('_')
                        h_num = int(id_parts[1])
                        l_num = int(id_parts[2])
                        
                        if "：" in item_name:
                            text_content = item_name.split("：", 1)[1]
                        elif ":" in item_name:
                             text_content = item_name.split(":", 1)[1]
                        else:
                            text_content = item_name
                            
                        # Clean quotes
                        text_content = text_content.replace('"', '')
                        
                        yao_key = f"{h_num}-{l_num}"
                        yao_texts[yao_key] = text_content
                        
                    except Exception:
                        pass
    except Exception as e:
        print(f"Error reading CSV: {e}")

    # Load Modern Translations
    translations_path = "data/yao_phrases_384.json"
    modern_phrases = {}
    if os.path.exists(translations_path):
        with open(translations_path, 'r', encoding='utf-8') as f:
            modern_phrases = json.load(f)

    if target_hex:
        # Generate 6 lines for this hexagram
        for key, val in hex_data['hexagrams'].items():
            if val['number'] == target_hex:
                print(f"Generating Yao layouts for Hexagram {target_hex}: {key}")
                val['name'] = key
                
                for line_num in range(1, 7):
                     yao_key = f"{target_hex}-{line_num}" # Note: key format in json is "1-1" (int-int) match logic below
                     
                     # Check if we have a direct modern translation
                     # The json keys are strings like "1-1", "1-2"
                     # Our h_num is int, l_num is int.
                     
                     lookup_key = f"{target_hex}-{line_num}"
                     subtitle = ""
                     
                     if lookup_key in modern_phrases:
                         subtitle = modern_phrases[lookup_key].get('modern', "")
                     
                     # Fallback to classic text if no modern phrase found (yet)
                     if not subtitle:
                        subtitle = yao_texts.get(lookup_key, "")
                     
                     # Create hexagram directory if it doesn't exist
                     hex_dir = os.path.join(output_dir, f"hex_{target_hex:02d}")
                     os.makedirs(hex_dir, exist_ok=True)
                     
                     output_filename = f"hexagram_{target_hex:02d}_line_{line_num}_layout.png"
                     output_path = os.path.join(hex_dir, output_filename)
                     create_yao_layout(output_path, val, line_num, subtitle)
                break
    else:
        print("Please specify --hex <number>")

if __name__ == "__main__":
    main()
