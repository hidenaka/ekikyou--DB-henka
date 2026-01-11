import json
import os
import sys
from PIL import Image, ImageDraw, ImageFont

# Correct Japanese Font on macOS
FONT_PATH = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc" 
# Fallback if specific weight not found, though W6 is standard
if not os.path.exists(FONT_PATH):
    FONT_PATH = "/System/Library/Fonts/Hiragino Sans GB.ttc"

# Dimensions
IMG_SIZE = 512
CENTER_X = IMG_SIZE // 2

# Layout Config
LINES_START_Y = 320  # Start drawing lines from this Y pixel
LINE_HEIGHT = 12     # Thickness of the line
LINE_GAP = 12        # Vertical gap between lines
LINE_WIDTH = 260     # Total width of the line visual
YIN_GAP = 40         # Gap in the middle of a Yin line

TEXT_Y = 460         # Y position for the Hexagram Name
TEXT_SIZE = 48       # Font size

# Color Palette (Simple Map for now, can be expanded)
# We will use a "Gold" or "White" for lines depending on contrast.
# For now, default to Gold/White with a subtle shadow/outline if needed.
LINE_COLOR = (255, 255, 255, 230) # White-ish
TEXT_COLOR = (255, 255, 255, 255)

# Trigram Binary Data (Bottom to Top)
# 1 = Yang, 0 = Yin
TRIGRAMS = {
    "乾": [1, 1, 1], "兌": [1, 1, 0], "離": [1, 0, 1], "震": [1, 0, 0],
    "巽": [0, 1, 1], "坎": [0, 1, 0], "艮": [0, 0, 1], "坤": [0, 0, 0]
}

def get_hexagram_lines(hex_data):
    upper = TRIGRAMS[hex_data['upper']]
    lower = TRIGRAMS[hex_data['lower']]
    # Combined: Lower (lines 1-3) then Upper (lines 4-6)
    return lower + upper

def create_base_layout(output_path, hex_data):
    """
    Creates a clean base layout image with just:
    1. Background color (based on Upper Trigram)
    2. The 6 Lines (Yao) correctly drawn
    3. The Hexagram Name
    
    This serves as the 'structure reference' for Image-to-Image generation.
    """
    # Determine background color based on Upper Trigram
    upper_name = hex_data['upper']
    # Default colors mapping
    COLOR_MAP = {
        "乾": "#2D1B4E", # Heaven - Deep Purple
        "兌": "#87CEEB", # Lake - Sky Blue
        "離": "#E34234", # Fire - Red
        "震": "#000080", # Thunder - Navy
        "巽": "#50C878", # Wind - Green
        "坎": "#000080", # Water - Navy
        "艮": "#8B4513", # Mountain - Brown
        "坤": "#CD853F"  # Earth - Terracotta
    }
    bg_color = COLOR_MAP.get(upper_name, "#2D1B4E")
    
    # Create base image
    img = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw Lines
    lines = get_hexagram_lines(hex_data) # [L1...L6]
    start_y_for_line_6 = 310
    
    # Line color: Gold/Yellow for high contrast reference
    layout_line_color = (255, 215, 0, 255) # Gold
    
    for i in range(6):
        # Top (Line 6) to Bottom (Line 1)
        line_idx = 5 - i 
        is_yang = lines[line_idx] == 1
        current_y = start_y_for_line_6 + (i * (LINE_HEIGHT + LINE_GAP))
        
        left_x = CENTER_X - (LINE_WIDTH // 2)
        right_x = CENTER_X + (LINE_WIDTH // 2)
        
        if is_yang:
            draw.rectangle([left_x, current_y, right_x, current_y + LINE_HEIGHT], fill=layout_line_color)
        else:
            mid_left = CENTER_X - (YIN_GAP // 2)
            mid_right = CENTER_X + (YIN_GAP // 2)
            draw.rectangle([left_x, current_y, mid_left, current_y + LINE_HEIGHT], fill=layout_line_color)
            draw.rectangle([mid_right, current_y, right_x, current_y + LINE_HEIGHT], fill=layout_line_color)

    # Draw Text
    try:
        font = ImageFont.truetype(FONT_PATH, TEXT_SIZE)
        text = hex_data['name']
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = CENTER_X - (text_w // 2)
        draw.text((text_x, TEXT_Y), text, font=font, fill=(255, 255, 255, 255))
    except Exception as e:
        print(f"Font Error: {e}")
        
    img.save(output_path)
    print(f"Created layout: {output_path}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <json_path> <output_dir> [--layout-only]")
        return

    json_path = sys.argv[1]
    output_dir = sys.argv[2]
    mode = "fix"
    if len(sys.argv) > 3 and sys.argv[3] == "--layout-only":
        mode = "layout"
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    if mode == "layout":
        # Generate base layouts for ALL 64 hexagrams
        for key, val in data['hexagrams'].items():
            num = val['number']
            val['name'] = key
            output_filename = f"hexagram_{num:02d}_{key}_layout.png"
            output_path = os.path.join(output_dir, output_filename)
            create_base_layout(output_path, val)
    else:
        # Fix existing images
        for filename in os.listdir(output_dir):
            if not filename.endswith(".png") or "fixed" in filename or "layout" in filename:
                continue
            if not filename.startswith("hexagram_"):
                continue
                
            parts = filename.split('_')
            try:
                num = int(parts[1])
                hex_key = None
                for key, val in data['hexagrams'].items():
                    if val['number'] == num:
                        hex_key = key
                        target_data = val
                        target_data['name'] = key
                        break
                
                if hex_key:
                    print(f"Processing Hexagram {num}: {hex_key}")
                    input_path = os.path.join(output_dir, filename)
                    output_path = input_path.replace(".png", "_fixed.png")
                    draw_hexagram_overlay(input_path, output_path, target_data)
                    
            except Exception as e:
                print(f"Skipping {filename}: {e}")

if __name__ == "__main__":
    main()
