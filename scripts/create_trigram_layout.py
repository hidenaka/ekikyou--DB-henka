import os
import sys
from PIL import Image, ImageDraw, ImageFont

# Font configuration
FONT_PATH = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "/System/Library/Fonts/Hiragino Sans GB.ttc"

# Dimensions
IMG_SIZE = 512
CENTER_X = IMG_SIZE // 2

# Layout Config - Adapted for Trigram Icon
LINES_START_Y = 380       # Moved down slightly
LINE_HEIGHT = 18          # Slightly thinner lines
LINE_GAP = 14
LINE_WIDTH = 220          # Slightly narrower lines
YIN_GAP = 30

MAIN_TEXT_Y = 60          # Moved up
SUB_TEXT_Y = 160          # Moved up
MAIN_TEXT_SIZE = 100      # Smaller, more elegant
SUB_TEXT_SIZE = 40        # Distinctly smaller (subtitle)

# Trigram Data (Bottom to Top)
# 1 = Yang, 0 = Yin
TRIGRAMS = {
    "乾": {"lines": [1, 1, 1], "nature": "天", "color": "#2D1B4E", "text_color": "#FFFFFF"}, # Heaven - Deep Purple
    "兌": {"lines": [1, 1, 0], "nature": "沢", "color": "#87CEEB", "text_color": "#FFFFFF"}, # Lake - Sky Blue
    "離": {"lines": [1, 0, 1], "nature": "火", "color": "#E34234", "text_color": "#FFFFFF"}, # Fire - Red
    "震": {"lines": [1, 0, 0], "nature": "雷", "color": "#000080", "text_color": "#FFFFFF"}, # Thunder - Navy
    "巽": {"lines": [0, 1, 1], "nature": "風", "color": "#50C878", "text_color": "#FFFFFF"}, # Wind - Green
    "坎": {"lines": [0, 1, 0], "nature": "水", "color": "#000080", "text_color": "#FFFFFF"}, # Water - Navy
    "艮": {"lines": [0, 0, 1], "nature": "山", "color": "#8B4513", "text_color": "#FFFFFF"}, # Mountain - Brown
    "坤": {"lines": [0, 0, 0], "nature": "地", "color": "#CD853F", "text_color": "#FFFFFF"}  # Earth - Terracotta
}

def create_trigram_layout(name, output_path):
    data = TRIGRAMS.get(name)
    if not data:
        print(f"Unknown trigram: {name}")
        return

    img = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), data["color"])
    draw = ImageDraw.Draw(img)

    # Draw Text
    try:
        # Main Character (Trigram Name) e.g., 乾
        font_main = ImageFont.truetype(FONT_PATH, MAIN_TEXT_SIZE)
        bbox = draw.textbbox((0, 0), name, font=font_main)
        text_w = bbox[2] - bbox[0]
        text_x = CENTER_X - (text_w // 2)
        draw.text((text_x, MAIN_TEXT_Y), name, font=font_main, fill=data["text_color"])

        # Sub Character (Nature) e.g., 天
        # Draw "Nature" attribute smaller, perhaps with a subtle decorative line or distinct placement?
        # For now, just hierarchy. "Qian" is the Power, "Heaven" is the Image.
        font_sub = ImageFont.truetype(FONT_PATH, SUB_TEXT_SIZE)
        nature_text = data["nature"]
        bbox_sub = draw.textbbox((0, 0), nature_text, font=font_sub)
        text_w_sub = bbox_sub[2] - bbox_sub[0]
        text_x_sub = CENTER_X - (text_w_sub // 2)
        draw.text((text_x_sub, SUB_TEXT_Y), nature_text, font=font_sub, fill=(255, 255, 255, 200)) # Slightly transparent for hierarchy

    except Exception as e:
        print(f"Font Error: {e}")

    # Draw Lines
    lines = data["lines"] # Bottom to Top [1, 1, 1]
    # We traditionally draw from Bottom up, or Top down visually?
    # Usually standard references list Bottom line first.
    # Visually, we draw Line 3 (Top) at top y, Line 1 (Bottom) at bottom y.
    
    # Visual Order: Top (Line 3), Middle (Line 2), Bottom (Line 1)
    # lines array is [Line 1 (Bottom), Line 2, Line 3 (Top)]
    
    line_visual_order = [lines[2], lines[1], lines[0]] # Top to Bottom

    layout_line_color = (255, 215, 0, 255) # Gold

    for i, is_yang in enumerate(line_visual_order):
        current_y = LINES_START_Y + (i * (LINE_HEIGHT + LINE_GAP))
        
        left_x = CENTER_X - (LINE_WIDTH // 2)
        right_x = CENTER_X + (LINE_WIDTH // 2)
        
        if is_yang:
            draw.rectangle([left_x, current_y, right_x, current_y + LINE_HEIGHT], fill=layout_line_color)
        else:
            mid_left = CENTER_X - (YIN_GAP // 2)
            mid_right = CENTER_X + (YIN_GAP // 2)
            draw.rectangle([left_x, current_y, mid_left, current_y + LINE_HEIGHT], fill=layout_line_color)
            draw.rectangle([mid_right, current_y, right_x, current_y + LINE_HEIGHT], fill=layout_line_color)

    # Save
    img.save(output_path)
    print(f"Created layout: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_trigram_layout.py <trigram_name> <output_path>")
    else:
        create_trigram_layout(sys.argv[1], sys.argv[2])
