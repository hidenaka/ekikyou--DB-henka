import os
import sys
import subprocess

# Define base directory
BASE_DIR = "/Users/nakanohideaki/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB"
IMAGES_DIR = os.path.join(BASE_DIR, "images/yao_icons")
SCRIPT_PATH = os.path.join(BASE_DIR, "scripts/generate_yao_icons_openai.py")

def check_structure(hex_num):
    hex_dir = os.path.join(IMAGES_DIR, f"hex_{hex_num:02d}")
    if not os.path.exists(hex_dir):
        return set() # Nothing exists
    
    existing_lines = set()
    files = os.listdir(hex_dir)
    for line_num in range(1, 7):
        # Check for any valid artwork file for this line
        # Logic: must contain "line_{num}_", must end with ".png", must NOT contain "layout"
        found = False
        for f in files:
            if f"line_{line_num}_" in f and f.endswith(".png") and "layout" not in f:
                found = True
                break
        
        if found:
            existing_lines.add(line_num)
    
    return existing_lines

def main():
    print("Starting smart batch generation...")
    
    for hex_num in range(1, 65):
        existing_lines = check_structure(hex_num)
        
        if len(existing_lines) == 6:
            print(f"Hexagram {hex_num:02d}: All lines exist. Skipping.")
            continue
            
        print(f"Hexagram {hex_num:02d}: Missing lines. Checking individually...")
        
        # The generation script takes: hex_number [output_dir] [start_line]
        # It loops from start_line to 6.
        # This is slightly inefficient if we have holes (e.g. 1, 3 exist, missing 2).
        # But usually we generate in order.
        # To be precise, we can modify the gen script or just call it.
        # The gen script does:
        # for line_num in range(start_line, 7):
        #    ...
        #    if not os.path.exists(output_path): generate()
        #
        # But the gen script's "exists" check only looks for the NEW filename format.
        # So if we run it, it WILL generate duplicates for existing "hidden_jp" style files.
        # To avoid this, we should really modify the generation script's check logic 
        # OR we pass a "force" flag? No.
        
        # Best approach: Iterate here and call the generation function directly implies importing.
        # But let's stick to calling the script for now, but we can't easily skip arbitrary lines with the current script arguments.
        # Current script: start_line to 6.
        
        # Let's write a temporary wrapper that imports the function?
        # Creating a python script that imports generate_yao_icons_openai is cleaner.
        pass

if __name__ == "__main__":
    # We will invoke a python script that does the import and heavy lifting
    # This file IS that script.
    
    # Add scripts dir to path to allow import
    sys.path.append(os.path.join(BASE_DIR, "scripts"))
    import generate_yao_icons_openai as gen_script
    
    # Monkey patch or use the function?
    # validation: gen_script.generate_image_dalle3 exists
    # We can reconstruct the logic here.
    
    gen_script.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
    # Ensure key is loaded
    if not gen_script.OPENAI_API_KEY:
        key_file = os.path.join(BASE_DIR, ".openai_key")
        if os.path.exists(key_file):
            with open(key_file, "r") as f:
                gen_script.OPENAI_API_KEY = f.read().strip()
    
    # Load Data once
    hex_json_path = os.path.join(BASE_DIR, "data/diagnostic/hexagram_64.json")
    yao_phrases_path = os.path.join(BASE_DIR, "data/yao_phrases_384.json")
    
    with open(hex_json_path, 'r') as f:
        hex_data = json.load(f)['hexagrams']
    with open(yao_phrases_path, 'r') as f:
        yao_phrases = json.load(f)

    # Loop
    for target_hex in range(1, 65):
        existing = check_structure(target_hex)
        if len(existing) == 6:
            print(f"Hex {target_hex}: Complete.")
            continue
            
        # Find Target Info
        target_info = None
        for key, val in hex_data.items():
            if val['number'] == target_hex:
                target_info = val
                target_info['name'] = key
                break
        
        hex_dir = os.path.join(IMAGES_DIR, f"hex_{target_hex:02d}")
        os.makedirs(hex_dir, exist_ok=True)

        for line_num in range(1, 7):
            if line_num in existing:
                continue # Skip if ANY art file exists
                
            print(f"Generating: Hex {target_hex} Line {line_num}")
            
            # Logic from original script
            yao_key = f"{target_hex}-{line_num}"
            phrase_info = yao_phrases.get(yao_key, {})
            # Note: The json uses "1-1", "1-2" etc.
            # But wait, original script lines 106-108:
            # phrase_info = yao_phrases.get(yao_key, {})
            # So yes, key is f"{target_hex}-{line_num}"
            
            modern_text = phrase_info.get('modern', '')
            classic_text = phrase_info.get('classic', '')
            
            prompt = f"""
I-Ching Hexagram Icon.
Hexagram: {target_hex} ({target_info['name']}). Line {line_num}.
Theme: {target_info['meaning']}. {target_info['image']}
Specific Line Meaning: "{classic_text}" which means "{modern_text}".
Visual Subject: A mystical scene representing the meaning "{modern_text}".
Style: High-quality, spiritual, flat vector art, gold and deep navy blue color scheme.
Key Elements: 
- A central circular composition.
- Six horizontal lines in the background (Hexagram structure).
- The {line_num}th line from the bottom is Glowing Gold.
- The other lines are dim/dark.
- Text "{modern_text}" written elegantly in Japanese at the bottom.
Atmosphere: {target_info['keywords']}
"""
            output_filename = f"hex_{target_hex:02d}_line_{line_num}_dalle3.png"
            output_path = os.path.join(hex_dir, output_filename)
            
            # Call generation
            success = gen_script.generate_image_dalle3(prompt, output_path)
            if not success:
                print(f"Failed to generate {output_path}")

