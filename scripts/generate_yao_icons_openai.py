import os
import sys
import json
import base64
import urllib.request
import urllib.error
from datetime import datetime

# You can set your API key here or passed as an env var
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def generate_image_dalle3(prompt, output_path):
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set.")
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

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 generate_yao_icons_openai.py <hex_number> [output_dir]")
        sys.exit(1)
        
    target_hex = int(sys.argv[1])
    # Default to new standard path if not provided
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "images/yao_icons"
    
    # Load Data
    hex_json_path = "data/diagnostic/hexagram_64.json"
    yao_phrases_path = "data/yao_phrases_384.json"
    
    with open(hex_json_path, 'r') as f:
        hex_data = json.load(f)['hexagrams']
        
    with open(yao_phrases_path, 'r') as f:
        yao_phrases = json.load(f)
        
    # Find Target Hexagram
    target_info = None
    for key, val in hex_data.items():
        if val['number'] == target_hex:
            target_info = val
            target_info['name'] = key
            break
            
    if not target_info:
        print(f"Hexagram {target_hex} not found.")
        sys.exit(1)

    print(f"Starting Generation for Hexagram {target_hex}: {target_info['name']}")
    
    # Generate for lines 1 to 6
    for line_num in range(1, 7):
        yao_key = f"{target_hex}-{line_num}"
        phrase_info = yao_phrases.get(yao_key, {})
        modern_text = phrase_info.get('modern', '')
        classic_text = phrase_info.get('classic', '')
        
        # Construct Prompt
        # Note: We can't pass the "Layout Image" to DALL-E 3 as a reference in the same way.
        # So we must rely on a strong text description or use the 'edit' endpoint (DALL-E 2 only).
        # For DALL-E 3, we describe the scene vividly.
        # To strictly keep text, we might need a post-processing step to overlay text 
        # OR we ask DALL-E 3 to render the text (it is better at text now).
        
        prompt = f"""
I-Ching Hexagram Icon.
Hexagram: {target_hex} ({target_info['name']}). Line {line_num}.
Theme: {target_info['description']}
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
        
        # Create hexagram directory if it doesn't exist
        hex_dir = os.path.join(output_dir, f"hex_{target_hex:02d}")
        os.makedirs(hex_dir, exist_ok=True)
        
        output_path = os.path.join(hex_dir, output_filename)
        
        if not os.path.exists(output_path):
            generate_image_dalle3(prompt, output_path)
        else:
            print(f"Skipping {output_filename}, already exists.")

if __name__ == "__main__":
    main()
