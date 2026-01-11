
import re
import os

WALKTHROUGH_PATH = "/Users/hideakimacbookair/.gemini/antigravity/brain/dae5cfd8-cf24-4bd6-a8c4-d85928f02ca7/walkthrough.md"
BASE_IMG_DIR = "/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/images/yao_icons"

def fix_paths():
    with open(WALKTHROUGH_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to find image links: ![alt](path)
    # We want to capture the path group
    pattern = r'!\[([^\]]*)\]\(([^)]+)\)'

    def replace_match(match):
        alt_text = match.group(1)
        current_path = match.group(2)

        # If already absolute, skip
        if current_path.startswith('/'):
            return match.group(0)

        # Try to determine hex number from filename
        # Format: hex_01_... or hex_10_...
        hex_match = re.match(r'hex_(\d+)_', os.path.basename(current_path))
        if not hex_match:
            print(f"Skipping {current_path}: Cannot extract hex number")
            return match.group(0)

        hex_num = int(hex_match.group(1))
        hex_dir_name = f"hex_{hex_num:02d}"
        
        # Construct expected absolute path
        abs_path = os.path.join(BASE_IMG_DIR, hex_dir_name, current_path)

        if os.path.exists(abs_path):
            print(f"Fixed: {current_path} -> {abs_path}")
            return f'![{alt_text}]({abs_path})'
        else:
            # Fallback: check if it exists in the artifact dir? 
            # Or just warn
            print(f"WARNING: File not found at expected path: {abs_path}")
            # Try searching recursively? No, structure is known.
            return match.group(0)

    new_content = re.sub(pattern, replace_match, content)

    if new_content != content:
        with open(WALKTHROUGH_PATH, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("Updated walkthrough.md")
    else:
        print("No changes made.")

if __name__ == "__main__":
    fix_paths()
