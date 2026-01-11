import os
import shutil

BASE_DIR = "/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/images/yao_icons"
ARCHIVE_DIR = os.path.join(BASE_DIR, "_archive")

def main():
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)
    
    # Process each hex folder
    for i in range(1, 65):
        hex_dir_name = f"hex_{i:02d}"
        hex_dir_path = os.path.join(BASE_DIR, hex_dir_name)
        
        if not os.path.exists(hex_dir_path):
            print(f"Directory not found: {hex_dir_path}")
            continue
        
        files = os.listdir(hex_dir_path)
        print(f"\n=== Processing {hex_dir_name} ===")
        
        for file in files:
            file_path = os.path.join(hex_dir_path, file)
            
            # 1. Remove prompts_for_manual_generation.txt
            if file == "prompts_for_manual_generation.txt":
                os.remove(file_path)
                print(f"  Deleted: {file}")
                continue
            
            # 2. Rename files without _icon suffix to have _icon suffix
            # Target: hexagram_XX_line_Y.png -> hexagram_XX_line_Y_icon.png
            if file.endswith(".png") and "_icon" not in file:
                # Check if it's in hex_XX_line_Y format (old naming)
                if file.startswith(f"hexagram_{i}_line_"):
                    new_name = file.replace(".png", "_icon.png")
                    new_path = os.path.join(hex_dir_path, new_name)
                    if not os.path.exists(new_path):
                        os.rename(file_path, new_path)
                        print(f"  Renamed: {file} -> {new_name}")
                    else:
                        # Duplicate exists, remove the one without _icon
                        os.remove(file_path)
                        print(f"  Removed duplicate: {file}")
                # Old format like hex_27_line_5_deviating_path_jp_xxx.png - rename to standard
                elif file.startswith(f"hex_{i:02d}_line_") or file.startswith(f"hex_{i}_line_"):
                    # Extract line number
                    parts = file.split("_")
                    if len(parts) >= 4:
                        line_num = parts[3]  # hex_XX_line_Y_...
                        new_name = f"hexagram_{i}_line_{line_num}_icon.png"
                        new_path = os.path.join(hex_dir_path, new_name)
                        if not os.path.exists(new_path):
                            os.rename(file_path, new_path)
                            print(f"  Renamed: {file} -> {new_name}")
                        else:
                            os.remove(file_path)
                            print(f"  Removed duplicate: {file}")
            
            # 3. Handle JPG files in hex_32 - rename to .png and add _icon
            if file.endswith(".jpg"):
                if "_icon" in file:
                    new_name = file.replace(".jpg", ".png")
                else:
                    new_name = file.replace(".jpg", "_icon.png")
                new_path = os.path.join(hex_dir_path, new_name)
                os.rename(file_path, new_path)
                print(f"  Renamed JPG: {file} -> {new_name}")

if __name__ == "__main__":
    main()
