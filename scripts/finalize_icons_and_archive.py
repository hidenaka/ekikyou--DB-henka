import os
import shutil

BASE_DIR = "/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/images/yao_icons"
ARCHIVE_DIR = os.path.join(BASE_DIR, "_archive")

def main():
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)
        print(f"Created archive directory: {ARCHIVE_DIR}")

    # 1. Rename icons for Hexagrams 51-64
    print("--- Renaming Hexagram 51-64 Icons ---")
    for i in range(51, 65):
        hex_dir_name = f"hex_{i:02d}"
        hex_dir_path = os.path.join(BASE_DIR, hex_dir_name)
        
        if not os.path.exists(hex_dir_path):
            print(f"Directory not found: {hex_dir_path}")
            continue

        for line_num in range(1, 7):
            # Target file pattern: hexagram_{n}_line_{m}.png
            # Desired pattern: hexagram_{n}_line_{m}_icon.png
            
            old_filename = f"hexagram_{i}_{'line_' + str(line_num)}.png" # Based on user input "hexagram_51_line_1.png"
            new_filename = f"hexagram_{i}_{'line_' + str(line_num)}_icon.png"
            
            old_path = os.path.join(hex_dir_path, old_filename)
            new_path = os.path.join(hex_dir_path, new_filename)
            
            # Also handle cases where the file might be named slightly differently if my assumption is wrong
            # The prompt says "hexagram_51_line_1.png"
            
            if os.path.exists(old_path):
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_filename} -> {new_filename}")
                else:
                    print(f"Skipped (Target exists): {new_filename}")
            else:
                # Check directly for "line_{n}.png" just in case, but rely on user specific "hexagram_51..." first
                pass

    # 2. Archive prompts and layouts for ALL Hexagrams (01-64)
    print("\n--- Archiving Prompts and Layouts ---")
    
    # We'll scan all directories in BASE_DIR that start with "hex_"
    all_items = os.listdir(BASE_DIR)
    hex_dirs = [d for d in all_items if d.startswith("hex_") and os.path.isdir(os.path.join(BASE_DIR, d))]
    
    for hex_dir in hex_dirs:
        hex_dir_path = os.path.join(BASE_DIR, hex_dir)
        files = os.listdir(hex_dir_path)
        
        for file in files:
            # Criteria for archiving:
            # 1. Ends with "_prompt.txt"
            # 2. Ends with "_layout.png"
            # 3. is "prompts_for_manual_generation.txt"
            
            should_archive = False
            if file.endswith("_prompt.txt"):
                should_archive = True
            elif file.endswith("_layout.png"):
                should_archive = True
            elif file == "prompts_for_manual_generation.txt":
                should_archive = True
                
            if should_archive:
                src_path = os.path.join(hex_dir_path, file)
                dst_path = os.path.join(ARCHIVE_DIR, file)
                
                # Handle duplicate filenames in archive (though unlikely with hex naming)
                if os.path.exists(dst_path):
                    print(f"Warning: File already exists in archive: {file}")
                    # Simple conflict resolution: append timestamp or skip? 
                    # For now, let's just overwrite or skip? Overwriting is risky. 
                    # Let's skip and notify.
                    continue
                    
                shutil.move(src_path, dst_path)
                print(f"Archived: {file} from {hex_dir}")

if __name__ == "__main__":
    main()
