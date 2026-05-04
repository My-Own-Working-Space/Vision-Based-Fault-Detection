import pandas as pd
import os
import shutil

# 1. Paths configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.normpath(os.path.join(current_dir, "../dataset/train"))
CSV_FILE = os.path.join(BASE_DIR, "_classes.csv")
CLASSES = ['Clean-Insulator', 'Dirt-Insulator']

print(f"--- Checking path: {BASE_DIR} ---")

# 2. Read CSV
if not os.path.exists(CSV_FILE):
    print(f"ERROR: CSV file not found at {CSV_FILE}")
else:
    try:
        print(f"Reading file: {CSV_FILE}")
        df = pd.read_csv(CSV_FILE)
        df.columns = df.columns.str.strip()
        
        # 3. Create class folders
        for cls in CLASSES:
            path = os.path.join(BASE_DIR, cls)
            os.makedirs(path, exist_ok=True)
            print(f"Created/Checked directory: {path}")

        # 4. Move images
        print("Starting to classify images...")
        count = 0
        for index, row in df.iterrows():
            file_name = row['filename'].strip()
            src_path = os.path.join(BASE_DIR, file_name)
            
            for cls in CLASSES:
                if row[cls] == 1:
                    dest_path = os.path.join(BASE_DIR, cls, file_name)
                    if os.path.exists(src_path):
                        shutil.move(src_path, dest_path)
                        count += 1
                    break 

        print(f"Done! Moved {count} images to their respective folders.")
    except Exception as e:
        print(f"Error during processing: {e}")