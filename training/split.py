import pandas as pd
import os
import shutil

# 1. Cấu hình đường dẫn
BASE_DIR = "../dataset/train" # Đường dẫn đến thư mục train hiện tại
CSV_FILE = os.path.join(BASE_DIR, "_classes.csv")
CLASSES = ['Clean-Insulator', 'Dirt-Insulator'] # Tên các cột nhãn trong CSV

# 2. Đọc tệp CSV
df = pd.read_csv(CSV_FILE)
# Loại bỏ khoảng trắng thừa ở tên cột nếu có
df.columns = df.columns.str.strip()

# 3. Tạo các thư mục nhãn
for cls in CLASSES:
    os.makedirs(os.path.join(BASE_DIR, cls), exist_ok=True)

# 4. Duyệt qua từng hàng trong CSV để di chuyển ảnh
print("Đang bắt đầu phân loại ảnh...")
for index, row in df.iterrows():
    file_name = row['filename'].strip()
    src_path = os.path.join(BASE_DIR, file_name)
    
    # Kiểm tra xem ảnh thuộc nhãn nào (giá trị 1)
    for cls in CLASSES:
        if row[cls] == 1:
            dest_path = os.path.join(BASE_DIR, cls, file_name)
            if os.path.exists(src_path):
                shutil.move(src_path, dest_path)
            break 

print("Hoàn thành! Bây giờ bạn có thể dùng datasets.ImageFolder.")