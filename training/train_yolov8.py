import os
from ultralytics import YOLO
from roboflow import Roboflow

# ==========================================
# 1. Cấu hình (Học viên điền thông tin vào đây)
# ==========================================
ROBOFLOW_API_KEY = "YOUR_ROBOFLOW_API_KEY"  # <-- Điền API Key của bạn
WORKSPACE = "uav-inspection"
PROJECT_ID = "fault-detection-grid"
VERSION = 1

# ==========================================
# 2. Tải Dataset từ Roboflow
# ==========================================
def download_data():
    print("--- Đang tải dữ liệu từ Roboflow... ---")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT_ID)
    dataset = project.version(VERSION).download("yolov8")
    return dataset.location

# ==========================================
# 3. Huấn luyện mô hình YOLOv8
# ==========================================
def train_model(data_path):
    print(f"--- Bắt đầu huấn luyện YOLOv8 với dữ liệu tại: {data_path} ---")
    
    # Sử dụng mô hình nano (n) để chạy nhanh trên Laptop,
    # Hoặc mô hình medium (m) nếu có GPU mạnh.
    model = YOLO("yolov8n.pt") 

    # Các thông số huấn luyện theo yêu cầu:
    # 50 epochs, imgsz=640, batch size tự động
    results = model.train(
        data=f"{data_path}/data.yaml",
        epochs=50,
        imgsz=640,
        batch=-1,  # Tự động điều chỉnh theo bộ nhớ GPU
        name="uav_fault_detection",
        device=0    # Chạy trên GPU nếu có (nếu không có sẽ tự động chuyển sang CPU)
    )
    
    print("--- Huấn luyện hoàn tất! Weights được lưu tại: runs/detect/uav_fault_detection/ ---")
    return model

# ==========================================
# 4. Kiểm tra (Validation) & Đánh giá
# ==========================================
def validate_model(model):
    print("--- Đang đánh giá mô hình trên tập Validation... ---")
    metrics = model.val() # Tự động xuất Confusion Matrix ra folder 'runs'
    
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print("--- Kiểm tra file confusion_matrix.png trong thư mục runs để xem chi tiết độ chính xác ---")

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    try:
        # Nếu chạy trên Google Colab, bạn có thể thêm đoạn code mount Drive ở đây
        # from google.colab import drive
        # drive.mount('/content/drive')
        
        path = download_data()
        trained_model = train_model(path)
        validate_model(trained_model)
        
        # Xuất biểu đồ (nếu chạy trong notebook sẽ hiển thị hình ảnh)
        print("Done!")
        
    except Exception as e:
        print(f"Lỗi: {e}")
        print("Mẹo: Đảm bảo bạn đã cài đặt 'ultralytics' và 'roboflow' qua pip.")
