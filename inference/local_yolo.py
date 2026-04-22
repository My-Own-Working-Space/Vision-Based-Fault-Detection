import os
import requests
import cv2
from ultralytics import YOLO

# ==========================================
# 1. Cấu hình
# ==========================================
# Đường dẫn tới thư mục model của bạn
MODEL_PATH = "models/best"
# URL của Dashboard ASP.NET
DASHBOARD_API = "http://localhost:5000/api/faults"

# ==========================================
# 2. Khởi tạo Model
# ==========================================
print(f"--- Đang tải mô hình YOLOv8 từ: {MODEL_PATH} ---")
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    exit(1)

def run_local_inference(image_path):
    """
    Chạy nhận diện trên một ảnh bằng mô hình local
    """
    print(f"--- Đang phân tích (Local): {image_path} ---")
    
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy file {image_path}")
        return

    # Chạy dự đoán
    # imgsz=640, conf=0.5 (ngưỡng 50%)
    results = model.predict(source=image_path, imgsz=640, conf=0.5)

    found_fault = False
    
    for result in results:
        # Lấy thông tin hộp (boxes)
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])
            
            print(f"- Phát hiện: {class_name} ({conf*100:.1f}%)")
            
            # Gửi thông tin về Dashboard (giả lập tọa độ cho test)
            send_to_dashboard(class_name, conf, image_path)
            found_fault = True

    if not found_fault:
        print("Không phát hiện đối tượng nào.")

def send_to_dashboard(fault_type, confidence, image_path):
    """
    Gửi dữ liệu về Dashboard ASP.NET Core
    """
    # Map class name sang loại lỗi tiếng Việt (nếu cần)
    fault_map = {
        "broken disc": "Sứ nứt (Local)",
        "insulator": "Sứ bình thường",
        "broken": "Lỗi Sứ",
        "rust": "Rỉ sét"
    }
    
    display_name = fault_map.get(fault_type.lower(), fault_type)
    
    payload = {
        "towerId": "T-110KV-LOCAL", # Mã trụ giả lập
        "faultType": display_name,
        "confidenceScore": float(confidence),
        "imagePath": f"/uploads/{os.path.basename(image_path)}",
        "latitude": 21.0315, # Tọa độ giả lập hơi lệch so với mẫu cũ để dễ phân biệt
        "longitude": 105.8502
    }
    
    try:
        res = requests.post(DASHBOARD_API, json=payload)
        status_icon = "✅" if res.status_code == 200 else "❌"
        print(f"{status_icon} Gửi Dashboard: {display_name} ({confidence*100:.1f}%)")
    except Exception as e:
        print(f"❌ Lỗi kết nối Dashboard: {e}")

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    # Test với ảnh mẫu của bạn
    SAMPLE_IMAGE = "inference/fault-insulator.jpeg" 
    
    if os.path.exists(SAMPLE_IMAGE):
        run_local_inference(SAMPLE_IMAGE)
    else:
        print(f"Vui lòng chuẩn bị file {SAMPLE_IMAGE} để test.")
