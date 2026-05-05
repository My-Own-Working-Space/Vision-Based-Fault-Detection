# Vision-Based Fault Detection (UAV Inspection)

Hệ thống nhận diện lỗi lưới điện 110kV sử dụng Drone AI kết hợp sức mạnh của **YOLOv8** (Phát hiện vật thể) và **Custom CNN** (Phân loại chi tiết).

## Tính năng mới: Real-time Hybrid Detection
Dự án đã được nâng cấp với luồng xử lý thời gian thực mạnh mẽ:
- **YOLOv8 (Detector):** Xác định chính xác vị trí của Sứ cách điện (Insulator) trong khung hình.
- **Custom CNN (Refiner):** Sử dụng kiến trúc MobileNetV3 để phân loại sâu trạng thái của sứ (Sạch vs Bẩn/Nứt).
- **IP Camera Integration:** Kết nối trực tiếp với smartphone (thông qua IP Webcam) để giả lập camera của Drone.
- **FastAPI Backend:** Xử lý và stream video đã được AI phân tích lên trình duyệt web.

## Luồng hoạt động (Workflow)
```text
📱 Camera Drone/Phone → Stream Video → FastAPI Server  
→ YOLOv8 (Tìm vị trí Sứ)  
→ Tự động Crop vùng ảnh Sứ  
→ Custom CNN (Soi lỗi Sạch/Bẩn)  
→ Output (Vẽ khung xanh nếu sạch, đỏ nếu bẩn + Gắn nhãn chi tiết)
```

## Cấu trúc dự án
- `/training`: Chứa mã nguồn huấn luyện mô hình (YOLOv8 và Custom CNN MobileNetV3).
- `/inference`: Hệ thống nhận diện thực tế, bao gồm server FastAPI (`app.py`).
- `/models`: Lưu trữ các file trọng số model (`.pt` và `.pth`).
- `/web`: Dashboard quản lý lỗi (ASP.NET Core 9.0).

## Hướng dẫn khởi chạy hệ thống Real-time

### 1. Chuẩn bị Camera
Cài đặt app **IP Webcam** trên điện thoại, bật server và lấy link (VD: `http://192.168.1.x:8080/video`).

### 2. Cài đặt thư viện
```bash
pip install fastapi uvicorn opencv-python ultralytics torch torchvision pillow
```

### 3. Chạy hệ thống
```bash
cd inference
export IP_CAMERA_URL="link_camera_cua_ban"
python app.py
```
Sau đó truy cập `http://localhost:8000/video` để xem kết quả.

## Kết quả huấn luyện
- **YOLOv8:** Huấn luyện trên tập dữ liệu UAV chuyên dụng (Roboflow).
- **CNN:** Đạt độ chính xác cao trong việc phân biệt các vết bẩn nhỏ và nứt vỡ trên bề mặt sứ.