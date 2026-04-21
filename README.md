# Vision-Based-Fault-Detection

Hệ thống nhận diện lỗi lưới điện 110kV sử dụng Drone AI (YOLOv8 & Roboflow).

## Giới thiệu
Dự án được xây dựng nhằm hỗ trợ việc kiểm tra định kỳ lưới điện cao thế, tự động phát hiện các lỗi như **Sứ nứt (Broken Insulator)** và **Rỉ sét (Rust)** thông qua hình ảnh từ Drone.

## Cấu trúc dự án
- `/training`: Script huấn luyện mô hình YOLOv8.
- `/inference`: Script nhận diện lỗi sử dụng Roboflow Serverless Workflow.
- `/web`: Dashboard quản lý lỗi xây dựng trên ASP.NET Core 9.0 Razor Pages.
- `/integration`: Các công cụ kết nối dữ liệu giữa Drone và Dashboard.

## Hướng dẫn cài đặt
1. **Python Setup**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Web Dashboard Setup**:
   ```bash
   cd web/FaultDetection.Web
   dotnet run
   ```

## Liên hệ
- Tác giả: Minh Châu
- Project: UAV Power Grid Inspection AI
