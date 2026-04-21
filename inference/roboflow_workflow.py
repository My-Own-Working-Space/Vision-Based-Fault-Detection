import os
import cv2
import json
import base64
import requests
from inference_sdk import InferenceHTTPClient

# ==========================================
# 1. Cấu hình Roboflow Workflow
# ==========================================
API_URL = "https://serverless.roboflow.com"
API_KEY = "3VRKN4GLeDVKPJ9eQFkG"
WORKSPACE = "pham-hoang-minh-chau"
WORKFLOW_ID = "general-segmentation-api"

# Cấu hình Dashboard (để gửi lỗi về)
DASHBOARD_API = "http://localhost:5000/api/faults"

# ==========================================
# 2. Khởi tạo Client
# ==========================================
client = InferenceHTTPClient(
    api_url=API_URL,
    api_key=API_KEY
)

def run_segmentation(image_path):
    """
    Chạy workflow segmentation trên một ảnh và trả về kết quả
    """
    print(f"--- Đang phân tích ảnh: {image_path} ---")
    
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy file {image_path}")
        return None

    try:
        # Chạy workflow
        result = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW_ID,
            images={
                "image": image_path
            },
            parameters={
                "classes": "insulator, snow, broken disc"
            },
            use_cache=True
        )
        return result
    except Exception as e:
        print(f"Lỗi khi gọi Roboflow API: {e}")
        return None

def process_results(result, image_path):
    """
    Xử lý kết quả trả về và gửi dữ liệu về Dashboard nếu có lỗi (broken disc)
    """
    if not result:
        return

    # In kết quả thô để debug (Bật lên nếu gặp lỗi)
    # print("DEBUG RAW RESULT:")
    # print(json.dumps(result, indent=2))

    # Giả sử workflow trả về danh sách các detections/predictions
    # Lưu ý: Cấu trúc JSON trả về phụ thuộc vào cách bạn thiết kế Workflow trên Roboflow
    # Ở đây chúng ta sẽ tìm các 'broken disc' (Sứ nứt)
    
    # MOCK logic để bóc tách thông tin (Cần điều chỉnh theo output thực tế của workflow bạn)
    # Thông thường workflow result có dạng: result[0]['outputs']...
    
    # Lấy dữ liệu predictions từ kết quả
    # Cấu trúc thực tế từ debug: [ { "result": { "outputs": { "predictions": [...] } } } ]
    
    # 1. Lọc lấy phần dữ liệu chính
    main_doc = result[0] if isinstance(result, list) and len(result) > 0 else result
    
    # 2. Đi sâu vào các cấp độ (result -> outputs)
    if isinstance(main_doc, dict) and "result" in main_doc:
        main_doc = main_doc["result"]
    
    outputs = main_doc.get("outputs", {})
    
    # 3. Tìm danh sách predictions
    predictions = []
    if isinstance(outputs, list) and len(outputs) > 0:
        predictions = outputs[0].get("predictions", [])
    elif isinstance(outputs, dict):
        # Nếu outputs là dict, có thể predictions nằm trực tiếp bên trong 
        # hoặc nằm trong một key động (tên bước của workflow)
        if "predictions" in outputs:
            predictions = outputs["predictions"]
        else:
            # Tìm trong tất cả các keys của outputs
            for key in outputs:
                if isinstance(outputs[key], list) and len(outputs[key]) > 0:
                    if isinstance(outputs[key][0], dict) and "predictions" in outputs[key][0]:
                        predictions = outputs[key][0]["predictions"]
                        break
                    elif key == "predictions": # Trường hợp predictions là list trực tiếp
                        predictions = outputs[key]
                        break

    if not predictions:
        print("Cảnh báo: Không tìm thấy danh sách 'predictions' trong kết quả.")
        return

    print(f"--- Kết quả phân tích (Tìm thấy {len(predictions)} đối tượng) ---")
    found_fault = False

    for pred in predictions:
        # Bảo vệ nếu pred không phải là dict
        if not isinstance(pred, dict):
            continue
            
        class_name = pred.get('class', '').strip()
        confidence = pred.get('confidence', 0)
        
        if not class_name: continue
        
        print(f"- Phát hiện: {class_name} ({confidence*100:.1f}%)")
        
        # Kiểm tra lỗi (broken disc / insulator broken)
        if ('broken' in class_name.lower() or 'disc' in class_name.lower()) and confidence > 0.4:
            found_fault = True
            send_to_dashboard(class_name, confidence, image_path)
        
        if class_name == 'broken disc' and confidence > 0.6:
            found_fault = True
            send_to_dashboard(class_name, confidence, image_path)

    if not found_fault:
        print("Không phát hiện lỗi nghiêm trọng.")

def send_to_dashboard(fault_type, confidence, image_path):
    """
    Gửi dữ liệu về Dashboard ASP.NET Core
    """
    payload = {
        "towerId": "T-110KV-ROBO", # Mã trụ giả lập
        "faultType": "Sứ nứt (Broken)",
        "confidenceScore": float(confidence),
        "imagePath": f"/uploads/{os.path.basename(image_path)}",
        "latitude": 21.0305,
        "longitude": 105.8522
    }
    
    try:
        res = requests.post(DASHBOARD_API, json=payload)
        if res.status_code == 200:
            print(f"Đã gửi cảnh báo '{fault_type}' về Dashboard.")
        else:
            print(f"Lỗi khi gửi Dashboard: {res.status_code}")
    except Exception as e:
        print(f"Không thể kết nối Dashboard: {e}")

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    # Test với một ảnh mẫu (Bạn cần chuẩn bị ảnh YOUR_IMAGE.jpg)
    SAMPLE_IMAGE = "fault-insulator.jpeg" 
    
    # Nếu chưa có ảnh, script sẽ báo lỗi file không tồn tại
    res = run_segmentation(SAMPLE_IMAGE)
    if res:
        process_results(res, SAMPLE_IMAGE)
        print("\nKết quả thô từ Roboflow:")
        print(res)
