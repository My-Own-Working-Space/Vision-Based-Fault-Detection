import requests
import json
import time
import random

# ==========================================
# Cấu hình Kết nối
# ==========================================
API_URL = "http://localhost:5000/api/faults"  # Địa chỉ Dashboard ASP.NET
TOWER_IDS = ["T-110KV-01", "T-110KV-02", "T-110KV-03", "T-110KV-04"]
FAULT_TYPES = ["Bình thường", "Sứ nứt", "Rỉ sét"]

def send_fault_to_dashboard(tower_id, fault_type, confidence, lat, lon):
    """
    Gửi dữ liệu lỗi về Website ASP.NET Core
    """
    payload = {
        "towerId": tower_id,
        "faultType": fault_type,
        "confidenceScore": confidence,
        "imagePath": "/uploads/mock_drone_capture.jpg",
        "latitude": lat,
        "longitude": lon
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        print(f"--- Đang gửi: {tower_id} - {fault_type} ({confidence*100:.1f}%) ---")
        response = requests.post(API_URL, data=json.dumps(payload), headers=headers)
        
        if response.status_code == 200:
            print("Successfully sent to dashboard ✅")
            return True
        else:
            print(f"Failed to send: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error connecting to dashboard: {e}")
        return False

# ==========================================
# Giả lập Drone bay và nhận diện (Simulate)
# ==========================================
def simulate_drone_mission():
    print("🚀 Bắt đầu giả lập Drone tuần tra lưới điện...")
    
    # Tọa độ khu vực Hà Nội (giả lập)
    base_lat = 21.0285
    base_lon = 105.8542
    
    for i in range(10):
        # Giả lập tọa độ di chuyển
        curr_lat = base_lat + random.uniform(-0.01, 0.01)
        curr_lon = base_lon + random.uniform(-0.01, 0.01)
        
        tower = random.choice(TOWER_IDS)
        fault = random.choice(FAULT_TYPES)
        conf = random.uniform(0.75, 0.99)
        
        # Chỉ gửi nếu phát hiện lỗi hoặc ngẫu nhiên gửi trạng thái bình thường
        send_fault_to_dashboard(tower, fault, conf, curr_lat, curr_lon)
        
        time.sleep(3) # Đợi 3 giây giữa mỗi trạm

if __name__ == "__main__":
    simulate_drone_mission()
