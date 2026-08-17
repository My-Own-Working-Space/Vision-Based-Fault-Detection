# Vision-Based Fault Detection (UAV Power-Line Inspection Service)

Hệ thống nhận diện lỗi lưới điện 110kV sử dụng Drone AI kết hợp sức mạnh của **YOLO11** (Phát hiện khẩn cấp/biên) và **RF-DETR** (Phân tích ngoại tuyến chính xác cao).

## Kiến trúc Hệ thống (Architecture)

Dự án được triển khai dưới dạng monorepo với 2 module chạy độc lập chia sẻ chung các schemas và contract giao tiếp với backend ASP.NET Core:

1. **Edge Raspberry Module (`edge_raspberry/`)**:
   - Sử dụng **YOLO11** (CPU-only, tối ưu hóa cho ARM64).
   - Dùng để phát hiện nhanh các bất thường trong lúc bay.
   - Nhận nhiệm vụ từ queue `ai.analysis.edge.requested`.
2. **Server PC Module (`server_pc/`)**:
   - Sử dụng **RF-DETR** (Hỗ trợ GPU CUDA / fallback CPU).
   - Dùng để phân tích chi tiết hình ảnh/video độ phân giải cao sau khi drone upload.
   - Nhận nhiệm vụ từ queue `ai.analysis.server.requested`.
3. **Shared Module (`shared/`)**:
   - Chứa DTO, schemas chuẩn hóa, logic tải media, class mapping, xử lý bbox và callback retry.
   - Chứa client Roboflow Workflow dùng `inference-sdk` để gọi workflow hosted khi cần tích hợp inference qua Roboflow Serverless.

---

## Cấu trúc thư mục

```text
ai-project/
├── shared/                 # Contract & utilities chung
│   ├── schemas/            # Schemas request/result
│   ├── services/           # Tải file, callback, mapping
│   ├── messaging/          # RabbitMQ client kết nối tự phục hồi
│   ├── utils/              # Bbox normalization & logging
│   └── config/             # BaseSettings
│
├── edge_raspberry/         # Module chạy trên Raspberry Pi (YOLO11)
│   ├── app/                # Main app, detector, consumer, settings
│   ├── Dockerfile          # Build đa kiến trúc (ARM64/AMD64)
│   └── requirements.txt
│
├── server_pc/              # Module chạy trên PC/Server (RF-DETR)
│   ├── app/                # Main app, detector, consumer, settings
│   ├── Dockerfile          # Hỗ trợ CUDA GPU
│   └── requirements.txt
│
└── docker-compose.yml      # Orchestration dịch vụ
```

---

## Hướng dẫn Khởi chạy Dịch vụ

### 1. Cài đặt Cấu hình Môi trường
Production AI dùng một file env riêng, không dùng chung `.env` của PMS:
```bash
cp .env.example .env
```

Điền các giá trị thật vào `.env`, đặc biệt `RABBITMQ_PASS`, `AI_SERVICE_KEY` và `ROBOFLOW_API_KEY`. File `.env` của AI không được commit.

### 2. Khởi chạy bằng Docker Compose
Dùng đúng một file Compose production. RabbitMQ và Gateway được dùng từ PMS, không tạo broker riêng:
```bash
docker compose up -d --build
```

### 3. Chạy từng Module độc lập cục bộ (Local Development)

#### Thiết lập Virtual Environment chung:
```bash
python3 -m venv .venv
source .venv/bin/python
pip install -r edge_raspberry/requirements.txt   # Cho Edge development
# HOẶC
pip install -r server_pc/requirements.txt        # Cho Server development
```

#### Khởi chạy Edge Module (YOLO11):
```bash
export DEVICE_PROFILE=edge
python3 -m edge_raspberry.app.main
```
Ứng dụng sẽ lắng nghe tại cổng `8001`.

#### Khởi chạy Server Module (Roboflow thật cho server, Harness chỉ local/test):
```bash
export DEVICE_PROFILE=server
export SERVER_INFERENCE_BACKEND=roboflow
export ROBOFLOW_API_KEY=CHANGE_ME
python3 -m server_pc.app.main
```
Dịch vụ sẽ khởi động và lắng nghe tại cổng `8002`. Production nên dùng `SERVER_INFERENCE_BACKEND=roboflow` để chạy workflow/model thật. `SERVER_INFERENCE_BACKEND=harness` chỉ dùng fake provider cho local/offline tests. Để quay lại RF-DETR local, đặt `SERVER_INFERENCE_BACKEND=local`.

Docker `server_pc` production mặc định dùng Roboflow hosted workflow và không cài RF-DETR/PyTorch CUDA. Deploy riêng server PC bằng:

```bash
docker compose up -d --build server_pc
```

Chỉ khi cần backend RF-DETR local mới build thêm dependency ML CPU-only:

```bash
INSTALL_LOCAL_ML=true SERVER_INFERENCE_BACKEND=local docker compose up -d --build server_pc
```

---

## Roboflow Workflow Integration

Workflow hosted đã được tích hợp trong `shared/services/roboflow_workflow_client.py`. Production `server_pc` nên chạy backend `roboflow` để gọi workflow/model thật. Backend `harness` dùng fake provider và chỉ phù hợp cho local/offline validation.

- Workspace slug: `les-workspace-ijdwd`
- Workflow slug: `evn-object-detection-vevn-object-detection-cnyo0-2-yolo11n-t1-logic`
- API URL: production dùng service nội bộ `http://roboflow-inference:9001`
- Declared input: `image`
- Declared runtime parameters: none
- Declared output key: `predictions`

Thiết lập API key qua biến môi trường, không commit secret:
```bash
export ROBOFLOW_API_KEY="<key-from-app.roboflow.com/settings/api>"
```

Ví dụ gọi workflow cho một ảnh HTTPS:
```python
from shared.services.roboflow_workflow_client import run_evn_object_detection_workflow

runs = run_evn_object_detection_workflow(
    "https://media.roboflow.com/notebooks/examples/dog.jpeg",
)
detection_result = runs[0].as_detection_result()
```

Client dùng `InferenceHTTPClient.run_workflow(...)`, timeout theo attempt, retry với exponential backoff, và parse output theo key thực tế của workflow. Nếu workflow về sau thêm output dạng ảnh base64, truyền `output_dir` để decode ra file thay vì log payload base64.

Lưu ý: lần kiểm tra live ngày 2026-07-19 bằng Roboflow MCP trả về lỗi server-side 500 vì workflow wrapper đang bind `model_id` vào child workflow, trong khi child workflow chỉ khai báo `class_agnostic_nms`, `confidence`, `image`, `iou_threshold`, và `max_detections`. Cần publish lại workflow trên Roboflow trước khi smoke test live có thể pass.

---

## PMS Integration

Khi chạy cùng UAV PMS microservices, AI service phải dùng chung RabbitMQ với PMS. Luồng chính hiện tại là:

`AIInspectionService -> RabbitMQ request -> Python AI consume -> RabbitMQ result -> AIInspectionService consume result`

Với `server_pc`, HTTP callback không còn nằm trong luồng tích hợp PMS. Kết quả AI được trả về bằng RabbitMQ result event. Không chạy RabbitMQ riêng cho luồng tích hợp PMS, nếu không consumer sẽ nghe nhầm broker và job AI sẽ không được xử lý.

Local host mode, khi PMS gateway đang chạy ở `http://localhost:5194` và RabbitMQ publish port `5672`:

```bash
cd /home/minhchau/Documents/Vision-Based-Fault-Detection
RABBITMQ_HOST=localhost \
RABBITMQ_PORT=5672 \
ALLOW_PRIVATE_IPS=true \
.venv/bin/uvicorn server_pc.app.main:app --host 0.0.0.0 --port 8002
```

Docker mode, khi PMS compose đã tạo network `uavpms_org_default` và các container `uav-rabbitmq`, `uav-gateway` đang chạy:

```bash
docker compose up -d --build
```

`.env` của AI chứa các cấu hình tích hợp:

- `RABBITMQ_HOST=uav-rabbitmq`
- `AI_SERVICE_KEY` lấy cùng giá trị với `AIService_ServiceKey` bên PMS
- `RESULT_EXCHANGE=identity-exchange`
- `RESULT_ROUTING_KEY=identity.event.aianalysisresultevent`
- `RESULT_QUEUE_NAME=ai.analysis.result`
- `RESULT_RETRY_QUEUE_NAME=ai.analysis.result.retry`
- `RESULT_DEAD_LETTER_QUEUE=ai.analysis.result.dead-letter`
- `MESSAGE_RETRY_LIMIT=3`
- `PMS_DOCKER_NETWORK` mặc định là `uavpms_org_default`; đổi biến này nếu PMS compose tạo network tên khác
- `ROBOFLOW_API_URL` không cần khai báo vì Compose cố định vào `http://roboflow-inference:9001`

Kiểm tra nhanh:

```bash
curl http://localhost:8002/health
curl http://localhost:8002/ready
```

RabbitMQ binding kỳ vọng:

```text
identity-exchange -> ai.analysis.server.requested -> identity.event.aianalysisrequestedevent.server
ai.analysis.server.requested -> dispatcher -> ai.analysis.server.image.requested
ai.analysis.server.requested -> dispatcher -> ai.analysis.server.video.requested
identity-exchange -> ai.analysis.edge.requested -> identity.event.aianalysisrequestedevent.edge
ai.analysis.edge.requested -> dispatcher -> ai.analysis.edge.image.requested
ai.analysis.edge.requested -> dispatcher -> ai.analysis.edge.video.requested
identity-exchange -> ai.analysis.result -> identity.event.aianalysisresultevent
ai.analysis.result.retry --TTL--> identity-exchange(identity.event.aianalysisresultevent)
ai.analysis.result.dead-letter <- DLX khi result payload invalid hoặc xử lý vượt retry limit
```

Mỗi runtime chạy một ingress dispatcher và hai consumer độc lập cho ảnh/video.
Vì vậy một video đang xử lý không giữ consumer ảnh. Có thể scale từng runtime bằng
nhiều process/container; RabbitMQ sẽ phân phối job trong từng media queue cho các
consumer cạnh tranh.

## Kiểm thử (Testing)

Chạy bộ unit test kiểm tra bbox, class mapping, callback retry, và routing rules:
```bash
.venv/bin/python -m unittest discover -s shared/tests -p "test_*.py"
```

Chạy smoke test live Roboflow sau khi workflow published hoạt động:
```bash
export ROBOFLOW_API_KEY="<key-from-app.roboflow.com/settings/api>"
export RUN_ROBOFLOW_SMOKE_TEST=1
.venv/bin/python -m unittest shared.tests.test_roboflow_workflow_client.TestRoboflowWorkflowSmoke
```

## Các API Endpoint chính (cả 2 Runtimes)
- `GET /health`: Kiểm tra trạng thái tiến trình (liveness).
- `GET /ready`: Kiểm tra mô hình đã load thành công chưa (readiness).
- `POST /api/analyze`: Endpoint RESTful gửi yêu cầu phân tích bất đồng bộ.
