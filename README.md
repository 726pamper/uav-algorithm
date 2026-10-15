# UAV AI 算法推理管道

基于 **DeepStream + TensorRT** 的无人机实时视频分析系统。单机支持多路任务并发推理、算法动态热切换与视觉大模型协同识别，覆盖目标检测、OBB 旋转框、语义分割、SAM3 分割等多种视觉任务，并通过 MQTT 与上层平台完成指令交互与事件上报。

```
平台 (MQTT) ──→ mqtt_redis.py ──→ DeepStream 推理子进程（每任务一个）
                    │                      │
                  Redis ←──────────────────┘   检测结果 (Pub/Sub)
                    │
          redis_result_processor.py ──→ MinIO (存储) + MQTT (事件上报)
```

---

## 一、核心设计

### 1. 主控编排与多路并发

`mqtt_redis.py` 作为编排主控，为每路任务用 `multiprocessing` 拉起**独立推理子进程**（`spawn` 模式，全新解释器 + 独立日志 + 独立 GPU 分配），打通 **RTSP 拉流 → TensorRT 推理 → RTMP 推流** 闭环。子进程崩溃不影响主控，主控退出会级联清理子进程。

### 2. 指令通道与进程内热更新

平台指令经 MQTT 到达主控后，通过 **Redis Pub/Sub**（频道名 = 设备 SN）下发给对应推理子进程，实现算法列表、置信度阈值、检测区域、大模型提示词的**进程内热更新** —— 动态增删算法、切换模型**无需重启推理进程**。

任务状态同时写入 Redis 快照（`inference:task:{task_id}`，24h TTL），服务重启后可恢复任务视图，避免状态丢失导致的指令错配。

### 3. 视频流 / 图片双模式

按平台下发的机场类型字段自动路由：

| 模式 | 推理模块 | 输入 | 输出 |
|---|---|---|---|
| 视频流 | `common/video_trt_inference.py` | RTSP 拉流 | RTMP 推流 / 仅推理 |
| 图片 | `common/deepstream_trt_inference.py` | MinIO 图片（拍照任务） | 检测结果 + 事件 |

图片模式兼容「无检测模型、纯视觉大模型识别」场景（跳过 CNN，直接走 VLM），并支持只推理不推流的低带宽模式。

### 4. 可扩展后处理算法框架

`postprocess_utils/` 下以 `ALGO_ID` + `EXPORT(detect/draw)` 约定实现算法模块，系统自动扫描注册，**新增算法对主流程零侵入**：

```
postprocess_utils/
├── normal_det.py / normal_obb.py / normal_seg.py     # 通用检测 / 旋转框 / 分割
├── normal_sam3.py / people_sam3.py                   # SAM3 分割
├── License_plate.py                                  # 车牌 OCR（ONNX PaddleOCR）
├── car_count.py / person_count.py                    # 车辆 / 人员计数
├── car_speed_lines.py / vehicle_speed_tracker.py     # 车辆测速
├── fire_exit.py / charging_pile.py                   # 消防通道占用 / 充电桩
├── gs_violation.py / illegal_construction.py         # 高速违章 / 违规施工
├── floating_debris.py / weiting_car.py               # 漂浮物 / 违停
├── llm_draw.py                                       # 大模型结果可视化
├── geo_area_checker.py                               # 地理围栏与区域判定
└── chinese_name.py                                   # 中英文类名映射
```

### 5. 视觉大模型双链路

| 链路 | 触发 | 说明 |
|---|---|---|
| **实时识别** | 每 N 帧 / 每张照片 | 按用户提示词做 VLM 目标识别，结构化 JSON 输出类别与坐标 |
| **二次复审** | 初检命中后 | 对 YOLO 初检结果做 LLM 语义级复核，降低误检率，可按算法粒度开关 |

实时链路支持平台下发大模型权限字段（`llmPermission`）：`0` = 该组织无大模型权限，算法端忽略全部提示词；`1` = 允许。权限在「任务启动」与「动态切换」两个通道均可生效，并按任务记忆，字段缺省时沿用该任务最近一次的权限值。

### 6. 结果消费与告警链路

推理子进程将检测帧与元数据写入 Redis 并发布到 `alarm-images`；`redis_result_processor.py` 订阅消费，完成算法后处理、MinIO 存储（桶自动创建）与 MQTT `ai-event` 事件上报。

### 7. 稳定性与性能

- **GPU 负载均衡**：`common/gpu_load_balancer.py` 支持轮询 / 最少占用 / 加权 / 自适应策略，按任务分配 GPU
- **TensorRT 引擎封装**：`common/trt_inference_torch.py` 统一 detect / obb / segment 三类任务的模型预热与推理接口
- **握手鉴权**：主控与平台通过公钥校验完成握手，未完成握手前丢弃业务指令
- **守护与自愈**：`common/watchdog.py` 心跳检测 + `anyshell/` 下的 supervisor 托管与 systemd timer 容器探活
- **分阶段埋点**：对大模型链路按「JPG 编码 → 网关往返 → 结果解析 → Redis 上报」分段计时，便于定位耗时瓶颈

---

## 二、目录结构

```
.
├── mqtt_redis.py                    # 主控：MQTT 指令接收、任务生命周期、Redis 状态同步
├── redis_result_processor.py        # 结果消费：后处理、MinIO 上传、MQTT 事件推送
├── common/
│   ├── deepstream_trt_inference.py  # 图片模式推理进程
│   ├── video_trt_inference.py       # 视频流模式推理进程
│   ├── trt_inference_torch.py       # TensorRT 引擎封装（ultralytics）
│   ├── llm_infer.py                 # 大模型实时识别 / 二次复审调用
│   ├── config.py                    # 算法清单（模型路径、阈值、提示词等）
│   ├── config_loader.py             # profile 化配置加载（main / test）
│   ├── gpu_load_balancer.py         # GPU 负载均衡策略
│   ├── watchdog.py                  # 心跳看门狗
│   └── healthcheck.py               # 健康检查
├── postprocess_utils/               # 40+ 算法模块（自动注册）
├── utils/                           # Redis / MinIO / 类型转换等工具
├── anyshell/                        # supervisor 配置与部署脚本
└── UAV-AI-System-Flow.plantuml      # 完整数据流时序图（含算法路径与 LLM 路径对比）
```

---

## 三、运行环境

| 组件 | 版本 / 说明 |
|---|---|
| 推理 | TensorRT 10.x、DeepStream 7.x、PyTorch、ultralytics、CuPy |
| 通信 | paho-mqtt、redis-py |
| 存储 | MinIO（S3 兼容） |
| 视觉 | OpenCV、Shapely、PaddleOCR（ONNX） |
| 大模型 | OpenAI SDK（兼容 DashScope 等网关） |

### 需要自行准备的模型与字体

为保证仓库轻量，以下二进制资源**未包含**在本仓库中，需自行下载后放入对应目录：

```
postprocess_utils/fonts/wqy-zenhei.ttc                 # 中文绘制字体（文泉驿正黑）
postprocess_utils/onnxocr/fonts/simfang.ttf            # OCR 绘制字体
postprocess_utils/onnxocr/models/ppocrv4|ppocrv5/...   # PaddleOCR det/rec/cls ONNX 模型
```

PaddleOCR 模型下载与转换为 ONNX 的完整步骤见 [`postprocess_utils/onnxocr/readme.md`](postprocess_utils/onnxocr/readme.md)。

算法模型（`.engine` / `.pt`）路径在 `common/config.py` 中按算法编号配置，请按你的实际存放位置调整。

---

## 四、快速开始

### 1. 配置

配置采用 profile 机制（`common/config-dev.yaml` 指定 `profile: main|test`），也可用环境变量临时覆盖：

```bash
export UAV_PROFILE=test
```

编辑对应 profile 文件，填入你的环境地址与凭据：

```yaml
minio:
  endpoint: "127.0.0.1:9000"
  access_key: "minioadmin"
  secret_key: "minioadmin"

mqtt:
  host: '127.0.0.1'
  port: 1883
  username: "mqtt_user"
  password: "change_me"
  topics:
    algorithm_control: "ai/calculate/event"
    photo_notify: "ai/calculate/control/photo"
    photo_result: "ai/calculate/photo_result"

redis:
  host: 127.0.0.1
  port: 6379
  password: change_me
  database: 1
```

大模型 API Key 通过环境变量注入（不要写进代码或提交到仓库）：

```bash
export DASHSCOPE_API_KEY="sk-your-key"
```

### 2. 启动

```bash
# 主控服务
python mqtt_redis.py

# 结果处理器（可独立运行）
python redis_result_processor.py
```

容器内使用 supervisor 托管：

```bash
supervisord -c /etc/supervisor/supervisord.conf   # 启动
supervisorctl status                              # 查看状态
supervisorctl reload                              # 调整 /etc/supervisor/ 内文件后重新加载
supervisorctl restart all                         # 更新代码后重启服务
```

### 3. 容器化部署

```bash
cd anyshell
# 按注释修改路径与镜像名后执行
./start_uav_ai.sh
```

---

## 五、数据流

完整链路（含算法路径与 LLM 路径的分步对比）见 [`UAV-AI-System-Flow.plantuml`](UAV-AI-System-Flow.plantuml)，可用 PlantUML 渲染。

---

## 六、说明

本仓库为个人技术整理版本，已移除生产环境的地址、凭据与模型文件；示例配置中的密码均为占位符，部署时请替换为实际值。
