"""
图片检测推理子进程入口（与 mqtt_redis 编排侧配合）。

本模块是 ``common/deepstream_trt_inference.py`` 的「图片模式」改造副本：
- 移除 GStreamer / DeepStream / pyds 依赖（不再拉 RTSP 流、不再有 Pad Probe 与推流）。
- 保留 ``MODEL_DICT`` 多算法引擎管理与 ``initialize_inference_engine()``、Redis Pub/Sub 热切换回调 ``on_message``。
- 平台无人机手动拍照上传 MinIO 后，编排侧通过 Redis 频道 ``photo_task:{task_id}`` 下发图片任务，
  本进程从 MinIO（或 HTTP URL）读取图片 → 调用已加载的 ``MODEL_DICT`` 引擎推理 →
  按原格式写入 Redis（``image:*`` / ``metadata:*``）并 Publish ``alarm-images``。
- 后半段 ``redis_result_processor.py`` 消费链路 100% 兼容，metadata 结构与原系统一致。

兼容约定
--------
- ``fast_upload_to_redis`` / ``FastImageEncoder`` / ``build_detect_frame_from_detections`` 等全部沿用原实现，
  保证写入 Redis 的 ``metadata:{task_id}:{ts}`` 字段（image_key / detections / detect_frame / TASK /
  algorithmnum / model_code / second_review / filepath / uavsn / AREA / timestamp / task_id）与原系统完全一致。
- 新增 ``force`` 参数用于手动拍照场景绕过原 3s 节流（默认 False，保持原行为）。
"""

import sys
import os
import queue
import threading
import time
import json
import math
import traceback
import urllib.request

from PIL import Image
import io
import torch
import numpy as np
import cv2
import base64
from importlib import reload
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from minio import Minio

import common.config
from common.config_loader import get_profile, get_config_path
from common.trt_inference_torch import TRTInferenceEngine
from common.load_SAM3 import Sam3VideoLoader, ReleaseSam3Model
from common.llm_infer import realtime_llm_infer
from postprocess_utils.chinese_name import CLASS_NAME_TO_CHINESE
from utils.redis_helper import RedisHelper
from utils.minio_util import init_minio_client
from postprocess_utils.car_speed_lines import update_speed_from_det_tensor
from postprocess_utils.vehicle_speed_tracker import process_vehicle_speeds_with_lanes

MODEL_CONIFG = common.config.algo_config

# --- 推理与跟踪默认参数（可被 Redis 配置覆盖 CONF_THRESHOLD） ---
MODEL_ENGINE_PATH = None
CONF_THRESHOLD = 0.20
IOU_THRESHOLD = 0.65
ENABLE_TRACKING = True
TRACKER_CONFIG = "bytetrack.yaml"
CONFIG_ALGORITHMS = None
CLASS_NAMES = {}

model_type = None
YAML_PATH = get_config_path()
print(f"[图片推理子进程] 当前环境: {get_profile()}, 配置文件: {YAML_PATH}")
FILEPATH = None
MODEL_DICT = None
MINIO_CLIENT = None
REDIS_CLIENT = None
TASK_ID = None
ALGORITHMNUM = None
THRESH = {'car': 0.5, 'vehicle': 0.5, 'person': 0.5, 'build_rubbish': 0.5, 'live_rubbish': 0.5}
UAVSN = None

# --- 告警图上传节流（按 task_id）---
LAST_UPLOAD_TIME = {}
UPLOAD_INTERVAL = 3.0
UPLOAD_THREAD_POOL = None
LLM_THREAD_POOL = None
ENABLE_COMPRESSION = True
COMPRESSION_QUALITY = 80

STOP_REQUESTED = False
AREAJTR = None

_PHOTO_TASK_QUEUE = queue.Queue()   # 新增：拍照任务队列
_PHOTO_PROCESSING = False           # 新增：是否正在处理
_SEEN_PHOTO_KEYS = {}   # object_key -> 上次处理时间戳（照片去重）

# --- photo_result MQTT publisher ---
_PHOTO_MQTT_CLIENT = None
_PHOTO_MQTT_TOPIC = None

# --- DashScope 兼容 OpenAI：异步调用与节流 ---
LLM_CLIENT = None
LLM_UPLOAD_INTERVAL = 1.0
LAST_LLM_UPLOAD_TIME = {}
AVA_LLM_INFER_TIME = 0
# platform llmPermission: 1 = LLM detection allowed, 0 = disabled, absent = keep current
LLM_PERMISSION = 1

def normalize_llm_permission(value, default: int = 1) -> int:
    """把平台 llmPermission 归一化为 0（无权限）/ 1（有权限）。

    None 或无法识别的值返回 ``default``：老平台不下发该字段时保持原有行为。
    """
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in ("0", "false", "no", "off"):
        return 0
    if text in ("1", "true", "yes", "on"):
        return 1
    return default


def llm_entry_prompt_text(llm_config) -> str:
    """Prompt text of one LLM entry; "" when the platform left it empty.

    The platform sends LLMItem objects whose prompts[0].prompt can be empty
    ("LLM not configured"); such an entry must NOT enable the LLM path.
    Accepts both the current {"prompts": "<json str>"} and the legacy
    top-level {"prompt": "..."} shape.
    """
    try:
        entry = llm_config or {}
        prompts_raw = entry.get("prompts")
        if prompts_raw:
            prompts_list = json.loads(prompts_raw) if isinstance(prompts_raw, str) else prompts_raw
            if isinstance(prompts_list, list) and prompts_list:
                text = str((prompts_list[0] or {}).get("prompt") or "").strip()
                if text:
                    return text
        return str(entry.get("prompt") or "").strip()
    except Exception:
        return ""


# --- 推理临界区与帧计数 ---
inference_engine = None
ENGINE_LOCK = threading.RLock()
COUNT_FRAMES = 0

NUM_GPUS = 2
INFERENCE_EXECUTOR = None
GPU_INDEX = 0

# --- 算法 69：按 TASK_ID 隔离的车速状态（绝对速 → 相对速 → 上报/OSD） ---
TRACK_STATE_DS = {}
TRACK_ABSOLUTE_SPEEDS_DS = {}
TRACK_RELATIVE_SPEEDS_DS = {}

LLM_PROMPT = {}

ALGORITHM_SECOND_REVIEW = {}


def initialize_inference_engine():
    """根据全局 ``ALGORITHMNUM`` 与 ``CONFIG_ALGORITHMS`` 填充 ``MODEL_DICT``。

    策略:
        - 非 SAM3: ``TRTInferenceEngine`` 绑定 ``cuda:0``（避免 TRT engine 跨设备）。
        - SAM3: ``Sam3VideoLoader`` 绑定 ``cuda:1``，独立 ``frame_skip``，关闭 ByteTrack。
        - 线程池 ``INFERENCE_EXECUTOR`` 仅初始化一次。

    Raises:
        RuntimeError: 无任何模型加载成功。

    Note:
        调用前须已由 ``on_message`` / ``main`` 写好 ``ALGORITHMNUM`` 列表。
    """
    global inference_engine, CONF_THRESHOLD, MODEL_DICT, ALGORITHMNUM, CONFIG_ALGORITHMS
    global GPU_INDEX, INFERENCE_EXECUTOR
    with ENGINE_LOCK:
        try:
            GPU_INDEX = 0

            if INFERENCE_EXECUTOR is None:
                INFERENCE_EXECUTOR = ThreadPoolExecutor(max_workers=NUM_GPUS)
                print(f"[并行推理] 线程池已初始化，max_workers={NUM_GPUS}")

            print(f"[初始化] 使用双卡并行推理，NUM_GPUS={NUM_GPUS}")

            MODEL_DICT = {}

            for aid in ALGORITHMNUM:
                model_path = CONFIG_ALGORITHMS[aid].get("modelPath", None)
                modelType = CONFIG_ALGORITHMS[aid].get("modelType", None)
                frame_skip = CONFIG_ALGORITHMS[aid].get("frameSkip", 1)
                enable_tracking = CONFIG_ALGORITHMS[aid].get("enableTracking", ENABLE_TRACKING)
                text_prompt = CONFIG_ALGORITHMS[aid].get("text_prompt",None)
                print(f"model_path: {model_path}, modelType:{modelType}, frame_skip: {frame_skip}, enable_tracking: {enable_tracking}")
                if model_path is None or len(model_path) < 3:
                    print(f"Need Check ModelPath,{model_path}")
                    return

                if modelType == "SAM3":
                    assigned_gpu = 1
                    print(f"[GPU 分配] 模型{aid}(SAM3) -> GPU{assigned_gpu} (独占)")
                else:
                    assigned_gpu = 0
                    print(f"[GPU 分配] 模型{aid} -> GPU{assigned_gpu} (统一)")

                if modelType == "SAM3":
                    try:
                        model = Sam3VideoLoader()
                        MODEL_DICT[f"{aid}"] = {
                            'engine': model,
                            'gpu_id': assigned_gpu,
                            'model_type': 'SAM3',
                            'frame_skip': 60,
                            'frame_counter': 0,
                            "text_prompt":text_prompt,
                            'enable_tracking': False,
                            'mem_usage_gb': 7.0
                        }
                        print(f"[GPU{assigned_gpu}] SAM3 模型初始化成功")
                    except Exception as e:
                        print(f"[GPU{assigned_gpu}] SAM3 模型初始化失败：{e}")
                        continue
                else:
                    try:
                        engine_instance = TRTInferenceEngine(
                            model_path,
                            device=f"cuda:{assigned_gpu}",
                        )
                        _custom_sz = CONFIG_ALGORITHMS[aid].get("imgsz")
                        if _custom_sz:
                            engine_instance.input_size = (
                                (_custom_sz, _custom_sz) if isinstance(_custom_sz, int) else tuple(_custom_sz)
                            )
                            print(f"[TRT] input_size 已按config覆盖: {engine_instance.input_size}")
                        MODEL_DICT[f"{aid}"] = {
                            'engine': engine_instance,
                            'gpu_id': assigned_gpu,
                            'model_type': modelType,
                            'frame_skip': frame_skip,
                            'frame_counter': 0,
                            'enable_tracking': enable_tracking,
                            'mem_usage_gb': 0.8
                        }
                        print(f"[GPU{assigned_gpu}] 推理引擎初始化成功 (model={aid}, type={modelType})")
                    except Exception as e:
                        print(f"[GPU{assigned_gpu}] 推理引擎初始化失败：{e}")
                        continue

            if not MODEL_DICT:
                raise RuntimeError("No models loaded successfully")

            print(f"[初始化完成] 共加载 {len(MODEL_DICT)} 个模型")
            for code, info in MODEL_DICT.items():
                tracking_str = "跟踪" if info['enable_tracking'] else "检测"
                print(f"  - 模型{code}: GPU{info['gpu_id']}, 隔{info['frame_skip']}帧，{tracking_str}")

        except Exception as e:
            inference_engine = None
            print(f"Error initializing TensorRT inference engine: {e}")
            raise


def compute_obb_polygon(cx: float, cy: float, w: float, h: float, angle: float):
    """由中心、宽高与弧度角生成旋转矩形四角（图像坐标系，与推理输出约定一致）。"""
    cosa = math.cos(angle)
    sina = math.sin(angle)
    dw = w / 2.0
    dh = h / 2.0
    pts = [(dw, dh), (-dw, dh), (-dw, -dh), (dw, -dh)]
    polygon = []
    for x, y in pts:
        xx = cx + x * cosa - y * sina
        yy = cy + x * sina + y * cosa
        polygon.append((xx, yy))
    return polygon


def polygon_to_aabb(polygon):
    """旋转框顶点的外接轴对齐矩形 ``(x1,y1,x2,y2)``，用于 JSON 告警中的 bbox 摘要。"""
    xs = [pt[0] for pt in polygon]
    ys = [pt[1] for pt in polygon]
    return min(xs), min(ys), max(xs), max(ys)

def build_points_from_detection(det: dict) -> list:
    """将检测结果统一转换为前端二次编辑使用的 points。

    优先使用 polygon/points/xy,普通检测 bbox 自动转换为四点矩形。
    输出坐标必须保持为原图像素坐标。

    优先级polygon/points/xy -> bbox
    """
    #优先读取 polygon / points / xy
    polygon_pts = det.get("polygon") or det.get("points") or det.get("xy")
    #如果有 polygon/points/xy，就逐点转换（多边形）
    if polygon_pts:
        points = []
        for p in polygon_pts:
            if isinstance(p, dict):
                x = p.get("X", p.get("x", 0))
                y = p.get("Y", p.get("y", 0))

            #如果点是数组
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                x, y = p[0], p[1]
            else:
                continue
            points.append({"X": float(x), "Y": float(y)})
        return points

    #如果没有polygon / points / xy，就读取bbox(普通检测框)
    bbox = det.get("bbox", {})
    #如果bbox是字典的格式
    if isinstance(bbox, dict):
        x1 = bbox.get("x1", bbox.get("X1", 0))
        y1 = bbox.get("y1", bbox.get("Y1", 0))
        x2 = bbox.get("x2", bbox.get("X2", 0))
        y2 = bbox.get("y2", bbox.get("Y2", 0))

    #如果是列表的形式
    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x1, y1, x2, y2 = bbox[:4]

    #如果没有有效的bbox格式，就返回空列表
    else:
        return []


    #返回坐标信息
    return [
        {"X": float(x1), "Y": float(y1)},
        {"X": float(x2), "Y": float(y1)},
        {"X": float(x2), "Y": float(y2)},
        {"X": float(x1), "Y": float(y2)},
    ]


def build_detect_frame_from_detections(detections: list, default_model_code: str = None) -> dict:
    """构建前端二次编辑使用的 detect_frame 节点。

    新结构：按算法（model_code）分组 en_names/cn_names，boxes 中增加 method 和 cn_name 字段。

    注意：
    不直接使用模型原始 class_id 作为前端 class_id。
    因为多算法场景下，不同算法的 class_id 会出现重复，
    这里按 (model_code, label) 组合重新生成前端唯一 class_id。

    Args:
        detections: 检测结果列表，每个元素可含 model_code 字段
        default_model_code: 当检测项无 model_code 时使用的默认算法标识
    """
    # 按算法分组的类别表  {model_code: {label_to_id: int, en_names: list, cn_names: list}}
    algo_groups = {}

    # 检测框的信息
    boxes = []

    # 遍历检测结果
    for det in detections:
        raw_class_id = det.get("class_id")
        if raw_class_id is None:
            continue

        # 获取类别的英文名称
        label = (
            det.get("class_name")
            or det.get("label")
            or f"class_{int(raw_class_id)}"
        )

        # 过滤后处理/分割区域，不传给前端二次编辑
        if label.startswith("class_"):
            continue

        # 过滤掉只有多边形点、没有 bbox 的数据，比如应急车道后处理轮廓点
        if (det.get("polygon") or det.get("points") or det.get("xy")) and not det.get("bbox"):
            continue

        # 获取算法标识（model_code），优先从检测项中取，否则使用默认值
        model_code = str(det.get("model_code", default_model_code or "unknown"))

        # 获取中文名
        cn_label = CLASS_NAME_TO_CHINESE.get(label, label)

        # 生成新的检测框的点位格式
        points = build_points_from_detection(det)
        if len(points) < 3:
            continue

        # 初始化该算法的分组
        if model_code not in algo_groups:
            algo_groups[model_code] = {
                "label_to_id": {},
                "en_names": [],
                "cn_names": []
            }

        group = algo_groups[model_code]

        # 在算法内重新生成唯一的 class_id
        if label not in group["label_to_id"]:
            front_class_id = len(group["label_to_id"])
            group["label_to_id"][label] = front_class_id

            # 添加英文类别表
            group["en_names"].append({
                str(front_class_id): label
            })

            # 添加中文类别表
            group["cn_names"].append({
                str(front_class_id): cn_label
            })
        else:
            front_class_id = group["label_to_id"][label]

        # 添加检测框
        boxes.append({
            "class_id": front_class_id,
            "label": label,
            "method": model_code,
            "cn_name": cn_label,
            "points": points,
        })

    # 构建返回结构：按算法名分组 + boxes
    result = {}
    for model_code, group in algo_groups.items():
        result[model_code] = {
            "en_names": group["en_names"],
            "cn_names": group["cn_names"]
        }
    result["boxes"] = boxes

    return result

def parse_llm_objects_for_detect_frame(raw_str):
    """解析大模型返回结果，提取 objects 列表。

    手动拍照/实时大模型链路中，大模型返回的数据可能是：
    1. JSON 字符串
    2. ```json ... ``` 包裹的 Markdown 代码块
    3. {"objects": [...]} 字典结构
    4. 直接返回 [...] 列表结构

    本函数统一把这些格式解析成 objects 列表，
    供后续生成前端二次编辑 detect_frame 使用。
    """
    if not raw_str:
        return []

    raw_str = raw_str.strip()
    if raw_str.startswith("```"):
        raw_str = raw_str.split("```")[1]
    raw_str = raw_str.replace("json", "", 1).strip()

    try:
        data = json.loads(raw_str)
        print(f"[LLM Result] result of handling : {data}")
    except Exception:
        return []

    if isinstance(data, dict):
        objects = data.get("objects", [])
        return objects if isinstance(objects, list) else []

    if isinstance(data, list):
        return data

    return []


def build_llm_detections_from_objects(
    objects: list,
    image_height: int,
    image_width: int,
    model_code: str = "llm",
    coordinate_mode: str = "pixel",
) -> list:
    """将大模型 objects 转成与普通算法一致的 detections 列表。"""

    detections = []
    seen_boxes = set()
    label_to_class_id = {}

    for obj in objects:
        bbox = obj.get("bbox_2d")
        if not bbox or len(bbox) != 4:
            continue

        label = obj.get("label") or obj.get("category") or "target"

        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except Exception:
            continue

        # LLM 返回 0~1000 归一化坐标时，换算到原图像素坐标
        if coordinate_mode == "normalized_1000":
            x1 = x1 * image_width / 1000.0
            y1 = y1 * image_height / 1000.0
            x2 = x2 * image_width / 1000.0
            y2 = y2 * image_height / 1000.0

        # 如果已经是原图像素坐标，直接使用
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        x1 = max(0.0, min(x1, float(image_width)))
        y1 = max(0.0, min(y1, float(image_height)))
        x2 = max(0.0, min(x2, float(image_width)))
        y2 = max(0.0, min(y2, float(image_height)))

        if x2 <= x1 or y2 <= y1:
            continue

        box_key = (label, round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3))
        if box_key in seen_boxes:
            continue
        seen_boxes.add(box_key)

        if label not in label_to_class_id:
            label_to_class_id[label] = len(label_to_class_id)

        class_id = label_to_class_id[label]
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        detections.append({
            "class_id": class_id,
            "class_name": label,
            "label": label,
            "conf": float(obj.get("score", obj.get("confidence", 1.0))),
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "model_code": str(model_code),
        })

    return detections


def build_llm_detect_frame_from_objects(
    objects: list,
    image_height: int,
    image_width: int,
    emit_boxes: bool = False,
    model_code: str = "llm",
    coordinate_mode: str = "pixel",
) -> dict:
    """将大模型 objects 转成与普通算法完全一致的 detect_frame 结构。"""

    detections = build_llm_detections_from_objects(
        objects=objects,
        image_height=image_height,
        image_width=image_width,
        model_code=model_code,
        coordinate_mode=coordinate_mode,
    )

    detect_frame = build_detect_frame_from_detections(
        detections,
        default_model_code=str(model_code),
    )

    if not emit_boxes:
        detect_frame["boxes"] = []

    return detect_frame

class FastImageEncoder:
    """告警 JPEG + 检测 JSON 组装：统一走 OpenCV JPEG，支持 detect/obb/segment/SAM3/merged。

    约定:
        ``encode_with_detection_info`` 的 ``task`` 与推理引擎 ``task`` 字段对齐；segment 分支对 tensor 结构有特殊切片。
        算法 69 在 ``track_speeds`` 非空时按 ``TASK_ID``、``track_id``、类别 0（car）写入 ``speed_kmh``。
    """

    @staticmethod
    def encode_gpu_to_jpeg(gpu_array: np.ndarray, quality: int = 95) -> bytes:
        """将 tensor / numpy / cupy 数组编码为 JPEG（兼容原 DeepStream 链路）。"""
        # 如果是 PyTorch tensor，转换为 numpy
        if isinstance(gpu_array, torch.Tensor):
            gpu_array = gpu_array.cpu().numpy()
        elif 'cp' in sys.modules and isinstance(gpu_array, sys.modules['cp'].ndarray):
            # CuPy -> NumPy （这是必要的 CPU 拷贝，但只有一次）
            gpu_array = sys.modules['cp'].asnumpy(gpu_array)

        # 确保格式正确
        if gpu_array.dtype != np.uint8:
            if gpu_array.max() <= 1.0:
                gpu_array = (gpu_array * 255).astype(np.uint8)
            else:
                gpu_array = gpu_array.astype(np.uint8)

        # 处理 RGBA -> BGR（OpenCV 需要）
        if len(gpu_array.shape) == 3 and gpu_array.shape[2] == 4:
            gpu_array = cv2.cvtColor(gpu_array, cv2.COLOR_RGBA2BGR)

        # 编码为 JPEG
        ret, encoded = cv2.imencode('.jpg', gpu_array,
                                   [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ret:
            raise RuntimeError("Failed to encode image to JPEG")

        return encoded.tobytes()

    @staticmethod
    def encode_sam3_detection_info(
        gpu_tensor: torch.Tensor,
        sam3_output,
        quality: int = 80,
        model_code: str = None,
    ) -> tuple:
        """SAM3 模式：编码 SAM3 分割模型输出。"""
        # 转为 numpy（GPU->CPU 拷贝只发生一次）
        if isinstance(gpu_tensor, torch.Tensor):
            image_np = gpu_tensor.cpu().numpy()
        else:
            image_np = np.asarray(gpu_tensor)

        # JPEG 编码
        jpeg_bytes = FastImageEncoder.encode_gpu_to_jpeg(image_np, quality)

        # 提取 SAM3 结果
        results_obj = sam3_output[0] if isinstance(sam3_output, list) else sam3_output

        detections_list = []

        # 检查是否有边界框
        if hasattr(results_obj, 'boxes') and results_obj.boxes is not None:
            boxes_xyxy = results_obj.boxes.xyxy  # (N, 4) - [x1, y1, x2, y2]

            # 获取类别名称映射
            sam3_names = results_obj.names if hasattr(results_obj, 'names') else {}
            if isinstance(sam3_names, dict):
                class_id_to_name = sam3_names
            else:
                class_id_to_name = {i: name for i, name in enumerate(sam3_names)} if sam3_names else {}

            # 遍历每个检测框
            for idx, box_coords in enumerate(boxes_xyxy):
                try:
                    x1, y1, x2, y2 = [float(v) for v in box_coords]

                    # 计算宽高和中心点
                    w = x2 - x1
                    h = y2 - y1
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    # 获取置信度（如果存在）
                    confidence = 1.0
                    if hasattr(results_obj.boxes, 'conf') and results_obj.boxes.conf is not None:
                        confidence = float(results_obj.boxes.conf[idx])

                    # 获取类别ID
                    class_id = 0
                    if hasattr(results_obj.boxes, 'cls') and results_obj.boxes.cls is not None:
                        class_id = int(results_obj.boxes.cls[idx])

                    # 获取类别名称
                    class_name = class_id_to_name.get(class_id, f"class_{class_id}")

                    # 构建检测信息字典
                    det_dict = {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "cx": float(cx),
                        "cy": float(cy),
                        "w": float(w),
                        "h": float(h),
                        "conf": float(confidence),
                        "class_id": int(class_id),
                        "class_name": class_name,
                        "bbox": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2)
                        }
                    }
                    if model_code is not None:
                        det_dict["model_code"] = str(model_code)

                    detections_list.append(det_dict)

                except Exception as e:
                    print(f"[SAM3-Encode] 处理检测框 {idx} 失败: {e}")
                    continue

        detection_json = json.dumps(detections_list, ensure_ascii=False)
        return jpeg_bytes, detection_json

    @staticmethod
    def encode_with_detection_info(
        gpu_tensor: torch.Tensor,
        detections: list,
        quality: int = 80,
        task: bool = True,
        with_tracking: bool = False,
        track_speeds: dict = None,
        names: dict = None,
        model_code: str = None,
    ) -> tuple:
        """
        快速编码 + 检测信息打包

        Args:
            gpu_tensor: GPU上的图像tensor
            detections: 检测结果数组
                - OBB模式（无跟踪）: (N, 7) = [cx, cy, w, h, angle, conf, class_id]
                - OBB模式（有跟踪）: (N, 8) = [cx, cy, w, h, angle, conf, class_id, track_id]
                - DET模式（无跟踪）: (N, 6) = [x1, y1, x2, y2, conf, class_id]
                - DET模式（有跟踪）: (N, 7) = [x1, y1, x2, y2, conf, class_id, track_id]
                - SAM3模式: ultralytics.engine.results.Results 对象
            quality: JPEG质量
            task: 任务类型（detect/obb/segment/SAM3）
            with_tracking: 是否包含跟踪ID（默认False）
            track_speeds: 实时车速字典 {track_id: speed_kmh}，用于算法69直接传递速度
            model_code: 当前算法的编号，用于前端分组

        Returns:
            (jpeg_bytes, detection_json_str)
        """
        # print(f"LOOK THIS TOO: {names}")
        # 特殊处理 SAM3 模式
        if task == 'SAM3':
            return FastImageEncoder.encode_sam3_detection_info(
                gpu_tensor,
                detections,
                quality=quality,
                model_code=model_code,
            )

        # 转为 numpy（GPU->CPU 拷贝只发生一次）
        if task !='segment':
            detections = detections[0].detach().cpu().numpy()
        if isinstance(gpu_tensor, torch.Tensor):
            image_np = gpu_tensor.cpu().numpy()
        else:
            image_np = np.asarray(gpu_tensor)

        # JPEG 编码
        jpeg_bytes = FastImageEncoder.encode_gpu_to_jpeg(image_np, quality)

        # 检测信息（直接 JSON 编码，不需要拷贝）
        detections_list = []
        for det in detections:
            if task=='obb':
                # OBB模式：解析旋转框参数
                if with_tracking and len(det) >= 8:
                    cx, cy, w, h, angle, conf, class_id, track_id = det[:8]
                    track_id = int(track_id) if track_id >= 0 else None
                else:
                    cx, cy, w, h, angle, conf, class_id = det[:7]
                    track_id = None

                polygon = compute_obb_polygon(cx, cy, w, h, angle)
                x1, y1, x2, y2 = polygon_to_aabb(polygon)

                det_dict = {
                    "cx": float(cx),
                    "cy": float(cy),
                    "w": float(w),
                    "h": float(h),
                    "angle": float(angle),
                    "conf": float(conf),
                    "class_id": int(class_id),
                    "class_name": names[int(class_id)],
                    "polygon": [{"x": float(px), "y": float(py)} for px, py in polygon],
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2)
                    }
                }
                if track_id is not None:
                    det_dict["track_id"] = track_id
                if model_code is not None:
                    det_dict["model_code"] = str(model_code)
                detections_list.append(det_dict)

            elif task=='segment':
                # segment模式：解析旋转框参数
                if with_tracking and len(det) >= 5:
                    xy, xyxy, conf, cls, track_id = det[:5]
                    # track_id = int(track_id) if track_id >= 0 else None
                    for xy_,xyxy_,conf_,cls_,track_id_ in zip(xy, xyxy, conf, cls, track_id):
                        x1,y1,x2,y2 = xyxy_
                        if isinstance(xy_,np.ndarray):
                            xy_ = xy_.tolist()
                        w = x2 - x1
                        h = y2 - y1
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        track_id = int(track_id_) if track_id_ >= 0 else None
                        class_id = cls_
                        det_dict = {
                            "xy": xy_,
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                            "cx": float(cx),
                            "cy": float(cy),
                            "w": float(w),
                            "h": float(h),
                            "conf": float(conf_),
                            "class_id": int(class_id),
                            "class_name": names[int(class_id)],
                            "bbox": {
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2)
                            }
                        }
                        if track_id is not None:
                            det_dict["track_id"] = track_id
                        if model_code is not None:
                            det_dict["model_code"] = str(model_code)
                        detections_list.append(det_dict)
                else:
                    xy, xyxy, conf, cls = det[:4]
                    track_id = None
                    for xy_,xyxy_,conf_,cls_ in zip(xy, xyxy, conf, cls):
                        x1,y1,x2,y2 = xyxy_
                        if isinstance(xy_,np.ndarray):
                            xy_ = xy_.tolist()
                        w = x2 - x1
                        h = y2 - y1
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0

                        det_dict = {
                            "xy": xy_,
                            "conf": float(conf_),
                            "class_id": int(cls_),
                            "class_name": names[int(class_id)],
                            "bbox": {
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2)
                            }
                        }
                        if track_id is not None:
                            det_dict["track_id"] = track_id
                        if model_code is not None:
                            det_dict["model_code"] = str(model_code)
                        detections_list.append(det_dict)
            else:
                # DET模式：解析水平框参数
                if with_tracking and len(det) >= 7:
                    x1, y1, x2, y2, conf, class_id, cxn, cyn, w, h, track_id = det[:11]
                    track_id = int(track_id) if track_id >= 0 else None
                else:
                    x1, y1, x2, y2, conf, class_id, cxn, cyn, w, h = det[:10]
                    track_id = None
                # print("ago: ",w,h)
                w = x2 - x1
                h = y2 - y1
                # print("now ",w,h)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                det_dict = {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "cx": float(cx),
                    "cy": float(cy),
                    "cxn": float(cxn),
                    "cyn":float(cyn),
                    "w": float(w),
                    "h": float(h),
                    "conf": float(conf),
                    "class_id": int(class_id),
                    "class_name": names[int(class_id)],
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2)
                    }
                }
                if track_id is not None:
                    det_dict["track_id"] = track_id
                    # 算法69：只对类别0（car）添加相对速度，相对速度不存在则不添加速度字段
                    if track_speeds is not None and track_id >= 0 and int(class_id) == 0:
                        # 先按照当前task_id获取相对速度字典，再根据track_id获取具体速度
                        task_speeds = track_speeds.get(TASK_ID, {})
                        speed_kmh = task_speeds.get(int(track_id), None)
                        if speed_kmh is not None:
                            det_dict["speed_kmh"] = int(speed_kmh)
                if model_code is not None:
                    det_dict["model_code"] = str(model_code)
                detections_list.append(det_dict)

        detection_json = json.dumps(detections_list, ensure_ascii=False)

        return jpeg_bytes, detection_json

    @staticmethod
    def encode_with_merged_detection_info(
        gpu_tensor: torch.Tensor,
        merged_detections: list,
        quality: int = 95,
        with_tracking: bool = False,
        track_speeds: dict = None,
    ) -> tuple:
        """
        处理合并的多模型检测结果

        Args:
            gpu_tensor: GPU上的图像tensor
            merged_detections: 合并后的检测结果列表
                [{'model_code': str, 'task': str, 'results': tensor}, ...]
            quality: JPEG质量
            with_tracking: 是否包含跟踪ID（默认False）
            track_speeds: 实时车速字典 {track_id: speed_kmh}

        Returns:
            (jpeg_bytes, detection_json_str)
        """
        # 转为 numpy（GPU->CPU 拷贝只发生一次）
        if isinstance(gpu_tensor, torch.Tensor):
            image_np = gpu_tensor.cpu().numpy()
        else:
            image_np = np.asarray(gpu_tensor)

        # JPEG 编码

        jpeg_bytes = FastImageEncoder.encode_gpu_to_jpeg(image_np, quality)

        # 处理所有模型的检测结果
        detections_list = []
        for model_det in merged_detections:
            model_code = model_det['model_code']
            task = model_det['task']
            results = model_det['results']
            names = model_det['names']
            # print(f"LOOK THIS: {names}")
            # 转换检测结果格式
            if task == 'SAM3':
                # SAM3模式：解析 Results 对象
                results_obj = results[0] if isinstance(results, list) else results

                if hasattr(results_obj, 'boxes') and results_obj.boxes is not None:
                    boxes_xyxy = results_obj.boxes.xyxy  # (N, 4) - [x1, y1, x2, y2]

                    # 获取类别名称映射
                    sam3_names = results_obj.names if hasattr(results_obj, 'names') else {}
                    if isinstance(sam3_names, dict):
                        class_id_to_name = sam3_names
                    else:
                        class_id_to_name = {i: name for i, name in enumerate(sam3_names)} if sam3_names else {}

                    # 遍历每个检测框
                    for idx, box_coords in enumerate(boxes_xyxy):
                        try:
                            x1, y1, x2, y2 = [float(v) for v in box_coords]

                            # 计算宽高和中心点
                            w = x2 - x1
                            h = y2 - y1
                            cx = (x1 + x2) / 2.0
                            cy = (y1 + y2) / 2.0

                            # 获取置信度（如果存在）
                            confidence = 1.0
                            if hasattr(results_obj.boxes, 'conf') and results_obj.boxes.conf is not None:
                                confidence = float(results_obj.boxes.conf[idx])

                            # 获取类别ID
                            class_id = 0
                            if hasattr(results_obj.boxes, 'cls') and results_obj.boxes.cls is not None:
                                class_id = int(results_obj.boxes.cls[idx])

                            # 获取类别名称
                            class_name = class_id_to_name.get(class_id, f"class_{class_id}")

                            # 构建检测信息字典
                            det_dict = {
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2),
                                "cx": float(cx),
                                "cy": float(cy),
                                "w": float(w),
                                "h": float(h),
                                "conf": float(confidence),
                                "class_id": int(class_id),
                                "class_name": class_name,
                                "bbox": {
                                    "x1": float(x1),
                                    "y1": float(y1),
                                    "x2": float(x2),
                                    "y2": float(y2)
                                },
                                "model_code": model_code,  # 添加模型标识
                                "task": task,
                            }

                            detections_list.append(det_dict)

                        except Exception as e:
                            print(f"[SAM3-Merged] 处理检测框 {idx} 失败: {e}")
                            continue
                continue

            if task != 'segment':
                detections = results[0].detach().cpu().numpy()
            else:
                detections = results

            # 处理每个检测结果
            for det in detections:
                if task == 'obb':
                    # OBB模式：解析旋转框参数
                    if with_tracking and len(det) >= 8:
                        cx, cy, w, h, angle, conf, class_id, track_id = det[:8]
                        track_id = int(track_id) if track_id >= 0 else None
                    else:
                        cx, cy, w, h, angle, conf, class_id = det[:7]
                        track_id = None

                    polygon = compute_obb_polygon(cx, cy, w, h, angle)
                    x1, y1, x2, y2 = polygon_to_aabb(polygon)

                    det_dict = {
                        "cx": float(cx),
                        "cy": float(cy),
                        "w": float(w),
                        "h": float(h),
                        "angle": float(angle),
                        "conf": float(conf),
                        "class_id": int(class_id),
                        "class_name": names[int(class_id)],
                        "polygon": [{"x": float(px), "y": float(py)} for px, py in polygon],
                        "bbox": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2)
                        },
                        "model_code": model_code,  # 添加模型标识
                        "task": task,
                    }
                    if track_id is not None:
                        det_dict["track_id"] = track_id
                    detections_list.append(det_dict)

                elif task == 'segment':
                    # segment模式：解析分割结果
                    if with_tracking and len(det) >= 5:
                        xy, xyxy, conf, cls, track_id = det[:5]
                        for xy_, xyxy_, conf_, cls_, track_id_ in zip(xy, xyxy, conf, cls, track_id):
                            x1,y1,x2,y2 = xyxy_
                            if isinstance(xy_, np.ndarray):
                                xy_ = xy_.tolist()
                            w = x2 - x1
                            h = y2 - y1
                            cx = (x1 + x2) / 2.0
                            cy = (y1 + y2) / 2.0
                            track_id = int(track_id_) if track_id_ >= 0 else None
                            class_id = cls_
                            det_dict = {
                                "xy": xy_,
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2),
                                "cx": float(cx),
                                "cy": float(cy),
                                "w": float(w),
                                "h": float(h),
                                "conf": float(conf_),
                                "class_id": int(class_id),
                                # "class_name": CLASS_NAMES.get(int(class_id), f"Class_{int(class_id)}"),
                                "bbox": {
                                    "x1": float(x1),
                                    "y1": float(y1),
                                    "x2": float(x2),
                                    "y2": float(y2)
                                },
                                "model_code": model_code,  # 添加模型标识
                                "task": task,
                            }
                            if track_id is not None:
                                det_dict["track_id"] = track_id
                            detections_list.append(det_dict)
                    else:
                        xy, xyxy, conf, cls = det[:4]
                        track_id = None
                        for xy_, xyxy_, conf_, cls_ in zip(xy, xyxy, conf, cls):
                            x1,y1,x2,y2 = xyxy_
                            if isinstance(xy_, np.ndarray):
                                xy_ = xy_.tolist()
                            w = x2 - x1
                            h = y2 - y1
                            cx = (x1 + x2) / 2.0
                            cy = (y1 + y2) / 2.0

                            det_dict = {
                                "xy": xy_,
                                "conf": float(conf_),
                                "class_id": int(cls_),
                                "class_name": names[int(cls_)],
                                "bbox": {
                                    "x1": float(x1),
                                    "y1": float(y1),
                                    "x2": float(x2),
                                    "y2": float(y2)
                                },
                                "model_code": model_code,  # 添加模型标识
                                "task": task,
                            }
                            if track_id is not None:
                                det_dict["track_id"] = track_id
                            detections_list.append(det_dict)

                else:
                    # DET模式：解析水平框参数
                    cxn = cyn = None  # 初始化变量

                    if len(det) == 6:
                        # 无跟踪格式： [x1, y1, x2, y2, conf, class_id]
                        x1, y1, x2, y2, conf, class_id = det[:6]
                        track_id = None
                    elif len(det) == 7:
                        # 7元素格式： [x1, y1, x2, y2, conf, class_id, track_id]
                        x1, y1, x2, y2, conf, class_id, track_id = det[:7]
                        track_id = int(track_id) if with_tracking and track_id >= 0 else None
                    elif len(det) == 11:
                        # 11元素格式： [x1, y1, x2, y2, conf, class_id, cxn, cyn, w, h, track_id]
                        x1, y1, x2, y2, conf, class_id, cxn, cyn, w, h, track_id = det[:11]
                        track_id = int(track_id) if with_tracking and track_id >= 0 else None
                    else:
                        # 异常长度，使用前6个元素，尝试从最后获取track_id
                        x1, y1, x2, y2, conf, class_id = det[:6]
                        track_id = int(det[-1]) if with_tracking and len(det) > 6 and det[-1] >= 0 else None

                    w = x2 - x1
                    h = y2 - y1
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    # print(names)

                    det_dict = {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "cx": float(cx),
                        "cy": float(cy),
                        "w": float(w),
                        "h": float(h),
                        "conf": float(conf),
                        "class_id": int(class_id),
                        "class_name": names[int(class_id)],
                        "bbox": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2)
                        },
                        "model_code": model_code,  # 添加模型标识
                        "task": task,
                    }

                    # 在11元素格式时添加归一化坐标
                    if cxn is not None and cyn is not None:
                        det_dict["cxn"] = float(cxn)
                        det_dict["cyn"] = float(cyn)
                    if track_id is not None:
                        det_dict["track_id"] = track_id
                        # 算法69：只对类别0（car）添加相对速度，相对速度不存在则不添加速度字段
                        if track_speeds is not None and track_id is not None and int(class_id) == 0:
                            # 先按照当前task_id获取相对速度字典，再根据track_id获取具体速度
                            task_speeds = track_speeds.get(TASK_ID, {})
                            speed_kmh = task_speeds.get(int(track_id), None)
                            if speed_kmh is not None:
                                det_dict["speed_kmh"] = int(speed_kmh)
                    detections_list.append(det_dict)

        detection_json = json.dumps(detections_list, ensure_ascii=False)

        return jpeg_bytes, detection_json

def init_llm_api():
    """构造 OpenAI 兼容客户端（DashScope base_url）；缺 Key 返回 None，LLM 分支自动跳过。"""
    try:
        api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print(
                "[LLM] 请导出 DASHSCOPE_API_KEY 或 OPENAI_API_KEY"
            )
            return None
        client = OpenAI(
            api_key=api_key,
            base_url="https://aigw.telecomjs.com/v1",
        )
        print(f"初始化成功:{client}")
        return client

    except Exception as e:
        print(f"大模型API加载失败，{e}")
        return None


class BoundedExecutor:
    """带并发上限的线程池：信号量满时 **非阻塞丢弃** 任务，防止请求堆积。"""

    def __init__(self, max_workers, max_queue_size):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = threading.Semaphore(max_queue_size)

    def submit(self, fn, *args, **kwargs):
        if not self.semaphore.acquire(blocking=False):
            print("队列满，丢弃任务")
            return
        def wrapper():
            try:
                fn(*args, **kwargs)
            finally:
                self.semaphore.release()
        self.executor.submit(wrapper)


# ============================================================
# 图片获取：MinIO / HTTP
# ============================================================

def _init_minio():
    """惰性初始化 MinIO 客户端（首次图片下载时创建）。"""
    global MINIO_CLIENT
    if MINIO_CLIENT is None:
        try:
            MINIO_CLIENT = init_minio_client(YAML_PATH)
            print("[MinIO] 客户端初始化成功")
        except Exception as e:
            print(f"[MinIO] 客户端初始化失败：{e}")
            MINIO_CLIENT = None
    return MINIO_CLIENT

# data 就是你从minio拿到的 bytes: data = resp.read()

def resize_image_to_4k(image_bytes: bytes) -> bytes:
    """
    将图片二进制resize到标准4K 3840x2160 (16:9)
    返回处理后的图片二进制(jpg格式)
    """
    # 字节流转PIL Image对象
    input_stream = io.BytesIO(image_bytes)
    img = Image.open(input_stream)

    print("zzzzzzzzzzzzzzzzzzzzzzzz")
    print(img.size)

    target_w, target_h = 4032, 3024

    # 方式1：直接拉伸到4K（会改变原图比例，不推荐）
    # resized_img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # 方式2：保持原图比例缩放，长边适配，不会拉伸变形【推荐】
    img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    resized_img = img

    resized_img = resized_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    print("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY")
    print(img.size)

    # 输出为bytes，quality控制jpg质量
    output_stream = io.BytesIO()
    # 如果原图是png，想输出png把format改成"PNG"
    resized_img.save(output_stream, format="JPEG", quality=85, optimize=True)
    out_bytes = output_stream.getvalue()

    print(f"resize后大小: {len(out_bytes)} bytes, 尺寸:{resized_img.size}")
    return out_bytes

def download_image_bytes(object_key=None, bucket=None, image_url=None):
    """获取图片字节流。

    优先级：
        - 若提供 ``object_key``：从 MinIO ``get_object`` 读取（``bucket`` 缺省时使用 ``alarm-images``）。
        - 否则若提供 ``image_url``：HTTP(S) 下载。

    Returns:
        bytes（图片原始字节）；失败返回 None。
    """
    # 1) MinIO 对象
    if object_key:
        client = _init_minio()
        if client is None:
            print("[图片下载] MinIO 客户端不可用")
            return None
        try:
            bucket_name = bucket or "alarm-images"
            resp = client.get_object(bucket_name, object_key)
            data = resp.read()
            print("BBBBBBBBBBBBBBBBBBBBBBBBBB")
            print(len(data))
            

            data = resize_image_to_4k(data)
            print("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
            print(len(data))

            resp.close()
            resp.release_conn()
            print(f"[图片下载] MinIO 成功: {bucket_name}/{object_key}, {len(data)} bytes")
            return data
        except Exception as e:
            print(f"[图片下载] MinIO 失败: {e}")
            return None

    # 2) HTTP URL
    if image_url:
        try:
            req = urllib.request.Request(image_url, headers={"User-Agent": "uav-image-infer/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            print(f"[图片下载] HTTP 成功: {image_url}, {len(data)} bytes")
            return data
        except Exception as e:
            print(f"[图片下载] HTTP 失败: {e}")
            return None

    print("[图片下载] 未提供 object_key 或 image_url")
    return None


def decode_image(image_bytes):
    """解码图片字节为 BGR ndarray (H, W, 3) uint8。失败返回 None。"""
    if not image_bytes:
        return None
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        print("[图片解码] cv2.imdecode 返回 None")
    return img


# ============================================================
# 图片推理主逻辑
# ============================================================

def _run_single_inference_image(model_code, model_info, enable_tracking_flag,
                                img_rgb_tensor, img_bgr_np):
    """单模型图片推理函数（与 DeepStream probe 的 ``run_single_inference`` 结果结构一致）。

    Args:
        img_rgb_tensor: RGB torch tensor (H, W, 3) uint8，已置于模型目标 GPU；供 TRT 引擎使用。
        img_bgr_np: BGR numpy (H, W, 3) uint8；供 SAM3 ``set_image`` 与 JPEG 编码使用。
    """
    selected_engine = model_info['engine']
    actual_gpu_id = model_info['gpu_id']
    second_review_flag = ALGORITHM_SECOND_REVIEW.get(str(model_code), 0)
    local_conf = CONF_THRESHOLD

    try:
        if model_info.get('model_type') == "SAM3":
            # SAM3 直接使用 BGR 图像
            selected_engine.set_image(img_bgr_np)
            text_prompt = model_info.get("text_prompt", "car")
            results = selected_engine(text=[text_prompt])
            num_detections = results[0].boxes.shape[0] if hasattr(results[0], 'boxes') else 0
            return {
                'model_code': model_code,
                "second_review": second_review_flag,
                'results': results,
                'task': "SAM3",
                'num_detections': num_detections,
                'success': True
            }

        # TRT 引擎
        # 根据配置选择使用跟踪或普通检测
        if enable_tracking_flag:
            results, names = selected_engine.track(
                img_rgb_tensor,
                input_size=selected_engine.input_size,
                conf_threshold=local_conf,
                iou_threshold=IOU_THRESHOLD,
                return_torch=True,
                tracker=TRACKER_CONFIG,
                persist=True
            )
        else:
            results, names = selected_engine(
                img_rgb_tensor,
                input_size=selected_engine.input_size,
                conf_threshold=local_conf,
                iou_threshold=IOU_THRESHOLD,
                return_torch=True
            )

        # 获取检测数量
        if isinstance(results[0], list):
            num_detections = len(results[0])
        else:
            num_detections = results[0].shape[0] if results[0].shape[0] > 0 else 0

        # 保存结果
        return {
            'model_code': model_code,
            "second_review": second_review_flag,  # 二次复审标志
            'results': results,
            "names": names,
            'task': selected_engine.task,
            'num_detections': num_detections,
            'success': True
        }

    except Exception as e:
        print(f"[图片推理] 模型{model_code}推理失败：{e}")
        traceback.print_exc()
        return {
            'model_code': model_code,
            "second_review": second_review_flag,
            'results': None,
            "names": None,
            'task': None,
            'num_detections': 0,
            'success': False
        }


def process_image_task(img_bgr_np, task_id, algo_list=None, force=False, source_object_key=None):
    """单张图片检测主流程：解码 → 多模型推理 → 算法69 融合 → 写 Redis + publish。

    与原 ``frame_inference_probe`` 的推理/上报段保持等价：
        - 按 GPU 分组串行推理（ByteTrack 需帧序，单张图片场景同帧内串行即可）。
        - 多算法合并走 ``encode_with_merged_detection_info``，单算法走 ``encode_with_detection_info``。
        - 算法 69 融合分割+检测，计算相对车速。
        - LLM 路径：若 ``LLM_PROMPT`` 非空，异步提交 ``_request_llm_and_upload``。

    Args:
        img_bgr_np: BGR ndarray (H, W, 3) uint8。
        task_id: 任务/设备标识。
        force: True 时绕过 3s 节流（手动拍照场景建议 True）。

    Returns:
        True 表示有检测结果并已上报；False 表示无检测或被节流。
    """
    global COUNT_FRAMES, TRACK_ABSOLUTE_SPEEDS_DS, TRACK_RELATIVE_SPEEDS_DS

    if img_bgr_np is None:
        print("[图片推理] 图片为空，跳过")
        return False

    if source_object_key:
        print(f"[图片推理] ===== 开始推理照片: {source_object_key} =====")
        print(f"[TRACE][{str(source_object_key).rsplit('/',1)[-1]}] stage=6_inference_start ts={int(time.time()*1000)}")

    COUNT_FRAMES += 1
    frame_start_time = time.time()

    with ENGINE_LOCK:
        local_model_dict = dict(MODEL_DICT) if MODEL_DICT else {}
        local_conf = CONF_THRESHOLD

    # 按指定算法筛选（兼容对象数组[{algorithmNum}]与纯字符串数组["编号"]）
    if algo_list:
        algo_set = {str(a.get("algorithmNum")) if isinstance(a, dict) else str(a) for a in algo_list}
        local_model_dict = {k: v for k, v in local_model_dict.items() if k in algo_set}

    if not local_model_dict:
        # 纯 LLM 模式：无检测模型但有 LLM prompt 时，走纯 LLM 实时识别
        if LLM_PROMPT:
            try:
                for pk, llm_config in LLM_PROMPT.items():
                    prompt_text = "请分析这张图片"
                    prompts_raw = llm_config.get("prompts")
                    if prompts_raw:
                        prompts_list = json.loads(prompts_raw) if isinstance(prompts_raw, str) else prompts_raw
                        if isinstance(prompts_list, list) and len(prompts_list) > 0:
                            prompt_text = prompts_list[0].get("prompt", prompt_text)
                    LLM_THREAD_POOL.submit(_request_llm_and_upload, TASK_ID, img_bgr_np, prompt_text, source_object_key)
                    break  # 单张图片仅用第一个 prompt
                return True
            except Exception as e:
                print(f"[图片 #{COUNT_FRAMES}] 纯 LLM 请求失败：{e}")
                return False
        print(f"[图片 #{COUNT_FRAMES}] 推理引擎未初始化，跳过")
        return False

    height, width = img_bgr_np.shape[:2]

    # 预备 RGB tensor（TRT 引擎）—— 置于 cuda:0，run_single_inference 内会按需迁移
    try:
        img_rgb_np = cv2.cvtColor(img_bgr_np, cv2.COLOR_BGR2RGB)
        img_rgb_tensor = torch.from_numpy(img_rgb_np).to(torch.device("cuda:0"))
    except Exception as e:
        print(f"[图片推理] tensor 构造失败：{e}")
        return False

    # 用于 JPEG 编码 / SAM3 的 BGR tensor（置于 cuda:0，FastImageEncoder 会 .cpu().numpy()）
    img_bgr_tensor = torch.from_numpy(img_bgr_np).to(torch.device("cuda:0"))

    t2 = time.time()
    total_num_detections = 0

    # ========== 按 GPU 分组串行推理 ==========
    # 将模型按 GPU 分组，同一 GPU 上的模型串行执行，不同 GPU 上的模型并行执行
    gpu0_models = []
    gpu1_models = []
    # print(MODEL_DICT)
    for model_code, model_info in local_model_dict.items():
        # 根据 enable_tracking 标志和模型类型决定推理方式
        enable_tracking_flag = model_info.get('enable_tracking', ENABLE_TRACKING)
        # 分配到对应 GPU 队列
        if model_info['gpu_id'] == 0:
            gpu0_models.append((model_code, model_info, enable_tracking_flag))
        else:
            gpu1_models.append((model_code, model_info, enable_tracking_flag))

    # ========== 执行串行推理（当前帧） ==========
    # 注：由于跟踪器需要保持帧顺序，暂时使用串行执行
    # 后续可根据需要启用真正的并行（使用 ThreadPoolExecutor）
    all_detections = []

    # GPU0 上的模型（串行执行）
    for model_code, model_info, enable_tracking_flag in gpu0_models:
        result = _run_single_inference_image(model_code, model_info, enable_tracking_flag,
                                             img_rgb_tensor, img_bgr_np)
        if result['success']:
            all_detections.append(result)
            total_num_detections += result['num_detections']

    # GPU1 上的模型（串行执行）
    for model_code, model_info, enable_tracking_flag in gpu1_models:
        result = _run_single_inference_image(model_code, model_info, enable_tracking_flag,
                                             img_rgb_tensor, img_bgr_np)
        if result['success']:
            all_detections.append(result)
            total_num_detections += result['num_detections']

    for _r in all_detections:
        _n = _r.get('num_detections', 0)
        print(f"[模型输出] 算法={_r['model_code']} task={_r['task']} "
              f"检测数={_n} second_review={_r.get('second_review')}")
        if _n > 0 and _r.get('task') in ('detect', 'obb'):
            _results = _r.get('results') or []
            if not _results:
                continue
            _t = _results[0]
            for _i, _d in enumerate(_t):  # ← 去掉 [:5]，打印全部
                print(f"    raw[{_i}] {_d.tolist()}")

    inference_total_time = (time.time() - t2) * 1000

    # ========== 算法69：车道判断和相对速度调整 ==========
    if ALGORITHMNUM == "69" and total_num_detections > 0:
        try:
            # 收集分割和检测结果
            vehicle_data = []
            segment_data = []  # 初始化segment_data

            for det_info in all_detections:
                model_code = det_info['model_code']
                results = det_info['results']
                task = det_info['task']

                if model_code == "69" and task == 'segment':
                    # 收集分割结果 - 从tensor格式数据中提取轮廓点位
                    segment_data = []

                    # 处理tensor格式的分割数据 results[0] = [xy, xyxy, conf, cls, track_id]
                    if len(results) > 0 and len(results[0]) >= 4:
                        try:
                            detections = results[0]  # [xy, xyxy, conf, cls, track_id]

                            if len(detections) >= 4:
                                xy_list, xyxy_list, conf_list, cls_list = detections[:4]

                                # 遍历每个检测结果
                                num_detections = len(xy_list) if hasattr(xy_list, '__len__') else 0

                                for i in range(num_detections):
                                    try:
                                        # 获取xy坐标
                                        xy_coords = xy_list[i] if i < len(xy_list) else None
                                        if xy_coords is not None:
                                            if isinstance(xy_coords, np.ndarray):
                                                xy_coords = xy_coords.tolist()
                                            elif hasattr(xy_coords, 'tolist'):
                                                xy_coords = xy_coords.tolist()

                                            # 获取class_id
                                            class_id = 0  # 默认值
                                            if i < len(cls_list):
                                                cls_val = cls_list[i]
                                                class_id = int(cls_val.item() if hasattr(cls_val, 'item') else cls_val)

                                            # 构造segment_item
                                            segment_item = {
                                                "xy": xy_coords,
                                                "class_id": class_id,
                                                "model_code": model_code  # 添加模型代码标识
                                            }
                                            segment_data.append(segment_item)

                                    except Exception as e:
                                        continue

                        except Exception as e:
                            segment_data = []
                elif model_code == "999" and task == 'detect':
                    # 收集车辆检测结果
                    vehicle_data = results[0] if len(results) > 0 else []

            # 进行车道判断和相对速度调整
            if segment_data and vehicle_data is not None:
                try:
                    # 1. 先计算绝对速度
                    task_track_state = TRACK_STATE_DS.get(TASK_ID, {})
                    task_absolute_speeds = TRACK_ABSOLUTE_SPEEDS_DS.get(TASK_ID, {})
                    update_speed_from_det_tensor(vehicle_data, task_track_state, task_absolute_speeds)
                    # 更新绝对速度状态
                    TRACK_STATE_DS[TASK_ID] = task_track_state
                    TRACK_ABSOLUTE_SPEEDS_DS[TASK_ID] = task_absolute_speeds

                    # 2. 使用绝对速度计算相对速度
                    relative_speeds = process_vehicle_speeds_with_lanes(
                        segment_data, vehicle_data, task_absolute_speeds,  # 传入绝对速度数据
                        TASK_ID, REDIS_CLIENT
                    )
                    # 保存相对速度用于OSD显示
                    TRACK_RELATIVE_SPEEDS_DS[TASK_ID] = relative_speeds
                    print(f"[算法69] 计算了 {len(relative_speeds)} 辆车的相对速度")
                except Exception as e:
                    print(f"[算法69] 相对速度计算失败：{e}")
                    TRACK_RELATIVE_SPEEDS_DS[TASK_ID] = {}
        except Exception as e:
            print(f"[图片 #{COUNT_FRAMES}] [算法69调试] 速度调整失败：{e}")

    # ========== 上传到 Redis ==========
    upload_time = 0
    if total_num_detections > 0:
        try:
            t3 = time.time()

            # 收集所有有检测结果的模型
            valid_detections = [det_info for det_info in all_detections if det_info['num_detections'] > 0]

            if len(local_model_dict) == 1:
                # 单模型算法：直接检查唯一的模型是否有检测结果
                if all_detections[0]['num_detections'] > 0:
                    det_info = all_detections[0]
                    # 获取该算法的二次复审标志
                    second_review_flag = ALGORITHM_SECOND_REVIEW.get(det_info['model_code'], 0)
                    UPLOAD_THREAD_POOL.submit(
                        fast_upload_to_redis,
                        TASK_ID,
                        img_bgr_tensor.detach(),
                        det_info['results'],
                        det_info['task'],
                        ENABLE_TRACKING,
                        det_info['model_code'],
                        det_info['names'],
                        second_review_flag,
                        force,  # 绕过节流
                        source_object_key,
                    )
            else:
                # 多模型算法：合并所有检测结果并上报
                if len(valid_detections) > 0:
                    merged_results = []
                    merged_second_review_flags = {}
                    for det_info in valid_detections:
                        merged_results.append({
                            'model_code': det_info['model_code'],
                            'task': det_info['task'],
                            'results': det_info['results'],
                            "names": det_info['names']
                        })
                        # 收集每个模型的二次复审标志
                        merged_second_review_flags[det_info['model_code']] = ALGORITHM_SECOND_REVIEW.get(det_info['model_code'], 0)

                    # print(f"合并后的检测结果：{merged_results}")

                    # 多模型合并时，如果任一算法开启二次复审则上报 1
                    any_second_review = 1 if any(merged_second_review_flags.values()) else 0
                    UPLOAD_THREAD_POOL.submit(
                        fast_upload_to_redis,
                        TASK_ID,
                        img_bgr_tensor.detach(),
                        merged_results,
                        'merged',
                        ENABLE_TRACKING,
                        None,
                        None,
                        any_second_review,
                        force,
                        source_object_key,
                    )
            upload_time = (time.time() - t3) * 1000
        except Exception as e:
            print(f"[图片 #{COUNT_FRAMES}] 上传失败：{e}")

    # ========== LLM 路径：手动拍照也走一次实时识别 ==========
    try:
        if LLM_PROMPT:
            for pk, llm_config in LLM_PROMPT.items():
                prompt_text = "请分析这张图片"
                prompts_raw = llm_config.get("prompts")
                if prompts_raw:
                    prompts_list = json.loads(prompts_raw) if isinstance(prompts_raw, str) else prompts_raw
                    if isinstance(prompts_list, list) and len(prompts_list) > 0:
                        prompt_text = prompts_list[0].get("prompt", prompt_text)
                # 提交到线程池异步执行（每帧只提交一次，避免重复事件）
                LLM_THREAD_POOL.submit(_request_llm_and_upload, TASK_ID, img_bgr_np, prompt_text) 
                break  # 单张图片仅用第一个 prompt
    except Exception as e:
        print(f"[图片 #{COUNT_FRAMES}] LLM 请求失败：{e}")

    total_time = (time.time() - frame_start_time) * 1000
    print(f"[图片 #{COUNT_FRAMES}] 性能统计 | "
          f"尺寸：{width}x{height} | "
          f"检测数：{total_num_detections} | "
          f"总耗时：{total_time:.2f}ms | "
          f"推理：{inference_total_time:.2f}ms | "
          f"上传：{upload_time:.2f}ms | "
          f"模型数：{len(all_detections)}/{len(local_model_dict)} | "
          f"模型编号：{ALGORITHMNUM}")

    return total_num_detections > 0


def fast_upload_to_redis(task_id: str, gpu_tensor, detections_np, task: str = 'detect',
                          with_tracking: bool = False, model_code: str = None, names: dict = None,
                          second_review_flag: int = 0, force: bool = False, source_object_key: str = None):
    """将当前帧 JPEG 与检测列表写入 Redis，并向频道 ``alarm-images`` 发布 **metadata 的 key**。

    存储形态:
        - ``image:{task_id}:{ts}`` → JPEG bytes（TTL 600s）
        - ``metadata:{task_id}:{ts}`` → JSON（含 ``image_key``、``detections``、``TASK``、``second_review`` 等）
        - Pub/Sub payload = metadata 的 Redis key 字符串（消费者再 GET）

    Args:
        task: ``detect`` / ``obb`` / ``segment`` / ``SAM3`` / ``merged``；合并模式走 ``encode_with_merged_detection_info``。
        second_review_flag: 写入 metadata，供下游 redis_result_processor 是否走 LLM 复审。
        force: True 时绕过 ``UPLOAD_INTERVAL`` 节流（手动拍照场景使用），默认 False 保持原行为。
    """
    global LAST_UPLOAD_TIME
    try:
        current_time = time.time()

        if not force and task_id in LAST_UPLOAD_TIME:
            if current_time - LAST_UPLOAD_TIME[task_id] < UPLOAD_INTERVAL:
                return
        track_speeds_for_upload = None
        if ALGORITHMNUM == "69":
            track_speeds_for_upload = TRACK_RELATIVE_SPEEDS_DS
            print(f"[DEBUG] track_speeds_for_upload: {track_speeds_for_upload}")
        if task == 'merged':
            jpeg_bytes, detection_json = FastImageEncoder.encode_with_merged_detection_info(
                gpu_tensor,
                detections_np,
                quality=COMPRESSION_QUALITY,
                with_tracking=with_tracking,
                track_speeds=track_speeds_for_upload
            )
        else:
            jpeg_bytes, detection_json = FastImageEncoder.encode_with_detection_info(
                gpu_tensor,
                detections_np,
                quality=COMPRESSION_QUALITY,
                task=task,
                with_tracking=with_tracking,
                track_speeds=track_speeds_for_upload,
                names = names,
                model_code = model_code,
            )

        if REDIS_CLIENT:
            timestamp = int(time.time() * 1000)

            image_key = f"image:{task_id}:{timestamp}"
            REDIS_CLIENT.client.setex(image_key, 600, jpeg_bytes)
            detections = json.loads(detection_json)
            detect_frame = build_detect_frame_from_detections(detections, default_model_code=model_code)

            metadata = {
                "image_key": image_key,
                "detections": detections,
                "detect_frame": detect_frame,
                "timestamp": timestamp,
                "task_id": task_id,
                "algorithmnum": model_code if model_code is not None else ALGORITHMNUM,
                "filepath": FILEPATH,
                "uavsn": UAVSN,
                "AREA": AREAJTR,
                'TASK': task,
                'model_code': model_code,
                "second_review": second_review_flag,
                "source_object_key": source_object_key or "",
            }

            metadata_key = f"metadata:{task_id}:{timestamp}"
            REDIS_CLIENT.client.setex(
                metadata_key,
                600,
                json.dumps(metadata, ensure_ascii=False)
            )
            print(f"[图片推理] 已写入 Redis 并 publish: {metadata_key}, 源照片: {source_object_key or ''}")

            REDIS_CLIENT.publish(f"alarm-images", metadata_key)
            _src = str(source_object_key).rsplit("/",1)[-1] if source_object_key else str(task_id)
            print(f"[TRACE][{_src}] stage=7_redis_publish ts={int(time.time()*1000)} meta={metadata_key}") 
            LAST_UPLOAD_TIME[task_id] = current_time

    except Exception:
        import traceback
        traceback.print_exc()


def llm_upload_to_redis(task_id: str, img_array, llm_result: str, task: str = 'llm', prompt: str = "", source_object_key=None):
    """LLM 可视化链路专用：JPEG + metadata（含 ``llm_result``、``prompt``），发布键到 ``alarm-images``。"""
    global LAST_LLM_UPLOAD_TIME, LLM_CLIENT 
    t_up = time.time()
    _src = str(source_object_key).rsplit('/', 1)[-1] if source_object_key else str(task_id)
    try:
        current_time = time.time()

        if task_id in LAST_LLM_UPLOAD_TIME:
            if current_time - LAST_LLM_UPLOAD_TIME[task_id] < LLM_UPLOAD_INTERVAL:
                return

        t_p = time.time()
        llm_dets = parse_llm_objects_for_detect_frame(llm_result)
        if not llm_dets:
            print("[LLM] 没有解析到任何目标，跳过 Redis 上报")
            return

        ret, encoded = cv2.imencode('.jpg', img_array,
                                   [cv2.IMWRITE_JPEG_QUALITY, 95])
        height, width = img_array.shape[:2]
        t_enc = time.time()

        detect_frame = build_llm_detect_frame_from_objects(
            llm_dets,
            image_width=width,
            image_height=height,
            emit_boxes=True,
            model_code="llm",
            coordinate_mode="normalized_1000",   # 如果你的 bbox_2d 实际是 0~1000，就改成 "normalized_1000"
        )
        t_df = time.time()

        if not detect_frame.get("boxes"):
            print("[LLM] detect_frame boxes is null, jump Redis publish")
            return

        if REDIS_CLIENT:
            t_r = time.time()
            timestamp = int(time.time() * 1000)

            image_key = f"image:{task_id}:{timestamp}"
            REDIS_CLIENT.client.setex(image_key, 600, encoded.tobytes())

            metadata = {
                "image_key": image_key,
                "timestamp": timestamp,
                "task_id": task_id,
                "filepath": FILEPATH,
                "uavsn": UAVSN,
                "prompt": prompt,
                "TASK": task,
                "llm_result": llm_result,
                "detect_frame": detect_frame,
                "source_object_key": source_object_key or "",
            }

            metadata_key = f"metadata:{task_id}:{timestamp}"
            REDIS_CLIENT.client.setex(
                metadata_key,
                600,
                json.dumps(metadata, ensure_ascii=False)
            )

            REDIS_CLIENT.publish(f"alarm-images", metadata_key)
            LAST_LLM_UPLOAD_TIME[task_id] = current_time
            t_r_end = time.time()

            print(f"[LLM-TIME] [{_src}] stage=upload细分 parse+jpg编码={(t_enc - t_p) * 1000:.0f}ms detect_frame构建={(t_df - t_enc) * 1000:.0f}ms redis写入+publish={(t_r_end - t_r) * 1000:.0f}ms | upload合计={(t_r_end - t_up) * 1000:.0f}ms")
            print(f"[LLM] 已上传 LLM 结果到 Redis: {metadata_key}")

    except Exception:
        import traceback
        print(f"[LLM] 上传失败：{traceback.format_exc()}")


def _request_llm_and_upload(task_id: str, img_array, prompt: str, source_object_key=None):
    """线程池任务：``realtime_llm_infer`` → 成功则 ``llm_upload_to_redis``。"""
    global LLM_CLIENT, AVA_LLM_INFER_TIME, COUNT_FRAMES
    t_total = time.time()
    _src = str(source_object_key).rsplit('/', 1)[-1] if source_object_key else str(task_id)
    try:
        if not LLM_CLIENT:
            print("[LLM] 客户端未初始化")
            return

        if img_array.dtype != np.uint8:
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)

        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

        t0 = time.time()
        ret, img_bytes = cv2.imencode('.jpg', img_array)
        if not ret:
            print("[LLM] 图片编码失败")
            return
        img_b64 = base64.b64encode(img_bytes.tobytes()).decode("utf-8")
        img_data_url = f"data:image/jpeg;base64,{img_b64}"
        t1 = time.time()
        print(f"[LLM-TIME] [{_src}] stage=encode_jpg_b64 cost={(t1 - t0) * 1000:.0f}ms b64={len(img_b64) // 1024}KB")

        success, llm_result, spend_time = realtime_llm_infer(LLM_CLIENT, img_data_url, prompt)
        AVA_LLM_INFER_TIME += spend_time
        t2 = time.time()
        print(f"[LLM-TIME] [{_src}] stage=llm_request(网关请求) cost={(t2 - t1) * 1000:.0f}ms")
        if success and llm_result:
            print(f"[LLM] 请求成功：{llm_result}")
            objects = parse_llm_objects_for_detect_frame(llm_result)
            t3 = time.time()
            print(f"[LLM-TIME] [{_src}] stage=parse_objects count={len(objects)} cost={(t3 - t2) * 1000:.0f}ms")
            if not objects:
                print("[LLM] 没有解析到任何目标，跳过上报")
                return
            llm_upload_to_redis(task_id, img_array, llm_result, prompt=prompt, source_object_key=source_object_key)
            t4 = time.time()
            print(f"[LLM-TIME] [{_src}] stage=upload_to_redis cost={(t4 - t3) * 1000:.0f}ms")
            print(f"[LLM-TIME] [{_src}] stage=SUBPROCESS_TOTAL(编码→上报) cost={(t4 - t_total) * 1000:.0f}ms")
        else:
            print(f"[LLM] 请求失败或无结果, 网关耗时{(time.time() - t1) * 1000:.0f}ms")

    except Exception as e:
        print(f"[LLM] 请求异常：{e} 已耗时{(time.time() - t_total) * 1000:.0f}ms")
        import traceback
        traceback.print_exc()


def on_message(msg):
    """Redis 订阅回调：解析 JSON，增量更新算法列表、阈值、``MODEL_DICT`` 与 ``LLM_PROMPT``。

    载荷字段（与 mqtt_redis TaskManager 发布格式对齐）:
        ``start_algorithm_list`` / ``stop_algorithm_list`` / ``start_llm_list`` / ``stop_llm_list``。

    已知注意:
        ``CONF_THRESHOLD`` 更新处若 ``THRESH.values()`` 非纯数值或为空，可能触发异常并被捕获；
        ``min(float(THRESH.values()), 0.45)`` 若写成对 dict_values 直接 float 会失败，二次开发时勿随意改写。

    SAM3:
        停止列表释放引擎时 SAM3 路径调用 ``ReleaseSam3Model``；异常时可能残留显存，宜避免单任务反复挂载卸载 SAM3。
    """
    global MODEL_DICT, CLASS_NAMES, CONF_THRESHOLD, ALGORITHMNUM, FILEPATH, UAVSN, AREAJTR, LLM_PROMPT, ALGORITHM_SECOND_REVIEW

    payload_str = None

    try:
        if isinstance(msg, dict):
            msg_type = msg.get('type')

            if msg_type in ('subscribe', 'psubscribe'):
                print(f"[Redis] 订阅确认: {msg_type}")
                return

            if msg_type in ('unsubscribe', 'punsubscribe', 'pong'):
                return

            data = msg.get('data')
            if data is None:
                print(f"[Redis] 警告: 消息中无数据字段，完整消息: {msg}")
                return

            if isinstance(data, bytes):
                payload_str = data.decode('utf-8')
            else:
                payload_str = str(data)

        elif hasattr(msg, 'payload'):
            p = msg.payload
            if isinstance(p, bytes):
                payload_str = p.decode('utf-8')
            else:
                payload_str = str(p)
        else:
            payload_str = str(msg)

        if not payload_str:
            print("[Redis] 警告: 消息内容为空")
            return

        print(f"[Redis] 收到原始消息: {payload_str[:200]}")

        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError as e:
            print(f"[Redis] 错误: JSON 解析失败: {e}, 原始内容: {payload_str}")
            return

        start_algorithm_list = payload.get("start_algorithm_list", [])
        stop_algorithm_list = payload.get("stop_algorithm_list",[])
        for algo in start_algorithm_list:
            ALGORITHMNUM.append(algo['algorithmNum'])
            ALGORITHM_SECOND_REVIEW[str(algo['algorithmNum'])] = algo['second_review']

            for d in algo['paramers']:
                THRESH.update(d)
        try:
            conf_thresh = min(float(THRESH.values()), 0.45)
            CONF_THRESHOLD = conf_thresh
            print(f"[Redis] 已更新置信度阈值: {CONF_THRESHOLD}")
        except Exception as e:
            print(f"[Redis] 警告: 置信度阈值解析失败: {e}")

        if len(start_algorithm_list)>0:
            try:
                initialize_inference_engine()
                print(f"[Redis] 推理引擎初始化成功")
            except Exception as e:
                print(f"[Redis] 错误: 推理引擎初始化失败: {e}")
                raise
        if len(stop_algorithm_list)>0:
            stop_list = [str(i.get("algorithmNum")) for i in stop_algorithm_list]
            ALGORITHM_SECOND_REVIEW = { str(k): v for k, v in ALGORITHM_SECOND_REVIEW.items() if int(k) not in stop_list }
            engines = list(MODEL_DICT.values()) if MODEL_DICT else []
            ALGORITHMNUM = [i for i in ALGORITHMNUM if str(i) not in stop_list]
            print(f"删除后的ALGORITHMNUM:{ALGORITHMNUM},stoplist:{stop_list},engines:{MODEL_DICT}")
            if len(engines) > 0:
                try:
                    for k, v in list(MODEL_DICT.items()):
                        if k in stop_list:
                            model_now = v.get("engine", None)
                            print(f"待清除的model_now ：{model_now}")
                            if model_now is None:
                                continue
                            if v.get("model_type") == "SAM3":
                                try:
                                    ReleaseSam3Model(model_now)
                                    del MODEL_DICT[k]
                                except Exception as e:
                                    print("SAM3释放失败:", e)
                            else:
                                try:
                                    model_now._del()
                                    del MODEL_DICT[k]
                                    torch.cuda.empty_cache()
                                except:
                                    print(f"模型释放失败: {k}, error: {e}")
                except Exception as e:
                    print(f"[Redis] 错误: 推理引擎清理失败: {e}")

        start_llm_list = payload.get("start_llm_list", None)
        stop_llm_list = payload.get("stop_llm_list", None)

        # 平台 llmPermission：0 无权限（忽略提示词），1 有权限，缺省保持原值（老平台兼容）
        llm_permission = payload.get("llm_permission", None)
        if llm_permission is not None:
            global LLM_PERMISSION
            LLM_PERMISSION = normalize_llm_permission(llm_permission, LLM_PERMISSION)
            print(f"[LLM] llmPermission={llm_permission} -> LLM_PERMISSION={LLM_PERMISSION}")
            if LLM_PERMISSION == 0:
                LLM_PROMPT.clear()
                print("[LLM] llmPermission=0 无权限，已关闭大模型检测（提示词一律忽略）")

        if start_llm_list:
            for llm_config in start_llm_list:
                prompt_key = llm_config.get("prompt", llm_config.get("name", "default"))
                if LLM_PERMISSION == 0:
                    print(f"[LLM] 无大模型权限(llmPermission=0)，跳过启用: {prompt_key}")
                    continue
                if not llm_entry_prompt_text(llm_config):
                    print(f"[LLM] skip entry with empty prompt (platform not configured): {prompt_key}")
                    continue
                LLM_PROMPT[prompt_key] = llm_config
            print(f"[LLM] 新增 LLM 提示词：{list(start_llm_list)}")
            print(f"[LLM] 当前 LLM_PROMPT keys: {list(LLM_PROMPT.keys())}")

        if stop_llm_list:
            for stop_config in stop_llm_list:
                stop_prompt_key = stop_config.get("prompt", stop_config.get("name", "default"))
                if stop_prompt_key in LLM_PROMPT:
                    existing_config = LLM_PROMPT[stop_prompt_key]
                    is_match = all(existing_config.get(k) == v for k, v in stop_config.items())
                    if is_match:
                        del LLM_PROMPT[stop_prompt_key]
                        print(f"[LLM] 删除 LLM 提示词：{stop_prompt_key}")
                    else:
                        print(f"[LLM] 提示词 {stop_prompt_key} 字段不匹配，未删除")
                else:
                    print(f"[LLM] 提示词 {stop_prompt_key} 不存在，跳过删除")
            print(f"[LLM] 当前 LLM_PROMPT keys: {list(LLM_PROMPT.keys())}")

    except Exception as e:
        import traceback
        print(f"[Redis] 严重错误: {e}")
        traceback.print_exc()


# ============================================================
# photo_result 终态反馈（算法→后端）
# ============================================================

def _init_photo_mqtt():
    """惰性初始化用于发布 photo_result 的 MQTT 客户端。

    从当前 profile 配置读取 mqtt.host/port/username/password 与 topics.photo_result，
    创建独立 paho 客户端（clientid 加后缀 ``-photo``，避免与编排侧 clientid 冲突）。
    """
    global _PHOTO_MQTT_CLIENT, _PHOTO_MQTT_TOPIC
    if _PHOTO_MQTT_CLIENT is not None:
        return _PHOTO_MQTT_CLIENT
    try:
        import paho.mqtt.client as _mqtt
        from common.config_loader import load_config as _load
        _cfg = _load()
        _mqtt_cfg = _cfg.get('mqtt', {})
        _topics = _mqtt_cfg.get('topics', {})
        _host = _mqtt_cfg.get('host', '127.0.0.1')
        _port = int(_mqtt_cfg.get('port', 1883))
        _username = _mqtt_cfg.get('username')
        _password = _mqtt_cfg.get('password')
        _clientid = _mqtt_cfg.get('clientid', 'uav_algorithm_control') + '-photo'
        _PHOTO_MQTT_TOPIC = _topics.get('photo_result', 'ai/calculate/photo_result')

        client = _mqtt.Client(client_id=_clientid, clean_session=True)
        if _username:
            client.username_pw_set(_username, _password)
        client.connect(_host, _port, keepalive=60)
        client.loop_start()
        _PHOTO_MQTT_CLIENT = client
        print(f"[图片推理] photo_result MQTT 已连接: {_host}:{_port}, topic={_PHOTO_MQTT_TOPIC}")
    except Exception as e:
        print(f"[图片推理] photo_result MQTT 初始化失败：{e}")
        _PHOTO_MQTT_CLIENT = None
    return _PHOTO_MQTT_CLIENT


def _publish_photo_result(task_id, object_key, status, reason=None):
    """向 ``ai/calculate/photo_result`` 发布 QoS1 终态反馈。

    Args:
        task_id: 与 photo_notify 中的 task_id 一致，唯一关联键
        object_key: 原样回传，仅用于后端校验
        status: ``fail``
        reason: status=fail 时必填：download_404 / download_error / decode_error / inference_exception
    """
    try:
        client = _init_photo_mqtt()
        if client is None:
            print(f"[图片推理] photo_result 发布跳过：MQTT 客户端不可用, status={status}, task_id={task_id}")
            return
        payload = {
            "definition": "photo_result",
            "task_id": str(task_id),
            "object_key": object_key,
            "status": status,
            "reason": reason,
            "timestamp": int(time.time() * 1000),
        }
        info = client.publish(_PHOTO_MQTT_TOPIC, json.dumps(payload, ensure_ascii=False), qos=1)
        print(f"[图片推理] 发布 photo_result: status={status}, task_id={task_id}, rc={info.rc}")
    except Exception as e:
        print(f"[图片推理] 发布 photo_result 异常：{e}")
        traceback.print_exc()


def _on_photo_message(msg):
    """Redis 订阅回调：处理编排侧下发的手动拍照任务。

    载荷（由 mqtt_redis ``_handle_photo_notify`` 发布到 ``photo_task:{task_id}``）:
        {
          "type": "photo",
          "image_url": "http://.../x.jpg",     # 与 object_key 二选一
          "object_key": "drone/DRONE_01/x.jpg", # MinIO 对象 key
          "bucket": "alarm-images",              # 可选，默认 alarm-images
          "task_id": "DRONE_01",
          "timestamp": 1754411160000
        }

    流程: 幂等检查 → 下载图片 → 解码 BGR → ``process_image_task``（force=True 绕过节流）
          → 按终态发布 ``photo_result``（无目标 / 失败），有目标走原有 alarm-images 链路。
    """
    payload_str = None
    _inflight_marked = False
    try:
        if isinstance(msg, dict):
            msg_type = msg.get('type')
            if msg_type in ('subscribe', 'psubscribe'):
                print(f"[图片推理] 订阅确认: {msg_type}")
                return
            if msg_type in ('unsubscribe', 'punsubscribe', 'pong'):
                return
            data = msg.get('data')
            if data is None:
                return
            payload_str = data.decode('utf-8') if isinstance(data, bytes) else str(data)
        elif hasattr(msg, 'payload'):
            p = msg.payload
            payload_str = p.decode('utf-8') if isinstance(p, bytes) else str(p)
        else:
            payload_str = str(msg)

        if not payload_str:
            return

        print(f"[图片推理] 收到拍照任务: {payload_str[:200]}")
        payload = json.loads(payload_str)

        if payload.get("type") != "photo":
            return

        image_url = payload.get("image_url")
        object_key = payload.get("object_key")
        bucket = payload.get("bucket")
        photo_task_id = payload.get("task_id") or TASK_ID

        # ---- 拍照任务去重 ----
        if object_key:
            now_ts = time.time()
            if now_ts - _SEEN_PHOTO_KEYS.get(object_key, 0) < 600:
                print(f"[图片推理] 重复拍照任务，已去重: {object_key}")
                return
            _SEEN_PHOTO_KEYS[object_key] = now_ts
            if len(_SEEN_PHOTO_KEYS) > 2000:
                for k in list(_SEEN_PHOTO_KEYS.keys())[:1000]:
                    _SEEN_PHOTO_KEYS.pop(k, None)
        print(f"[图片推理] 去重通过，将处理: {object_key}")

        _PHOTO_TASK_QUEUE.put({
            "image_url": image_url,
            "object_key": object_key,
            "bucket": bucket,
            "photo_task_id": photo_task_id,
        })
        print(f"[图片推理] 拍照任务已入队: task_id={photo_task_id}, queue_size={_PHOTO_TASK_QUEUE.qsize()}")

        # 如果没有消费者在跑，启动一个
        global _PHOTO_PROCESSING
        if not _PHOTO_PROCESSING:
            _PHOTO_PROCESSING = True
            t = threading.Thread(target=_photo_queue_consumer, daemon=True)
            t.start()

    except Exception as e:
        print(f"[图片推理] 处理拍照任务异常：{e}")
        traceback.print_exc()

def _photo_queue_consumer():
    """队列消费者：逐一处理拍照任务"""
    while True:
        try:
            task = _PHOTO_TASK_QUEUE.get(timeout=5)
        except queue.Empty:
            global _PHOTO_PROCESSING
            _PHOTO_PROCESSING = False
            return

        _process_single_photo(task)
        _PHOTO_TASK_QUEUE.task_done()


def _process_single_photo(task):
    """处理单张图片"""
    image_url = task["image_url"]
    object_key = task["object_key"]
    bucket = task["bucket"]
    photo_task_id = task["photo_task_id"]
    _trace = str(object_key).rsplit("/", 1)[-1]
    print(f"[TRACE][{_trace}] stage=3_photo_task_dequeue ts={int(time.time()*1000)}")

    # 下载图片
    t0 = time.time()
    image_bytes = download_image_bytes(object_key=object_key, bucket=bucket, image_url=image_url)
    if image_bytes:
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print(f"[图片推理] 下载耗时: {time.time()-t0:.2f}s, 大小: {len(image_bytes)} bytes")
        print(f"[TRACE][{_trace}] stage=4_download_done cost={time.time()-t0:.2f}s size={len(image_bytes)}")   
    if image_bytes is None:
        print(f"[图片推理] 图片下载失败: url={image_url}, key={object_key}")
        _publish_photo_result(photo_task_id, object_key, "fail", reason="download_error")
        return

    # 解码
    img_bgr = decode_image(image_bytes)
    if img_bgr is None:
        print("[图片推理] 图片解码失败")
        _publish_photo_result(photo_task_id, object_key, "fail", reason="decode_error")
        return
    print(f"[TRACE][{_trace}] stage=5_decode_done ts={int(time.time()*1000)}")
    # 推理
    has_target = process_image_task(img_bgr, photo_task_id, force=True,
                                    source_object_key=object_key)
    if not has_target:
        print(f"[图片推理] 推理无目标: task_id={photo_task_id}")
    else:
        print(f"[图片推理] 推理有目标，已写入 Redis 并 publish alarm-images: task_id={photo_task_id}")

def main(input_rtsp=None, output_rtmp=None, task_id=None, algorithm_list=None,
         filePath=None, assigned_gpu_id=0, llm_list=None):
    """子进程入口（图片模式）：reload 配置 → 初始化模型与 Redis → 订阅 ``task_id`` 与
    ``photo_task:{task_id}`` → 进入空闲等待循环（由拍照任务驱动推理）。

    Args:
        input_rtsp / output_rtmp: 图片模式下不使用（保留参数以兼容 ``_inference_wrapper`` 调用契约）。
        task_id: Redis 订阅频道名（与设备 SN 一致）；亦为全局 ``TASK_ID``。
        algorithm_list: 启动时的算法列表；空列表则不加载引擎直至 Redis 下发 start。
        filePath / assigned_gpu_id: 传入推理逻辑（路径上报等）。
        llm_list: 初始 LLM 配置列表，写入 ``LLM_PROMPT``。
    """
    global THRESH, CONF_THRESHOLD, CONFIG_ALGORITHMS, CLASS_NAMES, TASK_ID, REDIS_CLIENT, MODEL_DICT, UPLOAD_THREAD_POOL, COUNT_FRAMES, ALGORITHMNUM, FILEPATH, UAVSN, ENABLE_TRACKING, STOP_REQUESTED, TRACK_STATE_DS, TRACK_ABSOLUTE_SPEEDS_DS, TRACK_RELATIVE_SPEEDS_DS
    global LLM_PROMPT, LLM_CLIENT, LLM_THREAD_POOL

    import common.config
    reload(common.config)
    CONFIG_ALGORITHMS = common.config.model_path
    TASK_ID = str(task_id)
    ALGORITHMNUM = []
    UAVSN  = str(task_id)
    FILEPATH = str(filePath)
    THRESH = {}
    LLM_THREAD_POOL = BoundedExecutor(4, 10)
    LLM_PROMPT = {}
    if llm_list:
        for llm_config in llm_list:
            prompt_key = llm_config.get("prompt", llm_config.get("name", "default"))
            if not llm_entry_prompt_text(llm_config):
                print(f"[LLM] skip entry with empty prompt (platform not configured): {prompt_key}")
                continue
            LLM_PROMPT[prompt_key] = llm_config
        print(f"[LLM] 初始化 LLM_PROMPT: {list(LLM_PROMPT.keys())}")
    else:
        print("[LLM] 无 LLM 配置")
    if not algorithm_list:
        print(f"初始无算法运行，等待 Redis 下发 start_algorithm_list")
    else:
        for algo in algorithm_list:
            ALGORITHMNUM.append(algo['algorithmNum'])
            for d in algo['paramers']:
                THRESH.update(d)
        if not THRESH:
            pass
        else:
            CONF_THRESHOLD = min(THRESH.values())
        initialize_inference_engine()

    if TASK_ID not in TRACK_STATE_DS:
        TRACK_STATE_DS[TASK_ID] = {}
    if TASK_ID not in TRACK_ABSOLUTE_SPEEDS_DS:
        TRACK_ABSOLUTE_SPEEDS_DS[TASK_ID] = {}
    if TASK_ID not in TRACK_RELATIVE_SPEEDS_DS:
        TRACK_RELATIVE_SPEEDS_DS[TASK_ID] = {}
    UPLOAD_THREAD_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="upload")

    try:
        print(f"[初始化] 连接 Redis, YAML_PATH={YAML_PATH}")
        REDIS_CLIENT = RedisHelper(YAML_PATH)
        print(f"[初始化] Redis 客户端创建成功")

        # ---- 单实例看门狗：同任务只允许一个推理进程消费 photo_task ----
        _lock_key = f"photo_worker_lock:{task_id}"
        _my_pid = str(os.getpid())
        REDIS_CLIENT.client.set(_lock_key, _my_pid, ex=90)   # 新进程无条件接管

        def _single_instance_watchdog():
            while True:
                time.sleep(20)
                try:
                    v = REDIS_CLIENT.client.get(_lock_key)
                    if v is None:
                        REDIS_CLIENT.client.set(_lock_key, _my_pid, ex=90)
                    elif v.decode() != _my_pid:
                        print(f"[图片推理] 检测到新推理进程(pid={v.decode()})已接管，"
                              f"本进程(pid={_my_pid})退出")
                        os._exit(0)
                    else:
                        REDIS_CLIENT.client.expire(_lock_key, 90)
                except Exception:
                    pass

        threading.Thread(target=_single_instance_watchdog, daemon=True).start()
        print(f"[图片推理] 单实例锁已接管: {_lock_key} -> pid={_my_pid}")
        # ---------------------------------------------------------------

        LLM_CLIENT = init_llm_api()

        # 订阅 1: task_id 频道 —— 算法/LLM 热切换
        def redis_subscribe_worker():
            global REDIS_CLIENT, STOP_REQUESTED
            print(f"[Redis] 订阅热切换频道: {task_id}")
            REDIS_CLIENT.subscribe(str(task_id), on_message)
            print("[Redis] hot-swap subscribe returned, reconnecting...")

        sub_thread = threading.Thread(target=redis_subscribe_worker, daemon=True, name=f"redis-sub-{task_id}")
        sub_thread.start()
        print(f"[初始化] 热切换订阅线程已启动")

        # 订阅 2: photo_task:{task_id} 频道 —— 手动拍照图片任务
        photo_channel = f"photo_task:{task_id}"
        print(f"[初始化] 手动拍照图片任务任务开始订阅：{photo_channel}")
        def redis_photo_worker():
            global REDIS_CLIENT, STOP_REQUESTED
            print(f"[Redis] 订阅拍照任务频道: {photo_channel}")
            REDIS_CLIENT.subscribe(photo_channel, _on_photo_message)
            print("[Redis] photo subscribe returned, reconnecting...")

        photo_thread = threading.Thread(target=redis_photo_worker, daemon=True, name=f"redis-photo-{task_id}")
        photo_thread.start()
        print(f"[初始化] 拍照任务订阅线程已启动")

    except Exception as e:
        print(f"[初始化] 严重错误: Redis 初始化失败: {e}")
        import traceback
        traceback.print_exc()

    # 主循环：图片模式下不再有 GStreamer 管线，进入空闲等待，
    # 由上述两个 Redis 订阅线程驱动（热切换 + 拍照推理）。
    stop_event = threading.Event()

    def _signal_handler(signum, frame):
        print(f"[主] 收到信号 {signum}，准备退出图片推理子进程")
        nonlocal stop_event
        STOP_REQUESTED = True
        stop_event.set()

    import signal as _signal
    _signal.signal(_signal.SIGTERM, _signal_handler)
    _signal.signal(_signal.SIGINT, _signal_handler)

    print(f"[主] 图片推理子进程就绪，task_id={TASK_ID}，等待拍照任务...")

    while not STOP_REQUESTED:
        # 每 30s 打印一次存活心跳，便于 supervisor 看门狗观测
        try:
            stop_event.wait(timeout=30)
        except Exception:
            pass

    print("[主] 图片推理子进程退出")

    try:
        if UPLOAD_THREAD_POOL:
            print("Shutting down upload thread pool...")
            UPLOAD_THREAD_POOL.shutdown(wait=True)
            print("Upload thread pool shut down")
    except Exception:
        pass


# 本地调试：未调用 main()；需传入算法列表与 Redis 初始化时请通过 mqtt_redis 子进程入口。
if __name__ == "__main__":
    input_rtsp = "rtsp://127.0.0.1:8554/live/test"
    output_rtmp = "rtmp://127.0.0.1:1935/live/test_ai"

    if len(sys.argv) > 1:
        input_rtsp = sys.argv[1]
    if len(sys.argv) > 2:
        output_rtmp = sys.argv[2]


