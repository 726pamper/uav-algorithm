"""
DeepStream + TensorRT 实时推理子进程入口（与 mqtt_redis 编排侧配合）。

职责概要
--------
- 构建 GStreamer 管线：拉流 → 硬解 → nvstreammux → RGBA(NVMM) → 在 ``inference_pad`` 上挂 Pad Probe
  做推理与 OSD → 编码 → 推流（RTMP / RTSP / GB28181 风格 UDP RTP）。
- 通过全局 ``MODEL_DICT`` 管理多算法引擎；TensorRT 模型默认 cuda:0，SAM3 固定 cuda:1（与引擎绑定策略一致）。
- 订阅 Redis 频道 ``task_id``（见 ``main``），接收 ``start_algorithm_list`` / ``stop_algorithm_list`` / LLM 配置，
  与编排服务下发的消息格式对齐。
- 检测结果经 ``fast_upload_to_redis`` 写入 Redis 并 publish ``alarm-images``（负载为 metadata key），供
  ``redis_result_processor`` 消费。

排查建议
--------
- 黑屏/无推理：确认 probe 是否挂上 ``inference_pad``、``MODEL_DICT`` 是否非空、Redis 是否已成功初始化引擎。
- 车速相关：算法 69 使用 ``TRACK_*_DS``；上报与 OSD 一致依赖 ``TRACK_RELATIVE_SPEEDS_DS``。
- LLM：依赖环境变量 ``DASHSCOPE_API_KEY`` 或 ``OPENAI_API_KEY``；``BoundedExecutor`` 防止异步请求堆积。
"""

import sys
import gi
import os
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds
import queue
import threading
import ctypes
import time
import cupy as cp
import torch
import json
import math
import traceback
import common.config
import numpy as np
import cv2
import base64
from importlib import reload
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from common.trt_inference_torch import TRTInferenceEngine
from common.load_SAM3 import Sam3VideoLoader, ReleaseSam3Model
from postprocess_utils.chinese_name import CLASS_NAME_TO_CHINESE
from common.llm_infer import realtime_llm_infer
from utils.redis_helper import RedisHelper
from postprocess_utils.car_speed_lines import update_speed_from_det_tensor
from postprocess_utils.vehicle_speed_tracker import process_vehicle_speeds_with_lanes

# --- GStreamer  muxer / tiler 分辨率（与 nvstreammux、tiler 插件属性一致） ---
MUXER_WIDTH = 1920
MUXER_HEIGHT = 1080
TILED_WIDTH = 1920
TILED_HEIGHT = 1080
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
from common.config_loader import get_profile, get_config_path
YAML_PATH = get_config_path()
print(f"[推理子进程] 当前环境: {get_profile()}, 配置文件: {YAML_PATH}")
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

# --- 管线重连（EOS/ERROR 退出 MainLoop 后指数退避重建） ---
STOP_REQUESTED = False
RECONNECT_BACKOFF_BASE = 2
RECONNECT_MAX_BACKOFF = 60
AREAJTR = None

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


# --- 推理临界区与帧计数（probe 内频繁读取 MODEL_DICT） ---
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
        - 线程池 ``INFERENCE_EXECUTOR`` 仅初始化一次；当前帧内推理仍以 GPU 分组串行为主（见 probe 注释）。

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
        """
        在 GPU 上的数据直接编码为 JPEG（避免 CPU 拷贝）

        Args:
            gpu_array: CuPy 数组或 PyTorch GPU tensor
            quality: JPEG 质量 (1-100)

        Returns:
            JPEG 字节数据
        """
        # 如果是 PyTorch tensor，转换为 numpy
        if isinstance(gpu_array, torch.Tensor):
            gpu_array = gpu_array.cpu().numpy()
        elif isinstance(gpu_array, cp.ndarray):
            # CuPy -> NumPy （这是必要的 CPU 拷贝，但只有一次）
            gpu_array = cp.asnumpy(gpu_array)

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
        """
        SAM3 模式：编码 SAM3 分割模型输出

        Args:
            gpu_tensor: GPU上的图像tensor
            sam3_output: ultralytics.engine.results.Results 对象，包含：
                - boxes: ultralytics.engine.results.Boxes 对象
                - masks: ultralytics.engine.results.Masks 对象（可选）
                - names: 类别名称列表
            quality: JPEG质量
            model_code: 当前算法的编号，用于前端分组

        Returns:
            (jpeg_bytes, detection_json_str)
        """
        # 转为 numpy（GPU->CPU 拷贝只发生一次）
        if isinstance(gpu_tensor, torch.Tensor):
            image_np = gpu_tensor.cpu().numpy()
        else:
            image_np = cp.asnumpy(gpu_tensor)

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
            image_np = cp.asnumpy(gpu_tensor)

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
            image_np = cp.asnumpy(gpu_tensor)

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
    """构造 OpenAI 兼容客户端（DashScope base_url）；缺 Key 返回 None，probe 内 LLM 分支自动跳过。"""
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
    """带并发上限的线程池：信号量满时 **非阻塞丢弃** 任务，防止推理帧堆积拖垮延迟。"""

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

def frame_inference_probe(pad, info, u_data):
    """RGBA NVMM 缓冲区的 Pad Probe：GPU 表面 → Torch → 多模型推理 → Redis/OSD。

    流程摘要:
        1. ``pyds.get_nvds_buf_surface_gpu`` + CuPy 拷贝出独立显存，再 DLPack 转 Torch（避免 UnownedMemory 悬垂）。
        2. ``ENGINE_LOCK`` 下拷贝 ``MODEL_DICT`` 快照；按 ``gpu_id`` 分两路由 ``run_single_inference`` **串行**执行
           （ByteTrack 等需帧序时不宜在同帧内乱序并行）。
        3. ``ALGORITHMNUM == "69"``：融合分割与检测 tensor，更新 ``TRACK_*_DS`` 并算相对车速。
        4. 有检测时 ``clone`` 帧 tensor，线程池调用 ``fast_upload_to_redis``；再写入 ``pyds`` 元数据供 nvdsosd。

    性能日志:
        周期打印 GPU 映射、推理、上传、写 meta 耗时；定位瓶颈时优先对照四项毫秒数。

    Args:
        pad, info: GStreamer probe 标准参数；``u_data`` 未使用。
    """
    global COUNT_FRAMES, TRACK_ABSOLUTE_SPEEDS_DS, TRACK_RELATIVE_SPEEDS_DS, LAST_UPLOAD_TIME, ALGORITHMNUM
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    l_frame = batch_meta.frame_meta_list

    while l_frame:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)

        # ========== 计时器：数据获取 ==========
        frame_start_time = time.time()

        # ========== 初始化所有可能的变量 ==========
        torch_tensor = None
        torch_tensor_on_gpu = None
        cp_arr = None
        unowned_mem = None
        mem_ptr = None
        dlpack = None
        all_detections = []

        try:
            data_type, shape, strides, dataptr, size = pyds.get_nvds_buf_surface_gpu(
                hash(buf), frame_meta.batch_id
            )

            ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
            ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
            c_data_ptr = ctypes.pythonapi.PyCapsule_GetPointer(dataptr, None)

            # ========== 1. GPU内存映射 ==========
            t1 = time.time()
            owner = None
            unowned_mem = cp.cuda.UnownedMemory(c_data_ptr, size, owner)
            mem_ptr = cp.cuda.MemoryPointer(unowned_mem, 0)
            cp_arr = cp.ndarray(
                shape=shape,
                dtype=data_type,
                memptr=mem_ptr,
                strides=strides,
                order='C'
            )

            # 关键：直接从 CuPy 数组克隆一份独立的 GPU 内存
            # 这样可以确保不依赖于 UnownedMemory 的生命周期
            cp_arr_owned = cp.array(cp_arr, copy=True)
            dlpack = cp.from_dlpack(cp_arr_owned)
            torch_tensor = torch.utils.dlpack.from_dlpack(dlpack)
            gpu_mapping_time = (time.time() - t1) * 1000

            # 立即释放 UnownedMemory 相关对象，因为我们已经有了独立的拷贝
            cp_arr = None
            dlpack = None
            unowned_mem = None
            mem_ptr = None

            COUNT_FRAMES += 1
            # ================= 注意大模型是脱离于正常检测模型的 ================
            # 当 LLM_PROMPT 不为空时，触发大模型分析
            try:
                if LLM_CLIENT and LLM_PROMPT:
                    current_time = time.time()
                    # 检查上传时间间隔
                    if COUNT_FRAMES % 60 == 0:
                        # 获取第一个 prompt 用于请求
                        img_np = torch_tensor.cpu().numpy()
                        prompt_key = list(LLM_PROMPT.keys()) if LLM_PROMPT else None
                        submitted = False  # 防止多 prompt key 导致重复投递
                        for pk in prompt_key:
                            if pk and not submitted:
                                llm_config = LLM_PROMPT[pk]
                                prompt_text = "请分析这张图片"
                                prompts_raw = llm_config.get("prompts")
                                if prompts_raw:
                                    prompts_list = json.loads(prompts_raw) if isinstance(prompts_raw, str) else prompts_raw
                                    if isinstance(prompts_list, list) and len(prompts_list) > 0:
                                        dede_bug = prompts_list[0].get("prompt")
                                        if prompts_list[0].get("prompt"):
                                            prompt_text = prompts_list[0].get("prompt", prompt_text)
                                            print(f"当前的提示词==============================={prompt_text}")
                                            img_np_for_llm = img_np

                                            # 提交到线程池异步执行（每帧只提交一次，避免重复事件）
                                            LLM_THREAD_POOL.submit(
                                                _request_llm_and_upload,
                                                TASK_ID,
                                                img_np_for_llm,
                                                prompt_text
                                            )
                                            submitted = True

            except Exception as e:
                print(f"[帧 #{COUNT_FRAMES}] LLM 请求失败：{e}")

            # ========== 2. 推理（包含预处理、推理、ultralytics后处理） ==========
            with ENGINE_LOCK:
                local_model_dict = dict(MODEL_DICT) if MODEL_DICT else {}
                local_conf = CONF_THRESHOLD

            if not local_model_dict:
                print(f"[帧 #{COUNT_FRAMES}] 推理引擎未初始化")
                l_frame = l_frame.next
                continue

            t2 = time.time()
            total_num_detections = 0

            # ========== 并行推理：按 GPU 分组 ==========
            # 将模型按 GPU 分组，同一 GPU 上的模型串行执行，不同 GPU 上的模型并行执行
            gpu0_models = []
            gpu1_models = []
            # print(MODEL_DICT)
            for model_code, model_info in local_model_dict.items():
                # 隔帧处理：检查是否需要跳过此帧
                model_info['frame_counter'] = model_info.get('frame_counter', 0) + 1
                frame_skip = model_info.get('frame_skip', 1)

                if model_info['frame_counter'] % frame_skip != 0:
                    print(f"[帧 #{COUNT_FRAMES}] 模型{model_code} 跳过 (隔{frame_skip}帧)")
                    continue

                # 根据 enable_tracking 标志和模型类型决定推理方式
                enable_tracking_flag = model_info.get('enable_tracking', ENABLE_TRACKING)
                model_type_val = model_info.get('model_type', 'detect')
                text_prompt = model_info.get("text_prompt","car")
                # 分配到对应 GPU 队列
                if model_info['gpu_id'] == 0:
                    gpu0_models.append((model_code, model_info, enable_tracking_flag))
                else:
                    gpu1_models.append((model_code, model_info, enable_tracking_flag))

            # ========== 定义单模型推理函数 ==========
            def run_single_inference(model_code, model_info, enable_tracking_flag):
                """单模型推理函数（线程安全）"""
                selected_engine = model_info['engine']
                actual_gpu_id = model_info['gpu_id']
                second_review_flag = ALGORITHM_SECOND_REVIEW.get(str(model_code), 0)

                try:
                    # 确保张量在正确的 GPU 上
                    if hasattr(torch_tensor, 'device'):
                        current_device = torch_tensor.device
                        target_device = torch.device(f"cuda:{actual_gpu_id}")
                        if current_device != target_device:
                            tensor_on_gpu = torch_tensor.to(target_device)
                        else:
                            tensor_on_gpu = torch_tensor
                    else:
                        tensor_on_gpu = torch_tensor

                    # 根据配置选择使用跟踪或普通检测
                    if enable_tracking_flag:
                        results,names = selected_engine.track(
                            tensor_on_gpu,
                            input_size=selected_engine.input_size,
                            conf_threshold=local_conf,
                            iou_threshold=IOU_THRESHOLD,
                            return_torch=True,
                            tracker=TRACKER_CONFIG,
                            persist=True
                        )
                    else:
                        if model_info.get('model_type') == "SAM3":
                            # SAM3 模型特殊处理
                            image = tensor_on_gpu.cpu().numpy()  # RGBA --> RGB HWC
                            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                            selected_engine.set_image(img_bgr)
                            results = selected_engine(text=[text_prompt])
                        else:
                            results,names = selected_engine(
                                tensor_on_gpu,
                                input_size=selected_engine.input_size,
                                conf_threshold=local_conf,
                                iou_threshold=IOU_THRESHOLD,
                                return_torch=True
                            )

                    # 获取检测数量
                    if isinstance(results[0], list):
                        num_detections = len(results[0])
                    elif model_info.get('model_type') == "SAM3":
                        num_detections = results[0].boxes.shape[0]
                    else:
                        num_detections = results[0].shape[0] if results[0].shape[0] > 0 else 0
                    # 保存结果
                    if model_info.get('model_type') == "SAM3":
                        return {
                            'model_code': model_code,
                            "second_review": second_review_flag,  # 二次复审标志
                            'results': results,
                            'task': "SAM3",
                            'num_detections': num_detections,
                            'success': True
                        }
                    else:
                        return {
                            'model_code': model_code,
                            "second_review": second_review_flag,  # 二次复审标志
                            'results': results,
                            "names":names,
                            'task': selected_engine.task,
                            'num_detections': num_detections,
                            'success': True
                        }

                except Exception as e:
                    print(f"[帧 #{COUNT_FRAMES}] 模型{model_code}推理失败：{e}")
                    traceback.print_exc()
                    return {
                        'model_code': model_code,
                        "second_review": second_review_flag,  # 二次复审标志,
                        'results': None,
                        "names":None,
                        'task': None,
                        'num_detections': 0,
                        'success': False
                    }

            # ========== 执行串行推理（当前帧） ==========
            # 注：由于跟踪器需要保持帧顺序，暂时使用串行执行
            # 后续可根据需要启用真正的并行（使用 ThreadPoolExecutor）
            all_detections = []

            # GPU0 上的模型（串行执行）
            gpu0_results = []
            for model_code, model_info, enable_tracking_flag in gpu0_models:
                result = run_single_inference(model_code, model_info, enable_tracking_flag)
                if result['success']:
                    gpu0_results.append(result)
                    total_num_detections += result['num_detections']


            # GPU1 上的模型（串行执行）
            gpu1_results = []
            for model_code, model_info, enable_tracking_flag in gpu1_models:
                result = run_single_inference(model_code, model_info, enable_tracking_flag)
                if result['success']:
                    gpu1_results.append(result)
                    total_num_detections += result['num_detections']

            # ========== 合并结果 ==========
            all_detections = gpu0_results + gpu1_results

            # 打印并行推理信息
            if gpu1_models and gpu0_models:
                print(f"[帧 #{COUNT_FRAMES}] 双卡调度：GPU0={len(gpu0_models)}个模型，GPU1={len(gpu1_models)}个模型")
            elif gpu0_models:
                print(f"[帧 #{COUNT_FRAMES}] 推理：GPU0={len(gpu0_models)}个模型")
            elif gpu1_models:
                print(f"[帧 #{COUNT_FRAMES}] 推理：GPU1={len(gpu1_models)}个模型")

            inference_total_time = (time.time() - t2) * 1000

            # ========== 2.5. 算法69：车道判断和相对速度调整 ==========
            if ALGORITHMNUM == "69" and total_num_detections > 0:
                try:
                    # 收集分割和检测结果
                    # segment_data = []  # 初始化segment_data
                    vehicle_data = []
                    segment_data = []  # 初始化segment_data

                    for det_info in all_detections:
                        model_code = det_info['model_code']
                        results = det_info['results']
                        task = det_info['task']

                        print(f"[DEBUG] 检测结果: model_code={model_code}, task={task}, num_detections={det_info['num_detections']}")

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
                    print(f"[DEBUG] segment_data长度: {len(segment_data)}, vehicle_data长度: {len(vehicle_data) if hasattr(vehicle_data, '__len__') else 'N/A'}")
                    if segment_data and vehicle_data is not None:
                        try:
                            # 1. 先计算绝对速度
                            task_track_state = TRACK_STATE_DS.get(TASK_ID, {})
                            task_absolute_speeds = TRACK_ABSOLUTE_SPEEDS_DS.get(TASK_ID, {})
                            update_speed_from_det_tensor(
                                vehicle_data,
                                task_track_state,
                                task_absolute_speeds
                            )
                            # 更新绝对速度状态
                            TRACK_STATE_DS[TASK_ID] = task_track_state
                            TRACK_ABSOLUTE_SPEEDS_DS[TASK_ID] = task_absolute_speeds
                            print(f"[DEBUG] TRACK_ABSOLUTE_SPEEDS_DS[{TASK_ID}] updated: {task_absolute_speeds}")

                            # 2. 使用绝对速度计算相对速度
                            relative_speeds = process_vehicle_speeds_with_lanes(
                                segment_data,
                                vehicle_data,
                                task_absolute_speeds,  # 传入绝对速度数据
                                TASK_ID,
                                REDIS_CLIENT
                            )

                            # 保存相对速度用于OSD显示
                            TRACK_RELATIVE_SPEEDS_DS[TASK_ID] = relative_speeds

                            print(f"[算法69] 计算了 {len(relative_speeds)} 辆车的相对速度")

                        except Exception as e:
                            print(f"[算法69] 相对速度计算失败: {e}")
                            TRACK_RELATIVE_SPEEDS_DS[TASK_ID] = {}
                except Exception as e:
                    print(f"[帧 #{COUNT_FRAMES}] [算法69调试] 速度调整失败: {e}")

            # ========== 3. 上传到Redis（如果启用） ==========
            upload_time = 0
            if total_num_detections > 0:
                try:
                    t3 = time.time()

                    # 确保数据独立，避免 nvdsosd 绘制红色框时修改共享内存
                    torch_tensor_clone = torch_tensor.clone()


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
                                torch_tensor_clone.detach(),
                                det_info['results'],
                                det_info['task'],
                                ENABLE_TRACKING,
                                det_info['model_code'],
                                det_info['names'],
                                second_review_flag  # 二次复审标志
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
                                torch_tensor_clone.detach(),
                                merged_results,
                                'merged',  # 标识为合并模式
                                ENABLE_TRACKING,
                                None,  # model_code 为 None 表示合并模式
                                None,  # names
                                any_second_review  # 二次复审标志
                            )
                    upload_time = (time.time() - t3) * 1000
                except Exception as e:
                    print(f"[帧 #{COUNT_FRAMES}] 上传失败: {e}")



            t4 = time.time()
            # ========== 4. 添加到 DeepStream 元数据 ==========
            if total_num_detections > 0:
                # 为每个模型的结果添加到 DeepStream 元数据
                for det_info in all_detections:
                    if det_info['num_detections'] > 0:
                        results = det_info['results']
                        task_mode = det_info['task']
                        names = det_info['names']
                        try:
                            if task_mode == 'obb':
                                add_detections_to_metadata_obb(batch_meta, frame_meta, names, results[0], ENABLE_TRACKING)
                            elif task_mode == 'SAM3':
                                add_sam3_detections_to_metadata(batch_meta, frame_meta, results, None)
                            else:
                                add_detections_to_metadata_det(batch_meta, frame_meta, names, results[0], ENABLE_TRACKING)
                        except Exception as e:
                            print(f"[帧 #{COUNT_FRAMES}] 模型 {det_info['model_code']} 绘制失败: {e}")

            add_metadata_time = (time.time() - t4) * 1000

            # ========== 5. 打印性能信息和检测数量 ==========
            total_time = (time.time() - frame_start_time) * 1000

            # 计算每模型平均耗时（用于评估并行效果）
            avg_time_per_model = inference_total_time / max(1, len(all_detections))

            print(f"[帧 #{COUNT_FRAMES}] 性能统计 | /"
                f"检测数：{total_num_detections} | "
                f"总耗时：{total_time:.2f}ms | "
                f"推理：{inference_total_time:.2f}ms | "
                f"GPU 映射：{gpu_mapping_time:.2f}ms | "
                f"上传：{upload_time:.2f}ms | "
                f"元数据：{add_metadata_time:.2f}ms | "
                f"模型数：{len(all_detections)}/{len(local_model_dict)} | "
                f"平均/模型：{avg_time_per_model:.2f}ms |"
                f"模型编号：{ALGORITHMNUM} |"
                f"提示词列表： {LLM_PROMPT} |"
                f"当前模型的二次复审状态：{ALGORITHM_SECOND_REVIEW}")
        except Exception as e:
            print(f"[帧 #{COUNT_FRAMES}] 帧处理异常: {e}")
            traceback.print_exc()

        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK


def fast_upload_to_redis(task_id: str, gpu_tensor: torch.Tensor, detections_np, task: str = 'detect', with_tracking: bool = False, model_code: str = None, names: dict = None, second_review_flag: int = 0):
    """将当前帧 JPEG 与检测列表写入 Redis，并向频道 ``alarm-images`` 发布 **metadata 的 key**。

    存储形态:
        - ``image:{task_id}:{ts}`` → JPEG bytes（TTL 600s）
        - ``metadata:{task_id}:{ts}`` → JSON（含 ``image_key``、``detections``、``TASK``、``second_review`` 等）
        - Pub/Sub payload = metadata 的 Redis key 字符串（消费者再 GET）

    节流:
        同一 ``task_id`` 受 ``UPLOAD_INTERVAL`` 限制；与 probe 异步线程池配合，勿在函数内阻塞过久。

    Args:
        task: ``detect`` / ``obb`` / ``segment`` / ``SAM3`` / ``merged``；合并模式走 ``encode_with_merged_detection_info``。
        second_review_flag: 写入 metadata，供下游 redis_result_processor 是否走 LLM 复审。
    """
    global LAST_UPLOAD_TIME
    try:
        current_time = time.time()

        if task_id in LAST_UPLOAD_TIME:
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
                "inference_mode": "video",
            }

            metadata_key = f"metadata:{task_id}:{timestamp}"
            REDIS_CLIENT.client.setex(
                metadata_key,
                600,
                json.dumps(metadata, ensure_ascii=False)
            )

            REDIS_CLIENT.publish(f"alarm-images", metadata_key)
            LAST_UPLOAD_TIME[task_id] = current_time


    except Exception:
        import traceback
        traceback.print_exc()


def llm_upload_to_redis(task_id: str, img_array: torch.Tensor, llm_result: str, task: str = 'llm', prompt: str = ""):
    """LLM 可视化链路专用：JPEG + metadata（含 ``llm_result``、``prompt``），发布键到 ``alarm-images``。

    Note:
        参数名为 ``img_array``，实现中按 numpy/OpenCV 使用；调用方 ``_request_llm_and_upload`` 已做 RGBA→BGR 与 uint8。
    """
    global LAST_LLM_UPLOAD_TIME, LLM_CLIENT
    try:
        current_time = time.time()

        if task_id in LAST_LLM_UPLOAD_TIME:
            if current_time - LAST_LLM_UPLOAD_TIME[task_id] < LLM_UPLOAD_INTERVAL:
                return

        llm_dets = parse_llm_objects_for_detect_frame(llm_result)
        if not llm_dets:
            print("[LLM] 没有解析到任何目标，跳过 Redis 上报")
            return

        ret, encoded = cv2.imencode('.jpg', img_array,
                                   [cv2.IMWRITE_JPEG_QUALITY, 95])
        height, width = img_array.shape[:2]

        # 前端二次编辑字段：包含类别表和可编辑框坐标。
        # redis_result_processor 会透传到 MQTT payload 的 data.detect_frame。
        detect_frame = build_llm_detect_frame_from_objects(
            llm_dets,
            image_width=width,
            image_height=height,
            emit_boxes=True,
            model_code="llm",
            coordinate_mode="normalized_1000",   # 如果你的 bbox_2d 实际是 0~1000，就改成 "normalized_1000"
        )

        if not detect_frame.get("boxes"):
            print("[LLM] detect_frame boxes is null, jump Redis publish")
            return

        if REDIS_CLIENT:
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
                "inference_mode": "video",
            }

            metadata_key = f"metadata:{task_id}:{timestamp}"
            REDIS_CLIENT.client.setex(
                metadata_key,
                600,
                json.dumps(metadata, ensure_ascii=False)
            )

            REDIS_CLIENT.publish(f"alarm-images", metadata_key)
            LAST_LLM_UPLOAD_TIME[task_id] = current_time


            print(f"[LLM] 已上传 LLM 结果到 Redis: {metadata_key}")

    except Exception:
        import traceback
        print(f"[LLM] 上传失败：{traceback.format_exc()}")

def _request_llm_and_upload(task_id: str, img_array: torch.Tensor, prompt: str):
    """线程池任务：``realtime_llm_infer`` → 成功则 ``llm_upload_to_redis``。

    兼容 numpy CPU 图像（由 probe 侧 ``torch_tensor.cpu().numpy()`` 传入）；内部完成 RGBA/BGR 与 JPEG。
    """
    global LLM_CLIENT, AVA_LLM_INFER_TIME, COUNT_FRAMES
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

        ret, img_bytes = cv2.imencode('.jpg', img_array)
        if not ret:
            print("[LLM] 图片编码失败")
            return
        img_b64 = base64.b64encode(img_bytes.tobytes()).decode("utf-8")
        img_data_url = f"data:image/jpeg;base64,{img_b64}"
        success, llm_result, spend_time = realtime_llm_infer(LLM_CLIENT, img_data_url, prompt)
        AVA_LLM_INFER_TIME += spend_time
        print(f"LLM_AVA_SPEND_TIME:{max(1, AVA_LLM_INFER_TIME / int(COUNT_FRAMES / 60))}")

        if success and llm_result:
            print(f"[LLM] 请求成功：{llm_result}")
            objects = parse_llm_objects_for_detect_frame(llm_result)  # 解析
            if not objects:
                print("[LLM] 没有解析到任何目标，跳过上报")
                return

            print(f"[LLM] 请求成功：{llm_result}...")
            llm_upload_to_redis(task_id, img_array, llm_result, prompt=prompt)
        else:
            print(f"[LLM] 请求失败或无结果")

    except Exception as e:
        print(f"[LLM] 请求异常：{e}")
        import traceback
        traceback.print_exc()


def add_detections_to_metadata_obb(batch_meta, frame_meta, names, detections: torch.Tensor, with_tracking: bool = False):
    """将 OBB tensor 转为 ``NvDsObjectMeta`` 供 nvdsosd 绘制。

    Tensor 形状:
        无跟踪 (N,7): cx,cy,w,h,angle,conf,cls；有跟踪 (N,8) 末尾为 track_id。
    """
    try:
        COLOR_MAP = {
            0: (1.0, 0.0, 0.0, 1.0),
            1: (0.0, 1.0, 0.0, 1.0),
            2: (1.0, 1.0, 0.0, 1.0),
        }
        DEFAULT_COLOR = (0.0, 0.0, 1.0, 1.0)
        BORDER_WIDTH = 3
        TEXT_Y_OFFSET = -25

        # 直接使用全局配置的帧尺寸
        frame_w = MUXER_WIDTH
        frame_h = MUXER_HEIGHT

        for idx, det in enumerate(detections):
            if with_tracking and det.shape[0] >= 8:
                # 有跟踪ID
                cx, cy, w, h, angle, conf, class_id, track_id = det.tolist()
                track_id = int(track_id) if track_id >= 0 else None
            else:
                # 无跟踪ID
                cx, cy, w, h, angle, conf, class_id = det.tolist()[:7]
                track_id = None

            # 将旋转框转换为水平矩形框
            poly = compute_obb_polygon(cx, cy, w, h, angle)
            min_x, min_y, max_x, max_y = polygon_to_aabb(poly)

            # 以最小坐标作为绘制的起点
            left = min_x
            top = min_y
            right = max_x
            bottom = max_y

            # 强制边界裁剪坐标（确保不为负数且不超出画面）
            left = max(0.0, min(left, frame_w - 1.0))
            right = max(left + 1.0, min(right, frame_w))
            top = max(0.0, min(top, frame_h - 1.0))
            bottom = max(top + 1.0, min(bottom, frame_h))

            width = max(1.0, right - left)
            height = max(1.0, bottom - top)

            # DeepStream 需要整数像素 (使用四舍五入避免系统性偏移)
            left_i, top_i = round(left), round(top)
            width_i, height_i = round(width), round(height)

            obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)

            obj_meta.class_id = int(class_id)
            obj_meta.confidence = float(conf)
            obj_meta.obj_label = names[int(class_id)]

            # 设置跟踪ID（如果存在）
            if track_id is not None and track_id >= 0:
                obj_meta.object_id = track_id

            rect = obj_meta.rect_params
            rect.left = left_i
            rect.top = top_i
            rect.width = width_i
            rect.height = height_i
            rect.border_width = BORDER_WIDTH
            rect.has_bg_color = 0
            rect.border_color.set(*COLOR_MAP.get(int(class_id), DEFAULT_COLOR))

            txt = obj_meta.text_params
            # 显���文本：算法69时改为 **km/h，其它算法保持原样
            if ALGORITHMNUM == "69":
                if track_id is not None and track_id >= 0:
                    txt.display_text = f"{obj_meta.obj_label} **km/h ID:{track_id}"
                else:
                    txt.display_text = f"{obj_meta.obj_label} **km/h"
            else:
                # 显示文本包含跟踪ID（如果存在）
                if track_id is not None and track_id >= 0:
                    txt.display_text = f"{obj_meta.obj_label} {conf:.2f} ID:{track_id}"
                else:
                    txt.display_text = f"{obj_meta.obj_label} {conf:.2f}"
            txt.x_offset = int(left)
            txt.y_offset = int(max(0, top + TEXT_Y_OFFSET))

            txt.font_params.font_name = "Serif"
            txt.font_params.font_size = 10
            txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            txt.set_bg_clr = 1  # 启用背景色（属性赋值）
            txt.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)  # 设置背景颜色（方法调用）

            pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)

    except Exception:
        pass

def add_detections_to_metadata_segment(batch_meta, frame_meta, results, with_tracking: bool = False,task:str='segment'):
    """分割分支 OSD：由 ``results[0]`` 的打包结构（xy/xyxy/conf/cls/track）生成 ``NvDsObjectMeta``。

    与 ``encode_with_detection_info(..., task='segment')`` 使用同一套 tensor 打包约定；掩膜若存在，
    当前实现仍以框/meta 为主（DeepStream 文本框绘制），细粒度轮廓见注释代码路径。
    """
    try:
        COLOR_MAP = {
            0: (1.0, 0.0, 0.0, 1.0),
            1: (0.0, 1.0, 0.0, 1.0),
            2: (1.0, 1.0, 0.0, 1.0),
        }
        DEFAULT_COLOR = (0.0, 0.0, 1.0, 1.0)
        BORDER_WIDTH = 3
        TEXT_Y_OFFSET = -25

        frame_w = MUXER_WIDTH
        frame_h = MUXER_HEIGHT

        # 解析结果
        detections = results[0]  # 检测结果 (N, 6/7)

        # 解析检测结果
        if with_tracking and len(detections)>=5:
            _, xyxy, conf, cls, track_id = detections
            track_id = int(track_id) if track_id >= 0 else None
            for xyxy_, conf_, cls_, track_id_ in zip(xyxy, conf, cls, track_id):
                x1,y1,x2,y2 = xyxy_
                conf = conf_
                cls = cls
                left = x1
                top = y1
                right = x2
                bottom = y2
                class_id = cls_
                track_id = track_id_
                # 边界约束
                left = max(0.0, min(x1, frame_w - 1.0))
                right = max(left + 1.0, min(x2, frame_w))
                top = max(0.0, min(y1, frame_h - 1.0))
                bottom = max(top + 1.0, min(y2, frame_h))

                width = max(1.0, right - left)
                height = max(1.0, bottom - top)

                left_i, top_i = round(left), round(top)
                width_i, height_i = round(width), round(height)

                obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)

                obj_meta.class_id = int(class_id)
                obj_meta.confidence = float(conf)
                obj_meta.obj_label = CLASS_NAMES.get(int(class_id), f"cls_{class_id}")

                if track_id is not None and track_id >= 0:
                    obj_meta.object_id = track_id

                rect = obj_meta.rect_params
                rect.left = left_i
                rect.top = top_i
                rect.width = width_i
                rect.height = height_i
                rect.border_width = BORDER_WIDTH
                rect.has_bg_color = 0
                rect.border_color.set(*COLOR_MAP.get(int(class_id), DEFAULT_COLOR))

                txt = obj_meta.text_params

                if track_id is not None and track_id >= 0:
                    txt.display_text = f"{obj_meta.obj_label} {conf:.2f} ID:{track_id}"
                else:
                    txt.display_text = f"{obj_meta.obj_label} {conf:.2f}"

                txt.x_offset = int(left)
                txt.y_offset = int(max(0, top + TEXT_Y_OFFSET))

                txt.font_params.font_name = "Serif"
                txt.font_params.font_size = 10
                txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                txt.set_bg_clr = 1
                txt.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

                pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)

        else:
            _, xyxy, conf, cls = detections
            track_id = None

            for xyxy_, conf_, cls_ in zip(xyxy, conf, cls):
                x1,y1,x2,y2 = xyxy_
                conf = conf_
                cls = cls_
                left = x1
                top = y1
                right = x2
                bottom = y2

                # 边界约束
                left = max(0.0, min(x1, frame_w - 1.0))
                right = max(left + 1.0, min(x2, frame_w))
                top = max(0.0, min(y1, frame_h - 1.0))
                bottom = max(top + 1.0, min(y2, frame_h))

                width = max(1.0, right - left)
                height = max(1.0, bottom - top)

                left_i, top_i = round(left), round(top)
                width_i, height_i = round(width), round(height)

                obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)

                obj_meta.class_id = int(class_id)
                obj_meta.confidence = float(conf)
                obj_meta.obj_label = CLASS_NAMES.get(int(class_id), f"cls_{class_id}")
                # 算法69：类别显示为"car"
                # if ALGORITHMNUM == "69":
                    # obj_meta.obj_label = "car"

                if track_id is not None and track_id >= 0:
                    obj_meta.object_id = track_id

                rect = obj_meta.rect_params
                rect.left = left_i
                rect.top = top_i
                rect.width = width_i
                rect.height = height_i
                rect.border_width = BORDER_WIDTH
                rect.has_bg_color = 0
                rect.border_color.set(*COLOR_MAP.get(int(class_id), DEFAULT_COLOR))

                txt = obj_meta.text_params

                if track_id is not None and track_id >= 0:
                    txt.display_text = f"{obj_meta.obj_label} {conf:.2f} ID:{track_id}"
                else:
                    txt.display_text = f"{obj_meta.obj_label} {conf:.2f}"

                txt.x_offset = int(left)
                txt.y_offset = int(max(0, top + TEXT_Y_OFFSET))

                txt.font_params.font_name = "Serif"
                txt.font_params.font_size = 10
                txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                txt.set_bg_clr = 1
                txt.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

                pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)

    except Exception:
        pass

def add_detections_to_metadata_det(batch_meta, frame_meta, names, detections: torch.Tensor, with_tracking: bool = False):
    """水平框 DET：tensor 行格式含扩展字段（cxn/cyn/w/h 等），与 TRT 引擎 ``return_torch`` 输出对齐。

    ``ALGORITHMNUM == "69"``：从 ``TRACK_RELATIVE_SPEEDS_DS[TASK_ID]`` 取 track 级速度附加到 ``display_text``。
    """
    try:
        COLOR_MAP = {
            0: (1.0, 0.0, 0.0, 1.0),
            1: (0.0, 1.0, 0.0, 1.0),
            2: (1.0, 1.0, 0.0, 1.0),
        }
        DEFAULT_COLOR = (0.0, 0.0, 1.0, 1.0)
        BORDER_WIDTH = 3
        TEXT_Y_OFFSET = -25

        # 算法69：使用相对速度进行显示
        if ALGORITHMNUM == "69":
            # 使用相对速度进行OSD显示
            task_relative_speeds = TRACK_RELATIVE_SPEEDS_DS.get(TASK_ID, {})
            print(f"[DEBUG] OSD开始: TRACK_RELATIVE_SPEEDS_DS={TRACK_RELATIVE_SPEEDS_DS}, TASK_ID={TASK_ID}, task_relative_speeds={task_relative_speeds}")

        # 直接使用全局配置的帧尺寸
        frame_w = MUXER_WIDTH
        frame_h = MUXER_HEIGHT

        for idx, det in enumerate(detections):
            if with_tracking and det.shape[0] >= 11:
                # 有跟踪ID
                x1, y1, x2, y2, conf, class_id, _, _, _, _, track_id = det.tolist()
                track_id = int(track_id) if track_id >= 0 else None
            else:
                # 无跟踪ID
                x1, y1, x2, y2, conf, class_id, _, _, _, _,= det.tolist()[:10]
                track_id = None

            # 算法69：只绘制类别0（car）的目标，其他类别过滤掉
            if ALGORITHMNUM == "69" and int(class_id) != 0:
                continue

            # DET模式：直接使用水平框坐标
            left = x1
            top = y1
            right = x2
            bottom = y2

            # 强制边界裁剪坐标（确保不为负数且不超出画面）
            left = max(0.0, min(left, frame_w - 1.0))
            right = max(left + 1.0, min(right, frame_w))
            top = max(0.0, min(top, frame_h - 1.0))
            bottom = max(top + 1.0, min(bottom, frame_h))

            width = max(1.0, right - left)
            height = max(1.0, bottom - top)

            # DeepStream 需要整数像素 (使用四舍五入避免系统性偏移)
            left_i, top_i = round(left), round(top)
            width_i, height_i = round(width), round(height)

            obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)

            obj_meta.class_id = int(class_id)
            obj_meta.confidence = float(conf)
            # 算法69：类别显示为"car"
            if ALGORITHMNUM == "69":
                obj_meta.obj_label = "car"
            else:
                obj_meta.obj_label = names[int(class_id)]

            # 设置跟踪ID（如果存在）
            if track_id is not None and track_id >= 0:
                obj_meta.object_id = track_id

            rect = obj_meta.rect_params
            rect.left = left_i
            rect.top = top_i
            rect.width = width_i
            rect.height = height_i
            rect.border_width = BORDER_WIDTH
            rect.has_bg_color = 0
            rect.border_color.set(*COLOR_MAP.get(int(class_id), DEFAULT_COLOR))

            txt = obj_meta.text_params
            # 算法69：在 OSD 中只显示相对速度 km/h，相对速度计算失败则不显示速度
            if ALGORITHMNUM == "69":
                if track_id is not None and track_id >= 0:
                    # 只显示相对速度，如果没有相对速度则不显示速度
                    relative_speed = task_relative_speeds.get(track_id, None)
                    print(f"[DEBUG] OSD速度检查: track_id={track_id}, task_relative_speeds={task_relative_speeds}, relative_speed={relative_speed}")
                    if relative_speed is not None:
                        speed_kmh_int = int(relative_speed)  # 车速向下取整为整数
                        txt.display_text = f"{obj_meta.obj_label} {speed_kmh_int}km/h ID:{track_id}"
                        print(f"[相对速度] track_id={track_id}: {relative_speed:.1f}km/h")
                    else:
                        # 相对速度未计算，不显示速度
                        txt.display_text = f"{obj_meta.obj_label} ID:{track_id}"
                else:
                    txt.display_text = f"{obj_meta.obj_label}"
            else:
                # 显示文本包含跟踪ID（如果存在）
                if track_id is not None and track_id >= 0:
                    txt.display_text = f"{obj_meta.obj_label} {conf:.2f} ID:{track_id}"
                else:
                    txt.display_text = f"{obj_meta.obj_label} {conf:.2f}"
            txt.x_offset = int(left)
            txt.y_offset = int(max(0, top + TEXT_Y_OFFSET))

            txt.font_params.font_name = "Serif"
            txt.font_params.font_size = 10
            txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            txt.set_bg_clr = 1  # 启用背景色（属性赋值）
            txt.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)  # 设置背景颜色（方法调用）

            pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)

    except Exception:
        pass

def add_sam3_detections_to_metadata(batch_meta, frame_meta, sam3_output, class_names: dict = None):
    """SAM3：读取 ultralytics ``Results`` 的 ``boxes`` 写 OSD；掩膜字段存在亦主要走框/meta（与编码侧一致）。

    ``class_names`` 可覆盖；默认取 ``results.names``。
    """
    try:
        COLOR_MAP = {
            0: (1.0, 0.0, 0.0, 1.0),
            1: (0.0, 1.0, 0.0, 1.0),
            2: (1.0, 1.0, 0.0, 1.0),
        }
        DEFAULT_COLOR = (0.0, 0.0, 1.0, 1.0)
        BORDER_WIDTH = 3
        TEXT_Y_OFFSET = -25

        # 直接使用全局配置的帧尺寸
        frame_w = MUXER_WIDTH
        frame_h = MUXER_HEIGHT

        # 提取 SAM3 输出中的关键信息
        # sam3_output 是 Results 对象，通过 results[0] 访问
        results_obj = sam3_output[0] if isinstance(sam3_output, list) else sam3_output

        # 获取边界框（XYXY格式）
        if not hasattr(results_obj, 'boxes') or results_obj.boxes is None:
            print("[SAM3] 没有检测到边界框")
            return

        boxes_xyxy = results_obj.boxes.xyxy  # (N, 4) - [x1, y1, x2, y2]

        # 获取类别名称映射
        sam3_names = results_obj.names if hasattr(results_obj, 'names') else {}
        if isinstance(sam3_names, dict):
            class_id_to_name = sam3_names
        else:
            # names 可能是列表，转换为字典
            class_id_to_name = {i: name for i, name in enumerate(sam3_names)}

        # 遍历每个检测框
        for idx, box_coords in enumerate(boxes_xyxy):
            try:
                x1, y1, x2, y2 = [float(v) for v in box_coords]

                # 边界约束
                left = max(0.0, min(x1, frame_w - 1.0))
                right = max(left + 1.0, min(x2, frame_w))
                top = max(0.0, min(y1, frame_h - 1.0))
                bottom = max(top + 1.0, min(y2, frame_h))

                width = max(1.0, right - left)
                height = max(1.0, bottom - top)

                # DeepStream 需要整数像素
                left_i, top_i = round(left), round(top)
                width_i, height_i = round(width), round(height)

                # 获取置信度（如果存在）
                confidence = 1.0  # SAM3 默认置信度为 1.0
                if hasattr(results_obj.boxes, 'conf') and results_obj.boxes.conf is not None:
                    confidence = float(results_obj.boxes.conf[idx])

                # 获取类别ID
                class_id = 0  # 默认类别ID
                if hasattr(results_obj.boxes, 'cls') and results_obj.boxes.cls is not None:
                    class_id = int(results_obj.boxes.cls[idx])

                # 创建对象元数据
                obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
                obj_meta.class_id = class_id
                obj_meta.confidence = confidence

                # 设置对象标签（优先使用配置中的 text_prompt）
                try:
                    obj_meta.obj_label = MODEL_CONIFG[ALGORITHMNUM].get("text_prompt", class_id_to_name.get(class_id, f"class_{class_id}"))
                except Exception:
                    obj_meta.obj_label = class_id_to_name.get(class_id, f"class_{class_id}")

                # 设置矩形边框
                rect = obj_meta.rect_params
                rect.left = left_i
                rect.top = top_i
                rect.width = width_i
                rect.height = height_i
                rect.border_width = BORDER_WIDTH
                rect.has_bg_color = 0
                rect.border_color.set(*COLOR_MAP.get(class_id, DEFAULT_COLOR))

                # 设置文本
                txt = obj_meta.text_params
                txt.display_text = f"{obj_meta.obj_label} {confidence:.2f}"
                txt.x_offset = int(left)
                txt.y_offset = int(max(0, top + TEXT_Y_OFFSET))

                txt.font_params.font_name = "Serif"
                txt.font_params.font_size = 10
                txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                txt.set_bg_clr = 1
                txt.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

                # 添加到帧元数据
                pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)

            except Exception as e:
                print(f"[SAM3] 处理检测框 {idx} 失败: {e}")
                continue

    except Exception as e:
        print(f"[SAM3] 检测处理错误: {e}")
        traceback.print_exc()

def bus_call(bus, message, loop):
    """兼容保留：EOS/ERROR 时退出 ``GLib.MainLoop``（当前主路径更多使用 ``main`` 内 ``local_bus_call``）。"""
    t = message.type
    if t == Gst.MessageType.EOS:
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, dbg = message.parse_error()
        loop.quit()
    return True


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


def main(input_rtsp, output_rtmp, task_id, algorithm_list, filePath, assigned_gpu_id=0, llm_list=None):
    """子进程入口：reload 配置 → 初始化模型与 Redis → 订阅 ``task_id`` → GStreamer 主循环（可重连）。

    Args:
        input_rtsp / output_rtmp: 支持 rtmp://、rtsp://、默认 RTSP 输入；输出支持 rtmp、rtspclientsink、gb28181:// UDP。
        task_id: Redis 订阅频道名（与设备 SN 一致）；亦为全局 ``TASK_ID``。
        algorithm_list: 启动时的算法列表；空列表则不加载引擎直至 Redis 下发 start。
        filePath / assigned_gpu_id: 传入推理逻辑（路径上报等）；后者当前更多保留给编排侧语义。
        llm_list: 初始 LLM 配置列表，写入 ``LLM_PROMPT``。

    线程:
        ``redis_subscribe_worker`` 守护线程阻塞于 ``RedisHelper.subscribe``，断线返回后由封装侧重连（见 RedisHelper）。
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
        print(f"初始无算法运行")
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


        LLM_CLIENT = init_llm_api()

        def redis_subscribe_worker():
            """阻塞订阅 ``task_id``；连接断开导致 ``subscribe`` 返回时依赖 RedisHelper 外层重连。"""
            global REDIS_CLIENT, STOP_REQUESTED

            print(f"[Redis] 订阅线程启动，频道: {task_id}")

            REDIS_CLIENT.subscribe(str(task_id), on_message)
            print("[Redis] subscribe returned, reconnecting...")

        sub_thread = threading.Thread(target=redis_subscribe_worker, daemon=True, name=f"redis-sub-{task_id}")
        sub_thread.start()
        print(f"[初始化] Redis 订阅线程已启动")

    except Exception as e:
        print(f"[初始化] 严重错误: Redis 初始化失败: {e}")
        import traceback
        traceback.print_exc()

    Gst.init(None)

    def build_pipeline():
        """根据输入/输出 URL 拼装 ``Gst.parse_launch`` 字符串；失败返回 None。

        Probe 挂载点:
            元素名 ``inference_pad``（sink pad）附加 ``frame_inference_probe``。
        """
        is_rtmp_input = input_rtsp.startswith('rtmp://')
        is_rtsp_input = input_rtsp.startswith('rtsp://')

        is_rtsp_output = output_rtmp.startswith('rtsp://')
        is_gb28181_output = output_rtmp.startswith('gb28181://') or 'gb28181' in output_rtmp.lower()

        if is_rtmp_input:
            input_src = (
                f'rtmpsrc location="{input_rtsp}" timeout=5 '
                f'! flvdemux ! queue max-size-buffers=10 leaky=downstream '
                f'! h264parse '
            )
            print("[Gst] 使用 RTMP 输入流（优化版）")
        elif is_rtsp_input:
            input_src = (
                f'rtspsrc location="{input_rtsp}" protocols=tcp is-live=true latency=300 '
                f'! rtph264depay '
                f'! h264parse '
            )
            print("[Gst] 使用 RTSP 输入流")
        else:
            input_src = (
                f'rtspsrc location="{input_rtsp}" protocols=tcp is-live=true latency=300 '
                f'! rtph264depay '
                f'! h264parse '
            )
            print("[Gst] 使用 RTSP 输入流 (默认)")

        if is_gb28181_output:
            print("[Gst] 使用 GB28181 协议推流 (UDP 方式)")
            try:
                addr = output_rtmp.replace('gb28181://', '')
                if '/' in addr:
                    host_port = addr.split('/')[0]
                else:
                    host_port = addr

                if ':' in host_port:
                    gb_host, gb_port = host_port.split(':')
                    gb_port = int(gb_port)
                else:
                    gb_host = host_port
                    gb_port = 5060

                print(f"[Gst] GB28181 目标: {gb_host}:{gb_port}")
            except Exception as e:
                print(f"[Gst] 解析 GB28181 地址失败: {e}, 使用默认值")
                gb_host = "127.0.0.1"
                gb_port = 5060

            pipeline_desc = (
                input_src +
                f'! nvv4l2decoder discard-corrupted-frames=true skip-frames=decode_non_ref num-extra-surfaces=2 '
                f'! m.sink_0 nvstreammux name=m batch-size=1 width={MUXER_WIDTH} height={MUXER_HEIGHT} '
                f'sync-inputs=false align-inputs=false attach-sys-ts=true live-source=true '
                f'! nvvideoconvert '
                f'! nvmultistreamtiler width={TILED_WIDTH} height={TILED_HEIGHT} '
                f'! nvvideoconvert '
                f'! video/x-raw(memory:NVMM), format=RGBA '
                f'! nvvideoconvert name=inference_pad '
                f'! nvdsosd '
                f'! nvvideoconvert '
                f'! nvv4l2h264enc bitrate=6000000 '
                f'! h264parse config-interval=-1 '
                f'! rtph264pay pt=96 '
                f'! udpsink host={gb_host} port={gb_port} async=false sync=false'
            )
        elif is_rtsp_output:
            print("[Gst] 使用 RTSP 协议推流 (UDP 方式)")
            try:
                addr = output_rtmp.replace('rtsp://', '')
                if '/' in addr:
                    host_port = addr.split('/')[0]
                else:
                    host_port = addr

                if ':' in host_port:
                    rtsp_host, rtsp_port = host_port.rsplit(':', 1)
                    rtsp_port = int(rtsp_port)
                else:
                    rtsp_host = host_port
                    rtsp_port = 5540

                print(f"[Gst] RTSP 目标: {rtsp_host}:{rtsp_port}")
            except Exception as e:
                print(f"[Gst] 解析 RTSP 地址失败: {e}, 使用默认值")
                rtsp_host = "127.0.0.1"
                rtsp_port = 5540

            pipeline_desc = (
                input_src +
                f'! nvv4l2decoder discard-corrupted-frames=true skip-frames=decode_non_ref num-extra-surfaces=2 '
                f'! m.sink_0 nvstreammux name=m batch-size=1 width={MUXER_WIDTH} height={MUXER_HEIGHT} '
                f'sync-inputs=false align-inputs=false attach-sys-ts=true live-source=true '
                f'! nvvideoconvert '
                f'! nvmultistreamtiler width={TILED_WIDTH} height={TILED_HEIGHT} '
                f'! nvvideoconvert '
                f'! video/x-raw(memory:NVMM), format=RGBA '
                f'! nvvideoconvert name=inference_pad '
                f'! nvdsosd '
                f'! nvvideoconvert '
                f'! nvv4l2h264enc bitrate=6000000 '
                f'! h264parse config-interval=-1 '
                f'! rtspclientsink location="{output_rtmp}" protocols=tcp '
            )
        else:
            print("[Gst] 使用 RTMP 协议推流")
            pipeline_desc = (
                input_src +
                f'! nvv4l2decoder '
                f'! m.sink_0 nvstreammux name=m batch-size=1 width={MUXER_WIDTH} height={MUXER_HEIGHT} '
                f'sync-inputs=false align-inputs=false live-source=true '
                f'! nvvideoconvert '
                f'! nvmultistreamtiler width={TILED_WIDTH} height={TILED_HEIGHT} '
                f'! nvvideoconvert '
                f'! video/x-raw(memory:NVMM), format=RGBA '
                f'! nvvideoconvert name=inference_pad '
                f'! nvdsosd '
                f'! nvvideoconvert '
                f'! nvv4l2h264enc bitrate=6000000 iframeinterval=30 '
                f'! h264parse config-interval=-1 '
                f'! flvmux ! rtmpsink location="{output_rtmp}" sync=false async=false'
            )
        try:
            # inference-only: cut the push tail, end with fakesink
            if '! nvdsosd' in pipeline_desc:
                pipeline_desc = pipeline_desc.split('! nvdsosd')[0] + '! fakesink sync=false async=false'
                print("[Gst] inference-only (fakesink)")
            pipeline = Gst.parse_launch(pipeline_desc)
            return pipeline
        except Exception as e:
            print(f"[Gst] 构建 pipeline 失败: {e}")
            return None

    backoff = 1
    try:
        while not STOP_REQUESTED:
            pipeline = build_pipeline()
            if not pipeline:
                print(f"[Gst] pipeline 构建失败，{backoff}s 后重试")
                time.sleep(backoff)
                backoff = min(backoff * RECONNECT_BACKOFF_BASE, RECONNECT_MAX_BACKOFF)
                continue
            backoff = 1

            inference_pad = pipeline.get_by_name("inference_pad")
            if inference_pad:
                pad_sinkpad = inference_pad.get_static_pad("sink")
                pad_sinkpad.add_probe(Gst.PadProbeType.BUFFER, frame_inference_probe, None)

            loop = GLib.MainLoop()
            bus = pipeline.get_bus()
            bus.add_signal_watch()

            def local_bus_call(bus_obj, message, loop_obj):
                """EOS/ERROR 时 quit MainLoop，外层循环重建 pipeline。"""
                t = message.type
                if t == Gst.MessageType.EOS:
                    print("[Gst] 收到 EOS，准备重连")
                    loop_obj.quit()
                elif t == Gst.MessageType.ERROR:
                    err, dbg = message.parse_error()
                    print(f"[Gst] ERROR: {err}, {dbg}")
                    loop_obj.quit()
                return True

            bus.connect("message", local_bus_call, loop)

            pipeline.set_state(Gst.State.PLAYING)
            print("[Gst] pipeline 运行中")
            try:
                loop.run()
            except KeyboardInterrupt:
                print("[主] KeyboardInterrupt，停止服务并退出")
                STOP_REQUESTED = True
                try:
                    loop.quit()
                except Exception:
                    pass
                break
            finally:
                try:
                    pipeline.set_state(Gst.State.NULL)
                except Exception:
                    pass

            if STOP_REQUESTED:
                break

            print("[Gst] pipeline 停止或异常结束，准备重连")
            time.sleep(backoff)
            backoff = min(backoff * RECONNECT_BACKOFF_BASE, RECONNECT_MAX_BACKOFF)

    except Exception as e:
        print(f"[主] 主循环发生异常: {e}")
    finally:
        STOP_REQUESTED = True
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


