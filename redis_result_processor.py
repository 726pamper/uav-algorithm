"""
Redis 告警消费与可视化上报（与 DeepStream 发布端协议耦合）。

数据通路:
    1. Pub/Sub: 订阅 ``channel``（默认与 bucket 同名），消息体为 **字符串**；该字符串视为 Redis **键**。
    2. 使用同一 Redis 连接 ``GET(该键)`` 得到完整 JSON，交给 ``RedisResultProcessor._process_message``。
    3. JSON 内需包含 ``image_key``（JPEG 二进制存储键）、``detections``、``algorithmnum``、``TASK`` 等；
       图像经 ``_decode_frame`` 解码，经 ``ALGO_REGISTRY`` 中对应算法的 detect/draw 处理。
    4. 结果上传 MinIO，并通过 MQTT 发布 ``definition: ai-event``。

算法扩展:
    在 ``postprocess_utils`` 下新增模块，导出 ``ALGO_ID`` 与 ``EXPORT``（通常含 ``detect`` / ``draw``），
    由 ``RedisResultProcessor.auto_register_algos`` 扫描注册；键为算法编号字符串。

注意:
    - 二次复审依赖环境变量 ``DASHSCOPE_API_KEY`` 或 ``OPENAI_API_KEY``；未配置时 ``llm_client`` 为 None，
      ``_perform_llm_review`` 会跳过并清理临时文件。
    - 多算法合并上报时检测项应带 ``model_code``；算法 75、69、79 等分支见 ``_process_multi_algorithm``。
"""

import os
import importlib
import yaml
import sys
import uuid
import json
import time
import threading
import tempfile
from typing import Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import paho.mqtt.client as mqtt
import redis
import numpy as np
import copy
import cv2
from openai import OpenAI
import logging
from common.llm_infer import Second_review
from common.config import model_prompt
from postprocess_utils.onnxocr.onnx_paddleocr import ONNXPaddleOcr

from utils.minio_util import MinioHelper, init_minio_client

logger = logging.getLogger(__name__)


class _ThreadIdFilter(logging.Filter):
    """自动为每条日志注入当前线程 ID"""

    def filter(self, record):
        record.thread_id = threading.get_ident()
        return True


logger.addFilter(_ThreadIdFilter())


class RedisResultProcessor:
    """见模块文档字符串；本类负责订阅、解析、算法路由、MinIO 与 MQTT 侧上报。"""

    DEFAULT_COLORS = {
        0: (0, 0, 255),
        1: (0, 255, 0),
        2: (0, 255, 255),
    }

    def __init__(
        self,
        redis_client,
        minio_helper: MinioHelper,
        bucket_name: str = "alarm-images",
        channel: str = "alarm-images",
        mqtt_client = None,
        class_names: Optional[Dict[int, str]] = None,
    ):
        """
        Args:
            redis_client: 上游传入的 Redis 客户端；内部会尽量拷贝连接参数另建客户端供 Pub/Sub 与 GET 使用。
            minio_helper: 必填；用于上传绘制结果与原图。
            bucket_name: 默认逻辑桶名（与业务约定一致即可）。
            channel: Pub/Sub 频道名；消息 payload = 存储完整元数据的 Redis key。
            mqtt_client: 用于发布 ai-event；LLM 仅结果链路亦依赖。
            class_names: 可选类别 id -> 名称映射，供绘制层使用。
        """
        if not minio_helper:
            raise ValueError("minio_helper is required")
        self._user_redis = redis_client
        self.mqtt_client = mqtt_client
        self._redis: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self.minio_helper = minio_helper
        self.bucket_name = bucket_name
        self._stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.channel = channel
        # 线程池用于处理消息
        self._executor: Optional[ThreadPoolExecutor] = None
        self.class_names = class_names or {}
        self.do_task_id = {}
        self._violation_tracking = {}
        self._fire_exit_tracking = {}
        self._normal_tracking = {}
        self._violation_threshold = 0.0
        self._gs_tracking = {}
        self.ALGO_REGISTRY = {}
        self.auto_register_algos()
        logger.info(f"已注册算法: {list(self.ALGO_REGISTRY.keys())}")
        self.ocr_model = ONNXPaddleOcr(use_angle_cls=False, use_gpu=False)
        self._universal_tracking = {}
        self.llm_client = self._init_llm_api()

    def start(self):
        """启动后台线程与线程池；重复调用且线程仍存活时直接返回。"""
        if self._thread and self._thread.is_alive():
            return
        self._stopped.clear()
        self._executor = ThreadPoolExecutor(max_workers=5)
        self._thread = threading.Thread(target=self._run, name=f"redis-processor")
        self._thread.start()
        logger.info("RedisResultProcessor started")

    def stop(self, timeout: float = 5.0):
        """设置停止标志，解除 Pub/Sub 阻塞，关闭连接并回收线程池。

        Args:
            timeout: 等待消费线程退出的最大秒数。
        """
        self._stopped.set()
        try:
            if self._pubsub:
                try:
                    self._pubsub.unsubscribe(self.channel)
                except Exception:
                    pass
                try:
                    self._pubsub.close()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if self._redis and hasattr(self._redis, 'close'):
                try:
                    self._redis.close()
                except Exception:
                    pass
        except Exception:
            pass

        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        logger.info("RedisResultProcessor stopped")

    def auto_register_algos(self, root="postprocess_utils"):
        """扫描 ``root`` 目录下 ``*.py``，将含 ``ALGO_ID`` + ``EXPORT`` 的模块注册到 ``ALGO_REGISTRY``。

        每个模块单独 try/except：单个模块 import 失败只跳过它自己，不再中断后续模块。
        旧实现把 try 包在整个 for 循环外，任一模块导入失败（如缺少 Pillow / shapely）
        会导致其后所有算法都注册不上，且 ``os.listdir`` 顺序不确定 → 表现为"时好时坏"。
        """
        path = os.path.join(os.path.dirname(__file__), root)
        try:
            files = sorted(os.listdir(path))
        except Exception:
            logger.exception(f"[auto_register_algos] 无法列出目录: {path}")
            return

        failed = []
        for file in files:
            if not file.endswith(".py") or file.startswith("__"):
                continue
            module_name = f"{root}.{file[:-3]}"
            try:
                module = importlib.import_module(module_name)
            except Exception as e:
                failed.append(module_name)
                logger.exception(f"[auto_register_algos] 模块 {module_name} 导入失败，已跳过: {e}")
                continue

            if hasattr(module, "ALGO_ID") and hasattr(module, "EXPORT"):
                self.ALGO_REGISTRY[module.ALGO_ID] = module.EXPORT

        if failed:
            logger.warning(f"[auto_register_algos] 以下模块注册失败（不影响其它算法）: {failed}")

    def _create_redis_client(self) -> redis.Redis:
        """构建专用于本类的 Redis 客户端（Pub/Sub + 二进制 GET）。

        优先从 ``_user_redis.connection_pool`` 拷贝 host/port/db/password，并强制
        ``decode_responses=False``（JPEG 与 JSON 均需原始字节/字符串由上层处理）。
        拷贝失败时回退到调用方传入的同一客户端实例。
        """
        try:
            if self._user_redis is None:
                return redis.Redis(decode_responses=False, socket_timeout=5, socket_connect_timeout=5)

            pool = getattr(self._user_redis, 'connection_pool', None)
            if pool is not None:
                conn_kwargs = getattr(pool, 'connection_kwargs', None) or getattr(pool, 'connection_pool', None) or {}
                safe_kwargs = dict(conn_kwargs)
                safe_kwargs.update({
                    'decode_responses': False,
                    'socket_timeout': 5,
                    'socket_connect_timeout': 5
                })
                return redis.Redis(**safe_kwargs)
        except Exception as e:
            logger.warning(f"Failed to create redis from provided client: {e}")

        return self._user_redis

    def _connect_pubsub(self):
        """初始化 ``_redis`` 与 ``_pubsub``，并 ``SUBSCRIBE`` 至 ``self.channel``。"""
        self._redis = self._create_redis_client()
        self._pubsub = self._redis.pubsub(ignore_subscribe_messages=True)
        self._pubsub.subscribe(self.channel)
        logger.info(f"[RedisResultProcessor] Subscribed to channel: {self.channel}")

    def _run(self):
        """消费循环：带指数退避重连；每条 Pub/Sub 消息解析为 key 再 ``GET`` JSON。

        异常处理:
            ``ConnectionError`` / ``OSError`` / ``ValueError``：清空 Pub/Sub 与客户端，退避后重建。
            其它未预期异常：记录栈并短暂 sleep，避免死转。
        """
        backoff = 1.0
        while not self._stopped.is_set():
            try:
                if not self._pubsub:
                    try:
                        self._connect_pubsub()
                        backoff = 1.0
                    except Exception as e:
                        logger.warning(f"Failed to connect pubsub: {e}")
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 30.0)
                        continue

                msg = self._pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if not msg:
                    continue

                if msg.get('type') == 'message':
                    data_str = msg.get('data')
                    logger.debug(f"收到消息: {data_str}")
                    if isinstance(data_str, bytes):
                        try:
                            data_str = data_str.decode('utf-8')
                        except Exception:
                            logger.warning("Received non-UTF8 message, skipping")
                            continue
                    try:
                        logger.info(f"[RedisResultProcessor] 收到 alarm-images: {data_str}")
                        logger.info(f"[TRACE] stage=8_processor_recv ts={int(time.time()*1000)} meta={data_str}")
                        message_data = json.loads(self._redis.get(data_str))

                    except Exception as e:
                        logger.error(f"Failed to parse message JSON: {e}")
                        continue
                    
                    if self._executor:
                        self._executor.submit(self._process_message, message_data)
                    else:
                        self._process_message(message_data)

            except (redis.exceptions.ConnectionError, OSError, ValueError) as e:
                logger.warning(f"[RedisResultProcessor] Listener error, reconnecting: {e}")
                try:
                    if self._pubsub:
                        try:
                            self._pubsub.close()
                        except Exception:
                            pass
                finally:
                    self._pubsub = None

                try:
                    if self._redis:
                        try:
                            self._redis.close()
                        except Exception:
                            pass
                finally:
                    self._redis = None

                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            except Exception as e:
                logger.exception(f"Unexpected error in RedisResultProcessor loop: {e}")
                time.sleep(1.0)

        try:
            if self._pubsub:
                self._pubsub.unsubscribe(self.channel)
                self._pubsub.close()
        except Exception:
            pass
        try:
            if self._redis and hasattr(self._redis, 'close'):
                self._redis.close()
        except Exception:
            pass

    def _get_realtime_coordinates(self, task_id: str) -> Optional[dict]:
        """读取 Redis 中坐标缓存（当前业务链路中多为预留，调用处常注释关闭）。

        Redis Key 构造为 ``coord:task{task_id}``（与写入端约定一致方可命中）。

        Returns:
            {"latitude", "longitude", "speed"} 或 None；字段缺失或 JSON 非法时返回 None 并打 warning。
        """
        try:
            if not task_id or not self._redis:
                return None

            coord_key = f"coord:task{task_id}"
            coord_data = self._redis.get(coord_key)

            if coord_data is None:
                logger.debug(f"No coordinate data found for task_id: {task_id}")
                return None

            # 解析JSON格式的坐标数据
            if isinstance(coord_data, bytes):
                coord_data = coord_data.decode('utf-8')

            coord_info = (json.loads(coord_data)).get("coord")
            if 'latitude' in coord_info and 'longitude' in coord_info and 'speed' in coord_info:
                return {
                    "latitude": coord_info['latitude'],
                    "longitude": coord_info['longitude'],
                    "speed": coord_info['speed']
                }
            else:
                logger.warning(f"Coordinate data missing latitude/longitude/speed: {coord_info}")
                return None

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse coordinate data from Redis: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error getting realtime coordinates: {e}")
            return None

    def _init_llm_api(self):
        """初始化 DashScope 兼容 OpenAI 客户端；失败或缺 Key 时返回 None（二次复审短路）。"""
        try:
            api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get(
                "OPENAI_API_KEY"
            )
            if not api_key:
                logger.warning(
                    "[LLM] 未设置 API Key：请配置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY（阿里云百炼 DashScope）"
                )
                return None
            logger.info("初始化大模型 Client（使用环境变量中的 API Key）")
            return OpenAI(
                api_key=api_key,
                base_url="https://aigw.telecomjs.com/v1",
            )

        except Exception as e:
            logger.exception("大模型 API 加载失败: %s", e)
            return None

    def _process_message(self, message_data: dict):
        """单条告警元数据入口：按 ``TASK`` 与 ``algorithmnum`` 形态路由。

        分支:
            - ``TASK == 'llm'``: 流式大模型结果落盘与 MQTT，不经 ALGO_REGISTRY 检测链。
            - ``algorithmnum`` 为长度>1 的 list: 多模型，按 ``model_code`` 分组（见 ``_process_multi_algorithm``）。
            - 否则: 单算法 ``_process_single_algorithm``。

        异常:
            ``filepath`` 等字段缺失时可能抛错，由外层记录 exception；排查时对照发布端 JSON 字段名。
        """
        start_time = time.time()
        try:
            image_key = message_data.get("image_key")
            file_path = message_data.get("filepath")
            bucket_name = file_path.split('/')[1]
            uavSN = message_data.get("uavsn")
            algorithmnum = message_data.get("algorithmnum")
            dets = message_data.get("detections", [])
            detect_frame = message_data.get("detect_frame", {})
            ts = int(message_data.get("timestamp", 0))
            area = message_data.get("AREA",None)
            task = str(message_data.get('TASK', None))
            sr_key = message_data.get("second_review",0)
            # video-stream tasks: omit "pixel" from the platform payload
            is_video = str(message_data.get("inference_mode", "")).strip().lower() == "video"

            # ---- 从 Redis 读取区域信息 + 实时坐标 ----
            area_from_redis = None
            coordinates = None
            try:
                if uavSN and self._redis:
                    area_key = f"area:task:{uavSN}"
                    area_data = self._redis.get(area_key)
                    if area_data:
                        if isinstance(area_data, bytes):
                            area_data = area_data.decode('utf-8')
                        area_from_redis = area_data
                    coordinates = self._get_realtime_coordinates(task_id=uavSN)
            except Exception:
                pass
            # 优先用 area 字段（直接下发），否则用 Redis 缓存
            effective_area = area if area else area_from_redis
            # -------------------------------------------

            logger.info(f"当前事件的状态：\n 算法编号：{algorithmnum} \n 任务类型: {task} \n 二次复审: {sr_key} \n 源照片: {message_data.get('source_object_key', '')}")

            # 手动拍照/实时大模型链路：
            # deepstream_trt_inference 已经在 metadata 中写入 detect_frame，
            # 这里将其作为 coordinates 传入 _process_llm_result，
            # 最终透传到 MQTT payload 的 data.detect_frame，供前端二次编辑使用。


            if task == 'llm':
                self._process_llm_result(
                    message_data=message_data,
                    image_key=image_key,
                    bucket_name=bucket_name,
                    file_path=file_path,
                    uavSN=uavSN,
                    ts=ts,
                    coordinates=coordinates,
                    detect_frame=detect_frame,
                    is_video=is_video,
                )
                return



            if isinstance(algorithmnum, list) and len(algorithmnum) > 1:
                self._process_multi_algorithm(
                    dets=dets,
                    algorithmnum_list=algorithmnum,
                    image_key=image_key,
                    bucket_name=bucket_name,
                    file_path=file_path,
                    uavSN=uavSN,
                    ts=ts,
                    area=effective_area,
                    task=task,
                    second=sr_key,
                    coordinates=coordinates,
                    detect_frame=detect_frame,
                    is_video=is_video
                )

            else:
                algo_str = str(algorithmnum[0]) if isinstance(algorithmnum, list) else str(algorithmnum)
                self._process_single_algorithm(
                    dets=dets,
                    algo_str=algo_str,
                    image_key=image_key,
                    bucket_name=bucket_name,
                    file_path=file_path,
                    uavSN=uavSN,
                    ts=ts,
                    area=effective_area,
                    task=task,
                    second=sr_key,
                    coordinates=coordinates,
                    detect_frame=detect_frame,
                    is_video=is_video,
                )
            logger.info(f"up->mqtt need time: {time.time() - start_time:.3f}s")
            logger.info(f"[TRACE] stage=9_draw_upload_done 画框上传耗时={time.time() - start_time:.3f}s")
        except Exception as e:
            logger.exception(f"[RedisResultProcessor] Failed to process message data: {e}")

    def _process_single_algorithm(
        self,
        dets: list,
        algo_str: str,
        image_key: str,
        bucket_name: str,
        file_path: str,
        uavSN: str,
        ts: int,
        task: str,
        area: Optional[str] = None,
        second: int = 0,
        coordinates: Optional[dict] = None,
        detect_frame: Optional[dict] = None,
        is_video: bool = False,
    ):
        """对单一算法号执行 ``EXPORT['detect']``，必要时解码帧并 ``_save_result``。

        注册表解析:
            ``detect_key`` 优先为 ``algo_str``（若在 ``ALGO_REGISTRY`` 中），否则回退为 ``task`` 再失败则按 task 名二次回退。

        违停/区域类算法:
            ``detect_key in {'54','55','57','78'}`` 时注入 ``threshold``；``task=='detect'`` 且带 ``area`` 时注入区域与坐标。

        日志:
            ``need_draw`` 为 False 时打印 Skip 行，便于区分「无违规」与「解码失败」。
        """
        tracking_state = self._universal_tracking.setdefault(algo_str, {})
        tracking_state.clear()

        detect_key = algo_str if algo_str in self.ALGO_REGISTRY else task
        threadId = str(uuid.uuid4())
        kwargs = {}
        if detect_key in ['54', '55', '57', '78']:
            kwargs = {'threshold': self._violation_threshold}
        elif detect_key == 'detect' and area is not None:
            kwargs = {'area': area, 'coordinates': coordinates}

        # 透传 area + coordinates（normal_det 等模块使用 + ai-event 上报）
        if area is not None:
            kwargs['area'] = area
        if coordinates is not None:
            kwargs['coordinates'] = coordinates

        # ---- 区域排除（所有算法通用）----
        if area is not None and coordinates is not None:
            from postprocess_utils.geo_area_checker import GeoAreaChecker
            try:
                if isinstance(area, str):
                    area_list = json.loads(area)
                else:
                    area_list = area
                geo_checker = GeoAreaChecker(mqtt_client=None)
                latitude = coordinates.get('latitude')
                longitude = coordinates.get('longitude')
                if latitude is not None and longitude is not None:
                    in_exclusion, _ = geo_checker.check_position_in_area(
                        latitude, longitude, area_list
                    )
                    if in_exclusion:
                        logger.info(f"Process Thread ID : {threadId} | [RedisResultProcessor] Skip: algo: {detect_key} in exclusion zone, dets: {len(dets)}, ts: {ts}")
                        return
            except Exception:
                pass
        # --------------------------------------------------------

        registry_entry = self.ALGO_REGISTRY.get(detect_key)
        if registry_entry and 'detect' in registry_entry:
            try:
                logger.info(f"Process Thread ID : {threadId} | [RedisResultProcessor] RegistryEntry is detect in {registry_entry} ts: {ts}")
                draw_info, need_draw = registry_entry['detect'](dets, tracking_state, **kwargs)
            except TypeError:
                draw_info, need_draw = registry_entry['detect'](dets, tracking_state)
            except Exception:
                logger.exception(f"Error running detect for key: {detect_key}")
                draw_info, need_draw = None, False
        else:
            fallback_key = task if task in self.ALGO_REGISTRY else None
            logger.info(f"Process Thread ID : {threadId} | [RedisResultProcessor] RegistryEntry is task in {task} Fallback key : {fallback_key} ts: {ts}")
            if fallback_key:
                try:
                    draw_info, need_draw = self.ALGO_REGISTRY[fallback_key]['detect'](dets, tracking_state, area, coordinates) if fallback_key == 'detect' and area is not None else self.ALGO_REGISTRY[fallback_key]['detect'](dets, tracking_state)
                except Exception:
                    logger.exception(f"Fallback detect failed for task: {fallback_key}")
                    draw_info, need_draw = None, False
            else:
                logger.warning(f"No detect function for algo_str: {algo_str}, task: {task}")
                draw_info, need_draw = None, False

        logger.info(f"Process Thread ID : {threadId} | [RedisResultProcessor] Review TrackingState : {tracking_state} and DetectKey : {detect_key}, RegistryEntry info : {registry_entry}, Draw info : {draw_info}, Need Draw : {need_draw} ts: {ts}")
        if need_draw and draw_info:
            try:
                frame_np, jpeg_bytes = self._decode_frame(image_key)
            except Exception:
                logger.exception(f"[RedisResultProcessor] Failed to decode frame for key: {image_key}")
                return

            event_detect_frame = self._normalize_detect_frame_for_editing(detect_frame, is_video=is_video)

            if (algo_str != '75' and event_detect_frame
                    and event_detect_frame.get("boxes")):
                try:
                    event_detect_frame["boxes"] = [
                        b for b in event_detect_frame["boxes"]
                        if str(b.get("method", "")) == str(algo_str)
                    ]
                except Exception:
                    pass

            if algo_str in ['54', '57']:
                event_detect_frame = self._build_violation_detect_frame_from_draw_info(
                    draw_info=draw_info,
                    algorithmnum=algo_str,
                    is_video=is_video,
                )
                # _build_violation_detect_frame_from_draw_info 在无违停项时返回 None
                if event_detect_frame is None:
                    logger.info(f"Process Thread ID : {threadId} | [RedisResultProcessor] Skip save: no violation items in draw_info, algo={algo_str}")
                    return

            if algo_str == '75' and need_draw and draw_info:
                event_detect_frame = self._apply_gs_violation_labels(event_detect_frame, draw_info)

            # 兜底检查：确保 boxes 不为空
            if not event_detect_frame or not event_detect_frame.get("boxes"):
                logger.info(f"Process Thread ID : {threadId} | [RedisResultProcessor] Skip save: detect_frame has no boxes, algo={algo_str}")
                return

            logger.info(f"Process Thread ID : {threadId} | [RedisResultProcessor] Final detect_frame boxes count: {len(detect_frame.get('boxes', []) if detect_frame else [])}")
            self._save_result(
                frame_rgb=frame_np,
                draw_info=draw_info,
                ts_ms=ts,
                bucket_name=bucket_name,
                file_path=file_path,
                uavSN=uavSN,
                algorithmnum=algo_str,
                task=task,
                coordinates=coordinates,
                detect_frame=event_detect_frame,
                second=second,
                tracking_state=tracking_state,
                threadId=threadId,
            )
            logger.info(f"Process Thread ID : {threadId} | [RedisResultProcessor] Violation detected: {len(draw_info)} items to draw, timestamp: {ts}")
        else:
            logger.info(f"Process Thread ID : {threadId} | [RedisResultProcessor] Skip save >>>>> need_draw: {need_draw} draw_info: {bool(draw_info)} dets: {len(dets)} track_size: {len(tracking_state)} ts: {ts}")

    def _build_points_from_draw_item(self, item: dict) -> list:
        polygon = item.get("polygon")
        if polygon is not None:
            try:
                pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
                if len(pts) >= 3:
                    return [{"X": float(x), "Y": float(y)} for x, y in pts]
            except Exception:
                pass

        bbox = item.get("bbox")
        if isinstance(bbox, dict):
            x1 = float(bbox.get("x1", 0))
            y1 = float(bbox.get("y1", 0))
            x2 = float(bbox.get("x2", 0))
            y2 = float(bbox.get("y2", 0))
            return [
                {"X": x1, "Y": y1},
                {"X": x2, "Y": y1},
                {"X": x2, "Y": y2},
                {"X": x1, "Y": y2},
            ]

        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            return [
                {"X": x1, "Y": y1},
                {"X": x2, "Y": y1},
                {"X": x2, "Y": y2},
                {"X": x1, "Y": y2},
            ]
        print("输出的结果。。。。。。。。。。。")
        return []

    def _build_violation_detect_frame_from_draw_info(self, draw_info: list, algorithmnum: str, is_video: bool = False) -> Optional[dict]:
        """从 draw_info 构建违停 detect_frame；无违停项时返回 None，避免空 boxes 上报。"""
        from postprocess_utils.chinese_name import CLASS_NAME_TO_CHINESE

        method = str(algorithmnum)
        label_to_id = {}
        en_names = []
        cn_names = []
        boxes = []
        logger.info(f"[RedisResultProcessor] Execute AlgorithmNum 54, 57 -> method : {method} : draw_info : {draw_info}")
        for item in draw_info:
            # 只保留真正违停的目标框，不再传 parking_spot / tcw 区域框
            if not item.get("is_violation", False):
                continue

            raw_label = item.get("label", item.get("type", "unknown"))
            label = self._normalize_violation_label(raw_label, method)
            cn_name = CLASS_NAME_TO_CHINESE.get(label, label)

            points = self._build_points_from_draw_item(item)
            if len(points) < 3:
                continue

            if label not in label_to_id:
                class_id = len(label_to_id)
                label_to_id[label] = class_id
                en_names.append({str(class_id): label})
                cn_names.append({str(class_id): cn_name})
            else:
                class_id = label_to_id[label]

            boxes.append({
                "class_id": class_id,
                "label": label,
                "method": method,
                "cn_name": cn_name,
                "points": points,
                "editable": True,
                "box_role": "target",
                "is_violation": True,
            })

        # 没有任何违停项，返回 None 让调用方直接跳过
        if not boxes:
            logger.info(f"[RedisResultProcessor] _build_violation_detect_frame_from_draw_info: no violation items, returning None for algo={method}")
            return None

        result = {
            method: {
                "en_names": en_names,
                "cn_names": cn_names,
            },
            "boxes": boxes,
        }
        if not is_video:
            result["pixel"] = "4K"
        return result

    def _build_plate_detect_frame_from_draw_info(self, draw_info: list, algorithmnum: str = "35") -> Optional[dict]:
        """算法 35：把 OCR 识别到的车牌号写入 detect_frame（``label`` / ``cn_name``）。

        与 ``_build_violation_detect_frame_from_draw_info`` 同构。关键点：
        ``en_names``/``cn_names`` 是**按 class_id 索引**的，所以**每个车牌号必须分配
        独立 class_id**——否则两个不同车牌号共用一个 id，前端类别表会错乱（多车牌场景）。

        识别失败的框保留原类别名、照常上报（**不返回 None**，避免二次编辑页无框可选）。

        Returns:
            detect_frame dict（含 ``{"35": {...}, "boxes": [...]}``）；无任何框时返回 None。
        """
        from postprocess_utils.chinese_name import CLASS_NAME_TO_CHINESE

        method = str(algorithmnum)
        label_to_id = {}
        en_names = []
        cn_names = []
        boxes = []

        for item in draw_info:
            if item.get("type") != "detection":
                continue

            points = self._build_points_from_draw_item(item)
            if len(points) < 3:
                continue

            plate = str(item.get("ocr_text") or "").strip()
            if plate:
                label, cn_name = plate, plate
            else:
                label = item.get("label") or "license"
                cn_name = CLASS_NAME_TO_CHINESE.get(label, label)

            if label not in label_to_id:
                class_id = len(label_to_id)
                label_to_id[label] = class_id
                en_names.append({str(class_id): label})
                cn_names.append({str(class_id): cn_name})
            else:
                class_id = label_to_id[label]

            boxes.append({
                "class_id": class_id,
                "label": label,
                "method": method,
                "cn_name": cn_name,
                "points": points,
                "editable": True,
                "box_role": "target",
                "is_violation": False,
            })

        if not boxes:
            return None

        return {
            method: {
                "en_names": en_names,
                "cn_names": cn_names,
            },
            "boxes": boxes,
        }

    def _normalize_violation_label(self, label: str, algorithmnum: str) -> str:
        method = str(algorithmnum)

        if method == "54" and label == "car":
            return "illegal_parking_car"

        if method == "57" and label == "electric_vehicle":
            return "illegal_parking_electric_vehicle"

        return label

    def _apply_gs_violation_labels(self, event_detect_frame: dict, draw_info: list) -> dict:
        """
        算法 75 专用：不重建 detect_frame，而是基于现有 detect_frame 修改违停车辆的 label。

        应急车道内的车辆：label → 'illegal_parking_emergency'，cn_name → '应急车道违停'
        应急车道外的车辆：保持原样（label='car'，cn_name='汽车'）
        """
        from postprocess_utils.chinese_name import CLASS_NAME_TO_CHINESE

        if not event_detect_frame or not draw_info:
            return event_detect_frame

        # 从 draw_info 中提取违停车辆的 bbox（用于匹配 detect_frame 中的 box）
        violation_bboxes = []
        for item in draw_info:
            if item.get('is_violation') and item.get('bbox'):
                b = item['bbox']
                violation_bboxes.append({
                    'x1': b['x1'], 'y1': b['y1'],
                    'x2': b['x2'], 'y2': b['y2'],
                })

        if not violation_bboxes:
            return event_detect_frame

        violation_label = 'illegal_parking_emergency'
        violation_cn = CLASS_NAME_TO_CHINESE.get(violation_label, '应急车道违停')

        method_key = '75'
        if method_key not in event_detect_frame:
            event_detect_frame[method_key] = {'en_names': [], 'cn_names': []}

        # 检查是否已有该标签，避免重复添加
        has_label = any(
            list(d.values())[0] == violation_label
            for d in event_detect_frame[method_key].get('en_names', [])
            if d
        )
        if not has_label:
            max_id = 0
            for d in event_detect_frame[method_key].get('en_names', []):
                for k in d:
                    max_id = max(max_id, int(k))
            class_id = max_id + 1 if max_id > 0 else 0
            event_detect_frame[method_key]['en_names'].append({str(class_id): violation_label})
            event_detect_frame[method_key]['cn_names'].append({str(class_id): violation_cn})

        # 匹配并修改 boxes：违停车辆标为 illegal_parking_emergency
        for box in event_detect_frame.get('boxes', []):
            points = box.get('points', [])
            if len(points) < 4:
                continue
            xs = [p['X'] for p in points]
            ys = [p['Y'] for p in points]
            box_x1, box_y1 = min(xs), min(ys)
            box_x2, box_y2 = max(xs), max(ys)

            for vb in violation_bboxes:
                if (abs(box_x1 - vb['x1']) < 2.0 and abs(box_y1 - vb['y1']) < 2.0
                        and abs(box_x2 - vb['x2']) < 2.0 and abs(box_y2 - vb['y2']) < 2.0):
                    box['label'] = violation_label
                    box['cn_name'] = violation_cn
                    box['method'] = method_key
                    box['is_violation'] = True
                    break

        return event_detect_frame

    def _build_event_image_fields(
        self,
        object_name: Optional[str],
        raw_object_name: Optional[str],
        coordinates: Optional[dict],
        task: Optional[str] = None,
        event_source: Optional[str] = None,
    ):
        """
        统一组装平台事件上报里的图片字段。

        目标：
        1. 详情页显示带框结果图
        2. 二次编辑页使用原图 + detect_frame
        3. 不影响普通算法当前逻辑
        """

        '''
        result_object_key = object_name or raw_object_name
        source_object_key = raw_object_name or object_name
        '''

        result_object_key = raw_object_name
        source_object_key = raw_object_name


        if event_source in {"llm_event", "llm_review"}:
            return {
                "object_key": result_object_key,
                #"object_key": source_object_key,
                "raw_object_key": source_object_key,
                "detect_frame": coordinates,
        }

        return {
            "object_key": result_object_key,
            "raw_object_key": source_object_key,
            "detect_frame": coordinates,
    }

    def _normalize_detect_frame_for_editing(self, detect_frame: Optional[dict], is_video: bool = False) -> dict:
        """
        把普通算法的 detect_frame 规范成前端可编辑结构。
        """
        if not isinstance(detect_frame, dict):
            return {}

        normalized = copy.deepcopy(detect_frame)
        boxes = normalized.get("boxes", [])

        if not isinstance(boxes, list):
            normalized["boxes"] = []
            return normalized

        for box in boxes:
            if not isinstance(box, dict):
                continue

            box.setdefault("editable", True)
            box.setdefault("box_role", "target")
            box.setdefault("is_violation", False)

        if not is_video:
            normalized["pixel"] = "4K"

        return normalized

    def _process_multi_algorithm(
        self,
        dets: list,
        algorithmnum_list: list,
        image_key: str,
        bucket_name: str,
        file_path: str,
        uavSN: str,
        ts: int,
        task: str,
        area: Optional[str] = None,
        second: int = 0,
        coordinates: Optional[dict] = None,
        detect_frame: Optional[dict] = None,
        is_video: bool = False,
    ):
        """按检测项 ``model_code`` 分组后逐组调用 ``_process_single_algorithm``。

        特殊规则:
            - ``model_code in ('75','69')``: 传入**完整** ``dets``（多模型联合上下文，与单码分组不同）。
            - ``model_code == '79'``: 当前实现中跳过（显式 continue）。
        """
        dets_by_model: Dict[str, list] = {}
        for det in dets:
            model_code = det.get("model_code")
            if model_code is None:
                logger.warning(f"Detection without model_code, skipping: {det}")
                continue
            if model_code not in dets_by_model:
                dets_by_model[model_code] = []
            dets_by_model[model_code].append(det)

        for model_code, model_dets in dets_by_model.items():
            task = model_dets[-1].get('task','detect')
            # 统一转为字符串比较，防止 JSON 中的整数 model_code 导致类型不匹配
            model_code_str = str(model_code)
            if model_code_str in ('75', '69'):
                self._process_single_algorithm(
                    dets=dets,
                    algo_str=model_code_str,
                    image_key=image_key,
                    bucket_name=bucket_name,
                    file_path=file_path,
                    uavSN=uavSN,
                    ts=ts,
                    task=task,
                    area=area,
                    second=second,
                    coordinates=coordinates,
                    detect_frame=detect_frame,
                    is_video=is_video,
                )
            elif model_code_str == '79':
                continue
            else:
                self._process_single_algorithm(
                    dets=model_dets,
                    algo_str=model_code_str,
                    image_key=image_key,
                    bucket_name=bucket_name,
                    file_path=file_path,
                    uavSN=uavSN,
                    ts=ts,
                    task=task,
                    area=area,
                    second=second,
                    coordinates=coordinates,
                    detect_frame=detect_frame,
                    is_video=is_video,
                )


    def second_parse_llm_output(self, raw_str):
        """解析二次复审模型返回文本，抽取 result/according_to（容忍叙述文字与代码块包裹）。"""
        if not raw_str:
            return []

        raw_str = raw_str.strip()

        # 1) 优先提取 ```json ... ``` 代码块内容（模型常把 JSON 包在代码块里）
        if "```" in raw_str:
            for seg in raw_str.split("```"):
                seg = seg.strip()
                if seg.startswith("json"):
                    seg = seg[4:].strip()
                if seg.startswith("{") and "}" in seg:
                    raw_str = seg
                    break

        # 2) 若仍未拿到对象，直接从第一个 '{' 截到最后一个 '}'（容忍前置叙述文字）
        start = raw_str.find("{")
        end = raw_str.rfind("}")
        if start != -1 and end > start:
            raw_str = raw_str[start:end + 1]

        try:
            data = json.loads(raw_str)
        except Exception as e:
            logger.error("JSON解析失败: %s", e)
            return []

        if isinstance(data, dict):
            return data if "result" in data else []
        elif isinstance(data, list):
            return data

        return []

    def realtime_parse_llm_output(self, raw_str):
        """解析实时流 LLM 返回中的目标列表（``objects`` 或 JSON 数组），供 ``ALGO_REGISTRY['llm']['draw']`` 使用。"""
        if not raw_str:
            return []

        raw_str = raw_str.strip()
        if raw_str.startswith("```"):
            raw_str = raw_str.split("```")[1]
        raw_str = raw_str.replace("json", "", 1).strip()

        try:
            data = json.loads(raw_str)
        except Exception as e:
            logger.error("JSON解析失败: %s", e)
            return []

        if isinstance(data, dict):
            if "objects" in data:
                return data["objects"]
            return []
        elif isinstance(data, list):
            return data

        return []

    def _process_llm_result(
    self,
    message_data: dict,
    image_key: str,
    bucket_name: str,
    file_path: str,
    uavSN: str,
    ts: int,
    coordinates=None,
    detect_frame=None,
    is_video: bool = False,
):
        """处理 TASK == 'llm' 的结果，并按 ai-event 标准结构发送。"""
        tmp_path = None
        raw_tmp_path = None

        try:
            t_start = time.time()
            _src = str(message_data.get("source_object_key", "") or "").rsplit("/", 1)[-1] or image_key
            _lag = int(time.time() * 1000) - int(message_data.get("timestamp", 0) or 0)
            logger.info(f"[LLM-TIME] [{_src}] stage=transit_lag(子进程上报→处理器收到)={_lag}ms")
            frame_bgr, _ = self._decode_frame(image_key)
            logger.info(f"[LLM-TIME] [{_src}] stage=decode_frame cost={(time.time() - t_start) * 1000:.0f}ms")
            # LLM 事件幂等锁：防止同一个 Redis metadata/image_key 被重复消费后重复上报
            dedupe_key = f"dedupe:llm_event:{image_key}"
            try:
                if self._redis:
                    locked = self._redis.set(dedupe_key, "1", nx=True, ex=600)
                    if not locked:
                        logger.info(f"[RedisResultProcessor] [LLM] Skip duplicate event: {image_key}")
                        return
            except Exception:
                logger.exception(f"[RedisResultProcessor] [LLM] Failed to set dedupe lock: {dedupe_key}")

            prompt = message_data.get("prompt", "")
            logger.info(f"[RedisResultProcessor] [LLM] 860 line message date.prompt: {prompt}")
            model_code = message_data.get("model_code", None)
            classes = message_data.get("classes", None)
            algorithm_num = message_data.get("algorithmnum", None)
            according_to = message_data.get("according_to", "")

            if isinstance(algorithm_num, list):
                if len(algorithm_num) == 1:
                    algorithm_num = algorithm_num[0]
                elif len(algorithm_num) == 0:
                    algorithm_num = None

            if file_path is None:
                file_path = ""

            file_path = "/".join(str(file_path).lstrip("/").split("/")[1:])
            object_name = f"{file_path}llm_{ts}.jpg"
            raw_object_name = f"{file_path}raw_llm_{ts}.jpg"

            t_minio = time.time()
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                cv2.imwrite(tmp_path, frame_bgr)

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as raw_tmp_file:
                raw_tmp_path = raw_tmp_file.name
                cv2.imwrite(raw_tmp_path, frame_bgr.copy())

            self.minio_helper.upload_file(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=tmp_path,
                content_type="image/jpeg",
            )
            self.minio_helper.upload_file(
                bucket_name=bucket_name,
                object_name=raw_object_name,
                file_path=raw_tmp_path,
                content_type="image/jpeg",
            )
            logger.info(f"[LLM-TIME] [{_src}] stage=write_tmp+minio_upload(2张) cost={(time.time() - t_minio) * 1000:.0f}ms")

            tid = str(uuid.uuid4())

            if isinstance(detect_frame, dict) and "pixel" not in detect_frame and not is_video:
                detect_frame["pixel"] = "4K"

            payloads = {
                "tid": tid,
                "data": {
                    "minio_bulk": bucket_name,
                    "object_key": raw_object_name,
                    "raw_object_key": raw_object_name,
                    "prompt": prompt,
                    "uavSN": uavSN,
                    "xy": (0.5, 0.5),
                    "detect_frame": detect_frame,
                    "according_to": according_to,
                },
                "timestamp": int(time.time() * 1000),
                "target": uavSN,
                "definition": "ai-event",
                "result": 0,
            }

            global MQTT_TOPIC_ALGORITHM_CONTROL
            t_pub = time.time()
            self.mqtt_client.publish(
                MQTT_TOPIC_ALGORITHM_CONTROL,
                payload=json.dumps(payloads, ensure_ascii=False),
                qos=1,
            )
            logger.info(f"[LLM-TIME] [{_src}] stage=mqtt_publish(ai-event) cost={(time.time() - t_pub) * 1000:.0f}ms | PROCESSOR_TOTAL={(time.time() - t_start) * 1000:.0f}ms")

            logger.info(f"[RedisResultProcessor] [LLM] Successful to pub msg: {payloads}")
            logger.info(
                f"[RedisResultProcessor] [LLM] MinIO 上传成功: "
                f"{bucket_name}/{object_name}, raw={raw_object_name}"
            )

        except Exception as e:
            logger.exception(f"[RedisResultProcessor] [LLM] Failed to process LLM result: {e}")

        finally:
            for path in (tmp_path, raw_tmp_path):
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except Exception:
                        pass

    def _decode_frame(self, image_key):
        """根据 ``image_key`` 从 Redis 读取 JPEG，解码为 BGR ndarray。

        Returns:
            tuple: ``(frame_bgr: numpy.ndarray, jpeg_bytes: bytes)``

        Raises:
            ValueError: Key 不存在或 ``cv2.imdecode`` 失败。
        """
        jpeg_bytes = self._redis.get(image_key)
        if jpeg_bytes is None:
            raise ValueError(f"Key {image_key} not found or expired in Redis.")

        np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Failed to decode image from key: {image_key}")

        return frame, jpeg_bytes

    def _save_result(self, frame_rgb: np.ndarray, draw_info, ts_ms: int, bucket_name=None, file_path=None, uavSN=None, algorithmnum=None,  task=None,  coordinates=None, detect_frame=None, second=0, tracking_state=None, threadId=None):
        """将 ``draw_info`` 渲染到帧上，写临时 JPEG，按策略上传 MinIO 并 MQTT 上报。

        绘制键 ``detect_key`` 选取（与单路处理一致）:
            1. 算法号在注册表中则用之。
            2. 否则在 **非 75** 场景下可回退 ``task`` 或 ``detect``/``obb``/``segment``。

        算法 75:
            禁止回退到 ``segment`` 的 draw（避免把分割掩模画进告警图）；未命中注册表时直接放弃绘制。

        算法 35:
            先 ``ocr_draw`` 再链式 ``ocr``（传入 ``self.ocr_model``）。

        二次复审:
            ``second`` 真时走 ``_llm_executor`` 异步 ``_perform_llm_review``；否则同步双传 MinIO 并发 MQTT。

        Args:
            frame_rgb: 已解码 BGR 图像（参数名历史原因仍写 rgb，实为 OpenCV BGR）。
            draw_info: 各算法 detect 阶段返回的绘制元数据。
            ts_ms: 毫秒时间戳，用于对象名防碰撞。
        """
        algo_str = str(algorithmnum) if algorithmnum is not None else None

        detect_key = None
        if algo_str and algo_str in self.ALGO_REGISTRY:
            detect_key = algo_str
        elif task and task in self.ALGO_REGISTRY:
            if algo_str == '75':
                logger.warning(f"Process Thread ID : {threadId} | Algorithm 75 should use algo_str='75' as detect_key, not task: {task}. Skipping task-level draw.")
            else:
                detect_key = task
        else:
            if algo_str == '75':
                logger.warning(f"Process Thread ID : {threadId} | Algorithm 75 not found in ALGO_REGISTRY, cannot fallback to task-level draw")
            else:
                for fallback in ('detect', 'obb', 'segment'):
                    if fallback in self.ALGO_REGISTRY:
                        detect_key = fallback
                        break

        img_bgr = None
        name = None
        center_point = None

        if detect_key:
            registry_entry = self.ALGO_REGISTRY.get(detect_key, {})
            draw_func = registry_entry.get('draw')
            if draw_func:
                try:
                    if detect_key == '80':
                        img_bgr, name  = draw_func(frame_rgb.copy(), draw_info, tracking_state)
                    else:
                        img_bgr, name = draw_func(frame_rgb.copy(), draw_info)
                except Exception:
                    logger.exception(f"Process Thread ID : {threadId} | Registry draw failed for key={detect_key}")
                    img_bgr, name = None, None

        if img_bgr is None:
            if algo_str == '35' and isinstance(self.ALGO_REGISTRY.get('35'), dict) and 'ocr' in self.ALGO_REGISTRY['35']:
                try:
                    img_bgr, name = self.ALGO_REGISTRY["35"]['ocr_draw'](frame_rgb.copy(), draw_info)
                except Exception:
                    logger.exception("Process Thread ID : {threadId} | Fallback normal draw failed for algo 35")
                    img_bgr, name = None, None

                try:
                    ocr_func = self.ALGO_REGISTRY['35'].get('ocr')
                    if ocr_func:
                        img_bgr, name = ocr_func(img_bgr, draw_info, model=self.ocr_model)
                except Exception:
                    logger.exception("Process Thread ID : {threadId} | OCR callback failed for algo 35")

        # ---- 算法 35：把 OCR 车牌号逐个写回 detect_frame（仅 35；其他算法走不到这里）----
        # draw_info 的 item 已被 License_plate.ocr() 写入 ocr_text/ocr_score，
        # 这里重建 detect_frame，让每个车牌号成为独立 class_id，供前端二次编辑使用。
        # 位置必须在 OCR 之后、下面的 second(LLM 复审) 分支之前，两条上报路径才都覆盖。
        if algo_str == '35' and isinstance(detect_frame, dict):
            try:
                rebuilt = self._build_plate_detect_frame_from_draw_info(draw_info, algo_str)
                if rebuilt:
                    # 保留原 detect_frame 的 pixel 标记（承载 is_video 语义，避免给本函数加参数）
                    if "pixel" in detect_frame:
                        rebuilt["pixel"] = detect_frame["pixel"]
                    detect_frame = rebuilt
                    # classes 也统一用清洗后的车牌号，避免水印/广告字混入
                    # （draw_info 的 ocr_text 已被 License_plate._pick_plate 过滤）
                    plates = [str(it.get("ocr_text")) for it in draw_info if it.get("ocr_text")]
                    if plates:
                        name = "、".join(plates)
                    logger.info(
                        f"Process Thread ID : {threadId} | [algo35] detect_frame 已写回车牌号: "
                        f"{[b.get('label') for b in rebuilt['boxes']]}"
                    )
            except Exception:
                logger.exception(
                    f"Process Thread ID : {threadId} | [algo35] 构建车牌 detect_frame 失败，沿用原 detect_frame"
                )

        if img_bgr is None and detect_key and detect_key not in (algo_str,):
            if algo_str == '75' and detect_key == 'segment':
                logger.error(f"Process Thread ID : {threadId} | Algorithm 75 should not fallback to segment draw function. This would draw segmentation mask which is not allowed.")
                pass
            else:
                registry_entry = self.ALGO_REGISTRY.get(detect_key, {})
                draw_func = registry_entry.get('draw')
                if draw_func:
                    try:
                        if detect_key == "detect":
                            img_bgr, name  = draw_func(frame_rgb.copy(), draw_info)  #, center_point

                        else:
                            img_bgr, name = draw_func(frame_rgb.copy(), draw_info)
                    except Exception:
                        logger.exception(f"Registry task-level draw failed for key={detect_key}")
                        img_bgr, name = None, None

        if img_bgr is None or name is None:
            logger.warning(f"Process Thread ID : {threadId} | No image produced for algo: {algo_str}, skip upload")
            return

        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                cv2.imwrite(tmp_path, img_bgr)

            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                row_tmp_path = tmp_file.name
                cv2.imwrite(row_tmp_path, frame_rgb.copy())

            if file_path is None:
                file_path = ""
            file_path = "/".join(str(file_path).lstrip("/").split("/")[1:])
            object_name = f"{file_path}{ts_ms}.jpg"
            raw_object_name = f"{file_path}raw_{ts_ms}.jpg"


            if not hasattr(self, '_llm_executor'):
                self._llm_executor = ThreadPoolExecutor(max_workers=2)

            logger.info(f"Process Thread ID : {threadId} | 当前大模型复审状态：{second}")
            if second:
                llm_future = self._llm_executor.submit(self._perform_llm_review,  img_bgr, bucket_name, algo_str, algorithmnum, name, object_name, raw_object_name, uavSN, tmp_path, row_tmp_path, coordinates,detect_frame)
            else:
                self.minio_helper.upload_file(
                    bucket_name=bucket_name,
                    object_name=object_name,
                    file_path=tmp_path,
                    content_type="image/jpeg"
                )
                self.minio_helper.upload_file(
                    bucket_name=bucket_name,
                    object_name=raw_object_name,
                    file_path=row_tmp_path,
                    content_type="image/jpeg"
                )

                prompt = ""
                
                try:
                    if algorithmnum is not None:
                        prompt = model_prompt.get(int(algorithmnum), "")
                except Exception:
                    prompt = ""

                # tid = str(uuid.uuid4())

                payloads = {
                    "tid": threadId,
                    "data": {
                        "minio_bulk": bucket_name,
                        "object_key": raw_object_name,
                        "raw_object_key": raw_object_name,
                        "model_code": algo_str,
                        "classes": name,
                        "algorithmNum": algorithmnum,
                        "prompt": prompt,
                        "uavSN": uavSN,
                        "xy": (0.5, 0.5),
                        "coordinates": coordinates,
                        "detect_frame": detect_frame,
                        "according_to": "",
                    },
                    "timestamp": int(time.time() * 1000),
                    "target": uavSN,
                    "definition": "ai-event",
                    "result": 0
                }
                self.mqtt_client.publish(MQTT_TOPIC_ALGORITHM_CONTROL, payload=json.dumps(payloads), qos=1)
                logger.info(f"[TRACE] stage=10_mqtt_publish_final ts={int(time.time()*1000)} path=直发")
                logger.info(f"Process Thread ID : {threadId} | [RedisResultProcessor] Successful to pub msg two : {MQTT_TOPIC_ALGORITHM_CONTROL} : {payloads} ")
                logger.info(f"Process Thread ID : {threadId} | [RedisResultProcessor] MinIO 上传成功: {bucket_name}/{object_name}, raw={raw_object_name}")

                try:
                    os.unlink(tmp_path)
                    os.unlink(row_tmp_path)
                except Exception:
                    pass

                logger.info(f"Process Thread ID : {threadId} | [RedisResultProcessor] Uploaded image to MinIO: {self.bucket_name}/{object_name}")
        except Exception as e:
            logger.exception(f"Process Thread ID : {threadId} | [RedisResultProcessor] Failed to upload image to MinIO: {e}")

    def _perform_llm_review(self, img_bgr, bucket_name, algo_str, algorithmnum, name, object_name, raw_object_name, uavSN, tmp_path, row_tmp_path, coordinates, detect_frame):
        """异步二次复审：``model_prompt[algorithmnum]`` + ``Second_review``；仅当模型肯定时上传并 MQTT。

        失败/否决策:
            ``llm_client`` 缺失、API 异常、或 ``result`` 为假：不上传 MinIO；临时文件在 finally 中清理。

        Note:
            ``algorithmnum`` 需为 ``model_prompt`` 合法整型下标，否则 KeyError 由上层 exception 记录。
        """
        try:
            if not self.llm_client:
                logger.warning(
                    "[LLM] Client 未初始化，跳过二次复审，请配置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY"
                )
                try:
                    os.unlink(tmp_path)
                    os.unlink(row_tmp_path)
                except OSError:
                    pass
                return
            _, jpeg_bytes = cv2.imencode('.jpg', img_bgr)
            jpeg_bytes = jpeg_bytes.tobytes()

            prompt = model_prompt.get(int(algorithmnum), "")
            logger.info(f"当前复审的Prompt是：{prompt}")

            _,llm_result = Second_review(self.llm_client, jpeg_bytes, prompt)
            logger.info(f"ori-llm_result:{llm_result}")

            llm_result = self.second_parse_llm_output(llm_result)
            logger.info(f"llm_result:{llm_result}")
            if not isinstance(llm_result, dict):
                logger.error("[LLM] 复审输出解析失败，跳过本次事件: %s", llm_result)
                return
            llm_as_result =  llm_result.get('result',False)
            according_to = llm_result.get('according_to', '无法正确解析')

            if llm_as_result:

                self.minio_helper.upload_file(
                        bucket_name=bucket_name,
                        object_name=object_name,
                        file_path=tmp_path,
                        content_type="image/jpeg"
                    )
                self.minio_helper.upload_file(
                    bucket_name=bucket_name,
                    object_name=raw_object_name,
                    file_path=row_tmp_path,
                    content_type="image/jpeg"
                )


                tid = str(uuid.uuid4())

                payloads = {
                    "tid": tid,
                    "data": {
                        "minio_bulk": bucket_name,
                        "object_key": raw_object_name,
                        "raw_object_key": raw_object_name,
                        "model_code": algo_str,
                        "classes": name,
                        "algorithmNum": algorithmnum,
                        "prompt": prompt,
                        "uavSN": uavSN,
                        "xy": (0.5, 0.5),
                        "coordinates": coordinates,
                        "detect_frame": detect_frame,
                        "according_to": according_to,
                    },
                    "timestamp": int(time.time() * 1000),
                    "target": uavSN,
                    "definition": "ai-event",
                    "result": 0
                }

                global MQTT_TOPIC_ALGORITHM_CONTROL
                self.mqtt_client.publish(MQTT_TOPIC_ALGORITHM_CONTROL, payload=json.dumps(payloads), qos=1)
                logger.info(f"[TRACE] stage=10_mqtt_publish_final ts={int(time.time()*1000)} path=LLM复审")
                logger.info(f"[RedisResultProcessor] Successful to pub msg one : {MQTT_TOPIC_ALGORITHM_CONTROL} :{payloads} ")
                logger.info(f"[RedisResultProcessor] MinIO 上传成功: {bucket_name}/{object_name}, raw={raw_object_name}")
            else:
                logger.info(f"复审后无待检测目标,跳过")
            try:
                os.unlink(tmp_path)
                os.unlink(row_tmp_path)
            except Exception as e:
                logger.error(f"=== perform_llm_review === 清理文件出错： {e}")
        except Exception as e:
            logger.exception(f"[RedisResultProcessor] Failed to perform LLM review: {e}")

    def _color_for(self, class_id: int) -> Tuple[int, int, int]:
        """返回 BGR 颜色；未在 ``DEFAULT_COLORS`` 内时用确定性哈希生成，保证同类 id 颜色稳定。"""
        if class_id in self.DEFAULT_COLORS:
            return self.DEFAULT_COLORS[class_id]
        rng = (37 * class_id + 17) % 255
        return (int((rng * 3) % 255), int((rng * 7) % 255), int((rng * 11) % 255))

def on_message(client, userdata, msg):
    """占位回调；本进程以 Redis 消费为主，MQTT 侧主要用于发布。"""
    pass

def on_connect(client, userdata, flags, rc):
    """Paho 连接成功：订阅控制与数据同步主题（与 ``__main__`` 内变量一致）。"""
    if rc == 0:
        logger.info(f"Connected to MQTT broker: {MQTT_BROKER}, port: {MQTT_PORT}")
        client.subscribe([(MQTT_TOPIC_ALGORITHM_CONTROL, 1),(MQTT_TOPIC_DATA_SYNC, 1)])
        logger.info(f"Subscribed to topic: { MQTT_TOPIC_ALGORITHM_CONTROL}")
    else:
        logger.info(f"Failed to connect to MQTT broker, return code: {rc}")

def on_disconnect(client, userdata, rc):
    """非零 return code 时尝试 ``reconnect``（简单重连策略，生产可换用循环器）。"""
    if rc != 0:
        logger.info(f"Unexpected MQTT disconnection, return code: {rc}")
        logger.info(f"Attempting to reconnect...")
        try:
            client.reconnect()
        except Exception as e:
            logger.error(f"Reconnect failed: %s", e)

def mqtt_client_connect(mqtt_client):
    """阻塞 connect 并 ``loop_start``；失败打印错误并 ``sys.exit(1)``。"""
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        mqtt_client.loop_start()
    except Exception as e:
        logger.error(f"Error connecting to MQTT broker: %s", e)
        sys.exit(1)

def disconnect(mqtt_client):
    """停止网络循环并断开 broker。"""
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    logger.info(f"Disconnected from MQTT broker")

# 直接运行：读取 YAML，连接 Redis / MQTT / MinIO，启动消费者（非库引用场景）。
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force = True
    )

    from common.config_loader import get_profile, get_config_path, load_config
    logger.info(f"[结果处理器] 当前环境: {get_profile()}, 配置文件: {get_config_path()}")

    _config = load_config()
    redis_config = _config.get('redis',{})
    mqtt_config = _config.get('mqtt', {})
    topics = mqtt_config.get('topics', {})
    MQTT_TOPIC_ALGORITHM_CONTROL = topics.get('algorithm_control', 'ai/calculate/event')
    MQTT_TOPIC_DATA_SYNC = topics.get('data_sync', 'ai/calculate/event')
    MQTT_BROKER = mqtt_config.get('host', '58.215.0.14')
    MQTT_PORT = mqtt_config.get('port', 1883)
    MQTT_USERNAME = mqtt_config.get('username', 'dock2')
    MQTT_PASSWORD = mqtt_config.get('password', '123456')
    MQTT_CLIENT_ID = f"uploads_{time.strftime('%H%M%S', time.localtime(time.time()))}"


    REDIS_HOST = redis_config.get("host","localhost")
    REDIS_PORT = redis_config.get("port","6379")
    REDIS_DB = redis_config.get("database","0")
    REDIS_PASSWORD = redis_config.get('password',None)
    REDIS_KEY_PREFIX = "inference:task:"
    try:
        redis_client = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=REDIS_DB,
                    password=REDIS_PASSWORD,
                    decode_responses=False,
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
        redis_client.ping()
        mqtt_client = mqtt.Client(client_id=MQTT_CLIENT_ID)
        mqtt_client.on_connect = on_connect
        mqtt_client.on_message = on_message
        mqtt_client.on_disconnect = on_disconnect
        mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        mqtt_client_connect(mqtt_client)
        minio_client = init_minio_client(get_config_path())
        minio_helper = MinioHelper(minio_client)
        results_processer = RedisResultProcessor(redis_client, minio_helper,mqtt_client=mqtt_client)
        results_processer.start()
        time.sleep(2)
        while True:
            print("wait")
            time.sleep(1)
    except KeyboardInterrupt:
        results_processer.stop()
        mqtt_client.disconnect()
