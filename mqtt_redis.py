#!/usr/bin/env python3
"""
UAV 算法推理编排服务（MQTT + Redis + 子进程 DeepStream）。

职责概要
--------
- 经 MQTT 接收平台侧控制指令（启动/停止/切换算法、航线索引等），校验握手后执行业务逻辑。
- 将任务进程状态写入 Redis；向推理子进程通过 Redis Pub/Sub 下发算法启停与模型切换消息。
- 以 multiprocessing 子进程拉起 ``common.deepstream_trt_inference.main``，实现拉流、推理与推流闭环。

协议约定
--------------------
入站消息体为 JSON，统一使用字段 ``definition`` 表示消息类型，``data`` 为业务负载，``target`` 通常为设备 SN。
握手完成前除 ``handshake_auth_reply`` 相关流程外，其余业务消息一律丢弃。
应答通过 ``Config.MQTT_SEND_SERVER_REPLY`` 主题发布，载荷中 ``definition`` 与请求侧成对（如
``serve_start_reply`` → ``serve_start``）。
"""

from multiprocessing.context import Process
from multiprocessing.context import Process
import os
import sys
import json
import time
import signal
import logging
import multiprocessing as mp
import uuid
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from importlib import reload
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path
import yaml
import paho.mqtt.client as mqtt
import redis
from utils.red import RedisAlgorithmManager
from common.watchdog import Watchdog

HEARTBEAT_FILE = "/tmp/uav_pipeline_heartbeat"


# 可选：启用 common.gpu_load_balancer 时在 TaskManager 中初始化 GPU 分配策略（当前未接入）。

def load_config():
    """加载服务侧 YAML 配置（MQTT、Redis、日志等）。

    通过 ``common/config_loader`` 切换。
    切换环境只需修改 ``common/config-dev.yaml`` 中的 ``profile`` 值。

    Returns:
        dict: 解析后的配置根对象；文件缺失或解析失败时进程退出。

    Note:
        默认路径为容器内 ``/tensorrt/uav-algorithm-pipeline/common/config-dev-main.yaml``。
    """
    from common.config_loader import get_profile, get_config_path, load_config as _load
    print(f"[启动] 当前环境: {get_profile()}, 配置文件: {get_config_path()}")
    return _load()


_config = load_config()
print(f"_config{_config}")
mqtt_config = _config.get('mqtt', {})
redis_config = _config.get('redis', {})
topics = mqtt_config.get('topics', {})
logging_config = _config.get('logging', {})


def touch_heartbeat():
    """更新外部健康检查依赖的心跳文件时间戳（供 Watchdog/编排探活）。"""
    with open(HEARTBEAT_FILE, "w") as f:
        f.write(str(time.time()))


class Config:
    """全局静态配置：MQTT、Redis、推理进程与握手参数。

    主题与 Redis Key 约定：
        MQTT_SEND_SERVER_REPLY: 平台可见的控制侧应答主题。
        MQTT_TOPIC_HANDSHAKE: 握手及周期性 ``serve_status`` 上报主题。
        MQTT_TOPIC_DATA_SYNC: 配置中的数据同步主题（与日志启动信息一致）。
        REDIS_KEY_PREFIX: 任务快照 ``inference:task:{task_id}``。
        REDIS_KEY_ROUTE_PREFIX: 航线索引缓存 ``route:task:{device_sn}``，TTL 600s。
    """

    MQTT_BROKER = mqtt_config.get('host', '58.215.0.14')
    MQTT_PORT = mqtt_config.get('port', 1883)
    MQTT_USERNAME = mqtt_config.get('username', 'dock2')
    MQTT_PASSWORD = mqtt_config.get('password', '123456')
    MQTT_SEND_SERVER_REPLY = "ai/calculate/control/server_reply"
    MQTT_SED_EVENT_REPLY = "ai/calculate/event/event_reply"
    MQTT_TOPIC_DATA_SYNC = topics.get('data_sync', 'ai/calculate/event/event_reply')
    MQTT_TOPIC_ALGORITHM_CONTROL = topics.get('algorithm_control', 'ai/calculate/control/server')
    MQTT_TOPIC_HANDSHAKE = "ai/calculate/event"

    MQTT_CLIENT_ID = mqtt_config.get('clientid', 'mqtt_forwarder')

    REDIS_HOST = redis_config.get("host", "localhost")
    REDIS_PORT = redis_config.get("port", "6379")
    REDIS_DB = redis_config.get("database", "0")
    REDIS_PASSWORD = redis_config.get('password', None)
    REDIS_KEY_PREFIX = "inference:task:"
    REDIS_KEY_ROUTE_PREFIX = "route:task:"

    DEEPSTREAM_SCRIPT = os.getenv("DEEPSTREAM_SCRIPT", "./deepstream_trt_inference.py")
    MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "20"))

    # 图片模式：True 时推理子进程改用 common.image_trt_inference（MinIO 图片检测），
    # 不再拉 RTSP 流；平台手动拍照通知经 MQTT_TOPIC_PHOTO_NOTIFY 下发。
    IMAGE_MODE = bool(_config.get('image', {}).get('enabled', False))
    # 平台无人机手动拍照上报主题（配置项 topics.photo_notify）
    MQTT_TOPIC_PHOTO_NOTIFY = topics.get('photo_notify', 'ai/calculate/control/photo')
    # 推理终态反馈主题（算法→后端，配置项 topics.photo_result）
    MQTT_TOPIC_PHOTO_RESULT = topics.get('photo_result', 'ai/calculate/photo_result')

    HANDSHAKE_TIMEOUT = 60
    HANDSHAKE_RETRY_INTERVAL = 15
    HANDSHAKE_MAX_RETRIES = 99999

    LOG_LEVEL = logging_config.get('level', 'INFO')
    LOG_FILE = logging_config.get('file', 'inference_service.log')

    STATUS_REPORT_INTERVAL = 5


_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), Config.LOG_FILE)
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)


def _inference_wrapper(log_path: str, *args):
    """子进程入口 wrapper：先重定向 stdout/stderr 到独立日志文件，再调用 main。

    必须是模块级函数（非嵌套），因为 multiprocessing spawn 模式需要 pickle target。

    当配置文件 ``mock.enabled: true`` 时，路由到 ``common.mock_inference.main``，
    绕过 DeepStream + TensorRT，使用本地图片模拟帧输入。
    """
    import sys as _sys
    _sys.stdout = open(log_path, 'a')
    _sys.stderr = open(log_path, 'a')

    # args[-1] carries the inference mode: image | video
    mode = str(args[-1]) if args else "image"
    call_args = args[:-1] if args else args

    mock_config = _config.get('mock', {})
    if mock_config.get('enabled', False):
        from common.mock_inference import main as _main
    elif mode == "video":
        from common.video_trt_inference import main as _main
        print("[ENTRY] mode=video -> common.video_trt_inference")
    else:
        from common.deepstream_trt_inference import main as _main
        print("[ENTRY] mode=image -> common.deepstream_trt_inference")

    _main(*call_args)


def run_deepstream(
        uav_sn,
        input_rtsp,
        output_rtmp,
        algorithm_list,
        filePath,
        assigned_gpu_id,
        task_id,
        llm_list,
        mode="image",
):
    """推理子进程入口：拉起单路 DeepStream 管线进程并阻塞直至退出。

    当前实现为每任务一个 multiprocessing.Process，入口为本模块 run_deepstream；
    其内部再 spawn 子进程执行 common.deepstream_trt_inference.main，
    以便统一信号处理与日志语义。

    Args:
        uav_sn: 无人机/设备序列号，传入推理进程。
        input_rtsp: 输入流地址（RTSP/RTMP 等，由下游管线解析）。
        output_rtmp: 输出流地址。
        algorithm_list: 算法清单，元素含 ``algorithmNum``、``paramers``、``second_review`` 等。
        filePath: 任务关联路径（如录制或配置目录，由下游使用）。
        assigned_gpu_id: 期望 GPU 索引；实际可见设备由子进程环境决定。
        task_id: 任务标识，通常与设备 SN 一致，用于 Redis 频道订阅。
        llm_list: 可选的大模型侧配置列表，传入推理进程。

    Note:
        SIGTERM/SIGINT 将终止当前进程内的推理子进程并退出。
    """
    processes = []

    def signal_handler(signum, frame):
        """转发终止信号：优雅终止推理子进程，必要时 SIGKILL。"""
        logger.info(f"[{task_id}] 收到信号 {signum}，正在终止所有算法子进程...")
        for p in processes:
            if p.is_alive():
                logger.info(f"[{task_id}] 终止子进程 PID: {p.pid}")
                p.terminate()
        # 给子进程一点时间退出
        time.sleep(2)
        for p in processes:
            if p.is_alive():
                logger.warning(f"[{task_id}] 子进程 {p.pid} 未退出，强制杀死")
                p.kill()
        logger.info(f"[{task_id}] 所有算法子进程已终止，主进程退出")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        logger.info(f"[{task_id}] 启动推理服务，分配GPU: {assigned_gpu_id}")
        logger.info(f"[{task_id}] 输入流: {input_rtsp}")
        logger.info(f"[{task_id}] 输出流: {output_rtmp}")
        logger.info(f"[{task_id}] 算法列表: {[a.get('algorithmNum') for a in algorithm_list]}")

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"inference_{task_id}.log")

        p = mp.Process(
            target=_inference_wrapper,
            args=(log_path, input_rtsp, output_rtmp, task_id, algorithm_list, filePath, assigned_gpu_id, llm_list, mode),
            name=f"inference-{task_id}"
        )
        p.start()
        processes.append(p)
        logger.info(f"[{task_id}] 启动算法进程, PID: {p.pid}, 推理日志: {log_path}")

        logger.info(f"[{task_id}] 主控进程 PID: {os.getpid()}，等待子进程...")

        # 等待所有算法进程结束
        for p in processes:
            p.join()

        logger.info(f"[{task_id}] 所有推理完成")

    except KeyboardInterrupt:
        logger.info(f"[{task_id}] 任务被用户中断")
        signal_handler(signal.SIGINT, None)  # 手动调用清理
    except Exception as e:
        logger.error(f"[{task_id}] 推理错误: {e}", exc_info=True)
    finally:
        logger.info(f"[{task_id}] 子进程终止 (PID: {os.getpid()})")


class TaskManager:
    """推理任务生命周期与 Redis 协调。

    维护内存中的 ``task_id -> Process`` 映射，并与 Redis 中的任务快照同步；
    通过 ``redis.publish(task_id, ...)`` 向 ``deepstream_trt_inference`` 子进程下发
    算法增量启动与局部停止指令。
    """

    def __init__(self, num_gpus=2):
        self.tasks: Dict[str, mp.Process] = {}
        self.redis_client = None
        self.gpu_load_balancer = None
        self.num_gpus = num_gpus

        self._init_redis()

        signal.signal(signal.SIGCHLD, self._handle_sigchld)

    def _init_redis(self):
        """建立 Redis 连接并 Ping；失败时 ``redis_client`` 置为 None，上层调用需容错。"""
        try:
            self.redis_client = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                db=Config.REDIS_DB,
                password=Config.REDIS_PASSWORD,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            logger.info(f"Redis connected: {Config.REDIS_HOST}:{Config.REDIS_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None

    def save_task_status(self, task_id: str, status: str, info: dict = None):
        """将任务快照写入 Redis（TTL 24h）。

        Args:
            task_id: 任务键，一般为设备 SN，对应 Redis Key ``inference:task:{task_id}``。
            status: 枚举语义字符串，如 ``running`` / ``stopped`` / ``exited`` / ``error``。
            info: 可选扩展字段，与 ``task_id``、``status``、``timestamp`` 合并存储。
        """
        if not self.redis_client:
            return
        try:
            task_data = {
                "task_id": task_id,
                "status": status,
                "timestamp": datetime.now().isoformat(),
            }
            if info:
                task_data.update(info)
            key = f"{Config.REDIS_KEY_PREFIX}{task_id}"
            self.redis_client.setex(
                key,
                86400,  # 24小时过期
                json.dumps(task_data)
            )
            logger.debug(f"Task status saved to Redis: {task_id} -> {status}")
        except Exception as e:
            logger.error(f"Failed to save task status to Redis: {e}")

    def get_task_status(self, task_id: str) -> Optional[dict]:
        """读取任务快照 JSON；无键或 Redis 不可用时返回 None。"""
        if not self.redis_client:
            return None
        try:
            key = f"{Config.REDIS_KEY_PREFIX}{task_id}"
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Failed to get task status from Redis: {e}")
        return None

    def switch_model(self, channel: str, new_algorithm_list: list, new_llm_list: list,
                     llm_permission=None
                     ) -> bool:
        """热增量：仅对「新增」的算法与 LLM 配置下发 Redis 消息。

        根据当前 Redis 快照中的 ``algorithm_list`` 计算差集，向频道 ``channel``（设备 SN）
        发布 ``start_algorithm_list`` / ``start_llm_list``，供推理进程内订阅处理。
        若既无新算法也无新 LLM 配置，返回 True 且不发布。

        Args:
            channel: Redis Pub/Sub 频道名，须与运行中任务的 ``task_id`` 一致。
            new_algorithm_list: 期望的全集算法列表（含 ``algorithmNum``、``paramers`` 等）。
            new_llm_list: 待追加启用的 LLM 配置列表。
            llm_permission: 平台大模型权限（0 无权限 / 1 有权限）；None 表示本次不下发。

        Returns:
            bool: 发布成功且状态更新成功为 True；无任务快照或 Redis 异常为 False。
        """
        try:
            status_data = self.get_task_status(channel)
            if not status_data:
                logger.error(f"No task status found for {channel}")
                return False

            current_list = status_data.get("algorithm_list", [])
            current_ids = [a.get('algorithmNum') for a in current_list]

            new_ids = [a.get('algorithmNum') for a in new_algorithm_list]

            add_ids = [i for i in new_ids if i not in current_ids]

            if not add_ids and not new_llm_list and llm_permission is None:
                logger.info(f"No new algorithms to add for task {channel}")
                return True

            add_algorithm_list = [
                a for a in new_algorithm_list
                if a.get("algorithmNum") in add_ids
            ]

            message = {
                "start_algorithm_list": add_algorithm_list,
                "start_llm_list": new_llm_list,
            }
            # 透传给推理子进程：0 关闭大模型检测权限，1 允许（缺省则子进程保持原值）
            if llm_permission is not None:
                message["llm_permission"] = llm_permission
            print(message)
            try:
                self.redis_client.publish(str(channel), json.dumps(message))

                status_data["algorithm_list"] = [message]
                if llm_permission is not None:
                    status_data["llm_permission"] = llm_permission

                self.save_task_status(channel, "running", status_data)

                logger.info(f"Published switch message (add only): {message}")
                return True

            except Exception as e:
                logger.error(f"Failed to publish switch message: {e}")
                return False


        except Exception as e:
            logger.error(f"Error in switch_model: {e}")
            return False

    def start_task(self, task_id, input_rtsp=None, output_rtmp=None,
                   algorithm_list=None, filePath=None, uavSN=None,
                   llm_list=None, mode="image", llm_permission=None,
                   ) -> tuple:
        """启动一条推理任务：创建 ``run_deepstream`` 子进程并写入 Redis 快照。

        Args:
            task_id: 任务 ID，与 Redis 订阅频道及快照 Key 一致，通常为设备 SN。
            input_rtsp: 输入流 URL。
            output_rtmp: 输出流 URL。
            algorithm_list: 初始算法列表。
            filePath: 下游使用的路径参数。
            uavSN: 设备序列号，传入子进程。
            llm_list: 可选 LLM 配置。

        Returns:
            并发已满时返回 False（历史行为，非二元组）。
            否则 (True, "") 表示启动成功；(False, str(exception)) 表示异常。

        Note:
            GPU 索引当前默认为 0；若挂载 ``gpu_load_balancer`` 则由均衡器选择。
        """
        active_tasks = sum(1 for p in self.tasks.values() if p.is_alive())
        if active_tasks >= Config.MAX_CONCURRENT_TASKS:
            logger.warning(f"Max concurrent tasks reached: {active_tasks}/{Config.MAX_CONCURRENT_TASKS}")
            return False

        try:
            # ==== 防重复启动：同任务已有存活进程，先停旧的再起新的 ====
            old_process = self.tasks.get(task_id)
            if old_process is not None:
                if old_process.is_alive():
                    logger.warning(
                        f"Task {task_id} 已有运行中进程 (PID: {old_process.pid})，先停止旧进程")
                    self.stop_task(task_id)
                    time.sleep(2)   # 给旧进程退出时间，释放 GPU/端口
                else:
                    self.tasks.pop(task_id, None)
            # ========================================================
            assigned_gpu_id = 0
            if self.gpu_load_balancer:
                assigned_gpu_id = self.gpu_load_balancer.select_gpu()
                logger.info(f"[GPU负载均衡] 为task_{task_id}分配GPU_ID={assigned_gpu_id}")
            else:
                logger.warning(f"[GPU负载均衡] 未初始化，使用默认GPU_0")

            process = mp.Process(
                target=run_deepstream,
                args=(uavSN, input_rtsp, output_rtmp, algorithm_list, filePath, assigned_gpu_id, task_id, llm_list, mode),
                name=f"inference-{task_id}"
            )
            process.start()

            self.tasks[task_id] = process

            self.save_task_status(task_id, "running", {
                "input_rtsp": input_rtsp,
                "output_rtmp": output_rtmp,
                "pid": process.pid,
                # "algorithm_list": algorithm_list,
                "assigned_gpu_id": assigned_gpu_id,
                "filePath": filePath,
                "uavSN": uavSN,
                "llm": llm_list or [],
                "mode": mode,
                "llm_permission": llm_permission,
            })

            logger.info(
                f"Task {task_id} started (PID: {process.pid}, GPU_ID: {assigned_gpu_id}, LLM: {llm_list}) with {len(algorithm_list)} algorithms")
            return True, ""

        except Exception as e:
            logger.error(f"Failed to start task {task_id}: {e}")
            self.save_task_status(task_id, "error", {"error": str(e)})
            return False, str(e)

    def stop_task(self, task_id):
        """停止整条任务：对托管进程 SIGTERM，超时后 SIGKILL，更新 Redis 为 ``stopped``。"""
        if task_id not in self.tasks:
            logger.warning(f"Task {task_id} not found")
            return False

        process = self.tasks[task_id]

        logger.info(f"self.tasks:{self.tasks}")

        if not process.is_alive():
            logger.warning(f"Task {task_id} is not running")
            del self.tasks[task_id]
            return False

        try:
            logger.info(f"Stopping task {task_id} (PID: {process.pid})")

            # 发送 SIGTERM
            process.terminate()

            # 等待最多 10 秒
            process.join(timeout=10)

            # 如果还在运行，强制杀死
            if process.is_alive():
                logger.warning(f"Task {task_id} did not terminate, killing...")
                process.kill()
                process.join()

            # 保存状态
            self.save_task_status(task_id, "stopped")

            del self.tasks[task_id]
            logger.info(f"Task {task_id} stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop task {task_id}: {e}")
            return False

    def stop_algorithms(self, channel: str, old_algo_list: list, old_llm: list) -> bool:
        """热卸载：向推理进程发布 ``stop_algorithm_list`` / ``stop_llm_list``，并更新本地快照。

        Args:
            channel: 与任务 ``task_id`` 一致的 Redis 频道。
            old_algo_list: 待停止算法项列表（含 ``algorithmNum``）。
            old_llm: 待停止的 LLM 配置项列表。

        Returns:
            bool: Redis 发布与状态写回成功为 True。
        """
        try:
            status_data = self.get_task_status(channel)
            if not status_data:
                logger.error(f"No task status found for {channel}")
                return False

            current_list = status_data.get("algorithm_list", [])

            current_ids = [a.get('algorithmNum') for a in current_list]

            old_ids = set(a.get('algorithmNum') for a in old_algo_list)

            new_algorithm_list = [
                algo for algo in current_list
                if algo.get('algorithmNum') not in old_ids
            ]

            message = {
                "stop_algorithm_list": old_algo_list,
                "stop_llm_list": old_llm
            }

            try:
                self.redis_client.publish(str(channel), json.dumps(message))

                status_data["algorithm_list"] = new_algorithm_list

                if "llm_list" in status_data:
                    old_llm_set = set(old_llm)
                    status_data["llm_list"] = [
                        llm for llm in status_data["llm_list"]
                        if llm not in old_llm_set
                    ]

                self.save_task_status(channel, "stopping", status_data)

                logger.info(f"Published stop message to subprocess: {message}")
                return True

            except Exception as e:
                logger.error(f"Failed to publish stop message: {e}")
                return False

        except Exception as e:
            logger.error(f"Error in stop_algorithms: {e}")
            return False

    def _handle_sigchld(self, signum, frame):
        """非阻塞回收子进程，同步移除 tasks 并将 Redis 标记为 exited。"""
        try:
            while True:
                pid, status = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
                # 找到对应的 task_id 并清理
                for task_id, proc in list[tuple[str, Process]](self.tasks.items()):
                    if proc.pid == pid:
                        logger.info(f"Child process for task {task_id} exited")
                        del self.tasks[task_id]
                        self.save_task_status(task_id, "exited")
                        break
        except ChildProcessError:
            pass

    def get_active_tasks(self):
        """枚举当前存活进程及其算法摘要，供 MQTT serve_status 周期性上报。

        Returns:
            list[dict]: 每项含 task_id、algorithms、pid、gpu_id。
        """
        active = []
        for task_id, process in list[tuple[str, Process]](self.tasks.items()):
            if process.is_alive():
                status_data = self.get_task_status(task_id)
                algorithms = []
                if status_data and "algorithm_list" in status_data:
                    for algo in status_data["algorithm_list"]:
                        algo_num = str(algo.get('algorithmNum'))
                        algorithms.append({
                            "algorithmNum": algo_num,
                            "char": f"算法{algo_num}",
                        })
                active.append({
                    "task_id": task_id,
                    "algorithms": algorithms,
                    "pid": process.pid,
                    "gpu_id": status_data.get("assigned_gpu_id") if status_data else None,
                })
            else:
                del self.tasks[task_id]
                self.save_task_status(task_id, "exited")
        return active

    def cleanup_tasks(self):
        """服务关停时依次 stop_task，释放所有托管进程。"""
        logger.info("Cleaning up all tasks...")
        for task_id in list(self.tasks.keys()):
            self.stop_task(task_id)


class MQTTService:
    """MQTT 客户端封装：握手门控、业务消息分发、周期性状态主动上报。

    入站主题（实现上与 Config 中配置保持一致，on_connect 内显式订阅）：
        - 控制类：ai/calculate/control/server
        - 事件/握手回调：ai/calculate/event/event_reply

    出站主题：
        - 控制应答：Config.MQTT_SEND_SERVER_REPLY
        - 握手与 serve_status：Config.MQTT_TOPIC_HANDSHAKE
    """

    def __init__(self, task_manager: TaskManager):
        self.task_manager = task_manager
        self.client = mqtt.Client(client_id=Config.MQTT_CLIENT_ID)
        self.model_dict = None
        self.algo_name_map = {}
        self.load_model_config()
        self.ParameterSynchronization = RedisAlgorithmManager(redis_config=redis_config)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        self.thread_pool = ThreadPoolExecutor(max_workers=5)

        self.handshake_state = False
        self.handshake_lock = threading.Lock()
        self.handshake_tid = str(uuid.uuid4())
        self.handshake_retry_count = 0
        self.handshake_thread = None
        self.handshake_timer = None
        self.should_stop_handshake = False
        self.public_key = "example_public_key_from_client"
        self.private_key = "example_private_key_from_b_side"

        if Config.MQTT_USERNAME and Config.MQTT_PASSWORD:
            self.client.username_pw_set(Config.MQTT_USERNAME, Config.MQTT_PASSWORD)

    def load_model_config(self):
        """热加载 common.config 中的算法清单，构建 model_dict 与算法展示名映射。"""
        import common.config
        reload(common.config)
        config_algorithms = common.config.algo_config
        self.model_dict = self.Convert_model_path(config_algorithms)
        # ===== 新增：构建算法名称映射 =====
        self.algo_name_map = {}
        for algo_dict in config_algorithms:
            self.algo_name_map[str(algo_dict['algorithmNum'])] = algo_dict.get("modelChar",
                                                                               f"算法{algo_dict['algorithmNum']}")

    def _generate_handshake_message(self, result_code: int = 3):
        """组装握手阶段 JSON 载荷。

        Args:
            result_code: 3 请求对端认证；0 认证通过并附带插件列表；1 认证失败。

        Returns:
            dict: 符合平台约定的握手消息体（含 definition / tid / data / result）。
        """
        if result_code == 3:
            message = {
                "tid": self.handshake_tid,
                "data": {},
                "timestamp": int(time.time() * 1000),  # 毫秒级时间戳
                "target": "",
                "definition": "handshake_auth_reply",
                "result": result_code
            }
        elif result_code == 0:
            algorithms = self.ParameterSynchronization._get_now_config()

            message = {
                "tid": self.handshake_tid,
                "data": {
                    "pluginList": algorithms
                },
                "timestamp": 16546546,
                "target": "",
                "definition": "handshake_process_reply",
                "result": 0
            }
        elif result_code == 1:
            message = {
                "tid": self.handshake_tid,
                "data": {},
                "timestamp": 16546546,
                "target": "",
                "definition": "handshake_process",
                "result": -1
            }
        return message

    def _verify_public_key(self, public_key: str) -> bool:
        """将客户端上报的 authKey 与部署侧 auth_token.json 中授权值做字符串恒等比对。"""
        try:
            # 读取 auth_token.json 中的授权公钥
            auth_token_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "common", "auth_token.json")
            try:
                with open(auth_token_path, 'r', encoding='utf-8') as f:
                    auth_data = json.load(f)
                authorized_key = auth_data.get("authKey", "")
            except FileNotFoundError:
                logger.error(f"[握手认证] 授权文件不存在：{auth_token_path}")
                return False
            except json.JSONDecodeError as e:
                logger.error(f"[握手认证] 授权文件格式错误：{e}")
                return False

            if not authorized_key:
                logger.error(f"[握手认证] 授权文件中没有有效的 authKey")
                return False

            # 比对客户端发送的公钥与授权公钥
            if public_key == authorized_key:
                logger.info(f"[握手认证] 公钥验证成功")
                return True
            else:
                logger.warning(f"[握手认证] 公钥不匹配，验证失败")
                logger.debug(f"[握手认证] 客户端公钥：{public_key[:20]}...")
                logger.debug(f"[握手认证] 授权公钥：{authorized_key[:20]}...")
                return False

        except Exception as e:
            logger.error(f"[握手认证] 公钥验证异常：{e}")
            return False

    def _send_handshake_message(self, result_code: int = 3):
        """向 MQTT_TOPIC_HANDSHAKE 发布握手载荷（QoS 1）。"""
        try:
            message = self._generate_handshake_message(result_code)
            payload = json.dumps(message, ensure_ascii=False)

            self.client.publish(
                Config.MQTT_TOPIC_HANDSHAKE,
                payload,
                qos=1
            )
            logger.info(f"[握手认证] 发送握手消息到主题: {Config.MQTT_TOPIC_HANDSHAKE}, "
                        f"tid: {self.handshake_tid}, result: {result_code}")
        except Exception as e:
            logger.error(f"[握手认证] 发送握手消息失败: {e}")

    def _handshake_timeout_handler(self):
        """单次握手等待超时：复位 handshake_state 并进入重试调度。"""
        logger.warning(f"[握手认证] 握手超时 (等待时间: {Config.HANDSHAKE_TIMEOUT}s), "
                       f"回归握手重试模式")
        with self.handshake_lock:
            self.handshake_state = False
        self._start_handshake_retry()

    def _start_handshake_retry(self):
        """在 HANDSHAKE_RETRY_INTERVAL 后再次发送 result_code=3 并重启超时定时器。"""
        if self.should_stop_handshake:
            logger.info("[握手认证] 握手已停止")
            return

        if self.handshake_retry_count >= Config.HANDSHAKE_MAX_RETRIES:
            logger.error(f"[握手认证] 握手重试次数已达上限 ({Config.HANDSHAKE_MAX_RETRIES}次), "
                         "请检查网络连接和A端配置")
            return

        self.handshake_retry_count += 1
        logger.info(f"[握手认证] 安排握手重试 (第 {self.handshake_retry_count}/{Config.HANDSHAKE_MAX_RETRIES} 次), "
                    f"将在 {Config.HANDSHAKE_RETRY_INTERVAL}s 后重试")

        # 生成新的事务ID
        self.handshake_tid = str(uuid.uuid4())

        # 等待后发送握手消息
        def retry_after_delay():
            time.sleep(Config.HANDSHAKE_RETRY_INTERVAL)
            if not self.should_stop_handshake:
                self._send_handshake_message(result_code=3)
                self._set_handshake_timeout()

        retry_thread = threading.Thread(target=retry_after_delay, daemon=True)
        retry_thread.start()

    def _set_handshake_timeout(self):
        """启动 HANDSHAKE_TIMEOUT 秒后触发的超时 Timer（守护线程）。"""
        if self.handshake_timer:
            self.handshake_timer.cancel()

        self.handshake_timer = threading.Timer(
            Config.HANDSHAKE_TIMEOUT,
            self._handshake_timeout_handler
        )
        self.handshake_timer.daemon = True
        self.handshake_timer.start()
        logger.debug(f"[握手认证] 握手超时计时器已设置 ({Config.HANDSHAKE_TIMEOUT}s)")

    def _complete_handshake(self, success: bool):
        """取消超时定时器；成功则允许业务消息，失败则触发重试链。"""
        if self.handshake_timer:
            self.handshake_timer.cancel()
            self.handshake_timer = None

        with self.handshake_lock:
            self.handshake_state = success

        if success:
            logger.info(f"[握手认证] ✓ 握手成功完成 (tid: {self.handshake_tid}), "
                        "系统已准备就绪")
            self.handshake_retry_count = 0  # 重置重试计数
        else:
            logger.warning(f"[握手认证] ✗ 握手认证失败 (tid: {self.handshake_tid}), "
                           "回归握手重试模式")
            self._start_handshake_retry()

    def start_handshake(self):
        """MQTT 连接建立后异步发送首次握手（result_code=3）并启动超时监视。"""
        logger.info("[握手认证] 启动握手流程...")

        def handshake_proc():
            try:
                # 发送初始握手消息
                self._send_handshake_message(result_code=3)
                # 设置超时
                self._set_handshake_timeout()
            except Exception as e:
                logger.error(f"[握手认证] 握手启动失败: {e}")
                self._start_handshake_retry()

        self.handshake_thread = self.thread_pool.submit(handshake_proc)

    def on_connect(self, client, userdata, flags, rc):
        """Paho 回调：订阅控制与事件应答主题，并启动握手序列。"""
        if rc == 0:
            logger.info(f"Connected to MQTT broker: {Config.MQTT_BROKER}:{Config.MQTT_PORT}")
            control_topic = "ai/calculate/control/server"
            event_reply_topic = "ai/calculate/event/event_reply"
            client.subscribe([
                (control_topic, 1),
                (event_reply_topic, 1)
            ])
            logger.info(f"Subscribed to control topic: {control_topic}")
            logger.info(f"Subscribed to event reply topic: {event_reply_topic}")
            # 图片模式：额外订阅平台手动拍照通知主题
            if Config.IMAGE_MODE:
                client.subscribe([(Config.MQTT_TOPIC_PHOTO_NOTIFY, 1)])
                logger.info(f"[图片模式] 订阅拍照通知主题: {Config.MQTT_TOPIC_PHOTO_NOTIFY}")
            self.start_handshake()
        else:
            logger.error(f"Failed to connect to MQTT broker, return code: {rc}")

    def on_disconnect(self, client, userdata, rc):
        """Paho 回调：非零 rc 时记录异常断开（重连由客户端库与 loop 策略决定）。"""
        if rc != 0:
            logger.warning(f"Unexpected MQTT disconnection, return code: {rc}")
            logger.info("Attempting to reconnect...")

    def on_message(self, client, userdata, msg):
        """Paho 回调：将解析与业务逻辑投递至线程池，避免阻塞网络循环。"""
        self.thread_pool.submit(self.msg_process, msg)

    def Convert_model_path(self, model_config):
        """将 algo_config 列表转为 algorithmNum(str) -> 模型路径与类型字典。"""
        result = {
            str(d.get("algorithmNum")): {"modelPath": d.get("modelPath", ""), "withTrack": d.get("withTrack", False),
                                         "modelType": d.get("modelType", False)} for d in model_config}
        return result

    def _handle_photo_notify(self, data: dict):
        """图片模式：处理平台手动拍照通知，向推理子进程下发图片检测任务。

        平台消息字段（兼容多种命名，按优先级取值）:
            - imageUrl / image_url / objectKey / object_key / url
            - bucket (可选，缺省 alarm-images)
            - taskId / task_id / device_sn / target / droneSn

        下发方式: 向 Redis 频道 ``photo_task:{task_id}`` 发布 JSON，
        由 ``common.image_trt_inference._on_photo_message`` 消费并执行推理。
        """
        try:
            logger.info(f"[图片模式] 拍照通知原始载荷: {json.dumps(data, ensure_ascii=False)[:400]}")
            info = data.get("data", {}) if isinstance(data.get("data"), dict) else {}
            # 合并外层与 data 内层字段，兼容两种消息结构
            merged = {**info, **{k: v for k, v in data.items() if k != "data"}}

            image_url = merged.get("imageUrl") or merged.get("image_url") or merged.get("url")
            object_key = merged.get("objectKey") or merged.get("object_key")
            bucket = merged.get("bucket") or "alarm-images"

            task_id = merged.get("target")

            if not task_id:
                logger.warning(f"[图片模式] 拍照通知缺少 taskId，跳过: {data}")
                return

            if not image_url and not object_key:
                logger.warning(f"[图片模式] 拍照通知缺少 imageUrl/objectKey，跳过: {data}")
                return

            # ---- 拍照通知去重：同一 object_key 在窗口内只下发一次 ----
            if object_key:
                now_ts = time.time()
                last_ts = getattr(self, "_photo_seen", {}).get(object_key, 0)
                if now_ts - last_ts < 600:
                    logger.info(f"[图片模式] 重复拍照通知，已去重: {object_key}")
                    return
                if not hasattr(self, "_photo_seen"):
                    self._photo_seen = {}
                self._photo_seen[object_key] = now_ts
                # 防止无限增长：超过 2000 条时丢弃最旧的一半
                if len(self._photo_seen) > 2000:
                    kept = dict(sorted(self._photo_seen.items(),
                                       key=lambda kv: kv[1])[-1000:])
                    self._photo_seen = kept
            # ------------------------------------------------------

            photo_job = {
                "type": "photo",
                "image_url": image_url,
                "object_key": object_key,
                "bucket": bucket,
                "task_id": str(task_id),
                "timestamp": int(time.time() * 1000),
            }

            logger.info(
                f"[图片模式] 收到拍照通知: image_url={image_url}, object_key={object_key}, task_id={task_id}, bucket={bucket}")
            _trace = str(object_key).rsplit("/", 1)[-1]
            logger.info(f"[TRACE][{_trace}] stage=1_photo_notify_recv ts={int(time.time()*1000)}")
            channel = f"photo_task:{task_id}"
            if self.task_manager.redis_client:
                self.task_manager.redis_client.publish(channel, json.dumps(photo_job, ensure_ascii=False))
                logger.info(f"[图片模式] 已下发拍照任务到 {channel}: "
                            f"key={object_key}, url={image_url}")
                logger.info(f"[TRACE][{_trace}] stage=2_photo_task_publish channel={channel}")
            else:
                logger.error("[图片模式] Redis 不可用，无法下发拍照任务")

        except Exception as e:
            logger.error(f"[图片模式] 处理拍照通知失败: {e}", exc_info=True)

    def msg_process(self, msg):
        """解析 MQTT 负载并分发业务处理器。

         处理顺序：
             1. topic 为 event_reply 且 definition 为 handshake_auth_reply：完成握手门控。
             2. 若 handshake_state 为 False：丢弃其余消息。
             3. 按 definition 路由：
                - serve_air_route_reply：航线索引写 Redis，应答 serve_air_route。
                - serve_switch_reply：部分停算 + 增量启算，应答 serve_switch。
                - serve_start_reply：整任务启动，应答 serve_start。
                - serve_stop_reply：整任务停止，应答 serve_stop。
         """
        try:
            payload = msg.payload.decode('utf-8')

            data = json.loads(payload)
            logger.info(f"Received message on topic {msg.topic} : {payload}...")

            if msg.topic == "ai/calculate/event/event_reply" and data.get("definition") == "handshake_auth_reply":
                logger.info(f"[握手认证] 收到A端握手认证请求 (tid: {data.get('tid')})")

                client_public_key = data.get("data", {}).get("authKey", "")
                client_tid = data.get("tid", "")

                if self._verify_public_key(client_public_key):
                    logger.info(f"[握手认证] ✓ 公钥验证成功 (tid: {client_tid})")
                    self.handshake_tid = client_tid
                    self._send_handshake_message(result_code=0)
                    self._complete_handshake(success=True)
                else:
                    logger.error(f"[握手认证] ✗ 公钥验证失败 (tid: {client_tid}), 用户无授权")
                    self.handshake_tid = client_tid
                    self._send_handshake_message(result_code=1)
                    self._complete_handshake(success=False)
                return

            if not self.handshake_state:
                logger.warning(f"[握手认证] ✗ 握手未完成，拒绝接收消息: {data.get('definition', 'unknown')}")
                return

            # 图片模式：手动拍照通知 → 下发图片任务到推理子进程
            if Config.IMAGE_MODE and msg.topic == Config.MQTT_TOPIC_PHOTO_NOTIFY:
                self._handle_photo_notify(data)
                return

            message_type = data.get("definition")
            info = data.get("data", {})
            if not message_type:
                logger.error("Missing 'definition' in message")
                return

            logger.info(f"Received message on topic {msg.topic} : {payload[:30]}...")

            # --- serve_air_route_reply：航线索引缓存 ---
            if message_type == "serve_air_route_reply":
                device_sn = data.get("target")
                if not device_sn:
                    logger.error("Missing target (device SN) in serve_air_route_reply message")
                    return

                new_air_route_index_id = info.get("newAirRouteIndexId")
                if not new_air_route_index_id:
                    logger.error("Missing newAirRouteIndexId in serve_air_route_reply message")
                    response = {
                        "tid": data.get("tid"),
                        "data": {},
                        "timestamp": int(time.time() * 1000),
                        "target": device_sn,
                        "definition": "serve_air_route",
                        "result": 1004
                    }
                    self.client.publish(
                        Config.MQTT_SEND_SERVER_REPLY,
                        json.dumps(response, ensure_ascii=False),
                        qos=1
                    )
                    return

                result_code = 0
                try:
                    if self.task_manager.redis_client:
                        key = f"{Config.REDIS_KEY_ROUTE_PREFIX}{device_sn}"
                        route_info = {
                            "newAirRouteIndexId": new_air_route_index_id,
                            "timestamp": int(time.time() * 1000)
                        }
                        self.task_manager.redis_client.setex(key, 600, json.dumps(route_info))
                        logger.info(f"Updated air route index for device {device_sn}: {new_air_route_index_id}")
                    else:
                        logger.error("Redis client not available")
                        result_code = 1004
                except Exception as e:
                    logger.error(f"Failed to save air route index to Redis: {e}")
                    result_code = 1004

                response = {
                    "tid": data.get("tid"),
                    "data": {},
                    "timestamp": int(time.time() * 1000),
                    "target": device_sn,
                    "definition": "serve_air_route",
                    "result": result_code
                }
                self.client.publish(
                    Config.MQTT_SEND_SERVER_REPLY,
                    json.dumps(response, ensure_ascii=False),
                    qos=1
                )

            # --- serve_switch_reply：算法/LLM 热切换 ---
            elif message_type == "serve_switch_reply":
                logger.info(f"========== 收到 serve_switch_reply 消息 ==========")
                logger.info(f"完整消息: {payload}")
                device_sn = data.get("target")
                if not device_sn:
                    logger.error("Missing target (device SN) in serve_switch_reply message")
                    return

                old_algo_list = info.get("oldAlgorithmList", [])
                new_algo_list = info.get("newAlgorithmList", [])
                logger.info(f"[切换] 设备 {device_sn} 请求切换，新算法列表: {new_algo_list}")
                logger.info(f"[切换] 设备 {device_sn} 请求关闭，旧算法列表: {old_algo_list}")

                new_llm = info.get("newLLM", [])
                old_llm = info.get("oldLLM", [])
                llm_permission = info.get("llmPermission", None)
                if llm_permission is None:
                    # 平台本次未下发权限字段：沿用该任务最近一次的权限（start 时记录的）
                    _status = self.task_manager.get_task_status(device_sn) or {}
                    llm_permission = _status.get("llm_permission", None)
                    logger.info(f"[切换] llmPermission 缺省，沿用任务快照值: {llm_permission}")
                logger.info(f"[切换] 设备 {device_sn} 大模型权限 llmPermission: {llm_permission}")
                logger.info(f"[切换] 设备 {device_sn} 请求切换，新大模型列表: {new_llm}")
                success = False
                try:
                    if old_algo_list or old_llm:
                        success = self.task_manager.stop_algorithms(
                            channel=device_sn,
                            old_algo_list=old_algo_list,
                            old_llm=old_llm,
                        )
                    logger.info(f"stop_model 返回: {success}")
                except Exception as e:
                    logger.error(f"stop_model 异常: {e}", exc_info=True)
                    success = False

                constructed_list = []
                if len(new_algo_list) > 0:
                    for algo in new_algo_list:
                        algorithmNum = algo.get("algorithmNum")
                        if not algorithmNum:
                            logger.error(f"算法项缺少 algorithmNum: {algo}")
                            continue
                        logger.info(f"处理算法 {algorithmNum}")

                        model_params = algo.get("modelParams", [])
                        second_review = 1 if str(algorithmNum) == "93" else algo.get('second_review', 0)

                        constructed_list.append({
                            "algorithmNum": algorithmNum,
                            "paramers": model_params,
                            "second_review": second_review,
                        })

                try:
                    success = self.task_manager.switch_model(
                        channel=device_sn,
                        new_algorithm_list=constructed_list,
                        new_llm_list=new_llm,
                        llm_permission=llm_permission,
                    )
                    logger.info(f"switch_model 返回: {success}")
                except Exception as e:
                    logger.error(f"switch_model 异常: {e}", exc_info=True)
                    success = False

                response = {
                    "tid": data.get("tid"),
                    "data": {},
                    "timestamp": int(time.time() * 1000),
                    "target": device_sn,
                    "definition": "serve_switch",
                    "result": 0 if success else -1
                }

                try:
                    self.client.publish(
                        Config.MQTT_SEND_SERVER_REPLY,
                        json.dumps(response, ensure_ascii=False),
                        qos=1
                    )
                    logger.info(f"已发送切换响应到 {Config.MQTT_SEND_SERVER_REPLY}: {response}")
                except Exception as e:
                    logger.error(f"发布切换响应失败: {e}")

            # --- serve_start_reply：整任务启动 ---
            elif message_type == "serve_start_reply":

                algo_data = info.get("algorithmList", [])
                logger.info(f"algo_data 原始内容: {algo_data}")
                task_id = data.get("tid")
                input_rtsp = info.get("inputStreamPath")
                output_rtmp = info.get("outputStreamPath")
                filePath = info.get("filePath")
                # airport type: 3=image mode, 2=video mode
                airport_type = info.get("airportType", info.get("dockType"))
                if str(airport_type) == "3":
                    task_mode = "image"
                elif str(airport_type) == "2":
                    task_mode = "video"
                else:
                    task_mode = "image" if Config.IMAGE_MODE else "video"
                    logger.warning(f"airportType={airport_type} unrecognized, fallback mode={task_mode}")
                if task_mode == "video":
                    output_rtmp = output_rtmp or "no-push://inference-only"
                logger.info(f"airportType={airport_type} -> mode={task_mode}")
                airRouteIndexId = info.get("airRouteIndexId")
                device_sn = data.get("target")

                # 图片模式不依赖拉流/推流地址，仅校验 filePath（bucket 派生自 filePath）
                if task_mode == "image":
                    if not filePath:
                        logger.error("[图片模式] 缺少必需顶层字段: filePath，终止处理")
                        return
                    # 流地址缺省时填占位值，保持下游参数完整
                    input_rtsp = input_rtsp or "image://manual-photo"
                    output_rtmp = output_rtmp or "image://manual-photo"
                else:
                    if not input_rtsp or not filePath:
                        logger.error("缺少必需顶层字段: inputStreamPath, outputStreamPath 或 filePath，终止处理")
                        return
                if not device_sn:
                    logger.error("  缺少 target (设备SN)，终止处理")
                    return
                logger.info(f"设备SN: {device_sn}")

                algorithms_info = []
                for idx, algo in enumerate(algo_data):
                    logger.info(f"---- 处理第 {idx + 1} 个算法 ----")
                    algorithmNum = algo.get("algorithmNum")
                    if not algorithmNum:
                        logger.error(f"  第 {idx + 1} 个算法缺少 algorithmNum，终止处理")
                        return
                    logger.info(f"算法编号 algorithmNum: {algorithmNum}")

                    model_params = algo.get("modelParams", [])
                    second_review = 1 if str(algorithmNum) == "93" else algo.get('second_review', 0)

                    algorithms_info.append({
                        "algorithmNum": algorithmNum,
                        "paramers": model_params,
                        "second_review": second_review,
                    })
                logger.info(f"===== 所有算法信息收集完成，共 {len(algorithms_info)} 个 =====")

                llm_list = info.get("algorithmStart", [])
                llm_permission = info.get("llmPermission", None)
                logger.info(f"[大模型权限] llmPermission={llm_permission}")
                if llm_permission is not None and str(llm_permission).strip().lower() in ("0", "false", "no", "off"):
                    logger.info(f"[大模型权限] llmPermission=0 无权限，忽略大模型配置 {len(llm_list)} 条")
                    llm_list = []
                logger.info(f"LLM 配置：{llm_list}")

                success, msg = self.task_manager.start_task(
                    task_id=device_sn,
                    input_rtsp=input_rtsp,
                    output_rtmp=output_rtmp,
                    algorithm_list=algorithms_info,
                    filePath=filePath,
                    uavSN=device_sn,
                    llm_list=llm_list,
                    mode=task_mode,
                    llm_permission=llm_permission,
                )
                logger.info(f" 启动状态 success = {success, msg}")

                response = {
                    "tid": data.get("tid"),
                    "data": {} if success else msg,
                    "timestamp": int(time.time() * 1000),
                    "target": device_sn,
                    "definition": "serve_start",
                    "result": 0 if success else -1
                }
                logger.info(f"构造的响应消息: {response}")

                try:
                    self.client.publish(
                        Config.MQTT_SEND_SERVER_REPLY,
                        json.dumps(response, ensure_ascii=False),
                        qos=1
                    )
                    logger.info(f"已发布响应到主题 {Config.MQTT_SEND_SERVER_REPLY}")
                except Exception as e:
                    logger.error(f"  发布响应时发生异常: {e}", exc_info=True)

                logger.info("========== serve_start_reply 分支处理完毕 ==========")

            # --- serve_stop_reply：整任务停止 ---
            elif message_type == "serve_stop_reply":
                device_sn = data.get("target")
                if not device_sn:
                    logger.error("Missing target (device SN) in stop message")
                    return

                success = self.task_manager.stop_task(device_sn)

                response = {
                    "tid": data.get("tid"),
                    "data": {},
                    "timestamp": int(time.time() * 1000),
                    "target": device_sn,
                    "definition": "serve_stop",
                    "result": 0 if success else -1
                }

                try:
                    self.client.publish(
                        Config.MQTT_SEND_SERVER_REPLY,
                        json.dumps(response, ensure_ascii=False),
                        qos=1
                    )
                    logger.info(f"Sent stop response to {Config.MQTT_SEND_SERVER_REPLY}: {response}")
                except Exception as e:
                    logger.error(f"Failed to publish stop response: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON message: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)

    def publish_status_periodically(self):
        """主动上报运行中算法清单（definition=serve_status，发往 MQTT_TOPIC_HANDSHAKE）。"""
        try:
            active_tasks = self.task_manager.get_active_tasks()
            if not active_tasks:
                return

            data = {}
            for task in active_tasks:
                device_sn = task["task_id"]
                algorithms = task["algorithms"]
                algo_list = []
                for algo in algorithms:
                    algo_num = algo["algorithmNum"]
                    name = self.algo_name_map.get(str(algo_num), f"算法{algo_num}")
                    algo_list.append({
                        "algorithmNum": algo_num,
                        "char": name
                    })
                data[device_sn] = algo_list

            message = {
                "tid": str(uuid.uuid4()),
                "data": data,
                "timestamp": int(time.time() * 1000),
                "target": "",
                "definition": "serve_status"
            }

            self.client.publish(
                Config.MQTT_TOPIC_HANDSHAKE,
                json.dumps(message, ensure_ascii=False),
                qos=1
            )
            logger.debug(f"Published periodic status: {len(data)} devices")
        except Exception as e:
            logger.error(f"Failed to publish periodic status: {e}", exc_info=True)

    def connect(self):
        """阻塞式 connect + loop_start，失败向上抛出。"""
        try:
            self.client.connect(Config.MQTT_BROKER, Config.MQTT_PORT, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            raise

    def disconnect(self):
        """停止握手线程与 Timer，关闭 MQTT loop 并断开连接。"""
        self.should_stop_handshake = True
        if self.handshake_timer:
            self.handshake_timer.cancel()
        logger.info("[握手认证] 握手流程已停止")

        self.client.loop_stop()
        self.client.disconnect()
        logger.info("Disconnected from MQTT broker")


class InferenceService:
    """进程级编排：组装 TaskManager 与 MQTTService，驱动心跳与周期状态上报。"""

    def __init__(self):
        self.task_manager = TaskManager()
        self.mqtt_service = MQTTService(self.task_manager)
        self.running = False
        self.last_status_time = 0

    def start(self):
        """连接 MQTT、启动 Watchdog、注册信号量，并进入心跳与状态上报主循环。"""
        logger.info("=" * 60)
        logger.info("Starting Inference Service")
        logger.info("=" * 60)
        logger.info(f"MQTT Broker: {Config.MQTT_BROKER}:{Config.MQTT_PORT}")
        logger.info(f"Subscribe Topic: {Config.MQTT_TOPIC_ALGORITHM_CONTROL}")
        logger.info(f"Status Topic: {Config.MQTT_TOPIC_DATA_SYNC}")
        logger.info(f"Redis: {Config.REDIS_HOST}:{Config.REDIS_PORT}")
        logger.info(f"DeepStream Script: {Config.DEEPSTREAM_SCRIPT}")
        logger.info(f"Max Concurrent Tasks: {Config.MAX_CONCURRENT_TASKS}")
        if Config.IMAGE_MODE:
            logger.info(
                f"[图片模式] 已启用: 推理子进程 = common.image_trt_inference, 拍照主题 = {Config.MQTT_TOPIC_PHOTO_NOTIFY}")
        logger.info("=" * 60)

        try:
            try:
                self.mqtt_service.connect()
            except Exception as conn_e:
                # MQTT / Redis 外部依赖不可用时优雅降级：保持进程存活，便于本地试跑
                logger.warning(
                    f"[降级] MQTT/Redis 连接失败，进入本地降级模式（握手循环不阻塞）：{conn_e}"
                )
                logger.warning("[降级] 如需完整功能，请配置可用的 MQTT/Redis 并重启服务。")
                # 置空 mqtt_client，避免 publish_* 访问 None 崩溃
                try:
                    self.mqtt_service.mqtt_client = None
                except Exception:
                    pass

            self.running = True
            self.last_status_time = time.time()
            watchdog = None
            try:
                watchdog = Watchdog(
                    heartbeat_file=HEARTBEAT_FILE,
                    enable_nvml=False,  # Windows 本地试跑默认关闭 NVML，无 CUDA/驱动时不崩溃
                    process_name="uav-algorithm-pipeline"
                )
                watchdog.start()
            except Exception as wd_e:
                logger.warning(f"[降级] Watchdog 启动失败，已跳过：{wd_e}")

            try:
                signal.signal(signal.SIGINT, self._signal_handler)
                signal.signal(signal.SIGTERM, self._signal_handler)
            except Exception:
                # Windows 某些非主场景下 signal 不可注册，静默跳过
                pass

            logger.info("Service started, waiting for messages...")

            while self.running:
                try:
                    touch_heartbeat()
                except Exception:
                    pass
                now = time.time()
                if now - self.last_status_time >= Config.STATUS_REPORT_INTERVAL:
                    try:
                        self.mqtt_service.publish_status_periodically()
                    except Exception as pub_e:
                        # 降级模式下没有连接，静默忽略发布
                        if self.mqtt_service.mqtt_client is not None:
                            logger.debug(f"状态发布失败：{pub_e}")
                    self.last_status_time = now
                time.sleep(1)

                active_tasks = self.task_manager.get_active_tasks()
                if active_tasks:
                    logger.info(f"Active tasks: {len(active_tasks)}")

        except KeyboardInterrupt:
            logger.info("Service interrupted by user")
        except Exception as e:
            logger.error(f"Service error: {e}", exc_info=True)
        finally:
            self.stop()

    def stop(self):
        """停止主循环标志、清理子任务并断开 MQTT。"""
        logger.info("Stopping service...")
        self.running = False

        self.task_manager.cleanup_tasks()
        self.mqtt_service.disconnect()

        logger.info("Service stopped")

    def _signal_handler(self, signum, frame):
        """SIGINT/SIGTERM：置 running=False 以退出主循环。"""
        logger.info(f"Received signal {signum}")
        self.running = False


def main():
    """CLI 入口：强制 multiprocessing spawn，实例化 InferenceService 并阻塞运行。"""
    try:
        mp.set_start_method('spawn', force=True)
        logger.info("Multiprocessing start method set to: spawn")
    except RuntimeError:
        logger.debug("Start method already set")
    service = InferenceService()
    service.start()


if __name__ == "__main__":
    main()