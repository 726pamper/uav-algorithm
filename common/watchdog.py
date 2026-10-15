import os
import time
import threading
import logging
import signal

logger = logging.getLogger("watchdog")
logger.setLevel(logging.INFO)

HEARTBEAT_TIMEOUT = 30  # 秒
CHECK_INTERVAL = 5
HEARTBEAT_FILE = "/tmp/uav_pipeline_heartbeat"

class Watchdog:
    def __init__(
        self,
        heartbeat_file: str,
        enable_nvml: bool = False,
        process_name: str = "service"
    ):
        self.heartbeat_file = heartbeat_file
        self.enable_nvml = enable_nvml
        self.process_name = process_name
        self._stop = threading.Event()

    # ======================
    # NVML 检查（仅主服务）
    # ======================
    def check_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            _ = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()
        except Exception as e:
            logger.critical("[%s] NVML check failed: %s", self.process_name, e)
            self.fatal_exit()

    # ======================
    # 心跳检查（通用）
    # ======================
    def check_heartbeat(self):
        if not os.path.exists(self.heartbeat_file):
            logger.critical("[%s] Heartbeat file missing", self.process_name)
            self.fatal_exit()

        last_mtime = os.path.getmtime(self.heartbeat_file)
        if time.time() - last_mtime > HEARTBEAT_TIMEOUT:
            logger.critical("[%s] Heartbeat timeout", self.process_name)
            self.fatal_exit()

    # ======================
    # 判死：直接杀进程
    # ======================
    def fatal_exit(self):
        logger.critical("[%s] Watchdog fatal, exiting process", self.process_name)
        os._exit(1)

    # ======================
    # watchdog 主循环
    # ======================
    def run(self):
        logger.info("[%s] Watchdog started", self.process_name)

        while not self._stop.is_set():
            if self.enable_nvml:
                self.check_nvml()

            self.check_heartbeat()
            time.sleep(CHECK_INTERVAL)

    def start(self):
        t = threading.Thread(target=self.run, daemon=True)
        t.start()

    def stop(self):
        self._stop.set()

