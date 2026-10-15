from importlib import reload
import yaml
import redis
import json
import os
import time
import logging
import sys
import threading
from typing import List, Dict, Optional, Union
from redis.exceptions import RedisError
from datetime import datetime
import common.config

# ------------------------------
# 1. 基础配置
# ------------------------------
DEFAULT_REDIS_CONFIG = {
    "REDIS_HOST": "192.168.1.15",
    "REDIS_PORT": 6379,
    "REDIS_DB": 0,
    "REDIS_PASSWORD": None
}

ALGORITHM_DICT_KEY = "ai_info"
JOB_STATUS_KEY_PREFIX = "ai_job_status"
UAV_LOCATION_KEY_PREFIX = "ai_location"
logging_config = {}
DEFAULT_CONFIG_JSON_PATH = "/tensorrt/uav-algorithm-pipeline/config.json"
CONFIG_CHECK_INTERVAL = 10  # config.json 变更检测间隔（秒）

LOG_LEVEL = logging_config.get('level', 'INFO')
LOG_FILE = logging_config.get('file', 'inference_service.log')
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('inference_service.log')
    ]
)
logger = logging.getLogger(__name__)


# ------------------------------
# 2. Redis连接初始化
# ------------------------------
def get_redis_instance(config) -> redis.Redis:
    if not hasattr(get_redis_instance, "instance"):
        redis_conn = redis.Redis(
            host=config["host"],
            port=config["port"],
            db=config["database"],
            password=config["password"],
            decode_responses=True,
            socket_timeout=5
        )
        try:
            redis_conn.ping()
            print(f"[Redis] 连接成功（地址：{config['host']}:{config['port']}）")

        except Exception as e:
            raise ConnectionError(f"[Redis] 连接失败：{str(e)}") from e
    return redis_conn


# ------------------------------
# 3. 核心工具类
# ------------------------------
class RedisAlgorithmManager:
    def __init__(
            self,
            config_json_path: str = DEFAULT_CONFIG_JSON_PATH,
            check_interval: int = CONFIG_CHECK_INTERVAL,
            redis_config: str = None
    ):
        self.redis = get_redis_instance(redis_config)
        self.config_json_path = config_json_path
        self.check_interval = check_interval  # 变更检测间隔
        self.last_config_mtime = 0  # 记录config.json上次修改时间
        self.config_check_thread = None  # 后台检测线程
        self.running = False  # 线程运行标记
        self.algo_config = None
        self.start()

    def start(self):
        """启动配置变更检测线程"""
        if not self.running:
            self.running = True
            # self.config_check_thread = threading.Thread(target=self._config_check_loop)
            # self.config_check_thread.daemon = True
            # self.config_check_thread.start()    
            # logger.info("[Redis] 配置变更检测线程已启动")   

    def stop(self):
        """停止配置变更检测线程"""
        self.running = False
        if self.config_check_thread:
            self.config_check_thread.join()
            logger.info("[Redis] 配置变更检测线程已停止")

    def sync_status_clear(self):
        """清空同步状态 0  —————>  1 """
        # try:
        pass

    def _get_now_config(self):
        """获取当前的config.py配置"""
        try:
            if self.algo_config:
                return self.algo_config
            else:
                reload(common.config)
                self.algo_config = common.config.algo_config
                if self.algo_config:
                    # self.redis.set(ALGORITHM_DICT_KEY, json.dumps(self.algo_config))
                    return self.algo_config
                else:
                    logger.warning("[Redis] 未找到有效的算法配置")
                    return None
        except Exception as e:
            logger.error(f"[Redis] 获取config.py失败：{str(e)}")
            return None

    def _config_check_loop(self):
        """后台检测config.json是否有变更"""
        while self.running:
            try:
                reload(common.config)
                self.algo_config = common.config.algo_config
                if self.algo_config:
                    # self.redis.set(ALGORITHM_DICT_KEY, json.dumps(self.algo_config))
                    time.sleep(self.check_interval)
                else:
                    logger.warning("[Redis] 未找到有效的算法配置")
                    time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"[Redis] 配置变更检测失败：{str(e)}")
                time.sleep(self.check_interval)