import redis
import yaml
import threading
import logging
from typing import Any, Callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class RedisHelper:
    """
        要求版本大于等于 pip install redis>=5.0.1
    """
    def __init__(self, config_path: str = "config-dev.yaml"):
        self.config = self._load_config(config_path)
        self.pool = redis.ConnectionPool(
            host=self.config["host"],
            port=self.config["port"],
            password=self.config.get("password") or None,
            db=self.config.get("database", 0),
            decode_responses=True  # 自动转为 str
        )
        self.client = redis.Redis(connection_pool=self.pool)
        logging.info("RedisHelper initialized: %s:%s (db=%s)",
                     self.config["host"], self.config["port"], self.config["database"])

    def _load_config(self, path: str) -> dict:
        """加载 YAML 配置文件"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if "redis" not in data:
            raise ValueError("配置文件中缺少 'redis' 节点")
        return data["redis"]

    # -------------------- 基本操作 --------------------
    def set(self, key: str, value: Any, ex: int = None):
        """设置键值，可选过期时间(秒)"""
        return self.client.set(key, value, ex=ex)

    def get(self, key: str):
        """获取键值"""
        return self.client.get(key)

    def delete(self, key: str):
        """删除键"""
        return self.client.delete(key)

    def exists(self, key: str) -> bool:
        """判断 key 是否存在"""
        return self.client.exists(key) > 0

    def incr(self, key: str, amount: int = 1):
        return self.client.incr(key, amount)

    def decr(self, key: str, amount: int = 1):
        return self.client.decr(key, amount)

    # -------------------- 哈希操作 --------------------
    def hset(self, name: str, key: str, value: Any):
        return self.client.hset(name, key, value)

    def hget(self, name: str, key: str):
        return self.client.hget(name, key)

    def hgetall(self, name: str):
        return self.client.hgetall(name)

    # -------------------- 列表操作 --------------------
    def lpush(self, name: str, *values):
        return self.client.lpush(name, *values)

    def rpush(self, name: str, *values):
        return self.client.rpush(name, *values)

    def lpop(self, name: str):
        return self.client.lpop(name)

    def rpop(self, name: str):
        return self.client.rpop(name)

    def lrange(self, name: str, start: int = 0, end: int = -1):
        return self.client.lrange(name, start, end)

    # -------------------- 集合操作 --------------------
    def sadd(self, name: str, *values):
        return self.client.sadd(name, *values)

    def smembers(self, name: str):
        return self.client.smembers(name)

    # -------------------- 发布/订阅 --------------------
    def publish(self, channel: str, message: str):
        """发布消息"""
        logging.info("Publish: [%s] %s", channel, "message")
        return self.client.publish(channel, message)

    def subscribe(self, channel: str, callback: Callable[[str], None]):
        """
        订阅消息并在新线程中处理
        :param channel: 频道名
        :param callback: 消息处理回调函数 callback(message)
        """
        pubsub = self.client.pubsub()
        pubsub.subscribe(channel)
        logging.info("Subscribed to channel: %s", channel)

        def listen():
            for msg in pubsub.listen():
                if msg["type"] == "message":
                    logging.info("Received from [%s]: %s", channel, msg["data"])
                    callback(msg["data"])

        thread = threading.Thread(target=listen, daemon=True)
        thread.start()

    def close(self):
        """关闭连接"""
        self.pool.disconnect()
        logging.info("Redis connection closed.")


# -------------------- 使用示例 --------------------
if __name__ == "__main__":
    from common.config_loader import get_config_path

    r = RedisHelper(get_config_path())


    def on_message(msg):
        print("收到消息：", msg)

    # 订阅
    r.subscribe("uav_status_update", on_message)
    # 发布
    message = str({"jobId": 110, "uavSN" : "yyyyyyyyy"})
    r.publish("uav_status_update", message)

    import time
    time.sleep(3)
