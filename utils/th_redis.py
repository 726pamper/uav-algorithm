import time
import threading
from redis_helper import RedisHelper

def test_pub_sub():
    r = RedisHelper("../config-dev.yaml")

    # 定义消息回调
    def on_message(msg):
        print(f"收到消息：{msg}")

    # 启动订阅线程
    sub_thread = threading.Thread(target=lambda: r.subscribe("uav_status_update", on_message), daemon=True)
    sub_thread.start()

    # 延迟 1 秒，确保订阅者启动完成
    time.sleep(1)

    # 启动发布线程
    def publisher():
        for i in range(5):
            message = str({"jobId": 100 + i, "uavSN": f"SN_{i:03d}"})
            print(f"发布消息：{message}")
            r.publish("uav_status_update", message)
            time.sleep(1)

    pub_thread = threading.Thread(target=publisher, daemon=True)
    pub_thread.start()

    # 主线程等待子线程执行完
    pub_thread.join()
    time.sleep(2)
    r.close()

if __name__ == "__main__":
    test_pub_sub()
