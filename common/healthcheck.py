import os
import sys
import time

HEARTBEAT_FILE = "/tmp/uav_pipeline_heartbeat"
TIMEOUT = 30  # 秒

if not os.path.exists(HEARTBEAT_FILE):
    sys.exit(1)

mtime = os.path.getmtime(HEARTBEAT_FILE)
if time.time() - mtime > TIMEOUT:
    sys.exit(1)

sys.exit(0)
