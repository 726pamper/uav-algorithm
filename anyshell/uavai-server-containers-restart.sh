#!/bin/bash

# 1. 定义名称和变量
AI_CONTAINER_NAME="uav-ai"
REDIS_CONTAINER_NAME="uav-ai-redis"
IMAGE_NAME="uav-ai:latest"
NETWORK_NAME="uav-network"

# 2. 创建一个 Docker 网络
docker network create $NETWORK_NAME 2>/dev/null || true

# 3. 停止并删除旧容器
echo "清理旧容器"
docker rm -f $AI_CONTAINER_NAME $REDIS_CONTAINER_NAME 2>/dev/null

# 4. 启动 Redis 容器
echo "正在启动 Redis..."
docker run -d \
  --name $REDIS_CONTAINER_NAME \
  --network $NETWORK_NAME \
  --restart always \
  -p 6379:6379 \
  redis:latest

# 5. 启动算法容器 
# 注意：这里我们显式加上驱动能力，确保万无一失
echo "正在启动算法容器..."
docker run -d \
  --name $AI_CONTAINER_NAME \
  --network $NETWORK_NAME \
  --gpus all \
  --restart always \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e REDIS_HOST=$REDIS_CONTAINER_NAME \
  -e REDIS_PORT=6379 \
  -v /path/to/uav_ai/:/tensorrt \
  $IMAGE_NAME \
  sleep infinity 

# 6. 等待 5 秒，确保容器内部的 NVIDIA 驱动挂载和链接完全就绪
echo "等待系统环境初始化 (5s)..."
sleep 5

# 7. 手动触发 supervisord
echo "激活任务管理系统 (supervisord)..."
docker exec -d $AI_CONTAINER_NAME supervisord  -c /etc/supervisor/supervisord.conf

echo "------------------------------------------------"
echo "全部启动完成！"
echo "查看算法日志请执行: docker exec -it $AI_CONTAINER_NAME tail -f /var/log/mqtt_redis.err.log"
echo "------------------------------------------------"