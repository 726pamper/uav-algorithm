# 1. 定义名称和变量
AI_CONTAINER_NAME="uav-ai"
REDIS_CONTAINER_NAME="uav-ai-redis"
IMAGE_NAME="uav-ai:latest"
NETWORK_NAME="uav-network"

sleep 3

docker exec -d $AI_CONTAINER_NAME supervisord -n -c /etc/supervisor/supervisord.conf

