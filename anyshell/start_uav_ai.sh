# 首次部署脚本（按你的实际环境修改路径、镜像名与容器名后执行）
#
# 约定：
#   宿主机代码目录  -> 容器内 /tensorrt/uav-algorithm-pipeline
#   容器使用 host 网络（推理服务需要直连 RTSP/RTMP 与平台 MQTT）

# 1. 宿主机上存放代码与模型的目录
workdir=/path/to/TensorRT

# 2. 启动算法容器（挂载代码目录、开放 GPU）
docker run -itd --gpus all --restart=always --network=host \
  --name uav-ai \
  -v ${workdir}:/tensorrt \
  uav-ai:latest

# 3. 安装服务看门狗（systemd timer 定时探活并自动拉起容器内服务）
cd ${workdir}/uav-algorithm-pipeline/anyshell && cp -r supervisor-watchdog* /etc/systemd/system
sudo systemctl daemon-reload
sudo systemctl start supervisor-watchdog.timer
sudo systemctl enable supervisor-watchdog.timer

# 4. 首次手动启动容器内服务
docker exec -it uav-ai bash -c "cd /tensorrt/uav-algorithm-pipeline/anyshell && ./restart.sh"
