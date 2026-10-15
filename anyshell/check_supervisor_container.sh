#!/usr/bin/env bash

CONTAINER_ID="uav-ai"

DOCKER_EXEC="docker exec -i"
SUPERVISOR_STATUS_CMD="supervisorctl status"
SUPERVISOR_RELOAD_CMD="supervisorctl reload"
SUPERVISORD_START_CMD="supervisord -c /etc/supervisor/supervisord.conf"

LOG_TAG="[supervisor-watchdog]"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') ${LOG_TAG} $1"
}

# ---------- check container running ----------
if ! docker inspect -f '{{.State.Running}}' "${CONTAINER_ID}" 2>/dev/null | grep -q true; then
    log "Container ${CONTAINER_ID} is not running, attempting to start"

    if ! docker start "${CONTAINER_ID}" >/dev/null 2>&1; then
        log "Failed to start container ${CONTAINER_ID}"
        exit 1
    fi

    # 给容器一点启动时间
    sleep 3

    # 二次确认
    if ! docker inspect -f '{{.State.Running}}' "${CONTAINER_ID}" 2>/dev/null | grep -q true; then
        log "Container ${CONTAINER_ID} failed to enter running state"
        exit 1
    fi

    log "Container ${CONTAINER_ID} started successfully"
fi


# ===============================
# 2. NVML / GPU 状态检查（最高优先级）
# ===============================
NVML_OUTPUT=$(${DOCKER_EXEC} "${CONTAINER_ID}" nvidia-smi 2>&1)

if echo "${NVML_OUTPUT}" | grep -qi "Failed to initialize NVML"; then
    log "NVML initialization failed, restarting container ${CONTAINER_ID}"
    docker restart "${CONTAINER_ID}" >/dev/null
    exit 10
fi

# ===============================
# 3. supervisor 状态检查
# ===============================
STATUS_OUTPUT=$(${DOCKER_EXEC} "${CONTAINER_ID}" ${SUPERVISOR_STATUS_CMD} 2>&1)
RET=$?

# ---------- socket refused ----------
if echo "${STATUS_OUTPUT}" | grep -Eqi "supervisor\.sock.*(no such file|refused connection)"; then
    log "Supervisor socket error (not available)"

    # 检查 supervisord 进程是否存在
    if ! ${DOCKER_EXEC} "${CONTAINER_ID}" pgrep -x supervisord >/dev/null 2>&1; then
        log "supervisord not running, starting supervisord"
        ${DOCKER_EXEC} "${CONTAINER_ID}" ${SUPERVISORD_START_CMD} >/dev/null 2>&1
        exit 5
    else
        log "supervisord process exists but socket unavailable (manual intervention recommended)"
        exit 6
    fi
fi

# ---------- supervisorctl 执行失败 ----------
if [ ${RET} -ne 0 ]; then
    log "supervisorctl failed: ${STATUS_OUTPUT}"
    exit 3
fi

# ===============================
# 4. 检测非 RUNNING 服务
# ===============================
NOT_RUNNING=$(echo "${STATUS_OUTPUT}" | grep -v RUNNING || true)

if [ -n "${NOT_RUNNING}" ]; then
    log "Detected non-RUNNING supervisor services:"
    echo "${NOT_RUNNING}"

    log "Reloading supervisor"
    ${DOCKER_EXEC} "${CONTAINER_ID}" ${SUPERVISOR_RELOAD_CMD} >/dev/null 2>&1
    exit 4
fi

# ===============================
# 5. 正常
# ===============================
log "Supervisor services OK and NVML OK"
exit 0
