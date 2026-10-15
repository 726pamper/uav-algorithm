import math
import time
import cv2
import numpy as np

ALGO_ID = "69"

# 路段定义：两个y坐标区间，距离30米
ROAD_SEGMENT_1_Y_RANGE = (109, 248)  # 第一段y坐标范围
ROAD_SEGMENT_2_Y_RANGE = (656, 925)  # 第二段y坐标范围
SEGMENT_DISTANCE_METERS = 24.0       # 两段之间的距离（米）

def _get_homography_and_ppm():
    """
    计算并返回单应矩阵 H 和 ppm（每米多少像素）。
    """
    src_pts = np.array(
        # [
            # [570, 32],
            # [705, 32],
            # [684, 1058],
            # [400, 1058],
        # ],
        [
            [524, 33],
            [659, 33],
            [645, 1058],
            [361, 1059],
        ],
        dtype=np.float32,
    )

    width_m = 11.25
    height_m = 150.0
    dst_world = np.array(
        [[0, 0], [width_m, 0], [width_m, height_m], [0, height_m]],
        dtype=np.float32,
    )

    ppm = 100.0
    dst_pts = dst_world * ppm

    H, _ = cv2.findHomography(src_pts, dst_pts)
    if H is None:
        raise RuntimeError("Failed to compute homography matrix.")

    return H, ppm


def calc_real_distance(p1_img, p2_img):
    """
    计算两点在地面真实世界中的距离（米）。
    """
    H, ppm = _get_homography_and_ppm()

    pts = np.array([[p1_img, p2_img]], dtype=np.float32)

    mapped = cv2.perspectiveTransform(pts, H)[0]
    mapped_m = mapped / ppm

    p1_m = mapped_m[0]
    p2_m = mapped_m[1]
    distance_m = math.hypot(float(p1_m[0] - p2_m[0]), float(p1_m[1] - p2_m[1]))

    return distance_m


def _get_vehicle_segment(y_coord):
    """
    根据y坐标判断车辆在哪个路段。

    Args:
        y_coord: 车辆中心的y坐标

    Returns:
        int: 0=不在路段内, 1=第一段, 2=第二段
    """
    if ROAD_SEGMENT_1_Y_RANGE[0] <= y_coord <= ROAD_SEGMENT_1_Y_RANGE[1]:
        return 1
    elif ROAD_SEGMENT_2_Y_RANGE[0] <= y_coord <= ROAD_SEGMENT_2_Y_RANGE[1]:
        return 2
    else:
        return 0


def update_speed_from_det_tensor(detections, track_state, track_speeds):
    """
    基于 DET 结果使用路段时间测量计算每个 track_id 的车速（km/h）。

    通过车辆经过固定路段的时间来计算速度，两段间距离30米。
    车辆进入第一段后计算速度，直到第二段计算新速度前保持不变。

    - detections: Torch tensor，形状 (N,11) = [x1,y1,x2,y2,conf,class_id,cxn,cyn,w,h,track_id]
    - track_state: {track_id: {"current_segment": int, "segment_enter_time": float, "speed_calculated": bool, "last_speed_kmh": float}}
    - track_speeds: {track_id: speed_kmh}，用于 OSD 显示
    """
    if detections is None:
        return

    try:
        import numpy as _np  # 局部导入避免命名冲突

        arr = _np.asarray(detections)
        if arr.size == 0:
            return
    except Exception:
        return

    current_time = time.time()

    try:
        for det in detections:
            if det.shape[0] < 11:
                continue
            # 11元素格式: [x1, y1, x2, y2, conf, class_id, cxn, cyn, w, h, track_id]
            x1, y1, x2, y2, conf, class_id, cxn, cyn, w, h, track_id = det.tolist()[:11]
            track_id = int(track_id)
            if track_id < 0:
                continue
            # 只处理 car（class_id == 0）
            if int(class_id) != 0:
                continue

            # 获取车辆中心点坐标
            cx = (float(x1) + float(x2)) / 2.0
            cy = (float(y1) + float(y2)) / 2.0

            # 判断车辆当前在哪个路段
            current_segment = _get_vehicle_segment(cy)

            # 初始化车辆状态
            if track_id not in track_state:
                track_state[track_id] = {
                    "current_segment": 0,        # 当前路段 (0=路段外, 1=第一段, 2=第二段)
                    "segment_enter_time": 0.0,   # 进入当前路段的时间
                    "speed_calculated": False,   # 该路段是否已计算过速度
                    "last_speed_kmh": 0.0        # 当前速度
                }

            vehicle_state = track_state[track_id]
            prev_segment = vehicle_state["current_segment"]

            # 检测路段变化
            if current_segment != prev_segment:
                if current_segment > 0:  # 进入路段l
                    # 记录进入时间
                    vehicle_state["current_segment"] = current_segment
                    vehicle_state["segment_enter_time"] = current_time
                    vehicle_state["speed_calculated"] = False
                    print(f"[路段测速] 车辆{track_id} 进入路段{current_segment}, y={cy:.1f}")

                else:  # 离开路段
                    # 计算通过时间和速度
                    if prev_segment > 0 and not vehicle_state["speed_calculated"]:
                        enter_time = vehicle_state["segment_enter_time"]
                        pass_time = current_time - enter_time

                        if pass_time > 0.3:  # 至少0.2秒，避免瞬间通过
                            # 速度 = 距离 / 时间 (km/h)
                            speed_m_s = SEGMENT_DISTANCE_METERS / pass_time
                            speed_kmh = speed_m_s * 3.6

                            # 检查速度是否超过160km/h上限
                            if speed_kmh > 160.0:
                                print(f"[路段测速] 车辆{track_id} 速度异常({speed_kmh:.1f}km/h > 160km/h)，跳过计算")
                            else:
                                # 更新状态
                                vehicle_state["last_speed_kmh"] = speed_kmh
                                vehicle_state["speed_calculated"] = True

                                # 更新速度显示
                                track_speeds[track_id] = int(speed_kmh)

                                print(f"[路段测速] 车辆{track_id} 离开路段{prev_segment}, "
                                      f"通过时间={pass_time:.3f}s, 速度={speed_kmh:.1f}km/h")
                        else:
                            print(f"[路段测速] 车辆{track_id} 通过路段{prev_segment}过快，跳过计算")

                    # 重置路段状态
                    vehicle_state["current_segment"] = 0
                    vehicle_state["segment_enter_time"] = 0.0

            # 如果车辆仍在路段内，保持当前状态

    except Exception as e:
        # DeepStream 侧出错时不影响主流程
        print(f"[路段测速] Error in segment-based calculation: {e}")
        pass


def speed_detections(dets, tracking, speed_threshold_kmh=132.0):
    """
    检测车辆超速，超速时返回所有车辆信息。
    """
    # print(f"[调试] 接收到的检测结果 dets: {dets}")

    vehicles = []
    has_overspeed = False

    for det in dets:
        if det.get("model_code") != "999":
            continue

        track_id = det.get("track_id")
        if track_id is None:
            continue

        # 允许 "car" 和 "left_lane" 类名（后者是车辆）
        class_name = det.get("class_name", "")
        class_id = int(det.get("class_id", -1))

        speed = det.get("speed")
        if speed is None:
            speed = det.get("speed_kmh")  # 尝试使用speed_kmh字段
        if speed is None:
            continue

        speed = int(float(speed))
        print(f"[调试] 获取的速度值 speed: {speed}")

        is_overspeed = speed > speed_threshold_kmh

        # 更新跟踪状态
        if track_id not in tracking:
            tracking[track_id] = {"last_speed_kmh": speed, "overspeed": is_overspeed}
        else:
            tracking[track_id]["last_speed_kmh"] = speed
            tracking[track_id]["overspeed"] = is_overspeed

        # 收集车辆信息
        bbox = det.get("bbox")
        if not bbox:
            cx, cy = det.get("cx", 0), det.get("cy", 0)
            bbox = {
                "x1": det.get("x1", cx), "y1": det.get("y1", cy),
                "x2": det.get("x2", cx), "y2": det.get("y2", cy)
            }

        vehicles.append({
            "bbox": bbox,
            "track_id": int(track_id),
            "speed_kmh": speed,
            "is_overspeed": is_overspeed
        })

        if is_overspeed:
            has_overspeed = True

    # 有超速时才绘制所有车辆
    return vehicles if has_overspeed else [], has_overspeed


def speed_draw_detection_results(frame_bgr, draw_info):
    """
    绘制车辆：超速红色，正常绿色。
    """
    img = frame_bgr.copy()
    for info in draw_info:
        bbox = info.get("bbox", {})
        speed_kmh = info.get("speed_kmh", 0)
        is_overspeed = info.get("is_overspeed", False)

        x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
        x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))

        color = (0, 0, 255) if is_overspeed else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{speed_kmh} km/h", (x1, max(0, y1 - 5)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img, "car"

EXPORT = {
    'detect':speed_detections ,
    'draw': speed_draw_detection_results,
}
