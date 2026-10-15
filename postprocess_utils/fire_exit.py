import cv2
import time
import numpy as np
from .chinese_name import CLASS_NAME_TO_CHINESE

ft = cv2.freetype.createFreeType2()
ft.loadFontData(
    fontFileName="/tensorrt/uav-algorithm-pipeline/postprocess_utils/fonts/wqy-zenhei.ttc",
    idx=0
)

ALGO_ID = "55"


def point_in_box(pt, box):
    """检查点是否在矩形框内
    box: [x_min, y_min, x_max, y_max]
    pt: (x, y)
    """
    x, y = pt
    x_min, y_min, x_max, y_max = box
    return x_min <= x <= x_max and y_min <= y <= y_max


def get_center_from_box(box):
    """计算矩形框的中心点
    box: [x_min, y_min, x_max, y_max]
    """
    x_min, y_min, x_max, y_max = box
    return int((x_min + x_max) / 2), int((y_min + y_max) / 2)


def fire_exit_detections(detections, tracking, threshold=1):
    """
    Args:
        detections: JSON格式的检测结果列表，每个元素包含标准检测信息:
            {"bbox": {"x1": float, "y1": float, "x2": float, "y2": float},
             "conf": float, "class_id": int, "class_name": str, 
             "track_id": int (可选)}
        tracking: 消防通道跟踪状态字典:
            {track_id: {"t0": float or None, "seen": float,
                        "occupied": bool, "violating": bool, "alerted": bool}}
        threshold: 判定"占用消防通道"的时间阈值（秒）

    Returns:
        draw_info: 绘制信息列表，元素格式示例：
            - 消防通道:
              {"type": "fire_exit", "bbox": [x1,y1,x2,y2],
               "color": (B,G,R), "track_id": int}
            - 车辆:
              {"type": "car", "bbox": [x1,y1,x2,y2],
               "center": (cx,cy), "color": (B,G,R), "label": "car"}
        need_draw: bool，本帧是否新产生"占用消防通道"的告警
    """
    cars = []
    fire_exits = []

    # 解析检测结果，分离车辆和消防通道 
    for det in detections:
        class_name = det.get("class_name", "")

        # 获取 bbox
        bbox = det.get("bbox")
        if not bbox or not isinstance(bbox, dict):
            # 如果没有 bbox，跳过该检测结果
            continue
        x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
        x2, y2 = bbox.get("x2", 0), bbox.get("y2", 0)
        box = [int(x1), int(y1), int(x2), int(y2)]

        track_id = det.get("track_id", -1)
        if track_id is None or track_id < 0:
            track_id = -1

        if class_name == "car":
            # 直接从 bbox 计算中心点
            center = get_center_from_box(box)
            cars.append(
                {
                    "center": center,
                    "box": box,
                    "cls": class_name,
                }
            )
        elif class_name == "fire_exit":
            fire_exits.append(
                {
                    "id": int(track_id),
                    "box": box,
                }
            )

    draw_info = []
    violation_results = []
    current_time = time.time()
    need_draw = False

    # 更新消防通道状态并检测占用 
    for fire_exit in fire_exits:
        track_id = fire_exit["id"]
        fire_exit_box = fire_exit["box"]

        # 初始化 tracking
        if track_id not in tracking:
            tracking[track_id] = {
                "t0": None,  # 首次被占用时间
                "seen": current_time,
                "occupied": False,
                "violating": False,
                "alerted": False,
            }
        else:
            tracking[track_id]["seen"] = current_time

        # 检查哪些车在该消防通道内
        cars_in_fire_exit = []
        for car in cars:
            if point_in_box(car["center"], fire_exit_box):
                cars_in_fire_exit.append(car)

        is_currently_occupied = len(cars_in_fire_exit) > 0

        if is_currently_occupied:
            # 第一次被占用时记录时间
            if not tracking[track_id]["occupied"]:
                tracking[track_id]["t0"] = current_time
                tracking[track_id]["occupied"] = True
                tracking[track_id]["violating"] = False

            occupied_duration = current_time - tracking[track_id]["t0"]
            was_violating = tracking[track_id]["violating"]
            now_violating = occupied_duration >= threshold

            # 本帧刚刚满足“占用阈值”
            if now_violating and not was_violating:
                tracking[track_id]["violating"] = True
                if not tracking[track_id]["alerted"]:
                    tracking[track_id]["alerted"] = True
                    need_draw = True
                    violation_results.append(
                        ["fire_exit", *fire_exit_box, track_id, True]
                    )
        else:
            # 当前未被占用，则清空占用状态
            if tracking[track_id]["occupied"]:
                tracking[track_id]["occupied"] = False
                tracking[track_id]["t0"] = None
                tracking[track_id]["violating"] = False
            # 根据需求：同一个消防通道只告警一次，因此不重置 alerted

        # 确定消防通道颜色
        occupied_duration = 0
        if tracking[track_id]["occupied"] and tracking[track_id]["t0"] is not None:
            occupied_duration = current_time - tracking[track_id]["t0"]

        if tracking[track_id]["occupied"]:
            if occupied_duration >= threshold:
                color = (0, 0, 255)  # 被占用：红
            else:
                color = (0, 255, 255)  # 待判定：黄
        else:
            color = (0, 255, 0)  # 空闲：绿

        draw_info.append(
            {
                "type": "fire_exit",
                "bbox": fire_exit_box,
                "color": color,
                "track_id": track_id,
            }
        )

    # 车辆绘制信息（统一用黄色框）
    for car in cars:
        box = car["box"]
        center = car["center"]
        draw_info.append(
            {
                "type": "car",
                "bbox": box,
                "center": center,
                "color": (0, 255, 255),
                "label": car["cls"],
            }
        )

    # 清理过期 tracking
    for track_id in list(tracking.keys()):
        is_alive = any(f["id"] == track_id for f in fire_exits)
        is_occupied_marked = tracking[track_id]["occupied"]
        last_seen_time = tracking[track_id]["seen"]
        time_since_last_seen = current_time - last_seen_time

        if not is_alive:
            should_keep = is_occupied_marked and time_since_last_seen < 30
            if not should_keep:
                tracking.pop(track_id)

    return draw_info, need_draw


def fire_exit_draw_detection_results(image, draw_info):
    """ 

    Args:
        image: 输入图像 (BGR numpy array)
        draw_info: 来自 `fire_exit_detections` 的绘制信息列表

    Returns:
        img_bgr: 绘制后的图像
        violation_class_name: 违规则返回 "fire_exit"，否则为 "fire_exit"
    """
    img_bgr = image.copy()
    violation = False

    for item in draw_info:
        item_type = item.get("type")
        bbox = item.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, bbox)

        if item_type == "fire_exit":
            color = item.get("color", (0, 255, 0))
            label_cn = CLASS_NAME_TO_CHINESE.get("fire_exit", "fire_exit")
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            ft.putText(
                img_bgr,
                str(label_cn),
                (x1, max(0, y1 + 20)),
                fontHeight=20,
                color=color,
                thickness=-1,
                line_type=cv2.LINE_AA,
                bottomLeftOrigin=False,
            )
            # 红色视为已违规占用
            if color == (0, 0, 255):
                violation = True

        elif item_type == "car":
            color = item.get("color", (0, 255, 255))
            label = item.get("label", "car")
            label_cn = CLASS_NAME_TO_CHINESE.get(label, label)
            center = item.get("center", (x1, y1))
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            ft.putText(
                img_bgr,
                str(label_cn),
                tuple(center),
                fontHeight=20,
                color=color,
                thickness=-1,
                line_type=cv2.LINE_AA,
                bottomLeftOrigin=False,
            )

    violation_class_name = "fire_exit" if violation else "fire_exit"
    return img_bgr, violation_class_name


EXPORT = {
    "detect": fire_exit_detections,
    "draw": fire_exit_draw_detection_results,
}