import cv2
import time
import numpy as np
from .chinese_name import CLASS_NAME_TO_CHINESE

ft = cv2.freetype.createFreeType2()
ft.loadFontData(
    fontFileName="/tensorrt/uav-algorithm-pipeline/postprocess_utils/fonts/wqy-zenhei.ttc",
    idx=0
)

ALGO_ID = "77"  # 充电桩算法编号


def get_center_from_box(box):
    """计算矩形框的中心点

    Args:
        box: [x1, y1, x2, y2]

    Returns:
        (cx, cy): 中心点坐标
    """
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def box_in_box(inner_box, outer_box):
    """判断一个矩形框是否在另一个矩形框内

    Args:
        inner_box: [x1, y1, x2, y2] 内部框
        outer_box: [x1, y1, x2, y2] 外部框

    Returns:
        bool: 如果内部框的中心点在外部框内，返回 True
    """
    inner_cx, inner_cy = get_center_from_box(inner_box)
    outer_x1, outer_y1, outer_x2, outer_y2 = outer_box

    return (outer_x1 <= inner_cx <= outer_x2 and
            outer_y1 <= inner_cy <= outer_y2)


def charging_pile_detections(detections, tracking):
    """
    充电桩算法检测结果处理

    Args:
        detections: JSON格式的检测结果列表，每个元素包含标准检测信息:
            {"bbox": {"x1": float, "y1": float, "x2": float, "y2": float},
             "conf": float, "class_id": int, "class_name": str,
             "track_id": int (可选)}
        tracking: 跟踪状态字典:
            {track_id: {"status": str, "seen": float}}
            status 可能的值: "normal" (正常充电枪摆放), "charging" (充电中), "abnormal" (异常)

    Returns:
        draw_info: 绘制信息列表，只包含 plug 的绘制信息:
            {"type": "plug", "bbox": [x1,y1,x2,y2],
             "color": (B,G,R), "label": str, "status": str, "track_id": int}
        need_draw: bool，本帧是否有异常状态需要上报
    """
    plugs = []
    chargers = []
    cars = []

    # 解析检测结果，分离三种类别
    for det in detections:
        class_name = det.get("class_name", "").lower()

        # 获取 bbox
        bbox = det.get("bbox")
        if not bbox or not isinstance(bbox, dict):
            continue

        x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
        x2, y2 = bbox.get("x2", 0), bbox.get("y2", 0)
        box = [int(x1), int(y1), int(x2), int(y2)]

        track_id = det.get("track_id", -1)
        if track_id is None or track_id < 0:
            track_id = -1

        if class_name == "plug":
            plugs.append({
                "box": box,
                "track_id": track_id,
                "conf": det.get("conf", 0.0),
                "class_id": det.get("class_id", -1)
            })
        elif class_name == "charger":
            chargers.append({
                "box": box,
                "track_id": track_id
            })
        elif class_name == "car":
            cars.append({
                "box": box,
                "track_id": track_id
            })

    draw_info = []
    need_draw = False
    current_time = time.time()

    # 处理每个 plug，判断其状态
    for plug in plugs:
        plug_box = plug["box"]
        track_id = plug["track_id"]

        # 初始化跟踪状态
        if track_id not in tracking:
            tracking[track_id] = {
                "status": "abnormal",  # 默认异常
                "seen": current_time,
                "alerted": False
            }
        else:
            tracking[track_id]["seen"] = current_time

        # 判断 plug 的状态
        status = "abnormal"  # 默认异常
        label = "异常"
        color = (0, 0, 255)  # 红色表示异常

        # 检查 plug 是否在 charger 内（正常充电枪摆放，未充电）
        in_charger = False
        for charger in chargers:
            if box_in_box(plug_box, charger["box"]):
                in_charger = True
                status = "normal"
                label = "正常"
                color = (0, 255, 0)  # 绿色表示正常
                break

        # 如果不在 charger 内，检查是否在 car 内（充电中）
        if not in_charger:
            in_car = False
            for car in cars:
                if box_in_box(plug_box, car["box"]):
                    in_car = True
                    status = "charging"
                    label = "充电中"
                    color = (0, 255, 255)  # 黄色表示充电中
                    break

        # 更新跟踪状态
        previous_status = tracking[track_id]["status"]
        tracking[track_id]["status"] = status

        # 上报策略：只有异常状态需要上报
        # 如果状态从非异常变为异常，需要上报
        if status == "abnormal":
            if previous_status != "abnormal":
                # 状态刚变为异常，需要上报
                need_draw = True
                tracking[track_id]["alerted"] = True
            elif not tracking[track_id].get("alerted", False):
                # 异常状态持续但未告警过，也需要上报
                need_draw = True
                tracking[track_id]["alerted"] = True
        else:
            # 恢复正常状态，重置告警标志（下次异常时再上报）
            if previous_status == "abnormal":
                tracking[track_id]["alerted"] = False

        # 只添加 plug 到绘制信息（charger 和 car 不绘制）
        draw_info.append({
            "type": "plug",
            "bbox": plug_box,
            "color": color,
            "label": label,
            "status": status,
            "track_id": track_id,
            "conf": plug["conf"]
        })

    # 清理过期的跟踪记录
    for track_id in list(tracking.keys()):
        is_alive = any(p["track_id"] == track_id for p in plugs)
        last_seen_time = tracking[track_id]["seen"]
        time_since_last_seen = current_time - last_seen_time

        if not is_alive and time_since_last_seen > 30:
            tracking.pop(track_id)

    return draw_info, need_draw


def charging_pile_draw_detection_results(image, draw_info):
    """
    绘制充电桩检测结果（只绘制 plug）

    Args:
        image: 输入图像 (BGR numpy array)
        draw_info: 来自 `charging_pile_detections` 的绘制信息列表

    Returns:
        img_bgr: 绘制后的图像
        violation_class_name: 类别名称，异常时返回 "charging_pile_abnormal"，否则返回 "charging_pile"
    """
    img_bgr = image.copy()
    violation_class_name = "charging_pile"
    has_abnormal = False

    for item in draw_info:
        item_type = item.get("type")
        if item_type != "plug":
            continue

        bbox = item.get("bbox") or []
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        color = item.get("color", (0, 0, 255))
        label = item.get("label", "plug")
        status = item.get("status", "abnormal")
        track_id = item.get("track_id", -1)
        conf = item.get("conf", 0.0)

        # 绘制矩形框
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

        # 构建标签文本
        if track_id >= 0:
            label_text = f"{label} ID:{track_id} {conf:.2f}"
        else:
            label_text = f"{label} {conf:.2f}"

        # 使用中文字体绘制标签
        label_cn = CLASS_NAME_TO_CHINESE.get("plug", "充电枪")
        if status == "abnormal":
            label_cn = f"{label_cn}(异常)"
            has_abnormal = True
        elif status == "charging":
            label_cn = f"{label_cn}(充电中)"
        else:
            label_cn = f"{label_cn}(正常)"

        ft.putText(
            img_bgr,
            label_cn,
            (x1, max(0, y1 - 25)),
            fontHeight=30,
            color=color,
            thickness=-1,
            line_type=cv2.LINE_AA,
            bottomLeftOrigin=False
        )

    if has_abnormal:
        violation_class_name = "charging_pile_abnormal"

    return img_bgr, violation_class_name


EXPORT = {
    "detect": charging_pile_detections,
    "draw": charging_pile_draw_detection_results,
}
