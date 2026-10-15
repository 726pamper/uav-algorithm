import cv2, time
import numpy as np
from .chinese_name import CLASS_NAME_TO_CHINESE

ft = cv2.freetype.createFreeType2()
ft.loadFontData(
    fontFileName="/tensorrt/uav-algorithm-pipeline/postprocess_utils/fonts/wqy-zenhei.ttc",
    idx=0
)

ALGO_ID = "57"


def point_in_obb(pt, obb):
    obb = obb.reshape(-1, 2).astype(np.float32)
    return cv2.pointPolygonTest(obb, pt, False) >= 0


def get_center(obb):
    pts = obb.reshape(-1, 2)
    return int(pts[:, 0].mean()), int(pts[:, 1].mean())


def vehicle_detections(detections, tracking, threshold=0):
    """
    Args:
        detections: JSON格式的检测结果列表，每个元素包含 OBB 信息:
            {"polygon": [{"x": float, "y": float}, ...], "cx": float, "cy": float,
             "w": float, "h": float, "angle": float, "conf": float,
             "class_id": int, "class_name": str, "track_id": int (可选),
             "bbox": {"x1": float, "y1": float, "x2": float, "y2": float}}
        tracking: 跟踪状态字典，格式: {track_id: {"t0": float, "seen": float,
                                                   "ever_in": bool, "violating": bool, "alerted": bool}}
        threshold: 违停判定时间阈值（秒），默认1秒

    Returns:
        draw_info: 需要绘制的信息列表，每个元素包含：
            - 车辆: {"type": "electric_vehicle", "polygon": np.array,
                    "bbox": [x1, y1, x2, y2], "center": (cx, cy),
                    "color": (B, G, R), "label": str, "track_id": int}
            - 停车位: {"type": "parking_spot", "polygon": np.array,
                      "bbox": [x1, y1, x2, y2], "center": (cx, cy),
                      "label": str}
        need_draw: bool, 是否需要绘制（有违停车辆时为True，否则为False）
    """
    cars = []  # 存储车辆信息：{id, center, obb, box, cls}
    parking_spots = []  # 存储停车位信息：{obb, box}

    # === 步骤1：分类检测结果，分离车辆和停车位 ===
    for det in detections:
        class_name = det.get("class_name", "")
        class_id = det.get("class_id", -1)

        polygon = None
        polygon_list = det.get("polygon")
        if isinstance(polygon_list, list) and len(polygon_list) > 0:
            if isinstance(polygon_list[0], dict):
                polygon = np.array([[p["x"], p["y"]] for p in polygon_list], dtype=np.float32)
            else:
                polygon = np.array(polygon_list, dtype=np.float32).reshape(-1, 2)

        obb = polygon
        bbox = det.get("bbox")
        if not bbox or not isinstance(bbox, dict):
            # 如果没有 bbox，跳过该检测结果
            continue
        x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
        x2, y2 = bbox.get("x2", 0), bbox.get("y2", 0)
        bounding_box = np.array([x1, y1, x2, y2], dtype=int)
        center = get_center(obb) if obb is not None else (int((x1 + x2) / 2), int((y1 + y2) / 2))

        track_id = det.get("track_id", -1)
        if track_id is None or track_id < 0:
            track_id = -1

        # 根据类别分类
        if class_name == "electric_vehicle":
            cars.append({
                "id": track_id,
                "center": center,
                "obb": obb,
                "box": bounding_box,
                "cls": class_name
            })
        elif class_name == "tcw":  # 停车位
            parking_spots.append({
                "obb": obb,
                "box": bounding_box
            })
    # 没有检测到非机动车，或者没有检测到停车位时，无法判断违停，直接不上报
    if not cars or not parking_spots:
        return [], False

    # 用于存储绘制信息
    draw_info = []

    # 添加停车位绘制信息
    for spot in parking_spots:
        center_x, center_y = get_center(spot["obb"]) if spot["obb"] is not None else (
            int((spot["box"][0] + spot["box"][2]) / 2),
            int((spot["box"][1] + spot["box"][3]) / 2)
        )
        draw_info.append({
            "type": "parking_spot",
            "polygon": spot["obb"],
            "bbox": spot["box"].tolist(),
            "center": (center_x, center_y),
            "label": "tcw",
            "editable": False,
            "box_role": "region",
            "is_violation": False,
        })

    # 用于存储"本帧新出现的违停车辆"的检测结果
    violation_results = []
    current_time = time.time()
    need_draw = False  # 是否需要绘制的标志

    # 更新车辆跟踪状态并检测违停
    for car in cars:
        track_id = car["id"]
        car_center = car["center"]

        # 初始化或更新跟踪记录
        if track_id not in tracking:
            tracking[track_id] = {
                "t0": current_time,
                "seen": current_time,
                "ever_in": False,
                "violating": False,
                "alerted": False
            }
        else:
            tracking[track_id]["seen"] = current_time

        # 检查车辆中心点是否在任何停车位内
        is_in_parking_spot = any(
            point_in_obb(car_center, spot["obb"])
            for spot in parking_spots
            if spot["obb"] is not None
        )

        if is_in_parking_spot:
            tracking[track_id]["ever_in"] = True
            if tracking[track_id]["violating"]:
                tracking[track_id]["violating"] = False
            if tracking[track_id]["alerted"]:
                tracking[track_id]["alerted"] = False

        # 计算车辆停留时间
        first_seen_time = tracking[track_id]["t0"]
        stay_duration = current_time - first_seen_time

        # 违停判断
        has_entered_parking = tracking[track_id]["ever_in"]
        was_violating = tracking[track_id]["violating"]
        now_violating = (stay_duration >= threshold and not has_entered_parking and len(parking_spots) > 0)

        if now_violating and not was_violating:
            tracking[track_id]["violating"] = True

        # 当前是否处于违停状态（用于绘制颜色）
        is_currently_violating = tracking[track_id]["violating"]
        is_pending_violation = (not has_entered_parking) and (stay_duration < threshold)

        # 确定颜色：违停红色，待判定黄色，正常绿色
        color = (0, 255, 0)  # 正常：绿色 (BGR)
        if is_pending_violation:
            color = (0, 255, 255)  # 待判定：黄色
        if is_currently_violating:
            color = (0, 0, 255)  # 违停：红色

        # 添加车辆绘制信息
        draw_info.append({
            "type": "electric_vehicle",
            "polygon": car["obb"],
            "bbox": car["box"].tolist(),
            "center": car_center,
            "color": color,
            "label": "electric_vehicle",
            "track_id": track_id,
            "editable": True,
            "box_role": "target",
            "is_violation": is_currently_violating,
        })

        # 仅在"本帧刚刚变为违停"且尚未告警过时，设置需要绘制标志
        if now_violating and not was_violating and not tracking[track_id]["alerted"]:
            tracking[track_id]["alerted"] = True
            need_draw = True
            violation_results.append([car["cls"], *car["box"], track_id, True])

    # 如果有违停车辆，将所有停车位也加入违停结果（用于截图）
    if violation_results:
        for spot in parking_spots:
            violation_results.append(["tcw", *spot["box"], None])

    # 清除过期的跟踪记录
    for track_id in list(tracking.keys()):
        is_alive = any(car["id"] == track_id for car in cars)
        is_violation_marked = tracking[track_id]["violating"]
        last_seen_time = tracking[track_id]["seen"]
        time_since_last_seen = current_time - last_seen_time

        if not is_alive:
            should_keep = is_violation_marked and time_since_last_seen < 30
            if not should_keep:
                tracking.pop(track_id)

    # 如果不需要绘制，返回空列表和False
    if not need_draw:
        return [], False

    return draw_info, True


def vehicle_draw_detection_results(image, draw_info):
    """
    根据OBB绘制信息绘制检测结果

    Args:
        image: 输入图像 (BGR numpy array)
        draw_info: 来自vehicle_detections的绘制信息列表，包含车辆与停车位的多边形坐标

    Returns:
        drawn_image: 绘制后的图像 (BGR格式)
        violation_class_name: 违停车辆类别名称（如果有违停车辆），否则为 "electric_vehicle"
    """
    img_bgr = image.copy()
    violation_class_name = "electric_vehicle"  # 类别名称

    # 根据draw_info进行绘制
    for item in draw_info:
        item_type = item.get("type")

        if item_type == "electric_vehicle":
            polygon = item.get("polygon")
            bbox = item.get("bbox")
            color = item.get("color", (0, 255, 0))  # 绿色 (BGR)
            label = item.get("label", "electric_vehicle")
            label_cn = CLASS_NAME_TO_CHINESE.get(label, label)
            track_id = item.get("track_id", -1)

            if polygon is not None:
                # 使用旋转框多边形进行绘制，而不是最小外接矩形
                pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_bgr, [pts], isClosed=True, color=color, thickness=2)

                # 为了放置文字标签，仍然计算一个轴对齐外接框，但不再调用 obb_to_aabb
                xs = pts[:, 0, 0]
                ys = pts[:, 0, 1]
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
            elif bbox:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            else:
                continue

            # 绘制标签：仅显示中文，不改变英文 label 参与的逻辑
            label_text = label_cn if track_id < 0 else f"{label_cn}"
            ft.putText(
                img_bgr,
                str(label_text),
                (x1, max(0, y1 + 20)),
                fontHeight=20,
                color=color,
                thickness=-1,
                line_type=cv2.LINE_AA,
                bottomLeftOrigin=False,
            )

            # 记录违停车辆类别名称（红色表示违停），仍然使用英文 label
            if color == (0, 0, 255):  # 红色表示违停 (BGR)
                violation_class_name = "illegal_parking_electric_vehicle"

        elif item_type == "parking_spot":
            polygon = item.get("polygon")
            bbox = item.get("bbox")
            label = item.get("label", "tcw")
            label_cn = CLASS_NAME_TO_CHINESE.get(label, label)

            if polygon is not None:
                # 使用旋转框多边形进行绘制，而不是最小外接矩形
                pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_bgr, [pts], isClosed=True, color=(255, 0, 0), thickness=1)

                # 为了放置文字标签，仍然计算一个轴对齐外接框，但不再调用 obb_to_aabb
                xs = pts[:, 0, 0]
                ys = pts[:, 0, 1]
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
            elif bbox:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 1)
            else:
                continue
            ft.putText(
                img_bgr,
                str(label_cn),
                (x1, max(0, y1 + 20)),
                fontHeight=20,
                color=(255, 0, 0),
                thickness=-1,
                line_type=cv2.LINE_AA,
                bottomLeftOrigin=False,
            )

    return img_bgr, violation_class_name


EXPORT = {
    "detect": vehicle_detections,
    "draw": vehicle_draw_detection_results,
}