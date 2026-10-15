import cv2
import time
import json
import numpy as np
from .geo_area_checker import GeoAreaChecker
from .chinese_name import CLASS_NAME_TO_CHINESE

ft = cv2.freetype.createFreeType2()
ft.loadFontData(
    fontFileName="/tensorrt/uav-algorithm-pipeline/postprocess_utils/fonts/wqy-zenhei.ttc",
    idx=0
)

ALGO_ID = 'detect'


def normal_detections(detections, tracking, area=None, coordinates=None):
    """
    普通检测结果处理，相同 track_id 只绘制一次

    Args:
        detections: JSON格式的检测结果列表，每个元素包含标准检测信息:
            {"bbox": {"x1": float, "y1": float, "x2": float, "y2": float},
             "conf": float, "class_id": int, "class_name": str,
             "track_id": int (可选)}
        tracking: 跟踪状态字典:
            {track_id: {"timestamp": float, "is_draw": bool}}
        area: List | Dict 违禁区域经纬度 不为None的话开启
        coordinates: Dict 实时经纬度 {"latitude": float, "longitude": float}
    Returns:
        draw_info: 绘制信息列表，元素格式：
            {"type": "detection", "bbox": [x1,y1,x2,y2],
             "color": (B,G,R), "label": str, "track_id": int}
        need_draw: bool，本帧是否有新目标需要绘制
    """
    draw_info = []
    current_time = time.time()
    need_draw = False

    # 判断是否需要进行地理区域检测
    should_check_geo_area = area is not None and coordinates is not None
    geo_violation_detected = False
    violated_areas = []
    # print(f"should_check_geo_area:{should_check_geo_area}, area:{area}, coordinates: {coordinates}"),
    # 如果需要进行地理区域检测，先检查是否在违禁区域内
    if should_check_geo_area:
        try:
            # 解析area信息（如果是字符串的话）
            if isinstance(area, str):
                area_list = json.loads(area)
            else:
                area_list = area

            # 使用GeoAreaChecker进行地理区域检测
            geo_checker = GeoAreaChecker(mqtt_client=None)
            latitude = coordinates.get('latitude')
            longitude = coordinates.get('longitude')

            if latitude is not None and longitude is not None:
                geo_violation_detected, violated_areas = geo_checker.check_position_in_area(
                    latitude, longitude, area_list
                )

                if geo_violation_detected:
                    print(f"[normal_detections] Geo area violation detected in areas: {violated_areas}")
        except Exception as e:
            print(f"[normal_detections] Error checking geo area: {e}")
            geo_violation_detected = False

    # 处理每个检测结果
    for det in detections:
        # 获取 bbox
        bbox = det.get("bbox")
        if not bbox or not isinstance(bbox, dict):
            continue

        x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
        x2, y2 = bbox.get("x2", 0), bbox.get("y2", 0)
        box = [int(x1), int(y1), int(x2), int(y2)]
        center_point = (det.get("cxn", 0.5), det.get("cyn", 0.5))  # 双击AIM功能 必需参数

        track_id = det.get("track_id", -1)
        if track_id is None or track_id < 0:
            track_id = -1

        class_name = det.get("class_name", "")
        class_id = det.get("class_id", -1)

        # 初始化或更新跟踪记录
        if track_id not in tracking:
            tracking[track_id] = {
                "timestamp": current_time,
                "is_draw": False
            }
        else:
            tracking[track_id]["timestamp"] = current_time

        # 如果该 track_id 尚未绘制过，则添加到绘制列表
        # 注意：对无 track_id（-1）的检测，不进行 is_draw 过滤，
        # 确保同一张图片内多个无跟踪目标都能被绘制
        if track_id < 0 or not tracking[track_id]["is_draw"]:
            if track_id >= 0:
                tracking[track_id]["is_draw"] = True
            need_draw = True

            # 根据地理区域违禁状态选择颜色
            # 如果检测到地理区域违禁，使用红色；否则使用绿色
            if geo_violation_detected:
                color = (0, 0, 255)  # BGR: Red for violation
                label_suffix = ""
            else:
                color = (0, 255, 0)  # BGR: Green
                label_suffix = ""

            draw_info.append({
                "type": "detection",
                "bbox": box,
                "color": color,
                "label": class_name + label_suffix,
                "track_id": track_id,
                "class_id": class_id,
                "geo_violation": geo_violation_detected,
                "violated_areas": violated_areas,
                "center_point": center_point,
            })

    # 清理过期的跟踪记录
    for track_id in list(tracking.keys()):
        is_alive = any(det.get("track_id", -1) == track_id for det in detections)
        last_timestamp_time = tracking[track_id]["timestamp"]
        time_since_last_timestamp = current_time - last_timestamp_time

        if not is_alive and time_since_last_timestamp > 30:
            tracking.pop(track_id)

    # 如果检测到地理区域违禁，则需要绘制
    if geo_violation_detected and not draw_info:
        # 即使没有检测到物体，如果在违禁区域也需要标记
        need_draw = True

    return draw_info, need_draw


def normal_draw_detection_results(image, draw_info):
    """
    绘制检测结果

    Args:
        image: 输入图像 (BGR numpy array)
        draw_info: 来自 `normal_detections` 的绘制信息列表

    Returns:
        img_bgr: 绘制后的图像
        violation_class_name: 类别名称（如果有检测结果），否则为 "normal"
    """
    img_bgr = image.copy()
    violation_class_name = "normal"

    for item in draw_info:
        item_type = item.get("type")
        bbox = item.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, bbox)

        if item_type == "detection":
            color = item.get("color", (0, 0, 255))
            label = item.get("label", "")
            track_id = item.get("track_id", -1)
            geo_violation = item.get("geo_violation", False)
            violated_areas = item.get("violated_areas", [])
            center_point = item.get("center_point", (0.5, 0.5))
            # 绘制矩形框
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

            # 将类别名转换为中文
            chinese_label = CLASS_NAME_TO_CHINESE.get(label, None)
            # 绘制标签
            # label_text = chinese_label if track_id < 0 else f"{chinese_label} ID:{track_id}"
            # label_text = chinese_label if chinese_label != None else f"{chinese_label}"
            ft.putText(
                img_bgr,
                str(chinese_label),
                (x1, min(img_bgr.shape[0] - 20, y2 - 25)),  # org
                fontHeight=20,  # 字体高度，整型
                color=(0, 255, 0),
                thickness=-1,
                line_type=cv2.LINE_AA,
                bottomLeftOrigin=False  # 必须提供
            )

            # 如果有地理区域违禁信息，在图像上添加警告文本
            if geo_violation and violated_areas:
                warning_text = f"{violated_areas}"
                ft.putText(
                    img_bgr,
                    warning_text,
                    # chinese_label,
                    (x1, min(img_bgr.shape[0] - 20, y2 + 25)),  # org
                    fontHeight=20,  # 字体高度，整型
                    color=(0, 0, 255),
                    thickness=-1,
                    line_type=cv2.LINE_AA,
                    bottomLeftOrigin=False  # 必须提供
                )

            # 记录类别名称
            if label:
                violation_class_name = chinese_label or label
                # 如果有地理区域违禁，添加地理区域信息到类别名称
                if geo_violation and violated_areas:
                    violation_class_name = f"{chinese_label or label}_GEO_VIOLATION_{','.join(violated_areas)}"

    return img_bgr, violation_class_name  # , center_point


EXPORT = {
    "detect": normal_detections,
    "draw": normal_draw_detection_results,
}
