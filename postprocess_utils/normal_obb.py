import cv2
import time
import numpy as np
from .chinese_name import CLASS_NAME_TO_CHINESE

ft = cv2.freetype.createFreeType2()
ft.loadFontData(
    fontFileName="/tensorrt/uav-algorithm-pipeline/postprocess_utils/fonts/wqy-zenhei.ttc",
    idx=0
)

ALGO_ID = "obb"


def normal_obb_detections(detections, tracking):
    """
    普通OBB检测结果处理，相同 track_id 只绘制一次

    Args:
        detections: JSON格式的检测结果列表，每个元素包含OBB检测信息:
            {"polygon": [{"x": float, "y": float}, ...],
             "cx": float, "cy": float, "w": float, "h": float, "angle": float,
             "conf": float, "class_id": int, "class_name": str,
             "track_id": int (可选),
             "bbox": {"x1": float, "y1": float, "x2": float, "y2": float}}
        tracking: 跟踪状态字典:
            {track_id: {"timestamp": float, "is_draw": bool}}

    Returns:
        draw_info: 绘制信息列表，元素格式：
            {"type": "obb", "polygon": np.array, "bbox": [x1,y1,x2,y2],
             "color": (B,G,R), "label": str, "track_id": int}
        need_draw: bool，本帧是否有新目标需要绘制
    """
    draw_info = []
    current_time = time.time()
    need_draw = False

    # 处理每个检测结果
    for det in detections:
        # 获取 polygon（OBB 四个顶点）
        polygon = det.get("polygon")
        if polygon is None:
            continue

        # 转换 polygon 为 numpy 数组
        if isinstance(polygon, list):
            if len(polygon) > 0 and isinstance(polygon[0], dict):
                # 格式: [{"x": float, "y": float}, ...]
                polygon_array = np.array([[pt.get("x", 0), pt.get("y", 0)] for pt in polygon], dtype=np.float32)
            else:
                # 格式: [[x, y], ...]
                polygon_array = np.array(polygon, dtype=np.float32)
        else:
            polygon_array = np.array(polygon, dtype=np.float32)

        if polygon_array.size == 0:
            continue

        # 获取 bbox（用于标签位置）
        bbox = det.get("bbox")
        if bbox and isinstance(bbox, dict):
            x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
            x2, y2 = bbox.get("x2", 0), bbox.get("y2", 0)
            box = [int(x1), int(y1), int(x2), int(y2)]
        else:
            # 从 polygon 计算 AABB
            xs = polygon_array[:, 0]
            ys = polygon_array[:, 1]
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())
            box = [x1, y1, x2, y2]

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

            # 默认颜色为绿色
            color = (0, 255, 0)

            draw_info.append({
                "type": "obb",
                "polygon": polygon_array,
                "bbox": box,
                "color": color,
                "label": class_name,
                "track_id": track_id,
                "class_id": class_id
            })

    # 清理过期的跟踪记录
    for track_id in list(tracking.keys()):
        is_alive = any(det.get("track_id", -1) == track_id for det in detections)
        last_timestamp_time = tracking[track_id]["timestamp"]
        time_since_last_timestamp = current_time - last_timestamp_time

        if not is_alive and time_since_last_timestamp > 30:
            tracking.pop(track_id)

    return draw_info, need_draw


def normal_draw_obb_results(image, draw_info):
    """
    绘制OBB检测结果

    Args:
        image: 输入图像 (BGR numpy array)
        draw_info: 来自 `normal_obb_detections` 的绘制信息列表

    Returns:
        img_bgr: 绘制后的图像
        violation_class_name: 类别名称（如果有检测结果），否则为 "normal"
    """
    img_bgr = image.copy()
    violation_class_name = "normal"

    for item in draw_info:
        item_type = item.get("type")
        if item_type != "obb":
            continue

        polygon = item.get("polygon")
        bbox = item.get("bbox") or []
        color = item.get("color", (0, 0, 255))
        label = item.get("label", "")
        label_cn = CLASS_NAME_TO_CHINESE.get(label, label)
        track_id = item.get("track_id", -1)

        if polygon is not None and polygon.size > 0:
            # 绘制旋转框多边形
            pts = polygon.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_bgr, [pts], isClosed=True, color=color, thickness=2)

            # 计算标签位置（使用 bbox 或从 polygon 计算）
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
            else:
                # 从 polygon 计算 AABB
                xs = pts[:, 0, 0]
                ys = pts[:, 0, 1]
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()), int(ys.max())
        elif len(bbox) == 4:
            # 如果没有 polygon，使用 bbox 绘制矩形
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        else:
            continue

        # 绘制标签（使用中文 FreeType 字体）
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

        # 记录类别名称
        if label:
            violation_class_name = label

    return img_bgr, violation_class_name


EXPORT = {
    "detect": normal_obb_detections,
    "draw": normal_draw_obb_results,
}