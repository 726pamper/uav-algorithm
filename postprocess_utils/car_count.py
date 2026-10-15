import cv2
import numpy as np
import time
from .chinese_name import CLASS_NAME_TO_CHINESE

ft = cv2.freetype.createFreeType2()
ft.loadFontData(
    fontFileName="/tensorrt/uav-algorithm-pipeline/postprocess_utils/fonts/wqy-zenhei.ttc",
    idx=0
)

ALGO_ID = "80"


def normal_detections(detections, tracking):
    """图片模式：统计本张照片内的车辆数（快照口径）。

    与旧版（视频流过线计数）的差异：
        - 不再依赖 track_id 跨帧轨迹与过线判定；
        - 本张照片内每个 car 检出都进入绘制列表，车辆数 = car 检出条数；
        - tracking 参数仅为兼容注册表调用签名保留，内部不再使用。
    """
    draw_info = []
    need_draw = False

    for det in detections:
        bbox = det.get("bbox")
        if not bbox or not isinstance(bbox, dict):
            continue

        x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
        x2, y2 = bbox.get("x2", 0), bbox.get("y2", 0)
        box = [int(x1), int(y1), int(x2), int(y2)]

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        center_point = (cx, cy)

        track_id = det.get("track_id", -1)
        if track_id is None or track_id < 0:
            track_id = -1

        class_name = det.get("class_name", "")
        class_id = det.get("class_id", -1)

        # 本张照片内每辆车都计入（同一照片内目标互不相同，无需去重）
        if class_name == "car":
            need_draw = True
            draw_info.append({
                "type": "detection",
                "bbox": box,
                "color": (0, 255, 0),
                "label": class_name,
                "track_id": track_id,
                "class_id": class_id,
                "center_point": center_point,
            })

    return draw_info, need_draw


def normal_draw_detection_results(image, draw_info, tracking_state):
    """绘制本张照片的车辆框，并在左上角标注本张照片车辆数。"""
    img_bgr = image.copy()
    violation_class_name = "normal"

    # 本张照片车辆数 = 绘制列表中 car 的条数
    car_count = sum(1 for item in draw_info if item.get("label") == "car")

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

            # 绘制矩形框
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

            # 将类别名转换为中文
            chinese_label = CLASS_NAME_TO_CHINESE.get(label, None)

            # 绘制标签
            label_text = chinese_label if track_id < 0 else f"{chinese_label} ID:{track_id}"

            ft.putText(
                img_bgr,
                str(chinese_label),
                (x1, min(img_bgr.shape[0] - 20, y2 - 25)),
                fontHeight=20,
                color=(0, 255, 0),
                thickness=-1,
                line_type=cv2.LINE_AA,
                bottomLeftOrigin=False
            )

    # 在画面左上角绘制本张照片车辆数
    count_text = f"车辆数: {car_count}"
    ft.putText(
        img_bgr,
        count_text,
        (20, 50),
        fontHeight=30,
        color=(255, 0, 0),
        thickness=-1,
        line_type=cv2.LINE_AA,
        bottomLeftOrigin=False,
    )

    return img_bgr, violation_class_name


EXPORT = {
    "detect": normal_detections,
    "draw": normal_draw_detection_results,
}