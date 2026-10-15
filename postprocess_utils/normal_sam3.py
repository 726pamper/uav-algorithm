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

ALGO_ID = 'SAM3'


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
    need_draw = False

    # 处理每个检测结果
    for det in detections:
        # 获取 bbox
        bbox = det.get("bbox")
        if not bbox or not isinstance(bbox, dict):
            continue

        x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
        x2, y2 = bbox.get("x2", 0), bbox.get("y2", 0)
        box = [int(x1), int(y1), int(x2), int(y2)]

        class_name = det.get("class_name", "")
        class_id = det.get("class_id", -1)

        color = (0, 255, 0)  # BGR: Green

        need_draw = True
        draw_info.append({
            "type": "sam3",
            "bbox": box,
            "color": color,
            "label": class_name,
            "class_id": class_id,
        })

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

        if item_type == "sam3":
            color = item.get("color", (0, 0, 255))
            label = item.get("label", "")

            # 绘制矩形框
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

            # 将类别名转换为中文
            chinese_label = CLASS_NAME_TO_CHINESE.get(label, None)

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

            # 记录类别名称
            if label:
                violation_class_name = chinese_label

    return img_bgr, violation_class_name,  # center_point


EXPORT = {
    "detect": normal_detections,
    "draw": normal_draw_detection_results,
}
