import cv2
import time
from .chinese_name import CLASS_NAME_TO_CHINESE

ft = cv2.freetype.createFreeType2()
ft.loadFontData(
    fontFileName="/tensorrt/uav-algorithm-pipeline/postprocess_utils/fonts/wqy-zenhei.ttc",
    idx=0
)

ALGO_ID = "93"

TARGET_CLASS = "person"
RISK_TEXT = "有溺水风险"
RISK_COLOR = (0, 0, 255)
TRACK_EXPIRE_TIME = 30


def get_box_from_detection(det):
    """从检测结果中解析 bbox。

    支持两种格式：
    1. {"bbox": {"x1": float, "y1": float, "x2": float, "y2": float}}
    2. {"x1": float, "y1": float, "x2": float, "y2": float}
    """
    bbox = det.get("bbox")
    if bbox and isinstance(bbox, dict):
        x1 = bbox.get("x1", 0)
        y1 = bbox.get("y1", 0)
        x2 = bbox.get("x2", 0)
        y2 = bbox.get("y2", 0)
    else:
        x1 = det.get("x1", 0)
        y1 = det.get("y1", 0)
        x2 = det.get("x2", 0)
        y2 = det.get("y2", 0)

    box = [int(x1), int(y1), int(x2), int(y2)]
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def get_center_from_box(box):
    """计算矩形框中心点。

    box: [x_min, y_min, x_max, y_max]
    """
    x_min, y_min, x_max, y_max = box
    return int((x_min + x_max) / 2), int((y_min + y_max) / 2)


def drowning_risk_detections(detections, tracking, threshold=0.5):
    """
    Args:
        detections: JSON 格式的检测结果列表，每个元素包含标准检测信息:
            {"bbox": {"x1": float, "y1": float, "x2": float, "y2": float},
             "conf": float, "class_id": int, "class_name": str,
             "track_id": int (可选)}
        tracking: 人员溺水风险跟踪状态字典:
            {track_id: {"seen": float, "alerted": bool}}
        threshold: person 检测置信度阈值，低于该阈值的目标不显示

    Returns:
        draw_info: 绘制信息列表，元素格式示例:
            {"type": "person", "bbox": [x1,y1,x2,y2],
             "center": (cx,cy), "color": (B,G,R),
             "label": "person", "text": "有溺水风险",
             "conf": float, "track_id": int}
        need_draw: bool，本帧是否检测到达到阈值的人员风险目标
    """
    draw_info = []
    current_time = time.time()
    need_draw = False

    if not isinstance(detections, list):
        return draw_info, need_draw

    alive_track_ids = set()

    for det in detections:
        class_name = det.get("class_name", "")
        if class_name != TARGET_CLASS:
            continue

        conf = float(det.get("conf", 0.0) or 0.0)
        if conf < threshold:
            continue

        box = get_box_from_detection(det)
        if box is None:
            continue

        track_id = det.get("track_id", -1)
        if track_id is None or track_id < 0:
            track_id = -1

        center = get_center_from_box(box)
        alive_track_ids.add(track_id)

        if track_id not in tracking:
            tracking[track_id] = {
                "seen": current_time,
                "alerted": False,
            }
        else:
            tracking[track_id]["seen"] = current_time

        if not tracking[track_id]["alerted"]:
            tracking[track_id]["alerted"] = True
            need_draw = True

        draw_info.append(
            {
                "type": TARGET_CLASS,
                "bbox": box,
                "center": center,
                "color": RISK_COLOR,
                "label": TARGET_CLASS,
                "text": RISK_TEXT,
                "conf": conf,
                "track_id": track_id,
            }
        )

    for track_id in list(tracking.keys()):
        last_seen_time = tracking[track_id].get("seen", 0)
        if track_id not in alive_track_ids and current_time - last_seen_time > TRACK_EXPIRE_TIME:
            tracking.pop(track_id)

    return draw_info, need_draw


def drowning_risk_draw_detection_results(image, draw_info):
    """
    Args:
        image: 输入图像 (BGR numpy array)
        draw_info: 来自 `drowning_risk_detections` 的绘制信息列表

    Returns:
        img_bgr: 绘制后的图像
        violation_class_name: 检测到风险返回 "person"，否则返回 "normal"
    """
    img_bgr = image.copy()
    has_risk = False

    for item in draw_info:
        item_type = item.get("type")
        if item_type != TARGET_CLASS:
            continue

        bbox = item.get("bbox") or []
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        color = item.get("color", RISK_COLOR)
        label = item.get("label", TARGET_CLASS)
        conf = item.get("conf")
        text = item.get("text", RISK_TEXT)

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

        if conf is not None:
            label_text = f"{text} {conf:.2f}"
        else:
            label_cn = CLASS_NAME_TO_CHINESE.get(label, label)
            label_text = f"{text} {label_cn}"

        ft.putText(
            img_bgr,
            str(label_text),
            (x1, max(0, y1 - 10)),
            fontHeight=20,
            color=color,
            thickness=-1,
            line_type=cv2.LINE_AA,
            bottomLeftOrigin=False,
        )
        has_risk = True

    violation_class_name = TARGET_CLASS if has_risk else "normal"
    return img_bgr, violation_class_name


EXPORT = {
    "detect": drowning_risk_detections,
    "draw": drowning_risk_draw_detection_results,
}
