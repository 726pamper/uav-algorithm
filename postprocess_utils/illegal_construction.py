"""违规建筑 YOLO-detect 后处理（算法 87）。

同一目标不重复告警：通过空间位置 IoU 判断是否为同一目标，
已告警过的目标在消失前不再重复推送告警。

Note:
    可通过移除下方区域排除代码块来关闭本算法的区域过滤。
"""

import cv2
import time
from .chinese_name import CLASS_NAME_TO_CHINESE

ft = cv2.freetype.createFreeType2()
ft.loadFontData(
    fontFileName="/tensorrt/uav-algorithm-pipeline/postprocess_utils/fonts/wqy-zenhei.ttc",
    idx=0
)

ALGO_ID = "87"

TARGET_CLASSES = {"illegal_construction"}
CONSTRUCTION_COLOR = (0, 0, 255)

MIN_CONF = 0.25
MIN_AREA_RATIO = 0.0001
MAX_AREA_RATIO = 0.50
NMS_IOU = 0.50

DEDUP_IOU_THRESH = 0.20
DEDUP_EXPIRE_TIME = 60.0


def _is_target_class(class_name: str, class_id: int = -1) -> bool:
    if class_name in TARGET_CLASSES:
        return True
    return class_id == 0


def _parse_bbox(det) -> list[int] | None:
    bbox = det.get("bbox")
    if bbox and isinstance(bbox, dict):
        x1 = int(bbox.get("x1", 0))
        y1 = int(bbox.get("y1", 0))
        x2 = int(bbox.get("x2", 0))
        y2 = int(bbox.get("y2", 0))
    else:
        x1 = int(det.get("x1", 0))
        y1 = int(det.get("y1", 0))
        x2 = int(det.get("x2", 0))
        y2 = int(det.get("y2", 0))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def _is_duplicate_target(bbox, alarmed_boxes: list, iou_thresh: float = DEDUP_IOU_THRESH) -> bool:
    if not alarmed_boxes:
        return False
    for ab in alarmed_boxes:
        if _bbox_iou(bbox, ab) >= iou_thresh:
            return True
    return False


def _estimate_frame_size(detections) -> tuple[int, int]:
    max_x, max_y = 1920, 1080
    for det in detections:
        box = _parse_bbox(det)
        if not box:
            continue
        max_x = max(max_x, box[2])
        max_y = max(max_y, box[3])
    return max_x, max_y


def _filter_det_candidate(det, img_w: int, img_h: int) -> dict | None:
    class_name = det.get("class_name", "")
    class_id = det.get("class_id", -1)
    if not _is_target_class(class_name, class_id):
        return None

    conf = float(det.get("conf", 0.0) or 0.0)
    if conf < MIN_CONF:
        return None

    box = _parse_bbox(det)
    if not box:
        return None

    x1, y1, x2, y2 = box
    frame_area = max(img_w * img_h, 1)
    area_ratio = (x2 - x1) * (y2 - y1) / frame_area
    if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
        return None

    track_id = det.get("track_id", -1)
    if track_id is None or track_id < 0:
        track_id = -1

    return {
        "type": "detection",
        "bbox": box,
        "color": CONSTRUCTION_COLOR,
        "label": class_name or "illegal_construction",
        "track_id": track_id,
        "class_id": class_id,
        "conf": conf,
    }


def _nms_candidates(candidates: list[dict]) -> list[dict]:
    if len(candidates) <= 1:
        return candidates
    candidates = sorted(candidates, key=lambda c: c["conf"], reverse=True)
    kept = []
    for cand in candidates:
        if all(_bbox_iou(cand["bbox"], k["bbox"]) < NMS_IOU for k in kept):
            kept.append(cand)
    return kept


def illegal_construction_detections(detections, tracking, task_id=None, **kwargs):
    """违规建筑检测后处理（YOLO-detect）。

    流程：
    1. 按目标类别 + 置信度 + 面积比例过滤候选
    2. 帧内 NMS（防同一目标被重复框出）
    3. 跨帧 IoU 去重（已告警目标不重复上报）
    4. 过期告警记录清理
    """
    draw_info = []
    current_time = time.time()
    need_draw = False

    if "__total_count__" not in tracking:
        tracking["__total_count__"] = 0
    if "__alarmed_boxes__" not in tracking:
        tracking["__alarmed_boxes__"] = []

    if not isinstance(detections, list):
        return draw_info, need_draw

    img_w, img_h = _estimate_frame_size(detections)

    candidates = []
    for det in detections:
        cand = _filter_det_candidate(det, img_w, img_h)
        if cand:
            candidates.append(cand)
    candidates = _nms_candidates(candidates)

    tracking["__alarmed_boxes__"] = [
        ab for ab in tracking["__alarmed_boxes__"]
        if current_time - ab.get("timestamp", 0) < DEDUP_EXPIRE_TIME
    ]

    alarmed_boxes = [ab["bbox"] for ab in tracking["__alarmed_boxes__"]]
    new_alarmed = []

    for cand in candidates:
        track_id = cand["track_id"]
        bbox = cand["bbox"]

        if track_id not in tracking:
            tracking[track_id] = {"timestamp": current_time, "is_draw": False}
        else:
            tracking[track_id]["timestamp"] = current_time

        if _is_duplicate_target(bbox, alarmed_boxes):
            for ab in tracking["__alarmed_boxes__"]:
                if _bbox_iou(bbox, ab["bbox"]) >= DEDUP_IOU_THRESH:
                    ab["timestamp"] = current_time
                    break
            tracking[track_id]["is_draw"] = True
            continue

        should_draw = not tracking[track_id]["is_draw"]

        if should_draw:
            tracking[track_id]["is_draw"] = True
            new_alarmed.append({"bbox": bbox, "timestamp": current_time})
            need_draw = True
            draw_info.append(cand)

    tracking["__alarmed_boxes__"].extend(new_alarmed)

    for tid in list(tracking.keys()):
        if tid in ("__total_count__", "__frame_count__", "__alarmed_boxes__"):
            continue
        is_alive = any(
            (_filter_det_candidate(det, img_w, img_h) or {}).get("track_id", -1) == tid
            for det in detections
        )
        if not is_alive and current_time - tracking[tid].get("timestamp", 0) > 30:
            tracking.pop(tid)

    return draw_info, need_draw


def illegal_construction_draw_results(image, draw_info):
    """绘制违规建筑检测框。"""
    img_bgr = image.copy()
    violation_class_name = "normal"

    count = sum(1 for item in draw_info if item.get("label") in TARGET_CLASSES)

    for item in draw_info:
        if item.get("type") != "detection":
            continue

        bbox = item.get("bbox") or []
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        color = item.get("color", CONSTRUCTION_COLOR)
        label = item.get("label", "illegal_construction")
        track_id = item.get("track_id", -1)
        conf = item.get("conf")
        label_cn = CLASS_NAME_TO_CHINESE.get(label, "违规建筑")

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

        if conf is not None:
            label_text = f"{label_cn} {conf:.2f}"
        elif track_id >= 0:
            label_text = f"{label_cn} ID:{track_id}"
        else:
            label_text = label_cn

        ft.putText(img_bgr, label_text, (x1, max(0, y1 - 10)),
                   font_height=20, color=color)
        violation_class_name = label_cn

    if count > 0:
        ft.putText(img_bgr, f"违规建筑: {count}", (20, 50),
                   font_height=30, color=(0, 0, 255))

    return img_bgr, violation_class_name


EXPORT = {
    "detect": illegal_construction_detections,
    "draw": illegal_construction_draw_results,
}
