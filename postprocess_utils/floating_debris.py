"""水面漂浮物 YOLO-detect 后处理（算法 86）。

同一目标不重复告警：通过空间位置 IoU 判断是否为同一目标，
已告警过的目标在消失前不再重复推送告警。
"""

import cv2
import time
from .chinese_name import CLASS_NAME_TO_CHINESE

ft = cv2.freetype.createFreeType2()
ft.loadFontData(
    fontFileName="/tensorrt/uav-algorithm-pipeline/postprocess_utils/fonts/wqy-zenhei.ttc",
    idx=0
)

ALGO_ID = "86"

TARGET_CLASSES = {"floating_debris", "float"}
DEBRIS_COLOR = (255, 140, 0)  # BGR: 水面蓝青色，区别于违规建筑的纯红色

MIN_CONF = 0.15
MIN_AREA_RATIO = 0.00005   # 极小漂浮物也要检测
MAX_AREA_RATIO = 0.60
NMS_IOU = 0.50

DEDUP_IOU_THRESH = 0.20
DEDUP_EXPIRE_TIME = 30.0   # 30 秒未出现视为消失，允许重新告警


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
    """检查当前 bbox 是否与已告警目标重叠（防同目标重复告警）。"""
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
        "color": DEBRIS_COLOR,
        "label": class_name or "floating_debris",
        "track_id": track_id,
        "class_id": class_id,
        "conf": conf,
    }


def _nms_candidates(candidates: list[dict]) -> list[dict]:
    """对候选框做 NMS，按 conf 降序保留。"""
    if len(candidates) <= 1:
        return candidates
    candidates = sorted(candidates, key=lambda c: c["conf"], reverse=True)
    kept = []
    for cand in candidates:
        if all(_bbox_iou(cand["bbox"], k["bbox"]) < NMS_IOU for k in kept):
            kept.append(cand)
    return kept


def floating_debris_detections(detections, tracking, task_id=None, **kwargs):
    """水面漂浮物检测后处理（YOLO-detect）。

    流程：
    1. 按目标类别 + 置信度 + 面积比例过滤候选
    2. 帧内 NMS（防同一目标被重复框出）
    3. 跨帧 IoU 去重（已告警目标不重复上报）
    4. 过期告警记录清理

    Note:
        本算法不做区域排除。
    """
    draw_info = []
    current_time = time.time()
    need_draw = False

    # 初始化 tracking 中的内部元数据
    if "__total_count__" not in tracking:
        tracking["__total_count__"] = 0
    if "__alarmed_boxes__" not in tracking:
        tracking["__alarmed_boxes__"] = []

    if not isinstance(detections, list):
        return draw_info, need_draw

    img_w, img_h = _estimate_frame_size(detections)

    # ---- 1. 过滤候选 ----
    candidates = []
    for det in detections:
        cand = _filter_det_candidate(det, img_w, img_h)
        if cand:
            candidates.append(cand)

    # ---- 2. 帧内 NMS ----
    candidates = _nms_candidates(candidates)

    # ---- 3. 清理过期告警记录 ----
    tracking["__alarmed_boxes__"] = [
        ab for ab in tracking["__alarmed_boxes__"]
        if current_time - ab.get("timestamp", 0) < DEDUP_EXPIRE_TIME
    ]

    # ---- 4. 跨帧去重 + 生成 draw_info ----
    alarmed_boxes = [ab["bbox"] for ab in tracking["__alarmed_boxes__"]]
    new_alarmed = []

    for cand in candidates:
        track_id = cand["track_id"]
        bbox = cand["bbox"]

        # 初始化或更新 per-track_id 记录
        if track_id not in tracking:
            tracking[track_id] = {"timestamp": current_time, "is_draw": False}
        else:
            tracking[track_id]["timestamp"] = current_time

        # ---- 跨帧 IoU 去重 ----
        if _is_duplicate_target(bbox, alarmed_boxes):
            # 刷新对应 alarmed_box 的时间戳
            for ab in tracking["__alarmed_boxes__"]:
                if _bbox_iou(bbox, ab["bbox"]) >= DEDUP_IOU_THRESH:
                    ab["timestamp"] = current_time
                    break
            # 更新 track_id 的 is_draw 状态
            tracking[track_id]["is_draw"] = True
            continue

        # 新目标：判断是否需要绘制
        should_draw = not tracking[track_id]["is_draw"]

        if should_draw:
            tracking[track_id]["is_draw"] = True
            new_alarmed.append({"bbox": bbox, "timestamp": current_time})
            need_draw = True
            draw_info.append(cand)

    # 将本帧新告警的目标加入全局告警列表
    tracking["__alarmed_boxes__"].extend(new_alarmed)

    # ---- 5. 清理长期消失的 track_id ----
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


def floating_debris_draw_results(image, draw_info):
    """绘制水面漂浮物检测框。"""
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
        color = item.get("color", DEBRIS_COLOR)
        label = item.get("label", "floating_debris")
        track_id = item.get("track_id", -1)
        conf = item.get("conf")
        label_cn = CLASS_NAME_TO_CHINESE.get(label, "漂浮物")

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

        if conf is not None:
            label_text = f"{label_cn} {conf:.2f}"
        elif track_id >= 0:
            label_text = f"{label_cn} ID:{track_id}"
        else:
            label_text = label_cn

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
        violation_class_name = label_cn

    if count > 0:
        ft.putText(
            img_bgr,
            f"水面漂浮物: {count}",
            (20, 50),
            fontHeight=30,
            color=(255, 140, 0),
            thickness=-1,
            line_type=cv2.LINE_AA,
            bottomLeftOrigin=False,
        )

    return img_bgr, violation_class_name


EXPORT = {
    "detect": floating_debris_detections,
    "draw": floating_debris_draw_results,
}
