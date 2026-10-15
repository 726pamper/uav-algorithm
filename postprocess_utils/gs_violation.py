"""
高速应急车道违停检测模块
基于分割区域进行车辆占用检测
"""
import cv2
import numpy as np
import time
from .chinese_name import CLASS_NAME_TO_CHINESE

ft = cv2.freetype.createFreeType2()
ft.loadFontData(
    fontFileName="/tensorrt/uav-algorithm-pipeline/postprocess_utils/fonts/wqy-zenhei.ttc",
    idx=0
)

ALGO_ID = "75"

def gs_violation_detections(
    dets,
    tracking,
    seg_results=None,
    ts_ms=0,
    task_id=None,
    **kwargs
):
    """处理高速应急车道占用检测结果。"""
    ts_s = ts_ms / 1000.0
    draw_info = []
    need_draw = False

    lane_contours = []
    car_detections = []

    for det in dets:
        if not isinstance(det, dict):
            continue

        model_code = det.get("model_code")
        if model_code == "75" and det.get("xy") is not None:
            xy_coords = det["xy"]
            if isinstance(xy_coords, list):
                contour = np.array(xy_coords, dtype=np.float32)
                if contour.ndim == 2 and contour.shape[1] == 2:
                    contour = contour.reshape(-1, 1, 2)
                lane_contours.append(contour)
        elif model_code == "79" and det.get("bbox") is not None:
            class_name = det.get("class_name", "")
            class_id = int(det.get("class_id", -1))
            if class_name == "car" or class_id == 0 or class_name == "":
                car_detections.append(det)
    try:
        for _i, _det in enumerate(dets):
            if isinstance(_det, dict):
                _xy = _det.get("xy")
                print(f"[GS-DEBUG] det[{_i}] model_code={_det.get('model_code')} task={_det.get('task')} "
                      f"has_xy={_xy is not None} xy_pts={len(_xy) if isinstance(_xy, list) else 0} "
                      f"has_bbox={_det.get('bbox') is not None} class_name={_det.get('class_name')} "
                      f"class_id={_det.get('class_id')} track_id={_det.get('track_id')} conf={_det.get('conf')}")
        print(f"[GS-DEBUG] 汇总 lane_contours={len(lane_contours)} car_detections={len(car_detections)}")
    except Exception as _e:
        print(f"[GS-DEBUG] 打印异常: {_e}")
        
    if not lane_contours or not car_detections:
        return [], False

    current_time = time.time()

    for det in car_detections:
        track_id = det.get("track_id")
        if track_id is None:
            continue

        bbox = det.get("bbox", {})
        cx = (bbox.get("x1", 0) + bbox.get("x2", 0)) / 2.0
        cy = (bbox.get("y1", 0) + bbox.get("y2", 0)) / 2.0
        center_point = (cx, cy)

        is_in_lane = False
        for contour in lane_contours:
            if len(contour) > 0 and cv2.pointPolygonTest(contour, center_point, False) >= 0:
                is_in_lane = True
                break

        if is_in_lane:
            state = tracking.setdefault(track_id, {
                "violation_reported": False,
                "last_violation_time": 0,
            })
            last_report_time = state["last_violation_time"]
            time_since_last_report = ts_s - last_report_time
            if (not state["violation_reported"]) or time_since_last_report > 30:
                state["violation_reported"] = True
                state["last_violation_time"] = ts_s
                need_draw = True
                draw_info.append({
                    "type": "car",
                    "bbox": bbox,
                    "track_id": int(track_id),
                    "is_in_lane": True,
                    "center_point": center_point,
                    "label": "car",
                    "editable": True,
                    "box_role": "target",
                    "is_violation": True,
                })

    for det in car_detections:
        track_id = det.get("track_id")
        if track_id is None:
            continue
        if any(info["track_id"] == track_id for info in draw_info):
            continue

        bbox = det.get("bbox", {})
        center_point = (
            (bbox.get("x1", 0) + bbox.get("x2", 0)) / 2.0,
            (bbox.get("y1", 0) + bbox.get("y2", 0)) / 2.0,
        )
        draw_info.append({
            "type": "car",
            "bbox": bbox,
            "track_id": int(track_id),
            "is_in_lane": False,
            "center_point": center_point,
            "label": "car",
            "editable": True,
            "box_role": "target",
            "is_violation": False,
        })

    if not need_draw:
        return [], False

    return draw_info, True

def gs_violation_draw_detection_results(
    frame_bgr, draw_info
):
    """
    在图像上绘制高速应急车道占用检测结果
    分割车道内的车辆绘制为红色，分割车道外的车辆绘制为绿色
    仅绘制车辆
    """
    img = frame_bgr.copy()

    for i, info in enumerate(draw_info):
        bbox = info.get("bbox", {})  # 修复：确保使用正确的键名 'bbox'
        is_in_lane = info.get("is_in_lane", False)
        track_id = info.get("track_id", -1)

        x1 = int(bbox.get("x1", 0))
        y1 = int(bbox.get("y1", 0))
        x2 = int(bbox.get("x2", 0))
        y2 = int(bbox.get("y2", 0))

        # 根据是否在应急车道内选择颜色和标签
        if is_in_lane:
            color = (0, 0, 255)  # 红色 - 在应急车道内
            label = CLASS_NAME_TO_CHINESE.get('illegal_parking_emergency', '应急车道违停')
        else:
            color = (0, 255, 0)  # 绿色 - 在车道外
            label = CLASS_NAME_TO_CHINESE.get('car', '汽车')

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        ft.putText(
            img,
            str(label),
            (x1, max(0, y1 + 20)),
            fontHeight=15,
            color=color,
            thickness=-1,
            line_type=cv2.LINE_AA,
            bottomLeftOrigin=False,
        )

    return img, "illegal_parking_emergency"


EXPORT = {
    'detect': gs_violation_detections,
    'draw': gs_violation_draw_detection_results,
}
