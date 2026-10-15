import cv2
import time
import numpy as np
from .chinese_name import CLASS_NAME_TO_CHINESE

# ft = cv2.freetype.createFreeType2()
# ft.loadFontData(
#     fontFileName="/tensorrt/uav-algorithm-pipeline/postprocess_utils/fonts/wqy-zenhei.ttc",
#     idx=0
# )

ALGO_ID = "segment"

# 全局共享字典：存储segment结果，以task_id为key
# 格式：{task_id: {"results": [...], "timestamp": float}}
_SEGMENT_RESULTS_CACHE = {}


def normal_seg_detections(detections, tracking, task_id=None, **kwargs):
    """
    普通分割结果处理，相同 track_id 只绘制一次

    注意：detections列表中同时包含segment结果和detect结果
    本函数只处理有xy字段的项（segment结果）

    Args:
        detections: JSON格式的检测结果列表，每个元素可能包含：
            - segment结果: {"xy": list, "bbox": {...}, "x1": float, ...}
            - detect结果: {"bbox": {...}, "x1": float, ...} (没有xy字段)
        tracking: 跟踪状态字典:
            {track_id: {"timestamp": float, "is_draw": bool}}
        task_id: 任务ID（可选），用于存储segment结果供gs_violation使用
        **kwargs: 其他参数（可能包含task_id）

    Returns:
        draw_info: 绘制信息列表，元素格式：
            {"type": "segmentation", "bbox": [x1,y1,x2,y2], "mask": np.ndarray,
             "color": (B,G,R), "label": str, "track_id": int, "class_id": int}
        need_draw: bool，本帧是否有新目标需要绘制
    """
    # 从kwargs中获取task_id（如果未直接提供）
    if task_id is None:
        task_id = kwargs.get('task_id')

    # 在tracking中存储task_id（供后续使用）
    if task_id:
        tracking["_task_id"] = str(task_id)

    draw_info = []
    current_time = time.time()
    need_draw = False

    # 确保detections是列表类型
    if not isinstance(detections, list):
        print(f"[normal_seg] Warning: detections is not a list, type={type(detections)}")
        return draw_info, need_draw

    # 处理每个分割结果（只处理有xy字段的项）
    for det in detections:
        # 确保det是字典类型
        if not isinstance(det, dict):
            print(f"[normal_seg] Warning: det is not a dict, type={type(det)}, value={det}")
            continue

        # 只处理有xy字段的项（segment结果）
        if det.get("xy") is None:
            continue
        # 获取 mask 数据，支持两种格式：
        # 1. deepstream_trt_inference.py 中的 "xy" 字段（点列表）
        # 2. 原始的 "mask" 字段（直接掩码）
        xy_points = det.get("xy")
        mask = det.get("mask")

        mask_array = None

        # 处理 xy 点列表格式
        if xy_points is not None:
            # 确保xy_points是列表类型
            if not isinstance(xy_points, list):
                print(f"[normal_seg] Warning: xy_points is not a list, type={type(xy_points)}, value={xy_points}")
                xy_points = None

            if xy_points is not None and len(xy_points) > 0:
                # 获取 bbox（优先从bbox字典获取，其次从顶层字段获取）
                bbox_dict = det.get("bbox")
                if bbox_dict and isinstance(bbox_dict, dict):
                    x1 = float(bbox_dict.get("x1", 0))
                    y1 = float(bbox_dict.get("y1", 0))
                    x2 = float(bbox_dict.get("x2", 0))
                    y2 = float(bbox_dict.get("y2", 0))
                else:
                    x1 = float(det.get("x1", 0))
                    y1 = float(det.get("y1", 0))
                    x2 = float(det.get("x2", 0))
                    y2 = float(det.get("y2", 0))

                # 计算掩码大小（相对于bbox）
                width = int(x2 - x1)
                height = int(y2 - y1)

                if width > 0 and height > 0:
                    try:
                        # 创建空掩码（相对于bbox的大小）
                        mask_array = np.zeros((height, width), dtype=np.uint8)

                        # 将点列表转换为 numpy 数组
                        points = np.array(xy_points, dtype=np.float32)

                        # 关键修复：将绝对坐标转换为相对于bbox左上角的相对坐标
                        # xy_points 是绝对坐标（相对于整个图像），需要减去 bbox 的左上角坐标
                        if points.ndim == 2 and points.shape[1] == 2:
                            points_relative = points.copy()
                            points_relative[:, 0] = points[:, 0] - x1  # x坐标减去x1
                            points_relative[:, 1] = points[:, 1] - y1  # y坐标减去y1
                            # 转换为整数坐标
                            points_relative = points_relative.astype(np.int32)

                            # 确保坐标在有效范围内
                            points_relative[:, 0] = np.clip(points_relative[:, 0], 0, width - 1)
                            points_relative[:, 1] = np.clip(points_relative[:, 1], 0, height - 1)

                            # 填充多边形区域
                            cv2.fillPoly(mask_array, [points_relative], 255)
                        else:
                            # 如果格式不对，跳过
                            print(
                                f"[normal_seg] Warning: points format invalid, ndim={points.ndim}, shape={points.shape}")
                            mask_array = None
                    except Exception as e:
                        print(f"[normal_seg] Error processing xy_points: {e}")
                        import traceback
                        traceback.print_exc()
                        mask_array = None

        # 处理直接掩码格式
        elif mask is not None:
            if isinstance(mask, list):
                mask_array = np.array(mask, dtype=np.uint8)
            else:
                mask_array = np.array(mask, dtype=np.uint8)

        # 检查mask是否有效
        if mask_array is None or mask_array.size == 0:
            continue

        # 获取 bbox
        bbox = det.get("bbox")
        if bbox and isinstance(bbox, dict):
            x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
            x2, y2 = bbox.get("x2", 0), bbox.get("y2", 0)
            box = [int(x1), int(y1), int(x2), int(y2)]
        else:
            # 从 mask 计算 bbox
            try:
                rows, cols = np.where(mask_array > 0)
                if len(rows) == 0 or len(cols) == 0:
                    continue
                x1, y1 = int(cols.min()), int(rows.min())
                x2, y2 = int(cols.max()), int(rows.max())
                box = [x1, y1, x2, y2]
            except Exception:
                continue

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
        # 注意：对无 track_id（-1）的检测，不进行 is_draw 过滤
        if track_id < 0 or not tracking[track_id]["is_draw"]:
            if track_id >= 0:
                tracking[track_id]["is_draw"] = True
            need_draw = True

            # 默认颜色为绿色
            color = (0, 255, 0)

            draw_info.append({
                "type": "segmentation",
                "bbox": box,
                "mask": mask_array,
                "color": color,
                "label": class_name,
                "track_id": track_id,
                "class_id": class_id
            })

    # 将segment结果存储到全局缓存和tracking字典中，供gs_violation使用
    if draw_info:
        # 存储到tracking字典（向后兼容）
        tracking["_segment_results"] = {
            "results": draw_info,
            "timestamp": current_time
        }

        # 存储到全局缓存（以task_id为key，供gs_violation跨算法编号访问）
        if task_id:
            global _SEGMENT_RESULTS_CACHE
            _SEGMENT_RESULTS_CACHE[str(task_id)] = {
                "results": draw_info,
                "timestamp": current_time
            }

            # 清理过期的缓存（30秒过期）
            expired_keys = []
            for tid, data in _SEGMENT_RESULTS_CACHE.items():
                if current_time - data.get("timestamp", 0) > 30:
                    expired_keys.append(tid)
            for tid in expired_keys:
                _SEGMENT_RESULTS_CACHE.pop(tid, None)

    # 清理过期的跟踪记录
    for track_id in list(tracking.keys()):
        # 跳过特殊key
        if track_id == "_segment_results":
            # 清理过期的segment结果（30秒过期）
            seg_timestamp = tracking[track_id].get("timestamp", 0)
            if current_time - seg_timestamp > 30:
                tracking.pop(track_id)
            continue

        is_alive = any(det.get("track_id", -1) == track_id for det in detections)
        last_timestamp_time = tracking[track_id]["timestamp"]
        time_since_last_timestamp = current_time - last_timestamp_time

        if not is_alive and time_since_last_timestamp > 30:
            tracking.pop(track_id)

    return draw_info, need_draw


def normal_draw_seg_results(image, draw_info):
    """
    绘制分割结果

    Args:
        image: 输入图像 (BGR numpy array)
        draw_info: 来自 `normal_seg_detections` 的绘制信息列表

    Returns:
        img_bgr: 绘制后的图像
        violation_class_name: 类别名称（如果有分割结果），否则为 "normal"
    """
    img_bgr = image.copy()
    violation_class_name = "normal"

    for item in draw_info:
        item_type = item.get("type")
        if item_type != "segmentation":
            continue

        mask = item.get("mask")
        bbox = item.get("bbox") or []
        color = item.get("color", (0, 255, 0))
        label = item.get("label", "")
        label_cn = CLASS_NAME_TO_CHINESE.get(label, label)
        track_id = item.get("track_id", -1)

        if mask is not None and mask.size > 0:
            # 绘制分割掩码
            # mask 是相对于 bbox 的，需要放置到全图的正确位置
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                mask_h, mask_w = mask.shape[:2]
                bbox_h, bbox_w = y2 - y1, x2 - x1

                # 如果mask大小与bbox大小匹配，直接放置
                if mask_h == bbox_h and mask_w == bbox_w:
                    # 创建全图大小的mask
                    full_mask = np.zeros((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.uint8)
                    full_mask[y1:y2, x1:x2] = mask

                    # 创建彩色掩码
                    colored_mask = np.zeros_like(img_bgr)
                    colored_mask[full_mask > 0] = color
                    img_bgr = cv2.addWeighted(img_bgr, 0.7, colored_mask, 0.3, 0)
                else:
                    # mask大小不匹配，尝试resize
                    try:
                        resized_mask = cv2.resize(mask, (bbox_w, bbox_h), interpolation=cv2.INTER_NEAREST)
                        full_mask = np.zeros((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.uint8)
                        full_mask[y1:y2, x1:x2] = resized_mask

                        colored_mask = np.zeros_like(img_bgr)
                        colored_mask[full_mask > 0] = color
                        img_bgr = cv2.addWeighted(img_bgr, 0.7, colored_mask, 0.3, 0)
                    except Exception:
                        # 如果resize失败，至少绘制bbox
                        pass
            else:
                # 如果没有bbox信息，尝试直接使用mask（假设是全图大小）
                try:
                    if mask.shape[:2] == (img_bgr.shape[0], img_bgr.shape[1]):
                        colored_mask = np.zeros_like(img_bgr)
                        colored_mask[mask > 0] = color
                        img_bgr = cv2.addWeighted(img_bgr, 0.7, colored_mask, 0.3, 0)
                except Exception:
                    pass

        if len(bbox) == 4:
            # 绘制 bbox（如果有）
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

            # 绘制标签（使用中文 FreeType 字体）
            label_text = label_cn if track_id < 0 else f"{label_cn}"
            ft.putText(
                img_bgr,
                str(label_text),
                (x1, max(0, y1 + 20)),
                fontHeight=5,
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
    "detect": normal_seg_detections,
    "draw": normal_draw_seg_results,
}