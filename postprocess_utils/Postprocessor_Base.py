from typing import List, Dict, Tuple, Any, Optional
import time
import cv2
import numpy as np


class DetectionPostProcessor:
    """
    通用后处理基类。

    参数:
        min_conf: 最小置信度阈值，低于此值的检测会被过滤
        expire_time: 跟踪记录的过期时间（秒），超过此时间且未出现的 track 会被移除
        draw_once_per_track: 是否对同一 track_id 只绘制一次（True）
        class_map: 可选的 class_id -> class_name 映射
        palette: 可选的 class_id -> (B,G,R) 颜色映射；若未提供将用基于 id 的一致哈希颜色
    """

    def __init__(
            self,
            min_conf: float = 0.25,
            expire_time: float = 30.0,
            draw_once_per_track: bool = True,
            class_map: Optional[Dict[int, str]] = None,
            palette: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ) -> None:
        self.min_conf = float(min_conf)
        self.expire_time = float(expire_time)
        self.draw_once_per_track = draw_once_per_track
        self.class_map = class_map or {}
        self.palette = palette or {}

        # 跟踪状态字典: {track_id: {"timestamp": float, "is_draw": bool}}
        # 其中 track_id 为 int，未提供 track_id 的检测使用 -1
        self.tracking: Dict[int, Dict[str, Any]] = {}

    # ---------- 辅助函数 ----------
    @staticmethod
    def _clamp_bbox(x1: float, y1: float, x2: float, y2: float, img_wh: Optional[Tuple[int, int]] = None) -> List[int]:
        """把 bbox 转为 int 列表，并在有图像尺寸时裁剪到图像边界。"""
        x1_i = int(round(x1))
        y1_i = int(round(y1))
        x2_i = int(round(x2))
        y2_i = int(round(y2))
        if img_wh is not None:
            w, h = img_wh
            x1_i = max(0, min(x1_i, w - 1))
            x2_i = max(0, min(x2_i, w - 1))
            y1_i = max(0, min(y1_i, h - 1))
            y2_i = max(0, min(y2_i, h - 1))
        return [x1_i, y1_i, x2_i, y2_i]

    def _get_color(self, class_id: int, track_id: int) -> Tuple[int, int, int]:
        """返回与 class_id/track_id 一致的 (B,G,R) 颜色。优先使用 palette。"""
        if class_id in self.palette:
            return self.palette[class_id]
        key = track_id if track_id >= 0 else class_id
        # 基于哈希生成稳定颜色
        h = abs(hash(key))
        b = (h & 0xFF0000) >> 16
        g = (h & 0x00FF00) >> 8
        r = h & 0x0000FF
        return (int(b % 256), int(g % 256), int(r % 256))

    # ---------- 子类可覆盖的方法 ----------
    def convert_raw_detections(self, raw: Any) -> List[Dict[str, Any]]:
        """
        将模型原始输出转换为标准 detection 列表（字典列表）。
        子类应根据具体模型覆写此方法。

        默认实现假定 raw 已经是满足标准格式的列表，做简单过滤与校验。
        """
        detections: List[Dict[str, Any]] = []
        if not raw:
            return detections

        # 假定 raw 是列表，每个元素为 dict
        for det in raw:
            if not isinstance(det, dict):
                continue
            conf = float(det.get("conf", 0.0))
            if conf < self.min_conf:
                continue
            bbox = det.get("bbox")
            if not bbox or not isinstance(bbox, dict):
                continue
            class_id = int(det.get("class_id", -1))
            class_name = det.get("class_name") or self.class_map.get(class_id, str(class_id))
            track_id = det.get("track_id", -1)
            if track_id is None:
                track_id = -1
            detections.append({
                "bbox": {
                    "x1": float(bbox.get("x1", 0)),
                    "y1": float(bbox.get("y1", 0)),
                    "x2": float(bbox.get("x2", 0)),
                    "y2": float(bbox.get("y2", 0)),
                },
                "conf": conf,
                "class_id": class_id,
                "class_name": class_name,
                "track_id": int(track_id),
            })
        return detections

    # ---------- 主流程函数 ----------
    def process(self, raw_detections: Any, img_shape: Optional[Tuple[int, int]] = None) -> Tuple[
        List[Dict[str, Any]], bool]:
        """
        处理模型原始输出并返回绘制信息及是否需要绘制的标志。

        参数:
            raw_detections: 模型的原始输出（子类通过 convert_raw_detections 解析）
            img_shape: 可选，图像宽高 (w, h)，用于裁剪 bbox

        返回:
            draw_info: 元素格式与问题中一致的绘制信息列表
            need_draw: bool，本帧是否有新目标需要绘制
        """
        current_time = time.time()
        std_dets = self.convert_raw_detections(raw_detections)
        draw_info: List[Dict[str, Any]] = []
        need_draw = False

        # 更新 timestamp 时间并决定是否绘制
        for det in std_dets:
            bbox = det.get("bbox")
            if not bbox:
                continue
            x1 = bbox.get("x1", 0.0)
            y1 = bbox.get("y1", 0.0)
            x2 = bbox.get("x2", 0.0)
            y2 = bbox.get("y2", 0.0)

            box = self._clamp_bbox(x1, y1, x2, y2, img_shape)
            track_id = int(det.get("track_id", -1) or -1)
            class_id = int(det.get("class_id", -1) or -1)
            class_name = det.get("class_name", self.class_map.get(class_id, str(class_id)))

            if track_id not in self.tracking:
                self.tracking[track_id] = {"timestamp": current_time, "is_draw": False}
            else:
                self.tracking[track_id]["timestamp"] = current_time

            # 决定是否绘制：如果配置了 draw_once_per_track，则只为尚未绘制的 track 绘制一次
            if not self.draw_once_per_track or not self.tracking[track_id]["is_draw"]:
                self.tracking[track_id]["is_draw"] = True
                need_draw = True
                color = self._get_color(class_id, track_id)
                draw_info.append({
                    "type": "detection",
                    "bbox": box,
                    "color": color,
                    "label": class_name,
                    "track_id": track_id,
                    "class_id": class_id,
                })

        # 清理过期 track
        for t_id in list(self.tracking.keys()):
            last_timestamp = self.tracking[t_id].get("timestamp", 0.0)
            if (current_time - last_timestamp) > self.expire_time:
                del self.tracking[t_id]

        return draw_info, need_draw

    def draw(self, image: np.ndarray, draw_info: List[Dict[str, Any]]) -> Tuple[np.ndarray, str]:
        """
        在图像上绘制 draw_info 并返回绘制结果与主要类别名（第一个非空 label）。
        返回值与用户最初提供的 normal_draw_detection_results 接口保持一致。
        """
        img = image.copy()
        main_label = "normal"
        for item in draw_info:
            t = item.get("type")
            bbox = item.get("bbox") or []
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            if t == "detection":
                color = item.get("color", (0, 255, 0))
                label = item.get("label", "")
                track_id = item.get("track_id", -1)
                # rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label_text = label if track_id < 0 else f"{label} ID:{track_id}"
                cv2.putText(img, label_text, (x1, max(0, y1 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if label:
                    main_label = label
            elif t == "obb":
                polygon = item.get("polygon")
                bbox = item.get("bbox") or []
                color = item.get("color", (0, 255, 0))
                label = item.get("label", "")
                track_id = item.get("track_id", -1)

                if polygon is not None and polygon.size > 0:
                    # 绘制旋转框多边形
                    pts = polygon.astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

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
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                else:
                    continue

                # 绘制标签
                # label_text = label if track_id < 0 else f"{label} ID:{track_id}"
                label_text = label if track_id < 0 else f"{label}"
                cv2.putText(
                    img,
                    label_text,
                    (x1, max(0, y1 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
                if label:
                    main_label = label
        return img, main_label

    # ---------- 其他工具方法 ----------
    def reset(self) -> None:
        """重置内部跟踪状态"""
        self.tracking.clear()





