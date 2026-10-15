import cv2
import numpy as np
import time
from .chinese_name import CLASS_NAME_TO_CHINESE

ft = cv2.freetype.createFreeType2()
ft.loadFontData(
    fontFileName="/tensorrt/uav-algorithm-pipeline/postprocess_utils/fonts/wqy-zenhei.ttc",
    idx=0
)

ALGO_ID = "llm"


def llm_draw_detection_results(jpeg_bytes, draw_info):
    """
    绘制检测结果

    Args:
        image: 输入图像 (BGR numpy array)
        draw_info: 来自 `normal_detections` 的绘制信息列表

    Returns:
        img_bgr: 绘制后的图像
    """
    np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"无法读取图片")

    i_h, i_w = frame.shape[:2]

    # 你的模型是基于 1000x1000 推理的
    h_, w_ = i_h / 1000, i_w / 1000

    for d in draw_info:
        box = d.get("bbox_2d")
        label = d.get("label", "目标物")
        # d.get("category") or d.get("label")
        if not box or len(box) != 4:
            continue

        # 坐标缩放还原
        x1, y1, x2, y2 = [
            int(v * w_) if idx % 2 == 0 else int(v * h_)
            for idx, v in enumerate(box)
        ]

        # 画框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (215, 0, 0), 3)

        # 画标签
        ft.putText(
            frame,
            label,
            (x1, min(frame.shape[0] - 20, y2 - 25)),
            fontHeight=30,
            color=(0, 255, 255),  # 白色文字
            thickness=-1,
            line_type=cv2.LINE_AA,
            bottomLeftOrigin=False,
        )

    return frame


EXPORT = {
    "draw": llm_draw_detection_results,
}