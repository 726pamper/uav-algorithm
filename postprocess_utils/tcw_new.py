"""
tcw_new.engine 模型的后处理模块
复用 weiting_car.py 的逻辑，因为类别和判断逻辑完全一致
类别顺序：先 tcw 再 car（不影响逻辑，因为通过 class_name 判断）
"""

from .normal_det import normal_detections, normal_draw_detection_results
import cv2
import numpy as np

ft = cv2.freetype.createFreeType2()
ft.loadFontData(
    fontFileName="/tensorrt/uav-algorithm-pipeline/postprocess_utils/fonts/wqy-zenhei.ttc",
    idx=0
)

ALGO_ID = "78"


def custom_car_draw_detection_results(image, draw_info):
    """
    根据OBB绘制信息绘制检测结果，并添加车辆计数
    
    Args:
        image: 输入图像 (BGR numpy array)
        draw_info: 来自car_detections的绘制信息列表，包含车辆与停车位的多边形坐标
    
    Returns:
        drawn_image: 绘制后的图像 (BGR格式)
        violation_class_name: 违停车辆类别名称（如果有违停车辆），否则为 "car"
    """
    # 先调用原始绘制函数
    img_bgr, violation_class_name = car_draw_detection_results(image, draw_info)
    
    # 统计车辆数量
    car_count = sum(1 for item in draw_info if item.get("type") == "car")
    
    # 在画面左上角绘制车辆计数
    count_text = f"车辆计数: {car_count}"
    ft.putText(
        img_bgr,
        count_text,
        (20, 50),
        fontHeight=30,
        color=(255, 255, 255),  # 白色文字
        thickness=-1,
        line_type=cv2.LINE_AA,
        bottomLeftOrigin=False,
    )
    
    return img_bgr, violation_class_name


EXPORT = {
    "detect": normal_detections,
    "draw": normal_draw_detection_results,
}

