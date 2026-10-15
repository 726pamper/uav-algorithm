"""车牌 OCR 后处理（算法 35）。

链路:
    检测阶段（EXPORT 无 'detect'，由 redis_result_processor 回退到 task='detect'
    → normal_det）产出车牌框；``_save_result`` 中对 algo 35 先 ``ocr_draw`` 画框，
    再调本模块 ``ocr`` 做文字识别。

依赖:
    - onnxocr（ONNXPaddleOcr，实例为 redis_result_processor.self.ocr_model）
    - Pillow（Image / ImageDraw / ImageFont）

修复记录（原实现 OCR 结果拿不到的原因）:
    1. ``text_visual.create_blank_img`` 用 ``np.ones(..., dtype=np.int8) * 255``，
       int8 无法承载 255（numpy>=2 直接 OverflowError；老版本得到 int16，
       PIL.fromarray 同样不支持）→ ``draw_ocr`` 抛异常 → ``ocr`` 整体失败。
       改为 ``dtype=np.uint8``。
    2. ``draw_ocr`` / ``text_visual`` 对"某裁剪框未识别到文字"（空列表）会
       ``scores[i][0]`` / ``txt[0]`` 越界（IndexError）。已加空值保护。
    3. ``ocr`` 原先把各裁剪框内 OCR 得到的局部坐标直接画到全图，位置错乱；
       现按裁剪原点 (x1,y1) 平移回全图坐标，并把各框结果展平为单层列表。
"""

import cv2
import math
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import re
from .chinese_name import CLASS_NAME_TO_CHINESE

ALGO_ID = "35"
module_dir = Path(__file__).resolve().parent
DEFAULT_FONT = str(module_dir / "onnxocr/fonts/simfang.ttf")

# ---- 显示样式（对齐平台目标样式：绿框 + 上方"车辆" + 下方车牌号）----
BOX_BGR = (0, 255, 0)      # 框与文字颜色：绿色
LABEL_TEXT = "车辆"         # 框上方固定文案

# 车牌：省份简称 + 字母 + 5~6 位字母数字（含新能源 8 位）
_PLATE_RE = re.compile(
    r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼]"
    r"[A-Z][A-Z0-9]{5,6}"
)


def _pick_plate(texts, scores):
    """从多条识别文本里挑最像车牌的一条；挑不到则取分数最高的一条。

    目的是避免水印/广告字被写进 detect_frame（例如整图送检时会带出"汽车之家"）。
    特殊号牌（港澳/使领馆等）匹配不上时自动退化为"取最高分"，不会报错。
    """
    cand = [(str(t), float(s)) for t, s in zip(texts, scores) if str(t).strip()]
    if not cand:
        return "", 0.0
    for t, s in cand:
        if _PLATE_RE.search(t.replace("·", "").replace(" ", "")):
            return t, s
    return max(cand, key=lambda x: x[1])


def _font_size(img_shape):
    return max(16, int(max(img_shape[:2]) / 40))


def _pil_draw_texts(img_bgr, items, bgr_color, font_path=DEFAULT_FONT):
    """用 PIL 在图上写中文（cv2.putText 画不出中文，会变 "?"）。

    items: [(x, y, text)]，y 为文字**左上角**坐标。
    """
    if not items:
        return img_bgr
    size = _font_size(img_bgr.shape)
    try:
        font = ImageFont.truetype(font_path, size, encoding="utf-8")
    except Exception:
        font = ImageFont.load_default()

    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    drawer = ImageDraw.Draw(pil)
    rgb = (bgr_color[2], bgr_color[1], bgr_color[0])
    for x, y, text in items:
        drawer.text((max(0, int(x)), max(0, int(y))), str(text), rgb, font=font)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def resize_img(img, input_size=600):
    """限制最长边到 input_size 后返回缩放图。"""
    img = np.array(img)
    h, w = img.shape[:2]
    scale = float(input_size) / float(max(h, w))
    return cv2.resize(img, None, None, fx=scale, fy=scale)


def str_count(s):
    """统计中文个数（英文字符/数字/空格按半个中文字符计）。"""
    import string

    count_zh = count_pu = 0
    s_len = len(str(s))
    en_dg_count = 0
    for c in str(s):
        if c in string.ascii_letters or c.isdigit() or c.isspace():
            en_dg_count += 1
        elif c.isalpha():
            count_zh += 1
        else:
            count_pu += 1
    return s_len - math.ceil(en_dg_count / 2)


def _create_blank(img_h, img_w):
    """新建白底画布。必须是 uint8：int8 会导致 PIL.fromarray 失败。"""
    blank = np.ones((img_h, img_w), dtype=np.uint8) * 255
    im = Image.fromarray(blank).convert("RGB")
    return im, ImageDraw.Draw(im)


def text_visual(texts, scores, img_h=400, img_w=600, threshold=0.0, font_path=DEFAULT_FONT):
    """把识别文本渲染为右侧文本面板。

    texts / scores 均为**一维**列表（已展平）。
    """
    if scores is not None:
        assert len(texts) == len(scores), "texts 与 scores 数量必须一致"

    blank_img, draw_txt = _create_blank(img_h, img_w)
    font_size = 20
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    gap = font_size + 5

    txt_img_list = []
    count, index = 1, 0
    for idx, txt in enumerate(texts):
        score = float(scores[idx]) if scores is not None else 1.0
        if math.isnan(score) or score < threshold:
            continue

        index += 1
        txt = str(txt)
        first_line = True

        while str_count(txt) >= img_w // font_size - 4:
            tmp = txt
            txt = tmp[: img_w // font_size - 4]
            new_txt = f"{index}: {txt}" if first_line else f"    {txt}"
            first_line = False
            draw_txt.text((0, gap * count), new_txt, (0, 0, 0), font=font)
            txt = tmp[img_w // font_size - 4:]
            if count >= img_h // gap - 1:
                txt_img_list.append(np.array(blank_img))
                blank_img, draw_txt = _create_blank(img_h, img_w)
                count = 0
            count += 1

        new_txt = f"{index}: {txt}   {score:.3f}" if first_line else f"  {txt}  {score:.3f}"
        draw_txt.text((0, gap * count), new_txt, (0, 0, 0), font=font)

        if count >= img_h // gap - 1 and idx + 1 < len(texts):
            txt_img_list.append(np.array(blank_img))
            blank_img, draw_txt = _create_blank(img_h, img_w)
            count = 0
        count += 1

    txt_img_list.append(np.array(blank_img))
    if len(txt_img_list) == 1:
        out = np.array(txt_img_list[0])
    else:
        out = np.concatenate(txt_img_list, axis=1)
    return np.array(out)


def draw_ocr(image, boxes, txts=None, scores=None, drop_score=0.5, font_path=DEFAULT_FONT):
    """在给定框的**下方**绘制识别结果（绿色车牌号）。

    显示样式对齐平台目标：框由 ``ocr_draw`` 画（绿色）+ 上方"车辆"，
    本函数只负责在框下方补一行车牌号（不画框、不显示置信度分数）。

    Args:
        boxes: 长度 N 的列表，元素为该框的 (M,2) 绝对坐标点集
        txts / scores: 长度 N 的一维列表
    """
    img = np.array(image).copy()
    n = len(boxes)
    if scores is None:
        scores = [1.0] * n

    h = img.shape[0]
    size = _font_size(img.shape)

    items = []
    for i in range(n):
        b = boxes[i]
        if b is None or len(np.asarray(b)) == 0:
            continue
        try:
            s = float(scores[i])
        except (TypeError, ValueError):
            continue
        if math.isnan(s) or s < drop_score:
            continue

        text = str(txts[i]) if txts and i < len(txts) else ""
        if not text:
            continue

        pts = np.asarray(b).reshape(-1, 2)
        x = int(pts[:, 0].min())
        top = int(pts[:, 1].max()) + 4          # 默认画在框下方
        if top + size > h:                       # 底部空间不足则画到框上方
            top = max(0, int(pts[:, 1].min()) - size - 6)
        items.append((x, top, text))

    return _pil_draw_texts(img, items, BOX_BGR, font_path)


def numpy_crop(img, bbox):
    """按 bbox 列表裁图（保留供外部调用）。"""
    inputs = []
    for box in bbox:
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = map(int, box)
        if x2 <= x1 or y2 <= y1:
            continue
        inputs.append(img[y1:y2, x1:x2, :])
    return inputs


def _draw_plate_annotations(img_bgr, draw_info):
    """统一绘制：每个检测框画绿框 + 上方"车辆"；识别到车牌的在框下方写车牌号。"""
    size = _font_size(img_bgr.shape)
    texts = []
    boxes, txts, scores = [], [], []

    for item in draw_info:
        if item.get("type") != "detection":
            continue
        bbox = item.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), BOX_BGR, 2)

        top = y1 - size - 6
        if top < 0:                      # 贴顶时画到框内
            top = y1 + 4
        texts.append((x1, top, LABEL_TEXT))

        plate = str(item.get("ocr_text") or "").strip()
        if plate:
            boxes.append(np.asarray(bbox, dtype=np.float32).reshape(-1, 2))
            txts.append(plate)
            scores.append(float(item.get("ocr_score") or 1.0))

    img_bgr = _pil_draw_texts(img_bgr, texts, BOX_BGR)
    return draw_ocr(img_bgr, boxes, txts, scores, drop_score=0.0)


def ocr(img, draw_info, model):
    """裁剪车牌框 → ONNX OCR → 绘制（绿框 + 上方"车辆" + 下方车牌号）。

    顺序很重要：**先在干净图上识别，识别完再画框**。
    若先画框（ocr_draw 画），框线会盖住车牌首字污染 OCR 输入
    （实测 川A·88888 会变成 A·88888）。

    **副作用（多车牌方案依赖）**：把每个检测框识别到的车牌号写回对应的
    ``draw_info`` item（``item["ocr_text"]`` / ``item["ocr_score"]``）。
    ``draw_info`` 是同一个 list 对象，``redis_result_processor._save_result``
    可据此把车牌号逐个写进 ``detect_frame`` 的框（每个车牌号独立 class_id）。

    Returns:
        (img_bgr, "、".join(识别到的车牌号))
    """
    image = img.copy()
    plates = []

    # 1) 在**干净图**上识别（此时图上还没有框线，避免框线盖住车牌首字）
    for item in draw_info:
        bbox = item.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2, :]
        if crop is None or crop.size == 0:
            continue

        try:
            res = model.ocr(crop)
        except Exception:
            continue
        if not res or not res[0]:
            continue

        texts_this = [str(line[1][0]) for line in res[0]]
        scores_this = [float(line[1][1]) for line in res[0]]
        plate, plate_score = _pick_plate(texts_this, scores_this)
        if plate:
            # 写回该检测框（同一 list 对象，_save_result 直接可读）
            item["ocr_text"] = plate
            item["ocr_score"] = plate_score
            plates.append(plate)

    # 2) 识别完成后再统一绘制（框 + 上方"车辆" + 下方车牌号）
    image = _draw_plate_annotations(image, draw_info)
    return image, ("、".join(plates) if plates else "车牌")


def ocr_draw(image, draw_info):
    """本函数**不再绘制**：绘制统一放在 ``ocr()`` 里、识别之后做。

    原因：框架调用顺序是 ocr_draw → ocr，若在此处先画框，绿框线会盖住车牌
    首字，污染随后 OCR 的裁剪输入（实测 川A·88888 会变成 A·88888）。

    保留本函数以兼容 redis_result_processor 的调用约定：返回 (原图, 类别名)。
    """
    violation_class_name = "normal"
    for item in draw_info:
        label = item.get("label", "")
        if label:
            violation_class_name = label
    return image, violation_class_name


EXPORT = {
    "ocr_draw": ocr_draw,
    "ocr": ocr,
}
