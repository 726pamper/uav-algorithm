import torch
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
import time
from pathlib import Path
from ultralytics import YOLO

def _read_engine_input_size(engine_path):
    """直接反序列化 engine, 读输入 binding 的真实尺寸作为唯一事实来源。
    静态engine返回实际(h,w); 动态engine返回optimization profile的opt(h,w); 失败返回None。"""
    try:
        # 0) 关键: TRT反序列化需要已激活的CUDA上下文(校验GPU架构), 否则静默返回None
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda:0")   # 强制创建/激活主上下文(幂等, 已存在时无开销)
        import tensorrt as trt
        trt_logger = trt.Logger(trt.Logger.INFO)   # INFO: 若仍失败, 能看到TRT原生错误详情
        trt.init_libnvinfer_plugins(trt_logger, "")
        with open(engine_path, 'rb') as f:
            data = f.read()
        rt = trt.Runtime(trt_logger)
        eng = rt.deserialize_cuda_engine(data)
        if eng is None:
            print(f"[TRT] engine反序列化返回None (trt={trt.__version__}, {engine_path}) — 回退predictor逻辑")
            return None
        n = eng.num_io_tensors if hasattr(eng, 'num_io_tensors') else eng.num_bindings
        for i in range(n):
            if hasattr(eng, 'binding_is_input'):
                is_input = eng.binding_is_input(i)
                shape = tuple(eng.get_binding_shape(i))
            else:
                name = eng.get_tensor_name(i)
                is_input = eng.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                shape = tuple(eng.get_tensor_shape(name))
            if not is_input:
                continue
            h, w = int(shape[2]), int(shape[3])
            if h > 0 and w > 0:
                return (h, w)                          # 静态: 真实构建尺寸
            if hasattr(eng, 'get_profile_shape'):
                opt = tuple(eng.get_profile_shape(0, i)[1])
                return (int(opt[2]), int(opt[3]))
            return None
    except Exception as e:
        print(f"[TRT] 读取engine输入尺寸失败(将回退predictor逻辑): {e}")
    return None

class TRTInferenceEngine:

    def __init__(self, engine_path: str, device: str = "cuda:0"):
        """
        初始化 TensorRT 推理引擎（使用 ultralytics）

        Args:
            engine_path: TensorRT engine 文件路径
            device: CUDA 设备 (默认 "cuda:0")
            is_obb: 是否为 OBB 检测（如果未显式指定，则根据文件名推断）
        """
        self.device = torch.device(device)
        engine_name = Path(engine_path).name.lower()
        self.task = None
        # 判断模型类型：优先使用is_obb参数，否则根据文件名推断

        if "obb" in engine_name:
            self.task = 'obb'
        elif "seg" in engine_name:
            self.task = 'segment'
        else:
            # 默认为DET（水平框检测）
            self.task = 'detect'

        # 使用 ultralytics 加载 TensorRT engine
        self.model = YOLO(engine_path, task=self.task)

        _sz = _read_engine_input_size(engine_path)
        if _sz:
            self.input_size = _sz
            print(f"[TRT] input_size 自适应engine binding: {self.input_size}")
        else:
            try:
                if hasattr(self.model, 'predictor') and self.model.predictor:
                    self.input_size = self.model.predictor.args.imgsz
                else:
                    self.input_size = (640, 640)
                if isinstance(self.input_size, int):
                    self.input_size = (self.input_size, self.input_size)
                elif isinstance(self.input_size, list):
                    self.input_size = tuple(self.input_size)
            except Exception:
                self.input_size = (640, 640)
            print(f"[TRT] input_size 回退predictor逻辑: {self.input_size}")

        # Warmup（预热模型，消除首次推理的额外开销）
        self._warmup()

    def _warmup(self, num_iterations: int = 3):
        """
        预热模型，消除首次推理的 CUDA 初始化开销

        Args:
            num_iterations: 预热迭代次数
        """
        try:
            # 创建虚拟输入
            dummy_input = torch.zeros(
                (1, 3, self.input_size[0], self.input_size[1]),
                dtype=torch.float32,
                device=self.device
            )

            # 预热推理
            for i in range(num_iterations):
                _ = self.model(
                    dummy_input,
                    imgsz=self.input_size,
                    conf=0.25,
                    iou=0.45,
                    verbose=False,
                    device=self.device
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        except Exception as e:
            print(f"[TRT] warmup 失败（不影响加载，但输入尺寸可能不匹配）: {e}")

    def _convert_results(self, results) -> List[torch.Tensor]:
        """

        Returns:
            List[Tensor]: 每个 tensor 形状为 (N, 7), 格式为 [x1, y1, x2, y2, track_id, conf, class_id]
        """
        return self._simple_convert_results(results)

    def track(
            self,
            image: torch.Tensor,
            input_size: Tuple[int, int] = (640, 640),
            conf_threshold: float = 0.25,
            iou_threshold: float = 0.45,
            return_torch: bool = True,
            tracker: str = "bytetrack.yaml",
            persist: bool = True
    ):
        """
        使用 ByteTrack 进行目标跟踪

        Args:
            image: 输入 tensor (H, W, C) 或 (C, H, W) 或 (B, C, H, W) 格式
            input_size: 模型输入尺寸 (height, width)，默认(640, 640)
            conf_threshold: 置信度阈值
            iou_threshold: IOU 阈值（用于 NMS）
            return_torch: 是否返回 torch tensor
            tracker: 跟踪器配置文件名，默认 "bytetrack.yaml"
            persist: 是否保持跟踪状态（跨帧跟踪）

        Returns:
            包含跟踪ID的tensor列表
            - OBB模式: (N, 8) = [cx, cy, w, h, angle, conf, class_id, track_id]
            - DET模式: (N, 7) = [x1, y1, x2, y2, conf, class_id, track_id]
        """
        try:
            if not image.is_cuda:
                image = image.to(self.device)

            # 记录原图尺寸（用于后续坐标缩放）
            original_h, original_w = self._get_image_size(image)

            # RGBA → RGB转换
            image = self._convert_rgba_to_rgb(image)

            # 格式转换为BCHW（Ultralytics对torch.Tensor要求BCHW）
            image = self._convert_to_bchw(image)

            # 保存letterbox前的尺寸（用于坐标转换）
            letterbox_h, letterbox_w = image.shape[2], image.shape[3]

            # 进行letterbox预处理
            image, (scale_y, scale_x), (pad_h, pad_w) = self._letterbox(
                image,
                new_shape=input_size,
                color=0.0,  # 填充黑色
                auto=True,  # 自动计算最小填充（确保能被32整除）
                scaleFill=False,  # 保持宽高比
                scaleup=True  # 允许放大
            )

            # 数据类型转换和归一化
            if image.dtype == torch.uint8:
                image = image.float() / 255.0  # uint8 [0, 255] -> float32 [0.0, 1.0]
            elif image.dtype in [torch.float32, torch.float16, torch.float64]:
                if image.max() > 1.0:
                    image = image / 255.0

            # 使用CUDA Events异步测量时间（减少阻塞）
            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.time()

            # 调用Ultralytics跟踪方法
            results = self.model.track(
                image,  # 已经是 (1, 3, 640, 640) BCHW格式，float32，归一化到[0,1]
                imgsz=input_size,  # 告诉Ultralytics输入尺寸（用于坐标缩放）
                conf=conf_threshold,
                iou=iou_threshold,
                tracker=tracker,  # 使用ByteTrack跟踪器
                persist=persist,  # 保持跟踪状态
                verbose=False,  # 关闭详细输出（提高性能）
                device=self.device
            )

            # 测量推理时间
            if torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize()
            else:
                pass

            # 后处理：格式转换和坐标缩放（包含跟踪ID）
            if return_torch:
                # 将坐标从letterbox后的图像转换回原图尺寸
                converted_results = self._simple_convert_results(
                    results,
                    original_size=(original_h, original_w) if original_h and original_w else None,
                    model_input_size=input_size,
                    letterbox_scales=(scale_y, scale_x),
                    letterbox_pad=(pad_h, pad_w),
                    with_tracking=True  # 启用跟踪模式
                )
                return converted_results
            else:
                # 直接返回ultralytics Results对象
                return results

        except Exception as e:
            import traceback
            traceback.print_exc()
            # 返回空结果而不是崩溃
            if return_torch:
                empty_shape = (0, 8) if self.task == 'obb' else (0, 7)
                return [torch.empty(empty_shape, device=self.device)]
            else:
                empty_shape = (0, 8) if self.task == 'obb' else (0, 7)
                return [np.empty(empty_shape)]

    def preprocess_tensor(self, image: torch.Tensor, target_size: Tuple[int, int] = None) -> torch.Tensor:
        """
        预处理 GPU tensor 以适配 ultralytics

        Args:
            image: GPU tensor (H, W, C) 或 (C, H, W) 格式
            target_size: 目标尺寸 (height, width)，如果为 None 则不 resize

        Returns:
            处理后的 tensor，格式为 (1, 3, H, W) - BCHW
        """
        # 确保在正确的设备上
        if not image.is_cuda:
            image = image.to(self.device)

        # 如果是 (H, W, C) 格式，转换为 (C, H, W)
        if image.dim() == 3 and image.shape[2] in [1, 3, 4]:
            image = image.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        elif image.dim() == 2:
            image = image.unsqueeze(0)  # (H, W) -> (1, H, W)

        # 处理 RGBA: 只取前3个通道 (RGB)
        if image.shape[0] == 4:
            image = image[:3, :, :]  # 丢弃 Alpha 通道
        elif image.shape[0] == 1:
            # 灰度图，复制到3个通道
            image = image.repeat(3, 1, 1)
        elif image.shape[0] != 3:
            raise ValueError(f"Unexpected number of channels: {image.shape[0]}, expected 1, 3, or 4")

        # 添加 batch 维度: (C, H, W) -> (1, C, H, W)
        if image.dim() == 3:
            image = image.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)

        # 转换为 float32 并归一化到 [0, 1]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0  # uint8 [0, 255] -> float32 [0.0, 1.0]
        elif image.dtype in [torch.float32, torch.float16, torch.float64]:
            # 如果已经是浮点类型，检查是否需要归一化
            if image.max() > 1.0:
                image = image / 255.0

        # Resize 到目标尺寸（如果指定）
        if target_size is not None:
            target_h, target_w = target_size
            current_h, current_w = image.shape[2], image.shape[3]

            if (current_h, current_w) != (target_h, target_w):
                # 使用双线性插值 resize
                image = torch.nn.functional.interpolate(
                    image,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                )

        # 确保内存连续
        image = image.contiguous()

        return image

    def __call__(
            self,
            image: torch.Tensor,
            input_size: Tuple[int, int] = (640, 640),
            conf_threshold: float = 0.25,
            iou_threshold: float = 0.45,
            return_torch: bool = True
    ):
        """

        Args:
            image: 输入 tensor (H, W, C) 或 (C, H, W) 或 (B, C, H, W) 格式
            input_size: 模型输入尺寸 (height, width)，默认(640, 640)
            conf_threshold: 置信度阈值
            iou_threshold: IOU 阈值（用于 NMS）
            return_torch: 是否返回 torch tensor

        Returns:
            ultralytics Results 对象或转换后的tensor列表
        """
        try:
            if not image.is_cuda:
                image = image.to(self.device)

            # 记录原图尺寸（用于后续坐标缩放）
            original_h, original_w = self._get_image_size(image)

            # RGBA → RGB转换
            image = self._convert_rgba_to_rgb(image)

            # 格式转换为BCHW（Ultralytics对torch.Tensor要求BCHW）
            image = self._convert_to_bchw(image)

            # （按比例缩放 + 填充黑色）
            # 保存letterbox前的尺寸（用于坐标转换）
            letterbox_h, letterbox_w = image.shape[2], image.shape[3]

            # 进行letterbox预处理
            image, (scale_y, scale_x), (pad_h, pad_w) = self._letterbox(
                image,
                new_shape=input_size,
                color=0.0,  # 填充黑色
                auto=True,  # 自动计算最小填充（确保能被32整除）
                scaleFill=False,  # 保持宽高比
                scaleup=True  # 允许放大
            )

            # 数据类型转换和归一化
            if image.dtype == torch.uint8:
                image = image.float() / 255.0  # uint8 [0, 255] -> float32 [0.0, 1.0]
            elif image.dtype in [torch.float32, torch.float16, torch.float64]:
                if image.max() > 1.0:
                    image = image / 255.0

            # 使用CUDA Events异步测量时间（减少阻塞）
            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.time()

            # 调用Ultralytics（不再进行resize，只做推理和NMS）
            # 告诉Ultralytics输入尺寸，以便它正确缩放坐标
            results = self.model(
                image,  # 已经是 (1, 3, 640, 640) BCHW格式，float32，归一化到[0,1]
                imgsz=input_size,  # 告诉Ultralytics输入尺寸（用于坐标缩放）
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,  # 关闭详细输出（提高性能）
                device=self.device
            )

            # 测量推理时间
            if torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize()
            else:
                pass

            # 后处理：格式转换和坐标缩放
            if return_torch:
                # 将坐标从letterbox后的图像转换回原图尺寸
                converted_results = self._simple_convert_results(
                    results,
                    original_size=(original_h, original_w) if original_h and original_w else None,
                    model_input_size=input_size,
                    letterbox_scales=(scale_y, scale_x),
                    letterbox_pad=(pad_h, pad_w)
                )
                return converted_results
            else:
                # 直接返回ultralytics Results对象
                return results

        except Exception as e:
            import traceback
            traceback.print_exc()
            # 返回空结果而不是崩溃
            if return_torch:
                empty_shape = (0, 7) if self.task == 'obb' else (0, 6)
                return [torch.empty(empty_shape, device=self.device)]
            else:
                empty_shape = (0, 7) if self.task == 'obb' else (0, 6)
                return [np.empty(empty_shape)]

    def _get_image_size(self, image: torch.Tensor) -> Tuple[Optional[int], Optional[int]]:
        """
        获取图像尺寸（支持多种格式）

        Args:
            image: 输入tensor，可以是 (H, W, C), (C, H, W), 或 (B, C, H, W)

        Returns:
            (height, width) 元组，如果无法确定则返回 (None, None)
        """
        if image.dim() == 4:  # (B, C, H, W)
            return image.shape[2], image.shape[3]
        elif image.dim() == 3:
            if image.shape[0] in [1, 3, 4]:  # (C, H, W)
                return image.shape[1], image.shape[2]
            else:  # (H, W, C)
                return image.shape[0], image.shape[1]
        else:
            return None, None

    def _convert_rgba_to_rgb(self, image: torch.Tensor) -> torch.Tensor:
        """
        将RGBA转换为RGB（Ultralytics不支持4通道，必须手动处理）

        支持多种格式：
        - (H, W, 4) → (H, W, 3)
        - (4, H, W) → (3, H, W)
        - (B, 4, H, W) → (B, 3, H, W)
        - (B, H, W, 4) → (B, H, W, 3)

        Args:
            image: 输入tensor，可能包含4通道（RGBA）

        Returns:
            转换后的tensor，确保是RGB（3通道）
        """
        if image.dim() == 3:
            if image.shape[-1] == 4:  # (H, W, 4)
                image = image[:, :, :3]  # → (H, W, 3)
            elif image.shape[0] == 4:  # (4, H, W)
                image = image[:3, :, :]  # → (3, H, W)
        elif image.dim() == 4:
            if image.shape[1] == 4:  # (B, 4, H, W)
                image = image[:, :3, :, :]  # → (B, 3, H, W)
            elif image.shape[-1] == 4:  # (B, H, W, 4) - 不太常见
                image = image[:, :, :, :3]  # → (B, H, W, 3)

        return image

    def _letterbox(
            self,
            image: torch.Tensor,
            new_shape: Tuple[int, int] = (640, 640),
            color: float = 0.0,
            auto: bool = True,
            scaleFill: bool = False,
            scaleup: bool = True
    ) -> Tuple[torch.Tensor, Tuple[float, float], Tuple[int, int]]:
        """
        Ultralytics标准的letterbox预处理：按比例缩放 + 填充黑色


        Returns:
            (processed_image, scale, (pad_h, pad_w))
            - processed_image: 处理后的tensor，格式为 (1, C, new_h, new_w)
            - scale: 缩放比例
            - (pad_h, pad_w): 上下和左右的填充像素数
        """
        # 确保是 (1, C, H, W) 格式
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # interpolate 需要浮点类型，如果是 uint8 需要先转换
        # 只转换数据类型，不进行归一化（归一化在 letterbox 之后进行）
        if image.dtype == torch.uint8:
            image = image.float()  # uint8 -> float32，值范围 [0, 255]

        shape = image.shape[-2:]  # (H, W)
        new_h, new_w = new_shape

        # 计算缩放比例
        r = min(new_h / shape[0], new_w / shape[1])
        if not scaleup:  # 只允许缩小，不允许放大
            r = min(r, 1.0)

        # 计算新尺寸（保持宽高比）
        new_unpad = (int(round(shape[0] * r)), int(round(shape[1] * r)))

        if auto:  # 最小矩形填充，确保能被32整除
            # 调整new_unpad使其能被32整除（向下取整到最近的32的倍数）
            new_unpad = (new_unpad[0] - new_unpad[0] % 32, new_unpad[1] - new_unpad[1] % 32)
            # 重新计算实际的缩放比例（基于调整后的new_unpad）
            r = min(new_unpad[0] / shape[0], new_unpad[1] / shape[1])
        elif scaleFill:  # 拉伸填充
            new_unpad = (new_h, new_w)
            r = min(new_h / shape[0], new_w / shape[1])

        # 计算需要填充的像素（确保最终尺寸等于new_shape）
        dw, dh = new_w - new_unpad[1], new_h - new_unpad[0]

        # 计算上下和左右的填充（平均分配）
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:  # 需要resize
            # 使用双线性插值resize（现在 image 已经是 float 类型）
            image = torch.nn.functional.interpolate(
                image,
                size=new_unpad,
                mode='bilinear',
                align_corners=False
            )

        # 计算填充值（整数，对称）
        # torch.nn.functional.pad的参数顺序是: (pad_left, pad_right, pad_top, pad_bottom)
        pad_w_left = int(round(dw))
        pad_w_right = int(round(dw))
        pad_h_top = int(round(dh))
        pad_h_bottom = int(round(dh))

        # 如果填充不是整数，分配余数
        dw_remainder = dw - pad_w_left
        dh_remainder = dh - pad_h_top
        if abs(dw_remainder) > 0.5:
            pad_w_right += 1 if dw_remainder > 0 else -1
        if abs(dh_remainder) > 0.5:
            pad_h_bottom += 1 if dh_remainder > 0 else -1

        # 填充（左右、上下）
        # pad参数: (left, right, top, bottom)
        image = torch.nn.functional.pad(
            image,
            (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom),
            mode='constant',
            value=color
        )

        # 记录最终的纵横缩放比例（避免后续坐标还原时只使用单一比例）
        scale_y = new_unpad[0] / shape[0] if shape[0] > 0 else 1.0
        scale_x = new_unpad[1] / shape[1] if shape[1] > 0 else 1.0

        # 确保是 (1, C, H, W) 格式
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # 返回上边和左边的填充值（用于坐标转换）
        return image, (scale_y, scale_x), (pad_h_top, pad_w_left)

    def _convert_to_bchw(self, image: torch.Tensor) -> torch.Tensor:
        """
        将tensor转换为BCHW格式（Ultralytics对torch.Tensor要求BCHW格式）
        """
        # 确保图片在GPU内存上
        if not image.is_cuda:
            image = image.to(self.device)

        if image.dim() == 3:
            # 判断是 (H, W, C) 还是 (C, H, W)
            if image.shape[-1] in [1, 3]:  # (H, W, C)
                image = image.permute(2, 0, 1)  # → (C, H, W)
            # 现在应该是 (C, H, W)，添加batch维度
            image = image.unsqueeze(0)  # → (1, C, H, W)
        elif image.dim() == 4:
            # 已经是 (B, C, H, W)，保持不变
            pass
        elif image.dim() == 2:
            # (H, W) → (1, 1, H, W)
            image = image.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported tensor dimension: {image.dim()}, shape: {image.shape}")

        # 确保内存连续
        if not image.is_contiguous():
            image = image.contiguous()

        return image

    def _simple_convert_results(
            self,
            results,
            original_size: Optional[Tuple[int, int]] = None,
            model_input_size: Optional[Tuple[int, int]] = None,
            letterbox_scales: Optional[Tuple[float, float]] = None,
            letterbox_pad: Optional[Tuple[int, int]] = None,
            with_tracking: bool = False
    ) -> List[torch.Tensor]:
        # 根据模型类型和是否跟踪确定空结果的形状
        if with_tracking:
            if self.task == 'obb':
                empty_shape = (0, 8)  # OBB + tracking: [cx, cy, w, h, angle, conf, cls, track_id]
            elif self.task == 'segment':
                empty_shape = (0, 33)  # SEG + tracking 默认为空33个点位
            else:
                empty_shape = (0, 7)  # DET + tracking: [x1, y1, x2, y2, conf, cls, track_id]
        else:
            if self.task == 'obb':
                empty_shape = (0, 7)  # OBB: [cx, cy, w, h, angle, conf, cls]
            elif self.task == 'segment':
                empty_shape = (0, 32)  # SEG 默认32个点位
            else:
                empty_shape = (0, 6)  # DET: [x1, y1, x2, y2, conf, cls]

        if not results:
            return [torch.empty(empty_shape, device=self.device)]

        if (
                original_size is None
                or letterbox_scales is None
                or letterbox_pad is None
        ):
            letterbox_scales = None
            letterbox_pad = None
        else:
            orig_h, orig_w = original_size
            scale_y, scale_x = letterbox_scales
            pad_h, pad_w = letterbox_pad

        converted = []
        for idx, result in enumerate(results):
            if self.task == 'obb':
                # OBB模式：处理旋转框
                boxes = result.obb
                if boxes is None or len(boxes) == 0:
                    converted.append(torch.empty((0, 8 if with_tracking else 7), device=self.device))
                    continue

                xywhr = boxes.xywhr.clone()
                if letterbox_scales is not None and letterbox_pad is not None:
                    # 将坐标从letterbox后的图像转换回原图尺寸
                    xywhr[:, 0] = (xywhr[:, 0] - pad_w) / scale_x  # cx
                    xywhr[:, 1] = (xywhr[:, 1] - pad_h) / scale_y  # cy
                    xywhr[:, 2] = xywhr[:, 2] / scale_x  # w
                    xywhr[:, 3] = xywhr[:, 3] / scale_y  # h
                    # angle不需要缩放

                conf = boxes.conf.unsqueeze(1)
                cls = boxes.cls.unsqueeze(1)

                if with_tracking:
                    # 获取跟踪ID（如果存在）
                    if hasattr(boxes, 'id') and boxes.id is not None:
                        track_id = boxes.id.unsqueeze(1).float()  # 转换为float以匹配其他数据
                    else:
                        # 如果没有跟踪ID，使用-1作为占位符
                        track_id = torch.full((len(xywhr), 1), -1.0, device=self.device)
                    converted.append(torch.cat([xywhr, conf, cls, track_id], dim=1))  # (N, 8)
                else:
                    converted.append(torch.cat([xywhr, conf, cls], dim=1))  # (N, 7)

            elif self.task == 'segment':
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    converted.append(torch.empty((0, 7 if with_tracking else 6), device=self.device))
                    continue
                xy = result.masks.xy
                xyxy = boxes.xyxy

                # 转换mask轮廓点坐标从letterbox后的图像转换回原图尺寸
                # 注意：只转换掩码坐标，边界框坐标保持不变（用户需求）
                if letterbox_scales is not None and letterbox_pad is not None:
                    # xy是列表格式，每个元素是一个mask的轮廓点坐标 (M, 2)，其中M是点数，2是x,y坐标
                    # 不同mask的点数可能不同，无法直接批量转换为tensor
                    if xy is not None and len(xy) > 0:
                        try:
                            # 逐个处理每个mask的坐标转换
                            converted_xy = []
                            for mask_coords in xy:
                                if mask_coords is not None and len(mask_coords) > 0:
                                    # 将单个mask的坐标转换为torch tensor进行转换
                                    mask_tensor = torch.tensor(mask_coords, device=self.device, dtype=torch.float32)
                                    # 对每个点的x,y坐标进行转换：(coord - pad) / scale
                                    # 注意：这里可能产生负数，将负数裁剪为0
                                    mask_tensor[:, 0] = torch.clamp((mask_tensor[:, 0] - pad_w) / scale_x,
                                                                    min=0.0)  # x坐标转换，负数变0
                                    mask_tensor[:, 1] = torch.clamp((mask_tensor[:, 1] - pad_h) / scale_y,
                                                                    min=0.0)  # y坐标转换，负数变0
                                    # 转回numpy格式并添加到结果列表
                                    converted_xy.append(mask_tensor.cpu().numpy())
                                else:
                                    converted_xy.append(mask_coords)
                            xy = converted_xy
                        except Exception as e:
                            print(f"[Warning] Failed to convert segment mask coordinates: {e}")
                            # 如果转换失败，保持原坐标

                conf = boxes.conf.unsqueeze(1)
                cls = boxes.cls.unsqueeze(1)
                if with_tracking:
                    # 获取跟踪ID（如果存在）
                    if hasattr(boxes, 'id') and boxes.id is not None:
                        track_id = boxes.id.unsqueeze(1).float()  # 转换为float以匹配其他数据
                    else:
                        # 如果没有跟踪ID，使用-1作为占位符
                        track_id = torch.full((len(xy), 1), -1.0, device=self.device)
                    converted.append([xy, xyxy, conf, cls, track_id])  # (N, 3+point)
                else:
                    converted.append([xy, xyxy, conf, cls])  # (N, 7)

            else:
                # DET模式：处理水平框
                boxes = result.boxes

                if boxes is None or len(boxes) == 0:
                    converted.append(torch.empty((0, 7 if with_tracking else 6), device=self.device))
                    continue

                # 获取xyxy格式的框（相对于letterbox后的图像）
                xyxy = boxes.xyxy.clone()
                xywhn = boxes.xywhn.clone()
                if letterbox_scales is not None and letterbox_pad is not None:
                    # 将坐标从letterbox后的图像转换回原图尺寸
                    xyxy[:, 0] = (xyxy[:, 0] - pad_w) / scale_x  # x1
                    xyxy[:, 1] = (xyxy[:, 1] - pad_h) / scale_y  # y1
                    xyxy[:, 2] = (xyxy[:, 2] - pad_w) / scale_x  # x2
                    xyxy[:, 3] = (xyxy[:, 3] - pad_h) / scale_y  # y2

                conf = boxes.conf.unsqueeze(1)
                cls = boxes.cls.unsqueeze(1)
                if with_tracking:
                    # 获取跟踪ID（如果存在）
                    if hasattr(boxes, 'id') and boxes.id is not None:
                        track_id = boxes.id.unsqueeze(1).float()  # 转换为float以匹配其他数据
                    else:
                        # 如果没有跟踪ID，使用-1作为占位符
                        track_id = torch.full((len(xyxy), 1), -1.0, device=self.device)
                    converted.append(torch.cat([xyxy, conf, cls, xywhn, track_id],
                                               dim=1))  # (N, 11): [x1, y1, x2, y2, conf, cls, cxn, cyn, w, h, track_id]
                else:
                    converted.append(torch.cat([xyxy, conf, cls, xywhn],
                                               dim=1))  # (N, 10): [x1, y1, x2, y2, conf, cls, cxn, cyn, w, h,]

        return converted, self.model.names

    def _del(self):
        """清理资源"""
        if hasattr(self, 'model'):
            del self.model


# ---------------- 使用示例 ----------------
def main():
    """使用示例"""
    # 初始化推理引擎
    engine_path = "model.engine"  # TensorRT engine 文件路径
    inference_engine = TRTInferenceEngine(engine_path, device="cuda:0")

    # 方式 1: 从文件读取图像
    image_path = "test.jpg"
    image_bgr = cv2.imread(image_path)

    # 转换为 torch tensor 并移到 GPU
    image_tensor = torch.from_numpy(image_bgr).to("cuda:0")

    # 推理
    start_time = time.time()
    results = inference_engine.predict(
        image_tensor,
        input_size=(640, 640),
        conf_threshold=0.25,
        iou_threshold=0.45,
        return_torch=True
    )
    inference_time = (time.time() - start_time) * 1000


if __name__ == "__main__":
    main()

