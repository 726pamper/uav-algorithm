from ultralytics.models.sam import SAM3SemanticPredictor
import torch
import gc


def Sam3VideoLoader():
    overrides = dict(conf=0.6, task="segment", mode="predict", imgsz=640,
                     model="/tensorrt/checkpoints/sam3.pt", half=True, save=False)
    predictor = SAM3SemanticPredictor(overrides=overrides)
    return predictor


def ReleaseSam3Model(predictor):
    """释放 SAM3 模型资源"""
    if predictor is not None:
        # 清理 predictor 内部可能持有的张量
        if hasattr(predictor, 'model'):
            del predictor.model
        if hasattr(predictor, 'results'):
            del predictor.results
        del predictor

    # 清理 CUDA 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 强制垃圾回收
    gc.collect()


if __name__ == "__main__":
    Sam3VideoLoader()
