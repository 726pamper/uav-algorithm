"""
多GPU负载均衡器 - 用于4张NVIDIA T4显卡的推理分配
支持轮询、最少占用、加权负载均衡等策略
"""

import threading
import time
import logging
from typing import Dict, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class LoadBalanceStrategy(Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"           # 轮询
    LEAST_LOADED = "least_loaded"         # 最少占用
    WEIGHTED = "weighted"                 # 加权（基于历史耗时）
    ADAPTIVE = "adaptive"                 # 自适应（根据实时利用率）


@dataclass
class GPUStats:
    """GPU统计信息"""
    gpu_id: int
    device_string: str
    total_inferences: int = 0
    failed_inferences: int = 0
    avg_inference_time: float = 0.0
    current_queue_depth: int = 0
    utilization_rate: float = 0.0  # 0-1
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    last_updated: float = 0.0
    inference_times: deque = None  # 最近100次推理耗时
    
    def __post_init__(self):
        if self.inference_times is None:
            self.inference_times = deque(maxlen=100)
    
    def update_inference_time(self, elapsed_ms: float):
        """更新推理耗时统计"""
        self.inference_times.append(elapsed_ms)
        if len(self.inference_times) > 0:
            self.avg_inference_time = np.mean(list(self.inference_times))
    
    def get_load_score(self, strategy: LoadBalanceStrategy = LoadBalanceStrategy.ADAPTIVE) -> float:
        """
        获取GPU的负载评分（越低越好，表示负载越轻）
        
        - ROUND_ROBIN: 轮询，不看负载
        - LEAST_LOADED: 基于队列深度和当前利用率
        - WEIGHTED: 基于平均推理耗时和队列深度
        - ADAPTIVE: 综合考虑所有因素
        """
        if strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return 0.0  # 轮询不看负载
        
        elif strategy == LoadBalanceStrategy.LEAST_LOADED:
            # 队列深度权重70%，利用率权重30%
            return self.current_queue_depth * 0.7 + self.utilization_rate * 0.3
        
        elif strategy == LoadBalanceStrategy.WEIGHTED:
            # 考虑平均推理耗时和队列深度
            # 耗时越长说明GPU性能使用越充分，应该少分配
            time_weight = self.avg_inference_time / 100.0  # 归一化
            queue_weight = self.current_queue_depth
            return time_weight * 0.4 + queue_weight * 0.6
        
        elif strategy == LoadBalanceStrategy.ADAPTIVE:
            # 综合评分：队列深度(40%) + 利用率(30%) + 推理耗时(20%) + 失败率(10%)
            failure_rate = self.failed_inferences / max(1, self.total_inferences)
            time_normalized = min(1.0, self.avg_inference_time / 200.0)
            
            score = (
                self.current_queue_depth * 0.4 +
                self.utilization_rate * 0.3 +
                time_normalized * 0.2 +
                failure_rate * 0.1
            )
            return score
        
        return 0.0


class GPULoadBalancer:
    """
    多GPU负载均衡器
    
    使用方式：
    ```python
    balancer = GPULoadBalancer(
        num_gpus=4,
        strategy=LoadBalanceStrategy.ADAPTIVE
    )
    
    # 选择GPU
    gpu_id = balancer.select_gpu()
    
    # 执行推理
    try:
        result = inference_engine.run_on_gpu(gpu_id, data)
        balancer.record_inference(gpu_id, inference_time_ms, success=True)
    except Exception as e:
        balancer.record_inference(gpu_id, inference_time_ms, success=False)
    ```
    """
    
    def __init__(
        self,
        num_gpus: int = 4,
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.ADAPTIVE,
        update_interval: float = 1.0,
        max_queue_depth: int = 32
    ):
        """
        初始化负载均衡器
        
        Args:
            num_gpus: GPU数量
            strategy: 负载均衡策略
            update_interval: 统计信息更新间隔（秒）
            max_queue_depth: 最大队列深度阈值
        """
        self.num_gpus = num_gpus
        self.strategy = strategy
        self.update_interval = update_interval
        self.max_queue_depth = max_queue_depth
        self.stats = {}
        # 初始化GPU统计信息
        self.gpu_stats: Dict[int, GPUStats] = {}
        for i in range(num_gpus):
            self.gpu_stats[i] = GPUStats(
                gpu_id=i,
                device_string=f"cuda:{i}",
                memory_total_mb=15360  # T4显卡16GB内存
            )
        
        # 轮询计数器
        self.round_robin_index = 0
        
        # 线程同步
        self.stats_lock = threading.RLock()
        self.last_update_time = time.time()
        
        logger.info(f"[GPULoadBalancer] 初始化成功: {num_gpus}张GPU, 策略={strategy.value}")
    
    def select_gpu(self) -> int:
        """
        选择下一个GPU进行推理
        
        Returns:
            GPU ID (0-N)
        """
        with self.stats_lock:
            if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
                gpu_id = self.round_robin_index % self.num_gpus
                self.round_robin_index += 1
                return gpu_id
            
            else:
                # 其他策略：选择负载最低的GPU
                scores = {
                    gpu_id: stats.get_load_score(self.strategy)
                    for gpu_id, stats in self.gpu_stats.items()
                }
                best_gpu = min(scores, key=scores.get)
                
                logger.debug(
                    f"[GPULoadBalancer] 选择GPU: {best_gpu}, "
                    f"评分={scores[best_gpu]:.2f}, "
                    f"所有评分={{{', '.join(f'GPU{i}:{s:.2f}' for i,s in scores.items())}}}"
                )
                
                return best_gpu
    
    def record_inference(
        self,
        gpu_id: int,
        elapsed_time_ms: float,
        success: bool = True,
        queue_depth: int = None
    ):
        """
        记录推理结果（应在推理完成后调用）
        
        Args:
            gpu_id: GPU ID
            elapsed_time_ms: 推理耗时（毫秒）
            success: 是否成功
            queue_depth: 当前队列深度（可选）
        """
        if gpu_id < 0 or gpu_id >= self.num_gpus:
            logger.warning(f"[GPULoadBalancer] 无效的GPU ID: {gpu_id}")
            return
        
        with self.stats_lock:
            stats = self.gpu_stats[gpu_id]
            
            # 更新统计信息
            stats.total_inferences += 1
            if not success:
                stats.failed_inferences += 1
            
            # 更新推理耗时
            stats.update_inference_time(elapsed_time_ms)
            
            # 更新队列深度
            if queue_depth is not None:
                stats.current_queue_depth = queue_depth
            
            stats.last_updated = time.time()
            
            # 定期打印统计信息
            if time.time() - self.last_update_time > self.update_interval:
                self._print_stats()
                self.last_update_time = time.time()
    
    def update_gpu_utilization(self, gpu_id: int, utilization_rate: float):
        """
        更新GPU利用率（可选，用于更精确的负载评估）
        
        Args:
            gpu_id: GPU ID
            utilization_rate: 利用率 (0.0-1.0)
        """
        if gpu_id < 0 or gpu_id >= self.num_gpus:
            return
        
        with self.stats_lock:
            self.gpu_stats[gpu_id].utilization_rate = min(1.0, max(0.0, utilization_rate))
    
    def get_stats(self, gpu_id: int = None) -> Dict:
        """
        获取GPU统计信息
        
        Args:
            gpu_id: 指定GPU ID，None则返回所有GPU统计
        
        Returns:
            统计信息字典
        """
        with self.stats_lock:
            if gpu_id is not None:
                if gpu_id not in self.gpu_stats:
                    return {}
                stats = self.gpu_stats[gpu_id]
                return {
                    "gpu_id": stats.gpu_id,
                    "device": stats.device_string,
                    "total_inferences": stats.total_inferences,
                    "failed_inferences": stats.failed_inferences,
                    "success_rate": 1.0 - (stats.failed_inferences / max(1, stats.total_inferences)),
                    "avg_inference_time_ms": round(stats.avg_inference_time, 2),
                    "queue_depth": stats.current_queue_depth,
                    "utilization_rate": round(stats.utilization_rate, 2),
                    "load_score": round(stats.get_load_score(self.strategy), 2),
                }
            else:
                return {
                    gpu_id: {
                        "gpu_id": stats.gpu_id,
                        "device": stats.device_string,
                        "total_inferences": stats.total_inferences,
                        "failed_inferences": stats.failed_inferences,
                        "success_rate": 1.0 - (stats.failed_inferences / max(1, stats.total_inferences)),
                        "avg_inference_time_ms": round(stats.avg_inference_time, 2),
                        "queue_depth": stats.current_queue_depth,
                        "utilization_rate": round(stats.utilization_rate, 2),
                        "load_score": round(stats.get_load_score(self.strategy), 2),
                    }
                    for gpu_id, stats in self.gpu_stats.items()
                }
    
    def _print_stats(self):
        """打印统计信息"""
        logger.info("[GPULoadBalancer] ========== GPU统计信息 ==========")
        for gpu_id in range(self.num_gpus):
            stats = self.stats[gpu_id]
            success_rate = 1.0 - (stats.failed_inferences / max(1, stats.total_inferences))
            logger.info(
                f"[GPU{gpu_id}] "
                f"推理数={stats.total_inferences} | "
                f"成功率={success_rate:.1%} | "
                f"平均耗时={stats.avg_inference_time:.2f}ms | "
                f"队列={stats.current_queue_depth} | "
                f"利用率={stats.utilization_rate:.1%} | "
                f"负载评分={stats.get_load_score(self.strategy):.2f}"
            )
    
    def reset_stats(self, gpu_id: int = None):
        """重置统计信息"""
        with self.stats_lock:
            if gpu_id is not None:
                if gpu_id in self.gpu_stats:
                    self.gpu_stats[gpu_id] = GPUStats(
                        gpu_id=gpu_id,
                        device_string=f"cuda:{gpu_id}",
                        memory_total_mb=16384
                    )
            else:
                for i in range(self.num_gpus):
                    self.gpu_stats[i] = GPUStats(
                        gpu_id=i,
                        device_string=f"cuda:{i}",
                        memory_total_mb=16384
                    )
