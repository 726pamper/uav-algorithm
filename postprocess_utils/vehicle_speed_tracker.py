"""
车辆测速跟踪模块
基于车道分割和车辆检测进行相对速度计算
算法任务69：集成69车道分割和999车辆检测模型
"""
import cv2
import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional

# 尝试导入redis，如果失败则设置为None
try:
    import redis
except ImportError:
    redis = None

# ALGO_ID = "69"


class VehicleSpeedTracker:
    """
    车辆测速跟踪器
    负责集成车道分割和车辆检测，计算车辆相对速度
    """

    def __init__(self):
        self.task_lane_masks = {}  # {task_id: {lane_id: mask_points}} 按任务隔离的车道掩码缓存
        self.last_coords = {}  # {task_id: {"lat": float, "lng": float}} 按任务隔离的上次经纬度

    def get_drone_speed_from_redis(self, task_id: str, redis_client) -> float:
        """
        从Redis获取无人机速度
        先检查经纬度是否变化，如果未变化则返回0

        Args:
            task_id: 任务ID
            redis_client: Redis客户端

        Returns:
            float: 无人机速度(km/h)，如果位置未变化则返回0
        """
        try:
            coord_key = f"coord:task{task_id}"
            coord_data = redis_client.get(coord_key)

            if coord_data is None:
                print(f"[无人机速度] Redis中未找到坐标数据，键: {coord_key}")
                return 0.0

            # 解析坐标数据
            if isinstance(coord_data, bytes):
                coord_data = coord_data.decode('utf-8')

            coord_json = json.loads(coord_data)
            coord_info = coord_json.get("coord")

            if not coord_info:
                print(f"[无人机速度] 坐标数据无效: {coord_info}")
                return 0.0

            # 获取当前经纬度
            current_lat = coord_info.get('latitude')
            current_lng = coord_info.get('longitude')

            if current_lat is None or current_lng is None:
                print(f"[无人机速度] 坐标数据缺少经纬度: latitude={current_lat}, longitude={current_lng}")
                return 0.0

            # 获取上次经纬度
            last_coord = self.last_coords.get(task_id, {})
            last_lat = last_coord.get('latitude')
            last_lng = last_coord.get('longitude')

            # 比较经纬度是否相同
            coords_unchanged = (last_lat == current_lat and last_lng == current_lng)

            # 更新存储的上次坐标
            self.last_coords[task_id] = {'latitude': current_lat, 'longitude': current_lng}

            if coords_unchanged:
                print(f"[无人机速度] 经纬度未变化(latitude:{current_lat:.6f}, longitude:{current_lng:.6f})，无人机速度设为0")
                return 0.0

            # 经纬度发生变化，获取速度
            if 'speed' in coord_info:
                # 将 m/s 转换为 km/h
                speed_ms = float(coord_info['speed'])
                speed_kmh = speed_ms * 3.6

                print(f"[无人机速度] 经纬度变化(latitude:{last_lat:.6f},longitude:{last_lng:.6f} -> latitude:{current_lat:.6f},longitude:{current_lng:.6f})，获取成功: 转换前速度{speed_ms:.1f} m/s，转换后速度{speed_kmh:.1f} km/h")
                return speed_kmh
            else:
                print(f"[无人机速度] 坐标数据缺少速度字段: {coord_info}")
                return 0.0

        except json.JSONDecodeError as e:
            print(f"[无人机速度] JSON解析失败: {e}, 数据: {coord_data[:200] if coord_data else 'None'}")
            return 0.0
        except Exception as e:
            print(f"[无人机速度] 获取失败: {e}")
            return 0.0

    def process_lane_segmentation(self, segment_results: List[Dict[str, Any]], task_id: str) -> Dict[str, List[List[float]]]:
        """
        处理车道分割结果，提取车道掩码点位

        Args:
            segment_results: 分割结果列表 [{"xy": [[x,y],...], "class_id": int}, ...]
            task_id: 任务ID，用于数据隔离

        Returns:
            Dict[str, List[List[float]]]: {lane_type: [[x,y], ...]}
        """
        lane_masks = {"left_lane": [], "right_lane": []}

        for segment_item in segment_results:
            xy_coords = segment_item.get("xy")
            class_id = segment_item.get("class_id", 0)

            if xy_coords is not None and isinstance(xy_coords, list):
                lane_type = "left_lane" if class_id == 0 else "right_lane"
                lane_masks[lane_type].extend(xy_coords)

        # 按任务隔离缓存车道掩码
        if task_id not in self.task_lane_masks:
            self.task_lane_masks[task_id] = {}
        self.task_lane_masks[task_id] = lane_masks
        return lane_masks

    def is_point_in_lane(self, point: Tuple[float, float], lane_type: str, task_id: str) -> bool:
        """
        判断点是否在指定车道内

        Args:
            point: (x, y) 点坐标
            lane_type: 车道类型 "left_lane" 或 "right_lane"
            task_id: 任务ID，用于数据隔离

        Returns:
            bool: 是否在车道内
        """
        task_masks = self.task_lane_masks.get(task_id, {})
        if lane_type not in task_masks:
            return False

        lane_points = task_masks[lane_type]
        if len(lane_points) < 3:  # 需要至少3个点才能构成多边形
            return False

        try:
            # 转换为numpy数组
            contour = np.array(lane_points, dtype=np.float32)
            if contour.ndim == 2 and contour.shape[1] == 2:
                contour = contour.reshape(-1, 1, 2)

                # 使用OpenCV的pointPolygonTest判断点是否在多边形内
                result = cv2.pointPolygonTest(contour, point, False)
                return result >= 0  # >=0表示在多边形内或边界上
        except Exception as e:
            print(f"[车道检测] 点车道判断失败: {e}")

        return False

    def determine_vehicle_lane(self, vehicle_center: Tuple[float, float], task_id: str) -> Optional[str]:
        """
        根据车辆中心点位置判断所在车道

        Args:
            vehicle_center: (cx, cy) 车辆中心点坐标
            task_id: 任务ID，用于数据隔离

        Returns:
            str or None: "left_lane", "right_lane" 或 None
        """
        if self.is_point_in_lane(vehicle_center, "left_lane", task_id):
            return "left_lane"
        elif self.is_point_in_lane(vehicle_center, "right_lane", task_id):
            return "right_lane"

        return None

    def calculate_relative_speed(self, absolute_speed_kmh: float, vehicle_lane: Optional[str],
                               drone_speed_kmh: float) -> float:
        """
        根据车道信息计算相对速度

        Args:
            absolute_speed_kmh: 车辆绝对速度(km/h)
            vehicle_lane: 车辆所在车道 ("left_lane", "right_lane" 或 None)
            drone_speed_kmh: 无人机速度(km/h)

        Returns:
            float: 相对速度(km/h)
        """
        if vehicle_lane is None:
            # 无车道信息，返回绝对速度
            return absolute_speed_kmh

        if vehicle_lane == "left_lane":
            # 左侧车道：相对速度 = 车辆速度 - 无人机速度
            relative_speed = absolute_speed_kmh - drone_speed_kmh
        elif vehicle_lane == "right_lane":
            # 右侧车道：相对速度 = 车辆速度 + 无人机速度
            relative_speed = absolute_speed_kmh + drone_speed_kmh
        else:
            relative_speed = absolute_speed_kmh

        # 确保速度不小于0
        return max(0, relative_speed)

    def parse_vehicle_detection(self, vehicle_det) -> Optional[Dict[str, Any]]:
        """
        解析车辆检测结果

        Args:
            vehicle_det: 车辆检测结果（tensor格式: [x1, y1, x2, y2, conf, class_id, cxn, cyn, w, h, track_id]）

        Returns:
            dict or None: 解析后的车辆信息
        """
        try:
            # tensor格式 [x1, y1, x2, y2, conf, class_id, cxn, cyn, w, h, track_id]
            if len(vehicle_det) >= 11:
                x1, y1, x2, y2, conf, class_id, cxn, cyn, w, h, track_id = vehicle_det.tolist()[:11]
                track_id = int(track_id)

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                return {
                    "track_id": track_id,
                    "class_id": int(class_id),
                    "center": (cx, cy),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                }
        except Exception as e:
            print(f"[车辆解析] 解析失败: {e}")

        return None

    def process_frame(self, segment_data: List[Dict[str, Any]], vehicle_data: List,
                     absolute_speeds: Dict[int, float], task_id: str, redis_client) -> Dict[int, float]:
        """
        处理单帧数据，计算所有车辆的相对速度

        Args:
            segment_data: 车道分割数据
            vehicle_data: 车辆检测数据
            absolute_speeds: 车辆绝对速度 {track_id: speed_kmh}
            task_id: 任务ID
            redis_client: Redis客户端

        Returns:
            Dict[int, float]: {track_id: relative_speed_kmh}
        """
        # 1. 处理车道分割数据
        lane_masks = self.process_lane_segmentation(segment_data, task_id)

        # 2. 获取无人机速度
        drone_speed_kmh = self.get_drone_speed_from_redis(task_id, redis_client)

        # 3. 计算每辆车的相对速度
        relative_speeds = {}

        if vehicle_data is not None:
            # print(f"[测速跟踪] 处理 {len(vehicle_data)} 辆车辆")
            for i, vehicle_det in enumerate(vehicle_data):
                # print(f"[测速跟踪] 车辆{i}原始数据: {vehicle_det.tolist() if hasattr(vehicle_det, 'tolist') else vehicle_det}")

                vehicle_info = self.parse_vehicle_detection(vehicle_det)
                if vehicle_info is None:
                    print(f"[测速跟踪] 车辆{i}解析失败")
                    continue

                track_id = vehicle_info["track_id"]
                vehicle_center = vehicle_info["center"]

                # print(f"[测速跟踪] 车辆{i} -> track_id={track_id}, center={vehicle_center}")

                # 获取车辆绝对速度
                absolute_speed = absolute_speeds.get(track_id, 0.0)
                print(f"[DEBUG] track_id={track_id}, absolute_speeds dict: {absolute_speeds}, retrieved speed: {absolute_speed}")

                # 判断车辆所在车道
                vehicle_lane = self.determine_vehicle_lane(vehicle_center, task_id)

                # 计算相对速度
                relative_speed = self.calculate_relative_speed(
                    absolute_speed, vehicle_lane, drone_speed_kmh
                )

                relative_speeds[track_id] = relative_speed

                print(f"[测速跟踪] 车辆{track_id}: 绝对速度={absolute_speed:.1f}km/h, "
                      f"车道={vehicle_lane}, 无人机速度={drone_speed_kmh:.1f}km/h, "
                      f"相对速度={relative_speed:.1f}km/h")

        return relative_speeds



# 全局单例实例
_speed_tracker = None

def get_speed_tracker() -> VehicleSpeedTracker:
    """获取全局速度跟踪器实例"""
    global _speed_tracker
    if _speed_tracker is None:
        _speed_tracker = VehicleSpeedTracker()
    return _speed_tracker

def process_vehicle_speeds_with_lanes(
    segment_data: List[Dict[str, Any]],
    vehicle_data: List,
    absolute_speeds: Dict[int, float],
    task_id: str,
    redis_client
) -> Dict[int, float]:
    """
    处理车辆测速的主入口函数
    集成车道分割和车辆检测，计算相对速度

    Args:
        segment_data: 车道分割数据 [{"xy": [[x,y],...], "class_id": int}, ...]
        vehicle_data: 车辆检测数据 [tensor格式检测结果]
        absolute_speeds: 车辆绝对速度 {track_id: speed_kmh}
        task_id: 任务ID
        redis_client: Redis客户端

    Returns:
        Dict[int, float]: {track_id: relative_speed_kmh}
    """
    tracker = get_speed_tracker()
    return tracker.process_frame(segment_data, vehicle_data, absolute_speeds, task_id, redis_client)
