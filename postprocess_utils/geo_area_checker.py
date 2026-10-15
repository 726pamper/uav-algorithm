"""
不需要ALGO_ID, 属于可随机调用插件,功能是进行经纬度区域判定，基于平台绘制好的违禁区域与飞机实时的经纬度进行判定
地理位置区域检测模块
用于检测无人机是否在违禁区域内，支持多边形、圆形等形状的区域检测
"""

import json
import threading
import logging
from typing import Dict, List, Optional, Tuple
import paho.mqtt.client as mqtt
from shapely.geometry import Point, Polygon, shape
from shapely.ops import unary_union


logger = logging.getLogger(__name__)


class GeoAreaChecker:
    """
    地理位置区域检测器
    支持订阅MQTT获取实时无人机位置，并判断是否在违禁区域内
    """

    def __init__(self, mqtt_client: mqtt.Client):
        """
        初始化地理区域检测器
        
        Args:
            mqtt_client: MQTT客户端实例
        """
        self._mqtt_client = mqtt_client
        self._position_cache: Dict[str, Dict[str, float]] = {}  # {device_sn: {latitude, longitude}}
        self._speed_cache: Dict[str, float] = {}  # {device_sn: speed}
        self._subscription_lock = threading.Lock()
        self._subscribed_devices: set = set()  # 已订阅的设备集合
        self._area_cache: Dict[str, List[Dict]] = {}  # {uavSN: area_list}
        
    def subscribe_device_position(self, device_sn: str, mqtt_client: mqtt.Client = None):
        """
        订阅指定设备的位置信息
        
        Args:
            device_sn: 无人机SN码
            mqtt_client: 可选的MQTT客户端实例，如果为None则使用初始化时的客户端
        """
        if mqtt_client is None:
            mqtt_client = self._mqtt_client
            
        with self._subscription_lock:
            if device_sn in self._subscribed_devices:
                logger.warning(f"Device {device_sn} already subscribed")
                return
            
            topic = f"thing/product/{device_sn}/osd"
            try:
                mqtt_client.subscribe(topic, qos=1)
                self._subscribed_devices.add(device_sn)
                logger.info(f"Subscribed to device position topic: {topic}")
            except Exception as e:
                logger.error(f"Failed to subscribe to {topic}: {e}")

    def unsubscribe_device_position(self, device_sn: str, mqtt_client: mqtt.Client = None):
        """
        取消订阅指定设备的位置信息
        
        Args:
            device_sn: 无人机SN码
            mqtt_client: 可选的MQTT客户端实例
        """
        if mqtt_client is None:
            mqtt_client = self._mqtt_client
            
        with self._subscription_lock:
            if device_sn not in self._subscribed_devices:
                return
            
            topic = f"thing/product/{device_sn}/osd"
            try:
                mqtt_client.unsubscribe(topic)
                self._subscribed_devices.discard(device_sn)
                logger.info(f"Unsubscribed from device position topic: {topic}")
            except Exception as e:
                logger.error(f"Failed to unsubscribe from {topic}: {e}")

    def update_position(self, device_sn: str, latitude: float, longitude: float):
        """
        更新设备位置缓存
        
        Args:
            device_sn: 无人机SN码
            latitude: 纬度
            longitude: 经度
        """
        self._position_cache[device_sn] = {
            "latitude": latitude,
            "longitude": longitude
        }
        logger.debug(f"Updated position for {device_sn}: ({latitude}, {longitude})")

    def get_position(self, device_sn: str) -> Optional[Tuple[float, float]]:
        """
        获取设备的最后已知位置

        Args:
            device_sn: 无人机SN码

        Returns:
            (latitude, longitude) 或 None
        """
        pos = self._position_cache.get(device_sn)
        if pos:
            return (pos.get("latitude"), pos.get("longitude"))
        return None

    def update_speed(self, device_sn: str, speed: float):
        """
        更新设备速度缓存

        Args:
            device_sn: 无人机SN码
            speed: 速度值
        """
        self._speed_cache[device_sn] = speed
        logger.debug(f"Updated speed for {device_sn}: {speed}")

    def get_speed(self, device_sn: str) -> Optional[float]:
        """
        获取设备的最后已知速度

        Args:
            device_sn: 无人机SN码

        Returns:
            速度值或None
        """
        return self._speed_cache.get(device_sn)

    def parse_area_from_message(self, area_json_str: str) -> Optional[List[Dict]]:
        """
        从消息字符串解析area信息
        
        Args:
            area_json_str: area的JSON字符串
            
        Returns:
            解析后的area列表或None
        """
        try:
            if isinstance(area_json_str, str):
                area_list = json.loads(area_json_str)
            else:
                area_list = area_json_str
            
            if isinstance(area_list, list):
                return area_list
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse area JSON: {e}")
            return None

    def cache_area(self, uav_sn: str, area_list: List[Dict]):
        """
        缓存无人机的违禁区域信息
        
        Args:
            uav_sn: 无人机SN码
            area_list: 区域列表
        """
        self._area_cache[uav_sn] = area_list
        logger.debug(f"Cached area for UAV {uav_sn}: {len(area_list)} areas")

    def get_cached_area(self, uav_sn: str) -> Optional[List[Dict]]:
        """
        获取缓存的违禁区域信息
        
        Args:
            uav_sn: 无人机SN码
            
        Returns:
            区域列表或None
        """
        return self._area_cache.get(uav_sn)

    def is_point_in_polygon(self, latitude: float, longitude: float, 
                           polygon_coords: List[List[float]]) -> bool:
        """
        检查点是否在多边形内（用于Polygon类型区域）
        
        Args:
            latitude: 纬度
            longitude: 经度
            polygon_coords: 多边形坐标列表 [[lon, lat], [lon, lat], ...]
            
        Returns:
            True 如果点在多边形内，False 否则
        """
        try:
            # 确保坐标格式正确 (longitude, latitude)
            if not polygon_coords :
                logger.warning("Invalid polygon coordinates")
                return False
            
            # 创建点和多边形
            point = Point(longitude, latitude)
            polygon = Polygon(polygon_coords)
            
            # 检查点是否在多边形内
            return polygon.contains(point) or polygon.touches(point)
        except Exception as e:
            logger.error(f"Error checking point in polygon: {e}")
            return False

    def is_point_in_circle(self, latitude: float, longitude: float,
                          center_lat: float, center_lon: float, 
                          radius_m: float) -> bool:
        """
        检查点是否在圆形区域内
        
        Args:
            latitude: 目标点纬度
            longitude: 目标点经度
            center_lat: 圆心纬度
            center_lon: 圆心经度
            radius_m: 圆的半径（米）
            
        Returns:
            True 如果点在圆形区域内，False 否则
        """
        try:
            # 使用Haversine公式计算两点间距离
            from math import radians, cos, sin, asin, sqrt
            
            lat1, lon1, lat2, lon2 = map(radians, [latitude, longitude, center_lat, center_lon])
            
            # Haversine公式
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * asin(sqrt(a))
            
            # 地球半径（米）
            r = 6371000
            distance = c * r
            
            return distance <= radius_m
        except Exception as e:
            logger.error(f"Error checking point in circle: {e}")
            return False

    def is_point_near_line(self, latitude: float, longitude: float,
                           line_coords: List[List[float]],
                           buffer_width_m: float = 10.0) -> bool:
        """
        检查点是否在线段附近（buffer 范围内）。

        支持折线：点到任意线段的最短距离 ≤ buffer_width 即视为在线上。

        Args:
            latitude: 目标点纬度
            longitude: 目标点经度
            line_coords: 线段坐标 [[lon, lat], [lon, lat], ...]
            buffer_width_m: 线宽（米），点距线小于此距离视为命中

        Returns:
            True 如果点在线的 buffer 范围内
        """
        try:
            from math import radians, cos, sin, asin, sqrt, degrees, acos

            if not line_coords or len(line_coords) < 2:
                return False

            point_lat, point_lon = radians(latitude), radians(longitude)
            R = 6371000  # 地球半径（米）

            for i in range(len(line_coords) - 1):
                lon1, lat1 = line_coords[i]
                lon2, lat2 = line_coords[i + 1]
                lat1, lon1 = radians(lat1), radians(lon1)
                lat2, lon2 = radians(lat2), radians(lon2)

                # 计算点到线段的大圆距离（近似：用平面几何在局部坐标下计算）
                dlat_line = lat2 - lat1
                dlon_line = lon2 - lon1
                dlat_p = point_lat - lat1
                dlon_p = point_lon - lon1

                # 转换为近似米制坐标（以点1为原点）
                y1 = 0.0
                x1 = 0.0
                y2 = dlat_line * R
                x2 = dlon_line * cos((lat1 + lat2) / 2) * R
                yp = dlat_p * R
                xp = dlon_p * cos((lat1 + point_lat) / 2) * R

                # 点到线段的最短距离
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    dist = sqrt((xp - x1) ** 2 + (yp - y1) ** 2)
                else:
                    t = ((xp - x1) * dx + (yp - y1) * dy) / (dx * dx + dy * dy)
                    t = max(0.0, min(1.0, t))
                    proj_x = x1 + t * dx
                    proj_y = y1 + t * dy
                    dist = sqrt((xp - proj_x) ** 2 + (yp - proj_y) ** 2)

                if dist <= buffer_width_m:
                    return True

            return False
        except Exception as e:
            logger.error(f"Error checking point near line: {e}")
            return False

    def is_point_in_sector(self, latitude: float, longitude: float,
                           center_lat: float, center_lon: float,
                           radius_m: float, start_angle: float,
                           end_angle: float) -> bool:
        """
        检查点是否在扇形区域内。

        Args:
            latitude: 目标点纬度
            longitude: 目标点经度
            center_lat: 扇形圆心纬度
            center_lon: 扇形圆心经度
            radius_m: 扇形半径（米）
            start_angle: 起始角度（度，0=正北，顺时针）
            end_angle: 终止角度（度，0=正北，顺时针）

        Returns:
            True 如果点在扇形内
        """
        try:
            from math import radians, cos, sin, asin, sqrt, degrees, atan2

            # 先检查距离
            if not self.is_point_in_circle(latitude, longitude,
                                           center_lat, center_lon, radius_m):
                return False

            # 计算方位角（从圆心到目标点）
            lat1, lon1 = radians(center_lat), radians(center_lon)
            lat2, lon2 = radians(latitude), radians(longitude)

            dlon = lon2 - lon1
            y = sin(dlon) * cos(lat2)
            x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
            bearing = degrees(atan2(y, x))
            bearing = (bearing + 360) % 360  # 0=正北，顺时针

            # 归一化角度范围
            start = start_angle % 360
            end = end_angle % 360

            if start <= end:
                return start <= bearing <= end
            else:
                # 跨越 0 度的情况 (如 350° ~ 10°)
                return bearing >= start or bearing <= end

        except Exception as e:
            logger.error(f"Error checking point in sector: {e}")
            return False

    def check_position_in_area(self, latitude: float, longitude: float,
                              area_list: List[Dict]) -> Tuple[bool, List[str]]:
        """
        检查位置是否在任何违禁区域内

        Args:
            latitude: 无人机纬度
            longitude: 无人机经度
            area_list: 区域列表

        Returns:
            (是否在违禁区域, 违禁区域名称列表)
        """
        violated_areas = []

        if not area_list:
            return False, []

        try:
            for area in area_list:
                if not isinstance(area, dict):
                    continue

                shape_type = area.get("shapeType", "").lower()
                area_name = area.get("name", "Unknown")
                area_id = area.get("id", "")
                geometry = area.get("geometryData", {})

                is_violated = False
                if shape_type == "polygon":
                    # 处理多边形区域
                    coordinates = geometry.get("coordinates", [])
                    if coordinates and len(coordinates) >= 3:
                        # coordinates格式: [[lon, lat], [lon, lat], ...]
                        is_violated = self.is_point_in_polygon(
                            latitude, longitude, coordinates
                        )

                elif shape_type == "circle":
                    # 处理圆形区域
                    point_list = geometry.get("point", [])
                    radius = geometry.get("radius", 0)

                    if point_list and len(point_list) > 0 and radius > 0:
                        center_lon, center_lat = point_list[0]
                        is_violated = self.is_point_in_circle(
                            latitude, longitude, center_lat, center_lon, radius
                        )

                elif shape_type == "line":
                    # 处理线型区域（折线 + buffer 宽度）
                    coordinates = geometry.get("coordinates", [])
                    buffer_width = geometry.get("width", geometry.get("radius", 10.0))
                    if coordinates and len(coordinates) >= 2:
                        is_violated = self.is_point_near_line(
                            latitude, longitude, coordinates, buffer_width
                        )

                elif shape_type == "sector":
                    # 处理扇形区域
                    point_list = geometry.get("point", [])
                    radius = geometry.get("radius", 0)
                    start_angle = geometry.get("startAngle", 0)
                    end_angle = geometry.get("endAngle", 0)
                    if point_list and len(point_list) > 0 and radius > 0:
                        center_lon, center_lat = point_list[0]
                        is_violated = self.is_point_in_sector(
                            latitude, longitude, center_lat, center_lon,
                            radius, start_angle, end_angle
                        )

                if is_violated:
                    violated_areas.append(f"{area_name}(ID:{area_id})")
                    logger.warning(f"Position violation detected in area: {area_name} (ID:{area_id})")
        
        except Exception as e:
            logger.error(f"Error checking position in area: {e}")
        
        return len(violated_areas) > 0, violated_areas

    def handle_mqtt_message(self, topic: str, payload: str, uav_sn: str = None):
        """
        处理来自MQTT的OSD消息

        Args:
            topic: MQTT主题
            payload: 消息负载
            uav_sn: 无人机SN码（用于查询缓存的area）

        Returns:
            (是否违禁, 违禁区域列表, 位置和速度信息)
            位置和速度信息包含: {"device_sn": str, "latitude": float, "longitude": float, "speed": float}
        """
        try:
            msg_data = json.loads(payload)
            
            # 从OSD消息中提取sub_device信息
            sub_device = msg_data.get("sub_device", {})
            device_sn = sub_device.get("device_sn")
            
            if not device_sn:
                logger.warning("No device_sn found in OSD message")
                return False, [], None
            
            # 提取位置信息
            latitude = msg_data.get("latitude")
            longitude = msg_data.get("longitude")

            if latitude is None or longitude is None:
                logger.warning(f"No position data in OSD message for {device_sn}")
                return False, [], None

            # 提取速度信息
            speed = msg_data.get("speed")

            # 更新位置和速度缓存
            self.update_position(device_sn, latitude, longitude)
            if speed is not None:
                self.update_speed(device_sn, speed)
            
            # 获取缓存的area信息
            if uav_sn:
                area_list = self.get_cached_area(uav_sn)
                if area_list:
                    is_violated, violated_areas = self.check_position_in_area(
                        latitude, longitude, area_list
                    )
                    return is_violated, violated_areas, {
                        "device_sn": device_sn,
                        "latitude": latitude,
                        "longitude": longitude,
                        "speed": speed
                    }

            return False, [], {
                "device_sn": device_sn,
                "latitude": latitude,
                "longitude": longitude,
                "speed": speed
            }
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse MQTT message: {e}")
            return False, [], None
        except Exception as e:
            logger.error(f"Error handling MQTT message: {e}")
            return False, [], None


def create_geo_checker_with_subscription(mqtt_client: mqtt.Client, 
                                       uav_sn: str,
                                       area_info: str) -> GeoAreaChecker:
    """
    创建GeoAreaChecker实例并自动订阅设备位置
    
    Args:
        mqtt_client: MQTT客户端实例
        uav_sn: 无人机SN码
        area_info: area的JSON字符串
        
    Returns:
        配置好的GeoAreaChecker实例
    """
    checker = GeoAreaChecker(mqtt_client)
    
    # 解析area信息
    area_list = checker.parse_area_from_message(area_info)
    if area_list:
        checker.cache_area(uav_sn, area_list)
        # 订阅UAV的OSD主题以获取sub_device信息
        checker.subscribe_device_position(uav_sn, mqtt_client)
        logger.info(f"GeoAreaChecker initialized for UAV {uav_sn} with {len(area_list)} areas")
    else:
        logger.warning(f"Failed to parse area information for UAV {uav_sn}")
    
    return checker
