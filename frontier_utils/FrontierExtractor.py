#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""完整的边界提取器类（带ROS节点复用、边界点获取API）"""
import rospy
import numpy as np
import scipy.ndimage as ndimage
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


class FrontierExtractor:
    """ROS地图边界提取器类（带高效区域查询API）"""

    # 区域类型常量
    REGION_UNKNOWN = -1  # 未知区域
    REGION_FREE = 0  # 已知空闲区域
    REGION_OCCUPIED = 100  # 已知占用区域
    REGION_INVALID = -2  # 无效坐标（超出地图范围）

    def __init__(self):
        """初始化：复用已有的ROS节点，避免重复初始化"""
        # 关键：检查ROS节点是否已初始化，避免和RosTester冲突
        if not rospy.core.is_initialized():
            rospy.init_node('frontier_extractor', anonymous=True)
        else:
            rospy.loginfo("FrontierExtractor复用已初始化的ROS节点")

        # 配置参数（可通过ROS参数服务器配置）
        # 定义 “大连通区域” 的判定阈值（单位：米）。作用：如果一个连通边界区域的物理跨度（长 / 宽）超过该值，就用 KMeans 拆分成 2 个小边界点，避免单个大区域只输出 1 个边界点
        self.max_span = rospy.get_param('~max_span', 3.0)
        # DBSCAN 聚类的 “邻域半径”（单位：米）
        # ✅ 作用：距离 ≤ 该值的多个边界点，会被合并成 1 个中心点（你代码中注释了该逻辑，暂未生效）	- 默认 0.5（米）：
        # ✔️ 0.3：合并极近的边界点（输出更稀疏）；
        # ✔️ 0.8：合并更多邻近点，减少边界点数量；
        # - 若开启该逻辑：浅灰色边界点密集时，设 0.5 可避免边界点过多，同时保留核心边界
        self.dbscan_eps = rospy.get_param('~dbscan_eps', 1)
        # 过滤 “孤立噪声点”：只有连通区域的像素数 ≥ 该值，才会被保留为有效边界
        # ✅ 对浅灰色的影响：浅灰色边界往往是小面积连通区，该值过大会直接过滤掉	- 默认 4 → 建议改为 2~3：
        # ✔️ 2：保留极小的浅灰色边界（避免漏提）；
        # ✔️ 4：过滤小噪声，但可能漏提细窄的浅灰色边界；
        # - 不要设为 1：会提取大量单点噪声，导致边界点杂乱
        self.min_cluster_size = rospy.get_param('~min_cluster_size', 8)
        self.marker_scale = rospy.get_param('~marker_scale', 0.3)

        # ROS话题
        self._pub_frontier = rospy.Publisher(
            '/frontier_clusters', Marker, queue_size=10, latch=True
        )
        self._sub_map = rospy.Subscriber(
            '/map', OccupancyGrid, self._map_callback, queue_size=1
        )

        # 核心地图数据（内部维护）
        self._map_grid = None
        self._map_resolution = None
        self._map_origin_x = None
        self._map_origin_y = None
        self._map_width = None
        self._map_height = None
        self._latest_centroids = []  # 存储最新边界中心点

        self.FREE_THRESHOLD = 50
        self.OCCUPIED_THRESHOLD = 80

        # 等待地图数据加载
        self._wait_for_map()
        rospy.loginfo("边界提取器初始化完成（带区域查询API）")

    def _wait_for_map(self):
        """等待地图数据加载完成"""
        rospy.loginfo("等待/map话题数据...")
        while self._map_grid is None and not rospy.is_shutdown():
            try:
                # 尝试获取一次地图数据（避免依赖回调）
                msg = rospy.wait_for_message('/map', OccupancyGrid, timeout=5.0)
                self._map_callback(msg)
            except rospy.ROSException:
                rospy.logwarn("未接收到/map话题数据，继续等待...")
                rospy.sleep(1.0)

    def _map_callback(self, msg: OccupancyGrid):
        """地图回调：更新地图数据并提取边界点"""
        try:
            # 更新核心地图元数据
            self._map_resolution = msg.info.resolution
            self._map_origin_x = msg.info.origin.position.x
            self._map_origin_y = msg.info.origin.position.y
            self._map_width = msg.info.width
            self._map_height = msg.info.height
            self._map_grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))

            # 提取边界点
            frontier_mask = self._extract_frontier_mask(self._map_grid)
            centroids = self._cluster_frontier_regions(
                frontier_mask, self._map_resolution, self._map_origin_x, self._map_origin_y
            )

            final_centroids = self._merge_close_centroids(centroids)
            # final_centroids = centroids

            self._latest_centroids = final_centroids
            self._publish_frontier_markers(final_centroids, msg.header.frame_id)

        except Exception as e:
            rospy.logerr(f"处理地图时出错: {str(e)}")

    def _extract_frontier_mask(self, grid: np.ndarray) -> np.ndarray:
        """提取边界点掩码（空闲区域且邻域包含未知区域）"""
        frontier_mask = np.zeros_like(grid, dtype=bool)
        for y in range(1, grid.shape[0] - 1):
            for x in range(1, grid.shape[1] - 1):
                # 原来：if grid[y, x] == 0 and -1 in grid[y-1:y+2, x-1:x+2]:
                # 现在：0~50 都算可走/边界，把浅灰色包含进来
                if 0 <= grid[y, x] <= 50 and -1 in grid[y - 1:y + 2, x - 1:x + 2]:
                    frontier_mask[y, x] = True
        return frontier_mask

    def _cluster_frontier_regions(self, mask: np.ndarray, res: float, origin_x: float, origin_y: float) -> list:
        """连通区域分析，大区域用K-Means拆分"""
        labeled_array, num_regions = ndimage.label(mask, structure=np.ones((3, 3)))
        centroids = []
        for region_id in range(1, num_regions + 1):
            y_coords, x_coords = np.where(labeled_array == region_id)
            if len(y_coords) < self.min_cluster_size:
                continue
            cluster_pixels = np.column_stack((x_coords, y_coords))
            span_pixels = np.max(cluster_pixels, axis=0) - np.min(cluster_pixels, axis=0)
            max_span_meters = np.max(span_pixels) * res
            if max_span_meters > self.max_span:
                kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(cluster_pixels)
                for sub_label in range(2):
                    sub_cluster = cluster_pixels[kmeans.labels_ == sub_label]
                    if len(sub_cluster) > 0:
                        center = np.mean(sub_cluster, axis=0)
                        centroids.append([center[0] * res + origin_x, center[1] * res + origin_y])
            else:
                center = np.mean(cluster_pixels, axis=0)
                centroids.append([center[0] * res + origin_x, center[1] * res + origin_y])
        return centroids

    def _merge_close_centroids(self, centroids: list) -> list:
        """DBSCAN融合近距离聚类中心"""
        if not centroids:
            return []
        centroids_np = np.array(centroids)
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=1).fit(centroids_np)
        final_centroids = []
        for label in set(dbscan.labels_):
            cluster_points = centroids_np[dbscan.labels_ == label]
            final_centroids.append(np.mean(cluster_points, axis=0))
        return final_centroids

    def _publish_frontier_markers(self, centroids: list, frame_id: str):
        """发布边界点可视化Marker"""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "frontier_clusters"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration()
        marker.scale.x = self.marker_scale
        marker.scale.y = self.marker_scale
        marker.scale.z = self.marker_scale
        marker.color.b = 1.0
        marker.color.a = 1.0
        for point in centroids:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 0.1
            marker.points.append(p)
        self._pub_frontier.publish(marker)

    # ==================== 对外提供的核心API ====================
    def get_latest_frontier_centroids(self):
        """获取最新的边界中心点列表（返回副本，避免外部修改）"""
        return [list(centroid) for centroid in self._latest_centroids]

    def get_region_type(self, world_x: float, world_y: float) -> int:
        """查询世界坐标的区域类型"""
        x_pixel, y_pixel = self._world_to_pixel(world_x, world_y)
        if x_pixel == -1 or y_pixel == -1:
            return self.REGION_INVALID
        grid_value = self._map_grid[y_pixel, x_pixel]

        if grid_value == -1:
            return self.REGION_UNKNOWN
        # 这里也改成区间判断
        elif 0 <= grid_value <= self.FREE_THRESHOLD:
            return self.REGION_FREE
        elif grid_value >= self.OCCUPIED_THRESHOLD:  # 深灰/黑色算占用
            return self.REGION_OCCUPIED
        else:
            return self.REGION_UNKNOWN

    def _world_to_pixel(self, world_x: float, world_y: float) -> tuple:
        """世界坐标转像素坐标"""
        if None in [self._map_resolution, self._map_origin_x, self._map_origin_y,
                    self._map_width, self._map_height]:
            return -1, -1
        x_pixel = int((world_x - self._map_origin_x) / self._map_resolution)
        y_pixel = int((world_y - self._map_origin_y) / self._map_resolution)
        if 0 <= x_pixel < self._map_width and 0 <= y_pixel < self._map_height:
            return x_pixel, y_pixel
        else:
            return -1, -1

    def pixel_to_world(self, x_pixel: int, y_pixel: int) -> tuple:
        """像素坐标转世界坐标（补充API）"""
        if None in [self._map_resolution, self._map_origin_x, self._map_origin_y]:
            return None, None
        world_x = x_pixel * self._map_resolution + self._map_origin_x
        world_y = y_pixel * self._map_resolution + self._map_origin_y
        return world_x, world_y

    def is_unknown_region(self, world_x: float, world_y: float) -> bool:
        """判断是否为未知区域"""
        return self.get_region_type(world_x, world_y) == self.REGION_UNKNOWN