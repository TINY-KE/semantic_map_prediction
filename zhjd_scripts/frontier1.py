#!/usr/bin/env python
import rospy
import numpy as np
import scipy.ndimage as ndimage
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class FrontierExtractor:
    def __init__(self):
        rospy.init_node('frontier_extractor')
        self.pub = rospy.Publisher('/frontier_clusters', Marker, queue_size=10)
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        rospy.loginfo("基于连通区域的边界提取器已启动...")

    def map_callback(self, msg):
        res = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))

        # 1. 识别边界点 (Mask)
        # 这里的逻辑：当前点是空闲(0)，且周围邻域内有未知(-1)
        # 使用图像膨胀/腐蚀来加速边界获取，或者简单的矩阵切片
        frontier_mask = np.zeros_like(grid, dtype=bool)

        # 快速判断：利用 NumPy 切片获取周围邻域
        # 这里的循环是性能瓶颈，但在 Python 中，对中等地图尚可接受
        for y in range(1, msg.info.height - 1):
            for x in range(1, msg.info.width - 1):
                if grid[y, x] == 0:
                    if -1 in grid[y - 1:y + 2, x - 1:x + 2]:
                        frontier_mask[y, x] = True

        # 2. 连通区域标记 (Connected Components)
        # structure 定义 8 邻域连通
        labeled_array, num_features = ndimage.label(frontier_mask, structure=np.ones((3, 3)))

        # 3. 计算每个连通块的中心
        centroids = []
        for i in range(1, num_features + 1):
            y_coords, x_coords = np.where(labeled_array == i)

            # 过滤过小的孤立噪声点（可选）
            if len(y_coords) < 5: continue

            center_y = np.mean(y_coords)
            center_x = np.mean(x_coords)

            world_x = center_x * res + origin_x
            world_y = center_y * res + origin_y
            centroids.append([world_x, world_y])

        # 4. 发布结果
        self.publish_markers(centroids, msg.header.frame_id)
        rospy.loginfo(f"检测到 {num_features} 个连通区域，发布 {len(centroids)} 个目标")

    def publish_markers(self, centroids, frame_id):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.b = 1.0  # 红色点
        marker.color.a = 1.0
        for c in centroids:
            marker.points.append(Point(c[0], c[1], 0.1))
        self.pub.publish(marker)


if __name__ == '__main__':
    try:
        FrontierExtractor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass