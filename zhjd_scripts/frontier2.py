#!/usr/bin/env python
import rospy
import numpy as np
import scipy.ndimage as ndimage
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN # 使用聚类算法

class FrontierExtractor:
    def __init__(self):
        rospy.init_node('frontier_extractor')
        self.pub = rospy.Publisher('/frontier_clusters', Marker, queue_size=10)
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        # 最大允许的边界区域跨度（单位：米）
        self.max_span = 2.0
        rospy.loginfo("基于连通区域与K-Means拆分的边界提取器已启动...")

    def map_callback(self, msg):
        res = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))

        # 1. 识别边界点 (Mask)
        frontier_mask = np.zeros_like(grid, dtype=bool)
        for y in range(1, msg.info.height - 1):
            for x in range(1, msg.info.width - 1):
                if grid[y, x] == 0:
                # if grid[y, x] != -1:
                    if -1 in grid[y - 1:y + 2, x - 1:x + 2]:
                        frontier_mask[y, x] = True

        # 2. 连通区域标记
        labeled_array, num_features = ndimage.label(frontier_mask, structure=np.ones((3, 3)))

        # 3. 计算每个连通块的中心（支持递归拆分）
        centroids = []
        for i in range(1, num_features + 1):
            y_coords, x_coords = np.where(labeled_array == i)
            if len(y_coords) < 4: continue

            # 将像素坐标转为 (x, y) 矩阵
            cluster_points = np.column_stack((x_coords, y_coords))

            # 计算当前连通块的跨度
            span = np.max(cluster_points, axis=0) - np.min(cluster_points, axis=0)
            max_span_meters = np.max(span) * res

            # 如果区域跨度过大，使用 K-Means 拆分
            if max_span_meters > self.max_span:
                kmeans = KMeans(n_clusters=2, n_init=10).fit(cluster_points)
                for sub_label in range(2):
                    sub_cluster = cluster_points[kmeans.labels_ == sub_label]
                    center = np.mean(sub_cluster, axis=0)
                    centroids.append([center[0] * res + origin_x, center[1] * res + origin_y])
            else:
                center = np.mean(cluster_points, axis=0)
                centroids.append([center[0] * res + origin_x, center[1] * res + origin_y])

        # 4. --- 第二轮处理：融合距离小于 0.5 米的目标点 ---
        if len(centroids) > 0:
            centroids = np.array(centroids)
            # 使用 DBSCAN 对中心点进行二次聚类，eps=0.5 即 0.5 米
            # min_samples=1 保证即使只有一个点，它也会被保留，不会丢失目标
            second_clustering = DBSCAN(eps=0.5, min_samples=1).fit(centroids)

            final_centroids = []
            for label in set(second_clustering.labels_):
                # 提取该簇内所有的点
                group = centroids[second_clustering.labels_ == label]
                # 合并这些点：取它们的中心作为最终目标
                final_centroids.append(np.mean(group, axis=0))

            centroids = final_centroids

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
        marker.color.b = 1.0
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