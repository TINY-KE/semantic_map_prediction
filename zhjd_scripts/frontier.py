#!/usr/bin/env python
import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from sklearn.cluster import DBSCAN  # 使用聚类算法


class EfficientFrontierExtractor:
    def __init__(self):
        rospy.init_node('efficient_frontier')
        self.pub = rospy.Publisher('/frontier_clusters', Marker, queue_size=10)
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)

        rospy.loginfo("高效边界提取器已启动...")

    def map_callback(self, msg):
        res, origin_x, origin_y = msg.info.resolution, msg.info.origin.position.x, msg.info.origin.position.y
        grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))

        # 1. 提取原始点
        points = []
        step = 1
        for y in range(1, msg.info.height - 1, step):  # 更大的采样步长以提高效率
            for x in range(1, msg.info.width - 1, step):
                if grid[y, x] == 0:
                    if -1 in grid[y - 1:y + 2, x - 1:x + 2]:
                        points.append([x * res + origin_x, y * res + origin_y])

        if not points: return

        # 2. 进行聚类
        points = np.array(points)
        points = np.array(points)
        # DBSCAN参数：
        # eps为邻域范围
        # min_samples： 一个区域内至少需要多少个点，才能被算法认定为一个“核心簇”（即一个有意义的区域）。
        # self.dbscan = DBSCAN(eps=1.0, min_samples=2)
        self.dbscan = DBSCAN(eps=0.2, min_samples=2)
        labels = self.dbscan.fit_predict(points)

        # 3. 计算每个聚类的中心点 (作为最佳导航目标)
        centroids = []
        for label in set(labels):
            cluster = points[labels == label]

            if label == -1:
                # 如果是 -1，说明这些点原本被判定为噪点
                # 我们不能取它们的均值，而是把每一个点都当成一个独立的“探索目标”
                for point in cluster:
                    centroids.append(point)
            else:
                # 如果不是 -1，说明是聚类好的区域，取均值
                centroids.append(np.mean(cluster, axis=0))

            # for point in cluster:
            #     centroids.append(point)

        # 4. 可视化聚类中心
        self.publish_markers(centroids, msg.header.frame_id)
        rospy.loginfo(f"原始边界点数: {len(points)}, 聚类后导航目标数: {len(centroids)}")

    def publish_markers(self, centroids, frame_id):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x, marker.scale.y, marker.scale.z = 0.3, 0.3, 0.3
        marker.color.b, marker.color.a = 1.0, 1.0  # 红色点，代表探索目标
        for c in centroids:
            marker.points.append(Point(c[0], c[1], 0.2))
        self.pub.publish(marker)
        rospy.loginfo(f"正在发布目标点坐标: {c[0]}, {c[1]}")


if __name__ == '__main__':
    try:
        EfficientFrontierExtractor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass