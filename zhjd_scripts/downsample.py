#!/usr/bin/env python
import rospy
import numpy as np
import scipy.ndimage as ndimage
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from sklearn.cluster import KMeans, DBSCAN


class DownSampler:
    def __init__(self):
        rospy.init_node('frontier_extractor')
        self.pub_markers = rospy.Publisher('/frontier_clusters', Marker, queue_size=10)
        # 发布降采样后的新地图
        self.pub_map = rospy.Publisher('/map_downsampled', OccupancyGrid, queue_size=1)
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)

        self.target_res = 0.1
        self.max_span = 2.0
        rospy.loginfo("高效边界提取与地图降采样节点已启动...")

    def map_callback(self, msg):
        # 1. 降采样地图
        orig_res = msg.info.resolution
        scale = orig_res / self.target_res
        grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))

        # 使用最近邻插值缩放地图 (order=0)
        downsampled_grid = ndimage.zoom(grid, scale, order=0)

        # 空闲区域的mask
        free_space_mask = (downsampled_grid == 0)

        # 发布新地图供 Rviz 查看
        new_map = OccupancyGrid()
        new_map.header = msg.header
        new_map.info = msg.info
        new_map.info.resolution = self.target_res
        new_map.info.width = downsampled_grid.shape[1]
        new_map.info.height = downsampled_grid.shape[0]
        new_map.data = downsampled_grid.flatten().astype(np.int8).tolist()
        self.pub_map.publish(new_map)

        # 2. 识别边界点 (基于降采样后的地图)
        frontier_mask = np.zeros_like(downsampled_grid, dtype=bool)
        for y in range(1, new_map.info.height - 1):
            for x in range(1, new_map.info.width - 1):
                if downsampled_grid[y, x] == 0:
                    if -1 in downsampled_grid[y - 1:y + 2, x - 1:x + 2]:
                        frontier_mask[y, x] = True

        # 3. 膨胀与连通性标记
        dilated_mask = ndimage.binary_dilation(frontier_mask, iterations=1)
        labeled_array, num_features = ndimage.label(dilated_mask, structure=np.ones((3, 3)))

        # 4. 计算中心点 (K-Means 切分)
        centroids = []
        for i in range(1, num_features + 1):
            y_coords, x_coords = np.where(labeled_array == i)
            if len(y_coords) < 5: continue
            points = np.column_stack((x_coords, y_coords))

            span = np.max(points, axis=0) - np.min(points, axis=0)
            if np.max(span) * self.target_res > self.max_span:
                kmeans = KMeans(n_clusters=2, n_init=10).fit(points)
                for lbl in range(2):
                    c = np.mean(points[kmeans.labels_ == lbl], axis=0)
                    centroids.append([c[0] * self.target_res + msg.info.origin.position.x,
                                      c[1] * self.target_res + msg.info.origin.position.y])
            else:
                c = np.mean(points, axis=0)
                centroids.append([c[0] * self.target_res + msg.info.origin.position.x,
                                  c[1] * self.target_res + msg.info.origin.position.y])

        # 5. 二轮融合处理 (距离小于 0.5 米的融合)
        if centroids:
            final = []
            db = DBSCAN(eps=0.5, min_samples=1).fit(centroids)
            for lbl in set(db.labels_):
                final.append(np.mean(np.array(centroids)[db.labels_ == lbl], axis=0))
            self.publish_markers(final, msg.header.frame_id)

    def publish_markers(self, centroids, frame_id):
        m = Marker()
        m.header.frame_id = frame_id
        m.type = Marker.SPHERE_LIST
        m.scale.x = m.scale.y = m.scale.z = 0.3
        m.color.r = 1.0;
        m.color.a = 1.0
        for c in centroids: m.points.append(Point(c[0], c[1], 0.1))
        self.pub_markers.publish(m)


if __name__ == '__main__':
    try:
        DownSampler()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass