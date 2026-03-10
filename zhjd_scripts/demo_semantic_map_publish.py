#!/usr/bin/env python
import rospy
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


class SemanticMarkerPublisher:
    def __init__(self):
        rospy.init_node('semantic_marker_node')
        self.pub = rospy.Publisher('/semantic_markers', Marker, queue_size=1)
        rospy.loginfo("语义 Marker 发布器已启动...")

    def publish_semantic_map(self, semantic_grid, res, origin_x, origin_y, height):
        """
        semantic_grid: [300, 300] numpy array (0-26)
        res: 分辨率
        """
        # 创建一个复合 Marker，包含 27 个 Cube_List
        # Rviz 中可以为每一种语义类分配颜色
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.scale.x = res
        marker.scale.y = res
        marker.scale.z = 0.01  # 设为 0.05 可以让它看起来像一层薄膜

        # 定义颜色映射 (27种类别，这里随机生成)
        # 你可以根据实际含义定义：例如 0=空闲(白色), 1=墙(灰色)等
        np.random.seed(42)
        color_palette = np.random.rand(27, 4)

        # 遍历地图，将属于类别的点加入对应的 Marker 中
        # 为了高效，我们将 Marker 点集存入
        for cat in range(27):
            y_indices, x_indices = np.where(semantic_grid == cat)
            for y, x in zip(y_indices, x_indices):
                # 转换为世界坐标
                px = x * res + origin_x
                py = y * res + origin_y

                marker.points.append(Point(px, py, height))
                # 设置颜色
                color = ColorRGBA(color_palette[cat, 0], color_palette[cat, 1],
                                  color_palette[cat, 2], 0.8)
                marker.colors.append(color)

        self.pub.publish(marker)


# 示例调用逻辑
if __name__ == '__main__':
    # 模拟一个 300x300 的语义栅格地图
    semantic_map = np.random.randint(0, 27, (300, 300))

    pub = SemanticMarkerPublisher()
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        pub.publish_semantic_map(semantic_map, 0.1, -15.0, -15.0, 1)
        rate.sleep()