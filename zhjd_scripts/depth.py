#!/usr/bin/env python
# coding: utf-8

import rospy
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# 图像尺寸与相机内参
W = 128
H = 128
hfov_deg = 79
hfov = hfov_deg * np.pi / 180.
fx = (W / 2) / np.tan(hfov / 2)
fy = fx
cx = W / 2
cy = H / 2

# 颜色映射（类别 ID → RGB）
color_mapping_27 = {
    0:  (255, 255, 255),   # white
    1:  (128, 128, 0),     # olive
    2:  (0, 0, 255),       # blue
    3:  (255, 0, 0),       # red
    4:  (255, 0, 255),     # magenta
    5:  (0, 255, 255),     # cyan
    6:  (255, 165, 0),     # orange
    7:  (255, 255, 0),     # yellow
    8:  (128, 128, 128),   # gray
    9:  (128, 0, 0),       # maroon
    10: (255, 20, 147),    # deep pink
    11: (0, 128, 0),       # dark green
    12: (128, 0, 128),     # purple
    13: (0, 128, 128),     # teal
    14: (0, 0, 128),       # navy
    15: (210, 105, 30),    # chocolate
    16: (188, 143, 143),   # rosy brown
    17: (0, 255, 0),       # green
    18: (255, 215, 0),     # gold
    19: (0, 0, 0),         # black
    20: (192, 192, 192),   # silver
    21: (138, 43, 226),    # blue violet
    22: (255, 127, 80),    # coral
    23: (238, 130, 238),   # violet
    24: (245, 245, 220),   # beige
    25: (139, 69, 19),     # saddle brown
    26: (64, 224, 208)     # turquoise
}

def depth_to_pointcloud(depth, mask):
    """将深度图中的选中像素转换为相机坐标系下的 3D 点云"""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u[mask]
    v = v[mask]
    z = depth[mask]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack((x, y, z), axis=-1)  # (N, 3)

def transform_pointcloud_to_world(points, position):
    """将点云从机器人坐标系变换到世界坐标系"""
    x_pose, y_pose, o = position
    cos_o = np.cos(o)
    sin_o = np.sin(o)

    x_cam = points[:, 0]
    y_cam = points[:, 1]
    z_cam = points[:, 2]

    x_world = cos_o * x_cam - sin_o * y_cam + x_pose
    y_world = sin_o * x_cam + cos_o * y_cam + y_pose
    z_world = z_cam + 1.0  # 可选高度偏移量

    return np.stack((x_world, y_world, z_world), axis=-1)

def publish_marker_pointcloud(points, marker_id=0, color=(0.0, 1.0, 0.0), scale=0.05):
    """使用 visualization_msgs/Marker 发布点云为小球列表"""
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "semantic_points"
    marker.id = marker_id
    marker.type = Marker.SPHERE_LIST
    marker.action = Marker.ADD

    marker.scale.x = scale
    marker.scale.y = scale
    marker.scale.z = scale

    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = 1.0

    for x, y, z in points:
        pt = Point()
        pt.x = float(x)
        pt.y = float(y)
        pt.z = float(z)
        marker.points.append(pt)

    marker.lifetime = rospy.Duration(0)
    marker_pub.publish(marker)

if __name__ == '__main__':
    rospy.init_node("semantic_pointcloud_publisher")
    marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=10)
    rospy.sleep(1.0)  # 等待初始化

    # 加载数据
    path = "/home/robotlab/dataset/semantic/semantic_datasets/data_v6/test_old/2azQ1b91cZZ/ep_1_1_2azQ1b91cZZ.npz"
    data = np.load(path)
    all_sseg = data["ssegs"]       # (N, H, W)
    all_depth = data["depth_imgs"] # (N, H, W)
    all_pose = data["abs_pose"]    # (N, 3)

    for i in range(10):
        depth = np.squeeze(all_depth[i])
        sseg = np.squeeze(all_sseg[i])
        position = np.squeeze(all_pose[i])

        rospy.loginfo("帧 %d，位姿: %s", i, position)

        # 相机 → 机器人坐标系变换
        R_robot_to_cam = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        # R_cam_to_robot = R_robot_to_cam.T

        # 遍历所有类别
        for class_id in range(27):
            mask = (sseg == class_id)
            if not np.any(mask):
                continue

            points_cam = depth_to_pointcloud(depth, mask)
            points_robot = points_cam @ R_robot_to_cam
            points_world = transform_pointcloud_to_world(points_robot, position)

            # 可选过滤：仅保留所有坐标小于 2 的点
            mask_filter = np .all(points_world < 2, axis=1)
            points_world = points_world[mask_filter]

            if points_world.shape[0] == 0:
                continue

            rgb = color_mapping_27.get(class_id, (255, 255, 255))
            color = tuple([c / 255.0 for c in rgb])
            marker_id = i * 100 + class_id  # 保证唯一 ID

            publish_marker_pointcloud(points_world, marker_id=marker_id, color=color, scale=0.03)

        rospy.sleep(0.5)  # 每帧间隔