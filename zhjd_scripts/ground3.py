#!/usr/bin/env python
# coding: utf-8
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.semantic_grid import SemanticGrid
from datasets.util import viz_utils, map_utils

# 图像分辨率与相机参数
W, H = 128, 128
hfov_deg = 79
hfov = hfov_deg * np.pi / 180.0
fx = (W / 2) / np.tan(hfov / 2)
fy = fx
cx, cy = W / 2, H / 2

# 类别颜色映射（27类）
color_mapping_27 = {
    0:  (255, 255, 255),   1:  (128, 128, 0),     2:  (0, 0, 255),
    3:  (255, 0, 0),       4:  (255, 0, 255),     5:  (0, 255, 255),
    6:  (255, 165, 0),     7:  (255, 255, 0),     8:  (128, 128, 128),
    9:  (128, 0, 0),      10: (255, 20, 147),    11: (0, 128, 0),
    12: (128, 0, 128),    13: (0, 128, 128),     14: (0, 0, 128),
    15: (210, 105, 30),   16: (188, 143, 143),   17: (0, 255, 0),
    18: (255, 215, 0),    19: (0, 0, 0),         20: (192, 192, 192),
    21: (138, 43, 226),   22: (255, 127, 80),    23: (238, 130, 238),
    24: (245, 245, 220),  25: (139, 69, 19),     26: (64, 224, 208)
}

def depth_to_pointcloud(depth, mask):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u[mask]
    v = v[mask]
    z = depth[mask]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack((x, y, z), axis=-1), u, v

def transform_pointcloud_to_world(points, position):
    x_pose, y_pose, o = position
    cos_o = np.cos(o)
    sin_o = np.sin(o)

    x_cam = points[:, 0]
    y_cam = points[:, 1]
    z_cam = points[:, 2]

    x_world = cos_o * x_cam - sin_o * y_cam + x_pose
    y_world = sin_o * x_cam + cos_o * y_cam + y_pose
    z_world = z_cam + 1.0  # 可选：相机高度修正

    return np.stack((x_world, y_world, z_world), axis=-1)

if __name__ == '__main__':
    # 加载数据
    path = "/home/robotlab/dataset/semantic/semantic_datasets/data_v6/test_old/2azQ1b91cZZ/ep_1_1_2azQ1b91cZZ.npz"
    data = np.load(path)
    all_sseg = data["ssegs"]       # (N, H, W)
    all_depth = data["depth_imgs"] # (N, H, W)
    all_pose = data["abs_pose"]    # (N, 3)

    # 地图参数
    spatial_labels = 27
    grid_dim = (184, 184)
    cell_size = 0.1

    # 初始化全局地图张量
    global_grid = torch.zeros((spatial_labels, *grid_dim), dtype=torch.float32).to("cuda")

    for i in range(10):
        depth = np.squeeze(all_depth[i])
        sseg = np.squeeze(all_sseg[i])
        position = np.squeeze(all_pose[i])

        # 相机 → 机器人坐标变换
        R_robot_to_cam = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])

        mask = (sseg > -1)
        if not np.any(mask):
            continue

        # 相机点云和像素坐标
        points_cam, u, v = depth_to_pointcloud(depth, mask)
        points_robot = points_cam @ R_robot_to_cam
        points_world = transform_pointcloud_to_world(points_robot, position)

        local3D = points_world[np.newaxis, ...]  # shape: [1, N, 3]
        points2D = np.stack((u, v), axis=-1)[np.newaxis, ...]  # shape: [1, N, 2]

        ssegs_3 = sseg[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
        ssegs_3 = torch.from_numpy(ssegs_3).float().to("cuda")
        points2D = torch.from_numpy(points2D).float().to("cuda")
        local3D = torch.from_numpy(local3D).float().to("cuda")

        # 构建语义栅格
        ego_grid_sseg_3 = map_utils.ground_projection_my(
            points2D, local3D, ssegs_3,
            sseg_labels=spatial_labels,
            grid_dim=grid_dim,
            cell_size=cell_size
        )  # shape: [t, 27, 184, 184]
        print("ego_grid_sseg_3.shape: ", ego_grid_sseg_3.shape)

        # 累加到全局地图
        global_grid += ego_grid_sseg_3.squeeze(0)

    # 多帧融合后取 argmax 得到最终语义图
    label_map = global_grid.argmax(dim=0).cpu().numpy()  # shape: (H, W)
    ego_vis = viz_utils.colorEncode(label_map, color_mapping_27)

    # 可视化融合地图
    plt.figure(figsize=(8, 8))
    plt.imshow(ego_vis)
    plt.title("Fused Semantic Map (10 Frames)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()