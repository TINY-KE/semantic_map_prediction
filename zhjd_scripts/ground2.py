#!/usr/bin/env python
# coding: utf-8
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import torch  # ✅ 添加 PyTorch
from models.semantic_grid import SemanticGrid
from datasets.util import viz_utils, map_utils

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
    return np.stack((x, y, z), axis=-1), u, v  # 返回相机坐标点和像素坐标

def transform_pointcloud_to_world(points, position):
    x_pose, y_pose, o = position
    cos_o = np.cos(o)
    sin_o = np.sin(o)

    x_cam = points[:, 0]
    y_cam = points[:, 1]
    z_cam = points[:, 2]

    x_world = cos_o * x_cam - sin_o * y_cam + x_pose
    y_world = sin_o * x_cam + cos_o * y_cam + y_pose
    z_world = z_cam + 1.0  # 可选：相机高度偏移

    return np.stack((x_world, y_world, z_world), axis=-1)

if __name__ == '__main__':
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

        # 相机坐标系 → 机器人坐标系变换矩阵
        R_robot_to_cam = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])

        mask = (sseg > -1)
        if not np.any(mask):
            continue

        # 得到相机点云和对应像素坐标 (u, v)
        points_cam, u, v = depth_to_pointcloud(depth, mask)
        points_robot = points_cam @ R_robot_to_cam
        points_world = transform_pointcloud_to_world(points_robot, position)

        # local3D: [1, N, 3]
        local3D = points_world[np.newaxis, ...]

        # points2D: [1, N, 2]
        points2D = np.stack((u, v), axis=-1)[np.newaxis, ...]

        # ssegs_3: [1, 1, H, W] → 转成 PyTorch Tensor
        ssegs_3 = sseg[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
        ssegs_3 = torch.from_numpy(ssegs_3).float().to('cuda')  # ✅ 转成 Tensor 并放到 GPU

        # 可选：如果你后面也用 GPU，可以一起转
        points2D = torch.from_numpy(points2D).float().to('cuda')
        local3D = torch.from_numpy(local3D).float().to('cuda')

        # 构建语义栅格图
        spatial_labels = 27
        grid_dim = (184, 184)
        cell_size = 0.1

        ego_grid_sseg_3 = map_utils.ground_projection_my(
            points2D, local3D, ssegs_3,
            sseg_labels=spatial_labels,
            grid_dim=grid_dim,
            cell_size=cell_size
        )  # shape: [27, H, W]

        # print("ego_grid_sseg_3.shape: ",ego_grid_sseg_3)
        # 可视化当前帧的语义地图
        # 假设 ego_grid_sseg_3 是 torch.Tensor，shape = [1, 27, 384, 384]
        label_map = ego_grid_sseg_3[0].argmax(dim=0)  # shape: (384, 384) -> 2D
        print(label_map)
        # 调用 colorEncode
        ego_vis = viz_utils.colorEncode(label_map, color_mapping_27)

        plt.figure(figsize=(36, 36))
        plt.imshow(ego_vis)
        plt.title(f"Frame {i} - Ground Projected Semantic Map")
        plt.axis("off")
        plt.tight_layout()
        plt.show()