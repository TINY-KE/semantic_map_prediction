import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import habitat
from habitat.config.default import get_config
import habitat.utils.visualizations.maps as map_util
from datasets.util import map_utils
import datasets.util.utils as utils
import datasets.util.viz_utils as viz_utils
from models.img_segmentation import get_img_segmentor_from_options
import test_utils as tutils

import torchvision.transforms as transforms

import gzip
import json


# ----------------------------
# 数据集类
# ----------------------------
class ObjNavEpisodeDataset(Dataset):
    def __init__(self, episode_files):
        self.episodes_file_list = episode_files

    def __len__(self):
        return len(self.episodes_file_list)

    def __getitem__(self, idx):
        ep_file = self.episodes_file_list[idx]
        ep = np.load(ep_file)

        print("ep['abs_pose']维度：", ep['abs_pose'].shape)
        abs_pose = ep['abs_pose'][-4:]
        ego_grid_crops_spatial = torch.from_numpy(ep['ego_grid_crops_spatial'][-4:])
        step_ego_grid_crops_spatial = torch.from_numpy(ep['step_ego_grid_crops_spatial'][-4:])
        gt_grid_crops_spatial = torch.from_numpy(ep['gt_grid_crops_spatial'][-4:])
        gt_grid_crops_objects = torch.from_numpy(ep['gt_grid_crops_objects'][-4:])

        # 计算相对位姿
        rel_pose = []
        for i in range(abs_pose.shape[0]):
            rel_pose.append(utils.get_rel_pose(pos2=abs_pose[i], pos1=abs_pose[0]))

        item = {
            'pose': torch.from_numpy(np.asarray(rel_pose)).float(),
            'abs_pose': torch.from_numpy(abs_pose).float(),
            'ego_grid_crops_spatial': ego_grid_crops_spatial,
            'step_ego_grid_crops_spatial': step_ego_grid_crops_spatial,
            'gt_grid_crops_spatial': gt_grid_crops_spatial,
            'gt_grid_crops_objects': gt_grid_crops_objects,

            'images': torch.from_numpy(ep['images'][-4:]),
            'gt_segm': torch.from_numpy(ep['ssegs'][-4:]).type(torch.int64),
            'depth_imgs': torch.from_numpy(ep['depth_imgs'][-4:]),

            'pred_ego_crops_sseg': torch.from_numpy(ep['pred_ego_crops_sseg'][-4:]),
            'step_ego_grid_27': torch.from_numpy(ep['step_ego_grid_27'][-4:])
        }

        return item

# ----------------------------
# 可视化函数
# ----------------------------
def visualize_item(item, timestep=0):
    rgb = item['images'][timestep]          # [3, H, W]
    # rgb_np = normalize_rgb(rgb).transpose(1, 2, 0)  # [H, W, 3]
    segm = item['gt_segm'][timestep][0]     # [H, W]
    depth = item['depth_imgs'][timestep][0] # [H, W]

    rgb_np = rgb.permute(1, 2, 0).numpy().astype(np.uint8)
    segm_np = segm.numpy()
    depth_np = depth.numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(rgb_np)
    axs[0].set_title("RGB Image")
    axs[0].axis("off")

    axs[1].imshow(segm_np, cmap="tab20")
    axs[1].set_title("Semantic Segmentation")
    axs[1].axis("off")

    axs[2].imshow(depth_np, cmap="viridis")
    axs[2].set_title("Depth Map")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()

def normalize_rgb(rgb_tensor):
    """
    将 RGB float tensor 转换为 uint8 显示图像
    支持范围为 [0, 1] 或 [-1, 1]
    """
    rgb_np = rgb_tensor.detach().cpu().numpy()
    if rgb_np.dtype in [np.float32, np.float64]:
        if rgb_np.max() <= 1.0 and rgb_np.min() >= 0.0:
            rgb_np = rgb_np * 255.0
        elif rgb_np.min() >= -1.0 and rgb_np.max() <= 1.0:
            rgb_np = (rgb_np + 1.0) * 127.5
    rgb_np = np.clip(rgb_np, 0, 255).astype(np.uint8)
    return rgb_np

def tensor_to_np(t):
    """确保 Tensor 是 numpy 格式，且 squeeze 掉 batch/channel 维度"""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
        if t.ndim == 4:
            t = t[0]
        if t.ndim == 3 and t.shape[0] == 1:
            t = t[0]
        return t.numpy()
    return t

def visualize_all_fields(item, timestep=0):
    """
    显示所有字段的可视化图
    """
    # rgb = tensor_to_np(item['images'][timestep]).transpose(1, 2, 0).astype(np.uint8)  # [H,W,3]
    rgb = normalize_rgb(item['images'][timestep]).transpose(1, 2, 0)
    segm = tensor_to_np(item['gt_segm'][timestep])
    depth = tensor_to_np(item['depth_imgs'][timestep])
    pred_semantic_grid_map = tensor_to_np(item['pred_ego_crops_sseg'][timestep])   # (64, 64)，int64， 栅格存着 物体的label
    egocentric_spatial_grid_map = tensor_to_np(item['ego_grid_crops_spatial'][timestep])  # (3, 64, 64)，float32， 栅格存着 空间占据的概率
    step_ego_grid_crops_spatial = tensor_to_np(item['step_ego_grid_crops_spatial'][timestep])  # # (3, 64, 64)，float32， 栅格存着 物体种类的概率 【输入】
    GroundTruth_spatial_crop = tensor_to_np(item['gt_grid_crops_spatial'][timestep])
    GroundTruth_object_crop = tensor_to_np(item['gt_grid_crops_objects'][timestep])
    step_grid_27 = tensor_to_np(item['step_ego_grid_27'][timestep])  # # (27, 64, 64)，float32， 每个维度对应一个物体，栅格存着 对应物体种类的概率 【输入】，应该是来自

    # 如果预测语义是 C x H x W，取 argmax
    if pred_semantic_grid_map.ndim == 3 and pred_semantic_grid_map.shape[0] > 1:
        pred_semantic_grid_map = np.argmax(pred_semantic_grid_map, axis=0)

    # 如果是网格图（多通道），只取第一个通道可视化
    def first_channel(img):
        return img if img.ndim == 2 else img[0]

    fig, axs = plt.subplots(4, 3, figsize=(10, 10))
    axs = axs.flatten()

    axs[0].imshow(rgb)
    axs[0].set_title("RGB Image")

    axs[1].imshow(segm, cmap='tab20')
    axs[1].set_title("GT Segmentation")

    axs[2].imshow(depth, cmap='viridis')
    axs[2].set_title("Depth")



    axs[3].imshow(pred_semantic_grid_map, cmap='tab20')
    axs[3].set_title("Predicted Semantic Grid Map")

    # TODO: 为什么是三个维度，
    axs[4].imshow(first_channel(egocentric_spatial_grid_map), cmap='gray')
    axs[4].set_title("Egocentric Spatial Grid Map")

    # TODO: 为什么是三个维度， step_ego_grid_crops_spatial. 这个是输入进work2神经网络的
    axs[5].imshow(first_channel(step_ego_grid_crops_spatial), cmap='gray')
    axs[5].set_title("Step Ego Grid Spatial")



    axs[6].imshow(GroundTruth_object_crop, cmap='tab20')
    axs[6].set_title("GroundTruth Object Crop")

    axs[7].imshow(first_channel(GroundTruth_spatial_crop), cmap='gray')
    axs[7].set_title("GroundTruth Spatial Crop")

    # TODO: 为什么是三个维度， step_ego_grid_27. 这个是输入进work2神经网络的
    # axs[8].imshow((step_grid_27), cmap='gray')
    # axs[8].set_title("Step Ego Grid 27")
    axs[8].imshow(step_grid_27.argmax(axis=0), cmap='tab20')
    axs[8].set_title("step_grid_27 - argmax 语义图")




    # axs[9].imshow(step_ego_grid_crops_spatial[0], cmap='gray')
    # axs[9].set_title("Step Ego Grid Channel 0")
    #
    # axs[10].imshow(step_ego_grid_crops_spatial[1], cmap='gray')
    # axs[10].set_title("Step Ego Grid Channel 1")
    #
    # axs[11].imshow(step_ego_grid_crops_spatial[2], cmap='gray')
    # axs[11].set_title("Step Ego Grid Channel 2")

    axs[9].imshow(egocentric_spatial_grid_map[0], cmap='gray')
    axs[9].set_title("egocentric_spatial_grid_map 0")

    axs[10].imshow(egocentric_spatial_grid_map[1], cmap='gray')
    axs[10].set_title("egocentric_spatial_grid_map 1")

    axs[11].imshow(egocentric_spatial_grid_map[2], cmap='gray')
    axs[11].set_title("egocentric_spatial_grid_map 2")

    print("egocentric_spatial_grid_map shape:", egocentric_spatial_grid_map.shape)
    print("                            dtype:", egocentric_spatial_grid_map.dtype)
    print("step_grid_27 shape:", step_grid_27.shape)
    print("                       dtype:", step_grid_27.dtype)
    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# ----------------------------
# 主函数入口
# ----------------------------
if __name__ == "__main__":
    root_path = "/home/robotlab/dataset/semantic/semantic_datasets/data_v6/test_old/2azQ1b91cZZ"
    ep_path = root_path + '/' + 'ep_92_148_2azQ1b91cZZ.npz'
    root_path = "/home/robotlab/dataset/semantic/semantic_datasets/data_v6/test_old/2azQ1b91cZZ"
    ep_path = root_path + '/' + 'ep_1_1_2azQ1b91cZZ.npz'
    # root_path = "/home/robotlab/dataset/MP3D_dataset/v1/tasks/mp3d_habitat_scenes_dir/NPZ/train/HxpKQynjfin"
    # ep_path = root_path + '/' + 'ep_1_1_HxpKQynjfin.npz'

    if not os.path.exists(ep_path):
        print(f"❌ 文件未找到: {ep_path}")
        exit(1)

    dataset = ObjNavEpisodeDataset([ep_path])
    item = dataset[0]
    print(item.keys())  # 看看 item 里有哪些字段
    print("RGB 维度：", item['images'].shape)
    print("深度图维度：", item['depth_imgs'].shape)
    print("语义分割维度：", item['gt_segm'].shape)
    print("相机位姿维度：", item['abs_pose'].shape)

    # for t in range(4):
    #     print(f"\n=== 可视化时间步 {t} ===")
    #     visualize_item(item, timestep=t)

    for t in range(10):
        print(f"🕒 时间步 {t}")
        visualize_all_fields(item, timestep=t)