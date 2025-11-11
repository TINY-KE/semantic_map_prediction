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

    def length(self):
        return len(self.episodes_file_list)

    def __getitem__(self, idx):
        ep_file = self.episodes_file_list[idx]
        ep = np.load(ep_file)

        abs_pose = ep['abs_pose'][-10:]
        ego_grid_crops_spatial = torch.from_numpy(ep['ego_grid_crops_spatial'][-10:])
        step_ego_grid_crops_spatial = torch.from_numpy(ep['step_ego_grid_crops_spatial'][-10:])
        gt_grid_crops_spatial = torch.from_numpy(ep['gt_grid_crops_spatial'][-10:])
        gt_grid_crops_objects = torch.from_numpy(ep['gt_grid_crops_objects'][-10:])

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

            'images': torch.from_numpy(ep['images'][-10:]),
            'gt_segm': torch.from_numpy(ep['ssegs'][-10:]).type(torch.int64),
            'depth_imgs': torch.from_numpy(ep['depth_imgs'][-10:]),

            'pred_ego_crops_sseg': torch.from_numpy(ep['pred_ego_crops_sseg'][-10:]),
            'step_ego_grid_27': torch.from_numpy(ep['step_ego_grid_27'][-10:])
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


def visualize_all_fields_colorized(item, timestep=0):
    """
    使用 viz_utils.colorize_grid 可视化 episode 的关键字段。
    自动补齐输入维度并统一输出维度以适配 imshow。
    """

    # === 取数据 ===
    rgb = normalize_rgb(item['images'][timestep]).transpose(1, 2, 0)
    segm = tensor_to_np(item['gt_segm'][timestep])
    depth = tensor_to_np(item['depth_imgs'][timestep])
    pred_semantic_grid_map = tensor_to_np(item['pred_ego_crops_sseg'][timestep])
    ego_spatial = tensor_to_np(item['ego_grid_crops_spatial'][timestep])
    step_ego_spatial = tensor_to_np(item['step_ego_grid_crops_spatial'][timestep])
    gt_spatial = tensor_to_np(item['gt_grid_crops_spatial'][timestep])
    gt_objects = tensor_to_np(item['gt_grid_crops_objects'][timestep])
    step_grid_27 = tensor_to_np(item['step_ego_grid_27'][timestep])

    # # === 如果预测语义是 C×H×W，取 argmax ===
    # if pred_semantic_grid_map.ndim == 3 and pred_semantic_grid_map.shape[0] > 1:
    #     pred_semantic_grid_map = np.argmax(pred_semantic_grid_map, axis=0)

    def to_5d(t):
        t = torch.tensor(t)
        while t.ndim < 5:
            t = t.unsqueeze(0)  # 在最前面添加一个新维度。例如原来是 (64, 64) → 变成 (1, 64, 64)
        return t

    # === 用 colorize_grid 上色 ===
    def color_and_extract(grid, color_mapping):
        colorized = viz_utils.colorize_grid(to_5d(grid), color_mapping=color_mapping)
        # 输出可能是 (3,H,W) 或 (1,3,H,W) 或 (1,1,3,H,W)
        colorized = torch.tensor(colorized)
        if colorized.ndim == 5:
            colorized = colorized[0, 0]
        elif colorized.ndim == 4:
            colorized = colorized[0]
        # 现在 colorized 应为 (3,H,W)
        return colorized.permute(1, 2, 0)  # 转为 (H,W,3)

    color_ego_spatial = color_and_extract(ego_spatial, 3)
    color_step_spatial = color_and_extract(step_ego_spatial, 3)
    color_gt_spatial = color_and_extract(gt_spatial, 3)
    color_gt_objects = color_and_extract(gt_objects, 27)
    color_pred_semantic = color_and_extract(pred_semantic_grid_map, 27)
    color_step_grid27 = color_and_extract(step_grid_27.argmax(axis=0), 27)

    # === 绘图 ===
    fig, axs = plt.subplots(3, 3, figsize=(20, 20))
    axs = axs.flatten()

    axs[0].imshow(rgb)
    axs[0].set_title("RGB Image")

    axs[1].imshow(segm, cmap='tab20')
    axs[1].set_title("GT Segmentation")

    axs[2].imshow(depth, cmap='viridis')
    axs[2].set_title("Depth")





    axs[3].imshow(color_pred_semantic)
    axs[3].set_title("Predicted Semantic ")


    axs[4].imshow(color_step_grid27)
    axs[4].set_title("L2M results Input (step_ego_grid_27)")

    axs[5].imshow(color_gt_objects)
    axs[5].set_title("GT Objects ")



    axs[6].imshow(color_ego_spatial)
    axs[6].set_title("Egocentric Spatial ")

    axs[7].imshow(color_step_spatial)
    axs[7].set_title("Step Ego Spatial (step_ego_grid_crops_spatial)")

    axs[8].imshow(color_gt_spatial)
    axs[8].set_title("GT Spatial ")



    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"✅ timestep={timestep} 可视化完成。")



# ----------------------------
# 主函数入口
# ----------------------------
if __name__ == "__main__":
    root_path = "/home/robotlab/dataset/semantic/semantic_datasets/data_v6/test/2azQ1b91cZZ"
    ep_path = root_path + '/' + 'ep_1_1_2azQ1b91cZZ.npz'

    # ep_path = '/home/robotlab/dataset/MP3D_dataset/v1/tasks/mp3d_habitat/NPZ/train/HxpKQynjfin/ep_1_1_HxpKQynjfin.npz'
    if not os.path.exists(ep_path):
        print(f"❌ 文件未找到: {ep_path}")
        exit(1)

    data = np.load(ep_path)
    print(data.files)
    # work1: ['abs_pose', 'ego_grid_crops_spatial', 'step_ego_grid_crops_spatial', 'gt_grid_crops_spatial', 'gt_grid_crops_objects', 'images', 'ssegs', 'depth_imgs']
    # work2: ['abs_pose', 'ego_grid_crops_spatial', 'step_ego_grid_crops_spatial', 'gt_grid_crops_spatial', 'gt_grid_crops_objects', 'images', 'ssegs', 'depth_imgs', 'pred_ego_crops_sseg', 'step_ego_grid_27']
    # models/predictors/map_predictor_model.py 中的唯一输入是step_ego_grid_27, 这也是work1的输出。
    # 2️⃣ 打印 abs_pose 的 shape
    print("abs_pose shape:", data['abs_pose'].shape)  # 序列长度为10，参考L2M的store_episodes_parallel.py, 设定了每个episode（轨迹）中只保留最后的10个pose。

    # 3️⃣ 输出序列长度（时间步数量）
    print("abs_pose 序列长度:", data['abs_pose'].shape[0])

    print(f"{'step_ego_grid_crops_spatial shape:':<35} {data['step_ego_grid_crops_spatial'].shape}")
    print(f"{'step_ego_grid_27 shape:':<35} {data['step_ego_grid_27'].shape}")


    dataset = ObjNavEpisodeDataset([ep_path])
    item = dataset[0]

    step_ego_crops = item['ego_grid_crops_spatial']
    T, _, cH, cW = step_ego_crops.shape
    print(f"T (timesteps): {T}")
    print(f"C (channels): {_}")
    print(f"cH (crop height): {cH}")
    print(f"cW (crop width): {cW}")

    # for t in range(4):
    #     print(f"\n=== 可视化时间步 {t} ===")
    #     visualize_item(item, timestep=t)

    for t in range(10):
        print(f"🕒 时间步 {t}")
        visualize_all_fields_colorized(item, timestep=t)