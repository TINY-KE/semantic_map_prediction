import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

# ----------------------------
# 自定义颜色映射（27类）
# ----------------------------
color_mapping_27 = {
    0:  (255, 255, 255),  # white
    1:  (128, 128, 0),    # olive
    2:  (0, 0, 255),      # blue
    3:  (255, 0, 0),      # red
    4:  (255, 0, 255),    # magenta
    5:  (0, 255, 255),    # cyan
    6:  (255, 165, 0),    # orange
    7:  (255, 255, 0),    # yellow
    8:  (128, 128, 128),  # gray
    9:  (128, 0, 0),      # maroon
    10: (255, 20, 147),   # deep pink
    11: (0, 128, 0),      # dark green
    12: (128, 0, 128),    # purple
    13: (0, 128, 128),    # teal
    14: (0, 0, 128),      # navy
    15: (210, 105, 30),   # chocolate
    16: (188, 143, 143),  # rosy brown
    17: (0, 255, 0),      # green
    18: (255, 215, 0),    # gold
    19: (0, 0, 0),        # black
    20: (192, 192, 192),  # silver
    21: (138, 43, 226),   # blue violet
    22: (255, 127, 80),   # coral
    23: (238, 130, 238),  # violet
    24: (245, 245, 220),  # beige
    25: (139, 69, 19),    # saddle brown
    26: (64, 224, 208)    # turquoise
}

# ----------------------------
# 数据集类
# ----------------------------
class SimpleSegmentationDataset(Dataset):
    def __init__(self, npz_file_path):
        if not os.path.exists(npz_file_path):
            raise FileNotFoundError(f"❌ 文件不存在: {npz_file_path}")
        self.data = np.load(npz_file_path)
        self.images = self.data["images"]    # (N, 3, H, W)
        self.ssegs = self.data["ssegs"]    # (N, H, W) or (N, 1, H, W)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])           # (3, H, W)
        sseg = torch.from_numpy(self.ssegs[idx])             # (H, W) or (1, H, W)
        if sseg.ndim == 3:
            sseg = sseg[0]  # squeeze channel dim if needed
        return {
            "image": image,
            "sseg": sseg
        }

# ----------------------------
# 将语义图索引转换为 RGB 彩色图
# ----------------------------
def colorize_sseg(sseg, color_map):
    h, w = sseg.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)

    for label_id, color in color_map.items():
        mask = sseg == label_id
        color_image[mask] = color

    return color_image

# ----------------------------
# 可视化函数
# ----------------------------
def visualize_image_and_sseg(item, timestep=0):
    image = item["image"]
    sseg = item["sseg"]

    # --- 修正 RGB 图像维度 ---
    rgb_tensor = image.detach().cpu()

    if rgb_tensor.ndim == 3 and rgb_tensor.shape[0] == 3:
        # (3, H, W) → (H, W, 3)
        rgb_np = rgb_tensor.permute(1, 2, 0).numpy()
    elif rgb_tensor.ndim == 3 and rgb_tensor.shape[2] == 3:
        # (H, W, 3)
        rgb_np = rgb_tensor.numpy()
    else:
        raise ValueError(f"Unsupported image shape: {rgb_tensor.shape}")

    # --- 修正数值范围 ---
    if rgb_np.dtype == np.float32 or rgb_np.max() <= 1.0:
        rgb_np = (rgb_np * 255).clip(0, 255).astype(np.uint8)
    else:
        rgb_np = rgb_np.astype(np.uint8)
    sseg_np = sseg.numpy()
    print(f"[调试] 图像值范围: min={rgb_np.min()}, max={rgb_np.max()}, dtype={rgb_np.dtype}")

    # --- 语义图上色 ---
    segm_color = colorize_sseg(sseg_np, color_mapping_27)

    # --- 可视化 ---
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(rgb_np)
    axs[0].set_title("RGB Image")
    axs[0].axis("off")

    axs[1].imshow(segm_color)
    axs[1].set_title("Semantic Segmentation")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()

# ----------------------------
# 主函数入口
# ----------------------------
if __name__ == "__main__":
    # root_path = "/home/robotlab/dataset/semantic/semantic_datasets/data_v6/test_old/2azQ1b91cZZ"
    # npz_file_path = root_path + '/' + 'ep_1_1_2azQ1b91cZZ.npz'

    npz_file_path = "/home/robotlab/work/semantic-segmentation-pytorch/save_results/all_data.npz"

    # [debug] 先打印文件中有哪些 key
    data = np.load(npz_file_path)
    print("📦 文件中实际包含的字段 (keys):", list(data.keys()))
    data.close()

    dataset = SimpleSegmentationDataset(npz_file_path)

    for t in range(len(dataset)):
        print(f"🕒 时间步 {t}")
        item = dataset[t]
        visualize_image_and_sseg(item, timestep=t)