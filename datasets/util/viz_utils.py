
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import math
import torch
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb
import datasets.util.map_utils as map_utils
import datasets.util.viz_utils as viz_utils
import cv2

'''
MP3D original semantic labels and reduced set correspondence
# Original set from here: https://github.com/niessner/Matterport/blob/master/metadata/mpcat40.tsv
0 void 0
1 wall 15 structure
2 floor 17 free-space
3 chair 1
4 door 2
5 table 3
6 picture 18
7 cabinet 19
8 cushion 4
9 window 15 structure
10 sofa 5
11 bed 6
12 curtain 16 other
13 chest_of_drawers 20
14 plant 7
15 sink 8
16 stairs 17 free-space
17 ceiling 17 free-space
18 toilet 9
19 stool 21
20 towel 22
21 mirror 16 other
22 tv_monitor 10
23 shower 11
24 column 15 structure
25 bathtub 12
26 counter 13
27 fireplace 23
28 lighting 16 other
29 beam 16 other
30 railing 16 other
31 shelving 16 other
32 blinds 16 other
33 gym_equipment 24
34 seating 25
35 board_panel 16 other
36 furniture 16 other
37 appliances 14
38 clothes 26
39 objects 16 other
40 misc 16 other
'''
# 27 categories which include the 21 object categories in the habitat challenge
label_conversion_40_27 = {-1:0, 0:0, 1:15, 2:17, 3:1, 4:2, 5:3, 6:18, 7:19, 8:4, 9:15, 10:5, 11:6, 12:16, 13:20, 14:7, 15:8, 16:17, 17:17,
                    18:9, 19:21, 20:22, 21:16, 22:10, 23:11, 24:15, 25:12, 26:13, 27:23, 28:16, 29:16, 30:16, 31:16, 32:16,
                    33:24, 34:25, 35:16, 36:16, 37:14, 38:26, 39:16, 40:16}


# BGR  # wacv 20230823 the note for the color is wrong!
# color_mapping_27 = {
#     0:(255,255,255), # white
#     1:(128,128,0), # olive (dark yellow)
#     2:(0,0,255), # blue
#     3:(255,0,0), # red
#     4:(255,0,255), # magenta
#     5:(0,255,255), # cyan
#     6:(255,165,0), # orange
#     7:(255,255,0), # yellow
#     8:(128,128,128), # gray
#     9:(128,0,0), # maroon
#     10:(255,20,147), # pink
#     11:(0,128,0), # dark green
#     12:(128,0,128), # purple
#     13:(0,128,128), # teal
#     14:(0,0,128), # navy (dark blue)
#     15:(210,105,30), # chocolate
#     16:(188,143,143), # rosy brown
#     # 17:(0,255,0), # green
#     # 17: (217, 239, 226),  # light green1
#     17: (179, 224, 197),  # light green2
#     # 17: (229, 195, 156),  # light blue
#     18:(255,215,0), # gold
#     19:(0,0,0), # black
#     20:(192,192,192), # silver
#     21:(138,43,226), # blue violet
#     22:(255,127,80), # coral
#     23:(238,130,238), # violet
#     24:(245,245,220), # beige
#     25:(139,69,19), # saddle brown
#     26:(64,224,208) # turquoise
# }


name_mapping_27 = {
    0:  "空类别",           # 白色 white                       空类别 / 无类别 (void)
    1:  "椅子",             # 橄榄色 olive                     椅子 (chair)  ***
    2:  "门",               # 蓝色 blue                        门 (door)  ***%%
    3:  "桌子",             # 红色 red                         桌子 (table)  ***
    4:  "靠垫cushion",             # 洋红色 magenta                   靠垫 / 坐垫 (cushion)  ***
    5:  "沙发",             # 青色 cyan                        沙发 (sofa)  ***
    6:  "床",               # 橙色 orange                      床 (bed)  ***
    7:  "植物",             # 黄色 yellow                      植物 (plant)
    8:  "洗手池",           # 灰色 gray                        洗手池 / 水槽 (sink)
    9:  "马桶",             # 栗色 maroon                      马桶 (toilet)
    10: "电视",             # 深粉红 deep pink                 电视 / 显示器 (tv_monitor)  ***%%
    11: "淋浴器",           # 深绿色 dark green               淋浴器 (shower)
    12: "浴缸",             # 紫色 purple                      浴缸 (bathtub)  ***%%
    13: "工作台counter",           # 水鸭色 teal                      操作台 / 工作台 (counter)  ***
    14: "家电",             # 藏青色 navy                     家电 (appliances)
    15: "墙",         # 巧克力色 chocolate              建筑结构 (structure)
    16: "其他",             # 褐玫瑰色 rosy brown             其他 / 杂项 (other)
    17: "可行走区域",       # 绿色 green                      空闲空间 / 可行走区域 (free-space)
    18: "画",             # 金色 gold                       图片 / 挂画 (picture)
    19: "橱柜cabinet",             # 黑色 black                      橱柜 / 柜子 (cabinet)  ***
    20: "抽屉",           # 银色 silver                     抽屉柜 (chest_of_drawers)
    21: "凳子",             # 蓝紫色 blue violet              凳子 (stool)
    22: "毛巾",             # 珊瑚色 coral                    毛巾 (towel)
    23: "壁炉",             # 紫罗兰色 violet                 壁炉 (fireplace)
    24: "健身器材",         # 米色 / 浅卡其 beige            健身器材 (gym_equipment)
    25: "座位",             # 马鞍棕 saddle brown            座位（综合类）(seating)
    26: "衣服",             # 绿松石色 turquoise              衣物 (clothes)
}

color_mapping_27 = {
    0:  (255, 255, 255),   # 白色 white                       空类别 / 无类别 (void)
    1:  (128, 128, 0),     # 橄榄色 olive                     椅子 (chair)  ***
    2:  (0, 0, 255),       # 蓝色 blue                        门 (door)  ***
    3:  (255, 0, 0),       # 红色 red                         桌子 (table)  ***
    4:  (255, 0, 255),     # 洋红色 magenta                   靠垫 / 坐垫 (cushion)  ***
    5:  (0, 255, 255),     # 青色 cyan                        沙发 (sofa)  ***
    6:  (255, 165, 0),     # 橙色 orange                      床 (bed)  ***
    7:  (255, 255, 0),     # 黄色 yellow                      植物 (plant)
    8:  (128, 128, 128),   # 灰色 gray                        洗手池 / 水槽 (sink)
    9:  (128, 0, 0),       # 栗色 maroon                      马桶 (toilet)
    10: (255, 20, 147),    # 深粉红 deep pink                 电视 / 显示器 (tv_monitor)  ***
    11: (0, 128, 0),       # 深绿色 dark green               淋浴器 (shower)
    12: (128, 0, 128),     # 紫色 purple                      浴缸 (bathtub)  ***
    13: (0, 128, 128),     # 水鸭色 teal                      操作台 / 工作台 (counter)  ***
    14: (0, 0, 128),       # 藏青色 navy                     家电 (appliances)
    15: (210, 105, 30),    # 巧克力色 chocolate              建筑结构 (structure)
    16: (188, 143, 143),   # 褐玫瑰色 rosy brown             其他 / 杂项 (other)
    17: (0, 255, 0),       # 绿色 green                      空闲空间 / 可行走区域 (free-space)   $$$
    18: (255, 215, 0),     # 金色 gold                       图片 / 挂画 (picture)
    19: (0, 0, 0),         # 黑色 black                      橱柜 / 柜子 (cabinet)  ***
    20: (192, 192, 192),   # 银色 silver                     抽屉柜 (chest_of_drawers)
    21: (138, 43, 226),    # 蓝紫色 blue violet              凳子 (stool)
    22: (255, 127, 80),    # 珊瑚色 coral                    毛巾 (towel)
    23: (238, 130, 238),   # 紫罗兰色 violet                 壁炉 (fireplace)
    24: (245, 245, 220),   # 米色 / 浅卡其 beige            健身器材 (gym_equipment)
    25: (139, 69, 19),     # 马鞍棕 saddle brown            座位（综合类）(seating)
    26: (64, 224, 208)     # 绿松石色 turquoise              衣物 (clothes)
}


# three label classification (0:void, 1:occupied, 2:free)
label_conversion_40_3 = {-1:0, 0:0, 1:1, 2:2, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1, 13:1, 14:1, 15:1, 16:2, 17:2,
                    18:1, 19:1, 20:1, 21:1, 22:1, 23:1, 24:1, 25:1, 26:1, 27:1, 28:1, 29:1, 30:1, 31:1, 32:1,
                    33:1, 34:1, 35:1, 36:1, 37:1, 38:1, 39:1, 40:1}
# RGB
color_mapping_3 = {
    0:(200,200,200), # unknown
    1:(0,0,0), # obstacle
    2:(255,255,255), # free
}

# 保存图像数据为 .png 文件。
def write_img(img, savepath, name):
    # img: T x 3 x dim x dim, assumed normalized
    for i in range(img.shape[0]):
        vis_img = img[i,:,:,:].cpu().numpy()
        vis_img = np.transpose(vis_img, (1,2,0))
        im_path = savepath + str(i) + "_" + name + ".png"
        cv2.imwrite(im_path, vis_img[:,:,::-1]*255.0)

# 保存深度图为 .png 文件，并应用颜色映射（JET）。
def write_depth_img(img, savepath, name):
    # img: T x 1 x dim x dim, assumed normalized
    for i in range(img.shape[0]):

        vis_img_tmp = img[i][0].cpu().numpy()
        min = np.min(vis_img_tmp)
        max = np.max(vis_img_tmp)

        # print('vis_img_tmp', vis_img_tmp)
        vis_img = (img[i][0].cpu().numpy() / (max-min)*255).astype(np.uint8)
        # print('vis_img', vis_img)
        # print('vis_img shape', vis_img.shape)
        vis_img_color = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)

        # vis_img = np.transpose(vis_img, (1,2,0))
        im_path = savepath + str(i) + "_" + name + ".png"
        # cv2.imwrite(im_path, vis_img[:,:,::-1]*255.0)
        # vis_img.save(im_path)
        # cv2.cvtColor()
        cv2.imwrite(im_path, vis_img_color)
        # cv2.imwrite(im_path, vis_img[:,:,::-1]*255.0)


# 把一个语义网格（occupancy / semantic grid）从多通道标签或概率图，转换成彩色的 RGB 图像，方便可视化或写入 TensorBoard 视频。
# 输入： grid 是一个五维张量，形状为(B, T, C, H, W)
# 输出： (B, T, 3, H, W) 的张量，其中 3 是 RGB 通道数。
def colorize_grid(grid, color_mapping=27): # to pass into tensorboardX video
    # Input: grid -- B x T x C x grid_dim x grid_dim, where C=1,T=1 when gt and C=41,T>=1 for other
    # Output: grid_img -- B x T x 3 x grid_dim x grid_dim
    grid = grid.detach().cpu().numpy()
    grid_img = np.zeros((grid.shape[0], grid.shape[1], grid.shape[3], grid.shape[4], 3),  dtype=np.uint8)

    # 如果 原grid 有多个通道（C > 1），取最大概率
    if grid.shape[2] > 1:
        # For cells where prob distribution is all zeroes (or uniform), argmax returns arbitrary number (can be true for the accumulated maps)
        # 每个像素在 C 个通道上代表每个类别的概率； 取最大概率 np.amax(grid, axis=2)；
        grid_prob_max = np.amax(grid, axis=2)
        # 如果最大值 ≤ 0.05，说明这个像素所有类别都“不确定”，属于“未观测区域”；
        inds = np.asarray(grid_prob_max<=0.05).nonzero() # if no label has prob higher than k then assume unobserved
        # 把这些像素的类别 0 通道设为 1（强制认为它是 “void / unknown”）；
        grid[inds[0], inds[1], 0, inds[2], inds[3]] = 1 # assign label 0 (void) to be the dominant label
        # 取每个像素概率最大的类别编号，得到整数标签图。
        grid = np.argmax(grid, axis=2) # B x T x grid_dim x grid_dim。即 取了 axis=2 之后，第2维（C2）就被“压掉”了。
    else:
        grid = grid.squeeze(2)

    if color_mapping==27:
        color_mapping = color_mapping_27
    else:
        color_mapping = color_mapping_3
    for label in color_mapping.keys():
        grid_img[ grid==label ] = color_mapping[label]

    #当前 grid_img 是 (B, T, H, W, 3)，通过 transpose(0, 1, 4, 2, 3) → (B, T, 3, H, W)
    return torch.tensor(grid_img.transpose(0, 1, 4, 2, 3), dtype=torch.uint8)


def write_tensor_imgSegm(img, savepath, name, t=None):
    # pred: T x C x dim x dim
    if img.shape[1] > 1:
        img = torch.argmax(img.cpu(), dim=1, keepdim=True) # T x 1 x cH x cW
    img_labels = img.squeeze(1)

    for i in range(img_labels.shape[0]):
        img0 = img_labels[i,:,:]

        vis_img = np.zeros((img0.shape[0], img0.shape[1], 3), dtype=np.uint8)
        for label in color_mapping_27.keys():
            vis_img[ img0==label ] = color_mapping_27[label]
        
        if t is None:
            im_path = savepath + str(i) + "_" + name + ".png"
        else:
            im_path = savepath + name + "_" + str(t) + "_" + str(i) + ".png"
        # cv2.imwrite(im_path, vis_img[:,:,::-1])
        cv2.imwrite(im_path, vis_img)



def display_sample(rgb_obs, depth_obs, sseg_img=None, savepath=None):
    # sseg_img is semantic observation from Matterport habitat
    depth_obs = depth_obs / np.amax(depth_obs) # normalize for visualization
    rgb_img = Image.fromarray(rgb_obs, mode="RGB")
    depth_img = Image.fromarray((depth_obs * 255).astype(np.uint8), mode="L")
    
    if sseg_img is not None:
        semantic_img = Image.new("P", (sseg_img.shape[1], sseg_img.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((sseg_img.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")

        arr = [rgb_img, depth_img, semantic_img]
        n=3
    else:
        arr = [rgb_img, depth_img]
        n=2

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, n, i+1)
        ax.axis('off')
        plt.imshow(data)
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()


# zhjd 定制
#  ensemble_object_maps.shape:  torch.Size([3, 1, 10, 27, 64, 64])  ensemble_num, B, T, C, cH, cW
#  pred_maps_objects.shape:  torch.Size([1, 10, 27, 64, 64]  B, T, _, cH, cW
def save_ensembles(ensemble_object_maps, pred_maps_objects, save_img_dir_):
    B, T, _, cH, cW = pred_maps_objects.shape
    for t in range(T):
        ensemble1 = color_and_extract(ensemble_object_maps[0, 0, t, :, :, :], 27)
        ensemble2 = color_and_extract(ensemble_object_maps[1, 0, t, :, :, :], 27)
        ensemble3 = color_and_extract(ensemble_object_maps[2, 0, t, :, :, :], 27)
        ensemble4 = color_and_extract(ensemble_object_maps[3, 0, t, :, :, :], 27)
        ## FIXME: 先用平均图替代
        # pred_maps_objects_single = color_and_extract(pred_maps_objects[0, t, :, :, :], 27)
        # ensemble4 = pred_maps_objects_single

        # === 四宫格保存本地===
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        axs = axs.flatten()

        imgs = [
            (ensemble1, "ensemble1", None),
            (ensemble2, "ensemble2", None),
            (ensemble3, "ensemble3", None),
            (ensemble4, "ensemble4", None)
        ]

        for i, (img, title, cmap) in enumerate(imgs):
            axs[i].imshow(img, cmap=cmap)
            axs[i].set_title(title)
            axs[i].axis('off')

        plt.tight_layout()

        # === 保存图片 ===
        save_file = os.path.join(save_img_dir_, f"ensemble_t{t}.png")
        plt.savefig(save_file, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()
    print(f"✅ 已保存集成模型的结果: {save_img_dir_}")


# zhjd 定制
#  step_geo_grid.shape:  torch.Size([1, 10, 27, 300, 300])
#  step_uncertainty.shape:  torch.Size([1, 10, 27, 300, 300]
def save_uncertainty(step_geo_grid, step_uncertainty, pose_coords_list, save_img_dir_, timestamp_length):
# def save_uncertainty(sg, ltg, pose_coords, save_img_dir_, timestamp_length):
    step_geo_grid = step_geo_grid.squeeze(0)  # 变为[10, 27, 300, 300]
    step_uncertainty = step_uncertainty.squeeze(0)  # 变为[10, 27, 300, 300]
    for sem_lbl in [1, 3, 4, 5, 6, 13, 19]:
        class_name = name_mapping_27.get(sem_lbl, "未知类别")
        for t in range(timestamp_length):
            # 1. 提取该类别的预测图（概率图）
            target_pred = step_geo_grid[t, sem_lbl, :, :].unsqueeze(0)  # [1, H, W]
            # ZHJD: 将等于 1/C 的位置置为 0. 去除屏幕边缘的黄色区域，为了美观
            mask = (target_pred == (1.0 / 27.0))
            target_pred[mask] = 0.0
            target_pred = target_pred.permute(1, 2, 0).cpu().numpy() * 255.0

            # 2. 提取该类别的不确定性图
            target_uncertainty = step_uncertainty[t, sem_lbl, :, :].unsqueeze(0)
            target_uncertainty = target_uncertainty.permute(1, 2, 0).cpu().numpy()
            target_uncertainty /= np.amax(target_uncertainty)+ 1e-6  # 避免除 0
            target_uncertainty = target_uncertainty * 255.0
            #  3. 获取整个语义地图的彩色图
            # color_sem_grid = colorize_grid(sg.sem_grid.unsqueeze(1))
            # im = color_sem_grid[0, 0, :, :, :].permute(1, 2, 0).cpu().numpy()
            color_sem_grid = colorize_grid(step_geo_grid[t].unsqueeze(0).unsqueeze(0))  # shape: [1, 1, H, W, 3]
            im = color_sem_grid[0, 0].permute(1, 2, 0).cpu().numpy()

            #  5. 裁剪中心区域（100x100）
            # crop viz inputs to 128 x 128
            area_size = 100  # area around the agent to be evaluated
            # 只关注 agent 周围的局部区域，避免图太大
            area_start = int((im.shape[0] / 2) - (area_size / 2))
            area_end = int((im.shape[0] / 2) + (area_size / 2))
            # 把彩色语义图、不确定性图、预测图都裁成 100x100
            im = im[area_start:area_end, area_start:area_end, :]
            target_uncertainty = target_uncertainty[area_start:area_end, area_start:area_end, :]
            target_pred = target_pred[area_start:area_end, area_start:area_end, :]

            #  6. 平移坐标（匹配裁剪后坐标系）
            # translate coords   减去 area_start 是为了把坐标对齐到裁剪后的图像中
            # ltg[0, 0, 0] -= area_start
            # ltg[0, 0, 1] -= area_start
            pose_x = pose_coords_list[t, 0, 0, 0].item() - area_start
            pose_y = pose_coords_list[t, 0, 0, 1].item() - area_start

            # 7. 可视化并保存图片
            # 把三个图（语义地图、预测图、不确定性图）用 matplotlib 拼成 3 个 subplot
            # 其中第一张图上添加了 agent 当前的位置（蓝色）和目标点位置（洋红色）
            arr = [im, target_pred, target_uncertainty]
            plt.figure(figsize=(20, 15))
            for i, data in enumerate(arr):
                ax = plt.subplot(1, 3, i + 1)
                ax.axis('off')
                plt.imshow(data)
                if i == 0:
                    plt.scatter(pose_x, pose_y, color="blue", s=50)
                    # plt.scatter(ltg[0, 0, 0], ltg[0, 0, 1], color="magenta", s=50)

            # 8. 保存图像为 PNG
            filename = f"{class_name}_time-{t}.png"
            filepath = save_img_dir_ + filename
            plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=200)
            plt.close()
    print(f"✅ 已保存方差的结果: {save_img_dir_}")



# zhjd 定制
#  step_geo_grid.shape:  torch.Size([1, 10, 27, 300, 300])
#  step_uncertainty.shape:  torch.Size([1, 10, 27, 300, 300]
def save_uncertainty_ros(step_geo_grid, step_uncertainty, pose_coords_list, save_img_dir_, global_time):
    step_geo_grid = step_geo_grid.squeeze(0)  # 变为[10, 27, 300, 300]
    step_uncertainty = step_uncertainty.squeeze(0)  # 变为[10, 27, 300, 300]
    for sem_lbl in [1, 3, 4, 5, 6, 13, 19]:
        class_name = name_mapping_27.get(sem_lbl, "未知类别")
        t = 0
        # 1. 提取该类别的预测图（概率图）
        target_pred = step_geo_grid[t, sem_lbl, :, :].unsqueeze(0)  # [1, H, W]
        # ZHJD: 将等于 1/C 的位置置为 0. 去除屏幕边缘的黄色区域，为了美观
        mask = (target_pred == (1.0 / 27.0))
        target_pred[mask] = 0.0
        target_pred = target_pred.permute(1, 2, 0).cpu().numpy() * 255.0

        # 2. 提取该类别的不确定性图
        target_uncertainty = step_uncertainty[t, sem_lbl, :, :].unsqueeze(0)
        target_uncertainty = target_uncertainty.permute(1, 2, 0).cpu().numpy()
        target_uncertainty /= np.amax(target_uncertainty)+ 1e-6  # 避免除 0
        target_uncertainty = target_uncertainty * 255.0
        #  3. 获取整个语义地图的彩色图
        # color_sem_grid = colorize_grid(sg.sem_grid.unsqueeze(1))
        # im = color_sem_grid[0, 0, :, :, :].permute(1, 2, 0).cpu().numpy()
        color_sem_grid = colorize_grid(step_geo_grid[t].unsqueeze(0).unsqueeze(0))  # shape: [1, 1, H, W, 3]
        im = color_sem_grid[0, 0].permute(1, 2, 0).cpu().numpy()

        #  5. 裁剪中心区域（100x100）
        # crop viz inputs to 128 x 128
        area_size = 100  # area around the agent to be evaluated
        # 只关注 agent 周围的局部区域，避免图太大
        area_start = int((im.shape[0] / 2) - (area_size / 2))
        area_end = int((im.shape[0] / 2) + (area_size / 2))
        # 把彩色语义图、不确定性图、预测图都裁成 100x100
        im = im[area_start:area_end, area_start:area_end, :]
        target_uncertainty = target_uncertainty[area_start:area_end, area_start:area_end, :]
        target_pred = target_pred[area_start:area_end, area_start:area_end, :]

        #  6. 平移坐标（匹配裁剪后坐标系）
        # translate coords   减去 area_start 是为了把坐标对齐到裁剪后的图像中
        # ltg[0, 0, 0] -= area_start
        # ltg[0, 0, 1] -= area_start
        pose_x = pose_coords_list[t, 0, 0, 0].item() - area_start
        pose_y = pose_coords_list[t, 0, 0, 1].item() - area_start

        # 7. 可视化并保存图片
        # 把三个图（语义地图、预测图、不确定性图）用 matplotlib 拼成 3 个 subplot
        # 其中第一张图上添加了 agent 当前的位置（蓝色）和目标点位置（洋红色）
        arr = [im, target_pred, target_uncertainty]
        plt.figure(figsize=(20, 15))
        for i, data in enumerate(arr):
            ax = plt.subplot(1, 3, i + 1)
            ax.axis('off')
            plt.imshow(data)
            if i == 0:
                plt.scatter(pose_x, pose_y, color="blue", s=50)
                # plt.scatter(ltg[0, 0, 0], ltg[0, 0, 1], color="magenta", s=50)

        # 8. 保存图像为 PNG
        filename = f"{class_name}_time-{global_time}.png"
        filepath = save_img_dir_ + filename
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()
    print(f"✅ 已保存方差的结果: {save_img_dir_}")


# 将语义地图（semantic map）、预测结果和不确定性图可视化并保存为一张图片
# test_ds	测试集对象，含点云等信息
# sg	语义地图对象（semantic grid），含有 sem_grid 和 per_class_uncertainty_map
# sem_lbl	要可视化的语义类别索引（如“桌子”、“椅子”等）
# abs_pose	当前 agent 的世界坐标位置
# ltg	long-term goal（目标点）坐标
# pose_coords	当前 agent 在栅格地图中的坐标
# agent_height	agent 身高，用于投影点云
# save_img_dir_	保存图片的路径前缀
# t	当前时间步编号（用于命名）
def save_visual_steps(test_ds, sg, sem_lbl, abs_pose, ltg, pose_coords, agent_height, save_img_dir_, t):
    # 1. 提取该类别的预测图（概率图）
    target_pred = sg.sem_grid[:,sem_lbl,:,:]
    target_pred = target_pred.permute(1,2,0).cpu().numpy()*255.0
    # 2. 提取该类别的不确定性图
    target_uncertainty = sg.per_class_uncertainty_map[:,sem_lbl,:,:].permute(1,2,0).cpu().numpy()
    target_uncertainty /= np.amax(target_uncertainty)
    target_uncertainty = target_uncertainty*255.0
    #  3. 获取整个语义地图的彩色图
    color_sem_grid = colorize_grid(sg.sem_grid.unsqueeze(1))
    im = color_sem_grid[0,0,:,:,:].permute(1,2,0).cpu().numpy()
    #  4. 获取地面真实语义 crop 图（用于评估）
    pose_ = np.asarray(abs_pose).reshape(1,3)
    gt_grid_crops_objects = map_utils.get_gt_crops(pose_, test_ds.pcloud, test_ds.label_seq_objects, agent_height,
                                            test_ds.grid_dim, test_ds.crop_size, test_ds.cell_size)
    color_gt_crop = colorize_grid(gt_grid_crops_objects.unsqueeze(0))
    im_gt_crop = color_gt_crop[0,0,:,:,:].permute(1,2,0).cpu().numpy()

    #  5. 裁剪中心区域（100x100）
    # crop viz inputs to 128 x 128
    area_size = 100 # area around the agent to be evaluated
    area_start = int( (im.shape[0] / 2) - (area_size / 2) )
    area_end = int( (im.shape[0] / 2) + (area_size / 2) )
    im = im[area_start:area_end, area_start:area_end,:]
    target_uncertainty = target_uncertainty[area_start:area_end, area_start:area_end,:]
    target_pred = target_pred[area_start:area_end, area_start:area_end,:]

    # translate coords
    ltg[0,0,0] -= area_start
    ltg[0,0,1] -= area_start
    pose_coords[0,0,0] -= area_start
    pose_coords[0,0,1] -= area_start

    # 7. 可视化并保存图片
    # 把三个图（语义地图、预测图、不确定性图）用 matplotlib 拼成 3 个 subplot
    # 其中第一张图上添加了 agent 当前的位置（蓝色）和目标点位置（洋红色）
    arr = [ im,
            target_pred,
            target_uncertainty
            ]
    n=len(arr)
    plt.figure(figsize=(20 ,15))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i+1)
        ax.axis('off')
        plt.imshow(data)
        if i==0:
            plt.scatter(ltg[0,0,0], ltg[0,0,1], color="magenta", s=50)
            plt.scatter(pose_coords[0,0,0], pose_coords[0,0,1], color="blue", s=50)

    # 8. 保存图像为 PNG
    plt.savefig(save_img_dir_+str(t)+'.png', bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()


# def save_map_pred_steps(spatial_in, spatial_pred, objects_pred, ego_img_segm, save_img_dir_, t):
#
#     color_spatial_in = colorize_grid(spatial_in.unsqueeze(0), color_mapping=3)
#     im_spatial_in = color_spatial_in[0,0,:,:,:].permute(1,2,0).cpu().numpy()
#
#     color_spatial_pred = colorize_grid(spatial_pred, color_mapping=3)
#     im_spatial_pred = color_spatial_pred[0,0,:,:,:].permute(1,2,0).cpu().numpy()
#
#     color_objects_pred = colorize_grid(objects_pred, color_mapping=27)
#     im_objects_pred = color_objects_pred[0,0,:,:,:].permute(1,2,0).cpu().numpy()
#
#     color_ego_img_segm = colorize_grid(ego_img_segm, color_mapping=27)
#     im_ego_img_segm = color_ego_img_segm[0,0,:,:,:].permute(1,2,0).cpu().numpy()
#
#     arr = [ im_spatial_in,
#             im_spatial_pred,
#             im_objects_pred,
#             im_ego_img_segm
#             ]
#     n=len(arr)
#     plt.figure(figsize=(20 ,15))
#     for i, data in enumerate(arr):
#         ax = plt.subplot(1, n, i+1)
#         ax.axis('off')
#         plt.imshow(data)
#     plt.savefig(save_img_dir_+"map_step_"+str(t)+'.png', bbox_inches='tight', pad_inches=0, dpi=200)
#     plt.close()







# zhjd
def add_border(img, color=(255, 0, 0), thickness=5):
    # 如果是 (C,H,W)，转为 (H,W,C)
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.permute(1, 2, 0)

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.copy()

    h, w, c = img.shape
    img[:thickness, :, :] = color  # top
    img[-thickness:, :, :] = color  # bottom
    img[:, :thickness, :] = color  # left
    img[:, -thickness:, :] = color  # right
    return img

def to_5d(t):
    t = torch.tensor(t)
    while t.ndim < 5:
        t = t.unsqueeze(0)  # 在最前面添加一个新维度。例如原来是 (64, 64) → 变成 (1, 64, 64)
    return t

fix_extract = 0

# === 用 colorize_grid 上色 ===
def color_and_extract(grid, color_mapping):
    colorized = colorize_grid(to_5d(grid), color_mapping=color_mapping)
    # 输出可能是 (3,H,W) 或 (1,3,H,W) 或 (1,1,3,H,W)
    colorized = torch.tensor(colorized)
    # 将五/四维度转为三维度
    if colorized.ndim == 5:
        colorized = colorized[0, fix_extract]  # 默认显示的是第二维（time）的第 0 帧。
    elif colorized.ndim == 4:
        colorized = colorized[fix_extract]
    # 现在 colorized 应为 (3,H,W)
    colorized.permute(1, 2, 0) # 转为 (H,W,3)
    colorized_border = add_border(colorized, color=(10, 10, 10), thickness=1)
    return colorized_border

def show_image_color_and_extract(tensor_or_array, title="image", color_mapping=27):
    if isinstance(tensor_or_array, torch.Tensor):
        img = tensor_or_array.detach().cpu().numpy()
    else:
        img = np.array(tensor_or_array)
    img = color_and_extract(img, color_mapping=color_mapping)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_image(tensor_or_array, title="image"):
    if isinstance(tensor_or_array, torch.Tensor):
        img = tensor_or_array.detach().cpu().numpy()
    else:
        img = np.array(tensor_or_array)
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


def show_image_sseg_2d_label(tensor_or_array, title="image"):

    if isinstance(tensor_or_array, torch.Tensor):
        img = tensor_or_array.detach().cpu().numpy()
    else:
        img = np.array(tensor_or_array)

    # 2️⃣ 确保是二维标签图 [H,W]
    # assert img.ndim == 2, f" [zhjd-debug] Expected 2D label map, got shape {img.shape}"
    if img.ndim == 5:
        img = img[0, fix_extract, 0]  # 默认显示的是第二维（time）的第 0 帧。
    elif img.ndim == 4:
        img = img[0, fix_extract]
    elif img.ndim == 4:
        img = img[fix_extract]

    H, W = img.shape
    rgb_img = np.zeros((H, W, 3), dtype=np.uint8)

    # 3️⃣ 将每个 label 转成 RGB
    for lbl, color in color_mapping_27.items():
        mask = img == lbl
        rgb_img[mask] = color

    # 4️⃣ 显示结果
    plt.imshow(rgb_img)
    plt.title(title)
    plt.axis('off')
    plt.show()



def save_all_infos_and_mapprediction_origin(batch, pred_maps_objects, savepath, name):
    # batch: ['abs_pose', 'ego_grid_crops_spatial', 'step_ego_grid_crops_spatial', 'gt_grid_crops_spatial', 'gt_grid_crops_objects', 'images', 'ssegs', 'depth_imgs', 'pred_ego_crops_sseg', 'step_ego_grid_27']
    images = batch['images']
    ssegs = batch['gt_segm']  # 物体分割真值
    depth_imgs = batch['depth_imgs']
    pred_ego_crops_sseg = batch['pred_ego_crops_sseg']  # net3的输出

    step_ego_grid_27 = batch['step_ego_grid_27']
    ##### RSMP的输出 pred_maps_objects
    gt_grid_crops_objects = batch['gt_grid_crops_objects']

    ego_grid_crops_spatial = batch['ego_grid_crops_spatial']  # 当前帧几何地图
    step_ego_grid_crops_spatial = batch['step_ego_grid_crops_spatial']  # 多帧融合几何地图
    gt_grid_crops_spatial = batch['gt_grid_crops_spatial']  # 语义地图真值

    B, T, _, cH, cW = step_ego_grid_27.shape
    for t in range(T):
        # print(f"🕒 时间步 {t}")
        images_single = images[0, t, :, :, :].detach().cpu().permute(1, 2, 0).numpy()
        ssegs_single = ssegs[0, t, :, :, :].detach().cpu().permute(1, 2, 0).numpy()
        depth_imgs_single = depth_imgs[0, t, :, :, :].detach().cpu().permute(1, 2, 0).numpy()

        step_ego_grid_27_single = color_and_extract(step_ego_grid_27[0, t, :, :, :], 27)
        pred_maps_objects_single = color_and_extract(pred_maps_objects[0, t, :, :, :], 27)
        gt_grid_crops_objects_single = color_and_extract(gt_grid_crops_objects[0, t, :, :, :], 27)

        ego_grid_crops_spatial_single = color_and_extract(ego_grid_crops_spatial[0, t, :, :, :], 3)
        step_ego_grid_crops_spatial_single = color_and_extract(step_ego_grid_crops_spatial[0, t, :, :, :], 3)
        gt_grid_crops_spatial_single = color_and_extract(gt_grid_crops_spatial[0, t, :, :, :], 3)

        # fig, axs = plt.subplots(3, 3, figsize=(20, 20))
        # axs = axs.flatten()
        #
        # axs[0].imshow(images_single)
        # axs[0].set_title("RGB Image")
        #
        # axs[1].imshow(ssegs_single, cmap='tab20')
        # axs[1].set_title("GT Segmentation")
        #
        # axs[2].imshow(depth_imgs_single, cmap='viridis')
        # axs[2].set_title("Depth")
        #
        # axs[3].imshow(step_ego_grid_27_single, cmap='viridis')
        # axs[3].set_title("RGB Project")
        #
        # axs[4].imshow(pred_maps_objects_single, cmap='viridis')
        # axs[4].set_title("Refined Semantic Map")
        #
        # axs[5].imshow(gt_grid_crops_objects_single, cmap='viridis')
        # axs[5].set_title("GT Semantic Map")
        #
        # axs[6].imshow(ego_grid_crops_spatial_single, cmap='viridis')
        # axs[6].set_title("Depth Project")
        #
        # axs[7].imshow(step_ego_grid_crops_spatial_single, cmap='viridis')
        # axs[7].set_title("Refined Occupied Map")
        #
        # axs[8].imshow(gt_grid_crops_spatial_single, cmap='viridis')
        # axs[8].set_title("GT Occupied Map")
        #
        # for ax in axs:
        #     ax.axis('off')
        #
        # plt.tight_layout()
        # plt.show()

        # === 九宫格保存本地===
        fig, axs = plt.subplots(3, 3, figsize=(20, 20))
        axs = axs.flatten()

        imgs = [
            (images_single, "RGB Image", None),
            (ssegs_single, "GT Segmentation", 'tab20'),
            (depth_imgs_single, "Depth", 'viridis'),
            (step_ego_grid_27_single, "RGB Project", None),
            (pred_maps_objects_single, "Refined Semantic Map", None),
            (gt_grid_crops_objects_single, "GT Semantic Map", None),
            (ego_grid_crops_spatial_single, "Depth Project", None),
            (step_ego_grid_crops_spatial_single, "Refined Occupied Map", None),
            (gt_grid_crops_spatial_single, "GT Occupied Map", None)
        ]

        for i, (img, title, cmap) in enumerate(imgs):
            axs[i].imshow(img, cmap=cmap)
            axs[i].set_title(title)
            axs[i].axis('off')

        plt.tight_layout()

        # === 保存图片 ===
        save_file = os.path.join(savepath, f"{name}_t{t}.png")
        plt.savefig(save_file, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()
        print(f"✅ 已保存: {save_file}")



def save_all_infos_and_mapprediction_slam(batch, pred_maps_objects, savepath, name):
    # batch: ['abs_pose', 'ego_grid_crops_spatial', 'step_ego_grid_crops_spatial', 'gt_grid_crops_spatial', 'gt_grid_crops_objects', 'images', 'ssegs', 'depth_imgs', 'pred_ego_crops_sseg', 'step_ego_grid_27']
    images = batch['images']
    ssegs = batch['gt_segm']  # 物体分割真值
    depth_imgs = batch['depth_imgs']
    step_ego_grid_27 = batch['step_ego_grid_27']



    B, T, _, cH, cW = step_ego_grid_27.shape
    for t in range(T):
        # print(f"🕒 时间步 {t}")
        # RGB图
        images_single = images[0, t, :, :, :].detach().cpu().numpy()

        # --- 语义图上色 ---
        ssegs_single = ssegs[0, t, :, :].detach().cpu().numpy()
        segm_color = colorize_sseg(ssegs_single, color_mapping_27)

        # 深度图上色
        depth_imgs_single = depth_imgs[0, t, :, :].detach().cpu().numpy()
        depth_norm = cv2.normalize(depth_imgs_single, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)  # (H, W, 3)

        # step_ego_grid_27_single = color_and_extract(step_ego_grid_27[0, t, :, :, :], 27)
        # pred_maps_objects_single = color_and_extract(pred_maps_objects[0, t, :, :, :], 27)  # RSMP的输出 pred_maps_objects
        step_ego_grid_27_single = colorEncode(step_ego_grid_27[0, t, :, :, :].argmax(axis=0))

        # 是否将预测图叠加到step_ego_grid_27上
        flag_map_overlay = 0
        if flag_map_overlay == 0:
            pred_maps_objects_single = colorEncode(pred_maps_objects[0, t, :, :, :].argmax(axis=0))
        elif flag_map_overlay == 1:
            pred_maps_objects_bottom = step_ego_grid_27[0, t, :, :, :]      # [27,64,64]
            pred_maps_objects_top = pred_maps_objects[0, t, :, :, :]        # [27,64,64]
            # 计算 bottom 层每个栅格的 argmax
            val_bottom, idx_bottom = torch.max(pred_maps_objects_bottom, dim=0)  # idx_bottom shape: [64, 64]
            # 2. 构造掩码 (Mask)：找出 argmax 为 0 的位置
            fail_mask = (idx_bottom == 0) | (idx_bottom == 17)
            # 3. 初始化最终的融合地图
            # 我们先完整复制 bottom 的数据
            fused_map = pred_maps_objects_bottom.clone()
            # 4. 执行叠加逻辑：
            # 在所有 fail_mask 为 True 的坐标点，用 top 的 27 维概率向量替换掉 bottom 的
            # 这里使用广播机制处理 27 个通道
            fused_map[:, fail_mask] = pred_maps_objects_top[:, fail_mask]
            pred_maps_objects_single = colorEncode(fused_map.argmax(axis=0))
        elif flag_map_overlay == 2:
            pred_maps_objects_bottom = step_ego_grid_27[0, t, :, :, :]      # [27,64,64]
            pred_maps_objects_top = pred_maps_objects[0, t, :, :, :]        # [27,64,64]
            # 计算 bottom 层每个栅格的 argmax
            val_bottom, idx_bottom = torch.max(pred_maps_objects_bottom, dim=0)  # idx_bottom shape: [64, 64]
            # 2. 构造掩码 (Mask)：找出 argmax 为 0 的位置
            fail_mask = (idx_bottom != 15) # 只保留贝叶斯更新中的墙
            # 3. 初始化最终的融合地图
            # 我们先完整复制 bottom 的数据
            fused_map = pred_maps_objects_bottom.clone()
            # 4. 执行叠加逻辑：
            # 在所有 fail_mask 为 True 的坐标点，用 top 的 27 维概率向量替换掉 bottom 的
            # 这里使用广播机制处理 27 个通道
            fused_map[:, fail_mask] = pred_maps_objects_top[:, fail_mask]
            pred_maps_objects_single = colorEncode(fused_map.argmax(axis=0))

        # === 六宫格保存本地===
        fig, axs = plt.subplots(3, 2, figsize=(30, 30))
        axs = axs.flatten()

        imgs = [
            (images_single, "RGB Image", None),
            (ssegs_single, "GT Segmentation", 'tab20'),
            (depth_imgs_single, "Depth", 'viridis'),
            (step_ego_grid_27_single, "RGB Project", None),
            (pred_maps_objects_single, "Refined Semantic Map", None),
        ]

        for i, (img, title, cmap) in enumerate(imgs):
            axs[i].imshow(img, cmap=cmap)
            axs[i].set_title(title)
            axs[i].axis('off')

        plt.tight_layout()

        # === 保存图片 ===
        save_file = os.path.join(savepath, f"{name}_t{t}.png")
        plt.savefig(save_file, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()
        print(f"✅ 已保存: {save_file}")



def save_all_infos_and_mapprediction_Global(batch, local_pred_maps_objects, global_maps_objects, savepath, name):
    # batch: ['abs_pose', 'ego_grid_crops_spatial', 'step_ego_grid_crops_spatial', 'gt_grid_crops_spatial',
    # 'gt_grid_crops_objects', 'images', 'ssegs', 'depth_imgs', 'pred_ego_crops_sseg', 'step_ego_grid_27']
    images = batch['images']
    ssegs = batch['gt_segm']  # 物体分割真值
    depth_imgs = batch['depth_imgs']
    pred_ego_crops_sseg = batch['pred_ego_crops_sseg']  # net3的输出

    step_ego_grid_27 = batch['step_ego_grid_27']
    ##### RSMP的输出 pred_maps_objects
    gt_grid_crops_objects = batch['gt_grid_crops_objects']

    ego_grid_crops_spatial = batch['ego_grid_crops_spatial']  # 当前帧几何地图
    step_ego_grid_crops_spatial = batch['step_ego_grid_crops_spatial']  # 多帧融合几何地图
    gt_grid_crops_spatial = batch['gt_grid_crops_spatial']  # 语义地图真值

    B, T, _, cH, cW = step_ego_grid_27.shape
    for t in range(T):
        # print(f"🕒 时间步 {t}")
        images_single = images[0, t, :, :, :].detach().cpu().permute(1, 2, 0).numpy()
        ssegs_single = ssegs[0, t, :, :, :].detach().cpu().permute(1, 2, 0).numpy()
        depth_imgs_single = depth_imgs[0, t, :, :, :].detach().cpu().permute(1, 2, 0).numpy()

        step_ego_grid_27_single = color_and_extract(step_ego_grid_27[0, t, :, :, :], 27)
        pred_maps_objects_single = color_and_extract(local_pred_maps_objects[0, t, :, :, :], 27)
        gt_grid_crops_objects_single = color_and_extract(gt_grid_crops_objects[0, t, :, :, :], 27)

        global_maps_objects_single = color_and_extract(global_maps_objects[0, t, :, :, :], 27)
        # ego_grid_crops_spatial_single = color_and_extract(ego_grid_crops_spatial[0, t, :, :, :], 3)
        step_ego_grid_crops_spatial_single = color_and_extract(step_ego_grid_crops_spatial[0, t, :, :, :], 3)
        gt_grid_crops_spatial_single = color_and_extract(gt_grid_crops_spatial[0, t, :, :, :], 3)


        # === 九宫格保存本地===
        fig, axs = plt.subplots(3, 3, figsize=(20, 20))
        axs = axs.flatten()

        imgs = [
            (images_single, "RGB Image", None),
            (ssegs_single, "GT Segmentation", 'tab20'),
            (depth_imgs_single, "Depth", 'viridis'),
            (step_ego_grid_27_single, "RGB Project", None),
            (pred_maps_objects_single, "Refined Semantic Map", None),
            (gt_grid_crops_objects_single, "GT Semantic Map", None),
            (global_maps_objects_single, "Global Semantic Map", None),
            (step_ego_grid_crops_spatial_single, "Refined Occupied Map", None),
            (gt_grid_crops_spatial_single, "GT Occupied Map", None)
        ]

        for i, (img, title, cmap) in enumerate(imgs):
            axs[i].imshow(img, cmap=cmap)
            axs[i].set_title(title)
            axs[i].axis('off')

        plt.tight_layout()

        # === 保存图片 ===
        save_file = os.path.join(savepath, f"{name}_t{t}.png")
        plt.savefig(save_file, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()
    print(f"✅ 已保存输入信息和EGO语义地图: {savepath}")


def save_all_infos_and_mapprediction_Global_forSLAM(batch, local_pred_maps_objects, global_maps_objects, savepath, name):
    # batch: ['abs_pose', 'ego_grid_crops_spatial', 'step_ego_grid_crops_spatial', 'gt_grid_crops_spatial',
    # 'gt_grid_crops_objects', 'images', 'ssegs', 'depth_imgs', 'pred_ego_crops_sseg', 'step_ego_grid_27']
    images = batch['images']
    ssegs = batch['gt_segm']  # 物体分割真值
    depth_imgs = batch['depth_imgs']

    step_ego_grid_27 = batch['step_ego_grid_27']

    B, T, _, cH, cW = step_ego_grid_27.shape
    for t in range(T):
        # print(f"🕒 时间步 {t}")
        images_single = images[0, t, :, :, :].detach().cpu().permute(1, 2, 0).numpy()
        ssegs_single = ssegs[0, t, :, :, :].detach().cpu().permute(1, 2, 0).numpy()
        depth_imgs_single = depth_imgs[0, t, :, :, :].detach().cpu().permute(1, 2, 0).numpy()

        step_ego_grid_27_single = color_and_extract(step_ego_grid_27[0, t, :, :, :], 27)
        pred_maps_objects_single = color_and_extract(local_pred_maps_objects[0, t, :, :, :], 27)

        global_maps_objects_single = color_and_extract(global_maps_objects[0, t, :, :, :], 27)


        # === 九宫格保存本地===
        fig, axs = plt.subplots(3, 2, figsize=(20, 20))
        axs = axs.flatten()

        imgs = [
            (images_single, "RGB Image", None),
            (ssegs_single, "MIT Segmentation", 'tab20'),
            (depth_imgs_single, "Depth", 'viridis'),
            (step_ego_grid_27_single, "RGB Project", None),
            (pred_maps_objects_single, "Refined Semantic Map", None),
            (global_maps_objects_single, "Global Semantic Map", None)
        ]

        for i, (img, title, cmap) in enumerate(imgs):
            axs[i].imshow(img, cmap=cmap)
            axs[i].set_title(title)
            axs[i].axis('off')

        plt.tight_layout()

        # === 保存图片 ===
        save_file = os.path.join(savepath, f"{name}_t{t}.png")
        plt.savefig(save_file, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()
    print(f"✅ 已保存输入信息和EGO语义地图: {savepath}")


import os
import torch
import matplotlib.pyplot as plt
import numpy as np


def save_Global_forSLAM(global_maps_objects, savepath, name):
    """
    参数:
        global_maps_objects: [B, T, 27, H, W]
        savepath: 保存路径
        name: 文件名前缀
    """
    # 确保保存路径存在
    os.makedirs(savepath, exist_ok=True)

    # 1. 维度提取
    # 注意：这里我们使用 .detach().cpu() 确保数据在内存中处理
    global_maps = global_maps_objects.detach().cpu()
    B, T, C, cH, cW = global_maps.shape

    print(f"开始渲染并保存全局地图序列 (共 {T} 帧)...")

    # 2. 预先创建 Figure 对象，避免在循环中重复创建
    # 这样能极大提高保存速度
    fig, ax = plt.subplots(figsize=(10, 10))

    for t in range(T):
        # 3. 提取当前帧并转换
        # global_maps[0, t] 形状为 [27, H, W]
        # argmax(0) 得到类别索引图 [H, W]
        current_grid = global_maps[0, t]

        # 使用你定义的上色逻辑
        # 假设 color_and_extract 接受 [27, H, W] 并返回 RGB 图像
        global_maps_objects_single = color_and_extract(current_grid, 27)

        # 4. 清除上一帧内容并显示新帧
        ax.clear()
        ax.imshow(global_maps_objects_single)
        ax.set_title(f"{name} - Timestep {t:03d}")
        ax.axis('off')

        # 5. 构造文件名并保存
        save_file = os.path.join(savepath, f"{name}_t{t:03d}.png")
        # 使用较低的 dpi 可以显著加快保存速度，除非你需要极高精度
        plt.savefig(save_file, bbox_inches='tight', pad_inches=0.1, dpi=150)

    # 6. 最后关闭 Figure 释放资源
    plt.close(fig)
    print(f"✅ 已保存 {T} 张全局语义地图至: {savepath}")


def save_Global_forROS(global_maps_objects, global_map_uncertainty, savepath, name):
    """
    参数:
        global_maps_objects: [B, T, 27, H, W]
        global_uncertainty: [B, T, 27, H, W]
        savepath: 保存路径
        name: 文件名前缀
    """
    # 确保保存路径存在
    os.makedirs(savepath, exist_ok=True)

    # 1. 维度提取
    # 注意：这里我们使用 .detach().cpu() 确保数据在内存中处理
    global_maps = global_maps_objects.detach().cpu()
    global_uncertainty = global_map_uncertainty.detach().cpu()
    B, T, C, cH, cW = global_maps.shape

    print(f"开始渲染并保存全局地图序列 (共 {T} 帧)...")

    # 2. 预先创建 Figure 对象，避免在循环中重复创建
    #  修改 Figure 为 1x2 的布局
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for t in range(T):
        # 3. 提取当前帧并转换
        # global_maps[0, t] 形状为 [27, H, W]
        # argmax(0) 得到类别索引图 [H, W]
        current_grid = global_maps[0, t]

        # 使用你定义的上色逻辑
        # 假设 color_and_extract 接受 [27, H, W] 并返回 RGB 图像
        global_maps_objects_single = color_and_extract(current_grid, 27)

        # --- 不确定性部分 (sigma_map) ---
        current_uncertainty = global_uncertainty[0, t]
        max_uncertainty, _ = torch.max(current_uncertainty, dim=0, keepdim=True)
        sigma_map = torch.sqrt(max_uncertainty).squeeze().numpy() # [H, W]

        # 2. 渲染两张子图
        axes[0].clear()
        axes[0].imshow(global_maps_objects_single)
        axes[0].set_title("Semantic Map")
        axes[0].axis('off')

        axes[1].clear()
        # 使用 'jet' 或 'magma' 热力图显示不确定性，越亮代表越不确定
        axes[1].imshow(sigma_map, vmin=0, vmax=1)
        axes[1].set_title("Uncertainty (Sigma)")
        axes[1].axis('off')

        # 5. 构造文件名并保存
        save_file = os.path.join(savepath, f"{name}.png")
        # 使用较低的 dpi 可以显著加快保存速度，除非你需要极高精度
        plt.savefig(save_file, bbox_inches='tight', pad_inches=0.1, dpi=150)

    # 6. 最后关闭 Figure 释放资源
    plt.close(fig)
    print(f"✅ 已保存 {T} 张全局语义地图至: {savepath}")

# zhjd
def colorEncode(label_map, color_mapping=color_mapping_27):
    """
    将单通道标签图转换为彩色图像（RGB）。

    参数:
        label_map: numpy.ndarray, shape (H, W)，每个像素是类别 ID
        color_mapping: dict[int, tuple[int, int, int]]，类别 ID → RGB 颜色

    返回:
        RGB 图像: numpy.ndarray, shape (H, W, 3)，dtype=uint8
    """
    # 保证输入是 numpy，并 squeeze 掉多余维度
    if isinstance(label_map, torch.Tensor):
        label_map = label_map.detach().cpu().numpy()

    label_map = np.squeeze(label_map)  # 去掉多余维度，例如 (1, H, W) → (H, W)

    if label_map.ndim != 2:
        raise ValueError(f"[colorEncode] 输入的 label_map 必须是 2D，但实际是 {label_map.shape}")

    h, w = label_map.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    for label_id, color in color_mapping.items():
        color_img[label_map == label_id] = color

    return color_img




# zhjd
def colorEncode(label_map):
    """
    将单通道标签图转换为彩色图像（RGB）。

    参数:
        label_map: numpy.ndarray, shape (H, W)，每个像素是类别 ID
        color_mapping: dict[int, tuple[int, int, int]]，类别 ID → RGB 颜色

    返回:
        RGB 图像: numpy.ndarray, shape (H, W, 3)，dtype=uint8
    """
    color_mapping = color_mapping_27

    # 保证输入是 numpy，并 squeeze 掉多余维度
    if isinstance(label_map, torch.Tensor):
        label_map = label_map.detach().cpu().numpy()

    label_map = np.squeeze(label_map)  # 去掉多余维度，例如 (1, H, W) → (H, W)

    if label_map.ndim != 2:
        raise ValueError(f"[colorEncode] 输入的 label_map 必须是 2D，但实际是 {label_map.shape}")

    h, w = label_map.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    for label_id, color in color_mapping.items():
        color_img[label_map == label_id] = color

    return color_img


def colorize_sseg(sseg, color_map):
    h, w = sseg.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)

    for label_id, color in color_map.items():
        mask = sseg == label_id
        color_image[mask] = color

    return color_image