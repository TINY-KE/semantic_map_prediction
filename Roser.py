import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataloader import HabitatDataOffline
from datasets.dataloader import HabitatDataOfflineMPv2, HabitatDataOfflineSLAM

from models.predictors import get_predictor_rsmp
from models.img_segmentation import get_img_segmentor_from_options
import datasets.util.utils as utils
import datasets.util.viz_utils as viz_utils
import datasets.util.map_utils as map_utils
import test_utils as tutils
from models.semantic_grid import SemanticGrid
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import metrics
import json
import cv2
from models.networks.resnetUnet import Visualizer
import time

from collections import deque
import rospy
import message_filters
# from nav_msgs.msg import OccupancyGrid
# from geometry_msgs.msg import PoseStamped
from StepEgoMapPose_msgs.msg import StepEgoMapPose
from geometry_msgs.msg import PoseStamped # 确保在文件顶部导入

from ros_utils.FrontierExtractor import FrontierExtractor
from ros_utils.SemanticMapPublisher import  SemanticMarkerPublisher, AsyncDualSemanticPublisher

class RosTester(object):
    """ Implements testing for prediction models
    """
    def __init__(self, options):
        self.options = options
        print("options:")
        for k in self.options.__dict__.keys():
            print(k, self.options.__dict__[k])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 0. 完整加载 Ensemble 模型逻辑
        self.models_dict = {}  # keys are the ids of the models in the ensemble
        ensemble_exp_rsmp = os.listdir(
            self.options.ensemble_dir_rsmp)  # ensemble_dir should be a dir that holds multiple experiments
        ensemble_exp_rsmp.sort()  # in case the models are numbered put them in order
        for n in range(self.options.ensemble_size):
            print("     [zhjd-slam-search] RosTester Init Loading model ", n)
            self.models_dict[n] = {'predictor_model': get_predictor_rsmp(self.options)}
            self.models_dict[n] = {k: v.to(self.device) for k, v in self.models_dict[n].items()}

            # Needed only for models trained with multi-gpu setting
            self.models_dict[n]['predictor_model'] = nn.DataParallel(self.models_dict[n]['predictor_model'])

            checkpoint_dir = self.options.ensemble_dir_rsmp + "/" + ensemble_exp_rsmp[n]
            print('checkpoint_dir', checkpoint_dir)

            latest_checkpoint = tutils.get_latest_model(save_dir=checkpoint_dir)
            print("Model", n, "loading checkpoint", latest_checkpoint)
            self.models_dict[n] = tutils.load_model(models=self.models_dict[n], checkpoint_file=latest_checkpoint)
            self.models_dict[n]["predictor_model"].eval()




        # 1. 状态缓冲区：T=10
        self.batch_size = 1
        self.grid_buffer = deque(maxlen=self.batch_size)
        self.pose_buffer = deque(maxlen=self.batch_size)
        self.seq_buffer = deque(maxlen=self.batch_size)

        # 2. 初始化全局语义地图[修改地图中心]
        self.spatial_labels = 3
        self.object_labels = 27
        # self.grid_dim = (200, 200)  # 749办公室 7*12米
        # 为什么以下两者是相反的
        h_length_positive = 500  # 对应的是机器人坐标系的y轴
        h_length_negative = 500
        w_length_positive = 500  # 对应的是机器人坐标系的x轴
        w_length_negative = 500

        self.grid_dim = (h_length_positive+h_length_negative, w_length_positive+w_length_negative)  # 749办公室 7*12米
        self.cell_size = 0.1
        self.crop_size = (64, 64)
        self.sg = SemanticGrid(self.batch_size, self.grid_dim, self.crop_size[0], self.cell_size,
                          spatial_labels=self.spatial_labels, object_labels=self.object_labels, origin=None )

        # 3.边界和地图处理
        self.FrontierExtractor = FrontierExtractor(self.sg)
        self.semantic_map_publisher = AsyncDualSemanticPublisher(raw_topic="/semantic_global_map", filtered_topic="/semantic_global_map_free_only")
        self.semantic_map_publisher_height = 3

        # 5. ROS 订阅和发布
        rospy.Subscriber("/step_ego_map_pose", StepEgoMapPose, self.ros_callback)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal_sigma', PoseStamped, queue_size=1)

        self.num_flag = 0
        self.prev_time = time.time()  # 初始化时间戳
        self.fps = 0.0




    def ros_callback(self, msg):
        """处理自定义消息"""
        # print(" [1] Start ros_callback ")
        # 将 flat data 还原并转为 Tensor
        grid_np = np.array(msg.grid27.data, dtype=np.float32).reshape(27, 64, 64)
        robot_pose_list = [pose.data for pose in msg.robot_pose]
        robot_pose_np = np.array(robot_pose_list, dtype=np.float32)

        self.grid_buffer.append(torch.from_numpy(grid_np).float())
        self.pose_buffer.append(torch.from_numpy(robot_pose_np).float())
        self.seq_buffer.append(msg.header.seq)  # ✨ 同步记录
        if len(self.grid_buffer) == self.batch_size:
            self.run_online_inference()

    def run_online_inference(self):
        current_seqs = list(self.seq_buffer)
        last_seq = current_seqs[-1]  # 通常使用本批次最后一帧的序号作为保存标识
        print(f"\n ------------  {self.num_flag} 开始推理，对应 ROS {last_seq} -------------")
        # 构造 Batch: [B=1, T=1, C=27, H=64, W=64]
        grid_seq = torch.stack(list(self.grid_buffer))  # [batch_size, 27, 64, 64]
        pose_seq = torch.stack(list(self.pose_buffer))  # [batch_size, 3]


        print("[当前机器人的位置]： ",pose_seq)
        batch = {
            'step_ego_grid_27': grid_seq.unsqueeze(0).to(self.device), # [1, batch_size, 27, 64, 64]
            'rel_pose': pose_seq.unsqueeze(0).to(self.device),        # [1, batch_size, 3]
            # 设置为全0张量，且保持与 pose_seq 相同的 batch 和 time 维度
            # 'abs_pose': torch.zeros_like(pose_seq).unsqueeze(0).to(self.device)
        }

        # 1. 秒级时间戳（浮点数，含小数）
        time0 = time.time()
        time1 = time.time()
        # print(f"当前秒级时间戳 1: {time.time()}")  # 输出示例：1773070000.123456
        # step_ego_grid_27 = batch['step_ego_grid_27']
        # print("   [zhjd-debug] step_ego_grid_27.shape: ", step_ego_grid_27.shape)
        # viz_utils.show_image_color_and_extract(step_ego_grid_27, "Beyesin Map", 27)

        with torch.no_grad():
            ensemble_outs = []
            for n in range(self.options.ensemble_size):
                out = self.models_dict[n]['predictor_model'](batch, False, step_name="online")
                ensemble_outs.append(out['pred_maps_objects'].clone())

            ensemble_outs = torch.stack(ensemble_outs)

            # 平均预测
            # pred_maps_objects = torch.mean(ensemble_outs, dim=0)
            pred_maps_objects = ensemble_outs[0]

            time2 = time.time()
            print(f" [耗时]推理耗时: {time2 - time1}")  # 输出示例：1773070000.123456
            time1 = time2
            _rel_pose = batch['rel_pose'].squeeze(0)

            # 计算不确定性地图
            # add crop uncertainty to uncertainty map  用于物体搜索的目标选择
            B, T, C, cH, cW = pred_maps_objects.shape
            ### Estimate the variance of each class for each location # 1 x B x T x object_classes x crop_dim x crop_dim
            ensemble_var = torch.zeros((1, B, T, C, cH, cW), dtype=torch.float32).to(self.device)
            for i in range(ensemble_outs.shape[3]):  # num of classes
                ensemble_class = ensemble_outs[:, :, :, i, :, :]
                ensemble_class_var = torch.var(ensemble_class, dim=0, keepdim=True)
                ensemble_var[:, :, :, i, :, :] = ensemble_class_var
            # 方差的取值范围是 [0, 0.25]（概率∈[0,1] 时，方差理论最大值 = 0.25）；
            # 因此你的 per_class_uncertainty_crop 最大值 M ≤ 0.25（实际中通常在 0~0.2 之间）。
            per_class_uncertainty = ensemble_var.squeeze(0)  # B x T x C x cH x cW
            step_uncertainty = self.sg.register_per_class_uncertainty_ros_without_rot(per_class_uncertainty_crop=per_class_uncertainty, pose=_rel_pose)


            # 选择长期目标
            eucost_map = self.get_Epistemic_Uncertainty_cost_map(self.sg)
            robot_pose_world = pose_seq[0].cpu().numpy()
            robot_x_px = int((robot_pose_world[0] - self.FrontierExtractor.target_origin_x) / self.FrontierExtractor.target_cell_size)
            robot_y_px = int((robot_pose_world[1] - self.FrontierExtractor.target_origin_y) / self.FrontierExtractor.target_cell_size)
            robot_pose_px = [robot_x_px, robot_y_px]
            ltg = self.get_long_term_goal(eucost_map, robot_pose_px)
            if ltg is not None:
                print(f" 发布探索点(栅格坐标): {ltg[0]}, {ltg[1]}")
                flag_pub_ltg = True
                if flag_pub_ltg:
                    goal_x_idx = ltg[0]
                    goal_y_idx = ltg[1]
                    # 转换为物理坐标 (单位：米)
                    # 公式：物理位置 = (索引 * 分辨率) + 地图原点
                    # 注意：这里的物理 x/y 映射关系需与你的 SemanticGrid 定义一致
                    world_goal_x = goal_x_idx * self.cell_size + self.sg.origin[0]
                    world_goal_y = goal_y_idx * self.cell_size + self.sg.origin[1]
                    goal_msg = PoseStamped()
                    goal_msg.header.stamp = rospy.Time.now()
                    goal_msg.header.frame_id = "map"  # 或者是你 SLAM 的全局坐标系
                    goal_msg.pose.position.x = world_goal_x
                    goal_msg.pose.position.y = world_goal_y
                    goal_msg.pose.position.z = 0.0
                    goal_msg.pose.orientation.z = 1.0
                    self.goal_pub.publish(goal_msg)
                    print(
                        f" [Ros-Goal] 发送导航点: 像素({goal_x_idx}, {goal_y_idx}) -> 物理({world_goal_x:.2f}, {world_goal_y:.2f})")
            time2 = time.time()
            print(f" [耗时]路径点规划耗时: {time2 - time1}")  # 输出示例：1773070000.123456
            time1 = time2

            # 6.3 用墙来修正预测
            step_ego_grid_27 = batch['step_ego_grid_27']
            for t in range(T):
                pred_maps_objects_bottom = step_ego_grid_27[0, t, :, :, :]  # [27,64,64]
                pred_maps_objects_top = pred_maps_objects[0, t, :, :, :]  # [27,64,64]
                # 计算 bottom 层每个栅格的 argmax
                val_bottom, idx_bottom = torch.max(pred_maps_objects_bottom, dim=0)  # idx_bottom shape: [64, 64]
                # 2. 构造掩码 (Mask)：找出 argmax 为 0 的位置
                fail_mask = (idx_bottom != 15)  # 只保留贝叶斯更新中的墙
                # 3. 初始化最终的融合地图
                # 我们先完整复制 bottom 的数据
                fused_map = pred_maps_objects_bottom.clone()
                # 4. 执行叠加逻辑：
                # 在所有 fail_mask 为 True 的坐标点，用 top 的 27 维概率向量替换掉 bottom 的
                # 这里使用广播机制处理 27 个通道
                fused_map[:, fail_mask] = pred_maps_objects_top[:, fail_mask]
                pred_maps_objects[0, t, :, :, :] = fused_map
            # viz_utils.show_image_color_and_extract(pred_maps_objects, "Predicted Map", 27)

            # 更新全局地图 (调用你原本的方法)
            step_geo_grid = self.sg.register_sem_pred_ros_without_rot(prediction_crop=pred_maps_objects, pose=_rel_pose)

            flag_rviz_2dmap = False
            # 在 run_online_inference 内部
            if flag_rviz_2dmap:
                # 计算cartographer地图的显示区域
                carto_origin_x = self.FrontierExtractor._map_origin_x  # Cartographer地图原点x
                carto_origin_y = self.FrontierExtractor._map_origin_y  # Cartographer地图原点y
                carto_res = self.FrontierExtractor._map_resolution  # Cartographer地图分辨率（米/像素）
                carto_pixel_w = self.FrontierExtractor._map_width  # Cartographer地图像素宽度
                carto_pixel_h = self.FrontierExtractor._map_height  # Cartographer地图像素高度
                # 计算Cartographer地图的物理范围（世界坐标）
                carto_min_x = carto_origin_x  # 最小x（原点x）
                carto_max_x = carto_origin_x + carto_pixel_w * carto_res  # 最大x（原点x + 宽度*分辨率）
                carto_min_y = carto_origin_y  # 最小y（原点y）
                carto_max_y = carto_origin_y + carto_pixel_h * carto_res  # 最大y（原点y + 高度*分辨率）
                max_area = [carto_min_x, carto_max_x, carto_min_y, carto_max_y]
                self.semantic_map_publisher.async_publish(
                    step_geo_grid.squeeze(0),  # 去掉批次维度，变成 [1, 27, 200, 200]
                    # free_mask= self.FrontierExtractor.target_free_mask,
                    None,
                    res=0.1,
                    origin_x=self.sg.origin[0],
                    origin_y=self.sg.origin[1],
                    height=self.semantic_map_publisher_height
                    # ,
                    # max_area = max_area
                )

            time2 = time.time()
            print(f" [耗时]全局语义地图更新耗时: {time2 - time1}")  # 输出示例：1773070000.123456
            time1 = time2

            # 保存地图

            if self.options.save_nav_images:
                # save_img_dir_ = self.options.save_img_dir + '/ep_' + str(tstep)  + '/'
                save_img_dir_ = f"{self.options.save_img_dir}/ros/7_floor"
                print("     [zhjd-ros] save_img_dir_: ", save_img_dir_)
                if not os.path.exists(save_img_dir_):
                    os.makedirs(save_img_dir_)
                # viz_utils.save_all_infos_and_mapprediction_ros(batch, pred_maps_objects, savepath=save_img_dir_, name=f'path_{self.num_flag:03d}')
                # viz_utils.save_Global_forROS(step_geo_grid, step_uncertainty, savepath=save_img_dir_, name=f"global_{self.num_flag:03d}")
                # viz_utils.save_free_mask
                # viz_utils.save_uncertainty_ros(step_geo_grid, step_uncertainty, pose_coords_list.clone().cpu().numpy(), save_img_dir_, global_time=self.num_flag)
                if self.num_flag % 30 == 0:
                    viz_utils.save_only_Global_forROS(step_geo_grid, savepath=save_img_dir_, name=f"global_seq_{last_seq:06d}")


            time2 = time.time()
            print(f" [耗时]全局: {time2 - time0}")  # 输出示例：1773070000.123456
            curr_time = time.time()
            dt = curr_time - self.prev_time
            self.fps = 1.0 / dt if dt > 0 else 0.0
            self.prev_time = curr_time
            print(f" ------------ FPS: 【{self.fps:.2f}】 \n")
            self.num_flag = self.num_flag + 1

    def get_cost_map(self, sg, sem_lbl, a_1, a_2):
        p_map = sg.sem_grid[:, sem_lbl, :, :]
        sigma_map = torch.sqrt(sg.per_class_uncertainty_map[:, sem_lbl, :, :])
        return p_map + torch.sign(a_2 - p_map) * a_1 * sigma_map

    def get_Epistemic_Uncertainty_cost_map(self, sg):
        # per_class_uncertainty_map [batch, 27, H, W]
        max_uncertainty, _ = torch.max(sg.per_class_uncertainty_map, dim=1, keepdim=False) # [B, H, W]
        # max_uncertainty, _ = torch.max(sg.per_class_uncertainty_map, dim=1, keepdim=True) # [B, 1, H, W]
        # sg.per_class_uncertainty_map[:, 0, :, :]  [B, H, W]
        sigma_map = torch.sqrt(max_uncertainty)
        return sigma_map


    # def get_long_term_goal(self, cost_map):
    #     # 1. 预处理 cost_map，确保是 [H, W] 的 numpy 数组
    #     # 假设 cost_map 维度是 [B, 1, H, W]
    #     map_np = cost_map.detach().cpu().numpy()[0, :, :]
    #
    #     # 2. 获取已经对齐好的 free_mask [H, W]
    #     # 确保它是布尔类型
    #     free_mask = self.FrontierExtractor.target_free_mask
    #     if free_mask is None:
    #         rospy.logwarn("Free mask is None, returning zero goal")
    #         return torch.zeros(2, dtype=torch.int64, device=self.device)
    #
    #     # 3. 使用 mask 过滤掉非空闲区域
    #     # 将非空闲区域的分数设为极小值（比如 -1e9），这样 argmax 永远不会选到它们
    #     masked_map = np.where(free_mask, map_np, -1e9)
    #
    #     # 4. 找到最大值的索引 (一维索引)
    #     idx_1d = np.argmax(masked_map)
    #     final_score = masked_map.flat[idx_1d]
    #
    #     # 5. 转换为二维像素坐标 [row_idx, col_idx]
    #     # 注意：NumPy 的索引顺序通常是 (y, x) 对应 (row, col)
    #     y_idx, x_idx = np.unravel_index(idx_1d, masked_map.shape)
    #
    #     # --- 输出调试信息 ---
    #     if final_score < -1e8:
    #         rospy.logwarn("警告：所有区域都被过滤，未找到有效的 Free 区域目标点！")
    #     else:
    #         print(f"目标点选择成功: 像素坐标({x_idx}, {y_idx}), 最终得分: {final_score:.4f}")
    #
    #     # 6. 返回目标点 [x_pixel, y_pixel]
    #     # 这里的顺序要和你之前的 goal = [cell_j, cell_j] (推测你想写 [x, y]) 保持一致
    #     if final_score > 0.63:
    #         return torch.tensor([x_idx, y_idx], dtype=torch.int64, device=self.device)
    #         # 否则，进入边界点 (Frontier) 评价模式
    #     else:
    #         # rospy.loginfo("--- [Mode: Frontier] 语义得分过低，转向评价边界点 ---")
    #         frontiers = self.FrontierExtractor.get_latest_frontier_centroids()
    #
    #         if not frontiers:
    #             rospy.logwarn("未发现任何边界点，返回None")
    #             return None
    #
    #         best_frontier_score = -1e9
    #         best_frontier_pixel = [x_idx, y_idx]  # 默认兜底
    #
    #         for f_world in frontiers:
    #             # 将 World 坐标 (f_world[0], f_world[1]) 转换为 Target Map 的像素坐标
    #             # 转换公式：(World - Origin) / Cell_Size
    #             f_x_px = int(
    #                 (f_world[0] - self.FrontierExtractor.target_origin_x) / self.FrontierExtractor.target_cell_size)
    #             f_y_px = int(
    #                 (f_world[1] - self.FrontierExtractor.target_origin_y) / self.FrontierExtractor.target_cell_size)
    #
    #             # 边界检查：确保点在 300x300 (或者你设定的 grid_dim) 范围内
    #             if 0 <= f_x_px < self.grid_dim[0] and 0 <= f_y_px < self.grid_dim[1]:
    #                 # 从 cost_map 中提取该边界点位置的分数
    #                 f_score = map_np[f_y_px, f_x_px]
    #
    #                 if f_score > best_frontier_score:
    #                     best_frontier_score = f_score
    #                     best_frontier_pixel = [f_x_px, f_y_px]
    #             else:
    #                 # 如果边界点在 target_map 范围外，通常跳过
    #                 continue
    #
    #         print(
    #             f"--- [Mode: Frontier] 已选定最佳边界点: 像素({best_frontier_pixel[0]}, {best_frontier_pixel[1]}), 得分: {best_frontier_score:.4f} ---")
    #         return torch.tensor(best_frontier_pixel, dtype=torch.int64, device=self.device)


    # （1）将自由区间内score大于6.3的点，选出来，从中选择一个最近的点
    # （2）如果没有大于6.3的点，则直接选择一个距离最近的frontier（即不考虑frontier的socre）
    def get_long_term_goal(self, cost_map, robot_pose_px):
        """
        robot_pose_px: [x, y] 机器人当前的像素坐标
        """
        # 1. 预处理 cost_map [H, W]
        map_np = cost_map.detach().cpu().numpy()[0, :, :]

        # 2. 获取 Free Mask
        free_mask = self.FrontierExtractor.target_free_mask
        if free_mask is None:
            rospy.logwarn("Free mask is None, returning None")
            return None

        # --- 核心修复：检查并对齐维度 ---
        # 如果 map_np 是 (120, 600) 而 free_mask 是 (600, 120)
        if map_np.shape != free_mask.shape:
            # print(f"DEBUG: 对齐维度 {map_np.shape} vs {free_mask.shape}")
            # 通常我们需要将 free_mask 转置以匹配 cost_map 的 [H, W] 顺序
            free_mask = free_mask.T

        # --- 策略 (1): 寻找自由区间内 Score > 0.63 的点中最近的一个 ---
        # 找到所有满足条件的像素索引 (y_indices, x_indices)
        high_score_mask = (map_np > 0.63) & (free_mask > 0)
        y_high, x_high = np.where(high_score_mask)

        if len(x_high) > 0:
            # 计算这些高分点到机器人的距离
            dists = np.sqrt((x_high - robot_pose_px[0]) ** 2 + (y_high - robot_pose_px[1]) ** 2)
            nearest_idx = np.argmin(dists)

            target_px = [x_high[nearest_idx], y_high[nearest_idx]]
            print(
                f"--- [Mode: 认知不确定性 High Score] 发现 {len(x_high)} 个目标，选择最近点: {target_px}, 距离: {dists[nearest_idx]:.2f} ---")
            return torch.tensor(target_px, dtype=torch.int64, device=self.device)

        # --- 策略 (2): 如果没有高分点，则选择距离最近的 Frontier ---
        else:
            rospy.loginfo("--- [Mode: 最近边界点] 无高分目标，寻找最近边界点 ---")
            frontiers = self.FrontierExtractor.get_latest_frontier_centroids()

            if not frontiers:
                rospy.logwarn("未发现任何边界点，探索失败")
                return None

            best_f_px = None
            min_f_dist = float('inf')

            for f_world in frontiers:
                # World -> Pixel
                f_x_px = int(
                    (f_world[0] - self.FrontierExtractor.target_origin_x) / self.FrontierExtractor.target_cell_size)
                f_y_px = int(
                    (f_world[1] - self.FrontierExtractor.target_origin_y) / self.FrontierExtractor.target_cell_size)

                # 边界检查
                if 0 <= f_x_px < self.grid_dim[0] and 0 <= f_y_px < self.grid_dim[1]:
                    # 计算到机器人的像素距离
                    dist = np.sqrt((f_x_px - robot_pose_px[0]) ** 2 + (f_y_px - robot_pose_px[1]) ** 2)
                    if dist < min_f_dist:
                        min_f_dist = dist
                        best_f_px = [f_x_px, f_y_px]

            if best_f_px is not None:
                print(f"--- [Mode: Nearest Frontier] 已选定最近边界点: {best_f_px}, 距离: {min_f_dist:.2f} ---")
                return torch.tensor(best_f_px, dtype=torch.int64, device=self.device)
            else:
                return None

