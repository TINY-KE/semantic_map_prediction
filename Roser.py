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
from geometry_msgs.msg import PoseStamped, Point
from StepEgoMapPose_msgs.msg import StepEgoMapPose
from geometry_msgs.msg import PoseStamped # 确保在文件顶部导入
from visualization_msgs.msg import Marker

from ros_utils.FrontierExtractor import FrontierExtractor
from ros_utils.SemanticMapPublisher import  SemanticMarkerPublisher, AsyncDualSemanticPublisher, AsyncUncertaintyMarkerPublisher

class RosTester(object):
    """ Implements testing for prediction models
    """
    def __init__(self, options):
        self.options = options
        print("options:")
        for k in self.options.__dict__.keys():
            print(k, self.options.__dict__[k])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        flag_load_debug = True
        if flag_load_debug:
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
        # self.sg = SemanticGrid(self.batch_size, self.grid_dim, self.crop_size[0], self.cell_size,
        #                   spatial_labels=self.spatial_labels, object_labels=self.object_labels, origin=None,
        #                   recovered_map_path =  "/home/robotlab/Downloads/global_map.png")
        self.sg = SemanticGrid(self.batch_size, self.grid_dim, self.crop_size[0], self.cell_size,
                               spatial_labels=self.spatial_labels, object_labels=self.object_labels, origin=None,
                               recovered_map_path="/home/robotlab/Downloads/ruihai_global_seq_182185.png")

        # 3.边界和地图处理
        # rospy.loginfo("Models loaded, stabilizing system...")
        # rospy.sleep(2.0)  # 让出 CPU 时间片，让 ROS 处理队列中的地图包
        self.FrontierExtractor = FrontierExtractor(self.sg)
        self.semantic_map_publisher = AsyncDualSemanticPublisher(raw_topic="/semantic_global_map", filtered_topic="/semantic_global_map_free_only")
        self.semantic_map_publisher_height = 6
        self.uncertainty_map_publisher = AsyncUncertaintyMarkerPublisher(raw_topic="/uncertainty_global_map", filtered_topic="/uncertainty_global_map_free_only")
        self.uncertainty_map_publisher_height = -6
        # 初始化历史边界坐标 [min_x, max_x, min_y, max_y]
        # 使用 float('inf') 确保第一次更新能成功
        self.history_min_x = float('inf')
        self.history_max_x = float('-inf')
        self.history_min_y = float('inf')
        self.history_max_y = float('-inf')

        # 5. ROS 订阅和发布
        rospy.Subscriber("/step_ego_map_pose", StepEgoMapPose, self.ros_callback)
        # self.goal_pub = rospy.Publisher('/move_base_simple/goal_sigma', PoseStamped, queue_size=1)
        self.potential_points_pub = rospy.Publisher('/potential_exploration_points', Marker, queue_size=1)
        self.ltg_marker_pub = rospy.Publisher('/long_term_goal_marker', Marker, queue_size=1)
        self.ltg_marker_pub_2 = rospy.Publisher('/long_term_goal_marker_2', Marker, queue_size=1)

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

        # 坐标
        robot_pose_list = [pose.data for pose in msg.robot_pose]  # 假设格式是 [x, y, theta]
        curr_x = robot_pose_list[0]
        curr_y = robot_pose_list[1]
        # 更新历史最值
        if curr_x < self.history_min_x: self.history_min_x = curr_x
        if curr_x > self.history_max_x: self.history_max_x = curr_x
        if curr_y < self.history_min_y: self.history_min_y = curr_y
        if curr_y > self.history_max_y: self.history_max_y = curr_y

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


            # # 选择长期目标
            eucost_map = self.get_Epistemic_Uncertainty_cost_map(self.sg)
            robot_pose_world = pose_seq[0].cpu().numpy()
            robot_x_px = int((robot_pose_world[0] - self.FrontierExtractor.target_origin_x) / self.FrontierExtractor.target_cell_size)
            robot_y_px = int((robot_pose_world[1] - self.FrontierExtractor.target_origin_y) / self.FrontierExtractor.target_cell_size)
            robot_pose_px = [robot_x_px, robot_y_px]
            ltg = self.get_long_term_goal(eucost_map, robot_pose_px)
            if ltg is not None:
                self.publish_ltg_marker(ltg.cpu().numpy())
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

            flag_rviz_2dmap = True
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
                if last_seq > 500:
                    min_compined = carto_max_x
                else:
                    min_compined = min(carto_max_x, self.history_max_x + 3.2)
                max_area = [carto_min_x, min_compined, carto_min_y, carto_max_y]
                self.semantic_map_publisher.async_publish(
                    step_geo_grid.squeeze(0),  # 去掉批次维度，变成 [1, 27, 200, 200]
                    free_mask= self.FrontierExtractor.target_free_mask,
                    # None,
                    res=0.1,
                    origin_x=self.sg.origin[0],
                    origin_y=self.sg.origin[1],
                    height=self.semantic_map_publisher_height
                    ,
                    max_area = max_area
                )
                self.uncertainty_map_publisher.async_publish(
                    step_uncertainty.squeeze(0),  # 去掉批次维度，变成 [1, 27, 200, 200]
                    free_mask=self.FrontierExtractor.target_free_mask,
                    # None,
                    res=0.1,
                    origin_x=self.sg.origin[0],
                    origin_y=self.sg.origin[1],
                    height=self.uncertainty_map_publisher_height,
                    max_area=max_area,
                    threshold=0,
                    robot_pose_px = robot_pose_px
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

    def publish_potential_points_marker(self, pts_px):
        """
        pts_px: 像素坐标列表 [[x1, y1], [x2, y2], ...]
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "potential_points"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD

        # 尺寸：直径 0.35 米的球体，稍微大一点点更显眼
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        # 颜色：粉色 (R: 1.0, G: 0.0, B: 1.0)
        marker.color.r = 1.0
        marker.color.g = 0.08  # 稍微带一点点暖色调的深粉
        marker.color.b = 0.58
        marker.color.a = 1  # 提高不透明度，让它更跳脱

        for pt in pts_px:
            # 像素坐标 -> 物理世界坐标
            world_x = pt[0] * self.cell_size + self.sg.origin[0]
            world_y = pt[1] * self.cell_size + self.sg.origin[1]

            p = Point()
            p.x = world_x
            p.y = world_y
            p.z = self.uncertainty_map_publisher_height + 0.12
            marker.points.append(p)

        self.potential_points_pub.publish(marker)

    def publish_ltg_marker(self, goal_px):
        """
        以亮绿色圆柱体形式发布最终选定的导航点 (仅发布一个高度)
        """
        if goal_px is None:
            return

        # 1. 物理坐标转换 (确保处理的是 Numpy 数组)
        if isinstance(goal_px, torch.Tensor):
            goal_px = goal_px.detach().cpu().numpy()

        world_x = goal_px[0] * self.cell_size + self.sg.origin[0]
        world_y = goal_px[1] * self.cell_size + self.sg.origin[1]

        # 2. 创建 Marker
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "final_goal"
        marker.id = 100  # 固定 ID 确保新目标会覆盖旧目标
        marker.type = Marker.CYLINDER  # <--- 修改为圆柱体
        marker.action = Marker.ADD

        # 尺寸：直径 0.5m (x, y)，高度 0.2m (z)
        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 0.4

        # 颜色：亮绿色
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # 不透明

        # 位置
        marker.pose.position.x = world_x
        marker.pose.position.y = world_y
        # 设置在地面层高度 (0.15m)
        marker.pose.position.z = 0.15
        marker.pose.orientation.w = 1.0

        # (可选) 设置为 0 表示 Marker 永远不会自动消失，直到被新 ID 覆盖
        marker.lifetime = rospy.Duration(0)

        self.ltg_marker_pub.publish(marker)

        marker.pose.position.z = 0.15 + self.uncertainty_map_publisher_height
        self.ltg_marker_pub_2.publish(marker)


    # （1）将自由区间内score大于0.6的点，选出来，从中选择一个最近的点
    # （2）如果没有大于6.3的点，则直接选择一个距离最近的frontier（即不考虑frontier的socre）
    def get_long_term_goal(self, cost_map, robot_pose_px):
        """
        robot_pose_px: [x, y] 机器人当前的像素坐标 (Grid Coordinates)
        cost_map: 认知不确定性地图 (Sigma Map)
        """
        # 1. 预处理 cost_map 转换为 [H, W] 的 numpy 数组
        map_np = cost_map.detach().cpu().numpy()
        if map_np.ndim == 3:
            map_np = map_np[0, :, :]
        elif map_np.ndim == 4:
            map_np = map_np[0, 0, :, :]

        # 2. 获取并检查 Free Mask
        free_mask = self.FrontierExtractor.target_free_mask
        if free_mask is None:
            rospy.logwarn("--- [LTG] Free mask is None, skipping goal selection ---")
            return None

        # 维度对齐检查
        if map_np.shape != free_mask.shape:
            if map_np.shape == free_mask.T.shape:
                free_mask = free_mask.T
            else:
                rospy.logerr(f"--- [LTG] Shape mismatch: Map {map_np.shape}, Mask {free_mask.shape} ---")
                return None

        # ==========================================================
        # 新增：距离过滤逻辑 (只在机器人附近选择)
        # ==========================================================
        H, W = map_np.shape
        # 定义搜索半径 (例如：10米 = 10 / 0.1 = 100 像素)
        search_radius_m = 5.0
        search_radius_px = int(search_radius_m / self.cell_size)

        # 创建网格索引
        Y, X = np.ogrid[:H, :W]
        # 计算每个点到机器人像素位置的平方距离 (避免开方以提高速度)
        dist_sq = (X - robot_pose_px[0]) ** 2 + (Y - robot_pose_px[1]) ** 2
        # 生成圆形距离掩码
        distance_mask = dist_sq <= search_radius_px ** 2

        # ==========================================================
        # 策略 (1): 可视化逻辑 - 寻找附近的 Top 4 潜力点
        # ==========================================================
        # 复合过滤：必须在 Free 区域 且 在机器人附近
        nearby_free_mask = (free_mask > 0) & distance_mask
        y_free, x_free = np.where(nearby_free_mask)

        if len(x_free) > 0:
            free_scores = map_np[y_free, x_free]
            sorted_indices = np.argsort(free_scores)[::-1]

            potential_pts = []
            for idx in sorted_indices:
                pt = [x_free[idx], y_free[idx]]
                if all(np.linalg.norm(np.array(pt) - np.array(prev_pt)) > 10 for prev_pt in potential_pts):
                    potential_pts.append(pt)
                if len(potential_pts) >= 4:
                    break
            self.publish_potential_points_marker(potential_pts)

        # ==========================================================
        # 策略 (2): 导航决策逻辑 - 附近的高分点
        # ==========================================================
        # 复合过滤：高分 且 Free 且 附近
        nearby_high_score_mask = (map_np > 0.6) & (free_mask > 0) & distance_mask
        y_high, x_high = np.where(nearby_high_score_mask)

        if len(x_high) > 0:
            # 在附近的候选点中选最近的
            dists = np.sqrt((x_high - robot_pose_px[0]) ** 2 + (y_high - robot_pose_px[1]) ** 2)
            nearest_idx = np.argmin(dists)

            target_px = [x_high[nearest_idx], y_high[nearest_idx]]
            print(f"--- [Mode: 认知不确定性] 附近(半径{search_radius_m}m)发现目标: {target_px} ---")
            return torch.tensor(target_px, dtype=torch.int64, device=self.device)

        # ==========================================================
        # 策略 (3): 兜底逻辑 - 附近没有高分，选最近边界点 (Frontier)
        # ==========================================================
        else:
            rospy.loginfo("--- [Mode: 边界探索] 半径内无高分，寻找边界点 ---")
            frontiers = self.FrontierExtractor.get_latest_frontier_centroids()

            if not frontiers:
                return None

            best_f_px = None
            min_f_dist = float('inf')

            for f_world in frontiers:
                f_x_px = int(
                    (f_world[0] - self.FrontierExtractor.target_origin_x) / self.FrontierExtractor.target_cell_size)
                f_y_px = int(
                    (f_world[1] - self.FrontierExtractor.target_origin_y) / self.FrontierExtractor.target_cell_size)

                if 0 <= f_x_px < W and 0 <= f_y_px < H:
                    dist = np.sqrt((f_x_px - robot_pose_px[0]) ** 2 + (f_y_px - robot_pose_px[1]) ** 2)
                    # 你可以选择是否也对 Frontier 进行半径限制
                    # if dist > search_radius_px: continue

                    if dist < min_f_dist:
                        min_f_dist = dist
                        best_f_px = [f_x_px, f_y_px]

            if best_f_px is not None:
                return torch.tensor(best_f_px, dtype=torch.int64, device=self.device)
            else:
                return None


    def get_long_term_goal2(self, cost_map, robot_pose_px):
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

            scores = map_np[y_high, x_high]
            sorted_indices = np.argsort(scores)[::-1]

            potential_pts = []
            for idx in sorted_indices:
                pt = [x_high[idx], y_high[idx]]
                # 距离去重，防止 4 个粉点叠在一起
                if all(np.linalg.norm(np.array(pt) - np.array(prev_pt)) > 8 for prev_pt in potential_pts):
                    potential_pts.append(pt)
                if len(potential_pts) >= 4:
                    break

            # 发布粉色标记点
            self.publish_potential_points_marker(potential_pts)

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

