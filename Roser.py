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

class RosTester(object):
    """ Implements testing for prediction models
    """
    def __init__(self, options):
        self.options = options
        print("options:")
        for k in self.options.__dict__.keys():
            print(k, self.options.__dict__[k])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. 完整加载 Ensemble 模型逻辑
        self.models_dict = {} # keys are the ids of the models in the ensemble
        ensemble_exp_rsmp = os.listdir(self.options.ensemble_dir_rsmp) # ensemble_dir should be a dir that holds multiple experiments
        ensemble_exp_rsmp.sort() # in case the models are numbered put them in order
        for n in range(self.options.ensemble_size):

            print("     [zhjd-slam-search] RosTester Init Loading model ", n)
            self.models_dict[n] = {'predictor_model': get_predictor_rsmp(self.options)}
            self.models_dict[n] = {k:v.to(self.device) for k,v in self.models_dict[n].items()}

            # Needed only for models trained with multi-gpu setting
            self.models_dict[n]['predictor_model'] = nn.DataParallel(self.models_dict[n]['predictor_model'])

            checkpoint_dir = self.options.ensemble_dir_rsmp + "/" + ensemble_exp_rsmp[n]
            print('checkpoint_dir', checkpoint_dir)

            latest_checkpoint = tutils.get_latest_model(save_dir=checkpoint_dir)
            print("Model", n, "loading checkpoint", latest_checkpoint)
            self.models_dict[n] = tutils.load_model(models=self.models_dict[n], checkpoint_file=latest_checkpoint)
            self.models_dict[n]["predictor_model"].eval()

        # 2. 状态缓冲区：T=10
        self.batch_size = 1
        self.grid_buffer = deque(maxlen=self.batch_size)
        self.pose_buffer = deque(maxlen=self.batch_size)

        # 4. 初始化全局语义地图
        self.spatial_labels = 3
        self.object_labels = 27
        self.grid_dim = (150, 150)  # 749办公室 7*12米
        self.cell_size = 0.1
        self.crop_size = (64, 64)
        self.sg = SemanticGrid(self.batch_size, self.grid_dim, self.crop_size[0], self.cell_size,
                          spatial_labels=self.spatial_labels, object_labels=self.object_labels)


        # 5. ROS 订阅和发布
        rospy.Subscriber("/step_ego_map_pose", StepEgoMapPose, self.ros_callback)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)

        # grid_sub = message_filters.Subscriber("/ego_grid", OccupancyGrid)
        # pose_sub = message_filters.Subscriber("/robot_pose", PoseStamped)
        # # 同步器：slop 是允许的时间误差，单位秒
        # ts = message_filters.ApproximateTimeSynchronizer([grid_sub, pose_sub], queue_size=10, slop=0.05)
        # ts.registerCallback(self.ros_sync_callback)

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

        if len(self.grid_buffer) == self.batch_size:
            self.run_online_inference()

    def run_online_inference(self):
        print(f"\n ------------  {self.num_flag} 开始推理 -------------")
        # 构造 Batch: [B=1, T=1, C=27, H=64, W=64]
        grid_seq = torch.stack(list(self.grid_buffer))  # [batch_size, 27, 64, 64]
        pose_seq = torch.stack(list(self.pose_buffer))  # [batch_size, 3]
        print(pose_seq)
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
            print(f"                      推理耗时: {time2 - time1}")  # 输出示例：1773070000.123456
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
            # Use cost map to decide next long term direction
            ltg = self.get_long_term_goal(self.sg, eucost_map)
            print(f" 发布探索点: {ltg[0,0,0]}, {ltg[0,0,1]}")
            flag_pub_ltg = True
            if flag_pub_ltg:
                goal_x_idx = ltg[0, 0, 0].item()
                goal_y_idx = ltg[0, 0, 1].item()
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
                goal_msg.pose.orientation.w = 1.0
                self.goal_pub.publish(goal_msg)
                print(
                    f" [Ros-Goal] 发送导航点: 像素({goal_x_idx}, {goal_y_idx}) -> 物理({world_goal_x:.2f}, {world_goal_y:.2f})")
            time2 = time.time()
            print(f" [耗时]路径点规划耗时: {time2 - time1}")  # 输出示例：1773070000.123456
            time1 = time2

            # 更新全局地图 (调用你原本的方法)
            step_geo_grid = self.sg.register_sem_pred_ros_without_rot(prediction_crop=pred_maps_objects, pose=_rel_pose)
            time2 = time.time()
            print(f" [耗时]全局语义地图更新耗时: {time2 - time1}")  # 输出示例：1773070000.123456
            time1 = time2

            # viz_utils.show_image_color_and_extract(pred_maps_objects, "Predicted Map", 27)

            if self.options.save_nav_images:
                # save_img_dir_ = self.options.save_img_dir + '/ep_' + str(tstep)  + '/'
                save_img_dir_ = f"{self.options.save_img_dir}/ros/1/"
                print("     [zhjd-ros] save_img_dir_: ", save_img_dir_)
                if not os.path.exists(save_img_dir_):
                    os.makedirs(save_img_dir_)
                # viz_utils.save_all_infos_and_mapprediction_slam(batch, pred_maps_objects, savepath=save_img_dir_, name='path')
                viz_utils.save_Global_forROS(step_geo_grid, step_uncertainty, savepath=save_img_dir_, name=f"global_{self.num_flag:03d}")
                # viz_utils.save_uncertainty_ros(step_geo_grid, step_uncertainty, pose_coords_list.clone().cpu().numpy(), save_img_dir_, global_time=self.num_flag)

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

    def get_long_term_goal(self, sg, cost_map):
        ### Choose long term goal
        goal = torch.zeros((sg.per_class_uncertainty_map.shape[0], 1, 2), dtype=torch.int64, device=self.device)
        # explored_grid = map_utils.get_explored_grid(sg.proj_grid)
        # current_UNexplored_map = 1-explored_grid
        # unexplored_cost_map = cost_map * current_UNexplored_map
        # unexplored_cost_map = unexplored_cost_map.squeeze(1)
        unexplored_cost_map = cost_map.squeeze(1)
        for b in range(unexplored_cost_map.shape[0]):
            map_ = unexplored_cost_map[b,:,:]
            inds = utils.unravel_index(map_.argmax(), map_.shape)
            goal[b,0,0] = inds[1]
            goal[b,0,1] = inds[0]
        return goal