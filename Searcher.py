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


class SearchTester(object):
    """ Implements testing for prediction models
    """
    def __init__(self, options):
        self.options = options
        print("options:")
        for k in self.options.__dict__.keys():
            print(k, self.options.__dict__[k])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.test_ds = HabitatDataOfflineMPv2(options, config_file=options.config_test_file)

        ensemble_exp_rsmp = os.listdir(self.options.ensemble_dir_rsmp) # ensemble_dir should be a dir that holds multiple experiments
        ensemble_exp_rsmp.sort() # in case the models are numbered put them in order
        N = len(ensemble_exp_rsmp) # number of models in the ensemble
        self.models_dict = {} # keys are the ids of the models in the ensemble
        for n in range(self.options.ensemble_size):
            # self.models_dict[n] = {'predictor_model': get_predictor_from_options(self.options)}

            print("     [zhjd-debug-search] SearchTester Init predictor_model ...")
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

        self.spatial_classes = {0:"void", 1:"occupied", 2:"free"}
        self.object_classes = {0:"void", 17:"floor", 15:'wall', 3:"table", 4:"cushion", 13:"counter", 1:"chair", 5:"sofa", 6:"bed"}

        # initialize res dicts
        self.results_spatial = {}
        self.results_objects = {}
        for object_ in list(self.object_classes.values()):
            self.results_objects[object_] = {}
        self.results_objects['objects_all'] = {}
        self.results_objects['original_result'] = {}

        for spatial in list(self.spatial_classes.values()):
            self.results_spatial[spatial] = {}
        self.results_spatial['spatial_all'] = {}

    def test_semantic_map(self):
        # 1. 数据加载器
        print("     [zhjd-debug-search], params test_batch_size: ", self.options.test_batch_size, ", num_workers:", self.options.num_workers)
        test_data_loader = DataLoader(self.test_ds,
                                      # 这是数据集（Dataset）对象，表示测试数据集。它通常是一个继承自 torch.utils.data.Dataset 类的自定义类，用来定义如何加载和预处理数据。
                                      # self.test_ds 中包含了测试数据的所有样本和标签，并实现了如何在 __getitem__ 方法中获取一个样本。
                                batch_size=self.options.test_batch_size,
                                num_workers=self.options.num_workers,
                                pin_memory=self.options.pin_memory,
                                shuffle=False)
        # 2. 初始化变量
        batch = None
        self.options.test_iters = len(test_data_loader) # the length of dataloader depends on the batch size
        print("   [zhjd-debug-search] self.options.test_iters : ", self.options.test_iters )
        object_labels = list(range(self.options.n_object_classes))
        spatial_labels = list(range(self.options.n_spatial_classes))
        overall_confusion_matrix_objects, overall_confusion_matrix_spatial = None, None

        # 3. 遍历测试数据集
        # tstep：当前批次的索引（从 0 开始），通常用于跟踪训练或测试的进度。
        # batch：包含当前批次数据的字典或张量，通常包括输入数据（如图像）和目标标签（如标签图像或类标签）。在循环中，batch 会被用于模型推理或计算损失。
        for tstep, batch in enumerate(tqdm(test_data_loader,   # tqdm 会显示当前进度、每秒处理的批次数以及估计的剩余时间。
                                           desc='Testing',
                                           total=self.options.test_iters)):
            # ep_name = batch['name']
            # print("   [zhjd-debug] batch: ", batch.size())

            # 用于把一个 batch（批次数据）移动到 GPU，同时跳过不需要放上 GPU 的字段。
            # batch = {k: v.to(self.device) for k, v in batch.items() if k != 'name'}
            scene_id = batch['scene_id'][0]  # 比如 ['TbHJrupSAjP']
            epsoid_name = batch['epsoid_name'][0]  # 比如 ['ep_43_72_TbHJrupSAjP']
            skip_keys = {'name', 'epsoid_name', 'scene_id'}
            batch = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
                if k not in skip_keys
            }


            # 4. 模型推理（Ensemble 集成模型）
            with torch.no_grad():
                # # 获取 ground truth
                gt_crops_objects = batch['gt_grid_crops_objects'].cpu()  # B x T x 1 x cH x cW

                # # 初始化 ensemble 输出列表
                ensemble_object_maps, ensemble_spatial_maps = [], []
                N = len(self.models_dict) # numbers of models in the ensemble
                print("   [zhjd-debug-search] range(self.options.ensemble_size): ", self.options.ensemble_size)
                for n in range(self.options.ensemble_size):
                    pred_output = self.models_dict[n]['predictor_model'](batch)
                    ensemble_object_maps.append(pred_output['pred_maps_objects'].clone())

                # 5. 集成模型平均预测
                ensemble_object_maps = torch.stack(ensemble_object_maps)  # N x B x T x C x cH x cW
                print("   [zhjd-debug-search] ensemble_object_maps.shape: ", ensemble_object_maps.shape)  # torch.Size([1, 1, 10, 27, 64, 64])

                # TODO: 各模型预测的平均语义图
                # Getting the mean predictions from the ensemble
                pred_maps_objects = torch.mean(ensemble_object_maps, dim=0)  # B x T x C x cH x cW
                print("   [zhjd-debug-search] pred_maps_objects.shape: ", pred_maps_objects.shape)

                # argmax，只用于可视化
                # Decide label for each location based on predition probs
                pred_labels_objects = torch.argmax(pred_maps_objects.cpu(), dim=2, keepdim=True) # B x T x 1 x cH x cW


                # 6. todo: 将预测逐步融合进全局地图
                # 初始化全局语义地图
                sg = SemanticGrid(1, self.test_ds.grid_dim, self.test_ds.crop_size[0], self.test_ds.cell_size,
                                  spatial_labels=self.test_ds.spatial_labels, object_labels=self.test_ds.object_labels)
                # 6.1 计算位姿
                abs_poses = []  # 每步的绝对位姿，用于坐标变换与地图配准
                # agent_height = []
                # print("   [zhjd-debug-search] batch['abs_pose'].shape: ", batch['abs_pose'].shape)
                abs_poses = batch['abs_pose']
                _abs_poses = torch.tensor(abs_poses, dtype=torch.float32, device=batch['abs_pose'].device).squeeze(0)
                _abs_poses = _abs_poses.to(self.device)
                # print("   [zhjd-debug-search] _abs_poses.shape: ", _abs_poses.shape)
                # agent_height.append(0.09859601)  # 0.09859601
                # print("   [zhjd-debug-search] batch['rel_pose'].shape: ", batch['rel_pose'].shape)
                rel = batch['rel_pose']
                _rel_pose = torch.tensor(rel, dtype=torch.float32, device=batch['rel_pose'].device).squeeze(0)
                _rel_pose = _rel_pose.to(self.device)
                # pose_coords_list：把位姿投到地图格子坐标（便于在地图上定位 agent）。
                # print("_abs_poses: ", _abs_poses)
                # _abs_poses:  tensor([[ 3.1302, -4.6391,  2.1866],
                #         [ 3.1302, -4.6391,  2.7102],
                #         [ 2.9031, -4.5346,  2.7102],
                #         [ 2.9031, -4.5346, -3.0494],
                #         [ 2.6542, -4.5576, -3.0494],
                #         [ 2.6542, -4.5576,  2.7102],
                #         [ 2.6542, -4.5576,  2.1866],
                #         [ 2.5098, -4.3535,  2.1866],
                #         [ 2.3654, -4.1495,  2.1866],
                #         [ 2.2210, -3.9454,  2.1866]], device='cuda:0')
                # print("_rel_pose: ", _rel_pose)
                # _rel_pose:  tensor([[ 0.0000,  0.0000,  0.0000],
                #         [ 0.0000,  0.0000,  0.5236],
                #         [-0.2271,  0.1045,  0.5236],
                #         [-0.2271,  0.1045,  1.0472],
                #         [-0.4760,  0.0815,  1.0472],
                #         [-0.4760,  0.0815,  0.5236],
                #         [-0.4760,  0.0815,  0.0000],
                #         [-0.6204,  0.2856,  0.0000],
                #         [-0.7648,  0.4897,  0.0000],
                #         [-0.9092,  0.6938,  0.0000]], device='cuda:0')
                pose_coords_list = tutils.get_coord_pose_for_robot_center(sg, _rel_pose, _abs_poses[0], self.test_ds.grid_dim[0],
                                                                     self.test_ds.cell_size, self.device)  # B x T x 3
                # print("pose_coords_list: ", pose_coords_list)  # 10个, shape为[10, 1, ]
                # print("pose_coords_list.shape: ", pose_coords_list.shape)  # torch.Size([10, 1, 2])
                # tensor([[[149, 149]]], device='cuda:0')

                # 6.2 计算不确定度
                # add crop uncertainty to uncertainty map  用于物体搜索的目标选择
                B, T, C, cH, cW = pred_maps_objects.shape
                ### Estimate the variance of each class for each location # 1 x B x T x object_classes x crop_dim x crop_dim
                ensemble_var = torch.zeros((1, B, T, C, cH, cW), dtype=torch.float32).to(self.device)
                for i in range(ensemble_object_maps.shape[3]):  # num of classes
                    ensemble_class = ensemble_object_maps[:, :, :, i, :, :]
                    ensemble_class_var = torch.var(ensemble_class, dim=0, keepdim=True)
                    ensemble_var[:, :, :, i, :, :] = ensemble_class_var
                # TODO: 各类像素的不确定性图
                per_class_uncertainty = ensemble_var.squeeze(0)  # B x T x C x cH x cW
                step_uncertainty = sg.register_per_class_uncertainty(per_class_uncertainty_crop=per_class_uncertainty, pose=_rel_pose, abs_pose=_abs_poses)
                # 6.3 全局地图
                # 添加到全局地图 add semantic prediction to semantic map
                step_geo_grid = sg.register_sem_pred(prediction_crop=pred_maps_objects, pose=_rel_pose, abs_pose=_abs_poses)  # 形状为 torch.Size([1, 10, 27, 400, 400])
                print("   [zhjd-debug-search] step_geo_grid.shape: ", step_geo_grid.shape)
                print("   [zhjd-debug-search] step_uncertainty.shape: ", step_uncertainty.shape)

                # ltg_dist = torch.linalg.norm(ltg.clone().float()-pose_coords.float())*self.options.cell_size # distance to current long-term goal


                # 7. （可选）可视化保存
                # Option to save visualizations of steps
                if self.options.save_nav_images:
                    # save_img_dir_ = self.options.save_img_dir + '/ep_' + str(tstep)  + '/'
                    save_img_dir_ = f"{self.options.save_img_dir}/{scene_id}/{epsoid_name}/"
                    save_img_dir_ = f"{self.options.save_img_dir}/{scene_id}/{epsoid_name}/"
                    print("     [zhjd-debug-search] save_img_dir_: ", save_img_dir_)
                    if not os.path.exists(save_img_dir_):
                        os.makedirs(save_img_dir_)
                    # viz_utils.save_all_infos_and_mapprediction_Global(batch, pred_maps_objects, step_geo_grid, savepath=save_img_dir_, name='path')
                    # viz_utils.save_uncertainty(sg, ltg.clone().cpu().numpy(), pose_coords.clone().cpu().numpy(), save_img_dir_)
                    viz_utils.save_uncertainty(step_geo_grid, step_uncertainty, pose_coords_list.clone().cpu().numpy(), save_img_dir_, timestamp_length=10)

                # 7. 混淆矩阵计算
                current_confusion_matrix_objects = confusion_matrix(y_true=gt_crops_objects.flatten(), y_pred=pred_labels_objects.flatten(), labels=object_labels)
                current_confusion_matrix_objects = torch.tensor(current_confusion_matrix_objects)

                if overall_confusion_matrix_objects is None:
                    overall_confusion_matrix_objects = current_confusion_matrix_objects
                else:
                    overall_confusion_matrix_objects += current_confusion_matrix_objects

        # 7. 计算指标（metrics）
        mAcc_obj = metrics.overall_pixel_accuracy(overall_confusion_matrix_objects)
        class_mAcc_obj, per_class_Acc = metrics.per_class_pixel_accuracy(overall_confusion_matrix_objects)
        mIoU_obj, per_class_IoU = metrics.jaccard_index(overall_confusion_matrix_objects)
        mF1_obj, per_class_F1 = metrics.F1_Score(overall_confusion_matrix_objects)

        self.results_objects['original_result'] = {
            'mean_interesction_over_union_objects': mIoU_obj.numpy().tolist(),
            'mean_f1_score_objects': mF1_obj.numpy().tolist(),
            'overall_pixel_accuracy_objects': mAcc_obj.numpy().tolist(),
            'per_class_pixel_accuracy_objects': class_mAcc_obj.numpy().tolist(),
            'per_class_IoU': per_class_IoU.numpy().tolist(),
            'per_class_mAcc_obj': per_class_Acc.numpy().tolist(),
            'per_class_F1': per_class_F1.numpy().tolist()}

        print("\nSemantic prediction results:")
        classes = list(self.object_classes.keys())
        classes.sort()
        per_class_Acc = per_class_Acc[classes]
        per_class_IoU = per_class_IoU[classes]
        per_class_F1 = per_class_F1[classes]
        for i in range(len(classes)):
            lbl = classes[i]
            self.results_objects[self.object_classes[lbl]] = {'Acc': per_class_Acc[i].item(),
                                                              'IoU': per_class_IoU[i].item(),
                                                              'F1': per_class_F1[i].item()}
            print("Class:", self.object_classes[lbl], "Acc:", per_class_Acc[i], "IoU:", per_class_IoU[i], "F1:", per_class_F1[i])
        mean_per_class_Acc = torch.mean(per_class_Acc)
        mean_per_class_IoU = torch.mean(per_class_IoU)
        mean_per_class_F1 = torch.mean(per_class_F1)
        print("mAcc:", mean_per_class_Acc, "mIoU:", mean_per_class_IoU, "mF1:", mean_per_class_F1)
        self.results_objects['objects_all']['mAcc'] = mean_per_class_Acc.item()
        self.results_objects['objects_all']['mIoU'] = mean_per_class_IoU.item()
        self.results_objects['objects_all']['mF1'] = mean_per_class_F1.item()

        res = {
            # **self.results_spatial,
            **self.results_objects}
        with open(self.options.log_dir+'/sem_map_results.json', 'w') as outfile:
            json.dump(res, outfile, indent=4)

        # save the confusion matrices
        filepath = self.options.log_dir+'/confusion_matrices.npz'
        np.savez_compressed(filepath,
                            # overall_confusion_matrix_spatial=overall_confusion_matrix_spatial,
                            overall_confusion_matrix_objects=overall_confusion_matrix_objects)

        print()
        print(overall_confusion_matrix_objects)

    def get_long_term_goal(self, sg, cost_map):
        ### Choose long term goal
        goal = torch.zeros((sg.per_class_uncertainty_map.shape[0], 1, 2), dtype=torch.int64, device=self.device)
        explored_grid = map_utils.get_explored_grid(sg.proj_grid)
        current_UNexplored_map = 1-explored_grid
        unexplored_cost_map = cost_map * current_UNexplored_map
        unexplored_cost_map = unexplored_cost_map.squeeze(1)
        for b in range(unexplored_cost_map.shape[0]):
            map_ = unexplored_cost_map[b,:,:]
            inds = utils.unravel_index(map_.argmax(), map_.shape)
            goal[b,0,0] = inds[1]
            goal[b,0,1] = inds[0]
        return goal


class SearchSLAMer(object):
    """ Implements testing for prediction models
    """
    def __init__(self, options):
        self.options = options
        print("options:")
        for k in self.options.__dict__.keys():
            print(k, self.options.__dict__[k])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.test_ds = HabitatDataOfflineSLAM(options, config_file=options.config_test_file)

        ensemble_exp_rsmp = os.listdir(self.options.ensemble_dir_rsmp) # ensemble_dir should be a dir that holds multiple experiments
        ensemble_exp_rsmp.sort() # in case the models are numbered put them in order
        N = len(ensemble_exp_rsmp) # number of models in the ensemble
        self.models_dict = {} # keys are the ids of the models in the ensemble
        for n in range(self.options.ensemble_size):
            # self.models_dict[n] = {'predictor_model': get_predictor_from_options(self.options)}

            print("     [zhjd-debug-slam] SearchSLAMer Init predictor_model ...")
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

        self.spatial_classes = {0:"void", 1:"occupied", 2:"free"}
        self.object_classes = {0:"void", 17:"floor", 15:'wall', 3:"table", 4:"cushion", 13:"counter", 1:"chair", 5:"sofa", 6:"bed"}

        # initialize res dicts
        self.results_spatial = {}
        self.results_objects = {}
        for object_ in list(self.object_classes.values()):
            self.results_objects[object_] = {}
        self.results_objects['objects_all'] = {}
        self.results_objects['original_result'] = {}

        for spatial in list(self.spatial_classes.values()):
            self.results_spatial[spatial] = {}
        self.results_spatial['spatial_all'] = {}

    def test_semantic_map(self):
        # 1. 数据加载器
        print("     [zhjd-debug-slam], params test_batch_size: ", self.options.test_batch_size, ", num_workers:", self.options.num_workers)
        test_data_loader = DataLoader(self.test_ds,
                                      # 这是数据集（Dataset）对象，表示测试数据集。它通常是一个继承自 torch.utils.data.Dataset 类的自定义类，用来定义如何加载和预处理数据。
                                      # self.test_ds 中包含了测试数据的所有样本和标签，并实现了如何在 __getitem__ 方法中获取一个样本。
                                batch_size=self.options.test_batch_size,
                                num_workers=self.options.num_workers,
                                pin_memory=self.options.pin_memory,
                                shuffle=False)
        # 2. 初始化变量
        batch = None
        self.options.test_iters = len(test_data_loader) # the length of dataloader depends on the batch size
        print("   [zhjd-debug-slam] self.options.test_iters : ", self.options.test_iters )
        object_labels = list(range(self.options.n_object_classes))
        spatial_labels = list(range(self.options.n_spatial_classes))

        # 3. 遍历测试数据集
        # tstep：当前批次的索引（从 0 开始），通常用于跟踪训练或测试的进度。
        # batch：包含当前批次数据的字典或张量，通常包括输入数据（如图像）和目标标签（如标签图像或类标签）。在循环中，batch 会被用于模型推理或计算损失。
        for tstep, batch in enumerate(tqdm(test_data_loader,   # tqdm 会显示当前进度、每秒处理的批次数以及估计的剩余时间。
                                           desc='Testing',
                                           total=self.options.test_iters)):

            # 用于把一个 batch（批次数据）移动到 GPU，同时跳过不需要放上 GPU 的字段。
            # batch = {k: v.to(self.device) for k, v in batch.items() if k != 'name'}
            scene_id = batch['scene_id'][0]  # 比如 ['TbHJrupSAjP']
            epsoid_name = batch['epsoid_name'][0]  # 比如 ['ep_43_72_TbHJrupSAjP']
            skip_keys = {'name', 'epsoid_name', 'scene_id'}
            batch = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
                if k not in skip_keys
            }


            # 4. 模型推理（Ensemble 集成模型）
            with torch.no_grad():
                # # 获取 ground truth  【没有】

                # # 初始化 ensemble 输出列表
                ensemble_object_maps, ensemble_spatial_maps = [], []
                N = len(self.models_dict) # numbers of models in the ensemble
                print("   [zhjd-debug-slam] range(self.options.ensemble_size): ", range(self.options.ensemble_size))
                for n in range(self.options.ensemble_size):
                    pred_output = self.models_dict[n]['predictor_model'](batch)
                    ensemble_object_maps.append(pred_output['pred_maps_objects'].clone())

                # 5. 集成模型平均预测
                ensemble_object_maps = torch.stack(ensemble_object_maps)  # N x B x T x C x cH x cW
                print("   [zhjd-debug-slam] ensemble_object_maps.shape: ", ensemble_object_maps.shape)

                # Getting the mean predictions from the ensemble
                pred_maps_objects = torch.mean(ensemble_object_maps, dim=0)  # B x T x C x cH x cW
                print("   [zhjd-debug-slam] pred_maps_objects.shape: ", pred_maps_objects.shape)
                # 显示输入的语义栅格投影图
                # step_ego_grid_27 = batch['step_ego_grid_27']
                # viz_utils.show_image_color_and_extract(step_ego_grid_27,"Predicted Map L2M", 27)
                # viz_utils.show_image_color_and_extract(pred_maps_objects,"Predicted Map RSMP", 27)
                # viz_utils.show_image_sseg_2d_label(gt_crops_objects, "GT")

                # Decide label for each location based on predition probs
                pred_labels_objects = torch.argmax(pred_maps_objects.cpu(), dim=2, keepdim=True) # B x T x 1 x cH x cW

                # 6. （可选）可视化保存
                # Option to save visualizations of steps
                if self.options.save_nav_images:
                    # save_img_dir_ = self.options.save_img_dir + '/ep_' + str(tstep)  + '/'
                    save_img_dir_ = f"{self.options.save_img_dir}/{scene_id}/{epsoid_name}/"
                    print("     [zhjd-debug-slam] save_img_dir_: ", save_img_dir_)
                    if not os.path.exists(save_img_dir_):
                        os.makedirs(save_img_dir_)
                    viz_utils.save_all_infos_and_mapprediction_slam(batch, pred_maps_objects, savepath=save_img_dir_, name='path')

