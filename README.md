## 测试SALM版本  #数据集读取用的是HabitatDataOfflineSLAM

python main.py --name slam_2026_1_19 --ensemble_dir  path-model/smp   --log_dir /home/robotlab/semantic-map-prediction/zhjd_logs     --stored_episodes_dir /home/robotlab/dataset/semantic/semantic_datasets/data_slam/    --save_nav_images

## 在MP3D NPZ数据集上计算方差和概率
 + python main.py --name temp_1_21.1  --ensemble_dir  path-model/   --log_dir /home/robotlab/semantic-map-prediction/zhjd_logs    --sem_map_test --stored_episodes_dir /home/robotlab/dataset/semantic/semantic_datasets/data_v6/   --ensemble_size 4   --save_nav_images
 + 实现步骤 
   + 融合全局地图: 【完成】 
   + 保存全局地图为本地图片：【完成】
   + dataload的时候，帧数多一些，会有问题吗？
     + 暂时忽略
   + 计算均值和方差，并保存target_pred和target_uncertainty为图片【完成】
     + 目前只使用了两个参数不同的预测模型
   + 保存四个模型的预测地图 
     + 训练一个新模型 
       + smp 原版              
       + train_for_CJME_1  训练了50轮
       + train_for_CJME_2_from_zeros  虽然中间断了，但是可用
       + train_for_CJME_2_from_zeros_2  训练了50轮

     + python main.py --name train_for_CJME_1 --batch_size 1 --num_workers 1 --is_train --log_dir /home/robotlab/semantic-map-prediction/zhjd_logs  --stored_episodes_dir /home/robotlab/dataset/semantic/semantic_datasets/data_v6/  --num_epochs 50 --ensemble_dir path-model/smp
     + 新模型保存在 /home/robotlab/semantic-map-prediction/zhjd_logs/train_for_CJME_1/checkpoints
     + 可视化训练误差 tensorboard --logdir=~/semantic-map-prediction/zhjd_logs/train_for_CJME_1/tensorboard
     + 取代论文中四个集成模型图片
     + 
   + 下一步, 从TbHJrupSAjP、oLBMNvg9in8两个场景中，超出反应target_pred和target_uncertainty的轨迹，加入到论文中 



## 图神经网络的图片保存
+ self.models_dict[n] = {'predictor_model': get_predictor_rsmp(self.options)
  + 调用处：
  + self.models_dict[n]['predictor_model'](batch)
+ def get_predictor_rsmp(options):
+ def get_mp_network(options):
  + return MapPredictorAM(segmentation_model=get_mp_network(options),
                                  map_loss_scale=options.map_loss_scale)
+ ResNetUNetDAMLastLayerv2
        ResNetUNetDAMLastLayerv2(n_channel_in=options.n_object_classes, n_class_out=options.n_object_classes)
+  class ResNetUNetDAMLastLayerv2(nn.Module):
    其中AE模块在整个网络中承担了跨位置关系建模的功能，通过pool参数在不同尺度上建立特征间的关系
+ class AE(nn.Module):
    sm_y = self.gnn(A, y) + y       语义关系推理
    sp_y = self.spatialgnn(y) + y   空间关系推理
    y = sm_y + sp_y         相加融合
  + class SpatialGNN(nn.Module):  说明SpatialGNN是嵌套在AE内部的


## 滨州王佳莉家
+ 在SLAM NPZ数据集上预测地图
  +  python main.py --name slam_search_2_24.1  --ensemble_dir  path-model/   --log_dir /home/robotlab/semantic-map-prediction/zhjd_logs   --stored_episodes_dir /home/robotlab/dataset/semantic/semantic_datasets/data_binzhou_wjl_2/   --ensemble_size 1    --save_nav_images
+ 在SLAM NPZ数据集上计算方差和概率
  + python main.py --name slam_search_2_22.1  --ensemble_dir  path-model/   --log_dir /home/robotlab/semantic-map-prediction/zhjd_logs   --stored_episodes_dir /home/robotlab/dataset/semantic/semantic_datasets/data_binzhou_wjl/   --ensemble_size 4    --save_nav_images
+ 由于SLAM数据集中，虽然被分成了多个NPZ文件，但是相互之间是连续的，因此需要融合
  + 再MIT中

## 749实验室
+ 使用ROS接收栅格地图
python main.py --name slam_search_3_12.1  --ensemble_dir  path-model/   --log_dir /home/robotlab/semantic-map-prediction/zhjd_logs    --ensemble_size 1    --is_ros  --save_nav_images
+ 完成ros发布
+ 接收地图和frontier
+ 