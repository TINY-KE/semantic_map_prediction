## 测试SALM版本  #数据集读取用的是HabitatDataOfflineSLAM

python main.py --name slam_2026_1_19 --ensemble_dir  path-model/smp   --log_dir /home/robotlab/semantic-map-prediction/zhjd_logs     --stored_episodes_dir /home/robotlab/dataset/semantic/semantic_datasets/data_slam/    --save_nav_images

## 在NPZ数据集上计算方差和概率
 + python main.py --name temp_1_21.1  --ensemble_dir  path-model/   --log_dir /home/robotlab/semantic-map-prediction/zhjd_logs    --sem_map_test --stored_episodes_dir /home/robotlab/dataset/semantic/semantic_datasets/data_v6/   --ensemble_size 4
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

   + 研究图神经网络：
     + 位置：
       + ResNetUNetDAMLastLayerv2(n_channel_in=options.n_object_classes, n_class_out=options.n_object_classes)
       + 