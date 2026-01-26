## 基于关系推理和主动训练的室内语义地图预测方法

### 配置环境 

创建conda环境:

```
conda create -n myenv python=3.8 -y
conda activate myenv
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -y
pip install -r requirements.txt

```

安装 habitat-sim 和 habitat-lab:

```` 
# 克隆 habitat-lab
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout v0.2.1

# 初始化 submodule habitat-sim
git submodule update --init --recursive
cd habitat-sim
git checkout v0.2.2
````


Clone the repository and install other requirements:

```
https://github.com/TINY-KE/semantic_map_prediction.git
cd semantic_map_prediction;
当前路径为$ROOT_PATH 
```

### 下载数据集


下载数据集 [here](https://www.dropbox.com/scl/fi/4dpko4s8fhm1bj3lbx9ng/datasets.zip?rlkey=nuvpibd5cus5cioiqtk3v1fz2&dl=0),

```
your-datasets-path/
  mp3d_objnav_episodes_tmp/
    train/
      1LXtFkjw3qL/
        ep_1_40970_1LXtFkjw3qL.npz
        ...
    val/
      VVfe2KiqLaN/
        ep_1_16987_VVfe2KiqLaN.npz
        ...
    test/
      2azQ1b91cZZ/
        ep_1_1_2azQ1b91cZZ.npz
        ...
```



### 使用方法

#### 训练

从零开始训练:

```
python main.py --name train-x --batch_size 1 --num_workers 1 --is_train --log_dir you-log-path --stored_episodes_dir you-datasets-path/mp3d_objnav_episodes_tmp/ --num_epochs 50
```



#### 预训练模型: 

由 [此处](https://www.dropbox.com/scl/fo/z2kj03w1eq86sx33n91h2/h?rlkey=6wlpclxzblollb4wyhkf8i6a2&dl=0)下载, 并按以下路径放置

```
$ROOT_PATH/
  model-path/
      model-name/
        model/
          smp.pt
```

#### 在NPZ数据集上运行:

```
python main.py --name test --ensemble_dir model-path/model-name/ --log_dir your-log-dir --sem_map_test --stored_episodes_dir you-datasets-path/mp3d_objnav_episodes_tmp/ 
```

#### 在SLAM数据集上运行:

待整理


#### 在Habitat仿真环境中运行:

待整理
