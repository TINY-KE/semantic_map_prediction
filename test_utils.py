
import numpy as np
import quaternion
import datasets.util.utils as utils
import datasets.util.map_utils as map_utils
import torch
import os

def get_latest_model(save_dir):
    checkpoint_list = []
    for dirpath, _, filenames in os.walk(save_dir):
        for filename in filenames:
            if filename.endswith('.pt'):
                print('model path ',os.path.join(dirpath, filename))

                checkpoint_list.append(os.path.abspath(os.path.join(dirpath, filename)))
    checkpoint_list = sorted(checkpoint_list)
    latest_checkpoint = None if (len(checkpoint_list) is 0) else checkpoint_list[-1]
    return latest_checkpoint


def load_model(models, checkpoint_file):
    # Load the latest checkpoint
    checkpoint = torch.load(checkpoint_file)
    # print('checkpoint models ', checkpoint['models']['predictor_model'].keys())

    for model in models:
        if model in checkpoint['models']:
            models[model].load_state_dict(checkpoint['models'][model])
        else:
            raise Exception("Missing model in checkpoint: {}".format(model))
    return models


def get_2d_pose(position, rotation):
    # position is 3-element list
    # rotation is 4-element list representing a quaternion
    position = np.asarray(position, dtype=np.float32)
    x = -position[2]
    y = -position[0]
    height = position[1]

    rotation = np.quaternion(rotation[0], rotation[1], rotation[2], rotation[3])
    axis = quaternion.as_euler_angles(rotation)[0]
    if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
        o = quaternion.as_euler_angles(rotation)[1]
    else:
        o = 2*np.pi - quaternion.as_euler_angles(rotation)[1]
    if o > np.pi:
        o -= 2 * np.pi

    pose = x, y, o
    return pose, height

# zhjd 改编
# 根据相对位姿 rel_pose 和初始位姿 init_pose，通过空间变换（spatial transformer）计算出 当前位姿在语义地图网格中的坐标。
# sg	包含 spatialTransformer 方法的类（通常是 Semantic Grid 模型）
# rel_pose	相对起始位置的 pose（T × 3）
# init_pose	初始位姿（起点 pose），shape 应为 [T, 3]
# grid_dim	地图的宽/高（正方形）
# cell_size	地图中每个 cell 的真实世界尺寸（如 0.1m）
# device	执行模型的设备（如 'cuda'）
def get_coord_pose_for_robot_center(sg, rel_pose, init_pose, grid_dim, cell_size, device):
    # Create a grid where the starting location is always at the center looking upwards (like the ground-projected grids)
    # Then use the spatial transformer to move that location at the right place
    # 判断变量 init_pose 是否是一个 list（列表） 或者 tuple（元组） 类型。如果是列表或元组，就将它转换为张量，并加一个 batch 维度。
    if isinstance(init_pose, list) or isinstance(init_pose, tuple):
        init_pose = torch.tensor(init_pose).unsqueeze(0)
    else:
        init_pose = init_pose.unsqueeze(0)
    init_pose = init_pose.to(device)

    zero_pose = torch.tensor([[0., 0., 0.]]).to(device)

    zero_coords = map_utils.discretize_coords(x=zero_pose[:,0],
                                            z=zero_pose[:,1],
                                            grid_dim=(grid_dim, grid_dim),
                                            cell_size=cell_size)

    pose_grid = torch.zeros((rel_pose.shape[0], 1, grid_dim, grid_dim), dtype=torch.float32).to(device)

    for i in range(rel_pose.shape[0]):
        pose_grid[i, 0, zero_coords[0, 0], zero_coords[0, 1]] = 1

    # 位移后的ego地图  grid -- sequence len * number of classes * grid_dim * grid_dim
    pose_grid_transf = sg.spatialTransformer(grid=pose_grid, pose=rel_pose, abs_pose=init_pose)
    # print("pose_grid_transf.shape: ", pose_grid_transf.shape)     得到 torch.Size([10, 1, 300, 300])

    pose_coords_list = []
    for i in range(pose_grid_transf.shape[0]):
        pose_grid_transf_single = pose_grid_transf[i].squeeze(0)
        # print("pose_grid_transf_single.shape: ", pose_grid_transf_single.shape)     # 得到torch.Size([300, 300]

        # 找到 pose_grid_transf_single 张量中最大值的位置坐标（索引），并将其从扁平索引转换为多维索引（坐标）。
        inds = utils.unravel_index(pose_grid_transf_single.argmax(), pose_grid_transf_single.shape)
        # print("inds: ", inds)     # 得到  tensor([  0,   0, 149, 149])

        pose_coord = torch.zeros((1, 1, 2), dtype=torch.int64).to(device)
        pose_coord[0,0,0] = inds[1] # inds is y,x
        pose_coord[0,0,1] = inds[0]
        # print("pose_coord: ", pose_coord)     # 得到
        pose_coords_list.append(pose_coord)

    return torch.stack(pose_coords_list)   #  torch.Size([10, 1, 2])


def get_closest_target_location(sg, pose_coords, sem_lbl, cell_size, sem_thresh):
    # Uses euclidean distance, not geodesic
    pose_coords = pose_coords.squeeze(0)

    sem_lbl_map = sg.sem_grid[:, sem_lbl, :, :] # B x H x W
    sem_lbl_map = sem_lbl_map.squeeze(0)
    inds = torch.nonzero(torch.where(sem_lbl_map>=sem_thresh, 1, 0))
    
    if inds.shape[0]==0: # semantic target not in the map yet
        return None, None, None

    obj_coords = torch.zeros(inds.size()).to(pose_coords.device)
    obj_coords[:,0] = inds[:,1]
    obj_coords[:,1] = inds[:,0]

    dist = torch.linalg.norm(obj_coords - pose_coords, dim=1) * cell_size
    min_ind = torch.argmin(dist)
    min_dist = dist[min_ind]

    coord = torch.zeros((2), dtype=torch.int64).to(pose_coords.device)
    coord[0], coord[1] = int(obj_coords[min_ind,0]), int(obj_coords[min_ind,1])
    prob = sem_lbl_map[coord[1], coord[0]] 
    return coord, min_dist, prob


def get_cost_map(sg, sem_lbl, a_1, a_2):
    p_map = sg.sem_grid[:, sem_lbl, :, :]
    sigma_map = torch.sqrt(sg.per_class_uncertainty_map[:, sem_lbl, :, :])
    return p_map + torch.sign(a_2-p_map) * a_1 * sigma_map


def decide_stop(dist, stop_dist):
    # If closest occurence of semantic target is less than a theshold (0.8m?) then stop
    if dist is None: # case when object has not been observed yet
        return False
    if dist <= stop_dist:
        return True
    else:
        return False


# Return success, SPL, soft_SPL, distance_to_goal measures
def get_metrics(sim,
                episode_goal_positions,
                success_distance,
                start_end_episode_distance,
                agent_episode_distance,
                stop_signal):

    curr_pos = sim.get_agent_state().position

    # returns distance to the closest goal position
    distance_to_goal = sim.geodesic_distance(curr_pos, episode_goal_positions)

    if distance_to_goal <= success_distance and stop_signal:
        success = 1.0
    else:
        success = 0.0

    spl = success * (start_end_episode_distance / max(start_end_episode_distance, agent_episode_distance) )

    ep_soft_success = max(0, (1 - distance_to_goal / start_end_episode_distance) )
    soft_spl = ep_soft_success * (start_end_episode_distance / max(start_end_episode_distance, agent_episode_distance) )

    metrics = {'distance_to_goal':distance_to_goal,
               'success':success,
               'spl':spl,
               'softspl':soft_spl}
    return metrics


