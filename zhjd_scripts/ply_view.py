import open3d as o3d

# 1. 读取PLY文件
ply_path = "your_mp3d_scene.ply"
pcd = o3d.io.read_point_cloud(ply_path)  # 读取点云
# 如果是网格文件（MP3D部分PLY是网格），用：mesh = o3d.io.read_triangle_mesh(ply_path)

# 2. 检查是否读取成功
if pcd.is_empty():
    print("PLY文件读取失败，请检查路径！")
else:
    print(f"点云数量：{len(pcd.points)}")

    # 3. 可视化（会弹出3D窗口，支持鼠标旋转/缩放/平移）
    o3d.visualization.draw_geometries([pcd], window_name="MP3D PLY 3D可视化")