import cv2
import numpy as np
import torch
import rospy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ros_utils.SemanticMapPublisher import  SemanticMarkerPublisher, AsyncDualSemanticPublisher

def visualize_local_image_in_rviz(image_path, publisher, res=0.1, origin_x=0.0, origin_y=0.0):
    """
    读取本地图片并利用 AsyncDualSemanticPublisher 发布到 RViz
    """
    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        rospy.logerr(f"无法读取图片: {image_path}")
        return

    # 2. 将图片转换为灰度图，并缩放到 0-26 的类别区间
    # 这样 argmax 之后每一像素的类别就对应图片原本的亮度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    # 归一化到 0-26 (你的类别总数是 27)
    semantic_indices = (gray / 255.0 * 26).astype(np.int64)

    # 3. 构造 [27, H, W] 的 One-hot 概率分布
    # 因为 publish_semantic_map 内部会做 np.argmax(sseg_np, axis=0)
    ego_grid_sseg = np.zeros((27, H, W), dtype=np.float32)

    # 利用 numpy 索引技巧快速填充
    for cat in range(27):
        ego_grid_sseg[cat][semantic_indices == cat] = 1.0

    # 4. (可选) 构造一个全 True 的 free_mask 确保图片全部显示
    free_mask = np.ones((H, W), dtype=bool)

    # 5. 调用异步发布器
    # 注意：origin_x/y 决定了图片左下角在 RViz 中的位置
    rospy.loginfo(f"正在发布图片可视化: {W}x{H}, 分辨率: {res}")
    publisher.async_publish(
        ego_grid_sseg=ego_grid_sseg,
        free_mask=free_mask,
        res=res,
        origin_x=origin_x,
        origin_y=origin_y,
        height=0.5,  # 稍微抬高一点防止被地面遮挡
        max_area=[-500, 500, -500, 500]  # 给一个足够大的显示区域
    )


# -------------------------- 使用示例 --------------------------
if __name__ == '__main__':
    try:
        # 初始化节点（如果主程序没运行）
        if not rospy.core.is_initialized():
            rospy.init_node('image_to_rviz_test', anonymous=True)

        # 实例化你的异步发布器
        my_publisher = AsyncDualSemanticPublisher(
            raw_topic="/image_raw_marker",
            filtered_topic="/image_filtered_marker"
        )

        # 等待发布器连接
        rospy.sleep(1.0)

        # 路径替换为你的本地图片
        img_path = "/home/robotlab/Desktop/360250808092037974.png"

        # 持续发布以便在 RViz 中查看
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            visualize_local_image_in_rviz(
                img_path,
                my_publisher,
                res=0.1,  # 每个像素占 0.05米
                origin_x=-5.0,  # 图片原点 X
                origin_y=-5.0  # 图片原点 Y
            )
            rate.sleep()

    except rospy.ROSInterruptException:
        pass