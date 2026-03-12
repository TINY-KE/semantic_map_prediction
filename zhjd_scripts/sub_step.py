#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from StepEgoMapPose_msgs.msg import StepEgoMapPose

# 颜色映射表 (与你发送端一致)
color_mapping_27 = {
    0: (255, 255, 255), 1: (128, 128, 0), 2: (0, 0, 255), 3: (255, 0, 0),
    4: (255, 0, 255), 5: (0, 255, 255), 6: (255, 165, 0), 7: (255, 255, 0),
    8: (128, 128, 128), 9: (128, 0, 0), 10: (255, 20, 147), 11: (0, 128, 0),
    12: (128, 0, 128), 13: (0, 128, 128), 14: (0, 0, 128), 15: (210, 105, 30),
    16: (188, 143, 143), 17: (0, 255, 0), 18: (255, 215, 0), 19: (0, 0, 0),
    20: (192, 192, 192), 21: (138, 43, 226), 22: (255, 127, 80), 23: (238, 130, 238),
    24: (245, 245, 220), 25: (139, 69, 19), 26: (64, 224, 208)
}

# 转换为 NumPy 查找表
colors_27 = np.zeros((27, 3), dtype=np.uint8)
for i, color in color_mapping_27.items():
    colors_27[i] = color

class SemanticMapListener:
    def __init__(self):
        self.sub = rospy.Subscriber('/step_ego_map_pose', StepEgoMapPose, self.callback)
        rospy.loginfo("Semantic Map Listener Started...")

    def callback(self, msg):
        try:
            # 1. 解析 MultiArray 维度
            # 预期: C=27, H=64, W=64 (或 200)
            dims = {d.label: d.size for d in msg.grid27.layout.dim}
            C = dims.get('channels', 27)
            H = dims.get('height', 64)
            W = dims.get('width', 64)

            # 2. 转换数据为 NumPy 数组
            grid_flat = np.array(msg.grid27.data, dtype=np.float32)
            grid_3d = grid_flat.reshape((C, H, W))

            # 3. 语义可视化：取概率最大的类别
            # 在通道维度(C)上取最大值的索引
            semantic_map = np.argmax(grid_3d, axis=0).astype(np.uint8)

            # 4. 映射颜色 (BGR 格式用于 OpenCV)
            # 注意：colors_27 是 RGB，OpenCV 需要 BGR，所以用 [:, ::-1]
            color_map = colors_27[semantic_map][:, :, ::-1]

            # 5. 放大显示 (64x64 太小，放大到 400x400)
            display_map = cv2.resize(color_map, (400, 400), interpolation=cv2.INTER_NEAREST)

            # 6. 叠加机器人坐标信息
            rx = msg.robot_pose[0].data if isinstance(msg.robot_pose, list) else msg.robot_pose.x
            ry = msg.robot_pose[1].data if isinstance(msg.robot_pose, list) else msg.robot_pose.y
            cv2.putText(display_map, f"Pose: {rx:.2f}, {ry:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 7. 显示
            cv2.imshow("Local Semantic Ego Map", display_map)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"Visualization error: {e}")

if __name__ == '__main__':
    rospy.init_node('semantic_map_visualizer')
    listener = SemanticMapListener()
    rospy.spin()
    cv2.destroyAllWindows()