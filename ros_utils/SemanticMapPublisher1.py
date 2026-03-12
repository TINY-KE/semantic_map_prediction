#!/usr/bin/env python
import rospy
import numpy as np
import torch
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import threading  # 新增：线程库
import queue      # 新增：队列（线程安
        
# -------------------------- 语义Marker发布器（适配你的配色） --------------------------
class SemanticMarkerPublisher:
    def __init__(self, marker_topic="/semantic_global_map"):
        """
        初始化语义地图发布器（复用已初始化的ROS节点，不再重复init_node）
        :param marker_topic: Marker发布的ROS话题名
        """
        # 关键修复：移除 rospy.init_node()，复用主程序已初始化的节点
        if not rospy.core.is_initialized():
            # 兜底：若主程序未初始化，才初始化（避免完全无节点的情况）
            rospy.init_node('semantic_ego_marker_node', anonymous=True, disable_signals=True)
        
        # 初始化Publisher（复用现有节点）
        self.pub = rospy.Publisher(marker_topic, Marker, queue_size=10)
        self.rate = rospy.Rate(1)  # 1Hz发布频率
        rospy.loginfo(f"Ego语义地图Marker发布器已启动，话题：{marker_topic}")
        
        # 加载你的27类配色（RGB 255 → 0-1浮点数，添加透明度）
        self.color_palette = self._get_custom_semantic_colors()

    def _get_custom_semantic_colors(self):
        """
        加载你的 color_mapping_27 配色表
        转换规则：RGB(255,255,255) → (1.0,1.0,1.0)，透明度固定为0.8
        """
        # 你的原始配色（RGB 255格式）
        color_mapping_27 = {
            0: (255, 255, 255), 1: (128, 128, 0), 2: (0, 0, 255), 3: (255, 0, 0), 
            4: (255, 0, 255), 5: (0, 255, 255), 6: (255, 165, 0), 7: (255, 255, 0), 
            8: (128, 128, 128), 9: (128, 0, 0), 10: (255, 20, 147), 11: (0, 128, 0), 
            12: (128, 0, 128), 13: (0, 128, 128), 14: (0, 0, 128), 15: (210, 105, 30), 
            16: (188, 143, 143), 17: (0, 255, 0), 18: (255, 215, 0), 19: (0, 0, 0), 
            20: (192, 192, 192), 21: (138, 43, 226), 22: (255, 127, 80), 23: (238, 130, 238), 
            24: (245, 245, 220), 25: (139, 69, 19), 26: (64, 224, 208)
        }
        
        # 转换为ROS Marker需要的格式：(R/255, G/255, B/255, 透明度)
        colors = []
        for idx in range(27):
            r, g, b = color_mapping_27[idx]
            # 转换为0-1浮点数，透明度固定为0.8（可根据需要调整）
            colors.append((r/255.0, g/255.0, b/255.0, 0.8))
        
        return np.array(colors)

    def ego_grid_to_semantic_map(self, ego_grid_sseg):
        """
        将Ego语义栅格（概率分布）转换为类别索引数组
        ego_grid_sseg: torch张量 [1, 27, H, W] 或 numpy数组 [27, H, W]
        返回: numpy数组 [H, W] (0-26)
        """
        # 1. 处理张量/数组兼容
        if isinstance(ego_grid_sseg, torch.Tensor):
            semantic_grid = ego_grid_sseg.cpu().numpy()
        else:
            semantic_grid = ego_grid_sseg
        
        # 2. 去除批次维度（若有）
        if semantic_grid.ndim == 4:  # [1,27,H,W]
            semantic_grid = semantic_grid.squeeze(0)
        
        # 3. 取每个栅格概率最大的类别
        semantic_grid = np.argmax(semantic_grid, axis=0)
        return semantic_grid

    def publish_semantic_map(self, ego_grid_sseg, free_mask=None, res=0.1, origin_x=-15.0, origin_y=-15.0, height=0.0,
                             max_area=[-100, 100, -100, 100]):
        """
        :param ego_grid_sseg: [27, H, W] 的语义概率分布
        :param free_mask: [H, W] 的布尔矩阵，True 表示该区域为空闲（可通行）
        """
        # 1. 预处理数据
        sseg_np = self._process_input(ego_grid_sseg)
        if sseg_np.ndim == 4: sseg_np = sseg_np.squeeze(0)

        semantic_indices = np.argmax(sseg_np, axis=0)  # [H, W]

        # 2. 如果提供了 free_mask，则将非 free 区域的类别设为无效（如 -1）
        if free_mask is not None:
            f_mask_np = self._process_input(free_mask)
            if f_mask_np.ndim == 3: f_mask_np = f_mask_np.squeeze(0)
            # 只有 free_mask 为 True 的地方才保留语义，其余设为 -1
            semantic_indices = np.where(f_mask_np > 0, semantic_indices, -1)

        # 3. 创建 Marker
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "filtered_semantic_map"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.scale.x = res
        marker.scale.y = res
        marker.scale.z = 0.02
        marker.pose.orientation.w = 1.0

        points = []
        colors = []
        min_x, max_x, min_y, max_y = max_area

        # 4. 遍历类别进行物理过滤和区域过滤
        for cat in range(27):
            y_indices, x_indices = np.where(semantic_indices == cat)
            if len(y_indices) == 0: continue

            # 转换为世界坐标
            px = x_indices * res + origin_x
            py = y_indices * res + origin_y

            # 区域限制过滤
            spatial_mask = (px >= min_x) & (px <= max_x) & (py >= min_y) & (py <= max_y)
            px_f = px[spatial_mask]
            py_f = py[spatial_mask]

            if len(px_f) == 0: continue

            # 批量填充
            color_rgba = ColorRGBA(
                self.color_palette[cat, 0], self.color_palette[cat, 1],
                self.color_palette[cat, 2], self.color_palette[cat, 3]
            )
            for x, y in zip(px_f, py_f):
                points.append(Point(x, y, height))
                colors.append(color_rgba)

        marker.points = points
        marker.colors = colors
        self.pub.publish(marker)
        rospy.loginfo_once(f"语义地图已与FreeMask融合发布，点数: {len(points)}")


# -------------------------- 异步语义地图发布器 --------------------------
class AsyncSemanticMarkerPublisher(SemanticMarkerPublisher):
    def __init__(self, marker_topic="/semantic_ego_markers", queue_size=5):
        # 关键修复：父类仅传 marker_topic，num_classes 无需传递（固定27类）
        super().__init__(marker_topic=marker_topic)
        self.publish_queue = queue.Queue(maxsize=queue_size)
        self.is_running = True
        self.publish_thread = threading.Thread(target=self._async_publish_worker, daemon=True)
        self.publish_thread.start()
        rospy.loginfo("异步语义地图发布线程已启动")

    def _async_publish_worker(self):
        """子线程发布逻辑"""
        while self.is_running and not rospy.is_shutdown():
            try:
                data = self.publish_queue.get(timeout=1.0)
                ego_grid_sseg, res, origin_x, origin_y, height, max_area = data  # 新增max_area
                super().publish_semantic_map(ego_grid_sseg, res, origin_x, origin_y, height, max_area)
                self.publish_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                rospy.logerr(f"异步发布失败：{e}")
                import traceback
                rospy.logerr(traceback.format_exc())

    def async_publish_semantic_map(self, ego_grid_sseg, res=0.1, origin_x=-10.0, origin_y=-10.0, height=-0.5, max_area = [-100,100,-100,100]):
        """主线程非阻塞发布"""
        try:
            if self.publish_queue.full():
                try:
                    self.publish_queue.get_nowait()
                except queue.Empty:
                    pass
            self.publish_queue.put_nowait((ego_grid_sseg, res, origin_x, origin_y, height, max_area))
        except Exception as e:
            rospy.logerr(f"放入发布队列失败：{e}")

    def async_publish_semantic_map_and_freemask(self, ego_grid_sseg, free_mask, res=0.1, origin_x=-10.0, origin_y=-10.0, height=-0.5, max_area = [-100,100,-100,100]):
        """主线程非阻塞发布"""
        try:
            if self.publish_queue.full():
                try:
                    self.publish_queue.get_nowait()
                except queue.Empty:
                    pass
            self.publish_queue.put_nowait((ego_grid_sseg, res, origin_x, origin_y, height, max_area))
        except Exception as e:
            rospy.logerr(f"放入发布队列失败：{e}")

    def stop(self):
        """停止异步线程"""
        self.is_running = False
        if self.publish_thread.is_alive():
            self.publish_thread.join(timeout=2.0)
        rospy.loginfo("异步语义地图发布线程已停止")



# -------------------------- 测试用例（单独运行时） --------------------------
if __name__ == '__main__':
    try:
        # 初始化发布器（此时会初始化节点，因为是单独运行）
        pub = SemanticMarkerPublisher()
        
        # 模拟Ego语义栅格（替换为你的实际Ego地图）
        H, W = 300, 300
        ego_grid_sseg = torch.rand(1, 27, H, W)  # 模拟概率分布

        # 循环发布
        rate = rospy.Rate(1)  # 1Hz
        while not rospy.is_shutdown():
            pub.publish_semantic_map(ego_grid_sseg, res=0.1, origin_x=-15.0, origin_y=-15.0, height=0.0)
            rate.sleep()

    except rospy.ROSInterruptException:
        rospy.loginfo("Ego语义地图发布器已停止")
    except Exception as e:
        rospy.logerr(f"发布失败：{e}")