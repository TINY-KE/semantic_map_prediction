#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import torch
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import threading
import queue


# -------------------------- 语义Marker双重发布器 --------------------------
class SemanticMarkerPublisher:
    def __init__(self, raw_topic="/semantic_raw_map", filtered_topic="/semantic_free_only_map"):
        if not rospy.core.is_initialized():
            rospy.init_node('semantic_dual_publisher_node', anonymous=True, disable_signals=True)

        # 定义两个发布者
        self.pub_raw = rospy.Publisher(raw_topic, Marker, queue_size=10)
        self.pub_filtered = rospy.Publisher(filtered_topic, Marker, queue_size=10)

        self.color_palette = self._get_custom_semantic_colors()

        rospy.loginfo(f"双重语义发布器启动: \n 原图: {raw_topic} \n 过滤图: {filtered_topic}")

    def _get_custom_semantic_colors(self):
        color_mapping_27 = {
            0: (255, 255, 255), 1: (128, 128, 0), 2: (0, 0, 255), 3: (255, 0, 0),
            4: (255, 0, 255), 5: (0, 255, 255), 6: (255, 165, 0), 7: (255, 255, 0),
            8: (128, 128, 128), 9: (128, 0, 0), 10: (255, 20, 147), 11: (0, 128, 0),
            12: (128, 0, 128), 13: (0, 128, 128), 14: (0, 0, 128), 15: (210, 105, 30),
            16: (188, 143, 143), 17: (0, 255, 0), 18: (255, 215, 0), 19: (0, 0, 0),
            20: (192, 192, 192), 21: (138, 43, 226), 22: (255, 127, 80), 23: (238, 130, 238),
            24: (245, 245, 220), 25: (139, 69, 19), 26: (64, 224, 208)
        }
        colors = []
        for idx in range(27):
            r, g, b = color_mapping_27[idx]
            colors.append((r / 255.0, g / 255.0, b / 255.0, 0.8))
        return np.array(colors)

    def _to_numpy(self, data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data

    def _create_marker(self, ns, frame_id, res, height):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.scale.x = res
        marker.scale.y = res
        marker.scale.z = 0.02
        marker.pose.orientation.w = 1.0
        marker.color.a = 1.0
        return marker

    def publish_semantic_map(self, ego_grid_sseg, free_mask=None, res=0.1, origin_x=-15.0, origin_y=-15.0, height=0.0,
                             max_area=[-100, 100, -100, 100], y_offset_cells = 0):
        # 1. 数据预处理
        sseg_np = self._to_numpy(ego_grid_sseg)
        if sseg_np.ndim == 4: sseg_np = sseg_np.squeeze(0)
        raw_indices = np.argmax(sseg_np, axis=0)

        # 2. 准备偏移量 (米)
        offset_y_meters = y_offset_cells * res

        # 3. 内部填充函数：增加 offset 参数
        def fill_marker(target_marker, indices_map, y_offset=0.0):
            min_x, max_x, min_y, max_y = max_area
            for cat in range(27):
                y_idx, x_idx = np.where(indices_map == cat)
                if len(y_idx) == 0: continue

                px = x_idx * res + origin_x
                py = y_idx * res + origin_y

                # 物理边界过滤 (注意：边界判定通常基于原始位置)
                mask = (px >= min_x) & (px <= max_x) & (py >= min_y) & (py <= max_y)
                px_f, py_f = px[mask], py[mask]

                color = ColorRGBA(*self.color_palette[cat])
                for x, y in zip(px_f, py_f):
                    # 关键点：在发布坐标时加上偏移
                    target_marker.points.append(Point(x, y + y_offset, height))
                    target_marker.colors.append(color)

        # 4. 发布原图 (偏移为 0)
        raw_marker = self._create_marker("raw_semantic", "map", res, height)
        fill_marker(raw_marker, raw_indices, y_offset=0.0)
        self.pub_raw.publish(raw_marker)

        # 5. 发布过滤图 (带偏移)
        if free_mask is not None:
            f_mask_np = self._to_numpy(free_mask)
            if f_mask_np.ndim == 3: f_mask_np = f_mask_np.squeeze(0)
            if f_mask_np.shape != raw_indices.shape: f_mask_np = f_mask_np.T

            # 仅保留在 free_mask 区域内的索引
            filtered_indices = np.where(f_mask_np > 0, raw_indices, -1)

            filtered_marker = self._create_marker("free_filtered_semantic", "map", res, height + 0.02)
            # 同步偏移：地图内容被 mask 截取后，整体平移 offset_y_meters
            fill_marker(filtered_marker, filtered_indices, y_offset=offset_y_meters)
            self.pub_filtered.publish(filtered_marker)

        rospy.loginfo_once("Dual Semantic Markers Published.")


# -------------------------- 异步双重发布器 --------------------------
class AsyncDualSemanticPublisher(SemanticMarkerPublisher):
    def __init__(self, raw_topic="/semantic_global_map", filtered_topic="/semantic_global_map_free_only", queue_size=5):
        super().__init__(raw_topic, filtered_topic)
        self.publish_queue = queue.Queue(maxsize=queue_size)
        self.is_running = True
        self.publish_thread = threading.Thread(target=self._worker, daemon=True)
        self.publish_thread.start()
        self.y_offset_cells = 40
    def _worker(self):
        while self.is_running and not rospy.is_shutdown():
            try:
                args = self.publish_queue.get(timeout=1.0)
                self.publish_semantic_map(*args)
                self.publish_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                rospy.logerr(f"Async Publish Error: {e}")

    def async_publish(self, ego_grid_sseg, free_mask=None, res=0.1, origin_x=-10.0, origin_y=-10.0, height=0.0,
                      max_area=[-500, 500, -500, 500], y_offset_cells = 0):
        try:
            if self.publish_queue.full(): self.publish_queue.get_nowait()
            self.publish_queue.put_nowait((ego_grid_sseg, free_mask, res, origin_x, origin_y, height, max_area, y_offset_cells))
        except Exception as e:
            rospy.logerr(f"Queue Error: {e}")

    def stop(self):
        self.is_running = False
        self.publish_thread.join(timeout=1.0)


import matplotlib.pyplot as plt  # 用于获取伪彩色映射

import rospy
import numpy as np
import torch
import matplotlib.cm as cm
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import threading
import queue


class AsyncUncertaintyMarkerPublisher(SemanticMarkerPublisher):
    def __init__(self, raw_topic="/uncertainty_raw_map",
                 filtered_topic="/uncertainty_aligned_map",
                 queue_size=5):
        super(AsyncUncertaintyMarkerPublisher, self).__init__(raw_topic, filtered_topic)
        # self.cmap = cm.get_cmap('jet')
        self.cmap = cm.get_cmap('viridis')
        self.publish_queue = queue.Queue(maxsize=queue_size)
        self.is_running = True
        self.publish_thread = threading.Thread(target=self._worker, daemon=True)
        self.publish_thread.start()
        # self.y_offset_cells = -45  # 默认值
        rospy.loginfo(f"🚀 异步不确定性发布器(自动降维集成版)启动")

    def _worker(self):
        while self.is_running and not rospy.is_shutdown():
            try:
                args = self.publish_queue.get(timeout=1.0)
                self.process_and_publish(*args)
                self.publish_queue.task_done()
            except queue.Empty:
                continue

    def async_publish(self, uncertainty_grid, free_mask=None, res=0.1, origin_x=-15.0, origin_y=-15.0, height=0.0,
                      max_area=[-100, 100, -100, 100], threshold=0.05, robot_pose_px = None, y_offset_cells = 0):
        """
        现在你可以直接传入 sg.per_class_uncertainty_map [B, 27, H, W]
        """
        try:
            # --- 关键集成：在入队前或入队后处理降维 ---
            # 为了主线程负担最小，我们将原始数据丢进队列，在 worker 线程里处理 torch.max
            if self.publish_queue.full():
                self.publish_queue.get_nowait()

            self.publish_queue.put_nowait(
                (uncertainty_grid, free_mask, res, origin_x, origin_y, height, max_area, threshold, robot_pose_px, y_offset_cells))
        except Exception as e:
            rospy.logerr(f"Uncertainty Queue Error: {e}")



    def process_and_publish(self, uncertainty_grid, free_mask, res, origin_x, origin_y, height, max_area, threshold,
                            robot_pose_px, y_offset_cells):
        # 1. 维度处理
        unc_np = self._to_numpy(uncertainty_grid)
        if unc_np.ndim == 4:
            uncertainty_grid, _ = torch.max(uncertainty_grid, dim=1)
            unc_np = self._to_numpy(uncertainty_grid)
        if unc_np.ndim == 3: unc_np = unc_np.squeeze(0)

        # 2. 【核心修改】步长下采样：每 3 个栅格取 1 个
        # 这样处理 1000x1000 的图实际上只处理约 333x333，速度飞快
        stride = 1
        unc_downsampled = unc_np[::stride, ::stride]

        # 3. 向量化获取高分索引
        y_local, x_local = np.where(unc_downsampled > threshold)
        if len(y_local) == 0: return

        # 还原回原始坐标系的索引
        y_idx = y_local * stride
        x_idx = x_local * stride
        vals = unc_downsampled[y_local, x_local]

        # 4. 向量化物理坐标计算 (注意：res 也要乘以步长使 Marker 大小看起来连续，或者保持 res 只显示散点)
        # 这里建议将 Marker 的 scale 设置为 res * stride，这样视觉上是满的
        display_res = res * stride
        px = x_idx * res + origin_x
        py = y_idx * res + origin_y

        # 5. 向量化过滤：物理边界
        min_x, max_x, min_y, max_y = max_area
        spatial_mask = (px >= min_x) & (px <= max_x) & (py >= min_y) & (py <= max_y)
        px, py, vals, x_idx, y_idx = px[spatial_mask], py[spatial_mask], vals[spatial_mask], x_idx[spatial_mask], y_idx[
            spatial_mask]

        # 6. 向量化计算：距离降级 (10米外 > 0.6 的设为 0.6)
        dist_threshold_m = 10.0
        rx_px, ry_px = robot_pose_px[0], robot_pose_px[1]
        dist_px_sq = (x_idx - rx_px) ** 2 + (y_idx - ry_px) ** 2

        dist_mask = dist_px_sq > (dist_threshold_m / res) ** 2
        vals[dist_mask & (vals > 0.6)] = 0.6

        # 7. 向量化映射颜色
        norm_vals = np.clip(vals / 0.8, 0.0, 1.0)
        colors_rgba = self.cmap(norm_vals)

        # 8. 快速构建 Marker
        raw_marker = self._create_marker("raw_unc", "map", display_res, height)
        aligned_marker = self._create_marker("aligned_unc", "map", display_res, height + 0.01)

        # 统一缩放 Marker 大小，使其看起来依然是连续的
        raw_marker.scale.x = raw_marker.scale.y = display_res

        points = [Point(x, y, height) for x, y in zip(px, py)]
        colors = [ColorRGBA(c[0], c[1], c[2], 0.8) for c in colors_rgba]

        raw_marker.points = points
        raw_marker.colors = colors

        # 9. Aligned Map 过滤
        if free_mask is not None:
            if free_mask.shape != unc_np.shape: free_mask = free_mask.T
            # 在原始 free_mask 中采样
            f_mask_at_pts = free_mask[y_idx, x_idx] > 0

            indices = np.where(f_mask_at_pts)[0]
            # 👇 只对过滤后的点应用 Y 轴偏移（30格）
            offset_y_meters = y_offset_cells * res  # 30格 × 分辨率

            # 8. 快速构建 Raw Marker (无偏移)
            raw_marker = self._create_marker("raw_unc", "map", display_res, height)
            raw_marker.points = [Point(x, y, height) for x, y in zip(px, py)]
            raw_marker.colors = [ColorRGBA(c[0], c[1], c[2], 0.8) for c in colors_rgba]
            self.pub_raw.publish(raw_marker)

            # 9. 构建 Aligned Marker (内容被 mask 过滤，但坐标整体偏移)
            if free_mask is not None:
                if free_mask.shape != unc_np.shape: free_mask = free_mask.T
                # 在原始索引位置采样 mask
                f_mask_at_pts = free_mask[y_idx, x_idx] > 0
                indices = np.where(f_mask_at_pts)[0]

                aligned_marker = self._create_marker("aligned_unc", "map", display_res, height + 0.01)
                # 关键：px[i], py[i] 是原位置，加 offset 变成偏移位置
                aligned_marker.points = [
                    Point(px[i], py[i] + offset_y_meters, height + 0.01)
                    for i in indices
                ]
                aligned_marker.colors = [ColorRGBA(colors_rgba[i][0], colors_rgba[i][1], colors_rgba[i][2], 0.8) for i in
                                         indices]
                self.pub_filtered.publish(aligned_marker)