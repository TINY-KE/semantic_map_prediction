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
                             max_area=[-100, 100, -100, 100]):
        # 1. 数据预处理
        sseg_np = self._to_numpy(ego_grid_sseg)
        if sseg_np.ndim == 4: sseg_np = sseg_np.squeeze(0)
        raw_indices = np.argmax(sseg_np, axis=0)  # 原始类别图 [H, W]

        # 2. 准备 Marker
        frame_id = "map"
        raw_marker = self._create_marker("raw_semantic", frame_id, res, height)

        filtered_marker = None
        filtered_indices = None
        if free_mask is not None:
            filtered_marker = self._create_marker("free_filtered_semantic", frame_id, res, height + 0.02)
            f_mask_np = self._to_numpy(free_mask)
            if f_mask_np.ndim == 3: f_mask_np = f_mask_np.squeeze(0)
            # 仅保留空闲区域，其余设为 -1
            filtered_indices = np.where(f_mask_np > 0, raw_indices, -1)

        # 3. 核心提取函数 (内部闭包优化性能)
        def fill_marker(target_marker, indices_map):
            min_x, max_x, min_y, max_y = max_area
            for cat in range(27):
                y_idx, x_idx = np.where(indices_map == cat)
                if len(y_idx) == 0: continue

                px = x_idx * res + origin_x
                py = y_idx * res + origin_y

                # 物理边界过滤
                mask = (px >= min_x) & (px <= max_x) & (py >= min_y) & (py <= max_y)
                px_f, py_f = px[mask], py[mask]
                if len(px_f) == 0: continue

                color = ColorRGBA(*self.color_palette[cat])
                for x, y in zip(px_f, py_f):
                    target_marker.points.append(Point(x, y, height))
                    target_marker.colors.append(color)

        # 4. 执行填充与发布
        fill_marker(raw_marker, raw_indices)
        self.pub_raw.publish(raw_marker)

        if filtered_marker is not None:
            fill_marker(filtered_marker, filtered_indices)
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
                      max_area=[-500, 500, -500, 500]):
        try:
            if self.publish_queue.full(): self.publish_queue.get_nowait()
            self.publish_queue.put_nowait((ego_grid_sseg, free_mask, res, origin_x, origin_y, height, max_area))
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
                      max_area=[-100, 100, -100, 100], threshold=0.05):
        """
        现在你可以直接传入 sg.per_class_uncertainty_map [B, 27, H, W]
        """
        try:
            # --- 关键集成：在入队前或入队后处理降维 ---
            # 为了主线程负担最小，我们将原始数据丢进队列，在 worker 线程里处理 torch.max
            if self.publish_queue.full():
                self.publish_queue.get_nowait()

            self.publish_queue.put_nowait(
                (uncertainty_grid, free_mask, res, origin_x, origin_y, height, max_area, threshold))
        except Exception as e:
            rospy.logerr(f"Uncertainty Queue Error: {e}")

    def process_and_publish(self, uncertainty_grid, free_mask, res, origin_x, origin_y, height, max_area, threshold):
        # ========== 1. 维度处理（保持不变） ==========
        unc_np = self._to_numpy(uncertainty_grid)
        if unc_np.ndim == 4:
            uncertainty_grid, _ = torch.max(uncertainty_grid, dim=1)
            unc_np = self._to_numpy(uncertainty_grid)
        if unc_np.ndim == 3:
            unc_np = unc_np.squeeze(0)  # 最终维度 [H, W]

        # ========== 2. Free Mask 处理（保持不变） ==========
        f_mask_np = None
        if free_mask is not None:
            f_mask_np = self._to_numpy(free_mask)
            if f_mask_np.ndim == 3:
                f_mask_np = f_mask_np.squeeze(0)
            # 强制对齐 mask 维度（关键：和语义图保持一致）
            if f_mask_np.shape != unc_np.shape:
                f_mask_np = f_mask_np.T
                rospy.logwarn(f"Uncertainty mask维度({f_mask_np.shape})自动转置为{unc_np.shape}")

        # ========== 3. 完全复刻语义发布器的 max_area 过滤逻辑 ==========
        frame_id = "map"
        raw_marker = self._create_marker("raw_unc", frame_id, res, height)
        aligned_marker = self._create_marker("aligned_unc", frame_id, res, height + 0.01)

        # 核心修复：复用语义发布器的 fill_marker 逻辑（仅修改颜色映射）
        def fill_unc_marker(target_marker, unc_map, use_free_mask=False):
            min_x, max_x, min_y, max_y = max_area
            # 遍历所有栅格（和语义发布器一致：逐点判断，而非先筛选再过滤）
            for y_idx in range(unc_map.shape[0]):
                for x_idx in range(unc_map.shape[1]):
                    # 1. 不确定性阈值过滤（threshold=0 时不生效）
                    if unc_map[y_idx, x_idx] <= threshold:
                        continue

                    # 2. 计算物理坐标（和语义发布器完全一致）
                    px = x_idx * res + origin_x
                    py = y_idx * res + origin_y

                    # 3. max_area 过滤（和语义发布器完全一致）
                    if not (min_x <= px <= max_x and min_y <= py <= max_y):
                        continue

                    # 4. Free Mask 过滤（仅 aligned_marker 使用）
                    if use_free_mask and f_mask_np is not None:
                        # 确保 mask 索引和栅格索引一致
                        if not f_mask_np[y_idx, x_idx]:
                            continue

                    # 5. 颜色映射（保留）
                    vmax = np.max(unc_map) if np.max(unc_map) > 0 else 1.0
                    norm_val = np.clip(unc_map[y_idx, x_idx] / vmax, 0.0, 1.0)
                    color_rgba = self.cmap(norm_val)

                    # 6. 添加点（和语义发布器一致）
                    point = Point(px, py, height if not use_free_mask else height + 0.01)
                    color = ColorRGBA(color_rgba[0], color_rgba[1], color_rgba[2], 0.8)
                    target_marker.points.append(point)
                    target_marker.colors.append(color)

        # ========== 4. 执行填充和发布 ==========
        # 填充 raw_unc（仅 max_area 过滤，无 free_mask）
        fill_unc_marker(raw_marker, unc_np, use_free_mask=False)
        # 填充 aligned_unc（max_area + free_mask 过滤）
        fill_unc_marker(aligned_marker, unc_np, use_free_mask=True)

        # 空值处理（避免 RViz 报错）
        if len(raw_marker.points) == 0:
            raw_marker.action = Marker.DELETE
        if len(aligned_marker.points) == 0:
            aligned_marker.action = Marker.DELETE

        # 发布
        self.pub_raw.publish(raw_marker)
        self.pub_filtered.publish(aligned_marker)