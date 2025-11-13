#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import yaml
import cv2
import numpy as np
import torch
import importlib
from types import SimpleNamespace
from pathlib import Path
from collections import deque

# ------------------------
# 工具
# ------------------------
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def to_namespace(d):
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, to_namespace(v) if isinstance(v, dict) else v)
    return ns

def letterbox(img, new_shape=(480, 480)):
    h0, w0 = img.shape[:2]
    r = min(new_shape[0] / h0, new_shape[1] / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    return img, r, (dw, dh)

def torch_from_image(img):
    if img.ndim == 2:
        t = torch.from_numpy(img)[None, None]
    else:
        t = torch.from_numpy(img[:,:,::-1].transpose(2,0,1))[None]
    return t.float() / 255.0

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

# ------------------------
# 数据源
# ------------------------
class FolderStream:
    def __init__(self, rgb_dir, depth_dir, ext_rgb=".png", ext_d=".npy", loop=True):
        self.rgbs = sorted([str(x) for x in Path(rgb_dir).glob(f"*{ext_rgb}")])
        self.depths = sorted([str(x) for x in Path(depth_dir).glob(f"*{ext_d}")])
        self.i = 0
        self.loop = loop
        if len(self.rgbs) == 0 or len(self.depths) == 0:
            raise RuntimeError("folder inputs empty")
        if len(self.rgbs) != len(self.depths):
            n = min(len(self.rgbs), len(self.depths))
            self.rgbs, self.depths = self.rgbs[:n], self.depths[:n]

    def read(self):
        if self.i >= len(self.rgbs):
            if self.loop:
                self.i = 0
            else:
                return False, (None, None)
        rgb = cv2.imread(self.rgbs[self.i], cv2.IMREAD_COLOR)
        d = np.load(self.depths[self.i])
        self.i += 1
        return True, (rgb, d)

class CameraStream:
    def __init__(self, cam_idx=0, depth_folder=None):
        self.cap = cv2.VideoCapture(cam_idx)
        self.depth_folder = depth_folder
        self.i = 0
        if not self.cap.isOpened():
            raise RuntimeError("camera open failed")

    def read(self):
        ok, rgb = self.cap.read()
        if not ok:
            return False, (None, None)
        d = None
        if self.depth_folder:
            depth_path = os.path.join(self.depth_folder, f"{self.i:06d}.npy")
            if os.path.exists(depth_path):
                d = np.load(depth_path)
            else:
                d = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
        else:
            d = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
        self.i += 1
        return True, (rgb, d)

# ------------------------
# 动态加载模型
# ------------------------
def dynamic_import(module_path, class_name):
    m = importlib.import_module(module_path)
    cls = getattr(m, class_name)
    return cls

def load_torch_module(ckpt_path, map_location="cpu"):
    obj = torch.load(ckpt_path, map_location=map_location)
    return obj

class L2MWrapper(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.device)
        if cfg.entry.type == "jit":
            self.net = torch.jit.load(cfg.entry.path, map_location=self.device)
        elif cfg.entry.type == "torch":
            self.net = load_torch_module(cfg.entry.path, map_location=self.device)
        else:
            Net = dynamic_import(cfg.entry.module, cfg.entry.class_name)
            self.net = Net(cfg.entry.extra if hasattr(cfg.entry, "extra") else None)
            if hasattr(cfg, "weights") and cfg.weights:
                state = torch.load(cfg.weights, map_location=self.device)
                if "state_dict" in state: state = state["state_dict"]
                self.net.load_state_dict(state, strict=False)
        self.net.to(self.device).eval()
        self.in_size = (cfg.input.h, cfg.input.w)

    @torch.no_grad()
    def forward(self, rgb, depth):
        rgb = letterbox(rgb, self.in_size)[0]
        depth = letterbox(depth, self.in_size)[0]
        if depth.ndim == 3: depth = depth[...,0]
        r = torch_from_image(rgb).to(self.device)                # [1,3,H,W]
        d = torch.from_numpy(depth).float()[None,None].to(self.device)  # [1,1,H,W]
        out = self.net(r, d)                                     # 期望返回特征或语义图
        return out

class RSMPNetWrapper(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.device)
        if cfg.entry.type == "jit":
            self.net = torch.jit.load(cfg.entry.path, map_location=self.device)
        elif cfg.entry.type == "torch":
            self.net = load_torch_module(cfg.entry.path, map_location=self.device)
        else:
            Net = dynamic_import(cfg.entry.module, cfg.entry.class_name)
            self.net = Net(cfg.entry.extra if hasattr(cfg.entry, "extra") else None)
            if hasattr(cfg, "weights") and cfg.weights:
                state = torch.load(cfg.weights, map_location=self.device)
                if "state_dict" in state: state = state["state_dict"]
                self.net.load_state_dict(state, strict=False)
        self.net.to(self.device).eval()
        self.in_size = (cfg.input.h, cfg.input.w)

    @torch.no_grad()
    def forward(self, l2m_out, rgb=None, depth=None):
        if isinstance(l2m_out, torch.Tensor):
            feat = l2m_out
        elif isinstance(l2m_out, dict) and "feat" in l2m_out:
            feat = l2m_out["feat"]
        else:
            raise RuntimeError("unsupported l2m output")
        feat = feat.to(self.device)
        out = self.net(feat)                                     # 期望输出语义概率或栅格
        return out

# ------------------------
# 可视化
# ------------------------
def colorize_logits(logits, palette=None):
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().float()
        if logits.dim() == 4: logits = logits[0]
        if logits.shape[0] > 1:
            pred = torch.argmax(logits, dim=0).numpy().astype(np.uint8)
        else:
            pred = (logits[0] > 0.5).numpy().astype(np.uint8)
    else:
        pred = logits
    if palette is None:
        palette = np.array([
            [0,0,0], [128,64,128], [244,35,232], [70,70,70],
            [102,102,156], [190,153,153], [153,153,153], [250,170,30],
            [220,220, 0], [107,142, 35], [152,251,152], [70,130,180],
            [220, 20, 60], [255, 0, 0], [ 0, 0,142], [0, 0,70]
        ], dtype=np.uint8)
    h, w = pred.shape[:2]
    c = np.zeros((h,w,3), dtype=np.uint8)
    classes = np.unique(pred)
    for k in classes:
        c[pred==k] = palette[k % len(palette)]
    return c

# ------------------------
# 主流程
# ------------------------
class IntegratedRunner:
    def __init__(self, cfg_path):
        self.cfg = to_namespace(load_yaml(cfg_path))
        self.device = torch.device(self.cfg.device)

        if self.cfg.input.type == "folder":
            self.stream = FolderStream(self.cfg.input.rgb_dir, self.cfg.input.depth_dir,
                                       ext_rgb=self.cfg.input.rgb_ext, ext_d=self.cfg.input.depth_ext,
                                       loop=self.cfg.input.loop)
        elif self.cfg.input.type == "camera":
            self.stream = CameraStream(self.cfg.input.cam_index, depth_folder=self.cfg.input.depth_folder)
        else:
            raise RuntimeError("unknown input.type")

        self.l2m = L2MWrapper(self.cfg.l2m)
        self.rsmp = RSMPNetWrapper(self.cfg.rsmp)
        self.show = self.cfg.viz.show
        self.save_dir = self.cfg.viz.save_dir
        self.fps_limit = self.cfg.runtime.fps_limit
        if self.save_dir: ensure_dir(self.save_dir)
        self.tq = deque(maxlen=60)

    def step(self, rgb, depth):
        l2m_out = self.l2m(rgb, depth)
        rsmp_out = self.rsmp(l2m_out, rgb=rgb, depth=depth)
        return l2m_out, rsmp_out

    def run(self):
        win = "integrated"
        if self.show: cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        idx = 0
        while True:
            t0 = time.time()
            ok, (rgb, depth) = self.stream.read()
            if not ok: break
            l2m_out, rsmp_out = self.step(rgb, depth)
            vis = colorize_logits(rsmp_out)
            vis = cv2.resize(vis, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            overlay = (0.6 * rgb + 0.4 * vis).astype(np.uint8)
            t1 = time.time()
            self.tq.append(1.0 / max(1e-6, (t1 - t0)))
            fps = sum(self.tq) / len(self.tq)
            cv2.putText(overlay, f"FPS: {fps:.1f}", (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
            if self.show:
                cv2.imshow(win, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'): break
            if self.save_dir:
                cv2.imwrite(os.path.join(self.save_dir, f"frame_{idx:06d}.png"), overlay)
            idx += 1
            if self.fps_limit > 0:
                stay = max(0.0, 1.0/self.fps_limit - (time.time()-t0))
                if stay > 1e-3: time.sleep(stay)
        if self.show:
            try: cv2.destroyAllWindows()
            except: pass

# ------------------------
# 入口
# ------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python run_integrated.py configs/integrated.yaml")
        sys.exit(1)
    runner = IntegratedRunner(sys.argv[1])
    runner.run()

if __name__ == "__main__":
    main()
