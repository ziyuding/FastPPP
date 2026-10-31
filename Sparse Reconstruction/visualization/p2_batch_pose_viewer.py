#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''p2_batch_pose_viewer.py — 稀疏相机位姿与双环轨迹批量交互质检工具
- 功能: 批量遍历 COLMAP 稀疏重建结果 (相机视锥体 + 稀疏特征点云)
- 特性:
• 支持快捷键一键切盆 (下一盆/上一盆)'''

from __future__ import annotations

import os
import re
import sys
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def natural_sort_key(p: Path):
    m = re.match(r"^(\d+)", p.name)
    return (int(m.group(1)) if m else 99999, p.name)


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    '''四元数 → 旋转矩阵 (COLMAP: q = [qw, qx, qy, qz])'''
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2],
    ])


def parse_sparse_data(sample_dir: Path):
    '''解析 cameras.txt, images.txt, points3D.txt'''
    cameras_txt = sample_dir / "cameras.txt"
    images_txt = sample_dir / "images.txt"
    points_txt = sample_dir / "points3D.txt"

    if not (cameras_txt.exists() and images_txt.exists() and points_txt.exists()):
        return None, None, None, None

    cameras = {}
    with open(cameras_txt, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            w, h = int(parts[2]), int(parts[3])
            params = list(map(float, parts[4:]))
            cameras[cam_id] = {"model": model, "width": w, "height": h, "params": params}

    images = []
    with open(images_txt, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        if len(parts) >= 10:
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            cam_id = int(parts[8])
            name = parts[9]
            images.append({
                "image_id": image_id,
                "qvec": np.array([qw, qx, qy, qz]),
                "tvec": np.array([tx, ty, tz]),
                "camera_id": cam_id,
                "name": name,
            })
        i += 2

    pts = []
    colors = []
    with open(points_txt, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 7:
                x, y, z = map(float, parts[1:4])
                r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
                pts.append([x, y, z])
                colors.append([r / 255.0, g / 255.0, b / 255.0])

    pts_np = np.array(pts, dtype=np.float64) if pts else np.empty((0, 3))
    colors_np = np.array(colors, dtype=np.float64) if colors else np.empty((0, 3))

    return cameras, images, pts_np, colors_np


def build_frustums_lineset(cameras, images, scale=0.05, every_n_cam=1):
    '''构建所有相机的视锥体单合并 LineSet (高低环双色区分)'''
    all_points = []
    all_lines = []
    all_colors = []

    COLOR_HIGH = [0.15, 0.85, 1.0]
    COLOR_LOW = [1.0, 0.35, 0.1]
    COLOR_DEFAULT = [0.9, 0.2, 0.2]

    v_offset = 0
    for idx, img in enumerate(images):
        if idx % every_n_cam != 0:
            continue
        cam = cameras.get(img["camera_id"])
        if not cam:
            continue

        w, h = cam["width"], cam["height"]
        params = cam["params"]
        f, cx, cy = params[0], params[1], params[2] if len(params) >= 3 else (max(w, h), w / 2, h / 2)

        corners_2d = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        corners_norm = (corners_2d - np.array([cx, cy])) / f
        corners_3d = np.hstack([corners_norm, np.ones((4, 1))]) * scale

        R = qvec2rotmat(img["qvec"])
        t = img["tvec"]
        center = -R.T @ t
        corners_world = (R.T @ corners_3d.T).T + center

        pts = np.vstack([center.reshape(1, 3), corners_world])
        lines = [
            [v_offset + 0, v_offset + 1],
            [v_offset + 0, v_offset + 2],
            [v_offset + 0, v_offset + 3],
            [v_offset + 0, v_offset + 4],
            [v_offset + 1, v_offset + 2],
            [v_offset + 2, v_offset + 3],
            [v_offset + 3, v_offset + 4],
            [v_offset + 4, v_offset + 1],
        ]

        name = img["name"].lower()
        if "high" in name or "cam1" in name or "cam_high" in name:
            col = COLOR_HIGH
        elif "low" in name or "cam2" in name or "cam_low" in name:
            col = COLOR_LOW
        else:
            col = COLOR_DEFAULT

        all_points.append(pts)
        all_lines.extend(lines)
        all_colors.extend([col] * 8)
        v_offset += 5

    line_set = o3d.geometry.LineSet()
    if all_points:
        line_set.points = o3d.utility.Vector3dVector(np.vstack(all_points))
        line_set.lines = o3d.utility.Vector2iVector(all_lines)
        line_set.colors = o3d.utility.Vector3dVector(all_colors)
    else:
        line_set.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
        line_set.lines = o3d.utility.Vector2iVector(np.empty((0, 2), dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(np.empty((0, 3)))

    return line_set


class BatchPoseViewer:
    def __init__(self, sample_dirs: list[Path], initial_idx: int = 0, frustum_scale: float = 0.05, every_n_cam: int = 1):
        self.sample_dirs = sample_dirs
        self.total = len(sample_dirs)
        self.idx = max(0, min(initial_idx, self.total - 1))
        self.frustum_scale = frustum_scale
        self.every_n_cam = every_n_cam

        self.pcd = o3d.geometry.PointCloud()
        self.frustums = o3d.geometry.LineSet()
        self.coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frustum_scale * 4)

    def load_sample_geometry(self, s_dir: Path):
        cameras, images, pts, colors = parse_sparse_data(s_dir)
        if cameras is None:
            return 0, 0, False

        self.pcd.points = o3d.utility.Vector3dVector(pts)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        ls = build_frustums_lineset(cameras, images, scale=self.frustum_scale, every_n_cam=self.every_n_cam)
        self.frustums.points = ls.points
        self.frustums.lines = ls.lines
        self.frustums.colors = ls.colors

        n_cams = len(images)
        n_pts = len(pts)
        return n_cams, n_pts, True

    def update_sample(self, vis, idx: int, reset_view: bool = False):
        self.idx = (idx + self.total) % self.total
        s_dir = self.sample_dirs[self.idx]

        n_cams, n_pts, ok = self.load_sample_geometry(s_dir)
        vis.update_geometry(self.pcd)
        vis.update_geometry(self.frustums)

        if reset_view:
            vis.reset_view_point(True)

        print("\n" + "=" * 72)
        print(f"  📷 稀疏位姿质检 [{self.idx + 1}/{self.total}] : {s_dir.name}")
        if ok:
            print(f"  • 注册相机: {n_cams} 张  (🔵 高位相机环: 青色  |  🟠 低位相机环: 橙红色)")
            print(f"  • 稀疏特征点: {n_pts:,} 个")
        else:
            print(f"  • ⚠️ 读取失败或缺少 sparse_txt 文件")
        print(f"  • 导航: [D / →] 下一盆  |  [A / ←] 上一盆  |  [B] 背景  |  [= / -] 点大小  |  [R] 重置视角  |  [Q] 退出")
        print("=" * 72)

    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="COLMAP Sparse Poses Batch Viewer", width=1600, height=900)

        first_dir = self.sample_dirs[self.idx]
        self.load_sample_geometry(first_dir)
        vis.add_geometry(self.pcd)
        vis.add_geometry(self.frustums)
        vis.add_geometry(self.coord)

        opt = vis.get_render_option()
        opt.point_size = 1.5
        opt.background_color = [0.05, 0.05, 0.05]
        opt.light_on = True

        state = {"left": False, "right": False, "px": None, "py": None}

        def _on_button(v, btn, act, mods):
            if btn == 0:
                state["left"] = act == 1
            elif btn == 1:
                state["right"] = act == 1
            if act == 1:
                state["px"] = state["py"] = None
            return False

        def _on_move(v, x, y):
            if state["px"] is not None:
                dx, dy = x - state["px"], y - state["py"]
                ctrl = v.get_view_control()
                if state["left"]:
                    ctrl.rotate(dx, dy)
                elif state["right"]:
                    ctrl.translate(dx, dy)
            state["px"], state["py"] = x, y
            return state["left"] or state["right"]

        vis.register_mouse_button_callback(_on_button)
        vis.register_mouse_move_callback(_on_move)

        def _next_sample(v):
            self.update_sample(v, self.idx + 1)

        def _prev_sample(v):
            self.update_sample(v, self.idx - 1)

        bg_colors = [
            np.array([0.05, 0.05, 0.05]),
            np.array([1.0, 1.0, 1.0]),
            np.array([0.2, 0.2, 0.25]),
        ]
        bg_idx = 0

        def _toggle_bg(v):
            nonlocal bg_idx
            bg_idx = (bg_idx + 1) % len(bg_colors)
            v.get_render_option().background_color = bg_colors[bg_idx]
            print(f"[背景切换] → {['深黑', '纯白', '蓝灰'][bg_idx]}")

        def _point_bigger(v):
            o = v.get_render_option()
            o.point_size = min(20.0, o.point_size + 1.0)
            print(f"[点大小] → {o.point_size}")

        def _point_smaller(v):
            o = v.get_render_option()
            o.point_size = max(1.0, o.point_size - 1.0)
            print(f"[点大小] → {o.point_size}")

        def _reset_view(v):
            v.reset_view_point(True)
            print("[视角] 已重置")

        def _screenshot(v):
            s_name = self.sample_dirs[self.idx].name
            out_p = REPO_ROOT / f"pose_{s_name}.png"
            v.capture_screen_image(str(out_p))
            print(f"📸 截图已保存至: {out_p}")

        def _quit(v):
            v.destroy_window()
            sys.exit(0)

        for key in [ord("D"), ord("N"), 262]:
            vis.register_key_callback(key, _next_sample)
        for key in [ord("A"), ord("P"), 263]:
            vis.register_key_callback(key, _prev_sample)

        vis.register_key_callback(ord("B"), _toggle_bg)
        vis.register_key_callback(ord("="), _point_bigger)
        vis.register_key_callback(ord("+"), _point_bigger)
        vis.register_key_callback(ord("-"), _point_smaller)
        vis.register_key_callback(ord("_"), _point_smaller)
        vis.register_key_callback(ord("R"), _reset_view)
        vis.register_key_callback(ord("S"), _screenshot)
        vis.register_key_callback(ord("F"), _screenshot)
        for key in [ord("Q"), 256]:
            vis.register_key_callback(key, _quit)

        print("\n" + "=" * 72)
        print(f"  📷 稀疏位姿批量质检工具已就绪 (共 {self.total} 盆)")
        print(f"  • 当前样本: {first_dir.name}")
        print(f"  • 导航: [D / →] 下一盆  |  [A / ←] 上一盆  |  [B] 背景  |  [= / -] 点大小  |  [R] 重设视角  |  [Q] 退出")
        print("=" * 72)

        vis.run()
        vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="批量检查稀疏相机位姿双环轨迹")
    parser.add_argument("--day", type=str, default="Day3_20260916", help="数据集日期，如 Day3_20260916, Day1_20260914, Day0_20260912")
    parser.add_argument("--dir", type=str, default=None, help="稀疏位姿目录路径 (默认为 /data_ssd/ddd/Dataset/fqxGrass_pcds/<day>/sparse_poses)")
    parser.add_argument("--samples", nargs="+", default=None, help="指定一个或多个样本名称进行检查 (如 104_21-B 164_253-B 218_176)")
    parser.add_argument("--start", type=str, default=None, help="起始样本编号或名称 (如 104_21-B 或 104)")
    parser.add_argument("--frustum-scale", type=float, default=0.05, help="相机视锥体尺寸 (默认 0.05)")
    parser.add_argument("--every-n-cam", type=int, default=1, help="视锥体采样间隔 (默认 1 表示显示全部相机)")
    args = parser.parse_args()

    if args.dir:
        base_dir = Path(args.dir)
    else:
        base_dir = Path(f"/data_ssd/ddd/Dataset/fqxGrass_pcds/{args.day}/sparse_poses")

    if not base_dir.exists():
        print(f"错误: 目录不存在: {base_dir}", file=sys.stderr)
        sys.exit(1)

    all_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    all_dirs.sort(key=natural_sort_key)

    if not all_dirs:
        print(f"错误: 目录下未发现任何稀疏位姿文件夹: {base_dir}", file=sys.stderr)
        sys.exit(1)

    if args.samples:
        target_set = set(args.samples)
        filtered = [d for d in all_dirs if any(tgt in d.name for tgt in target_set)]
        if not filtered:
            print(f"错误: 指定样本 {args.samples} 未匹配到任何目录", file=sys.stderr)
            sys.exit(1)
        all_dirs = filtered

    start_idx = 0
    if args.start:
        for i, d in enumerate(all_dirs):
            if args.start == d.name or d.name.startswith(f"{args.start}_") or args.start in d.name:
                start_idx = i
                break

    viewer = BatchPoseViewer(
        sample_dirs=all_dirs,
        initial_idx=start_idx,
        frustum_scale=args.frustum_scale,
        every_n_cam=args.every_n_cam,
    )
    viewer.run()


if __name__ == "__main__":
    main()
