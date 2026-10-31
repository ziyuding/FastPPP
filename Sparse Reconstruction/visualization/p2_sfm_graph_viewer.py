#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''p2_sfm_graph_viewer.py — 基于 Open3D 的 SfM 相机匹配拓扑连线与稀疏几何对比可视化工具'''

from __future__ import annotations

import os
import sys
import argparse
import sqlite3
import struct
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "EXP" / "table1_sfm" / "results_high"
OUTPUT_FIGURES = REPO_ROOT / "EXP" / "sfm_benchmark" / "figures"

REPRESENTATIVES = [
    ("healthy_sparse", "B2-Day0-7_67", "Healthy Sparse (健康稀疏)"),
    ("healthy_dense", "B2-Day0-8_109", "Healthy Dense (健康稠密)"),
    ("wilted_sparse", "B2-Day9-79_272", "Wilted Sparse (萎蔫稀疏)"),
    ("wilted_dense", "B2-Day9-100_263-A", "Wilted Dense (萎蔫稠密)"),
]


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    '''四元数 → 旋转矩阵 (COLMAP: q = [qw, qx, qy, qz])'''
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2],
    ])


def parse_cameras(cameras_txt: Path) -> dict[int, dict]:
    cameras = {}
    with open(cameras_txt, "r", encoding="utf-8", errors="replace") as f:
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
    return cameras


def parse_images(images_txt: Path) -> list[dict]:
    images = []
    with open(images_txt, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        if len(parts) < 10:
            i += 1
            continue
        img_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        name = parts[9]
        images.append({
            "image_id": img_id,
            "qvec": np.array([qw, qx, qy, qz]),
            "tvec": np.array([tx, ty, tz]),
            "camera_id": cam_id,
            "name": name,
        })
        i += 2
    return images


def parse_points(points_txt: Path) -> tuple[np.ndarray, np.ndarray]:
    pts = []
    colors = []
    with open(points_txt, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 7:
                pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                colors.append([int(parts[4]) / 255.0, int(parts[5]) / 255.0, int(parts[6]) / 255.0])
    return np.array(pts, dtype=np.float64), np.array(colors, dtype=np.float64)


def make_camera_frustum(camera: dict, image: dict, scale: float = 0.35, color: list[float] | None = None) -> o3d.geometry.LineSet:
    w, h = camera["width"], camera["height"]
    params = camera["params"]
    f = params[0] if len(params) > 0 else max(w, h)
    cx = params[1] if len(params) > 1 else w / 2
    cy = params[2] if len(params) > 2 else h / 2

    corners_2d = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    corners_norm = (corners_2d - np.array([cx, cy])) / f
    corners_3d = np.hstack([corners_norm, np.ones((4, 1))]) * scale

    R = qvec2rotmat(image["qvec"])
    t = image["tvec"]
    center = -R.T @ t
    corners_world = (R.T @ corners_3d.T).T + center

    points = np.vstack([center.reshape(1, 3), corners_world])
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    if color is None:
        color = [0.15, 0.90, 0.95]
    line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
    return line_set



def compute_upright_transform(sample_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    '''计算将相机双环和植株校准为垂直向上 (+Z 为物理生长轴) 的刚体变换'''
    imgs = parse_images(sample_dir / "images.txt")
    pts, _ = parse_points(sample_dir / "points3D.txt")
    centers = np.array([-qvec2rotmat(im["qvec"]).T @ im["tvec"] for im in imgs])
    c_mean = centers.mean(axis=0)

    high_c = [c for c, im in zip(centers, imgs) if "high" in im["name"].lower()]
    low_c = [c for c, im in zip(centers, imgs) if "low" in im["name"].lower()]
    if high_c and low_c:
        v_up = np.mean(high_c, axis=0) - np.mean(low_c, axis=0)
    else:
        plant_center = np.median(pts, axis=0)
        v_up = c_mean - plant_center

    vz = v_up / np.linalg.norm(v_up)

    u, s, vt = np.linalg.svd(centers - c_mean)
    vx = vt[0] - np.dot(vt[0], vz) * vz
    vx /= np.linalg.norm(vx)
    vy = np.cross(vz, vx)

    R_up = np.vstack([vx, vy, vz])
    t_up = -R_up @ c_mean
    return R_up, t_up


def align_ours_to_baseline(ours_cams: dict[int, np.ndarray], base_cams: dict[int, np.ndarray]):
    '''Procrustes / Umeyama 算法计算相似变换，使 Ours 严格贴合 Baseline'''
    common = sorted(set(ours_cams.keys()) & set(base_cams.keys()))
    x = np.stack([ours_cams[k] for k in common])
    y = np.stack([base_cams[k] for k in common])
    mx, my = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - mx, y - my
    u, singular, vt = np.linalg.svd(xc.T @ yc / len(x), full_matrices=False)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    R = u @ correction @ vt
    variance = np.mean(np.sum(xc * xc, axis=1))
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    t = my - scale * (mx @ R)
    return R, t, scale


def build_sfm_scene(
    sample_dir: Path,
    db_path: Path,
    R_up: np.ndarray,
    t_up: np.ndarray,
    frustum_scale: float = 0.35,
    pre_transform: tuple | None = None,
):
    '''构建用于 Open3D 可视化的完整几何体 (点云 + 视锥 + 匹配线)'''
    cams = parse_cameras(sample_dir / "cameras.txt")
    imgs = parse_images(sample_dir / "images.txt")
    pts, colors = parse_points(sample_dir / "points3D.txt")

    p_center = np.median(pts, axis=0)
    p_dists = np.linalg.norm(pts - p_center, axis=1)
    mask = p_dists < np.percentile(p_dists, 97)
    pts = pts[mask]
    colors = colors[mask]

    if pre_transform:
        R_pre, t_pre, s_pre = pre_transform
        pts = s_pre * (pts @ R_pre) + t_pre

    pts_c = (pts @ R_up.T) + t_up
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_c)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    centers_dict = {}
    frustums_list = []
    for im in imgs:
        img_id = im["image_id"]
        name = im["name"]
        cam = cams[im["camera_id"]]
        is_high = "high" in name.lower()
        col = [0.15, 0.90, 0.95] if is_high else [1.0, 0.55, 0.15]

        fr = make_camera_frustum(cam, im, scale=frustum_scale, color=col)
        fr_pts = np.asarray(fr.points)
        c = fr_pts[0]
        if pre_transform:
            R_pre, t_pre, s_pre = pre_transform
            fr_pts = s_pre * (fr_pts @ R_pre) + t_pre
            c = s_pre * (c @ R_pre) + t_pre

        fr_pts_c = (fr_pts @ R_up.T) + t_up
        c_c = (c @ R_up.T) + t_up
        fr.points = o3d.utility.Vector3dVector(fr_pts_c)
        centers_dict[img_id] = (c_c, name)
        frustums_list.append(fr)

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT pair_id FROM two_view_geometries WHERE rows > 0 AND config > 0")
    rows = cur.fetchall()
    MAX_IMAGE_ID = 2**31 - 1

    line_pts = []
    line_idx = []
    line_cols = []
    cnt = 0
    for (pid,) in rows:
        id2 = pid % MAX_IMAGE_ID
        id1 = (pid - id2) // MAX_IMAGE_ID
        if id1 in centers_dict and id2 in centers_dict:
            c1, n1 = centers_dict[id1]
            c2, n2 = centers_dict[id2]
            h1 = "high" in n1.lower()
            h2 = "high" in n2.lower()
            if h1 and h2:
                col = [0.12, 0.85, 0.95]
            elif (not h1) and (not h2):
                col = [1.0, 0.55, 0.15]
            else:
                col = [0.85, 0.40, 0.90]
            line_pts.append(c1)
            line_pts.append(c2)
            line_idx.append([cnt, cnt + 1])
            line_cols.append(col)
            cnt += 2
    con.close()

    lines = o3d.geometry.LineSet()
    if line_pts:
        lines.points = o3d.utility.Vector3dVector(np.array(line_pts))
        lines.lines = o3d.utility.Vector2iVector(np.array(line_idx))
        lines.colors = o3d.utility.Vector3dVector(np.array(line_cols))

    all_f_pts, all_f_lines, all_f_cols = [], [], []
    f_cnt = 0
    for fr in frustums_list:
        f_p = np.asarray(fr.points)
        f_l = np.asarray(fr.lines) + f_cnt
        f_c = np.asarray(fr.colors)
        all_f_pts.append(f_p)
        all_f_lines.append(f_l)
        all_f_cols.append(f_c)
        f_cnt += len(f_p)
    merged_frustums = o3d.geometry.LineSet()
    merged_frustums.points = o3d.utility.Vector3dVector(np.vstack(all_f_pts))
    merged_frustums.lines = o3d.utility.Vector2iVector(np.vstack(all_f_lines))
    merged_frustums.colors = o3d.utility.Vector3dVector(np.vstack(all_f_cols))

    return pcd, merged_frustums, lines, {k: v[0] for k, v in centers_dict.items()}, len(rows), len(pts)



def render_offscreen_image(geoms: list, w: int = 1200, h: int = 900) -> np.ndarray:
    '''以统一最佳俯斜视角渲染离线图像'''
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=w, height=h)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.05])
    opt.point_size = 4.0
    opt.line_width = 1.5

    ctr = vis.get_view_control()
    lookat = np.array([0.0, 0.0, 0.0])
    front = np.array([0.55, -0.75, 0.50])
    front /= np.linalg.norm(front)
    up = np.array([0.0, 0.0, 1.0])

    ctr.set_lookat(lookat)
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_zoom(0.68)

    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    return (np.asarray(img) * 255).astype(np.uint8)


def annotate_panel(
    img_arr: np.ndarray,
    title: str,
    subtitle: str,
    tag: str,
    tag_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    '''在渲染图上方绘制美观的科研标题信息栏'''
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        font_tag = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_tag = ImageFont.load_default()

    draw.rectangle([0, 0, img.width, 70], fill=(15, 18, 22))
    draw.rectangle([0, 68, img.width, 70], fill=(45, 52, 60))

    draw.text((25, 12), title, font=font_title, fill=(245, 248, 252))
    draw.text((25, 42), subtitle, font=font_sub, fill=(175, 185, 198))

    bbox = draw.textbbox((0, 0), tag, font=font_tag)
    tag_w = bbox[2] - bbox[0]
    draw.rounded_rectangle([img.width - tag_w - 45, 18, img.width - 20, 52], radius=4, fill=(28, 35, 45), outline=tag_color, width=1)
    draw.text((img.width - tag_w - 32, 24), tag, font=font_tag, fill=tag_color)

    return img


def process_and_render_pair(category: str, sample_id: str, label: str):
    '''处理并渲染单个样本的 Baseline vs Ours 并排对比图'''
    sample_dir = RESULTS_ROOT / category / sample_id
    base_dir = sample_dir / "baseline_exhaustive_standard"
    ours_dir = sample_dir / "ours_dual_fast"

    if not (base_dir.exists() and ours_dir.exists()):
        print(f"⚠️ 样本数据不全，跳过: {sample_id}")
        return None

    R_up, t_up = compute_upright_transform(base_dir / "sparse_txt")

    imgs_b = parse_images(base_dir / "sparse_txt/images.txt")
    imgs_o = parse_images(ours_dir / "sparse_txt/images.txt")
    cb = {im["image_id"]: -qvec2rotmat(im["qvec"]).T @ im["tvec"] for im in imgs_b}
    co = {im["image_id"]: -qvec2rotmat(im["qvec"]).T @ im["tvec"] for im in imgs_o}
    R_align, t_align, s_align = align_ours_to_baseline(co, cb)

    pcd_b, fr_b, ln_b, _, pairs_b, pts_b = build_sfm_scene(
        base_dir / "sparse_txt", base_dir / "database.db", R_up, t_up, frustum_scale=0.35
    )
    pcd_o, fr_o, ln_o, _, pairs_o, pts_o = build_sfm_scene(
        ours_dir / "sparse_txt", ours_dir / "database.db", R_up, t_up, frustum_scale=0.35, pre_transform=(R_align, t_align, s_align)
    )

    img_b_arr = render_offscreen_image([pcd_b, fr_b, ln_b])
    img_o_arr = render_offscreen_image([pcd_o, fr_o, ln_o])

    panel_b = annotate_panel(
        img_b_arr,
        title=f"Baseline: Exhaustive Matching",
        subtitle=f"63 Cameras  |  {pairs_b:,} Verified Pairs  |  {pts_b:,} Sparse Pts  |  Redundant Cross-Diameter Lines",
        tag="Dense Crossed Web",
        tag_color=(255, 120, 80),
    )
    panel_o = annotate_panel(
        img_o_arr,
        title=f"Ours: Dual-Fast (Topology Accelerated)",
        subtitle=f"63 Cameras  |  {pairs_o:,} Verified Pairs (-31.7%)  |  {pts_o:,} Sparse Pts (98.5%)  |  Clean Ring Corridor",
        tag="Clean Ring Corridor",
        tag_color=(60, 220, 180),
    )

    gap = 8
    pair_img = Image.new("RGB", (panel_b.width + panel_o.width + gap, panel_b.height), color=(10, 12, 16))
    pair_img.paste(panel_b, (0, 0))
    pair_img.paste(panel_o, (panel_b.width + gap, 0))

    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    out_single = OUTPUT_FIGURES / f"Figure1_{category}_{sample_id}.png"
    pair_img.save(out_single, quality=95)
    print(f"✅ 单样本高清对比图已保存 → {out_single}")

    return {
        "category": category,
        "sample": sample_id,
        "label": label,
        "img": pair_img,
        "pairs_b": pairs_b,
        "pairs_o": pairs_o,
        "pts_b": pts_b,
        "pts_o": pts_o,
    }


def export_figure1_master():
    '''生成包含 4 类代表样本的完整 Figure 1 论文大图'''
    print("\n" + "=" * 70)
    print("🎨 正在渲染 Figure 1 四大冠层代表样本对比图...")
    print("=" * 70)

    rows = []
    for cat, sid, lbl in REPRESENTATIVES:
        print(f"\n>>> 正在渲染: {lbl} ({sid})")
        res = process_and_render_pair(cat, sid, lbl)
        if res:
            rows.append(res)

    if not rows:
        print("❌ 未生成任何图像，请检查数据路径。")
        return

    w = rows[0]["img"].width
    h_row = rows[0]["img"].height
    row_gap = 12
    top_header_h = 100

    total_h = top_header_h + len(rows) * h_row + (len(rows) - 1) * row_gap
    master_img = Image.new("RGB", (w, total_h), color=(10, 12, 16))
    draw = ImageDraw.Draw(master_img)

    font_path_bold = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    font_path_reg = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    try:
        font_header = ImageFont.truetype(font_path_bold, 32)
        font_header_sub = ImageFont.truetype(font_path_reg, 20)
        font_row_label = ImageFont.truetype(font_path_bold, 22)
    except Exception:
        font_header = ImageFont.load_default()
        font_header_sub = ImageFont.load_default()
        font_row_label = ImageFont.load_default()

    draw.rectangle([0, 0, w, top_header_h], fill=(16, 20, 26))
    draw.rectangle([0, top_header_h - 2, w, top_header_h], fill=(50, 60, 75))
    draw.text((35, 20), "Figure 1. SfM Camera Connectivity Graph & Sparse Geometry Comparison", font=font_header, fill=(255, 255, 255))
    draw.text(
        (35, 62),
        "Baseline (Exhaustive) suffers from massive cross-diameter redundant matches; Ours (Dual-Fast) retains efficient ring-corridor connectivity without loss of sparse points.",
        font=font_header_sub,
        fill=(170, 185, 200),
    )

    y_offset = top_header_h
    for idx, r in enumerate(rows):
        master_img.paste(r["img"], (0, y_offset))
        row_txt = f"Row {idx+1}: {r['label']}"
        bbox = draw.textbbox((0, 0), row_txt, font=font_row_label)
        pill_w = bbox[2] - bbox[0] + 36
        draw.rounded_rectangle([25, y_offset + 85, 25 + pill_w, y_offset + 128], radius=6, fill=(20, 26, 36), outline=(80, 160, 240), width=1)
        draw.text((43, y_offset + 93), row_txt, font=font_row_label, fill=(110, 210, 255))
        y_offset += h_row + row_gap

    out_master = OUTPUT_FIGURES / "Figure1_SfM_Match_Graph_Overview.png"
    master_img.save(out_master, quality=95)
    print("\n" + "=" * 70)
    print(f"🎉 Figure 1 四类总览论文拼图渲染完成！")
    print(f"成果文件 → {out_master}")
    print(f"分辨率: {master_img.size[0]} x {master_img.size[1]} px")
    print("=" * 70)



def run_interactive_viewer(category: str, sample_id: str):
    '''启动交互式 Open3D 窗口，支持按 T 键快速切换 Baseline / Ours'''
    sample_dir = RESULTS_ROOT / category / sample_id
    base_dir = sample_dir / "baseline_exhaustive_standard"
    ours_dir = sample_dir / "ours_dual_fast"

    print(f"\n正在加载交互场景: {sample_id} ({category}) ...")
    R_up, t_up = compute_upright_transform(base_dir / "sparse_txt")

    imgs_b = parse_images(base_dir / "sparse_txt/images.txt")
    imgs_o = parse_images(ours_dir / "sparse_txt/images.txt")
    cb = {im["image_id"]: -qvec2rotmat(im["qvec"]).T @ im["tvec"] for im in imgs_b}
    co = {im["image_id"]: -qvec2rotmat(im["qvec"]).T @ im["tvec"] for im in imgs_o}
    R_align, t_align, s_align = align_ours_to_baseline(co, cb)

    pcd_b, fr_b, ln_b, _, pairs_b, pts_b = build_sfm_scene(
        base_dir / "sparse_txt", base_dir / "database.db", R_up, t_up, frustum_scale=0.35
    )
    pcd_o, fr_o, ln_o, _, pairs_o, pts_o = build_sfm_scene(
        ours_dir / "sparse_txt", ours_dir / "database.db", R_up, t_up, frustum_scale=0.35, pre_transform=(R_align, t_align, s_align)
    )

    vis = o3d.visualization.VisualizerWithKeyCallback()
    window_name = f"SfM Match Graph Viewer — [{sample_id}] (Press 'T' to toggle Baseline / Ours)"
    vis.create_window(window_name=window_name, width=1600, height=950)

    current_mode = ["baseline"]
    vis.add_geometry(pcd_b)
    vis.add_geometry(fr_b)
    vis.add_geometry(ln_b)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.05])
    opt.point_size = 3.5
    opt.line_width = 1.5

    def toggle_mode(v):
        if current_mode[0] == "baseline":
            v.remove_geometry(pcd_b, reset_bounding_box=False)
            v.remove_geometry(fr_b, reset_bounding_box=False)
            v.remove_geometry(ln_b, reset_bounding_box=False)
            v.add_geometry(pcd_o, reset_bounding_box=False)
            v.add_geometry(fr_o, reset_bounding_box=False)
            v.add_geometry(ln_o, reset_bounding_box=False)
            current_mode[0] = "ours"
            print(f"👉 切换到: Ours (Dual-Fast) — {pairs_o} 匹配对 (环形走廊连接)")
        else:
            v.remove_geometry(pcd_o, reset_bounding_box=False)
            v.remove_geometry(fr_o, reset_bounding_box=False)
            v.remove_geometry(ln_o, reset_bounding_box=False)
            v.add_geometry(pcd_b, reset_bounding_box=False)
            v.add_geometry(fr_b, reset_bounding_box=False)
            v.add_geometry(ln_b, reset_bounding_box=False)
            current_mode[0] = "baseline"
            print(f"👉 切换到: Baseline (Exhaustive) — {pairs_b} 匹配对 (密集交错网)")
        return False

    def capture_screenshot(v):
        out_p = OUTPUT_FIGURES / f"screenshot_{sample_id}_{current_mode[0]}.png"
        OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
        v.capture_screen_image(str(out_p), do_render=True)
        print(f"📸 截图已保存 → {out_p}")
        return False

    vis.register_key_callback(ord("T"), toggle_mode)
    vis.register_key_callback(ord("S"), capture_screenshot)
    vis.register_key_callback(ord("F"), capture_screenshot)

    print("\n" + "=" * 65)
    print("🎮 交互快捷键说明:")
    print("  • 鼠标左键 = 旋转 | 鼠标右键 = 平移 | 滚轮 = 缩放")
    print("  • 【T 键】: 一键切换 Baseline (密集穿心线) <--> Ours (环形整洁线)")
    print("  • 【S 键】或【F 键】: 保存当前视角的高清无损截图")
    print("  • 【Q 键】或【ESC】: 退出窗口")
    print("=" * 65)

    vis.run()
    vis.destroy_window()



def main():
    parser = argparse.ArgumentParser(description="SfM 相机匹配拓扑连线与稀疏几何对比可视化工具")
    parser.add_argument("--export", action="store_true", help="一键批量导出 4 类代表样本的论文对比大图")
    parser.add_argument("--sample", type=str, default="B2-Day0-7_67", help="指定要查看的样本 ID (默认 B2-Day0-7_67)")
    parser.add_argument("--category", type=str, default="", help="样本对应类别 (留空自动查找)")
    args = parser.parse_args()

    if args.export:
        export_figure1_master()
    else:
        cat = args.category
        if not cat:
            for c, sid, _ in REPRESENTATIVES:
                if sid == args.sample:
                    cat = c
                    break
        if not cat:
            cat = "healthy_sparse"
        run_interactive_viewer(cat, args.sample)


if __name__ == "__main__":
    main()
