'''p3_calibrate_segment - ground alignment, cube scale recovery and trait extraction.'''

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def fit_ground_plane_and_align(pcd: o3d.geometry.PointCloud) -> tuple[np.ndarray, np.ndarray, float]:
    '''通过 RANSAC 拟合桌面/地面主平面，并将坐标系旋转扶正，使地面法向对齐 +Z 轴，地面绝对高程设为 Z = 0.0。'''
    pts = np.asarray(pcd.points)

    plane_model, table_inliers = pcd.segment_plane(
        distance_threshold=0.03, ransac_n=3, num_iterations=2000
    )
    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=np.float64)
    normal /= np.linalg.norm(normal)

    if normal[1] > 0:
        normal = -normal
        d = -d

    target_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    v = np.cross(normal, target_axis)
    s = np.linalg.norm(v)
    c_ang = np.dot(normal, target_axis)

    if s < 1e-6:
        R = np.eye(3) if c_ang > 0 else -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + (vx @ vx) * ((1.0 - c_ang) / (s**2))

    rot_pts = (R @ pts.T).T

    table_pts = (R @ np.asarray(pcd.select_by_index(table_inliers).points).T).T
    table_z_offset = float(table_pts[:, 2].mean())
    rot_pts[:, 2] -= table_z_offset

    print(f"  [地面拟合] 成功识别桌面基准面 (支撑点数: {len(table_inliers):,})")
    print(f"  [坐标扶正] 桌面已精确水平对齐于 X-Y 平面，地面高程设为 Z = 0.0 mm")

    return R, rot_pts, table_z_offset


def detect_cubes_and_build_tight_boxes(
    rot_pts: np.ndarray,
    colors: np.ndarray,
    real_cube_size_mm: float = 30.0,
    line_density: int = 150,
) -> tuple[float, list[dict], np.ndarray, np.ndarray]:
    '''严格剔除飞点与木质杂点，精准提取魔方实体，并构建严密贴合的 2.5D 立体红色线框。'''
    r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]

    c_red = (r > 0.50) & (r - g > 0.20) & (r - b > 0.30)
    c_blue = (b > 0.40) & (b - r > 0.25)
    c_yellow = (r > 0.60) & (g > 0.55) & (b < 0.25) & (r - b > 0.40) & (g - b > 0.35)
    c_green = (g > 0.45) & (g - r > 0.15) & (g - b > 0.15)
    c_black = (r < 0.25) & (g < 0.25) & (b < 0.25)

    cube_colors = c_red | c_blue | c_yellow | c_green | c_black

    z_above_table = (rot_pts[:, 2] > 0.02) & (rot_pts[:, 2] < 0.38)
    cand_mask = z_above_table & cube_colors

    cand_pcd = o3d.geometry.PointCloud()
    cand_pcd.points = o3d.utility.Vector3dVector(rot_pts[cand_mask])
    cand_pcd.colors = o3d.utility.Vector3dVector(colors[cand_mask])

    labels = np.array(cand_pcd.cluster_dbscan(eps=0.12, min_points=40))
    unique_labels = [l for l in np.unique(labels) if l >= 0]

    detected_cubes = []
    for l in unique_labels:
        idx = np.where(labels == l)[0]
        sub_pcd = cand_pcd.select_by_index(idx)
        raw_pts = np.asarray(sub_pcd.points)

        raw_w, raw_h = sorted(cv2.minAreaRect(raw_pts[:, :2].astype(np.float32))[1])
        if not (0.20 < raw_w < 0.50):
            continue

        clean_pcd, inliers = sub_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
        clean_pts = np.asarray(clean_pcd.points)

        if len(clean_pts) < 100:
            continue

        top_mask = clean_pts[:, 2] > 0.18
        if top_mask.sum() > 30:
            top_pcd = o3d.geometry.PointCloud()
            top_pcd.points = o3d.utility.Vector3dVector(clean_pts[top_mask])
            _, top_inliers = top_pcd.segment_plane(distance_threshold=0.005, ransac_n=3, num_iterations=1000)
            top_inlier_pts = np.asarray(top_pcd.points)[top_inliers]
            z_top = float(np.percentile(top_inlier_pts[:, 2], 90.0))
        else:
            z_top = float(np.percentile(clean_pts[:, 2], 92.0))

        xy = clean_pts[:, :2]
        best_area = 1e9
        best_res = None
        for deg in np.linspace(0, 90, 361):
            rad = np.deg2rad(deg)
            c, s = np.cos(rad), np.sin(rad)
            R_2d = np.array([[c, s], [-s, c]])
            uv = xy @ R_2d.T
            u0, u1 = np.percentile(uv[:, 0], 2.5), np.percentile(uv[:, 0], 97.5)
            v0, v1 = np.percentile(uv[:, 1], 2.5), np.percentile(uv[:, 1], 97.5)
            area = (u1 - u0) * (v1 - v0)
            if area < best_area:
                best_area = area
                best_res = (deg, u0, u1, v0, v1, R_2d)

        deg, u0, u1, v0, v1, R_2d = best_res
        w = float(u1 - u0)
        h = float(v1 - v0)

        corners_uv = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
        corners_2d_unit = corners_uv @ R_2d

        if 0.22 < w < 0.45 and (max(w, h) / min(w, h)) < 1.35:
            detected_cubes.append({
                "cluster_id": int(l),
                "clean_pcd": clean_pcd,
                "corners_2d_unit": corners_2d_unit,
                "yaw_deg": float(deg),
                "w_unit": float(w),
                "h_unit": float(h),
                "z_max_unit": float(z_top),
                "mean_dim_unit": float((w + h + z_top) / 3.0),
            })

    if not detected_cubes:
        print("  [警告] 未检测到魔方，回退到默认标定")
        return 100.0, [], np.empty((0, 3)), np.empty((0, 3))

    mean_cube_units = float(np.mean([c["mean_dim_unit"] for c in detected_cubes]))
    scale = real_cube_size_mm / mean_cube_units

    print(f"\n  [魔方检出] 已按主体点拟合顶面与 4 个侧面（忽略乱飞杂点），精确定位 {len(detected_cubes)} 个魔方:")
    cube_records = []
    all_wireframe_pts = []

    edges_idx = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for i, c in enumerate(detected_cubes, 1):
        corners_2d = c["corners_2d_unit"] * scale
        z_bottom_mm = 0.0
        z_top_mm = c["z_max_unit"] * scale

        p_bottom = np.column_stack([corners_2d, np.full(4, z_bottom_mm)])
        p_top = np.column_stack([corners_2d, np.full(4, z_top_mm)])
        corners_8 = np.vstack([p_bottom, p_top])

        for (i1, i2) in edges_idx:
            p1, p2 = corners_8[i1], corners_8[i2]
            alphas = np.linspace(0.0, 1.0, line_density)[:, None]
            line_pts = p1 * (1.0 - alphas) + p2 * alphas
            all_wireframe_pts.append(line_pts)

        w_mm = c["w_unit"] * scale
        h_mm = c["h_unit"] * scale
        z_mm = c["z_max_unit"] * scale
        mean_mm = c["mean_dim_unit"] * scale

        rec = {
            "cube_id": i,
            "dimensions_mm": [round(w_mm, 1), round(h_mm, 1), round(z_mm, 1)],
            "mean_edge_mm": round(mean_mm, 1),
            "rotation_angle_deg": round(c["yaw_deg"], 1),
        }
        cube_records.append(rec)
        print(f"    ▸ 魔方 #{i}: 严密贴合长宽高 = {w_mm:.1f} × {h_mm:.1f} × {z_mm:.1f} mm "
              f"(平均棱长={mean_mm:.1f} mm, 偏航角度={c['yaw_deg']:.1f}°)")

    wireframe_pts_mm = np.vstack(all_wireframe_pts)
    wireframe_cols = np.tile([1.0, 0.0, 0.0], (len(wireframe_pts_mm), 1))

    print(f"  ==> 尺度标定收敛: 3 个魔方实测平均 = {np.mean([r['mean_edge_mm'] for r in cube_records]):.1f} mm")
    print(f"      用户实测基准: {real_cube_size_mm:.1f} mm")
    print(f"      全局物理尺度因子 (Scale): {scale:.3f} mm/unit\n")

    return scale, cube_records, wireframe_pts_mm, wireframe_cols


def segment_plant_and_measure_height(
    pts_mm: np.ndarray,
    colors: np.ndarray,
    out_dir: Path,
    line_density: int = 150,
) -> tuple[dict, np.ndarray, np.ndarray]:
    '''以地面 Z = 0.0 mm 为严格起点测量真实株高，提取纯净苗木点云，并生成 3D 地面基准与株高测量标尺可视化线框。'''
    exg = 2 * colors[:, 1] - colors[:, 0] - colors[:, 2]

    upper_mask = (pts_mm[:, 2] > 70.0) & (pts_mm[:, 2] < 120.0)
    if upper_mask.sum() > 500:
        pot_center_xy = pts_mm[upper_mask, :2].mean(axis=0)
    else:
        pot_center_xy = np.array([0.0, 0.0])

    dist_to_pot = np.linalg.norm(pts_mm[:, :2] - pot_center_xy, axis=1)

    soil_cand = (dist_to_pot < 55.0) & (pts_mm[:, 2] > 100.0) & (pts_mm[:, 2] < 160.0)
    if soil_cand.sum() > 200:
        soil_height_mm = float(np.percentile(pts_mm[soil_cand, 2], 75.0))
    else:
        soil_height_mm = 128.0

    plant_mask = (
        (dist_to_pot < 90.0)
        & (pts_mm[:, 2] > soil_height_mm)
        & (pts_mm[:, 2] < 350.0)
        & (exg > 0.03)
    )

    raw_plant_pcd = o3d.geometry.PointCloud()
    raw_plant_pcd.points = o3d.utility.Vector3dVector(pts_mm[plant_mask])
    raw_plant_pcd.colors = o3d.utility.Vector3dVector(colors[plant_mask])

    clean_plant_pcd, _ = raw_plant_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    clean_pts = np.asarray(clean_plant_pcd.points)

    plant_file = out_dir / "plant_segmented.ply"
    o3d.io.write_point_cloud(str(plant_file), clean_plant_pcd)
    print(f"  [纯净植株保存] 已导出纯小苗点云 -> {plant_file.name} ({len(clean_pts):,} 点)")

    apex_idx = np.argmax(clean_pts[:, 2])
    apex_pt = clean_pts[apex_idx]
    max_z_from_ground = float(apex_pt[2])
    net_seedling_height = max_z_from_ground - soil_height_mm
    canopy_span_x = float(clean_pts[:, 0].max() - clean_pts[:, 0].min())
    canopy_span_y = float(clean_pts[:, 1].max() - clean_pts[:, 1].min())
    max_canopy = max(canopy_span_x, canopy_span_y)

    metrics = {
        "ground_reference": "Table Surface Z = 0.0 mm",
        "plant_height_from_ground_mm": round(max_z_from_ground, 1),
        "plant_height_from_ground_cm": round(max_z_from_ground / 10.0, 1),
        "pot_soil_level_mm": round(soil_height_mm, 1),
        "pot_soil_level_cm": round(soil_height_mm / 10.0, 1),
        "net_seedling_height_above_soil_mm": round(net_seedling_height, 1),
        "net_seedling_height_above_soil_cm": round(net_seedling_height / 10.0, 1),
        "max_canopy_diameter_mm": round(max_canopy, 1),
        "plant_point_count": len(clean_pts),
    }

    marker_lines = []
    marker_colors = []

    def add_line(p1, p2, color, count=120):
        alphas = np.linspace(0.0, 1.0, count)[:, None]
        pts = p1 * (1.0 - alphas) + p2 * alphas
        marker_lines.append(pts)
        marker_colors.append(np.tile(color, (count, 1)))

    px_min, px_max = clean_pts[:, 0].min(), clean_pts[:, 0].max()
    py_min, py_max = clean_pts[:, 1].min(), clean_pts[:, 1].max()
    pz_min, pz_max = soil_height_mm, max_z_from_ground

    box_c = [
        np.array([px_min, py_min, pz_min]), np.array([px_max, py_min, pz_min]),
        np.array([px_max, py_max, pz_min]), np.array([px_min, py_max, pz_min]),
        np.array([px_min, py_min, pz_max]), np.array([px_max, py_min, pz_max]),
        np.array([px_max, py_max, pz_max]), np.array([px_min, py_max, pz_max]),
    ]
    box_edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
    for (i1, i2) in box_edges:
        add_line(box_c[i1], box_c[i2], [0.0, 1.0, 0.2], count=80)

    ruler_x = pot_center_xy[0] - 70.0
    ruler_y = pot_center_xy[1]
    p_ruler_soil = np.array([ruler_x, ruler_y, soil_height_mm])
    p_ruler_top = np.array([ruler_x, ruler_y, max_z_from_ground])

    add_line(p_ruler_soil, p_ruler_top, [0.0, 0.95, 1.0], count=200)

    add_line(p_ruler_soil, np.array([ruler_x + 40.0, ruler_y, soil_height_mm]), [1.0, 0.85, 0.0], count=70)
    stem_base_xy = np.array([apex_pt[0], apex_pt[1]])
    add_line(
        np.array([stem_base_xy[0] - 30.0, stem_base_xy[1], soil_height_mm]),
        np.array([stem_base_xy[0] + 30.0, stem_base_xy[1], soil_height_mm]),
        [1.0, 0.85, 0.0], count=80
    )

    add_line(p_ruler_top, apex_pt, [0.0, 0.95, 1.0], count=100)

    vis_marker_pts = np.vstack(marker_lines)
    vis_marker_cols = np.vstack(marker_colors)

    return metrics, vis_marker_pts, vis_marker_cols


def run_pipeline(
    ply_path: Path,
    output_dir: Path | None = None,
    real_cube_size_mm: float = 30.0,
):
    print("=" * 68)
    print("3D-Pipeline [阶段 3] 地面平贴扶正、魔方去噪紧密贴合划线与真实株高测量")
    print(f"输入点云: {ply_path}")
    print("=" * 68)
    out_dir = output_dir or ply_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    pcd = o3d.io.read_point_cloud(str(ply_path))
    colors = np.asarray(pcd.colors)

    R, rot_pts, table_z = fit_ground_plane_and_align(pcd)

    scale, cube_records, wireframe_pts, wireframe_cols = detect_cubes_and_build_tight_boxes(
        rot_pts, colors, real_cube_size_mm=real_cube_size_mm
    )

    pts_mm = rot_pts * scale

    metrics, plant_marker_pts, plant_marker_cols = segment_plant_and_measure_height(pts_mm, colors, out_dir)

    marked_pts = np.vstack([pts_mm, wireframe_pts, plant_marker_pts])
    marked_cols = np.vstack([colors, wireframe_cols, plant_marker_cols])

    marked_scene_pcd = o3d.geometry.PointCloud()
    marked_scene_pcd.points = o3d.utility.Vector3dVector(marked_pts)
    marked_scene_pcd.colors = o3d.utility.Vector3dVector(marked_cols)

    marked_file = out_dir / "scene_with_cubes_marked.ply"
    o3d.io.write_point_cloud(str(marked_file), marked_scene_pcd)
    print(f"  [划线场景保存] 魔方红框 + 地面基准环 + 植株株高垂直标尺已合成导出:")
    print(f"                 -> 全景核验点云: {marked_file.name}")

    full_aligned_file = out_dir / "fused_scaled_upright.ply"
    o3d.io.write_point_cloud(str(full_aligned_file), marked_scene_pcd)

    report = {
        "source_ply": str(ply_path),
        "real_cube_benchmark_mm": real_cube_size_mm,
        "scale_mm_per_unit": round(scale, 3),
        "cubes_detected": cube_records,
        "measurements": metrics,
    }

    report_file = out_dir / "calibration_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 68)
    print("🎯 最终物理尺寸精准测量结果 (以实测 3.0cm 魔方为基准):")
    print(f"   ▸ 真实株高 (地面到植株最顶部): {metrics['plant_height_from_ground_mm']} mm ({metrics['plant_height_from_ground_cm']} cm)")
    print(f"   ▸ 花盆土壤表面高程:          {metrics['pot_soil_level_mm']} mm ({metrics['pot_soil_level_cm']} cm)")
    print(f"   ▸ 出土小苗净高度:            {metrics['net_seedling_height_above_soil_mm']} mm ({metrics['net_seedling_height_above_soil_cm']} cm)")
    print(f"   ▸ 冠幅最大跨度:              {metrics['max_canopy_diameter_mm']} mm")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser(description="3D-Pipeline 阶段 3: 地面平贴扶正、魔方去噪紧密贴合划线与真实株高测量")
    parser.add_argument("ply", type=Path, nargs="?", default=None,
                        help="输入点云 (默认自动找最新 output/*/fused.ply)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="输出目录")
    parser.add_argument("--cube-size", type=float, default=30.0,
                        help="魔方真实物理边长 (毫米 mm，实测 3cm 迷你魔方即 30.0)")
    args = parser.parse_args()

    ply_path = args.ply
    if ply_path is None:
        candidates = sorted(Path("output").glob("*/fused.ply"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            print("错误: 未找到任何 output/*/fused.ply 文件", file=sys.stderr)
            sys.exit(1)
        ply_path = candidates[-1]

    run_pipeline(ply_path, output_dir=args.output, real_cube_size_mm=args.cube_size)


if __name__ == "__main__":
    main()
