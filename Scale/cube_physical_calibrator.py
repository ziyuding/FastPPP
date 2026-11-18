'''cube_physical_calibrator - reference-cube scale recovery and ground alignment.'''

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d


def fit_ground_plane_and_align(pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray, float]:
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

    return R, rot_pts, table_z_offset


def detect_cubes_and_get_scale(
    rot_pts: np.ndarray,
    colors: np.ndarray,
    real_cube_size_mm: float = 30.0,
) -> Tuple[float, List[Dict]]:
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
            c_val, s_val = np.cos(rad), np.sin(rad)
            R_2d = np.array([[c_val, s_val], [-s_val, c_val]])
            uv = xy @ R_2d.T
            u0, u1 = np.percentile(uv[:, 0], 2.5), np.percentile(uv[:, 0], 97.5)
            v0, v1 = np.percentile(uv[:, 1], 2.5), np.percentile(uv[:, 1], 97.5)
            area = (u1 - u0) * (v1 - v0)
            if area < best_area:
                best_area = area
                best_res = (deg, u0, u1, v0, v1, R_2d)

        deg, u0, u1, v0, v1, R_2d = best_res
        w_unit = u1 - u0
        h_unit = v1 - v0
        z_max_unit = z_top
        mean_dim_unit = (w_unit + h_unit + z_max_unit) / 3.0

        detected_cubes.append({
            "w_unit": w_unit,
            "h_unit": h_unit,
            "z_max_unit": z_max_unit,
            "mean_dim_unit": mean_dim_unit,
            "yaw_deg": deg,
        })

    if not detected_cubes:
        return 130.0, []

    mean_cube_units = float(np.mean([c["mean_dim_unit"] for c in detected_cubes]))
    scale_mm_per_unit = real_cube_size_mm / mean_cube_units

    cube_records = []
    for i, c in enumerate(detected_cubes, 1):
        w_mm = c["w_unit"] * scale_mm_per_unit
        h_mm = c["h_unit"] * scale_mm_per_unit
        z_mm = c["z_max_unit"] * scale_mm_per_unit
        mean_mm = c["mean_dim_unit"] * scale_mm_per_unit
        cube_records.append({
            "cube_id": i,
            "dimensions_mm": [round(w_mm, 1), round(h_mm, 1), round(z_mm, 1)],
            "mean_edge_mm": round(mean_mm, 1),
            "rotation_angle_deg": round(c["yaw_deg"], 1),
        })

    return scale_mm_per_unit, cube_records


def resolve_sample_physical_pointcloud(
    ply_path: Path,
    custom_scale: Optional[float] = None,
    real_cube_size_mm: float = 30.0,
) -> Tuple[o3d.geometry.PointCloud, Dict]:
    ply_path = Path(ply_path)
    pcd = o3d.io.read_point_cloud(str(ply_path))
    raw_pts = np.asarray(pcd.points)
    raw_colors = np.asarray(pcd.colors) if pcd.has_colors() else np.tile([0.2, 0.8, 0.3], (len(raw_pts), 1))

    stem = ply_path.stem
    m = re.match(r"^(\d+_[A-Za-z0-9\-]+)", stem)
    sample_id = m.group(1) if m else stem.split("_")[0]
    pure_id = stem.split("_")[0]

    calib_info = {
        "sample_id": sample_id,
        "ply_path": str(ply_path),
        "source": "auto",
        "scale_mm_per_unit": 130.0,
        "scale_to_cm": 13.0,
        "ground_z_offset": 0.0,
        "cubes_count": 0,
    }

    raw_scene_candidates = [
        ply_path.parent.parent / "00_Raw_GS_Scene" / f"{sample_id}_raw_scene.ply",
        ply_path.parent.parent / "00_Raw_GS_Scene" / f"{pure_id}_raw_scene.ply",
    ]
    if ply_path.parent.parent.joinpath("00_Raw_GS_Scene").exists():
        for r_ply in ply_path.parent.parent.joinpath("00_Raw_GS_Scene").glob(f"*{pure_id}*.ply"):
            raw_scene_candidates.append(r_ply)

    found_raw_scene = None
    for rc in raw_scene_candidates:
        if rc.exists():
            found_raw_scene = rc
            break

    if custom_scale is not None:
        scale_mm = float(custom_scale)
        scale_to_cm = scale_mm / 10.0
        calib_info["source"] = "manual_custom_scale"
        calib_info["scale_mm_per_unit"] = scale_mm
        calib_info["scale_to_cm"] = scale_to_cm
        pts_cm = raw_pts * scale_to_cm
    elif found_raw_scene is not None:
        try:
            raw_pcd = o3d.io.read_point_cloud(str(found_raw_scene))
            R, rot_pts, table_z = fit_ground_plane_and_align(raw_pcd)
            scale_mm, cubes = detect_cubes_and_get_scale(rot_pts, np.asarray(raw_pcd.colors), real_cube_size_mm)
            scale_to_cm = scale_mm / 10.0
            calib_info["source"] = f"00_Raw_GS_Scene ({found_raw_scene.name})"
            calib_info["scale_mm_per_unit"] = scale_mm
            calib_info["scale_to_cm"] = scale_to_cm
            calib_info["cubes_count"] = len(cubes)
            pts_cm = raw_pts * scale_to_cm
        except Exception:
            pts_cm = raw_pts * 13.0
    else:
        scale_to_cm = 13.0
        calib_info["source"] = "standard_benchmark_scale (130.0 mm/unit)"
        calib_info["scale_mm_per_unit"] = 130.0
        calib_info["scale_to_cm"] = 13.0
        pts_cm = raw_pts * scale_to_cm

    z_min = float(np.percentile(pts_cm[:, 2], 0.5))
    if z_min < 0:
        pts_cm[:, 2] -= z_min

    phys_pcd = o3d.geometry.PointCloud()
    phys_pcd.points = o3d.utility.Vector3dVector(pts_cm)
    phys_pcd.colors = o3d.utility.Vector3dVector(raw_colors)

    return phys_pcd, calib_info
