'''p1_useColmap - thin Python wrapper around the COLMAP command line tools.'''

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


COLMAP_BIN = Path(os.environ.get("COLMAP_BIN", "colmap"))
COLMAP_BIN_WIN = Path(os.environ.get("COLMAP_BIN_WIN", "colmap.exe"))


def _resolve_colmap() -> Path:
    '''自动找到 colmap 可执行文件。'''
    candidates = [
        COLMAP_BIN,
        COLMAP_BIN_WIN,
        Path(os.environ.get("COLMAP_BIN", "colmap")),
        Path("/usr/local/bin/colmap"),
        Path("/usr/bin/colmap"),
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    cm = shutil.which(str(COLMAP_BIN))
    if cm:
        return Path(cm)
    raise FileNotFoundError(
        "找不到 colmap。请设置环境变量 COLMAP_BIN 或确保 colmap 在系统 PATH 中。"
    )


def _run(cmd: list[str], *, step_name: str = "", cwd: Path | None = None, timeout: int | None = None):
    '''运行一个 colmap 子进程，实时打印输出，失败抛异常。'''
    label = f"[{step_name}]" if step_name else "[colmap]"
    print(f"{label} {' '.join(map(shlex.quote, cmd))}")

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        cwd=str(cwd) if cwd else None,
        bufsize=1,
    )
    for line in proc.stdout:
        print(f"  {label} {line.rstrip()}")

    rc = proc.wait(timeout=timeout)
    elapsed = time.perf_counter() - t0

    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd, output=None, stderr=None)
    print(f"{label} 完成 ({elapsed:.1f}s)\n")



def _detect_camera_setup(image_dir: Path) -> str:
    '''根据目录结构自动推断相机数量。'''
    subdirs = [d for d in image_dir.iterdir() if d.is_dir()]
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    img_subdirs = [
        d for d in subdirs
        if any(f.suffix.lower() in img_exts for f in d.iterdir() if f.is_file())
    ]
    if len(img_subdirs) >= 2:
        return "per_folder"
    return "1"


def _next_output_dir(base_dir: Path) -> Path:
    '''返回 output/N/ 的下一个可用编号目录。'''
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    for d in base_dir.iterdir():
        if d.is_dir() and d.name.isdigit():
            existing.append(int(d.name))
    next_n = max(existing) + 1 if existing else 1
    return base_dir / str(next_n)



class ColmapPipeline:
    '''封装 COLMAP CLI 的重建流水线。'''

    def __init__(self, colmap_exe: Path, project_dir: Path, image_dir: Path):
        self.colmap = colmap_exe
        self.project = Path(project_dir)
        self.images = Path(image_dir).resolve()
        self.db = self.project / "database.db"
        self.sparse_dir = self.project / "sparse"
        self.dense_dir = self.project / "dense"

    def _colmap(self, subcommand: str, *args: str) -> list[str]:
        return [str(self.colmap), subcommand] + list(args)


    def extract_features(
        self,
        camera_model: str = "SIMPLE_RADIAL",
        camera_params: str | None = None,
        quality: str = "high",
        overwrite: bool = False,
        single_camera: str = "1",
    ):
        '''提取 SIFT 特征。'''
        quality_map = {
            "low":     (1200, 4000),
            "medium":  (1600, 6000),
            "high":    (2400, 8192),
            "extreme": (3200, 16384),
        }
        max_size, max_feat = quality_map.get(quality, quality_map["high"])

        cmd = self._colmap(
            "feature_extractor",
            "--database_path", str(self.db),
            "--image_path", str(self.images),
            "--ImageReader.camera_model", camera_model,
            "--FeatureExtraction.max_image_size", str(max_size),
            "--SiftExtraction.max_num_features", str(max_feat),
        )

        if camera_params:
            cmd += ["--ImageReader.camera_params", camera_params]

        if single_camera == "per_folder":
            cmd += ["--ImageReader.single_camera_per_folder", "1"]
        else:
            cmd += ["--ImageReader.single_camera", single_camera]

        if overwrite:
            if self.db.exists():
                self.db.unlink()
                print(f"  已删除旧数据库: {self.db}")

        label = f"SfM:extract [single_camera={single_camera}]"
        _run(cmd, step_name=label)

    def _generate_dual_camera_pairs(self, out_file: Path, window: int = 8):
        '''为双相机（高低双环轨道）生成结构化配对列表：'''
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        high_dir = self.images / "cam_high"
        low_dir = self.images / "cam_low"

        if not (high_dir.is_dir() and low_dir.is_dir()):
            subdirs = [d for d in self.images.iterdir() if d.is_dir()]
            if len(subdirs) >= 2:
                high_dir, low_dir = subdirs[0], subdirs[1]
            else:
                imgs = sorted([f.name for f in self.images.iterdir() if f.suffix.lower() in img_exts])
                N = len(imgs)
                pairs = set()
                for i in range(N):
                    for off in range(1, window + 1):
                        j = (i + off) % N
                        pairs.add(tuple(sorted([imgs[i], imgs[j]])))
                with open(out_file, "w") as f:
                    for p1, p2 in sorted(pairs):
                        f.write(f"{p1} {p2}\n")
                return

        high_imgs = sorted([f"{high_dir.name}/{f.name}" for f in high_dir.iterdir() if f.suffix.lower() in img_exts])
        low_imgs = sorted([f"{low_dir.name}/{f.name}" for f in low_dir.iterdir() if f.suffix.lower() in img_exts])

        pairs = set()
        Nh, Nl = len(high_imgs), len(low_imgs)
        for i in range(Nh):
            for off in range(1, window + 1):
                j = (i + off) % Nh
                pairs.add(tuple(sorted([high_imgs[i], high_imgs[j]])))
        for i in range(Nl):
            for off in range(1, window + 1):
                j = (i + off) % Nl
                pairs.add(tuple(sorted([low_imgs[i], low_imgs[j]])))
        for i, h_name in enumerate(high_imgs):
            ratio = i / Nh if Nh > 0 else 0
            l_center = int(round(ratio * Nl))
            for off in range(-window, window + 1):
                if Nl > 0:
                    j = (l_center + off) % Nl
                    pairs.add(tuple(sorted([h_name, low_imgs[j]])))

        with open(out_file, "w") as f:
            for p1, p2 in sorted(pairs):
                f.write(f"{p1} {p2}\n")

    def match_features(
        self,
        mode: str = "exhaustive",
        overlap: int = 10,
        gpu_index: int = 0,
    ):
        '''特征匹配。'''
        if mode == "dual_fast":
            pairs_file = self.project / "pairs.txt"
            self._generate_dual_camera_pairs(pairs_file, window=overlap)
            cmd = self._colmap(
                "matches_importer",
                "--database_path", str(self.db),
                "--match_list_path", str(pairs_file),
                "--match_type", "pairs",
                "--FeatureMatching.gpu_index", str(gpu_index),
            )
        elif mode == "exhaustive":
            cmd = self._colmap(
                "exhaustive_matcher",
                "--database_path", str(self.db),
                "--FeatureMatching.gpu_index", str(gpu_index),
            )
        elif mode == "sequential":
            cmd = self._colmap(
                "sequential_matcher",
                "--database_path", str(self.db),
                "--FeatureMatching.gpu_index", str(gpu_index),
                "--SequentialMatching.overlap", str(overlap),
            )
        else:
            cmd = self._colmap(
                "vocab_tree_matcher",
                "--database_path", str(self.db),
                "--FeatureMatching.gpu_index", str(gpu_index),
            )
        _run(cmd, step_name=f"SfM:match [{mode}, overlap={overlap}]" if mode in ("sequential", "dual_fast") else f"SfM:match [{mode}]")

    def sparse_recon(self, fast: bool = False):
        '''增量式 SfM 稀疏重建，跑完自动导出 TXT 位姿。'''
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._colmap(
            "mapper",
            "--database_path", str(self.db),
            "--image_path", str(self.images),
            "--output_path", str(self.sparse_dir),
        )
        if fast:
            cmd += [
                "--Mapper.multiple_models", "0",
                "--Mapper.ba_global_frames_ratio", "1.3",
                "--Mapper.ba_global_max_num_iterations", "20",
                "--Mapper.ba_local_max_num_iterations", "15",
            ]
        _run(cmd, step_name="SfM:mapper [fast]" if fast else "SfM:mapper")

        best_sparse = self._get_best_sparse_model()
        if not (best_sparse / "images.bin").exists():
            print("⚠ mapper 未生成 images.bin，跳过导出（可能全部注册失败）")
            return

        self._export_sparse_to_txt(best_sparse)
        self._print_sparse_summary()
        self.verify_sfm_quality()

    def _get_best_sparse_model(self) -> Path:
        '''获取点数最多或最完整的主稀疏模型路径（防止存在 0, 1 等多模型时选错）。'''
        submodels = [d for d in self.sparse_dir.iterdir() if d.is_dir() and (d / "images.bin").exists()]
        if not submodels:
            return self.sparse_dir / "0"
        return max(submodels, key=lambda d: (d / "points3D.bin").stat().st_size if (d / "points3D.bin").exists() else 0)

    def verify_sfm_quality(self, max_step_thresh: float = 1.2, max_loop_gap: float = 1.5) -> bool:
        '''自动质检 SfM 轨迹完整性、连续性与闭环度，支持单相机与双相机独立环轨道检测，并自动量化首尾冗余重复帧。'''
        import json
        out_dir = self.project / "sparse_txt"
        images_txt = out_dir / "images.txt"
        if not images_txt.exists():
            print("\n❌ [SfM 质检报警] images.txt 不存在，SfM 失败！")
            return False

        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        disk_img_paths = [f for f in self.images.rglob("*") if f.is_file() and f.suffix.lower() in img_exts]
        total_disk_imgs = len(disk_img_paths)

        poses = []
        with open(images_txt, "r") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            img_name = parts[-1]
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            R = np.array([
                [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
            ])
            pos = -R.T @ np.array([tx, ty, tz])
            view_dir = R.T @ np.array([0.0, 0.0, 1.0])
            norm_v = np.linalg.norm(view_dir)
            if norm_v > 1e-6:
                view_dir = view_dir / norm_v
            poses.append((img_name, pos, view_dir))

        reg_count = len(poses)
        reg_ratio = reg_count / total_disk_imgs if total_disk_imgs > 0 else 1.0

        groups = {}
        for p in poses:
            img_name = p[0]
            parent_name = Path(img_name).parent.name
            group_key = parent_name if parent_name else "camera"
            groups.setdefault(group_key, []).append(p)

        is_passed = True
        warnings = []
        camera_reports = {}
        all_redundant_frames = []

        if reg_ratio < 0.85:
            is_passed = False
            warnings.append(f"总注册图片过少 ({reg_count}/{total_disk_imgs} < 85%)")

        print("\n" + "=" * 65)
        print("          🔍 SfM 轨迹质检与质量诊断报告 (双环独立核验)")
        print("=" * 65)
        print(f"  • 总注册成功率 : {reg_count}/{total_disk_imgs} ({reg_ratio*100:.1f}%)")

        for g_name, g_poses in sorted(groups.items()):
            g_poses.sort(key=lambda x: x[0])
            coords = np.array([p[1] for p in g_poses])
            vdirs = np.array([p[2] for p in g_poses])
            n_frames = len(coords)
            diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1) if n_frames > 1 else np.array([])
            mean_step = float(diffs.mean()) if len(diffs) > 0 else 0.0
            max_step = float(diffs.max()) if len(diffs) > 0 else 0.0
            loop_gap = float(np.linalg.norm(coords[0] - coords[-1])) if n_frames > 1 else 0.0

            check_loop_k = min(10, n_frames // 4)
            if n_frames > check_loop_k * 2:
                head_loop = coords[:check_loop_k]
                tail_loop = coords[-check_loop_k:]
                min_loop_gap = float(np.min(
                    np.linalg.norm(tail_loop[:, None, :] - head_loop[None, :, :], axis=2)
                ))
            else:
                min_loop_gap = loop_gap

            jumps = []
            for i, d in enumerate(diffs):
                if d > max(max_step_thresh, mean_step * 3.0):
                    jumps.append((g_poses[i][0], g_poses[i+1][0], float(d)))

            check_k = min(15, n_frames // 3)
            head_coords = coords[:check_k]
            tail_coords = coords[-check_k:]
            head_dirs = vdirs[:check_k]
            tail_dirs = vdirs[-check_k:]

            redundant = []
            for t_idx, (t_pt, t_dir) in enumerate(zip(tail_coords, tail_dirs)):
                actual_tail_idx = n_frames - check_k + t_idx
                tail_name = g_poses[actual_tail_idx][0]
                dists = np.linalg.norm(head_coords - t_pt, axis=1)
                angles = np.arccos(np.clip(np.sum(head_dirs * t_dir, axis=1), -1.0, 1.0)) * 180.0 / np.pi
                best_h_idx = int(np.argmin(dists))
                best_dist = float(dists[best_h_idx])
                best_angle = float(angles[best_h_idx])

                is_overlap = (best_dist < max(0.15, mean_step * 1.2) and best_angle < 12.0)
                is_stationary = False
                if actual_tail_idx > 0 and len(diffs) >= actual_tail_idx:
                    step_from_prev = diffs[actual_tail_idx - 1]
                    if step_from_prev < mean_step * 0.15:
                        is_stationary = True

                if is_overlap or is_stationary:
                    redundant.append({
                        "tail_image": tail_name,
                        "tail_index": actual_tail_idx,
                        "overlaps_with": g_poses[best_h_idx][0],
                        "dist": best_dist,
                        "angle_deg": best_angle,
                        "reason": "head_tail_overlap" if is_overlap else "stationary_tail"
                    })
                    all_redundant_frames.append(tail_name)

            cam_passed = True
            cam_warns = []
            if jumps:
                cam_passed = False
                jump_str = ", ".join(f"{p1}->{p2} (跳跃={d:.2f})" for p1, p2, d in jumps[:2])
                cam_warns.append(f"轨迹跳跃断裂: {jump_str}")
            if min_loop_gap > max(max_loop_gap, mean_step * 3.5):
                cam_passed = False
                cam_warns.append(f"环拍未闭合 (首尾最小间距={min_loop_gap:.2f} > 允许阈值, 字典首尾={loop_gap:.2f})")

            if not cam_passed:
                is_passed = False
                warnings.extend([f"[{g_name}] {w}" for w in cam_warns])

            camera_reports[g_name] = {
                "frames": n_frames,
                "mean_step": round(mean_step, 4),
                "max_step": round(max_step, 4),
                "loop_gap": round(loop_gap, 4),
                "min_loop_gap": round(min_loop_gap, 4),
                "passed": cam_passed,
                "warnings": cam_warns,
                "redundant_count": len(redundant),
                "redundant_frames": redundant,
            }

            status_sym = "✅" if cam_passed else "❌"
            print(f"\n  --- 相机轨道 [{g_name}] {status_sym} ---")
            print(f"      帧数: {n_frames} | 平均步长: {mean_step:.3f} | 最大步长: {max_step:.3f} | 首尾最小闭合间距: {min_loop_gap:.3f}")
            if redundant:
                red_names = [Path(r["tail_image"]).name for r in redundant]
                print(f"      💡 发现首尾冗余帧 ({len(redundant)} 张): {', '.join(red_names[:5])}{'...' if len(red_names) > 5 else ''}")
                for r in redundant[:3]:
                    print(f"         ↳ 尾部 {Path(r['tail_image']).name} 与 头部 {Path(r['overlaps_with']).name} 重合: 间距={r['dist']:.3f}, 夹角={r['angle_deg']:.1f}°")

        print("\n" + "-" * 65)
        if is_passed:
            print("  ==> 【质检通过】✅ 双环相机轨迹均平滑闭合！")
        else:
            print("  ==> 【质检报警】❌ 存在轨迹断裂或未闭环！")
            for w in warnings:
                print(f"      - ⚠ {w}")
        print("=" * 65 + "\n")

        report_data = {
            "is_passed": is_passed,
            "total_disk_imgs": total_disk_imgs,
            "total_registered": reg_count,
            "reg_ratio": round(reg_ratio, 4),
            "warnings": warnings,
            "camera_reports": camera_reports,
            "all_redundant_frames": all_redundant_frames,
        }
        report_path = self.project / "sfm_quality_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"  [质检与冗余分析已保存] → {report_path}\n")

        return is_passed

    def _export_sparse_to_txt(self, sparse0: Path):
        '''将稀疏模型导出为 TXT 格式。'''
        out_dir = self.project / "sparse_txt"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._colmap(
            "model_converter",
            "--input_path", str(sparse0),
            "--output_path", str(out_dir),
            "--output_type", "TXT",
        )
        _run(cmd, step_name="export:TXT")
        print(f"  相机位姿 TXT 已导出 → {out_dir}")
        print(f"    cameras.txt  — 相机内参")
        print(f"    images.txt   — 每张图片的位姿 (qw,qx,qy,qz, tx,ty,tz)")
        print(f"    points3D.txt — 稀疏三维点")

    def _print_sparse_summary(self):
        '''打印稀疏重建摘要。'''
        out_dir = self.project / "sparse_txt"
        images_txt = out_dir / "images.txt"
        cameras_txt = out_dir / "cameras.txt"
        points_txt = out_dir / "points3D.txt"

        n_images = 0
        if images_txt.exists():
            with open(images_txt, "r") as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    n_images += 1
                    f.readline()

        n_cameras = 0
        if cameras_txt.exists():
            with open(cameras_txt, "r") as f:
                n_cameras = sum(1 for line in f if line.strip() and not line.startswith("#"))

        n_points = 0
        if points_txt.exists():
            with open(points_txt, "r") as f:
                n_points = sum(1 for line in f if line.strip() and not line.startswith("#"))

        print(f"\n{'─' * 40}")
        print(f"  稀疏重建摘要")
        print(f"{'─' * 40}")
        print(f"  注册图片: {n_images}")
        print(f"  相机模型: {n_cameras}")
        print(f"  稀疏点云: {n_points}")
        print(f"  位姿文件: {images_txt}")
        print(f"{'─' * 40}\n")

    def run_sfm(
        self,
        camera_model: str = "SIMPLE_RADIAL",
        camera_params: str | None = None,
        quality: str = "high",
        match_mode: str = "exhaustive",
        overlap: int = 10,
        single_camera: str = "1",
        fast_mapper: bool = True,
    ):
        '''一键 SfM（特征提取 + 匹配 + 稀疏重建）。'''
        print("=" * 50)
        print(f"SfM 管线开始 | 图片目录: {self.images}")
        print(f"               工作目录: {self.project}")
        print(f"               相机模式: single_camera={single_camera}")
        print(f"               相机先验: camera_params={camera_params}")
        print(f"               匹配模式: mode={match_mode}, overlap={overlap}")
        print(f"               极速建图: fast_mapper={fast_mapper}")
        print("=" * 50)
        self.project.mkdir(parents=True, exist_ok=True)
        self.extract_features(
            camera_model=camera_model,
            camera_params=camera_params,
            quality=quality,
            single_camera=single_camera,
        )
        self.match_features(mode=match_mode, overlap=overlap)
        self.sparse_recon(fast=fast_mapper)
        print("SfM 完成 [OK]")


    def image_undistorter(self):
        '''去畸变，为稠密重建准备图像（保持原始分辨率）。'''
        best_sparse = self._get_best_sparse_model()
        cmd = self._colmap(
            "image_undistorter",
            "--image_path", str(self.images),
            "--input_path", str(best_sparse),
            "--output_path", str(self.dense_dir),
            "--output_type", "COLMAP",
        )
        _run(cmd, step_name="MVS:undistort")

    def patch_match_stereo(
        self,
        gpu_index: int = 0,
        window_radius: int = 5,
        num_iterations: int = 3,
        max_image_size: int = -1,
        geom_consistency: bool = True,
    ):
        '''PatchMatch 稠密立体匹配（CUDA 加速）。'''
        cmd = self._colmap(
            "patch_match_stereo",
            "--workspace_path", str(self.dense_dir),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.gpu_index", str(gpu_index),
            "--PatchMatchStereo.window_radius", str(window_radius),
            "--PatchMatchStereo.num_iterations", str(num_iterations),
            "--PatchMatchStereo.max_image_size", str(max_image_size),
            "--PatchMatchStereo.geom_consistency", "1" if geom_consistency else "0",
        )
        _run(cmd, step_name="MVS:stereo")

    def stereo_fusion(
        self,
        output_ply: str = "fused.ply",
        input_type: str = "geometric",
    ):
        '''融合深度图生成稠密点云。'''
        out_path = self.project / output_ply
        cmd = self._colmap(
            "stereo_fusion",
            "--workspace_path", str(self.dense_dir),
            "--workspace_format", "COLMAP",
            "--input_type", input_type,
            "--output_path", str(out_path),
        )
        _run(cmd, step_name="MVS:fusion")
        print(f"稠密点云已保存 → {out_path}")
        return out_path

    def run_mvs(
        self,
        output_ply: str = "fused.ply",
        max_image_size: int = 1000,
        num_iterations: int = 3,
        geom_consistency: bool = True,
    ):
        '''一键 MVS 稠密重建（需要先完成 SfM）。'''
        print("=" * 50)
        print(f"MVS 稠密重建开始 [max_image_size={max_image_size}, num_iterations={num_iterations}, geom_consistency={geom_consistency}]")
        print("=" * 50)
        if self.dense_dir.exists():
            shutil.rmtree(self.dense_dir, ignore_errors=True)
        self.image_undistorter()
        self.patch_match_stereo(
            max_image_size=max_image_size,
            num_iterations=num_iterations,
            geom_consistency=geom_consistency,
        )
        input_type = "geometric" if geom_consistency else "photometric"
        return self.stereo_fusion(output_ply, input_type=input_type)


    def export(self, output_type: str = "PLY"):
        '''将稀疏模型导出为 PLY/TXT/NVM 等格式。'''
        input_sparse = self.sparse_dir / "0"
        out_file = self.project / f"sparse_model.{output_type.lower()}"
        cmd = self._colmap(
            "model_converter",
            "--input_path", str(input_sparse),
            "--output_path", str(out_file),
            "--output_type", output_type,
        )
        _run(cmd, step_name="export")
        print(f"模型已导出 → {out_file}")
        return out_file



def main():
    parser = argparse.ArgumentParser(
        description="p1_sfm_recon — COLMAP 三维重建流水线"
    )
    parser.add_argument(
        "images", type=Path,
        help="输入图片目录（也可是 p0 抽帧的输出目录）",
    )
    parser.add_argument(
        "-w", "--workspace", type=Path, default=None,
        help="工作目录（存放 database、sparse、dense），默认在图片目录旁创建",
    )
    parser.add_argument(
        "--step", choices=("features", "match", "sparse", "sfm",
                          "undistort", "stereo", "fusion", "mvs",
                          "export"),
        help="单步执行；不传则跑完整 SfM",
    )
    parser.add_argument(
        "--dense", action="store_true",
        help="SfM 完成后接续跑 MVS 稠密重建",
    )
    parser.add_argument(
        "--match-mode", choices=("exhaustive", "sequential", "vocab_tree", "dual_fast"),
        default="exhaustive",
        help="匹配策略：exhaustive(全两两), sequential(单序列), dual_fast(双相机极速时序匹配), vocab_tree",
    )
    parser.add_argument(
        "--fast", action="store_true", default=True,
        help="开启极速建图模式（优化 BA 频率与单模型追踪，提速 2~3 倍）",
    )
    parser.add_argument(
        "--camera-model", default="SIMPLE_RADIAL",
        help="相机模型，默认 SIMPLE_RADIAL",
    )
    parser.add_argument(
        "--camera-params", default=None,
        help="显式指定相机初始内参（例如 '1473.0,540,1200,0.06'），防止多视角退化漂移",
    )
    parser.add_argument(
        "--single-camera", choices=("1", "0", "per_folder", "auto"),
        default="auto",
        help="相机内参共享模式：1=全部共享(单相机), 0=各自标定(多相机混拍), "
             "per_folder=按子文件夹共享(双相机), auto=自动检测子文件夹结构。默认 auto",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="使用 sequential_matcher（视频序列推荐开启）",
    )
    parser.add_argument(
        "--overlap", type=int, default=10,
        help="时序匹配帧重叠数，默认 10",
    )
    parser.add_argument(
        "--mvs-size", type=int, default=1000,
        help="稠密重建时图片最大边长（默认 1000，1080P极速高质推荐；显存和计算量极低）",
    )
    parser.add_argument(
        "--mvs-iter", type=int, default=3,
        help="PatchMatch 迭代次数（默认 3，相比 5 提速 40%% 且表面精度无损）",
    )
    parser.add_argument(
        "--no-geom-consistency", action="store_true",
        help="关闭几何一致性校验（极致提速至 1-2 分钟，仅用于快速初筛）",
    )
    parser.add_argument(
        "--quality", choices=("low", "medium", "high", "extreme"),
        default="high",
        help="特征提取质量，默认 high",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU 索引，默认 0",
    )
    parser.add_argument(
        "--force-mvs", action="store_true",
        help="忽略 SfM 质量报警，强制继续执行 MVS 稠密重建",
    )
    parser.add_argument(
        "--colmap-bin", type=Path, default=None,
        help="colmap.exe 路径（自动检测）",
    )
    args = parser.parse_args()

    try:
        colmap_bin = args.colmap_bin or _resolve_colmap()
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"colmap → {colmap_bin}\n")

    images = args.images.resolve()
    if not images.is_dir():
        print(f"错误: 图片目录不存在 — {images}", file=sys.stderr)
        sys.exit(1)
    script_dir = Path(__file__).resolve().parent
    output_root = script_dir / "output"

    if args.workspace:
        workspace = args.workspace
    else:
        try:
            rel = images.relative_to(output_root)
            top = Path(rel.parts[0])
            if top.name.isdigit():
                workspace = output_root / top.name
                print(f"从图片路径推断工作目录: {workspace}")
            else:
                raise ValueError
        except (ValueError, IndexError):
            workspace = _next_output_dir(output_root)
            print(f"自动分配工作目录: {workspace}")

    pipe = ColmapPipeline(colmap_bin, workspace, images)

    def _safe_run_mvs():
        '''执行 MVS 前的安全质量卡点检查。'''
        if not pipe.verify_sfm_quality() and not args.force_mvs:
            print("\n🛑 [已拦截] SfM 质量质检未通过（轨迹存在断裂或未闭合）！")
            print("   系统已自动阻断 MVS 运行，防止浪费计算时间。")
            print("   请先根据质检报告修复稀疏建图，或追加 --force-mvs 强制执行。\n")
            sys.exit(1)
        pipe.run_mvs(
            max_image_size=args.mvs_size,
            num_iterations=args.mvs_iter,
            geom_consistency=not args.no_geom_consistency,
        )

    if args.single_camera == "auto":
        single_camera = _detect_camera_setup(images)
        print(f"自动检测相机模式: single_camera={single_camera}")
        if single_camera == "per_folder":
            print("  检测到多个含图片的子文件夹 → 按文件夹分组相机内参")
        else:
            print("  所有图片在同一目录 → 单相机模式")
    else:
        single_camera = args.single_camera

    match_mode = "sequential" if args.sequential else args.match_mode

    step = args.step

    if step == "features":
        pipe.extract_features(
            camera_model=args.camera_model,
            camera_params=args.camera_params,
            quality=args.quality,
            single_camera=single_camera,
        )
    elif step == "match":
        pipe.match_features(mode=match_mode, overlap=args.overlap, gpu_index=args.gpu)
    elif step == "sparse":
        pipe.sparse_recon()
    elif step == "sfm":
        pipe.run_sfm(
            camera_model=args.camera_model,
            camera_params=args.camera_params,
            quality=args.quality,
            match_mode=match_mode,
            overlap=args.overlap,
            single_camera=single_camera,
            fast_mapper=args.fast,
        )
    elif step == "undistort":
        pipe.image_undistorter()
    elif step == "stereo":
        pipe.patch_match_stereo(
            gpu_index=args.gpu,
            max_image_size=args.mvs_size,
            num_iterations=args.mvs_iter,
            geom_consistency=not args.no_geom_consistency,
        )
    elif step == "fusion":
        pipe.stereo_fusion(input_type="geometric" if not args.no_geom_consistency else "photometric")
    elif step == "mvs":
        _safe_run_mvs()
    elif step == "export":
        pipe.export()
    else:
        best_sparse = pipe._get_best_sparse_model()
        sfm_done = (best_sparse / "images.bin").exists()
        if sfm_done and not args.dense:
            print("SfM 已完成，无需重复运行。加 --dense 跑稠密重建。")
        elif sfm_done and args.dense:
            print("检测到已有稀疏模型，检查质量后进入 MVS\n")
            _safe_run_mvs()
        else:
            pipe.run_sfm(
                camera_model=args.camera_model,
                camera_params=args.camera_params,
                quality=args.quality,
                match_mode=match_mode,
                overlap=args.overlap,
                single_camera=single_camera,
                fast_mapper=args.fast,
            )
            if args.dense:
                _safe_run_mvs()

    print("done.")


if __name__ == "__main__":
    main()
