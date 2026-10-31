#!/usr/bin/env python3
'''Compare COLMAP sparse reconstructions and verified image-pair graphs.'''

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "EXP" / "table1_sfm" / "results_high"
OUTPUT = ROOT / "EXP" / "table1_sfm" / "figures"
MAX_IMAGE_ID = 2**31 - 1
METHODS = (
    ("baseline_exhaustive_standard", "Baseline | Exhaustive + Standard"),
    ("ours_dual_fast", "Ours | Dual-Fast + Fast Mapper"),
)
REPRESENTATIVES = (
    ("healthy_sparse", "B2-Day0-7_67", "Healthy sparse"),
    ("healthy_dense", "B2-Day0-8_109", "Healthy dense"),
    ("wilted_sparse", "B2-Day9-79_272", "Wilted sparse"),
    ("wilted_dense", "B2-Day9-100_263-A", "Wilted dense"),
)
EDGE_COLORS = {
    "high-high": "#40d9e8",
    "low-low": "#ff9a52",
    "cross-ring": "#d879df",
}


@dataclass
class Reconstruction:
    centers: dict[str, np.ndarray]
    rings: dict[str, str]
    points: np.ndarray
    colors: np.ndarray
    edges: dict[str, np.ndarray]


def parse_images(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    centers: dict[str, np.ndarray] = {}
    rings: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        fields = line.split()
        if len(fields) < 10:
            i += 1
            continue
        qw, qx, qy, qz = map(float, fields[1:5])
        t = np.asarray(list(map(float, fields[5:8])))
        name = fields[9]
        rotation = np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
        ])
        low_name = name.lower()
        ring = "high" if "high" in low_name else "low" if "low" in low_name else "unknown"
        centers[name] = -rotation.T @ t
        rings[name] = ring
        i += 2
    return centers, rings


def parse_points(path: Path, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) >= 7:
                rows.append((float(fields[1]), float(fields[2]), float(fields[3]),
                             int(fields[4]), int(fields[5]), int(fields[6])))
    data = np.asarray(rows, dtype=np.float64)
    if len(data) > max_points:
        rng = np.random.default_rng(seed)
        data = data[np.sort(rng.choice(len(data), max_points, replace=False))]
    if not len(data):
        return np.empty((0, 3)), np.empty((0, 3))
    return data[:, :3], data[:, 3:6] / 255.0


def parse_verified_edges(db_path: Path, centers: dict[str, np.ndarray], rings: dict[str, str]):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        image_names = dict(con.execute("SELECT image_id, name FROM images"))
        pairs = con.execute(
            "SELECT pair_id FROM two_view_geometries WHERE rows > 0 AND config > 0"
        ).fetchall()
    finally:
        con.close()

    grouped = {key: [] for key in EDGE_COLORS}
    for (pair_id,) in pairs:
        image_id2 = pair_id % MAX_IMAGE_ID
        image_id1 = (pair_id - image_id2) // MAX_IMAGE_ID
        name1, name2 = image_names.get(image_id1), image_names.get(image_id2)
        if name1 not in centers or name2 not in centers:
            continue
        ring1, ring2 = rings.get(name1, "unknown"), rings.get(name2, "unknown")
        kind = "high-high" if ring1 == ring2 == "high" else (
            "low-low" if ring1 == ring2 == "low" else "cross-ring"
        )
        grouped[kind].append([centers[name1], centers[name2]])
    return {key: np.asarray(value, dtype=np.float64).reshape((-1, 2, 3))
            for key, value in grouped.items()}, len(pairs)


def load_reconstruction(category: str, sample: str, method: str, max_points: int, seed: int):
    folder = RESULTS / category / sample / method
    sparse = folder / "sparse_txt"
    centers, rings = parse_images(sparse / "images.txt")
    points, colors = parse_points(sparse / "points3D.txt", max_points, seed)
    edges, verified_count = parse_verified_edges(folder / "database.db", centers, rings)
    return Reconstruction(centers, rings, points, colors, edges), verified_count


def align_similarity(source: Reconstruction, target: Reconstruction) -> tuple[np.ndarray, np.ndarray, float]:
    common = sorted(set(source.centers) & set(target.centers))
    if len(common) < 3:
        raise RuntimeError(f"Need at least 3 common registered images for alignment; found {len(common)}")
    x = np.stack([source.centers[name] for name in common])
    y = np.stack([target.centers[name] for name in common])
    mx, my = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - mx, y - my
    u, singular, vt = np.linalg.svd(xc.T @ yc / len(x), full_matrices=False)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    variance = np.mean(np.sum(xc * xc, axis=1))
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    translation = my - scale * (mx @ rotation)
    return rotation, translation, scale


def transform_reconstruction(rec: Reconstruction, rotation: np.ndarray, translation: np.ndarray, scale: float):
    rec.centers = {name: scale * (point @ rotation) + translation
                   for name, point in rec.centers.items()}
    rec.points = scale * (rec.points @ rotation) + translation
    for kind, lines in rec.edges.items():
        rec.edges[kind] = scale * (lines @ rotation) + translation


def ring_order(name: str) -> tuple[int, int, str]:
    import re
    match = re.search(r"_(\d+)(?:\.[^.]+)?$", name)
    number = int(match.group(1)) if match else 0
    ring = 0 if "high" in name.lower() else 1
    return ring, number, name


def draw_reconstruction(ax, rec: Reconstruction, title: str, n_pairs: int, point_size: float):
    if len(rec.points):
        ax.scatter(*rec.points.T, c=rec.colors, s=point_size, alpha=0.62,
                   linewidths=0, depthshade=False, rasterized=True)

    for kind, segments in rec.edges.items():
        if len(segments):
            collection = Line3DCollection(segments, colors=EDGE_COLORS[kind],
                                          linewidths=0.45, alpha=0.12)
            ax.add_collection3d(collection)

    centers_by_ring = {"high": [], "low": []}
    for name, center in rec.centers.items():
        ring = rec.rings.get(name, "unknown")
        color = "#35d6e8" if ring == "high" else "#ff9a52" if ring == "low" else "#eeeeee"
        ax.scatter(*center, c=color, s=12, marker="o", edgecolors="#111111",
                   linewidths=0.25, depthshade=False)
        if ring in centers_by_ring:
            centers_by_ring[ring].append((name, center))

    for ring, color in (("high", "#35d6e8"), ("low", "#ff9a52")):
        ordered = sorted(centers_by_ring[ring], key=lambda item: ring_order(item[0]))
        if len(ordered) > 1:
            trajectory = np.stack([center for _, center in ordered])
            ax.plot(*trajectory.T, color=color, linewidth=1.8, alpha=1.0)

    ax.set_title(f"{title}\n{len(rec.centers)} cameras | {n_pairs:,} verified edges", fontsize=10, pad=8)
    ax.title.set_color("#f2f5f8")
    ax.set_xlabel("X", labelpad=-4)
    ax.set_ylabel("Y", labelpad=-4)
    ax.set_zlabel("Z", labelpad=-4)
    ax.tick_params(labelsize=7, pad=-2)
    ax.xaxis.label.set_color("#d5dbe2")
    ax.yaxis.label.set_color("#d5dbe2")
    ax.zaxis.label.set_color("#d5dbe2")
    ax.tick_params(colors="#b8c0ca")
    ax.grid(False)
    ax.set_facecolor("#11151b")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.07, 0.09, 0.12, 1.0))
        axis.pane.set_edgecolor((0.30, 0.34, 0.38, 0.35))
    ax.view_init(elev=25, azim=-55)


def set_common_limits(axes, reconstructions: list[Reconstruction]):
    all_points = [rec.points for rec in reconstructions if len(rec.points)]
    all_points += [np.stack(list(rec.centers.values())) for rec in reconstructions if rec.centers]
    points = np.vstack(all_points)
    lo, hi = np.percentile(points, [1.0, 99.0], axis=0)
    mid = (lo + hi) / 2
    half = max(float(np.max(hi - lo)) / 2, 1e-3) * 1.08
    for ax in axes:
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)
        ax.set_box_aspect((1, 1, 1))


def render_one(category: str, sample: str, label: str, max_points: int, point_size: float, out_dir: Path):
    baseline, baseline_edges = load_reconstruction(category, sample, METHODS[0][0], max_points, 1)
    ours, ours_edges = load_reconstruction(category, sample, METHODS[1][0], max_points, 2)
    rotation, translation, scale = align_similarity(ours, baseline)
    transform_reconstruction(ours, rotation, translation, scale)

    fig = plt.figure(figsize=(13.2, 6.2), facecolor="#090c10")
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d", facecolor="#11151b") for i in range(2)]
    fig.suptitle(f"{label} | {sample}", color="white", fontsize=14, y=0.98)
    draw_reconstruction(axes[0], baseline, METHODS[0][1], baseline_edges, point_size)
    draw_reconstruction(axes[1], ours, METHODS[1][1], ours_edges, point_size)
    set_common_limits(axes, [baseline, ours])
    fig.text(0.5, 0.025,
             "Verified image-pair links: cyan high-ring | orange low-ring | magenta cross-ring   "
             "Camera trajectories: bright ring-colored paths   |   Point cloud: COLMAP RGB",
             ha="center", color="#d5dbe2", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.89, bottom=0.10, wspace=0.02)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{category}_{sample}_match_graph.png"
    fig.savefig(out, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out, baseline, ours, baseline_edges, ours_edges


def render_overview(rows, out_dir: Path):
    fig = plt.figure(figsize=(13.2, 23.5), facecolor="#090c10")
    axes = []
    row_axes = []
    row_recs = []
    for row_idx, (label, sample, baseline, ours, base_count, ours_count) in enumerate(rows):
        for col_idx, (rec, title, count) in enumerate(((baseline, METHODS[0][1], base_count),
                                                        (ours, METHODS[1][1], ours_count))):
            ax = fig.add_subplot(4, 2, row_idx * 2 + col_idx + 1, projection="3d", facecolor="#11151b")
            draw_reconstruction(ax, rec, f"{label} | {title}", count, 0.20)
            axes.append(ax)
        row_axes.append(axes[-2:])
        row_recs.append((baseline, ours))
    for pair_axes, pair_recs in zip(row_axes, row_recs):
        set_common_limits(pair_axes, list(pair_recs))
    fig.suptitle("COLMAP Match Graph: Baseline vs Dual-Fast", color="white", fontsize=16, y=0.995)
    fig.text(0.5, 0.004,
             "Verified image-pair links: cyan high-ring | orange low-ring | magenta cross-ring; "
             "camera paths are bright; sparse points use COLMAP RGB",
             ha="center", color="#d5dbe2", fontsize=9)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.025, hspace=0.13, wspace=0.02)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "table1_four_classes_match_graph_overview.png"
    fig.savefig(out, dpi=145, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--max-points", type=int, default=18000)
    parser.add_argument("--point-size", type=float, default=0.55)
    parser.add_argument("--categories", nargs="+", choices=[x[0] for x in REPRESENTATIVES],
                        default=[x[0] for x in REPRESENTATIVES])
    args = parser.parse_args()

    rendered = []
    for category, sample, label in REPRESENTATIVES:
        if category not in args.categories:
            continue
        out, baseline, ours, base_count, ours_count = render_one(
            category, sample, label, args.max_points, args.point_size, args.output_dir
        )
        rendered.append((label, sample, baseline, ours, base_count, ours_count))
        print(f"{out} | cameras={len(baseline.centers)} | verified pairs={base_count}/{ours_count}")
    if len(rendered) == len(REPRESENTATIVES):
        print(render_overview(rendered, args.output_dir))


if __name__ == "__main__":
    main()
