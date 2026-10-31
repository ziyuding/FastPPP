#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Figure 1 column B -- 63x63 matching adjacency matrices (band-matrix view).'''
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from p2_sfm_graph_viewer import (
    align_ours_to_baseline, compute_upright_transform, parse_images, qvec2rotmat,
)

RESULTS_ROOT = REPO_ROOT / "EXP" / "table1_sfm" / "results_high"
OUT_DEFAULT = REPO_ROOT / "EXP" / "FINAL_PAPER_ASSETS" / "figure1_panels"
METHODS = [("baseline_exhaustive_standard", "baseline", "Exhaustive matching"),
           ("ours_dual_fast", "ours", "Dual-Fast matching")]
NAVY = "#1B365D"
CMAP = LinearSegmentedColormap.from_list("w2navy", ["#FFFFFF", NAVY])


def build(category: str, sample: str, method_dir: str, value: str,
          order: str) -> tuple[np.ndarray, list[str], dict]:
    d = RESULTS_ROOT / category / sample / method_dir
    base = RESULTS_ROOT / category / sample / "baseline_exhaustive_standard"
    R_up, _ = compute_upright_transform(base / "sparse_txt")
    imgs = parse_images(d / "sparse_txt/images.txt")

    pre = None
    if method_dir == "ours_dual_fast":
        ib = parse_images(base / "sparse_txt/images.txt")
        io = imgs
        cb = {im["image_id"]: -qvec2rotmat(im["qvec"]).T @ im["tvec"] for im in ib}
        co = {im["image_id"]: -qvec2rotmat(im["qvec"]).T @ im["tvec"] for im in io}
        pre = align_ours_to_baseline(co, cb)

    centers = {}
    for im in imgs:
        v = -qvec2rotmat(im["qvec"]).T @ im["tvec"]
        if pre:
            Rp, tp, sp = pre
            v = sp * (v @ Rp) + tp
        centers[im["image_id"]] = v @ R_up.T

    if order == "azimuth":
        def key(im):
            x, y, _ = centers[im["image_id"]]
            return math.atan2(y, x)
    else:
        def key(im):
            return im["image_id"]
    ordered = sorted(imgs, key=key)
    index = {im["image_id"]: k for k, im in enumerate(ordered)}
    names = [im["name"].split("/")[-1] for im in ordered]

    N = len(ordered)
    M = np.zeros((N, N), dtype=float)
    con = sqlite3.connect(d / "database.db")
    if value == "raw":
        rows = con.execute("SELECT pair_id, rows FROM matches WHERE rows > 0").fetchall()
    else:
        rows = con.execute("SELECT pair_id, rows FROM two_view_geometries "
                           "WHERE rows > 0 AND config > 0").fetchall()
    con.close()
    MAX_ID = 2 ** 31 - 1
    used = 0
    for pid, n in rows:
        j = pid % MAX_ID
        i = (pid - j) // MAX_ID
        if i not in index or j not in index:
            continue
        a, b = index[i], index[j]
        M[a, b] = M[b, a] = max(M[a, b], n)
        used += 1
    stats = {"n_images": N, "pairs": used, "sum_matches": int(M.sum() / 2),
             "max": int(M.max()), "nonzero": int((M > 0).sum() // 2)}
    return M, names, stats


def draw(M: np.ndarray, out: Path, title: str, stats: dict, N: int) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 6.8), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    vmax = float(np.percentile(M[M > 0], 97)) if (M > 0).any() else 1.0
    ax.imshow(np.ma.masked_where(M <= 0, M), cmap=CMAP, vmin=0, vmax=vmax,
              interpolation="nearest")
    ax.set_xlim(-0.5, N - 0.5); ax.set_ylim(N - 0.5, -0.5)
    ticks = list(range(0, N, 5))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([str(t + 1) for t in ticks], fontsize=10)
    ax.set_yticklabels([str(t + 1) for t in ticks], fontsize=10)
    ax.set_xlabel("Image View Index (1 to %d)" % N, fontsize=12)
    ax.set_ylabel("Image View Index (1 to %d)" % N, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    for sp in ax.spines.values():
        sp.set_color("0.75"); sp.set_linewidth(0.8)
    ax.tick_params(length=3, color="0.6")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out.name}   {stats['pairs']} pairs / {stats['nonzero']} lit cells "
          f"/ {stats['sum_matches']} matches / max {stats['max']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default="healthy_dense")
    ap.add_argument("--sample", default="B2-Day0-8_109")
    ap.add_argument("--value", choices=("raw", "verified"), default="raw",
                    help="raw = matches found by the matcher (what was compared); "
                         "verified = only geometrically verified inliers")
    ap.add_argument("--order", choices=("azimuth", "file"), default="azimuth",
                    help="azimuth (default) sorts views around the turntable so that "
                         "neighbouring indices are neighbouring viewpoints")
    ap.add_argument("--outdir", default=str(OUT_DEFAULT))
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    for method_dir, tag, label in METHODS:
        M, names, stats = build(args.category, args.sample, method_dir,
                                args.value, args.order)
        title = (f"{label} — {args.sample}\n"
                 f"{stats['pairs']:,} compared pairs · {stats['sum_matches']:,} matches")
        draw(M, out / f"fig1_match_matrix_{tag}.png", title, stats, stats["n_images"])


if __name__ == "__main__":
    main()
