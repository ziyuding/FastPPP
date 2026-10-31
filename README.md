# FastPPP: A Fast Plant Phenotyping Pipeline via 3D Gaussian Splatting for High-Throughput Analysis of Potted Alfalfa

This repository contains the official implementation of the paper:

> **FastPPP: A Fast Plant Phenotyping Pipeline via 3D Gaussian Splatting for High-Throughput Analysis of Potted Alfalfa**

---

## News

- **[Oct 2026]** Our paper has been submitted.
- **[Oct 2026]** This repository is now populated with the algorithmic components
  described in the paper, together with four sample plant point clouds.

## Overview

FastPPP is an end-to-end 3D reconstruction and high-throughput phenotyping
pipeline tailored for large-scale potted plant populations.

| Folder | What it does |
|---|---|
| `Video Sampling` | 1080P video -> frames at a fixed interval |
| `Sparse Reconstruction` | **Dual-FastSfM**: circular-topology image pairing, then incremental SfM. `visualization/` holds the match-matrix and co-visibility viewers |
| `Gaussian Splatting` | 3DGS training with **ABBS** and **VCP** (see the README inside) |
| `Scale and Segmentation` | 30 mm reference cube -> millimetre scale, ground alignment, pot / soil removal |

Trait computation (canopy volume, plant height, leaf length and width, green
ratio, fresh-biomass inversion) follows section 2.5 of the paper and is not
released here.

---

## Performance

20 scenes (healthy / wilted x sparse / dense), 63 views each, half resolution,
10,000 iterations.

| Method | PSNR (dB) | SSIM | LPIPS | Time (s/pot) | Gaussians |
|---|---:|---:|---:|---:|---:|
| 2DGS | 23.99 | 0.870 | 0.064 | 273.3 | 290k |
| 3DGS | 26.58 | 0.916 | 0.036 | 226.1 | 646k |
| **FastPPP** | **26.56** | **0.928** | 0.039 | **115.3** | **335k** |

Against 3DGS: PSNR unchanged, SSIM +0.012, **1.96x faster**, **48 % fewer primitives**.

---

## Cohort

1,116 pot-sessions = 186 pots x 6 dates (D0, D1, D3, D5, D7, D9),
71 control and 115 salt-stressed plants.

---

## Sample point clouds

Four cleaned, plant-only clouds - the same four shown in the paper - are in
`sample_pointclouds/`, one binary `plant.ply` per folder
(`x y z red green blue`, millimetres, Z = 0 on the table plane).

```
sample_pointclouds/
  healthy_dense/    plant.ply   42,398 vertices
  healthy_sparse/   plant.ply    5,258 vertices
  wilted_dense/     plant.ply   55,385 vertices
  wilted_sparse/    plant.ply    7,847 vertices
```

---

## Data and full pipeline availability

The 1,116-session dataset, trained checkpoints, the batch orchestration scripts
and the trait-computation module are **not** included. They are available from
the corresponding author upon reasonable request, subject to institutional
data-sharing policy.

---

## Acknowledgement

`Gaussian Splatting/FastGS-master` is a modified copy of
[FastGS](https://github.com/fastgs/FastGS) (arXiv:2511.04283), which itself builds
on [3DGS](https://github.com/graphdeco-inria/gaussian-splatting),
[DropGaussian](https://github.com/DCVL-3D/DropGaussian_release),
[Taming-3DGS](https://github.com/humansensinglab/taming-3dgs),
[Speedy-Splat](https://github.com/j-alex-hanson/speedy-splat) and
[AbsGS](https://github.com/TY424/AbsGS). Only the ABBS tile multiplier and the
VCP pruning hooks are new; everything else follows the upstream code.

---

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{fastppp2026,
  title   = {FastPPP: A Fast Plant Phenotyping Pipeline via 3D Gaussian
             Splatting for High-Throughput Analysis of Potted Alfalfa},
  author  = {<authors>},
  journal = {Plant Phenomics},
  year    = {2026},
  doi     = {<doi>}
}
```

---

*For any inquiries regarding the paper or code, please contact the authors.*
