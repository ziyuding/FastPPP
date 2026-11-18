# FastPPP: A Fast Plant Phenotyping Pipeline via 3D Gaussian Splatting for High-Throughput Analysis of Potted Alfalfa

## Overview

FastPPP is a fast 3D reconstruction and high-throughput phenotyping pipeline
tailored for large-scale potted plant populations.

| Folder | What it does |
|---|---|
| `Video Sampling` | 1080P video -> frames at a fixed interval |
| `Sparse Reconstruction` | **Dual-FastSfM**: circular-topology image pairing, then incremental SfM. `visualization/` holds the match-matrix and co-visibility viewers |
| `Gaussian Splatting` | 3DGS training with **ABBS** and **VCP** (see the README inside) |
| `Scale` | 30 mm reference cube -> millimetre scale and ground alignment |

The released code ends at a calibrated, ground-aligned point cloud. Everything
downstream of that — plant extraction (pot wall, soil surface, label tags, black
cloth), trait computation and biomass inversion — follows section 2.5 of the
paper and is **not** released here.

## Collaboration

We work with groups using the same acquisition rig. The usual arrangement is
straightforward: you capture, we run the full reconstruction-and-phenotyping
pipeline on your data and return the calibrated point clouds together with the
trait tables.

If you are interested, contact the corresponding author with a short description
of your experiment (species, pot geometry, number of plants, number of dates).

## Acknowledgement

`Gaussian Splatting/FastGS-master` is a modified copy of
[FastGS](https://github.com/fastgs/FastGS) (arXiv:2511.04283), which itself builds
on [3DGS](https://github.com/graphdeco-inria/gaussian-splatting).

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
