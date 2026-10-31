# FastPPP: A Fast Plant Phenotyping Pipeline via 3D Gaussian Splatting for High-Throughput Analysis of Potted Alfalfa





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


---

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
