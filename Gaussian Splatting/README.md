# Gaussian Splatting

`FastGS-master/` is a modified copy of [FastGS](https://github.com/fastgs/FastGS)
(arXiv:2511.04283). The training framework, the CUDA rasteriser and the
densification schedule come from upstream and stay under their original licence
(`FastGS-master/LICENSE_GAUSSIAN_SPLATTING.md`).

Two modules are introduced in this work: **ABBS** and **VCP**.

---

## Adaptive Bounding-Box Splatting (ABBS)

Standard 3DGS projects each primitive to an axis-aligned bounding box built from
a fixed 3σ radius, so elongated stems and inclined leaflets touch a large number
of background tiles that contribute no gradient.

ABBS derives the truncation radius from the primitive opacity and the two
screen-space variances of the projected covariance:

```
r_eff(k) = sqrt( 2 * ln( alpha_k / tau_alpha ) ),   alpha_k > tau_alpha
tau_alpha = 1/255
```

The AABB is then
`[mu_x - r_eff*sx, mu_x + r_eff*sx] x [mu_y - r_eff*sy, mu_y + r_eff*sy]`.
Low-opacity primitives get `r_eff < 3`, the box shrinks along the major axis of
the ellipse and the per-tile depth-sorted list gets shorter.

Implementation:

```
arguments/__init__.py             mult          (tile shrinkage factor, 0.5)
gaussian_renderer/__init__.py     raster_settings(mult=...)
submodules/diff-gaussian-rasterization_fastgs/   applies mult to the tile range
```

Measured effect: -9.5 % wall clock overall, up to -14.2 % on sparse wilted
canopies, with a PSNR change of -0.02 dB (inside the run-to-run spread).

## View-Consistent Pruning (VCP)

Density control in standard 3DGS is driven by single-view photometric loss
alone, which lets translucent floaters survive at foreground/background
boundaries. VCP scores every primitive by how inconsistent it is across views.

Each density-control interval samples training views, renders them, and computes
the pixel-wise L1 error `e(u,v) = || C_hat(u,v) - C_gt(u,v) ||_1`. Pixels above
`tau_loss` are marked high-error; the differentiable rasteriser then propagates
that error back onto the contributing primitives with their rasterisation
weights `omega_k(u,v)`, together with the flagged frequency `N_flag(k)`:

```
S_vcp(k) = Norm( sum_n sum_(u,v) in Omega_n  I(e > tau_loss)
                 * omega_k(u,v) * e(u,v) * N_flag(k) )
```

`Norm` is min-max over the current pool, so `S_vcp` lies in [0, 1]. A high score
means the primitive fits one viewpoint while damaging the others — the signature
of an edge floater. The score is folded into the routine density-control prune,
and a *strong* prune additionally drops primitives with
`alpha_k < alpha_min` or `S_vcp(k) > tau_s`.

Implementation:

```
utils/fast_utils.py             compute_gaussian_score_fastgs()
scene/gaussian_model.py         densify_and_prune_fastgs(pruning_score=...)
                                final_prune_fastgs(min_opacity, pruning_score, score_thresh)
train.py                        calls both inside the training loop
```

Measured effect: 522k -> 335k primitives (-36 %), -8.4 % wall clock, at a cost
of 0.39 dB PSNR. Over the 1,116-session cohort this removes roughly 42 GB of
model storage.

### Hyperparameters

| Symbol | Value | Meaning |
|---|---|---|
| `tau_loss` | 0.1 | high-error pixel L1 threshold |
| `tau_s` | 0.9 | consistency-score prune threshold |
| `alpha_min` | 0.1 | minimum opacity for strong prune |
| `tau_alpha` | 1/255 | visibility threshold used by ABBS |
| mult | 0.5 | ABBS tile shrinkage factor |
| density control | start 500, every 500 | routine prune cadence |
| strong prune | every 8,000 | threshold-based elimination |

---

## Not included

Training data, trained checkpoints, the batch orchestration for the 1,116
pot-sessions, and the ablation builds (`--no_vcp`, ABBS disabled). Available
from the corresponding author on request.

---

## Environment

Python 3.8, PyTorch 2.1.2 + CUDA 11.8, single RTX 3090 (24 GB). The rasteriser
must be compiled from `submodules/diff-gaussian-rasterization_fastgs` — note the
`_fastgs` suffix, it is not the original 3DGS package.

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

The Gaussian Splatting backbone should also be cited:

```bibtex
@article{fastgs2025,
  title   = {FastGS: Training 3D Gaussian Splatting in 100 Seconds},
  author  = {<FastGS authors>},
  journal = {arXiv preprint arXiv:2511.04283},
  year    = {2025}
}
```
