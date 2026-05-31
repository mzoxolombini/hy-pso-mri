# HyPSO-MRI

**Adaptive Hyper-Heuristic Control of Particle Swarm Optimisation for Multilevel Intensity Partitioning of Brain MRI Images**

> Mbini, M. & Nyathi, T. — Department of Computer Science, University of Pretoria, South Africa.

---

## Overview

This repository contains the research paper and Python implementations of **HyPSO** and **MCLPSO** for unsupervised, training-free segmentation of brain MRI images via multilevel thresholding.

HyPSO couples **canonical Particle Swarm Optimisation (PSO)** with a **meta-level hyper-heuristic controller** that adaptively selects among five hill-climbing local-search variants, applied exclusively to the global-best particle. PSO's velocity and position update equations remain entirely unchanged — the controller sits above the swarm as an independent, modular layer.

The objective function is **2D Rényi entropy** (order *q* = 0.5) computed on the joint histogram of the original image and its nonlocal-means-filtered counterpart, capturing both spectral and spatial information.

---

## Repository Contents

| File | Description |
|---|---|
| `hypso_mri_segmentation.py` | Full Python implementation of HyPSO |
| `mclpso_mri_segmentation.py` | Comprehensive Learning PSO (MCLPSO) implementation for multilevel MRI segmentation using the same 2D Rényi entropy objective |
| `Image_segmentation_using_Memetic_PSO (1).pdf` | Research paper (Mbini & Nyathi) |
| `README.md` | This file |

---

## Algorithm

### Architecture

```
┌─────────────────────────────────────────────┐
│         Hyper-Heuristic Controller          │
│  Choice Function  │  SHC │ SAHC │ StHC │   │
│  (reward: ΔH/Δt)  │ FCHC │ RRHC          │
└───────────────────┬─────────────────────────┘
                    │ applied to global best g
┌───────────────────▼─────────────────────────┐
│              Canonical PSO Swarm            │
│   p₁   p₂   g   p₄   p₅  …  pₙ            │
│  (velocity/position equations unchanged)    │
└─────────────────────────────────────────────┘
```

### Five Low-Level Heuristics (LLH pool)

| ID | Name | Description |
|---|---|---|
| SHC | Steepest-ascent HC | First-improvement, fixed scan order |
| SAHC | Steepest-Ascent HC | Best-improvement, exhaustive scan |
| StHC | Stochastic HC | Random neighbour; always accepts improvements, accepts worse moves with *p* = 0.05 |
| FCHC | First-Choice HC | Random sampling; first improvement; terminates after 10*(n−1) failures |
| RRHC | Random-Restart HC | Full SHC + random restart on stagnation (up to 3 restarts) |

### Hyperparameters (Table 1, paper)

| Parameter | Symbol | Default | Description |
|---|---|---|---|
| Swarm size | *N* | 50 | Number of PSO particles |
| Evaluation budget | *FE_max* | 10,000 × *D* | Total fitness evaluations |
| Inertia weight | *ω* | 0.729 | PSO inertia |
| Acceleration coefficients | *c₁, c₂* | 1.494 | Cognitive / social scaling |
| Rényi order | *q* | 0.5 | Entropy order parameter |
| Neighbourhood step | *δ* | 1 | Perturbation size for LLH |
| Local-search interval | *τ* | 2 | Iterations between LLH calls |
| Exploration probability | *ε* | 0.15 | Epsilon-greedy exploration rate |
| Reward window | *W* | 10 | Sliding window length for choice function |
| Max RRHC restarts | *R_max* | 3 | Random restarts per RRHC invocation |

---

## Installation

**Python 3.8+** is required. Only NumPy is strictly required; scikit-image and SciPy are optional but recommended.

```bash
# Clone the repository
git clone https://github.com/mzoxolombini/hy-pso-mri.git
cd hy-pso-mri

# Install dependencies
pip install numpy                          # required
pip install scikit-image scipy             # recommended (better NL-means filter + SSIM)
```

---

## Quick Start

### Run the built-in demo

```bash
python hypso_mri_segmentation.py
```

This runs HyPSO on a synthetic 128×128 multi-region image and prints the optimal thresholds, Rényi entropy, and SSIM score.

### Segment your own MRI image

```python
import numpy as np
from hypso_mri_segmentation import segment_mri, compute_ssim

# Load a grayscale MRI image (H x W, uint8)
# Example using PIL:
# from PIL import Image
# image = np.array(Image.open("brain_mri.png").convert("L"))

thresholds, segmented, fitness = segment_mri(
    image,
    n_thresholds=4,        # produces 5 intensity segments
    verbose=True,
)

print("Optimal thresholds:", thresholds.tolist())
print("2D Rényi entropy:  ", fitness)
print("SSIM:              ", compute_ssim(image, segmented))
```

### Use HyPSO directly on a precomputed histogram

```python
from hypso_mri_segmentation import (
    nonlocal_means_filter,
    build_joint_histogram,
    hypso,
)

filtered   = nonlocal_means_filter(image)
joint_hist = build_joint_histogram(image, filtered)

thresholds, fitness = hypso(
    joint_hist,
    D=4,            # number of thresholds
    N=50,           # swarm size
    FE_max=40_000,  # evaluation budget (= 10,000 * D)
    seed=42,        # for reproducibility
    verbose=True,
)
```

---

## API Reference

### `segment_mri(image, n_thresholds, hypso_kwargs, verbose)`
Full pipeline: filtering → joint histogram → HyPSO → label map.
- **image** — 2-D NumPy array (grayscale, uint8 or float)
- **n_thresholds** — number of threshold levels *D* (default 4)
- **hypso_kwargs** — dict of keyword arguments forwarded to `hypso()`
- Returns `(thresholds, segmented_image, best_fitness)`

### `hypso(joint_hist, D, N, FE_max, ...)`
Runs the HyPSO optimisation loop.
- **joint_hist** — 256×256 normalised joint histogram
- **D** — problem dimension (number of thresholds)
- **FE_max** — total function-evaluation budget (default `10_000 * D`)
- Returns `(best_thresholds, best_fitness)`

### `compute_ssim(image, segmented)`
Computes SSIM between the original image and the label map reconstructed with per-segment intensity means. Requires scikit-image.

---

## Evaluation

Results in the paper are reported as **reconstruction fidelity (SSIM)** — how well the thresholded image preserves the structure of the original. Experiments used:

- **Dataset:** 400 brain MRI images (Kaggle; no expert-annotated masks)
- **Threshold levels:** th ∈ {5, 10, 15, 20}
- **Trials:** 30 independent runs per configuration
- **Baselines:** PSO, MPSO, PSO-sono, EAPSO, FPSO, ACEPSO, SQPPSO, MLCPSO

HyPSO significantly outperformed PSO, MPSO, PSO-sono, EAPSO, and FPSO on SSIM (*p* < 0.05) and achieved statistical equivalence to ACEPSO and SQPPSO at 13–35% lower runtime.

---

## Authors

- **Mzoxolo Mbini** — [@mzoxolombini](https://github.com/mzoxolombini) — University of Pretoria
- **Thambo Nyathi** — University of Pretoria
