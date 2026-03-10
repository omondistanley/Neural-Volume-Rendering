# Neural Volume Rendering

A PyTorch implementation of **differentiable volume rendering** and **neural surface rendering** for 3D scene representation and inverse rendering. The pipeline runs locally with PyTorch3D and produces RGB images, depth maps, and spiral renderings from implicit volumes or SDFs.

## Overview

This repository implements:

- **Emission-absorption (EA) volume rendering** — Ray generation from cameras, stratified point sampling along rays, and transmittance-weighted color/depth aggregation
- **Differentiable inverse rendering** — Optimizing implicit volume parameters (e.g. box pose and size) from image supervision via gradient descent
- **Neural Radiance Fields (NeRF)** — MLP-based implicit volume with positional encoding and optional view-dependent color
- **Neural surface rendering** — Sphere tracing for SDFs, neural SDF from point clouds with eikonal regularization, and VolSDF (SDF-to-density conversion for volume rendering)

No external APIs; everything runs on your machine after environment setup.

---

## Pipeline Architecture

```
Input: cameras + image size (or dataset)
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│  Ray generation (ray_utils.py)                               │
│  get_pixels_from_image → get_rays_from_pixels                │
│  NDC [-1, 1] → world-space ray origins & directions          │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  Point sampling (sampler.py)                                 │
│  StratifiedRaysampler: sample distances [near, far]          │
│  sample_points = origins + directions × t                    │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  Implicit evaluation (implicit.py)                           │
│  SDFVolume / NeuralRadianceField / NeuralSurface             │
│  → density, feature (color) per sample point                 │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  Volume rendering (renderer.py)                              │
│  _compute_weights: transmittance T, alpha, weights           │
│  _aggregate: weighted sum → color, depth                     │
│  Output: RGB image, depth map                                │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
              images/ (GIFs, depth, visualizations)
```

**Surface path (sphere tracing):** same ray generation → `SphereTracingRenderer.sphere_tracing` → SDF queries → intersection points → `get_color` → output image.

---

## Setup

### Requirements

- Python 3.x (see `environment.yml`)
- PyTorch, PyTorch3D
- CUDA-capable GPU recommended for training and NeRF
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)

### Install

```bash
conda env create -f environment.yml
conda activate l3d
```

### Data

- Core data is under `data/`.
- For the **materials** scene (NeRF/VolSDF), download the zip from [Google Drive](https://drive.google.com/file/d/1v_0w1bx6m-SMZdqu3IFO71FEsu-VJJyb/view?usp=sharing) and unzip into `data/`.

---

## Quick Start

| Task | Command |
|------|--------|
| Volume render (box) | `python volume_rendering_main.py --config-name=box` |
| Train box (inverse rendering) | `python volume_rendering_main.py --config-name=train_box` |
| Train NeRF (lego) | `python volume_rendering_main.py --config-name=nerf_lego` |
| Sphere tracing (torus) | `python -m surface_rendering_main --config-name=torus_surface` |
| Train neural SDF (point cloud) | `python -m surface_rendering_main --config-name=points_surface` |
| VolSDF (lego) | `python -m surface_rendering_main --config-name=volsdf_surface` |

Outputs (GIFs, depth, visualizations) are written to `images/`.

---

## Repository Structure

| Path | Role |
|------|------|
| `volume_rendering_main.py` | Entry point: render, train box, train NeRF |
| `surface_rendering_main.py` | Entry point: sphere tracing, neural SDF, VolSDF |
| `ray_utils.py` | Pixel grids, ray generation, random pixel sampling |
| `sampler.py` | Stratified point sampling along rays |
| `renderer.py` | VolumeRenderer, SphereTracingRenderer, VolSDF renderer, sdf_to_density |
| `implicit.py` | SDFVolume, NeuralRadianceField, NeuralSurface, SDF primitives |
| `losses.py` | Eikonal loss, sphere loss, sampling helpers |
| `configs/*.yaml` | Hydra configs (box, train_box, nerf_lego, torus_surface, etc.) |
| `notes.md` | Detailed implementation notes and references |

---

## Features

- **Volume rendering:** Transmittance-based weights, alpha compositing, color and depth output
- **SDF primitives:** Sphere, box, torus; extensible via [SDF formulas](https://iquilezles.org/articles/distfunctions/)
- **NeRF:** Positional encoding (`HarmonicEmbedding`), skip connections, view-dependent color
- **Neural SDF:** MLP with eikonal regularization; mesh extraction via marching cubes for visualization
- **VolSDF:** SDF-to-density conversion (alpha/beta); joint geometry and color from images

