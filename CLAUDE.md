# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SeaSplat is a research implementation for real-time rendering of underwater scenes using 3D Gaussian Splatting (3DGS) constrained with a physically grounded underwater image formation model. It extends the [INRIA 3DGS codebase](https://github.com/graphdeco-inria/gaussian-splatting) with underwater-specific attenuation and backscatter modeling (the "DeepSeeColor" module).

## Setup

Requires CUDA-capable GPU, Python 3.10, and PyTorch with CUDA support.

```bash
conda create --name seasplat_py310 -y python=3.10
conda activate seasplat_py310
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

Docker alternative: `docker build --tag seasplat --build-arg USER_ID=$(id -u) -f Dockerfile .`

## Key Commands

### Training
```bash
python train.py -s DATASET_PATH --exp EXPERIMENT_NAME --do_seathru --seathru_from_iter 10000
```
`DATASET_PATH` must contain `images/` (undistorted) and `sparse/0/` (COLMAP outputs). Output goes to `DATASET_PATH/experiments/MMDDYYYY/EXPERIMENT_NAME/`.

### Rendering from trained model
```bash
python render_uw.py -m MODEL_PATH --seathru [--skip_train] [--skip_test]
```

### Evaluation metrics
```bash
python metrics.py -m MODEL_PATH [--comparison_dir with_water|render] [--skip_train]
```

### Data preparation (COLMAP)
```bash
python convert.py -s SOURCE_PATH [--camera OPENCV] [--resize]
```

## Architecture

The codebase follows this data flow during training:

1. **Scene loading** (`scene/`): COLMAP or Blender data is loaded via `dataset_readers.py`, cameras are set up via `cameras.py`, and a `GaussianModel` is initialized from the point cloud.

2. **Rendering** (`gaussian_renderer/`): `render()` rasterizes Gaussians to RGB+alpha via the CUDA `diff-gaussian-rasterization` submodule. `render_depth()` uses the same rasterizer but overrides colors with camera-space Z coordinates to produce depth maps.

3. **Underwater image formation** (`deepseecolor/`): After `seathru_from_iter` iterations, the underwater model activates:
   - `BackscatterNetV2`: models `B_inf * (1 - exp(-beta_b * z))` — depth-dependent scattering
   - `AttenuateNetV3` (default): models `exp(-beta_d * z)` — depth-dependent color attenuation
   - Combined: `underwater_image = clamp(J * attenuation + backscatter, 0, 1)`
   - These small networks have separate optimizers and are updated on an alternating schedule with the Gaussians (`update_bs_at_interval` / `update_bs_at_count`)

4. **Loss functions** (`deepseecolor/losses.py`, `deepseecolor/depth_losses.py`, `utils/loss_utils.py`):
   - Standard 3DGS: L1 + D-SSIM reconstruction loss
   - Dark channel prior (`DarkChannelPriorLossV3`): encourages haze-free restored images
   - Gray world prior: pushes restored scene toward balanced color channels
   - Depth smoothness, alpha background, RGB saturation, and attenuation losses
   - All controlled by `OptimizationParams` flags and lambdas

5. **Post-training**: renders train/test splits, computes PSNR/SSIM/LPIPS, saves to `eval_metrics.json`.

## Key Configuration

All training hyperparameters are defined in `arguments/__init__.py` via `ModelParams`, `OptimizationParams`, and `PipelineParams`. Important non-obvious defaults:

- `sh_degree = 0` (not 3 as in original 3DGS)
- `learn_background = True` with `bg_from_bs = True` (background learning hands off to backscatter model once SeaThru activates)
- `use_at_v3 = True` (simplest attenuation model is default)
- `seathru_from_iter = 9_000_000` (effectively disabled unless `--seathru_from_iter` is passed)
- `do_seathru = False` (must pass `--do_seathru` to enable underwater model)

## CUDA Submodules

- `submodules/diff-gaussian-rasterization`: Modified differentiable Gaussian rasterizer (returns alpha channel alongside RGB)
- `submodules/simple-knn`: CUDA KNN for initial Gaussian scale estimation

These must be installed as pip packages (`pip install submodules/...`) and require a CUDA build toolchain.
