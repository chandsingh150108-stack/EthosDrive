# ASP Safety Layer — Full Setup Guide

This guide walks you through setting up everything needed to run the ASP Safety Layer project, including the neural planner (SparseDrive) with real CUDA inference.

**Who is this for?** Team members with an RTX 4050 laptop (6GB VRAM) running Linux or Windows with WSL2.

**How long does it take?** About 30–60 minutes depending on your internet speed.

---

## Before You Start

Make sure you have:
- An NVIDIA GeForce RTX 4050 laptop (6GB VRAM)
- At least 10 GB free disk space
- Internet connection
- A terminal / command line

---

## Step 1: Check Your GPU is Visible

Run this command:
```bash
nvidia-smi
```

**What this does:** Checks if your GPU is detected by the operating system.

**What you should see:** A table showing your GPU name (`NVIDIA GeForce RTX 4050`), driver version, and CUDA version.

### If you see an error

| Error | What it means | How to fix it |
|-------|---------------|---------------|
| `nvidia-smi: command not found` | NVIDIA driver not installed | Download and install from https://www.nvidia.com/download/index.aspx (select GeForce RTX 4050, your OS) |
| `NVIDIA-SMI has failed...` | Driver installed but GPU not accessible | On Linux: run `sudo modprobe nvidia`. On Windows: restart your computer. On WSL2: make sure you have the Windows NVIDIA driver installed (not the Linux one) |
| GPU shows but CUDA version is 12.x | Driver is newer than CUDA 11.8 | This is fine! CUDA 11.8 works with newer drivers. Continue to Step 2 |

---

## Step 2: Install Miniconda

Miniconda lets you create isolated Python environments so different projects don't conflict.

**On Linux:**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

**On Windows (WSL2):**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Follow the prompts. When it asks about adding conda to your PATH, say **yes**.

After installation, close and reopen your terminal, then verify:
```bash
conda --version
```

**What you should see:** `conda 24.x.x` or similar version number.

### If you see an error

| Error | How to fix it |
|-------|---------------|
| `conda: command not found` after install | Run: `source ~/.bashrc` (Linux) or restart your terminal |
| Still not found | Run: `export PATH="$HOME/miniconda3/bin:$PATH"` then try again |

---

## Step 3: Create the Project Environment

This creates an isolated Python 3.10 environment for the project.

```bash
conda create -n asp-safety python=3.10 -y
```

Then activate it:
```bash
conda activate asp-safety
```

**Verify:**
```bash
python --version
```

**What you should see:** `Python 3.10.x`

> **Important:** You must run `conda activate asp-safety` every time you open a new terminal before working on this project.

### If you see an error

| Error | How to fix it |
|-------|---------------|
| `CondaError: The target prefix is the base prefix` | Run `conda deactivate` first, then try again |
| Python version shows 3.13 or other | The environment wasn't created properly. Run: `conda env remove -n asp-safety` then repeat this step |

---

## Step 4: Install CUDA 11.8 Toolkit

This installs the CUDA compiler (`nvcc`) needed to compile custom GPU code.

```bash
conda install -n asp-safety -c nvidia/label/cuda-11.8.0 cuda-toolkit -y
```

**What this does:** Installs CUDA 11.8 inside your conda environment (does not affect your system).

**Verify:**
```bash
conda activate asp-safety
nvcc --version
```

**What you should see:** Something like `Cuda compilation tools, release 11.8`

### If you see an error

| Error | How to fix it |
|-------|---------------|
| `nvcc: command not found` | Make sure you ran `conda activate asp-safety` first |
| `Package not found` | Try: `conda install -c nvidia cuda-toolkit=11.8 -y` |
| Very slow download | Try adding `--solver=libmamba` to the command |

---

## Step 5: Install PyTorch with CUDA 11.8

This installs PyTorch (the deep learning framework) with GPU support.

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118
```

**What this does:** Downloads PyTorch pre-built for CUDA 11.8 so you don't have to compile it yourself.

**Verify:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'PyTorch version: {torch.__version__}')"
```

**What you should see:**
```
CUDA available: True
PyTorch version: 2.1.0+cu118
```

### If you see an error

| Error | How to fix it |
|-------|---------------|
| `CUDA available: False` | PyTorch was installed without CUDA. Uninstall and reinstall: `pip uninstall torch torchvision torchaudio` then repeat this step. Make sure you copied the `--index-url` part exactly |
| `torch` installs but version shows `2.14.0+cpu` or similar | You have a different torch installed. Run `pip uninstall torch -y` then repeat this step |
| Download is very large (>2GB) | That's normal, the CUDA version is big. Wait for it to finish |

---

## Step 6: Install Project Dependencies

From the project root directory (`asp-safety/`):

```bash
pip install -r requirements.txt
```

**What this does:** Installs the Python packages the project needs (numpy, clingo, nuScenes, etc.).

### If you see an error

| Error | How to fix it |
|-------|---------------|
| `ERROR: pip's dependency resolver does not take into account...` | Usually safe to ignore. The packages will still work |
| `clingo` fails to install | Try: `conda install -c conda-forge clingo=5.7.1 -y` instead |
| `nuscenes-devkit` fails | Try: `pip install nuscenes-devkit --no-deps` then install missing deps one by one |

---

## Step 7: Install SparseDrive Dependencies

```bash
cd SparseDrive
pip install -r requirement.txt
```

**What this does:** Installs the specific packages SparseDrive needs (OpenCV, mmcv, mmdet, etc.).

### If you see an error

This step has the most potential issues. Here's what to do:

**`mmcv_full` fails to build (the usual case):**
Skip the source build — install the pre-built wheel for the WORKING
pin (1.7.2; the upstream spec's 1.7.1 is superseded — DC-5, see
`docs/environment.md`). Note the index is `torch2.1.0`, not `torch2.1`:
```bash
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
```
(CPU-only machine: use `.../cpu/torch2.1.0/` as the index.)
Then install the rest:
```bash
pip install opencv-python==4.8.1.78 pyquaternion==0.9.9 nuscenes-devkit==1.1.10 \
    yapf==0.33.0 tensorboard==2.14.0 motmetrics==1.1.3 pandas \
    flash-attn==2.3.2 prettytable==3.7.0 scikit-learn==1.3.0
```

**`flash-attn==2.3.2` build fails:**
This is a common issue. Try these in order:
```bash
# Option A: Install without build isolation
pip install flash-attn==2.3.2 --no-build-isolation

# Option B: Install a newer version (still compatible)
pip install flash-attn

# Option C: Skip it entirely (some features won't work but the project still runs)
# Just continue to the next step
```

**`numpy` version conflict:**
The SparseDrive requirements pin `numpy==1.23.5` but other packages may need newer numpy. Allow upgrades:
```bash
pip install "numpy>=1.23.5"
```

**`urllib3` or other small package errors:**
Install packages one by one, skipping the one that failed:
```bash
pip install mmcv-full==1.7.2 mmdet==2.28.2 pyquaternion==0.9.9 \
    nuscenes-devkit==1.1.10 yapf==0.33.0 tensorboard==2.14.0 \
    motmetrics==1.1.3 opencv-python==4.8.1.78 prettytable==3.7.0 scikit-learn==1.3.0
```

**Verify:**
```bash
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import mmcv; print(f'mmcv: {mmcv.__version__}')"
```

### Go back to project root when done:
```bash
cd ..
```

---

## Step 8: Compile Custom CUDA Operations

> **No-compile path (no admin rights / no MSVC / no CUDA toolkit):** you
> can SKIP this step entirely. The planner wrapper uses the pure-torch
> `deformable_aggregation` fallback + the SDPA `flash_attn` path
> (`src/neural_planner.py`; kernel math verified to 8.3e-07 — see
> `docs/environment.md` install note 7). This is the path used by the
> working env measured end-to-end (CPU 3.9 s, CUDA 2.4 s per frame).

This compiles the custom GPU kernel that SparseDrive uses for deformable attention.

```bash
cd SparseDrive/projects/mmdet3d_plugin/ops
python setup.py develop
```

**What this does:** Compiles C++/CUDA code into a Python module. Takes 2–5 minutes.

**What you should see:** Compilation messages ending with `Finished CUDA extensions setup` or similar success message.

### If you see an error

| Error | How to fix it |
|-------|---------------|
| `nvcc not found` | CUDA toolkit not in PATH. Run: `export PATH="$CONDA_PREFIX/bin:$PATH"` then try again |
| `fatal error: cuda_runtime.h: No such file` | CUDA toolkit not installed properly. Re-run Step 4 |
| `error: command 'nvcc' failed` | Same as above — make sure `nvcc --version` works |
| `torch.cuda.is_available() returns False` | PyTorch doesn't see your GPU. Re-run Step 5 |
| Compilation takes forever or runs out of memory | RTX 4050 has 6GB VRAM which should be enough. Try: `export TORCH_CUDA_ARCH_LIST="8.9"` before running the command |

**Verify:**
```bash
python -c "import deformable_aggregation_ext; print('Custom ops: OK')"
```

### Go back to project root when done:
```bash
cd ../../../../..
```

---

## Step 9: Run the Test Suite

```bash
python tests/test_sparse_drive.py
```

**What you should see:**
```
Testing SparseDrivePlanner...
  [PASS] Initialization successful
  [PASS] Prediction successful
    Trajectory shape: (6, 2)
    Boxes shape: (N, 7)
    Scores shape: (N,)
  [PASS] Correctly raised ValueError for wrong shape
  [PASS] Projection matrix construction (RQ-2)
  [PASS] Postprocess golden values (T-5)
  [PASS] Postprocess fail-loud (P-13)
  [PASS] STEP_DT contract pinned (P-1)
  [PASS] Placeholder determinism (T-12)
  [PASS] Temporal sequence API (N-2)
  [PASS] estimate_ego_status layout (N-2)
```

### If you see an error

| Error | How to fix it |
|-------|---------------|
| `ModuleNotFoundError: No module named 'cv2'` | Step 7 didn't complete. Run: `pip install opencv-python==4.8.1.78` |
| `ModuleNotFoundError: No module named 'mmcv'` | Step 7 didn't complete. Run: `pip install mmcv_full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html` |
| `ImportError: deformable_aggregation_ext` | Step 8 didn't complete. Re-run it |
| Tests pass but say "Using placeholder planner" | The weights file is missing. Make sure `SparseDrive/ckpt/sparsedrive_stage2.pth` exists (986MB) |

---

## Step 10: Smoke Test Real Inference

This confirms the full pipeline works with your GPU.

```bash
python -c "
from src.neural_planner import SparseDrivePlanner
import numpy as np

print('Loading SparseDrive model...')
planner = SparseDrivePlanner('SparseDrive/ckpt/sparsedrive_stage2.pth', 'cuda')

print('Running inference...')
images = np.random.randn(6, 3, 256, 704).astype(np.float32)
intrinsics = np.eye(3, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
extrinsics = np.eye(4, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
ego2global = np.eye(4, dtype=np.float32)

result = planner.predict(images, intrinsics, extrinsics, ego2global, 0.0)
print(f'Trajectory: {result["trajectory"].shape}')
print(f'Boxes: {result["boxes"].shape}')
print(f'Velocities: {result["velocities"].shape}')
print('SUCCESS: Real inference is working!')
"
```

**What you should see:**
```
Loading SparseDrive model...
SparseDrive model loaded from SparseDrive/ckpt/sparsedrive_stage2.pth
Running inference...
Trajectory: (6, 2)
Boxes: (N, 7)
Velocities: (N, 3)
SUCCESS: Real inference is working!
```

`N` varies with the input (random noise yields hundreds of low-score
detections; the pipeline keeps score>=0.5, cap 20). If you see this,
you're done! The full setup is complete.

---

## Quick Verification Checklist

Run all of these to confirm everything is working:
```bash
conda activate asp-safety
nvidia-smi                                          # GPU visible
nvcc --version                                      # CUDA 11.8
python -c "import torch; print(torch.cuda.is_available())"   # True
python -c "import clingo; print(clingo.__version__)"         # 5.7.1
python -c "import cv2; print(cv2.__version__)"               # 4.8.x
python -c "import mmcv; print(mmcv.__version__)"             # 1.7.2
python tests/test_sparse_drive.py                            # All PASS
```

---

## Troubleshooting Quick Reference

| Problem | Solution |
|---------|----------|
| `nvidia-smi` not found | Install NVIDIA driver from nvidia.com |
| `conda` not found | Run `source ~/.bashrc` or restart terminal |
| `nvcc` not found | Run `conda activate asp-safety` or `export PATH="$CONDA_PREFIX/bin:$PATH"` |
| PyTorch says CUDA unavailable | Reinstall: `pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118` |
| `mmcv` build fails | Use pre-built: `pip install mmcv_full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html` |
| `flash-attn` build fails | Try `pip install flash-attn --no-build-isolation` or skip it |
| Custom ops import error | Re-run `python setup.py develop` in `SparseDrive/projects/mmdet3d_plugin/ops` |
| Tests say "placeholder planner" | Weights file missing at `SparseDrive/ckpt/sparsedrive_stage2.pth` |

---

## Starting Fresh? Complete One-Liner

If something is badly broken and you want to start over:
```bash
conda env remove -n asp-safety -y
conda create -n asp-safety python=3.10 -y
conda activate asp-safety
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit -y
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
cd SparseDrive && pip install mmcv_full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html && cd ..
```
Then continue from Step 7 for the remaining SparseDrive dependencies
(pinning notes: `numpy==1.26.4`, `setuptools==75.8.0`,
`opencv-python==4.8.1.78`, `mmdet==2.28.2`; do NOT install mmdet3d;
see `docs/environment.md`).

---

## Step 11 (Optional): CARLA Install + Smoke Setup

Closed-loop work is scoped to install + setup only (S11/S12); no bridge
or scenarios yet. 6 GB VRAM is the stated minimum — Low quality,
synchronous fixed-timestep when the bridge lands.

```bash
conda activate asp-safety
pip install carla==0.9.15
python -c "import carla; print('client ok')"
```

Download `CARLA_0.9.15.zip` (7.22 GB) from
`https://downloads.carlasim.com/Windows/CARLA_0.9.15.zip`
(the GitHub release page lists packages but hosts no binaries),
extract to `C:\CARLA` (NOT the repo), then smoke-test with the
bundled scripts (default map only — first `load_world` of another
town can exceed client timeouts on this hardware; the nav cache
downloads on demand):

```bash
python experiments/carla_launch.py   # server up, port 2000 open
python experiments/carla_smoke.py    # spawn ego, read one camera
```

**Verified 2026-09-26:** server 0.9.15, default map
`Carla/Maps/Town10HD_Opt`, camera frame (256, 704, 3) uint8
(planner input resolution), clean actor destroy + server shutdown.
First launch crashed mid-map-load at ~5.6/6.1 GB VRAM; relaunch
succeeded in ~10 s once shader caches were warm. Keep
`experiments/carla_launch.py` for server lifecycle; kill stray
servers with `taskkill /F /IM CarlaUE4.exe`.
VRAM retreat order on OOM: smaller sensors -> `-RenderOffScreen`
-> CPU planner. Detail + rates: session S11.
