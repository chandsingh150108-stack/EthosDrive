# Environment Record — ASP Safety Layer (N-4)

Canonical runtime: conda env `asp-safety`. Captured 2026-09-26 from
`C:\Users\chand\anaconda3\envs\asp-safety` (Python 3.10.21, torch 2.1.0+cu118,
CUDA available, RTX 4050 Laptop).

## Machine facts (S11 probe, read-only)

- GPU: NVIDIA GeForce RTX 4050 Laptop, 6 GB VRAM, driver 581.57
  (CARLA minimum; 8 GB recommended — Low quality mandatory).
- CPU: AMD Ryzen 5 6600H, 6 cores (traffic-manager single thread
  caps NPC counts at ~20).
- RAM 16 GB; C: 76 GB free (packaged CARLA ~= 20 GB fits).
- No admin rights (prebuilt CARLA needs none); VCRedist x64 present;
  TCP 2000/2001 free.
- CARLA home: `C:\CARLA`, 0.9.15 stable (INSTALLED 2026-09-26:
  7,755,135,185-byte zip verified, extracted to
  `C:\CARLA\WindowsNoEditor\CarlaUE4.exe`; client `carla==0.9.15`
  in env; default map Town10HD_Opt).

## Pinned versions (measured, `pip freeze`)

```
torch==2.1.0+cu118
torchvision==0.16.0
torchaudio==2.1.0
mmcv-full==1.7.2
mmdet==2.28.2
nuscenes-devkit==1.2.0
numpy==1.26.4
setuptools==75.8.0
opencv-python==4.8.1.78
opencv-python-headless==4.11.0.86
clingo==5.7.1
scipy==1.15.3
pandas==2.3.3
matplotlib==3.10.9
seaborn==0.13.2
pytest==9.1.1
PyYAML==6.0.3
carla==0.9.15
einops==0.8.2
prettytable==3.18.0
motmetrics==1.4.0
ipython==8.39.0
black==26.5.1
flake8==7.4.1
mypy==2.3.1
```

## Install notes (must reproduce in this order)

1. `conda create -n asp-safety python=3.10 -y`
2. torch 2.1.0 + cu118 from `https://download.pytorch.org/whl/cu118`.
3. mmcv-full 1.7.2 from the matching index
   `https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/` (CPU index is
   `.../cpu/torch2.1.0/`). Then `mmdet==2.28.2`, `prettytable`, `motmetrics`,
   `ipython`, `einops`.
4. Pin `numpy==1.26.4`, `setuptools==75.8.0`, `opencv-python==4.8.1.78`.
5. `clingo==5.7.1`, `nuscenes-devkit==1.2.0`, scipy/pandas/matplotlib/seaborn,
   pytest/black/flake8/mypy.
6. Do NOT `pip install mmdet3d` for the planner path (mmcv 1.x vs 2.x
   conflict). Note (2026-09-26 drift): `mmdet3d==1.1.1` is currently
   importable in this env — harmless because the wrapper uses mmdet
   registries only, but a clean rebuild should leave it out.
7. No MSVC / CUDA toolkit on this machine (no admin) — compiled ops are NOT
   built; the wrapper uses pure-torch `deformable_aggregation` and SDPA
   `flash_attn` fallbacks instead (see `src/neural_planner.py`).

## Data

- `C:/data/sets/nuscenes` — v1.0-mini + map expansion (`maps/` present).
- The expansion zip extracts as `expansion/*.json` + `basemap/*.png`
  at the data root, but the devkit expects `maps/expansion/*.json`
  and `maps/basemap/*.png`: copy both directories under `maps/`
  (done 2026-09-26; originals left in place).
- `asp-safety/SparseDrive/ckpt/sparsedrive_stage2.pth` — 1,033,678,905 B (~986 MB).
- `opencode.json` must keep `external_directory: allow` so agents can read `C:/data`.

## Dependency-pin divergence vs upstream spec (DC-5, closed 2026-10-21)

- Upstream SparseDrive (`SparseDrive/docs/quick_start.md`,
  `requirement.txt`) pins `torch==1.13.0+cu116` (torchvision 0.14.0)
  and the spec row references `mmcv_full==1.7.1`. The working env uses
  **torch 2.1.0+cu118 + mmcv-full 1.7.2 + mmdet 2.28.2** (mmdet3d
  excluded; mmcv 1.x vs 2.x conflict — install note 6).
- Canonical pins for THIS repo: `requirements.txt` + the freeze above.
  Upstream spec files are left untouched on purpose (do not "fix"
  upstream); `SETUP_GUIDE.md` Step 7 troubleshooting quotes 1.7.2.
- Compiled CUDA ops are never required here: the no-compile fallback
  path (pure-torch deformable + SDPA) is the verified end-to-end path
  (install note 7).
- Non-conda drift (S13 note): the system-Python dev shell reads
  clingo **5.8.2** while the canonical env pin is 5.7.1. Catalog code
  avoids the constructs that differ (`#abs`); 5.7.1 stays canonical.
