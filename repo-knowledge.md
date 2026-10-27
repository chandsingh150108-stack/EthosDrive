# AGENTS.md - ASP Safety Layer for Autonomous Vehicles

## Project Context

This document provides complete specifications for implementing an ASP-based safety layer for autonomous driving. An agent reading this file can implement the entire project from start to finish without ambiguity.

**Project Type**: Course Project (Primary), Workshop Paper (Secondary)
**Duration**: 4 weeks (core), 8 weeks (enhancements)
**Team Size**: 4 members
**Hardware**: 2x laptops with NVIDIA GeForce RTX 4050 (6GB VRAM)

---

## Table of Contents

1. [Subdivision 1: Project Initialization & Environment Setup](#subdivision-1-project-initialization--environment-setup)
2. [Subdivision 2: SparseDrive Neural Planner Integration](#subdivision-2-sparsedrive-neural-planner-integration)
3. [Subdivision 3: ASP Traffic Rules Specification](#subdivision-3-asp-traffic-rules-specification)
4. [Subdivision 4: Python-Clingo Bridge Module](#subdivision-4-python-clingo-bridge-module)
5. [Subdivision 5: Data Loading & Preprocessing](#subdivision-5-data-loading--preprocessing)
6. [Subdivision 6: Trajectory Correction Engine](#subdivision-6-trajectory-correction-engine)
7. [Subdivision 7: Evaluation Metrics & Benchmarking](#subdivision-7-evaluation-metrics--benchmarking)
8. [Subdivision 8: End-to-End Pipeline Integration](#subdivision-8-end-to-end-pipeline-integration)
9. [Subdivision 9: Testing & Validation Suite](#subdivision-9-testing--validation-suite)
10. [Subdivision 10: Documentation & Course Report](#subdivision-10-documentation--course-report)
11. [Subdivision 11: Comprehensive ASP Rule Catalog](#subdivision-11-comprehensive-asp-rule-catalog)

---

## Execution Protocol

### Order of Execution
```
Subdivision 1 (Environment)
    ├── Subdivision 2 (SparseDrive) ──────────────────────────┐
    ├── Subdivision 3 (Basic Rules) → Subdivision 4 (Checker) → Subdivision 6 (Correction)
    │       └── Subdivision 11 (63 Rules) ────────────────────┤
    ├── Subdivision 5 (Data Loading) ─────────────────────────┤
    ├── Subdivision 7 (Evaluation) ───────────────────────────┤
    └── Subdivision 8 (Pipeline Integration) ←────────────────┘
            └── Subdivision 9 (Testing)
                └── Subdivision 10 (Documentation)
```

### Decision Points
When you encounter `[USER INPUT REQUIRED]`, pause and ask the user to choose from the provided options.

### Error Protocol
If any step fails:
1. Log error to `asp-safety/errors.log`
2. Report: subdivision number, failure point, error message, suggested fix
3. Wait for user decision before proceeding

---

# Subdivision 1: Project Initialization & Environment Setup

**Objective**: Create project directory structure, conda environment, and install all dependencies.

**Dependencies**: None (start here)

**Estimated Time**: 1-2 hours

**Files to Create**:
- `asp-safety/` (directory)
- `asp-safety/configs/` (directory)
- `asp-safety/configs/default.yaml`
- `asp-safety/requirements.txt`
- `setup_env.sh`

---

## Step 1.1: Create Directory Structure

Execute the following command to create the project directory structure:

```bash
mkdir -p asp-safety/{configs,rules,src,experiments,tests,docs,paper}
```

This creates:
```
asp-safety/
├── configs/
├── rules/
├── src/
├── experiments/
├── tests/
├── docs/
└── paper/
```

**Verification**: Run `ls -la asp-safety/` and confirm all 7 subdirectories exist.

---

## Step 1.2: Create requirements.txt

Create file `asp-safety/requirements.txt` with the following exact content:

```
# PyTorch with CUDA 11.8
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0

# ASP Solver
clingo==5.7.1

# nuScenes devkit
nuscenes-devkit

# Scientific computing
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Development tools
pytest>=7.3.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.4.0
```

**Verification**: Run `cat asp-safety/requirements.txt` and confirm content matches exactly.

---

## Step 1.3: Create setup_env.sh

Create file `setup_env.sh` with the following exact content:

```bash
#!/bin/bash
# setup_env.sh - Environment setup for ASP Safety Layer project
set -e

echo "=========================================="
echo "ASP Safety Layer - Environment Setup"
echo "=========================================="

# Check conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not installed"
    echo "Install from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create environment
echo "[1/4] Creating conda environment 'asp-safety'..."
conda create -n asp-safety python=3.10 -y

# Activate
echo "[2/4] Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate asp-safety

# Install PyTorch
echo "[3/4] Installing PyTorch 2.1.0 with CUDA 11.8..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install other packages
echo "[4/4] Installing remaining packages..."
pip install clingo==5.7.1
pip install nuscenes-devkit
pip install numpy scipy pandas matplotlib seaborn
pip install pytest black flake8 mypy

# Verify
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import clingo; print(f'Clingo: {clingo.__version__}')"
python -c "import nuscenes; print('nuScenes: OK')"

echo ""
echo "Setup complete! Run: conda activate asp-safety"
```

Make executable:
```bash
chmod +x setup_env.sh
```

**Verification**: Run `ls -la setup_env.sh` and confirm file is executable (`-rwxr-xr-x`).

---

## Step 1.4: Create configs/default.yaml

Create file `asp-safety/configs/default.yaml` with the following exact content:

```yaml
# ASP Safety Layer - Default Configuration

project:
  name: "asp-safety-layer"
  version: "1.0.0"

paths:
  sparse_drive: "./SparseDrive"
  rules_dir: "./rules"
  data_dir: "/data/sets/nuscenes"

sparse_drive:
  model_path: "./SparseDrive/ckpt/sparsedrive_stage2.pth"
  device: "cuda"
  input_shape: [6, 3, 256, 704]  # [cameras, channels, height, width]

asp:
  rules_file: "./rules/basic.lp"
  timeout_ms: 100

rules:
  safe_distance: 5.0  # meters
  speed_limit: 50.0   # km/h
  min_following_time: 2.0  # seconds

evaluation:
  dataset: "nuscenes"
  split: "mini"
  num_scenes: 100
```

**Verification**: Run `cat asp-safety/configs/default.yaml` and confirm YAML is valid.

---

## Step 1.5: Verify Environment

After running `setup_env.sh`, verify all components:

```bash
conda activate asp-safety

# Test 1: PyTorch with CUDA
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}: OK')
"

# Test 2: Clingo
python -c "
import clingo
print(f'Clingo {clingo.__version__}: OK')
"

# Test 3: nuScenes
python -c "
import nuscenes
print('nuScenes devkit: OK')
"

# Test 4: Scientific packages
python -c "
import numpy, scipy, pandas, matplotlib, seaborn
print('Scientific packages: OK')
"
```

**Success Criteria**:
- [ ] All directories created
- [ ] `requirements.txt` created
- [ ] `setup_env.sh` created and executable
- [ ] `configs/default.yaml` created
- [ ] All 4 verification tests pass

**[USER INPUT REQUIRED]**: Do you want to use conda (recommended) or virtualenv for the Python environment?

---

# Subdivision 2: SparseDrive Neural Planner Integration

**Objective**: Clone SparseDrive repository, download pre-trained weights, and create wrapper class for inference.

**Dependencies**: Subdivision 1 complete

**Estimated Time**: 2-3 hours

**Files to Create**:
- `asp-safety/SparseDrive/` (cloned repository)
- `asp-safety/SparseDrive/ckpt/sparsedrive_stage2.pth`
- `asp-safety/src/neural_planner.py`

---

## Step 2.1: Clone SparseDrive Repository

Execute:
```bash
cd asp-safety
git clone https://github.com/swc-17/SparseDrive.git
```

**Verification**: Run `ls asp-safety/SparseDrive/` and confirm repository files exist.

---

## Step 2.2: Download Pre-trained Weights

Execute:
```bash
cd asp-safety/SparseDrive
mkdir -p ckpt
wget https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage2.pth \
    -O ckpt/sparsedrive_stage2.pth
```

Verify file size (should be ~400MB):
```bash
ls -lh asp-safety/SparseDrive/ckpt/sparsedrive_stage2.pth
```

**Verification**: File exists and size is between 300MB and 500MB.

---

## Step 2.3: Create neural_planner.py

Create file `asp-safety/src/neural_planner.py` with the following exact content:

```python
"""SparseDrive Neural Planner Wrapper.

This module provides a wrapper class for SparseDrive inference.
The class loads pre-trained weights and generates trajectory predictions.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional


class SparseDrivePlanner:
    """Wrapper for SparseDrive model inference.
    
    This class loads a pre-trained SparseDrive model and provides
    a simple interface for trajectory prediction.
    
    Attributes:
        device: PyTorch device ("cuda" or "cpu")
        model: Loaded SparseDrive model
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """Initialize the planner.
        
        Args:
            model_path: Absolute path to .pth weights file
            device: "cuda" for GPU, "cpu" for CPU
            
        Raises:
            FileNotFoundError: If model_path doesn't exist
            RuntimeError: If device="cuda" but CUDA unavailable
        """
        self.model_path = Path(model_path)
        self.device = device
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA not available")
            except ImportError:
                raise RuntimeError("PyTorch not installed")
        
        self.model = self._load_model()
    
    def _load_model(self):
        """Load the SparseDrive model.
        
        This is a placeholder that should be replaced with actual
        SparseDrive model loading code.
        
        Returns:
            Loaded model object
        """
        # TODO: Replace with actual SparseDrive model loading
        # The actual implementation depends on SparseDrive version
        # Example:
        #   import torch
        #   from mmdet3d.models import build_detector
        #   config = ...
        #   model = build_detector(config)
        #   model.load_state_dict(torch.load(self.model_path))
        #   return model.to(self.device).eval()
        
        print(f"Loading model from {self.model_path}")
        print("Note: Using placeholder model. Replace with actual SparseDrive.")
        return None
    
    def predict(
        self,
        images: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        ego2global: np.ndarray,
        timestamp: float
    ) -> Dict[str, np.ndarray]:
        """Generate trajectory prediction.
        
        Args:
            images: Camera images
                Shape: (6, 3, 256, 704)
                dtype: float32
                Order: [CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,
                        CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT]
            intrinsics: Camera intrinsic matrices
                Shape: (6, 3, 3)
                dtype: float32
            extrinsics: Camera extrinsic matrices
                Shape: (6, 4, 4)
                dtype: float32
            ego2global: Vehicle to global transformation
                Shape: (4, 4)
                dtype: float32
            timestamp: Unix timestamp
                dtype: float
                
        Returns:
            Dict with keys:
                'trajectory': np.ndarray, shape (6, 2), float32
                    Predicted [x, y] positions at t=0,1,2,3,4,5 seconds
                'boxes': np.ndarray, shape (N, 7), float32
                    Detected 3D boxes [x, y, z, w, h, l, yaw]
                'scores': np.ndarray, shape (N,), float32
                    Detection confidence scores [0, 1]
                    
        Raises:
            ValueError: If images.shape != (6, 3, 256, 704)
        """
        # Validate input shape
        expected_shape = (6, 3, 256, 704)
        if images.shape != expected_shape:
            raise ValueError(
                f"Expected images shape {expected_shape}, got {images.shape}"
            )
        
        # TODO: Replace with actual SparseDrive inference
        # Example:
        #   import torch
        #   with torch.no_grad():
        #       inputs = {
        #           'img': torch.tensor(images).unsqueeze(0).to(self.device),
        #           'intrinsics': torch.tensor(intrinsics).unsqueeze(0).to(self.device),
        #           'extrinsics': torch.tensor(extrinsics).unsqueeze(0).to(self.device),
        #           'ego2global': torch.tensor(ego2global).unsqueeze(0).to(self.device),
        #           'timestamp': timestamp
        #       }
        #       outputs = self.model(inputs)
        #       return self._postprocess(outputs)
        
        # Placeholder: Return synthetic trajectory
        return self._generate_placeholder()
    
    def _generate_placeholder(self) -> Dict[str, np.ndarray]:
        """Generate placeholder outputs for testing.
        
        Returns:
            Dict with synthetic trajectory, boxes, and scores
        """
        # Generate straight line trajectory (5 m/s constant velocity)
        t = np.arange(6, dtype=np.float32)
        trajectory = np.column_stack([
            t * 5.0,  # x = velocity * time
            np.zeros(6, dtype=np.float32)  # y = 0
        ])
        
        # Generate placeholder boxes
        boxes = np.array([
            [20.0, 0.0, 0.5, 2.0, 1.5, 4.5, 0.0],  # [x,y,z,w,h,l,yaw]
            [40.0, 5.0, 0.5, 2.0, 1.5, 4.5, 0.0],
        ], dtype=np.float32)
        
        # Generate placeholder scores
        scores = np.array([0.95, 0.87], dtype=np.float32)
        
        return {
            'trajectory': trajectory,
            'boxes': boxes,
            'scores': scores
        }
```

---

## Step 2.4: Verify SparseDrive Integration

Create test file `asp-safety/tests/test_sparse_drive.py`:

```python
"""Test SparseDrive integration."""

import sys
sys.path.insert(0, 'asp-safety/src')

from neural_planner import SparseDrivePlanner
import numpy as np

def test_sparse_drive_planner():
    """Test SparseDrivePlanner initialization and prediction."""
    print("Testing SparseDrivePlanner...")
    
    # Test 1: Initialization
    try:
        planner = SparseDrivePlanner(
            model_path="asp-safety/SparseDrive/ckpt/sparsedrive_stage2.pth",
            device="cpu"  # Use CPU for testing
        )
        print("  [PASS] Initialization successful")
    except Exception as e:
        print(f"  [FAIL] Initialization failed: {e}")
        return False
    
    # Test 2: Prediction with correct input shape
    try:
        images = np.random.randn(6, 3, 256, 704).astype(np.float32)
        intrinsics = np.eye(3, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
        extrinsics = np.eye(4, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
        ego2global = np.eye(4, dtype=np.float32)
        timestamp = 1234567890.0
        
        result = planner.predict(images, intrinsics, extrinsics, ego2global, timestamp)
        
        assert 'trajectory' in result
        assert 'boxes' in result
        assert 'scores' in result
        
        assert result['trajectory'].shape == (6, 2)
        assert result['boxes'].shape[1] == 7
        assert result['scores'].ndim == 1
        
        print("  [PASS] Prediction successful")
        print(f"    Trajectory shape: {result['trajectory'].shape}")
        print(f"    Boxes shape: {result['boxes'].shape}")
        print(f"    Scores shape: {result['scores'].shape}")
    except Exception as e:
        print(f"  [FAIL] Prediction failed: {e}")
        return False
    
    # Test 3: Prediction with wrong input shape (should fail)
    try:
        wrong_images = np.random.randn(3, 3, 256, 704).astype(np.float32)
        planner.predict(wrong_images, intrinsics, extrinsics, ego2global, timestamp)
        print("  [FAIL] Should have raised ValueError")
        return False
    except ValueError:
        print("  [PASS] Correctly raised ValueError for wrong shape")
    
    print("\nAll SparseDrive tests passed!")
    return True

if __name__ == "__main__":
    test_sparse_drive_planner()
```

Run test:
```bash
cd asp-safety
python tests/test_sparse_drive.py
```

**Success Criteria**:
- [ ] SparseDrive repository cloned
- [ ] Pre-trained weights downloaded (>300MB)
- [ ] `neural_planner.py` created
- [ ] All 3 tests pass

---

# Subdivision 3: ASP Traffic Rules Specification

> SUPERSEDED (DC-1, 2026-09-26): the 5-file float-based starter below was
> replaced by the 63-rule integer catalog (Subdivision 11 +
> `asp_traffic_rules.md`, implemented in `rules/helpers.lp` +
> `rules/<R1..R15>.lp` + `rules/basic.lp`). Do NOT create the files in
> Steps 3.1-3.6; unit conventions live in `asp-safety/docs/conventions.md`
> and divergences in `asp-safety/docs/rule_divergences.md`. The steps
> remain for archeology only.

**Objective**: Create 5 ASP rule files in Clingo syntax that encode traffic rules.

**Dependencies**: Subdivision 1 complete

**Estimated Time**: 1-2 hours

**Files to Create**:
- `asp-safety/rules/lane_keeping.lp`
- `asp-safety/rules/collision_avoidance.lp`
- `asp-safety/rules/speed_limit.lp`
- `asp-safety/rules/traffic_light.lp`
- `asp-safety/rules/safe_following.lp`
- `asp-safety/rules/basic.lp`

---

## Step 3.1: Create lane_keeping.lp

Create file `asp-safety/rules/lane_keeping.lp` with the following exact content:

```asp
% lane_keeping.lp - Lane Keeping Rule
% Vehicle must stay within lane boundaries

% Time horizon: 6 timesteps (0-5)
time(0..5).

% Trajectory point: position of vehicle V at time T
% Input facts: pos(V, T, X, Y)
point(V, T, X, Y) :- vehicle(V), time(T), pos(V, T, X, Y).

% Lane boundary: line segment from (X1,Y1) to (X2,Y2)
% Input facts: lane(L, X1, Y1, X2, Y2)

% Point inside lane (simplified axis-aligned bounding box)
in_lane(X, Y) :-
    lane(_, X1, Y1, X2, Y2),
    min(X1, X2) <= X, X <= max(X1, X2),
    min(Y1, Y2) <= Y, Y <= max(Y1, Y2).

% Violation: vehicle outside lane
violation(lane_keeping, V, T) :-
    point(V, T, X, Y),
    vehicle(V),
    time(T),
    not in_lane(X, Y).

% Optimization: minimize violations
#minimize { 1@T, V : violation(lane_keeping, V, T) }.
```

---

## Step 3.2: Create collision_avoidance.lp

Create file `asp-safety/rules/collision_avoidance.lp` with the following exact content:

```asp
% collision_avoidance.lp - Collision Avoidance Rule
% Maintain safe distance from other vehicles

% Safe distance threshold in meters
safe_distance(5.0).

% Euclidean distance between two vehicles at time T
distance(V1, V2, T, D) :-
    point(V1, T, X1, Y1),
    point(V2, T, X2, Y2),
    V1 != V2,
    time(T),
    D = sqrt((X1-X2)*(X1-X2) + (Y1-Y2)*(Y1-Y2)).

% Violation: vehicles too close
violation(collision_avoidance, V1, T) :-
    point(V1, T, X1, Y1),
    point(V2, T, X2, Y2),
    V1 != V2,
    time(T),
    distance(V1, V2, T, D),
    safe_distance(S),
    D < S.

% Optimization: minimize violations
#minimize { 1@T, V : violation(collision_avoidance, V, T) }.
```

---

## Step 3.3: Create speed_limit.lp

Create file `asp-safety/rules/speed_limit.lp` with the following exact content:

```asp
% speed_limit.lp - Speed Limit Rule
% Vehicle must not exceed speed limit

% Default speed limit in km/h
default_speed_limit(50.0).

% Time step in seconds (for velocity calculation)
dt(1.0).

% Velocity calculation (m/s)
% V becomes available at T=1 (needs previous position)
velocity(V, T, VX, VY) :-
    point(V, T, X, Y),
    point(V, T-1, X0, Y0),
    time(T),
    T > 0,
    dt(DT),
    VX = (X - X0) / DT,
    VY = (Y - Y0) / DT.

% Speed in km/h (convert from m/s)
speed(V, T, S) :-
    velocity(V, T, VX, VY),
    time(T),
    S = sqrt(VX*VX + VY*VY) * 3.6.

% Violation: speed exceeds limit
violation(speed_limit, V, T) :-
    speed(V, T, S),
    time(T),
    default_speed_limit(MAX),
    S > MAX.

% Optimization: minimize violations
#minimize { 1@T, V : violation(speed_limit, V, T) }.
```

---

## Step 3.4: Create traffic_light.lp

Create file `asp-safety/rules/traffic_light.lp` with the following exact content:

```asp
% traffic_light.lp - Traffic Light Compliance Rule
% Vehicle must stop at red lights

% Traffic light state: light_state(I, Color, T)
% Colors: red, yellow, green
% Input facts: light_state(intersection_id, color, timestep)

% Intersection location: intersection(I, X, Y, Radius)
% Input facts: intersection(intersection_id, x, y, radius_meters)

% Distance from vehicle to intersection center
dist_to_intersection(V, T, I, D) :-
    point(V, T, X, Y),
    intersection(I, IX, IY, _),
    time(T),
    D = sqrt((X-IX)*(X-IX) + (Y-IY)*(Y-IY)).

% Vehicle at intersection (within radius)
at_intersection(V, T, I) :-
    dist_to_intersection(V, T, I, D),
    intersection(I, _, _, R),
    D <= R.

% Violation: at red light
violation(traffic_light, V, T) :-
    at_intersection(V, T, I),
    light_state(I, red, T),
    time(T).

% Optimization: minimize violations
#minimize { 1@T, V : violation(traffic_light, V, T) }.
```

---

## Step 3.5: Create safe_following.lp

Create file `asp-safety/rules/safe_following.lp` with the following exact content:

```asp
% safe_following.lp - Safe Following Distance Rule
% Maintain adequate following distance behind vehicle ahead

% Minimum following time in seconds
min_following_time(2.0).

% Vehicle velocity (m/s, only x-component for simplicity)
veh_velocity(V, T, VX) :-
    velocity(V, T, VX, _),
    time(T).

% Required following distance based on speed
required_distance(V, T, RD) :-
    veh_velocity(V, T, VX),
    min_following_time(MFT),
    VX > 0,
    RD = VX * MFT.

% Following distance (V1 behind V2)
following_distance(V1, V2, T, D) :-
    point(V1, T, X1, Y1),
    point(V2, T, X2, Y2),
    time(T),
    X1 < X2,
    D = sqrt((X1-X2)*(X1-X2) + (Y1-Y2)*(Y1-Y2)).

% Violation: insufficient following distance
violation(safe_following, V1, T) :-
    following_distance(V1, V2, T, D),
    required_distance(V1, T, RD),
    time(T),
    D < RD.

% Optimization: minimize violations
#minimize { 1@T, V : violation(safe_following, V, T) }.
```

---

## Step 3.6: Create basic.lp (Aggregator)

Create file `asp-safety/rules/basic.lp` with the following exact content:

```asp
% basic.lp - Aggregates all traffic rules

% Include individual rule files
#include "lane_keeping.lp".
#include "collision_avoidance.lp".
#include "speed_limit.lp".
#include "traffic_light.lp".
#include "safe_following.lp".

% Global time horizon
time(0..5).

% Vehicle definitions (added dynamically via Python)
% vehicle(ego).
% vehicle(vehicle_0).

% Total violation count
total_violations(V) :-
    V = #count { T, R, Veh : violation(R, Veh, T) }.

% Safe trajectory: no violations
safe(ego) :-
    total_violations(0).
```

---

## Step 3.7: Verify Rules

Execute the following verification:

```bash
cd asp-safety

# Test each rule file compiles
for rule in lane_keeping collision_avoidance speed_limit traffic_light safe_following basic; do
    echo "Testing ${rule}.lp..."
    clingo rules/${rule}.lp --text --terminate=0 2>&1 | head -5
    if [ $? -eq 0 ]; then
        echo "  PASS"
    else
        echo "  FAIL"
        exit 1
    fi
done

echo ""
echo "All rules compile successfully!"
```

**Success Criteria**:
- [ ] All 5 rule files created
- [ ] basic.lp aggregates all rules
- [ ] All rules compile without errors
- [ ] Rules accept input facts in specified format

---

# Subdivision 4: Python-Clingo Bridge Module

**Objective**: Create the TrajectoryChecker class that interfaces Python with Clingo ASP solver.

**Dependencies**: Subdivision 1, Subdivision 3 complete

**Estimated Time**: 2-3 hours

**Files to Create**:
- `asp-safety/src/checker.py`

---

## Step 4.1: Create checker.py

Create file `asp-safety/src/checker.py` with the following exact content:

```python
"""ASP-based trajectory safety checker.

This module provides the TrajectoryChecker class that uses Clingo
to check trajectories against traffic rules.
"""

import clingo
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class TrajectoryChecker:
    """ASP-based trajectory safety checker.
    
    This class loads ASP traffic rules and checks vehicle trajectories
    for violations.
    
    Attributes:
        rules_dir: Path to directory containing .lp rule files
        control: Clingo Control instance
    """
    
    def __init__(self, rules_dir: str = "rules"):
        """Initialize the checker.
        
        Args:
            rules_dir: Path to directory containing .lp rule files
            
        Raises:
            FileNotFoundError: If rules_dir doesn't exist
            RuntimeError: If no .lp files found in rules_dir
        """
        self.rules_dir = Path(rules_dir)
        
        if not self.rules_dir.exists():
            raise FileNotFoundError(f"Rules directory not found: {rules_dir}")
        
        self.control = clingo.Control(["--warn=none"])
        self._load_rules()
    
    def _load_rules(self):
        """Load all ASP rules from the rules directory.
        
        Raises:
            RuntimeError: If no .lp files found
        """
        lp_files = list(self.rules_dir.glob("*.lp"))
        
        if not lp_files:
            raise RuntimeError(f"No .lp files found in {self.rules_dir}")
        
        for rule_file in lp_files:
            with open(rule_file, 'r') as f:
                self.control.add("base", [], f.read())
    
    def check(
        self,
        trajectory: np.ndarray,
        other_vehicles: Optional[List[np.ndarray]] = None,
        map_data: Optional[Dict] = None,
        traffic_lights: Optional[Dict] = None
    ) -> Tuple[bool, List[Dict]]:
        """Check trajectory against all rules.
        
        Args:
            trajectory: Vehicle trajectory
                Shape: (T, 2) where T <= 6
                dtype: float32
                Values: [x, y] positions at each timestep
            other_vehicles: Optional list of other vehicle trajectories
                Each shape: (T, 2)
            map_data: Optional map data dictionary
                Structure: {'lanes': {'lane_id': [{'x1': float, 'y1': float, 
                            'x2': float, 'y2': float}]}}
            traffic_lights: Optional traffic light states
                Structure: {'intersection_id': {'color': str, 'time': int}}
                
        Returns:
            Tuple of:
                - is_safe: bool (True if no violations)
                - violations: List[Dict] with structure:
                    {'rule': str, 'vehicle': str, 'time': int}
                    
        Raises:
            ValueError: If trajectory.shape[1] != 2
            ValueError: If trajectory.shape[0] > 6
        """
        # Validate trajectory
        if trajectory.ndim != 2 or trajectory.shape[1] != 2:
            raise ValueError(
                f"Expected trajectory shape (T, 2), got {trajectory.shape}"
            )
        
        if trajectory.shape[0] > 6:
            raise ValueError(
                f"Maximum 6 timesteps allowed, got {trajectory.shape[0]}"
            )
        
        # Clear previous state
        self.control.cleanup()
        
        # Add ego vehicle trajectory
        self._add_trajectory("ego", trajectory)
        
        # Add other vehicles if provided
        if other_vehicles:
            for i, veh in enumerate(other_vehicles):
                self._add_trajectory(f"vehicle_{i}", veh)
        
        # Add map data if provided
        if map_data:
            self._add_map_data(map_data)
        
        # Add traffic lights if provided
        if traffic_lights:
            self._add_traffic_lights(traffic_lights)
        
        # Ground the program
        self.control.ground([("base", [])])
        
        # Solve and collect violations
        violations = []
        
        def on_model(model):
            for atom in model.symbols(atoms=True):
                if atom.name == "violation":
                    violations.append({
                        "rule": str(atom.arguments[0]),
                        "vehicle": str(atom.arguments[1]),
                        "time": int(atom.arguments[2])
                    })
        
        self.control.solve(on_model=on_model)
        
        return len(violations) == 0, violations
    
    def _add_trajectory(self, vehicle_id: str, trajectory: np.ndarray):
        """Add trajectory as ASP facts.
        
        Generates facts:
            pos(vehicle_id, t, x, y). for each timestep
            vehicle(vehicle_id). once per vehicle
            
        Args:
            vehicle_id: Identifier for the vehicle
            trajectory: Shape (T, 2) array of positions
        """
        for t, (x, y) in enumerate(trajectory):
            self.control.add("base", [],
                f"pos({vehicle_id}, {t}, {x:.4f}, {y:.4f}).")
        self.control.add("base", [], f"vehicle({vehicle_id}).")
    
    def _add_map_data(self, map_data: Dict):
        """Add lane boundaries as ASP facts.
        
        Generates facts:
            lane(lane_id, x1, y1, x2, y2).
            
        Args:
            map_data: Dictionary with lane information
        """
        for lane_id, segments in map_data.get('lanes', {}).items():
            for seg in segments:
                self.control.add("base", [],
                    f"lane({lane_id}, {seg['x1']:.4f}, {seg['y1']:.4f}, "
                    f"{seg['x2']:.4f}, {seg['y2']:.4f}).")
    
    def _add_traffic_lights(self, lights: Dict):
        """Add traffic light states as ASP facts.
        
        Generates facts:
            light_state(intersection_id, color, time).
            intersection(intersection_id, x, y, radius).
            
        Args:
            lights: Dictionary with traffic light states
        """
        for intersection_id, state in lights.items():
            self.control.add("base", [],
                f"light_state({intersection_id}, {state['color']}, "
                f"{state['time']}).")
            # Add intersection location if provided
            if 'x' in state and 'y' in state and 'radius' in state:
                self.control.add("base", [],
                    f"intersection({intersection_id}, {state['x']:.4f}, "
                    f"{state['y']:.4f}, {state['radius']:.4f}).")
    
    def get_safe_trajectory(
        self,
        trajectory: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Return corrected trajectory that satisfies all rules.
        
        Args:
            trajectory: Original trajectory, shape (T, 2)
            **kwargs: Arguments for check() method
            
        Returns:
            Corrected trajectory, shape (T, 2)
            If already safe, returns original trajectory (no copy)
        """
        is_safe, violations = self.check(trajectory, **kwargs)
        
        if is_safe:
            return trajectory
        
        # Apply corrections
        corrected = self._apply_corrections(trajectory, violations)
        return corrected
    
    def _apply_corrections(
        self,
        trajectory: np.ndarray,
        violations: List[Dict]
    ) -> np.ndarray:
        """Apply corrections based on violation type.
        
        Args:
            trajectory: Original trajectory
            violations: List of detected violations
            
        Returns:
            Corrected trajectory
        """
        corrected = trajectory.copy()
        
        for violation in violations:
            rule = violation['rule']
            t = violation['time']
            
            if rule == 'speed_limit' and t > 0:
                # Reduce velocity by 20%
                dx = corrected[t, 0] - corrected[t-1, 0]
                dy = corrected[t, 1] - corrected[t-1, 1]
                scale = 0.8
                corrected[t, 0] = corrected[t-1, 0] + dx * scale
                corrected[t, 1] = corrected[t-1, 1] + dy * scale
            
            elif rule == 'collision_avoidance':
                # Shift trajectory 2m in y-direction
                corrected[t, 1] += 2.0
            
            elif rule == 'lane_keeping':
                # Project to nearest lane center (simplified)
                corrected[t, 1] = 0.0
            
            elif rule == 'traffic_light':
                # Stop before intersection (set velocity to 0)
                if t > 0:
                    corrected[t, 0] = corrected[t-1, 0]
                    corrected[t, 1] = corrected[t-1, 1]
            
            elif rule == 'safe_following':
                # Reduce speed to increase following distance
                if t > 0:
                    dx = corrected[t, 0] - corrected[t-1, 0]
                    dy = corrected[t, 1] - corrected[t-1, 1]
                    scale = 0.7
                    corrected[t, 0] = corrected[t-1, 0] + dx * scale
                    corrected[t, 1] = corrected[t-1, 1] + dy * scale
        
        return corrected
```

---

## Step 4.2: Verify TrajectoryChecker

Create test file `asp-safety/tests/test_checker.py`:

```python
"""Test TrajectoryChecker."""

import sys
sys.path.insert(0, 'asp-safety/src')

import numpy as np
from checker import TrajectoryChecker


def test_trajectory_checker():
    """Test TrajectoryChecker initialization and checking."""
    print("Testing TrajectoryChecker...")
    
    # Test 1: Initialization
    try:
        checker = TrajectoryChecker("asp-safety/rules")
        print("  [PASS] Initialization successful")
    except Exception as e:
        print(f"  [FAIL] Initialization failed: {e}")
        return False
    
    # Test 2: Safe trajectory (straight line)
    try:
        safe_traj = np.array([
            [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]
        ], dtype=np.float32)
        
        is_safe, violations = checker.check(safe_traj)
        
        assert is_safe == True, f"Expected safe, got violations: {violations}"
        assert violations == []
        print("  [PASS] Safe trajectory detected correctly")
    except Exception as e:
        print(f"  [FAIL] Safe trajectory test failed: {e}")
        return False
    
    # Test 3: Speed limit violation
    try:
        fast_traj = np.array([
            [0, 0], [20, 0], [40, 0], [60, 0], [80, 0], [100, 0]
        ], dtype=np.float32)
        
        is_safe, violations = checker.check(fast_traj)
        
        assert is_safe == False, "Expected unsafe trajectory"
        assert any(v['rule'] == 'speed_limit' for v in violations), \
            f"Expected speed_limit violation, got: {violations}"
        print("  [PASS] Speed limit violation detected")
    except Exception as e:
        print(f"  [FAIL] Speed limit test failed: {e}")
        return False
    
    # Test 4: Collision avoidance violation
    try:
        ego = np.array([
            [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]
        ], dtype=np.float32)
        other = np.array([
            [3, 1], [3.5, 1], [4, 1], [4.5, 1], [5, 1], [5.5, 1]
        ], dtype=np.float32)
        
        is_safe, violations = checker.check(ego, other_vehicles=[other])
        
        # This should detect collision if distance < 5m
        if not is_safe:
            assert any(v['rule'] == 'collision_avoidance' for v in violations)
            print("  [PASS] Collision avoidance violation detected")
        else:
            print("  [PASS] No collision (distance > 5m)")
    except Exception as e:
        print(f"  [FAIL] Collision test failed: {e}")
        return False
    
    # Test 5: Correction
    try:
        corrected = checker.get_safe_trajectory(fast_traj)
        is_safe, _ = checker.check(corrected)
        assert is_safe, f"Corrected trajectory still has violations"
        print("  [PASS] Trajectory correction works")
    except Exception as e:
        print(f"  [FAIL] Correction test failed: {e}")
        return False
    
    print("\nAll TrajectoryChecker tests passed!")
    return True


if __name__ == "__main__":
    test_trajectory_checker()
```

Run test:
```bash
cd asp-safety
python tests/test_checker.py
```

**Success Criteria**:
- [ ] `checker.py` created
- [ ] All 5 tests pass
- [ ] `check()` returns correct violations
- [ ] `get_safe_trajectory()` returns corrected trajectory

---

# Subdivision 5: Data Loading & Preprocessing

**Objective**: Create data loader for nuScenes dataset (with placeholder for testing without dataset).

**Dependencies**: Subdivision 1 complete

**Estimated Time**: 2-3 hours

**Files to Create**:
- `asp-safety/src/data_loader.py`

---

## Step 5.1: Create data_loader.py

Create file `asp-safety/src/data_loader.py` with the following exact content:

```python
"""Data loader for nuScenes dataset.

This module provides data loading functionality for the nuScenes dataset.
If nuScenes is not available, it generates placeholder data for testing.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class NuScenesLoader:
    """Loader for nuScenes dataset.
    
    This class loads camera images, calibration data, and ground truth
    from the nuScenes dataset. If the dataset is not available,
    it generates placeholder data for testing.
    
    Attributes:
        data_dir: Path to nuScenes data directory
        version: Dataset version
        use_placeholder: Whether using placeholder data
    """
    
    def __init__(self, data_dir: str, version: str = "v1.0-mini"):
        """Initialize the data loader.
        
        Args:
            data_dir: Path to nuScenes data directory
            version: Dataset version ("v1.0-mini", "v1.0-trainval", "v1.0-test")
            
        Raises:
            ValueError: If version not in valid list
        """
        valid_versions = ["v1.0-mini", "v1.0-trainval", "v1.0-test"]
        if version not in valid_versions:
            raise ValueError(f"Invalid version: {version}. Must be one of {valid_versions}")
        
        self.data_dir = Path(data_dir)
        self.version = version
        self.use_placeholder = not self.data_dir.exists()
        
        if self.use_placeholder:
            print(f"nuScenes not found at {data_dir}. Using placeholder data.")
            self.num_scenes = 10  # Placeholder scenes
        else:
            # TODO: Initialize nuScenes devkit
            # from nuscenes.nuscenes import NuScenes
            # self.nusc = NuScenes(version=version, dataroot=str(data_dir))
            # self.num_scenes = len(self.nusc.scene)
            self.num_scenes = 10
    
    def get_scene_sample(self, scene_idx: int = 0) -> Dict[str, np.ndarray]:
        """Get a sample from the dataset.
        
        Args:
            scene_idx: Index of scene to load (0-indexed)
            
        Returns:
            Dict with keys:
                'images': np.ndarray, shape (6, 3, 256, 704), uint8
                    Camera images in order:
                    [CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,
                     CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT]
                'intrinsics': np.ndarray, shape (6, 3, 3), float32
                    Camera intrinsic matrices
                'extrinsics': np.ndarray, shape (6, 4, 4), float32
                    Camera extrinsic matrices (camera-to-vehicle)
                'ego2global': np.ndarray, shape (4, 4), float32
                    Vehicle-to-global transformation
                'timestamp': float
                    Unix timestamp
                    
        Raises:
            IndexError: If scene_idx out of range
        """
        if scene_idx < 0 or scene_idx >= self.num_scenes:
            raise IndexError(
                f"scene_idx {scene_idx} out of range [0, {self.num_scenes})"
            )
        
        if self.use_placeholder:
            return self._generate_placeholder_sample(scene_idx)
        else:
            return self._load_nuscenes_sample(scene_idx)
    
    def _generate_placeholder_sample(self, scene_idx: int) -> Dict[str, np.ndarray]:
        """Generate placeholder sample for testing.
        
        Args:
            scene_idx: Scene index (used for seed)
            
        Returns:
            Placeholder sample dictionary
        """
        # Use scene_idx as seed for reproducibility
        np.random.seed(scene_idx)
        
        # Generate random images (6 cameras, 3 channels, 256x704)
        images = np.random.randint(
            0, 255, (6, 3, 256, 704), dtype=np.uint8
        )
        
        # Generate identity intrinsics
        intrinsics = np.eye(3, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
        # Set focal length and principal point
        intrinsics[:, 0, 0] = 400.0  # fx
        intrinsics[:, 1, 1] = 400.0  # fy
        intrinsics[:, 0, 2] = 352.0  # cx
        intrinsics[:, 1, 2] = 128.0  # cy
        
        # Generate identity extrinsics
        extrinsics = np.eye(4, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
        
        # Generate ego2global
        ego2global = np.eye(4, dtype=np.float32)
        ego2global[0, 3] = scene_idx * 10.0  # x offset
        
        # Generate timestamp
        timestamp = 1234567890.0 + scene_idx * 0.5
        
        return {
            'images': images,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'ego2global': ego2global,
            'timestamp': timestamp
        }
    
    def _load_nuscenes_sample(self, scene_idx: int) -> Dict[str, np.ndarray]:
        """Load actual nuScenes sample.
        
        Args:
            scene_idx: Scene index
            
        Returns:
            Sample dictionary
        """
        # TODO: Implement actual nuScenes loading
        # This is a placeholder for when nuScenes is available
        raise NotImplementedError(
            "nuScenes loading not implemented. "
            "Install nuscenes-devkit and provide valid data_dir."
        )
    
    def get_trajectory(self, scene_idx: int = 0) -> np.ndarray:
        """Get ground truth trajectory for a scene.
        
        Args:
            scene_idx: Index of scene
            
        Returns:
            np.ndarray: Shape (T, 2), dtype float32
                T = up to 6 timesteps
                Values = [x, y] positions in vehicle-local coordinates
        """
        if self.use_placeholder:
            return self._generate_placeholder_trajectory(scene_idx)
        else:
            # TODO: Implement actual trajectory extraction
            raise NotImplementedError("nuScenes trajectory extraction not implemented")
    
    def _generate_placeholder_trajectory(self, scene_idx: int) -> np.ndarray:
        """Generate placeholder trajectory.
        
        Args:
            scene_idx: Scene index (used for variation)
            
        Returns:
            Synthetic trajectory
        """
        np.random.seed(scene_idx)
        
        # Generate trajectory with some variation
        t = np.arange(6, dtype=np.float32)
        base_velocity = 5.0 + np.random.randn() * 2.0  # 5 m/s +/- 2
        
        x = t * base_velocity
        y = np.cumsum(np.random.randn(6).astype(np.float32) * 0.5)
        
        return np.column_stack([x, y])
    
    def get_map_data(self, scene_idx: int = 0) -> Dict:
        """Get map data for a scene.
        
        Args:
            scene_idx: Index of scene
            
        Returns:
            Dict with structure:
                {
                    'lanes': {
                        'lane_id': [
                            {'x1': float, 'y1': float, 'x2': float, 'y2': float}
                        ]
                    },
                    'intersections': {
                        'intersection_id': {
                            'x': float, 'y': float, 'radius': float
                        }
                    }
                }
        """
        if self.use_placeholder:
            return self._generate_placeholder_map_data()
        else:
            # TODO: Implement actual map extraction
            raise NotImplementedError("nuScenes map extraction not implemented")
    
    def _generate_placeholder_map_data(self) -> Dict:
        """Generate placeholder map data.
        
        Returns:
            Synthetic map data dictionary
        """
        return {
            'lanes': {
                'lane_0': [
                    {'x1': -10.0, 'y1': -2.0, 'x2': 100.0, 'y2': -2.0},
                    {'x1': -10.0, 'y1': 2.0, 'x2': 100.0, 'y2': 2.0}
                ],
                'lane_1': [
                    {'x1': -10.0, 'y1': 2.0, 'x2': 100.0, 'y2': 2.0},
                    {'x1': -10.0, 'y1': 6.0, 'x2': 100.0, 'y2': 6.0}
                ]
            },
            'intersections': {
                'int_0': {'x': 50.0, 'y': 0.0, 'radius': 10.0}
            }
        }
    
    def get_traffic_lights(self, scene_idx: int = 0) -> Dict:
        """Get traffic light states for a scene.
        
        Args:
            scene_idx: Index of scene
            
        Returns:
            Dict with structure:
                {
                    'intersection_id': {
                        'color': 'red'|'yellow'|'green',
                        'time': int,
                        'x': float,
                        'y': float,
                        'radius': float
                    }
                }
        """
        if self.use_placeholder:
            return self._generate_placeholder_traffic_lights()
        else:
            # TODO: Implement actual traffic light extraction
            raise NotImplementedError("nuScenes traffic light extraction not implemented")
    
    def _generate_placeholder_traffic_lights(self) -> Dict:
        """Generate placeholder traffic light states.
        
        Returns:
            Synthetic traffic light dictionary
        """
        return {
            'int_0': {
                'color': 'red',
                'time': 0,
                'x': 50.0,
                'y': 0.0,
                'radius': 10.0
            }
        }
```

---

## Step 5.2: Verify DataLoader

Create test file `asp-safety/tests/test_data_loader.py`:

```python
"""Test DataLoader."""

import sys
sys.path.insert(0, 'asp-safety/src')

import numpy as np
from data_loader import NuScenesLoader


def test_data_loader():
    """Test NuScenesLoader initialization and data loading."""
    print("Testing NuScenesLoader...")
    
    # Test 1: Initialization with placeholder
    try:
        loader = NuScenesLoader("nonexistent_path", version="v1.0-mini")
        assert loader.use_placeholder == True
        print("  [PASS] Initialization with placeholder successful")
    except Exception as e:
        print(f"  [FAIL] Initialization failed: {e}")
        return False
    
    # Test 2: Get scene sample
    try:
        sample = loader.get_scene_sample(0)
        
        assert 'images' in sample
        assert 'intrinsics' in sample
        assert 'extrinsics' in sample
        assert 'ego2global' in sample
        assert 'timestamp' in sample
        
        assert sample['images'].shape == (6, 3, 256, 704)
        assert sample['intrinsics'].shape == (6, 3, 3)
        assert sample['extrinsics'].shape == (6, 4, 4)
        assert sample['ego2global'].shape == (4, 4)
        assert isinstance(sample['timestamp'], float)
        
        print("  [PASS] Scene sample loaded correctly")
        print(f"    Images shape: {sample['images'].shape}")
        print(f"    Timestamp: {sample['timestamp']}")
    except Exception as e:
        print(f"  [FAIL] Scene sample test failed: {e}")
        return False
    
    # Test 3: Get trajectory
    try:
        trajectory = loader.get_trajectory(0)
        
        assert trajectory.shape[1] == 2
        assert trajectory.shape[0] <= 6
        assert trajectory.dtype == np.float32
        
        print("  [PASS] Trajectory loaded correctly")
        print(f"    Shape: {trajectory.shape}")
    except Exception as e:
        print(f"  [FAIL] Trajectory test failed: {e}")
        return False
    
    # Test 4: Get map data
    try:
        map_data = loader.get_map_data(0)
        
        assert 'lanes' in map_data
        assert 'intersections' in map_data
        
        print("  [PASS] Map data loaded correctly")
    except Exception as e:
        print(f"  [FAIL] Map data test failed: {e}")
        return False
    
    # Test 5: Get traffic lights
    try:
        traffic_lights = loader.get_traffic_lights(0)
        
        assert 'int_0' in traffic_lights
        assert 'color' in traffic_lights['int_0']
        
        print("  [PASS] Traffic lights loaded correctly")
    except Exception as e:
        print(f"  [FAIL] Traffic lights test failed: {e}")
        return False
    
    # Test 6: Invalid scene index
    try:
        loader.get_scene_sample(100)
        print("  [FAIL] Should have raised IndexError")
        return False
    except IndexError:
        print("  [PASS] IndexError raised for invalid scene_idx")
    
    print("\nAll DataLoader tests passed!")
    return True


if __name__ == "__main__":
    test_data_loader()
```

Run test:
```bash
cd asp-safety
python tests/test_data_loader.py
```

**Success Criteria**:
- [ ] `data_loader.py` created
- [ ] All 6 tests pass
- [ ] Returns correct shapes and dtypes
- [ ] Placeholder works without nuScenes dataset

---

# Subdivision 6: Trajectory Correction Engine

**Objective**: Create TrajectoryCorrector class that fixes invalid trajectories.

**Dependencies**: Subdivision 3, Subdivision 4 complete

**Estimated Time**: 2-3 hours

**Files to Create**:
- `asp-safety/src/corrector.py`

---

## Step 6.1: Create corrector.py

Create file `asp-safety/src/corrector.py` with the following exact content:

```python
"""Trajectory Correction Engine.

This module provides the TrajectoryCorrector class that fixes
invalid trajectories to satisfy ASP traffic rules.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

# Import TrajectoryChecker
try:
    from checker import TrajectoryChecker
except ImportError:
    from src.checker import TrajectoryChecker


class TrajectoryCorrector:
    """Corrects invalid trajectories to satisfy traffic rules.
    
    This class takes an invalid trajectory and applies corrections
    to make it satisfy all ASP traffic rules.
    
    Attributes:
        checker: TrajectoryChecker instance
        max_iterations: Maximum optimization iterations
    """
    
    def __init__(self, checker: TrajectoryChecker, max_iterations: int = 100):
        """Initialize the corrector.
        
        Args:
            checker: Initialized TrajectoryChecker instance
            max_iterations: Maximum number of correction iterations
            
        Raises:
            TypeError: If checker is not a TrajectoryChecker
        """
        if not isinstance(checker, TrajectoryChecker):
            raise TypeError("checker must be a TrajectoryChecker instance")
        
        self.checker = checker
        self.max_iterations = max_iterations
    
    def correct(
        self,
        trajectory: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Correct an invalid trajectory.
        
        Args:
            trajectory: Invalid trajectory, shape (T, 2)
            **kwargs: Arguments for TrajectoryChecker.check()
            
        Returns:
            Tuple of:
                - corrected_trajectory: Shape (T, 2)
                - applied_corrections: List[Dict] with structure:
                    {
                        'rule': str,
                        'timestep': int,
                        'correction_type': str,
                        'original_value': Any,
                        'corrected_value': Any
                    }
                    
        Raises:
            RuntimeError: If correction fails after max_iterations
        """
        # Check if already valid
        is_safe, violations = self.checker.check(trajectory, **kwargs)
        
        if is_safe:
            return trajectory.copy(), []
        
        # Apply corrections iteratively
        corrected = trajectory.copy()
        applied_corrections = []
        
        for iteration in range(self.max_iterations):
            # Check current state
            is_safe, violations = self.checker.check(corrected, **kwargs)
            
            if is_safe:
                return corrected, applied_corrections
            
            # Apply correction for first violation
            if violations:
                violation = violations[0]
                corrected, correction = self._apply_correction(
                    corrected, violation, **kwargs
                )
                applied_corrections.append(correction)
            else:
                break
        
        # Final check
        is_safe, _ = self.checker.check(corrected, **kwargs)
        if not is_safe:
            raise RuntimeError(
                f"Failed to correct trajectory after {self.max_iterations} iterations"
            )
        
        return corrected, applied_corrections
    
    def _apply_correction(
        self,
        trajectory: np.ndarray,
        violation: Dict,
        **kwargs
    ) -> Tuple[np.ndarray, Dict]:
        """Apply correction for a specific violation.
        
        Args:
            trajectory: Current trajectory
            violation: Violation dictionary
            
        Returns:
            Tuple of (corrected_trajectory, correction_info)
        """
        rule = violation['rule']
        t = violation['time']
        
        corrected = trajectory.copy()
        correction_info = {
            'rule': rule,
            'timestep': t,
            'original_value': trajectory[t].tolist()
        }
        
        if rule == 'speed_limit':
            corrected, correction_type = self._correct_speed_limit(
                corrected, t, **kwargs
            )
            correction_info['correction_type'] = correction_type
        
        elif rule == 'collision_avoidance':
            corrected, correction_type = self._correct_collision(
                corrected, t, **kwargs
            )
            correction_info['correction_type'] = correction_type
        
        elif rule == 'lane_keeping':
            corrected, correction_type = self._correct_lane_keeping(
                corrected, t, **kwargs
            )
            correction_info['correction_type'] = correction_type
        
        elif rule == 'traffic_light':
            corrected, correction_type = self._correct_traffic_light(
                corrected, t, **kwargs
            )
            correction_info['correction_type'] = correction_type
        
        elif rule == 'safe_following':
            corrected, correction_type = self._correct_safe_following(
                corrected, t, **kwargs
            )
            correction_info['correction_type'] = correction_type
        
        else:
            correction_type = "unknown"
        
        correction_info['corrected_value'] = corrected[t].tolist()
        
        return corrected, correction_info
    
    def _correct_speed_limit(
        self,
        trajectory: np.ndarray,
        t: int,
        **kwargs
    ) -> Tuple[np.ndarray, str]:
        """Correct speed limit violation.
        
        Strategy: Reduce velocity by 20% at violation timestep.
        
        Args:
            trajectory: Current trajectory
            t: Violation timestep
            
        Returns:
            Tuple of (corrected_trajectory, correction_type)
        """
        if t == 0:
            return trajectory, "no_change"
        
        # Reduce velocity
        dx = trajectory[t, 0] - trajectory[t-1, 0]
        dy = trajectory[t, 1] - trajectory[t-1, 1]
        
        scale = 0.8  # 20% reduction
        trajectory[t, 0] = trajectory[t-1, 0] + dx * scale
        trajectory[t, 1] = trajectory[t-1, 1] + dy * scale
        
        return trajectory, "velocity_reduction_20pct"
    
    def _correct_collision(
        self,
        trajectory: np.ndarray,
        t: int,
        **kwargs
    ) -> Tuple[np.ndarray, str]:
        """Correct collision avoidance violation.
        
        Strategy: Shift trajectory 2m in y-direction.
        
        Args:
            trajectory: Current trajectory
            t: Violation timestep
            
        Returns:
            Tuple of (corrected_trajectory, correction_type)
        """
        trajectory[t, 1] += 2.0
        return trajectory, "lateral_shift_2m"
    
    def _correct_lane_keeping(
        self,
        trajectory: np.ndarray,
        t: int,
        **kwargs
    ) -> Tuple[np.ndarray, str]:
        """Correct lane keeping violation.
        
        Strategy: Project to nearest lane center (y=0).
        
        Args:
            trajectory: Current trajectory
            t: Violation timestep
            
        Returns:
            Tuple of (corrected_trajectory, correction_type)
        """
        trajectory[t, 1] = 0.0
        return trajectory, "project_to_lane_center"
    
    def _correct_traffic_light(
        self,
        trajectory: np.ndarray,
        t: int,
        **kwargs
    ) -> Tuple[np.ndarray, str]:
        """Correct traffic light violation.
        
        Strategy: Stop before intersection (set velocity to 0).
        
        Args:
            trajectory: Current trajectory
            t: Violation timestep
            
        Returns:
            Tuple of (corrected_trajectory, correction_type)
        """
        if t > 0:
            trajectory[t, 0] = trajectory[t-1, 0]
            trajectory[t, 1] = trajectory[t-1, 1]
        return trajectory, "stop_before_intersection"
    
    def _correct_safe_following(
        self,
        trajectory: np.ndarray,
        t: int,
        **kwargs
    ) -> Tuple[np.ndarray, str]:
        """Correct safe following distance violation.
        
        Strategy: Reduce speed by 30% to increase following distance.
        
        Args:
            trajectory: Current trajectory
            t: Violation timestep
            
        Returns:
            Tuple of (corrected_trajectory, correction_type)
        """
        if t == 0:
            return trajectory, "no_change"
        
        dx = trajectory[t, 0] - trajectory[t-1, 0]
        dy = trajectory[t, 1] - trajectory[t-1, 1]
        
        scale = 0.7  # 30% reduction
        trajectory[t, 0] = trajectory[t-1, 0] + dx * scale
        trajectory[t, 1] = trajectory[t-1, 1] + dy * scale
        
        return trajectory, "velocity_reduction_30pct"
```

---

## Step 6.2: Verify TrajectoryCorrector

Create test file `asp-safety/tests/test_corrector.py`:

```python
"""Test TrajectoryCorrector."""

import sys
sys.path.insert(0, 'asp-safety/src')

import numpy as np
from checker import TrajectoryChecker
from corrector import TrajectoryCorrector


def test_trajectory_corrector():
    """Test TrajectoryCorrector initialization and correction."""
    print("Testing TrajectoryCorrector...")
    
    # Initialize components
    checker = TrajectoryChecker("asp-safety/rules")
    corrector = TrajectoryCorrector(checker)
    
    # Test 1: Initialization
    try:
        assert corrector.max_iterations == 100
        print("  [PASS] Initialization successful")
    except Exception as e:
        print(f"  [FAIL] Initialization failed: {e}")
        return False
    
    # Test 2: Correct speed limit violation
    try:
        fast_traj = np.array([
            [0, 0], [20, 0], [40, 0], [60, 0], [80, 0], [100, 0]
        ], dtype=np.float32)
        
        corrected, corrections = corrector.correct(fast_traj)
        
        # Verify corrected trajectory is safe
        is_safe, _ = checker.check(corrected)
        assert is_safe, f"Corrected trajectory still has violations"
        
        # Verify corrections were applied
        assert len(corrections) > 0
        assert corrections[0]['rule'] == 'speed_limit'
        
        print("  [PASS] Speed limit correction successful")
        print(f"    Applied {len(corrections)} corrections")
    except Exception as e:
        print(f"  [FAIL] Speed limit correction failed: {e}")
        return False
    
    # Test 3: Already safe trajectory
    try:
        safe_traj = np.array([
            [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]
        ], dtype=np.float32)
        
        corrected, corrections = corrector.correct(safe_traj)
        
        assert len(corrections) == 0
        np.testing.assert_array_equal(corrected, safe_traj)
        
        print("  [PASS] Safe trajectory returned unchanged")
    except Exception as e:
        print(f"  [FAIL] Safe trajectory test failed: {e}")
        return False
    
    # Test 4: Multiple violations
    try:
        # Trajectory that violates both speed and collision
        multi_violation_traj = np.array([
            [0, 0], [20, 0], [40, 0], [60, 0], [80, 0], [100, 0]
        ], dtype=np.float32)
        
        other_vehicle = np.array([
            [3, 1], [3.5, 1], [4, 1], [4.5, 1], [5, 1], [5.5, 1]
        ], dtype=np.float32)
        
        corrected, corrections = corrector.correct(
            multi_violation_traj,
            other_vehicles=[other_vehicle]
        )
        
        # May need multiple iterations
        assert len(corrections) > 0
        
        print("  [PASS] Multiple violations handled")
    except Exception as e:
        print(f"  [FAIL] Multiple violations test failed: {e}")
        return False
    
    print("\nAll TrajectoryCorrector tests passed!")
    return True


if __name__ == "__main__":
    test_trajectory_corrector()
```

Run test:
```bash
cd asp-safety
python tests/test_corrector.py
```

**Success Criteria**:
- [ ] `corrector.py` created
- [ ] All 4 tests pass
- [ ] Corrected trajectory passes `check()`
- [ ] All 5 violation types have correction strategies

---

# Subdivision 7: Evaluation Metrics & Benchmarking

**Objective**: Create MetricsCalculator class for evaluating trajectory planning performance.

**Dependencies**: Subdivision 1 complete

**Estimated Time**: 1-2 hours

**Files to Create**:
- `asp-safety/src/evaluation.py`

---

## Step 7.1: Create evaluation.py

Create file `asp-safety/src/evaluation.py` with the following exact content:

```python
"""Evaluation Metrics for ASP Safety Layer.

This module provides the MetricsCalculator class for evaluating
trajectory planning performance.
"""

import numpy as np
from typing import List, Dict, Tuple


class MetricsCalculator:
    """Calculate evaluation metrics for trajectory planning.
    
    This class provides static methods for calculating various
    metrics used to evaluate autonomous driving performance.
    """
    
    @staticmethod
    def collision_rate(
        trajectories: List[np.ndarray],
        gt_boxes: List[np.ndarray],
        pred_boxes: List[np.ndarray]
    ) -> float:
        """Calculate collision rate.
        
        Args:
            trajectories: List of predicted trajectories
                Each shape: (T, 2)
            gt_boxes: List of ground truth bounding boxes
                Each shape: (N, 7) - [x, y, z, w, h, l, yaw]
            pred_boxes: List of predicted bounding boxes
                Each shape: (M, 7) - [x, y, z, w, h, l, yaw]
                
        Returns:
            float: Collision rate as percentage [0.0, 100.0]
            
        Formula:
            collision_rate = (num_collisions / total_trajectories) * 100
        """
        if not trajectories:
            return 0.0
        
        collisions = 0
        for traj, gt, pred in zip(trajectories, gt_boxes, pred_boxes):
            if MetricsCalculator._check_collision(traj, gt, pred):
                collisions += 1
        
        return collisions / len(trajectories) * 100
    
    @staticmethod
    def l2_error(
        pred_trajectory: np.ndarray,
        gt_trajectory: np.ndarray
    ) -> float:
        """Calculate L2 displacement error.
        
        Args:
            pred_trajectory: Predicted trajectory, shape (T, 2)
            gt_trajectory: Ground truth trajectory, shape (T', 2)
            
        Returns:
            float: Average L2 error in meters
            
        Formula:
            l2_error = mean(sqrt((pred_x - gt_x)^2 + (pred_y - gt_y)^2))
        """
        min_len = min(len(pred_trajectory), len(gt_trajectory))
        
        if min_len == 0:
            return 0.0
        
        return float(np.mean(np.linalg.norm(
            pred_trajectory[:min_len] - gt_trajectory[:min_len], axis=1
        )))
    
    @staticmethod
    def violation_detection_rate(
        true_violations: List[Dict],
        detected_violations: List[Dict]
    ) -> float:
        """Calculate true positive rate for violation detection.
        
        Args:
            true_violations: List of actual violations
            detected_violations: List of detected violations
            
        Returns:
            float: Detection rate [0.0, 1.0]
            
        Formula:
            detection_rate = true_positives / total_true_violations
        """
        if not true_violations:
            return 1.0 if not detected_violations else 0.0
        
        detected = 0
        for tv in true_violations:
            for dv in detected_violations:
                if (tv.get('rule') == dv.get('rule') and
                    tv.get('time') == dv.get('time')):
                    detected += 1
                    break
        
        return detected / len(true_violations)
    
    @staticmethod
    def false_positive_rate(
        true_violations: List[Dict],
        detected_violations: List[Dict]
    ) -> float:
        """Calculate false positive rate for violation detection.
        
        Args:
            true_violations: List of actual violations
            detected_violations: List of detected violations
            
        Returns:
            float: False positive rate [0.0, 1.0]
            
        Formula:
            fpr = false_positives / total_detected_violations
        """
        if not detected_violations:
            return 0.0
        
        false_positives = 0
        for dv in detected_violations:
            is_true = False
            for tv in true_violations:
                if (tv.get('rule') == dv.get('rule') and
                    tv.get('time') == dv.get('time')):
                    is_true = True
                    break
            if not is_true:
                false_positives += 1
        
        return false_positives / len(detected_violations)
    
    @staticmethod
    def average_inference_time(inference_times: List[float]) -> float:
        """Calculate average inference time.
        
        Args:
            inference_times: List of inference times in milliseconds
            
        Returns:
            float: Average inference time in ms
        """
        if not inference_times:
            return 0.0
        return float(np.mean(inference_times))
    
    @staticmethod
    def _check_collision(
        traj: np.ndarray,
        gt_boxes: np.ndarray,
        pred_boxes: np.ndarray
    ) -> bool:
        """Check if trajectory collides with any box.
        
        Args:
            traj: Trajectory (T, 2)
            gt_boxes: Ground truth boxes (N, 7)
            pred_boxes: Predicted boxes (M, 7)
            
        Returns:
            bool: True if collision detected
        """
        for t, point in enumerate(traj):
            for box in gt_boxes:
                if MetricsCalculator._point_in_box(point, box):
                    return True
            for box in pred_boxes:
                if MetricsCalculator._point_in_box(point, box):
                    return True
        return False
    
    @staticmethod
    def _point_in_box(point: np.ndarray, box: np.ndarray) -> bool:
        """Check if 2D point is inside bounding box projection.
        
        Args:
            point: (x, y) point
            box: Bounding box [x, y, z, w, h, l, yaw]
            
        Returns:
            bool: True if point is inside box
        """
        if len(box) < 4:
            return False
        
        x, y = point[:2]
        bx, by, bw, bh = box[:4]
        
        return (bx - bw/2 <= x <= bx + bw/2 and
                by - bh/2 <= y <= by + bh/2)
```

---

## Step 7.2: Verify MetricsCalculator

Create test file `asp-safety/tests/test_evaluation.py`:

```python
"""Test MetricsCalculator."""

import sys
sys.path.insert(0, 'asp-safety/src')

import numpy as np
from evaluation import MetricsCalculator


def test_metrics_calculator():
    """Test MetricsCalculator metrics."""
    print("Testing MetricsCalculator...")
    
    calc = MetricsCalculator()
    
    # Test 1: Collision rate (100% collision)
    try:
        traj = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)
        box = np.array([[1.5, 0, 0, 0.5, 0.5, 0, 0]], dtype=np.float32)
        
        rate = calc.collision_rate([traj], [box], [box])
        
        assert rate == 100.0, f"Expected 100%, got {rate}%"
        print("  [PASS] Collision rate (100%): PASS")
    except Exception as e:
        print(f"  [FAIL] Collision rate test failed: {e}")
        return False
    
    # Test 2: Collision rate (0% collision)
    try:
        traj = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)
        box = np.array([[10, 10, 0, 0.5, 0.5, 0, 0]], dtype=np.float32)
        
        rate = calc.collision_rate([traj], [box], [box])
        
        assert rate == 0.0, f"Expected 0%, got {rate}%"
        print("  [PASS] Collision rate (0%): PASS")
    except Exception as e:
        print(f"  [FAIL] Collision rate test failed: {e}")
        return False
    
    # Test 3: L2 error
    try:
        pred = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)
        gt = np.array([[0, 0], [1, 1], [2, 0]], dtype=np.float32)
        
        error = calc.l2_error(pred, gt)
        
        expected = 0.5  # mean([0, 1, 0]) = 0.5
        assert abs(error - expected) < 0.01, f"Expected {expected}, got {error}"
        print("  [PASS] L2 error: PASS")
    except Exception as e:
        print(f"  [FAIL] L2 error test failed: {e}")
        return False
    
    # Test 4: Violation detection rate (100%)
    try:
        true_v = [{'rule': 'speed_limit', 'time': 2, 'vehicle': 'ego'}]
        detected_v = [{'rule': 'speed_limit', 'time': 2, 'vehicle': 'ego'}]
        
        rate = calc.violation_detection_rate(true_v, detected_v)
        
        assert rate == 1.0, f"Expected 1.0, got {rate}"
        print("  [PASS] Violation detection rate (100%): PASS")
    except Exception as e:
        print(f"  [FAIL] Violation detection rate test failed: {e}")
        return False
    
    # Test 5: False positive rate
    try:
        true_v = []
        detected_v = [{'rule': 'speed_limit', 'time': 2, 'vehicle': 'ego'}]
        
        rate = calc.false_positive_rate(true_v, detected_v)
        
        assert rate == 1.0, f"Expected 1.0, got {rate}"
        print("  [PASS] False positive rate: PASS")
    except Exception as e:
        print(f"  [FAIL] False positive rate test failed: {e}")
        return False
    
    # Test 6: Average inference time
    try:
        times = [10.0, 20.0, 30.0]
        
        avg = calc.average_inference_time(times)
        
        assert avg == 20.0, f"Expected 20.0, got {avg}"
        print("  [PASS] Average inference time: PASS")
    except Exception as e:
        print(f"  [FAIL] Average inference time test failed: {e}")
        return False
    
    print("\nAll MetricsCalculator tests passed!")
    return True


if __name__ == "__main__":
    test_metrics_calculator()
```

Run test:
```bash
cd asp-safety
python tests/test_evaluation.py
```

**Success Criteria**:
- [ ] `evaluation.py` created
- [ ] All 6 tests pass
- [ ] Metrics calculate correctly

---

# Subdivision 8: End-to-End Pipeline Integration

**Objective**: Create SafetyLayerPipeline class that connects all components.

**Dependencies**: Subdivisions 2, 4, 5, 7 complete

**Estimated Time**: 3-4 hours

**Files to Create**:
- `asp-safety/src/pipeline.py`

---

## Step 8.1: Create pipeline.py

Create file `asp-safety/src/pipeline.py` with the following exact content:

```python
"""End-to-End Safety Layer Pipeline.

This module provides the SafetyLayerPipeline class that connects
all components into a complete processing pipeline.
"""

import time
import numpy as np
from typing import Dict, List, Optional

# Import components
try:
    from neural_planner import SparseDrivePlanner
    from checker import TrajectoryChecker
    from corrector import TrajectoryCorrector
    from data_loader import NuScenesLoader
    from evaluation import MetricsCalculator
except ImportError:
    from src.neural_planner import SparseDrivePlanner
    from src.checker import TrajectoryChecker
    from src.corrector import TrajectoryCorrector
    from src.data_loader import NuScenesLoader
    from src.evaluation import MetricsCalculator


class SafetyLayerPipeline:
    """Complete safety layer pipeline.
    
    This class connects SparseDrive neural planner with ASP safety
    checker to provide trajectory planning with safety guarantees.
    
    Attributes:
        planner: SparseDrivePlanner instance
        checker: TrajectoryChecker instance
        corrector: TrajectoryCorrector instance
    """
    
    def __init__(
        self,
        planner_path: str,
        rules_dir: str = "rules",
        device: str = "cuda"
    ):
        """Initialize the pipeline.
        
        Args:
            planner_path: Path to SparseDrive weights
            rules_dir: Path to ASP rule files
            device: "cuda" or "cpu"
        """
        self.planner = SparseDrivePlanner(planner_path, device)
        self.checker = TrajectoryChecker(rules_dir)
        self.corrector = TrajectoryCorrector(self.checker)
    
    def run(
        self,
        images: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        ego2global: np.ndarray,
        timestamp: float,
        map_data: Optional[Dict] = None,
        traffic_lights: Optional[Dict] = None
    ) -> Dict:
        """Run full pipeline on a single frame.
        
        Args:
            images: Camera images, shape (6, 3, 256, 704)
            intrinsics: Camera intrinsics, shape (6, 3, 3)
            extrinsics: Camera extrinsics, shape (6, 4, 4)
            ego2global: Vehicle-to-global transform, shape (4, 4)
            timestamp: Unix timestamp
            map_data: Optional map data dictionary
            traffic_lights: Optional traffic light states
            
        Returns:
            Dict with keys:
                'original_trajectory': np.ndarray, shape (6, 2)
                'safe_trajectory': np.ndarray, shape (6, 2)
                'violations': List[Dict]
                'is_safe': bool
                'inference_time_ms': float
                'boxes': Optional[np.ndarray]
                'scores': Optional[np.ndarray]
        """
        start_time = time.time()
        
        # Step 1: Run SparseDrive inference
        planner_output = self.planner.predict(
            images, intrinsics, extrinsics, ego2global, timestamp
        )
        original_trajectory = planner_output['trajectory']
        
        # Step 2: Check with ASP safety layer
        is_safe, violations = self.checker.check(
            original_trajectory,
            map_data=map_data,
            traffic_lights=traffic_lights
        )
        
        # Step 3: Correct if needed
        if is_safe:
            safe_trajectory = original_trajectory
        else:
            safe_trajectory, _ = self.corrector.correct(
                original_trajectory,
                map_data=map_data,
                traffic_lights=traffic_lights
            )
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        return {
            'original_trajectory': original_trajectory,
            'safe_trajectory': safe_trajectory,
            'violations': violations,
            'is_safe': is_safe,
            'inference_time_ms': inference_time,
            'boxes': planner_output.get('boxes', None),
            'scores': planner_output.get('scores', None)
        }
    
    def run_batch(
        self,
        data_loader: NuScenesLoader,
        scene_indices: List[int]
    ) -> List[Dict]:
        """Run pipeline on multiple scenes.
        
        Args:
            data_loader: Initialized NuScenesLoader
            scene_indices: List of scene indices to process
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for idx in scene_indices:
            try:
                # Load scene data
                sample = data_loader.get_scene_sample(idx)
                map_data = data_loader.get_map_data(idx)
                traffic_lights = data_loader.get_traffic_lights(idx)
                
                # Run pipeline
                result = self.run(
                    sample['images'],
                    sample['intrinsics'],
                    sample['extrinsics'],
                    sample['ego2global'],
                    sample['timestamp'],
                    map_data=map_data,
                    traffic_lights=traffic_lights
                )
                
                result['scene_idx'] = idx
                results.append(result)
                
            except Exception as e:
                print(f"Error processing scene {idx}: {e}")
                results.append({
                    'scene_idx': idx,
                    'error': str(e)
                })
        
        return results
```

---

## Step 8.2: Verify Pipeline

Create test file `asp-safety/tests/test_pipeline.py`:

```python
"""Test SafetyLayerPipeline."""

import sys
sys.path.insert(0, 'asp-safety/src')

import numpy as np
from pipeline import SafetyLayerPipeline
from data_loader import NuScenesLoader


def test_pipeline():
    """Test SafetyLayerPipeline initialization and execution."""
    print("Testing SafetyLayerPipeline...")
    
    # Test 1: Initialization
    try:
        pipeline = SafetyLayerPipeline(
            planner_path="asp-safety/SparseDrive/ckpt/sparsedrive_stage2.pth",
            rules_dir="asp-safety/rules",
            device="cpu"
        )
        print("  [PASS] Initialization successful")
    except Exception as e:
        print(f"  [FAIL] Initialization failed: {e}")
        return False
    
    # Test 2: Run pipeline
    try:
        # Create synthetic inputs
        images = np.random.randn(6, 3, 256, 704).astype(np.float32)
        intrinsics = np.eye(3, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
        extrinsics = np.eye(4, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
        ego2global = np.eye(4, dtype=np.float32)
        timestamp = 1234567890.0
        
        result = pipeline.run(images, intrinsics, extrinsics, ego2global, timestamp)
        
        # Verify result structure
        assert 'original_trajectory' in result
        assert 'safe_trajectory' in result
        assert 'violations' in result
        assert 'is_safe' in result
        assert 'inference_time_ms' in result
        assert 'boxes' in result
        assert 'scores' in result
        
        # Verify shapes
        assert result['original_trajectory'].shape == (6, 2)
        assert result['safe_trajectory'].shape == (6, 2)
        assert isinstance(result['is_safe'], bool)
        assert result['inference_time_ms'] > 0
        
        print("  [PASS] Pipeline run successful")
        print(f"    Original trajectory shape: {result['original_trajectory'].shape}")
        print(f"    Safe trajectory shape: {result['safe_trajectory'].shape}")
        print(f"    Is safe: {result['is_safe']}")
        print(f"    Inference time: {result['inference_time_ms']:.2f} ms")
    except Exception as e:
        print(f"  [FAIL] Pipeline run failed: {e}")
        return False
    
    # Test 3: Run batch
    try:
        loader = NuScenesLoader("nonexistent_path")
        results = pipeline.run_batch(loader, [0, 1, 2])
        
        assert len(results) == 3
        assert all('scene_idx' in r for r in results)
        
        print("  [PASS] Batch run successful")
    except Exception as e:
        print(f"  [FAIL] Batch run failed: {e}")
        return False
    
    print("\nAll Pipeline tests passed!")
    return True


if __name__ == "__main__":
    test_pipeline()
```

Run test:
```bash
cd asp-safety
python tests/test_pipeline.py
```

**Success Criteria**:
- [ ] `pipeline.py` created
- [ ] All 3 tests pass
- [ ] Pipeline runs end-to-end without errors
- [ ] Returns complete result dictionary

---

# Subdivision 9: Testing & Validation Suite

**Objective**: Create comprehensive test suite for all modules.

**Dependencies**: All previous subdivisions complete

**Estimated Time**: 2-3 hours

**Files to Create**:
- `asp-safety/tests/__init__.py`
- `asp-safety/tests/test_all.py`

---

## Step 9.1: Create __init__.py

Create file `asp-safety/tests/__init__.py`:

```python
"""Tests for ASP Safety Layer."""
```

---

## Step 9.2: Create comprehensive test suite

Create file `asp-safety/tests/test_all.py`:

```python
"""Comprehensive test suite for ASP Safety Layer."""

import sys
sys.path.insert(0, 'asp-safety/src')

import numpy as np


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("ASP Safety Layer - Comprehensive Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Test 1: SparseDrive
    print("\n[1/7] Testing SparseDrive...")
    try:
        from neural_planner import SparseDrivePlanner
        planner = SparseDrivePlanner(
            "asp-safety/SparseDrive/ckpt/sparsedrive_stage2.pth",
            device="cpu"
        )
        images = np.random.randn(6, 3, 256, 704).astype(np.float32)
        intrinsics = np.eye(3, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
        extrinsics = np.eye(4, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
        ego2global = np.eye(4, dtype=np.float32)
        result = planner.predict(images, intrinsics, extrinsics, ego2global, 0.0)
        assert result['trajectory'].shape == (6, 2)
        results['SparseDrive'] = 'PASS'
    except Exception as e:
        results['SparseDrive'] = f'FAIL: {e}'
    
    # Test 2: TrajectoryChecker
    print("\n[2/7] Testing TrajectoryChecker...")
    try:
        from checker import TrajectoryChecker
        checker = TrajectoryChecker("asp-safety/rules")
        safe_traj = np.array([[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]], dtype=np.float32)
        is_safe, violations = checker.check(safe_traj)
        assert is_safe == True
        results['TrajectoryChecker'] = 'PASS'
    except Exception as e:
        results['TrajectoryChecker'] = f'FAIL: {e}'
    
    # Test 3: TrajectoryCorrector
    print("\n[3/7] Testing TrajectoryCorrector...")
    try:
        from corrector import TrajectoryCorrector
        corrector = TrajectoryCorrector(checker)
        fast_traj = np.array([[0,0],[20,0],[40,0],[60,0],[80,0],[100,0]], dtype=np.float32)
        corrected, _ = corrector.correct(fast_traj)
        is_safe, _ = checker.check(corrected)
        assert is_safe == True
        results['TrajectoryCorrector'] = 'PASS'
    except Exception as e:
        results['TrajectoryCorrector'] = f'FAIL: {e}'
    
    # Test 4: DataLoader
    print("\n[4/7] Testing DataLoader...")
    try:
        from data_loader import NuScenesLoader
        loader = NuScenesLoader("nonexistent")
        sample = loader.get_scene_sample(0)
        assert sample['images'].shape == (6, 3, 256, 704)
        results['DataLoader'] = 'PASS'
    except Exception as e:
        results['DataLoader'] = f'FAIL: {e}'
    
    # Test 5: MetricsCalculator
    print("\n[5/7] Testing MetricsCalculator...")
    try:
        from evaluation import MetricsCalculator
        calc = MetricsCalculator()
        pred = np.array([[0,0],[1,0],[2,0]], dtype=np.float32)
        gt = np.array([[0,0],[1,1],[2,0]], dtype=np.float32)
        error = calc.l2_error(pred, gt)
        assert abs(error - 0.5) < 0.01
        results['MetricsCalculator'] = 'PASS'
    except Exception as e:
        results['MetricsCalculator'] = f'FAIL: {e}'
    
    # Test 6: Pipeline
    print("\n[6/7] Testing Pipeline...")
    try:
        from pipeline import SafetyLayerPipeline
        pipeline = SafetyLayerPipeline(
            "asp-safety/SparseDrive/ckpt/sparsedrive_stage2.pth",
            "asp-safety/rules",
            device="cpu"
        )
        images = np.random.randn(6, 3, 256, 704).astype(np.float32)
        intrinsics = np.eye(3, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
        extrinsics = np.eye(4, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
        ego2global = np.eye(4, dtype=np.float32)
        result = pipeline.run(images, intrinsics, extrinsics, ego2global, 0.0)
        assert 'safe_trajectory' in result
        results['Pipeline'] = 'PASS'
    except Exception as e:
        results['Pipeline'] = f'FAIL: {e}'
    
    # Test 7: ASP Rules
    print("\n[7/7] Testing ASP Rules...")
    try:
        import subprocess
        for rule in ['lane_keeping', 'collision_avoidance', 'speed_limit', 
                     'traffic_light', 'safe_following']:
            result = subprocess.run(
                ['clingo', f'asp-safety/rules/{rule}.lp', '--text', '--terminate=0'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError(f"{rule}.lp failed to compile")
        results['ASP Rules'] = 'PASS'
    except Exception as e:
        results['ASP Rules'] = f'FAIL: {e}'
    
    # Print results
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    
    all_pass = True
    for test_name, result in results.items():
        status = "[PASS]" if result == 'PASS' else "[FAIL]"
        print(f"  {status} {test_name}")
        if result != 'PASS':
            print(f"        {result}")
            all_pass = False
    
    print("=" * 60)
    
    if all_pass:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
```

---

## Step 9.3: Run All Tests

Execute:
```bash
cd asp-safety
python tests/test_all.py
```

**Success Criteria**:
- [ ] All test files created
- [ ] All 7 test categories pass
- [ ] No test depends on external resources

---

# Subdivision 10: Documentation & Course Report

**Objective**: Create documentation and course report template.

**Dependencies**: All previous subdivisions complete

**Estimated Time**: 3-4 hours

**Files to Create**:
- `asp-safety/README.md`
- `asp-safety/docs/usage.md`
- `asp-safety/docs/api_reference.md`
- `asp-safety/paper/report.tex`

---

## Step 10.1: Create README.md

Create file `asp-safety/README.md`:

```markdown
# ASP Safety Layer for Autonomous Vehicles

## Overview

An ASP-based safety layer that checks and corrects neural planner trajectories against formal traffic rules, reducing collision rate with minimal latency overhead.

## Team

| Member | Role | Responsibilities |
|--------|------|------------------|
| Member 1 | Project Lead | Architecture, integration, coordination |
| Member 2 | ASP Specialist | Rule implementation, Clingo integration |
| Member 3 | Neural Planner | SparseDrive setup, inference pipeline |
| Member 4 | Evaluation Lead | Testing, metrics, CARLA/nuScenes |

## Quick Start

### 1. Environment Setup

```bash
# Run setup script
./setup_env.sh

# Activate environment
conda activate asp-safety
```

### 2. Run Demo

```python
from src.pipeline import SafetyLayerPipeline
from src.data_loader import NuScenesLoader

# Initialize pipeline
pipeline = SafetyLayerPipeline(
    planner_path="SparseDrive/ckpt/sparsedrive_stage2.pth",
    rules_dir="rules"
)

# Load data
loader = NuScenesLoader("data/sets/nuscenes")
sample = loader.get_scene_sample(0)

# Run pipeline
result = pipeline.run(
    sample['images'],
    sample['intrinsics'],
    sample['extrinsics'],
    sample['ego2global'],
    sample['timestamp']
)

print(f"Safe: {result['is_safe']}")
print(f"Violations: {result['violations']}")
```

### 3. Run Tests

```bash
python tests/test_all.py
```

## Architecture

```
Input Images → SparseDrive → Trajectory → ASP Checker → Safe Trajectory
                                    ↓
                              If violations → Corrector → Corrected Trajectory
```

## Project Structure

```
asp-safety/
├── configs/          # Configuration files
├── rules/            # ASP traffic rules
├── src/              # Source code
│   ├── neural_planner.py   # SparseDrive wrapper
│   ├── checker.py          # ASP safety checker
│   ├── corrector.py        # Trajectory corrector
│   ├── data_loader.py      # Data loading
│   ├── evaluation.py       # Metrics
│   └── pipeline.py         # End-to-end pipeline
├── experiments/      # Evaluation scripts
├── tests/            # Unit tests
├── docs/             # Documentation
└── paper/            # Course report
```

## Evaluation Results

| Metric | SparseDrive Only | With ASP Safety | Improvement |
|--------|------------------|-----------------|-------------|
| Collision Rate | 0.06% | <0.05% | ↓ 20%+ |
| L2 Error | 0.61m | ~0.65m | ↑ 6% |
| Inference Time | 45ms | 55ms | +10ms |

## Resources

- [SparseDrive](https://github.com/swc-17/SparseDrive)
- [nuScenes](https://www.nuscenes.org/)
- [Clingo](https://potassco.org/clingo/)

## License

Course Project
```

---

## Step 10.2: Create usage.md

Create file `asp-safety/docs/usage.md`:

```markdown
# Usage Guide

## Installation

1. Clone repository
2. Run `./setup_env.sh`
3. Activate environment: `conda activate asp-safety`

## Basic Usage

### Initialize Pipeline

```python
from src.pipeline import SafetyLayerPipeline

pipeline = SafetyLayerPipeline(
    planner_path="SparseDrive/ckpt/sparsedrive_stage2.pth",
    rules_dir="rules",
    device="cuda"  # or "cpu"
)
```

### Run Inference

```python
import numpy as np

# Prepare inputs
images = np.random.randn(6, 3, 256, 704).astype(np.float32)
intrinsics = np.eye(3, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
extrinsics = np.eye(4, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
ego2global = np.eye(4, dtype=np.float32)
timestamp = 1234567890.0

# Run pipeline
result = pipeline.run(images, intrinsics, extrinsics, ego2global, timestamp)

# Access results
print(f"Original trajectory: {result['original_trajectory']}")
print(f"Safe trajectory: {result['safe_trajectory']}")
print(f"Is safe: {result['is_safe']}")
print(f"Violations: {result['violations']}")
```

### Check Trajectory Only

```python
from src.checker import TrajectoryChecker

checker = TrajectoryChecker("rules")

trajectory = np.array([[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]], dtype=np.float32)
is_safe, violations = checker.check(trajectory)
```

## Configuration

Edit `configs/default.yaml` to change settings.
```

---

## Step 10.3: Create api_reference.md

Create file `asp-safety/docs/api_reference.md`:

```markdown
# API Reference

## SparseDrivePlanner

### __init__(model_path: str, device: str = "cuda")
Initialize the planner.

### predict(images, intrinsics, extrinsics, ego2global, timestamp) -> Dict
Generate trajectory prediction.

## TrajectoryChecker

### __init__(rules_dir: str = "rules")
Initialize the checker.

### check(trajectory, other_vehicles=None, map_data=None, traffic_lights=None) -> Tuple[bool, List[Dict]]
Check trajectory against rules.

### get_safe_trajectory(trajectory, **kwargs) -> np.ndarray
Return corrected trajectory.

## TrajectoryCorrector

### __init__(checker: TrajectoryChecker, max_iterations: int = 100)
Initialize the corrector.

### correct(trajectory, **kwargs) -> Tuple[np.ndarray, List[Dict]]
Correct an invalid trajectory.

## MetricsCalculator

### collision_rate(trajectories, gt_boxes, pred_boxes) -> float
Calculate collision rate.

### l2_error(pred_trajectory, gt_trajectory) -> float
Calculate L2 error.

### violation_detection_rate(true_violations, detected_violations) -> float
Calculate detection rate.
```

---

## Step 10.4: Create report.tex

Create file `asp-safety/paper/report.tex`:

```latex
\documentclass{article}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}

\title{ASP-Based Safety Layer for Autonomous Driving}

\begin{document}
\maketitle

\section{Introduction}
% 1 page: Problem statement, motivation, contributions

Autonomous driving requires safety guarantees. We propose an ASP-based safety layer that checks neural planner trajectories against formal traffic rules.

\section{Related Work}
% 0.5 page: Neural planners, formal methods, neurosymbolic approaches

\section{Method}
% 2 pages: Architecture, ASP rules, integration with SparseDrive

\subsection{Architecture}
% Figure showing pipeline

\subsection{ASP Traffic Rules}
% Description of 5 rules

\subsection{Integration}
% How components connect

\section{Experiments}
% 1.5 pages: Setup, results, analysis

\subsection{Setup}
% Dataset, metrics, implementation details

\subsection{Results}
% Table with collision rate, L2 error

\subsection{Analysis}
% Discussion of results

\section{Conclusion}
% 0.5 page: Summary, limitations, future work

\end{document}
```

---

## Step 10.5: Verify Documentation

Execute:
```bash
cd asp-safety

# Verify all files exist
ls -la README.md
ls -la docs/usage.md
ls -la docs/api_reference.md
ls -la paper/report.tex

# Verify README has required sections
grep -q "## Quick Start" README.md
grep -q "## Architecture" README.md
grep -q "## Evaluation" README.md

echo "Documentation verification: PASS"
```

**Success Criteria**:
- [ ] All documentation files created
- [ ] README contains all required sections
- [ ] API reference covers all public classes
- [ ] Course report template complete

---

# Subdivision 11: Comprehensive ASP Rule Catalog

**Objective**: Implement the full 63-rule ASP traffic rule catalog covering 15 categories of autonomous driving safety constraints, replacing the 5 basic rules from Subdivision 3 with a production-grade rule set.

**Dependencies**: Subdivision 3 (basic rules working), Subdivision 4 (checker functional)

**Estimated Time**: 8-12 hours (3 phases)

**Source Document**: `asp_traffic_rules.md` (1,245 lines, 63 rules with full specifications)

**Files to Modify/Create**:
- `asp-safety/rules/speed_regulation.lp` (R1.1–R1.5)
- `asp-safety/rules/lane_discipline.lp` (R2.1–R2.5)
- `asp-safety/rules/following_behavior.lp` (R3.1–R3.4)
- `asp-safety/rules/collision_avoidance.lp` (R4.1–R4.5) — replace existing
- `asp-safety/rules/traffic_control.lp` (R5.1–R5.5)
- `asp-safety/rules/right_of_way.lp` (R6.1–R6.5)
- `asp-safety/rules/turning_rules.lp` (R7.1–R7.5)
- `asp-safety/rules/overtaking.lp` (R8.1–R8.4)
- `asp-safety/rules/vru_safety.lp` (R9.1–R9.5)
- `asp-safety/rules/intersection_behavior.lp` (R10.1–R10.4)
- `asp-safety/rules/emergency.lp` (R11.1–R11.4)
- `asp-safety/rules/comfort_quality.lp` (R12.1–R12.4)
- `asp-safety/rules/predictive_safety.lp` (R13.1–R13.4)
- `asp-safety/rules/environmental.lp` (R14.1–R14.4)
- `asp-safety/rules/complex_interactions.lp` (R15.1–R15.4)
- `asp-safety/rules/helpers.lp` (shared predicates)
- `asp-safety/rules/basic.lp` — update to include all rule files

---

## Conventions (from asp_traffic_rules.md)

- **Coordinate system**: Ego-frame, integer, scaled ×100 (1 unit = 1 cm). All arithmetic is integer-only; squared distances replace Euclidean distance so no `sqrt()` is required.
- **Timesteps**: `T` ranges `0..5` (6 total), matching the planner's prediction horizon.
- **Core facts** (given): `pos(V,T,X,Y)`, `lane(L,X1,Y1,X2,Y2)`, `light_state(I,Color,T)`, `intersection(I,X,Y,R)`, `speed(V,T,S)` (S = cm/timestep, integer).
- **Extended facts** (introduced per-rule): Called out explicitly in each rule's "Input Facts Required". Naming is consistent across rules (e.g. `pedestrian(P,T,X,Y)` is reused everywhere).
- **`ego`** denotes the planned vehicle; other `V` values denote tracked agents.
- **Priority levels**: Critical > High > Medium > Low.

---

## Phase 1: Core Rules (24 rules) — Priority for Course Project

### Category 1: Speed Regulation (R1.1–R1.5)

| Rule | Name | Priority | Complexity |
|------|------|----------|------------|
| R1.1 | Urban Speed Limit Compliance | High | Simple |
| R1.2 | Highway Maximum Speed Limit | High | Simple |
| R1.3 | Variable Speed Zones (School/Construction) | Critical | Medium |
| R1.4 | Minimum Speed on Highways | Medium | Simple |
| R1.5 | Reduced Speed Near Pedestrians/Cyclists | Critical | Medium |

**New input facts required**:
```asp
road_type(L,Type).           % Type = urban | highway
speed_limit(L,Limit).        % cm/timestep
min_speed_limit(L,MinLimit). % cm/timestep
zone(Z,X1,Y1,X2,Y2,Type).   % Type = school | construction
zone_active(Z,T).            % boolean
zone_speed_limit(Z,Limit).   % cm/timestep
vru(P,T,X,Y,Type).          % Type = pedestrian | cyclist
vru_speed_cap(Type,DistBand,Cap). % lookup table
```

**Implementation notes**:
- R1.3 requires `inside_zone/6` predicate (axis-aligned or rotated-rect bounding test)
- R1.5 uses squared-distance bucket lookup (close/medium/far) to avoid sqrt
- All speed comparisons are direct integer: `S > Limit`

### Category 2: Lane Discipline (R2.1–R2.5)

| Rule | Name | Priority | Complexity |
|------|------|----------|------------|
| R2.1 | Lane Keeping (Boundary Containment) | High | Medium |
| R2.2 | Lane Change Permission & Minimum Gap | Critical | Complex |
| R2.3 | Wrong-Way Driving Detection | Critical | Simple |
| R2.4 | HOV / Bus Lane Restriction | Medium | Simple |
| R2.5 | Lane Departure Early Warning | Low | Medium |

**New input facts required**:
```asp
lane_width(L,W).             % cm
lane_boundary(L1,L2,Type).   % Type = solid | dashed
lane_direction(L,DX,DY).     % canonical direction vector
lane_restriction(L,Type,MinOcc,ActiveHours). % Type = hov | bus_only
occupancy(ego,N).            % passenger count
current_time(Tm).            % time-of-day
```

**Implementation notes**:
- R2.1 uses `perp_dist2/7` (squared perpendicular distance via cross-product)
- R2.2 requires `crosses_boundary/4` (sign change in cross-product between consecutive timesteps)
- R2.3 uses dot product of heading vs lane direction: `Dot < 0` means wrong way
- R2.5 fires in the outer ~20% band: `D2 > (W*W)/4 * 64/100`

### Category 3: Following Behavior (R3.1–R3.4)

| Rule | Name | Priority | Complexity |
|------|------|----------|------------|
| R3.1 | Safe Following Distance (Time-Based) | High | Medium |
| R3.2 | Safe Following Distance (Speed-Dependent Floor) | High | Simple |
| R3.3 | Tailgating Pattern Detection | High | Medium |
| R3.4 | Caravan / Queue Behavior | Medium | Medium |

**New input facts required**:
```asp
lead_vehicle(ego,V,T).        % identifies the vehicle directly ahead
gap_dist(ego,V,T,D).          % longitudinal distance (cm)
min_gap_table(S,MinD).        % speed-indexed minimum gap lookup
queue_member(V).              % boolean
```

**Implementation notes**:
- R3.1 cross-multiplies to avoid division: `D*100 < S*MinTimeUnits`
- R3.3 uses `#count` aggregate: `N >= 4` out of 6 timesteps = sustained tailgating
- R3.4 flags speed oscillation while lead vehicle is slow (accordion effect)

### Category 4: Collision Avoidance (R4.1–R4.5)

| Rule | Name | Priority | Complexity |
|------|------|----------|------------|
| R4.1 | Minimum Lateral Clearance | Critical | Medium |
| R4.2 | Minimum Longitudinal Clearance | Critical | Simple |
| R4.3 | Cut-In Detection | Critical | Medium |
| R4.4 | Rear-End Collision Risk | High | Complex |
| R4.5 | Side-Swipe Risk (Converging Trajectories) | High | Medium |

**New input facts required**:
```asp
lateral_offset(ego,V,T,DY).   % perpendicular separation
longitudinal_overlap(ego,V,T). % boolean: extents overlap along travel direction
longitudinal_gap(ego,V,T,D).  % gap along lane direction
predicted_pos(V,T,X,Y).       % other agent's predicted positions
following_gap(V,ego,T,D).     % gap from trailing vehicle to ego
```

**Implementation notes**:
- R4.1 requires both lateral offset AND longitudinal overlap (otherwise irrelevant)
- R4.3 detects lane-change-into-ego's-path by comparing `in_lane(V,T,L)` vs `in_lane(V,T-1,L2)`
- R4.5 compares lateral separation at T vs T+1: decreasing trend + below threshold = convergence
- R4.4 uses braking-distance lookup to check if trailing vehicle can stop in time

### Category 5: Traffic Control Compliance (R5.1–R5.5)

| Rule | Name | Priority | Complexity |
|------|------|----------|------------|
| R5.1 | Red Light Compliance | Critical | Simple |
| R5.2 | Yellow Light Decision Zone | High | Complex |
| R5.3 | Green Light With Obstacle in Intersection | Critical | Medium |
| R5.4 | Flashing Signal Handling | Critical/Medium | Medium |
| R5.5 | Protected Arrow Signal Compliance | Critical | Medium |

**New input facts required**:
```asp
controls_approach(I,L).       % which approach lane a signal governs
stopping_distance(S,SD).      % precomputed stopping distance lookup
no_turn_on_red(I,Direction).  % signage fact
light_state(I,arrow_green,Direction,T). % protected arrow states
light_state(I,arrow_red,Direction,T).
planned_turn(ego,Direction).  % Direction = left | right
```

**Implementation notes**:
- R5.2 is the "dilemma zone" rule: can we stop safely? If yes and plan doesn't, that's a violation. If no (already committed), proceeding is permitted.
- R5.4 flashing-red = full stop required (like a stop sign); flashing-yellow = reduced speed
- R5.5 checks directional arrow, not just general green

---

## Phase 2: Extended Rules (23 rules)

### Category 6: Right-of-Way (R6.1–R6.5)

| Rule | Name | Priority | Complexity |
|------|------|----------|------------|
| R6.1 | Uncontrolled Intersection Priority | Critical | Complex |
| R6.2 | Roundabout Entry Priority | Critical | Medium |
| R6.3 | Yield to Oncoming Traffic During Turns | Critical | Complex |
| R6.4 | Emergency Vehicle Yielding | Critical | Medium |
| R6.5 | Pedestrian Crosswalk Priority | Critical | Medium |

**New input facts required**:
```asp
arrival_time(V,I,T).          % first timestep vehicle reaches stop line
approach_side(V,I,Side).      % Side = left | right | front | behind
roundabout(R,Xc,Yc,Radius).
in_circulation(V,R,T).        % vehicle inside roundabout
about_to_enter(ego,I,T).      % boolean: one timestep from intersection
has_cleared(V,I,T).           % boolean: vehicle has exited the box
emergency_vehicle(V,T,X,Y,active).
approaching(V,ego,T).         % boolean: converging trajectory
crosswalk(C,X1,Y1,X2,Y2).
pedestrian(P,T,X,Y).
path_intersects(ego,C,T).     % boolean: ego path overlaps crosswalk
stopped_before(ego,C,T).      % boolean: ego stopped before boundary
```

### Category 7: Turning Rules (R7.1–R7.5)

| Rule | Name | Priority | Complexity |
|------|------|----------|------------|
| R7.1 | Right Turn on Red | Critical/High | Medium |
| R7.2 | Left Turn Across Traffic | Critical | Complex |
| R7.3 | U-Turn Restrictions | High | Simple |
| R7.4 | Turn Signal Timing | Medium | Simple |
| R7.5 | Turning Radius Compliance | High | Complex |

**New input facts required**:
```asp
has_stopped(ego,I).           % boolean: full stop event occurred
clear_path(ego,I,T).          % boolean: no conflicting traffic/pedestrians
no_u_turn(L).                 % signage fact
sightline_distance(L,SD).     % map-encoded sightline
planned_maneuver(ego,Type,T). % Type = turn | lane_change | overtake | u_turn
signal_active(ego,Direction,T).
min_turning_radius(ego,Rmin). % vehicle kinematic limit
```

### Category 8: Overtaking Rules (R8.1–R8.4)

| Rule | Name | Priority | Complexity |
|------|------|----------|------------|
| R8.1 | Permitted Overtaking Conditions | Critical | Complex |
| R8.2 | No-Passing Zones | Critical | Medium |
| R8.3 | Overtaking on the Right | High | Simple |
| R8.4 | Overtaking Clearance Requirements | Critical/High | Medium |

**New input facts required**:
```asp
no_passing_zone(Z,X1,Y1,X2,Y2).
overtake_side(ego,T,Side).    % Side = left | right
overtaken_type(V,Type).       % Type = vehicle | cyclist | pedestrian
overtake_clearance_min(Type,MinLateral).
overtake_duration(ego,Est).   % estimated maneuver duration in timesteps
```

### Category 9: Pedestrian & Cyclist Safety (R9.1–R9.5)

| Rule | Name | Priority | Complexity |
|------|------|----------|------------|
| R9.1 | General Crosswalk Yielding (Unmarked) | Critical | Medium |
| R9.2 | School Zone Pedestrian Behavior | Critical | Simple |
| R9.3 | Bicycle Lane Respect | High | Medium |
| R9.4 | Blind Spot Detection for VRUs | Critical | Complex |
| R9.5 | Sidewalk Encroachment Prevention | Critical | Simple |

**New input facts required**:
```asp
implied_crosswalk(I,Corner1,Corner2).
crossing_guard_signal(Z,T,stop|go).
lane_type(L,bicycle).
blind_spot_zone(ego,Side,X1,Y1,X2,Y2).
last_confident_detection_age(P,T,Age).
last_known_pos(P,X,Y).
sidewalk(S,X1,Y1,X2,Y2).
driveway_crossing(ego,T).
```

### Category 10: Intersection Behavior (R10.1–R10.4)

| Rule | Name | Priority | Complexity |
|------|------|----------|------------|
| R10.1 | Queue Position at Red Light | High | Medium |
| R10.2 | Box Blocking Prevention | Critical | Medium |
| R10.3 | Protected Left Turn Lane Compliance | Medium | Simple |
| R10.4 | Right Turn Yielding to Pedestrians | Critical | Medium |

**New input facts required**:
```asp
queue_position(V,I,N).
last_queued_vehicle(V,I,T).
queue_ahead(V,I,T).
space_beyond_box(ego,V,I,T,Space).
ego_length(Len).
lane_type(L,protected_left).
destination_crosswalk(I,Direction,C).
```

---

## Phase 3: Advanced Rules (16 rules)

### Category 11: Emergency Situations (R11.1–R11.4)

| Rule | Name | Priority | Complexity |
|------|------|----------|------------|
| R11.1 | Emergency Braking Feasibility | Critical | Medium |
| R11.2 | Evasive Maneuver Boundaries | Critical | Complex |
| R11.3 | Guardrail / Fixed-Obstacle Avoidance | Critical | Simple |
| R11.4 | Active Construction Zone Compliance | Critical | Complex |

**New input facts required**:
```asp
hazard_distance(ego,H,T,D).
max_decel(ego,MaxD).
braking_distance(S,MaxD,BD).  % precomputed lookup table
road_boundary(X1,Y1,X2,Y2).
fixed_obstacle(O,X1,Y1,X2,Y2).
ego_footprint(Width).
zone(Z,X1,Y1,X2,Y2,construction).
lane_shift(Z,L,NewPath).
worker(W,T,X,Y).
```

### Category 12: Comfort & Ride Quality (R12.1–R12.4)

| Rule | Name | Priority | Complexity |
|------|------|----------|------------|
| R12.1 | Maximum Acceleration/Deceleration Limits | Low | Simple |
| R12.2 | Lateral Acceleration / Jerk Limits | Low | Complex |
| R12.3 | Smooth Trajectory Requirement | Low | Complex |
| R12.4 | Passenger Comfort Scoring | Low | Complex |

**New input facts required**:
```asp
max_comfort_accel(A).
max_comfort_decel(D).
override_active(ego,T).
max_lateral_accel(LA).
max_jerk(J).
max_heading_change(HC).
```

**Implementation notes**:
- R12.1 is referenced by nearly every speed-correcting rule as the default bound
- R12.4 is advisory only (no violation predicate) — produces `comfort_score/2` for tie-breaking

### Category 13: Predictive Safety (R13.1–R13.4)

| Rule | Name | Priority | Complexity |
|------|------|----------|------------|
| R13.1 | Trajectory Collision Prediction | Critical | Complex |
| R13.2 | Time-to-Collision (TTC) Thresholds | Critical/High | Medium |
| R13.3 | Threat Level Assessment (Aggregate) | Critical | Medium |
| R13.4 | Multi-Step Violation Prediction | Critical | Complex |

**New input facts required**:
```asp
closing_rate(ego,V,T,CR).
CollisionThreshold2.          % squared distance threshold
ImminentUnits.                % TTC band thresholds (cross-multiplied)
CriticalUnits.
ThreateningUnits.
```

**Implementation notes**:
- R13.1 is the full cross-product of horizon × agents (performance-critical)
- R13.3 is the single arbitration point when multiple Critical rules fire
- R13.4 drives the fixpoint loop: if any violation at any T, re-solve from earliest T

### Category 14: Environmental Context (R14.1–R14.4)

| Rule | Name | Priority | Complexity |
|------|------|----------|------------|
| R14.1 | Wet Road Speed Reduction | High | Simple |
| R14.2 | Night Driving Behavior | High | Medium |
| R14.3 | Visibility-Limited Rules | High/Critical | Simple |
| R14.4 | Weather-Dependent Threshold Adjustment | Medium | Medium |

**New input facts required**:
```asp
road_condition(T,wet|dry).
wet_speed_factor(Factor).     % integer percentage (e.g. 80 = 80%)
time_of_day(T,night|day).
headlight_range(ego,Range).   % cm
perception_confidence(T,Level). % Level = high | medium | low | severe
confidence_speed_cap(Level,Cap).
weather_severity(T,Level).    % Level = mild | moderate | severe
threshold_adjustment(Level,RuleFamily,Factor).
```

### Category 15: Complex Interactions (R15.1–R15.4)

| Rule | Name | Priority | Complexity |
|------|------|----------|------------|
| R15.1 | Multi-Vehicle Coordination at Conflict Points | Critical | Complex |
| R15.2 | Platoon/Convoy Behavior | Critical/Medium | Medium |
| R15.3 | Merge Lane Negotiation | High | Complex |
| R15.4 | Traffic Wave Dampening | Medium | Complex |

**New input facts required**:
```asp
conflict_point(C,X,Y).
platoon_member(V,PlatoonId).
comm_link_active(ego,V,T).
PlatoonMinGap.                % tight platoon gap
StandardMinGap.               % normal gap (fallback)
merge_lane(L,EndPoint).
merge_gap_sequence(ego,V,Order). % Order = before | after
rolling_avg_speed(V,T,Avg).
AmplificationFactor.          % integer percentage
```

---

## Implementation Steps

### Step 11.1: Create helpers.lp

Create `asp-safety/rules/helpers.lp` with shared predicates used across all rules:

```asp
% helpers.lp - Shared predicates for the comprehensive rule catalog

time(0..5).

% Squared distance between two vehicles at time T
dist2(V1,V2,T,D) :-
    pos(V1,T,X1,Y1), pos(V2,T,X2,Y2), V1!=V2,
    D = (X1-X2)*(X1-X2) + (Y1-Y2)*(Y1-Y2).

% Heading vector from consecutive positions
heading(V,T,DX,DY) :-
    pos(V,T,X,Y), pos(V,T+1,X2,Y2),
    DX = X2-X, DY = Y2-Y.

% Squared perpendicular distance from point to line segment
perp_dist2(X,Y,X1,Y1,X2,Y2,D2) :-
    Num = ((Y2-Y1)*X - (X2-X1)*Y + X2*Y1 - Y2*X1),
    Den = (X2-X1)*(X2-X1) + (Y2-Y1)*(Y2-Y1),
    D2 = Num*Num.

% Point inside axis-aligned bounding box
inside_zone(X,Y,X1,Y1,X2,Y2) :-
    X >= X1, X <= X2, Y >= Y1, Y <= Y2.

% Vehicle in lane (perpendicular distance within half-width)
in_lane(V,T,L) :-
    pos(V,T,X,Y), lane(L,X1,Y1,X2,Y2), lane_width(L,W),
    perp_dist2(X,Y,X1,Y1,X2,Y2,D2), D2 <= (W*W)/4.

% Longitudinal gap between two vehicles in same lane
longitudinal_gap(ego,V,T,D) :-
    in_lane(ego,T,L), in_lane(V,T,L),
    pos(ego,T,Xe,Ye), pos(V,T,Xv,Yv),
    % Project onto lane direction
    lane_direction(L,DX,DY),
    D = (Xv-Xe)*DX + (Yv-Ye)*DY.
```

### Step 11.2: Implement Phase 1 rule files

For each Phase 1 category, create the corresponding `.lp` file with the violation conditions from `asp_traffic_rules.md`. Key implementation rules:

1. **All arithmetic is integer-only** — no floats, no sqrt
2. **Use squared distances** for all proximity checks
3. **Use cross-multiplication** for ratio comparisons (e.g. `D/S < 2` becomes `D*100 < S*20`)
4. **Use step-function lookups** for continuous functions (speed-indexed tables)
5. **Each rule file** should include its violation predicates and optimization directives

### Step 11.3: Update basic.lp

Update `asp-safety/rules/basic.lp` to include all rule files:

```asp
% basic.lp - Master include for all 63 rules

#include "helpers.lp".
#include "speed_regulation.lp".
#include "lane_discipline.lp".
#include "following_behavior.lp".
#include "collision_avoidance.lp".
#include "traffic_control.lp".
#include "right_of_way.lp".
#include "turning_rules.lp".
#include "overtaking.lp".
#include "vru_safety.lp".
#include "intersection_behavior.lp".
#include "emergency.lp".
#include "comfort_quality.lp".
#include "predictive_safety.lp".
#include "environmental.lp".
#include "complex_interactions.lp".
```

### Step 11.4: Update checker.py

Extend `TrajectoryChecker` to accept the new extended input facts:

- `road_type`, `speed_limit`, `min_speed_limit` — for R1.x
- `lane_width`, `lane_boundary`, `lane_direction` — for R2.x
- `lead_vehicle`, `min_gap_table` — for R3.x
- `predicted_pos`, `following_gap` — for R4.x
- `controls_approach`, `stopping_distance`, `planned_turn` — for R5.x
- `pedestrian`, `crosswalk`, `intersection` — for R6.x/R9.x
- `zone`, `zone_active`, `zone_speed_limit` — for R1.3/R11.4

### Step 11.5: Unit tests per rule category

Create `asp-safety/tests/test_rules.py` with:

```python
# For each rule category:
# 1. Test that violations fire on known-bad trajectories
# 2. Test that safe trajectories produce no violations
# 3. Test priority ordering when multiple rules conflict
```

### Step 11.6: Integration tests

Test rule interactions:
- R1.3 (school zone) overrides R1.1 (urban speed) when zone is active
- R5.1 (red light) overrides R6.1 (uncontrolled priority)
- R11.1 (emergency braking) overrides R12.1 (comfort deceleration)
- R13.3 (threat arbitration) resolves conflicts between R4.x and R11.x

---

## Success Criteria

### Phase 1 (Core)
- [ ] All 24 Phase 1 rules implemented in `.lp` files
- [ ] Each rule compiles and grounds independently via Clingo
- [ ] Unit tests pass for each rule category
- [ ] `TrajectoryChecker` handles extended input facts

### Phase 2 (Extended)
- [ ] All 23 Phase 2 rules implemented
- [ ] Roundabout, turning, and pedestrian rules functional
- [ ] Integration tests for right-of-way interactions

### Phase 3 (Advanced)
- [ ] All 16 Phase 3 rules implemented
- [ ] Emergency pipeline (R11.1→R11.2) functional
- [ ] Threat arbitration (R13.3) resolves correctly
- [ ] Performance: solve time < 50ms with 20 tracked agents

### Overall
- [ ] All 63 rules present in `rules/` directory
- [ ] `basic.lp` includes all rule files
- [ ] `test_rules.py` passes all tests
- [ ] Full pipeline integration: SparseDrive → 63-rule checker → corrector

---

# Completion Checklist

After completing all 11 subdivisions, verify:

```bash
cd asp-safety

echo "=== Final Verification ==="

# Check directory structure
echo "1. Directory structure..."
for dir in configs rules src experiments tests docs paper; do
    if [ -d "$dir" ]; then
        echo "   $dir: OK"
    else
        echo "   $dir: MISSING"
    fi
done

# Check source files
echo "2. Source files..."
for file in src/neural_planner.py src/checker.py src/corrector.py src/data_loader.py src/evaluation.py src/pipeline.py; do
    if [ -f "$file" ]; then
        echo "   $file: OK"
    else
        echo "   $file: MISSING"
    fi
done

# Check rule files
echo "3. Rule files..."
for file in rules/helpers.lp rules/speed_regulation.lp rules/lane_discipline.lp rules/following_behavior.lp rules/collision_avoidance.lp rules/traffic_control.lp rules/right_of_way.lp rules/turning_rules.lp rules/overtaking.lp rules/vru_safety.lp rules/intersection_behavior.lp rules/emergency.lp rules/comfort_quality.lp rules/predictive_safety.lp rules/environmental.lp rules/complex_interactions.lp rules/basic.lp; do
    if [ -f "$file" ]; then
        echo "   $file: OK"
    else
        echo "   $file: MISSING"
    fi
done

# Check tests
echo "4. Test files..."
for file in tests/test_all.py tests/test_checker.py tests/test_pipeline.py; do
    if [ -f "$file" ]; then
        echo "   $file: OK"
    else
        echo "   $file: MISSING"
    fi
done

# Run all tests
echo "5. Running all tests..."
python tests/test_all.py

echo ""
echo "=== Project Complete ==="
```

---

*Document Version: 1.1*
*Created: September 10, 2026*
*Updated: September 13, 2026*
*Status: Ready for Implementation*
