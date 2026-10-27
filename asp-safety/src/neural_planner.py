"""SparseDrive Neural Planner Wrapper.

This module provides a wrapper class for SparseDrive inference.
The class loads pre-trained weights and generates trajectory predictions.

SparseDrive outputs:
- final_planning: (6, 2) - cumulative (x,y) at t=0.5,1.0,1.5,2.0,2.5,3.0s
- boxes_3d: (N, 7) - [x, y, z, w, l, h, yaw] in lidar frame
- scores_3d: (N,) - detection confidence scores

Integration requires:
1. SparseDrive repo cloned (done)
2. Custom CUDA ops compiled: cd SparseDrive/projects/mmdet3d_plugin/ops && python setup.py develop
3. mmdet3d dependencies installed (see SparseDrive/requirement.txt)
4. Pre-trained weights downloaded
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, List


class SparseDrivePlanner:
    """Wrapper for SparseDrive model inference.
    
    This class loads a pre-trained SparseDrive model and provides
    a simple interface for trajectory prediction.
    
    The model outputs trajectory as cumulative (x, y) positions in the
    ego/lidar coordinate frame at 0.5s intervals for 3 seconds ahead.
    
    Attributes:
        model_path: Path to .pth weights file
        device: PyTorch device ("cuda" or "cpu")
        model: Loaded SparseDrive model (None if not loaded)
        use_placeholder: Whether using placeholder data
    """
    
    # SparseDrive output constants
    EGO_FUT_TS = 6       # number of planning timesteps
    EGO_FUT_INTERVAL = 0.5  # seconds between timesteps
    EGO_FUT_MODES = 6    # number of planning modes
    NUM_COMMANDS = 3      # navigation command heads; index Convention
    # UNVERIFIED (cmd sweep 2026-09-26: no consistent left/straight/right
    # mapping -- all heads curve +y on the scene-0 left turn, cmd2 wins
    # straight scenes, cmd1 never wins). Keep default [1,0,0]; revisit
    # under N-2 warm inference, which may sharpen head selection.
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """Initialize the planner.
        
        Args:
            model_path: Path to .pth weights file
            device: "cuda" for GPU, "cpu" for CPU
        """
        self.model_path = Path(model_path)
        self.device = device
        self.use_placeholder = not self.model_path.exists()
        self.model = None
        
        if self.use_placeholder:
            print(f"SparseDrive weights not found at {model_path}")
            print("Using placeholder planner. Download weights for real inference.")
        else:
            try:
                self.model = self._load_model()
            except Exception as e:
                print(f"Failed to load SparseDrive model: {e}")
                print("Falling back to placeholder planner.")
                self.use_placeholder = True
    
    def _load_model(self):
        """Load the SparseDrive model using mmdet3d framework.
        
        Requires:
        - mmdet3d installed
        - Custom CUDA ops compiled
        - mmcv_full installed
        
        Returns:
            Loaded model in eval mode
        """
        try:
            import sys
            sparse_drive_root = str(Path(__file__).parent.parent / "SparseDrive")
            if sparse_drive_root not in sys.path:
                sys.path.insert(0, sparse_drive_root)
            
            import torch
            # SparseDrive registers into mmdet's registries (see
            # tools/test.py: `from mmdet.models import build_detector`
            # after importing the plugin). Importing the plugin package
            # registers SparseDrive + all submodules.
            import projects.mmdet3d_plugin  # noqa: F401
            from mmdet.models import build_detector
            
            config_path = Path(sparse_drive_root) / "projects/configs/sparsedrive_small_stage2.py"
            if not config_path.exists():
                raise RuntimeError(
                    f"SparseDrive config not found: {config_path}"
                )
            
            # Load config
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", str(config_path))
            if spec is None or spec.loader is None:
                raise RuntimeError(
                    f"Cannot load SparseDrive config module from {config_path}"
                )
            cfg_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cfg_module)
            
            # Anchor .npy paths in the config are relative ('data/kmeans').
            # Resolve them against the SparseDrive root so loading works
            # from any CWD. Synthetic stand-ins (correct shapes, zeros)
            # are acceptable: the checkpoint overwrites every anchor
            # Parameter via load_state_dict below.
            def _fix_anchors(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if isinstance(v, str) and v.startswith('data/kmeans'):
                            obj[k] = str(Path(sparse_drive_root) / v)
                        else:
                            _fix_anchors(v)
                elif isinstance(obj, (list, tuple)):
                    for v in obj:
                        _fix_anchors(v)
            _fix_anchors(cfg_module.model)
            
            # Build model
            model = build_detector(cfg_module.model)
            
            # Load checkpoint (weights_only blocks arbitrary pickle payloads)
            checkpoint = torch.load(
                self.model_path, map_location="cpu", weights_only=True)
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            
            model = model.to(self.device).eval()
            if self.device == "cpu":
                # mmcv auto_fp16 casts to Half, whose matmuls have no CPU
                # kernel. Disable the wrappers for CPU fp32 inference
                # (numerics differ slightly from fp16-CUDA reference;
                # golden values are pinned as CPU-fp32).
                for m in model.modules():
                    if hasattr(m, 'fp16_enabled'):
                        m.fp16_enabled = False
            print(f"SparseDrive model loaded from {self.model_path}")
            return model
            
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import SparseDrive dependencies: {e}\n"
                "Install requirements: cd SparseDrive && pip install -r requirement.txt\n"
                "Compile ops: cd SparseDrive/projects/mmdet3d_plugin/ops && python setup.py develop"
            )
    
    def predict(
        self,
        images: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        ego2global: np.ndarray,
        timestamp: float,
        nav_cmd: Optional[np.ndarray] = None,
        ego_status: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Generate trajectory prediction.
        
        Args:
            images: Camera images, shape (6, 3, 256, 704), uint8 or float32
                (NuScenesLoader emits uint8; normalized float32 accepted too)
                Order: [CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,
                        CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT]
            intrinsics: Camera intrinsic matrices, shape (6, 3, 3), float32
            extrinsics: Camera extrinsic matrices (camera-to-vehicle), shape (6, 4, 4), float32
            ego2global: Vehicle-to-global transformation, shape (4, 4), float32
            timestamp: Unix timestamp, float
            nav_cmd: Optional one-hot navigation command, shape (3,)
                Index convention UNVERIFIED (see NUM_COMMANDS note).
                Defaults to [1, 0, 0]. Selects the
                command-conditioned planning head
                (decoder.py: `cmd = gt_ego_fut_cmd.argmax(-1)`).
            ego_status: Optional ego dynamic state, shape (18,)
                [vx, vy, ...]. Defaults to zeros. Only consumed across
                frames via the temporal queue
                (instance_queue.prev_ego_status); with cold-reset
                single-frame inference it has no effect (full derivation
                is N-2 work).
                
        Returns:
            Dict with keys:
                'trajectory': np.ndarray, shape (6, 2), float32
                    Predicted [x, y] positions at t=0.5,1.0,1.5,2.0,2.5,3.0s
                    in ego/lidar frame (x=forward, y=left)
                'boxes': np.ndarray, shape (N, 7), float32
                    Detected 3D boxes [x, y, z, w, l, h, yaw]
                'scores': np.ndarray, shape (N,), float32
                    Detection confidence scores [0, 1]
        """
        expected_shape = (6, 3, 256, 704)
        if images.shape != expected_shape:
            raise ValueError(f"Expected images shape {expected_shape}, got {images.shape}")
        
        if self.use_placeholder or self.model is None:
            return self._generate_placeholder()

        # Independent frames by default: the temporal banks need img_metas
        # (per-camera T_global + timestamp) that single-frame callers don't
        # supply, so a warm bank crashes the second call with
        # KeyError 'img_metas'. Reset to cold state; the N-2 warm-queue
        # harness will drive consecutive frames through an explicit
        # temporal API that provides those metas instead.
        self.reset_temporal()
        return self._run_inference(images, intrinsics, extrinsics,
                                   ego2global, timestamp,
                                   nav_cmd=nav_cmd, ego_status=ego_status)
    
    def begin_sequence(self):
        """Start a temporally-linked frame sequence (N-2).

        Clears the banks so the first predict_frame() call is cold;
        subsequent calls fuse history until end_sequence(). For
        independent frames (open-loop eval, tests) use predict(), which
        cold-resets on every call.
        """
        self.reset_temporal()

    def end_sequence(self):
        """End a sequence; drop history so the next scene starts cold."""
        self.reset_temporal()

    def predict_frame(
        self,
        images: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        ego2global: np.ndarray,
        timestamp: float,
        nav_cmd: Optional[np.ndarray] = None,
        ego_status: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Predict one frame of a sequence WITHOUT resetting history (N-2).

        Same contract as predict(), but the temporal banks persist across
        calls: the caller must supply per-frame ego_status (use
        data_loader.estimate_ego_status) and this method builds the
        img_metas (T_global per frame) the banks need -- the missing
        piece that crashes warm predict() calls (KeyError 'img_metas').
        Pair with begin_sequence()/end_sequence().
        """
        expected_shape = (6, 3, 256, 704)
        if images.shape != expected_shape:
            raise ValueError(f"Expected images shape {expected_shape}, got {images.shape}")

        if self.use_placeholder or self.model is None:
            return self._generate_placeholder()

        return self._run_inference(images, intrinsics, extrinsics,
                                   ego2global, timestamp,
                                   nav_cmd=nav_cmd, ego_status=ego_status,
                                   temporal=True)

    def reset_temporal(self):
        """Clear cross-frame temporal state (instance banks/queues).

        The model fuses temporal history when cached anchors exist;
        across INDEPENDENT scenes that history is wrong (and crashes
        on missing img_metas). Call once per scene for open-loop eval.
        Only touches modules exposing bank/queue state (reset +
        cached_anchor/anchor_queue attrs); nothing else defines reset().
        """
        if self.model is None:
            return
        for m in self.model.modules():
            if callable(getattr(m, 'reset', None)) and (
                    hasattr(m, 'cached_anchor')
                    or hasattr(m, 'anchor_queue')):
                m.reset()

    def _run_inference(
        self,
        images: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        ego2global: np.ndarray,
        timestamp: float,
        nav_cmd: Optional[np.ndarray] = None,
        ego_status: Optional[np.ndarray] = None,
        temporal: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Run actual SparseDrive inference.
        
        The model expects:
        - img: (B, 6, 3, 256, 704) normalized images
        - projection_mat: (B, 6, 4, 4) lidar2img = extrinsics @ intrinsics
        - image_wh: (B, 6, 2) image width, height per camera
        - ego_status: (B, D) ego vehicle status vector
        - gt_ego_fut_cmd: (B, 3) navigation command one-hot
        - timestamp: (B,) frame timestamps
        """
        import torch
        
        # Normalize images (nuScenes standard)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        img_normalized = (images - mean[None, :, None, None]) / std[None, :, None, None]
        
        # Lidar-to-image projection, K*[R|t] embedded in 4x4 (RQ-2).
        projection_mat = self._build_projection_mat(intrinsics, extrinsics)
        
        # Image dimensions (all cameras same size)
        image_wh = np.array([[704, 256]] * 6, dtype=np.float32)
        
        # Ego status: [vx, vy, ax, ay, yaw_rate, ...] (simplified).
        # Cold single-frame inference ignores it (see predict docstring);
        # zeros unless the caller supplies measured state.
        if ego_status is None:
            ego_status = np.zeros(18, dtype=np.float32)
        else:
            ego_status = np.asarray(ego_status, dtype=np.float32)
            assert ego_status.shape == (18,), \
                f"ego_status must be (18,), got {ego_status.shape}"

        # Navigation command: one-hot over [straight, left, right].
        # Selects the planning head; hardcoded straight by default.
        if nav_cmd is None:
            gt_ego_fut_cmd = np.array([1, 0, 0], dtype=np.float32)
        else:
            gt_ego_fut_cmd = np.asarray(nav_cmd, dtype=np.float32)
            assert gt_ego_fut_cmd.shape == (3,), \
                f"nav_cmd must be (3,), got {gt_ego_fut_cmd.shape}"
        
        with torch.no_grad():
            inputs = {
                'img': torch.tensor(img_normalized).unsqueeze(0).to(self.device),
                'projection_mat': torch.tensor(projection_mat).unsqueeze(0).to(self.device),
                'image_wh': torch.tensor(image_wh).unsqueeze(0).to(self.device),
                'ego_status': torch.tensor(ego_status).unsqueeze(0).to(self.device),
                'gt_ego_fut_cmd': torch.tensor(gt_ego_fut_cmd).unsqueeze(0).to(self.device),
                'timestamp': torch.tensor([timestamp], dtype=torch.float64).to(self.device),
            }
            if temporal:
                # Temporal banks fuse history via relative ego motion:
                # T_temp2cur = T_global_inv(new) @ T_global(old), plus
                # timestamp deltas (instance_bank.get / instance_queue.get).
                # ego2global IS the lidar-frame T_global, so no
                # per-camera expansion is needed (batch-size-1 metas).
                g = np.asarray(ego2global, dtype=np.float64)
                inputs['img_metas'] = [{
                    'T_global': g,
                    'T_global_inv': np.linalg.inv(g),
                }]

            result = self.model(**inputs)
        
        return self._postprocess(result)
    
    @staticmethod
    def _build_projection_mat(
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ) -> np.ndarray:
        """Build lidar-to-image projection matrices (RQ-2).
        
        projection = K @ [R|t] embedded in a 4x4, where K is the 3x3
        camera intrinsic matrix and [R|t] is the top 3 rows of the 4x4
        lidar-to-camera rigid transform. The previous code computed
        ``extrinsics @ intrinsics`` ((4,4) @ (3,3)), which raises a
        shape-mismatch error as written.
        
        NOTE: this assumes ``extrinsics`` maps lidar-frame points into
        the camera frame. Confirm the convention against the SparseDrive
        data pipeline (``nuscenes_3d_dataset.py`` / ``loading.py``) when
        real calibration flows through; the function is pure numpy so
        the mapping is unit-testable without torch.
        
        Args:
            intrinsics: (6, 3, 3) camera intrinsic matrices
            extrinsics: (6, 4, 4) lidar-to-camera transforms
            
        Returns:
            (6, 4, 4) projection matrices, bottom row [0, 0, 0, 1]
        """
        if intrinsics.shape != (6, 3, 3):
            raise ValueError(
                f"Expected intrinsics shape (6, 3, 3), got {intrinsics.shape}"
            )
        if extrinsics.shape != (6, 4, 4):
            raise ValueError(
                f"Expected extrinsics shape (6, 4, 4), got {extrinsics.shape}"
            )
        projection_mat = np.zeros((6, 4, 4), dtype=np.float32)
        for cam_idx in range(6):
            projection_mat[cam_idx, :3, :4] = (
                intrinsics[cam_idx] @ extrinsics[cam_idx][:3, :]
            )
            projection_mat[cam_idx, 3, 3] = 1.0
        return projection_mat
    
    @staticmethod
    def _to_numpy(arr):
        """Convert a tensor/array output to numpy (P-13).

        Handles torch tensors when torch is installed; passes numpy
        arrays (and lists) through otherwise, so the postprocess path
        is testable without torch.
        """
        try:
            import torch
            if isinstance(arr, torch.Tensor):
                return arr.detach().cpu().numpy()
        except ImportError:
            pass
        return np.asarray(arr)
    
    def _postprocess(self, raw_output) -> Dict[str, np.ndarray]:
        """Convert SparseDrive raw output to standard format.
        
        Fail-loud (P-13): a missing or misshapen ``final_planning``
        raises instead of falling back to the placeholder, so model
        errors surface at integration time instead of hiding behind
        synthetic data. Empty boxes/scores remain legal (no detections).
        
        Args:
            raw_output: List of dicts from model.forward()
                Each dict has 'img_bbox' key containing:
                - final_planning: (6, 2) cumulative ego trajectory
                - boxes_3d: (N, 10) decoded boxes
                    [x,y,z,w,l,h,yaw,vx,vy,vz]; sliced to [:, :7]
                - scores_3d: (N,) detection scores
        
        Returns:
            Standardized output dict
        """
        if not raw_output or 'img_bbox' not in raw_output[0]:
            raise KeyError(
                "model output missing 'img_bbox'; refusing placeholder "
                "fallback so the integration error stays visible"
            )
        img_bbox = raw_output[0]['img_bbox']
        
        # Extract planning trajectory (required: a plan must exist).
        if 'final_planning' not in img_bbox:
            raise KeyError(
                "model output missing 'final_planning'; refusing "
                "placeholder fallback so the integration error stays visible"
            )
        planning = self._to_numpy(img_bbox['final_planning'])
        if planning.shape != (6, 2):
            raise ValueError(
                f"Expected final_planning shape (6, 2), got {planning.shape}"
            )
        trajectory = planning.astype(np.float32)
        
        # Extract detection boxes (empty is legal: no detections).
        # The model emits DECODED 10-dim boxes
        # [x,y,z,w,l,h,yaw,vx,vy,vz] (see decode_box); the pipeline
        # contract keeps the 7-dim footprint [x,y,z,w,l,h,yaw] and the
        # planar velocity [vx,vy,vz] separately (P-18: velocities drive
        # constant-velocity agent trajectories for the checker).
        if 'boxes_3d' in img_bbox:
            boxes = self._to_numpy(img_bbox['boxes_3d']).astype(np.float32)
            if boxes.size and boxes.shape[1] != 10:
                raise ValueError(
                    f"Expected boxes_3d shape (N, 10), got {boxes.shape}"
                )
            if boxes.size:
                velocities = boxes[:, 7:10]
                boxes = boxes[:, :7]
            else:
                boxes = boxes.reshape(0, 7)
                velocities = np.zeros((0, 3), dtype=np.float32)
        else:
            boxes = np.zeros((0, 7), dtype=np.float32)
            velocities = np.zeros((0, 3), dtype=np.float32)
        
        # Extract scores (must match boxes when both present).
        if 'scores_3d' in img_bbox:
            scores = self._to_numpy(img_bbox['scores_3d']).astype(np.float32)
            if scores.shape[0] != boxes.shape[0]:
                raise ValueError(
                    f"scores_3d ({scores.shape[0]}) mismatches "
                    f"boxes_3d ({boxes.shape[0]})"
                )
        else:
            scores = np.zeros((boxes.shape[0],), dtype=np.float32)

        return {
            'trajectory': trajectory,
            'boxes': boxes,
            'scores': scores,
            'velocities': velocities,
        }

    def _generate_placeholder(self) -> Dict[str, np.ndarray]:
        """Generate placeholder outputs for testing.
        
        Simulates a vehicle driving at 2.5 m/s (9 km/h). Waypoints are
        2.5 m apart, interpreted at the canonical STEP_DT=1.0 s (see
        checker.STEP_DT). NOTE: the real SparseDrive model emits 0.5 s
        waypoints; when real inference lands, its output must be
        resampled to STEP_DT before check() (P-1).
        """
        np.random.seed(42)
        
        # 6 timesteps at 0.5s intervals, ~5 m/s constant velocity
        dt = self.EGO_FUT_INTERVAL
        cruise = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float32)

        # Cumulative x positions (forward)
        x_cum = np.cumsum(cruise * dt)
        # Slight lateral drift
        y_cum = np.cumsum(np.array([0.0, 0.1, 0.05, -0.05, 0.0, 0.05], dtype=np.float32))
        
        trajectory = np.column_stack([x_cum, y_cum]).astype(np.float32)
        
        # Placeholder boxes [x, y, z, w, l, h, yaw] (canonical layout:
        # w=width/y-extent, l=length/x-extent, h=height/vertical).
        boxes = np.array([
            [20.0, 0.0, 0.5, 2.0, 4.5, 1.5, 0.0],
            [40.0, 5.0, 0.5, 2.0, 4.5, 1.5, 0.0],
        ], dtype=np.float32)
        
        scores = np.array([0.95, 0.87], dtype=np.float32)

        # Placeholder detections are static (no velocity estimates).
        velocities = np.zeros((2, 3), dtype=np.float32)

        return {
            'trajectory': trajectory,
            'boxes': boxes,
            'scores': scores,
            'velocities': velocities,
        }
