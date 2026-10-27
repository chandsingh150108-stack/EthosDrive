"""Data loader for nuScenes dataset.

This module provides data loading functionality for the nuScenes dataset.
If nuScenes is not available, it generates placeholder data for testing.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Camera order expected by the planner contract.
NUSC_CAM_ORDER = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]

# Planner input resolution (W, H).
PLANNER_W, PLANNER_H = 704, 256

# Real-map extraction budgets (perf vs completeness, forward item 3).
# Tile/crosswalk counts dominate real-scene grounding cost; the P-5
# verdict ledger lives in experiments/sweep_verdict.md.
MAP_QUERY_RADIUS = 40.0
MAP_LANE_TOKENS = 32
MAP_TILE_CELLS = 48
MAP_CROSSWALK_POLYS = 8
MAP_WALKWAY_POLYS = 8
MAP_CENTERLINE_RES = 2.0


def _pose_to_mat(translation, rotation) -> np.ndarray:
    """Compose a 4x4 transform from nuScenes translation + quaternion.

    nuScenes quaternions are (w, x, y, z).
    """
    from pyquaternion import Quaternion
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = Quaternion(rotation).rotation_matrix
    mat[:3, 3] = np.asarray(translation, dtype=np.float64)
    return mat


def estimate_ego_status(
    prev_ego2global: np.ndarray,
    ego2global: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Estimate the 18-dim ego dynamic state from consecutive ego poses.

    Layout follows the upstream converter
    (SparseDrive/tools/data_converter/nuscenes_converter.py::get_ego_status):
    [ax,ay,az, wx,wy,wz, vx,vy,vz, steer, 0*8]. Upstream falls back to
    all-zeros when CAN bus is absent (always, on mini); pose-derived
    planar velocity is strictly more informative and uses the same
    indices the checkpoint was trained on (status[6]=vx feeds the ego
    anchor in instance_queue.prepare_planning).

    Only planar velocity is observable from two poses; acceleration,
    angular rate, steering and padding stay zero.
    """
    status = np.zeros(18, dtype=np.float32)
    if dt <= 0:
        return status
    rel = np.linalg.inv(prev_ego2global.astype(np.float64)) @ \
        ego2global.astype(np.float64)
    status[6] = rel[0, 3] / dt  # vx: forward, m/s (ego frame x=forward)
    status[7] = rel[1, 3] / dt  # vy: left, m/s (ego frame y=left)
    return status


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
            # Real nuScenes scene count when devkit is installed;
            # falls back to 10 with a warning otherwise.
            try:
                from nuscenes.nuscenes import NuScenes
                self.nusc = NuScenes(
                    version=version, dataroot=str(data_dir))
                self.num_scenes = len(self.nusc.scene)
            except Exception as e:
                print(f"nuScenes devkit unavailable ({e}); "
                      f"using placeholder data.")
                self.use_placeholder = True
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
        """Load an actual nuScenes sample (MC-3).

        Uses the middle sample of the scene for variety. Images are read
        with cv2 (BGR->RGB), resized to the planner resolution, with
        intrinsics scaled to match. Extrinsics are lidar-to-camera
        4x4 transforms (cam_T_lidar), matching _build_projection_mat.
        The chosen sample token is cached so trajectory/map calls for
        the same scene stay consistent.
        """
        scene = self.nusc.scene[scene_idx]
        # Middle sample: deterministic, avoids scene-boundary effects.
        nbr = scene['nbr_samples']
        target = nbr // 2
        token = scene['first_sample_token']
        for _ in range(target):
            token = self.nusc.get('sample', token)['next']
        self._sample_tokens = getattr(self, '_sample_tokens', {})
        self._sample_tokens[scene_idx] = token
        return self._load_sample_by_token(token)

    def get_scene_frames(
        self, scene_idx: int, n_past: int = 4
    ) -> list:
        """Load consecutive frames ending at the scene's canonical sample.

        Walks BACKWARD from the middle sample (via sample['prev']) so
        history always exists, and returns oldest->newest sample dicts
        (same schema as get_scene_sample). Powers the N-2 warm-queue
        harness: feed all but the last frame to warm the temporal banks,
        evaluate on the last.

        Falls back to repeating the canonical sample if the scene start
        is reached (mini scenes are short).
        """
        if scene_idx < 0 or scene_idx >= self.num_scenes:
            raise IndexError(
                f"scene_idx {scene_idx} out of range [0, {self.num_scenes})"
            )
        if self.use_placeholder:
            base = self._generate_placeholder_sample(scene_idx)
            return [dict(base) for _ in range(n_past + 1)]

        # Ensure the canonical token is cached.
        self._load_nuscenes_sample(scene_idx)
        token = self._sample_tokens[scene_idx]
        tokens = [token]
        for _ in range(n_past):
            prev = self.nusc.get('sample', tokens[-1])['prev']
            tokens.append(prev if prev else tokens[-1])
        tokens.reverse()
        return [self._load_sample_by_token(t) for t in tokens]

    def _load_sample_by_token(self, token: str) -> Dict[str, np.ndarray]:
        """Load one sample dict by sample token (shared by canonical + frames)."""
        import cv2

        sample = self.nusc.get('sample', token)
        images, intrinsics, extrinsics = [], [], []
        ego2global = None
        timestamp = None

        # Lidar pose for cam_T_lidar composition.
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_sd = self.nusc.get('sample_data', lidar_token)
        lidar_cal = self.nusc.get(
            'calibrated_sensor', lidar_sd['calibrated_sensor_token'])
        lidar_ego = self.nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        ego_T_lidar = _pose_to_mat(
            lidar_cal['translation'], lidar_cal['rotation'])
        # Canonical ego frame + timestamp come from LIDAR_TOP.
        ego2global = _pose_to_mat(
            lidar_ego['translation'],
            lidar_ego['rotation']).astype(np.float32)
        timestamp = float(lidar_sd['timestamp']) / 1e6

        for cam in NUSC_CAM_ORDER:
            sd = self.nusc.get('sample_data', sample['data'][cam])
            img_path = str(self.data_dir / sd['filename'])
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"image missing: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h0, w0 = img.shape[:2]
            img = cv2.resize(img, (PLANNER_W, PLANNER_H),
                             interpolation=cv2.INTER_LINEAR)
            images.append(img.transpose(2, 0, 1))  # HWC -> CHW

            cal = self.nusc.get(
                'calibrated_sensor', sd['calibrated_sensor_token'])
            K = np.array(cal['camera_intrinsic'], dtype=np.float32)
            # Scale intrinsics to the resized resolution.
            K[0, :] *= PLANNER_W / w0
            K[1, :] *= PLANNER_H / h0
            intrinsics.append(K)

            ego_T_cam = _pose_to_mat(
                cal['translation'], cal['rotation'])
            cam_T_lidar = np.linalg.inv(ego_T_cam) @ ego_T_lidar
            extrinsics.append(cam_T_lidar.astype(np.float32))

        return {
            'images': np.stack(images).astype(np.uint8),
            'intrinsics': np.stack(intrinsics).astype(np.float32),
            'extrinsics': np.stack(extrinsics).astype(np.float32),
            'ego2global': ego2global,
            'timestamp': timestamp,
        }
    
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
            return self._load_nuscenes_trajectory(scene_idx)
    
    def _load_nuscenes_trajectory(self, scene_idx: int) -> np.ndarray:
        """Load ground-truth ego trajectory (MC-3).

        Walks the sample `next` chain from the cached scene sample,
        taking every 2nd pose (nuScenes is 2 Hz; STEP_DT=1.0 s per
        checker.STEP_DT). Poses are expressed in the first pose's
        local frame (x=forward, y=left). Returns up to 6 waypoints;
        fewer if the scene ends first.
        """
        tokens = getattr(self, '_sample_tokens', {})
        if scene_idx not in tokens:
            self.get_scene_sample(scene_idx)
            tokens = self._sample_tokens
        token = tokens[scene_idx]

        poses = []
        cur = token
        for _ in range(11):  # 1 anchor + 10 hops -> 6 waypoints at even hops
            if not cur:
                break
            sample = self.nusc.get('sample', cur)
            lidar_sd = self.nusc.get(
                'sample_data', sample['data']['LIDAR_TOP'])
            ego_pose = self.nusc.get('ego_pose', lidar_sd['ego_pose_token'])
            poses.append(_pose_to_mat(
                ego_pose['translation'], ego_pose['rotation']))
            cur = sample['next']
        if len(poses) < 2:
            raise RuntimeError(
                f"scene {scene_idx} has <2 samples from middle; "
                "cannot build a trajectory")
        # Every 2nd pose starting at the anchor (1 s spacing).
        picks = poses[0::2][:6]
        base_inv = np.linalg.inv(picks[0])
        local = [(base_inv @ p)[:2, 3] for p in picks]
        return np.array(local, dtype=np.float32)

    def get_tracked_agents(
        self, scene_idx: int = 0, max_agents: int = 20
    ) -> Dict[str, Dict]:
        """Extract tracked-agent trajectories from annotations (replay).

        Walks the anchor sample + next 10 samples (0..5 s at 2 Hz),
        groups `sample_annotation` records by `instance_token`, and
        interpolates each track to the integer-second grid T=0..5 in
        the anchor ego frame (same grid/frame as `get_trajectory`).

        Returns:
            Dict agent_id -> {'trajectory': (6, 2) float32,
                              'parked': bool (displacement < 0.5 m),
                              'category': str}
            Empty dict on placeholder data or when nothing tracks
            long enough (>= 2 distinct times spanning >= 3 s).

        Only `vehicle.*` and `human.*` categories are kept. Positions
        are annotation box centers (global) transformed into ego
        frame -- true tracked histories, unlike CV-projected single
        detections (see docs/replay_eval_design.md).
        """
        if self.use_placeholder:
            return {}
        tokens = getattr(self, '_sample_tokens', {})
        if scene_idx not in tokens:
            self.get_scene_sample(scene_idx)
            tokens = self._sample_tokens
        token = tokens[scene_idx]

        anchor_ts = None
        base_inv = None
        tracks: Dict[str, Dict] = {}
        cur = token
        for _ in range(11):
            if not cur:
                break
            sample = self.nusc.get('sample', cur)
            lidar_sd = self.nusc.get(
                'sample_data', sample['data']['LIDAR_TOP'])
            ego_pose = self.nusc.get('ego_pose', lidar_sd['ego_pose_token'])
            ts = float(lidar_sd['timestamp']) / 1e6
            if anchor_ts is None:
                anchor_ts = ts
                base_inv = np.linalg.inv(_pose_to_mat(
                    ego_pose['translation'], ego_pose['rotation']))
            t_rel = ts - anchor_ts
            for ann_tok in sample['anns']:
                try:
                    ann = self.nusc.get('sample_annotation', ann_tok)
                except KeyError:
                    continue
                cat = str(ann.get('category_name', ''))
                if not (cat.startswith('vehicle')
                        or cat.startswith('human')):
                    continue
                inst = ann['instance_token']
                tr = tracks.setdefault(inst, {'times': [], 'pts': [],
                                              'category': cat})
                tr['times'].append(t_rel)
                tr['pts'].append((float(ann['translation'][0]),
                                  float(ann['translation'][1])))
            cur = sample['next']

        out: Dict[str, Dict] = {}
        targets = np.arange(6, dtype=float)
        for inst, tr in tracks.items():
            times = np.array(tr['times'], dtype=float)
            order = np.argsort(times)
            times = times[order]
            pts = np.array(tr['pts'], dtype=float)[order]
            if len(times) < 2 or times[-1] - times[0] < 3.0:
                continue
            xs = np.interp(targets, times, pts[:, 0])
            ys = np.interp(targets, times, pts[:, 1])
            glob = np.column_stack(
                [xs, ys, np.zeros(6), np.ones(6)])
            local = (base_inv @ glob.T).T[:, :2].astype(np.float32)
            disp = float(np.linalg.norm(local[-1] - local[0]))
            out[inst[:8]] = {'trajectory': local,
                             'parked': disp < 0.5,
                             'category': tr['category']}
            if len(out) >= max_agents:
                break
        return out
    
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
            return self._generate_placeholder_map_data(scene_idx)
        else:
            return self._load_nuscenes_map_data(scene_idx)
    
    def _load_nuscenes_map_data(self, scene_idx: int) -> Dict:
        """Build map data from the real nuScenes map expansion (N-1).

        Queries NuScenesMap around the sample ego pose (40 m radius):
          lanes: centerline segments from arcline_path_3, widths from
              polygon-edge medians (3.5 m fallback), lane_direction
              from path direction, lane_type bicycle preserved (R9.3).
          intersections: is_intersection road_segment polygons
              (centroid + max-radius, capped at 30 m).
          crosswalks/sidewalks: ped_crossing/walkway polygon bboxes.
        Deliberately OMITTED (unknown, not inventable): road_type and
        speed_limit. R1.1/R1.2/R8.3/min-speed rules needing them stay
        silent; the default 50 km/h cap still applies. The old
        hardcoded urban/13.4 ego-path lane caused GT false positives
        (e.g. scene 7); unknown-classification silence is the honest
        no-knowledge behavior. Lane dividers/boundaries, signal phases
        and stop-line roles are not extracted (logged limits of N-1).
        """
        sample = self._load_nuscenes_sample(scene_idx)
        ego_x = float(sample['ego2global'][0, 3])
        ego_y = float(sample['ego2global'][1, 3])

        scene = self.nusc.scene[scene_idx]
        log = self.nusc.get('log', scene['log_token'])
        map_name = log['location']
        self._maps = getattr(self, '_maps', {})
        if map_name not in self._maps:
            from nuscenes.map_expansion.map_api import NuScenesMap
            self._maps[map_name] = NuScenesMap(
                dataroot=str(self.data_dir), map_name=map_name)
        nmap = self._maps[map_name]

        recs = nmap.get_records_in_radius(
            ego_x, ego_y, MAP_QUERY_RADIUS,
            ['lane', 'lane_connector', 'road_segment', 'ped_crossing',
             'walkway'])

        lanes, widths, directions, lane_types = {}, {}, {}, {}
        lane_tokens = (recs.get('lane', []) +
                       recs.get('lane_connector', []))[:MAP_LANE_TOKENS]
        try:
            centerlines = nmap.discretize_lanes(lane_tokens,
                                                MAP_CENTERLINE_RES)
        except (KeyError, TypeError, ValueError):
            centerlines = {}
        for tok in lane_tokens:
            try:
                pts = [(float(x), float(y))
                       for x, y, _ in centerlines.get(tok, [])]
                if len(pts) < 2:
                    continue
                try:
                    rec = nmap.get(
                        'lane', tok, )
                except (KeyError, TypeError):
                    rec = nmap.get('lane_connector', tok)
                lid = f"lane_{tok[:8]}"
                segs = []
                for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
                    if abs(x2 - x1) + abs(y2 - y1) < 1e-6:
                        continue
                    segs.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
                if not segs:
                    continue
                lanes[lid] = segs
                widths[lid] = self._lane_width_nmap(
                    nmap, rec.get('polygon_token', ''), pts)
                dx, dy = pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]
                norm = float(np.linalg.norm([dx, dy])) or 1.0
                directions[lid] = [dx / norm, dy / norm]
                if str(rec.get('lane_type', '')).upper() == 'BIKE':
                    lane_types[lid] = 'bicycle'
            except (KeyError, TypeError, ValueError, IndexError):
                continue

        intersections = {}
        for tok in recs.get('road_segment', []):
            try:
                rs = nmap.get('road_segment', tok)
                if not rs.get('is_intersection'):
                    continue
                poly = nmap.extract_polygon(rs['polygon_token'])
                xs, ys = poly.exterior.coords.xy
                cx, cy = float(np.mean(xs)), float(np.mean(ys))
                r = min(30.0, max(
                    float(np.hypot(x - cx, y - cy))
                    for x, y in zip(xs, ys)))
                if r < 1e-6:
                    continue
                intersections[f"int_{tok[:8]}"] = {
                    'x': cx, 'y': cy, 'radius': r}
            except (KeyError, TypeError, ValueError):
                continue

        # Nearest-first ordering + caps (perf): tile count dominates
        # real-scene grounding, and far polygons cannot affect a 6-step
        # horizon anyway.
        def _nearest(layer, key, limit):
            scored = []
            for tok in recs.get(layer, []):
                try:
                    rec = nmap.get(layer, tok)
                    d2 = self._poly_dist2(
                        nmap, rec[key], ego_x, ego_y)
                except (KeyError, TypeError, ValueError):
                    continue
                scored.append((d2, tok))
            scored.sort(key=lambda t: t[0])
            return [tok for _, tok in scored[:limit]]

        crosswalks = []
        for tok in _nearest('ped_crossing', 'polygon_token',
                            MAP_CROSSWALK_POLYS):
            try:
                pc = nmap.get('ped_crossing', tok)
                for x1, y1, x2, y2 in self._tiled_boxes(
                        nmap, pc['polygon_token']):
                    crosswalks.append({'id': f"cw_{tok[:8]}",
                                       'x1': x1, 'y1': y1,
                                       'x2': x2, 'y2': y2})
            except (KeyError, TypeError, ValueError):
                continue
        sidewalks = []
        for tok in _nearest('walkway', 'polygon_token',
                            MAP_WALKWAY_POLYS):
            try:
                w = nmap.get('walkway', tok)
                for x1, y1, x2, y2 in self._tiled_boxes(
                        nmap, w['polygon_token']):
                    sidewalks.append({'id': f"sw_{tok[:8]}",
                                      'x1': x1, 'y1': y1,
                                      'x2': x2, 'y2': y2})
            except (KeyError, TypeError, ValueError):
                continue

        out = {
            'lanes': lanes,
            'lane_width': widths,
            'lane_direction': directions,
            'intersections': intersections,
        }
        if lane_types:
            out['lane_type'] = lane_types
        if crosswalks:
            out['crosswalks'] = crosswalks
        if sidewalks:
            out['sidewalks'] = sidewalks
        return self._map_to_ego_frame(out, sample['ego2global'])

    @staticmethod
    def _map_to_ego_frame(map_data: Dict, ego2global: np.ndarray) -> Dict:
        """Express map geometry in the ego-local frame.

        Trajectories (pred + GT) live in ego frame; map layers are
        global. Without this transform every lane/intersection test
        compares across frames and misfires.
        """
        inv = np.linalg.inv(np.asarray(ego2global, dtype=np.float64))
        rot = inv[:3, :3]

        def _pt(x, y):
            p = inv @ np.array([x, y, 0.0, 1.0])
            return float(p[0]), float(p[1])

        def _vec(dx, dy):
            v = rot @ np.array([dx, dy, 0.0])
            return [float(v[0]), float(v[1])]

        ego = {'lanes': {}, 'lane_width': {}, 'lane_direction': {},
               'intersections': {}}
        for lid, segs in map_data.get('lanes', {}).items():
            out_segs = []
            for seg in segs:
                x1, y1 = _pt(seg['x1'], seg['y1'])
                x2, y2 = _pt(seg['x2'], seg['y2'])
                out_segs.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
            ego['lanes'][lid] = out_segs
        ego['lane_width'] = dict(map_data.get('lane_width', {}))
        for lid, d in map_data.get('lane_direction', {}).items():
            ego['lane_direction'][lid] = _vec(d[0], d[1])
        for iid, st in map_data.get('intersections', {}).items():
            x, y = _pt(st['x'], st['y'])
            ego['intersections'][iid] = {'x': x, 'y': y,
                                         'radius': st['radius']}
        for key in ('lane_type',):
            if key in map_data:
                ego[key] = dict(map_data[key])

        def _boxes(key):
            boxes = []
            for b in map_data.get(key, []):
                x1, y1 = _pt(b['x1'], b['y1'])
                x2, y2 = _pt(b['x2'], b['y2'])
                boxes.append({'id': b['id'],
                              'x1': min(x1, x2), 'y1': min(y1, y2),
                              'x2': max(x1, x2), 'y2': max(y1, y2)})
            return boxes

        # Small rotations keep tiles near-axis-aligned; re-boxing the
        # rotated corners is a bounded approximation (documented).
        if 'crosswalks' in map_data:
            ego['crosswalks'] = _boxes('crosswalks')
        if 'sidewalks' in map_data:
            ego['sidewalks'] = _boxes('sidewalks')
        return ego

    @staticmethod
    def _lane_width_nmap(nmap, polygon_token, pts) -> float:
        """Estimate lane width: 2x median centerline-to-edge distance."""
        try:
            poly = nmap.extract_polygon(polygon_token)
            ring = list(poly.exterior.coords)
            dists = []
            for px, py in pts:
                dists.append(min(
                    float(np.hypot(px - rx, py - ry)) for rx, ry in ring))
            w = 2.0 * float(np.median(dists)) if dists else 0.0
            if 1.5 <= w <= 8.0:
                return w
        except (KeyError, TypeError, ValueError):
            pass
        return 3.5

    @staticmethod
    def _tiled_boxes(nmap, polygon_token, cell=2.5,
                     max_cells=MAP_TILE_CELLS):
        """Tile a polygon into small boxes (centers inside the polygon).

        Axis-aligned bboxes of diagonal strips cover road area and
        cause false positives; 2.5 m tiles bound the over-approximation
        to one cell. Yields (x1, y1, x2, y2) tuples, at most max_cells
        (perf: tile count dominates real-scene grounding cost).
        """
        poly = nmap.extract_polygon(polygon_token)
        from shapely.geometry import Point
        minx, miny, maxx, maxy = poly.bounds
        boxes = []
        y = miny
        while y < maxy and len(boxes) < max_cells:
            x = minx
            while x < maxx and len(boxes) < max_cells:
                if poly.contains(Point(x + cell / 2.0, y + cell / 2.0)):
                    boxes.append((x, y, min(x + cell, maxx),
                                  min(y + cell, maxy)))
                x += cell
            y += cell
        return boxes

    @staticmethod
    def _poly_dist2(nmap, polygon_token, x, y) -> float:
        """Squared distance from (x, y) to a polygon centroid (or inf)."""
        try:
            poly = nmap.extract_polygon(polygon_token)
            cx, cy = poly.centroid.x, poly.centroid.y
            return (x - cx) ** 2 + (y - cy) ** 2
        except (KeyError, TypeError, ValueError, AttributeError):
            return float('inf')
    
    def _generate_placeholder_map_data(self, scene_idx: int = 0) -> Dict:
        """Generate placeholder map data.

        Scene 9 is the HOSTILE scene (T-9/P-14): a fixed obstacle sits
        on the planner path and the red intersection is reachable, so
        placeholder runs exercise the violation -> correction loop
        instead of only the safe path. Other scenes share the base
        two-lane geometry with a reachable red at (12, 0).

        Returns:
            Synthetic map data dictionary. Lanes are axis-aligned
            centerline segments with widths/directions/types so the
            lane_discipline (R2.x) and speed_regulation (R1.x) rules
            can resolve lane membership.
        """
        out = {
            'lanes': {
                'lane_0': [
                    {'x1': -10.0, 'y1': 0.0, 'x2': 100.0, 'y2': 0.0}
                ],
                'lane_1': [
                    {'x1': -10.0, 'y1': 4.0, 'x2': 100.0, 'y2': 4.0}
                ]
            },
            'lane_width': {
                'lane_0': 4.0,
                'lane_1': 4.0,
            },
            'lane_direction': {
                'lane_0': [1.0, 0.0],
                'lane_1': [1.0, 0.0],
            },
            'road_type': {
                'lane_0': 'urban',
                'lane_1': 'urban',
            },
            'speed_limit': {
                'lane_0': 13.4,
                'lane_1': 13.4,
            },
            'lane_boundary': [
                ['lane_0', 'lane_1', 'dashed'],
            ],
            'controls_approach': {
                'int_0': 'lane_0',
            },
            'intersections': {
                # Reachable on the placeholder path (ends ~15 m):
                # T3/T4 enter the box and must hold.
                'int_0': {'x': 12.0, 'y': 0.0, 'radius': 2.0}
            }
        }
        if scene_idx == 9:
            # Stalled vehicle straddling the path centerline.
            out['fixed_obstacles'] = [
                {'id': 'obs_9', 'x1': 8.0, 'y1': -1.0,
                 'x2': 12.0, 'y2': 1.0},
            ]
        return out
    
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
            return self._generate_placeholder_traffic_lights(scene_idx)
        else:
            # nuScenes provides no traffic-signal states. Return empty
            # (no knowledge) rather than inventing phases: signal rules
            # stay silent, which is the honest behavior.
            return {}
    
    def _generate_placeholder_traffic_lights(
        self, scene_idx: int = 0
    ) -> Dict:
        """Generate placeholder traffic light states.

        The red intersection matches the placeholder map (reachable at
        (12, 0), T3/T4 entry). Scene 9 keeps the same red on top of
        its obstacle.

        Returns:
            Synthetic traffic light dictionary
        """
        return {
            'int_0': {
                'color': 'red',
                # Span the horizon so entry (not just T0) is checked;
                # T0 starts outside the box (see map placeholder).
                'times': [0, 1, 2, 3, 4, 5],
                'x': 12.0,
                'y': 0.0,
                'radius': 2.0
            }
        }
