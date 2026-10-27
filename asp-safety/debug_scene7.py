import sys
sys.path.insert(0, r'C:\Users\chand\OneDrive\Desktop\EDI\asp-safety\src')
import json, time
import numpy as np
from checker import TrajectoryChecker
from corrector import CorrectorQP
from pipeline import SafetyLayerPipeline

dump = json.loads(open(r'C:\Users\chand\OneDrive\Desktop\EDI\asp-safety\experiments\trajs.json').read())
checker = TrajectoryChecker(r'C:\Users\chand\OneDrive\Desktop\EDI\asp-safety\rules')

s = dump['7']
pred = np.asarray(s['pred_1s'], dtype=np.float32)
agents = SafetyLayerPipeline._detections_to_trajectories(
    np.asarray(s['boxes'], dtype=np.float32),
    np.asarray(s.get('velocities', np.zeros((0,3))), dtype=np.float32),
    np.asarray(s.get('scores', np.zeros((0,))), dtype=np.float32), len(pred))
kw = dict(map_data=s['map_data'], other_vehicles=agents or None)

from constraint_optimizer import ConstraintOptimizer
orig_solve = __import__('constraint_optimizer').ConstraintOptimizer.solve
def traced(self, traj, cons, params=None):
    w, info = __import__('constraint_optimizer').ConstraintOptimizer.solve(self, traj, cons, params)
    print(f'  solve: status={info["status"]} iters={info["iterations"]} joint_iters={info.get("joint_iters")}')
    print(f'  constraint counts: xb={len(cons.get("x_bounds",[]))} yb={len(cons.get("y_bounds",[]))} yr={len(cons.get("y_bounds_range",[]))} xr={len(cons.get("x_bounds_range",[]))}')
    return w, info

import constraint_optimizer as co
orig_solve = co.ConstraintOptimizer.solve
def traced(self, traj, cons, params=None):
    w, info = orig_solve(self, traj, cons, params)
    print('  solve: status=%s iters=%s joint_iters=%s' % (
        info['status'], info['iterations'], info.get('joint_iters')))
    print('  constraint counts: xb=%s yb=%s yr=%s xr=%s' % (
        len(cons.get('x_bounds',[])), len(cons.get('y_bounds',[])),
        len(cons.get('y_bounds_range',[])), len(cons.get('x_bounds_range',[]))))
    return w, info

import constraint_optimizer as co
co.ConstraintOptimizer.solve = traced

from checker import TrajectoryChecker
from corrector import CorrectorQP
from pipeline import SafetyLayerPipeline

dump = json.loads(open(r'C:\Users\chand\OneDrive\Desktop\EDI\asp-safety\experiments\trajs.json').read())
checker = TrajectoryChecker(r'C:\Users\chand\OneDrive\Desktop\EDI\asp-safety\rules')

s = dump['7']
pred = np.asarray(s['pred_1s'], dtype=np.float32)
agents = SafetyLayerPipeline._detections_to_trajectories(
    np.asarray(s['boxes'], dtype=np.float32),
    np.asarray(s.get('velocities', np.zeros((0,3))), dtype=np.float32),
    np.asarray(s.get('scores', np.zeros((0,))), dtype=np.float32), len(pred))
kw = dict(map_data=s['map_data'], other_vehicles=agents or None)

qp = CorrectorQP(checker, heuristic_fallback=False, max_rounds=3)
print('Scene 7 with tracing (3 rounds):')
try:
    fixed, entries = qp.correct(pred, **kw)
    print('FIXED')
except RuntimeError as e:
    print('FAILED:', str(e)[:80])
ok, v = checker.check(pred, **kw)
print('initial violations:', sorted({x['rule'] for x in v}))
print('initial bounds:', {k: len(vv) for k, vv in checker.check(pred, include_constraints=True, **kw)[2].items() if vv})