"""CARLA closed-loop metrics aggregator (P25-C).

Parses run JSONs produced by run_closed_loop.py under
experiments/runs/ and emits the frozen metric set (nextstep.md 7.2)

    python experiments/carla_metrics.py [--runs "experiments/runs/*.json"]

Writes experiments/carla_metrics.json and prints a per-run table.

Metrics (mean over completed runs unless noted):
  collision_rate_pct   share of runs whose ego collided
  violations_per_mile  ASP violations / driven miles (mean per run)
  intervention_rate    replans where the shield materially changed the plan
  replan_wall_ms       per-replan wall time (planner+checker+corrector)
  frame_drop_rate      dropped sensor synchros / replans
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MILES_PER_KM = 0.621371


def _rows(run):
    return run.get("rows", [])


def _n_violations(run):
    return sum(len(r.get("violations", [])) for r in _rows(run))


def _n_corrected(run):
    n = 0
    for r in _rows(run):
        if run.get("engine", "off") == "off":
            break
        l2 = r.get("safe_vs_plan_l2")
        flags = r.get("fix_flags", [])
        touched = (l2 is not None and l2 > 0.0) or \
                  ("revert_to_plan" in flags)
        n += int(touched)
    return n


def _exec_ms_mean(run):
    rows = _rows(run)
    if not rows:
        return None
    vals = [r.get("exec_ms") for r in rows if r.get("exec_ms")]
    return sum(vals) / len(vals) if vals else None


def aggregate(paths):
    per_run = []
    for p in paths:
        try:
            run = json.load(open(p))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [skip] {p}: {e}")
            continue
        rows = _rows(run)
        n_rep = max(run.get("replans", 0), len(rows), 1)
        km = run.get("sim_km", 0.0)
        item = {
            "file": os.path.basename(p),
            "scenario": run.get("scenario"),
            "seed": run.get("seed"),
            "engine": run.get("engine"),
            "plan_engine": run.get("config", {}).get(
                "plan_engine") or run.get("plan_engine"),
            "status": run.get("status"),
            "collided": run.get("collided"),
            "sim_km": km,
            "violations": _n_violations(run),
            "violations_per_mile": (_n_violations(run)
                                    / max(km * MILES_PER_KM, 1e-6)),
            "replans": n_rep,
            "corrected_rows": _n_corrected(run),
            "intervention_rate": _n_corrected(run) / n_rep,
            "frame_drops": run.get("frame_drops", 0),
            "exec_ms_mean": _exec_ms_mean(run),
        }
        per_run.append(item)

    def _mean(key):
        vals = [it[key] for it in per_run if it[key] is not None]
        return sum(vals) / len(vals) if vals else None

    summary = per_run and {
        "n_runs": len(per_run),
        "collision_rate_pct": 100.0 * sum(
            1 for it in per_run if it["collided"]) / len(per_run),
        "violations_per_mile": _mean("violations_per_mile"),
        "intervention_rate": _mean("intervention_rate"),
        "replan_wall_ms_mean": _mean("exec_ms_mean"),
        "frame_drop_rate": _mean("frame_drops") / max(
            _mean("replans"), 1.0),
        "per_run": per_run,
    } or {}
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="experiments/runs/*.json",
                    help="glob of run JSON(s); paths relative to repo root")
    ap.add_argument("--out", default=str(ROOT / "experiments"
                                         / "carla_metrics.json"))
    args = ap.parse_args(argv)

    pattern = args.runs
    if not os.path.isabs(pattern) and not pattern.startswith("experiments"):
        pattern = str(ROOT / pattern)
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"no runs matched: {pattern}")
        return 1
    print(f"aggregating {len(paths)} run(s)")

    summary = aggregate(paths)
    if not summary:
        print("no valid runs parsed")
        return 1

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)

    print(f"\ncollision_rate_pcnt  {summary['collision_rate_pct']:.1f}")
    print(f"viol_per_mile        {summary['violations_per_mile']:.2f}")
    print(f"intervention_rate    {summary['intervention_rate']:.3f}")
    print(f"replan_wall_ms_mean  {summary['replan_wall_ms_mean']:.0f}")
    print(f"frame_drop_rate      {summary['frame_drop_rate']:.4f}")

    print("\n" + "  ".join(
        f"{h:>11}" for h in ["run", "scen", "eng", "plan",
                             "km", "viol", "intv", "ms", "col", "status"]))
    for it in summary["per_run"]:
        name = it["file"][:24]
        print(f"{name[:11]:>11}  {it['scenario'] or '-':>4} "
              f"{(it['engine'] or '-'):>9} {(it['plan_engine'] or '-'):>6} "
              f"{it['sim_km']:>6.2f} {it['violations']:>4} "
              f"{it['intervention_rate']:>6.3f} "
              f"{(it['exec_ms_mean'] or 0):>7.0f} "
              f"{str(it['collided']):>5} {it['status']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
