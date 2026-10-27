"""CLI driver for the CARLA closed-loop harness (P25-B/C/D).

One invocation = one closed-loop run -> one JSON artifact under
experiments/runs/{scenario}_{seed}_{engine}.json with a
run-id timestamp dedupe suffix.

    python experiments/run_closed_loop.py \
        --scenario S1 --seed 1 --engine off \
        [--replan-ticks 40 --max-minutes 6 --max-distance 600 \
         --device cuda]

The frozen pilot matrix (nextstep.md 7.2) is driven by callers doing
5 scenarios x 3 engines x 1 seed; per-scenario triples run
back-to-back for the deterministic sim common across the triple.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from carla_bridge import BridgeConfig, CarlaBridge  # noqa: E402
from carla_context import (  # noqa: E402,F401 (kept so --smoke-context
    build_agent_context)     # uses the full builders path)


def cli_main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="S1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--engine", default="off",
                    choices=["off", "heuristic", "qp"])
    ap.add_argument("--replan-ticks", type=int, default=40)
    ap.add_argument("--plan-engine", default="neural",
                    choices=["neural", "lane"])
    ap.add_argument("--max-minutes", type=float, default=6.0)
    ap.add_argument("--max-distance", type=float, default=600.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dbg", action="store_true",
                    help="emit per-~2s control diagnostics ([ctrl] lines)")
    ap.add_argument("--view", action="store_true",
                    help="spectator ego-follow CAMERA (visible server)")
    ap.add_argument("--log-dir", default=str(ROOT / "experiments" / "runs"))
    ap.add_argument("--out", default=None,
                    help="override output JSON path")
    args = ap.parse_args(argv)

    cfg = BridgeConfig(
        scenario=args.scenario, seed=args.seed, engine=args.engine,
        plan_engine=args.plan_engine, replan_ticks=args.replan_ticks,
        max_minutes=args.max_minutes,
        max_distance_m=args.max_distance, device=args.device,
        dbg_lines=args.dbg, view_ego=args.view)
    bridge = CarlaBridge(cfg)
    t0 = time.time()
    summary = bridge.run()

    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = (Path(args.out) if args.out else
                out_dir / f"{cfg.scenario}_{cfg.seed}_{cfg.engine}.json"
                ).with_name(
        (Path(args.out).stem if args.out
         else f"{cfg.scenario}_{cfg.seed}_{cfg.engine}") + f"_{stamp}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=1, default=lambda o: "unserializable")

    wall = time.time() - t0
    print(f"[run_closed] {cfg.scenario} seed={cfg.seed} "
          f"engine={cfg.engine}: status={summary['status']} "
          f"sim_km={summary['sim_km']} replans={summary['replans']} "
          f"collided={summary['collided']} give_up_rate="
          f"{summary['give_up_rate']} wall={wall:.0f}s "
          f"-> {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
