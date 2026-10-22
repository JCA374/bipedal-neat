#!/usr/bin/env python3
"""Run a genome on analysis terrains and log every step, for failure analysis.

Per episode it stores the full per-step trace: raw observation (24), action (4),
hull x / y / angle / angular velocity / vx / vy, terrain height under the hull and
the terrain slope under and ahead of it, and the running reward. Output is gzipped
JSON lines (a header line, then one episode per line) in results/episodes/.

Seeds start at 3e7: disjoint from training (<1e6), hall of fame (1e7+) and
evaluate.py (2e7+).

  python experiments/record_episodes.py --run runs/baseline --episodes 300
  python experiments/analyze_falls.py results/episodes/baseline.jsonl.gz
"""
import argparse
import gzip
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import common  # noqa: E402

ANALYSIS_SEED0 = 30_000_000
_G = {}


def terrain_at(u, x):
    tx, ty = np.asarray(u.terrain_x), np.asarray(u.terrain_y, dtype=float)
    return float(np.interp(x, tx, ty))


def slope(u, x0, x1):
    return (terrain_at(u, x1) - terrain_at(u, x0)) / (x1 - x0)


def run(seed):
    genome, config = _G["genome"], _G["config"]
    net = common.make_net(genome, config)
    env = common.make_env()
    obs, _ = env.reset(seed=int(seed))
    u = env.unwrapped
    rows, total, fell, steps = [], 0.0, False, 0
    for steps in range(1, 1601):
        a = net.activate(common.scale_obs(obs))
        h = u.hull
        x = float(h.position.x)
        rows.append([round(float(v), 4) for v in obs] + [round(float(v), 4) for v in a] + [
            round(x, 3), round(float(h.position.y), 3), round(float(h.angle), 4),
            round(float(h.angularVelocity), 4), round(float(h.linearVelocity.x), 4),
            round(float(h.linearVelocity.y), 4), round(terrain_at(u, x), 3),
            round(slope(u, x - 0.5, x + 0.5), 4), round(slope(u, x + 0.5, x + 3.0), 4), round(total, 2)])
        obs, r, term, trunc, _ = env.step(np.asarray(a, dtype=np.float32))
        total += float(r)
        if term:
            fell = bool(u.game_over or r <= -100)
            break
        if trunc:
            break
    env.close()
    return {"seed": seed, "reward": round(total, 2), "steps": steps, "fell": fell,
            "final_x": round(float(u.hull.position.x), 2), "trace": rows}


COLUMNS = [f"obs{i}" for i in range(24)] + [f"act{i}" for i in range(4)] + \
          ["x", "y", "angle", "ang_vel", "vx", "vy", "ground_y", "slope_under", "slope_ahead", "score"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/baseline", help="run dir, milestone dir or genome.pkl")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    genome, config, meta = common.resolve_genome(args.run)
    _G.update(genome=genome, config=config)
    name = args.name or os.path.basename(os.path.normpath(args.run))
    out = os.path.join("results", "episodes", f"{name}.jsonl.gz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    seeds = [ANALYSIS_SEED0 + i for i in range(args.episodes)]
    header = {"model": name, "run": args.run, "genome_key": meta.get("genome_key"),
              "generation": meta.get("generation"), "columns": COLUMNS, "seed0": ANALYSIS_SEED0,
              "topology": common.topology(genome, config)}
    t0 = time.time()
    with gzip.open(out, "wt") as f, mp.get_context("fork").Pool(args.workers) as pool:
        f.write(json.dumps({"header": header}) + "\n")
        for i, res in enumerate(pool.imap_unordered(run, seeds), 1):
            f.write(json.dumps(res, separators=(",", ":")) + "\n")
            if i % 20 == 0 or i == len(seeds):
                print(f"\r{i}/{len(seeds)} episodes ({time.time() - t0:.0f}s)", end="", flush=True)
    print(f"\nSaved {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
