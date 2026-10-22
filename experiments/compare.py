#!/usr/bin/env python3
"""Paired comparison of two saved genomes on the same new terrains, with bootstrap CIs.

  python experiments/compare.py runs/baseline/champion_v1 runs/baseline --episodes 100 \
      --out runs/baseline/eval/compare_v1.json

Seeds 4e7+ (disjoint from training, hall of fame, evaluate.py and the fall analysis).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import common  # noqa: E402

COMPARE0 = 40_000_000
_G = {}


def _ep(task):
    which, seed = task
    key = f"net{which}"
    if key not in _G:
        _G[key] = common.make_net(_G[which][0], _G[which][1])
        _G.setdefault("env", common.make_env())
    r, _, _, fell = common.run_episode(_G[key], _G["env"], seed, early_stop=False)
    return which, seed, r, fell


def boot_ci(d, n=5000):
    m = np.random.default_rng(0).choice(d, (n, len(d))).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="baseline genome (run dir, best dir or genome.pkl)")
    ap.add_argument("b", help="challenger")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    for k, path in (("a", args.a), ("b", args.b)):
        g, c, meta = common.resolve_genome(path)
        _G[k] = (g, c, meta)
    seeds = [COMPARE0 + i for i in range(args.episodes)]
    with mp.get_context("fork").Pool(args.workers) as pool:
        res = pool.map(_ep, [(w, s) for w in ("a", "b") for s in seeds], chunksize=2)
    R = {w: np.array([r for ww, s, r, f in sorted(res, key=lambda x: x[1]) if ww == w]) for w in "ab"}
    F = {w: np.array([f for ww, s, r, f in sorted(res, key=lambda x: x[1]) if ww == w], float) for w in "ab"}
    d = R["b"] - R["a"]
    out = {"episodes": args.episodes, "seed0": COMPARE0,
           "a": {"path": args.a, "mean": float(R["a"].mean()), "falls": int(F["a"].sum()),
                 "generation": _G["a"][2].get("generation"), **common.topology(_G["a"][0], _G["a"][1])},
           "b": {"path": args.b, "mean": float(R["b"].mean()), "falls": int(F["b"].sum()),
                 "generation": _G["b"][2].get("generation"), **common.topology(_G["b"][0], _G["b"][1])},
           "diff": float(d.mean()), "diff_ci": boot_ci(d),
           "fall_diff": float((F["b"] - F["a"]).mean()), "fall_diff_ci": boot_ci(F["b"] - F["a"]),
           "b_wins_share": float((d > 0).mean())}
    print(json.dumps(out, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
