"""N-episode evaluation of a saved genome (no early stop, raw reward).

  python evaluate.py --run runs/baseline               # best/ champion, 100 episodes
  python evaluate.py --run runs/baseline/milestones/100 --episodes 100
"""
import argparse
import json
import multiprocessing as mp
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import common

EVAL_SEED_BASE = 20_000_000  # disjoint from training (<1e6) and hall-of-fame (1e7+) seeds


_G = {}


def _episode(seed):
    # The genome is loaded once in the parent and inherited via fork, so a training run
    # saving a new champion mid-evaluation can't mix two genomes into one result.
    if "env" not in _G:
        _G["env"] = common.make_env()
        _G["net"] = common.make_net(_G["genome"], _G["config"])
    r, steps, _, fell = common.run_episode(_G["net"], _G["env"], seed, early_stop=False)
    return r, steps, fell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<name>, a milestone dir, or a genome.pkl")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed-base", type=int, default=EVAL_SEED_BASE)
    ap.add_argument("--out", default=None, help="also write the result (with all rewards) as JSON")
    args = ap.parse_args()

    genome, config, meta = common.resolve_genome(args.run)
    _G.update(genome=genome, config=config)
    topo = common.topology(genome, config)
    seeds = [args.seed_base + i for i in range(args.episodes)]
    with mp.get_context("fork").Pool(args.workers) as pool:
        res = pool.map(_episode, seeds, chunksize=1)
    rewards = np.array([r[0] for r in res])
    out = {
        "run": args.run, "episodes": args.episodes,
        "mean": round(float(rewards.mean()), 2), "std": round(float(rewards.std()), 2),
        "min": round(float(rewards.min()), 2), "max": round(float(rewards.max()), 2),
        "solved_frac": round(float((rewards >= 300).mean()), 3),
        "fell_frac": round(float(np.mean([r[2] for r in res])), 3),
        "mean_steps": round(float(np.mean([r[1] for r in res])), 1),
        **topo,
        "has_hidden_layer": topo["hidden_in_use"] >= 1,
        "logged_hof_mean": meta.get("mean"),
    }
    print(json.dumps(out, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({**out, "genome_key": meta.get("genome_key"), "generation": meta.get("generation"),
                       "rewards": [round(float(r), 2) for r in rewards]}, f, indent=2)


if __name__ == "__main__":
    main()
