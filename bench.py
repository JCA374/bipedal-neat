"""Phase 0 benchmark: env steps/s with a real NEAT net, on 1..N workers, plus RSS per worker.

  python bench.py --workers 1 4 6 7 8
"""
import argparse
import multiprocessing as mp
import os
import random
import resource
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import neat

import common


def _work(args):
    n_steps, seed = args
    random.seed(seed)
    config = common.load_config()
    genome = neat.DefaultGenome(seed)
    genome.configure_new(config.genome_config)
    for _ in range(20):  # grow it a bit, like a mid-run genome
        genome.mutate(config.genome_config)
    net = common.make_net(genome, config)
    env = common.make_env()
    steps, ep, t0 = 0, 0, time.time()
    while steps < n_steps:
        # No early stop: random nets fall fast; this measures step + activate cost incl. resets.
        _, n, _, _ = common.run_episode(net, env, seed * 1000 + ep, early_stop=False)
        steps += n
        ep += 1
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return steps, time.time() - t0, rss_mb


def mem_avail_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) // 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 6, 7, 8])
    ap.add_argument("--steps", type=int, default=20000, help="steps per worker")
    args = ap.parse_args()
    print(f"MemAvailable before: {mem_avail_mb()} MB")
    for w in args.workers:
        ctx = mp.get_context("fork")
        t0 = time.time()
        with ctx.Pool(w) as pool:
            res = pool.map(_work, [(args.steps, i + 1) for i in range(w)])
            avail = mem_avail_mb()
        wall = time.time() - t0
        total = sum(r[0] for r in res)
        per_core = sum(r[0] / r[1] for r in res) / w
        print(f"workers={w}: total {total / wall:7.0f} steps/s | per worker {per_core:6.0f} steps/s "
              f"| RSS/worker {max(r[2] for r in res):4.0f} MB | MemAvailable {avail} MB")


if __name__ == "__main__":
    main()
