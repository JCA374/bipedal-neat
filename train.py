"""NEAT evolution for BipedalWalker-v3.

  python train.py --name baseline --hours 2
  python train.py --name baseline --resume --hours 8

Fitness = mean raw reward over K seeds (shared by the whole generation, rotated
every generation), with early termination on stall. Genomes with no hidden
node in use are penalised and can never become the champion.
"""
import argparse
import csv
import gzip
import json
import multiprocessing as mp
import os
import pickle
import random
import shutil
import signal
import time
from itertools import count
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import neat

import common

MILESTONES = [("first_positive", 0.0), ("100", 100.0), ("200", 200.0), ("300", 300.0)]
COMPAT_MIN, COMPAT_MAX = 0.3, 3.0  # bounds for the adaptive compatibility_threshold
HELD_OUT_BASE = 10_000_000  # training seeds are < 1e6, held-out seeds start here
STATS_FIELDS = ["generation", "best_fitness", "mean_fitness", "n_species", "best_nodes",
                "best_conns", "best_hidden_in_use", "best_depth", "linear_frac",
                "env_steps_gen", "env_steps_total", "steps_saved_total", "gen_seconds",
                "wall_seconds", "hof_best_mean", "compat_threshold", "mem_avail_mb"]


# ---------------------------------------------------------------- workers
_W = {}


def _init_worker(config_path):
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # parent handles Ctrl-C
    _W["config"] = common.load_config(config_path)
    _W["env"] = common.make_env()


def linear_penalty(fitness):
    return fitness * 0.5 if fitness > 0 else fitness - 50.0


def eval_genome(task):
    genome, seeds, early_stop = task
    config, env = _W["config"], _W["env"]
    net = common.make_net(genome, config)
    rewards, steps, saved = [], 0, 0
    for s in seeds:
        r, n, sv, _ = common.run_episode(net, env, s, early_stop=early_stop)
        rewards.append(r)
        steps += n
        saved += sv
    hidden = common.topology(genome, config)["hidden_in_use"]
    return {"rewards": rewards, "steps": steps, "saved": saved, "hidden": hidden}


# ---------------------------------------------------------------- helpers
def mem_avail_mb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def gen_seeds(base_seed, generation, k):
    rng = np.random.default_rng(base_seed + generation)
    return [int(x) for x in rng.integers(0, 1_000_000, size=k)]


def held_out_seeds(n):
    return [HELD_OUT_BASE + i for i in range(n)]


def save_checkpoint(path, p, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    indexer, p.species.indexer = p.species.indexer, None  # itertools.count pickling is deprecated
    try:
        with gzip.open(tmp, "wb", compresslevel=5) as f:
            # Deliberately NOT storing the neat.Config: resume always uses the current .ini.
            pickle.dump({"generation": p.generation, "population": p.population,
                         "species": p.species, "random_state": random.getstate(),
                         "state": state}, f, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        p.species.indexer = indexer
    tmp.replace(path)
    print(f"  checkpoint -> {path}")


def unify_initial_hidden_keys(p, config):
    """neat-python 0.92 gives every new genome its own hidden-node ids (12..19, 20..27, ...),
    so with num_hidden > 0 all initial genomes look fully disjoint and each becomes its
    own species. Renumber every genome's initial hidden nodes to the same ids, then
    re-speciate."""
    gc = config.genome_config
    first = len(gc.output_keys)
    for g in p.population.values():
        hidden = sorted(k for k in g.nodes if k not in gc.output_keys)
        remap = {old: first + i for i, old in enumerate(hidden)}
        nodes = {}
        for old, ng in g.nodes.items():
            ng.key = remap.get(old, old)
            nodes[ng.key] = ng
        g.nodes = nodes
        conns = {}
        for (a, b), cg in g.connections.items():
            cg.key = (remap.get(a, a), remap.get(b, b))
            conns[cg.key] = cg
        g.connections = conns
    gc.node_indexer = count(first + gc.num_hidden)
    p.species = config.species_set_type(config.species_set_config, p.reporters)
    p.species.speciate(config, p.population, p.generation)


def restore_population(path, config):
    """Rebuild a Population from a checkpoint using the CURRENT config (lesson #2)."""
    with gzip.open(path, "rb") as f:
        d = pickle.load(f)
    random.setstate(d["random_state"])
    species = d["species"]
    p = neat.Population(config, (d["population"], species, d["generation"]))
    # Re-attach current config/reporters to restored objects that captured the old ones.
    species.species_set_config = config.species_set_config
    species.reporters = p.reporters
    species.indexer = count(max(species.species, default=0) + 1)
    p.reproduction.genome_indexer = count(max(p.population) + 1)
    max_node = max(max(g.nodes) for g in p.population.values())
    config.genome_config.node_indexer = count(max_node + 1)
    return p, d["state"]


def effective_params(config, args):
    gc, rc = config.genome_config, config.reproduction_config
    return {
        "pop_size": config.pop_size,
        "num_hidden": gc.num_hidden,
        "initial_connection": gc.initial_connection,
        "feed_forward": gc.feed_forward,
        "conn_add_prob": gc.conn_add_prob,
        "node_add_prob": gc.node_add_prob,
        "node_delete_prob": gc.node_delete_prob,
        "weight_mutate_rate": gc.weight_mutate_rate,
        "weight_mutate_power": gc.weight_mutate_power,
        "compatibility_threshold": config.species_set_config.compatibility_threshold,
        "max_stagnation": config.stagnation_config.max_stagnation,
        "elitism": rc.elitism,
        "survival_threshold": rc.survival_threshold,
        "k_seeds": args.k,
        "early_stop": not args.no_early_stop,
        "workers": args.workers,
    }


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default=str(common.DEFAULT_CONFIG))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--generations", type=int, default=None, help="stop at this total generation")
    ap.add_argument("--hours", type=float, default=None, help="wall-clock budget for this invocation")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--k", type=int, default=3, help="seeds per genome per generation")
    ap.add_argument("--pop-size", type=int, default=None, help="override pop_size from the ini")
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--hof-every", type=int, default=10)
    ap.add_argument("--hof-top", type=int, default=5)
    ap.add_argument("--hof-seeds", type=int, default=10)
    ap.add_argument("--target-species", type=int, default=10,
                    help="adapt compatibility_threshold toward this many species (0 = fixed ini value)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop-at", type=float, default=None,
                    help="stop once the held-out champion mean reaches this (e.g. 300)")
    ap.add_argument("--runs-dir", default=str(common.ROOT / "runs"))
    args = ap.parse_args(argv)

    run_dir = Path(args.runs_dir) / args.name
    # overrides.json in the run dir changes args for the next (resumed) chunk without
    # touching a running queue script, e.g. {"k": 5, "pop_size": 300, "reason": "..."}.
    ov_path = run_dir / "overrides.json"
    overrides = json.loads(ov_path.read_text()) if (args.resume and ov_path.exists()) else {}
    if overrides:
        if "k" in overrides:
            args.k = int(overrides["k"])
        if "pop_size" in overrides:
            args.pop_size = int(overrides["pop_size"])
        print(f"Overrides from {ov_path}: {overrides}")
    ckpt_dir = run_dir / "checkpoints"
    ckpt_path = ckpt_dir / "latest.pkl.gz"
    config = common.load_config(args.config)
    if args.pop_size:
        config.pop_size = args.pop_size

    if args.resume:
        if not ckpt_path.exists():
            raise SystemExit(f"--resume: no checkpoint at {ckpt_path}")
        p, state = restore_population(ckpt_path, config)
        if args.target_species and state.get("compat_threshold"):
            # The threshold is a controlled variable when adaptive; continue from where it was.
            config.species_set_config.compatibility_threshold = min(
                COMPAT_MAX, max(COMPAT_MIN, state["compat_threshold"]))
        print(f"Resumed {args.name} at generation {p.generation}")
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise SystemExit(f"{run_dir} exists; use --resume or a new --name")
        random.seed(args.seed)
        p = neat.Population(config)
        unify_initial_hidden_keys(p, config)
        state = {"compat_threshold": config.species_set_config.compatibility_threshold,
                 "env_steps_total": 0, "steps_saved_total": 0, "wall_seconds": 0.0,
                 "best_score": None, "milestones": {}, "hof": [], "seed": args.seed}
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.ini")

    params = effective_params(config, args)
    print("Effective params:", json.dumps(params))
    hist = state.setdefault("param_history", [])
    same = lambda h: {k: v for k, v in h.items() if k not in ("generation", "workers", "compatibility_threshold")} == \
        {k: v for k, v in params.items() if k not in ("workers", "compatibility_threshold")}
    if not hist or not same(hist[-1]):
        hist.append({"generation": p.generation, **params, **({"reason": overrides["reason"]} if overrides.get("reason") else {})})
    with open(run_dir / "run_config.json", "w") as f:
        json.dump({"args": vars(args), "effective": params, "history": state["param_history"]}, f, indent=2)

    stats_path = run_dir / "stats.csv"
    new_stats = not stats_path.exists()
    stats_f = open(stats_path, "a", newline="")
    stats_w = csv.DictWriter(stats_f, fieldnames=STATS_FIELDS)
    if new_stats:
        stats_w.writeheader()

    pool = mp.get_context("fork").Pool(args.workers, initializer=_init_worker,
                                       initargs=(args.config,))
    t_start = time.time()
    wall0 = state["wall_seconds"]

    def wall():
        return wall0 + time.time() - t_start

    def hall_of_fame(genomes, generation):
        """Re-evaluate top genomes WITH a hidden layer on held-out seeds, no early stop."""
        cands = [g for g in genomes if common.has_hidden_layer(g, config)]
        cands.sort(key=lambda g: g.fitness, reverse=True)
        cands = cands[:args.hof_top]
        if not cands:
            print("  HOF: no candidate with a hidden layer")
            return None
        seeds = held_out_seeds(args.hof_seeds)
        # One task per (genome, seed): full 1600-step episodes, so balance them across workers.
        flat = pool.map(eval_genome, [(g, [s], False) for g in cands for s in seeds], chunksize=1)
        results = []
        for i in range(len(cands)):
            chunk = flat[i * len(seeds):(i + 1) * len(seeds)]
            results.append({"rewards": [c["rewards"][0] for c in chunk],
                            "steps": sum(c["steps"] for c in chunk)})
        best_mean = None
        for g, r in zip(cands, results):
            state["env_steps_total"] += r["steps"]
            mean, std = float(np.mean(r["rewards"])), float(np.std(r["rewards"]))
            topo = common.topology(g, config)
            entry = {"generation": generation, "genome_key": g.key, "mean": mean, "std": std,
                     "train_fitness": g.fitness, **topo}
            state["hof"].append(entry)
            best_mean = mean if best_mean is None else max(best_mean, mean)
            if state["best_score"] is None or mean > state["best_score"]:
                state["best_score"] = mean
                meta = {**entry, "env_steps_total": state["env_steps_total"],
                        "steps_saved_total": state["steps_saved_total"],
                        "wall_seconds": wall(), "held_out_seeds": seeds,
                        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                save_best(run_dir / "best", g, meta, args.config)
                print(f"  HOF: new champion mean={mean:.1f} ±{std:.1f} "
                      f"(hidden={topo['hidden_in_use']}, depth={topo['depth']})")
                for name, thr in MILESTONES:
                    if mean > thr and name not in state["milestones"]:
                        state["milestones"][name] = meta
                        save_best(run_dir / "milestones" / name, g, meta, args.config)
                        print(f"  MILESTONE {name} reached at generation {generation}")
        state["hof"] = sorted(state["hof"], key=lambda e: e["mean"], reverse=True)[:20]
        with open(run_dir / "hof.json", "w") as f:
            json.dump(state["hof"], f, indent=2)
        return best_mean

    def fitness_fn(genomes, cfg):
        gen = p.generation
        t0 = time.time()
        seeds = gen_seeds(state["seed"], gen, args.k)
        results = pool.map(eval_genome, [(g, seeds, not args.no_early_stop) for _, g in genomes],
                           chunksize=max(1, len(genomes) // (args.workers * 4)))
        steps = saved = linear = 0
        for (_, g), r in zip(genomes, results):
            f = float(np.mean(r["rewards"]))
            if r["hidden"] == 0:
                f = linear_penalty(f)
                linear += 1
            g.fitness = f
            steps += r["steps"]
            saved += r["saved"]
        state["env_steps_total"] += steps
        state["steps_saved_total"] += saved
        hof_best = None
        if args.hof_every and (gen + 1) % args.hof_every == 0:
            hof_best = hall_of_fame([g for _, g in genomes], gen)
        best = max((g for _, g in genomes), key=lambda g: g.fitness)
        topo = common.topology(best, cfg)
        fits = [g.fitness for _, g in genomes]
        row = {"generation": gen, "best_fitness": round(best.fitness, 2),
               "mean_fitness": round(float(np.mean(fits)), 2), "n_species": len(p.species.species),
               "best_nodes": topo["nodes"], "best_conns": topo["enabled_conns"],
               "best_hidden_in_use": topo["hidden_in_use"], "best_depth": topo["depth"],
               "linear_frac": round(linear / len(genomes), 3), "env_steps_gen": steps,
               "env_steps_total": state["env_steps_total"],
               "steps_saved_total": state["steps_saved_total"],
               "gen_seconds": round(time.time() - t0, 1), "wall_seconds": round(wall(), 1),
               "hof_best_mean": "" if hof_best is None else round(hof_best, 2),
               "mem_avail_mb": mem_avail_mb()}
        row["compat_threshold"] = round(cfg.species_set_config.compatibility_threshold, 4)
        if args.target_species:
            n = len(p.species.species)
            thr = cfg.species_set_config.compatibility_threshold
            if n < args.target_species:
                thr *= 0.95
            elif n > args.target_species:
                thr *= 1.05
            # Clamp: neat never merges existing species, so the species count only falls
            # through stagnation and an unclamped controller winds up (0.6 -> 38 in run 1).
            cfg.species_set_config.compatibility_threshold = min(COMPAT_MAX, max(COMPAT_MIN, thr))
            state["compat_threshold"] = cfg.species_set_config.compatibility_threshold
        stats_w.writerow(row)
        stats_f.flush()
        print(f"gen {gen:4d} best {best.fitness:7.1f} mean {row['mean_fitness']:7.1f} "
              f"species {row['n_species']:2d} hidden {topo['hidden_in_use']:2d} "
              f"steps {steps/1e3:6.0f}k saved {saved/1e3:6.0f}k {row['gen_seconds']:5.1f}s "
              f"mem {row['mem_avail_mb']}MB")

    def checkpoint():
        state["wall_seconds"] = wall()
        save_checkpoint(ckpt_path, p, state)

    try:
        while True:
            if args.generations is not None and p.generation >= args.generations:
                break
            if args.hours is not None and time.time() - t_start >= args.hours * 3600:
                print("Time budget reached")
                break
            if args.stop_at is not None and state["best_score"] is not None and state["best_score"] >= args.stop_at:
                print(f"Champion reached {state['best_score']:.1f} >= --stop-at {args.stop_at}")
                break
            p.run(fitness_fn, 1)
            if p.generation % args.checkpoint_every == 0:
                checkpoint()
    except KeyboardInterrupt:
        print("\nInterrupted, saving checkpoint")
    finally:
        checkpoint()
        pool.terminate()
        stats_f.close()
    print(f"Done at generation {p.generation}. Best held-out mean: {state['best_score']}")
    return p, state


def save_best(dirpath, genome, meta, config_path):
    dirpath = Path(dirpath)
    common.save_genome(dirpath / "genome.pkl", genome, meta)
    shutil.copy(config_path, dirpath / "config.ini")
    with open(dirpath / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
