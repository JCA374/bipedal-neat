#!/usr/bin/env python3
"""Re-select the champion from a run's latest population with a stricter test.

The hall of fame picks the champion on 10 terrains, where a fall-prone but fast
walker can get lucky. This takes the top genomes of the checkpointed population
(by training fitness, hidden layer required), runs them on SELECT seeds, and
then compares the best few against the current champion, paired, on COMPARE
seeds with bootstrap confidence intervals.

  python experiments/reselect.py --run runs/baseline [--top 40] [--apply]

Seeds: select 5e7+, compare 4e7+ (both disjoint from training, HOF, evaluate.py
and the fall analysis).
"""
import argparse
import gzip
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import common  # noqa: E402
import train  # noqa: E402

SELECT0, COMPARE0 = 50_000_000, 40_000_000
_G = {}


def _ep(task):
    gi, seed = task
    if gi not in _G["nets"]:
        _G["nets"][gi] = common.make_net(_G["genomes"][gi], _G["config"])
    if "env" not in _G:
        _G["env"] = common.make_env()
    r, steps, _, fell = common.run_episode(_G["nets"][gi], _G["env"], seed, early_stop=False)
    return gi, seed, r, fell


def run_all(pool, gis, seeds):
    res = pool.map(_ep, [(g, s) for g in gis for s in seeds], chunksize=4)
    R = {g: np.zeros(len(seeds)) for g in gis}
    F = {g: np.zeros(len(seeds)) for g in gis}
    idx = {s: i for i, s in enumerate(seeds)}
    for g, s, r, fell in res:
        R[g][idx[s]] = r
        F[g][idx[s]] = fell
    return R, F


def boot_ci(d, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, (n, len(d))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/baseline")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--select-seeds", type=int, default=30)
    ap.add_argument("--finalists", type=int, default=4)
    ap.add_argument("--compare-seeds", type=int, default=100)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--apply", action="store_true", help="make the winner best/ if it beats the champion")
    args = ap.parse_args()
    run = args.run
    config = common.load_config(os.path.join(run, "config.ini"))
    with gzip.open(os.path.join(run, "checkpoints", "latest.pkl.gz"), "rb") as f:
        ck = pickle.load(f)
    pop = [g for g in ck["population"].values() if common.has_hidden_layer(g, config)]
    champ, cmeta = common.load_genome(os.path.join(run, "best", "genome.pkl"))
    # Unevaluated checkpoint populations have fitness None: rank those by structure-free order.
    pop.sort(key=lambda g: (g.fitness is not None, g.fitness or 0), reverse=True)
    cands = pop[:args.top]
    genomes = {"champion": champ, **{f"g{g.key}": g for g in cands}}
    _G.update(genomes=genomes, config=config, nets={})
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        sel = [SELECT0 + i for i in range(args.select_seeds)]
        R, F = run_all(pool, list(genomes), sel)
        ranked = sorted(genomes, key=lambda g: R[g].mean(), reverse=True)
        print(f"selection on {len(sel)} terrains ({time.time() - t0:.0f}s):")
        for g in ranked[:8]:
            print(f"  {g:>10}: mean {R[g].mean():7.1f}  falls {int(F[g].sum()):2d}/{len(sel)}")
        finalists = [g for g in ranked if g != "champion"][:args.finalists]
        cmp_seeds = [COMPARE0 + i for i in range(args.compare_seeds)]
        R2, F2 = run_all(pool, ["champion"] + finalists, cmp_seeds)
    out = {"select_seeds": len(sel), "compare_seeds": len(cmp_seeds),
           "champion": {"mean": float(R2["champion"].mean()), "falls": int(F2["champion"].sum()),
                        "genome_key": cmeta.get("genome_key"), "generation": cmeta.get("generation")},
           "finalists": []}
    print(f"\npaired comparison on {len(cmp_seeds)} new terrains:")
    print(f"  champion  : mean {R2['champion'].mean():7.1f}  falls {int(F2['champion'].sum())}")
    best = None
    for g in finalists:
        d = R2[g] - R2["champion"]
        lo, hi = boot_ci(d)
        df = F2[g] - F2["champion"]
        flo, fhi = boot_ci(df)
        rec = {"name": g, "mean": float(R2[g].mean()), "falls": int(F2[g].sum()), "diff": float(d.mean()),
               "diff_ci": [lo, hi], "fall_diff_ci": [flo, fhi], **common.topology(genomes[g], config)}
        out["finalists"].append(rec)
        print(f"  {g:>10}: mean {R2[g].mean():7.1f}  falls {int(F2[g].sum()):2d}  "
              f"diff {d.mean():+6.1f} [95% CI {lo:+.1f}, {hi:+.1f}]  hidden {rec['hidden_in_use']} depth {rec['depth']}")
        if lo > 0 and (best is None or rec["diff"] > best["diff"]):
            best = rec
    out["winner"] = best["name"] if best else None
    os.makedirs(os.path.join(run, "eval"), exist_ok=True)
    with open(os.path.join(run, "eval", "reselect.json"), "w") as f:
        json.dump(out, f, indent=2)
    if not best:
        print("\nNo finalist is better than the champion with 95% confidence; champion kept.")
        return
    print(f"\nWinner: {best['name']} ({best['diff']:+.1f} vs champion, CI {best['diff_ci'][0]:+.1f} to {best['diff_ci'][1]:+.1f})")
    if args.apply:
        g = genomes[best["name"]]
        # Score on the standard 10 hall-of-fame terrains too, so best/meta stays comparable.
        env, net = common.make_env(), common.make_net(g, config)
        hof = [common.run_episode(net, env, s, early_stop=False)[0] for s in train.held_out_seeds(10)]
        meta = {**cmeta, "genome_key": g.key, "generation": ck["generation"], "mean": float(np.mean(hof)),
                "std": float(np.std(hof)), **common.topology(g, config),
                "selected_by": f"reselect.py: {len(sel)} select + {len(cmp_seeds)} paired compare terrains",
                "replaced": {"genome_key": cmeta.get("genome_key"), "generation": cmeta.get("generation"),
                             "mean": cmeta.get("mean")}}
        train.save_best(os.path.join(run, "best"), g, meta, os.path.join(run, "config.ini"))
        print(f"best/ updated (10-terrain HOF mean {meta['mean']:.1f})")


if __name__ == "__main__":
    main()
