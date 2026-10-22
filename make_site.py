"""Build the GitHub Pages site in docs/ from a training run.

  python make_site.py --run runs/baseline

Renders one short clip per stage (random start, each milestone, current champion)
at 2x speed, H.264 so browsers play it, plus a concatenated progress.mp4 for the
later video. Embeds stats, hall of fame and the champion's network in index.html.
"""
import argparse
import csv
import json
import os
import random
import subprocess
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import neat
import numpy as np
from neat.graphs import feed_forward_layers

import common
from record_video import load_video_helpers

DOCS = common.ROOT / "docs"
CLIPS = DOCS / "clips"
HOF_SEEDS = [10_000_000 + i for i in range(10)]  # the hall-of-fame held-out terrains (train.held_out_seeds)
SPEEDUP = 2        # keep every 2nd frame at 50 fps -> 2x real time
HOLD_FRAMES = 40   # freeze the last frame ~0.8 s
MILESTONE_ORDER = ["first_positive", "100", "200", "300"]
MILESTONE_TITLE = {"first_positive": "First positive score", "100": "Milestone 100",
                   "200": "Milestone 200", "300": "Milestone 300 (solved)"}
INPUT_NAMES = ["hull angle", "hull ang. vel", "vel x", "vel y",
               "hip 1 angle", "hip 1 speed", "knee 1 angle", "knee 1 speed", "leg 1 contact",
               "hip 2 angle", "hip 2 speed", "knee 2 angle", "knee 2 speed", "leg 2 contact"] + \
              [f"lidar {i}" for i in range(10)]
OUTPUT_NAMES = ["hip 1", "knee 1", "hip 2", "knee 2"]


def median_seed(genome, config):
    """The held-out terrain whose reward is the median of the 10: a representative run,
    not a lucky or unlucky one."""
    net = common.make_net(genome, config)
    env = common.make_env()
    rewards = [common.run_episode(net, env, s, early_stop=False)[0] for s in HOF_SEEDS]
    order = np.argsort(rewards)
    i = order[len(order) // 2]
    return HOF_SEEDS[i], rewards


def render_clip(genome, config, out_mp4, label, helpers):
    seed, rewards = median_seed(genome, config)
    net = common.make_net(genome, config)
    env = common.make_env(render_mode="rgb_array")
    frames, scores = [], []

    def on_frame(e, total):
        frames.append(e.render())
        scores.append(total)

    total, steps, _, fell = common.run_episode(net, env, seed, early_stop=False, on_frame=on_frame)
    env.close()
    frames = frames[::SPEEDUP] + [frames[-1]] * HOLD_FRAMES
    scores = scores[::SPEEDUP] + [total] * HOLD_FRAMES
    h, w, _ = frames[0].shape
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", "50", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "26", "-preset", "slow", "-movflags", "+faststart", str(out_mp4)],
        stdin=subprocess.PIPE)
    for f, s in zip(frames, scores):
        proc.stdin.write(helpers.add_score(helpers.add_label(f, label), s).tobytes())
    proc.stdin.close()
    proc.wait()
    poster = out_mp4.with_suffix(".jpg")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-sseof", "-0.5", "-i", str(out_mp4),
                    "-frames:v", "1", "-q:v", "4", str(poster)], check=True)
    return {"reward_clip": round(total, 1), "terrain": seed - HOF_SEEDS[0] + 1,
            "rewards10": [round(r, 1) for r in rewards], "steps": steps, "fell": fell,
            "seconds": round(len(frames) / 50, 1)}


def network_data(genome, config):
    gc = config.genome_config
    conns = [cg for cg in genome.connections.values() if cg.enabled]
    layers = [sorted(l) for l in feed_forward_layers(gc.input_keys, gc.output_keys,
                                                     [c.key for c in conns])]
    used = set(gc.input_keys) | {n for l in layers for n in l}
    # Outputs always drawn in the last column; hidden layers in between.
    hidden_layers = [[n for n in l if n not in gc.output_keys] for l in layers]
    hidden_layers = [l for l in hidden_layers if l]
    cols = [list(gc.input_keys)] + hidden_layers + [list(gc.output_keys)]
    nodes = []
    for ci, col in enumerate(cols):
        for ri, k in enumerate(col):
            if k in gc.input_keys:
                name = INPUT_NAMES[-k - 1]
            elif k in gc.output_keys:
                name = OUTPUT_NAMES[k]
            else:
                name = f"hidden {k}"
            nodes.append({"id": k, "col": ci, "row": ri, "n": len(col), "name": name,
                          "bias": round(genome.nodes[k].bias, 3) if k in genome.nodes else None})
    edges = [{"a": c.key[0], "b": c.key[1], "w": round(c.weight, 3)} for c in conns
             if c.key[0] in used and c.key[1] in used]
    return {"nodes": nodes, "edges": edges, "cols": len(cols)}


def read_stats(run):
    rows = []
    with open(run / "stats.csv") as f:
        for r in csv.DictReader(f):
            rows.append({"g": int(r["generation"]), "best": float(r["best_fitness"]),
                         "mean": float(r["mean_fitness"]), "species": int(r["n_species"]),
                         "hidden": int(r["best_hidden_in_use"]), "depth": int(r["best_depth"]),
                         "steps": int(r["env_steps_total"]), "wall": float(r["wall_seconds"]),
                         "hof": float(r["hof_best_mean"]) if r["hof_best_mean"] else None})
    return rows


def _copy_analysis(run):
    """Publish fall analyses (failure-analysis-report method): the diagnosis of the champion
    before the fix (results/fall_analysis/champion_v1 -> docs/analysis-v1/) and the current
    champion (results/fall_analysis/<run> -> docs/analysis/). Returns the diagnosis summary."""
    import shutil
    out = None
    for name, dst_name in (("champion_v1", "analysis-v1"), (run.name, "analysis")):
        src = common.ROOT / "results" / "fall_analysis" / name
        if not (src / "summary.json").exists():
            continue
        dst = DOCS / dst_name
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            shutil.copy(f, dst / f.name)
        if out is None:
            out = (src, dst_name)
    if out is None:
        return None
    src, dst_name = out
    s = json.loads((src / "summary.json").read_text())
    feats = sorted(s["features"].values(), key=lambda v: -v["auc"])
    return {"dir": dst_name, "has_current": (DOCS / "analysis" / "report.html").exists() and dst_name != "analysis",
            "episodes": s["episodes"], "falls": s["falls"], "forward": s["fall_forward_share"],
            "onset": s["onset_to_fall_steps"], "mean_reward": s["mean_reward"],
            "terrain_before": s["terrain_before_fall"], "terrain_all": s["terrain_all"],
            "top": [{"label": v["label"], "auc": round(v["auc"], 2), "pc": round(v["partial_corr"], 2)} for v in feats[:5]],
            "max_added": round(max(v["partial_corr"] for v in s["features"].values()), 2)}


def _json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/baseline")
    ap.add_argument("--skip-clips", action="store_true")
    args = ap.parse_args()
    run = Path(args.run)
    CLIPS.mkdir(parents=True, exist_ok=True)
    helpers = load_video_helpers()
    config = common.load_config(run / "config.ini")

    stages = []
    random.seed(0)
    g0 = neat.DefaultGenome(0)
    g0.configure_new(config.genome_config)
    stages.append(("gen0", g0, {"generation": 0, "mean": None}, "Generation 0: random network",
                   "Random weights. 24 inputs, 8 hidden, 4 outputs."))
    seen = set()
    for m in MILESTONE_ORDER:
        d = run / "milestones" / m
        if (d / "genome.pkl").exists():
            g, meta = common.load_genome(d / "genome.pkl")
            seen.add(meta["genome_key"])
            stages.append((f"m_{m}", g, meta, MILESTONE_TITLE[m],
                           f"Generation {meta['generation']}. First champion over {m.replace('first_positive', '0')} on {len(meta.get('held_out_seeds') or [0] * 10)} unseen terrains."))
    g, meta = common.load_genome(run / "best" / "genome.pkl")
    if meta["genome_key"] not in seen:
        stages.append(("best", g, meta, "Current champion",
                       f"Generation {meta['generation']}. Best average on {len(meta.get('held_out_seeds') or [0] * 10)} unseen terrains so far."))

    clips = []
    for key, genome, meta, title, desc in stages:
        out = CLIPS / f"{key}.mp4"
        label = f"Gen {meta['generation']}" + (f"  avg {meta['mean']:+.0f}" if meta.get("mean") is not None else "")
        info = {"seconds": None}
        if not args.skip_clips or not out.exists():
            print(f"rendering {key} ...")
            info = render_clip(genome, config, out, label, helpers)
        topo = common.topology(genome, config)
        clips.append({"key": key, "title": title, "desc": desc, "generation": meta["generation"],
                      "mean": None if meta.get("mean") is None else round(meta["mean"], 1),
                      "std": None if meta.get("std") is None else round(meta["std"], 1),
                      **info, **topo})

    # One concatenated progress video for the later explainer.
    listing = CLIPS / "concat.txt"
    listing.write_text("".join(f"file '{c['key']}.mp4'\n" for c in clips))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(listing), "-c", "copy", str(CLIPS / "progress.mp4")], check=True)
    listing.unlink()

    stats = read_stats(run)
    champ, champ_meta = common.load_genome(run / "best" / "genome.pkl")
    milestones = {}
    for m in MILESTONE_ORDER:
        p = run / "milestones" / m / "meta.json"
        if p.exists():
            mm = json.loads(p.read_text())
            milestones[m] = {"generation": mm["generation"], "mean": round(mm["mean"], 1),
                             "wall_min": round(mm["wall_seconds"] / 60, 1),
                             "env_steps": mm["env_steps_total"]}
    data = {
        "built": time.strftime("%Y-%m-%d %H:%M"),
        "run": run.name,
        "stats": stats,
        "clips": clips,
        "milestones": milestones,
        "champion": {"generation": champ_meta["generation"], "mean": round(champ_meta["mean"], 1),
                     "std": round(champ_meta["std"], 1), **common.topology(champ, config)},
        "network": network_data(champ, config),
        "params": json.loads((run / "run_config.json").read_text())["effective"],
        "eval": _json(run / "eval" / "best.json"),
        "eval_milestones": {p.stem.replace("milestone_", ""): {k: v for k, v in _json(p).items() if k != "rewards"}
                            for p in sorted((run / "eval").glob("milestone_*.json"),
                                          key=lambda p: MILESTONE_ORDER.index(p.stem.replace("milestone_", ""))
                                          if p.stem.replace("milestone_", "") in MILESTONE_ORDER else 99)},
        "videos": (_json(DOCS / "video" / "videos.json") or {}).get("videos", []),
        "history": json.loads((run / "run_config.json").read_text()).get("history", []),
        "hof_terrains": len(champ_meta.get("held_out_seeds") or [0] * 10),
        "compare": _json(run / "eval" / "compare_v1.json"),
        "analysis": _copy_analysis(run),
    }
    if data["eval"] and data["eval"].get("genome_key") != champ_meta.get("genome_key"):
        data["eval"] = None  # stale: evaluated an older champion
    tpl = (common.ROOT / "site_template.html").read_text()
    (DOCS / "index.html").write_text(tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))))
    (DOCS / ".nojekyll").write_text("")
    print(f"site -> {DOCS / 'index.html'} ({len(clips)} clips)")


if __name__ == "__main__":
    main()
