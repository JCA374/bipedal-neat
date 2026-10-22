#!/usr/bin/env python3
"""Failure analysis of a walker: why it falls and what comes before the fall.

Builds results/fall_analysis/<model>/report.html (+ PNGs and summary.json) from
record_episodes.py output, following the failure-analysis-report method:
gallery, where, drift, prediction (AUC + partial correlation beyond the 24 inputs),
causes, and onset.

  python experiments/analyze_falls.py results/episodes/baseline.jsonl.gz
"""
import argparse
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "video_tools"))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import common  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "font.family": "DejaVu Sans",
})

INPUT_LABELS = ["hull angle", "hull angular velocity", "velocity x", "velocity y",
                "hip 1 angle", "hip 1 speed", "knee 1 angle", "knee 1 speed", "leg 1 contact",
                "hip 2 angle", "hip 2 speed", "knee 2 angle", "knee 2 speed", "leg 2 contact"] + \
               [f"lidar {i}" for i in range(10)]
CANDIDATES = {
    "hull_height": "hull height above ground",
    "slope_under": "terrain slope under hull",
    "slope_ahead": "terrain slope ahead (0.5-3 m)",
    "airborne": "steps with both feet off the ground",
    "since_contact_change": "steps since a foot touched down or lifted",
    "act_sat": "saturated motors (|torque| > 0.98)",
    "act_jerk": "torque change since last step",
    "pitch_trend": "hull angle change over 10 steps",
    "vx_trend": "speed change over 20 steps",
    "knee_min": "most bent knee (min knee angle)",
    "leg_spread": "hip 1 - hip 2 angle",
    "progress_50": "distance walked in the last 50 steps",
    "lidar_min": "closest lidar reading",
}
SHORT = {"obs0": "hull angle", "obs1": "hull angular velocity", "obs2": "forward speed",
         "hull_height": "hull height", "pitch_trend": "pitch change (10 steps)", "act_sat": "saturated motors",
         "knee_min": "most bent knee", "since_contact_change": "steps since foot change"}
KEY_CURRENT = ["obs0", "obs1", "obs2", "obs3"]
DRIFT = ["obs0", "obs1", "obs2", "hull_height", "pitch_trend", "act_sat", "knee_min", "since_contact_change"]
WARNING, NORMAL_GAP, WINDOW = 50, 300, 300


def label(k):
    return INPUT_LABELS[int(k[3:])] if k.startswith("obs") else CANDIDATES.get(k, k)


def load(path):
    with gzip.open(path, "rt") as f:
        header = json.loads(f.readline())["header"]
        eps = [json.loads(l) for l in f]
    return header, eps


def features(trace, cols):
    T = np.asarray(trace, dtype=float)
    c = {n: T[:, i] for i, n in enumerate(cols)}
    n = len(T)
    F = {f"obs{i}": c[f"obs{i}"] for i in range(24)}
    F["hull_height"] = c["y"] - c["ground_y"]
    F["slope_under"], F["slope_ahead"] = c["slope_under"], c["slope_ahead"]
    both_off = (c["obs8"] < 0.5) & (c["obs13"] < 0.5)
    air = np.zeros(n)
    since = np.zeros(n)
    contacts = np.stack([c["obs8"] > 0.5, c["obs13"] > 0.5], 1)
    for t in range(1, n):
        air[t] = air[t - 1] + 1 if both_off[t] else 0
        since[t] = 0 if (contacts[t] != contacts[t - 1]).any() else since[t - 1] + 1
    F["airborne"], F["since_contact_change"] = air, since
    A = np.stack([c[f"act{i}"] for i in range(4)], 1)
    F["act_sat"] = (np.abs(A) > 0.98).sum(1).astype(float)
    F["act_jerk"] = np.r_[0, np.abs(np.diff(A, axis=0)).sum(1)]
    lag = lambda v, k: np.r_[np.full(k, v[0]), v[:-k]] if n > k else np.full(n, v[0])
    F["pitch_trend"] = c["angle"] - lag(c["angle"], 10)
    F["vx_trend"] = c["vx"] - lag(c["vx"], 20)
    F["knee_min"] = np.minimum(c["obs6"], c["obs11"])
    F["leg_spread"] = c["obs4"] - c["obs9"]
    F["progress_50"] = c["x"] - lag(c["x"], 50)
    F["lidar_min"] = T[:, 14:24].min(1)
    F["_x"], F["_angle"], F["_vx"] = c["x"], c["angle"], c["vx"]
    return F


def auc(scores, labels):
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = labels.astype(bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def residualize(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ beta


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d else 0.0


def save(fig, out_dir, name):
    fig.savefig(os.path.join(out_dir, name), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return name


def render_frames(run, seed, steps):
    """Game frames of one analysis episode at the given steps (re-simulated)."""
    from vcommon import Player
    genome, config, _ = common.resolve_genome(run)
    p = Player(genome, config, seed)
    return [p.frame(s).copy() for s in sorted(steps)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    header, eps = load(args.episodes)
    model, cols = header["model"], header["columns"]
    out_dir = args.out or os.path.join("results", "fall_analysis", model)
    os.makedirs(out_dir, exist_ok=True)
    feats = [features(e["trace"], cols) for e in eps]
    fell = np.array([e["fell"] for e in eps])
    rewards = np.array([e["reward"] for e in eps])
    keys = [f"obs{i}" for i in range(24)] + list(CANDIDATES)
    print(f"{len(eps)} episodes, {fell.sum()} falls")

    # --- snapshots every 10 steps (skip the start-up)
    S = {k: [] for k in keys}
    ttf, is_fall_ep, ep_idx = [], [], []
    for i, (e, F) in enumerate(zip(eps, feats)):
        n = e["steps"]
        for t in range(60, n, 10):
            for k in keys:
                S[k].append(F[k][t])
            ttf.append(n - t if e["fell"] else np.inf)
            is_fall_ep.append(e["fell"])
            ep_idx.append(i)
    S = {k: np.array(v) for k, v in S.items()}
    ttf = np.array(ttf)
    soon = ttf <= WARNING
    normal = ttf > NORMAL_GAP
    sections, summary = [], {"model": model, "episodes": len(eps), "falls": int(fell.sum()),
                             "mean_reward": float(rewards.mean()), "warning_steps": WARNING, "features": {}}

    # --- 1. gallery
    fall_eps = sorted([e for e in eps if e["fell"]], key=lambda e: e["reward"])
    ok_eps = sorted([e for e in eps if not e["fell"]], key=lambda e: e["reward"])
    pick = fall_eps[:: max(1, len(fall_eps) // 9)][:9] + [ok_eps[0], ok_eps[len(ok_eps) // 2], ok_eps[-1]]
    fig, axes = plt.subplots(3, 4, figsize=(13, 6.4))
    for ax, e in zip(axes.flat, pick):
        fr = render_frames(header["run"], e["seed"], [max(0, e["steps"] - 1)])[0]
        ax.imshow(fr)
        ax.set_axis_off()
        tag = "fell" if e["fell"] else "time up"
        ax.set_title(f"{tag} · step {e['steps']} · {e['reward']:+.0f}", fontsize=8, fontweight="normal",
                     color=SERIES[7] if e["fell"] else INK_2)
    for ax in list(axes.flat)[len(pick):]:
        ax.set_axis_off()
    sections.append(("What does failure look like?", save(fig, out_dir, "gallery.png"),
                     f"Final frame of 9 falls (sorted by score, worst first) and 3 episodes that ran out of time "
                     f"(worst, median, best). {fell.sum()} of {len(eps)} episodes end in a fall; every other episode "
                     f"ends at the 1600-step limit before the end of the course."))

    # --- 2. where
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    allx = np.concatenate([F["_x"] for F in feats])
    fx = np.array([e["final_x"] for e in eps if e["fell"]])
    bins = np.linspace(0, 90, 31)
    axes[0].hist(allx, bins, density=True, color=GRID, label="all steps")
    axes[0].hist(fx, bins, density=True, color=SERIES[7], alpha=0.85, label="falls")
    axes[0].set(title="Position on the course", xlabel="hull x (course is ~88 long)", ylabel="density")
    axes[0].legend(frameon=False)
    ang_n = np.concatenate([F["_angle"][60:max(61, e["steps"] - NORMAL_GAP)] if e["fell"] else F["_angle"][60:]
                            for e, F in zip(eps, feats)])
    vx_n = np.concatenate([F["_vx"][60:max(61, e["steps"] - NORMAL_GAP)] if e["fell"] else F["_vx"][60:]
                           for e, F in zip(eps, feats)])
    ang_f = np.concatenate([F["_angle"][-WARNING:] for e, F in zip(eps, feats) if e["fell"]])
    vx_f = np.concatenate([F["_vx"][-WARNING:] for e, F in zip(eps, feats) if e["fell"]])
    rng = [[-1.6, 0.6], [-1.5, 3.0]]
    for ax, (a, v, t) in zip(axes[1:], [(ang_n, vx_n, "Normal walking"), (ang_f, vx_f, f"Last {WARNING} steps before a fall")]):
        ax.hist2d(a, v, bins=40, range=rng, cmap=BLUES, density=True)
        ax.set(title=t, xlabel="hull angle (rad, negative = nose down)", ylabel="forward speed")
        ax.grid(False)
    sections.append(("Where does it happen?", save(fig, out_dir, "where.png"),
                     "Left: falls are spread along the course, not tied to one place. Middle and right: hull angle "
                     "against forward speed. Normal walking stays in a tight spot; before a fall the hull pitches "
                     "nose-down while still moving forward."))

    # --- 3. drift
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.6))
    fig.subplots_adjust(hspace=0.55, wspace=0.28)
    drift_onset = {}
    for ax, k in zip(axes.flat, DRIFT):
        M = np.full((fell.sum(), WINDOW), np.nan)
        for j, (e, F) in enumerate([(e, F) for e, F in zip(eps, feats) if e["fell"]]):
            v = F[k][-WINDOW:]
            M[j, WINDOW - len(v):] = v
        x = np.arange(-WINDOW + 1, 1)
        med, q1, q3 = np.nanmedian(M, 0), np.nanpercentile(M, 25, 0), np.nanpercentile(M, 75, 0)
        col = SERIES[0] if k.startswith("obs") else SERIES[1]
        ax.fill_between(x, q1, q3, color=col, alpha=0.18, lw=0)
        ax.plot(x, med, color=col)
        nm = np.median(S[k][normal])
        lo, hi = np.percentile(S[k][normal], [10, 90])
        ax.axhline(nm, color=INK_2, ls="--", lw=1)
        out = np.where((med > hi) | (med < lo))[0]
        # first step of the final run outside the normal 10-90% band
        onset = None
        if len(out):
            run_start = out[-1]
            while run_start - 1 in set(out):
                run_start -= 1
            onset = int(x[run_start])
        drift_onset[k] = onset
        ax.set_title(SHORT.get(k, label(k)), fontsize=10)
        ax.set_xlabel("steps before the fall")
    sections.append(("What drifts before failure?", save(fig, out_dir, "drift.png"),
                     f"Median (line) and interquartile band over the last {WINDOW} steps of every fall. Dashed: the "
                     "typical value during normal walking. Blue = inputs the network already sees, orange = "
                     "candidate signals it doesn't."))

    # --- 4. predict
    X_cur = np.column_stack([S[f"obs{i}"] for i in range(24)])
    auc_v, pc_v = {}, {}
    fe = np.isfinite(ttf) & (ttf <= WINDOW)
    y = residualize(np.log(ttf[fe]), X_cur[fe])
    for k in keys:
        a = auc(S[k], soon)
        auc_v[k] = (max(a, 1 - a), "high" if a >= 0.5 else "low")
        pc_v[k] = abs(corr(residualize(S[k][fe], X_cur[fe]), y)) if not k.startswith("obs") else 0.0
    order = sorted(keys, key=lambda k: auc_v[k][0], reverse=True)[:16]
    best_cur = max((k for k in keys if k.startswith("obs")), key=lambda k: auc_v[k][0])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ys = np.arange(len(order))[::-1]
    axes[0].barh(ys, [auc_v[k][0] for k in order], color=[SERIES[0] if k.startswith("obs") else SERIES[1] for k in order])
    axes[0].set_yticks(ys, [f"{label(k)} ({auc_v[k][1]} = danger)" for k in order], fontsize=8)
    axes[0].axvline(0.5, color=INK_2, lw=1)
    axes[0].axvline(auc_v[best_cur][0], color=SERIES[0], ls="--", lw=1)
    axes[0].set(xlim=(0.4, 1.0), title=f"AUC: does it warn of a fall within {WARNING} steps?", xlabel="AUC")
    for yy, k in zip(ys, order):
        axes[0].text(auc_v[k][0] + 0.005, yy, f"{auc_v[k][0]:.2f}", va="center", fontsize=8, color=INK_2)
    cand = sorted(CANDIDATES, key=lambda k: pc_v[k], reverse=True)
    yc = np.arange(len(cand))[::-1]
    axes[1].barh(yc, [pc_v[k] for k in cand], color=SERIES[1])
    axes[1].set_yticks(yc, [label(k) for k in cand], fontsize=8)
    for yy, k in zip(yc, cand):
        axes[1].text(pc_v[k] + 0.003, yy, f"{pc_v[k]:.2f}", va="center", fontsize=8, color=INK_2)
    axes[1].set(title="Added information beyond the 24 inputs", xlabel="|partial correlation| with log time-to-fall")
    sections.append(("What predicts failure?", save(fig, out_dir, "predict.png"),
                     f"Left: AUC of each signal alone for \"falls within {WARNING} steps\" on snapshots every 10 steps "
                     f"(the network has no value estimate, so the dashed line is the best single input, "
                     f"{label(best_cur)}). Right: correlation with time-to-fall after removing what the 24 inputs "
                     "already explain. A high bar here is information the network does not have."))

    # --- 5. causes
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    fall_ang = np.array([F["_angle"][-1] for e, F in zip(eps, feats) if e["fell"]])
    fwd = float((fall_ang < 0).mean())
    axes[0].bar(["nose down\n(forward)", "nose up\n(backward)"], [fwd, 1 - fwd], color=[SERIES[7], SERIES[0]])
    axes[0].set(title="Which way does it fall?", ylabel="share of falls", ylim=(0, 1.1))
    for i, v in enumerate([fwd, 1 - fwd]):
        axes[0].text(i, v + 0.03, f"{v:.0%}", ha="center", fontsize=9)
    sl_all = np.concatenate([F["slope_under"] for F in feats])
    sl_f = np.array([F["slope_under"][-WARNING:].mean() for e, F in zip(eps, feats) if e["fell"]])
    cats = lambda s: [np.mean(s < -0.05), np.mean(np.abs(s) <= 0.05), np.mean(s > 0.05)]
    w = 0.38
    xx = np.arange(3)
    axes[1].bar(xx - w / 2, cats(sl_all), w, color=GRID, label="all steps (chance)")
    axes[1].bar(xx + w / 2, cats(sl_f), w, color=SERIES[7], label="before falls")
    axes[1].set_xticks(xx, ["downhill", "flat", "uphill"])
    axes[1].set(title="Terrain in the last second", ylabel="share")
    axes[1].legend(frameon=False)
    def contact_state(F, idx):
        a, b = F["obs8"][idx] > 0.5, F["obs13"][idx] > 0.5
        return np.where(a & b, 2, np.where(a | b, 1, 0))
    cs_all = np.concatenate([contact_state(F, slice(60, None)) for F in feats])
    cs_f = np.concatenate([contact_state(F, slice(-WARNING, None)) for e, F in zip(eps, feats) if e["fell"]])
    sh = lambda c: [np.mean(c == 0), np.mean(c == 1), np.mean(c == 2)]
    axes[2].bar(xx - w / 2, sh(cs_all), w, color=GRID, label="all steps (chance)")
    axes[2].bar(xx + w / 2, sh(cs_f), w, color=SERIES[7], label="last second before falls")
    axes[2].set_xticks(xx, ["no foot down", "one foot", "both feet"])
    axes[2].set(title="Feet on the ground", ylabel="share of steps")
    axes[2].legend(frameon=False)
    sections.append(("What causes it?", save(fig, out_dir, "causes.png"),
                     "Left: the direction of the fall from the final hull angle. Middle: terrain slope under the hull "
                     "in the last second before falls, compared with all steps. Right: how many feet are on the "
                     "ground in the last second, compared with normal."))
    summary["fall_forward_share"] = fwd
    summary["terrain_before_fall"] = dict(zip(["downhill", "flat", "uphill"], map(float, cats(sl_f))))
    summary["terrain_all"] = dict(zip(["downhill", "flat", "uphill"], map(float, cats(sl_all))))
    summary["contacts_before_fall"] = dict(zip(["none", "one", "both"], map(float, sh(cs_f))))
    summary["contacts_all"] = dict(zip(["none", "one", "both"], map(float, sh(cs_all))))

    # --- 6. onset: last step where |hull angle| and |pitch trend| were within normal 75th pct
    a75 = np.percentile(np.abs(S["obs0"][normal]), 75)
    p75 = np.percentile(np.abs(S["pitch_trend"][normal]), 75)
    onsets = []
    for e, F in zip(eps, feats):
        if not e["fell"]:
            continue
        healthy = np.where((np.abs(F["obs0"]) <= a75) & (np.abs(F["pitch_trend"]) <= p75))[0]
        onsets.append((e, int(healthy[-1]) if len(healthy) else 0))
    dist = np.array([e["steps"] - o for e, o in onsets])
    ex = sorted(onsets, key=lambda eo: eo[0]["steps"] - eo[1])
    ex = [ex[len(ex) // 4], ex[len(ex) // 2], ex[3 * len(ex) // 4]]
    fig, axes = plt.subplots(3, 4, figsize=(14, 6.6), gridspec_kw={"width_ratios": [1, 1, 1, 1]})
    for r, (e, o) in enumerate(ex):
        d = e["steps"] - 1 - o
        steps = [o, o + d // 3, o + 2 * d // 3, e["steps"] - 1]
        frs = render_frames(header["run"], e["seed"], steps)
        for c, (fr, s) in enumerate(zip(frs, steps)):
            ax = axes[r, c]
            ax.imshow(fr)
            ax.set_axis_off()
            name = "last healthy" if c == 0 else "the fall" if c == 3 else f"+{s - o} steps"
            ax.set_title(f"{name} (step {s})", fontsize=8, fontweight="normal", color=INK_2)
    sections.append(("How does it start?", save(fig, out_dir, "onset.png"),
                     f"Three falls (short, median and long run-up): the last healthy step (hull angle and pitch rate "
                     f"both within the normal 75th percentile), two steps in between, and the fall. Median distance "
                     f"from onset to fall: {np.median(dist):.0f} steps ({np.median(dist) / 50:.1f} s); middle half "
                     f"{np.percentile(dist, 25):.0f}-{np.percentile(dist, 75):.0f} steps."))
    summary["onset_to_fall_steps"] = {"median": float(np.median(dist)), "q25": float(np.percentile(dist, 25)),
                                      "q75": float(np.percentile(dist, 75))}

    # --- summary table
    rows = ""
    table_keys = KEY_CURRENT + sorted(CANDIDATES, key=lambda k: auc_v[k][0], reverse=True)
    for k in table_keys:
        mid = float(np.median(S[k][normal]))
        at = float(np.median([F[k][-1] for e, F in zip(eps, feats) if e["fell"]]))
        onset = drift_onset.get(k)
        rows += (f"<tr><td>{label(k)}{' <span class=cur>input</span>' if k.startswith('obs') else ''}</td>"
                 f"<td>{mid:.3g}</td><td>{at:.3g}</td><td>{auc_v[k][0]:.2f} ({auc_v[k][1]})</td>"
                 f"<td>{'' if k.startswith('obs') else f'{pc_v[k]:.2f}'}</td>"
                 f"<td>{'' if onset is None else f'{onset} steps'}</td></tr>")
        summary["features"][k] = {"label": label(k), "median_normal": mid, "median_at_fall": at,
                                  "auc": auc_v[k][0], "danger": auc_v[k][1], "partial_corr": pc_v[k],
                                  "leaves_normal_at": onset}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    secs = "".join(f'<section><h2>{t}</h2><p>{d}</p><img src="{img}" alt="{t}"></section>' for t, img, d in sections)
    topo = header.get("topology", {})
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fall analysis: {model}</title>
<style>body{{font-family:system-ui,sans-serif;background:{SURFACE};color:{INK};max-width:1200px;margin:24px auto;padding:0 16px;line-height:1.5}}
h1{{margin-bottom:4px}} .sub{{color:{INK_2}}} img{{max-width:100%;border:1px solid {GRID};border-radius:6px}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;font-size:14px;margin:12px 0}} td,th{{padding:4px 10px;border-bottom:1px solid {GRID};text-align:right}}
td:first-child,th:first-child{{text-align:left}} .cur{{font-size:11px;color:#fff;background:{SERIES[0]};border-radius:4px;padding:1px 5px}}
section{{margin:28px 0}} a{{color:{SERIES[0]}}}</style></head><body>
<p class="sub"><a href="../index.html">← NEAT Bipedal Walker</a></p>
<h1>Fall analysis: {model}</h1>
<p class="sub">Champion of generation {header.get('generation')} ({topo.get('hidden_in_use')} hidden nodes, depth {topo.get('depth')}) on
{len(eps)} analysis terrains (seeds {header['seed0']}+, never used in training or testing): {fell.sum()} falls ({fell.mean():.0%}), mean
reward {rewards.mean():.1f}. Snapshots every 10 steps; "falls soon" = within {WARNING} steps (1 s); "normal" = more than {NORMAL_GAP} steps before a fall, or an episode without one.</p>
<div class="tw"><table><tr><th>signal</th><th>median, normal</th><th>median at fall</th><th>AUC (danger when)</th>
<th>added info (partial corr.)</th><th>leaves normal range at</th></tr>{rows}</table></div>
<p class="sub">"Leaves normal range" = the step before the fall where the median trajectory last leaves the 10-90% band of normal walking (drift signals only).</p>
{secs}</body></html>"""
    with open(os.path.join(out_dir, "report.html"), "w") as f:
        f.write(html)
    print(f"Wrote {out_dir}/report.html")


if __name__ == "__main__":
    main()
