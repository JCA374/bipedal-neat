"""Shared pieces for the explainer scenes: Pillow drawing helpers (same look as
../lunarlander-neat), an episode player that re-simulates instead of storing frames
(1600 frames would be >1 GB per episode on a 7.6 GB laptop), and `Context`, which loads
the run and caches the expensive bits (traces, evaluations) keyed by the champion.
"""
import hashlib
import json
import os
import pickle
import random
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import common  # noqa: E402
import neat  # noqa: E402
from neat.graphs import feed_forward_layers  # noqa: E402
from timeline import Scene  # noqa: E402,F401

W, H, FPS = 1280, 720, 30
BG = (11, 16, 32)
PANEL = (22, 30, 54)
FG = (232, 236, 245)
MUTED = (140, 152, 180)
AMBER = (255, 184, 64)
BLUE = (96, 168, 255)
GREEN = (84, 212, 140)
RED = (240, 96, 96)
VIOLET = (190, 120, 255)
ORANGE = (235, 104, 52)

HOF_SEEDS = [10_000_000 + i for i in range(10)]
EVAL_SEED0 = 20_000_000
MILESTONES = ["first_positive", "100", "200", "300"]
INPUT_SHORT = ["hull angle", "ang. vel", "vel x", "vel y",
               "hip1 ang", "hip1 spd", "knee1 ang", "knee1 spd", "foot1",
               "hip2 ang", "hip2 spd", "knee2 ang", "knee2 spd", "foot2"] + [f"lidar {i}" for i in range(10)]
OUTPUT_SHORT = ["hip 1", "knee 1", "hip 2", "knee 2"]
# Prior results on the same laptop (../bipedal-gpt, ../bipedal-claude-ppo, ../bipedal-claude).
PRIOR = {"PPO": 271.6, "TD3": 308.0}

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
_fonts = {}


def font(size, bold=False, mono=False):
    key = (size, bold, mono)
    if key not in _fonts:
        name = "DejaVuSansMono" if mono else "DejaVuSans"
        if bold:
            name += "-Bold"
        _fonts[key] = ImageFont.truetype(os.path.join(FONT_DIR, name + ".ttf"), size)
    return _fonts[key]


def mix(c1, c2, a):
    a = max(0.0, min(1.0, a))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * a) for i in range(3))


def ease(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def fade(t, start, dur=0.4):
    return ease((t - start) / dur)


def blank():
    return Image.new("RGB", (W, H), BG)


def text(d, xy, s, size=24, fill=FG, a=1.0, bold=False, mono=False, anchor="la", bg=BG):
    if a <= 0:
        return
    d.text(xy, s, font=font(size, bold, mono), fill=mix(bg, fill, a), anchor=anchor)


def heading(d, title, t, sub=None):
    a = fade(t, 0)
    text(d, (60, 42), title, 38, FG, a, bold=True)
    d.rectangle([60, 92, 60 + int(120 * a), 96], fill=AMBER)
    if sub:
        text(d, (200, 84), sub, 18, MUTED, a)


def wrap(s, width):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + [cur] if cur else lines


def paragraph(d, xy, s, width=40, size=22, fill=FG, a=1.0, gap=1.45):
    x, y = xy
    for i, line in enumerate(wrap(s, width)):
        text(d, (x, y + i * size * gap), line, size, fill, a)


def paste_frame(img, frame, box, border=None, dim=0.0):
    x, y, w, h = box
    tile = Image.fromarray(frame).resize((w, h), Image.BILINEAR)
    if dim > 0:
        tile = Image.blend(tile, Image.new("RGB", (w, h), BG), dim)
    img.paste(tile, (x, y))
    if border:
        ImageDraw.Draw(img).rectangle([x - 3, y - 3, x + w + 2, y + h + 2], outline=border, width=3)


def cover_frame(img, frame, dim=0.0):
    """Fill the whole 16:9 canvas with a 3:2 game frame (scaled up, cropped top/bottom)."""
    tile = Image.fromarray(frame).resize((W, int(W * frame.shape[0] / frame.shape[1])), Image.BILINEAR)
    top = (tile.height - H) // 2 + 40   # keep a bit more ground than sky
    tile = tile.crop((0, top, W, top + H))
    if dim > 0:
        tile = Image.blend(tile, Image.new("RGB", (W, H), BG), dim)
    img.paste(tile, (0, 0))


def label(d, xy, s, size=17):
    """Readable label on top of a game frame: light text on a dark box."""
    f = font(size, True)
    x, y = xy
    bb = d.textbbox((x, y), s, font=f)
    d.rectangle([bb[0] - 6, bb[1] - 4, bb[2] + 6, bb[3] + 4], fill=(20, 24, 40))
    d.text((x, y), s, font=f, fill=FG)


def title_card(name, kicker, title, sub, dur=5.0):
    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        text(d, (W // 2, 250), kicker, 26, AMBER, fade(t, 0.1), bold=True, anchor="mm")
        text(d, (W // 2, 330), title, 56, FG, fade(t, 0.3, 0.6), bold=True, anchor="mm")
        u = fade(t, 0.5)
        d.rectangle([W // 2 - int(80 * u), 380, W // 2 + int(80 * u), 384], fill=AMBER)
        text(d, (W // 2, 430), sub, 24, MUTED, fade(t, 0.7, 0.6), anchor="mm")
        return img
    return Scene(name, render, [dur])


# ----------------------------------------------------------------------------
# Episodes
# ----------------------------------------------------------------------------

class Player:
    """Frames of one deterministic episode, re-simulated on demand.

    frame(step) is cheap for increasing steps (the normal case while rendering a
    video) and restarts the episode when asked to go back (loops, stills)."""

    def __init__(self, genome, config, seed, max_steps=1600):
        self.genome, self.config, self.seed, self.max_steps = genome, config, seed, max_steps
        self.env = None
        self._restart()

    def _restart(self):
        if self.env is not None:
            self.env.close()
        self.env = common.make_env(render_mode="rgb_array")
        self.net = common.make_net(self.genome, self.config)
        self.obs, _ = self.env.reset(seed=int(self.seed))
        self.step, self.done, self.total = 0, False, 0.0
        self.cur = self.env.render()

    def frame(self, step):
        step = int(step)
        if step < self.step:
            self._restart()
        while self.step < step and not self.done and self.step < self.max_steps:
            a = self.net.activate(common.scale_obs(self.obs))
            self.obs, r, term, trunc, _ = self.env.step(np.asarray(a, dtype=np.float32))
            self.total += float(r)
            self.step += 1
            self.done = term or trunc
            if self.step >= step or self.done:
                self.cur = self.env.render()
        return self.cur

    def at_time(self, t, speed=1.0, loop_pause=1.5, length=None):
        """Frame for scene time t at `speed` x real time (50 steps/s), looping."""
        n = length or self.max_steps
        k = int(t * 50 * speed) % (n + int(loop_pause * 50 * speed))
        return self.frame(min(k, n))


def trace_episode(genome, config, seed, early_stop=False, max_steps=1600):
    """Per-step record: scaled obs, actions, all node values, reward, hull x/y/angle."""
    env = common.make_env()
    net = common.make_net(genome, config)
    obs, _ = env.reset(seed=int(seed))
    hull = env.unwrapped.hull
    rec = {"obs": [], "act": [], "vals": [], "r": [], "x": [], "y": [], "angle": []}
    total, fell, stalled = 0.0, False, False
    for step in range(max_steps):
        x = common.scale_obs(obs)
        a = net.activate(x)
        rec["obs"].append(x)
        rec["act"].append(np.array(a))
        rec["vals"].append(net.values.copy())
        obs, r, term, trunc, _ = env.step(np.asarray(a, dtype=np.float32))
        total += float(r)
        rec["r"].append(total)
        rec["x"].append(hull.position.x)
        rec["y"].append(hull.position.y)
        rec["angle"].append(hull.angle)
        if term:
            fell = r <= -100
            break
        if trunc:
            break
        if early_stop and len(rec["x"]) > common.STALL_WINDOW and \
                rec["x"][-1] - rec["x"][-1 - common.STALL_WINDOW] < common.STALL_MIN_DX:
            stalled = True
            break
    env.close()
    out = {k: np.array(v) for k, v in rec.items()}
    out.update(total=total, steps=len(rec["r"]), fell=fell, stalled=stalled, seed=seed,
               keys=_value_keys(genome, config))
    return out


def _value_keys(genome, config):
    """Node key for each index of FastNet.values (same ordering as FastNet)."""
    gc = config.genome_config
    conns = [cg.key for cg in genome.connections.values() if cg.enabled]
    layers = [sorted(l) for l in feed_forward_layers(gc.input_keys, gc.output_keys, conns)]
    keys = list(gc.input_keys) + [k for l in layers for k in l]
    keys += [k for k in gc.output_keys if k not in keys]
    return keys


def median_seed(genome, config, seeds=HOF_SEEDS):
    env = common.make_env()
    net = common.make_net(genome, config)
    rewards = [common.run_episode(net, env, s, early_stop=False)[0] for s in seeds]
    env.close()
    order = np.argsort(rewards)
    return seeds[order[len(order) // 2]], rewards


def net_layout(genome, config, box):
    """{node key: (x, y)}: inputs left, outputs right, hidden layers between (FF layers)."""
    gc = config.genome_config
    conns = [cg.key for cg in genome.connections.values() if cg.enabled]
    layers = [sorted(l) for l in feed_forward_layers(gc.input_keys, gc.output_keys, conns)]
    hidden = [[n for n in l if n not in gc.output_keys] for l in layers]
    hidden = [l for l in hidden if l]
    cols = [list(gc.input_keys)] + hidden + [list(gc.output_keys)]
    x0, y0, x1, y1 = box
    pos = {}
    for ci, col in enumerate(cols):
        x = x0 + ci * (x1 - x0) / (len(cols) - 1)
        for ri, k in enumerate(col):
            pos[k] = (x, y0 + (ri + 0.5) * (y1 - y0) / len(col))
    return pos, cols


def draw_net(d, genome, config, box, values=None, a=1.0, labels=True, hl=None, edge_a=0.55,
             node_r=(4, 8)):
    """Draw a network. values: {node key: activation} colours the nodes by sign/size."""
    pos, cols = net_layout(genome, config, box)
    gc = config.genome_config
    conns = [cg for cg in genome.connections.values() if cg.enabled and cg.key[0] in pos and cg.key[1] in pos]
    maxw = max([abs(c.weight) for c in conns] + [1.0])
    for c in sorted(conns, key=lambda c: abs(c.weight)):
        (xa, ya), (xb, yb) = pos[c.key[0]], pos[c.key[1]]
        col = BLUE if c.weight >= 0 else ORANGE
        ea = edge_a * a * (0.25 + 0.75 * abs(c.weight) / maxw)
        if hl is not None:
            ea = a * (0.9 if (c.key[0] in hl or c.key[1] in hl) else 0.05)
        wdt = max(1, int(round(0.6 + 2.6 * abs(c.weight) / maxw)))
        d.line([(xa, ya), (xb, yb)], fill=mix(BG, col, ea), width=wdt)
    for k, (x, y) in pos.items():
        isin, isout = k in gc.input_keys, k in gc.output_keys
        r = node_r[0] if isin else node_r[1]
        fill = PANEL
        if values is not None and k in values:
            v = float(np.clip(values[k], -1, 1))
            fill = mix(PANEL, BLUE if v >= 0 else ORANGE, abs(v))
        elif isout:
            fill = mix(PANEL, AMBER, 0.8)
        outline = mix(BG, FG if not isin else MUTED, a)
        d.ellipse([x - r, y - r, x + r, y + r], fill=mix(BG, fill, a), outline=outline, width=2)
        if labels and isin:
            text(d, (x - 10, y), INPUT_SHORT[-k - 1], 12, MUTED, a, anchor="rm")
        if labels and isout:
            text(d, (x + 14, y), OUTPUT_SHORT[k], 16, FG, a, anchor="lm")
    return pos, cols


def line_chart(d, box, series, xmax, ymin, ymax, t_frac=1.0, grid_step=50, a=1.0, xlabel="generation",
               marks=None):
    """series: [(xs, ys, colour, width, step?)]; t_frac reveals the lines left to right."""
    x0, y0, x1, y1 = box
    X = lambda v: x0 + v / max(1, xmax) * (x1 - x0)
    Y = lambda v: y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)
    v = int(np.ceil(ymin / grid_step) * grid_step)
    while v <= ymax:
        d.line([(x0, Y(v)), (x1, Y(v))], fill=mix(BG, MUTED, 0.45 if v == 0 else 0.15), width=1)
        text(d, (x0 - 8, Y(v)), str(v), 14, MUTED, a, anchor="rm")
        v += grid_step
    if (ymax - ymin) / grid_step > 12:
        pass
    xs_step = next(s for s in (10, 20, 50, 100, 200, 500, 1000) if xmax / s <= 10)
    for g in range(0, int(xmax) + 1, xs_step):
        text(d, (X(g), y1 + 16), str(g), 14, MUTED, a, anchor="mm")
    text(d, ((x0 + x1) / 2, y1 + 38), xlabel, 14, MUTED, a, anchor="mm")
    cut = t_frac * xmax
    for g, label in (marks or []):
        if g <= cut:
            d.line([(X(g), y0), (X(g), y1)], fill=mix(BG, MUTED, 0.5), width=1)
            text(d, (X(g) + 5, y0 + 4), label, 13, MUTED, a)
    for xs, ys, col, wdt, step in series:
        pts = []
        prev = None
        for xv, yv in zip(xs, ys):
            if yv is None or xv > cut:
                continue
            if step and prev is not None:
                pts.append((X(xv), Y(prev)))
            pts.append((X(xv), Y(yv)))
            prev = yv
        if len(pts) > 1:
            d.line(pts, fill=mix(BG, col, a), width=wdt, joint="curve")
    return X, Y


# ----------------------------------------------------------------------------
# Run + cache
# ----------------------------------------------------------------------------

class Context:
    def __init__(self, run_dir, cache_dir):
        self.dir = run_dir
        self.config = common.load_config(os.path.join(run_dir, "config.ini"))
        self.stats = _read_stats(os.path.join(run_dir, "stats.csv"))
        self.champ, self.champ_meta = common.load_genome(os.path.join(run_dir, "best", "genome.pkl"))
        self.milestones = {}
        for m in MILESTONES:
            p = os.path.join(run_dir, "milestones", m, "genome.pkl")
            if os.path.exists(p):
                self.milestones[m] = common.load_genome(p)
        self.G = self.stats[-1]["g"] + 1
        self.topo = common.topology(self.champ, self.config)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.key = hashlib.sha1(pickle.dumps(sorted((k, round(c.weight, 6), c.enabled) for k, c in
                                                    self.champ.connections.items()))).hexdigest()[:10]
        # Random starting genome, same recipe as make_site.py.
        random.seed(0)
        self.gen0 = neat.DefaultGenome(0)
        self.gen0.configure_new(self.config.genome_config)
        self.champ_seed, self.champ_rewards10 = self.cached("champ_seed", lambda: median_seed(self.champ, self.config))
        self.trace = self.cached("champ_trace", lambda: trace_episode(self.champ, self.config, self.champ_seed))
        self.eval100 = self.cached("eval100", self._eval100)
        self.stage_seeds = self.cached("stage_seeds", self._stage_seeds)
        self.fall_example = self.cached("fall_example", self._fall_example)

    def cached(self, name, fn):
        path = os.path.join(self.cache_dir, f"{name}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as fh:
                key, value = pickle.load(fh)
            if key == self.key:
                return value
        print(f"  computing {name} ...", file=sys.stderr)
        value = fn()
        with open(path, "wb") as fh:
            pickle.dump((self.key, value), fh)
        return value

    def _eval100(self):
        """Prefer the queue's evaluate.py output; otherwise compute 100 episodes here."""
        p = os.path.join(self.dir, "eval", "best.json")
        if os.path.exists(p):
            with open(p) as fh:
                d = json.load(fh)
            if d.get("genome_key") == self.champ_meta.get("genome_key") and "rewards" in d:
                return d
        env = common.make_env()
        net = common.make_net(self.champ, self.config)
        res = [common.run_episode(net, env, EVAL_SEED0 + i, early_stop=False) for i in range(100)]
        env.close()
        rewards = [r[0] for r in res]
        return {"rewards": rewards, "mean": float(np.mean(rewards)), "std": float(np.std(rewards)),
                "fell_frac": float(np.mean([r[3] for r in res])), "genome_key": self.champ_meta.get("genome_key")}

    def stages(self):
        """[(key, title, genome, meta)] gen 0, each milestone, and the champion if new."""
        out = [("gen0", "Random start", self.gen0, {"generation": 0, "mean": None})]
        seen = set()
        for m, (g, meta) in self.milestones.items():
            seen.add(meta["genome_key"])
            out.append((m, {"first_positive": "First positive"}.get(m, f"Over {m}"), g, meta))
        if self.champ_meta["genome_key"] not in seen:
            out.append(("best", "Champion", self.champ, self.champ_meta))
        return out

    def _stage_seeds(self):
        return {k: median_seed(g, self.config)[0] for k, _, g, _ in self.stages()}

    def _fall_example(self):
        """A random genome and terrain that falls within ~200 steps (for the reward scene)."""
        rng = random.Random(7)
        for i in range(200):
            random.seed(rng.randrange(10**6))
            g = neat.DefaultGenome(1)
            g.configure_new(self.config.genome_config)
            for _ in range(3):
                g.mutate(self.config.genome_config)
            tr = trace_episode(g, self.config, 1000 + i, max_steps=400)
            if tr["fell"] and 60 < tr["steps"] < 220:
                return {"genome": g, "seed": 1000 + i, "steps": tr["steps"], "total": tr["total"]}
        return None


def _read_stats(path):
    import csv
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({"g": int(r["generation"]), "best": float(r["best_fitness"]),
                         "mean": float(r["mean_fitness"]), "species": int(r["n_species"]),
                         "nodes": int(r["best_nodes"]), "conns": int(r["best_conns"]),
                         "hidden": int(r["best_hidden_in_use"]), "depth": int(r["best_depth"]),
                         "steps": int(r["env_steps_total"]), "saved": int(r["steps_saved_total"]),
                         "wall": float(r["wall_seconds"]),
                         "hof": float(r["hof_best_mean"]) if r["hof_best_mean"] else None,
                         "compat": float(r["compat_threshold"]) if r.get("compat_threshold") else None})
    return rows
