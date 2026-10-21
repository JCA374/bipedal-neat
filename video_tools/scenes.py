"""Scenes of the NEAT BipedalWalker explainer. One factory per scene; each returns a
Scene whose beats match the `## <scene>[.n]` sections in narration (see narration.py).
"""
import numpy as np
from PIL import ImageDraw

import vcommon as C
from vcommon import (AMBER, BG, BLUE, FG, GREEN, MUTED, ORANGE, PANEL, RED, VIOLET, H, W,
                     Player, Scene, blank, draw_net, ease, fade, heading, line_chart, mix,
                     paragraph, paste_frame, text)


def _bars(d, box, labels, vals, a=1.0, color=BLUE, neg=ORANGE, size=14, label_w=120):
    """Horizontal bars for values in [-1, 1], centred."""
    x0, y0, x1, y1 = box
    n = len(labels)
    rh = (y1 - y0) / n
    mid = x0 + label_w + (x1 - x0 - label_w) / 2
    half = (x1 - x0 - label_w) / 2 - 4
    for i, (lab, v) in enumerate(zip(labels, vals)):
        y = y0 + i * rh + rh / 2
        text(d, (x0 + label_w - 10, y), lab, size, MUTED, a, anchor="rm")
        d.rectangle([x0 + label_w, y - rh * 0.32, x1, y + rh * 0.32], fill=mix(BG, PANEL, a))
        v = float(np.clip(v, -1, 1))
        if v >= 0:
            d.rectangle([mid, y - rh * 0.32, mid + v * half, y + rh * 0.32], fill=mix(BG, color, a))
        else:
            d.rectangle([mid + v * half, y - rh * 0.32, mid, y + rh * 0.32], fill=mix(BG, neg, a))
    d.line([(mid, y0), (mid, y1)], fill=mix(BG, MUTED, 0.5 * a))


def _sync_step(t, speed=1.0, n=1600, pause=1.5):
    k = int(t * 50 * speed) % (n + int(pause * 50 * speed))
    return min(k, n - 1)


# ----------------------------------------------------------------------------

def scene_title(ctx, name="title"):
    p = Player(ctx.champ, ctx.config, ctx.champ_seed)

    def render(t, b):
        img = blank()
        C.cover_frame(img, p.at_time(t + 4), dim=0.55)
        d = ImageDraw.Draw(img)
        text(d, (W // 2, 270), "NEAT learns to walk", 64, FG, fade(t, 0.3, 0.8), bold=True, anchor="mm")
        u = fade(t, 0.8)
        d.rectangle([W // 2 - int(110 * u), 320, W // 2 + int(110 * u), 325], fill=AMBER)
        text(d, (W // 2, 370), "Evolving the weights and the shape of a neural network", 26, FG,
             fade(t, 1.1, 0.6), anchor="mm")
        text(d, (W // 2, 410), "BipedalWalker · neat-python · one 4-core laptop", 20, MUTED, fade(t, 1.5, 0.6), anchor="mm")
        return img
    return Scene(name, render, [7.0])


def scene_task(ctx, name="task"):
    """Beats: the walker; four motors; the goal."""
    p = Player(ctx.champ, ctx.config, ctx.champ_seed)
    tr = ctx.trace

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "The task", t, "Gymnasium BipedalWalker-v3")
        k, lt, _ = b.local(t)
        box = (60, 125, 600, 400)
        step = _sync_step(t)
        paste_frame(img, p.frame(step), box, border=PANEL)
        d = ImageDraw.Draw(img)
        x = 720
        if k == 0:
            text(d, (x, 150), "A body and two legs", 30, AMBER, fade(t, b.at(0) + 0.3), bold=True)
            paragraph(d, (x, 200), "A box-shaped hull on two legs, each with a hip and a knee. "
                      "The terrain is bumpy and different in every episode.", 34, 21, FG, fade(t, b.at(0) + 0.6))
        elif k == 1:
            text(d, (x, 150), "Four motors", 30, AMBER, fade(t, b.at(1) + 0.3), bold=True)
            paragraph(d, (x, 200), "Every step, the walker sets a torque between -1 and +1 on each joint.",
                      34, 21, FG, fade(t, b.at(1) + 0.6))
            _bars(d, (x, 300, 1220, 440), C.OUTPUT_SHORT, tr["act"][min(step, tr["steps"] - 1)],
                  fade(t, b.at(1) + 1.0), label_w=80, size=16)
        else:
            text(d, (x, 150), "The goal", 30, AMBER, fade(t, b.at(2) + 0.3), bold=True)
            paragraph(d, (x, 200), "Walk to the far end without falling, within 1600 steps: "
                      "32 seconds of simulated time.", 34, 21, FG, fade(t, b.at(2) + 0.6))
        # progress bar along the course
        xpos = tr["x"][min(step, tr["steps"] - 1)]
        frac = np.clip((xpos - tr["x"][0]) / 88.0, 0, 1)
        d.rectangle([60, 550, 660, 562], fill=PANEL)
        d.rectangle([60, 550, 60 + frac * 600, 562], fill=GREEN)
        text(d, (60, 580), f"distance {frac * 100:3.0f}% of the course", 16, MUTED)
        text(d, (660, 580), f"step {step:4d} / 1600", 16, MUTED, anchor="ra")
        return img
    return Scene(name, render, [6.0, 6.0, 6.0])


def scene_senses(ctx, name="senses"):
    """Beats: body; legs; lidar."""
    p = Player(ctx.champ, ctx.config, ctx.champ_seed)
    tr = ctx.trace
    groups = [(0, 4, "Body: 4 numbers", "tilt, spin and speed of the hull"),
              (4, 14, "Legs: 10 numbers", "angle and speed of each joint, and whether each foot touches the ground"),
              (14, 24, "Lidar: 10 numbers", "ten rays measure the distance to the ground ahead")]

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "What the walker senses", t, "24 numbers, every step")
        k, _, _ = b.local(t)
        step = _sync_step(t, speed=0.6)
        paste_frame(img, p.frame(step), (60, 140, 540, 360), border=PANEL)
        d = ImageDraw.Draw(img)
        lo, hi, title, sub = groups[k]
        text(d, (700, 140), title, 28, AMBER, fade(t, b.at(k) + 0.2), bold=True)
        paragraph(d, (700, 182), sub, 44, 18, FG, fade(t, b.at(k) + 0.4))
        obs = tr["obs"][min(step, tr["steps"] - 1)]
        _bars(d, (700, 250, 1220, 250 + 30 * (hi - lo)), C.INPUT_SHORT[lo:hi], obs[lo:hi],
              fade(t, b.at(k) + 0.5), label_w=110)
        text(d, (60, 680), "Each number is scaled to the range -1 to +1 by fixed factors, "
             "so one saved file is all a trained walker needs.", 17, MUTED, fade(t, 1.0))
        return img
    return Scene(name, render, [6.0, 7.0, 6.0])


def scene_reward(ctx, name="reward"):
    """Beats: progress pays; torque costs; falling."""
    tr = ctx.trace
    fe = ctx.fall_example
    pf = Player(fe["genome"], ctx.config, fe["seed"]) if fe else None
    pc = Player(ctx.champ, ctx.config, ctx.champ_seed)

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "How it is scored", t, "the environment's own reward")
        k, lt, _ = b.local(t)
        if k < 2:
            step = _sync_step(t, speed=1.5)
            paste_frame(img, pc.frame(step), (60, 130, 460, 307), border=PANEL)
            d = ImageDraw.Draw(img)
            # cumulative reward chart
            n = tr["steps"]
            xs = np.arange(n)
            line_chart(d, (640, 150, 1220, 430), [(xs, list(tr["r"][:step + 1]) + [None] * (n - step - 1), BLUE, 3, False)],
                       n, -20, 150 if tr["total"] < 140 else 350, 1.0, 50, xlabel="step")
            text(d, (640, 120), f"running score: {tr['r'][min(step, n - 1)]:+.0f}", 20, FG)
            msg = ("Moving forward earns points: about 300 for walking the whole course."
                   if k == 0 else "Every bit of motor torque costs a little, so wasted effort lowers the score.")
            paragraph(d, (60, 480), msg, 90, 22, FG, fade(t, b.at(k) + 0.3))
        else:
            if pf:
                paste_frame(img, pf.at_time(t - b.at(2), speed=0.8, length=fe["steps"]), (60, 130, 460, 307), border=RED)
            d = ImageDraw.Draw(img)
            text(d, (600, 150), "Falling costs 100", 34, RED, fade(t, b.at(2) + 0.3), bold=True)
            paragraph(d, (600, 205), "If the hull touches the ground, the episode ends at once with -100.", 44, 22, FG,
                      fade(t, b.at(2) + 0.6))
            text(d, (600, 330), "Solved = an average of 300", 28, GREEN, fade(t, b.at(2) + 1.6), bold=True)
            paragraph(d, (600, 380), "over 100 random terrains.", 44, 22, FG, fade(t, b.at(2) + 1.9))
        return img
    return Scene(name, render, [6.0, 6.0, 7.0])


def scene_network(ctx, name="network"):
    """Beats: the network; live signals; outputs are torques."""
    p = Player(ctx.champ, ctx.config, ctx.champ_seed)
    tr = ctx.trace
    keys = tr["keys"]

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "The pilot is a neural network", t, f"the champion: {ctx.topo['hidden_in_use']} hidden nodes")
        k, _, _ = b.local(t)
        step = _sync_step(t, speed=0.5)
        vals = None
        if k >= 1:
            v = tr["vals"][min(step, tr["steps"] - 1)]
            vals = {key: v[i] for i, key in enumerate(keys)}
        draw_net(d, ctx.champ, ctx.config, (150, 130, 780, 640), values=vals, a=fade(t, 0.2, 0.8))
        if k >= 1:
            paste_frame(img, p.frame(step), (900, 140, 330, 220), border=PANEL)
            d = ImageDraw.Draw(img)
        msgs = ["Signals flow left to right: 24 sensor inputs, through hidden nodes, to 4 motors.",
                "Blue means positive, orange negative. Thicker lines are stronger connections.",
                "The four outputs on the right are the joint torques, sent to the motors every step."]
        paragraph(d, (900, 400 if k >= 1 else 200), msgs[k], 30, 20, FG, fade(t, b.at(k) + 0.3))
        return img
    return Scene(name, render, [7.0, 8.0, 6.0])


def scene_hidden(ctx, name="hidden"):
    """Beats: start 24-8-4; the guard."""
    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "At least one hidden layer", t, "a hard requirement for this project")
        k, _, _ = b.local(t)
        draw_net(d, ctx.gen0, ctx.config, (140, 130, 620, 640), a=fade(t, 0.2), edge_a=0.25)
        text(d, (380, 660), "every network starts as 24 → 8 → 4", 18, MUTED, fade(t, 0.5), anchor="mm")
        if k == 0:
            text(d, (700, 160), "No shortcuts", 30, AMBER, fade(t, b.at(0) + 0.3), bold=True)
            paragraph(d, (700, 210), "Inputs connect only to the 8 hidden nodes, and the hidden nodes to the motors. "
                      "Every signal has to pass through a hidden layer.", 40, 21, FG, fade(t, b.at(0) + 0.6))
        else:
            text(d, (700, 160), "A guard in the code", 30, AMBER, fade(t, b.at(1) + 0.3), bold=True)
            items = ["Count the hidden nodes on a real input → output path.",
                     "None? The network's fitness is cut, so evolution avoids it.",
                     "And it can never be saved as a champion."]
            for i, s in enumerate(items):
                a = fade(t, b.at(1) + 0.8 + 1.2 * i)
                d.ellipse([700, 222 + 70 * i, 716, 238 + 70 * i], fill=mix(BG, GREEN, a))
                paragraph(d, (730, 218 + 70 * i), s, 40, 20, FG, a)
        return img
    return Scene(name, render, [8.0, 9.0])


TOY_POS = {"i1": (0, 0.15), "i2": (0, 0.5), "i3": (0, 0.85), "h1": (0.5, 0.3), "h2": (0.5, 0.7),
           "h3": (0.75, 0.5), "o1": (1, 0.3), "o2": (1, 0.7)}


def _toy(d, box, edges, nodes, a=1.0, hl=()):
    x0, y0, x1, y1 = box
    P = lambda n: (x0 + TOY_POS[n][0] * (x1 - x0), y0 + TOY_POS[n][1] * (y1 - y0))
    for (u, v), w, on in edges:
        if u not in nodes or v not in nodes:
            continue
        col = BLUE if w >= 0 else ORANGE
        ea = a * (0.95 if (u, v) in hl else 0.6) * (1 if on else 0.15)
        d.line([P(u), P(v)], fill=mix(BG, col, ea), width=max(1, int(1 + 3 * abs(w))))
    for n in nodes:
        x, y = P(n)
        r = 9
        fill = AMBER if n.startswith("o") else PANEL
        d.ellipse([x - r, y - r, x + r, y + r], fill=mix(BG, fill, a), outline=mix(BG, FG, a), width=2)


def scene_neat(ctx, name="neat"):
    """Beats: population; mutation; selection and crossover; species."""
    base = [(("i1", "h1"), 0.8, True), (("i2", "h1"), -0.5, True), (("i2", "h2"), 0.6, True),
            (("i3", "h2"), -0.9, True), (("h1", "o1"), 0.9, True), (("h2", "o2"), 0.7, True),
            (("h2", "o1"), -0.3, True)]
    nodes = ["i1", "i2", "i3", "h1", "h2", "o1", "o2"]
    rng = np.random.default_rng(1)
    pop = [[(e, w + rng.normal(0, 0.5), on) for e, w, on in base] for _ in range(24)]
    species_of = [int(i % 4 == 0) + int(i % 7 == 0) for i in range(24)]

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "How NEAT evolves networks", t, "NeuroEvolution of Augmenting Topologies")
        k, lt, _ = b.local(t)
        if k == 0:
            for i in range(24):
                cx, cy = 80 + (i % 8) * 140, 140 + (i // 8) * 150
                _toy(d, (cx, cy, cx + 100, cy + 100), pop[i], nodes, fade(t, b.at(0) + 0.05 * i))
            paragraph(d, (80, 600), "Start with a population of 200 random networks. Each one is a genome: "
                      "a list of nodes and weighted connections.", 90, 22, FG, fade(t, b.at(0) + 0.8))
        elif k == 1:
            u1 = ease((t - b.at(1) - 1.0) / 1.2)
            u2 = ease((t - b.at(1) - 3.5) / 1.2)
            u3 = ease((t - b.at(1) - 6.0) / 1.2)
            edges = [(e, w + (0.6 * u1 if e == ("i2", "h1") else 0), on) for e, w, on in base]
            nn = list(nodes)
            hl = [("i2", "h1")]
            if u2 > 0:
                edges.append((("i1", "h2"), 0.7 * u2, True))
                hl = [("i1", "h2")]
            if u3 > 0:
                nn.append("h3")
                edges = [(e, w, on and not (e == ("h2", "o1"))) for e, w, on in edges]
                edges += [(("h2", "h3"), 1.0 * u3, True), (("h3", "o1"), -0.3 * u3, True)]
                hl = [("h2", "h3"), ("h3", "o1")]
            _toy(d, (380, 150, 900, 520), edges, nn, 1.0, hl)
            labels = [("change a weight", u1), ("add a connection", u2), ("add a node", u3)]
            for i, (s, u) in enumerate(labels):
                text(d, (80 + 400 * i, 580), s, 26, AMBER if u > 0 else MUTED, fade(t, b.at(1) + 0.4), bold=True)
            text(d, (80, 630), "Adding a node splits a connection, so networks grow deeper over time.", 19, MUTED,
                 fade(t, b.at(1) + 6.5))
        elif k == 2:
            order = np.argsort([-sum(w for _, w, _ in g) for g in pop])
            for r, i in enumerate(order):
                cx, cy = 80 + (r % 8) * 140, 140 + (r // 8) * 150
                keep = r < 5
                _toy(d, (cx, cy, cx + 100, cy + 100), pop[i], nodes, 1.0 if keep else 0.25)
                if keep:
                    d.rectangle([cx - 10, cy - 10, cx + 110, cy + 110], outline=GREEN, width=2)
            paragraph(d, (80, 600), "Each generation, the best fifth survive and have children: copies, "
                      "or mixes of two parents, plus mutations. The two very best pass on unchanged.", 90, 22, FG,
                      fade(t, b.at(2) + 0.3))
        else:
            cols = [BLUE, GREEN, VIOLET]
            for i in range(24):
                s = species_of[i]
                cx, cy = 80 + (i % 8) * 140, 140 + (i // 8) * 150
                _toy(d, (cx, cy, cx + 100, cy + 100), pop[i], nodes, 1.0)
                u = fade(t, b.at(3) + 0.8)
                d.rectangle([cx - 10, cy - 10, cx + 110, cy + 110], outline=mix(BG, cols[s], u), width=3)
            paragraph(d, (80, 600), "Similar networks form species and mostly compete inside their own species. "
                      "A new structure gets a few generations to prove itself before it has to beat everyone.",
                      90, 22, FG, fade(t, b.at(3) + 0.3))
        return img
    return Scene(name, render, [7.0, 10.0, 8.0, 8.0])


def scene_fitness(ctx, name="fitness"):
    """Beats: three shared terrains; early stop; the stand-still trap."""
    p0 = Player(ctx.gen0, ctx.config, ctx.stage_seeds.get("gen0", C.HOF_SEEDS[0]))
    terrains = [Player(ctx.gen0, ctx.config, s).frame(0) for s in (501, 502, 503)]

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "Fitness: how a network is judged", t)
        k, lt, _ = b.local(t)
        if k == 0:
            for i in range(3):
                x = 60 + i * 400
                paste_frame(img, terrains[i], (x, 150, 360, 240), border=PANEL)
                C.label(ImageDraw.Draw(img), (x + 10, 160), f"terrain {i + 1}", 16)
            d = ImageDraw.Draw(img)
            paragraph(d, (60, 440), "Every network walks the same 3 random terrains, and its fitness is the "
                      "average score. The terrains change every generation, so nobody can memorise them.",
                      90, 22, FG, fade(t, b.at(0) + 0.4))
        elif k == 1:
            step = min(int((t - b.at(1)) * 50 * 0.9), 300)
            paste_frame(img, p0.frame(step), (60, 130, 560, 373), border=PANEL)
            d = ImageDraw.Draw(img)
            prog = min(step, 150) / 150
            d.rectangle([700, 200, 1200, 220], fill=PANEL)
            d.rectangle([700, 200, 700 + 500 * prog, 220], fill=AMBER)
            text(d, (700, 170), "150 steps without 1 unit of forward progress", 18, MUTED)
            if step >= 150:
                text(d, (700, 260), "STOP", 44, RED, fade(t, b.at(1) + 3.2), bold=True)
                paragraph(d, (700, 330), "Early stop: the episode ends. A network that just stands there "
                          "doesn't burn 1600 steps of compute.", 40, 21, FG, fade(t, b.at(1) + 3.4))
        else:
            vals = [("stand still", -9, MUTED), ("try to walk, then fall", -100, RED), ("walk the course", 300, GREEN)]
            x0, zero = 160, 480
            for i, (lab, v, col) in enumerate(vals):
                a = fade(t, b.at(2) + 0.4 + 0.8 * i)
                x = x0 + i * 330
                h = v * 0.6
                d.rectangle([x, min(zero, zero - h), x + 140, max(zero, zero - h)], fill=mix(BG, col, a))
                text(d, (x + 70, zero - h - 26 if v > 0 else zero - h + 24), f"{v:+d}", 24, FG, a, bold=True, anchor="mm")
                text(d, (x + 70, 640), lab, 20, FG, a, anchor="mm")
            d.line([(120, zero), (1180, zero)], fill=MUTED)
            paragraph(d, (120, 140), "The catch: standing still scores better than a brave attempt that falls. "
                      "That makes standing still a trap.", 80, 22, FG, fade(t, b.at(2) + 0.2))
        return img
    return Scene(name, render, [8.0, 9.0, 9.0])


def scene_hof(ctx, name="hof"):
    """Beats: noisy best; hall of fame."""
    rows = ctx.stats

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "Picking the champion", t, "the hall of fame")
        k, _, _ = b.local(t)
        gs = [r["g"] for r in rows]
        champ, ch = None, []
        for r in rows:
            if r["hof"] is not None and (champ is None or r["hof"] > champ):
                champ = r["hof"]
            ch.append(champ)
        series = [(gs, [r["best"] for r in rows], ORANGE, 2, False)]
        if k >= 1:
            series.append((gs, ch, BLUE, 4, True))
        ymax = max(r["best"] for r in rows) + 30
        line_chart(d, (110, 150, 840, 600), series, gs[-1], -150, ymax, 1.0, 50)
        text(d, (900, 160), "Best of each generation", 24, ORANGE, 1.0, bold=True)
        paragraph(d, (900, 200), "Noisy: 3 terrains can be lucky or unlucky.", 30, 19, FG, fade(t, 0.5))
        if k >= 1:
            text(d, (900, 300), "Champion", 24, BLUE, fade(t, b.at(1) + 0.3), bold=True)
            paragraph(d, (900, 340), "Every 10 generations the top 5 walk 10 terrains they have never seen, "
                      "with no early stop. The best average becomes the champion.", 30, 19, FG, fade(t, b.at(1) + 0.6))
        return img
    return Scene(name, render, [7.0, 9.0])


def scene_curve(ctx, name="curve"):
    """Beats: stuck; breakthrough; to the end."""
    rows = ctx.stats
    gs = [r["g"] for r in rows]
    champ, ch = None, []
    for r in rows:
        if r["hof"] is not None and (champ is None or r["hof"] > champ):
            champ = r["hof"]
        ch.append(champ)
    marks = []
    for m in C.MILESTONES:
        if m in ctx.milestones:
            marks.append((ctx.milestones[m][1]["generation"], "first > 0" if m == "first_positive" else f"≥ {m}"))
    first = marks[0][0] if marks else gs[-1] // 2

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "The training run", t, f"{ctx.G} generations")
        k, lt, dur = b.local(t)
        stops = [first / gs[-1] * 0.95, min(1.0, (first + 35) / gs[-1]), 1.0]
        start = 0 if k == 0 else stops[k - 1]
        frac = start + (stops[k] - start) * ease(lt / max(1.0, dur - 1.0))
        ymax = max(max(r["best"] for r in rows), max([c for c in ch if c is not None] + [0])) + 30
        line_chart(d, (110, 140, 1180, 600),
                   [(gs, [r["mean"] for r in rows], GREEN, 2, False), (gs, [r["best"] for r in rows], ORANGE, 2, False),
                    (gs, ch, BLUE, 4, True)], gs[-1], -150, ymax, frac, 50, marks=marks)
        for i, (lab, col) in enumerate([("champion (10 unseen terrains)", BLUE), ("best of generation", ORANGE),
                                        ("population average", GREEN)]):
            d.rectangle([120 + i * 330, 662, 140 + i * 330, 666], fill=col)
            text(d, (148 + i * 330, 664), lab, 16, MUTED, anchor="lm")
        return img
    return Scene(name, render, [8.0, 8.0, 9.0])


def scene_stages(ctx, name="stages"):
    """Beats: the stages side by side; the champion."""
    stages = ctx.stages()
    players = [(key, title, meta, Player(g, ctx.config, ctx.stage_seeds[key])) for key, title, g, meta in stages]
    pc = Player(ctx.champ, ctx.config, ctx.champ_seed)

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "Watching it learn", t, "each champion on a typical terrain, 2x speed")
        k, lt, _ = b.local(t)
        if k == 0:
            n = len(players)
            cols = 2 if n == 4 else 3
            tw = 460 if n == 4 else 370
            th = int(tw * 2 / 3)
            rows_n = (n + cols - 1) // cols
            y0 = 130 if rows_n > 1 else 220
            for i, (key, title, meta, p) in enumerate(players[:6]):
                x = (W - cols * tw - (cols - 1) * 30) // 2 + (i % cols) * (tw + 30)
                y = y0 + (i // cols) * (th + 30)
                paste_frame(img, p.at_time(t, speed=2.0), (x, y, tw, th), border=PANEL)
                d = ImageDraw.Draw(img)
                lab = f"{title} · gen {meta['generation']}" + (f" · avg {meta['mean']:+.0f}" if meta.get("mean") is not None else "")
                C.label(d, (x + 10, y + 10), lab, 16)
        else:
            paste_frame(img, pc.at_time(t - b.at(1), speed=1.5), (140, 120, 1000, 567), border=AMBER)
            d = ImageDraw.Draw(img)
            C.label(d, (160, 140), f"Champion · generation {ctx.champ_meta['generation']} · avg {ctx.champ_meta['mean']:+.0f}", 22)
        return img
    return Scene(name, render, [12.0, 10.0], min_beats=[10.0, 8.0])


def scene_growth(ctx, name="growth"):
    """Beats: size over time; the champion's structure."""
    rows = ctx.stats

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "The network grows", t, "structure of the best network each generation")
        k, _, _ = b.local(t)
        gs = [r["g"] for r in rows]
        if k == 0:
            line_chart(d, (110, 150, 1180, 600), [(gs, [r["hidden"] for r in rows], VIOLET, 3, True),
                                                  (gs, [r["depth"] for r in rows], AMBER, 3, True)],
                       gs[-1], 0, max(max(r["hidden"] for r in rows), 8) + 3, ease((t - 0.3) / 4), 2)
            text(d, (130, 640), "hidden nodes in use", 18, VIOLET)
            text(d, (400, 640), "layers (input to output)", 18, AMBER)
        else:
            draw_net(d, ctx.champ, ctx.config, (150, 140, 780, 650), a=fade(t, b.at(1)))
            tp = ctx.topo
            paragraph(d, (900, 180), f"The champion has {tp['hidden_in_use']} hidden nodes in {tp['depth'] - 1} "
                      f"hidden layers and {tp['enabled_conns']} connections. It started as 8 nodes in one layer; "
                      "everything else was added by mutation and kept because it helped.", 26, 20, FG,
                      fade(t, b.at(1) + 0.5))
        return img
    return Scene(name, render, [8.0, 9.0])


def scene_species(ctx, name="species"):
    """Beats: species over time; the bug."""
    rows = ctx.stats

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "Species", t, "and a bug found along the way")
        k, _, _ = b.local(t)
        gs = [r["g"] for r in rows]
        line_chart(d, (110, 150, 760, 600), [(gs, [r["species"] for r in rows], GREEN, 3, False)],
                   gs[-1], 0, max(r["species"] for r in rows) + 5, 1.0, 10)
        text(d, (130, 640), "number of species", 18, GREEN)
        if k == 0:
            paragraph(d, (820, 180), "A setting called the compatibility threshold decides how different two "
                      "networks must be to count as different species. The code adjusts it to aim for about 10.",
                      30, 20, FG, fade(t, b.at(0) + 0.3))
        else:
            paragraph(d, (820, 180), "In the first hour the adjustment ran away: neat-python never merges species, "
                      "so the count barely fell and the threshold kept rising, to 38. It is now kept between "
                      "0.3 and 3, and the run was resumed from its checkpoint.", 30, 20, FG, fade(t, b.at(1) + 0.3))
        return img
    return Scene(name, render, [8.0, 11.0])


def scene_results(ctx, name="results"):
    """Beats: 100 terrains; falls; comparison."""
    ev = ctx.eval100
    rewards = np.array(ev["rewards"])

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "The final test", t, "100 terrains it has never seen")
        k, _, _ = b.local(t)
        if k < 2:
            bins = np.arange(-150, 351, 25)
            hist, _ = np.histogram(np.clip(rewards, -149, 349), bins)
            x0, y1, bw = 110, 560, 50
            mx = max(hist.max(), 1)
            for i, h in enumerate(hist):
                a = fade(t, 0.3 + 0.04 * i)
                col = RED if bins[i] < 0 else BLUE
                if k == 1 and bins[i] >= 0:
                    col = mix(BG, BLUE, 0.35)
                hh = h / mx * 360
                d.rectangle([x0 + i * bw + 2, y1 - hh, x0 + (i + 1) * bw - 2, y1], fill=mix(BG, col, a))
            for v in (-100, 0, 100, 200, 300):
                xx = x0 + (v + 150) / 25 * bw
                text(d, (xx, y1 + 18), str(v), 14, MUTED, anchor="mm")
            text(d, ((x0 + x0 + 20 * bw) / 2, y1 + 44), "score per episode", 14, MUTED, anchor="mm")
            if k == 0:
                text(d, (110, 130), f"average {ev['mean']:+.0f}  (spread {ev['std']:.0f})", 30, FG, fade(t, 0.5), bold=True)
            else:
                text(d, (110, 130), f"falls in {ev['fell_frac'] * 100:.0f} of 100 episodes", 30, RED, fade(t, b.at(1) + 0.3), bold=True)
        else:
            bars = [("NEAT (this run)", ev["mean"], AMBER), ("PPO", C.PRIOR["PPO"], MUTED), ("TD3", C.PRIOR["TD3"], MUTED)]
            for i, (lab, v, col) in enumerate(bars):
                a = fade(t, b.at(2) + 0.4 + 0.6 * i)
                y = 220 + i * 120
                d.rectangle([360, y, 360 + max(0, v) * 2.6, y + 70], fill=mix(BG, col, a))
                text(d, (340, y + 35), lab, 22, FG, a, anchor="rm")
                text(d, (370 + max(0, v) * 2.6, y + 35), f"{v:.0f}", 22, FG, a, bold=True, anchor="lm")
            x300 = 360 + 300 * 2.6
            d.line([(x300, 190), (x300, 590)], fill=mix(BG, GREEN, fade(t, b.at(2) + 2.0)), width=2)
            text(d, (x300, 175), "solved: 300", 16, GREEN, fade(t, b.at(2) + 2.0), anchor="mm")
            text(d, (110, 630), "PPO and TD3: gradient-based runs on the same laptop (earlier projects).", 17, MUTED,
                 fade(t, b.at(2) + 1.0))
        return img
    return Scene(name, render, [8.0, 7.0, 9.0])


def scene_laptop(ctx, name="laptop"):
    last = ctx.stats[-1]

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "What it took", t)
        tiles = [(f"{last['wall'] / 3600:.1f} h", "of training"), (f"{last['steps'] / 1e6:.0f} M", "simulated steps"),
                 (f"{last['saved'] / 1e6:.0f} M", "steps skipped by early stop"), ("4 cores", "no GPU, ~50 MB per worker")]
        for i, (big, small) in enumerate(tiles):
            a = fade(t, 0.4 + 0.5 * i)
            x = 80 + i * 290
            d.rounded_rectangle([x, 200, x + 260, 380], 14, fill=mix(BG, PANEL, a))
            text(d, (x + 24, 250), big, 44, AMBER, a, bold=True)
            text(d, (x + 24, 320), small, 18, FG, a)
        paragraph(d, (80, 450), "Every network is evaluated on its own, so NEAT spreads across CPU cores with no "
                  "GPU and almost no memory. For much of this run it shared the laptop with another training job.",
                  90, 22, FG, fade(t, 2.5))
        return img
    return Scene(name, render, [11.0])


def scene_outro(ctx, name="outro"):
    p = Player(ctx.champ, ctx.config, ctx.champ_seed)

    def render(t, b):
        img = blank()
        C.cover_frame(img, p.at_time(t + 10), dim=0.6)
        d = ImageDraw.Draw(img)
        text(d, (W // 2, 300), "No gradients. No GPU. Just evolution.", 44, FG, fade(t, 0.3), bold=True, anchor="mm")
        text(d, (W // 2, 370), "Code, clips and data: jca374.github.io/bipedal-neat", 24, AMBER, fade(t, 1.0), anchor="mm")
        return img
    return Scene(name, render, [8.0])
