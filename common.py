"""Shared helpers: env creation, static obs scaling, episode rollout, topology stats.

Everything inference needs is the genome pickle + the .ini config. There is no
running normalisation (lesson #1 in PLAN.md).
"""
import os
import pickle
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import gymnasium as gym
import neat
import numpy as np
from neat.graphs import feed_forward_layers

ENV_ID = "BipedalWalker-v3"
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "neat-bipedal.ini"

# Static per-dimension divisors, from the 99th percentile of |obs| (see bench.py).
# 0 hull angle, 1 hull ang. vel, 2 vx, 3 vy,
# 4-7 hip1 angle/speed, knee1 angle/speed, 8 contact1,
# 9-12 hip2 angle/speed, knee2 angle/speed, 13 contact2, 14-23 lidar.
OBS_SCALE = np.array(
    [1.0, 0.25, 0.5, 0.5,
     1.2, 2.0, 1.2, 2.0, 1.0,
     1.2, 2.0, 1.2, 2.0, 1.0]
    + [1.0] * 10,
    dtype=np.float64,
)

# Early termination: stop if the hull moves less than this over the window.
STALL_WINDOW = 150
STALL_MIN_DX = 1.0


def scale_obs(obs):
    return np.clip(np.asarray(obs, dtype=np.float64) / OBS_SCALE, -1.0, 1.0)


def load_config(path=DEFAULT_CONFIG):
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                       neat.DefaultSpeciesSet, neat.DefaultStagnation, str(path))


def make_env(render_mode=None):
    return gym.make(ENV_ID, render_mode=render_mode)


class FastNet:
    """numpy equivalent of neat.nn.FeedForwardNetwork for tanh/sum nodes.

    Node values live in one vector ordered [inputs, layer1, layer2, ...] so each
    layer is a contiguous slice computed with one matmul. neat's tanh is
    tanh(clamp(2.5 * (bias + response * sum), +-60)); the 2.5 and response are
    folded into W and b, and the clamp is a no-op for tanh in float64.
    test_neat.py checks it matches neat's network.
    """

    def __init__(self, genome, config):
        gc = config.genome_config
        conns = [cg for cg in genome.connections.values() if cg.enabled]
        layers = [sorted(l) for l in
                  feed_forward_layers(gc.input_keys, gc.output_keys, [cg.key for cg in conns])]
        keys = list(gc.input_keys) + [k for l in layers for k in l]
        keys += [k for k in gc.output_keys if k not in keys]  # unreachable outputs stay 0
        idx = {k: i for i, k in enumerate(keys)}
        self.n_in = len(gc.input_keys)
        self.out_idx = np.array([idx[k] for k in gc.output_keys])
        self.layers = []
        for layer in layers:
            ng = [genome.nodes[n] for n in layer]
            if any(g.activation != "tanh" or g.aggregation != "sum" for g in ng):
                raise ValueError("FastNet supports only tanh/sum nodes")
            lo = idx[layer[0]]
            W = np.zeros((len(layer), len(keys)))
            for cg in conns:
                a, b = cg.key
                if b in idx and lo <= idx[b] < lo + len(layer):
                    W[idx[b] - lo, idx[a]] += cg.weight
            resp = np.array([g.response for g in ng])
            W *= (2.5 * resp)[:, None]
            b = 2.5 * np.array([g.bias for g in ng])
            self.layers.append((lo, lo + len(layer), W, b))
        self.values = np.zeros(len(keys))

    def activate(self, inputs):
        v = self.values
        v[:self.n_in] = inputs
        for lo, hi, W, b in self.layers:
            v[lo:hi] = np.tanh(W @ v + b)
        return v[self.out_idx]


def make_net(genome, config):
    return FastNet(genome, config)


def run_episode(net, env, seed, early_stop=True, max_steps=1600, on_frame=None):
    """Roll out one episode. Returns (total_reward, steps, steps_saved, fell).

    steps_saved = steps cut off by early termination (1600 - steps when stalled).
    on_frame(env, total_reward) is called before every step, for rendering.
    """
    obs, _ = env.reset(seed=int(seed))
    hull = env.unwrapped.hull
    xs = []
    total = 0.0
    steps = 0
    fell = False
    for steps in range(1, max_steps + 1):
        if on_frame is not None:
            on_frame(env, total)
        action = net.activate(scale_obs(obs))
        obs, reward, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
        total += float(reward)
        if terminated:
            fell = reward <= -100
            break
        if truncated:
            break
        if early_stop:
            xs.append(hull.position.x)
            if len(xs) > STALL_WINDOW and xs[-1] - xs[-1 - STALL_WINDOW] < STALL_MIN_DX:
                return total, steps, max_steps - steps, False
    if on_frame is not None:
        on_frame(env, total)
    return total, steps, 0, fell


def topology(genome, config):
    """Hidden nodes actually on an input->output path, and number of FF layers.

    feed_forward_layers only keeps nodes that are reachable from the inputs AND
    needed for an output, so dangling hidden nodes are not counted.
    """
    gc = config.genome_config
    conns = [cg.key for cg in genome.connections.values() if cg.enabled]
    layers = feed_forward_layers(gc.input_keys, gc.output_keys, conns)
    used = set().union(*layers) if layers else set()
    hidden_in_use = len(used - set(gc.output_keys))
    return {
        "hidden_in_use": hidden_in_use,
        "depth": len(layers),
        "nodes": len(genome.nodes),
        "enabled_conns": len(conns),
    }


def has_hidden_layer(genome, config):
    return topology(genome, config)["hidden_in_use"] >= 1


def save_genome(path, genome, meta=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"genome": genome, "meta": meta or {}}, f)


def load_genome(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["genome"], d.get("meta", {})


def resolve_genome(run_or_file):
    """Accept runs/<name>, runs/<name>/best, or a .pkl path. Returns (genome, config, meta)."""
    p = Path(run_or_file)
    if p.is_dir():
        if (p / "best" / "genome.pkl").exists():
            p = p / "best"
        genome_path = p / "genome.pkl"
    else:
        genome_path = p
    genome, meta = load_genome(genome_path)
    ini = genome_path.parent / "config.ini"
    config = load_config(ini if ini.exists() else DEFAULT_CONFIG)
    return genome, config, meta
