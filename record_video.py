"""Record an mp4 at the comparison spec (600x400, 50 fps, 1600 steps, seed 42).

  python record_video.py --run runs/baseline --out videos/neat_seed42.mp4

Reuses add_label / add_score / write_video from ../comparison_videos so the NEAT
panel looks identical to the PPO/TD3 ones.
"""
import argparse
import importlib.util
import sys
import types
from pathlib import Path

import common

COMPARISON = common.ROOT.parent / "comparison_videos" / "create_comparison_videos.py"


def load_video_helpers():
    # That script imports stable_baselines3 at module level; we only need the cv2
    # helpers, so stub SB3 rather than installing it here.
    if "stable_baselines3" not in sys.modules:
        sb3 = types.ModuleType("stable_baselines3")
        sb3.PPO = sb3.TD3 = object
        vec = types.ModuleType("stable_baselines3.common.vec_env")
        vec.DummyVecEnv = vec.VecNormalize = object
        sys.modules.update({"stable_baselines3": sb3,
                            "stable_baselines3.common": types.ModuleType("stable_baselines3.common"),
                            "stable_baselines3.common.vec_env": vec})
    spec = importlib.util.spec_from_file_location("create_comparison_videos", COMPARISON)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def record_neat_episode(run, seed=42, max_steps=1600):
    """Same contract as create_comparison_videos.record_episode: (frames, total, cumulative)."""
    genome, config, _ = common.resolve_genome(run)
    net = common.make_net(genome, config)
    env = common.make_env(render_mode="rgb_array")
    frames, cumulative = [], []

    def on_frame(e, total):
        frames.append(e.render())
        cumulative.append(total)

    total, steps, _, _ = common.run_episode(net, env, seed, early_stop=False,
                                            max_steps=max_steps, on_frame=on_frame)
    env.close()
    frames, cumulative = frames[:max_steps], cumulative[:max_steps]
    # Freeze the final frame if the episode ended early, like the comparison script.
    frames += [frames[-1]] * (max_steps - len(frames))
    cumulative += [total] * (max_steps - len(cumulative))
    return frames, total, cumulative


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--label", default="NEAT")
    args = ap.parse_args()
    h = load_video_helpers()
    frames, total, cum = record_neat_episode(args.run, args.seed, args.steps)
    frames = [h.add_score(h.add_label(f, args.label), s) for f, s in zip(frames, cum)]
    out = Path(args.out or Path(args.run) / f"neat_seed{args.seed}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    h.write_video(frames, out, args.fps)
    print(f"reward {total:.1f}, {len(frames)} frames {frames[0].shape[1]}x{frames[0].shape[0]} -> {out}")


if __name__ == "__main__":
    main()
