"""Watch a saved genome: python play.py --run runs/baseline [--episodes 3] [--seed 42]"""
import argparse

import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    genome, config, meta = common.resolve_genome(args.run)
    print("topology:", common.topology(genome, config), "| logged HOF mean:", meta.get("mean"))
    net = common.make_net(genome, config)
    env = common.make_env(render_mode="human")
    try:
        for ep in range(args.episodes):
            r, steps, _, fell = common.run_episode(net, env, args.seed + ep, early_stop=False)
            print(f"episode {ep}: reward {r:.1f}, steps {steps}, fell {fell}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
