# Training log

## Run 1: `baseline` (2026-10-20)
- Config: pop 200, K=3, 24→8→4 `full_nodirect`, early stop, 4 workers, sharing the CPU with a lunar-lander NEAT run on 8 threads.
- Gen 0–69: training-best stuck at −7…−9. That is the "stand still until early stop" optimum.
- The HOF re-eval (full episodes, 10 held-out seeds) still found walkers:
  - Gen 69: champion +23.3 ±8.1 (9 hidden, depth 4), `first_positive` milestone.
  - Gen 88: best +23.6.
- About 16–35 s per generation under contention.
- **Bug found:** the adaptive `compatibility_threshold` wound up from 0.6 to 38. neat never merges existing species, so their count only drops via stagnation, and the controller kept pushing. Species spiked to 63 at gen 5.
- **Fix (gen 88):** clamp to [0.3, 3.0]. Stopped with Ctrl-C, resumed from the checkpoint; the restored threshold is clamped to 3.0. Everything else is unchanged.
