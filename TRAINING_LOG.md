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
- Gen 99: champion +111.1 (milestone 100). Gen 149: +132. Gen 199: +154. Gen 238: **+157.7** on 10 held-out terrains.
- 100-episode check of the gen-149 champion: +140.1 ± 12, 0 falls (`evaluate.py`).
- End of the 2 h budget at gen 238.

## Run 1, phase 3: overnight (`overnight.sh`, 2026-10-20 → 21)
- Resumed `baseline` in 1 h chunks until 05:00, or until the champion reaches 300 (`--stop-at 300`). 6 workers when the CPU is free, 4 when another training run is active.
- No config changes, so the curve stays one continuous run.
- **Plateau (gen 238 → 333):** champion stuck at ~158 (157.7 → 158.2). The 100-episode check shows a careful, slow walker with almost no falls.
- **Change (PLAN phase 4, step 1), from the 23:30 chunk:** K 3 → 5 seeds per genome and pop 200 → 300, via `runs/baseline/overrides.json`, which `train.py` reads on resume (so the running `overnight.sh` didn't need editing). Changes are recorded in `run_config.json` → `history`.
