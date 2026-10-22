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
- K=5 / pop 300 from gen 430: champion 163 → 168 (gen ~470) → **199.96** (gen ~520) on 10 held-out terrains, then flat until the 05:00 deadline (gen 674). Milestone 200 was missed by 0.04.
- **Final champion, 100 unseen terrains:** mean **+179.0 ± 75.1**, median 206, middle half 191–220, falls 17/100, 0 runs ≥ 300. 6 hidden nodes in 5 hidden layers, 37 connections. Evolution pruned the 24→8→4 start (224 connections) into a deep, sparse chain.
- Milestones on 100 terrains: first_positive +19.9 ± 8.3 (0 falls); milestone 100 +103.3 ± 46.8 (4 falls). The gen-149 champion: +140.1 ± 12 (0 falls).
- Trade-off: the later champions walk faster and score more, but fall more. Falls are now what limits the score.

## Next ideas (not run yet)
1. Action repeat 2 (PLAN phase 4, step 2): more forward motion per decision.
2. Fitness shaping against falls, e.g. no early-stop credit for standing, or a hull-stability bonus. Report raw reward only.
3. More time: the champion improved in bursts (gen 99, 238, 520), so a longer run may break the plateau again.

## Fall analysis of the gen-674 champion (2026-10-22, `experiments/`)
- `record_episodes.py` (300 terrains, seeds 3e7+) + `analyze_falls.py` (failure-analysis-report method). Report: `docs/analysis-v1/report.html`.
- 39/300 falls (13%). 90% are forward pitch-overs. The walker normally leans back ~0.7 rad. Before a fall it speeds up (vx 0.27 → 0.45 over ~60 steps), the hull rises (1.57 → 1.9), then it pitches past vertical. Last healthy step → fall: median 15 steps (0.3 s).
- Terrain is not the cause: 77% flat before falls vs 67% overall. No candidate signal adds much beyond the 24 inputs (max partial correlation 0.19), so this is a selection/policy problem, not missing sensors.
- Also: no episode reaches the end of the course. All 261 non-falling episodes run out of time at x ≈ 74 of ~88.
- `reselect.py`: re-selecting from the final population (top 40, 30 selection terrains, paired on 100 new terrains) found safer walkers (4–7 falls vs 16) but not better ones. Best difference −3.4 [95% CI −19.5, +13.9]. Speed and falls trade off, so the champion was kept.
- The champion re-scored on 20 hall-of-fame terrains: 200.0 → 179.9. Its 10-terrain score was lucky.

## Run 1, phase 4: fall penalty (2026-10-22 12:34 → 17:00)
- Resumed from gen 674 with `overrides.json`: K=5, pop 300, **fall_penalty 100** (training fitness only; all reported scores stay raw), **hof_seeds 20**. The previous champion is kept as `runs/baseline/champion_v1` for a paired comparison on 100 new terrains (`experiments/compare.py`, seeds 4e7+).
- 4 workers: two lunar-lander runs shared the CPU.
- Result at the 17:00 deadline (gen 740): HOF champion from gen 709, 188.8 on 20 terrains; 4 hidden nodes in 1 hidden layer, 30 connections.
- 100 eval terrains: **+173.2 ± 63, falls 9%**. v1: +179.0 ± 75, falls 17%.
- Paired on 100 new terrains (`compare.py`): v1 173.3 (17 falls) vs new 167.9 (12 falls). Difference **−5.4, 95% CI [−26.5, +16.0]**; fall difference CI [−0.15, +0.05]; the new champion wins on 40% of terrains.
- Fall analysis of the new champion: 27/300 falls (9%) vs 39/300 before, still 93% forward pitch-overs.
- **Conclusion:** the fall penalty cut falls by about a third to a half, but the score did not change significantly. Speed and falls still trade off. The new champion is published: same score, fewer falls.
- Ops note: `make_site.py` was OOM-killed in the queue (it kept 1600 frames per clip in RAM, while two other training runs filled swap). Clips now stream to ffmpeg, peak 157 MB.
