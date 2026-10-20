# Plan: NEAT BipedalWalker on this laptop (`bipedal-neat/`)


## Context

We've already trained BipedalWalker-v3 three times on this laptop:
- `bipedal-gpt/` (PPO): best was 271.6, and the "best" checkpoint was broken.
- `bipedal-claude-ppo/` (PPO, cleaner version).
- `bipedal-claude/` (TD3): solved it, 308 to 319 points in about 2 hours.

All three used gradient-based RL with Stable-Baselines3. Now we want a **gradient-free, neuroevolution baseline**. NEAT (NeuroEvolution of Augmenting Topologies) evolves both the weights and the shape of the network.

The goal is a NEAT walker that trains on **this laptop** and plugs into the same milestone and comparison-video workflow, so it can be shown next to PPO and TD3.

**Hardware** (measured; matches `bipedal-claude/CLAUDE.md`):
- CPU: Intel i5-8250U, 4 cores / 8 threads.
- RAM: 7.6 GB total, about 4 GB really free. Swap is already in use.
- No GPU, Python 3.12.

**Capacity check (2026-10-20, measured):**
- About 4.2–5 GB RAM available; 1.4 GB swap was already in use (Firefox + Claude sessions). Disk: 41 GB free.
- Random-policy env, idle machine: 1 core 3.5k steps/s; 4 workers 12.9k; 6 workers 14.2k; 8 workers 15.5k. 8 workers is only ~9% over 6 (hyperthreads), so **6 is the default**.
- RAM per worker: ~50 MB.
- Env cost is dominated by Box2D plus `np.clip` on scalars inside `BipedalWalker.step`. That stays untouched, for comparability.
- **The machine is shared**: other projects (e.g. `lunarlander-neat-wind`) sometimes run NEAT on 4 threads. Check `uptime` before benchmarking or starting long runs.

NEAT suits this machine well. It needs no GPU, no replay buffer, and almost no RAM per worker. It scales with CPU cores, because each genome is evaluated on its own.

## Lessons carried over from the earlier projects

| Earlier finding | What we do in NEAT |
|---|---|
| A missing or mismatched VecNormalize file wrecked eval and playback (all 3 projects) | **No running normalisation.** Use fixed, static input scaling and clipping in code. One saved file (genome pickle + config) is everything inference needs. |
| `TD3.load()` silently kept the old hyperparameters, and the fine-tune collapsed to -170 (TRAINING_LOG Run 3) | `neat.Checkpointer.restore_checkpoint` **also restores the config stored inside the checkpoint**. On resume we rebuild `neat.Config` from the current `.ini` and attach it to the restored population. Print the effective params at startup. Add a test for it. |
| The model peaked and then got worse; the latest model wasn't the best (TD3 970k→1M; GPT "best" broken) | Keep a **hall of fame**. Re-evaluate the top genomes on held-out seeds and pick the champion by the **mean over many seeds**, never by one lucky episode. |
| The same model gave different scores on different seeds (comparison README) | Fitness = mean reward over K seeds per generation. All genomes in a generation share the same seeds. Seeds rotate every generation. The final score is a 100-episode eval. |
| More than 8 parallel envs caused swap thrashing, and 16 crashed (OOM) | `ParallelEvaluator` with **6 workers by default** (`--workers` flag), plus `OMP_NUM_THREADS=1`. Watch RAM in phase 0. |
| Milestone saves + metadata (wall clock, step count) made comparisons easy | Same milestone names: `first_positive/100/200/300`. Same metadata JSON, plus generation and **total env steps**, so sample efficiency is comparable to the PPO and TD3 tables. |
| Simple monolithic `train.py` + smoke test worked well (claude-ppo) | Same layout: one `train.py`, small helper scripts, `pytest -m smoke`. |
| Pinned deps avoided breakage | Same pins as the RL projects: `gymnasium[box2d]==0.29.1`, `numpy<2`. Add `neat-python==0.92` and `opencv-python`. box2d already builds on this Python 3.12 (see `bipedal-claude-ppo/.venv`). |
| Comparison videos are 600×400 per panel, 50 fps, 1600 steps, seed 42 (`comparison_videos/create_comparison_videos.py`) | `record_video.py` uses the same spec. Later, add a third "NEAT" panel by reusing `record_episode`, `add_label`, `add_score` and `write_video` from that script. |

## Layout

```
bipedal-neat/
  PLAN.md                  # this file
  README.md, CLAUDE.md     # written after phase 2, once commands are real
  requirements.txt
  config/neat-bipedal.ini  # NEAT hyperparameters
  train.py                 # evolution loop, parallel eval, checkpoints, milestones, hall of fame, resume
  common.py                # make_env, scale_obs, run_episode(genome, config, seed, early_stop) -> (reward, steps)
  evaluate.py              # N-episode eval of a saved genome (default 100), prints mean/std
  play.py                  # render_mode="human" playback
  record_video.py          # mp4 at the comparison spec
  bench.py                 # phase 0: env steps/s per core + RAM per worker
  tests/test_neat.py       # smoke + resume-applies-new-config + deterministic run_episode
  runs/<run-name>/         # checkpoints/, milestones/, best/, stats.csv, run_config.json
```

## Key design decisions

**Inputs and outputs**
- Inputs: all 24 observations. Scale them statically (the per-dimension bounds of the BipedalWalker obs are known) and clip to [-1, 1].
- Outputs: 4 values with tanh activation, which matches the action range [-1, 1].

**Network: at least one hidden layer, guaranteed** (changed 2026-10-20)
- Start feed-forward with a hidden layer: `num_hidden = 8`, `initial_connection = full_nodirect` (24→8→4, no direct input→output links), `feed_forward = True`. With `num_hidden = 0` NEAT often stays linear for the whole run.
- `node_delete_prob = 0.05` (low). Direct links and extra nodes can still evolve.
- Code-level guarantee: `common.topology()` counts hidden nodes actually on an input→output path (`neat.graphs.feed_forward_layers`). Genomes with 0 in use get a fitness penalty (×0.5 if positive, −50 if negative). Only genomes with ≥1 in use can enter the hall of fame, `best/` or a milestone.
- neat-python 0.92 gotcha: `num_hidden > 0` gives every initial genome *different* hidden-node ids, so every genome becomes its own species. `train.unify_initial_hidden_keys` renumbers them to shared ids and re-speciates.
- Try recurrence only if phase 3 stalls.

**Speciation**
- Initial genome distance is ~0.55 and drifts to ~1.5, so the textbook `compatibility_threshold = 3.0` gives one species. Start at 0.6 and adapt it ×0.95/×1.05 per generation toward `--target-species 10`. The current value is stored in the checkpoint and logged in `stats.csv`.

**Network evaluation speed**
- `common.FastNet`: a numpy equivalent of `neat.nn.FeedForwardNetwork` that does one matmul per layer. Activate takes 9 µs instead of 108 µs, and a test checks it matches neat to 1e-9.

**Fitness**
- Fitness = mean raw env reward over **K=3 seeds** (K=5 late in training).
- Raw reward keeps scores directly comparable with PPO and TD3.

**Early termination** (the big speed win)
- End an episode when the hull's x position advances less than ~1.0 unit over 150 steps.
- Without it, NEAT's common local optimum (stand still or shuffle) burns the full 1600 steps per episode.
- Report both "steps simulated" and "steps saved".

**Starting NEAT config**
- `pop_size = 200`, `elitism = 2`, `survival_threshold = 0.2`
- `compatibility_threshold = 3.0`, `max_stagnation = 20`, `species_elitism = 2`
- `weight_mutate_rate = 0.8`, `weight_mutate_power = 0.5`
- `conn_add_prob = 0.3`, `node_add_prob = 0.1`
- `activation_default = tanh`

All of these are kept in the `.ini` and none are hard-coded.

**Each generation**
- Log per generation to `stats.csv`: best, mean, species count, node and connection counts, env steps, wall time.
- Every 10 generations: re-evaluate the top 5 genomes on 10 held-out seeds. The best mean goes to `best/`, and any milestone it crosses is saved.

**Checkpoints**
- Our own gzip pickle (`checkpoints/latest.pkl.gz`) every 10 generations, plus on Ctrl-C and at exit. It deliberately does **not** store `neat.Config`. `neat.Checkpointer` was not used: it pickles the config, which is exactly lesson #2.
- `--resume` rebuilds the Population from the current `.ini` and re-attaches it to the restored species set, which otherwise keeps the old `species_set_config`. It also resets the genome/node id counters, which neat restarts at 1 and would otherwise collide.

**Budget**
- `--hours` / `--generations` stop conditions, so runs fit "while I'm out" or overnight.

## Phases

**Phase 0: setup and benchmark (~20 min)**
- Create `.venv` and install pinned deps.
- Run `bench.py`: random-policy env steps/s on 1 core and on N workers, plus RSS per worker.
- Pick `--workers` (start at 6, try 7).
- Go if: at least 10k env steps/s in total and RAM stays above 1.5 GB free.

**Phase 1: smoke run (~5 min)**
- `pytest -m smoke`: pop 10, 2 generations, early stop on.
- Then a 10-generation run, to check that checkpoint, resume, evaluate and play work end to end.

**Phase 2: baseline run (~1–2 h)**
- Pop 200, K=3.
- Expect `first_positive` early (shuffling forward). The real question is whether it reaches 100.
- Write README.md and CLAUDE.md with the real commands and the measured speed.

**Phase 3: long run (overnight, ~8 h)**
- Resume from the baseline, or start fresh with tuned params.
- Targets: 100 (expected), 200 (likely with time), 300 (stretch; NEAT on BipedalWalker usually plateaus at 200–300).

**Phase 4: if it plateaus**
- Try these in order, one change per run, and log every run in `TRAINING_LOG.md` like the TD3 project did:
  1. Raise K to 5 and pop to 300.
  2. Action repeat of 2.
  3. Allow recurrent nets.
  4. Fitness shaping, e.g. a small bonus for hull stability.
- Report scores with raw reward only.

**Phase 5: analysis and comparison**
- Run `evaluate.py` with 100 episodes on the champion and on each milestone.
- Make `record_video.py` mp4s and add a NEAT panel to the comparison videos.
- Write a `MODEL_COMPARISON` section: NEAT vs PPO vs TD3 on score, env steps, wall clock and network size. NEAT's evolved topology is a nice thing to show; render it with `graphviz` if installed.
- Use the `failure-analysis-report` skill (as set out in `~/.claude/CLAUDE.md`) on the champion's failed episodes: why it falls, and which terrain or state comes before the fall.

## Verification

- `pytest -m smoke` passes. The tests cover:
  - The resume test: change `pop_size` in the ini, resume, and assert the new value is in use.
  - The deterministic-episode test: same genome + same seed gives the same reward.
- `python train.py --generations 3 --workers 6 --name smoke` creates `runs/smoke/` with `stats.csv`, a checkpoint and `best/`.
- `python train.py --resume --name smoke --generations 5` continues from generation 3, not 0.
- `python evaluate.py --run runs/<name> --episodes 100` gives a mean that matches the logged hall-of-fame score within its noise.
- `python play.py --run runs/<name>` renders, and `record_video.py` writes a 1600-step mp4 at 50 fps and 600×400.
- During long runs, check `free -h`: swap must not grow (lesson #5).
