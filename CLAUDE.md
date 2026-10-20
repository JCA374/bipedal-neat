# bipedal-neat: notes for Claude

NEAT on BipedalWalker-v3, as a baseline against `../bipedal-claude` (TD3) and `../bipedal-claude-ppo`. `PLAN.md` has the design; `README.md` has the commands. Use `.venv/bin/python` (numpy<2, gymnasium 0.29.1, neat-python 0.92).

## Hard requirement
The champion must have **≥1 hidden layer**:
- `num_hidden = 8`, `full_nodirect` in the ini.
- `common.topology()["hidden_in_use"]`.
- Penalty in `train.fitness_fn`.
- Filter in `train.hall_of_fame`.

Don't set `num_hidden = 0` or `full_direct` without replacing those guards.

## Gotchas (all fixed; don't regress)
- neat-python 0.92 gives each new genome its own hidden-node ids, so every genome becomes its own species. `train.unify_initial_hidden_keys` fixes this; it is tested.
- The textbook `compatibility_threshold = 3.0` gives one species here (distances ~0.5–1.5). It is adaptive via `--target-species`.
- Resume never uses a pickled config. The restored species set and the id counters are re-pointed at the current config (`train.restore_population`). This is tested by the smoke test.
- `common.FastNet` replaces `neat.nn.FeedForwardNetwork` (12× faster activate). It must stay numerically equal; this is tested.
- Early stop (Δx < 1 over 150 steps) is training-only. The HOF, `evaluate.py`, `play.py` and videos all run full episodes with raw reward.
- Seeds: training < 1e6, hall of fame 1e7+, `evaluate.py` 2e7+.

## Machine
- i5-8250U 4C/8T, ~4 GB free RAM, no GPU. ~50 MB per worker.
- Other projects (e.g. `../../lunarlander/lunarlander-neat-wind`) run NEAT jobs on all 8 threads. Check `uptime` before benchmarking. Use `--workers 4` when sharing, 6 when idle.
- Run `free -h` during long runs; swap must not grow.
