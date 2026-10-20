# bipedal-neat

**Live page with progress clips:** https://jca374.github.io/bipedal-neat/

NEAT (neuroevolution, gradient-free) on BipedalWalker-v3. It is a baseline to compare with the PPO (`../bipedal-claude-ppo`) and TD3 (`../bipedal-claude`) projects. See `PLAN.md` for the design and the lessons carried over from those projects.

**Every champion has at least one hidden layer.** The population starts as 24→8→4 (`full_nodirect`). Genomes with no hidden node on an input→output path are penalised, and they can never become `best/` or a milestone.

## Setup
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q            # 6 tests incl. smoke train + resume-applies-new-config
.venv/bin/python bench.py                # steps/s per worker count (check `uptime` first: shared machine)
```

## Train
```bash
source .venv/bin/activate
python train.py --name baseline --hours 2 --workers 4      # fresh run
python train.py --name baseline --resume --hours 8         # continue; the CURRENT ini is applied
```
Useful flags: `--generations N` (total), `--k 3` (seeds per genome), `--pop-size`, `--target-species 10`, `--hof-every 10`, `--no-early-stop`.

Output in `runs/<name>/`:
- `stats.csv`: one row per generation. Best/mean fitness, species, `best_hidden_in_use`, `best_depth`, env steps, steps saved by early stop, seconds, compat threshold, free RAM.
- `best/`: champion by **mean over 10 held-out seeds** (genome.pkl + config.ini + meta.json).
- `milestones/{first_positive,100,200,300}/`: the same files, saved when first crossed.
- `hof.json`: top re-evaluated genomes. `checkpoints/latest.pkl.gz`: the resume point.

## Use a trained genome
```bash
python evaluate.py --run runs/baseline --episodes 100       # mean/std + hidden_in_use, depth
python play.py --run runs/baseline                          # window
python record_video.py --run runs/baseline                  # 600x400, 50 fps, 1600 steps, seed 42
```

## Site
```bash
python make_site.py --run runs/baseline     # rebuilds docs/ (clips: median of 10 held-out terrains, 2x speed)
```
`docs/clips/progress.mp4` puts all the stage clips back to back, as raw material for a later video.
`record_video.py` borrows the label/score helpers from `../comparison_videos/` (sibling folder, not in this repo).
