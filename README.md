# bipedal-neat

**Live page with the narrated explainer, chapter videos and progress clips:** https://jca374.github.io/bipedal-neat/

**Result (run `baseline`, 740 generations, ~12 h on a 4-core laptop):** the champion (gen 709) averages **+173 ± 63 over 100 unseen terrains** and falls in 9 of 100. It has 4 hidden nodes in one hidden layer and 30 connections. A fall analysis (`experiments/`, report on the page) showed that 90% of falls were forward pitch-overs after risky speed-ups. A training-only fall penalty then halved the fall rate (17% → 9%) without a significant change in score: paired difference −5.4, 95% CI [−26.5, +16.0]. For comparison, PPO reached 272 and TD3 reached 308–319 on the same laptop. Details are in `TRAINING_LOG.md`.

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

## Overnight queue
```bash
DEADLINE=05:00 TARGET=300 nohup ./overnight.sh > overnight.log 2>&1 &
```
The queue trains in 1 h chunks (4 or 6 workers depending on what else is running), evaluates on 100 terrains, writes the narration from the final numbers (`video_tools/narration.py`), voices it with the voice-text pipeline (Kokoro + Whisper check), renders `docs/video/` (explainer + 5 chapters), rebuilds the site and pushes.
To change settings for the next chunk without touching the running script, use `runs/<name>/overrides.json` (`{"k": 5, "pop_size": 300, "reason": "..."}`).

## Experiments (failure analysis and validation)
```bash
python experiments/record_episodes.py --run runs/baseline --episodes 300   # every step logged, seeds 3e7+
python experiments/analyze_falls.py results/episodes/baseline.jsonl.gz      # results/fall_analysis/baseline/report.html
python experiments/reselect.py --run runs/baseline                         # re-pick champion from the population (paired, bootstrap)
python experiments/compare.py runs/baseline/champion_v1 runs/baseline      # paired comparison on 100 new terrains
```

## Site
```bash
python make_site.py --run runs/baseline     # rebuilds docs/ (clips: median of 10 held-out terrains, 2x speed)
```
`docs/clips/progress.mp4` puts all the stage clips back to back, as raw material for a later video.
`record_video.py` borrows the label/score helpers from `../comparison_videos/` (sibling folder, not in this repo).
