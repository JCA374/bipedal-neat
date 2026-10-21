#!/usr/bin/env bash
# Overnight queue: finish training, evaluate, voice + render the explainer videos,
# rebuild the site and publish. Safe to re-run: every step picks up where it is.
#
#   nohup ./overnight.sh > overnight.log 2>&1 &
#
# Env knobs: DEADLINE (HH:MM, stop training then; default 05:00), TARGET (default 300),
#            RUN (default baseline).
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
V=~/.claude/skills/voice-text/scripts/voice.sh
RUN=${RUN:-baseline}
RUN_DIR=runs/$RUN
TARGET=${TARGET:-300}
DEADLINE=${DEADLINE:-05:00}
deadline_ts=$(date -d "$DEADLINE" +%s)
[ "$deadline_ts" -lt "$(date +%s)" ] && deadline_ts=$(date -d "tomorrow $DEADLINE" +%s)

log() { echo "[$(date '+%H:%M:%S')] $*"; }
best_score() { $PY -c "import json;print(json.load(open('$RUN_DIR/best/meta.json'))['mean'])" 2>/dev/null || echo -999; }
# Another project's training (e.g. lunarlander-neat-wind) sharing the CPU?
other_neat_running() { pgrep -af "python.*train[.]py" | grep -v -- "--name $RUN" | grep -q .; }

# ---------------------------------------------------------------- 1. training
log "waiting for any running '$RUN' training to finish"
while pgrep -f "train[.]py --name $RUN" >/dev/null; do sleep 30; done
log "champion so far: $(best_score)"

while :; do
  now=$(date +%s); left=$(( deadline_ts - now ))
  best=$(best_score)
  if $PY -c "import sys; sys.exit(0 if float('$best') >= $TARGET else 1)"; then
    log "target $TARGET reached (champion $best), training done"; break
  fi
  if [ "$left" -lt 900 ]; then log "deadline $DEADLINE reached (champion $best)"; break; fi
  if other_neat_running; then W=4; else W=6; fi
  hours=$($PY -c "print(round(min(1.0, $left/3600 - 0.05), 2))")
  log "training chunk: ${hours} h, $W workers (champion $best)"
  $PY -u train.py --name "$RUN" --resume --hours "$hours" --workers "$W" --stop-at "$TARGET" >> runs_${RUN}.log 2>&1
  rc=$?
  [ $rc -ne 0 ] && { log "train.py exited $rc, see runs_${RUN}.log"; tail -5 runs_${RUN}.log; break; }
done

# ---------------------------------------------------------------- 2. evaluation
log "evaluating champion and milestones on 100 unseen terrains"
if other_neat_running; then W=4; else W=6; fi
$PY evaluate.py --run "$RUN_DIR" --episodes 100 --workers $W --out "$RUN_DIR/eval/best.json" | tail -20
for m in "$RUN_DIR"/milestones/*/; do
  name=$(basename "$m")
  $PY evaluate.py --run "$m" --episodes 100 --workers $W --out "$RUN_DIR/eval/milestone_$name.json" > /dev/null \
    && log "  milestone $name: $($PY -c "import json;d=json.load(open('$RUN_DIR/eval/milestone_$name.json'));print(d['mean'], '+-', d['std'], 'falls', d['fell_frac'])")"
done

# ---------------------------------------------------------------- 3. narration + voice
log "building narration from the final numbers"
$PY video_tools/narration.py "$RUN_DIR" 2>/dev/null > video_tools/narration/explainer.md
nsec=$(grep -c '^## ' video_tools/narration/explainer.md)
log "  $nsec narration sections"
voice_ok=0
if [ "$nsec" -eq 38 ] && \
   $V new video_tools/narration/explainer.md --kind voiceover --slug explainer --force && \
   $V prep voice/explainer && $V narrate voice/explainer; then
  $V scan voice/explainer > voice/scan.txt 2>&1
  $V listen voice/explainer > voice/listen.txt 2>&1 || true     # flags are for review, not fatal
  if $V export voice/explainer; then
    voice_ok=1
    $V status voice/explainer 2>&1 | tail -15
    log "  listen-check report: voice/listen.txt"; tail -20 voice/listen.txt
  fi
fi
[ $voice_ok -eq 1 ] || log "voice pipeline failed: videos will render silent"

# ---------------------------------------------------------------- 4. videos + site
log "rendering explainer and chapter videos"
$PY video_tools/make_explainer.py "$RUN_DIR" 2>&1 | grep -v pkg_res | tail -60
log "rebuilding site"
$PY make_site.py --run "$RUN_DIR" 2>&1 | grep -v pkg_res | tail -5

# ---------------------------------------------------------------- 5. publish
log "tests"
$PY -m pytest -q 2>&1 | tail -2
log "publishing"
git add -A
git commit -q -m "Overnight: training to generation $(tail -1 $RUN_DIR/stats.csv | cut -d, -f1), evaluation, narrated explainer and chapter videos

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin main && log "pushed" || log "nothing pushed"
for i in $(seq 1 30); do
  s=$(gh api repos/JCA374/bipedal-neat/pages/builds/latest --jq .status 2>/dev/null); [ "$s" = built ] && break; sleep 10
done
log "pages build: $s  ->  https://jca374.github.io/bipedal-neat/"
log "QUEUE DONE"
