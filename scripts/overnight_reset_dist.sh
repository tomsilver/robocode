#!/bin/bash
# Unattended overnight driver for the StickButton2D reset-distribution experiment.
#
# Phase 1  synthesis: the remaining (condition, seed) cells on the band distribution,
#          capped at MAX_SYNTHESIS concurrent runs so the agent rate limit holds.
# Phase 2  cross-evaluation: every band-trained policy re-scored on the standard
#          distribution, at the same pinned eval_seed.
#
# The two phases need different kinder sources, so they are strictly serialized: the
# checkout only ever changes while no run is starting up. A synthesis run snapshots
# its source into a temp mount at startup and holds its config in memory, so a later
# checkout cannot affect a run already under way.
#
# Launch detached so it outlives an ssh session:
#   setsid nohup bash scripts/overnight_reset_dist.sh > overnight.log 2>&1 < /dev/null &
set -uo pipefail

OUT_ROOT="outputs/reset_dist_a_2026-08-01"
KG="third-party/kindergarden"
BAND_REF="exp/sb-reset-a"
STANDARD_REF="f646ff8"
EVAL_SEED=1000
BUDGET=20.0
MAX_SYNTHESIS=3
MAX_EVAL_JOBS=5
SEEDS=(24 424)
CONDS=(noprims lowlevel bilevel)

log() { echo "[$(date '+%F %T')] $*"; }

# Top-level synthesis runs only: forked eval children share the parent's cmdline, and
# load_dir runs are evaluations rather than synthesis.
count_synthesis() {
    local n=0 pid ppid pcmd cmd
    while read -r pid ppid cmd; do
        case "$cmd" in *load_dir*) continue ;; esac
        pcmd=$(ps -o cmd= -p "$ppid" 2>/dev/null)
        case "$pcmd" in *run_experiment.py*) continue ;; esac
        n=$((n + 1))
    done < <(ps -eo pid,ppid,cmd | grep "[r]un_experiment.py")
    echo "$n"
}

stick_lower_bound() {
    .venv/bin/python -c "
from kinder.envs.kinematic2d.stickbutton2d import StickButton2DEnvConfig as C
print(f'{C().stick_init_pose_bounds[0].x:.4f}')" 2>/dev/null
}

require_bounds() {
    local want=$1 got
    got=$(stick_lower_bound)
    if [ "$got" != "$want" ]; then
        log "FATAL: stick lower bound is '$got', expected '$want'"
        exit 1
    fi
    log "verified stick lower bound = $got"
}

# ---------------------------------------------------------------- phase 0: settle
log "waiting for any in-flight evaluations to finish"
while [ "$(ps -eo pid,ppid,cmd | grep "[r]un_experiment.py" | grep -c load_dir)" -gt 0 ]; do
    sleep 30
done
log "no evaluations running"

# ---------------------------------------------------------------- phase 1: synthesis
log "switching kinder to the band distribution ($BAND_REF)"
git -C "$KG" checkout --quiet "$BAND_REF" || { log "FATAL: checkout failed"; exit 1; }
require_bounds "3.4350"
git -C "$KG" rev-parse HEAD > "$OUT_ROOT/kindergarden_sha.txt"

for seed in "${SEEDS[@]}"; do
    for cond in "${CONDS[@]}"; do
        run_dir="$OUT_ROOT/$cond/s$seed"
        if [ -f "$run_dir/results.json" ]; then
            log "skip $cond s$seed (already has results.json)"
            continue
        fi
        # A restart must never relaunch a cell that is still running: the duplicate
        # would write into the same directory and clobber hours of work.
        if ps -eo cmd | grep -q "[h]ydra.run.dir=$run_dir\$"; then
            log "skip $cond s$seed (already running)"
            continue
        fi
        while [ "$(count_synthesis)" -ge "$MAX_SYNTHESIS" ]; do sleep 60; done
        # Re-check just before launching: a run must never start on the wrong source.
        require_bounds "3.4350"
        mkdir -p "$run_dir"
        log "launching synthesis $cond s$seed"
        nohup .venv/bin/python experiments/run_experiment.py \
            --config-name "$cond" \
            approach=agentic \
            approach/backend=claude_sonnet \
            approach.container_backend=apptainer \
            approach.max_budget_usd="$BUDGET" \
            approach.blackbox=false \
            num_eval_tasks=100 \
            seed="$seed" \
            eval_seed="$EVAL_SEED" \
            environment=stickbutton2d_generalized \
            hydra.run.dir="$run_dir" \
            > "$run_dir/launch.log" 2>&1 &
        # Let the run snapshot its source before anything else can touch the tree.
        sleep 120
    done
done

log "all synthesis cells launched; waiting for them to finish"
while [ "$(count_synthesis)" -gt 0 ]; do sleep 60; done
log "synthesis complete"

# ---------------------------------------------------------------- phase 2: cross-eval
log "switching kinder to the standard distribution ($STANDARD_REF)"
git -C "$KG" checkout --quiet "$STANDARD_REF" || { log "FATAL: checkout failed"; exit 1; }
require_bounds "0.0000"

eval_one() {
    local cond=$1 seed=$2 src=$3 dest=$4
    mkdir -p "$dest"
    .venv/bin/python experiments/run_experiment.py \
        --config-name "$cond" \
        approach=agentic \
        approach/backend=claude_sonnet \
        approach.container_backend=apptainer \
        approach.load_dir="$src" \
        approach.blackbox=false \
        num_eval_tasks=100 \
        seed="$seed" \
        eval_seed="$EVAL_SEED" \
        environment=stickbutton2d_generalized \
        hydra.run.dir="$dest" \
        > "$dest/eval.log" 2>&1 \
        && log "eval done $cond s$seed" \
        || log "eval FAILED $cond s$seed"
}

for cond in "${CONDS[@]}"; do
    for seed in 42 24 424; do
        src="$OUT_ROOT/$cond/s$seed"
        [ -f "$src/sandbox/approach.py" ] || { log "no policy for $cond s$seed"; continue; }
        dest="$OUT_ROOT/band_trained_on_standard_suite/$cond/s$seed"
        [ -f "$dest/results.json" ] && { log "skip eval $cond s$seed"; continue; }
        while [ "$(jobs -rp | wc -l)" -ge "$MAX_EVAL_JOBS" ]; do wait -n; done
        eval_one "$cond" "$seed" "$src" "$dest" &
    done
done
wait
log "cross-evaluation complete"

# ---------------------------------------------------------------- phase 3: summary
log "summary"
.venv/bin/python experiments/analyze_reset_dist.py "$OUT_ROOT" 2>&1 | sed 's/^/    /'
log "OVERNIGHT RUN FINISHED"
