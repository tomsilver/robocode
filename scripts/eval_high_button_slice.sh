#!/bin/bash
# Mirror test: score every policy on the high-button slice.
#
# The wall band asks whether a bottom-grasping policy can recover when a bottom
# grasp is impossible. This asks the mirror question: whether a side-grasping
# policy can recover when a single side grasp cannot lift the stick high enough.
# The slice is drawn from the STANDARD distribution but conditioned on holding a
# button above the side-grasp reach ceiling, so it has the statistical power the
# uniform suite lacks (only ~4 such instances per 100).
#
# Waits for the overnight driver to finish, which leaves the standard kinder
# source checked out -- the distribution the slice was built from.
set -uo pipefail

OUT_ROOT="outputs/reset_dist_a_2026-08-01"
SLICE="$OUT_ROOT/slices/high_button.json"
JULY_A="outputs/validation_fix_eval_count_leak_2026-07-23/full_20"
JULY_B="outputs/validation_fix_eval_count_leak_2026-07-24_new/seeds_24_424"
JOBS=5

log() { echo "[$(date '+%F %T')] $*"; }

log "waiting for the overnight driver to finish"
while ! grep -q "OVERNIGHT RUN FINISHED" "$OUT_ROOT/overnight.log" 2>/dev/null; do
    sleep 120
done
log "overnight driver finished"

run_one() {
    local cond=$1 label=$2 src=$3 dest=$4
    .venv/bin/python experiments/eval_policy_on_slice.py \
        --slice "$SLICE" --policy "$src" --cond "$cond" --out "$dest" \
        > "$dest.log" 2>&1 \
        && log "slice eval done $label" \
        || log "slice eval FAILED $label (see $dest.log)"
}

for cond in noprims lowlevel bilevel; do
    for seed in 42 24 424; do
        # Policies trained on the standard distribution.
        if [ "$seed" = "42" ]; then
            src="$JULY_A/$cond/stickbutton2d_generalized/s$seed"
        else
            src="$JULY_B/$cond/stickbutton2d_generalized/s$seed"
        fi
        dest="$OUT_ROOT/high_button_slice/standard_trained/$cond/s$seed"
        if [ -f "$src/sandbox/approach.py" ] && [ ! -f "$dest/results.json" ]; then
            mkdir -p "$(dirname "$dest")"
            while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
            run_one "$cond" "standard/$cond/s$seed" "$src" "$dest" &
        fi

        # Policies trained on the band distribution.
        src="$OUT_ROOT/$cond/s$seed"
        dest="$OUT_ROOT/high_button_slice/band_trained/$cond/s$seed"
        if [ -f "$src/sandbox/approach.py" ] && [ ! -f "$dest/results.json" ]; then
            mkdir -p "$(dirname "$dest")"
            while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
            run_one "$cond" "band/$cond/s$seed" "$src" "$dest" &
        fi
    done
done
wait

log "high-button slice results"
for f in $(find "$OUT_ROOT/high_button_slice" -name results.json | sort); do
    rate=$(.venv/bin/python -c "
import json; print(f\"{json.load(open('$f'))['solve_rate']:.2f}\")" 2>/dev/null)
    log "    $rate  ${f%/results.json}"
done
log "HIGH BUTTON SLICE FINISHED"
