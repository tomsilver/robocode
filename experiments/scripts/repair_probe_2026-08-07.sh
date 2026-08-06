#!/usr/bin/env bash
set -euo pipefail

# Retained-repair assay on the frozen 08-05 opus5 stickbutton s424 program.
# Three conditions per agent seed given (default: 42), identical config except
# the instance list handed to the agent: none / four mid-distribution / four
# wall-band instances. The standard-suite eval built into each run measures
# retention; edge-suite (band) scoring happens post-hoc with the
# results-viewer slice harness, never on the demo seeds.
#
# Run from the repair-probe checkout with the project environment active.
# Agent seeds come from the arguments (default: 42); add seeds in later
# invocations as earlier ones finish:
#
#   bash experiments/scripts/repair_probe_2026-08-07.sh       # seed 42
#   bash experiments/scripts/repair_probe_2026-08-07.sh 24    # later

python -c "import hydra" 2>/dev/null \
  || { echo "activate the project environment first" >&2; exit 1; }

SEEDS=("$@")
[[ ${#SEEDS[@]} -eq 0 ]] && SEEDS=(42)

ROOT=$(git rev-parse --show-toplevel)
PROBE="$ROOT/experiments/repair_probe"
STAMP=2026-08-07
BUDGET=20.0

COMMON=(--config-name noprims
  approach=agentic approach/backend=claude_opus5
  approach.container_backend=apptainer approach.blackbox=false
  environment=stickbutton2d_generalized
  num_eval_tasks=100 render_videos=false
  approach.seed_program="$PROBE/runB_approach.py")

mkdir -p "$ROOT/outputs"
{
  echo "repair probe $STAMP"
  echo "branch: $(git -C "$ROOT" branch --show-current) @ $(git -C "$ROOT" rev-parse HEAD)"
  echo "seed program: repair_probe/runB_approach.py (opus5 stickbutton s424, 08-05)"
  echo "conditions: none / mid / band; agent seeds: ${SEEDS[*]}; budget: $BUDGET"
} | tee "$ROOT/outputs/repair_probe_${STAMP}_manifest.txt"

fail=0
for seed in "${SEEDS[@]}"; do
  for cond in none mid band; do
    extra=()
    case "$cond" in
      mid)  extra=(approach.seed_instances="$PROBE/instances_mid.json") ;;
      band) extra=(approach.seed_instances="$PROBE/instances_band.json") ;;
    esac
    outdir="outputs/repair_probe_${STAMP}/${cond}/stickbutton2d_generalized/s${seed}"
    (cd "$ROOT" && python experiments/run_experiment.py "${COMMON[@]}" "${extra[@]:+${extra[@]}}" \
      seed="$seed" approach.max_budget_usd="$BUDGET" hydra.run.dir="$outdir") \
      || { echo "!! $cond s$seed failed"; fail=1; }
    test -f "$ROOT/$outdir/results.json" || { echo "!! $cond s$seed missing results"; fail=1; }
  done
done
echo "=== repair probe done (fail=$fail) ==="
exit "$fail"
