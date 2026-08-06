#!/usr/bin/env bash
set -euo pipefail

# Two-shelf strict-vs-twin synthesis campaign.
#
# Run from the strict checkout (branch two-shelf-clutteredstorage) with the
# project environment active. Seeds come from the arguments (default: 42);
# add seeds in later invocations as earlier ones finish:
#
#   bash experiments/scripts/two_shelf_arms_2026-08-07.sh          # seed 42
#   bash experiments/scripts/two_shelf_arms_2026-08-07.sh 24 424   # later
#
# The twin arm launches from a sibling worktree on branch two-shelf-twin
# (created here if absent), so each agent reads source whose defaults match
# the sampler it observes. A cheap capped smoke run validates plumbing before
# the paid arms (skipped if its results already exist); arms run as
# strict/twin pairs on matched seeds.

SEEDS=("$@")
[[ ${#SEEDS[@]} -eq 0 ]] && SEEDS=(42)
BUDGET=20.0
STAMP=2026-08-07

python -c "import hydra" 2>/dev/null \
  || { echo "activate the project environment first" >&2; exit 1; }

STRICT_ROOT=$(git rev-parse --show-toplevel)
TWIN_ROOT="${STRICT_ROOT}-twin"

[[ $(git -C "$STRICT_ROOT" branch --show-current) == two-shelf-clutteredstorage ]] \
  || { echo "run from the two-shelf-clutteredstorage checkout" >&2; exit 1; }

if [[ ! -d "$TWIN_ROOT" ]]; then
  git -C "$STRICT_ROOT" fetch origin two-shelf-twin:two-shelf-twin || true
  git -C "$STRICT_ROOT" worktree add "$TWIN_ROOT" two-shelf-twin
  git -C "$TWIN_ROOT" submodule update --init third-party/kindergarden
  ln -sf "$STRICT_ROOT/robocode-sandbox.sif" "$TWIN_ROOT/robocode-sandbox.sif"
fi
[[ $(git -C "$TWIN_ROOT" branch --show-current) == two-shelf-twin ]] \
  || { echo "twin worktree is not on two-shelf-twin" >&2; exit 1; }

COMMON=(--config-name noprims
  approach=agentic approach/backend=claude_opus5
  approach.container_backend=apptainer approach.blackbox=false
  environment=two_shelf_clutteredstorage2d
  num_eval_tasks=100 render_videos=false)

run() { # root label seed budget
  local root=$1 label=$2 seed=$3 budget=$4
  local outdir="outputs/two_shelf_${label}_${STAMP}/noprims/two_shelf_clutteredstorage2d/s${seed}"
  (cd "$root" && python experiments/run_experiment.py "${COMMON[@]}" \
    seed="$seed" approach.max_budget_usd="$budget" hydra.run.dir="$outdir")
  test -f "$root/$outdir/results.json"
}

mkdir -p "$STRICT_ROOT/outputs"
{
  echo "two-shelf arms $STAMP"
  echo "strict: $(git -C "$STRICT_ROOT" rev-parse HEAD) (two-shelf-clutteredstorage)"
  echo "twin:   $(git -C "$TWIN_ROOT" rev-parse HEAD) (two-shelf-twin)"
  echo "seeds: ${SEEDS[*]}  budget: $BUDGET  eval_seed: unset (follows seed)"
  echo "smoke: strict seed 7, budget 2.0, num_eval_tasks 10"
} | tee "$STRICT_ROOT/outputs/two_shelf_${STAMP}_manifest.txt"

SMOKE_DIR="outputs/two_shelf_smoke_${STAMP}/noprims/two_shelf_clutteredstorage2d/s7"
if [[ -f "$STRICT_ROOT/$SMOKE_DIR/results.json" ]]; then
  echo "=== smoke already passed, skipping ==="
else
  echo "=== smoke (strict, capped) ==="
  (cd "$STRICT_ROOT" && python experiments/run_experiment.py "${COMMON[@]}" \
    seed=7 approach.max_budget_usd=2.0 num_eval_tasks=10 \
    hydra.run.dir="$SMOKE_DIR")
  test -f "$STRICT_ROOT/$SMOKE_DIR/results.json"
  echo "=== smoke passed, launching arms ==="
fi

fail=0
for seed in "${SEEDS[@]}"; do
  run "$STRICT_ROOT" strict "$seed" "$BUDGET" & p_strict=$!
  run "$TWIN_ROOT" twin "$seed" "$BUDGET" & p_twin=$!
  wait "$p_strict" || { echo "!! strict s$seed failed"; fail=1; }
  wait "$p_twin" || { echo "!! twin s$seed failed"; fail=1; }
done
echo "=== all arms done (fail=$fail) ==="
exit "$fail"
