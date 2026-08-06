#!/usr/bin/env bash
set -euo pipefail

# First-pass overnight queue: one seed of everything, strictly in sequence.
#
#   smoke (strict, $2 capped)
#   -> strict s42 + twin s42 (the matched pair, in parallel)
#   -> repair probe none/mid/band at s42 (sequential)
#
# Run from the strict checkout (branch two-shelf-clutteredstorage) with the
# project environment active:
#
#   bash experiments/scripts/overnight_queue_2026-08-07.sh
#
# Further seeds are added only after these finish, by re-invoking the two
# campaign scripts with explicit seeds (e.g. `two_shelf_arms_2026-08-07.sh 24`).
# The repair probe runs from a sibling worktree on branch repair-probe,
# created here if absent.

STRICT_ROOT=$(git rev-parse --show-toplevel)
REPAIR_ROOT="${STRICT_ROOT}-repair"

echo "[$(date -Is)] === queue start ==="
bash "$STRICT_ROOT/experiments/scripts/two_shelf_arms_2026-08-07.sh" 42

if [[ ! -d "$REPAIR_ROOT" ]]; then
  git -C "$STRICT_ROOT" fetch origin repair-probe:repair-probe || true
  git -C "$STRICT_ROOT" worktree add "$REPAIR_ROOT" repair-probe
  git -C "$REPAIR_ROOT" submodule update --init third-party/kindergarden
  ln -sf "$STRICT_ROOT/robocode-sandbox.sif" "$REPAIR_ROOT/robocode-sandbox.sif"
fi
[[ $(git -C "$REPAIR_ROOT" branch --show-current) == repair-probe ]] \
  || { echo "repair worktree is not on repair-probe" >&2; exit 1; }
[[ -e "$REPAIR_ROOT/robocode-sandbox.sif" ]] \
  || { echo "missing sif in $REPAIR_ROOT" >&2; exit 1; }
[[ -f "$REPAIR_ROOT/third-party/kindergarden/src/kinder/core.py" ]] \
  || { echo "kindergarden submodule not checked out in $REPAIR_ROOT" >&2; exit 1; }

echo "[$(date -Is)] === repair probe starting ==="
(cd "$REPAIR_ROOT" && bash experiments/scripts/repair_probe_2026-08-07.sh 42)

echo "[$(date -Is)] === first pass done: strict/twin s42 + repair none/mid/band s42 ==="
