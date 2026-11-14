# robocode, top down: a guide to the repository

robocode is the lab's code for AgenticGenPlan: a coding agent writes one Python policy for a robot task, and the frozen policy is evaluated. The `abstract-tree` branch adds my search-explanations analyzer, which measures how much such a policy plans by search and whether its moves follow those plans.

## Folders

| Folder | What it holds |
| --- | --- |
| `src/robocode/planning_explanations/search/` | The analyzer. The `policy_*.py` files analyze real agent policies; the other files are earlier milestones on toy tasks |
| `src/robocode/planning_explanations/` (top level, `stress/`) | The first prototype, on small Python programs; `stress/README.md` describes its test programs |
| `docs/abstract-tree-analyzer.md` | The analyzer's method, assumptions, metrics and current results |
| `tests/planning_explanations/` | The analyzer's 260 tests |
| `experiments/` | `run_experiment.py` runs an agent on a task; `policy_history.py` analyzes every version an agent committed and plots the changes |
| `src/robocode/` (other packages) | The shared AgenticGenPlan code: agents (`approaches/`), tasks (`environments/`), sandbox and agent tools (`utils/`, `mcp/`) |
| `third-party/` | Submodules, including the KinDER environments |

Inside `search/`, a policy passes through `policy_runner.py` (records every loop), `policy_discovery.py` (finds and freezes explanations), `policy_verify.py` and `policy_causality.py` (edit probes, plan use), and `policy_report.py` (metrics).

## Where to look first

1. **[`docs/abstract-tree-analyzer.md`](docs/abstract-tree-analyzer.md)**: what the analyzer does, on worked examples, with every assumption and metric and the current results.
2. **`search/policy_fixtures.py`**: small control policies (`PLAN_ONCE`, `UNUSED_PLAN`, `REACTIVE`) that show what the analyzer must tell apart.
3. **`RULES` at the top of `search/policy_report.py`**: the exact definition of every metric.
4. **`tests/planning_explanations/test_policy_loops.py`**: one small example per kind of search the analyzer must flag.

## Where the runs are

- **Agent runs** are not in git. About 1,000 development runs (29 KinDER environments, 5 seeds each, several models) sit in a shared Google Drive folder. `experiments/results_viewer.py` downloads and browses them. Each run holds the agent's log, every policy version it committed (`sandbox/.git`) and `results.json`.
- **Analysis results** are in `outputs/`, also not in git, one folder per analysis. Example: `outputs/search-policy-sol424-history-v12/` has `history.json`, the plots and `findings.md` for Sol's 70 versions.
- **One agent policy is in git** as a test case: GPT-5.6 Sol's ClutteredRetrieval2D policy in `search/policies/`.
