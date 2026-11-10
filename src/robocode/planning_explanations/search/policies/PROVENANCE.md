# Milestone 8 policy

`clutteredretrieval2d_codex_gpt56sol.py.txt` is a byte-identical copy of an
agent-written policy. The `.txt` suffix keeps repository linters off generated
code; the analyzer reads it as Python source.

| Field | Value |
| --- | --- |
| Environment | KinDER ClutteredRetrieval2D, generalized over obstruction counts |
| Agent | Codex GPT-5.6 Sol, Blackbox, strict, no primitives, 60 s timeout |
| Run | `2026-09-07_20-32-39/replicate_424` in the robocode results-viewer cache |
| SHA-256 | `c799970f416b862aa3c31502908784459bd75cc2a75a36559ffdb394ef087551` |
| Length | 529 lines (median among the 143 cached policies that use `heappush`) |

`clutteredretrieval2d_env_config.json` is that run's `env_config.json`. The
policy's author designed for 1, 3, 5 and 10 obstructions; evaluation also uses
15, 20 and 25.
