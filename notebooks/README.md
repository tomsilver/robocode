# Paper results tables

`paper_results.ipynb` contains the table-generation portion of the notebook used
for the ICRA draft. `results_lib.py` loads saved run results, selects replicates,
and formats Tables I and II. Notebook outputs are cleared in Git.

## Draft status

This publishes the existing local generator, which is behind the current
manuscript. It still selects Kinematic3D Shelf rather than BaseMotion, omits the
One-shot row, and does not reproduce the current GenPlan cells with the original
local input roots. Those cohort/method updates and the helper's existing
Pandas typing/lint issues remain to be resolved before treating this as an exact
reproduction of the latest paper. Do not overwrite manuscript tables with this
version without reviewing the comparison cell.

Validation on 625 selected local runs confirmed byte-identical Table I and II
LaTeX between this PR and the original local generator. Both differ from the
current manuscript tables for the reasons above.

## Run

From the repository root:

```bash
cd notebooks
uvx --from jupyterlab --with pandas --with numpy jupyter lab paper_results.ipynb
```

Edit the configuration cell to point `ROOTS` at your downloaded result directories
and `PDDLSTREAM_REEVALUATION` at the audited planner reevaluation export, then run
the cells. The example paths under `../paper_data/` are placeholders, not included
datasets. No Docker, simulator, LLM calls or new evaluations are needed.

## Required inputs and reproducibility

For the manuscript snapshot, use the same curated final-run archives and GenPlan
results as the original notebook. Preserve the experiment-ID/timestamp/replicate
directory hierarchy containing each `results.json`. Arbitrary local rerun folder
names are not a substitute for the experiment IDs the loader parses.

The notebook retains the original precedence, exclusion and reevaluation rules:

- Earlier roots take priority over later roots for duplicate experiment/replicate
  identities. The configuration's root names are used by the exclusion rules.
- Superseded September 3–7 Dynamic3D batches are removed before deduplication.
- The failed PR2Blocked synthesis attempt is explicitly excluded, retaining its
  reason next to the identifier.
- The audited September 14 PDDLStream reevaluation replaces paired outcomes and
  timings only after source hashes and all 100 episode identities match. Supply
  its `manifest.json`, `full/summary.json`, and
  `full/<environment>-<replicate>/episodes.jsonl` files. The input files are not
  modified. Missing audit data raises an error rather than silently changing the
  reported results.

These result archives and the audited export are external inputs and are not
bundled in this PR. Input paths must be configured before execution. Loading a
new collection of results changes the table values; this is a generator, not a
frozen release of the paper's data. The ongoing network-restricted reruns are not
automatically substituted into the original cohort.

Optional Drive sync uses `DRIVE_FOLDER_ID` and an existing `robocode-drive` rclone
remote. It is off by default; extracted local archives are sufficient.

## Outputs

The notebook writes:

- `out/table1_paper.tex`: Table I's complete LaTeX block.
- `out/table2_paper.tex`: Table II's complete LaTeX block.

Tables report the mean and [minimum, maximum] across five finished replicates.
Cells with fewer than five remain blank. Bold marks the best mean and best
maximum per environment among the main-setting methods; the source-access row
is not included in that comparison. The final diagnostic cell displays partial
cells separately.

Set `OVERLEAF` to a local paper checkout to compare against
`tables/table-main.tex` and `tables/table-pending.tex`. Writes require explicitly
setting `WRITE_OVERLEAF = True`; the default only compares.

Figures and synthesis-process notebooks are outside this table-focused PR.
