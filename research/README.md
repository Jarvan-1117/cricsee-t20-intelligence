# T20 win probability and Net Win Contribution

This checkout includes selected result tables and figures rather than the complete
data and model cache. Read [Publication and reproduction scope](PUBLICATION.md)
before executing notebooks. Run research commands from this directory. The
prototype application is developed separately on the `dev-prototype` branch.

Research question: how can player quality and dynamic availability define
remaining resources, and do those resources improve ball-by-ball T20I win
prediction and calibration? NWC is the downstream contribution analysis.

## Current workflow

### Python environment

Use this project's `.venv/Scripts/python.exe` (Python 3.10.6 on Windows).
`requirements-lock.txt` records the exact package versions; `requirements.txt`
lists the direct runtime dependencies. To recreate the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name t20-project-3 --display-name "Python (T20 Project 3.0)"
```

Open this folder in VS Code and select **Python (T20 Project 3.0)** as the notebook
kernel. Notebook 01–10 metadata names that kernel. An already-running kernel does
not switch automatically: restart/select the kernel and confirm
`sys.executable` points to this checkout's `research\\.venv\\Scripts\\python.exe`.
Do not save an older editor buffer over externally updated notebook metadata.

Run checks with `.\.venv\Scripts\python.exe -m pip check` and
`.\.venv\Scripts\python.exe -B -m unittest discover -s tests`.
Recreating the environment does not require retraining or rewriting saved results.
See [Jupyter kernel registration](https://ipython.readthedocs.io/en/stable/install/kernel_install.html)
and [VS Code notebook kernel selection](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).

### Experiment and product stages

The implemented sequence is **01-06 -> 07 -> 08 -> 09 -> 10**. Stage 08 selects
four models across feature groups using validation Log Loss. Stage 09 uses raw prediction streams to calculate
partial operational NWC. Stage 10 exports verified player-match histories with
traditional statistics and partial NWC for the current test cohort. The
prototype API serves this package to its player-history pages. Complete
historical-player coverage is outside this test-cohort export.

Execution status: Stage 08 has completed the cross-group selection, training/
validation diagnostics, test evaluation and permutation importance. Its verified
raw probability handoff covers 228 test matches and 53,850 events. Stage 09
consumes these four cross-group-selected streams. Stage 10 publishes the
corresponding 542-player / 5,016-player-match package. The local API and
player-history pages use this verified package. Each API process pins one
snapshot; restart it after a subsequent export to activate that snapshot.

Use [Current workflow and methods](docs/current_workflow.md) for current
instructions, selection rules, output paths and interpretation limits.

| Notebook | Responsibility |
| --- | --- |
| 01 | Source audit, cleaning and standard-match cohort |
| 02 | Pre-event game state and prior-date team background |
| 03 | Separate prior-date batting and bowling quality metrics |
| 04 | Player availability and nominal individual bowling quotas |
| 05 | Waiting-batter sums and capacity-weighted bowling sums |
| 06 | Five controlled groups, four candidate algorithm families |
| 07 | RBW versus batting, bowling and combined progress interactions |
| 08 | Validation-selected feature group, algorithm and parameters; test evaluation and importance |
| 09 | Partial NWC, reconciliation and terminal-case audit |
| 10 | Chronological batting/bowling histories and verified product data export |

For export instructions and field definitions, see
[Player match history export](docs/player_history_export.md).

## Data and resources

The study covers men's and women's T20 internationals between Australia,
England, India, New Zealand, Pakistan, South Africa, Sri Lanka, West Indies,
Bangladesh, Ireland and Zimbabwe. Main exclusions include adjusted-target and
reduced-over matches, Super Overs and unsupported outcomes. Normal successful
chases and all-out endings need not reach 120 legal balls. Source event order
is checked against JSON; wides and no-balls remain events.

Stages 01-03 retain 1,627 matches / 385,786 events. Stage 04 identifies 1,521
resource-ready matches / 360,721 events; 106 review-required matches remain
separately recorded. All experimental groups use the same resource-ready cohort.

- Both innings predict first-batting-team victory using pre-event inputs.
- Team/player history uses strictly earlier dates with same-date batching.
- Batting quality: strike rate, average, boundary rate and dot rate.
- Bowling quality: economy, wicket rate and dot rate.
- Batting resources sum each metric over available players who have never
  entered the innings. Exclude the current pair and previously entered players.
- Bowling resources sum remaining nominal balls / 6 times each bowling metric,
  including the current bowler's remaining capacity.
- No weighted mean, historical-use allocation or relative-quality centering.
- Direct striker/non-striker/bowler quality and evidence are excluded from model
  inputs. Current-player diagnostics in 03 are not direct predictors.
- Individual nominal capacities may sum above the innings budget; these are
  upper bounds, not a forecast of the captain's actual allocation.

## Experiments and final model selection

06 compares R0 (constant), R1 (state/background), RB (+batting resources),
RW (+bowling resources) and RBW (+both). 07 reuses RBW and adds IB (batting
resource/progress products), IW (bowling products) and IBW (all seven products).
Products use nominal pre-event remaining balls / 120. Original resources remain.

Both stages use Logistic Regression, Random Forest, HistGradientBoosting and
XGBoost, with three candidate settings per family. 06 completed 196 fits,
including four constants, retaining 68 family winners and 20 group/track winners.
07 completed 144 additional fits (48 per added arm).

Whole-match/date splits contain 1,065 training, 228 validation and 228 test
matches: 252,670 / 54,201 / 53,850 events. Models/preprocessing fit training
data only; no train-plus-validation refit. First innings omit chase-only fields.

**08 selects the feature group, algorithm and parameters separately for each
track by validation event Log Loss.** Combine the eight within-group winners
from 06 + 07, deduplicate RBW, then choose the lowest validation score. Exact
cross-group ties follow R0/R1/RB/RW/RBW/IB/IW/IBW order. Test scores do not enter
selection. The eight-arm test table is an exploratory comparison, not reselection.

| Track | Selected group | Algorithm |
| --- | --- | --- |
| W1 | RW | XGBoost |
| W2 | RBW | HistGradientBoosting |
| M1 | IBW | Logistic Regression |
| M2 | IBW | Random Forest |

08 reports Log Loss, Brier, accuracy, F1, AUC, calibration and paired whole-match
uncertainty. Ten-repeat permutation importance covers actual inputs and feature
blocks of the four selected models. Importance describes reliance, not causal
effects or proof that adding a resource improves predictions.

Section 08.6 also reports per-transformed-feature XGBoost Gain for W1 and
Random Forest normalised impurity importance (MDI) for M2, under
`native_importance/`. W2 has no supported public native Gain interface and
Gain is not applicable to M1's linear model. The native metrics supplement,
rather than replace, permutation importance; their numeric scales differ.

## NWC and limits

09 uses the four verified validation-selected raw 08 streams. It does not fit or select
models, or generate new predictions. Within-innings changes are next pre-event minus current pre-event
probability. The final match event closes to the audited outcome without
clipping, smoothing or redistribution. The first-innings last event and
innings transition form an unallocated boundary bundle.

Ordinary no-extra/no-exit and validated bowled/LBW intervals use the current
striker/bowler opposite-sign proxy. Other exits, extras and shared responsibility
remain residuals. Role changes are flagged, not causally isolated. Missing
player allocation is not zero contribution.

Current coverage: 228 test matches / 53,850 events; 48,469 allocated and
5,381 residual events (90.01%). The terminal audit inspects every case
at or above 25 percentage points against source and accounting checks.
These checks do not certify prediction quality. Audit counts are derived
from the current ledger, not hardcoded.

Player totals are partial operational NWC, not complete causal player value,
an ability ranking or complete historical coverage. Removing direct quality
does not remove historical quality from resource context. Exploratory comparisons
and validation selection do not establish independent confirmation. Do not
repeatedly tune from test results or importance.

## Running and saved files

Use the existing project interpreter. Notebooks install no packages. False run
flags read verified results; True explicitly requests computation. Missing or
stale results never automatically trigger training. Text edits need no retraining.

08 uses `resource_sums_v2_joint_selection_v1`, with a frozen selection snapshot,
completed test evaluation, raw probability handoff and `development_diagnostics/`
train/validation tables. 09 uses `resource_sums_v2_joint_selection_v1` for its
ledger, reconciliation, player totals, figures and terminal audit.
07 uses `resource_sums_v2/four_way_interactions_v1`. 06 models are under
`outputs/06_model_training_and_comparison/models/resource_sums_v2/`.

This publication contains notebooks in `notebooks/`, reusable code in `src/`,
checks in `tests/`, experiment settings in `config/`, method documents in `docs/`
and selected reports and figures in `outputs/`. Private reference materials,
raw/processed data, candidate model files and complete runtime manifests are
not included. See `PUBLICATION.md` for inputs required to reproduce the workflow.

The method specifications are [Stage 07](docs/stage07_four_way_interactions_protocol.md),
[Stage 08](docs/stage08_joint_selection_protocol.md) and
[Stage 09](docs/stage09_joint_selection_protocol.md). Stage 08/09 result verification
checks the recorded hashes of their input protocols, code and artifacts.
Reload notebooks from disk after external edits instead of overwriting them
with older VS Code editor buffers.
