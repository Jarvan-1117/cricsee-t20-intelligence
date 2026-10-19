# Current 01-10 workflow and methods

This guide describes the 01-10 experiment and product-export pipeline. Stage 08 selects four
models across feature groups by validation Log Loss; Stage 09 consumes raw prediction streams for partial
operational NWC. See [Stage 07](stage07_four_way_interactions_protocol.md),
[Stage 08](stage08_joint_selection_protocol.md) and
[Stage 09](stage09_joint_selection_protocol.md) for the method specifications.

Execution status: Stage 08 has completed cross-group selection, training/validation
diagnostics, test evaluation and permutation importance. The verified raw handoff
covers 228 test matches and 53,850 events. Stage 09 consumes this cross-group
handoff for its ledger, player totals and terminal diagnostics. Stage 10 exports
the corresponding 542-player / 5,016-player-match package, consumed by the local
API and player-history pages. The API pins one verified snapshot at startup.
A selection snapshot alone is not an evaluated prediction stream.

## 01-05: data, quality and resources

01 audits source events and exports the clean standard-match cohort. Pending
matches never enter the included cohort. 02 constructs pre-event state and
prior-date team background. Optional team identity and recent-10-event columns
remain inspectable but are not used in the current common model input block.
03 calculates separate smoothed player metrics using dates strictly before the
match, with same-date batching. Its current-actor quality/evidence fields are
diagnostics, not direct model inputs. 04 tracks availability and nominal quotas.
05 combines the shared player profiles and availability under `resource_sums_v2`.

Batting resources sum four separate metrics over available players who have
never entered the innings. Exclude both current batters, dismissed/previously
entered players and unavailable/replaced-out players. Entering as non-striker
removes a player immediately, without waiting for a faced ball.

Bowling resources sum remaining nominal balls / 6 times each of three bowling
metrics, including the current bowler's remaining capacity. Use the Stage 04
over-slot rules, not a blanket 24-minus-balls formula. No weighted average,
relative-quality centering, future workload allocation or innings-budget
normalization. Individual upper bounds can exceed the innings budget in total.
An empty pool produces zero sums. Keep different metric units separate.

No standalone striker, non-striker or bowler quality/evidence enters the model
allowlists. Historical quality still enters through the resource context.
Higher economy or batting dot rate is not universally better quality.

## 06: five controlled groups

Compare R0 (constant), R1 (state/background), RB (+batting), RW (+bowling)
and RBW (+both). All use 1,521 resource-ready matches / 360,721 events and
the same complete-date, whole-match chronological split within gender.
Training/validation/test counts are 1,065/228/228 matches and
252,670/54,201/53,850 events. Both innings target first-batting-team victory;
first innings omit chase-only fields.

The four algorithms are Logistic Regression, Random Forest, HistGradientBoosting
and XGBoost, with three settings each. XGBoost uses 200 CPU histogram trees and
learning-rate/depth pairs (0.05, 3), (0.1, 3), (0.05, 6), without early stopping.
Preprocessing and fitting use training data only; no train-plus-validation refit.
Total fits: 196 including four constants. There are 68 family winners and 20
group/track winners. Validation event Log Loss selects algorithms/parameters;
other metrics are supplementary.

`selected_models.csv` contains all 20 group/track winners. The four RBW entries
in `selected_systems.csv` are resource controls reused by 07. Its fixed-RBW
policy is an internal Stage 06 contract, not the final Stage 08 selection rule.
The best-results table across five groups is a Stage 06 summary, not joint selection.

## 07: symmetric resource/progress interactions

Compare RBW, IB, IW and IBW. Added arms retain all RBW inputs and add four
batting products, three bowling products or all seven products, respectively.
Each equals its resource metric times `scheduled_balls_remaining_estimate / 120`.
This uses nominal pre-event progress, never future realized innings duration.

Inherit the 06 cohort, split, four algorithms, grids and seed. Each added arm
fits 48 candidates (four tracks x four algorithms x three settings): **144 new
fits**. RBW is reused, not refitted. Report parameter-matched, within-family
and selected-system comparisons separately. Matching hyperparameters does not
freeze learned coefficients, trees or preprocessing.

07 reports an algorithm/parameter winner for each arm and track, plus a
supplementary best-across-arms table with exact ties resolved in RBW/IB/IW/IBW
order. All 16 within-arm winners feed 08; the across-arms table does not select
the final feature group. 07 never scores test data or computes NWC. Small
validation differences do not establish significance.

## 08: validation-selected final models and eight-arm evaluation

For each of W1/W2/M1/M2, select the feature group, algorithm and parameters
by validation event Log Loss. Verify the complete 06/07 grid and its within-group
winners, combine R0/R1/RB/RW/RBW/IB/IW/IBW and deduplicate RBW. The lowest
validation score wins; exact cross-group ties follow that group order. Freeze
all 32 comparison models and four final selections before reading test values.
The test table cannot change the selected models.

Current Stage 08 selections are W1 RW + XGBoost; W2 RBW + HistGradientBoosting;
M1 IBW + Logistic Regression; M2 IBW + Random Forest.

Section 08.1 reports training and validation event-weighted Log Loss and Brier
for all 32 within-group winners, highlighting the four final selections. The
same saved pipeline scores both splits; preprocessing is not fitted again.
Both gaps are validation minus training. A gap is diagnostic evidence, not an
automatic overfitting verdict: chronological population changes can also affect
it. Training scores and gaps never enter selection, and test events are not scored.

`probability_development_diagnostics.py --score` computes this diagnostic package
without executing test evaluation. Results are under the Stage 08 report folder
in `development_diagnostics/`: `train_validation_metrics.csv`,
`preferred_train_validation_metrics.csv`, protocol, summary and hash manifest.
`RUN_DEVELOPMENT_DIAGNOSTICS = False` reads verified diagnostics in the notebook;
True explicitly rescores the saved models on training and validation data only.

All groups use identical test events. Report event/match Log Loss and Brier,
accuracy, F1, AUC and ten-bin calibration diagnostics. Accuracy/F1 threshold is
0.5. No fitting, refitting, calibrator or terminal 0/1 override. Use 2,000 paired
whole-match bootstrap draws within gender. Pointwise 95% intervals omit selection
uncertainty and are not multiplicity-adjusted; temporal/team dependence remains.

Ten-repeat permutation importance covers actual inputs and nonempty
state/resource/interaction blocks of the four selected models. Products are
shuffled as separate columns, not reconstructed. Correlation and inconsistent
shuffled states limit interpretation. Importance describes reliance, not causal
effects; shuffle SD is not a sampling confidence interval. Keep negative values.
Do not retune from test results or importance. The comparisons are exploratory,
not independent confirmation.

Native importance applicability for the selected models: W1 XGBoost supports
Gain (mean gain across splits using a feature); W2 HistGradientBoosting has no
public `feature_importances_` in the installed implementation; M1 Logistic
Regression has coefficients rather than tree Gain; M2 Random Forest exposes
impurity-based importance, not XGBoost Gain. The common report uses permutation
importance. The native supplement exports W1 mean split Gain and M2 normalised
MDI under `native_importance/`, with explicit status rows for W2 and M1.
Each transformed feature maps to its raw input; phase one-hot indicators are
separate and their mean gains are not summed. Removed constants have missing
scores; unused retained XGBoost features have zero Gain. These measures must
not be merged into a common Gain column. `RUN_NATIVE_IMPORTANCE = False` reads
verified results; True extracts from saved models without fitting or prediction.
See the
[XGBoost API](https://xgboost.readthedocs.io/en/release_3.0.0/python/python_api.html#xgboost.Booster.get_score),
[Random Forest API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
and [HistGradientBoosting API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).

## 09: partial NWC and terminal checks

09 consumes verified Stage 08 predictions from W1 RW / XGBoost,
W2 RBW / HistGradientBoosting, M1 IBW / Logistic Regression and M2 IBW /
Random Forest. The input checks require the cross-group validation-selection
contract and exactly one frozen model per track. Its consumer is
`src/nwc_attribution_joint.py`, with shared accounting functions in
`src/nwc_attribution.py`.

Use only the verified four raw Stage 08 prediction streams, one row per event.
No training, reselection or prediction calls. Keep the first-batting-team
perspective and innings-first-event coverage. Within-innings changes are next
pre-event minus current pre-event probability, including intervening observed
state/personnel changes rather than a fully isolated causal ball effect.

Keep the first-innings final event plus innings transition in one unallocated
boundary bundle. Close the final match event to the audited outcome (0/1).
Never clip, delete, smooth or redistribute large terminal changes.

Ordinary no-extra/no-exit and validated bowled/LBW intervals use the current
striker/bowler opposite-sign proxy. Other exits, extras, penalties and shared
responsibility remain residuals. Role changes are flagged, not causally isolated.
Missing player allocation is not zero. Reconcile allocations plus residuals.

The terminal audit selects all cases with absolute terminal WPA at least 25
percentage points from the current ledger. Last-five-event views are context,
not smoothing. The current selected-case count and source/accounting outcomes
are reported dynamically in notebook 09 and its versioned terminal audit.
Passing these checks does not certify probabilities or causal attribution.

Current scope: 228 test matches / 53,850 events, not complete player histories.
48,469 events have player allocations and 5,381 are residuals. Partial operational
NWC is not a full ability ranking. Stage 10 exports traditional statistics and
partial contributions for this cohort. The prototype displays chronological
batting and bowling histories through its read-only API; full-career coverage
is not provided by this test-cohort package.

## Active locations

| Stage | Reports | Processed data |
| --- | --- | --- |
| 01 | `outputs/01_data_audit_and_cleaning` | `data/processed/matches` |
| 02 | `outputs/02_game_state_features` | `data/processed/game_state` |
| 03 | `outputs/03_player_quality_features` | `data/processed/historical_features` |
| 04 | `outputs/04_dynamic_player_availability` | `data/processed/player_availability` |
| 05 | `outputs/05_remaining_resource_features` | `data/processed/remaining_resources` |
| 06 | `outputs/06_model_training_and_comparison` | `data/processed/model_comparison` |
| 07 | `outputs/07_resource_representation_experiments/resource_sums_v2/four_way_interactions_v1` | `data/processed/resource_experiments/resource_sums_v2/four_way_interactions_v1` |
| 08 | `outputs/08_probability_evaluation/resource_sums_v2_joint_selection_v1` | `data/processed/probability_evaluation/resource_sums_v2_joint_selection_v1` |
| 09 | `outputs/09_nwc_attribution/resource_sums_v2_joint_selection_v1` | `data/processed/nwc_attribution/resource_sums_v2_joint_selection_v1` |

06 model files are in its `models/resource_sums_v2` subfolder. 07 candidate
models are in the `ib/models`, `iw/models` and `ibw/models` subfolders of its
report directory. Retain the candidates used by the comparison tables, not only
the four final selected models. Result manifests record input and output hashes;
figure manifests identify the associated plots. The 09 `terminal_case_audit`
subfolder contains the terminal-case checks.

Original reference documents are in `docs/references/`, the supplied feature
engineering notebook is in `docs/original_materials/`, and source data is in
`data/raw/`. Configuration, tests, dependency requirements and method documents
are part of the reproducible project. Notebook 10 exports player-match histories
without training or prediction. It verifies Stage 09, aligns audited events to
source JSON, joins stable player IDs, aggregates batting/bowling statistics and
preserves NWC missingness and coverage. Its package pointer is
`outputs/10_results_and_visualisation/player_history_v1/current.json`.
See [Player match history export](player_history_export.md) for field definitions,
run instructions and the publication checks. APIs and frontend integration are
separate consumers of this package.

## Execution and integrity checks

Notebook run switches default to False, so opening a notebook reads and verifies
published results. For an intentional computation, enable only the relevant
switch, execute that stage, then return the switch to False before saving.

| Notebook | Explicit computation switches |
| --- | --- |
| 01 | `RUN_AUDIT` regenerates audit reports; `EXPORT_CLEAN_DATA` publishes the reviewed clean cohort |
| 02-05 | `RUN_STAGE` builds the stage's feature or availability export |
| 06-07 | `RUN_STAGE` trains the declared candidates; 07 `RUN_PREFLIGHT` performs input checks |
| 08 | `RUN_DEVELOPMENT_DIAGNOSTICS`, `RUN_EVALUATION`, `RUN_NATIVE_IMPORTANCE` control the three separate reports |
| 09 | `RUN_ATTRIBUTION` builds the contribution ledger; `SAVE_TERMINAL_AUDIT` exports source diagnostics |
| 10 | `RUN_EXPORT` publishes the verified player-history package |

In 07, `RESUME_COMPLETED_ARMS=True` reuses only verified completed arms during
an explicitly requested training run. `SAVE_FIGURES` writes presentation files;
it does not train models or rebuild the underlying numerical results.

False run flags read verified results; True explicitly requests stage execution.
Figure saving and terminal-audit export are separate from training/attribution.
Wording edits need no numerical rerun. Reload notebooks from disk after external
edits rather than saving an old VS Code editor buffer over the new text.

The cross-group Stage 08 protocol is `stage08_joint_selection_protocol.md`.
Stage 09 verifies `stage09_joint_selection_protocol.md` and its producer's
cross-group contract. Stage 10 verifies the joint-selection Stage 09 outputs.
Its immutable snapshots have their own input/output hashes, and `current.json`
identifies the active export. The shared NWC accounting implementation remains
in `src/nwc_attribution.py`; the active producer interface is in
`src/nwc_attribution_joint.py`.
Verification checks those inputs and the corresponding code, model registries
and output files before reading results. A missing file or mismatched hash
stops execution; it does not trigger automatic fitting or evaluation.

The module `src/probability_evaluation_common.py` supplies shared metrics,
calibration, bootstrap and input-validation functions to Stage 08. It has no
standalone evaluation entry point or experiment-specific configuration.
