# Stage 07: four-way resource/progress interactions

## Question and scope

Do explicit interactions between remaining resources and innings progress improve
validation predictions beyond the same RBW state/background and resource inputs?
Stage 06 supplies the selected RBW models used as controls.

| Code | Original inputs retained | Additional features |
| --- | --- | --- |
| RBW | State/background + both dynamic resource blocks | None |
| IB | Same | Four batting-resource metrics times remaining-ball fraction |
| IW | Same | Three bowling-resource metrics times remaining-ball fraction |
| IBW | Same | All seven products |

The fraction is `scheduled_balls_remaining_estimate / 120` in every arm. This
is a nominal pre-event value, not the future realized duration. Initial fraction
one reproduces the original metric, and fraction zero makes the added product
zero without removing the original resource. Every metric remains separate.
No current striker/non-striker/bowler direct historical quality is introduced.
No explicit products does not imply that a tree model cannot learn interactions.

## Fixed controls and selection

Reuse the Stage 06 whole-match, same-date chronological split within gender,
eligible cohort, target, seed, train-only preprocessing and candidate grid.
First-innings models omit structurally unavailable chase fields. Stage 07 uses
the shared chronological split, without calibration fitting, test scoring or
NWC computation. Each added representation fits 48 candidates: four tracks,
four algorithms and three parameter settings. The algorithms are Logistic
Regression, Random Forest, HistGradientBoosting and XGBoost. Total fits in
Stage 07: 144; the RBW controls are loaded from Stage 06.

Within each representation use validation event Log Loss and the configured
family/candidate tie-break to choose an algorithm and parameter setting for
each track. A supplementary table identifies the lowest validation event
Log Loss among RBW, IB, IW and IBW, with exact ties resolved in that order.
This table does not select the final feature group: Stage 08 fixes IBW for all
four tracks and evaluates all eight unique Stage 06/07 groups as comparisons.
Accuracy, Brier, match-mean losses and phase diagnostics are supplementary,
not alternative selection criteria. A small point-estimate improvement does
not establish statistical significance.

Report parameter-matched RBW contrasts, separately tuned within-family contrasts
and selected-system contrasts. Parameter matching does not fix learned weights
or trees; selected-system comparisons may change both representation and family.
Do not suppress negative or small effects. These are exploratory representation
comparisons, not independent confirmation or causal resource effects.

## Artifacts and handoff

Current reports: `outputs/07_resource_representation_experiments/resource_sums_v2/four_way_interactions_v1/`.
Validation streams: the matching `data/processed/resource_experiments/` directory.
Separate `ib/`, `iw/`, `ibw/` subdirectories retain all fitted candidates and
fingerprinted validation outputs. `selected_models.csv` contains 16 group/track
systems. `selected_systems.csv` and `frozen_protocol.json` record the four
supplementary across-arm winners with model hashes, paths and exact input
columns. Stage 08 obtains the within-arm winners from `selected_models.csv`,
uses the four IBW entries as final models, and includes RBW, IB and IW in its
comparison set alongside R0, R1, RB and RW from Stage 06.

Stage 08 evaluates the frozen models without test-driven selection and exports
the four IBW prediction streams to Stage 09. Stage 09 calculates partial
operational NWC from that verified handoff.

## Execution

`python -B src/resource_interaction_experiments.py --preflight` is read-only.
`--train` explicitly trains the three added arms; `--train --resume` reuses only
verified completed arms after interruption. In the notebook, `RUN_STAGE=False`
loads verified results without fitting. Missing results remain pending; stale
results stop rather than automatically retraining.
