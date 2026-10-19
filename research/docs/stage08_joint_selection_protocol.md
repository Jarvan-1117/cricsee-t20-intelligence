# Stage 08: validation-selected probability models

For each of W1, W2, M1 and M2, choose the feature group, algorithm and
hyperparameters using validation event Log Loss. Candidate groups are
R0, R1, RB, RW, RBW, IB, IW and IBW. R0 is a constant-probability baseline;
the seven learned groups each compare Logistic Regression, Random Forest,
HistGradientBoosting and XGBoost using the Stage 06 parameter grid.

## Selection

Verify the complete Stage 06 and Stage 07 manifests, shared split and model
hashes. Check candidate coverage and reproduce the within-group winners from
the candidate tables using the upstream tie rule (Log Loss, family order,
candidate index). Deduplicate the shared RBW control. Select the lowest
validation event Log Loss among the eight group winners in each track.
Exact cross-group ties use the group order listed above, never test metrics.
Taking the best of the within-group winners also minimises the validation
score over the complete candidate grid.

Freeze all 32 comparison systems and four final systems before reading test
values. Model fitting uses training data only. Stage 08 does not fit models,
refit on train plus validation, fit calibrators or alter feature definitions.

## Evaluation

Report the four selected systems as the final-model results. The eight-arm
table is an exploratory comparison, in fixed group order; it does not promote
a model based on test performance. Report event/match Log Loss and Brier,
accuracy, F1 at threshold 0.5, AUC and ten fixed-width calibration bins.
Use 2,000 paired whole-match bootstrap draws, shared across innings and arms
within gender, with pointwise 95% intervals. RBW versus R1 is the primary
resource contrast. The test cohort has previously been inspected: it is not
a new untouched confirmatory sample. Intervals condition on fitted models
and do not account for selection uncertainty.

Explain the four selected models using marginal permutation importance for
each actual raw input and each nonempty state/resource/interaction block.
Ten fixed-seed repeats yield mean Log Loss increase and shuffle SD, not a
sampling confidence interval or causal effect. Keep negative values. Shuffle
explicit interaction inputs separately; correlation and inconsistent shuffled
products limit interpretation. Constant models have no feature importance.

## Contract and execution

Contract: `resource_sums_v2_joint_selection_v1`.
Policy: `cross_group_validation_log_loss_v1`.
Engine: `probability_evaluation_joint.py`.
Reports: `outputs/08_probability_evaluation/resource_sums_v2_joint_selection_v1`.
Predictions: `data/processed/probability_evaluation/resource_sums_v2_joint_selection_v1`.

`--freeze` verifies candidates and writes a validation-only selection snapshot;
it does not read test predictions, produce test metrics or publish an NWC
handoff. `--evaluate` freezes the registry before evaluating saved models and
exports four raw pre-event probability streams, one per test event. The target
is always a win for the first-batting team. Terminal closure and player NWC
allocation are outside Stage 08. Consumers must verify the contract, evaluation
ID and model identities before using the exported stream.

The notebook uses `RUN_EVALUATION = False` for verified reading. Missing or
stale evaluation outputs stop explicitly; no automatic training or evaluation.
