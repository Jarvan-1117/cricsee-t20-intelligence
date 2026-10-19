# Stage 08: fixed-IBW final models

Authorized 8 October 2026. Four tracks use the same feature-group definition:
state/background, both remaining-resource blocks and all seven resource/progress
products. First innings omit structurally unavailable chase-only fields. This
is a structural choice for consistency, not a claim that IBW wins every comparison.

## Selection and inputs

For each of W1, W2, M1 and M2, inherit the IBW algorithm and parameters already
selected by Stage 07 validation event Log Loss. Do not select the best feature
group across 06/07. Upstream family/candidate tie rules remain unchanged. Keep
R0/R1/RB/RW/RBW/IB/IW/IBW as eight comparison arms; deduplicate the shared RBW
control. Verify both upstream stages, model hashes, allowlists and frozen splits.

Freeze all 32 comparison systems and four fixed-IBW final systems before reading
test values. A better validation/test score from a non-IBW arm cannot replace
the final group. Stage 06 and Stage 07 are not refitted. No train+validation
refit, new grid, temporal split, probability calibrator or feature modification.

## Evaluation and importance

Retain identical test matches/events, event/match Log Loss and Brier, accuracy,
F1 at threshold 0.5, AUC and ten-bin calibration. Retain the same paired
contrasts, 2,000 whole-match bootstrap draws, seed and pointwise 95% intervals.
RBW versus R1 remains the primary resource contrast; IBW comparisons remain
reported with the other planned comparisons. No test-driven reselection.

Explain only the four fixed-IBW final models. Permute each actual input and
the five disjoint blocks (state/background, batting resources, bowling resources,
batting interactions, bowling interactions), using ten fixed-seed repeats.
Shuffle explicit product columns independently; do not reconstruct products.
Report mean Log Loss increase and shuffle SD, retaining negative importance.
These are descriptive model-reliance measures, not causal effects or sampling
confidence intervals. Correlated features and inconsistent shuffled states
limit interpretation. Do not retune from test importance or scores.

## Version and boundary

Contract: `resource_sums_v2_fixed_ibw_v1`.
Selection policy: `fixed_ibw_within_arm_validation_log_loss_v1`.
Reports: `outputs/08_probability_evaluation/resource_sums_v2_fixed_ibw_v1`.
Data: `data/processed/probability_evaluation/resource_sums_v2_fixed_ibw_v1`.

The active notebook imports `probability_evaluation_fixed_ibw` and its matching
visualisation module. The old joint-selection engine/results stay unchanged so
the existing Stage 09 remains readable. Export four IBW raw pre-event probability
streams, covering the same test events exactly once. No terminal 0/1 override
or NWC is computed by Stage 08. Stage 09 requires a separate explicit interface
update before consuming this new contract; do not relabel old NWC as IBW NWC.

False run flags read verified results; True explicitly evaluates without training.
Missing/stale results stop rather than automatically run. Comparisons remain
exploratory and conditional on the fitted models and selection procedure.
