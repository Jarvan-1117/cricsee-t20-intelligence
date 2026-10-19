# Stage 09: consume four fixed-IBW Stage 08 prediction streams

This is an input/interface migration, not a change to NWC calculation or
allocation. It replaces the joint feature-group selection contract.

## Frozen input contract

Read verified `resource_sums_v2_fixed_ibw_v1` Stage 08 outputs. All four tracks
must use IBW: state/background, both original remaining-resource blocks, and
both resource/progress interaction blocks. Algorithms and parameters were
selected within IBW on validation event Log Loss, not test performance.

Verify the full 32-model comparison registry and its four preferred IBW rows,
the evaluation ID, model paths/hashes, feature lists, version and prediction
path. Require exactly one raw pre-event probability per test event and the
first-batting-team target in both innings. Reject old joint/fixed-RBW results,
non-IBW preferred rows, adjusted probabilities and incomplete/duplicate data.
The producer's frozen `requires_stage09_migration` status is accepted by this
explicitly migrated consumer; do not rewrite Stage 08 artifacts or identities.

No model fitting, prediction calls, calibration or reselection. Test-match
coverage is inherited from Stage 08 and is not expanded to full player history.

## Unchanged computation

Preserve event validation, allocation priority and rule table, observed
next-pre-event interval differences, current-event striker/bowler pair proxies,
role-change flags, all residual categories, player totals and reconciliation.
Keep first-event coverage and the inseparable first-innings final-event/innings
transition bundle unallocated. Close the final match event to the audited 0/1
outcome, without clipping, deleting, smoothing or redistributing its movement.
Missing player allocation is not zero. These remain partial operational
contributions, not identified causal responsibility or complete ability rankings.

Preserve figure definitions and scales. Re-evaluate all terminal cases with
absolute terminal WPA at least 0.25 using the unchanged raw-source audit.
The last five events provide context, not smoothing; case counts are dynamic.

## Versioned outputs and acceptance

Reports: `outputs/09_nwc_attribution/resource_sums_v2_fixed_ibw_v1`.
Ledger: `data/processed/nwc_attribution/resource_sums_v2_fixed_ibw_v1`.
Preserve all older output directories and back up the prior implementation.

Verify unchanged core function source/AST and unchanged rule configuration
apart from the version/model policy. On identical probabilities, old and new
implementations must produce identical ledgers and player/team summaries.
Check raw input equality, keys, first/terminal events, reconciliation, role
allocation masks and terminal sources. Men's models are unchanged, so compare
their new/old results within numerical tolerance; women's changes must follow
only from the new probability streams. No pass threshold is prediction tuning.

The authorized migration includes one explicit attribution run, figure export
and terminal audit. Keep notebook `RUN_ATTRIBUTION=False` afterward for verified
reading; missing/stale output never causes an automatic run. Diagnostic export
is independent of attribution. Do not label an old cached notebook result as
fixed-IBW output.
