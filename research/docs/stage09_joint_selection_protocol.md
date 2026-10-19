# Stage 09: event-interval NWC attribution

## Prediction contract

Read the verified Stage 08 `resource_sums_v2_joint_selection_v1` handoff.
For W1, W2, M1 and M2, Stage 08 selects the feature group, algorithm and
parameters using validation event Log Loss only. Stage 09 requires exactly
one frozen model per track, with matching model hashes, features and event keys.
It reads each test event's raw pre-event first-batting-team win probability.
There is no training, prediction, calibration fitting or model selection here.

## Accounting and allocation

Within an innings, the interval is the next pre-event probability minus the
current pre-event probability. The first-innings last event and innings
transition form one inseparable team-level bundle. The final second-innings
event closes to the source-audited binary outcome, without clipping or smoothing.

Ordinary no-extra/no-exit intervals and source-validated bowled/LBW intervals
without extras use the current striker/bowler pair convention. Contributions
are opposite in sign, expressed in each player's team perspective. Other exits,
extras, penalties and the innings-boundary bundle remain team residuals.
Role changes are flagged, not automatically excluded. Missing player allocation
is not an observed zero contribution. Batting and bowling totals are separate.

Player totals plus team residuals must reconcile to the match's outcome minus
its first pre-event probability (tolerance `1e-10`). Every event, including both
innings' first events and the final event, appears exactly once.

## Outputs and interpretation

The versioned report contains allocation rules, event coverage, player-match
totals, match/team reconciliation, boundary bundles and terminal events. The
processed directory contains the full interval ledger. A manifest binds these
artifacts to configuration, shared accounting implementation, producer handoff
and source data. Figures identify their attribution ID and source hash.

Terminal diagnostics inspect absolute final-event changes of at least 25
percentage points against raw JSON source facts. Last-five-event context is
descriptive; it does not redistribute contributions or select models.

These are partial operational contributions, not identified causal player
values. Allocation coverage does not establish causal accuracy. Historical
quality can affect probabilities through resource context. The boundary bundle
is not a pure model-transition gap, and accounting consistency does not certify
probability calibration. Coverage is the frozen test cohort, not entire careers.
