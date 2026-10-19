# Player match history export

Stage 10 exports chronological player-match histories for the verified Stage 09
test cohort. It reads results without training, prediction or model selection.
The export is not a full-career database or an API service. Its source is
`nwc_attribution_joint`, using the Stage 09 `resource_sums_v2_joint_selection_v1`
ledger and player/team tables. Stage 08 selects each track's feature group,
algorithm and parameters by validation Log Loss; Stage 10 only consumes the
verified resulting contributions. The package carries the attribution and
evaluation IDs so consumers can identify its source.

## Run and inspect

Open `notebooks/10_results_and_visualisation.ipynb`. Set `RUN_EXPORT = True` to
build the package; use False to verify and read the published package. Set
`PLAYER_ID` and `GENDER` to inspect two chronological tables for one player.
Successful exports do not require any upstream notebook to run again.

The equivalent command, from the project root, is:

```text
python -B src/player_history_export.py --export
```

Without `--export`, the command verifies the current package and its sources.
The notebook does not modify the export flag automatically.

## Package files

`outputs/10_results_and_visualisation/player_history_v1/current.json` points to
an immutable, content-identified directory under `snapshots/`. Consumers must
resolve this pointer, not select a directory by timestamp.

| File | Purpose |
| --- | --- |
| history.json | Schema 2 product payload: players, aliases, teams, matches, nested batting/bowling history and definitions |
| batting_history.csv | One row per listed player/match, batting statistics and NWC state |
| bowling_history.csv | Corresponding bowling table; dates in ascending order |
| verification.json | Source alignment, all-player statistical checks and team contribution reconciliation |
| manifest.json | Input/output hashes, dataset version and generation timestamp |

Rows include listed players who did not bat or bowl. `match_participation`
distinguishes initial XI, observed participants and unconfirmed catalogue-only
entries. An unconfirmed entry does not count as a confirmed match appearance.
Player identity uses Cricsheet IDs, not name/team concatenation. Gender remains
part of every query scope. Names and team associations are recorded per match.

The JSON is the authoritative product contract. CSV blanks represent nulls;
consumers must not replace them with zero. No model files or local source paths
are exposed in the product payload. The manifest is an internal provenance file.

## Traditional statistics

Statistics use every audited event in the exported cohort, not just events with
player allocations. Each event is aligned to the source JSON by match, innings
and event sequence; participants, scores, extras and exits must agree.

- Runs: batter runs. Balls faced: recorded non-wide events, including no-balls,
  following the existing source-event convention. Strike rate is 100 times runs
  divided by balls faced, null when the denominator is zero.
- Fours and sixes: batter runs of 4 or 6, excluding `runs.non_boundary` events.
  The source distinguishes a boundary from four or six completed runs.
  [Cricsheet delivery format](https://cricsheet.org/format/json/#delivery-data).
- Dismissals: show the recorded exit kind or not-out/did-not-bat state; retain
  the exit list. A non-striker run-out belongs to the dismissed player, not
  automatically to the striker. Retired hurt is not a dismissal count.
- Bowler runs conceded: batter runs plus wides and no-ball extras. Byes,
  leg-byes and separate penalties are excluded, consistent with bowler debits
  in [MCC Law 18](https://www.lords.org/mcc/the-laws-of-cricket/scoring-runs).
- Bowler wickets: bowled, caught, caught and bowled, lbw, stumped, hit wicket.
  Run-outs, retired out and obstructing the field are not bowler wickets.
- Bowling balls: actual legal deliveries. Overs display uses six-ball
  equivalents (for example, 22 legal balls is `3.4`), not a decimal number or
  a reconstruction of miscounted umpire over slots. Economy is six times
  conceded runs divided by actual legal deliveries; zero denominator is null.
- Dot balls: legal deliveries with zero total event runs. This explicitly
  excludes scoring byes and leg-byes even when not charged to the bowler.

Rate values are not rounded in the export. The presentation layer may format
rates to two decimals. These are source-derived statistics, not a separately
certified official scorecard.

## NWC and missingness

Each role has a nested `nwc` object. `available` means at least one event has
an allocation; only those events are summed. A zero is valid in that state.
`unallocated` means the player participated in the role but has no allocated
events. `not_participated` means the role was not performed by a confirmed
participant. `unavailable` means match participation is not established.
The latter three states have null NWC, never zero-filled NWC.

Entering as non-striker counts as batting participation even without facing a
ball. Stage 09 batting role events are striker events, so a non-striker-only
appearance can have zero role events and null allocation coverage. Role-event
counts and balls faced are not interchangeable.

NWC is copied from Stage 09 after checking every role against the event ledger.
Values are probability differences, positive for the player's team; multiply by
100 only when displaying percentage points. Terminal contributions are included
in partial NWC and also exposed as a component, not added a second time.
Unallocated first-innings boundaries and other team residuals are not assigned
to players. Partial NWC is not a complete causal or ability ranking.

## Verification and publication

The export verifies the Stage 09 dependency chain, fingerprints every extra
source, validates stable IDs and event joins, and reconciles all player
statistics against the independent Stage 03 accounting table. Boundary counts
are separately checked using source flags instead of Stage 03's quality-feature
boundary proxy. All player NWC totals plus team residuals reconcile to Stage 09.

Input hashes are checked again after computation. Output is built and verified
in a temporary staging directory before publication. The current pointer is
updated atomically only after a complete package exists. Failed exports cannot
replace the current package with an incomplete one. Repeating an export with
identical sources reuses the same snapshot. No upstream files are written.

The prototype's player-history API reads this Schema 2 package and pins a
verified snapshot at startup. After publication, restart the API to activate
the new snapshot and refresh the browser's data session. The independent R0
demo uses a different contract and is not a consumer of these NWC values.
