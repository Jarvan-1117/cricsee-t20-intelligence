# Published results and reproduction scope

This repository includes the experiment code, configuration, tests, method
documents, notebooks with saved results, and a selected result snapshot.

## Included results

- 01–05: audit/coverage summaries, metric definitions and diagnostic figures.
- 06: selected models, chronological split summary, validation differences and figures.
- 07: selected models, interaction comparisons, experiment summary and figures.
- 08: validation-selected cross-group model registry, all-group and final-model
  test metrics, paired comparisons, calibration bins, per-feature and block
  permutation importance, sample counts and eight figures.
  `development_diagnostics/` contains all-group and selected-model training/
  validation metrics, their gaps and a summary. `native_importance/` contains
  W1 XGBoost Gain, M2 Random Forest MDI, per-model applicability and a summary;
  these are different metrics, not interchangeable importance scales.
- 09: allocation rules and coverage, match/team reconciliation, player-match
  partial NWC, terminal-audit summary and six figures, using the four selected
  Stage 08 prediction streams.
- 10: verification report for the currently verified player-history export.
  The report retains its source snapshot directory identifier for provenance;
  the complete product payload and snapshot pointer are not included.

CSV filenames and their source directory layout are retained. Code, experiment
configuration and method protocols are copied without computational changes.
Any local machine prefixes removed from notebook display outputs do not change
cell source or numerical results.

## Published model and result identity

Stage 08 selects each track's feature group, algorithm and parameters by validation
Log Loss only: W1 uses RW/XGBoost, W2 RBW/HistGradientBoosting, M1 IBW/Logistic
Regression and M2 IBW/Random Forest. Test comparisons are exploratory and do not
reselect models. Both Stage 08 and 09 results are under
`resource_sums_v2_joint_selection_v1/`.

- Evaluation ID: `31bd1c42bb6e0ec84f9824da5fe496de9fc7e0680f32c96bc2db445e30fa3bea`.
- Attribution ID: `f6a25bf2480b950fe31e69672273d7a4c06807f72e6efce1098ba6261d48baae`.
- Stage 10 snapshot: `8e9f523ce24dc6af3a9d0f393469720e8911aec76848d0ffaf3777d0da37a21f`.

The verified cohort contains 228 test matches and 53,850 events. The player-history
export contains 542 players and 5,016 player-match rows. Only its verification
report is published here; the API package must be generated separately. The
prototype application is outside this research publication.

## Not included

Raw and processed data, trained model files, full event-level prediction/attribution
tables, complete product history packages, environments, caches, editor settings,
private reference documents and the prototype application are not included.

The `outputs/` folder is a selected publication snapshot, not a complete set of
artifacts accepted by runtime hash verification. Full-stage manifests are not
provided as if omitted files were present. Successful inspection of a figure does
not establish that the complete experiment can execute from this checkout.

## Required data for a full run

The cleaning stage expects these paths relative to `research/`:

```text
data/raw/t20i_deliveries_data.csv
data/raw/t20i_matches_data.csv
data/raw/cricsheet/t20s_female_json.zip
data/raw/cricsheet/t20s_male_json.zip
```

The CSV inputs are the project's supplied datasets and must be obtained from
their owner. They cannot be reconstructed by simply renaming a download.
Cricsheet archives must correspond to the source version used by the experiment;
a later download may contain different matches or corrections. Obtain the exact
input snapshot for exact reproduction. Source access and redistribution rights
must be checked separately; this repository does not grant rights to omitted data.

Install the Python dependencies from the provided requirement files. Run commands
from `research/`. Prepare the inputs, then explicitly execute 01–05, training and
selection in 06–07, evaluation in 08, attribution in 09 and history export in 10.
Run flags defaulting to false read existing verified outputs; they do not rebuild
missing data automatically. A full rerun can be expensive.

Saved notebook outputs are a record for reading. Inspect run flags before using
Run All. Do not disable integrity checks to make a partial checkout appear complete.

## Test prerequisites

The full test command is `python -B -m unittest discover -s tests` from `research/`.
It includes artifact-backed checks, not only self-contained unit tests. In this
selected-results checkout, four probability integration test classes require
the omitted `outputs/05_remaining_resource_features/feature_contract.json`;
the Stage 07 completeness test requires the full verified Stage 07 artifacts,
including its manifest, models and validation predictions. Generate or restore
those artifacts before expecting the entire suite to pass. They are deliberately
not replaced by stub manifests, and the checks are not disabled for publication.

Notebook presentation checks can run without the omitted datasets and models:
`python -B -m unittest discover -s tests -p test_notebook_workflow.py`.
