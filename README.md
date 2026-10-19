# CricSee — T20 Intelligence

T20 cricket win probability modelling and partial Net Win Contribution (NWC) analysis.

> Upload in progress: research source files and the presentation video have not yet been published to this repository.

## Project presentation

The completed local presentation combines the research background, experiments and results with the interactive prototype walkthrough (approximately 10 minutes 46 seconds, English narration and subtitles). The video download will be added after the upload completes.

## Planned repository structure

```text
research/
  notebooks/     Stages 01–10
  src/           Reusable implementations
  tests/         Automated checks
  config/        Experiment settings
  docs/          Method and export documentation
  outputs/       Selected published tables, summaries and figures
```

The local prototype directory currently contains only a placeholder; its frontend and backend code still need to be supplied.

## Research scope

Four IBW models are selected by validation performance within the fixed feature group. NWC uses the resulting probabilities and explicit partial allocation rules. The reported player contributions are not causal ability ratings or complete career histories. Current NWC coverage is 228 test matches.

Raw/processed datasets, trained models and full intermediate artifacts are not bundled in the prepared source publication.
