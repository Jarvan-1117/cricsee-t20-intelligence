# CricSee — T20 Intelligence

T20 cricket win probability modelling and partial Net Win Contribution (NWC) analysis.

## Project presentation

The complete presentation combines the research background, experiments and
results with the interactive prototype walkthrough (approximately 10 minutes
46 seconds, English narration and subtitles).

- [Download the full presentation (MP4)](https://github.com/Jarvan-1117/cricsee-t20-intelligence/releases/download/presentation-v1/CricSee.mp4)
- [Presentation release and details](https://github.com/Jarvan-1117/cricsee-t20-intelligence/releases/tag/presentation-v1)

The video is distributed as a release asset rather than stored in Git history.

## Repository structure

```text
research/
  notebooks/     Stages 01–10
  src/           Reusable implementations
  tests/         Automated checks
  config/        Experiment settings
  docs/          Method and export documentation
  outputs/       Selected published tables, summaries and figures
```

The prototype frontend and backend are not included in this publication.

## Start here

- [Research overview](research/README.md)
- [Workflow and methods](research/docs/current_workflow.md)
- [Notebook 08 — model evaluation](research/notebooks/08_probability_evaluation.ipynb)
- [Notebook 09 — partial NWC](research/notebooks/09_nwc_attribution.ipynb)
- [Player history export](research/docs/player_history_export.md)
- [Published results and reproduction scope](research/PUBLICATION.md)

Four IBW models are selected by validation performance within the fixed feature
group. NWC uses the resulting probabilities and explicit partial allocation rules.
The reported player contributions are not causal ability ratings or complete
career histories. Current NWC coverage is 228 test matches.

## Running locally

Run research commands from `research/`, not the repository root. Create a local
Python environment and install `requirements-lock.txt` (full environment) or
`requirements.txt` (runtime dependencies). Open notebooks with that interpreter.

Raw/processed datasets, trained models and full intermediate artifacts are not
bundled. The selected results are for inspection, not a complete runtime cache.
Default read/verification modes may stop when omitted artifacts are absent; this
is intentional. Restore the complete verified artifacts or prepare the required
source data and explicitly run the pipeline before attempting a full execution.
See [publication scope](research/PUBLICATION.md) for the required input paths.
