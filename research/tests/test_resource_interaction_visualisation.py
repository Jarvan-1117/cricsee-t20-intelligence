"""Synthetic rendering tests for the new interaction comparison figures."""
import io
from pathlib import Path
import sys
import unittest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import resource_interaction_experiments as stage
import resource_interaction_visualisation as viz


class FigureTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_all_figures_render(self):
        selected = pd.DataFrame([dict(track=t, arm_code=c, validation_event_log_loss=.4+i*.001,
                                      validation_accuracy=.7+i*.002)
                                 for t in stage.TRACKS for i,c in enumerate(stage.CODES)])
        comparisons = pd.DataFrame([dict(track=t, treatment=c, comparison_mode=mode,
             reference_family=family, validation_log_loss_gain=.001)
             for t in stage.TRACKS for c in stage.CODES[1:]
             for mode in ['fixed_baseline_parameters', 'selected_system'] for family in stage.base.FAMILIES])
        phases = pd.DataFrame([dict(track=t, arm_code=c, phase=p, validation_event_log_loss=.5)
                              for t in stage.TRACKS for c in stage.CODES for p in stage.common.PHASES])
        figures = [viz.design_figure(), viz.metric_figure(selected, 'validation_event_log_loss', 'Log Loss'),
                   viz.metric_figure(selected, 'validation_accuracy', 'Accuracy'),
                   viz.gain_figure(comparisons), viz.phase_figure(phases)]
        for fig in figures:
            output = io.BytesIO()
            fig.savefig(output, format='png')
            self.assertGreater(len(output.getvalue()), 1000)


if __name__ == '__main__': unittest.main()
