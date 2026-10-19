"""Stage 06 chart semantics and notebook syntax; synthetic values only."""
import json
from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import model_comparison as stage
import model_comparison_visualisation as viz
from test_model_comparison import fixture


def chart_tables():
    rows = []
    for track in viz.TRACKS:
        for i, arm in enumerate(stage.ARM_CODES):
            for j, family in enumerate(stage.FAMILIES if arm != 'constant' else ['constant']):
                rows.append({'track': track, 'arm': arm, 'family': family, 'candidate': 0,
                    'validation_event_log_loss': .7 - i * .025 + j * .01, 'validation_brier': .2 - i * .01})
    winners = pd.DataFrame(rows)
    selected = pd.DataFrame([stage.best_record(g.to_dict('records')) for _, g in winners.groupby(['track', 'arm'])])
    return winners, selected, stage.comparison_deltas(winners, selected)


class PresentationTests(unittest.TestCase):
    def tearDown(self): plt.close('all')

    def test_split_counts(self):
        _, _, _, splits, _ = fixture()
        fig = viz.split_figure(stage.split_summary(splits))
        self.assertEqual(sum(p.get_height() for ax in fig.axes for p in ax.patches), len(splits))
        fig.canvas.draw()

    def test_heatmap_dimensions(self):
        winners, _, _ = chart_tables(); fig = viz.heatmap_figure(winners)
        self.assertEqual(len(fig.axes), 4)
        self.assertEqual(fig.axes[0].images[0].get_array().shape, (4, 4))
        self.assertIn('XGBoost', [tick.get_text() for tick in fig.axes[0].get_xticklabels()])
        fig.canvas.draw()

    def test_missing_arm_rejected(self):
        winners, _, _ = chart_tables()
        with self.assertRaises(ValueError): viz.heatmap_figure(winners[winners.arm.ne('both_resources')])

    def test_selected_groups_and_actual_bar_values(self):
        _, selected, _ = chart_tables(); fig = viz.selected_figure(selected)
        self.assertEqual(len(fig.axes[0].patches), 5)
        self.assertAlmostEqual(fig.axes[0].patches[0].get_width(), .7)
        fig.canvas.draw()

    def test_gain_sign_not_inverted(self):
        _, _, deltas = chart_tables(); fig = viz.gains_figure(deltas)
        self.assertAlmostEqual(fig.axes[0].patches[0].get_width(), .025)
        fig.canvas.draw()

    def test_notebook_read_default_and_syntax(self):
        notebook = json.loads((ROOT / 'notebooks/06_model_training_and_comparison.ipynb').read_text(encoding='utf-8'))
        cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        self.assertEqual(len(cells), 9)
        for c in cells: compile(c, 'stage06', 'exec')
        self.assertIn('RUN_STAGE = False', cells[0])
        self.assertIn('if RUN_STAGE:', cells[1])
        self.assertNotIn('if stale', '\n'.join(cells))


if __name__ == '__main__': unittest.main()
