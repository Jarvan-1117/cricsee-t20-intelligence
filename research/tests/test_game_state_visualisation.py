"""Stage 02 figures and notebook syntax, using synthetic data only."""
import json
from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import game_state_features as gs
import game_state_visualisation as viz
from test_game_state_features import fixture, ROOT


class GameStateVisualisationTests(unittest.TestCase):
    def setUp(self):
        d, m, i = fixture()
        self.frame, self.background = gs.build_features(d, m, i, gs.read_config(ROOT))

    def tearDown(self):
        plt.close('all')

    def test_match_figure_is_pre_event_and_has_two_innings(self):
        fig = viz.match_figure(self.frame, 1)
        self.assertEqual(len(fig.axes), 4)
        self.assertEqual(list(fig.axes[0].lines[0].get_ydata()), [5, 6, 7, 11, 11, 13])
        self.assertIn('no win-probability model fitted', fig._suptitle.get_text())
        fig.canvas.draw()

    def test_coverage_reconciles_with_events(self):
        coverage = self.frame.groupby(['gender', 'match_innings', 'innings_phase']).agg(events=('source_row', 'size')).reset_index()
        fig = viz.overview_figure({'matches': 2, 'innings': 4, 'events': len(self.frame)}, coverage)
        self.assertEqual(sum(p.get_width() for p in fig.axes[-1].patches), len(self.frame))
        fig.canvas.draw()

    def test_history_figure_and_missing_gender(self):
        fig = viz.history_figure(self.background)
        self.assertEqual(len(fig.axes), 2)
        self.assertIn('Historical', fig.axes[1].get_ylabel())
        fig.canvas.draw()

    def test_missing_example_rejected(self):
        with self.assertRaises(ValueError): viz.match_figure(self.frame, 999)

    def test_extreme_chase_rate_is_not_clipped(self):
        self.frame.loc[self.frame.match_innings.eq(2), 'required_run_rate'] = 350.0
        fig = viz.match_figure(self.frame, 1)
        ax = fig.axes[3]
        self.assertEqual(ax.get_yscale(), 'symlog')
        self.assertEqual(max(ax.lines[-1].get_ydata()), 350.0)
        self.assertIn('log above', ax.get_ylabel())
        fig.canvas.draw()

    def test_notebook_code_compiles_and_has_safe_default(self):
        notebook = json.loads((ROOT / 'notebooks/02_game_state_features.ipynb').read_text(encoding='utf-8'))
        code = []
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source']); compile(source, f'cell_{i}', 'exec'); code.append(source)
        self.assertIn('RUN_STAGE = False', '\n'.join(code))
        self.assertNotIn('subprocess', '\n'.join(code))


if __name__ == '__main__':
    unittest.main()
