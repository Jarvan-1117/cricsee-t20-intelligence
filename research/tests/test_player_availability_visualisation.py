"""Figure content, rendering, manifest linkage and notebook controls."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import player_availability as pa
import player_availability_visualisation as viz
from test_player_availability import fixture, replay


class AvailabilityVisualisationTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.transitions, _, _ = replay(fixture())

    def tearDown(self):
        plt.close("all")

    def test_batting_series_and_labels(self):
        fig = viz.batting_figure(self.frame)
        self.assertEqual(list(fig.axes[0].lines[0].get_ydata()), [11, 11])
        self.assertEqual(list(fig.axes[0].lines[1].get_ydata()), [9, 9])
        self.assertIn("pre-event", fig.axes[0].get_xlabel())
        fig.canvas.draw()

    def test_bowling_values(self):
        fig = viz.bowling_figure(self.frame, {"id_B1": "B1"})
        self.assertEqual(list(fig.axes[0].lines[0].get_ydata()), [24, 23])
        fig.canvas.draw()

    def test_overview_counts(self):
        summary = {"matches": 2, "replayed_events": 3, "ready_matches": 1, "review_matches": 1, "quarantined_matches": 0}
        coverage = pd.DataFrame({"gender": ["male", "female"], "status": ["ready", "review_required"], "matches": [1, 1]})
        fig = viz.overview_figure(summary, coverage)
        self.assertEqual(sum(p.get_width() for p in fig.axes[-1].patches), 2)
        fig.canvas.draw()

    def test_empty_timeline_renders(self):
        fig = viz.timeline_figure(self.frame, pd.DataFrame(columns=pa.TIMELINE_COLUMNS), {})
        fig.canvas.draw()

    def test_after_event_offset(self):
        t = pd.DataFrame([{"match_id": 1, "match_innings": 1, "event_sequence": 1,
                           "player_id": "id_A1", "event": "temporary_retirement", "timing": "after"}])
        fig = viz.timeline_figure(self.frame, t, {"id_A1": "A1"})
        self.assertEqual(float(fig.axes[0].collections[0].get_offsets()[0, 0]), 1.5)
        fig.canvas.draw()

    def test_empty_examples_rejected(self):
        with self.assertRaises(ValueError): viz.batting_figure(pd.DataFrame())
        with self.assertRaises(ValueError): viz.bowling_figure(pd.DataFrame(), {})

    def test_figure_manifest_linkage(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            fig = viz.batting_figure(self.frame)
            with patch.object(viz, "sha256", return_value="test-hash"):
                path = viz.save_figures(root, {"availability_id": "id1"}, {"test.png": fig})
            saved = json.loads((path / "figure_manifest.json").read_text())
            self.assertEqual(saved["availability_id"], "id1")
            self.assertEqual(saved["figure_sha256"], {"test.png": "test-hash"})

    def test_notebook_syntax_and_explicit_switch(self):
        notebook = json.loads((ROOT / "notebooks/04_dynamic_player_availability.ipynb").read_text())
        code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        for cell in code: compile(cell, "stage04", "exec")
        self.assertEqual(len(code), 8)
        self.assertIn("RUN_STAGE = False", code[0])
        self.assertIn("if RUN_STAGE:", code[1])
        self.assertNotIn("if stale", "\n".join(code))


if __name__ == "__main__":
    unittest.main()
