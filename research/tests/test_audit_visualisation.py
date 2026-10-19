"""Presentation tests use synthetic counts, not fixed production totals."""
from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import audit_visualisation as viz


class AuditVisualisationTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_overview_widths_follow_counts(self):
        s = {"audited_matches": 12, "match_counts": {"included": 7, "excluded": 4, "pending_review": 1}}
        fig = viz.overview_figure(s)
        self.assertEqual([p.get_width() for p in fig.axes[-1].patches], [7, 4, 1])
        fig.canvas.draw()

    def test_bad_totals_are_not_displayed_as_valid(self):
        with self.assertRaises(ValueError):
            viz.overview_figure({"audited_matches": 3, "match_counts": {"included": 4}})

    def test_exclusion_reasons_are_not_deduplicated_or_summed(self):
        reasons = pd.DataFrame([{"status": "excluded", "reason": r, "matches": 2} for r in ["tie", "super_over"]])
        fig = viz.exclusion_figure(reasons, 2)
        self.assertEqual([p.get_width() for p in fig.axes[0].patches], [2, 2])
        self.assertIn("overlap", fig.texts[0].get_text())

    def test_cohort_counts_and_gap_years(self):
        data = pd.DataFrame({"match_id": [1, 2, 3], "match_date": ["2020-01-01", "2022-01-01", "2022-06-01"],
                             "gender": ["female", "male", "male"], "csv_events": [200, 230, 220]})
        original = data.copy(deep=True)
        counts, annual, _, _ = viz.cohort_tables(data)
        self.assertEqual(int(counts.matches.sum()), 3)
        self.assertEqual(int(counts.events.sum()), 650)
        self.assertEqual(annual.index.tolist(), [2020, 2021, 2022])
        self.assertEqual(int(annual.loc[2021].sum()), 0)
        viz.cohort_figure(data).canvas.draw()
        pd.testing.assert_frame_equal(data, original)

    def test_empty_reason_chart_and_unknown_pending_reason(self):
        viz.exclusion_figure(pd.DataFrame(columns=["status", "reason", "matches"]), 0).canvas.draw()
        data = pd.DataFrame({"match_id": [1], "match_date": ["2020-01-01"], "gender": ["male"],
                             "csv_events": [20], "pending_reasons": ["source_conflict"]})
        self.assertEqual(viz.pending_table(data)["Review needed"].iloc[0], "source_conflict")

    def test_html_escapes_source_strings(self):
        html = viz.table(pd.DataFrame({"value": ["<script>bad</script>"]}), "<Title>", True).data
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;Title&gt;", html)
        self.assertIn("<details>", html)


if __name__ == "__main__":
    unittest.main()
