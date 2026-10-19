"""Stage 05 presentation selection, labels, plotted values and notebook syntax."""
import json
from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import remaining_resource_features as rf
import remaining_resource_visualisation as viz
from test_remaining_resource_features import resource_case


class ResourceVisualisationTests(unittest.TestCase):
    def setUp(self):
        events,profiles,lineups=resource_case();config=rf.read_config(ROOT)
        lookup=rf.match_profiles(profiles,lineups,config)
        rows=[];details=[]
        for row in events.itertuples(index=False):
            result,detail=rf.event_resources(row,lookup,config)
            result.update(match_date=row.match_date,batting_team=row.batting_team,bowling_team=row.bowling_team,source_over=row.source_over)
            rows.append(result);details.append(detail)
        self.frame=pd.DataFrame(rows);self.details=pd.DataFrame(details)

    def tearDown(self):
        plt.close('all')

    def test_batting_dimensions_and_uncentered_sums(self):
        fig=viz.batting_figure(self.frame)
        self.assertEqual(len(fig.axes),4)
        self.assertEqual(list(fig.axes[0].lines[0].get_ydata()),[900,900])
        self.assertIn('smoothed',fig.axes[0].get_ylabel())
        fig.canvas.draw()

    def test_bowling_weighted_sum_per_innings(self):
        fig=viz.bowling_quality_figure(self.frame)
        self.assertEqual(len(fig.axes),6)
        self.assertEqual(len(fig.axes[0].lines),2)
        self.assertEqual(fig.axes[0].lines[0].get_ydata()[0],352)
        fig.canvas.draw()

    def test_capacity_plot_does_not_allocate_innings_budget(self):
        fig=viz.capacity_figure(self.frame.iloc[0],self.details.iloc[0],{})
        self.assertEqual(sum(p.get_width() for p in fig.axes[0].patches),44)
        self.assertIn('44 overs',fig.axes[0].get_title(loc='left'))
        fig.canvas.draw()

    def test_coverage_counts(self):
        summary={'upstream_matches':3,'matches':2,'events':3,'review_or_quarantined_matches':1}
        coverage=pd.DataFrame({'gender':['male','female','female'],'included_in_resource_sample':[True,True,False],'matches':[1,1,1]})
        fig=viz.overview_figure(summary,coverage)
        self.assertEqual(sum(p.get_width() for p in fig.axes[-1].patches),3)
        fig.canvas.draw()

    def test_calendar_example_selection_ignores_outcome(self):
        f=pd.DataFrame({'match_id':[3,2,1],'match_date':['2021-01-01','2020-01-01','2019-01-01'],'outcome':[0,0,1]})
        self.assertEqual(int(viz.select_example(f).match_id.iloc[0]),2)
        f['outcome']=1-f['outcome']
        self.assertEqual(int(viz.select_example(f).match_id.iloc[0]),2)

    def test_example_fallback_and_empty(self):
        f=pd.DataFrame({'match_id':[2,1],'match_date':['2019-01-01','2018-01-01']})
        self.assertEqual(int(viz.select_example(f).match_id.iloc[0]),1)
        with self.assertRaises(ValueError):viz.select_example(f.iloc[:0])

    def test_notebook_syntax_and_default_read_mode(self):
        n=json.loads((ROOT/'notebooks/05_remaining_resource_features.ipynb').read_text(encoding='utf-8'))
        cells=[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
        self.assertEqual(len(cells),8)
        for c in cells:compile(c,'stage05','exec')
        self.assertIn('RUN_STAGE = False',cells[0]);self.assertIn('if RUN_STAGE:',cells[1])
        self.assertNotIn('if stale','\n'.join(cells))


if __name__=='__main__':
    unittest.main()
