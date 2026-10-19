import json
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from test_player_quality_features import fixture, ROOT, pq
import player_quality_visualisation as viz


class PlayerQualityVisualisationTests(unittest.TestCase):
    def setUp(self):
        d,roster=fixture()
        linked=pq.link_event_identities(d,roster)
        self.profiles=pq.build_profiles(pq.match_contributions(linked,roster),pq.read_config(ROOT))

    def tearDown(self): plt.close('all')

    def test_profile_coverage_is_one_hundred_percent_per_role(self):
        coverage=pq.coverage_report(self.profiles)
        summary={'matches':2,'events':18,'profiles':44,'players_by_gender':{'male':22}}
        fig=viz.coverage_figure(summary,coverage)
        self.assertAlmostEqual(sum(p.get_width() for p in fig.axes[-1].patches),200)
        fig.canvas.draw()

    def test_all_distribution_dimensions_render(self):
        fig=viz.distribution_figure(self.profiles)
        self.assertEqual(len(fig.axes),4)
        self.assertIn('Credited wickets',fig.axes[3].get_xlabel())
        fig.canvas.draw()

    def test_smoothing_scatter_handles_absent_gender(self):
        fig=viz.smoothing_figure(self.profiles)
        self.assertIn('Sparse history',fig._suptitle.get_text())
        fig.canvas.draw()

    def test_notebook_compiles_and_defaults_to_no_rebuild(self):
        nb=json.loads((ROOT/'notebooks/03_player_quality_features.ipynb').read_text(encoding='utf-8'))
        sources=[]
        for i,c in enumerate(nb['cells']):
            if c['cell_type']=='code':
                source=''.join(c['source']); compile(source,f'cell_{i}','exec'); sources.append(source)
        self.assertIn('RUN_STAGE = False','\n'.join(sources))
        self.assertNotIn('.fit(', '\n'.join(sources))


if __name__=='__main__': unittest.main()
