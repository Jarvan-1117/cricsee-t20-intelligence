"""Shared metric and current fixed-IBW integration checks using synthetic data."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import probability_evaluation_fixed_ibw as stage
from probability_fixtures import fixture, FrozenPredictor






class MathematicsTests(unittest.TestCase):
    def test_common_helpers_have_no_execution_or_selection_entry_point(self):
        for name in ['run', 'read_config', 'expected_config', 'build_registry',
                     'prepare_registry', 'input_fingerprints', 'verified_output']:
            self.assertFalse(hasattr(stage.common, name), name)

    def test_bins_boundaries_and_empty_bins(self):
        table,ece=stage.calibration_table([0,1,1],[0,.1,1],[1,2,3])
        self.assertEqual(table.events.tolist(),[1,1,0,0,0,0,0,0,0,1])
        self.assertTrue(table.loc[table.events.eq(0),'observed_win_rate'].isna().all())
        self.assertAlmostEqual(ece,.3)

    def test_event_weighted_ece_not_mean_of_bins(self):
        table,ece=stage.calibration_table([0,0,0,1],[.05,.05,.05,.95],[1,1,2,2])
        self.assertAlmostEqual(ece,.05)
        self.assertEqual(table.matches.sum(),3)  # Same match may enter more than one bin.

    def test_invalid_probabilities_rejected(self):
        for p in [[np.nan],[-.1],[1.1]]:
            with self.assertRaises(ValueError): stage.calibration_table([1],p,[1])

    def test_paired_gain_sign_and_weighting(self):
        ref=pd.DataFrame({'events':[1,3],'log_loss_sum':[.6,1.8],'brier_sum':[.2,.6]},index=[1,2])
        treatment=pd.DataFrame({'events':[1,3],'log_loss_sum':[.1,2.1],'brier_sum':[.1,.9]},index=[1,2])
        counts=np.array([[2,0],[0,2],[1,1]],dtype=float)
        gains=stage.paired_gain(ref,treatment,counts)
        self.assertAlmostEqual(gains['event_log_loss'][0],.05)
        self.assertAlmostEqual(gains['match_log_loss'][0],.2)
        np.testing.assert_allclose(gains['event_log_loss'][1],[.5,-.1,.05])
        self.assertLess(gains['brier'][0],0)

    def test_bootstrap_repeated_matches_and_seed(self):
        a=stage.draw_match_counts(7,50,32933); b=stage.draw_match_counts(7,50,32933)
        np.testing.assert_array_equal(a,b)
        self.assertTrue((a.sum(axis=1)==7).all()); self.assertGreater(a.max(),1)

    def test_pair_coverage_mismatch_rejected(self):
        frame=pd.DataFrame({'events':[1,2],'log_loss_sum':[1,2],'brier_sum':[1,2]},index=[1,2])
        with self.assertRaises(ValueError): stage.paired_gain(frame,frame.iloc[::-1],np.ones((2,2)))

    def test_missing_results_never_evaluates(self):
        with tempfile.TemporaryDirectory() as d, patch.object(stage,'run') as run:
            with self.assertRaisesRegex(FileNotFoundError,'RUN_EVALUATION'): stage.verified_output(d)
            run.assert_not_called()

    def test_config_rejects_calibrator_fit(self):
        c=stage.read_config(ROOT); c['fit_calibrators']=True
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'config'; target.mkdir()
            (Path(d)/stage.CONFIG_PATH).write_text(json.dumps(c))
            with self.assertRaises(ValueError): stage.read_config(d)

    def test_permutation_preserves_block_donors_and_source(self):
        x=pd.DataFrame({'a':[1.,2.,3.], 'b':[10.,20.,30.], 'phase':['a','b','c'], 'fixed':[9,8,7]})
        original=x.copy(deep=True)
        shuffled=stage.permuted_input(x,['a','b','phase'],np.array([2,0,1]))
        pd.testing.assert_frame_equal(x,original)
        np.testing.assert_array_equal(shuffled.b,10*shuffled.a)
        self.assertEqual(shuffled.phase.tolist(),['c','a','b'])
        self.assertEqual(shuffled.fixed.tolist(),[9,8,7])
        with self.assertRaises(ValueError): stage.permuted_input(x,['a'],[0,0,2])

    def test_protocol_is_eight_arms_with_fixed_ibw(self):
        config=stage.read_config(ROOT)
        self.assertEqual(config['arms'],['R0','R1','RB','RW','RBW','IB','IW','IBW'])
        self.assertEqual(config['primary_pair'],['RBW','R1'])
        self.assertEqual(config['preferred_arm'],'IBW')
        self.assertEqual(config['importance']['repeats'],10)
        self.assertTrue(stage.REPORT_DIR.endswith('/resource_sums_v2_fixed_ibw_v1'))
        self.assertFalse(any(hasattr(stage,name) for name in ['e1','e2','e3']))


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='stage08_synthetic_')
        cls.root=Path(cls.temp.name)
        cls.config,cls.registry,cls.data,cls.splits=fixture(cls.root)
        cls.inputs={'synthetic_only':'no real test scoring'}
        cls.upstream={'stage06':{'experiment_id':'synthetic-stage06'},'stage07':{'experiment_id':'synthetic-stage07'}}
        cls.before={p.relative_to(cls.root).as_posix():stage.sha256(p) for p in cls.root.rglob('*') if p.is_file()}
        original_loader=stage.load_test_data
        def checked_loader(root,registry):
            protocol=json.loads((Path(root)/stage.REPORT_DIR/'frozen_protocol.json').read_text())
            if protocol['models']!=registry: raise AssertionError('Protocol must precede test access.')
            return original_loader(root,registry)
        with patch.object(stage,'prepare_registry',return_value=(cls.config,cls.inputs,cls.upstream,cls.registry)), \
             patch.object(stage,'input_fingerprints',return_value=(cls.inputs,cls.upstream)), \
             patch.object(stage,'load_test_data',side_effect=checked_loader), \
             patch.object(stage.base,'make_pipeline',side_effect=AssertionError('No training pipeline')):
            cls.summary=stage.run(cls.root,progress=lambda *args:None)
        cls.out=cls.root/stage.REPORT_DIR
        cls.pred=pd.read_csv(cls.root/stage.DATA_DIR/'test_predictions.csv.gz')
        cls.metrics=pd.read_csv(cls.out/'test_metrics.csv')
        cls.cal=pd.read_csv(cls.out/'calibration_bins.csv')
        cls.ci=pd.read_csv(cls.out/'paired_comparisons.csv')

    @classmethod
    def tearDownClass(cls):
        plt.close('all'); cls.temp.cleanup()

    def test_upstream_files_unchanged(self):
        for path,digest in self.before.items(): self.assertEqual(stage.sha256(self.root/path),digest)

    def test_counts_and_no_fits(self):
        self.assertEqual(self.summary['models_evaluated'],32)
        self.assertEqual(self.summary['training_fits'],0); self.assertEqual(self.summary['calibration_fits'],0)
        self.assertEqual(self.summary['test_matches'],8); self.assertEqual(self.summary['test_events'],48)
        self.assertEqual(len(self.metrics),32); self.assertEqual(len(self.cal),320); self.assertEqual(len(self.ci),192)
        self.assertEqual(len(self.pred),384)

    def test_direct_quality_and_extra_features_rejected(self):
        for column in ['striker_batting_strike_rate','non_striker_batting_average',
                       'bowler_bowling_economy','audit_waiting_count','unexpected_input']:
            registry=copy.deepcopy(self.registry)
            registry[1]['features'].append(column)
            with self.assertRaisesRegex(ValueError,'allowlist'):
                stage.validate_registry(registry,self.config)

    def test_importance_complete_reproducible_and_no_source_mutation(self):
        before=self.data.copy(deep=True)
        result,details=stage.permutation_importance(self.root,self.data,self.registry,self.pred,self.config,
                                                  progress=lambda *a:None)
        saved=pd.read_csv(self.out/'feature_importance.csv')
        saved_details=pd.read_csv(self.root/stage.DATA_DIR/'importance_repeats.csv.gz')
        pd.testing.assert_frame_equal(result,saved,check_dtype=False,atol=1e-12)
        pd.testing.assert_frame_equal(details,saved_details,check_dtype=False,atol=1e-12)
        pd.testing.assert_frame_equal(self.data,before)
        self.assertEqual(set(result.arm_code),{'IBW'})
        self.assertTrue(result.repeats.eq(10).all())
        self.assertEqual(len(result[result.unit_type.eq('block')]),20)
        for row in self.registry:
            if row['preferred']:
                part=result[result.track.eq(row['track']) & result.unit_type.eq('feature')]
                self.assertEqual(set(part.unit),set(row['features']))
                # The fake predictor uses pre_score only: unused inputs must have zero importance.
                self.assertTrue(part.loc[part.unit.ne('pre_score'),'importance_mean'].eq(0).all())
        independently=details.permuted_log_loss-details.baseline_log_loss
        np.testing.assert_allclose(details.importance,independently,rtol=0,atol=1e-15)
        grouped=details.groupby(['track','unit_type','unit']).importance
        summary=result.set_index(['track','unit_type','unit'])
        np.testing.assert_allclose(grouped.std().sort_index(),summary.importance_std.sort_index(),atol=1e-12)

    def test_importance_refuses_reordered_or_changed_baseline(self):
        pred=self.pred.copy()
        pred.loc[pred.arm_code.eq('IBW'),'raw_model_probability']=.123
        with self.assertRaisesRegex(ValueError,'baseline predictions'):
            stage.permutation_importance(self.root,self.data,self.registry,pred,self.config,progress=lambda *a:None)

    def test_main_registry_is_not_selected_by_test_or_global_validation_rank(self):
        registry=copy.deepcopy(self.registry)
        for row in registry:
            row['validation_event_log_loss']=.9 if row['arm_code']=='IBW' else .01
        stage.validate_registry(registry,self.config)
        self.assertEqual(sum(r['preferred'] for r in registry),4)

    def test_preferred_handoff_covers_test_once(self):
        p=pd.read_csv(self.root/stage.DATA_DIR/'preferred_test_predictions.csv.gz')
        self.assertEqual(len(p),48); self.assertFalse(p.duplicated(stage.base.KEYS).any())
        for track,g in p.groupby('track'): self.assertEqual(set(g.arm_code),{stage.PREFERRED_ARM})
        self.assertNotIn('attribution_probability',p.columns)

    def test_match_and_event_metrics_independent_calculation(self):
        for row in self.metrics.itertuples():
            p=self.pred[self.pred.track.eq(row.track)&self.pred.arm_code.eq(row.arm_code)]
            y=p[stage.base.TARGET].to_numpy(); prob=p.raw_model_probability.to_numpy()
            loss=-(y*np.log(prob)+(1-y)*np.log1p(-prob))
            self.assertAlmostEqual(loss.mean(),row.test_event_log_loss)
            self.assertAlmostEqual(((y-prob)**2).mean(),row.test_brier)

    def test_bootstrap_points_match_metric_differences(self):
        m=self.metrics.set_index(['track','arm_code'])
        for row in self.ci.itertuples():
            expected=m.loc[(row.track,row.reference),'test_'+row.metric]-m.loc[(row.track,row.treatment),'test_'+row.metric]
            self.assertAlmostEqual(expected,row.gain)
            self.assertLessEqual(row.ci_lower,row.ci_upper)
            self.assertEqual(row.bootstrap_replicates,2000)
        self.assertEqual(len(self.ci[self.ci.comparison_role.eq('primary')]),16)

    def test_bootstrap_one_draw_matrix_per_gender(self):
        original=stage.draw_match_counts
        with patch.object(stage,'draw_match_counts',wraps=original) as draws:
            result,_=stage.bootstrap_comparisons(self.pred,self.config)
            self.assertEqual(draws.call_count,2)
        pd.testing.assert_frame_equal(result,self.ci,check_dtype=False,atol=1e-12,rtol=1e-10)

    def test_empty_bins_and_raw_zero_one_not_changed(self):
        p=self.pred.copy()
        p['raw_model_probability']=np.where(p[stage.base.TARGET].eq(1),1.,0.)
        m,c=stage.evaluate_metrics(p,self.config)
        self.assertTrue(m.test_brier.eq(0).all())
        self.assertTrue(np.isfinite(m.test_event_log_loss).all())
        self.assertTrue(c.loc[c.events.eq(0),'mean_probability'].isna().all())
        self.assertEqual(set(p.raw_model_probability),{0.,1.})

    def test_auc_unavailable_for_single_class(self):
        p=self.pred.copy(); p[stage.base.TARGET]=1
        m,_=stage.evaluate_metrics(p,self.config)
        self.assertTrue(m.test_auc.isna().all())

    def test_non_test_and_missing_rows_rejected(self):
        data=self.data.copy(); data.loc[0,'split']='train'
        with self.assertRaises(ValueError): stage.common.validate_test_data(data,self.splits,self.registry)
        with self.assertRaises(ValueError): stage.common.validate_test_data(self.data.iloc[1:],self.splits,self.registry)

    def test_first_innings_future_target_rejected(self):
        data=self.data.copy(); data.loc[data.match_innings.eq(1),'target_runs']=150
        with self.assertRaisesRegex(ValueError,'chase'): stage.common.validate_test_data(data,self.splits,self.registry)

    def test_missing_or_wrong_prediction_labels_rejected(self):
        with self.assertRaises(ValueError): stage.validate_predictions(self.pred.iloc[1:],self.data)
        p=self.pred.copy(); p.loc[0,stage.base.TARGET]=1-p.loc[0,stage.base.TARGET]
        with self.assertRaisesRegex(ValueError,'labels'): stage.validate_predictions(p,self.data)

    def test_changed_preferred_registry_rejected(self):
        r=copy.deepcopy(self.registry); r[0]['preferred']=not r[0]['preferred']
        with self.assertRaisesRegex(ValueError,'Preferred'): stage.validate_registry(r,self.config)
        r=copy.deepcopy(self.registry)
        r[0]['validation_event_log_loss']=.001
        stage.validate_registry(r,self.config)  # Cross-group best must NOT replace fixed IBW.
        self.assertEqual({row['arm_code'] for row in r if row['preferred']},{'IBW'})

    def test_model_hash_change_rejected(self):
        r=copy.deepcopy(self.registry); r[0]['model_sha256']='invalid'
        with self.assertRaisesRegex(ValueError,'changed'): stage.predict_frozen(self.root,self.data,r)

    def test_model_feature_order_checked(self):
        r=copy.deepcopy(self.registry); r[1]['features']=r[1]['features'][::-1]
        with self.assertRaisesRegex(ValueError,'feature order'):
            stage.predict_frozen(self.root,self.data,r,progress=lambda *a:None)

    def test_read_mode_and_tamper_checks(self):
        with patch.object(stage,'input_fingerprints',return_value=(self.inputs,self.upstream)), \
             patch.object(stage,'read_config',return_value=self.config), \
             patch.object(stage,'build_registry',return_value=self.registry):
            self.assertEqual(stage.verified_output(self.root)['evaluation_id'],self.summary['evaluation_id'])
        with patch.object(stage,'input_fingerprints',return_value=({'changed':True},self.upstream)):
            with self.assertRaisesRegex(ValueError,'stale'): stage.verified_output(self.root)
        path=self.out/'test_metrics.csv'; original=path.read_bytes()
        try:
            path.write_bytes(original+b'\n')
            with patch.object(stage,'input_fingerprints',return_value=(self.inputs,self.upstream)):
                with self.assertRaisesRegex(ValueError,'artifact changed'): stage.verified_output(self.root)
        finally: path.write_bytes(original)

if __name__=='__main__': unittest.main()
