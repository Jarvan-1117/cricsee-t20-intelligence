"""Fixed-IBW Stage 08 regression; all training/prediction fixtures are synthetic."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import probability_evaluation_fixed_ibw as stage
import probability_evaluation_fixed_ibw_visualisation as viz
from probability_fixtures import fixture, FrozenPredictor




class FixedIBWTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='joint08_synthetic_'); cls.root=Path(cls.tmp.name)
        cls.config,cls.registry,cls.data,cls.splits=fixture(cls.root)
        cls.inputs={'synthetic':'joint selection'}
        cls.upstream={'stage06':{'experiment_id':'synthetic06'},'stage07':{'experiment_id':'synthetic07'}}
        cls.before={p:stage.sha256(p) for p in cls.root.rglob('*') if p.is_file()}
        loader=stage.load_test_data
        def checked_loader(root,registry):
            protocol=json.loads((Path(root)/stage.REPORT_DIR/'frozen_protocol.json').read_text())
            if protocol['models']!=registry: raise AssertionError('Freeze must precede test access.')
            return loader(root,registry)
        with patch.object(stage,'prepare_registry',return_value=(cls.config,cls.inputs,cls.upstream,cls.registry)), \
             patch.object(stage,'input_fingerprints',return_value=(cls.inputs,cls.upstream)), \
             patch.object(stage,'load_test_data',side_effect=checked_loader), \
             patch.object(stage.base,'make_pipeline',side_effect=AssertionError('No training')):
            cls.summary=stage.run(cls.root,lambda *a:None)
        cls.out=cls.root/stage.REPORT_DIR
        cls.pred=pd.read_csv(cls.root/stage.DATA_DIR/'test_predictions.csv.gz')
        cls.metrics=pd.read_csv(cls.out/'test_metrics.csv')
        cls.bins=pd.read_csv(cls.out/'calibration_bins.csv')
        cls.ci=pd.read_csv(cls.out/'paired_comparisons.csv')

    @classmethod
    def tearDownClass(cls): plt.close('all'); cls.tmp.cleanup()

    def test_counts_no_fitting_no_upstream_writes(self):
        self.assertEqual(self.summary['models_evaluated'],32)
        self.assertEqual(self.summary['test_events'],48)
        self.assertEqual(len(self.pred),384)
        self.assertEqual(len(self.bins),320)
        self.assertEqual(len(self.ci),192)
        self.assertEqual(self.summary['training_fits'],0)
        self.assertEqual(self.summary['calibration_fits'],0)
        for path,digest in self.before.items(): self.assertEqual(stage.sha256(path),digest)

    def test_fixed_group_ignores_other_arms_validation_and_test_advantage(self):
        for arm in stage.ARM_ORDER:
            rows=copy.deepcopy(self.registry)
            for row in rows:
                row['validation_event_log_loss']=.01 if row['arm_code']==arm else .9
                row['test_log_loss']=100 if row['arm_code']==arm else 0
            self.assertEqual(set(stage.select_preferred(rows).values()),{'IBW'})

    def test_ties_deterministic_and_invalid_scores_rejected(self):
        rows=copy.deepcopy(self.registry)
        for row in rows: row['validation_event_log_loss']=.5
        self.assertEqual(set(stage.select_preferred(rows[::-1]).values()),{'IBW'})
        rows[0]['validation_event_log_loss']=np.nan
        with self.assertRaises(ValueError): stage.select_preferred(rows)

    def test_registry_missing_duplicate_quality_and_wrong_preferred(self):
        for mutation in ['missing','duplicate','quality','flag','source']:
            r=copy.deepcopy(self.registry)
            if mutation=='missing': r.pop()
            if mutation=='duplicate': r[-1]=r[0]
            if mutation=='quality': r[1]['features'].append('striker_batting_strike_rate')
            if mutation=='flag': r[0]['preferred']=not r[0]['preferred']
            if mutation=='source': r[0]['source_stage']='07'
            with self.assertRaises(ValueError): stage.validate_registry(r,self.config)

    def test_interactions_reconstructed_exactly_and_no_future_chase(self):
        stage.interactions.Extension('IBW').validate_features(self.data)
        for source,product in zip(stage.interactions.raw_features('IBW'),stage.interactions.interaction_features('IBW')):
            np.testing.assert_allclose(self.data[product],self.data[source]*.5)
        for row in self.registry:
            if row['track'].endswith('_1'): self.assertFalse(set(row['features'])&set(stage.base.CHASE))

    def test_only_test_rows_and_equal_coverage(self):
        self.assertEqual(set(self.data.split),{'test'})
        with self.assertRaises(ValueError): stage.validate_predictions(self.pred.iloc[1:],self.data)
        p=self.pred.copy(); p.loc[0,stage.base.TARGET]=1-p.loc[0,stage.base.TARGET]
        with self.assertRaises(ValueError): stage.validate_predictions(p,self.data)

    def test_handoff_once_and_selected_from_both_stages(self):
        h=json.loads((self.out/'nwc_handoff.json').read_text())
        self.assertEqual(h['status'],'requires_stage09_migration')
        pred=pd.read_csv(self.root/stage.DATA_DIR/'preferred_test_predictions.csv.gz')
        self.assertEqual(len(pred),48); self.assertFalse(pred.duplicated(stage.base.KEYS).any())
        self.assertEqual({r['arm_code'] for r in h['preferred_models']},{'IBW'})
        self.assertNotIn('attribution_probability',pred)

    def test_importance_actual_blocks_constant_na_and_negative_retained(self):
        status=pd.read_csv(self.out/'importance_status.csv')
        self.assertEqual(set(status.arm_code), {'IBW'})
        self.assertEqual(set(status.status), {'evaluated'})
        imp=pd.read_csv(self.out/'feature_importance.csv')
        self.assertNotIn('R0',set(imp.arm_code))
        self.assertIn('batting_interactions',set(imp.unit))
        for row in self.registry:
            if row['preferred'] and row['features']:
                units=stage.importance_units(row['features'])
                actual=sum([r['columns'] for r in units if r['unit_type']=='block'],[])
                self.assertCountEqual(actual,row['features'])
                self.assertEqual(len(actual),len(set(actual)))
        imp['importance_mean']=-.1
        fig=viz.importance_figure(imp,'block',status)
        self.assertTrue(all(p.get_width()<0 for ax in fig.axes for p in ax.patches)); plt.close(fig)

    def test_f1_and_gain_points_match_metrics(self):
        self.assertTrue(self.metrics.test_f1.between(0,1).all())
        m=self.metrics.set_index(['track','arm_code'])
        for row in self.ci.itertuples():
            self.assertAlmostEqual(row.gain,m.loc[(row.track,row.reference),'test_'+row.metric]-m.loc[(row.track,row.treatment),'test_'+row.metric])

    def test_visual_order_and_missing_data(self):
        try:
            fig=viz.performance_figure(self.metrics)
            labels=[t.get_text().replace(' *','') for t in fig.axes[0].get_yticklabels()]
            self.assertEqual(labels,stage.ARM_ORDER)
            with self.assertRaises(ValueError): viz.performance_figure(self.metrics.iloc[1:])
            fig=viz.calibration_figure(self.bins,['R1','IBW'],'Synthetic QA')
            self.assertEqual(len(fig.axes),4)
            fig.canvas.draw()
            with self.assertRaises(ValueError):
                viz.calibration_figure(self.bins.iloc[1:],['R0'],'Synthetic QA')
            with self.assertRaises(ValueError):
                viz.comparison_figure(self.ci.iloc[4:],'event_log_loss')
        finally:
            plt.close('all')

    def test_one_bootstrap_draw_per_gender(self):
        with patch.object(stage,'draw_match_counts',wraps=stage.draw_match_counts) as draws:
            stage.bootstrap_comparisons(self.pred,self.config)
            self.assertEqual(draws.call_count,2)

    def test_changed_model_rejected(self):
        r=copy.deepcopy(self.registry); r[0]['model_sha256']='invalid'
        with self.assertRaisesRegex(ValueError,'changed'): stage.predict_frozen(self.root,self.data,r)

    def test_importance_rejects_non_ibw_preferred_flags(self):
        rows=copy.deepcopy(self.registry)
        for r in rows: r['preferred']=r['arm_code']=='R0'
        with self.assertRaisesRegex(ValueError, 'fixed-IBW'):
            stage.permutation_importance(self.root,self.data,rows,self.pred,self.config)

    def test_verified_read_and_tamper(self):
        with patch.object(stage,'input_fingerprints',return_value=(self.inputs,self.upstream)), \
             patch.object(stage,'read_config',return_value=self.config), \
             patch.object(stage,'build_registry',return_value=self.registry):
            self.assertEqual(stage.verified_output(self.root)['evaluation_id'],self.summary['evaluation_id'])
        with patch.object(stage,'input_fingerprints',return_value=({},self.upstream)):
            with self.assertRaisesRegex(ValueError,'stale'): stage.verified_output(self.root)


    def test_build_registry_deduplicates_rbw_and_rejects_changed_control(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            config,rows,_,_=fixture(root)
            baseout=root/stage.base.REPORT_DIR
            extout=root/stage.interactions.REPORT_DIR
            (baseout/'models').mkdir(parents=True); extout.mkdir(parents=True)
            main=[]; extensions=[]
            for r in rows:
                record={k:r[k] for k in ['track','arm','arm_code','family','candidate','validation_event_log_loss']}
                record['parameters']=json.dumps(r['parameters'])
                if r['source_stage']=='06':
                    name=r['track']+'__'+r['arm_code']+'.joblib'
                    joblib.dump(FrozenPredictor(r['features']),baseout/'models'/name)
                    record['model_file']=name
                    main.append(dict(record))
                    if r['arm_code']=='RBW':
                        record.update(model_path=stage.base.REPORT_DIR+'/models/'+name,
                                      model_sha256=stage.sha256(baseout/'models'/name))
                        extensions.append(dict(record))
                else:
                    record.update(model_path=r['model_path'],model_sha256=r['model_sha256'])
                    extensions.append(record)
            pd.DataFrame(main).to_csv(baseout/'selected_models.csv',index=False)
            pd.DataFrame(extensions).to_csv(extout/'selected_models.csv',index=False)
            registry=stage.build_registry(root,config)
            self.assertEqual(len(registry),32)
            self.assertEqual(sum(r['arm_code']=='RBW' for r in registry),4)
            self.assertEqual(stage.select_preferred(registry),stage.select_preferred(rows))
            extensions[0]['validation_event_log_loss']+=.1
            pd.DataFrame(extensions).to_csv(extout/'selected_models.csv',index=False)
            with self.assertRaisesRegex(ValueError,'RBW control'): stage.build_registry(root,config)

    def test_mismatched_upstream_and_missing_outputs_stop(self):
        with patch.object(stage.base,'verified_output',return_value={'experiment_id':'new06'}), \
             patch.object(stage.interactions,'verified_output',return_value={'upstream_stage06_id':'old06'}):
            with self.assertRaisesRegex(ValueError,'current Stage 06'): stage.input_fingerprints(self.root)
        with tempfile.TemporaryDirectory() as d,patch.object(stage,'run') as run:
            with self.assertRaisesRegex(FileNotFoundError,'RUN_EVALUATION'): stage.verified_output(d)
            run.assert_not_called()

    def test_new_handoff_accepted_by_fixed_ibw_stage09(self):
        import nwc_attribution as nwc
        handoff=json.loads((self.out/'nwc_handoff.json').read_text())
        protocol=json.loads((self.out/'frozen_protocol.json').read_text())
        registry=nwc.validate_handoff(handoff,protocol,self.summary)
        self.assertEqual(len(registry),4)
        self.assertEqual({r['arm_code'] for r in registry.values()},{'IBW'})

    def test_real_xgboost_pipeline_predicts_with_joint_registry(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            config,rows,data,_=fixture(root)
            row=next(r for r in rows if r['track']==stage.TRACKS[0] and r['arm_code']=='IBW')
            part=data[data.gender.eq('female')&data.match_innings.eq(1)]
            model=stage.base.make_pipeline('xgboost',dict(n_estimators=2,learning_rate=.1,max_depth=2),
                row['features'],dict(seed=32933,threads=1))
            model.fit(part[row['features']],part[stage.base.TARGET])  # Synthetic data only.
            path=root/row['model_path']; joblib.dump(model,path)
            row.update(family='xgboost',model_sha256=stage.sha256(path))
            pred=stage.predict_frozen(root,data,rows,lambda *a:None)
            saved=pred[pred.track.eq(row['track'])&pred.arm_code.eq('IBW')]
            np.testing.assert_allclose(saved.raw_model_probability,
                stage.base.probability(model,part[row['features']]),rtol=0,atol=1e-15)


if __name__=='__main__': unittest.main()
