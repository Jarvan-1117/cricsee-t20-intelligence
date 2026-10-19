"""XGBoost integration: unchanged controls, train-only fitting, shared selection."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import model_comparison as stage
from test_model_comparison import fixture


class XGBoostTests(unittest.TestCase):
    def test_grid_and_fit_count(self):
        config = stage.read_config(ROOT)
        self.assertEqual(stage.FAMILIES, ['logistic_regression', 'random_forest', 'hist_gradient_boosting', 'xgboost'])
        self.assertEqual(config['candidate_grid']['logistic_regression'], [{'C':.01},{'C':.1},{'C':1.}])
        self.assertEqual(config['candidate_grid']['random_forest'], [
            {'n_estimators':200,'max_depth':8,'min_samples_leaf':100},
            {'n_estimators':200,'max_depth':14,'min_samples_leaf':50},
            {'n_estimators':200,'max_depth':None,'min_samples_leaf':30}])
        self.assertEqual(config['candidate_grid']['hist_gradient_boosting'], [
            {'max_iter':200,'learning_rate':.05,'max_leaf_nodes':15},
            {'max_iter':200,'learning_rate':.1,'max_leaf_nodes':15},
            {'max_iter':200,'learning_rate':.05,'max_leaf_nodes':31}])
        self.assertEqual(len(config['candidate_grid']['xgboost']), 3)
        self.assertEqual(4*len(stage.ARMS)*sum(map(len, config['candidate_grid'].values()))+4, 196)
        self.assertEqual(config['split_fractions'], [.7,.15,.15])
        self.assertEqual(config['selection_metric'], 'validation_event_log_loss')
        self.assertEqual(config['preferred_arm'], 'both_resources')
        self.assertFalse(config['refit_train_validation'])

    def test_cpu_seed_thread_limit_and_no_early_stopping(self):
        config, arms, _, _, _ = fixture()
        pipeline = stage.make_pipeline('xgboost', config['candidate_grid']['xgboost'][0],
                                       stage.track_columns(arms['both_resources'], 1), config)
        model = pipeline.named_steps['model']
        params = model.get_params()
        self.assertEqual(params['objective'], 'binary:logistic')
        self.assertEqual(params['eval_metric'], 'logloss')
        self.assertEqual(params['tree_method'], 'hist')
        self.assertEqual(params['device'], 'cpu')
        self.assertEqual(params['random_state'], config['seed'])
        self.assertEqual(params['n_jobs'], config['threads'])
        self.assertIsNone(params['early_stopping_rounds'])

    def test_shared_preprocessing_no_validation_refit_or_eval_set(self):
        config, arms, _, _, data = fixture()
        columns = stage.track_columns(arms['both_resources'], 2)
        train = data[data.gender.eq('female') & data.match_innings.eq(2) & data.split.eq('train')]
        val = data[data.gender.eq('female') & data.match_innings.eq(2) & data.split.eq('validation')].copy()
        val['pre_score'] = 1e6
        pipeline = stage.make_pipeline('xgboost', config['candidate_grid']['xgboost'][0], columns, config)
        model = pipeline.named_steps['model']
        with threadpool_limits(limits=1), patch.object(model, 'fit', wraps=model.fit) as fit:
            pipeline.fit(train[columns], train[stage.TARGET])
            self.assertEqual(fit.call_count, 1)
            self.assertNotIn('eval_set', fit.call_args.kwargs)
            self.assertEqual(len(fit.call_args.args[0]), len(train))
        prep = pipeline.named_steps['preprocess'].named_transformers_['numeric']
        self.assertNotIn('scale', prep.named_steps)
        before = prep.named_steps['impute'].statistics_.copy()
        with threadpool_limits(limits=1):
            probability = stage.probability(pipeline, val[columns])
        np.testing.assert_array_equal(before, prep.named_steps['impute'].statistics_)
        self.assertTrue(np.isfinite(probability).all())
        self.assertEqual(probability.dtype, np.dtype('float64'))
        self.assertTrue(((probability>=0)&(probability<=1)).all())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'xgb.joblib'
            joblib.dump(pipeline,path)
            with threadpool_limits(limits=1):
                restored=stage.probability(joblib.load(path),val[columns])
            np.testing.assert_array_equal(probability,restored)
            export=Path(folder)/'probability.csv'
            pd.DataFrame({'probability':probability}).to_csv(export,index=False)
            reloaded=pd.read_csv(export).probability.to_numpy()
            np.testing.assert_allclose(reloaded,probability,atol=1e-15,rtol=0)

    def test_xgboost_can_win_by_validation_but_ties_preserve_original_order(self):
        records=[dict(family=f,candidate=0,validation_event_log_loss=.5) for f in stage.FAMILIES]
        records[-1]['validation_event_log_loss']=.4
        self.assertIs(stage.best_record(records),records[-1])
        records[-1]['validation_event_log_loss']=.5
        self.assertIs(stage.best_record(records[::-1]),records[0])

    def test_grid_cannot_override_fixed_fit_policy(self):
        original=stage.read_config(ROOT)
        for field,value in [('early_stopping_rounds',10),('eval_set',[]),('device','cuda'),('random_state',999)]:
            config=deepcopy(original)
            config['candidate_grid']['xgboost'][0][field]=value
            with tempfile.TemporaryDirectory() as folder:
                p=Path(folder)/'config/model_comparison.json'
                p.parent.mkdir()
                p.write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError,'parameter override'):
                    stage.read_config(folder)


if __name__=='__main__': unittest.main()
