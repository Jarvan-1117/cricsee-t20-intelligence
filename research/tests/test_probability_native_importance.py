"""Native importance alignment and contract tests; synthetic models only."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import probability_native_importance as native


def fixture(family='xgboost', track='female_innings_1'):
    row = dict(track=track, arm_code='RW', family=family, model_sha256='model',
        model_path=track+'.joblib', features=['pre_score', 'no_legal_ball_yet', 'innings_phase'])
    names = np.array(['pre_score', 'no_legal_ball_yet', 'innings_phase_powerplay', 'innings_phase_middle'])
    booster = SimpleNamespace(feature_names=None, num_features=lambda: 3,
        get_score=lambda importance_type: {'f0':2., 'f2':4.} if importance_type=='gain' else {'f0':1., 'f2':2.})
    estimator = SimpleNamespace(n_features_in_=3, get_booster=lambda: booster)
    if family=='random_forest': estimator.feature_importances_=np.array([.2,.3,.5])
    prep = SimpleNamespace(get_feature_names_out=lambda: names,
        named_transformers_={'phase':SimpleNamespace(get_feature_names_out=lambda cols:names[2:])})
    model = SimpleNamespace(feature_names_in_=np.array(row['features']),
        named_steps={'preprocess':prep,'nonconstant':SimpleNamespace(get_support=lambda:np.array([True,False,True,True])),
                     'model':estimator})
    return model, row


class NativeImportanceTests(unittest.TestCase):
    def test_xgboost_index_mapping_zero_and_missing_are_distinct(self):
        model,row=fixture()
        records,status=native.extract_model(model,row)
        self.assertEqual(status['metric'],'xgboost_mean_split_gain')
        self.assertEqual([r['importance'] for r in records],[2.,None,0.,4.])
        self.assertEqual(records[1]['status'],'constant_removed')
        self.assertFalse(records[2]['used_in_split'])
        self.assertEqual(records[2]['raw_feature'],'innings_phase')
        self.assertEqual(records[3]['raw_feature'],'innings_phase')
        self.assertEqual(len(records),4)  # No misleading sum across phase dummies.

    def test_rf_mdi_is_normalised_and_not_gain(self):
        model,row=fixture('random_forest')
        records,status=native.extract_model(model,row)
        self.assertEqual(status['metric'],'random_forest_normalized_mdi')
        self.assertAlmostEqual(sum(r['importance'] for r in records if r['retained']),1.)
        self.assertTrue(all(r['used_in_split'] is None for r in records))

    def test_unsupported_and_not_applicable_models_have_no_zero_rows(self):
        for family,expected in [('hist_gradient_boosting','unsupported_public_interface'),
                                ('logistic_regression','not_applicable_linear_model')]:
            model,row=fixture(family)
            records,status=native.extract_model(model,row)
            self.assertEqual(records,[])
            self.assertEqual(status['status'],expected)

    def test_mapping_dimension_and_named_keys_rejected(self):
        for kind in ['raw','dimension','name','key']:
            model,row=fixture()
            if kind=='raw': model.feature_names_in_=model.feature_names_in_[::-1]
            if kind=='dimension': model.named_steps['model'].n_features_in_=100
            booster=model.named_steps['model'].get_booster()
            if kind=='name': booster.feature_names=['wrong','order','names']
            if kind=='key': booster.get_score=lambda importance_type:{'f99':1.}
            with self.assertRaises(ValueError): native.extract_model(model,row)

    def test_named_xgboost_features_supported_in_exact_order(self):
        model,row=fixture()
        booster=model.named_steps['model'].get_booster()
        booster.feature_names=['pre_score','innings_phase_powerplay','innings_phase_middle']
        booster.get_score=lambda importance_type:{'pre_score':2.}
        records,_=native.extract_model(model,row)
        self.assertEqual([r['importance'] for r in records],[2.,None,0.,0.])

    def test_invalid_rf_scores_are_rejected(self):
        for values in [[1,1,1],[.5,-.1,.6],[float('nan'),0,0],[1,0]]:
            model,row=fixture('random_forest')
            model.named_steps['model'].feature_importances_=np.array(values)
            with self.assertRaises(ValueError): native.extract_model(model,row)

    def test_export_load_no_predictions_and_upstream_files_untouched(self):
        families=['xgboost','hist_gradient_boosting','logistic_regression','random_forest']
        fixtures=[fixture(f,t) for f,t in zip(families,native.stage.TRACKS)]
        registry=[r for _,r in fixtures]
        models={r['model_path']:m for m,r in fixtures}
        evaluation={'evaluation_id':'evaluation','preferred':{r['track']:r['arm_code'] for r in registry}}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            protected=root/'protected.txt'; protected.write_text('do not change')
            digest=native.sha256(protected)
            prepared=(evaluation,registry,{'protected.txt':digest},'native')
            with patch.object(native,'prepare',return_value=prepared), \
                 patch.object(native.joblib,'load',side_effect=lambda p:models[p.name]), \
                 patch.object(native.stage.base,'probability',side_effect=AssertionError('No prediction')), \
                 patch.object(native.stage.base,'make_pipeline',side_effect=AssertionError('No fitting')):
                summary=native.run(root)
                loaded,features,status=native.load_tables(root)
                self.assertEqual(summary,loaded)
                self.assertEqual(summary['exported_models'],2)
                self.assertEqual(len(status),4)
                self.assertEqual(len(features),8)
                self.assertEqual(native.sha256(protected),digest)
                self.assertTrue(features.loc[features.status.eq('constant_removed'),'importance'].isna().all())
                with patch.object(native,'sha256',return_value='tampered'):
                    with self.assertRaisesRegex(ValueError,'artifact changed'): native.load_tables(root)
                wrong=(evaluation,registry,{},'changed')
                with patch.object(native,'prepare',return_value=wrong):
                    with self.assertRaisesRegex(ValueError,'stale'): native.load_tables(root)

    def test_missing_results_never_automatically_export(self):
        with tempfile.TemporaryDirectory() as d,patch.object(native,'run') as run:
            with self.assertRaisesRegex(FileNotFoundError,'RUN_NATIVE_IMPORTANCE'): native.load_tables(d)
            run.assert_not_called()

    def test_notebook_supplement_is_read_only_and_keeps_permutation_figures(self):
        nb=json.loads((ROOT/'notebooks/08_probability_evaluation.ipynb').read_text(encoding='utf-8'))
        source='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        self.assertIn('RUN_NATIVE_IMPORTANCE = False',source)
        self.assertIn("display(figures['feature_importance.png'])",source)
        self.assertIn("display(figures['resource_block_importance.png'])",source)
        cell=next(''.join(c['source']) for c in nb['cells'] if 'native_summary, native_features' in ''.join(c['source']))
        supplement=cell[cell.index('if RUN_NATIVE_IMPORTANCE:'):]
        records,status=native.extract_model(*fixture())
        outputs=[]
        namespace=dict(RUN_NATIVE_IMPORTANCE=False,native_importance=native,ROOT=ROOT,stage=native.stage,
            display=outputs.append,show_details=lambda *args:outputs.append(args))
        with patch.object(native,'load_tables',return_value=({'native_importance_id':'test','notes':native.NOTES},pd.DataFrame(records),pd.DataFrame([status]))), \
             patch.object(native,'run') as run:
            exec(compile(supplement,'native_notebook','exec'),namespace)
            run.assert_not_called()
        self.assertEqual(len(outputs),3)


if __name__=='__main__': unittest.main()
