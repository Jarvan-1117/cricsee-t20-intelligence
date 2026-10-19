"""Stage 06 isolation, reproducibility, selection and small synthetic end-to-end tests."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import model_comparison as stage


def fixture():
    config = stage.read_config(ROOT)
    config['threads'] = 1
    config['candidate_grid'] = {'logistic_regression': [{'C': .1}],
        'random_forest': [{'n_estimators': 2, 'max_depth': 3, 'min_samples_leaf': 2}],
        'hist_gradient_boosting': [{'max_iter': 2, 'learning_rate': .1, 'max_leaf_nodes': 3}],
        'xgboost': [{'n_estimators': 2, 'learning_rate': .1, 'max_depth': 2}]}
    contract = stage.resources.feature_contract(stage.game_state.feature_contract())
    arms = stage.resolve_arms(contract)
    cohort = pd.DataFrame([{'match_id': offset + i, 'gender': gender,
        'match_date': f'2020-01-{i + 1:02}', 'source_events': 4,
        'included_in_resource_sample': True}
        for gender, offset in [('female', 100), ('male', 200)] for i in range(20)])
    splits = stage.chronological_splits(cohort, config['split_fractions'])
    rng = np.random.default_rng(42); rows = []
    for match in splits.itertuples(index=False):
        for innings in [1, 2]:
            for event in [1, 2]:
                row = {c: float(rng.normal()) for c in set(sum(arms.values(), []))}
                for c in stage.resources.BAT_FEATURES + stage.resources.BOWL_FEATURES:
                    row[c] = abs(row[c])
                row[stage.resources.BAT_FEATURES[-1]] = 0.
                row.update(match_id=match.match_id, gender=match.gender, match_date=match.match_date,
                    split=match.split, match_innings=innings, event_sequence=event,
                    source_row=len(rows), innings_phase='powerplay' if event == 1 else 'middle',
                    first_batting_team_win=int(match.match_id % 2))
                if innings == 1:
                    row.update({c: np.nan for c in stage.CHASE})
                rows.append(row)
    full = pd.DataFrame(rows)
    return config, arms, cohort, splits, full[full.split.ne('test')].copy()


class SplitAndIsolationTests(unittest.TestCase):
    def setUp(self):
        self.config, self.arms, self.cohort, self.splits, self.data = fixture()

    def test_five_groups_and_nested_controls(self):
        self.assertEqual(len(stage.ARM_CODES), 5)
        self.assertEqual(self.arms['constant'], [])
        self.assertEqual(set(self.arms['both_resources']) - set(self.arms['state']),
                         set(stage.resources.BAT_FEATURES + stage.resources.BOWL_FEATURES))
        self.assertEqual(self.arms['batting_resources'], self.arms['state'] + stage.resources.BAT_FEATURES)
        self.assertEqual(self.arms['bowling_resources'], self.arms['state'] + stage.resources.BOWL_FEATURES)

    def test_direct_quality_evidence_or_aliases_cannot_enter_training(self):
        forbidden = stage.resources.quality.feature_contract()['current_player_block'] + ['striker_quality_copy', 'audit_current_bowler_capacity']
        for field in forbidden:
            arms = deepcopy(self.arms); arms['both_resources'].append(field)
            with self.assertRaises(ValueError): stage.validate_development(self.data, arms)

    def test_legacy_contract_rejected(self):
        contract = stage.resources.feature_contract(stage.game_state.feature_contract())
        contract['contract_version'] = 'legacy'
        with self.assertRaises(ValueError): stage.resolve_arms(contract)

    def test_preflight_records_identical_event_keys_for_all_arms(self):
        report = stage.preflight_audit(self.data, self.arms, self.splits, self.config)
        self.assertFalse(report['training_started'])
        rows = pd.DataFrame(report['rows'])
        self.assertEqual(len(rows), 4 * 2 * 5)
        self.assertEqual(rows.groupby(['track', 'split']).event_keys_sha256.nunique().max(), 1)
        self.assertEqual(rows.groupby(['track', 'split']).events.nunique().max(), 1)
        self.assertEqual(len(report['excluded_direct_quality_fields']), 19)

    def test_outcome_independent_date_splits(self):
        changed = self.cohort.sample(frac=1, random_state=5).assign(first_batting_team_win=99)
        pd.testing.assert_frame_equal(self.splits, stage.chronological_splits(changed, [.7, .15, .15]))

    def test_equal_date_groups_never_split(self):
        c = self.cohort.copy(); c['match_date'] = c.match_date.str.slice(0, 8) + ((c.match_id % 20) // 3 + 1).astype(str).str.zfill(2)
        s = stage.chronological_splits(c, [.7, .15, .15])
        self.assertEqual(s.groupby(['gender', 'match_date']).split.nunique().max(), 1)
        for _, g in s.groupby('gender'):
            self.assertLess(g[g.split.eq('train')].match_date.max(), g[g.split.eq('validation')].match_date.min())
            self.assertLess(g[g.split.eq('validation')].match_date.max(), g[g.split.eq('test')].match_date.min())

    def test_reviews_excluded_from_every_arm(self):
        c = self.cohort.copy(); c.loc[0, 'included_in_resource_sample'] = False
        self.assertNotIn(c.match_id.iloc[0], stage.chronological_splits(c, [.7, .15, .15]).match_id.tolist())

    def test_duplicate_match_rejected(self):
        with self.assertRaises(ValueError):
            stage.chronological_splits(pd.concat([self.cohort, self.cohort.iloc[:1]]), [.7, .15, .15])

    def test_insufficient_dates_rejected(self):
        with self.assertRaises(ValueError):
            stage.chronological_splits(self.cohort.assign(match_date='2020-01-01'), [.7, .15, .15])

    def test_development_passes_and_negative_legacy_resources_rejected(self):
        stage.validate_development(self.data, self.arms)
        altered = self.data.copy(); altered.loc[altered.index[0], stage.resources.BAT_FEATURES[0]] = -1
        with self.assertRaises(ValueError): stage.validate_development(altered, self.arms)

    def test_test_rows_refused_before_training(self):
        frame = self.data.copy(); frame.loc[frame.index[0], 'split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'no test'):
            stage.fit_development(frame, self.arms, self.config, 'not-created', 'not-created.csv.gz')

    def test_duplicate_event_rejected(self):
        with self.assertRaises(ValueError):
            stage.validate_development(pd.concat([self.data, self.data.iloc[:1]]), self.arms)

    def test_inconsistent_match_outcome_rejected(self):
        f = self.data.copy(); f.loc[f.index[0], stage.TARGET] = 1 - f[stage.TARGET].iloc[0]
        with self.assertRaises(ValueError): stage.validate_development(f, self.arms)

    def test_cross_split_match_rejected(self):
        f = self.data.copy(); f.loc[f.index[0], 'split'] = 'validation'
        with self.assertRaises(ValueError): stage.validate_development(f, self.arms)

    def test_structural_chase_missingness_not_imputed(self):
        first = stage.track_columns(self.arms['state'], 1)
        second = stage.track_columns(self.arms['state'], 2)
        self.assertFalse(set(first) & set(stage.CHASE))
        self.assertTrue(set(stage.CHASE) <= set(second))
        f = self.data.copy(); f.loc[f.match_innings.eq(1), 'target_runs'] = 200
        with self.assertRaises(ValueError): stage.validate_development(f, self.arms)

    def test_nonfinite_resources_not_silently_dropped(self):
        f = self.data.copy(); f.loc[f.index[0], stage.resources.BAT_FEATURES[0]] = np.nan
        with self.assertRaises(ValueError): stage.validate_development(f, self.arms)

    def test_one_class_training_rejected(self):
        f = self.data.copy(); f.loc[f.split.eq('train'), stage.TARGET] = 1
        with self.assertRaises(ValueError): stage.validate_development(f, self.arms)

    def test_identifier_or_target_not_a_predictor(self):
        contract = stage.resources.feature_contract(stage.game_state.feature_contract())
        for forbidden in ['match_id', stage.TARGET]:
            c = deepcopy(contract); c['main_arms']['state'].append(forbidden)
            with self.assertRaises(ValueError): stage.resolve_arms(c)

    def test_hgb_early_stopping_disabled(self):
        model = stage.make_pipeline('hist_gradient_boosting', self.config['candidate_grid']['hist_gradient_boosting'][0],
                                    stage.track_columns(self.arms['state'], 2), self.config)
        self.assertIs(model.named_steps['model'].early_stopping, False)

    def test_preprocessing_fits_training_only(self):
        columns = stage.track_columns(self.arms['both_resources'], 2)
        train = self.data[self.data.gender.eq('female') & self.data.match_innings.eq(2) & self.data.split.eq('train')]
        val = train.copy(); val['pre_score'] = 1e6
        model = stage.make_pipeline('logistic_regression', {'C': .1}, columns, self.config)
        model.fit(train[columns], train[stage.TARGET])
        prep = model.named_steps['preprocess']; numeric = [c for c in columns if c != 'innings_phase']
        scaler = prep.named_transformers_['numeric'].named_steps['scale']
        self.assertAlmostEqual(scaler.mean_[numeric.index('pre_score')], train.pre_score.mean())
        before = scaler.mean_.copy(); stage.probability(model, val[columns])
        np.testing.assert_array_equal(before, scaler.mean_)
        self.assertIn(stage.resources.BAT_FEATURES[-1], stage.feature_audit(model, columns)['train_constant_columns_removed'])

    def test_selection_log_loss_not_accuracy(self):
        rows = [{'family': 'logistic_regression', 'candidate': 0, 'validation_event_log_loss': .3, 'validation_accuracy': .6},
                {'family': 'random_forest', 'candidate': 0, 'validation_event_log_loss': .4, 'validation_accuracy': .9}]
        self.assertIs(stage.best_record(rows), rows[0])
        rows[1]['validation_event_log_loss'] = .3
        self.assertIs(stage.best_record(rows[::-1]), rows[0])

    def test_main_models_fixed_rbw_and_cross_group_winners_display_only(self):
        rows = []
        expected_arms = {'female_innings_1': 'batting_resources', 'female_innings_2': 'both_resources',
                         'male_innings_1': 'bowling_resources', 'male_innings_2': 'state'}
        for gender in ['female', 'male']:
            for innings in [1, 2]:
                track = f'{gender}_innings_{innings}'
                for arm, code in stage.ARM_CODES.items():
                    rows.append({'track': track, 'arm': arm, 'arm_code': code, 'candidate': 0,
                                 'family': 'constant' if arm == 'constant' else 'logistic_regression',
                                 'validation_event_log_loss': .1 if arm == expected_arms[track] else .9,
                                 'test_log_loss': 0 if arm == 'both_resources' else 100})
        selected = pd.DataFrame(rows)
        result = stage.select_cross_group_best(selected)
        self.assertEqual(len(result), 4)
        self.assertEqual(result.set_index('track').arm.to_dict(), expected_arms)
        self.assertTrue(result.validation_event_log_loss.eq(.1).all())
        pd.testing.assert_frame_equal(result, stage.select_cross_group_best(selected.sample(frac=1, random_state=1)))
        main = stage.select_final_systems(selected)
        self.assertEqual(len(main), 4)
        self.assertEqual(set(main.arm_code), {'RBW'})
        self.assertTrue(main.loc[main.track.ne('female_innings_2'), 'validation_event_log_loss'].eq(.9).all())
        self.assertFalse(main[['track', 'arm']].equals(result[['track', 'arm']]))
        with self.assertRaises(ValueError):
            stage.select_final_systems(selected[~((selected.track == 'female_innings_1') & (selected.arm == 'both_resources'))])
        with self.assertRaises(ValueError):
            stage.select_final_systems(pd.concat([selected, selected.iloc[:1]]))
        with self.assertRaises(ValueError):
            stage.select_final_systems(selected.assign(validation_event_log_loss=np.nan))
        # Constant baseline is eligible; exact ties are independent of input row order.
        selected['validation_event_log_loss'] = .1
        self.assertEqual(set(stage.select_cross_group_best(selected.sample(frac=1)).arm), {'constant'})
        selected.loc[selected.arm.eq('constant'), 'validation_event_log_loss'] = .9
        self.assertEqual(set(stage.select_cross_group_best(selected.sample(frac=1)).arm), {'state'})
        self.assertEqual(set(stage.select_final_systems(selected).arm), {'both_resources'})

    def test_main_resource_policy_cannot_be_changed_in_config_silently(self):
        for field, value in [('preferred_arm', 'state'), ('selection_policy', 'best_across_arms_v1')]:
            config = deepcopy(self.config); config[field] = value
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / 'config'; path.mkdir()
                (path / 'model_comparison.json').write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, 'policy'):
                    stage.read_config(d)

    def test_metrics_match_weighting_and_single_class_auc(self):
        scores = stage.metrics([1, 1, 0], [.8, .8, .8], [1, 1, 2], 1e-15)
        self.assertNotEqual(scores['validation_event_log_loss'], scores['validation_match_log_loss'])
        self.assertIsNone(stage.metrics([1, 1], [.5, 1], [1, 1], 1e-15)['validation_auc'])
        self.assertTrue(np.isfinite(stage.metrics([1, 0], [0., 1.], [1, 2], 1e-15)['validation_event_log_loss']))


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='t20_stage06_test_')
        cls.root = Path(cls.temp.name)
        config, arms, cohort, splits, data = fixture()
        cls.config, cls.arms, cls.splits, cls.data = config, arms, splits, data
        cls.inputs = {'files': {'synthetic': 'not-real-results'}, 'runtime': {'mode': 'test'}}
        prepared = (config, cls.inputs, {'resource_id': 'synthetic'}, arms, splits, data)
        with patch.object(stage, 'prepare', return_value=prepared), patch.object(stage, 'input_fingerprints', return_value=(cls.inputs, {})), patch('builtins.print'):
            cls.summary = stage.run(cls.root)
        cls.out = cls.root / stage.REPORT_DIR

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_groups_tracks_and_models(self):
        self.assertEqual(self.summary['candidate_fits'], 68)
        self.assertEqual(self.summary['saved_family_models'], 68)
        self.assertEqual(self.summary['selected_arm_models'], 20)
        self.assertEqual(self.summary['test_predictions'], 0)
        self.assertEqual(len(list((self.out / 'models' / stage.VERSION).glob('*.joblib'))), 68)

    def test_main_registry_selects_algorithm_and_parameters_within_rbw(self):
        systems = pd.read_csv(self.out / 'selected_systems.csv')
        winners = pd.read_csv(self.out / 'family_winners.csv')
        self.assertEqual(len(systems), 4)
        self.assertEqual(set(systems.arm_code), {'RBW'})
        for row in systems.itertuples(index=False):
            candidates = winners[winners.track.eq(row.track) & winners.arm.eq('both_resources')]
            expected = stage.best_record(candidates.to_dict('records'))
            self.assertEqual(row.model_file, expected['model_file'])
        self.assertEqual(self.summary['preferred_arm'], 'both_resources')
        self.assertEqual(self.summary['selection_policy'], stage.SELECTION_POLICY)

    def test_rehashed_nonwinning_registry_is_still_rejected(self):
        path = self.out / 'selected_systems.csv'
        manifest_path = self.out / 'manifest.json'
        original, original_manifest = path.read_bytes(), manifest_path.read_bytes()
        try:
            selected = pd.read_csv(self.out / 'selected_models.csv')
            altered = pd.read_csv(path)
            other = selected[selected.track.eq(altered.iloc[0].track) & selected.arm.ne(altered.iloc[0].arm)].iloc[0]
            altered.loc[0] = other
            altered.to_csv(path, index=False)
            manifest = json.loads(original_manifest)
            manifest['output_sha256'][path.relative_to(self.root).as_posix()] = stage.sha256(path)
            manifest_path.write_text(json.dumps(manifest))
            with patch.object(stage, 'input_fingerprints', return_value=(self.inputs, {})):
                with self.assertRaisesRegex(ValueError, 'within-RBW'):
                    stage.verified_output(self.root)
        finally:
            path.write_bytes(original)
            manifest_path.write_bytes(original_manifest)

    def test_predictions_cover_identical_validation_events(self):
        predictions = pd.read_csv(self.root / stage.DATA_DIR / 'validation_predictions.csv.gz')
        expected = set(self.data.loc[self.data.split.eq('validation'), 'match_id'])
        self.assertEqual(set(predictions.match_id), expected)
        for track, group in predictions.groupby('track'):
            reference = None
            for _, arm in group.groupby(['arm', 'family']):
                keys = set(map(tuple, arm[stage.KEYS].to_numpy()))
                if reference is None: reference = keys
                self.assertEqual(keys, reference)

    def test_saved_model_reproduces_predictions(self):
        winners = pd.read_csv(self.out / 'family_winners.csv')
        row = winners[(winners.arm == 'both_resources') & (winners.family == 'logistic_regression')].iloc[0]
        model = joblib.load(self.out / 'models' / row.model_file)
        val = self.data[self.data.gender.eq('female') & self.data.match_innings.eq(1) & self.data.split.eq('validation')]
        p = stage.probability(model, val[stage.track_columns(self.arms['both_resources'], 1)])
        self.assertAlmostEqual(stage.metrics(val[stage.TARGET], p, val.match_id, 1e-15)['validation_event_log_loss'], row.validation_event_log_loss)

    def test_constant_uses_training_prevalence_only(self):
        model = joblib.load(self.out / 'models' / stage.VERSION / 'female_innings_1__constant__constant.joblib')
        train = self.data[self.data.gender.eq('female') & self.data.match_innings.eq(1) & self.data.split.eq('train')]
        self.assertAlmostEqual(stage.probability(model, np.zeros((1, 1)))[0], train[stage.TARGET].mean())

    def test_delta_sign_and_comparisons(self):
        deltas = pd.read_csv(self.out / 'validation_deltas.csv')
        self.assertEqual(len(deltas), 4 * 5 * (len(stage.FAMILIES) + 1))
        selected = pd.read_csv(self.out / 'selected_models.csv')
        row = deltas[deltas.view.eq('selected_system')].iloc[0]
        group = selected[selected.track.eq(row.track)].set_index('arm')
        self.assertAlmostEqual(row.validation_log_loss_gain,
            group.loc[row.reference].validation_event_log_loss - group.loc[row.treatment].validation_event_log_loss)

    def test_manifest_verification_and_changed_config_rejection(self):
        with patch.object(stage, 'input_fingerprints', return_value=(self.inputs, {})):
            self.assertEqual(stage.verified_output(self.root)['experiment_id'], self.summary['experiment_id'])
        with patch.object(stage, 'input_fingerprints', return_value=({'changed': True}, {})):
            with self.assertRaises(ValueError): stage.verified_output(self.root)

    def test_notebook_all_cells_read_saved_results_without_training(self):
        notebook = json.loads((ROOT / 'notebooks/06_model_training_and_comparison.ipynb').read_text(encoding='utf-8'))
        cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        namespace = {}
        final_registry_before = (self.out / 'selected_systems.csv').read_bytes()
        with patch('builtins.print'):
            exec(compile(cells[0], 'stage06_setup', 'exec'), namespace)
            namespace.update(ROOT=self.root, OUTPUT=self.out, display=lambda *args, **kwargs: None)
            with patch.object(stage, 'input_fingerprints', return_value=(self.inputs, {})), patch.object(stage, 'run') as train:
                for cell in cells[1:]:
                    exec(compile(cell, 'stage06_read_mode', 'exec'), namespace)
                train.assert_not_called()
        self.assertEqual((self.out / 'selected_systems.csv').read_bytes(), final_registry_before)
        supplementary = namespace['supplementary_best']
        self.assertEqual(len(supplementary), 4)
        selected = pd.read_csv(self.out / 'selected_models.csv')
        for row in supplementary.itertuples(index=False):
            expected = stage.best_record(selected[selected.track.eq(row.track)].to_dict('records'))
            self.assertEqual(row.model_file, expected['model_file'])
        figures = self.out / 'figures'
        self.assertEqual(len(list(figures.glob('*.png'))), 4)
        self.assertIs(json.loads((figures / 'figure_manifest.json').read_text())['test_scored'], False)

    def test_tampered_export_is_rejected(self):
        path = self.out / 'selected_systems.csv'
        original = path.read_bytes()
        try:
            path.write_bytes(original + b'\n')
            with patch.object(stage, 'input_fingerprints', return_value=(self.inputs, {})):
                with self.assertRaisesRegex(ValueError, 'artifact changed'): stage.verified_output(self.root)
        finally:
            path.write_bytes(original)

    def test_missing_output_never_trains(self):
        with tempfile.TemporaryDirectory() as d, patch.object(stage, 'run') as fit:
            with self.assertRaises(FileNotFoundError): stage.verified_output(d)
            fit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
