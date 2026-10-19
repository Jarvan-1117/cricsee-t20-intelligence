"""Shared Stage 07 engine checks using the current IB adapter and synthetic spies."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import resource_interaction_experiments as interactions
import resource_extension_common as common
from test_model_comparison import fixture as base_fixture


class EngineExtension(interactions.Extension):
    """Expose the common engine through a current interaction adapter for tests."""
    def __init__(self, code):
        super().__init__(code)
        self.EXPECTED_CONFIG = self.read_config(ROOT)

    def run(self, root):
        return common.run(root, self)

    def verified_output(self, root):
        return common.verified_output(root, self)


stage = EngineExtension('IB')


def fixture():
    config, arms, _, splits, data = base_fixture()
    data['scheduled_balls_remaining_estimate'] = np.where(data.event_sequence.eq(1), 120., 30.)
    joined, audit = stage.build_features(ROOT, data)
    return config, arms, splits, joined, audit


def reference_fixture(data, config, extension):
    rows, predictions = [], []
    for track in common.TRACKS:
        gender, _, number = track.split('_')
        val = data[data.gender.eq(gender) & data.match_innings.eq(int(number)) & data.split.eq('validation')]
        for arm in extension.REFERENCES:
            for family in common.base.FAMILIES:
                probability = np.full(len(val), .5)
                rows.append(dict(track=track, arm=arm, arm_code=common.base.ARM_CODES[arm],
                    family=family, candidate=0, parameters=json.dumps(config['candidate_grid'][family][0], sort_keys=True),
                    model_file='reference-not-loaded.joblib',
                    **common.base.metrics(val[common.base.TARGET], probability, val.match_id, config['probability_epsilon'])))
                frame = val[common.base.META + [common.base.TARGET, 'innings_phase']].copy()
                frame['track'], frame['arm'], frame['family'] = track, arm, family
                frame['raw_model_probability'] = probability
                predictions.append(frame)
    winners = pd.DataFrame(rows)
    selected = pd.DataFrame([common.base.best_record(g.to_dict('records')) for _, g in winners.groupby(['track','arm'])])
    return winners, selected, pd.concat(predictions, ignore_index=True)


class SpyPipeline:
    classes_ = np.array([0, 1])
    calls = []
    def fit(self, x, y):
        type(self).calls.append(('fit', list(x.index), list(x.columns), len(y)))
        return self
    def predict_proba(self, x):
        type(self).calls.append(('predict', list(x.index), list(x.columns), len(x)))
        return np.full((len(x), 2), .5)


class InteractionTests(unittest.TestCase):
    def setUp(self):
        self.config, self.arms, self.splits, self.data, self.audit = fixture()
        self.columns = stage.resolve_columns(self.arms)

    def test_exact_additions_and_v2_allowlist(self):
        self.assertEqual(self.columns, self.arms['both_resources'] + stage.FEATURES)
        common.validate_development(self.data, self.columns, stage)
        self.assertEqual(len(stage.FEATURES), 4)

    def test_formula_preserves_nonnegative_resource(self):
        for original, feature in zip(common.resources.BAT_FEATURES, stage.FEATURES):
            np.testing.assert_allclose(self.data[feature], self.data[original] * self.data.scheduled_balls_remaining_estimate / 120.)
        self.assertFalse(self.data[stage.FEATURES].lt(0).any().any())

    def test_zero_budget_does_not_remove_original_resources(self):
        raw = self.data.drop(columns=stage.FEATURES)
        raw.loc[raw.index[0], 'scheduled_balls_remaining_estimate'] = 0
        result, _ = stage.build_features(ROOT, raw)
        self.assertEqual(result.loc[0, stage.FEATURES].sum(), 0)
        pd.testing.assert_frame_equal(result[common.resources.BAT_FEATURES], raw[common.resources.BAT_FEATURES])

    def test_negative_and_invalid_progress_rejected(self):
        for value in [-1, 121, np.nan]:
            data = self.data.copy()
            data.loc[data.index[0], 'scheduled_balls_remaining_estimate'] = value
            with self.assertRaises(ValueError): stage.validate_features(data)
        data = self.data.copy()
        data.loc[data.index[0], common.resources.BAT_FEATURES[0]] = -1
        with self.assertRaises(ValueError): stage.validate_features(data)

    def test_incorrect_product_rejected(self):
        data = self.data.copy()
        data.loc[data.index[0], stage.FEATURES[0]] += .1
        with self.assertRaises(ValueError): stage.validate_features(data)

    def test_join_missing_duplicate_or_wrong_keys_rejected(self):
        data = self.data.drop(columns=stage.FEATURES)
        extra = self.data[common.base.KEYS + stage.FEATURES]
        changed = extra.copy()
        changed.iloc[0, changed.columns.get_loc('source_row')] = -999
        for part in [extra.iloc[1:], pd.concat([extra.iloc[1:], extra.iloc[1:2]]), changed]:
            with self.assertRaises(ValueError): common.join_extra(data, part, stage.FEATURES)
        joined = common.join_extra(data, extra.sample(frac=1, random_state=1), stage.FEATURES)
        pd.testing.assert_frame_equal(joined[data.columns], data.reset_index(drop=True))

    def test_direct_quality_extra_and_legacy_columns_rejected(self):
        for field in common.resources.quality.feature_contract()['current_player_block'] + ['audit_capacity', 'player_id']:
            with self.assertRaises(ValueError):
                common.validate_development(self.data.assign(**{field: 1}), self.columns + [field], stage)

    def test_test_rows_rejected_before_pipeline_or_writes(self):
        data = self.data.copy()
        data.loc[data.index[0], 'split'] = 'test'
        with tempfile.TemporaryDirectory() as folder, patch.object(common.base, 'make_pipeline') as make:
            with self.assertRaisesRegex(ValueError, 'no test'):
                common.fit_extension(data, self.columns, self.config, Path(folder)/'models', Path(folder)/'p.csv.gz', stage)
            make.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_missing_results_do_not_fallback_or_train(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(common, 'run') as run:
            old = Path(folder)/'outputs/07_resource_representation_experiments'
            old.mkdir(parents=True)
            (old/'manifest.json').write_text('{"status":"complete"}')
            with self.assertRaisesRegex(FileNotFoundError, 'v2'): stage.verified_output(folder)
            run.assert_not_called()

    def test_stale_v2_outputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(common, 'input_fingerprints', return_value=({'new':'hash'}, {})):
            out = Path(folder)/stage.REPORT_DIR
            out.mkdir(parents=True)
            (out/'manifest.json').write_text('{"status":"complete","contract_version":"old","inputs":{}}')
            with self.assertRaisesRegex(ValueError, 'stale'): stage.verified_output(folder)

    def test_config_requires_frozen_policy(self):
        self.assertEqual(stage.read_config(ROOT), stage.EXPECTED_CONFIG)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/stage.CONFIG_PATH
            path.parent.mkdir()
            config = deepcopy(interactions.EXPECTED_CONFIG)
            config['test_scored'] = True
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError): stage.read_config(folder)

    def test_first_innings_structural_omissions(self):
        actual = common.base.track_columns(self.columns, 1)
        self.assertFalse(set(actual) & set(common.base.CHASE + ['match_innings']))
        self.assertTrue(set(stage.FEATURES).issubset(actual))


class EngineTests(unittest.TestCase):
    def test_reference_reconstruction_and_wrong_labels(self):
        config, _, _, data, _ = fixture()
        winners, selected, predictions = reference_fixture(data, config, stage)
        with tempfile.TemporaryDirectory() as folder:
            out, processed = Path(folder)/common.base.REPORT_DIR, Path(folder)/common.base.DATA_DIR
            out.mkdir(parents=True); processed.mkdir(parents=True)
            winners.to_csv(out/'family_winners.csv', index=False)
            selected.to_csv(out/'selected_models.csv', index=False)
            predictions.drop(columns='innings_phase').to_csv(processed/'validation_predictions.csv.gz', index=False)
            checked = common.baseline_tables(folder, data, config, stage.REFERENCES)
            self.assertEqual(len(checked[0]), 4 * len(stage.REFERENCES) * len(common.base.FAMILIES))
            bad = predictions.drop(columns='innings_phase').copy()
            bad.loc[0, common.base.TARGET] = 1-bad.loc[0, common.base.TARGET]
            bad.to_csv(processed/'validation_predictions.csv.gz', index=False)
            with self.assertRaisesRegex(ValueError, 'labels'): common.baseline_tables(folder, data, config, stage.REFERENCES)

    def test_run_and_verifier_with_spies_no_real_training(self):
        self.exercise_engine(stage, fixture())

    def exercise_engine(self, stage, fixture_data):
        config, arms, splits, data, audit = fixture_data
        winners, selected, refs = reference_fixture(data, config, stage)
        inputs = {'synthetic':'no-real-model-fit'}
        prepared = dict(policy=stage.EXPECTED_CONFIG, config=config, inputs=inputs,
            upstream={'experiment_id':'synthetic-stage06'}, columns=stage.resolve_columns(arms),
            splits=splits, data=data, audit=audit, reference_winners=winners,
            reference_selected=selected, reference_predictions=refs)
        SpyPipeline.calls = []
        with tempfile.TemporaryDirectory() as folder:
            split_path = Path(folder)/common.base.DATA_DIR/'match_splits.csv'
            split_path.parent.mkdir(parents=True)
            splits.to_csv(split_path, index=False)
            before = split_path.read_bytes()
            with patch.object(common, 'prepare', return_value=prepared), patch.object(common, 'input_fingerprints', return_value=(inputs, prepared['upstream'])), patch.object(common.base, 'make_pipeline', side_effect=lambda *a: SpyPipeline()), patch.object(common.base, 'feature_audit', return_value={}), patch('builtins.print'):
                summary = stage.run(folder)
                checked = stage.verified_output(folder)
            self.assertEqual(summary, checked)
            self.assertEqual(summary['candidate_fits'], 4 * len(common.base.FAMILIES))  # Tiny spy grid.
            self.assertFalse(summary['main_model_promotion'])
            self.assertEqual(summary['test_predictions'], 0)
            self.assertEqual(before, split_path.read_bytes())
            for kind, indices, columns, _ in SpyPipeline.calls:
                expected = 'train' if kind == 'fit' else 'validation'
                self.assertTrue(data.loc[indices, 'split'].eq(expected).all())
                self.assertFalse(set(columns) & set(common.resources.quality.feature_contract()['current_player_block']))
            deltas = pd.read_csv(Path(folder)/stage.REPORT_DIR/'validation_comparisons.csv')
            self.assertEqual(len(deltas), 4 * (2 * len(common.base.FAMILIES) + 1) * len(stage.REFERENCES))
            self.assertEqual(set(deltas.reference_code), {common.base.ARM_CODES[a] for a in stage.REFERENCES})
            for row in deltas[deltas.comparison_mode.eq('fixed_baseline_parameters')].itertuples():
                self.assertEqual(row.treatment_family, row.reference_family)
                self.assertEqual(json.loads(row.treatment_parameters), json.loads(row.reference_parameters))

    def test_fixed_parameter_comparison_does_not_use_retuned_winner(self):
        config, _, _, data, _ = fixture()
        refs, ref_selected, _ = reference_fixture(data, config, stage)
        candidates = refs[refs.arm.eq('both_resources')].copy()
        candidates['arm'], candidates['arm_code'] = stage.ARM, stage.CODE
        alternative = candidates.copy()
        alternative['candidate'] = 1
        alternative['parameters'] = '{"different":1}'
        alternative['validation_event_log_loss'] -= .05
        candidates = pd.concat([candidates, alternative])
        winners = alternative
        selected = pd.DataFrame([common.base.best_record(g.to_dict('records')) for _, g in winners.groupby('track')])
        compared = common.comparisons(candidates, winners, selected, refs, ref_selected, stage)
        self.assertTrue(compared[compared.comparison_mode.eq(common.MODES[0])].treatment_candidate.eq(0).all())
        self.assertTrue(compared[compared.comparison_mode.eq(common.MODES[1])].treatment_candidate.eq(1).all())

    def test_empty_phase_is_missing_not_zero(self):
        config, _, _, data, _ = fixture()
        _, selected, predictions = reference_fixture(data, config, stage)
        frames = [predictions[predictions.track.eq(r.track) & predictions.arm.eq(r.arm) & predictions.family.eq(r.family)].assign(candidate=r.candidate)
                  for r in selected.itertuples()]
        phases = common.phase_diagnostics(pd.concat(frames), config['probability_epsilon'], stage)
        self.assertTrue(phases[phases.phase.eq('death')].validation_event_log_loss.isna().all())


if __name__ == '__main__': unittest.main()
