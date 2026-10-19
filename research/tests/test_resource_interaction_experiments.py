"""Symmetric Stage 07 checks; synthetic data and spies, not production training."""
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
import resource_interaction_experiments as stage
from test_model_comparison import fixture
import test_resource_extension_common as engine_tests


class InteractionTests(unittest.TestCase):
    def setUp(self):
        self.config, self.arms, _, self.splits, self.raw = fixture()
        self.raw['scheduled_balls_remaining_estimate'] = np.where(self.raw.event_sequence.eq(1), 120., 30.)

    def test_four_separate_definitions(self):
        self.assertEqual([len(stage.interaction_features(c)) for c in stage.CODES], [0, 4, 3, 7])
        self.assertEqual(stage.interaction_features('IBW'), stage.interaction_features('IB') + stage.interaction_features('IW'))
        with self.assertRaises(ValueError): stage.Extension('E2')

    def test_formula_all_arms_preserves_original_rows_and_resources(self):
        for code in stage.CODES[1:]:
            ext = stage.Extension(code)
            data, audit = ext.build_features(ROOT, self.raw)
            pd.testing.assert_frame_equal(data[self.raw.columns], self.raw.reset_index(drop=True))
            stage.common.validate_development(data, ext.resolve_columns(self.arms), ext)
            for original, feature in zip(stage.raw_features(code), ext.FEATURES):
                np.testing.assert_allclose(data[feature], self.raw[original] * self.raw.scheduled_balls_remaining_estimate / 120.)
            self.assertEqual(audit.events.sum(), len(self.raw))

    def test_batting_features_match_explicit_formula(self):
        expected = self.raw.copy().reset_index(drop=True)
        for feature in stage.resources.BAT_FEATURES:
            expected['progress_interaction_' + feature] = (
                expected[feature] * expected.scheduled_balls_remaining_estimate / 120.)
        actual, _ = stage.Extension('IB').build_features(ROOT, self.raw)
        pd.testing.assert_frame_equal(actual, expected)

    def test_zero_and_full_progress(self):
        for balls in [0., 120.]:
            raw = self.raw.copy()
            raw['scheduled_balls_remaining_estimate'] = balls
            for code in stage.CODES[1:]:
                ext = stage.Extension(code)
                data, _ = ext.build_features(ROOT, raw)
                np.testing.assert_allclose(data[ext.FEATURES], raw[stage.raw_features(code)].to_numpy() * balls / 120.)
                pd.testing.assert_frame_equal(data[raw.columns], raw.reset_index(drop=True))

    def test_invalid_progress_and_resource_fail(self):
        for value in [-1., 121., np.nan, np.inf]:
            data = self.raw.copy()
            data.loc[0, 'scheduled_balls_remaining_estimate'] = value
            with self.assertRaises(ValueError): stage.Extension('IBW').build_features(ROOT, data)
        for feature in stage.raw_features('IBW'):
            data = self.raw.copy()
            data.loc[0, feature] = -1.
            with self.assertRaises(ValueError): stage.Extension('IBW').build_features(ROOT, data)

    def test_changed_product_rejected(self):
        ext = stage.Extension('IW')
        data, _ = ext.build_features(ROOT, self.raw)
        data.loc[0, ext.FEATURES[0]] += 1.
        with self.assertRaisesRegex(ValueError, 'reconstruction'): ext.validate_features(data)

    def test_direct_quality_and_identifiers_not_in_allowlist(self):
        ext = stage.Extension('IBW')
        data, _ = ext.build_features(ROOT, self.raw)
        for name in stage.resources.quality.feature_contract()['current_player_block'] + ['player_id', 'audit_capacity']:
            with self.assertRaises(ValueError):
                stage.common.validate_development(data.assign(**{name: 1}), ext.resolve_columns(self.arms) + [name], ext)

    def test_no_first_innings_chase_fields(self):
        columns = stage.Extension('IBW').resolve_columns(self.arms)
        self.assertFalse(set(stage.base.track_columns(columns, 1)) & set(stage.base.CHASE))
        self.assertTrue(set(stage.base.CHASE) <= set(stage.base.track_columns(columns, 2)))

    def test_test_rows_rejected_before_any_fit_or_write(self):
        ext = stage.Extension('IBW')
        data, _ = ext.build_features(ROOT, self.raw)
        data.loc[0, 'split'] = 'test'
        with tempfile.TemporaryDirectory() as folder, patch.object(stage.base, 'make_pipeline') as make:
            with self.assertRaisesRegex(ValueError, 'no test'):
                stage.common.fit_extension(data, ext.resolve_columns(self.arms), self.config,
                                           Path(folder)/'models', Path(folder)/'p.csv.gz', ext)
            make.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_policy_changes_fail_closed(self):
        self.assertEqual(stage.read_config(ROOT), stage.EXPECTED_CONFIG)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / stage.CONFIG_PATH
            path.parent.mkdir()
            policy = deepcopy(stage.EXPECTED_CONFIG)
            policy['test_scored'] = True
            path.write_text(json.dumps(policy))
            with self.assertRaises(ValueError): stage.read_config(folder)

    def test_missing_result_never_runs_training(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(stage, 'run') as run:
            with self.assertRaises(FileNotFoundError): stage.verified_output(folder)
            run.assert_not_called()

    def test_all_new_arms_use_shared_engine_with_spies(self):
        for code in stage.CODES[1:]:
            ext = stage.Extension(code)
            data, audit = ext.build_features(ROOT, self.raw)
            ext.EXPECTED_CONFIG = ext.read_config(ROOT)
            ext.run = lambda root, ext=ext: stage.common.run(root, ext)
            ext.verified_output = lambda root, ext=ext: stage.common.verified_output(root, ext)
            engine_tests.EngineTests().exercise_engine(ext, (self.config, self.arms, self.splits, data, audit))

    def test_preflight_no_fit_and_same_four_group_cohort(self):
        ext = stage.Extension('IB')
        data, _ = ext.build_features(ROOT, self.raw)
        prepared = dict(data=data, inputs={'synthetic': 'hash'}, upstream={'experiment_id': 'synthetic'}, config=self.config)
        with patch.object(stage.common, 'prepare', return_value=prepared), patch.object(stage.base, 'make_pipeline') as make:
            checked = stage.preflight(ROOT)
        self.assertEqual(checked['expected_new_candidate_fits'], 3 * 4 * len(stage.base.FAMILIES))
        self.assertTrue(checked['shared_rows_and_dates'])
        self.assertFalse(checked['test_scored'])
        make.assert_not_called()


class SelectionTests(unittest.TestCase):
    def table(self):
        return pd.DataFrame([dict(track=t, arm_code=c, candidate=0,
            validation_event_log_loss=.5, test_log_loss=.01 if c == 'IBW' else .99)
            for t in stage.TRACKS for c in stage.CODES])

    def test_exact_ties_use_prespecified_order_not_test(self):
        selected = stage.select_systems(self.table())
        self.assertEqual(selected.arm_code.tolist(), ['RBW'] * 4)

    def test_validation_selects_independently_for_each_track(self):
        table = self.table()
        for track, code in zip(stage.TRACKS, stage.CODES):
            table.loc[table.track.eq(track) & table.arm_code.eq(code), 'validation_event_log_loss'] = .4
        self.assertEqual(stage.select_systems(table).arm_code.tolist(), stage.CODES)

    def test_incomplete_duplicate_or_nonfinite_rejected(self):
        table = self.table()
        for broken in [table.iloc[1:], pd.concat([table.iloc[1:], table.iloc[1:2]]),
                       table.assign(validation_event_log_loss=np.nan)]:
            with self.assertRaises(ValueError): stage.select_systems(broken)


class WorkflowTests(unittest.TestCase):
    def test_full_workflow_resume_and_tamper_detection_with_spies(self):
        config, arms, _, splits, raw = fixture()
        raw['scheduled_balls_remaining_estimate'] = 60.
        inputs, upstream = {'fixture': 'only-synthetic-validation'}, {'experiment_id': 'synthetic06'}
        def prepared(root, ext):
            data, audit = ext.build_features(root, raw)
            winners, selected, refs = engine_tests.reference_fixture(data, config, ext)
            return dict(policy=ext.read_config(root), config=config, inputs=inputs, upstream=upstream,
                columns=ext.resolve_columns(arms), splits=splits, data=data, audit=audit,
                reference_winners=winners, reference_selected=selected, reference_predictions=refs)
        checks = dict(status='passed', input_fingerprints=inputs, expected_new_candidate_fits=3 * 4 * len(stage.base.FAMILIES),
                      development_matches=int(raw.match_id.nunique()), development_events=len(raw), rows=[])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = root / stage.CONFIG_PATH
            cfg.parent.mkdir()
            cfg.write_text(json.dumps(stage.EXPECTED_CONFIG))
            split = root / stage.base.DATA_DIR / 'match_splits.csv'
            split.parent.mkdir(parents=True)
            splits.to_csv(split, index=False)
            initial = prepared(root, stage.Extension('IB'))['reference_selected'].copy()
            initial['model_file'] = [track+'.joblib' for track in initial.track]
            initial['train_events'] = 4
            initial['train_matches'] = 2
            base_out = root / stage.base.REPORT_DIR
            (base_out/'models').mkdir(parents=True)
            initial.to_csv(base_out/'selected_models.csv', index=False)
            for filename in initial.model_file:
                (base_out/'models'/filename).write_bytes(b'SYNTHETIC CONTROL; NEVER LOADED')
            engine_tests.SpyPipeline.calls = []
            with patch.object(stage, 'preflight', return_value=checks), \
                 patch.object(stage.common, 'prepare', side_effect=prepared), \
                 patch.object(stage.common, 'input_fingerprints', return_value=(inputs, upstream)), \
                 patch.object(stage, 'input_fingerprints', return_value=(inputs, upstream)), \
                 patch.object(stage.base, 'read_config', return_value=config), \
                 patch.object(stage.base, 'make_pipeline', side_effect=lambda *a: engine_tests.SpyPipeline()), \
                 patch.object(stage.base, 'feature_audit', return_value={}), patch('builtins.print'):
                summary = stage.run(root)
                self.assertEqual(stage.verified_output(root), summary)
                self.assertEqual(summary['new_candidate_fits'], 3 * 4 * len(stage.base.FAMILIES))
                self.assertFalse(summary['test_scored'])
                registry = json.loads((root/stage.REPORT_DIR/'frozen_protocol.json').read_text())
                self.assertEqual([r['arm_code'] for r in registry['models']], ['RBW']*4)
                calls = len(engine_tests.SpyPipeline.calls)
                stage.run(root, resume=True)
                self.assertEqual(len(engine_tests.SpyPipeline.calls), calls)
                table = root/stage.REPORT_DIR/'selected_systems.csv'
                table.write_text(table.read_text()+'\n')
                with self.assertRaisesRegex(ValueError, 'artifact changed'):
                    stage.verified_output(root)


if __name__ == '__main__': unittest.main()
