"""Stage 09 producer contract and shared accounting regression checks."""
import copy
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import nwc_attribution as accounting
import nwc_attribution_joint as stage
import nwc_attribution_joint_visualisation as viz
import nwc_terminal_audit as original_audit
import nwc_terminal_audit_joint as audit
import test_nwc_attribution as fixtures


def contract():
    handoff, protocol, summary, p, d, meta = fixtures.handoff_fixture()
    preferred = dict(zip(stage.evaluation.TRACKS, ['RW', 'RBW', 'IBW', 'IBW']))
    for row in protocol['models']:
        row['preferred'] = row['arm_code'] == preferred[row['track']]
        row['validation_event_log_loss'] = .1 if row['preferred'] else .9
    protocol.update(config=stage.evaluation.expected_config(), preferred=preferred)
    keys = ['track', 'arm_code', 'family', 'model_path', 'model_sha256', 'features']
    handoff.update(status='raw_predictions_ready', contract_version=stage.VERSION,
        selection_policy=stage.evaluation.POLICY,
        prediction_path=stage.evaluation.DATA_DIR+'/preferred_test_predictions.csv.gz',
        preferred_models=[{k:r[k] for k in keys} for r in protocol['models'] if r['preferred']])
    summary.update(contract_version=stage.VERSION, selection_policy=stage.evaluation.POLICY,
                   preferred=preferred)
    p['arm_code'] = p.track.map(preferred)
    return handoff, protocol, summary, p, d, meta


class JointTests(unittest.TestCase):
    def test_shared_accounting_is_same_function_not_reimplemented(self):
        for name in ['validate_events', 'allocation_rule', 'allocation_rules',
                     'build_ledger', 'summarise', 'team_accounting']:
            self.assertIs(getattr(stage, name), getattr(accounting, name))
        self.assertEqual(inspect.getsource(audit.compare_cases),
                         inspect.getsource(original_audit.compare_cases))
        self.assertEqual(stage.KEYS, accounting.KEYS)
        self.assertEqual(stage.DIRECT_DISMISSALS, accounting.DIRECT_DISMISSALS)

    def test_config_diff_is_only_producer_version_and_policy(self):
        a, b = stage.read_config(ROOT), accounting.read_config(ROOT)
        self.assertEqual({k for k in a if a[k] != b[k]}, {'contract_version', 'main_model_policy'})
        self.assertIn('src/nwc_attribution.py', stage.SOURCE_FILES)
        self.assertIn('src/nwc_attribution_joint.py', stage.SOURCE_FILES)

    def test_four_cross_group_models_and_prediction_contract(self):
        h, p, s, predictions, _, _ = contract()
        registry = stage.validate_handoff(h, p, s)
        self.assertEqual({r['arm_code'] for r in registry.values()}, {'RW', 'RBW', 'IBW'})
        stage.validate_prediction_handoff(predictions, registry, s)

    def test_wrong_handoff_identity_path_status_perspective_rejected(self):
        h, p, s, _, _, _ = contract()
        for k, v in [('status', 'requires_stage09_migration'), ('evaluation_id', 'wrong'),
                     ('prediction_path', 'wrong'), ('prediction_timing', 'post-event'),
                     ('probability_column', 'attribution_probability'),
                     ('target', 'batting_team_win'), ('contract_version', accounting.VERSION)]:
            candidate = dict(h, **{k:v})
            with self.subTest(field=k), self.assertRaises(ValueError):
                stage.validate_handoff(candidate, p, s)

    def test_winner_flags_cannot_override_validation_scores(self):
        h, p, s, _, _, _ = contract()
        p['models'][0]['validation_event_log_loss'] = .001
        with self.assertRaises(ValueError): stage.validate_handoff(h, p, s)

    def test_hash_features_duplicates_and_missing_models_rejected(self):
        h, p, s, _, _, _ = contract()
        for mode in ['hash', 'features', 'duplicate', 'missing']:
            candidate = copy.deepcopy(h)
            if mode == 'hash': candidate['preferred_models'][0]['model_sha256'] = 'bad'
            if mode == 'features': candidate['preferred_models'][0]['features'].append('quality')
            if mode == 'duplicate': candidate['preferred_models'].append(candidate['preferred_models'][0])
            if mode == 'missing': candidate['preferred_models'].pop()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                stage.validate_handoff(candidate, p, s)

    def test_missing_duplicate_and_adjusted_predictions_rejected(self):
        h, p, s, predictions, _, _ = contract()
        registry = stage.validate_handoff(h, p, s)
        for bad in [predictions.iloc[:-1], pd.concat([predictions, predictions.iloc[:1]]),
                    predictions.assign(attribution_probability=.5), predictions.assign(arm_code='IBW')]:
            with self.assertRaises(ValueError): stage.validate_prediction_handoff(bad, registry, s)

    def test_identical_inputs_have_identical_ledger_and_summaries(self):
        _, _, _, _, d, meta = contract()
        a, b = accounting.build_ledger(d, meta), stage.build_ledger(d, meta)
        pd.testing.assert_frame_equal(a, b)
        for old, new in zip(accounting.summarise(a), stage.summarise(b)):
            pd.testing.assert_frame_equal(old, new)

    def test_run_verification_and_no_fallback(self):
        h, _, s, _, d, meta = contract()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(FileNotFoundError): stage.verified_output(root)
            with patch.object(stage, 'read_config', return_value=stage.read_config(ROOT)), \
                 patch.object(stage, 'input_fingerprints', return_value=({'synthetic':'hash'}, s)), \
                 patch.object(stage, 'load_inputs', return_value=(d, meta, h)):
                result = stage.run(root, progress=lambda _:None)
                self.assertEqual(stage.verified_output(root)['events'], 12)
                self.assertEqual(result['training_fits'], 0)
                self.assertEqual(result['prediction_calls'], 0)
                (root/stage.REPORT_DIR/'team_reconciliation.csv').write_text('tampered')
                with self.assertRaisesRegex(ValueError, 'artifact changed'): stage.verified_output(root)


if __name__ == '__main__': unittest.main()
