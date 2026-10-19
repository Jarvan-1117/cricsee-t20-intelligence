"""Synthetic saved-model diagnostics; never fit production models or score test events."""
import copy
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import probability_development_diagnostics as diagnostics
from probability_fixtures import fixture

stage = diagnostics.stage
base = stage.base


class DevelopmentDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='development_diagnostics_')
        cls.root = Path(cls.tmp.name)
        cls.config, cls.registry, _, cls.splits = fixture(cls.root, stage_module=stage)
        cls.splits['split'] = ['train', 'train', 'validation', 'test', 'test', 'test']*2
        raw = pd.read_csv(cls.root/base.resources.DATA_DIR/'model_input.csv.gz')
        raw['split'] = raw.match_id.map(cls.splits.set_index('match_id').split)
        cls.raw = raw[raw.split.ne('test')].copy()
        cls.data, _ = stage.interactions.Extension('IBW').build_features(cls.root, cls.raw)
        records = []
        for row in cls.registry:
            model = joblib.load(cls.root/row['model_path'])
            gender, _, innings = row['track'].split('_')
            subset = cls.data[cls.data.gender.eq(gender) & cls.data.match_innings.eq(int(innings))]
            record = {k: row[k] for k in ['track', 'arm_code', 'family', 'candidate']}
            for split in ['train', 'validation']:
                part = subset[subset.split.eq(split)]
                x = part[row['features']] if row['features'] else np.zeros((len(part), 1))
                p = base.probability(model, x)
                metrics = base.metrics(part[base.TARGET], p, part.match_id, 1e-15)
                for name in diagnostics.METRICS+['events', 'matches']:
                    record[split+'_'+name] = metrics['validation_'+name]
            row['validation_event_log_loss'] = record['validation_event_log_loss']
            records.append(record)
        cls.reference = pd.DataFrame(records).set_index(['track', 'arm_code', 'family', 'candidate'])
        preferred = stage.select_preferred(cls.registry)
        for row in cls.registry:
            row['preferred'] = row['arm_code'] == preferred[row['track']]
        target = cls.root/diagnostics.SOURCE_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/diagnostics.SOURCE_PATH, target)
        cls.inputs = {diagnostics.SOURCE_PATH: stage.sha256(target)}
        cls.prepared = (cls.config, cls.inputs, cls.registry, cls.reference, 'selection', 'diagnostic')
        cls.before = {p: stage.sha256(p) for p in cls.root.rglob('*') if p.is_file()}
        def loader(root):
            protocol = json.loads((root/diagnostics.REPORT_DIR/'protocol.json').read_text())
            if protocol['models'] != cls.registry:
                raise AssertionError('Registry must be frozen before scoring.')
            return cls.data, cls.splits
        with patch.object(diagnostics, 'prepare', return_value=cls.prepared), \
             patch.object(diagnostics, 'load_development', side_effect=loader), \
             patch.object(stage, 'input_fingerprints', return_value=({}, {})), \
             patch.object(stage, 'load_test_data', side_effect=AssertionError('No test access')), \
             patch.object(stage, 'run', side_effect=AssertionError('No test evaluation')), \
             patch.object(base, 'make_pipeline', side_effect=AssertionError('No fitting')):
            cls.summary = diagnostics.run(cls.root, lambda *args: None)
        cls.out = cls.root/diagnostics.REPORT_DIR
        cls.table = pd.read_csv(cls.out/'train_validation_metrics.csv')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_groups_same_model_both_splits_and_no_source_writes(self):
        self.assertEqual(len(self.table), 32)
        self.assertEqual(int(self.table.preferred.sum()), 4)
        self.assertFalse(self.summary['test_scored'])
        self.assertEqual(self.summary['training_fits'], 0)
        self.assertEqual(self.summary['split_counts']['train'], {'events': 24, 'matches': 4})
        for path, digest in self.before.items():
            self.assertEqual(stage.sha256(path), digest)
        self.assertFalse((self.root/stage.REPORT_DIR/'manifest.json').exists())
        for row in self.table.itertuples():
            ref = self.reference.loc[(row.track, row.arm_code, row.family, row.candidate)]
            for metric in diagnostics.METRICS:
                self.assertAlmostEqual(getattr(row, 'train_'+metric), ref['train_'+metric])
                self.assertAlmostEqual(getattr(row, 'validation_'+metric), ref['validation_'+metric])
                self.assertAlmostEqual(getattr(row, metric+'_gap'), ref['validation_'+metric]-ref['train_'+metric])

    def test_gap_is_not_innings_gap_or_selection_input_and_negative_retained(self):
        altered = self.table.copy()
        altered['train_event_log_loss'] = altered.validation_event_log_loss+1
        altered['event_log_loss_gap'] = -1.
        diagnostics.validate_table(altered, self.registry)
        self.assertTrue(altered.preferred.equals(self.table.preferred))
        selected = stage.select_preferred(self.registry)
        changed = copy.deepcopy(self.registry)
        for row in changed:
            row.update(train_event_log_loss=-100, event_log_loss_gap=100)
        self.assertEqual(stage.select_preferred(changed), selected)

    def test_saved_validation_brier_log_loss_and_counts_must_reproduce(self):
        for name in ['validation_brier', 'validation_event_log_loss', 'train_events', 'validation_matches']:
            reference = self.reference.copy()
            reference.iloc[0, reference.columns.get_loc(name)] += 1
            with self.assertRaises(ValueError):
                diagnostics.score_models(self.root, self.data, self.splits, self.registry, reference, self.config, lambda *a: None)

    def test_reject_test_leakage_metadata_missing_and_duplicate_events(self):
        for kind in ['test', 'metadata', 'missing', 'duplicate', 'label']:
            data = self.data.copy()
            if kind == 'test': data.loc[data.index[0], 'split'] = 'test'
            if kind == 'metadata': data.loc[data.index[0], 'match_date'] = '1999-01-01'
            if kind == 'missing': data = data.iloc[1:]
            if kind == 'duplicate': data = pd.concat([data, data.iloc[:1]])
            if kind == 'label': data.loc[data.index[0], base.TARGET] = 9
            with self.assertRaises(ValueError): diagnostics.validate_data(data, self.splits)

    def test_feature_and_model_hash_checks(self):
        rows = copy.deepcopy(self.registry)
        rows[0]['model_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'model changed'):
            diagnostics.score_models(self.root, self.data, self.splits, rows, self.reference, self.config)
        rows = copy.deepcopy(self.registry)
        rows[1]['features'] = rows[1]['features'][::-1]
        with self.assertRaisesRegex(ValueError, 'allowlist'):
            diagnostics.score_models(self.root, self.data, self.splits, rows, self.reference, self.config)

    def test_verified_read_and_tampered_or_missing_artifacts(self):
        with patch.object(diagnostics, 'prepare', return_value=self.prepared):
            self.assertEqual(diagnostics.verified_output(self.root), self.summary)
            with patch.object(diagnostics, 'sha256', return_value='tampered'):
                with self.assertRaisesRegex(ValueError, 'artifact changed'):
                    diagnostics.verified_output(self.root)
        wrong = list(self.prepared); wrong[1] = {}
        with patch.object(diagnostics, 'prepare', return_value=wrong):
            with self.assertRaisesRegex(ValueError, 'stale'):
                diagnostics.verified_output(self.root)
        with tempfile.TemporaryDirectory() as folder, patch.object(diagnostics, 'run') as run:
            with self.assertRaisesRegex(FileNotFoundError, 'RUN_DEVELOPMENT_DIAGNOSTICS'):
                diagnostics.verified_output(folder)
            run.assert_not_called()

    def test_loader_uses_existing_development_contract_and_adds_interactions(self):
        with patch.object(base, 'prepare', return_value=(None, None, None, None, self.splits, self.raw)):
            data, splits = diagnostics.load_development(self.root)
        pd.testing.assert_frame_equal(data, self.data)
        pd.testing.assert_frame_equal(splits, self.splits)
        self.assertEqual(set(data.split), {'train', 'validation'})

    def test_notebook_diagnostics_work_without_test_evaluation(self):
        nb = json.loads((ROOT/'notebooks/08_probability_evaluation.ipynb').read_text())
        cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        self.assertIn('RUN_DEVELOPMENT_DIAGNOSTICS = False', cells[0])
        namespace = {'__name__': 'development_test'}
        with patch.object(Path, 'cwd', return_value=ROOT), patch.object(importlib, 'reload', side_effect=lambda x: x):
            exec(compile(cells[0], '08_setup', 'exec'), namespace)
        displayed = []
        namespace.update(ROOT=self.root, display=displayed.append)
        with patch.object(diagnostics, 'prepare', return_value=self.prepared), \
             patch.object(diagnostics, 'run') as run, \
             patch.object(stage, 'verified_output', side_effect=AssertionError('Test results not required')):
            exec(compile(cells[1], '08_diagnostics', 'exec'), namespace)
            run.assert_not_called()
        self.assertEqual(len(displayed[0]), 4)
        self.assertIn('event_log_loss_gap', displayed[0].columns)
        self.assertIn('<details ', displayed[1].data)


if __name__ == '__main__':
    unittest.main()
