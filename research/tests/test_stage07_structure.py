"""Read-only checks for the symmetric four-way Stage 07 notebook."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StructureTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads((ROOT / 'notebooks/07_resource_representation_experiments.ipynb').read_text(encoding='utf-8-sig'))

    def test_four_current_representations(self):
        text = '\n'.join(''.join(c['source']) for c in self.notebook['cells'])
        for name in ['RBW', 'IB', 'IW', 'IBW']:
            self.assertIn(name, text)
        self.assertNotIn('RUN_E2', text)
        self.assertNotIn('static_resource_experiment', text)

    def test_code_compiles_without_execution(self):
        for index, cell in enumerate(self.notebook['cells']):
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']), filename=f'notebook07-cell-{index}')

    def test_explicit_switches_false_and_v2_status(self):
        setup = ast.parse(''.join(self.notebook['cells'][1]['source']))
        switches = {}
        for node in ast.walk(setup):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.startswith('RUN_'):
                        switches[target.id] = ast.literal_eval(node.value)
        self.assertEqual(switches, {'RUN_STAGE': False, 'RUN_PREFLIGHT': False})
        self.assertIn('resource_sums_v2', ''.join(self.notebook['cells'][0]['source']))

    def test_local_experiment_imports_exist(self):
        setup = ast.parse(''.join(self.notebook['cells'][1]['source']))
        names = {alias.name for node in ast.walk(setup) if isinstance(node, ast.Import) for alias in node.names}
        expected = {'resource_interaction_experiments', 'resource_interaction_visualisation'}
        self.assertTrue(expected.issubset(names))
        for name in expected:
            self.assertTrue((ROOT / 'src' / (name + '.py')).is_file())

    def test_versioned_independent_output_paths(self):
        source = (ROOT / 'src/resource_interaction_experiments.py').read_text(encoding='utf-8')
        self.assertIn('four_way_interactions_v1', source)
        self.assertIn('VERSION = common.VERSION', source)

    def test_current_four_way_results_complete(self):
        config = json.loads((ROOT / 'config/resource_interaction_experiments.json').read_text(encoding='utf-8'))
        version = Path(config['contract_version']) / config['design_version']
        report = Path('outputs/07_resource_representation_experiments') / version
        data = Path('data/processed/resource_experiments') / version
        manifest = json.loads((ROOT / report / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['status'], 'complete')
        required = [report / name for name in [
            'experiment_summary.json', 'selected_models.csv', 'selected_systems.csv',
            'validation_comparisons.csv', 'feature_groups.json', 'frozen_protocol.json',
            'input_audit.csv', 'phase_diagnostics.csv', 'preflight.json']]
        required += [report / code / 'manifest.json' for code in ['ib', 'iw', 'ibw']]
        required.append(data / 'selected_validation_predictions.csv.gz')
        for relative in required:
            with self.subTest(artifact=relative.as_posix()):
                self.assertIn(relative.as_posix(), manifest['output_sha256'])
                path = ROOT / relative
                self.assertTrue(path.is_file(), f'Missing current Stage 07 artifact: {relative}')
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                 manifest['output_sha256'][relative.as_posix()])
        summary = json.loads((ROOT / report / 'experiment_summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(summary['experiment_id'], manifest['experiment_id'])
        self.assertEqual(summary['contract_version'], config['contract_version'])
        self.assertEqual(summary['design_version'], config['design_version'])
        self.assertEqual(summary['arms'], ['RBW', 'IB', 'IW', 'IBW'])
        with (ROOT / report / 'selected_models.csv').open(encoding='utf-8', newline='') as handle:
            models = list(csv.DictReader(handle))
        expected = {(f'{gender}_innings_{innings}', arm)
                    for gender in ['female', 'male'] for innings in [1, 2]
                    for arm in ['RBW', 'IB', 'IW', 'IBW']}
        self.assertEqual(len(models), len(expected))
        self.assertEqual({(row['track'], row['arm_code']) for row in models}, expected)
        for row in models:
            with self.subTest(track=row['track'], arm=row['arm_code']):
                self.assertTrue((ROOT / row['model_path']).is_file(), row['model_path'])


if __name__ == '__main__':
    unittest.main()
