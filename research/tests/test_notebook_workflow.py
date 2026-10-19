"""Notebook presentation contract: current method, saved text and required fields."""
import ast
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


def visible_text(notebook):
    parts = []
    for cell in notebook['cells']:
        parts.append(''.join(cell['source']))
        for output in cell.get('outputs', []):
            if 'text' in output:
                parts.append(''.join(output['text']))
            for mime in ['text/plain', 'text/html']:
                value = output.get('data', {}).get(mime, '')
                parts.append(''.join(value) if isinstance(value, list) else value)
    return '\n'.join(parts)


class NotebookWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebooks = {p.name[:2]:json.loads(p.read_text(encoding='utf-8'))
                         for p in sorted((ROOT/'notebooks').glob('*.ipynb'))}

    def test_all_ten_notebooks_compile_without_saved_errors(self):
        self.assertEqual(set(self.notebooks), {f'{i:02d}' for i in range(1,11)})
        for name, notebook in self.notebooks.items():
            for index, cell in enumerate(notebook['cells']):
                with self.subTest(notebook=name, cell=index):
                    if cell['cell_type'] == 'code':
                        ast.parse(''.join(cell['source']))
                    self.assertFalse(any(o.get('output_type') == 'error'
                                         for o in cell.get('outputs', [])))

    def test_no_obsolete_selection_or_migration_claims_in_source_or_saved_text(self):
        patterns = [r'fixed[- ]IBW', r'fixes IBW', r'fixed to IBW',
                    r'within-IBW validation winners', r'does not select a final feature group',
                    r'Stage 09 requires a separate update', r'Higher allocation coverage reflects a changed convention',
                    r'new-design arms', r'修订后的实验|旧版输出另存|规则保持不变',
                    r'complete 01-09 workflow']
        for name, notebook in self.notebooks.items():
            text = visible_text(notebook)
            for pattern in patterns:
                with self.subTest(notebook=name, pattern=pattern):
                    self.assertIsNone(re.search(pattern, text, re.I))

    def test_both_candidate_stages_describe_validation_only_eight_group_selection(self):
        for name in ['06', '07']:
            sources = '\n'.join(''.join(c['source']) for c in self.notebooks[name]['cells'])
            self.assertIn('Stage 08', sources)
            self.assertIn('eight', sources)
            self.assertIn('validation', sources)

    def test_no_repeated_completion_instructions_in_results(self):
        phrases = ['After rebuilding: set RUN_STAGE', 'After the first successful run',
                   'After successful execution, set RUN_ATTRIBUTION',
                   'Audit only: no clean dataset was exported by this notebook.']
        for name, notebook in self.notebooks.items():
            for phrase in phrases:
                with self.subTest(notebook=name, phrase=phrase):
                    self.assertNotIn(phrase, visible_text(notebook))

    def test_batting_history_includes_count_and_exit_status_in_code_and_output(self):
        notebook = self.notebooks['10']
        cell = next(c for c in notebook['cells']
                    if "display(history_view('batting'" in ''.join(c['source']))
        source = ''.join(cell['source'])
        self.assertIn("'dismissals'", source)
        self.assertIn("'dismissal'", source)
        self.assertTrue(any('dismissals' in ''.join(o.get('data', {}).get('text/html', []))
                            for o in cell.get('outputs', [])))


if __name__ == '__main__': unittest.main()
