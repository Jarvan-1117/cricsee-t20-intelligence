"""Stage 10 source accounting, null semantics and atomic package tests."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import player_history_export as stage


def fixture():
    names = [('a', 'A'), ('n', 'A'), ('dnb', 'A'), ('b', 'B'), ('p', 'B')]
    registry = pd.DataFrame([dict(match_id='10', match_date='2024-01-01', gender='female',
        team=team, player_name=pid.upper(), player_id=pid) for pid, team in names])
    rows = []
    for seq, runs, extras, kind, out, nonboundary, value in [
        (1, 4, {}, '', '', False, .1),
        (2, 0, {'wides': 2}, '', '', False, None),
        (3, 6, {'noballs': 1}, '', '', False, None),
        (4, 0, {'byes': 2}, '', '', False, None),
        (5, 0, {}, 'run out', 'N', False, None),
        (6, 4, {}, '', '', True, 0.)]:
        rows.append(dict(source_row=seq, match_id='10', match_innings=1, event_sequence=seq,
            gender='female', match_date='2024-01-01', batting_team='A', bowling_team='B',
            striker='A', striker_id='a', non_striker='N' if seq < 6 else 'DNB',
            non_striker_id='n' if seq < 6 else 'dnb', bowler='B', bowler_id='b',
            runs_batter=runs, runs_total=runs+sum(extras.values()), non_boundary=nonboundary,
            legal_delivery=int(not extras.get('wides', 0) and not extras.get('noballs', 0)),
            wicket_kind=kind, player_out=out, player_allocated=value is not None,
            striker_partial_nwc=value, bowler_partial_nwc=-value if value is not None else None,
            is_terminal_event=False, **{k: extras.get(k, 0) for k in stage.EXTRAS}))
    events = pd.DataFrame(rows)
    records = [dict(match_id='10', gender='female', team=team, player_id=pid,
        role=role, role_events=6, allocated_events=2, unallocated_events=4,
        partial_nwc=value, terminal_partial_nwc=0., allocation_coverage=2/6)
        for pid, team, role, value in [('a', 'A', 'batting', .1), ('b', 'B', 'bowling', -.1)]]
    frames = dict(registry=registry, lineups=registry.copy(), nwc=pd.DataFrame(records))
    docs = {'10': {'info': dict(gender='female', dates=['2024-01-01'], teams=['A', 'B'],
        players={'A': ['A', 'N', 'DNB'], 'B': ['B', 'P']},
        registry={'people': {pid.upper(): pid for pid, _ in names}}, outcome={'winner': 'A'})}}
    return events, frames, docs


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.events, self.frames, self.docs = fixture()

    def result(self):
        rows, stats = stage.build_history(self.events, self.frames, self.docs)
        return {r['player_id']: r for r in rows}, stats

    def test_wide_and_no_ball_faces_and_boundary_flag(self):
        rows, _ = self.result()
        a = rows['a']['batting']
        self.assertEqual((a['runs'], a['balls_faced'], a['fours'], a['sixes']), (14, 5, 1, 1))
        self.assertEqual(a['strike_rate'], 280.)

    def test_conceded_legal_balls_dots_and_runout(self):
        rows, _ = self.result()
        b = rows['b']['bowling']
        self.assertEqual((b['runs_conceded'], b['legal_balls_bowled'], b['dot_balls'], b['wickets']), (17, 4, 1, 0))
        self.assertEqual((b['overs_display'], b['economy']), ('0.4', 25.5))
        self.assertEqual(rows['n']['batting']['dismissal'], 'run out')
        self.assertEqual(rows['n']['batting']['dismissals'], 1)
        self.assertEqual(rows['a']['batting']['dismissals'], 0)

    def test_non_striker_only_is_batting_not_zero_contribution(self):
        rows, _ = self.result()
        self.assertTrue(rows['dnb']['batted'])
        self.assertEqual(rows['dnb']['batting']['balls_faced'], 0)
        self.assertIsNone(rows['dnb']['batting']['strike_rate'])
        n = rows['dnb']['batting']['nwc']
        self.assertEqual(n['status'], 'unallocated')
        self.assertEqual(n['role_events'], 0)
        self.assertIsNone(n['partial_nwc'])
        self.assertIsNone(n['allocation_coverage'])

    def test_did_not_bat_or_bowl(self):
        rows, _ = self.result()
        p = rows['p']
        self.assertFalse(p['batted']); self.assertFalse(p['bowled'])
        self.assertEqual(p['batting']['dismissal'], 'did not bat')
        self.assertEqual(p['batting']['nwc']['status'], 'not_participated')
        self.assertIsNone(p['batting']['runs'])
        self.assertIsNone(p['bowling']['economy'])

    def test_catalogue_only_is_unknown(self):
        self.frames['lineups'] = self.frames['lineups'][self.frames['lineups'].player_id.ne('p')]
        rows, _ = self.result()
        self.assertIsNone(rows['p']['batted'])
        self.assertEqual(rows['p']['batting']['nwc']['status'], 'unavailable')

    def test_zero_legal_balls_has_no_economy(self):
        self.events['legal_delivery'] = 0
        rows, _ = self.result()
        self.assertTrue(rows['b']['bowled'])
        self.assertEqual(rows['b']['bowling']['overs_display'], '0.0')
        self.assertIsNone(rows['b']['bowling']['economy'])

    def test_bowler_wicket_not_retirement(self):
        for kind, expected in [('bowled', 1), ('caught', 1), ('stumped', 1),
                               ('retired out', 0), ('retired hurt', 0), ('obstructing the field', 0)]:
            with self.subTest(kind=kind):
                self.events.loc[4, ['wicket_kind', 'player_out']] = [kind, 'A']
                rows, _ = self.result()
                self.assertEqual(rows['b']['bowling']['wickets'], expected)
                self.assertEqual(rows['a']['batting']['dismissals'], int(kind != 'retired hurt'))

    def test_unknown_exit_rejected(self):
        self.events.loc[4, 'wicket_kind'] = 'mystery'
        with self.assertRaisesRegex(ValueError, 'Unknown dismissal'):
            self.result()

    def test_retirement_return_keeps_exit_history_but_finishes_not_out(self):
        self.events.loc[4, ['wicket_kind', 'player_out']] = ['retired hurt', 'A']
        rows, _ = self.result()
        self.assertEqual(rows['a']['batting']['dismissal'], 'not out')
        self.assertEqual(rows['a']['batting']['exits'][0]['kind'], 'retired hurt')
        self.assertEqual(rows['a']['batting']['dismissals'], 0)

    def test_wrong_identity_and_duplicate_registry_rejected(self):
        self.events.loc[0, 'striker_id'] = 'b'
        with self.assertRaisesRegex(ValueError, 'registry mismatch'):
            self.result()
        self.events, self.frames, self.docs = fixture()
        self.frames['registry'] = pd.concat([self.frames['registry'], self.frames['registry'].iloc[:1]])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.result()

    def test_nwc_missing_role_rejected(self):
        self.frames['nwc'] = self.frames['nwc'].iloc[:1]
        with self.assertRaisesRegex(ValueError, 'Missing Stage 09'):
            self.result()

    def test_nwc_total_counts_coverage_and_terminal_rejected(self):
        for column, value in [('partial_nwc', .8), ('role_events', 99),
                              ('allocation_coverage', .9), ('terminal_partial_nwc', 1.)]:
            with self.subTest(column=column):
                self.events, self.frames, self.docs = fixture()
                self.frames['nwc'].loc[0, column] = value
                with self.assertRaises(ValueError):
                    self.result()

    def test_null_and_real_zero_remain_distinct(self):
        source = dict(role_events=1, allocated_events=1, unallocated_events=0,
            partial_nwc=0., terminal_partial_nwc=0., allocation_coverage=1.)
        result = stage.nwc_record(source, [(0., True)], True)
        self.assertEqual(result['status'], 'available'); self.assertEqual(result['partial_nwc'], 0.)
        source.update(allocated_events=0, unallocated_events=1, partial_nwc=float('nan'), allocation_coverage=0.)
        result = stage.nwc_record(source, [(float('nan'), True)], True)
        self.assertEqual(result['status'], 'unallocated'); self.assertIsNone(result['partial_nwc'])
        self.assertIsNone(result['terminal_partial_nwc'])

    def test_json_is_strict_and_csv_projects_both_roles(self):
        rows, _ = stage.build_history(self.events, self.frames, self.docs)
        payload = stage.product_payload(rows, self.docs, 'a'*64,
            dict(attribution_id='09', upstream_evaluation_id='08'))
        json.dumps(payload, allow_nan=False)
        self.assertEqual(len(stage.role_table(rows, 'batting')), len(rows))
        self.assertEqual(len(stage.role_table(rows, 'bowling')), len(rows))
        self.assertEqual(payload['matches'][0]['match_id'], '10')


class SourceTests(unittest.TestCase):
    def document(self):
        return {'info': {'balls_per_over': 6}, 'innings': [
            {'team': 'A', 'overs': [{'over': 0, 'deliveries': [
                {'batter': 'A', 'non_striker': 'N', 'bowler': 'B',
                 'runs': {'batter': 4, 'extras': 0, 'total': 4, 'non_boundary': True}}]}]},
            {'team': 'B', 'overs': []}]}

    def test_source_non_boundary_and_display_coordinates(self):
        doc = self.document()
        raw = stage.source_events(doc, '10')
        clean = raw.drop(columns='non_boundary')
        result = stage.attach_source_flags(clean, {'10': doc})
        self.assertTrue(result.non_boundary.iloc[0])
        clean.loc[0, 'runs_batter'] = 6
        with self.assertRaisesRegex(ValueError, 'disagree'):
            stage.attach_source_flags(clean, {'10': doc})

    def test_duplicate_or_missing_source_event_rejected(self):
        doc = self.document(); clean = stage.source_events(doc, '10').drop(columns='non_boundary')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            stage.attach_source_flags(pd.concat([clean, clean]), {'10': doc})
        clean.loc[0, 'event_sequence'] = 2
        with self.assertRaisesRegex(ValueError, 'coverage'):
            stage.attach_source_flags(clean, {'10': doc})

    def test_multiple_exits_fail_closed(self):
        doc = self.document()
        doc['innings'][0]['overs'][0]['deliveries'][0]['wickets'] = [
            {'kind': 'retired hurt', 'player_out': 'N'}, {'kind': 'bowled', 'player_out': 'A'}]
        with self.assertRaisesRegex(ValueError, 'Multiple exits'):
            stage.source_events(doc, '10')


class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.events, self.frames, docs = fixture()
        self.rows, self.stats = stage.build_history(self.events, self.frames, docs)
        # Explicit expectations, not recomputed through the exporter.
        self.frames['contributions'] = pd.DataFrame([
            dict(match_id='10', player_id=pid, batting_runs=runs, balls_faced=balls,
                 batting_dismissals=out, legal_balls_bowled=legal, bowling_runs_conceded=conceded,
                 bowler_wickets=0, bowling_dot_balls=dots)
            for pid, runs, balls, out, legal, conceded, dots in [
                ('a', 14, 5, 0, 0, 0, 0), ('n', 0, 0, 1, 0, 0, 0),
                ('dnb', 0, 0, 0, 0, 0, 0), ('b', 0, 0, 0, 4, 17, 1),
                ('p', 0, 0, 0, 0, 0, 0)]])
        self.frames['teams'] = pd.DataFrame([
            dict(match_id='10', team=team, player_interval_total=sign*.1,
                 team_residual=sign*.2, expected_probability_change=sign*.3)
            for team, sign in [('A', 1), ('B', -1)]])
        self.frames['innings'] = pd.DataFrame([dict(match_id='10', match_innings=1,
            events=6, actual_legal_balls=4, penalty_runs_pre=0, penalty_runs_post=0, official_total=19)])

    def check(self):
        return stage.verify_totals(self.rows, self.stats, self.frames, self.events)

    def test_independent_stats_and_team_residuals_pass(self):
        result = self.check()
        self.assertEqual(result['status'], 'passed')
        self.assertLess(result['max_team_closure_error'], 1e-10)

    def test_wrong_statistic_rejected(self):
        self.frames['contributions'].loc[0, 'batting_runs'] = 15
        with self.assertRaisesRegex(ValueError, 'statistic mismatch'):
            self.check()

    def test_missing_player_statistic_rejected(self):
        self.frames['contributions'] = self.frames['contributions'].iloc[:-1]
        with self.assertRaisesRegex(ValueError, 'coverage mismatch'):
            self.check()

    def test_team_residual_mismatch_rejected(self):
        self.frames['teams'].loc[0, 'team_residual'] = .3
        with self.assertRaisesRegex(ValueError, 'reconciliation failed'):
            self.check()

    def test_innings_total_mismatch_rejected(self):
        self.frames['innings'].loc[0, 'official_total'] = 20
        with self.assertRaisesRegex(ValueError, 'Innings accounting'):
            self.check()


class NotebookTests(unittest.TestCase):
    def test_current_joint_attribution_inputs_and_schema(self):
        import nwc_attribution_joint
        self.assertIs(stage.attribution, nwc_attribution_joint)
        self.assertEqual(stage.SCHEMA_VERSION, 2)
        self.assertEqual(stage.LEDGER, nwc_attribution_joint.DATA_DIR + '/event_ledger.csv.gz')
        self.assertEqual(stage.PLAYER_NWC, nwc_attribution_joint.REPORT_DIR + '/player_match_partial_nwc.csv')
        self.assertEqual(stage.TEAM_NWC, nwc_attribution_joint.REPORT_DIR + '/team_reconciliation.csv')
        self.assertIn(nwc_attribution_joint.REPORT_DIR + '/manifest.json', stage.SOURCES)
        self.assertFalse(any('fixed_ibw' in name for name in stage.SOURCES))

    def test_notebook_compiles_and_defaults_to_verified_read_mode(self):
        notebook = json.loads((ROOT/'notebooks/10_results_and_visualisation.ipynb').read_text(encoding='utf-8'))
        self.assertEqual(notebook['nbformat'], 4)
        sources = []
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                compile(source, f'cell{index}', 'exec')
                sources.append(source)
        joined = '\n'.join(sources)
        self.assertIn('RUN_EXPORT = False', joined)
        self.assertIn('stage.verified_output(ROOT)', joined)
        self.assertIn("history_view('batting'", joined)
        self.assertIn("history_view('bowling'", joined)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        events, frames, docs = fixture()
        rows, _ = stage.build_history(events, frames, docs)
        self.payload = stage.product_payload(rows, docs, 'a'*64,
            dict(attribution_id='09', upstream_evaluation_id='08'))

    def test_verified_package_and_idempotent_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = stage.publish(root, self.payload, {'status': 'passed'}, {'source': 'hash'})
            stage.verify_package(output, 'a'*64)
            before = (output/'manifest.json').read_bytes()
            again = stage.publish(root, self.payload, {'status': 'passed'}, {'source': 'hash'})
            self.assertEqual(output, again)
            self.assertEqual(before, (again/'manifest.json').read_bytes())
            (output/'history.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                stage.verify_package(output, 'a'*64)

    def test_failed_publication_keeps_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage.publish(root, self.payload, {'status': 'passed'}, {})
            pointer = root/stage.REPORT_DIR/'current.json'; before = pointer.read_bytes()
            altered = copy.deepcopy(self.payload); altered['dataset_version'] = 'b'*64
            with patch.object(stage, 'verify_package', side_effect=ValueError('test failure')):
                with self.assertRaises(ValueError):
                    stage.publish(root, altered, {}, {})
            self.assertEqual(before, pointer.read_bytes())
            self.assertFalse((root/stage.REPORT_DIR/'snapshots'/('b'*64)).exists())

    def test_invalid_version_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.payload['dataset_version'] = '../outside'
            with self.assertRaisesRegex(ValueError, 'Invalid dataset version'):
                stage.publish(Path(tmp), self.payload, {}, {})

    def test_read_mode_detects_stale_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); stage.publish(root, self.payload, {}, {'source': 'hash'})
            with patch.object(stage, 'fingerprints', return_value=({'source': 'changed'}, {})):
                with self.assertRaisesRegex(ValueError, 'sources changed'):
                    stage.verified_output(root)


if __name__ == '__main__':
    unittest.main()
