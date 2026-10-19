"""Stage 05 v2 arithmetic, membership, exclusions and fail-closed contracts."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import remaining_resource_features as rf
import game_state_features as game_state
from test_player_availability import fixture, replay, inputs, delivery


def resource_case():
    doc = fixture()
    events, _, _, _ = replay(doc)
    _, cat = inputs(doc)
    profiles = cat.copy()
    profiles['match_id'] = 1
    profiles['gender'] = 'male'
    profiles['match_date'] = '2020-01-01'
    profiles['history_last_date'] = '2019-12-01'
    profiles['prior_balls_faced'] = 100
    profiles['prior_legal_balls_bowled'] = 30
    profiles['prior_listed_matches'] = 5
    for metric, value in zip(rf.METRICS, [100., 20., .1, .4, 8., .04, .4]):
        profiles['reference_' + metric] = value
        profiles['smoothed_' + metric] = value
    return events, profiles, cat


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.config = rf.read_config(ROOT)
        self.events, self.profiles, self.lineups = resource_case()

    def prepared(self):
        return rf.match_profiles(self.profiles, self.lineups, self.config)

    def compute(self, row=None, lookup=None):
        row = self.events.iloc[0].to_dict() if row is None else row
        return rf.event_resources(SimpleNamespace(**row), self.prepared() if lookup is None else lookup, self.config)

    def test_reference_players_have_nonzero_sums_in_original_units(self):
        out, detail = self.compute()
        np.testing.assert_allclose([out[c] for c in rf.BAT_FEATURES], [900, 180, .9, 3.6])
        np.testing.assert_allclose([out[c] for c in rf.BOWL_FEATURES], np.array([8, .04, .4]) * 44)
        self.assertEqual(out['audit_total_remaining_overs'], 44)
        self.assertEqual(out['audit_batting_sum_error'] + out['audit_bowling_sum_error'], 0)

    def test_current_pair_quality_has_no_batting_resource_effect(self):
        before, _ = self.compute()
        mask = self.profiles.player_name.isin(['A1', 'A2'])
        self.profiles.loc[mask, 'smoothed_batting_strike_rate'] = 999
        after, _ = self.compute()
        self.assertEqual(before, after)

    def test_waiting_batter_quality_is_summed_not_averaged(self):
        self.profiles.loc[self.profiles.player_name.eq('A3'), 'smoothed_batting_strike_rate'] = 150
        out, _ = self.compute()
        self.assertEqual(out[rf.BAT_FEATURES[0]], 950)

    def test_batting_and_bowling_quality_are_role_separate(self):
        before, _ = self.compute()
        self.profiles.loc[self.profiles.team.eq('A'), 'smoothed_bowling_economy'] = 99
        self.profiles.loc[self.profiles.team.eq('B'), 'smoothed_batting_strike_rate'] = 999
        after, _ = self.compute()
        self.assertEqual(before, after)

    def test_current_bowler_quality_remains_in_weighted_capacity(self):
        before, _ = self.compute()
        self.profiles.loc[self.profiles.player_name.eq('B1'), 'smoothed_bowling_economy'] = 10
        after, _ = self.compute()
        self.assertEqual(after[rf.BOWL_FEATURES[0]] - before[rf.BOWL_FEATURES[0]], 4 * (10 - 8))

    def test_economy_and_batting_dot_rate_are_not_sign_flipped(self):
        before, _ = self.compute()
        self.profiles.loc[self.profiles.player_name.eq('A3'), 'smoothed_batting_dot_rate'] = .6
        self.profiles.loc[self.profiles.player_name.eq('B1'), 'smoothed_bowling_economy'] = 10
        after, _ = self.compute()
        self.assertAlmostEqual(after[rf.BAT_FEATURES[3]] - before[rf.BAT_FEATURES[3]], .2)
        self.assertGreater(after[rf.BOWL_FEATURES[0]], before[rf.BOWL_FEATURES[0]])

    def test_returned_batter_is_excluded_without_changing_stage04_count(self):
        row = self.events.iloc[0].to_dict()
        states = json.loads(row['batting_state_json'])
        s = next(s for s in states if s[0] == 'id_A3')
        s[2], s[3] = 'waiting_returned', 1
        row['batting_state_json'] = json.dumps(states)
        out, _ = self.compute(row)
        self.assertEqual(out['audit_waiting_batter_count'], 8)
        self.assertEqual(out['audit_excluded_returned_count'], 1)
        self.assertEqual(out[rf.BAT_FEATURES[0]], 800)

    def test_unavailable_replaced_and_dismissed_batters_excluded(self):
        for status in ['temporarily_unavailable', 'replaced_out', 'dismissed']:
            with self.subTest(status=status):
                row = self.events.iloc[0].to_dict()
                states = json.loads(row['batting_state_json'])
                next(s for s in states if s[0] == 'id_A3')[2] = status
                row.update(batting_state_json=json.dumps(states), waiting_batter_count=8, available_batter_count=10)
                out, _ = self.compute(row)
                self.assertEqual(out[rf.BAT_FEATURES[0]], 800)

    def test_new_non_striker_removed_before_facing_a_ball(self):
        doc = fixture()
        ev = doc['innings'][0]['overs'][0]['deliveries']
        ev[0]['wickets'] = [{'player_out': 'A2', 'kind': 'run out'}]
        ev[1]['non_striker'] = 'A3'
        frame, _, _, _ = replay(doc)
        out, detail = self.compute(frame.iloc[1].to_dict())
        self.assertEqual(out['audit_waiting_batter_count'], 8)
        a3 = next(s for s in json.loads(detail['batting_resource_json']) if s[0] == 'id_A3')
        self.assertEqual(a3[1:4], ['current_batter', 0, 0])

    def test_inconsistent_entry_flag_fails_closed(self):
        row = self.events.iloc[0].to_dict()
        states = json.loads(row['batting_state_json'])
        next(s for s in states if s[0] == 'id_A3')[3] = 1
        row['batting_state_json'] = json.dumps(states)
        with self.assertRaisesRegex(ValueError, 'prior entry'): self.compute(row)

    def test_empty_waiting_pool_zero_sums(self):
        row = self.events.iloc[0].to_dict()
        states = json.loads(row['batting_state_json'])
        for s in states:
            if s[2] != 'current_batter': s[2] = 'dismissed'
        row.update(batting_state_json=json.dumps(states), waiting_batter_count=0, available_batter_count=2)
        out, _ = self.compute(row)
        self.assertEqual(sum(out[c] for c in rf.BAT_FEATURES), 0)

    def test_three_completed_legal_balls_leave_21_for_current_bowler(self):
        doc = fixture()
        doc['innings'][0]['overs'][0]['deliveries'] = [delivery() for _ in range(4)]
        frame, _, _, _ = replay(doc)
        out, detail = self.compute(frame.iloc[3].to_dict())
        b1 = next(s for s in json.loads(detail['bowling_resource_json']) if s[0] == 'id_B1')
        self.assertEqual(b1[2:4], [21, 3.5])
        self.assertEqual(b1[-3], 3.5 * 8)
        self.assertEqual(out['audit_current_bowler_capacity'], 21)

    def test_special_over_uses_slots_not_24_minus_legal(self):
        doc = fixture()
        doc['innings'][0]['overs'] = [
            {'over': 0, 'deliveries': [delivery() for _ in range(7)]},
            {'over': 1, 'deliveries': [delivery(bowler='B2')]}]
        frame, _, _, _ = replay(doc)
        _, detail = self.compute(frame.iloc[7].to_dict())
        b1 = next(s for s in json.loads(detail['bowling_resource_json']) if s[0] == 'id_B1')
        self.assertEqual(b1[2:4], [18, 3])  # NOT 24 - 7 = 17.

    def test_no_usage_allocation_or_budget_normalization(self):
        before, _ = self.compute()
        row = self.events.iloc[0].to_dict()
        row['nominal_scheduled_slots_remaining'] = 1
        self.profiles['prior_listed_matches'] = 999999
        self.profiles['prior_legal_balls_bowled'] = 999999
        self.profiles[[c for c in self.profiles if c.startswith('reference_')]] = 999
        after, _ = self.compute(row)
        np.testing.assert_array_equal([before[c] for c in rf.BAT_FEATURES + rf.BOWL_FEATURES], [after[c] for c in rf.BAT_FEATURES + rf.BOWL_FEATURES])

    def test_zero_nominal_budget_does_not_erase_personal_capacity(self):
        row = self.events.iloc[0].to_dict()
        row['nominal_scheduled_slots_remaining'] = 0
        out, _ = self.compute(row)
        self.assertEqual(out['audit_zero_nominal_budget'], 1)
        self.assertEqual(out[rf.BOWL_FEATURES[0]], 352)

    def test_exhausted_or_unavailable_bowlers_contribute_zero(self):
        for status in ['quota_exhausted', 'temporarily_unavailable', 'replaced_out', 'suspended']:
            row = self.events.iloc[0].to_dict()
            states = json.loads(row['bowling_state_json'])
            s = next(s for s in states if s[0] == 'id_B3')
            s[2], s[-1] = status, 0
            row.update(bowling_state_json=json.dumps(states), bowlers_with_nominal_capacity=10)
            lookup = self.prepared()
            lookup['id_B3']['quality'][4] = 1000
            out, _ = self.compute(row, lookup)
            self.assertEqual(out[rf.BOWL_FEATURES[0]], 320)

    def test_all_zero_capacities_zero_bowling_sums(self):
        row = self.events.iloc[0].to_dict()
        states = json.loads(row['bowling_state_json'])
        for s in states:
            s[-1] = 0
            if s[2] != 'current_bowler': s[2] = 'quota_exhausted'
        row.update(bowling_state_json=json.dumps(states), bowlers_with_nominal_capacity=0, current_bowler_nominal_capacity=0)
        out, _ = self.compute(row)
        self.assertEqual(sum(out[c] for c in rf.BOWL_FEATURES), 0)

    def test_invalid_capacities_rejected(self):
        for cap in [-1, 25, .5, float('nan')]:
            row = self.events.iloc[0].to_dict()
            states = json.loads(row['bowling_state_json']); states[0][-1] = cap
            row['bowling_state_json'] = json.dumps(states)
            with self.assertRaises(ValueError): self.compute(row)

    def test_unavailable_player_with_positive_capacity_rejected(self):
        row = self.events.iloc[0].to_dict(); states = json.loads(row['bowling_state_json'])
        next(s for s in states if s[0] == 'id_B3')[2] = 'quota_exhausted'
        row['bowling_state_json'] = json.dumps(states)
        with self.assertRaises(ValueError): self.compute(row)

    def test_per_player_reconciliation_detects_tampering(self):
        out, details = self.compute()
        for side in ['batting', 'bowling']:
            altered = deepcopy(details); records = json.loads(altered[side + '_resource_json'])
            records[0][-1] += 1; altered[side + '_resource_json'] = json.dumps(records)
            with self.assertRaises(ValueError): rf.reconcile_details(out, altered, 1e-8)
        out[rf.BOWL_FEATURES[0]] += 1
        with self.assertRaises(ValueError): rf.reconcile_details(out, details, 1e-8)

    def test_random_capacity_weighted_sums(self):
        rng = np.random.default_rng(47)
        for _ in range(60):
            row = self.events.iloc[0].to_dict(); states = json.loads(row['bowling_state_json']); lookup = self.prepared()
            expected = np.zeros(3)
            for s in states:
                s[-1] = int(rng.integers(0, 25))
                q = np.array([rng.uniform(1, 20), rng.random(), rng.random()])
                lookup[s[0]]['quality'][4:] = q
                expected += s[-1] / 6 * q
            row.update(bowling_state_json=json.dumps(states), current_bowler_nominal_capacity=next(s[-1] for s in states if s[2] == 'current_bowler'), bowlers_with_nominal_capacity=sum(s[-1] > 0 for s in states))
            out, _ = self.compute(row, lookup)
            np.testing.assert_allclose([out[c] for c in rf.BOWL_FEATURES], expected)

    def test_same_date_or_future_quality_history_rejected(self):
        for date in ['2020-01-01', '2020-02-01']:
            self.profiles['history_last_date'] = date
            with self.assertRaises(ValueError): self.prepared()

    def test_nonfinite_negative_or_invalid_quality_rejected(self):
        for value in [np.nan, -1, np.inf]:
            self.profiles['smoothed_batting_average'] = value
            with self.assertRaises(ValueError): self.prepared()

    def test_missing_identity_and_uncertainty_fail_closed(self):
        lookup = self.prepared(); del lookup['id_A3']
        with self.assertRaises(ValueError): self.compute(lookup=lookup)
        row = self.events.iloc[0].to_dict(); row['known_uncertainty'] = 1
        with self.assertRaises(ValueError): self.compute(row)

    def test_future_outcomes_do_not_change_earlier_resources(self):
        doc = fixture(); before, _, _, _ = replay(doc)
        doc['innings'][0]['overs'][0]['deliveries'][1]['wickets'] = [{'player_out': 'A1', 'kind': 'bowled'}]
        after, _, _, _ = replay(doc)
        self.assertEqual(self.compute(before.iloc[0].to_dict()), self.compute(after.iloc[0].to_dict()))

    def test_v2_five_arms_exclude_direct_quality_and_evidence(self):
        c = rf.feature_contract(game_state.feature_contract())
        self.assertEqual(c['contract_version'], rf.VERSION)
        self.assertEqual(len(c['main_arms']), 5)
        self.assertEqual(c['main_arms']['constant'], [])
        self.assertEqual(c['auxiliary_arms'], {})
        base = c['main_arms']['state']
        for name, extra in [('batting_resources', rf.BAT_FEATURES), ('bowling_resources', rf.BOWL_FEATURES), ('both_resources', rf.BAT_FEATURES + rf.BOWL_FEATURES)]:
            self.assertEqual(c['main_arms'][name], base + extra)
        for fields in c['main_arms'].values():
            self.assertEqual(len(fields), len(set(fields)))
            self.assertFalse(set(fields) & set(rf.quality.feature_contract()['current_player_block']))
            self.assertFalse(any(f.startswith('audit_') for f in fields))
            self.assertFalse(set(fields) & set(rf.CAPACITY_FEATURES + rf.STATIC_FEATURES + rf.ALT_FEATURES + rf.INTERACTIONS))

    def test_poisoned_base_contract_rejected(self):
        base = deepcopy(game_state.feature_contract())
        base['base_numeric'] += rf.quality.feature_contract()['current_player_block'][:1]
        with self.assertRaises(ValueError): rf.feature_contract(base)

    def test_missing_export_requires_explicit_build(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(FileNotFoundError): rf.verified_output(Path(name))

    def test_legacy_or_stale_export_never_rebuilt_automatically(self):
        for extra in [{}, {'contract_version': rf.VERSION, 'input_sha256': {'old': 'old'}}]:
            with tempfile.TemporaryDirectory() as name:
                root = Path(name); directory = root / rf.DATA_DIR; directory.mkdir(parents=True)
                (directory / 'manifest.json').write_text(json.dumps({'status': 'complete', 'input_sha256': {}, **extra}))
                with patch.object(rf, 'input_fingerprints', return_value=({}, {})), patch.object(rf, 'run') as rebuild:
                    with self.assertRaises(ValueError): rf.verified_output(root)
                    rebuild.assert_not_called()

    def test_corrupt_export_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); directory = root / rf.DATA_DIR; directory.mkdir(parents=True)
            artifact = directory / 'resource.csv'; artifact.write_text('changed')
            (directory / 'manifest.json').write_text(json.dumps({'status': 'complete', 'contract_version': rf.VERSION, 'input_sha256': {}, 'output_sha256': {artifact.relative_to(root).as_posix(): 'wrong'}}))
            with patch.object(rf, 'input_fingerprints', return_value=({}, {})):
                with self.assertRaises(ValueError): rf.verified_output(root)


if __name__ == '__main__':
    unittest.main()
