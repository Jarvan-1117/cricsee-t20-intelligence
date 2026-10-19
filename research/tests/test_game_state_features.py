"""Stage 02 timing, orientation, history and artifact-contract regression tests."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import game_state_features as gs

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    events, matches, meta = [], [], []
    for match_id, date, opponent, gender in [(1, "2020-01-01", "B", "male"), (2, "2020-01-02", "C", "male")]:
        matches.append({"match_id": match_id, "match_date": date, "gender": gender})
        first_total = 0
        for no, team, bowling in [(1, "A", opponent), (2, opponent, "A")]:
            score = 5 if no == 1 else 0
            wickets = legal_count = 0
            plays = [(0, 1, 1, 0), (0, 0, 1, 0), (0, 1, 4, 1), (1, 1, 0, 0), (6, 1, 2, 0), (15, 1, 0, 0)]
            if no == 2: plays = [(0, 1, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0)]
            for seq, (over, legal, runs, wicket) in enumerate(plays, 1):
                score += runs; wickets += wicket; legal_count += legal
                events.append({"match_id": match_id, "source_row": len(events), "match_date": date,
                    "gender": gender, "match_innings": no, "event_sequence": seq, "source_over": over,
                    "batting_team": team, "bowling_team": bowling, "striker": team + "1", "non_striker": team + "2",
                    "bowler": bowling + "1", "match_winner": "A", "toss_winner": "A", "toss_decision": "bat",
                    "legal_delivery": legal, "runs_total": runs, "innings_score": score,
                    "innings_wickets": wickets, "cumulative_legal_deliveries": legal_count})
            if no == 1: first_total = score + 7
            meta.append({"match_id": match_id, "match_innings": no, "batting_team": team,
                         "penalty_runs_pre": 5 if no == 1 else 0, "penalty_runs_post": 7 if no == 1 else 0,
                         "official_total": first_total if no == 1 else score,
                         "official_target": np.nan if no == 1 else first_total + 1,
                         "miscounted_overs": "{}", "absent_hurt": "[]"})
    return pd.DataFrame(events), pd.DataFrame(matches), pd.DataFrame(meta)


class GameStateTests(unittest.TestCase):
    def setUp(self):
        self.config = gs.read_config(ROOT)
        self.d, self.m, self.i = fixture()

    def build(self):
        return gs.build_features(self.d, self.m, self.i, self.config)

    def test_complete_feature_contract_and_fixed_target(self):
        frame, _ = self.build()
        self.assertTrue(all(gs.validate_features(frame, self.m, len(self.d)).values()))
        self.assertTrue(frame[gs.TARGET].eq(1).all())
        self.assertTrue(set(gs.BASE_NUMERIC + gs.BASE_CATEGORICAL + gs.RECENT_CANDIDATES + gs.IDENTITY_CANDIDATES) <= set(frame))
        self.assertNotIn(gs.TARGET, gs.BASE_NUMERIC)
        self.assertNotIn("toss_elected_bat", gs.BASE_NUMERIC)

    def test_pre_event_score_and_wicket(self):
        frame, _ = self.build()
        local = frame.query('match_id == 1 and match_innings == 1')
        self.assertEqual(local.pre_score.tolist(), [5, 6, 7, 11, 11, 13])
        self.assertEqual(local.wickets_remaining.tolist(), [10, 10, 10, 9, 9, 9])
        self.assertEqual(local.runs_prev_10_events.tolist(), [0, 1, 2, 6, 6, 8])
        self.assertEqual(local.wickets_prev_10_events.tolist(), [0, 0, 0, 1, 1, 1])

    def test_illegal_deliveries_are_events_not_legal_balls(self):
        frame, _ = self.build()
        local = frame.query('match_id == 1 and match_innings == 1')
        self.assertEqual(local.legal_balls_elapsed.iloc[1:3].tolist(), [1, 1])
        self.assertEqual(local.audit_recent_window_events.iloc[1:3].tolist(), [1, 2])

    def test_first_event_wide_and_penalty_do_not_make_infinite_rate(self):
        index = self.d.query('match_id == 1 and match_innings == 1').index
        self.d.loc[index[0], 'legal_delivery'] = 0
        self.d.loc[index, 'cumulative_legal_deliveries'] -= 1
        frame, _ = self.build()
        first_two = frame.query('match_id == 1 and match_innings == 1').head(2)
        self.assertTrue(first_two.no_legal_ball_yet.eq(1).all())
        self.assertTrue(first_two.current_run_rate.eq(0).all())

    def test_penalty_target_is_only_available_in_second_innings(self):
        frame, _ = self.build()
        self.assertTrue(frame.loc[frame.match_innings.eq(1), 'target_runs'].isna().all())
        self.assertEqual(frame.query('match_id == 1 and match_innings == 2').target_runs.iloc[0], 21)
        self.assertEqual(frame.query('match_id == 1 and match_innings == 1').pre_score.iloc[-1], 13)

    def test_phase_uses_observed_over_not_total_legal_balls(self):
        frame, _ = self.build()
        local = frame.query('match_id == 1 and match_innings == 1')
        self.assertEqual(local.innings_phase.tolist(), ['powerplay'] * 4 + ['middle', 'death'])
        self.assertEqual(local.scheduled_balls_remaining_estimate.iloc[-2:].tolist(), [84, 30])

    def test_clock_never_uses_future_miscount_or_absence_metadata(self):
        before, _ = self.build()
        self.i['miscounted_overs'] = '{"0": {"balls": 7}}'
        self.i['absent_hurt'] = '["A9"]'
        self.i['official_total'] = 999
        after, _ = self.build()
        pd.testing.assert_frame_equal(before, after)

    def test_zero_slot_miscount_rate_is_finite_and_flagged(self):
        idx = self.d.query('match_id == 1 and match_innings == 2').index
        # Synthetic already-observed seventh legal event in the final over.
        base = self.d.loc[idx[0]].copy()
        additions = []
        for seq in range(1, 8):
            row = base.copy(); row['event_sequence'] = seq; row['source_row'] = 100 + seq
            row['source_over'] = 19; row['legal_delivery'] = 1; row['runs_total'] = 0
            row['innings_score'] = 0; row['innings_wickets'] = 0; row['cumulative_legal_deliveries'] = seq
            additions.append(row)
        self.d = pd.concat([self.d.drop(idx), pd.DataFrame(additions)], ignore_index=True)
        frame, _ = self.build()
        last = frame.query('match_id == 1 and match_innings == 2').iloc[-1]
        self.assertEqual(last.scheduled_balls_remaining_estimate, 0)
        self.assertTrue(last.audit_rate_denominator_floored)
        self.assertTrue(np.isfinite(last.required_run_rate))

    def test_changing_current_outcome_does_not_change_current_predictors(self):
        before, _ = self.build()
        idx = self.d.query('match_id == 1 and match_innings == 1').index
        self.d.loc[idx[2], 'runs_total'] += 2
        self.d.loc[idx[2:], 'innings_score'] += 2
        after, _ = self.build()
        columns = gs.BASE_NUMERIC + gs.BASE_CATEGORICAL + gs.RECENT_CANDIDATES
        pd.testing.assert_frame_equal(before.loc[idx[:3], columns], after.loc[idx[:3], columns])

    def test_window_is_ten_prior_events_and_resets(self):
        f = pd.DataFrame({'match_id': [1] * 15, 'match_innings': [1] * 13 + [2] * 2})
        sums = gs.prior_event_sum(f, pd.Series([1] * 15))
        self.assertEqual(sums.tolist(), list(range(11)) + [10, 10, 0, 1])

    def test_sorting_preserves_source_event_order(self):
        before, _ = self.build()
        self.d = self.d.sample(frac=1, random_state=2)
        after, _ = self.build()
        pd.testing.assert_frame_equal(before, after)

    def test_duplicate_or_misordered_source_fails(self):
        self.d.loc[1, 'source_row'] = self.d.loc[0, 'source_row']
        with self.assertRaises(ValueError): self.build()

    def test_current_match_result_does_not_change_its_background(self):
        _, before = self.build()
        self.d.loc[self.d.match_id.eq(1), 'match_winner'] = 'B'
        _, after = self.build()
        columns = [c for c in before if '_prior_' in c or '_recent_' in c]
        pd.testing.assert_series_equal(before.loc[0, columns], after.loc[0, columns])
        self.assertNotEqual(before.loc[1, 'first_team_prior_win_rate'], after.loc[1, 'first_team_prior_win_rate'])

    def test_same_day_results_cannot_update_each_other(self):
        self.d['match_date'] = '2020-01-01'; self.m['match_date'] = '2020-01-01'
        _, background = self.build()
        self.assertTrue(background.first_team_prior_matches.eq(0).all())
        self.assertTrue(background.first_team_prior_win_rate.eq(.5).all())
        self.assertTrue(background.first_team_recent_win_rate.eq(.5).all())

    def test_histories_are_gender_specific(self):
        self.d.loc[self.d.match_id.eq(2), 'gender'] = 'female'
        self.m.loc[self.m.match_id.eq(2), 'gender'] = 'female'
        _, background = self.build()
        self.assertTrue(background.first_team_prior_matches.eq(0).all())

    def test_history_count_and_laplace_rate(self):
        _, background = self.build()
        self.assertEqual(background.first_team_prior_matches.tolist(), [0, 1])
        self.assertAlmostEqual(background.first_team_prior_win_rate.iloc[1], 2/3)
        self.assertEqual(background.second_team_prior_win_rate.iloc[1], .5)

    def test_recent_history_decays_and_same_date_order_is_irrelevant(self):
        records = gs.match_records(self.d, self.m, self.i)
        rows = []
        for i in range(25):
            row = records.iloc[0].copy(); row['match_id'] = i
            row['match_date'] = pd.Timestamp('2020-01-01') + pd.Timedelta(days=i)
            row['match_winner'] = 'A' if i < 12 else 'B'; rows.append(row)
        h = gs.team_history(pd.DataFrame(rows), self.config)
        self.assertLess(h.first_team_recent_win_rate.iloc[-1], h.first_team_prior_win_rate.iloc[-1])
        rows[1]['match_date'] = rows[0]['match_date']
        a = gs.team_history(pd.DataFrame(rows), self.config).sort_values('match_id').reset_index(drop=True)
        b = gs.team_history(pd.DataFrame(rows).iloc[::-1], self.config).sort_values('match_id').reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)

    def test_stale_output_never_rebuilds_automatically(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / gs.DATA_DIR; folder.mkdir(parents=True)
            (folder / 'manifest.json').write_text(json.dumps({'status': 'complete', 'input_sha256': {'x': 'old'}}))
            with patch.object(gs, 'input_fingerprints', return_value=({'x': 'new'}, {})):
                with self.assertRaisesRegex(ValueError, 'no automatic rebuild'): gs.verified_output(temp)

    def test_tampered_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / gs.DATA_DIR; folder.mkdir(parents=True)
            (folder / 'table.csv').write_text('changed')
            (folder / 'manifest.json').write_text(json.dumps({'status': 'complete', 'input_sha256': {},
                'output_sha256': {gs.DATA_DIR + '/table.csv': 'incorrect'}}))
            with patch.object(gs, 'input_fingerprints', return_value=({}, {})):
                with self.assertRaisesRegex(ValueError, 'output changed'): gs.verified_output(temp)


if __name__ == '__main__':
    unittest.main()
