"""Terminal diagnostics check accounting, never infer calibration from outliers."""
import copy
import unittest

from test_nwc_attribution import fixture
import nwc_attribution as attribution
import nwc_terminal_audit as audit


class TerminalAuditTests(unittest.TestCase):
    def setUp(self):
        self.events, self.meta = fixture()
        self.ledger = attribution.build_ledger(self.events, self.meta)
        self.sources = {
            mid: (self.events[self.events.match_id.eq(mid)].fillna('').to_dict('records'),
                  self.meta[self.meta.match_id.eq(mid)].to_dict('records'), 'B', set())
            for mid in [10, 20]}

    def result(self, threshold=.25):
        return audit.compare_cases(self.events, self.ledger, self.meta, self.sources, threshold)

    def test_valid_accounting_and_probability_perspectives(self):
        before = self.ledger.copy(deep=True)
        cases = self.result()
        self.assertEqual(len(cases), 2)
        self.assertTrue(cases.all_checks_pass.all())
        self.assertTrue(cases.first_team_probability_pct.eq(40).all())
        self.assertTrue(cases.chasing_team_probability_pct.eq(60).all())
        self.assertTrue(cases.terminal_change_pp.eq(-40).all())
        self.assertTrue(self.ledger.equals(before))

    def test_no_selected_cases(self):
        self.assertTrue(self.result(.5).empty)

    def test_threshold_is_inclusive(self):
        self.assertEqual(len(self.result(.4)), 2)

    def test_invalid_threshold(self):
        for value in [-1, 2, float('nan')]:
            with self.assertRaises(ValueError): self.result(value)

    def test_small_stored_delta_does_not_hide_case(self):
        self.ledger.loc[self.ledger.is_terminal_event, 'interval_wpa_first_team'] = .01
        cases = self.result()
        self.assertEqual(len(cases), 2)
        self.assertFalse(cases.wpa_arithmetic_ok.any())

    def test_wrong_endpoint_and_raw_probability_detected(self):
        mask = self.ledger.is_terminal_event
        self.ledger.loc[mask, 'interval_endpoint_probability'] = .2
        self.ledger.loc[mask, 'raw_model_probability'] = .1
        cases = self.result()
        self.assertFalse(cases.terminal_endpoint_ok.any())
        self.assertFalse(cases.raw_probability_preserved.any())

    def test_source_delivery_conflict_reported(self):
        self.sources = copy.deepcopy(self.sources)
        self.sources[10][0][-1]['runs_total'] = 3
        case = self.result().set_index('match_id').loc[10]
        self.assertFalse(case.delivery_ok)
        self.assertFalse(case.pre_state_ok)

    def test_source_target_and_winner_conflicts_reported(self):
        rows, meta, _, issues = self.sources[10]
        meta[-1]['official_target'] = 5
        self.sources[10] = rows, meta, 'A', issues
        case = self.result().set_index('match_id').loc[10]
        self.assertFalse(case.target_ok)
        self.assertFalse(case.winner_ok)

    def test_miscount_capacity_is_audit_only(self):
        self.sources[10][1][-1]['miscounted_overs'] = '{"0": {"balls": 7}}'
        case = self.result().set_index('match_id').loc[10]
        self.assertEqual(case.audited_scheduled_legal_balls_remaining, 119)

    def test_decoder_issue_cannot_pass(self):
        self.sources[10][-1].add('unexplained_over_length')
        case = self.result().set_index('match_id').loc[10]
        self.assertFalse(case.source_decode_ok)
        self.assertFalse(case.all_checks_pass)


if __name__ == '__main__':
    unittest.main()
