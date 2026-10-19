"""No-pair endings must be supported by player identities, not wicket inflation."""
import copy
import unittest

from test_data_preparation import fixture, delivery, evaluate
from data_preparation import audit_batting_exhaustion


def injury_fixture(absent=True, retired=False):
    doc = fixture(first_runs=1)
    players = [f"P{i}" for i in range(11)]
    doc["info"]["players"] = {"India": players}
    doc["info"]["outcome"] = {"winner": "Australia", "by": {"runs": 120}}
    stream = []
    if retired:
        stream.append(delivery(batter="P8" if absent else "P9", non_striker="P10",
                              wickets=[{"player_out": "P8" if absent else "P9", "kind": "retired hurt"}]))
    count = 10 - int(absent) - int(retired)
    for i in range(count):
        stream.append(delivery(batter=f"P{i}", non_striker="P10",
                              wickets=[{"player_out": f"P{i}", "kind": "bowled"}]))
    inn = doc["innings"][1]
    inn["overs"] = [{"over": k // 6, "deliveries": stream[k:k + 6]} for k in range(0, len(stream), 6)]
    if absent:
        inn["absent_hurt"] = ["P9"]
    return doc


def evidence(doc):
    return audit_batting_exhaustion(doc["info"], doc["innings"][1])


class InjuryCompletionTests(unittest.TestCase):
    def test_three_supported_endings_keep_true_wicket_count(self):
        for absent, retired, wickets in [(True, False, 9), (False, True, 9), (True, True, 8)]:
            with self.subTest(absent=absent, retired=retired):
                doc = injury_fixture(absent, retired)
                original = copy.deepcopy(doc)
                decision, clean, reports, _ = evaluate(doc)
                self.assertEqual(decision["status"], "included")
                self.assertEqual(reports[1]["final_wickets"], wickets)
                self.assertEqual(clean.iloc[-1].innings_wickets, wickets)
                self.assertEqual(reports[1]["audit_completion_basis"], "no_available_batting_pair")
                self.assertTrue(reports[1]["audit_no_batting_pair"])
                self.assertFalse(any(c.startswith('audit_remaining') for c in clean))
                self.assertEqual(doc, original)

    def test_missing_or_duplicate_or_oversized_roster_stays_pending(self):
        for players in ([], ["P0"] * 11, [f"P{i}" for i in range(10)], [f"P{i}" for i in range(12)]):
            doc = injury_fixture()
            doc["info"]["players"]["India"] = players
            self.assertEqual(evaluate(doc)[0]["status"], "pending_review")

    def test_absent_player_cannot_also_have_batted(self):
        doc = injury_fixture()
        doc["innings"][1]["absent_hurt"] = ["P0"]
        self.assertIn("completion_availability_conflict", evidence(doc)["audit_completion_issues"])
        self.assertFalse(evidence(doc)["audit_no_batting_pair"])

    def test_unknown_or_repeated_absence_is_not_accepted(self):
        for absent in (["Unknown"], ["P9", "P9"]):
            doc = injury_fixture()
            doc["innings"][1]["absent_hurt"] = absent
            self.assertFalse(evidence(doc)["audit_no_batting_pair"])

    def test_duplicate_dismissal_is_not_an_extra_unavailable_player(self):
        doc = injury_fixture()
        doc["innings"][1]["overs"][-1]["deliveries"][-1]["wickets"][0]["player_out"] = "P0"
        self.assertFalse(evidence(doc)["audit_no_batting_pair"])
        self.assertIn("completion_exit_conflict", evidence(doc)["audit_completion_issues"])

    def test_returning_batter_removes_temporary_unavailability(self):
        doc = injury_fixture(False, True)
        # P9 returns as non-striker immediately after retiring, and stays not out.
        for over in doc["innings"][1]["overs"]:
            for d in over["deliveries"]:
                if d["batter"] != "P9":
                    d["non_striker"] = "P9"
        result = evidence(doc)
        self.assertFalse(result["audit_no_batting_pair"])
        self.assertEqual(result["audit_unreturned_retirements"], "[]")
        self.assertIn('"P9"', result["audit_return_events"])
        self.assertEqual(evaluate(doc)[0]["status"], "pending_review")

    def test_retired_out_is_a_dismissal_not_temporary_absence(self):
        doc = injury_fixture(False, True)
        doc["innings"][1]["overs"][0]["deliveries"][0]["wickets"][0]["kind"] = "retired out"
        decision, _, reports, _ = evaluate(doc)
        self.assertEqual(decision["status"], "included")
        self.assertEqual(reports[1]["final_wickets"], 10)
        self.assertEqual(reports[1]["audit_completion_basis"], "ten_dismissals")

    def test_events_after_exhaustion_are_not_truncated(self):
        doc = injury_fixture()
        doc["innings"][1]["overs"][-1]["deliveries"].append(delivery(batter="P10", non_striker="P8"))
        self.assertIn("completion_events_after_no_batting_pair", evidence(doc)["audit_completion_issues"])
        self.assertEqual(evaluate(doc)[0]["status"], "pending_review")

    def test_replacements_need_separate_review(self):
        doc = injury_fixture()
        doc["innings"][1]["overs"][0]["deliveries"][0]["replacements"] = {"match": [{"in": "New", "out": "P9"}]}
        self.assertIn("completion_replacements_need_review", evidence(doc)["audit_completion_issues"])
        self.assertEqual(evaluate(doc)[0]["status"], "pending_review")

    def test_wrong_score_still_fails_csv_verification(self):
        from test_data_preparation import csv_events
        doc = injury_fixture()
        frame = csv_events(doc)
        frame.loc[frame.index[-1], "innings_score"] = 5
        self.assertIn("unexplained_csv_score", evaluate(doc, frame)[0]["pending_reasons"])

    def test_no_extra_terminal_features_in_delivery_table(self):
        _, clean, _, _ = evaluate(injury_fixture())
        self.assertNotIn("audit_exhaustion_event", clean)
        self.assertNotIn("absent_hurt", clean)


if __name__ == "__main__":
    unittest.main()
