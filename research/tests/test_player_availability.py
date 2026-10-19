"""Small event replays: timing, identity, quota and fail-closed handoff."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import player_availability as pa


def delivery(batter="A1", other="A2", bowler="B1", **extra):
    return {"batter": batter, "non_striker": other, "bowler": bowler,
            "runs": {"batter": 0, "extras": 0, "total": 0}, **extra}


def fixture():
    players = {t: [f"{t}{n}" for n in range(1, 12)] for t in ["A", "B"]}
    doc = {"info": {"players": players, "gender": "male", "dates": ["2020-01-01"],
                   "registry": {"people": {n: "id_" + n for names in players.values() for n in names}}},
           "innings": [{"team": "A", "overs": [{"over": 0, "deliveries": [delivery(), delivery()]}]},
                       {"team": "B", "overs": [{"over": 0, "deliveries": [delivery("B1", "B2", "A1")]}]}]}
    return doc


def inputs(doc):
    rows = []
    sequence = {1: 0, 2: 0}
    for inn, team, over, number, ev in pa.source_events(doc):
        sequence[inn] += 1
        extras = ev.get("extras", {})
        rows.append({"match_id": 1, "source_row": len(rows), "match_innings": inn,
            "event_sequence": sequence[inn], "gender": "male", "match_date": "2020-01-01",
            "batting_team": team, "bowling_team": "B" if team == "A" else "A",
            "source_over": over, "source_delivery_in_over": number, "striker": ev["batter"],
            "non_striker": ev["non_striker"], "bowler": ev["bowler"], "runs_total": ev["runs"]["total"],
            "legal_delivery": int(not(extras.get("wides", 0) or extras.get("noballs", 0)))})
    cat = pd.DataFrame([{"team": t, "player_name": n, "player_id": doc["info"]["registry"]["people"][n]}
                        for t,names in doc["info"]["players"].items() for n in names])
    return pd.DataFrame(rows), cat


def replay(doc):
    return pa.replay_match(doc, *inputs(doc))


def state(frame, row, side, name):
    schema = pa.BAT_SCHEMA if side == "batting" else pa.BOWL_SCHEMA
    return next(dict(zip(schema, a)) for a in json.loads(frame.iloc[row][side + "_state_json"]) if a[0] == "id_" + name)


class AvailabilityTests(unittest.TestCase):
    def test_current_pair_and_pre_event_quota(self):
        frame, _, issues, _ = replay(fixture())
        self.assertEqual(frame.available_batter_count.tolist(), [11, 11, 11])
        self.assertEqual(frame.waiting_batter_count.tolist(), [9, 9, 9])
        self.assertEqual(frame.current_bowler_nominal_capacity.tolist(), [24, 23, 24])
        self.assertEqual(frame.current_bowler_prior_legal_balls.tolist(), [0, 1, 0])
        self.assertEqual(issues, [])

    def test_entry_as_non_striker_is_known_before_facing(self):
        doc = fixture(); ev = doc['innings'][0]['overs'][0]['deliveries']
        ev[0]['wickets'] = [{'player_out': 'A2', 'kind': 'run out'}]
        ev[1]['non_striker'] = 'A3'
        ev.append(delivery(other='A3'))
        frame, _, _, _ = replay(doc)
        first = state(frame, 1, 'batting', 'A3')
        self.assertEqual(first['status'], 'current_batter')
        self.assertEqual(first['previously_observed_batting'], 0)
        self.assertEqual(state(frame, 2, 'batting', 'A3')['previously_observed_batting'], 1)
        self.assertEqual(frame.waiting_batter_count.tolist()[:3], [9, 8, 8])

    def test_current_capacity_after_three_balls_is_21(self):
        doc = fixture()
        doc['innings'][0]['overs'][0]['deliveries'] = [delivery() for _ in range(4)]
        frame, _, _, _ = replay(doc)
        self.assertEqual(frame.iloc[3].current_bowler_nominal_capacity, 21)

    def test_completed_short_over_consumes_a_slot_not_five_balls(self):
        doc = fixture()
        doc['innings'][0]['overs'] = [
            {'over': 0, 'deliveries': [delivery() for _ in range(5)]},
            {'over': 1, 'deliveries': [delivery(bowler='B2')]}]
        frame, _, _, _ = replay(doc)
        b1 = state(frame, 5, 'bowling', 'B1')
        self.assertEqual(b1['personal_legal_balls'], 5)
        self.assertEqual(b1['nominal_capacity'], 18)

    def test_completed_long_over_consumes_a_slot_not_seven_balls(self):
        doc = fixture()
        doc['innings'][0]['overs'] = [
            {'over': 0, 'deliveries': [delivery() for _ in range(7)]},
            {'over': 1, 'deliveries': [delivery(bowler='B2')]}]
        frame, _, _, _ = replay(doc)
        b1 = state(frame, 7, 'bowling', 'B1')
        self.assertEqual(b1['personal_legal_balls'], 7)
        self.assertEqual(b1['nominal_capacity'], 18)

    def test_dismissal_only_changes_next_event(self):
        doc = fixture(); events = doc["innings"][0]["overs"][0]["deliveries"]
        events[0]["wickets"] = [{"player_out": "A1", "kind": "bowled"}]
        events[1]["batter"] = "A3"
        frame, timeline, _, _ = replay(doc)
        self.assertEqual(frame.available_batter_count.tolist()[:2], [11, 10])
        self.assertEqual(state(frame, 0, "batting", "A1")["status"], "current_batter")
        self.assertEqual(state(frame, 1, "batting", "A1")["status"], "dismissed")
        self.assertEqual(timeline[0]["timing"], "after")

    def test_non_striker_run_out(self):
        doc = fixture(); events = doc["innings"][0]["overs"][0]["deliveries"]
        events[0]["wickets"] = [{"player_out": "A2", "kind": "run out"}]
        events[1]["non_striker"] = "A3"
        f, _, _, _ = replay(doc)
        self.assertEqual(state(f, 1, "batting", "A2")["status"], "dismissed")

    def test_retirement_not_wicket_and_observed_return(self):
        doc = fixture(); events = doc["innings"][0]["overs"][0]["deliveries"]
        events[0]["wickets"] = [{"player_out": "A1", "kind": "retired hurt"}]
        events[1]["batter"] = "A3"
        events[1]["wickets"] = [{"player_out": "A3", "kind": "bowled"}]
        events.append(delivery())
        frame, timeline, issues, _ = replay(doc)
        self.assertEqual(frame.dismissed_slot_count.tolist()[:3], [0, 0, 1])
        self.assertEqual(state(frame, 1, "batting", "A1")["status"], "temporarily_unavailable")
        self.assertEqual(state(frame, 2, "batting", "A1")["status"], "current_batter")
        self.assertIn("observed_batting_return", [t["event"] for t in timeline])
        self.assertTrue(any(i["blocks_resources"] for i in issues))

    def test_retired_out_is_dismissal(self):
        doc = fixture(); ev = doc["innings"][0]["overs"][0]["deliveries"]
        ev[0]["wickets"] = [{"player_out": "A1", "kind": "retired out"}]; ev[1]["batter"] = "A3"
        f, _, _, _ = replay(doc)
        self.assertEqual(f.iloc[1].dismissed_slot_count, 1)

    def test_wide_and_noball_do_not_consume_legal_balls(self):
        for key in ["wides", "noballs"]:
            doc = fixture(); ev = doc["innings"][0]["overs"][0]["deliveries"]
            ev[0]["extras"] = {key: 1}; ev[0]["runs"] = {"batter": 0, "extras": 1, "total": 1}
            f, _, _, _ = replay(doc)
            self.assertEqual(f.iloc[1].current_bowler_nominal_capacity, 24)
            self.assertEqual(f.iloc[1].current_bowler_slot_overs_started, 1)

    def test_untimed_absence_does_not_change_features(self):
        base = fixture(); altered = deepcopy(base); altered["innings"][0]["absent_hurt"] = ["A11"]
        before, _, _, _ = replay(base); after, _, issues, _ = replay(altered)
        pd.testing.assert_frame_equal(before, after)
        self.assertEqual(issues[0]["reason"], "untimed_absent_hurt")
        self.assertIsNone(issues[0]["source_row"])

    def test_future_outcome_cannot_change_earlier_snapshot(self):
        doc = fixture(); before, _, _, _ = replay(doc)
        doc["innings"][0]["overs"][0]["deliveries"][-1]["wickets"] = [{"player_out": "A1", "kind": "bowled"}]
        after, _, _, _ = replay(doc)
        pd.testing.assert_frame_equal(before.iloc[:2], after.iloc[:2])

    def test_future_replacement_not_activated_early_and_quota_inherited(self):
        doc = fixture(); doc["info"]["players"]["B"].append("B12"); doc["info"]["registry"]["people"]["B12"] = "id_B12"
        ev = doc["innings"][0]["overs"][0]["deliveries"]
        ev[1]["bowler"] = "B12"
        ev[1]["replacements"] = {"match": [{"team": "B", "out": "B1", "in": "B12", "reason": "concussion_substitute"}]}
        doc["innings"][1]["overs"][0]["deliveries"][0]["batter"] = "B12"
        f, _, issues, initial = replay(doc)
        self.assertNotIn("B12", initial["B"])
        self.assertNotIn("id_B12", f.iloc[0].bowling_state_json)
        sub = state(f, 1, "bowling", "B12")
        self.assertEqual(sub["slot_legal_balls"], 1)
        self.assertEqual(sub["personal_legal_balls"], 0)
        self.assertEqual(sub["nominal_capacity"], 23)
        self.assertTrue(issues[0]["blocks_resources"])

    def test_role_replacement_consumes_two_over_slots(self):
        doc = fixture(); ev = doc["innings"][0]["overs"][0]["deliveries"]
        ev[1]["bowler"] = "B2"; ev[1]["replacements"] = {"role": [{"in": "B2", "out": "B1", "role": "bowler", "reason": "injury"}]}
        f, _, _, _ = replay(doc)
        self.assertEqual(state(f, 1, "bowling", "B1")["slot_overs_started"], 1)
        self.assertEqual(state(f, 1, "bowling", "B2")["slot_overs_started"], 1)
        self.assertEqual(state(f, 1, "bowling", "B1")["nominal_capacity"], 0)

    def test_missing_outgoing_only_inferred_in_same_over(self):
        doc = fixture(); ev = doc["innings"][0]["overs"][0]["deliveries"]
        ev[1]["bowler"] = "B2"; ev[1]["replacements"] = {"role": [{"in": "B2", "role": "bowler", "reason": "injury"}]}
        f, _, issues, _ = replay(doc)
        self.assertEqual(state(f, 1, "bowling", "B1")["status"], "temporarily_unavailable")
        self.assertNotIn("unknown_outgoing_role_player", [i["reason"] for i in issues])
        doc["innings"][0]["overs"].append({"over": 1, "deliveries": [delivery(bowler="B3", replacements={"role": [{"in": "B3", "role": "bowler", "reason": "injury"}]})]})
        _, _, issues, _ = replay(doc)
        self.assertIn("unknown_outgoing_role_player", [i["reason"] for i in issues])

    def test_miscount_no_future_quota_forecast(self):
        doc = fixture(); doc["innings"][0]["overs"][0]["deliveries"] = [delivery() for _ in range(7)]
        f, _, issues, _ = replay(doc)
        self.assertEqual(f.iloc[6].pre_over_legal_balls, 6)
        self.assertEqual(f.iloc[6].current_bowler_nominal_capacity, 18)
        self.assertEqual(f.iloc[6].observed_current_opportunity, 1)
        self.assertFalse(issues[0]["blocks_resources"])
        doc["innings"][0]["miscounted_overs"] = {"0": {"balls": 7}}
        after, _, _, _ = replay(doc)
        pd.testing.assert_frame_equal(f, after)

    def test_capacity_zero_at_fourth_over_seventh_ball(self):
        self.assertEqual(pa.nominal_capacity({0, 2, 4, 6}, 6, 6, True), 0)
        self.assertEqual(pa.nominal_capacity({0, 2, 4, 6}, 7, 0, False), 0)
        self.assertEqual(pa.nominal_capacity({0}, 0, 5, True), 19)
        self.assertEqual(pa.nominal_capacity({0}, 0, 5, False), 18)

    def test_fifth_over_and_consecutive_over_are_reviewed(self):
        doc = fixture(); doc["innings"][0]["overs"] = [{"over": i, "deliveries": [delivery()]} for i in range(5)]
        _, _, issues, _ = replay(doc)
        self.assertIn("observed_quota_exceeded", [i["reason"] for i in issues])
        self.assertIn("consecutive_over_slot", [i["reason"] for i in issues])

    def test_suspended_bowler_cannot_return(self):
        doc = fixture(); ev = doc["innings"][0]["overs"][0]["deliveries"]
        ev[1]["bowler"] = "B2"; ev[1]["replacements"] = {"role": [{"in": "B2", "out": "B1", "role": "bowler", "reason": "excluded - high full pitched balls"}]}
        ev.append(delivery())
        with self.assertRaises(pa.ReplayError): replay(doc)

    def test_dismissed_player_cannot_return(self):
        doc = fixture(); doc["innings"][0]["overs"][0]["deliveries"][0]["wickets"] = [{"player_out": "A1", "kind": "bowled"}]
        with self.assertRaises(pa.ReplayError): replay(doc)

    def test_identity_mismatch_stops_not_cold_start(self):
        doc = fixture(); frame, cat = inputs(doc); cat.loc[0, "player_id"] = "wrong"
        with self.assertRaises(pa.ReplayError): pa.replay_match(doc, frame, cat)

    def test_reordered_source_stops(self):
        doc = fixture(); frame, cat = inputs(doc)
        with self.assertRaises(pa.ReplayError): pa.replay_match(doc, frame.iloc[::-1], cat)

    def test_incomplete_lineup_stops(self):
        doc = fixture(); doc["info"]["players"]["A"].remove("A11")
        with self.assertRaises(pa.ReplayError): replay(doc)

    def test_unexplained_departure_requires_review(self):
        doc = fixture(); doc["innings"][0]["overs"][0]["deliveries"][1]["batter"] = "A3"
        f, _, issues, _ = replay(doc)
        self.assertEqual(f.iloc[0].known_uncertainty, 0)
        self.assertEqual(f.iloc[1].known_uncertainty, 1)
        self.assertIn("unexplained_batting_departure", [i["reason"] for i in issues])

    def test_resource_gate_rejects_unknown_review_and_empty_requests(self):
        audit = pd.DataFrame({"match_id": [1, 2], "ready_for_resource_features": [True, False]})
        self.assertEqual(pa.require_ready_matches(audit, [1]), {1})
        for ids in [[], [2], [3], [1, 2]]:
            with self.assertRaises(ValueError): pa.require_ready_matches(audit, ids)

    def test_config_and_contract(self):
        self.assertEqual(pa.read_config(ROOT)["stage"], pa.STAGE)
        self.assertEqual(pa.feature_contract()["batting_state_schema"], pa.BAT_SCHEMA)
        self.assertIn('no innings-budget allocation', pa.feature_contract()['capacity_aggregation'])
        self.assertIn('waiting_not_yet_batted', pa.feature_contract()['never_entered_resource_pool'])


if __name__ == "__main__":
    unittest.main()
