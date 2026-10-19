"""Boundary tests for Stage 01; fixtures never read or modify project raw data."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_preparation import (audit_match, decode_states, parse_boolean, read_config,
                              rooted, run, sha256, verified_audit)

CONFIG = read_config(ROOT)
META = {"match_date": "2020-01-01", "gender": "male", "game": "Australia vs India", "metadata_issue": ""}


def delivery(score=0, **updates):
    row = {"batter": "Batter One", "non_striker": "Batter Two", "bowler": "Bowler",
           "runs": {"batter": score, "extras": 0, "total": score}}
    row.update(updates)
    return row


def fixture(first_runs=0, last_over_balls=6):
    overs = [{"over": n, "deliveries": [delivery(first_runs) for _ in range(last_over_balls if n == 19 else 6)]} for n in range(20)]
    total = sum(d["runs"]["total"] for o in overs for d in o["deliveries"])
    return {"info": {"gender": "male", "match_type": "T20", "balls_per_over": 6, "overs": 20,
                     "teams": ["Australia", "India"], "dates": ["2020-01-01"],
                     "toss": {"winner": "Australia", "decision": "bat"}, "outcome": {"winner": "India"}},
            "innings": [{"team": "Australia", "overs": overs},
                        {"team": "India", "target": {"overs": 20, "runs": total + 1},
                         "overs": [{"over": 0, "deliveries": [delivery(total + 1)]}]}]}


def csv_events(doc):
    rows, _, _ = decode_states(doc)
    frame = pd.DataFrame(rows)
    frame["source_row"] = range(len(frame))
    frame["match_id"] = 1
    frame["over"] = frame.source_over
    frame["ball"] = frame.source_delivery_in_over
    frame["bowling_team"] = frame.batting_team.map({"Australia": "India", "India": "Australia"})
    frame["match_winner"] = doc["info"]["outcome"].get("winner", "India")
    frame["winning_outcome"] = "10 wickets"
    frame["toss_winner"] = "Australia"
    frame["toss_decision"] = "bat"
    frame["venue"] = "Example"
    return frame


def evaluate(doc, frame=None, meta=None):
    return audit_match(csv_events(doc) if frame is None else frame, META if meta is None else meta, doc, CONFIG)


class EventAuditTests(unittest.TestCase):
    def test_normal_successful_chase_is_complete_without_120_balls(self):
        decision, clean, innings, changes = evaluate(fixture())
        self.assertEqual(decision["status"], "included")
        self.assertEqual(len(clean), 121)
        self.assertEqual(innings[1]["actual_legal_balls"], 1)
        self.assertEqual(changes, [])

    def test_complete_defended_target(self):
        doc = fixture(first_runs=1)
        doc["innings"][1]["overs"] = [{"over": n, "deliveries": [delivery() for _ in range(6)]} for n in range(20)]
        doc["info"]["outcome"]["winner"] = "Australia"
        self.assertEqual(evaluate(doc)[0]["status"], "included")

    def test_all_out_before_twenty_overs(self):
        doc = fixture()
        doc["innings"][0]["overs"] = [{"over": n, "deliveries": [delivery(wickets=[{"kind": "bowled", "player_out": f"Player {n}-{j}"}]) for j in range(count)]} for n, count in enumerate([6, 4])]
        self.assertEqual(evaluate(doc)[0]["status"], "included")

    def test_documented_119_and_121_ball_innings_are_retained(self):
        for n in (5, 7):
            with self.subTest(n=n):
                doc = fixture(last_over_balls=n)
                doc["innings"][0]["miscounted_overs"] = {"19": {"balls": n}}
                decision, _, innings, _ = evaluate(doc)
                self.assertEqual(decision["status"], "included")
                self.assertEqual(innings[0]["actual_legal_balls"], 114 + n)

    def test_unexplained_121_balls_needs_review(self):
        decision = evaluate(fixture(last_over_balls=7))[0]
        self.assertEqual(decision["status"], "pending_review")
        self.assertIn("unexplained_over_length", decision["pending_reasons"])

    def test_integer_string_miscount_is_supported_and_preserved(self):
        doc = fixture(last_over_balls=7)
        doc["innings"][0]["miscounted_overs"] = {"19": {"balls": "7"}}
        decision, _, innings, _ = evaluate(doc)
        self.assertEqual(decision["status"], "included")
        self.assertEqual(innings[0]["actual_legal_balls"], 121)
        self.assertEqual(json.loads(innings[0]["miscounted_overs"])["19"]["balls"], "7")

    def test_invalid_miscount_not_silently_coerced(self):
        for value in ("unknown", "7.5", -1, 0, True, None):
            with self.subTest(value=value):
                doc = fixture()
                frame = csv_events(doc)
                doc["innings"][0]["miscounted_overs"] = {"19": {"balls": value}}
                self.assertIn("invalid_miscount_metadata", evaluate(doc, frame)[0]["pending_reasons"])

    def test_injury_completion_is_explicitly_pending_without_counting_wicket(self):
        doc = fixture()
        doc["innings"][0]["overs"] = doc["innings"][0]["overs"][:18]
        doc["innings"][0]["absent_hurt"] = ["Injured Player"]
        decision, _, innings, _ = evaluate(doc)
        self.assertEqual(decision["status"], "pending_review")
        self.assertIn("injury_related_completion_needs_review", decision["pending_reasons"])
        self.assertEqual(innings[0]["final_wickets"], 0)

    def test_miscount_metadata_must_match(self):
        doc = fixture(last_over_balls=7)
        doc["innings"][0]["miscounted_overs"] = {"19": {"balls": 5}}
        self.assertIn("miscount_metadata_mismatch", evaluate(doc)[0]["pending_reasons"])

    def test_wide_retained_without_advancing_legal_clock(self):
        doc = fixture()
        doc["innings"][0]["overs"][0]["deliveries"].insert(0, delivery(extras={"wides": 1}, runs={"batter": 0, "extras": 1, "total": 1}))
        doc["innings"][1]["target"]["runs"] = 2
        doc["innings"][1]["overs"][0]["deliveries"] = [delivery(2)]
        decision, clean, _, _ = evaluate(doc)
        self.assertEqual(decision["status"], "included")
        self.assertEqual(clean.iloc[0].cumulative_legal_deliveries, 0)
        self.assertEqual(len(clean), 122)

    def test_no_ball_is_not_legal(self):
        doc = fixture()
        doc["innings"][0]["overs"][0]["deliveries"].insert(0, delivery(extras={"noballs": 1}, runs={"batter": 0, "extras": 1, "total": 1}))
        doc["innings"][1]["target"]["runs"] = 2
        doc["innings"][1]["overs"][0]["deliveries"] = [delivery(2)]
        decision, clean, _, _ = evaluate(doc)
        self.assertEqual(decision["status"], "included")
        self.assertEqual(clean.iloc[0].legal_delivery, 0)

    def test_display_coordinates_do_not_reorder_events(self):
        doc = fixture()
        frame = csv_events(doc)
        frame.loc[1, ["over", "ball"]] = [0, 1]
        decision, clean, _, changes = evaluate(doc, frame)
        self.assertEqual(decision["status"], "included")
        self.assertEqual(clean.source_row.tolist(), list(range(len(frame))))
        self.assertEqual(decision["display_coordinate_repeats"], 1)
        self.assertTrue(any(c["field"] == "ball" for c in changes))

    def test_unexplained_changed_event_is_not_repaired(self):
        doc = fixture()
        frame = csv_events(doc)
        frame.loc[0, "runs_total"] = 3
        decision, clean, _, changes = evaluate(doc, frame)
        self.assertEqual(decision["status"], "pending_review")
        self.assertIsNone(clean)
        self.assertEqual(changes, [])

    def test_post_innings_penalty_changes_target_not_delivery_score(self):
        doc = fixture()
        doc["innings"][0]["penalty_runs"] = {"post": 5}
        doc["innings"][1]["target"]["runs"] = 6
        doc["innings"][1]["overs"][0]["deliveries"] = [delivery(6)]
        decision, clean, reports, _ = evaluate(doc)
        self.assertEqual(decision["status"], "included")
        self.assertEqual(clean.loc[clean.match_innings.eq(1)].innings_score.iloc[-1], 0)
        self.assertEqual(reports[0]["official_total"], 5)
        self.assertEqual(reports[1]["official_target"], 6)

    def test_pre_innings_penalty_is_a_logged_correction(self):
        doc = fixture()
        doc["innings"][0]["penalty_runs"] = {"pre": 5}
        doc["innings"][1]["target"]["runs"] = 6
        doc["innings"][1]["overs"][0]["deliveries"] = [delivery(6)]
        frame = csv_events(doc)
        frame.loc[frame.match_innings.eq(1), "innings_score"] = 0
        decision, clean, _, changes = evaluate(doc, frame)
        self.assertEqual(decision["status"], "included")
        self.assertEqual(clean.iloc[0].innings_score, 5)
        self.assertEqual(clean.iloc[0].original_innings_score, 0)
        self.assertTrue(any(c["field"] == "innings_score" for c in changes))

    def test_retirements_have_distinct_wicket_semantics(self):
        for kind, wickets in (("retired hurt", 0), ("retired not out", 0), ("retired out", 1)):
            with self.subTest(kind=kind):
                doc = fixture()
                doc["innings"][0]["overs"][0]["deliveries"][0]["wickets"] = [{"kind": kind, "player_out": "Batter One"}]
                frame = csv_events(doc)
                frame.loc[0, ["player_out", "wicket_kind"]] = ""
                frame.loc[0, "is_wicket"] = False
                frame.loc[frame.match_innings.eq(1), "innings_wickets"] = 0
                decision, clean, _, changes = evaluate(doc, frame)
                self.assertEqual(decision["status"], "included")
                self.assertEqual(clean.iloc[0].innings_wickets, wickets)
                self.assertTrue(changes)

    def test_true_string_false_string_and_missing(self):
        self.assertFalse(parse_boolean("FALSE"))
        self.assertTrue(parse_boolean("TRUE"))
        self.assertIsNone(parse_boolean(None))
        doc = fixture()
        frame = csv_events(doc)
        frame["is_wicket"] = "FALSE"
        self.assertEqual(evaluate(doc, frame)[0]["status"], "included")

    def test_missing_player_out_without_wicket_is_normal(self):
        doc = fixture()
        frame = csv_events(doc)
        frame["player_out"], frame["wicket_kind"] = None, None
        self.assertEqual(evaluate(doc, frame)[0]["status"], "included")

    def test_missing_core_value_not_filled_with_zero(self):
        doc = fixture()
        frame = csv_events(doc)
        frame.loc[0, "runs_total"] = float("nan")
        self.assertEqual(evaluate(doc, frame)[0]["status"], "pending_review")

    def test_events_after_target_are_not_truncated(self):
        doc = fixture()
        doc["innings"][1]["overs"][0]["deliveries"].append(delivery())
        decision, clean, _, _ = evaluate(doc)
        self.assertIn("events_after_successful_chase", decision["pending_reasons"])
        self.assertIsNone(clean)

    def test_incomplete_innings_needs_review(self):
        doc = fixture()
        doc["innings"][0]["overs"] = doc["innings"][0]["overs"][:18]
        self.assertIn("innings_completion_unresolved", evaluate(doc)[0]["pending_reasons"])

    def test_special_formats_excluded_not_labelled_corrupt(self):
        for case in ("dls", "shortened", "tie", "no_result", "super_over", "awarded"):
            with self.subTest(case=case):
                doc = fixture()
                frame = csv_events(doc)
                if case == "dls": doc["info"]["outcome"]["method"] = "D/L"
                elif case == "shortened": doc["innings"][1]["target"]["overs"] = 15
                elif case == "tie": doc["info"]["outcome"] = {"result": "tie"}
                elif case == "no_result": doc["info"]["outcome"] = {"result": "no result"}
                elif case == "super_over": doc["innings"][1]["super_over"] = True
                elif case == "awarded": frame["winning_outcome"] = "awarded"
                self.assertEqual(evaluate(doc, frame)[0]["status"], "excluded")

    def test_missing_source_is_pending(self):
        self.assertEqual(audit_match(csv_events(fixture()), META, None, CONFIG)[0]["status"], "pending_review")

    def test_no_lineup_information_is_not_a_stage_one_exclusion(self):
        self.assertNotIn("players", fixture()["info"])
        self.assertEqual(evaluate(fixture())[0]["status"], "included")

    def test_contradictory_info_overs_requires_target_and_progression(self):
        doc = fixture()
        doc["info"]["overs"] = 7
        decision = evaluate(doc)[0]
        self.assertEqual(decision["status"], "included")
        self.assertIn("info_overs_conflict", decision["warnings"])
        doc["innings"][0]["overs"] = doc["innings"][0]["overs"][:7]
        self.assertEqual(evaluate(doc)[0]["status"], "pending_review")

    def test_metadata_conflict_is_pending(self):
        meta = {**META, "match_date": "2021-01-01"}
        self.assertEqual(evaluate(fixture(), meta=meta)[0]["status"], "pending_review")

    def test_unexplained_target_is_pending_not_ordinary_penalty(self):
        doc = fixture()
        doc["innings"][1]["target"]["runs"] = 10
        self.assertIn("unexplained_target_difference", evaluate(doc)[0]["pending_reasons"])

    def test_raw_frame_is_unchanged(self):
        doc = fixture()
        frame = csv_events(doc)
        original = frame.copy(deep=True)
        evaluate(doc, frame)
        pd.testing.assert_frame_equal(frame, original)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="t20_stage01_test_")
        self.root = Path(self.temp.name)
        for directory in ("src", "config", "data/raw/cricsheet"):
            (self.root / directory).mkdir(parents=True)
        shutil.copyfile(ROOT / "config/experiment.yaml", self.root / "config/experiment.yaml")
        shutil.copyfile(ROOT / "src/data_preparation.py", self.root / "src/data_preparation.py")
        pd.DataFrame([{"match_id": 1, "date": "2020-01-01", "gender": "male", "game": "Australia vs India"}]).to_csv(self.root / CONFIG["paths"]["matches"], index=False)
        csv_events(fixture()).drop(columns="source_row").to_csv(self.root / CONFIG["paths"]["deliveries"], index=False)
        for gender in ("female", "male"):
            with ZipFile(self.root / CONFIG["paths"][gender + "_json"], "w") as archive:
                if gender == "male": archive.writestr("1.json", json.dumps(fixture()))

    def tearDown(self):
        self.temp.cleanup()

    def test_audit_then_export_with_verified_fingerprints(self):
        inputs = {p: sha256(self.root / p) for p in CONFIG["paths"].values()}
        summary = run(self.root)
        self.assertEqual(summary["match_counts"]["included"], 1)
        self.assertFalse((self.root / "data/processed/matches").exists())
        verified_audit(self.root)
        exported = run(self.root, export_clean=True)
        self.assertTrue(exported["clean_export_written"])
        self.assertEqual(len(pd.read_csv(self.root / "data/processed/matches/clean_deliveries.csv.gz")), 121)
        innings = pd.read_csv(self.root / "data/processed/matches/innings_metadata.csv")
        self.assertEqual(len(innings), 2)
        self.assertEqual(innings.loc[innings.match_innings.eq(2), "official_target"].iloc[0], 1)
        self.assertEqual(inputs, {p: sha256(self.root / p) for p in CONFIG["paths"].values()})

    def test_export_requires_audit(self):
        with self.assertRaises(FileNotFoundError): run(self.root, export_clean=True)

    def test_stale_audit_is_rejected(self):
        run(self.root)
        with (self.root / "src/data_preparation.py").open("a") as stream: stream.write("\n# changed\n")
        with self.assertRaisesRegex(ValueError, "stale"): verified_audit(self.root)

    def test_modified_report_is_rejected(self):
        run(self.root)
        with (self.root / "outputs/01_data_audit_and_cleaning/included_matches.csv").open("a") as stream: stream.write("\n")
        with self.assertRaisesRegex(ValueError, "report changed"): verified_audit(self.root)

    def test_path_escape_is_rejected(self):
        with self.assertRaises(ValueError): rooted(self.root, "../outside.csv")


if __name__ == "__main__":
    unittest.main()
