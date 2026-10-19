"""Stage 01: source-verified match audit; clean export is a separate opt-in.

No training, lineup exclusion, learned imputation or prediction features here.
JSON is corroborating evidence, not permission to overwrite unexplained CSV
differences. Raw rows stay immutable. Original values and approved changes are
preserved in the clean event table and in a field-level correction ledger.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd

STAGE = "01_data_audit_and_cleaning"
EXTRAS = ("wides", "noballs", "byes", "legbyes", "penalty")
TEMPORARY_RETIREMENTS = {"retired hurt", "retired not out"}
RETIREMENTS = TEMPORARY_RETIREMENTS | {"retired out"}
DISMISSALS = {"bowled", "caught", "caught and bowled", "lbw", "stumped",
              "hit wicket", "run out", "obstructing the field", "hit the ball twice",
              "timed out", "retired out"}
MATCH_REQUIRED = {"match_id", "date", "gender", "game"}
EVENT_REQUIRED = {"match_id", "batting_team", "bowling_team", "over", "ball",
                  "striker", "non_striker", "bowler", "runs_batter", "runs_extras",
                  "runs_total", "innings_score", "innings_wickets", "is_wicket",
                  "player_out", "wicket_kind", "match_winner", "winning_outcome",
                  "toss_winner", "toss_decision", *EXTRAS}
CORRECTION_COLUMNS = ["match_id", "source_row", "field", "original_value",
                      "canonical_value", "reason", "evidence"]
FIXED_EXCLUSIONS = {"tie", "no_result", "awarded", "adjusted_method", "reduced_overs",
                    "super_over", "declared_or_forfeited"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rooted(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Path must be inside project: {relative}")
    return path


def read_config(root):
    # JSON is a YAML subset. No implicit booleans or optional YAML dependency.
    config = json.loads(rooted(root, "config/experiment.yaml").read_text(encoding="utf-8"))
    if (config["stage"] != STAGE or config["scheduled_overs"] != 20
            or config["balls_per_over"] != 6 or config["match_type"] != "T20"
            or set(config["scope_exclusions"]) != FIXED_EXCLUSIONS
            or config["unresolved_policy"] != "pending_review; never silently repair or truncate"):
        raise ValueError("Unsupported cleaning policy; update implementation and tests explicitly.")
    if len(config["team_scope"]) != 11 or len(set(config["team_scope"])) != 11:
        raise ValueError("Expected the explicitly agreed eleven-team population.")
    return config


def fingerprints(root, config):
    paths = list(config["paths"].values()) + ["config/experiment.yaml", "src/data_preparation.py"]
    return {name: sha256(rooted(root, name)) for name in paths}


def text(value):
    return "" if pd.isna(value) else str(value)


def parse_boolean(value):
    if pd.isna(value):
        return None
    token = str(value).strip().lower()
    if token in {"true", "1", "1.0"}:
        return True
    if token in {"false", "0", "0.0"}:
        return False
    return None


def nonnegative_integer(value):
    return (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value) and value >= 0 and float(value).is_integer())


def miscount_ball_count(value):
    """Accept source integer strings without changing the preserved metadata."""
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        value = int(value)
    return int(value) if nonnegative_integer(value) and value > 0 else None


def require_columns(frame, required, name):
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")


def source_events(doc):
    for innings_no, innings in enumerate(doc["innings"], 1):
        for over in innings["overs"]:
            for delivery_no, delivery in enumerate(over["deliveries"], 1):
                yield innings_no, innings["team"], over["over"], delivery_no, delivery


def metadata_checks(doc, meta, local, config):
    """Separate research exclusions from unresolved identity/data problems."""
    excluded, pending, warnings = set(), set(), set()
    info = doc["info"]
    innings = doc["innings"]
    teams = info.get("teams", [])
    csv_teams = set(local.batting_team.dropna()) | set(local.bowling_team.dropna())
    if len(teams) != 2 or len(set(teams)) != 2 or csv_teams != set(teams):
        pending.add("team_identity_mismatch")
    if any(team not in config["team_scope"] for team in teams):
        excluded.add("outside_team_scope")
    if info.get("gender") not in config["genders"]:
        excluded.add("outside_gender_scope")
    if info.get("gender") != meta.get("gender"):
        pending.add("gender_mismatch")
    if not info.get("dates") or str(info["dates"][0]) != meta.get("match_date"):
        pending.add("date_missing_or_mismatch")
    if info.get("match_type") != config["match_type"]:
        excluded.add("outside_t20_scope")
    if info.get("balls_per_over") != 6:
        excluded.add("non_six_ball_over_format")
    outcome = info.get("outcome", {})
    csv_outcomes = " ".join(local.winning_outcome.dropna().astype(str).unique()).lower()
    result = str(outcome.get("result", "")).lower()
    if result in {"tie", "tied"} or re.search(r"\btie(?:d)?\b", csv_outcomes):
        excluded.add("tie")
    if result == "no result" or "no result" in csv_outcomes:
        excluded.add("no_result")
    if "award" in csv_outcomes or "award" in str(outcome).lower():
        excluded.add("awarded")
    if outcome.get("method") or re.search(r"dls|d/l|duckworth", csv_outcomes):
        excluded.add("adjusted_method")
    if (any(i.get("super_over") for i in innings) or outcome.get("eliminator")
            or outcome.get("bowl_out")):
        excluded.add("super_over")
    if any(i.get("declared") or i.get("forfeited") for i in innings):
        excluded.add("declared_or_forfeited")
    target_overs = innings[1].get("target", {}).get("overs") if len(innings) >= 2 else None
    if target_overs is not None and target_overs != 20:
        excluded.add("reduced_or_nonstandard_overs")
    if info.get("overs") != 20:
        if target_overs == 20:
            warnings.add("info_overs_conflict_checked_against_explicit_target_and_events")
        elif nonnegative_integer(info.get("overs")) and info["overs"] < 20:
            excluded.add("reduced_or_nonstandard_overs")
        else:
            pending.add("overs_limit_unresolved")
    if len(innings) != 2:
        pending.add("not_two_regular_innings")
    elif set(i["team"] for i in innings) != set(teams):
        pending.add("innings_team_mismatch")
    winner = outcome.get("winner", "")
    csv_winners = set(local.match_winner.dropna())
    if winner not in teams or csv_winners != {winner} or local.match_winner.isna().any():
        pending.add("winner_missing_or_mismatch")
    toss = info.get("toss", {})
    if (toss.get("winner") not in teams or toss.get("decision") not in {"bat", "field"}
            or local.toss_winner.isna().any() or local.toss_decision.isna().any()
            or set(local.toss_winner.dropna()) != {toss.get("winner")}
            or set(local.toss_decision.dropna()) != {toss.get("decision")}):
        pending.add("toss_missing_or_mismatch")
    return excluded, pending, warnings


def audit_batting_exhaustion(info, innings):
    """Retrospective terminal evidence only, never a pre-delivery feature.

    Absence has no event timestamp. Do not infer its live availability time from
    this check. Match replacements require a separate eligibility reconstruction.
    A missing roster does not exclude otherwise complete standard innings.
    """
    players = info.get("players", {}).get(innings["team"], [])
    absent_list = innings.get("absent_hurt", [])
    evidence = {"audit_no_batting_pair": False, "audit_roster_size": len(players),
                "audit_remaining_batters": "[]", "audit_unreturned_retirements": "[]",
                "audit_return_events": "[]", "audit_exhaustion_event": None,
                "audit_completion_issues": ""}
    issues = set()
    if (len(players) != 11 or len(set(players)) != 11
            or not all(isinstance(p, str) and p for p in players)):
        evidence["audit_completion_issues"] = "completion_roster_missing_or_invalid"
        return evidence
    roster, absent = set(players), set(absent_list)
    if len(absent) != len(absent_list) or not absent <= roster:
        issues.add("completion_absence_invalid")
    dismissed, retired, seen = set(), set(), set()
    returns, exhausted = [], []
    sequence, final_exit = 0, False
    for over in innings["overs"]:
        for d in over["deliveries"]:
            sequence += 1
            actors = {d["batter"], d["non_striker"]}
            if len(actors) != 2 or not actors <= roster or actors & dismissed:
                issues.add("completion_actor_conflict")
            if d.get("replacements"):
                issues.add("completion_replacements_need_review")
            seen |= actors
            for player in sorted(actors & retired):
                returns.append({"event": sequence, "player": player})
            retired -= actors
            exits = d.get("wickets", [])
            final_exit = bool(exits)
            for w in exits:
                player, kind = w["player_out"], w["kind"]
                if player not in roster or player in dismissed or player in retired:
                    issues.add("completion_exit_conflict")
                if kind in TEMPORARY_RETIREMENTS:
                    retired.add(player)
                elif kind in DISMISSALS:
                    dismissed.add(player)
                else:
                    issues.add("completion_unknown_exit")
            if len(roster - (dismissed | absent | retired)) < 2:
                exhausted.append(sequence)
    if absent & seen or absent & (dismissed | retired) or dismissed & retired:
        issues.add("completion_availability_conflict")
    remaining = roster - (dismissed | absent | retired)
    if exhausted and exhausted != [sequence]:
        issues.add("completion_events_after_no_batting_pair")
    evidence.update({
        "audit_no_batting_pair": bool(not issues and final_exit and exhausted == [sequence]
                                      and len(remaining) == 1 and (absent or retired)),
        "audit_remaining_batters": json.dumps(sorted(remaining)),
        "audit_unreturned_retirements": json.dumps(sorted(retired)),
        "audit_return_events": json.dumps(returns),
        "audit_exhaustion_event": exhausted[0] if exhausted else None,
        "audit_completion_issues": ";".join(sorted(issues)),
    })
    return evidence


def decode_states(doc):
    """Audit complete innings without treating future metadata as model inputs.

    Only event facts and cumulative counters are returned. Full-innings totals,
    completion evidence and miscount lengths are separately marked audit-only.
    No remaining-ball or run-rate predictor is constructed in Stage 01.
    """
    rows, reports, pending = [], [], set()
    totals = []
    for innings in doc["innings"]:
        penalty = innings.get("penalty_runs", {})
        pre, post = penalty.get("pre", 0), penalty.get("post", 0)
        if not all(nonnegative_integer(v) for v in (pre, post)):
            pending.add("invalid_innings_penalty")
            return [], [], pending
        totals.append(sum(d["runs"]["total"] for o in innings["overs"] for d in o["deliveries"]) + pre + post)
    target = doc["innings"][1].get("target", {}).get("runs", totals[0] + 1)
    if not nonnegative_integer(target) or target != totals[0] + 1:
        pending.add("unexplained_target_difference")
    for innings_no, innings in enumerate(doc["innings"], 1):
        overs = innings["overs"]
        if [o["over"] for o in overs] != list(range(len(overs))) or len(overs) > 20:
            pending.add("invalid_over_sequence")
        pre = innings.get("penalty_runs", {}).get("pre", 0)
        post = innings.get("penalty_runs", {}).get("post", 0)
        score, wickets, legal_balls, event_sequence = pre, 0, 0, 0
        miscounts = innings.get("miscounted_overs", {})
        for key, detail in miscounts.items():
            if str(key) not in {str(o["over"]) for o in overs} or miscount_ball_count(detail.get("balls")) is None:
                pending.add("invalid_miscount_metadata")
        if "invalid_miscount_metadata" in pending:
            return [], [], pending
        last_over_finished = False
        all_out_seen = False
        for over_index, over in enumerate(overs):
            expected = miscount_ball_count(miscounts.get(str(over["over"]), {}).get("balls", 6))
            legal_in_over = 0
            for delivery_no, d in enumerate(over["deliveries"], 1):
                event_sequence += 1
                extra = d.get("extras", {})
                counts = [d["runs"]["batter"], d["runs"]["extras"], d["runs"]["total"], *extra.values()]
                if not all(nonnegative_integer(v) for v in counts):
                    pending.add("invalid_source_run_value")
                if d["runs"]["total"] != d["runs"]["batter"] + d["runs"]["extras"] or d["runs"]["extras"] != sum(extra.values()):
                    pending.add("source_run_accounting_mismatch")
                if set(extra) - set(EXTRAS):
                    pending.add("unsupported_extra_kind")
                if wickets >= 10:
                    pending.add("events_after_all_out")
                if innings_no == 2 and score >= target:
                    pending.add("events_after_successful_chase")
                ws = d.get("wickets", [])
                if any(w["kind"] not in DISMISSALS | TEMPORARY_RETIREMENTS for w in ws):
                    pending.add("unsupported_dismissal_kind")
                if len(ws) > 1:
                    pending.add("multiple_wickets_need_review")
                dismissal_count = sum(w["kind"] not in TEMPORARY_RETIREMENTS for w in ws)
                legal = not (extra.get("wides", 0) or extra.get("noballs", 0))
                legal_balls += int(legal)
                legal_in_over += int(legal)
                score += d["runs"]["total"]
                wickets += dismissal_count
                all_out_seen |= wickets == 10
                if wickets > 10:
                    pending.add("over_ten_wickets")
                rows.append({
                    "match_innings": innings_no, "event_sequence": event_sequence,
                    "source_over": over["over"], "source_delivery_in_over": delivery_no,
                    "batting_team": innings["team"], "striker": d["batter"],
                    "non_striker": d["non_striker"], "bowler": d["bowler"],
                    "runs_batter": d["runs"]["batter"], "runs_extras": d["runs"]["extras"],
                    "runs_total": d["runs"]["total"], **{k: extra.get(k, 0) for k in EXTRAS},
                    "player_out": ws[0]["player_out"] if ws else "",
                    "wicket_kind": ws[0]["kind"] if ws else "",
                    "is_wicket": bool(dismissal_count), "legal_delivery": int(legal),
                    "innings_score": score, "innings_wickets": wickets,
                    "cumulative_legal_deliveries": legal_balls,
                })
            if legal_in_over > expected or (over_index < len(overs) - 1 and legal_in_over != expected):
                pending.add("unexplained_over_length")
            if str(over["over"]) in miscounts and legal_in_over != expected:
                pending.add("miscount_metadata_mismatch")
            last_over_finished = over["over"] == 19 and legal_in_over == expected
        completed = last_over_finished or all_out_seen or (innings_no == 2 and totals[1] >= target)
        completion_basis = ("target_reached" if innings_no == 2 and totals[1] >= target
                            else "ten_dismissals" if all_out_seen
                            else "scheduled_overs_complete" if last_over_finished else "unresolved")
        exhaustion = {"audit_no_batting_pair": False, "audit_roster_size": None,
                      "audit_remaining_batters": "[]", "audit_unreturned_retirements": "[]",
                      "audit_return_events": "[]", "audit_exhaustion_event": None,
                      "audit_completion_issues": ""}
        if not completed:
            exhaustion = audit_batting_exhaustion(doc["info"], innings)
            completed = exhaustion["audit_no_batting_pair"]
            if completed:
                completion_basis = "no_available_batting_pair"
            else:
                pending.update(filter(None, exhaustion["audit_completion_issues"].split(";")))
        if not completed:
            pending.add("innings_completion_unresolved")
            if innings.get("absent_hurt") or any(
                w["kind"] in TEMPORARY_RETIREMENTS
                for o in overs for d in o["deliveries"] for w in d.get("wickets", [])
            ):
                pending.add("injury_related_completion_needs_review")
        reports.append({"match_innings": innings_no, "batting_team": innings["team"],
                        "events": event_sequence, "actual_legal_balls": legal_balls,
                        "final_wickets": wickets, "penalty_runs_pre": pre,
                        "penalty_runs_post": post, "official_total": totals[innings_no - 1],
                        "official_target": target if innings_no == 2 else None,
                        "audit_complete": completed,
                        "audit_completion_basis": completion_basis, **exhaustion,
                        "miscounted_overs": json.dumps(miscounts, sort_keys=True),
                        "absent_hurt": json.dumps(innings.get("absent_hurt", []))})
    expected_winner = doc["innings"][0]["team"] if totals[0] > totals[1] else doc["innings"][1]["team"]
    if totals[0] == totals[1] or doc["info"]["outcome"].get("winner") != expected_winner:
        pending.add("winner_inconsistent_with_totals")
    return rows, reports, pending


def compare_events(local, rows, innings_reports, match_id):
    """Allow only independently explained retirements/penalties/coordinates."""
    problems, corrections = set(), []
    if len(local) != len(rows):
        return None, {"event_count_mismatch"}, []
    if not rows:
        return None, {"empty_event_stream"}, []
    source = pd.DataFrame(rows)
    local = local.sort_values("source_row", kind="stable").reset_index(drop=True)
    comparable = ["batting_team", "striker", "non_striker", "bowler", "runs_batter",
                  "runs_extras", "runs_total", *EXTRAS]
    for column in comparable:
        if local[column].isna().any() or not local[column].eq(source[column]).all():
            problems.add("source_event_mismatch_" + column)
    teams = list(dict.fromkeys(source.batting_team))
    if len(teams) == 2:
        opposition = source.batting_team.map({teams[0]: teams[1], teams[1]: teams[0]})
        if not local.bowling_team.eq(opposition).all():
            problems.add("source_event_mismatch_bowling_team")
    flags = local.is_wicket.map(parse_boolean)
    if flags.isna().any():
        problems.add("invalid_or_missing_wicket_flag")
    for column in ["over", "ball", "innings_score", "innings_wickets", "runs_batter", "runs_extras", "runs_total", *EXTRAS]:
        if not local[column].map(nonnegative_integer).all():
            problems.add("invalid_or_missing_numeric_" + column)
    if problems:
        return None, problems, corrections

    def record(index, column, replacement, reason):
        old = local.at[index, column]
        if text(old) == text(replacement) or (column == "is_wicket" and parse_boolean(old) == replacement):
            return
        corrections.append({"match_id": match_id, "source_row": int(local.at[index, "source_row"]),
                            "field": column, "original_value": text(old), "canonical_value": text(replacement),
                            "reason": reason, "evidence": f"{match_id}.json:innings={source.at[index, 'match_innings']};over={source.at[index, 'source_over']};event={source.at[index, 'source_delivery_in_over']}"})

    for index, row in source.iterrows():
        old_out, old_kind = text(local.at[index, "player_out"]), text(local.at[index, "wicket_kind"])
        source_retirement = row.wicket_kind in RETIREMENTS
        omitted_retirement = source_retirement and old_out == "" and old_kind == ""
        if (old_out, old_kind) != (row.player_out, row.wicket_kind) and not omitted_retirement:
            problems.add("source_dismissal_mismatch")
        if flags.iloc[index] != row.is_wicket and not source_retirement:
            problems.add("wicket_flag_mismatch")
        for column in ("player_out", "wicket_kind", "is_wicket"):
            record(index, column, row[column], "explicit_source_retirement")
        # Display coordinates do not establish event order. Match all event facts first.
        record(index, "over", row.source_over, "source_verified_over_coordinate")
        record(index, "ball", row.source_delivery_in_over, "source_verified_event_coordinate")
    csv_cumulative_runs = local.runs_total.groupby(source.match_innings).cumsum()
    csv_cumulative_wickets = flags.astype(int).groupby(source.match_innings).cumsum()
    for index, row in source.iterrows():
        # The known CSV omissions concern innings penalties and retirement events.
        if local.at[index, "innings_score"] not in {csv_cumulative_runs.iloc[index], row.innings_score}:
            problems.add("unexplained_csv_score")
        if local.at[index, "innings_wickets"] not in {csv_cumulative_wickets.iloc[index], row.innings_wickets}:
            problems.add("unexplained_csv_wickets")
        record(index, "innings_score", row.innings_score, "source_penalty_aware_cumulative_score")
        record(index, "innings_wickets", row.innings_wickets, "source_retirement_aware_cumulative_wickets")
    if problems:
        return None, problems, []  # Proposed changes to unresolved matches are not approved corrections.
    clean = local.copy()
    for column in ("over", "ball", "innings_score", "innings_wickets", "player_out", "wicket_kind", "is_wicket"):
        clean["original_" + column] = clean[column]
    for column in source:
        clean[column] = source[column].to_numpy()
    clean["over"] = clean.source_over
    clean["ball"] = clean.source_delivery_in_over
    clean["audit_source_verified"] = True
    return clean, problems, corrections


def audit_match(local, meta, doc, config):
    mid = int(local.match_id.iloc[0])
    excluded, pending, warnings = set(), set(), set()
    clean, reports, corrections = None, [], []
    if meta is None:
        pending.add("match_metadata_missing")
        meta = {}
    if meta.get("metadata_issue"):
        pending.add(meta["metadata_issue"])
    if doc is None:
        pending.add("source_json_missing")
        csv_teams = set(local.batting_team.dropna()) | set(local.bowling_team.dropna())
        if any(t not in config["team_scope"] for t in csv_teams):
            excluded.add("outside_team_scope")
    else:
        try:
            ex, pe, wa = metadata_checks(doc, meta, local, config)
            excluded |= ex
            pending |= pe
            warnings |= wa
            if not excluded and not pending:
                rows, reports, pe = decode_states(doc)
                pending |= pe
                if not pending:
                    clean, pe, corrections = compare_events(local, rows, reports, mid)
                    pending |= pe
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            pending.add("source_structure_unresolved")
            warnings.add(f"{type(exc).__name__}: {str(exc)[:180]}")
    status = "excluded" if excluded else "pending_review" if pending else "included"
    if status != "included":
        clean, corrections = None, []
    elif clean is None:
        raise AssertionError("An included match must have verified canonical events.")
    decision = {"match_id": mid, "match_date": meta.get("match_date", ""),
                "gender": meta.get("gender", ""), "csv_events": len(local),
                "status": status, "exclusion_reasons": ";".join(sorted(excluded)),
                "pending_reasons": ";".join(sorted(pending)),
                "warnings": ";".join(sorted(warnings)),
                "event_audit_performed": bool(reports),
                "source_json_available": doc is not None,
                "display_coordinate_repeats": int(local.duplicated(["batting_team", "over", "ball"]).sum()),
                "approved_corrections": len(corrections)}
    return decision, clean, [{"match_id": mid, **r} for r in reports], corrections


def load_inputs(root, config):
    matches = pd.read_csv(rooted(root, config["paths"]["matches"]))
    events = pd.read_csv(rooted(root, config["paths"]["deliveries"]), low_memory=False)
    require_columns(matches, MATCH_REQUIRED, "matches")
    require_columns(events, EVENT_REQUIRED, "deliveries")
    for frame in (matches, events):
        if not frame.match_id.map(nonnegative_integer).all():
            raise ValueError("Missing/invalid match IDs prevent reliable linkage. Sources unchanged.")
        frame["match_id"] = frame.match_id.astype("int64")
    if "source_row" in events:
        raise ValueError("Raw events unexpectedly contain a reserved source_row column.")
    events.insert(0, "source_row", np.arange(len(events), dtype=np.int64))
    missingness = pd.concat([
        pd.DataFrame({"source": name, "column": frame.columns,
                      "missing_count": frame.isna().sum().to_numpy(), "rows": len(frame)})
        for name, frame in (("matches", matches), ("deliveries", events))], ignore_index=True)
    metadata = {}
    for mid, rows in matches.groupby("match_id", sort=False):
        first = rows.iloc[0]
        date = pd.to_datetime(first["date"], errors="coerce")
        metadata[int(mid)] = {"match_date": date.strftime("%Y-%m-%d") if pd.notna(date) else "",
                              "gender": text(first.gender), "game": text(first.game),
                              "metadata_issue": "duplicate_match_metadata" if len(rows) != 1 else
                              "invalid_match_date" if pd.isna(date) else ""}
    return matches, events, metadata, missingness


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def verified_audit(root):
    """Reject missing, unfinished, tampered or stale report snapshots."""
    root = Path(root).resolve()
    path = root / "outputs" / STAGE / "audit_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError("Audit is not complete; run Stage 01 audit first.")
    if summary["input_sha256"] != fingerprints(root, read_config(root)):
        raise ValueError("Audit is stale: raw inputs, config or implementation changed.")
    for relative, expected in summary["report_sha256"].items():
        if sha256(rooted(root, relative)) != expected:
            raise ValueError(f"Audit report changed: {relative}")
    return summary


def run(root, export_clean=False):
    root = Path(root).resolve()
    config = read_config(root)
    inputs = fingerprints(root, config)
    # Export requires a previous successful audit of the exact inputs and code.
    approved = verified_audit(root) if export_clean else None
    out = rooted(root, "outputs/" + STAGE)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "audit_summary.json"
    write_json(summary_path, {"status": "running", "mode": "export" if export_clean else "audit_only"})
    try:
        matches, events, metadata, missingness = load_inputs(root, config)
        decisions, canonical, innings_reports, corrections = [], [], [], []
        with ExitStack() as stack:
            archives = {g: stack.enter_context(ZipFile(rooted(root, config["paths"][g + "_json"]))) for g in config["genders"]}
            archive_names = {g: set(a.namelist()) for g, a in archives.items()}
            for number, (mid, local) in enumerate(events.groupby("match_id", sort=True), 1):
                meta = metadata.get(int(mid))
                gender = meta.get("gender") if meta else None
                filename = f"{mid}.json"
                doc = None
                if gender in archives and filename in archive_names[gender]:
                    try:
                        doc = json.loads(archives[gender].read(filename))
                    except (ValueError, KeyError):
                        pass
                local = local.copy()
                local["match_date"] = meta.get("match_date", "") if meta else ""
                local["gender"] = gender
                decision, clean, audits, edits = audit_match(local, meta, doc, config)
                decisions.append(decision)
                innings_reports.extend(audits)
                corrections.extend(edits)
                if clean is not None:
                    canonical.append(clean)
                if number % 500 == 0:
                    print(f"Audited {number} matches", flush=True)
        for mid in sorted(set(metadata) - set(events.match_id)):
            meta = metadata[mid]
            decisions.append({"match_id": mid, "match_date": meta["match_date"], "gender": meta["gender"],
                              "csv_events": 0, "status": "pending_review", "exclusion_reasons": "",
                              "pending_reasons": "no_delivery_records", "warnings": "",
                              "event_audit_performed": False, "source_json_available": False,
                              "display_coordinate_repeats": 0, "approved_corrections": 0})
        decision_frame = pd.DataFrame(decisions).sort_values("match_id")
        clean = pd.concat(canonical, ignore_index=True) if canonical else pd.DataFrame()
        included = decision_frame.loc[decision_frame.status.eq("included")]
        if clean.empty or clean.source_row.duplicated().any() or len(clean) != int(included.csv_events.sum()):
            raise ValueError("No verified data or a coverage/identity invariant failed.")
        clean = clean.sort_values(["match_id", "match_innings", "source_row"], kind="stable")
        if clean.groupby(["match_id", "match_innings"]).source_row.diff().dropna().le(0).any():
            raise ValueError("Canonical events are not in source order.")
        # Primary and supplemental counts stay separate; reasons can overlap.
        reason_counts = Counter()
        for row in decisions:
            field = "exclusion_reasons" if row["status"] == "excluded" else "pending_reasons"
            for reason in filter(None, row[field].split(";")):
                reason_counts[(row["status"], reason)] += 1
        reasons = pd.DataFrame([{"status": s, "reason": r, "matches": n} for (s, r), n in sorted(reason_counts.items())],
                               columns=["status", "reason", "matches"])
        tables = {"match_decisions.csv": decision_frame,
                  "included_matches.csv": included,
                  "excluded_matches.csv": decision_frame.loc[decision_frame.status.eq("excluded")],
                  "pending_matches.csv": decision_frame.loc[decision_frame.status.eq("pending_review")],
                  "reason_counts.csv": reasons,
                  "missing_values.csv": missingness,
                  "innings_audit.csv": pd.DataFrame(innings_reports),
                  "corrections.csv": pd.DataFrame(corrections, columns=CORRECTION_COLUMNS),
                  "clean_event_preview.csv": clean.head(20)}
        if inputs != fingerprints(root, config):
            raise ValueError("Inputs changed during audit; no clean dataset published.")
        for name, table in tables.items():
            table.to_csv(out / name, index=False)
        audit_id = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        summary = {"status": "complete", "mode": "export" if export_clean else "audit_only",
                   "audit_id": audit_id, "created_utc": datetime.now(timezone.utc).isoformat(),
                   "input_sha256": inputs, "python_version": sys.version.split()[0],
                   "pandas_version": pd.__version__, "numpy_version": np.__version__,
                   "source_match_rows": len(matches), "source_delivery_rows": len(events),
                   "audited_matches": len(decision_frame),
                   "match_counts": {s: int(decision_frame.status.eq(s).sum()) for s in ("included", "excluded", "pending_review")},
                   "included_events": len(clean),
                   "included_by_gender": included.groupby("gender").agg(matches=("match_id", "size"), events=("csv_events", "sum")).reset_index().to_dict("records"),
                   "approved_field_corrections": len(corrections),
                   "report_sha256": {str((out / name).relative_to(root)).replace("\\", "/"): sha256(out / name) for name in tables},
                   "clean_export_written": False,
                   "notes": ["Out-of-scope matches receive metadata screening; event-level checks focus on otherwise eligible matches.",
                             "Excluded matches may also have unresolved issues. Counts of reasons overlap.",
                             "No automatic row deletion or missing-value imputation. Missing core values require review.",
                             "Roster coverage and eligibility are investigated in Stage 04, not used to filter this cohort.",
                             "Audit totals, final outcomes and miscount lengths must never become live prediction features.",
                             "No train/validation/test split or learned feature is created by Stage 01."]}
        if approved and (approved["audit_id"] != audit_id or approved["report_sha256"] != summary["report_sha256"]):
            raise ValueError("Export differs from the reviewed audit. Review a new audit before export.")
        if export_clean:
            target = rooted(root, "data/processed/matches")
            target.mkdir(parents=True, exist_ok=True)
            write_json(target / "manifest.json", {"status": "writing", "audit_id": audit_id})
            clean.to_csv(target / "clean_deliveries.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
            included.to_csv(target / "clean_matches.csv", index=False)
            # This preserves target/penalty evidence for Stage 02, together with
            # future audit facts. It is NOT a model feature matrix.
            innings_metadata = tables["innings_audit.csv"]
            innings_metadata = innings_metadata.loc[innings_metadata.match_id.isin(included.match_id)]
            innings_metadata.to_csv(target / "innings_metadata.csv", index=False)
            summary["clean_export_written"] = True
            outputs = {str(p.relative_to(root)).replace("\\", "/"): sha256(p) for p in
                       [target / "clean_deliveries.csv.gz", target / "clean_matches.csv", target / "innings_metadata.csv"]}
            write_json(target / "manifest.json", {"status": "complete", "audit_id": audit_id,
                       "input_sha256": inputs, "output_sha256": outputs,
                       "matches": len(included), "events": len(clean),
                       "metadata_usage": "Use pre-innings penalties at innings start and official target only in innings 2; final totals, completion flags, absent_hurt and full miscounts are audit-only",
                       "state_timing": "canonical event outcomes; Stage 02 constructs pre-delivery predictors"})
        write_json(summary_path, summary)
        print(json.dumps({k: summary[k] for k in ("mode", "match_counts", "included_events", "clean_export_written")}, indent=2), flush=True)
        return summary
    except Exception as exc:
        write_json(summary_path, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--export", action="store_true", help="Export after reviewing a successful audit of the same inputs and code")
    args = parser.parse_args()
    run(args.root, export_clean=args.export)
