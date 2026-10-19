"""Stage 02: deterministic pre-delivery state and prior-date team background.

No splitting, fitting, feature selection, player quality or NWC calculation.
Only the explicit feature contract may be used to construct model matrices.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from data_preparation import rooted, sha256, verified_audit, write_json

STAGE = "02_game_state_features"
DATA_DIR = "data/processed/game_state"
REPORT_DIR = "outputs/" + STAGE
TARGET = "first_batting_team_win"
KEYS = ["match_id", "match_innings"]
BASE_NUMERIC = [
    "match_innings", "pre_score", "wickets_remaining",
    "scheduled_balls_remaining_estimate", "current_run_rate", "no_legal_ball_yet",
    "target_runs", "runs_required", "required_run_rate", "run_rate_differential",
    "toss_first_team_won",
    "first_team_prior_win_rate", "second_team_prior_win_rate",
    "first_team_recent_win_rate", "second_team_recent_win_rate",
]
BASE_CATEGORICAL = ["innings_phase"]
IDENTITY_CANDIDATES = ["first_batting_team", "second_batting_team"]
RECENT_CANDIDATES = ["runs_prev_10_events", "wickets_prev_10_events"]
HISTORY_COLUMNS = ["prior_matches", "prior_win_rate", "recent_win_rate"]
ID_COLUMNS = ["match_id", "source_row", "event_sequence", "match_date", "gender",
              "first_batting_team", "second_batting_team", "batting_team", "bowling_team",
              "striker", "non_striker", "bowler"]


def read_config(root):
    config = json.loads((Path(root) / "config/game_state.json").read_text(encoding="utf-8"))
    expected = {"stage": STAGE, "prediction_timing": "pre_delivery", "target": TARGET,
                "recent_event_window": 10, "phase_over_boundaries": [6, 15],
                "scheduled_overs": 20, "balls_per_over": 6,
                "history_scope": "Stage 01 included matches, strictly earlier dates, within gender",
                "same_date_policy": "Read all snapshots before updating any result on that date",
                "recent_form_same_date_policy": "Equal weight within date; decay older history by the number of matches"}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Unsupported Stage 02 policy; update implementation and tests explicitly.")
    for key in ["team_form_half_life_matches", "team_rate_prior_wins", "team_rate_prior_losses"]:
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    return config


def feature_contract():
    return {
        "target": TARGET, "probability_perspective": "first-innings batting team in both innings",
        "prediction_timing": "immediately before each recorded event",
        "base_numeric": BASE_NUMERIC, "base_categorical": BASE_CATEGORICAL,
        "optional_team_identity": IDENTITY_CANDIDATES,
        "optional_recent_events": RECENT_CANDIDATES,
        "selection": "Stage 06 compares optional blocks using validation only; freeze the same background for every resource arm.",
        "first_innings_chase_policy": "Four chase columns are NA, not future targets. Separate innings models omit them; a unified model must use an innings-aware mapping fitted on training only.",
        "forbidden_automatic_selection": "Never select all numeric columns. Outcome, audit_*, identifiers, actors and history exposure counts are not base predictors.",
        "redundancy": "pre_wickets, legal_balls_elapsed and source_over are diagnostic alternatives. toss_elected_bat duplicates toss_first_team_won in the retained cohort and is diagnostic only. No strength difference is added alongside both team rates.",
        "gender": "Keep genders separate when fitting Stage 06 models.",
        "history_evaluation": "Rolling-origin information: completed earlier-date validation/test matches may update later history. No model fitting or selection on test results.",
        "nwc": "Stage 02 supplies pre-event states only. Stage 08 must explicitly construct post-event/terminal states; shifting predictions alone is not an attribution definition.",
    }


def prior_event_sum(frame, values, window=10):
    """Past recorded events within an innings, including wides/no-balls."""
    groups = [frame[k] for k in KEYS]
    shifted = values.groupby(groups, sort=False).shift(1)
    return shifted.groupby(groups, sort=False).transform(
        lambda s: s.rolling(window, min_periods=1).sum()).fillna(0)


def match_records(deliveries, matches, innings):
    if matches.match_id.duplicated().any() or innings.duplicated(KEYS).any():
        raise ValueError("Duplicate match or innings key")
    if set(deliveries.match_id) != set(matches.match_id) or set(innings.match_id) != set(matches.match_id):
        raise ValueError("Stage 01 table coverage differs")
    if not innings.groupby("match_id").match_innings.apply(lambda x: set(x) == {1, 2}).all():
        raise ValueError("Expected exactly two innings per match")
    constant = ["match_date", "gender", "match_winner", "toss_winner", "toss_decision"]
    if deliveries.groupby("match_id")[constant].nunique(dropna=False).ne(1).any().any():
        raise ValueError("Inconsistent match metadata")
    if deliveries[constant].isna().any().any():
        raise ValueError("Missing match metadata")
    records = deliveries.groupby("match_id", sort=False)[constant].first().reset_index()
    for no, name in [(1, "first_batting_team"), (2, "second_batting_team")]:
        lookup = innings.loc[innings.match_innings.eq(no), ["match_id", "batting_team"]]
        records = records.merge(lookup.rename(columns={"batting_team": name}), on="match_id", validate="one_to_one")
    if records.first_batting_team.eq(records.second_batting_team).any():
        raise ValueError("A match must have two different teams")
    for column in ["match_winner", "toss_winner"]:
        if not (records[column].eq(records.first_batting_team) | records[column].eq(records.second_batting_team)).all():
            raise ValueError(f"Invalid {column}")
    if not records.toss_decision.isin(["bat", "field"]).all():
        raise ValueError("Unknown toss decision")
    expected_first = np.where(records.toss_decision.eq("bat"), records.toss_winner,
                              np.where(records.toss_winner.eq(records.first_batting_team),
                                       records.second_batting_team, records.first_batting_team))
    if not (expected_first == records.first_batting_team).all():
        raise ValueError("Toss decision contradicts innings order")
    records["match_date"] = pd.to_datetime(records.match_date, errors="raise").dt.normalize()
    verified = matches.set_index("match_id").loc[records.match_id]
    if not np.array_equal(pd.to_datetime(verified.match_date).to_numpy(), records.match_date.to_numpy()):
        raise ValueError("Match dates differ from approved cohort")
    if not np.array_equal(verified.gender.to_numpy(), records.gender.to_numpy()):
        raise ValueError("Genders differ from approved cohort")
    if not records.gender.isin(["female", "male"]).all():
        raise ValueError("Unsupported gender")
    records[TARGET] = records.match_winner.eq(records.first_batting_team).astype("int8")
    return records.sort_values(["match_date", "match_id"], kind="stable").reset_index(drop=True)


def team_history(records, config):
    """Read entire date group, then update. No ordering assumptions within a day.

    Weighted wins/counts decay by 2**(-matches / half_life). For same-date
    batches use equal weights whose total equals a sequential geometric sum.
    Both long-term and recent rates use the configured Beta pseudo-counts.
    """
    decay = 2 ** (-1 / config["team_form_half_life_matches"])
    prior_w = config["team_rate_prior_wins"]
    prior_n = prior_w + config["team_rate_prior_losses"]
    state = defaultdict(lambda: [0, 0, 0.0, 0.0])  # wins, matches, weighted wins, weighted matches
    result = []
    for _, day in records.groupby("match_date", sort=True):
        updates = defaultdict(list)
        for row in day.itertuples(index=False):
            out = {"match_id": row.match_id}
            for role, team in [("first_team", row.first_batting_team), ("second_team", row.second_batting_team)]:
                key = (row.gender, team)
                wins, count, recent_w, recent_n = state[key]
                out[role + "_prior_matches"] = count
                out[role + "_prior_win_rate"] = (wins + prior_w) / (count + prior_n)
                out[role + "_recent_win_rate"] = (recent_w + prior_w) / (recent_n + prior_n)
                updates[key].append(int(row.match_winner == team))
            result.append(out)
        for key, outcomes in updates.items():
            wins, count, recent_w, recent_n = state[key]
            size = len(outcomes)
            factor = decay ** size
            weight = (1 - factor) / (1 - decay)
            state[key] = [wins + sum(outcomes), count + size,
                          recent_w * factor + weight * np.mean(outcomes), recent_n * factor + weight]
    return records.merge(pd.DataFrame(result), on="match_id", validate="one_to_one")


def build_features(deliveries, matches, innings, config):
    """Return one pre-event row per verified delivery and one background row per match."""
    d = deliveries.sort_values(KEYS + ["source_row"], kind="stable").reset_index(drop=True).copy()
    if d.source_row.duplicated().any() or d.duplicated(KEYS + ["event_sequence"]).any():
        raise ValueError("Duplicate event identity")
    grouped = d.groupby(KEYS, sort=False)
    if not d.event_sequence.eq(grouped.cumcount() + 1).all():
        raise ValueError("Source order contradicts event sequence")
    if not d.source_over.between(0, 19).all() or grouped.source_over.diff().fillna(0).lt(0).any():
        raise ValueError("Invalid observed over progression")
    if not d.legal_delivery.isin([0, 1]).all():
        raise ValueError("Invalid legal-delivery flag")
    meta = innings.set_index(KEYS)
    event_keys = pd.MultiIndex.from_frame(d[KEYS])
    pre_penalties = pd.Series(meta.penalty_runs_pre.reindex(event_keys).to_numpy(), index=d.index)
    if pre_penalties.isna().any():
        raise ValueError("Missing innings metadata")
    if not np.array_equal(meta.batting_team.reindex(event_keys).to_numpy(), d.batting_team.to_numpy()):
        raise ValueError("Innings team mismatch")
    pre_score = grouped.innings_score.shift(1).fillna(pre_penalties)
    pre_wickets = grouped.innings_wickets.shift(1).fillna(0)
    wicket_count = d.innings_wickets - pre_wickets
    if not wicket_count.isin([0, 1]).all() or not (d.innings_score - pre_score).eq(d.runs_total).all():
        raise ValueError("Cumulative scores or wickets do not reconcile")
    if not d.cumulative_legal_deliveries.eq(grouped.legal_delivery.cumsum()).all():
        raise ValueError("Cumulative legal count does not reconcile")
    pre_legal = grouped.legal_delivery.cumsum() - d.legal_delivery
    over_legal = d.groupby(KEYS + ["source_over"], sort=False).legal_delivery.cumsum() - d.legal_delivery
    records = team_history(match_records(d, matches, innings), config)
    out = d[["match_id", "source_row", "match_innings", "event_sequence", "batting_team", "bowling_team",
             "striker", "non_striker", "bowler"]].merge(
        records.drop(columns="match_winner"), on="match_id", validate="many_to_one", sort=False)
    expected_batting = np.where(out.match_innings.eq(1), out.first_batting_team, out.second_batting_team)
    expected_bowling = np.where(out.match_innings.eq(1), out.second_batting_team, out.first_batting_team)
    if not (out.batting_team.eq(expected_batting) & out.bowling_team.eq(expected_bowling)).all():
        raise ValueError("Batting/bowling orientation inconsistent")
    out["pre_score"] = pre_score.astype(int)
    out["pre_wickets"] = pre_wickets.astype(int)
    out["wickets_remaining"] = 10 - out.pre_wickets
    out["legal_balls_elapsed"] = pre_legal.astype(int)
    out["source_over"] = d.source_over.astype(int)
    # Nominal slots, NOT hindsight-adjusted final innings length. A current
    # over may already have six legal balls if its end was miscounted.
    slots = (19 - d.source_over) * 6 + (6 - over_legal).clip(lower=0)
    out["scheduled_balls_remaining_estimate"] = slots.astype(int)
    out["audit_pre_legal_in_over"] = over_legal.astype(int)
    out["audit_overrun_before_event"] = over_legal.ge(6)
    out["audit_rate_denominator_floored"] = slots.eq(0)
    out["no_legal_ball_yet"] = pre_legal.eq(0).astype("int8")
    out["current_run_rate"] = (6 * pre_score / pre_legal.clip(lower=1)).where(pre_legal.gt(0), 0.0)
    out["innings_phase"] = np.select([d.source_over.lt(6), d.source_over.lt(15)],
                                     ["powerplay", "middle"], default="death")
    chase = out.match_innings.eq(2)
    out["target_runs"] = pd.Series(meta.official_target.reindex(event_keys).to_numpy(), index=d.index).where(chase)
    if out.loc[chase, "target_runs"].isna().any():
        raise ValueError("Missing chase target")
    out["runs_required"] = (out.target_runs - out.pre_score).where(chase)
    if out.loc[chase, "runs_required"].le(0).any() or out.wickets_remaining.le(0).any():
        raise ValueError("Pre-event state already terminal")
    out["required_run_rate"] = (6 * out.runs_required / slots.clip(lower=1)).where(chase)
    out["run_rate_differential"] = (out.current_run_rate - out.required_run_rate).where(chase)
    out["toss_first_team_won"] = out.toss_winner.eq(out.first_batting_team).astype("int8")
    out["toss_elected_bat"] = out.toss_decision.eq("bat").astype("int8")
    out["runs_prev_10_events"] = prior_event_sum(d, d.runs_total).astype(int)
    out["wickets_prev_10_events"] = prior_event_sum(d, wicket_count).astype(int)
    out["audit_recent_window_events"] = grouped.cumcount().clip(upper=10)
    out["audit_runs_total"] = d.runs_total
    out["audit_wicket_count"] = wicket_count.astype(int)
    out["audit_post_score"] = d.innings_score
    out["audit_legal_delivery"] = d.legal_delivery
    return out, records


def validate_features(frame, matches, event_count):
    chase_fields = ["target_runs", "runs_required", "required_run_rate", "run_rate_differential"]
    first = frame.match_innings.eq(1)
    checks = {
        "one_row_per_source_event": len(frame) == event_count and frame.source_row.is_unique,
        "same_match_cohort": set(frame.match_id) == set(matches.match_id),
        "two_innings_per_match": frame.groupby("match_id").match_innings.nunique().eq(2).all(),
        "fixed_binary_target": frame[TARGET].isin([0, 1]).all() and frame.groupby("match_id")[TARGET].nunique().eq(1).all(),
        "first_innings_chase_fields_absent": frame.loc[first, chase_fields].isna().all().all(),
        "second_innings_chase_fields_present": frame.loc[~first, chase_fields].notna().all().all(),
        "no_missing_common_features": frame[[c for c in BASE_NUMERIC + BASE_CATEGORICAL + IDENTITY_CANDIDATES + RECENT_CANDIDATES if c not in chase_fields]].notna().all().all(),
        "finite_numeric_features": np.isfinite(frame[BASE_NUMERIC + RECENT_CANDIDATES].fillna(0).to_numpy(dtype=float)).all(),
        "valid_wickets": frame.wickets_remaining.between(1, 10).all(),
        "nonnegative_score_and_slots": frame.pre_score.ge(0).all() and frame.scheduled_balls_remaining_estimate.between(0, 120).all(),
        "empty_window_at_innings_start": frame.loc[frame.event_sequence.eq(1), RECENT_CANDIDATES].eq(0).all().all(),
        "phase_matches_observed_over": frame.innings_phase.eq(np.select([frame.source_over.lt(6), frame.source_over.lt(15)], ["powerplay", "middle"], default="death")).all(),
    }
    checks = {k: bool(v) for k, v in checks.items()}
    if not all(checks.values()):
        raise ValueError(f"Stage 02 checks failed: {[k for k,v in checks.items() if not v]}")
    return checks


def input_fingerprints(root):
    root = Path(root)
    audit = verified_audit(root)
    path = root / "data/processed/matches/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("audit_id") != audit["audit_id"]:
        raise ValueError("Stage 01 clean export is incomplete or stale. Review/export Stage 01 first.")
    required = {f"data/processed/matches/{name}" for name in
                ["clean_matches.csv", "clean_deliveries.csv.gz", "innings_metadata.csv"]}
    if set(manifest["output_sha256"]) != required:
        raise ValueError("Unexpected Stage 01 exported file contract")
    for relative, expected in manifest["output_sha256"].items():
        if sha256(rooted(root, relative)) != expected:
            raise ValueError(f"Stage 01 exported file changed: {relative}")
    paths = sorted(required | {"data/processed/matches/manifest.json", "config/game_state.json",
                               "src/game_state_features.py"})
    return {name: sha256(root / name) for name in paths}, manifest


def verified_output(root):
    root = Path(root)
    path = root / DATA_DIR / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError("No Stage 02 output. Set RUN_STAGE = True once to build features.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    current, _ = input_fingerprints(root)
    if manifest.get("status") != "complete" or manifest.get("input_sha256") != current:
        raise ValueError("Stage 02 output is stale/incomplete. Set RUN_STAGE = True explicitly; no automatic rebuild.")
    for relative, expected in manifest["output_sha256"].items():
        if sha256(rooted(root, relative)) != expected:
            raise ValueError(f"Stage 02 output changed: {relative}")
    summary = json.loads((root / REPORT_DIR / "feature_summary.json").read_text(encoding="utf-8"))
    return summary


def feature_dictionary(frame):
    descriptions = {
        "match_innings": "Innings indicator (1 or 2); target perspective does not change.",
        "pre_score": "Cumulative score before event; includes pre-innings and earlier event penalties only.",
        "pre_wickets": "Dismissals before event; temporary retirement is not a wicket.",
        "wickets_remaining": "10 minus dismissals; NOT roster-based available batter count.",
        "legal_balls_elapsed": "Observed legal deliveries completed before this event.",
        "source_over": "Observed zero-based over coordinate; never future miscount length.",
        "scheduled_balls_remaining_estimate": "6*(19-source_over)+max(6-prior legal balls in current over,0). Nominal slots, not guaranteed balls left.",
        "current_run_rate": "6*pre_score/legal_balls_elapsed; zero before first legal ball, flagged separately.",
        "no_legal_ball_yet": "One until the first legal delivery has completed.",
        "innings_phase": "Observed overs 1-6 powerplay, 7-15 middle, 16-20 death (analysis convention).",
        "target_runs": "Official chase target, available only in innings 2; NA in innings 1.",
        "runs_required": "Official target minus pre-score; innings 2 only.",
        "required_run_rate": "6*runs_required/max(nominal remaining slots,1); innings 2 only. Zero-slot miscounts flagged.",
        "run_rate_differential": "Current minus required rate; innings 2 only.",
        "toss_first_team_won": "Toss winner is first-innings batting team.",
        "toss_elected_bat": "Toss winner elected to bat. Retained as an alternative; redundant with toss_first_team_won in this standard cohort.",
        "runs_prev_10_events": "Total runs including extras in previous up-to-10 recorded events of this innings, excluding current event.",
        "wickets_prev_10_events": "True dismissals in previous up-to-10 recorded events of this innings; excludes temporary retirements.",
        TARGET: "Label only: first-innings batting team eventually won, constant across both innings.",
    }
    rows = []
    for column in frame.columns:
        role = ("label" if column == TARGET else "base_numeric" if column in BASE_NUMERIC
                else "base_categorical" if column in BASE_CATEGORICAL else "optional_identity" if column in IDENTITY_CANDIDATES
                else "optional_recent_events" if column in RECENT_CANDIDATES else "audit_only" if column.startswith("audit_")
                else "identifier_or_diagnostic")
        definition = descriptions.get(column, "Identifier / diagnostic only; excluded from the default model feature list.")
        if column.endswith("_prior_win_rate"):
            definition = "Strict prior-date gender-specific cohort wins/counts with configured Beta pseudo-counts. No current-date results."
        elif column.endswith("_recent_win_rate"):
            definition = "Prior-date match-count-decayed wins/counts with configured Beta pseudo-counts; half-life in config."
        rows.append({"feature": column, "role": role, "definition": definition,
                     "missing_rows": int(frame[column].isna().sum()), "dtype": str(frame[column].dtype)})
    return pd.DataFrame(rows)


def run(root):
    root = Path(root).resolve()
    config = read_config(root)
    inputs, upstream = input_fingerprints(root)
    source = root / "data/processed/matches"
    deliveries = pd.read_csv(source / "clean_deliveries.csv.gz", low_memory=False)
    matches = pd.read_csv(source / "clean_matches.csv")
    innings = pd.read_csv(source / "innings_metadata.csv")
    frame, background = build_features(deliveries, matches, innings, config)
    checks = validate_features(frame, matches, len(deliveries))
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    output, report = root / DATA_DIR, root / REPORT_DIR
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", {"status": "writing", "feature_id": fingerprint})
    frame.to_csv(output / "game_state_features.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    background.to_csv(output / "match_background.csv", index=False)
    feature_dictionary(frame).to_csv(report / "feature_dictionary.csv", index=False)
    write_json(report / "feature_contract.json", feature_contract())
    coverage = frame.groupby(["gender", "match_innings", "innings_phase"], sort=True).agg(
        events=("source_row", "size"), matches=("match_id", "nunique")).reset_index()
    coverage.to_csv(report / "phase_coverage.csv", index=False)
    frame.head(30).to_csv(report / "feature_preview.csv", index=False)
    anomalies = frame.loc[frame.audit_overrun_before_event | frame.audit_rate_denominator_floored,
                          ["match_id", "match_innings", "event_sequence", "source_over", "legal_balls_elapsed",
                           "audit_pre_legal_in_over", "scheduled_balls_remaining_estimate", "audit_rate_denominator_floored"]]
    anomalies.to_csv(report / "nominal_clock_exceptions.csv", index=False)
    summary = {
        "status": "complete", "stage": STAGE, "created_utc": datetime.now(timezone.utc).isoformat(),
        "feature_id": fingerprint, "upstream_audit_id": upstream["audit_id"],
        "matches": len(matches), "events": len(frame), "innings": len(innings),
        "matches_by_gender": matches.groupby("gender").size().to_dict(),
        "events_by_gender": frame.groupby("gender").size().to_dict(),
        "checks": checks, "config": config, "models_trained": 0,
        "overrun_events": len(anomalies), "rate_denominator_floored_events": int(frame.audit_rate_denominator_floored.sum()),
        "python_version": sys.version.split()[0], "pandas_version": pd.__version__,
        "limitations": ["Nominal wickets are not actual available batters; Stage 04 owns dynamic availability.",
                        "Team history uses the retained cohort only, not every international match ever played.",
                        "Recent event and identity blocks remain optional until validation comparison in Stage 06.",
                        "No prediction improvement or calibration claim is established by this feature export."]}
    write_json(report / "feature_summary.json", summary)
    current, _ = input_fingerprints(root)
    if current != inputs:
        raise ValueError("Inputs changed during export; output is marked incomplete. Re-run explicitly.")
    paths = [output / "game_state_features.csv.gz", output / "match_background.csv"] + [report / name for name in
             ["feature_dictionary.csv", "feature_contract.json", "phase_coverage.csv", "feature_preview.csv",
              "nominal_clock_exceptions.csv", "feature_summary.json"]]
    write_json(output / "manifest.json", {"status": "complete", "feature_id": fingerprint,
               "upstream_audit_id": upstream["audit_id"], "input_sha256": inputs,
               "output_sha256": {p.relative_to(root).as_posix(): sha256(p) for p in paths}})
    print(json.dumps({k: summary[k] for k in ["stage", "matches", "events", "models_trained"]}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    run(args.root)
