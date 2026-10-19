"""Stage 03: ID-linked prior-date player profiles, separate quality dimensions.

No models, split selection, in-match availability or remaining-resource sums.
Profiles for eventual replacements are lookup records, not pregame predictors.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

import game_state_features as state
from data_preparation import DISMISSALS, TEMPORARY_RETIREMENTS, rooted, sha256, write_json

STAGE = "03_player_quality_features"
DATA_DIR = "data/processed/historical_features"
REPORT_DIR = "outputs/" + STAGE
BOWLER_WICKETS = {"bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"}
COUNTS = ["batting_runs", "balls_faced", "batting_dismissals", "boundary_balls", "batting_dot_balls",
          "bowling_runs_conceded", "legal_balls_bowled", "bowler_wickets", "bowling_dot_balls",
          "batting_matches", "bowling_matches", "observed_matches", "listed_matches"]
# name: numerator, denominator, reporting multiplier, pseudo-observation unit
METRICS = {
    "batting_strike_rate": ("batting_runs", "balls_faced", 100., "balls"),
    "batting_average": ("batting_runs", "batting_dismissals", 1., "dismissals"),
    "batting_boundary_rate": ("boundary_balls", "balls_faced", 1., "balls"),
    "batting_dot_rate": ("batting_dot_balls", "balls_faced", 1., "balls"),
    "bowling_economy": ("bowling_runs_conceded", "legal_balls_bowled", 6., "balls"),
    "bowling_wicket_rate": ("bowler_wickets", "legal_balls_bowled", 1., "balls"),
    "bowling_dot_rate": ("bowling_dot_balls", "legal_balls_bowled", 1., "balls"),
}
BATTING_METRICS = [m for m in METRICS if m.startswith("batting_")]
BOWLING_METRICS = [m for m in METRICS if m.startswith("bowling_")]
ROLE_METRICS = {"striker": BATTING_METRICS, "non_striker": BATTING_METRICS, "bowler": BOWLING_METRICS}
KEYS = ["match_id", "gender", "match_date", "player_id"]


def read_config(root):
    config = json.loads((Path(root) / "config/player_quality.json").read_text(encoding="utf-8"))
    fixed = {"stage": STAGE, "history_scope": "Stage 01 retained cohort only",
             "time_rule": "strictly earlier match dates; same-date results update only after all snapshots",
             "identity_rule": "Cricsheet registry ID within gender; names are display labels",
             "reference_policy": "Gender-specific prior-date pooled rates, regularised by fixed cold-start pseudo-observations",
             "participant_policy": "Profile every listed participant; listing is NOT proof of starting XI or live availability",
             "batting_ball_definition": "recorded event without wides; includes no-ball events",
             "batting_dot_definition": "zero batter runs on a non-wide event",
             "bowling_dot_definition": "zero total event runs on a legal delivery"}
    if any(config.get(k) != v for k, v in fixed.items()):
        raise ValueError("Unsupported Stage 03 policy; change implementation and tests explicitly.")
    for key in ["player_pseudo_balls", "player_pseudo_dismissals", "population_pseudo_balls", "population_pseudo_dismissals"]:
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if set(config["cold_start"]) != set(METRICS):
        raise ValueError("Cold-start reference metrics differ from contract")
    for name, value in config["cold_start"].items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid cold-start value: {name}")
        if name.endswith("_rate") and name != "batting_strike_rate" and value >= 1:
            raise ValueError(f"Initial proportion/rate must lie in (0,1): {name}")
    return config


def roster_from_document(doc, match):
    """Identity catalogue only; do NOT classify starting/active/replacement roles."""
    info = doc["info"]
    if info["gender"] != match.gender or str(info["dates"][0]) != str(match.match_date)[:10]:
        raise ValueError(f"Roster metadata mismatch for {match.match_id}")
    teams = {match.first_batting_team, match.second_batting_team}
    if set(info["teams"]) != teams or set(info["players"]) != teams:
        raise ValueError(f"Roster team mismatch for {match.match_id}")
    registry = info["registry"]["people"]
    rows, seen_ids, seen_names = [], set(), set()
    for team, names in info["players"].items():
        if len(names) < 11 or len(names) != len(set(names)):
            raise ValueError(f"Incomplete/duplicate participant list: {match.match_id}, {team}")
        for name in names:
            pid = registry.get(name)
            if not isinstance(pid, str) or not pid.strip() or pid in seen_ids or name in seen_names:
                raise ValueError(f"Missing/ambiguous player ID: {match.match_id}, {name}")
            seen_ids.add(pid); seen_names.add(name)
            rows.append({"match_id": int(match.match_id), "match_date": str(match.match_date)[:10],
                         "gender": match.gender, "team": team, "player_name": name, "player_id": pid,
                         "catalogue_scope": "listed_participant_not_live_availability"})
    return rows


def load_rosters(root, background):
    rows = []
    with ExitStack() as stack:
        archives = {g: stack.enter_context(ZipFile(Path(root) / f"data/raw/cricsheet/t20s_{g}_json.zip"))
                    for g in ["female", "male"]}
        for match in background.itertuples(index=False):
            doc = json.loads(archives[match.gender].read(f"{match.match_id}.json"))
            rows.extend(roster_from_document(doc, match))
    result = pd.DataFrame(rows)
    result["match_date"] = pd.to_datetime(result.match_date).dt.normalize()
    return result


def link_event_identities(deliveries, rosters):
    d = deliveries.copy()
    if d.source_row.duplicated().any() or rosters.duplicated(["match_id", "player_id"]).any():
        raise ValueError("Duplicate event or participant identity")
    if rosters.duplicated(["match_id", "player_name"]).any():
        raise ValueError("Ambiguous participant name within match")
    d["match_date"] = pd.to_datetime(d.match_date).dt.normalize()
    lookup = rosters[["match_id", "gender", "match_date", "player_name", "player_id", "team"]]
    for role in ["striker", "non_striker", "bowler", "player_out"]:
        d = d.merge(lookup.rename(columns={"player_name": role, "player_id": role + "_id", "team": role + "_team"}),
                    on=["match_id", "gender", "match_date", role], how="left", sort=False, validate="many_to_one")
        required = d[role].notna() & d[role].ne("")
        if role != "player_out" and not required.all():
            raise ValueError(f"Missing actor name: {role}")
        if d.loc[required, role + "_id"].isna().any():
            raise ValueError(f"Unresolved actor identity: {role}; do not silently use a cold-start profile")
        team_column = "bowling_team" if role == "bowler" else "batting_team"
        if not d.loc[required, role + "_team"].eq(d.loc[required, team_column]).all():
            raise ValueError(f"Actor assigned to wrong team: {role}")
    if not np.array_equal(deliveries.source_row.to_numpy(), d.source_row.to_numpy()):
        raise ValueError("Identity join changed event order")
    kind = d.wicket_kind.fillna("")
    if not kind.isin(DISMISSALS_AND_RETIREMENTS | {""}).all():
        raise ValueError("Unknown dismissal/retirement kind")
    if not (kind.ne("") == d.player_out_id.notna()).all():
        raise ValueError("Exit kind and player identity disagree")
    return d


DISMISSALS_AND_RETIREMENTS = DISMISSALS | TEMPORARY_RETIREMENTS


def match_contributions(d, rosters):
    """Event accounting aggregated only after match, before prior-date replay."""
    bat = d[["match_id", "gender", "match_date", "striker_id"]].rename(columns={"striker_id": "player_id"}).copy()
    bat["batting_runs"] = d.runs_batter
    bat["balls_faced"] = d.wides.eq(0).astype(int)
    bat["boundary_balls"] = (d.wides.eq(0) & d.runs_batter.isin([4, 6])).astype(int)
    bat["batting_dot_balls"] = (d.wides.eq(0) & d.runs_batter.eq(0)).astype(int)
    dismissal = d.loc[d.wicket_kind.isin(DISMISSALS), ["match_id", "gender", "match_date", "player_out_id"]].rename(columns={"player_out_id": "player_id"})
    dismissal["batting_dismissals"] = 1
    bowl = d[["match_id", "gender", "match_date", "bowler_id"]].rename(columns={"bowler_id": "player_id"}).copy()
    bowl["bowling_runs_conceded"] = d.runs_batter + d.wides + d.noballs
    bowl["legal_balls_bowled"] = d.legal_delivery
    bowl["bowler_wickets"] = d.wicket_kind.isin(BOWLER_WICKETS).astype(int)
    bowl["bowling_dot_balls"] = (d.legal_delivery.eq(1) & d.runs_total.eq(0)).astype(int)
    def appearances(roles, name):
        pieces = [d.loc[d[role + "_id"].notna(), ["match_id", "gender", "match_date", role + "_id"]].rename(columns={role + "_id": "player_id"}) for role in roles]
        out = pd.concat(pieces, ignore_index=True).drop_duplicates(KEYS)
        out[name] = 1
        return out
    pieces = [bat, dismissal, bowl, appearances(["striker", "non_striker", "player_out"], "batting_matches"),
              appearances(["bowler"], "bowling_matches"),
              appearances(["striker", "non_striker", "bowler", "player_out"], "observed_matches")]
    totals = pd.concat(pieces, ignore_index=True).groupby(KEYS, as_index=False, sort=False).sum(numeric_only=True)
    result = rosters.merge(totals, on=KEYS, how="left", validate="one_to_one")
    result["listed_matches"] = 1  # Source listing, NOT an assertion that all are initial XI.
    result[COUNTS] = result[COUNTS].fillna(0).astype("int64")
    if result[COUNTS].lt(0).any().any():
        raise ValueError("Negative player contribution")
    expected = {
        "batting_runs": d.runs_batter.sum(), "balls_faced": d.wides.eq(0).sum(),
        "batting_dismissals": d.wicket_kind.isin(DISMISSALS).sum(),
        "bowling_runs_conceded": (d.runs_batter + d.wides + d.noballs).sum(),
        "legal_balls_bowled": d.legal_delivery.sum(), "bowler_wickets": d.wicket_kind.isin(BOWLER_WICKETS).sum()}
    for key, value in expected.items():
        if result[key].sum() != value:
            raise ValueError(f"Player accounting failed: {key}")
    return result


def profile_values(totals, population, config):
    """Raw and regularised rates with separately visible evidence strength."""
    t = dict(zip(COUNTS, totals)); pool = dict(zip(COUNTS, population))
    values = {"prior_" + key: int(t[key]) for key in COUNTS}
    for metric, (numerator, denominator, multiplier, unit) in METRICS.items():
        pop_k = config["population_pseudo_" + unit]
        player_k = config["player_pseudo_" + unit]
        seed = config["cold_start"][metric]
        reference = (multiplier * pool[numerator] + seed * pop_k) / (pool[denominator] + pop_k)
        values["reference_" + metric] = reference
        values["raw_" + metric] = multiplier * t[numerator] / t[denominator] if t[denominator] > 0 else np.nan
        values["smoothed_" + metric] = (multiplier * t[numerator] + reference * player_k) / (t[denominator] + player_k)
    values["no_prior_batting_balls"] = int(t["balls_faced"] == 0)
    values["no_prior_bowling_balls"] = int(t["legal_balls_bowled"] == 0)
    values["no_prior_batting_dismissals"] = int(t["batting_dismissals"] == 0)
    values["no_prior_bowler_wickets"] = int(t["bowler_wickets"] == 0)
    values["batting_ball_reliability"] = t["balls_faced"] / (t["balls_faced"] + config["player_pseudo_balls"])
    values["batting_average_reliability"] = t["batting_dismissals"] / (t["batting_dismissals"] + config["player_pseudo_dismissals"])
    values["bowling_ball_reliability"] = t["legal_balls_bowled"] / (t["legal_balls_bowled"] + config["player_pseudo_balls"])
    values["reference_prior_balls_faced"] = int(pool["balls_faced"])
    values["reference_prior_balls_bowled"] = int(pool["legal_balls_bowled"])
    return values


def build_profiles(contributions, config):
    data = contributions.sort_values(["match_date", "gender", "match_id", "player_id"], kind="stable")
    history = defaultdict(lambda: np.zeros(len(COUNTS), dtype=np.int64))
    population = defaultdict(lambda: np.zeros(len(COUNTS), dtype=np.int64))
    last_seen = {}
    rows = []
    for date, day in data.groupby("match_date", sort=True):
        for row in day.itertuples(index=False):
            key = (row.gender, row.player_id)
            record = {"match_id": row.match_id, "match_date": date, "gender": row.gender,
                      "team": row.team, "player_name": row.player_name, "player_id": row.player_id,
                      "catalogue_scope": row.catalogue_scope,
                      "history_last_date": last_seen.get(key, pd.NaT)}
            record.update(profile_values(history[key], population[row.gender], config))
            rows.append(record)
        # All snapshots on this date are emitted before any current-day update.
        for row in day.itertuples(index=False):
            additions = np.asarray([getattr(row, key) for key in COUNTS], dtype=np.int64)
            key = (row.gender, row.player_id)
            history[key] += additions
            population[row.gender] += additions
            last_seen[key] = date
    return pd.DataFrame(rows)


def role_columns(role):
    metrics = ["smoothed_" + m for m in ROLE_METRICS[role]]
    evidence = (["no_prior_bowling_balls", "bowling_ball_reliability"] if role == "bowler" else
                ["no_prior_batting_balls", "batting_ball_reliability", "batting_average_reliability"])
    return metrics, evidence


def attach_current_players(states, linked, profiles):
    keys = ["match_id", "source_row", "match_innings", "event_sequence"]
    if len(states) != len(linked) or set(states.source_row) != set(linked.source_row):
        raise ValueError("Stage 02 and clean event coverage differ")
    ordered = linked.set_index("source_row").loc[states.source_row]
    for column in ["match_id", "match_innings", "event_sequence", "striker", "non_striker", "bowler"]:
        if not np.array_equal(states[column].to_numpy(), ordered[column].to_numpy()):
            raise ValueError(f"Stage 02 actor/event alignment mismatch: {column}")
    ids = linked[keys + [role + "_id" for role in ROLE_METRICS]]
    out = states.merge(ids, on=keys, how="left", validate="one_to_one", sort=False)
    for role in ROLE_METRICS:
        quality, evidence = role_columns(role)
        columns = quality + evidence
        lookup = profiles[["match_id", "gender", "player_id", *columns]].rename(
            columns={"player_id": role + "_id", **{c: role + "_" + c for c in columns}})
        out = out.merge(lookup, on=["match_id", "gender", role + "_id"], how="left", validate="many_to_one", sort=False)
    # No fallback zero on a failed ID join: missing identity is not an unseen player.
    additions = [c for c in out if c not in states]
    if out[additions].isna().any().any():
        raise ValueError("Missing current-player profile after identity join")
    pd.testing.assert_frame_equal(states.reset_index(drop=True), out[states.columns].reset_index(drop=True), check_dtype=True)
    return out


def feature_contract():
    quality = [role + "_" + c for role in ROLE_METRICS for c in role_columns(role)[0]]
    evidence = [role + "_" + c for role in ROLE_METRICS for c in role_columns(role)[1]]
    return {
        "target": state.TARGET, "prediction_timing": "pre_delivery, qualities fixed from strictly earlier dates",
        "inherits": "Stage 02 explicit background contract and optional blocks; never auto-select numeric columns",
        "current_player_quality": quality, "current_player_evidence": evidence,
        "current_player_block": quality + evidence,
        "design": "Four separate batting dimensions for each batter and three separate bowling dimensions; evidence fields accompany the block. No arbitrary single composite score.",
        "identity": "Stable registry IDs for joins only, not model features. Names are display labels.",
        "population": "Gender-specific prior-date retained-cohort history; not complete career statistics.",
        "profile_catalogue": "Listed participants include eventual replacements. Profile values are based on pre-date history, but membership is retrospective: never aggregate all catalogue rows as pregame resources. Stage 04 reconstructs active players.",
        "raw_rates": "Zero denominator remains NA in raw diagnostics. Smoothed metrics are finite; ID join failures stop, never become cold-start values.",
        "smoothing": "Prespecified pseudo-exposure defaults in config/player_quality.json. No current/future-date pool, full-dataset fitting, or test-driven tuning.",
        "handoff": "Stage 04 decides dynamic availability. Stage 05 combines availability with these shared profiles. Stage 06 evaluates predictive benefit; no benefit established here.",
        "evaluation": "Date-forward rolling history may include earlier completed held-out matches; fitting and selection follow the later Stage 06 protocol, never use future results.",
    }


def validate(profiles, current, states, rosters):
    smoothed = ["smoothed_" + m for m in METRICS]
    no_history = profiles.prior_observed_matches.eq(0)
    checks = {
        "one_profile_per_listed_player_match": len(profiles) == len(rosters) and not profiles.duplicated(["match_id", "player_id"]).any(),
        "profile_cohort_preserved": set(profiles.match_id) == set(states.match_id),
        "same_event_count": len(current) == len(states) and current.source_row.is_unique,
        "same_source_order": np.array_equal(current.source_row, states.source_row),
        "finite_smoothed_metrics": np.isfinite(profiles[smoothed].to_numpy()).all(),
        "nonnegative_smoothed_metrics": profiles[smoothed].ge(0).all().all(),
        "nonnegative_prior_counts": profiles[["prior_" + c for c in COUNTS]].ge(0).all().all(),
        "history_strictly_before_date": (profiles.history_last_date.isna() | (pd.to_datetime(profiles.history_last_date) < pd.to_datetime(profiles.match_date))).all(),
        "no_missing_model_features": current[feature_contract()["current_player_block"]].notna().all().all(),
        "unseen_players_equal_prior_reference": all(np.allclose(profiles.loc[no_history, "smoothed_" + m], profiles.loc[no_history, "reference_" + m]) for m in METRICS),
        "raw_zero_denominators_remain_missing": all(profiles.loc[profiles["prior_" + spec[1]].eq(0), "raw_" + metric].isna().all() for metric, spec in METRICS.items()),
    }
    for metric in ["batting_boundary_rate", "batting_dot_rate", "bowling_dot_rate"]:
        checks[metric + "_bounded"] = profiles["smoothed_" + metric].between(0, 1).all()
    checks = {k: bool(v) for k, v in checks.items()}
    if not all(checks.values()):
        raise ValueError(f"Stage 03 checks failed: {[k for k,v in checks.items() if not v]}")
    pd.testing.assert_frame_equal(states.reset_index(drop=True), current[states.columns].reset_index(drop=True))
    return checks


def input_fingerprints(root):
    root = Path(root)
    upstream = state.verified_output(root)
    paths = ["config/player_quality.json", "src/player_quality_features.py",
             "data/processed/matches/clean_deliveries.csv.gz",
             "data/processed/matches/manifest.json",
             "data/processed/game_state/game_state_features.csv.gz",
             "data/processed/game_state/match_background.csv",
             "data/processed/game_state/manifest.json",
             "data/raw/cricsheet/t20s_female_json.zip", "data/raw/cricsheet/t20s_male_json.zip"]
    return {p: sha256(root / p) for p in paths}, upstream


def verified_output(root):
    root = Path(root)
    path = root / DATA_DIR / "player_quality_manifest.json"
    if not path.exists():
        raise FileNotFoundError("No Stage 03 export. Set RUN_STAGE = True explicitly to build it.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    current, _ = input_fingerprints(root)
    if manifest.get("status") != "complete" or manifest.get("input_sha256") != current:
        raise ValueError("Stage 03 output stale/incomplete. Set RUN_STAGE = True; no automatic rebuild.")
    for relative, expected in manifest["output_sha256"].items():
        if sha256(rooted(root, relative)) != expected:
            raise ValueError(f"Stage 03 output changed: {relative}")
    return json.loads((root / REPORT_DIR / "quality_summary.json").read_text(encoding="utf-8"))


def coverage_report(profiles):
    rows = []
    for gender, group in profiles.groupby("gender"):
        for role, column in [("batting", "prior_balls_faced"), ("bowling", "prior_legal_balls_bowled")]:
            counts = group[column]
            for label, mask in [("No prior balls", counts.eq(0)), ("1-119 prior balls", counts.between(1, 119)),
                                ("120+ prior balls", counts.ge(120))]:
                rows.append({"gender": gender, "role": role, "history_band": label,
                             "player_match_profiles": int(mask.sum()), "proportion": float(mask.mean())})
    return pd.DataFrame(rows)


def metric_dictionary():
    definitions = {
        "batting_strike_rate": "Batter runs per 100 non-wide events (no-ball events included).",
        "batting_average": "Batter runs per true dismissal; temporary retirement excluded. No dismissal -> raw NA, not zero.",
        "batting_boundary_rate": "Non-wide events with batter runs 4 or 6 divided by faced events; not boundary-run share.",
        "batting_dot_rate": "Non-wide events with zero batter runs divided by faced events; byes may occur.",
        "bowling_economy": "6*(batter runs + wides + no-ball extras)/legal balls; byes, legbyes and separate penalties excluded.",
        "bowling_wicket_rate": "Bowler-credited dismissals per legal ball; run out, retired out, timed out etc excluded. Not a Bernoulli probability.",
        "bowling_dot_rate": "Legal events with zero total runs divided by legal balls; different from batting dot rate.",
    }
    return pd.DataFrame([{"metric": m, "definition": definitions[m], "numerator": spec[0], "denominator": spec[1],
                          "multiplier": spec[2], "pseudo_unit": spec[3],
                          "raw_zero_denominator": "NA", "snapshot": "strictly before match date"} for m,spec in METRICS.items()])


def run(root):
    root = Path(root).resolve()
    config = read_config(root)
    inputs, upstream = input_fingerprints(root)
    d = pd.read_csv(root / "data/processed/matches/clean_deliveries.csv.gz", low_memory=False)
    states = pd.read_csv(root / "data/processed/game_state/game_state_features.csv.gz", low_memory=False)
    background = pd.read_csv(root / "data/processed/game_state/match_background.csv")
    rosters = load_rosters(root, background)
    linked = link_event_identities(d, rosters)
    contributions = match_contributions(linked, rosters)
    profiles = build_profiles(contributions, config)
    current = attach_current_players(states, linked, profiles)
    checks = validate(profiles, current, states, rosters)
    identity = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    output, report = root / DATA_DIR, root / REPORT_DIR
    output.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    write_json(output / "player_quality_manifest.json", {"status": "writing", "quality_id": identity})
    files = []
    for name, frame in [("player_match_profiles.csv.gz", profiles), ("player_match_contributions.csv.gz", contributions),
                        ("current_player_features.csv.gz", current), ("participant_registry.csv", rosters)]:
        path = output / name
        compression = {"method": "gzip", "mtime": 0} if name.endswith(".gz") else None
        frame.to_csv(path, index=False, compression=compression); files.append(path)
    coverage = coverage_report(profiles)
    role_rows = [{"role": role, "gender": gender, "events": len(group),
                  "no_prior_balls_events": int(group[role + ("_no_prior_bowling_balls" if role == "bowler" else "_no_prior_batting_balls")].sum())}
                 for role in ROLE_METRICS for gender,group in current.groupby("gender")]
    registry_audit = rosters.groupby(["match_id", "gender", "team"]).agg(listed_participants=("player_id", "size")).reset_index()
    registry_audit["more_than_eleven_listed"] = registry_audit.listed_participants.gt(11)
    aliases = rosters.groupby(["gender", "player_id"]).agg(display_names=("player_name", lambda x: " | ".join(sorted(set(x)))),
                distinct_names=("player_name", "nunique"), listed_matches=("match_id", "nunique")).reset_index()
    references = profiles[["gender", "match_date", *["reference_" + m for m in METRICS],
                          "reference_prior_balls_faced", "reference_prior_balls_bowled"]].drop_duplicates()
    for name, frame in [("history_coverage.csv", coverage), ("current_role_coverage.csv", pd.DataFrame(role_rows)),
                        ("participant_catalogue_audit.csv", registry_audit), ("player_aliases.csv", aliases),
                        ("metric_dictionary.csv", metric_dictionary()), ("prior_date_references.csv", references),
                        ("profile_preview.csv", profiles.head(30))]:
        path = report / name; frame.to_csv(path, index=False); files.append(path)
    contract = feature_contract()
    write_json(report / "feature_contract.json", contract); files.append(report / "feature_contract.json")
    summary = {"status": "complete", "stage": STAGE, "created_utc": datetime.now(timezone.utc).isoformat(),
               "quality_id": identity, "upstream_feature_id": upstream["feature_id"],
               "matches": int(profiles.match_id.nunique()), "events": len(current), "profiles": len(profiles),
               "players_by_gender": profiles.groupby("gender").player_id.nunique().to_dict(),
               "listed_profiles_by_gender": profiles.groupby("gender").size().to_dict(),
               "matches_with_extra_listed_participants": int(registry_audit.loc[registry_audit.more_than_eleven_listed, "match_id"].nunique()),
               "ids_with_multiple_display_names": int(aliases.distinct_names.gt(1).sum()),
               "current_quality_features": len(contract["current_player_quality"]),
               "current_evidence_features": len(contract["current_player_evidence"]),
               "checks": checks, "config": config, "models_trained": 0,
               "accounting": {key: int(contributions[key].sum()) for key in COUNTS},
               "limitations": ["Only retained-cohort history, not full careers.",
                               "Prespecified smoothing defaults are not established as optimal.",
                               "Listed participants are not the starting XI or the active-player set.",
                               "Quality is a prior-date statistical profile, not causal skill or a calibrated prediction.",
                               "No claim of improved prediction; Stage 06-07 must evaluate this."]}
    write_json(report / "quality_summary.json", summary); files.append(report / "quality_summary.json")
    current_inputs, _ = input_fingerprints(root)
    if current_inputs != inputs:
        raise ValueError("Inputs changed during export; incomplete output must be regenerated explicitly.")
    write_json(output / "player_quality_manifest.json", {"status": "complete", "quality_id": identity,
               "input_sha256": inputs, "output_sha256": {p.relative_to(root).as_posix(): sha256(p) for p in files}})
    print(json.dumps({k:summary[k] for k in ["matches", "events", "profiles", "players_by_gender", "models_trained"]}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    run(parser.parse_args().root)
