"""Stage 04: record-based pre-event availability, separate from player quality.

The event ledger is diagnostic until joined to the separate whole-match audit.
Untimed absence metadata and future review outcomes NEVER become predictors.
No claim of complete medical eligibility, penalty-time or historical-rule replay.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

import player_quality_features as quality
from data_preparation import DISMISSALS, TEMPORARY_RETIREMENTS, rooted, sha256, write_json

STAGE = "04_dynamic_player_availability"
DATA_DIR = "data/processed/player_availability"
REPORT_DIR = "outputs/" + STAGE
KEYS = ["source_row", "match_id", "match_innings", "event_sequence"]
BAT_SCHEMA = ["player_id", "resource_slot_id", "status", "previously_observed_batting"]
BOWL_SCHEMA = ["player_id", "resource_slot_id", "status", "personal_legal_balls",
               "slot_legal_balls", "slot_overs_started", "nominal_capacity"]
TIMELINE_COLUMNS = KEYS + ["timing", "event", "player_id", "detail"]
ISSUE_COLUMNS = ["match_id", "match_innings", "source_row", "reason", "detail", "blocks_resources"]


class ReplayError(ValueError):
    """An unexplained source/state conflict; quarantine the whole match."""


def compact(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def read_config(root):
    value = json.loads((Path(root) / "config/player_availability.json").read_text(encoding="utf-8"))
    if (value["stage"], value["balls_per_over"], value["maximum_overs_per_bowler"], value["scheduled_overs"]) != (STAGE, 6, 4, 20):
        raise ValueError("Unsupported availability policy: update implementation and tests explicitly")
    return value


def source_events(doc):
    for inn, innings in enumerate(doc["innings"], 1):
        if innings.get("super_over"):
            raise ReplayError("Unexpected Super Over")
        for over in innings["overs"]:
            for number, event in enumerate(over["deliveries"], 1):
                yield inn, innings["team"], over["over"], number, event


def initial_rosters(doc):
    """Retrospective XI reconstruction, not a forward lineup-prediction task.

    Only explicit incoming replacement identities are removed. No future batting
    or bowling appearances are used. Activation still occurs at the recorded time.
    """
    initial = {team: set(names) for team, names in doc["info"]["players"].items()}
    incoming_seen = set()
    for _, _, _, _, event in source_events(doc):
        for swap in event.get("replacements", {}).get("match", []):
            if not all(k in swap for k in ("team", "in", "out", "reason")):
                raise ReplayError("Incomplete match replacement")
            team, incoming, outgoing = swap["team"], swap["in"], swap["out"]
            if team not in initial or incoming == outgoing or incoming in incoming_seen:
                raise ReplayError("Ambiguous replacement chain")
            listed = doc["info"]["players"][team]
            if incoming not in listed or outgoing not in listed:
                raise ReplayError("Replacement identity absent from team catalogue")
            incoming_seen.add(incoming)
            initial[team].remove(incoming)
    if len(initial) != 2 or any(len(names) != 11 for names in initial.values()):
        raise ReplayError("Cannot reconstruct exactly eleven initial players per team")
    return initial


def nominal_capacity(started, current_over, pre_over_legal, is_current, allowed=True):
    """Upper-bound ball slots, NOT an actual future allocation or eligibility.

    An observed start, even with a wide/no-ball, uses an over slot. Interrupted
    parts count for each bowler. Extra balls in a miscounted over are not forecast:
    zero nominal current-over slots can coexist with one observed opportunity.
    """
    if not allowed:
        return 0
    overs = set(started)
    if is_current:
        overs.add(current_over)
    return max(0, 4 - len(overs)) * 6 + (max(0, 6 - pre_over_legal) if is_current else 0)


def replay_match(doc, local, catalogue):
    """Return event rows, source-linked transitions, review issues and initial XI.

    All event quantities are emitted BEFORE processing runs, wickets and extras.
    Full-innings absence metadata is inspected only after the replay has ended.
    """
    source = list(source_events(doc))
    if len(source) != len(local) or local.source_row.duplicated().any():
        raise ReplayError("Event coverage differs from Stage 01")
    if not local.source_row.is_monotonic_increasing:
        raise ReplayError("Source rows are not in original order")
    mid = int(local.match_id.iloc[0])
    info = doc["info"]
    if info["gender"] != local.gender.iloc[0] or str(info["dates"][0]) != str(local.match_date.iloc[0])[:10]:
        raise ReplayError("Match date/gender mismatch")
    initial = initial_rosters(doc)
    registry = info["registry"]["people"]
    expected = {(team, name, registry[name]) for team, names in info["players"].items() for name in names}
    actual = set(catalogue[["team", "player_name", "player_id"]].itertuples(index=False, name=None))
    if expected != actual or len({pid for _, _, pid in expected}) != len(expected):
        raise ReplayError("Stage 03 catalogue identity mismatch")
    active = {team: set(names) for team, names in initial.items()}
    activated = {team: set(names) for team, names in initial.items()}
    slots = {name: name for names in initial.values() for name in names}
    dismissed, seen_batting = defaultdict(set), defaultdict(set)
    unavailable_bat, unavailable_bowl, banned = set(), set(), set()
    started, over_bowlers = defaultdict(set), defaultdict(set)
    slot_legal, personal_legal, over_legal = defaultdict(int), defaultdict(int), defaultdict(int)
    last_bowler, last_pair = {}, {}
    rows, timeline, issues = [], [], []
    known_issues = set()

    def log(row, kind, name="", detail="", timing="before"):
        timeline.append({**{k: int(getattr(row, k)) for k in KEYS}, "timing": timing,
                         "event": kind, "player_id": registry.get(name, ""), "detail": detail})

    def issue(row, reason, detail="", block=True, known=True):
        # Match-level issues are auditable; only already observed reasons affect
        # the separate diagnostic known_uncertainty field in subsequent rows.
        if not any(x["reason"] == reason and x["match_innings"] == row.match_innings for x in issues):
            issues.append({"match_id": mid, "match_innings": int(row.match_innings),
                           "source_row": int(row.source_row), "reason": reason,
                           "detail": detail, "blocks_resources": bool(block)})
        if block and known:
            known_issues.add(reason)

    for row, (inn, batting, over, number, event) in zip(local.itertuples(index=False), source):
        bowling = next(t for t in active if t != batting)
        expected_event = (inn, batting, bowling, over, number, event["batter"], event["non_striker"], event["bowler"], event["runs"]["total"])
        observed = (row.match_innings, row.batting_team, row.bowling_team, row.source_over,
                    row.source_delivery_in_over, row.striker, row.non_striker, row.bowler, row.runs_total)
        if expected_event != observed:
            raise ReplayError(f"Event alignment conflict at source row {row.source_row}")
        extras = event.get("extras", {})
        legal = int(not (extras.get("wides", 0) or extras.get("noballs", 0)))
        if legal != row.legal_delivery:
            raise ReplayError("Legal-delivery definition differs from Stage 01")
        replacements = event.get("replacements", {})
        if set(replacements) - {"match", "role"}:
            raise ReplayError("Unsupported replacement type")
        for swap in replacements.get("match", []):
            team, incoming, outgoing = swap["team"], swap["in"], swap["out"]
            if outgoing not in active[team] or incoming in active[team] or incoming in slots:
                raise ReplayError("Invalid active-lineup transition")
            active[team].remove(outgoing); active[team].add(incoming)
            activated[team].add(incoming)
            slots[incoming] = slots[outgoing]
            if outgoing in banned:
                banned.add(incoming)
            issue(row, "match_replacement_restrictions_unverified", compact(swap))
            log(row, "match_replacement", incoming, compact(swap))
        for swap in replacements.get("role", []):
            role, incoming, reason = swap.get("role"), swap.get("in"), swap.get("reason", "unknown")
            if role not in {"bowler", "batter"}:
                raise ReplayError("Unknown replacement role")
            team = bowling if role == "bowler" else batting
            if incoming not in active[team]:
                raise ReplayError("Role replacement is not an active team member")
            outgoing = swap.get("out")
            if outgoing is None and role == "bowler":
                outgoing = last_bowler.get((inn, over))  # never the next/future bowler
            if outgoing is not None:
                if outgoing not in active[team] or outgoing == incoming:
                    raise ReplayError("Invalid outgoing role replacement")
                if role == "bowler" and reason.startswith("excluded"):
                    banned.add(outgoing)
                elif role == "bowler":
                    unavailable_bowl.add(outgoing)
                else:
                    unavailable_bat.add(outgoing)
                # Injury may affect the other role, but the record does not say
                # how. Preserve role-specific evidence and block formal use.
            else:
                issue(row, "unknown_outgoing_role_player", compact(swap))
            issue(row, "role_replacement_restrictions_unverified", compact(swap))
            log(row, "role_replacement", outgoing or "", compact(swap))

        pair = {event["batter"], event["non_striker"]}
        if len(pair) != 2:
            raise ReplayError("Current batting pair is not two distinct players")
        for missing in last_pair.get(inn, set()) - pair:
            if missing in active[batting] and slots[missing] not in dismissed[inn] and missing not in unavailable_bat:
                unavailable_bat.add(missing)
                issue(row, "unexplained_batting_departure", missing)
                log(row, "unexplained_batting_departure", missing, "Actor disappeared without a recorded exit")
        for name in pair:
            if name not in active[batting] or slots[name] in dismissed[inn]:
                raise ReplayError(f"Current batter is inactive or dismissed: {name}")
            if name in unavailable_bat:
                unavailable_bat.remove(name)
                log(row, "observed_batting_return", name, "Return known from current pre-event actor")
        bowler = event["bowler"]
        if bowler not in active[bowling] or bowler in banned:
            raise ReplayError(f"Current bowler inactive or suspended: {bowler}")
        if bowler in unavailable_bowl:
            unavailable_bowl.remove(bowler)
            issue(row, "bowling_return_penalty_time_unverified", bowler)
            log(row, "observed_bowling_return", bowler, "Medical/penalty-time details unavailable")
        previous = last_bowler.get((inn, over))
        if previous and previous != bowler and not replacements:
            issue(row, "unexplained_mid_over_bowler_change", f"{previous} -> {bowler}")
        slot = slots[bowler]
        new_start = over not in started[(inn, slot)]
        if new_start and slot in over_bowlers[(inn, over - 1)]:
            issue(row, "consecutive_over_slot", bowler)
        if new_start and len(started[(inn, slot)]) >= 4:
            issue(row, "observed_quota_exceeded", bowler)
            log(row, "observed_quota_exceeded", bowler, "More than four distinct over slots")
        pre_over_legal = over_legal[(inn, over)]
        if pre_over_legal >= 6:
            issue(row, "observed_over_continuation", "Nominal current-over slots exhausted", block=False)
            log(row, "observed_over_continuation", bowler, "Current opportunity observed; future length unknown")
        started[(inn, slot)].add(over)  # current bowler selection is already known
        over_bowlers[(inn, over)].add(slot)

        bat_states, bowl_states = [], []
        available, waiting = [], []
        for name in sorted(activated[batting], key=lambda n: registry[n]):
            if name not in active[batting]:
                status = "replaced_out"
            elif slots[name] in dismissed[inn]:
                status = "dismissed"
            elif name in unavailable_bat:
                status = "temporarily_unavailable"
            elif name in pair:
                status = "current_batter"
            else:
                status = "waiting_returned" if name in seen_batting[inn] else "waiting_not_yet_batted"
            if status in {"current_batter", "waiting_returned", "waiting_not_yet_batted"}:
                available.append(name)
                if name not in pair:
                    waiting.append(name)
            bat_states.append([registry[name], registry[slots[name]], status, int(name in seen_batting[inn])])
        capacities = {}
        for name in sorted(activated[bowling], key=lambda n: registry[n]):
            if name not in active[bowling]:
                status = "replaced_out"
            elif name in banned:
                status = "suspended"
            elif name in unavailable_bowl:
                status = "temporarily_unavailable"
            else:
                status = "current_bowler" if name == bowler else "inferred_available"
            allowed = status in {"current_bowler", "inferred_available"}
            key = (inn, slots[name])
            capacity = nominal_capacity(started[key], over, pre_over_legal, name == bowler, allowed)
            if allowed and capacity == 0 and name != bowler:
                status = "quota_exhausted"
            capacities[name] = capacity
            bowl_states.append([registry[name], registry[slots[name]], status,
                                personal_legal[(inn, name)], slot_legal[key], len(started[key]), capacity])
        if not pair.issubset(available) or len(available) != len(waiting) + 2 or len(available) > 11:
            raise ReplayError("Batting-state partition does not reconcile")
        nominal_left = max(0, 19 - over) * 6 + max(0, 6 - pre_over_legal)
        rows.append({**{k: int(getattr(row, k)) for k in KEYS}, "gender": row.gender,
                     "match_date": str(row.match_date)[:10], "batting_team": batting, "bowling_team": bowling,
                     "source_over": int(over), "striker_id": registry[event["batter"]],
                     "non_striker_id": registry[event["non_striker"]], "bowler_id": registry[bowler],
                     "available_batter_count": len(available), "waiting_batter_count": len(waiting),
                     "temporarily_unavailable_batter_count": sum(s[2] == "temporarily_unavailable" for s in bat_states),
                     "dismissed_slot_count": len(dismissed[inn]),
                     "bowlers_with_nominal_capacity": sum(c > 0 for c in capacities.values()),
                     "current_bowler_nominal_capacity": capacities[bowler],
                     "current_bowler_prior_legal_balls": personal_legal[(inn, bowler)],
                     "current_bowler_slot_prior_legal_balls": slot_legal[(inn, slot)],
                     "current_bowler_slot_overs_started": len(started[(inn, slot)]),
                     "nominal_scheduled_slots_remaining": nominal_left,
                     "pre_over_legal_balls": pre_over_legal,
                     "observed_current_opportunity": 1,
                     "observed_over_continuation": int(pre_over_legal >= 6),
                     "known_uncertainty": int(bool(known_issues)),
                     "batting_state_json": compact(bat_states), "bowling_state_json": compact(bowl_states)})

        # All event consequences are applied only after the snapshot above.
        seen_batting[inn].update(pair)
        for wicket in event.get("wickets", []):
            name, kind = wicket["player_out"], wicket["kind"]
            if name not in active[batting] or slots[name] in dismissed[inn]:
                raise ReplayError("Exit refers to inactive/already dismissed player")
            if kind in TEMPORARY_RETIREMENTS:
                unavailable_bat.add(name)
                issue(row, "retirement_return_conditions_unverified", kind)
                log(row, "temporary_retirement", name, kind, "after")
            elif kind in DISMISSALS:
                dismissed[inn].add(slots[name])
                log(row, "dismissal", name, kind, "after")
            else:
                raise ReplayError(f"Unknown exit kind: {kind}")
        slot_legal[(inn, slot)] += legal
        personal_legal[(inn, bowler)] += legal
        over_legal[(inn, over)] += legal
        last_bowler[(inn, over)] = bowler
        last_pair[inn] = pair

    # Retrospective review only: never use untimed absences as ball-time inputs.
    for inn, innings in enumerate(doc["innings"], 1):
        if innings.get("absent_hurt"):
            issues.append({"match_id": mid, "match_innings": inn, "source_row": None,
                           "reason": "untimed_absent_hurt", "detail": compact(innings["absent_hurt"]),
                           "blocks_resources": True})
    return pd.DataFrame(rows), timeline, issues, initial


def feature_contract():
    return {
        "prediction_timing": "pre-event; current actors and documented pre-event substitutions known",
        "batting_state_schema": BAT_SCHEMA, "bowling_state_schema": BOWL_SCHEMA,
        "join_keys": KEYS, "quality_join": ["match_id", "gender", "player_id"],
        "base_available_meaning": "inferred from known records, not confirmed full legal/medical eligibility",
        "initial_XI": "retrospectively reconstructed from participant lists minus explicit incoming replacements; no future appearances used",
        "capacity": "nominal over-slot upper bound, not actual allocation, next-bowler eligibility or future miscount length",
        "capacity_aggregation": "Individual upper bounds, not remaining innings balls. Stage 05 v2 sums each nominal capacity / 6 times bowling quality, including the current bowler; no innings-budget allocation or normalization. Total individual capacity need not fit the innings budget.",
        "current_pair": "included in available_batter_count, excluded from waiting_batter_count; avoid double counting in Stage 05",
        "never_entered_resource_pool": "Stage 05 v2 uses waiting_not_yet_batted with previously_observed_batting=0 only. waiting_returned is not a never-entered resource. Current pair status excludes a new non-striker immediately even when their prior-observation flag is zero.",
        "absent_hurt": "untimed full-innings metadata appears in review only, never in event states",
        "formal_gate": "load_resource_events requires explicit whole-match IDs and rejects every match not ready_for_resource_features; freeze identical cohort/dates for all arms",
        "forbidden_predictors": ["match_availability_audit", "review_issues", "initial_lineup audit", "future availability", "outcomes", "player IDs"],
        "diagnostics_not_predictors": ["known_uncertainty", "observed_over_continuation", "observed_current_opportunity"],
        "scope": "No quality weighting, no model, no split, no probability or NWC. Stage 01-03 unchanged.",
    }


def input_fingerprints(root):
    root = Path(root)
    upstream = quality.verified_output(root)
    paths = ["config/player_availability.json", "src/player_availability.py",
             "data/processed/matches/clean_deliveries.csv.gz",
             "data/processed/historical_features/participant_registry.csv",
             "data/processed/historical_features/player_quality_manifest.json",
             "data/raw/cricsheet/t20s_female_json.zip", "data/raw/cricsheet/t20s_male_json.zip"]
    return {p: sha256(root / p) for p in paths}, upstream


def verified_output(root):
    root = Path(root)
    path = root / DATA_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError("No Stage 04 export. Set RUN_STAGE = True explicitly.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    inputs, _ = input_fingerprints(root)
    if manifest.get("status") != "complete" or manifest.get("input_sha256") != inputs:
        raise ValueError("Stage 04 output stale/incomplete. Set RUN_STAGE = True; no automatic rebuild.")
    for relative, expected in manifest["output_sha256"].items():
        if sha256(rooted(root, relative)) != expected:
            raise ValueError(f"Stage 04 output changed: {relative}")
    return json.loads((root / REPORT_DIR / "availability_summary.json").read_text(encoding="utf-8"))


def require_ready_matches(audit, match_ids):
    requested = set(map(int, match_ids))
    ready = set(audit.loc[audit.ready_for_resource_features.eq(True), "match_id"])
    if not requested or not requested.issubset(ready):
        raise ValueError(f"Formal resources require explicit ready whole matches; rejected: {sorted(requested - ready)}")
    return requested


def load_resource_events(root, match_ids):
    """Fail-closed downstream entry; never silently shrink an experimental arm."""
    verified_output(root)
    root = Path(root)
    audit = pd.read_csv(root / REPORT_DIR / "match_availability_audit.csv")
    requested = require_ready_matches(audit, match_ids)
    data = pd.read_csv(root / DATA_DIR / "event_availability.csv.gz")
    return data.loc[data.match_id.isin(requested)].reset_index(drop=True)


def run(root):
    root = Path(root).resolve()
    config = read_config(root)
    inputs, upstream = input_fingerprints(root)
    d = pd.read_csv(root / "data/processed/matches/clean_deliveries.csv.gz", low_memory=False)
    catalogue = pd.read_csv(root / "data/processed/historical_features/participant_registry.csv")
    cat_groups = {int(mid): frame for mid, frame in catalogue.groupby("match_id", sort=False)}
    output, report = root / DATA_DIR, root / REPORT_DIR
    output.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    write_json(output / "manifest.json", {"status": "writing", "availability_id": identity})
    event_path = output / "event_availability.csv.gz"
    all_timeline, all_issues, audits, lineups, numeric = [], [], [], [], []
    event_coverage = d[KEYS].copy(); event_coverage["replay_status"] = "quarantined"
    written_keys, example_frames = [], {}
    # Stream one match at a time: player-state JSON remains small in memory.
    with ExitStack() as stack:
        archives = {g: stack.enter_context(ZipFile(root / f"data/raw/cricsheet/t20s_{g}_json.zip")) for g in ("female", "male")}
        handle = stack.enter_context(gzip.open(event_path, "wt", encoding="utf-8", newline=""))
        header = True
        for index, (mid, local) in enumerate(d.groupby("match_id", sort=False), 1):
            gender = local.gender.iloc[0]
            doc = json.loads(archives[gender].read(f"{mid}.json"))
            try:
                frame, timeline, issues, initial = replay_match(doc, local, cat_groups[int(mid)])
            except ReplayError as exc:
                issues = [{"match_id": int(mid), "match_innings": None, "source_row": None,
                           "reason": "replay_conflict", "detail": str(exc), "blocks_resources": True}]
                frame, timeline, initial = pd.DataFrame(), [], {}
            all_issues.extend(issues); all_timeline.extend(timeline)
            ready = len(frame) == len(local) and not any(i["blocks_resources"] for i in issues)
            status = "ready" if ready else ("review_required" if len(frame) else "quarantined")
            audits.append({"match_id": int(mid), "gender": gender, "match_date": str(local.match_date.iloc[0])[:10],
                           "source_events": len(local), "replayed_events": len(frame), "status": status,
                           "ready_for_resource_features": ready,
                           "reasons": " | ".join(sorted({i["reason"] for i in issues if i["blocks_resources"]}))})
            for team, names in initial.items():
                lineups.extend({"match_id": int(mid), "gender": gender, "team": team,
                               "player_id": doc["info"]["registry"]["people"][name], "player_name": name,
                               "evidence": "participant catalogue minus explicit incoming replacements"} for name in sorted(names))
            if len(frame):
                frame.to_csv(handle, index=False, header=header); header = False
                written_keys.append(frame[KEYS])
                event_coverage.loc[event_coverage.match_id.eq(mid), "replay_status"] = status
                numeric.append(frame.drop(columns=["batting_state_json", "bowling_state_json"]))
                if ready and "ordinary" not in example_frames:
                    example_frames["ordinary"] = frame
                if any(t["event"] == "temporary_retirement" for t in timeline) and "retirement" not in example_frames:
                    example_frames["retirement"] = frame
            if index % 200 == 0:
                print(f"Replayed {index}/{d.match_id.nunique()} matches", flush=True)
    if not numeric:
        raise ValueError("No matches replayed; inspect source and catalogue")
    events = pd.concat(numeric, ignore_index=True)
    key_frame = pd.concat(written_keys, ignore_index=True)
    expected_keys = d.loc[d.match_id.isin(events.match_id), KEYS].reset_index(drop=True)
    pd.testing.assert_frame_equal(key_frame, expected_keys)
    if not events.available_batter_count.eq(events.waiting_batter_count + 2).all():
        raise ValueError("Current pair counted incorrectly")
    if events.current_bowler_nominal_capacity.lt(0).any() or events.current_bowler_nominal_capacity.gt(24).any():
        raise ValueError("Invalid nominal capacity")
    audit = pd.DataFrame(audits)
    issues = pd.DataFrame(all_issues, columns=ISSUE_COLUMNS)
    timeline = pd.DataFrame(all_timeline, columns=TIMELINE_COLUMNS)
    coverage = audit.groupby(["gender", "status"], as_index=False).agg(matches=("match_id", "size"), events=("source_events", "sum"))
    reasons = issues.groupby(["reason", "blocks_resources"], as_index=False).agg(matches=("match_id", "nunique"), records=("match_id", "size"))
    files = [event_path]
    for path, table in [(output / "event_coverage.csv.gz", event_coverage),
                        (output / "initial_lineups.csv", pd.DataFrame(lineups)),
                        (output / "availability_transitions.csv.gz", timeline),
                        (report / "match_availability_audit.csv", audit),
                        (report / "review_issues.csv", issues), (report / "coverage.csv", coverage),
                        (report / "review_reason_counts.csv", reasons), (report / "event_preview.csv", events.head(30))]:
        table.to_csv(path, index=False); files.append(path)
    for label, frame in example_frames.items():
        path = report / f"example_{label}.csv.gz"
        frame.to_csv(path, index=False); files.append(path)
    summary = {"status": "complete", "stage": STAGE, "created_utc": datetime.now(timezone.utc).isoformat(),
               "availability_id": identity, "upstream_quality_id": upstream["quality_id"],
               "matches": int(d.match_id.nunique()), "events": len(d), "replayed_matches": int(events.match_id.nunique()),
               "replayed_events": len(events), "ready_matches": int(audit.ready_for_resource_features.sum()),
               "ready_events": int(audit.loc[audit.ready_for_resource_features, "source_events"].sum()),
               "review_matches": int(audit.status.eq("review_required").sum()),
               "quarantined_matches": int(audit.status.eq("quarantined").sum()),
               "example_matches": {k: int(v.match_id.iloc[0]) for k,v in example_frames.items()},
               "models_trained": 0, "config": config,
               "checks": {"all_source_events_accounted_for": len(event_coverage) == len(d),
                          "replayed_source_order_unchanged": True, "current_pair_partition": True,
                          "nonnegative_bounded_nominal_capacity": True, "whole_match_resource_gate": True,
                          "stage03_cohort_unchanged": int(d.match_id.nunique()) == upstream["matches"] and len(d) == upstream["events"]},
               "limitations": ["Unrecorded injury and fielding absences are not observable.",
                               "Review flags may be retrospective; never use them as predictors.",
                               "No penalty-time or historical match-referee restriction simulator.",
                               "Nominal capacities are not jointly feasible allocations or next-bowler choices.",
                               "No resource-quality aggregation, training or accuracy claim."]}
    if not all(summary["checks"].values()):
        raise ValueError("Stage 04 summary checks failed")
    for name, value in [("availability_summary.json", summary), ("feature_contract.json", feature_contract())]:
        path = report / name; write_json(path, value); files.append(path)
    current, _ = input_fingerprints(root)
    if current != inputs:
        raise ValueError("Inputs changed during Stage 04 export")
    write_json(output / "manifest.json", {"status": "complete", "availability_id": identity,
               "input_sha256": inputs, "output_sha256": {p.relative_to(root).as_posix(): sha256(p) for p in files}})
    print(json.dumps({k:summary[k] for k in ["matches", "events", "replayed_matches", "ready_matches", "review_matches", "quarantined_matches"]}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    run(parser.parse_args().root)
