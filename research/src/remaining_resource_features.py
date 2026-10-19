"""Stage 05 v2: never-entered batting sums and remaining-capacity bowling sums.

No training, splitting, future participation inference, or upstream cohort edits.
Historical Stage 06-07 consumers must be migrated explicitly.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import player_availability as availability
import player_quality_features as quality
from data_preparation import rooted, sha256, write_json

STAGE = '05_remaining_resource_features'
VERSION = 'resource_sums_v2'
DATA_DIR = 'data/processed/remaining_resources'
REPORT_DIR = 'outputs/' + STAGE
KEYS = availability.KEYS
BAT_METRICS = quality.BATTING_METRICS
BOWL_METRICS = quality.BOWLING_METRICS
METRICS = BAT_METRICS + BOWL_METRICS
BAT_FEATURES = ['waiting_resource_' + m for m in BAT_METRICS]
BOWL_FEATURES = ['bowling_resource_' + m for m in BOWL_METRICS]
CAPACITY_DIAGNOSTICS = ['audit_current_bowler_capacity', 'audit_bowlers_with_capacity', 'audit_total_remaining_overs']
BAT_DETAIL_SCHEMA = ['player_id', 'status', 'previously_observed_batting', 'included'] + ['smoothed_' + m for m in BAT_METRICS] + BAT_FEATURES
BOWL_DETAIL_SCHEMA = ['player_id', 'status', 'nominal_capacity', 'remaining_overs'] + ['smoothed_' + m for m in BOWL_METRICS] + BOWL_FEATURES
# Names retained only so archived consumers can be imported. V2 does not export
# or allowlist these legacy features, nor run their old allocation experiments.
CAPACITY_FEATURES = ['resource_current_bowler_capacity', 'resource_bowlers_with_capacity', 'resource_allocation_shortfall_fraction']
ALT_FEATURES = ['capacity_weighted_resource_' + m for m in BOWL_METRICS]
STATIC_FEATURES = ['initial_XI_resource_' + m for m in METRICS]
INTERACTIONS = ['progress_interaction_' + name for name in BAT_FEATURES]
ALLOCATION_SCHEMA = ['player_id', 'nominal_capacity', 'prior_usage_weight', 'reserved_current_over', 'usage_allocated_balls', 'capacity_allocated_balls']


def read_config(root):
    c = json.loads((Path(root) / 'config/remaining_resources.json').read_text(encoding='utf-8'))
    fixed = {
        'stage': STAGE, 'contract_version': VERSION,
        'cohort_policy': 'all whole matches ready in Stage 04; reviews retained separately',
        'prediction_timing': 'pre-event; no future appearances or allocations',
        'quality_policy': 'Stage 03 prior-date smoothed metrics; separate dimensions; no centering',
        'batting_policy': 'sum available never-entered batters only; exclude current pair and previously entered players',
        'bowling_policy': 'sum Stage 04 remaining nominal balls / 6 times bowling quality; include current bowler',
        'direct_quality_policy': 'exclude striker, non-striker and bowler direct quality and evidence',
        'capacity_policy': 'individual upper bounds, not an allocated innings budget; no historical-use weights',
        'zero_pool_policy': 'empty batting pool or zero eligible bowling capacities produce zero sums',
    }
    if set(c) != set(fixed) | {'tolerance'} or any(c.get(k) != v for k, v in fixed.items()):
        raise ValueError('Unsupported Stage 05 v2 policy; change implementation and tests explicitly')
    if not np.isfinite(c['tolerance']) or not 0 < c['tolerance'] <= 1e-6:
        raise ValueError('Invalid reconciliation tolerance')
    return c


def match_profiles(profiles, lineups, config):
    """Use smoothed prior-date values, never a predicted future workload."""
    if profiles.empty or profiles.player_id.duplicated().any() or lineups.player_id.duplicated().any():
        raise ValueError('Empty or duplicate player identity in profiles/initial XI')
    if any(profiles[c].nunique() != 1 for c in ['gender', 'match_id', 'match_date']):
        raise ValueError('Profile match/gender/date inconsistent')
    history_date = pd.to_datetime(profiles.history_last_date)
    if not (history_date.isna() | history_date.lt(pd.to_datetime(profiles.match_date))).all():
        raise ValueError('Player quality must use strictly earlier dates')
    if lineups.groupby('team').size().tolist() != [11, 11]:
        raise ValueError('Resource scope requires two initial XIs')
    values = profiles[['smoothed_' + m for m in METRICS]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Smoothed quality must be finite and nonnegative')
    rates = [i for i, m in enumerate(METRICS) if m.endswith('_rate') and m != 'batting_strike_rate']
    if (values[:, rates] > 1).any():
        raise ValueError('Quality rate outside [0, 1]')
    counts = profiles[['prior_balls_faced', 'prior_legal_balls_bowled']].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('Invalid historical evidence counts')
    lookup = {r.player_id: {'team': r.team, 'quality': values[i].copy(),
                          'no_batting_history': int(r.prior_balls_faced == 0),
                          'no_bowling_history': int(r.prior_legal_balls_bowled == 0)}
              for i, r in enumerate(profiles.itertuples(index=False))}
    for r in lineups.itertuples(index=False):
        if r.player_id not in lookup or lookup[r.player_id]['team'] != r.team:
            raise ValueError('Initial-XI identity/quality join failed')
    return lookup


def event_resources(row, lookup, config):
    bat, bowl = json.loads(row.batting_state_json), json.loads(row.bowling_state_json)
    for states, team, width in [(bat, row.batting_team, len(availability.BAT_SCHEMA)), (bowl, row.bowling_team, len(availability.BOWL_SCHEMA))]:
        if not states or any(len(s) != width for s in states) or len({s[0] for s in states}) != len(states):
            raise ValueError('Player-state schema or uniqueness failed')
        if any(s[0] not in lookup or lookup[s[0]]['team'] != team for s in states):
            raise ValueError('State-to-quality ID join failed; do not treat this as cold start')
    if int(row.known_uncertainty):
        raise ValueError('Uncertain event escaped whole-match resource gate')
    pair = {row.striker_id, row.non_striker_id}
    if len(pair) != 2 or {s[0] for s in bat if s[2] == 'current_batter'} != pair:
        raise ValueError('Current-pair identity mismatch')
    for s in bat:
        if s[2] not in {'current_batter', 'waiting_not_yet_batted', 'waiting_returned', 'dismissed', 'temporarily_unavailable', 'replaced_out'} or s[3] not in (0, 1):
            raise ValueError('Unknown batting status or entered-history flag')
        if (s[2] == 'waiting_not_yet_batted' and s[3] != 0) or (s[2] == 'waiting_returned' and s[3] != 1):
            raise ValueError('Waiting status disagrees with prior entry evidence')
    # Stage 04 general waiting includes returned players. Current pair members
    # can have prior flag zero on entry but are already on the field.
    general = [s for s in bat if s[2] in {'waiting_not_yet_batted', 'waiting_returned'}]
    if len(general) != row.waiting_batter_count or row.available_batter_count != len(general) + 2:
        raise ValueError('Batting partition disagrees with Stage 04')
    waiting = {s[0] for s in bat if s[2] == 'waiting_not_yet_batted' and s[3] == 0}
    bat_details = []
    for pid, _, status, seen in bat:
        q = np.asarray(lookup[pid]['quality'][:len(BAT_METRICS)], dtype=float)
        included = int(pid in waiting)
        bat_details.append([pid, status, seen, included, *q.tolist(), *(q * included).tolist()])
    bat_resource = np.array([d[-len(BAT_METRICS):] for d in bat_details]).sum(axis=0)
    ids = [s[0] for s in bowl]
    current = [i for i, s in enumerate(bowl) if s[2] == 'current_bowler']
    if len(current) != 1 or ids[current[0]] != row.bowler_id:
        raise ValueError('Current-bowler identity mismatch')
    caps = np.array([s[-1] for s in bowl], dtype=float)
    if not np.isfinite(caps).all() or (caps < 0).any() or (caps > 24).any() or not np.equal(caps, np.floor(caps)).all():
        raise ValueError('Invalid Stage 04 nominal capacity')
    for s, cap in zip(bowl, caps):
        if s[2] not in {'current_bowler', 'inferred_available', 'quota_exhausted', 'temporarily_unavailable', 'replaced_out', 'suspended'}:
            raise ValueError('Unknown bowling status')
        if s[2] not in {'current_bowler', 'inferred_available'} and cap != 0:
            raise ValueError('Unavailable or exhausted player has positive capacity')
    if caps[current[0]] != row.current_bowler_nominal_capacity or int((caps > 0).sum()) != row.bowlers_with_nominal_capacity:
        raise ValueError('Bowling capacity diagnostics disagree with Stage 04')
    overs = caps / 6.
    bowl_details = []
    for s, weight in zip(bowl, overs):
        q = np.asarray(lookup[s[0]]['quality'][len(BAT_METRICS):], dtype=float)
        bowl_details.append([s[0], s[2], float(s[-1]), float(weight), *q.tolist(), *(q * weight).tolist()])
    bowl_resource = np.array([d[-len(BOWL_METRICS):] for d in bowl_details]).sum(axis=0)
    out = {k: int(getattr(row, k)) for k in KEYS}
    out.update(zip(BAT_FEATURES, bat_resource))
    out.update(zip(BOWL_FEATURES, bowl_resource))
    total = float(overs.sum())
    out.update({'audit_waiting_batter_count': len(waiting),
        'audit_excluded_returned_count': sum(s[2] == 'waiting_returned' for s in bat),
        'audit_nominal_budget': float(row.nominal_scheduled_slots_remaining),
        'audit_zero_nominal_budget': int(row.nominal_scheduled_slots_remaining == 0),
        'audit_current_bowler_capacity': float(caps[current[0]]),
        'audit_bowlers_with_capacity': int((caps > 0).sum()), 'audit_total_remaining_overs': total,
        'audit_waiting_players_without_history': sum(lookup[p]['no_batting_history'] for p in waiting),
        'audit_capacity_fraction_without_history': sum(w * lookup[p]['no_bowling_history'] for p, w in zip(ids, overs)) / total if total else 0.})
    details = {k: out[k] for k in KEYS}
    details['batting_resource_json'] = availability.compact(bat_details)
    details['bowling_resource_json'] = availability.compact(bowl_details)
    out.update(reconcile_details(out, details, config['tolerance']))
    return out, details


def reconcile_details(values, details, tolerance):
    """Independently multiply and sum serialized per-player contributions."""
    errors = {}
    for side, schema, features, metrics in [('batting', BAT_DETAIL_SCHEMA, BAT_FEATURES, BAT_METRICS), ('bowling', BOWL_DETAIL_SCHEMA, BOWL_FEATURES, BOWL_METRICS)]:
        records = json.loads(details[side + '_resource_json'])
        if not records or any(len(r) != len(schema) for r in records):
            raise ValueError('Invalid resource detail schema')
        data = np.array([r[4:] for r in records], dtype=float)
        n = len(metrics)
        weights = np.array([r[3] for r in records], dtype=float)
        expected_weights = np.array([int(r[1] == 'waiting_not_yet_batted' and r[2] == 0) if side == 'batting' else r[2] / 6. for r in records])
        contributions = data[:, :n] * expected_weights[:, None]
        actual = np.array([values[f] for f in features])
        error = max(float(np.max(np.abs(weights - expected_weights))), float(np.max(np.abs(contributions - data[:, n:]))), float(np.max(np.abs(contributions.sum(axis=0) - actual))))
        if not np.isfinite(data).all() or not np.isfinite(actual).all() or not np.isfinite(error) or error > tolerance:
            raise ValueError(f'{side} resource contribution reconciliation failed')
        errors['audit_' + side + '_sum_error'] = error
    return errors


def feature_contract(base_contract):
    base = base_contract['base_numeric'] + base_contract['base_categorical']
    forbidden = quality.feature_contract()['current_player_block']
    arms = {'constant': [], 'state': base, 'batting_resources': base + BAT_FEATURES,
            'bowling_resources': base + BOWL_FEATURES, 'both_resources': base + BAT_FEATURES + BOWL_FEATURES}
    for fields in arms.values():
        if len(fields) != len(set(fields)) or set(fields) & set(forbidden):
            raise ValueError('Direct quality/evidence or duplicate field in resource allowlist')
    return {
        'contract_version': VERSION, 'target': base_contract['target'], 'prediction_timing': 'pre-event',
        'perspective': 'First-batting-team win in both innings; resources describe current batting and bowling teams.',
        'main_arms': arms, 'auxiliary_arms': {},
        'arm_codes': {'R0': 'constant', 'R1': 'state', 'RB': 'batting_resources', 'RW': 'bowling_resources', 'RBW': 'both_resources'},
        'legacy_arm_policy': 'New identifiers; old F/E arms are historical. Stage 06-08 migration required.',
        'batting_features': BAT_FEATURES, 'bowling_features': BOWL_FEATURES,
        'new_availability_features': [], 'audit_only_capacity_fields': CAPACITY_DIAGNOSTICS,
        'excluded_direct_quality_fields': forbidden,
        'optional_background': {k: base_contract[k] for k in ['optional_team_identity', 'optional_recent_events']},
        'background_rule': 'Same frozen state/background for nonconstant arms. No extra availability arm or allocation shortfall.',
        'quality': 'Uncentered Stage 03 prior-date smoothed metrics; separate units, not a composite.',
        'batting': 'Sum available never-entered batters only. Current pair and all previously entered players excluded.',
        'bowling': 'Sum individual nominal capacity / 6 times each bowling metric; includes current bowler. No allocation/mean.',
        'resource_units': {**{f: 'sum of player ' + m for f, m in zip(BAT_FEATURES, BAT_METRICS)}, **{f: 'remaining overs times player ' + m for f, m in zip(BOWL_FEATURES, BOWL_METRICS)}},
        'direction': 'Economy and batting dot rate are lower-is-better quality metrics. No sign flip. Sums also depend on pool size/capacity.',
        'zero_meaning': 'Empty pool or zero eligible capacity yields zero. Zero on one metric does not prove no players remain.',
        'batting_detail_schema': BAT_DETAIL_SCHEMA, 'bowling_detail_schema': BOWL_DETAIL_SCHEMA,
        'capacity_scope': 'All Stage 04 inferred-available bowlers, not future actual bowlers. Individual upper bounds may exceed innings budget in total.',
        'cohort': 'Identical Stage 04 ready whole matches for all arms. No fits or splits here.',
        'historical_scope': 'Unchanged Stage 03 prior-date histories from retained Stage 01 cohort.',
        'forbidden_predictors': 'Direct current quality/evidence, audits/reviews, IDs, other outcomes, state JSON, future participation. Never all numeric columns.',
        'independence_limit': 'Current-bowler quality remains indirectly in resources. Not statistical independence or causal NWC.',
        'implementation_choices': ['Exclude non-striker direct quality/evidence too.', 'Retain quality-weighted current-bowler remaining capacity.'],
        'evaluation': 'Exploratory follow-up on inspected test period; validation-only selection. No performance conclusion at Stage 05.',
    }


def input_fingerprints(root):
    root = Path(root)
    upstream = availability.verified_output(root)
    paths = ['config/remaining_resources.json', 'src/remaining_resource_features.py', 'data/processed/player_availability/manifest.json',
             'data/processed/historical_features/player_match_profiles.csv.gz', 'data/processed/historical_features/current_player_features.csv.gz', 'outputs/02_game_state_features/feature_contract.json']
    return {p: sha256(root / p) for p in paths}, upstream


def verified_output(root):
    root = Path(root)
    path = root / DATA_DIR / 'manifest.json'
    if not path.exists():
        raise FileNotFoundError('No Stage 05 export. Set RUN_STAGE = True explicitly.')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    inputs, _ = input_fingerprints(root)
    if manifest.get('status') != 'complete' or manifest.get('input_sha256') != inputs or manifest.get('contract_version') != VERSION:
        raise ValueError('Stage 05 output stale/incomplete or legacy. Set RUN_STAGE = True; no automatic rebuild.')
    for relative, expected in manifest['output_sha256'].items():
        if sha256(rooted(root, relative)) != expected:
            raise ValueError(f'Stage 05 artifact changed: {relative}')
    contract = json.loads((root / REPORT_DIR / 'feature_contract.json').read_text(encoding='utf-8'))
    base = json.loads((root / 'outputs/02_game_state_features/feature_contract.json').read_text(encoding='utf-8'))
    if contract != feature_contract(base):
        raise ValueError('Stage 05 feature contract mismatch')
    return json.loads((root / REPORT_DIR / 'resource_summary.json').read_text(encoding='utf-8'))


def run(root):
    root = Path(root).resolve()
    config = read_config(root)
    inputs, upstream = input_fingerprints(root)
    audit = pd.read_csv(root / availability.REPORT_DIR / 'match_availability_audit.csv')
    requested = list(audit.loc[audit.ready_for_resource_features.eq(True), 'match_id'])
    events = availability.load_resource_events(root, requested)
    profiles = pd.read_csv(root / 'data/processed/historical_features/player_match_profiles.csv.gz', low_memory=False)
    lineups = pd.read_csv(root / availability.DATA_DIR / 'initial_lineups.csv')
    current = pd.read_csv(root / 'data/processed/historical_features/current_player_features.csv.gz', low_memory=False)
    current = current.loc[current.match_id.isin(requested)].reset_index(drop=True)
    pd.testing.assert_frame_equal(events[KEYS].reset_index(drop=True), current[KEYS])
    for name in ['striker_id', 'non_striker_id', 'bowler_id']:
        np.testing.assert_array_equal(events[name], current[name])
    np.testing.assert_array_equal(events.nominal_scheduled_slots_remaining, current.scheduled_balls_remaining_estimate)
    contract = feature_contract(json.loads((root / 'outputs/02_game_state_features/feature_contract.json').read_text(encoding='utf-8')))
    forbidden = contract['excluded_direct_quality_fields']
    if set(forbidden) - set(current):
        raise ValueError('Stage 03 direct quality schema differs; review before export')
    inherited = current.drop(columns=forbidden)
    output, report = root / DATA_DIR, root / REPORT_DIR
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    write_json(output / 'manifest.json', {'status': 'writing', 'resource_id': identity, 'contract_version': VERSION})
    pg = {int(mid): g for mid, g in profiles.groupby('match_id', sort=False)}
    lg = {int(mid): g for mid, g in lineups.groupby('match_id', sort=False)}
    all_rows, example_details = [], None
    example_id = int(events.match_id.iloc[0])
    detail_path = output / 'resource_contributions.csv.gz'
    with gzip.open(detail_path, 'wt', encoding='utf-8', newline='') as handle:
        for index, (mid, frame) in enumerate(events.groupby('match_id', sort=False), 1):
            if not frame.gender.eq(pg[int(mid)].gender.iloc[0]).all() or not frame.match_date.eq(pg[int(mid)].match_date.iloc[0]).all():
                raise ValueError('Quality snapshot date/gender differs from event')
            lookup = match_profiles(pg[int(mid)], lg[int(mid)], config)
            rows, details = [], []
            for row in frame.itertuples(index=False):
                values, detail = event_resources(row, lookup, config)
                rows.append(values)
                details.append(detail)
            all_rows.extend(rows)
            local_details = pd.DataFrame(details)
            local_details.to_csv(handle, index=False, header=index == 1)
            if int(mid) == example_id:
                example_details = local_details
            if index % 200 == 0:
                print(f'Constructed resource sums: {index}/{len(requested)} matches', flush=True)
    resources = pd.DataFrame(all_rows)
    pd.testing.assert_frame_equal(resources[KEYS], inherited[KEYS])
    additions = [c for c in resources if c not in KEYS]
    if set(additions) & set(inherited):
        raise ValueError('Resource export would overwrite inherited fields')
    if not np.isfinite(resources[additions].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite resource output; no row dropping allowed')
    model_input = pd.concat([inherited, resources[additions]], axis=1)
    pd.testing.assert_frame_equal(model_input[inherited.columns], inherited)
    if set(forbidden) & set(model_input):
        raise ValueError('Direct quality/evidence leaked into model input')
    for name, columns in contract['main_arms'].items():
        if len(set(columns)) != len(columns) or set(columns) - set(model_input):
            raise ValueError(f'Invalid feature allowlist: {name}')
    cohort = audit.copy()
    cohort['included_in_resource_sample'] = cohort.match_id.isin(requested)
    coverage = cohort.groupby(['gender', 'included_in_resource_sample'], as_index=False).agg(matches=('match_id', 'size'), events=('source_events', 'sum'))
    feature_summary = resources[BAT_FEATURES + BOWL_FEATURES].describe().T.reset_index(names='feature')
    checks = resources.groupby('match_id', as_index=False).agg(events=('source_row', 'size'), max_batting_sum_error=('audit_batting_sum_error', 'max'), max_bowling_sum_error=('audit_bowling_sum_error', 'max'), zero_nominal_budget_events=('audit_zero_nominal_budget', 'sum'))
    files = [detail_path]
    for path, table in [(output / 'resource_event_features.csv.gz', resources), (output / 'model_input.csv.gz', model_input), (output / 'cohort.csv', cohort),
                       (report / 'coverage.csv', coverage), (report / 'resource_checks.csv', checks), (report / 'resource_feature_summary.csv', feature_summary),
                       (report / 'review_matches.csv', cohort.loc[~cohort.included_in_resource_sample]),
                       (report / 'resource_preview.csv', model_input[KEYS + BAT_FEATURES + BOWL_FEATURES].head(20)),
                       (report / 'example_match.csv.gz', model_input.loc[model_input.match_id.eq(example_id)]), (report / 'example_contributions.csv.gz', example_details)]:
        table.to_csv(path, index=False)
        files.append(path)
    summary = {'status': 'complete', 'stage': STAGE, 'contract_version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'resource_id': identity, 'upstream_availability_id': upstream['availability_id'], 'upstream_matches': upstream['matches'], 'upstream_events': upstream['events'],
        'matches': len(requested), 'events': len(resources), 'review_or_quarantined_matches': int((~cohort.included_in_resource_sample).sum()),
        'matches_by_gender': cohort.loc[cohort.included_in_resource_sample].groupby('gender').size().to_dict(), 'example_match_id': example_id,
        'models_trained': 0, 'splits_created': 0, 'config': config,
        'checks': {'same_whole_match_cohort_for_all_arms': True, 'stage03_source_files_unchanged': True, 'direct_quality_and_evidence_removed_from_model_input': True,
                   'stage04_source_order_and_actor_ids_match': True, 'finite_resources': True, 'per_player_sums_reconciled': True, 'no_usage_allocation_or_centering': True},
        'reconciliation': {'max_batting_sum_error': float(checks.max_batting_sum_error.max()), 'max_bowling_sum_error': float(checks.max_bowling_sum_error.max()), 'zero_nominal_budget_events': int(resources.audit_zero_nominal_budget.sum())},
        'limitations': ['Record-based availability upper bounds, not full medical/legal eligibility.', 'Separate quality metrics, not expected runs or a composite score.',
                        'Individual bowling capacities may exceed innings budget in total; no future allocation is predicted.',
                        'Current-bowler quality remains indirectly; no independence or causal NWC claim.',
                        'Stage 06-08 migration required; old F/E results do not evaluate these features.', 'Inspected test period; no improvement established here.']}
    for name, value in [('resource_summary.json', summary), ('feature_contract.json', contract)]:
        path = report / name
        write_json(path, value)
        files.append(path)
    final_inputs, _ = input_fingerprints(root)
    if final_inputs != inputs:
        raise ValueError('Inputs changed during resource export')
    write_json(output / 'manifest.json', {'status': 'complete', 'resource_id': identity, 'contract_version': VERSION, 'input_sha256': inputs,
        'output_sha256': {p.relative_to(root).as_posix(): sha256(p) for p in files}})
    print(json.dumps({k: summary[k] for k in ['matches', 'events', 'reconciliation', 'models_trained']}, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    run(parser.parse_args().root)
