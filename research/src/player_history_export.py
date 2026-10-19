"""Stage 10: verified, read-only-source player-match history export.

No training, prediction, model selection or NWC allocation is performed here.
JSON is the product contract; two CSV projections support notebook inspection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile
from zipfile import ZipFile

import pandas as pd

import nwc_attribution_joint as attribution
from data_preparation import DISMISSALS, EXTRAS, TEMPORARY_RETIREMENTS, rooted, sha256

REPORT_DIR = 'outputs/10_results_and_visualisation/player_history_v1'
SCHEMA_VERSION = 2
KEYS = ['source_row', 'match_id', 'match_innings', 'event_sequence']
REGISTRY = 'data/processed/historical_features/participant_registry.csv'
LINEUPS = 'data/processed/player_availability/initial_lineups.csv'
CLEAN = 'data/processed/matches/clean_deliveries.csv.gz'
INNINGS = 'data/processed/matches/innings_metadata.csv'
LEDGER = attribution.DATA_DIR + '/event_ledger.csv.gz'
PLAYER_NWC = attribution.REPORT_DIR + '/player_match_partial_nwc.csv'
TEAM_NWC = attribution.REPORT_DIR + '/team_reconciliation.csv'
CONTRIBUTIONS = 'data/processed/historical_features/player_match_contributions.csv.gz'
MANIFESTS = ['data/processed/historical_features/player_quality_manifest.json',
             'data/processed/player_availability/manifest.json']
SOURCES = [REGISTRY, LINEUPS, CLEAN, INNINGS, LEDGER, PLAYER_NWC, TEAM_NWC,
           CONTRIBUTIONS, *MANIFESTS, attribution.REPORT_DIR + '/manifest.json',
           'data/raw/cricsheet/t20s_female_json.zip',
           'data/raw/cricsheet/t20s_male_json.zip', 'src/player_history_export.py',
           'docs/player_history_export.md']
BOWLER_WICKETS = {'bowled', 'caught', 'caught and bowled', 'lbw', 'stumped', 'hit wicket'}
DEFINITIONS = {
    'scope': 'Current Stage 09 test matches only, not complete career histories.',
    'balls_faced': 'Recorded non-wide events, including no-balls; source-event convention.',
    'fours_sixes': 'Batter runs of 4/6 only when source runs.non_boundary is false or absent.',
    'runs_conceded': 'Batter runs + wides + no-ball extras; excludes byes, leg-byes and penalties.',
    'bowler_wickets': 'bowled, caught, caught and bowled, lbw, stumped, hit wicket.',
    'dot_balls': 'Legal deliveries with zero total event runs; byes/leg-byes are not dots.',
    'overs_display': 'Six-ball equivalent from actual legal balls, not umpire over slots; never decimal arithmetic.',
    'rates': 'strike_rate=100*runs/balls_faced; economy=6*runs_conceded/legal_balls; null on zero denominator.',
    'partial_nwc': 'Stage 09 allocated intervals only, in probability units, positive for player team; UI multiplies by 100 for percentage points.',
    'coverage': 'allocated_events/role_events from Stage 09; batting role_events count striker events, not all non-striker participation.',
    'missing': 'No allocation is null, not zero. Observed non-striker-only batting has zero role_events and null coverage.',
    'participation': 'Initial XI or observed batting/bowling establishes match participation; catalogue-only is not proof of playing.',
    'dismissal': 'Final recorded exit kind, not out, did not bat, or null if participation is unconfirmed; all exit records retained.',
}


def read_table(root, relative):
    return pd.read_csv(root / relative, dtype={'match_id': str, 'player_id': str,
        'striker_id': str, 'non_striker_id': str, 'bowler_id': str})


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                               indent=2) + '\n', encoding='utf-8')


def team_id(name):
    return 'team_' + hashlib.sha256(name.encode('utf-8')).hexdigest()[:16]


def fingerprints(root):
    summary = attribution.verified_output(root)
    hashes = {name: sha256(rooted(root, name)) for name in SOURCES}
    # Extra Stage 03/04 inputs must agree with their published provenance too.
    for name in MANIFESTS:
        manifest = json.loads((root / name).read_text(encoding='utf-8'))
        if manifest.get('status') != 'complete':
            raise ValueError('Incomplete upstream manifest: ' + name)
        for key in ['input_sha256', 'output_sha256']:
            for relative, expected in manifest[key].items():
                if relative in hashes and hashes[relative] != expected:
                    raise ValueError('Upstream artifact mismatch: ' + relative)
    return hashes, summary


def unique(frame, keys, label):
    if frame.empty or frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError('Empty, missing or duplicate keys: ' + label)


def source_events(doc, match_id):
    """Explicit source-order keys; no join on repeated display ball coordinates."""
    info = doc['info']
    if info['balls_per_over'] != 6 or len(doc['innings']) != 2:
        raise ValueError('Unsupported match format: ' + match_id)
    rows = []
    for inn, innings in enumerate(doc['innings'], 1):
        sequence = 0
        for over in innings['overs']:
            for delivery in over['deliveries']:
                sequence += 1
                wickets = delivery.get('wickets', [])
                if len(wickets) > 1:
                    raise ValueError('Multiple exits require an explicit schema mapping: ' + match_id)
                rows.append(dict(match_id=match_id, match_innings=inn,
                    event_sequence=sequence, source_over=over['over'],
                    batting_team=innings['team'], striker=delivery['batter'],
                    non_striker=delivery['non_striker'], bowler=delivery['bowler'],
                    runs_batter=delivery['runs']['batter'], runs_total=delivery['runs']['total'],
                    runs_extras=delivery['runs']['extras'],
                    player_out=wickets[0]['player_out'] if wickets else '',
                    wicket_kind=wickets[0]['kind'] if wickets else '',
                    non_boundary=bool(delivery['runs'].get('non_boundary', False)),
                    **{extra: delivery.get('extras', {}).get(extra, 0) for extra in EXTRAS}))
    return pd.DataFrame(rows)


def attach_source_flags(clean, documents):
    raw = pd.concat([source_events(doc, mid) for mid, doc in documents.items()], ignore_index=True)
    keys = ['match_id', 'match_innings', 'event_sequence']
    unique(clean, keys, 'clean events'); unique(raw, keys, 'source events')
    joined = clean.merge(raw, on=keys, how='outer', suffixes=('', '_json'),
                         indicator=True, validate='one_to_one')
    if not joined['_merge'].eq('both').all():
        raise ValueError('Source JSON and clean event coverage differ')
    for col in raw.columns.difference(keys + ['non_boundary']):
        if not joined[col].fillna('').eq(joined[col + '_json'].fillna('')).all():
            raise ValueError('Source JSON and clean events disagree: ' + col)
    return joined[list(clean.columns) + ['non_boundary']]


def load_inputs(root, summary):
    ledger = read_table(root, LEDGER)
    unique(ledger, KEYS, 'ledger')
    ids = set(ledger.match_id)
    if len(ids) != summary['matches'] or len(ledger) != summary['events']:
        raise ValueError('Stage 09 cohort mismatch')
    frames = {}
    for label, relative in [('clean', CLEAN), ('registry', REGISTRY), ('lineups', LINEUPS),
                            ('innings', INNINGS), ('nwc', PLAYER_NWC), ('teams', TEAM_NWC),
                            ('contributions', CONTRIBUTIONS)]:
        data = read_table(root, relative)
        frames[label] = data[data.match_id.isin(ids)].copy()
    if set(frames['nwc'].match_id) != ids or set(frames['registry'].match_id) != ids:
        raise ValueError('Incomplete player history cohort')
    with ExitStack() as stack:
        archives = {g: stack.enter_context(ZipFile(root / f'data/raw/cricsheet/t20s_{g}_json.zip'))
                    for g in ['male', 'female']}
        documents = {}
        for mid, events in ledger.groupby('match_id'):
            doc = json.loads(archives[events.gender.iloc[0]].read(mid + '.json'))
            if (doc['info']['gender'] != events.gender.iloc[0]
                    or doc['info']['dates'][0] != events.match_date.iloc[0]):
                raise ValueError('Match metadata mismatch: ' + mid)
            documents[mid] = doc
    frames['clean'] = attach_source_flags(frames['clean'], documents)
    # Compare common fields as well as keys before supplementing the ledger.
    clean = frames['clean']
    unique(clean, KEYS, 'clean events')
    common = [c for c in clean.columns if c in ledger.columns and c not in KEYS]
    merged = ledger.merge(clean[KEYS + common + ['runs_batter', 'non_boundary']],
        on=KEYS, suffixes=('', '_clean'), how='outer', indicator=True, validate='one_to_one')
    if not merged['_merge'].eq('both').all():
        raise ValueError('Ledger and clean events differ')
    for col in common:
        if not merged[col].fillna('').eq(merged[col + '_clean'].fillna('')).all():
            raise ValueError('Ledger and clean event field mismatch: ' + col)
    events = merged[list(ledger.columns) + ['runs_batter', 'non_boundary']]
    return events.sort_values(['match_id', 'match_innings', 'event_sequence']), frames, documents


def empty_stats():
    return dict(batted=False, bowled=False, runs=0, balls_faced=0, fours=0, sixes=0,
        dismissals=0, exits=[], legal_balls_bowled=0, runs_conceded=0, wickets=0,
        dot_balls=0, last_batting_event=None, role_batting=[], role_bowling=[])


def aggregate_stats(events, registry):
    unique(registry, ['match_id', 'player_id'], 'participant registry')
    unique(registry, ['match_id', 'player_name'], 'participant names within match')
    lookup = {(r.match_id, r.player_name): (r.player_id, r.team) for r in registry.itertuples()}
    stats = {(r.match_id, r.player_id): empty_stats() for r in registry.itertuples()}
    for e in events.itertuples():
        for role, team in [('striker', e.batting_team), ('non_striker', e.batting_team), ('bowler', e.bowling_team)]:
            if lookup.get((e.match_id, getattr(e, role))) != (getattr(e, role + '_id'), team):
                raise ValueError('Actor registry mismatch: ' + role)
        bat = stats[(e.match_id, e.striker_id)]
        bowl = stats[(e.match_id, e.bowler_id)]
        bat['batted'] = True
        stats[(e.match_id, e.non_striker_id)]['batted'] = True
        for pid in [e.striker_id, e.non_striker_id]:
            stats[(e.match_id, pid)]['last_batting_event'] = (int(e.match_innings), int(e.event_sequence))
        bowl['bowled'] = True
        bat['runs'] += int(e.runs_batter)
        bat['balls_faced'] += int(e.wides == 0)
        bat['fours'] += int(e.runs_batter == 4 and not e.non_boundary)
        bat['sixes'] += int(e.runs_batter == 6 and not e.non_boundary)
        bowl['legal_balls_bowled'] += int(e.legal_delivery)
        bowl['runs_conceded'] += int(e.runs_batter + e.wides + e.noballs)
        bowl['dot_balls'] += int(e.legal_delivery == 1 and e.runs_total == 0)
        kind = '' if pd.isna(e.wicket_kind) else e.wicket_kind
        if kind not in DISMISSALS | TEMPORARY_RETIREMENTS | {''}:
            raise ValueError('Unknown dismissal: ' + kind)
        bowl['wickets'] += int(kind in BOWLER_WICKETS)
        if kind:
            out = lookup.get((e.match_id, e.player_out))
            if out is None or out[1] != e.batting_team:
                raise ValueError('Unresolved dismissed player')
            player = stats[(e.match_id, out[0])]
            player['batted'] = True
            player['dismissals'] += int(kind in DISMISSALS)
            player['exits'].append(dict(kind=kind, innings=int(e.match_innings),
                                       event_sequence=int(e.event_sequence)))
        for target, role, value in [(bat, 'batting', e.striker_partial_nwc),
                                    (bowl, 'bowling', e.bowler_partial_nwc)]:
            if bool(e.player_allocated) != pd.notna(value):
                raise ValueError('Allocation flag and value disagree')
            target['role_' + role].append((value, bool(e.is_terminal_event)))
    return stats


def nwc_record(source, observations, participated):
    """Copy published Stage 09 values after an independent ledger cross-check."""
    count = len(observations)
    values = [float(v) for v, _ in observations if pd.notna(v)]
    if count and source is None:
        raise ValueError('Missing Stage 09 player role')
    if source is not None:
        if (int(source['role_events']) != count or int(source['allocated_events']) != len(values)
                or int(source['unallocated_events']) != count - len(values)):
            raise ValueError('Role event counts disagree with Stage 09')
        expected = math.fsum(values) if values else None
        actual = source['partial_nwc']
        if expected is None:
            if pd.notna(actual):
                raise ValueError('Unallocated contribution must be null')
        elif not math.isclose(float(actual), expected, abs_tol=1e-10, rel_tol=0):
            raise ValueError('Player NWC differs from ledger')
        terminal = math.fsum(float(v) for v, end in observations if end and pd.notna(v))
        if not math.isclose(float(source['terminal_partial_nwc']), terminal, abs_tol=1e-10, rel_tol=0):
            raise ValueError('Terminal NWC differs from ledger')
        if not count or not math.isclose(float(source['allocation_coverage']), len(values)/count, abs_tol=1e-10):
            raise ValueError('Invalid allocation coverage')
    if values:
        status, reason = 'available', 'allocated_intervals_only'
    elif participated is True:
        status, reason = 'unallocated', 'no_allocated_intervals' if count else 'no_striker_events'
    elif participated is False:
        status, reason = 'not_participated', 'did_not_participate_in_role'
    else:
        status, reason = 'unavailable', 'catalogue_listing_only'
    return dict(status=status, reason=reason, partial_nwc=float(source['partial_nwc']) if values else None,
        terminal_partial_nwc=float(source['terminal_partial_nwc']) if values else None,
        role_events=count, allocated_events=len(values), unallocated_events=count-len(values),
        allocation_coverage=len(values)/count if count else None)


def build_history(events, frames, documents):
    registry, nwc = frames['registry'], frames['nwc']
    unique(nwc, ['match_id', 'player_id', 'role'], 'NWC roles')
    unique(frames['lineups'], ['match_id', 'player_id'], 'initial XI')
    stats = aggregate_stats(events, registry)
    source_nwc = {(r['match_id'], r['player_id'], r['role']): r for r in nwc.to_dict('records')}
    initial = {(r.match_id, r.player_id) for r in frames['lineups'].itertuples()}
    registry_map = {(r.match_id, r.player_id): r for r in registry.itertuples()}
    if not initial.issubset(registry_map):
        raise ValueError('Initial XI not in participant registry')
    for row in frames['lineups'].itertuples():
        ref = registry_map[(row.match_id, row.player_id)]
        if row.team != ref.team or row.gender != ref.gender:
            raise ValueError('Initial XI identity mismatch')
    for (mid, pid, role), row in source_nwc.items():
        ref = registry_map.get((mid, pid))
        if ref is None or role not in {'batting', 'bowling'} or row['team'] != ref.team or row['gender'] != ref.gender:
            raise ValueError('NWC participant mismatch')
    rows = []
    for r in registry.itertuples():
        key = (r.match_id, r.player_id)
        s = stats[key]; info = documents[r.match_id]['info']
        if r.gender != info['gender'] or r.match_date != info['dates'][0]:
            raise ValueError('Registry match metadata mismatch')
        if info['registry']['people'].get(r.player_name) != r.player_id or r.player_name not in info['players'].get(r.team, []):
            raise ValueError('Registry and source roster mismatch')
        observed = s['batted'] or s['bowled']
        confirmed = key in initial or observed
        batted = s['batted'] if confirmed else None
        bowled = s['bowled'] if confirmed else None
        opponent, = [t for t in info['teams'] if t != r.team]
        batting_keys = ['runs', 'balls_faced', 'fours', 'sixes', 'dismissals']
        bowling_keys = ['legal_balls_bowled', 'runs_conceded', 'wickets', 'dot_balls']
        batting = {k: s[k] if batted else None for k in batting_keys}
        bowling = {k: s[k] if bowled else None for k in bowling_keys}
        last_exit = s['exits'][-1] if s['exits'] else None
        returned = (last_exit is not None and last_exit['kind'] in TEMPORARY_RETIREMENTS
                    and s['last_batting_event'] is not None
                    and s['last_batting_event'] > (last_exit['innings'], last_exit['event_sequence']))
        batting.update(strike_rate=100*s['runs']/s['balls_faced'] if s['balls_faced'] else None,
            dismissal=last_exit['kind'] if last_exit and not returned else ('not out' if batted else 'did not bat' if batted is False else None),
            exits=s['exits'], nwc=nwc_record(source_nwc.get((*key, 'batting')), s['role_batting'], batted))
        balls = s['legal_balls_bowled']
        bowling.update(overs_display=f'{balls//6}.{balls%6}' if bowled else None,
            economy=6*s['runs_conceded']/balls if balls else None,
            nwc=nwc_record(source_nwc.get((*key, 'bowling')), s['role_bowling'], bowled))
        rows.append(dict(match_id=r.match_id, date=r.match_date, gender=r.gender,
            player_id=r.player_id, player_name=r.player_name, team_id=team_id(r.team), team_name=r.team,
            opponent_id=team_id(opponent), opponent_name=opponent,
            match_participation='initial_xi' if key in initial else 'observed' if observed else 'listed_only',
            batted=batted, bowled=bowled, batting=batting, bowling=bowling,
            stats_status='available' if confirmed else 'unavailable', missing_fields=[] if confirmed else ['role_participation']))
    rows.sort(key=lambda r: (r['date'], r['match_id'], r['gender'], r['player_id']))
    return rows, stats


def verify_totals(rows, stats, frames, events):
    """Check all players against an independently generated Stage 03 table and 09 teams."""
    unique(frames['contributions'], ['match_id', 'player_id'], 'Stage 03 statistics')
    mapping = {'runs': 'batting_runs', 'balls_faced': 'balls_faced', 'dismissals': 'batting_dismissals',
               'legal_balls_bowled': 'legal_balls_bowled', 'runs_conceded': 'bowling_runs_conceded',
               'wickets': 'bowler_wickets', 'dot_balls': 'bowling_dot_balls'}
    if set(stats) != set(zip(frames['contributions'].match_id, frames['contributions'].player_id)):
        raise ValueError('Stage 03 player coverage mismatch')
    for old in frames['contributions'].itertuples():
        s = stats[(old.match_id, old.player_id)]
        for new, source in mapping.items():
            if s[new] != getattr(old, source):
                raise ValueError(f'Traditional statistic mismatch: {old.match_id}/{old.player_id}/{new}')
    grouped = defaultdict(list)
    for r in rows:
        for role in ['batting', 'bowling']:
            value = r[role]['nwc']['partial_nwc']
            if value is not None:
                grouped[(r['match_id'], r['team_name'])].append(value)
    unique(frames['teams'], ['match_id', 'team'], 'team reconciliation')
    if len(frames['teams']) != 2*events.match_id.nunique():
        raise ValueError('Missing team reconciliation')
    errors = []
    for t in frames['teams'].itertuples():
        total = math.fsum(grouped[(t.match_id, t.team)])
        error = max(abs(total-t.player_interval_total),
                    abs(total+t.team_residual-t.expected_probability_change))
        if error > 1e-10:
            raise ValueError('Team NWC reconciliation failed')
        errors.append(error)
    for meta in frames['innings'].itertuples():
        part = events[events.match_id.eq(meta.match_id) & events.match_innings.eq(meta.match_innings)]
        if (len(part) != meta.events or int(part.legal_delivery.sum()) != meta.actual_legal_balls
                or int(part.runs_total.sum()) + meta.penalty_runs_pre + meta.penalty_runs_post != meta.official_total):
            raise ValueError('Innings accounting mismatch')
    return dict(status='passed', player_match_rows=len(rows), verified_statistics=list(mapping),
        verified_nwc_roles=len(frames['nwc']), verified_team_rows=len(errors),
        max_team_closure_error=max(errors), non_boundary_scoring_events=int(events.non_boundary.sum()),
        innings_checked=len(frames['innings']), event_rows=len(events),
        source_json_alignment='all events matched by match/innings/sequence and values',
        training_fits=0, prediction_calls=0)


def role_table(rows, role):
    records = []
    for r in rows:
        common = {k: v for k, v in r.items() if k not in {'batting', 'bowling', 'missing_fields'}}
        block = r[role]
        records.append({**common, **{k: v for k, v in block.items() if k not in {'nwc', 'exits'}},
                        **{'nwc_' + k: v for k, v in block['nwc'].items()}})
    return pd.DataFrame(records)


def product_payload(rows, documents, version, upstream):
    people = defaultdict(list)
    for r in rows:
        people[(r['gender'], r['player_id'])].append(r)
    players = [dict(player_id=pid, gender=g, name=rs[-1]['player_name'],
        aliases=sorted({r['player_name'] for r in rs}), team_ids=sorted({r['team_id'] for r in rs}),
        listed_match_count=len(rs), confirmed_match_count=sum(r['match_participation'] != 'listed_only' for r in rs))
        for (g, pid), rs in sorted(people.items())]
    teams = sorted({t for d in documents.values() for t in d['info']['teams']})
    matches = [dict(match_id=mid, date=d['info']['dates'][0], gender=d['info']['gender'],
        venue=d['info'].get('venue'), team_ids=[team_id(t) for t in d['info']['teams']],
        winner_team_id=team_id(d['info']['outcome']['winner'])) for mid, d in documents.items()]
    matches.sort(key=lambda m: (m['date'], m['match_id']))
    counts = {}
    for gender in ['female', 'male']:
        part = [m for m in matches if m['gender'] == gender]
        counts[gender] = dict(matches=len(part), players=sum(p['gender'] == gender for p in players),
            player_match_rows=sum(r['gender'] == gender for r in rows),
            date_from=min(m['date'] for m in part) if part else None,
            date_to=max(m['date'] for m in part) if part else None)
    return dict(schema_version=SCHEMA_VERSION, dataset_version=version, scope='test_matches',
        scope_label='Current recorded test matches; not complete player careers',
        upstream_attribution_id=upstream['attribution_id'], upstream_evaluation_id=upstream['upstream_evaluation_id'],
        counts_by_gender=counts, stats_definitions=DEFINITIONS, players=players,
        teams=[dict(team_id=team_id(t), name=t) for t in teams], matches=matches, rows=rows)


def verify_package(directory, expected_version):
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete' or manifest.get('dataset_version') != expected_version:
        raise ValueError('Incomplete or wrong product package')
    required = {'history.json', 'batting_history.csv', 'bowling_history.csv', 'verification.json'}
    if set(manifest['output_sha256']) != required:
        raise ValueError('Incomplete product file list')
    for name, value in manifest['output_sha256'].items():
        if sha256(rooted(directory, name)) != value:
            raise ValueError('Product artifact changed: ' + name)
    return manifest


def publish(root, payload, checks, inputs):
    base = rooted(root, REPORT_DIR)
    snapshots = base/'snapshots'; snapshots.mkdir(parents=True, exist_ok=True)
    version = payload['dataset_version']
    # Version is computed by this module, never accepted as an arbitrary path.
    if len(version) != 64 or any(c not in '0123456789abcdef' for c in version):
        raise ValueError('Invalid dataset version')
    target = snapshots/version
    if target.exists():
        manifest = verify_package(target, version)
        if manifest['inputs'] != inputs:
            raise ValueError('Existing package inputs differ')
    else:
        with tempfile.TemporaryDirectory(prefix='.staging-', dir=snapshots) as tmp:
            staging = Path(tmp)/'package'; staging.mkdir()
            dump_json(staging/'history.json', payload)
            for role in ['batting', 'bowling']:
                role_table(payload['rows'], role).to_csv(staging/f'{role}_history.csv', index=False)
            dump_json(staging/'verification.json', checks)
            names = ['history.json', 'batting_history.csv', 'bowling_history.csv', 'verification.json']
            dump_json(staging/'manifest.json', dict(status='complete', schema_version=SCHEMA_VERSION,
                dataset_version=version, generated_at=datetime.now(timezone.utc).isoformat(), inputs=inputs,
                output_sha256={name: sha256(staging/name) for name in names}))
            verify_package(staging, version)
            staging.rename(target)
    # One small atomic pointer update; a failed export cannot publish a partial package.
    with tempfile.TemporaryDirectory(prefix='.pointer-', dir=base) as tmp:
        pointer = Path(tmp)/'current.json'
        dump_json(pointer, dict(dataset_version=version,
            package='snapshots/' + version, manifest_sha256=sha256(target/'manifest.json')))
        pointer.replace(base/'current.json')
    return target


def run(root, progress=print):
    root = Path(root).resolve()
    progress('Verifying Stage 09 and source identities; no training or prediction.')
    inputs, upstream = fingerprints(root)
    version = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    events, frames, documents = load_inputs(root, upstream)
    progress('Building chronological player histories and checking every player/match.')
    rows, stats = build_history(events, frames, documents)
    checks = verify_totals(rows, stats, frames, events)
    payload = product_payload(rows, documents, version, upstream)
    # Validate strict JSON before publishing or changing current.json.
    json.dumps(payload, allow_nan=False)
    final_inputs, _ = fingerprints(root)
    if inputs != final_inputs:
        raise ValueError('Inputs changed during export; no package published')
    directory = publish(root, payload, checks, inputs)
    progress('Published ' + str(directory))
    return dict(dataset_version=version, package=str(directory), matches=len(documents),
                players=len(payload['players']), **checks)


def verified_output(root):
    root = Path(root).resolve(); base = rooted(root, REPORT_DIR)
    pointer = json.loads((base/'current.json').read_text(encoding='utf-8'))
    version = pointer['dataset_version']
    if (len(version) != 64 or any(c not in '0123456789abcdef' for c in version)
            or pointer['package'] != 'snapshots/' + version):
        raise ValueError('Invalid package pointer')
    directory = rooted(base, pointer['package'])
    if sha256(directory/'manifest.json') != pointer['manifest_sha256']:
        raise ValueError('Package manifest changed')
    manifest = verify_package(directory, version)
    inputs, _ = fingerprints(root)
    if manifest['inputs'] != inputs:
        raise ValueError('Export sources changed; run Stage 10 explicitly')
    return directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--export', action='store_true', help='Explicitly build and publish history data')
    args = parser.parse_args()
    print(json.dumps(run(args.root) if args.export else {'verified_package': str(verified_output(args.root))}, indent=2))
