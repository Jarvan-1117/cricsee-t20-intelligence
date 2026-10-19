"""Stage 09: complete interval ledger, rule-based PARTIAL player allocation.

No invented post-event predictions. An observed next-pre-event interval is not
a pure delivery effect. The first-innings last event and innings transition are
inseparable with the frozen pre-event models and remain an unallocated bundle.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import probability_evaluation_joint as evaluation
# Shared accounting implementation; the producer interface is defined below.
from nwc_attribution import (
    validate_events, allocation_rule, allocation_rules, build_ledger, summarise, team_accounting,
)
from data_preparation import DISMISSALS, EXTRAS, TEMPORARY_RETIREMENTS, rooted, sha256, write_json

STAGE = '09_nwc_attribution'
VERSION = 'resource_sums_v2_joint_selection_v1'
MODEL_POLICY = 'four_cross_group_validation_selected_from_verified_stage08'
REPORT_DIR = 'outputs/' + STAGE + '/' + VERSION
DATA_DIR = 'data/processed/nwc_attribution/' + VERSION
KEYS = ['source_row', 'match_id', 'match_innings', 'event_sequence']
DIRECT_DISMISSALS = {'bowled', 'lbw'}
SOURCE_FILES = [
    'data/processed/game_state/game_state_features.csv.gz',
    'data/processed/matches/clean_deliveries.csv.gz',
    'data/processed/matches/innings_metadata.csv',
    'data/processed/player_availability/event_availability.csv.gz',
    evaluation.DATA_DIR + '/preferred_test_predictions.csv.gz',
    evaluation.REPORT_DIR + '/manifest.json',
    evaluation.REPORT_DIR + '/nwc_handoff.json',
    'config/nwc_attribution_joint.json', 'src/nwc_attribution.py', 'src/nwc_attribution_joint.py',
    evaluation.REPORT_DIR + '/frozen_protocol.json',
    evaluation.REPORT_DIR + '/evaluation_summary.json',
    'docs/stage09_joint_selection_protocol.md',
]
LIMITATIONS = [
    'PARTIAL operational player allocation, not complete player NWC or causal player value.',
    'Next-pre-event probabilities contain observed intervening state updates, not isolated ball effects.',
    'First-innings last-event effect and innings transition are inseparable and jointly unallocated.',
    'Role changes are flagged, not excluded: their effects are not separated from event-interval changes.',
    'Ordinary no-extra/no-exit intervals and validated bowled/LBW intervals use the current-event striker/bowler convention.',
    'Other exits, extras and post-innings penalties stay in team residuals, without guessed individual responsibility.',
    'Player allocation is a bookkeeping convention, not identified responsibility or official scorebook NWC.',
    'Allocation coverage does not establish causal accuracy.',
    'Terminal WPA can be large because of genuine uncertainty or model error; it is neither clipped nor smoothed.',
    'No terminal post-state model probability is fabricated, and no causal closure-error decomposition is claimed.',
    'Frozen models inherit Stage 08 generalisation/calibration limitations; test results do not change selection.',
    'Player totals cannot be compared as complete ability rankings, especially across differing coverage.',
    'Each track uses its Stage 08 validation-selected feature group, algorithm and parameters; Stage 09 does not select models.',
    'Removing direct current-player quality does not make resource-conditioned NWC independent of historical player quality.',
]


def read_config(root):
    c = json.loads((Path(root)/'config/nwc_attribution_joint.json').read_text(encoding='utf-8'))
    expected = dict(stage=STAGE, target='first_batting_team_win', probability_timing='pre-event',
        contract_version=VERSION, main_model_policy=MODEL_POLICY,
        within_innings='observed_next_pre_event_interval',
        first_innings_boundary='unallocated_last_event_plus_innings_transition',
        terminal='audited_outcome_minus_final_pre_event_probability',
        allocation='event_interval_participant_proxy_with_explicit_team_residuals',
        role_change_policy='flag_only_not_automatic_exclusion', direct_dismissals=['bowled','lbw'],
        training_fits=0, tolerance=1e-10)
    if c != expected:
        raise ValueError('Stage 09 policy changed; review explicitly before attribution.')
    return c


def input_fingerprints(root):
    summary = evaluation.verified_output(root)
    if (summary.get('contract_version') != VERSION or summary.get('selection_policy') != evaluation.POLICY
            or set(summary.get('preferred', {})) != set(evaluation.TRACKS)
            or not set(summary['preferred'].values()).issubset(evaluation.ARM_ORDER)):
        raise ValueError('Stage 09 requires the verified cross-group Stage 08 models.')
    root = Path(root)
    return {p: sha256(root/p) for p in SOURCE_FILES}, summary


def validate_handoff(handoff, protocol, summary):
    """Reject legacy, duplicate or supplementary registries before attribution."""
    if (handoff.get('prediction_timing') != 'pre-event'
            or handoff.get('target') != 'first_batting_team_win'
            or handoff.get('probability_column') != 'raw_model_probability'):
        raise ValueError('Unsupported probability perspective/timing.')
    if (summary.get('contract_version') != VERSION or summary.get('selection_policy') != evaluation.POLICY
            or protocol.get('config') != evaluation.expected_config()
            or handoff.get('contract_version') != VERSION or handoff.get('selection_policy') != evaluation.POLICY
            or handoff.get('status') != 'raw_predictions_ready'
            or handoff.get('evaluation_id') != summary.get('evaluation_id')
            or protocol.get('evaluation_id') != summary.get('evaluation_id')
            or handoff.get('prediction_path') != evaluation.DATA_DIR + '/preferred_test_predictions.csv.gz'):
        raise ValueError('Handoff must reference the current cross-group Stage 08 evaluation and prediction path.')
    evaluation.validate_registry(protocol['models'], protocol['config'])
    preferred = evaluation.select_preferred(protocol['models'])
    if summary.get('preferred') != preferred or protocol.get('preferred') != preferred:
        raise ValueError('Frozen preferred models disagree with cross-group validation-only selection.')
    keys = ['track', 'arm_code', 'family', 'model_path', 'model_sha256', 'features']
    expected = [{k: r[k] for k in keys} for r in protocol['models'] if r['preferred']]
    models = handoff.get('preferred_models', [])
    if (len(models) != 4 or len({r['track'] for r in models}) != 4
            or sorted(models, key=lambda r: r['track']) != sorted(expected, key=lambda r: r['track'])
            or {r['track']: r['arm_code'] for r in models} != preferred):
        raise ValueError('Main registry must exactly match the four Stage 08 cross-group models.')
    return {r['track']: r for r in models}


def validate_prediction_handoff(p, registry, summary):
    if (p.empty or p.duplicated(KEYS).any() or p[KEYS].isna().any().any()
            or set(p.track) != set(registry) or len(p) != summary['test_events']
            or p.match_id.nunique() != summary['test_matches']):
        raise ValueError('Incomplete/duplicate preferred predictions.')
    if not p.preferred.eq(True).all() or 'attribution_probability' in p.columns:
        raise ValueError('Expected raw main-model predictions, not adjusted attribution values.')
    for t,r in registry.items():
        part = p[p.track.eq(t)]
        gender, _, innings = t.split('_')
        if (not part.arm_code.eq(r['arm_code']).all() or not part.family.eq(r['family']).all()
                or not part.gender.eq(gender).all() or not part.match_innings.eq(int(innings)).all()):
            raise ValueError('Predictions differ from frozen preferred registry.')


def load_inputs(root):
    root = Path(root)
    directory = root/evaluation.REPORT_DIR
    handoff = json.loads((directory/'nwc_handoff.json').read_text(encoding='utf-8'))
    protocol = json.loads((directory/'frozen_protocol.json').read_text(encoding='utf-8'))
    summary = json.loads((directory/'evaluation_summary.json').read_text(encoding='utf-8'))
    registry = validate_handoff(handoff, protocol, summary)
    p = pd.read_csv(rooted(root, handoff['prediction_path']))
    validate_prediction_handoff(p, registry, summary)
    for r in registry.values():
        if sha256(rooted(root,r['model_path'])) != r['model_sha256']:
            raise ValueError('Frozen model changed.')
    ids = set(p.match_id)
    state_cols = KEYS + ['batting_team','bowling_team','striker','non_striker','bowler',
        'first_batting_team','second_batting_team','pre_score','pre_wickets',
        'legal_balls_elapsed','source_over','target_runs','audit_pre_legal_in_over']
    event_cols = KEYS + ['runs_total','runs_extras','innings_score','innings_wickets',
        'wicket_kind','player_out','legal_delivery','cumulative_legal_deliveries','match_winner'] + list(EXTRAS)
    actor_cols = KEYS + ['striker_id','non_striker_id','bowler_id','known_uncertainty']
    d = p.copy()
    for relative, cols in [(SOURCE_FILES[0],state_cols),(SOURCE_FILES[1],event_cols),(SOURCE_FILES[3],actor_cols)]:
        part = pd.read_csv(root/relative,usecols=cols)
        part = part[part.match_id.isin(ids)]
        if len(part)!=len(p) or part.duplicated(KEYS).any():
            raise ValueError('Source cohort does not cover preferred predictions exactly.')
        d = d.merge(part,on=KEYS,validate='one_to_one',how='outer',indicator=True)
        if not d['_merge'].eq('both').all():
            raise ValueError('Source event key mismatch.')
        d = d.drop(columns='_merge')
    meta = pd.read_csv(root/SOURCE_FILES[2])
    return d, meta[meta.match_id.isin(ids)].copy(), handoff


def run(root, progress=print):
    root=Path(root).resolve(); config=read_config(root)
    inputs,upstream=input_fingerprints(root)
    data,meta,handoff=load_inputs(root)
    if len(data)!=upstream['test_events'] or data.match_id.nunique()!=upstream['test_matches']:
        raise ValueError('Stage 08 test cohort mismatch.')
    identity=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
    out,processed=root/REPORT_DIR,root/DATA_DIR
    out.mkdir(parents=True,exist_ok=True); processed.mkdir(parents=True,exist_ok=True)
    write_json(out/'manifest.json',{'status':'running','attribution_id':identity})
    progress('Building event-interval ledger with explicit player/team rules; no predictions or training.')
    ledger=build_ledger(data,meta)
    matches,players,coverage=summarise(ledger,config['tolerance'])
    teams=team_accounting(ledger,players,config['tolerance'])
    tables={processed/'event_ledger.csv.gz':ledger,
        out/'match_reconciliation.csv':matches,out/'player_match_partial_nwc.csv':players,
        out/'allocation_coverage.csv':coverage,
        out/'allocation_rules.csv':allocation_rules(),out/'team_reconciliation.csv':teams,
        out/'team_residual_events.csv':ledger[~ledger.player_allocated],
        out/'boundary_bundles.csv':ledger[ledger.interval_kind.eq('last_event_plus_innings_transition')],
        out/'terminal_events.csv':ledger[ledger.is_terminal_event]}
    for path,frame in tables.items(): frame.to_csv(path,index=False,compression='gzip' if path.suffix=='.gz' else None)
    summary=dict(stage=STAGE,contract_version=VERSION,attribution_id=identity,upstream_evaluation_id=upstream['evaluation_id'],
        created_utc=datetime.now(timezone.utc).isoformat(),matches=len(matches),events=len(ledger),
        first_events_recorded=int(ledger.is_first_event.sum()),terminal_events=int(ledger.is_terminal_event.sum()),
        boundary_bundles=int(ledger.interval_kind.eq('last_event_plus_innings_transition').sum()),
        allocated_events=int(ledger.player_allocated.sum()),unallocated_events=int((~ledger.player_allocated).sum()),
        allocation_coverage=float(ledger.player_allocated.mean()),max_closure_error=float(matches.closure_error.max()),
        maximum_absolute_terminal_wpa=float(ledger.absolute_terminal_wpa.max()),
        allocated_role_change_events=int((ledger.player_allocated & ledger.role_changed).sum()),
        max_team_closure_error=float(teams.closure_error.max()),
        training_fits=0,prediction_calls=0,player_output_status='rule_based_interval_NWC_with_team_residuals_not_causal',
        preferred=upstream['preferred'],main_model_policy=MODEL_POLICY,limitations=LIMITATIONS)
    write_json(out/'attribution_summary.json',summary)
    write_json(out/'attribution_protocol.json',{'config':config,'handoff':handoff,'limitations':LIMITATIONS})
    final_inputs,_=input_fingerprints(root)
    if inputs!=final_inputs: raise ValueError('Inputs changed; Stage 09 remains incomplete.')
    paths=list(tables)+[out/'attribution_summary.json',out/'attribution_protocol.json']
    write_json(out/'manifest.json',dict(status='complete',contract_version=VERSION,attribution_id=identity,inputs=inputs,
        output_sha256={p.relative_to(root).as_posix():sha256(p) for p in paths}))
    return summary


def verified_output(root):
    root=Path(root); path=root/REPORT_DIR/'manifest.json'
    if not path.is_file():
        raise FileNotFoundError('No Stage 09 results. Set RUN_ATTRIBUTION = True explicitly for the first run.')
    m=json.loads(path.read_text(encoding='utf-8')); inputs,upstream=input_fingerprints(root)
    if m.get('status')!='complete' or m.get('inputs')!=inputs or m.get('contract_version')!=VERSION:
        raise ValueError('Stage 09 stale/incomplete. Review and rerun explicitly; no automatic execution.')
    for relative,expected in m['output_sha256'].items():
        if sha256(rooted(root,relative))!=expected: raise ValueError('Stage 09 artifact changed: '+relative)
    summary=json.loads((root/REPORT_DIR/'attribution_summary.json').read_text(encoding='utf-8'))
    if (summary.get('contract_version') != VERSION or summary.get('preferred') != upstream['preferred']
            or summary.get('main_model_policy') != MODEL_POLICY
            or summary.get('upstream_evaluation_id') != upstream['evaluation_id']
            or summary.get('events') != upstream['test_events'] or summary.get('matches') != upstream['test_matches']):
        raise ValueError('Stage 09 summary does not match the current Stage 08 main-model cohort.')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--attribute',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.root) if args.attribute else verified_output(args.root),indent=2))
