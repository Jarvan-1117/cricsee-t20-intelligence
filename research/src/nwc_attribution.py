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

import probability_evaluation_fixed_ibw as evaluation
from data_preparation import DISMISSALS, EXTRAS, TEMPORARY_RETIREMENTS, rooted, sha256, write_json

STAGE = '09_nwc_attribution'
VERSION = 'resource_sums_v2_fixed_ibw_v1'
MODEL_POLICY = 'four_fixed_ibw_validation_selected_from_verified_stage08'
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
    'config/nwc_attribution.json', 'src/nwc_attribution.py',
    evaluation.REPORT_DIR + '/frozen_protocol.json',
    evaluation.REPORT_DIR + '/evaluation_summary.json',
    'docs/stage09_fixed_ibw_protocol.md',
]
LIMITATIONS = [
    'PARTIAL operational player allocation, not complete player NWC or causal player value.',
    'Next-pre-event probabilities contain observed intervening state updates, not isolated ball effects.',
    'First-innings last-event effect and innings transition are inseparable and jointly unallocated.',
    'Role changes are flagged, not excluded: their effects are not separated from event-interval changes.',
    'Ordinary no-extra/no-exit intervals and validated bowled/LBW intervals use the current-event striker/bowler convention.',
    'Other exits, extras and post-innings penalties stay in team residuals, without guessed individual responsibility.',
    'Player allocation is a bookkeeping convention, not identified responsibility or official scorebook NWC.',
    'Higher allocation coverage reflects a changed convention, not evidence of improved causal accuracy.',
    'Terminal WPA can be large because of genuine uncertainty or model error; it is neither clipped nor smoothed.',
    'No terminal post-state model probability is fabricated, and no causal closure-error decomposition is claimed.',
    'Frozen models inherit Stage 08 generalisation/calibration limitations; test results do not change selection.',
    'Player totals cannot be compared as complete ability rankings, especially across differing coverage.',
    'All four systems use fixed IBW with within-group validation-selected algorithms/parameters from Stage 08; Stage 09 never reselects models.',
    'Removing direct current-player quality does not make resource-conditioned NWC independent of historical player quality.',
]


def read_config(root):
    c = json.loads((Path(root)/'config/nwc_attribution.json').read_text(encoding='utf-8'))
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
            or set(summary['preferred'].values()) != {'IBW'}):
        raise ValueError('Stage 09 requires the verified fixed-IBW Stage 08 models.')
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
            or handoff.get('status') != 'requires_stage09_migration'
            or handoff.get('evaluation_id') != summary.get('evaluation_id')
            or protocol.get('evaluation_id') != summary.get('evaluation_id')
            or handoff.get('prediction_path') != evaluation.DATA_DIR + '/preferred_test_predictions.csv.gz'):
        raise ValueError('Handoff must reference the current fixed-IBW Stage 08 evaluation and prediction path.')
    evaluation.validate_registry(protocol['models'], protocol['config'])
    preferred = evaluation.select_preferred(protocol['models'])
    if summary.get('preferred') != preferred or protocol.get('preferred') != preferred:
        raise ValueError('Frozen preferred models disagree with fixed-IBW validation-only selection.')
    keys = ['track', 'arm_code', 'family', 'model_path', 'model_sha256', 'features']
    expected = [{k: r[k] for k in keys} for r in protocol['models'] if r['preferred']]
    models = handoff.get('preferred_models', [])
    if (len(models) != 4 or len({r['track'] for r in models}) != 4
            or sorted(models, key=lambda r: r['track']) != sorted(expected, key=lambda r: r['track'])
            or {r['track']: r['arm_code'] for r in models} != preferred):
        raise ValueError('Main registry must exactly match the four Stage 08 fixed-IBW models.')
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


def validate_events(d, meta):
    """Fail closed on unknown endpoints, event reordering or outcome conflicts."""
    if d.empty or d.duplicated(KEYS).any() or d[KEYS].isna().any().any():
        raise ValueError('Empty/duplicate event ledger.')
    probs = d.raw_model_probability.to_numpy(float)
    if not np.isfinite(probs).all() or ((probs<0)|(probs>1)).any():
        raise ValueError('Invalid raw probability.')
    if not d.first_batting_team_win.isin([0,1]).all():
        raise ValueError('Invalid binary outcome.')
    if d.known_uncertainty.ne(0).any():
        raise ValueError('Uncertain resource record entered frozen cohort.')
    actor_cols=['striker_id','non_striker_id','bowler_id','striker','non_striker','bowler']
    if d[actor_cols].isna().any().any() or d.striker_id.eq(d.non_striker_id).any():
        raise ValueError('Missing/invalid participant identity.')
    numeric=['pre_score','pre_wickets','runs_total','runs_extras','innings_score','innings_wickets',
             'legal_delivery','legal_balls_elapsed','cumulative_legal_deliveries'] + list(EXTRAS)
    if not np.isfinite(d[numeric].to_numpy(float)).all() or (d[numeric]<0).any().any():
        raise ValueError('Invalid event accounting value.')
    if not d.legal_delivery.isin([0,1]).all() or d.runs_extras.gt(d.runs_total).any():
        raise ValueError('Invalid legal delivery/extras accounting.')
    if not d[list(EXTRAS)].sum(axis=1).eq(d.runs_extras).all():
        raise ValueError('Extra components do not reconcile.')
    kinds=d.wicket_kind.fillna('').str.strip()
    if not kinds.isin(DISMISSALS | TEMPORARY_RETIREMENTS | {''}).all():
        raise ValueError('Unknown exit type; review instead of inventing an allocation.')
    direct=kinds.isin(DIRECT_DISMISSALS)
    if (direct & (~d.player_out.eq(d.striker) | (d.innings_wickets-d.pre_wickets).ne(1))).any():
        raise ValueError('Direct dismissal does not identify the current striker and one wicket.')
    if meta.duplicated(['match_id','match_innings']).any() or len(meta)!=2*d.match_id.nunique():
        raise ValueError('Incomplete/duplicate innings metadata.')
    metadata = meta.set_index(['match_id','match_innings'])
    for mid, match in d.groupby('match_id',sort=False):
        if set(match.match_innings)!={1,2}:
            raise ValueError('Both innings required.')
        for col in ['first_batting_team_win','first_batting_team','second_batting_team','gender','match_winner']:
            if match[col].isna().any() or match[col].nunique()!=1:
                raise ValueError('Inconsistent match metadata: '+col)
        first, second = match.first_batting_team.iloc[0], match.second_batting_team.iloc[0]
        y = int(match.first_batting_team_win.iloc[0])
        if first==second or match.match_winner.iloc[0] != (first if y else second):
            raise ValueError('Outcome/perspective mismatch.')
        for inn, part in match.groupby('match_innings',sort=False):
            part = part.sort_values('source_row',kind='stable')
            if list(part.event_sequence)!=list(range(1,len(part)+1)):
                raise ValueError('Event order must follow source rows, not display coordinates.')
            if not part.batting_team.eq(first if inn==1 else second).all() or not part.bowling_team.eq(second if inn==1 else first).all():
                raise ValueError('Team orientation mismatch.')
            m = metadata.loc[(mid,inn)]
            if not bool(m.audit_complete) or m.events!=len(part) or m.batting_team!=part.batting_team.iloc[0]:
                raise ValueError('Innings metadata/coverage mismatch.')
            if not np.array_equal(part.innings_score.to_numpy(),part.pre_score.to_numpy()+part.runs_total.to_numpy()):
                raise ValueError('Score update mismatch.')
            if part.pre_wickets.iloc[0]!=0 or part.legal_balls_elapsed.iloc[0]!=0 or not (part.innings_wickets-part.pre_wickets).isin([0,1]).all():
                raise ValueError('Invalid first-event/wicket state.')
            if not np.array_equal(part.legal_balls_elapsed.to_numpy()+part.legal_delivery.to_numpy(),part.cumulative_legal_deliveries.to_numpy()):
                raise ValueError('Legal-ball update mismatch.')
            if len(part)>1 and (not np.array_equal(part.pre_score.iloc[1:].to_numpy(),part.innings_score.iloc[:-1].to_numpy()) or not np.array_equal(part.pre_wickets.iloc[1:].to_numpy(),part.innings_wickets.iloc[:-1].to_numpy())):
                raise ValueError('Pre/post event continuity mismatch.')
            if len(part)>1 and not np.array_equal(part.legal_balls_elapsed.iloc[1:].to_numpy(),part.cumulative_legal_deliveries.iloc[:-1].to_numpy()):
                raise ValueError('Legal-ball continuity mismatch.')
            last = part.iloc[-1]
            if part.pre_score.iloc[0]!=m.penalty_runs_pre or last.innings_score+m.penalty_runs_post!=m.official_total or last.innings_wickets!=m.final_wickets or last.cumulative_legal_deliveries!=m.actual_legal_balls:
                raise ValueError('Official endpoint does not reconcile.')
            basis = m.audit_completion_basis
            if basis not in {'target_reached','ten_dismissals','scheduled_overs_complete','no_available_batting_pair'}:
                raise ValueError('Unverified innings endpoint.')
            if basis=='ten_dismissals' and last.innings_wickets!=10:
                raise ValueError('False all-out endpoint.')
            if basis=='no_available_batting_pair' and not bool(m.audit_no_batting_pair):
                raise ValueError('Unverified batting exhaustion.')
            if basis=='scheduled_overs_complete':
                miscounts=json.loads(m.miscounted_overs)
                expected=miscounts.get('19',{}).get('balls',6)
                if last.source_over!=19 or last.audit_pre_legal_in_over+last.legal_delivery!=int(expected):
                    raise ValueError('Incomplete scheduled final over.')
            if inn==2:
                if not part.target_runs.eq(m.official_target).all() or m.official_target!=metadata.loc[(mid,1)].official_total+1:
                    raise ValueError('Target mismatch.')
                if part.pre_score.ge(m.official_target).any():
                    raise ValueError('Event after chase completed.')
                if m.official_total==m.official_target-1:
                    raise ValueError('Tied regulation result cannot close to binary outcome.')
                if int(m.official_total<m.official_target)!=y or (basis=='target_reached')!=(m.official_total>=m.official_target):
                    raise ValueError('Terminal result disagrees with audited score.')


def allocation_rule(row, boundary, post_penalty):
    """Priority-ordered project conventions, NOT a cricket-law causal metric."""
    if boundary:
        return 'team_first_innings_boundary'
    if post_penalty:
        return 'team_post_innings_penalty'
    kind='' if pd.isna(row.wicket_kind) else str(row.wicket_kind).strip()
    if row.penalty:
        return 'team_penalty_event'
    if row.runs_extras and kind:
        return 'team_mixed_extras_and_exit'
    if row.runs_extras:
        if row.wides or row.noballs:
            return 'team_wide_or_no_ball_interval'
        return 'team_bye_or_legbye_interval'
    if kind in DIRECT_DISMISSALS:
        return 'player_direct_dismissal_proxy'
    if kind=='run out':
        return 'team_run_out_shared_responsibility'
    if kind in {'caught','caught and bowled','stumped'}:
        return 'team_fielding_exit_not_decomposed'
    if kind:
        return 'team_other_exit_or_retirement'
    return 'player_ordinary_interval_proxy'


def allocation_rules():
    return pd.DataFrame([
        ('first_innings_boundary','team','First-innings last event plus transition; endpoint unidentified.'),
        ('post_innings_penalty','team','Terminal/boundary interval includes a post-innings award.'),
        ('penalty_event','team','Do not infer individual responsibility from a team penalty.'),
        ('extras_and_exit','team','Do not divide a mixed interval by runs or guess responsibility.'),
        ('wide_or_no_ball','team','Whole interval retained at team level; no automatic striker credit.'),
        ('bye_or_legbye','team','No individual batting/bowling split inferred.'),
        ('bowled_or_lbw_without_extras','player_pair','Current striker and bowler receive opposite interval values; dismissed striker must match source.'),
        ('run_out','team','Do not force a non-striker or fielding event onto the striker/bowler pair.'),
        ('caught_caught_and_bowled_stumped','team','Fielding/exit interval not individually decomposed by this protocol.'),
        ('other_exit_or_retirement','team','No personal retirement/responsibility inference.'),
        ('ordinary_without_extras_or_exit','player_pair','Current-event striker/bowler proxy, including dot balls and ordinary strike changes.'),
        ('role_changes','flag_only','Record strike swaps, batting-pair changes and next bowler changes; do not claim isolated effects.'),
    ],columns=['case','destination','convention'])


def build_ledger(events, meta):
    validate_events(events,meta)
    d=events.sort_values(['match_id','match_innings','source_row'],kind='stable').reset_index(drop=True).copy()
    groups=d.groupby(['match_id','match_innings'],sort=False)
    last=groups.cumcount(ascending=False).eq(0)
    d['is_first_event']=groups.cumcount().eq(0)
    d['is_terminal_event']=last & d.match_innings.eq(2)
    boundary=last & d.match_innings.eq(1)
    d['probability_before']=d.raw_model_probability
    d['interval_endpoint_probability']=groups.raw_model_probability.shift(-1)
    starts=d[d.match_innings.eq(2)&d.is_first_event].set_index('match_id').raw_model_probability
    d.loc[boundary,'interval_endpoint_probability']=d.loc[boundary,'match_id'].map(starts)
    d.loc[d.is_terminal_event,'interval_endpoint_probability']=d.loc[d.is_terminal_event,'first_batting_team_win']
    d['interval_kind']='next_pre_event'
    d.loc[boundary,'interval_kind']='last_event_plus_innings_transition'
    d.loc[d.is_terminal_event,'interval_kind']='audited_terminal'
    d['interval_wpa_first_team']=d.interval_endpoint_probability-d.probability_before
    # The boundary endpoint is NOT the first-innings post-event probability.
    d['event_interval_wpa_first_team']=d.interval_wpa_first_team.where(~boundary)
    d['boundary_bundle_first_team']=d.interval_wpa_first_team.where(boundary,0.)
    reasons=[]; changes=[]
    next_roles={c:groups[c].shift(-1) for c in ['striker_id','non_striker_id','bowler_id']}
    post_penalties=meta.set_index(['match_id','match_innings']).penalty_runs_post
    for i,row in d.iterrows():
        flags=[]
        if not last.iloc[i]:
            current_pair={row.striker_id,row.non_striker_id}
            following_pair={next_roles['striker_id'].iloc[i],next_roles['non_striker_id'].iloc[i]}
            if current_pair!=following_pair:
                flags.append('batting_pair_changed')
            elif row.striker_id!=next_roles['striker_id'].iloc[i]:
                flags.append('strike_swapped')
            if row.bowler_id!=next_roles['bowler_id'].iloc[i]: flags.append('bowler_changed')
        changes.append(';'.join(flags) if flags else ('no_next_event' if last.iloc[i] else 'unchanged'))
        reasons.append(allocation_rule(row,bool(boundary.iloc[i]),
            bool(last.iloc[i] and post_penalties.loc[(row.match_id,row.match_innings)]!=0)))
    d['allocation_reason']=reasons
    d['role_change_flags']=changes
    d['role_changed']=~d.role_change_flags.isin(['unchanged','no_next_event'])
    d['player_allocated']=d.allocation_reason.str.startswith('player_')
    d['allocation_scope']=np.where(d.player_allocated,'player_pair','team_residual')
    d['allocated_first_team_wpa']=d.interval_wpa_first_team.where(d.player_allocated,0.)
    d['unallocated_first_team_wpa']=d.interval_wpa_first_team.where(~d.player_allocated,0.)
    sign=np.where(d.match_innings.eq(1),1.,-1.)
    # NaN means not allocated, not an observed zero player contribution.
    d['striker_partial_nwc']=(d.interval_wpa_first_team*sign).where(d.player_allocated)
    d['bowler_partial_nwc']=-d.striker_partial_nwc
    d['terminal_first_team_wpa']=d.interval_wpa_first_team.where(d.is_terminal_event,0.)
    d['absolute_terminal_wpa']=d.terminal_first_team_wpa.abs()
    d['partial_nwc_excluding_terminal']=d.striker_partial_nwc.where(~d.is_terminal_event,0.)
    return d


def summarise(d, tolerance=1e-10):
    if not np.isfinite(d.interval_wpa_first_team).all() or not np.allclose(
            d.interval_wpa_first_team,d.allocated_first_team_wpa+d.unallocated_first_team_wpa,atol=tolerance,rtol=0):
        raise ValueError('Event allocation accounting failed.')
    if not np.allclose((d.striker_partial_nwc+d.bowler_partial_nwc)[d.player_allocated],0,atol=tolerance,rtol=0):
        raise ValueError('Player pair accounting failed.')
    if d.loc[~d.player_allocated,['striker_partial_nwc','bowler_partial_nwc']].notna().any().any():
        raise ValueError('Unallocated player value must remain missing.')
    matches=[]
    for mid,g in d.groupby('match_id',sort=False):
        expected=float(g.first_batting_team_win.iloc[0]-g.probability_before.iloc[0])
        total=float(g.interval_wpa_first_team.sum())
        allocated=float(g.allocated_first_team_wpa.sum()); unallocated=float(g.unallocated_first_team_wpa.sum())
        error=max(abs(total-expected),abs(total-allocated-unallocated))
        if error>tolerance or g.is_terminal_event.sum()!=1 or g.is_first_event.sum()!=2:
            raise ValueError('Match closure or endpoint coverage failed.')
        matches.append(dict(match_id=mid,gender=g.gender.iloc[0],match_date=g.match_date.iloc[0],
            events=len(g),allocated_events=int(g.player_allocated.sum()),unallocated_events=int((~g.player_allocated).sum()),
            start_probability=float(g.probability_before.iloc[0]),outcome=int(g.first_batting_team_win.iloc[0]),
            probability_change=expected,allocated_first_team_wpa=allocated,unallocated_first_team_wpa=unallocated,
            boundary_bundle=float(g.boundary_bundle_first_team.sum()),terminal_wpa=float(g.terminal_first_team_wpa.sum()),
            closure_error=error))
    rows=[]
    for role,idcol,namecol,teamcol,value in [('batting','striker_id','striker','batting_team','striker_partial_nwc'),
                                           ('bowling','bowler_id','bowler','bowling_team','bowler_partial_nwc')]:
        for _,r in d.iterrows():
            rows.append(dict(match_id=r.match_id,gender=r.gender,team=r[teamcol],player_id=r[idcol],player=r[namecol],role=role,
                role_events=1,allocated_events=int(r.player_allocated),unallocated_events=int(not r.player_allocated),
                partial_nwc=r[value],terminal_partial_nwc=r[value] if r.is_terminal_event and r.player_allocated else 0.,
                role_change_allocated_events=int(r.player_allocated and r.role_changed),
                ordinary_proxy_events=int(r.allocation_reason=='player_ordinary_interval_proxy'),
                direct_dismissal_proxy_events=int(r.allocation_reason=='player_direct_dismissal_proxy'),
                terminal_allocated_events=int(r.player_allocated and r.is_terminal_event)))
    roles=pd.DataFrame(rows)
    players=roles.groupby(['match_id','gender','team','player_id','player','role'],as_index=False).agg(
        role_events=('role_events','sum'),allocated_events=('allocated_events','sum'),unallocated_events=('unallocated_events','sum'),
        partial_nwc=('partial_nwc',lambda s:s.sum(min_count=1)),terminal_partial_nwc=('terminal_partial_nwc','sum'),
        role_change_allocated_events=('role_change_allocated_events','sum'),
        ordinary_proxy_events=('ordinary_proxy_events','sum'),direct_dismissal_proxy_events=('direct_dismissal_proxy_events','sum'),
        terminal_allocated_events=('terminal_allocated_events','sum'))
    players['allocation_coverage']=players.allocated_events/players.role_events
    players['partial_nwc_excluding_terminal']=players.partial_nwc-players.terminal_partial_nwc
    if not np.allclose(players.loc[players.partial_nwc.notna(),'partial_nwc'],
            players.loc[players.partial_nwc.notna(),'partial_nwc_excluding_terminal']+
            players.loc[players.partial_nwc.notna(),'terminal_partial_nwc'],atol=tolerance,rtol=0):
        raise ValueError('Non-terminal + terminal player decomposition failed.')
    if not np.allclose(players.groupby('match_id').partial_nwc.sum(),0,atol=tolerance,rtol=0):
        raise ValueError('Allocated batting/bowling pair sum failed.')
    coverage=d.groupby(['track','allocation_reason'],as_index=False).agg(events=('source_row','size'),
        first_team_wpa=('interval_wpa_first_team','sum'),absolute_wpa=('interval_wpa_first_team',lambda s:s.abs().sum()))
    coverage['track_event_fraction']=coverage.events/coverage.groupby('track').events.transform('sum')
    return pd.DataFrame(matches),players,coverage


def team_accounting(d, players, tolerance=1e-10):
    """Reconcile each team's actual player totals PLUS team residual, not just paired zero-sum."""
    rows=[]
    sums=players.groupby(['match_id','team']).partial_nwc.sum()
    for mid,g in d.groupby('match_id',sort=False):
        for team,sign in [(g.first_batting_team.iloc[0],1.),(g.second_batting_team.iloc[0],-1.)]:
            player_sum=float(sums.get((mid,team),0.))
            expected_player_sum=float(sign*g.allocated_first_team_wpa.sum())
            residual=float(sign*g.unallocated_first_team_wpa.sum())
            expected=float(sign*(g.first_batting_team_win.iloc[0]-g.probability_before.iloc[0]))
            error=max(abs(player_sum-expected_player_sum),abs(player_sum+residual-expected))
            if error>tolerance: raise ValueError('Team/player accounting failed.')
            rows.append(dict(match_id=mid,team=team,player_interval_total=player_sum,
                team_residual=residual,expected_probability_change=expected,closure_error=error))
    return pd.DataFrame(rows)


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
