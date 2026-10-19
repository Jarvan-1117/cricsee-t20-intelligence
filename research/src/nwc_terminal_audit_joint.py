"""Read-only terminal-case diagnostics; never fit, predict, or rewrite attribution.

Selection by observed absolute terminal WPA is descriptive, not a calibration
sample. A passed accounting check does not certify the model probability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

import data_preparation as preparation
import nwc_attribution_joint as attribution

THRESHOLD = 0.25
OUTPUT_DIR = attribution.REPORT_DIR + '/terminal_case_audit'


def compare_cases(events, ledger, meta, sources, threshold=THRESHOLD):
    """Compare saved endpoints with source facts; retain failed checks visibly.

    sources maps match ID to (decoded rows, decoded innings metadata, raw winner,
    decoder issues). Source completion metadata is retrospective audit evidence,
    never a model input. Candidate selection uses BOTH saved and recomputed WPA
    so an incorrectly stored small WPA cannot hide a large terminal movement.
    """
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be between zero and one.')
    ends = events.sort_values(attribution.KEYS[1:3] + ['source_row']).groupby(
        ['match_id', 'match_innings'], sort=False).tail(1)
    ends = ends[ends.match_innings.eq(2)].copy()
    if ledger.duplicated(attribution.KEYS).any():
        raise ValueError('Duplicate saved ledger keys.')
    indexed = ledger.set_index(attribution.KEYS)
    metadata = meta.set_index(['match_id', 'match_innings'])
    results = []
    for _, event in ends.iterrows():
        key = tuple(event[k] for k in attribution.KEYS)
        saved = indexed.loc[key]
        expected_wpa = float(event.first_batting_team_win - event.raw_model_probability)
        if max(abs(expected_wpa), abs(float(saved.interval_wpa_first_team))) < threshold:
            continue
        mid = int(event.match_id)
        rows, reports, winner, issues = sources[mid]
        raw = pd.DataFrame(rows)
        raw = raw[raw.match_innings.eq(2)].sort_values('event_sequence')
        last = raw.iloc[-1]
        endpoint = next(r for r in reports if r['match_innings'] == 2)
        m = metadata.loc[(mid, 2)]
        same = lambda a, b: bool(np.isclose(float(a), float(b), atol=1e-10, rtol=0))
        text = lambda v: '' if pd.isna(v) else str(v).strip()
        raw_pre_score = last.innings_score - last.runs_total
        raw_pre_wickets = last.innings_wickets - int(last.wicket_kind in preparation.DISMISSALS)
        raw_pre_balls = last.cumulative_legal_deliveries - last.legal_delivery
        checks = {
            'source_decode_ok': not issues,
            'final_event_ok': same(event.event_sequence, last.event_sequence)
                and len(raw) == len(events[(events.match_id == mid) & (events.match_innings == 2)])
                and same(event.source_over, last.source_over),
            'pre_state_ok': same(event.pre_score, raw_pre_score)
                and same(event.pre_wickets, raw_pre_wickets)
                and same(event.legal_balls_elapsed, raw_pre_balls),
            'delivery_ok': all(same(event[c], last[c]) for c in
                ['runs_total', 'runs_extras', 'innings_score', 'innings_wickets',
                 'legal_delivery', *preparation.EXTRAS])
                and all(text(event[c]) == text(last[c]) for c in
                    ['striker', 'non_striker', 'bowler', 'wicket_kind', 'player_out']),
            'target_ok': same(event.target_runs, endpoint['official_target'])
                and same(m.official_target, endpoint['official_target']),
            'ending_ok': bool(endpoint['audit_complete'])
                and m.audit_completion_basis == endpoint['audit_completion_basis']
                and same(m.official_total, endpoint['official_total'])
                and same(m.penalty_runs_post, endpoint['penalty_runs_post']),
            'winner_ok': winner == event.match_winner
                and int(winner == event.first_batting_team) == event.first_batting_team_win,
            'raw_probability_preserved': same(saved.raw_model_probability, event.raw_model_probability)
                and same(saved.probability_before, event.raw_model_probability),
            'terminal_endpoint_ok': bool(saved.is_terminal_event)
                and saved.interval_kind == 'audited_terminal'
                and same(saved.interval_endpoint_probability, event.first_batting_team_win),
            'wpa_arithmetic_ok': same(saved.interval_wpa_first_team, expected_wpa)
                and same(saved.terminal_first_team_wpa, expected_wpa)
                and same(saved.absolute_terminal_wpa, abs(expected_wpa)),
        }
        failed = ';'.join(k for k, ok in checks.items() if not ok)
        miscounts = json.loads(endpoint['miscounted_overs'])
        quota = sum(int(miscounts.get(str(o), {}).get('balls', 6)) for o in range(20))
        results.append(dict(match_id=mid, gender=event.gender, match_date=event.match_date,
            first_batting_team=event.first_batting_team, chasing_team=event.second_batting_team,
            arm_code=event.get('arm_code', ''), family=event.get('family', ''),
            pre_score=int(event.pre_score), pre_wickets=int(event.pre_wickets),
            target=int(event.target_runs), runs_needed=int(event.target_runs-event.pre_score),
            audited_scheduled_legal_balls_remaining=int(quota-raw_pre_balls),
            final_event_runs=int(event.runs_total), final_event_extras=int(event.runs_extras),
            final_event_exit=text(event.wicket_kind), final_legal_delivery=int(event.legal_delivery),
            post_innings_penalty=int(endpoint['penalty_runs_post']),
            official_final_score=int(endpoint['official_total']),
            completion_basis=endpoint['audit_completion_basis'], winner=winner,
            first_team_probability_pct=100*float(event.raw_model_probability),
            chasing_team_probability_pct=100*(1-float(event.raw_model_probability)),
            first_team_outcome=int(event.first_batting_team_win),
            terminal_change_pp=100*expected_wpa, absolute_change_pp=100*abs(expected_wpa),
            **checks, all_checks_pass=not failed, failed_checks=failed,
            source_issues=';'.join(sorted(issues)),
            interpretation='accounting_consistent_probability_not_certified' if not failed else 'review_failed_checks'))
    return pd.DataFrame(results).sort_values('absolute_change_pp', ascending=False).reset_index(drop=True) if results else pd.DataFrame()


def inspect(root, threshold=THRESHOLD):
    """Verify current provenance, re-read raw JSON, return diagnostics in memory."""
    root = Path(root)
    upstream = attribution.verified_output(root)
    events, meta, _ = attribution.load_inputs(root)
    attribution.validate_events(events, meta)
    ledger = pd.read_csv(root/attribution.DATA_DIR/'event_ledger.csv.gz')
    if set(map(tuple, ledger[attribution.KEYS].to_numpy())) != set(map(tuple, events[attribution.KEYS].to_numpy())):
        raise ValueError('Ledger/source event coverage mismatch.')
    ends = events.sort_values(['match_id', 'match_innings', 'source_row']).groupby(
        ['match_id', 'match_innings']).tail(1)
    ends = ends[ends.match_innings.eq(2)]
    final_keys = set(map(tuple, ends[attribution.KEYS].to_numpy()))
    flagged_keys = set(map(tuple, ledger.loc[ledger.is_terminal_event, attribution.KEYS].to_numpy()))
    if final_keys != flagged_keys:
        raise ValueError('Saved terminal flags do not identify the true final events.')
    # Read only the selected matches from the archives. The comparison also uses
    # the stored delta to detect disagreements with the re-derived selection.
    saved_delta = ledger.set_index(attribution.KEYS).interval_wpa_first_team
    sources = {}; raw_hashes = {}
    config = preparation.read_config(root)
    for gender, part in ends.groupby('gender'):
        with ZipFile(root/config['paths'][gender+'_json']) as archive:
            for _, row in part.iterrows():
                delta = row.first_batting_team_win-row.raw_model_probability
                stored = saved_delta.loc[tuple(row[k] for k in attribution.KEYS)]
                if max(abs(delta), abs(stored)) < threshold:
                    continue
                mid = int(row.match_id)
                content = archive.read(f'{mid}.json')
                doc = json.loads(content)
                rows, reports, issues = preparation.decode_states(doc)
                sources[mid] = (rows, reports, doc['info']['outcome'].get('winner'), issues)
                raw_hashes[f'{gender}/{mid}.json'] = hashlib.sha256(content).hexdigest()
    cases = compare_cases(events, ledger, meta, sources, threshold)
    ids = cases.match_id.tolist() if not cases.empty else []
    context = ledger[ledger.match_id.isin(ids) & ledger.match_innings.eq(2)].sort_values(
        ['match_id', 'source_row']).groupby('match_id', sort=False).tail(5).copy()
    context = context[['match_id', 'event_sequence', 'pre_score', 'pre_wickets',
        'legal_balls_elapsed', 'target_runs', 'striker', 'bowler', 'runs_total',
        'wicket_kind', 'raw_model_probability', 'interval_endpoint_probability',
        'interval_wpa_first_team', 'is_terminal_event']]
    summary = dict(attribution_id=upstream['attribution_id'], threshold_absolute_wpa=threshold,
        test_matches=upstream['matches'], selected_cases=len(cases),
        passed_cases=int(cases.all_checks_pass.sum()) if not cases.empty else 0,
        failed_cases=int((~cases.all_checks_pass).sum()) if not cases.empty else 0,
        training_fits=0, prediction_calls=0, attribution_changes=0,
        source_json_sha256=raw_hashes,
        audit_code_sha256=preparation.sha256(Path(__file__)),
        attribution_manifest_sha256=preparation.sha256(root/attribution.REPORT_DIR/'manifest.json'),
        limitations=[
            'Outcome-selected cases are descriptive, not an unbiased calibration sample.',
            'Passing source/accounting checks does not establish that a forecast is well calibrated.',
            'Remaining balls use retrospective source-audited over lengths for diagnosis only.',
            'No terminal clipping, reassignment, model reselection or test-driven tuning.',
        ])
    return cases, context, summary


def export(root):
    cases, context, summary = inspect(root)
    out = Path(root)/OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    cases.to_csv(out/'cases.csv', index=False)
    context.to_csv(out/'last_five_events.csv', index=False)
    summary['output_sha256'] = {name: preparation.sha256(out/name)
        for name in ['cases.csv', 'last_five_events.csv']}
    preparation.write_json(out/'summary.json', summary)
    return cases, context, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--export', action='store_true', help='Write separate diagnostics only.')
    args = parser.parse_args()
    cases, _, summary = export(args.root) if args.export else inspect(args.root)
    print(cases.to_string(index=False))
    print(json.dumps(summary, indent=2))
