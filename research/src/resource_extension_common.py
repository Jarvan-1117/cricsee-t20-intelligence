"""Shared Stage 07 v2 engine: verified development-only controls, no promotion."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import model_comparison as base
import remaining_resource_features as resources
from data_preparation import rooted, sha256, write_json

VERSION = 'resource_sums_v2'
TRACKS = ['female_innings_1', 'female_innings_2', 'male_innings_1', 'male_innings_2']
PHASES = ['powerplay', 'middle', 'death']
MODES = ['fixed_baseline_parameters', 'within_family_selected', 'selected_system']


def input_fingerprints(root, ext):
    root = Path(root)
    upstream = base.verified_output(root)
    if upstream.get('contract_version') != VERSION:
        raise ValueError('Expected verified resource_sums_v2 Stage 06.')
    paths = [ext.CONFIG_PATH, ext.SOURCE_PATH, 'src/resource_extension_common.py',
             'src/model_comparison.py', 'src/remaining_resource_features.py',
             base.REPORT_DIR + '/manifest.json', base.DATA_DIR + '/match_splits.csv',
             resources.REPORT_DIR + '/feature_contract.json'] + ext.EXTRA_INPUT_PATHS
    return {p: sha256(root / p) for p in paths}, upstream


def validate_development(data, columns, ext):
    arms = base.resolve_arms(resources.feature_contract(base.game_state.feature_contract()))
    base.validate_development(data, arms)
    if columns != ext.resolve_columns(arms):
        raise ValueError('Extension inputs differ from exact allowlist; no direct quality/audit fields.')
    ext.validate_features(data)
    for innings in [1, 2]:
        numeric = [c for c in base.track_columns(columns, innings) if c != 'innings_phase']
        if not np.isfinite(data.loc[data.match_innings.eq(innings), numeric].to_numpy(dtype=float)).all():
            raise ValueError('Nonfinite extension predictors.')


def join_extra(data, extra, columns):
    if extra.duplicated(base.KEYS).any() or len(extra) != len(data):
        raise ValueError('Extension events differ from development cohort.')
    if set(columns) & set(data.columns):
        raise ValueError('Extension columns already present.')
    result = data.merge(extra[base.KEYS + columns], on=base.KEYS, how='left', sort=False, validate='one_to_one')
    pd.testing.assert_frame_equal(result[data.columns].reset_index(drop=True), data.reset_index(drop=True))
    if result[columns].isna().any().any():
        raise ValueError('Missing extension event/feature.')
    return result


def baseline_tables(root, data, config, references):
    out = Path(root) / base.REPORT_DIR
    winners = pd.read_csv(out / 'family_winners.csv')
    selected = pd.read_csv(out / 'selected_models.csv')
    winners = winners[winners.arm.isin(references)].copy()
    selected = selected[selected.arm.isin(references)].copy()
    if (len(winners) != 4 * len(references) * len(base.FAMILIES)
            or winners.duplicated(['track', 'arm', 'family']).any()
            or len(selected) != 4 * len(references) or selected.duplicated(['track', 'arm']).any()):
        raise ValueError('Incomplete/duplicate Stage 06 references.')
    for track in TRACKS:
        for arm in references:
            group = winners[winners.track.eq(track) & winners.arm.eq(arm)]
            if set(group.family) != set(base.FAMILIES):
                raise ValueError('Missing reference algorithm.')
            for row in group.itertuples(index=False):
                if json.loads(row.parameters) != config['candidate_grid'][row.family][int(row.candidate)]:
                    raise ValueError('Baseline parameters differ from inherited grid.')
            chosen = selected[selected.track.eq(track) & selected.arm.eq(arm)]
            best = base.best_record(group.to_dict('records'))
            if len(chosen) != 1 or chosen.iloc[0].family != best['family'] or int(chosen.iloc[0].candidate) != best['candidate']:
                raise ValueError('Baseline selection differs from validation rule.')
    parts = []
    for chunk in pd.read_csv(Path(root) / base.DATA_DIR / 'validation_predictions.csv.gz', chunksize=100000):
        parts.append(chunk[chunk.arm.isin(references)].copy())
    predictions = pd.concat(parts, ignore_index=True)
    if predictions.duplicated(['track', 'arm', 'family'] + base.KEYS).any():
        raise ValueError('Duplicate baseline predictions.')
    if set(predictions.track) != set(TRACKS) or set(predictions.family) != set(base.FAMILIES):
        raise ValueError('Unexpected baseline prediction track/family.')
    val = data[data.split.eq('validation')]
    for row in winners.itertuples(index=False):
        gender, _, number = row.track.split('_')
        expected = val[val.gender.eq(gender) & val.match_innings.eq(int(number))].set_index(base.KEYS).sort_index()
        group = predictions[predictions.track.eq(row.track) & predictions.arm.eq(row.arm)
                            & predictions.family.eq(row.family)].set_index(base.KEYS).sort_index()
        if not group.index.equals(expected.index) or not group[[base.TARGET, 'match_date', 'gender']].equals(expected[[base.TARGET, 'match_date', 'gender']]):
            raise ValueError('Baseline validation coverage/labels differ.')
        p = group.raw_model_probability.to_numpy()
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError('Invalid baseline probability.')
        for key, value in base.metrics(group[base.TARGET], p, group.index.get_level_values('match_id'), config['probability_epsilon']).items():
            if value is not None and not np.isclose(value, getattr(row, key), atol=1e-12, rtol=1e-10):
                raise ValueError('Baseline metrics disagree with saved predictions.')
    return winners, selected, predictions.merge(val[base.KEYS + ['innings_phase']], on=base.KEYS, how='left', validate='many_to_one')


def prepare(root, ext):
    root = Path(root)
    policy = ext.read_config(root)
    inputs, upstream = input_fingerprints(root, ext)
    config, _, _, arms, splits, data = base.prepare(root)
    if config != json.loads((root / base.REPORT_DIR / 'experiment_config.json').read_text(encoding='utf-8')):
        raise ValueError('Stage 06 configuration changed.')
    pd.testing.assert_frame_equal(splits, pd.read_csv(root / base.DATA_DIR / 'match_splits.csv'))
    data, audit = ext.build_features(root, data)
    columns = ext.resolve_columns(arms)
    validate_development(data, columns, ext)
    winners, selected, predictions = baseline_tables(root, data, config, ext.REFERENCES)
    return dict(policy=policy, config=config, inputs=inputs, upstream=upstream, columns=columns,
                splits=splits, data=data, audit=audit, reference_winners=winners,
                reference_selected=selected, reference_predictions=predictions)


def preflight(root, ext):
    p = prepare(root, ext)
    return dict(status='passed', extension=ext.CODE, contract_version=VERSION,
                upstream_stage06_id=p['upstream']['experiment_id'],
                development_matches=int(p['data'].match_id.nunique()), development_events=len(p['data']),
                expected_candidate_fits=4 * sum(len(p['config']['candidate_grid'][f]) for f in base.FAMILIES),
                input_columns=p['columns'], audit=p['audit'].to_dict('records'),
                training_fits=0, test_predictions=0, E1_result_dependency=False, main_model_promotion=False)


def fit_extension(data, columns, config, model_dir, prediction_path, ext, progress=print):
    validate_development(data, columns, ext)
    model_dir, prediction_path = Path(model_dir), Path(prediction_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    rows, audits = [], []
    total = 4 * sum(len(config['candidate_grid'][f]) for f in base.FAMILIES)
    count, first = 0, True
    with gzip.open(prediction_path, 'wt', encoding='utf-8', newline='') as stream:
        for track in TRACKS:
            gender, _, number = track.split('_')
            group = data[data.gender.eq(gender) & data.match_innings.eq(int(number))]
            train, val = group[group.split.eq('train')], group[group.split.eq('validation')]
            features = base.track_columns(columns, int(number))
            for family in base.FAMILIES:
                for index, parameters in enumerate(config['candidate_grid'][family]):
                    model = base.make_pipeline(family, parameters, features, config)
                    with threadpool_limits(limits=config['threads']), warnings.catch_warnings():
                        warnings.simplefilter('error', ConvergenceWarning)
                        model.fit(train[features], train[base.TARGET])
                        probability = base.probability(model, val[features])
                    filename = f'{track}__{ext.CODE}__{family}__candidate_{index}.joblib'
                    joblib.dump(model, model_dir / filename)
                    rows.append(dict(track=track, arm=ext.ARM, arm_code=ext.CODE, family=family,
                        candidate=index, parameters=json.dumps(parameters, sort_keys=True),
                        train_events=len(train), train_matches=train.match_id.nunique(), model_file=filename,
                        **base.metrics(val[base.TARGET], probability, val.match_id, config['probability_epsilon'])))
                    audits.append(dict(track=track, family=family, candidate=index, **base.feature_audit(model, features)))
                    export = val[base.META + [base.TARGET, 'innings_phase']].copy()
                    export['track'], export['arm'], export['family'] = track, ext.ARM, family
                    export['candidate'], export['raw_model_probability'] = index, probability
                    export.to_csv(stream, index=False, header=first)
                    first = False
                    count += 1
                    progress(f'[{count}/{total}] {track} | {ext.CODE} | {family} {index}: validation log loss {rows[-1]["validation_event_log_loss"]:.6f}')
    candidates = pd.DataFrame(rows)
    winners = pd.DataFrame([base.best_record(g.to_dict('records')) for _, g in candidates.groupby(['track', 'family'], sort=False)])
    selected = pd.DataFrame([base.best_record(g.to_dict('records')) for _, g in winners.groupby('track', sort=False)])
    return candidates, winners, selected, audits


def comparisons(candidates, winners, selected, reference_winners, reference_selected, ext):
    rows = []
    def append(mode, treatment, reference):
        if treatment.track != reference.track:
            raise ValueError('Cannot compare different tracks.')
        if mode == MODES[0] and (treatment.family != reference.family or json.loads(treatment.parameters) != json.loads(reference.parameters)):
            raise ValueError('Fixed comparison changed algorithm/parameters.')
        rows.append(dict(comparison_mode=mode, track=treatment.track, treatment=ext.CODE,
            reference=reference.arm, reference_code=base.ARM_CODES[reference.arm],
            comparison_role='primary' if reference.arm == 'both_resources' else 'supporting',
            treatment_family=treatment.family, reference_family=reference.family,
            treatment_candidate=int(treatment.candidate), reference_candidate=int(reference.candidate),
            treatment_parameters=treatment.parameters, reference_parameters=reference.parameters,
            treatment_log_loss=treatment.validation_event_log_loss, reference_log_loss=reference.validation_event_log_loss,
            validation_log_loss_gain=reference.validation_event_log_loss - treatment.validation_event_log_loss,
            validation_match_log_loss_gain=reference.validation_match_log_loss - treatment.validation_match_log_loss,
            validation_brier_gain=reference.validation_brier - treatment.validation_brier))
    for ref in reference_winners.itertuples(index=False):
        same = candidates[candidates.track.eq(ref.track) & candidates.family.eq(ref.family) & candidates.candidate.eq(ref.candidate)]
        best = winners[winners.track.eq(ref.track) & winners.family.eq(ref.family)]
        if len(same) != 1 or len(best) != 1:
            raise ValueError('Missing matched candidate/family winner.')
        append(MODES[0], same.iloc[0], ref)
        append(MODES[1], best.iloc[0], ref)
    for ref in reference_selected.itertuples(index=False):
        best = selected[selected.track.eq(ref.track)]
        if len(best) != 1:
            raise ValueError('Missing selected extension model.')
        append(MODES[2], best.iloc[0], ref)
    return pd.DataFrame(rows)


def selected_predictions(predictions, selected, references, reference_selected, ext):
    frames = []
    for row in selected.itertuples(index=False):
        frames.append(predictions[predictions.track.eq(row.track) & predictions.family.eq(row.family) & predictions.candidate.eq(row.candidate)].copy())
    for row in reference_selected.itertuples(index=False):
        part = references[references.track.eq(row.track) & references.arm.eq(row.arm) & references.family.eq(row.family)].copy()
        part['candidate'] = row.candidate
        frames.append(part)
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(['track', 'arm'] + base.KEYS).any():
        raise ValueError('Duplicate selected predictions.')
    if set(result.track) != set(TRACKS) or set(result.arm) != set(ext.REFERENCES + [ext.ARM]):
        raise ValueError('Missing selected systems.')
    for track in TRACKS:
        group = result[result.track.eq(track)]
        reference = group[group.arm.eq('both_resources')].set_index(base.KEYS).sort_index()
        for arm in ext.REFERENCES + [ext.ARM]:
            other = group[group.arm.eq(arm)].set_index(base.KEYS).sort_index()
            if not other.index.equals(reference.index) or not other[[base.TARGET, 'innings_phase']].equals(reference[[base.TARGET, 'innings_phase']]):
                raise ValueError('Selected validation coverage/labels/phase differ.')
    return result


def phase_diagnostics(predictions, epsilon, ext):
    rows = []
    for (track, arm), group in predictions.groupby(['track', 'arm'], sort=False):
        if group.family.nunique() != 1 or group.candidate.nunique() != 1:
            raise ValueError('No phase-specific selection permitted.')
        for phase in PHASES:
            part = group[group.innings_phase.eq(phase)]
            values = base.metrics(part[base.TARGET], part.raw_model_probability, part.match_id, epsilon) if len(part) else {
                'validation_event_log_loss': None, 'validation_match_log_loss': None, 'validation_brier': None,
                'validation_match_brier': None, 'validation_accuracy': None, 'validation_auc': None,
                'validation_events': 0, 'validation_matches': 0}
            rows.append(dict(track=track, arm=arm, arm_code=ext.CODE if arm == ext.ARM else base.ARM_CODES[arm],
                             phase=phase, family=group.family.iloc[0], candidate=int(group.candidate.iloc[0]), **values))
    return pd.DataFrame(rows)


def run(root, ext):
    root = Path(root).resolve()
    p = prepare(root, ext)
    out, processed = root / ext.REPORT_DIR, root / ext.DATA_DIR
    out.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps(p['inputs'], sort_keys=True).encode()).hexdigest()
    write_json(out / 'manifest.json', dict(status='running', contract_version=VERSION, experiment_id=identity))
    candidates, winners, selected, audits = fit_extension(p['data'], p['columns'], p['config'], out / 'models',
        processed / 'validation_predictions.csv.gz', ext, progress=lambda s: print(s, flush=True))
    predictions = selected_predictions(pd.read_csv(processed / 'validation_predictions.csv.gz'), selected,
                                       p['reference_predictions'], p['reference_selected'], ext)
    tables = dict(candidate_metrics=candidates, family_winners=winners, selected_models=selected,
        reference_family_winners=p['reference_winners'], reference_selected_models=p['reference_selected'],
        validation_comparisons=comparisons(candidates, winners, selected, p['reference_winners'], p['reference_selected'], ext),
        phase_diagnostics=phase_diagnostics(predictions, p['config']['probability_epsilon'], ext),
        feature_audit=p['audit'], split_summary=base.split_summary(p['splits']))
    for name, table in tables.items():
        table.to_csv(out / (name + '.csv'), index=False)
    predictions.to_csv(processed / 'selected_validation_predictions.csv.gz', index=False, compression='gzip')
    summary = dict(stage='07_resource_representation_experiments', contract_version=VERSION,
        extension=ext.CODE, experiment_id=identity, upstream_stage06_id=p['upstream']['experiment_id'],
        created_utc=datetime.now(timezone.utc).isoformat(), candidate_fits=len(candidates), saved_candidate_models=len(candidates),
        selected_models=len(selected), matches=len(p['splits']), events=int(p['splits'].source_events.sum()),
        development_events=len(p['data']), test_predictions=0, test_scored=False, calibration_fits=0,
        main_model_promotion=False, comparison_modes=MODES, limitations=ext.LIMITATIONS)
    payloads = dict(experiment_config=dict(extension=p['policy'], inherited_stage06_config=p['config'],
        stage06_experiment_id=p['upstream']['experiment_id'], split_sha256=sha256(root / base.DATA_DIR / 'match_splits.csv')),
        feature_groups=dict(contract_version=VERSION, target=base.TARGET, columns=p['columns'], references=ext.REFERENCES,
                            primary_reference='both_resources', added_or_replacement_features=ext.FEATURES, prediction_timing='pre-event'),
        preprocessing_audit=audits, experiment_summary=summary)
    for name, value in payloads.items():
        write_json(out / (name + '.json'), value)
    final_inputs, _ = input_fingerprints(root, ext)
    if p['inputs'] != final_inputs:
        raise ValueError('Inputs changed while running; outputs remain incomplete.')
    paths = [out / (name + '.csv') for name in tables] + [out / (name + '.json') for name in payloads]
    paths += [processed / name for name in ['validation_predictions.csv.gz', 'selected_validation_predictions.csv.gz']]
    paths += [out / 'models' / name for name in candidates.model_file]
    write_json(out / 'manifest.json', dict(status='complete', contract_version=VERSION, experiment_id=identity,
        inputs=p['inputs'], output_sha256={path.relative_to(root).as_posix(): sha256(path) for path in paths}))
    return summary


def verified_output(root, ext):
    root = Path(root)
    path = root / ext.REPORT_DIR / 'manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f'No {ext.CODE} resource_sums_v2 results. Historical outputs are not reused; train explicitly only after approval.')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    inputs, upstream = input_fingerprints(root, ext)
    identity = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    if (manifest.get('status') != 'complete' or manifest.get('inputs') != inputs
            or manifest.get('contract_version') != VERSION or manifest.get('experiment_id') != identity):
        raise ValueError(f'{ext.CODE} output stale/incomplete; no automatic training.')
    for relative, expected in manifest['output_sha256'].items():
        if sha256(rooted(root, relative)) != expected:
            raise ValueError('Extension artifact changed: ' + relative)
    summary = json.loads((root / ext.REPORT_DIR / 'experiment_summary.json').read_text(encoding='utf-8'))
    if (summary['upstream_stage06_id'] != upstream['experiment_id'] or summary['contract_version'] != VERSION
            or summary['extension'] != ext.CODE or summary['experiment_id'] != identity):
        raise ValueError('Extension summary provenance mismatch.')
    return summary
