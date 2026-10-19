"""Training/validation diagnostics for frozen Stage 08 models; no fitting or test scoring."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import probability_evaluation_joint as stage
from data_preparation import rooted, sha256, write_json

base = stage.base
REPORT_DIR = stage.REPORT_DIR + '/development_diagnostics'
SOURCE_PATH = 'src/probability_development_diagnostics.py'
METRICS = ['event_log_loss', 'brier']
TABLES = ['train_validation_metrics', 'preferred_train_validation_metrics']
LIMITATIONS = [
    'Training scores are in-sample, not independent generalisation estimates.',
    'Validation scores also guided model selection and are not independent confirmation.',
    'Positive gap means validation loss exceeds training loss; it is not a standalone overfitting verdict.',
    'Chronological population changes and match composition can also affect the gap.',
    'Log Loss and Brier are event-weighted; deliveries from one match are not independent outcomes.',
    'These diagnostics do not change feature groups, algorithms, parameters or selected models.',
]


def prepare(root):
    config, inputs, upstream, registry = stage.prepare_registry(root)
    selection_id = hashlib.sha256(json.dumps(dict(inputs=inputs, registry=registry), sort_keys=True).encode()).hexdigest()
    fingerprints = dict(inputs, **{SOURCE_PATH: sha256(Path(root)/SOURCE_PATH)})
    identity = hashlib.sha256(json.dumps(dict(inputs=fingerprints, registry=registry), sort_keys=True).encode()).hexdigest()
    candidates = stage.validate_candidate_winners(root, registry)
    reference = candidates.set_index(['track', 'arm_code', 'family', 'candidate'])
    return config, fingerprints, registry, reference, selection_id, identity


def load_development(root):
    # The shared loader checks the complete cohort/split and removes test rows.
    _, _, _, _, splits, data = base.prepare(root)
    data, _ = stage.interactions.Extension('IBW').build_features(root, data)
    return data, splits


def validate_data(data, splits):
    arms = base.resolve_arms(base.resources.feature_contract(base.game_state.feature_contract()))
    base.validate_development(data, arms)
    stage.interactions.Extension('IBW').validate_features(data)
    if splits.match_id.duplicated().any() or set(splits.split) != set(base.SPLITS):
        raise ValueError('Invalid frozen match split.')
    lookup = splits.set_index('match_id')
    expected = lookup[lookup.split.isin(['train', 'validation'])]
    if set(data.match_id) != set(expected.index):
        raise ValueError('Development cohort differs from frozen split.')
    for key in ['split', 'gender', 'match_date']:
        if not data[key].eq(data.match_id.map(lookup[key])).all():
            raise ValueError('Development metadata differs from frozen split: '+key)
    actual_counts = data.groupby('match_id').size().sort_index()
    if not np.array_equal(actual_counts.to_numpy(), expected.source_events.sort_index().to_numpy()):
        raise ValueError('Development event counts differ from frozen split.')


def score_models(root, data, splits, registry, reference, config, progress=print):
    validate_data(data, splits)
    stage.validate_registry(registry, config)
    rows = []
    for row in registry:
        path = rooted(root, row['model_path'])
        if sha256(path) != row['model_sha256']:
            raise ValueError('Diagnostic model changed after selection.')
        model = joblib.load(path)
        columns = row['features']
        if columns and list(model.feature_names_in_) != columns:
            raise ValueError('Diagnostic model feature order differs.')
        if not columns and row['arm_code'] != 'R0':
            raise ValueError('Only R0 can have no model inputs.')
        saved = reference.loc[(row['track'], row['arm_code'], row['family'], row['candidate'])]
        gender, _, innings = row['track'].split('_')
        subset = data[data.gender.eq(gender) & data.match_innings.eq(int(innings))]
        result = {k: row[k] for k in ['track', 'arm_code', 'family', 'candidate', 'preferred', 'model_path', 'model_sha256']}
        result['parameters'] = json.dumps(row['parameters'], sort_keys=True)
        for split in ['train', 'validation']:
            part = subset[subset.split.eq(split)]
            x = part[columns] if columns else np.zeros((len(part), 1))
            with threadpool_limits(limits=2):
                p = base.probability(model, x)
            metrics = base.metrics(part[base.TARGET], p, part.match_id, config['probability_epsilon'])
            for metric in METRICS + ['events', 'matches']:
                result[split+'_'+metric] = metrics['validation_'+metric]
            for count in ['events', 'matches']:
                if result[split+'_'+count] != saved[split+'_'+count]:
                    raise ValueError('Diagnostic row counts differ from fitted candidate.')
        for metric in METRICS:
            value = result['validation_'+metric]
            if not np.isclose(value, saved['validation_'+metric], rtol=0, atol=1e-10):
                raise ValueError('Validation '+metric+' does not reproduce the saved result.')
            result[metric+'_gap'] = value-result['train_'+metric]
        if not np.isclose(result['validation_event_log_loss'], row['validation_event_log_loss'], rtol=0, atol=1e-10):
            raise ValueError('Validation Log Loss differs from frozen registry.')
        rows.append(result)
        progress('Development scores: '+row['track']+' | '+row['arm_code'])
    table = pd.DataFrame(rows)
    validate_table(table, registry)
    return table


def validate_table(table, registry):
    expected = {(r['track'], r['arm_code']): r for r in registry}
    if (len(table) != 32 or table.duplicated(['track', 'arm_code']).any()
            or set(zip(table.track, table.arm_code)) != set(expected)):
        raise ValueError('Incomplete development diagnostic table.')
    for row in table.to_dict('records'):
        model = expected[(row['track'], row['arm_code'])]
        for key in ['family', 'candidate', 'preferred', 'model_path', 'model_sha256']:
            if row[key] != model[key]:
                raise ValueError('Diagnostic identity differs from frozen registry.')
        if json.loads(row['parameters']) != model['parameters']:
            raise ValueError('Diagnostic parameters differ from frozen registry.')
        for metric in METRICS:
            values = [row['train_'+metric], row['validation_'+metric], row[metric+'_gap']]
            if not np.isfinite(values).all() or min(values[:2]) < 0:
                raise ValueError('Invalid diagnostic score.')
            if not np.isclose(values[2], values[1]-values[0], rtol=0, atol=1e-12):
                raise ValueError('Gap must equal validation minus training.')
        if not np.isclose(row['validation_event_log_loss'], model['validation_event_log_loss'], rtol=0, atol=1e-10):
            raise ValueError('Diagnostic validation score differs from selected model.')


def run(root, progress=print):
    root = Path(root).resolve()
    config, inputs, registry, reference, selection_id, identity = prepare(root)
    out = root/REPORT_DIR
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'manifest.json', dict(status='running', diagnostic_id=identity))
    # Freeze model identities before scoring either split. There is no selector after scoring.
    write_json(out/'protocol.json', dict(diagnostic_id=identity, selection_id=selection_id,
        models=registry, splits=['train', 'validation'], metrics=METRICS,
        weighting='equal delivery-event weight', gap='validation minus training',
        training_fits=0, test_scored=False, purpose='diagnostic only; no model reselection'))
    data, splits = load_development(root)
    table = score_models(root, data, splits, registry, reference, config, progress)
    preferred = table[table.preferred].reset_index(drop=True)
    table.to_csv(out/(TABLES[0]+'.csv'), index=False)
    preferred.to_csv(out/(TABLES[1]+'.csv'), index=False)
    summary = dict(status='complete', diagnostic_id=identity, selection_id=selection_id,
        created_utc=datetime.now(timezone.utc).isoformat(), models_scored=len(table),
        preferred=stage.select_preferred(registry), training_fits=0, test_scored=False,
        split_counts={s: dict(events=int(data.split.eq(s).sum()),
            matches=int(data.loc[data.split.eq(s), 'match_id'].nunique())) for s in ['train', 'validation']},
        limitations=LIMITATIONS)
    write_json(out/'summary.json', summary)
    final_inputs, _ = stage.input_fingerprints(root)
    final_inputs[SOURCE_PATH] = sha256(root/SOURCE_PATH)
    if final_inputs != inputs or any(sha256(rooted(root, r['model_path'])) != r['model_sha256'] for r in registry):
        raise ValueError('Inputs changed during diagnostics; outputs remain incomplete.')
    paths = [out/(name+'.csv') for name in TABLES] + [out/'summary.json', out/'protocol.json']
    write_json(out/'manifest.json', dict(status='complete', diagnostic_id=identity, inputs=inputs,
        output_sha256={p.relative_to(root).as_posix(): sha256(p) for p in paths}))
    return summary


def verified_output(root):
    root = Path(root)
    out = root/REPORT_DIR
    if not (out/'manifest.json').is_file():
        raise FileNotFoundError('No development diagnostics. Set RUN_DEVELOPMENT_DIAGNOSTICS = True explicitly.')
    _, inputs, registry, _, selection_id, identity = prepare(root)
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete' or manifest.get('inputs') != inputs or manifest.get('diagnostic_id') != identity:
        raise ValueError('Development diagnostics stale/incomplete; no automatic scoring.')
    expected_paths = {REPORT_DIR+'/'+name+'.csv' for name in TABLES} | {REPORT_DIR+'/summary.json', REPORT_DIR+'/protocol.json'}
    if set(manifest['output_sha256']) != expected_paths:
        raise ValueError('Incomplete diagnostic manifest.')
    for path, digest in manifest['output_sha256'].items():
        if sha256(rooted(root, path)) != digest:
            raise ValueError('Diagnostic artifact changed: '+path)
    table = pd.read_csv(out/(TABLES[0]+'.csv'))
    validate_table(table, registry)
    preferred = pd.read_csv(out/(TABLES[1]+'.csv'))
    pd.testing.assert_frame_equal(preferred, table[table.preferred].reset_index(drop=True))
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    summary = json.loads((out/'summary.json').read_text(encoding='utf-8'))
    if (protocol['models'] != registry or protocol['selection_id'] != selection_id
            or summary['selection_id'] != selection_id or summary['diagnostic_id'] != identity
            or summary['preferred'] != stage.select_preferred(registry) or summary['models_scored'] != 32
            or summary['training_fits'] != 0 or summary['test_scored'] is not False):
        raise ValueError('Diagnostic summary/registry mismatch.')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--score', action='store_true', help='Score saved models on train/validation only.')
    args = parser.parse_args()
    print(json.dumps(run(args.root) if args.score else verified_output(args.root), indent=2))
