"""Native importance supplement for verified Stage 08 models; no fitting/prediction."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import probability_evaluation_joint as stage
from data_preparation import rooted, sha256, write_json

REPORT_DIR = stage.REPORT_DIR + '/native_importance'
SOURCE_PATH = 'src/probability_native_importance.py'
FILES = ['feature_importance.csv', 'model_status.csv', 'summary.json']
NOTES = [
    'XGBoost gain is the mean training split gain for each retained transformed feature, not a percentage.',
    'Random Forest importance is normalised mean impurity decrease (MDI), not XGBoost Gain.',
    'Native values are training-model summaries, not test loss changes or causal effects.',
    'Different metrics and models are not on a common scale; do not rank tracks by these values.',
    'One-hot phase indicators are listed separately and mapped to their raw input; their gains are not summed.',
    'A retained feature unused by XGBoost has zero gain; a removed constant has no native score (missing, not zero).',
    'Correlated inputs and split-selection/cardinality biases limit native importance interpretation.',
    'Permutation importance remains the common feature-level and block-level report for all four models.',
]


def prepare(root):
    root = Path(root)
    evaluation = stage.verified_output(root)
    protocol = json.loads((root/stage.REPORT_DIR/'frozen_protocol.json').read_text(encoding='utf-8'))
    registry = [r for r in protocol['models'] if r['preferred']]
    if len(registry) != 4 or {r['track']: r['arm_code'] for r in registry} != evaluation['preferred']:
        raise ValueError('Native importance requires the four frozen final models.')
    paths = [SOURCE_PATH, stage.REPORT_DIR+'/manifest.json'] + [r['model_path'] for r in registry]
    inputs = {p: sha256(rooted(root, p)) for p in paths}
    if any(inputs[r['model_path']] != r['model_sha256'] for r in registry):
        raise ValueError('Native importance model changed.')
    identity = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    return evaluation, registry, inputs, identity


def transformed_features(model, row):
    if list(model.feature_names_in_) != row['features']:
        raise ValueError('Native importance raw feature order differs.')
    prep = model.named_steps['preprocess']
    names = list(prep.get_feature_names_out())
    keep = np.asarray(model.named_steps['nonconstant'].get_support(), dtype=bool)
    if len(names) != len(keep) or len(set(names)) != len(names):
        raise ValueError('Invalid transformed feature mapping.')
    encoder = prep.named_transformers_['phase']
    phase_names = set(encoder.get_feature_names_out(['innings_phase']))
    raw_names = []
    for name in names:
        if name in row['features']:
            raw_names.append(name)
        elif name in phase_names:
            raw_names.append('innings_phase')
        else:
            raise ValueError('Unmapped transformed feature: '+name)
    estimator = model.named_steps['model']
    if estimator.n_features_in_ != int(keep.sum()):
        raise ValueError('Estimator feature count differs from retained inputs.')
    return names, raw_names, keep


def extract_model(model, row):
    names, raw_names, keep = transformed_features(model, row)
    estimator = model.named_steps['model']
    status = {k: row[k] for k in ['track', 'arm_code', 'family', 'model_sha256']}
    status.update(retained_features=int(keep.sum()), removed_constant_features=int((~keep).sum()))
    if row['family'] == 'hist_gradient_boosting':
        if hasattr(estimator, 'feature_importances_'):
            raise ValueError('HistGradientBoosting interface changed; review native metric semantics.')
        status.update(metric='not_exported', status='unsupported_public_interface',
            explanation='Installed HistGradientBoosting has no public Gain/feature_importances_ interface; use permutation importance.')
        return [], status
    if row['family'] == 'logistic_regression':
        status.update(metric='not_applicable', status='not_applicable_linear_model',
            explanation='Logistic Regression is not a tree model; coefficients are not Gain. Use permutation importance.')
        return [], status
    retained = [name for name, selected in zip(names, keep) if selected]
    if row['family'] == 'xgboost':
        booster = estimator.get_booster()
        keys = booster.feature_names or ['f'+str(i) for i in range(len(retained))]
        if booster.feature_names is not None and list(booster.feature_names) != retained:
            raise ValueError('Named XGBoost features differ from pipeline feature order.')
        if booster.num_features() != len(retained) or len(keys) != len(retained) or len(set(keys)) != len(keys):
            raise ValueError('XGBoost feature indices do not match retained pipeline columns.')
        gain = booster.get_score(importance_type='gain')
        splits = booster.get_score(importance_type='weight')
        if not set(gain).issubset(keys) or set(gain) != set(splits):
            raise ValueError('Unknown or inconsistent XGBoost split feature keys.')
        values = np.array([gain.get(key, 0.) for key in keys], dtype=float)
        used = [key in splits for key in keys]
        metric = 'xgboost_mean_split_gain'
        explanation = 'Mean gain over training splits using each transformed feature; unnormalised.'
    elif row['family'] == 'random_forest':
        values = np.asarray(estimator.feature_importances_, dtype=float)
        used = [None]*len(retained)
        metric = 'random_forest_normalized_mdi'
        explanation = 'Normalised mean decrease in impurity across trees; not XGBoost Gain.'
        if values.sum() != 0 and not np.isclose(values.sum(), 1., atol=1e-12):
            raise ValueError('Random Forest MDI does not sum to one.')
    else:
        raise ValueError('Unsupported final-model family: '+row['family'])
    if values.shape != (len(retained),) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid native importance values.')
    status.update(metric=metric, status='exported', explanation=explanation)
    rows, index = [], 0
    for name, raw_name, selected in zip(names, raw_names, keep):
        record = {k: row[k] for k in ['track', 'arm_code', 'family', 'model_sha256']}
        record.update(feature=name, raw_feature=raw_name, metric=metric,
            retained=bool(selected), status='retained' if selected else 'constant_removed',
            importance=float(values[index]) if selected else None,
            used_in_split=used[index] if selected else None)
        rows.append(record)
        if selected:
            index += 1
    return rows, status


def run(root):
    root = Path(root).resolve()
    evaluation, registry, inputs, identity = prepare(root)
    out = root/REPORT_DIR
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'manifest.json', dict(status='running', native_importance_id=identity))
    records, statuses = [], []
    for row in registry:
        rows, status = extract_model(joblib.load(rooted(root, row['model_path'])), row)
        records.extend(rows)
        statuses.append(status)
    features = pd.DataFrame(records, columns=['track', 'arm_code', 'family', 'model_sha256',
        'feature', 'raw_feature', 'metric', 'retained', 'status', 'importance', 'used_in_split'])
    features.to_csv(out/'feature_importance.csv', index=False)
    pd.DataFrame(statuses).to_csv(out/'model_status.csv', index=False)
    summary = dict(native_importance_id=identity, evaluation_id=evaluation['evaluation_id'],
        preferred=evaluation['preferred'], models=registry, feature_rows=len(features),
        exported_models=sum(s['status']=='exported' for s in statuses),
        training_fits=0, prediction_calls=0, notes=NOTES)
    write_json(out/'summary.json', summary)
    if any(sha256(rooted(root, p)) != digest for p, digest in inputs.items()):
        raise ValueError('Inputs changed during native importance extraction.')
    write_json(out/'manifest.json', dict(status='complete', native_importance_id=identity, inputs=inputs,
        output_sha256={REPORT_DIR+'/'+name: sha256(out/name) for name in FILES}))
    return summary


def load_tables(root):
    root = Path(root)
    out = root/REPORT_DIR
    if not (out/'manifest.json').is_file():
        raise FileNotFoundError('No native importance supplement. Set RUN_NATIVE_IMPORTANCE = True explicitly.')
    evaluation, registry, inputs, identity = prepare(root)
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('status') != 'complete' or manifest.get('inputs') != inputs
            or manifest.get('native_importance_id') != identity):
        raise ValueError('Native importance stale/incomplete; no automatic extraction.')
    if set(manifest['output_sha256']) != {REPORT_DIR+'/'+name for name in FILES}:
        raise ValueError('Incomplete native importance manifest.')
    for path, digest in manifest['output_sha256'].items():
        if sha256(rooted(root, path)) != digest:
            raise ValueError('Native importance artifact changed: '+path)
    summary = json.loads((out/'summary.json').read_text(encoding='utf-8'))
    if (summary['models'] != registry or summary['evaluation_id'] != evaluation['evaluation_id']
            or summary['native_importance_id'] != identity or summary['preferred'] != evaluation['preferred']):
        raise ValueError('Native importance model identity differs from Stage 08.')
    return summary, pd.read_csv(out/'feature_importance.csv'), pd.read_csv(out/'model_status.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.root) if args.export else load_tables(args.root)[0], indent=2))
