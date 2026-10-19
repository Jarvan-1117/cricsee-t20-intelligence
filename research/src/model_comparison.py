"""Stage 06 v2: five arms, date-group splits and validation-only selection.

No test predictions, test metrics, calibration fitting, extensions, or NWC.
Saved pipelines are fitted on TRAIN ONLY, not refitted on validation. Historical
features remain the Stage 03 prior-date rolling histories for every arm.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

import remaining_resource_features as resources
import game_state_features as game_state
from data_preparation import rooted, sha256, write_json

STAGE = '06_model_training_and_comparison'
VERSION = 'resource_sums_v2'
PREFERRED_ARM = 'both_resources'
SELECTION_POLICY = 'fixed_rbw_main_v2'
SELECTION_DESCRIPTION = ('Main resource models: fixed RBW inputs in all four tracks; select '
    'algorithm/parameters by validation event log loss within RBW. Retain all five comparison '
    'arms. Cross-group winners are display-only and never replace the main registry. '
    'Exact ties: constant, LR, RF, HGB, XGBoost, then candidate index and arm order. No test selection.')
REPORT_DIR = 'outputs/' + STAGE
DATA_DIR = 'data/processed/model_comparison'
TARGET = 'first_batting_team_win'
KEYS = resources.KEYS
META = list(dict.fromkeys(KEYS + ['gender', 'match_date']))
ARM_CODES = {'constant': 'R0', 'state': 'R1', 'batting_resources': 'RB',
             'bowling_resources': 'RW', 'both_resources': 'RBW'}
ARMS = list(ARM_CODES)[1:]
FAMILIES = ['logistic_regression', 'random_forest', 'hist_gradient_boosting', 'xgboost']
CHASE = ['target_runs', 'runs_required', 'required_run_rate', 'run_rate_differential']
SPLITS = ['train', 'validation', 'test']


def read_config(root):
    c = json.loads((Path(root) / 'config/model_comparison.json').read_text(encoding='utf-8'))
    if (c['stage'] != STAGE or c.get('contract_version') != VERSION or c['target'] != TARGET
            or c.get('preferred_arm') != PREFERRED_ARM or c.get('selection_policy') != SELECTION_POLICY
            or c['selection_metric'] != 'validation_event_log_loss'
            or c['optional_background'] != [] or c['refit_train_validation'] is not False):
        raise ValueError('Unsupported Stage 06 policy; change design and tests explicitly.')
    f = np.asarray(c['split_fractions'], dtype=float)
    if f.shape != (3,) or not np.isfinite(f).all() or (f <= 0).any() or not np.isclose(f.sum(), 1):
        raise ValueError('Three positive split fractions summing to one required.')
    if not isinstance(c['threads'], int) or c['threads'] < 1 or not 0 < c['probability_epsilon'] < 1e-6:
        raise ValueError('Invalid thread count or probability clipping epsilon.')
    if set(c['candidate_grid']) != set(FAMILIES) or any(not c['candidate_grid'][f] for f in FAMILIES):
        raise ValueError('All four candidate families need a nonempty prespecified grid.')
    allowed = {'logistic_regression': {'C'},
               'random_forest': {'n_estimators', 'max_depth', 'min_samples_leaf'},
               'hist_gradient_boosting': {'max_iter', 'learning_rate', 'max_leaf_nodes'},
               'xgboost': {'n_estimators', 'learning_rate', 'max_depth'}}
    if any(set(p) != allowed[f] for f in FAMILIES for p in c['candidate_grid'][f]):
        raise ValueError('Unsupported candidate parameter override.')
    return c


def resolve_arms(contract):
    expected = resources.feature_contract(game_state.feature_contract())
    if contract != expected or contract.get('contract_version') != VERSION:
        raise ValueError('Expected the exact Stage 05 v2 contract; no legacy groups or extra predictors.')
    if contract['arm_codes'] != {code: arm for arm, code in ARM_CODES.items()}:
        raise ValueError('Arm identifiers differ from the frozen Stage 05 contract.')
    arms = {name: list(contract['main_arms'][name]) for name in ARM_CODES}
    for name, columns in arms.items():
        # match_innings is a permitted state feature but is dropped per track.
        forbidden = (set(columns) & set(META + [TARGET])) - {'match_innings'}
        direct = set(columns) & set(contract['excluded_direct_quality_fields'])
        if len(set(columns)) != len(columns) or forbidden or direct or any(c.startswith(('audit_', 'striker_', 'non_striker_', 'bowler_')) for c in columns):
            raise ValueError('Duplicate columns or identifiers/outcome in predictors: ' + name)
    return arms


def chronological_splits(cohort, fractions):
    """Choose nearest cumulative match-count boundaries without splitting dates."""
    selected = cohort.loc[cohort.included_in_resource_sample.eq(True)].copy()
    if selected.empty or selected.match_id.duplicated().any():
        raise ValueError('Empty or duplicate resource cohort.')
    if set(selected.gender) != {'female', 'male'}:
        raise ValueError('Both gender populations required.')
    selected['match_date'] = pd.to_datetime(selected.match_date, errors='raise').dt.strftime('%Y-%m-%d')
    if selected.match_date.isna().any() or selected[['match_id', 'source_events']].isna().any().any():
        raise ValueError('Missing split metadata.')
    rows = []
    for gender, group in selected.groupby('gender', sort=True):
        dates = group.groupby('match_date').size().sort_index()
        if len(dates) < 3:
            raise ValueError('At least three distinct match dates per gender required.')
        counts = dates.cumsum().to_numpy()
        n = len(group)
        first = min(range(len(dates) - 2), key=lambda i: (abs(counts[i] - n * fractions[0]), i))
        second = min(range(first + 1, len(dates) - 1),
                     key=lambda i: (abs(counts[i] - n * sum(fractions[:2])), i))
        group = group[['match_id', 'gender', 'match_date', 'source_events']].copy()
        group['split'] = np.where(group.match_date <= dates.index[first], 'train',
                                   np.where(group.match_date <= dates.index[second], 'validation', 'test'))
        rows.append(group)
    result = pd.concat(rows).sort_values(['gender', 'match_date', 'match_id']).reset_index(drop=True)
    for gender, group in result.groupby('gender'):
        if set(group.split) != set(SPLITS) or group.groupby('match_date').split.nunique().max() != 1:
            raise ValueError('Invalid chronological date groups: ' + gender)
    return result


def split_summary(splits):
    return splits.groupby(['gender', 'split']).agg(
        matches=('match_id', 'size'), events=('source_events', 'sum'),
        first_date=('match_date', 'min'), last_date=('match_date', 'max')).reset_index()


def input_fingerprints(root):
    root = Path(root)
    upstream = resources.verified_output(root)
    paths = ['config/model_comparison.json', 'src/model_comparison.py',
             resources.DATA_DIR + '/manifest.json', resources.DATA_DIR + '/model_input.csv.gz',
             resources.DATA_DIR + '/cohort.csv', resources.REPORT_DIR + '/feature_contract.json']
    hashes = {p: sha256(root / p) for p in paths}
    runtime = {'python': platform.python_version(), 'numpy': np.__version__,
               'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'joblib': joblib.__version__,
               'xgboost': xgboost.__version__}
    return {'files': hashes, 'runtime': runtime}, upstream


def validate_development(data, arms):
    expected = resolve_arms(resources.feature_contract(game_state.feature_contract()))
    if arms != expected:
        raise ValueError('Training inputs differ from the five frozen feature groups.')
    if set(data.split) != {'train', 'validation'}:
        raise ValueError('Training engine accepts development data only; no test rows.')
    if data.duplicated(KEYS).any() or data[META].isna().any().any():
        raise ValueError('Duplicate event keys or missing event metadata.')
    if not data[TARGET].isin([0, 1]).all() or data.groupby('match_id')[TARGET].nunique().max() != 1:
        raise ValueError('Invalid or inconsistent match target.')
    if (data.groupby('match_id').match_innings.agg(set) != {1, 2}).any():
        raise ValueError('Both innings of every development match required.')
    if not data.match_innings.isin([1, 2]).all() or not data.gender.isin(['female', 'male']).all():
        raise ValueError('Unexpected model track.')
    for _, match in data.groupby('match_id', sort=False):
        if match['split'].nunique() != 1 or match.gender.nunique() != 1 or match.match_date.nunique() != 1:
            raise ValueError('Match crosses split, gender or date.')
    for gender in ['female', 'male']:
        g = data[data.gender.eq(gender)]
        if set(g.split) != {'train', 'validation'}:
            raise ValueError('Each gender needs nonempty training and validation data.')
        if g.loc[g.split.eq('train'), 'match_date'].max() >= g.loc[g.split.eq('validation'), 'match_date'].min():
            raise ValueError('Training must strictly precede validation dates.')
        for innings in [1, 2]:
            for split in ['train', 'validation']:
                part = g[g.match_innings.eq(innings) & g.split.eq(split)]
                if part.empty or (split == 'train' and part[TARGET].nunique() != 2):
                    raise ValueError('Each training track needs both classes and nonempty validation.')
    columns = sorted(set(sum(arms.values(), [])))
    if not data.loc[data.match_innings.eq(1), CHASE].isna().all().all():
        raise ValueError('First innings must not have future chase targets.')
    for innings in [1, 2]:
        numeric = [c for c in columns if c != 'innings_phase' and not (innings == 1 and c in CHASE)]
        if not np.isfinite(data.loc[data.match_innings.eq(innings), numeric].to_numpy(dtype=float)).all():
            raise ValueError('Unexpected missing/nonfinite development features; do not silently drop events.')
    if not data.innings_phase.isin(['powerplay', 'middle', 'death']).all():
        raise ValueError('Unknown innings phase.')
    if data[resources.BAT_FEATURES + resources.BOWL_FEATURES].lt(0).any().any():
        raise ValueError('V2 resource sums cannot be negative; legacy centred features are not accepted.')


def preflight_audit(data, arms, splits, config):
    """No model fit: record exact row membership and feature allowlists per arm."""
    validate_development(data, arms)
    if splits.groupby(['gender', 'match_date'])['split'].nunique().max() != 1:
        raise ValueError('A date group crosses splits.')
    for gender, group in splits.groupby('gender'):
        for left, right in zip(SPLITS, SPLITS[1:]):
            if group.loc[group.split.eq(left), 'match_date'].max() >= group.loc[group.split.eq(right), 'match_date'].min():
                raise ValueError('Train/validation/test chronology overlaps: ' + gender)
    rows = []
    for (gender, innings, split), part in data.groupby(['gender', 'match_innings', 'split'], sort=True):
        key_hash = hashlib.sha256(part[KEYS].to_csv(index=False).encode('utf-8')).hexdigest()
        for arm, code in ARM_CODES.items():
            columns = track_columns(arms[arm], innings)
            rows.append({'track': f'{gender}_innings_{innings}', 'split': split, 'arm_code': code,
                         'events': len(part), 'matches': int(part.match_id.nunique()),
                         'event_keys_sha256': key_hash, 'input_columns': columns})
    return {'status': 'passed', 'contract_version': VERSION, 'training_started': False,
            'arms': ARM_CODES, 'excluded_direct_quality_fields': resources.quality.feature_contract()['current_player_block'],
            'same_development_rows_all_arms': True, 'whole_match_date_groups': True,
            'no_current_player_direct_quality_or_evidence': True, 'no_test_rows_in_training_engine': True,
            'match_split_sha256': hashlib.sha256(splits.to_csv(index=False).encode('utf-8')).hexdigest(),
            'config': config, 'rows': rows}


def prepare(root):
    root = Path(root)
    config = read_config(root)
    inputs, upstream = input_fingerprints(root)
    contract = json.loads((root / resources.REPORT_DIR / 'feature_contract.json').read_text(encoding='utf-8'))
    base = json.loads((root / 'outputs/02_game_state_features/feature_contract.json').read_text(encoding='utf-8'))
    if contract != resources.feature_contract(base):
        raise ValueError('Stage 05 feature contract does not match its implementation.')
    arms = resolve_arms(contract)
    cohort = pd.read_csv(root / resources.DATA_DIR / 'cohort.csv')
    splits = chronological_splits(cohort, config['split_fractions'])
    split_path = root / DATA_DIR / 'match_splits.csv'
    if split_path.exists():
        old = pd.read_csv(split_path)
        if not old.equals(splits):
            raise ValueError('Frozen split differs. Archive the previous experiment before changing the cohort/split policy.')
    columns = list(dict.fromkeys(META + [TARGET] + sum(arms.values(), [])))
    lookup = splits.set_index('match_id')
    counts = {}; parts = []
    for chunk in pd.read_csv(root / resources.DATA_DIR / 'model_input.csv.gz', usecols=columns, chunksize=50000):
        if not chunk.match_id.isin(lookup.index).all():
            raise ValueError('Model input includes matches outside the common cohort.')
        if not chunk.gender.eq(chunk.match_id.map(lookup.gender)).all() or not chunk.match_date.eq(chunk.match_id.map(lookup.match_date)).all():
            raise ValueError('Event metadata disagrees with the cohort.')
        for mid, count in chunk.groupby('match_id').size().items():
            counts[mid] = counts.get(mid, 0) + int(count)
        chunk['split'] = chunk.match_id.map(lookup['split'])
        # Test values are never used for validation, preprocessing, fitting or scoring.
        parts.append(chunk.loc[chunk.split.ne('test')].copy())
    if counts != lookup.source_events.to_dict():
        raise ValueError('Events missing or added relative to Stage 05 whole-match cohort.')
    data = pd.concat(parts, ignore_index=True)
    validate_development(data, arms)
    return config, inputs, upstream, arms, splits, data


def track_columns(columns, innings):
    return [c for c in columns if c != 'match_innings' and not (innings == 1 and c in CHASE)]


def make_pipeline(family, parameters, columns, config):
    numeric = [c for c in columns if c != 'innings_phase']
    numeric_steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
    if family == 'logistic_regression':
        numeric_steps.append(('scale', StandardScaler()))
    prep = ColumnTransformer([
        ('numeric', Pipeline(numeric_steps), numeric),
        ('phase', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['innings_phase'])
    ], sparse_threshold=0, verbose_feature_names_out=False)
    seed = config['seed']
    if family == 'logistic_regression':
        estimator = LogisticRegression(max_iter=2000, solver='lbfgs', random_state=seed, **parameters)
    elif family == 'random_forest':
        estimator = RandomForestClassifier(criterion='log_loss', max_features='sqrt',
            n_jobs=config['threads'], random_state=seed, **parameters)
    elif family == 'hist_gradient_boosting':
        estimator = HistGradientBoostingClassifier(early_stopping=False, min_samples_leaf=30,
            l2_regularization=1., random_state=seed, **parameters)
    elif family == 'xgboost':
        estimator = XGBClassifier(objective='binary:logistic', eval_metric='logloss',
            tree_method='hist', device='cpu', early_stopping_rounds=None,
            n_jobs=config['threads'], random_state=seed, verbosity=0, **parameters)
    else:
        raise ValueError('Unknown candidate family: ' + family)
    return Pipeline([('preprocess', prep), ('nonconstant', VarianceThreshold()), ('model', estimator)])


def probability(model, x):
    classes = list(model.classes_)
    if classes != [0, 1]:
        raise ValueError('Binary model classes must be [0, 1].')
    # XGBoost returns float32. Export float64 values so CSV reload reproduces
    # the exact numeric predictions used for metrics, like the other families.
    p = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid predicted probability.')
    return p


def metrics(y, p, match_ids, epsilon):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    clipped = np.clip(p, epsilon, 1 - epsilon)
    event_loss = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
    by_match = pd.DataFrame({'match_id': np.asarray(match_ids), 'loss': event_loss, 'brier': (y - p)**2}).groupby('match_id').mean()
    return {'validation_event_log_loss': float(event_loss.mean()),
            'validation_match_log_loss': float(by_match.loss.mean()),
            'validation_brier': float(brier_score_loss(y, p)),
            'validation_match_brier': float(by_match.brier.mean()),
            'validation_accuracy': float(accuracy_score(y, p >= .5)),
            'validation_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            'validation_events': len(y), 'validation_matches': len(by_match)}


def best_record(records):
    """Log loss only; exact ties use a prespecified family order then grid index."""
    return min(records, key=lambda r: (r['validation_event_log_loss'],
               (['constant'] + FAMILIES).index(r['family']), r['candidate']))


def validate_selected_models(selected):
    """Require a complete five-arm comparison before deriving either table."""
    tracks = {f'{gender}_innings_{innings}' for gender in ['female', 'male'] for innings in [1, 2]}
    if set(selected.track) != tracks or selected.duplicated(['track', 'arm']).any():
        raise ValueError('Expected unique per-track/per-arm selected models for all four tracks.')
    if not np.isfinite(selected.validation_event_log_loss.to_numpy(dtype=float)).all():
        raise ValueError('Final selection requires finite validation log loss.')
    for _, group in selected.groupby('track', sort=True):
        if set(group.arm) != set(ARM_CODES) or not group.arm_code.eq(group.arm.map(ARM_CODES)).all():
            raise ValueError('Final selection requires all five arms with matching identifiers.')
        if not group.loc[group.arm.eq('constant'), 'family'].eq('constant').all() or not group.loc[group.arm.ne('constant'), 'family'].isin(FAMILIES).all():
            raise ValueError('Unexpected algorithm for a comparison arm.')


def select_final_systems(selected):
    """Main-model registry for subsequent evaluation/NWC: four RBW rows only."""
    validate_selected_models(selected)
    return selected.loc[selected.arm.eq(PREFERRED_ARM)].sort_values('track').reset_index(drop=True)


def select_cross_group_best(selected):
    """Display-only cross-group summary; not a downstream model registry."""
    validate_selected_models(selected)
    rows = []
    arm_order = {arm: i for i, arm in enumerate(ARM_CODES)}
    for _, group in selected.groupby('track', sort=True):
        # Stable arm order breaks otherwise exact family/parameter/score ties.
        group = group.sort_values('arm', key=lambda values: values.map(arm_order), kind='stable')
        rows.append(best_record(group.to_dict('records')))
    return pd.DataFrame(rows).reset_index(drop=True)


def feature_audit(model, columns):
    transformed = list(model.named_steps['preprocess'].get_feature_names_out())
    support = model.named_steps['nonconstant'].get_support()
    return {'input_columns': columns, 'retained_transformed_columns': [c for c, keep in zip(transformed, support) if keep],
            'train_constant_columns_removed': [c for c, keep in zip(transformed, support) if not keep]}


def fit_development(data, arms, config, model_dir, prediction_path, progress=print):
    """Fit all arms on shared train rows. Only development data accepted."""
    validate_development(data, arms)
    model_dir, prediction_path = Path(model_dir), Path(prediction_path)
    model_dir.mkdir(parents=True, exist_ok=True); prediction_path.parent.mkdir(parents=True, exist_ok=True)
    candidates, winners, features = [], [], []
    count = 0
    total = 4 * len(ARMS) * sum(len(config['candidate_grid'][f]) for f in FAMILIES)
    first_write = True
    with gzip.open(prediction_path, 'wt', encoding='utf-8', newline='') as stream:
        for gender in ['female', 'male']:
            for innings in [1, 2]:
                track = f'{gender}_innings_{innings}'
                subset = data[data.gender.eq(gender) & data.match_innings.eq(innings)]
                train = subset[subset.split.eq('train')]; val = subset[subset.split.eq('validation')]
                for arm in ARM_CODES:
                    columns = track_columns(arms[arm], innings) if arm != 'constant' else []
                    families = FAMILIES if columns else ['constant']
                    for family in families:
                        grid = config['candidate_grid'][family] if columns else [{}]
                        best = None; best_model = None; best_p = None
                        for index, params in enumerate(grid):
                            if columns:
                                model = make_pipeline(family, params, columns, config)
                                with threadpool_limits(limits=config['threads']), warnings.catch_warnings():
                                    warnings.simplefilter('error', ConvergenceWarning)
                                    model.fit(train[columns], train[TARGET])
                                    p = probability(model, val[columns])
                                count += 1
                            else:
                                model = DummyClassifier(strategy='prior').fit(np.zeros((len(train), 1)), train[TARGET])
                                p = probability(model, np.zeros((len(val), 1)))
                            row = {'track': track, 'arm': arm, 'arm_code': ARM_CODES[arm],
                                   'family': family, 'candidate': index, 'parameters': json.dumps(params, sort_keys=True),
                                   'train_events': len(train), 'train_matches': train.match_id.nunique(),
                                   **metrics(val[TARGET], p, val.match_id, config['probability_epsilon'])}
                            candidates.append(row)
                            if best is None or best_record([best, row]) is row:
                                best, best_model, best_p = row, model, p
                            progress(f'[{count}/{total}] {track} | {ARM_CODES[arm]} | {family} {index}: validation log loss {row["validation_event_log_loss"]:.6f}')
                        name = f'{track}__{arm}__{family}.joblib'
                        joblib.dump(best_model, model_dir / name)
                        winners.append({**best, 'model_file': name})
                        if columns:
                            features.append({'track': track, 'arm': arm, 'family': family, **feature_audit(best_model, columns)})
                        export = val[META + [TARGET]].copy()
                        export['track'] = track; export['arm'] = arm; export['family'] = family
                        export['raw_model_probability'] = best_p
                        export.to_csv(stream, index=False, header=first_write)
                        first_write = False
    selected = [best_record(group.to_dict('records')) for _, group in pd.DataFrame(winners).groupby(['track', 'arm'], sort=False)]
    # All five arms remain in the study; only RBW supplies the four main models.
    systems = select_final_systems(pd.DataFrame(selected))
    return pd.DataFrame(candidates), pd.DataFrame(winners), pd.DataFrame(selected), pd.DataFrame(systems), features


def comparison_deltas(winners, selected):
    rows = []
    pairs = [('batting_resources', 'state'), ('bowling_resources', 'state'),
             ('both_resources', 'state'), ('both_resources', 'bowling_resources'),
             ('both_resources', 'batting_resources')]
    for view, table in [('within_family', winners[winners.family.ne('constant')]), ('selected_system', selected)]:
        keys = ['track', 'family'] if view == 'within_family' else ['track']
        for key, group in table.groupby(keys, sort=False):
            by_arm = group.set_index('arm')
            for treatment, reference in pairs:
                a, b = by_arm.loc[treatment], by_arm.loc[reference]
                rows.append({'view': view, 'track': a.track, 'treatment': treatment, 'reference': reference,
                    'treatment_family': a.family, 'reference_family': b.family,
                    'validation_log_loss_gain': float(b.validation_event_log_loss - a.validation_event_log_loss),
                    'validation_brier_gain': float(b.validation_brier - a.validation_brier)})
    return pd.DataFrame(rows)


def run(root):
    root = Path(root).resolve()
    config, inputs, upstream, arms, splits, data = prepare(root)
    preflight = preflight_audit(data, arms, splits, config)
    out = root / REPORT_DIR; processed = root / DATA_DIR
    out.mkdir(parents=True, exist_ok=True); processed.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    write_json(out / 'manifest.json', {'status': 'running', 'experiment_id': identity, 'contract_version': VERSION})
    write_json(out / 'preflight_audit.json', preflight)
    split_path = processed / 'match_splits.csv'
    if not split_path.exists():
        splits.to_csv(split_path, index=False)
    split_summary(splits).to_csv(out / 'split_summary.csv', index=False)
    write_json(out / 'experiment_config.json', config)
    write_json(out / 'feature_groups.json', {'contract_version': VERSION, 'arms': arms, 'arm_codes': ARM_CODES,
        'preferred_arm': PREFERRED_ARM, 'selection_policy': SELECTION_POLICY,
        'structural_omissions': {'all_tracks': ['match_innings'], 'innings_1': CHASE},
        'target': TARGET, 'optional_background': [], 'test_scored': False})
    print('Preflight passed: 5 arms; identical rows/splits; no direct current-player quality. Starting train-only fits; no test scoring.', flush=True)
    c, f, s, systems, features = fit_development(data, arms, config, out / 'models' / VERSION, processed / 'validation_predictions.csv.gz',
        progress=lambda message: print(message, flush=True))
    for table in [f, s, systems]:
        table['model_file'] = VERSION + '/' + table['model_file']
    for name, frame in [('candidate_metrics', c), ('family_winners', f), ('selected_models', s), ('selected_systems', systems),
                         ('validation_deltas', comparison_deltas(f, s))]:
        frame.to_csv(out / (name + '.csv'), index=False)
    write_json(out / 'preprocessing_audit.json', features)
    summary = {'stage': STAGE, 'contract_version': VERSION, 'experiment_id': identity, 'upstream_resource_id': upstream['resource_id'],
        'created_utc': datetime.now(timezone.utc).isoformat(), 'matches': len(splits),
        'events': int(splits.source_events.sum()), 'development_events': len(data),
        'arms': list(ARM_CODES), 'tracks': sorted(s.track.unique()), 'candidate_fits': len(c),
        'saved_family_models': len(f), 'selected_arm_models': len(s), 'test_predictions': 0,
        'test_scored': False, 'fit_scope': 'train only; no train+validation refit',
        'preflight_passed_before_training': True, 'learned_candidate_fits': int(c.family.ne('constant').sum()),
        'constant_fits': int(c.family.eq('constant').sum()),
        'preferred_arm': PREFERRED_ARM, 'selection_policy': SELECTION_POLICY,
        'selection': SELECTION_DESCRIPTION,
        'constant': 'training-event outcome prevalence per gender/innings; not Naive Bayes',
        'histories': 'prior-date rolling histories shared across arms; not frozen at train boundary',
        'interpretation': 'Exploratory validation comparisons, not final evidence of improvement or calibration.',
        'limitations': ['Conservative resource-ready cohort; review matches excluded identically from all arms.',
            'Previously inspected historical matches are not a new untouched confirmatory test set.',
            'Separate innings models share a target but need not be continuous across the innings break.',
            'Within-family comparisons tune each arm; not a pure fixed-hyperparameter feature effect.',
            'No post-hoc calibration fitted. Log loss/Brier improvements are not pure calibration improvements.']}
    write_json(out / 'comparison_summary.json', summary)
    paths = [out / (n + '.csv') for n in ['split_summary', 'candidate_metrics', 'family_winners', 'selected_models', 'selected_systems', 'validation_deltas']]
    paths += [out / (n + '.json') for n in ['experiment_config', 'feature_groups', 'preprocessing_audit', 'comparison_summary', 'preflight_audit']]
    paths += [split_path, processed / 'validation_predictions.csv.gz'] + [out / 'models' / n for n in f.model_file]
    final_inputs, _ = input_fingerprints(root)
    if final_inputs != inputs:
        raise ValueError('Inputs/code/runtime changed during training. Outputs remain incomplete; rerun explicitly.')
    write_json(out / 'manifest.json', {'status': 'complete', 'contract_version': VERSION, 'experiment_id': identity, 'inputs': inputs,
        'preferred_arm': PREFERRED_ARM, 'selection_policy': SELECTION_POLICY,
        'output_sha256': {p.relative_to(root).as_posix(): sha256(p) for p in paths}})
    return summary


def verified_output(root):
    root = Path(root)
    path = root / REPORT_DIR / 'manifest.json'
    if not path.is_file():
        raise FileNotFoundError('No Stage 06 results. Set RUN_STAGE = True for the first full run. No automatic training.')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    inputs, _ = input_fingerprints(root)
    if (manifest.get('status') != 'complete' or manifest.get('inputs') != inputs or manifest.get('contract_version') != VERSION
            or manifest.get('preferred_arm') != PREFERRED_ARM or manifest.get('selection_policy') != SELECTION_POLICY):
        raise ValueError('Stage 06 results incomplete/stale. Rebuild explicitly; no automatic training.')
    for relative, expected in manifest['output_sha256'].items():
        if sha256(rooted(root, relative)) != expected:
            raise ValueError('Stage 06 artifact changed: ' + relative)
    out = root / REPORT_DIR
    expected = select_final_systems(pd.read_csv(out / 'selected_models.csv')).sort_values('track').reset_index(drop=True)
    actual = pd.read_csv(out / 'selected_systems.csv').sort_values('track').reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError('Main registry must exactly match the four within-RBW selections, not the supplementary winners.')
    summary = json.loads((out / 'comparison_summary.json').read_text(encoding='utf-8'))
    if summary.get('preferred_arm') != PREFERRED_ARM or summary.get('selection_policy') != SELECTION_POLICY:
        raise ValueError('Summary does not describe the fixed-RBW main-model policy.')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--train', action='store_true', help='Explicitly run the full training grid.')
    args = parser.parse_args()
    print(json.dumps(run(args.root) if args.train else verified_output(args.root), indent=2))
