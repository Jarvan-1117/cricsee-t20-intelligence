"""Stage 07: symmetric batting/bowling progress interactions, validation only.

The frozen Stage 06 RBW controls and historical extension engines are read-only.
This module creates a separate registry for the next Stage 08; it never activates
models in existing Stage 08/09 artifacts.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import resource_extension_common as common
from data_preparation import rooted, sha256, write_json

base, resources = common.base, common.resources
VERSION = common.VERSION
DESIGN = 'four_way_interactions_v1'
CONFIG_PATH = 'config/resource_interaction_experiments.json'
SOURCE_PATH = 'src/resource_interaction_experiments.py'
REPORT_DIR = 'outputs/07_resource_representation_experiments/' + VERSION + '/' + DESIGN
DATA_DIR = 'data/processed/resource_experiments/' + VERSION + '/' + DESIGN
TRACKS = common.TRACKS
CODES = ['RBW', 'IB', 'IW', 'IBW']
ARM_NAMES = dict(RBW='both_resources', IB='batting_progress_interactions',
                 IW='bowling_progress_interactions', IBW='both_progress_interactions')
LABELS = dict(RBW='No added interaction', IB='Batting interaction',
              IW='Bowling interaction', IBW='Both interactions')
EXPECTED_CONFIG = dict(stage='07_resource_representation_experiments', contract_version=VERSION,
    design_version=DESIGN, arms=CODES, progress='scheduled_balls_remaining_estimate / 120',
    retain_original_resources=True, selection_metric='validation_event_log_loss',
    tie_break=CODES, candidate_grid='inherit frozen Stage 06', fit_scope='train only',
    test_scored=False, downstream_activation='requires separate Stage 08 migration')
LIMITATIONS = [
    'Exploratory representation comparisons; independent validation is needed for stronger generalization claims.',
    'Validation-selected gains are subject to selection uncertainty, not independent confirmation.',
    'Remaining-ball fraction is nominal pre-event progress, not actual future innings length.',
    'No explicit products does not mean no learned interactions: tree models can learn them.',
    'Matched parameters do not hold learned preprocessing, coefficients or trees fixed.',
    'Resources remain conditioned on historical quality; these comparisons are not causal attribution.',
    'No test metrics, calibration fitting, NWC or automatic downstream activation in this stage.']


def read_config(root):
    config = json.loads((Path(root) / CONFIG_PATH).read_text(encoding='utf-8'))
    if config != EXPECTED_CONFIG:
        raise ValueError('Four-way interaction policy differs from the frozen design.')
    return config


def raw_features(code):
    if code not in CODES:
        raise ValueError('Unknown interaction arm: ' + code)
    return (list(resources.BAT_FEATURES) if code in ['IB', 'IBW'] else []) + (
            list(resources.BOWL_FEATURES) if code in ['IW', 'IBW'] else [])


def interaction_features(code):
    return ['progress_interaction_' + name for name in raw_features(code)]


class Extension:
    """Adapter to the tested development-only engine without modifying old engines."""
    CONFIG_PATH, SOURCE_PATH = CONFIG_PATH, SOURCE_PATH
    EXTRA_INPUT_PATHS = []
    REFERENCES = ['both_resources']
    LIMITATIONS = LIMITATIONS

    def __init__(self, code):
        if code not in CODES[1:]:
            raise ValueError('Only added interaction arms require fitting.')
        self.CODE, self.ARM = code, ARM_NAMES[code]
        self.FEATURES = interaction_features(code)
        self.REPORT_DIR = REPORT_DIR + '/' + code.lower()
        self.DATA_DIR = DATA_DIR + '/' + code.lower()

    def read_config(self, root):
        return dict(design=read_config(root), arm=self.CODE)

    def resolve_columns(self, arms):
        return arms['both_resources'] + self.FEATURES

    def validate_features(self, data):
        progress = data.scheduled_balls_remaining_estimate.to_numpy(dtype=float)
        raw = data[raw_features(self.CODE)].to_numpy(dtype=float)
        actual = data[self.FEATURES].to_numpy(dtype=float)
        if not np.isfinite(progress).all() or ((progress < 0) | (progress > 120)).any():
            raise ValueError('Progress must use finite pre-event nominal balls in [0,120].')
        if (not np.isfinite(raw).all() or (raw < 0).any() or not np.isfinite(actual).all()
                or not np.allclose(actual, raw * progress[:, None] / 120., atol=1e-12, rtol=1e-12)):
            raise ValueError('Resource interaction reconstruction failed.')

    def build_features(self, root, data):
        extra = data[base.KEYS].copy()
        for original, feature in zip(raw_features(self.CODE), self.FEATURES):
            extra[feature] = data[original] * data.scheduled_balls_remaining_estimate / 120.
        joined = common.join_extra(data, extra, self.FEATURES)
        self.validate_features(joined)
        audit = joined.groupby(['gender', 'match_innings', 'split']).size().reset_index(name='events')
        audit['max_reconstruction_error'] = 0.
        return joined, audit


def input_fingerprints(root):
    inputs, upstream = common.input_fingerprints(Path(root), Extension('IB'))
    return inputs, upstream


def preflight(root):
    root = Path(root)
    p = common.prepare(root, Extension('IB'))
    raw = p['data'].drop(columns=interaction_features('IB'))
    arms = base.resolve_arms(resources.feature_contract(base.game_state.feature_contract()))
    rows = []
    for code in CODES:
        if code == 'RBW':
            data, columns = raw, arms['both_resources']
        else:
            ext = Extension(code)
            data, _ = ext.build_features(root, raw)
            columns = ext.resolve_columns(arms)
            common.validate_development(data, columns, ext)
        for (gender, innings, split), group in data.groupby(['gender', 'match_innings', 'split']):
            rows.append(dict(arm_code=code, track=f'{gender}_innings_{innings}', split=split,
                events=len(group), matches=int(group.match_id.nunique()),
                event_keys_sha256=hashlib.sha256(group[base.KEYS].to_csv(index=False).encode()).hexdigest(),
                input_columns=base.track_columns(columns, innings)))
    audit = pd.DataFrame(rows)
    if (audit.groupby(['track', 'split']).event_keys_sha256.nunique() != 1).any():
        raise ValueError('Interaction groups do not share exactly the same events.')
    return dict(status='passed', design_version=DESIGN, input_fingerprints=p['inputs'],
        upstream_stage06_id=p['upstream']['experiment_id'],
        development_matches=int(raw.match_id.nunique()), development_events=len(raw),
        expected_new_candidate_fits=3 * 4 * sum(len(p['config']['candidate_grid'][f]) for f in base.FAMILIES),
        added_features={code: interaction_features(code) for code in CODES},
        shared_rows_and_dates=True, direct_current_player_quality=False,
        training_fits=0, test_scored=False, rows=rows)


def select_systems(selected):
    """Validation-only choice among four representations; ties favor fewer changes."""
    if (set(selected.track) != set(TRACKS) or len(selected) != 16
            or selected.duplicated(['track', 'arm_code']).any()
            or any(set(g.arm_code) != set(CODES) for _, g in selected.groupby('track'))):
        raise ValueError('Expected four complete representations in every track.')
    if not np.isfinite(selected.validation_event_log_loss).all():
        raise ValueError('Nonfinite selection metric.')
    chosen = selected.assign(_arm_order=selected.arm_code.map({c: i for i, c in enumerate(CODES)}))
    return (chosen.sort_values(['track', 'validation_event_log_loss', '_arm_order', 'candidate'])
            .groupby('track', sort=False).head(1).drop(columns='_arm_order').reset_index(drop=True))


def baseline_model_path(row):
    return base.REPORT_DIR + '/models/' + row['model_file']


def summarise(root, checks):
    root = Path(root)
    out, processed = root / REPORT_DIR, root / DATA_DIR
    out.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(root / base.REPORT_DIR / 'selected_models.csv')
    baseline = baseline[baseline.arm_code.eq('RBW')].copy()
    baseline['model_path'] = baseline.apply(baseline_model_path, axis=1)
    selected, predictions, comparisons, phases = [baseline], [], [], []
    extension_ids = {}
    for code in CODES[1:]:
        ext = Extension(code)
        summary = common.verified_output(root, ext)
        extension_ids[code] = summary['experiment_id']
        table = pd.read_csv(root / ext.REPORT_DIR / 'selected_models.csv')
        table['model_path'] = ext.REPORT_DIR + '/models/' + table.model_file
        selected.append(table)
        pred = pd.read_csv(root / ext.DATA_DIR / 'selected_validation_predictions.csv.gz')
        phase = pd.read_csv(root / ext.REPORT_DIR / 'phase_diagnostics.csv')
        if code != 'IB':
            pred = pred[pred.arm.eq(ext.ARM)]
            phase = phase[phase.arm.eq(ext.ARM)]
        predictions.append(pred)
        phases.append(phase)
        comparisons.append(pd.read_csv(root / ext.REPORT_DIR / 'validation_comparisons.csv'))
    selected = pd.concat(selected, ignore_index=True)
    selected['model_sha256'] = selected.model_path.map(lambda p: sha256(rooted(root, p)))
    systems = select_systems(selected)
    pred = pd.concat(predictions, ignore_index=True)
    pred['arm_code'] = pred.arm.map({v: k for k, v in ARM_NAMES.items()})
    if pred.arm_code.isna().any() or pred.duplicated(['track', 'arm_code'] + base.KEYS).any():
        raise ValueError('Invalid combined validation coverage.')
    config = base.read_config(root)
    for track, group in pred.groupby('track'):
        reference = group[group.arm_code.eq('RBW')].set_index(base.KEYS).sort_index()
        for code in CODES:
            part = group[group.arm_code.eq(code)].set_index(base.KEYS).sort_index()
            if (not part.index.equals(reference.index)
                    or not part[[base.TARGET, 'match_date', 'gender']].equals(reference[[base.TARGET, 'match_date', 'gender']])):
                raise ValueError('Combined validation coverage/labels differ.')
            record = selected[selected.track.eq(track) & selected.arm_code.eq(code)].iloc[0]
            for key, value in base.metrics(part[base.TARGET], part.raw_model_probability,
                                          part.index.get_level_values('match_id'), config['probability_epsilon']).items():
                if value is not None and not np.isclose(value, record[key], atol=1e-12, rtol=1e-10):
                    raise ValueError('Combined validation metrics do not reproduce.')
    tables = dict(selected_models=selected, selected_systems=systems,
        validation_comparisons=pd.concat(comparisons, ignore_index=True),
        phase_diagnostics=pd.concat(phases, ignore_index=True),
        input_audit=pd.DataFrame(checks['rows']))
    for name, table in tables.items():
        table.to_csv(out / (name + '.csv'), index=False)
    pred.to_csv(processed / 'selected_validation_predictions.csv.gz', index=False, compression='gzip')
    arms = base.resolve_arms(resources.feature_contract(base.game_state.feature_contract()))
    feature_columns = {code: arms['both_resources'] + interaction_features(code) for code in CODES}
    registry = []
    for row in systems.to_dict('records'):
        row['input_columns'] = base.track_columns(feature_columns[row['arm_code']], int(row['track'][-1]))
        row['parameters'] = json.loads(row['parameters'])
        registry.append(row)
    inputs, upstream = input_fingerprints(root)
    if inputs != checks['input_fingerprints']:
        raise ValueError('Inputs changed during the interaction experiment.')
    identity = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    summary = dict(status='complete', stage='07_resource_representation_experiments',
        contract_version=VERSION, design_version=DESIGN, experiment_id=identity,
        created_utc=datetime.now(timezone.utc).isoformat(), upstream_stage06_id=upstream['experiment_id'],
        arms=CODES, new_candidate_fits=checks['expected_new_candidate_fits'],
        reused_rbw_selected_models=4, selected_models=16, downstream_candidates=4,
        development_matches=checks['development_matches'], development_events=checks['development_events'],
        validation_prediction_rows=len(pred), test_predictions=0, test_scored=False,
        calibration_fits=0, downstream_activated=False, extension_ids=extension_ids, limitations=LIMITATIONS)
    payloads = dict(experiment_summary=summary, preflight=checks, feature_groups=feature_columns,
        frozen_protocol=dict(design=read_config(root), inherited_stage06_config=config,
            fit_scope='train only', selection_metric='validation_event_log_loss',
            exact_tie_order=CODES, models=registry, activation_status='awaiting_stage08_migration'))
    for name, payload in payloads.items():
        write_json(out / (name + '.json'), payload)
    paths = [out / (name + '.csv') for name in tables] + [out / (name + '.json') for name in payloads]
    paths += [processed / 'selected_validation_predictions.csv.gz']
    paths += [root / Extension(code).REPORT_DIR / 'manifest.json' for code in CODES[1:]]
    write_json(out / 'manifest.json', dict(status='complete', experiment_id=identity, inputs=inputs,
        output_sha256={p.relative_to(root).as_posix(): sha256(p) for p in paths}))
    return summary


def run(root, resume=False):
    root = Path(root).resolve()
    checks = preflight(root)
    out = root / REPORT_DIR
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'manifest.json', dict(status='running', design_version=DESIGN))
    write_json(out / 'preflight.json', checks)
    print('Four-way preflight passed. Same samples/dates/grid; no direct current-player quality. No test scoring.', flush=True)
    for code in CODES[1:]:
        ext = Extension(code)
        if resume and (root / ext.REPORT_DIR / 'manifest.json').is_file():
            try:
                common.verified_output(root, ext)
            except (ValueError, FileNotFoundError):
                print('Incomplete or stale arm will be explicitly retrained:', code, flush=True)
            else:
                print('Reusing verified completed arm:', code, flush=True)
                continue
        print('Training representation:', code, flush=True)
        common.run(root, ext)
    return summarise(root, checks)


def verified_output(root):
    root = Path(root)
    path = root / REPORT_DIR / 'manifest.json'
    if not path.is_file():
        raise FileNotFoundError('No four-way Stage 07 results. Set RUN_STAGE=True explicitly to train.')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    inputs, upstream = input_fingerprints(root)
    identity = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    if manifest.get('status') != 'complete' or manifest.get('inputs') != inputs or manifest.get('experiment_id') != identity:
        raise ValueError('Four-way Stage 07 results are incomplete/stale; no automatic training.')
    for relative, expected in manifest['output_sha256'].items():
        if sha256(rooted(root, relative)) != expected:
            raise ValueError('Interaction artifact changed: ' + relative)
    for code in CODES[1:]:
        common.verified_output(root, Extension(code))
    summary = json.loads((root / REPORT_DIR / 'experiment_summary.json').read_text(encoding='utf-8'))
    if summary['upstream_stage06_id'] != upstream['experiment_id'] or summary['experiment_id'] != identity:
        raise ValueError('Four-way provenance mismatch.')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Reuse only verified complete arms during explicit training')
    args = parser.parse_args()
    if args.resume and not args.train:
        parser.error('--resume requires --train')
    result = run(args.root, args.resume) if args.train else preflight(args.root) if args.preflight else verified_output(args.root)
    print(json.dumps(result, indent=2))
