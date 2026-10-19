"""Stage 08: fixed IBW final models, with eight-arm evaluation and model reliance.

No training, calibration fitting, test-driven selection or NWC.
Shared numerical helpers are in probability_evaluation_common.py.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

import model_comparison as base
import resource_interaction_experiments as interactions
import probability_evaluation_common as common
from data_preparation import rooted, sha256, write_json

STAGE = '08_probability_evaluation'
VERSION = 'resource_sums_v2_fixed_ibw_v1'
POLICY = 'fixed_ibw_within_arm_validation_log_loss_v1'
PREFERRED_ARM = 'IBW'
REPORT_DIR = 'outputs/' + STAGE + '/' + VERSION
DATA_DIR = 'data/processed/probability_evaluation/' + VERSION
CONFIG_PATH = 'config/probability_evaluation_fixed_ibw.json'
PROTOCOL_PATH = 'docs/stage08_fixed_ibw_protocol.md'
TRACKS = ['female_innings_1', 'female_innings_2', 'male_innings_1', 'male_innings_2']
ARM_ORDER = ['R0', 'R1', 'RB', 'RW', 'RBW', 'IB', 'IW', 'IBW']
PAIRS = [['RBW', 'R1'], ['RB', 'R1'], ['RW', 'R1'], ['RBW', 'RW'],
         ['RBW', 'RB'], ['R1', 'R0']] + [[arm, ref] for ref in ['RBW', 'R1'] for arm in ['IB', 'IW', 'IBW']]
INTERVAL_METRICS = ['event_log_loss', 'match_log_loss', 'brier', 'match_brier']
calibration_table = common.calibration_table
match_losses = common.match_losses
draw_match_counts = common.draw_match_counts
paired_gain = common.paired_gain
event_log_loss = common.event_log_loss
permuted_input = common.permuted_input


def expected_config():
    return dict(stage=STAGE, contract_version=VERSION, target=base.TARGET,
        resource_contract_version=base.VERSION, arms=list(ARM_ORDER), selection_policy=POLICY,
        selection_metric='validation_event_log_loss', tie_break=list(ARM_ORDER),
        comparisons=[list(pair) for pair in PAIRS], preferred_arm=PREFERRED_ARM,
        selection_scope='IBW only; inherit Stage 07 within-arm algorithm and parameter selection',
        primary_pair=['RBW', 'R1'], calibration_bins=10, bootstrap_replicates=2000,
        bootstrap_seed=32933, confidence_level=.95, probability_epsilon=1e-15,
        fit_models=False, fit_calibrators=False, refit_train_validation=False,
        scope='frozen chronological test split; same event cohort for all arms',
        bootstrap_unit='paired whole match within gender; same draws for both innings and all arms',
        interval_policy='pointwise percentile intervals; not multiplicity-adjusted',
        importance=dict(method='marginal permutation of raw input columns',
            models='four fixed-IBW models with validation-selected algorithms and parameters',
            dataset='complete frozen test split per track', metric='event_log_loss_increase',
            repeats=10, seed=32934,
            units='each actual model input and nonempty disjoint state/resource/interaction blocks',
            permutation='within track across events; shared row permutation within each block',
            uncertainty='sample standard deviation across permutations; not a confidence interval',
            purpose='descriptive model reliance only; no feature selection or model promotion',
            constant_model='not applicable: no input features; never fabricate zero feature scores'))


def read_config(root):
    c = json.loads((Path(root)/CONFIG_PATH).read_text(encoding='utf-8'))
    if c != expected_config():
        raise ValueError('Stage 08 fixed-IBW protocol changed; review before evaluation.')
    return c


def input_fingerprints(root):
    root = Path(root)
    upstream = {'stage06': base.verified_output(root), 'stage07': interactions.verified_output(root)}
    if upstream['stage07']['upstream_stage06_id'] != upstream['stage06']['experiment_id']:
        raise ValueError('Stage 07 does not reference the current Stage 06 experiment.')
    paths = [CONFIG_PATH, PROTOCOL_PATH, 'src/probability_evaluation_fixed_ibw.py',
             'src/probability_evaluation_common.py', 'src/resource_interaction_experiments.py',
             base.REPORT_DIR+'/manifest.json', interactions.REPORT_DIR+'/manifest.json',
             base.DATA_DIR+'/match_splits.csv', base.resources.REPORT_DIR+'/feature_contract.json']
    return {p: sha256(root/p) for p in paths}, upstream


def feature_columns():
    groups = base.resolve_arms(base.resources.feature_contract(base.game_state.feature_contract()))
    columns = {base.ARM_CODES[arm]: cols for arm, cols in groups.items()}
    for code in interactions.CODES[1:]:
        columns[code] = interactions.Extension(code).resolve_columns(groups)
    return columns


def select_preferred(registry):
    """Fix the feature group to IBW; upstream validation chose its algorithm/parameters."""
    preferred = {}
    for track in TRACKS:
        rows = [r for r in registry if r['track'] == track]
        if len(rows) != len(ARM_ORDER) or {r['arm_code'] for r in rows} != set(ARM_ORDER):
            raise ValueError('Expected eight distinct candidates in every track.')
        if any(not np.isfinite(r['validation_event_log_loss']) for r in rows):
            raise ValueError('Nonfinite validation selection score.')
        winner = next(r for r in rows if r['arm_code'] == PREFERRED_ARM)
        preferred[track] = winner['arm_code']
    return preferred


def build_registry(root, config):
    """Merge upstream within-arm winners, deduplicating the shared RBW control."""
    root = Path(root)
    columns = feature_columns()
    a = pd.read_csv(root/base.REPORT_DIR/'selected_models.csv')
    b = pd.read_csv(root/interactions.REPORT_DIR/'selected_models.csv')
    for table, codes in [(a, ARM_ORDER[:5]), (b, interactions.CODES)]:
        if (len(table) != 4*len(codes) or table.duplicated(['track','arm_code']).any()
                or set(zip(table.track, table.arm_code)) != {(t,c) for t in TRACKS for c in codes}):
            raise ValueError('Incomplete upstream within-arm winner table.')
    for track in TRACKS:
        left = a[a.track.eq(track) & a.arm_code.eq('RBW')].iloc[0]
        right = b[b.track.eq(track) & b.arm_code.eq('RBW')].iloc[0]
        path = base.REPORT_DIR + '/models/' + left.model_file
        if (left.family != right.family or int(left.candidate) != int(right.candidate)
                or json.loads(left.parameters) != json.loads(right.parameters)
                or left.validation_event_log_loss != right.validation_event_log_loss
                or right.model_path != path or right.model_sha256 != sha256(rooted(root,path))):
            raise ValueError('Stage 07 RBW control differs from Stage 06.')
    rows = []
    for source, table in [('06', a), ('07', b[b.arm_code.ne('RBW')])]:
        for row in table.itertuples(index=False):
            path = (base.REPORT_DIR+'/models/'+row.model_file) if source == '06' else row.model_path
            features = base.track_columns(columns[row.arm_code], int(row.track.rsplit('_',1)[1]))
            rows.append(dict(track=row.track, arm_code=row.arm_code, arm=row.arm,
                family=row.family, candidate=int(row.candidate), parameters=json.loads(row.parameters),
                validation_event_log_loss=float(row.validation_event_log_loss),
                model_path=path, model_sha256=sha256(rooted(root,path)), features=features,
                source_stage=source))
    preferred = select_preferred(rows)
    for row in rows: row['preferred'] = preferred[row['track']] == row['arm_code']
    rows.sort(key=lambda r: (TRACKS.index(r['track']), ARM_ORDER.index(r['arm_code'])))
    validate_registry(rows, config)
    return rows


def validate_registry(registry, config):
    keys = [(r['track'],r['arm_code']) for r in registry]
    if len(keys) != 32 or set(keys) != {(t,a) for t in TRACKS for a in ARM_ORDER}:
        raise ValueError('Expected exactly 32 unique frozen track/arm systems.')
    preferred = select_preferred(registry)
    columns = feature_columns()
    for row in registry:
        if row['preferred'] != (preferred[row['track']] == row['arm_code']):
            raise ValueError('Preferred flag differs from the fixed-IBW policy.')
        expected = base.track_columns(columns[row['arm_code']], int(row['track'].rsplit('_',1)[1]))
        if row['features'] != expected:
            raise ValueError('Features differ from the exact resource/interaction allowlist.')
        if row['source_stage'] != ('06' if row['arm_code'] in ARM_ORDER[:5] else '07'):
            raise ValueError('Unexpected source stage.')
    if (config['selection_policy'] != POLICY or config['tie_break'] != ARM_ORDER
            or config.get('preferred_arm') != PREFERRED_ARM):
        raise ValueError('Unsupported selection policy.')


def prepare_registry(root):
    config = read_config(root)
    inputs, upstream = input_fingerprints(root)
    return config, inputs, upstream, build_registry(root, config)


def load_test_data(root, registry):
    root = Path(root)
    splits = pd.read_csv(root/base.DATA_DIR/'match_splits.csv')
    if splits.match_id.duplicated().any() or set(splits.split) != set(base.SPLITS):
        raise ValueError('Invalid frozen split.')
    derived = interactions.interaction_features('IBW')
    columns = list(dict.fromkeys(base.META + [base.TARGET,'innings_phase'] + base.CHASE +
        [c for row in registry for c in row['features'] if c not in derived]))
    ids = set(splits.loc[splits.split.eq('test'),'match_id'])
    parts = [chunk[chunk.match_id.isin(ids)].copy() for chunk in pd.read_csv(
        root/base.resources.DATA_DIR/'model_input.csv.gz', usecols=columns, chunksize=50000)]
    data = pd.concat(parts, ignore_index=True)
    data['split'] = 'test'
    data, _ = interactions.Extension('IBW').build_features(root, data)
    common.validate_test_data(data, splits, registry)
    return data, splits


def validate_predictions(predictions, data):
    if predictions.duplicated(['track','arm_code']+base.KEYS).any() or len(predictions) != len(data)*8:
        raise ValueError('Incomplete or duplicate eight-arm prediction coverage.')
    p = predictions.raw_model_probability.to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError('Invalid probability.')
    for track in TRACKS:
        gender, _, innings = track.split('_')
        expected = data[data.gender.eq(gender)&data.match_innings.eq(int(innings))].set_index(base.KEYS).sort_index()
        group = predictions[predictions.track.eq(track)]
        if set(group.arm_code) != set(ARM_ORDER): raise ValueError('Missing prediction arm.')
        preferred = []
        for arm, part in group.groupby('arm_code'):
            actual = part.set_index(base.KEYS).sort_index()
            if (not actual.index.equals(expected.index) or
                not actual[[base.TARGET,'gender','match_date','innings_phase']].equals(
                    expected[[base.TARGET,'gender','match_date','innings_phase']])):
                raise ValueError('Prediction coverage/labels differ across arms.')
            if part.preferred.nunique() != 1: raise ValueError('Inconsistent preferred flags.')
            if part.preferred.iloc[0]: preferred.append(arm)
        if preferred != [PREFERRED_ARM]: raise ValueError('Exactly one IBW preferred model per track required.')


def predict_frozen(root, data, registry, progress=print):
    frames = []
    for row in registry:
        gender, _, innings = row['track'].split('_')
        part = data[data.gender.eq(gender)&data.match_innings.eq(int(innings))]
        path = rooted(root,row['model_path'])
        if sha256(path) != row['model_sha256']: raise ValueError('Model changed after freeze.')
        model = joblib.load(path)
        if row['features']:
            if list(model.feature_names_in_) != row['features']: raise ValueError('Pipeline feature order differs.')
            x = part[row['features']]
        else:
            if row['arm_code'] != 'R0': raise ValueError('Only R0 may have no features.')
            x = np.zeros((len(part),1))
        with threadpool_limits(limits=2): p = base.probability(model,x)
        frame = part[base.META+[base.TARGET,'innings_phase']].copy()
        for key in ['track','arm_code','family','preferred']: frame[key] = row[key]
        frame['raw_model_probability'] = p
        frames.append(frame)
        progress('Evaluated frozen '+row['track']+' | '+row['arm_code'])
    predictions = pd.concat(frames,ignore_index=True)
    validate_predictions(predictions,data)
    return predictions


def evaluate_metrics(predictions, config):
    metrics, bins = common.evaluate_metrics(predictions,config)
    f1 = {(t,a): f1_score(g[base.TARGET],g.raw_model_probability.ge(.5),zero_division=0)
          for (t,a),g in predictions.groupby(['track','arm_code'])}
    metrics['test_f1'] = [f1[(r.track,r.arm_code)] for r in metrics.itertuples()]
    return metrics,bins


def bootstrap_comparisons(predictions, config):
    losses = match_losses(predictions,config['probability_epsilon'])
    rows = []
    for gender_index, gender in enumerate(['female','male']):
        ids = sorted(losses.loc[losses.gender.eq(gender),'match_id'].unique())
        counts = draw_match_counts(len(ids),config['bootstrap_replicates'],config['bootstrap_seed']+gender_index)
        for innings in [1,2]:
            track = f'{gender}_innings_{innings}'
            tables = {}
            for arm in ARM_ORDER:
                part = losses[losses.track.eq(track)&losses.arm_code.eq(arm)].set_index('match_id')
                if set(part.index)!=set(ids) or part.index.duplicated().any():
                    raise ValueError('Bootstrap match coverage differs.')
                tables[arm] = part.loc[ids]
            for treatment, reference in config['comparisons']:
                gains = paired_gain(tables[reference],tables[treatment],counts)
                for metric,(point,distribution) in gains.items():
                    alpha=(1-config['confidence_level'])/2
                    lo,hi=np.quantile(distribution,[alpha,1-alpha])
                    rows.append(dict(track=track,treatment=treatment,reference=reference,
                        comparison_role='primary' if [treatment,reference]==config['primary_pair'] else 'supporting',
                        metric=metric,gain=point,ci_lower=float(lo),ci_upper=float(hi),
                        contains_zero=bool(lo<=0<=hi),matches=len(ids),
                        bootstrap_replicates=config['bootstrap_replicates']))
    return pd.DataFrame(rows),losses


def importance_units(features):
    batting,bowling=base.resources.BAT_FEATURES,base.resources.BOWL_FEATURES
    ib,iw=interactions.interaction_features('IB'),interactions.interaction_features('IW')
    blocks={'state_background':[c for c in features if c not in batting+bowling+ib+iw],
            'batting_resources':[c for c in features if c in batting],
            'bowling_resources':[c for c in features if c in bowling],
            'batting_interactions':[c for c in features if c in ib],
            'bowling_interactions':[c for c in features if c in iw]}
    return ([dict(unit_type='feature',unit=c,columns=[c]) for c in features] +
            [dict(unit_type='block',unit=name,columns=cols) for name,cols in blocks.items() if cols])


def permutation_importance(root,data,registry,predictions,config,progress=print):
    validate_registry(registry,config)
    detailed=[]
    for track_index,track in enumerate(TRACKS):
        choices=[r for r in registry if r['track']==track and r['preferred']]
        if len(choices)!=1: raise ValueError('Expected one validation-selected model.')
        row=choices[0]
        if not row['features']: continue  # Explicitly recorded as not applicable in status table.
        gender,_,innings=track.split('_')
        part=data[data.gender.eq(gender)&data.match_innings.eq(int(innings))].sort_values(base.KEYS).reset_index(drop=True)
        reference=predictions[predictions.track.eq(track)&predictions.arm_code.eq(row['arm_code'])].sort_values(base.KEYS).reset_index(drop=True)
        if not part[base.KEYS+[base.TARGET]].equals(reference[base.KEYS+[base.TARGET]]):
            raise ValueError('Importance event alignment differs.')
        path=rooted(root,row['model_path'])
        if sha256(path)!=row['model_sha256']: raise ValueError('Importance model changed.')
        model=joblib.load(path)
        if list(model.feature_names_in_)!=row['features']: raise ValueError('Importance feature order differs.')
        x=part[row['features']].copy(); y=part[base.TARGET].to_numpy()
        with threadpool_limits(limits=2): baseline_p=base.probability(model,x)
        if not np.allclose(baseline_p,reference.raw_model_probability,rtol=0,atol=1e-12):
            raise ValueError('Importance baseline predictions differ.')
        baseline=event_log_loss(y,baseline_p,config['probability_epsilon'])
        seed=config['importance']['seed']+track_index
        rng=np.random.default_rng(seed)
        permutations=[rng.permutation(len(x)) for _ in range(config['importance']['repeats'])]
        for unit in importance_units(row['features']):
            for repeat,permutation in enumerate(permutations,1):
                shuffled=permuted_input(x,unit['columns'],permutation)
                with threadpool_limits(limits=2): p=base.probability(model,shuffled)
                loss=event_log_loss(y,p,config['probability_epsilon'])
                detailed.append(dict(track=track,arm_code=row['arm_code'],family=row['family'],
                    unit_type=unit['unit_type'],unit=unit['unit'],columns=json.dumps(unit['columns']),
                    repeat=repeat,seed=seed,events=len(x),matches=int(part.match_id.nunique()),
                    baseline_log_loss=baseline,permuted_log_loss=loss,importance=loss-baseline))
            progress('Importance '+track+' | '+unit['unit'])
    detail_columns=['track','arm_code','family','unit_type','unit','columns','repeat','seed','events',
                    'matches','baseline_log_loss','permuted_log_loss','importance']
    details=pd.DataFrame(detailed,columns=detail_columns)
    keys=['track','arm_code','family','unit_type','unit','columns']
    summary=details.groupby(keys,sort=False,as_index=False).agg(
        importance_mean=('importance','mean'),importance_std=('importance','std'),
        repeats=('importance','size'),baseline_log_loss=('baseline_log_loss','first'),
        events=('events','first'),matches=('matches','first'))
    if not np.isfinite(summary[['importance_mean','importance_std']].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite importance.')
    summary['interpretation']='descriptive marginal reliance; shuffle SD, not confidence interval or causal effect'
    return summary,details


def run(root,progress=print):
    root=Path(root).resolve()
    config,inputs,upstream,registry=prepare_registry(root)
    out,processed=root/REPORT_DIR,root/DATA_DIR
    out.mkdir(parents=True,exist_ok=True); processed.mkdir(parents=True,exist_ok=True)
    identity=hashlib.sha256(json.dumps(dict(inputs=inputs,registry=registry),sort_keys=True).encode()).hexdigest()
    preferred=select_preferred(registry)
    write_json(out/'manifest.json',dict(status='running',evaluation_id=identity))
    write_json(out/'frozen_protocol.json',dict(evaluation_id=identity,frozen_utc=datetime.now(timezone.utc).isoformat(),
        config=config,inputs=inputs,upstream_ids={k:v['experiment_id'] for k,v in upstream.items()},
        models=registry,preferred=preferred,selection_source=POLICY,
        importance_units={r['track']:importance_units(r['features']) for r in registry if r['preferred']}))
    progress('Four fixed-IBW selections frozen. Reading test data; no fitting or reselection.')
    data,splits=load_test_data(root,registry)
    predictions=predict_frozen(root,data,registry,progress)
    metrics,bins=evaluate_metrics(predictions,config)
    with threadpool_limits(limits=2): comparisons,losses=bootstrap_comparisons(predictions,config)
    importance,repeats=permutation_importance(root,data,registry,predictions,config,progress)
    preferred_predictions=predictions[predictions.preferred].copy()
    if len(preferred_predictions)!=len(data) or preferred_predictions.duplicated(base.KEYS).any():
        raise ValueError('Handoff must cover every test event exactly once.')
    selected=pd.DataFrame([r for r in registry if r['preferred']])
    status=selected[['track','arm_code','family']].copy()
    status['status']=['evaluated' if r['features'] else 'not_applicable_constant_model' for r in registry if r['preferred']]
    tables=dict(test_metrics=metrics,calibration_bins=bins,paired_comparisons=comparisons,
        preferred_model_metrics=metrics[metrics.preferred],selected_systems=selected,
        feature_importance=importance,importance_status=status,split_summary=base.split_summary(splits))
    for name,table in tables.items(): table.to_csv(out/(name+'.csv'),index=False)
    for name,table in dict(test_predictions=predictions,preferred_test_predictions=preferred_predictions,
                           match_losses=losses,importance_repeats=repeats).items():
        table.to_csv(processed/(name+'.csv.gz'),index=False,compression='gzip')
    summary=dict(stage=STAGE,contract_version=VERSION,resource_contract_version=base.VERSION,
        evaluation_id=identity,created_utc=datetime.now(timezone.utc).isoformat(),selection_policy=POLICY,
        upstream_stage06_id=upstream['stage06']['experiment_id'],upstream_stage07_id=upstream['stage07']['experiment_id'],
        models_evaluated=len(registry),test_matches=int(data.match_id.nunique()),test_events=len(data),
        prediction_rows=len(predictions),preferred_prediction_rows=len(preferred_predictions),preferred=preferred,
        training_fits=0,calibration_fits=0,comparison_rows=len(comparisons),
        bootstrap_replicates=config['bootstrap_replicates'],importance_rows=len(importance),
        importance_models=int(status.status.eq('evaluated').sum()),importance_repeat_rows=len(repeats),
        limitations=['Exploratory model comparisons; validation selection is not independent confirmation.',
            'No test-driven selection, refitting, calibration fitting, terminal override or NWC.',
            'Whole-match intervals are pointwise, condition on fitted models and omit selection uncertainty.',
            'Match bootstrap does not remove temporal or team dependence.',
            'Log loss and Brier do not isolate calibration; fixed-bin ECE depends on binning and sample size.',
            'Importance shuffles actual input columns, including products separately: correlations and inconsistent products may distort interpretation.',
            'Permutation importance is descriptive, not causal; shuffle SD is not a sampling confidence interval.',
            'All four final models use IBW; other arms are controls, not competing final feature-group choices.',
            'Stage 09 must explicitly migrate to the new selection contract before consuming this handoff.'])
    write_json(out/'evaluation_summary.json',summary)
    write_json(out/'nwc_handoff.json',dict(evaluation_id=identity,contract_version=VERSION,selection_policy=POLICY,
        status='requires_stage09_migration',prediction_timing='pre-event',target=base.TARGET,
        probability_column='raw_model_probability',preferred_models=[{k:r[k] for k in
        ['track','arm_code','family','model_path','model_sha256','features']} for r in registry if r['preferred']],
        prediction_path=DATA_DIR+'/preferred_test_predictions.csv.gz'))
    final_inputs,_=input_fingerprints(root)
    if final_inputs!=inputs or any(sha256(rooted(root,r['model_path']))!=r['model_sha256'] for r in registry):
        raise ValueError('Upstream inputs changed during evaluation; output remains incomplete.')
    paths=[out/(name+'.csv') for name in tables]
    paths += [out/(name+'.json') for name in ['frozen_protocol','evaluation_summary','nwc_handoff']]
    paths += [processed/(name+'.csv.gz') for name in ['test_predictions','preferred_test_predictions','match_losses','importance_repeats']]
    write_json(out/'manifest.json',dict(status='complete',evaluation_id=identity,inputs=inputs,
        output_sha256={p.relative_to(root).as_posix():sha256(p) for p in paths}))
    return summary


def verified_output(root):
    root=Path(root); path=root/REPORT_DIR/'manifest.json'
    if not path.is_file(): raise FileNotFoundError('No fixed-IBW Stage 08 results. Set RUN_EVALUATION = True explicitly.')
    manifest=json.loads(path.read_text(encoding='utf-8'))
    inputs,_=input_fingerprints(root)
    if manifest.get('status')!='complete' or manifest.get('inputs')!=inputs:
        raise ValueError('Fixed-IBW Stage 08 results stale/incomplete; no automatic evaluation.')
    for relative,expected in manifest['output_sha256'].items():
        if sha256(rooted(root,relative))!=expected: raise ValueError('Stage 08 artifact changed: '+relative)
    config=read_config(root)
    protocol=json.loads((root/REPORT_DIR/'frozen_protocol.json').read_text(encoding='utf-8'))
    registry=build_registry(root,config)
    identity=hashlib.sha256(json.dumps(dict(inputs=inputs,registry=registry),sort_keys=True).encode()).hexdigest()
    if (protocol['config']!=config or protocol['models']!=registry or protocol['preferred']!=select_preferred(registry)
            or protocol['evaluation_id']!=identity or manifest['evaluation_id']!=identity):
        raise ValueError('Frozen protocol differs from fixed-IBW within-arm validation selection.')
    summary=json.loads((root/REPORT_DIR/'evaluation_summary.json').read_text(encoding='utf-8'))
    if (summary['preferred']!=select_preferred(registry) or summary['models_evaluated']!=32
            or summary['selection_policy']!=POLICY or summary['evaluation_id']!=identity):
        raise ValueError('Fixed-IBW Stage 08 summary mismatch.')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--evaluate',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.root) if args.evaluate else verified_output(args.root),indent=2))
