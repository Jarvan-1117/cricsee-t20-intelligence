"""Shared probability metrics, calibration, bootstrap and input validation.

Pure evaluation helpers: no model selection, file exports or execution entry point.
"""
import numpy as np
import pandas as pd
import model_comparison as base


def validate_test_data(data, splits, registry):
    test = splits[splits.split.eq('test')].set_index('match_id')
    if data.empty or data.duplicated(base.KEYS).any() or data[base.META].isna().any().any():
        raise ValueError('Empty/duplicate/missing test event identifiers.')
    if not data.match_id.isin(test.index).all() or set(data.split) != {'test'}:
        raise ValueError('Evaluation input contains non-test matches.')
    if data.groupby('match_id').size().to_dict() != test.source_events.to_dict():
        raise ValueError('Test event coverage differs from the frozen cohort.')
    if not data.gender.eq(data.match_id.map(test.gender)).all() or not data.match_date.eq(data.match_id.map(test.match_date)).all():
        raise ValueError('Test metadata differs from the frozen split.')
    if not data[base.TARGET].isin([0,1]).all() or data.groupby('match_id')[base.TARGET].nunique().ne(1).any():
        raise ValueError('Invalid/inconsistent test outcome.')
    if (data.groupby('match_id').match_innings.agg(set) != {1,2}).any():
        raise ValueError('Both innings required for every test match.')
    if not data.loc[data.match_innings.eq(1),base.CHASE].isna().all().all():
        raise ValueError('First-innings chase fields must remain unavailable.')
    if not data.innings_phase.isin(['powerplay','middle','death']).all():
        raise ValueError('Unknown test innings phase.')
    for row in registry:
        gender, _, innings = row['track'].split('_')
        part = data[data.gender.eq(gender) & data.match_innings.eq(int(innings))]
        if part.empty:
            raise ValueError('Empty test track.')
        numeric = [c for c in row['features'] if c != 'innings_phase']
        if not np.isfinite(part[numeric].to_numpy(dtype=float)).all():
            raise ValueError('Nonfinite test features; never drop events to repair coverage.')


def calibration_table(y, p, match_ids, bins=10):
    """Fixed-width, event-weighted calibration. Empty bins remain unavailable."""
    y, p, match_ids = np.asarray(y), np.asarray(p,dtype=float), np.asarray(match_ids)
    if len(y)==0 or len(y)!=len(p) or len(y)!=len(match_ids) or not np.isin(y,[0,1]).all():
        raise ValueError('Invalid calibration observations.')
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid calibration probability.')
    index = np.minimum((p*bins).astype(int), bins-1)
    rows = []
    for b in range(bins):
        mask = index==b; count = int(mask.sum())
        rows.append({'bin': b, 'lower': b/bins, 'upper': (b+1)/bins,
                     'events': count, 'matches': int(len(np.unique(match_ids[mask]))),
                     'mean_probability': float(p[mask].mean()) if count else np.nan,
                     'observed_win_rate': float(y[mask].mean()) if count else np.nan})
    table = pd.DataFrame(rows)
    valid = table.events.gt(0)
    ece = float(((table.loc[valid,'mean_probability']-table.loc[valid,'observed_win_rate']).abs()
                 * table.loc[valid,'events']).sum()/len(y))
    return table, ece


def evaluate_metrics(predictions, config):
    rows, calibrations = [], []
    for (track, arm), group in predictions.groupby(['track','arm_code'], sort=False):
        if group.family.nunique()!=1 or group.preferred.nunique()!=1:
            raise ValueError('Mixed model identity in one track/arm.')
        values = base.metrics(group[base.TARGET], group.raw_model_probability,
                              group.match_id, config['probability_epsilon'])
        values = {key.replace('validation_','test_'): value for key,value in values.items()}
        cal, ece = calibration_table(group[base.TARGET],group.raw_model_probability,
                                     group.match_id,config['calibration_bins'])
        rows.append({'track':track,'arm_code':arm,'family':group.family.iloc[0],
                     'preferred':bool(group.preferred.iloc[0]),'test_ece':ece,**values})
        cal['track']=track; cal['arm_code']=arm; calibrations.append(cal)
    return pd.DataFrame(rows), pd.concat(calibrations,ignore_index=True)


def match_losses(predictions, epsilon):
    p = np.clip(predictions.raw_model_probability.to_numpy(),epsilon,1-epsilon)
    y = predictions[base.TARGET].to_numpy()
    data = predictions[['track','arm_code','gender','match_id','match_innings']].copy()
    data['log_loss_sum'] = -(y*np.log(p)+(1-y)*np.log1p(-p))
    data['brier_sum'] = (y-predictions.raw_model_probability.to_numpy())**2
    data['events'] = 1
    return data.groupby(['track','arm_code','gender','match_id','match_innings'],sort=False)[
        ['log_loss_sum','brier_sum','events']].sum().reset_index()


def draw_match_counts(matches, replicates, seed):
    if matches<1 or replicates<2:
        raise ValueError('Bootstrap requires matches and at least two replicates.')
    rng = np.random.default_rng(seed)
    draws = rng.integers(0,matches,size=(replicates,matches))
    return np.array([np.bincount(draw,minlength=matches) for draw in draws],dtype=float)


def paired_gain(reference, treatment, counts):
    """Positive = reference loss minus treatment loss. Preserve repeated matches."""
    if not reference.index.equals(treatment.index) or not np.array_equal(reference.events,treatment.events):
        raise ValueError('Paired bootstrap requires identical ordered match coverage.')
    if (reference.events<=0).any() or counts.shape[1]!=len(reference):
        raise ValueError('Invalid match counts.')
    result = {}
    for name, column in [('log_loss','log_loss_sum'),('brier','brier_sum')]:
        delta = reference[column].to_numpy()-treatment[column].to_numpy()
        events = reference.events.to_numpy()
        result['event_log_loss' if name=='log_loss' else 'brier'] = (
            float(delta.sum()/events.sum()), (counts@delta)/(counts@events))
        per_match = delta/events
        result['match_log_loss' if name=='log_loss' else 'match_brier'] = (
            float(per_match.mean()), (counts@per_match)/counts.sum(axis=1))
    return result


def permuted_input(x, columns, permutation):
    """Copy before shuffling; block columns share donor rows, including categories."""
    permutation = np.asarray(permutation)
    if not np.array_equal(np.sort(permutation), np.arange(len(x))):
        raise ValueError('Permutation must contain every row exactly once.')
    result = x.copy(deep=True)
    for column in columns:
        result[column] = x[column].to_numpy()[permutation]
    return result


def event_log_loss(y, p, epsilon):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y)
    if len(y) != len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid importance probabilities.')
    p = np.clip(p, epsilon, 1 - epsilon)
    return float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean())

