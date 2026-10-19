"""Synthetic eight-arm evaluation fixtures; no production models or data scoring."""
from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import probability_evaluation_fixed_ibw as stage


class FrozenPredictor:
    """A prediction-only fake: any attempted fit fails the integration test."""
    classes_ = np.array([0,1])
    def __init__(self, features, shift=0):
        self.feature_names_in_ = np.array(features)
        self.shift = shift
    def fit(self,*args,**kwargs):
        raise AssertionError('Evaluation must never fit a model.')
    def predict_proba(self,x):
        score = x.pre_score.to_numpy() if isinstance(x,pd.DataFrame) else np.zeros(len(x))
        p = 1/(1+np.exp(-(score+self.shift)))
        return np.c_[1-p,p]


def fixture(root, preferred=None, stage_module=None):
    stage = stage_module or globals()['stage']
    config = stage.read_config(ROOT)
    contract = json.loads((ROOT/stage.base.resources.REPORT_DIR/'feature_contract.json').read_text())
    groups = stage.base.resolve_arms(contract)
    columns = {stage.base.ARM_CODES[k]:v for k,v in groups.items()}
    union = list(dict.fromkeys(c for values in columns.values() for c in values))
    splits, events = [], []
    rng = np.random.default_rng(32933)
    for gender,offset in [('female',100),('male',200)]:
        for i in range(6):
            split = ['train','validation','test','test','test','test'][i]
            date = f'2020-01-{i+1:02}'
            splits.append({'match_id':offset+i,'gender':gender,'match_date':date,'source_events':6,'split':split})
            for innings in [1,2]:
                for event in range(3):
                    row = {c:float(rng.normal()) for c in union}
                    row.update(match_id=offset+i,gender=gender,match_date=date,source_row=len(events),
                               match_innings=innings,event_sequence=event+1,
                               innings_phase=['powerplay','middle','death'][event],
                               first_batting_team_win=i%2)
                    if innings==1: row.update({c:np.nan for c in stage.base.CHASE})
                    events.append(row)
    splits = pd.DataFrame(splits); events = pd.DataFrame(events)
    source = root/stage.base.resources.DATA_DIR; source.mkdir(parents=True)
    events.to_csv(source/'model_input.csv.gz',index=False,compression='gzip')
    splitdir = root/stage.base.DATA_DIR; splitdir.mkdir(parents=True)
    splits.to_csv(splitdir/'match_splits.csv',index=False)
    models = root/'fixture_models'; models.mkdir()
    resources = stage.base.resources.BAT_FEATURES + stage.base.resources.BOWL_FEATURES
    events[resources] = events[resources].abs()
    events['scheduled_balls_remaining_estimate'] = 60.
    events.to_csv(source/'model_input.csv.gz', index=False, compression='gzip')
    preferred=preferred or {t:'IBW' for t in stage.TRACKS}
    registry=[]
    columns=stage.feature_columns()
    for track in stage.TRACKS:
        for i,arm in enumerate(stage.ARM_ORDER):
            features=stage.base.track_columns(columns[arm],int(track.rsplit('_',1)[1]))
            path=root/'fixture_models'/(track+'__'+arm+'.joblib')
            joblib.dump(FrozenPredictor(features,i/20),path)
            registry.append(dict(track=track,arm_code=arm,arm=arm,family='synthetic_prediction_only',
                candidate=0,parameters={},features=features,model_path=path.relative_to(root).as_posix(),
                model_sha256=stage.sha256(path),source_stage='06' if i<5 else '07',
                preferred=preferred[track]==arm,validation_event_log_loss=.1 if preferred[track]==arm else .3))
    stage.validate_registry(registry,config)
    data,splits=stage.load_test_data(root,registry)
    return config,registry,data,splits

