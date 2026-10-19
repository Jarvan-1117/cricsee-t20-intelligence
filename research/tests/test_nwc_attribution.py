"""Synthetic boundary/attribution tests. No models fitted or test-set tuning."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import nwc_attribution as stage
import nwc_attribution_visualisation as viz


def fixture():
    rows=[]; metadata=[]
    for mid,gender in [(10,'female'),(20,'male')]:
        for inn,runs in [(1,[1,0,2]),(2,[0,0,4])]:
            score=0
            for event,run in enumerate(runs,1):
                rows.append(dict(source_row=mid*100+inn*10+event,match_id=mid,match_innings=inn,event_sequence=event,
                    track=f'{gender}_innings_{inn}',gender=gender,match_date='2020-01-01',
                    batting_team='A' if inn==1 else 'B',bowling_team='B' if inn==1 else 'A',
                    first_batting_team='A',second_batting_team='B',match_winner='B',first_batting_team_win=0,
                    striker='batter'+str(inn),non_striker='partner'+str(inn),bowler='bowler'+str(inn),
                    striker_id='s'+str(inn),non_striker_id='n'+str(inn),bowler_id='b'+str(inn),known_uncertainty=0,
                    pre_score=score,pre_wickets=0,legal_balls_elapsed=event-1,source_over=0,
                    audit_pre_legal_in_over=event-1,target_runs=4 if inn==2 else np.nan,
                    runs_total=run,runs_extras=0,wides=0,noballs=0,byes=0,legbyes=0,penalty=0,
                    innings_score=score+run,innings_wickets=0,
                    wicket_kind='',player_out='',legal_delivery=1,cumulative_legal_deliveries=event,
                    raw_model_probability=([.6,.65,.7] if inn==1 else [.55,.5,.4])[event-1]))
                score+=run
            metadata.append(dict(match_id=mid,match_innings=inn,batting_team='A' if inn==1 else 'B',
                events=3,actual_legal_balls=3,final_wickets=0,penalty_runs_pre=0,penalty_runs_post=0,
                official_total=score,official_target=4 if inn==2 else np.nan,audit_complete=True,
                audit_completion_basis='no_available_batting_pair' if inn==1 else 'target_reached',
                audit_no_batting_pair=inn==1,miscounted_overs='{}'))
    return pd.DataFrame(rows),pd.DataFrame(metadata)


def handoff_fixture():
    """Synthetic 32-model fixed-IBW protocol; no fit or real evaluation access."""
    evaluation=stage.evaluation
    columns=evaluation.feature_columns()
    preferred={track:'IBW' for track in evaluation.TRACKS}
    rows=[]
    for track in evaluation.TRACKS:
        for code in evaluation.ARM_ORDER:
            rows.append(dict(track=track,arm_code=code,family='constant' if code=='R0' else 'logistic_regression',
                model_path=f'fixture_models/{track}__{code}.joblib',model_sha256='synthetic-hash',
                features=evaluation.base.track_columns(columns[code],int(track[-1])),
                source_stage='06' if code in evaluation.ARM_ORDER[:5] else '07',
                validation_event_log_loss=.1 if code==preferred[track] else .9,preferred=code==preferred[track]))
    protocol={'evaluation_id':'synthetic','config':evaluation.expected_config(),'models':rows,'preferred':preferred}
    keys=['track','arm_code','family','model_path','model_sha256','features']
    handoff=dict(evaluation_id='synthetic',prediction_timing='pre-event',target='first_batting_team_win',
        probability_column='raw_model_probability',status='requires_stage09_migration',
        contract_version=stage.VERSION,selection_policy=evaluation.POLICY,
        prediction_path=evaluation.DATA_DIR+'/preferred_test_predictions.csv.gz',
        preferred_models=[{k:r[k] for k in keys} for r in rows if r['preferred']])
    summary=dict(evaluation_id='synthetic',contract_version=stage.VERSION,preferred=preferred,
                 selection_policy=evaluation.POLICY,
                 test_events=12,test_matches=2)
    d,meta=fixture()
    p=d[stage.KEYS+['track','gender','match_date','first_batting_team_win','raw_model_probability']].copy()
    p['arm_code']=p.track.map(preferred); p['family']='logistic_regression'; p['preferred']=True
    return handoff,protocol,summary,p,d,meta


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.handoff,self.protocol,self.summary,self.p,self.d,self.meta=handoff_fixture()

    def test_fixed_ibw_validation_winners_and_versioned_paths(self):
        registry=stage.validate_handoff(self.handoff,self.protocol,self.summary)
        self.assertEqual(len(registry),4)
        self.assertEqual({r['arm_code'] for r in registry.values()},{'IBW'})
        stage.validate_prediction_handoff(self.p,registry,self.summary)
        self.assertTrue(stage.REPORT_DIR.endswith('/resource_sums_v2_fixed_ibw_v1'))
        self.assertTrue(stage.DATA_DIR.endswith('/resource_sums_v2_fixed_ibw_v1'))

    def test_handoff_identity_path_and_probability_contract(self):
        for key,value in [('evaluation_id','old'),('prediction_path','data/old_predictions.csv.gz'),
                          ('probability_column','attribution_probability'),('prediction_timing','post-event'),
                          ('target','batting_team_win'),('status','legacy')]:
            handoff=copy.deepcopy(self.handoff); handoff[key]=value
            with self.assertRaises(ValueError): stage.validate_handoff(handoff,self.protocol,self.summary)

    def test_duplicate_or_supplementary_main_models_rejected(self):
        for change in ['duplicate','missing','legacy_arm','different_model','different_features']:
            handoff=copy.deepcopy(self.handoff)
            if change=='duplicate': handoff['preferred_models'].append(handoff['preferred_models'][0])
            elif change=='missing': handoff['preferred_models'].pop()
            elif change=='legacy_arm': handoff['preferred_models'][0]['arm_code']='E2'
            elif change=='different_model': handoff['preferred_models'][0]['model_sha256']='other'
            else: handoff['preferred_models'][0]['features'].append('striker_smoothed_batting_average')
            with self.assertRaises(ValueError): stage.validate_handoff(handoff,self.protocol,self.summary)

    def test_legacy_summary_or_protocol_rejected(self):
        summary=dict(self.summary,contract_version='legacy')
        with self.assertRaises(ValueError): stage.validate_handoff(self.handoff,self.protocol,summary)
        protocol=copy.deepcopy(self.protocol); protocol['models'][4]['features'].append('bowler_quality')
        with self.assertRaises(ValueError): stage.validate_handoff(self.handoff,protocol,self.summary)

    def test_non_ibw_arms_cannot_be_main_even_with_better_validation(self):
        for code in stage.evaluation.ARM_ORDER:
            protocol=copy.deepcopy(self.protocol); handoff=copy.deepcopy(self.handoff)
            preferred={track:code for track in stage.evaluation.TRACKS}
            for row in protocol['models']:
                row['preferred']=row['arm_code']==code
                row['validation_event_log_loss']=.1 if row['preferred'] else .9
            protocol['preferred']=preferred
            keys=['track','arm_code','family','model_path','model_sha256','features']
            handoff['preferred_models']=[{k:r[k] for k in keys} for r in protocol['models'] if r['preferred']]
            summary=dict(self.summary,preferred=preferred)
            if code=='IBW':
                registry=stage.validate_handoff(handoff,protocol,summary)
                self.assertEqual({r['arm_code'] for r in registry.values()},{'IBW'})
            else:
                with self.assertRaises(ValueError): stage.validate_handoff(handoff,protocol,summary)

    def test_validation_selection_mapping_cannot_be_overridden(self):
        summary=copy.deepcopy(self.summary); summary['preferred'][stage.evaluation.TRACKS[0]]='R1'
        with self.assertRaisesRegex(ValueError,'validation-only'): stage.validate_handoff(self.handoff,self.protocol,summary)
        protocol=copy.deepcopy(self.protocol)
        for row in protocol['models']: row['validation_event_log_loss']=.001 if row['arm_code']=='R0' else .9
        registry=stage.validate_handoff(self.handoff,protocol,self.summary)
        self.assertEqual({r['arm_code'] for r in registry.values()},{'IBW'})

    def test_old_joint_contract_rejected(self):
        handoff=copy.deepcopy(self.handoff)
        # Keep the incompatible wire contract as data, without importing its implementation.
        handoff.update(contract_version='resource_sums_v2_joint_selection_v1',
                       selection_policy='stage06_stage07_validation_log_loss_v1',
                       prediction_path='data/processed/probability_evaluation/'
                           'resource_sums_v2_joint_selection_v1/preferred_test_predictions.csv.gz')
        with self.assertRaises(ValueError): stage.validate_handoff(handoff,self.protocol,self.summary)

    def test_verified_summary_requires_ibw_for_every_track(self):
        for track in stage.evaluation.TRACKS:
            summary=copy.deepcopy(self.summary); summary['preferred'][track]='RBW'
            with patch.object(stage.evaluation,'verified_output',return_value=summary):
                with self.assertRaisesRegex(ValueError,'fixed-IBW'): stage.input_fingerprints(ROOT)

    def test_new_input_version_never_falls_back_to_old_outputs(self):
        summary=dict(self.summary,contract_version='resource_sums_v2')
        with patch.object(stage.evaluation,'verified_output',return_value=summary):
            with self.assertRaisesRegex(ValueError,'fixed-IBW'): stage.input_fingerprints(ROOT)
        with tempfile.TemporaryDirectory() as d,patch.object(stage,'run') as run:
            old=Path(d)/'outputs/09_nwc_attribution/resource_sums_v2'
            old.mkdir(parents=True); (old/'manifest.json').write_text('{"status":"complete"}')
            with self.assertRaises(FileNotFoundError): stage.verified_output(d)
            run.assert_not_called()

    def test_prediction_metadata_missing_events_and_adjustment_rejected(self):
        registry=stage.validate_handoff(self.handoff,self.protocol,self.summary)
        variants=[self.p.iloc[:-1],pd.concat([self.p,self.p.iloc[:1]])]
        for key,value in [('arm_code','R1'),('preferred',False),('gender','male'),('match_innings',2),
                          ('family','random_forest')]:
            p=self.p.copy(); p.loc[0,key]=value; variants.append(p)
        adjusted=self.p.copy(); adjusted['attribution_probability']=0; variants.append(adjusted)
        for p in variants:
            with self.assertRaises(ValueError): stage.validate_prediction_handoff(p,registry,self.summary)

    def test_real_loader_on_synthetic_files_without_model_loading(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for row in self.handoff['preferred_models']:
                path=root/row['model_path']; path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b'synthetic model bytes never loaded')
                row['model_sha256']=stage.sha256(path)
                for model in self.protocol['models']:
                    if model['track']==row['track'] and model['preferred']:
                        model['model_sha256']=row['model_sha256']
            directory=root/stage.evaluation.REPORT_DIR; directory.mkdir(parents=True)
            for name,value in [('nwc_handoff',self.handoff),('frozen_protocol',self.protocol),
                               ('evaluation_summary',self.summary)]:
                (directory/(name+'.json')).write_text(json.dumps(value))
            pred=root/self.handoff['prediction_path']; pred.parent.mkdir(parents=True)
            self.p.to_csv(pred,index=False,compression='gzip')
            for index,relative in enumerate(stage.SOURCE_FILES[:4]):
                path=root/relative; path.parent.mkdir(parents=True,exist_ok=True)
                (self.meta if index==2 else self.d).to_csv(path,index=False,
                    compression='gzip' if path.suffix=='.gz' else None)
            with patch.object(stage.evaluation.base.joblib,'load',side_effect=AssertionError('No model loading')):
                events,meta,handoff=stage.load_inputs(root)
                ledger=stage.build_ledger(events,meta)
            self.assertEqual(len(ledger),12)
            np.testing.assert_allclose(ledger.raw_model_probability,self.p.raw_model_probability)
            self.assertEqual(handoff,self.handoff)
            model_path=root/self.handoff['preferred_models'][0]['model_path']
            model_path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Frozen model changed'): stage.load_inputs(root)

    def test_first_event_delta_is_included_and_raw_values_unchanged(self):
        ledger=stage.build_ledger(self.d,self.meta)
        first=ledger[ledger.is_first_event]
        np.testing.assert_allclose(first.interval_wpa_first_team,[.05,-.05,.05,-.05])
        np.testing.assert_array_equal(ledger.raw_model_probability,self.d.raw_model_probability)

    def test_terminal_change_not_spread_to_previous_events(self):
        changed=self.d.copy(); changed.loc[changed.event_sequence.eq(3)&changed.match_innings.eq(2),'raw_model_probability']=.99
        before=stage.build_ledger(self.d,self.meta); after=stage.build_ledger(changed,self.meta)
        np.testing.assert_allclose(after.loc[after.is_terminal_event,'interval_wpa_first_team'],-.99)
        early=before.event_sequence.eq(1)
        np.testing.assert_allclose(before.loc[early,'interval_wpa_first_team'],after.loc[early,'interval_wpa_first_team'])
        # The preceding interval legitimately changes with its new observed endpoint.
        matches,players,_=stage.summarise(after)
        self.assertLess(matches.closure_error.max(),1e-12)
        self.assertLess(stage.team_accounting(after,players).closure_error.max(),1e-12)


class LedgerTests(unittest.TestCase):
    def setUp(self): self.d,self.meta=fixture()

    def test_complete_ledger_partial_allocation_and_telescoping(self):
        d=stage.build_ledger(self.d,self.meta)
        m,p,c=stage.summarise(d)
        self.assertEqual(len(d),12)
        self.assertEqual(d.is_first_event.sum(),4)
        self.assertEqual(d.is_terminal_event.sum(),2)
        self.assertEqual(d.player_allocated.sum(),10)
        self.assertLess(m.closure_error.max(),1e-12)
        self.assertTrue(np.allclose(m.probability_change,-.6))
        self.assertTrue(np.allclose(p.groupby('match_id').partial_nwc.sum(),0))

    def test_first_innings_last_is_unidentified_not_zero_or_pure_gap(self):
        d=stage.build_ledger(self.d,self.meta)
        b=d[d.interval_kind.eq('last_event_plus_innings_transition')]
        self.assertTrue(b.event_interval_wpa_first_team.isna().all())
        self.assertTrue(b.striker_partial_nwc.isna().all())
        self.assertTrue(np.allclose(b.boundary_bundle_first_team,-.15))

    def test_terminal_not_clipped_or_used_as_raw_prediction(self):
        d=stage.build_ledger(self.d,self.meta)
        t=d[d.is_terminal_event]
        self.assertTrue(np.allclose(t.raw_model_probability,.4))
        self.assertTrue(np.allclose(t.interval_endpoint_probability,0))
        self.assertTrue(np.allclose(t.striker_partial_nwc,.4))
        self.assertTrue(np.allclose(t.bowler_partial_nwc,-.4))

    def test_role_change_flagged_without_automatic_exclusion(self):
        self.d.loc[1,'striker_id']='replacement'
        d=stage.build_ledger(self.d,self.meta)
        self.assertTrue(d.player_allocated.iloc[0])
        self.assertIn('batting_pair_changed',d.role_change_flags.iloc[0])
        self.assertEqual(d.striker_id.iloc[0],'s1')

    def test_non_striker_swap_allocated_and_flagged(self):
        self.d.loc[1,['striker_id','non_striker_id']]=['n1','s1']
        d=stage.build_ledger(self.d,self.meta)
        self.assertTrue(d.player_allocated.iloc[0])
        self.assertEqual(d.role_change_flags.iloc[0],'strike_swapped')

    def test_bowler_change_retains_current_bowler_proxy(self):
        self.d.loc[1,'bowler_id']='newbowler'
        d=stage.build_ledger(self.d,self.meta)
        self.assertTrue(d.player_allocated.iloc[0])
        self.assertEqual(d.role_change_flags.iloc[0],'bowler_changed')
        self.assertEqual(d.bowler_id.iloc[0],'b1')

    def test_wicket_and_runout_not_forced_to_striker(self):
        self.d.loc[0,'wicket_kind']='run out'
        d=stage.build_ledger(self.d,self.meta)
        self.assertTrue(pd.isna(d.striker_partial_nwc.iloc[0]))

    def test_extras_unallocated_and_kept(self):
        self.d.loc[0,'runs_extras']=1
        self.d.loc[0,'byes']=1
        d=stage.build_ledger(self.d,self.meta)
        self.assertEqual(len(d),len(self.d))
        self.assertEqual('team_bye_or_legbye_interval',d.allocation_reason.iloc[0])

    def test_final_wide_is_terminal_but_not_player_allocated(self):
        self.d.loc[5,['runs_extras','legal_delivery','cumulative_legal_deliveries']]=[4,0,2]
        self.d.loc[5,'wides']=4
        self.meta.loc[1,'actual_legal_balls']=2
        d=stage.build_ledger(self.d,self.meta)
        self.assertTrue(d.is_terminal_event.iloc[5]); self.assertFalse(d.player_allocated.iloc[5])

    def test_final_penalty_accounted_but_unallocated(self):
        self.d.loc[5,['runs_total','innings_score']]=[0,0]
        self.meta.loc[1,'penalty_runs_post']=4
        d=stage.build_ledger(self.d,self.meta)
        self.assertIn('post_innings_penalty',d.allocation_reason.iloc[5])
        self.assertFalse(d.player_allocated.iloc[5])

    def test_bad_outcome_rejected(self):
        self.d.loc[self.d.match_id.eq(10),'first_batting_team_win']=1
        with self.assertRaisesRegex(ValueError,'Outcome'): stage.build_ledger(self.d,self.meta)

    def test_false_terminal_rejected(self):
        self.meta.loc[1,'audit_completion_basis']='unresolved'
        with self.assertRaisesRegex(ValueError,'endpoint'): stage.build_ledger(self.d,self.meta)

    def test_wrong_target_rejected(self):
        self.meta.loc[1,'official_target']=5
        with self.assertRaisesRegex(ValueError,'Target'): stage.build_ledger(self.d,self.meta)

    def test_wrong_source_order_rejected(self):
        self.d.loc[0,'source_row']=1015
        with self.assertRaisesRegex(ValueError,'Event order'): stage.build_ledger(self.d,self.meta)

    def test_unsorted_input_restores_source_order(self):
        d=stage.build_ledger(self.d.sample(frac=1,random_state=1),self.meta)
        self.assertEqual(list(d.source_row),list(self.d.source_row))

    def test_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            stage.build_ledger(pd.concat([self.d,self.d.iloc[:1]]),self.meta)

    def test_invalid_probability_rejected(self):
        for value in [np.nan,1.1,-.1]:
            d=self.d.copy(); d.loc[0,'raw_model_probability']=value
            with self.assertRaisesRegex(ValueError,'probability'): stage.build_ledger(d,self.meta)

    def test_incomplete_metadata_rejected(self):
        with self.assertRaisesRegex(ValueError,'metadata'): stage.build_ledger(self.d,self.meta.iloc[:-1])

    def test_no_allocation_is_missing_not_zero(self):
        self.d['wicket_kind']='retired hurt'
        d=stage.build_ledger(self.d,self.meta); _,players,_=stage.summarise(d)
        self.assertTrue(players.partial_nwc.isna().all())

    def test_config_and_missing_output(self):
        self.assertEqual(stage.read_config(ROOT)['training_fits'],0)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(FileNotFoundError,'RUN_ATTRIBUTION'): stage.verified_output(td)

    def test_bowled_and_lbw_allocate_source_striker(self):
        for kind in ['bowled','lbw']:
            d=self.d.copy(); meta=self.meta.copy()
            d.loc[1,'wicket_kind']=kind; d.loc[1,'player_out']=d.loc[1,'striker']
            d.loc[[1,2],'innings_wickets']=1; d.loc[2,'pre_wickets']=1
            meta.loc[0,'final_wickets']=1
            result=stage.build_ledger(d,meta)
            self.assertEqual(result.allocation_reason.iloc[1],'player_direct_dismissal_proxy')
            self.assertTrue(result.player_allocated.iloc[1])

    def test_direct_dismissal_wrong_batter_fails_closed(self):
        self.d.loc[1,'wicket_kind']='bowled'; self.d.loc[1,'player_out']='partner1'
        with self.assertRaisesRegex(ValueError,'Direct dismissal'): stage.build_ledger(self.d,self.meta)

    def test_shared_fielding_events_remain_team_residuals(self):
        for kind in ['caught','caught and bowled','stumped','run out','retired hurt','hit wicket']:
            d=self.d.copy(); d.loc[1,'wicket_kind']=kind
            result=stage.build_ledger(d,self.meta)
            self.assertFalse(result.player_allocated.iloc[1])
            self.assertTrue(result.allocation_reason.iloc[1].startswith('team_'))

    def test_mixed_extra_and_exit_not_split_by_runs(self):
        self.d.loc[0,['runs_extras','byes']]=[1,1]
        self.d.loc[0,'wicket_kind']='run out'
        d=stage.build_ledger(self.d,self.meta)
        self.assertEqual(d.allocation_reason.iloc[0],'team_mixed_extras_and_exit')

    def test_penalty_event_priority(self):
        self.d.loc[0,['runs_extras','penalty']]=[1,1]
        d=stage.build_ledger(self.d,self.meta)
        self.assertEqual(d.allocation_reason.iloc[0],'team_penalty_event')

    def test_extra_components_fail_closed(self):
        self.d.loc[0,'runs_extras']=1
        with self.assertRaisesRegex(ValueError,'Extra components'): stage.build_ledger(self.d,self.meta)

    def test_team_and_player_totals_reconcile(self):
        d=stage.build_ledger(self.d,self.meta); _,players,_=stage.summarise(d)
        teams=stage.team_accounting(d,players)
        self.assertEqual(len(teams),4)
        self.assertLess(teams.closure_error.max(),1e-12)
        self.assertTrue(np.allclose(players.partial_nwc,
            players.partial_nwc_excluding_terminal+players.terminal_partial_nwc))
        players.loc[0,'partial_nwc']+=.1
        with self.assertRaisesRegex(ValueError,'Team/player'): stage.team_accounting(d,players)

    def test_player_plot_separates_terminal(self):
        d=stage.build_ledger(self.d,self.meta); _,players,_=stage.summarise(d)
        fig=viz.player_figure(players,10)
        self.assertEqual([a.get_title() for a in fig.axes],['Non-terminal','Terminal','Total'])
        plt.close(fig)

    def test_event_allocation_corruption_rejected(self):
        d=stage.build_ledger(self.d,self.meta)
        d.loc[0,'allocated_first_team_wpa']+=.01
        with self.assertRaisesRegex(ValueError,'Event allocation'): stage.summarise(d)

    def test_unknown_player_is_not_silently_omitted(self):
        self.d.loc[0,'striker_id']=None
        with self.assertRaisesRegex(ValueError,'identity'): stage.build_ledger(self.d,self.meta)

    def test_false_scheduled_endpoint(self):
        self.meta.loc[0,'audit_completion_basis']='scheduled_overs_complete'
        with self.assertRaisesRegex(ValueError,'final over'): stage.build_ledger(self.d,self.meta)

    def test_post_target_record_rejected(self):
        self.d.loc[4,['pre_score','innings_score']]=[4,4]
        with self.assertRaises(ValueError): stage.build_ledger(self.d,self.meta)

    def test_run_verify_and_figures(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            handoff,_,upstream,_,_,_=handoff_fixture()
            with patch.object(stage,'read_config',return_value=stage.read_config(ROOT)), \
                 patch.object(stage,'input_fingerprints',return_value=({'synthetic':'hash'},upstream)), \
                 patch.object(stage,'load_inputs',return_value=(self.d,self.meta,handoff)):
                summary=stage.run(root,progress=lambda _:None)
                self.assertEqual(stage.verified_output(root)['events'],12)
                figures=viz.build_figures(root)
                self.assertEqual(len(figures),6)
                viz.save_figures(root,summary,figures)
                plt.close('all')
                path=root/stage.REPORT_DIR/'match_reconciliation.csv'
                path.write_text('changed')
                with self.assertRaisesRegex(ValueError,'artifact changed'): stage.verified_output(root)


if __name__=='__main__': unittest.main()
