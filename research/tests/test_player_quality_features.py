"""Independent small-case checks for Stage 03 identities and historical timing."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import player_quality_features as pq


def fixture():
    roster, events = [], []
    for mid, date in [(1, "2020-01-01"), (2, "2020-01-02")]:
        for team in ["A", "B"]:
            for number in range(1, 12):
                name = team + str(number)
                roster.append({"match_id": mid, "gender": "male", "match_date": pd.Timestamp(date),
                    "team": team, "player_name": name, "player_id": 'id_' + name,
                    "catalogue_scope": "listed_participant_not_live_availability"})
        # Wide, no-ball boundary, byes, legbyes, penalty, run out, retired hurt,
        # a bowler-credited wicket, and a legal dot. Retirement is not a dismissal.
        plays = [(0,1,0,0,0,0,'', ''), (4,0,1,0,0,0,'',''), (0,0,0,2,0,0,'',''),
                 (0,0,0,0,1,0,'',''), (0,0,0,0,0,5,'',''),
                 (0,0,0,0,0,0,'run out','A2'), (0,0,0,0,0,0,'retired hurt','A1'),
                 (0,0,0,0,0,0,'bowled','A3'), (0,0,0,0,0,0,'','')]
        for seq,(runs,wide,no,byes,legbyes,penalty,kind,out) in enumerate(plays,1):
            events.append({"match_id":mid,"gender":"male","match_date":date,"source_row":len(events),
                "match_innings":1,"event_sequence":seq,"batting_team":"A","bowling_team":"B",
                "striker":"A1","non_striker":"A2","bowler":"B1","player_out":out or np.nan,
                "wicket_kind":kind or np.nan,"runs_batter":runs,"wides":wide,"noballs":no,
                "byes":byes,"legbyes":legbyes,"penalty":penalty,
                "runs_total":runs+wide+no+byes+legbyes+penalty,"legal_delivery":int(not(wide or no))})
    return pd.DataFrame(events),pd.DataFrame(roster)


class PlayerQualityTests(unittest.TestCase):
    def setUp(self):
        self.d,self.roster=fixture()
        self.config=pq.read_config(ROOT)

    def pipeline(self):
        linked=pq.link_event_identities(self.d,self.roster)
        totals=pq.match_contributions(linked,self.roster)
        profiles=pq.build_profiles(totals,self.config)
        return linked,totals,profiles

    def test_stat_accounting_extras_dismissals_and_participation(self):
        _,c,_=self.pipeline()
        first=c.loc[c.match_id.eq(1)].set_index('player_id')
        self.assertEqual(first.loc['id_A1','batting_runs'],4)
        self.assertEqual(first.loc['id_A1','balls_faced'],8)
        self.assertEqual(first.loc['id_A1','boundary_balls'],1)
        self.assertEqual(first.loc['id_A1','batting_dismissals'],0)
        self.assertEqual(first.loc['id_A2','batting_dismissals'],1)
        self.assertEqual(first.loc['id_A3','batting_dismissals'],1)
        self.assertEqual(first.loc['id_B1','bowling_runs_conceded'],6)
        self.assertEqual(first.loc['id_B1','legal_balls_bowled'],7)
        self.assertEqual(first.loc['id_B1','bowler_wickets'],1)
        self.assertEqual(first.loc['id_B1','bowling_dot_balls'],4)
        self.assertEqual(first.loc['id_A11','observed_matches'],0)
        self.assertEqual(first.loc['id_A11','listed_matches'],1)
        self.assertEqual(first.loc['id_A2','batting_matches'],1)

    def test_first_date_uses_cold_start_only(self):
        _,_,p=self.pipeline()
        first=p.loc[p.match_id.eq(1)]
        for metric,reference in self.config['cold_start'].items():
            self.assertTrue(np.allclose(first['smoothed_'+metric],reference))
            self.assertTrue(first['raw_'+metric].isna().all())

    def test_second_date_has_only_first_date_personal_history(self):
        _,c,p=self.pipeline()
        first=c.loc[c.match_id.eq(1)].set_index('player_id')
        second=p.loc[p.match_id.eq(2)].set_index('player_id')
        for name in pq.COUNTS:
            np.testing.assert_array_equal(first[name],second.loc[first.index,'prior_'+name])

    def test_changing_today_cannot_change_today_profiles_or_reference(self):
        _,_,before=self.pipeline()
        self.d.loc[self.d.match_id.eq(2),'runs_batter']+=20
        self.d.loc[self.d.match_id.eq(2),'runs_total']+=20
        _,_,after=self.pipeline()
        pd.testing.assert_frame_equal(before,after)

    def test_same_date_results_never_inform_each_other(self):
        self.d['match_date']='2020-01-01'; self.roster['match_date']=pd.Timestamp('2020-01-01')
        _,_,p=self.pipeline()
        self.assertTrue(p.prior_balls_faced.eq(0).all())
        self.assertTrue(p.reference_prior_balls_faced.eq(0).all())

    def test_genders_have_separate_history_and_references(self):
        self.d.loc[self.d.match_id.eq(2),'gender']='female'
        self.roster.loc[self.roster.match_id.eq(2),'gender']='female'
        _,_,p=self.pipeline()
        self.assertTrue(p.prior_balls_faced.eq(0).all())
        self.assertTrue(p.reference_prior_balls_faced.eq(0).all())

    def test_alias_same_id_keeps_history(self):
        self.roster.loc[(self.roster.match_id==2)&(self.roster.player_name=='A1'),'player_name']='New name'
        self.d.loc[self.d.match_id.eq(2),'striker']='New name'
        self.d.loc[self.d.match_id.eq(2)&self.d.player_out.eq('A1'),'player_out']='New name'
        _,_,p=self.pipeline()
        row=p.loc[(p.match_id==2)&(p.player_id=='id_A1')].iloc[0]
        self.assertEqual(row.player_name,'New name')
        self.assertEqual(row.prior_balls_faced,8)

    def test_same_name_different_id_does_not_inherit_history(self):
        self.roster.loc[(self.roster.match_id==2)&(self.roster.player_name=='A1'),'player_id']='different_person'
        _,_,p=self.pipeline()
        self.assertEqual(p.loc[p.player_id.eq('different_person'),'prior_balls_faced'].iloc[0],0)

    def test_identity_failure_is_not_a_cold_start(self):
        self.d.loc[0,'striker']='Unknown'
        with self.assertRaisesRegex(ValueError,'Unresolved actor'): self.pipeline()

    def test_wrong_team_and_duplicate_identity_fail(self):
        self.d.loc[0,'bowler']='A11'
        with self.assertRaisesRegex(ValueError,'wrong team'): self.pipeline()
        self.d,self.roster=fixture()
        self.roster.loc[1,'player_id']=self.roster.loc[0,'player_id']
        with self.assertRaisesRegex(ValueError,'Duplicate'): self.pipeline()

    def test_zero_dismissals_stays_raw_missing_but_smoothed_finite(self):
        _,_,p=self.pipeline()
        row=p.loc[(p.match_id==2)&(p.player_id=='id_A1')].iloc[0]
        self.assertTrue(np.isnan(row.raw_batting_average))
        self.assertTrue(np.isfinite(row.smoothed_batting_average))
        self.assertEqual(row.no_prior_batting_dismissals,1)

    def test_raw_and_smoothed_formula_and_evidence_weight(self):
        personal=np.zeros(len(pq.COUNTS),dtype=int); pop=personal.copy()
        personal[pq.COUNTS.index('batting_runs')]=120
        personal[pq.COUNTS.index('balls_faced')]=60
        v=pq.profile_values(personal,pop,self.config)
        self.assertAlmostEqual(v['raw_batting_strike_rate'],200)
        self.assertAlmostEqual(v['smoothed_batting_strike_rate'],(12000+110*120)/180)
        self.assertAlmostEqual(v['batting_ball_reliability'],1/3)

    def test_different_bowling_exposure_remains_distinguishable(self):
        a=np.zeros(len(pq.COUNTS),dtype=int)
        for name,value in [('bowling_runs_conceded',100),('bowler_wickets',5),('legal_balls_bowled',60)]: a[pq.COUNTS.index(name)]=value
        b=a.copy(); b[pq.COUNTS.index('legal_balls_bowled')]=120
        av=pq.profile_values(a,np.zeros_like(a),self.config); bv=pq.profile_values(b,np.zeros_like(a),self.config)
        self.assertNotEqual(av['smoothed_bowling_economy'],bv['smoothed_bowling_economy'])
        self.assertNotEqual(av['smoothed_bowling_wicket_rate'],bv['smoothed_bowling_wicket_rate'])

    def test_current_profile_join_preserves_original_rows_and_columns(self):
        linked,_,p=self.pipeline()
        original=self.d[['match_id','source_row','match_innings','event_sequence','gender','striker','non_striker','bowler']].copy()
        original['first_batting_team_win']=1
        out=pq.attach_current_players(original,linked,p)
        pd.testing.assert_frame_equal(out[original.columns],original)
        self.assertEqual(len(pq.feature_contract()['current_player_quality']),11)
        self.assertEqual(len(pq.feature_contract()['current_player_evidence']),8)

    def test_all_listed_players_have_profiles_but_are_not_declared_active(self):
        _,_,p=self.pipeline()
        self.assertEqual(len(p),44)
        self.assertTrue(p.catalogue_scope.eq('listed_participant_not_live_availability').all())
        self.assertNotIn('is_active',p)

    def test_roster_over_eleven_allowed_without_assigning_starting_xi(self):
        names={t:[f'{t}{i}' for i in range(1,12)] for t in ['A','B']}; names['A'].append('Sub')
        doc={'info':{'gender':'male','dates':['2020-01-01'],'teams':['A','B'],'players':names,
                      'registry':{'people':{name:'id_'+name for group in names.values() for name in group}}}}
        match=SimpleNamespace(match_id=1,match_date='2020-01-01',gender='male',first_batting_team='A',second_batting_team='B')
        rows=pq.roster_from_document(doc,match)
        self.assertEqual(len(rows),23)
        self.assertNotIn('initial_XI',rows[0])
        del doc['info']['registry']['people']['Sub']
        with self.assertRaisesRegex(ValueError,'Missing/ambiguous'): pq.roster_from_document(doc,match)

    def test_shuffled_contributions_same_profiles(self):
        _,c,p=self.pipeline()
        other=pq.build_profiles(c.sample(frac=1,random_state=42),self.config)
        pd.testing.assert_frame_equal(p,other)

    def test_unknown_exit_kind_fails(self):
        self.d.loc[5,'wicket_kind']='unknown dismissal'
        with self.assertRaisesRegex(ValueError,'Unknown dismissal'): self.pipeline()

    def test_raw_zero_wickets_is_zero_rate_not_missing_with_exposure(self):
        values=np.zeros(len(pq.COUNTS),dtype=int); values[pq.COUNTS.index('legal_balls_bowled')]=12
        v=pq.profile_values(values,np.zeros_like(values),self.config)
        self.assertEqual(v['raw_bowling_wicket_rate'],0)
        self.assertGreater(v['smoothed_bowling_wicket_rate'],0)

    def test_invalid_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'config'; folder.mkdir()
            bad=dict(self.config); bad['player_pseudo_balls']=0
            (folder/'player_quality.json').write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'positive'): pq.read_config(temp)

    def test_missing_and_stale_outputs_stop_without_rebuilding(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError,'RUN_STAGE'): pq.verified_output(temp)
            folder=Path(temp)/pq.DATA_DIR; folder.mkdir(parents=True)
            (folder/'player_quality_manifest.json').write_text(json.dumps({'status':'complete','input_sha256':{'x':'old'}}))
            with patch.object(pq,'input_fingerprints',return_value=({'x':'new'},{})):
                with self.assertRaisesRegex(ValueError,'no automatic rebuild'): pq.verified_output(temp)


if __name__=='__main__': unittest.main()
