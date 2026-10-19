"""Stage 05 v2 presentation: separate quality sums, not win predictions."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import remaining_resource_features as resource
from data_preparation import sha256, write_json

STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
         "axes.spines.right": False, "figure.facecolor": "white"}
INNINGS_COLORS = {1: "#277DA8", 2: "#BF7A30"}
EXAMPLE_DATE = "2020-01-01"  # Calendar rule only: no outcomes or feature magnitudes.


def select_example(frame):
    if frame.empty:
        raise ValueError("No resource events to display")
    dates = frame[["match_id", "match_date"]].drop_duplicates().sort_values(["match_date", "match_id"])
    eligible = dates.loc[dates.match_date.ge(EXAMPLE_DATE)]
    chosen = int((eligible if len(eligible) else dates).match_id.iloc[0])
    return frame.loc[frame.match_id.eq(chosen)].reset_index(drop=True)


def load_example(root):
    columns = resource.KEYS + ["match_date", "batting_team", "bowling_team", "source_over",
              "audit_waiting_batter_count", "audit_nominal_budget", "audit_total_remaining_overs",
              "audit_current_bowler_capacity"] + resource.BAT_FEATURES + resource.BOWL_FEATURES
    data = pd.read_csv(Path(root) / resource.DATA_DIR / "model_input.csv.gz", usecols=columns)
    return select_example(data)


def load_example_contributions(root, match_id):
    pieces = []
    for part in pd.read_csv(Path(root) / resource.DATA_DIR / "resource_contributions.csv.gz", chunksize=25000):
        selected = part.loc[part.match_id.eq(match_id)]
        if len(selected):
            pieces.append(selected)
        elif pieces:
            break  # The export stores complete matches contiguously.
    if not pieces:
        raise ValueError("Example resource contribution evidence is missing")
    return pd.concat(pieces, ignore_index=True)


def overview_figure(summary, coverage):
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(12, 5.4), layout="constrained")
        grid = fig.add_gridspec(2, 4, height_ratios=[1, 2.5])
        for index,(label,value) in enumerate([
            ("Upstream matches (unchanged)", summary["upstream_matches"]),
            ("Shared resource cohort", summary["matches"]), ("Events in every comparison arm", summary["events"]),
            ("Matches held for review", summary["review_or_quarantined_matches"]) ]):
            ax = fig.add_subplot(grid[0,index]);ax.axis('off')
            ax.text(.02,.8,label,transform=ax.transAxes,fontsize=9,color='#52657B')
            ax.text(.02,.15,f'{value:,}',transform=ax.transAxes,fontsize=25,weight='bold',color='#17365D')
        ax=fig.add_subplot(grid[1,:]);left=np.zeros(2);genders=['female','male']
        for flag,label,color in [(True,'Resource sample','#287F71'),(False,'Retained for review','#C58B32')]:
            counts=np.array([coverage.loc[coverage.gender.eq(g)&coverage.included_in_resource_sample.eq(flag),'matches'].sum() for g in genders])
            ax.barh(genders,counts,left=left,label=label,color=color)
            for y,n,start in zip(genders,counts,left):
                if n>20:ax.text(start+n/2,y,str(int(n)),ha='center',va='center',color='white')
            left+=counts
        ax.set_xlabel('Whole matches; same rows and future date split for every experimental arm')
        ax.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.2))
        fig.suptitle('Stage 05: resource construction, not model evaluation',fontsize=16,weight='bold')
        return fig


def batting_figure(frame):
    titles=['Sum of strike rates','Sum of batting averages','Sum of boundary-event rates','Sum of dot-ball rates']
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(2,2,figsize=(12,7.2),layout='constrained')
        for ax,column,title in zip(axes.flat,resource.BAT_FEATURES,titles):
            for inn,part in frame.groupby('match_innings',sort=True):
                ax.step(part.event_sequence,part[column],where='post',lw=1.6,color=INNINGS_COLORS[inn],
                        label=f'Innings {inn}: {part.batting_team.iloc[0]}')
            ax.axhline(0,color='#8796A5',ls='--',lw=1)
            ax.set_title(title,loc='left',weight='bold');ax.set_xlabel('Pre-event index within innings')
            ax.set_ylabel('Sum of smoothed player metrics');ax.legend(frameon=False,fontsize=8)
            ax.grid(axis='y',alpha=.15)
        fig.suptitle(f'Match {int(frame.match_id.iloc[0])}: waiting-batter resource dimensions',fontsize=15,weight='bold')
        fig.supxlabel('Only available players who have NEVER entered this innings. Current pair and returned players excluded. Separate metric units.',fontsize=9)
        return fig


def bowling_quality_figure(frame):
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(2,3,figsize=(12,7.1),layout='constrained')
        for inn,row in enumerate(axes,1):
            part=frame.loc[frame.match_innings.eq(inn)]
            for ax,col,title in zip(row,resource.BOWL_FEATURES,['Economy resource','Wicket-rate resource','Dot-rate resource']):
                if part.empty:continue
                ax.plot(part.event_sequence,part[col],color='#287F71',lw=1.6,label='Capacity-weighted sum')
                ax.axhline(0,color='#8796A5',lw=.8)
                ax.set_title(f'Innings {inn}: {title}',loc='left',fontsize=10,weight='bold')
                ax.set_xlabel('Pre-event index');ax.set_ylabel('Sum of remaining overs x metric')
                ax.grid(axis='y',alpha=.15);ax.legend(frameon=False,fontsize=8)
        fig.suptitle(f'Match {int(frame.match_id.iloc[0])}: remaining bowling resource sums',fontsize=14,weight='bold')
        fig.supxlabel('Includes current bowler remaining capacity. No allocation or weighted mean. Higher economy is NOT better quality; separate units.',fontsize=9)
        return fig


def capacity_figure(snapshot, detail, names):
    array=json.loads(detail.bowling_resource_json)
    data=pd.DataFrame(array,columns=resource.BOWL_DETAIL_SCHEMA).sort_values(['remaining_overs','player_id'],ascending=[False,True])
    y=np.arange(len(data))
    with plt.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(12,7.4))
        fig.subplots_adjust(left=.16,right=.98,top=.86,bottom=.25)
        colors=['#C58B32' if s=='current_bowler' else '#287F71' for s in data.status]
        ax.barh(y,data.remaining_overs,height=.65,color=colors)
        from matplotlib.patches import Patch
        handles=[Patch(color='#C58B32',label='Current bowler'),Patch(color='#287F71',label='Other bowlers')]
        ax.set_yticks(y,[names.get(p,p) for p in data.player_id]);ax.invert_yaxis()
        ax.set_xlabel('Remaining overs (nominal ball capacity / 6)')
        fig.legend(handles=handles,frameon=False,loc='lower center',bbox_to_anchor=(.56,.07),ncol=2,fontsize=9)
        ax.grid(axis='x',alpha=.15)
        ax.set_title(f'Match {int(snapshot.match_id)} | innings {int(snapshot.match_innings)} | event {int(snapshot.event_sequence)}\n'
                     f'Individual capacities total {snapshot.audit_total_remaining_overs:g} overs; innings nominal slots {snapshot.audit_nominal_budget:g}',
                     loc='left',fontsize=13,weight='bold')
        fig.text(.5,.025,'Individual upper bounds can exceed the remaining innings budget. No future bowling schedule is predicted.',ha='center',fontsize=9)
        return fig


def build_figures(root, summary):
    root=Path(root);frame=load_example(root);mid=int(frame.match_id.iloc[0])
    details=load_example_contributions(root,mid)
    candidates=frame.loc[frame.match_innings.eq(2)&frame.source_over.ge(10)]
    snapshot=(candidates if len(candidates) else frame).iloc[0]
    detail=details.loc[details.source_row.eq(snapshot.source_row)].iloc[0]
    catalogue=pd.read_csv(root/'data/processed/historical_features/participant_registry.csv')
    local=catalogue.loc[catalogue.match_id.eq(mid)]
    figures={
        'resource_coverage.png':overview_figure(summary,pd.read_csv(root/resource.REPORT_DIR/'coverage.csv')),
        'waiting_batting_resources.png':batting_figure(frame),
        'remaining_bowling_resources.png':bowling_quality_figure(frame),
        'bowling_capacity_example.png':capacity_figure(snapshot,detail,dict(zip(local.player_id,local.player_name))),
    }
    for fig in figures.values():fig.resource_example_match_id=mid
    return figures


def save_figures(root, summary, figures):
    root=Path(root);directory=root/resource.REPORT_DIR/'figures';directory.mkdir(parents=True,exist_ok=True)
    hashes={}
    for name,fig in figures.items():
        path=directory/name;fig.savefig(path,dpi=150,bbox_inches='tight');hashes[name]=sha256(path)
    write_json(directory/'figure_manifest.json',{'resource_id':summary['resource_id'],
        'visualisation_sha256':sha256(root/'src/remaining_resource_visualisation.py'),
        'example_rule':f'Earliest resource match on/after {EXAMPLE_DATE}; otherwise earliest. No outcome-based selection.',
        'example_match_id':getattr(next(iter(figures.values())),'resource_example_match_id',None),'figure_sha256':hashes})
    return directory
