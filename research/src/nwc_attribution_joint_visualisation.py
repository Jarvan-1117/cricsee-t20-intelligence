"""Stage 09 figures: observed intervals and explicitly partial allocation."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import nwc_attribution_joint as stage
from data_preparation import sha256, write_json


def match_probability_figure(frame, legal_clock=True):
    d=frame.sort_values(['match_innings','source_row']).copy()
    fig,ax=plt.subplots(figsize=(13,5),layout='constrained')
    first=d.first_batting_team.iloc[0]; second=d.second_batting_team.iloc[0]
    offset=0
    for inn,g in d.groupby('match_innings',sort=True):
        if legal_clock:
            # Nominal 120-slot visual clock; keep original legal counts in ledger.
            x=(inn-1)*120+np.minimum(g.source_over.to_numpy()*6+g.audit_pre_legal_in_over.to_numpy(),120)
            end=(inn-1)*120+min(float(g.source_over.iloc[-1]*6+g.audit_pre_legal_in_over.iloc[-1]+g.legal_delivery.iloc[-1]),120)
        else:
            x=offset+np.arange(len(g)); end=offset+len(g)
        ax.plot(x,g.probability_before,label=f'Innings {inn}: observed pre-event probability',linewidth=1.5)
        if inn==2:
            y=float(g.first_batting_team_win.iloc[-1])
            ax.plot([x[-1],end],[g.probability_before.iloc[-1],y],':',color='#b24b45',linewidth=2,label='Terminal closure (not model prediction)')
            ax.scatter([end],[y],marker='s',s=25,color='#b24b45',zorder=4)
        else:
            boundary=120 if legal_clock else len(g)
            # Do not draw a false first-innings post-event probability.
            ax.axvline(boundary,linestyle=':',color='#777')
        offset+=len(g)
    ax.axhline(.5,linestyle='--',color='#777',linewidth=1)
    ax.set_ylim(-.03,1.03); ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel(f'{first} win probability')
    if legal_clock:
        ax.set_xticks([0,30,60,90,120,150,180,210,240],['0','30','60','90','120 / 0','30','60','90','120'])
        ax.set_xlim(-2,242)
        ax.set_xlabel(f'{first} innings (0-120)                         {second} innings (0-120)\nNominal over-slot clock; extras share positions; over-count exceptions capped for display only')
    else:
        ax.set_xlabel('Recorded event boundary (includes wides/no-balls)')
    ax.set_title(f'Match {int(d.match_id.iloc[0])}: {first} vs {second}\nFirst-innings final event + innings transition remain an unallocated bundle')
    ax.grid(alpha=.15); ax.legend(loc='best',fontsize=8)
    return fig


def event_figure(frame):
    d=frame.sort_values(['match_innings','source_row'])
    x=np.arange(1,len(d)+1); allocated=d.player_allocated.to_numpy(bool)
    boundary=d.interval_kind.eq('last_event_plus_innings_transition').to_numpy()
    values=d.interval_wpa_first_team.to_numpy()*100
    fig,ax=plt.subplots(figsize=(13,4),layout='constrained')
    direct=d.allocation_reason.eq('player_direct_dismissal_proxy').to_numpy()
    for mask,color,label in [(allocated&~direct,'#2878a5','Ordinary interval proxy'),
        (direct,'#8c6bb1','Bowled/LBW interval proxy'),
        (~allocated&~boundary,'#c58e47','Team residual'),(boundary,'#777','Last event + innings transition bundle')]:
        ax.bar(x[mask],values[mask],color=color,label=label,width=.9)
    term=d.is_terminal_event.to_numpy(bool)
    ax.scatter(x[term],values[term],marker='D',facecolors='none',edgecolors='#b24b45',s=70,label='Terminal event')
    ax.axhline(0,color='#555',linewidth=.8)
    ax.set_xlabel('Recorded event number across both innings')
    ax.set_ylabel('First-batting-team change (percentage points)')
    ax.set_title('Observed-state probability ledger | not isolated delivery effects')
    ax.legend(fontsize=8); return fig


def coverage_figure(d):
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    tracks=stage.evaluation.TRACKS
    ordinary=np.array([d.loc[d.track.eq(t),'allocation_reason'].eq('player_ordinary_interval_proxy').mean() for t in tracks])
    direct=np.array([d.loc[d.track.eq(t),'allocation_reason'].eq('player_direct_dismissal_proxy').mean() for t in tracks])
    ax.barh(tracks,ordinary,color='#2878a5',label='Ordinary interval proxy')
    ax.barh(tracks,direct,left=ordinary,color='#8c6bb1',label='Bowled/LBW proxy')
    ax.barh(tracks,1-ordinary-direct,left=ordinary+direct,color='#c58e47',label='Team residual')
    ax.set_xlim(0,1); ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel('Fraction of recorded events (not fraction of causal contribution)')
    ax.set_title('Allocation coverage by rule | higher coverage is not proof of better attribution')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=3); return fig


def terminal_figure(d):
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,gender in zip(axes,['female','male']):
        g=d[d.is_terminal_event&d.gender.eq(gender)]
        ax.scatter(g.probability_before,g.interval_wpa_first_team*100,s=16,alpha=.6)
        ax.axhline(0,color='#777',linewidth=.8); ax.set_xlim(0,1)
        ax.set_xlabel('Final pre-event first-team probability')
        ax.set_ylabel('Terminal change (percentage points)'); ax.set_title(gender.title())
    fig.suptitle('Terminal outcome minus pre-event probability | no clipping or smoothing')
    return fig


def player_figure(players, mid):
    # Alphabetical order, not a purported ability ranking; keep batting/bowling separate.
    g=players[players.match_id.eq(mid)].sort_values(['team','player','role'])
    fig,axes=plt.subplots(1,3,sharey=True,figsize=(15,max(5,len(g)*.24)),layout='constrained')
    y=np.arange(len(g)); valid=g.partial_nwc.notna()
    labels=[f'{r.player} ({r.role})  {r.allocated_events}/{r.role_events} allocated'+(' [NA]' if pd.isna(r.partial_nwc) else '') for r in g.itertuples()]
    cols=['partial_nwc_excluding_terminal','terminal_partial_nwc','partial_nwc']
    limit=max(1.,float(g.loc[valid,cols].abs().max().max())*100*1.12) if valid.any() else 1.
    for ax,col,title,color in zip(axes,cols,['Non-terminal','Terminal','Total'],['#2878a5','#c58e47','#667580']):
        ax.barh(y[valid],g.loc[valid,col]*100,color=color)
        ax.axvline(0,color='#777',linewidth=.8); ax.set_xlim(-limit,limit)
        ax.set_title(title); ax.set_xlabel('Percentage points'); ax.grid(axis='x',alpha=.15)
    axes[0].set_yticks(y,labels,fontsize=8); axes[0].invert_yaxis()
    fig.suptitle('Rule-based player interval contributions | non-terminal + terminal = total\nAlphabetical, not an ability ranking. Team residuals excluded; NA means no allocated events.')
    return fig


def build_figures(root, match_id=None):
    root=Path(root)
    stage.verified_output(root)  # Require verified attribution for the plotted cohort.
    d=pd.read_csv(root/stage.DATA_DIR/'event_ledger.csv.gz')
    players=pd.read_csv(root/stage.REPORT_DIR/'player_match_partial_nwc.csv')
    # Deterministic metadata-only choice, never cherry-pick large WPA or a good model fit.
    if match_id is None:
        match_id=int(d.sort_values(['match_date','match_id']).match_id.iloc[0])
    g=d[d.match_id.eq(int(match_id))]
    if g.empty: raise ValueError('Example match must belong to the frozen test cohort.')
    return {'match_probability_120':match_probability_figure(g),
        'match_probability_events':match_probability_figure(g,False),
        'match_interval_changes':event_figure(g),'player_partial_contributions':player_figure(players,int(match_id)),
        'allocation_coverage':coverage_figure(d),'terminal_changes':terminal_figure(d)}


def save_figures(root,summary,figures):
    root=Path(root); out=root/stage.REPORT_DIR/'figures'; out.mkdir(parents=True,exist_ok=True)
    hashes={}
    for name,fig in figures.items():
        path=out/(name+'.png'); fig.savefig(path,dpi=150); hashes[path.name]=sha256(path)
    write_json(out/'figure_manifest.json',{'attribution_id':summary['attribution_id'],
        'visualisation_sha256':sha256(Path(__file__)),'figure_sha256':hashes})
