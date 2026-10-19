"""Stage 08 validation-selected charts. Fixed arm order, never test-driven ranking."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import probability_evaluation_joint as stage
from data_preparation import sha256, write_json

COLORS=dict(zip(stage.ARM_ORDER,['#aaaaaa','#808080','#2878a5','#35916f','#d18b2c','#9467bd','#c45b82','#4169b1']))


def performance_figure(metrics):
    fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
    for ax,track in zip(axes.flat,stage.TRACKS):
        part=metrics[metrics.track.eq(track)]
        if len(part)!=8 or set(part.arm_code)!=set(stage.ARM_ORDER): raise ValueError('Incomplete eight-arm metrics.')
        part=part.set_index('arm_code').loc[stage.ARM_ORDER]
        ax.barh(range(8),part.test_event_log_loss,color=['#2878a5' if p else '#b4c7d8' for p in part.preferred])
        ax.set_yticks(range(8),[a+(' *' if part.loc[a,'preferred'] else '') for a in stage.ARM_ORDER])
        ax.invert_yaxis(); ax.set_xlim(left=0); ax.grid(axis='x',alpha=.2)
        ax.set_title(track.replace('_',' ').title()); ax.set_xlabel('Test event log loss (lower is better)')
    fig.suptitle('Eight frozen comparison arms | * = feature group, algorithm and parameters selected on validation\nFixed arm order, not test ranking')
    return fig


def calibration_figure(bins,arms,title):
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for ax,track in zip(axes.flat,stage.TRACKS):
        ax.plot([0,1],[0,1],'--',color='#777',label='Perfect calibration')
        for arm in arms:
            part=bins[bins.track.eq(track)&bins.arm_code.eq(arm)].sort_values('bin')
            if len(part)!=10 or set(part.bin)!=set(range(10)): raise ValueError('Incomplete calibration bins.')
            ax.plot(part.mean_probability,part.observed_win_rate,'o-',markersize=4,color=COLORS[arm],label=arm)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted win probability',ylabel='Observed win fraction')
        ax.set_title(track.replace('_',' ').title()); ax.grid(alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=len(labels))
    fig.suptitle(title+'\nTen fixed-width bins; empty bins are gaps; no fitted calibration')
    return fig


def preferred_bin_support_figure(bins,metrics):
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,track in zip(axes.flat,stage.TRACKS):
        selected=metrics[metrics.track.eq(track)&metrics.preferred]
        if len(selected)!=1: raise ValueError('Expected one validation-selected model.')
        arm=selected.arm_code.iloc[0]
        part=bins[bins.track.eq(track)&bins.arm_code.eq(arm)].sort_values('bin')
        ax.bar(range(10),part.events,color='#b4c7d8')
        for i,row in enumerate(part.itertuples()): ax.annotate(str(row.matches),(i,row.events),ha='center',va='bottom',fontsize=8)
        ax.set_xticks(range(10),[f'{i/10:.1f}' for i in range(10)])
        ax.set_ylim(0,max(float(part.events.max()),1)*1.2)
        ax.set_title(track.replace('_',' ').title()+' | '+arm)
        ax.set_xlabel('Probability bin lower edge'); ax.set_ylabel('Events')
    fig.suptitle('Support for four validation-selected models\nBars = events; labels = matches per bin (not additive)')
    return fig


def comparison_figure(comparisons,metric):
    fig,axes=plt.subplots(2,2,figsize=(14,12),layout='constrained')
    order=[tuple(p) for p in stage.PAIRS]
    for ax,track in zip(axes.flat,stage.TRACKS):
        part=comparisons[comparisons.track.eq(track)&comparisons.metric.eq(metric)]
        if len(part)!=len(order) or part.duplicated(['treatment','reference']).any(): raise ValueError('Incomplete comparisons.')
        part=part.set_index(['treatment','reference']).loc[order]
        for i,row in enumerate(part.itertuples()):
            color='#2878a5' if row.comparison_role=='primary' else '#7b8794'
            ax.plot([row.ci_lower,row.ci_upper],[i,i],color=color,linewidth=2)
            ax.plot(row.gain,i,'o',color=color)
        ax.axvline(0,color='#333',linewidth=1); ax.grid(axis='x',alpha=.2)
        ax.set_yticks(range(len(order)),[f'{a} vs {b}' for a,b in order]); ax.invert_yaxis()
        ax.set_title(track.replace('_',' ').title()); ax.set_xlabel('Reference loss minus treatment loss; positive = improvement')
    fig.suptitle(metric.replace('_',' ').title()+' | Paired whole-match bootstrap\n95% pointwise intervals; not multiplicity-adjusted')
    return fig


def importance_figure(importance,unit_type,status):
    fig,axes=plt.subplots(2,2,figsize=(17,17 if unit_type=='feature' else 9),layout='constrained')
    for ax,track in zip(axes.flat,stage.TRACKS):
        record=status[status.track.eq(track)]
        if len(record)!=1: raise ValueError('Missing importance status.')
        arm=record.arm_code.iloc[0]
        ax.set_title(track.replace('_',' ').title()+' | '+arm)
        part=importance[importance.track.eq(track)&importance.unit_type.eq(unit_type)]
        if record.status.iloc[0]=='not_applicable_constant_model':
            ax.text(.5,.5,'Not applicable: constant model\nNo input features',ha='center',va='center',transform=ax.transAxes)
            ax.set_axis_off(); continue
        if part.empty or part.unit.duplicated().any() or set(part.arm_code)!={arm}: raise ValueError('Invalid importance table.')
        part=part.sort_values(['importance_mean','unit'],ascending=[False,True])
        labels=[c.replace('progress_interaction_','progress x ').replace('waiting_resource_batting_','bat: ')
                 .replace('bowling_resource_bowling_','bowl: ').replace('_',' ') for c in part.unit]
        y=np.arange(len(part))
        ax.barh(y,part.importance_mean,color='#2878a5')
        ax.errorbar(part.importance_mean,y,xerr=part.importance_std,fmt='none',ecolor='#555',capsize=2)
        ax.set_yticks(y,labels,fontsize=8); ax.invert_yaxis(); ax.axvline(0,color='#333'); ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('Test log-loss increase after input-column permutation')
    fig.suptitle('Validation-selected model reliance | '+unit_type.title()+' importance\n'
                 'Mean +/- shuffle SD, not confidence intervals; products shuffled as input columns')
    return fig


def build_figures(root):
    out=Path(root)/stage.REPORT_DIR
    metrics=pd.read_csv(out/'test_metrics.csv'); bins=pd.read_csv(out/'calibration_bins.csv')
    comparisons=pd.read_csv(out/'paired_comparisons.csv'); importance=pd.read_csv(out/'feature_importance.csv')
    status=pd.read_csv(out/'importance_status.csv')
    return {'test_log_loss.png':performance_figure(metrics),
        'calibration_main.png':calibration_figure(bins,['R0','R1','RB','RW','RBW'],'Stage 06 arms'),
        'calibration_interactions.png':calibration_figure(bins,['RBW','IB','IW','IBW'],'Stage 07 representations'),
        'preferred_bin_support.png':preferred_bin_support_figure(bins,metrics),
        'paired_log_loss.png':comparison_figure(comparisons,'event_log_loss'),
        'paired_brier.png':comparison_figure(comparisons,'brier'),
        'feature_importance.png':importance_figure(importance,'feature',status),
        'resource_block_importance.png':importance_figure(importance,'block',status)}


def save_figures(root,summary,figures):
    out=Path(root)/stage.REPORT_DIR/'figures'; out.mkdir(parents=True,exist_ok=True)
    hashes={}
    for name,fig in figures.items():
        fig.savefig(out/name,dpi=150,bbox_inches='tight'); hashes[name]=sha256(out/name)
    write_json(out/'figure_manifest.json',dict(evaluation_id=summary['evaluation_id'],
        visualisation_sha256=sha256(Path(__file__)),figure_sha256=hashes))
