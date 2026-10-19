"""Validation-only Stage 06 presentation; never loads or scores test data."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_preparation import sha256, write_json
from model_comparison import REPORT_DIR, ARM_CODES, ARMS, FAMILIES, SPLITS

TRACKS = ['female_innings_1', 'female_innings_2', 'male_innings_1', 'male_innings_2']
SHORT = {'logistic_regression': 'Logistic', 'random_forest': 'Random forest',
         'hist_gradient_boosting': 'Hist. gradient boost', 'xgboost': 'XGBoost', 'constant': 'Constant'}
LABELS = {'constant': 'R0: Constant', 'state': 'R1: State', 'batting_resources': 'RB: + Batting',
          'bowling_resources': 'RW: + Bowling', 'both_resources': 'RBW: + Both'}
COLORS = ['#3975b5', '#e69b35', '#369776']
FAMILY_COLORS = COLORS + ['#9564ad']


def split_figure(frame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    for ax, gender in zip(axes, ['female', 'male']):
        group = frame[frame.gender.eq(gender)].set_index('split').reindex(SPLITS)
        bars = ax.bar(SPLITS, group.matches, color=COLORS)
        ax.bar_label(bars, padding=4)
        ax.set_ylim(0, group.matches.max() * 1.2)
        ax.set_title(gender.title() + ' - whole matches')
        ax.set_ylabel('Matches')
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Shared chronological splits | Test set reserved, not evaluated')
    return fig


def heatmap_figure(winners):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    for ax, track in zip(axes.flat, TRACKS):
        group = winners[winners.track.eq(track) & winners.family.ne('constant')]
        matrix = group.pivot(index='arm', columns='family', values='validation_event_log_loss').reindex(index=ARMS, columns=FAMILIES)
        values = matrix.to_numpy()
        if not np.isfinite(values).all():
            raise ValueError('Incomplete family comparison; do not plot missing arms as zero.')
        ax.imshow(values, cmap='YlGnBu', aspect='auto', vmin=values.min() - .005, vmax=values.max() + .02)
        midpoint = (values.max() + values.min()) / 2
        for (r, c), value in np.ndenumerate(values):
            ax.text(c, r, f'{value:.4f}', ha='center', va='center', color='white' if value > midpoint else '#172635')
        ax.set_xticks(range(len(FAMILIES)), [SHORT[f] for f in FAMILIES], rotation=12)
        ax.set_yticks(range(len(ARMS)), [LABELS[a] for a in ARMS])
        ax.set_title(track.replace('_', ' ').title())
    fig.suptitle('Validation log loss - lower is better\nBest parameter setting per algorithm/arm; colour scales differ by panel')
    return fig


def selected_figure(selected):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    for ax, track in zip(axes.flat, TRACKS):
        group = selected[selected.track.eq(track)].set_index('arm').reindex(ARM_CODES)
        values = group.validation_event_log_loss.to_numpy()
        y = np.arange(len(group))
        ax.barh(y, values, color=['#9ca6ae'] + ['#3975b5'] * (len(ARM_CODES) - 1))
        for i, (_, row) in enumerate(group.iterrows()):
            ax.text(row.validation_event_log_loss + .008, i, f'{row.validation_event_log_loss:.4f}  {SHORT[row.family]}', va='center', fontsize=8)
        ax.set_yticks(y, [LABELS[a] for a in group.index]); ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.55)
        ax.set_xlabel('Validation log loss (lower is better)')
        ax.set_title(track.replace('_', ' ').title())
    fig.suptitle('Selected model per feature group | Train-only fits, validation selection')
    return fig


def gains_figure(deltas):
    treatments = ['batting_resources', 'bowling_resources', 'both_resources']
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    for ax, track in zip(axes.flat, TRACKS):
        group = deltas[deltas.track.eq(track) & deltas.view.eq('within_family') & deltas.reference.eq('state')]
        y = np.arange(len(treatments))
        width = .69 / len(FAMILIES)
        for index, family in enumerate(FAMILIES):
            values = group[group.treatment_family.eq(family)].set_index('treatment').reindex(treatments).validation_log_loss_gain
            ax.barh(y + (index - (len(FAMILIES) - 1) / 2) * width, values,
                    height=width * .95, label=SHORT[family], color=FAMILY_COLORS[index])
        ax.axvline(0, color='#333333', linewidth=1)
        ax.set_yticks(y, [LABELS[t] for t in treatments]); ax.invert_yaxis()
        ax.set_xlabel('R1 log loss minus resource-arm log loss\nPositive = better than state/background')
        ax.set_title(track.replace('_', ' ').title()); ax.grid(axis='x', alpha=.2)
    axes[0, 0].legend(fontsize=8, loc='best')
    fig.suptitle('Resource gains within the same algorithm | Validation only; no significance claim')
    return fig


def build_figures(root):
    out = Path(root) / REPORT_DIR
    return {'split_counts.png': split_figure(pd.read_csv(out / 'split_summary.csv')),
            'validation_family_heatmap.png': heatmap_figure(pd.read_csv(out / 'family_winners.csv')),
            'selected_models.png': selected_figure(pd.read_csv(out / 'selected_models.csv')),
            'resource_validation_gains.png': gains_figure(pd.read_csv(out / 'validation_deltas.csv'))}


def save_figures(root, summary, figures):
    directory = Path(root) / REPORT_DIR / 'figures'
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, figure in figures.items():
        figure.savefig(directory / name, dpi=150, bbox_inches='tight')
        hashes[name] = sha256(directory / name)
    write_json(directory / 'figure_manifest.json', {'experiment_id': summary['experiment_id'],
        'visualisation_sha256': sha256(Path(__file__)), 'test_scored': False, 'figure_sha256': hashes})
