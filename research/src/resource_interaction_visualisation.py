"""Validation-only figures for the four-way Stage 07 interaction design."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import resource_interaction_experiments as stage
from data_preparation import sha256, write_json

SHORT = dict(zip(stage.TRACKS, ['W1', 'W2', 'M1', 'M2']))
COLORS = dict(RBW='#66758b', IB='#287eae', IW='#bf792b', IBW='#358667')


def design_figure():
    fig, ax = plt.subplots(figsize=(11, 3.8), layout='constrained')
    ax.axis('off')
    rows = [[c, 'State/background + both resources',
             'None' if c == 'RBW' else ('4 batting products' if c == 'IB' else
             '3 bowling products' if c == 'IW' else 'All 7 products')]
            for c in stage.CODES]
    table = ax.table(cellText=rows, colLabels=['Arm', 'Shared inputs', 'Explicit additions'],
                     loc='center', cellLoc='left', colWidths=[.12, .52, .36])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#d9e1e8')
        if r == 0:
            cell.set_facecolor('#e6edf3')
            cell.set_text_props(weight='bold')
    ax.set_title('Resource x nominal remaining-ball fraction (same fraction in all arms)\n'
                 'Original resources retained | Validation-only selection | RBW reused from Stage 06', pad=18)
    return fig


def metric_figure(selected, metric, label):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7), layout='constrained')
    for ax, track in zip(axes.flat, stage.TRACKS):
        table = selected[selected.track.eq(track)].set_index('arm_code').loc[stage.CODES]
        values = table[metric].to_numpy()
        for y, code in enumerate(stage.CODES):
            ax.scatter(values[y], y, s=75, color=COLORS[code], zorder=3)
            ax.annotate(f'{values[y]:.5f}', (values[y], y), xytext=(8, 0),
                        textcoords='offset points', va='center', fontsize=9)
        span = max(np.ptp(values), .002)
        ax.set_xlim(values.min() - span * .35, values.max() + span * 1.1)
        ax.set_ylim(3.5, -.5)
        ax.set_yticks(range(4), stage.CODES)
        ax.set_title(SHORT[track])
        ax.set_xlabel(label)
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Each arm uses its own validation-selected algorithm and parameters\n'
                 'Zoomed axes differ by track; point differences are not significance tests')
    return fig


def gain_figure(comparisons):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout='constrained')
    for ax, track in zip(axes.flat, stage.TRACKS):
        table = comparisons[comparisons.track.eq(track)]
        for index, code in enumerate(stage.CODES[1:]):
            fixed = table[table.treatment.eq(code) & table.comparison_mode.eq('fixed_baseline_parameters')]
            for offset, family in zip(np.linspace(-.24, .24, len(stage.base.FAMILIES)), stage.base.FAMILIES):
                row = fixed[fixed.reference_family.eq(family)].iloc[0]
                ax.scatter(row.validation_log_loss_gain, index + offset, marker='o', s=35,
                           color=COLORS[code], alpha=.65)
                ax.annotate({'logistic_regression':'LR', 'random_forest':'RF', 'hist_gradient_boosting':'HGB', 'xgboost':'XGB'}[family],
                            (row.validation_log_loss_gain, index+offset), xytext=(5, 0),
                            textcoords='offset points', fontsize=7, va='center')
            row = table[table.treatment.eq(code) & table.comparison_mode.eq('selected_system')].iloc[0]
            ax.scatter(row.validation_log_loss_gain, index + .36, marker='D', s=85,
                       edgecolors='black', color=COLORS[code], zorder=4)
        ax.axvline(0, color='#555555', linewidth=.8)
        ax.set_yticks(range(3), stage.CODES[1:])
        ax.set_ylim(2.6, -.6)
        ax.margins(x=.3)
        ax.set_title(SHORT[track])
        ax.set_xlabel('RBW Log Loss minus interaction Log Loss (positive = better)')
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Validation gains versus RBW\nCircles: RBW family parameters held fixed | Diamonds: separately selected systems')
    return fig


def phase_figure(phases):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    for ax, track in zip(axes.flat, stage.TRACKS):
        for code in stage.CODES:
            table = phases[phases.track.eq(track) & phases.arm_code.eq(code)].set_index('phase').reindex(stage.common.PHASES)
            ax.plot(range(3), table.validation_event_log_loss, marker='o', color=COLORS[code], label=code)
        ax.set_xticks(range(3), ['Powerplay', 'Middle', 'Death'])
        ax.set_title(SHORT[track])
        ax.set_ylabel('Validation Log Loss')
        ax.grid(alpha=.2)
    axes[0,0].legend(ncol=2)
    fig.suptitle('Descriptive phase diagnostics | Overall-track selected models\n'
                 'No phase-specific selection; missing phases stay missing')
    return fig


def build_figures(root):
    stage.verified_output(root)
    out = Path(root) / stage.REPORT_DIR
    selected = pd.read_csv(out / 'selected_models.csv')
    return {'design.png': design_figure(),
            'validation_log_loss.png': metric_figure(selected, 'validation_event_log_loss', 'Log Loss (lower is better)'),
            'validation_accuracy.png': metric_figure(selected, 'validation_accuracy', 'Accuracy (higher is better; not selection metric)'),
            'validation_gains.png': gain_figure(pd.read_csv(out / 'validation_comparisons.csv')),
            'phase_diagnostics.png': phase_figure(pd.read_csv(out / 'phase_diagnostics.csv'))}


def save_figures(root, summary, figures):
    folder = Path(root) / stage.REPORT_DIR / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, fig in figures.items():
        fig.savefig(folder / name, dpi=150, bbox_inches='tight')
        hashes[name] = sha256(folder / name)
    write_json(folder / 'figure_manifest.json', dict(experiment_id=summary['experiment_id'],
        visualisation_sha256=sha256(Path(__file__)), test_scored=False, figure_sha256=hashes))
