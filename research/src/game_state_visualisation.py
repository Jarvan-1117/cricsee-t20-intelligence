"""English-only Stage 02 feature diagnostics, never win-probability predictions."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
         "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"}
PHASES = ["powerplay", "middle", "death"]
COLORS = ["#287BB0", "#237B65", "#BD7A17"]


def overview_figure(summary, coverage):
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(12, 5.5), layout="constrained")
        grid = fig.add_gridspec(2, 4, height_ratios=[1, 3])
        for i, (label, value) in enumerate([("Matches", summary["matches"]), ("Innings", summary["innings"]),
                                           ("Pre-event rows", summary["events"]), ("Models trained", 0)]):
            ax = fig.add_subplot(grid[0, i]); ax.axis("off")
            ax.text(.03, .85, label, transform=ax.transAxes, color="#52657B")
            ax.text(.03, .2, f"{value:,}", transform=ax.transAxes, fontsize=25, weight="bold", color="#17365D")
        ax = fig.add_subplot(grid[1, :])
        data = coverage.pivot(index=["gender", "match_innings"], columns="innings_phase", values="events").fillna(0)
        y = np.arange(len(data)); left = np.zeros(len(data))
        for phase, color in zip(PHASES, COLORS):
            values = data.reindex(columns=PHASES, fill_value=0)[phase].to_numpy()
            ax.barh(y, values, left=left, color=color, label=phase.title())
            for position, amount, start in zip(y, values, left):
                if amount: ax.text(start + amount / 2, position, f"{amount:,}", ha="center", va="center", color="white", fontsize=9)
            left += values
        ax.set_yticks(y, [f"{gender.title()} / innings {innings}" for gender, innings in data.index])
        ax.set_xlabel("Recorded events (not legal balls)")
        ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.2))
        fig.suptitle("Stage 02: feature coverage", x=.01, ha="left", weight="bold", fontsize=17)
        return fig


def choose_example(frame):
    # Deterministic choice by event count only; never choose by model performance.
    counts = frame.groupby("match_id").size().sort_values(kind="stable")
    return int(counts.index[len(counts) // 2])


def match_figure(frame, match_id):
    data = frame.loc[frame.match_id.eq(match_id)].sort_values(["match_innings", "event_sequence"])
    if data.empty: raise ValueError("Example match not found")
    first = data.iloc[0]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
        for column, no in enumerate([1, 2]):
            local = data.loc[data.match_innings.eq(no)]
            x = local.event_sequence.to_numpy()
            for ax in axes[:, column]:
                for phase, color in zip(PHASES, COLORS):
                    section = local.loc[local.innings_phase.eq(phase)]
                    if not section.empty:
                        ax.axvspan(section.event_sequence.min() - .5, section.event_sequence.max() + .5, color=color, alpha=.10)
                ax.grid(axis="y", alpha=.18)
                ax.set_xlabel("Recorded event (state BEFORE event)")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax = axes[0, column]
            ax.plot(x, local.pre_score, color="#17365D", label="Pre-event score", lw=2)
            after_wicket = local.pre_wickets.diff().fillna(0).gt(0)
            ax.scatter(local.loc[after_wicket, "event_sequence"], local.loc[after_wicket, "pre_score"],
                       color="#BA493A", marker="x", s=45, label="First state after dismissal", zorder=4)
            ax.set_ylabel("Runs")
            ax.set_title(f"Innings {no}: {local.batting_team.iloc[0]} batting", loc="left", weight="bold")
            ax.legend(frameon=False, fontsize=9)
            ax = axes[1, column]
            ax.plot(x, local.current_run_rate, color="#237B65", label="Current run rate", lw=1.8)
            if no == 2:
                ax.plot(x, local.required_run_rate, color="#BD7A17", label="Required run rate", lw=1.8)
            ax.set_ylabel("Runs per over")
            # Retain extreme late-chase rates without flattening the current
            # rate curve or silently clipping data. Declare the nonlinear axis.
            if no == 2 and local.required_run_rate.max() > 40:
                ax.set_yscale("symlog", linthresh=20)
                ticks = [v for v in [0, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
                         if v <= local.required_run_rate.max() * 1.15]
                ax.set_yticks(ticks, [str(v) for v in ticks])
                ax.set_ylabel("Runs per over (linear to 20, log above)")
            ax.legend(frameon=False, fontsize=9)
        fig.suptitle(f"Match {match_id}: {first.first_batting_team} vs {first.second_batting_team}\n"
                     "Observed features only - no win-probability model fitted", fontsize=15, weight="bold")
        fig.supxlabel("Background: blue = overs 1-6; green = overs 7-15; amber = overs 16-20. Final post-event state is not plotted.", fontsize=9)
        return fig


def history_figure(background):
    parts = []
    for role, name in [("first_team", "first_batting_team"), ("second_team", "second_batting_team")]:
        parts.append(background[["match_date", "gender", name, role + "_prior_win_rate", role + "_recent_win_rate"]].rename(
            columns={name: "team", role + "_prior_win_rate": "long_term", role + "_recent_win_rate": "recent"}))
    data = pd.concat(parts, ignore_index=True)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
        for ax, gender in zip(axes, ["female", "male"]):
            local = data.loc[data.gender.eq(gender)]
            if local.empty:
                ax.text(.5, .5, "No matches", transform=ax.transAxes, ha="center"); continue
            team = local.groupby("team").size().sort_values(ascending=False, kind="stable").index[0]
            local = local.loc[local.team.eq(team)].sort_values("match_date").drop_duplicates("match_date")
            dates = pd.to_datetime(local.match_date)
            ax.plot(dates, local.long_term, color="#287BB0", label="Long-term prior win rate")
            ax.plot(dates, local.recent, color="#BD7A17", label="Recent weighted prior win rate")
            ax.axhline(.5, ls="--", color="#8192A3", lw=.8)
            ax.set_ylim(0, 1); ax.set_ylabel("Historical win rate (not live win probability)")
            ax.set_title(f"{gender.title()}: {team}", loc="left", weight="bold")
            ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", alpha=.18)
        fig.suptitle("Team background available BEFORE each match date", weight="bold", fontsize=15)
        fig.supxlabel("History uses the retained cohort only. Example: most represented team in each gender. Rates include prior smoothing.", fontsize=9)
        return fig


def save_figures(frame, background, summary, coverage, destination):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    example = choose_example(frame)
    figures = {"feature_coverage": overview_figure(summary, coverage),
               "example_pre_delivery_states": match_figure(frame, example),
               "team_history": history_figure(background)}
    for name, fig in figures.items():
        fig.savefig(destination / (name + ".png"), dpi=150, facecolor="white")
        plt.close(fig)
    return example
