"""Stage 04 coverage, inferred availability, nominal quotas and source timeline."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_preparation import sha256, write_json
from player_availability import BAT_SCHEMA, BOWL_SCHEMA, DATA_DIR, REPORT_DIR

STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
         "axes.spines.right": False, "figure.facecolor": "white"}
COLORS = {"ready": "#287F71", "review_required": "#C58B32", "quarantined": "#B7545D"}


def overview_figure(summary, coverage):
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(12, 5.6), layout="constrained")
        grid = fig.add_gridspec(2, 4, height_ratios=[1, 2.5])
        cards = [("Stage 01 matches (unchanged)", summary["matches"]),
                 ("Events replayed", summary["replayed_events"]),
                 ("Ready for resource features", summary["ready_matches"]),
                 ("Review / quarantined matches", summary["review_matches"] + summary["quarantined_matches"])]
        for index, (label, value) in enumerate(cards):
            ax = fig.add_subplot(grid[0, index]); ax.axis("off")
            ax.text(.02, .8, label, transform=ax.transAxes, fontsize=9, color="#52657B")
            ax.text(.02, .15, f"{value:,}", transform=ax.transAxes, fontsize=25, weight="bold", color="#17365D")
        ax = fig.add_subplot(grid[1, :]); genders = ["female", "male"]
        left = np.zeros(2)
        for status, color in COLORS.items():
            values = np.array([coverage.loc[coverage.gender.eq(g) & coverage.status.eq(status), "matches"].sum() for g in genders])
            ax.barh(genders, values, left=left, color=color, label=status.replace("_", " ").title())
            for y, start, count in zip(genders, left, values):
                if count >= 20: ax.text(start + count / 2, y, str(int(count)), ha="center", va="center", color="white")
            left += values
        ax.set_xlabel("Whole matches; review status is audit-only, not a predictor")
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.2))
        fig.suptitle("Stage 04: availability reconstruction and resource-readiness gate", fontsize=15, weight="bold")
        return fig


def batting_figure(frame):
    if frame.empty:
        raise ValueError("No example events")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained", sharey=True)
        for inn, ax in enumerate(axes, 1):
            part = frame.loc[frame.match_innings.eq(inn)]
            if part.empty:
                ax.text(.5, .5, "No replayed innings", transform=ax.transAxes, ha="center"); continue
            ax.step(part.event_sequence, part.available_batter_count, where="post", label="Inferred available (includes current pair)", color="#287F71", lw=2)
            ax.step(part.event_sequence, part.waiting_batter_count, where="post", label="Waiting (excludes current pair)", color="#277DA8", lw=1.7)
            ax.step(part.event_sequence, part.temporarily_unavailable_batter_count, where="post", label="Recorded temporary unavailability", color="#C58B32", lw=1.2)
            ax.set_title(f"Innings {inn}: {part.batting_team.iloc[0]}", loc="left", weight="bold")
            ax.set_xlabel("Recorded event within innings (pre-event state)")
            ax.set_ylim(-.4, 11.8); ax.set_yticks(range(0, 12, 2)); ax.grid(axis="y", alpha=.2)
        axes[0].set_ylabel("Players")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=1, frameon=False, fontsize=9)
        fig.suptitle(f"Match {int(frame.match_id.iloc[0])}: available is not the same as 10 minus wickets", fontsize=14, weight="bold")
        return fig


def bowling_figure(frame, names):
    if frame.empty:
        raise ValueError("No example events")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained", sharey=True)
        for inn, ax in enumerate(axes, 1):
            part = frame.loc[frame.match_innings.eq(inn)]
            if part.empty: continue
            snapshots = [{a[0]: dict(zip(BOWL_SCHEMA, a)) for a in json.loads(s)} for s in part.bowling_state_json]
            # Retrospective display selection ONLY, not a feature or eligibility rule.
            shown = list(dict.fromkeys(part.bowler_id))
            for pid in shown:
                values = [s.get(pid, {}).get("nominal_capacity", np.nan) for s in snapshots]
                ax.step(part.event_sequence, values, where="post", lw=1.5, label=names.get(pid, pid))
            ax.set_title(f"Innings {inn}: {part.bowling_team.iloc[0]}", loc="left", weight="bold")
            ax.set_xlabel("Recorded event within innings (pre-event state)")
            ax.set_ylim(-.5, 25); ax.set_yticks([0, 6, 12, 18, 24]); ax.grid(axis="y", alpha=.2)
            ax.legend(frameon=False, fontsize=8, loc="upper right")
        axes[0].set_ylabel("Nominal remaining ball slots per player")
        fig.suptitle(f"Match {int(frame.match_id.iloc[0])}: bowling capacities, not future allocations", fontsize=14, weight="bold")
        fig.supxlabel("Shown: players observed bowling somewhere in this match (display only). Capacities cannot simply be added as innings balls.", fontsize=9)
        return fig


def timeline_figure(frame, transitions, names):
    selected = transitions.loc[transitions.match_id.eq(int(frame.match_id.iloc[0])) & transitions.event.ne("dismissal")].copy()
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), layout="constrained")
        for inn, ax in enumerate(axes, 1):
            events = selected.loc[selected.match_innings.eq(inn)]
            part = frame.loc[frame.match_innings.eq(inn)]
            ax.set_title(f"Innings {inn}", loc="left", weight="bold")
            ax.set_xlabel("Recorded event; +0.5 denotes a consequence AFTER that event")
            if events.empty:
                ax.text(.5, .5, "No recorded special transitions", transform=ax.transAxes, ha="center", color="#52657B")
                ax.set_yticks([]); continue
            players = list(dict.fromkeys(events.player_id.fillna("unknown")))
            for number, row in enumerate(events.itertuples(index=False)):
                pid = row.player_id if pd.notna(row.player_id) else "unknown"
                y = players.index(pid); x = row.event_sequence + (.5 if row.timing == "after" else 0)
                color = "#C58B32" if row.timing == "after" else "#287F71"
                ax.scatter([x], [y], marker="D" if row.timing == "after" else "o", color=color, s=50, zorder=3)
                ax.annotate(row.event.replace("_", " "), (x, y), xytext=(5, 12 if number % 2 == 0 else -20),
                            textcoords="offset points", fontsize=8, color=color)
            ax.set_yticks(range(len(players)), [names.get(p, p) for p in players])
            ax.set_ylim(-.7, len(players) - .3)
            max_event = int(part.event_sequence.max()) if len(part) else int(events.event_sequence.max())
            ax.set_xlim(0, max_event + max(25, max_event * .15)); ax.grid(axis="x", alpha=.2)
        fig.suptitle(f"Match {int(frame.match_id.iloc[0])}: recorded exits, returns and replacements", fontsize=14, weight="bold")
        return fig


def build_figures(root, summary):
    root = Path(root); report = root / REPORT_DIR
    catalogue = pd.read_csv(root / "data/processed/historical_features/participant_registry.csv")
    example = "retirement" if "retirement" in summary["example_matches"] else "ordinary"
    frame = pd.read_csv(report / f"example_{example}.csv.gz")
    local_names = catalogue.loc[catalogue.match_id.eq(int(frame.match_id.iloc[0]))]
    names = dict(zip(local_names.player_id, local_names.player_name))
    ordinary = pd.read_csv(report / "example_ordinary.csv.gz") if "ordinary" in summary["example_matches"] else frame
    ordinary_names = catalogue.loc[catalogue.match_id.eq(int(ordinary.match_id.iloc[0]))]
    transitions = pd.read_csv(root / DATA_DIR / "availability_transitions.csv.gz")
    return {
        "availability_coverage.png": overview_figure(summary, pd.read_csv(report / "coverage.csv")),
        "batting_availability.png": batting_figure(frame),
        "bowling_capacity.png": bowling_figure(ordinary, dict(zip(ordinary_names.player_id, ordinary_names.player_name))),
        "availability_timeline.png": timeline_figure(frame, transitions, names),
    }


def save_figures(root, summary, figures):
    root = Path(root); directory = root / REPORT_DIR / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, figure in figures.items():
        path = directory / name
        figure.savefig(path, dpi=150, bbox_inches="tight")
        hashes[name] = sha256(path)
    write_json(directory / "figure_manifest.json", {"availability_id": summary["availability_id"],
        "visualisation_sha256": sha256(root / "src/player_availability_visualisation.py"), "figure_sha256": hashes})
    return directory
