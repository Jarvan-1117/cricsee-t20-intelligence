"""Presentation-only views of verified Stage 01 reports; no cleaning or training."""
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import pandas as pd
from IPython.display import HTML

COLORS = {"included": "#237B65", "excluded": "#697C91", "pending_review": "#BD7A17",
          "female": "#7458A5", "male": "#287BB0", "text": "#243447"}
STYLE = {"font.family": "DejaVu Sans", "font.size": 11,
         "axes.titlesize": 14, "axes.titleweight": "bold", "axes.labelcolor": COLORS["text"],
         "text.color": COLORS["text"], "axes.spines.top": False,
         "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"}
REASONS = {"outside_team_scope": "Outside the 11-team scope",
           "reduced_or_nonstandard_overs": "Reduced / nonstandard over limit",
           "adjusted_method": "Adjusted-method result (e.g. DLS)",
           "no_result": "No result", "tie": "Tie", "super_over": "Super Over",
           "awarded": "Awarded outcome", "declared_or_forfeited": "Declared / forfeited innings"}


def table(frame, title, collapsed=False):
    """Readable, escaped HTML; long tables remain inspectable, not truncated."""
    html = frame.to_html(index=False, border=0, escape=True, na_rep="Not available")
    content = ('<div style="overflow-x:auto;max-width:1100px">' + html + '</div>')
    style = """<style>.audit-table table{border-collapse:collapse;width:100%;font-size:13px}
    .audit-table th{text-align:left;border-bottom:2px solid #8192a3;padding:9px}
    .audit-table td{padding:8px 9px;border-bottom:1px solid #dce3eb;vertical-align:top}
    .audit-table summary{cursor:pointer;padding:12px 0;font-weight:600}
    .audit-table{margin:10px 0 20px}</style>"""
    if collapsed:
        content = f'<details><summary>{escape(title)} ({len(frame):,} rows)</summary>{content}</details>'
    else:
        content = f'<h4>{escape(title)}</h4>{content}'
    return HTML(style + '<div class="audit-table">' + content + '</div>')


def _count_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E4E9EF", linewidth=.7)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))


def overview_figure(summary):
    counts = summary["match_counts"]
    total = summary["audited_matches"]
    if total <= 0 or sum(counts.values()) != total:
        raise ValueError("Overview requires nonempty, reconciled match counts.")
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(12, 4.3), layout="constrained")
        grid = fig.add_gridspec(2, 4, height_ratios=[1, 1.25])
        items = [("Source matches", total, COLORS["text"]),
                 ("Included", counts["included"], COLORS["included"]),
                 ("Excluded", counts["excluded"], COLORS["excluded"]),
                 ("Pending review", counts["pending_review"], COLORS["pending_review"])]
        for i, (label, value, color) in enumerate(items):
            ax = fig.add_subplot(grid[0, i]); ax.axis("off")
            ax.text(.03, .78, label, transform=ax.transAxes, fontsize=11)
            ax.text(.03, .27, f"{value:,}", transform=ax.transAxes, fontsize=29, color=color, weight="bold")
            ax.plot([.03, .9], [.1, .1], transform=ax.transAxes, color=color, lw=2)
        ax = fig.add_subplot(grid[1, :])
        left = 0
        for status, label in [("included", "Included"), ("excluded", "Excluded"), ("pending_review", "Pending review")]:
            value = counts[status]
            ax.barh(0, value, left=left, color=COLORS[status], height=.38,
                    label=f"{label}: {value:,} ({value / total:.1%})")
            if value / total > .08:
                ax.text(left + value / 2, 0, f"{value / total:.1%}", ha="center", va="center", color="white", weight="bold")
            left += value
        ax.set_xlim(0, total); ax.set_ylim(-.5, .6); ax.set_yticks([])
        ax.set_xlabel("Matches (mutually exclusive decisions)")
        _count_axis(ax)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.3), ncol=3, frameon=False, fontsize=10)
        fig.suptitle("Stage 01: initial audit results", x=.01, ha="left", fontsize=17, weight="bold")
        return fig


def exclusion_figure(reasons, excluded_count):
    data = reasons.loc[reasons.status.eq("excluded")].sort_values("matches", ascending=True)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(12, max(3.3, .43 * len(data) + 1.6)), layout="constrained")
        if data.empty:
            ax.text(.5, .5, "No excluded matches", ha="center", transform=ax.transAxes)
            ax.set_axis_off()
        else:
            labels = [REASONS.get(r, r.replace("_", " ")) for r in data.reason]
            bars = ax.barh(labels, data.matches, color=COLORS["excluded"], height=.62)
            ax.bar_label(bars, labels=[f"{int(v):,}" for v in data.matches], padding=6)
            ax.set_xlim(0, max(data.matches) * 1.14)
            ax.set_xlabel("Matches flagged for each reason")
            _count_axis(ax)
        ax.set_title(f"Why {excluded_count:,} matches were excluded", loc="left", pad=15)
        fig.supxlabel("Reasons overlap. A match may appear in several bars; do not add these counts.", fontsize=10)
        return fig


def cohort_tables(included):
    dates = pd.to_datetime(included.match_date, errors="raise")
    if dates.isna().any() or included.empty:
        raise ValueError("Included cohort requires nonmissing dates and at least one match.")
    if not included.gender.isin(["female", "male"]).all():
        raise ValueError("Unexpected gender category in included cohort.")
    gender = included.groupby("gender").agg(matches=("match_id", "size"), events=("csv_events", "sum")).reindex(["female", "male"], fill_value=0)
    annual = included.assign(year=dates.dt.year).groupby(["year", "gender"]).size().unstack(fill_value=0)
    annual = annual.reindex(index=range(dates.dt.year.min(), dates.dt.year.max() + 1), columns=["female", "male"], fill_value=0)
    return gender, annual, dates.min(), dates.max()


def cohort_figure(included):
    gender, annual, earliest, latest = cohort_tables(included)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 2.4]}, layout="constrained")
        ax, trend = axes
        bars = ax.barh(["Female", "Male"], gender.matches, color=[COLORS[g] for g in gender.index], height=.5)
        ax.bar_label(bars, padding=5, labels=[f"{v:,}" for v in gender.matches])
        ax.invert_yaxis(); ax.set_xlim(0, gender.matches.max() * 1.28)
        ax.set_title("Included matches", loc="left"); ax.set_xlabel("Matches"); _count_axis(ax)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        for g, marker in [("female", "o"), ("male", "s")]:
            trend.plot(annual.index, annual[g], color=COLORS[g], marker=marker, markersize=4, label=g.title(), linewidth=2)
        trend.set_title("Included matches by year", loc="left")
        trend.set_ylim(bottom=0); trend.set_xlabel("Year"); trend.set_ylabel("Matches")
        trend.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
        trend.yaxis.set_major_locator(MaxNLocator(integer=True))
        trend.grid(axis="y", color="#E4E9EF"); trend.legend(frameon=False)
        fig.supxlabel(f"Included match dates: {earliest:%Y-%m-%d} to {latest:%Y-%m-%d}. Boundary years may be partial; this is not a train/test split.", fontsize=9)
        return fig


def pending_table(pending):
    result = pending[["match_id", "match_date", "gender", "csv_events"]].copy()
    result["Review needed"] = pending.pending_reasons.map(
        lambda s: "Verify innings ended with no available batting pair" if s == "injury_related_completion_needs_review;innings_completion_unresolved" else s.replace(";", "; "))
    return result.rename(columns={"match_id": "Match ID", "match_date": "Date", "gender": "Gender", "csv_events": "Events"})


def special_case_table(innings, decisions, corrections):
    """Select factual examples from current reports, never hardcoded match IDs."""
    cases = []
    eligible = innings.loc[innings.match_id.isin(decisions.loc[decisions.status.eq("included"), "match_id"])]
    selections = [
        ("Documented miscount", eligible.miscounted_overs.ne("{}"), "miscounted_overs", "Retained after event verification"),
        ("Pre-innings penalty", eligible.penalty_runs_pre.ne(0), "penalty_runs_pre", "Included in the initial innings score"),
        ("Post-innings penalty", eligible.penalty_runs_post.ne(0), "penalty_runs_post", "Official total / target adjusted, not earlier deliveries"),
    ]
    for label, mask, field, handling in selections:
        for _, row in eligible.loc[mask].head(1).iterrows():
            cases.append({"Case": label, "Match ID": int(row.match_id), "Evidence": f"Innings {int(row.match_innings)}: {row[field]}", "Handling": handling})
    retirements = corrections.loc[corrections.field.eq("wicket_kind") & corrections.canonical_value.isin(["retired hurt", "retired not out"])]
    for _, row in retirements.head(1).iterrows():
        cases.append({"Case": "Temporary retirement", "Match ID": int(row.match_id), "Evidence": row.canonical_value, "Handling": "Retirement recorded without adding a wicket"})
    if "audit_completion_basis" in eligible:
        for _, row in eligible.loc[eligible.audit_completion_basis.eq("no_available_batting_pair")].head(1).iterrows():
            cases.append({"Case": "No available batting pair", "Match ID": int(row.match_id),
                          "Evidence": f"{int(row.final_wickets)} dismissals; one available batter remains",
                          "Handling": "Normal completion verified from roster, absence and retirement evidence"})
    for _, row in decisions.loc[decisions.status.eq("pending_review")].head(1).iterrows():
        cases.append({"Case": "Unresolved ending", "Match ID": int(row.match_id), "Evidence": row.pending_reasons.replace(";", "; "), "Handling": "Pending, not published to the clean cohort"})
    return pd.DataFrame(cases, columns=["Case", "Match ID", "Evidence", "Handling"])


def save_figure(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name + ".png")
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    return path
