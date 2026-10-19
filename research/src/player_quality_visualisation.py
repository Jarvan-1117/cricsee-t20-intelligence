"""Stage 03 evidence and metric diagnostics; no model performance claims."""
import matplotlib.pyplot as plt
import numpy as np

STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
         "axes.spines.right": False, "figure.facecolor": "white"}
BANDS = ["No prior balls", "1-119 prior balls", "120+ prior balls"]
COLORS = ["#9BA9B7", "#D39635", "#267F72"]
GENDER_COLORS = {"female": "#7458A5", "male": "#287BB0"}


def coverage_figure(summary, coverage):
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(12, 5.7), layout="constrained")
        grid = fig.add_gridspec(2, 4, height_ratios=[1, 3])
        values = [("Matches", summary["matches"]), ("Player-match profiles", summary["profiles"]),
                  ("Current-player event rows", summary["events"]),
                  ("Unique players (within gender)", sum(summary["players_by_gender"].values()))]
        for i,(label,value) in enumerate(values):
            ax=fig.add_subplot(grid[0,i]); ax.axis("off")
            ax.text(.02,.8,label,transform=ax.transAxes,fontsize=9,color="#52657B")
            ax.text(.02,.2,f"{value:,}",transform=ax.transAxes,fontsize=25,weight="bold",color="#17365D")
        ax=fig.add_subplot(grid[1,:])
        table=coverage.pivot(index=["gender","role"],columns="history_band",values="player_match_profiles").reindex(columns=BANDS,fill_value=0).fillna(0)
        totals=table.sum(axis=1).to_numpy(); left=np.zeros(len(table)); y=np.arange(len(table))
        for label,color in zip(BANDS,COLORS):
            fractions=table[label].to_numpy()/totals*100
            ax.barh(y,fractions,left=left,label=label,color=color)
            for pos,v,start in zip(y,fractions,left):
                if v>5: ax.text(start+v/2,pos,f"{v:.1f}%",ha="center",va="center",color="white",fontsize=10)
            left+=fractions
        ax.set_yticks(y,[f"{g.title()} / {role}" for g,role in table.index])
        ax.set_xlim(0,100); ax.set_xlabel("Listed player-match profiles (%) - not unique players or active lineups")
        ax.legend(ncol=3,loc="upper center",bbox_to_anchor=(.5,1.22),frameon=False)
        fig.suptitle("Stage 03: how much prior player history is available?",fontsize=16,weight="bold",x=.01,ha="left")
        return fig


def distribution_figure(profiles):
    specs=[("batting_strike_rate","Batting strike rate","Runs per 100 faced events",1),
           ("batting_average","Batting average","Runs per dismissal",1),
           ("bowling_economy","Bowling economy","Runs conceded per 6 legal balls",1),
           ("bowling_wicket_rate","Bowler wicket rate","Credited wickets per 100 legal balls",100)]
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(2,2,figsize=(12,7.8),layout="constrained")
        for ax,(metric,title,label,multiplier) in zip(axes.flat,specs):
            column="smoothed_"+metric
            bins=np.histogram_bin_edges(profiles[column]*multiplier,bins=35)
            for gender,color in GENDER_COLORS.items():
                values=profiles.loc[profiles.gender.eq(gender),column]*multiplier
                if not values.empty: ax.hist(values,bins=bins,density=True,histtype="step",lw=1.8,label=gender.title(),color=color)
            ax.set_title(title,loc="left",weight="bold"); ax.set_xlabel(label); ax.set_ylabel("Density")
            ax.legend(frameon=False); ax.grid(axis="y",alpha=.18)
        fig.suptitle("Separate, smoothed quality dimensions",fontsize=16,weight="bold")
        fig.supxlabel("All listed player-match snapshots, including cold starts; repeated players are not independent observations. No ranking or model accuracy is inferred.",fontsize=9)
        return fig


def smoothing_figure(profiles):
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout="constrained")
        for ax,gender in zip(axes,["female","male"]):
            # One first observed sparse-history snapshot per ID: reproducible,
            # no performance-based selection, no oversized dense cloud.
            group=profiles.loc[profiles.gender.eq(gender)&profiles.prior_balls_faced.between(1,119)].sort_values(["match_date","match_id"]).drop_duplicates("player_id")
            if group.empty:
                ax.text(.5,.5,"No sparse-history profiles",transform=ax.transAxes,ha="center"); continue
            x=group.raw_batting_strike_rate; y=group.smoothed_batting_strike_rate
            points=ax.scatter(x,y,c=group.prior_balls_faced,cmap="viridis",s=22,alpha=.7,vmin=1,vmax=119)
            bound=max(float(x.max()),float(y.max()))*1.05
            ax.plot([0,bound],[0,bound],ls="--",color="#8192A3",lw=1,label="Raw = smoothed")
            ax.set_xlim(0,bound); ax.set_ylim(0,bound)
            ax.set_xlabel("Raw batting strike rate"); ax.set_ylabel("Smoothed batting strike rate")
            ax.set_title(f"{gender.title()}: {len(group):,} player snapshots",loc="left",weight="bold")
            ax.legend(frameon=False,fontsize=9)
            fig.colorbar(points,ax=ax,label="Prior faced events",shrink=.8)
        fig.suptitle("Sparse history: individual estimates shrink toward a prior-date reference",fontsize=14,weight="bold")
        fig.supxlabel("First snapshot with 1-119 prior faced events for each player. Unseen players use the reference and are omitted from this raw-versus-smoothed plot.",fontsize=9)
        return fig
