"""
viz.py — Shared Plotly visualization helpers for CIV MCP Demo Scenarios.

All functions accept a pandas DataFrame (already collected from Spark) and
return a ``plotly.graph_objects.Figure`` (or display it inline via
``fig.show()`` when ``show=True``).

Usage in the notebook:
    from viz import (
        plot_s1, plot_s2, plot_s3, plot_s4, plot_s5, plot_s6,
        plot_s7, plot_s8, plot_s9, plot_s10,
        plot_d1_styled, plot_d2, plot_d3, plot_d4, plot_d5,
        plot_summary_cards, plot_summary_hazard_and_dist,
        CLASS_COLORS, TIER_COLORS,
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Shared color palettes ─────────────────────────────────────────────────────
CLASS_COLORS: dict[str, str] = {
    "substation": "#e63946",
    "plant": "#f4a261",
    "dam": "#2a9d8f",
    "communication_tower": "#457b9d",
    "mobile_phone_tower": "#1d3557",
    "pipeline": "#a8dadc",
    "transformer": "#e9c46a",
    "generator": "#264653",
    "storage_tank": "#8ecae6",
    "water_tower": "#023047",
}

TIER_COLORS: dict[int, str] = {
    1: "#b7e4c7",
    2: "#74c69d",
    3: "#52b788",
    4: "#2d6a4f",
    5: "#1b4332",
}

_SENTINEL_COLOR = "#999999"

PLOTLY_TEMPLATE = "plotly_dark"


def _class_color_series(series: pd.Series) -> list[str]:
    return [CLASS_COLORS.get(c, _SENTINEL_COLOR) for c in series]


def _tier_color_series(series: pd.Series) -> list[str]:
    return [TIER_COLORS.get(int(t), _SENTINEL_COLOR) for t in series]


# ── Scenario 1: Multi-Hazard Overlap ─────────────────────────────────────────
def plot_s1(df: pd.DataFrame, show: bool = True) -> go.Figure:
    """Horizontal bar of top-20 vuln scores + grouped hazard exposure bar by class."""
    labels = [f"{r['class'][:14]} T{r['criticality_tier']}" for _, r in df.iterrows()]
    mean_score = df["vuln_score"].astype(float).mean()

    # Aggregate flood/wildfire counts by asset class for right panel
    agg = (
        df.groupby("class")
        .agg(
            flood_total=("flood_count", "sum"),
            wildfire_total=("wildfire_count", "sum"),
            asset_count=("asset_id", "count"),
        )
        .reset_index()
        .sort_values("flood_total", ascending=False)
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Vulnerability Score — Top 20 Multi-Hazard Assets",
            "Total Hazard Exposures by Class",
        ),
        horizontal_spacing=0.14,
    )

    # Chart A – horizontal bars (ranked by vuln score)
    fig.add_trace(
        go.Bar(
            y=labels,
            x=df["vuln_score"].astype(float),
            orientation="h",
            marker_color=_class_color_series(df["class"]),
            name="Vuln Score",
            hovertemplate="%{y}: %{x:.1f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_vline(
        x=mean_score,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean {mean_score:.0f}",
        row=1,
        col=1,
    )

    # Chart B – grouped bar by class: flood vs wildfire totals
    fig.add_trace(
        go.Bar(
            x=agg["class"],
            y=agg["flood_total"],
            name="Flood Exposures",
            marker_color="#457b9d",
            hovertemplate="%{x} — Flood: %{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=agg["class"],
            y=agg["wildfire_total"],
            name="Wildfire Exposures",
            marker_color="#e63946",
            hovertemplate="%{x} — Wildfire: %{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    # Annotate asset count per class
    for _, r in agg.iterrows():
        fig.add_annotation(
            x=r["class"],
            y=max(r["flood_total"], r["wildfire_total"]) + 0.5,
            text=f"n={r['asset_count']}",
            showarrow=False,
            font=dict(size=9, color="white"),
            row=1,
            col=2,
        )

    fig.update_layout(
        title="Scenario 1 — Multi-Hazard Overlap: Top 20 Assets",
        template=PLOTLY_TEMPLATE,
        height=540,
        barmode="group",
        legend=dict(orientation="h", y=-0.14),
        yaxis=dict(autorange="reversed"),
    )
    fig.update_xaxes(title_text="Vulnerability Score", row=1, col=1)
    fig.update_xaxes(title_text="Asset Class", tickangle=35, row=1, col=2)
    fig.update_yaxes(title_text="Total Exposure Count", row=1, col=2)
    if show:
        fig.show()
        return None
    return fig


# ── Scenario 2: Social Media Sentiment Hotspots ───────────────────────────────
def plot_s2(df: pd.DataFrame, show: bool = True) -> go.Figure:
    """Stacked sentiment bar + ranked lollipop of avg sentiment colored by tier."""
    short_ids = [
        f"{r['class'][:12]} T{r['criticality_tier']}" for _, r in df.iterrows()
    ]
    # Sort right-panel by avg_sentiment ascending so most negative is at top
    df_sorted = df.copy()
    df_sorted["_label"] = short_ids
    df_sorted = df_sorted.sort_values("avg_sentiment", ascending=True).reset_index(
        drop=True
    )

    tier_palette = {
        1: "#b7e4c7",
        2: "#74c69d",
        3: "#52b788",
        4: "#2d6a4f",
        5: "#1b4332",
    }
    dot_colors = [
        tier_palette.get(int(t), "#999") for t in df_sorted["criticality_tier"]
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Sentiment Breakdown — Top 15 Assets",
            "Avg Sentiment Score (ranked, colored by tier)",
        ),
        horizontal_spacing=0.14,
    )

    for col_name, color, label in [
        ("negative_tweet_count", "#e63946", "Negative"),
        ("neutral_tweet_count", "#adb5bd", "Neutral"),
        ("positive_tweet_count", "#2a9d8f", "Positive"),
    ]:
        fig.add_trace(
            go.Bar(
                x=short_ids,
                y=df[col_name],
                name=label,
                marker_color=color,
                hovertemplate=f"{label}: %{{y}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Lollipop: horizontal lines from 0 to sentiment score, then dot
    sent_vals = df_sorted["avg_sentiment"].astype(float)
    labels_sorted = df_sorted["_label"].tolist()
    for i, (lbl, val, col) in enumerate(zip(labels_sorted, sent_vals, dot_colors)):
        fig.add_trace(
            go.Scatter(
                x=[0, val],
                y=[lbl, lbl],
                mode="lines",
                line=dict(color="#555555", width=1.5),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )
    fig.add_trace(
        go.Scatter(
            x=sent_vals,
            y=labels_sorted,
            mode="markers",
            marker=dict(
                size=10,
                color=dot_colors,
                line=dict(width=1, color="white"),
            ),
            text=[f"Tier {t}" for t in df_sorted["criticality_tier"]],
            customdata=df_sorted["vuln_score"].astype(float),
            hovertemplate="<b>%{y}</b><br>Sentiment: %{x:.3f}<br>Vuln: %{customdata:.1f}  %{text}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=2)
    fig.add_vline(
        x=-0.05,
        line_dash="dot",
        line_color="#e63946",
        annotation_text="Neg threshold",
        row=1,
        col=2,
    )

    fig.update_layout(
        barmode="stack",
        title="Scenario 2 — Social Media Sentiment Hotspots",
        template=PLOTLY_TEMPLATE,
        height=540,
        legend=dict(orientation="h", y=-0.14),
    )
    fig.update_xaxes(tickangle=40, row=1, col=1)
    fig.update_xaxes(title_text="Avg Sentiment Score", row=1, col=2)
    fig.update_yaxes(title_text="Tweet Count", row=1, col=1)
    if show:
        fig.show()
        return None
    return fig


# ── Scenario 3: Geofence Monitoring ──────────────────────────────────────────
def plot_s3(df: pd.DataFrame, show: bool = True) -> go.Figure:
    """Top-10 geofences by tweet volume + sentiment heatmap table."""
    # Limit to top 10 most active geofences
    df_top = df.head(10).copy()
    short_fence = [gid[:12] + "…" for gid in df_top["geofence_id"]]
    df_top["_label"] = short_fence

    sent_colors = [
        "#e63946" if float(s) < -0.05 else ("#2a9d8f" if float(s) > 0.05 else "#adb5bd")
        for s in df_top["avg_fence_sentiment"]
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Top {len(df_top)} Geofences by Tweet Activity",
            "Negative Tweet Share per Geofence<br>(bar = % negative of total)",
        ),
        horizontal_spacing=0.14,
    )

    # Chart A: grouped bar — total vs negative tweets, sorted by total
    fig.add_trace(
        go.Bar(
            y=short_fence[::-1],
            x=df_top["total_tweets"].values[::-1],
            orientation="h",
            name="Total Tweets",
            marker_color="#457b9d",
            hovertemplate="Total: %{x:,}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            y=short_fence[::-1],
            x=df_top["total_negative_tweets"].values[::-1],
            orientation="h",
            name="Negative Tweets",
            marker_color="#e63946",
            hovertemplate="Negative: %{x:,}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Chart B: % negative ratio as horizontal bar, colored by sentiment bucket
    neg_pct = (
        df_top["total_negative_tweets"].astype(float)
        / df_top["total_tweets"].astype(float).replace(0, float("nan"))
        * 100
    ).fillna(0)
    fig.add_trace(
        go.Bar(
            y=short_fence[::-1],
            x=neg_pct.values[::-1],
            orientation="h",
            marker_color=sent_colors[::-1],
            customdata=df_top["avg_fence_sentiment"].astype(float).values[::-1],
            hovertemplate="%{y}<br>Neg %: %{x:.1f}%<br>Sentiment: %{customdata:.3f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_vline(
        x=20,
        line_dash="dot",
        line_color="#e63946",
        annotation_text="20% threshold",
        row=1,
        col=2,
    )

    # Annotate asset count on right panel
    for i, (lbl, n) in enumerate(
        zip(short_fence[::-1], df_top["assets_in_fence"].values[::-1])
    ):
        fig.add_annotation(
            x=neg_pct.values[::-1][i] + 0.5,
            y=lbl,
            text=f"{n} assets",
            showarrow=False,
            font=dict(size=9, color="white"),
            xanchor="left",
            row=1,
            col=2,
        )

    fig.update_layout(
        barmode="overlay",
        title=f"Scenario 3 — Geofence Monitoring: Top {len(df_top)} of {len(df)} Active Geofences",
        template=PLOTLY_TEMPLATE,
        height=540,
        legend=dict(orientation="h", y=-0.12),
    )
    fig.update_xaxes(title_text="Tweet Count", row=1, col=1)
    fig.update_xaxes(title_text="% Negative Tweets", row=1, col=2)
    if show:
        fig.show()
        return None
    return fig


# ── Scenario 4: Combined Risk ─────────────────────────────────────────────────
def plot_s4(df: pd.DataFrame, show: bool = True) -> go.Figure:
    """Geofence donut + vuln vs negative-tweets scatter."""
    gf_counts = df["geofence_status"].value_counts()
    inside_mask = df["geofence_status"].str.startswith("Inside")

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "pie"}, {"type": "scatter"}]],
        subplot_titles=(
            "Geofence Coverage of Highest-Concern Assets",
            "Vulnerability vs Negative Tweets<br>(size = criticality tier)",
        ),
        horizontal_spacing=0.14,
    )

    fig.add_trace(
        go.Pie(
            labels=gf_counts.index.tolist(),
            values=gf_counts.values.tolist(),
            hole=0.55,
            marker_colors=["#e63946", "#adb5bd"],
            textinfo="label+percent",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["negative_tweet_count"],
            y=df["vuln_score"].astype(float),
            mode="markers",
            marker=dict(
                size=df["criticality_tier"] * 6,
                color=["#e63946" if i else "#adb5bd" for i in inside_mask],
                line=dict(width=0.5, color="black"),
            ),
            text=df["class"],
            hovertemplate="<b>%{text}</b><br>Neg tweets: %{x}  Vuln: %{y}<extra></extra>",
            name="Assets",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Scenario 4 — Combined Risk: Multi-Hazard AND Social Alert",
        template=PLOTLY_TEMPLATE,
        height=520,
        showlegend=True,
    )
    fig.update_xaxes(title_text="Negative Tweet Count", row=1, col=2)
    fig.update_yaxes(title_text="Vulnerability Score", row=1, col=2)
    if show:
        fig.show()
        return None
    return fig


# ── Scenario 5: Tier-5 Risk Profile ──────────────────────────────────────────
def plot_s5(df: pd.DataFrame, show: bool = True) -> go.Figure:
    """3-panel: asset count, grouped hazard exposure, avg vuln + social alerts."""
    colors = _class_color_series(df["class"])

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Total Tier-5 Assets by Class",
            "Hazard Exposure by Class",
            "Avg Vuln Score + Social Alerts",
        ),
        specs=[[{}, {}, {"secondary_y": True}]],
        horizontal_spacing=0.10,
    )

    fig.add_trace(
        go.Bar(
            x=df["class"],
            y=df["asset_count"],
            marker_color=colors,
            name="Asset Count",
            hovertemplate="%{x}: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    for col_name, color, label in [
        ("flood_exposed", "#457b9d", "Flood"),
        ("wildfire_exposed", "#e63946", "Wildfire"),
        ("multi_hazard", "#f4a261", "Multi-Hazard"),
    ]:
        fig.add_trace(
            go.Bar(
                x=df["class"],
                y=df[col_name],
                name=label,
                marker_color=color,
                hovertemplate=f"{label}: %{{y}}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    fig.add_trace(
        go.Bar(
            x=df["class"],
            y=df["avg_vuln_score"].astype(float),
            marker_color=colors,
            opacity=0.7,
            name="Avg Vuln Score",
            hovertemplate="Avg Vuln: %{y}<extra></extra>",
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=df["class"],
            y=df["social_alert"],
            mode="lines+markers",
            marker=dict(symbol="diamond", size=9, color="purple"),
            line=dict(dash="dash", color="purple"),
            name="Social Alerts",
            hovertemplate="Social Alerts: %{y}<extra></extra>",
            yaxis="y4",
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        barmode="group",
        title="Scenario 5 — Tier-5 Asset Risk Profile by Class",
        template=PLOTLY_TEMPLATE,
        height=520,
        legend=dict(orientation="h", y=-0.18),
    )
    for col_idx in [1, 2, 3]:
        fig.update_xaxes(tickangle=40, row=1, col=col_idx)
    if show:
        fig.show()
        return None
    return fig


# ── Scenario 6: Seasonal Vulnerability Timeline ───────────────────────────────
def plot_s6(
    df_top: pd.DataFrame, df_summary: pd.DataFrame, show: bool = True
) -> go.Figure:
    """Gantt-style window comparison + scatter of Mar-Sep assets."""
    window_config = {
        "Mar-Jun": {"start": 3, "duration": 4, "color": "#457b9d"},
        "Jun-Sep": {"start": 6, "duration": 4, "color": "#e63946"},
        "Mar-Sep": {"start": 3, "duration": 7, "color": "#f4a261"},
    }
    month_names = [
        "",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Hazard Exposure Windows (Ukraine)",
            "Mar-Sep Assets: Vuln Score vs Social Signal<br>(size = criticality tier)",
        ),
        horizontal_spacing=0.12,
    )

    for wnd, cfg in window_config.items():
        row = df_summary[df_summary["peak_vulnerability_window"] == wnd]
        count = int(row["asset_count"].iloc[0]) if len(row) else 0
        # Draw as a horizontal bar from start to start+duration
        fig.add_trace(
            go.Bar(
                x=[cfg["duration"]],
                y=[wnd],
                base=[cfg["start"] - 1],
                orientation="h",
                name=f"{wnd} — {count:,} assets",
                marker_color=cfg["color"],
                opacity=0.85,
                hovertemplate=f"{wnd}: {count:,} assets, {cfg['duration']} months<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.update_xaxes(
        tickvals=list(range(12)),
        ticktext=month_names[1:],
        title_text="Month",
        row=1,
        col=1,
    )

    colors_s6 = _class_color_series(df_top["class"])
    fig.add_trace(
        go.Scatter(
            x=df_top["negative_tweet_count"],
            y=df_top["vuln_score"].astype(float),
            mode="markers",
            marker=dict(
                size=df_top["criticality_tier"] * 6,
                color=colors_s6,
                line=dict(width=0.5, color="black"),
            ),
            text=df_top["class"],
            hovertemplate="<b>%{text}</b><br>Neg tweets: %{x}  Vuln: %{y}<extra></extra>",
            name="Mar-Sep Assets",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        barmode="overlay",
        title="Scenario 6 — Seasonal Vulnerability Timeline",
        template=PLOTLY_TEMPLATE,
        height=520,
        legend=dict(orientation="h", y=-0.18),
    )
    fig.update_xaxes(title_text="Negative Tweet Count", row=1, col=2)
    fig.update_yaxes(title_text="Vulnerability Score", row=1, col=2)
    if show:
        fig.show()
        return None
    return fig


# ── Scenario 7: Settlement Impact ────────────────────────────────────────────
def plot_s7(df: pd.DataFrame, show: bool = True) -> go.Figure:
    """Stacked risk-component bar (vuln + facility + social) + ranked lollipop."""
    labels = [f"{r['class'][:12]} T{r['criticality_tier']}" for _, r in df.iterrows()]
    df = df.copy()
    df["_label"] = labels

    # Decompose composite risk into its three additive components
    df["_vuln_part"] = df["vuln_score"].astype(float)
    df["_facility_part"] = df["facility_count"].astype(float) * 5.0
    df["_social_part"] = df["negative_tweet_count"].astype(float) * 0.1

    # Sort descending by composite_risk_score for both charts
    df = df.sort_values("composite_risk_score", ascending=False).reset_index(drop=True)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Composite Risk Components — Top 10 Settlement-Impact Assets",
            "Facilities vs Social Signal<br>(dot size = vuln score, colored by class)",
        ),
        horizontal_spacing=0.14,
    )

    # Chart A: stacked horizontal bar showing 3 components
    for part_col, color, label in [
        ("_social_part", "#f4a261", "Social Signal (neg×0.1)"),
        ("_facility_part", "#457b9d", "Facility Proximity (×5)"),
        ("_vuln_part", "#e63946", "Vulnerability Score"),
    ]:
        fig.add_trace(
            go.Bar(
                y=labels,
                x=df[part_col],
                orientation="h",
                name=label,
                marker_color=color,
                hovertemplate=f"{label}: %{{x:.1f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Chart B: lollipop — ranked by composite_risk_score, sized by vuln, colored by class
    colors_s7 = _class_color_series(df["class"])
    for i, (lbl, fac, twt, col) in enumerate(
        zip(labels, df["facility_count"], df["negative_tweet_count"], colors_s7)
    ):
        fig.add_trace(
            go.Scatter(
                x=[0, fac],
                y=[lbl, lbl],
                mode="lines",
                line=dict(color="#555555", width=1.5),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )
    fig.add_trace(
        go.Scatter(
            x=df["facility_count"],
            y=labels,
            mode="markers",
            marker=dict(
                size=df["vuln_score"].astype(float) / 6 + 8,
                color=colors_s7,
                line=dict(width=1, color="white"),
            ),
            customdata=df[["negative_tweet_count", "composite_risk_score"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>Facilities: %{x}"
                "<br>Neg Tweets: %{customdata[0]}"
                "<br>Composite Risk: %{customdata[1]:.1f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        barmode="stack",
        title="Scenario 7 — Settlement Impact: Cascading Risk to Populated Areas",
        template=PLOTLY_TEMPLATE,
        height=540,
        legend=dict(orientation="h", y=-0.14),
        yaxis=dict(autorange="reversed"),
    )
    fig.update_xaxes(title_text="Risk Score (stacked components)", row=1, col=1)
    fig.update_xaxes(
        title_text="Nearby Facility Count (hospitals/schools)", row=1, col=2
    )
    if show:
        fig.show()
        return None
    return fig


# ── Scenario 8: Communication Infrastructure ─────────────────────────────────
def plot_s8(
    df_summary: pd.DataFrame, df_top: pd.DataFrame, show: bool = True
) -> go.Figure:
    """Grouped hazard exposure bar + ranked lollipop of tweet volume by tower."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Hazard Exposure by Tower Type",
            "Top 20 Towers: Tweet Volume (ranked)<br>(dot color = class, size = wildfire count)",
        ),
        horizontal_spacing=0.14,
    )

    for col_name, color, label in [
        ("in_wildfire_zone", "#e63946", "Wildfire Zone"),
        ("in_flood_zone", "#457b9d", "Flood Zone"),
        ("multi_hazard", "#f4a261", "Multi-Hazard"),
    ]:
        fig.add_trace(
            go.Bar(
                x=df_summary["class"],
                y=df_summary[col_name],
                name=label,
                marker_color=color,
                hovertemplate=f"{label}: %{{y}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Annotate total towers on left panel
    for _, row in df_summary.iterrows():
        fig.add_annotation(
            x=row["class"],
            y=max(row["in_wildfire_zone"], row["in_flood_zone"], row["multi_hazard"])
            + 5,
            text=f"n={row['total_towers']:,}",
            showarrow=False,
            font=dict(size=10),
            row=1,
            col=1,
        )

    # Right panel: ranked lollipop — sort by tweet_count descending
    df_rank = (
        df_top.copy().sort_values("tweet_count", ascending=True).reset_index(drop=True)
    )
    rank_labels = [
        f"{r['class'][:10]} T{r['criticality_tier']}" for _, r in df_rank.iterrows()
    ]
    colors_s8 = _class_color_series(df_rank["class"])

    for lbl in rank_labels:
        fig.add_trace(
            go.Scatter(
                x=[
                    0,
                    df_rank.loc[df_rank.index[rank_labels.index(lbl)], "tweet_count"],
                ],
                y=[lbl, lbl],
                mode="lines",
                line=dict(color="#555555", width=1.2),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )
    fig.add_trace(
        go.Scatter(
            x=df_rank["tweet_count"],
            y=rank_labels,
            mode="markers",
            marker=dict(
                size=df_rank["wildfire_count"].clip(lower=0) / 2 + 8,
                color=colors_s8,
                line=dict(width=1, color="white"),
            ),
            customdata=df_rank[
                ["negative_tweet_count", "wildfire_count", "vuln_score"]
            ].values,
            hovertemplate=(
                "<b>%{y}</b><br>Tweets: %{x:,}"
                "<br>Neg Tweets: %{customdata[0]:,}"
                "<br>Wildfire Count: %{customdata[1]}"
                "<br>Vuln Score: %{customdata[2]:.1f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_vline(
        x=float(df_rank["tweet_count"].mean()),
        line_dash="dash",
        line_color="gray",
        annotation_text="Mean",
        row=1,
        col=2,
    )

    fig.update_layout(
        barmode="group",
        title="Scenario 8 — Communication Infrastructure Resilience",
        template=PLOTLY_TEMPLATE,
        height=560,
        legend=dict(orientation="h", y=-0.12),
    )
    fig.update_xaxes(title_text="Tower Count", row=1, col=1)
    fig.update_xaxes(title_text="Total Tweet Count", row=1, col=2)
    if show:
        fig.show()
        return None
    return fig


# ── Scenario 9: Dam Safety ────────────────────────────────────────────────────
def plot_s9(df: pd.DataFrame, show: bool = True) -> go.Figure:
    """Compound risk horizontal bar + jittered flood vs facility scatter with top-5 labels."""
    tier_colors = _tier_color_series(df["criticality_tier"])
    labels = [
        f"T{r['criticality_tier']} | {r['asset_id'][:10]}…" for _, r in df.iterrows()
    ]
    social_colors = [
        "#e63946" if bool(s) else "#adb5bd" for s in df["social_alert_flag"]
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Top 20 Dams by Compound Risk Score",
            "Flood Exposure vs Population Proximity<br>(jittered; red = social alert; size = risk)",
        ),
        horizontal_spacing=0.14,
    )

    fig.add_trace(
        go.Bar(
            y=labels[::-1],
            x=df["compound_risk_score"].astype(float)[::-1],
            orientation="h",
            marker_color=tier_colors[::-1],
            name="Compound Risk",
            hovertemplate="%{y}: %{x:.1f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    mean_risk = float(df["compound_risk_score"].astype(float).mean())
    fig.add_vline(
        x=mean_risk,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Mean {mean_risk:.0f}",
        row=1,
        col=1,
    )

    # Jitter scatter — add small random offsets so stacked points separate
    rng = np.random.default_rng(42)
    jitter_x = rng.uniform(-0.35, 0.35, size=len(df))
    jitter_y = rng.uniform(-0.25, 0.25, size=len(df))

    x_vals = df["flood_count"].astype(float) + jitter_x
    y_vals = df["facility_count"].astype(float) + jitter_y

    # Clamp bubble size so the range is readable
    risk_vals = df["compound_risk_score"].astype(float)
    sizes = (
        (risk_vals - risk_vals.min()) / (risk_vals.max() - risk_vals.min() + 1e-9)
    ) * 20 + 8

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker=dict(
                size=sizes,
                color=social_colors,
                opacity=0.75,
                line=dict(width=0.8, color="white"),
            ),
            text=labels,
            customdata=df["compound_risk_score"].astype(float),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Flood: %{x:.1f}  Facilities: %{y:.1f}<br>"
                "Compound Risk: %{customdata:.1f}<extra></extra>"
            ),
            name="Dams",
        ),
        row=1,
        col=2,
    )

    # Label top 5 highest-risk dams
    top5_idx = df["compound_risk_score"].astype(float).nlargest(5).index
    for i in top5_idx:
        fig.add_annotation(
            x=x_vals.iloc[i],
            y=y_vals.iloc[i],
            text=labels[i],
            showarrow=True,
            arrowhead=2,
            arrowsize=0.8,
            font=dict(size=8, color="white"),
            arrowcolor="white",
            ax=25,
            ay=-18,
            row=1,
            col=2,
        )

    fig.update_layout(
        title="Scenario 9 — Dam Safety: Flood + Settlement + Social Compound Risk",
        template=PLOTLY_TEMPLATE,
        height=540,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Compound Risk Score", row=1, col=1)
    fig.update_xaxes(title_text="Flood Proximity Count (jittered)", row=1, col=2)
    fig.update_yaxes(title_text="Nearby Facility Count (jittered)", row=1, col=2)
    if show:
        fig.show()
        return None
    return fig


# ── Scenario 10: Cross-Demo Comparison ───────────────────────────────────────
def plot_s10(
    df_dist: pd.DataFrame, df_civ_profile: pd.DataFrame, show: bool = True
) -> go.Figure:
    """Score distribution histogram + risk flag percentage bars."""
    total = int(df_civ_profile["analytic_units"].iloc[0])
    avg = float(df_civ_profile["score_avg"].iloc[0])
    mh = int(df_civ_profile["multi_hazard_count"].iloc[0])
    sa = int(df_civ_profile["social_alert_count"].iloc[0])

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Vulnerability Score Distribution (Ukraine CIV)",
            f"Key Risk Flags  (Total: {total:,} CIV assets)",
        ),
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Bar(
            x=df_dist["score_bucket"].astype(float),
            y=df_dist["asset_count"],
            width=40,
            marker_color="#457b9d",
            name="Asset Count",
            hovertemplate="Bucket %{x}: %{y} assets<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_vline(
        x=avg,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean = {avg:.0f}",
        row=1,
        col=1,
    )

    pcts = [mh / total * 100, sa / total * 100]
    vals = [mh, sa]
    flags = ["Multi-Hazard Assets", "Social Alert Assets"]
    fig.add_trace(
        go.Bar(
            y=flags,
            x=pcts,
            orientation="h",
            marker_color=["#f4a261", "#e63946"],
            text=[f"{v:,} ({p:.1f}%)" for v, p in zip(vals, pcts)],
            textposition="outside",
            name="Risk Flags",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Scenario 10 — Cross-Demo Profile: Ukraine CIV Data Overview",
        template=PLOTLY_TEMPLATE,
        height=520,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Vulnerability Score Bucket (width=50)", row=1, col=1)
    fig.update_yaxes(title_text="Asset Count", row=1, col=1)
    fig.update_xaxes(title_text="% of Total Assets", row=1, col=2)
    if show:
        fig.show()
        return None
    return fig


# ── Scenario D1: Priority Targeting (Plotly styled table) ────────────────────
def plot_d1_styled(df: pd.DataFrame, show: bool = True) -> go.Figure:
    """Interactive Plotly table styled as analyst briefing."""

    def _flag(val):
        if val is True or val == 1 or str(val).lower() == "true":
            return "YES"
        return "NO"

    display_df = df.copy()
    display_df["multi_hazard_flag"] = display_df["multi_hazard_flag"].apply(_flag)
    display_df["social_alert_flag"] = display_df["social_alert_flag"].apply(_flag)

    header_labels = [
        "Asset ID",
        "Class",
        "Tier",
        "Vuln Score",
        "Multi-Hazard",
        "Social Alert",
        "Flood Cnt",
        "Wildfire Cnt",
        "Neg Tweets",
    ]

    cell_values = [
        display_df["asset_id"].tolist(),
        display_df["class"].tolist(),
        display_df["criticality_tier"].tolist(),
        [f"{v:.2f}" for v in display_df["vuln_score"].astype(float)],
        display_df["multi_hazard_flag"].tolist(),
        display_df["social_alert_flag"].tolist(),
        display_df["flood_count"].tolist(),
        display_df["wildfire_count"].tolist(),
        display_df["negative_tweet_count"].tolist(),
    ]

    row_fill = []
    for i in range(len(display_df)):
        has_mh = "YES" in display_df["multi_hazard_flag"].iloc[i]
        has_sa = "YES" in display_df["social_alert_flag"].iloc[i]
        if has_mh and has_sa:
            row_fill.append("#4a0e0e")
        elif has_mh or has_sa:
            row_fill.append("#2d2d00")
        else:
            row_fill.append("#1e1e2e")

    fig = go.Figure(
        go.Table(
            header=dict(
                values=[f"<b>{h}</b>" for h in header_labels],
                fill_color="#264653",
                font=dict(color="white", size=12),
                align="center",
            ),
            cells=dict(
                values=cell_values,
                fill_color=[row_fill] * len(header_labels),
                font=dict(color="white", size=11),
                align=[
                    "left",
                    "left",
                    "center",
                    "right",
                    "center",
                    "center",
                    "right",
                    "right",
                    "right",
                ],
            ),
        )
    )
    fig.update_layout(
        title="Analyst Priority Targeting List — Top 10 Most Vulnerable CIV Assets",
        template=PLOTLY_TEMPLATE,
        height=420,
    )
    if show:
        fig.show()
        return None
    return fig


# ── Scenario D2: Compound Cluster Analysis ────────────────────────────────────
def plot_d2(
    df: pd.DataFrame, avg_flood: float, avg_wildfire: float, show: bool = True
) -> go.Figure:
    """Asset count bar + quadrant scatter of flood vs wildfire."""
    colors = _class_color_series(df["class"])

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Compound Exposure Count<br>(flood > avg AND wildfire > avg)",
            "Flood vs Wildfire Exposure by Class<br>(bubble = compound asset count ÷ 10)",
        ),
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Bar(
            x=df["class"],
            y=df["compound_hazard_assets"],
            marker_color=colors,
            name="Compound Hazard Assets",
            hovertemplate="%{x}: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["avg_flood_count"].astype(float),
            y=df["avg_wildfire_count"].astype(float),
            mode="markers+text",
            text=df["class"],
            textposition="top right",
            textfont=dict(size=8),
            marker=dict(
                size=df["compound_hazard_assets"] / 10 + 6,
                color=colors,
                line=dict(width=0.5, color="black"),
            ),
            hovertemplate="%{text}<br>Avg Flood: %{x:.1f}  Avg Wildfire: %{y:.1f}<extra></extra>",
            name="Classes",
        ),
        row=1,
        col=2,
    )
    fig.add_vline(
        x=avg_flood,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Avg flood={avg_flood:.1f}",
        row=1,
        col=2,
    )
    fig.add_hline(
        y=avg_wildfire,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Avg wildfire={avg_wildfire:.1f}",
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Scenario D2 — Compound Multi-Hazard Cluster Analysis",
        template=PLOTLY_TEMPLATE,
        height=520,
        showlegend=False,
    )
    fig.update_xaxes(tickangle=40, row=1, col=1)
    fig.update_xaxes(title_text="Avg Flood Count (above-avg assets)", row=1, col=2)
    fig.update_yaxes(title_text="Avg Wildfire Count (above-avg assets)", row=1, col=2)
    if show:
        fig.show()
        return None
    return fig


# ── Scenario D3: Sentiment Threat Surface ────────────────────────────────────
def plot_d3(df: pd.DataFrame, show: bool = True) -> go.Figure:
    """Negative-ratio histogram + tier 4/5 stacked bar by class."""
    df_agg = (
        df.groupby(["class", "criticality_tier"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    tier4 = df_agg.get(4, pd.Series([0] * len(df_agg))).tolist()
    tier5 = df_agg.get(5, pd.Series([0] * len(df_agg))).tolist()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Distribution of Negative Tweet Ratios<br>(Tier 4-5, ratio > 50%)",
            "Threat Surface by Asset Class<br>(Tier 4+5, >50% negative sentiment)",
        ),
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Histogram(
            x=df["negative_ratio"].astype(float),
            nbinsx=15,
            marker_color="#e63946",
            name="Ratio Distribution",
            hovertemplate="Ratio %{x:.2f}: %{y} assets<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_vline(
        x=0.5,
        line_dash="dash",
        line_color="white",
        annotation_text="50% threshold",
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=df_agg["class"],
            y=tier4,
            name="Tier 4",
            marker_color="#2d6a4f",
            hovertemplate="Tier 4: %{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=df_agg["class"],
            y=tier5,
            name="Tier 5",
            marker_color="#1b4332",
            hovertemplate="Tier 5: %{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        barmode="stack",
        title="Scenario D3 — Social Sentiment Threat Surface",
        template=PLOTLY_TEMPLATE,
        height=520,
        legend=dict(orientation="h", y=-0.15),
    )
    fig.update_xaxes(title_text="Negative Tweet Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Asset Count", row=1, col=1)
    fig.update_xaxes(tickangle=40, row=1, col=2)
    fig.update_yaxes(title_text="Asset Count (>50% negative ratio)", row=1, col=2)
    if show:
        fig.show()
        return None
    return fig


# ── Scenario D4: Seasonal Exposure Planning ──────────────────────────────────
def plot_d4(by_class: pd.DataFrame, show: bool = True) -> go.Figure:
    """Stacked social-alert bar + avg flood vs wildfire scatter by class."""
    colors = _class_color_series(by_class["class"])

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Tier-5 Assets in Mar-Sep Window<br>(Social Alert Status)",
            "Avg Hazard Exposure by Class<br>(bubble = asset count ÷ 5)",
        ),
        horizontal_spacing=0.12,
    )

    no_alert = (by_class["count"] - by_class["social_alert"]).tolist()
    fig.add_trace(
        go.Bar(
            x=by_class["class"],
            y=by_class["social_alert"].tolist(),
            name="Social Alert ",
            marker_color="#e63946",
            hovertemplate="Alert: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=by_class["class"],
            y=no_alert,
            name="No Alert",
            marker_color="#adb5bd",
            hovertemplate="No Alert: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=by_class["avg_flood"],
            y=by_class["avg_wildfire"],
            mode="markers+text",
            text=by_class["class"],
            textposition="top right",
            textfont=dict(size=8),
            marker=dict(
                size=by_class["count"] / 5 + 6,
                color=colors,
                line=dict(width=0.5, color="black"),
            ),
            hovertemplate="%{text}<br>Avg Flood: %{x:.1f}  Avg Wildfire: %{y:.1f}<extra></extra>",
            name="Classes",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        barmode="stack",
        title="Scenario D4 — Seasonal Exposure Window Planning",
        template=PLOTLY_TEMPLATE,
        height=520,
        legend=dict(orientation="h", y=-0.15),
    )
    fig.update_xaxes(tickangle=40, row=1, col=1)
    fig.update_xaxes(title_text="Avg Flood Count", row=1, col=2)
    fig.update_yaxes(title_text="Asset Count", row=1, col=1)
    fig.update_yaxes(title_text="Avg Wildfire Count", row=1, col=2)
    if show:
        fig.show()
        return None
    return fig


# ── Scenario D5: Geofence Hot Spot ───────────────────────────────────────────
def plot_d5(
    df: pd.DataFrame, df_detail: pd.DataFrame, top_fence_id: str, show: bool = True
) -> go.Figure:
    """Risk score ranking bar + top-geofence class breakdown."""
    short_ids = [gid[:14] + "…" for gid in df["geofence_id"]]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Geofence Composite Risk Score Ranking (Top 10)",
            f"Top Geofence: Asset Class Breakdown<br>({top_fence_id[:22]}…)",
        ),
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Bar(
            y=short_ids[::-1],
            x=df["risk_score"].astype(int)[::-1],
            orientation="h",
            marker=dict(
                color=df["risk_score"].astype(int)[::-1],
                colorscale="YlOrRd",
            ),
            name="Risk Score",
            hovertemplate="%{y}: %{x:,}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_vline(
        x=float(df["risk_score"].astype(float).mean()),
        line_dash="dash",
        line_color="gray",
        annotation_text="Mean",
        row=1,
        col=1,
    )

    detail_labels = [
        f"{r['class']} T{r['criticality_tier']}" for _, r in df_detail.iterrows()
    ]
    colors_detail = _class_color_series(df_detail["class"])
    fig.add_trace(
        go.Bar(
            y=detail_labels[::-1],
            x=df_detail["asset_count"].tolist()[::-1],
            orientation="h",
            marker_color=colors_detail[::-1],
            name="Asset Count",
            hovertemplate="%{y}: %{x}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_vline(
        x=float(df_detail["asset_count"].mean()),
        line_dash="dash",
        line_color="red",
        annotation_text="Mean",
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Scenario D5 — Geofence Hot Spot Intelligence Report",
        template=PLOTLY_TEMPLATE,
        height=520,
        showlegend=False,
    )
    fig.update_xaxes(
        title_text="Risk Score (high_tier_assets × fence_neg_tweets)", row=1, col=1
    )
    fig.update_xaxes(title_text="Asset Count", row=1, col=2)
    if show:
        fig.show()
        return None
    return fig


# ── Final Summary: KPI Cards ──────────────────────────────────────────────────
def plot_summary_cards(stats: "pd.Series", show: bool = True) -> go.Figure:
    """6-panel KPI card grid using indicator traces."""
    total = int(stats["total_assets"])
    metrics = [
        ("Total Assets", total, None, "#457b9d"),
        ("Asset Classes", int(stats["asset_classes"]), None, "#2a9d8f"),
        (
            "Multi-Hazard Assets",
            int(stats["multi_hazard_assets"]),
            f"{100 * int(stats['multi_hazard_assets']) / total:.1f}% of total",
            "#e63946",
        ),
        (
            "Social Alert Assets",
            int(stats["social_alert_assets"]),
            f"{100 * int(stats['social_alert_assets']) / total:.1f}% of total",
            "#f4a261",
        ),
        (
            "Score Range",
            f"{float(stats['min_score']):.0f} – {float(stats['max_score']):.0f}",
            None,
            "#264653",
        ),
        ("Avg Vuln Score", float(stats["avg_score"]), None, "#1b4332"),
    ]

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[m[0] for m in metrics],
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )
    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]

    for (r, c), (label, value, delta_text, color) in zip(positions, metrics):
        fig.add_trace(
            go.Indicator(
                mode="number" if isinstance(value, str) else "number+delta",
                value=value if not isinstance(value, str) else 0,
                number=dict(
                    font=dict(size=40, color="white"),
                    valueformat=",",
                    suffix="" if isinstance(value, str) else "",
                ),
                title=dict(
                    text=(value if isinstance(value, str) else "")
                    + (
                        f"<br><span style='font-size:13px;color:rgba(255,255,255,0.8)'>{delta_text}</span>"
                        if delta_text
                        else ""
                    ),
                    font=dict(size=22, color="white"),
                ),
                domain=dict(row=r - 1, column=c - 1),
            ),
            row=r,
            col=c,
        )

    # Use colored shapes as card backgrounds
    shapes = []
    x_positions = [0, 0.355, 0.71]
    y_positions = [0.55, 0.0]
    for idx, ((r, c), (_, _, _, color)) in enumerate(zip(positions, metrics)):
        xi = c - 1
        yi = r - 1
        shapes.append(
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=x_positions[xi],
                x1=x_positions[xi] + 0.28,
                y0=y_positions[yi],
                y1=y_positions[yi] + 0.44,
                fillcolor=color,
                opacity=0.9,
                line=dict(width=0),
                layer="below",
            )
        )

    fig.update_layout(
        title=" CIV Demo — Intelligence Summary Dashboard",
        template=PLOTLY_TEMPLATE,
        height=480,
        shapes=shapes,
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
    )
    if show:
        fig.show()
        return None
    return fig


# ── Final Summary: Hazard Donut + Score Distribution ─────────────────────────
def plot_summary_hazard_and_dist(
    stats: "pd.Series", df_dist: pd.DataFrame, show: bool = True
) -> go.Figure:
    """Donut of hazard exposure + score distribution histogram."""
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=(
            "Asset Hazard Exposure Breakdown",
            "Vulnerability Score Distribution",
        ),
        horizontal_spacing=0.14,
    )

    fig.add_trace(
        go.Pie(
            labels=["Flood Only", "Wildfire Only", "Both Hazards", "No Hazard"],
            values=[
                int(stats["flood_only"]),
                int(stats["wildfire_only"]),
                int(stats["both_hazards"]),
                int(stats["no_hazard"]),
            ],
            hole=0.55,
            marker_colors=["#457b9d", "#e63946", "#f4a261", "#adb5bd"],
            textinfo="label+percent",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=df_dist["bucket"].astype(float),
            y=df_dist["cnt"],
            width=40,
            marker_color="#457b9d",
            name="Asset Count",
            hovertemplate="Bucket %{x}: %{y} assets<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_vline(
        x=float(stats["avg_score"]),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Avg = {float(stats['avg_score']):.0f}",
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Hazard Exposure Profile & Score Distribution",
        template=PLOTLY_TEMPLATE,
        height=480,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Vulnerability Score (bucket width = 50)", row=1, col=2)
    fig.update_yaxes(title_text="Asset Count", row=1, col=2)
    if show:
        fig.show()
        return None
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Setup Pipeline Visualizations (CIV_Demo_Setup_Ukraine)
# ─────────────────────────────────────────────────────────────────────────────


# ── Setup Step 1: AOI Localities ──────────────────────────────────────────────
def plot_setup_step1_localities(
    country_pd: pd.DataFrame, show: bool = True
) -> go.Figure:
    """Bar chart of AOI localities grouped by country and subtype."""
    fig = px.bar(
        country_pd,
        x="country",
        y="count",
        color="subtype",
        barmode="group",
        title="AOI Localities by Country and Subtype",
        labels={
            "country": "Country Code",
            "count": "Division Count",
            "subtype": "Subtype",
        },
        color_discrete_sequence=px.colors.qualitative.Bold,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(
        title_font_size=18,
        xaxis_title="Country Code",
        yaxis_title="Number of Divisions",
        legend_title="Subtype",
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f1a",
        font_color="#e0e0e0",
    )
    if show:
        fig.show()
        return None
    return fig


# ── Setup Step 2: Critical Infrastructure Assets ──────────────────────────────
def plot_setup_step2_assets(class_pd: pd.DataFrame, show: bool = True) -> go.Figure:
    """Horizontal bar of critical infrastructure asset counts by class."""
    fig = px.bar(
        class_pd,
        x="count",
        y="class",
        orientation="h",
        title="Critical Infrastructure Assets by Class — Ukraine",
        labels={"count": "Asset Count", "class": "Infrastructure Class"},
        color="count",
        color_continuous_scale="Reds",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        title_font_size=18,
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f1a",
        font_color="#e0e0e0",
        coloraxis_showscale=False,
    )
    if show:
        fig.show()
        return None
    return fig


# ── Setup Step 3: Flood Hazard Proximity ──────────────────────────────────────
def plot_setup_step3_flood(
    flood_pd: pd.DataFrame,
    top_flood_pd: pd.DataFrame,
    flood_buffer_m: int,
    show: bool = True,
) -> go.Figure:
    """Two-figure display: flood count histogram + top-15 assets by flood proximity."""
    fig_hist = px.histogram(
        flood_pd[flood_pd["flood_water_count"] > 0],
        x="flood_water_count",
        nbins=40,
        title=f"Flood Hazard — Distribution of Water Feature Counts within {flood_buffer_m} m",
        labels={
            "flood_water_count": "Water Features within Buffer",
            "count": "Number of Assets",
        },
        color_discrete_sequence=["#1f77b4"],
        template=PLOTLY_TEMPLATE,
    )
    fig_hist.update_layout(
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f1a",
        font_color="#e0e0e0",
        title_font_size=16,
    )
    if show:
        fig_hist.show()

    fig_top = px.bar(
        top_flood_pd,
        x="flood_water_count",
        y="asset_id",
        orientation="h",
        color="flood_water_count",
        color_continuous_scale="Blues",
        title=f"Top 15 Assets by Flood Water Feature Proximity ({flood_buffer_m} m)",
        labels={"flood_water_count": "Water Feature Count", "asset_id": "Asset ID"},
        template=PLOTLY_TEMPLATE,
    )
    fig_top.update_layout(
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f1a",
        font_color="#e0e0e0",
        coloraxis_showscale=False,
        title_font_size=16,
    )
    if show:
        fig_top.show()
        return None
    return fig_top


# ── Setup Step 4: Wildfire Hazard Proximity ───────────────────────────────────
def plot_setup_step4_wildfire(
    wildfire_pd: pd.DataFrame,
    top_wf_pd: pd.DataFrame,
    wildfire_buffer_m: int,
    show: bool = True,
) -> go.Figure:
    """Two-figure display: wildfire count histogram + top-15 assets by wildfire proximity."""
    fig_hist = px.histogram(
        wildfire_pd[wildfire_pd["wildfire_landcover_count"] > 0],
        x="wildfire_landcover_count",
        nbins=40,
        title=f"Wildfire Hazard — Distribution of Fire-Prone Land Cover within {wildfire_buffer_m} m",
        labels={
            "wildfire_landcover_count": "Fire-Prone Land Cover Features",
            "count": "Number of Assets",
        },
        color_discrete_sequence=["#d62728"],
        template=PLOTLY_TEMPLATE,
    )
    fig_hist.update_layout(
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f1a",
        font_color="#e0e0e0",
        title_font_size=16,
    )
    if show:
        fig_hist.show()

    fig_top = px.bar(
        top_wf_pd,
        x="wildfire_landcover_count",
        y="asset_id",
        orientation="h",
        color="wildfire_landcover_count",
        color_continuous_scale="Oranges",
        title=f"Top 15 Assets by Wildfire Land Cover Proximity ({wildfire_buffer_m} m)",
        labels={
            "wildfire_landcover_count": "Fire-Prone Features",
            "asset_id": "Asset ID",
        },
        template=PLOTLY_TEMPLATE,
    )
    fig_top.update_layout(
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f1a",
        font_color="#e0e0e0",
        coloraxis_showscale=False,
        title_font_size=16,
    )
    if show:
        fig_top.show()
        return None
    return fig_top


# ── Setup Step 5: Settlement & Facility / Building Proximity ──────────────────
def plot_setup_step5_settlement(
    top_fac_pd: pd.DataFrame,
    top_bld_pd: pd.DataFrame,
    settlement_m: int,
    building_m: int,
    show: bool = True,
) -> go.Figure:
    """Side-by-side horizontal bars: top-15 by facility count and by building density."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Top 15 by Facility Count ({settlement_m} m)",
            f"Top 15 by Building Density ({building_m} m)",
        ),
        shared_yaxes=False,
    )
    fig.add_trace(
        go.Bar(
            x=top_fac_pd["facility_count"],
            y=top_fac_pd["asset_id"],
            orientation="h",
            name="Facilities",
            marker=dict(
                color=top_fac_pd["facility_count"],
                colorscale="Greens",
                showscale=False,
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=top_bld_pd["building_count"],
            y=top_bld_pd["asset_id"],
            orientation="h",
            name="Buildings",
            marker=dict(
                color=top_bld_pd["building_count"],
                colorscale="Purples",
                showscale=False,
            ),
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_layout(
        title_text="Settlement & Building Proximity — Top Assets",
        title_font_size=18,
        showlegend=False,
        template=PLOTLY_TEMPLATE,
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f1a",
        font_color="#e0e0e0",
        height=500,
    )
    if show:
        fig.show()
        return None
    return fig


# ── Setup Step 6: Asset Criticality Tiers ────────────────────────────────────
def plot_setup_step6_tiers(tier_class_pd: pd.DataFrame, show: bool = True) -> go.Figure:
    """Sunburst of tier x class breakdown.

    ``tier_class_pd`` must include a ``tier_label`` column (e.g. "Tier 1").
    """
    fig = px.sunburst(
        tier_class_pd,
        path=["tier_label", "class"],
        values="count",
        color="criticality_tier",
        color_continuous_scale="RdYlGn_r",
        title="Critical Infrastructure — Tier and Class Breakdown",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(
        title_font_size=18,
        paper_bgcolor="#0f0f1a",
        font_color="#e0e0e0",
        coloraxis_showscale=False,
        height=550,
    )
    if show:
        fig.show()
        return None
    return fig


# ── Setup Step 7: Seasonal Hazard Windows ────────────────────────────────────
_WINDOW_COLORS: dict[str, str] = {
    "Mar-Jun": "#1f77b4",
    "Jun-Sep": "#d62728",
    "Mar-Sep": "#ff7f0e",
    "None": "#7f7f7f",
}


def plot_setup_step7_windows(window_pd: pd.DataFrame, show: bool = True) -> go.Figure:
    """Bar chart of assets per peak vulnerability window, colored by season."""
    fig = px.bar(
        window_pd,
        x="peak_vulnerability_window",
        y="count",
        color="peak_vulnerability_window",
        color_discrete_map=_WINDOW_COLORS,
        title="Assets by Peak Vulnerability Window",
        labels={"peak_vulnerability_window": "Hazard Window", "count": "Asset Count"},
        template=PLOTLY_TEMPLATE,
        text="count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        title_font_size=18,
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f1a",
        font_color="#e0e0e0",
        yaxis_title="Number of Assets",
    )
    if show:
        fig.show()
        return None
    return fig


# ── Setup Step 8: Social Media Sentiment Proximity ────────────────────────────
def plot_setup_step8_sentiment_gauge(
    total_neg: int,
    total_pos: int,
    total_neu: int,
    mean_sent: float,
    tweet_proximity_m: int,
    show: bool = True,
) -> go.Figure:
    """Gauge indicator showing mean sentiment score across all assets."""
    fig = go.Figure()
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=round(mean_sent, 4),
            title={"text": "Mean Sentiment Score (All Assets)", "font": {"size": 16}},
            gauge={
                "axis": {"range": [-1, 1], "tickcolor": "#e0e0e0"},
                "bar": {"color": "#ff7f0e" if mean_sent < 0 else "#2ca02c"},
                "steps": [
                    {"range": [-1, -0.2], "color": "#d62728"},
                    {"range": [-0.2, 0.2], "color": "#7f7f7f"},
                    {"range": [0.2, 1], "color": "#2ca02c"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": 0,
                },
            },
            delta={
                "reference": 0,
                "increasing": {"color": "#2ca02c"},
                "decreasing": {"color": "#d62728"},
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="#0f0f1a",
        font_color="#e0e0e0",
        height=300,
        annotations=[
            dict(
                text=f"Neg: {total_neg:,}  |  Pos: {total_pos:,}  |  Neu: {total_neu:,}",
                x=0.5,
                y=-0.15,
                showarrow=False,
                xref="paper",
                yref="paper",
                font=dict(size=13, color="#aaaaaa"),
            )
        ],
    )
    if show:
        fig.show()
        return None
    return fig


def plot_setup_step8_sentiment_bar(
    top_social_pd: pd.DataFrame,
    tweet_proximity_m: int,
    show: bool = True,
) -> go.Figure:
    """Stacked sentiment bar for top-20 assets by tweet volume."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Negative",
            x=top_social_pd["asset_id"],
            y=top_social_pd["negative_tweet_count"],
            marker_color="#d62728",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Neutral",
            x=top_social_pd["asset_id"],
            y=top_social_pd["neutral_tweet_count"],
            marker_color="#7f7f7f",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Positive",
            x=top_social_pd["asset_id"],
            y=top_social_pd["positive_tweet_count"],
            marker_color="#2ca02c",
        )
    )
    fig.update_layout(
        barmode="stack",
        title=f"Top 20 Assets by Tweet Volume — Sentiment Breakdown ({tweet_proximity_m} m)",
        xaxis_title="Asset ID",
        yaxis_title="Tweet Count",
        title_font_size=16,
        template=PLOTLY_TEMPLATE,
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f1a",
        font_color="#e0e0e0",
        legend=dict(orientation="h", y=1.08),
        xaxis_tickangle=-45,
        height=450,
    )
    if show:
        fig.show()
        return None
    return fig


# ── Setup Step 9: Geofence Asset Overlay ─────────────────────────────────────
def plot_setup_step9_geofence(
    geofence_pd: pd.DataFrame, show: bool = True
) -> go.Figure:
    """Dual-axis bar+line: asset count and negative tweets per geofence."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            name="Assets in Geofence",
            x=geofence_pd["geofence_id"].astype(str),
            y=geofence_pd["asset_count"],
            marker_color="#1f77b4",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name="Negative Tweets",
            x=geofence_pd["geofence_id"].astype(str),
            y=geofence_pd["negative_tweets"],
            mode="lines+markers",
            line=dict(color="#d62728", width=2),
            marker=dict(size=8),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title_text="Geofence Coverage — Assets and Negative Tweet Activity",
        title_font_size=16,
        template=PLOTLY_TEMPLATE,
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f1a",
        font_color="#e0e0e0",
        legend=dict(orientation="h", y=1.08),
        height=400,
    )
    fig.update_yaxes(title_text="Asset Count", secondary_y=False)
    fig.update_yaxes(title_text="Negative Tweet Count", secondary_y=True)
    if show:
        fig.show()
        return None
    return fig


# ── Setup Step 10 (continued): Vulnerability Distribution & Bubble ─────────────
def plot_setup_step10_vuln_dist(vuln_pd: pd.DataFrame, show: bool = True) -> go.Figure:
    """Overlay histogram of vulnerability score distribution colored by asset class."""
    fig = px.histogram(
        vuln_pd,
        x="vulnerability_score",
        nbins=50,
        color="class",
        title="Vulnerability Score Distribution by Infrastructure Class",
        labels={"vulnerability_score": "Vulnerability Score", "count": "Asset Count"},
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.add_vline(
        x=vuln_pd["vulnerability_score"].quantile(0.9),
        line_dash="dash",
        line_color="white",
        annotation_text="P90",
        annotation_position="top right",
    )
    fig.update_layout(
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f1a",
        font_color="#e0e0e0",
        title_font_size=16,
        barmode="overlay",
    )
    fig.update_traces(opacity=0.75)
    if show:
        fig.show()
        return None
    return fig


def plot_setup_step10_vuln_bubble(
    vuln_pd: pd.DataFrame, show: bool = True
) -> go.Figure:
    """Bubble scatter of flood vs wildfire exposure, sized by vulnerability score.

    ``vuln_pd`` must include a ``flag`` column with values such as
    "Multi-Hazard + Alert", "Multi-Hazard", "Social Alert", or "Standard".
    """
    fig = px.scatter(
        vuln_pd,
        x="flood_count",
        y="wildfire_count",
        size="vulnerability_score",
        color="class",
        symbol="flag",
        hover_data=[
            "asset_id",
            "criticality_tier",
            "negative_tweet_count",
            "vulnerability_score",
        ],
        title="Flood vs Wildfire Exposure — Bubble Size = Vulnerability Score",
        labels={
            "flood_count": "Flood Feature Count",
            "wildfire_count": "Wildfire Land Cover Count",
        },
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=px.colors.qualitative.Bold,
        size_max=30,
    )
    fig.update_layout(
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f1a",
        font_color="#e0e0e0",
        title_font_size=16,
        height=550,
        legend=dict(orientation="v", x=1.02),
    )
    if show:
        fig.show()
        return None
    return fig


# ── Setup Pipeline Summary: KPI Indicator Dashboard ──────────────────────────
def plot_setup_pipeline_summary(
    steps: "list[tuple[str, int]]",
    output_prefix: str,
    table_suffix: str,
    geofence_table: str,
    tweet_table: str,
    w_flood: float,
    w_wildfire: float,
    w_settle: float,
    w_critical: float,
    w_social: float,
    show: bool = True,
) -> go.Figure:
    """Grid of go.Indicator number cards summarizing pipeline step counts."""
    cols = 5
    rows = -(-len(steps) // cols)  # ceiling division

    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "indicator"}] * cols for _ in range(rows)],
    )
    for idx, (label, value) in enumerate(steps):
        r = idx // cols + 1
        c = idx % cols + 1
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=value,
                title={"text": label, "font": {"size": 12, "color": "#aaaaaa"}},
                number={"font": {"size": 28, "color": "#ffffff"}, "valueformat": ","},
            ),
            row=r,
            col=c,
        )
    fig.update_layout(
        title=dict(
            text=(
                "<b>CIV Ukraine Pipeline — Complete</b><br>"
                f"<sup>Output: {output_prefix}.*{table_suffix} &nbsp;|&nbsp; "
                f"Social: {geofence_table}, {tweet_table} &nbsp;|&nbsp; "
                f"Weights: flood={w_flood} wildfire={w_wildfire} settle={w_settle} "
                f"critical={w_critical} social={w_social}</sup>"
            ),
            font=dict(size=18, color="#e0e0e0"),
            x=0.5,
        ),
        paper_bgcolor="#0f0f1a",
        plot_bgcolor="#0f0f1a",
        font_color="#e0e0e0",
        height=200 * rows + 120,
        margin=dict(l=20, r=20, t=110, b=20),
    )
    if show:
        fig.show()
        return None
    return fig


# ---------------------------------------------------------------------------
# Setup – Step 11 helpers
# ---------------------------------------------------------------------------


def plot_setup_step11_tables(tables_pd, output_prefix: str = "", show: bool = True):
    """Render a styled table of all persisted Iceberg tables in a given catalog prefix."""
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[f"<b>{c}</b>" for c in tables_pd.columns],
                    fill_color="#1f3a5f",
                    font=dict(color="white", size=13),
                    align="left",
                    height=32,
                ),
                cells=dict(
                    values=[tables_pd[c] for c in tables_pd.columns],
                    fill_color="#0f0f1a",
                    font=dict(color="#e0e0e0", size=12),
                    align="left",
                    height=28,
                ),
            )
        ]
    )
    title_text = (
        f"Persisted Iceberg Tables in <b>{output_prefix}</b>"
        if output_prefix
        else "Persisted Iceberg Tables"
    )
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=16, color="#e0e0e0"),
            x=0,
        ),
        paper_bgcolor="#0f0f1a",
        font_color="#e0e0e0",
        margin=dict(l=10, r=10, t=60, b=10),
        height=max(200, 50 + len(tables_pd) * 32),
    )
    if show:
        fig.show()
        return None
    return fig


def plot_setup_step11_top5(
    top5_pd, output_prefix: str = "", table_suffix: str = "", show: bool = True
):
    """Render a styled spot-check table of the top-5 most vulnerable assets."""
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[f"<b>{c}</b>" for c in top5_pd.columns],
                    fill_color="#3a1f5f",
                    font=dict(color="white", size=13),
                    align="left",
                    height=32,
                ),
                cells=dict(
                    values=[top5_pd[c] for c in top5_pd.columns],
                    fill_color="#0f0f1a",
                    font=dict(color="#e0e0e0", size=12),
                    align="left",
                    height=28,
                ),
            )
        ]
    )
    ref = (
        f"{output_prefix}.enhanced_vulnerability{table_suffix}"
        if output_prefix
        else "enhanced_vulnerability"
    )
    fig.update_layout(
        title=dict(
            text=f"Top 5 Most Vulnerable Assets (persisted: {ref})",
            font=dict(size=15, color="#e0e0e0"),
            x=0,
        ),
        paper_bgcolor="#0f0f1a",
        font_color="#e0e0e0",
        margin=dict(l=10, r=10, t=60, b=10),
        height=280,
    )
    if show:
        fig.show()
        return None
    return fig
