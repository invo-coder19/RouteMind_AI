"""
dashboard/app.py
================
LLM Cost Autopilot — Cost & Quality Dashboard (Phase 4.2 — UI/UX Refinement).

Run with:
    streamlit run dashboard/app.py

Sections (top to bottom)
-------------------------
1. Headline: total $ saved vs. baseline — single dominant number
2. Cost over time: actual vs. baseline dual line chart
3. Routing distribution: donut (tier) + horizontal bar (model)
4. Quality & escalation trend: dual-axis line chart
5. Recent escalations: sortable/scrollable st.dataframe

Theme is centralised in .streamlit/config.toml.
All remaining hex codes here are CSS custom properties — one place to change.

Data contract: no changes to db/queries.py or its return shapes.
"""
from __future__ import annotations

import pathlib
import sys

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from db.queries import (  # noqa: E402
    get_baseline_comparison,
    get_cost_over_time,
    get_quality_metrics,
    get_recent_escalations,
    get_routing_distribution,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LLM Cost Autopilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Design tokens — kept in one block so colours are easy to update
# ---------------------------------------------------------------------------
# These mirror .streamlit/config.toml semantically.
# Use these variables throughout; never paste hex inline below this block.

BG          = "#0b0f1a"   # page background
SURFACE     = "#131928"   # card / panel background
SURFACE2    = "#1c2333"   # slightly raised (chart bg, row hover)
BORDER      = "#1e2d40"   # subtle card border
PRIMARY     = "#4ade80"   # green — savings, pass, positive
PRIMARY_DIM = "#22c55e"   # darker green for fill / area
DANGER      = "#f87171"   # red — escalation, baseline, negative
ACCENT      = "#60a5fa"   # blue — spend, neutral metric
PURPLE      = "#a78bfa"   # purple — request count
TEXT        = "#f1f5f9"   # high-contrast body
MUTED       = "#94a3b8"   # labels, captions
GRID        = "#1a2236"   # chart gridlines

# Chart colour sequences — consistent shades of the palette, not rainbows
TIER_COLOURS = {
    "simple":   PRIMARY,
    "moderate": ACCENT,
    "complex":  PURPLE,
}
TIER_SEQ = [PRIMARY, ACCENT, PURPLE]

FONT = "Inter, system-ui, -apple-system, sans-serif"

# ---------------------------------------------------------------------------
# Global CSS — single <style> block, zero scattered markdown later
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <style>
    /* ── Google Fonts ─────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Reset / Base ─────────────────────────────────────────────── */
    html, body, [class*="css"], .stApp {{
        font-family: {FONT};
        background-color: {BG};
        color: {TEXT};
    }}

    /* Hide Streamlit chrome */
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding: 2rem 2.5rem 3rem; max-width: 1400px; }}

    /* ── Page header ──────────────────────────────────────────────── */
    .page-title {{
        font-size: 1.55rem;
        font-weight: 700;
        color: {TEXT};
        letter-spacing: -0.02em;
        line-height: 1.2;
        margin: 0 0 2px;
    }}
    .page-subtitle {{
        font-size: 0.875rem;
        color: {MUTED};
        margin: 0 0 1.5rem;
    }}

    /* ── Section labels ───────────────────────────────────────────── */
    .section-label {{
        font-size: 0.7rem;
        font-weight: 600;
        color: {MUTED};
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 0 0 0.35rem;
    }}
    .section-rule {{
        border: none;
        border-top: 1px solid {BORDER};
        margin: 0 0 1.2rem;
    }}

    /* ── Headline metric card ─────────────────────────────────────── */
    .headline-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 22px 26px 18px;
        position: relative;
        transition: box-shadow 0.2s ease;
    }}
    .headline-card:hover {{
        box-shadow: 0 0 0 1px {PRIMARY}33, 0 4px 24px rgba(0,0,0,0.4);
    }}
    .headline-card.primary {{
        border-color: {PRIMARY}55;
        background: linear-gradient(145deg, {SURFACE} 60%, {PRIMARY}08 100%);
    }}
    .card-super {{
        font-size: 0.68rem;
        font-weight: 600;
        color: {MUTED};
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 6px;
    }}
    .card-value {{
        font-size: 2.6rem;
        font-weight: 700;
        color: {PRIMARY};
        line-height: 1.05;
        letter-spacing: -0.03em;
    }}
    .card-value.neutral {{ color: {ACCENT}; }}
    .card-value.purple  {{ color: {PURPLE}; }}
    .card-value.negative {{ color: {DANGER}; }}
    .card-sub {{
        font-size: 0.82rem;
        color: {MUTED};
        margin-top: 5px;
    }}
    .card-trend {{
        position: absolute;
        top: 18px; right: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 20px;
        background: {PRIMARY}1a;
        color: {PRIMARY};
    }}
    .card-trend.down {{
        background: {DANGER}1a;
        color: {DANGER};
    }}

    /* ── Filter pill row ──────────────────────────────────────────── */
    div[data-testid="stSelectbox"] > div {{
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
    }}

    /* ── Dataframe ─────────────────────────────────────────────────  */
    .stDataFrame {{
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid {BORDER};
    }}

    /* ── Empty state ──────────────────────────────────────────────── */
    .empty-state {{
        text-align: center;
        padding: 48px 24px;
        color: {MUTED};
        font-size: 0.9rem;
        background: {SURFACE};
        border: 1px dashed {BORDER};
        border-radius: 12px;
    }}
    .empty-state .icon {{
        font-size: 2rem;
        margin-bottom: 8px;
    }}

    /* ── Footer ───────────────────────────────────────────────────── */
    .dash-footer {{
        text-align: center;
        color: {BORDER};
        font-size: 0.73rem;
        padding-top: 2rem;
        border-top: 1px solid {BORDER};
        margin-top: 2rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Shared Plotly layout helper
# ---------------------------------------------------------------------------

def _base_layout(**overrides) -> dict:
    """Return a consistent Plotly layout dict for all charts."""
    base = dict(
        plot_bgcolor=SURFACE2,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, family=FONT, size=12),
        margin=dict(l=4, r=4, t=36, b=4),
        legend=dict(
            orientation="h",
            x=0,
            y=1.12,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        xaxis=dict(
            gridcolor=GRID,
            linecolor=BORDER,
            tickcolor=BORDER,
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor=GRID,
            linecolor=BORDER,
            tickcolor=BORDER,
            showgrid=True,
            zeroline=False,
        ),
        hoverlabel=dict(
            bgcolor=SURFACE,
            bordercolor=BORDER,
            font=dict(color=TEXT, family=FONT, size=12),
        ),
        height=310,
    )
    base.update(overrides)
    return base


# ── Section header helper ──────────────────────────────────────────────────

def _section(label: str) -> None:
    st.markdown(
        f'<p class="section-label">{label}</p><hr class="section-rule">',
        unsafe_allow_html=True,
    )


def _empty(msg: str, icon: str = "📭") -> None:
    st.markdown(
        f'<div class="empty-state"><div class="icon">{icon}</div>{msg}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.markdown(
    '<p class="page-title">⚡ LLM Cost Autopilot</p>'
    '<p class="page-subtitle">Live routing performance &amp; cost savings — Postgres-backed, auto-refreshes every 2 min</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Filter row — compact segmented-style control
# ---------------------------------------------------------------------------

filter_col, _, _ = st.columns([1, 1, 4])
with filter_col:
    days_back: int = st.selectbox(
        "Time window",
        options=[7, 14, 30, 60, 90],
        index=2,
        format_func=lambda d: f"Last {d} days",
        label_visibility="collapsed",
    )

# ---------------------------------------------------------------------------
# Data loading — cached per window, 2-min TTL
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120, show_spinner="Fetching data…")
def load_all_data(days: int) -> dict:
    """Load all dashboard data for the given time window."""
    try:
        return {
            "baseline":    get_baseline_comparison(),
            "cost_time":   get_cost_over_time(days=days),
            "routing":     get_routing_distribution(),
            "quality":     get_quality_metrics(days=days),
            "escalations": get_recent_escalations(limit=25),
            "error":       None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


data = load_all_data(days_back)

if data.get("error"):
    st.error(
        f"**Database connection failed:** {data['error']}\n\n"
        "Check that Postgres is running and `DATABASE_URL` is set in `.env`, "
        "then run `python db/init_db.py`."
    )
    st.stop()

df_baseline    = data["baseline"]
df_cost        = data["cost_time"]
df_routing     = data["routing"]
df_quality     = data["quality"]
df_escalations = data["escalations"]

# ---------------------------------------------------------------------------
# 1. Headline metrics — largest element on the page
# ---------------------------------------------------------------------------

_section("COST SAVINGS OVERVIEW")

total_actual   = float(df_baseline["actual_cost"].sum())   if not df_baseline.empty else 0.0
total_baseline = float(df_baseline["baseline_cost"].sum()) if not df_baseline.empty else 0.0
total_savings  = total_baseline - total_actual
savings_pct    = (total_savings / total_baseline * 100)    if total_baseline > 0 else 0.0
total_requests = int(df_cost["request_count"].sum())       if not df_cost.empty else 0

# ── Trend indicator (half-period comparison) ──────────────────────────────
# TODO: get_baseline_comparison() returns all-time aggregates, not period-split.
# A per-period trend indicator requires a split query. Mark as TODO until
# get_baseline_comparison() supports a `days` param.
_savings_trend_html = ""  # placeholder — no fabricated trend

h1, h2, h3, h4 = st.columns(4, gap="small")

with h1:
    v_class = "card-value" if total_savings >= 0 else "card-value negative"
    st.markdown(
        f"""<div class="headline-card primary">
            <div class="card-super">Total Saved (All Time)</div>
            <div class="{v_class}">${total_savings:,.4f}</div>
            <div class="card-sub">vs. routing everything to premium tier</div>
            {_savings_trend_html}
        </div>""",
        unsafe_allow_html=True,
    )

with h2:
    v_class = "card-value" if savings_pct >= 0 else "card-value negative"
    st.markdown(
        f"""<div class="headline-card">
            <div class="card-super">Cost Reduction</div>
            <div class="{v_class}">{savings_pct:.1f}%</div>
            <div class="card-sub">vs. all-premium baseline</div>
        </div>""",
        unsafe_allow_html=True,
    )

with h3:
    st.markdown(
        f"""<div class="headline-card">
            <div class="card-super">Actual Spend</div>
            <div class="card-value neutral">${total_actual:,.4f}</div>
            <div class="card-sub">smart routing applied</div>
        </div>""",
        unsafe_allow_html=True,
    )

with h4:
    st.markdown(
        f"""<div class="headline-card">
            <div class="card-super">Requests — last {days_back}d</div>
            <div class="card-value purple">{total_requests:,}</div>
            <div class="card-sub">routed by classifier</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2. Cost over time
# ---------------------------------------------------------------------------

_section("COST OVER TIME")

if df_cost.empty:
    _empty("No request data for this time window.", "📉")
else:
    df_cm = df_cost.copy()
    if not df_baseline.empty:
        df_cm = df_cm.merge(df_baseline[["date", "baseline_cost"]], on="date", how="left")
        df_cm["baseline_cost"] = df_cm["baseline_cost"].fillna(0)

    fig_cost = go.Figure()

    # Area fill for actual — subtle green fill
    fig_cost.add_trace(go.Scatter(
        x=df_cm["date"],
        y=df_cm["total_cost"],
        name="Actual cost",
        mode="lines+markers",
        line=dict(color=PRIMARY, width=2),
        marker=dict(size=4, color=PRIMARY),
        fill="tozeroy",
        fillcolor=f"{PRIMARY}12",
        hovertemplate="<b>%{x|%b %d}</b><br>Actual: $%{y:.4f}<extra></extra>",
    ))

    if "baseline_cost" in df_cm.columns:
        fig_cost.add_trace(go.Scatter(
            x=df_cm["date"],
            y=df_cm["baseline_cost"],
            name="Baseline (all-premium)",
            mode="lines",
            line=dict(color=DANGER, width=1.5, dash="dash"),
            hovertemplate="<b>%{x|%b %d}</b><br>Baseline: $%{y:.4f}<extra></extra>",
        ))

    fig_cost.update_layout(
        **_base_layout(
            yaxis=dict(
                title="USD",
                gridcolor=GRID,
                tickformat="$.4f",
                tickfont=dict(size=11),
            ),
        )
    )
    st.plotly_chart(fig_cost, use_container_width=True)

# ---------------------------------------------------------------------------
# 3. Routing distribution
# ---------------------------------------------------------------------------

_section("ROUTING DISTRIBUTION")

if df_routing.empty:
    _empty("No routing data yet.", "🗂️")
else:
    c_donut, c_bar = st.columns(2, gap="medium")

    with c_donut:
        tier_agg = (
            df_routing.groupby("complexity_tier")["request_count"]
            .sum()
            .reset_index()
        )
        # Consistent tier ordering
        tier_order = ["simple", "moderate", "complex"]
        tier_agg["_order"] = tier_agg["complexity_tier"].map(
            {t: i for i, t in enumerate(tier_order)}
        )
        tier_agg = tier_agg.sort_values("_order").drop(columns="_order")

        fig_donut = go.Figure(go.Pie(
            labels=tier_agg["complexity_tier"],
            values=tier_agg["request_count"],
            hole=0.55,
            marker=dict(
                colors=[TIER_COLOURS.get(t, ACCENT) for t in tier_agg["complexity_tier"]],
                line=dict(color=SURFACE, width=2),
            ),
            textfont=dict(color=TEXT, size=12),
            hovertemplate="<b>%{label}</b><br>%{value:,} requests (%{percent})<extra></extra>",
        ))
        fig_donut.update_layout(
            **_base_layout(
                title=dict(text="By Complexity Tier", font=dict(size=13, color=MUTED), x=0.01),
                showlegend=True,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=300,
            )
        )
        # Donut centre annotation
        total_r = tier_agg["request_count"].sum()
        fig_donut.add_annotation(
            text=f"<b>{total_r:,}</b><br><span style='font-size:10px'>total</span>",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=15, color=TEXT, family=FONT),
            align="center",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with c_bar:
        df_sorted = df_routing.sort_values("request_count", ascending=True)
        fig_bar = go.Figure()

        for tier in tier_order:
            subset = df_sorted[df_sorted["complexity_tier"] == tier]
            if subset.empty:
                continue
            fig_bar.add_trace(go.Bar(
                y=subset["model_id_used"],
                x=subset["request_count"],
                name=tier.capitalize(),
                orientation="h",
                marker_color=TIER_COLOURS.get(tier, ACCENT),
                hovertemplate=f"<b>%{{y}}</b><br>{tier}: %{{x:,}} reqs<extra></extra>",
            ))

        fig_bar.update_layout(
            **_base_layout(
                title=dict(text="By Model", font=dict(size=13, color=MUTED), x=0.01),
                barmode="stack",
                xaxis=dict(title="Requests", gridcolor=GRID, tickformat=","),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11)),
                height=300,
            )
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------------------------
# 4. Quality & escalation trend
# ---------------------------------------------------------------------------

_section("QUALITY & ESCALATION TREND")

if df_quality.empty:
    _empty("Run requests through the verification loop to see quality data here.", "🎯")
else:
    fig_q = go.Figure()

    # Primary y — quality score (green)
    fig_q.add_trace(go.Scatter(
        x=df_quality["date"],
        y=df_quality["avg_quality"],
        name="Avg Quality Score",
        mode="lines+markers",
        line=dict(color=PRIMARY, width=2),
        marker=dict(size=4, color=PRIMARY),
        yaxis="y1",
        hovertemplate="<b>%{x|%b %d}</b><br>Quality: %{y:.3f}<extra></extra>",
    ))

    # Secondary y — escalation rate (red, dashed)
    fig_q.add_trace(go.Scatter(
        x=df_quality["date"],
        y=df_quality["escalation_rate"],
        name="Escalation Rate %",
        mode="lines+markers",
        line=dict(color=DANGER, width=1.5, dash="dot"),
        marker=dict(size=4, symbol="diamond", color=DANGER),
        yaxis="y2",
        hovertemplate="<b>%{x|%b %d}</b><br>Escalation: %{y:.1f}%<extra></extra>",
    ))

    fig_q.update_layout(
        **_base_layout(
            yaxis=dict(
                title="Quality Score",
                gridcolor=GRID,
                range=[0, 1.05],
                tickformat=".2f",
                tickfont=dict(size=11),
            ),
            yaxis2=dict(
                title="Escalation %",
                overlaying="y",
                side="right",
                gridcolor="rgba(0,0,0,0)",
                range=[0, 100],
                tickformat=".0f",
                ticksuffix="%",
                tickfont=dict(size=11),
            ),
        )
    )
    st.plotly_chart(fig_q, use_container_width=True)

# ---------------------------------------------------------------------------
# 5. Recent escalations — sortable, scrollable, column-configured
# ---------------------------------------------------------------------------

_section("RECENT ESCALATIONS — SYSTEM SELF-CORRECTIONS")

if df_escalations.empty:
    _empty(
        "No escalations on record. The router's cheap models are meeting quality standards "
        "— or the verification loop hasn't processed requests yet.",
        "✅",
    )
else:
    disp = df_escalations.copy()
    disp["timestamp"] = disp["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    # Keep quality_score numeric so ProgressColumn works
    disp = disp.rename(columns={
        "timestamp":          "Time (UTC)",
        "complexity_tier":    "Tier",
        "model_id_used":      "Cheap Model",
        "reference_model_id": "Reference Model",
        "quality_score":      "Quality Score",
        "judge_justification":"Judge Note",
        "prompt_preview":     "Prompt",
    })
    disp = disp.drop(columns=["request_id"], errors="ignore")

    st.dataframe(
        disp,
        use_container_width=True,
        hide_index=True,
        height=340,
        column_config={
            "Quality Score": st.column_config.ProgressColumn(
                "Quality Score",
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
                width="small",
            ),
            "Tier": st.column_config.TextColumn("Tier", width="small"),
            "Time (UTC)": st.column_config.TextColumn("Time (UTC)", width="medium"),
            "Prompt": st.column_config.TextColumn("Prompt", width="large"),
            "Judge Note": st.column_config.TextColumn("Judge Note", width="large"),
        },
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    f'<div class="dash-footer">LLM Cost Autopilot &nbsp;·&nbsp; '
    f'Phase 4.2 Dashboard &nbsp;·&nbsp; '
    f'Data cached for 2 minutes &nbsp;·&nbsp; '
    f'Postgres live</div>',
    unsafe_allow_html=True,
)
