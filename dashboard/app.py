"""
dashboard/app.py
================
LLM Cost Autopilot — Cost & Quality Dashboard (Phase 4).

Run with:
    streamlit run dashboard/app.py

What it shows (top to bottom)
------------------------------
1. Headline metric: total $ saved vs. baseline, and % savings
2. Cost over time: daily line chart
3. Routing distribution: pie + bar charts by tier and model
4. Quality & escalation trend: dual-axis line chart
5. Recent escalations table: spot-check real failures

Date-range filter at the top applies to all time-series charts.

All data is read live from Postgres via db/queries.py.
No mock or hardcoded data.
"""
from __future__ import annotations

import pathlib
import sys

import plotly.express as px
import plotly.graph_objects as go
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
    page_title="LLM Cost Autopilot — Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark theme
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main { background-color: #0e1117; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #232b3e 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 24px 28px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.8rem;
        font-weight: 700;
        color: #48bb78;
        line-height: 1.1;
    }
    .metric-value.negative { color: #fc8181; }
    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-sub {
        font-size: 1.1rem;
        color: #68d391;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        padding: 8px 0 4px 0;
        border-bottom: 2px solid #2d3748;
        margin-bottom: 16px;
    }

    /* Streamlit overrides */
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    div[data-testid="metric-container"] { background: #1a1f2e; border-radius: 10px; padding: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("## ⚡ LLM Cost Autopilot")
st.markdown(
    "<p style='color:#718096;margin-top:-8px;'>Live routing performance & cost savings dashboard</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Date-range filter (applies to all time-series charts)
# ---------------------------------------------------------------------------

col_filter1, col_filter2, _ = st.columns([1, 1, 4])
with col_filter1:
    days_back = st.selectbox(
        "Time window",
        options=[7, 14, 30, 60, 90],
        index=2,
        format_func=lambda d: f"Last {d} days",
    )

st.divider()

# ---------------------------------------------------------------------------
# Data loading — cached per time window
# ---------------------------------------------------------------------------


@st.cache_data(ttl=120, show_spinner="Loading data from database…")
def load_all_data(days: int) -> dict:
    """Load all dashboard data for the given time window."""
    try:
        return {
            "baseline": get_baseline_comparison(),
            "cost_time": get_cost_over_time(days=days),
            "routing": get_routing_distribution(),
            "quality": get_quality_metrics(days=days),
            "escalations": get_recent_escalations(limit=25),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


data = load_all_data(days_back)

if data.get("error"):
    st.error(
        f"**Database connection failed:** {data['error']}\n\n"
        "Make sure Postgres is running and `DATABASE_URL` is set in your `.env` file, "
        "then run `python db/init_db.py` to create the tables."
    )
    st.stop()

df_baseline = data["baseline"]
df_cost = data["cost_time"]
df_routing = data["routing"]
df_quality = data["quality"]
df_escalations = data["escalations"]

# ---------------------------------------------------------------------------
# 1. Headline metrics
# ---------------------------------------------------------------------------

st.markdown('<div class="section-header">💰 Cost Savings vs. Baseline</div>', unsafe_allow_html=True)

total_actual = float(df_baseline["actual_cost"].sum()) if not df_baseline.empty else 0.0
total_baseline = float(df_baseline["baseline_cost"].sum()) if not df_baseline.empty else 0.0
total_savings = total_baseline - total_actual
savings_pct = (total_savings / total_baseline * 100) if total_baseline > 0 else 0.0
total_requests = int(df_cost["request_count"].sum()) if not df_cost.empty else 0

value_class = "metric-value" if total_savings >= 0 else "metric-value negative"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""<div class="metric-card">
            <div class="{value_class}">${total_savings:,.4f}</div>
            <div class="metric-label">Total Saved (All Time)</div>
            <div class="metric-sub">vs. all-premium routing</div>
        </div>""",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""<div class="metric-card">
            <div class="{value_class}">{savings_pct:.1f}%</div>
            <div class="metric-label">Cost Reduction</div>
            <div class="metric-sub">vs. baseline model</div>
        </div>""",
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-value" style="color:#63b3ed;">${total_actual:,.4f}</div>
            <div class="metric-label">Actual Spend</div>
            <div class="metric-sub">smart routing applied</div>
        </div>""",
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-value" style="color:#a78bfa;">{total_requests:,}</div>
            <div class="metric-label">Total Requests</div>
            <div class="metric-sub">last {days_back} days</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2. Cost over time
# ---------------------------------------------------------------------------

st.markdown('<div class="section-header">📈 Cost Over Time</div>', unsafe_allow_html=True)

if df_cost.empty:
    st.info("No request data found for the selected time window.")
else:
    # Merge with baseline for the same date range
    df_cost_merged = df_cost.copy()
    if not df_baseline.empty:
        df_cost_merged = df_cost_merged.merge(
            df_baseline[["date", "baseline_cost"]],
            on="date",
            how="left",
        )
        df_cost_merged["baseline_cost"] = df_cost_merged["baseline_cost"].fillna(0)

    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(
        x=df_cost_merged["date"],
        y=df_cost_merged["total_cost"],
        name="Actual cost",
        mode="lines+markers",
        line=dict(color="#48bb78", width=2.5),
        marker=dict(size=5),
        fill="tozeroy",
        fillcolor="rgba(72,187,120,0.08)",
    ))
    if "baseline_cost" in df_cost_merged.columns:
        fig_cost.add_trace(go.Scatter(
            x=df_cost_merged["date"],
            y=df_cost_merged["baseline_cost"],
            name="Baseline (all premium)",
            mode="lines",
            line=dict(color="#fc8181", width=2, dash="dash"),
        ))
    fig_cost.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#a0aec0", family="Inter"),
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis=dict(
            title="Cost (USD)",
            gridcolor="#1a1f2e",
            tickformat="$.4f",
        ),
        xaxis=dict(gridcolor="#1a1f2e"),
        height=320,
    )
    st.plotly_chart(fig_cost, use_container_width=True)

# ---------------------------------------------------------------------------
# 3. Routing distribution
# ---------------------------------------------------------------------------

st.markdown('<div class="section-header">🗂️ Routing Distribution</div>', unsafe_allow_html=True)

if df_routing.empty:
    st.info("No routing data available.")
else:
    col_pie, col_bar = st.columns(2)

    with col_pie:
        # Tier-level pie
        tier_agg = (
            df_routing.groupby("complexity_tier")["request_count"]
            .sum()
            .reset_index()
        )
        fig_pie = px.pie(
            tier_agg,
            values="request_count",
            names="complexity_tier",
            title="By Complexity Tier",
            color_discrete_sequence=["#48bb78", "#63b3ed", "#a78bfa"],
            hole=0.45,
        )
        fig_pie.update_layout(
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="#a0aec0", family="Inter"),
            margin=dict(l=0, r=0, t=40, b=0),
            height=320,
            showlegend=True,
            legend=dict(orientation="v"),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        # Model-level bar
        fig_bar = px.bar(
            df_routing.sort_values("request_count", ascending=True),
            x="request_count",
            y="model_id_used",
            orientation="h",
            title="By Model",
            color="complexity_tier",
            color_discrete_map={
                "simple": "#48bb78",
                "moderate": "#63b3ed",
                "complex": "#a78bfa",
            },
            labels={"request_count": "Requests", "model_id_used": "Model"},
        )
        fig_bar.update_layout(
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="#a0aec0", family="Inter"),
            margin=dict(l=0, r=0, t=40, b=0),
            height=320,
            legend_title_text="Tier",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------------------------
# 4. Quality & escalation trend
# ---------------------------------------------------------------------------

st.markdown('<div class="section-header">🎯 Quality & Escalation Trend</div>', unsafe_allow_html=True)

if df_quality.empty:
    st.info("No verification data found. Run some requests through the verification loop first.")
else:
    fig_quality = go.Figure()

    # Primary axis: avg quality score
    fig_quality.add_trace(go.Scatter(
        x=df_quality["date"],
        y=df_quality["avg_quality"],
        name="Avg Quality Score",
        mode="lines+markers",
        line=dict(color="#48bb78", width=2.5),
        marker=dict(size=5),
        yaxis="y1",
    ))

    # Secondary axis: escalation rate %
    fig_quality.add_trace(go.Scatter(
        x=df_quality["date"],
        y=df_quality["escalation_rate"],
        name="Escalation Rate (%)",
        mode="lines+markers",
        line=dict(color="#fc8181", width=2, dash="dot"),
        marker=dict(size=5, symbol="diamond"),
        yaxis="y2",
    ))

    fig_quality.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#a0aec0", family="Inter"),
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=0, r=0, t=30, b=0),
        height=320,
        yaxis=dict(
            title="Avg Quality Score",
            gridcolor="#1a1f2e",
            range=[0, 1.05],
            tickformat=".2f",
        ),
        yaxis2=dict(
            title="Escalation Rate (%)",
            overlaying="y",
            side="right",
            gridcolor="#1a1f2e",
            range=[0, 100],
            tickformat=".1f",
        ),
        xaxis=dict(gridcolor="#1a1f2e"),
    )
    st.plotly_chart(fig_quality, use_container_width=True)

# ---------------------------------------------------------------------------
# 5. Recent escalations table
# ---------------------------------------------------------------------------

st.markdown('<div class="section-header">🚨 Recent Escalations (System Self-Corrections)</div>', unsafe_allow_html=True)

if df_escalations.empty:
    st.success(
        "No escalations recorded. The cheap model is meeting quality standards — or "
        "verification hasn't run yet."
    )
else:
    # Format for display
    display_df = df_escalations.copy()
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")
    display_df["quality_score"] = display_df["quality_score"].apply(lambda x: f"{x:.3f}")
    display_df = display_df.rename(columns={
        "timestamp": "Time",
        "complexity_tier": "Tier",
        "model_id_used": "Cheap Model",
        "reference_model_id": "Reference Model",
        "quality_score": "Score",
        "judge_justification": "Judge Note",
        "prompt_preview": "Prompt (preview)",
    })
    display_df = display_df.drop(columns=["request_id"], errors="ignore")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0.0, max_value=1.0, format="%.3f"
            ),
        },
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.markdown(
    "<p style='text-align:center;color:#4a5568;font-size:0.78rem;'>"
    "LLM Cost Autopilot · Phase 4 Dashboard · Data refreshes every 2 minutes"
    "</p>",
    unsafe_allow_html=True,
)
