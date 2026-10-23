"""
visualizer.py
=============
Streamlit Observability Dashboard for the Secure AI Agent.
Visualizes real-time metrics across:

  📊 Cost & Latency Monitor
  🧠 Cache Performance (L1 + L2)
  🔒 Security Filter Activity
  🏥🎓 Domain Usage (Education vs Healthcare)
  👥 Multi-Tenant Activity
  📈 Retrieval Quality (Precision, Recall, MRR)
  🚨 Audit Log & Threat Timeline

Usage:
    streamlit run observability/visualizer.py

Dependencies:
    pip install streamlit plotly pandas
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import random
import time
from datetime import datetime, timedelta
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Secure AI Agent — Observability",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Mock Data Generator
# Replace these with real imports from your monitoring modules:
#   from cost_monitor import get_cost_metrics
#   from cache import get_cache_stats
#   from security_filter import get_filter_stats
#   from auto_eval_pipeline import get_eval_scores
# ─────────────────────────────────────────────────────────────────────────────

def _seed(offset=0):
    random.seed(int(time.time() / 30) + offset)

def get_cost_metrics():
    _seed(1)
    now = datetime.now()
    hours = [(now - timedelta(hours=i)).strftime("%H:%M") for i in range(23, -1, -1)]
    return {
        "total_cost_usd":     round(random.uniform(1.2, 4.8), 3),
        "cost_today_usd":     round(random.uniform(0.3, 1.2), 3),
        "avg_cost_per_query": round(random.uniform(0.002, 0.008), 4),
        "total_tokens_used":  random.randint(80000, 250000),
        "hourly_cost": [
            {"hour": h, "cost": round(random.uniform(0.01, 0.25), 3)}
            for h in hours
        ],
        "cost_by_tenant": {
            "anna_univ":  round(random.uniform(0.4, 1.2), 3),
            "vit":        round(random.uniform(0.3, 0.9), 3),
            "hospital_a": round(random.uniform(0.5, 1.5), 3),
        },
    }

def get_latency_metrics():
    _seed(2)
    return {
        "avg_latency_ms":    round(random.uniform(120, 280), 1),
        "p50_latency_ms":    round(random.uniform(90, 180), 1),
        "p95_latency_ms":    round(random.uniform(300, 600), 1),
        "p99_latency_ms":    round(random.uniform(600, 1100), 1),
        "latency_by_stage": {
            "Query Rewrite":    round(random.uniform(15, 45), 1),
            "Hybrid Retrieval": round(random.uniform(50, 150), 1),
            "Re-ranking":       round(random.uniform(20, 60), 1),
            "LLM Generation":   round(random.uniform(80, 250), 1),
            "Security Filter":  round(random.uniform(5, 20), 1),
            "Cache Lookup":     round(random.uniform(2, 8), 1),
        },
    }

def get_cache_stats():
    _seed(3)
    total = random.randint(400, 800)
    l1_hits = int(total * random.uniform(0.30, 0.45))
    l2_hits = int((total - l1_hits) * random.uniform(0.28, 0.38))
    misses  = total - l1_hits - l2_hits
    return {
        "total_requests": total,
        "l1_hits":        l1_hits,
        "l2_hits":        l2_hits,
        "misses":         misses,
        "hit_rate_pct":   round((l1_hits + l2_hits) / total * 100, 1),
        "l1_hit_rate":    round(l1_hits / total * 100, 1),
        "l2_hit_rate":    round(l2_hits / total * 100, 1),
        "estimated_saved_usd": round((l1_hits + l2_hits) * 0.004, 3),
    }

def get_security_stats():
    _seed(4)
    total = random.randint(50, 120)
    blocked = random.randint(15, 35)
    return {
        "total_queries":     total,
        "blocked":           blocked,
        "allowed":           total - blocked,
        "block_rate_pct":    round(blocked / total * 100, 1),
        "attacks_by_type": {
            "Prompt Injection":    random.randint(2, 8),
            "Jailbreak":           random.randint(1, 5),
            "PII Extraction":      random.randint(1, 4),
            "Cross-Tenant Leak":   random.randint(1, 3),
            "Role Escalation":     random.randint(1, 4),
            "System Prompt Leak":  random.randint(0, 3),
            "Other":               random.randint(1, 5),
        },
        "recent_threats": [
            {
                "time":     (datetime.now() - timedelta(minutes=random.randint(1, 120))).strftime("%H:%M"),
                "tenant":   random.choice(["anna_univ", "vit", "hospital_a"]),
                "category": random.choice(["Jailbreak", "Prompt Injection", "PII Extraction", "Role Escalation"]),
                "severity": random.choice(["🔴 Critical", "🟠 High", "🟡 Medium"]),
            }
            for _ in range(8)
        ],
    }

def get_retrieval_metrics():
    _seed(5)
    return {
        "precision_at_5":  round(random.uniform(0.60, 0.85), 3),
        "recall_at_5":     round(random.uniform(0.55, 0.80), 3),
        "mrr":             round(random.uniform(0.50, 0.75), 3),
        "hit_rate_pct":    round(random.uniform(70, 92), 1),
        "isolation_pct":   round(random.uniform(95, 100), 1),
        "by_domain": {
            "education":  {
                "precision": round(random.uniform(0.62, 0.88), 3),
                "recall":    round(random.uniform(0.58, 0.82), 3),
            },
            "healthcare": {
                "precision": round(random.uniform(0.65, 0.90), 3),
                "recall":    round(random.uniform(0.60, 0.85), 3),
            },
        },
    }

def get_tenant_activity():
    _seed(6)
    tenants = ["anna_univ", "vit", "hospital_a"]
    now = datetime.now()
    return {
        "queries_by_tenant": {t: random.randint(40, 200) for t in tenants},
        "active_users":      {t: random.randint(5, 40)   for t in tenants},
        "domain_split": {"Education": random.randint(55, 75), "Healthcare": random.randint(25, 45)},
        "query_timeline": [
            {
                "time":   (now - timedelta(minutes=i * 5)).strftime("%H:%M"),
                "anna_univ":  random.randint(0, 8),
                "vit":        random.randint(0, 6),
                "hospital_a": random.randint(0, 5),
            }
            for i in range(23, -1, -1)
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.shields.io/badge/Secure_AI_Agent-v1.0-blue", use_container_width=True)
    st.markdown("## 🛡️ Observability Dashboard")
    st.markdown("**Secure Multi-Tenant Document Intelligence Agent**")
    st.markdown("---")

    selected_view = st.radio(
        "View",
        ["📊 Overview", "💰 Cost & Latency", "🧠 Cache", "🔒 Security", "📈 Retrieval Quality", "👥 Tenant Activity"],
        index=0,
    )

    st.markdown("---")
    auto_refresh = st.toggle("Auto-refresh (30s)", value=False)
    if auto_refresh:
        st.caption("🔄 Refreshing every 30 seconds")

    if st.button("🔄 Refresh Now", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    st.caption("📍 Chennai, India")


# ─────────────────────────────────────────────────────────────────────────────
# Load all metrics
# ─────────────────────────────────────────────────────────────────────────────

cost    = get_cost_metrics()
lat     = get_latency_metrics()
cache   = get_cache_stats()
sec     = get_security_stats()
retr    = get_retrieval_metrics()
tenant  = get_tenant_activity()


# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

if selected_view == "📊 Overview":
    st.title("📊 System Overview")
    st.caption("Real-time health of the Secure AI Agent pipeline")

    # ── KPI Row 1 ────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Cost Today",      f"${cost['cost_today_usd']}",       delta="-12% vs yesterday")
    c2.metric("⚡ Avg Latency",     f"{lat['avg_latency_ms']} ms",       delta="-8 ms")
    c3.metric("🧠 Cache Hit Rate",  f"{cache['hit_rate_pct']}%",         delta="+2.1%")
    c4.metric("🔒 Threats Blocked", f"{sec['blocked']}",                 delta=f"+{random.randint(1,5)} today")
    c5.metric("📈 Retrieval MRR",   f"{retr['mrr']}",                    delta="+0.03")

    st.markdown("---")

    # ── KPI Row 2 ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎓 Education Queries", tenant["queries_by_tenant"]["anna_univ"] + tenant["queries_by_tenant"]["vit"])
    c2.metric("🏥 Healthcare Queries", tenant["queries_by_tenant"]["hospital_a"])
    c3.metric("🔢 Total Tokens",      f"{cost['total_tokens_used']:,}")
    c4.metric("🏢 Active Tenants",    "3")

    st.markdown("---")

    # ── Query Timeline + Domain Pie ──────────────────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📉 Query Volume (Last 2 Hours)")
        df_tl = pd.DataFrame(tenant["query_timeline"])
        fig = go.Figure()
        colors = {"anna_univ": "#4F8EF7", "vit": "#F7934F", "hospital_a": "#4FF79E"}
        labels = {"anna_univ": "Anna Univ", "vit": "VIT", "hospital_a": "Hospital A"}
        for col in ["anna_univ", "vit", "hospital_a"]:
            fig.add_trace(go.Scatter(
                x=df_tl["time"], y=df_tl[col],
                mode="lines+markers", name=labels[col],
                line=dict(color=colors[col], width=2),
                fill="tozeroy", opacity=0.15,
            ))
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                          legend=dict(orientation="h"), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("🌐 Domain Split")
        fig2 = go.Figure(go.Pie(
            labels=list(tenant["domain_split"].keys()),
            values=list(tenant["domain_split"].values()),
            hole=0.55,
            marker_colors=["#4F8EF7", "#F74F6E"],
        ))
        fig2.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                           showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Pipeline Stage Latency ────────────────────────────────────────────────
    st.subheader("⚡ Pipeline Stage Latency Breakdown")
    stages = lat["latency_by_stage"]
    fig3 = go.Figure(go.Bar(
        x=list(stages.values()),
        y=list(stages.keys()),
        orientation="h",
        marker_color=["#4F8EF7", "#4FF79E", "#F7934F", "#F74F6E", "#A44FF7", "#F7E34F"],
        text=[f"{v} ms" for v in stages.values()],
        textposition="outside",
    ))
    fig3.update_layout(height=280, xaxis_title="ms",
                       margin=dict(l=0, r=0, t=10, b=0),
                       plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# COST & LATENCY
# ─────────────────────────────────────────────────────────────────────────────

elif selected_view == "💰 Cost & Latency":
    st.title("💰 Cost & Latency Monitor")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cost",         f"${cost['total_cost_usd']}")
    c2.metric("Cost Today",         f"${cost['cost_today_usd']}")
    c3.metric("Avg / Query",        f"${cost['avg_cost_per_query']}")
    c4.metric("Tokens Used",        f"{cost['total_tokens_used']:,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Hourly Cost ($)")
        df_cost = pd.DataFrame(cost["hourly_cost"])
        fig = px.area(df_cost, x="hour", y="cost", color_discrete_sequence=["#4F8EF7"])
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🏢 Cost by Tenant")
        tenants_cost = cost["cost_by_tenant"]
        fig2 = go.Figure(go.Bar(
            x=list(tenants_cost.keys()),
            y=list(tenants_cost.values()),
            marker_color=["#4F8EF7", "#F7934F", "#4FF79E"],
            text=[f"${v}" for v in tenants_cost.values()],
            textposition="outside",
        ))
        fig2.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                           plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("⚡ Latency Percentiles")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P50", f"{lat['p50_latency_ms']} ms")
    c2.metric("Avg", f"{lat['avg_latency_ms']} ms")
    c3.metric("P95", f"{lat['p95_latency_ms']} ms")
    c4.metric("P99", f"{lat['p99_latency_ms']} ms")

    st.subheader("🔍 Latency by Pipeline Stage")
    stages = lat["latency_by_stage"]
    df_stages = pd.DataFrame({"stage": list(stages.keys()), "ms": list(stages.values())})
    fig3 = px.bar(df_stages, x="stage", y="ms", color="ms",
                  color_continuous_scale="Blues", text="ms")
    fig3.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0),
                       plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────────────────────────────────────

elif selected_view == "🧠 Cache":
    st.title("🧠 Cache Performance (L1 + L2)")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Requests",    cache["total_requests"])
    c2.metric("Overall Hit Rate",  f"{cache['hit_rate_pct']}%",  delta="+2.1%")
    c3.metric("L1 Hit Rate",       f"{cache['l1_hit_rate']}%")
    c4.metric("L2 Hit Rate",       f"{cache['l2_hit_rate']}%")
    c5.metric("💰 Cost Saved",     f"${cache['estimated_saved_usd']}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🥧 Cache Hit Distribution")
        fig = go.Figure(go.Pie(
            labels=["L1 Hits", "L2 Hits", "Misses"],
            values=[cache["l1_hits"], cache["l2_hits"], cache["misses"]],
            hole=0.5,
            marker_colors=["#4FF79E", "#4F8EF7", "#F74F6E"],
        ))
        fig.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Cache Layer Breakdown")
        layers = ["L1 Hits", "L2 Hits", "Misses"]
        vals   = [cache["l1_hits"], cache["l2_hits"], cache["misses"]]
        colors = ["#4FF79E", "#4F8EF7", "#F74F6E"]
        fig2 = go.Figure(go.Bar(
            x=layers, y=vals,
            marker_color=colors,
            text=vals, textposition="outside",
        ))
        fig2.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0),
                           plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    st.info(
        f"💡 Cache is saving **${cache['estimated_saved_usd']}** in API costs by avoiding "
        f"**{cache['l1_hits'] + cache['l2_hits']} redundant LLM calls** today."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECURITY
# ─────────────────────────────────────────────────────────────────────────────

elif selected_view == "🔒 Security":
    st.title("🔒 Security Filter Activity")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Queries",  sec["total_queries"])
    c2.metric("✅ Allowed",     sec["allowed"])
    c3.metric("🚫 Blocked",     sec["blocked"])
    c4.metric("Block Rate",     f"{sec['block_rate_pct']}%", delta="+1.2%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔴 Attacks by Type")
        atk = sec["attacks_by_type"]
        fig = go.Figure(go.Bar(
            x=list(atk.values()),
            y=list(atk.keys()),
            orientation="h",
            marker_color="#F74F6E",
            text=list(atk.values()),
            textposition="outside",
        ))
        fig.update_layout(height=340, margin=dict(l=0,r=0,t=10,b=0),
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("✅ Allowed vs Blocked")
        fig2 = go.Figure(go.Pie(
            labels=["Allowed", "Blocked"],
            values=[sec["allowed"], sec["blocked"]],
            hole=0.5,
            marker_colors=["#4FF79E", "#F74F6E"],
        ))
        fig2.update_layout(height=340, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("🚨 Recent Threat Log")
    threats = sorted(sec["recent_threats"], key=lambda x: x["time"], reverse=True)
    df_threats = pd.DataFrame(threats)
    df_threats.columns = ["Time", "Tenant", "Attack Category", "Severity"]
    st.dataframe(df_threats, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL QUALITY
# ─────────────────────────────────────────────────────────────────────────────

elif selected_view == "📈 Retrieval Quality":
    st.title("📈 Retrieval Quality Metrics")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precision@5",        retr["precision_at_5"])
    c2.metric("Recall@5",           retr["recall_at_5"])
    c3.metric("MRR",                retr["mrr"])
    c4.metric("Hit Rate",           f"{retr['hit_rate_pct']}%")
    c5.metric("🔒 Tenant Isolation", f"{retr['isolation_pct']}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Metrics Comparison (Radar)")
        categories = ["Precision@5", "Recall@5", "MRR", "Hit Rate (norm)", "Isolation (norm)"]
        edu  = retr["by_domain"]["education"]
        hc   = retr["by_domain"]["healthcare"]
        fig  = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[edu["precision"], edu["recall"], retr["mrr"],
               retr["hit_rate_pct"]/100, retr["isolation_pct"]/100],
            theta=categories, fill="toself", name="Education",
            line_color="#4F8EF7",
        ))
        fig.add_trace(go.Scatterpolar(
            r=[hc["precision"], hc["recall"], retr["mrr"],
               retr["hit_rate_pct"]/100, retr["isolation_pct"]/100],
            theta=categories, fill="toself", name="Healthcare",
            line_color="#F74F6E",
        ))
        fig.update_layout(height=360, polar=dict(radialaxis=dict(range=[0, 1])),
                          margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎓🏥 Domain Comparison")
        domains = ["Education", "Healthcare"]
        prec_vals = [edu["precision"], hc["precision"]]
        rec_vals  = [edu["recall"],    hc["recall"]]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Precision", x=domains, y=prec_vals,
                              marker_color="#4F8EF7", text=[f"{v:.3f}" for v in prec_vals],
                              textposition="outside"))
        fig2.add_trace(go.Bar(name="Recall",    x=domains, y=rec_vals,
                              marker_color="#F7934F", text=[f"{v:.3f}" for v in rec_vals],
                              textposition="outside"))
        fig2.update_layout(height=360, barmode="group", yaxis=dict(range=[0, 1]),
                           margin=dict(l=0,r=0,t=10,b=0),
                           plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("🔒 Tenant Isolation Status")
    isolation_data = [
        {"Test":           "Anna Univ → VIT leak",
         "Status":         "✅ PASS", "Leaked Docs": 0},
        {"Test":           "VIT → Anna Univ leak",
         "Status":         "✅ PASS", "Leaked Docs": 0},
        {"Test":           "Hospital A → Education leak",
         "Status":         "✅ PASS", "Leaked Docs": 0},
        {"Test":           "Cross-domain isolation",
         "Status":         "✅ PASS", "Leaked Docs": 0},
    ]
    st.dataframe(pd.DataFrame(isolation_data), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TENANT ACTIVITY
# ─────────────────────────────────────────────────────────────────────────────

elif selected_view == "👥 Tenant Activity":
    st.title("👥 Multi-Tenant Activity Dashboard")

    c1, c2, c3 = st.columns(3)
    for col, (t_id, label, icon) in zip(
        [c1, c2, c3],
        [("anna_univ", "Anna University", "🎓"),
         ("vit",       "VIT",            "🎓"),
         ("hospital_a","Hospital A",     "🏥")],
    ):
        with col:
            st.metric(
                f"{icon} {label}",
                f"{tenant['queries_by_tenant'][t_id]} queries",
                delta=f"{tenant['active_users'][t_id]} active users",
            )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Queries by Tenant")
        t_data = tenant["queries_by_tenant"]
        fig = go.Figure(go.Bar(
            x=list(t_data.keys()),
            y=list(t_data.values()),
            marker_color=["#4F8EF7", "#F7934F", "#4FF79E"],
            text=list(t_data.values()),
            textposition="outside",
        ))
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🌐 Domain Split")
        fig2 = go.Figure(go.Pie(
            labels=list(tenant["domain_split"].keys()),
            values=list(tenant["domain_split"].values()),
            hole=0.55,
            marker_colors=["#4F8EF7", "#F74F6E"],
        ))
        fig2.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📉 Per-Tenant Query Timeline (Last 2 Hours)")
    df_tl = pd.DataFrame(tenant["query_timeline"])
    fig3 = go.Figure()
    colors = {"anna_univ": "#4F8EF7", "vit": "#F7934F", "hospital_a": "#4FF79E"}
    labels = {"anna_univ": "Anna Univ 🎓", "vit": "VIT 🎓", "hospital_a": "Hospital A 🏥"}
    for col in ["anna_univ", "vit", "hospital_a"]:
        fig3.add_trace(go.Scatter(
            x=df_tl["time"], y=df_tl[col],
            mode="lines+markers", name=labels[col],
            line=dict(color=colors[col], width=2),
        ))
    fig3.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                       legend=dict(orientation="h"),
                       plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "🛡️ **Secure Multi-Tenant Document Intelligence Agent** | "
    "Education 🎓 × Healthcare 🏥 | "
    f"Dashboard refreshed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)