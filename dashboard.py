"""
dashboard.py — Evaluation Dashboard
-------------------------------------
Run: streamlit run dashboard.py
(uvicorn main:app --reload also running)
"""

import streamlit as st
import requests
import pandas as pd
import time

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="📊 EduBot Dashboard", layout="wide")
st.title("📊 EduBot — Evaluation Dashboard")

# ── Auto refresh ───────────────────────────────────────────────────────────────
if st.button("🔄 Refresh"):
    st.rerun()

st.caption(f"Last updated: {time.strftime('%H:%M:%S')}")

# ── Fetch data ─────────────────────────────────────────────────────────────────
try:
    summary = requests.get(f"{API}/monitor/summary", timeout=5).json()
    recent  = requests.get(f"{API}/monitor/recent?n=50", timeout=5).json().get("recent", [])
    cache   = requests.get(f"{API}/cache/stats", timeout=5).json()
    health  = requests.get(f"{API}/health", timeout=5).json()
    api_ok  = True
except:
    api_ok  = False
    st.error("❌ API not running! Start uvicorn first.")
    st.stop()

# ── Health ─────────────────────────────────────────────────────────────────────
st.success(f"✅ API Status: {health.get('status', 'Unknown')}")

# ── Top metrics ────────────────────────────────────────────────────────────────
st.divider()
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Queries", summary.get("total_queries", 0))
with col2:
    st.metric("Avg Latency", f"{summary.get('avg_latency_sec', 0)}s")
with col3:
    st.metric("Total Cost", f"${summary.get('total_cost_usd', 0):.6f}")
with col4:
    st.metric("Total Tokens", summary.get("total_tokens", 0))
with col5:
    st.metric("Cache Entries", cache.get("valid_entries", 0))

# ── Charts ─────────────────────────────────────────────────────────────────────
st.divider()

if recent:
    df = pd.DataFrame(recent)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("⏱️ Latency per Query")
        if "latency_sec" in df.columns:
            st.line_chart(df["latency_sec"])

    with col_right:
        st.subheader("🔍 Search Type Breakdown")
        breakdown = summary.get("search_type_breakdown", {})
        if breakdown:
            breakdown_df = pd.DataFrame(
                list(breakdown.items()),
                columns=["Search Type", "Count"]
            )
            st.bar_chart(breakdown_df.set_index("Search Type"))

    # ── Token usage ────────────────────────────────────────────────────────────
    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🪙 Token Usage per Query")
        if "total_tokens" in df.columns:
            st.bar_chart(df["total_tokens"])

    with col_b:
        st.subheader("📊 Status Breakdown")
        if "status" in df.columns:
            status_counts = df["status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            st.bar_chart(status_counts.set_index("Status"))

    # ── Recent queries table ───────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Recent Queries")
    display_cols = ["timestamp", "question", "latency_sec", "search_type", "total_tokens", "cost_usd", "status"]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available].tail(10), use_container_width=True)

else:
    st.info("No queries yet — ask some questions first!")

# ── Cache stats ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("⚡ Cache Stats")
col_c1, col_c2, col_c3 = st.columns(3)
with col_c1:
    st.metric("Memory Entries", cache.get("memory_entries", 0))
with col_c2:
    st.metric("Disk Entries", cache.get("disk_entries", 0))
with col_c3:
    st.metric("TTL", f"{cache.get('ttl_seconds', 0)}s")

# ── Clear buttons ──────────────────────────────────────────────────────────────
st.divider()
col_d1, col_d2 = st.columns(2)
with col_d1:
    if st.button("🗑️ Clear Monitor Log"):
        requests.delete(f"{API}/monitor/clear")
        st.success("Monitor log cleared!")
        st.rerun()
with col_d2:
    if st.button("🗑️ Clear Cache"):
        requests.delete(f"{API}/cache/clear")
        st.success("Cache cleared!")
        st.rerun()
# ── Eval Report ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("🧪 Evaluation Report")

try:
    eval_data = requests.get(f"{API}/eval/report", timeout=5).json()
    eval_summary = eval_data.get("summary", {})
    eval_results = eval_data.get("results", [])

    # Summary metrics
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    with col_e1:
        st.metric("Total Tested", eval_summary.get("total_tested", 0))
    with col_e2:
        st.metric("Avg Eval Score", eval_summary.get("avg_eval_score", 0))
    with col_e3:
        grounding = eval_summary.get("grounding_distribution", {})
        grounded = grounding.get("grounded", 0) + grounding.get("partial", 0)
        total = sum(grounding.values()) or 1
        st.metric("Grounded Rate", f"{round(grounded/total*100, 1)}%")
    with col_e4:
        grades = eval_summary.get("grade_distribution", {})
        a_grades = grades.get("A 🟢", 0)
        st.metric("A-Grade Answers", a_grades)

    # Grade distribution chart
    if grades:
        st.subheader("📊 Grade Distribution")
        grade_df = pd.DataFrame(
            list(grades.items()),
            columns=["Grade", "Count"]
        )
        st.bar_chart(grade_df.set_index("Grade"))

    # Grounding distribution
    if grounding:
        st.subheader("🛡️ Grounding Distribution")
        ground_df = pd.DataFrame(
            list(grounding.items()),
            columns=["Label", "Count"]
        )
        st.bar_chart(ground_df.set_index("Label"))

    # Per-question results table
    if eval_results:
        st.subheader("📋 Per-Question Results")
        rows = []
        for r in eval_results:
            rows.append({
                "Question": r.get("question", "")[:50],
                "Grade": r.get("eval_grade", ""),
                "Score": r.get("eval_overall_score", 0),
                "Grounding": r.get("grounding_label", ""),
                "G-Score": r.get("grounding_score", 0),
                "Search": r.get("search_type", ""),
            })
        eval_df = pd.DataFrame(rows)
        st.dataframe(eval_df, use_container_width=True)

    st.caption(f"Last eval run: {eval_summary.get('last_run', 'Never')}")

    # Re-run eval button
    if st.button("🔁 Re-run Evaluation"):
        st.info("Run `python run_eval.py` in terminal to update eval results.")

except Exception as e:
    st.warning(f"⚠️ No eval report found. Run `python run_eval.py` first. ({e})")