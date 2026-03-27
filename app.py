from __future__ import annotations
 
import os
import time
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
PREDICT_URL = f"{API_BASE}/predict"
BATCH_URL = f"{API_BASE}/predict/batch"
HEALTH_URL = f"{API_BASE}/health"
 
TIER_COLOR = {
    "Low": "#22c55e",
    "Medium": "#f59e0b",
    "High": "#f97316",
    "Critical": "#ef4444",
}
 
st.set_page_config(
    page_title="Customer Retention Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;600;700&family=Space+Mono:wght@400;700&display=swap');
 
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
 
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-card .label { color: #94a3b8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.1em; }
    .metric-card .value { color: #f1f5f9; font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; margin: 4px 0; }
    .metric-card .sub   { color: #64748b; font-size: 0.75rem; }
 
    .risk-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        font-family: 'Space Mono', monospace;
    }
 
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)
 
def check_api_health() -> dict | None:
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        return r.json()
    except Exception:
        return None
 
 
def predict_single(payload: dict) -> dict | None:
    try:
        r = requests.post(PREDICT_URL, json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None
 
 
def predict_batch(customers: list[dict]) -> dict | None:
    try:
        r = requests.post(BATCH_URL, json={"customers": customers}, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None
 
 
def gauge_chart(prob: float, tier: str) -> go.Figure:
    color = TIER_COLOR[tier]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"size": 40, "family": "Space Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#475569"},
                "bar": {"color": color},
                "bgcolor": "#1e293b",
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0, 30], "color": "#14532d"},
                    {"range": [30, 55], "color": "#78350f"},
                    {"range": [55, 75], "color": "#7c2d12"},
                    {"range": [75, 100], "color": "#450a0a"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.85,
                    "value": round(prob * 100, 1),
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="#0f172a",
        font_color="#f1f5f9",
        height=260,
        margin=dict(t=20, b=10, l=20, r=20),
    )
    return fig
 
 
with st.sidebar:
    st.markdown("## Customer Retention")
    st.markdown("---")
 
    health = check_api_health()
    if health and health.get("model_loaded"):
        st.success("API Online")
        st.caption(f"Model: `{health.get('model_name', '—')}`")
        st.caption(f"Stage: `{health.get('model_stage', '—')}`")
    else:
        st.error("API Offline")
        st.caption(f"Endpoint: `{API_BASE}`")
 
    st.markdown("---")
    mode = st.radio("Mode", ["Single Customer", "Batch Upload"], label_visibility="collapsed")
    st.markdown("---")
 
    st.markdown("#### Feature Guide")
    st.markdown(
        """
| Feature | Meaning |
|---|---|
| Frequency | Unique invoices |
| Monetary | Total revenue (£) |
| F_Score | Freq. quintile 1–5 |
| M_Score | Monetary quintile 1–5 |
        """
    )
 
st.title("Customer Retention Intelligence")
st.markdown(
    "Predict churn risk from RFM profiles using your MLflow-registered XGBoost model."
)
st.markdown("---")
 
if mode == "Single Customer":
    st.subheader("Single Customer Prediction")
 
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        frequency = st.number_input("Frequency (invoices)", min_value=1, max_value=5000, value=8)
    with col2:
        monetary = st.number_input("Monetary (£)", min_value=1.0, max_value=200000.0, value=1250.0, step=50.0)
    with col3:
        f_score = st.slider("F_Score", 1, 5, 3)
    with col4:
        m_score = st.slider("M_Score", 1, 5, 3)
 
    if st.button("Predict Churn Risk", use_container_width=True):
        if health and health.get("model_loaded"):
            with st.spinner("Running inference…"):
                payload = {
                    "Frequency": frequency,
                    "Monetary": monetary,
                    "F_Score": f_score,
                    "M_Score": m_score,
                }
                result = predict_single(payload)
 
            if result:
                pred = result["prediction"]
                prob = pred["churn_probability"]
                tier = pred["risk_tier"]
                color = TIER_COLOR[tier]
 
                st.markdown("---")
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    st.plotly_chart(gauge_chart(prob, tier), use_container_width=True)
                with c2:
                    st.markdown(
                        f"""
                        <div class="metric-card" style="margin-top:20px">
                            <div class="label">Churn Probability</div>
                            <div class="value">{prob*100:.1f}%</div>
                            <div class="sub">Model confidence</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="label">Risk Tier</div>
                            <div class="value" style="color:{color}">{tier}</div>
                            <div class="sub">{"Immediate action needed" if tier == "Critical" else "Monitor closely" if tier == "High" else "Standard monitoring" if tier == "Medium" else "Healthy — maintain"}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c3:
                    st.markdown(
                        f"""
                        <div class="metric-card" style="margin-top:20px">
                            <div class="label">Recommended Action</div>
                            <div class="value" style="font-size:1.1rem; margin-top:10px">
                                {"VIP Discount Offer" if tier == "Critical" else "Personal Outreach" if tier == "High" else "Re-engagement Email" if tier == "Medium" else "Loyalty Reward"}
                            </div>
                            <div class="sub" style="margin-top:12px">{"Deploy $20 coupon within 48h" if tier in ("Critical","High") else "Automated nurture sequence"}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.warning("API is not reachable. Start the FastAPI server first.")
 
else:
    st.subheader("Batch Prediction")
    st.markdown(
        "Upload a CSV with columns: `Frequency`, `Monetary`, `F_Score`, `M_Score` "
        "(and optionally `Customer ID`)."
    )
 
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
 
    if uploaded:
        df = pd.read_csv(uploaded)
        required = {"Frequency", "Monetary", "F_Score", "M_Score"}
        if not required.issubset(df.columns):
            st.error(f"Missing columns: {required - set(df.columns)}")
        else:
            st.markdown(f"**{len(df):,} customers loaded.** Preview:")
            st.dataframe(df.head(), use_container_width=True)
 
            if st.button("Run Batch Prediction", use_container_width=True):
                if health and health.get("model_loaded"):
                    customers = df[list(required)].to_dict(orient="records")
                    with st.spinner(f"Scoring {len(customers)} customers…"):
                        result = predict_batch(customers)
 
                    if result:
                        preds = result["predictions"]
                        df["churn_probability"] = [p["churn_probability"] for p in preds]
                        df["risk_tier"] = [p["risk_tier"] for p in preds]
                        df["churn_label"] = [p["churn_label"] for p in preds]
 
                        st.markdown("---")
                        k1, k2, k3, k4 = st.columns(4)
                        tier_counts = df["risk_tier"].value_counts()
 
                        with k1:
                            st.metric("Total Customers", f"{len(df):,}")
                        with k2:
                            churned = df["churn_label"].sum()
                            st.metric("Predicted Churners", f"{churned:,}", delta=f"{churned/len(df)*100:.1f}%")
                        with k3:
                            critical = tier_counts.get("Critical", 0)
                            st.metric("Critical Risk", f"{critical:,}")
                        with k4:
                            avg = df["churn_probability"].mean()
                            st.metric("Avg Churn Prob", f"{avg*100:.1f}%")
 
                        col_a, col_b = st.columns(2)
                        with col_a:
                            tier_order = ["Low", "Medium", "High", "Critical"]
                            counts = [tier_counts.get(t, 0) for t in tier_order]
                            fig_bar = go.Figure(
                                go.Bar(
                                    x=tier_order,
                                    y=counts,
                                    marker_color=[TIER_COLOR[t] for t in tier_order],
                                )
                            )
                            fig_bar.update_layout(
                                title="Customers by Risk Tier",
                                paper_bgcolor="#0f172a",
                                plot_bgcolor="#0f172a",
                                font_color="#f1f5f9",
                                xaxis=dict(gridcolor="#1e293b"),
                                yaxis=dict(gridcolor="#1e293b"),
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)
 
                        with col_b:
                            fig_hist = px.histogram(
                                df,
                                x="churn_probability",
                                nbins=30,
                                color_discrete_sequence=["#6366f1"],
                                title="Churn Probability Distribution",
                            )
                            fig_hist.update_layout(
                                paper_bgcolor="#0f172a",
                                plot_bgcolor="#0f172a",
                                font_color="#f1f5f9",
                                xaxis=dict(gridcolor="#1e293b"),
                                yaxis=dict(gridcolor="#1e293b"),
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
 
                        st.markdown("### Full Results")
                        st.dataframe(
                            df.sort_values("churn_probability", ascending=False),
                            use_container_width=True,
                        )
 
                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download Results CSV",
                            csv,
                            "churn_predictions.csv",
                            "text/csv",
                            use_container_width=True,
                        )
                else:
                    st.warning("API is not reachable.")
 
st.markdown("---")
st.caption("Customer Retention Intelligence · Built with FastAPI + MLflow + Streamlit")