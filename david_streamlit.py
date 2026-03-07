
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Ensure imports work from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import C, UP, DOWN, SIDEWAYS, NIFTY_SYMBOL
from data_engine import load_all_data
from feature_forge import engineer_features
from models.ensemble_classifier import EnsembleClassifier
from models.regime_detector import RegimeDetector
from models.range_predictor import RangePredictor
from models.sr_engine import SREngine
from analyzers.whipsaw_detector import WhipsawDetector
from analyzers.iron_condor_analyzer import IronCondorAnalyzer
from analyzers.bounce_analyzer import BounceAnalyzer

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="David Oracle v1.0",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode aesthetics
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #41444C;
        text-align: center;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .up-text { color: #00FF7F; }
    .down-text { color: #FF4B4B; }
    .side-text { color: #FFD700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZATION & CACHING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_oracle():
    """Load data and models once."""
    df_raw = load_all_data()
    df, features = engineer_features(df_raw)
    
    ensemble = EnsembleClassifier()
    if not ensemble.load():
        ensemble.train(df, features)
        ensemble.save()
        
    regime = RegimeDetector()
    if not regime.load():
        regime.train(df)
        regime.save()
        
    range_pred = RangePredictor()
    if not range_pred.load():
        range_pred.train(df, features)
        range_pred.save()
        
    sr = SREngine()
    whipsaw = WhipsawDetector()
    condor = IronCondorAnalyzer()
    bounce = BounceAnalyzer()
    
    return {
        "df_raw": df_raw,
        "df": df,
        "features": features,
        "ensemble": ensemble,
        "regime": regime,
        "range_pred": range_pred,
        "sr": sr,
        "whipsaw": whipsaw,
        "condor": condor,
        "bounce": bounce
    }

with st.spinner("Waking up David... (Loading models & data)"):
    oracle = load_oracle()

df = oracle["df"]
current_price = float(df["close"].iloc[-1])
vix = float(oracle["df_raw"]["vix"].iloc[-1]) if "vix" in oracle["df_raw"].columns else 15.0
last_date = df["date"].iloc[-1].strftime("%Y-%m-%d")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/eagle.png", width=80)
    st.title("David Oracle")
    st.markdown(f"**NIFTY**: {current_price:,.2f}")
    st.markdown(f"**VIX**: {vix:.2f}")
    st.markdown(f"**Date**: {last_date}")
    
    st.markdown("---")
    
    mode = st.radio("Navigation", [
        "Dashboard", 
        "Forecast & Ranges",
        "Strategy Lab"
    ])
    
    st.markdown("---")
    if st.button("♻️ Sync Data & Retrain"):
        with st.spinner("Fetching latest data and retraining..."):
            st.cache_resource.clear()
        st.success("Data synced! Reloading...")
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if mode == "Dashboard":
    st.title("🦅 Prophet Dashboard")
    
    # 1. Main Predictions
    pred = oracle["ensemble"].predict_today(df)
    regime_info = oracle["regime"].get_regime_with_micro_direction(df, pred)
    whipsaw = oracle["whipsaw"].analyze(df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔮 Verdict")
        direction = pred["direction"]
        conf = pred["confidence"] * 100
        
        color = "green" if direction == UP else "red" if direction == DOWN else "orange"
        st.markdown(f"<h1 style='color:{color}; text-align:center;'>{direction}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center;'>Confidence: {conf:.0f}%</h3>", unsafe_allow_html=True)
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = conf,
            title = {'text': "AI Confidence"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 40], 'color': "#333"},
                    {'range': [40, 70], 'color': "#555"},
                    {'range': [70, 100], 'color': "#777"}
                ]
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🏛️ Regime")
        r_label = regime_info["regime"]
        st.markdown(f"<h2 style='text-align:center;'>{r_label}</h2>", unsafe_allow_html=True)
        
        # Probabilities
        probs = regime_info.get("state_probs", {})
        if probs:
            df_probs = pd.DataFrame(list(probs.items()), columns=["State", "Prob"])
            fig = go.Figure(go.Bar(
                x=df_probs["Prob"],
                y=df_probs["State"],
                orientation='h',
                marker_color=['#00FF7F', '#7CFC00', '#FFD700', '#FFA500', '#FF4500']
            ))
            fig.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)",
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown("### ⚡ Whipsaw / Chop")
        is_chop = whipsaw["is_choppy"]
        prob_chop = whipsaw["whipsaw_prob"]
        
        status = "⚠️ CHOPPY" if is_chop else "✅ TRENDING"
        color = "orange" if is_chop else "green"
        
        st.markdown(f"<h2 style='color:{color}; text-align:center;'>{status}</h2>", unsafe_allow_html=True)
        st.progress(prob_chop / 100)
        st.caption(f"Chop Probability: {prob_chop:.0f}%")
        
        st.markdown("**Signals:**")
        for key, val in whipsaw["signals"].items():
            icon = "🔴" if val["weight"] > 0 else "🟢"
            st.text(f"{icon} {key}: {val['signal']}")

    st.markdown("---")
    
    # 2. Support & Resistance
    st.subheader("🔴 Support & Resistance")
    supports, resistances = oracle["sr"].find_levels(oracle["df_raw"])
    
    sr_col1, sr_col2 = st.columns(2)
    
    with sr_col1:
        st.markdown("#### Resistance (Overhead)")
        for r in resistances[:3]:
            dist = ((r['price'] - current_price) / current_price) * 100
            st.markdown(f"**R**: {r['price']:,.0f} (+{dist:.1f}%) — *Str: {r['strength']:.1f}*")
            
    with sr_col2:
        st.markdown("#### Support (Below)")
        for s in supports[:3]:
            dist = ((current_price - s['price']) / current_price) * 100
            st.markdown(f"**S**: {s['price']:,.0f} (-{dist:.1f}%) — *Str: {s['strength']:.1f}*")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: FORECAST & RANGES
# ─────────────────────────────────────────────────────────────────────────────
elif mode == "Forecast & Ranges":
    st.title("📈 Price Forecast")
    
    ranges = oracle["range_pred"].predict_range(df, current_price)
    
    tab7, tab30 = st.tabs(["7-Day Forecast", "30-Day Forecast"])
    
    with tab7:
        if 7 in ranges:
            r = ranges[7]
            
            fig = go.Figure()
            
            # Current Price Line
            fig.add_trace(go.Scatter(
                x=[0, 7], y=[current_price, current_price],
                mode="lines", name="Current Spot",
                line=dict(color="white", dash="dash")
            ))
            
            # Fan Chart / Range
            x_vals = [0, 7]
            
            # 90th percentile area
            fig.add_trace(go.Scatter(
                x=x_vals + x_vals[::-1],
                y=[current_price, r['p90']] + [r['p10'], current_price],
                fill='toself',
                fillcolor='rgba(0, 255, 255, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='80% Confidence'
            ))
            
            # 50th percentile area
            fig.add_trace(go.Scatter(
                x=x_vals + x_vals[::-1],
                y=[current_price, r['p75']] + [r['p25'], current_price],
                fill='toself',
                fillcolor='rgba(0, 255, 255, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='50% Confidence'
            ))
            
            # Median
            fig.add_trace(go.Scatter(
                x=[0, 7], y=[current_price, r['p50']],
                mode="lines+markers", name="Median Path",
                line=dict(color="cyan", width=3)
            ))
            
            fig.update_layout(
                title="7-Day Probability Cone",
                xaxis_title="Days from now",
                yaxis_title="Nifty Level",
                height=500,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Low (10%)", f"{r['p10']:,.0f}")
            c2.metric("Median (50%)", f"{r['p50']:,.0f}")
            c3.metric("High (90%)", f"{r['p90']:,.0f}")
            
    with tab30:
        if 30 in ranges:
            r = ranges[30]
            st.success(f"30-Day Target Range (80% Conf): **{r['p10']:,.0f} — {r['p90']:,.0f}**")
            # Similar plot could be added here

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: STRATEGY LAB
# ─────────────────────────────────────────────────────────────────────────────
elif mode == "Strategy Lab":
    st.title("🧪 Strategy Lab")
    
    st.subheader("🛡️ Iron Condor Analyzer")
    strike = st.number_input("Enter Strike Price to Test:", value=26000, step=100)
    days = st.slider("Timeframe (Days)", 1, 30, 5)
    
    if st.button("Analyze Strike"):
        res = oracle["condor"].analyze_strike(oracle["df_raw"], strike, days)
        
        st.markdown("#### Probability Analysis")
        col1, col2, col3 = st.columns(3)
        col1.metric("Touch Prob", f"{res['touch_prob']:.1f}%")
        col2.metric("Recovery Prob", f"{res['recovery_prob']:.1f}%")
        col3.metric("Firefight Level", f"{res['firefight_level']:,.0f}")
        
        if res['touch_prob'] > 60:
            st.error("🚨 HIGH RISK! High probability of touching this strike.")
        elif res['touch_prob'] > 35:
            st.warning("⚠️ MODERATE RISK. Keep monitoring.")
        else:
            st.success("✅ SAFE ZONE. Low touch probability.")
            
    st.markdown("---")
    
    st.subheader("🔄 Bounce-Back Calculator")
    target = st.number_input("Enter Target Price (Dip/Rally):", value=25000, step=100)
    
    if st.button("Check Bounce Probability"):
        res = oracle["bounce"].analyze(oracle["df_raw"], target)
        
        # Display table
        data = []
        for d, vals in res["timeframes"].items():
            data.append({
                "Days": d,
                "Recovery %": f"{vals['recovery_prob']:.1f}%",
                "Avg Days": f"{vals['avg_recovery_days']:.1f}"
            })
        st.table(pd.DataFrame(data))
