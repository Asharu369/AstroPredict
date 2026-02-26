# AstroPredict Phase-1 Dashboard 

import streamlit as st
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# =========================
# CONFIGURATION
# =========================
API_URL = "https://astropredict-api.onrender.com/predict/now"
REFRESH_INTERVAL_MS = 10000  # 10 seconds

st.set_page_config(
    page_title="AstroPredict Demo",
    layout="wide",
    page_icon="☀️",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM STYLING
# =========================
st.markdown("""
<style>
.stApp { 
    background-color: #0e1117; 
    color: #ffffff; 
}

/* Risk Level Cards */
.risk-card {
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    margin: 20px 0;
    border: 3px solid;
}

.risk-low {
    background: linear-gradient(135deg, #065f46 0%, #047857 100%);
    border-color: #10b981;
}

.risk-moderate {
    background: linear-gradient(135deg, #92400e 0%, #b45309 100%);
    border-color: #f59e0b;
}

.risk-elevated {
    background: linear-gradient(135deg, #9a3412 0%, #c2410c 100%);
    border-color: #fb923c;
}

.risk-high {
    background: linear-gradient(135deg, #991b1b 0%, #b91c1c 100%);
    border-color: #ef4444;
}

.risk-card h1 {
    margin: 0;
    font-size: 56px;
    font-weight: bold;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.risk-card h3 {
    margin: 10px 0 0 0;
    font-size: 24px;
    opacity: 0.9;
}

/* Info Boxes */
.info-box {
    padding: 20px;
    border-radius: 10px;
    background-color: #1e293b;
    border: 1px solid #334155;
    margin: 10px 0;
}

/* Metric Cards */
.metric-card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    border: 1px solid #334155;
}

.metric-card h2 {
    font-size: 32px;
    margin: 10px 0;
    color: #3b82f6;
}

.metric-card p {
    margin: 5px 0;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR — SYSTEM INFO
# =========================
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/"
        "Solar_Orbiter_artists_impression.jpg/640px-Solar_Orbiter_artists_impression.jpg",
        use_container_width=True
    )

    st.markdown("## 🎓 AstroPredict Phase-1")
    st.markdown("**Demonstration**")
   
    
    st.markdown("---")
    
    st.info("""
**Purpose:**
Live demonstration of AI-based solar flare forecasting 
using LSTM and BiLSTM neural networks.

**What You're Seeing:**
- Real-time GOES X-ray data (NOAA)
- Scenario-matched magnetic patterns
- 24-hour flare probability forecast
- Model consensus analysis
    """)
    
    st.warning("""
**⚠️ Demonstration Mode**

This is a **research prototype**, not an operational system.

**Data Sources:**
- ✅ Live X-ray flux (NOAA GOES-16)
- ⚠️ Magnetic fields (historical scenarios)

**Forecast Horizon:** Next 24 hours  
**Target:** M-class or stronger flares
    """)
    
    st.markdown("---")
    
    if st.button("🔄 Manual Refresh", use_container_width=True):
        st.rerun()

# =========================
# AUTO REFRESH (USING PROPER METHOD)
# =========================
# Auto-refresh every 30 seconds using built-in component
st_autorefresh(interval=REFRESH_INTERVAL_MS, key="astropredict_refresh")

# =========================
# HEADER
# =========================
st.title("☀️ AstroPredict: Solar Flare Forecasting System")
st.markdown("##### Phase-1 Real-Time Demonstration")

# =========================
# DATA FETCH
# =========================
@st.cache_data(ttl=10)  # Cache for 10 seconds
def fetch_prediction():
    try:
        response = requests.get(API_URL, timeout=5)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API returned status {response.status_code}"
    except requests.Timeout:
        return None, "Connection timeout - check if backend is running"
    except requests.ConnectionError:
        return None, "Cannot connect to backend"
    except Exception as e:
        return None, str(e)

data, error = fetch_prediction()

if error:
    st.error(f"⚠️ **Backend Connection Error**")
    st.code(f"Error: {error}\n\nPlease ensure the API is running:\nuvicorn main:app --reload", language="bash")
    st.stop()

if not data:
    st.error("⚠️ No data received from API")
    st.stop()

# =========================
# PARSE DATA
# =========================
flux = data.get("goes_flux", 0.0)
state = data.get("activity_state", "UNKNOWN")
ensemble = data["probabilities"]["ensemble"]
lstm_val = data["probabilities"]["lstm"]
bilstm_val = data["probabilities"]["bilstm"]
timestamp = data["timestamp"]

# Extract date and time
try:
    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    utc_date = dt.strftime("%Y-%m-%d")
    utc_time = dt.strftime("%H:%M:%S")
    forecast_end = dt + timedelta(hours=24)
    forecast_end_str = forecast_end.strftime("%Y-%m-%d %H:%M:%S UTC")
except:
    utc_date = "Unknown"
    utc_time = "Unknown"
    forecast_end_str = "Unknown"

# =========================
# RISK CLASSIFICATION
# =========================
def get_risk_info(probability):
    if probability < 0.25:
        return {
            "level": "LOW",
            "class": "risk-low",
            "color": "#10b981",
            "emoji": "🟢",
            "description": "Solar conditions are stable. No significant threat detected."
        }
    elif probability < 0.50:
        return {
            "level": "MODERATE",
            "class": "risk-moderate",
            "color": "#f59e0b",
            "emoji": "🟡",
            "description": "Significant flare potential detected. Enhanced situational awareness required."
        }
    elif probability < 0.75:
        return {
            "level": "ELEVATED",
            "class": "risk-elevated",
            "color": "#fb923c",
            "emoji": "🟠",
            "description": "High probability of major flare event. Preparedness awareness advised."
        }
    else:
        return {
            "level": "HIGH",
            "class": "risk-high",
            "color": "#ef4444",
            "emoji": "🔴",
            "description": "High probability of major flare event. Immediate preparation advised."
        }

risk_info = get_risk_info(ensemble)

# =========================
# PRIMARY FORECAST DISPLAY
# =========================
st.markdown("---")
st.markdown("## 🎯 Current 24-Hour Forecast")

# Main risk display
st.markdown(f"""
<div class="risk-card {risk_info['class']}">
    <h3>{risk_info['emoji']} {risk_info['level']} RISK</h3>
    <h1>{ensemble*100:.0f}%</h1>
    <p style="font-size: 18px; margin-top: 10px;">Probability of M-class or stronger flare</p>
</div>
""", unsafe_allow_html=True)

# Forecast details
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-box">
        <h4>📅 Forecast Window</h4>
        <p><strong>Valid From:</strong> {start}</p>
        <p><strong>Valid Until:</strong> {end}</p>
        <p style="margin-top: 10px; color: #94a3b8;">
            <em>Flare may occur at any time in this 24-hour period</em>
        </p>
    </div>
    """.format(start=f"{utc_date} {utc_time} UTC", end=forecast_end_str), 
    unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="info-box">
        <h4>🧠 Model Assessment</h4>
        <p>{risk_info['description']}</p>
        <p style="margin-top: 10px;">
            <strong>Confidence:</strong> Ensemble of 2 models<br>
            <strong>Threshold:</strong> ≥M-class flares
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-box">
        <h4>🎬 Potential Stakeholders (Conceptual)</h4>
        <ul style="text-align: left; color: #94a3b8;">
            <li>Power grid control centers</li>
            <li>Satellite operators (ISRO)</li>
            <li>Aviation authorities (DGCA)</li>
            <li>Telecom network managers</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# =========================
# UNDERSTANDING THE FORECAST
# =========================
st.markdown("---")
st.markdown("## 📖 Understanding This Forecast")

explain_col1, explain_col2 = st.columns(2)

with explain_col1:
    st.markdown(f"""
    <div class="info-box">
        <h4>❓ What does "{int(ensemble*100)}%" mean?</h4>
        <p style="line-height: 1.8;">
        This is a <strong>probability over a time window</strong>, not a prediction of exact timing.
        </p>
        <ul style="line-height: 1.8; color: #94a3b8;">
            <li>✅ A flare <strong>may occur</strong> anytime in the next 24 hours</li>
            <li>❌ We <strong>cannot</strong> predict the exact minute</li>
            <li>✅ Higher probability = Higher likelihood</li>
            <li>✅ Think: "60% chance of rain today"</li>
        </ul>
        <p style="margin-top: 10px; font-style: italic; color: #94a3b8;">
        Just like weather forecasting, we predict the <em>likelihood</em> over a period, 
        not the <em>exact moment</em> of occurrence.
        </p>
    </div>
    """, unsafe_allow_html=True)

with explain_col2:
    st.markdown("""
    <div class="info-box">
        <h4>🔬 How This Forecast Works</h4>
        <p style="line-height: 1.8;"><strong>Step 1:</strong> Fetch live X-ray flux from NOAA satellites</p>
        <p style="line-height: 1.8;"><strong>Step 2:</strong> Match current conditions to historical magnetic patterns</p>
        <p style="line-height: 1.8;"><strong>Step 3:</strong> AI models analyze 6-hour magnetic evolution</p>
        <p style="line-height: 1.8;"><strong>Step 4:</strong> Ensemble combines predictions for reliability</p>
        <p style="margin-top: 15px; padding: 10px; background-color: #0f172a; border-radius: 5px;">
        <strong>Training Data:</strong> 1.3M samples from 309 active regions (2010-2025)
        </p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# CURRENT SOLAR CONTEXT
# =========================
st.markdown("---")
st.markdown("## 🌞 Current Solar Observations")

solar_col1, solar_col2, solar_col3, solar_col4 = st.columns(4)

with solar_col1:
    st.markdown(f"""
    <div class="metric-card">
        <p>⚡ X-Ray Flux</p>
        <h2>{flux:.2e}</h2>
        <p>W/m² (GOES-16)</p>
    </div>
    """, unsafe_allow_html=True)

with solar_col2:
    state_color = "#10b981" if state == "QUIET" else "#f59e0b"
    st.markdown(f"""
    <div class="metric-card">
        <p>🌍 Activity State</p>
        <h2 style="color: {state_color};">{state}</h2>
        <p>Current classification</p>
    </div>
    """, unsafe_allow_html=True)

with solar_col3:
    st.markdown(f"""
    <div class="metric-card">
        <p>📡 Data Source</p>
        <h2>NOAA</h2>
        <p>GOES-16 Satellite</p>
    </div>
    """, unsafe_allow_html=True)

with solar_col4:
    st.markdown(f"""
    <div class="metric-card">
        <p>🕐 Observation Time</p>
        <h2>{utc_time}</h2>
        <p>UTC (GMT)</p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# MODEL CONSENSUS
# =========================
st.markdown("---")
st.markdown("## 🧠 AI Model Consensus Analysis")

model_col1, model_col2 = st.columns([3, 2])

with model_col1:
    # Calculate model agreement
    model_spread = abs(lstm_val - bilstm_val)
    
    if model_spread < 0.15:
        agreement = "HIGH AGREEMENT"
        agreement_color = "green"
        agreement_text = "✅ Both models see similar patterns. Forecast is reliable."
    elif model_spread < 0.30:
        agreement = "MODERATE AGREEMENT"
        agreement_color = "orange"
        agreement_text = "⚠️ Models show some divergence. Monitor situation closely."
    else:
        agreement = "LOW AGREEMENT"
        agreement_color = "red"
        agreement_text = "❌ Significant disagreement. High uncertainty in forecast."
    
    st.markdown(f"### Model Consensus: :{agreement_color}[{agreement}]")
    st.caption(agreement_text)
    st.markdown(f"**Spread:** {model_spread*100:.1f}% difference between models")
    
    st.markdown("---")
    
    # LSTM
    st.markdown(f"**LSTM Model:** {lstm_val*100:.1f}%")
    st.progress(float(lstm_val))
    st.caption("↗️ Long Short-Term Memory - Analyzes gradual buildup in magnetic patterns")
    
    st.markdown("")
    
    # BiLSTM
    st.markdown(f"**BiLSTM Model:** {bilstm_val*100:.1f}%")
    st.progress(float(bilstm_val))
    st.caption("↔️ Bidirectional LSTM - Captures forward and backward temporal context")
    
    st.markdown("---")
    
    # Ensemble
    st.markdown(f"**Ensemble Decision:** {ensemble*100:.1f}%")
    st.progress(float(ensemble))
    st.caption("🎯 Equal-weight average (reduces false alarms)")

with model_col2:
    # Risk gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ensemble * 100,
        title={"text": "Risk Index", "font": {"size": 18, "color": "white"}},
        number={"suffix": "%", "font": {"size": 48, "color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white"},
            "bar": {"color": risk_info['color']},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 2,
            "bordercolor": "#475569",
            "steps": [
                {"range": [0, 25], "color": "#1e293b"},
                {"range": [25, 50], "color": "#334155"},
                {"range": [50, 75], "color": "#475569"},
                {"range": [75, 100], "color": "#64748b"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.8,
                "value": 75
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk level legend
    st.markdown("""
    <div style="font-size: 12px; line-height: 1.8;">
    <strong>Risk Levels:</strong><br>
    🟢 LOW (0-25%)<br>
    🟡 MODERATE (25-50%)<br>
    🟠 ELEVATED (50-75%)<br>
    🔴 HIGH (75-100%)
    </div>
    """, unsafe_allow_html=True)

# =========================
# TECHNICAL DETAILS (EXPANDABLE)
# =========================
with st.expander("🔧 Technical Details & Model Architecture"):
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.markdown("### Data Pipeline")
        st.code("""
Component          | Details
-------------------|------------------
X-ray Flux         | NOAA GOES-16 (1-min cadence)
Magnetic Fields    | SHARP parameters (12-min)
Context Matching   | Historical scenario library
Features Used      | 13 magnetic parameters
Lookback Window    | 6 hours (30 timesteps)
        """, language="text")
        
        st.markdown("### Model Performance (Test Set)")
        st.code("""
Model     | ROC-AUC | TSS   
----------|---------|-------
LSTM      | 0.822   | 0.521 
BiLSTM    | 0.835   | 0.516 
Ensemble  | 0.832   | 0.535 
        """, language="text")
    
    with tech_col2:
        st.markdown("### Model Architecture")
        st.code("""
LSTM Model:
- Input: (30, 13) [timesteps, features]
- Layers: LSTM(128) → Dropout(0.3) → Dense(1)
- Activation: Sigmoid


BiLSTM Model:
- Input: (30, 13)
- Layers: Bidirectional(LSTM(128)) → Dropout(0.3) → Dense(1)
- Activation: Sigmoid


Ensemble:
- Method: Equal-weight average
- Formula: (LSTM + BiLSTM) / 2
        """, language="python")
        
        st.markdown("### Key Features")
        st.caption("""
**Magnetic Parameters:**
- USFLUX, TOTUSJH, TOTUSJZ, TOTPOT
- R_VALUE, SAVNCPP, ABSNJZH, MEANALP
- Plus 5 engineered features (derivatives, std)
        """)

# =========================
# SYSTEM STATUS FOOTER
# =========================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("🟢 **System Status:** DEMONSTRATION ACTIVE")
    st.caption(f"Last refresh: {utc_time} UTC")

with footer_col2:
    st.caption(f"🔄 **Auto-refresh:** Every 30 seconds")
    st.caption(f"API: {API_URL}")

with footer_col3:
    st.caption("🎓 **AstroPredict Phase-1**")

# =========================
# SCIENTIFIC NOTE
# =========================
st.info(f"""
**ℹ️ Scientific Note:** {data.get('note', 'Scenario-based demonstration')}
""")