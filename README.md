# ☀️ AstroPredict: AI-Based Solar Flare Early Warning System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://astropredict2026.streamlit.app/)
[![API Status](https://img.shields.io/badge/API-Render-46E3B7?style=for-the-badge&logo=render)](https://astropredict-api.onrender.com/health)
[![GitHub](https://img.shields.io/badge/Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/Asharu369/AstroPredict)

> **Integrated M.Tech Artificial Intelligence Thesis — VIT Bhopal University, 2025–2026**
> Student: Muhammed Asharudheen A (21MIM10034) | Supervisor: Dr. Abdul Rahman

---

## 🔴 Live Demo

👉 **[Open AstroPredict Dashboard](https://astropredict2026.streamlit.app/)**
🔗 **[Backend API](https://astropredict-api.onrender.com/health)**

The dashboard shows:
- Real-time GOES X-ray flux from NOAA GOES-16 satellite
- AI-based 24-hour probabilistic solar flare forecast (≥M-class)
- LSTM vs BiLSTM model consensus analysis
- Risk classification: LOW / MODERATE / ELEVATED / HIGH

---

## 📌 Project Summary

AstroPredict is a research-oriented prototype for probabilistic forecasting of ≥M-class solar flares using temporally-aware machine learning models trained on NASA SHARP (Space-weather HMI Active Region Patch) magnetic field parameters.

**Core contributions:**
- Strict HARPNUM-level train/test isolation — zero temporal data leakage
- Explicit quantification of temporal gain: LSTM achieves +127% TSS over static XGBoost baseline
- Multi-metric evaluation across 7 standard space weather forecasting metrics
- End-to-end modular pipeline: data ingestion → training → evaluation → live deployment
- Honest reporting under realistic rare-event conditions

> This project is a Master's thesis research prototype. It is not an operational warning authority.

---

## 📊 Problem Context

Solar flares disrupt:
- Satellite communication systems
- Aviation communication and navigation
- GPS infrastructure accuracy
- Power grid stability

Reliable prediction is difficult due to extreme class imbalance, nonlinear magnetic field evolution, temporal data leakage in prior studies, and 24–48 hour SHARP magnetogram publication latency. AstroPredict addresses all four through strict region-level validation and sequence-based temporal modeling.

---

## 🧠 Dataset & Experimental Design

| Parameter | Value |
|-----------|-------|
| Total observations | ~1.36 million temporal samples |
| Solar Active Regions | 309 unique HARPNUMs |
| Training set | 247 HARPNUMs |
| Test set | 62 HARPNUMs (zero overlap with training) |
| Test samples | 264,163 |
| Forecast horizon | 24 hours |
| Target class | ≥M-class solar flare |
| Data source | NASA SHARP via Zenodo (2010–2024) |

Strict HARPNUM-level separation prevents any temporal leakage between training and test data — a methodological weakness identified in several prior published studies.

---

## ⚙️ Feature Engineering

**Base SHARP magnetic parameters:**
USFLUX, TOTUSJH, TOTUSJZ, TOTPOT, R_VALUE, SAVNCPP, ABSNJZH, MEANALP

**Temporal engineering:**
- 6-hour lookback window (30 timesteps × 12-minute cadence)
- Delta features (first-order temporal differences)
- Rolling mean statistics
- Rolling standard deviation

**Final input shape:** `(30 timesteps × 13 features)` per sequence

---

## 🤖 Models

| Model | Type | Description |
|-------|------|-------------|
| XGBoost | Static baseline | Tabular snapshot-based gradient boosting |
| LSTM | Temporal | 128-unit LSTM → Dropout(0.3) → Dense(1, sigmoid) |
| BiLSTM | Bidirectional temporal | Bidirectional(LSTM(128)) → Dropout(0.3) → Dense(1, sigmoid) |
| Ensemble | Combined | Equal-weight average of LSTM + BiLSTM outputs |

**Evaluation metrics:** Accuracy, Precision, Recall, F1, TSS, HSS, ROC-AUC
*(TSS is primary metric due to class imbalance)*

---

## 📈 Final Verified Results

*Held-out test set: 264,163 samples, 62 HARPNUMs, zero overlap with training*

| Model | TSS | ROC-AUC | Recall | F1 |
|-------|-----|---------|--------|----|
| XGBoost (baseline) | 0.179 | 0.624 | 0.535 | 0.677 |
| LSTM | 0.406 | 0.776 | 0.769 | 0.756 |
| BiLSTM | 0.448 | 0.805 | 0.767 | 0.779 |
| Ensemble | 0.444 | 0.797 | 0.785 | 0.773 |

**Key finding:** Temporal modeling improves TSS by ~127% over static baseline, confirming that magnetic field *evolution* carries predictive information beyond single-timestep snapshots. The SHARP-only feature ceiling (TSS 0.40–0.55) is consistent with published literature under leakage-free evaluation.

---

## 🏗️ System Architecture

```
NOAA GOES-16 (Live X-ray flux)
        ↓
Activity State Classification (QUIET / ACTIVE)
        ↓
SHARP Context Library (Historical magnetic scenarios)
        ↓
┌─────────────────┐    ┌──────────────────┐
│   LSTM Model    │    │   BiLSTM Model   │
│   TSS: 0.406    │    │   TSS: 0.448     │
│   AUC:  0.776   │    │   AUC:  0.805    │
└─────────────────┘    └──────────────────┘
           ↓                    ↓
      Equal-Weight Ensemble (TSS: 0.444)
           ↓
  24-Hour Flare Probability Output
```

**Backend (Render.com):** FastAPI inference service
- `GET /health` — system status
- `GET /predict/now` — live probabilistic forecast

**Frontend (Streamlit Cloud):** Interactive dashboard
- Live risk display with gauge visualization
- Model consensus analysis
- Solar observation metrics

---

## 🔬 How the Forecast Works

1. Fetch live GOES X-ray flux from NOAA GOES-16
2. Classify current solar activity state (QUIET / ACTIVE)
3. Retrieve context-matched historical SHARP magnetic sequence
4. Evaluate 6-hour magnetic evolution using LSTM & BiLSTM
5. Combine model outputs via equal-weight ensemble
6. Output 24-hour probabilistic ≥M-class flare risk

> **Note on real-time design:** Due to SHARP data publication latency (24–48 hours), the system retrieves context-matched historical magnetic sequences for inference. This is a scientifically documented and justified design choice, explicitly discussed in the thesis.

---

## 📁 Repository Structure

```
AstroPredict/
├── notebooks/
│   ├── 00_SHARP_Context_Builder.ipynb
│   ├── 01_Data_Preparation.ipynb
│   ├── 02_Model_Training.ipynb
│   ├── 03_Evaluation_Results.ipynb
│   ├── 04_RealTime_Pipeline.ipynb
│   └── 05_Interface_Demo.ipynb
├── models/
│   ├── lstm_phase1_model.keras
│   ├── bilstm_phase1_model.keras
│   ├── sharp_context.pkl
│   └── README.md
├── dashboard.py          ← Streamlit frontend
├── main.py               ← FastAPI backend
├── requirements.txt
└── .gitignore
```

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/Asharu369/AstroPredict.git
cd AstroPredict

# Install dependencies
pip install -r requirements.txt

# Terminal 1 — Start API backend
python main.py

# Terminal 2 — Start dashboard
streamlit run dashboard.py
```

Open `http://localhost:8501` in your browser.

---

## 📡 Data Sources

| Source | Description | Access |
|--------|-------------|--------|
| NASA SHARP (Zenodo) | Historical SHARP magnetic parameters (2010–2024) | [zenodo.org/record/7416899](https://zenodo.org/record/7416899) |
| NOAA GOES-16 | Live X-ray flux (1-minute cadence) | [swpc.noaa.gov](https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json) |

---

## 🎓 Academic Context

| Field | Detail |
|-------|--------|
| Program | Integrated M.Tech — Artificial Intelligence |
| Institution | VIT Bhopal University |
| School | SCAI — School of Computing Science, Engineering and AI |
| Academic Year | 2025–2026 |
| Student | Muhammed Asharudheen A (21MIM10034) |
| Supervisor | Dr. Abdul Rahman |
| Thesis Title | AstroPredict: AI-Based Solar Flare Early Warning System |

