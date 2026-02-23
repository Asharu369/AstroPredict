# AstroPredict Phase-1 FastAPI Backend
# Scenario-Based Real-Time Solar Flare Prediction

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pandas as pd
import requests
import joblib
import tensorflow as tf
import random
from datetime import datetime, timezone

# =========================
# INITIALIZE APP
# =========================
app = FastAPI(
    title="AstroPredict API",
    description="Scenario-Based Real-Time Solar Flare Prediction System",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# =========================
# LOAD MODELS & CONTEXT
# =========================
@app.on_event("startup")
def load_assets():
    global lstm_model, bilstm_model, context_lib

    lstm_model = tf.keras.models.load_model("models/lstm_phase1_model.keras")
    bilstm_model = tf.keras.models.load_model("models/bilstm_phase1_model.keras")
    context_lib = joblib.load("models/sharp_context.pkl")

    print("✅ Models & context loaded")

# =========================
# SAFE GOES FETCH
# =========================
def fetch_goes_current_safe():
    GOES_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"

    try:
        r = requests.get(GOES_URL, timeout=5)
        r.raise_for_status()
        data = r.json()

        if not data or len(data) == 0:
            raise ValueError("Empty GOES response")

        df = pd.DataFrame(data)
        df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True)
        latest = df.iloc[-1]

        flux = float(latest.get("flux", np.nan))
        ts = latest.get("time_tag", datetime.now(timezone.utc))

        return flux, ts

    except Exception:
        # Safe fallback — demo must NEVER crash
        return np.nan, datetime.now(timezone.utc)


# =========================
# CLASSIFY SOLAR ACTIVITY
# =========================
def classify_activity(flux):
    if np.isnan(flux):
        return "UNKNOWN"
    elif flux < 1e-6:
        return "QUIET"
    else:
        return "ACTIVE"

# =========================
# CONTEXT RETRIEVAL
# =========================
def get_context_window(state):
    if state == "UNKNOWN":
        state = "QUIET"  # fallback
    return random.choice(context_lib[state])


# =========================
# MODEL INFERENCE
# =========================
def run_models(sharp_window):
    X = np.expand_dims(sharp_window, axis=0)

    lstm_prob = float(lstm_model.predict(X, verbose=0)[0][0])
    bilstm_prob = float(bilstm_model.predict(X, verbose=0)[0][0])

    return {
        "lstm": round(lstm_prob, 4),
        "bilstm": round(bilstm_prob, 4),
        "ensemble": round((lstm_prob + bilstm_prob) / 2, 4)
    }

# =========================
# API RESPONSE MODEL
# =========================
class PredictionResponse(BaseModel):
    timestamp: str
    goes_flux: float
    activity_state: str
    probabilities: dict
    note: str

# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
def health():
    return {"status": "OK"}

# =========================
# REAL-TIME DEMO ENDPOINT
# =========================
@app.get("/predict/now", response_model=PredictionResponse)
def predict_now():
    flux, ts = fetch_goes_current_safe()
    state = classify_activity(flux)
    sharp_window = get_context_window(state)
    probs = run_models(sharp_window)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "goes_flux": flux,
        "activity_state": state,
        "probabilities": probs,
        "note": "Scenario-based real-time simulation using context-matched historical magnetic data"
    }
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)