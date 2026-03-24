"""
Bi-LSTM + Attention inference for landslide temporal forecasting.

Given a country name (or observation sequence), predicts:
  1. Most likely landslide TYPE
  2. Occurrence PROBABILITY
  3. FUTURE FORECAST — predicted type + probability for next N steps
  4. ATTENTION WEIGHTS — which past events influenced the prediction most
  5. PEAK RISK MONTH from historical data

Same interface as phase2_hmm/hmm_predict.py for pipeline compatibility.
"""

import os
import sys
import calendar
import math

import numpy as np
import torch
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from phase2_lstm.temporal_model import BiLSTMAttentionModel
from phase2_lstm.data_preprocessing import (
    load_glc_data,
    encode_features,
    build_sequences,
    event_to_feature_vector,
    FEATURE_DIM,
    N_TYPES,
    N_TRIGGERS,
    TYPE_MAP,
    TRIGGER_MAP,
    SIZE_ORDER,
)


def load_model_and_encoders(device=None):
    """
    Load trained Bi-LSTM model and label encoders.

    Returns:
        (model, type_encoder, trigger_encoder, device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint = torch.load(config.LSTM_MODEL_PATH, map_location=device, weights_only=False)
    cfg = checkpoint["model_config"]

    model = BiLSTMAttentionModel(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        n_types=cfg["n_types"],
        n_forecast_steps=cfg["n_forecast_steps"],
        dropout=cfg.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load encoders
    type_encoder = joblib.load(config.LSTM_TYPE_ENCODER_PATH)
    trigger_encoder = joblib.load(config.LSTM_TRIGGER_ENCODER_PATH)

    print(f"Loaded Bi-LSTM+Attention model (val_acc={checkpoint.get('val_acc', 0):.4f})")
    return model, type_encoder, trigger_encoder, device


def build_country_sequence(country, type_encoder, trigger_encoder, risk_profile=None, df=None):
    """
    Build a feature sequence for a given country from NASA GLC data.

    If df is not provided, loads it from disk. Returns the most recent events
    as a feature tensor ready for the model.
    """
    if df is None:
        df = load_glc_data()
        df, _, _ = encode_features(df)

    country_data = df[df["country_name"] == country].sort_values(["year", "month"])

    if len(country_data) == 0:
        # Country not in catalog — create a synthetic seed from risk profile
        if risk_profile is not None and country in risk_profile.index:
            r = risk_profile.loc[country]
            # Create a single dummy event with dominant type
            try:
                type_code = int(type_encoder.transform([r["dominant_type"]])[0])
            except Exception:
                type_code = 0
            vec = np.zeros(FEATURE_DIM, dtype=np.float32)
            vec[type_code] = 1.0  # type one-hot
            vec[13] = 0.25  # medium size
            vec[14] = 0.1   # low fatality
            vec[15] = math.sin(2 * math.pi * r["peak_month"] / 12.0)
            vec[16] = math.cos(2 * math.pi * r["peak_month"] / 12.0)
            vec[17] = 0.8   # recent
            return np.array([vec], dtype=np.float32)
        else:
            # Completely unknown — use generic seed
            vec = np.zeros(FEATURE_DIM, dtype=np.float32)
            vec[0] = 1.0  # "Landslide" type
            return np.array([vec], dtype=np.float32)

    # Build feature vectors for each event
    year_min = int(df["year"].min())
    year_max = int(df["year"].max())

    features = []
    for _, row in country_data.tail(100).iterrows():  # Keep last 100 events
        vec = event_to_feature_vector(row, type_encoder, trigger_encoder, year_min, year_max)
        features.append(vec)

    return np.array(features, dtype=np.float32)


def classify_and_forecast(
    model, type_encoder, risk_profile,
    country=None, obs_sequence=None,
    n_forecast_steps=3,
    trigger_encoder=None, df_encoded=None,
):
    """
    Main prediction function for Phase 2 (Bi-LSTM + Attention).

    Same interface as phase2_hmm.hmm_predict.classify_and_forecast()
    for pipeline compatibility.

    Args:
        model:            Trained BiLSTMAttentionModel.
        type_encoder:     Fitted LabelEncoder for types.
        risk_profile:     DataFrame from get_country_risk_profile().
        country:          Country name.
        obs_sequence:     Optional pre-built feature array (T, 18).
        n_forecast_steps: How many future steps to forecast.
        trigger_encoder:  Fitted LabelEncoder for triggers.
        df_encoded:       Pre-encoded DataFrame (optional, to avoid reloading).

    Returns:
        dict with: current_type, occurrence_probability, future_forecast,
                   risk_stats, peak_risk_month, attention_weights, country
    """
    device = next(model.parameters()).device

    # Build or use provided sequence
    if obs_sequence is not None:
        features = obs_sequence
    else:
        if trigger_encoder is None:
            trigger_encoder = joblib.load(config.LSTM_TRIGGER_ENCODER_PATH)
        features = build_country_sequence(
            country, type_encoder, trigger_encoder,
            risk_profile=risk_profile, df=df_encoded,
        )

    # Convert to tensor
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    lengths_tensor = torch.tensor([len(features)], dtype=torch.long).to(device)

    # Run inference
    type_probs, occ_prob, forecast_probs, attn_weights = model.predict_step(
        features_tensor, lengths_tensor
    )

    # Extract results
    type_probs_np = type_probs[0].cpu().numpy()
    best_type_idx = int(np.argmax(type_probs_np))
    current_type = type_encoder.classes_[best_type_idx]

    occurrence_probability = float(occ_prob[0].cpu().numpy())
    occurrence_probability = max(0.05, min(0.95, occurrence_probability))

    # Build forecast
    forecast_probs_np = forecast_probs[0].cpu().numpy()  # (n_steps, n_types)
    future_forecast = []
    for step in range(forecast_probs_np.shape[0]):
        step_probs = forecast_probs_np[step]
        best_idx = int(np.argmax(step_probs))
        ls_type = type_encoder.classes_[best_idx]
        future_forecast.append({
            "step": step + 1,
            "landslide_type": ls_type,
            "probability": float(step_probs[best_idx]),
            "all_probs": {type_encoder.classes_[i]: float(step_probs[i])
                         for i in range(len(type_encoder.classes_))},
        })

    # Attention weights for interpretability
    attn_np = attn_weights[0].cpu().numpy()  # (num_heads, T, T)

    # Risk stats from NASA GLC
    risk_stats = {}
    peak_month_name = "N/A"
    if country and risk_profile is not None and country in risk_profile.index:
        r = risk_profile.loc[country]
        risk_stats = {
            "events_in_catalog": int(r["event_count"]),
            "events_per_year":   float(r["events_per_year"]),
            "avg_fatalities":    float(r["avg_fatalities"]),
            "dominant_trigger":  r["dominant_trigger"],
        }
        peak_month_name = calendar.month_name[int(r["peak_month"])]

    return {
        "current_type":           current_type,
        "current_type_probs":     {type_encoder.classes_[i]: float(type_probs_np[i])
                                   for i in range(len(type_encoder.classes_))},
        "occurrence_probability": round(occurrence_probability, 4),
        "hidden_state":           best_type_idx,  # Compatibility with HMM interface
        "future_forecast":        future_forecast,
        "risk_stats":             risk_stats,
        "peak_risk_month":        peak_month_name,
        "country":                country or "Unknown",
        "attention_weights":      attn_np,
        "sequence_length":        len(features),
    }


def format_phase2_output(result: dict) -> str:
    """Human-readable summary of Phase 2 Bi-LSTM results."""
    lines = [
        f"PHASE 2 — Bi-LSTM + Attention Temporal Forecast",
        f"{'=' * 55}",
        f"Current Type         : {result['current_type']}",
        f"Occurrence Prob.     : {result['occurrence_probability'] * 100:.1f}%",
        f"Peak Risk Month      : {result['peak_risk_month']}",
        f"Sequence Length       : {result['sequence_length']} past events",
    ]

    # Type probabilities
    lines.append(f"\nType Probabilities:")
    for type_name, prob in sorted(result["current_type_probs"].items(),
                                   key=lambda x: -x[1]):
        marker = " <--" if type_name == result["current_type"] else ""
        lines.append(f"  {type_name:25s}: {prob * 100:.1f}%{marker}")

    if result["risk_stats"]:
        r = result["risk_stats"]
        lines += [
            f"\nHistorical Risk ({result['country']}):",
            f"  Events in catalog  : {r['events_in_catalog']}",
            f"  Events/year        : {r['events_per_year']}",
            f"  Avg fatalities     : {r['avg_fatalities']:.1f}",
            f"  Main trigger       : {r['dominant_trigger']}",
        ]

    lines.append(f"\nFuture Forecast (next {len(result['future_forecast'])} events):")
    for f in result["future_forecast"]:
        lines.append(
            f"  Step {f['step']}: {f['landslide_type']:25s} (prob={f['probability'] * 100:.1f}%)"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    from phase2_lstm.data_preprocessing import get_country_risk_profile

    model, type_enc, trig_enc, device = load_model_and_encoders()

    df = load_glc_data()
    df_enc, _, _ = encode_features(df)
    risk = get_country_risk_profile(df_enc)

    for country in ["India", "Nepal", "Philippines", "United States"]:
        print(f"\n{'=' * 55}")
        result = classify_and_forecast(
            model, type_enc, risk,
            country=country,
            trigger_encoder=trig_enc,
            df_encoded=df_enc,
        )
        print(format_phase2_output(result))
