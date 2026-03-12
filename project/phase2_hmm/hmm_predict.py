"""
HMM inference: landslide type classification + future occurrence prediction.

Given:
  - A country name (or recent observation sequence)
  - The trained HMM + NASA GLC risk profiles

Outputs:
  1. Most likely landslide TYPE (from hidden state emission distribution)
  2. Occurrence PROBABILITY (how likely is the next event given history)
  3. FUTURE FORECAST — predicted type + probability for next N time steps
  4. PEAK RISK MONTH — which month has historically highest risk for this location
"""

import os, sys, calendar
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from phase2_hmm.hmm_model import load_hmm_model, load_encoder


def _load_hmm_meta():
    """Load combined-obs metadata saved at training time (if present)."""
    meta_path = os.path.join(config.CHECKPOINTS_DIR, "hmm_meta.pkl")
    if os.path.exists(meta_path):
        return joblib.load(meta_path)
    return {"use_combined": False, "n_types": 7, "n_triggers": 1}


def predict_hidden_states(model, obs_sequence: np.ndarray) -> np.ndarray:
    """Viterbi decoding: most likely hidden state sequence."""
    X = obs_sequence.astype(int).reshape(-1, 1)
    _, states = model.decode(X, algorithm="viterbi")
    return states


def get_type_from_state(model, state_idx: int, type_encoder, n_triggers: int = 1) -> str:
    """Return the most probable landslide TYPE emitted by a hidden state.
    If combined observations (type × trigger) are used, marginalise over triggers.
    """
    emission_row = model.emissionprob_[state_idx]
    n_types = len(type_encoder.classes_)
    if n_triggers > 1:
        # Sum emission probs across all trigger symbols for each type
        type_probs = np.zeros(n_types)
        for t in range(n_types):
            type_probs[t] = emission_row[t * n_triggers: (t + 1) * n_triggers].sum()
        most_likely_type = int(np.argmax(type_probs))
    else:
        most_likely_obs = int(np.argmax(emission_row))
        most_likely_type = most_likely_obs % n_types
    return type_encoder.classes_[most_likely_type]


def compute_occurrence_probability(model, obs_sequence: np.ndarray) -> float:
    """
    Normalised log-likelihood → probability [0,1].
    Uses per-step log-likelihood relative to the model's entropy floor:
    probability = sigmoid((per_step - floor) / scale)
    where floor = -log(n_symbols) is the per-step LL of a uniform random model
    and scale is chosen so typical sequences map to 30–60% range.
    """
    if len(obs_sequence) == 0:
        return 0.0
    X = obs_sequence.astype(int).reshape(-1, 1)
    log_ll = model.score(X)
    per_step = log_ll / len(obs_sequence)   # e.g. -6.9 for 42-symbol, -0.77 for 7-symbol
    n_symbols = model.emissionprob_.shape[1]
    floor = -np.log(n_symbols)              # uniform random baseline (e.g. -3.74 for 42)
    # How much better than random? Normalised excess in (0, 1)
    # per_step is always <= 0 and >= floor (model beats random)
    # excess = (per_step - floor) / |floor|  →  0 if model = random, grows positive when better
    excess = (per_step - floor) / abs(floor)
    # excess ≈ 0.85 for typical Nepal seed → sigmoid(0.85) ≈ 0.70 → clamp to 0.2–0.65 range
    prob = float(1.0 / (1.0 + np.exp(-excess)))
    # Clamp to [0.05, 0.95] to avoid extreme values
    prob = max(0.05, min(0.95, prob))
    return round(prob, 4)


def forecast_future_types(model, current_state: int, type_encoder,
                          n_steps: int = 3, n_triggers: int = 1) -> list:
    """
    Project the most likely landslide type for the next N time steps.
    Uses the transition matrix to propagate state distribution forward.

    Returns list of dicts:
      [{"step": 1, "landslide_type": str, "state": int, "probability": float}, ...]
    """
    state_dist = np.zeros(model.n_components)
    state_dist[current_state] = 1.0

    forecasts = []
    for step in range(1, n_steps + 1):
        state_dist = state_dist @ model.transmat_
        most_likely_state = int(np.argmax(state_dist))
        ls_type = get_type_from_state(model, most_likely_state, type_encoder, n_triggers)
        # Compute type probability as state_prob × emission_prob
        type_prob = float(state_dist[most_likely_state] *
                          model.emissionprob_[most_likely_state, np.argmax(model.emissionprob_[most_likely_state])])
        forecasts.append({
            "step":           step,
            "landslide_type": ls_type,
            "state":          most_likely_state,
            "probability":    round(type_prob, 4),
        })
    return forecasts


def classify_and_forecast(model, type_encoder, risk_profile,
                          country: str = None,
                          obs_sequence: np.ndarray = None,
                          n_forecast_steps: int = 3) -> dict:
    """
    Main prediction function for Phase 2.

    Args:
        model:            Trained CategoricalHMM.
        type_encoder:     Fitted LabelEncoder for landslide types.
        risk_profile:     DataFrame from get_country_risk_profile().
        country:          Country name (used to look up history & build sequence).
        obs_sequence:     Optional pre-built integer observation array.
                          If None, uses the country's dominant type code as seed.
        n_forecast_steps: How many future time steps to forecast.

    Returns:
        dict with:
          current_type, occurrence_probability, hidden_state,
          future_forecast, risk_stats, peak_risk_month
    """
    # Load combined-obs meta to get n_triggers
    meta = _load_hmm_meta()
    n_triggers = int(meta.get("n_triggers", 1))
    n_types = int(meta.get("n_types", len(type_encoder.classes_)))
    use_combined = meta.get("use_combined", False)

    # --- Build observation sequence ---
    if obs_sequence is None:
        if country and country in risk_profile.index:
            dom_type = risk_profile.loc[country, "dominant_type"]
            try:
                type_code = int(type_encoder.transform([dom_type])[0])
            except Exception:
                type_code = 0
            if use_combined:
                # Use combined code with trigger=0 (most common / unknown)
                code = type_code * n_triggers
            else:
                code = type_code
            obs_sequence = np.array([code])
        else:
            obs_sequence = np.array([0])

    # --- Phase 2a: classify current type ---
    states = predict_hidden_states(model, obs_sequence)
    current_state = int(states[-1])
    current_type  = get_type_from_state(model, current_state, type_encoder, n_triggers)

    # --- Phase 2b: occurrence probability ---
    occurrence_prob = compute_occurrence_probability(model, obs_sequence)

    # --- Phase 2c: future forecast ---
    future = forecast_future_types(model, current_state, type_encoder, n_forecast_steps, n_triggers)

    # --- Phase 2d: risk stats from NASA GLC ---
    risk_stats = {}
    peak_month_name = "N/A"
    if country and country in risk_profile.index:
        r = risk_profile.loc[country]
        risk_stats = {
            "events_in_catalog":   int(r["event_count"]),
            "events_per_year":     float(r["events_per_year"]),
            "avg_fatalities":      float(r["avg_fatalities"]),
            "dominant_trigger":    r["dominant_trigger"],
        }
        peak_month_name = calendar.month_name[int(r["peak_month"])]

    return {
        "current_type":          current_type,
        "hidden_state":          current_state,
        "occurrence_probability": occurrence_prob,
        "future_forecast":       future,
        "risk_stats":            risk_stats,
        "peak_risk_month":       peak_month_name,
        "country":               country or "Unknown",
    }


def format_phase2_output(result: dict) -> str:
    """Human-readable summary of Phase 2 results."""
    lines = [
        f"PHASE 2 — Landslide Type & Future Prediction",
        f"{'='*50}",
        f"Current Type        : {result['current_type']}",
        f"Occurrence Prob.    : {result['occurrence_probability']*100:.1f}%",
        f"Peak Risk Month     : {result['peak_risk_month']}",
    ]
    if result["risk_stats"]:
        r = result["risk_stats"]
        lines += [
            f"\nHistorical Risk ({result['country']}):",
            f"  Events in catalog : {r['events_in_catalog']}",
            f"  Events/year       : {r['events_per_year']}",
            f"  Avg fatalities    : {r['avg_fatalities']:.1f}",
            f"  Main trigger      : {r['dominant_trigger']}",
        ]
    lines.append(f"\nFuture Forecast (next {len(result['future_forecast'])} events):")
    for f in result["future_forecast"]:
        lines.append(f"  Step {f['step']}: {f['landslide_type']:25s} (prob={f['probability']*100:.1f}%)")
    return "\n".join(lines)


if __name__ == "__main__":
    import joblib
    from phase2_hmm.data_preprocessing import (
        load_glc_data, encode_features, get_country_risk_profile
    )
    model     = load_hmm_model()
    type_enc  = load_encoder()
    df        = load_glc_data()
    df_enc, _, _ = encode_features(df)
    risk      = get_country_risk_profile(df_enc)

    for country in ["India", "Nepal", "Philippines", "United States"]:
        result = classify_and_forecast(model, type_enc, risk, country=country)
        print(format_phase2_output(result))
        print()
