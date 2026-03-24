"""
Enhanced data preprocessing for Phase 2 Bi-LSTM + Attention model.
Uses the NASA Global Landslide Catalog (GLC) with richer feature encoding.

Features per event (18-dim):
  - Landslide type:  7-dim one-hot
  - Trigger:         6-dim one-hot
  - Size:            1-dim normalized [0,1]
  - Fatality count:  1-dim log-scaled
  - Month:           2-dim cyclical (sin, cos)
  - Year:            1-dim normalized

Source: NASA Open Data Portal — 11,033 global landslide events (1970-2019)
"""

import os
import sys
import warnings
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

warnings.filterwarnings("ignore")

GLC_PATH = os.path.join(config.EXCEL_DIR, "nasa_glc.csv")

TYPE_MAP = {
    "landslide": "Landslide", "debris_flow": "Debris Flow",
    "mudslide": "Mudslide", "rock_fall": "Rockfall",
    "earth_flow": "Earth Flow", "translational_slide": "Translational Slide",
    "complex": "Complex Landslide", "lahar": "Mudslide",
    "creep": "Landslide", "topple": "Rockfall",
    "snow_avalanche": "Rockfall", "riverbank_collapse": "Landslide",
    "other": "Landslide", "unknown": "Landslide",
}

TRIGGER_MAP = {
    "downpour": "Heavy Rain", "rain": "Heavy Rain",
    "continuous_rain": "Heavy Rain", "monsoon": "Heavy Rain",
    "tropical_cyclone": "Storm", "snowfall_snowmelt": "Snowmelt",
    "freeze_thaw": "Snowmelt", "earthquake": "Earthquake",
    "mining": "Human Activity", "construction": "Human Activity",
    "flooding": "Heavy Rain", "dam_embankment_collapse": "Human Activity",
    "no_apparent_trigger": "Unknown", "unknown": "Unknown", "other": "Unknown",
}

SIZE_ORDER = {"small": 0, "medium": 1, "large": 2, "very_large": 3, "catastrophic": 4, "unknown": 1}

# Fixed number of feature dimensions
N_TYPES = 7
N_TRIGGERS = 6
FEATURE_DIM = N_TYPES + N_TRIGGERS + 1 + 1 + 2 + 1  # = 18


def load_glc_data(filepath=None):
    """Load and clean the NASA GLC dataset."""
    if filepath is None:
        filepath = GLC_PATH
    df = pd.read_csv(filepath, low_memory=False)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date", "landslide_category", "country_name"])
    df = df[df["landslide_category"].str.strip() != ""].copy()
    df["year"] = df["event_date"].dt.year.astype(int)
    df["month"] = df["event_date"].dt.month.astype(int)
    df["fatality_count"] = pd.to_numeric(df["fatality_count"], errors="coerce").fillna(0).astype(int)
    df["landslide_size"] = df["landslide_size"].fillna("unknown").str.lower().str.strip()
    df["landslide_trigger"] = df["landslide_trigger"].fillna("unknown").str.lower().str.strip()
    print(f"Loaded {len(df)} records | {df['year'].min()}-{df['year'].max()} | "
          f"{df['country_name'].nunique()} countries")
    return df


def encode_features(df):
    """Encode categorical features and return enriched dataframe + encoders."""
    df = df.copy()
    df["ls_type"] = df["landslide_category"].str.lower().str.strip().map(TYPE_MAP).fillna("Landslide")
    df["ls_trigger"] = df["landslide_trigger"].str.lower().map(TRIGGER_MAP).fillna("Unknown")
    df["size_code"] = df["landslide_size"].map(SIZE_ORDER).fillna(1).astype(int)

    type_encoder = LabelEncoder()
    df["type_code"] = type_encoder.fit_transform(df["ls_type"])

    trigger_encoder = LabelEncoder()
    df["trigger_code"] = trigger_encoder.fit_transform(df["ls_trigger"])

    print(f"\nLandslide types ({len(type_encoder.classes_)}):")
    for i, c in enumerate(type_encoder.classes_):
        print(f"  {i}: {c:25s}  ({(df['ls_type'] == c).sum():5d} events)")

    return df, type_encoder, trigger_encoder


def event_to_feature_vector(row, type_encoder, trigger_encoder, year_min, year_max):
    """
    Convert a single event row to an 18-dim feature vector.

    Features:
      [0:7]   type one-hot (7 dim)
      [7:13]  trigger one-hot (6 dim)
      [13]    size normalized (1 dim)
      [14]    log fatality count (1 dim)
      [15]    month sin (1 dim) — cyclical encoding
      [16]    month cos (1 dim) — cyclical encoding
      [17]    year normalized (1 dim)
    """
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)

    # Type one-hot
    type_idx = row["type_code"]
    if 0 <= type_idx < N_TYPES:
        vec[type_idx] = 1.0

    # Trigger one-hot
    trig_idx = row["trigger_code"]
    if 0 <= trig_idx < N_TRIGGERS:
        vec[N_TYPES + trig_idx] = 1.0

    # Size normalized [0, 1]
    vec[13] = row["size_code"] / 4.0

    # Log fatality count
    vec[14] = math.log1p(row["fatality_count"]) / 10.0  # normalize roughly to [0, 1]

    # Month cyclical encoding
    month = row["month"]
    vec[15] = math.sin(2 * math.pi * month / 12.0)
    vec[16] = math.cos(2 * math.pi * month / 12.0)

    # Year normalized
    year_range = max(year_max - year_min, 1)
    vec[17] = (row["year"] - year_min) / year_range

    return vec


def build_sequences(df, type_encoder, trigger_encoder, min_seq_len=3, max_seq_len=200):
    """
    Build per-country temporal sequences of feature vectors for LSTM training.

    Each sequence is a time-ordered list of events for one country.
    Training target: given events [1..t-1], predict type of event t.

    Args:
        df:             Encoded dataframe.
        type_encoder:   Fitted LabelEncoder for types.
        trigger_encoder: Fitted LabelEncoder for triggers.
        min_seq_len:    Minimum events per country to include.
        max_seq_len:    Maximum sequence length (truncate older events).

    Returns:
        sequences: list of (features_array, target_types, country_name)
                   features_array: (T, 18) numpy array
                   target_types:   (T,) numpy array of type codes
    """
    df_sorted = df.sort_values(["country_name", "year", "month"])
    year_min = int(df["year"].min())
    year_max = int(df["year"].max())

    sequences = []
    for country, grp in df_sorted.groupby("country_name"):
        if len(grp) < min_seq_len:
            continue

        # Convert each event to feature vector
        events = grp.tail(max_seq_len)  # Keep most recent events
        features = []
        types = []
        for _, row in events.iterrows():
            vec = event_to_feature_vector(row, type_encoder, trigger_encoder, year_min, year_max)
            features.append(vec)
            types.append(int(row["type_code"]))

        features = np.array(features, dtype=np.float32)
        types = np.array(types, dtype=np.int64)
        sequences.append((features, types, country))

    total_events = sum(len(s[1]) for s in sequences)
    print(f"\nBuilt {len(sequences)} country sequences | "
          f"Total events: {total_events} | "
          f"Seq lengths: {min(len(s[1]) for s in sequences)}-{max(len(s[1]) for s in sequences)}")

    return sequences


def get_country_risk_profile(df):
    """Per-country stats for risk assessment."""
    rows = []
    for country, grp in df.groupby("country_name"):
        year_span = max(1, grp["year"].max() - grp["year"].min() + 1)
        rows.append({
            "country":          country,
            "event_count":      len(grp),
            "avg_fatalities":   round(grp["fatality_count"].mean(), 2),
            "dominant_type":    grp["ls_type"].mode()[0],
            "dominant_trigger": grp["ls_trigger"].mode()[0],
            "events_per_year":  round(len(grp) / year_span, 2),
            "peak_month":       int(grp["month"].mode()[0]),
            "year_span":        year_span,
        })
    return pd.DataFrame(rows).set_index("country").sort_values("event_count", ascending=False)


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class LandslideTemporalDataset(Dataset):
    """
    PyTorch Dataset for Bi-LSTM training.

    For each sequence of T events:
      Input:  features[0:T-1]  (past events)
      Target: types[1:T]       (next event type at each step)
      Occurrence: 1.0 if any landslide in next step, else 0.0
    """

    def __init__(self, sequences, n_forecast_steps=3):
        self.samples = []
        self.n_forecast_steps = n_forecast_steps

        for features, types, country in sequences:
            if len(features) < 3:
                continue

            # Input: all events except last
            input_feats = features[:-1]        # (T-1, 18)
            # Target type: the next event type at each step
            target_type = types[-1]             # scalar: final event type
            # Target occurrence: 1.0 (there IS a next event in the catalog)
            target_occ = 1.0

            # Forecast targets: last n_forecast_steps types
            n = min(self.n_forecast_steps, len(types))
            forecast_targets = types[-n:]  # (n,)
            # Pad if needed
            if len(forecast_targets) < self.n_forecast_steps:
                pad = np.full(self.n_forecast_steps - len(forecast_targets), types[-1], dtype=np.int64)
                forecast_targets = np.concatenate([forecast_targets, pad])

            self.samples.append({
                "features": input_feats,
                "target_type": target_type,
                "target_occ": target_occ,
                "forecast_targets": forecast_targets,
                "country": country,
                "seq_len": len(input_feats),
            })

        print(f"Dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "features": torch.tensor(s["features"], dtype=torch.float32),
            "target_type": torch.tensor(s["target_type"], dtype=torch.long),
            "target_occ": torch.tensor(s["target_occ"], dtype=torch.float32),
            "forecast_targets": torch.tensor(s["forecast_targets"], dtype=torch.long),
            "seq_len": s["seq_len"],
        }


def collate_fn(batch):
    """
    Custom collate: pad sequences to same length within a batch.
    """
    max_len = max(item["seq_len"] for item in batch)

    features_padded = []
    target_types = []
    target_occs = []
    forecast_targets = []
    lengths = []

    for item in batch:
        feat = item["features"]
        seq_len = item["seq_len"]

        # Pad to max_len
        if seq_len < max_len:
            pad = torch.zeros(max_len - seq_len, feat.shape[1])
            feat = torch.cat([feat, pad], dim=0)

        features_padded.append(feat)
        target_types.append(item["target_type"])
        target_occs.append(item["target_occ"])
        forecast_targets.append(item["forecast_targets"])
        lengths.append(seq_len)

    return {
        "features": torch.stack(features_padded),           # (B, T, 18)
        "target_type": torch.stack(target_types),            # (B,)
        "target_occ": torch.stack(target_occs),              # (B,)
        "forecast_targets": torch.stack(forecast_targets),   # (B, n_steps)
        "lengths": torch.tensor(lengths, dtype=torch.long),  # (B,)
    }


# ─── Legacy compatibility ────────────────────────────────────────────────────
# These functions maintain the same interface as phase2_hmm/data_preprocessing.py

def build_observation_sequences(df, group_by="country_name", use_combined=False):
    """Legacy HMM-compatible interface (used by pipeline for backward compat)."""
    df_sorted = df.sort_values([group_by, "year", "month"])
    obs_sequences, lengths, group_labels = [], [], []

    if use_combined:
        n_triggers = df["trigger_code"].max() + 1
        df_sorted = df_sorted.copy()
        df_sorted["obs_code"] = (
            df_sorted["type_code"] * int(n_triggers) + df_sorted["trigger_code"]
        ).astype(int)
        obs_col = "obs_code"
    else:
        obs_col = "type_code"

    for name, grp in df_sorted.groupby(group_by):
        seq = grp[obs_col].values.astype(int)
        if len(seq) >= 2:
            obs_sequences.append(seq)
            lengths.append(len(seq))
            group_labels.append(name)

    return obs_sequences, lengths, group_labels
