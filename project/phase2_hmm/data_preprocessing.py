"""
Data preprocessing for Phase 2 HMM using the NASA Global Landslide Catalog (GLC).
Source: NASA Open Data Portal — 11,033 global landslide events (1970-2019)

Landslide types: Landslide, Debris Flow, Mudslide, Rockfall, Earth Flow,
                 Translational Slide, Complex Landslide
"""

import os, sys, warnings
import numpy as np
import pandas as pd
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


def load_glc_data(filepath=None):
    if filepath is None:
        filepath = GLC_PATH
    df = pd.read_csv(filepath, low_memory=False)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date", "landslide_category", "country_name"])
    df = df[df["landslide_category"].str.strip() != ""].copy()
    df["year"]  = df["event_date"].dt.year.astype(int)
    df["month"] = df["event_date"].dt.month.astype(int)
    df["fatality_count"] = pd.to_numeric(df["fatality_count"], errors="coerce").fillna(0).astype(int)
    df["landslide_size"]    = df["landslide_size"].fillna("unknown").str.lower().str.strip()
    df["landslide_trigger"] = df["landslide_trigger"].fillna("unknown").str.lower().str.strip()
    print(f"Loaded {len(df)} records | {df['year'].min()}-{df['year'].max()} | "
          f"{df['country_name'].nunique()} countries | {df['landslide_category'].nunique()} raw categories")
    return df


def encode_features(df):
    df = df.copy()
    df["ls_type"]    = df["landslide_category"].str.lower().str.strip().map(TYPE_MAP).fillna("Landslide")
    df["ls_trigger"] = df["landslide_trigger"].str.lower().map(TRIGGER_MAP).fillna("Unknown")
    df["size_code"]  = df["landslide_size"].map(SIZE_ORDER).fillna(1).astype(int)

    type_encoder = LabelEncoder()
    df["type_code"] = type_encoder.fit_transform(df["ls_type"])

    trigger_encoder = LabelEncoder()
    df["trigger_code"] = trigger_encoder.fit_transform(df["ls_trigger"])

    print(f"\nLandslide types ({len(type_encoder.classes_)}):")
    for i, c in enumerate(type_encoder.classes_):
        print(f"  {i}: {c:25s}  ({(df['ls_type']==c).sum():5d} events)")

    return df, type_encoder, trigger_encoder


def build_observation_sequences(df, group_by="country_name", use_combined=False):
    """
    Build per-country temporal sequences for HMM.

    Args:
        df:            Encoded dataframe (must have type_code + trigger_code columns).
        group_by:      Grouping column (default: country_name).
        use_combined:  If True, observation = type_code * n_triggers + trigger_code
                       (combined multi-feature symbol). If False, uses type_code only.

    Returns:
        (obs_sequences, lengths, group_labels)
    """
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
    total = sum(lengths)
    print(f"\nBuilt {len(obs_sequences)} sequences | Total: {total} obs | "
          f"min={min(lengths)} max={max(lengths)} mean={total/len(lengths):.1f}")
    if use_combined:
        n_types = df["type_code"].max() + 1
        print(f"Combined symbols: {n_types} types × {n_triggers} triggers = {int(n_types * n_triggers)}")
    return obs_sequences, lengths, group_labels


def get_country_risk_profile(df):
    """Per-country stats for future occurrence probability estimates."""
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


if __name__ == "__main__":
    df = load_glc_data()
    df_enc, type_enc, trig_enc = encode_features(df)
    seqs, lengths, labels = build_observation_sequences(df_enc)
    risk = get_country_risk_profile(df_enc)
    print("\nTop 10 high-risk countries:")
    print(risk.head(10).to_string())
