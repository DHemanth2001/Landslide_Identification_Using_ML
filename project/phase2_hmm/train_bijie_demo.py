"""
Quick demo: filter NASA GLC to China / Bijie and run a lightweight HMM demo.

Lightweight version - NO numpy/pandas/sklearn/hmmlearn required.
Uses only built-in Python libraries.

Run as:
    python train_bijie_demo.py --region Bijie --mock

The outputs are explicitly labelled as "DEMO / SIMULATED" when mock mode is used.
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from collections import defaultdict
import random


# Lightweight GLC CSV path
GLC_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "excel", "nasa_glc.csv")
RANDOM_SEED = 42


def load_glc_csv(filepath):
    """Load NASA GLC CSV without pandas."""
    records = []
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Using synthetic data.")
        return []
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row and 'country_name' in row and 'landslide_category' in row:
                    records.append(row)
    except Exception as e:
        print(f"Error reading CSV: {e}. Using synthetic data.")
   
    print(f"Loaded {len(records)} records from {filepath}")
    return records


def filter_to_china_region(records, region="Bijie"):
    """Filter records to China and optionally a region."""
    china_records = [r for r in records 
                     if r.get('country_name', '').strip().lower() == 'china']
    
    if region:
        region_lower = region.lower()
        region_records = [
            r for r in china_records
            if region_lower in (r.get('admin_division_name', '') or '').lower()
            or region_lower in (r.get('location_description', '') or '').lower()
            or region_lower in (r.get('gazeteer_closest_point', '') or '').lower()
        ]
        return region_records if region_records else china_records
    return china_records


class SimpleLabelEncoder:
    """Lightweight label encoder without sklearn."""
    def __init__(self):
        self.classes_ = []
        self.class_to_code = {}
        self.code_to_class = {}
    
    def fit(self, values):
        unique = sorted(set(v for v in values if v))
        self.classes_ = unique
        self.class_to_code = {c: i for i, c in enumerate(unique)}
        self.code_to_class = {i: c for c, i in self.class_to_code.items()}
        return self
    
    def transform(self, values):
        return [self.class_to_code.get(v, 0) for v in values]
    
    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)


def encode_glc_records(records):
    """Encode landslide types and triggers."""
    type_map = {
        "landslide": "Landslide", "debris_flow": "Debris Flow",
        "mudslide": "Mudslide", "rock_fall": "Rockfall",
        "earth_flow": "Earth Flow", "translational_slide": "Translational Slide",
        "complex": "Complex Landslide", "lahar": "Mudslide",
        "creep": "Landslide", "topple": "Rockfall",
        "snow_avalanche": "Rockfall", "riverbank_collapse": "Landslide",
        "other": "Landslide", "unknown": "Landslide",
    }
    
    trigger_map = {
        "downpour": "Heavy Rain", "rain": "Heavy Rain",
        "continuous_rain": "Heavy Rain", "monsoon": "Heavy Rain",
        "tropical_cyclone": "Storm", "snowfall_snowmelt": "Snowmelt",
        "freeze_thaw": "Snowmelt", "earthquake": "Earthquake",
        "mining": "Human Activity", "construction": "Human Activity",
        "flooding": "Heavy Rain", "dam_embankment_collapse": "Human Activity",
        "no_apparent_trigger": "Unknown", "unknown": "Unknown", "other": "Unknown",
    }
    
    for rec in records:
        cat = (rec.get('landslide_category') or 'unknown').lower().strip()
        rec['ls_type'] = type_map.get(cat, 'Landslide')
        
        trig = (rec.get('landslide_trigger') or 'unknown').lower().strip()
        rec['ls_trigger'] = trigger_map.get(trig, 'Unknown')
    
    # Encode
    type_encoder = SimpleLabelEncoder()
    types = [r['ls_type'] for r in records]
    type_encoder.fit_transform(types)
    
    trigger_encoder = SimpleLabelEncoder()
    triggers = [r['ls_trigger'] for r in records]
    trigger_encoder.fit_transform(triggers)
    
    for rec in records:
        rec['type_code'] = type_encoder.class_to_code.get(rec['ls_type'], 0)
        rec['trigger_code'] = trigger_encoder.class_to_code.get(rec['ls_trigger'], 0)
    
    return records, type_encoder, trigger_encoder


class DummyHMM:
    """Tiny fake HMM using only built-in Python. Generates demo probabilities."""
    def __init__(self, n_components=8, n_symbols=42):
        self.n_components = n_components
        self.n_symbols = n_symbols
        # Fake matrices (seeded, but simple)
        random.seed(RANDOM_SEED)
        self.startprob_ = [1.0 / n_components] * n_components
        self.transmat_ = [[1.0 / n_components] * n_components for _ in range(n_components)]
        self.emissionprob_ = [[random.random() for _ in range(n_symbols)] 
                              for _ in range(n_components)]
        # Normalize
        for row in self.emissionprob_:
            s = sum(row)
            for i in range(len(row)):
                row[i] /= s if s > 0 else 1.0


def get_hotspot(records):
    """Extract coarse hotspot from lat/lon records."""
    lats = []
    lons = []
    for r in records:
        try:
            lat = float(r.get('latitude', 0))
            lon = float(r.get('longitude', 0))
            if lat and lon:
                lats.append(lat)
                lons.append(lon)
        except:
            pass
    
    if lats and lons:
        avg_lat = sum(lats) / len(lats)
        avg_lon = sum(lons) / len(lons)
        count = len(records)
        return avg_lat, avg_lon, count
    return None


def run_demo(region="Bijie", mock=True):
    """Demo: filter to Bijie, produce Phase 2-style report."""
    records = load_glc_csv(GLC_PATH)
    if not records:
        # Synthetic fallback
        print("Generating synthetic Bijie data for demo...")
        records = [
            {'country_name': 'China', 'admin_division_name': 'Bijie', 'location_description': 'Test slide',
             'latitude': 27.3, 'longitude': 105.3, 'landslide_category': 'translational_slide',
             'landslide_trigger': 'heavy_rain'},
            {'country_name': 'China', 'admin_division_name': 'Bijie', 'location_description': 'Test debris',
             'latitude': 27.2, 'longitude': 105.2, 'landslide_category': 'debris_flow',
             'landslide_trigger': 'continuous_rain'},
        ]
    
    # Filter
    df_china = [r for r in records if r.get('country_name', '').strip().lower() == 'china']
    print(f"China records: {len(df_china)}")
    
    df_region = filter_to_china_region(df_china, region)
    if not df_region:
        print(f"No records for {region}, using China-wide data")
        df_region = df_china
    
    print(f"Region records: {len(df_region)}")
    
    # Encode
    df_enc, type_enc, trig_enc = encode_glc_records(df_region)
    
    print(f"\nLandslide types found: {type_enc.classes_}")
    print(f"Triggers found: {trig_enc.classes_}")
    
    # Build hotspot
    hotspot = get_hotspot(df_region)
    hotspot_text = "N/A"
    if hotspot:
        lat, lon, cnt = hotspot
        hotspot_text = f"Lat {lat:.3f}, Lon {lon:.3f} (historical events ≈ {cnt})"
    
    # Create mock HMM
    model = DummyHMM(n_components=8, n_symbols=len(type_enc.classes_) * len(trig_enc.classes_))
    
    # Fake Phase 2 output
    current_type = type_enc.classes_[0] if type_enc.classes_ else "Landslide"
    occurrence_prob = 0.32
    peak_month = "July"
    
    # Print report
    header = "DEMO / SIMULATED OUTPUT (mock HMM, no training)"
    print("\n" + "=" * 60)
    print(header)
    print("=" * 60 + "\n")
    
    print(f"PHASE 2 — {region} (China) Regional Temporal Forecast")
    print("=" * 50)
    print(f"Current Type        : {current_type}")
    print(f"Occurrence Prob.    : {occurrence_prob*100:.1f}%")
    print(f"Peak Risk Month     : {peak_month}")
    print(f"\nHistorical Risk ({region}):")
    print(f"  Events in catalog : {len(df_region)}")
    print(f"  Main types        : {', '.join(type_enc.classes_[:3])}")
    print(f"  Main trigger      : {trig_enc.classes_[0] if trig_enc.classes_ else 'Unknown'}")
    print(f"\nFuture Forecast (next 3 events):")
    print(f"  Step 1: {current_type:25s} (prob=45.0%)")
    print(f"  Step 2: Debris Flow           (prob=38.0%)")
    print(f"  Step 3: Rotational Slide      (prob=31.0%)")
    print(f"\nEstimated hotspot (coarse): {hotspot_text}")
    print("\nNote: All values are demo approximations. Full training required for production.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 Bijie demo with lightweight mock HMM")
    parser.add_argument("--region", type=str, default="Bijie", help="Target region (e.g., Bijie)")
    parser.add_argument("--mock", action="store_true", help="Enable mock HMM mode (demo output, no training)")
    args = parser.parse_args()
    
    run_demo(region=args.region, mock=True)
