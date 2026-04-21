"""
Integrated Phase 1 → Phase 2 Pipeline with Visualization

Phase 1: Binary landslide detection on satellite image
Phase 2: Regional temporal forecasting + hotspot visualization (if Phase 1 = Landslide)

Usage:
    python run_integrated_pipeline.py --image /path/to/image.jpg --lat 27.3 --lon 105.3
    python run_integrated_pipeline.py --image /path/to/image.jpg --lat 27.3 --lon 105.3 --region Bijie
    
Output:
    - Phase 1: Landslide detected/not detected + confidence
    - Phase 2 (if landslide): Type, probability, peak month, hotspot with red circle on China map
"""

import argparse
import os
import sys
import csv
import json
from datetime import datetime
from collections import defaultdict
import random
from pathlib import Path

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# Try imports for Phase 1
try:
    import torch
    from phase1_alexnet.ensemble_predict import load_ensemble, predict_ensemble
    PHASE1_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Phase 1 not available: {e}")
    PHASE1_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class IntegratedPipeline:
    """Combined Phase 1 (image classification) → Phase 2 (regional forecasting)."""
    
    def __init__(self):
        self.phase1_available = PHASE1_AVAILABLE
        self.device = None
        self.convnext = None
        self.swinv2 = None
        
        # Phase 1: Load ensemble
        if self.phase1_available:
            print("[PHASE 1] Loading ensemble (ConvNeXt-CBAM-FPN + SwinV2-Small)...")
            try:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.convnext, self.swinv2, _ = load_ensemble(self.device)
                print("[PHASE 1] [OK] Ensemble loaded.\n")
            except Exception as e:
                print(f"[PHASE 1] Error loading ensemble: {e}")
                self.phase1_available = False
        
        # Phase 2: Load lightweight Bijie data (no heavy dependencies)
        print("[PHASE 2] Loading NASA GLC data for China/Bijie...")
        self.glc_data = self._load_glc_csv()
        print(f"[PHASE 2] [OK] Loaded {len(self.glc_data)} NASA GLC records.\n")
    
    def _load_glc_csv(self):
        """Load NASA GLC CSV without pandas."""
        glc_path = os.path.join(config.DATA_DIR, "excel", "nasa_glc.csv")
        records = []
        if not os.path.exists(glc_path):
            print(f"Warning: {glc_path} not found.")
            return []
        
        try:
            with open(glc_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row and 'latitude' in row and 'longitude' in row:
                        try:
                            row['latitude'] = float(row.get('latitude', 0))
                            row['longitude'] = float(row.get('longitude', 0))
                            records.append(row)
                        except:
                            pass
        except Exception as e:
            print(f"Error reading CSV: {e}")
        
        return records
    
    def _get_nearby_events(self, lat, lon, radius_km=50.0):
        """Filter NASA GLC events within radius of given lat/lon (rough approximation)."""
        # Rough conversion: 1 degree ≈ 111 km
        radius_degrees = radius_km / 111.0
        nearby = []
        
        for record in self.glc_data:
            rec_lat = record.get('latitude', 0)
            rec_lon = record.get('longitude', 0)
            
            if abs(rec_lat - lat) < radius_degrees and abs(rec_lon - lon) < radius_degrees:
                nearby.append(record)
        
        return nearby
    
    def _get_regional_forecast(self, events, region="China"):
        """Extract Phase 2 forecast from nearby events."""
        if not events:
            return {
                'current_type': 'Landslide (Unknown Type)',
                'occurrence_probability': 0.25,
                'peak_risk_month': 'July',
                'future_forecast': [
                    {'step': 1, 'landslide_type': 'Translational Slide', 'probability': 0.35},
                    {'step': 2, 'landslide_type': 'Debris Flow', 'probability': 0.30},
                    {'step': 3, 'landslide_type': 'Mudflow', 'probability': 0.25},
                ],
                'event_count': 0,
                'dominant_type': 'Unknown',
                'dominant_trigger': 'Unknown',
                'seasonal_distribution': {},
            }
        
        # Count types
        type_counts = defaultdict(int)
        trigger_counts = defaultdict(int)
        month_counts = defaultdict(int)
        
        for event in events:
            ls_type = event.get('landslide_category', 'Landslide').replace('_', ' ').title()
            trigger = event.get('landslide_trigger', 'Unknown').replace('_', ' ').title()
            month = event.get('month', 7)  # Default to July
            
            type_counts[ls_type] += 1
            trigger_counts[trigger] += 1
            month_counts[month] += 1
        
        # Get dominant types
        dominant_type = max(type_counts, key=type_counts.get) if type_counts else 'Translational Slide'
        dominant_trigger = max(trigger_counts, key=trigger_counts.get) if trigger_counts else 'Heavy Rain'
        peak_month_num = max(month_counts, key=month_counts.get) if month_counts else 7
        peak_month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][peak_month_num - 1] if 1 <= peak_month_num <= 12 else 'July'
        
        # Occurrence probability (based on frequency)
        total_years = 50  # Approximate span of NASA GLC
        event_frequency = len(events) / total_years
        occurrence_prob = min(0.9, event_frequency * 0.2)  # Scale to 0-0.9 range
        
        # Sort types by frequency for forecast
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        forecast_types = [t[0] for t in sorted_types[:3]] if sorted_types else ['Translational Slide', 'Debris Flow', 'Mudflow']
        
        return {
            'current_type': dominant_type,
            'occurrence_probability': occurrence_prob,
            'peak_risk_month': peak_month,
            'future_forecast': [
                {'step': i+1, 'landslide_type': forecast_types[i] if i < len(forecast_types) else 'Landslide', 
                 'probability': max(0.2, occurrence_prob - i*0.1)}
                for i in range(3)
            ],
            'event_count': len(events),
            'dominant_type': dominant_type,
            'dominant_trigger': dominant_trigger,
            'seasonal_distribution': dict(month_counts),
        }
    
    def predict(self, image_path=None, latitude=None, longitude=None, region="Bijie", **kwargs):
        """
        Integrated prediction:
        1. Phase 1: Classify image (landslide or not)
        2. Phase 2: If landslide, forecast using nearby NASA GLC events
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'location': {'latitude': latitude, 'longitude': longitude, 'region': region},
            'phase1': None,
            'phase2': None,
            'visualization_path': None,
        }
        
        # ── Phase 1: Classify Image ──
        if image_path and self.phase1_available and os.path.exists(image_path):
            print(f"[PHASE 1] Classifying image: {image_path}")
            try:
                phase1_result = predict_ensemble(image_path, self.convnext, self.swinv2, self.device)
                result['phase1'] = phase1_result
                is_landslide = phase1_result.get('is_landslide', phase1_result.get('label') != 'non_landslide')
                confidence = phase1_result.get('confidence', 0.0)
                print(f"[PHASE 1] Result: {phase1_result.get('label', 'Unknown')} (confidence: {confidence*100:.1f}%)\n")
            except Exception as e:
                print(f"[PHASE 1] Error: {e}\n")
                is_landslide = False
        else:
            # Demo mode: assume landslide
            is_landslide = True
            result['phase1'] = {
                'label': 'landslide',
                'is_landslide': True,
                'confidence': 0.95,
                'message': '[DEMO] Assuming landslide detected'
            }
            print(f"[PHASE 1] DEMO MODE: Assuming landslide detected (no image provided)\n")
        
        # ── Phase 2: Regional Forecasting ──
        if is_landslide and latitude is not None and longitude is not None:
            print(f"[PHASE 2] Analyzing regional risk (lat={latitude:.3f}, lon={longitude:.3f})...")
            
            # Get nearby events (100 km radius)
            nearby_events = self._get_nearby_events(latitude, longitude, radius_km=100.0)
            print(f"[PHASE 2] Found {len(nearby_events)} nearby NASA GLC events\n")
            
            # Generate forecast
            forecast = self._get_regional_forecast(nearby_events, region=region)
            result['phase2'] = forecast
            
            # Print Phase 2 output
            print("=" * 70)
            print(f"PHASE 2 — Regional Temporal Forecast ({region})")
            print("=" * 70)
            print(f"Location            : Lat {latitude:.3f}, Lon {longitude:.3f}")
            print(f"Current Type        : {forecast['current_type']}")
            print(f"Occurrence Prob.    : {forecast['occurrence_probability']*100:.1f}%")
            print(f"Peak Risk Month     : {forecast['peak_risk_month']}")
            print(f"Dominant Trigger    : {forecast['dominant_trigger']}")
            print(f"Historical Events   : {forecast['event_count']}")
            print(f"\nFuture Forecast (next 3 events):")
            for f in forecast['future_forecast']:
                print(f"  Step {f['step']}: {f['landslide_type']:25s} (prob={f['probability']*100:.1f}%)")
            print(f"\nSeasonal Distribution: {dict(forecast['seasonal_distribution'])}")
            print("=" * 70 + "\n")
            
            # Visualize hotspot (SVG, no dependencies required)
            viz_path = self._visualize_hotspot(latitude, longitude, nearby_events, region)
            result['visualization_path'] = viz_path
        
        elif is_landslide:
            print("[PHASE 2] SKIPPED: Location not provided. Use --lat and --lon to enable Phase 2.\n")
        else:
            print("[PHASE 2] SKIPPED: Phase 1 did not detect landslide.\n")
        
        return result
    
    def _visualize_hotspot(self, lat, lon, nearby_events, region_name):
        """Create SVG visualization of hotspot with red circle on China map."""
        try:
            # China bounding box (rough)
            map_width = 800
            map_height = 600
            min_lat, max_lat = 18, 54
            min_lon, max_lon = 73, 135
            
            # Scale function
            def scale_coords(latitude, longitude):
                x = ((longitude - min_lon) / (max_lon - min_lon)) * map_width
                y = ((max_lat - latitude) / (max_lat - min_lat)) * map_height
                return x, y
            
            # Create SVG
            svg_lines = [
                f'<svg width="{map_width + 100}" height="{map_height + 150}" xmlns="http://www.w3.org/2000/svg">',
                '<style>',
                '  .title { font-size: 18px; font-weight: bold; fill: black; }',
                '  .label { font-size: 12px; fill: black; }',
                '  .event { r: 3; fill: blue; opacity: 0.6; }',
                '  .hotspot { r: 15; fill: red; }',
                '</style>',
                f'<title>Landslide Risk Hotspot - {region_name}, China</title>',
                f'<text x="50" y="30" class="title">Landslide Risk Hotspot — {region_name}, China</text>',
                f'<text x="50" y="50" class="label">Red star = Predicted Hotspot | Blue dots = Historical Events | Red dashed circle = Risk Zone</text>',
                '',
                '<!-- Map background -->',
                f'<rect x="40" y="70" width="{map_width}" height="{map_height}" fill="#e6f2ff" stroke="black" stroke-width="1"/>',
                '',
                '<!-- Latitude/Longitude grid -->',
            ]
            
            # Add grid lines
            for lat_val in range(20, 55, 5):
                y = ((max_lat - lat_val) / (max_lat - min_lat)) * map_height + 70
                svg_lines.append(f'<line x1="40" y1="{y}" x2="{map_width + 40}" y2="{y}" stroke="lightgray" stroke-width="0.5" stroke-dasharray="2,2"/>')
                svg_lines.append(f'<text x="5" y="{y + 4}" class="label" font-size="10">{lat_val}°</text>')
            
            for lon_val in range(75, 135, 10):
                x = ((lon_val - min_lon) / (max_lon - min_lon)) * map_width + 40
                svg_lines.append(f'<line x1="{x}" y1="70" x2="{x}" y2="{map_height + 70}" stroke="lightgray" stroke-width="0.5" stroke-dasharray="2,2"/>')
                svg_lines.append(f'<text x="{x - 15}" y="{map_height + 90}" class="label" font-size="10">{lon_val}°</text>')
            
            svg_lines.append('')
            svg_lines.append('<!-- Historical events (blue dots) -->')
            
            # Plot nearby events
            for event in nearby_events:
                event_lat = event.get('latitude', 0)
                event_lon = event.get('longitude', 0)
                x, y = scale_coords(event_lat, event_lon)
                svg_lines.append(f'<circle cx="{x + 40}" cy="{y + 70}" class="event"/>')
            
            svg_lines.append('')
            svg_lines.append('<!-- Predicted hotspot (red star) -->')
            
            # Plot hotspot
            center_x, center_y = scale_coords(lat, lon)
            center_x += 40
            center_y += 70
            
            # Red star (5-pointed)
            star_size = 15
            svg_lines.append(f'<polygon points="{center_x},{center_y - star_size} ')
            svg_lines.append(f'{center_x + star_size*0.38},{center_y - star_size*0.38} ')
            svg_lines.append(f'{center_x + star_size},{center_y} ')
            svg_lines.append(f'{center_x + star_size*0.38},{center_y + star_size*0.38} ')
            svg_lines.append(f'{center_x},{center_y + star_size} ')
            svg_lines.append(f'{center_x - star_size*0.38},{center_y + star_size*0.38} ')
            svg_lines.append(f'{center_x - star_size},{center_y} ')
            svg_lines.append(f'{center_x - star_size*0.38},{center_y - star_size*0.38}" ')
            svg_lines.append('fill="red" stroke="darkred" stroke-width="1"/>')
            
            # Red dashed circle (50 km radius)
            radius_pixels = (50.0 / 111.0) / (max_lon - min_lon) * map_width
            svg_lines.append(f'<circle cx="{center_x}" cy="{center_y}" r="{radius_pixels}" ')
            svg_lines.append('fill="none" stroke="red" stroke-width="2" stroke-dasharray="5,5"/>')
            
            # Legend text
            svg_lines.append(f'<text x="50" y="{map_height + 120}" class="label">Hotspot Location: Lat {lat:.4f}, Lon {lon:.4f}</text>')
            svg_lines.append(f'<text x="50" y="{map_height + 140}" class="label">Historical events nearby: {len(nearby_events)}</text>')
            
            svg_lines.append('</svg>')
            
            # Save file
            output_dir = os.path.join(config.PLOTS_DIR, "integrated_pipeline")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"hotspot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg")
            
            with open(output_path, 'w') as f:
                f.write('\n'.join(svg_lines))
            
            print(f"[OK] SVG visualization saved: {output_path}\n")
            return output_path
        
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Integrated Phase 1→2 Pipeline: Image classification + Regional forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classification only (Phase 1):
  python run_integrated_pipeline.py --image data/sample.jpg
  
  # Full pipeline (Phase 1 + Phase 2):
  python run_integrated_pipeline.py --image data/sample.jpg --lat 27.3 --lon 105.3 --region Bijie
  
  # Phase 2 only (demo mode, no image):
  python run_integrated_pipeline.py --lat 27.3 --lon 105.3 --region Bijie
"""
    )
    parser.add_argument("--image", type=str, default=None, help="Path to satellite image (Phase 1)")
    parser.add_argument("--lat", type=float, default=None, help="Latitude of image location")
    parser.add_argument("--lon", type=float, default=None, help="Longitude of image location")
    parser.add_argument("--region", type=str, default="Bijie", help="Region name (for display)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("INTEGRATED LANDSLIDE IDENTIFICATION PIPELINE")
    print("Phase 1: Binary Classification | Phase 2: Temporal Forecasting + Visualization")
    print("=" * 70 + "\n")
    
    pipeline = IntegratedPipeline()
    result = pipeline.predict(
        image_path=args.image,
        latitude=args.lat,
        longitude=args.lon,
        region=args.region,
    )
    
    # Output JSON summary
    print(f"\n[OK] Pipeline complete. Result summary:")
    print(f"  Phase 1 Label: {result['phase1']['label'] if result['phase1'] else 'N/A'}")
    print(f"  Phase 2 Active: {result['phase2'] is not None}")
    if result['visualization_path']:
        print(f"  Visualization: {result['visualization_path']}")
