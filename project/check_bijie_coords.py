import csv

bijie_events = []
with open('data/excel/nasa_glc.csv', 'r', encoding='utf-8', errors='ignore') as f:
    reader = csv.DictReader(f)
    for row in reader:
        admin = row.get('admin_division_name', '').lower()
        loc = row.get('location_description', '').lower()
        
        if admin == 'bijie' or 'bijie' in loc:
            try:
                lat = float(row.get('latitude', 0))
                lon = float(row.get('longitude', 0))
                bijie_events.append({
                    'lat': lat, 
                    'lon': lon, 
                    'type': row.get('landslide_category', 'Unknown'),
                    'admin': admin,
                    'loc': loc[:50]
                })
            except:
                pass

print(f'Found {len(bijie_events)} Bijie events\n')
for i, e in enumerate(bijie_events[:10]):
    print(f"{i+1}. Lat {e['lat']:.4f}, Lon {e['lon']:.4f} - {e['type']} | Admin: {e['admin']}")

if bijie_events:
    lats = [e['lat'] for e in bijie_events]
    lons = [e['lon'] for e in bijie_events]
    print(f"\nCoordinate range:")
    print(f"  Lat: {min(lats):.3f} to {max(lats):.3f}")
    print(f"  Lon: {min(lons):.3f} to {max(lons):.3f}")
    print(f"  Center: ({sum(lats)/len(lats):.3f}, {sum(lons)/len(lons):.3f})")
