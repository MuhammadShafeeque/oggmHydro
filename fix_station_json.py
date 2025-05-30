import json
import os

# Read the current JSON file
json_file_path = '/workspace/tutorials/tutorials/station_data/batura_glacier_stations.json'

with open(json_file_path, 'r') as f:
    data = json.load(f)

# Update the structure to include elevation in metadata
for station_id, station_data in data['stations'].items():
    # Add elevation to metadata as expected by OGGM
    if 'metadata' not in station_data:
        station_data['metadata'] = {}
    
    # Copy elevation from coordinates to metadata
    if 'coordinates' in station_data and 'elevation' in station_data['coordinates']:
        station_data['metadata']['elevation'] = station_data['coordinates']['elevation']
    
    # Also add other metadata
    station_data['metadata']['latitude'] = station_data['coordinates']['lat']
    station_data['metadata']['longitude'] = station_data['coordinates']['lon']
    
    print(f"Updated {station_id}: elevation = {station_data['metadata']['elevation']}m")

# Save the updated JSON file
with open(json_file_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"✅ Updated JSON file saved to: {json_file_path}")
print("✅ Elevation data is now available at both 'metadata.elevation' and 'coordinates.elevation'")
