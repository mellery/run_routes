#!/usr/bin/env python3
"""
Minimal test of Phase 1 core functions
"""

import rasterio
from route import haversine_distance, get_elevation_from_raster

def minimal_test():
    print("=== Minimal Phase 1 Test ===")
    
    # Test 1: Haversine distance
    print("1. Testing haversine distance...")
    lat1, lon1 = 37.13, -80.41
    lat2, lon2 = 37.14, -80.42
    distance = haversine_distance(lat1, lon1, lat2, lon2)
    print(f"   Distance between ({lat1}, {lon1}) and ({lat2}, {lon2}): {distance:.2f}m")
    
    # Test 2: SRTM file info
    print("\n2. Checking SRTM file...")
    srtm_file = 'srtm_38_03.tif'
    try:
        with rasterio.open(srtm_file) as src:
            print(f"   SRTM bounds: {src.bounds}")
            print(f"   SRTM shape: {src.shape}")
            print(f"   SRTM CRS: {src.crs}")
    except Exception as e:
        print(f"   Error reading SRTM file: {e}")
    
    # Test 3: Elevation extraction
    print("\n3. Testing elevation extraction...")
    test_coords = [
        (37.13, -80.41),  # Should be within Christiansburg area
        (37.14, -80.42),
        (37.15, -80.43)
    ]
    
    for lat, lon in test_coords:
        elevation = get_elevation_from_raster(srtm_file, lat, lon)
        if elevation is not None:
            print(f"   Elevation at ({lat}, {lon}): {elevation:.1f}m")
        else:
            print(f"   No elevation data for ({lat}, {lon})")
    
    print("\n=== Minimal Test Complete ===")

if __name__ == "__main__":
    minimal_test()