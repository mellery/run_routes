#!/usr/bin/env python3
"""
Validate elevation accuracy against known landmarks in Virginia
"""

from route import get_elevation_from_raster

def validate_elevation_accuracy():
    """Test elevation data against known landmarks"""
    print("=== Elevation Validation for Virginia ===")
    
    srtm_file = 'srtm_20_05.tif'
    
    # Known elevations in Virginia (approximate)
    landmarks = [
        ("Virginia Tech campus, Blacksburg", 37.2284, -80.4234, 634),  # VT campus
        ("Christiansburg downtown", 37.1299, -80.4094, 620),  # Downtown Christiansburg
        ("Radford University", 37.1317, -80.5764, 538),  # Radford
        ("Blue Ridge Parkway (near Roanoke)", 37.2710, -79.9414, 853),  # Higher elevation area
        ("New River Valley", 37.1500, -80.5000, 580),  # General valley area
    ]
    
    print(f"Testing {len(landmarks)} known locations...")
    print()
    
    for name, lat, lon, expected_elev in landmarks:
        measured_elev = get_elevation_from_raster(srtm_file, lat, lon)
        
        if measured_elev is not None:
            difference = measured_elev - expected_elev
            percent_error = abs(difference) / expected_elev * 100
            
            status = "✅" if abs(difference) < 50 else "⚠️" if abs(difference) < 100 else "❌"
            
            print(f"{status} {name}")
            print(f"   Expected: {expected_elev}m")
            print(f"   Measured: {measured_elev:.1f}m")
            print(f"   Difference: {difference:+.1f}m ({percent_error:.1f}% error)")
            print()
        else:
            print(f"❌ {name}")
            print(f"   Could not read elevation data")
            print()
    
    # Test elevation range for Christiansburg area
    print("=== Christiansburg Area Elevation Analysis ===")
    
    # Sample points around Christiansburg
    test_points = []
    center_lat, center_lon = 37.1299, -80.4094
    
    # Create a grid of points around Christiansburg
    for lat_offset in [-0.02, -0.01, 0, 0.01, 0.02]:
        for lon_offset in [-0.02, -0.01, 0, 0.01, 0.02]:
            test_lat = center_lat + lat_offset
            test_lon = center_lon + lon_offset
            elev = get_elevation_from_raster(srtm_file, test_lat, test_lon)
            if elev is not None:
                test_points.append(elev)
    
    if test_points:
        print(f"Sampled {len(test_points)} points around Christiansburg:")
        print(f"   Min elevation: {min(test_points):.1f}m")
        print(f"   Max elevation: {max(test_points):.1f}m")
        print(f"   Average elevation: {sum(test_points)/len(test_points):.1f}m")
        print(f"   Elevation range: {max(test_points) - min(test_points):.1f}m")
    
    print("\n=== Validation Complete ===")

if __name__ == "__main__":
    validate_elevation_accuracy()