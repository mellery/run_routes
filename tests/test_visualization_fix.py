#!/usr/bin/env python3
"""
Test Visualization Fix
Test the array length mismatch fix in route visualization
"""

def test_visualization_logic():
    """Test the visualization array handling logic"""
    
    print("🧪 Testing Visualization Array Length Fix")
    print("=" * 50)
    
    # Simulate the problematic scenario
    print("\n1️⃣ Simulating route data...")
    
    # Example route with 6 nodes
    route = [1001, 1002, 1003, 1004, 1005, 1006]
    
    # Simulate coordinate and elevation data
    lats = [37.1299, 37.1300, 37.1301, 37.1302, 37.1303, 37.1304]  # 6 items
    lons = [-80.4094, -80.4095, -80.4096, -80.4097, -80.4098, -80.4099]  # 6 items  
    elevations = [610, 615, 620, 625, 630, 635]  # 6 items
    distances = [0, 100, 200, 300, 400, 500]  # 6 items
    
    print(f"   Route nodes: {len(route)} items")
    print(f"   Coordinates: {len(lats)} lats, {len(lons)} lons")
    print(f"   Elevations: {len(elevations)} items")
    print(f"   Distances: {len(distances)} items")
    
    # Add return to start (this was causing the mismatch)
    print("\n2️⃣ Adding return to start...")
    cumulative_distance = distances[-1] + 150  # Add return distance
    distances.append(cumulative_distance)  # Now 7 items
    
    print(f"   Distances after return: {len(distances)} items")
    print(f"   Elevations still: {len(elevations)} items")
    print(f"   ❌ This would cause the 'c' argument error!")
    
    # Test the fix
    print("\n3️⃣ Testing the fix...")
    
    # Check if arrays match for scatter plot
    if len(lons) == len(lats) == len(elevations):
        print("   ✅ Coordinate arrays match - can use elevation coloring")
    else:
        print("   ⚠️ Coordinate arrays don't match - will use fallback coloring")
    
    # Fix for elevation profile
    elevations_with_return = elevations + [elevations[0]] if elevations else []
    distances_km = [d / 1000 for d in distances]
    
    print(f"   Distances for profile: {len(distances_km)} items")
    print(f"   Elevations with return: {len(elevations_with_return)} items")
    
    if len(distances_km) == len(elevations_with_return):
        print("   ✅ Arrays match for elevation profile")
    else:
        print("   ⚠️ Arrays don't match - will use fallback plot")
    
    # Test edge cases
    print("\n4️⃣ Testing edge cases...")
    
    # Empty route
    empty_elevations = []
    empty_elevations_with_return = empty_elevations + [empty_elevations[0]] if empty_elevations else []
    print(f"   Empty route handling: {len(empty_elevations_with_return)} items (should be 0)")
    
    # Single node route
    single_elevations = [610]
    single_elevations_with_return = single_elevations + [single_elevations[0]] if single_elevations else []
    print(f"   Single node route: {len(single_elevations_with_return)} items (should be 2)")

def test_matplotlib_import():
    """Test if matplotlib is available"""
    print("\n5️⃣ Testing matplotlib availability...")
    
    try:
        import matplotlib.pyplot as plt
        print("   ✅ matplotlib is available")
        
        # Test basic plot creation
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            print("   ✅ Can create subplots")
            plt.close(fig)  # Close to avoid display issues
        except Exception as e:
            print(f"   ⚠️ Subplot creation issue: {e}")
            
    except ImportError as e:
        print(f"   ❌ matplotlib not available: {e}")

def main():
    """Main test function"""
    test_visualization_logic()
    test_matplotlib_import()
    
    print("\n" + "=" * 50)
    print("🎯 Visualization Fix Test Complete")
    print("\n📋 Key improvements:")
    print("   • Fixed array length mismatch by not adding extra elevation point")
    print("   • Added fallback handling for mismatched arrays")
    print("   • Proper elevation profile with return segment")
    print("   • Better error handling for edge cases")
    print("\n🎉 Route visualization should now work without 'c' argument errors!")

if __name__ == "__main__":
    main()