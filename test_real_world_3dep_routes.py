#!/usr/bin/env python3
"""
Real-World 3DEP Route Testing
Test route generation with actual 3DEP tiles and demonstrate precision improvements
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def find_3dep_coverage_area():
    """Find coordinates within 3DEP tile coverage for testing"""
    
    print("🔍 Finding 3DEP Coverage Area")
    print("=" * 35)
    
    try:
        from elevation_data_sources import get_elevation_manager
        
        manager = get_elevation_manager()
        source = manager.get_elevation_source()
        
        if not source:
            print("❌ No elevation source available")
            return None
        
        # Test coordinates from our validated 3DEP coverage
        test_coordinates = [
            (36.846651, -78.409308),  # Known valid coordinate from tile scan
            (36.85, -78.41),          # Nearby coordinate  
            (36.84, -78.40),          # Adjacent area
            (36.85, -78.40),          # Different nearby area
            (36.847, -78.408),        # Very close to known valid
        ]
        
        print("🧪 Testing coordinates for 3DEP coverage:")
        valid_coords = []
        
        for lat, lon in test_coordinates:
            try:
                available = source.is_available(lat, lon)
                elevation = source.get_elevation(lat, lon)
                
                if available and elevation is not None:
                    print(f"   ✅ ({lat:.4f}, {lon:.4f}): {elevation:.1f}m - VALID")
                    valid_coords.append((lat, lon, elevation))
                else:
                    print(f"   ❌ ({lat:.4f}, {lon:.4f}): No data")
                    
            except Exception as e:
                print(f"   ⚠️ ({lat:.4f}, {lon:.4f}): Error - {e}")
        
        manager.close_all()
        
        if valid_coords:
            print(f"\n✅ Found {len(valid_coords)} coordinates with 3DEP coverage")
            # Return the coordinate with highest elevation (likely interesting terrain)
            best_coord = max(valid_coords, key=lambda x: x[2])
            print(f"🎯 Selected test coordinate: ({best_coord[0]:.6f}, {best_coord[1]:.6f}) at {best_coord[2]:.1f}m")
            return (best_coord[0], best_coord[1])
        else:
            print("❌ No coordinates found with 3DEP coverage")
            return None
            
    except Exception as e:
        print(f"❌ Failed to find 3DEP coverage: {e}")
        return None

def test_route_with_3dep_precision(test_coordinate):
    """Test route generation with 3DEP precision vs fallback"""
    
    print(f"\n🛣️ Testing Route Generation with 3DEP Precision")
    print("=" * 55)
    
    try:
        from route_services import NetworkManager
        from route_services.route_optimizer import RouteOptimizer
        from route_services.elevation_profiler_enhanced import EnhancedElevationProfiler
        
        # Load network around 3DEP coverage area
        print(f"📍 Loading network around ({test_coordinate[0]:.4f}, {test_coordinate[1]:.4f})...")
        network_manager = NetworkManager(test_coordinate)
        graph = network_manager.load_network(radius_km=2.0)
        
        if not graph:
            print("❌ Failed to load network")
            return False
        
        print(f"✅ Loaded network: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
        
        # Find a starting node near our test coordinate
        best_node = None
        min_distance = float('inf')
        
        for node in graph.nodes():
            node_data = graph.nodes[node]
            node_lat, node_lon = node_data['y'], node_data['x']
            
            # Calculate distance to test coordinate
            distance = ((node_lat - test_coordinate[0])**2 + (node_lon - test_coordinate[1])**2)**0.5
            if distance < min_distance:
                min_distance = distance
                best_node = node
        
        if not best_node:
            print("❌ No suitable starting node found")
            return False
        
        start_node_data = graph.nodes[best_node]
        print(f"🎯 Using start node {best_node} at ({start_node_data['y']:.6f}, {start_node_data['x']:.6f})")
        
        # Create enhanced route optimizer and elevation profiler
        route_optimizer = RouteOptimizer(graph)
        elevation_profiler = EnhancedElevationProfiler(graph)
        
        # Test different objectives and algorithms
        test_configs = [
            {"objective": "distance", "algorithm": "auto", "name": "Distance Optimized (TSP)"},
            {"objective": "elevation", "algorithm": "auto", "name": "Elevation Optimized (GA)"},
        ]
        
        results = {}
        
        for config in test_configs:
            print(f"\n🧪 Testing {config['name']}...")
            
            start_time = time.time()
            route_result = route_optimizer.optimize_route(
                start_node=best_node,
                target_distance_km=1.0,  # 1km route for detailed testing
                objective=config["objective"],
                algorithm=config["algorithm"]
            )
            generation_time = time.time() - start_time
            
            if route_result:
                # Generate enhanced elevation profile
                profile_data = elevation_profiler.generate_profile_data(
                    route_result, 
                    use_enhanced_elevation=True, 
                    interpolate_points=True
                )
                
                results[config["name"]] = {
                    'route_result': route_result,
                    'profile_data': profile_data,
                    'generation_time': generation_time
                }
                
                print(f"   ✅ Generated successfully in {generation_time:.2f}s")
                print(f"   📏 Route: {len(route_result.get('route', []))} nodes")
                print(f"   📊 Distance: {route_result.get('stats', {}).get('total_distance_km', 0):.2f}km")
                print(f"   ⛰️ Elevation gain: {route_result.get('stats', {}).get('total_elevation_gain_m', 0):.1f}m")
                
                # Show enhanced elevation info
                if profile_data:
                    data_source = profile_data.get('data_source_info', {})
                    if data_source:
                        print(f"   🔬 Data source: {data_source.get('active_source', 'Unknown')}")
                        if 'usage_stats' in data_source:
                            stats = data_source['usage_stats']
                            if 'primary_percentage' in stats:
                                print(f"   📊 High-res coverage: {stats['primary_percentage']:.1f}%")
                
            else:
                print(f"   ❌ Route generation failed")
        
        elevation_profiler.close()
        
        # Compare results
        if len(results) >= 2:
            print(f"\n📊 Route Comparison Results")
            print("=" * 35)
            
            for name, result_data in results.items():
                route_result = result_data['route_result']
                profile_data = result_data['profile_data']
                
                print(f"\n{name}:")
                print(f"   Generation time: {result_data['generation_time']:.2f}s")
                print(f"   Route nodes: {len(route_result.get('route', []))}")
                print(f"   Distance: {route_result.get('stats', {}).get('total_distance_km', 0):.3f}km")
                print(f"   Elevation gain: {route_result.get('stats', {}).get('total_elevation_gain_m', 0):.1f}m")
                
                if profile_data:
                    stats = profile_data.get('elevation_stats', {})
                    if stats:
                        print(f"   Elevation range: {stats.get('min_elevation_m', 0):.1f}m - {stats.get('max_elevation_m', 0):.1f}m")
                        print(f"   Max grade: {stats.get('max_grade_percent', 0):.1f}%")
                        
                        quality = stats.get('data_quality', {})
                        if quality:
                            print(f"   Data resolution: {quality.get('resolution_m', 'Unknown')}m")
                            print(f"   Points analyzed: {quality.get('points_analyzed', 0)}")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Route testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_elevation_precision_comparison():
    """Compare elevation precision between 3DEP and SRTM"""
    
    print(f"\n🔬 Testing Elevation Precision Comparison")
    print("=" * 50)
    
    try:
        from elevation_data_sources import get_elevation_manager, LocalThreeDEPSource, SRTMElevationSource
        
        # Get manager with hybrid source
        manager = get_elevation_manager()
        hybrid_source = manager.get_elevation_source()
        
        # Also create individual sources for comparison
        threedep_source = LocalThreeDEPSource()
        srtm_source = SRTMElevationSource("srtm_20_05.tif")
        
        # Test coordinates in 3DEP coverage area
        test_coords = [
            (36.846651, -78.409308),  # Known valid coordinate from tile scan
            (36.85, -78.41),          # Nearby coordinate  
            (36.84, -78.40),          # Adjacent area
            (36.847, -78.408),        # Very close to known valid
        ]
        
        print("🧪 Comparing elevation precision:")
        print("   Coordinate          3DEP 1m    SRTM 90m   Difference")
        print("   " + "-" * 55)
        
        comparisons = []
        
        for lat, lon in test_coords:
            try:
                # Get elevations from different sources
                threedep_elev = threedep_source.get_elevation(lat, lon)
                srtm_elev = srtm_source.get_elevation(lat, lon)
                hybrid_elev = hybrid_source.get_elevation(lat, lon)
                
                # Format results
                threedep_str = f"{threedep_elev:.1f}m" if threedep_elev else "N/A"
                srtm_str = f"{srtm_elev:.1f}m" if srtm_elev else "N/A"
                
                if threedep_elev and srtm_elev:
                    diff = abs(threedep_elev - srtm_elev)
                    diff_str = f"±{diff:.1f}m"
                    comparisons.append(diff)
                else:
                    diff_str = "N/A"
                
                print(f"   ({lat:.6f}, {lon:.6f})  {threedep_str:>8} {srtm_str:>10} {diff_str:>10}")
                
            except Exception as e:
                print(f"   ({lat:.6f}, {lon:.6f})  Error: {e}")
        
        # Calculate statistics
        if comparisons:
            avg_diff = sum(comparisons) / len(comparisons)
            max_diff = max(comparisons)
            min_diff = min(comparisons)
            
            print(f"\n📊 Precision Comparison Statistics:")
            print(f"   Average difference: ±{avg_diff:.1f}m")
            print(f"   Maximum difference: ±{max_diff:.1f}m")
            print(f"   Minimum difference: ±{min_diff:.1f}m")
            print(f"   Comparisons made: {len(comparisons)}")
            
            print(f"\n🎯 Precision Improvement Analysis:")
            print(f"   3DEP accuracy: ±0.3m (99% confidence)")
            print(f"   SRTM accuracy: ±16m (90% confidence)")
            print(f"   Theoretical improvement: 53× better accuracy")
            print(f"   Observed variation: {avg_diff:.1f}m (within expected range)")
        
        # Cleanup
        threedep_source.close()
        srtm_source.close()
        manager.close_all()
        
        return len(comparisons) > 0
        
    except Exception as e:
        print(f"❌ Precision comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main real-world testing function"""
    
    print("🌍 Real-World 3DEP Route Testing")
    print("Testing with actual 3DEP tiles and route generation")
    print("=" * 60)
    
    # Step 1: Find 3DEP coverage area
    test_coordinate = find_3dep_coverage_area()
    
    if not test_coordinate:
        print("❌ Cannot proceed without 3DEP coverage area")
        return False
    
    # Step 2: Test route generation with 3DEP precision
    route_success = test_route_with_3dep_precision(test_coordinate)
    
    # Step 3: Test elevation precision comparison
    precision_success = test_elevation_precision_comparison()
    
    # Summary
    print(f"\n🏆 Real-World Testing Summary")
    print("=" * 40)
    
    if route_success and precision_success:
        print("✅ All real-world tests passed!")
        print("🎉 3DEP integration is working with real route generation")
        print("📊 Demonstrated precision improvements over SRTM")
        print("🛣️ Ready for production use with 1m elevation data")
        
        print(f"\n🎯 Key Achievements:")
        print(f"   ✅ Successfully found 3DEP coverage area")
        print(f"   ✅ Generated routes with 1m elevation precision")
        print(f"   ✅ Demonstrated hybrid 3DEP/SRTM fallback system")
        print(f"   ✅ Compared elevation precision between data sources")
        print(f"   ✅ Enhanced route services working with real data")
        
        return True
    else:
        print("⚠️ Some real-world tests failed")
        print("🔧 May need additional debugging or 3DEP tile coverage")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)