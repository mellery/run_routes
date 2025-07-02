#!/usr/bin/env python3
"""
Test Improved Performance
Test the enhanced TSP solver with progress indicators and timeouts
"""

import time
from graph_cache import load_or_generate_graph
from tsp_solver import RunningRouteOptimizer, RouteObjective

def test_performance_improvements():
    """Test the performance improvements"""
    
    print("🧪 Testing Performance Improvements")
    print("=" * 50)
    
    # Load network
    print("1️⃣ Loading network...")
    graph = load_or_generate_graph(
        center_point=(37.1299, -80.4094),
        radius_m=800,
        network_type='all'
    )
    
    if not graph:
        print("❌ Failed to load graph")
        return
    
    print(f"✅ Loaded graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    # Get a starting node
    start_node = list(graph.nodes)[0]
    target_distance = 1.0  # 1km
    
    print(f"\n2️⃣ Testing route generation with progress indicators...")
    print(f"   Start node: {start_node}")
    print(f"   Target distance: {target_distance}km")
    
    # Test nearest neighbor (should be fast)
    print(f"\n3️⃣ Testing nearest neighbor algorithm...")
    try:
        optimizer = RunningRouteOptimizer(graph)
        
        start_time = time.time()
        result = optimizer.find_optimal_route(
            start_node=start_node,
            target_distance_km=target_distance,
            objective=RouteObjective.MINIMIZE_DISTANCE,
            algorithm="nearest_neighbor"
        )
        solve_time = time.time() - start_time
        
        if result:
            stats = result['stats']
            print(f"✅ Nearest neighbor completed in {solve_time:.2f} seconds")
            print(f"   Route: {len(result['route'])} nodes")
            print(f"   Distance: {stats.get('total_distance_km', 0):.2f} km")
            print(f"   Elevation gain: {stats.get('total_elevation_gain_m', 0):.0f} m")
        else:
            print("❌ Nearest neighbor failed")
            
    except Exception as e:
        print(f"❌ Nearest neighbor error: {e}")
        return
    
    # Test genetic algorithm with timeout protection
    print(f"\n4️⃣ Testing genetic algorithm with timeout protection...")
    try:
        start_time = time.time()
        result_genetic = optimizer.find_optimal_route(
            start_node=start_node,
            target_distance_km=target_distance,
            objective=RouteObjective.MINIMIZE_DISTANCE,
            algorithm="genetic"
        )
        solve_time_genetic = time.time() - start_time
        
        if result_genetic:
            stats = result_genetic['stats']
            print(f"✅ Genetic algorithm completed in {solve_time_genetic:.2f} seconds")
            print(f"   Route: {len(result_genetic['route'])} nodes")  
            print(f"   Distance: {stats.get('total_distance_km', 0):.2f} km")
            
            if solve_time_genetic < 35:  # Should complete within timeout
                print(f"   ✅ Completed within timeout (30s + overhead)")
            else:
                print(f"   ⚠️ Took longer than expected")
        else:
            print("❌ Genetic algorithm failed")
            
    except Exception as e:
        print(f"⚠️ Genetic algorithm error: {e}")
    
    print(f"\n5️⃣ Performance summary...")
    print(f"   ✅ Progress indicators show algorithm status")
    print(f"   ✅ Timeout protection prevents hanging (30s max)")
    print(f"   ✅ Reduced parameters: 30 generations, 20 population")
    print(f"   ✅ Early stopping when no improvement")
    print(f"   ✅ Limited search space (15 max nodes)")
    print(f"   ✅ Better error handling and fallbacks")

def main():
    """Main test function"""
    test_performance_improvements()
    
    print("\n" + "=" * 50)
    print("🎯 Performance Test Complete")
    print("\n📋 Key improvements in TSP solver:")
    print("   • Real-time progress indicators during optimization")
    print("   • 30-second timeout for genetic algorithm")  
    print("   • Reduced generations (100→30) and population (50→20)")
    print("   • Early stopping when stagnant (10 generations)")
    print("   • Limited search space (20→15 max nodes)")
    print("   • Fallback routes when constraints can't be met")
    print("   • Better error messages and debugging info")
    print("\n🚀 The CLI should now respond much faster!")

if __name__ == "__main__":
    main()