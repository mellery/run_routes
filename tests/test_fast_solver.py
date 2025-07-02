#!/usr/bin/env python3
"""
Test Fast TSP Solver
Quick test of the fast solver without distance matrix precomputation
"""

import time
from graph_cache import load_or_generate_graph
from tsp_solver_fast import FastRunningRouteOptimizer, RouteObjective

def test_fast_solver():
    """Test the fast solver performance"""
    
    print("🧪 Testing Fast TSP Solver")
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
    
    print(f"\n2️⃣ Testing fast route generation...")
    print(f"   Start node: {start_node}")
    print(f"   Target distance: {target_distance}km")
    print(f"   Objective: maximize elevation")
    
    # Test fast solver
    try:
        optimizer = FastRunningRouteOptimizer(graph)
        
        start_time = time.time()
        result = optimizer.find_optimal_route(
            start_node=start_node,
            target_distance_km=target_distance,
            objective=RouteObjective.MAXIMIZE_ELEVATION,  # This was the problematic objective
            algorithm="nearest_neighbor"
        )
        solve_time = time.time() - start_time
        
        if result:
            stats = result['stats']
            print(f"\n✅ Fast solver completed in {solve_time:.2f} seconds")
            print(f"   Route: {len(result['route'])} nodes")
            print(f"   Distance: {stats.get('total_distance_km', 0):.2f} km")
            print(f"   Elevation gain: {stats.get('total_elevation_gain_m', 0):.0f} m")
            print(f"   Algorithm: {result.get('algorithm', 'unknown')}")
            print(f"   Objective: {result.get('objective', 'unknown')}")
            
            # Check if it's within reasonable time
            if solve_time < 30:
                print(f"   ✅ Completed within reasonable time")
            else:
                print(f"   ⚠️ Took longer than expected")
                
        else:
            print("❌ Fast solver failed")
            
    except Exception as e:
        print(f"❌ Fast solver error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n3️⃣ Performance comparison...")
    print(f"   Fast solver benefits:")
    print(f"   • No distance matrix precomputation")
    print(f"   • On-demand distance calculation with caching")
    print(f"   • Progress indicators throughout")
    print(f"   • Should handle large networks much better")

def main():
    """Main test function"""
    test_fast_solver()
    
    print("\n" + "=" * 50)
    print("🎯 Fast Solver Test Complete")
    print("\n📋 If this test worked quickly, the CLI should now be responsive!")

if __name__ == "__main__":
    main()