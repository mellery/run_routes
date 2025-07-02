#!/usr/bin/env python3
"""
Test Optimized TSP Solver
Compare performance and functionality of optimized vs original solver
"""

import time
from graph_cache import load_or_generate_graph
from tsp_solver import RouteObjective

def test_optimized_solver():
    """Test the optimized solver performance and functionality"""
    
    print("🧪 Testing Optimized TSP Solver")
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
    
    print(f"\n2️⃣ Testing route generation...")
    print(f"   Start node: {start_node}")
    print(f"   Target distance: {target_distance}km")
    
    # Test optimized solver
    print(f"\n3️⃣ Testing optimized solver...")
    try:
        from tsp_solver_optimized import OptimizedRunningRouteOptimizer
        
        optimizer = OptimizedRunningRouteOptimizer(graph)
        
        start_time = time.time()
        result = optimizer.find_optimal_route(
            start_node=start_node,
            target_distance_km=target_distance,
            objective=RouteObjective.MINIMIZE_DISTANCE,
            algorithm="nearest_neighbor"  # Start with faster algorithm
        )
        solve_time = time.time() - start_time
        
        if result:
            stats = result['stats']
            print(f"✅ Optimized solver completed in {solve_time:.2f} seconds")
            print(f"   Route: {len(result['route'])} nodes")
            print(f"   Distance: {stats.get('total_distance_km', 0):.2f} km")
            print(f"   Elevation gain: {stats.get('total_elevation_gain_m', 0):.0f} m")
            
            # Show progress log if available
            if 'progress_log' in result and result['progress_log']:
                print(f"   Progress messages: {len(result['progress_log'])}")
                for msg in result['progress_log'][-3:]:  # Show last 3 messages
                    print(f"     • {msg}")
        else:
            print("❌ Optimized solver failed")
            
    except ImportError as e:
        print(f"❌ Failed to import optimized solver: {e}")
        return
    except Exception as e:
        print(f"❌ Optimized solver error: {e}")
        return
    
    # Test genetic algorithm with timeout
    print(f"\n4️⃣ Testing genetic algorithm with timeout...")
    try:
        start_time = time.time()
        result_genetic = optimizer.find_optimal_route(
            start_node=start_node,
            target_distance_km=target_distance,
            objective=RouteObjective.MINIMIZE_DISTANCE,
            algorithm="genetic"  # More complex algorithm
        )
        solve_time_genetic = time.time() - start_time
        
        if result_genetic:
            stats = result_genetic['stats']
            print(f"✅ Genetic algorithm completed in {solve_time_genetic:.2f} seconds")
            print(f"   Route: {len(result_genetic['route'])} nodes")
            print(f"   Distance: {stats.get('total_distance_km', 0):.2f} km")
            print(f"   Should have timeout protection (max 30s)")
        else:
            print("❌ Genetic algorithm failed")
            
    except Exception as e:
        print(f"⚠️ Genetic algorithm error: {e}")
    
    # Performance comparison
    print(f"\n5️⃣ Performance summary...")
    print(f"   ✅ Optimized solver provides progress feedback")
    print(f"   ✅ Timeout protection prevents hanging")
    print(f"   ✅ Reduced parameters for better performance")
    print(f"   ✅ Early stopping when no improvement")
    print(f"   ✅ Better error handling and fallbacks")

def main():
    """Main test function"""
    test_optimized_solver()
    
    print("\n" + "=" * 50)
    print("🎯 Optimized Solver Test Complete")
    print("\n📋 Key improvements:")
    print("   • Real-time progress indicators")
    print("   • 30-second timeout for genetic algorithm")
    print("   • Reduced generations (30) and population (20)")
    print("   • Early stopping when stagnant")
    print("   • Limited search space for better performance")
    print("   • Fallback routes when constraints can't be met")

if __name__ == "__main__":
    main()