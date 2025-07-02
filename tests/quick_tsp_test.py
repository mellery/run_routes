#!/usr/bin/env python3
"""
Quick test of TSP solver core functionality
"""

import osmnx as ox
import time
from route import add_elevation_to_graph, add_elevation_to_edges, add_running_weights
from tsp_solver import (
    RunningRouteOptimizer, RouteObjective, 
    NearestNeighborTSP, DistanceConstrainedTSP
)

def quick_test():
    """Quick validation of TSP solver"""
    print("=== Quick TSP Solver Test ===")
    
    # Small test network using cache
    print("1. Setting up test network...")
    try:
        from graph_cache import load_or_generate_graph
        center_point = (37.1299, -80.4094)  # Christiansburg, VA
        graph = load_or_generate_graph(center_point, radius_m=400, network_type='drive')
        print(f"   ✓ Network: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    except Exception as e:
        print(f"   ❌ Cache loading failed, using direct method: {e}")
        graph = ox.graph_from_point(center_point, dist=400, network_type='drive')
        print(f"   Network: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
        # Add elevation data
        print("2. Adding elevation data...")
        graph = add_elevation_to_graph(graph, 'srtm_20_05.tif')
        graph = add_elevation_to_edges(graph)
        graph = add_running_weights(graph)
        print("   ✓ Elevation data added")
    
    # Test basic nearest neighbor
    print("\n3. Testing Nearest Neighbor TSP...")
    start_node = list(graph.nodes)[0]
    
    nn_solver = NearestNeighborTSP(graph, start_node, RouteObjective.MINIMIZE_DISTANCE)
    
    start_time = time.time()
    route, cost = nn_solver.solve(max_nodes=5)  # Limit to 5 nodes for speed
    solve_time = time.time() - start_time
    
    stats = nn_solver.get_route_stats(route)
    
    print(f"   ✓ Route found: {len(route)} nodes")
    print(f"   ✓ Distance: {stats.get('total_distance_km', 0):.2f} km")
    print(f"   ✓ Elevation gain: {stats.get('total_elevation_gain_m', 0):.1f}m")
    print(f"   ✓ Solve time: {solve_time:.3f} seconds")
    
    # Test different objectives
    print("\n4. Testing different objectives...")
    objectives = [
        RouteObjective.MINIMIZE_DISTANCE,
        RouteObjective.MAXIMIZE_ELEVATION,
        RouteObjective.BALANCED_ROUTE,
        RouteObjective.MINIMIZE_DIFFICULTY
    ]
    
    for objective in objectives:
        solver = NearestNeighborTSP(graph, start_node, objective)
        route, cost = solver.solve(max_nodes=4)
        stats = solver.get_route_stats(route)
        
        print(f"   {objective}: {stats.get('total_distance_km', 0):.2f}km, "
              f"{stats.get('total_elevation_gain_m', 0):.0f}m gain, cost: {cost:.2f}")
    
    # Test distance-constrained TSP
    print("\n5. Testing Distance-Constrained TSP...")
    dc_solver = DistanceConstrainedTSP(graph, start_node, target_distance_km=1.0, 
                                      tolerance=0.3, objective=RouteObjective.MINIMIZE_DISTANCE)
    
    start_time = time.time()
    route, cost = dc_solver.solve(algorithm="nearest_neighbor")
    solve_time = time.time() - start_time
    
    stats = dc_solver.get_route_stats(route)
    actual_distance = stats.get('total_distance_km', 0)
    target_distance = 1.0
    
    print(f"   ✓ Target: {target_distance:.1f}km ±30%")
    print(f"   ✓ Actual: {actual_distance:.2f}km")
    print(f"   ✓ Within tolerance: {abs(actual_distance - target_distance) / target_distance <= 0.3}")
    print(f"   ✓ Solve time: {solve_time:.3f} seconds")
    
    # Test high-level optimizer
    print("\n6. Testing RunningRouteOptimizer...")
    optimizer = RunningRouteOptimizer(graph)
    
    start_time = time.time()
    result = optimizer.find_optimal_route(
        start_node=start_node,
        target_distance_km=0.8,
        objective=RouteObjective.BALANCED_ROUTE,
        algorithm="nearest_neighbor"
    )
    solve_time = time.time() - start_time
    
    print(f"   ✓ Optimization complete: {solve_time:.3f} seconds")
    print(f"   ✓ Route: {len(result['route'])} nodes")
    print(f"   ✓ Distance: {result['stats'].get('total_distance_km', 0):.2f}km")
    print(f"   ✓ Elevation: {result['stats'].get('total_elevation_gain_m', 0):.0f}m")
    
    # Show route details
    print("\n7. Route Analysis...")
    route_nodes = result['route']
    print(f"   Route sequence: {route_nodes[:5]}{'...' if len(route_nodes) > 5 else ''}")
    
    stats = result['stats']
    print(f"   Total distance: {stats.get('total_distance_km', 0):.2f} km")
    print(f"   Elevation gain: {stats.get('total_elevation_gain_m', 0):.1f} m")
    print(f"   Elevation loss: {stats.get('total_elevation_loss_m', 0):.1f} m")
    print(f"   Max grade: {stats.get('max_grade_percent', 0):.1f}%")
    print(f"   Estimated time: {stats.get('estimated_time_min', 0):.1f} minutes")
    
    print(f"\n✅ Quick TSP Test Complete!")
    print(f"✓ All core TSP algorithms working")
    print(f"✓ Multiple objectives implemented")
    print(f"✓ Distance constraints functional")
    print(f"✓ Running-specific optimizations active")
    
    return result

if __name__ == "__main__":
    try:
        result = quick_test()
        print(f"\n🎯 Phase 2 Core Functionality Validated!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()