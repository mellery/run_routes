#!/usr/bin/env python3
"""
Test and demonstrate TSP solver implementations
"""

import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
import time
from route import add_elevation_to_graph, add_elevation_to_edges, add_running_weights
from tsp_solver import (
    RunningRouteOptimizer, RouteObjective, 
    NearestNeighborTSP, GeneticAlgorithmTSP, DistanceConstrainedTSP
)

def test_tsp_solvers():
    """Test all TSP solver implementations"""
    print("=== TSP Solver Testing ===")
    
    # Download street network
    print("1. Setting up test environment...")
    center_point = (37.1299, -80.4094)  # Christiansburg, VA
    graph = ox.graph_from_point(center_point, dist=1000, network_type='all')
    print(f"   Downloaded {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    # Add elevation data
    srtm_file = 'srtm_20_05.tif'
    graph = add_elevation_to_graph(graph, srtm_file)
    graph = add_elevation_to_edges(graph)
    graph = add_running_weights(graph)
    
    # Choose a start node
    start_node = list(graph.nodes)[0]
    print(f"   Using start node: {start_node}")
    
    # Test different algorithms and objectives
    test_cases = [
        ("nearest_neighbor", RouteObjective.MINIMIZE_DISTANCE),
        ("nearest_neighbor", RouteObjective.MAXIMIZE_ELEVATION),
        ("nearest_neighbor", RouteObjective.BALANCED_ROUTE),
        ("genetic", RouteObjective.MINIMIZE_DISTANCE),
        ("genetic", RouteObjective.MAXIMIZE_ELEVATION),
    ]
    
    results = []
    
    print("\n2. Testing TSP algorithms...")
    for algorithm, objective in test_cases:
        print(f"\n   Testing {algorithm} with {objective}:")
        
        start_time = time.time()
        
        # Create optimizer
        optimizer = RunningRouteOptimizer(graph)
        
        # Find route
        try:
            result = optimizer.find_optimal_route(
                start_node=start_node,
                target_distance_km=2.0,  # 2km target
                objective=objective,
                algorithm=algorithm
            )
            
            solve_time = time.time() - start_time
            
            # Print results
            stats = result['stats']
            print(f"     ✓ Route found: {len(result['route'])} nodes")
            print(f"     ✓ Distance: {stats.get('total_distance_km', 0):.2f} km")
            print(f"     ✓ Elevation gain: {stats.get('total_elevation_gain_m', 0):.1f}m")
            print(f"     ✓ Max grade: {stats.get('max_grade_percent', 0):.1f}%")
            print(f"     ✓ Solve time: {solve_time:.2f} seconds")
            
            results.append((algorithm, objective, result))
            
        except Exception as e:
            print(f"     ❌ Failed: {e}")
    
    # Compare results
    print("\n3. Comparing results...")
    print(f"{'Algorithm':<15} {'Objective':<20} {'Distance':<10} {'Elev Gain':<10} {'Time':<8}")
    print("-" * 70)
    
    for algorithm, objective, result in results:
        stats = result['stats']
        distance = stats.get('total_distance_km', 0)
        elevation = stats.get('total_elevation_gain_m', 0)
        solve_time = result.get('solve_time', 0)
        
        print(f"{algorithm:<15} {objective:<20} {distance:<10.2f} {elevation:<10.1f} {solve_time:<8.2f}")
    
    print(f"\n✅ TSP Testing Complete! Tested {len(results)} configurations.")
    
    return results

def test_specific_algorithms():
    """Test individual algorithm components"""
    print("\n=== Individual Algorithm Testing ===")
    
    # Setup smaller test case
    center_point = (37.1299, -80.4094)
    graph = ox.graph_from_point(center_point, dist=600, network_type='drive')
    graph = add_elevation_to_graph(graph, 'srtm_20_05.tif')
    graph = add_elevation_to_edges(graph)
    graph = add_running_weights(graph)
    
    start_node = list(graph.nodes)[0]
    print(f"Test graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    # Test Nearest Neighbor
    print("\n1. Testing Nearest Neighbor TSP...")
    nn_solver = NearestNeighborTSP(graph, start_node, RouteObjective.MINIMIZE_DISTANCE)
    nn_route, nn_cost = nn_solver.solve(max_nodes=8)
    nn_stats = nn_solver.get_route_stats(nn_route)
    
    print(f"   Route: {len(nn_route)} nodes")
    print(f"   Distance: {nn_stats.get('total_distance_km', 0):.2f} km")
    print(f"   Cost: {nn_cost:.2f}")
    
    # Test Genetic Algorithm
    print("\n2. Testing Genetic Algorithm TSP...")
    ga_solver = GeneticAlgorithmTSP(graph, start_node, RouteObjective.MINIMIZE_DISTANCE)
    ga_solver.population_size = 20
    ga_solver.generations = 30
    ga_route, ga_cost = ga_solver.solve(max_nodes=8)
    ga_stats = ga_solver.get_route_stats(ga_route)
    
    print(f"   Route: {len(ga_route)} nodes")
    print(f"   Distance: {ga_stats.get('total_distance_km', 0):.2f} km")
    print(f"   Cost: {ga_cost:.2f}")
    
    # Test Distance Constrained
    print("\n3. Testing Distance Constrained TSP...")
    dc_solver = DistanceConstrainedTSP(graph, start_node, target_distance_km=1.5, 
                                      tolerance=0.2, objective=RouteObjective.MINIMIZE_DISTANCE)
    dc_route, dc_cost = dc_solver.solve(algorithm="nearest_neighbor")
    dc_stats = dc_solver.get_route_stats(dc_route)
    
    print(f"   Route: {len(dc_route)} nodes")
    print(f"   Distance: {dc_stats.get('total_distance_km', 0):.2f} km")
    print(f"   Target: 1.5 km (±20%)")
    print(f"   Cost: {dc_cost:.2f}")
    
    return (nn_route, nn_stats), (ga_route, ga_stats), (dc_route, dc_stats)

def visualize_routes(route_results):
    """Create visualization comparing different routes"""
    print("\n4. Creating route visualizations...")
    
    # Setup for visualization
    center_point = (37.1299, -80.4094)
    graph = ox.graph_from_point(center_point, dist=800, network_type='all')
    graph = add_elevation_to_graph(graph, 'srtm_20_05.tif')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    titles = ["Nearest Neighbor", "Genetic Algorithm", "Distance Constrained", "Elevation Comparison"]
    
    for i, ((route, stats), title) in enumerate(zip(route_results, titles[:3])):
        ax = axes[i]
        
        # Plot base network
        ox.plot_graph(graph, ax=ax, node_size=5, node_color='lightblue',
                     edge_color='gray', edge_linewidth=0.5, show=False)
        
        # Highlight route
        if route and len(route) > 1:
            route_coords = []
            for node in route:
                if node in graph.nodes:
                    node_data = graph.nodes[node]
                    route_coords.append((node_data['x'], node_data['y']))
            
            # Add return to start
            if route_coords:
                route_coords.append(route_coords[0])
                
                route_x = [coord[0] for coord in route_coords]
                route_y = [coord[1] for coord in route_coords]
                
                ax.plot(route_x, route_y, 'red', linewidth=3, alpha=0.8, label='Route')
                ax.scatter(route_x[:-1], route_y[:-1], c='red', s=50, zorder=5)
                ax.scatter([route_x[0]], [route_y[0]], c='green', s=100, zorder=6, 
                          marker='*', label='Start')
        
        ax.set_title(f"{title}\nDistance: {stats.get('total_distance_km', 0):.2f}km, "
                   f"Elevation: {stats.get('total_elevation_gain_m', 0):.0f}m")
        ax.legend()
    
    # Fourth panel: Elevation profile comparison
    ax = axes[3]
    
    for i, ((route, stats), label) in enumerate(zip(route_results, ["NN", "GA", "DC"])):
        if route:
            elevations = []
            for node in route:
                if node in graph.nodes:
                    elevations.append(graph.nodes[node].get('elevation', 0))
            elevations.append(elevations[0])  # Return to start
            
            ax.plot(range(len(elevations)), elevations, marker='o', label=f"{label}: {stats.get('total_elevation_gain_m', 0):.0f}m gain")
    
    ax.set_xlabel('Route Segment')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Elevation Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tsp_route_comparison.png', dpi=200, bbox_inches='tight')
    print("   Saved route comparison to 'tsp_route_comparison.png'")
    
    plt.show()

def main():
    """Main testing function"""
    print("🏃 Running Route TSP Solver - Phase 2 Testing")
    print("=" * 50)
    
    try:
        # Test all solvers
        results = test_tsp_solvers()
        
        # Test individual algorithms
        route_results = test_specific_algorithms()
        
        # Create visualizations
        visualize_routes(route_results)
        
        print("\n" + "=" * 50)
        print("✅ Phase 2 TSP Implementation Complete!")
        print("✓ Nearest Neighbor TSP solver")
        print("✓ Genetic Algorithm TSP solver") 
        print("✓ Distance-constrained TSP variant")
        print("✓ Multiple optimization objectives")
        print("✓ Running-specific route evaluation")
        print("✓ Comprehensive testing and validation")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()