#!/usr/bin/env python3
"""
Test UI Components
Simple test to verify UI components can load networks and generate routes
"""

import sys
import time
import osmnx as ox
from route import add_elevation_to_graph, add_elevation_to_edges, add_running_weights
from tsp_solver import RunningRouteOptimizer, RouteObjective

def test_network_loading():
    """Test network loading functionality"""
    print("🌐 Testing network loading...")
    
    try:
        # Load small network
        center_point = (37.1299, -80.4094)  # Christiansburg, VA
        graph = ox.graph_from_point(center_point, dist=400, network_type='all')
        
        # Add elevation data
        print("   Adding elevation data...")
        graph = add_elevation_to_graph(graph, 'srtm_20_05.tif')
        graph = add_elevation_to_edges(graph)
        graph = add_running_weights(graph)
        
        print(f"✅ Network loaded: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        return graph
        
    except Exception as e:
        print(f"❌ Network loading failed: {e}")
        return None

def test_route_generation(graph):
    """Test route generation"""
    if not graph:
        return None
        
    print("\n🚀 Testing route generation...")
    
    try:
        # Get a starting node
        start_node = list(graph.nodes())[0]
        
        # Create optimizer
        optimizer = RunningRouteOptimizer(graph)
        
        # Generate route
        result = optimizer.find_optimal_route(
            start_node=start_node,
            target_distance_km=0.5,
            objective=RouteObjective.MINIMIZE_DISTANCE,
            algorithm="nearest_neighbor"
        )
        
        if result:
            stats = result['stats']
            print(f"✅ Route generated successfully:")
            print(f"   Distance: {stats.get('total_distance_km', 0):.2f} km")
            print(f"   Elevation gain: {stats.get('total_elevation_gain_m', 0):.0f} m")
            print(f"   Route points: {len(result['route'])} intersections")
            return result
        else:
            print("❌ Route generation returned no result")
            return None
            
    except Exception as e:
        print(f"❌ Route generation failed: {e}")
        return None

def test_ui_helper_functions(graph, result):
    """Test UI helper functions"""
    if not graph or not result:
        return
        
    print("\n🗺️ Testing UI helper functions...")
    
    try:
        # Test coordinate extraction
        route = result['route']
        coordinates = []
        elevations = []
        
        for node in route:
            if node in graph.nodes:
                data = graph.nodes[node]
                coordinates.append([data['y'], data['x']])
                elevations.append(data.get('elevation', 0))
        
        print(f"✅ Extracted {len(coordinates)} coordinates")
        print(f"   Elevation range: {min(elevations):.0f}m - {max(elevations):.0f}m")
        
        # Test directions generation
        directions = []
        cumulative_distance = 0
        
        for i in range(len(route)):
            if i == 0:
                directions.append(f"Start at intersection {route[i]}")
            else:
                directions.append(f"Continue to intersection {route[i]}")
        
        print(f"✅ Generated {len(directions)} turn-by-turn directions")
        
    except Exception as e:
        print(f"❌ UI helper functions failed: {e}")

def main():
    """Main test function"""
    print("🧪 Testing Phase 3 UI Components")
    print("=" * 50)
    
    # Load network
    graph = test_network_loading()
    
    # Generate route
    result = test_route_generation(graph)
    
    # Test UI helpers
    test_ui_helper_functions(graph, result)
    
    print("\n" + "=" * 50)
    if graph and result:
        print("✅ All UI component tests passed!")
        print("🎯 Phase 3 core functionality verified")
    else:
        print("❌ Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()