#!/usr/bin/env python3
"""
Test script for Phase 1 implementation
Tests elevation integration and distance calculations on a smaller subset
"""

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import numpy as np
import rasterio
from math import radians, cos, sin, asin, sqrt

# Import functions from route.py
from route import (
    haversine_distance, get_elevation_from_raster, add_elevation_to_graph,
    add_elevation_to_edges, add_running_weights, get_nodes_within_distance,
    create_distance_constrained_subgraph
)

def test_phase1():
    print("=== Phase 1 Implementation Test ===")
    
    # Get a smaller street network for testing
    print("1. Downloading smaller street network...")
    # Use a smaller area for testing
    graph = ox.graph_from_address("100 N Main St, Christiansburg, VA", 
                                  dist=1000, network_type='all')
    print(f"   Downloaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    # Test haversine distance calculation
    print("\n2. Testing haversine distance calculation...")
    node_list = list(graph.nodes(data=True))
    if len(node_list) >= 2:
        node1, data1 = node_list[0]
        node2, data2 = node_list[1]
        distance = haversine_distance(data1['y'], data1['x'], data2['y'], data2['x'])
        print(f"   Distance between nodes {node1} and {node2}: {distance:.2f} meters")
    
    # Test elevation data extraction
    print("\n3. Testing elevation data extraction...")
    srtm_file = 'srtm_38_03.tif'
    try:
        # Test a few sample coordinates
        test_coords = [(37.13, -80.41), (37.14, -80.42), (37.15, -80.43)]
        for lat, lon in test_coords:
            elevation = get_elevation_from_raster(srtm_file, lat, lon)
            if elevation is not None:
                print(f"   Elevation at ({lat}, {lon}): {elevation:.1f}m")
            else:
                print(f"   No elevation data for ({lat}, {lon})")
    except Exception as e:
        print(f"   Error testing elevation extraction: {e}")
    
    # Add elevation data to graph
    print("\n4. Adding elevation data to graph...")
    graph = add_elevation_to_graph(graph, srtm_file)
    
    # Count nodes with elevation data
    nodes_with_elevation = sum(1 for _, data in graph.nodes(data=True) 
                              if data.get('elevation', 0) > 0)
    print(f"   Nodes with elevation data: {nodes_with_elevation}/{len(graph.nodes)}")
    
    # Add elevation data to edges
    print("\n5. Adding elevation data to edges...")
    graph = add_elevation_to_edges(graph)
    
    # Add running weights
    print("\n6. Adding running-specific weights...")
    graph = add_running_weights(graph)
    
    # Test distance-constrained subgraph
    print("\n7. Testing distance-constrained subgraph...")
    start_node = list(graph.nodes)[0]  # Use first node as start
    subgraph = create_distance_constrained_subgraph(graph, start_node, 0.5)  # 500m radius
    
    # Show statistics
    print(f"\n=== Final Statistics ===")
    elevations = [data.get('elevation', 0) for _, data in graph.nodes(data=True)]
    print(f"Elevation range: {min(elevations):.1f}m to {max(elevations):.1f}m")
    
    edge_gains = [data.get('elevation_gain', 0) for _, _, data in graph.edges(data=True)]
    edge_grades = [data.get('grade', 0) for _, _, data in graph.edges(data=True)]
    print(f"Max elevation gain: {max(edge_gains):.1f}m")
    print(f"Max grade: {max(edge_grades):.1f}%")
    
    # Simple visualization
    print(f"\n8. Creating visualization...")
    node_elevations = [data.get('elevation', 0) for _, data in graph.nodes(data=True)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Network with elevation-colored nodes
    ox.plot_graph(graph, ax=ax1, node_size=20, 
                  node_color=node_elevations, 
                  edge_color='gray', edge_linewidth=0.5, show=False)
    ax1.set_title('Street Network with Elevation')
    
    # Plot 2: Subgraph
    ox.plot_graph(subgraph, ax=ax2, node_size=30, node_color='red',
                  edge_color='blue', edge_linewidth=1, show=False)
    ax2.set_title(f'Subgraph (500m radius, {len(subgraph.nodes)} nodes)')
    
    plt.tight_layout()
    plt.savefig('phase1_test_results.png', dpi=150, bbox_inches='tight')
    print("   Saved visualization to 'phase1_test_results.png'")
    
    print("\n=== Phase 1 Test Complete ===")
    return graph

if __name__ == "__main__":
    test_graph = test_phase1()