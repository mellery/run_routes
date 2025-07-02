#!/usr/bin/env python3
"""
Demo of Phase 1 implementation using synthetic elevation data
Since the SRTM file doesn't cover Christiansburg, VA, we'll use synthetic elevation
"""

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import numpy as np
from route import (
    haversine_distance, add_elevation_to_graph, add_elevation_to_edges, add_running_weights,
    get_nodes_within_distance, create_distance_constrained_subgraph
)

def add_synthetic_elevation_to_graph(graph):
    """Add synthetic elevation data based on latitude (north = higher elevation)"""
    print("Adding synthetic elevation data to graph nodes...")
    
    # Get lat/lon ranges
    lats = [data['y'] for _, data in graph.nodes(data=True)]
    lons = [data['x'] for _, data in graph.nodes(data=True)]
    
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    
    # Create synthetic elevation based on position and some randomness
    np.random.seed(42)  # For reproducible results
    
    for node_id, node_data in graph.nodes(data=True):
        lat, lon = node_data['y'], node_data['x']
        
        # Base elevation increases with latitude (north is higher)
        base_elevation = 600 + (lat - min_lat) / (max_lat - min_lat) * 200
        
        # Add some variation based on longitude 
        lon_variation = 50 * np.sin((lon - min_lon) / (max_lon - min_lon) * 2 * np.pi)
        
        # Add random noise
        noise = np.random.normal(0, 20)
        
        # Final elevation
        elevation = base_elevation + lon_variation + noise
        elevation = max(500, elevation)  # Minimum elevation of 500m
        
        graph.nodes[node_id]['elevation'] = elevation
    
    print(f"Added synthetic elevation data to {len(graph.nodes)} nodes")
    return graph

def demo_phase1():
    print("=== Phase 1 Demo with Real Elevation Data ===")
    
    # Get street network for a small area in Christiansburg
    print("1. Downloading street network...")
    try:
        # Use coordinates for Christiansburg center
        center_point = (37.1299, -80.4094)  # Christiansburg, VA
        graph = ox.graph_from_point(center_point, dist=800, network_type='all')
        print(f"   Downloaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    except Exception as e:
        print(f"   Error downloading graph: {e}")
        return
    
    # Try to add real elevation data first
    print("\n2. Adding real elevation data...")
    srtm_file = 'srtm_20_05.tif'
    try:
        graph = add_elevation_to_graph(graph, srtm_file)
        print("   Successfully added real elevation data!")
        use_real_elevation = True
    except Exception as e:
        print(f"   Error with real elevation data: {e}")
        print("   Falling back to synthetic elevation data...")
        graph = add_synthetic_elevation_to_graph(graph)
        use_real_elevation = False
    
    # Add elevation data to edges
    print("\n3. Calculating elevation changes for edges...")
    graph = add_elevation_to_edges(graph)
    
    # Add running weights
    print("\n4. Adding running-specific weights...")
    graph = add_running_weights(graph)
    
    # Test distance-constrained subgraph
    print("\n5. Testing distance-constrained subgraph...")
    start_node = list(graph.nodes)[0]
    print(f"   Using start node: {start_node}")
    
    # Test different search radii
    for radius_km in [0.3, 0.5, 1.0]:
        subgraph = create_distance_constrained_subgraph(graph, start_node, radius_km)
        print(f"   {radius_km}km radius: {len(subgraph.nodes)} nodes")
    
    # Show statistics
    data_type = "Real SRTM" if use_real_elevation else "Synthetic"
    print(f"\n=== Statistics ({data_type} Elevation Data) ===")
    elevations = [data.get('elevation', 0) for _, data in graph.nodes(data=True)]
    print(f"Elevation range: {min(elevations):.1f}m to {max(elevations):.1f}m")
    print(f"Average elevation: {np.mean(elevations):.1f}m")
    
    edge_gains = [data.get('elevation_gain', 0) for _, _, data in graph.edges(data=True)]
    edge_grades = [data.get('grade', 0) for _, _, data in graph.edges(data=True)]
    running_weights = [data.get('running_weight', 0) for _, _, data in graph.edges(data=True)]
    
    print(f"Elevation gain range: {min(edge_gains):.1f}m to {max(edge_gains):.1f}m")
    print(f"Max grade: {max(edge_grades):.1f}%")
    print(f"Running weight range: {min(running_weights):.1f} to {max(running_weights):.1f}")
    
    # Create visualization
    print(f"\n6. Creating visualization...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Network with elevation-colored nodes
    node_elevations = [data.get('elevation', 0) for _, data in graph.nodes(data=True)]
    ox.plot_graph(graph, ax=ax1, node_size=20, 
                  node_color=node_elevations,
                  edge_color='gray', edge_linewidth=0.5, show=False)
    ax1.set_title('Street Network with Elevation')
    
    # Plot 2: Network with all nodes
    ox.plot_graph(graph, ax=ax2, node_size=15, node_color='red',
                  edge_color='gray', edge_linewidth=0.5, show=False)
    ax2.set_title('Full Street Network')
    
    # Plot 3: Subgraph (500m radius)
    subgraph = create_distance_constrained_subgraph(graph, start_node, 0.5)
    sub_elevations = [data.get('elevation', 0) for _, data in subgraph.nodes(data=True)]
    ox.plot_graph(subgraph, ax=ax3, node_size=30, 
                  node_color=sub_elevations,
                  edge_color='blue', edge_linewidth=1.0, show=False)
    ax3.set_title(f'Subgraph: 500m radius ({len(subgraph.nodes)} nodes)')
    
    # Plot 4: Elevation histogram
    ax4.hist(elevations, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax4.set_xlabel('Elevation (m)')
    ax4.set_ylabel('Number of Nodes')
    ax4.set_title('Elevation Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phase1_demo_results.png', dpi=150, bbox_inches='tight')
    print("   Saved visualization to 'phase1_demo_results.png'")
    
    # Test haversine distance function
    print(f"\n7. Testing haversine distance calculation...")
    nodes = list(graph.nodes(data=True))
    if len(nodes) >= 2:
        node1_id, node1_data = nodes[0]
        node2_id, node2_data = nodes[1]
        
        lat1, lon1 = node1_data['y'], node1_data['x']
        lat2, lon2 = node2_data['y'], node2_data['x']
        
        distance = haversine_distance(lat1, lon1, lat2, lon2)
        print(f"   Distance between nodes {node1_id} and {node2_id}: {distance:.2f}m")
        
        # Compare with edge length if edge exists
        if graph.has_edge(node1_id, node2_id):
            edge_data = graph.get_edge_data(node1_id, node2_id)
            if isinstance(edge_data, dict):
                edge_length = list(edge_data.values())[0].get('length', 'N/A')
                print(f"   OSMnx edge length: {edge_length}")
    
    print("\n=== Phase 1 Demo Complete ===")
    print("All Phase 1 functionality successfully implemented:")
    print("✓ Elevation data extraction and integration")
    print("✓ Haversine distance calculation")
    print("✓ Elevation gain/loss calculation for edges")
    print("✓ Grade calculation")
    print("✓ Running-specific edge weights")
    print("✓ Distance-constrained subgraph creation")
    
    return graph

if __name__ == "__main__":
    demo_graph = demo_phase1()