#!/usr/bin/env python3
"""
Create a 3D plot of the street network with elevation data

Usage:
    python plot_3d_streets.py [--dist DISTANCE] [--exaggeration FACTOR] [--network-type TYPE]

Arguments:
    --dist, -d          Network radius in meters (default: 800)
    --exaggeration, -e  Vertical exaggeration factor (default: 10)
    --network-type, -t  Network type: all, drive, walk, bike (default: all)
    --help, -h          Show this help message

Examples:
    python plot_3d_streets.py                           # Default: 800m, 10x exaggeration
    python plot_3d_streets.py -d 1200 -e 15            # 1200m radius, 15x exaggeration
    python plot_3d_streets.py --dist 600 --network-type drive  # 600m, drive roads only
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import osmnx as ox
import argparse
import sys
from route import add_elevation_to_graph

def plot_3d_street_network(dist=800, exaggeration=10, network_type='all'):
    """Create a 3D visualization of the street network with elevation"""
    print("=== 3D Street Network Visualization ===")
    print(f"Parameters: distance={dist}m, exaggeration={exaggeration}x, network_type={network_type}")
    
    # Download street network
    print("1. Downloading street network...")
    center_point = (37.1299, -80.4094)  # Christiansburg, VA
    graph = ox.graph_from_point(center_point, dist=dist, network_type=network_type)
    print(f"   Downloaded {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    # Add real elevation data
    print("2. Adding elevation data...")
    srtm_file = 'srtm_20_05.tif'
    graph = add_elevation_to_graph(graph, srtm_file)
    
    # Extract coordinates and elevations
    print("3. Extracting 3D coordinates...")
    lats = []
    lons = []
    elevations = []
    
    for node_id, data in graph.nodes(data=True):
        lats.append(data['y'])
        lons.append(data['x'])
        elevations.append(data.get('elevation', 0))
    
    # Convert to numpy arrays
    lats = np.array(lats)
    lons = np.array(lons)
    elevs = np.array(elevations)
    
    # Apply elevation exaggeration
    base_elevation = np.min(elevs)
    exaggerated_elevs = (elevs - base_elevation) * exaggeration + base_elevation
    
    print(f"   Elevation range: {np.min(elevs):.1f}m to {np.max(elevs):.1f}m")
    print(f"   Exaggerated range: {np.min(exaggerated_elevs):.1f}m to {np.max(exaggerated_elevs):.1f}m")
    
    # Create 3D plot
    print("4. Creating 3D visualization...")
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot nodes with elevation coloring
    scatter = ax.scatter(lons, lats, exaggerated_elevs, 
                        c=elevs, cmap='terrain', s=8, alpha=0.7)
    
    # Plot edges as lines (sample every 3rd edge for performance)
    print("5. Adding street connections...")
    edge_count = 0
    max_edges = min(len(graph.edges), 2000)  # Limit for performance
    
    for i, (u, v, data) in enumerate(graph.edges(data=True)):
        if i % 3 == 0 and edge_count < max_edges:  # Sample every 3rd edge
            if u in graph.nodes and v in graph.nodes:
                # Get coordinates for both nodes
                u_data = graph.nodes[u]
                v_data = graph.nodes[v]
                
                u_lat, u_lon = u_data['y'], u_data['x']
                v_lat, v_lon = v_data['y'], v_data['x']
                u_elev = (u_data.get('elevation', 0) - base_elevation) * exaggeration + base_elevation
                v_elev = (v_data.get('elevation', 0) - base_elevation) * exaggeration + base_elevation
                
                # Draw line between nodes
                ax.plot([u_lon, v_lon], [u_lat, v_lat], [u_elev, v_elev], 
                       'gray', alpha=0.4, linewidth=0.8)
                edge_count += 1
    
    print(f"   Plotted {edge_count} edges (sampled for performance)")
    
    # Customize the plot
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude') 
    ax.set_zlabel(f'Elevation (m) - {exaggeration}x Exaggerated')
    ax.set_title(f'3D Street Network - Christiansburg, VA\n({exaggeration}x Vertical Exaggeration, {dist}m radius)', 
                fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20, pad=0.1)
    cbar.set_label('Actual Elevation (m)', rotation=270, labelpad=20)
    
    # Set viewing angle for better perspective
    ax.view_init(elev=20, azim=45)
    
    # Make the plot look better
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"""
Street Network Statistics:
• Nodes: {len(graph.nodes):,}
• Edges: {len(graph.edges):,}
• Network Type: {network_type}
• Radius: {dist}m
• Elevation Range: {np.min(elevs):.1f}m - {np.max(elevs):.1f}m
• Elevation Span: {np.max(elevs) - np.min(elevs):.1f}m
• Vertical Exaggeration: {exaggeration}x
"""
    
    # Add text box with statistics
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='white', alpha=0.8), fontsize=10, fontfamily='monospace')
    
    # Save the plot
    plt.tight_layout()
    filename_base = f'3d_street_network_{dist}m_{exaggeration}x_{network_type}'
    plt.savefig(f'{filename_base}.png', dpi=300, bbox_inches='tight')
    print(f"6. Saved 3D visualization to '{filename_base}.png'")
    
    # Create a second view from a different angle
    print("7. Creating additional view...")
    ax.view_init(elev=60, azim=135)
    plt.savefig(f'{filename_base}_topview.png', dpi=300, bbox_inches='tight')
    print(f"   Saved top view to '{filename_base}_topview.png'")
    
    # Show interactive plot
    plt.show()
    
    print("\n=== 3D Visualization Complete ===")
    print(f"✓ Real elevation data with {exaggeration}x vertical exaggeration")
    print(f"✓ Street network overlaid on topography ({dist}m radius)")
    print(f"✓ Network type: {network_type}")
    print("✓ Color-coded elevation mapping")
    print("✓ Interactive 3D visualization")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Create a 3D plot of the street network with elevation data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_3d_streets.py                           # Default: 800m, 10x exaggeration
  python plot_3d_streets.py -d 1200 -e 15            # 1200m radius, 15x exaggeration
  python plot_3d_streets.py --dist 600 --network-type drive  # 600m, drive roads only
  python plot_3d_streets.py -d 2000 -e 5 -t all     # Large area, subtle exaggeration
        """
    )
    
    parser.add_argument(
        '--dist', '-d', 
        type=int, 
        default=800,
        help='Network radius in meters (default: 800)'
    )
    
    parser.add_argument(
        '--exaggeration', '-e',
        type=float,
        default=10.0,
        help='Vertical exaggeration factor (default: 10.0)'
    )
    
    parser.add_argument(
        '--network-type', '-t',
        type=str,
        default='all',
        choices=['all', 'drive', 'walk', 'bike'],
        help='Network type: all, drive, walk, bike (default: all)'
    )
    
    return parser.parse_args()

def main():
    """Main function with command line argument parsing"""
    try:
        args = parse_arguments()
        
        # Validate arguments
        if args.dist <= 0:
            print("Error: Distance must be positive")
            sys.exit(1)
        
        if args.exaggeration <= 0:
            print("Error: Exaggeration factor must be positive")
            sys.exit(1)
        
        # Call the plotting function with parsed arguments
        plot_3d_street_network(
            dist=args.dist,
            exaggeration=args.exaggeration,
            network_type=args.network_type
        )
        
    except KeyboardInterrupt:
        print("\nVisualization cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()