#!/usr/bin/env python3
"""
Simple 3D plot of street network with elevation
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import osmnx as ox
from route import add_elevation_to_graph

def create_simple_3d_plot():
    """Create a simplified 3D visualization"""
    print("=== Simple 3D Street Visualization ===")
    
    # Use smaller network for speed
    print("1. Downloading compact street network...")
    center_point = (37.1299, -80.4094)  # Christiansburg, VA
    graph = ox.graph_from_point(center_point, dist=600, network_type='drive')  # Only roads
    print(f"   Downloaded {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    # Add elevation data
    print("2. Adding elevation data...")
    srtm_file = 'srtm_20_05.tif'
    graph = add_elevation_to_graph(graph, srtm_file)
    
    # Extract node data
    print("3. Processing coordinates...")
    nodes_data = []
    for node_id, data in graph.nodes(data=True):
        nodes_data.append({
            'id': node_id,
            'lat': data['y'],
            'lon': data['x'],
            'elev': data.get('elevation', 0)
        })
    
    # Convert to arrays
    lats = np.array([n['lat'] for n in nodes_data])
    lons = np.array([n['lon'] for n in nodes_data])
    elevs = np.array([n['elev'] for n in nodes_data])
    
    # Apply 10x exaggeration
    base_elev = np.min(elevs)
    exag_elevs = (elevs - base_elev) * 10 + base_elev
    
    print(f"   Elevation range: {np.min(elevs):.1f}m to {np.max(elevs):.1f}m")
    print(f"   Exaggerated: {np.min(exag_elevs):.1f}m to {np.max(exag_elevs):.1f}m")
    
    # Create 3D plot
    print("4. Creating 3D plot...")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot nodes
    scatter = ax.scatter(lons, lats, exag_elevs, c=elevs, cmap='terrain', 
                        s=20, alpha=0.8, edgecolors='black', linewidth=0.1)
    
    # Add selected major streets (sample edges)
    print("5. Adding street connections...")
    edge_lines = []
    node_lookup = {n['id']: i for i, n in enumerate(nodes_data)}
    
    for i, (u, v) in enumerate(list(graph.edges())[:500]):  # Limit to first 500 edges
        if u in node_lookup and v in node_lookup:
            u_idx = node_lookup[u]
            v_idx = node_lookup[v]
            
            # Draw line
            ax.plot([lons[u_idx], lons[v_idx]], 
                   [lats[u_idx], lats[v_idx]], 
                   [exag_elevs[u_idx], exag_elevs[v_idx]], 
                   'gray', alpha=0.6, linewidth=1)
    
    # Customize plot
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Elevation (m) - 10x Exaggerated')
    ax.set_title('3D Street Network - Christiansburg, VA\n10x Vertical Exaggeration', 
                fontsize=16, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=30)
    cbar.set_label('Actual Elevation (m)', rotation=270, labelpad=20)
    
    # Set good viewing angle
    ax.view_init(elev=25, azim=45)
    
    # Add stats
    stats = f"""Network Stats:
Nodes: {len(graph.nodes):,}
Edges: {len(graph.edges):,}
Elevation Span: {np.max(elevs) - np.min(elevs):.1f}m
Vertical Exaggeration: 10x"""
    
    ax.text2D(0.02, 0.98, stats, transform=ax.transAxes, 
             verticalalignment='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Save plot
    plt.tight_layout()
    plt.savefig('3d_streets_simple.png', dpi=200, bbox_inches='tight')
    print("6. Saved to '3d_streets_simple.png'")
    
    # Create wireframe version
    print("7. Creating wireframe version...")
    fig2 = plt.figure(figsize=(12, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    # Create a grid for wireframe
    lon_grid = np.linspace(lons.min(), lons.max(), 20)
    lat_grid = np.linspace(lats.min(), lats.max(), 20)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    
    # Interpolate elevation onto grid
    from scipy.interpolate import griddata
    ELEV = griddata((lons, lats), exag_elevs, (LON, LAT), method='linear')
    
    # Plot wireframe
    ax2.plot_wireframe(LON, LAT, ELEV, alpha=0.7, color='blue', linewidth=0.8)
    
    # Add some nodes
    ax2.scatter(lons[::10], lats[::10], exag_elevs[::10], 
               c=elevs[::10], cmap='terrain', s=30, alpha=0.9)
    
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_zlabel('Elevation (m) - 10x Exaggerated')
    ax2.set_title('3D Topographic Wireframe - Christiansburg, VA', fontsize=14)
    ax2.view_init(elev=30, azim=60)
    
    plt.tight_layout()
    plt.savefig('3d_wireframe.png', dpi=200, bbox_inches='tight')
    print("8. Saved wireframe to '3d_wireframe.png'")
    
    print("\n=== 3D Visualization Complete ===")
    print("✓ Created 3D street network plot")
    print("✓ Applied 10x vertical exaggeration")
    print("✓ Generated wireframe topography")
    print("✓ Real Virginia elevation data visualized")

if __name__ == "__main__":
    try:
        # Install scipy if needed
        import scipy.interpolate
    except ImportError:
        print("Installing scipy for interpolation...")
        import subprocess
        subprocess.check_call(["pip", "install", "scipy"])
        import scipy.interpolate
    
    create_simple_3d_plot()