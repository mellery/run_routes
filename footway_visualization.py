#!/usr/bin/env python3
"""
Footway Filtering Visualization
Creates a map showing which segments get filtered out by the exclude-footways option
"""

import matplotlib.pyplot as plt
import numpy as np
from route_services import NetworkManager
import math

def create_footway_visualization():
    """Create visualization of footway filtering in downtown Christiansburg"""
    
    print("Creating footway filtering visualization for downtown Christiansburg...")
    
    # Downtown Christiansburg coordinates
    downtown_lat = 37.1299
    downtown_lon = -80.4094
    
    # Load network with 0.25 mile radius
    radius_miles = 0.25
    radius_km = radius_miles * 1.60934  # Convert to km
    
    print(f"Loading network with {radius_miles} mile ({radius_km:.2f}km) radius...")
    
    network_manager = NetworkManager(center_point=(downtown_lat, downtown_lon))
    graph = network_manager.load_network(radius_km=radius_km)
    
    if not graph:
        print("❌ Failed to load network")
        return
    
    print(f"✅ Loaded network: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    # Analyze segments by highway type
    segments_by_type = {}
    coordinates_by_type = {}
    
    for u, v, data in graph.edges(data=True):
        highway = data.get('highway', 'unknown')
        
        # Handle cases where highway might be a list
        if isinstance(highway, list):
            highway = highway[0] if highway else 'unknown'
        
        highway = str(highway)  # Ensure it's a string
        
        if highway not in segments_by_type:
            segments_by_type[highway] = 0
            coordinates_by_type[highway] = []
        
        segments_by_type[highway] += 1
        
        # Get coordinates for this edge
        u_data = graph.nodes[u]
        v_data = graph.nodes[v]
        
        coordinates_by_type[highway].append([
            [u_data['x'], v_data['x']],  # longitude
            [u_data['y'], v_data['y']]   # latitude
        ])
    
    print("\nSegment breakdown by highway type:")
    for highway, count in sorted(segments_by_type.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(graph.edges)) * 100
        status = "🚫 FILTERED" if highway == 'footway' else "✅ KEPT"
        print(f"  {highway}: {count} segments ({percentage:.1f}%) {status}")
    
    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Calculate map bounds
    all_lats = [data['y'] for _, data in graph.nodes(data=True)]
    all_lons = [data['x'] for _, data in graph.nodes(data=True)]
    
    lat_margin = (max(all_lats) - min(all_lats)) * 0.1
    lon_margin = (max(all_lons) - min(all_lons)) * 0.1
    
    bounds = [
        min(all_lons) - lon_margin,  # west
        max(all_lons) + lon_margin,  # east
        min(all_lats) - lat_margin,  # south
        max(all_lats) + lat_margin   # north
    ]
    
    # Plot 1: All segments (before filtering)
    ax1.set_title(f'Before Filtering: All Network Segments\n{len(graph.edges)} total segments', fontsize=14, fontweight='bold')
    
    # Plot different highway types with different colors
    colors = {
        'residential': 'blue',
        'primary': 'red', 
        'secondary': 'orange',
        'tertiary': 'green',
        'footway': 'purple',
        'service': 'gray',
        'unclassified': 'brown'
    }
    
    for highway, coords in coordinates_by_type.items():
        color = colors.get(highway, 'black')
        alpha = 0.8 if highway == 'footway' else 0.6
        linewidth = 2 if highway == 'footway' else 1
        
        for lon_pair, lat_pair in coords:
            ax1.plot(lon_pair, lat_pair, color=color, alpha=alpha, linewidth=linewidth)
    
    ax1.set_xlim(bounds[0], bounds[1])
    ax1.set_ylim(bounds[2], bounds[3])
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Add legend for plot 1
    legend_elements = []
    for highway, count in sorted(segments_by_type.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            color = colors.get(highway, 'black')
            alpha = 0.8 if highway == 'footway' else 0.6
            linewidth = 2 if highway == 'footway' else 1
            label = f'{highway} ({count})'
            legend_elements.append(plt.Line2D([0], [0], color=color, alpha=alpha, linewidth=linewidth, label=label))
    
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Plot 2: After filtering (footways removed)
    footway_count = segments_by_type.get('footway', 0)
    remaining_count = len(graph.edges) - footway_count
    
    ax2.set_title(f'After Filtering: Footways Removed\n{remaining_count} segments ({footway_count} footways filtered)', 
                  fontsize=14, fontweight='bold')
    
    # Plot everything except footways
    for highway, coords in coordinates_by_type.items():
        if highway != 'footway':  # Skip footways
            color = colors.get(highway, 'black')
            alpha = 0.6
            linewidth = 1
            
            for lon_pair, lat_pair in coords:
                ax2.plot(lon_pair, lat_pair, color=color, alpha=alpha, linewidth=linewidth)
    
    ax2.set_xlim(bounds[0], bounds[1])
    ax2.set_ylim(bounds[2], bounds[3])
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Add legend for plot 2 (without footways)
    legend_elements_filtered = []
    for highway, count in sorted(segments_by_type.items(), key=lambda x: x[1], reverse=True):
        if count > 0 and highway != 'footway':
            color = colors.get(highway, 'black')
            label = f'{highway} ({count})'
            legend_elements_filtered.append(plt.Line2D([0], [0], color=color, linewidth=1, label=label))
    
    ax2.legend(handles=legend_elements_filtered, loc='upper right', fontsize=8)
    
    # Add center point marker
    for ax in [ax1, ax2]:
        ax.plot(downtown_lon, downtown_lat, 'ko', markersize=8, markeredgecolor='yellow', markeredgewidth=2, label='Downtown Center')
    
    # Overall title
    fig.suptitle(f'Footway Filtering Visualization - Downtown Christiansburg, VA\n0.25 mile radius around ({downtown_lat:.4f}, {downtown_lon:.4f})', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the visualization
    filename = "downtown_christiansburg_footway_filtering.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✅ Visualization saved as: {filename}")
    
    # Create summary statistics
    print(f"\n📊 FILTERING IMPACT SUMMARY:")
    print(f"  Total area: 0.25 mile radius around downtown")
    print(f"  Total segments before: {len(graph.edges)}")
    print(f"  Footway segments: {footway_count} ({footway_count/len(graph.edges)*100:.1f}%)")
    print(f"  Segments after filtering: {remaining_count} ({remaining_count/len(graph.edges)*100:.1f}%)")
    print(f"  Reduction: {footway_count} segments removed")
    
    # Show most impacted areas
    if footway_count > 0:
        print(f"\n🚫 FILTERED SEGMENTS (footways):")
        print(f"  These {footway_count} segments will NOT be used in route planning")
        print(f"  Prevents redundant back-and-forth routing on parallel paths")
        print(f"  Can be re-enabled with --include-footways flag")

def create_interactive_map():
    """Create an interactive map with OpenStreetMap background"""
    try:
        import folium
        from route_services import NetworkManager
        
        print("Creating interactive map with OpenStreetMap background...")
        
        # Downtown Christiansburg coordinates
        downtown_lat = 37.1299
        downtown_lon = -80.4094
        radius_miles = 0.25
        radius_km = radius_miles * 1.60934
        
        # Load network
        network_manager = NetworkManager(center_point=(downtown_lat, downtown_lon))
        graph = network_manager.load_network(radius_km=radius_km)
        
        # Create folium map
        m = folium.Map(
            location=[downtown_lat, downtown_lon],
            zoom_start=16,
            tiles='OpenStreetMap'
        )
        
        # Add segments by type
        footway_count = 0
        other_count = 0
        
        for u, v, data in graph.edges(data=True):
            highway = data.get('highway', 'unknown')
            
            # Handle cases where highway might be a list
            if isinstance(highway, list):
                highway = highway[0] if highway else 'unknown'
            highway = str(highway)
            
            u_data = graph.nodes[u]
            v_data = graph.nodes[v]
            
            coords = [[u_data['y'], u_data['x']], [v_data['y'], v_data['x']]]
            
            if highway == 'footway':
                # Red for footways (will be filtered)
                folium.PolyLine(
                    coords,
                    color='red',
                    weight=3,
                    opacity=0.8,
                    popup=f"FILTERED: {highway}"
                ).add_to(m)
                footway_count += 1
            else:
                # Blue for kept segments
                color = 'blue' if highway in ['residential', 'primary', 'secondary', 'tertiary'] else 'green'
                folium.PolyLine(
                    coords,
                    color=color,
                    weight=2,
                    opacity=0.6,
                    popup=f"KEPT: {highway}"
                ).add_to(m)
                other_count += 1
        
        # Add center marker
        folium.Marker(
            [downtown_lat, downtown_lon],
            popup="Downtown Christiansburg Center",
            icon=folium.Icon(color='yellow', icon='star')
        ).add_to(m)
        
        # Add legend
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4>Footway Filtering</h4>
        <p><span style="color:red">●</span> Footways ({footway_count}) - FILTERED</p>
        <p><span style="color:blue">●</span> Roads ({other_count}) - KEPT</p>
        <p>0.25 mile radius</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save interactive map
        map_filename = "downtown_christiansburg_footway_map.html"
        m.save(map_filename)
        
        print(f"✅ Interactive map saved as: {map_filename}")
        print(f"   Red lines: {footway_count} footway segments (FILTERED)")
        print(f"   Blue/Green lines: {other_count} road segments (KEPT)")
        
    except ImportError:
        print("⚠️ Folium not available for interactive map")

if __name__ == "__main__":
    create_footway_visualization()
    create_interactive_map()