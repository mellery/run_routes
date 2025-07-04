#!/usr/bin/env python3
"""
Visualize filtered nodes for a specific target distance with OpenStreetMap background
"""

import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point
import math
from route_services import NetworkManager, RouteOptimizer

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using haversine formula"""
    R = 6371000  # Earth radius in meters
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def get_filtered_nodes_for_distance(graph, start_node, target_distance_km):
    """Get filtered nodes for a specific target distance"""
    
    # Get intersection nodes (same logic as RouteOptimizer)
    all_intersections = []
    for node_id, node_data in graph.nodes(data=True):
        if graph.degree(node_id) != 2:
            all_intersections.append({
                'node_id': node_id,
                'lat': node_data['y'],
                'lon': node_data['x'],
                'highway': node_data.get('highway', 'none')
            })
    
    # Find real intersections vs artifacts
    real_intersections = []
    artifacts = []
    
    for node in all_intersections:
        if node['highway'] in ['crossing', 'traffic_signals', 'stop', 'mini_roundabout']:
            real_intersections.append(node)
        else:
            artifacts.append(node)
    
    # Remove artifacts within 20m of real intersections
    proximity_threshold_m = 20.0
    artifacts_after_real_filtering = []
    
    for artifact in artifacts:
        too_close_to_real = False
        for real_node in real_intersections:
            distance = haversine_distance(
                artifact['lat'], artifact['lon'],
                real_node['lat'], real_node['lon']
            )
            if distance < proximity_threshold_m:
                too_close_to_real = True
                break
        
        if not too_close_to_real:
            artifacts_after_real_filtering.append(artifact)
    
    # Remove artifacts within 20m of other artifacts
    final_kept_artifacts = []
    
    for artifact in artifacts_after_real_filtering:
        too_close_to_kept = False
        for kept_artifact in final_kept_artifacts:
            distance = haversine_distance(
                artifact['lat'], artifact['lon'],
                kept_artifact['lat'], kept_artifact['lon']
            )
            if distance < proximity_threshold_m:
                too_close_to_kept = True
                break
        
        if not too_close_to_kept:
            final_kept_artifacts.append(artifact)
    
    # Combine final intersections
    candidate_nodes = [node['node_id'] for node in real_intersections] + \
                     [node['node_id'] for node in final_kept_artifacts]
    
    print(f"Total intersection candidates: {len(candidate_nodes)}")
    
    # Apply distance filtering (same as RouteOptimizer)
    # Stage 1: Straight-line filtering
    max_straight_line_km = (target_distance_km / 2.0) + 1.5
    start_data = graph.nodes[start_node]
    start_lat, start_lon = start_data['y'], start_data['x']
    max_radius_m = max_straight_line_km * 1000
    
    straight_line_filtered = []
    for node in candidate_nodes:
        node_data = graph.nodes[node]
        node_lat, node_lon = node_data['y'], node_data['x']
        distance = haversine_distance(start_lat, start_lon, node_lat, node_lon)
        
        if distance <= max_radius_m:
            straight_line_filtered.append(node)
    
    print(f"After straight-line filtering ({max_straight_line_km:.1f}km): {len(straight_line_filtered)}")
    
    # Stage 2: Road distance filtering  
    import networkx as nx
    max_road_distance_km = (target_distance_km / 2.0) + 1.0
    max_distance_m = max_road_distance_km * 1000
    
    road_filtered = []
    for node in straight_line_filtered:
        if node == start_node:
            road_filtered.append(node)
            continue
        
        try:
            path_length = nx.shortest_path_length(
                graph, start_node, node, weight='length'
            )
            
            if path_length <= max_distance_m:
                road_filtered.append(node)
                
        except nx.NetworkXNoPath:
            pass
        except Exception:
            pass
    
    if start_node not in road_filtered:
        road_filtered.append(start_node)
    
    print(f"After road distance filtering ({max_road_distance_km:.1f}km): {len(road_filtered)}")
    
    return {
        'all_candidates': candidate_nodes,
        'straight_line_filtered': straight_line_filtered,
        'final_filtered': road_filtered,
        'filtering_stats': {
            'max_straight_line_km': max_straight_line_km,
            'max_road_distance_km': max_road_distance_km,
            'total_candidates': len(candidate_nodes),
            'after_straight_line': len(straight_line_filtered),
            'final_count': len(road_filtered)
        }
    }

def create_visualization(graph, start_node, target_distance_km, filtered_data):
    """Create visualization with OpenStreetMap background"""
    
    # Prepare node data for visualization
    all_nodes = []
    final_nodes = []
    start_node_data = None
    
    for node_id, node_data in graph.nodes(data=True):
        point_data = {
            'node_id': node_id,
            'lat': node_data['y'],
            'lon': node_data['x'],
            'geometry': Point(node_data['x'], node_data['y'])
        }
        
        if node_id == start_node:
            start_node_data = point_data
        elif node_id in filtered_data['final_filtered']:
            final_nodes.append(point_data)
        elif node_id in filtered_data['all_candidates']:
            all_nodes.append(point_data)
    
    # Create GeoDataFrames
    if all_nodes:
        all_candidates_gdf = gpd.GeoDataFrame(all_nodes, crs='EPSG:4326')
    else:
        all_candidates_gdf = None
        
    if final_nodes:
        final_filtered_gdf = gpd.GeoDataFrame(final_nodes, crs='EPSG:4326')
    else:
        final_filtered_gdf = None
        
    if start_node_data:
        start_gdf = gpd.GeoDataFrame([start_node_data], crs='EPSG:4326')
    else:
        start_gdf = None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Set bounds around the area of interest
    start_lat = start_node_data['lat']
    start_lon = start_node_data['lon']
    
    # Calculate bounds (roughly 2km radius)
    lat_offset = 0.018  # ~2km at this latitude
    lon_offset = 0.025  # ~2km at this latitude
    
    ax.set_xlim(start_lon - lon_offset, start_lon + lon_offset)
    ax.set_ylim(start_lat - lat_offset, start_lat + lat_offset)
    
    # Convert to Web Mercator for contextily
    if all_candidates_gdf is not None and len(all_candidates_gdf) > 0:
        all_candidates_mercator = all_candidates_gdf.to_crs('EPSG:3857')
        all_candidates_mercator.plot(ax=ax, color='lightblue', markersize=8, alpha=0.6, 
                                   label=f'All Candidates ({len(all_candidates_gdf)})', zorder=2)
    
    if final_filtered_gdf is not None and len(final_filtered_gdf) > 0:
        final_filtered_mercator = final_filtered_gdf.to_crs('EPSG:3857')
        final_filtered_mercator.plot(ax=ax, color='red', markersize=12, alpha=0.8,
                                   label=f'Filtered Nodes ({len(final_filtered_gdf)})', zorder=3)
    
    if start_gdf is not None:
        start_mercator = start_gdf.to_crs('EPSG:3857')
        start_mercator.plot(ax=ax, color='green', markersize=100, marker='*', 
                          label=f'Start Node ({start_node})', zorder=4)
    
    # Add OpenStreetMap background
    try:
        # Convert axis limits to Web Mercator
        ax_mercator = plt.gca()
        ax_mercator.set_aspect('equal')
        
        # Get bounds in Web Mercator
        minx, maxx = ax.get_xlim()
        miny, maxy = ax.get_ylim()
        
        # Transform bounds to Web Mercator for contextily
        from pyproj import Transformer
        transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
        minx_merc, miny_merc = transformer.transform(start_lon - lon_offset, start_lat - lat_offset)
        maxx_merc, maxy_merc = transformer.transform(start_lon + lon_offset, start_lat + lat_offset)
        
        ax.set_xlim(minx_merc, maxx_merc)
        ax.set_ylim(miny_merc, maxy_merc)
        
        # Add basemap
        ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.7)
        
    except Exception as e:
        print(f"Warning: Could not add OpenStreetMap background: {e}")
        # Continue without background map
    
    # Formatting
    ax.set_title(f'Filtered Candidate Nodes for {target_distance_km}km Route\n'
               f'Start Node: {start_node} | Filtering: {filtered_data["filtering_stats"]["total_candidates"]} → '
               f'{filtered_data["filtering_stats"]["after_straight_line"]} → {filtered_data["filtering_stats"]["final_count"]} nodes',
               fontsize=14, fontweight='bold')
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add filtering info as text
    info_text = f"""Filtering Details:
• Target Distance: {target_distance_km:.1f}km
• Max Straight-line: {filtered_data['filtering_stats']['max_straight_line_km']:.1f}km  
• Max Road Distance: {filtered_data['filtering_stats']['max_road_distance_km']:.1f}km
• Total Intersections: {filtered_data['filtering_stats']['total_candidates']}
• After Straight-line: {filtered_data['filtering_stats']['after_straight_line']} 
• Final Candidates: {filtered_data['filtering_stats']['final_count']}"""
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Remove axis labels for cleaner look
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    plt.tight_layout()
    
    # Save the figure
    output_file = f'filtered_nodes_{target_distance_km}km.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    
    # Show the plot
    plt.show()

def main():
    """Main function to create the visualization"""
    
    # Parameters
    target_distance_km = 2.0
    start_node = 1529188403  # Default start node from CLAUDE.md
    
    print(f"Creating filtered nodes visualization for {target_distance_km}km target distance...")
    print(f"Start node: {start_node}")
    
    # Load network
    print("Loading network...")
    network_manager = NetworkManager()
    graph = network_manager.load_network()
    
    if not graph:
        print("❌ Failed to load network")
        return
    
    print(f"✅ Network loaded: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    # Get filtered nodes
    print(f"\nApplying candidate node filtering for {target_distance_km}km route...")
    filtered_data = get_filtered_nodes_for_distance(graph, start_node, target_distance_km)
    
    # Create visualization
    print(f"\nCreating visualization...")
    create_visualization(graph, start_node, target_distance_km, filtered_data)
    
    print("✅ Visualization complete!")

if __name__ == '__main__':
    main()