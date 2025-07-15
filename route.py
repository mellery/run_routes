import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import numpy as np
import rasterio
import os
from math import radians, cos, sin, asin, sqrt

# Function to download street network data
def get_street_data(place_name):
    return ox.graph_from_place(place_name, network_type='all')


# Function to calculate haversine distance between two points
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on Earth in meters"""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in meters
    r = 6371000
    return c * r

# Function to extract elevation from SRTM data for given coordinates
def get_elevation_from_raster(raster_path, lat, lon):
    """Extract elevation value from SRTM raster for given lat/lon coordinates"""
    try:
        with rasterio.open(raster_path) as src:
            # Check if coordinates are within raster bounds
            bounds = src.bounds
            if not (bounds.left <= lon <= bounds.right and bounds.bottom <= lat <= bounds.top):
                return None
                
            # Get the pixel coordinates for the lat/lon
            row, col = src.index(lon, lat)
            
            # Check if indices are within array bounds
            height, width = src.shape
            if not (0 <= row < height and 0 <= col < width):
                return None
                
            # Read the elevation value
            elevation = src.read(1)[row, col]
            # Handle no-data values
            if elevation == src.nodata:
                return None
            return float(elevation)
    except Exception as e:
        # Silently handle errors for coordinates outside the raster
        return None

# Function to add elevation data to all nodes in the graph
def add_elevation_to_graph(graph, raster_path):
    """Add elevation attribute to all nodes in the graph"""
    # Check if elevation data already exists
    if has_elevation_data(graph):
        print("✅ Elevation data already exists in graph, skipping elevation loading")
        return graph
    
    print("Adding elevation data to graph nodes...")
    nodes_with_elevation = 0
    
    for node_id, node_data in graph.nodes(data=True):
        lat = node_data['y']
        lon = node_data['x']
        elevation = get_elevation_from_raster(raster_path, lat, lon)
        
        if elevation is not None:
            graph.nodes[node_id]['elevation'] = elevation
            nodes_with_elevation += 1
        else:
            # Use a default elevation if SRTM data is not available
            graph.nodes[node_id]['elevation'] = 0.0
    
    print(f"Added elevation data to {nodes_with_elevation}/{len(graph.nodes)} nodes")
    return graph

def _process_nodes_batch(graph, elevation_source, fallback_raster):
    """Process nodes in batches for better performance"""
    print("   Using batch processing for elevation data...")
    
    total_nodes = len(graph.nodes)
    batch_size = min(1000, total_nodes)  # Process in batches of 1000
    nodes_with_3dep = 0
    nodes_with_srtm = 0
    nodes_with_fallback = 0
    
    node_list = list(graph.nodes(data=True))
    
    # Preload tiles for the graph area if supported
    if hasattr(elevation_source, 'preload_tiles_for_area'):
        try:
            # Calculate graph bounds
            lats = [data['y'] for _, data in node_list]
            lons = [data['x'] for _, data in node_list]
            center_lat = (min(lats) + max(lats)) / 2
            center_lon = (min(lons) + max(lons)) / 2
            
            # Estimate radius from bounds
            lat_range = max(lats) - min(lats)
            lon_range = max(lons) - min(lons)
            radius_km = max(lat_range, lon_range) * 111.0 / 2  # Convert degrees to km
            
            print(f"   Preloading elevation tiles for {radius_km:.1f}km radius...")
            elevation_source.preload_tiles_for_area(center_lat, center_lon, radius_km)
        except Exception as e:
            print(f"   ⚠️ Tile preloading failed: {e}")
    
    for batch_start in range(0, total_nodes, batch_size):
        batch_end = min(batch_start + batch_size, total_nodes)
        batch_nodes = node_list[batch_start:batch_end]
        
        # Extract coordinates for batch
        coordinates = [(node_data['y'], node_data['x']) for node_id, node_data in batch_nodes]
        
        # Batch query elevation source
        try:
            elevations = elevation_source.get_elevation_profile(coordinates)
        except Exception as e:
            print(f"   ⚠️ Batch elevation query failed: {e}")
            elevations = [None] * len(coordinates)
        
        # Process batch results
        for i, (node_id, node_data) in enumerate(batch_nodes):
            elevation = elevations[i] if i < len(elevations) and elevations[i] is not None else None
            
            if elevation is not None and elevation != 0.0:
                nodes_with_3dep += 1
            else:
                # Fallback to SRTM if needed
                if fallback_raster and os.path.exists(fallback_raster):
                    elevation = get_elevation_from_raster(fallback_raster, node_data['y'], node_data['x'])
                    if elevation is not None:
                        nodes_with_srtm += 1
                    else:
                        elevation = 0.0
                        nodes_with_fallback += 1
                else:
                    elevation = 0.0
                    nodes_with_fallback += 1
            
            graph.nodes[node_id]['elevation'] = elevation
        
        # Progress update
        percent = int((batch_end / total_nodes) * 100)
        if percent % 10 == 0 or batch_end == total_nodes:
            print(f"   Progress: {percent}% ({batch_end:,}/{total_nodes:,} nodes)")
    
    return nodes_with_3dep, nodes_with_srtm, nodes_with_fallback

def _process_nodes_individual(graph, elevation_source, fallback_raster):
    """Process nodes individually (fallback method)"""
    print("   Using individual processing for elevation data...")
    
    total_nodes = len(graph.nodes)
    nodes_processed = 0
    last_percent = -1
    nodes_with_3dep = 0
    nodes_with_srtm = 0
    nodes_with_fallback = 0
    
    for node_id, node_data in graph.nodes(data=True):
        lat = node_data['y']
        lon = node_data['x']
        elevation = None
        
        # Try 3DEP/enhanced source first
        if elevation_source:
            try:
                elevation = elevation_source.get_elevation(lat, lon)
                if elevation is not None:
                    nodes_with_3dep += 1
            except Exception:
                elevation = None
        
        # Fallback to SRTM raster file if 3DEP failed
        if elevation is None and fallback_raster and os.path.exists(fallback_raster):
            elevation = get_elevation_from_raster(fallback_raster, lat, lon)
            if elevation is not None:
                nodes_with_srtm += 1
        
        # Final fallback to default elevation
        if elevation is None:
            elevation = 0.0
            nodes_with_fallback += 1
        
        graph.nodes[node_id]['elevation'] = elevation
        
        # Update progress indicator
        nodes_processed += 1
        percent = int((nodes_processed / total_nodes) * 100)
        if percent != last_percent and percent % 10 == 0:  # Show every 10%
            print(f"   Progress: {percent}% ({nodes_processed:,}/{total_nodes:,} nodes)")
            last_percent = percent
    
    return nodes_with_3dep, nodes_with_srtm, nodes_with_fallback

def has_elevation_data(graph):
    """Check if graph already has elevation data
    
    Args:
        graph: NetworkX graph
        
    Returns:
        True if graph has elevation data for nodes
    """
    if not graph or len(graph.nodes) == 0:
        return False
    
    # Check first few nodes for elevation data
    sample_size = min(10, len(graph.nodes))
    sample_nodes = list(graph.nodes(data=True))[:sample_size]
    
    for node_id, data in sample_nodes:
        if 'elevation' not in data:
            return False
    
    return True

def add_enhanced_elevation_to_graph(graph, use_3dep=True, fallback_raster='elevation_data/srtm_90m/srtm_20_05.tif'):
    """Add high-resolution elevation data to graph nodes using 3DEP when available
    
    Args:
        graph: NetworkX graph
        use_3dep: Whether to try using 3DEP data first
        fallback_raster: SRTM raster file to use as fallback
        
    Returns:
        Graph with enhanced elevation data
    """
    # Check if elevation data already exists
    if has_elevation_data(graph):
        print("✅ Elevation data already exists in graph, skipping elevation loading")
        return graph
    
    print("Adding enhanced elevation data to graph nodes...")
    
    elevation_manager = None
    elevation_source = None
    nodes_with_3dep = 0
    nodes_with_srtm = 0
    nodes_with_fallback = 0
    
    # Try to initialize 3DEP elevation source
    if use_3dep:
        try:
            from elevation_data_sources import get_elevation_manager
            elevation_manager = get_elevation_manager()
            if elevation_manager:
                available_sources = elevation_manager.get_available_sources()
                if '3dep_local' in available_sources or '3dep' in available_sources:
                    elevation_source = elevation_manager.get_elevation_source()
                    print(f"   Using 3DEP 1m elevation data (primary)")
                elif 'srtm' in available_sources:
                    elevation_source = elevation_manager.get_elevation_source()
                    print(f"   Using SRTM elevation data from manager")
        except Exception as e:
            print(f"   ⚠️ 3DEP initialization failed: {e}")
            elevation_manager = None
            elevation_source = None
    
    # Process all nodes with batch optimization and progress indicator
    total_nodes = len(graph.nodes)
    
    # Try batch processing if elevation source supports it
    try:
        if elevation_source and hasattr(elevation_source, 'get_elevation_profile'):
            nodes_with_3dep, nodes_with_srtm, nodes_with_fallback = _process_nodes_batch(
                graph, elevation_source, fallback_raster
            )
        else:
            # Fallback to individual processing
            nodes_with_3dep, nodes_with_srtm, nodes_with_fallback = _process_nodes_individual(
                graph, elevation_source, fallback_raster
            )
    except Exception as e:
        print(f"   ⚠️ Batch processing failed, using individual processing: {e}")
        nodes_with_3dep, nodes_with_srtm, nodes_with_fallback = _process_nodes_individual(
            graph, elevation_source, fallback_raster
        )
    
    # Show final progress
    print(f"   Progress: 100% ({total_nodes:,}/{total_nodes:,} nodes) - Complete!")
    
    # Clean up elevation manager
    if elevation_manager:
        try:
            elevation_manager.close_all()
        except Exception:
            pass
    
    total_nodes = len(graph.nodes)
    print(f"   ✅ Enhanced elevation added to {total_nodes} nodes:")
    if nodes_with_3dep > 0:
        print(f"      • 3DEP 1m data: {nodes_with_3dep} nodes ({nodes_with_3dep/total_nodes*100:.1f}%)")
    if nodes_with_srtm > 0:
        print(f"      • SRTM 90m data: {nodes_with_srtm} nodes ({nodes_with_srtm/total_nodes*100:.1f}%)")
    if nodes_with_fallback > 0:
        print(f"      • Default elevation: {nodes_with_fallback} nodes ({nodes_with_fallback/total_nodes*100:.1f}%)")
    
    return graph

# Function to calculate elevation gain/loss and add to edges
def add_elevation_to_edges(graph):
    """Calculate elevation gain/loss for each edge and add as edge attributes"""
    print("Calculating elevation gain/loss for edges...")
    
    for u, v, key, edge_data in graph.edges(keys=True, data=True):
        elevation_u = graph.nodes[u].get('elevation', 0.0)
        elevation_v = graph.nodes[v].get('elevation', 0.0)
        
        # Calculate elevation change (positive = uphill from u to v)
        elevation_gain = elevation_v - elevation_u
        elevation_loss = max(0, elevation_u - elevation_v)
        elevation_climb = max(0, elevation_v - elevation_u)
        
        # Add elevation attributes to edge
        graph.edges[u, v, key]['elevation_gain'] = elevation_gain
        graph.edges[u, v, key]['elevation_loss'] = elevation_loss
        graph.edges[u, v, key]['elevation_climb'] = elevation_climb
        length = edge_data.get('length', 1.0)
        if length > 0:
            graph.edges[u, v, key]['grade'] = abs(elevation_gain) / length * 100
        else:
            graph.edges[u, v, key]['grade'] = 0.0
        
        # Calculate haversine distance if not already present
        if 'haversine_distance' not in edge_data:
            lat1, lon1 = graph.nodes[u]['y'], graph.nodes[u]['x']
            lat2, lon2 = graph.nodes[v]['y'], graph.nodes[v]['x']
            haversine_dist = haversine_distance(lat1, lon1, lat2, lon2)
            graph.edges[u, v, key]['haversine_distance'] = haversine_dist
    
    print("Added elevation data to all edges")
    return graph

# Function to add running-specific weights to edges
def add_running_weights(graph, elevation_weight=0.1, grade_penalty=2.0):
    """Add running-specific weights to edges based on distance and elevation"""
    print("Adding running-specific weights to edges...")
    
    for u, v, key, edge_data in graph.edges(keys=True, data=True):
        distance = edge_data.get('length', 0)
        elevation_gain = edge_data.get('elevation_gain', 0)
        grade = edge_data.get('grade', 0)
        
        # Base weight is distance
        running_weight = distance
        
        # Add elevation penalty/bonus
        if elevation_gain > 0:  # Uphill
            running_weight += elevation_gain * elevation_weight
        
        # Add grade penalty for steep sections
        if grade > 5.0:  # Steeper than 5% grade
            running_weight += distance * (grade / 100) * grade_penalty
        
        graph.edges[u, v, key]['running_weight'] = running_weight
    
    print("Added running weights to all edges")
    return graph

# Function to get nodes within a certain distance of a starting point
def get_nodes_within_distance(graph, start_node, max_distance_km):
    """Get intersection nodes within max_distance_km of the start_node"""
    if start_node not in graph.nodes:
        print(f"Start node {start_node} not found in graph")
        return []
    
    start_lat = graph.nodes[start_node]['y']
    start_lon = graph.nodes[start_node]['x']
    
    nearby_nodes = []
    max_distance_m = max_distance_km * 1000  # Convert to meters
    
    for node_id, node_data in graph.nodes(data=True):
        if node_id == start_node:
            nearby_nodes.append(node_id)
            continue
        
        # Only include intersection nodes (degree > 2) or important endpoints (degree == 1)
        # Skip geometry nodes (degree == 2) which are just points along road segments
        node_degree = graph.degree(node_id)
        if node_degree == 2:
            continue
            
        distance = haversine_distance(start_lat, start_lon, node_data['y'], node_data['x'])
        if distance <= max_distance_m:
            nearby_nodes.append(node_id)
    
    print(f"Found {len(nearby_nodes)} intersection nodes within {max_distance_km}km of start node")
    return nearby_nodes

# Function to create a subgraph with nodes within a certain distance
def create_distance_constrained_subgraph(graph, start_node, max_distance_km):
    """Create a subgraph containing only nodes within max_distance_km of start_node"""
    nearby_nodes = get_nodes_within_distance(graph, start_node, max_distance_km)
    subgraph = graph.subgraph(nearby_nodes).copy()
    print(f"Created subgraph with {len(subgraph.nodes)} nodes and {len(subgraph.edges)} edges")
    return subgraph

def main():
    """Main execution function"""
    print("Loading street network with elevation data...")
    
    try:
        # Use cached graph loader for faster execution
        from graph_cache import load_or_generate_graph
        
        center_point = (37.1299, -80.4094)  # Christiansburg, VA
        street_data = load_or_generate_graph(
            center_point=center_point,
            radius_m=1200,
            network_type='all'
        )
        
        if not street_data:
            print("❌ Failed to load street network")
            return
            
        print(f"✅ Loaded street network: {len(street_data.nodes)} nodes, {len(street_data.edges)} edges")
        
    except Exception as e:
        print(f"❌ Error loading cached graph, falling back to direct method: {e}")
        
        # Fallback to original method
        place = 'Christiansburg, Virginia, USA'
        srtm_file = 'elevation_data/srtm_90m/srtm_20_05.tif'

        # Get street data
        print("Downloading street data...")
        street_data = get_street_data(place)
        print(f"Street data downloaded! Number of nodes: {len(street_data.nodes)}")

        # Add elevation data to the graph
        street_data = add_elevation_to_graph(street_data, srtm_file)

        # Add elevation data to edges
        street_data = add_elevation_to_edges(street_data)

        # Add running-specific weights
        street_data = add_running_weights(street_data)

    # Analyze specific node
    specific_node_id = 216507089  # Replace with the actual node ID you want to check
    if specific_node_id in street_data.nodes:
        data = street_data.nodes[specific_node_id]
        print(f"Node {specific_node_id}: {data}")
        print(f"Node degree: {street_data.degree(specific_node_id)}")
    else:
        print(f"Node {specific_node_id} not found in the graph.")

    # Test distance-constrained subgraph
    print("\n=== Testing Distance-Constrained Subgraph ===")
    test_subgraph = create_distance_constrained_subgraph(street_data, specific_node_id, 2.0)  # 2km radius

    # Show some elevation statistics
    elevations = [data.get('elevation', 0) for node, data in street_data.nodes(data=True)]
    print(f"\nElevation Statistics:")
    print(f"Min elevation: {min(elevations):.1f}m")
    print(f"Max elevation: {max(elevations):.1f}m")
    print(f"Average elevation: {np.mean(elevations):.1f}m")

    # Show some edge statistics
    edge_gains = [data.get('elevation_gain', 0) for u, v, data in street_data.edges(data=True)]
    edge_grades = [data.get('grade', 0) for u, v, data in street_data.edges(data=True)]
    print(f"\nEdge Statistics:")
    print(f"Max elevation gain: {max(edge_gains):.1f}m")
    print(f"Max elevation loss: {min(edge_gains):.1f}m")
    print(f"Max grade: {max(edge_grades):.1f}%")

    # Plot the street data with elevation-colored nodes
    fig, ax = ox.plot_graph(street_data, node_size=8, node_color=[data.get('elevation', 0) for node, data in street_data.nodes(data=True)], 
                           edge_color='gray', edge_linewidth=0.5, show=False, close=False)
    plt.colorbar(ax.collections[0], ax=ax, label='Elevation (m)')
    plt.title('Street Network with Elevation Data')
    
    # Only show plot if not running in test environment
    if not os.environ.get('PYTEST_CURRENT_TEST') and not any('unittest' in arg for arg in sys.argv):
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    main()