import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import numpy as np
import rasterio
from math import radians, cos, sin, asin, sqrt

# Function to download street network data
def get_street_data(place_name):
    return ox.graph_from_place(place_name, network_type='all')

# Function to print node information
def print_node_info(graph):
    for node, data in graph.nodes(data=True):
        print(f"Node {node}: {data}")

# Function to print information for a specific node ID
def print_specific_node_info(graph, node_id):
    if node_id in graph.nodes:
        data = graph.nodes[node_id]
        print(f"Node {node_id}: {data}")
        print(graph.degree(node_id))
    else:
        print(f"Node {node_id} not found in the graph.")

# Function to print the length of each edge for a specific node ID
def print_edge_lengths(graph, node_id):
    if node_id in graph.nodes:
        for neighbor in graph.neighbors(node_id):
            edge_data = graph.get_edge_data(node_id, neighbor)
            for key, data in edge_data.items():
                print(f"Edge from {node_id} to {neighbor} has length {data['length']} meters.")
    else:
        print(f"Node {node_id} not found in the graph.")

# Function to remove nodes with only one connected street
def remove_single_street_nodes(graph):
    nodes_to_remove = [node for node, data in graph.nodes(data=True) if data.get('street_count', 0) <= 1]
    graph.remove_nodes_from(nodes_to_remove)
    return graph

# Function to remove nodes with only one neighbor
def remove_single_neighbor_nodes(graph):
    nodes_to_remove = [node for node, degree in dict(graph.degree()).items() if degree == 1]
    graph.remove_nodes_from(nodes_to_remove)
    return graph

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
        srtm_file = 'srtm_20_05.tif'

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

    # Print information for a specific node ID
    specific_node_id = 216507089  # Replace with the actual node ID you want to check
    print_specific_node_info(street_data, specific_node_id)

    # Print the length of each edge for a specific node ID
    print_edge_lengths(street_data, specific_node_id)

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
    plt.show()

if __name__ == "__main__":
    main()