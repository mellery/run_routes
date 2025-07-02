#!/usr/bin/env python3
"""
Generate Cached Graph with Elevation Data
Pre-processes street network with elevation data and saves to cache file
"""

import os
import pickle
import time
import argparse
import osmnx as ox
from route import add_elevation_to_graph, add_elevation_to_edges, add_running_weights

def generate_cached_graph(center_point, radius_m, network_type='all', cache_file=None):
    """
    Generate and cache a graph with elevation data
    
    Args:
        center_point: (lat, lon) tuple for network center
        radius_m: Network radius in meters
        network_type: OSMnx network type ('all', 'drive', 'walk', 'bike')
        cache_file: Optional custom cache filename
    
    Returns:
        Processed graph with elevation data
    """
    
    if cache_file is None:
        cache_file = f"cached_graph_{radius_m}m_{network_type}.pkl"
    
    print(f"🌐 Generating cached graph for Christiansburg, VA...")
    print(f"   Parameters: {radius_m}m radius, {network_type} network")
    print(f"   Cache file: {cache_file}")
    
    start_time = time.time()
    
    try:
        # Step 1: Download street network
        print("\n1️⃣ Downloading street network from OpenStreetMap...")
        step_start = time.time()
        
        graph = ox.graph_from_point(
            center_point, 
            dist=radius_m, 
            network_type=network_type
        )
        
        step_time = time.time() - step_start
        print(f"   ✅ Downloaded {len(graph.nodes)} nodes, {len(graph.edges)} edges ({step_time:.1f}s)")
        
        # Step 2: Add elevation data to nodes
        print("\n2️⃣ Adding elevation data to graph nodes...")
        step_start = time.time()
        
        graph = add_elevation_to_graph(graph, 'srtm_20_05.tif')
        
        step_time = time.time() - step_start
        print(f"   ✅ Added elevation data to all nodes ({step_time:.1f}s)")
        
        # Step 3: Add elevation data to edges
        print("\n3️⃣ Calculating elevation gain/loss for edges...")
        step_start = time.time()
        
        graph = add_elevation_to_edges(graph)
        
        step_time = time.time() - step_start
        print(f"   ✅ Added elevation data to all edges ({step_time:.1f}s)")
        
        # Step 4: Add running-specific weights
        print("\n4️⃣ Adding running-specific weights...")
        step_start = time.time()
        
        graph = add_running_weights(graph)
        
        step_time = time.time() - step_start
        print(f"   ✅ Added running weights to all edges ({step_time:.1f}s)")
        
        # Step 5: Save to cache
        print(f"\n5️⃣ Saving processed graph to cache...")
        step_start = time.time()
        
        # Create cache metadata
        cache_metadata = {
            'center_point': center_point,
            'radius_m': radius_m,
            'network_type': network_type,
            'generated_at': time.time(),
            'nodes_count': len(graph.nodes),
            'edges_count': len(graph.edges),
            'elevation_file': 'srtm_20_05.tif'
        }
        
        # Save graph and metadata
        cache_data = {
            'graph': graph,
            'metadata': cache_metadata
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        step_time = time.time() - step_start
        file_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
        print(f"   ✅ Saved to {cache_file} ({file_size_mb:.1f}MB, {step_time:.1f}s)")
        
        # Summary
        total_time = time.time() - start_time
        print(f"\n🎉 Graph generation complete!")
        print(f"   Total time: {total_time:.1f} seconds")
        print(f"   Cache file: {cache_file}")
        print(f"   Graph stats: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
        # Get elevation statistics
        elevations = [data.get('elevation', 0) for _, data in graph.nodes(data=True)]
        if elevations:
            print(f"   Elevation range: {min(elevations):.0f}m - {max(elevations):.0f}m")
        
        return graph
        
    except Exception as e:
        print(f"❌ Graph generation failed: {e}")
        # Clean up partial cache file
        if os.path.exists(cache_file):
            os.remove(cache_file)
        raise

def load_cached_graph(cache_file):
    """
    Load a cached graph with validation
    
    Args:
        cache_file: Path to cache file
        
    Returns:
        Graph if valid, None if invalid/missing
    """
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        print(f"📁 Loading cached graph from {cache_file}...")
        
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        graph = cache_data['graph']
        metadata = cache_data['metadata']
        
        # Validate cache
        if not hasattr(graph, 'nodes') or not hasattr(graph, 'edges'):
            print("❌ Invalid graph structure in cache")
            return None
        
        # Check if elevation data exists
        sample_node = next(iter(graph.nodes(data=True)))[1]
        if 'elevation' not in sample_node:
            print("❌ No elevation data in cached graph")
            return None
        
        # Check if running weights exist
        sample_edge = next(iter(graph.edges(data=True)))[2]
        if 'running_weight' not in sample_edge:
            print("❌ No running weights in cached graph")
            return None
        
        print(f"✅ Loaded cached graph successfully")
        print(f"   Generated: {time.ctime(metadata['generated_at'])}")
        print(f"   Network: {metadata['radius_m']}m {metadata['network_type']}")
        print(f"   Stats: {metadata['nodes_count']} nodes, {metadata['edges_count']} edges")
        
        return graph
        
    except Exception as e:
        print(f"❌ Failed to load cached graph: {e}")
        return None

def get_cache_filename(center_point, radius_m, network_type='all'):
    """Generate standardized cache filename"""
    lat, lon = center_point
    return f"cached_graph_{lat:.4f}_{lon:.4f}_{radius_m}m_{network_type}.pkl"

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description='Generate cached graph with elevation data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--radius', '-r',
        type=int,
        default=1200,
        help='Network radius in meters (default: 1200)'
    )
    
    parser.add_argument(
        '--network-type', '-t',
        choices=['all', 'drive', 'walk', 'bike'],
        default='all',
        help='Network type (default: all)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output cache filename (auto-generated if not specified)'
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force regeneration even if cache exists'
    )
    
    args = parser.parse_args()
    
    # Christiansburg, VA coordinates
    center_point = (37.1299, -80.4094)
    
    # Generate cache filename
    cache_file = args.output or get_cache_filename(center_point, args.radius, args.network_type)
    
    # Check if cache already exists
    if os.path.exists(cache_file) and not args.force:
        print(f"📁 Cache file {cache_file} already exists")
        print("   Use --force to regenerate")
        
        # Try to load and validate existing cache
        graph = load_cached_graph(cache_file)
        if graph:
            print("✅ Existing cache is valid")
            return
        else:
            print("❌ Existing cache is invalid, regenerating...")
    
    # Generate new cache
    try:
        generate_cached_graph(center_point, args.radius, args.network_type, cache_file)
        print(f"\n🎯 Cache generation complete: {cache_file}")
        
    except Exception as e:
        print(f"\n❌ Cache generation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())