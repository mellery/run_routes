#!/usr/bin/env python3
"""
Generate Cached Graph with Elevation Data
Pre-processes street network with elevation data and saves to cache file
Enhanced with OSMnx + 3DEP integration for improved accuracy and performance
"""

import os
import pickle
import time
import argparse
import osmnx as ox
from utils.graph_utils import add_elevation_to_edges, add_running_weights
from utils.cache import get_cache_filename
from osmnx_config import configure_osmnx, get_config
from utils.elevation import get_elevation_service

def generate_cached_graph(center_point, radius_m, network_type='all', cache_file=None):
    """
    Generate and cache a graph with elevation data.

    Uses unified elevation service with automatic 3DEP 1m → SRTM 90m fallback.

    Args:
        center_point: (lat, lon) tuple for network center
        radius_m: Network radius in meters
        network_type: OSMnx network type ('all', 'drive', 'walk', 'bike', 'running')
        cache_file: Optional custom cache filename

    Returns:
        Processed graph with elevation data
    """
    
    if cache_file is None:
        cache_file = get_cache_filename(center_point, radius_m, network_type)
    
    print(f"🌐 Generating cached graph for Christiansburg, VA...")
    print(f"   Parameters: {radius_m}m radius, {network_type} network")
    print(f"   Cache file: {cache_file}")
    
    start_time = time.time()
    
    try:
        # Configure OSMnx for optimal performance
        configure_osmnx(timeout=300, memory_gb=2)
        config = get_config()
        
        # Step 1: Download street network
        print("\n1️⃣ Downloading street network from OpenStreetMap...")
        step_start = time.time()
        
        # Get network configuration
        network_config = config.get_network_config(network_type)
        
        graph = ox.graph_from_point(
            center_point, 
            dist=radius_m,
            **network_config
        )
        
        # Optimize graph with proper projected consolidation
        graph = config.optimize_graph(graph, simplify=True, consolidate_intersections=True, preserve_nodes=False, tolerance=10.0)
        
        step_time = time.time() - step_start
        print(f"   ✅ Downloaded {len(graph.nodes)} nodes, {len(graph.edges)} edges ({step_time:.1f}s)")
        
        # Step 2: Add elevation data to nodes
        print("\n2️⃣ Adding elevation data to graph nodes...")
        step_start = time.time()

        # Use new unified elevation service (3DEP 1m with SRTM 90m fallback)
        print("   Using unified elevation service (3DEP 1m → SRTM 90m fallback)...")
        elevation_service = get_elevation_service()

        if not elevation_service.is_available():
            print("   ⚠️ Warning: No elevation sources available")
        else:
            elevation_service.add_elevation_to_graph(graph)

        step_time = time.time() - step_start
        print(f"   ✅ Added elevation data to all nodes ({step_time:.1f}s)")

        # Step 3: Add edge grades
        print("\n3️⃣ Calculating elevation gain/loss for edges...")
        step_start = time.time()
        graph = add_elevation_to_edges(graph)
        step_time = time.time() - step_start
        print(f"   ✅ Added elevation data to edges ({step_time:.1f}s)")
        
        # Step 4: Add running-specific weights
        print("\n4️⃣ Adding running-specific weights...")
        step_start = time.time()
        
        graph = add_running_weights(graph)
        
        step_time = time.time() - step_start
        print(f"   ✅ Added running weights to all edges ({step_time:.1f}s)")
        
        # Step 5: Save to cache
        print(f"\n5️⃣ Saving processed graph to cache...")
        step_start = time.time()
        
        # Analyze elevation data quality
        elevations = [data.get('elevation', 0) for _, data in graph.nodes(data=True)]
        has_elevation = any('elevation' in data for _, data in graph.nodes(data=True))
        elevation_range = (min(elevations), max(elevations)) if elevations else (0, 0)
        non_zero_elevations = sum(1 for e in elevations if e != 0)
        
        # Create cache metadata
        cache_metadata = {
            'center_point': center_point,
            'radius_m': radius_m,
            'network_type': network_type,
            'generated_at': time.time(),
            'nodes_count': len(graph.nodes),
            'edges_count': len(graph.edges),
            'elevation_source': 'unified_service',  # Uses utils.elevation (3DEP + SRTM)
            'elevation_data_quality': {
                'has_elevation': has_elevation,
                'elevation_range': elevation_range,
                'nodes_with_elevation': non_zero_elevations,
                'elevation_coverage_percent': (non_zero_elevations / len(graph.nodes)) * 100 if graph.nodes else 0
            }
        }
        
        # Save graph and metadata
        cache_data = {
            'graph': graph,
            'metadata': cache_metadata
        }
        
        # Ensure cache directory exists
        os.makedirs('cache', exist_ok=True)
        
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
        
        # Show elevation data quality if available
        if 'elevation_data_quality' in metadata:
            quality = metadata['elevation_data_quality']
            if quality['has_elevation']:
                coverage = quality['elevation_coverage_percent']
                min_elev, max_elev = quality['elevation_range']
                print(f"   Elevation: {coverage:.1f}% coverage, {min_elev:.0f}m - {max_elev:.0f}m range")
                if metadata.get('enhanced_elevation', False):
                    print(f"   Enhanced elevation: 3DEP 1m + SRTM 90m hybrid")
                else:
                    print(f"   Standard elevation: SRTM 90m resolution")
            else:
                print(f"   ⚠️ No elevation data in cache")
        
        return graph
        
    except Exception as e:
        print(f"❌ Failed to load cached graph: {e}")
        return None

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
        choices=['all', 'drive', 'walk', 'bike', 'running'],
        default='all',
        help='Network type (default: all, "running" uses custom filter)'
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
    
    parser.add_argument(
        '--enhanced-elevation', '-e',
        action='store_true',
        default=True,
        help='Use enhanced 3DEP elevation data when available (legacy method)'
    )
    
    parser.add_argument(
        '--no-enhanced-elevation',
        action='store_true',
        help='Disable enhanced elevation, use only SRTM data'
    )
    
    parser.add_argument(
        '--osmnx-elevation',
        action='store_true',
        default=True,
        help='Use OSMnx + 3DEP integration for best accuracy (default: True)'
    )
    
    parser.add_argument(
        '--no-osmnx-elevation',
        action='store_true',
        help='Disable OSMnx elevation integration, use legacy methods'
    )
    
    args = parser.parse_args()
    
    # Determine elevation modes
    use_enhanced_elevation = args.enhanced_elevation and not args.no_enhanced_elevation
    use_osmnx_elevation = args.osmnx_elevation and not args.no_osmnx_elevation
    
    # Print configuration
    print("🔧 Configuration:")
    print(f"   Network type: {args.network_type}")
    print(f"   Radius: {args.radius}m")
    print(f"   OSMnx elevation: {use_osmnx_elevation}")
    print(f"   Enhanced elevation (legacy): {use_enhanced_elevation}")
    
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
        generate_cached_graph(
            center_point, 
            args.radius, 
            args.network_type, 
            cache_file, 
            use_enhanced_elevation,
            use_osmnx_elevation
        )
        print(f"\n🎯 Cache generation complete: {cache_file}")
        
    except Exception as e:
        print(f"\n❌ Cache generation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())