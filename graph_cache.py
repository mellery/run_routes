#!/usr/bin/env python3
"""
Graph Cache Utilities
Shared functions for loading and managing cached graphs with elevation data
"""

import os
import subprocess
import sys
from generate_cached_graph import load_cached_graph, get_cache_filename, generate_cached_graph

def load_or_generate_graph(center_point=(37.1299, -80.4094), radius_m=1200, network_type='all', force_regenerate=False, use_enhanced_elevation=True):
    """
    Load cached graph or generate if not available
    
    Args:
        center_point: (lat, lon) tuple for network center
        radius_m: Network radius in meters
        network_type: OSMnx network type ('all', 'drive', 'walk', 'bike')
        force_regenerate: Force regeneration even if cache exists
        use_enhanced_elevation: Whether to use 3DEP elevation data when available
        
    Returns:
        Graph with elevation data and running weights
    """
    
    # Generate cache filename
    cache_file = get_cache_filename(center_point, radius_m, network_type)
    
    # Try to load existing cache first
    if not force_regenerate:
        graph = load_cached_graph(cache_file)
        if graph is not None:
            return graph
    
    # Cache doesn't exist or is invalid, generate new one
    print(f"🔄 No valid cache found, generating new graph...")
    print(f"   This may take a few minutes for the first time...")
    
    try:
        graph = generate_cached_graph(center_point, radius_m, network_type, cache_file, use_enhanced_elevation)
        return graph
        
    except Exception as e:
        print(f"❌ Failed to generate cached graph: {e}")
        
        # Fall back to calling the script directly
        print("🔄 Attempting to generate cache using external script...")
        try:
            result = subprocess.run([
                sys.executable, 'generate_cached_graph.py',
                '--radius', str(radius_m),
                '--network-type', network_type,
                '--output', cache_file
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            if result.returncode == 0:
                # Try loading the newly generated cache
                graph = load_cached_graph(cache_file)
                if graph is not None:
                    return graph
            else:
                print(f"❌ Cache generation script failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("❌ Cache generation timed out")
        except Exception as script_error:
            print(f"❌ Cache generation script error: {script_error}")
        
        # If all else fails, import and run directly (slowest option)
        print("⚠️ Falling back to direct graph generation (will be slow)...")
        try:
            import osmnx as ox
            from route import add_elevation_to_graph, add_enhanced_elevation_to_graph, add_elevation_to_edges, add_running_weights
            
            graph = ox.graph_from_point(center_point, dist=radius_m, network_type=network_type)
            if use_enhanced_elevation:
                graph = add_enhanced_elevation_to_graph(graph, use_3dep=True, fallback_raster='srtm_20_05.tif')
            else:
                graph = add_elevation_to_graph(graph, 'srtm_20_05.tif')
            graph = add_elevation_to_edges(graph)
            graph = add_running_weights(graph)
            
            print("✅ Direct graph generation successful")
            return graph
            
        except Exception as fallback_error:
            print(f"❌ Direct graph generation also failed: {fallback_error}")
            raise

def list_cached_graphs():
    """List all available cached graphs"""
    
    cache_files = [f for f in os.listdir('.') if f.startswith('cached_graph_') and f.endswith('.pkl')]
    
    if not cache_files:
        print("📁 No cached graphs found")
        return []
    
    print(f"📁 Found {len(cache_files)} cached graphs:")
    print("-" * 60)
    
    valid_caches = []
    for cache_file in sorted(cache_files):
        try:
            graph = load_cached_graph(cache_file)
            if graph is not None:
                file_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
                print(f"✅ {cache_file} ({file_size_mb:.1f}MB)")
                valid_caches.append(cache_file)
            else:
                print(f"❌ {cache_file} (invalid)")
        except:
            print(f"❌ {cache_file} (corrupted)")
    
    print("-" * 60)
    return valid_caches

def clean_cache(keep_latest=True):
    """Clean up old or invalid cache files"""
    
    cache_files = [f for f in os.listdir('.') if f.startswith('cached_graph_') and f.endswith('.pkl')]
    
    if not cache_files:
        print("📁 No cache files to clean")
        return
    
    removed_count = 0
    kept_count = 0
    
    # Sort by modification time (newest first)
    cache_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    
    for i, cache_file in enumerate(cache_files):
        try:
            # Keep the first (newest) file if keep_latest is True
            if keep_latest and i == 0:
                # Validate the newest file
                graph = load_cached_graph(cache_file)
                if graph is not None:
                    print(f"✅ Keeping latest cache: {cache_file}")
                    kept_count += 1
                    continue
                else:
                    print(f"❌ Latest cache invalid, removing: {cache_file}")
            
            # Remove older or invalid files
            os.remove(cache_file)
            print(f"🗑️ Removed: {cache_file}")
            removed_count += 1
            
        except Exception as e:
            print(f"❌ Failed to process {cache_file}: {e}")
    
    print(f"\n📊 Cache cleanup complete: {removed_count} removed, {kept_count} kept")

if __name__ == "__main__":
    # Command line interface for cache management
    import argparse
    
    parser = argparse.ArgumentParser(description='Graph cache management utilities')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List cached graphs')
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean cache files')
    clean_parser.add_argument('--all', action='store_true', help='Remove all cache files')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test cache loading')
    test_parser.add_argument('--radius', type=int, default=800, help='Test radius')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_cached_graphs()
    elif args.command == 'clean':
        clean_cache(keep_latest=not args.all)
    elif args.command == 'test':
        print(f"🧪 Testing cache loading with {args.radius}m radius...")
        graph = load_or_generate_graph(radius_m=args.radius)
        if graph:
            print(f"✅ Success: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        else:
            print("❌ Failed to load or generate graph")
    else:
        parser.print_help()