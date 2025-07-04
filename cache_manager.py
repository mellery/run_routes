#!/usr/bin/env python3
"""
Cache Management Utility
Provides tools for managing distance matrix caches
"""

import os
import argparse
from distance_cache import DistanceMatrixCache


def show_cache_info():
    """Show information about cached distance matrices"""
    print("=== DISTANCE MATRIX CACHE INFO ===")
    
    cache = DistanceMatrixCache()
    info = cache.get_cache_info()
    
    if 'error' in info:
        print(f"❌ Error accessing cache: {info['error']}")
        return
    
    print(f"Cache directory: {info['cache_dir']}")
    print(f"Total cache files: {info['total_files']}")
    print(f"Total cache size: {info['total_size_mb']:.2f} MB")
    
    if info['entries']:
        print(f"\nCached distance matrices:")
        print(f"{'Key':<16} {'Nodes':<8} {'Objective':<20} {'Age (hours)':<12} {'Size (MB)':<10}")
        print("-" * 80)
        
        for entry in sorted(info['entries'], key=lambda x: x['age_hours']):
            print(f"{entry['key']:<16} {entry['nodes']:<8} {entry['objective']:<20} "
                  f"{entry['age_hours']:<12.1f} {entry['size_mb']:<10.2f}")
    else:
        print("\nNo cached distance matrices found.")
    
    # Calculate potential time savings
    total_nodes = sum(entry['nodes'] for entry in info['entries'])
    if total_nodes > 0:
        # Estimate time savings (rough calculation)
        estimated_computation_time = sum(entry['nodes']**2 * 0.001 for entry in info['entries'])  # ~1ms per node pair
        print(f"\nEstimated computation time saved: {estimated_computation_time:.1f} seconds")


def clean_cache(max_age_hours=168):
    """Clean old cache entries"""
    print(f"=== CLEANING CACHE (older than {max_age_hours} hours) ===")
    
    cache = DistanceMatrixCache()
    cache.clean_old_cache(max_age_hours)


def clear_all_cache():
    """Clear all cache entries (with confirmation)"""
    print("=== CLEARING ALL CACHE ===")
    
    cache = DistanceMatrixCache()
    info = cache.get_cache_info()
    
    if info['total_files'] == 0:
        print("No cache files to clear.")
        return
    
    # Ask for confirmation
    response = input(f"Clear {info['total_files']} cache files ({info['total_size_mb']:.2f} MB)? [y/N]: ")
    
    if response.lower() in ['y', 'yes']:
        try:
            for entry in info['entries']:
                cache_file = os.path.join(cache.cache_dir, f"matrix_{entry['key']}.pkl")
                if os.path.exists(cache_file):
                    os.remove(cache_file)
            
            print(f"✅ Cleared {info['total_files']} cache files")
            
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")
    else:
        print("Cache clear cancelled.")


def test_cache_performance():
    """Test cache performance with sample data"""
    print("=== CACHE PERFORMANCE TEST ===")
    
    try:
        from route_services import NetworkManager, RouteOptimizer
        from tsp_solver import GeneticAlgorithmTSP
        import time
        
        print("Loading network...")
        nm = NetworkManager()
        graph = nm.load_network()
        optimizer = RouteOptimizer(graph)
        
        # Get test candidates
        all_candidates = optimizer._get_intersection_nodes()
        test_candidates = all_candidates[:100]  # Test with 100 candidates
        
        print(f"Testing cache performance with {len(test_candidates)} candidates...")
        
        # First run (no cache)
        print("\n1. First run (building cache)...")
        start_time = time.time()
        genetic_solver1 = GeneticAlgorithmTSP(graph, 1529188403, 'maximize_elevation', test_candidates)
        first_run_time = time.time() - start_time
        print(f"   First run: {first_run_time:.2f}s")
        
        # Second run (should use cache)
        print("\n2. Second run (using cache)...")
        start_time = time.time()
        genetic_solver2 = GeneticAlgorithmTSP(graph, 1529188403, 'maximize_elevation', test_candidates)
        second_run_time = time.time() - start_time
        print(f"   Second run: {second_run_time:.2f}s")
        
        # Calculate speedup
        if second_run_time > 0:
            speedup = first_run_time / second_run_time
            print(f"\n🚀 Cache speedup: {speedup:.1f}x faster")
            
            if speedup > 5:
                print("✅ Excellent cache performance!")
            elif speedup > 2:
                print("✅ Good cache performance!")
            else:
                print("⚠️ Limited cache benefit - may need debugging")
        
    except Exception as e:
        print(f"❌ Cache performance test failed: {e}")


def main():
    """Main cache management interface"""
    parser = argparse.ArgumentParser(description="Distance Matrix Cache Manager")
    parser.add_argument('command', choices=['info', 'clean', 'clear', 'test'], 
                       help='Cache management command')
    parser.add_argument('--max-age-hours', type=float, default=168,
                       help='Maximum age in hours for cache cleanup (default: 168 = 7 days)')
    
    args = parser.parse_args()
    
    if args.command == 'info':
        show_cache_info()
    elif args.command == 'clean':
        clean_cache(args.max_age_hours)
    elif args.command == 'clear':
        clear_all_cache()
    elif args.command == 'test':
        test_cache_performance()


if __name__ == "__main__":
    main()