#!/usr/bin/env python3
"""
Setup Cache Script
Pre-generates commonly used cached graphs for faster application startup
"""

import os
import sys
from generate_cached_graph import generate_cached_graph, get_cache_filename

def setup_common_caches():
    """Generate commonly used cache files"""
    
    print("🚀 Setting up graph caches for Running Route Optimizer")
    print("=" * 60)
    
    # Christiansburg, VA coordinates
    center_point = (37.1299, -80.4094)
    
    # Common configurations
    configs = [
        {'radius_m': 800, 'network_type': 'all', 'description': 'Default small area (all roads)'},
        {'radius_m': 1200, 'network_type': 'all', 'description': 'Default medium area (all roads)'},
        {'radius_m': 400, 'network_type': 'drive', 'description': 'Quick test area (drive only)'},
        {'radius_m': 800, 'network_type': 'drive', 'description': 'Medium area (drive only)'},
    ]
    
    total_configs = len(configs)
    success_count = 0
    
    for i, config in enumerate(configs, 1):
        print(f"\n📦 Cache {i}/{total_configs}: {config['description']}")
        print(f"   Parameters: {config['radius_m']}m radius, {config['network_type']} network")
        
        cache_file = get_cache_filename(center_point, config['radius_m'], config['network_type'])
        
        # Check if cache already exists
        if os.path.exists(cache_file):
            print(f"   ✅ Cache already exists: {cache_file}")
            success_count += 1
            continue
        
        try:
            # Generate cache
            generate_cached_graph(
                center_point=center_point,
                radius_m=config['radius_m'],
                network_type=config['network_type'],
                cache_file=cache_file
            )
            
            print(f"   ✅ Cache generated successfully: {cache_file}")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ Cache generation failed: {e}")
            continue
    
    print("\n" + "=" * 60)
    print(f"🎯 Cache setup complete: {success_count}/{total_configs} caches ready")
    
    if success_count == total_configs:
        print("✅ All caches generated successfully!")
        print("\n📋 Applications will now start much faster:")
        print("   • Web app: streamlit run running_route_app.py")
        print("   • CLI app: python cli_route_planner.py --interactive")
        print("   • Route demo: python route.py")
        print("   • TSP test: python quick_tsp_test.py")
        
    elif success_count > 0:
        print(f"⚠️ Partial success: {success_count} caches ready")
        print("   Applications will use cached data where available")
        
    else:
        print("❌ No caches generated successfully")
        print("   Applications will generate graphs on first run (slower)")
    
    return success_count == total_configs

def verify_setup():
    """Verify the cache setup by testing loading"""
    
    print("\n🧪 Verifying cache setup...")
    
    try:
        from graph_cache import load_or_generate_graph
        
        # Test loading a small cache
        print("   Testing 400m drive network...")
        graph = load_or_generate_graph(
            center_point=(37.1299, -80.4094),
            radius_m=400,
            network_type='drive'
        )
        
        if graph and len(graph.nodes) > 0:
            print(f"   ✅ Cache loading successful: {len(graph.nodes)} nodes")
            return True
        else:
            print("   ❌ Cache loading returned empty graph")
            return False
            
    except Exception as e:
        print(f"   ❌ Cache loading failed: {e}")
        return False

def main():
    """Main setup function"""
    
    # Check prerequisites
    if not os.path.exists('srtm_20_05.tif'):
        print("❌ Missing SRTM elevation data file: srtm_20_05.tif")
        print("   Please ensure the elevation data file is present")
        return 1
    
    try:
        import osmnx
        import rasterio
        print("✅ Required packages available")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("   Please install required dependencies")
        return 1
    
    # Generate caches
    success = setup_common_caches()
    
    # Verify setup
    if success:
        verify_success = verify_setup()
        if verify_success:
            print("\n🎉 Cache setup and verification complete!")
            return 0
        else:
            print("\n⚠️ Cache setup complete but verification failed")
            return 1
    else:
        print("\n❌ Cache setup failed")
        return 1

if __name__ == "__main__":
    exit(main())