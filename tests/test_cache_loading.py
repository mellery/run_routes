#!/usr/bin/env python3
"""
Test Cache Loading Performance
Compare cached vs uncached graph loading times
"""

import time
import os
from graph_cache import load_or_generate_graph, load_cached_graph

def test_cache_performance():
    """Test cached graph loading performance"""
    
    print("🧪 Testing Cache Loading Performance")
    print("=" * 50)
    
    center_point = (37.1299, -80.4094)
    
    # Test 1: Load from cache if available
    print("\n1️⃣ Testing cached graph loading...")
    start_time = time.time()
    
    try:
        graph = load_or_generate_graph(
            center_point=center_point,
            radius_m=800,
            network_type='all'
        )
        
        load_time = time.time() - start_time
        
        if graph:
            print(f"✅ Graph loaded successfully in {load_time:.2f} seconds")
            print(f"   Stats: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
            
            # Verify elevation data exists
            sample_node = next(iter(graph.nodes(data=True)))[1]
            has_elevation = 'elevation' in sample_node
            
            sample_edge = next(iter(graph.edges(data=True)))[2]
            has_running_weight = 'running_weight' in sample_edge
            
            print(f"   Elevation data: {'✅' if has_elevation else '❌'}")
            print(f"   Running weights: {'✅' if has_running_weight else '❌'}")
            
        else:
            print(f"❌ Graph loading failed after {load_time:.2f} seconds")
            
    except Exception as e:
        print(f"❌ Error during graph loading: {e}")
    
    # Test 2: Check what cache files exist
    print("\n2️⃣ Available cache files:")
    cache_files = [f for f in os.listdir('.') if f.startswith('cached_graph_') and f.endswith('.pkl')]
    
    if cache_files:
        for cache_file in sorted(cache_files):
            file_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
            print(f"   📁 {cache_file} ({file_size_mb:.1f}MB)")
    else:
        print("   📁 No cache files found")
    
    # Test 3: Test specific cache loading
    expected_cache = "cached_graph_37.1299_-80.4094_800m_all.pkl"
    print(f"\n3️⃣ Testing specific cache: {expected_cache}")
    
    if os.path.exists(expected_cache):
        start_time = time.time()
        
        try:
            graph = load_cached_graph(expected_cache)
            load_time = time.time() - start_time
            
            if graph:
                print(f"✅ Cache loaded directly in {load_time:.3f} seconds")
                print(f"   This is the speed boost users will see!")
            else:
                print(f"❌ Cache loading failed")
                
        except Exception as e:
            print(f"❌ Cache loading error: {e}")
    else:
        print(f"❌ Expected cache file not found")
        print("   Run: python generate_cached_graph.py --radius 800 --network-type all")

def main():
    """Main test function"""
    test_cache_performance()
    
    print("\n" + "=" * 50)
    print("🎯 Cache Performance Test Complete")
    print("\n📋 Next steps:")
    print("   • Run setup_cache.py to generate common caches")
    print("   • Test web app: streamlit run running_route_app.py")
    print("   • Test CLI: python cli_route_planner.py --interactive")

if __name__ == "__main__":
    main()