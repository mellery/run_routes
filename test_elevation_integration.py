#!/usr/bin/env python3
"""
Test script for elevation data source integration
Tests the LocalThreeDEPSource with existing route services
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_elevation_sources():
    """Test elevation data sources integration"""
    
    print("🧪 Testing Elevation Data Sources Integration")
    print("=" * 50)
    
    # Test 1: Import and initialize elevation sources
    print("\n1. Testing imports and initialization...")
    try:
        from elevation_data_sources import (
            ElevationDataManager, 
            LocalThreeDEPSource, 
            SRTMElevationSource,
            HybridElevationSource,
            ElevationConfig
        )
        print("✅ Successfully imported elevation data sources")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Create elevation data manager
    print("\n2. Testing elevation data manager...")
    try:
        manager = ElevationDataManager()
        available_sources = manager.get_available_sources()
        print(f"✅ Manager created. Available sources: {available_sources}")
        
        if available_sources:
            source_info = manager.get_source_info()
            print(f"📊 Source info: {source_info}")
        else:
            print("⚠️ No elevation sources available (expected - no data downloaded yet)")
            
    except Exception as e:
        print(f"❌ Manager creation failed: {e}")
        return False
    
    # Test 3: Test configuration system
    print("\n3. Testing configuration system...")
    try:
        config = ElevationConfig()
        print(f"✅ Default config created:")
        print(f"   Preferred source: {config.preferred_source}")
        print(f"   Fallback enabled: {config.fallback_enabled}")
        print(f"   3DEP directory: {config.threedep_data_directory}")
        print(f"   SRTM file: {config.srtm_file_path}")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    # Test 4: Test local 3DEP source initialization
    print("\n4. Testing Local 3DEP source...")
    try:
        local_3dep = LocalThreeDEPSource()
        tile_info = local_3dep.get_tile_info()
        print(f"✅ Local 3DEP source initialized:")
        print(f"   Tile count: {tile_info['tile_count']}")
        print(f"   Coverage area: {tile_info['total_coverage_area']}")
        print(f"   Resolution: {local_3dep.get_resolution()}m")
        
        if tile_info['tile_count'] == 0:
            print("⚠️ No tiles found - this is expected before download")
        
    except Exception as e:
        print(f"❌ Local 3DEP source test failed: {e}")
        return False
    
    # Test 5: Test SRTM source with existing file
    print("\n5. Testing SRTM source...")
    try:
        srtm_file = "srtm_38_03.tif"
        if os.path.exists(srtm_file):
            srtm_source = SRTMElevationSource(srtm_file)
            print(f"✅ SRTM source initialized with existing file:")
            print(f"   Resolution: {srtm_source.get_resolution()}m")
            print(f"   Coverage bounds: {srtm_source.get_coverage_bounds()}")
            
            # Test elevation lookup
            test_lat, test_lon = 37.1299, -80.4094
            elevation = srtm_source.get_elevation(test_lat, test_lon)
            print(f"   Test elevation at ({test_lat}, {test_lon}): {elevation}m")
        else:
            print(f"⚠️ SRTM file not found: {srtm_file}")
            print("   This is expected if SRTM data hasn't been moved to project root")
            
    except Exception as e:
        print(f"❌ SRTM source test failed: {e}")
        return False
    
    # Test 6: Test hybrid source concept
    print("\n6. Testing hybrid source concept...")
    try:
        # Create mock sources for testing
        local_3dep = LocalThreeDEPSource()
        
        # Only create hybrid if we have SRTM data
        srtm_file = "srtm_38_03.tif"
        if os.path.exists(srtm_file):
            srtm_source = SRTMElevationSource(srtm_file)
            hybrid_source = HybridElevationSource(local_3dep, srtm_source)
            
            print(f"✅ Hybrid source created:")
            print(f"   Resolution: {hybrid_source.get_resolution()}m")
            print(f"   Coverage bounds: {hybrid_source.get_coverage_bounds()}")
            
            # Test stats
            stats = hybrid_source.get_stats()
            print(f"   Initial stats: {stats}")
        else:
            print("⚠️ Skipping hybrid test - no SRTM data available")
            
    except Exception as e:
        print(f"❌ Hybrid source test failed: {e}")
        return False
    
    print("\n✅ All elevation source tests completed successfully!")
    return True

def test_route_services_integration():
    """Test integration with existing route services"""
    
    print("\n🔗 Testing Route Services Integration")
    print("=" * 50)
    
    # Test 1: Check if route services can be imported
    print("\n1. Testing route services import...")
    try:
        from route_services import NetworkManager, RouteOptimizer, ElevationProfiler
        print("✅ Route services imported successfully")
        
        # Test if ElevationProfiler can be enhanced
        print("📊 ElevationProfiler available for enhancement")
        
    except ImportError as e:
        print(f"❌ Route services import failed: {e}")
        print("   This might be expected if route services haven't been refactored yet")
        return False
    
    # Test 2: Check compatibility with existing elevation code
    print("\n2. Testing existing elevation code compatibility...")
    try:
        # Check if route.py exists and can be imported
        import route
        print("✅ Existing route.py imported successfully")
        
    except ImportError as e:
        print(f"❌ Existing route.py import failed: {e}")
        return False
    
    print("\n✅ Route services integration tests completed!")
    return True

def main():
    """Run all integration tests"""
    
    print("🚀 Starting Elevation Data Integration Tests")
    print("=" * 60)
    
    success = True
    
    # Test elevation sources
    if not test_elevation_sources():
        success = False
    
    # Test route services integration
    if not test_route_services_integration():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All integration tests passed!")
        print("\n📋 Next Steps:")
        print("1. Download 3DEP tiles using: python setup_3dep_data.py --download instructions")
        print("2. Index tiles using: python setup_3dep_data.py --index")
        print("3. Test with real data: python setup_3dep_data.py --test")
        print("4. Integrate with existing route services")
    else:
        print("❌ Some integration tests failed")
        print("   Please check the errors above and fix before proceeding")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)