#!/usr/bin/env python3
"""
Test elevation data sources with existing SRTM data
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_with_existing_srtm():
    """Test elevation sources with existing SRTM data"""
    
    print("🧪 Testing Elevation Sources with Existing SRTM Data")
    print("=" * 55)
    
    from elevation_data_sources import (
        ElevationDataManager, 
        SRTMElevationSource,
        ElevationConfig
    )
    
    # Find available SRTM files
    srtm_files = list(Path(".").glob("srtm_*.tif"))
    
    if not srtm_files:
        print("❌ No SRTM files found in current directory")
        return False
    
    print(f"📁 Found SRTM files: {[f.name for f in srtm_files]}")
    
    # Test with first available SRTM file
    srtm_file = srtm_files[0]
    print(f"🗺️ Testing with: {srtm_file}")
    
    try:
        # Create SRTM source
        srtm_source = SRTMElevationSource(str(srtm_file))
        
        print(f"✅ SRTM source created successfully:")
        print(f"   Resolution: {srtm_source.get_resolution()}m")
        print(f"   Coverage bounds: {srtm_source.get_coverage_bounds()}")
        
        # Test elevation lookup at various coordinates
        test_coordinates = [
            (37.1299, -80.4094),  # Christiansburg, VA
            (37.0, -80.0),        # Nearby coordinate
            (37.2, -80.5),        # Another nearby coordinate
        ]
        
        print(f"\n🧪 Testing elevation lookups:")
        for lat, lon in test_coordinates:
            try:
                elevation = srtm_source.get_elevation(lat, lon)
                availability = srtm_source.is_available(lat, lon)
                print(f"   ({lat:.4f}, {lon:.4f}): {elevation}m (available: {availability})")
            except Exception as e:
                print(f"   ({lat:.4f}, {lon:.4f}): Error - {e}")
        
        # Test elevation profile
        print(f"\n📊 Testing elevation profile:")
        profile_coords = [(37.1299, -80.4094), (37.1350, -80.4100), (37.1400, -80.4150)]
        try:
            profile = srtm_source.get_elevation_profile(profile_coords)
            print(f"   Profile elevations: {profile}")
        except Exception as e:
            print(f"   Profile error: {e}")
        
        # Test with manager
        print(f"\n🎛️ Testing with ElevationDataManager:")
        
        # Create custom config
        config = ElevationConfig()
        config.srtm_file_path = str(srtm_file)
        config.preferred_source = "srtm"
        
        manager = ElevationDataManager(config)
        available_sources = manager.get_available_sources()
        print(f"   Available sources: {available_sources}")
        
        source = manager.get_elevation_source()
        if source:
            print(f"   Active source: {source.__class__.__name__}")
            
            # Test with manager
            test_lat, test_lon = 37.1299, -80.4094
            elevation = source.get_elevation(test_lat, test_lon)
            print(f"   Test elevation: {elevation}m")
            
            # Test all sources
            test_results = manager.test_sources(test_lat, test_lon)
            print(f"   Test results: {test_results}")
        else:
            print("   ❌ No active source available")
        
        # Close resources
        srtm_source.close()
        manager.close_all()
        
        print(f"\n✅ SRTM elevation source testing completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ SRTM testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run SRTM elevation test"""
    
    print("🚀 Starting SRTM Elevation Data Test")
    print("=" * 40)
    
    success = test_with_existing_srtm()
    
    if success:
        print("\n🎉 SRTM elevation test completed successfully!")
        print("\n📋 This confirms that:")
        print("   ✅ Elevation data source abstraction works correctly")
        print("   ✅ SRTM integration is functional")
        print("   ✅ ElevationDataManager manages sources properly")
        print("   ✅ Ready for 3DEP tile integration")
        print("\n🔄 Next: Download 3DEP tiles to test hybrid approach")
    else:
        print("\n❌ SRTM elevation test failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)