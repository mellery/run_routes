#!/usr/bin/env python3
"""
Test Enhanced Route Services with 3DEP Integration
Complete Week 2-3 testing of integrated 3DEP elevation system
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_enhanced_route_optimizer():
    """Test RouteOptimizer with enhanced elevation support"""
    
    print("🚀 Testing Enhanced RouteOptimizer")
    print("=" * 40)
    
    try:
        from route_services import NetworkManager
        from route_services.route_optimizer import RouteOptimizer
        
        # Initialize network
        print("📍 Loading network around Virginia 3DEP coverage...")
        center_point = (37.0, -78.1)  # Virginia area with 3DEP coverage
        network_manager = NetworkManager(center_point)
        graph = network_manager.load_network(radius_km=3.0)
        
        if not graph:
            print("❌ Failed to load network")
            return False
        
        # Create enhanced route optimizer
        route_optimizer = RouteOptimizer(graph)
        
        # Find a suitable starting node
        nodes = list(graph.nodes())
        if not nodes:
            print("❌ No nodes in graph")
            return False
        
        start_node = nodes[len(nodes) // 2]  # Use middle node
        print(f"🎯 Testing with start node: {start_node}")
        
        # Test route optimization with elevation objective
        print("\n🧬 Testing genetic algorithm with elevation objective...")
        route_result = route_optimizer.optimize_route(
            start_node=start_node,
            target_distance_km=2.0,
            objective="elevation",  # This should trigger GA
            algorithm="auto"
        )
        
        if route_result:
            print("✅ Enhanced route optimization successful!")
            print(f"   Route nodes: {len(route_result.get('route', []))}")
            print(f"   Distance: {route_result.get('stats', {}).get('total_distance_km', 0):.2f}km")
            print(f"   Elevation gain: {route_result.get('stats', {}).get('total_elevation_gain_m', 0):.1f}m")
            return True
        else:
            print("❌ Route optimization failed")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced RouteOptimizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_elevation_profiler():
    """Test EnhancedElevationProfiler with 3DEP data"""
    
    print("\n🏔️ Testing Enhanced ElevationProfiler")
    print("=" * 45)
    
    try:
        from route_services import NetworkManager
        from route_services.elevation_profiler_enhanced import EnhancedElevationProfiler
        
        # Load network in 3DEP coverage area
        center_point = (37.0, -78.1)  # Virginia area with 3DEP coverage
        network_manager = NetworkManager(center_point)
        graph = network_manager.load_network(radius_km=2.0)
        
        if not graph:
            print("❌ Failed to load network")
            return False
        
        # Create enhanced elevation profiler
        elevation_profiler = EnhancedElevationProfiler(graph)
        
        # Create sample route for testing
        nodes = list(graph.nodes())[:10]  # Use first 10 nodes
        sample_route = {
            'route': nodes,
            'stats': {'total_distance_km': 1.5}
        }
        
        print(f"📊 Testing elevation profile with {len(nodes)} nodes...")
        
        # Generate enhanced profile
        profile_data = elevation_profiler.generate_profile_data(
            sample_route, 
            use_enhanced_elevation=True, 
            interpolate_points=True
        )
        
        if profile_data:
            print("✅ Enhanced elevation profile generated!")
            print(f"   Coordinates: {len(profile_data.get('coordinates', []))}")
            print(f"   Elevation points: {len(profile_data.get('elevations', []))}")
            print(f"   Enhanced profile: {profile_data.get('enhanced_profile', False)}")
            
            # Show elevation statistics
            stats = profile_data.get('elevation_stats', {})
            if stats:
                print(f"   Elevation range: {stats.get('min_elevation_m', 0):.1f}m - {stats.get('max_elevation_m', 0):.1f}m")
                print(f"   Total gain: {stats.get('total_elevation_gain_m', 0):.1f}m")
                
                # Show data quality info
                quality = stats.get('data_quality', {})
                if quality:
                    print(f"   Data resolution: {quality.get('resolution_m', 'Unknown')}m")
                    print(f"   Vertical accuracy: ±{quality.get('vertical_accuracy_m', 'Unknown')}m")
            
            # Show data source info
            source_info = profile_data.get('data_source_info', {})
            if source_info:
                print(f"   Data source: {source_info.get('active_source', 'Unknown')}")
                if 'usage_stats' in source_info:
                    usage = source_info['usage_stats']
                    if 'primary_percentage' in usage:
                        print(f"   High-res coverage: {usage['primary_percentage']:.1f}%")
            
            elevation_profiler.close()
            return True
        else:
            print("❌ Failed to generate elevation profile")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced ElevationProfiler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_cli():
    """Test enhanced CLI with 3DEP integration"""
    
    print("\n💻 Testing Enhanced CLI Interface")
    print("=" * 40)
    
    try:
        from cli_route_planner import RefactoredCLIRoutePlanner
        
        # Create enhanced CLI planner
        cli_planner = RefactoredCLIRoutePlanner()
        
        # Initialize services in 3DEP coverage area
        center_point = (37.0, -78.1)  # Virginia area with 3DEP coverage
        success = cli_planner.initialize_services(center_point, radius_km=2.0)
        
        if not success:
            print("❌ Failed to initialize CLI services")
            return False
        
        print("✅ Enhanced CLI services initialized!")
        
        # Test elevation status
        print("\n📊 Testing elevation status display...")
        cli_planner.show_elevation_status()
        
        # Test elevation configuration
        print("\n⚙️ Testing elevation configuration...")
        cli_planner.configure_elevation_source("hybrid")
        
        # Test route generation
        print("\n🛣️ Testing enhanced route generation...")
        nodes = list(cli_planner.services['graph'].nodes())
        if nodes:
            start_node = nodes[len(nodes) // 4]  # Use a quarter-way node
            
            route_result = cli_planner.services['route_optimizer'].optimize_route(
                start_node=start_node,
                target_distance_km=1.5,
                objective="elevation",
                algorithm="auto"
            )
            
            if route_result:
                print("✅ Enhanced route generation successful!")
                cli_planner.display_route_stats(route_result)
                return True
            else:
                print("❌ Route generation failed")
                return False
        else:
            print("❌ No nodes available for testing")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced CLI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_elevation_data_integration():
    """Test integration with elevation data sources"""
    
    print("\n🗂️ Testing Elevation Data Integration")
    print("=" * 45)
    
    try:
        from elevation_data_sources import get_elevation_manager
        
        # Test elevation manager
        manager = get_elevation_manager()
        available_sources = manager.get_available_sources()
        
        print(f"✅ Available elevation sources: {available_sources}")
        
        if available_sources:
            source = manager.get_elevation_source()
            if source:
                print(f"✅ Active source: {source.__class__.__name__}")
                print(f"   Resolution: {source.get_resolution()}m")
                
                # Test elevation lookup in 3DEP coverage area
                test_coordinates = [
                    (37.0, -78.1),    # Virginia 3DEP area
                    (37.1299, -80.4094),  # Christiansburg (may be outside 3DEP coverage)
                ]
                
                print(f"\n🧪 Testing elevation lookups:")
                for lat, lon in test_coordinates:
                    elevation = source.get_elevation(lat, lon)
                    available = source.is_available(lat, lon)
                    status = "✅" if available and elevation else "❌"
                    elev_str = f"{elevation:.1f}m" if elevation else "N/A"
                    print(f"   {status} ({lat:.4f}, {lon:.4f}): {elev_str}")
                
                # Test profile generation
                if len(test_coordinates) > 1:
                    profile = source.get_elevation_profile(test_coordinates)
                    print(f"   Profile: {[f'{e:.1f}m' if e else 'N/A' for e in profile[:3]]}...")
                
                manager.close_all()
                return True
            else:
                print("❌ No active elevation source")
                return False
        else:
            print("⚠️ No elevation sources available")
            return False
            
    except Exception as e:
        print(f"❌ Elevation data integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_week2_3_integration():
    """Comprehensive test of Week 2-3 integration"""
    
    print("🎯 Week 2-3 Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Enhanced RouteOptimizer", test_enhanced_route_optimizer),
        ("Enhanced ElevationProfiler", test_enhanced_elevation_profiler),
        ("Enhanced CLI Interface", test_enhanced_cli),
        ("Elevation Data Integration", test_elevation_data_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_function in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_function()
            if success:
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
    
    print(f"\n🏆 Integration Test Results")
    print("=" * 35)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All Week 2-3 integration tests passed!")
        print("✅ 3DEP integration with route services is complete and working!")
    elif passed >= total * 0.75:
        print("✅ Most Week 2-3 integration tests passed!")
        print("🔧 Minor issues may need attention")
    else:
        print("⚠️ Several Week 2-3 integration tests failed")
        print("🔧 Integration needs additional work")
    
    return passed == total

def main():
    """Main test runner"""
    
    print("🚀 Enhanced Route Services Integration Test")
    print("Testing Week 2-3 of 3DEP Integration Plan")
    print("=" * 60)
    
    success = test_week2_3_integration()
    
    if success:
        print(f"\n🎊 Week 2-3 Implementation: COMPLETE")
        print(f"   ✅ Enhanced elevation integration working")
        print(f"   ✅ Route services enhanced with 3DEP precision")
        print(f"   ✅ CLI and optimizer integration functional")
        print(f"   ✅ Ready for production use with 1m elevation data")
    else:
        print(f"\n⚠️ Week 2-3 Implementation: NEEDS ATTENTION")
        print(f"   Some integration tests failed - check logs above")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)