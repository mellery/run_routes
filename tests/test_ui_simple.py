#!/usr/bin/env python3
"""
Simple UI Test
Test that imports work and basic functionality is available
"""

def test_imports():
    """Test all required imports"""
    print("📦 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import folium
        from streamlit_folium import st_folium
        print("✅ Folium and streamlit-folium imported")
    except ImportError as e:
        print(f"❌ Folium imports failed: {e}")
        return False
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly imported")
    except ImportError as e:
        print(f"❌ Plotly imports failed: {e}")
        return False
    
    try:
        from route import add_elevation_to_graph, add_elevation_to_edges, add_running_weights
        from tsp_solver import RunningRouteOptimizer, RouteObjective
        print("✅ Core route modules imported")
    except ImportError as e:
        print(f"❌ Core module imports failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic TSP functionality without network loading"""
    print("\n🔧 Testing basic functionality...")
    
    try:
        from tsp_solver import RouteObjective
        
        # Test objective enum
        objectives = [
            RouteObjective.MINIMIZE_DISTANCE,
            RouteObjective.MAXIMIZE_ELEVATION,
            RouteObjective.BALANCED_ROUTE,
            RouteObjective.MINIMIZE_DIFFICULTY
        ]
        
        print(f"✅ Route objectives available: {len(objectives)}")
        
        # Test UI helper functions exist
        import running_route_app
        import cli_route_planner
        
        print("✅ UI modules can be imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_file_existence():
    """Test that required files exist"""
    print("\n📁 Testing file existence...")
    
    import os
    
    files_to_check = [
        'srtm_20_05.tif',
        'running_route_app.py',
        'cli_route_planner.py',
        'tsp_solver.py',
        'route.py'
    ]
    
    all_exist = True
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            all_exist = False
    
    return all_exist

def main():
    """Main test function"""
    print("🧪 Simple UI Component Test")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 3
    
    # Test imports
    if test_imports():
        tests_passed += 1
    
    # Test basic functionality
    if test_basic_functionality():
        tests_passed += 1
    
    # Test file existence
    if test_file_existence():
        tests_passed += 1
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ All simple tests passed!")
        print("🎯 Phase 3 UI components ready for use")
        
        print("\n📋 To use the applications:")
        print("  Web app: streamlit run running_route_app.py")
        print("  CLI app: python cli_route_planner.py --interactive")
        
    else:
        print("❌ Some tests failed")

if __name__ == "__main__":
    main()