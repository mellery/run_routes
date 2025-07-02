#!/usr/bin/env python3
"""
Smoke Tests - Real Dependencies
Quick validation that services work with actual dependencies
"""

import unittest
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import networkx as nx
    from route_services import NetworkManager, RouteOptimizer, RouteAnalyzer, ElevationProfiler, RouteFormatter
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Dependencies not available for smoke tests")
class SmokeTests(unittest.TestCase):
    """Quick tests using real dependencies to validate basic functionality"""
    
    def setUp(self):
        """Set up real test data"""
        # Create a simple real graph
        self.graph = nx.Graph()
        self.graph.add_node(1001, x=-80.4094, y=37.1299, elevation=610)
        self.graph.add_node(1002, x=-80.4095, y=37.1300, elevation=615)
        self.graph.add_node(1003, x=-80.4096, y=37.1301, elevation=620)
        self.graph.add_edge(1001, 1002, length=100)
        self.graph.add_edge(1002, 1003, length=150)
        self.graph.add_edge(1003, 1001, length=120)
        
        # Real route result (simplified)
        self.route_result = {
            'route': [1001, 1002, 1003, 1001],
            'stats': {
                'total_distance_km': 0.37,
                'total_elevation_gain_m': 10,
                'estimated_time_min': 3
            },
            'algorithm': 'nearest_neighbor',
            'objective': 'minimize_distance'
        }
    
    def test_network_manager_smoke(self):
        """Smoke test: NetworkManager basic functionality"""
        manager = NetworkManager()
        
        # Test basic properties
        self.assertEqual(manager.center_point, (37.1299, -80.4094))
        self.assertEqual(manager._graph_cache, {})
        
        # Test network stats with real graph
        stats = manager.get_network_stats(self.graph)
        self.assertEqual(stats['nodes'], 3)
        self.assertEqual(stats['edges'], 3)
        self.assertTrue(stats['has_elevation'])
    
    def test_route_optimizer_smoke(self):
        """Smoke test: RouteOptimizer basic functionality"""
        optimizer = RouteOptimizer(self.graph)
        
        # Should initialize with real TSP solver
        self.assertIn(optimizer._solver_type, ['fast', 'standard'])
        self.assertIsNotNone(optimizer._optimizer_class)
        
        # Test getting solver info
        info = optimizer.get_solver_info()
        self.assertIn('solver_type', info)
        self.assertIn('solver_class', info)
    
    def test_route_analyzer_smoke(self):
        """Smoke test: RouteAnalyzer with real haversine calculations"""
        analyzer = RouteAnalyzer(self.graph)
        
        # Test route analysis (uses real haversine_distance)
        analysis = analyzer.analyze_route(self.route_result)
        
        # Should have basic analysis structure (real implementation)
        self.assertIn('additional_stats', analysis)
        self.assertIn('total_segments', analysis['additional_stats'])
        
        # Test directions generation (uses real haversine_distance)
        directions = analyzer.generate_directions(self.route_result)
        self.assertIsInstance(directions, list)
        self.assertGreater(len(directions), 0)
        
        # Test difficulty rating
        difficulty = analyzer.get_route_difficulty_rating(self.route_result)
        self.assertIn('rating', difficulty)
        self.assertIn('score', difficulty)
    
    def test_elevation_profiler_smoke(self):
        """Smoke test: ElevationProfiler with real calculations"""
        profiler = ElevationProfiler(self.graph)
        
        # Test profile generation (uses real haversine_distance)
        profile_data = profiler.generate_profile_data(self.route_result)
        
        # Should have profile structure
        self.assertIn('elevations', profile_data)
        self.assertIn('distances_km', profile_data)
        self.assertIn('coordinates', profile_data)
        
        # Distances should be realistic (not just mock values)
        distances = profile_data['distances_km']
        self.assertGreater(len(distances), 0)
        self.assertEqual(distances[0], 0.0)  # Should start at 0
        
        # Test elevation zones (real method available)
        zones = profiler.get_elevation_zones(self.route_result, zone_count=3)
        self.assertIsInstance(zones, list)
    
    def test_route_formatter_smoke(self):
        """Smoke test: RouteFormatter basic functionality"""
        formatter = RouteFormatter()
        
        # Test CLI formatting
        cli_output = formatter.format_route_stats_cli(self.route_result)
        self.assertIsInstance(cli_output, str)
        self.assertIn('Route Statistics', cli_output)
        
        # Test web formatting
        web_output = formatter.format_route_stats_web(self.route_result)
        self.assertIsInstance(web_output, dict)
        self.assertIn('distance', web_output)
        
        # Test JSON export
        json_output = formatter.export_route_json(self.route_result)
        self.assertIsInstance(json_output, str)
        self.assertIn('route', json_output)
    
    def test_services_integration_smoke(self):
        """Smoke test: All services work together with real dependencies"""
        # Initialize all services
        network_manager = NetworkManager()
        route_optimizer = RouteOptimizer(self.graph)
        route_analyzer = RouteAnalyzer(self.graph)
        elevation_profiler = ElevationProfiler(self.graph)
        route_formatter = RouteFormatter()
        
        # Test workflow: analyze → profile → format
        analysis = route_analyzer.analyze_route(self.route_result)
        profile_data = elevation_profiler.generate_profile_data(self.route_result)
        formatted_output = route_formatter.format_route_stats_cli(self.route_result, analysis)
        
        # All should produce valid output
        self.assertIsInstance(analysis, dict)
        self.assertIsInstance(profile_data, dict)
        self.assertIsInstance(formatted_output, str)
        
        print("✅ All services work together with real dependencies!")


class DependencyTest(unittest.TestCase):
    """Test dependency availability"""
    
    def test_dependencies_available(self):
        """Test that all required dependencies are available"""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest(f"Dependencies not available: {IMPORT_ERROR}")
        
        # Test specific imports
        import networkx
        import numpy
        
        # Optional: test osmnx if available
        try:
            import osmnx
            osmnx_available = True
        except ImportError:
            osmnx_available = False
        
        print(f"✅ NetworkX: {networkx.__version__}")
        print(f"✅ NumPy: {numpy.__version__}")
        print(f"{'✅' if osmnx_available else '⚠️'} OSMnx: {'Available' if osmnx_available else 'Not available'}")


if __name__ == '__main__':
    print("🔥 Running Smoke Tests with Real Dependencies")
    print("=" * 60)
    unittest.main(verbosity=2)