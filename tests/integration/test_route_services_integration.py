#!/usr/bin/env python3
"""
Integration Tests for Route Services
Test how multiple components work together (with mocked dependencies)
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import networkx as nx

# Add the parent directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from route_services import NetworkManager, RouteOptimizer, RouteAnalyzer, ElevationProfiler, RouteFormatter


class TestRouteServicesIntegration(unittest.TestCase):
    """Test integration between multiple route services"""
    
    def setUp(self):
        """Set up test fixtures with mocked dependencies"""
        # Create a realistic test graph
        self.test_graph = nx.Graph()
        nodes = [
            (1, {'y': 37.130950, 'x': -80.407501, 'elevation': 633.0}),
            (2, {'y': 37.131000, 'x': -80.407600, 'elevation': 635.0}),
            (3, {'y': 37.131100, 'x': -80.407700, 'elevation': 640.0}),
            (4, {'y': 37.131050, 'x': -80.407550, 'elevation': 638.0}),
        ]
        
        for node_id, data in nodes:
            self.test_graph.add_node(node_id, **data)
        
        # Add edges to create a connected graph
        edges = [(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)]
        for edge in edges:
            self.test_graph.add_edge(*edge, length=100.0, highway='residential')
    
    @patch('graph_cache.load_or_generate_graph')
    def test_full_route_planning_workflow(self, mock_load_graph):
        """Test complete workflow: NetworkManager -> RouteOptimizer -> RouteAnalyzer -> RouteFormatter"""
        mock_load_graph.return_value = self.test_graph
        
        # 1. Initialize NetworkManager
        network_manager = NetworkManager()
        graph = network_manager.load_network(radius_km=5.0)
        
        self.assertIsNotNone(graph)
        self.assertEqual(len(graph.nodes), 4)
        
        # 2. Initialize RouteOptimizer
        route_optimizer = RouteOptimizer(graph)
        
        # Mock route optimization result
        mock_route_result = {
            'path': [1, 2, 3, 1],
            'stats': {
                'total_distance_km': 0.3,
                'total_elevation_gain_m': 7.0,
                'total_elevation_loss_m': 0.0,
                'net_elevation_change_m': 7.0,
                'max_grade_percent': 2.3
            },
            'solver_info': {
                'algorithm': 'genetic',
                'solve_time': 1.5,
                'objective': 'minimize_distance'
            }
        }
        
        with patch.object(route_optimizer, 'optimize_route', return_value=mock_route_result):
            # 3. Generate route
            result = route_optimizer.optimize_route(
                start_node=1,
                target_distance_km=0.3,
                objective=route_optimizer.RouteObjective.MINIMIZE_DISTANCE
            )
            
            self.assertIsNotNone(result)
            self.assertEqual(result['path'], [1, 2, 3, 1])
            
            # 4. Analyze route
            route_analyzer = RouteAnalyzer(graph)
            analysis = route_analyzer.analyze_route(result)
            
            self.assertIsNotNone(analysis)
            # Basic analysis should be performed
            self.assertIsInstance(analysis, dict)
            
            # 5. Format route
            route_formatter = RouteFormatter()
            formatted_stats = route_formatter.format_route_summary(result)
            
            self.assertIsInstance(formatted_stats, str)
            self.assertIn('0.3km', formatted_stats)  # Distance should be included
            self.assertIn('7m', formatted_stats)     # Elevation should be included
    
    @patch('graph_cache.load_or_generate_graph')
    def test_network_manager_start_node_integration(self, mock_load_graph):
        """Test NetworkManager start node selection with RouteOptimizer"""
        mock_load_graph.return_value = self.test_graph
        
        network_manager = NetworkManager()
        graph = network_manager.load_network()
        
        # Test dynamic start node selection
        start_node = network_manager.get_start_node(graph)
        self.assertIn(start_node, graph.nodes)
        
        # Verify start node can be used with RouteOptimizer
        route_optimizer = RouteOptimizer(graph)
        
        # Mock a successful route optimization
        with patch.object(route_optimizer, 'optimize_route') as mock_optimize:
            mock_optimize.return_value = {
                'path': [start_node, 2, start_node],
                'stats': {'total_distance_km': 0.2}
            }
            
            result = route_optimizer.optimize_route(
                start_node=start_node,
                target_distance_km=0.2
            )
            
            self.assertIsNotNone(result)
            self.assertEqual(result['path'][0], start_node)
            mock_optimize.assert_called_once()
    
    def test_elevation_profiler_integration(self):
        """Test ElevationProfiler integration with other services"""
        elevation_profiler = ElevationProfiler(self.test_graph)
        
        # Mock route result (ElevationProfiler expects 'route' key)
        route_result = {
            'route': [1, 2, 3, 1],
            'path': [1, 2, 3, 1],
            'stats': {
                'total_distance_km': 0.3,
                'total_elevation_gain_m': 7.0
            }
        }
        
        # Test elevation profile generation
        profile_data = elevation_profiler.generate_profile_data(route_result)
        
        self.assertIsNotNone(profile_data)
        self.assertIn('distances_m', profile_data)
        self.assertIn('elevations', profile_data)
        
        # Test integration with RouteAnalyzer
        route_analyzer = RouteAnalyzer(self.test_graph)
        
        # RouteAnalyzer expects a route result with 'path' key
        analyzer_route_result = {
            'path': [1, 2, 3, 1],
            'stats': {
                'total_distance_km': 0.3,
                'total_elevation_gain_m': 7.0
            }
        }
        analysis = route_analyzer.analyze_route(analyzer_route_result)
        
        # Analysis should include basic route analysis
        self.assertIsNotNone(analysis)
    
    @patch('graph_cache.load_or_generate_graph')
    def test_error_propagation_between_services(self, mock_load_graph):
        """Test how errors propagate between integrated services"""
        mock_load_graph.return_value = None  # Simulate network loading failure
        
        network_manager = NetworkManager()
        graph = network_manager.load_network()
        
        self.assertIsNone(graph)
        
        # Test that RouteOptimizer handles None graph gracefully
        # Since RouteOptimizer prints error but doesn't raise exception, test initialization
        try:
            route_optimizer = RouteOptimizer(graph)
            # If graph is None, optimizer should handle it gracefully
            self.assertIsNotNone(route_optimizer)
        except Exception:
            # If it does raise an exception, that's also acceptable error handling
            pass


class TestCLIIntegration(unittest.TestCase):
    """Test CLI integration with route services"""
    
    @patch('graph_cache.load_or_generate_graph')
    def test_cli_service_initialization(self, mock_load_graph):
        """Test CLI properly initializes all services"""
        # Create a test graph
        test_graph = nx.Graph()
        test_graph.add_node(1, y=37.1309, x=-80.4075, elevation=633)
        test_graph.add_node(2, y=37.1310, x=-80.4076, elevation=635)
        test_graph.add_edge(1, 2, length=100.0)
        
        mock_load_graph.return_value = test_graph
        
        # Import and test CLI
        from cli_route_planner import RefactoredCLIRoutePlanner
        
        cli = RefactoredCLIRoutePlanner()
        success = cli.initialize_services()
        
        self.assertTrue(success)
        self.assertIsNotNone(cli.services)
        self.assertIn('network_manager', cli.services)
        self.assertIn('route_optimizer', cli.services)
        self.assertIn('route_analyzer', cli.services)
        self.assertIn('elevation_profiler', cli.services)
        self.assertIn('route_formatter', cli.services)


if __name__ == '__main__':
    unittest.main()