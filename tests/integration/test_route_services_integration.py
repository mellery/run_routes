#!/usr/bin/env python3
"""
Integration tests for route services
Tests the route services working together end-to-end
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import networkx as nx
import sys
import os

# Add the parent directory to sys.path to import route_services
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from route_services import (
    NetworkManager, RouteOptimizer, RouteAnalyzer, 
    ElevationProfiler, RouteFormatter
)


class TestRouteServicesIntegration(unittest.TestCase):
    """Integration tests for route services working together"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a realistic mock graph
        self.mock_graph = nx.Graph()
        
        # Add nodes with realistic coordinates and elevations
        nodes_data = [
            (1001, -80.4094, 37.1299, 610),
            (1002, -80.4095, 37.1300, 620),
            (1003, -80.4096, 37.1301, 615),
            (1004, -80.4097, 37.1302, 630),
            (1005, -80.4098, 37.1303, 625),
            (1006, -80.4099, 37.1304, 635)
        ]
        
        for node_id, lon, lat, elevation in nodes_data:
            self.mock_graph.add_node(node_id, x=lon, y=lat, elevation=elevation)
        
        # Add edges
        edges = [
            (1001, 1002, 100),
            (1002, 1003, 120),
            (1003, 1004, 110),
            (1004, 1005, 90),
            (1005, 1006, 105),
            (1006, 1001, 200)  # Close the loop
        ]
        
        for node1, node2, length in edges:
            self.mock_graph.add_edge(node1, node2, length=length)
        
        # Mock route result from optimizer
        self.mock_route_result = {
            'route': [1001, 1002, 1003, 1004, 1005],
            'stats': {
                'total_distance_km': 2.5,
                'total_elevation_gain_m': 45,
                'total_elevation_loss_m': 15,
                'net_elevation_gain_m': 30,
                'max_grade_percent': 8.2,
                'estimated_time_min': 15
            },
            'cost': 2500,
            'solve_time': 3.2,
            'algorithm': 'nearest_neighbor',
            'objective': 'maximize_elevation',
            'target_distance_km': 2.5,
            'solver_info': {
                'solver_type': 'fast',
                'solve_time': 3.2,
                'algorithm_used': 'nearest_neighbor',
                'objective_used': 'maximize_elevation'
            }
        }
    
    @patch('graph_cache.load_or_generate_graph')
    def test_complete_route_planning_workflow(self, mock_load_graph):
        """Test complete route planning workflow using all services"""
        mock_load_graph.return_value = self.mock_graph
        
        # Step 1: Load network using NetworkManager
        network_manager = NetworkManager()
        graph = network_manager.load_network(radius_km=1.0)
        
        self.assertIsNotNone(graph)
        self.assertEqual(graph, self.mock_graph)
        
        # Step 2: Optimize route using RouteOptimizer
        with patch('tsp_solver_fast.FastRunningRouteOptimizer') as mock_optimizer_class:
            mock_optimizer_instance = Mock()
            mock_optimizer_class.return_value = mock_optimizer_instance
            mock_optimizer_instance.find_optimal_route.return_value = self.mock_route_result
            
            route_optimizer = RouteOptimizer(graph)
            route_result = route_optimizer.optimize_route(
                start_node=1001,
                target_distance_km=2.5,
                algorithm="nearest_neighbor"
            )
            
            self.assertIsNotNone(route_result)
            self.assertEqual(route_result['route'], [1001, 1002, 1003, 1004, 1005])
            self.assertIn('solver_info', route_result)
        
        # Step 3: Analyze route using RouteAnalyzer
        route_analyzer = RouteAnalyzer(graph)
        analysis = route_analyzer.analyze_route(route_result)
        
        self.assertIn('basic_stats', analysis)
        self.assertIn('additional_stats', analysis)
        self.assertIn('route_info', analysis)
        
        # Generate directions
        directions = route_analyzer.generate_directions(route_result)
        self.assertGreater(len(directions), 0)
        self.assertEqual(directions[0]['type'], 'start')
        self.assertEqual(directions[-1]['type'], 'finish')
        
        # Get difficulty rating
        difficulty = route_analyzer.get_route_difficulty_rating(route_result)
        self.assertIn('rating', difficulty)
        self.assertIn('score', difficulty)
        
        # Step 4: Generate elevation profile using ElevationProfiler
        elevation_profiler = ElevationProfiler(graph)
        
        with patch('route.haversine_distance') as mock_haversine:
            mock_haversine.return_value = 100  # constant distance
            
            profile_data = elevation_profiler.generate_profile_data(route_result)
            
            self.assertIn('elevations', profile_data)
            self.assertIn('distances_km', profile_data)
            self.assertIn('elevation_stats', profile_data)
            
            # Find peaks and valleys
            peaks_valleys = elevation_profiler.find_elevation_peaks_valleys(route_result)
            self.assertIn('peaks', peaks_valleys)
            self.assertIn('valleys', peaks_valleys)
        
        # Step 5: Format results using RouteFormatter
        route_formatter = RouteFormatter()
        
        # Format CLI output
        cli_stats = route_formatter.format_route_stats_cli(route_result, {'difficulty': difficulty})
        self.assertIn('Route Statistics:', cli_stats)
        self.assertIn('Distance:', cli_stats)
        
        # Format web output
        web_stats = route_formatter.format_route_stats_web(route_result, {'difficulty': difficulty})
        self.assertIn('distance', web_stats)
        self.assertIn('difficulty', web_stats)
        
        # Format directions
        formatted_directions = route_formatter.format_directions_web(directions)
        self.assertEqual(len(formatted_directions), len(directions))
        
        # Export to JSON
        json_export = route_formatter.export_route_json(
            route_result, analysis, directions, profile_data
        )
        self.assertIsInstance(json_export, str)
        
        # Create summary
        summary = route_formatter.format_route_summary(route_result)
        self.assertIn('2.5km', summary)
        self.assertIn('45m', summary)
    
    @patch('graph_cache.load_or_generate_graph')
    def test_network_manager_caching_integration(self, mock_load_graph):
        """Test network manager caching with multiple service instances"""
        mock_load_graph.return_value = self.mock_graph
        
        # Create network manager and load graph
        network_manager = NetworkManager()
        graph1 = network_manager.load_network(radius_km=1.0)
        
        # Load same graph again - should use cache
        graph2 = network_manager.load_network(radius_km=1.0)
        
        # Should only call load_or_generate_graph once
        self.assertEqual(mock_load_graph.call_count, 1)
        self.assertEqual(graph1, graph2)
        
        # Create multiple service instances with same graph
        route_optimizer = RouteOptimizer(graph1)
        route_analyzer = RouteAnalyzer(graph1)
        elevation_profiler = ElevationProfiler(graph1)
        
        # All should reference the same graph
        self.assertEqual(route_optimizer.graph, graph1)
        self.assertEqual(route_analyzer.graph, graph1)
        self.assertEqual(elevation_profiler.graph, graph1)
    
    def test_route_analysis_consistency(self):
        """Test consistency between analyzer and profiler data"""
        route_analyzer = RouteAnalyzer(self.mock_graph)
        elevation_profiler = ElevationProfiler(self.mock_graph)
        
        with patch('route.haversine_distance') as mock_haversine1:
            
            # Use constant distance for consistency
            mock_haversine1.return_value = 100
            
            # Analyze route
            analysis = route_analyzer.analyze_route(self.mock_route_result)
            directions = route_analyzer.generate_directions(self.mock_route_result)
            
            # Generate profile
            profile_data = elevation_profiler.generate_profile_data(self.mock_route_result)
            
            # Check consistency
            route_length = len(self.mock_route_result['route'])
            self.assertEqual(analysis['route_info']['route_length'], route_length)
            self.assertEqual(len(profile_data['coordinates']), route_length)
            
            # Directions should have start + segments + finish
            expected_direction_count = route_length + 1  # +1 for finish
            self.assertEqual(len(directions), expected_direction_count)
    
    @patch('tsp_solver_fast.FastRunningRouteOptimizer')
    def test_optimizer_validation_integration(self, mock_optimizer_class):
        """Test route optimizer validation with network manager"""
        mock_optimizer_instance = Mock()
        mock_optimizer_class.return_value = mock_optimizer_instance
        mock_optimizer_instance.find_optimal_route.return_value = self.mock_route_result
        
        network_manager = NetworkManager()
        route_optimizer = RouteOptimizer(self.mock_graph)
        
        # Test validation with valid node
        validation = route_optimizer.validate_parameters(
            start_node=1001,
            target_distance_km=2.5
        )
        self.assertTrue(validation['valid'])
        
        # Test validation with invalid node
        validation = route_optimizer.validate_parameters(
            start_node=9999,  # Not in graph
            target_distance_km=2.5
        )
        self.assertFalse(validation['valid'])
        
        # Test validation with network manager's node validation
        self.assertTrue(network_manager.validate_node_exists(self.mock_graph, 1001))
        self.assertFalse(network_manager.validate_node_exists(self.mock_graph, 9999))
    
    def test_formatter_output_consistency(self):
        """Test formatter output consistency across different formats"""
        route_analyzer = RouteAnalyzer(self.mock_graph)
        route_formatter = RouteFormatter()
        
        # Generate analysis data
        analysis = route_analyzer.analyze_route(self.mock_route_result)
        difficulty = route_analyzer.get_route_difficulty_rating(self.mock_route_result)
        directions = route_analyzer.generate_directions(self.mock_route_result)
        
        # Format in different ways
        cli_stats = route_formatter.format_route_stats_cli(
            self.mock_route_result, {'difficulty': difficulty}
        )
        web_stats = route_formatter.format_route_stats_web(
            self.mock_route_result, {'difficulty': difficulty}
        )
        
        # Extract same information from both formats
        # Distance should be consistent
        self.assertIn('2.50 km', cli_stats)
        self.assertEqual(web_stats['distance']['value'], '2.50 km')
        
        # Elevation gain should be consistent
        self.assertIn('45 m', cli_stats)
        self.assertEqual(web_stats['elevation_gain']['value'], '45 m')
        
        # Format directions consistently
        cli_directions = route_formatter.format_directions_cli(directions)
        web_directions = route_formatter.format_directions_web(directions)
        
        # Should have same number of steps
        step_count_cli = cli_directions.count('1. ') + cli_directions.count('2. ') + cli_directions.count('3. ')
        step_count_web = len(web_directions)
        self.assertEqual(len(directions), step_count_web)
    
    def test_error_propagation_integration(self):
        """Test how errors propagate through the service chain"""
        # Test with None graph
        route_optimizer = RouteOptimizer(None)
        result = route_optimizer.optimize_route(1001, 2.5)
        self.assertIsNone(result)
        
        # Test with empty route result
        route_analyzer = RouteAnalyzer(self.mock_graph)
        analysis = route_analyzer.analyze_route(None)
        self.assertEqual(analysis, {})
        
        directions = route_analyzer.generate_directions({})
        self.assertEqual(directions, [])
        
        # Test formatter with empty data
        route_formatter = RouteFormatter()
        stats = route_formatter.format_route_stats_cli(None)
        self.assertEqual(stats, "❌ No route data available")
        
        web_stats = route_formatter.format_route_stats_web(None)
        self.assertEqual(web_stats, {})
    
    @patch('graph_cache.load_or_generate_graph')
    def test_service_factory_pattern(self, mock_load_graph):
        """Test creating all services from a single network load"""
        mock_load_graph.return_value = self.mock_graph
        
        # Factory function to create all services
        def create_route_services(center_point=None, radius_km=1.0):
            """Factory to create all route services with shared graph"""
            network_manager = NetworkManager(center_point)
            graph = network_manager.load_network(radius_km)
            
            if not graph:
                return None
            
            return {
                'network_manager': network_manager,
                'route_optimizer': RouteOptimizer(graph),
                'route_analyzer': RouteAnalyzer(graph),
                'elevation_profiler': ElevationProfiler(graph),
                'route_formatter': RouteFormatter(),
                'graph': graph
            }
        
        # Create services
        services = create_route_services()
        self.assertIsNotNone(services)
        
        # Verify all services were created
        expected_services = [
            'network_manager', 'route_optimizer', 'route_analyzer',
            'elevation_profiler', 'route_formatter', 'graph'
        ]
        
        for service_name in expected_services:
            self.assertIn(service_name, services)
        
        # Verify all services share the same graph
        graph = services['graph']
        self.assertEqual(services['route_optimizer'].graph, graph)
        self.assertEqual(services['route_analyzer'].graph, graph)
        self.assertEqual(services['elevation_profiler'].graph, graph)


if __name__ == '__main__':
    unittest.main()