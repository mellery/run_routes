#!/usr/bin/env python3
"""
Unit tests for RouteOptimizer
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import networkx as nx
import sys
import os

# Add the parent directory to sys.path to import route_services
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from route_services.route_optimizer import RouteOptimizer


class TestRouteOptimizer(unittest.TestCase):
    """Test cases for RouteOptimizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a mock graph
        self.mock_graph = nx.Graph()
        self.mock_graph.add_node(1001, x=-80.4094, y=37.1299, elevation=610)
        self.mock_graph.add_node(1002, x=-80.4095, y=37.1300, elevation=615)
        self.mock_graph.add_node(1003, x=-80.4096, y=37.1301, elevation=620)
        self.mock_graph.add_edge(1001, 1002, length=100)
        self.mock_graph.add_edge(1002, 1003, length=150)
    
    @patch('route_services.route_optimizer.FastRunningRouteOptimizer')
    @patch('route_services.route_optimizer.RouteObjective')
    def test_initialization_fast_solver(self, mock_route_objective, mock_fast_solver):
        """Test initialization with fast solver available"""
        optimizer = RouteOptimizer(self.mock_graph)
        
        self.assertEqual(optimizer.graph, self.mock_graph)
        self.assertEqual(optimizer._solver_type, "fast")
        self.assertEqual(optimizer._optimizer_class, mock_fast_solver)
    
    @patch('route_services.route_optimizer.FastRunningRouteOptimizer', side_effect=ImportError)
    @patch('route_services.route_optimizer.RunningRouteOptimizer')
    @patch('route_services.route_optimizer.RouteObjective')
    def test_initialization_standard_solver(self, mock_route_objective, mock_standard_solver, mock_fast_solver):
        """Test initialization fallback to standard solver"""
        optimizer = RouteOptimizer(self.mock_graph)
        
        self.assertEqual(optimizer._solver_type, "standard")
        self.assertEqual(optimizer._optimizer_class, mock_standard_solver)
    
    @patch('route_services.route_optimizer.FastRunningRouteOptimizer', side_effect=ImportError)
    @patch('route_services.route_optimizer.RunningRouteOptimizer', side_effect=ImportError)
    def test_initialization_no_solver(self, mock_standard_solver, mock_fast_solver):
        """Test initialization with no solver available"""
        with self.assertRaises(ImportError):
            RouteOptimizer(self.mock_graph)
    
    @patch('route_services.route_optimizer.FastRunningRouteOptimizer')
    @patch('route_services.route_optimizer.RouteObjective')
    def test_optimize_route_success(self, mock_route_objective, mock_fast_solver):
        """Test successful route optimization"""
        # Mock the optimizer instance and its result
        mock_optimizer_instance = Mock()
        mock_fast_solver.return_value = mock_optimizer_instance
        
        mock_result = {
            'route': [1001, 1002, 1003],
            'stats': {'total_distance_km': 2.5},
            'algorithm': 'nearest_neighbor',
            'objective': 'minimize_distance'
        }
        mock_optimizer_instance.find_optimal_route.return_value = mock_result
        
        optimizer = RouteOptimizer(self.mock_graph)
        result = optimizer.optimize_route(
            start_node=1001,
            target_distance_km=2.5,
            algorithm="nearest_neighbor"
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result['route'], [1001, 1002, 1003])
        self.assertIn('solver_info', result)
        self.assertEqual(result['solver_info']['solver_type'], 'fast')
    
    @patch('route_services.route_optimizer.FastRunningRouteOptimizer')
    @patch('route_services.route_optimizer.RouteObjective')
    def test_optimize_route_invalid_start_node(self, mock_route_objective, mock_fast_solver):
        """Test optimization with invalid start node"""
        optimizer = RouteOptimizer(self.mock_graph)
        result = optimizer.optimize_route(
            start_node=9999,  # Non-existent node
            target_distance_km=2.5
        )
        
        self.assertIsNone(result)
    
    @patch('route_services.route_optimizer.FastRunningRouteOptimizer')
    @patch('route_services.route_optimizer.RouteObjective')
    def test_optimize_route_no_graph(self, mock_route_objective, mock_fast_solver):
        """Test optimization with no graph"""
        optimizer = RouteOptimizer(None)
        result = optimizer.optimize_route(
            start_node=1001,
            target_distance_km=2.5
        )
        
        self.assertIsNone(result)
    
    @patch('route_services.route_optimizer.FastRunningRouteOptimizer')
    @patch('route_services.route_optimizer.RouteObjective')
    def test_optimize_route_exception(self, mock_route_objective, mock_fast_solver):
        """Test optimization with exception"""
        mock_optimizer_instance = Mock()
        mock_fast_solver.return_value = mock_optimizer_instance
        mock_optimizer_instance.find_optimal_route.side_effect = Exception("Optimization error")
        
        optimizer = RouteOptimizer(self.mock_graph)
        result = optimizer.optimize_route(
            start_node=1001,
            target_distance_km=2.5
        )
        
        self.assertIsNone(result)
    
    @patch('route_services.route_optimizer.FastRunningRouteOptimizer')
    @patch('route_services.route_optimizer.RouteObjective')
    def test_get_available_objectives(self, mock_route_objective, mock_fast_solver):
        """Test getting available objectives"""
        optimizer = RouteOptimizer(self.mock_graph)
        objectives = optimizer.get_available_objectives()
        
        expected_keys = [
            "Shortest Route",
            "Maximum Elevation Gain", 
            "Balanced Route",
            "Easiest Route"
        ]
        
        self.assertEqual(list(objectives.keys()), expected_keys)
    
    @patch('route_services.route_optimizer.FastRunningRouteOptimizer')
    @patch('route_services.route_optimizer.RouteObjective')
    def test_get_available_algorithms(self, mock_route_objective, mock_fast_solver):
        """Test getting available algorithms"""
        optimizer = RouteOptimizer(self.mock_graph)
        algorithms = optimizer.get_available_algorithms()
        
        expected_algorithms = ["nearest_neighbor", "genetic"]
        self.assertEqual(algorithms, expected_algorithms)
    
    @patch('route_services.route_optimizer.FastRunningRouteOptimizer')
    @patch('route_services.route_optimizer.RouteObjective')
    def test_validate_parameters_valid(self, mock_route_objective, mock_fast_solver):
        """Test parameter validation with valid parameters"""
        optimizer = RouteOptimizer(self.mock_graph)
        validation = optimizer.validate_parameters(
            start_node=1001,
            target_distance_km=2.5,
            algorithm="nearest_neighbor"
        )
        
        self.assertTrue(validation['valid'])
        self.assertEqual(validation['errors'], [])
    
    @patch('route_services.route_optimizer.FastRunningRouteOptimizer')
    @patch('route_services.route_optimizer.RouteObjective')
    def test_validate_parameters_invalid(self, mock_route_objective, mock_fast_solver):
        """Test parameter validation with invalid parameters"""
        optimizer = RouteOptimizer(self.mock_graph)
        validation = optimizer.validate_parameters(
            start_node=9999,  # Invalid node
            target_distance_km=-1,  # Invalid distance
            algorithm="invalid_algorithm"
        )
        
        self.assertFalse(validation['valid'])
        self.assertGreater(len(validation['errors']), 0)
        self.assertIn("Start node 9999 not found in graph", validation['errors'])
        self.assertIn("Target distance must be positive", validation['errors'])
        self.assertIn("Unknown algorithm: invalid_algorithm", validation['errors'])
    
    @patch('route_services.route_optimizer.FastRunningRouteOptimizer')
    @patch('route_services.route_optimizer.RouteObjective')
    def test_validate_parameters_warnings(self, mock_route_objective, mock_fast_solver):
        """Test parameter validation with warnings"""
        optimizer = RouteOptimizer(self.mock_graph)
        validation = optimizer.validate_parameters(
            start_node=1001,
            target_distance_km=25.0  # Very long distance
        )
        
        self.assertTrue(validation['valid'])
        self.assertGreater(len(validation['warnings']), 0)
    
    @patch('route_services.route_optimizer.FastRunningRouteOptimizer')
    @patch('route_services.route_optimizer.RouteObjective')
    def test_get_solver_info(self, mock_route_objective, mock_fast_solver):
        """Test getting solver information"""
        optimizer = RouteOptimizer(self.mock_graph)
        info = optimizer.get_solver_info()
        
        expected_keys = [
            'solver_type',
            'solver_class',
            'available_objectives',
            'available_algorithms',
            'graph_nodes',
            'graph_edges'
        ]
        
        for key in expected_keys:
            self.assertIn(key, info)
        
        self.assertEqual(info['solver_type'], 'fast')
        self.assertEqual(info['graph_nodes'], 3)
        self.assertEqual(info['graph_edges'], 2)
    
    @patch('route_services.route_optimizer.FastRunningRouteOptimizer')
    @patch('route_services.route_optimizer.RouteObjective')
    def test_solver_type_property(self, mock_route_objective, mock_fast_solver):
        """Test solver_type property"""
        optimizer = RouteOptimizer(self.mock_graph)
        self.assertEqual(optimizer.solver_type, 'fast')
    
    @patch('route_services.route_optimizer.FastRunningRouteOptimizer')
    @patch('route_services.route_optimizer.RouteObjective')
    def test_route_objective_property(self, mock_route_objective, mock_fast_solver):
        """Test RouteObjective property"""
        optimizer = RouteOptimizer(self.mock_graph)
        self.assertEqual(optimizer.RouteObjective, mock_route_objective)


if __name__ == '__main__':
    unittest.main()