#!/usr/bin/env python3
"""
GA Integration Tests
Test genetic algorithm integration with route planning system
"""

import unittest
import time
import networkx as nx
from unittest.mock import Mock, patch

# Import route services
from route_services import NetworkManager, RouteOptimizer, RouteAnalyzer, ElevationProfiler


class TestGAIntegration(unittest.TestCase):
    """Test GA integration with route services"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test graph
        self.test_graph = self._create_test_graph()
        
        # Initialize services
        self.route_optimizer = RouteOptimizer(self.test_graph)
        self.route_analyzer = RouteAnalyzer(self.test_graph)
        self.elevation_profiler = ElevationProfiler(self.test_graph)
        
        # Test parameters
        self.start_node = 1
        self.target_distance = 2.0
        
    def _create_test_graph(self):
        """Create test graph with elevation data"""
        graph = nx.Graph()
        
        # Add nodes with coordinates and elevation
        nodes = [
            (1, -80.4094, 37.1299, 100),
            (2, -80.4000, 37.1300, 110),
            (3, -80.4050, 37.1350, 105),
            (4, -80.4100, 37.1250, 120),
            (5, -80.4000, 37.1400, 130),
            (6, -80.4150, 37.1200, 95)
        ]
        
        for node_id, x, y, elevation in nodes:
            graph.add_node(node_id, x=x, y=y, elevation=elevation)
        
        # Add edges with lengths
        edges = [
            (1, 2, 500), (2, 3, 600), (3, 4, 700), (4, 1, 800),
            (1, 3, 900), (2, 4, 650), (3, 5, 550), (4, 6, 750),
            (1, 6, 850), (2, 5, 700)
        ]
        
        for n1, n2, length in edges:
            graph.add_edge(n1, n2, length=length)
        
        return graph
    
    def test_ga_availability_detection(self):
        """Test GA availability detection"""
        # Get solver info
        solver_info = self.route_optimizer.get_solver_info()
        
        # Verify GA information is included
        self.assertIn('ga_available', solver_info)
        self.assertIn('available_algorithms', solver_info)
        
        # Check algorithm list includes auto
        algorithms = solver_info['available_algorithms']
        self.assertIn('auto', algorithms)
        
        if solver_info['ga_available']:
            self.assertIn('genetic', algorithms)
            self.assertIn('ga_optimizer', solver_info)
    
    def test_algorithm_selection_auto_mode(self):
        """Test automatic algorithm selection"""
        # Test with elevation objective (should select GA if available)
        result = self.route_optimizer.optimize_route(
            start_node=self.start_node,
            target_distance_km=self.target_distance,
            objective=self.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION,
            algorithm="auto"
        )
        
        if result:
            solver_info = result.get('solver_info', {})
            self.assertIn('algorithm_used', solver_info)
            
            # Should have selected appropriate algorithm
            algorithm_used = solver_info['algorithm_used']
            self.assertIn(algorithm_used, ['genetic', 'nearest_neighbor'])
    
    def test_ga_optimization_complete_workflow(self):
        """Test complete GA optimization workflow"""
        # Skip if GA not available
        solver_info = self.route_optimizer.get_solver_info()
        if not solver_info.get('ga_available', False):
            self.skipTest("GA not available")
        
        # Test GA optimization with retry for robustness
        result = None
        max_attempts = 3
        
        for attempt in range(max_attempts):
            result = self.route_optimizer.optimize_route(
                start_node=self.start_node,
                target_distance_km=self.target_distance,
                objective=self.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION,
                algorithm="genetic"
            )
            
            if result and result.get('route'):
                route = result['route']
                if len(route) > 1 and route[0] == self.start_node and route[-1] == self.start_node:
                    break  # Got a valid circular route
                    
            print(f"GA attempt {attempt + 1}: Invalid route, retrying...")
        
        # Verify result structure
        self.assertIsNotNone(result, "GA optimization failed after multiple attempts")
        self.assertIn('route', result)
        self.assertIn('stats', result)
        self.assertIn('solver_info', result)
        
        # Verify GA-specific information
        solver_info = result['solver_info']
        self.assertEqual(solver_info['solver_type'], 'genetic')
        self.assertIn('ga_generations', solver_info)
        self.assertIn('ga_convergence', solver_info)
        
        # Verify route validity
        route = result['route']
        self.assertIsInstance(route, list)
        self.assertGreater(len(route), 1)
        self.assertEqual(route[0], self.start_node, f"Route should start at {self.start_node}, got {route[0]}")
        self.assertEqual(route[-1], self.start_node, f"Route should end at {self.start_node}, got {route[-1]}. Full route: {route}")
    
    def test_ga_with_route_analyzer_integration(self):
        """Test GA integration with route analyzer"""
        # Skip if GA not available
        solver_info = self.route_optimizer.get_solver_info()
        if not solver_info.get('ga_available', False):
            self.skipTest("GA not available")
        
        # Generate GA route
        result = self.route_optimizer.optimize_route(
            start_node=self.start_node,
            target_distance_km=self.target_distance,
            objective=self.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION,
            algorithm="genetic"
        )
        
        if result:
            # Analyze route
            analysis = self.route_analyzer.analyze_route(result)
            
            # Verify analysis
            self.assertIsNotNone(analysis)
            self.assertIn('basic_stats', analysis)
            basic_stats = analysis['basic_stats']
            self.assertIn('total_distance_m', basic_stats)
            self.assertIn('total_elevation_gain_m', basic_stats)
    
    def test_ga_with_elevation_profiler_integration(self):
        """Test GA integration with elevation profiler"""
        # Skip if GA not available
        solver_info = self.route_optimizer.get_solver_info()
        if not solver_info.get('ga_available', False):
            self.skipTest("GA not available")
        
        # Generate GA route
        result = self.route_optimizer.optimize_route(
            start_node=self.start_node,
            target_distance_km=self.target_distance,
            objective=self.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION,
            algorithm="genetic"
        )
        
        if result:
            # Generate elevation profile
            profile = self.elevation_profiler.generate_profile_data(result)
            
            # Verify profile
            self.assertIsNotNone(profile)
            self.assertIn('distances_m', profile)
            self.assertIn('elevations', profile)
            self.assertGreater(len(profile['distances_m']), 0)
    
    
    def test_ga_error_handling(self):
        """Test GA error handling"""
        # Test with invalid start node
        result = self.route_optimizer.optimize_route(
            start_node=999,  # Invalid node
            target_distance_km=self.target_distance,
            objective=self.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION,
            algorithm="genetic"
        )
        
        self.assertIsNone(result)
        
        # Test with invalid distance
        result = self.route_optimizer.optimize_route(
            start_node=self.start_node,
            target_distance_km=-1.0,  # Invalid distance
            objective=self.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION,
            algorithm="genetic"
        )
        
        self.assertIsNone(result)
    
    def test_ga_parameter_validation(self):
        """Test GA parameter validation"""
        # Test parameter validation
        validation = self.route_optimizer.validate_parameters(
            start_node=self.start_node,
            target_distance_km=self.target_distance,
            algorithm="genetic"
        )
        
        self.assertIsInstance(validation, dict)
        self.assertIn('valid', validation)
        self.assertIn('errors', validation)
        self.assertIn('warnings', validation)
        
        # Valid parameters should pass
        if self.route_optimizer.get_solver_info().get('ga_available', False):
            self.assertTrue(validation['valid'])
    
    def test_ga_performance_reasonable(self):
        """Test GA performance is reasonable"""
        # Skip if GA not available
        solver_info = self.route_optimizer.get_solver_info()
        if not solver_info.get('ga_available', False):
            self.skipTest("GA not available")
        
        start_time = time.time()
        
        # Generate GA route
        result = self.route_optimizer.optimize_route(
            start_node=self.start_node,
            target_distance_km=self.target_distance,
            objective=self.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION,
            algorithm="genetic"
        )
        
        total_time = time.time() - start_time
        
        if result:
            # GA should complete in reasonable time for small graphs
            self.assertLess(total_time, 60.0)  # Should complete within 60 seconds
            
            # Verify solver timing information
            solver_info = result.get('solver_info', {})
            if 'solve_time' in solver_info:
                solve_time = solver_info['solve_time']
                self.assertGreater(solve_time, 0)
                self.assertLess(solve_time, 60.0)


class TestGAFallbackBehavior(unittest.TestCase):
    """Test GA fallback behavior when not available"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test graph
        self.test_graph = self._create_minimal_graph()
        
    def _create_minimal_graph(self):
        """Create minimal test graph"""
        graph = nx.Graph()
        
        # Add minimal nodes
        for i in range(1, 5):
            graph.add_node(i, x=-80.4 + i*0.01, y=37.1 + i*0.01, elevation=100 + i*10)
        
        # Add edges
        edges = [(1, 2, 500), (2, 3, 600), (3, 4, 700), (4, 1, 800)]
        for n1, n2, length in edges:
            graph.add_edge(n1, n2, length=length)
        
        return graph
    
    @patch('route_services.route_optimizer.GA_AVAILABLE', False)
    def test_ga_fallback_when_unavailable(self):
        """Test fallback behavior when GA is not available"""
        route_optimizer = RouteOptimizer(self.test_graph)
        
        # Request GA algorithm when not available
        result = route_optimizer.optimize_route(
            start_node=1,
            target_distance_km=2.0,
            objective=route_optimizer.RouteObjective.MAXIMIZE_ELEVATION,
            algorithm="genetic"
        )
        
        if result:
            # Should fall back to TSP
            solver_info = result['solver_info']
            self.assertNotEqual(solver_info['solver_type'], 'genetic')
    
    @patch('route_services.route_optimizer.GA_AVAILABLE', False)
    def test_auto_selection_without_ga(self):
        """Test auto selection when GA is not available"""
        route_optimizer = RouteOptimizer(self.test_graph)
        
        # Use auto selection
        result = route_optimizer.optimize_route(
            start_node=1,
            target_distance_km=2.0,
            objective=route_optimizer.RouteObjective.MAXIMIZE_ELEVATION,
            algorithm="auto"
        )
        
        if result:
            # Should use TSP solver
            solver_info = result['solver_info']
            self.assertIn(solver_info['algorithm_used'], ['nearest_neighbor'])


if __name__ == '__main__':
    unittest.main()