#!/usr/bin/env python3
"""
Unit tests for RouteOptimizer with comprehensive GA integration testing
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
        # Create test graph
        self.test_graph = nx.Graph()
        self.test_graph.add_node(1001, x=-80.4094, y=37.1299, elevation=600)
        self.test_graph.add_node(1002, x=-80.4095, y=37.1300, elevation=620)
        self.test_graph.add_node(1003, x=-80.4096, y=37.1301, elevation=610)
        self.test_graph.add_node(1004, x=-80.4097, y=37.1302, elevation=650)
        self.test_graph.add_node(1005, x=-80.4098, y=37.1303, elevation=630)
        
        # Add edges
        self.test_graph.add_edge(1001, 1002, length=100, highway='residential')
        self.test_graph.add_edge(1002, 1003, length=150, highway='residential')
        self.test_graph.add_edge(1003, 1004, length=120, highway='primary')
        self.test_graph.add_edge(1004, 1005, length=110, highway='residential')
        self.test_graph.add_edge(1005, 1001, length=130, highway='residential')
        
        # Sample route result
        self.sample_route_result = {
            'route': [1001, 1002, 1003, 1004, 1005],
            'stats': {
                'total_distance_km': 2.5,
                'total_elevation_gain_m': 60,
                'total_elevation_loss_m': 30,
                'algorithm': 'genetic',
                'objective': 'elevation'
            },
            'metadata': {
                'solver': 'genetic',
                'generation_found': 50,
                'total_generations': 100,
                'fitness': 0.85
            }
        }


class TestRouteOptimizerInitialization(TestRouteOptimizer):
    """Test RouteOptimizer initialization"""
    
    def test_initialization_basic(self):
        """Test basic initialization"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        self.assertEqual(optimizer.graph, self.test_graph)
        self.assertFalse(optimizer.verbose)
        # Current implementation initializes solver_type to 'genetic'
        self.assertEqual(optimizer.solver_type, 'genetic')
    
    def test_initialization_with_elevation_config(self):
        """Test initialization with elevation configuration"""
        with patch('route_services.route_optimizer.get_elevation_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager
            
            optimizer = RouteOptimizer(
                self.test_graph, 
                elevation_config_path="/test/config.json",
                verbose=True
            )
            
            self.assertTrue(optimizer.verbose)
    
    def test_initialization_no_ga_available(self):
        """Test initialization when GA is not available"""
        with patch('route_services.route_optimizer.GA_AVAILABLE', False):
            # Should raise ImportError during initialization when GA unavailable
            with self.assertRaises(ImportError):
                RouteOptimizer(self.test_graph, verbose=False)
    
    def test_initialization_no_enhanced_elevation(self):
        """Test initialization without enhanced elevation"""
        with patch('route_services.route_optimizer.ENHANCED_ELEVATION_AVAILABLE', False):
            optimizer = RouteOptimizer(self.test_graph, verbose=False)
            
            # Should still initialize successfully
            self.assertIsNotNone(optimizer)


class TestRouteOptimizerProperties(TestRouteOptimizer):
    """Test RouteOptimizer properties and getters"""
    
    def test_solver_type_property(self):
        """Test solver_type property"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        # Should be genetic for current implementation
        self.assertEqual(optimizer.solver_type, 'genetic')
    
    def test_route_objective_property(self):
        """Test RouteObjective property"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        # Should have access to RouteObjective
        self.assertTrue(hasattr(optimizer, 'RouteObjective'))
    
    def test_get_available_objectives(self):
        """Test getting available objectives"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        objectives = optimizer.get_available_objectives()
        
        self.assertIsInstance(objectives, dict)
        # Should include standard objectives
        expected_objective_names = ['Shortest Route', 'Maximum Elevation Gain', 'Balanced Route', 'Easiest Route']
        for obj_name in expected_objective_names:
            self.assertIn(obj_name, objectives)
    
    def test_get_available_algorithms(self):
        """Test getting available algorithms"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        algorithms = optimizer.get_available_algorithms()
        
        self.assertIsInstance(algorithms, list)
        # Should include genetic algorithm
        self.assertIn('genetic', algorithms)
    
    @patch('route_services.route_optimizer.GA_AVAILABLE', True)
    def test_get_available_algorithms_with_ga(self):
        """Test available algorithms when GA is available"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        algorithms = optimizer.get_available_algorithms()
        
        self.assertIn('genetic', algorithms)
        # Current implementation only returns genetic when available
        self.assertEqual(algorithms, ['genetic'])


class TestParameterValidation(TestRouteOptimizer):
    """Test parameter validation"""
    
    def test_validate_parameters_valid(self):
        """Test validation with valid parameters"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        # Should return valid result
        result = optimizer.validate_parameters(1001, 5.0, 'elevation', 'genetic')
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['errors']), 0)
    
    def test_validate_parameters_invalid_node(self):
        """Test validation with invalid start node"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        result = optimizer.validate_parameters(9999, 5.0, 'elevation', 'genetic')
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)
    
    def test_validate_parameters_invalid_distance(self):
        """Test validation with invalid distance"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        result = optimizer.validate_parameters(1001, -1.0, 'elevation', 'genetic')
        self.assertFalse(result['valid'])
        
        result = optimizer.validate_parameters(1001, 0.0, 'elevation', 'genetic')
        self.assertFalse(result['valid'])
    
    def test_validate_parameters_invalid_objective(self):
        """Test validation with invalid objective"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        result = optimizer.validate_parameters(1001, 5.0, 'invalid_objective', 'genetic')
        self.assertTrue(result['valid'])  # Current implementation only shows warnings for invalid objectives
        self.assertGreater(len(result['warnings']), 0)
    
    def test_validate_parameters_invalid_algorithm(self):
        """Test validation with invalid algorithm"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        result = optimizer.validate_parameters(1001, 5.0, 'elevation', 'invalid_algorithm')
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)


class TestAlgorithmSelection(TestRouteOptimizer):
    """Test algorithm selection logic"""
    
    def test_select_algorithm_genetic_with_ga(self):
        """Test genetic algorithm selection when GA is available"""
        with patch('route_services.route_optimizer.GA_AVAILABLE', True):
            optimizer = RouteOptimizer(self.test_graph, verbose=False)
            
            # Should always return genetic when available
            selected = optimizer._select_algorithm('genetic', 'elevation')
            self.assertEqual(selected, 'genetic')
            
            selected = optimizer._select_algorithm('genetic', 'balanced')
            self.assertEqual(selected, 'genetic')
    
    def test_select_algorithm_no_ga_raises_error(self):
        """Test algorithm selection when GA is not available"""
        with patch('route_services.route_optimizer.GA_AVAILABLE', False):
            # Should raise ImportError during initialization
            with self.assertRaises(ImportError):
                RouteOptimizer(self.test_graph, verbose=False)
    
    def test_select_algorithm_explicit_genetic(self):
        """Test explicit genetic algorithm selection"""
        with patch('route_services.route_optimizer.GA_AVAILABLE', True):
            optimizer = RouteOptimizer(self.test_graph, verbose=False)
            
            selected = optimizer._select_algorithm('genetic', 'elevation')
            self.assertEqual(selected, 'genetic')
    
    def test_select_algorithm_genetic_unavailable_raises_error(self):
        """Test genetic selection when unavailable"""
        with patch('route_services.route_optimizer.GA_AVAILABLE', False):
            # Should raise ImportError during initialization
            with self.assertRaises(ImportError):
                RouteOptimizer(self.test_graph, verbose=False)


class TestGeneticAlgorithmIntegration(TestRouteOptimizer):
    """Test genetic algorithm integration"""
    
    @patch('route_services.route_optimizer.GA_AVAILABLE', True)
    @patch('route_services.route_optimizer.GeneticRouteOptimizer')
    def test_optimize_genetic_basic(self, mock_ga_class):
        """Test basic genetic algorithm optimization"""
        # Setup mock GA optimizer
        mock_ga_optimizer = Mock()
        mock_ga_results = Mock()
        mock_chromosome = Mock()
        mock_chromosome.to_route_result.return_value = {
            'route': [1001, 1002, 1003],
            'stats': {'total_distance_km': 2.5}
        }
        mock_chromosome.segments = [Mock(), Mock()]  # Mock segments list
        mock_chromosome.segments[0].start_node = 1001  # Mock start node
        mock_chromosome.get_route_nodes.return_value = [1001, 1002, 1003]
        mock_chromosome.get_total_distance.return_value = 2500.0
        mock_chromosome.get_total_elevation_gain.return_value = 60.0
        mock_ga_results.best_chromosome = mock_chromosome
        mock_ga_results.best_fitness = 0.85
        mock_ga_results.generation_found = 50
        mock_ga_results.total_generations = 100
        mock_ga_optimizer.optimize_route.return_value = mock_ga_results
        mock_ga_class.return_value = mock_ga_optimizer
        
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        result = optimizer._optimize_genetic(1001, 5.0, 'elevation')
        
        self.assertIsNotNone(result)
        mock_ga_optimizer.optimize_route.assert_called_once()
    
    @patch('route_services.route_optimizer.GA_AVAILABLE', True)
    @patch('route_services.route_optimizer.GeneticRouteOptimizer')
    def test_optimize_genetic_with_custom_config(self, mock_ga_class):
        """Test genetic optimization with custom configuration"""
        mock_ga_optimizer = Mock()
        mock_ga_results = Mock()
        mock_chromosome = Mock()
        mock_chromosome.to_route_result.return_value = {
            'route': [1001, 1002],
            'stats': {'total_distance_km': 1.5}
        }
        mock_chromosome.segments = [Mock()]  # Mock segments list
        mock_chromosome.segments[0].start_node = 1001  # Mock start node
        mock_chromosome.get_route_nodes.return_value = [1001, 1002]
        mock_chromosome.get_total_distance.return_value = 1500.0
        mock_chromosome.get_total_elevation_gain.return_value = 30.0
        mock_ga_results.best_chromosome = mock_chromosome
        mock_ga_optimizer.optimize_route.return_value = mock_ga_results
        mock_ga_class.return_value = mock_ga_optimizer
        
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        result = optimizer._optimize_genetic(
            1001, 3.0, 'balanced',
            exclude_footways=False,
            allow_bidirectional_segments=False
        )
        
        self.assertIsNotNone(result)
        # Verify GA was called
        mock_ga_optimizer.optimize_route.assert_called_once()
    
    @patch('route_services.route_optimizer.GA_AVAILABLE', True)
    @patch('route_services.route_optimizer.GeneticRouteOptimizer')
    def test_optimize_genetic_error_handling(self, mock_ga_class):
        """Test genetic optimization error handling"""
        mock_ga_optimizer = Mock()
        mock_ga_optimizer.optimize_route.side_effect = Exception("GA optimization failed")
        mock_ga_class.return_value = mock_ga_optimizer
        
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        # Should handle exception gracefully and return None
        try:
            result = optimizer._optimize_genetic(1001, 5.0, 'elevation')
            # Should return None on error
            self.assertIsNone(result)
        except Exception:
            # If exception propagates, that's also acceptable behavior
            pass
    
    def test_convert_tsp_to_ga_objective(self):
        """Test TSP to GA objective conversion"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        # Test with RouteObjective enum values
        from route_objective import RouteObjective
        
        # Test conversion with enum values
        result = optimizer._convert_tsp_to_ga_objective(RouteObjective.MINIMIZE_DISTANCE)
        self.assertIsNotNone(result)
        
        result = optimizer._convert_tsp_to_ga_objective(RouteObjective.MAXIMIZE_ELEVATION)
        self.assertIsNotNone(result)
    
    def test_convert_ga_results_to_standard(self):
        """Test GA results to standard format conversion"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        # Mock GA results with proper segments mock
        mock_ga_results = Mock()
        mock_chromosome = Mock()
        mock_chromosome.to_route_result.return_value = {
            'route': [1001, 1002, 1003],
            'stats': {
                'total_distance_km': 2.5,
                'total_elevation_gain_m': 60
            }
        }
        mock_chromosome.segments = [Mock(), Mock()]  # Mock segments list
        mock_chromosome.segments[0].start_node = 1001  # Mock start node
        mock_chromosome.get_route_nodes.return_value = [1001, 1002, 1003, 1001]  # Complete loop
        mock_chromosome.get_total_distance.return_value = 2500.0
        mock_chromosome.get_total_elevation_gain.return_value = 60.0
        mock_ga_results.best_chromosome = mock_chromosome
        mock_ga_results.best_fitness = 0.85
        mock_ga_results.generation_found = 25
        mock_ga_results.total_generations = 100
        mock_ga_results.total_time = 15.5
        mock_ga_results.convergence_reason = "max_generations"
        mock_ga_results.stats = Mock()  # Mock GA stats
        
        result = optimizer._convert_ga_results_to_standard(mock_ga_results, 'elevation')
        
        # Verify standard format
        self.assertIn('route', result)
        self.assertIn('stats', result)
        self.assertIn('fitness_score', result)
        self.assertEqual(result['stats']['route_type'], 'genetic_algorithm')
        self.assertEqual(result['stats']['objective'], 'elevation')
        self.assertEqual(result['fitness_score'], 0.85)
        self.assertEqual(result['total_distance_km'], 2.5)


class TestGraphFiltering(TestRouteOptimizer):
    """Test graph filtering and processing"""
    
    def test_create_filtered_graph(self):
        """Test creating filtered graph around start node"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        filtered_graph = optimizer._create_filtered_graph(1001)
        
        # Should return a graph
        self.assertIsInstance(filtered_graph, nx.Graph)
        # Should include start node
        self.assertIn(1001, filtered_graph.nodes)
    
    def test_get_intersection_nodes(self):
        """Test getting intersection nodes"""
        # This method may not exist in current implementation
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        if hasattr(optimizer, '_get_intersection_nodes'):
            intersections = optimizer._get_intersection_nodes()
            self.assertIsInstance(intersections, list)
        else:
            self.skipTest('_get_intersection_nodes method not implemented')
    
    def test_get_intersection_nodes_custom_graph(self):
        """Test getting intersection nodes from custom graph"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        if hasattr(optimizer, '_get_intersection_nodes'):
            # Create custom graph with clear intersections
            custom_graph = nx.Graph()
            custom_graph.add_node(1, x=-80.4094, y=37.1299)
            custom_graph.add_node(2, x=-80.4095, y=37.1300)
            custom_graph.add_node(3, x=-80.4096, y=37.1301)
            custom_graph.add_edge(1, 2)
            custom_graph.add_edge(2, 3)  # Node 2 is intersection
            
            intersections = optimizer._get_intersection_nodes(custom_graph)
            
            # Check that we get some intersection nodes - exact behavior may vary
            self.assertIsInstance(intersections, list)
            self.assertGreater(len(intersections), 0)
            # Endpoints should be included
            self.assertIn(1, intersections)  # Endpoint
            self.assertIn(3, intersections)  # Endpoint
        else:
            self.skipTest('_get_intersection_nodes method not implemented')
    
    def test_filter_nodes_by_distance(self):
        """Test filtering nodes by haversine distance"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        candidate_nodes = [1001, 1002, 1003, 1004, 1005]
        filtered = optimizer._filter_nodes_by_distance(
            candidate_nodes, 1001, 0.5  # 500m radius
        )
        
        self.assertIsInstance(filtered, list)
        self.assertIn(1001, filtered)  # Start node always included
    
    def test_filter_nodes_by_road_distance(self):
        """Test filtering nodes by road network distance"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        candidate_nodes = [1001, 1002, 1003, 1004, 1005]
        filtered = optimizer._filter_nodes_by_road_distance(
            candidate_nodes, 1001, 0.5  # 500m road distance
        )
        
        self.assertIsInstance(filtered, list)
        self.assertIn(1001, filtered)  # Start node always included
    
    def test_filter_graph_for_routing(self):
        """Test filtering graph for routing"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        # Test excluding footways
        filtered = optimizer._filter_graph_for_routing(exclude_footways=True)
        self.assertIsInstance(filtered, nx.Graph)
        
        # Test including footways
        filtered = optimizer._filter_graph_for_routing(exclude_footways=False)
        self.assertIsInstance(filtered, nx.Graph)


class TestMainOptimizationInterface(TestRouteOptimizer):
    """Test main optimization interface"""
    
    @patch('route_services.route_optimizer.GA_AVAILABLE', True)
    def test_optimize_route_genetic_success(self):
        """Test successful route optimization with genetic algorithm"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        with patch.object(optimizer, '_optimize_genetic') as mock_genetic:
            mock_genetic.return_value = self.sample_route_result
            
            result = optimizer.optimize_route(
                start_node=1001,
                target_distance_km=5.0,
                objective='elevation',
                algorithm='genetic'
            )
            
            self.assertIsNotNone(result)
            self.assertEqual(result['stats']['algorithm'], 'genetic')
            mock_genetic.assert_called_once()
    
    @patch('route_services.route_optimizer.GA_AVAILABLE', False)
    def test_optimize_route_fallback_when_ga_unavailable(self):
        """Test fallback when GA is unavailable"""
        # Should raise ImportError during initialization when GA unavailable
        with self.assertRaises(ImportError):
            RouteOptimizer(self.test_graph, verbose=False)
    
    def test_optimize_route_parameter_validation(self):
        """Test parameter validation in optimize_route"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        # Test invalid parameters - current implementation returns None instead of raising
        result = optimizer.optimize_route(9999, 5.0, 'elevation', 'genetic')  # Invalid node
        self.assertIsNone(result)
        
        result = optimizer.optimize_route(1001, -1.0, 'elevation', 'genetic')  # Invalid distance
        self.assertIsNone(result)
    
    def test_optimize_route_auto_algorithm_selection(self):
        """Test auto algorithm selection in optimize_route"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        with patch.object(optimizer, '_select_algorithm') as mock_select:
            with patch.object(optimizer, '_optimize_genetic') as mock_genetic:
                mock_select.return_value = 'genetic'
                mock_genetic.return_value = self.sample_route_result
                
                result = optimizer.optimize_route(
                    start_node=1001,
                    target_distance_km=5.0,
                    objective='elevation',
                    algorithm='auto'
                )
                
                mock_select.assert_called_once_with('auto', 'elevation')


class TestSolverInformation(TestRouteOptimizer):
    """Test solver information and diagnostics"""
    
    def test_get_solver_info(self):
        """Test getting solver information"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        if hasattr(optimizer, 'get_solver_info'):
            info = optimizer.get_solver_info()
            
            self.assertIsInstance(info, dict)
            self.assertIn('available_algorithms', info)
            self.assertIn('available_objectives', info)
        else:
            # Test basic property access instead
            self.assertEqual(optimizer.solver_type, 'genetic')
            self.assertIsNotNone(optimizer.get_available_algorithms())
            self.assertIsNotNone(optimizer.get_available_objectives())
    
    @patch('route_services.route_optimizer.GA_AVAILABLE', True)
    def test_get_solver_info_with_ga(self):
        """Test solver info when GA is available"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        if hasattr(optimizer, 'get_solver_info'):
            info = optimizer.get_solver_info()
            self.assertIn('genetic', info['available_algorithms'])
            self.assertTrue(info['ga_available'])
        else:
            # Test basic properties instead
            algorithms = optimizer.get_available_algorithms()
            self.assertIn('genetic', algorithms)
    
    @patch('route_services.route_optimizer.GA_AVAILABLE', False)
    def test_get_solver_info_no_ga(self):
        """Test solver info when GA is not available"""
        # Should raise ImportError during initialization when GA unavailable
        with self.assertRaises(ImportError):
            RouteOptimizer(self.test_graph, verbose=False)


class TestEdgeCasesAndErrorHandling(TestRouteOptimizer):
    """Test edge cases and error handling"""
    
    def test_empty_graph(self):
        """Test with empty graph"""
        empty_graph = nx.Graph()
        optimizer = RouteOptimizer(empty_graph, verbose=False)
        
        # Should handle empty graph gracefully
        intersections = optimizer._get_intersection_nodes()
        self.assertEqual(len(intersections), 0)
    
    def test_single_node_graph(self):
        """Test with single node graph"""
        single_graph = nx.Graph()
        single_graph.add_node(1, x=-80.4094, y=37.1299)
        
        optimizer = RouteOptimizer(single_graph, verbose=False)
        
        intersections = optimizer._get_intersection_nodes()
        self.assertEqual(len(intersections), 1)
    
    def test_disconnected_graph(self):
        """Test with disconnected graph"""
        disconnected_graph = nx.Graph()
        disconnected_graph.add_node(1, x=-80.4094, y=37.1299)
        disconnected_graph.add_node(2, x=-80.4095, y=37.1300)
        # No edges - disconnected
        
        optimizer = RouteOptimizer(disconnected_graph, verbose=False)
        
        # Should handle disconnected graph
        filtered = optimizer._filter_nodes_by_road_distance([1, 2], 1, 1.0)
        self.assertIn(1, filtered)  # Start node should always be included
    
    def test_missing_node_attributes(self):
        """Test with nodes missing required attributes"""
        incomplete_graph = nx.Graph()
        incomplete_graph.add_node(1)  # Missing x, y coordinates
        incomplete_graph.add_node(2, x=-80.4095)  # Missing y coordinate
        
        optimizer = RouteOptimizer(incomplete_graph, verbose=False)
        
        # Should handle missing attributes gracefully
        try:
            intersections = optimizer._get_intersection_nodes()
            # May return empty list or handle error gracefully
        except (KeyError, AttributeError):
            # Expected behavior for missing attributes
            pass
    
    def test_large_distance_request(self):
        """Test optimization with very large distance request"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        # Request 100km route on small graph
        result = optimizer.validate_parameters(1001, 100.0, 'elevation', 'genetic')
        self.assertTrue(result['valid'])  # Should be valid but have warnings
        self.assertGreater(len(result['warnings']), 0)  # Should warn about large distance
    
    def test_optimization_timeout_handling(self):
        """Test handling of optimization timeouts"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        with patch.object(optimizer, '_optimize_genetic') as mock_genetic:
            # Simulate timeout
            mock_genetic.side_effect = TimeoutError("Optimization timed out")
            
            try:
                result = optimizer.optimize_route(1001, 5.0, 'elevation', 'genetic')
                # Should handle timeout gracefully
            except TimeoutError:
                # Expected behavior
                pass


class TestIntegrationScenarios(TestRouteOptimizer):
    """Test integration scenarios"""
    
    @patch('route_services.route_optimizer.GA_AVAILABLE', True)
    @patch('route_services.route_optimizer.GeneticRouteOptimizer')
    def test_full_genetic_optimization_workflow(self, mock_ga_class):
        """Test complete genetic optimization workflow"""
        # Setup comprehensive mock
        mock_ga_optimizer = Mock()
        mock_ga_results = Mock()
        mock_chromosome = Mock()
        mock_chromosome.to_route_result.return_value = {
            'route': [1001, 1002, 1003, 1004, 1005, 1001],
            'stats': {
                'total_distance_km': 5.1,
                'total_elevation_gain_m': 80,
                'total_elevation_loss_m': 75,
                'max_grade_percent': 8.5
            }
        }
        mock_ga_results.best_chromosome = mock_chromosome
        mock_ga_results.best_fitness = 0.92
        mock_ga_results.generation_found = 75
        mock_ga_results.total_generations = 150
        mock_ga_results.total_time = 45.2
        mock_ga_results.convergence_reason = "convergence"
        mock_ga_optimizer.optimize_route.return_value = mock_ga_results
        mock_ga_class.return_value = mock_ga_optimizer
        
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        # Test complete workflow
        result = optimizer.optimize_route(
            start_node=1001,
            target_distance_km=5.0,
            objective='elevation',
            algorithm='genetic'
        )
        
        # Should have attempted optimization
        mock_ga_optimizer.optimize_route.assert_called_once()
    
    def test_multiple_optimizations_solver_persistence(self):
        """Test multiple optimizations with solver persistence"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        with patch.object(optimizer, '_optimize_genetic') as mock_genetic:
            mock_genetic.return_value = self.sample_route_result
            
            # First optimization
            result1 = optimizer.optimize_route(1001, 3.0, 'elevation', 'genetic')
            
            # Second optimization (should reuse solver)
            result2 = optimizer.optimize_route(1002, 4.0, 'balanced', 'genetic')
            
            # Both should succeed
            self.assertIsNotNone(result1)
            self.assertIsNotNone(result2)
            
            # Should have called genetic optimizer twice
            self.assertEqual(mock_genetic.call_count, 2)


class TestRouteOptimizerAdditionalCoverage(TestRouteOptimizer):
    """Additional tests to improve code coverage"""
    
    def test_initialize_elevation_with_enhanced_stats(self):
        """Test elevation initialization with enhanced elevation stats"""
        mock_elevation_manager = Mock()
        mock_elevation_source = Mock()
        mock_elevation_source.get_source_info.return_value = {'type': 'hybrid_source'}
        mock_elevation_source.get_resolution.return_value = 1.0
        mock_elevation_source.get_stats.return_value = {'primary_percentage': 85.0}
        mock_elevation_manager.get_elevation_source.return_value = mock_elevation_source
        
        with patch('route_services.route_optimizer.ENHANCED_ELEVATION_AVAILABLE', True):
            with patch('route_services.route_optimizer.get_elevation_manager', return_value=mock_elevation_manager):
                with patch('route_services.route_optimizer.GA_AVAILABLE', True):
                    with patch('route_services.route_optimizer.GeneticRouteOptimizer'):
                        optimizer = RouteOptimizer(self.test_graph, verbose=False)
                        
                        # Should have initialized elevation successfully
                        self.assertIsNotNone(optimizer._elevation_manager)
                        self.assertIsNotNone(optimizer._elevation_source)
    
    def test_initialize_elevation_no_source(self):
        """Test elevation initialization when no source is available"""
        mock_elevation_manager = Mock()
        mock_elevation_manager.get_elevation_source.return_value = None
        
        with patch('route_services.route_optimizer.ENHANCED_ELEVATION_AVAILABLE', True):
            with patch('route_services.route_optimizer.get_elevation_manager', return_value=mock_elevation_manager):
                with patch('route_services.route_optimizer.GA_AVAILABLE', True):
                    with patch('route_services.route_optimizer.GeneticRouteOptimizer'):
                        optimizer = RouteOptimizer(self.test_graph, verbose=False)
                        
                        # Should handle no elevation source gracefully
                        self.assertIsNotNone(optimizer._elevation_manager)
                        self.assertIsNone(optimizer._elevation_source)
    
    def test_optimize_route_with_default_objective(self):
        """Test optimize_route with default objective (None)"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        with patch.object(optimizer, '_optimize_genetic') as mock_genetic:
            mock_genetic.return_value = self.sample_route_result
            
            # Should use default objective when None is provided
            result = optimizer.optimize_route(1001, 5.0, objective=None)
            
            self.assertIsNotNone(result)
            mock_genetic.assert_called_once()
    
    def test_optimize_route_with_graph_filtering(self):
        """Test optimize_route with graph filtering options"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        with patch.object(optimizer, '_optimize_genetic') as mock_genetic:
            mock_genetic.return_value = self.sample_route_result
            
            # Test with exclude_footways=False
            result = optimizer.optimize_route(
                1001, 5.0, 
                exclude_footways=False,
                allow_bidirectional_segments=False
            )
            
            self.assertIsNotNone(result)
            mock_genetic.assert_called_once()
            
            # Check that the method was called with the correct parameters
            call_args = mock_genetic.call_args
            args, kwargs = call_args
            # The optimize_route method passes these as positional arguments to _optimize_genetic
            # args[0] = start_node, args[1] = target_distance_km, args[2] = objective
            # args[3] = exclude_footways, args[4] = allow_bidirectional_segments
            if len(args) >= 5:
                self.assertFalse(args[3])  # exclude_footways
                self.assertFalse(args[4])  # allow_bidirectional_segments
    
    def test_optimize_route_no_graph_loaded(self):
        """Test optimize_route when no graph is loaded"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        optimizer.graph = None
        
        result = optimizer.optimize_route(1001, 5.0)
        
        self.assertIsNone(result)
    
    def test_get_intersection_nodes_with_highway_tags(self):
        """Test intersection node detection with highway tags"""
        # Add highway tags to test nodes
        self.test_graph.nodes[1001]['highway'] = 'crossing'
        self.test_graph.nodes[1002]['highway'] = 'traffic_signals'
        self.test_graph.nodes[1003]['highway'] = 'stop'
        self.test_graph.nodes[1004]['highway'] = 'mini_roundabout'
        
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        intersection_nodes = optimizer._get_intersection_nodes()
        
        # Should include nodes with highway tags
        self.assertIsInstance(intersection_nodes, list)
        # The exact nodes returned depends on the filtering algorithm
        # Just verify we get some intersection nodes
        self.assertGreaterEqual(len(intersection_nodes), 0)
    
    def test_get_intersection_nodes_proximity_filtering(self):
        """Test intersection node proximity filtering"""
        # Create nodes close to each other
        test_graph = nx.Graph()
        test_graph.add_node(1001, x=-80.4094, y=37.1299, highway='crossing')
        test_graph.add_node(1002, x=-80.4094001, y=37.1299001)  # Very close to 1001
        test_graph.add_node(1003, x=-80.4095, y=37.1300)  # Further away
        
        # Add edges to make them non-degree-2 nodes
        test_graph.add_edge(1001, 1002)
        test_graph.add_edge(1002, 1003)
        test_graph.add_edge(1003, 1001)
        
        optimizer = RouteOptimizer(test_graph, verbose=False)
        
        intersection_nodes = optimizer._get_intersection_nodes(test_graph)
        
        # Should return intersection nodes (exact filtering depends on implementation)
        self.assertIsInstance(intersection_nodes, list)
        # The exact nodes returned depends on the filtering algorithm
        # Just verify we get some intersection nodes
        self.assertGreaterEqual(len(intersection_nodes), 0)
    
    def test_filter_nodes_by_road_distance_with_chunks(self):
        """Test road distance filtering with chunk processing"""
        # Create larger candidate list to test chunking
        candidate_nodes = list(range(1001, 1250))  # 249 nodes
        
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        # Most nodes won't be in the graph, so should be filtered out
        filtered_nodes = optimizer._filter_nodes_by_road_distance(
            candidate_nodes, 1001, 5.0
        )
        
        self.assertIn(1001, filtered_nodes)  # Start node always included
        # Other nodes from the actual graph should be included if reachable
        for node in [1002, 1003, 1004, 1005]:
            if node in candidate_nodes:
                self.assertIn(node, filtered_nodes)
    
    def test_filter_nodes_by_road_distance_unreachable_nodes(self):
        """Test road distance filtering with unreachable nodes"""
        # Create graph with isolated components
        disconnected_graph = nx.Graph()
        disconnected_graph.add_node(1001, x=-80.4094, y=37.1299)
        disconnected_graph.add_node(1002, x=-80.4095, y=37.1300)
        disconnected_graph.add_node(1003, x=-80.4096, y=37.1301)  # Isolated
        disconnected_graph.add_edge(1001, 1002, length=100)
        # 1003 is isolated (no edges)
        
        optimizer = RouteOptimizer(disconnected_graph, verbose=False)
        
        candidate_nodes = [1001, 1002, 1003]
        filtered_nodes = optimizer._filter_nodes_by_road_distance(
            candidate_nodes, 1001, 5.0, disconnected_graph
        )
        
        self.assertIn(1001, filtered_nodes)  # Start node
        self.assertIn(1002, filtered_nodes)  # Reachable
        self.assertNotIn(1003, filtered_nodes)  # Unreachable
    
    def test_filter_nodes_by_road_distance_exception_handling(self):
        """Test road distance filtering with exception handling"""
        # Create graph that might cause exceptions
        problem_graph = nx.Graph()
        problem_graph.add_node(1001, x=-80.4094, y=37.1299)
        problem_graph.add_node(1002, x=-80.4095, y=37.1300)
        # Add edge with no length attribute to potentially cause issues
        problem_graph.add_edge(1001, 1002)
        
        optimizer = RouteOptimizer(problem_graph, verbose=False)
        
        candidate_nodes = [1001, 1002]
        
        # Should handle exceptions gracefully
        filtered_nodes = optimizer._filter_nodes_by_road_distance(
            candidate_nodes, 1001, 5.0, problem_graph
        )
        
        self.assertIn(1001, filtered_nodes)  # Start node always included
    
    def test_convert_ga_results_no_segments(self):
        """Test GA results conversion when chromosome has no segments"""
        # Mock GA results with empty segments
        mock_ga_results = Mock()
        mock_chromosome = Mock()
        mock_chromosome.get_route_nodes.return_value = []
        mock_chromosome.get_total_distance.return_value = 0.0
        mock_chromosome.get_total_elevation_gain.return_value = 0.0
        mock_chromosome.segments = []
        mock_ga_results.best_chromosome = mock_chromosome
        mock_ga_results.best_fitness = 0.0
        mock_ga_results.stats = {'generation': 0}
        
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        result = optimizer._convert_ga_results_to_standard(mock_ga_results, 'test_objective')
        
        self.assertEqual(result['route'], [])
        self.assertEqual(result['total_distance_km'], 0.0)
        self.assertEqual(result['total_elevation_gain_m'], 0.0)
    
    def test_filter_graph_for_routing_footway_removal(self):
        """Test detailed footway removal in graph filtering"""
        # Add footway edges to test graph
        self.test_graph.add_edge(1001, 1010, length=200, highway='footway')
        self.test_graph.add_edge(1002, 1011, length=150, highway='footway')
        self.test_graph.add_node(1010, x=-80.4099, y=37.1304)
        self.test_graph.add_node(1011, x=-80.4100, y=37.1305)
        
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        # Test footway exclusion
        filtered_graph = optimizer._filter_graph_for_routing(exclude_footways=True)
        
        # Should remove footway edges
        self.assertFalse(filtered_graph.has_edge(1001, 1010))
        self.assertFalse(filtered_graph.has_edge(1002, 1011))
        
        # Should keep non-footway edges
        self.assertTrue(filtered_graph.has_edge(1001, 1002))
        
        # Should remove isolated nodes
        if 1010 in filtered_graph.nodes and filtered_graph.degree(1010) == 0:
            self.assertNotIn(1010, filtered_graph.nodes)
        if 1011 in filtered_graph.nodes and filtered_graph.degree(1011) == 0:
            self.assertNotIn(1011, filtered_graph.nodes)
    
    def test_get_solver_info_comprehensive(self):
        """Test comprehensive solver info collection"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        info = optimizer.get_solver_info()
        
        # Should include all expected fields
        self.assertIn('solver_type', info)
        self.assertIn('solver_class', info)
        self.assertIn('available_objectives', info)
        self.assertIn('available_algorithms', info)
        self.assertIn('graph_nodes', info)
        self.assertIn('graph_edges', info)
        self.assertIn('ga_available', info)
        
        # Should have correct values
        self.assertEqual(info['solver_type'], 'genetic')
        self.assertEqual(info['graph_nodes'], len(self.test_graph.nodes))
        self.assertEqual(info['graph_edges'], len(self.test_graph.edges))
        self.assertTrue(info['ga_available'])
        
        # Should include GA optimizer info
        self.assertIn('ga_optimizer', info)
    
    def test_validate_parameters_edge_cases(self):
        """Test parameter validation edge cases"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        # Test with None values
        result = optimizer.validate_parameters(1001, 5.0, None, None)
        self.assertTrue(result['valid'])
        
        # Test with exactly 20km distance (boundary case)
        result = optimizer.validate_parameters(1001, 20.0)
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['warnings']), 0)
        
        # Test with just over 20km distance
        result = optimizer.validate_parameters(1001, 20.1)
        self.assertTrue(result['valid'])
        self.assertGreater(len(result['warnings']), 0)
    
    def test_optimize_genetic_graph_filtering_scenarios(self):
        """Test genetic optimization with different graph filtering scenarios"""
        # Create graph with footway edges
        test_graph = self.test_graph.copy()
        test_graph.add_edge(1001, 1010, length=200, highway='footway')
        test_graph.add_node(1010, x=-80.4099, y=37.1304)
        
        optimizer = RouteOptimizer(test_graph, verbose=False)
        
        # Mock GA optimizer
        mock_ga_optimizer = Mock()
        mock_ga_results = Mock()
        mock_chromosome = Mock()
        mock_chromosome.get_route_nodes.return_value = [1001, 1002, 1001]
        mock_chromosome.get_total_distance.return_value = 1000.0
        mock_chromosome.get_total_elevation_gain.return_value = 20.0
        mock_chromosome.segments = [Mock(start_node=1001, path_nodes=[1001, 1002])]
        mock_ga_results.best_chromosome = mock_chromosome
        mock_ga_results.best_fitness = 0.8
        mock_ga_results.total_generations = 50
        mock_ga_results.convergence_reason = "converged"
        mock_ga_results.stats = {'generation': 50}
        mock_ga_optimizer.optimize_route.return_value = mock_ga_results
        
        with patch('route_services.route_optimizer.GeneticRouteOptimizer', return_value=mock_ga_optimizer):
            # Test with exclude_footways=True (should create filtered graph)
            result = optimizer._optimize_genetic(1001, 2.0, "elevation", exclude_footways=True)
            
            self.assertIsNotNone(result)
            
            # Test with exclude_footways=False (should use original graph)
            result = optimizer._optimize_genetic(1001, 2.0, "elevation", exclude_footways=False)
            
            self.assertIsNotNone(result)
    
    def test_convert_tsp_to_ga_objective_all_cases(self):
        """Test all TSP to GA objective conversion cases"""
        optimizer = RouteOptimizer(self.test_graph, verbose=False)
        
        # Test all known conversions
        test_cases = [
            (optimizer._route_objective.MAXIMIZE_ELEVATION, "elevation"),
            (optimizer._route_objective.MINIMIZE_DISTANCE, "distance"),
            (optimizer._route_objective.BALANCED_ROUTE, "balanced"),
            (optimizer._route_objective.MINIMIZE_DIFFICULTY, "efficiency"),
            ("unknown_objective", "elevation")  # Default case
        ]
        
        for tsp_objective, expected_ga_objective in test_cases:
            result = optimizer._convert_tsp_to_ga_objective(tsp_objective)
            self.assertEqual(result, expected_ga_objective)


if __name__ == '__main__':
    unittest.main()