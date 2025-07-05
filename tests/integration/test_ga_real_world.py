#!/usr/bin/env python3
"""
GA Real-World Testing Framework
Test genetic algorithm with actual street networks and elevation data
"""

import unittest
import time
import os
import networkx as nx
from typing import Dict, Any, Optional

# Import route services
from route_services import NetworkManager, RouteOptimizer, RouteAnalyzer


class TestGARealWorld(unittest.TestCase):
    """Test GA with real street networks and elevation data"""
    
    @classmethod
    def setUpClass(cls):
        """Set up real-world test environment"""
        try:
            # Initialize network manager with real data
            cls.network_manager = NetworkManager()
            
            # Load small real network (0.5km radius to keep tests fast)
            cls.graph = cls.network_manager.load_network(radius_km=0.5)
            
            if not cls.graph:
                cls.skipTest("Could not load real street network")
            
            # Initialize optimizers
            cls.route_optimizer = RouteOptimizer(cls.graph)
            cls.route_analyzer = RouteAnalyzer(cls.graph)
            
            # Get solver info
            cls.solver_info = cls.route_optimizer.get_solver_info()
            
            # Find a valid starting node
            cls.start_node = cls._find_valid_start_node()
            
        except Exception as e:
            cls.skipTest(f"Could not initialize real-world testing: {e}")
    
    @classmethod
    def _find_valid_start_node(cls) -> Optional[int]:
        """Find a valid starting node in the graph"""
        if not cls.graph or len(cls.graph.nodes) == 0:
            return None
        
        # Find a node with good connectivity
        for node_id in cls.graph.nodes():
            if cls.graph.degree(node_id) >= 2:  # At least 2 connections
                return node_id
        
        # Fallback to any node
        return list(cls.graph.nodes())[0]
    
    def setUp(self):
        """Set up individual test"""
        if not hasattr(self.__class__, 'graph') or not self.__class__.graph:
            self.skipTest("Real street network not available")
        
        if not self.__class__.start_node:
            self.skipTest("No valid starting node found")
    
    def test_ga_with_real_network_small_route(self):
        """Test GA with real network for small route"""
        if not self.solver_info.get('ga_available', False):
            self.skipTest("GA not available")
        
        # Test small route (1km)
        result = self.route_optimizer.optimize_route(
            start_node=self.start_node,
            target_distance_km=1.0,
            objective=self.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION,
            algorithm="genetic"
        )
        
        if result:
            self._validate_real_world_result(result, 1.0)
    
    def test_ga_with_real_network_medium_route(self):
        """Test GA with real network for medium route"""
        if not self.solver_info.get('ga_available', False):
            self.skipTest("GA not available")
        
        # Test medium route (2km)
        result = self.route_optimizer.optimize_route(
            start_node=self.start_node,
            target_distance_km=2.0,
            objective=self.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION,
            algorithm="genetic"
        )
        
        if result:
            self._validate_real_world_result(result, 2.0)
    
    def test_ga_memory_usage_real_network(self):
        """Test GA memory usage with real network"""
        if not self.solver_info.get('ga_available', False):
            self.skipTest("GA not available")
        
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run GA optimization
        result = self.route_optimizer.optimize_route(
            start_node=self.start_node,
            target_distance_km=1.5,
            objective=self.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION,
            algorithm="genetic"
        )
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 500MB for small network)
        self.assertLess(memory_increase, 500, 
                       f"Memory usage increased by {memory_increase:.1f}MB")
        
        if result:
            self._validate_real_world_result(result, 1.5)
    
    def test_ga_performance_real_network(self):
        """Test GA performance with real network"""
        if not self.solver_info.get('ga_available', False):
            self.skipTest("GA not available")
        
        start_time = time.time()
        
        # Run GA optimization
        result = self.route_optimizer.optimize_route(
            start_node=self.start_node,
            target_distance_km=1.0,
            objective=self.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION,
            algorithm="genetic"
        )
        
        total_time = time.time() - start_time
        
        # Should complete in reasonable time for small network
        self.assertLess(total_time, 120.0, 
                       f"GA took {total_time:.1f}s, expected < 120s")
        
        if result:
            solver_info = result.get('solver_info', {})
            if 'ga_generations' in solver_info:
                generations = solver_info['ga_generations']
                self.assertGreater(generations, 0)
                self.assertLess(generations, 1000)  # Reasonable generation count
    
    def test_ga_vs_tsp_real_network_comparison(self):
        """Test GA vs TSP comparison on real network"""
        if not self.solver_info.get('ga_available', False):
            self.skipTest("GA not available")
        
        # Test parameters
        distance_km = 1.5
        objective = self.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION
        
        # Run TSP optimization
        tsp_result = self.route_optimizer.optimize_route(
            start_node=self.start_node,
            target_distance_km=distance_km,
            objective=objective,
            algorithm="nearest_neighbor"
        )
        
        # Run GA optimization
        ga_result = self.route_optimizer.optimize_route(
            start_node=self.start_node,
            target_distance_km=distance_km,
            objective=objective,
            algorithm="genetic"
        )
        
        if tsp_result and ga_result:
            # Both should produce valid results
            self._validate_real_world_result(tsp_result, distance_km)
            self._validate_real_world_result(ga_result, distance_km)
            
            # Compare key metrics
            tsp_stats = tsp_result['stats']
            ga_stats = ga_result['stats']
            
            # Both should have reasonable distances
            tsp_distance = tsp_stats.get('total_distance_km', 0)
            ga_distance = ga_stats.get('total_distance_km', 0)
            
            self.assertGreater(tsp_distance, 0)
            self.assertGreater(ga_distance, 0)
            
            # For elevation objective, GA might find better elevation gain
            if 'total_elevation_gain_m' in tsp_stats and 'total_elevation_gain_m' in ga_stats:
                tsp_elevation = tsp_stats['total_elevation_gain_m']
                ga_elevation = ga_stats['total_elevation_gain_m']
                
                # Both should have non-negative elevation gain
                self.assertGreaterEqual(tsp_elevation, 0)
                self.assertGreaterEqual(ga_elevation, 0)
    
    def test_ga_all_objectives_real_network(self):
        """Test GA with all objectives on real network"""
        if not self.solver_info.get('ga_available', False):
            self.skipTest("GA not available")
        
        objectives = [
            self.route_optimizer.RouteObjective.MINIMIZE_DISTANCE,
            self.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION,
            self.route_optimizer.RouteObjective.BALANCED_ROUTE,
            self.route_optimizer.RouteObjective.MINIMIZE_DIFFICULTY
        ]
        
        distance_km = 1.0
        
        for objective in objectives:
            with self.subTest(objective=objective):
                result = self.route_optimizer.optimize_route(
                    start_node=self.start_node,
                    target_distance_km=distance_km,
                    objective=objective,
                    algorithm="genetic"
                )
                
                if result:
                    self._validate_real_world_result(result, distance_km)
                    
                    # Verify objective-specific behavior
                    stats = result['stats']
                    
                    if objective == self.route_optimizer.RouteObjective.MINIMIZE_DISTANCE:
                        # Should prioritize shorter routes
                        distance = stats.get('total_distance_km', 0)
                        self.assertGreater(distance, 0)
                    
                    elif objective == self.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION:
                        # Should prioritize elevation gain
                        elevation = stats.get('total_elevation_gain_m', 0)
                        self.assertGreaterEqual(elevation, 0)
    
    def test_ga_edge_cases_real_network(self):
        """Test GA edge cases with real network"""
        if not self.solver_info.get('ga_available', False):
            self.skipTest("GA not available")
        
        # Test very small distance
        result = self.route_optimizer.optimize_route(
            start_node=self.start_node,
            target_distance_km=0.1,
            objective=self.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION,
            algorithm="genetic"
        )
        
        # Should handle gracefully (might return None or minimal route)
        if result:
            self.assertIn('route', result)
            self.assertGreater(len(result['route']), 0)
    
    def test_ga_auto_selection_real_network(self):
        """Test auto algorithm selection with real network"""
        # Test auto selection for elevation objective
        result = self.route_optimizer.optimize_route(
            start_node=self.start_node,
            target_distance_km=1.0,
            objective=self.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION,
            algorithm="auto"
        )
        
        if result:
            solver_info = result.get('solver_info', {})
            algorithm_used = solver_info.get('algorithm_used', '')
            
            # Should select appropriate algorithm
            if self.solver_info.get('ga_available', False):
                # With GA available, might select genetic for elevation
                self.assertIn(algorithm_used, ['genetic', 'nearest_neighbor'])
            else:
                # Without GA, should fall back to TSP
                self.assertEqual(algorithm_used, 'nearest_neighbor')
    
    def _validate_real_world_result(self, result: Dict[str, Any], expected_distance_km: float):
        """Validate real-world optimization result"""
        # Basic structure validation
        self.assertIsInstance(result, dict)
        self.assertIn('route', result)
        self.assertIn('stats', result)
        self.assertIn('solver_info', result)
        
        # Route validation
        route = result['route']
        self.assertIsInstance(route, list)
        self.assertGreater(len(route), 1)
        self.assertEqual(route[0], self.start_node)  # Starts correctly
        self.assertEqual(route[-1], self.start_node)  # Returns to start
        
        # All nodes should exist in graph
        for node in route:
            self.assertIn(node, self.graph.nodes)
        
        # Route should be connected
        for i in range(len(route) - 1):
            self.assertTrue(
                self.graph.has_edge(route[i], route[i + 1]),
                f"No edge between {route[i]} and {route[i + 1]}"
            )
        
        # Stats validation
        stats = result['stats']
        self.assertIn('total_distance_km', stats)
        self.assertIn('total_distance_m', stats)
        
        distance_km = stats['total_distance_km']
        self.assertGreater(distance_km, 0)
        
        # Distance should be reasonably close to target (within 50% tolerance for real networks)
        distance_ratio = distance_km / expected_distance_km
        self.assertGreater(distance_ratio, 0.5, 
                          f"Route too short: {distance_km:.2f}km vs {expected_distance_km:.2f}km target")
        self.assertLess(distance_ratio, 2.0, 
                       f"Route too long: {distance_km:.2f}km vs {expected_distance_km:.2f}km target")
        
        # Solver info validation
        solver_info = result['solver_info']
        self.assertIn('solver_type', solver_info)
        self.assertIn('solve_time', solver_info)
        self.assertIn('algorithm_used', solver_info)
        
        solve_time = solver_info['solve_time']
        self.assertGreater(solve_time, 0)
        self.assertLess(solve_time, 300)  # Should complete within 5 minutes
        
        # GA-specific validation
        if solver_info.get('solver_type') == 'genetic':
            self.assertIn('ga_generations', solver_info)
            self.assertIn('ga_convergence', solver_info)
            
            generations = solver_info['ga_generations']
            self.assertGreater(generations, 0)


class TestGAStressTest(unittest.TestCase):
    """Stress test GA with real network"""
    
    def setUp(self):
        """Set up stress test"""
        try:
            self.network_manager = NetworkManager()
            self.graph = self.network_manager.load_network(radius_km=0.3)  # Smaller for stress test
            
            if not self.graph:
                self.skipTest("Could not load network for stress test")
            
            self.route_optimizer = RouteOptimizer(self.graph)
            solver_info = self.route_optimizer.get_solver_info()
            
            if not solver_info.get('ga_available', False):
                self.skipTest("GA not available for stress test")
            
            # Find valid start node
            self.start_node = list(self.graph.nodes())[0]
            
        except Exception as e:
            self.skipTest(f"Could not initialize stress test: {e}")
    
    def test_ga_multiple_runs_consistency(self):
        """Test GA consistency across multiple runs"""
        results = []
        num_runs = 3  # Limited for CI/CD performance
        
        for i in range(num_runs):
            result = self.route_optimizer.optimize_route(
                start_node=self.start_node,
                target_distance_km=1.0,
                objective=self.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION,
                algorithm="genetic"
            )
            
            if result:
                results.append(result)
        
        # Should get some successful results
        self.assertGreater(len(results), 0)
        
        # All results should be valid
        for result in results:
            stats = result['stats']
            distance = stats.get('total_distance_km', 0)
            self.assertGreater(distance, 0)
            self.assertLess(distance, 3.0)  # Reasonable upper bound


if __name__ == '__main__':
    # Set test timeout
    unittest.main(verbosity=2)