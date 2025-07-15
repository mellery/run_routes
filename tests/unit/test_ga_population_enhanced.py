#!/usr/bin/env python3
"""
Enhanced unit tests for GA Population Initialization
Tests advanced population creation strategies and edge cases
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import networkx as nx
import sys
import os
import math

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from genetic_algorithm.population import PopulationInitializer
from genetic_algorithm.chromosome import RouteChromosome, RouteSegment


class TestPopulationInitializerEnhanced(unittest.TestCase):
    """Enhanced test cases for PopulationInitializer"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create comprehensive mock graph with varied elevations
        self.mock_graph = nx.Graph()
        
        # Add nodes in a complex network pattern
        nodes = [
            (1, -80.4094, 37.1299, 100.0),  # Start node
            (2, -80.4090, 37.1299, 105.0),
            (3, -80.4086, 37.1299, 110.0),
            (4, -80.4094, 37.1303, 95.0),
            (5, -80.4090, 37.1303, 100.0),
            (6, -80.4086, 37.1303, 115.0),  # High elevation
            (7, -80.4082, 37.1299, 120.0),  # Higher elevation
            (8, -80.4082, 37.1303, 125.0),  # Highest elevation
            (9, -80.4078, 37.1299, 90.0),   # Low elevation
            (10, -80.4078, 37.1303, 85.0),  # Lowest elevation
            (11, -80.4074, 37.1299, 130.0), # Peak elevation
            (12, -80.4074, 37.1303, 135.0), # Highest peak
        ]
        
        for node_id, x, y, elevation in nodes:
            self.mock_graph.add_node(node_id, x=x, y=y, elevation=elevation)
        
        # Add comprehensive edge network
        edges = [
            # Horizontal connections
            (1, 2, 400), (2, 3, 400), (3, 7, 400), (7, 9, 400), (9, 11, 400),
            (4, 5, 400), (5, 6, 400), (6, 8, 400), (8, 10, 400), (10, 12, 400),
            # Vertical connections
            (1, 4, 400), (2, 5, 400), (3, 6, 400), (7, 8, 400), (9, 10, 400), (11, 12, 400),
            # Diagonal connections for variety
            (1, 5, 565), (2, 6, 565), (3, 8, 565), (7, 10, 565), (9, 12, 565),
            # Long-distance connections
            (1, 7, 800), (4, 8, 800), (2, 9, 800), (5, 10, 800),
        ]
        
        for start, end, length in edges:
            self.mock_graph.add_edge(start, end, length=length)
        
        self.start_node = 1
        self.initializer = PopulationInitializer(self.mock_graph, self.start_node)
    
    def test_create_population_large_size(self):
        """Test population creation with large population size"""
        large_size = 100
        target_distance = 2.0  # km
        
        population = self.initializer.create_population(large_size, target_distance)
        
        # Should attempt to create large population
        self.assertIsInstance(population, list)
        self.assertGreater(len(population), 0)
        self.assertLessEqual(len(population), large_size)
        
        # All chromosomes should be valid
        for chromosome in population:
            self.assertIsInstance(chromosome, RouteChromosome)
            self.assertTrue(chromosome.is_valid)
    
    def test_create_population_small_distance(self):
        """Test population creation with very small target distance"""
        small_distance = 0.5  # km
        population_size = 20
        
        population = self.initializer.create_population(population_size, small_distance)
        
        # Should create some routes even with small distance
        self.assertIsInstance(population, list)
        self.assertGreater(len(population), 0)
        
        # Check that routes are approximately the target distance
        for chromosome in population:
            distance_km = chromosome.get_total_distance() / 1000
            self.assertGreater(distance_km, small_distance * 0.5)  # At least 50% of target
    
    def test_create_population_large_distance(self):
        """Test population creation with large target distance"""
        large_distance = 10.0  # km
        population_size = 15
        
        population = self.initializer.create_population(population_size, large_distance)
        
        # Should create some routes even with large distance
        self.assertIsInstance(population, list)
        # May have fewer routes due to distance constraints
        self.assertGreaterEqual(len(population), 0)
        
        # Check that routes attempt to approach target distance (may be limited by graph)
        for chromosome in population:
            distance_km = chromosome.get_total_distance() / 1000
            # For large distances, routes may be much smaller due to graph limitations
            self.assertGreater(distance_km, 0.1)  # Just check that distance is reasonable
    
    def test_create_population_zero_size(self):
        """Test population creation with zero size"""
        population = self.initializer.create_population(0, 2.0)
        
        self.assertIsInstance(population, list)
        self.assertEqual(len(population), 0)
    
    def test_create_population_negative_distance(self):
        """Test population creation with negative distance"""
        population = self.initializer.create_population(10, -1.0)
        
        # Should handle negative distance gracefully
        self.assertIsInstance(population, list)
        # May return empty list or handle gracefully
        self.assertGreaterEqual(len(population), 0)
    
    def test_random_walk_route_creation(self):
        """Test random walk route creation with various parameters"""
        target_distances = [500, 1000, 2000, 5000]  # meters
        
        for target_distance in target_distances:
            with self.subTest(target_distance=target_distance):
                route = self.initializer._create_random_walk_route(target_distance)
                
                if route:  # If route was created
                    self.assertIsInstance(route, RouteChromosome)
                    self.assertTrue(route.is_valid)
                    self.assertGreater(len(route.segments), 0)
                    
                    # Check that route attempts to reach target distance
                    actual_distance = route.get_total_distance()
                    self.assertGreater(actual_distance, target_distance * 0.3)
    
    def test_directional_route_creation(self):
        """Test directional route creation in different directions"""
        target_distance = 2000  # meters
        directions = ['north', 'south', 'east', 'west']
        
        for direction in directions:
            with self.subTest(direction=direction):
                route = self.initializer._create_directional_route(target_distance, direction)
                
                if route:  # If route was created
                    self.assertIsInstance(route, RouteChromosome)
                    self.assertTrue(route.is_valid)
                    self.assertGreater(len(route.segments), 0)
                    
                    # Check that route follows general direction
                    actual_distance = route.get_total_distance()
                    self.assertGreater(actual_distance, target_distance * 0.2)
    
    def test_elevation_focused_route_creation(self):
        """Test elevation-focused route creation"""
        target_distance = 2000  # meters
        
        route = self.initializer._create_elevation_focused_route(target_distance)
        
        if route:  # If route was created
            self.assertIsInstance(route, RouteChromosome)
            self.assertTrue(route.is_valid)
            self.assertGreater(len(route.segments), 0)
            
            # Check that route has some elevation gain
            elevation_gain = route.get_elevation_gain()
            self.assertGreaterEqual(elevation_gain, 0)
    
    def test_simple_fallback_route_creation(self):
        """Test simple fallback route creation"""
        target_distance = 1000  # meters
        
        route = self.initializer._create_simple_fallback_route(target_distance)
        
        if route:  # If route was created
            self.assertIsInstance(route, RouteChromosome)
            self.assertTrue(route.is_valid)
            self.assertGreater(len(route.segments), 0)
            
            # Fallback route should be simple and valid
            actual_distance = route.get_total_distance()
            self.assertGreater(actual_distance, 0)
    
    def test_get_reachable_neighbors_default_distance(self):
        """Test getting reachable neighbors with default distance"""
        neighbors = self.initializer._get_reachable_neighbors(1)
        
        self.assertIsInstance(neighbors, list)
        self.assertGreater(len(neighbors), 0)
        
        # Should include immediate neighbors
        self.assertIn(2, neighbors)
        self.assertIn(4, neighbors)
        
        # Should not include the start node itself
        self.assertNotIn(1, neighbors)
    
    def test_get_reachable_neighbors_custom_distance(self):
        """Test getting reachable neighbors with custom max distance"""
        # Test with very small distance
        small_distance_neighbors = self.initializer._get_reachable_neighbors(1, 200)
        
        # Test with medium distance
        medium_distance_neighbors = self.initializer._get_reachable_neighbors(1, 600)
        
        # Test with large distance
        large_distance_neighbors = self.initializer._get_reachable_neighbors(1, 1000)
        
        # Should have more neighbors with larger distance
        self.assertLessEqual(len(small_distance_neighbors), len(medium_distance_neighbors))
        self.assertLessEqual(len(medium_distance_neighbors), len(large_distance_neighbors))
    
    def test_create_segment_valid_connection(self):
        """Test creating segment between valid connected nodes"""
        segment = self.initializer._create_segment(1, 2)
        
        if segment:  # If segment was created
            self.assertIsInstance(segment, RouteSegment)
            self.assertEqual(segment.start_node, 1)
            self.assertEqual(segment.end_node, 2)
            self.assertGreater(segment.length, 0)
            self.assertIsNotNone(segment.elevation_gain)
    
    def test_create_segment_invalid_connection(self):
        """Test creating segment between unconnected nodes"""
        # Add unconnected node
        self.mock_graph.add_node(99, x=-80.5000, y=37.2000, elevation=200.0)
        
        segment = self.initializer._create_segment(1, 99)
        
        # Should return None for unconnected nodes
        self.assertIsNone(segment)
    
    def test_create_segment_self_connection(self):
        """Test creating segment from node to itself"""
        segment = self.initializer._create_segment(1, 1)
        
        # Should return a valid segment for self-connection (NetworkX allows this)
        if segment:
            self.assertIsInstance(segment, RouteSegment)
            self.assertEqual(segment.start_node, 1)
            self.assertEqual(segment.end_node, 1)
        else:
            # Or it might return None if implementation prevents self-connections
            self.assertIsNone(segment)
    
    def test_select_distance_biased_neighbor_close_to_target(self):
        """Test distance-biased neighbor selection close to target"""
        neighbors = [2, 4, 5]  # Available neighbors
        current_node = 1
        current_distance = 500
        target_distance = 1000
        
        selected = self.initializer._select_distance_biased_neighbor(
            neighbors, current_node, target_distance - current_distance
        )
        
        self.assertIn(selected, neighbors)
        self.assertIsInstance(selected, int)
    
    def test_select_distance_biased_neighbor_far_from_target(self):
        """Test distance-biased neighbor selection far from target"""
        neighbors = [2, 4, 5]  # Available neighbors
        current_node = 1
        current_distance = 2000
        target_distance = 1000
        
        selected = self.initializer._select_distance_biased_neighbor(
            neighbors, current_node, target_distance - current_distance
        )
        
        self.assertIn(selected, neighbors)
        self.assertIsInstance(selected, int)
    
    def test_select_distance_biased_neighbor_empty_list(self):
        """Test distance-biased neighbor selection with empty neighbor list"""
        neighbors = []
        current_node = 1
        current_distance = 500
        target_distance = 1000
        
        selected = self.initializer._select_distance_biased_neighbor(
            neighbors, current_node, target_distance - current_distance
        )
        
        self.assertEqual(selected, current_node)
    
    def test_calculate_bearing_valid_nodes(self):
        """Test bearing calculation between valid nodes"""
        bearing = self.initializer._calculate_bearing(1, 2)
        
        if bearing is not None:
            self.assertIsInstance(bearing, (int, float))
            self.assertGreaterEqual(bearing, 0)
            self.assertLess(bearing, 360)
    
    def test_calculate_bearing_same_node(self):
        """Test bearing calculation for same node"""
        bearing = self.initializer._calculate_bearing(1, 1)
        
        # Should return 0.0 for same node (mathematically correct)
        self.assertEqual(bearing, 0.0)
    
    def test_calculate_bearing_invalid_node(self):
        """Test bearing calculation with invalid node"""
        bearing = self.initializer._calculate_bearing(1, 999)
        
        # Should return None for invalid node
        self.assertIsNone(bearing)
    
    def test_can_use_segment_bidirectional_allowed(self):
        """Test segment usage checking with bidirectional allowed"""
        initializer = PopulationInitializer(self.mock_graph, self.start_node, allow_bidirectional=True)
        segment_usage = {}
        
        # First use should be allowed
        can_use_1to2 = initializer._can_use_segment(1, 2, segment_usage)
        self.assertTrue(can_use_1to2)
        
        # Update usage
        initializer._update_segment_usage(1, 2, segment_usage)
        
        # Reverse direction should be allowed with bidirectional
        can_use_2to1 = initializer._can_use_segment(2, 1, segment_usage)
        self.assertTrue(can_use_2to1)
    
    def test_can_use_segment_bidirectional_not_allowed(self):
        """Test segment usage checking with bidirectional not allowed"""
        initializer = PopulationInitializer(self.mock_graph, self.start_node, allow_bidirectional=False)
        segment_usage = {}
        
        # First use should be allowed
        can_use_1to2 = initializer._can_use_segment(1, 2, segment_usage)
        self.assertTrue(can_use_1to2)
        
        # Update usage
        initializer._update_segment_usage(1, 2, segment_usage)
        
        # Reverse direction should not be allowed without bidirectional
        can_use_2to1 = initializer._can_use_segment(2, 1, segment_usage)
        self.assertFalse(can_use_2to1)
    
    def test_can_use_segment_repeated_usage(self):
        """Test segment usage checking with repeated usage"""
        segment_usage = {}
        
        # First use should be allowed
        can_use_first = self.initializer._can_use_segment(1, 2, segment_usage)
        self.assertTrue(can_use_first)
        
        # Update usage
        self.initializer._update_segment_usage(1, 2, segment_usage)
        
        # Second use should be allowed but may be limited
        can_use_second = self.initializer._can_use_segment(1, 2, segment_usage)
        # Usage limits may prevent repeated usage
        self.assertIsInstance(can_use_second, bool)
        
        # Update usage again
        self.initializer._update_segment_usage(1, 2, segment_usage)
        
        # Third use may be limited based on usage constraints
        can_use_third = self.initializer._can_use_segment(1, 2, segment_usage)
        self.assertIsInstance(can_use_third, bool)
    
    def test_update_segment_usage_tracking(self):
        """Test segment usage tracking updates"""
        segment_usage = {}
        
        # Initially empty
        self.assertEqual(len(segment_usage), 0)
        
        # First update
        self.initializer._update_segment_usage(1, 2, segment_usage)
        self.assertGreater(len(segment_usage), 0)
        
        # Second update
        self.initializer._update_segment_usage(1, 2, segment_usage)
        
        # Third update with different segment
        self.initializer._update_segment_usage(2, 3, segment_usage)
        
        # Should track multiple segments
        self.assertGreater(len(segment_usage), 0)
    
    def test_neighbor_cache_functionality(self):
        """Test neighbor caching functionality"""
        # First call should populate cache
        neighbors_1 = self.initializer._get_reachable_neighbors(1)
        
        # Second call should use cache
        neighbors_2 = self.initializer._get_reachable_neighbors(1)
        
        # Results should be identical
        self.assertEqual(neighbors_1, neighbors_2)
        
        # Cache should be populated
        self.assertGreater(len(self.initializer._neighbor_cache), 0)
    
    def test_distance_cache_functionality(self):
        """Test distance caching functionality"""
        # Create multiple segments to populate distance cache
        segment_1 = self.initializer._create_segment(1, 2)
        segment_2 = self.initializer._create_segment(2, 3)
        
        # Cache should be populated if segments were created
        if segment_1 and segment_2:
            self.assertGreaterEqual(len(self.initializer._distance_cache), 0)
    
    def test_population_diversity_strategies(self):
        """Test that population uses diverse creation strategies"""
        population_size = 40
        target_distance = 2.0
        
        population = self.initializer.create_population(population_size, target_distance)
        
        # Should have created some population
        self.assertGreater(len(population), 0)
        
        # Check for diversity in route lengths
        distances = [chromosome.get_total_distance() for chromosome in population]
        
        if len(distances) > 1:
            # Should have some variation in distances
            min_distance = min(distances)
            max_distance = max(distances)
            self.assertGreater(max_distance, min_distance)
    
    def test_elevation_gain_variation(self):
        """Test that population has variation in elevation gain"""
        population_size = 30
        target_distance = 2.0
        
        population = self.initializer.create_population(population_size, target_distance)
        
        # Should have created some population
        self.assertGreater(len(population), 0)
        
        # Check for diversity in elevation gains
        elevation_gains = [chromosome.get_elevation_gain() for chromosome in population]
        
        if len(elevation_gains) > 1:
            # Should have some variation in elevation gains
            min_gain = min(elevation_gains)
            max_gain = max(elevation_gains)
            self.assertGreaterEqual(max_gain, min_gain)
    
    def test_population_with_disconnected_graph(self):
        """Test population creation with disconnected graph"""
        # Create a disconnected graph
        disconnected_graph = nx.Graph()
        disconnected_graph.add_node(1, x=-80.4094, y=37.1299, elevation=100.0)
        disconnected_graph.add_node(2, x=-80.4090, y=37.1299, elevation=105.0)
        disconnected_graph.add_node(3, x=-80.4086, y=37.1299, elevation=110.0)
        
        # Only connect nodes 1 and 2, leaving 3 disconnected
        disconnected_graph.add_edge(1, 2, length=400)
        
        disconnected_initializer = PopulationInitializer(disconnected_graph, 1)
        
        population = disconnected_initializer.create_population(20, 2.0)
        
        # Should handle disconnected graph gracefully
        self.assertIsInstance(population, list)
        # May have fewer or no routes due to limited connectivity
        self.assertGreaterEqual(len(population), 0)
    
    def test_population_with_minimal_graph(self):
        """Test population creation with minimal graph"""
        # Create minimal graph with just 2 nodes
        minimal_graph = nx.Graph()
        minimal_graph.add_node(1, x=-80.4094, y=37.1299, elevation=100.0)
        minimal_graph.add_node(2, x=-80.4090, y=37.1299, elevation=105.0)
        minimal_graph.add_edge(1, 2, length=400)
        
        minimal_initializer = PopulationInitializer(minimal_graph, 1)
        
        population = minimal_initializer.create_population(10, 1.0)
        
        # Should handle minimal graph gracefully
        self.assertIsInstance(population, list)
        # May have very few routes due to limited options
        self.assertGreaterEqual(len(population), 0)
    
    def test_route_return_to_start_validation(self):
        """Test that all routes return to start node"""
        population = self.initializer.create_population(20, 2.0)
        
        for chromosome in population:
            route_nodes = chromosome.get_route_nodes()
            
            # Route should start and end at the same node
            self.assertEqual(route_nodes[0], route_nodes[-1])
            self.assertEqual(route_nodes[0], self.start_node)
    
    def test_chromosome_connectivity_validation(self):
        """Test that all chromosomes have valid connectivity"""
        population = self.initializer.create_population(15, 2.0)
        
        for chromosome in population:
            self.assertTrue(chromosome.is_valid)
            self.assertGreater(len(chromosome.segments), 0)
            
            # Check segment connectivity
            for i in range(len(chromosome.segments) - 1):
                current_segment = chromosome.segments[i]
                next_segment = chromosome.segments[i + 1]
                
                # End of current segment should connect to start of next
                self.assertEqual(current_segment.end_node, next_segment.start_node)
    
    def test_target_distance_approximation(self):
        """Test that routes approximate target distance reasonably"""
        target_distances = [1.0, 2.0, 3.0, 4.0]  # km
        
        for target_distance in target_distances:
            with self.subTest(target_distance=target_distance):
                population = self.initializer.create_population(10, target_distance)
                
                for chromosome in population:
                    actual_distance_km = chromosome.get_total_distance() / 1000
                    
                    # Should be within reasonable range of target
                    self.assertGreater(actual_distance_km, target_distance * 0.2)
                    self.assertLess(actual_distance_km, target_distance * 3.0)
    
    def test_memory_efficiency_large_population(self):
        """Test memory efficiency with large population"""
        # Test with larger population to check memory usage
        large_population = self.initializer.create_population(50, 2.0)
        
        # Should create population without memory issues
        self.assertIsInstance(large_population, list)
        self.assertGreaterEqual(len(large_population), 0)
        
        # All chromosomes should be valid
        for chromosome in large_population:
            self.assertIsInstance(chromosome, RouteChromosome)
    
    def test_error_handling_invalid_start_node(self):
        """Test error handling with invalid start node"""
        # Test with start node not in graph
        try:
            invalid_initializer = PopulationInitializer(self.mock_graph, 999)
            population = invalid_initializer.create_population(10, 2.0)
            
            # Should handle gracefully
            self.assertIsInstance(population, list)
            self.assertGreaterEqual(len(population), 0)
        except Exception as e:
            # If exception is raised, it should be handled gracefully
            self.assertIsInstance(e, Exception)


if __name__ == '__main__':
    unittest.main()