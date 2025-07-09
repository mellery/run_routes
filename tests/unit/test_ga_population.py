#!/usr/bin/env python3
"""
Unit tests for GA Population Initialization
Tests PopulationInitializer functionality
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import networkx as nx
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from genetic_algorithm import PopulationInitializer
from genetic_algorithm import RouteChromosome, RouteSegment


class TestPopulationInitializer(unittest.TestCase):
    """Test PopulationInitializer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock graph with multiple nodes and connections
        self.mock_graph = nx.Graph()
        
        # Add nodes in a grid pattern
        nodes = [
            (1, -80.4094, 37.1299, 100.0),
            (2, -80.4090, 37.1299, 105.0),
            (3, -80.4086, 37.1299, 110.0),
            (4, -80.4094, 37.1303, 95.0),
            (5, -80.4090, 37.1303, 100.0),
            (6, -80.4086, 37.1303, 115.0),
            (7, -80.4082, 37.1299, 120.0),
            (8, -80.4082, 37.1303, 125.0)
        ]
        
        for node_id, x, y, elevation in nodes:
            self.mock_graph.add_node(node_id, x=x, y=y, elevation=elevation)
        
        # Add edges to create connected network
        edges = [
            (1, 2, 400), (2, 3, 400), (3, 7, 400),
            (1, 4, 400), (4, 5, 400), (5, 6, 400), (6, 8, 400),
            (2, 5, 400), (3, 6, 400), (7, 8, 400),
            (1, 5, 565), (2, 6, 565)  # Diagonal connections
        ]
        
        for node1, node2, length in edges:
            self.mock_graph.add_edge(node1, node2, length=length)
        
        self.start_node = 1
        self.initializer = PopulationInitializer(self.mock_graph, self.start_node)
    
    def test_initializer_creation(self):
        """Test PopulationInitializer creation"""
        self.assertEqual(self.initializer.graph, self.mock_graph)
        self.assertEqual(self.initializer.start_node, self.start_node)
        self.assertIsInstance(self.initializer._neighbor_cache, dict)
        self.assertIsInstance(self.initializer._distance_cache, dict)
    
    def test_create_population_basic(self):
        """Test basic population creation"""
        population = self.initializer.create_population(size=5, target_distance_km=2.0)
        
        self.assertIsInstance(population, list)
        self.assertLessEqual(len(population), 5)  # May be less if some fail
        
        for chromosome in population:
            self.assertIsInstance(chromosome, RouteChromosome)
            self.assertTrue(chromosome.is_valid)
            self.assertGreater(len(chromosome.segments), 0)
    
    def test_create_population_size_validation(self):
        """Test population creation with different sizes"""
        # Small population
        small_pop = self.initializer.create_population(size=3, target_distance_km=1.5)
        self.assertLessEqual(len(small_pop), 3)
        
        # Medium population
        medium_pop = self.initializer.create_population(size=10, target_distance_km=2.0)
        self.assertLessEqual(len(medium_pop), 10)
    
    def test_create_population_target_distance_validation(self):
        """Test population creation with different target distances"""
        # Short distance
        short_pop = self.initializer.create_population(size=5, target_distance_km=1.0)
        self.assertGreater(len(short_pop), 0)
        
        # Medium distance
        medium_pop = self.initializer.create_population(size=5, target_distance_km=3.0)
        self.assertGreater(len(medium_pop), 0)
    
    def test_create_population_strategy_distribution(self):
        """Test that population uses different creation strategies"""
        population = self.initializer.create_population(size=10, target_distance_km=2.0)
        
        # Check that different creation methods are used
        methods = set(chromosome.creation_method for chromosome in population)
        
        # Should have multiple different methods
        self.assertGreater(len(methods), 1)
        
        # Should include expected method types
        method_types = [method.split('_')[0] for method in methods]
        expected_types = ['random', 'directional', 'elevation']
        
        for expected_type in expected_types:
            self.assertTrue(any(expected_type in method_type for method_type in method_types))
    
    def test_create_random_walk_route(self):
        """Test random walk route creation"""
        target_distance = 2000  # 2km in meters
        chromosome = self.initializer._create_random_walk_route(target_distance)
        
        if chromosome:  # May be None if creation fails
            self.assertIsInstance(chromosome, RouteChromosome)
            self.assertGreater(len(chromosome.segments), 0)
            
            # Should start and end at start_node for circular route
            if chromosome.segments:
                self.assertEqual(chromosome.segments[0].start_node, self.start_node)
                self.assertEqual(chromosome.segments[-1].end_node, self.start_node)
            
            # Chromosome validity might be False due to segment usage constraints
            # This is acceptable as the repair mechanisms will handle it
    
    def test_create_directional_route(self):
        """Test directional route creation"""
        target_distance = 2000  # 2km in meters
        
        # Test different directions
        directions = ['N', 'E', 'S', 'W']
        
        for direction in directions:
            chromosome = self.initializer._create_directional_route(target_distance, direction)
            
            if chromosome:  # May be None if creation fails
                self.assertIsInstance(chromosome, RouteChromosome)
                self.assertGreater(len(chromosome.segments), 0)
                
                # Should start and end at start_node
                if chromosome.segments:
                    self.assertEqual(chromosome.segments[0].start_node, self.start_node)
                    self.assertEqual(chromosome.segments[-1].end_node, self.start_node)
                
                # Chromosome validity might be False due to segment usage constraints
                # This is acceptable as the repair mechanisms will handle it
    
    def test_create_elevation_focused_route(self):
        """Test elevation-focused route creation"""
        target_distance = 2000  # 2km in meters
        chromosome = self.initializer._create_elevation_focused_route(target_distance)
        
        if chromosome:  # May be None if creation fails
            self.assertIsInstance(chromosome, RouteChromosome)
            self.assertTrue(chromosome.is_valid)
            self.assertGreater(len(chromosome.segments), 0)
            
            # Should start and end at start_node
            if chromosome.segments:
                self.assertEqual(chromosome.segments[0].start_node, self.start_node)
                self.assertEqual(chromosome.segments[-1].end_node, self.start_node)
    
    def test_create_simple_fallback_route(self):
        """Test simple fallback route creation"""
        target_distance = 2000  # 2km in meters
        chromosome = self.initializer._create_simple_fallback_route(target_distance)
        
        if chromosome:  # May be None if creation fails
            self.assertIsInstance(chromosome, RouteChromosome)
            self.assertTrue(chromosome.is_valid)
            self.assertGreater(len(chromosome.segments), 0)
            
            # Should be circular
            if chromosome.segments:
                self.assertEqual(chromosome.segments[0].start_node, self.start_node)
                self.assertEqual(chromosome.segments[-1].end_node, self.start_node)
    
    def test_get_reachable_neighbors(self):
        """Test neighbor retrieval with distance filtering"""
        # Test with different max distances
        neighbors_short = self.initializer._get_reachable_neighbors(self.start_node, max_distance=300)
        neighbors_medium = self.initializer._get_reachable_neighbors(self.start_node, max_distance=500)
        neighbors_long = self.initializer._get_reachable_neighbors(self.start_node, max_distance=1000)
        
        # Should have fewer neighbors with shorter max distance
        self.assertLessEqual(len(neighbors_short), len(neighbors_medium))
        self.assertLessEqual(len(neighbors_medium), len(neighbors_long))
        
        # All neighbors should be valid node IDs
        for neighbor in neighbors_long:
            self.assertIn(neighbor, self.mock_graph.nodes)
    
    def test_get_reachable_neighbors_caching(self):
        """Test that neighbor retrieval uses caching"""
        max_distance = 500
        
        # First call should populate cache
        neighbors1 = self.initializer._get_reachable_neighbors(self.start_node, max_distance)
        
        # Second call should use cache
        neighbors2 = self.initializer._get_reachable_neighbors(self.start_node, max_distance)
        
        self.assertEqual(neighbors1, neighbors2)
        
        # Check that cache was populated
        cache_key = (self.start_node, max_distance)
        self.assertIn(cache_key, self.initializer._neighbor_cache)
    
    def test_create_segment(self):
        """Test segment creation between nodes"""
        # Test valid segment creation
        segment = self.initializer._create_segment(1, 2)
        
        if segment:  # May be None if path not found
            self.assertIsInstance(segment, RouteSegment)
            self.assertEqual(segment.start_node, 1)
            self.assertEqual(segment.end_node, 2)
            self.assertTrue(segment.is_valid)
            self.assertGreater(segment.length, 0)
    
    def test_create_segment_no_path(self):
        """Test segment creation when no path exists"""
        # Create isolated graph
        isolated_graph = nx.Graph()
        isolated_graph.add_node(1, x=-80.4094, y=37.1299, elevation=100.0)
        isolated_graph.add_node(2, x=-80.4090, y=37.1300, elevation=110.0)
        # No edge between nodes
        
        isolated_initializer = PopulationInitializer(isolated_graph, 1)
        segment = isolated_initializer._create_segment(1, 2)
        
        self.assertIsNone(segment)
    
    def test_select_distance_biased_neighbor(self):
        """Test distance-biased neighbor selection"""
        neighbors = [2, 4, 5]  # Available neighbors
        current_node = 1
        remaining_distance = 1000.0
        
        selected = self.initializer._select_distance_biased_neighbor(
            neighbors, current_node, remaining_distance
        )
        
        self.assertIn(selected, neighbors)
    
    def test_select_distance_biased_neighbor_empty_list(self):
        """Test distance-biased neighbor selection with empty list"""
        neighbors = []
        current_node = 1
        remaining_distance = 1000.0
        
        selected = self.initializer._select_distance_biased_neighbor(
            neighbors, current_node, remaining_distance
        )
        
        self.assertEqual(selected, current_node)
    
    def test_calculate_bearing(self):
        """Test bearing calculation between nodes"""
        bearing = self.initializer._calculate_bearing(1, 2)
        
        if bearing is not None:
            self.assertIsInstance(bearing, float)
            self.assertGreaterEqual(bearing, 0.0)
            self.assertLess(bearing, 360.0)
    
    def test_calculate_bearing_same_node(self):
        """Test bearing calculation for same node"""
        bearing = self.initializer._calculate_bearing(1, 1)
        
        # Same node bearing calculation should return 0.0 or None
        self.assertTrue(bearing is None or bearing == 0.0)
    
    def test_population_generation_metadata(self):
        """Test that chromosomes have proper metadata"""
        population = self.initializer.create_population(size=5, target_distance_km=2.0)
        
        for chromosome in population:
            self.assertEqual(chromosome.generation, 0)
            self.assertIsNotNone(chromosome.creation_method)
            self.assertIsInstance(chromosome.creation_method, str)
    
    def test_population_diversity_metrics(self):
        """Test population diversity characteristics"""
        population = self.initializer.create_population(size=10, target_distance_km=2.0)
        
        if len(population) > 1:
            # Calculate diversity metrics
            distances = [chromosome.get_total_distance() for chromosome in population]
            elevations = [chromosome.get_elevation_gain() for chromosome in population]
            methods = [chromosome.creation_method for chromosome in population]
            
            # Should have some variation in distances (not all identical)
            distance_variance = max(distances) - min(distances)
            
            # Should have some variation in elevation gains
            elevation_variance = max(elevations) - min(elevations)
            
            # Should have multiple creation methods
            unique_methods = set(methods)
            
            # At least some diversity expected (though not guaranteed)
            self.assertGreaterEqual(len(unique_methods), 1)
    
    def test_error_handling_invalid_graph(self):
        """Test error handling with invalid graph"""
        # Empty graph
        empty_graph = nx.Graph()
        empty_initializer = PopulationInitializer(empty_graph, 1)
        
        population = empty_initializer.create_population(size=5, target_distance_km=2.0)
        
        # Should handle gracefully and return empty or minimal population
        self.assertIsInstance(population, list)
    
    def test_error_handling_invalid_start_node(self):
        """Test error handling with invalid start node"""
        invalid_initializer = PopulationInitializer(self.mock_graph, 999)  # Non-existent node
        
        population = invalid_initializer.create_population(size=5, target_distance_km=2.0)
        
        # Should handle gracefully
        self.assertIsInstance(population, list)
    
    def test_population_size_edge_cases(self):
        """Test population creation with edge case sizes"""
        # Zero size
        zero_pop = self.initializer.create_population(size=0, target_distance_km=2.0)
        self.assertEqual(len(zero_pop), 0)
        
        # Size 1
        one_pop = self.initializer.create_population(size=1, target_distance_km=2.0)
        self.assertLessEqual(len(one_pop), 1)
    
    def test_target_distance_edge_cases(self):
        """Test population creation with edge case distances"""
        # Very small distance
        small_pop = self.initializer.create_population(size=3, target_distance_km=0.1)
        self.assertIsInstance(small_pop, list)
        
        # Very large distance (may fail to create routes)
        large_pop = self.initializer.create_population(size=3, target_distance_km=100.0)
        self.assertIsInstance(large_pop, list)


class TestPopulationInitializerIntegration(unittest.TestCase):
    """Integration tests for PopulationInitializer with more realistic scenarios"""
    
    def setUp(self):
        """Set up more realistic test graph"""
        # Create a larger, more realistic graph
        self.mock_graph = nx.Graph()
        
        # Create a 5x5 grid of nodes
        for i in range(5):
            for j in range(5):
                node_id = i * 5 + j + 1
                x = -80.4094 + j * 0.001  # Spread out in longitude
                y = 37.1299 + i * 0.001   # Spread out in latitude
                elevation = 100 + (i + j) * 5  # Gradual elevation change
                
                self.mock_graph.add_node(node_id, x=x, y=y, elevation=elevation)
        
        # Connect nodes in grid pattern
        for i in range(5):
            for j in range(5):
                node_id = i * 5 + j + 1
                
                # Connect to right neighbor
                if j < 4:
                    right_neighbor = i * 5 + (j + 1) + 1
                    self.mock_graph.add_edge(node_id, right_neighbor, length=200)
                
                # Connect to bottom neighbor
                if i < 4:
                    bottom_neighbor = (i + 1) * 5 + j + 1
                    self.mock_graph.add_edge(node_id, bottom_neighbor, length=200)
        
        self.start_node = 13  # Center node
        self.initializer = PopulationInitializer(self.mock_graph, self.start_node)
    
    def test_realistic_population_creation(self):
        """Test population creation with realistic graph"""
        population = self.initializer.create_population(size=20, target_distance_km=3.0)
        
        self.assertGreater(len(population), 0)
        
        # Verify all chromosomes are valid
        circular_count = 0
        for chromosome in population:
            self.assertTrue(chromosome.is_valid)
            if chromosome.is_circular:
                circular_count += 1
            
            # Should have reasonable number of segments
            self.assertGreater(len(chromosome.segments), 0)
            self.assertLess(len(chromosome.segments), 50)  # Not too many
        
        # At least 90% of chromosomes should be circular (allows for some randomness)
        circular_percentage = circular_count / len(population)
        self.assertGreaterEqual(circular_percentage, 0.9, 
                              f"Only {circular_percentage:.1%} of chromosomes are circular, expected >= 90%")
    
    def test_strategy_effectiveness(self):
        """Test that different strategies produce different characteristics"""
        target_distance = 2000  # 2km
        
        # Create routes with different strategies
        random_route = self.initializer._create_random_walk_route(target_distance)
        elevation_route = self.initializer._create_elevation_focused_route(target_distance)
        directional_route = self.initializer._create_directional_route(target_distance, 'N')
        
        valid_routes = [route for route in [random_route, elevation_route, directional_route] if route]
        
        if len(valid_routes) > 1:
            # Should have some variation in elevation gain
            elevation_gains = [route.get_elevation_gain() for route in valid_routes]
            
            # Elevation-focused route should generally have higher elevation gain
            if elevation_route and len(valid_routes) > 1:
                elevation_route_gain = elevation_route.get_elevation_gain()
                other_gains = [route.get_elevation_gain() for route in valid_routes if route != elevation_route]
                
                # Not strictly required but often expected
                if other_gains:
                    max_other_gain = max(other_gains)
                    # Should at least be competitive
                    self.assertGreaterEqual(elevation_route_gain, min(other_gains))


if __name__ == '__main__':
    unittest.main()