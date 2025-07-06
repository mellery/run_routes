#!/usr/bin/env python3
"""
Unit tests for GA Chromosome classes
Tests RouteSegment and RouteChromosome functionality
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import networkx as nx
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ga_chromosome import RouteSegment, RouteChromosome


class TestRouteSegment(unittest.TestCase):
    """Test RouteSegment class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock graph
        self.mock_graph = nx.Graph()
        self.mock_graph.add_node(1, x=-80.4094, y=37.1299, elevation=100.0)
        self.mock_graph.add_node(2, x=-80.4090, y=37.1300, elevation=110.0)
        self.mock_graph.add_node(3, x=-80.4086, y=37.1301, elevation=105.0)
        self.mock_graph.add_edge(1, 2, length=100.0)
        self.mock_graph.add_edge(2, 3, length=150.0)
    
    def test_segment_initialization(self):
        """Test segment initialization"""
        segment = RouteSegment(1, 2, [1, 2])
        
        self.assertEqual(segment.start_node, 1)
        self.assertEqual(segment.end_node, 2)
        self.assertEqual(segment.path_nodes, [1, 2])
        self.assertEqual(segment.length, 0.0)
        self.assertEqual(segment.elevation_gain, 0.0)
        self.assertTrue(segment.is_valid)
    
    def test_segment_properties_calculation(self):
        """Test segment property calculation"""
        segment = RouteSegment(1, 2, [1, 2])
        segment.calculate_properties(self.mock_graph)
        
        self.assertEqual(segment.length, 100.0)
        self.assertEqual(segment.elevation_gain, 10.0)  # 110 - 100
        self.assertEqual(segment.elevation_loss, 0.0)
        self.assertEqual(segment.net_elevation, 10.0)
        self.assertTrue(segment.is_valid)
        self.assertIsNotNone(segment.direction)
    
    def test_segment_multi_node_path(self):
        """Test segment with multiple nodes in path"""
        segment = RouteSegment(1, 3, [1, 2, 3])
        segment.calculate_properties(self.mock_graph)
        
        self.assertEqual(segment.length, 250.0)  # 100 + 150
        self.assertEqual(segment.elevation_gain, 10.0)  # Only positive changes
        self.assertEqual(segment.elevation_loss, 5.0)   # 110 -> 105
        self.assertEqual(segment.net_elevation, 5.0)    # 105 - 100
        self.assertTrue(segment.is_valid)
    
    def test_segment_invalid_path(self):
        """Test segment with invalid path"""
        # Test with non-existent edge
        mock_graph = nx.Graph()
        mock_graph.add_node(1, x=-80.4094, y=37.1299, elevation=100.0)
        mock_graph.add_node(2, x=-80.4090, y=37.1300, elevation=110.0)
        # No edge between nodes
        
        segment = RouteSegment(1, 2, [1, 2])
        segment.calculate_properties(mock_graph)
        
        self.assertFalse(segment.is_valid)
    
    def test_segment_empty_path(self):
        """Test segment with empty path"""
        segment = RouteSegment(1, 2, [])
        segment.calculate_properties(self.mock_graph)
        
        # Empty path gets defaulted to [start_node, end_node], so should be valid if edge exists
        self.assertTrue(segment.is_valid)
        self.assertEqual(segment.path_nodes, [1, 2])
    
    def test_segment_single_node_path(self):
        """Test segment with single node path"""
        segment = RouteSegment(1, 1, [1])
        segment.calculate_properties(self.mock_graph)
        
        self.assertFalse(segment.is_valid)
    
    def test_segment_direction_calculation(self):
        """Test direction calculation"""
        segment = RouteSegment(1, 2, [1, 2])
        segment.calculate_properties(self.mock_graph)
        
        # Should calculate some direction
        self.assertIsNotNone(segment.direction)
        self.assertIn(segment.direction, ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    
    def test_segment_get_summary_stats(self):
        """Test segment summary statistics"""
        segment = RouteSegment(1, 2, [1, 2])
        segment.calculate_properties(self.mock_graph)
        
        stats = segment.get_summary_stats()
        
        self.assertIn('length_m', stats)
        self.assertIn('elevation_gain_m', stats)
        self.assertIn('elevation_loss_m', stats)
        self.assertIn('net_elevation_m', stats)
        self.assertIn('max_grade_percent', stats)
        self.assertIn('direction', stats)
        self.assertIn('is_valid', stats)
        self.assertIn('node_count', stats)
        
        self.assertEqual(stats['length_m'], 100.0)
        self.assertEqual(stats['elevation_gain_m'], 10.0)
        self.assertTrue(stats['is_valid'])
        self.assertEqual(stats['node_count'], 2)
    
    def test_segment_copy(self):
        """Test segment copy method"""
        segment = RouteSegment(1, 2, [1, 2])
        segment.calculate_properties(self.mock_graph)
        
        segment_copy = segment.copy()
        
        self.assertEqual(segment.start_node, segment_copy.start_node)
        self.assertEqual(segment.end_node, segment_copy.end_node)
        self.assertEqual(segment.path_nodes, segment_copy.path_nodes)
        self.assertEqual(segment.length, segment_copy.length)
        self.assertEqual(segment.elevation_gain, segment_copy.elevation_gain)
        self.assertEqual(segment.is_valid, segment_copy.is_valid)
        
        # Ensure it's a deep copy
        segment_copy.length = 999.0
        self.assertNotEqual(segment.length, segment_copy.length)


class TestRouteChromosome(unittest.TestCase):
    """Test RouteChromosome class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock graph
        self.mock_graph = nx.Graph()
        self.mock_graph.add_node(1, x=-80.4094, y=37.1299, elevation=100.0)
        self.mock_graph.add_node(2, x=-80.4090, y=37.1300, elevation=110.0)
        self.mock_graph.add_node(3, x=-80.4086, y=37.1301, elevation=105.0)
        self.mock_graph.add_edge(1, 2, length=100.0)
        self.mock_graph.add_edge(2, 3, length=150.0)
        self.mock_graph.add_edge(3, 1, length=200.0)
        
        # Create test segments
        self.segment1 = RouteSegment(1, 2, [1, 2])
        self.segment1.calculate_properties(self.mock_graph)
        
        self.segment2 = RouteSegment(2, 3, [2, 3])
        self.segment2.calculate_properties(self.mock_graph)
        
        self.segment3 = RouteSegment(3, 1, [3, 1])
        self.segment3.calculate_properties(self.mock_graph)
    
    def test_chromosome_initialization_empty(self):
        """Test chromosome initialization with no segments"""
        chromosome = RouteChromosome()
        
        self.assertEqual(len(chromosome.segments), 0)
        self.assertIsNone(chromosome.fitness)
        self.assertIsNone(chromosome.distance)
        self.assertTrue(chromosome.is_valid)
        self.assertFalse(chromosome.is_circular)
    
    def test_chromosome_initialization_with_segments(self):
        """Test chromosome initialization with segments"""
        segments = [self.segment1, self.segment2]
        chromosome = RouteChromosome(segments)
        
        self.assertEqual(len(chromosome.segments), 2)
        self.assertEqual(chromosome.segments[0], self.segment1)
        self.assertEqual(chromosome.segments[1], self.segment2)
    
    def test_chromosome_add_segment(self):
        """Test adding segment to chromosome"""
        chromosome = RouteChromosome()
        chromosome.add_segment(self.segment1)
        
        self.assertEqual(len(chromosome.segments), 1)
        self.assertEqual(chromosome.segments[0], self.segment1)
    
    def test_chromosome_insert_segment(self):
        """Test inserting segment into chromosome"""
        chromosome = RouteChromosome([self.segment1, self.segment3])
        chromosome.insert_segment(1, self.segment2)
        
        self.assertEqual(len(chromosome.segments), 3)
        self.assertEqual(chromosome.segments[0], self.segment1)
        self.assertEqual(chromosome.segments[1], self.segment2)
        self.assertEqual(chromosome.segments[2], self.segment3)
    
    def test_chromosome_remove_segment(self):
        """Test removing segment from chromosome"""
        chromosome = RouteChromosome([self.segment1, self.segment2, self.segment3])
        removed = chromosome.remove_segment(1)
        
        self.assertEqual(len(chromosome.segments), 2)
        self.assertEqual(removed, self.segment2)
        self.assertEqual(chromosome.segments[0], self.segment1)
        self.assertEqual(chromosome.segments[1], self.segment3)
    
    def test_chromosome_remove_segment_invalid_index(self):
        """Test removing segment with invalid index"""
        chromosome = RouteChromosome([self.segment1])
        removed = chromosome.remove_segment(5)
        
        self.assertIsNone(removed)
        self.assertEqual(len(chromosome.segments), 1)
    
    def test_chromosome_validate_connectivity_valid(self):
        """Test connectivity validation for valid chromosome"""
        chromosome = RouteChromosome([self.segment1, self.segment2, self.segment3])
        is_valid = chromosome.validate_connectivity()
        
        self.assertTrue(is_valid)
        self.assertTrue(chromosome.is_valid)
        self.assertTrue(chromosome.is_circular)  # Returns to start
    
    def test_chromosome_validate_connectivity_invalid(self):
        """Test connectivity validation for invalid chromosome"""
        # Create disconnected segments
        segment_disconnected = RouteSegment(5, 6, [5, 6])  # Not connected to other segments
        chromosome = RouteChromosome([self.segment1, segment_disconnected])
        is_valid = chromosome.validate_connectivity()
        
        self.assertFalse(is_valid)
        self.assertFalse(chromosome.is_valid)
    
    def test_chromosome_validate_connectivity_empty(self):
        """Test connectivity validation for empty chromosome"""
        chromosome = RouteChromosome()
        is_valid = chromosome.validate_connectivity()
        
        self.assertFalse(is_valid)
        self.assertFalse(chromosome.is_valid)
    
    def test_chromosome_get_total_distance(self):
        """Test total distance calculation"""
        chromosome = RouteChromosome([self.segment1, self.segment2])
        distance = chromosome.get_total_distance()
        
        expected_distance = self.segment1.length + self.segment2.length
        self.assertEqual(distance, expected_distance)
        
        # Test caching
        distance2 = chromosome.get_total_distance()
        self.assertEqual(distance, distance2)
    
    def test_chromosome_get_elevation_gain(self):
        """Test elevation gain calculation"""
        chromosome = RouteChromosome([self.segment1, self.segment2])
        elevation_gain = chromosome.get_elevation_gain()
        
        expected_gain = self.segment1.elevation_gain + self.segment2.elevation_gain
        self.assertEqual(elevation_gain, expected_gain)
    
    def test_chromosome_get_elevation_loss(self):
        """Test elevation loss calculation"""
        chromosome = RouteChromosome([self.segment1, self.segment2])
        elevation_loss = chromosome.get_elevation_loss()
        
        expected_loss = self.segment1.elevation_loss + self.segment2.elevation_loss
        self.assertEqual(elevation_loss, expected_loss)
    
    def test_chromosome_get_net_elevation(self):
        """Test net elevation calculation"""
        chromosome = RouteChromosome([self.segment1, self.segment2])
        net_elevation = chromosome.get_net_elevation()
        
        expected_net = chromosome.get_elevation_gain() - chromosome.get_elevation_loss()
        self.assertEqual(net_elevation, expected_net)
    
    def test_chromosome_get_max_grade(self):
        """Test maximum grade calculation"""
        chromosome = RouteChromosome([self.segment1, self.segment2])
        max_grade = chromosome.get_max_grade()
        
        expected_max = max(self.segment1.max_grade, self.segment2.max_grade)
        self.assertEqual(max_grade, expected_max)
    
    def test_chromosome_get_max_grade_empty(self):
        """Test maximum grade calculation for empty chromosome"""
        chromosome = RouteChromosome()
        max_grade = chromosome.get_max_grade()
        
        self.assertEqual(max_grade, 0.0)
    
    def test_chromosome_get_route_nodes(self):
        """Test route nodes extraction"""
        chromosome = RouteChromosome([self.segment1, self.segment2])
        nodes = chromosome.get_route_nodes()
        
        expected_nodes = [1, 2, 3]  # Start of seg1, end of seg1, end of seg2
        self.assertEqual(nodes, expected_nodes)
    
    def test_chromosome_get_route_nodes_empty(self):
        """Test route nodes extraction for empty chromosome"""
        chromosome = RouteChromosome()
        nodes = chromosome.get_route_nodes()
        
        self.assertEqual(nodes, [])
    
    def test_chromosome_get_complete_path(self):
        """Test complete path extraction"""
        # Create segments with intermediate nodes
        segment1 = RouteSegment(1, 3, [1, 2, 3])
        segment2 = RouteSegment(3, 1, [3, 1])
        
        chromosome = RouteChromosome([segment1, segment2])
        path = chromosome.get_complete_path()
        
        # Should include nodes from the path (implementation may vary)
        self.assertIn(1, path)
        self.assertIn(2, path)
        # Note: Implementation might not include all nodes due to overlap handling
    
    def test_chromosome_calculate_diversity_score(self):
        """Test diversity score calculation"""
        # Set up segments with directions
        self.segment1.direction = 'N'
        self.segment2.direction = 'E'
        self.segment3.direction = 'S'
        
        chromosome = RouteChromosome([self.segment1, self.segment2, self.segment3])
        diversity = chromosome.calculate_diversity_score()
        
        self.assertGreater(diversity, 0.0)
        self.assertLessEqual(diversity, 1.0)
    
    def test_chromosome_calculate_diversity_score_empty(self):
        """Test diversity score calculation for empty chromosome"""
        chromosome = RouteChromosome()
        diversity = chromosome.calculate_diversity_score()
        
        self.assertEqual(diversity, 0.0)
    
    def test_chromosome_get_route_stats(self):
        """Test route statistics calculation"""
        chromosome = RouteChromosome([self.segment1, self.segment2])
        stats = chromosome.get_route_stats()
        
        required_keys = [
            'total_distance_km', 'total_distance_m', 'total_elevation_gain_m',
            'total_elevation_loss_m', 'net_elevation_gain_m', 'max_grade_percent',
            'segment_count', 'node_count', 'is_valid', 'is_circular',
            'diversity_score', 'estimated_time_min', 'unique_edges',
            'bidirectional_edges', 'usage_efficiency'
        ]
        
        for key in required_keys:
            self.assertIn(key, stats)
        
        self.assertEqual(stats['segment_count'], 2)
        self.assertEqual(stats['total_distance_km'], stats['total_distance_m'] / 1000)
    
    def test_chromosome_copy(self):
        """Test chromosome copy method"""
        chromosome = RouteChromosome([self.segment1, self.segment2])
        chromosome.fitness = 0.75
        chromosome.generation = 5
        
        chromosome_copy = chromosome.copy()
        
        self.assertEqual(len(chromosome.segments), len(chromosome_copy.segments))
        self.assertEqual(chromosome.fitness, chromosome_copy.fitness)
        self.assertEqual(chromosome.generation, chromosome_copy.generation)
        
        # Ensure it's a deep copy
        chromosome_copy.fitness = 0.99
        self.assertNotEqual(chromosome.fitness, chromosome_copy.fitness)
    
    def test_chromosome_to_route_result(self):
        """Test conversion to route result format"""
        chromosome = RouteChromosome([self.segment1, self.segment2, self.segment3])
        chromosome.fitness = 0.85
        chromosome.validate_connectivity()
        
        result = chromosome.to_route_result('maximize_elevation', 1.5)
        
        required_keys = ['route', 'cost', 'stats', 'solve_time', 'objective', 
                        'algorithm', 'target_distance_km', 'solver_info']
        
        for key in required_keys:
            self.assertIn(key, result)
        
        self.assertEqual(result['objective'], 'maximize_elevation')
        self.assertEqual(result['algorithm'], 'genetic')
        self.assertEqual(result['solve_time'], 1.5)
        self.assertIn('fitness', result['solver_info'])
    
    def test_chromosome_string_representation(self):
        """Test string representation"""
        chromosome = RouteChromosome([self.segment1])
        chromosome.fitness = 0.85
        
        str_repr = str(chromosome)
        
        self.assertIn('RouteChromosome', str_repr)
        self.assertIn('segments=1', str_repr)
        self.assertIn('fitness=0.850', str_repr)
        self.assertIn('valid=', str_repr)
    
    def test_chromosome_string_representation_no_fitness(self):
        """Test string representation with no fitness"""
        chromosome = RouteChromosome([self.segment1])
        
        str_repr = str(chromosome)
        
        self.assertIn('RouteChromosome', str_repr)
        self.assertIn('fitness=None', str_repr)
    
    def test_chromosome_cache_invalidation(self):
        """Test that cache is invalidated when segments change"""
        chromosome = RouteChromosome([self.segment1])
        
        # Calculate distance (should cache)
        distance1 = chromosome.get_total_distance()
        
        # Add segment
        chromosome.add_segment(self.segment2)
        
        # Distance should be recalculated
        distance2 = chromosome.get_total_distance()
        
        self.assertNotEqual(distance1, distance2)
        self.assertGreater(distance2, distance1)
    
    def test_chromosome_segment_usage_validation(self):
        """Test segment usage validation"""
        # Create a chromosome with repeated segments
        chromosome = RouteChromosome([self.segment1, self.segment1])  # Same segment twice
        
        # Should be invalid due to repeated segment usage
        is_valid = chromosome.validate_connectivity()
        self.assertFalse(is_valid)
        self.assertFalse(chromosome.is_valid)
    
    def test_chromosome_segment_usage_bidirectional(self):
        """Test segment usage with bidirectional travel"""
        # Create reverse segment
        reverse_segment = RouteSegment(2, 1, [2, 1])
        reverse_segment.calculate_properties(self.mock_graph)
        
        # Should be valid - same edge but different directions
        chromosome = RouteChromosome([self.segment1, reverse_segment])
        is_valid = chromosome.validate_connectivity()
        
        self.assertTrue(is_valid)
        self.assertTrue(chromosome.is_valid)
    
    def test_chromosome_segment_usage_stats(self):
        """Test segment usage statistics"""
        # Create reverse segment
        reverse_segment = RouteSegment(2, 1, [2, 1])
        reverse_segment.calculate_properties(self.mock_graph)
        
        chromosome = RouteChromosome([self.segment1, reverse_segment])
        chromosome.validate_connectivity()
        
        usage_stats = chromosome.get_segment_usage_stats()
        
        self.assertIn('total_unique_edges', usage_stats)
        self.assertIn('bidirectional_edges', usage_stats)
        self.assertIn('usage_efficiency', usage_stats)
        self.assertIn('edge_usage_details', usage_stats)
        
        # Should have 1 unique edge used in both directions
        self.assertEqual(usage_stats['total_unique_edges'], 1)
        self.assertEqual(usage_stats['bidirectional_edges'], 1)
        self.assertEqual(usage_stats['usage_efficiency'], 0.0)  # (1-1)/1 = 0
    
    def test_chromosome_segment_usage_limit_exceeded(self):
        """Test segment usage limit exceeded"""
        # Create three segments using the same edge in the same direction
        segment1_dup = RouteSegment(1, 2, [1, 2])
        segment1_dup.calculate_properties(self.mock_graph)
        
        chromosome = RouteChromosome([self.segment1, segment1_dup])
        is_valid = chromosome.validate_connectivity()
        
        # Should be invalid - same edge used twice in same direction
        self.assertFalse(is_valid)
        self.assertFalse(chromosome.is_valid)
    
    def test_chromosome_segment_usage_complex_route(self):
        """Test segment usage with complex route"""
        # Create a route that uses multiple edges
        chromosome = RouteChromosome([self.segment1, self.segment2, self.segment3])
        is_valid = chromosome.validate_connectivity()
        
        # Should be valid - all different edges
        self.assertTrue(is_valid)
        self.assertTrue(chromosome.is_valid)
        
        usage_stats = chromosome.get_segment_usage_stats()
        self.assertEqual(usage_stats['total_unique_edges'], 3)
        self.assertEqual(usage_stats['bidirectional_edges'], 0)
        self.assertEqual(usage_stats['usage_efficiency'], 1.0)  # (3-0)/3 = 1.0


if __name__ == '__main__':
    unittest.main()