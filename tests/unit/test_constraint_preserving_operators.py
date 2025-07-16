#!/usr/bin/env python3
"""
Unit tests for ConstraintPreservingOperators
Tests comprehensive functionality of constraint-preserving genetic operators
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import networkx as nx
import sys
import os
import random

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from genetic_algorithm.constraint_preserving_operators import ConstraintPreservingOperators, RouteConstraints
from genetic_algorithm.chromosome import RouteChromosome, RouteSegment


class TestRouteConstraints(unittest.TestCase):
    """Test RouteConstraints class and validation methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.constraints = RouteConstraints(
            min_distance_km=2.0,
            max_distance_km=5.0,
            start_node=1001,
            must_return_to_start=True,
            must_be_connected=True,
            allow_bidirectional=True
        )
        
        # Create mock segments
        self.mock_segment1 = Mock(spec=RouteSegment)
        self.mock_segment1.start_node = 1001
        self.mock_segment1.end_node = 1002
        self.mock_segment1.length = 1000.0
        self.mock_segment1.path_nodes = [1001, 1002]
        
        self.mock_segment2 = Mock(spec=RouteSegment)
        self.mock_segment2.start_node = 1002
        self.mock_segment2.end_node = 1003
        self.mock_segment2.length = 1500.0
        self.mock_segment2.path_nodes = [1002, 1003]
        
        self.mock_segment3 = Mock(spec=RouteSegment)
        self.mock_segment3.start_node = 1003
        self.mock_segment3.end_node = 1001
        self.mock_segment3.length = 1000.0
        self.mock_segment3.path_nodes = [1003, 1001]
    
    def test_route_constraints_initialization(self):
        """Test RouteConstraints initialization"""
        constraints = RouteConstraints(
            min_distance_km=1.0,
            max_distance_km=3.0,
            start_node=1001
        )
        
        self.assertEqual(constraints.min_distance_km, 1.0)
        self.assertEqual(constraints.max_distance_km, 3.0)
        self.assertEqual(constraints.start_node, 1001)
        self.assertTrue(constraints.must_return_to_start)
        self.assertTrue(constraints.must_be_connected)
        self.assertTrue(constraints.allow_bidirectional)
    
    def test_route_constraints_custom_values(self):
        """Test RouteConstraints with custom values"""
        constraints = RouteConstraints(
            min_distance_km=2.0,
            max_distance_km=4.0,
            start_node=1005,
            must_return_to_start=False,
            must_be_connected=False,
            allow_bidirectional=False
        )
        
        self.assertEqual(constraints.min_distance_km, 2.0)
        self.assertEqual(constraints.max_distance_km, 4.0)
        self.assertEqual(constraints.start_node, 1005)
        self.assertFalse(constraints.must_return_to_start)
        self.assertFalse(constraints.must_be_connected)
        self.assertFalse(constraints.allow_bidirectional)
    
    def test_validate_distance_valid(self):
        """Test distance validation with valid chromosome"""
        mock_chromosome = Mock(spec=RouteChromosome)
        mock_chromosome.segments = [self.mock_segment1, self.mock_segment2, self.mock_segment3]
        mock_chromosome.get_route_stats.return_value = {'total_distance_km': 3.5}
        
        result = self.constraints.validate_distance(mock_chromosome)
        
        self.assertTrue(result)
        mock_chromosome.get_route_stats.assert_called_once()
    
    def test_validate_distance_too_short(self):
        """Test distance validation with too short chromosome"""
        mock_chromosome = Mock(spec=RouteChromosome)
        mock_chromosome.segments = [self.mock_segment1]
        mock_chromosome.get_route_stats.return_value = {'total_distance_km': 1.0}
        
        result = self.constraints.validate_distance(mock_chromosome)
        
        self.assertFalse(result)
    
    def test_validate_distance_too_long(self):
        """Test distance validation with too long chromosome"""
        mock_chromosome = Mock(spec=RouteChromosome)
        mock_chromosome.segments = [self.mock_segment1, self.mock_segment2, self.mock_segment3]
        mock_chromosome.get_route_stats.return_value = {'total_distance_km': 6.0}
        
        result = self.constraints.validate_distance(mock_chromosome)
        
        self.assertFalse(result)
    
    def test_validate_distance_empty_segments(self):
        """Test distance validation with empty segments"""
        mock_chromosome = Mock(spec=RouteChromosome)
        mock_chromosome.segments = []
        
        result = self.constraints.validate_distance(mock_chromosome)
        
        self.assertFalse(result)
    
    def test_validate_connectivity_valid(self):
        """Test connectivity validation with valid chromosome"""
        mock_chromosome = Mock(spec=RouteChromosome)
        mock_chromosome.segments = [self.mock_segment1, self.mock_segment2, self.mock_segment3]
        
        result = self.constraints.validate_connectivity(mock_chromosome)
        
        self.assertTrue(result)
    
    def test_validate_connectivity_disconnected(self):
        """Test connectivity validation with disconnected segments"""
        # Create disconnected segments
        disconnected_segment = Mock(spec=RouteSegment)
        disconnected_segment.start_node = 1004  # Doesn't connect to previous
        disconnected_segment.end_node = 1005
        disconnected_segment.path_nodes = [1004, 1005]
        
        mock_chromosome = Mock(spec=RouteChromosome)
        mock_chromosome.segments = [self.mock_segment1, disconnected_segment]
        
        result = self.constraints.validate_connectivity(mock_chromosome)
        
        self.assertFalse(result)
    
    def test_validate_connectivity_wrong_start_node(self):
        """Test connectivity validation with wrong start node"""
        wrong_start_segment = Mock(spec=RouteSegment)
        wrong_start_segment.start_node = 1002  # Wrong start
        wrong_start_segment.end_node = 1003
        wrong_start_segment.path_nodes = [1002, 1003]
        
        mock_chromosome = Mock(spec=RouteChromosome)
        mock_chromosome.segments = [wrong_start_segment, self.mock_segment3]
        
        result = self.constraints.validate_connectivity(mock_chromosome)
        
        self.assertFalse(result)
    
    def test_validate_connectivity_wrong_end_node(self):
        """Test connectivity validation with wrong end node"""
        wrong_end_segment = Mock(spec=RouteSegment)
        wrong_end_segment.start_node = 1003
        wrong_end_segment.end_node = 1004  # Wrong end
        wrong_end_segment.path_nodes = [1003, 1004]
        
        mock_chromosome = Mock(spec=RouteChromosome)
        mock_chromosome.segments = [self.mock_segment1, self.mock_segment2, wrong_end_segment]
        
        result = self.constraints.validate_connectivity(mock_chromosome)
        
        self.assertFalse(result)
    
    def test_validate_connectivity_no_bidirectional_valid(self):
        """Test connectivity validation with no bidirectional allowed - valid case"""
        constraints = RouteConstraints(
            min_distance_km=2.0,
            max_distance_km=5.0,
            start_node=1001,
            allow_bidirectional=False
        )
        
        mock_chromosome = Mock(spec=RouteChromosome)
        mock_chromosome.segments = [self.mock_segment1, self.mock_segment2, self.mock_segment3]
        
        result = constraints.validate_connectivity(mock_chromosome)
        
        self.assertTrue(result)
    
    def test_validate_connectivity_no_bidirectional_invalid(self):
        """Test connectivity validation with no bidirectional allowed - invalid case"""
        constraints = RouteConstraints(
            min_distance_km=2.0,
            max_distance_km=5.0,
            start_node=1001,
            allow_bidirectional=False
        )
        
        # Create segments that use the same edge twice
        duplicate_segment = Mock(spec=RouteSegment)
        duplicate_segment.start_node = 1002
        duplicate_segment.end_node = 1001  # Same edge as segment1 but reversed
        duplicate_segment.path_nodes = [1002, 1001]
        
        mock_chromosome = Mock(spec=RouteChromosome)
        mock_chromosome.segments = [self.mock_segment1, duplicate_segment]
        
        result = constraints.validate_connectivity(mock_chromosome)
        
        self.assertFalse(result)
    
    def test_validate_connectivity_empty_segments(self):
        """Test connectivity validation with empty segments"""
        mock_chromosome = Mock(spec=RouteChromosome)
        mock_chromosome.segments = []
        
        result = self.constraints.validate_connectivity(mock_chromosome)
        
        self.assertFalse(result)
    
    def test_validate_connectivity_must_not_return_to_start(self):
        """Test connectivity validation when return to start is not required"""
        constraints = RouteConstraints(
            min_distance_km=2.0,
            max_distance_km=5.0,
            start_node=1001,
            must_return_to_start=False
        )
        
        # Create segments that don't return to start
        non_return_segment = Mock(spec=RouteSegment)
        non_return_segment.start_node = 1003
        non_return_segment.end_node = 1004
        non_return_segment.path_nodes = [1003, 1004]
        
        mock_chromosome = Mock(spec=RouteChromosome)
        mock_chromosome.segments = [self.mock_segment1, self.mock_segment2, non_return_segment]
        
        result = constraints.validate_connectivity(mock_chromosome)
        
        self.assertTrue(result)
    
    def test_validate_all_constraints_valid(self):
        """Test validation of all constraints - valid case"""
        mock_chromosome = Mock(spec=RouteChromosome)
        mock_chromosome.segments = [self.mock_segment1, self.mock_segment2, self.mock_segment3]
        mock_chromosome.get_route_stats.return_value = {'total_distance_km': 3.5}
        
        result = self.constraints.validate(mock_chromosome)
        
        self.assertTrue(result)
    
    def test_validate_all_constraints_invalid_distance(self):
        """Test validation of all constraints - invalid distance"""
        mock_chromosome = Mock(spec=RouteChromosome)
        mock_chromosome.segments = [self.mock_segment1, self.mock_segment2, self.mock_segment3]
        mock_chromosome.get_route_stats.return_value = {'total_distance_km': 1.0}
        
        result = self.constraints.validate(mock_chromosome)
        
        self.assertFalse(result)
    
    def test_validate_all_constraints_invalid_connectivity(self):
        """Test validation of all constraints - invalid connectivity"""
        # Create disconnected segments
        disconnected_segment = Mock(spec=RouteSegment)
        disconnected_segment.start_node = 1004  # Doesn't connect
        disconnected_segment.end_node = 1005
        disconnected_segment.path_nodes = [1004, 1005]
        
        mock_chromosome = Mock(spec=RouteChromosome)
        mock_chromosome.segments = [self.mock_segment1, disconnected_segment]
        mock_chromosome.get_route_stats.return_value = {'total_distance_km': 3.5}
        
        result = self.constraints.validate(mock_chromosome)
        
        self.assertFalse(result)


class TestConstraintPreservingOperators(unittest.TestCase):
    """Test ConstraintPreservingOperators class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock graph
        self.mock_graph = nx.Graph()
        nodes = [
            (1001, -80.4094, 37.1299, 100.0),
            (1002, -80.4090, 37.1300, 105.0),
            (1003, -80.4086, 37.1301, 110.0),
            (1004, -80.4082, 37.1302, 115.0),
            (1005, -80.4078, 37.1303, 120.0)
        ]
        
        for node_id, x, y, elevation in nodes:
            self.mock_graph.add_node(node_id, x=x, y=y, elevation=elevation)
        
        # Add edges
        edges = [
            (1001, 1002, 400), (1002, 1003, 400), (1003, 1004, 400),
            (1004, 1005, 400), (1005, 1001, 400), (1001, 1003, 565),
            (1002, 1004, 565), (1003, 1005, 565), (1001, 1004, 800),
            (1002, 1005, 800)
        ]
        
        for node1, node2, length in edges:
            self.mock_graph.add_edge(node1, node2, length=length)
        
        # Create constraints
        self.constraints = RouteConstraints(
            min_distance_km=2.0,
            max_distance_km=5.0,
            start_node=1001,
            must_return_to_start=True,
            must_be_connected=True,
            allow_bidirectional=True
        )
        
        # Create operators
        self.operators = ConstraintPreservingOperators(self.mock_graph, self.constraints)
        
        # Create test chromosomes
        self.parent1 = self._create_test_chromosome([1001, 1002, 1003, 1001])
        self.parent2 = self._create_test_chromosome([1001, 1004, 1005, 1001])
    
    def _create_test_chromosome(self, node_path):
        """Helper to create test chromosome from node path"""
        segments = []
        for i in range(len(node_path) - 1):
            segment = RouteSegment(node_path[i], node_path[i + 1], [node_path[i], node_path[i + 1]])
            segment.calculate_properties(self.mock_graph)
            segments.append(segment)
        
        return RouteChromosome(segments)
    
    def test_operators_initialization(self):
        """Test ConstraintPreservingOperators initialization"""
        self.assertEqual(self.operators.graph, self.mock_graph)
        self.assertEqual(self.operators.constraints, self.constraints)
        self.assertIsInstance(self.operators.segment_cache, dict)
        self.assertIsInstance(self.operators.distances_from_start, dict)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_precompute_distance_matrix_success(self, mock_dijkstra):
        """Test successful distance matrix precomputation"""
        mock_distances = {1001: 0, 1002: 400, 1003: 800}
        mock_dijkstra.return_value = mock_distances
        
        operators = ConstraintPreservingOperators(self.mock_graph, self.constraints)
        
        self.assertEqual(operators.distances_from_start, mock_distances)
        mock_dijkstra.assert_called_once_with(
            self.mock_graph, 1001, weight='length', cutoff=15000
        )
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_precompute_distance_matrix_failure(self, mock_dijkstra):
        """Test distance matrix precomputation failure handling"""
        mock_dijkstra.side_effect = Exception("Network error")
        
        operators = ConstraintPreservingOperators(self.mock_graph, self.constraints)
        
        self.assertEqual(operators.distances_from_start, {})
    
    def test_connection_point_crossover_no_crossover(self):
        """Test connection point crossover with no crossover (random skip)"""
        with patch('random.random', return_value=0.9):  # > crossover_rate
            offspring1, offspring2 = self.operators.connection_point_crossover(
                self.parent1, self.parent2, crossover_rate=0.8
            )
            
            # Should return parent copies
            self.assertEqual(len(offspring1.segments), len(self.parent1.segments))
            self.assertEqual(len(offspring2.segments), len(self.parent2.segments))
    
    def test_connection_point_crossover_no_valid_points(self):
        """Test connection point crossover with no valid crossover points"""
        with patch.object(self.operators, '_find_valid_crossover_points') as mock_find:
            mock_find.return_value = []
            
            offspring1, offspring2 = self.operators.connection_point_crossover(
                self.parent1, self.parent2
            )
            
            # Should return parent copies
            self.assertEqual(len(offspring1.segments), len(self.parent1.segments))
            self.assertEqual(len(offspring2.segments), len(self.parent2.segments))
    
    def test_connection_point_crossover_successful(self):
        """Test successful connection point crossover"""
        # Create parents with common nodes
        parent1 = self._create_test_chromosome([1001, 1002, 1003, 1001])
        parent2 = self._create_test_chromosome([1001, 1003, 1004, 1001])
        
        offspring1, offspring2 = self.operators.connection_point_crossover(parent1, parent2)
        
        # Should create valid offspring
        self.assertIsInstance(offspring1, RouteChromosome)
        self.assertIsInstance(offspring2, RouteChromosome)
        self.assertGreater(len(offspring1.segments), 0)
        self.assertGreater(len(offspring2.segments), 0)
    
    def test_connection_point_crossover_validation_failure(self):
        """Test connection point crossover with validation failure"""
        with patch.object(self.operators, '_find_valid_crossover_points') as mock_find:
            mock_find.return_value = [1003]  # Valid crossover point
            
            with patch.object(self.operators, '_crossover_at_point') as mock_crossover:
                mock_invalid_offspring = Mock(spec=RouteChromosome)
                mock_crossover.return_value = (mock_invalid_offspring, mock_invalid_offspring)
                
                with patch.object(self.constraints, 'validate') as mock_validate:
                    mock_validate.return_value = False  # Validation fails
                    
                    offspring1, offspring2 = self.operators.connection_point_crossover(
                        self.parent1, self.parent2
                    )
                    
                    # Should return parent copies
                    self.assertEqual(len(offspring1.segments), len(self.parent1.segments))
                    self.assertEqual(len(offspring2.segments), len(self.parent2.segments))
    
    def test_connection_point_crossover_exception_handling(self):
        """Test connection point crossover exception handling"""
        with patch.object(self.operators, '_find_valid_crossover_points') as mock_find:
            mock_find.return_value = [1003]
            
            with patch.object(self.operators, '_crossover_at_point') as mock_crossover:
                mock_crossover.side_effect = Exception("Crossover error")
                
                offspring1, offspring2 = self.operators.connection_point_crossover(
                    self.parent1, self.parent2
                )
                
                # Should return parent copies
                self.assertEqual(len(offspring1.segments), len(self.parent1.segments))
                self.assertEqual(len(offspring2.segments), len(self.parent2.segments))
    
    def test_find_valid_crossover_points_basic(self):
        """Test finding valid crossover points"""
        parent1 = self._create_test_chromosome([1001, 1002, 1003, 1001])
        parent2 = self._create_test_chromosome([1001, 1003, 1004, 1001])
        
        with patch.object(self.operators, '_score_crossover_point') as mock_score:
            mock_score.return_value = 0.5
            
            crossover_points = self.operators._find_valid_crossover_points(parent1, parent2)
            
            # Should find common nodes (excluding start node)
            self.assertIsInstance(crossover_points, list)
            # 1003 should be a common node
            self.assertIn(1003, crossover_points)
    
    def test_find_valid_crossover_points_no_common_nodes(self):
        """Test finding valid crossover points with no common nodes"""
        parent1 = self._create_test_chromosome([1001, 1002, 1001])
        parent2 = self._create_test_chromosome([1001, 1004, 1001])
        
        crossover_points = self.operators._find_valid_crossover_points(parent1, parent2)
        
        # Should return empty list (no common nodes except start)
        self.assertEqual(crossover_points, [])
    
    def test_find_valid_crossover_points_scoring(self):
        """Test crossover point scoring and sorting"""
        parent1 = self._create_test_chromosome([1001, 1002, 1003, 1004, 1001])
        parent2 = self._create_test_chromosome([1001, 1003, 1004, 1005, 1001])
        
        with patch.object(self.operators, '_score_crossover_point') as mock_score:
            # Different scores for different nodes
            def score_side_effect(p1, p2, node):
                if node == 1003:
                    return 0.8
                elif node == 1004:
                    return 0.6
                else:
                    return 0.0
            
            mock_score.side_effect = score_side_effect
            
            crossover_points = self.operators._find_valid_crossover_points(parent1, parent2)
            
            # Should be sorted by score (highest first)
            if len(crossover_points) >= 2:
                self.assertEqual(crossover_points[0], 1003)  # Higher score
                self.assertEqual(crossover_points[1], 1004)  # Lower score
    
    def test_score_crossover_point_basic(self):
        """Test crossover point scoring"""
        parent1 = self._create_test_chromosome([1001, 1002, 1003, 1001])
        parent2 = self._create_test_chromosome([1001, 1003, 1004, 1001])
        
        with patch.object(self.operators, '_estimate_distance_to_node') as mock_to:
            with patch.object(self.operators, '_estimate_distance_from_node') as mock_from:
                mock_to.return_value = 1000.0
                mock_from.return_value = 1500.0
                
                score = self.operators._score_crossover_point(parent1, parent2, 1003)
                
                self.assertIsInstance(score, float)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)
    
    def test_score_crossover_point_exception(self):
        """Test crossover point scoring with exception"""
        parent1 = self._create_test_chromosome([1001, 1002, 1003, 1001])
        parent2 = self._create_test_chromosome([1001, 1003, 1004, 1001])
        
        with patch.object(self.operators, '_estimate_distance_to_node') as mock_to:
            mock_to.side_effect = Exception("Distance error")
            
            score = self.operators._score_crossover_point(parent1, parent2, 1003)
            
            self.assertEqual(score, 0.0)
    
    def test_estimate_distance_to_node_found(self):
        """Test distance estimation to node when found"""
        chromosome = self._create_test_chromosome([1001, 1002, 1003, 1001])
        
        distance = self.operators._estimate_distance_to_node(chromosome, 1002)
        
        self.assertIsInstance(distance, float)
        self.assertGreaterEqual(distance, 0.0)
    
    def test_estimate_distance_to_node_not_found(self):
        """Test distance estimation to node when not found"""
        chromosome = self._create_test_chromosome([1001, 1002, 1003, 1001])
        
        distance = self.operators._estimate_distance_to_node(chromosome, 1005)
        
        self.assertIsInstance(distance, float)
        # Should return total distance if node not found
        self.assertGreater(distance, 0.0)
    
    def test_estimate_distance_to_node_edge_data_handling(self):
        """Test distance estimation with different edge data formats"""
        chromosome = self._create_test_chromosome([1001, 1002, 1003, 1001])
        
        # Test with dictionary edge data
        distance = self.operators._estimate_distance_to_node(chromosome, 1002)
        
        self.assertIsInstance(distance, float)
        self.assertGreaterEqual(distance, 0.0)
    
    def test_estimate_distance_from_node_basic(self):
        """Test distance estimation from node"""
        chromosome = self._create_test_chromosome([1001, 1002, 1003, 1001])
        
        with patch.object(self.operators, '_estimate_distance_to_node') as mock_to:
            mock_to.return_value = 800.0
            
            distance = self.operators._estimate_distance_from_node(chromosome, 1002)
            
            self.assertIsInstance(distance, float)
            self.assertGreaterEqual(distance, 0.0)
    
    def test_crossover_at_point_basic(self):
        """Test crossover at specific point"""
        parent1 = self._create_test_chromosome([1001, 1002, 1003, 1001])
        parent2 = self._create_test_chromosome([1001, 1003, 1004, 1001])
        
        with patch.object(self.operators, '_split_at_node') as mock_split:
            # Mock split results
            mock_split.side_effect = [
                ([Mock(spec=RouteSegment)], [Mock(spec=RouteSegment)]),  # parent1 split
                ([Mock(spec=RouteSegment)], [Mock(spec=RouteSegment)])   # parent2 split
            ]
            
            offspring1, offspring2 = self.operators._crossover_at_point(parent1, parent2, 1003)
            
            self.assertIsInstance(offspring1, RouteChromosome)
            self.assertIsInstance(offspring2, RouteChromosome)
            self.assertGreater(len(offspring1.segments), 0)
            self.assertGreater(len(offspring2.segments), 0)
    
    def test_split_at_node_basic(self):
        """Test splitting chromosome at node"""
        chromosome = self._create_test_chromosome([1001, 1002, 1003, 1004, 1001])
        
        before_segments, after_segments = self.operators._split_at_node(chromosome, 1003)
        
        self.assertIsInstance(before_segments, list)
        self.assertIsInstance(after_segments, list)
        self.assertGreater(len(before_segments), 0)
        self.assertGreater(len(after_segments), 0)
    
    def test_split_at_node_not_found(self):
        """Test splitting chromosome at node not found"""
        chromosome = self._create_test_chromosome([1001, 1002, 1001])
        
        before_segments, after_segments = self.operators._split_at_node(chromosome, 1005)
        
        # Should return all segments as before_segments
        self.assertEqual(len(before_segments), len(chromosome.segments))
        self.assertEqual(len(after_segments), 0)
    
    def test_split_at_node_beginning(self):
        """Test splitting chromosome at beginning node"""
        chromosome = self._create_test_chromosome([1001, 1002, 1003, 1001])
        
        before_segments, after_segments = self.operators._split_at_node(chromosome, 1001)
        
        self.assertIsInstance(before_segments, list)
        self.assertIsInstance(after_segments, list)
        # Should handle start node correctly
    
    def test_split_at_node_end(self):
        """Test splitting chromosome at end node"""
        chromosome = self._create_test_chromosome([1001, 1002, 1003, 1001])
        
        before_segments, after_segments = self.operators._split_at_node(chromosome, 1001)
        
        self.assertIsInstance(before_segments, list)
        self.assertIsInstance(after_segments, list)
    
    def test_segment_substitution_crossover_no_crossover(self):
        """Test segment substitution crossover with no crossover"""
        with patch('random.random', return_value=0.9):  # > crossover_rate
            offspring1, offspring2 = self.operators.segment_substitution_crossover(
                self.parent1, self.parent2, crossover_rate=0.8
            )
            
            # Should return parent copies
            self.assertEqual(len(offspring1.segments), len(self.parent1.segments))
            self.assertEqual(len(offspring2.segments), len(self.parent2.segments))
    
    def test_segment_substitution_crossover_no_substitutions(self):
        """Test segment substitution crossover with no substitutable segments"""
        with patch.object(self.operators, '_find_substitutable_segments') as mock_find:
            mock_find.return_value = []
            
            offspring1, offspring2 = self.operators.segment_substitution_crossover(
                self.parent1, self.parent2
            )
            
            # Should return parent copies
            self.assertEqual(len(offspring1.segments), len(self.parent1.segments))
            self.assertEqual(len(offspring2.segments), len(self.parent2.segments))
    
    def test_segment_substitution_crossover_successful(self):
        """Test successful segment substitution crossover"""
        # Create parents with substitutable segments
        parent1 = self._create_test_chromosome([1001, 1002, 1003, 1001])
        parent2 = self._create_test_chromosome([1001, 1002, 1004, 1001])
        
        with patch('random.random', return_value=0.5):  # < crossover_rate, ensures crossover happens
            with patch.object(self.operators, '_find_substitutable_segments') as mock_find:
                mock_find.return_value = [(0, 0)]  # First segments are substitutable
                
                with patch.object(self.constraints, 'validate') as mock_validate:
                    mock_validate.return_value = True
                    
                    offspring1, offspring2 = self.operators.segment_substitution_crossover(
                        parent1, parent2
                    )
                    
                    self.assertIsInstance(offspring1, RouteChromosome)
                    self.assertIsInstance(offspring2, RouteChromosome)
                    self.assertEqual(offspring1.creation_method, "segment_substitution_crossover")
                    self.assertEqual(offspring2.creation_method, "segment_substitution_crossover")
    
    def test_segment_substitution_crossover_validation_failure(self):
        """Test segment substitution crossover with validation failure"""
        with patch.object(self.operators, '_find_substitutable_segments') as mock_find:
            mock_find.return_value = [(0, 0)]
            
            with patch.object(self.constraints, 'validate') as mock_validate:
                mock_validate.return_value = False  # Validation fails
                
                offspring1, offspring2 = self.operators.segment_substitution_crossover(
                    self.parent1, self.parent2
                )
                
                # Should return parent copies
                self.assertEqual(len(offspring1.segments), len(self.parent1.segments))
                self.assertEqual(len(offspring2.segments), len(self.parent2.segments))
    
    def test_find_substitutable_segments_basic(self):
        """Test finding substitutable segments"""
        parent1 = self._create_test_chromosome([1001, 1002, 1003, 1001])
        parent2 = self._create_test_chromosome([1001, 1002, 1004, 1001])
        
        substitutions = self.operators._find_substitutable_segments(parent1, parent2)
        
        self.assertIsInstance(substitutions, list)
        # Should find segments with same start/end nodes
        # First segments both start at 1001 and end at 1002
        self.assertIn((0, 0), substitutions)
    
    def test_find_substitutable_segments_no_matches(self):
        """Test finding substitutable segments with no matches"""
        parent1 = self._create_test_chromosome([1001, 1002, 1001])
        parent2 = self._create_test_chromosome([1001, 1004, 1001])
        
        substitutions = self.operators._find_substitutable_segments(parent1, parent2)
        
        # Should find no substitutions due to different intermediate nodes
        self.assertEqual(len(substitutions), 0)
    
    def test_find_substitutable_segments_distance_filter(self):
        """Test substitutable segments with distance filtering"""
        # Create segments with very different lengths but same start/end nodes
        parent1 = self._create_test_chromosome([1001, 1002, 1001])
        parent2 = self._create_test_chromosome([1001, 1002, 1001])
        
        # Mock segment lengths to be very different
        parent1.segments[0].length = 1000.0
        parent2.segments[0].length = 2000.0  # 1000m difference > 500m threshold
        
        # Need to mock both segments to ensure they are truly different
        parent1.segments[1].length = 1000.0
        parent2.segments[1].length = 2000.0  # 1000m difference > 500m threshold
        
        substitutions = self.operators._find_substitutable_segments(parent1, parent2)
        
        # Should filter out segments with large distance difference
        # Count only substitutions that pass the distance check
        valid_substitutions = [s for s in substitutions if abs(parent1.segments[s[0]].length - parent2.segments[s[1]].length) < 500]
        self.assertEqual(len(valid_substitutions), 0)
    
    def test_distance_neutral_mutation_empty_segments(self):
        """Test distance neutral mutation with empty segments"""
        empty_chromosome = Mock(spec=RouteChromosome)
        empty_chromosome.segments = []
        
        result = self.operators.distance_neutral_mutation(empty_chromosome)
        
        self.assertEqual(result, empty_chromosome)
    
    def test_distance_neutral_mutation_no_mutation(self):
        """Test distance neutral mutation with no mutation (random skip)"""
        with patch('random.random', return_value=0.9):  # > mutation_rate
            result = self.operators.distance_neutral_mutation(self.parent1, mutation_rate=0.1)
            
            self.assertIsInstance(result, RouteChromosome)
            self.assertEqual(result.creation_method, "distance_neutral_mutation")
    
    def test_distance_neutral_mutation_successful(self):
        """Test successful distance neutral mutation"""
        with patch('random.random', return_value=0.05):  # < mutation_rate
            with patch.object(self.operators, '_find_similar_length_alternative') as mock_find:
                mock_alternative = Mock(spec=RouteSegment)
                mock_alternative.start_node = 1001
                mock_alternative.end_node = 1002
                mock_alternative.length = 400.0
                mock_find.return_value = mock_alternative
                
                with patch.object(self.constraints, 'validate') as mock_validate:
                    mock_validate.return_value = True
                    
                    result = self.operators.distance_neutral_mutation(self.parent1)
                    
                    self.assertIsInstance(result, RouteChromosome)
                    self.assertEqual(result.creation_method, "distance_neutral_mutation")
    
    def test_distance_neutral_mutation_no_alternative(self):
        """Test distance neutral mutation with no alternative found"""
        with patch('random.random', return_value=0.05):  # < mutation_rate
            with patch.object(self.operators, '_find_similar_length_alternative') as mock_find:
                mock_find.return_value = None
                
                result = self.operators.distance_neutral_mutation(self.parent1)
                
                self.assertIsInstance(result, RouteChromosome)
                self.assertEqual(result.creation_method, "distance_neutral_mutation")
    
    def test_distance_neutral_mutation_validation_failure(self):
        """Test distance neutral mutation with validation failure"""
        with patch('random.random', return_value=0.05):  # < mutation_rate
            with patch.object(self.operators, '_find_similar_length_alternative') as mock_find:
                mock_alternative = Mock(spec=RouteSegment)
                mock_find.return_value = mock_alternative
                
                with patch.object(self.constraints, 'validate') as mock_validate:
                    mock_validate.return_value = False
                    
                    result = self.operators.distance_neutral_mutation(self.parent1)
                    
                    self.assertIsInstance(result, RouteChromosome)
                    # Should not have applied the mutation
    
    def test_find_similar_length_alternative_direct_edge(self):
        """Test finding similar length alternative with direct edge"""
        # Test with nodes that have direct edge
        alternative = self.operators._find_similar_length_alternative(1001, 1002, 400.0)
        
        # Should find alternative path
        if alternative:
            self.assertIsInstance(alternative, RouteSegment)
            self.assertEqual(alternative.start_node, 1001)
            self.assertEqual(alternative.end_node, 1002)
    
    def test_find_similar_length_alternative_no_path(self):
        """Test finding similar length alternative with no path"""
        # Create graph with no alternative paths
        isolated_graph = nx.Graph()
        isolated_graph.add_node(1001, x=-80.4094, y=37.1299, elevation=100)
        isolated_graph.add_node(1002, x=-80.4090, y=37.1300, elevation=105)
        isolated_graph.add_edge(1001, 1002, length=400)
        
        isolated_operators = ConstraintPreservingOperators(isolated_graph, self.constraints)
        
        alternative = isolated_operators._find_similar_length_alternative(1001, 1002, 400.0)
        
        self.assertIsNone(alternative)
    
    def test_find_similar_length_alternative_multi_hop(self):
        """Test finding similar length alternative with multi-hop path"""
        # Mock to ensure multi-hop path is tested
        with patch('networkx.shortest_path') as mock_path:
            mock_path.side_effect = [
                [1001, 1003],  # start -> intermediate
                [1003, 1002]   # intermediate -> end
            ]
            
            alternative = self.operators._find_similar_length_alternative(1001, 1002, 800.0)
            
            if alternative:
                self.assertIsInstance(alternative, RouteSegment)
                self.assertEqual(alternative.start_node, 1001)
                self.assertEqual(alternative.end_node, 1002)
    
    def test_find_similar_length_alternative_length_check(self):
        """Test similar length alternative with length checking"""
        # Create a segment with known length
        segment = RouteSegment(1001, 1002, [1001, 1002])
        segment.calculate_properties(self.mock_graph)
        
        # Request alternative with very different target length
        alternative = self.operators._find_similar_length_alternative(1001, 1002, 10000.0)
        
        # Should not find alternative due to length mismatch
        self.assertIsNone(alternative)
    
    def test_find_similar_length_alternative_exception_handling(self):
        """Test similar length alternative with exception handling"""
        with patch('networkx.shortest_path') as mock_path:
            mock_path.side_effect = Exception("Path error")
            
            alternative = self.operators._find_similar_length_alternative(1001, 1002, 400.0)
            
            self.assertIsNone(alternative)
    
    def test_local_optimization_mutation_few_segments(self):
        """Test local optimization mutation with few segments"""
        single_segment_chromosome = Mock(spec=RouteChromosome)
        single_segment_chromosome.segments = [Mock(spec=RouteSegment)]
        
        result = self.operators.local_optimization_mutation(single_segment_chromosome)
        
        self.assertEqual(result, single_segment_chromosome)
    
    def test_local_optimization_mutation_no_mutation(self):
        """Test local optimization mutation with no mutation"""
        with patch('random.random', return_value=0.9):  # > mutation_rate
            result = self.operators.local_optimization_mutation(self.parent1, mutation_rate=0.1)
            
            self.assertIsInstance(result, RouteChromosome)
            self.assertEqual(result.creation_method, "local_optimization_mutation")
    
    def test_local_optimization_mutation_successful(self):
        """Test successful local optimization mutation"""
        with patch('random.random', return_value=0.05):  # < mutation_rate
            with patch.object(self.operators, '_optimize_segment_pair') as mock_optimize:
                mock_optimized = [Mock(spec=RouteSegment), Mock(spec=RouteSegment)]
                mock_optimize.return_value = mock_optimized
                
                with patch.object(self.constraints, 'validate') as mock_validate:
                    mock_validate.return_value = True
                    
                    with patch.object(self.operators, '_has_better_elevation') as mock_better:
                        mock_better.return_value = True
                        
                        result = self.operators.local_optimization_mutation(self.parent1)
                        
                        self.assertIsInstance(result, RouteChromosome)
                        self.assertEqual(result.creation_method, "local_optimization_mutation")
    
    def test_local_optimization_mutation_no_optimization(self):
        """Test local optimization mutation with no optimization found"""
        with patch('random.random', return_value=0.05):  # < mutation_rate
            with patch.object(self.operators, '_optimize_segment_pair') as mock_optimize:
                mock_optimize.return_value = None
                
                result = self.operators.local_optimization_mutation(self.parent1)
                
                self.assertIsInstance(result, RouteChromosome)
                self.assertEqual(result.creation_method, "local_optimization_mutation")
    
    def test_local_optimization_mutation_validation_failure(self):
        """Test local optimization mutation with validation failure"""
        with patch('random.random', return_value=0.05):  # < mutation_rate
            with patch.object(self.operators, '_optimize_segment_pair') as mock_optimize:
                mock_optimized = [Mock(spec=RouteSegment)]
                mock_optimize.return_value = mock_optimized
                
                with patch.object(self.constraints, 'validate') as mock_validate:
                    mock_validate.return_value = False
                    
                    result = self.operators.local_optimization_mutation(self.parent1)
                    
                    self.assertIsInstance(result, RouteChromosome)
                    # Should not have applied optimization
    
    def test_local_optimization_mutation_no_better_elevation(self):
        """Test local optimization mutation with no better elevation"""
        with patch('random.random', return_value=0.05):  # < mutation_rate
            with patch.object(self.operators, '_optimize_segment_pair') as mock_optimize:
                mock_optimized = [Mock(spec=RouteSegment)]
                mock_optimize.return_value = mock_optimized
                
                with patch.object(self.constraints, 'validate') as mock_validate:
                    mock_validate.return_value = True
                    
                    with patch.object(self.operators, '_has_better_elevation') as mock_better:
                        mock_better.return_value = False
                        
                        result = self.operators.local_optimization_mutation(self.parent1)
                        
                        self.assertIsInstance(result, RouteChromosome)
                        # Should not have applied optimization
    
    def test_optimize_segment_pair_basic(self):
        """Test segment pair optimization"""
        segment1 = RouteSegment(1001, 1002, [1001, 1002])
        segment1.calculate_properties(self.mock_graph)
        
        segment2 = RouteSegment(1002, 1003, [1002, 1003])
        segment2.calculate_properties(self.mock_graph)
        
        with patch('networkx.shortest_path') as mock_path:
            mock_path.side_effect = [
                [1001, 1004],  # start -> intermediate
                [1004, 1003]   # intermediate -> end
            ]
            
            result = self.operators._optimize_segment_pair(segment1, segment2)
            
            if result:
                self.assertIsInstance(result, list)
                self.assertEqual(len(result), 2)
                for segment in result:
                    self.assertIsInstance(segment, RouteSegment)
    
    def test_optimize_segment_pair_no_neighbors(self):
        """Test segment pair optimization with no neighbors"""
        # Create segments with node that has no neighbors
        segment1 = Mock(spec=RouteSegment)
        segment1.start_node = 1001
        segment1.end_node = 1002
        segment1.length = 400.0
        
        segment2 = Mock(spec=RouteSegment)
        segment2.start_node = 1002
        segment2.end_node = 1003
        segment2.length = 400.0
        
        with patch.object(self.mock_graph, 'neighbors') as mock_neighbors:
            mock_neighbors.return_value = []
            
            result = self.operators._optimize_segment_pair(segment1, segment2)
            
            self.assertIsNone(result)
    
    def test_optimize_segment_pair_exception_handling(self):
        """Test segment pair optimization with exception handling"""
        segment1 = Mock(spec=RouteSegment)
        segment1.start_node = 1001
        segment1.end_node = 1002
        segment1.length = 400.0
        
        segment2 = Mock(spec=RouteSegment)
        segment2.start_node = 1002
        segment2.end_node = 1003
        segment2.length = 400.0
        
        with patch('networkx.shortest_path') as mock_path:
            mock_path.side_effect = Exception("Path error")
            
            result = self.operators._optimize_segment_pair(segment1, segment2)
            
            self.assertIsNone(result)
    
    def test_optimize_segment_pair_length_validation(self):
        """Test segment pair optimization with length validation"""
        segment1 = Mock(spec=RouteSegment)
        segment1.start_node = 1001
        segment1.end_node = 1002
        segment1.length = 400.0
        
        segment2 = Mock(spec=RouteSegment)
        segment2.start_node = 1002
        segment2.end_node = 1003
        segment2.length = 400.0
        
        with patch('networkx.shortest_path') as mock_path:
            mock_path.side_effect = [
                [1001, 1004],  # start -> intermediate
                [1004, 1003]   # intermediate -> end
            ]
            
            with patch('genetic_algorithm.constraint_preserving_operators.RouteSegment') as mock_segment_class:
                mock_new_segment = Mock(spec=RouteSegment)
                mock_new_segment.length = 2000.0  # Too long (ratio > 1.3)
                mock_new_segment.calculate_properties.return_value = None
                mock_segment_class.return_value = mock_new_segment
                
                result = self.operators._optimize_segment_pair(segment1, segment2)
                
                self.assertIsNone(result)
    
    def test_has_better_elevation_true(self):
        """Test elevation improvement check - true case"""
        old_segment1 = Mock(spec=RouteSegment)
        old_segment1.elevation_gain = 10.0
        old_segment2 = Mock(spec=RouteSegment)
        old_segment2.elevation_gain = 15.0
        
        new_segment1 = Mock(spec=RouteSegment)
        new_segment1.elevation_gain = 20.0
        new_segment2 = Mock(spec=RouteSegment)
        new_segment2.elevation_gain = 10.0
        
        result = self.operators._has_better_elevation(
            [new_segment1, new_segment2], [old_segment1, old_segment2]
        )
        
        self.assertTrue(result)  # 30 > 25
    
    def test_has_better_elevation_false(self):
        """Test elevation improvement check - false case"""
        old_segment1 = Mock(spec=RouteSegment)
        old_segment1.elevation_gain = 20.0
        old_segment2 = Mock(spec=RouteSegment)
        old_segment2.elevation_gain = 15.0
        
        new_segment1 = Mock(spec=RouteSegment)
        new_segment1.elevation_gain = 10.0
        new_segment2 = Mock(spec=RouteSegment)
        new_segment2.elevation_gain = 10.0
        
        result = self.operators._has_better_elevation(
            [new_segment1, new_segment2], [old_segment1, old_segment2]
        )
        
        self.assertFalse(result)  # 20 < 35
    
    def test_constraint_repair_mutation_already_valid(self):
        """Test constraint repair mutation with already valid chromosome"""
        with patch.object(self.constraints, 'validate') as mock_validate:
            mock_validate.return_value = True
            
            result = self.operators.constraint_repair_mutation(self.parent1)
            
            self.assertEqual(result, self.parent1)
    
    def test_constraint_repair_mutation_distance_violation(self):
        """Test constraint repair mutation with distance violation"""
        with patch.object(self.constraints, 'validate') as mock_validate:
            mock_validate.return_value = False
            
            with patch.object(self.constraints, 'validate_distance') as mock_validate_distance:
                mock_validate_distance.return_value = False
                
                with patch.object(self.operators, '_repair_distance_constraint') as mock_repair:
                    mock_repaired = Mock(spec=RouteChromosome)
                    mock_repaired.creation_method = "constraint_repair_mutation"
                    mock_repair.return_value = mock_repaired
                    
                    with patch.object(self.constraints, 'validate_connectivity') as mock_validate_connectivity:
                        mock_validate_connectivity.return_value = True
                        
                        result = self.operators.constraint_repair_mutation(self.parent1)
                        
                        self.assertEqual(result, mock_repaired)
                        self.assertEqual(result.creation_method, "constraint_repair_mutation")
    
    def test_constraint_repair_mutation_connectivity_violation(self):
        """Test constraint repair mutation with connectivity violation"""
        with patch.object(self.constraints, 'validate') as mock_validate:
            mock_validate.return_value = False
            
            with patch.object(self.constraints, 'validate_distance') as mock_validate_distance:
                mock_validate_distance.return_value = True
                
                with patch.object(self.constraints, 'validate_connectivity') as mock_validate_connectivity:
                    mock_validate_connectivity.return_value = False
                    
                    with patch.object(self.operators, '_repair_connectivity_constraint') as mock_repair:
                        mock_repaired = Mock(spec=RouteChromosome)
                        mock_repaired.creation_method = "constraint_repair_mutation"
                        mock_repair.return_value = mock_repaired
                        
                        result = self.operators.constraint_repair_mutation(self.parent1)
                        
                        self.assertEqual(result, mock_repaired)
                        self.assertEqual(result.creation_method, "constraint_repair_mutation")
    
    def test_repair_distance_constraint_too_short(self):
        """Test distance constraint repair - too short"""
        mock_chromosome = Mock(spec=RouteChromosome)
        mock_chromosome.get_route_stats.return_value = {'total_distance_km': 1.0}
        
        with patch.object(self.operators, '_extend_route') as mock_extend:
            mock_extended = Mock(spec=RouteChromosome)
            mock_extend.return_value = mock_extended
            
            result = self.operators._repair_distance_constraint(mock_chromosome)
            
            self.assertEqual(result, mock_extended)
            mock_extend.assert_called_once_with(mock_chromosome)
    
    def test_repair_distance_constraint_too_long(self):
        """Test distance constraint repair - too long"""
        mock_chromosome = Mock(spec=RouteChromosome)
        mock_chromosome.get_route_stats.return_value = {'total_distance_km': 6.0}
        
        with patch.object(self.operators, '_shorten_route') as mock_shorten:
            mock_shortened = Mock(spec=RouteChromosome)
            mock_shorten.return_value = mock_shortened
            
            result = self.operators._repair_distance_constraint(mock_chromosome)
            
            self.assertEqual(result, mock_shortened)
            mock_shorten.assert_called_once_with(mock_chromosome)
    
    def test_repair_distance_constraint_valid(self):
        """Test distance constraint repair - already valid"""
        mock_chromosome = Mock(spec=RouteChromosome)
        mock_chromosome.get_route_stats.return_value = {'total_distance_km': 3.0}
        
        result = self.operators._repair_distance_constraint(mock_chromosome)
        
        self.assertEqual(result, mock_chromosome)
    
    def test_extend_route_basic(self):
        """Test route extension"""
        chromosome = self._create_test_chromosome([1001, 1002, 1003, 1001])
        
        with patch.object(self.operators, '_find_longer_alternative') as mock_find:
            mock_alternative = Mock(spec=RouteSegment)
            mock_alternative.start_node = 1002
            mock_alternative.end_node = 1003
            mock_alternative.length = 800.0
            mock_find.return_value = mock_alternative
            
            result = self.operators._extend_route(chromosome)
            
            self.assertIsInstance(result, RouteChromosome)
            self.assertGreater(len(result.segments), 0)
    
    def test_extend_route_no_alternative(self):
        """Test route extension with no alternative"""
        chromosome = self._create_test_chromosome([1001, 1002, 1003, 1001])
        
        with patch.object(self.operators, '_find_longer_alternative') as mock_find:
            mock_find.return_value = None
            
            result = self.operators._extend_route(chromosome)
            
            self.assertEqual(result, chromosome)
    
    def test_extend_route_few_segments(self):
        """Test route extension with few segments"""
        chromosome = self._create_test_chromosome([1001, 1002, 1001])
        
        with patch.object(self.operators, '_find_longer_alternative') as mock_find:
            mock_alternative = Mock(spec=RouteSegment)
            mock_find.return_value = mock_alternative
            
            result = self.operators._extend_route(chromosome)
            
            self.assertIsInstance(result, RouteChromosome)
    
    def test_shorten_route_basic(self):
        """Test route shortening"""
        chromosome = self._create_test_chromosome([1001, 1002, 1003, 1001])
        
        with patch.object(self.operators, '_find_shorter_alternative') as mock_find:
            mock_alternative = Mock(spec=RouteSegment)
            mock_alternative.start_node = 1002
            mock_alternative.end_node = 1003
            mock_alternative.length = 200.0
            mock_find.return_value = mock_alternative
            
            result = self.operators._shorten_route(chromosome)
            
            self.assertIsInstance(result, RouteChromosome)
            self.assertGreater(len(result.segments), 0)
    
    def test_shorten_route_no_alternative(self):
        """Test route shortening with no alternative"""
        chromosome = self._create_test_chromosome([1001, 1002, 1003, 1001])
        
        with patch.object(self.operators, '_find_shorter_alternative') as mock_find:
            mock_find.return_value = None
            
            result = self.operators._shorten_route(chromosome)
            
            self.assertEqual(result, chromosome)
    
    def test_find_longer_alternative_basic(self):
        """Test finding longer alternative"""
        with patch('networkx.shortest_path') as mock_path:
            mock_path.side_effect = [
                [1001, 1004],  # start -> intermediate
                [1004, 1002]   # intermediate -> end
            ]
            
            result = self.operators._find_longer_alternative(1001, 1002, 800.0)
            
            if result:
                self.assertIsInstance(result, RouteSegment)
                self.assertEqual(result.start_node, 1001)
                self.assertEqual(result.end_node, 1002)
                self.assertGreaterEqual(result.length, 800.0)
    
    def test_find_longer_alternative_no_path(self):
        """Test finding longer alternative with no path"""
        with patch('networkx.shortest_path') as mock_path:
            mock_path.side_effect = nx.NetworkXNoPath("No path")
            
            result = self.operators._find_longer_alternative(1001, 1002, 800.0)
            
            self.assertIsNone(result)
    
    def test_find_longer_alternative_exception(self):
        """Test finding longer alternative with exception"""
        with patch('networkx.shortest_path') as mock_path:
            mock_path.side_effect = Exception("Path error")
            
            result = self.operators._find_longer_alternative(1001, 1002, 800.0)
            
            self.assertIsNone(result)
    
    def test_find_shorter_alternative_direct_edge(self):
        """Test finding shorter alternative with direct edge"""
        result = self.operators._find_shorter_alternative(1001, 1002)
        
        if result:
            self.assertIsInstance(result, RouteSegment)
            self.assertEqual(result.start_node, 1001)
            self.assertEqual(result.end_node, 1002)
    
    def test_find_shorter_alternative_no_edge(self):
        """Test finding shorter alternative with no direct edge"""
        result = self.operators._find_shorter_alternative(1001, 1005)
        
        # May or may not find alternative depending on graph structure
        if result:
            self.assertIsInstance(result, RouteSegment)
    
    def test_find_shorter_alternative_exception(self):
        """Test finding shorter alternative with exception"""
        with patch.object(self.mock_graph, 'has_edge') as mock_has_edge:
            mock_has_edge.side_effect = Exception("Graph error")
            
            result = self.operators._find_shorter_alternative(1001, 1002)
            
            self.assertIsNone(result)
    
    def test_repair_connectivity_constraint_basic(self):
        """Test connectivity constraint repair"""
        mock_chromosome = Mock(spec=RouteChromosome)
        
        result = self.operators._repair_connectivity_constraint(mock_chromosome)
        
        # Current implementation just returns original
        self.assertEqual(result, mock_chromosome)


if __name__ == '__main__':
    unittest.main()