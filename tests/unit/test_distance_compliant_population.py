#!/usr/bin/env python3
"""
Unit tests for DistanceCompliantPopulationInitializer
Tests comprehensive functionality of distance-compliant population initialization
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import networkx as nx
import sys
import os
import time
import random

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from genetic_algorithm.distance_compliant_population import DistanceCompliantPopulationInitializer
from genetic_algorithm.chromosome import RouteChromosome, RouteSegment


class TestDistanceCompliantPopulationInitializer(unittest.TestCase):
    """Test DistanceCompliantPopulationInitializer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock graph with realistic structure
        self.mock_graph = nx.Graph()
        
        # Add nodes in a grid pattern for testing
        nodes = [
            (1001, -80.4094, 37.1299, 100.0),
            (1002, -80.4090, 37.1299, 105.0),
            (1003, -80.4086, 37.1299, 110.0),
            (1004, -80.4082, 37.1299, 115.0),
            (1005, -80.4078, 37.1299, 120.0),
            (1006, -80.4094, 37.1303, 95.0),
            (1007, -80.4090, 37.1303, 100.0),
            (1008, -80.4086, 37.1303, 105.0),
            (1009, -80.4082, 37.1303, 110.0),
            (1010, -80.4078, 37.1303, 115.0),
            (1011, -80.4094, 37.1307, 90.0),
            (1012, -80.4090, 37.1307, 95.0),
            (1013, -80.4086, 37.1307, 100.0),
            (1014, -80.4082, 37.1307, 105.0),
            (1015, -80.4078, 37.1307, 110.0)
        ]
        
        for node_id, x, y, elevation in nodes:
            self.mock_graph.add_node(node_id, x=x, y=y, elevation=elevation)
        
        # Add edges to create connected network
        edges = [
            # Horizontal connections
            (1001, 1002, 400), (1002, 1003, 400), (1003, 1004, 400), (1004, 1005, 400),
            (1006, 1007, 400), (1007, 1008, 400), (1008, 1009, 400), (1009, 1010, 400),
            (1011, 1012, 400), (1012, 1013, 400), (1013, 1014, 400), (1014, 1015, 400),
            # Vertical connections
            (1001, 1006, 400), (1006, 1011, 400),
            (1002, 1007, 400), (1007, 1012, 400),
            (1003, 1008, 400), (1008, 1013, 400),
            (1004, 1009, 400), (1009, 1014, 400),
            (1005, 1010, 400), (1010, 1015, 400),
            # Some diagonal connections for diversity
            (1001, 1007, 565), (1002, 1008, 565), (1003, 1009, 565),
            (1006, 1012, 565), (1007, 1013, 565), (1008, 1014, 565)
        ]
        
        for node1, node2, length in edges:
            self.mock_graph.add_edge(node1, node2, length=length)
        
        self.start_node = 1001
        
        # Mock the distance computation to avoid expensive calculations
        self.mock_distances = {
            1001: 0, 1002: 400, 1003: 800, 1004: 1200, 1005: 1600,
            1006: 400, 1007: 565, 1008: 800, 1009: 1200, 1010: 1600,
            1011: 800, 1012: 965, 1013: 1200, 1014: 1600, 1015: 2000
        }
        
        # Mock the nodes_by_distance structure
        self.mock_nodes_by_distance = {
            0: [(1001, 0)],
            0: [(1002, 400), (1006, 400)],
            1: [(1003, 800), (1007, 565), (1008, 800), (1011, 800)],
            2: [(1004, 1200), (1009, 1200), (1012, 965), (1013, 1200)],
            3: [(1005, 1600), (1010, 1600), (1014, 1600)],
            4: [(1015, 2000)]
        }
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_initializer_creation_basic(self, mock_dijkstra):
        """Test basic DistanceCompliantPopulationInitializer creation"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node, max_route_distance_km=10.0
        )
        
        self.assertEqual(initializer.graph, self.mock_graph)
        self.assertEqual(initializer.start_node, self.start_node)
        self.assertIsInstance(initializer.distances_from_start, dict)
        self.assertIsInstance(initializer.nodes_by_distance, dict)
        
        # Verify dijkstra was called with correct parameters (8km minimum)
        mock_dijkstra.assert_called_once_with(
            self.mock_graph, self.start_node, weight='length', cutoff=8000
        )
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_initializer_creation_with_adaptive_cutoff(self, mock_dijkstra):
        """Test initializer creation with adaptive cutoff distance"""
        mock_dijkstra.return_value = self.mock_distances
        
        # Test with small route distance
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node, max_route_distance_km=5.0
        )
        
        # Should use at least 8km cutoff
        expected_cutoff = 8000
        mock_dijkstra.assert_called_once_with(
            self.mock_graph, self.start_node, weight='length', cutoff=expected_cutoff
        )
        
        # Reset mock
        mock_dijkstra.reset_mock()
        
        # Test with large route distance
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node, max_route_distance_km=20.0
        )
        
        # Should use 50% of route distance
        expected_cutoff = 10000
        mock_dijkstra.assert_called_once_with(
            self.mock_graph, self.start_node, weight='length', cutoff=expected_cutoff
        )
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_nodes_by_distance_grouping(self, mock_dijkstra):
        """Test proper grouping of nodes by distance ranges"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        
        # Verify nodes are grouped into 500m bins
        self.assertIsInstance(initializer.nodes_by_distance, dict)
        
        # Check that nodes are grouped correctly (within 500m bins)
        for distance_bin, nodes in initializer.nodes_by_distance.items():
            for node, distance in nodes:
                expected_bin = int(distance // 500)
                self.assertEqual(distance_bin, expected_bin)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_population_basic(self, mock_dijkstra):
        """Test basic population creation"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.nodes_by_distance = self.mock_nodes_by_distance
        
        with patch.object(initializer, '_create_route_by_strategy') as mock_create_route:
            mock_chromosome = Mock(spec=RouteChromosome)
            mock_chromosome.creation_method = 'out_and_back'
            mock_create_route.return_value = mock_chromosome
            
            population = initializer.create_population(size=10, target_distance_km=5.0)
            
            self.assertIsInstance(population, list)
            self.assertLessEqual(len(population), 10)
            
            # Should have called create_route_by_strategy multiple times
            self.assertGreater(mock_create_route.call_count, 0)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_population_strategy_distribution_short_route(self, mock_dijkstra):
        """Test population creation strategy distribution for short routes"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.nodes_by_distance = self.mock_nodes_by_distance
        
        with patch.object(initializer, '_create_route_by_strategy') as mock_create_route:
            mock_chromosome = Mock(spec=RouteChromosome)
            mock_chromosome.creation_method = 'out_and_back'
            mock_create_route.return_value = mock_chromosome
            
            population = initializer.create_population(size=10, target_distance_km=5.0)
            
            # Verify different strategies were called
            called_strategies = [call[0][0] for call in mock_create_route.call_args_list]
            expected_strategies = ['out_and_back', 'triangle_route', 'figure_eight', 'spiral_out']
            
            # Should use all strategies for short routes
            for strategy in expected_strategies:
                self.assertIn(strategy, called_strategies)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_population_strategy_distribution_long_route(self, mock_dijkstra):
        """Test population creation strategy distribution for long routes"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.nodes_by_distance = self.mock_nodes_by_distance
        
        with patch.object(initializer, '_create_route_by_strategy') as mock_create_route:
            mock_chromosome = Mock(spec=RouteChromosome)
            mock_chromosome.creation_method = 'out_and_back'
            mock_create_route.return_value = mock_chromosome
            
            population = initializer.create_population(size=10, target_distance_km=20.0)
            
            # Verify different strategies were called
            called_strategies = [call[0][0] for call in mock_create_route.call_args_list]
            expected_strategies = ['out_and_back', 'triangle_route', 'figure_eight']
            
            # Should use simplified strategies for long routes (no spiral_out)
            for strategy in expected_strategies:
                self.assertIn(strategy, called_strategies)
            
            # Should not use spiral_out for long routes
            self.assertNotIn('spiral_out', called_strategies)
    
    @patch('networkx.single_source_dijkstra_path_length')
    @patch('genetic_algorithm.distance_compliant_population.time.time')
    def test_create_population_timeout_handling(self, mock_time, mock_dijkstra):
        """Test population creation with timeout handling"""
        mock_dijkstra.return_value = self.mock_distances
        
        # Mock time to simulate timeout
        mock_time.side_effect = [0, 0, 125, 125, 125, 125, 125, 125]  # Start, check, timeout (repeat timeout)
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.nodes_by_distance = self.mock_nodes_by_distance
        
        with patch.object(initializer, '_create_route_by_strategy') as mock_create_route:
            mock_chromosome = Mock(spec=RouteChromosome)
            mock_chromosome.creation_method = 'out_and_back'
            mock_create_route.return_value = mock_chromosome
            
            population = initializer.create_population(size=10, target_distance_km=5.0)
            
            # Should have stopped early due to timeout
            self.assertIsInstance(population, list)
            # May have fewer routes due to timeout
            self.assertLessEqual(len(population), 10)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_population_fill_remaining_slots(self, mock_dijkstra):
        """Test population creation fills remaining slots with out_and_back"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.nodes_by_distance = self.mock_nodes_by_distance
        
        # Mock to return None for some strategies but success for out_and_back
        def mock_create_route(strategy, *args):
            if strategy == 'out_and_back':
                mock_chromosome = Mock(spec=RouteChromosome)
                mock_chromosome.creation_method = strategy
                return mock_chromosome
            return None
        
        with patch.object(initializer, '_create_route_by_strategy', side_effect=mock_create_route):
            population = initializer.create_population(size=10, target_distance_km=5.0)
            
            # Should fill remaining slots with out_and_back
            self.assertGreater(len(population), 0)
            
            # Check that out_and_back was used for filling
            out_and_back_count = sum(1 for chromosome in population 
                                   if chromosome.creation_method == 'out_and_back')
            self.assertGreater(out_and_back_count, 0)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_route_by_strategy_dispatch(self, mock_dijkstra):
        """Test route creation strategy dispatch"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        
        # Test all valid strategies
        strategies = ['out_and_back', 'triangle_route', 'figure_eight', 'spiral_out']
        
        for strategy in strategies:
            with patch.object(initializer, f'_create_{strategy}') as mock_method:
                mock_chromosome = Mock(spec=RouteChromosome)
                mock_method.return_value = mock_chromosome
                
                result = initializer._create_route_by_strategy(
                    strategy, 5000.0, 4250.0, 5750.0
                )
                
                mock_method.assert_called_once_with(5000.0, 4250.0, 5750.0)
                self.assertEqual(result, mock_chromosome)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_route_by_strategy_invalid_strategy(self, mock_dijkstra):
        """Test route creation with invalid strategy defaults to out_and_back"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        
        with patch.object(initializer, '_create_out_and_back') as mock_method:
            mock_chromosome = Mock(spec=RouteChromosome)
            mock_method.return_value = mock_chromosome
            
            result = initializer._create_route_by_strategy(
                'invalid_strategy', 5000.0, 4250.0, 5750.0
            )
            
            mock_method.assert_called_once_with(5000.0, 4250.0, 5750.0)
            self.assertEqual(result, mock_chromosome)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_out_and_back_success(self, mock_dijkstra):
        """Test successful out-and-back route creation"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.nodes_by_distance = self.mock_nodes_by_distance
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = [(1003, 800), (1008, 800)]
            
            with patch.object(initializer, '_create_segment') as mock_create_segment:
                mock_segment = Mock(spec=RouteSegment)
                mock_segment.length = 2375.0  # Total 4750 (within range 4250-5750)
                mock_create_segment.return_value = mock_segment
                
                with patch('genetic_algorithm.distance_compliant_population.RouteChromosome') as mock_chromosome_class:
                    mock_chromosome = Mock(spec=RouteChromosome)
                    mock_chromosome.validate_connectivity.return_value = None
                    mock_chromosome_class.return_value = mock_chromosome
                    
                    result = initializer._create_out_and_back(5000.0, 4250.0, 5750.0)
                    
                    self.assertIsNotNone(result)
                    self.assertEqual(result, mock_chromosome)
                    mock_chromosome.validate_connectivity.assert_called_once()
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_out_and_back_no_target_nodes(self, mock_dijkstra):
        """Test out-and-back route creation with no target nodes"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = []
            
            result = initializer._create_out_and_back(5000.0, 4250.0, 5750.0)
            
            self.assertIsNone(result)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_out_and_back_segment_creation_failure(self, mock_dijkstra):
        """Test out-and-back route creation with segment creation failure"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = [(1003, 800), (1008, 800)]
            
            with patch.object(initializer, '_create_segment') as mock_create_segment:
                mock_create_segment.return_value = None  # Segment creation fails
                
                result = initializer._create_out_and_back(5000.0, 4250.0, 5750.0)
                
                self.assertIsNone(result)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_out_and_back_distance_validation(self, mock_dijkstra):
        """Test out-and-back route creation with distance validation"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = [(1003, 800), (1008, 800)]
            
            with patch.object(initializer, '_create_segment') as mock_create_segment:
                mock_segment = Mock(spec=RouteSegment)
                mock_segment.length = 1000.0  # Too short for distance range
                mock_create_segment.return_value = mock_segment
                
                result = initializer._create_out_and_back(5000.0, 4250.0, 5750.0)
                
                self.assertIsNone(result)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_triangle_route_success(self, mock_dijkstra):
        """Test successful triangle route creation"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        # Include node 1010 in distances for triangle route to work
        mock_distances_with_1010 = self.mock_distances.copy()
        mock_distances_with_1010[1010] = 1600  # Return distance
        initializer.distances_from_start = mock_distances_with_1010
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = [(1003, 1600), (1008, 1600)]  # dist_a
            
            with patch.object(initializer, '_find_nodes_at_distance_from_node') as mock_find_from_node:
                mock_find_from_node.return_value = [(1010, 1600)]  # dist_b
                
                with patch.object(initializer, '_create_segment') as mock_create_segment:
                    mock_segment = Mock(spec=RouteSegment)
                    mock_segment.length = 1500.0  # 3 segments = 4500 total, within range
                    mock_create_segment.return_value = mock_segment
                    
                    with patch('genetic_algorithm.distance_compliant_population.RouteChromosome') as mock_chromosome_class:
                        mock_chromosome = Mock(spec=RouteChromosome)
                        mock_chromosome.validate_connectivity.return_value = None
                        mock_chromosome_class.return_value = mock_chromosome
                        
                        result = initializer._create_triangle_route(5000.0, 4250.0, 5750.0)
                        
                        self.assertIsNotNone(result)
                        self.assertEqual(result, mock_chromosome)
                        mock_chromosome.validate_connectivity.assert_called_once()
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_triangle_route_no_waypoints(self, mock_dijkstra):
        """Test triangle route creation with no waypoints"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = []
            
            result = initializer._create_triangle_route(5000.0, 4250.0, 5750.0)
            
            self.assertIsNone(result)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_figure_eight_success(self, mock_dijkstra):
        """Test successful figure-eight route creation"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = [(1003, 800), (1008, 800)]
            
            with patch.object(initializer, '_create_segment') as mock_create_segment:
                mock_segment = Mock(spec=RouteSegment)
                mock_segment.length = 1200.0  # Total 4800 (within range 4250-5750)
                mock_create_segment.return_value = mock_segment
                
                with patch('genetic_algorithm.distance_compliant_population.RouteChromosome') as mock_chromosome_class:
                    mock_chromosome = Mock(spec=RouteChromosome)
                    mock_chromosome.validate_connectivity.return_value = None
                    mock_chromosome_class.return_value = mock_chromosome
                    
                    result = initializer._create_figure_eight(5000.0, 4250.0, 5750.0)
                    
                    self.assertIsNotNone(result)
                    self.assertEqual(result, mock_chromosome)
                    mock_chromosome.validate_connectivity.assert_called_once()
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_figure_eight_no_candidates(self, mock_dijkstra):
        """Test figure-eight route creation with no candidates"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = []
            
            result = initializer._create_figure_eight(5000.0, 4250.0, 5750.0)
            
            self.assertIsNone(result)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_figure_eight_same_waypoints(self, mock_dijkstra):
        """Test figure-eight route creation skips same waypoints"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = [(1003, 800), (1003, 800)]  # Same waypoint
            
            with patch.object(initializer, '_create_segment') as mock_create_segment:
                mock_segment = Mock(spec=RouteSegment)
                mock_segment.length = 1250.0
                mock_create_segment.return_value = mock_segment
                
                result = initializer._create_figure_eight(5000.0, 4250.0, 5750.0)
                
                # Should return None because same waypoints are skipped
                self.assertIsNone(result)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_spiral_out_success(self, mock_dijkstra):
        """Test successful spiral-out route creation"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.distances_from_start = self.mock_distances
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = [(1003, 800), (1008, 800)]
            
            with patch.object(initializer, '_find_nodes_with_return_distance') as mock_find_return:
                mock_find_return.return_value = [(1010, 800)]
                
                with patch.object(initializer, '_create_segment') as mock_create_segment:
                    mock_segment = Mock(spec=RouteSegment)
                    mock_segment.length = 1000.0
                    mock_create_segment.return_value = mock_segment
                    
                    with patch('genetic_algorithm.distance_compliant_population.RouteChromosome') as mock_chromosome_class:
                        mock_chromosome = Mock(spec=RouteChromosome)
                        mock_chromosome.validate_connectivity.return_value = None
                        mock_chromosome_class.return_value = mock_chromosome
                        
                        result = initializer._create_spiral_out(5000.0, 4250.0, 5750.0)
                        
                        self.assertIsNotNone(result)
                        self.assertEqual(result, mock_chromosome)
                        mock_chromosome.validate_connectivity.assert_called_once()
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_spiral_out_distance_too_large(self, mock_dijkstra):
        """Test spiral-out route creation with distance too large"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.distances_from_start = self.mock_distances
        
        # Test with very large target distance
        result = initializer._create_spiral_out(50000.0, 42500.0, 57500.0)
        
        self.assertIsNone(result)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_spiral_out_no_candidates(self, mock_dijkstra):
        """Test spiral-out route creation with no candidates"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.distances_from_start = self.mock_distances
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = []
            
            result = initializer._create_spiral_out(5000.0, 4250.0, 5750.0)
            
            self.assertIsNone(result)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_spiral_out_segment_creation_failure(self, mock_dijkstra):
        """Test spiral-out route creation with segment creation failure"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.distances_from_start = self.mock_distances
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = [(1003, 800)]
            
            with patch.object(initializer, '_create_segment') as mock_create_segment:
                mock_create_segment.return_value = None  # Segment creation fails
                
                result = initializer._create_spiral_out(5000.0, 4250.0, 5750.0)
                
                self.assertIsNone(result)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_find_nodes_at_distance_basic(self, mock_dijkstra):
        """Test basic node finding at target distance"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.distances_from_start = self.mock_distances
        
        # Find nodes at ~800m distance
        candidates = initializer._find_nodes_at_distance(800.0, tolerance=0.3)
        
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)
        
        # Check that returned nodes are within tolerance
        for node, distance in candidates:
            self.assertGreaterEqual(distance, 800.0 * 0.7)  # 30% tolerance
            self.assertLessEqual(distance, 800.0 * 1.3)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_find_nodes_at_distance_sorting(self, mock_dijkstra):
        """Test that nodes are sorted by distance to target"""
        # Create mock distances with multiple nodes at similar distances
        mock_distances_for_sorting = {
            1001: 0,
            1002: 900,  # Close to 1000
            1003: 1100,  # Close to 1000
            1004: 950,  # Closer to 1000
            1005: 1050,  # Close to 1000
        }
        mock_dijkstra.return_value = mock_distances_for_sorting
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.distances_from_start = mock_distances_for_sorting
        
        # Find nodes at ~1000m distance
        candidates = initializer._find_nodes_at_distance(1000.0, tolerance=0.5)
        
        if len(candidates) > 1:
            # Check that candidates are sorted by closeness to target
            prev_diff = None
            for node, distance in candidates:
                current_diff = abs(distance - 1000.0)
                if prev_diff is not None:
                    self.assertLessEqual(prev_diff, current_diff)  # prev <= current (increasing order)
                prev_diff = current_diff
        else:
            # If no candidates or only one, the test is still valid
            self.assertIsInstance(candidates, list)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_find_nodes_at_distance_from_node_basic(self, mock_dijkstra):
        """Test finding nodes at distance from specific node"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        
        with patch('networkx.single_source_dijkstra_path_length') as mock_specific_dijkstra:
            mock_specific_dijkstra.return_value = {1003: 0, 1008: 400, 1013: 800}
            
            candidates = initializer._find_nodes_at_distance_from_node(1003, 400.0, tolerance=0.3)
            
            self.assertIsInstance(candidates, list)
            # Should call dijkstra with limited cutoff
            mock_specific_dijkstra.assert_called_once_with(
                self.mock_graph, 1003, weight='length', cutoff=600.0
            )
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_find_nodes_at_distance_from_node_exception(self, mock_dijkstra):
        """Test finding nodes from specific node with exception handling"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        
        with patch('networkx.single_source_dijkstra_path_length') as mock_specific_dijkstra:
            mock_specific_dijkstra.side_effect = Exception("Network error")
            
            candidates = initializer._find_nodes_at_distance_from_node(1003, 400.0)
            
            self.assertEqual(candidates, [])
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_find_nodes_with_return_distance_basic(self, mock_dijkstra):
        """Test finding nodes with good return distance"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.distances_from_start = self.mock_distances
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = [(1003, 800), (1008, 800), (1013, 1200)]
            
            candidates = initializer._find_nodes_with_return_distance(800.0, 800.0, tolerance=0.3)
            
            self.assertIsInstance(candidates, list)
            # Should filter based on return distance
            for node, distance in candidates:
                return_distance = initializer.distances_from_start[node]
                self.assertLessEqual(abs(return_distance - 800.0), 800.0 * 0.3)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_segment_success(self, mock_dijkstra):
        """Test successful segment creation"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        
        with patch('networkx.shortest_path') as mock_shortest_path:
            mock_shortest_path.return_value = [1001, 1002, 1003]
            
            result = initializer._create_segment(1001, 1003)
            
            self.assertIsNotNone(result)
            self.assertIsInstance(result, RouteSegment)
            mock_shortest_path.assert_called_once_with(
                self.mock_graph, 1001, 1003, weight='length'
            )
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_segment_exception(self, mock_dijkstra):
        """Test segment creation with exception handling"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        
        with patch('networkx.shortest_path') as mock_shortest_path:
            mock_shortest_path.side_effect = Exception("No path found")
            
            result = initializer._create_segment(1001, 1003)
            
            self.assertIsNone(result)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_population_spiral_out_failure_handling(self, mock_dijkstra):
        """Test population creation handles spiral_out failures gracefully"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.nodes_by_distance = self.mock_nodes_by_distance
        
        call_count = 0
        def mock_create_route(strategy, *args):
            nonlocal call_count
            call_count += 1
            
            if strategy == 'spiral_out':
                return None  # Always fail for spiral_out
            elif strategy == 'out_and_back':
                mock_chromosome = Mock(spec=RouteChromosome)
                mock_chromosome.creation_method = strategy
                return mock_chromosome
            else:
                # Return success for other strategies
                mock_chromosome = Mock(spec=RouteChromosome)
                mock_chromosome.creation_method = strategy
                return mock_chromosome
        
        with patch.object(initializer, '_create_route_by_strategy', side_effect=mock_create_route):
            population = initializer.create_population(size=10, target_distance_km=5.0)
            
            # Should have skipped remaining spiral_out attempts
            self.assertGreater(len(population), 0)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_population_fallback_failure(self, mock_dijkstra):
        """Test population creation when even fallback fails"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.nodes_by_distance = self.mock_nodes_by_distance
        
        with patch.object(initializer, '_create_route_by_strategy') as mock_create_route:
            mock_create_route.return_value = None  # All strategies fail
            
            population = initializer.create_population(size=10, target_distance_km=5.0)
            
            # Should handle gracefully and return partial population
            self.assertIsInstance(population, list)
            self.assertEqual(len(population), 0)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_population_with_creation_method_assignment(self, mock_dijkstra):
        """Test that created routes have proper creation_method assigned"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.nodes_by_distance = self.mock_nodes_by_distance
        
        with patch.object(initializer, '_create_route_by_strategy') as mock_create_route:
            mock_chromosome = Mock(spec=RouteChromosome)
            mock_create_route.return_value = mock_chromosome
            
            population = initializer.create_population(size=5, target_distance_km=5.0)
            
            # Check that creation_method is assigned
            for chromosome in population:
                self.assertTrue(hasattr(chromosome, 'creation_method'))
                self.assertIsNotNone(chromosome.creation_method)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_population_progress_logging(self, mock_dijkstra):
        """Test population creation progress logging"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.nodes_by_distance = self.mock_nodes_by_distance
        
        with patch.object(initializer, '_create_route_by_strategy') as mock_create_route:
            mock_chromosome = Mock(spec=RouteChromosome)
            mock_chromosome.creation_method = 'out_and_back'
            mock_create_route.return_value = mock_chromosome
            
            with patch('builtins.print') as mock_print:
                population = initializer.create_population(size=20, target_distance_km=5.0)
                
                # Should have printed progress messages
                print_calls = [str(call) for call in mock_print.call_args_list]
                progress_messages = [call for call in print_calls if 'Created' in call and 'routes' in call]
                self.assertGreater(len(progress_messages), 0)

    # Additional comprehensive tests for edge cases and error conditions
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_triangle_route_success_with_validation(self, mock_dijkstra):
        """Test triangle route creation with proper distance validation"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        # Include node 1010 in distances for triangle route to work
        mock_distances_with_1010 = self.mock_distances.copy()
        mock_distances_with_1010[1010] = 1600  # Return distance
        initializer.distances_from_start = mock_distances_with_1010
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = [(1003, 1600), (1008, 1600)]  # dist_a
            
            with patch.object(initializer, '_find_nodes_at_distance_from_node') as mock_find_from_node:
                mock_find_from_node.return_value = [(1010, 1600)]  # dist_b
                
                with patch.object(initializer, '_create_segment') as mock_create_segment:
                    mock_segment = Mock(spec=RouteSegment)
                    mock_segment.length = 1500.0
                    mock_create_segment.return_value = mock_segment
                    
                    with patch('genetic_algorithm.distance_compliant_population.RouteChromosome') as mock_chromosome_class:
                        mock_chromosome = Mock(spec=RouteChromosome)
                        mock_chromosome.validate_connectivity.return_value = None
                        mock_chromosome_class.return_value = mock_chromosome
                        
                        result = initializer._create_triangle_route(5000.0, 4250.0, 5750.0)
                        
                        self.assertIsNotNone(result)
                        self.assertEqual(result, mock_chromosome)
                        # Should have called create_segment 3 times (3 segments in triangle)
                        self.assertEqual(mock_create_segment.call_count, 3)
                        mock_chromosome.validate_connectivity.assert_called_once()
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_triangle_route_segment_failure(self, mock_dijkstra):
        """Test triangle route creation with segment creation failure"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.distances_from_start = self.mock_distances
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = [(1003, 800), (1008, 800)]
            
            with patch.object(initializer, '_find_nodes_at_distance_from_node') as mock_find_from_node:
                mock_find_from_node.return_value = [(1010, 800)]
                
                with patch.object(initializer, '_create_segment') as mock_create_segment:
                    mock_create_segment.return_value = None  # Segment creation fails
                    
                    result = initializer._create_triangle_route(5000.0, 4250.0, 5750.0)
                    
                    self.assertIsNone(result)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_triangle_route_distance_validation_failure(self, mock_dijkstra):
        """Test triangle route creation with distance validation failure"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.distances_from_start = self.mock_distances
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = [(1003, 800), (1008, 800)]
            
            with patch.object(initializer, '_find_nodes_at_distance_from_node') as mock_find_from_node:
                mock_find_from_node.return_value = [(1010, 800)]
                
                with patch.object(initializer, '_create_segment') as mock_create_segment:
                    mock_segment = Mock(spec=RouteSegment)
                    mock_segment.length = 100.0  # Too short total distance
                    mock_create_segment.return_value = mock_segment
                    
                    result = initializer._create_triangle_route(5000.0, 4250.0, 5750.0)
                    
                    self.assertIsNone(result)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_triangle_route_exception_handling(self, mock_dijkstra):
        """Test triangle route creation with exception handling"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.distances_from_start = self.mock_distances
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = [(1003, 800), (1008, 800)]
            
            with patch.object(initializer, '_find_nodes_at_distance_from_node') as mock_find_from_node:
                mock_find_from_node.side_effect = Exception("Network error")
                
                result = initializer._create_triangle_route(5000.0, 4250.0, 5750.0)
                
                self.assertIsNone(result)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_figure_eight_segment_failure(self, mock_dijkstra):
        """Test figure-eight route creation with segment creation failure"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = [(1003, 800), (1008, 800)]
            
            with patch.object(initializer, '_create_segment') as mock_create_segment:
                mock_create_segment.return_value = None  # Segment creation fails
                
                result = initializer._create_figure_eight(5000.0, 4250.0, 5750.0)
                
                self.assertIsNone(result)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_figure_eight_distance_validation_failure(self, mock_dijkstra):
        """Test figure-eight route creation with distance validation failure"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = [(1003, 800), (1008, 800)]
            
            with patch.object(initializer, '_create_segment') as mock_create_segment:
                mock_segment = Mock(spec=RouteSegment)
                mock_segment.length = 100.0  # Too short total distance
                mock_create_segment.return_value = mock_segment
                
                result = initializer._create_figure_eight(5000.0, 4250.0, 5750.0)
                
                self.assertIsNone(result)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_figure_eight_exception_handling(self, mock_dijkstra):
        """Test figure-eight route creation with exception handling"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = [(1003, 800), (1008, 800)]
            
            with patch.object(initializer, '_create_segment') as mock_create_segment:
                mock_create_segment.side_effect = Exception("Network error")
                
                result = initializer._create_figure_eight(5000.0, 4250.0, 5750.0)
                
                self.assertIsNone(result)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_spiral_out_return_segment_failure(self, mock_dijkstra):
        """Test spiral-out route creation with return segment failure"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.distances_from_start = self.mock_distances
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = [(1003, 800)]
            
            with patch.object(initializer, '_create_segment') as mock_create_segment:
                def mock_segment_creation(start, end):
                    if end == self.start_node:  # Return segment
                        return None
                    mock_segment = Mock(spec=RouteSegment)
                    mock_segment.length = 1000.0
                    return mock_segment
                
                mock_create_segment.side_effect = mock_segment_creation
                
                result = initializer._create_spiral_out(5000.0, 4250.0, 5750.0)
                
                self.assertIsNone(result)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_spiral_out_return_segment_exception(self, mock_dijkstra):
        """Test spiral-out route creation with return segment exception"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.distances_from_start = self.mock_distances
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = [(1003, 800)]
            
            with patch.object(initializer, '_create_segment') as mock_create_segment:
                def mock_segment_creation(start, end):
                    if end == self.start_node:  # Return segment
                        raise Exception("Return segment error")
                    mock_segment = Mock(spec=RouteSegment)
                    mock_segment.length = 1000.0
                    return mock_segment
                
                mock_create_segment.side_effect = mock_segment_creation
                
                result = initializer._create_spiral_out(5000.0, 4250.0, 5750.0)
                
                self.assertIsNone(result)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_spiral_out_waypoint_exception(self, mock_dijkstra):
        """Test spiral-out route creation with waypoint segment exception"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.distances_from_start = self.mock_distances
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = [(1003, 800)]
            
            with patch.object(initializer, '_create_segment') as mock_create_segment:
                mock_create_segment.side_effect = Exception("Waypoint segment error")
                
                result = initializer._create_spiral_out(5000.0, 4250.0, 5750.0)
                
                self.assertIsNone(result)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_spiral_out_last_waypoint_logic(self, mock_dijkstra):
        """Test spiral-out route creation with last waypoint logic"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.distances_from_start = self.mock_distances
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = [(1003, 800)]
            
            with patch.object(initializer, '_find_nodes_with_return_distance') as mock_find_return:
                mock_find_return.return_value = [(1010, 800)]
                
                with patch.object(initializer, '_create_segment') as mock_create_segment:
                    mock_segment = Mock(spec=RouteSegment)
                    mock_segment.length = 1000.0
                    mock_create_segment.return_value = mock_segment
                    
                    with patch('genetic_algorithm.distance_compliant_population.RouteChromosome') as mock_chromosome_class:
                        mock_chromosome = Mock(spec=RouteChromosome)
                        mock_chromosome.validate_connectivity.return_value = None
                        mock_chromosome_class.return_value = mock_chromosome
                        
                        result = initializer._create_spiral_out(5000.0, 4250.0, 5750.0)
                        
                        self.assertIsNotNone(result)
                        self.assertEqual(result, mock_chromosome)
                        
                        # Should have used find_nodes_with_return_distance for last waypoint
                        mock_find_return.assert_called_once()
                        mock_chromosome.validate_connectivity.assert_called_once()
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_find_nodes_at_distance_empty_result(self, mock_dijkstra):
        """Test find_nodes_at_distance with no matching nodes"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.distances_from_start = self.mock_distances
        
        # Find nodes at distance that doesn't exist
        candidates = initializer._find_nodes_at_distance(10000.0, tolerance=0.1)
        
        self.assertEqual(candidates, [])
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_find_nodes_at_distance_from_node_cutoff_logic(self, mock_dijkstra):
        """Test find_nodes_at_distance_from_node cutoff logic"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        
        with patch('networkx.single_source_dijkstra_path_length') as mock_specific_dijkstra:
            mock_specific_dijkstra.return_value = {1003: 0, 1008: 400}
            
            # Test with large distance (should be limited to 3km)
            candidates = initializer._find_nodes_at_distance_from_node(1003, 5000.0, tolerance=0.3)
            
            # Should call with 3km cutoff, not 7.5km
            mock_specific_dijkstra.assert_called_once_with(
                self.mock_graph, 1003, weight='length', cutoff=3000
            )
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_find_nodes_with_return_distance_no_good_candidates(self, mock_dijkstra):
        """Test find_nodes_with_return_distance with no good candidates"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.distances_from_start = self.mock_distances
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = [(1015, 2000)]  # Too far for return distance
            
            candidates = initializer._find_nodes_with_return_distance(800.0, 800.0, tolerance=0.3)
            
            self.assertEqual(candidates, [])
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_out_and_back_exception_handling(self, mock_dijkstra):
        """Test out-and-back route creation with exception handling"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        
        with patch.object(initializer, '_find_nodes_at_distance') as mock_find_nodes:
            mock_find_nodes.return_value = [(1003, 800)]
            
            with patch.object(initializer, '_create_segment') as mock_create_segment:
                mock_create_segment.side_effect = Exception("Segment creation error")
                
                result = initializer._create_out_and_back(5000.0, 4250.0, 5750.0)
                
                self.assertIsNone(result)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_population_with_extra_fill_chromosome_assignment(self, mock_dijkstra):
        """Test population creation with extra fill chromosome assignment"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        initializer.nodes_by_distance = self.mock_nodes_by_distance
        
        # Mock to simulate needing extra fill
        call_count = 0
        def mock_create_route(strategy, *args):
            nonlocal call_count
            call_count += 1
            
            if call_count <= 5:  # First 5 calls succeed
                mock_chromosome = Mock(spec=RouteChromosome)
                mock_chromosome.creation_method = strategy
                return mock_chromosome
            elif strategy == 'out_and_back':  # Extra fill calls
                mock_chromosome = Mock(spec=RouteChromosome)
                mock_chromosome.creation_method = 'out_and_back_extra'
                return mock_chromosome
            return None
        
        with patch.object(initializer, '_create_route_by_strategy', side_effect=mock_create_route):
            population = initializer.create_population(size=10, target_distance_km=5.0)
            
            # Should have some chromosomes with 'out_and_back_extra' method
            extra_chromosomes = [c for c in population if c.creation_method == 'out_and_back_extra']
            self.assertGreater(len(extra_chromosomes), 0)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_population_size_zero(self, mock_dijkstra):
        """Test population creation with size zero"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        
        population = initializer.create_population(size=0, target_distance_km=5.0)
        
        self.assertEqual(len(population), 0)
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_create_population_negative_size(self, mock_dijkstra):
        """Test population creation with negative size"""
        mock_dijkstra.return_value = self.mock_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        
        population = initializer.create_population(size=-5, target_distance_km=5.0)
        
        self.assertEqual(len(population), 0)
    
    @patch('networkx.single_source_dijkstra_path_length')  
    def test_empty_distances_from_start(self, mock_dijkstra):
        """Test initializer with empty distances from start"""
        mock_dijkstra.return_value = {}
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        
        self.assertEqual(initializer.distances_from_start, {})
        self.assertEqual(initializer.nodes_by_distance, {})
    
    @patch('networkx.single_source_dijkstra_path_length')
    def test_nodes_by_distance_binning_edge_cases(self, mock_dijkstra):
        """Test nodes_by_distance binning with edge case distances"""
        # Test with distances that are exact multiples of 500
        edge_case_distances = {
            1001: 0,
            1002: 500,
            1003: 1000,
            1004: 1500,
            1005: 2000
        }
        mock_dijkstra.return_value = edge_case_distances
        
        initializer = DistanceCompliantPopulationInitializer(
            self.mock_graph, self.start_node
        )
        
        # Verify correct binning
        self.assertIn(0, initializer.nodes_by_distance)  # bin 0 (0m)
        self.assertIn(1, initializer.nodes_by_distance)  # bin 1 (500m)
        self.assertIn(2, initializer.nodes_by_distance)  # bin 2 (1000m)
        self.assertIn(3, initializer.nodes_by_distance)  # bin 3 (1500m)
        self.assertIn(4, initializer.nodes_by_distance)  # bin 4 (2000m)


if __name__ == '__main__':
    unittest.main()