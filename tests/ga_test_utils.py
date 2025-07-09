#!/usr/bin/env python3
"""
GA Test Utilities
Common test setup patterns and utilities for GA unit tests
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from genetic_algorithm import (
    RouteChromosome, RouteSegment, GAOperators, 
    GAFitnessEvaluator, FitnessObjective, PopulationInitializer, 
    GeneticRouteOptimizer
)


class GATestBase(unittest.TestCase):
    """Base test class with common GA test setup"""
    
    def setUp(self):
        """Set up common test fixtures"""
        # Create mock graph with standard test data
        self.mock_graph = self.create_mock_graph()
        
        # Create test route segments
        self.test_segments = self.create_test_segments()
        
        # Create test chromosomes
        self.test_chromosomes = self.create_test_chromosomes()
        
        # Create GA components
        self.operators = GAOperators(self.mock_graph)
        self.fitness_evaluator = GAFitnessEvaluator()
        self.population_initializer = PopulationInitializer(self.mock_graph, start_node=1)
    
    def create_mock_graph(self) -> nx.Graph:
        """Create standardized mock graph for testing
        
        Returns:
            NetworkX graph with standard test nodes and edges
        """
        graph = nx.Graph()
        
        # Add nodes with coordinates and elevation
        nodes = [
            (1, -80.4094, 37.1299, 100.0),
            (2, -80.4090, 37.1300, 110.0),
            (3, -80.4086, 37.1301, 105.0),
            (4, -80.4082, 37.1302, 115.0),
            (5, -80.4078, 37.1303, 95.0),
            (6, -80.4074, 37.1304, 120.0),
            (7, -80.4070, 37.1305, 90.0),
            (8, -80.4066, 37.1306, 130.0)
        ]
        
        for node_id, x, y, elevation in nodes:
            graph.add_node(node_id, x=x, y=y, elevation=elevation)
        
        # Add edges with lengths
        edges = [
            (1, 2, 100.0),
            (2, 3, 150.0),
            (3, 4, 120.0),
            (4, 5, 180.0),
            (5, 1, 200.0),
            (1, 3, 300.0),
            (2, 4, 250.0),
            (3, 5, 220.0),
            (4, 6, 160.0),
            (5, 6, 140.0),
            (6, 7, 110.0),
            (7, 8, 130.0),
            (8, 1, 350.0)
        ]
        
        for start, end, length in edges:
            graph.add_edge(start, end, length=length)
        
        return graph
    
    def create_test_segments(self) -> List[RouteSegment]:
        """Create standardized test route segments
        
        Returns:
            List of RouteSegment objects with calculated properties
        """
        segments = []
        
        # Create segments with different characteristics
        segment_configs = [
            (1, 2, [1, 2]),  # Short segment
            (2, 3, [2, 3]),  # Medium segment
            (3, 4, [3, 4]),  # Uphill segment
            (4, 5, [4, 5]),  # Downhill segment
            (1, 3, [1, 3]),  # Direct connection
            (2, 4, [2, 4]),  # Alternative path
        ]
        
        for start, end, path in segment_configs:
            segment = RouteSegment(start, end, path)
            segment.calculate_properties(self.mock_graph)
            segments.append(segment)
        
        return segments
    
    def create_test_chromosomes(self) -> List[RouteChromosome]:
        """Create standardized test chromosomes
        
        Returns:
            List of RouteChromosome objects
        """
        chromosomes = []
        
        # Simple chromosome with 2 segments
        chromosome1 = RouteChromosome([self.test_segments[0], self.test_segments[1]])
        chromosome1.is_valid = True
        chromosomes.append(chromosome1)
        
        # Complex chromosome with 4 segments
        chromosome2 = RouteChromosome([
            self.test_segments[0], self.test_segments[1], 
            self.test_segments[2], self.test_segments[3]
        ])
        chromosome2.is_valid = True
        chromosomes.append(chromosome2)
        
        # Invalid chromosome for testing
        chromosome3 = RouteChromosome([self.test_segments[0]])
        chromosome3.is_valid = False
        chromosomes.append(chromosome3)
        
        return chromosomes
    
    def create_mock_population(self, size: int = 10) -> List[RouteChromosome]:
        """Create mock population for testing
        
        Args:
            size: Population size
            
        Returns:
            List of RouteChromosome objects
        """
        population = []
        
        for i in range(size):
            # Create varied chromosomes
            if i % 3 == 0:
                # Short routes
                segments = [self.test_segments[0], self.test_segments[1]]
            elif i % 3 == 1:
                # Medium routes  
                segments = [self.test_segments[0], self.test_segments[2], self.test_segments[3]]
            else:
                # Long routes
                segments = [self.test_segments[0], self.test_segments[1], 
                           self.test_segments[2], self.test_segments[3]]
            
            chromosome = RouteChromosome(segments)
            chromosome.is_valid = True
            chromosome.fitness = 0.5 + (i / size) * 0.4  # Varied fitness
            population.append(chromosome)
        
        return population


class GATestUtils:
    """Utility functions for GA testing"""
    
    @staticmethod
    def create_simple_graph(num_nodes: int = 5) -> nx.Graph:
        """Create simple test graph
        
        Args:
            num_nodes: Number of nodes to create
            
        Returns:
            Simple connected graph
        """
        graph = nx.Graph()
        
        # Add nodes in a line
        for i in range(1, num_nodes + 1):
            graph.add_node(i, 
                          x=-80.4094 + i * 0.001, 
                          y=37.1299 + i * 0.001, 
                          elevation=100.0 + i * 10.0)
        
        # Add edges
        for i in range(1, num_nodes):
            graph.add_edge(i, i + 1, length=100.0)
        
        return graph
    
    @staticmethod
    def create_segment_with_properties(start: int, end: int, path: List[int],
                                     length: float = 100.0, elevation_gain: float = 10.0) -> RouteSegment:
        """Create route segment with specific properties
        
        Args:
            start: Start node ID
            end: End node ID
            path: Path nodes
            length: Segment length
            elevation_gain: Elevation gain
            
        Returns:
            RouteSegment with specified properties
        """
        segment = RouteSegment(start, end, path)
        segment.length = length
        segment.elevation_gain = elevation_gain
        segment.is_valid = True
        return segment
    
    @staticmethod
    def create_chromosome_with_fitness(segments: List[RouteSegment], 
                                     fitness: float = 0.5) -> RouteChromosome:
        """Create chromosome with specific fitness
        
        Args:
            segments: Route segments
            fitness: Fitness value
            
        Returns:
            RouteChromosome with specified fitness
        """
        chromosome = RouteChromosome(segments)
        chromosome.fitness = fitness
        chromosome.is_valid = True
        return chromosome
    
    @staticmethod
    def assert_chromosome_valid(test_case: unittest.TestCase, chromosome: RouteChromosome):
        """Assert chromosome is valid
        
        Args:
            test_case: Test case instance
            chromosome: Chromosome to validate
        """
        test_case.assertTrue(chromosome.is_valid, "Chromosome should be valid")
        test_case.assertIsNotNone(chromosome.segments, "Chromosome should have segments")
        test_case.assertGreater(len(chromosome.segments), 0, "Chromosome should have at least one segment")
    
    @staticmethod
    def assert_segment_valid(test_case: unittest.TestCase, segment: RouteSegment):
        """Assert segment is valid
        
        Args:
            test_case: Test case instance
            segment: Segment to validate
        """
        test_case.assertTrue(segment.is_valid, "Segment should be valid")
        test_case.assertIsNotNone(segment.path_nodes, "Segment should have path nodes")
        test_case.assertGreater(len(segment.path_nodes), 0, "Segment should have at least one node")
        test_case.assertGreaterEqual(segment.length, 0, "Segment length should be non-negative")
    
    @staticmethod
    def assert_fitness_valid(test_case: unittest.TestCase, fitness: float):
        """Assert fitness value is valid
        
        Args:
            test_case: Test case instance
            fitness: Fitness value to validate
        """
        test_case.assertIsInstance(fitness, (int, float), "Fitness should be numeric")
        test_case.assertGreaterEqual(fitness, 0, "Fitness should be non-negative")
        test_case.assertLessEqual(fitness, 1, "Fitness should be <= 1")
    
    @staticmethod
    def assert_population_valid(test_case: unittest.TestCase, population: List[RouteChromosome]):
        """Assert population is valid
        
        Args:
            test_case: Test case instance
            population: Population to validate
        """
        test_case.assertIsInstance(population, list, "Population should be a list")
        test_case.assertGreater(len(population), 0, "Population should not be empty")
        
        for i, chromosome in enumerate(population):
            test_case.assertIsInstance(chromosome, RouteChromosome, 
                                     f"Population member {i} should be RouteChromosome")
            GATestUtils.assert_chromosome_valid(test_case, chromosome)


class MockGAComponents:
    """Mock GA components for testing"""
    
    @staticmethod
    def create_mock_fitness_evaluator(objective: str = "elevation", 
                                    target_distance: float = 5.0) -> Mock:
        """Create mock fitness evaluator
        
        Args:
            objective: Fitness objective
            target_distance: Target distance
            
        Returns:
            Mock fitness evaluator
        """
        mock_evaluator = Mock()
        mock_evaluator.objective = objective
        mock_evaluator.target_distance_km = target_distance
        mock_evaluator.evaluate_fitness.return_value = 0.7
        mock_evaluator.evaluate_population.return_value = [0.8, 0.6, 0.7, 0.9]
        return mock_evaluator
    
    @staticmethod
    def create_mock_operators(graph: nx.Graph) -> Mock:
        """Create mock GA operators
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Mock GA operators
        """
        mock_operators = Mock()
        mock_operators.graph = graph
        mock_operators.crossover.return_value = (Mock(), Mock())
        mock_operators.mutate.return_value = Mock()
        mock_operators.select.return_value = [Mock(), Mock()]
        return mock_operators
    
    @staticmethod
    def create_mock_population_initializer(graph: nx.Graph) -> Mock:
        """Create mock population initializer
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Mock population initializer
        """
        mock_initializer = Mock()
        mock_initializer.graph = graph
        mock_initializer.initialize_population.return_value = [Mock() for _ in range(10)]
        return mock_initializer


class GATestDecorators:
    """Decorators for GA testing"""
    
    @staticmethod
    def requires_graph(test_func):
        """Decorator to ensure test has graph setup"""
        def wrapper(self):
            if not hasattr(self, 'mock_graph'):
                self.mock_graph = GATestUtils.create_simple_graph()
            return test_func(self)
        return wrapper
    
    @staticmethod
    def requires_segments(test_func):
        """Decorator to ensure test has segments setup"""
        def wrapper(self):
            if not hasattr(self, 'test_segments'):
                self.test_segments = GATestBase.create_test_segments(self)
            return test_func(self)
        return wrapper


# Common test data
TEST_COORDINATES = [
    (37.1299, -80.4094),
    (37.1300, -80.4090),
    (37.1301, -80.4086),
    (37.1302, -80.4082),
    (37.1303, -80.4078)
]

TEST_ELEVATIONS = [100.0, 110.0, 105.0, 115.0, 95.0, 120.0, 90.0, 130.0]

TEST_DISTANCES = [100.0, 150.0, 120.0, 180.0, 200.0, 300.0, 250.0, 220.0]

TEST_NODE_IDS = [1, 2, 3, 4, 5, 6, 7, 8]

# Test configuration
TEST_CONFIG = {
    'population_size': 10,
    'max_generations': 50,
    'target_distance_km': 5.0,
    'mutation_rate': 0.1,
    'crossover_rate': 0.8,
    'default_objective': 'elevation'
}