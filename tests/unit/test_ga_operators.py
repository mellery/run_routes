#!/usr/bin/env python3
"""
Unit tests for GA Operators
Tests crossover, mutation, and selection operators
"""

import unittest
from unittest.mock import Mock, patch
import networkx as nx
import random
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from genetic_algorithm import GAOperators
from genetic_algorithm import RouteChromosome, RouteSegment

class TestGAOperators(unittest.TestCase):
    """Test GAOperators class functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test graph
        self.graph = nx.Graph()
        self.graph.add_node(1, x=-80.4094, y=37.1299, elevation=100.0)
        self.graph.add_node(2, x=-80.4090, y=37.1300, elevation=110.0)
        self.graph.add_node(3, x=-80.4086, y=37.1301, elevation=105.0)
        self.graph.add_node(4, x=-80.4082, y=37.1302, elevation=115.0)
        self.graph.add_node(5, x=-80.4078, y=37.1303, elevation=95.0)
        
        # Add edges with lengths
        self.graph.add_edge(1, 2, length=100.0)
        self.graph.add_edge(2, 3, length=150.0)
        self.graph.add_edge(3, 4, length=120.0)
        self.graph.add_edge(4, 5, length=180.0)
        self.graph.add_edge(5, 1, length=200.0)
        self.graph.add_edge(1, 3, length=300.0)
        self.graph.add_edge(2, 4, length=250.0)
        
        # Create GA operators
        self.operators = GAOperators(self.graph)
        
        # Create test chromosomes
        self.segment1 = RouteSegment(1, 2, [1, 2])
        self.segment1.calculate_properties(self.graph)
        self.segment2 = RouteSegment(2, 3, [2, 3])
        self.segment2.calculate_properties(self.graph)
        self.segment3 = RouteSegment(3, 4, [3, 4])
        self.segment3.calculate_properties(self.graph)
        
        self.parent1 = RouteChromosome([self.segment1.copy(), self.segment2.copy()])
        self.parent1.fitness = 0.8
        self.parent2 = RouteChromosome([self.segment2.copy(), self.segment3.copy()])
        self.parent2.fitness = 0.6
        
        # Set random seed for reproducible tests
        random.seed(42)
    
    def test_operators_initialization(self):
        """Test GAOperators initialization"""
        operators = GAOperators(self.graph)
        self.assertIsNotNone(operators)
        self.assertEqual(operators.graph, self.graph)
        self.assertIsInstance(operators.segment_cache, dict)
    
    # =============================================================================
    # CROSSOVER OPERATOR TESTS
    # =============================================================================
    
    def test_segment_exchange_crossover_basic(self):
        """Test basic segment exchange crossover"""
        offspring1, offspring2 = self.operators.segment_exchange_crossover(self.parent1, self.parent2)
        
        self.assertIsInstance(offspring1, RouteChromosome)
        self.assertIsInstance(offspring2, RouteChromosome)
        self.assertEqual(offspring1.creation_method, "segment_exchange_crossover")
        self.assertEqual(offspring2.creation_method, "segment_exchange_crossover")
        self.assertIsNotNone(offspring1.parent_ids)
        self.assertIsNotNone(offspring2.parent_ids)
    
    def test_segment_exchange_crossover_no_common_nodes(self):
        """Test crossover with parents having no common nodes"""
        # Create parents with no common nodes
        segment_a = RouteSegment(1, 2, [1, 2])
        segment_a.calculate_properties(self.graph)
        segment_b = RouteSegment(4, 5, [4, 5])
        segment_b.calculate_properties(self.graph)
        
        parent_a = RouteChromosome([segment_a])
        parent_a.fitness = 0.5
        parent_b = RouteChromosome([segment_b])
        parent_b.fitness = 0.7
        
        offspring1, offspring2 = self.operators.segment_exchange_crossover(parent_a, parent_b)
        
        # Should return copies of parents when no crossover possible
        self.assertEqual(len(offspring1.segments), len(parent_a.segments))
        self.assertEqual(len(offspring2.segments), len(parent_b.segments))
    
    def test_segment_exchange_crossover_rate(self):
        """Test crossover rate parameter"""
        # Test with 0% crossover rate
        offspring1, offspring2 = self.operators.segment_exchange_crossover(
            self.parent1, self.parent2, crossover_rate=0.0
        )
        
        # Should return copies of original parents
        self.assertEqual(len(offspring1.segments), len(self.parent1.segments))
        self.assertEqual(len(offspring2.segments), len(self.parent2.segments))
    
    def test_path_splice_crossover_basic(self):
        """Test basic path splice crossover"""
        offspring1, offspring2 = self.operators.path_splice_crossover(self.parent1, self.parent2)
        
        self.assertIsInstance(offspring1, RouteChromosome)
        self.assertIsInstance(offspring2, RouteChromosome)
        # Creation method may include "_repaired" suffix due to segment usage validation
        self.assertIn("path_splice_crossover", offspring1.creation_method)
        self.assertIn("path_splice_crossover", offspring2.creation_method)
    
    def test_path_splice_crossover_empty_parents(self):
        """Test path splice crossover with empty parents"""
        empty_parent = RouteChromosome([])
        empty_parent.fitness = 0.0
        
        offspring1, offspring2 = self.operators.path_splice_crossover(empty_parent, self.parent1)
        
        # Should handle empty parents gracefully
        self.assertIsInstance(offspring1, RouteChromosome)
        self.assertIsInstance(offspring2, RouteChromosome)
    
    # =============================================================================
    # MUTATION OPERATOR TESTS
    # =============================================================================
    
    def test_segment_replacement_mutation_basic(self):
        """Test basic segment replacement mutation"""
        original_length = len(self.parent1.segments)
        mutated = self.operators.segment_replacement_mutation(self.parent1, mutation_rate=1.0)
        
        self.assertIsInstance(mutated, RouteChromosome)
        self.assertEqual(len(mutated.segments), original_length)
        # Should not be the same object
        self.assertIsNot(mutated, self.parent1)
    
    def test_segment_replacement_mutation_rate(self):
        """Test mutation rate parameter"""
        # Test with 0% mutation rate
        mutated = self.operators.segment_replacement_mutation(self.parent1, mutation_rate=0.0)
        
        # Should return unchanged copy
        self.assertEqual(len(mutated.segments), len(self.parent1.segments))
        self.assertIsNot(mutated, self.parent1)
    
    def test_segment_replacement_mutation_empty_chromosome(self):
        """Test mutation with empty chromosome"""
        empty_chromosome = RouteChromosome([])
        mutated = self.operators.segment_replacement_mutation(empty_chromosome)
        
        self.assertIsInstance(mutated, RouteChromosome)
        self.assertEqual(len(mutated.segments), 0)
    
    def test_route_extension_mutation_basic(self):
        """Test basic route extension mutation"""
        target_distance = 2.0  # 2km
        mutated = self.operators.route_extension_mutation(
            self.parent1, target_distance, mutation_rate=1.0
        )
        
        self.assertIsInstance(mutated, RouteChromosome)
        self.assertIsNot(mutated, self.parent1)
    
    def test_route_extension_mutation_already_optimal(self):
        """Test mutation when route is already near target distance"""
        # Current route distance in km
        current_distance = self.parent1.get_total_distance() / 1000
        
        # Set target very close to current distance
        mutated = self.operators.route_extension_mutation(
            self.parent1, current_distance, mutation_rate=1.0
        )
        
        # Should return copy without significant changes
        self.assertIsInstance(mutated, RouteChromosome)
    
    def test_route_extension_mutation_rate(self):
        """Test route extension mutation rate"""
        # Test with 0% mutation rate
        mutated = self.operators.route_extension_mutation(
            self.parent1, 5.0, mutation_rate=0.0
        )
        
        # Should return unchanged copy
        self.assertEqual(len(mutated.segments), len(self.parent1.segments))
    

    def test_tournament_selection_basic(self):
        """Test basic tournament selection"""
        population = [self.parent1, self.parent2]
        selected = self.operators.tournament_selection(population, tournament_size=2)
        
        self.assertIsInstance(selected, RouteChromosome)
        self.assertIn(selected, population)
        # Should select parent1 (higher fitness)
        self.assertEqual(selected.fitness, 0.8)
    
    def test_tournament_selection_single_individual(self):
        """Test tournament selection with single individual"""
        population = [self.parent1]
        selected = self.operators.tournament_selection(population, tournament_size=5)
        
        self.assertEqual(selected, self.parent1)
    
    def test_tournament_selection_empty_population(self):
        """Test tournament selection with empty population"""
        with self.assertRaises(ValueError):
            self.operators.tournament_selection([], tournament_size=3)
    
    def test_tournament_selection_size_larger_than_population(self):
        """Test tournament selection with tournament size larger than population"""
        population = [self.parent1, self.parent2]
        selected = self.operators.tournament_selection(population, tournament_size=10)
        
        # Should work and select from available individuals
        self.assertIsInstance(selected, RouteChromosome)
        self.assertIn(selected, population)
    
    def test_elitism_selection_basic(self):
        """Test basic elitism selection"""
        # Create population with varied fitness
        chromo1 = self.parent1.copy()
        chromo1.fitness = 0.9
        chromo2 = self.parent2.copy()
        chromo2.fitness = 0.7
        chromo3 = self.parent1.copy()
        chromo3.fitness = 0.5
        
        population = [chromo3, chromo1, chromo2]  # Mixed order
        elite = self.operators.elitism_selection(population, elite_size=2)
        
        self.assertEqual(len(elite), 2)
        self.assertEqual(elite[0].fitness, 0.9)  # Highest fitness first
        self.assertEqual(elite[1].fitness, 0.7)  # Second highest
    
    def test_elitism_selection_empty_population(self):
        """Test elitism selection with empty population"""
        elite = self.operators.elitism_selection([], elite_size=5)
        
        self.assertEqual(len(elite), 0)
    
    def test_elitism_selection_elite_size_larger_than_population(self):
        """Test elitism selection with elite size larger than population"""
        population = [self.parent1, self.parent2]
        elite = self.operators.elitism_selection(population, elite_size=10)
        
        # Should return entire population
        self.assertEqual(len(elite), 2)
    
    def test_diversity_selection_basic(self):
        """Test basic diversity selection"""
        # Create population with varied characteristics
        chromo1 = self.parent1.copy()
        chromo1.fitness = 0.8
        chromo2 = self.parent2.copy()
        chromo2.fitness = 0.6
        
        # Add a chromosome with different distance
        segment_long = RouteSegment(1, 5, [1, 5])  # Longer segment
        segment_long.calculate_properties(self.graph)
        chromo3 = RouteChromosome([segment_long])
        chromo3.fitness = 0.4
        
        population = [chromo1, chromo2, chromo3]
        selected = self.operators.diversity_selection(population, selection_size=2)
        
        self.assertEqual(len(selected), 2)
        self.assertIsInstance(selected[0], RouteChromosome)
        self.assertIsInstance(selected[1], RouteChromosome)
        # First selected should be highest fitness
        self.assertEqual(selected[0].fitness, 0.8)
    
    def test_diversity_selection_empty_population(self):
        """Test diversity selection with empty population"""
        selected = self.operators.diversity_selection([], selection_size=5)
        
        self.assertEqual(len(selected), 0)
    
    def test_diversity_selection_selection_size_larger_than_population(self):
        """Test diversity selection with selection size larger than population"""
        population = [self.parent1, self.parent2]
        selected = self.operators.diversity_selection(population, selection_size=10)
        
        # Should return entire population
        self.assertEqual(len(selected), 2)
    
    def test_survival_selection_basic(self):
        """Test basic survival selection"""
        # Create population with varied fitness
        chromo1 = self.parent1.copy()
        chromo1.fitness = 0.9
        chromo2 = self.parent2.copy()
        chromo2.fitness = 0.7
        chromo3 = self.parent1.copy()
        chromo3.fitness = 0.5
        chromo4 = self.parent2.copy()
        chromo4.fitness = 0.3
        
        population = [chromo1, chromo2, chromo3, chromo4]
        fitness_scores = [0.9, 0.7, 0.5, 0.3]
        
        # Select top 50% survivors
        survivors, survivor_fitness = self.operators.survival_selection(
            population, fitness_scores, survival_rate=0.5
        )
        
        self.assertEqual(len(survivors), 2)
        self.assertEqual(len(survivor_fitness), 2)
        self.assertEqual(survivor_fitness[0], 0.9)  # Best fitness first
        self.assertEqual(survivor_fitness[1], 0.7)  # Second best
    
    def test_survival_selection_fitness_threshold(self):
        """Test survival selection with fitness threshold"""
        # Create population where some don't meet threshold
        chromo1 = self.parent1.copy()
        chromo1.fitness = 0.8
        chromo2 = self.parent2.copy()
        chromo2.fitness = 0.6
        chromo3 = self.parent1.copy()
        chromo3.fitness = 0.05  # Below threshold
        
        population = [chromo1, chromo2, chromo3]
        fitness_scores = [0.8, 0.6, 0.05]
        
        # Filter with threshold of 0.1 and 100% survival rate to keep all above threshold
        survivors, survivor_fitness = self.operators.survival_selection(
            population, fitness_scores, min_fitness_threshold=0.1, survival_rate=1.0
        )
        
        # Should only keep first two (those above threshold)
        self.assertEqual(len(survivors), 2)
        self.assertEqual(survivor_fitness[0], 0.8)
        self.assertEqual(survivor_fitness[1], 0.6)
    
    def test_survival_selection_all_below_threshold(self):
        """Test survival selection when all below threshold"""
        population = [self.parent1, self.parent2]
        fitness_scores = [0.05, 0.03]  # All below threshold
        
        # Should keep best one even if below threshold
        survivors, survivor_fitness = self.operators.survival_selection(
            population, fitness_scores, min_fitness_threshold=0.1
        )
        
        self.assertEqual(len(survivors), 1)
        self.assertEqual(survivor_fitness[0], 0.05)  # Best of the bad ones
    
    def test_survival_selection_empty_population(self):
        """Test survival selection with empty population"""
        survivors, survivor_fitness = self.operators.survival_selection([], [])
        
        self.assertEqual(len(survivors), 0)
        self.assertEqual(len(survivor_fitness), 0)
    
    def test_survival_selection_mismatched_lengths(self):
        """Test survival selection with mismatched input lengths"""
        population = [self.parent1, self.parent2]
        fitness_scores = [0.8]  # Wrong length
        
        with self.assertRaises(ValueError):
            self.operators.survival_selection(population, fitness_scores)
    
    # =============================================================================
    # HELPER METHOD TESTS
    # =============================================================================
    

class TestGAOperatorsIntegration(unittest.TestCase):
    """Integration tests for GA operators"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        # Create more complex test graph
        self.graph = nx.Graph()
        
        # Add nodes in a grid pattern
        for i in range(5):
            for j in range(5):
                node_id = i * 5 + j + 1
                self.graph.add_node(
                    node_id,
                    x=-80.4094 + j * 0.001,
                    y=37.1299 + i * 0.001,
                    elevation=100 + (i + j) * 5
                )
        
        # Add edges to create connected graph
        for i in range(5):
            for j in range(5):
                node_id = i * 5 + j + 1
                # Connect to right neighbor
                if j < 4:
                    right_neighbor = i * 5 + (j + 1) + 1
                    self.graph.add_edge(node_id, right_neighbor, length=100)
                # Connect to bottom neighbor
                if i < 4:
                    bottom_neighbor = (i + 1) * 5 + j + 1
                    self.graph.add_edge(node_id, bottom_neighbor, length=100)
        
        self.operators = GAOperators(self.graph)
        
        # Create complex test chromosomes
        segments1 = [
            RouteSegment(1, 2, [1, 2]),
            RouteSegment(2, 7, [2, 7]),
            RouteSegment(7, 12, [7, 12])
        ]
        
        segments2 = [
            RouteSegment(1, 6, [1, 6]),
            RouteSegment(6, 11, [6, 11]),
            RouteSegment(11, 16, [11, 16])
        ]
        
        for segment in segments1:
            segment.calculate_properties(self.graph)
        for segment in segments2:
            segment.calculate_properties(self.graph)
        
        self.complex_parent1 = RouteChromosome(segments1)
        self.complex_parent1.fitness = 0.85
        self.complex_parent2 = RouteChromosome(segments2)
        self.complex_parent2.fitness = 0.75
    
    def test_full_crossover_workflow(self):
        """Test complete crossover workflow with complex chromosomes"""
        offspring1, offspring2 = self.operators.segment_exchange_crossover(
            self.complex_parent1, self.complex_parent2
        )
        
        # Validate offspring
        self.assertTrue(offspring1.validate_connectivity())
        self.assertTrue(offspring2.validate_connectivity())
        self.assertGreater(len(offspring1.segments), 0)
        self.assertGreater(len(offspring2.segments), 0)
        
        # Check metadata
        self.assertEqual(offspring1.creation_method, "segment_exchange_crossover")
        self.assertEqual(offspring2.creation_method, "segment_exchange_crossover")
    

