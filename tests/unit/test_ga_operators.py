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

from ga_operators import GAOperators
from ga_chromosome import RouteChromosome, RouteSegment


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
        self.assertEqual(offspring1.creation_method, "path_splice_crossover")
        self.assertEqual(offspring2.creation_method, "path_splice_crossover")
    
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
    
    def test_elevation_bias_mutation_basic(self):
        """Test basic elevation bias mutation"""
        mutated = self.operators.elevation_bias_mutation(
            self.parent1, objective="elevation", mutation_rate=1.0
        )
        
        self.assertIsInstance(mutated, RouteChromosome)
        self.assertIsNot(mutated, self.parent1)
    
    def test_elevation_bias_mutation_wrong_objective(self):
        """Test elevation bias mutation with non-elevation objective"""
        mutated = self.operators.elevation_bias_mutation(
            self.parent1, objective="distance", mutation_rate=1.0
        )
        
        # Should return unchanged copy for non-elevation objectives
        self.assertEqual(len(mutated.segments), len(self.parent1.segments))
    
    def test_elevation_bias_mutation_rate(self):
        """Test elevation bias mutation rate"""
        # Test with 0% mutation rate
        mutated = self.operators.elevation_bias_mutation(
            self.parent1, objective="elevation", mutation_rate=0.0
        )
        
        # Should return unchanged copy
        self.assertEqual(len(mutated.segments), len(self.parent1.segments))
    
    # =============================================================================
    # SELECTION OPERATOR TESTS
    # =============================================================================
    
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
    
    # =============================================================================
    # HELPER METHOD TESTS
    # =============================================================================
    
    def test_find_common_nodes(self):
        """Test finding common nodes between parents"""
        common_nodes = self.operators._find_common_nodes(self.parent1, self.parent2)
        
        self.assertIsInstance(common_nodes, list)
        # Should find node 2 and 3 as common (from overlapping segments)
        self.assertIn(2, common_nodes)
        self.assertIn(3, common_nodes)
    
    def test_find_common_nodes_no_overlap(self):
        """Test finding common nodes with no overlap"""
        # Create non-overlapping parents
        segment_a = RouteSegment(1, 2, [1, 2])
        segment_a.calculate_properties(self.graph)
        segment_b = RouteSegment(4, 5, [4, 5])
        segment_b.calculate_properties(self.graph)
        
        parent_a = RouteChromosome([segment_a])
        parent_b = RouteChromosome([segment_b])
        
        common_nodes = self.operators._find_common_nodes(parent_a, parent_b)
        
        self.assertEqual(len(common_nodes), 0)
    
    def test_create_segment(self):
        """Test segment creation"""
        segment = self.operators._create_segment(1, 3)
        
        self.assertIsInstance(segment, RouteSegment)
        self.assertEqual(segment.start_node, 1)
        self.assertEqual(segment.end_node, 3)
        self.assertGreater(segment.length, 0)
    
    def test_create_segment_no_path(self):
        """Test segment creation with no path"""
        # Add isolated node
        self.graph.add_node(99, x=-80.5000, y=37.2000, elevation=120.0)
        
        segment = self.operators._create_segment(1, 99)
        
        self.assertIsNone(segment)
    
    def test_create_segment_caching(self):
        """Test segment creation caching"""
        # Create same segment twice
        segment1 = self.operators._create_segment(1, 3)
        segment2 = self.operators._create_segment(1, 3)
        
        self.assertIsInstance(segment1, RouteSegment)
        self.assertIsInstance(segment2, RouteSegment)
        # Should be copies, not same object
        self.assertIsNot(segment1, segment2)
        # But should have same properties
        self.assertEqual(segment1.length, segment2.length)
    
    def test_get_elevation_neighbors(self):
        """Test getting elevation neighbors"""
        neighbors = self.operators._get_elevation_neighbors(1)  # elevation 100
        
        self.assertIsInstance(neighbors, list)
        # Should include node 2 (elevation 110) but not node 5 (elevation 95)
        self.assertIn(2, neighbors)
        self.assertNotIn(5, neighbors)
    
    def test_get_elevation_neighbors_no_higher_neighbors(self):
        """Test getting elevation neighbors from highest point"""
        neighbors = self.operators._get_elevation_neighbors(4)  # elevation 115 (highest)
        
        # Should return empty list as no neighbors are higher
        self.assertEqual(len(neighbors), 0)
    
    def test_repair_chromosome(self):
        """Test chromosome repair functionality"""
        # Create chromosome that might need repair
        chromosome = self.parent1.copy()
        
        # Invalidate cache to test repair
        chromosome._invalidate_cache()
        
        repaired = self.operators._repair_chromosome(chromosome)
        
        self.assertIsInstance(repaired, RouteChromosome)
        self.assertTrue(repaired.is_valid)


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
    
    def test_full_mutation_workflow(self):
        """Test complete mutation workflow with complex chromosomes"""
        # Test all mutation types
        mutated1 = self.operators.segment_replacement_mutation(self.complex_parent1)
        mutated2 = self.operators.route_extension_mutation(self.complex_parent1, 2.0)
        mutated3 = self.operators.elevation_bias_mutation(self.complex_parent1, "elevation")
        
        # Validate all mutations
        for mutated in [mutated1, mutated2, mutated3]:
            self.assertIsInstance(mutated, RouteChromosome)
            self.assertTrue(mutated.validate_connectivity())
            self.assertGreater(len(mutated.segments), 0)
    
    def test_population_evolution_cycle(self):
        """Test complete evolution cycle with selection and operators"""
        # Create initial population
        population = [
            self.complex_parent1,
            self.complex_parent2,
            self.complex_parent1.copy(),
            self.complex_parent2.copy()
        ]
        
        # Set different fitness values
        for i, chromo in enumerate(population):
            chromo.fitness = 0.9 - i * 0.1
        
        # Selection phase
        elite = self.operators.elitism_selection(population, elite_size=2)
        selected_parents = [
            self.operators.tournament_selection(population) for _ in range(4)
        ]
        
        # Crossover phase
        offspring = []
        for i in range(0, len(selected_parents), 2):
            if i + 1 < len(selected_parents):
                child1, child2 = self.operators.segment_exchange_crossover(
                    selected_parents[i], selected_parents[i + 1]
                )
                offspring.extend([child1, child2])
        
        # Mutation phase
        mutated_offspring = []
        for child in offspring:
            mutated = self.operators.segment_replacement_mutation(child, mutation_rate=0.3)
            mutated_offspring.append(mutated)
        
        # Validate entire evolution cycle
        self.assertGreater(len(elite), 0)
        self.assertGreater(len(offspring), 0)
        self.assertGreater(len(mutated_offspring), 0)
        
        # All individuals should be valid
        for individual in elite + offspring + mutated_offspring:
            self.assertIsInstance(individual, RouteChromosome)
            self.assertTrue(individual.validate_connectivity())


if __name__ == '__main__':
    unittest.main()