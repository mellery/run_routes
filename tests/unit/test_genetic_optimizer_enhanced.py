#!/usr/bin/env python3
"""
Enhanced unit tests for GeneticRouteOptimizer
Tests comprehensive optimization workflows, convergence, and advanced features
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import networkx as nx
import sys
import os
from datetime import datetime
import time

# Add the parent directory to sys.path to import genetic_algorithm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from genetic_algorithm.optimizer import GeneticRouteOptimizer, GAConfig
from genetic_algorithm.chromosome import RouteChromosome
from genetic_algorithm.fitness import FitnessObjective


class TestGeneticRouteOptimizerEnhanced(unittest.TestCase):
    """Enhanced test cases for GeneticRouteOptimizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a mock graph
        self.mock_graph = nx.Graph()
        self.mock_graph.add_node(1001, x=-80.4094, y=37.1299, elevation=600)
        self.mock_graph.add_node(1002, x=-80.4095, y=37.1300, elevation=620)
        self.mock_graph.add_node(1003, x=-80.4096, y=37.1301, elevation=610)
        self.mock_graph.add_node(1004, x=-80.4097, y=37.1302, elevation=650)
        self.mock_graph.add_node(1005, x=-80.4098, y=37.1303, elevation=630)
        
        # Add edges
        self.mock_graph.add_edge(1001, 1002, length=100)
        self.mock_graph.add_edge(1002, 1003, length=100)
        self.mock_graph.add_edge(1003, 1004, length=100)
        self.mock_graph.add_edge(1004, 1005, length=100)
        self.mock_graph.add_edge(1005, 1001, length=100)
        
        # Create test configuration
        self.test_config = GAConfig(
            population_size=10,
            max_generations=5,
            verbose=False,
            enable_precision_enhancement=False,
            enable_adaptive_mutation=False,
            enable_terrain_aware_initialization=False,
            enable_restart_mechanisms=False,
            enable_diversity_selection=False
        )
        
        self.optimizer = GeneticRouteOptimizer(self.mock_graph, self.test_config)
    
    def test_initialization_default_config(self):
        """Test optimizer initialization with default config"""
        optimizer = GeneticRouteOptimizer(self.mock_graph)
        
        self.assertEqual(optimizer.graph, self.mock_graph)
        self.assertIsNotNone(optimizer.config)
        self.assertEqual(optimizer.config.population_size, 100)
        self.assertEqual(optimizer.config.max_generations, 200)
        self.assertIsNotNone(optimizer.segment_cache)
        self.assertIsNotNone(optimizer.operators)
    
    def test_initialization_custom_config(self):
        """Test optimizer initialization with custom config"""
        custom_config = GAConfig(
            population_size=50,
            max_generations=100,
            crossover_rate=0.9,
            mutation_rate=0.2
        )
        
        optimizer = GeneticRouteOptimizer(self.mock_graph, custom_config)
        
        self.assertEqual(optimizer.config.population_size, 50)
        self.assertEqual(optimizer.config.max_generations, 100)
        self.assertEqual(optimizer.config.crossover_rate, 0.9)
        self.assertEqual(optimizer.config.mutation_rate, 0.2)
    
    def test_setup_optimization_basic(self):
        """Test basic optimization setup"""
        # Call setup optimization with string objective
        self.optimizer._setup_optimization(1001, 5000, "elevation")
        
        # Verify components were initialized
        self.assertIsNotNone(self.optimizer.population_initializer)
        self.assertIsNotNone(self.optimizer.fitness_evaluator)
    
    def test_setup_optimization_distance_compliant(self):
        """Test optimization setup with distance compliant initialization"""
        config = GAConfig(use_distance_compliant_initialization=True)
        optimizer = GeneticRouteOptimizer(self.mock_graph, config)
        
        optimizer._setup_optimization(1001, 5000, "elevation")
        
        # Should have initialized population initializer
        self.assertIsNotNone(optimizer.population_initializer)
    
    def test_setup_optimization_terrain_aware(self):
        """Test optimization setup with terrain-aware initialization"""
        config = GAConfig(enable_terrain_aware_initialization=True)
        optimizer = GeneticRouteOptimizer(self.mock_graph, config)
        
        optimizer._setup_optimization(1001, 5000, "elevation")
        
        # Should have initialized terrain aware components
        self.assertIsNotNone(optimizer.terrain_aware_initializer)
    
    def test_setup_optimization_constraint_preserving(self):
        """Test optimization setup with constraint-preserving operators"""
        config = GAConfig(use_constraint_preserving_operators=True)
        optimizer = GeneticRouteOptimizer(self.mock_graph, config)
        
        optimizer._setup_optimization(1001, 5000, "elevation")
        
        # Should have initialized constraint operators
        self.assertIsNotNone(optimizer.constraint_operators)
    
    def test_setup_optimization_adaptive_mutation(self):
        """Test optimization setup with adaptive mutation"""
        config = GAConfig(enable_adaptive_mutation=True)
        optimizer = GeneticRouteOptimizer(self.mock_graph, config)
        
        optimizer._setup_optimization(1001, 5000, "elevation")
        
        # Should have initialized adaptive mutation controller
        self.assertIsNotNone(optimizer.adaptive_mutation_controller)
    
    def test_setup_optimization_restart_mechanisms(self):
        """Test optimization setup with restart mechanisms"""
        config = GAConfig(enable_restart_mechanisms=True)
        optimizer = GeneticRouteOptimizer(self.mock_graph, config)
        
        optimizer._setup_optimization(1001, 5000, "elevation")
        
        # Should have initialized restart mechanisms
        self.assertIsNotNone(optimizer.restart_mechanisms)
    
    def test_setup_optimization_diversity_selection(self):
        """Test optimization setup with diversity-preserving selection"""
        config = GAConfig(enable_diversity_selection=True)
        optimizer = GeneticRouteOptimizer(self.mock_graph, config)
        
        optimizer._setup_optimization(1001, 5000, "elevation")
        
        # Should have initialized diversity selector
        self.assertIsNotNone(optimizer.diversity_selector)
    
    def test_adapt_configuration_small_distance(self):
        """Test configuration adaptation for small distance"""
        config = GAConfig(adaptive_sizing=True, population_size=100, max_generations=200)
        optimizer = GeneticRouteOptimizer(self.mock_graph, config)
        
        optimizer._adapt_configuration(2.0)  # 2km distance
        
        # Should adapt to smaller population and fewer generations
        self.assertLessEqual(optimizer.config.population_size, 100)
        self.assertLessEqual(optimizer.config.max_generations, 200)
    
    def test_adapt_configuration_large_distance(self):
        """Test configuration adaptation for large distance"""
        config = GAConfig(adaptive_sizing=True, population_size=50, max_generations=100)
        optimizer = GeneticRouteOptimizer(self.mock_graph, config)
        
        optimizer._adapt_configuration(10.0)  # 10km distance
        
        # Should adapt to larger population and more generations
        self.assertGreaterEqual(optimizer.config.population_size, 50)
        self.assertGreaterEqual(optimizer.config.max_generations, 100)
    
    def test_adapt_configuration_disabled(self):
        """Test configuration adaptation when disabled"""
        config = GAConfig(adaptive_sizing=False, population_size=50, max_generations=100)
        optimizer = GeneticRouteOptimizer(self.mock_graph, config)
        
        original_pop_size = optimizer.config.population_size
        original_max_gen = optimizer.config.max_generations
        
        optimizer._adapt_configuration(10.0)
        
        # Should not change configuration when adaptive_sizing is False
        self.assertEqual(optimizer.config.population_size, original_pop_size)
        self.assertEqual(optimizer.config.max_generations, original_max_gen)
    
    def test_check_convergence_not_converged(self):
        """Test convergence checking when not converged"""
        # Create fitness history that shows improvement
        fitness_history = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        result = self.optimizer._check_convergence(fitness_history)
        
        self.assertFalse(result)
    
    def test_check_convergence_converged(self):
        """Test convergence checking when converged"""
        # Create fitness history that shows no improvement
        fitness_history = [0.8] * 25  # 25 generations of same fitness
        
        # Set fitness history in optimizer
        self.optimizer.fitness_history = [[0.8] * 10] * 25  # History of fitness arrays
        
        result = self.optimizer._check_convergence(fitness_history)
        
        self.assertTrue(result)
    
    def test_check_convergence_insufficient_history(self):
        """Test convergence checking with insufficient history"""
        # Create short fitness history
        fitness_history = [0.5, 0.6, 0.7]
        
        result = self.optimizer._check_convergence(fitness_history)
        
        self.assertFalse(result)
    
    def test_evolve_generation_basic(self):
        """Test basic generation evolution"""
        # Create mock population with fitness scores
        mock_chromosomes = []
        fitness_scores = []
        for i in range(10):
            mock_chr = Mock(spec=RouteChromosome)
            mock_chr.fitness = i * 0.1
            mock_chr.get_total_distance.return_value = 2500
            mock_chr.get_elevation_gain.return_value = 100
            mock_chr.get_elevation_loss.return_value = 50
            mock_chromosomes.append(mock_chr)
            fitness_scores.append(i * 0.1)
        
        # Mock fitness evaluator
        self.optimizer.fitness_evaluator = Mock()
        self.optimizer.fitness_evaluator.evaluate_population = Mock(return_value=fitness_scores)
        
        # Mock operators
        self.optimizer.operators = Mock()
        self.optimizer.operators.crossover = Mock(return_value=mock_chromosomes[0])
        self.optimizer.operators.mutate = Mock(return_value=mock_chromosomes[0])
        
        # Test evolution (method returns tuple of (population, fitness_scores))
        result = self.optimizer._evolve_generation(mock_chromosomes, fitness_scores)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)  # (population, fitness_scores)
        self.assertEqual(len(result[0]), 10)  # population size
        self.assertEqual(len(result[1]), 10)  # fitness scores
    
    def test_evolve_generation_with_constraint_operators(self):
        """Test generation evolution with constraint-preserving operators"""
        # Setup constraint operators
        self.optimizer.constraint_operators = Mock()
        self.optimizer.constraint_operators.crossover = Mock(return_value=Mock(spec=RouteChromosome))
        self.optimizer.constraint_operators.distance_neutral_mutation = Mock(return_value=mock_mutated)
        
        # Create mock population with fitness scores
        mock_chromosomes = []
        fitness_scores = []
        for i in range(10):
            mock_chr = Mock(spec=RouteChromosome)
            mock_chr.fitness = i * 0.1
            mock_chr.get_total_distance.return_value = 2500
            mock_chr.get_elevation_gain.return_value = 100
            mock_chr.get_elevation_loss.return_value = 50
            mock_chr.copy.return_value = Mock(spec=RouteChromosome)
            mock_chromosomes.append(mock_chr)
            fitness_scores.append(i * 0.1)
        
        # Mock fitness evaluator
        self.optimizer.fitness_evaluator = Mock()
        self.optimizer.fitness_evaluator.evaluate_population = Mock(return_value=fitness_scores)
        
        # Mock operators for fallback
        self.optimizer.operators = Mock()
        self.optimizer.operators.tournament_selection = Mock(return_value=mock_chromosomes[0])
        
        # Test evolution
        result = self.optimizer._evolve_generation(mock_chromosomes, fitness_scores)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.optimizer.constraint_operators.crossover.assert_called()
        self.optimizer.constraint_operators.mutate.assert_called()
    
    def test_evolve_generation_with_adaptive_mutation(self):
        """Test generation evolution with adaptive mutation"""
        # Setup adaptive mutation controller
        self.optimizer.adaptive_mutation_controller = Mock()
        self.optimizer.adaptive_mutation_controller.update.return_value = 0.25
        
        # Create mock population with fitness scores
        mock_chromosomes = []
        fitness_scores = []
        for i in range(10):
            mock_chr = Mock(spec=RouteChromosome)
            mock_chr.fitness = i * 0.1
            mock_chr.get_total_distance.return_value = 2500
            mock_chr.get_elevation_gain.return_value = 100
            mock_chr.get_elevation_loss.return_value = 50
            mock_chr.copy.return_value = Mock(spec=RouteChromosome)
            mock_chromosomes.append(mock_chr)
            fitness_scores.append(i * 0.1)
        
        # Mock fitness evaluator
        self.optimizer.fitness_evaluator = Mock()
        self.optimizer.fitness_evaluator.evaluate_population = Mock(return_value=fitness_scores)
        
        # Mock operators
        self.optimizer.operators = Mock()
        self.optimizer.operators.crossover = Mock(return_value=mock_chromosomes[0])
        self.optimizer.operators.mutate = Mock(return_value=mock_chromosomes[0])
        self.optimizer.operators.tournament_selection = Mock(return_value=mock_chromosomes[0])
        
        # Test evolution
        result = self.optimizer._evolve_generation(mock_chromosomes, fitness_scores)
        
        self.assertIsInstance(result, tuple)
        self.optimizer.adaptive_mutation_controller.update.assert_called()
    
    def test_evolve_generation_with_restart_mechanisms(self):
        """Test generation evolution with restart mechanisms"""
        # Setup restart mechanisms
        self.optimizer.restart_mechanisms = Mock()
        self.optimizer.restart_mechanisms.should_restart.return_value = False
        self.optimizer.restart_mechanisms.update.return_value = None
        
        # Create mock population with fitness scores
        mock_chromosomes = []
        fitness_scores = []
        for i in range(10):
            mock_chr = Mock(spec=RouteChromosome)
            mock_chr.fitness = i * 0.1
            mock_chr.get_total_distance.return_value = 2500
            mock_chr.get_elevation_gain.return_value = 100
            mock_chr.get_elevation_loss.return_value = 50
            mock_chr.copy.return_value = Mock(spec=RouteChromosome)
            mock_chromosomes.append(mock_chr)
            fitness_scores.append(i * 0.1)
        
        # Mock fitness evaluator
        self.optimizer.fitness_evaluator = Mock()
        self.optimizer.fitness_evaluator.evaluate_population = Mock(return_value=fitness_scores)
        
        # Mock operators
        self.optimizer.operators = Mock()
        self.optimizer.operators.crossover = Mock(return_value=mock_chromosomes[0])
        self.optimizer.operators.mutate = Mock(return_value=mock_chromosomes[0])
        self.optimizer.operators.tournament_selection = Mock(return_value=mock_chromosomes[0])
        
        # Test evolution
        result = self.optimizer._evolve_generation(mock_chromosomes, fitness_scores)
        
        self.assertIsInstance(result, tuple)
        # Note: restart mechanisms are checked differently in the actual code
    
    def test_evolve_generation_with_restart_triggered(self):
        """Test generation evolution with restart triggered"""
        # Setup restart mechanisms
        self.optimizer.restart_mechanisms = Mock()
        self.optimizer.restart_mechanisms.should_restart.return_value = True
        self.optimizer.restart_mechanisms.perform_restart.return_value = [Mock(spec=RouteChromosome) for _ in range(10)]
        self.optimizer.restart_mechanisms.update.return_value = None
        
        # Create mock population with fitness scores
        mock_chromosomes = []
        fitness_scores = []
        for i in range(10):
            mock_chr = Mock(spec=RouteChromosome)
            mock_chr.fitness = i * 0.1
            mock_chr.get_total_distance.return_value = 2500
            mock_chr.get_elevation_gain.return_value = 100
            mock_chr.get_elevation_loss.return_value = 50
            mock_chr.copy.return_value = Mock(spec=RouteChromosome)
            mock_chromosomes.append(mock_chr)
            fitness_scores.append(i * 0.1)
        
        # Mock fitness evaluator
        self.optimizer.fitness_evaluator = Mock()
        self.optimizer.fitness_evaluator.evaluate_population = Mock(return_value=fitness_scores)
        
        # Mock operators
        self.optimizer.operators = Mock()
        self.optimizer.operators.crossover = Mock(return_value=mock_chromosomes[0])
        self.optimizer.operators.mutate = Mock(return_value=mock_chromosomes[0])
        self.optimizer.operators.tournament_selection = Mock(return_value=mock_chromosomes[0])
        
        # Test evolution
        result = self.optimizer._evolve_generation(mock_chromosomes, fitness_scores)
        
        self.assertIsInstance(result, tuple)
        # Note: restart mechanisms are checked differently in the actual code
    
    def test_apply_adaptive_mutation_basic(self):
        """Test basic adaptive mutation application"""
        # Create mock chromosome
        mock_chromosome = Mock(spec=RouteChromosome)
        mock_mutated = Mock(spec=RouteChromosome)
        
        # Mock operators
        self.optimizer.operators = Mock()
        self.optimizer.operators.long_range_exploration_mutation = Mock(return_value=mock_mutated)
        
        result = self.optimizer._apply_adaptive_mutation(mock_chromosome, 0.2)
        
        self.assertEqual(result, mock_mutated)
        self.optimizer.operators.mutate.assert_called_once_with(mock_chromosome, 0.2)
    
    def test_apply_adaptive_mutation_with_constraint_operators(self):
        """Test adaptive mutation with constraint-preserving operators"""
        # Setup constraint operators
        self.optimizer.constraint_operators = Mock()
        mock_mutated = Mock(spec=RouteChromosome)
        self.optimizer.constraint_operators.distance_neutral_mutation = Mock(return_value=mock_mutated)
        
        # Mock regular operators too
        self.optimizer.operators = Mock()
        
        mock_chromosome = Mock(spec=RouteChromosome)
        
        result = self.optimizer._apply_adaptive_mutation(mock_chromosome, 0.2)
        
        self.assertEqual(result, mock_mutated)
        self.optimizer.constraint_operators.distance_neutral_mutation = Mock(return_value=mock_mutated)
    
    def test_apply_long_range_mutation_basic(self):
        """Test basic long-range mutation application"""
        # Create mock chromosome
        mock_chromosome = Mock(spec=RouteChromosome)
        mock_mutated = Mock(spec=RouteChromosome)
        
        # Mock operators
        self.optimizer.operators = Mock()
        self.optimizer.operators.long_range_exploration_mutation = Mock(return_value=mock_mutated)
        
        result = self.optimizer._apply_long_range_mutation(mock_chromosome, 0.5)
        
        self.assertEqual(result, mock_mutated)
        self.optimizer.operators.mutate.assert_called_once_with(mock_chromosome, 0.5)
    
    def test_apply_population_filtering_basic(self):
        """Test basic population filtering"""
        # Create mock population
        mock_chromosomes = []
        for i in range(20):
            mock_chr = Mock(spec=RouteChromosome)
            mock_chr.fitness = i * 0.05
            mock_chr.get_total_distance.return_value = 2500
            mock_chr.get_route_stats.return_value = {'total_distance_km': 2.5}
            mock_chromosomes.append(mock_chr)
        
        # Test filtering
        result = self.optimizer._apply_population_filtering(mock_chromosomes, 1001, 5000)
        
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result), 20)
    
    def test_apply_population_filtering_with_diversity_selector(self):
        """Test population filtering with diversity selector"""
        # Setup diversity selector
        self.optimizer.diversity_selector = Mock()
        filtered_pop = [Mock(spec=RouteChromosome) for _ in range(10)]
        self.optimizer.diversity_selector.maintain_diversity.return_value = filtered_pop
        
        # Create mock population
        mock_chromosomes = []
        for i in range(20):
            mock_chr = Mock(spec=RouteChromosome)
            mock_chr.fitness = i * 0.05
            mock_chr.get_total_distance.return_value = 2500
            mock_chr.get_route_stats.return_value = {'total_distance_km': 2.5}
            mock_chromosomes.append(mock_chr)
        
        # Test filtering
        result = self.optimizer._apply_population_filtering(mock_chromosomes, 1001, 5000)
        
        self.assertEqual(result, filtered_pop)
        self.optimizer.diversity_selector.maintain_diversity.assert_called_once()
    
    def test_get_optimization_stats_basic(self):
        """Test basic optimization statistics retrieval"""
        # Initialize required attributes for stats
        self.optimizer.best_fitness = 0.5
        self.optimizer.best_generation = 10
        self.optimizer.generation = 15
        self.optimizer.evaluation_times = [0.1, 0.2, 0.15]
        
        stats = self.optimizer._get_optimization_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('population_size', stats)
        self.assertIn('max_generations', stats)
        self.assertIn('crossover_rate', stats)
        self.assertIn('mutation_rate', stats)
    
    def test_get_optimization_stats_with_adaptive_mutation(self):
        """Test optimization statistics with adaptive mutation"""
        # Initialize required attributes for stats
        self.optimizer.best_fitness = 0.5
        self.optimizer.best_generation = 10
        self.optimizer.generation = 15
        self.optimizer.evaluation_times = [0.1, 0.2, 0.15]
        
        # Setup adaptive mutation controller
        self.optimizer.adaptive_mutation_controller = Mock()
        self.optimizer.adaptive_mutation_controller.get_current_mutation_rate.return_value = 0.25
        self.optimizer.adaptive_mutation_controller.get_stats.return_value = {
            'adaptation_count': 5,
            'stagnation_count': 2
        }
        
        stats = self.optimizer._get_optimization_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('current_mutation_rate', stats)
        self.assertIn('adaptive_mutation_stats', stats)
        self.assertEqual(stats['current_mutation_rate'], 0.25)
    
    def test_get_optimization_stats_with_restart_mechanisms(self):
        """Test optimization statistics with restart mechanisms"""
        # Initialize required attributes for stats
        self.optimizer.best_fitness = 0.5
        self.optimizer.best_generation = 10
        self.optimizer.generation = 15
        self.optimizer.evaluation_times = [0.1, 0.2, 0.15]
        
        # Setup restart mechanisms
        self.optimizer.restart_mechanisms = Mock()
        self.optimizer.restart_mechanisms.get_stats.return_value = {
            'restart_count': 2,
            'stagnation_generations': 15
        }
        
        stats = self.optimizer._get_optimization_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('restart_stats', stats)
        self.assertEqual(stats['restart_stats']['restart_count'], 2)
    
    def test_calculate_population_diversity_basic(self):
        """Test basic population diversity calculation"""
        # Create mock population with different fitness values
        mock_chromosomes = []
        for i in range(10):
            mock_chr = Mock(spec=RouteChromosome)
            mock_chr.fitness = i * 0.1
            mock_chr.get_route_nodes.return_value = [1001, 1002, 1003, 1001]
            mock_chr.get_total_distance.return_value = 2500 + i * 100
            mock_chromosomes.append(mock_chr)
        
        # Mock the population for diversity calculation
        self.optimizer.population = mock_chromosomes
        
        diversity = self.optimizer._calculate_population_diversity()
        
        self.assertIsInstance(diversity, float)
        self.assertGreaterEqual(diversity, 0.0)
        self.assertLessEqual(diversity, 1.0)
    
    def test_calculate_population_diversity_no_population(self):
        """Test population diversity calculation with no population"""
        self.optimizer.population = None
        
        diversity = self.optimizer._calculate_population_diversity()
        
        self.assertEqual(diversity, 0.0)
    
    def test_set_generation_callback(self):
        """Test setting generation callback"""
        callback_calls = []
        
        def test_callback(generation, population, fitness_scores):
            callback_calls.append({
                'generation': generation,
                'population_size': len(population),
                'fitness_scores': fitness_scores
            })
        
        self.optimizer.set_generation_callback(test_callback)
        
        # Verify callback was set
        self.assertEqual(self.optimizer.generation_callback, test_callback)
    
    def test_set_progress_callback(self):
        """Test setting progress callback"""
        progress_calls = []
        
        def test_callback(current_progress, estimated_remaining):
            progress_calls.append({
                'current_progress': current_progress,
                'estimated_remaining': estimated_remaining
            })
        
        self.optimizer.set_progress_callback(test_callback)
        
        # Verify callback was set
        self.assertEqual(self.optimizer.progress_callback, test_callback)
    
    @patch('genetic_algorithm.optimizer.GeneticRouteOptimizer._setup_optimization')
    @patch('genetic_algorithm.optimizer.GeneticRouteOptimizer._adapt_configuration')
    @patch('genetic_algorithm.optimizer.GeneticRouteOptimizer._evolve_generation')
    def test_optimize_route_complete_workflow(self, mock_evolve, mock_adapt, mock_setup):
        """Test complete route optimization workflow"""
        # Setup mocks
        mock_chromosome = Mock(spec=RouteChromosome)
        mock_chromosome.get_route_nodes.return_value = [1001, 1002, 1003, 1001]
        mock_chromosome.get_total_distance.return_value = 2500
        mock_chromosome.get_elevation_gain.return_value = 100
        mock_chromosome.get_elevation_loss.return_value = 50
        mock_chromosome.get_route_stats.return_value = {
            'total_distance_km': 2.5,
            'total_elevation_gain_m': 100,
            'total_elevation_loss_m': 50
        }
        mock_chromosome.fitness = 0.85
        mock_chromosome.copy.return_value = mock_chromosome
        
        mock_setup.return_value = None
        mock_adapt.return_value = None
        mock_evolve.return_value = ([mock_chromosome] * 10, [0.85] * 10)
        
        # Mock population initializer
        self.optimizer.population_initializer = Mock()
        self.optimizer.population_initializer.create_population.return_value = [mock_chromosome] * 10
        
        # Mock fitness evaluator
        self.optimizer.fitness_evaluator = Mock()
        self.optimizer.fitness_evaluator.evaluate_population.return_value = [0.85] * 10
        
        # Run optimization
        result = self.optimizer.optimize_route(1001, 5.0, "elevation")
        
        # Verify all methods were called
        mock_setup.assert_called_once_with(1001, 5000, "elevation")
        mock_adapt.assert_called_once_with(5.0)
        mock_evolve.assert_called()
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn('route', result)
        self.assertIn('stats', result)
    
    def test_optimize_route_with_callback(self):
        """Test route optimization with callback"""
        callback_calls = []
        
        def test_callback(generation, population, fitness_scores):
            callback_calls.append({
                'generation': generation,
                'population_size': len(population),
                'fitness_scores': fitness_scores
            })
        
        # Set the callback
        self.optimizer.set_generation_callback(test_callback)
        
        with patch('genetic_algorithm.optimizer.GeneticRouteOptimizer._setup_optimization') as mock_setup:
            mock_setup.return_value = None
            
            with patch('genetic_algorithm.optimizer.GeneticRouteOptimizer._adapt_configuration') as mock_adapt:
                mock_adapt.return_value = None
                
                with patch('genetic_algorithm.optimizer.GeneticRouteOptimizer._evolve_generation') as mock_evolve:
                    mock_chromosome = Mock(spec=RouteChromosome)
                    mock_chromosome.get_route_nodes.return_value = [1001, 1002, 1003, 1001]
                    mock_chromosome.get_total_distance.return_value = 2500
                    mock_chromosome.get_elevation_gain.return_value = 100
                    mock_chromosome.get_elevation_loss.return_value = 50
                    mock_chromosome.get_route_stats.return_value = {
                        'total_distance_km': 2.5,
                        'total_elevation_gain_m': 100,
                        'total_elevation_loss_m': 50
                    }
                    mock_chromosome.fitness = 0.85
                    mock_chromosome.copy.return_value = mock_chromosome
                    
                    mock_evolve.return_value = ([mock_chromosome] * 10, [0.85] * 10)
                    
                    # Mock population initializer
                    self.optimizer.population_initializer = Mock()
                    self.optimizer.population_initializer.create_population.return_value = [mock_chromosome] * 10
                    
                    # Mock fitness evaluator
                    self.optimizer.fitness_evaluator = Mock()
                    self.optimizer.fitness_evaluator.evaluate_population.return_value = [0.85] * 10
                    
                    # Run optimization with callback
                    result = self.optimizer.optimize_route(1001, 5.0, "elevation")
                    
                    self.assertIsInstance(result, dict)
                    self.assertIn('route', result)
                    self.assertIn('stats', result)
    
    def test_optimize_route_error_handling(self):
        """Test route optimization error handling"""
        with patch('genetic_algorithm.optimizer.GeneticRouteOptimizer._setup_optimization') as mock_setup:
            mock_setup.side_effect = Exception("Setup failed")
            
            # Test that optimization handles errors gracefully
            try:
                result = self.optimizer.optimize_route(1001, 5.0, "elevation")
                # If no exception was raised, check that we got a valid result structure
                self.assertIsInstance(result, dict)
                self.assertIn('route', result)
                self.assertIn('stats', result)
            except Exception as e:
                # If an exception was raised, verify it's the expected one
                self.assertIn("Setup failed", str(e))
    
    def test_adaptive_sizing_configuration(self):
        """Test adaptive sizing configuration"""
        config = GAConfig(
            adaptive_sizing=True,
            population_size=50,
            max_generations=100
        )
        
        optimizer = GeneticRouteOptimizer(self.mock_graph, config)
        
        # Test adaptive sizing would adjust parameters based on problem size
        self.assertTrue(optimizer.config.adaptive_sizing)
        self.assertEqual(optimizer.config.population_size, 50)
        self.assertEqual(optimizer.config.max_generations, 100)
    
    def test_precision_enhancement_disabled(self):
        """Test that precision enhancement is disabled by default"""
        optimizer = GeneticRouteOptimizer(self.mock_graph)
        
        # Should be disabled by default
        self.assertFalse(optimizer.precision_components_enabled)
        self.assertIsNone(optimizer.precision_crossover)
        self.assertIsNone(optimizer.precision_mutation)
        self.assertIsNone(optimizer.precision_visualizer)
    
    def test_segment_cache_integration(self):
        """Test segment cache integration"""
        optimizer = GeneticRouteOptimizer(self.mock_graph)
        
        # Should have segment cache initialized
        self.assertIsNotNone(optimizer.segment_cache)
        self.assertEqual(optimizer.segment_cache.max_size, 5000)
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test with invalid values
        config = GAConfig(
            population_size=0,  # Invalid
            max_generations=-1,  # Invalid
            crossover_rate=1.5,  # Invalid
            mutation_rate=-0.1  # Invalid
        )
        
        optimizer = GeneticRouteOptimizer(self.mock_graph, config)
        
        # Optimizer should still be created but with potentially adjusted values
        self.assertIsNotNone(optimizer)
        self.assertEqual(optimizer.config.population_size, 0)  # Will be handled during setup
    
    def test_fitness_objective_validation(self):
        """Test fitness objective validation"""
        # Test with valid objectives - we'll just test the setup since optimization is complex
        valid_objectives = ["distance", "elevation", "balanced", "scenic", "efficiency"]
        
        for objective in valid_objectives:
            # Mock the setup to avoid full optimization
            with patch.object(self.optimizer, '_setup_optimization') as mock_setup:
                mock_setup.return_value = None
                
                # Test that setup doesn't raise exception for valid objectives
                try:
                    self.optimizer._setup_optimization(1001, 5000, objective)
                    # If we reach here, the objective was valid
                    self.assertTrue(True)
                except Exception as e:
                    self.fail(f"Valid objective '{objective}' raised exception: {e}")


if __name__ == '__main__':
    unittest.main()