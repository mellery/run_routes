#!/usr/bin/env python3
"""
Unit tests for Genetic Route Optimizer
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import networkx as nx
import time

from genetic_algorithm import GeneticRouteOptimizer
from genetic_algorithm.optimizer import GAConfig, GAResults
from genetic_algorithm import RouteChromosome, RouteSegment


class TestGeneticOptimizer(unittest.TestCase):
    """Test genetic route optimizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test graph
        self.test_graph = nx.Graph()
        nodes = [
            (1, -80.4094, 37.1299, 100),
            (2, -80.4000, 37.1300, 110),
            (3, -80.4050, 37.1350, 105),
            (4, -80.4100, 37.1250, 120)
        ]
        
        for node_id, x, y, elev in nodes:
            self.test_graph.add_node(node_id, x=x, y=y, elevation=elev)
        
        edges = [(1, 2, 100), (2, 3, 150), (3, 4, 200), (4, 1, 180), (1, 3, 250), (2, 4, 220)]
        for n1, n2, length in edges:
            self.test_graph.add_edge(n1, n2, length=length)
        
        # Create test configuration
        self.test_config = GAConfig(
            population_size=10,
            max_generations=20,
            elite_size=2,  # Reasonable elite size
            verbose=False
        )
        
        # Create optimizer
        self.optimizer = GeneticRouteOptimizer(self.test_graph, self.test_config)
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        # Test default configuration
        optimizer = GeneticRouteOptimizer(self.test_graph)
        self.assertIsInstance(optimizer.config, GAConfig)
        self.assertEqual(optimizer.config.population_size, 100)
        self.assertEqual(optimizer.config.max_generations, 200)
        
        # Test custom configuration
        custom_config = GAConfig(population_size=50, max_generations=100)
        optimizer = GeneticRouteOptimizer(self.test_graph, custom_config)
        self.assertEqual(optimizer.config.population_size, 50)
        self.assertEqual(optimizer.config.max_generations, 100)
        
        # Test components initialization
        self.assertIsNotNone(optimizer.operators)
        self.assertEqual(optimizer.generation, 0)
        self.assertIsNone(optimizer.best_chromosome)
    
    def test_ga_config_dataclass(self):
        """Test GA configuration dataclass"""
        # Test default values
        config = GAConfig()
        self.assertEqual(config.population_size, 100)
        self.assertEqual(config.max_generations, 200)
        self.assertEqual(config.crossover_rate, 0.8)
        self.assertEqual(config.mutation_rate, 0.15)
        self.assertEqual(config.elite_size, 10)
        self.assertTrue(config.adaptive_sizing)
        
        # Test custom values
        config = GAConfig(
            population_size=50,
            max_generations=150,
            crossover_rate=0.7,
            mutation_rate=0.2
        )
        self.assertEqual(config.population_size, 50)
        self.assertEqual(config.max_generations, 150)
        self.assertEqual(config.crossover_rate, 0.7)
        self.assertEqual(config.mutation_rate, 0.2)
    
    def test_ga_results_dataclass(self):
        """Test GA results dataclass"""
        # Create test chromosome
        segment = RouteSegment(1, 2, [1, 2])
        chromosome = RouteChromosome([segment])
        
        results = GAResults(
            best_chromosome=chromosome,
            best_fitness=0.85,
            generation_found=50,
            total_generations=100,
            total_time=30.5,
            convergence_reason="max_generations",
            population_history=[],
            fitness_history=[],
            stats={}
        )
        
        self.assertEqual(results.best_fitness, 0.85)
        self.assertEqual(results.generation_found, 50)
        self.assertEqual(results.total_generations, 100)
        self.assertEqual(results.total_time, 30.5)
        self.assertEqual(results.convergence_reason, "max_generations")
    
    @patch('genetic_route_optimizer.PopulationInitializer')
    @patch('genetic_route_optimizer.GAFitnessEvaluator')
    def test_setup_optimization(self, mock_fitness, mock_population):
        """Test optimization setup"""
        # Setup mocks
        mock_population.return_value = Mock()
        mock_fitness.return_value = Mock()
        
        # Test setup
        self.optimizer._setup_optimization(1, 5.0, "elevation")
        
        # Verify initialization
        mock_population.assert_called_once_with(self.test_graph, 1, True)
        mock_fitness.assert_called_once_with("elevation", 5.0, unittest.mock.ANY, enable_micro_terrain=True, allow_bidirectional_segments=True)
        
        # Verify reset
        self.assertEqual(self.optimizer.generation, 0)
        self.assertEqual(len(self.optimizer.population_history), 0)
        self.assertEqual(len(self.optimizer.fitness_history), 0)
    
    def test_adaptive_configuration(self):
        """Test adaptive configuration adjustment"""
        # Test small distance
        self.optimizer.config = GAConfig(population_size=100, max_generations=200)
        self.optimizer._adapt_configuration(2.0)
        self.assertLessEqual(self.optimizer.config.population_size, 50)
        self.assertLessEqual(self.optimizer.config.max_generations, 100)
        
        # Test large distance
        self.optimizer.config = GAConfig(population_size=100, max_generations=200)
        self.optimizer._adapt_configuration(10.0)
        self.assertGreaterEqual(self.optimizer.config.population_size, 100)
        self.assertGreaterEqual(self.optimizer.config.max_generations, 200)
        
        # Test elite size adjustment
        self.optimizer.config = GAConfig(population_size=100)
        self.optimizer._adapt_configuration(5.0)
        self.assertGreaterEqual(self.optimizer.config.elite_size, 5)
        self.assertLessEqual(self.optimizer.config.elite_size, 10)
    
    def test_evolve_generation(self):
        """Test single generation evolution"""
        # Create test population with proper size for the test config
        segment1 = RouteSegment(1, 2, [1, 2])
        segment1.length = 1000.0
        segment1.elevation_gain = 50.0
        
        segment2 = RouteSegment(2, 3, [2, 3])
        segment2.length = 1500.0
        segment2.elevation_gain = 30.0
        
        population = []
        fitness_scores = []
        
        # Use test config population size
        pop_size = self.optimizer.config.population_size
        for i in range(pop_size):
            chromosome = RouteChromosome([segment1.copy(), segment2.copy()])
            chromosome.fitness = 0.5 + i * 0.05 / pop_size  # Ascending fitness
            population.append(chromosome)
            fitness_scores.append(chromosome.fitness)
        
        # Mock operators with proper call tracking
        with patch.object(self.optimizer.operators, 'tournament_selection') as mock_selection, \
             patch.object(self.optimizer.operators, 'segment_exchange_crossover') as mock_crossover, \
             patch.object(self.optimizer.operators, 'segment_replacement_mutation') as mock_mutation1, \
             patch.object(self.optimizer.operators, 'route_extension_mutation') as mock_mutation2, \
             patch.object(self.optimizer, '_apply_population_filtering') as mock_filtering:
            
            # Setup mocks to return valid chromosomes
            mock_selection.return_value = population[0]  # Return first chromosome
            mock_crossover.return_value = (population[0].copy(), population[1].copy())
            mock_mutation1.return_value = population[0].copy()
            mock_mutation2.return_value = population[1].copy()
            
            # Setup filtering mock to pass through without filtering
            mock_filtering.side_effect = lambda pop, fit: (pop, fit)
            
            # Setup fitness evaluator
            self.optimizer.fitness_evaluator = Mock()
            self.optimizer.fitness_evaluator.target_distance_km = 5.0
            # Mock should return appropriate number of scores based on population size
            def mock_evaluate_population(population, graph):
                return [0.5 + i * 0.01 for i in range(len(population))]
            self.optimizer.fitness_evaluator.evaluate_population.side_effect = mock_evaluate_population
            
            # Test evolution
            new_population, new_fitness = self.optimizer._evolve_generation(population, fitness_scores)
            
            # Verify results
            self.assertEqual(len(new_population), self.optimizer.config.population_size)
            self.assertEqual(len(new_fitness), len(new_population))
            
            # For this small population, offspring generation should occur
            # Since elite_size is small, there should be offspring generation
            if pop_size > self.optimizer.config.elite_size:
                self.assertTrue(mock_selection.called)
                self.assertTrue(mock_crossover.called)
                self.assertTrue(mock_mutation1.called)
                self.assertTrue(mock_mutation2.called)
    
    def test_convergence_detection(self):
        """Test convergence detection"""
        # Test insufficient history
        convergence = self.optimizer._check_convergence([0.5, 0.6, 0.7])
        self.assertFalse(convergence)
        
        # Test no convergence (improving fitness)
        improving_history = []
        for i in range(25):
            gen_fitness = [0.5 + i * 0.01] * 10  # Improving each generation
            improving_history.append(gen_fitness)
        
        self.optimizer.fitness_history = improving_history
        convergence = self.optimizer._check_convergence([0.7] * 10)
        self.assertFalse(convergence)
        
        # Test convergence (stable fitness)
        stable_history = []
        for i in range(25):
            gen_fitness = [0.8] * 10  # Stable fitness
            stable_history.append(gen_fitness)
        
        self.optimizer.fitness_history = stable_history
        convergence = self.optimizer._check_convergence([0.8] * 10)
        self.assertTrue(convergence)
    
    @patch('genetic_route_optimizer.PopulationInitializer')
    @patch('genetic_route_optimizer.GAFitnessEvaluator')
    def test_optimize_route_basic(self, mock_fitness_class, mock_population_class):
        """Test basic route optimization"""
        # Setup mocks
        mock_population = Mock()
        mock_fitness = Mock()
        mock_population_class.return_value = mock_population
        mock_fitness_class.return_value = mock_fitness
        
        # Create test population
        segment = RouteSegment(1, 2, [1, 2])
        segment.length = 2500.0
        segment.elevation_gain = 100.0
        
        test_chromosome = RouteChromosome([segment])
        test_chromosome.fitness = 0.8
        test_population = [test_chromosome] * 5
        
        mock_population.create_population.return_value = test_population
        mock_fitness.target_distance_km = 5.0
        mock_fitness.evaluate_population.return_value = [0.8] * 5
        
        # Configure for quick execution
        self.optimizer.config.max_generations = 5
        self.optimizer.config.verbose = False
        
        # Mock evolution to avoid complex operator interactions
        def mock_evolve(pop, fitness):
            return pop, fitness
        
        self.optimizer._evolve_generation = Mock(side_effect=mock_evolve)
        
        # Test optimization
        results = self.optimizer.optimize_route(1, 5.0, "elevation")
        
        # Verify results
        self.assertIsInstance(results, GAResults)
        self.assertIsNotNone(results.best_chromosome)
        self.assertGreater(results.best_fitness, 0.0)
        self.assertGreaterEqual(results.total_generations, 1)
        self.assertGreater(results.total_time, 0.0)
        self.assertIn(results.convergence_reason, ["max_generations", "convergence"])
    
    def test_optimization_statistics(self):
        """Test optimization statistics calculation"""
        # Setup test data
        self.optimizer.fitness_history = [
            [0.5, 0.6, 0.7],  # Generation 0
            [0.6, 0.7, 0.8],  # Generation 1
            [0.7, 0.8, 0.9]   # Generation 2
        ]
        self.optimizer.evaluation_times = [1.0, 1.2, 1.1]
        self.optimizer.best_fitness = 0.9
        self.optimizer.best_generation = 2
        
        # Setup fitness evaluator
        mock_fitness = Mock()
        mock_fitness.objective.value = "elevation"
        mock_fitness.target_distance_km = 5.0
        self.optimizer.fitness_evaluator = mock_fitness
        
        # Test statistics
        stats = self.optimizer._get_optimization_stats()
        
        self.assertEqual(stats['total_evaluations'], 9)
        self.assertAlmostEqual(stats['avg_generation_time'], 1.1, places=1)
        self.assertEqual(len(stats['best_fitness_progression']), 3)
        self.assertEqual(len(stats['avg_fitness_progression']), 3)
        self.assertEqual(stats['convergence_generation'], 2)
        self.assertEqual(stats['objective'], "elevation")
        self.assertEqual(stats['target_distance'], 5.0)
    
    def test_population_diversity_calculation(self):
        """Test population diversity calculation"""
        # Test empty history
        diversity = self.optimizer._calculate_population_diversity()
        self.assertEqual(diversity, 0.0)
        
        # Create test population with diversity
        segment1 = RouteSegment(1, 2, [1, 2])
        segment1.length = 1000.0
        
        segment2 = RouteSegment(1, 2, [1, 2])
        segment2.length = 2000.0
        
        segment3 = RouteSegment(1, 2, [1, 2])
        segment3.length = 1500.0
        
        population = [
            RouteChromosome([segment1]),
            RouteChromosome([segment2]),
            RouteChromosome([segment3])
        ]
        
        self.optimizer.population_history = [population]
        
        # Test diversity calculation
        diversity = self.optimizer._calculate_population_diversity()
        self.assertGreater(diversity, 0.0)
    
    def test_callback_functionality(self):
        """Test callback functionality"""
        # Test generation callback
        generation_calls = []
        
        def generation_callback(gen, pop, fitness):
            generation_calls.append((gen, len(pop), len(fitness)))
        
        self.optimizer.set_generation_callback(generation_callback)
        self.assertEqual(self.optimizer.generation_callback, generation_callback)
        
        # Test progress callback
        progress_calls = []
        
        def progress_callback(progress, best_fitness):
            progress_calls.append((progress, best_fitness))
        
        self.optimizer.set_progress_callback(progress_callback)
        self.assertEqual(self.optimizer.progress_callback, progress_callback)
    
    @patch('genetic_route_optimizer.datetime')
    def test_visualization_generation(self, mock_datetime):
        """Test visualization generation"""
        # Setup mock datetime
        mock_datetime.now.return_value.strftime.return_value = "20241204_143000"
        
        # Create test data
        segment = RouteSegment(1, 2, [1, 2])
        population = [RouteChromosome([segment])]
        fitness_scores = [0.8]
        
        # Create mock visualizer
        mock_visualizer = Mock()
        
        # Test visualization generation
        self.optimizer._generate_visualization(population, fitness_scores, 10, mock_visualizer)
        
        # Verify visualizer was called
        mock_visualizer.save_population_map.assert_called_once()
        call_args = mock_visualizer.save_population_map.call_args
        
        # Verify arguments
        self.assertEqual(call_args[0][0], population)  # population
        self.assertEqual(call_args[0][1], 10)          # generation
        self.assertIn("ga_evolution_gen_010", call_args[0][2])  # filename
    
    def test_error_handling(self):
        """Test error handling in optimization"""
        # Test with invalid graph
        empty_graph = nx.Graph()
        optimizer = GeneticRouteOptimizer(empty_graph, self.test_config)
        
        # This should handle the empty graph gracefully
        with patch('genetic_route_optimizer.PopulationInitializer') as mock_pop:
            mock_pop.return_value.create_population.return_value = []
            
            try:
                results = optimizer.optimize_route(1, 5.0, "elevation")
                self.fail("Should have raised exception for empty population")
            except ValueError as e:
                self.assertIn("Failed to initialize population", str(e))
    
    def test_integration_with_real_components(self):
        """Test integration with real GA components"""
        # Use real components (not mocked) for integration test
        config = GAConfig(
            population_size=5,
            max_generations=3,
            verbose=False
        )
        
        optimizer = GeneticRouteOptimizer(self.test_graph, config)
        
        # This should run without errors
        try:
            results = optimizer.optimize_route(1, 2.0, "elevation")
            
            # Basic verification
            self.assertIsInstance(results, GAResults)
            self.assertIsNotNone(results.best_chromosome)
            self.assertGreaterEqual(results.best_fitness, 0.0)
            
        except Exception as e:
            self.fail(f"Integration test failed: {e}")


if __name__ == '__main__':
    unittest.main()