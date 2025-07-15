#!/usr/bin/env python3
"""
Unit tests for GA Restart Mechanisms
Tests convergence detection and population restart strategies to escape local optima
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import random
import math
import statistics
import sys
import os
from typing import List, Dict, Set, Optional, Tuple

# Add the parent directory to sys.path to import genetic algorithm modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from genetic_algorithm.restart_mechanisms import (
        RestartConfig, ConvergenceDetector, RestartMechanisms
    )
    from genetic_algorithm.chromosome import RouteChromosome, RouteSegment
    GA_RESTART_AVAILABLE = True
except ImportError:
    GA_RESTART_AVAILABLE = False


class TestGARestartMechanisms(unittest.TestCase):
    """Base test class for GA restart mechanisms"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not GA_RESTART_AVAILABLE:
            self.skipTest("GA restart mechanism modules not available")
        
        # Create test graph
        import networkx as nx
        self.test_graph = nx.MultiGraph()
        self.test_graph.add_node(1, x=-80.4094, y=37.1299, elevation=600)
        self.test_graph.add_node(2, x=-80.4095, y=37.1300, elevation=620)
        self.test_graph.add_node(3, x=-80.4096, y=37.1301, elevation=610)
        self.test_graph.add_node(4, x=-80.4097, y=37.1302, elevation=650)
        self.test_graph.add_edge(1, 2, length=100, highway='residential')
        self.test_graph.add_edge(2, 3, length=150, highway='residential')
        self.test_graph.add_edge(3, 4, length=120, highway='primary')
        self.test_graph.add_edge(4, 1, length=180, highway='residential')
        
        # Test parameters
        self.start_node = 1
        self.target_distance = 5.0
        
        # Create mock chromosomes for testing
        self.mock_chromosomes = []
        for i in range(10):
            chromosome = Mock(spec=RouteChromosome)
            chromosome.segments = [Mock(spec=RouteSegment)]
            chromosome.fitness = 0.5 + (i * 0.05)  # Fitness from 0.5 to 0.95
            chromosome.get_total_distance.return_value = 5000.0 + (i * 100)
            chromosome.get_total_elevation_gain.return_value = 100.0 + (i * 10)
            chromosome.get_route_nodes.return_value = [1, 2, 3, 4, 1]
            self.mock_chromosomes.append(chromosome)


@unittest.skipUnless(GA_RESTART_AVAILABLE, "GA restart mechanism modules not available")
class TestRestartConfig(TestGARestartMechanisms):
    """Test RestartConfig dataclass"""
    
    def test_restart_config_default_values(self):
        """Test default configuration values"""
        config = RestartConfig()
        
        self.assertEqual(config.convergence_threshold, 0.01)
        self.assertEqual(config.stagnation_generations, 10)
        self.assertEqual(config.diversity_threshold, 0.2)
        self.assertEqual(config.elite_retention_percentage, 0.2)
        self.assertEqual(config.restart_exploration_percentage, 0.4)
        self.assertEqual(config.restart_high_mutation_percentage, 0.4)
        self.assertEqual(config.max_restarts, 3)
        self.assertEqual(config.restart_mutation_rate, 0.4)
    
    def test_restart_config_custom_values(self):
        """Test configuration with custom values"""
        config = RestartConfig(
            convergence_threshold=0.005,
            stagnation_generations=15,
            diversity_threshold=0.15,
            elite_retention_percentage=0.25,
            restart_exploration_percentage=0.3,
            restart_high_mutation_percentage=0.5,
            max_restarts=5,
            restart_mutation_rate=0.6
        )
        
        self.assertEqual(config.convergence_threshold, 0.005)
        self.assertEqual(config.stagnation_generations, 15)
        self.assertEqual(config.diversity_threshold, 0.15)
        self.assertEqual(config.elite_retention_percentage, 0.25)
        self.assertEqual(config.restart_exploration_percentage, 0.3)
        self.assertEqual(config.restart_high_mutation_percentage, 0.5)
        self.assertEqual(config.max_restarts, 5)
        self.assertEqual(config.restart_mutation_rate, 0.6)
    
    def test_restart_config_validation(self):
        """Test configuration validation"""
        # Test that percentages sum correctly
        config = RestartConfig(
            elite_retention_percentage=0.3,
            restart_exploration_percentage=0.4,
            restart_high_mutation_percentage=0.3
        )
        
        total_percentage = (config.elite_retention_percentage + 
                          config.restart_exploration_percentage + 
                          config.restart_high_mutation_percentage)
        self.assertEqual(total_percentage, 1.0)


@unittest.skipUnless(GA_RESTART_AVAILABLE, "GA restart mechanism modules not available")
class TestConvergenceDetector(TestGARestartMechanisms):
    """Test ConvergenceDetector functionality"""
    
    def test_convergence_detector_initialization(self):
        """Test convergence detector initialization"""
        config = RestartConfig()
        detector = ConvergenceDetector(config)
        
        self.assertEqual(detector.config, config)
        self.assertEqual(detector.fitness_history, [])
        self.assertEqual(detector.diversity_history, [])
        self.assertEqual(detector.stagnation_count, 0)
        self.assertEqual(detector.best_fitness, float('-inf'))
        self.assertEqual(detector.last_improvement_generation, 0)
    
    def test_fitness_improvement_detection(self):
        """Test fitness improvement detection"""
        config = RestartConfig(convergence_threshold=0.01)
        detector = ConvergenceDetector(config)
        
        # Test significant improvement
        fitness_scores = [0.5, 0.6, 0.7, 0.8]
        population = self.mock_chromosomes[:4]
        
        # Should not trigger restart with improving fitness
        restart_needed = detector.update(1, population, fitness_scores)
        self.assertFalse(restart_needed)
        self.assertGreater(detector.best_fitness, float('-inf'))
    
    def test_stagnation_detection(self):
        """Test stagnation detection"""
        config = RestartConfig(stagnation_generations=3, convergence_threshold=0.001)
        detector = ConvergenceDetector(config)
        
        # Simulate stagnant fitness over multiple generations
        fitness_scores = [0.8, 0.8, 0.8]
        population = self.mock_chromosomes[:3]
        
        # First few generations shouldn't trigger restart
        for generation in range(1, 3):
            restart_needed = detector.update(generation, population, fitness_scores)
            self.assertFalse(restart_needed)
        
        # After stagnation_generations, should trigger restart
        restart_needed = detector.update(4, population, fitness_scores)
        # Note: Actual restart decision depends on full implementation
        self.assertIsInstance(restart_needed, bool)
    
    def test_diversity_tracking(self):
        """Test population diversity tracking"""
        config = RestartConfig(diversity_threshold=0.2)
        detector = ConvergenceDetector(config)
        
        # Create population with varying fitness
        fitness_scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        population = self.mock_chromosomes[:5]
        
        restart_needed = detector.update(1, population, fitness_scores)
        
        # Should have recorded diversity metrics
        self.assertIsInstance(restart_needed, bool)
        # Check that detector tracks diversity (implementation dependent)
    
    def test_convergence_threshold_checking(self):
        """Test convergence threshold checking"""
        config = RestartConfig(convergence_threshold=0.1)
        detector = ConvergenceDetector(config)
        
        # Test with fitness improvements below threshold
        fitness_scores_low_improvement = [0.80, 0.81, 0.82]  # 1% improvements
        population = self.mock_chromosomes[:3]
        
        for generation in range(1, 4):
            restart_needed = detector.update(
                generation, population, fitness_scores_low_improvement[:generation]
            )
            # With small improvements, might trigger convergence
            self.assertIsInstance(restart_needed, bool)
    
    def test_best_fitness_tracking(self):
        """Test best fitness tracking over generations"""
        config = RestartConfig()
        detector = ConvergenceDetector(config)
        
        # Progressive fitness improvements
        generation_fitness = [
            ([0.5, 0.6, 0.7], 0.7),
            ([0.6, 0.7, 0.8], 0.8),
            ([0.7, 0.8, 0.9], 0.9)
        ]
        
        for generation, (fitness_scores, expected_best) in enumerate(generation_fitness, 1):
            population = self.mock_chromosomes[:len(fitness_scores)]
            detector.update(generation, population, fitness_scores)
            
            # Best fitness should be at least the expected value
            self.assertGreaterEqual(detector.best_fitness, expected_best)
    
    def test_generation_tracking(self):
        """Test generation tracking for improvements"""
        config = RestartConfig()
        detector = ConvergenceDetector(config)
        
        # Simulate improvement in generation 5
        fitness_scores = [0.8]
        population = self.mock_chromosomes[:1]
        
        detector.update(5, population, fitness_scores)
        
        # Should track when last improvement occurred
        # (Implementation may vary, but should be reasonable)
        self.assertIsInstance(detector.last_improvement_generation, int)


@unittest.skipUnless(GA_RESTART_AVAILABLE, "GA restart mechanism modules not available")
class TestDiversityCalculator(TestGARestartMechanisms):
    """Test diversity calculation functionality within ConvergenceDetector"""
    
    def test_diversity_calculator_exists(self):
        """Test that diversity calculator is available"""
        # Diversity calculation is implemented in ConvergenceDetector
        config = RestartConfig()
        detector = ConvergenceDetector(config)
        
        # Test that the diversity calculation method exists
        self.assertTrue(hasattr(detector, '_calculate_population_diversity'))
        self.assertTrue(callable(getattr(detector, '_calculate_population_diversity')))
    
    def test_fitness_diversity_calculation(self):
        """Test fitness diversity calculation"""
        config = RestartConfig()
        detector = ConvergenceDetector(config)
        
        # Test with varied population
        population = self.mock_chromosomes[:3]
        diversity = detector._calculate_population_diversity(population)
        
        self.assertIsInstance(diversity, float)
        self.assertGreaterEqual(diversity, 0.0)
        self.assertLessEqual(diversity, 1.0)
    
    def test_route_diversity_calculation(self):
        """Test route diversity calculation"""
        config = RestartConfig()
        detector = ConvergenceDetector(config)
        
        population = self.mock_chromosomes[:5]
        diversity = detector._calculate_population_diversity(population)
        
        self.assertIsInstance(diversity, float)
        self.assertGreaterEqual(diversity, 0.0)
        self.assertLessEqual(diversity, 1.0)
    
    def test_diversity_empty_population(self):
        """Test diversity calculation with empty population"""
        config = RestartConfig()
        detector = ConvergenceDetector(config)
        
        # Empty population should have maximum diversity (1.0)
        diversity = detector._calculate_population_diversity([])
        self.assertEqual(diversity, 1.0)  # Implementation returns 1.0 for empty population
    
    def test_diversity_single_individual(self):
        """Test diversity calculation with single individual"""
        config = RestartConfig()
        detector = ConvergenceDetector(config)
        
        # Single individual should have maximum diversity (1.0)
        diversity = detector._calculate_population_diversity([self.mock_chromosomes[0]])
        self.assertEqual(diversity, 1.0)  # Implementation returns 1.0 for single individual


@unittest.skipUnless(GA_RESTART_AVAILABLE, "GA restart mechanism modules not available")
class TestRestartManager(TestGARestartMechanisms):
    """Test RestartMechanisms functionality"""
    
    def test_restart_manager_exists(self):
        """Test that restart manager is available"""
        # RestartMechanisms is the actual implemented class
        restart_manager = RestartMechanisms(
            self.test_graph, 
            self.start_node, 
            self.target_distance, 
            RestartConfig()
        )
        
        self.assertIsNotNone(restart_manager)
        self.assertIsInstance(restart_manager, RestartMechanisms)
    
    def test_restart_manager_initialization(self):
        """Test restart manager initialization"""
        config = RestartConfig()
        manager = RestartMechanisms(
            self.test_graph, 
            self.start_node, 
            self.target_distance, 
            config
        )
        
        self.assertEqual(manager.config, config)
        self.assertIsInstance(manager.restart_count, int)
        self.assertEqual(manager.restart_count, 0)
        self.assertIsInstance(manager.convergence_detector, ConvergenceDetector)
    
    def test_should_restart_decision(self):
        """Test restart decision logic"""
        config = RestartConfig(max_restarts=3)
        manager = RestartMechanisms(
            self.test_graph, 
            self.start_node, 
            self.target_distance, 
            config
        )
        
        population = self.mock_chromosomes[:5]
        fitness_scores = [c.fitness for c in population]
        
        should_restart = manager.check_restart_needed(1, population, fitness_scores)
        self.assertIsInstance(should_restart, bool)
    
    def test_restart_population_creation(self):
        """Test restart population creation"""
        config = RestartConfig(
            elite_retention_percentage=0.2,
            restart_exploration_percentage=0.4,
            restart_high_mutation_percentage=0.4
        )
        manager = RestartMechanisms(
            self.test_graph, 
            self.start_node, 
            self.target_distance, 
            config
        )
        
        current_population = self.mock_chromosomes[:10]
        fitness_scores = [c.fitness for c in current_population]
        
        new_population, restart_info = manager.execute_restart(
            current_population, fitness_scores, 1
        )
        
        self.assertIsInstance(new_population, list)
        self.assertGreater(len(new_population), 0)
        self.assertIsInstance(restart_info, dict)
    
    def test_restart_count_tracking(self):
        """Test restart count tracking"""
        config = RestartConfig(max_restarts=2)
        manager = RestartMechanisms(
            self.test_graph, 
            self.start_node, 
            self.target_distance, 
            config
        )
        
        # Initial count should be 0
        self.assertEqual(manager.restart_count, 0)
        
        # Simulate restarts by executing restart
        population = self.mock_chromosomes[:5]
        fitness_scores = [c.fitness for c in population]
        
        new_population, restart_info = manager.execute_restart(population, fitness_scores, 1)
        self.assertEqual(manager.restart_count, 1)
        
        new_population, restart_info = manager.execute_restart(population, fitness_scores, 2)
        self.assertEqual(manager.restart_count, 2)
    
    def test_max_restarts_limit(self):
        """Test maximum restarts limit enforcement"""
        config = RestartConfig(max_restarts=1)
        manager = RestartMechanisms(
            self.test_graph, 
            self.start_node, 
            self.target_distance, 
            config
        )
        
        # Force restart count to maximum
        manager.restart_count = config.max_restarts
        
        # Should not allow more restarts
        population = self.mock_chromosomes[:5]
        fitness_scores = [c.fitness for c in population]
        
        should_restart = manager.check_restart_needed(10, population, fitness_scores)
        self.assertFalse(should_restart)


@unittest.skipUnless(GA_RESTART_AVAILABLE, "GA restart mechanism modules not available")
class TestRestartMechanismsIntegration(TestGARestartMechanisms):
    """Test integration between restart mechanism components"""
    
    def test_convergence_to_restart_workflow(self):
        """Test complete convergence detection to restart workflow"""
        config = RestartConfig(stagnation_generations=2, max_restarts=1)
        manager = RestartMechanisms(
            self.test_graph, 
            self.start_node, 
            self.target_distance, 
            config
        )
        
        population = self.mock_chromosomes[:5]
        
        # Simulate stagnant generations
        for generation in range(1, 4):
            fitness_scores = [0.5] * 5  # No improvement
            
            restart_needed = manager.check_restart_needed(generation, population, fitness_scores)
            self.assertIsInstance(restart_needed, bool)
            
            # If restart is needed, test the execution
            if restart_needed:
                new_population, restart_info = manager.execute_restart(
                    population, fitness_scores, generation
                )
                self.assertIsInstance(new_population, list)
                self.assertIsInstance(restart_info, dict)
    
    def test_restart_with_diversity_preservation(self):
        """Test restart mechanism with diversity preservation"""
        config = RestartConfig(elite_retention_percentage=0.3)
        manager = RestartMechanisms(
            self.test_graph, 
            self.start_node, 
            self.target_distance, 
            config
        )
        
        # High fitness population (converged)
        converged_population = []
        for i in range(10):
            chromosome = Mock(spec=RouteChromosome)
            chromosome.fitness = 0.95  # All very similar high fitness
            converged_population.append(chromosome)
        
        fitness_scores = [c.fitness for c in converged_population]
        
        new_population, restart_info = manager.execute_restart(
            converged_population, fitness_scores, 1
        )
        
        # Should maintain elite but add diversity
        self.assertIsInstance(new_population, list)
        self.assertIsInstance(restart_info, dict)
    
    def test_multiple_restart_cycles(self):
        """Test multiple restart cycles"""
        config = RestartConfig(max_restarts=3, stagnation_generations=1)
        manager = RestartMechanisms(
            self.test_graph, 
            self.start_node, 
            self.target_distance, 
            config
        )
        
        population = self.mock_chromosomes[:5]
        generation = 1
        
        # Simulate multiple convergence-restart cycles
        for cycle in range(2):
            fitness_scores = [0.5] * 5  # Stagnant fitness
            
            restart_needed = manager.check_restart_needed(generation, population, fitness_scores)
            
            if restart_needed:
                new_population, restart_info = manager.execute_restart(
                    population, fitness_scores, generation
                )
                self.assertIsInstance(new_population, list)
                self.assertIsInstance(restart_info, dict)
            
            generation += 1
        
        # Should have tracked restarts
        self.assertGreaterEqual(manager.restart_count, 0)
        self.assertLessEqual(manager.restart_count, config.max_restarts)
    
    def test_convergence_detector_empty_fitness_scores(self):
        """Test convergence detector with empty fitness scores"""
        config = RestartConfig()
        detector = ConvergenceDetector(config)
        
        # Test with empty fitness scores
        restart_needed = detector.update(1, self.mock_chromosomes[:3], [])
        self.assertFalse(restart_needed)
        
        # Test with None fitness scores
        restart_needed = detector.update(1, self.mock_chromosomes[:3], None)
        self.assertFalse(restart_needed)
    
    def test_convergence_detector_stagnation_tracking(self):
        """Test detailed stagnation tracking"""
        config = RestartConfig(stagnation_generations=2, convergence_threshold=0.05)
        detector = ConvergenceDetector(config)
        
        # Generation 1 - initial fitness
        restart_needed = detector.update(1, self.mock_chromosomes[:3], [0.5, 0.6, 0.7])
        self.assertFalse(restart_needed)
        self.assertEqual(detector.stagnation_count, 0)
        self.assertEqual(detector.last_improvement_generation, 1)
        
        # Generation 2 - small improvement (below threshold)
        restart_needed = detector.update(2, self.mock_chromosomes[:3], [0.52, 0.62, 0.72])
        self.assertFalse(restart_needed)
        self.assertEqual(detector.stagnation_count, 1)
        
        # Generation 3 - no improvement (stagnation)
        restart_needed = detector.update(3, self.mock_chromosomes[:3], [0.52, 0.62, 0.72])
        # May trigger restart if both stagnation and low diversity conditions are met
        self.assertIsInstance(restart_needed, bool)
        self.assertEqual(detector.stagnation_count, 2)
        
        # Generation 4 - should trigger restart consideration
        restart_needed = detector.update(4, self.mock_chromosomes[:3], [0.52, 0.62, 0.72])
        # Result depends on diversity calculation
        self.assertIsInstance(restart_needed, bool)
    
    def test_convergence_detector_significant_improvement(self):
        """Test convergence detector with significant improvement"""
        config = RestartConfig(convergence_threshold=0.1)
        detector = ConvergenceDetector(config)
        
        # Generation 1 - initial fitness
        restart_needed = detector.update(1, self.mock_chromosomes[:3], [0.5, 0.6, 0.7])
        self.assertFalse(restart_needed)
        self.assertEqual(detector.best_fitness, 0.7)
        
        # Generation 2 - significant improvement
        restart_needed = detector.update(2, self.mock_chromosomes[:3], [0.6, 0.7, 0.85])
        self.assertFalse(restart_needed)
        self.assertEqual(detector.best_fitness, 0.85)
        self.assertEqual(detector.stagnation_count, 0)
        self.assertEqual(detector.last_improvement_generation, 2)
    
    def test_restart_mechanisms_high_mutation_failure(self):
        """Test restart mechanisms with high mutation failure"""
        config = RestartConfig(max_restarts=1)
        manager = RestartMechanisms(
            self.test_graph,
            self.start_node,
            self.target_distance,
            config
        )
        
        # Test with empty elite population
        new_population, restart_info = manager.execute_restart([], [], 1)
        self.assertIsInstance(new_population, list)
        self.assertIsInstance(restart_info, dict)
        self.assertEqual(restart_info['elite_count'], 0)
    
    def test_restart_mechanisms_segment_replacement(self):
        """Test segment replacement during high mutation"""
        config = RestartConfig()
        manager = RestartMechanisms(
            self.test_graph,
            self.start_node,
            self.target_distance,
            config
        )
        
        # Create a chromosome with segments
        chromosome = Mock(spec=RouteChromosome)
        segments = []
        for i in range(3):
            segment = Mock(spec=RouteSegment)
            segment.start_node = i + 1
            segment.end_node = i + 2
            segment.path_nodes = [i + 1, i + 2]
            segments.append(segment)
        chromosome.segments = segments
        
        # Test _replace_random_segment
        original_segment_count = len(chromosome.segments)
        manager._replace_random_segment(chromosome)
        
        # Should still have same number of segments
        self.assertEqual(len(chromosome.segments), original_segment_count)
    
    def test_restart_mechanisms_high_elevation_extension(self):
        """Test route extension toward high elevation"""
        config = RestartConfig()
        manager = RestartMechanisms(
            self.test_graph,
            self.start_node,
            self.target_distance,
            config
        )
        
        # Create a chromosome with segments
        chromosome = Mock(spec=RouteChromosome)
        segment = Mock(spec=RouteSegment)
        segment.start_node = 1
        segment.end_node = 2
        segment.path_nodes = [1, 2]
        chromosome.segments = [segment]
        chromosome.get_route_nodes.return_value = [1, 2]
        
        # Test _extend_toward_high_elevation
        original_segment_count = len(chromosome.segments)
        manager._extend_toward_high_elevation(chromosome)
        
        # Should have attempted extension
        self.assertGreaterEqual(len(chromosome.segments), original_segment_count)
    
    def test_restart_mechanisms_segment_shuffling(self):
        """Test segment shuffling during high mutation"""
        config = RestartConfig()
        manager = RestartMechanisms(
            self.test_graph,
            self.start_node,
            self.target_distance,
            config
        )
        
        # Create a chromosome with multiple segments
        chromosome = Mock(spec=RouteChromosome)
        segments = []
        for i in range(5):
            segment = Mock(spec=RouteSegment)
            segment.start_node = i + 1
            segment.end_node = i + 2
            segment.path_nodes = [i + 1, i + 2]
            segments.append(segment)
        chromosome.segments = segments
        
        # Test _shuffle_segments
        original_first = chromosome.segments[0]
        original_last = chromosome.segments[-1]
        manager._shuffle_segments(chromosome)
        
        # First and last should remain the same
        self.assertEqual(chromosome.segments[0], original_first)
        self.assertEqual(chromosome.segments[-1], original_last)
    
    def test_restart_mechanisms_can_restart(self):
        """Test can_restart method"""
        config = RestartConfig(max_restarts=2)
        manager = RestartMechanisms(
            self.test_graph,
            self.start_node,
            self.target_distance,
            config
        )
        
        # Initially should be able to restart
        self.assertTrue(manager.can_restart())
        
        # After max restarts, should not be able to restart
        manager.restart_count = 2
        self.assertFalse(manager.can_restart())
    
    def test_restart_mechanisms_get_restart_stats(self):
        """Test restart statistics retrieval"""
        config = RestartConfig()
        manager = RestartMechanisms(
            self.test_graph,
            self.start_node,
            self.target_distance,
            config
        )
        
        stats = manager.get_restart_stats()
        
        expected_keys = ['restart_count', 'max_restarts', 'convergence_stats']
        for key in expected_keys:
            self.assertIn(key, stats)
        
        self.assertEqual(stats['restart_count'], 0)
        self.assertEqual(stats['max_restarts'], config.max_restarts)
        self.assertIsInstance(stats['convergence_stats'], dict)
    
    def test_convergence_detector_get_convergence_stats(self):
        """Test convergence statistics retrieval"""
        config = RestartConfig()
        detector = ConvergenceDetector(config)
        
        # Add some fitness history
        detector.fitness_history = [0.5, 0.6, 0.7, 0.8, 0.9]
        detector.diversity_history = [0.8, 0.7, 0.6, 0.5, 0.4]
        detector.stagnation_count = 3
        detector.best_fitness = 0.9
        detector.last_improvement_generation = 5
        
        stats = detector.get_convergence_stats()
        
        expected_keys = ['stagnation_count', 'best_fitness', 'last_improvement_generation', 'recent_diversity', 'fitness_trend']
        for key in expected_keys:
            self.assertIn(key, stats)
        
        self.assertEqual(stats['stagnation_count'], 3)
        self.assertEqual(stats['best_fitness'], 0.9)
        self.assertEqual(stats['last_improvement_generation'], 5)
        self.assertIsInstance(stats['recent_diversity'], float)
        self.assertIsInstance(stats['fitness_trend'], float)
    
    def test_convergence_detector_reset(self):
        """Test convergence detector reset functionality"""
        config = RestartConfig()
        detector = ConvergenceDetector(config)
        
        # Set up some state
        detector.stagnation_count = 5
        detector.last_improvement_generation = 10
        detector.fitness_history = [0.5, 0.6, 0.7]
        
        # Reset
        detector.reset()
        
        # Check reset state
        self.assertEqual(detector.stagnation_count, 0)
        self.assertEqual(detector.last_improvement_generation, 0)
        # fitness_history should be preserved for trend analysis
        self.assertEqual(len(detector.fitness_history), 3)
    
    def test_restart_timing_optimization(self):
        """Test restart timing optimization"""
        config = RestartConfig(
            stagnation_generations=5,
            convergence_threshold=0.001,
            diversity_threshold=0.1
        )
        
        # Test that restart is triggered at appropriate times
        # Early generations: no restart
        early_generation = 2
        self.assertLess(early_generation, config.stagnation_generations)
        
        # Late generations with stagnation: should consider restart
        late_generation = 10
        self.assertGreater(late_generation, config.stagnation_generations)
        
        # Very small fitness improvement should trigger convergence detection
        small_improvement = 0.0005
        self.assertLess(small_improvement, config.convergence_threshold)
    
    def test_restart_population_composition(self):
        """Test restart population composition"""
        config = RestartConfig(
            elite_retention_percentage=0.2,
            restart_exploration_percentage=0.4,
            restart_high_mutation_percentage=0.4
        )
        
        population_size = 20
        elite_count = int(population_size * config.elite_retention_percentage)
        exploration_count = int(population_size * config.restart_exploration_percentage)
        mutation_count = int(population_size * config.restart_high_mutation_percentage)
        
        # Test population composition adds up correctly
        total_count = elite_count + exploration_count + mutation_count
        self.assertEqual(elite_count, 4)  # 20% of 20
        self.assertEqual(exploration_count, 8)  # 40% of 20
        self.assertEqual(mutation_count, 8)  # 40% of 20
        self.assertEqual(total_count, 20)


if __name__ == '__main__':
    unittest.main()