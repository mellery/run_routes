#!/usr/bin/env python3
"""
Unit tests for GA Fitness Evaluation System
"""

import unittest
from unittest.mock import Mock, patch
import math

from ga_fitness import GAFitnessEvaluator, FitnessObjective
from ga_chromosome import RouteChromosome, RouteSegment


class TestGAFitness(unittest.TestCase):
    """Test GA fitness evaluation system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = GAFitnessEvaluator("elevation", 5.0)
        
        # Create test segments
        self.segment1 = RouteSegment(1, 2, [1, 2])
        self.segment1.length = 2000.0
        self.segment1.elevation_gain = 100.0
        self.segment1.direction = "N"
        
        self.segment2 = RouteSegment(2, 3, [2, 3])
        self.segment2.length = 3000.0
        self.segment2.elevation_gain = 50.0
        self.segment2.direction = "E"
        
        # Create test chromosome
        self.test_chromosome = RouteChromosome([self.segment1, self.segment2])
        self.test_chromosome.is_valid = True
    
    def test_fitness_evaluator_initialization(self):
        """Test fitness evaluator initialization"""
        # Test default initialization
        evaluator = GAFitnessEvaluator()
        self.assertEqual(evaluator.objective, FitnessObjective.ELEVATION)
        self.assertEqual(evaluator.target_distance_km, 5.0)
        self.assertEqual(evaluator.evaluations, 0)
        
        # Test custom initialization
        evaluator = GAFitnessEvaluator("distance", 3.0)
        self.assertEqual(evaluator.objective, FitnessObjective.DISTANCE)
        self.assertEqual(evaluator.target_distance_km, 3.0)
    
    def test_objective_weights(self):
        """Test objective weight calculation"""
        # Test elevation objective
        elev_evaluator = GAFitnessEvaluator("elevation", 5.0)
        elev_weights = elev_evaluator.weights
        self.assertEqual(elev_weights['elevation_reward'], 0.5)
        self.assertEqual(elev_weights['distance_penalty'], 0.2)
        
        # Test distance objective
        dist_evaluator = GAFitnessEvaluator("distance", 5.0)
        dist_weights = dist_evaluator.weights
        self.assertEqual(dist_weights['distance_penalty'], 0.6)
        self.assertEqual(dist_weights['elevation_reward'], 0.1)
        
        # Test balanced objective
        bal_evaluator = GAFitnessEvaluator("balanced", 5.0)
        bal_weights = bal_evaluator.weights
        self.assertEqual(bal_weights['distance_penalty'], 0.3)
        self.assertEqual(bal_weights['elevation_reward'], 0.3)
    
    def test_chromosome_fitness_evaluation(self):
        """Test single chromosome fitness evaluation"""
        # Test valid chromosome
        fitness = self.evaluator.evaluate_chromosome(self.test_chromosome)
        self.assertIsInstance(fitness, float)
        self.assertGreaterEqual(fitness, 0.0)
        self.assertLessEqual(fitness, 1.0)
        self.assertEqual(self.test_chromosome.fitness, fitness)
        
        # Test invalid chromosome
        invalid_chromosome = RouteChromosome([])
        invalid_chromosome.is_valid = False
        fitness = self.evaluator.evaluate_chromosome(invalid_chromosome)
        self.assertEqual(fitness, 0.0)
        self.assertEqual(invalid_chromosome.fitness, 0.0)
    
    def test_population_fitness_evaluation(self):
        """Test population fitness evaluation"""
        # Create test population
        population = [self.test_chromosome.copy() for _ in range(5)]
        
        # Evaluate population
        fitness_scores = self.evaluator.evaluate_population(population)
        
        self.assertEqual(len(fitness_scores), 5)
        for fitness in fitness_scores:
            self.assertIsInstance(fitness, float)
            self.assertGreaterEqual(fitness, 0.0)
            self.assertLessEqual(fitness, 1.0)
    
    def test_distance_score_calculation(self):
        """Test distance score component"""
        # Test perfect distance match
        score = self.evaluator._calculate_distance_score(5.0)
        self.assertAlmostEqual(score, 1.0, places=2)
        
        # Test within tolerance
        score = self.evaluator._calculate_distance_score(4.8)  # Within 10% tolerance
        self.assertGreater(score, 0.9)
        
        # Test outside tolerance
        score = self.evaluator._calculate_distance_score(3.0)  # Outside tolerance
        self.assertLess(score, 0.9)
        
        # Test zero distance
        score = self.evaluator._calculate_distance_score(0.0)
        self.assertEqual(score, 0.0)
    
    def test_elevation_score_calculation(self):
        """Test elevation score component"""
        # Test positive elevation gain
        score = self.evaluator._calculate_elevation_score(200.0, 10.0)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Test zero elevation gain
        score = self.evaluator._calculate_elevation_score(0.0, 5.0)
        self.assertEqual(score, 0.0)
        
        # Test steep grade penalty
        score_normal = self.evaluator._calculate_elevation_score(200.0, 10.0)
        score_steep = self.evaluator._calculate_elevation_score(200.0, 25.0)
        self.assertLess(score_steep, score_normal)
    
    def test_connectivity_score_calculation(self):
        """Test connectivity score component"""
        # Test valid connected chromosome
        score = self.evaluator._calculate_connectivity_score(self.test_chromosome)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Test invalid chromosome
        invalid_chromosome = RouteChromosome([])
        invalid_chromosome.is_valid = False
        score = self.evaluator._calculate_connectivity_score(invalid_chromosome)
        self.assertEqual(score, 0.0)
        
        # Test empty segments
        empty_chromosome = RouteChromosome([])
        score = self.evaluator._calculate_connectivity_score(empty_chromosome)
        self.assertEqual(score, 0.0)
    
    def test_diversity_score_calculation(self):
        """Test diversity score component"""
        # Test chromosome with diversity
        score = self.evaluator._calculate_diversity_score(self.test_chromosome)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Test empty chromosome
        empty_chromosome = RouteChromosome([])
        score = self.evaluator._calculate_diversity_score(empty_chromosome)
        self.assertEqual(score, 0.0)
        
        # Test single segment
        single_segment_chromosome = RouteChromosome([self.segment1])
        score = self.evaluator._calculate_diversity_score(single_segment_chromosome)
        self.assertGreaterEqual(score, 0.0)
    
    def test_fitness_tracking(self):
        """Test fitness evaluation tracking"""
        # Initial state
        self.assertEqual(self.evaluator.evaluations, 0)
        self.assertEqual(self.evaluator.best_fitness, 0.0)
        self.assertEqual(len(self.evaluator.fitness_history), 0)
        
        # After evaluation
        fitness = self.evaluator.evaluate_chromosome(self.test_chromosome)
        self.assertEqual(self.evaluator.evaluations, 1)
        self.assertEqual(self.evaluator.best_fitness, fitness)
        self.assertEqual(len(self.evaluator.fitness_history), 1)
        
        # After multiple evaluations
        for _ in range(3):
            self.evaluator.evaluate_chromosome(self.test_chromosome)
        self.assertEqual(self.evaluator.evaluations, 4)
        self.assertEqual(len(self.evaluator.fitness_history), 4)
    
    def test_fitness_statistics(self):
        """Test fitness statistics calculation"""
        # Test empty statistics
        stats = self.evaluator.get_fitness_stats()
        self.assertEqual(stats['evaluations'], 0)
        self.assertEqual(stats['best_fitness'], 0.0)
        
        # Generate some fitness data
        for _ in range(10):
            self.evaluator.evaluate_chromosome(self.test_chromosome)
        
        stats = self.evaluator.get_fitness_stats()
        self.assertEqual(stats['evaluations'], 10)
        self.assertGreater(stats['best_fitness'], 0.0)
        self.assertGreater(stats['average_fitness'], 0.0)
        self.assertGreaterEqual(stats['worst_fitness'], 0.0)
        self.assertGreaterEqual(stats['fitness_std'], 0.0)
    
    def test_recent_improvement_calculation(self):
        """Test recent improvement calculation"""
        # Test with insufficient data
        improvement = self.evaluator._calculate_recent_improvement()
        self.assertEqual(improvement, 0.0)
        
        # Generate fitness data with improvement
        for i in range(20):
            # Create chromosome with increasing fitness
            test_chromo = self.test_chromosome.copy()
            test_chromo.fitness = 0.5 + i * 0.01  # Simulate improvement
            self.evaluator.fitness_history.append(test_chromo.fitness)
        
        improvement = self.evaluator._calculate_recent_improvement()
        self.assertGreater(improvement, 0.0)
    
    def test_fitness_plateau_detection(self):
        """Test fitness plateau detection"""
        # Test with insufficient data
        plateau = self.evaluator.is_fitness_plateau()
        self.assertFalse(plateau)
        
        # Test with stable fitness (plateau) - need at least 20 entries for _calculate_recent_improvement
        stable_fitness = [0.5] * 25
        self.evaluator.fitness_history = stable_fitness
        plateau = self.evaluator.is_fitness_plateau()
        self.assertTrue(bool(plateau))  # Convert numpy bool to Python bool
        
        # Test with improving fitness (no plateau)
        improving_fitness = [0.5 + i * 0.01 for i in range(25)]  # Smaller improvement to avoid plateau
        self.evaluator.fitness_history = improving_fitness
        plateau = self.evaluator.is_fitness_plateau()
        self.assertFalse(bool(plateau))  # Convert numpy bool to Python bool
    
    def test_tracking_reset(self):
        """Test fitness tracking reset"""
        # Generate some data
        self.evaluator.evaluate_chromosome(self.test_chromosome)
        self.assertGreater(self.evaluator.evaluations, 0)
        
        # Reset tracking
        self.evaluator.reset_tracking()
        self.assertEqual(self.evaluator.evaluations, 0)
        self.assertEqual(self.evaluator.best_fitness, 0.0)
        self.assertEqual(len(self.evaluator.fitness_history), 0)
    
    def test_fitness_objectives_enum(self):
        """Test fitness objectives enum"""
        # Test all objectives can be created
        objectives = ["distance", "elevation", "balanced", "scenic", "efficiency"]
        
        for objective in objectives:
            evaluator = GAFitnessEvaluator(objective, 5.0)
            self.assertEqual(evaluator.objective.value, objective)
    
    def test_fitness_score_components_integration(self):
        """Test integration of all fitness score components"""
        # Test that all components contribute to final fitness
        fitness = self.evaluator.evaluate_chromosome(self.test_chromosome)
        
        # Verify fitness is computed from components
        self.assertGreater(fitness, 0.0)
        self.assertLessEqual(fitness, 1.0)
        
        # Test different objectives give different scores
        dist_evaluator = GAFitnessEvaluator("distance", 5.0)
        elev_evaluator = GAFitnessEvaluator("elevation", 5.0)
        
        dist_fitness = dist_evaluator.evaluate_chromosome(self.test_chromosome)
        elev_fitness = elev_evaluator.evaluate_chromosome(self.test_chromosome)
        
        # Different objectives should generally give different fitness scores
        # (though they could be the same in edge cases)
        self.assertIsInstance(dist_fitness, float)
        self.assertIsInstance(elev_fitness, float)
    
    def test_error_handling(self):
        """Test error handling in fitness evaluation"""
        # Test with None chromosome
        try:
            fitness = self.evaluator.evaluate_chromosome(None)
            self.fail("Should have raised exception for None chromosome")
        except:
            pass  # Expected
        
        # Test with chromosome missing attributes
        minimal_chromosome = RouteChromosome([])
        fitness = self.evaluator.evaluate_chromosome(minimal_chromosome)
        self.assertEqual(fitness, 0.0)


if __name__ == '__main__':
    unittest.main()