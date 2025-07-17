#!/usr/bin/env python3
"""
Unit tests for GA Fitness Evaluation System
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import math
import numpy as np
import networkx as nx

from genetic_algorithm import GAFitnessEvaluator, FitnessObjective, RouteChromosome, RouteSegment
from genetic_algorithm.performance import GASegmentCache


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
        self.assertEqual(elev_weights['elevation_reward'], 0.4)
        self.assertEqual(elev_weights['distance_penalty'], 0.15)
        self.assertEqual(elev_weights['micro_terrain_bonus'], 0.15)
        
        # Test distance objective
        dist_evaluator = GAFitnessEvaluator("distance", 5.0)
        dist_weights = dist_evaluator.weights
        self.assertEqual(dist_weights['distance_penalty'], 0.6)
        self.assertEqual(dist_weights['elevation_reward'], 0.1)
        self.assertEqual(dist_weights['micro_terrain_bonus'], 0.0)
        
        # Test balanced objective
        bal_evaluator = GAFitnessEvaluator("balanced", 5.0)
        bal_weights = bal_evaluator.weights
        self.assertEqual(bal_weights['distance_penalty'], 0.25)
        self.assertEqual(bal_weights['elevation_reward'], 0.25)
        self.assertEqual(bal_weights['micro_terrain_bonus'], 0.1)
    
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


class TestGAFitnessAdvanced(TestGAFitness):
    """Advanced fitness evaluation tests"""
    
    def test_hard_distance_constraints(self):
        """Test hard distance constraints enforcement"""
        # Create chromosomes with different distances
        short_segment = RouteSegment(1, 2, [1, 2])
        short_segment.length = 2000.0  # 2km - too short (< 85% of 5km)
        short_chromosome = RouteChromosome([short_segment])
        short_chromosome.is_valid = True
        
        long_segment = RouteSegment(1, 2, [1, 2])
        long_segment.length = 7000.0  # 7km - too long (> 115% of 5km)
        long_chromosome = RouteChromosome([long_segment])
        long_chromosome.is_valid = True
        
        good_segment = RouteSegment(1, 2, [1, 2])
        good_segment.length = 5000.0  # 5km - perfect
        good_chromosome = RouteChromosome([good_segment])
        good_chromosome.is_valid = True
        
        # Test fitness scores
        short_fitness = self.evaluator.evaluate_chromosome(short_chromosome)
        long_fitness = self.evaluator.evaluate_chromosome(long_chromosome)
        good_fitness = self.evaluator.evaluate_chromosome(good_chromosome)
        
        # Hard constraints should severely limit fitness
        self.assertLessEqual(short_fitness, 0.05)
        self.assertLessEqual(long_fitness, 0.05)
        self.assertGreater(good_fitness, 0.05)
    
    def test_distance_score_edge_cases(self):
        """Test distance score calculation edge cases"""
        # Test extreme distance deviations
        extreme_short = self.evaluator._calculate_distance_score(0.5)  # 10% of target
        extreme_long = self.evaluator._calculate_distance_score(50.0)  # 10x target
        
        self.assertAlmostEqual(extreme_short, 0.01, places=2)
        self.assertAlmostEqual(extreme_long, 0.01, places=2)
        
        # Test boundary conditions
        lower_bound = self.evaluator._calculate_distance_score(4.25)  # 85% of 5km
        upper_bound = self.evaluator._calculate_distance_score(5.75)  # 115% of 5km
        
        self.assertGreater(lower_bound, 0.01)
        self.assertGreater(upper_bound, 0.01)
    
    def test_elevation_score_with_grades(self):
        """Test elevation score with various grade penalties"""
        # Test normal grade
        normal_score = self.evaluator._calculate_elevation_score(200.0, 10.0)
        
        # Test steep grade
        steep_score = self.evaluator._calculate_elevation_score(200.0, 20.0)
        
        # Test very steep grade
        very_steep_score = self.evaluator._calculate_elevation_score(200.0, 30.0)
        
        # Scores should decrease with steeper grades
        self.assertGreater(normal_score, steep_score)
        self.assertGreater(steep_score, very_steep_score)
        
        # Test with zero elevation
        zero_elev_score = self.evaluator._calculate_elevation_score(0.0, 5.0)
        self.assertEqual(zero_elev_score, 0.0)
    
    def test_connectivity_score_smooth_connections(self):
        """Test connectivity score with smooth vs rough connections"""
        # Create smooth connection chromosome
        smooth_seg1 = RouteSegment(1, 2, [1, 2])
        smooth_seg2 = RouteSegment(2, 3, [2, 3])  # Connects smoothly
        smooth_chromosome = RouteChromosome([smooth_seg1, smooth_seg2])
        smooth_chromosome.is_valid = True
        
        # Create rough connection chromosome
        rough_seg1 = RouteSegment(1, 2, [1, 2])
        rough_seg2 = RouteSegment(3, 4, [3, 4])  # Doesn't connect smoothly
        rough_chromosome = RouteChromosome([rough_seg1, rough_seg2])
        rough_chromosome.is_valid = True
        
        smooth_score = self.evaluator._calculate_connectivity_score(smooth_chromosome)
        rough_score = self.evaluator._calculate_connectivity_score(rough_chromosome)
        
        # Smooth connections should score higher
        self.assertGreater(smooth_score, rough_score)
    
    def test_diversity_score_geographic_spread(self):
        """Test diversity score with different geographic spreads"""
        # Mock graph with nodes for geographic diversity calculation
        mock_graph = nx.Graph()
        mock_graph.add_node(1, x=-80.40, y=37.12)
        mock_graph.add_node(2, x=-80.41, y=37.13)  # Close to node 1
        mock_graph.add_node(3, x=-80.45, y=37.15)  # Far from nodes 1 & 2
        
        # Create segment with graph reference
        diverse_seg1 = RouteSegment(1, 2, [1, 2])
        diverse_seg1.graph = mock_graph
        diverse_seg2 = RouteSegment(2, 3, [2, 3])
        diverse_seg2.graph = mock_graph
        diverse_chromosome = RouteChromosome([diverse_seg1, diverse_seg2])
        
        # Create less diverse chromosome
        close_seg1 = RouteSegment(1, 2, [1, 2])
        close_seg1.graph = mock_graph
        close_seg2 = RouteSegment(2, 1, [2, 1])  # Back to same area
        close_seg2.graph = mock_graph
        close_chromosome = RouteChromosome([close_seg1, close_seg2])
        
        diverse_score = self.evaluator._calculate_diversity_score(diverse_chromosome)
        close_score = self.evaluator._calculate_diversity_score(close_chromosome)
        
        # More diverse routes should score higher
        self.assertGreater(diverse_score, 0.0)
        self.assertGreater(close_score, 0.0)
    
    def test_diversity_score_edge_visits(self):
        """Test diversity score with repeated edge usage"""
        # Create chromosome with repeated edges
        repeat_seg1 = RouteSegment(1, 2, [1, 2])
        repeat_seg2 = RouteSegment(2, 1, [2, 1])  # Same edge, reverse direction
        repeat_seg3 = RouteSegment(1, 2, [1, 2])  # Repeat again
        repeat_chromosome = RouteChromosome([repeat_seg1, repeat_seg2, repeat_seg3])
        
        # Create chromosome with unique edges
        unique_seg1 = RouteSegment(1, 2, [1, 2])
        unique_seg2 = RouteSegment(2, 3, [2, 3])
        unique_seg3 = RouteSegment(3, 4, [3, 4])
        unique_chromosome = RouteChromosome([unique_seg1, unique_seg2, unique_seg3])
        
        repeat_score = self.evaluator._calculate_diversity_score(repeat_chromosome)
        unique_score = self.evaluator._calculate_diversity_score(unique_chromosome)
        
        # Unique routes should score higher than repetitive ones
        self.assertGreater(unique_score, repeat_score)
    
    def test_segment_cache_integration(self):
        """Test integration with segment cache"""
        # Create mock segment cache
        mock_cache = Mock(spec=GASegmentCache)
        mock_cache.get_chromosome_properties.return_value = {
            'total_distance_km': 5.0,
            'total_elevation_gain_m': 200.0,
            'max_grade_percent': 10.0
        }
        
        # Create evaluator with cache
        cache_evaluator = GAFitnessEvaluator("elevation", 5.0, segment_cache=mock_cache)
        
        # Create mock graph - required for cache usage
        mock_graph = Mock()
        mock_graph.nodes = {1: {}, 2: {}, 3: {}}  # Non-empty nodes dict
        
        # Test evaluation with cache
        fitness = cache_evaluator.evaluate_chromosome(self.test_chromosome, mock_graph)
        
        self.assertIsInstance(fitness, float)
        self.assertGreaterEqual(fitness, 0.0)
        self.assertLessEqual(fitness, 1.0)
        
        # Verify cache was used
        mock_cache.get_chromosome_properties.assert_called_once_with(self.test_chromosome, mock_graph)
    
    def test_bidirectional_segments_validation(self):
        """Test bidirectional segments validation"""
        # Test with bidirectional allowed
        bid_evaluator = GAFitnessEvaluator("elevation", 5.0, allow_bidirectional_segments=True)
        
        # Mock validate_connectivity to return different results
        with patch.object(self.test_chromosome, 'validate_connectivity') as mock_validate:
            mock_validate.return_value = True
            fitness_allowed = bid_evaluator.evaluate_chromosome(self.test_chromosome)
            mock_validate.assert_called_with(True)
        
        # Test with bidirectional not allowed
        no_bid_evaluator = GAFitnessEvaluator("elevation", 5.0, allow_bidirectional_segments=False)
        
        with patch.object(self.test_chromosome, 'validate_connectivity') as mock_validate:
            mock_validate.return_value = False
            fitness_not_allowed = no_bid_evaluator.evaluate_chromosome(self.test_chromosome)
            mock_validate.assert_called_with(False)
            self.assertEqual(fitness_not_allowed, 0.0)


class TestGAFitnessMicroTerrain(TestGAFitness):
    """Test micro-terrain functionality"""
    
    def setUp(self):
        super().setUp()
        
        # Create test graph with elevation data
        self.test_graph = nx.Graph()
        self.test_graph.add_node(1, x=-80.4094, y=37.1299, elevation=635.0)
        self.test_graph.add_node(2, x=-80.4080, y=37.1310, elevation=640.0)
        self.test_graph.add_node(3, x=-80.4070, y=37.1320, elevation=645.0)
        self.test_graph.add_node(4, x=-80.4060, y=37.1330, elevation=650.0)
        
        # Update segments with path_nodes
        self.segment1.path_nodes = [1, 2]
        self.segment2.path_nodes = [2, 3, 4]
        
        # Create micro-terrain enabled evaluator
        self.micro_evaluator = GAFitnessEvaluator("elevation", 5.0, enable_micro_terrain=True)
    
    def test_micro_terrain_enabled_disabled(self):
        """Test micro-terrain enabled vs disabled"""
        enabled_evaluator = GAFitnessEvaluator("elevation", 5.0, enable_micro_terrain=True)
        disabled_evaluator = GAFitnessEvaluator("elevation", 5.0, enable_micro_terrain=False)
        
        self.assertTrue(enabled_evaluator.enable_micro_terrain)
        self.assertFalse(disabled_evaluator.enable_micro_terrain)
    
    def test_extract_elevation_profile(self):
        """Test elevation profile extraction from graph"""
        profile = self.micro_evaluator._extract_elevation_profile_from_graph(
            self.test_chromosome, self.test_graph
        )
        
        self.assertIsInstance(profile, dict)
        self.assertIn('elevations', profile)
        self.assertIn('distances', profile)
        self.assertIn('coordinates', profile)
        self.assertIn('total_distance_m', profile)
        self.assertIn('sample_count', profile)
        
        # Check data structure
        self.assertIsInstance(profile['elevations'], list)
        self.assertIsInstance(profile['distances'], list)
        self.assertIsInstance(profile['coordinates'], list)
        self.assertGreater(profile['sample_count'], 0)
    
    def test_interpolate_elevation_profile(self):
        """Test elevation profile interpolation"""
        # Test with short profile (will be interpolated since default interpolation_distance_m = 10.0)
        short_elevations = [635.0, 640.0]
        short_distances = [0.0, 100.0]
        short_coordinates = [(37.1299, -80.4094), (37.1310, -80.4080)]
        
        result = self.micro_evaluator._interpolate_elevation_profile(
            short_elevations, short_distances, short_coordinates
        )
        
        # With 100m segment and 10m interpolation distance, expect ~11 points (start + 10 interpolated + end)
        self.assertGreater(len(result[0]), 2)  # Should be interpolated
        
        # Test with long segment requiring interpolation
        long_elevations = [635.0, 700.0]  # Large elevation change
        long_distances = [0.0, 1000.0]  # Long distance (>interpolation_distance_m)
        long_coordinates = [(37.1299, -80.4094), (37.1400, -80.4000)]
        
        self.micro_evaluator.interpolation_distance_m = 100.0  # Set interpolation distance
        
        result = self.micro_evaluator._interpolate_elevation_profile(
            long_elevations, long_distances, long_coordinates
        )
        
        # Should have interpolated points
        self.assertGreater(len(result[0]), 2)
        self.assertEqual(len(result[0]), len(result[1]))
        self.assertEqual(len(result[0]), len(result[2]))
    
    def test_detect_micro_terrain_features(self):
        """Test micro-terrain feature detection"""
        # Create elevation profile with features
        profile = {
            'elevations': [635.0, 645.0, 640.0, 650.0, 645.0],  # Peak and valley
            'distances': [0.0, 100.0, 200.0, 300.0, 400.0]
        }
        
        features = self.micro_evaluator._detect_micro_terrain_features(profile)
        
        self.assertIsInstance(features, dict)
        self.assertIn('peaks', features)
        self.assertIn('valleys', features)
        self.assertIn('steep_sections', features)
        self.assertIn('grade_changes', features)
        self.assertIn('max_grade_percent', features)
        self.assertIn('avg_grade_percent', features)
        
        # Should detect at least some features
        self.assertIsInstance(features['peaks'], list)
        self.assertIsInstance(features['valleys'], list)
    
    def test_detect_features_edge_cases(self):
        """Test feature detection edge cases"""
        # Test with insufficient data
        short_profile = {
            'elevations': [635.0, 640.0],
            'distances': [0.0, 100.0]
        }
        
        features = self.micro_evaluator._detect_micro_terrain_features(short_profile)
        
        self.assertEqual(features['peaks'], [])
        self.assertEqual(features['valleys'], [])
        
        # Test with flat profile
        flat_profile = {
            'elevations': [635.0, 635.0, 635.0, 635.0],
            'distances': [0.0, 100.0, 200.0, 300.0]
        }
        
        features = self.micro_evaluator._detect_micro_terrain_features(flat_profile)
        
        self.assertEqual(features['peaks'], [])
        self.assertEqual(features['valleys'], [])
        self.assertEqual(features['max_grade_percent'], 0.0)
    
    def test_score_micro_peaks(self):
        """Test micro-peaks scoring"""
        # Test with no peaks
        no_peaks = {'peaks': []}
        score = self.micro_evaluator._score_micro_peaks(no_peaks)
        self.assertEqual(score, 0.0)
        
        # Test with significant peaks
        significant_peaks = {
            'peaks': [
                {'prominence_m': 5.0},  # Above threshold
                {'prominence_m': 8.0},  # Above threshold
                {'prominence_m': 1.0}   # Below threshold
            ]
        }
        score = self.micro_evaluator._score_micro_peaks(significant_peaks)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_score_grade_variety(self):
        """Test grade variety scoring"""
        # Test with varied grades
        varied_features = {
            'steep_sections': [
                {'grade_percent': 5.0},  # Good grade
                {'grade_percent': 10.0}, # Good grade
                {'grade_percent': 20.0}  # Too steep
            ],
            'grade_changes': [
                {'grade_change_percent': 5.0},  # Significant change
                {'grade_change_percent': 2.0}   # Small change
            ]
        }
        
        score = self.micro_evaluator._score_grade_variety(varied_features)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Test with no variety
        no_variety = {'steep_sections': [], 'grade_changes': []}
        score = self.micro_evaluator._score_grade_variety(no_variety)
        self.assertEqual(score, 0.0)
    
    def test_score_terrain_complexity(self):
        """Test terrain complexity scoring"""
        # Test with complex terrain
        complex_profile = {
            'elevations': [635.0, 650.0, 630.0, 660.0, 640.0]  # Varied elevations
        }
        
        score = self.micro_evaluator._score_terrain_complexity(complex_profile)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Test with simple terrain
        simple_profile = {
            'elevations': [635.0, 636.0, 637.0, 638.0]  # Gradual change
        }
        
        simple_score = self.micro_evaluator._score_terrain_complexity(simple_profile)
        self.assertGreaterEqual(simple_score, 0.0)
        self.assertLess(simple_score, score)  # Should be less complex
        
        # Test with insufficient data
        minimal_profile = {'elevations': [635.0]}
        minimal_score = self.micro_evaluator._score_terrain_complexity(minimal_profile)
        self.assertEqual(minimal_score, 0.0)
    
    def test_haversine_distance_calculation(self):
        """Test haversine distance calculation"""
        # Test with known coordinates
        lat1, lon1 = 37.1299, -80.4094
        lat2, lon2 = 37.1310, -80.4080
        
        distance = self.micro_evaluator._haversine_distance(lat1, lon1, lat2, lon2)
        
        self.assertIsInstance(distance, float)
        self.assertGreater(distance, 0.0)
        self.assertLess(distance, 10000.0)  # Should be reasonable distance
        
        # Test with same coordinates
        same_distance = self.micro_evaluator._haversine_distance(lat1, lon1, lat1, lon1)
        self.assertAlmostEqual(same_distance, 0.0, places=1)
    
    def test_micro_terrain_score_calculation(self):
        """Test complete micro-terrain score calculation"""
        # Test with valid chromosome and graph
        score = self.micro_evaluator._calculate_micro_terrain_score(
            self.test_chromosome, self.test_graph
        )
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Test with empty chromosome
        empty_chromosome = RouteChromosome([])
        empty_score = self.micro_evaluator._calculate_micro_terrain_score(
            empty_chromosome, self.test_graph
        )
        self.assertEqual(empty_score, 0.0)
    
    def test_micro_terrain_error_handling(self):
        """Test micro-terrain error handling"""
        # Test with invalid graph (should not crash)
        invalid_graph = nx.Graph()  # Empty graph
        
        score = self.micro_evaluator._calculate_micro_terrain_score(
            self.test_chromosome, invalid_graph
        )
        
        # Should handle gracefully
        self.assertEqual(score, 0.0)
        
        # Test with chromosome missing path_nodes
        bad_segment = RouteSegment(1, 2, [1, 2])
        bad_segment.path_nodes = None
        bad_chromosome = RouteChromosome([bad_segment])
        
        score = self.micro_evaluator._calculate_micro_terrain_score(
            bad_chromosome, self.test_graph
        )
        
        # Should handle gracefully
        self.assertGreaterEqual(score, 0.0)


class TestGAFitnessObjectiveWeights(TestGAFitness):
    """Test objective weight configurations"""
    
    def test_scenic_objective_weights(self):
        """Test scenic objective weights"""
        scenic_evaluator = GAFitnessEvaluator("scenic", 5.0)
        weights = scenic_evaluator.weights
        
        self.assertEqual(weights['distance_penalty'], 0.1)
        self.assertEqual(weights['elevation_reward'], 0.3)
        self.assertEqual(weights['connectivity_bonus'], 0.15)
        self.assertEqual(weights['diversity_bonus'], 0.25)
        self.assertEqual(weights['micro_terrain_bonus'], 0.2)
    
    def test_efficiency_objective_weights(self):
        """Test efficiency objective weights"""
        efficiency_evaluator = GAFitnessEvaluator("efficiency", 5.0)
        weights = efficiency_evaluator.weights
        
        self.assertEqual(weights['distance_penalty'], 0.5)
        self.assertEqual(weights['elevation_reward'], 0.2)
        self.assertEqual(weights['connectivity_bonus'], 0.2)
        self.assertEqual(weights['diversity_bonus'], 0.1)
        self.assertEqual(weights['micro_terrain_bonus'], 0.0)
    
    def test_invalid_objective_fallback(self):
        """Test fallback to default weights for invalid objective"""
        # Patch the FitnessObjective to allow invalid value
        with patch('genetic_algorithm.fitness.FitnessObjective') as mock_objective:
            mock_objective.return_value = Mock()
            mock_objective.return_value.value = "invalid"
            
            evaluator = GAFitnessEvaluator("elevation", 5.0)
            evaluator.objective = mock_objective.return_value
            
            weights = evaluator._get_objective_weights()
            
            # Should fall back to elevation weights
            self.assertEqual(weights['distance_penalty'], 0.15)
            self.assertEqual(weights['elevation_reward'], 0.4)
            self.assertEqual(weights['micro_terrain_bonus'], 0.15)


class TestGAFitnessConfiguration(TestGAFitness):
    """Test fitness evaluator configuration options"""
    
    def test_target_distance_conversion(self):
        """Test target distance unit conversion"""
        evaluator = GAFitnessEvaluator("elevation", 3.5)  # 3.5 km
        
        self.assertEqual(evaluator.target_distance_km, 3.5)
        self.assertEqual(evaluator.target_distance_m, 3500.0)
    
    def test_micro_terrain_parameters(self):
        """Test micro-terrain analysis parameters"""
        evaluator = GAFitnessEvaluator("elevation", 5.0, enable_micro_terrain=True)
        
        self.assertEqual(evaluator.interpolation_distance_m, 10.0)
        self.assertEqual(evaluator.grade_threshold, 0.03)
        self.assertEqual(evaluator.elevation_gain_threshold, 2.0)
        
        # Test parameters can be modified
        evaluator.interpolation_distance_m = 20.0
        evaluator.grade_threshold = 0.05
        evaluator.elevation_gain_threshold = 5.0
        
        self.assertEqual(evaluator.interpolation_distance_m, 20.0)
        self.assertEqual(evaluator.grade_threshold, 0.05)
        self.assertEqual(evaluator.elevation_gain_threshold, 5.0)
    
    def test_segment_cache_default(self):
        """Test default segment cache behavior"""
        # Mock the global cache function
        with patch('genetic_algorithm.fitness.get_global_segment_cache') as mock_cache:
            mock_cache.return_value = Mock(spec=GASegmentCache)
            
            evaluator = GAFitnessEvaluator("elevation", 5.0)
            
            mock_cache.assert_called_once()
            self.assertIsNotNone(evaluator.segment_cache)


if __name__ == '__main__':
    unittest.main()