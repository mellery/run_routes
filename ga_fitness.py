#!/usr/bin/env python3
"""
Genetic Algorithm Fitness Evaluation System
Evaluates chromosome fitness based on multiple objectives
"""

import math
from typing import List, Dict, Any, Optional
from enum import Enum
import numpy as np

from ga_chromosome import RouteChromosome


class FitnessObjective(Enum):
    """Supported fitness objectives"""
    DISTANCE = "distance"
    ELEVATION = "elevation"
    BALANCED = "balanced"
    SCENIC = "scenic"
    EFFICIENCY = "efficiency"


class GAFitnessEvaluator:
    """Fitness evaluation system for genetic algorithm route optimization"""
    
    def __init__(self, objective: str = "elevation", target_distance_km: float = 5.0):
        """Initialize fitness evaluator
        
        Args:
            objective: Primary optimization objective
            target_distance_km: Target route distance in kilometers
        """
        self.objective = FitnessObjective(objective.lower())
        self.target_distance_km = target_distance_km
        self.target_distance_m = target_distance_km * 1000
        
        # Fitness weights for different objectives
        self.weights = self._get_objective_weights()
        
        # Performance tracking
        self.evaluations = 0
        self.best_fitness = 0.0
        self.fitness_history = []
        
    def _get_objective_weights(self) -> Dict[str, float]:
        """Get fitness weights based on objective"""
        if self.objective == FitnessObjective.DISTANCE:
            return {
                'distance_penalty': 0.6,
                'elevation_reward': 0.1,
                'connectivity_bonus': 0.2,
                'diversity_bonus': 0.1
            }
        elif self.objective == FitnessObjective.ELEVATION:
            return {
                'distance_penalty': 0.2,
                'elevation_reward': 0.5,
                'connectivity_bonus': 0.2,
                'diversity_bonus': 0.1
            }
        elif self.objective == FitnessObjective.BALANCED:
            return {
                'distance_penalty': 0.3,
                'elevation_reward': 0.3,
                'connectivity_bonus': 0.2,
                'diversity_bonus': 0.2
            }
        elif self.objective == FitnessObjective.SCENIC:
            return {
                'distance_penalty': 0.1,
                'elevation_reward': 0.4,
                'connectivity_bonus': 0.2,
                'diversity_bonus': 0.3
            }
        elif self.objective == FitnessObjective.EFFICIENCY:
            return {
                'distance_penalty': 0.5,
                'elevation_reward': 0.2,
                'connectivity_bonus': 0.2,
                'diversity_bonus': 0.1
            }
        else:
            # Default to elevation
            return {
                'distance_penalty': 0.2,
                'elevation_reward': 0.5,
                'connectivity_bonus': 0.2,
                'diversity_bonus': 0.1
            }
    
    def evaluate_chromosome(self, chromosome: RouteChromosome) -> float:
        """Evaluate fitness of a single chromosome
        
        Args:
            chromosome: Chromosome to evaluate
            
        Returns:
            Fitness score (0.0 - 1.0, higher is better)
        """
        if not chromosome.is_valid or not chromosome.segments:
            chromosome.fitness = 0.0
            return 0.0
        
        # Calculate base metrics
        stats = chromosome.get_route_stats()
        
        # Distance component
        distance_score = self._calculate_distance_score(stats['total_distance_km'])
        
        # Elevation component
        elevation_score = self._calculate_elevation_score(
            stats['total_elevation_gain_m'],
            stats['max_grade_percent']
        )
        
        # Connectivity component
        connectivity_score = self._calculate_connectivity_score(chromosome)
        
        # Diversity component
        diversity_score = self._calculate_diversity_score(chromosome)
        
        # Combine scores using weights
        fitness = (
            distance_score * self.weights['distance_penalty'] +
            elevation_score * self.weights['elevation_reward'] +
            connectivity_score * self.weights['connectivity_bonus'] +
            diversity_score * self.weights['diversity_bonus']
        )
        
        # Normalize to 0-1 range
        fitness = max(0.0, min(1.0, fitness))
        
        # Cache fitness value
        chromosome.fitness = fitness
        
        # Update tracking
        self.evaluations += 1
        if fitness > self.best_fitness:
            self.best_fitness = fitness
        self.fitness_history.append(fitness)
        
        return fitness
    
    def evaluate_population(self, population: List[RouteChromosome]) -> List[float]:
        """Evaluate fitness of entire population
        
        Args:
            population: Population to evaluate
            
        Returns:
            List of fitness scores
        """
        fitness_scores = []
        
        for chromosome in population:
            fitness = self.evaluate_chromosome(chromosome)
            fitness_scores.append(fitness)
        
        return fitness_scores
    
    def _calculate_distance_score(self, distance_km: float) -> float:
        """Calculate distance-based score component"""
        if distance_km <= 0:
            return 0.0
        
        # Penalty for distance deviation from target
        distance_error = abs(distance_km - self.target_distance_km)
        distance_tolerance = self.target_distance_km * 0.1  # 10% tolerance
        
        if distance_error <= distance_tolerance:
            # Within tolerance - high score
            return 1.0 - (distance_error / distance_tolerance) * 0.1
        else:
            # Outside tolerance - exponential penalty
            excess_ratio = (distance_error - distance_tolerance) / self.target_distance_km
            return max(0.0, 0.9 - excess_ratio * 2.0)
    
    def _calculate_elevation_score(self, elevation_gain_m: float, max_grade_percent: float) -> float:
        """Calculate elevation-based score component"""
        if elevation_gain_m <= 0:
            return 0.0
        
        # Base elevation score (logarithmic for diminishing returns)
        elevation_score = math.log(1 + elevation_gain_m / 10) / math.log(1 + 500)  # Normalized to ~500m max
        
        # Penalty for excessive grades
        grade_penalty = 0.0
        if max_grade_percent > 15:  # Steep grades
            grade_penalty = min(0.3, (max_grade_percent - 15) / 100)
        
        return max(0.0, elevation_score - grade_penalty)
    
    def _calculate_connectivity_score(self, chromosome: RouteChromosome) -> float:
        """Calculate connectivity-based score component"""
        if not chromosome.segments:
            return 0.0
        
        # Base connectivity score
        connectivity_score = 1.0 if chromosome.is_valid else 0.0
        
        # Bonus for smooth connections (fewer sharp turns)
        smooth_connections = 0
        total_connections = len(chromosome.segments) - 1
        
        if total_connections > 0:
            for i in range(total_connections):
                current_segment = chromosome.segments[i]
                next_segment = chromosome.segments[i + 1]
                
                # Check if segments connect smoothly
                if current_segment.end_node == next_segment.start_node:
                    smooth_connections += 1
            
            smoothness_ratio = smooth_connections / total_connections
            connectivity_score *= (0.5 + 0.5 * smoothness_ratio)
        
        return connectivity_score
    
    def _calculate_diversity_score(self, chromosome: RouteChromosome) -> float:
        """Calculate diversity-based score component"""
        if not chromosome.segments:
            return 0.0
        
        # Direction diversity - variety of travel directions
        directions = set()
        for segment in chromosome.segments:
            if segment.direction:
                directions.add(segment.direction)
        
        direction_diversity = len(directions) / 8.0  # 8 cardinal directions max
        
        # Segment length diversity - variety of segment lengths
        if len(chromosome.segments) > 1:
            segment_lengths = [seg.length for seg in chromosome.segments]
            length_std = np.std(segment_lengths)
            length_mean = np.mean(segment_lengths)
            length_diversity = min(1.0, length_std / max(length_mean, 1.0))
        else:
            length_diversity = 0.0
        
        # Combine diversity components
        diversity_score = (direction_diversity + length_diversity) / 2.0
        
        return diversity_score
    
    def get_fitness_stats(self) -> Dict[str, Any]:
        """Get fitness evaluation statistics"""
        if not self.fitness_history:
            return {
                'evaluations': 0,
                'best_fitness': 0.0,
                'average_fitness': 0.0,
                'worst_fitness': 0.0,
                'fitness_std': 0.0
            }
        
        return {
            'evaluations': self.evaluations,
            'best_fitness': self.best_fitness,
            'average_fitness': np.mean(self.fitness_history),
            'worst_fitness': min(self.fitness_history),
            'fitness_std': np.std(self.fitness_history),
            'recent_improvement': self._calculate_recent_improvement()
        }
    
    def _calculate_recent_improvement(self) -> float:
        """Calculate recent fitness improvement trend"""
        if len(self.fitness_history) < 10:
            return 0.0
        
        # Compare last 10 evaluations to previous 10
        recent_avg = np.mean(self.fitness_history[-10:])
        previous_avg = np.mean(self.fitness_history[-20:-10]) if len(self.fitness_history) >= 20 else 0.0
        
        return recent_avg - previous_avg
    
    def reset_tracking(self):
        """Reset fitness tracking statistics"""
        self.evaluations = 0
        self.best_fitness = 0.0
        self.fitness_history = []
    
    def is_fitness_plateau(self, generations: int = 10, threshold: float = 0.001) -> bool:
        """Check if fitness has plateaued (convergence detection)
        
        Args:
            generations: Number of recent generations to check
            threshold: Minimum improvement threshold
            
        Returns:
            True if fitness has plateaued
        """
        if len(self.fitness_history) < generations:
            return False
        
        recent_improvement = self._calculate_recent_improvement()
        return abs(recent_improvement) < threshold


def test_fitness_evaluator():
    """Test function for fitness evaluator"""
    print("Testing GA Fitness Evaluator...")
    
    # Create test chromosome
    from ga_chromosome import RouteSegment, RouteChromosome
    
    segment1 = RouteSegment(1, 2, [1, 2])
    segment1.length = 1000.0
    segment1.elevation_gain = 50.0
    segment1.direction = "N"
    
    segment2 = RouteSegment(2, 3, [2, 3])
    segment2.length = 1500.0
    segment2.elevation_gain = 30.0
    segment2.direction = "E"
    
    chromosome = RouteChromosome([segment1, segment2])
    chromosome.is_valid = True
    
    # Test different objectives
    objectives = ["distance", "elevation", "balanced"]
    
    for objective in objectives:
        evaluator = GAFitnessEvaluator(objective, target_distance_km=2.5)
        fitness = evaluator.evaluate_chromosome(chromosome)
        print(f"✅ {objective.title()} objective fitness: {fitness:.3f}")
    
    # Test population evaluation
    population = [chromosome] * 5
    evaluator = GAFitnessEvaluator("elevation", 2.5)
    fitness_scores = evaluator.evaluate_population(population)
    print(f"✅ Population evaluation: {len(fitness_scores)} scores")
    
    # Test fitness statistics
    stats = evaluator.get_fitness_stats()
    print(f"✅ Fitness stats: {stats['evaluations']} evaluations, best: {stats['best_fitness']:.3f}")
    
    print("✅ All fitness evaluator tests completed")


if __name__ == "__main__":
    test_fitness_evaluator()