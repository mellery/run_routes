#!/usr/bin/env python3
"""
Adaptive Mutation Rate Controller for Genetic Algorithm
Implements adaptive mutation rates based on convergence detection and diversity metrics
"""

import math
import statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .chromosome import RouteChromosome


@dataclass
class AdaptiveMutationConfig:
    """Configuration for adaptive mutation rate controller"""
    base_mutation_rate: float = 0.15  # Base mutation rate
    min_mutation_rate: float = 0.05   # Minimum mutation rate
    max_mutation_rate: float = 0.50   # Maximum mutation rate
    stagnation_threshold: int = 10    # Generations without improvement before considering stagnation
    diversity_threshold: float = 0.3  # Minimum diversity before boosting mutation
    stagnation_boost_factor: float = 1.5  # Multiplier for stagnation boost
    diversity_boost_factor: float = 1.8   # Multiplier for diversity boost
    convergence_boost_factor: float = 2.0  # Multiplier when population converges
    decay_rate: float = 0.95  # Rate at which boosts decay over time


class AdaptiveMutationController:
    """Controller that adapts mutation rates based on GA performance"""
    
    def __init__(self, config: Optional[AdaptiveMutationConfig] = None):
        """Initialize adaptive mutation controller
        
        Args:
            config: Configuration for adaptive mutation behavior
        """
        self.config = config or AdaptiveMutationConfig()
        self.reset()
    
    def reset(self):
        """Reset controller state for new optimization run"""
        self.current_mutation_rate = self.config.base_mutation_rate
        self.fitness_history = []
        self.diversity_history = []
        self.best_fitness_history = []
        self.stagnation_count = 0
        self.last_improvement_generation = 0
        self.boost_decay_factor = 1.0
        self.generation = 0
        
    def update(self, generation: int, population: List[RouteChromosome], 
               fitness_scores: List[float], best_fitness: float) -> float:
        """Update mutation rate based on current population state
        
        Args:
            generation: Current generation number
            population: Current population
            fitness_scores: Fitness scores for current population
            best_fitness: Best fitness achieved so far
            
        Returns:
            Updated mutation rate for next generation
        """
        self.generation = generation
        
        # Track fitness history
        avg_fitness = statistics.mean(fitness_scores) if fitness_scores else 0.0
        self.fitness_history.append(avg_fitness)
        self.best_fitness_history.append(best_fitness)
        
        # Calculate diversity
        diversity = self._calculate_population_diversity(population)
        self.diversity_history.append(diversity)
        
        # Check for improvement
        if self._has_improved(best_fitness):
            self.last_improvement_generation = generation
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
        
        # Calculate adaptive mutation rate
        self.current_mutation_rate = self._calculate_adaptive_rate(
            diversity, self.stagnation_count, generation
        )
        
        # Apply decay to boost factors
        self.boost_decay_factor *= self.config.decay_rate
        
        return self.current_mutation_rate
    
    def _has_improved(self, current_best: float) -> bool:
        """Check if fitness has improved significantly"""
        if len(self.best_fitness_history) < 2:
            return True
        
        previous_best = self.best_fitness_history[-2]
        improvement = current_best - previous_best
        
        # Consider improvement if change is > 0.1% of current fitness
        threshold = max(0.001, abs(current_best) * 0.001)
        return improvement > threshold
    
    def _calculate_population_diversity(self, population: List[RouteChromosome]) -> float:
        """Calculate diversity metric for the population
        
        Args:
            population: Current population
            
        Returns:
            Diversity score (0.0 = no diversity, 1.0 = maximum diversity)
        """
        if len(population) < 2:
            return 0.0
        
        # Calculate route diversity based on node usage
        all_nodes = set()
        route_node_sets = []
        
        for chromosome in population:
            route_nodes = set()
            for segment in chromosome.segments:
                route_nodes.add(segment.start_node)
                route_nodes.add(segment.end_node)
            route_node_sets.append(route_nodes)
            all_nodes.update(route_nodes)
        
        if not all_nodes:
            return 0.0
        
        # Calculate average pairwise diversity
        total_diversity = 0.0
        comparisons = 0
        
        for i in range(len(route_node_sets)):
            for j in range(i + 1, len(route_node_sets)):
                # Jaccard similarity coefficient
                intersection = len(route_node_sets[i] & route_node_sets[j])
                union = len(route_node_sets[i] | route_node_sets[j])
                
                if union > 0:
                    similarity = intersection / union
                    diversity = 1.0 - similarity
                    total_diversity += diversity
                    comparisons += 1
        
        return total_diversity / comparisons if comparisons > 0 else 0.0
    
    def _calculate_adaptive_rate(self, diversity: float, stagnation_count: int, 
                                generation: int) -> float:
        """Calculate adaptive mutation rate
        
        Args:
            diversity: Current population diversity
            stagnation_count: Number of generations without improvement
            generation: Current generation number
            
        Returns:
            Adaptive mutation rate
        """
        # Start with base rate
        rate = self.config.base_mutation_rate
        
        # Stagnation boost
        if stagnation_count >= self.config.stagnation_threshold:
            stagnation_boost = min(
                self.config.stagnation_boost_factor,
                1.0 + (stagnation_count - self.config.stagnation_threshold) * 0.1
            )
            rate *= stagnation_boost
        
        # Diversity boost
        if diversity < self.config.diversity_threshold:
            diversity_deficit = self.config.diversity_threshold - diversity
            diversity_boost = 1.0 + (diversity_deficit * self.config.diversity_boost_factor)
            rate *= diversity_boost
        
        # Convergence boost (when both stagnation and low diversity occur)
        if (stagnation_count >= self.config.stagnation_threshold and 
            diversity < self.config.diversity_threshold):
            rate *= self.config.convergence_boost_factor
        
        # Apply decay to prevent excessive mutation
        rate *= self.boost_decay_factor
        
        # Clamp to configured bounds
        rate = max(self.config.min_mutation_rate, 
                  min(self.config.max_mutation_rate, rate))
        
        return rate
    
    def get_mutation_strategies(self) -> Dict[str, float]:
        """Get mutation strategy weights based on current state
        
        Returns:
            Dictionary of mutation strategy names to weights
        """
        strategies = {
            'segment_replacement': 0.4,
            'route_extension': 0.3,
            'elevation_bias': 0.2,
            'long_range_exploration': 0.1
        }
        
        # Adjust strategy weights based on state
        if self.stagnation_count >= self.config.stagnation_threshold:
            # Favor more exploratory mutations during stagnation
            strategies['long_range_exploration'] = 0.3
            strategies['elevation_bias'] = 0.3
            strategies['segment_replacement'] = 0.2
            strategies['route_extension'] = 0.2
        
        if self.diversity_history and self.diversity_history[-1] < self.config.diversity_threshold:
            # Favor mutations that increase diversity
            strategies['long_range_exploration'] = 0.4
            strategies['route_extension'] = 0.3
            strategies['segment_replacement'] = 0.2
            strategies['elevation_bias'] = 0.1
        
        return strategies
    
    def get_mutation_intensity(self) -> str:
        """Get current mutation intensity level
        
        Returns:
            String describing mutation intensity ('low', 'medium', 'high', 'extreme')
        """
        if self.current_mutation_rate <= self.config.base_mutation_rate:
            return 'low'
        elif self.current_mutation_rate <= self.config.base_mutation_rate * 1.5:
            return 'medium'
        elif self.current_mutation_rate <= self.config.base_mutation_rate * 2.0:
            return 'high'
        else:
            return 'extreme'
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about adaptive mutation state
        
        Returns:
            Dictionary containing diagnostic information
        """
        current_diversity = self.diversity_history[-1] if self.diversity_history else 0.0
        
        return {
            'current_mutation_rate': self.current_mutation_rate,
            'base_mutation_rate': self.config.base_mutation_rate,
            'mutation_intensity': self.get_mutation_intensity(),
            'stagnation_count': self.stagnation_count,
            'current_diversity': current_diversity,
            'boost_decay_factor': self.boost_decay_factor,
            'generations_since_improvement': self.generation - self.last_improvement_generation,
            'mutation_strategies': self.get_mutation_strategies(),
            'fitness_trend': self._get_fitness_trend(),
            'diversity_trend': self._get_diversity_trend()
        }
    
    def _get_fitness_trend(self) -> str:
        """Get fitness trend description"""
        if len(self.fitness_history) < 5:
            return 'insufficient_data'
        
        recent_fitness = self.fitness_history[-5:]
        if recent_fitness[-1] > recent_fitness[0]:
            return 'improving'
        elif recent_fitness[-1] < recent_fitness[0]:
            return 'declining'
        else:
            return 'stagnant'
    
    def _get_diversity_trend(self) -> str:
        """Get diversity trend description"""
        if len(self.diversity_history) < 5:
            return 'insufficient_data'
        
        recent_diversity = self.diversity_history[-5:]
        if recent_diversity[-1] > recent_diversity[0]:
            return 'increasing'
        elif recent_diversity[-1] < recent_diversity[0]:
            return 'decreasing'
        else:
            return 'stable'
    
    def should_trigger_restart(self) -> bool:
        """Check if population should be restarted due to convergence
        
        Returns:
            True if restart is recommended
        """
        if len(self.diversity_history) < 10:
            return False
        
        # Check if diversity has been consistently low
        recent_diversity = self.diversity_history[-10:]
        avg_diversity = statistics.mean(recent_diversity)
        
        # Check if we've been stagnant for too long
        excessive_stagnation = self.stagnation_count > self.config.stagnation_threshold * 2
        
        # Check if mutation rate is at maximum but still no improvement
        max_mutation_reached = self.current_mutation_rate >= self.config.max_mutation_rate * 0.9
        
        return (avg_diversity < self.config.diversity_threshold * 0.5 and 
                excessive_stagnation and 
                max_mutation_reached)