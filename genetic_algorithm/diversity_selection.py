#!/usr/bin/env python3
"""
Diversity-Preserving Selection for Genetic Algorithm
Implements selection strategies that maintain population diversity and prevent premature convergence
"""

import random
import math
import statistics
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from .chromosome import RouteChromosome


@dataclass
class DiversitySelectionConfig:
    """Configuration for diversity-preserving selection"""
    diversity_threshold: float = 0.3  # Minimum diversity between selected individuals
    diversity_weight: float = 0.3  # Weight of diversity in selection decisions
    fitness_weight: float = 0.7  # Weight of fitness in selection decisions
    elite_percentage: float = 0.1  # Percentage of best individuals to always select
    novelty_window: int = 50  # Number of recent generations to consider for novelty
    spatial_diversity_weight: float = 0.4  # Weight of spatial diversity (node overlap)
    elevation_diversity_weight: float = 0.3  # Weight of elevation diversity
    distance_diversity_weight: float = 0.3  # Weight of distance diversity
    min_hamming_distance: int = 3  # Minimum Hamming distance between routes
    adaptive_threshold: bool = True  # Whether to adapt diversity threshold dynamically


class DiversityMetrics:
    """Calculates various diversity metrics for routes"""
    
    @staticmethod
    def spatial_diversity(route1: RouteChromosome, route2: RouteChromosome) -> float:
        """Calculate spatial diversity based on node overlap (Jaccard distance)
        
        Args:
            route1: First route chromosome
            route2: Second route chromosome
            
        Returns:
            Spatial diversity score (0.0 = identical, 1.0 = no overlap)
        """
        try:
            nodes1 = set(route1.get_route_nodes())
            nodes2 = set(route2.get_route_nodes())
            
            intersection = len(nodes1 & nodes2)
            union = len(nodes1 | nodes2)
            
            if union == 0:
                return 0.0
                
            # Jaccard distance = 1 - (intersection / union)
            return 1.0 - (intersection / union)
        except Exception:
            return 0.0
    
    @staticmethod
    def elevation_diversity(route1: RouteChromosome, route2: RouteChromosome) -> float:
        """Calculate elevation profile diversity
        
        Args:
            route1: First route chromosome
            route2: Second route chromosome
            
        Returns:
            Elevation diversity score (0.0 = identical, 1.0 = maximum difference)
        """
        try:
            # Get elevation gains
            gain1 = route1.get_elevation_gain()
            gain2 = route2.get_elevation_gain()
            
            # Get total distances
            dist1 = route1.get_total_distance()
            dist2 = route2.get_total_distance()
            
            # Calculate normalized differences
            max_gain = max(gain1, gain2, 1.0)  # Avoid division by zero
            max_dist = max(dist1, dist2, 1.0)
            
            gain_diversity = abs(gain1 - gain2) / max_gain
            dist_diversity = abs(dist1 - dist2) / max_dist
            
            # Combine elevation and distance diversity
            return (gain_diversity + dist_diversity) / 2.0
        except Exception:
            return 0.0
    
    @staticmethod
    def segment_diversity(route1: RouteChromosome, route2: RouteChromosome) -> float:
        """Calculate diversity based on segment structure
        
        Args:
            route1: First route chromosome
            route2: Second route chromosome
            
        Returns:
            Segment diversity score (0.0 = identical structure, 1.0 = completely different)
        """
        try:
            # Get segment endpoints
            edges1 = set()
            edges2 = set()
            
            for segment in route1.segments:
                edge = tuple(sorted([segment.start_node, segment.end_node]))
                edges1.add(edge)
            
            for segment in route2.segments:
                edge = tuple(sorted([segment.start_node, segment.end_node]))
                edges2.add(edge)
            
            # Calculate edge overlap
            intersection = len(edges1 & edges2)
            union = len(edges1 | edges2)
            
            if union == 0:
                return 0.0
                
            return 1.0 - (intersection / union)
        except Exception:
            return 0.0
    
    @staticmethod
    def combined_diversity(route1: RouteChromosome, route2: RouteChromosome,
                          config: DiversitySelectionConfig) -> float:
        """Calculate combined diversity score
        
        Args:
            route1: First route chromosome
            route2: Second route chromosome
            config: Diversity selection configuration
            
        Returns:
            Combined diversity score (0.0 = identical, 1.0 = maximum diversity)
        """
        spatial_div = DiversityMetrics.spatial_diversity(route1, route2)
        elevation_div = DiversityMetrics.elevation_diversity(route1, route2)
        segment_div = DiversityMetrics.segment_diversity(route1, route2)
        
        # Weighted combination
        combined = (config.spatial_diversity_weight * spatial_div +
                   config.elevation_diversity_weight * elevation_div +
                   config.distance_diversity_weight * segment_div)
        
        return combined
    
    @staticmethod
    def hamming_distance(route1: RouteChromosome, route2: RouteChromosome) -> int:
        """Calculate Hamming distance between routes
        
        Args:
            route1: First route chromosome
            route2: Second route chromosome
            
        Returns:
            Hamming distance (number of differing positions)
        """
        try:
            nodes1 = route1.get_route_nodes()
            nodes2 = route2.get_route_nodes()
            
            # Pad shorter route with None values
            max_len = max(len(nodes1), len(nodes2))
            nodes1 = nodes1 + [None] * (max_len - len(nodes1))
            nodes2 = nodes2 + [None] * (max_len - len(nodes2))
            
            # Count differences
            differences = sum(1 for n1, n2 in zip(nodes1, nodes2) if n1 != n2)
            return differences
        except Exception:
            return 0


class DiversityPreservingSelector:
    """Implements diversity-preserving selection strategies"""
    
    def __init__(self, config: DiversitySelectionConfig):
        self.config = config
        self.selection_history = []  # Track selection decisions
        self.diversity_history = []  # Track population diversity over time
        
    def select_population(self, population: List[RouteChromosome], 
                         fitness_scores: List[float],
                         generation: int) -> List[RouteChromosome]:
        """Select population while preserving diversity
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for population
            generation: Current generation number
            
        Returns:
            Selected population maintaining diversity
        """
        if not population or not fitness_scores:
            return population
            
        population_size = len(population)
        selected = []
        
        # Step 1: Select elite individuals (always include best performers)
        elite_count = max(1, int(population_size * self.config.elite_percentage))
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        elite_selected = [population[i] for i in elite_indices]
        selected.extend(elite_selected)
        
        print(f"   👑 Selected {elite_count} elite individuals")
        
        # Step 2: Select remaining individuals with diversity constraint
        remaining_count = population_size - len(selected)
        remaining_indices = [i for i in range(population_size) if i not in elite_indices]
        
        diverse_selected = self._select_diverse_individuals(
            population, fitness_scores, remaining_indices, selected, remaining_count, generation
        )
        selected.extend(diverse_selected)
        
        # Step 3: Track diversity metrics
        self._update_diversity_history(selected, generation)
        
        return selected[:population_size]
    
    def _select_diverse_individuals(self, population: List[RouteChromosome],
                                  fitness_scores: List[float], 
                                  candidate_indices: List[int],
                                  already_selected: List[RouteChromosome],
                                  target_count: int,
                                  generation: int) -> List[RouteChromosome]:
        """Select diverse individuals from candidates
        
        Args:
            population: Full population
            fitness_scores: Fitness scores
            candidate_indices: Indices of candidates to select from
            already_selected: Already selected individuals
            target_count: Number of individuals to select
            generation: Current generation
            
        Returns:
            List of diverse selected individuals
        """
        selected = []
        available_indices = candidate_indices.copy()
        current_threshold = self.config.diversity_threshold
        
        # Adapt threshold based on generation
        if self.config.adaptive_threshold:
            current_threshold = self._adapt_diversity_threshold(generation)
        
        for selection_round in range(target_count):
            if not available_indices:
                break
                
            # Find candidates that meet diversity criteria
            diverse_candidates = []
            
            for idx in available_indices:
                candidate = population[idx]
                
                # Check diversity against all selected individuals
                min_diversity = float('inf')
                
                # Check against already selected elite
                for selected_individual in already_selected:
                    diversity = DiversityMetrics.combined_diversity(
                        candidate, selected_individual, self.config
                    )
                    min_diversity = min(min_diversity, diversity)
                
                # Check against individuals selected in this round
                for selected_individual in selected:
                    diversity = DiversityMetrics.combined_diversity(
                        candidate, selected_individual, self.config
                    )
                    min_diversity = min(min_diversity, diversity)
                
                # Check Hamming distance constraint
                hamming_ok = True
                for selected_individual in already_selected + selected:
                    hamming_dist = DiversityMetrics.hamming_distance(candidate, selected_individual)
                    if hamming_dist < self.config.min_hamming_distance:
                        hamming_ok = False
                        break
                
                # Add to candidates if diversity criteria met
                if min_diversity >= current_threshold and hamming_ok:
                    diverse_candidates.append((idx, fitness_scores[idx], min_diversity))
            
            # Select best candidate among diverse options
            if diverse_candidates:
                # Score combines fitness and diversity
                scored_candidates = []
                for idx, fitness, diversity in diverse_candidates:
                    combined_score = (self.config.fitness_weight * fitness +
                                    self.config.diversity_weight * diversity)
                    scored_candidates.append((idx, combined_score))
                
                # Select best scoring candidate
                best_idx, best_score = max(scored_candidates, key=lambda x: x[1])
                selected.append(population[best_idx])
                available_indices.remove(best_idx)
                
            else:
                # No diverse candidates found - relax threshold
                current_threshold *= 0.8
                
                if current_threshold < 0.1:
                    # Threshold too low - select best remaining candidate
                    if available_indices:
                        remaining_fitness = [(idx, fitness_scores[idx]) for idx in available_indices]
                        best_idx, _ = max(remaining_fitness, key=lambda x: x[1])
                        selected.append(population[best_idx])
                        available_indices.remove(best_idx)
                        current_threshold = self.config.diversity_threshold  # Reset threshold
        
        print(f"   🎯 Selected {len(selected)} diverse individuals (threshold: {current_threshold:.3f})")
        
        return selected
    
    def _adapt_diversity_threshold(self, generation: int) -> float:
        """Adapt diversity threshold based on generation and history
        
        Args:
            generation: Current generation number
            
        Returns:
            Adapted diversity threshold
        """
        base_threshold = self.config.diversity_threshold
        
        # Increase threshold early in evolution to maintain diversity
        if generation < 10:
            multiplier = 1.5 - (generation / 20)  # 1.5 -> 1.0
            return base_threshold * multiplier
        
        # Decrease threshold later to allow convergence
        elif generation > 50:
            multiplier = max(0.5, 1.0 - (generation - 50) / 100)  # 1.0 -> 0.5
            return base_threshold * multiplier
        
        return base_threshold
    
    def _update_diversity_history(self, population: List[RouteChromosome], generation: int):
        """Update diversity history for tracking
        
        Args:
            population: Selected population
            generation: Current generation number
        """
        if len(population) < 2:
            return
            
        # Calculate average pairwise diversity
        diversities = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                diversity = DiversityMetrics.combined_diversity(
                    population[i], population[j], self.config
                )
                diversities.append(diversity)
        
        avg_diversity = statistics.mean(diversities) if diversities else 0.0
        
        self.diversity_history.append({
            'generation': generation,
            'avg_diversity': avg_diversity,
            'population_size': len(population),
            'min_diversity': min(diversities) if diversities else 0.0,
            'max_diversity': max(diversities) if diversities else 0.0
        })
        
        # Keep only recent history
        if len(self.diversity_history) > self.config.novelty_window:
            self.diversity_history.pop(0)
    
    def tournament_selection_with_diversity(self, population: List[RouteChromosome],
                                          fitness_scores: List[float],
                                          tournament_size: int = 3) -> RouteChromosome:
        """Tournament selection with diversity consideration
        
        Args:
            population: Population to select from
            fitness_scores: Fitness scores
            tournament_size: Size of tournament
            
        Returns:
            Selected individual
        """
        if len(population) <= tournament_size:
            # Select best individual if population is small
            best_idx = np.argmax(fitness_scores)
            return population[best_idx]
        
        # Select random tournament participants
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_individuals = [population[i] for i in tournament_indices]
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        # Calculate diversity bonus for each tournament participant
        scored_participants = []
        for i, (individual, fitness) in enumerate(zip(tournament_individuals, tournament_fitness)):
            # Calculate diversity relative to recent selections
            diversity_bonus = 0.0
            if len(self.selection_history) > 0:
                recent_selections = self.selection_history[-10:]  # Last 10 selections
                diversities = []
                for recent_selection in recent_selections:
                    diversity = DiversityMetrics.combined_diversity(
                        individual, recent_selection, self.config
                    )
                    diversities.append(diversity)
                
                diversity_bonus = statistics.mean(diversities) if diversities else 0.0
            
            # Combined score
            combined_score = (self.config.fitness_weight * fitness +
                            self.config.diversity_weight * diversity_bonus)
            scored_participants.append((i, combined_score))
        
        # Select best scoring participant
        best_idx, _ = max(scored_participants, key=lambda x: x[1])
        selected = tournament_individuals[best_idx]
        
        # Update selection history
        self.selection_history.append(selected)
        if len(self.selection_history) > self.config.novelty_window:
            self.selection_history.pop(0)
        
        return selected
    
    def get_diversity_stats(self) -> Dict[str, any]:
        """Get diversity statistics
        
        Returns:
            Dictionary with diversity metrics
        """
        if not self.diversity_history:
            return {}
        
        recent_history = self.diversity_history[-10:]
        
        return {
            'current_avg_diversity': recent_history[-1]['avg_diversity'] if recent_history else 0.0,
            'diversity_trend': self._calculate_diversity_trend(),
            'min_diversity': min(h['min_diversity'] for h in recent_history),
            'max_diversity': max(h['max_diversity'] for h in recent_history),
            'diversity_stability': self._calculate_diversity_stability(),
            'selection_history_size': len(self.selection_history)
        }
    
    def _calculate_diversity_trend(self) -> float:
        """Calculate diversity trend over recent generations"""
        if len(self.diversity_history) < 3:
            return 0.0
        
        recent_diversities = [h['avg_diversity'] for h in self.diversity_history[-5:]]
        
        # Simple linear trend
        x = list(range(len(recent_diversities)))
        y = recent_diversities
        
        n = len(x)
        if n < 2:
            return 0.0
            
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _calculate_diversity_stability(self) -> float:
        """Calculate diversity stability (inverse of variance)"""
        if len(self.diversity_history) < 3:
            return 1.0
        
        recent_diversities = [h['avg_diversity'] for h in self.diversity_history[-10:]]
        
        if len(recent_diversities) < 2:
            return 1.0
        
        variance = statistics.variance(recent_diversities)
        stability = 1.0 / (1.0 + variance)  # Higher stability = lower variance
        
        return stability


class NoveltySearch:
    """Implements novelty search to encourage exploration of new areas"""
    
    def __init__(self, config: DiversitySelectionConfig):
        self.config = config
        self.novelty_archive = []  # Archive of novel individuals
        self.max_archive_size = 100
        
    def calculate_novelty(self, individual: RouteChromosome,
                         population: List[RouteChromosome]) -> float:
        """Calculate novelty score for an individual
        
        Args:
            individual: Individual to evaluate
            population: Current population
            
        Returns:
            Novelty score (higher = more novel)
        """
        # Find k-nearest neighbors in behavior space
        k = min(15, len(population) + len(self.novelty_archive))
        
        # Calculate distances to all other individuals
        distances = []
        
        # Distances to population
        for other in population:
            if other != individual:
                distance = DiversityMetrics.combined_diversity(
                    individual, other, self.config
                )
                distances.append(distance)
        
        # Distances to novelty archive
        for archived in self.novelty_archive:
            distance = DiversityMetrics.combined_diversity(
                individual, archived, self.config
            )
            distances.append(distance)
        
        # Novelty = average distance to k nearest neighbors
        if distances:
            distances.sort(reverse=True)  # Sort by distance (descending)
            k_nearest = distances[:k]
            novelty = statistics.mean(k_nearest)
        else:
            novelty = 1.0  # Maximum novelty if no comparisons
        
        return novelty
    
    def update_archive(self, individual: RouteChromosome, novelty_score: float):
        """Update novelty archive with potentially novel individual
        
        Args:
            individual: Individual to potentially add
            novelty_score: Novelty score of individual
        """
        # Add to archive if novelty is high enough
        novelty_threshold = 0.5  # Minimum novelty for archiving
        
        if novelty_score > novelty_threshold:
            self.novelty_archive.append(individual.copy())
            
            # Maintain archive size
            if len(self.novelty_archive) > self.max_archive_size:
                # Remove oldest or least diverse individuals
                self.novelty_archive.pop(0)
    
    def get_archive_stats(self) -> Dict[str, any]:
        """Get novelty archive statistics"""
        return {
            'archive_size': len(self.novelty_archive),
            'max_archive_size': self.max_archive_size,
            'archive_diversity': self._calculate_archive_diversity()
        }
    
    def _calculate_archive_diversity(self) -> float:
        """Calculate diversity within novelty archive"""
        if len(self.novelty_archive) < 2:
            return 0.0
        
        diversities = []
        for i in range(len(self.novelty_archive)):
            for j in range(i + 1, len(self.novelty_archive)):
                diversity = DiversityMetrics.combined_diversity(
                    self.novelty_archive[i], self.novelty_archive[j], self.config
                )
                diversities.append(diversity)
        
        return statistics.mean(diversities) if diversities else 0.0