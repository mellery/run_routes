#!/usr/bin/env python3
"""
Restart Mechanisms for Genetic Algorithm
Implements convergence detection and population restart strategies to escape local optima
"""

import random
import math
import statistics
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .chromosome import RouteChromosome
from .terrain_aware_initialization import TerrainAwarePopulationInitializer, TerrainAwareConfig


@dataclass
class RestartConfig:
    """Configuration for restart mechanisms"""
    convergence_threshold: float = 0.01  # Fitness improvement threshold for convergence
    stagnation_generations: int = 10  # Generations without improvement before restart
    diversity_threshold: float = 0.2  # Minimum population diversity before restart
    elite_retention_percentage: float = 0.2  # Percentage of elite to retain during restart
    restart_exploration_percentage: float = 0.4  # Percentage of new population for exploration
    restart_high_mutation_percentage: float = 0.4  # Percentage for high mutation
    max_restarts: int = 3  # Maximum number of restarts per optimization
    restart_mutation_rate: float = 0.4  # High mutation rate for restart population


class ConvergenceDetector:
    """Detects when GA has converged prematurely and needs restart"""
    
    def __init__(self, config: RestartConfig):
        self.config = config
        self.fitness_history = []
        self.diversity_history = []
        self.stagnation_count = 0
        self.best_fitness = float('-inf')
        self.last_improvement_generation = 0
        
    def update(self, generation: int, population: List[RouteChromosome], 
               fitness_scores: List[float]) -> bool:
        """Update convergence metrics and detect if restart is needed
        
        Args:
            generation: Current generation number
            population: Current population
            fitness_scores: Fitness scores for population
            
        Returns:
            True if restart is needed, False otherwise
        """
        if not fitness_scores:
            return False
            
        # Update fitness tracking
        current_best = max(fitness_scores)
        avg_fitness = statistics.mean(fitness_scores)
        
        self.fitness_history.append(current_best)
        
        # Check for fitness improvement
        if current_best > self.best_fitness + self.config.convergence_threshold:
            self.best_fitness = current_best
            self.last_improvement_generation = generation
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
        
        # Calculate population diversity
        diversity = self._calculate_population_diversity(population)
        self.diversity_history.append(diversity)
        
        # Check restart conditions
        fitness_stagnation = self.stagnation_count >= self.config.stagnation_generations
        low_diversity = diversity < self.config.diversity_threshold
        
        if fitness_stagnation and low_diversity:
            print(f"   🔄 Restart triggered: stagnation={self.stagnation_count}, diversity={diversity:.3f}")
            return True
            
        return False
    
    def _calculate_population_diversity(self, population: List[RouteChromosome]) -> float:
        """Calculate population diversity based on route node overlap
        
        Args:
            population: Population to analyze
            
        Returns:
            Diversity score (0.0 = no diversity, 1.0 = maximum diversity)
        """
        if len(population) < 2:
            return 1.0
            
        # Get route nodes for each chromosome
        route_sets = []
        for chromosome in population:
            try:
                nodes = set(chromosome.get_route_nodes())
                if len(nodes) > 1:  # Valid route
                    route_sets.append(nodes)
            except:
                continue
                
        if len(route_sets) < 2:
            return 1.0
            
        # Calculate pairwise Jaccard distances
        distances = []
        for i in range(len(route_sets)):
            for j in range(i + 1, len(route_sets)):
                set1, set2 = route_sets[i], route_sets[j]
                
                # Jaccard distance = 1 - (intersection / union)
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                
                if union > 0:
                    jaccard_distance = 1.0 - (intersection / union)
                    distances.append(jaccard_distance)
        
        # Average pairwise distance as diversity measure
        if distances:
            return statistics.mean(distances)
        else:
            return 1.0
    
    def get_convergence_stats(self) -> Dict[str, any]:
        """Get convergence statistics"""
        recent_diversity = self.diversity_history[-5:] if len(self.diversity_history) >= 5 else self.diversity_history
        
        return {
            'stagnation_count': self.stagnation_count,
            'best_fitness': self.best_fitness,
            'last_improvement_generation': self.last_improvement_generation,
            'recent_diversity': statistics.mean(recent_diversity) if recent_diversity else 0.0,
            'fitness_trend': self._calculate_fitness_trend()
        }
    
    def _calculate_fitness_trend(self) -> float:
        """Calculate fitness improvement trend over recent generations"""
        if len(self.fitness_history) < 3:
            return 0.0
            
        recent_fitness = self.fitness_history[-5:]
        
        # Simple linear trend calculation
        x = list(range(len(recent_fitness)))
        y = recent_fitness
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        if n * sum_x2 - sum_x ** 2 == 0:
            return 0.0
            
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope
    
    def reset(self):
        """Reset convergence tracking after restart"""
        self.stagnation_count = 0
        self.last_improvement_generation = 0
        # Keep fitness history for trend analysis but reset stagnation tracking


class RestartMechanisms:
    """Implements population restart strategies to escape local optima"""
    
    def __init__(self, graph, start_node: int, target_distance_km: float, 
                 config: RestartConfig):
        self.graph = graph
        self.start_node = start_node
        self.target_distance_km = target_distance_km
        self.config = config
        self.restart_count = 0
        
        # Initialize convergence detector
        self.convergence_detector = ConvergenceDetector(config)
        
        # Initialize terrain-aware generator for exploration
        terrain_config = TerrainAwareConfig(
            elevation_gain_threshold=30.0,
            max_elevation_gain_threshold=60.0,
            high_elevation_percentage=0.5,
            very_high_elevation_percentage=0.3,
            exploration_radius_multiplier=1.8  # Wider exploration during restart
        )
        self.terrain_initializer = TerrainAwarePopulationInitializer(
            graph, start_node, target_distance_km, terrain_config
        )
        
    def check_restart_needed(self, generation: int, population: List[RouteChromosome],
                           fitness_scores: List[float]) -> bool:
        """Check if population restart is needed
        
        Args:
            generation: Current generation number
            population: Current population
            fitness_scores: Fitness scores for population
            
        Returns:
            True if restart is needed, False otherwise
        """
        if self.restart_count >= self.config.max_restarts:
            return False
            
        return self.convergence_detector.update(generation, population, fitness_scores)
    
    def execute_restart(self, population: List[RouteChromosome], 
                       fitness_scores: List[float],
                       generation: int) -> Tuple[List[RouteChromosome], Dict[str, any]]:
        """Execute population restart with targeted exploration
        
        Args:
            population: Current population
            fitness_scores: Current fitness scores
            generation: Current generation
            
        Returns:
            Tuple of (new_population, restart_info)
        """
        self.restart_count += 1
        population_size = len(population)
        
        print(f"   🔄 Executing restart {self.restart_count}/{self.config.max_restarts} at generation {generation}")
        
        # Step 1: Retain elite individuals
        elite_count = max(1, int(population_size * self.config.elite_retention_percentage))
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        elite_population = [population[i] for i in elite_indices]
        
        print(f"   👑 Retaining {elite_count} elite individuals")
        
        # Step 2: Find unexplored high-elevation nodes
        unexplored_peaks = self._find_unexplored_peaks(population)
        
        # Step 3: Generate exploration population
        exploration_count = int(population_size * self.config.restart_exploration_percentage)
        exploration_population = self._generate_exploration_population(
            exploration_count, unexplored_peaks
        )
        
        print(f"   🎯 Generated {len(exploration_population)} exploration routes toward {len(unexplored_peaks)} unexplored peaks")
        
        # Step 4: Generate high-mutation population
        mutation_count = int(population_size * self.config.restart_high_mutation_percentage)
        mutation_population = self._generate_high_mutation_population(
            elite_population, mutation_count
        )
        
        print(f"   🧬 Generated {len(mutation_population)} high-mutation routes")
        
        # Step 5: Combine populations
        new_population = elite_population + exploration_population + mutation_population
        
        # Step 6: Fill remaining slots with terrain-aware initialization
        remaining_count = population_size - len(new_population)
        if remaining_count > 0:
            additional_population = self.terrain_initializer.create_population(remaining_count)
            new_population.extend(additional_population)
            print(f"   🏔️  Generated {len(additional_population)} additional terrain-aware routes")
        
        # Ensure exact population size
        new_population = new_population[:population_size]
        
        # Step 7: Reset convergence detector
        self.convergence_detector.reset()
        
        # Step 8: Generate restart info
        restart_info = {
            'restart_number': self.restart_count,
            'generation': generation,
            'elite_count': len(elite_population),
            'exploration_count': len(exploration_population),
            'mutation_count': len(mutation_population),
            'additional_count': remaining_count,
            'unexplored_peaks': len(unexplored_peaks),
            'convergence_stats': self.convergence_detector.get_convergence_stats()
        }
        
        return new_population, restart_info
    
    def _find_unexplored_peaks(self, population: List[RouteChromosome]) -> List[int]:
        """Find high-elevation nodes not visited by current population
        
        Args:
            population: Current population to analyze
            
        Returns:
            List of unexplored high-elevation node IDs
        """
        # Get all nodes visited by population
        visited_nodes = set()
        for chromosome in population:
            try:
                nodes = chromosome.get_route_nodes()
                visited_nodes.update(nodes)
            except:
                continue
        
        # Find high-elevation nodes not in visited set
        start_elevation = self.graph.nodes[self.start_node].get('elevation', 0)
        elevation_threshold = start_elevation + 40  # 40m above start
        
        unexplored_peaks = []
        for node_id, node_data in self.graph.nodes(data=True):
            elevation = node_data.get('elevation', 0)
            if (elevation > elevation_threshold and 
                node_id not in visited_nodes and 
                node_id != self.start_node):
                unexplored_peaks.append(node_id)
        
        # Sort by elevation (highest first)
        unexplored_peaks.sort(key=lambda n: self.graph.nodes[n].get('elevation', 0), reverse=True)
        
        return unexplored_peaks[:20]  # Limit to top 20 unexplored peaks
    
    def _generate_exploration_population(self, count: int, 
                                       unexplored_peaks: List[int]) -> List[RouteChromosome]:
        """Generate population targeting unexplored high-elevation nodes
        
        Args:
            count: Number of exploration routes to generate
            unexplored_peaks: List of unexplored peak node IDs
            
        Returns:
            List of exploration route chromosomes
        """
        exploration_population = []
        
        if not unexplored_peaks:
            # If no unexplored peaks, generate wider terrain-aware routes
            return self.terrain_initializer.create_population(count)
        
        # Create routes toward unexplored peaks
        for i in range(count):
            if i < len(unexplored_peaks):
                target_node = unexplored_peaks[i]
            else:
                # Cycle through peaks if we need more routes
                target_node = unexplored_peaks[i % len(unexplored_peaks)]
            
            route = self._create_route_to_target(target_node)
            if route:
                route.creation_method = "restart_exploration"
                exploration_population.append(route)
        
        return exploration_population
    
    def _create_route_to_target(self, target_node: int) -> Optional[RouteChromosome]:
        """Create route to specific target node
        
        Args:
            target_node: Target node ID
            
        Returns:
            Route chromosome or None if creation fails
        """
        try:
            # Use terrain initializer's method for consistency
            target_info = {
                'node_id': target_node,
                'elevation': self.graph.nodes[target_node].get('elevation', 0),
                'elevation_gain': (self.graph.nodes[target_node].get('elevation', 0) - 
                                 self.graph.nodes[self.start_node].get('elevation', 0))
            }
            
            return self.terrain_initializer._create_route_to_target(target_info, "restart_exploration")
        except Exception as e:
            print(f"   ⚠️ Failed to create route to target {target_node}: {e}")
            return None
    
    def _generate_high_mutation_population(self, elite_population: List[RouteChromosome],
                                         count: int) -> List[RouteChromosome]:
        """Generate population with high mutation rates from elite individuals
        
        Args:
            elite_population: Elite individuals to mutate
            count: Number of mutated routes to generate
            
        Returns:
            List of highly mutated route chromosomes
        """
        mutation_population = []
        
        if not elite_population:
            return mutation_population
        
        for i in range(count):
            # Select random elite individual
            parent = random.choice(elite_population)
            
            # Create mutated copy
            mutated = parent.copy()
            
            # Apply high-intensity mutation
            mutated = self._high_intensity_mutation(mutated)
            
            if mutated:
                mutated.creation_method = "restart_high_mutation"
                mutation_population.append(mutated)
        
        return mutation_population
    
    def _high_intensity_mutation(self, chromosome: RouteChromosome) -> Optional[RouteChromosome]:
        """Apply high-intensity mutation to chromosome
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome or None if mutation fails
        """
        try:
            # Apply multiple mutations with high probability
            mutated = chromosome.copy()
            
            # Mutation 1: Replace multiple segments (50% chance)
            if random.random() < 0.5 and len(mutated.segments) > 2:
                num_replacements = random.randint(1, min(3, len(mutated.segments) // 2))
                for _ in range(num_replacements):
                    self._replace_random_segment(mutated)
            
            # Mutation 2: Extend route toward high elevation (40% chance)
            if random.random() < 0.4:
                self._extend_toward_high_elevation(mutated)
            
            # Mutation 3: Shuffle segment order (30% chance)
            if random.random() < 0.3 and len(mutated.segments) > 2:
                self._shuffle_segments(mutated)
            
            return mutated
        except Exception:
            return None
    
    def _replace_random_segment(self, chromosome: RouteChromosome):
        """Replace random segment with alternative path"""
        if len(chromosome.segments) < 2:
            return
            
        # Select random segment to replace
        segment_idx = random.randint(0, len(chromosome.segments) - 1)
        old_segment = chromosome.segments[segment_idx]
        
        # Find alternative path between same nodes
        try:
            import networkx as nx
            
            # Get multiple paths and select random one
            paths = list(nx.all_simple_paths(
                self.graph, 
                old_segment.start_node, 
                old_segment.end_node,
                cutoff=10  # Limit path length
            ))
            
            if len(paths) > 1:
                # Select different path
                alternative_path = random.choice([p for p in paths if p != old_segment.path_nodes])
                
                # Create new segment
                from .chromosome import RouteSegment
                new_segment = RouteSegment(
                    start_node=old_segment.start_node,
                    end_node=old_segment.end_node,
                    path_nodes=alternative_path
                )
                new_segment.calculate_properties(self.graph)
                
                chromosome.segments[segment_idx] = new_segment
                chromosome._invalidate_cache()
        except Exception:
            pass
    
    def _extend_toward_high_elevation(self, chromosome: RouteChromosome):
        """Extend route toward high elevation nodes"""
        if not chromosome.segments:
            return
            
        # Find highest elevation nodes near route
        route_nodes = set(chromosome.get_route_nodes())
        
        # Find nearby high-elevation nodes
        high_elevation_candidates = []
        for node_id, node_data in self.graph.nodes(data=True):
            if node_id not in route_nodes:
                elevation = node_data.get('elevation', 0)
                if elevation > 650:  # High elevation threshold
                    high_elevation_candidates.append((node_id, elevation))
        
        if high_elevation_candidates:
            # Select random high-elevation node
            target_node, _ = random.choice(high_elevation_candidates)
            
            # Try to add segment to this node
            try:
                import networkx as nx
                
                # Find connection from last node of route
                last_node = chromosome.segments[-1].end_node
                
                if nx.has_path(self.graph, last_node, target_node):
                    path = nx.shortest_path(self.graph, last_node, target_node, weight='length')
                    
                    if len(path) > 1:
                        from .chromosome import RouteSegment
                        new_segment = RouteSegment(
                            start_node=last_node,
                            end_node=target_node,
                            path_nodes=path
                        )
                        new_segment.calculate_properties(self.graph)
                        chromosome.add_segment(new_segment)
            except Exception:
                pass
    
    def _shuffle_segments(self, chromosome: RouteChromosome):
        """Shuffle segment order while maintaining connectivity"""
        if len(chromosome.segments) < 3:
            return
            
        # Simple shuffle of middle segments
        if len(chromosome.segments) >= 4:
            middle_segments = chromosome.segments[1:-1]
            random.shuffle(middle_segments)
            chromosome.segments = [chromosome.segments[0]] + middle_segments + [chromosome.segments[-1]]
            chromosome._invalidate_cache()
    
    def get_restart_stats(self) -> Dict[str, any]:
        """Get restart mechanism statistics"""
        return {
            'restart_count': self.restart_count,
            'max_restarts': self.config.max_restarts,
            'convergence_stats': self.convergence_detector.get_convergence_stats()
        }
    
    def can_restart(self) -> bool:
        """Check if more restarts are allowed"""
        return self.restart_count < self.config.max_restarts