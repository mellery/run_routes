#!/usr/bin/env python3
"""
Genetic Algorithm Operators
Consolidated operators including standard GA operations and precision-aware enhancements
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
import math
import numpy as np
import networkx as nx
import logging
from typing import List, Tuple, Optional, Dict, Any

from .chromosome import RouteChromosome, RouteSegment
from ga_common_imports import (
    random_choice_weighted, validate_route_connectivity, 
    get_route_bounds, calculate_distance
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GAOperators:
    """Collection of genetic algorithm operators for route optimization"""
    
    def __init__(self, graph: nx.Graph, allow_bidirectional: bool = True):
        """Initialize genetic operators
        
        Args:
            graph: NetworkX graph with elevation and distance data
            allow_bidirectional: Whether to allow segments to be used in both directions
        """
        self.graph = graph
        self.segment_cache = {}  # Cache for segment creation
        self.allow_bidirectional = allow_bidirectional
        
    # =============================================================================
    # CROSSOVER OPERATORS
    # =============================================================================
    
    def segment_exchange_crossover(self, parent1: RouteChromosome, parent2: RouteChromosome,
                                  crossover_rate: float = 0.8) -> Tuple[RouteChromosome, RouteChromosome]:
        """Exchange segments between parents at common connection points
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome  
            crossover_rate: Probability of performing crossover
            
        Returns:
            Tuple of two offspring chromosomes
        """
        if random.random() > crossover_rate or not parent1.segments or not parent2.segments:
            # No crossover due to rate or empty parents, return copies with metadata
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()
            offspring1.creation_method = "segment_exchange_crossover"
            offspring2.creation_method = "segment_exchange_crossover"
            return offspring1, offspring2
        
        # Find common connection points
        parent1_nodes = set()
        parent2_nodes = set()
        
        for segment in parent1.segments:
            parent1_nodes.add(segment.start_node)
            parent1_nodes.add(segment.end_node)
        
        for segment in parent2.segments:
            parent2_nodes.add(segment.start_node)
            parent2_nodes.add(segment.end_node)
        
        common_nodes = parent1_nodes.intersection(parent2_nodes)
        
        if len(common_nodes) < 2:
            # No suitable crossover points, return copies
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()
            offspring1.creation_method = "segment_exchange_crossover"
            offspring2.creation_method = "segment_exchange_crossover"
            return offspring1, offspring2
        
        # Select crossover points
        crossover_nodes = random.sample(list(common_nodes), min(2, len(common_nodes)))
        
        # Create offspring by exchanging segments
        offspring1 = self._exchange_segments(parent1, parent2, crossover_nodes)
        offspring2 = self._exchange_segments(parent2, parent1, crossover_nodes)
        
        # Set creation method
        offspring1.creation_method = "segment_exchange_crossover"
        offspring2.creation_method = "segment_exchange_crossover"
        
        return offspring1, offspring2
    
    def _exchange_segments(self, parent1: RouteChromosome, parent2: RouteChromosome,
                          crossover_nodes: List[int]) -> RouteChromosome:
        """Exchange segments between two parents at specified nodes"""
        # Simple implementation: copy first parent and try to insert segments from second
        offspring = parent1.copy()
        
        # Find segments in parent2 that connect crossover nodes
        for segment in parent2.segments:
            if (segment.start_node in crossover_nodes or 
                segment.end_node in crossover_nodes):
                # Try to insert this segment if it doesn't conflict
                if not self._would_create_conflict(offspring, segment):
                    offspring.add_segment(segment.copy())
        
        return offspring
    
    def _would_create_conflict(self, chromosome: RouteChromosome, segment: RouteSegment) -> bool:
        """Check if adding a segment would create conflicts"""
        # Check if segment would create invalid bidirectional usage
        if not self.allow_bidirectional:
            edge_key = tuple(sorted([segment.start_node, segment.end_node]))
            for existing_segment in chromosome.segments:
                existing_edge = tuple(sorted([existing_segment.start_node, existing_segment.end_node]))
                if edge_key == existing_edge:
                    return True
        
        return False
    
    def path_splice_crossover(self, parent1: RouteChromosome, parent2: RouteChromosome,
                             crossover_rate: float = 0.8) -> Tuple[RouteChromosome, RouteChromosome]:
        """Splice paths from both parents to create offspring
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            crossover_rate: Probability of performing crossover
            
        Returns:
            Tuple of two offspring chromosomes
        """
        if random.random() > crossover_rate or not parent1.segments or not parent2.segments:
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()
            offspring1.creation_method = "path_splice_crossover"
            offspring2.creation_method = "path_splice_crossover"
            return offspring1, offspring2
        
        # Select splice points
        splice_point1 = random.randint(0, len(parent1.segments) - 1)
        splice_point2 = random.randint(0, len(parent2.segments) - 1)
        
        # Create offspring by splicing
        offspring1_segments = (parent1.segments[:splice_point1] + 
                             parent2.segments[splice_point2:])
        offspring2_segments = (parent2.segments[:splice_point2] + 
                             parent1.segments[splice_point1:])
        
        offspring1 = RouteChromosome(offspring1_segments)
        offspring2 = RouteChromosome(offspring2_segments)
        
        # Set creation method
        offspring1.creation_method = "path_splice_crossover"
        offspring2.creation_method = "path_splice_crossover"
        
        return offspring1, offspring2
    
    # =============================================================================
    # MUTATION OPERATORS
    # =============================================================================
    
    def segment_replacement_mutation(self, chromosome: RouteChromosome, 
                                   mutation_rate: float = 0.1) -> RouteChromosome:
        """Replace segments with alternative paths
        
        Args:
            chromosome: Chromosome to mutate
            mutation_rate: Probability of mutating each segment
            
        Returns:
            Mutated chromosome
        """
        if not chromosome.segments:
            return chromosome
        
        mutated = chromosome.copy()
        mutated.creation_method = "segment_replacement_mutation"
        
        for i, segment in enumerate(mutated.segments):
            if random.random() < mutation_rate:
                # Try to find alternative path
                alternative_segment = self._find_alternative_segment(
                    segment.start_node, segment.end_node)
                if alternative_segment:
                    mutated.segments[i] = alternative_segment
        
        return mutated
    
    def _find_alternative_segment(self, start_node: int, end_node: int) -> Optional[RouteSegment]:
        """Find alternative path between two nodes"""
        try:
            # Find alternative path using networkx
            if self.graph.has_edge(start_node, end_node):
                # Simple case: direct connection exists
                path = [start_node, end_node]
            else:
                # Find shortest path
                path = nx.shortest_path(self.graph, start_node, end_node)
            
            if len(path) >= 2:
                alternative_segment = RouteSegment(start_node, end_node, path)
                alternative_segment.calculate_properties(self.graph)
                return alternative_segment
                
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        
        return None
    
    def route_extension_mutation(self, chromosome: RouteChromosome, 
                               target_distance_km: float,
                               mutation_rate: float = 0.1) -> RouteChromosome:
        """Extend route with new segments
        
        Args:
            chromosome: Chromosome to mutate
            target_distance_km: Target distance for the route
            mutation_rate: Probability of extending the route
            
        Returns:
            Mutated chromosome
        """
        if random.random() > mutation_rate or not chromosome.segments:
            return chromosome
        
        mutated = chromosome.copy()
        mutated.creation_method = "route_extension_mutation"
        
        # Try to extend from the end
        last_segment = mutated.segments[-1]
        extension_segment = self._find_extension_segment(last_segment.end_node)
        
        if extension_segment:
            mutated.add_segment(extension_segment)
        
        return mutated
    
    def _find_extension_segment(self, from_node: int) -> Optional[RouteSegment]:
        """Find segment to extend route from given node"""
        try:
            # Get neighbors
            neighbors = list(self.graph.neighbors(from_node))
            if not neighbors:
                return None
            
            # Select random neighbor
            to_node = random.choice(neighbors)
            
            # Create segment
            extension_segment = RouteSegment(from_node, to_node, [from_node, to_node])
            extension_segment.calculate_properties(self.graph)
            return extension_segment
            
        except Exception:
            return None
    
    def elevation_bias_mutation(self, chromosome: RouteChromosome, 
                              mutation_rate: float = 0.1) -> RouteChromosome:
        """Mutate segments with bias toward elevation gain
        
        Args:
            chromosome: Chromosome to mutate
            mutation_rate: Probability of mutating each segment
            
        Returns:
            Mutated chromosome
        """
        if not chromosome.segments:
            return chromosome
        
        mutated = chromosome.copy()
        mutated.creation_method = "elevation_bias_mutation"
        
        for i, segment in enumerate(mutated.segments):
            if random.random() < mutation_rate:
                # Find higher elevation alternative
                elevation_alternative = self._find_elevation_alternative(
                    segment.start_node, segment.end_node)
                if elevation_alternative:
                    mutated.segments[i] = elevation_alternative
        
        return mutated
    
    def _find_elevation_alternative(self, start_node: int, end_node: int) -> Optional[RouteSegment]:
        """Find alternative path with higher elevation gain"""
        try:
            # Get neighbors of start node
            neighbors = list(self.graph.neighbors(start_node))
            if not neighbors:
                return None
            
            # Filter neighbors with higher elevation
            start_elevation = self.graph.nodes[start_node].get('elevation', 0)
            higher_neighbors = [n for n in neighbors 
                             if self.graph.nodes[n].get('elevation', 0) > start_elevation]
            
            if not higher_neighbors:
                return None
            
            # Select neighbor with highest elevation
            best_neighbor = max(higher_neighbors, 
                              key=lambda n: self.graph.nodes[n].get('elevation', 0))
            
            # Create segment
            alternative_segment = RouteSegment(start_node, best_neighbor, [start_node, best_neighbor])
            alternative_segment.calculate_properties(self.graph)
            return alternative_segment
            
        except Exception:
            return None
    
    # =============================================================================
    # SELECTION OPERATORS
    # =============================================================================
    
    def tournament_selection(self, population: List[RouteChromosome], 
                           tournament_size: int = 3) -> RouteChromosome:
        """Select individual using tournament selection
        
        Args:
            population: Population to select from
            tournament_size: Number of individuals in tournament
            
        Returns:
            Selected chromosome
        """
        if not population:
            raise ValueError("Population cannot be empty")
        
        # Select tournament participants
        tournament_size = min(tournament_size, len(population))
        tournament = random.sample(population, tournament_size)
        
        # Select best individual from tournament
        best = max(tournament, key=lambda x: x.fitness or 0)
        return best
    
    def elitism_selection(self, population: List[RouteChromosome], 
                         elite_size: int = 2) -> List[RouteChromosome]:
        """Select elite individuals
        
        Args:
            population: Population to select from
            elite_size: Number of elite individuals to select
            
        Returns:
            List of elite chromosomes
        """
        if not population:
            return []
        
        # Sort by fitness (descending)
        sorted_population = sorted(population, key=lambda x: x.fitness or 0, reverse=True)
        
        # Return top elite_size individuals
        return sorted_population[:elite_size]
    
    def diversity_selection(self, population: List[RouteChromosome], 
                          selection_size: int = 10) -> List[RouteChromosome]:
        """Select individuals based on diversity
        
        Args:
            population: Population to select from
            selection_size: Number of individuals to select
            
        Returns:
            List of diverse chromosomes
        """
        if not population:
            return []
        
        if len(population) <= selection_size:
            return population.copy()
        
        selected = []
        remaining = population.copy()
        
        # Select first individual (highest fitness)
        best = max(remaining, key=lambda x: x.fitness or 0)
        selected.append(best)
        remaining.remove(best)
        
        # Select remaining individuals based on diversity
        while len(selected) < selection_size and remaining:
            # Calculate diversity scores for remaining individuals
            diversity_scores = []
            for candidate in remaining:
                diversity_score = self._calculate_diversity_from_selected(candidate, selected)
                diversity_scores.append((candidate, diversity_score))
            
            # Select individual with highest diversity score
            best_diverse = max(diversity_scores, key=lambda x: x[1])
            selected.append(best_diverse[0])
            remaining.remove(best_diverse[0])
        
        return selected
    
    def _calculate_diversity_from_selected(self, candidate: RouteChromosome, 
                                         selected: List[RouteChromosome]) -> float:
        """Calculate diversity score of candidate relative to selected individuals"""
        if not selected:
            return 1.0
        
        # Simple diversity measure: route node overlap
        candidate_nodes = set(candidate.get_route_nodes())
        
        total_similarity = 0.0
        for selected_individual in selected:
            selected_nodes = set(selected_individual.get_route_nodes())
            overlap = len(candidate_nodes.intersection(selected_nodes))
            total_nodes = len(candidate_nodes.union(selected_nodes))
            
            if total_nodes > 0:
                similarity = overlap / total_nodes
                total_similarity += similarity
        
        # Diversity is inverse of average similarity
        avg_similarity = total_similarity / len(selected)
        return 1.0 - avg_similarity


class PrecisionAwareCrossover:
    """Crossover operators that leverage 1m elevation precision for better offspring"""
    
    def __init__(self, graph: nx.Graph, elevation_analyzer: Optional[Any] = None):
        """Initialize precision-aware crossover
        
        Args:
            graph: NetworkX graph with route network
            elevation_analyzer: Optional elevation analyzer for micro-terrain detection
        """
        self.graph = graph
        self.elevation_analyzer = elevation_analyzer
        
        # Crossover parameters
        self.micro_terrain_preference = 0.3  # Probability of preferring micro-terrain features
        self.elevation_similarity_threshold = 5.0  # meters
        
    def terrain_guided_crossover(self, parent1_route: List[int], parent2_route: List[int], 
                                target_distance_km: float = 5.0) -> Tuple[List[int], List[int]]:
        """Crossover that preserves micro-terrain features from both parents
        
        Args:
            parent1_route: First parent route (list of node IDs)
            parent2_route: Second parent route (list of node IDs)
            target_distance_km: Target distance for offspring
            
        Returns:
            Tuple of two offspring routes
        """
        try:
            # Find micro-terrain features in both parents
            parent1_features = self._identify_micro_terrain_features(parent1_route)
            parent2_features = self._identify_micro_terrain_features(parent2_route)
            
            # Create offspring by combining features
            offspring1 = self._combine_terrain_features(parent1_route, parent2_features, target_distance_km)
            offspring2 = self._combine_terrain_features(parent2_route, parent1_features, target_distance_km)
            
            return offspring1, offspring2
            
        except Exception as e:
            logger.warning(f"Terrain-guided crossover failed: {e}")
            # Fallback to simple crossover
            return self._simple_crossover(parent1_route, parent2_route)
    
    def _identify_micro_terrain_features(self, route: List[int]) -> List[Dict[str, Any]]:
        """Identify micro-terrain features in a route"""
        features = []
        
        if len(route) < 3:
            return features
        
        # Analyze elevation changes along the route
        for i in range(1, len(route) - 1):
            prev_node = route[i-1]
            curr_node = route[i]
            next_node = route[i+1]
            
            if (prev_node in self.graph.nodes and 
                curr_node in self.graph.nodes and 
                next_node in self.graph.nodes):
                
                prev_elev = self.graph.nodes[prev_node].get('elevation', 0)
                curr_elev = self.graph.nodes[curr_node].get('elevation', 0)
                next_elev = self.graph.nodes[next_node].get('elevation', 0)
                
                # Check for peaks and valleys
                if curr_elev > prev_elev and curr_elev > next_elev:
                    features.append({
                        'type': 'peak',
                        'node': curr_node,
                        'elevation': curr_elev,
                        'prominence': min(curr_elev - prev_elev, curr_elev - next_elev)
                    })
                elif curr_elev < prev_elev and curr_elev < next_elev:
                    features.append({
                        'type': 'valley',
                        'node': curr_node,
                        'elevation': curr_elev,
                        'depth': min(prev_elev - curr_elev, next_elev - curr_elev)
                    })
        
        return features
    
    def _combine_terrain_features(self, base_route: List[int], 
                                 other_features: List[Dict[str, Any]], 
                                 target_distance_km: float) -> List[int]:
        """Combine terrain features from another route into base route"""
        offspring = base_route.copy()
        
        # Try to incorporate significant terrain features
        for feature in other_features:
            if feature['type'] == 'peak' and feature.get('prominence', 0) > 5.0:
                # Try to route through significant peaks
                peak_node = feature['node']
                if peak_node not in offspring:
                    # Find insertion point
                    insertion_point = self._find_best_insertion_point(offspring, peak_node)
                    if insertion_point is not None:
                        offspring.insert(insertion_point, peak_node)
        
        return offspring
    
    def _find_best_insertion_point(self, route: List[int], new_node: int) -> Optional[int]:
        """Find best point to insert a new node in the route"""
        if not route:
            return None
        
        best_insertion = None
        min_distance_increase = float('inf')
        
        for i in range(len(route)):
            # Calculate distance increase for inserting at position i
            if i == 0:
                # Insert at beginning
                if len(route) > 0:
                    distance_increase = self._calculate_distance_between_nodes(new_node, route[0])
                else:
                    distance_increase = 0
            elif i == len(route):
                # Insert at end
                distance_increase = self._calculate_distance_between_nodes(route[-1], new_node)
            else:
                # Insert in middle
                prev_node = route[i-1]
                next_node = route[i]
                
                original_distance = self._calculate_distance_between_nodes(prev_node, next_node)
                new_distance = (self._calculate_distance_between_nodes(prev_node, new_node) +
                               self._calculate_distance_between_nodes(new_node, next_node))
                distance_increase = new_distance - original_distance
            
            if distance_increase < min_distance_increase:
                min_distance_increase = distance_increase
                best_insertion = i
        
        return best_insertion
    
    def _calculate_distance_between_nodes(self, node1: int, node2: int) -> float:
        """Calculate distance between two nodes"""
        try:
            if self.graph.has_edge(node1, node2):
                edge_data = self.graph[node1][node2]
                if 0 in edge_data:
                    return edge_data[0].get('length', 0)
                else:
                    return edge_data.get('length', 0)
            else:
                # Use shortest path
                path_length = nx.shortest_path_length(self.graph, node1, node2, weight='length')
                return path_length
        except:
            return float('inf')
    
    def _simple_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Simple crossover fallback"""
        if not parent1 or not parent2:
            return parent1.copy(), parent2.copy()
        
        # Single point crossover
        crossover_point1 = random.randint(1, len(parent1) - 1)
        crossover_point2 = random.randint(1, len(parent2) - 1)
        
        offspring1 = parent1[:crossover_point1] + parent2[crossover_point2:]
        offspring2 = parent2[:crossover_point2] + parent1[crossover_point1:]
        
        return offspring1, offspring2


class PrecisionAwareMutation:
    """Mutation operators that leverage 1m elevation precision"""
    
    def __init__(self, graph: nx.Graph):
        """Initialize precision-aware mutation
        
        Args:
            graph: NetworkX graph with route network
        """
        self.graph = graph
        self.elevation_threshold = 2.0  # meters
    
    def micro_terrain_mutation(self, route: List[int], mutation_rate: float = 0.1) -> List[int]:
        """Mutate route to explore micro-terrain features
        
        Args:
            route: Route to mutate
            mutation_rate: Probability of mutation
            
        Returns:
            Mutated route
        """
        if random.random() > mutation_rate or len(route) < 3:
            return route.copy()
        
        mutated = route.copy()
        
        # Find mutation point
        mutation_point = random.randint(1, len(mutated) - 2)
        
        # Find nearby nodes with interesting elevation
        current_node = mutated[mutation_point]
        neighbors = list(self.graph.neighbors(current_node))
        
        if neighbors:
            # Select neighbor with highest elevation
            current_elevation = self.graph.nodes[current_node].get('elevation', 0)
            best_neighbor = max(neighbors, 
                              key=lambda n: self.graph.nodes[n].get('elevation', 0))
            
            neighbor_elevation = self.graph.nodes[best_neighbor].get('elevation', 0)
            
            # Only mutate if significant elevation difference
            if neighbor_elevation - current_elevation > self.elevation_threshold:
                mutated[mutation_point] = best_neighbor
        
        return mutated


# Export main classes
__all__ = [
    'GAOperators',
    'PrecisionAwareCrossover',
    'PrecisionAwareMutation'
]