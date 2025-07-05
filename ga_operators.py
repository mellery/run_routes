#!/usr/bin/env python3
"""
Genetic Algorithm Operators
Implements crossover, mutation, and selection operators for route optimization
"""

import random
import math
from typing import List, Tuple, Optional, Dict, Any
import networkx as nx
import numpy as np

from ga_chromosome import RouteChromosome, RouteSegment


class GAOperators:
    """Collection of genetic algorithm operators for route optimization"""
    
    def __init__(self, graph: nx.Graph):
        """Initialize genetic operators
        
        Args:
            graph: NetworkX graph with elevation and distance data
        """
        self.graph = graph
        self.segment_cache = {}  # Cache for segment creation
        
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
            offspring1.parent_ids = [id(parent1), id(parent2)]
            offspring2.parent_ids = [id(parent1), id(parent2)]
            return offspring1, offspring2
        
        # Find common nodes between parents
        common_nodes = self._find_common_nodes(parent1, parent2)
        
        if len(common_nodes) < 2:
            # No crossover possible, return copies with crossover metadata
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()
            offspring1.creation_method = "segment_exchange_crossover"
            offspring2.creation_method = "segment_exchange_crossover"
            offspring1.parent_ids = [id(parent1), id(parent2)]
            offspring2.parent_ids = [id(parent1), id(parent2)]
            return offspring1, offspring2
        
        # Select crossover points
        crossover_points = random.sample(common_nodes, 2)
        point1, point2 = min(crossover_points), max(crossover_points)
        
        # Create offspring by exchanging segments
        offspring1 = self._exchange_segments(parent1, parent2, point1, point2)
        offspring2 = self._exchange_segments(parent2, parent1, point1, point2)
        
        # Set metadata before repair
        offspring1.creation_method = "segment_exchange_crossover"
        offspring2.creation_method = "segment_exchange_crossover"
        offspring1.parent_ids = [id(parent1), id(parent2)]
        offspring2.parent_ids = [id(parent1), id(parent2)]
        
        # Repair and validate offspring
        offspring1 = self._repair_chromosome(offspring1)
        offspring2 = self._repair_chromosome(offspring2)
        
        return offspring1, offspring2
    
    def path_splice_crossover(self, parent1: RouteChromosome, parent2: RouteChromosome,
                             crossover_rate: float = 0.8) -> Tuple[RouteChromosome, RouteChromosome]:
        """Splice path segments from one parent into another
        
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
            offspring1.creation_method = "path_splice_crossover"
            offspring2.creation_method = "path_splice_crossover"
            offspring1.parent_ids = [id(parent1), id(parent2)]
            offspring2.parent_ids = [id(parent1), id(parent2)]
            return offspring1, offspring2
        
        # Select random segments to splice
        donor_segment_idx = random.randint(0, len(parent1.segments) - 1)
        donor_segment = parent1.segments[donor_segment_idx]
        
        # Find best insertion point in parent2
        insertion_point = self._find_best_insertion_point(parent2, donor_segment)
        
        if insertion_point is None:
            # No insertion point found, return copies with metadata
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()
            offspring1.creation_method = "path_splice_crossover"
            offspring2.creation_method = "path_splice_crossover"
            offspring1.parent_ids = [id(parent1), id(parent2)]
            offspring2.parent_ids = [id(parent1), id(parent2)]
            return offspring1, offspring2
        
        # Create offspring by splicing
        offspring1 = self._splice_segment(parent2, donor_segment, insertion_point)
        offspring2 = self._splice_segment(parent1, 
                                        parent2.segments[random.randint(0, len(parent2.segments) - 1)],
                                        self._find_best_insertion_point(parent1, parent2.segments[0]))
        
        # Handle None results and set metadata
        if offspring1 is None:
            offspring1 = parent2.copy()
        if offspring2 is None:
            offspring2 = parent1.copy()
        
        # Set metadata before repair
        offspring1.creation_method = "path_splice_crossover"
        offspring2.creation_method = "path_splice_crossover"
        offspring1.parent_ids = [id(parent1), id(parent2)]
        offspring2.parent_ids = [id(parent1), id(parent2)]
        
        # Repair and validate offspring
        offspring1 = self._repair_chromosome(offspring1)
        offspring2 = self._repair_chromosome(offspring2)
        
        return offspring1, offspring2
    
    # =============================================================================
    # MUTATION OPERATORS
    # =============================================================================
    
    def segment_replacement_mutation(self, chromosome: RouteChromosome, 
                                   mutation_rate: float = 0.1) -> RouteChromosome:
        """Replace segments with alternative paths between same nodes
        
        Args:
            chromosome: Chromosome to mutate
            mutation_rate: Probability of mutating each segment
            
        Returns:
            Mutated chromosome
        """
        if not chromosome.segments:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        mutations_made = 0
        
        for i, segment in enumerate(mutated.segments):
            if random.random() < mutation_rate:
                # Find alternative path between same nodes
                alternative_segment = self._find_alternative_segment(
                    segment.start_node, 
                    segment.end_node,
                    exclude_path=segment.path_nodes
                )
                
                if alternative_segment:
                    mutated.segments[i] = alternative_segment
                    mutations_made += 1
        
        if mutations_made > 0:
            mutated = self._repair_chromosome(mutated)
            mutated.creation_method = f"segment_replacement_mutation({mutations_made})"
        
        return mutated
    
    def route_extension_mutation(self, chromosome: RouteChromosome, 
                               target_distance_km: float,
                               mutation_rate: float = 0.1) -> RouteChromosome:
        """Add or remove segments to adjust route length toward target
        
        Args:
            chromosome: Chromosome to mutate
            target_distance_km: Target route distance
            mutation_rate: Probability of performing mutation
            
        Returns:
            Mutated chromosome
        """
        if random.random() > mutation_rate or not chromosome.segments:
            return chromosome.copy()
        
        current_distance = chromosome.get_total_distance() / 1000
        distance_error = current_distance - target_distance_km
        
        # Skip if already close to target
        tolerance = target_distance_km * 0.1
        if abs(distance_error) < tolerance:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        
        if distance_error < 0:
            # Route too short, add segments
            mutated = self._add_extension_segments(mutated, abs(distance_error) * 1000)
        else:
            # Route too long, remove segments  
            mutated = self._remove_excess_segments(mutated, target_distance_km)
        
        mutated = self._repair_chromosome(mutated)
        mutated.creation_method = f"route_extension_mutation({distance_error:+.2f}km)"
        
        return mutated
    
    def elevation_bias_mutation(self, chromosome: RouteChromosome,
                              objective: str = "elevation",
                              mutation_rate: float = 0.15) -> RouteChromosome:
        """Mutate routes to favor elevation objectives
        
        Args:
            chromosome: Chromosome to mutate
            objective: Route objective ("elevation", "distance", etc.)
            mutation_rate: Probability of performing mutation
            
        Returns:
            Mutated chromosome
        """
        if random.random() > mutation_rate or not chromosome.segments:
            return chromosome.copy()
        
        if objective != "elevation":
            return chromosome.copy()
        
        mutated = chromosome.copy()
        
        # Find segment with lowest elevation gain
        worst_segment_idx = min(
            range(len(mutated.segments)),
            key=lambda i: mutated.segments[i].elevation_gain
        )
        
        worst_segment = mutated.segments[worst_segment_idx]
        
        # Try to replace with elevation-seeking alternative using distant nodes
        distant_nodes = self._get_distant_reachable_nodes(worst_segment.start_node, min_distance=100, max_distance=1000)
        
        if distant_nodes:
            # Filter for nodes with higher elevation and select the best one
            elevation_candidates = []
            current_elevation = self.graph.nodes[worst_segment.start_node].get('elevation', 0)
            
            for node in distant_nodes:
                node_elevation = self.graph.nodes[node].get('elevation', 0)
                if node_elevation > current_elevation:
                    elevation_candidates.append((node, node_elevation))
            
            if elevation_candidates:
                # Select node with highest elevation
                best_node = max(elevation_candidates, key=lambda x: x[1])[0]
                new_segment = self._create_segment(worst_segment.start_node, best_node)
                
                if new_segment and new_segment.elevation_gain > worst_segment.elevation_gain:
                    mutated.segments[worst_segment_idx] = new_segment
                    mutated = self._repair_chromosome(mutated)
                    mutated.creation_method = "elevation_bias_mutation"
        
        return mutated
    
    # =============================================================================
    # SELECTION OPERATORS
    # =============================================================================
    
    def tournament_selection(self, population: List[RouteChromosome], 
                           tournament_size: int = 5) -> RouteChromosome:
        """Select parent using tournament selection
        
        Args:
            population: Population to select from
            tournament_size: Number of individuals in tournament
            
        Returns:
            Selected chromosome
        """
        if not population:
            raise ValueError("Cannot select from empty population")
        
        tournament_size = min(tournament_size, len(population))
        tournament = random.sample(population, tournament_size)
        
        # Select individual with highest fitness
        winner = max(tournament, key=lambda x: x.fitness if x.fitness is not None else 0)
        return winner
    
    def elitism_selection(self, population: List[RouteChromosome], 
                         elite_size: int = 10) -> List[RouteChromosome]:
        """Preserve best individuals across generations
        
        Args:
            population: Population to select from
            elite_size: Number of elite individuals to preserve
            
        Returns:
            List of elite chromosomes
        """
        if not population:
            return []
        
        # Sort by fitness (highest first)
        sorted_pop = sorted(population, 
                          key=lambda x: x.fitness if x.fitness is not None else 0, 
                          reverse=True)
        
        elite_size = min(elite_size, len(sorted_pop))
        return sorted_pop[:elite_size]
    
    def diversity_selection(self, population: List[RouteChromosome], 
                          selection_size: int = 50) -> List[RouteChromosome]:
        """Select individuals to maintain population diversity
        
        Args:
            population: Population to select from
            selection_size: Number of individuals to select
            
        Returns:
            List of selected chromosomes maintaining diversity
        """
        if not population:
            return []
        
        selection_size = min(selection_size, len(population))
        selected = []
        remaining = population.copy()
        
        # Always include best individual
        best = max(remaining, key=lambda x: x.fitness if x.fitness is not None else 0)
        selected.append(best)
        remaining.remove(best)
        
        # Select remaining individuals with diversity consideration
        for _ in range(selection_size - 1):
            if not remaining:
                break
            
            candidate = max(
                remaining,
                key=lambda x: self._calculate_selection_score(x, selected)
            )
            selected.append(candidate)
            remaining.remove(candidate)
        
        return selected
    
    # =============================================================================
    # HELPER METHODS
    # =============================================================================
    
    def _find_common_nodes(self, parent1: RouteChromosome, parent2: RouteChromosome) -> List[int]:
        """Find nodes that appear in both parent routes"""
        nodes1 = set()
        nodes2 = set()
        
        for segment in parent1.segments:
            nodes1.update(segment.path_nodes)
        
        for segment in parent2.segments:
            nodes2.update(segment.path_nodes)
        
        return list(nodes1.intersection(nodes2))
    
    def _exchange_segments(self, parent1: RouteChromosome, parent2: RouteChromosome,
                         point1: int, point2: int) -> RouteChromosome:
        """Exchange segments between crossover points"""
        # Find segment indices for crossover points
        p1_indices = self._find_segment_indices(parent1, point1, point2)
        p2_indices = self._find_segment_indices(parent2, point1, point2)
        
        if not p1_indices or not p2_indices:
            offspring = parent1.copy()
            offspring.creation_method = "segment_exchange_crossover"
            offspring.parent_ids = [id(parent1), id(parent2)]
            return offspring
        
        # Create offspring by combining segments
        offspring = RouteChromosome()
        offspring.creation_method = "segment_exchange_crossover"
        offspring.parent_ids = [id(parent1), id(parent2)]
        
        # Add segments before first crossover point from parent1
        for i in range(p1_indices[0]):
            offspring.add_segment(parent1.segments[i].copy())
        
        # Add segments between crossover points from parent2
        for i in range(p2_indices[0], p2_indices[1] + 1):
            offspring.add_segment(parent2.segments[i].copy())
        
        # Add segments after second crossover point from parent1
        for i in range(p1_indices[1] + 1, len(parent1.segments)):
            offspring.add_segment(parent1.segments[i].copy())
        
        return offspring
    
    def _find_segment_indices(self, chromosome: RouteChromosome, 
                            point1: int, point2: int) -> Optional[Tuple[int, int]]:
        """Find segment indices containing crossover points"""
        indices = []
        
        for i, segment in enumerate(chromosome.segments):
            if point1 in segment.path_nodes or point2 in segment.path_nodes:
                indices.append(i)
        
        if len(indices) >= 2:
            return (min(indices), max(indices))
        return None
    
    def _find_best_insertion_point(self, chromosome: RouteChromosome, 
                                 segment: RouteSegment) -> Optional[int]:
        """Find best insertion point for a segment"""
        if not chromosome.segments:
            return None
        
        # Look for nodes in the chromosome that could connect to the segment
        chromosome_nodes = set()
        for seg in chromosome.segments:
            chromosome_nodes.update(seg.path_nodes)
        
        # Find potential connection points
        if segment.start_node in chromosome_nodes:
            return segment.start_node
        elif segment.end_node in chromosome_nodes:
            return segment.end_node
        
        # Find closest node in chromosome to segment
        min_distance = float('inf')
        best_node = None
        
        for node in chromosome_nodes:
            if node in self.graph.nodes and segment.start_node in self.graph.nodes:
                try:
                    distance = nx.shortest_path_length(self.graph, node, segment.start_node, weight='length')
                    if distance < min_distance:
                        min_distance = distance
                        best_node = node
                except nx.NetworkXNoPath:
                    continue
        
        return best_node
    
    def _splice_segment(self, chromosome: RouteChromosome, segment: RouteSegment, 
                       insertion_point: int) -> Optional[RouteChromosome]:
        """Splice a segment into a chromosome at insertion point"""
        if not chromosome.segments or insertion_point is None:
            return None
        
        # Find insertion position
        insertion_idx = None
        for i, seg in enumerate(chromosome.segments):
            if insertion_point in seg.path_nodes:
                insertion_idx = i
                break
        
        if insertion_idx is None:
            return None
        
        # Create new chromosome with spliced segment
        offspring = RouteChromosome()
        
        # Add segments before insertion point
        for i in range(insertion_idx):
            offspring.add_segment(chromosome.segments[i].copy())
        
        # Add the spliced segment
        offspring.add_segment(segment.copy())
        
        # Add segments after insertion point
        for i in range(insertion_idx + 1, len(chromosome.segments)):
            offspring.add_segment(chromosome.segments[i].copy())
        
        return offspring
    
    def _find_alternative_segment(self, start_node: int, end_node: int, 
                                exclude_path: List[int] = None) -> Optional[RouteSegment]:
        """Find alternative path between nodes avoiding excluded path"""
        if exclude_path is None:
            exclude_path = []
        
        # Create temporary graph without excluded intermediate nodes
        temp_graph = self.graph.copy()
        for node in exclude_path[1:-1]:  # Keep start and end nodes
            if node in temp_graph:
                temp_graph.remove_node(node)
        
        try:
            path = nx.shortest_path(temp_graph, start_node, end_node, weight='length')
            if len(path) > 1 and path != exclude_path:
                return self._create_segment(start_node, end_node, path)
        except nx.NetworkXNoPath:
            pass
        
        return None
    
    def _create_segment(self, start_node: int, end_node: int, 
                       path: List[int] = None) -> Optional[RouteSegment]:
        """Create a route segment between nodes"""
        cache_key = (start_node, end_node)
        
        if cache_key in self.segment_cache:
            return self.segment_cache[cache_key].copy()
        
        if path is None:
            try:
                path = nx.shortest_path(self.graph, start_node, end_node, weight='length')
            except nx.NetworkXNoPath:
                return None
        
        segment = RouteSegment(start_node, end_node, path)
        segment.calculate_properties(self.graph)
        
        # Cache the segment
        self.segment_cache[cache_key] = segment
        
        return segment.copy()
    
    def _get_elevation_neighbors(self, node: int, max_distance: float = 1000) -> List[int]:
        """Get neighboring nodes with potential elevation gain"""
        if node not in self.graph.nodes:
            return []
        
        neighbors = []
        current_elevation = self.graph.nodes[node].get('elevation', 0)
        
        for neighbor in self.graph.neighbors(node):
            if self.graph.has_edge(node, neighbor):
                edge_data = self.graph[node][neighbor]
                edge_length = edge_data.get('length', 0) if not isinstance(edge_data, dict) or 0 not in edge_data else edge_data[0].get('length', 0)
                
                if edge_length <= max_distance:
                    neighbor_elevation = self.graph.nodes[neighbor].get('elevation', 0)
                    if neighbor_elevation > current_elevation:
                        neighbors.append(neighbor)
        
        return neighbors
    
    def _get_distant_reachable_nodes(self, base_node: int, min_distance: float = 200, 
                                   max_distance: float = 2000) -> List[int]:
        """Get nodes that are reachable but at a reasonable distance to create proper path segments"""
        if base_node not in self.graph.nodes:
            return []
        
        # Use faster BFS approach instead of calculating exact distances for all nodes
        import networkx as nx
        
        # Use BFS to find nodes at different hop distances
        all_neighbors = set()
        current_level = {base_node}
        
        # Expand 3-5 levels to get nodes at reasonable distances
        for level in range(1, 6):  # 1-5 hops away
            next_level = set()
            for node in current_level:
                neighbors = list(self.graph.neighbors(node))
                next_level.update(neighbors)
            
            # Remove already visited nodes
            next_level = next_level - all_neighbors - {base_node}
            all_neighbors.update(next_level)
            current_level = next_level
            
            if not next_level:  # No more nodes to explore
                break
        
        # Remove immediate neighbors to ensure we have multi-hop paths
        immediate_neighbors = set(self.graph.neighbors(base_node))
        distant_neighbors = all_neighbors - immediate_neighbors
        
        # Convert to list and limit for performance
        distant_list = list(distant_neighbors)
        if len(distant_list) > 50:
            distant_list = random.sample(distant_list, 50)
        
        return distant_list
    
    def _add_extension_segments(self, chromosome: RouteChromosome, 
                              target_extension: float) -> RouteChromosome:
        """Add segments to extend route length"""
        if not chromosome.segments:
            return chromosome
        
        extended = chromosome.copy()
        remaining_extension = target_extension
        
        # Try to add segments at random positions
        max_attempts = 5
        for _ in range(max_attempts):
            if remaining_extension <= 0:
                break
            
            # Find insertion point
            insertion_idx = random.randint(0, len(extended.segments))
            
            # Find nodes near insertion point for extension
            if insertion_idx < len(extended.segments):
                base_node = extended.segments[insertion_idx].start_node
            elif insertion_idx > 0:
                base_node = extended.segments[insertion_idx - 1].end_node
            else:
                continue
            
            # Find distant nodes for extension (not direct neighbors)
            distant_nodes = self._get_distant_reachable_nodes(base_node, min_distance=200, max_distance=remaining_extension)
            if distant_nodes:
                target_node = random.choice(distant_nodes)
                new_segment = self._create_segment(base_node, target_node)
                
                if new_segment and new_segment.length <= remaining_extension:
                    extended.segments.insert(insertion_idx, new_segment)
                    remaining_extension -= new_segment.length
        
        return extended
    
    def _remove_excess_segments(self, chromosome: RouteChromosome, 
                              target_distance_km: float) -> RouteChromosome:
        """Remove segments to reduce route length"""
        if not chromosome.segments:
            return chromosome
        
        reduced = chromosome.copy()
        target_distance = target_distance_km * 1000
        
        # Remove segments until we're close to target distance
        while len(reduced.segments) > 1:
            current_distance = reduced.get_total_distance()
            if current_distance <= target_distance * 1.1:  # 10% tolerance
                break
            
            # Remove shortest segment (least impact)
            shortest_idx = min(range(len(reduced.segments)), 
                             key=lambda i: reduced.segments[i].length)
            reduced.segments.pop(shortest_idx)
        
        return reduced
    
    def _repair_chromosome(self, chromosome: RouteChromosome) -> RouteChromosome:
        """Repair chromosome connectivity and validate"""
        if not chromosome.segments:
            return chromosome
        
        # Validate connectivity and repair if needed
        chromosome.validate_connectivity()
        
        # Recalculate properties
        for segment in chromosome.segments:
            segment.calculate_properties(self.graph)
        
        # Invalidate cached values
        chromosome._invalidate_cache()
        
        return chromosome
    
    def _calculate_selection_score(self, chromosome: RouteChromosome, 
                                 selected: List[RouteChromosome]) -> float:
        """Calculate selection score considering fitness and diversity"""
        base_fitness = chromosome.fitness if chromosome.fitness is not None else 0
        
        # Calculate diversity bonus
        diversity_bonus = 0
        for selected_chromo in selected:
            # Simple diversity measure based on route length difference
            length_diff = abs(chromosome.get_total_distance() - selected_chromo.get_total_distance())
            diversity_bonus += min(length_diff / 1000, 1.0)  # Normalize to km
        
        diversity_bonus = diversity_bonus / max(len(selected), 1)
        
        return base_fitness + diversity_bonus * 0.1


def test_ga_operators():
    """Test function for GA operators"""
    print("Testing GA Operators...")
    
    # Create a minimal test graph
    test_graph = nx.Graph()
    test_graph.add_node(1, x=-80.4094, y=37.1299, elevation=100)
    test_graph.add_node(2, x=-80.4000, y=37.1300, elevation=110)
    test_graph.add_node(3, x=-80.4050, y=37.1350, elevation=105)
    test_graph.add_edge(1, 2, length=100)
    test_graph.add_edge(2, 3, length=150)
    test_graph.add_edge(3, 1, length=200)
    
    operators = GAOperators(test_graph)
    print("✅ GA Operators created successfully")
    
    # Test chromosome creation
    from ga_chromosome import RouteSegment, RouteChromosome
    segment1 = RouteSegment(1, 2, [1, 2])
    segment1.calculate_properties(test_graph)
    segment2 = RouteSegment(2, 3, [2, 3])
    segment2.calculate_properties(test_graph)
    
    parent1 = RouteChromosome([segment1, segment2])
    parent1.fitness = 0.8
    parent2 = RouteChromosome([segment2])
    parent2.fitness = 0.6
    
    # Test crossover
    offspring1, offspring2 = operators.segment_exchange_crossover(parent1, parent2)
    print(f"✅ Crossover test completed: {len(offspring1.segments)}, {len(offspring2.segments)} segments")
    
    # Test mutation
    mutated = operators.segment_replacement_mutation(parent1)
    print(f"✅ Mutation test completed: {len(mutated.segments)} segments")
    
    # Test selection
    population = [parent1, parent2, offspring1, offspring2]
    selected = operators.tournament_selection(population)
    print(f"✅ Selection test completed: selected fitness = {selected.fitness}")
    
    print("✅ All GA operator tests completed")


if __name__ == "__main__":
    test_ga_operators()