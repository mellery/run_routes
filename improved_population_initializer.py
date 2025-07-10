#!/usr/bin/env python3
"""
Improved Population Initialization Algorithm
Generates distance-compliant initial populations for genetic algorithm
"""

import random
import math
import networkx as nx
import numpy as np
from typing import List, Optional, Tuple, Set
from collections import defaultdict
import sys
import os

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from genetic_algorithm.chromosome import RouteChromosome, RouteSegment


class ImprovedPopulationInitializer:
    """Creates distance-compliant initial populations using targeted route building"""
    
    def __init__(self, graph: nx.Graph, start_node: int, allow_bidirectional: bool = True):
        """Initialize improved population creator
        
        Args:
            graph: NetworkX graph with elevation data
            start_node: Starting node for all routes
            allow_bidirectional: Whether to allow bidirectional segment usage
        """
        self.graph = graph
        self.start_node = start_node
        self.allow_bidirectional = allow_bidirectional
        
        # Cache for performance
        self._neighbor_cache = {}
        self._distance_cache = {}
        
        print(f"ImprovedPopulationInitializer created for start node {start_node}")
    
    def create_population(self, size: int, target_distance_km: float) -> List[RouteChromosome]:
        """Create distance-compliant initial population
        
        Args:
            size: Population size
            target_distance_km: Target route distance
            
        Returns:
            List of RouteChromosome objects (all meeting distance constraints)
        """
        population = []
        target_distance_m = target_distance_km * 1000
        
        # Distance tolerance (85%-115% of target)
        min_distance = target_distance_m * 0.85
        max_distance = target_distance_m * 1.15
        
        print(f"Creating distance-compliant population of {size} chromosomes")
        print(f"Target: {target_distance_km}km ({target_distance_m}m)")
        print(f"Tolerance: {min_distance/1000:.2f}km - {max_distance/1000:.2f}km")
        
        # Strategy distribution for compliant routes
        strategy_counts = {
            'outbound_return': int(size * 0.3),      # 30% - Simple out-and-back
            'loop_builder': int(size * 0.25),        # 25% - Loop construction
            'distance_walker': int(size * 0.2),      # 20% - Distance-targeted walking
            'elevation_seeker': int(size * 0.15),    # 15% - Elevation-focused with distance control
            'hybrid_explorer': int(size * 0.1)       # 10% - Mixed strategy
        }
        
        # Ensure we get exactly the right number
        remaining = size - sum(strategy_counts.values())
        strategy_counts['outbound_return'] += remaining
        
        attempts = 0
        max_attempts = size * 10  # Prevent infinite loops
        
        for strategy, count in strategy_counts.items():
            print(f"Creating {count} {strategy.replace('_', ' ')} routes...")
            
            for i in range(count):
                attempts += 1
                if attempts > max_attempts:
                    print(f"⚠️ Max attempts reached, stopping at {len(population)} routes")
                    break
                
                # Generate route using specific strategy
                chromosome = self._create_distance_compliant_route(
                    target_distance_m, min_distance, max_distance, strategy
                )
                
                if chromosome and self._validate_distance_constraint(chromosome, min_distance, max_distance):
                    chromosome.creation_method = strategy
                    chromosome.generation = 0
                    population.append(chromosome)
                    
                    if len(population) % 10 == 0:
                        print(f"  Generated {len(population)}/{size} compliant routes...")
                
                if len(population) >= size:
                    break
            
            if len(population) >= size:
                break
        
        print(f"✅ Created {len(population)}/{size} distance-compliant chromosomes")
        
        # Fill remaining slots with best-effort routes if needed
        while len(population) < size and attempts < max_attempts:
            attempts += 1
            chromosome = self._create_best_effort_route(target_distance_m, min_distance, max_distance)
            if chromosome:
                chromosome.creation_method = "best_effort"
                chromosome.generation = 0
                population.append(chromosome)
        
        return population[:size]
    
    def _create_distance_compliant_route(self, target_distance_m: float, 
                                       min_distance: float, max_distance: float,
                                       strategy: str) -> Optional[RouteChromosome]:
        """Create a route using specified strategy that meets distance constraints"""
        
        if strategy == 'outbound_return':
            return self._create_outbound_return_route(target_distance_m, min_distance, max_distance)
        elif strategy == 'loop_builder':
            return self._create_loop_builder_route(target_distance_m, min_distance, max_distance)
        elif strategy == 'distance_walker':
            return self._create_distance_walker_route(target_distance_m, min_distance, max_distance)
        elif strategy == 'elevation_seeker':
            return self._create_elevation_seeker_route(target_distance_m, min_distance, max_distance)
        elif strategy == 'hybrid_explorer':
            return self._create_hybrid_explorer_route(target_distance_m, min_distance, max_distance)
        else:
            return self._create_best_effort_route(target_distance_m, min_distance, max_distance)
    
    def _create_outbound_return_route(self, target_distance_m: float, 
                                    min_distance: float, max_distance: float) -> Optional[RouteChromosome]:
        """Create simple out-and-back route with precise distance targeting"""
        
        # Target: go out for half distance, then return
        outbound_target = target_distance_m / 2
        current_node = self.start_node
        outbound_segments = []
        outbound_distance = 0.0
        visited_nodes = {current_node}
        
        print(f"    Creating outbound-return route: target {target_distance_m}m, outbound target {outbound_target}m")
        
        # Build outbound path
        max_outbound_segments = 20
        for i in range(max_outbound_segments):
            if outbound_distance >= outbound_target * 0.9:  # Close enough to turn around
                print(f"    Reached outbound target: {outbound_distance:.1f}m >= {outbound_target * 0.9:.1f}m")
                break
            
            # Find neighbors and prefer those that get us closer to target distance
            neighbors = self._get_reachable_neighbors(current_node, max_distance=1000)
            print(f"    Step {i}: at node {current_node}, found {len(neighbors)} neighbors, distance so far: {outbound_distance:.1f}m")
            
            if not neighbors:
                print(f"    No neighbors found at node {current_node}")
                break
            
            # Filter unvisited neighbors (prefer exploration)
            unvisited = [n for n in neighbors if n not in visited_nodes]
            candidates = unvisited if unvisited else neighbors[:5]  # Fallback to any neighbors
            print(f"    Candidates: {len(unvisited)} unvisited, using {len(candidates)} candidates")
            
            # Select neighbor that best fits remaining distance
            remaining_outbound = outbound_target - outbound_distance
            best_neighbor = self._select_distance_fitting_neighbor(
                candidates, current_node, remaining_outbound
            )
            
            if not best_neighbor:
                print(f"    No suitable neighbor found")
                break
            
            # Create segment
            segment = self._create_segment(current_node, best_neighbor)
            if not segment or not segment.is_valid:
                print(f"    Failed to create valid segment to {best_neighbor}")
                break
            
            print(f"    Created segment: {current_node} -> {best_neighbor}, length: {segment.length:.1f}m")
            outbound_segments.append(segment)
            outbound_distance += segment.length
            visited_nodes.add(best_neighbor)
            current_node = best_neighbor
        
        print(f"    Outbound phase complete: {len(outbound_segments)} segments, {outbound_distance:.1f}m")
        
        # Return path - direct route back to start
        if current_node != self.start_node and outbound_segments:
            print(f"    Creating return segment from {current_node} to {self.start_node}")
            return_segment = self._create_segment(current_node, self.start_node)
            if return_segment and return_segment.is_valid:
                all_segments = outbound_segments + [return_segment]
                
                # Check if total distance is within constraints
                total_distance = sum(seg.length for seg in all_segments)
                print(f"    Total distance: {total_distance:.1f}m, range: {min_distance:.1f}m - {max_distance:.1f}m")
                
                if min_distance <= total_distance <= max_distance:
                    chromosome = RouteChromosome(all_segments)
                    chromosome.validate_connectivity()
                    print(f"    ✅ Route created successfully: {total_distance:.1f}m")
                    return chromosome
                else:
                    print(f"    ❌ Route distance outside constraints")
            else:
                print(f"    ❌ Failed to create return segment")
        else:
            print(f"    ❌ No outbound segments or already at start")
        
        return None
    
    def _create_loop_builder_route(self, target_distance_m: float,
                                 min_distance: float, max_distance: float) -> Optional[RouteChromosome]:
        """Build a route as a loop, monitoring distance throughout"""
        
        current_node = self.start_node
        segments = []
        total_distance = 0.0
        visited_nodes = {current_node}
        max_segments = 30
        
        # Build loop segments until we're close to target
        for segment_count in range(max_segments):
            remaining_distance = target_distance_m - total_distance
            
            # If we're close to target, try to return to start
            if remaining_distance <= target_distance_m * 0.2:  # Within 20% of target
                return_segment = self._create_segment(current_node, self.start_node)
                if (return_segment and return_segment.is_valid and 
                    min_distance <= total_distance + return_segment.length <= max_distance):
                    segments.append(return_segment)
                    chromosome = RouteChromosome(segments)
                    chromosome.validate_connectivity()
                    return chromosome
            
            # Select next node for loop building
            neighbors = self._get_reachable_neighbors(current_node, max_distance=800)
            if not neighbors:
                break
            
            # Prefer unvisited nodes, but allow revisiting
            unvisited = [n for n in neighbors if n not in visited_nodes]
            candidates = unvisited if unvisited and len(unvisited) > 2 else neighbors
            
            # Select based on remaining distance needs
            next_node = self._select_distance_fitting_neighbor(
                candidates, current_node, remaining_distance / max(1, max_segments - segment_count)
            )
            
            if not next_node:
                break
            
            # Create segment
            segment = self._create_segment(current_node, next_node)
            if not segment or not segment.is_valid:
                break
            
            # Check if adding this segment keeps us within bounds
            new_total = total_distance + segment.length
            if new_total > max_distance:
                break  # Would exceed maximum, stop here
            
            segments.append(segment)
            total_distance = new_total
            visited_nodes.add(next_node)
            current_node = next_node
        
        # Final attempt to close loop if not already at start
        if current_node != self.start_node and segments:
            return_segment = self._create_segment(current_node, self.start_node)
            if (return_segment and return_segment.is_valid and 
                min_distance <= total_distance + return_segment.length <= max_distance):
                segments.append(return_segment)
                chromosome = RouteChromosome(segments)
                chromosome.validate_connectivity()
                return chromosome
        
        return None
    
    def _create_distance_walker_route(self, target_distance_m: float,
                                    min_distance: float, max_distance: float) -> Optional[RouteChromosome]:
        """Create route by walking with precise distance monitoring"""
        
        current_node = self.start_node
        segments = []
        total_distance = 0.0
        max_segments = 25
        
        # Walk while monitoring distance closely
        for _ in range(max_segments):
            remaining_distance = target_distance_m - total_distance
            
            # If we're very close to target, try to finish
            if remaining_distance <= target_distance_m * 0.15:
                # Try to find a path back to start that uses remaining distance
                return_segment = self._find_distance_fitting_return(
                    current_node, remaining_distance * 0.8, remaining_distance * 1.2
                )
                if return_segment:
                    segments.append(return_segment)
                    total_distance += return_segment.length
                    if min_distance <= total_distance <= max_distance:
                        chromosome = RouteChromosome(segments)
                        chromosome.validate_connectivity()
                        return chromosome
                break
            
            # Find next segment that fits well with remaining distance
            neighbors = self._get_reachable_neighbors(current_node, max_distance=600)
            if not neighbors:
                break
            
            # Target segment length for this step
            target_segment_length = remaining_distance / max(1, max_segments - len(segments))
            
            best_neighbor = self._select_distance_fitting_neighbor(
                neighbors, current_node, target_segment_length
            )
            
            if not best_neighbor:
                break
            
            segment = self._create_segment(current_node, best_neighbor)
            if not segment or not segment.is_valid:
                break
            
            segments.append(segment)
            total_distance += segment.length
            current_node = best_neighbor
        
        return None
    
    def _create_elevation_seeker_route(self, target_distance_m: float,
                                     min_distance: float, max_distance: float) -> Optional[RouteChromosome]:
        """Create route focused on elevation gain while meeting distance constraints"""
        
        current_node = self.start_node
        segments = []
        total_distance = 0.0
        max_segments = 25
        
        for _ in range(max_segments):
            remaining_distance = target_distance_m - total_distance
            
            # If close to target, prioritize returning to start
            if remaining_distance <= target_distance_m * 0.2:
                return_segment = self._create_segment(current_node, self.start_node)
                if (return_segment and return_segment.is_valid and 
                    min_distance <= total_distance + return_segment.length <= max_distance):
                    segments.append(return_segment)
                    chromosome = RouteChromosome(segments)
                    chromosome.validate_connectivity()
                    return chromosome
                break
            
            # Find neighbors with elevation consideration
            neighbors = self._get_reachable_neighbors(current_node, max_distance=800)
            if not neighbors:
                break
            
            # Score neighbors by elevation gain and distance fit
            current_elevation = self.graph.nodes[current_node].get('elevation', 0.0)
            neighbor_scores = []
            
            for neighbor in neighbors:
                neighbor_elevation = self.graph.nodes[neighbor].get('elevation', 0.0)
                elevation_gain = max(0, neighbor_elevation - current_elevation)
                
                # Estimate segment distance
                try:
                    segment_distance = nx.shortest_path_length(
                        self.graph, current_node, neighbor, weight='length'
                    )
                    
                    # Distance fit score (prefer segments that fit remaining distance well)
                    target_segment = remaining_distance / max(1, max_segments - len(segments))
                    distance_fit = 1.0 / (1.0 + abs(segment_distance - target_segment) / target_segment)
                    
                    # Elevation score (normalized)
                    elevation_score = min(elevation_gain / 10.0, 1.0)  # 10m = max score
                    
                    # Combined score (60% distance fit, 40% elevation)
                    total_score = 0.6 * distance_fit + 0.4 * elevation_score
                    neighbor_scores.append((neighbor, total_score, segment_distance))
                    
                except nx.NetworkXNoPath:
                    continue
            
            if not neighbor_scores:
                break
            
            # Select best neighbor
            neighbor_scores.sort(key=lambda x: x[1], reverse=True)
            best_neighbor = neighbor_scores[0][0]
            
            segment = self._create_segment(current_node, best_neighbor)
            if not segment or not segment.is_valid:
                break
            
            if total_distance + segment.length > max_distance:
                break
            
            segments.append(segment)
            total_distance += segment.length
            current_node = best_neighbor
        
        return None
    
    def _create_hybrid_explorer_route(self, target_distance_m: float,
                                    min_distance: float, max_distance: float) -> Optional[RouteChromosome]:
        """Create route using mixed strategies with distance targeting"""
        
        # Randomly choose sub-strategy
        strategies = ['outbound_return', 'loop_builder', 'distance_walker']
        chosen_strategy = random.choice(strategies)
        
        return self._create_distance_compliant_route(
            target_distance_m, min_distance, max_distance, chosen_strategy
        )
    
    def _create_best_effort_route(self, target_distance_m: float,
                                min_distance: float, max_distance: float) -> Optional[RouteChromosome]:
        """Create best-effort route when other strategies fail"""
        
        # Simple strategy: extend outbound until we can return within constraints
        current_node = self.start_node
        segments = []
        total_distance = 0.0
        
        # Build outbound path
        for _ in range(15):
            if total_distance >= target_distance_m * 0.7:  # 70% outbound
                break
            
            neighbors = self._get_reachable_neighbors(current_node)
            if not neighbors:
                break
            
            next_node = random.choice(neighbors)
            segment = self._create_segment(current_node, next_node)
            
            if segment and segment.is_valid:
                segments.append(segment)
                total_distance += segment.length
                current_node = next_node
        
        # Return to start
        if current_node != self.start_node:
            return_segment = self._create_segment(current_node, self.start_node)
            if return_segment and return_segment.is_valid:
                total_with_return = total_distance + return_segment.length
                if min_distance <= total_with_return <= max_distance:
                    segments.append(return_segment)
                    chromosome = RouteChromosome(segments)
                    chromosome.validate_connectivity()
                    return chromosome
        
        return None
    
    # Helper methods
    def _select_distance_fitting_neighbor(self, neighbors: List[int], current_node: int, 
                                        target_segment_distance: float) -> Optional[int]:
        """Select neighbor that best fits target segment distance"""
        
        if not neighbors:
            return None
        
        neighbor_distances = []
        for neighbor in neighbors:
            try:
                distance = nx.shortest_path_length(
                    self.graph, current_node, neighbor, weight='length'
                )
                fit_score = 1.0 / (1.0 + abs(distance - target_segment_distance) / max(target_segment_distance, 100))
                neighbor_distances.append((neighbor, distance, fit_score))
            except nx.NetworkXNoPath:
                continue
        
        if not neighbor_distances:
            return random.choice(neighbors)
        
        # Sort by fit score and add some randomness
        neighbor_distances.sort(key=lambda x: x[2], reverse=True)
        
        # Select from top 3 candidates
        top_candidates = neighbor_distances[:min(3, len(neighbor_distances))]
        return random.choice(top_candidates)[0]
    
    def _find_distance_fitting_return(self, current_node: int, 
                                    min_return_distance: float, max_return_distance: float) -> Optional[RouteSegment]:
        """Find a return path to start that fits distance constraints"""
        
        # Try direct return first
        try:
            direct_distance = nx.shortest_path_length(
                self.graph, current_node, self.start_node, weight='length'
            )
            if min_return_distance <= direct_distance <= max_return_distance:
                return self._create_segment(current_node, self.start_node)
        except nx.NetworkXNoPath:
            pass
        
        # Try indirect return through intermediate nodes
        neighbors = self._get_reachable_neighbors(current_node, max_distance=1000)
        for intermediate in neighbors[:10]:  # Try up to 10 intermediate nodes
            try:
                # Path: current -> intermediate -> start
                dist1 = nx.shortest_path_length(self.graph, current_node, intermediate, weight='length')
                dist2 = nx.shortest_path_length(self.graph, intermediate, self.start_node, weight='length')
                total_distance = dist1 + dist2
                
                if min_return_distance <= total_distance <= max_return_distance:
                    # Create multi-segment return path
                    seg1 = self._create_segment(current_node, intermediate)
                    seg2 = self._create_segment(intermediate, self.start_node)
                    
                    if seg1 and seg2 and seg1.is_valid and seg2.is_valid:
                        # Combine into single "return" segment for simplicity
                        # (In practice, you might want to return both segments)
                        return seg2  # Just return the final segment for now
                        
            except nx.NetworkXNoPath:
                continue
        
        return None
    
    def _validate_distance_constraint(self, chromosome: RouteChromosome, 
                                    min_distance: float, max_distance: float) -> bool:
        """Validate that chromosome meets distance constraints"""
        
        if not chromosome or not chromosome.segments:
            return False
        
        total_distance = sum(segment.length for segment in chromosome.segments)
        return min_distance <= total_distance <= max_distance
    
    # Reuse helper methods from original class
    def _get_reachable_neighbors(self, node: int, max_distance: float = 1000) -> List[int]:
        """Get neighbors within max_distance using caching"""
        cache_key = (node, max_distance)
        
        if cache_key in self._neighbor_cache:
            return self._neighbor_cache[cache_key]
        
        neighbors = []
        try:
            for neighbor in self.graph.neighbors(node):
                if self.graph.has_edge(node, neighbor):
                    edge_length = self.graph[node][neighbor].get('length', 0.0)
                    if edge_length <= max_distance:
                        neighbors.append(neighbor)
        except Exception:
            neighbors = []
        
        self._neighbor_cache[cache_key] = neighbors
        return neighbors
    
    def _create_segment(self, start_node: int, end_node: int) -> Optional[RouteSegment]:
        """Create a segment between two nodes"""
        try:
            path = nx.shortest_path(self.graph, start_node, end_node, weight='length')
            segment = RouteSegment(start_node, end_node, path)
            segment.calculate_properties(self.graph)
            return segment
        except nx.NetworkXNoPath:
            return None
        except Exception as e:
            return None


def test_improved_initializer():
    """Test the improved population initializer"""
    
    print("🧪 Testing Improved Population Initializer")
    print("=" * 60)
    
    # Load network
    from route_services import NetworkManager
    nm = NetworkManager()
    graph = nm.load_network(radius_km=2.5)
    
    # Apply filtering
    filtered_graph = graph.copy()
    edges_to_remove = []
    for u, v, data in filtered_graph.edges(data=True):
        if data.get('highway', '') == 'footway':
            edges_to_remove.append((u, v))
    filtered_graph.remove_edges_from(edges_to_remove)
    isolated_nodes = [node for node in filtered_graph.nodes() if filtered_graph.degree(node) == 0]
    filtered_graph.remove_nodes_from(isolated_nodes)
    
    # Test parameters
    start_node = 1529188403
    target_distance = 5.0
    population_size = 50
    
    print(f"Network: {len(filtered_graph.nodes)} nodes, {len(filtered_graph.edges)} edges")
    print(f"Target: {target_distance}km, Population: {population_size}")
    
    # Create improved initializer
    initializer = ImprovedPopulationInitializer(filtered_graph, start_node)
    population = initializer.create_population(population_size, target_distance)
    
    # Analyze results
    print(f"\n📊 Results Analysis:")
    print(f"Population created: {len(population)}/{population_size}")
    
    if population:
        distances = []
        methods = defaultdict(int)
        
        for chromosome in population:
            stats = chromosome.get_route_stats()
            distance_km = stats.get('total_distance_km', 0)
            distances.append(distance_km)
            methods[chromosome.creation_method] += 1
        
        distances = np.array(distances)
        
        print(f"\nDistance Statistics:")
        print(f"  Mean: {distances.mean():.2f}km ({distances.mean()/target_distance:.1%} of target)")
        print(f"  Std:  {distances.std():.2f}km")
        print(f"  Min:  {distances.min():.2f}km ({distances.min()/target_distance:.1%} of target)")
        print(f"  Max:  {distances.max():.2f}km ({distances.max()/target_distance:.1%} of target)")
        
        # Check constraint compliance
        min_allowed = target_distance * 0.85
        max_allowed = target_distance * 1.15
        compliant = ((distances >= min_allowed) & (distances <= max_allowed)).sum()
        
        print(f"\nConstraint Compliance:")
        print(f"  Target range: {min_allowed:.2f}km - {max_allowed:.2f}km")
        print(f"  Compliant: {compliant}/{len(population)} ({compliant/len(population):.1%})")
        
        print(f"\nMethod Distribution:")
        for method, count in methods.items():
            print(f"  {method.replace('_', ' ').title()}: {count}")
    
    print(f"\n✅ Improved initializer test completed")


if __name__ == "__main__":
    test_improved_initializer()