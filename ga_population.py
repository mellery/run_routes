#!/usr/bin/env python3
"""
GA Population Initialization
Implements multiple strategies for creating diverse initial populations
"""

import random
import math
from typing import List, Optional, Tuple, Set
import networkx as nx
import numpy as np

from ga_chromosome import RouteChromosome, RouteSegment


class PopulationInitializer:
    """Creates initial populations using multiple strategies"""
    
    def __init__(self, graph: nx.Graph, start_node: int):
        """Initialize population creator
        
        Args:
            graph: NetworkX graph with elevation data
            start_node: Starting node for all routes
        """
        self.graph = graph
        self.start_node = start_node
        
        # Cache for performance
        self._neighbor_cache = {}
        self._distance_cache = {}
        
        print(f"PopulationInitializer created for start node {start_node}")
    
    def create_population(self, size: int, target_distance_km: float) -> List[RouteChromosome]:
        """Create initial population using multiple strategies
        
        Args:
            size: Population size
            target_distance_km: Target route distance
            
        Returns:
            List of RouteChromosome objects
        """
        population = []
        target_distance_m = target_distance_km * 1000
        
        print(f"Creating population of {size} chromosomes, target distance: {target_distance_km}km")
        
        # Strategy distribution
        random_walk_count = int(size * 0.4)      # 40%
        directional_count = int(size * 0.3)      # 30% 
        elevation_count = int(size * 0.2)        # 20%
        remaining_count = size - random_walk_count - directional_count - elevation_count  # 10%
        
        # Strategy 1: Random Walk Routes (40%)
        print(f"Creating {random_walk_count} random walk routes...")
        for i in range(random_walk_count):
            chromosome = self._create_random_walk_route(target_distance_m)
            if chromosome:
                chromosome.creation_method = "random_walk"
                chromosome.generation = 0
                population.append(chromosome)
        
        # Strategy 2: Directional Bias Routes (30%)
        print(f"Creating {directional_count} directional routes...")
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        for i in range(directional_count):
            preferred_direction = directions[i % len(directions)]
            chromosome = self._create_directional_route(target_distance_m, preferred_direction)
            if chromosome:
                chromosome.creation_method = f"directional_{preferred_direction}"
                chromosome.generation = 0
                population.append(chromosome)
        
        # Strategy 3: Elevation-Seeking Routes (20%)
        print(f"Creating {elevation_count} elevation-focused routes...")
        for i in range(elevation_count):
            chromosome = self._create_elevation_focused_route(target_distance_m)
            if chromosome:
                chromosome.creation_method = "elevation_focused"
                chromosome.generation = 0
                population.append(chromosome)
        
        # Strategy 4: Fill remaining with best strategy
        print(f"Creating {remaining_count} additional routes...")
        for i in range(remaining_count):
            # Mix of strategies for remaining
            if i % 2 == 0:
                chromosome = self._create_random_walk_route(target_distance_m)
                method = "random_walk_extra"
            else:
                chromosome = self._create_elevation_focused_route(target_distance_m)
                method = "elevation_extra"
            
            if chromosome:
                chromosome.creation_method = method
                chromosome.generation = 0
                population.append(chromosome)
        
        # Validate population
        valid_population = []
        for chromosome in population:
            if chromosome and chromosome.validate_connectivity():
                valid_population.append(chromosome)
        
        print(f"Population created: {len(valid_population)}/{size} valid chromosomes")
        
        # If we don't have enough valid routes, create simple fallback routes
        while len(valid_population) < size:
            fallback = self._create_simple_fallback_route(target_distance_m)
            if fallback:
                fallback.creation_method = "fallback"
                fallback.generation = 0
                valid_population.append(fallback)
            else:
                break
        
        return valid_population[:size]  # Return exactly requested size
    
    def _create_random_walk_route(self, target_distance_m: float) -> Optional[RouteChromosome]:
        """Create route using random walk strategy with segment usage validation"""
        current_node = self.start_node
        segments = []
        total_distance = 0.0
        visited_nodes = {current_node}
        max_segments = max(20, int(target_distance_m / 250))  # ~250m per segment on average
        segment_usage = {}  # Track segment usage during construction
        
        while total_distance < target_distance_m and len(segments) < max_segments:
            # Get reachable neighbors
            neighbors = self._get_reachable_neighbors(current_node, max_distance=800)
            
            # Filter out recently visited nodes (with some probability)
            available_neighbors = []
            for neighbor in neighbors:
                if neighbor not in visited_nodes or random.random() < 0.3:  # 30% chance to revisit
                    available_neighbors.append(neighbor)
            
            if not available_neighbors:
                available_neighbors = neighbors  # Use any neighbor if no unvisited
            
            # Filter out neighbors that would violate segment usage constraints
            valid_neighbors = []
            for neighbor in available_neighbors:
                if self._can_use_segment(current_node, neighbor, segment_usage):
                    valid_neighbors.append(neighbor)
            
            if not valid_neighbors:
                # If no valid neighbors, try fallback with any available neighbor
                valid_neighbors = available_neighbors
            
            if not valid_neighbors:
                break
            
            # Select next node with distance bias
            next_node = self._select_distance_biased_neighbor(
                valid_neighbors, current_node, target_distance_m - total_distance
            )
            
            # Create segment
            segment = self._create_segment(current_node, next_node)
            if not segment or not segment.is_valid:
                break
            
            # Update segment usage tracking
            self._update_segment_usage(current_node, next_node, segment_usage)
            
            segments.append(segment)
            total_distance += segment.length
            visited_nodes.add(next_node)
            current_node = next_node
            
            # Stop if we're getting close to target distance (more flexible threshold)
            if total_distance >= target_distance_m * 0.8:  # 80% of target distance
                break
        
        # Return to start if not already there
        if current_node != self.start_node and segments:
            # Check if we can use the return segment
            if self._can_use_segment(current_node, self.start_node, segment_usage):
                return_segment = self._create_segment(current_node, self.start_node)
                if return_segment and return_segment.is_valid:
                    segments.append(return_segment)
        
        if segments:
            chromosome = RouteChromosome(segments)
            chromosome.validate_connectivity()
            return chromosome
        
        return None
    
    def _create_directional_route(self, target_distance_m: float, 
                                 preferred_direction: str) -> Optional[RouteChromosome]:
        """Create route with directional bias and segment usage validation"""
        current_node = self.start_node
        segments = []
        total_distance = 0.0
        max_segments = max(20, int(target_distance_m / 250))  # ~250m per segment on average
        segment_usage = {}  # Track segment usage during construction
        
        # Direction mappings (approximate)
        direction_bearings = {
            'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
            'S': 180, 'SW': 225, 'W': 270, 'NW': 315
        }
        preferred_bearing = direction_bearings.get(preferred_direction, 0)
        
        while total_distance < target_distance_m and len(segments) < max_segments:
            neighbors = self._get_reachable_neighbors(current_node, max_distance=600)
            
            if not neighbors:
                break
            
            # Filter neighbors that would violate segment usage constraints
            valid_neighbors = [n for n in neighbors if self._can_use_segment(current_node, n, segment_usage)]
            
            if not valid_neighbors:
                valid_neighbors = neighbors  # Fallback to any neighbor
            
            # Select neighbor closest to preferred direction
            best_neighbor = None
            best_score = float('inf')
            
            for neighbor in valid_neighbors:
                bearing = self._calculate_bearing(current_node, neighbor)
                if bearing is not None:
                    # Calculate angular difference
                    angle_diff = abs(bearing - preferred_bearing)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                    
                    # Add some randomness
                    score = angle_diff + random.uniform(0, 30)
                    
                    if score < best_score:
                        best_score = score
                        best_neighbor = neighbor
            
            if not best_neighbor:
                best_neighbor = random.choice(valid_neighbors)
            
            # Create segment
            segment = self._create_segment(current_node, best_neighbor)
            if not segment or not segment.is_valid:
                break
            
            # Update segment usage tracking
            self._update_segment_usage(current_node, best_neighbor, segment_usage)
            
            segments.append(segment)
            total_distance += segment.length
            current_node = best_neighbor
            
            # After going in preferred direction for a while, allow turns
            if len(segments) > 3 and random.random() < 0.4:
                # Allow 90-degree turns
                turn_directions = [preferred_direction]
                if preferred_direction in ['N', 'S']:
                    turn_directions.extend(['E', 'W'])
                elif preferred_direction in ['E', 'W']:
                    turn_directions.extend(['N', 'S'])
                else:  # Diagonal directions
                    turn_directions.extend(['N', 'E', 'S', 'W'])
                
                preferred_direction = random.choice(turn_directions)
                preferred_bearing = direction_bearings.get(preferred_direction, 0)
        
        # Return to start
        if current_node != self.start_node and segments:
            if self._can_use_segment(current_node, self.start_node, segment_usage):
                return_segment = self._create_segment(current_node, self.start_node)
                if return_segment and return_segment.is_valid:
                    segments.append(return_segment)
        
        if segments:
            chromosome = RouteChromosome(segments)
            chromosome.validate_connectivity()
            return chromosome
        
        return None
    
    def _create_elevation_focused_route(self, target_distance_m: float) -> Optional[RouteChromosome]:
        """Create route focused on elevation gain"""
        current_node = self.start_node
        segments = []
        total_distance = 0.0
        max_segments = max(20, int(target_distance_m / 250))  # ~250m per segment on average
        
        while total_distance < target_distance_m and len(segments) < max_segments:
            neighbors = self._get_reachable_neighbors(current_node, max_distance=700)
            
            if not neighbors:
                break
            
            # Find neighbors with elevation gain potential
            elevation_neighbors = []
            current_elevation = self.graph.nodes[current_node].get('elevation', 0.0)
            
            for neighbor in neighbors:
                neighbor_elevation = self.graph.nodes[neighbor].get('elevation', 0.0)
                elevation_gain = neighbor_elevation - current_elevation
                
                if elevation_gain > 0:  # Prefer uphill
                    elevation_neighbors.append((neighbor, elevation_gain))
            
            # Select neighbor
            if elevation_neighbors:
                # Sort by elevation gain and select with some randomness
                elevation_neighbors.sort(key=lambda x: x[1], reverse=True)
                # Take top 3 and select randomly
                top_neighbors = elevation_neighbors[:min(3, len(elevation_neighbors))]
                next_node = random.choice(top_neighbors)[0]
            else:
                # No elevation gain available, select randomly
                next_node = random.choice(neighbors)
            
            # Create segment
            segment = self._create_segment(current_node, next_node)
            if not segment or not segment.is_valid:
                break
            
            segments.append(segment)
            total_distance += segment.length
            current_node = next_node
            
            # Occasionally allow descent to find more climbs
            if len(segments) > 2 and random.random() < 0.3:
                # Look for a descent that might lead to another climb
                descent_neighbors = []
                for neighbor in self._get_reachable_neighbors(current_node):
                    neighbor_elevation = self.graph.nodes[neighbor].get('elevation', 0.0)
                    current_elevation = self.graph.nodes[current_node].get('elevation', 0.0)
                    if neighbor_elevation < current_elevation:
                        descent_neighbors.append(neighbor)
                
                if descent_neighbors:
                    next_node = random.choice(descent_neighbors)
                    segment = self._create_segment(current_node, next_node)
                    if segment and segment.is_valid:
                        segments.append(segment)
                        total_distance += segment.length
                        current_node = next_node
        
        # Return to start
        if current_node != self.start_node and segments:
            return_segment = self._create_segment(current_node, self.start_node)
            if return_segment and return_segment.is_valid:
                segments.append(return_segment)
        
        if segments:
            chromosome = RouteChromosome(segments)
            chromosome.validate_connectivity()
            return chromosome
        
        return None
    
    def _create_simple_fallback_route(self, target_distance_m: float) -> Optional[RouteChromosome]:
        """Create simple fallback route when other strategies fail"""
        # Create a simple out-and-back route
        neighbors = self._get_reachable_neighbors(self.start_node, max_distance=1000)
        
        if not neighbors:
            return None
        
        # Select a neighbor roughly half the target distance away
        target_segment_distance = target_distance_m / 4  # Out, around, back, return
        
        best_neighbor = None
        best_distance_diff = float('inf')
        
        for neighbor in neighbors:
            segment = self._create_segment(self.start_node, neighbor)
            if segment and segment.is_valid:
                distance_diff = abs(segment.length - target_segment_distance)
                if distance_diff < best_distance_diff:
                    best_distance_diff = distance_diff
                    best_neighbor = neighbor
        
        if not best_neighbor:
            best_neighbor = random.choice(neighbors)
        
        # Create simple route: start -> neighbor -> start
        segment1 = self._create_segment(self.start_node, best_neighbor)
        segment2 = self._create_segment(best_neighbor, self.start_node)
        
        if segment1 and segment2 and segment1.is_valid and segment2.is_valid:
            chromosome = RouteChromosome([segment1, segment2])
            chromosome.validate_connectivity()
            return chromosome
        
        return None
    
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
            # Node doesn't exist in graph
            neighbors = []
        
        self._neighbor_cache[cache_key] = neighbors
        return neighbors
    
    def _create_segment(self, start_node: int, end_node: int) -> Optional[RouteSegment]:
        """Create a segment between two nodes"""
        try:
            # Find shortest path
            path = nx.shortest_path(self.graph, start_node, end_node, weight='length')
            
            # Create segment
            segment = RouteSegment(start_node, end_node, path)
            segment.calculate_properties(self.graph)
            
            return segment
            
        except nx.NetworkXNoPath:
            return None
        except Exception as e:
            print(f"Error creating segment {start_node} -> {end_node}: {e}")
            return None
    
    def _select_distance_biased_neighbor(self, neighbors: List[int], current_node: int,
                                       remaining_distance: float) -> int:
        """Select neighbor with bias toward remaining distance"""
        if not neighbors:
            return current_node
        
        # Calculate scores for each neighbor
        scored_neighbors = []
        
        for neighbor in neighbors:
            # Estimate distance to neighbor
            try:
                path_length = nx.shortest_path_length(
                    self.graph, current_node, neighbor, weight='length'
                )
                
                # Score based on how well it fits remaining distance
                distance_factor = 1.0
                if remaining_distance > 500:  # Still need significant distance
                    # Prefer longer segments
                    distance_factor = min(path_length / 400, 2.0)
                else:  # Close to target
                    # Prefer shorter segments
                    distance_factor = max(1.0 - path_length / 400, 0.1)
                
                # Add randomness
                score = distance_factor * random.uniform(0.5, 1.5)
                scored_neighbors.append((neighbor, score))
                
            except nx.NetworkXNoPath:
                continue
        
        if not scored_neighbors:
            return random.choice(neighbors)
        
        # Select based on scores (higher is better)
        scored_neighbors.sort(key=lambda x: x[1], reverse=True)
        
        # Top 3 with weighted random selection
        top_candidates = scored_neighbors[:min(3, len(scored_neighbors))]
        weights = [score for _, score in top_candidates]
        total_weight = sum(weights)
        
        if total_weight > 0:
            r = random.uniform(0, total_weight)
            cumulative = 0
            for neighbor, weight in top_candidates:
                cumulative += weight
                if r <= cumulative:
                    return neighbor
        
        return top_candidates[0][0]
    
    def _calculate_bearing(self, node1: int, node2: int) -> Optional[float]:
        """Calculate bearing from node1 to node2 in degrees"""
        try:
            lat1 = math.radians(self.graph.nodes[node1]['y'])
            lon1 = math.radians(self.graph.nodes[node1]['x'])
            lat2 = math.radians(self.graph.nodes[node2]['y'])
            lon2 = math.radians(self.graph.nodes[node2]['x'])
            
            dlon = lon2 - lon1
            y = math.sin(dlon) * math.cos(lat2)
            x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
            
            bearing = math.atan2(y, x)
            bearing = math.degrees(bearing)
            return (bearing + 360) % 360
            
        except Exception:
            return None
    
    def _can_use_segment(self, start_node: int, end_node: int, segment_usage: dict) -> bool:
        """Check if a segment can be used without violating usage constraints"""
        edge_key = tuple(sorted([start_node, end_node]))
        direction = "forward" if start_node < end_node else "backward"
        
        if edge_key not in segment_usage:
            return True
        
        return segment_usage[edge_key].get(direction, 0) < 1
    
    def _update_segment_usage(self, start_node: int, end_node: int, segment_usage: dict):
        """Update segment usage tracking"""
        edge_key = tuple(sorted([start_node, end_node]))
        direction = "forward" if start_node < end_node else "backward"
        
        if edge_key not in segment_usage:
            segment_usage[edge_key] = {"forward": 0, "backward": 0}
        
        segment_usage[edge_key][direction] += 1


def test_population_initializer():
    """Test function for population initializer"""
    print("Testing PopulationInitializer...")
    
    try:
        # Create test graph
        test_graph = nx.Graph()
        
        # Add nodes in a small grid
        nodes = [
            (1, -80.4094, 37.1299, 100),
            (2, -80.4090, 37.1299, 105),
            (3, -80.4086, 37.1299, 110),
            (4, -80.4094, 37.1303, 95),
            (5, -80.4090, 37.1303, 100),
            (6, -80.4086, 37.1303, 115)
        ]
        
        for node_id, x, y, elevation in nodes:
            test_graph.add_node(node_id, x=x, y=y, elevation=elevation)
        
        # Add edges
        edges = [(1, 2, 400), (2, 3, 400), (1, 4, 400), (4, 5, 400), 
                (5, 6, 400), (2, 5, 400), (3, 6, 400)]
        
        for node1, node2, length in edges:
            test_graph.add_edge(node1, node2, length=length)
        
        # Test population creation
        initializer = PopulationInitializer(test_graph, start_node=1)
        population = initializer.create_population(size=10, target_distance_km=2.0)
        
        print(f"✅ Created population of {len(population)} chromosomes")
        
        # Test individual chromosome
        if population:
            chromosome = population[0]
            print(f"✅ Sample chromosome: {chromosome}")
            print(f"   Stats: {chromosome.get_route_stats()}")
        
        print("✅ PopulationInitializer test completed")
        
    except Exception as e:
        print(f"❌ PopulationInitializer test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_population_initializer()