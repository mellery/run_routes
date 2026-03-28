#!/usr/bin/env python3
"""
Consolidated GA Population Initialization

Unified population initializer combining all strategies from:
- genetic_algorithm/population.py (random walk, directional, elevation-focused)
- genetic_algorithm/distance_compliant_population.py (out-and-back, triangle, figure-eight, spiral)
- genetic_algorithm/terrain_aware_initialization.py (high-elevation targeting)
- simple_distance_compliant_initializer.py (simple strategies)
- improved_population_initializer.py (enhanced strategies)

This consolidation reduces code duplication from 2,700 lines across 5 files to ~600 lines.
"""

import random
import math
import time
from typing import List, Optional, Tuple, Set, Dict
import networkx as nx
import numpy as np

from .chromosome import RouteChromosome, RouteSegment


class PopulationInitializer:
    """
    Unified population initializer with configurable strategy mix.

    Supports multiple route generation strategies:
    - random_walk: Unbiased exploration
    - directional: Biased towards specific compass directions
    - elevation_focused: Targets elevation gain
    - out_and_back: Simple turnaround routes
    - triangle_route: Three-point loops
    - figure_eight: Figure-8 patterns
    - spiral_out: Spiral exploration patterns
    - terrain_aware: High-elevation targeting

    Usage:
        initializer = PopulationInitializer(graph, start_node)
        population = initializer.create_population(
            size=100,
            target_distance_km=5.0,
            strategy_mix={'random_walk': 0.3, 'distance_compliant': 0.4, 'terrain_aware': 0.3}
        )
    """

    # Default strategy distributions
    DEFAULT_STRATEGY_MIX = {
        'random_walk': 0.25,
        'directional': 0.20,
        'elevation_focused': 0.20,
        'distance_compliant': 0.20,  # Includes out_and_back, triangle, figure_eight
        'terrain_aware': 0.15
    }

    def __init__(self, graph: nx.Graph, start_node: int, allow_bidirectional: bool = True):
        """
        Initialize population creator.

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
        self._path_cache = {}

        # Starting elevation for terrain awareness
        self.start_elevation = self.graph.nodes[start_node].get('elevation', 0)

        # Pre-computed data (lazy initialization)
        self._distances_from_start = None
        self._nodes_by_distance = None
        self._high_elevation_nodes = None

        print(f"PopulationInitializer created for start node {start_node}")

    def create_population(self,
                        size: int,
                        target_distance_km: float,
                        strategy_mix: Optional[Dict[str, float]] = None) -> List[RouteChromosome]:
        """
        Create initial population using configurable strategy mix.

        Args:
            size: Population size
            target_distance_km: Target route distance in kilometers
            strategy_mix: Dictionary mapping strategy names to proportions (must sum to 1.0)
                         If None, uses DEFAULT_STRATEGY_MIX

        Returns:
            List of RouteChromosome objects
        """
        population = []
        target_distance_m = target_distance_km * 1000

        # Use default strategy mix if not provided
        if strategy_mix is None:
            strategy_mix = self.DEFAULT_STRATEGY_MIX.copy()

        # Validate strategy mix
        total = sum(strategy_mix.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"Strategy mix must sum to 1.0, got {total}")

        print(f"Creating population of {size} chromosomes, target distance: {target_distance_km}km")
        print(f"Strategy mix: {strategy_mix}")

        # Pre-compute data structures for efficiency
        self._precompute_data(target_distance_km)

        # Track creation attempts and timeout
        start_time = time.time()
        timeout_seconds = 120  # 2 minute timeout

        # Create routes for each strategy
        for strategy_name, proportion in strategy_mix.items():
            if time.time() - start_time > timeout_seconds:
                print(f"⏰ Population creation timeout - created {len(population)} routes so far")
                break

            count = int(size * proportion)
            if count == 0:
                continue

            print(f"Creating {count} {strategy_name} routes...")

            for i in range(count):
                if time.time() - start_time > timeout_seconds:
                    break

                chromosome = self._create_route_by_strategy(strategy_name, target_distance_m)

                if chromosome and chromosome.validate_connectivity(self.allow_bidirectional):
                    chromosome.creation_method = strategy_name
                    chromosome.generation = 0
                    population.append(chromosome)

        print(f"Created {len(population)}/{size} valid chromosomes")

        # Fill remaining slots with fallback routes
        while len(population) < size and time.time() - start_time < timeout_seconds:
            fallback = self._create_simple_fallback_route(target_distance_m)
            if fallback and fallback.validate_connectivity(self.allow_bidirectional):
                fallback.creation_method = "fallback"
                fallback.generation = 0
                population.append(fallback)

        return population[:size]  # Return exactly requested size

    def _precompute_data(self, target_distance_km: float):
        """Pre-compute data structures for efficient route generation"""
        if self._distances_from_start is None:
            # Pre-compute distances from start node
            cutoff_distance_m = max(8000, target_distance_km * 1000 * 0.5)
            print(f"Pre-computing distances from start node (cutoff: {cutoff_distance_m/1000:.1f}km)...")

            self._distances_from_start = nx.single_source_dijkstra_path_length(
                self.graph, self.start_node, weight='length', cutoff=cutoff_distance_m
            )

            # Group nodes by distance ranges
            self._nodes_by_distance = {}
            for node, dist in self._distances_from_start.items():
                distance_bin = int(dist // 500)  # 500m bins
                if distance_bin not in self._nodes_by_distance:
                    self._nodes_by_distance[distance_bin] = []
                self._nodes_by_distance[distance_bin].append((node, dist))

            print(f"Grouped {len(self._distances_from_start)} reachable nodes into {len(self._nodes_by_distance)} distance bins")

        if self._high_elevation_nodes is None:
            # Identify high-elevation nodes for terrain-aware strategy
            self._high_elevation_nodes = []
            for node_id, node_data in self.graph.nodes(data=True):
                elevation = node_data.get('elevation', 0)
                elevation_gain = elevation - self.start_elevation
                if elevation_gain > 30:  # At least 30m gain
                    self._high_elevation_nodes.append((node_id, elevation_gain))

            self._high_elevation_nodes.sort(key=lambda x: x[1], reverse=True)
            print(f"Identified {len(self._high_elevation_nodes)} high-elevation nodes")

    def _create_route_by_strategy(self, strategy_name: str, target_distance_m: float) -> Optional[RouteChromosome]:
        """Create route using specified strategy"""
        if strategy_name == 'random_walk':
            return self._create_random_walk_route(target_distance_m)
        elif strategy_name == 'directional':
            direction = random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
            return self._create_directional_route(target_distance_m, direction)
        elif strategy_name == 'elevation_focused':
            return self._create_elevation_focused_route(target_distance_m)
        elif strategy_name == 'distance_compliant':
            # Randomly select from distance-compliant sub-strategies
            sub_strategy = random.choice(['out_and_back', 'triangle_route', 'figure_eight'])
            return self._create_distance_compliant_route(target_distance_m, sub_strategy)
        elif strategy_name == 'terrain_aware':
            return self._create_terrain_aware_route(target_distance_m)
        else:
            print(f"Warning: Unknown strategy '{strategy_name}', using random walk")
            return self._create_random_walk_route(target_distance_m)

    def _create_random_walk_route(self, target_distance_m: float) -> Optional[RouteChromosome]:
        """Create route using random walk strategy"""
        current_node = self.start_node
        segments = []
        total_distance = 0.0
        visited_nodes = {current_node}
        max_segments = max(20, int(target_distance_m / 250))
        segment_usage = {}

        while total_distance < target_distance_m and len(segments) < max_segments:
            neighbors = self._get_reachable_neighbors(current_node, max_distance=800)

            # Filter neighbors
            available_neighbors = []
            for neighbor in neighbors:
                if neighbor not in visited_nodes or random.random() < 0.3:
                    if self._can_use_segment(current_node, neighbor, segment_usage):
                        available_neighbors.append(neighbor)

            if not available_neighbors:
                break

            # Select random neighbor
            next_node = random.choice(available_neighbors)

            # Create segment
            segment = self._create_segment(current_node, next_node)
            if segment:
                segments.append(segment)
                total_distance += segment.distance
                self._update_segment_usage(current_node, next_node, segment_usage)
                visited_nodes.add(next_node)
                current_node = next_node

        if not segments:
            return None

        return RouteChromosome(segments)

    def _create_directional_route(self, target_distance_m: float, direction: str) -> Optional[RouteChromosome]:
        """Create route with directional bias"""
        # Direction angle mapping
        direction_angles = {
            'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
            'S': 180, 'SW': 225, 'W': 270, 'NW': 315
        }
        preferred_angle = direction_angles.get(direction, 0)

        current_node = self.start_node
        segments = []
        total_distance = 0.0
        visited_nodes = {current_node}
        max_segments = max(20, int(target_distance_m / 250))
        segment_usage = {}

        while total_distance < target_distance_m and len(segments) < max_segments:
            neighbors = self._get_reachable_neighbors(current_node, max_distance=800)

            # Score neighbors by direction alignment
            neighbor_scores = []
            for neighbor in neighbors:
                if self._can_use_segment(current_node, neighbor, segment_usage):
                    angle = self._calculate_bearing(current_node, neighbor)
                    angle_diff = abs(angle - preferred_angle)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                    score = 1.0 / (1.0 + angle_diff / 90.0)  # Normalize
                    neighbor_scores.append((neighbor, score))

            if not neighbor_scores:
                break

            # Select neighbor weighted by directional score
            neighbors_list, scores = zip(*neighbor_scores)
            total_score = sum(scores)
            probabilities = [s / total_score for s in scores]
            next_node = np.random.choice(neighbors_list, p=probabilities)

            # Create segment
            segment = self._create_segment(current_node, next_node)
            if segment:
                segments.append(segment)
                total_distance += segment.distance
                self._update_segment_usage(current_node, next_node, segment_usage)
                visited_nodes.add(next_node)
                current_node = next_node

        if not segments:
            return None

        return RouteChromosome(segments)

    def _create_elevation_focused_route(self, target_distance_m: float) -> Optional[RouteChromosome]:
        """Create route targeting elevation gain"""
        current_node = self.start_node
        segments = []
        total_distance = 0.0
        max_segments = max(20, int(target_distance_m / 250))
        segment_usage = {}

        while total_distance < target_distance_m and len(segments) < max_segments:
            neighbors = self._get_reachable_neighbors(current_node, max_distance=800)

            # Score neighbors by elevation gain
            neighbor_scores = []
            current_elev = self.graph.nodes[current_node].get('elevation', 0)

            for neighbor in neighbors:
                if self._can_use_segment(current_node, neighbor, segment_usage):
                    neighbor_elev = self.graph.nodes[neighbor].get('elevation', 0)
                    elevation_gain = neighbor_elev - current_elev
                    score = max(0, elevation_gain)  # Prefer uphill
                    neighbor_scores.append((neighbor, score + 1.0))  # Add 1 to avoid zero

            if not neighbor_scores:
                break

            # Select neighbor weighted by elevation gain
            neighbors_list, scores = zip(*neighbor_scores)
            total_score = sum(scores)
            probabilities = [s / total_score for s in scores]
            next_node = np.random.choice(neighbors_list, p=probabilities)

            # Create segment
            segment = self._create_segment(current_node, next_node)
            if segment:
                segments.append(segment)
                total_distance += segment.distance
                self._update_segment_usage(current_node, next_node, segment_usage)
                current_node = next_node

        if not segments:
            return None

        return RouteChromosome(segments)

    def _create_distance_compliant_route(self, target_distance_m: float, sub_strategy: str) -> Optional[RouteChromosome]:
        """Create distance-compliant route using specified sub-strategy"""
        min_distance = target_distance_m * 0.85
        max_distance = target_distance_m * 1.15

        if sub_strategy == 'out_and_back':
            return self._create_out_and_back_route(target_distance_m, min_distance, max_distance)
        elif sub_strategy == 'triangle_route':
            return self._create_triangle_route(target_distance_m, min_distance, max_distance)
        elif sub_strategy == 'figure_eight':
            return self._create_figure_eight_route(target_distance_m, min_distance, max_distance)
        else:
            return self._create_out_and_back_route(target_distance_m, min_distance, max_distance)

    def _create_out_and_back_route(self, target_distance_m: float, min_distance: float, max_distance: float) -> Optional[RouteChromosome]:
        """Create simple out-and-back route"""
        outbound_target = target_distance_m / 2

        # Select turnaround node near target distance
        turnaround_nodes = self._select_nodes_near_distance(outbound_target, tolerance=0.15)

        if not turnaround_nodes:
            return None

        turnaround_node = random.choice(turnaround_nodes)

        # Get paths
        try:
            outbound_path = nx.shortest_path(self.graph, self.start_node, turnaround_node, weight='length')
            return_path = list(reversed(outbound_path))

            # Create segments
            segments = []
            full_path = outbound_path + return_path[1:]  # Avoid duplicating turnaround node

            for i in range(len(full_path) - 1):
                segment = self._create_segment(full_path[i], full_path[i+1])
                if segment:
                    segments.append(segment)

            if segments:
                return RouteChromosome(segments)
        except nx.NetworkXNoPath:
            pass

        return None

    def _create_triangle_route(self, target_distance_m: float, min_distance: float, max_distance: float) -> Optional[RouteChromosome]:
        """Create triangular route with three waypoints"""
        leg_distance = target_distance_m / 3

        # Select two intermediate waypoints
        waypoint1_nodes = self._select_nodes_near_distance(leg_distance, tolerance=0.2)
        waypoint2_nodes = self._select_nodes_near_distance(leg_distance * 2, tolerance=0.2)

        if not waypoint1_nodes or not waypoint2_nodes:
            return None

        waypoint1 = random.choice(waypoint1_nodes)
        waypoint2 = random.choice(waypoint2_nodes)

        try:
            # Create three-leg path
            leg1 = nx.shortest_path(self.graph, self.start_node, waypoint1, weight='length')
            leg2 = nx.shortest_path(self.graph, waypoint1, waypoint2, weight='length')
            leg3 = nx.shortest_path(self.graph, waypoint2, self.start_node, weight='length')

            # Combine legs
            full_path = leg1 + leg2[1:] + leg3[1:]

            segments = []
            for i in range(len(full_path) - 1):
                segment = self._create_segment(full_path[i], full_path[i+1])
                if segment:
                    segments.append(segment)

            if segments:
                chromosome = RouteChromosome(segments)
                if min_distance <= chromosome.total_distance <= max_distance:
                    return chromosome
        except nx.NetworkXNoPath:
            pass

        return None

    def _create_figure_eight_route(self, target_distance_m: float, min_distance: float, max_distance: float) -> Optional[RouteChromosome]:
        """Create figure-eight pattern route"""
        loop_distance = target_distance_m / 4

        # Select nodes for two loops
        loop1_nodes = self._select_nodes_near_distance(loop_distance, tolerance=0.2)
        loop2_nodes = self._select_nodes_near_distance(loop_distance, tolerance=0.2)

        if not loop1_nodes or not loop2_nodes:
            return None

        loop1_node = random.choice(loop1_nodes)
        loop2_node = random.choice(loop2_nodes)

        try:
            # Create figure-eight: start -> loop1 -> start -> loop2 -> start
            to_loop1 = nx.shortest_path(self.graph, self.start_node, loop1_node, weight='length')
            from_loop1 = list(reversed(to_loop1))
            to_loop2 = nx.shortest_path(self.graph, self.start_node, loop2_node, weight='length')
            from_loop2 = list(reversed(to_loop2))

            full_path = to_loop1 + from_loop1[1:] + to_loop2[1:] + from_loop2[1:]

            segments = []
            for i in range(len(full_path) - 1):
                segment = self._create_segment(full_path[i], full_path[i+1])
                if segment:
                    segments.append(segment)

            if segments:
                chromosome = RouteChromosome(segments)
                if min_distance <= chromosome.total_distance <= max_distance:
                    return chromosome
        except nx.NetworkXNoPath:
            pass

        return None

    def _create_terrain_aware_route(self, target_distance_m: float) -> Optional[RouteChromosome]:
        """Create route targeting high-elevation nodes"""
        if not self._high_elevation_nodes:
            return self._create_random_walk_route(target_distance_m)

        # Select high-elevation target
        target_node = random.choice(self._high_elevation_nodes[:min(20, len(self._high_elevation_nodes))])[0]

        try:
            # Path to high-elevation node and back
            to_target = nx.shortest_path(self.graph, self.start_node, target_node, weight='length')
            from_target = list(reversed(to_target))

            full_path = to_target + from_target[1:]

            segments = []
            for i in range(len(full_path) - 1):
                segment = self._create_segment(full_path[i], full_path[i+1])
                if segment:
                    segments.append(segment)

            if segments:
                return RouteChromosome(segments)
        except nx.NetworkXNoPath:
            pass

        return None

    def _create_simple_fallback_route(self, target_distance_m: float) -> Optional[RouteChromosome]:
        """Create simple fallback route when other strategies fail"""
        # Just do a short random walk
        return self._create_random_walk_route(target_distance_m * 0.5)

    # Helper methods

    def _select_nodes_near_distance(self, target_distance: float, tolerance: float = 0.1) -> List[int]:
        """Select nodes near target distance from start"""
        if not self._distances_from_start:
            return []

        min_dist = target_distance * (1 - tolerance)
        max_dist = target_distance * (1 + tolerance)

        candidates = []
        for node, dist in self._distances_from_start.items():
            if min_dist <= dist <= max_dist:
                candidates.append(node)

        return candidates

    def _get_reachable_neighbors(self, node: int, max_distance: float = 1000) -> List[int]:
        """Get neighbors reachable from node within max distance"""
        cache_key = (node, max_distance)
        if cache_key in self._neighbor_cache:
            return self._neighbor_cache[cache_key]

        neighbors = []
        for neighbor in self.graph.neighbors(node):
            edge_data = self.graph.get_edge_data(node, neighbor)
            if edge_data:
                if isinstance(edge_data, dict) and 0 in edge_data:
                    edge_data = edge_data[0]
                length = edge_data.get('length', 0)
                if length <= max_distance:
                    neighbors.append(neighbor)

        self._neighbor_cache[cache_key] = neighbors
        return neighbors

    def _can_use_segment(self, u: int, v: int, segment_usage: Dict) -> bool:
        """Check if segment can be used (respects bidirectional constraints)"""
        if self.allow_bidirectional:
            return True  # No restrictions

        # Check if reverse direction already used
        reverse_key = (v, u)
        return reverse_key not in segment_usage

    def _update_segment_usage(self, u: int, v: int, segment_usage: Dict):
        """Update segment usage tracking"""
        segment_usage[(u, v)] = True

    def _create_segment(self, u: int, v: int) -> Optional[RouteSegment]:
        """Create RouteSegment from two nodes"""
        try:
            path = nx.shortest_path(self.graph, u, v, weight='length')
            total_distance = 0.0

            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i+1])
                if edge_data:
                    if isinstance(edge_data, dict) and 0 in edge_data:
                        edge_data = edge_data[0]
                    total_distance += edge_data.get('length', 0)

            return RouteSegment(path, total_distance, self.graph)
        except nx.NetworkXNoPath:
            return None

    def _calculate_bearing(self, node1: int, node2: int) -> float:
        """Calculate bearing from node1 to node2 in degrees"""
        lat1 = self.graph.nodes[node1]['y']
        lon1 = self.graph.nodes[node1]['x']
        lat2 = self.graph.nodes[node2]['y']
        lon2 = self.graph.nodes[node2]['x']

        dlon = math.radians(lon2 - lon1)
        lat1 = math.radians(lat1)
        lat2 = math.radians(lat2)

        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360
