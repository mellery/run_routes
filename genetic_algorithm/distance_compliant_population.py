#!/usr/bin/env python3
"""
Distance-Compliant Population Initializer for Genetic Algorithm
A production-ready approach that creates populations meeting distance constraints from the start
"""

import random
import math
import time
import networkx as nx
import numpy as np
from typing import List, Optional, Tuple

from .chromosome import RouteChromosome, RouteSegment


class DistanceCompliantPopulationInitializer:
    """Creates distance-compliant routes using a simple but effective approach"""
    
    def __init__(self, graph: nx.Graph, start_node: int, max_route_distance_km: float = 25.0):
        self.graph = graph
        self.start_node = start_node
        
        # Pre-compute distance matrix for efficiency - use adaptive cutoff
        # For spiral_out routes, we need up to 40% of the route distance as outbound distance
        cutoff_distance_m = max(8000, max_route_distance_km * 1000 * 0.5)  # At least 8km or 50% of max route
        print(f"Pre-computing distances from start node (cutoff: {cutoff_distance_m/1000:.1f}km)...")
        self.distances_from_start = nx.single_source_dijkstra_path_length(
            graph, start_node, weight='length', cutoff=cutoff_distance_m
        )
        
        # Group nodes by distance ranges for efficient selection
        self.nodes_by_distance = {}
        for node, dist in self.distances_from_start.items():
            distance_bin = int(dist // 500)  # 500m bins
            if distance_bin not in self.nodes_by_distance:
                self.nodes_by_distance[distance_bin] = []
            self.nodes_by_distance[distance_bin].append((node, dist))
        
        print(f"Grouped {len(self.distances_from_start)} reachable nodes into {len(self.nodes_by_distance)} distance bins")
    
    def create_population(self, size: int, target_distance_km: float) -> List[RouteChromosome]:
        """Create distance-compliant population using simple strategies"""
        
        population = []
        target_distance_m = target_distance_km * 1000
        min_distance = target_distance_m * 0.85
        max_distance = target_distance_m * 1.15
        
        print(f"\nCreating {size} distance-compliant routes")
        print(f"Target: {target_distance_km}km ({target_distance_m}m)")
        print(f"Range: {min_distance/1000:.2f}km - {max_distance/1000:.2f}km")
        print(f"Max reachable distance: {max(self.distances_from_start.values(), default=0)/1000:.2f}km")
        
        # Adjust strategy mix based on route distance
        if target_distance_km > 15:
            # For long routes, prioritize simple strategies
            strategies = [
                ('out_and_back', 0.6),      # 60% - Simple out and back (most reliable)
                ('triangle_route', 0.3),     # 30% - Three-point triangle
                ('figure_eight', 0.1),       # 10% - Figure-8 pattern
                # Skip spiral_out for very long routes as it's computationally expensive
            ]
            print("🏃‍♂️ Using simplified strategies for long route")
        else:
            strategies = [
                ('out_and_back', 0.4),      # 40% - Simple out and back
                ('triangle_route', 0.3),     # 30% - Three-point triangle
                ('figure_eight', 0.2),       # 20% - Figure-8 pattern
                ('spiral_out', 0.1)          # 10% - Spiral outward
            ]
        
        # Add timeout for population creation (2 minutes max)
        start_time = time.time()
        timeout_seconds = 120
        
        for strategy_name, proportion in strategies:
            if time.time() - start_time > timeout_seconds:
                print(f"⏰ Population creation timeout - created {len(population)} routes so far")
                break
                
            count = int(size * proportion)
            print(f"\nCreating {count} {strategy_name} routes...")
            
            for i in range(count):
                if time.time() - start_time > timeout_seconds:
                    print(f"⏰ Timeout reached during {strategy_name} creation")
                    break
                    
                route = self._create_route_by_strategy(
                    strategy_name, target_distance_m, min_distance, max_distance
                )
                
                if route:
                    route.creation_method = strategy_name
                    population.append(route)
                    
                    if len(population) % 10 == 0:
                        print(f"  Created {len(population)} routes...")
                elif strategy_name == 'spiral_out':
                    # Skip remaining spiral_out attempts if they're failing
                    print(f"  Skipping remaining {strategy_name} attempts (not feasible for this distance)")
                    break
                else:
                    # Print progress for debugging
                    print(f"  Failed to create {strategy_name} route {i+1}/{count}")
        
        # Fill remaining with best strategy (prefer reliable strategies)
        while len(population) < size:
            route = self._create_route_by_strategy(
                'out_and_back', target_distance_m, min_distance, max_distance
            )
            if route:
                route.creation_method = 'out_and_back_extra'
                population.append(route)
            else:
                # If out_and_back fails, break to avoid infinite loop
                print(f"⚠️ Could only create {len(population)}/{size} routes - distance may be too large")
                break
        
        print(f"✅ Created {len(population)}/{size} distance-compliant routes")
        return population[:size]
    
    def _create_route_by_strategy(self, strategy: str, target_distance_m: float,
                                min_distance: float, max_distance: float) -> Optional[RouteChromosome]:
        """Create route using specified strategy"""
        
        if strategy == 'out_and_back':
            return self._create_out_and_back(target_distance_m, min_distance, max_distance)
        elif strategy == 'triangle_route':
            return self._create_triangle_route(target_distance_m, min_distance, max_distance)
        elif strategy == 'figure_eight':
            return self._create_figure_eight(target_distance_m, min_distance, max_distance)
        elif strategy == 'spiral_out':
            return self._create_spiral_out(target_distance_m, min_distance, max_distance)
        else:
            return self._create_out_and_back(target_distance_m, min_distance, max_distance)
    
    def _create_out_and_back(self, target_distance_m: float,
                           min_distance: float, max_distance: float) -> Optional[RouteChromosome]:
        """Create simple out-and-back route targeting specific distance"""
        
        # Target outbound distance (half of total minus some buffer for return variation)
        outbound_target = target_distance_m * 0.45  # 45% out, allows 55% for return variation
        
        # Find a node roughly at the target outbound distance
        target_nodes = self._find_nodes_at_distance(outbound_target, tolerance=0.3)
        
        if not target_nodes:
            return None
        
        # Try several target nodes
        for target_node, actual_distance in target_nodes[:5]:
            try:
                # Create outbound segment
                outbound_segment = self._create_segment(self.start_node, target_node)
                if not outbound_segment:
                    continue
                
                # Create return segment
                return_segment = self._create_segment(target_node, self.start_node)
                if not return_segment:
                    continue
                
                # Check total distance
                total_distance = outbound_segment.length + return_segment.length
                
                if min_distance <= total_distance <= max_distance:
                    segments = [outbound_segment, return_segment]
                    chromosome = RouteChromosome(segments)
                    chromosome.validate_connectivity()
                    return chromosome
                    
            except Exception:
                continue
        
        return None
    
    def _create_triangle_route(self, target_distance_m: float,
                             min_distance: float, max_distance: float) -> Optional[RouteChromosome]:
        """Create triangular route: start -> A -> B -> start"""
        
        # Target: each leg should be roughly 1/3 of total distance
        leg_target = target_distance_m / 3
        
        # Find first waypoint
        waypoint_a_candidates = self._find_nodes_at_distance(leg_target, tolerance=0.4)
        if not waypoint_a_candidates:
            return None
        
        for waypoint_a, dist_a in waypoint_a_candidates[:3]:
            try:
                # Find second waypoint that's roughly leg_target from waypoint_a
                # and creates a good triangle
                waypoint_b_candidates = self._find_nodes_at_distance_from_node(
                    waypoint_a, leg_target, tolerance=0.4
                )
                
                for waypoint_b, dist_b in waypoint_b_candidates[:3]:
                    # Check if we can get back to start from waypoint_b
                    return_distance = self.distances_from_start.get(waypoint_b, float('inf'))
                    
                    # Estimate total distance
                    estimated_total = dist_a + dist_b + return_distance
                    
                    if min_distance <= estimated_total <= max_distance:
                        # Create the three segments
                        seg1 = self._create_segment(self.start_node, waypoint_a)
                        seg2 = self._create_segment(waypoint_a, waypoint_b)
                        seg3 = self._create_segment(waypoint_b, self.start_node)
                        
                        if seg1 and seg2 and seg3:
                            total_actual = seg1.length + seg2.length + seg3.length
                            
                            if min_distance <= total_actual <= max_distance:
                                segments = [seg1, seg2, seg3]
                                chromosome = RouteChromosome(segments)
                                chromosome.validate_connectivity()
                                return chromosome
                                
            except Exception:
                continue
        
        return None
    
    def _create_figure_eight(self, target_distance_m: float,
                           min_distance: float, max_distance: float) -> Optional[RouteChromosome]:
        """Create figure-8 route: start -> A -> start -> B -> start"""
        
        # Each loop should be roughly half the total distance
        loop_target = target_distance_m / 4  # Each leg of loop
        
        # Find waypoints for two loops
        loop_a_candidates = self._find_nodes_at_distance(loop_target, tolerance=0.4)
        loop_b_candidates = self._find_nodes_at_distance(loop_target, tolerance=0.4)
        
        if not loop_a_candidates or not loop_b_candidates:
            return None
        
        for waypoint_a, _ in loop_a_candidates[:2]:
            for waypoint_b, _ in loop_b_candidates[:2]:
                if waypoint_a == waypoint_b:
                    continue
                
                try:
                    # Create figure-8: start -> A -> start -> B -> start
                    seg1 = self._create_segment(self.start_node, waypoint_a)
                    seg2 = self._create_segment(waypoint_a, self.start_node)
                    seg3 = self._create_segment(self.start_node, waypoint_b)
                    seg4 = self._create_segment(waypoint_b, self.start_node)
                    
                    if seg1 and seg2 and seg3 and seg4:
                        total_distance = seg1.length + seg2.length + seg3.length + seg4.length
                        
                        if min_distance <= total_distance <= max_distance:
                            segments = [seg1, seg2, seg3, seg4]
                            chromosome = RouteChromosome(segments)
                            chromosome.validate_connectivity()
                            return chromosome
                            
                except Exception:
                    continue
        
        return None
    
    def _create_spiral_out(self, target_distance_m: float,
                         min_distance: float, max_distance: float) -> Optional[RouteChromosome]:
        """Create spiral route with gradually increasing distance from start"""
        
        # Quick validation: if target distances are beyond our pre-computed range, skip
        max_waypoint_distance = target_distance_m * 0.35
        if max_waypoint_distance > max(self.distances_from_start.values(), default=0) * 0.9:
            # Not enough reachable nodes for this distance
            return None
        
        # Create a route that spirals outward with 4-5 waypoints
        num_waypoints = 4
        segments = []
        current_node = self.start_node
        total_distance = 0.0
        
        # Target distances for each waypoint (spiraling out)
        waypoint_distances = [
            target_distance_m * 0.15,  # 15% out
            target_distance_m * 0.25,  # 25% out
            target_distance_m * 0.35,  # 35% out
            target_distance_m * 0.20   # 20% back to start area
        ]
        
        for i, target_dist in enumerate(waypoint_distances):
            if i == len(waypoint_distances) - 1:
                # Last waypoint: choose one that allows good return to start
                candidates = self._find_nodes_with_return_distance(
                    target_dist, target_distance_m - total_distance, tolerance=0.4
                )
            else:
                candidates = self._find_nodes_at_distance(target_dist, tolerance=0.4)
            
            if not candidates:
                break
            
            next_node = candidates[0][0]  # Take the first candidate
            
            try:
                segment = self._create_segment(current_node, next_node)
                if not segment:
                    break
                
                segments.append(segment)
                total_distance += segment.length
                current_node = next_node
                
            except Exception:
                break
        
        # Add return segment
        if current_node != self.start_node and segments:
            try:
                return_segment = self._create_segment(current_node, self.start_node)
                if return_segment:
                    total_distance += return_segment.length
                    
                    if min_distance <= total_distance <= max_distance:
                        segments.append(return_segment)
                        chromosome = RouteChromosome(segments)
                        chromosome.validate_connectivity()
                        return chromosome
                        
            except Exception:
                pass
        
        return None
    
    def _find_nodes_at_distance(self, target_distance: float, tolerance: float = 0.3) -> List[Tuple[int, float]]:
        """Find nodes at approximately target distance from start"""
        
        min_dist = target_distance * (1 - tolerance)
        max_dist = target_distance * (1 + tolerance)
        
        candidates = []
        for node, distance in self.distances_from_start.items():
            if min_dist <= distance <= max_dist:
                candidates.append((node, distance))
        
        # Sort by how close to target distance
        candidates.sort(key=lambda x: abs(x[1] - target_distance))
        return candidates[:20]  # Return top 20 candidates
    
    def _find_nodes_at_distance_from_node(self, source_node: int, target_distance: float, 
                                        tolerance: float = 0.3) -> List[Tuple[int, float]]:
        """Find nodes at target distance from a specific node (optimized with smaller cutoff)"""
        
        try:
            # Use smaller cutoff for efficiency - only compute what we need
            cutoff = min(target_distance * 1.5, 3000)  # Limit to 3km max to avoid expensive computation
            distances = nx.single_source_dijkstra_path_length(
                self.graph, source_node, weight='length', cutoff=cutoff
            )
            
            min_dist = target_distance * (1 - tolerance)
            max_dist = target_distance * (1 + tolerance)
            
            candidates = []
            for node, distance in distances.items():
                if min_dist <= distance <= max_dist:
                    candidates.append((node, distance))
            
            candidates.sort(key=lambda x: abs(x[1] - target_distance))
            return candidates[:5]  # Limit to 5 candidates for speed
            
        except Exception:
            return []
    
    def _find_nodes_with_return_distance(self, outbound_target: float, return_target: float,
                                       tolerance: float = 0.3) -> List[Tuple[int, float]]:
        """Find nodes at outbound_target distance that have good return distance to start"""
        
        outbound_candidates = self._find_nodes_at_distance(outbound_target, tolerance)
        
        good_candidates = []
        for node, outbound_dist in outbound_candidates:
            return_dist = self.distances_from_start.get(node, float('inf'))
            if abs(return_dist - return_target) <= return_target * tolerance:
                good_candidates.append((node, outbound_dist))
        
        return good_candidates
    
    def _create_segment(self, start_node: int, end_node: int) -> Optional[RouteSegment]:
        """Create a segment between two nodes"""
        try:
            path = nx.shortest_path(self.graph, start_node, end_node, weight='length')
            segment = RouteSegment(start_node, end_node, path)
            segment.calculate_properties(self.graph)
            return segment
        except Exception:
            return None


