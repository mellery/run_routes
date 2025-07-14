#!/usr/bin/env python3
"""
Terrain-Aware Population Initialization
Initializes population with routes targeting high-elevation nodes relative to starting elevation
"""

import random
import math
import networkx as nx
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import statistics

from .chromosome import RouteChromosome, RouteSegment


@dataclass
class TerrainAwareConfig:
    """Configuration for terrain-aware population initialization"""
    elevation_gain_threshold: float = 30.0  # Minimum elevation gain to consider a node "high"
    max_elevation_gain_threshold: float = 100.0  # Maximum elevation gain for "very high" nodes
    high_elevation_percentage: float = 0.4  # Percentage of population to target high-elevation nodes
    very_high_elevation_percentage: float = 0.2  # Percentage targeting very high elevation
    exploration_radius_multiplier: float = 1.5  # Multiplier for exploration radius
    min_path_length: int = 2  # Minimum path length to target nodes
    max_path_length: int = 25  # Maximum path length to target nodes (more flexible)
    diversity_factor: float = 0.3  # Factor for maintaining diversity in target selection


class TerrainAwarePopulationInitializer:
    """Population initializer that targets high-elevation nodes relative to starting elevation"""
    
    def __init__(self, graph: nx.Graph, start_node: int, target_distance_km: float,
                 config: Optional[TerrainAwareConfig] = None):
        """Initialize terrain-aware population initializer
        
        Args:
            graph: NetworkX graph with elevation data
            start_node: Starting node ID
            target_distance_km: Target route distance in kilometers
            config: Configuration for terrain-aware initialization
        """
        self.graph = graph
        self.start_node = start_node
        self.target_distance_km = target_distance_km
        self.config = config or TerrainAwareConfig()
        
        # Calculate starting elevation
        self.start_elevation = self.graph.nodes[start_node].get('elevation', 0)
        
        # Analyze terrain around starting point
        self._analyze_terrain()
        
        # Pre-compute reachable nodes for efficiency
        self._precompute_reachable_nodes()
    
    def _analyze_terrain(self):
        """Analyze terrain characteristics around starting point"""
        # Calculate exploration radius based on target distance
        exploration_radius_m = self.target_distance_km * 1000 * self.config.exploration_radius_multiplier
        
        # Find all nodes within exploration radius
        self.nearby_nodes = []
        
        start_lat = self.graph.nodes[self.start_node]['y']
        start_lon = self.graph.nodes[self.start_node]['x']
        
        for node_id, node_data in self.graph.nodes(data=True):
            if node_id == self.start_node:
                continue
                
            # Calculate distance using haversine
            node_lat = node_data['y']
            node_lon = node_data['x']
            distance = self._haversine_distance(start_lat, start_lon, node_lat, node_lon)
            
            if distance <= exploration_radius_m:
                elevation = node_data.get('elevation', 0)
                elevation_gain = elevation - self.start_elevation
                
                self.nearby_nodes.append({
                    'node_id': node_id,
                    'elevation': elevation,
                    'elevation_gain': elevation_gain,
                    'distance': distance
                })
        
        # Sort by elevation gain (descending)
        self.nearby_nodes.sort(key=lambda x: x['elevation_gain'], reverse=True)
        
        # Categorize nodes by elevation gain
        self.high_elevation_nodes = []
        self.very_high_elevation_nodes = []
        self.moderate_elevation_nodes = []
        
        for node_info in self.nearby_nodes:
            elevation_gain = node_info['elevation_gain']
            
            if elevation_gain >= self.config.max_elevation_gain_threshold:
                self.very_high_elevation_nodes.append(node_info)
            elif elevation_gain >= self.config.elevation_gain_threshold:
                self.high_elevation_nodes.append(node_info)
            elif elevation_gain >= 0:  # Positive elevation gain
                self.moderate_elevation_nodes.append(node_info)
        
        # Statistics for debugging
        if self.high_elevation_nodes:
            high_gains = [n['elevation_gain'] for n in self.high_elevation_nodes]
            print(f"   🏔️  High elevation nodes: {len(self.high_elevation_nodes)} "
                  f"(gain: {min(high_gains):.0f}-{max(high_gains):.0f}m)")
        
        if self.very_high_elevation_nodes:
            very_high_gains = [n['elevation_gain'] for n in self.very_high_elevation_nodes]
            print(f"   ⛰️  Very high elevation nodes: {len(self.very_high_elevation_nodes)} "
                  f"(gain: {min(very_high_gains):.0f}-{max(very_high_gains):.0f}m)")
    
    def _precompute_reachable_nodes(self):
        """Pre-compute nodes reachable within target distance"""
        max_distance_m = self.target_distance_km * 1000 * 0.6  # 60% of target distance one-way
        
        try:
            # Get all shortest path lengths from start node
            distances = nx.single_source_dijkstra_path_length(
                self.graph, self.start_node, cutoff=max_distance_m, weight='length'
            )
            
            self.reachable_nodes = set(distances.keys())
            
            # Filter elevation nodes to only include reachable ones
            self.high_elevation_nodes = [
                n for n in self.high_elevation_nodes 
                if n['node_id'] in self.reachable_nodes
            ]
            
            self.very_high_elevation_nodes = [
                n for n in self.very_high_elevation_nodes 
                if n['node_id'] in self.reachable_nodes
            ]
            
            print(f"   🎯 Reachable high elevation nodes: {len(self.high_elevation_nodes)}")
            print(f"   🎯 Reachable very high elevation nodes: {len(self.very_high_elevation_nodes)}")
            
        except Exception as e:
            print(f"   ⚠️ Error pre-computing reachable nodes: {e}")
            self.reachable_nodes = set(self.graph.nodes())
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points"""
        R = 6371000  # Earth radius in meters
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    
    def create_population(self, population_size: int, target_distance_km: float = None) -> List[RouteChromosome]:
        """Create terrain-aware population
        
        Args:
            population_size: Number of chromosomes to create
            target_distance_km: Target distance (optional, uses initialization value if not provided)
            
        Returns:
            List of route chromosomes
        """
        # Use provided target distance or fall back to initialization value
        if target_distance_km is not None:
            self.target_distance_km = target_distance_km
        print(f"   🏔️  Creating terrain-aware population (start elev: {self.start_elevation:.0f}m)")
        
        population = []
        
        # Calculate population distribution
        very_high_count = int(population_size * self.config.very_high_elevation_percentage)
        high_count = int(population_size * self.config.high_elevation_percentage)
        moderate_count = int(population_size * 0.3)  # 30% moderate elevation
        random_count = population_size - very_high_count - high_count - moderate_count
        
        print(f"   📊 Population distribution: very_high={very_high_count}, "
              f"high={high_count}, moderate={moderate_count}, random={random_count}")
        
        # Create very high elevation routes
        if self.very_high_elevation_nodes:
            very_high_routes = self._create_elevation_targeted_routes(
                very_high_count, self.very_high_elevation_nodes, "very_high"
            )
            population.extend(very_high_routes)
        
        # Create high elevation routes
        if self.high_elevation_nodes:
            high_routes = self._create_elevation_targeted_routes(
                high_count, self.high_elevation_nodes, "high"
            )
            population.extend(high_routes)
        
        # Create moderate elevation routes
        if self.moderate_elevation_nodes:
            moderate_routes = self._create_elevation_targeted_routes(
                moderate_count, self.moderate_elevation_nodes, "moderate"
            )
            population.extend(moderate_routes)
        
        # Fill remaining with random routes
        remaining_needed = population_size - len(population)
        if remaining_needed > 0:
            random_routes = self._create_random_routes(remaining_needed)
            population.extend(random_routes)
        
        # Ensure we have exactly the right number
        population = population[:population_size]
        
        print(f"   ✅ Created {len(population)} terrain-aware routes")
        
        return population
    
    def _create_elevation_targeted_routes(self, count: int, target_nodes: List[Dict], 
                                        category: str) -> List[RouteChromosome]:
        """Create routes targeting specific elevation nodes
        
        Args:
            count: Number of routes to create
            target_nodes: List of target node information
            category: Category name for logging
            
        Returns:
            List of route chromosomes
        """
        routes = []
        
        if not target_nodes or count <= 0:
            return routes
        
        # Create routes targeting different nodes for diversity
        used_targets = set()
        
        for i in range(count):
            # Select target node with diversity consideration
            target_node = self._select_diverse_target(target_nodes, used_targets)
            
            if target_node:
                route = self._create_route_to_target(target_node, category)
                if route:
                    routes.append(route)
                    used_targets.add(target_node['node_id'])
        
        return routes
    
    def _select_diverse_target(self, target_nodes: List[Dict], 
                             used_targets: Set[int]) -> Optional[Dict]:
        """Select target node with diversity consideration
        
        Args:
            target_nodes: Available target nodes
            used_targets: Previously used target nodes
            
        Returns:
            Selected target node or None
        """
        # Filter out used targets
        available_targets = [n for n in target_nodes if n['node_id'] not in used_targets]
        
        if not available_targets:
            # If all targets used, reset and select from all
            available_targets = target_nodes
        
        # Select with weighted probability favoring higher elevation gains
        if len(available_targets) == 1:
            return available_targets[0]
        
        # Weight by elevation gain with some randomness
        weights = []
        for target in available_targets:
            # Base weight on elevation gain
            base_weight = max(1.0, target['elevation_gain'] + 50)  # Ensure positive
            
            # Add randomness factor for diversity
            randomness = random.uniform(0.7, 1.3)
            
            # Penalize distance slightly
            distance_penalty = 1.0 - (target['distance'] / 10000.0)  # Normalize to 10km
            distance_penalty = max(0.5, distance_penalty)
            
            final_weight = base_weight * randomness * distance_penalty
            weights.append(final_weight)
        
        # Weighted random selection
        total_weight = sum(weights)
        if total_weight <= 0:
            return random.choice(available_targets)
        
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return available_targets[i]
        
        return available_targets[-1]  # Fallback
    
    def _create_route_to_target(self, target_node: Dict, category: str) -> Optional[RouteChromosome]:
        """Create route to specific target node
        
        Args:
            target_node: Target node information
            category: Category for creation method
            
        Returns:
            Route chromosome or None
        """
        target_id = target_node['node_id']
        
        try:
            # Find path to target
            path_to_target = nx.shortest_path(self.graph, self.start_node, target_id, weight='length')
            
            # Check path length constraints
            if len(path_to_target) < self.config.min_path_length:
                return None
            
            if len(path_to_target) > self.config.max_path_length:
                # For very long paths, create a simplified route with fewer intermediate nodes
                # Keep start, target, and some intermediate nodes
                simplified_path = [path_to_target[0]]  # Start node
                
                # Add some intermediate nodes
                step_size = max(1, len(path_to_target) // 6)  # Divide path into ~6 segments
                for i in range(step_size, len(path_to_target) - 1, step_size):
                    simplified_path.append(path_to_target[i])
                
                simplified_path.append(path_to_target[-1])  # Target node
                path_to_target = simplified_path
                target_id = path_to_target[-1]
            
            # Create out-and-back route
            route_segments = []
            
            # Forward path
            for i in range(len(path_to_target) - 1):
                segment = RouteSegment(
                    start_node=path_to_target[i],
                    end_node=path_to_target[i + 1],
                    path_nodes=[path_to_target[i], path_to_target[i + 1]]
                )
                segment.calculate_properties(self.graph)
                route_segments.append(segment)
            
            # Return path (reversed)
            return_path = list(reversed(path_to_target))
            for i in range(len(return_path) - 1):
                segment = RouteSegment(
                    start_node=return_path[i],
                    end_node=return_path[i + 1],
                    path_nodes=[return_path[i], return_path[i + 1]]
                )
                segment.calculate_properties(self.graph)
                route_segments.append(segment)
            
            # Create chromosome
            chromosome = RouteChromosome(route_segments)
            chromosome.creation_method = f"terrain_aware_{category}_elevation"
            
            return chromosome
            
        except (nx.NetworkXNoPath, nx.NodeNotFound, Exception) as e:
            return None
    
    def _create_random_routes(self, count: int) -> List[RouteChromosome]:
        """Create random routes as fallback
        
        Args:
            count: Number of random routes to create
            
        Returns:
            List of route chromosomes
        """
        routes = []
        
        for _ in range(count):
            route = self._create_random_walk_route()
            if route:
                routes.append(route)
        
        return routes
    
    def _create_random_walk_route(self) -> Optional[RouteChromosome]:
        """Create a random walk route
        
        Returns:
            Route chromosome or None
        """
        try:
            route_segments = []
            current_node = self.start_node
            visited_nodes = {current_node}
            
            # Random walk for 3-6 steps
            steps = random.randint(3, 6)
            
            for _ in range(steps):
                # Get neighbors
                neighbors = [n for n in self.graph.neighbors(current_node) 
                           if n not in visited_nodes]
                
                if not neighbors:
                    # No unvisited neighbors, try any neighbor
                    neighbors = list(self.graph.neighbors(current_node))
                
                if not neighbors:
                    break
                
                # Select random neighbor
                next_node = random.choice(neighbors)
                
                # Create segment
                segment = RouteSegment(
                    start_node=current_node,
                    end_node=next_node,
                    path_nodes=[current_node, next_node]
                )
                segment.calculate_properties(self.graph)
                route_segments.append(segment)
                
                visited_nodes.add(next_node)
                current_node = next_node
            
            # Return to start
            if current_node != self.start_node:
                try:
                    return_path = nx.shortest_path(self.graph, current_node, self.start_node, weight='length')
                    
                    for i in range(len(return_path) - 1):
                        segment = RouteSegment(
                            start_node=return_path[i],
                            end_node=return_path[i + 1],
                            path_nodes=[return_path[i], return_path[i + 1]]
                        )
                        segment.calculate_properties(self.graph)
                        route_segments.append(segment)
                        
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    return None
            
            # Create chromosome
            if route_segments:
                chromosome = RouteChromosome(route_segments)
                chromosome.creation_method = "terrain_aware_random_walk"
                return chromosome
            
        except Exception:
            pass
        
        return None
    
    def get_initialization_stats(self) -> Dict[str, any]:
        """Get statistics about terrain-aware initialization
        
        Returns:
            Dictionary with initialization statistics
        """
        return {
            'start_elevation': self.start_elevation,
            'nearby_nodes_count': len(self.nearby_nodes),
            'high_elevation_nodes_count': len(self.high_elevation_nodes),
            'very_high_elevation_nodes_count': len(self.very_high_elevation_nodes),
            'moderate_elevation_nodes_count': len(self.moderate_elevation_nodes),
            'reachable_nodes_count': len(self.reachable_nodes) if hasattr(self, 'reachable_nodes') else 0,
            'elevation_gain_threshold': self.config.elevation_gain_threshold,
            'max_elevation_gain_threshold': self.config.max_elevation_gain_threshold,
            'exploration_radius_km': self.target_distance_km * self.config.exploration_radius_multiplier,
            'target_percentages': {
                'very_high': self.config.very_high_elevation_percentage,
                'high': self.config.high_elevation_percentage,
                'moderate': 0.3,
                'random': 1.0 - self.config.very_high_elevation_percentage - self.config.high_elevation_percentage - 0.3
            }
        }