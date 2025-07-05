#!/usr/bin/env python3
"""
GA Distance Calculation Optimizer
High-performance distance calculations with vectorization and caching
"""

import numpy as np
import networkx as nx
import time
import math
from typing import List, Dict, Tuple, Optional, Set, Union, Any
from dataclasses import dataclass
from collections import defaultdict
import heapq

from ga_chromosome import RouteChromosome, RouteSegment


@dataclass
class DistanceCalculationStats:
    """Statistics for distance calculations"""
    total_calculations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    vectorized_calculations: int = 0
    total_calculation_time: float = 0.0
    avg_calculation_time: float = 0.0
    cache_hit_rate: float = 0.0


class FastHaversine:
    """Optimized Haversine distance calculations using vectorization"""
    
    # Earth radius in meters
    EARTH_RADIUS_M = 6371000.0
    
    @staticmethod
    def vectorized_haversine(lat1: np.ndarray, lon1: np.ndarray, 
                           lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """Vectorized Haversine distance calculation
        
        Args:
            lat1, lon1: First point coordinates (arrays)
            lat2, lon2: Second point coordinates (arrays)
            
        Returns:
            Array of distances in meters
        """
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # Differences
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Haversine formula
        a = (np.sin(dlat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
        
        c = 2 * np.arcsin(np.sqrt(a))
        
        return FastHaversine.EARTH_RADIUS_M * c
    
    @staticmethod
    def single_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Single Haversine distance calculation
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in meters
        """
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Differences
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Haversine formula
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        
        c = 2 * math.asin(math.sqrt(a))
        
        return FastHaversine.EARTH_RADIUS_M * c


class OptimizedDistanceMatrix:
    """High-performance distance matrix with smart caching and vectorization"""
    
    def __init__(self, graph: nx.Graph, cache_size: int = 50000):
        """Initialize optimized distance matrix
        
        Args:
            graph: NetworkX graph
            cache_size: Maximum cache size for distance calculations
        """
        self.graph = graph
        self.cache_size = cache_size
        
        # Extract node coordinates for vectorized calculations
        self.node_coords = {}
        self.coord_arrays = None
        self.node_to_index = {}
        
        self._build_coordinate_arrays()
        
        # Distance caches
        self.exact_distance_cache = {}  # Node-to-node exact distances
        self.haversine_cache = {}       # Haversine distances
        self.path_cache = {}            # Shortest paths
        
        # Statistics
        self.stats = DistanceCalculationStats()
        
        # Precomputed distance matrices for common node sets
        self.matrix_cache = {}
        
    def _build_coordinate_arrays(self):
        """Build coordinate arrays for vectorized calculations"""
        nodes = list(self.graph.nodes())
        
        if not nodes:
            return
        
        # Extract coordinates
        coords = []
        for i, node in enumerate(nodes):
            node_data = self.graph.nodes[node]
            lat = node_data.get('y', 0.0)  # y is latitude
            lon = node_data.get('x', 0.0)  # x is longitude
            
            coords.append((lat, lon))
            self.node_coords[node] = (lat, lon)
            self.node_to_index[node] = i
        
        # Convert to numpy arrays for vectorized operations
        if coords:
            coords_array = np.array(coords)
            self.coord_arrays = {
                'lat': coords_array[:, 0],
                'lon': coords_array[:, 1],
                'nodes': np.array(nodes)
            }
    
    def get_haversine_distance(self, node1: int, node2: int) -> float:
        """Get Haversine (great-circle) distance between nodes
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            Haversine distance in meters
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = tuple(sorted([node1, node2]))
        if cache_key in self.haversine_cache:
            self.stats.cache_hits += 1
            return self.haversine_cache[cache_key]
        
        self.stats.cache_misses += 1
        
        # Calculate Haversine distance
        if node1 in self.node_coords and node2 in self.node_coords:
            lat1, lon1 = self.node_coords[node1]
            lat2, lon2 = self.node_coords[node2]
            
            distance = FastHaversine.single_haversine(lat1, lon1, lat2, lon2)
            
            # Cache result
            if len(self.haversine_cache) < self.cache_size:
                self.haversine_cache[cache_key] = distance
            
            self.stats.total_calculation_time += time.time() - start_time
            self.stats.total_calculations += 1
            
            return distance
        
        return float('inf')
    
    def get_network_distance(self, node1: int, node2: int) -> Optional[float]:
        """Get exact network distance between nodes
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            Network distance in meters or None if no path
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = tuple(sorted([node1, node2]))
        if cache_key in self.exact_distance_cache:
            self.stats.cache_hits += 1
            return self.exact_distance_cache[cache_key]
        
        self.stats.cache_misses += 1
        
        # Calculate network distance
        try:
            distance = nx.shortest_path_length(self.graph, node1, node2, weight='length')
            
            # Cache result
            if len(self.exact_distance_cache) < self.cache_size:
                self.exact_distance_cache[cache_key] = distance
            
            self.stats.total_calculation_time += time.time() - start_time
            self.stats.total_calculations += 1
            
            return distance
            
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Cache negative result
            if len(self.exact_distance_cache) < self.cache_size:
                self.exact_distance_cache[cache_key] = None
            
            return None
    
    def build_distance_matrix_vectorized(self, nodes: List[int]) -> np.ndarray:
        """Build distance matrix using vectorized operations
        
        Args:
            nodes: List of nodes to include in matrix
            
        Returns:
            Distance matrix (numpy array)
        """
        n = len(nodes)
        
        # Check if matrix is already cached
        cache_key = tuple(sorted(nodes))
        if cache_key in self.matrix_cache:
            self.stats.cache_hits += 1
            return self.matrix_cache[cache_key].copy()
        
        self.stats.cache_misses += 1
        start_time = time.time()
        
        # Extract coordinates for these nodes
        node_coords = []
        for node in nodes:
            if node in self.node_coords:
                node_coords.append(self.node_coords[node])
            else:
                node_coords.append((0.0, 0.0))  # Default for missing coordinates
        
        coords_array = np.array(node_coords)
        lats = coords_array[:, 0]
        lons = coords_array[:, 1]
        
        # Create coordinate meshgrids for vectorized calculation
        lat1_grid, lat2_grid = np.meshgrid(lats, lats)
        lon1_grid, lon2_grid = np.meshgrid(lons, lons)
        
        # Vectorized Haversine calculation
        distance_matrix = FastHaversine.vectorized_haversine(
            lat1_grid.flatten(), lon1_grid.flatten(),
            lat2_grid.flatten(), lon2_grid.flatten()
        ).reshape(n, n)
        
        # Set diagonal to 0
        np.fill_diagonal(distance_matrix, 0.0)
        
        # Cache the matrix
        if len(self.matrix_cache) < 100:  # Limit matrix cache size
            self.matrix_cache[cache_key] = distance_matrix.copy()
        
        self.stats.vectorized_calculations += n * n
        self.stats.total_calculation_time += time.time() - start_time
        
        return distance_matrix
    
    def get_nearest_neighbors_vectorized(self, center_node: int, 
                                       candidate_nodes: List[int],
                                       k: int = 10) -> List[Tuple[int, float]]:
        """Get k nearest neighbors using vectorized calculations
        
        Args:
            center_node: Center node
            candidate_nodes: Nodes to search among
            k: Number of neighbors to return
            
        Returns:
            List of (node, distance) tuples
        """
        if center_node not in self.node_coords:
            return []
        
        start_time = time.time()
        
        # Get center coordinates
        center_lat, center_lon = self.node_coords[center_node]
        
        # Extract candidate coordinates
        candidate_coords = []
        valid_candidates = []
        
        for node in candidate_nodes:
            if node in self.node_coords and node != center_node:
                candidate_coords.append(self.node_coords[node])
                valid_candidates.append(node)
        
        if not candidate_coords:
            return []
        
        # Vectorized distance calculation
        coords_array = np.array(candidate_coords)
        candidate_lats = coords_array[:, 0]
        candidate_lons = coords_array[:, 1]
        
        distances = FastHaversine.vectorized_haversine(
            np.full(len(candidate_lats), center_lat),
            np.full(len(candidate_lons), center_lon),
            candidate_lats,
            candidate_lons
        )
        
        # Get k nearest
        k = min(k, len(distances))
        nearest_indices = np.argpartition(distances, k)[:k]
        
        result = [(valid_candidates[i], distances[i]) for i in nearest_indices]
        result.sort(key=lambda x: x[1])  # Sort by distance
        
        self.stats.vectorized_calculations += len(candidate_coords)
        self.stats.total_calculation_time += time.time() - start_time
        
        return result
    
    def estimate_route_distance_fast(self, chromosome: RouteChromosome) -> float:
        """Fast route distance estimation using cached calculations
        
        Args:
            chromosome: Route chromosome
            
        Returns:
            Estimated total distance
        """
        if not chromosome.segments:
            return 0.0
        
        start_time = time.time()
        total_distance = 0.0
        
        for segment in chromosome.segments:
            if segment.length > 0:
                # Use cached segment length
                total_distance += segment.length
            else:
                # Estimate using Haversine distance
                if segment.start_node and segment.end_node:
                    haversine_dist = self.get_haversine_distance(
                        segment.start_node, segment.end_node
                    )
                    # Apply correction factor for network vs Haversine distance
                    network_dist = haversine_dist * 1.4  # Typical factor
                    total_distance += network_dist
        
        self.stats.total_calculation_time += time.time() - start_time
        self.stats.total_calculations += 1
        
        return total_distance


class GADistanceOptimizer:
    """Main distance optimization system for GA operations"""
    
    def __init__(self, graph: nx.Graph, config: Optional[Dict[str, Any]] = None):
        """Initialize distance optimizer
        
        Args:
            graph: NetworkX graph
            config: Configuration options
        """
        default_config = {
            'cache_size': 50000,
            'enable_vectorization': True,
            'enable_matrix_caching': True,
            'precompute_common_distances': True,
            'distance_estimation_threshold': 1000  # Use estimation for distances > 1km
        }
        
        self.config = {**default_config, **(config or {})}
        self.graph = graph
        
        # Initialize optimized distance matrix
        self.distance_matrix = OptimizedDistanceMatrix(graph, self.config['cache_size'])
        
        # Segment distance cache
        self.segment_distance_cache = {}
        
        # Common distance patterns
        self.common_distances = {}
        
        if self.config['precompute_common_distances']:
            self._precompute_common_distances()
    
    def _precompute_common_distances(self):
        """Precompute distances for commonly used node pairs"""
        # This would be called during initialization to warm up caches
        # For now, we'll implement this as a placeholder
        pass
    
    def calculate_chromosome_distance_optimized(self, chromosome: RouteChromosome) -> float:
        """Calculate chromosome distance using optimized methods
        
        Args:
            chromosome: Route chromosome
            
        Returns:
            Total route distance in meters
        """
        if not chromosome.segments:
            return 0.0
        
        # Check if we have cached total distance
        chromosome_id = id(chromosome)
        if chromosome_id in self.segment_distance_cache:
            self.distance_matrix.stats.cache_hits += 1
            return self.segment_distance_cache[chromosome_id]
        
        self.distance_matrix.stats.cache_misses += 1
        
        total_distance = 0.0
        
        for segment in chromosome.segments:
            if segment.length > 0:
                # Use pre-calculated segment length
                total_distance += segment.length
            else:
                # Calculate segment distance
                segment_distance = self._calculate_segment_distance_optimized(segment)
                total_distance += segment_distance
                
                # Update segment with calculated distance
                segment.length = segment_distance
        
        # Cache total distance
        self.segment_distance_cache[chromosome_id] = total_distance
        
        return total_distance
    
    def _calculate_segment_distance_optimized(self, segment: RouteSegment) -> float:
        """Calculate segment distance using optimized pathfinding
        
        Args:
            segment: Route segment
            
        Returns:
            Segment distance in meters
        """
        if not segment.path_nodes or len(segment.path_nodes) < 2:
            # Use Haversine distance as fallback
            return self.distance_matrix.get_haversine_distance(
                segment.start_node, segment.end_node
            )
        
        # Calculate distance along path
        total_distance = 0.0
        
        for i in range(len(segment.path_nodes) - 1):
            node1 = segment.path_nodes[i]
            node2 = segment.path_nodes[i + 1]
            
            # Try to get edge distance from graph
            if self.graph.has_edge(node1, node2):
                edge_data = self.graph[node1][node2]
                if isinstance(edge_data, dict):
                    edge_distance = edge_data.get('length', 0.0)
                else:
                    # MultiGraph case
                    edge_distance = min(data.get('length', 0.0) 
                                      for data in edge_data.values())
                
                if edge_distance > 0:
                    total_distance += edge_distance
                else:
                    # Use Haversine as fallback
                    total_distance += self.distance_matrix.get_haversine_distance(node1, node2)
            else:
                # Use Haversine distance
                total_distance += self.distance_matrix.get_haversine_distance(node1, node2)
        
        return total_distance
    
    def find_nearest_nodes_optimized(self, center_node: int, 
                                   max_distance: float = 2000.0,
                                   k: int = 20) -> List[Tuple[int, float]]:
        """Find nearest nodes using optimized search
        
        Args:
            center_node: Center node for search
            max_distance: Maximum search distance
            k: Maximum number of nodes to return
            
        Returns:
            List of (node, distance) tuples
        """
        # Get candidate nodes (rough filtering)
        candidate_nodes = list(self.graph.nodes())
        
        # Use vectorized nearest neighbor search
        nearest = self.distance_matrix.get_nearest_neighbors_vectorized(
            center_node, candidate_nodes, k * 2  # Get more candidates initially
        )
        
        # Filter by distance and refine with network distances
        result = []
        for node, haversine_dist in nearest:
            if haversine_dist <= max_distance * 1.5:  # Allow some buffer for network distance
                network_dist = self.distance_matrix.get_network_distance(center_node, node)
                if network_dist is not None and network_dist <= max_distance:
                    result.append((node, network_dist))
            
            if len(result) >= k:
                break
        
        return result[:k]
    
    def build_optimized_distance_matrix(self, nodes: List[int]) -> np.ndarray:
        """Build optimized distance matrix for a set of nodes
        
        Args:
            nodes: Nodes to include in matrix
            
        Returns:
            Distance matrix
        """
        if self.config['enable_vectorization'] and len(nodes) > 10:
            # Use vectorized calculation for larger matrices
            return self.distance_matrix.build_distance_matrix_vectorized(nodes)
        else:
            # Use traditional calculation for smaller matrices
            n = len(nodes)
            matrix = np.zeros((n, n))
            
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes[i+1:], i+1):
                    distance = self.distance_matrix.get_network_distance(node1, node2)
                    if distance is not None:
                        matrix[i, j] = distance
                        matrix[j, i] = distance
                    else:
                        matrix[i, j] = float('inf')
                        matrix[j, i] = float('inf')
            
            return matrix
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get distance optimization statistics
        
        Returns:
            Dictionary with performance metrics
        """
        stats = self.distance_matrix.stats
        
        # Update computed statistics
        if stats.total_calculations > 0:
            stats.avg_calculation_time = stats.total_calculation_time / stats.total_calculations
        
        if stats.cache_hits + stats.cache_misses > 0:
            stats.cache_hit_rate = stats.cache_hits / (stats.cache_hits + stats.cache_misses)
        
        return {
            'total_calculations': stats.total_calculations,
            'cache_hits': stats.cache_hits,
            'cache_misses': stats.cache_misses,
            'cache_hit_rate': stats.cache_hit_rate,
            'vectorized_calculations': stats.vectorized_calculations,
            'total_calculation_time': stats.total_calculation_time,
            'avg_calculation_time': stats.avg_calculation_time,
            'haversine_cache_size': len(self.distance_matrix.haversine_cache),
            'exact_distance_cache_size': len(self.distance_matrix.exact_distance_cache),
            'matrix_cache_size': len(self.distance_matrix.matrix_cache)
        }
    
    def clear_caches(self):
        """Clear all distance caches"""
        self.distance_matrix.haversine_cache.clear()
        self.distance_matrix.exact_distance_cache.clear()
        self.distance_matrix.matrix_cache.clear()
        self.segment_distance_cache.clear()


def test_distance_optimizer():
    """Test function for distance optimizer"""
    print("Testing GA Distance Optimizer...")
    
    # Create test graph with coordinates
    test_graph = nx.Graph()
    nodes = [
        (1, -80.4094, 37.1299, 100), (2, -80.4000, 37.1300, 110),
        (3, -80.4050, 37.1350, 105), (4, -80.4100, 37.1250, 120),
        (5, -80.4200, 37.1400, 95)
    ]
    
    for node_id, x, y, elev in nodes:
        test_graph.add_node(node_id, x=x, y=y, elevation=elev)
    
    edges = [(1, 2, 100), (2, 3, 150), (3, 4, 200), (4, 5, 180), 
             (5, 1, 250), (1, 3, 220), (2, 4, 160)]
    for n1, n2, length in edges:
        test_graph.add_edge(n1, n2, length=length)
    
    # Test optimizer
    optimizer = GADistanceOptimizer(test_graph)
    
    # Test Haversine calculation
    haversine_dist = optimizer.distance_matrix.get_haversine_distance(1, 3)
    print(f"✅ Haversine distance 1->3: {haversine_dist:.1f}m")
    
    # Test network distance
    network_dist = optimizer.distance_matrix.get_network_distance(1, 3)
    print(f"✅ Network distance 1->3: {network_dist}m")
    
    # Test vectorized matrix
    nodes_list = [1, 2, 3, 4]
    distance_matrix = optimizer.build_optimized_distance_matrix(nodes_list)
    print(f"✅ Distance matrix shape: {distance_matrix.shape}")
    
    # Test nearest neighbors
    nearest = optimizer.find_nearest_nodes_optimized(1, max_distance=500, k=3)
    print(f"✅ Nearest neighbors to node 1: {len(nearest)} found")
    
    # Test performance stats
    stats = optimizer.get_optimization_stats()
    print(f"✅ Performance stats: {stats['total_calculations']} calculations")
    print(f"   Cache hit rate: {stats['cache_hit_rate']:.2f}")
    
    print("✅ All distance optimizer tests completed")


if __name__ == "__main__":
    test_distance_optimizer()