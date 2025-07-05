#!/usr/bin/env python3
"""
GA Performance Caching System
Comprehensive caching for genetic algorithm route optimization
"""

import os
import pickle
import hashlib
import time
import threading
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from collections import OrderedDict
import networkx as nx
import numpy as np

from ga_chromosome import RouteSegment


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    size: int = 0
    memory_mb: float = 0.0
    hit_rate: float = 0.0
    total_requests: int = 0


class LRUCache:
    """Thread-safe LRU (Least Recently Used) cache implementation"""
    
    def __init__(self, max_size: int = 10000):
        """Initialize LRU cache
        
        Args:
            max_size: Maximum number of items to cache
        """
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.stats = CacheStats()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self.lock:
            self.stats.total_requests += 1
            
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.stats.hits += 1
                self._update_hit_rate()
                return value
            else:
                self.stats.misses += 1
                self._update_hit_rate()
                return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            if key in self.cache:
                # Update existing item
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used item
                self.cache.popitem(last=False)
            
            self.cache[key] = value
            self.stats.size = len(self.cache)
    
    def clear(self) -> None:
        """Clear all cached items"""
        with self.lock:
            self.cache.clear()
            self.stats.size = 0
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self.lock:
            # Estimate memory usage
            try:
                import sys
                memory_bytes = sum(sys.getsizeof(k) + sys.getsizeof(v) 
                                 for k, v in self.cache.items())
                self.stats.memory_mb = memory_bytes / (1024 * 1024)
            except:
                self.stats.memory_mb = 0.0
            
            return self.stats
    
    def _update_hit_rate(self) -> None:
        """Update hit rate statistics"""
        if self.stats.total_requests > 0:
            self.stats.hit_rate = self.stats.hits / self.stats.total_requests


class GAPerformanceCache:
    """High-performance caching system for GA operations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize performance cache
        
        Args:
            config: Cache configuration options
        """
        # Default configuration
        default_config = {
            'segment_cache_size': 50000,
            'distance_cache_size': 100000,
            'path_cache_size': 25000,
            'fitness_cache_size': 10000,
            'enable_disk_cache': True,
            'disk_cache_dir': 'ga_cache',
            'cache_compression': True,
            'max_memory_mb': 500
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Initialize in-memory caches
        self.segment_cache = LRUCache(self.config['segment_cache_size'])
        self.distance_cache = LRUCache(self.config['distance_cache_size'])
        self.path_cache = LRUCache(self.config['path_cache_size'])
        self.fitness_cache = LRUCache(self.config['fitness_cache_size'])
        
        # Disk cache setup
        if self.config['enable_disk_cache']:
            self.disk_cache_dir = self.config['disk_cache_dir']
            os.makedirs(self.disk_cache_dir, exist_ok=True)
        
        # Performance tracking
        self.cache_start_time = time.time()
        self.total_cache_saves = 0
        self.total_cache_loads = 0
        
        # Distance matrix cache for fast lookups
        self.distance_matrix = {}
        self.distance_matrix_dirty = False
        
    def get_segment(self, start_node: int, end_node: int, 
                   graph: nx.Graph) -> Optional[RouteSegment]:
        """Get cached route segment
        
        Args:
            start_node: Segment start node
            end_node: Segment end node
            graph: Network graph
            
        Returns:
            Cached segment or None
        """
        cache_key = f"seg_{start_node}_{end_node}"
        segment = self.segment_cache.get(cache_key)
        
        if segment is None:
            # Create and cache new segment
            try:
                path = nx.shortest_path(graph, start_node, end_node, weight='length')
                segment = RouteSegment(start_node, end_node, path)
                segment.calculate_properties(graph)
                
                # Cache the segment
                self.segment_cache.put(cache_key, segment)
                
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # Cache negative result to avoid repeated failures
                self.segment_cache.put(cache_key, None)
                return None
        
        return segment.copy() if segment else None
    
    def get_distance(self, node1: int, node2: int, graph: nx.Graph) -> Optional[float]:
        """Get cached distance between nodes
        
        Args:
            node1: First node
            node2: Second node
            graph: Network graph
            
        Returns:
            Distance in meters or None if no path
        """
        # Use symmetric key (smaller node first)
        key_nodes = tuple(sorted([node1, node2]))
        cache_key = f"dist_{key_nodes[0]}_{key_nodes[1]}"
        
        distance = self.distance_cache.get(cache_key)
        
        if distance is None:
            try:
                distance = nx.shortest_path_length(graph, node1, node2, weight='length')
                self.distance_cache.put(cache_key, distance)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # Cache negative result
                self.distance_cache.put(cache_key, -1.0)
                return None
        
        return distance if distance >= 0 else None
    
    def get_path(self, start_node: int, end_node: int, 
                graph: nx.Graph) -> Optional[List[int]]:
        """Get cached path between nodes
        
        Args:
            start_node: Path start node
            end_node: Path end node
            graph: Network graph
            
        Returns:
            List of nodes in path or None
        """
        cache_key = f"path_{start_node}_{end_node}"
        path = self.path_cache.get(cache_key)
        
        if path is None:
            try:
                path = nx.shortest_path(graph, start_node, end_node, weight='length')
                self.path_cache.put(cache_key, path)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                self.path_cache.put(cache_key, [])
                return None
        
        return path if path else None
    
    def get_fitness_components(self, chromosome_id: str) -> Optional[Dict[str, float]]:
        """Get cached fitness components for chromosome
        
        Args:
            chromosome_id: Unique chromosome identifier
            
        Returns:
            Cached fitness components or None
        """
        cache_key = f"fitness_{chromosome_id}"
        return self.fitness_cache.get(cache_key)
    
    def put_fitness_components(self, chromosome_id: str, 
                             components: Dict[str, float]) -> None:
        """Cache fitness components for chromosome
        
        Args:
            chromosome_id: Unique chromosome identifier
            components: Fitness components to cache
        """
        cache_key = f"fitness_{chromosome_id}"
        self.fitness_cache.put(cache_key, components)
    
    def build_distance_matrix(self, nodes: List[int], graph: nx.Graph,
                            progress_callback: Optional[callable] = None) -> np.ndarray:
        """Build optimized distance matrix for frequent lookups
        
        Args:
            nodes: List of nodes to include in matrix
            graph: Network graph
            progress_callback: Optional progress reporting function
            
        Returns:
            Distance matrix (numpy array)
        """
        n = len(nodes)
        matrix_key = f"matrix_{hash(tuple(sorted(nodes)))}"
        
        # Check if matrix is already cached
        if matrix_key in self.distance_matrix:
            return self.distance_matrix[matrix_key]
        
        # Build new matrix
        distance_matrix = np.full((n, n), np.inf)
        
        for i, node1 in enumerate(nodes):
            if progress_callback:
                progress_callback(i / n)
            
            # Set diagonal to 0
            distance_matrix[i, i] = 0.0
            
            for j, node2 in enumerate(nodes[i+1:], i+1):
                distance = self.get_distance(node1, node2, graph)
                if distance is not None:
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance  # Symmetric
        
        # Cache the matrix
        self.distance_matrix[matrix_key] = distance_matrix
        self.distance_matrix_dirty = True
        
        return distance_matrix
    
    def precompute_common_segments(self, graph: nx.Graph, start_node: int,
                                 max_distance: float = 2000.0,
                                 progress_callback: Optional[callable] = None) -> None:
        """Precompute segments for nodes within radius of start node
        
        Args:
            graph: Network graph
            start_node: Central node for precomputation
            max_distance: Maximum distance for precomputation
            progress_callback: Optional progress reporting function
        """
        # Get nodes within radius
        nearby_nodes = []
        
        try:
            # Use ego graph for efficiency
            ego_graph = nx.ego_graph(graph, start_node, radius=3)
            candidate_nodes = list(ego_graph.nodes())
            
            # Filter by actual distance
            for node in candidate_nodes:
                distance = self.get_distance(start_node, node, graph)
                if distance is not None and distance <= max_distance:
                    nearby_nodes.append(node)
            
        except Exception:
            # Fallback to all nodes (less efficient)
            nearby_nodes = [node for node in graph.nodes() 
                          if self.get_distance(start_node, node, graph) is not None]
        
        # Precompute segments between nearby nodes
        total_pairs = len(nearby_nodes) * (len(nearby_nodes) - 1) // 2
        computed = 0
        
        for i, node1 in enumerate(nearby_nodes):
            if progress_callback:
                progress_callback(computed / total_pairs)
            
            for node2 in nearby_nodes[i+1:]:
                # This will cache the segment
                self.get_segment(node1, node2, graph)
                computed += 1
    
    def save_to_disk(self, filename: Optional[str] = None) -> str:
        """Save cache to disk
        
        Args:
            filename: Optional filename for cache file
            
        Returns:
            Path to saved cache file
        """
        if not self.config['enable_disk_cache']:
            raise ValueError("Disk cache is disabled")
        
        if filename is None:
            timestamp = int(time.time())
            filename = f"ga_cache_{timestamp}.pkl"
        
        filepath = os.path.join(self.disk_cache_dir, filename)
        
        # Prepare cache data
        cache_data = {
            'segment_cache': dict(self.segment_cache.cache),
            'distance_cache': dict(self.distance_cache.cache),
            'path_cache': dict(self.path_cache.cache),
            'fitness_cache': dict(self.fitness_cache.cache),
            'distance_matrix': self.distance_matrix,
            'config': self.config,
            'timestamp': time.time()
        }
        
        # Save with optional compression
        with open(filepath, 'wb') as f:
            if self.config['cache_compression']:
                import bz2
                compressed_data = bz2.compress(pickle.dumps(cache_data))
                pickle.dump(compressed_data, f)
            else:
                pickle.dump(cache_data, f)
        
        self.total_cache_saves += 1
        return filepath
    
    def load_from_disk(self, filepath: str) -> bool:
        """Load cache from disk
        
        Args:
            filepath: Path to cache file
            
        Returns:
            True if loaded successfully
        """
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'rb') as f:
                if self.config['cache_compression']:
                    import bz2
                    compressed_data = pickle.load(f)
                    cache_data = pickle.loads(bz2.decompress(compressed_data))
                else:
                    cache_data = pickle.load(f)
            
            # Restore cache data
            for key, value in cache_data['segment_cache'].items():
                self.segment_cache.put(key, value)
            
            for key, value in cache_data['distance_cache'].items():
                self.distance_cache.put(key, value)
            
            for key, value in cache_data['path_cache'].items():
                self.path_cache.put(key, value)
            
            for key, value in cache_data['fitness_cache'].items():
                self.fitness_cache.put(key, value)
            
            self.distance_matrix = cache_data.get('distance_matrix', {})
            
            self.total_cache_loads += 1
            return True
            
        except Exception as e:
            print(f"Failed to load cache from {filepath}: {e}")
            return False
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics for all caches
        
        Returns:
            Dictionary with memory usage in MB
        """
        return {
            'segment_cache_mb': self.segment_cache.get_stats().memory_mb,
            'distance_cache_mb': self.distance_cache.get_stats().memory_mb,
            'path_cache_mb': self.path_cache.get_stats().memory_mb,
            'fitness_cache_mb': self.fitness_cache.get_stats().memory_mb,
            'total_mb': (self.segment_cache.get_stats().memory_mb +
                        self.distance_cache.get_stats().memory_mb +
                        self.path_cache.get_stats().memory_mb +
                        self.fitness_cache.get_stats().memory_mb)
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        uptime = time.time() - self.cache_start_time
        
        return {
            'uptime_seconds': uptime,
            'segment_cache': self.segment_cache.get_stats(),
            'distance_cache': self.distance_cache.get_stats(),
            'path_cache': self.path_cache.get_stats(),
            'fitness_cache': self.fitness_cache.get_stats(),
            'memory_usage': self.get_memory_usage(),
            'disk_operations': {
                'saves': self.total_cache_saves,
                'loads': self.total_cache_loads
            },
            'distance_matrices': len(self.distance_matrix)
        }
    
    def optimize_memory(self, target_mb: Optional[float] = None) -> Dict[str, int]:
        """Optimize memory usage by clearing least effective caches
        
        Args:
            target_mb: Target memory usage (uses config default if None)
            
        Returns:
            Dictionary with items cleared from each cache
        """
        if target_mb is None:
            target_mb = self.config['max_memory_mb']
        
        current_usage = self.get_memory_usage()['total_mb']
        
        if current_usage <= target_mb:
            return {'segment_cache': 0, 'distance_cache': 0, 'path_cache': 0, 'fitness_cache': 0}
        
        # Calculate items to clear based on hit rates and memory usage
        caches = [
            ('fitness_cache', self.fitness_cache),
            ('path_cache', self.path_cache),
            ('segment_cache', self.segment_cache),
            ('distance_cache', self.distance_cache)
        ]
        
        cleared = {}
        
        for cache_name, cache in caches:
            if current_usage <= target_mb:
                break
            
            stats = cache.get_stats()
            if stats.hit_rate < 0.5:  # Clear caches with low hit rates first
                items_to_clear = min(len(cache.cache) // 2, stats.size)
                
                # Clear least recently used items
                with cache.lock:
                    for _ in range(items_to_clear):
                        if cache.cache:
                            cache.cache.popitem(last=False)
                
                cleared[cache_name] = items_to_clear
                cache.stats.size = len(cache.cache)
                
                # Update current usage estimate
                current_usage = self.get_memory_usage()['total_mb']
            else:
                cleared[cache_name] = 0
        
        return cleared
    
    def clear_all_caches(self) -> None:
        """Clear all cached data"""
        self.segment_cache.clear()
        self.distance_cache.clear()
        self.path_cache.clear()
        self.fitness_cache.clear()
        self.distance_matrix.clear()
        self.distance_matrix_dirty = False


def test_performance_cache():
    """Test function for performance cache"""
    print("Testing GA Performance Cache...")
    
    # Create test graph
    test_graph = nx.Graph()
    nodes = [(1, -80.4094, 37.1299, 100), (2, -80.4000, 37.1300, 110),
             (3, -80.4050, 37.1350, 105), (4, -80.4100, 37.1250, 120)]
    
    for node_id, x, y, elev in nodes:
        test_graph.add_node(node_id, x=x, y=y, elevation=elev)
    
    edges = [(1, 2, 100), (2, 3, 150), (3, 4, 200), (4, 1, 180), (1, 3, 250), (2, 4, 220)]
    for n1, n2, length in edges:
        test_graph.add_edge(n1, n2, length=length)
    
    # Test cache
    cache = GAPerformanceCache({'segment_cache_size': 100, 'enable_disk_cache': False})
    
    # Test segment caching
    segment1 = cache.get_segment(1, 2, test_graph)
    segment2 = cache.get_segment(1, 2, test_graph)  # Should be cached
    
    print(f"✅ Segment caching: {segment1 is not None}, cache hit: {segment2 is not None}")
    
    # Test distance caching
    dist1 = cache.get_distance(1, 3, test_graph)
    dist2 = cache.get_distance(1, 3, test_graph)  # Should be cached
    
    print(f"✅ Distance caching: {dist1}, cache hit: {dist2}")
    
    # Test path caching
    path1 = cache.get_path(1, 4, test_graph)
    path2 = cache.get_path(1, 4, test_graph)  # Should be cached
    
    print(f"✅ Path caching: {len(path1) if path1 else 0} nodes, cache hit: {len(path2) if path2 else 0} nodes")
    
    # Test performance stats
    stats = cache.get_performance_stats()
    print(f"✅ Performance stats: {stats['segment_cache'].hits} segment hits")
    
    # Test memory optimization
    memory_usage = cache.get_memory_usage()
    print(f"✅ Memory usage: {memory_usage['total_mb']:.3f} MB")
    
    print("✅ All performance cache tests completed")


if __name__ == "__main__":
    test_performance_cache()