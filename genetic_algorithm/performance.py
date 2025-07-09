#!/usr/bin/env python3
"""
GA Segment-Level Caching System
Caches individual route segment properties to accelerate fitness evaluation
"""

import time
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import OrderedDict
import threading
import networkx as nx

from .chromosome import RouteSegment


@dataclass
class SegmentProperties:
    """Cached properties for a route segment"""
    distance_km: float
    elevation_gain_m: float
    elevation_loss_m: float
    net_elevation_m: float
    max_elevation_m: float
    min_elevation_m: float
    avg_grade_percent: float
    max_grade_percent: float
    num_nodes: int
    connectivity_score: float
    calculated_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Ensure all numeric values are valid"""
        # Replace None with 0.0 for any missing values
        for field_name in ['distance_km', 'elevation_gain_m', 'elevation_loss_m', 
                          'net_elevation_m', 'max_elevation_m', 'min_elevation_m',
                          'avg_grade_percent', 'max_grade_percent', 'connectivity_score']:
            value = getattr(self, field_name)
            if value is None or not isinstance(value, (int, float)):
                setattr(self, field_name, 0.0)
        
        if self.num_nodes is None or not isinstance(self.num_nodes, int):
            self.num_nodes = 0


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    calculations_saved: int = 0
    total_calculation_time_saved: float = 0.0
    avg_calculation_time: float = 0.0
    total_calculation_time: float = 0.0
    calculation_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate as percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.cache_hits / self.total_requests) * 100
    
    @property
    def time_saved_per_hit(self) -> float:
        """Average time saved per cache hit"""
        if self.cache_hits == 0:
            return 0.0
        return self.avg_calculation_time
    
    def update_calculation_time(self, calculation_time: float):
        """Update running average of calculation times"""
        self.total_calculation_time += calculation_time
        self.calculation_count += 1
        self.avg_calculation_time = self.total_calculation_time / self.calculation_count
    
    def record_cache_hit(self):
        """Record a cache hit and calculate time saved"""
        self.cache_hits += 1
        self.total_calculation_time_saved += self.avg_calculation_time


class GASegmentCache:
    """Thread-safe segment-level cache for GA fitness evaluation"""
    
    def __init__(self, max_size: int = 5000, enable_stats: bool = True):
        """Initialize segment cache
        
        Args:
            max_size: Maximum number of segments to cache
            enable_stats: Whether to track cache statistics
        """
        self.max_size = max_size
        self.enable_stats = enable_stats
        
        # Thread-safe cache using OrderedDict for LRU behavior
        self._cache: OrderedDict[str, SegmentProperties] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics tracking
        self.stats = CacheStats() if enable_stats else None
        
        # Cache metadata
        self._creation_time = time.time()
        self._last_cleanup = time.time()
        
    def get_segment_key(self, segment: RouteSegment) -> str:
        """Generate unique cache key for a route segment
        
        Args:
            segment: Route segment to generate key for
            
        Returns:
            Unique string identifier for the segment
        """
        if not segment or not hasattr(segment, 'path_nodes') or not segment.path_nodes:
            return "empty_segment"
        
        # Create key from path nodes and basic properties
        path_str = "-".join(str(node) for node in segment.path_nodes)
        direction = getattr(segment, 'direction', 0)
        
        # Include segment properties that affect caching
        key_data = f"{path_str}_{direction}_{len(segment.path_nodes)}"
        
        # Create hash for long paths to keep keys manageable
        if len(key_data) > 100:
            return hashlib.md5(key_data.encode()).hexdigest()
        
        return key_data
    
    def calculate_segment_properties(self, segment: RouteSegment, graph: nx.Graph) -> SegmentProperties:
        """Calculate all properties for a route segment
        
        Args:
            segment: Route segment to analyze
            graph: Network graph for calculations
            
        Returns:
            SegmentProperties with all calculated values
        """
        start_time = time.time()
        
        if not segment or not hasattr(segment, 'path_nodes') or not segment.path_nodes or len(segment.path_nodes) < 2:
            return SegmentProperties(
                distance_km=0.0, elevation_gain_m=0.0, elevation_loss_m=0.0,
                net_elevation_m=0.0, max_elevation_m=0.0, min_elevation_m=0.0,
                avg_grade_percent=0.0, max_grade_percent=0.0, num_nodes=0,
                connectivity_score=0.0
            )
        
        # Distance calculation
        distance_km = 0.0
        elevations = []
        grades = []
        
        for i in range(len(segment.path_nodes) - 1):
            node1, node2 = segment.path_nodes[i], segment.path_nodes[i + 1]
            
            # Get edge distance
            if graph.has_edge(node1, node2):
                edge_data = graph[node1][node2]
                edge_length = edge_data.get('length', 0)
                distance_km += edge_length / 1000.0  # Convert to km
            
            # Collect elevations
            if node1 in graph.nodes:
                elev = graph.nodes[node1].get('elevation', 0)
                elevations.append(elev)
        
        # Add last node elevation
        if segment.path_nodes[-1] in graph.nodes:
            elevations.append(graph.nodes[segment.path_nodes[-1]].get('elevation', 0))
        
        # Elevation analysis
        if len(elevations) >= 2:
            elevation_gain = sum(max(0, elevations[i+1] - elevations[i]) 
                               for i in range(len(elevations) - 1))
            elevation_loss = sum(max(0, elevations[i] - elevations[i+1]) 
                               for i in range(len(elevations) - 1))
            net_elevation = elevations[-1] - elevations[0]
            max_elevation = max(elevations)
            min_elevation = min(elevations)
            
            # Grade calculations
            for i in range(len(elevations) - 1):
                elev_change = abs(elevations[i+1] - elevations[i])
                if i < len(segment.path_nodes) - 1:
                    node1, node2 = segment.path_nodes[i], segment.path_nodes[i + 1]
                    if graph.has_edge(node1, node2):
                        edge_length = graph[node1][node2].get('length', 1)
                        if edge_length > 0:
                            grade = (elev_change / edge_length) * 100
                            grades.append(grade)
            
            avg_grade = sum(grades) / len(grades) if grades else 0.0
            max_grade = max(grades) if grades else 0.0
        else:
            elevation_gain = elevation_loss = net_elevation = 0.0
            max_elevation = min_elevation = elevations[0] if elevations else 0.0
            avg_grade = max_grade = 0.0
        
        # Connectivity score (simple measure based on node connectivity)
        connectivity_score = 0.0
        for node in segment.path_nodes:
            if node in graph.nodes:
                connectivity_score += len(list(graph.neighbors(node)))
        connectivity_score = connectivity_score / len(segment.path_nodes) if segment.path_nodes else 0.0
        
        calculation_time = time.time() - start_time
        
        # Update statistics for actual calculations
        if self.stats:
            self.stats.update_calculation_time(calculation_time)
        
        return SegmentProperties(
            distance_km=distance_km,
            elevation_gain_m=elevation_gain,
            elevation_loss_m=elevation_loss,
            net_elevation_m=net_elevation,
            max_elevation_m=max_elevation,
            min_elevation_m=min_elevation,
            avg_grade_percent=avg_grade,
            max_grade_percent=max_grade,
            num_nodes=len(segment.path_nodes),
            connectivity_score=connectivity_score
        )
    
    def get_segment_properties(self, segment: RouteSegment, graph: nx.Graph) -> SegmentProperties:
        """Get cached segment properties or calculate if not cached
        
        Args:
            segment: Route segment to get properties for
            graph: Network graph for calculations
            
        Returns:
            SegmentProperties for the segment
        """
        if self.stats:
            self.stats.total_requests += 1
        
        segment_key = self.get_segment_key(segment)
        
        with self._lock:
            # Check cache first
            if segment_key in self._cache:
                # Move to end (LRU)
                properties = self._cache.pop(segment_key)
                self._cache[segment_key] = properties
                
                if self.stats:
                    self.stats.record_cache_hit()
                
                return properties
            
            # Cache miss - calculate properties
            if self.stats:
                self.stats.cache_misses += 1
            
            properties = self.calculate_segment_properties(segment, graph)
            
            # Add to cache
            self._cache[segment_key] = properties
            
            # Enforce size limit (LRU eviction)
            if len(self._cache) > self.max_size:
                # Remove oldest entries
                excess = len(self._cache) - self.max_size
                for _ in range(excess):
                    self._cache.popitem(last=False)
            
            return properties
    
    def get_chromosome_properties(self, chromosome, graph: nx.Graph) -> Dict[str, float]:
        """Get aggregated properties for entire chromosome using cached segments
        
        Args:
            chromosome: Route chromosome with segments
            graph: Network graph for calculations
            
        Returns:
            Dictionary with aggregated chromosome properties
        """
        if not chromosome or not chromosome.segments:
            return {
                'total_distance_km': 0.0,
                'total_elevation_gain_m': 0.0,
                'total_elevation_loss_m': 0.0,
                'net_elevation_change_m': 0.0,
                'max_elevation_m': 0.0,
                'min_elevation_m': 0.0,
                'avg_grade_percent': 0.0,
                'max_grade_percent': 0.0,
                'avg_connectivity_score': 0.0,
                'total_nodes': 0
            }
        
        # Get properties for all segments
        segment_properties = []
        for segment in chromosome.segments:
            props = self.get_segment_properties(segment, graph)
            segment_properties.append(props)
        
        # Aggregate properties
        total_distance = sum(props.distance_km for props in segment_properties)
        total_elevation_gain = sum(props.elevation_gain_m for props in segment_properties)
        total_elevation_loss = sum(props.elevation_loss_m for props in segment_properties)
        net_elevation_change = sum(props.net_elevation_m for props in segment_properties)
        
        max_elevation = max((props.max_elevation_m for props in segment_properties), default=0.0)
        min_elevation = min((props.min_elevation_m for props in segment_properties), default=0.0)
        
        # Weighted averages
        total_nodes = sum(props.num_nodes for props in segment_properties)
        if total_nodes > 0:
            avg_grade = sum(props.avg_grade_percent * props.num_nodes for props in segment_properties) / total_nodes
            avg_connectivity = sum(props.connectivity_score * props.num_nodes for props in segment_properties) / total_nodes
        else:
            avg_grade = avg_connectivity = 0.0
        
        max_grade = max((props.max_grade_percent for props in segment_properties), default=0.0)
        
        return {
            'total_distance_km': total_distance,
            'total_elevation_gain_m': total_elevation_gain,
            'total_elevation_loss_m': total_elevation_loss,
            'net_elevation_change_m': net_elevation_change,
            'max_elevation_m': max_elevation,
            'min_elevation_m': min_elevation,
            'avg_grade_percent': avg_grade,
            'max_grade_percent': max_grade,
            'avg_connectivity_score': avg_connectivity,
            'total_nodes': total_nodes
        }
    
    def clear(self):
        """Clear all cached segments"""
        with self._lock:
            self._cache.clear()
            if self.stats:
                # Reset stats but preserve cumulative data
                self.stats.total_requests = 0
                self.stats.cache_hits = 0
                self.stats.cache_misses = 0
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cache state and performance"""
        with self._lock:
            cache_size = len(self._cache)
            
        info = {
            'cache_size': cache_size,
            'max_size': self.max_size,
            'memory_usage_percent': (cache_size / self.max_size) * 100 if self.max_size > 0 else 0,
            'uptime_seconds': time.time() - self._creation_time,
        }
        
        if self.stats:
            info.update({
                'total_requests': self.stats.total_requests,
                'cache_hits': self.stats.cache_hits,
                'cache_misses': self.stats.cache_misses,
                'hit_rate_percent': self.stats.hit_rate,
                'calculations_saved': self.stats.cache_hits,
                'total_time_saved_seconds': self.stats.total_calculation_time_saved,
                'avg_time_saved_per_hit_ms': self.stats.time_saved_per_hit * 1000,
                'avg_calculation_time_ms': self.stats.avg_calculation_time * 1000,
                'total_calculations_performed': self.stats.calculation_count
            })
        
        return info
    
    def print_cache_stats(self):
        """Print detailed cache statistics"""
        info = self.get_cache_info()
        
        print("\n" + "="*50)
        print("GA SEGMENT CACHE STATISTICS")
        print("="*50)
        print(f"Cache Size: {info['cache_size']:,} / {info['max_size']:,} segments")
        print(f"Memory Usage: {info['memory_usage_percent']:.1f}%")
        print(f"Uptime: {info['uptime_seconds']:.1f} seconds")
        
        if self.stats:
            print(f"\nPerformance:")
            print(f"  Total Requests: {info['total_requests']:,}")
            print(f"  Cache Hits: {info['cache_hits']:,}")
            print(f"  Cache Misses: {info['cache_misses']:,}")
            print(f"  Hit Rate: {info['hit_rate_percent']:.1f}%")
            print(f"  Calculations Saved: {info['calculations_saved']:,}")
            print(f"  Total Time Saved: {info['total_time_saved_seconds']:.3f}s")
            print(f"  Avg Calculation Time: {info['avg_calculation_time_ms']:.2f}ms")
            print(f"  Total Calculations: {info['total_calculations_performed']:,}")
        
        print("="*50)


# Global cache instance for easy access
_global_segment_cache: Optional[GASegmentCache] = None


def get_global_segment_cache() -> GASegmentCache:
    """Get or create the global segment cache instance"""
    global _global_segment_cache
    if _global_segment_cache is None:
        _global_segment_cache = GASegmentCache()
    return _global_segment_cache


def set_global_segment_cache(cache: GASegmentCache):
    """Set the global segment cache instance"""
    global _global_segment_cache
    _global_segment_cache = cache


def clear_global_segment_cache():
    """Clear the global segment cache"""
    global _global_segment_cache
    if _global_segment_cache:
        _global_segment_cache.clear()