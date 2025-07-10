#!/usr/bin/env python3
"""
OSMnx Performance Profiler
Monitor and optimize OSMnx operations for route planning
"""

import time
import psutil
import functools
import logging
from typing import Dict, Any, Callable, List
from dataclasses import dataclass
from contextlib import contextmanager
import networkx as nx

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for OSMnx operations"""
    operation: str
    duration: float
    memory_peak_mb: float
    memory_delta_mb: float
    network_nodes: int
    network_edges: int
    cache_hit: bool
    additional_info: Dict[str, Any]

class OSMnxProfiler:
    """Profiler for OSMnx operations"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.process = psutil.Process()
    
    @contextmanager
    def profile_operation(self, operation_name: str, **kwargs):
        """Context manager to profile an operation"""
        # Get initial memory
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = initial_memory
        start_time = time.time()
        
        # Monitor memory during operation
        def memory_monitor():
            nonlocal peak_memory
            current_memory = self.process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)
        
        try:
            yield memory_monitor
            
        finally:
            # Calculate metrics
            end_time = time.time()
            final_memory = self.process.memory_info().rss / 1024 / 1024
            
            metrics = PerformanceMetrics(
                operation=operation_name,
                duration=end_time - start_time,
                memory_peak_mb=peak_memory,
                memory_delta_mb=final_memory - initial_memory,
                network_nodes=kwargs.get('nodes', 0),
                network_edges=kwargs.get('edges', 0),
                cache_hit=kwargs.get('cache_hit', False),
                additional_info=kwargs
            )
            
            self.metrics.append(metrics)
            logger.info(f"{operation_name}: {metrics.duration:.2f}s, "
                       f"Peak memory: {metrics.memory_peak_mb:.1f}MB, "
                       f"Delta: {metrics.memory_delta_mb:+.1f}MB")
    
    def profile_function(self, operation_name: str):
        """Decorator to profile a function"""
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile_operation(operation_name) as monitor:
                    # Periodically check memory during long operations
                    result = func(*args, **kwargs)
                    monitor()
                    
                    # Add network info if result is a graph
                    if isinstance(result, nx.Graph):
                        self.metrics[-1].network_nodes = len(result.nodes)
                        self.metrics[-1].network_edges = len(result.edges)
                    
                    return result
            return wrapper
        return decorator
    
    def get_operation_stats(self, operation_name: str = None) -> Dict[str, Any]:
        """Get statistics for operations"""
        relevant_metrics = self.metrics
        if operation_name:
            relevant_metrics = [m for m in self.metrics if m.operation == operation_name]
        
        if not relevant_metrics:
            return {}
        
        durations = [m.duration for m in relevant_metrics]
        memory_peaks = [m.memory_peak_mb for m in relevant_metrics]
        memory_deltas = [m.memory_delta_mb for m in relevant_metrics]
        
        return {
            'count': len(relevant_metrics),
            'total_duration': sum(durations),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'avg_memory_peak': sum(memory_peaks) / len(memory_peaks),
            'max_memory_peak': max(memory_peaks),
            'avg_memory_delta': sum(memory_deltas) / len(memory_deltas),
            'cache_hit_rate': sum(1 for m in relevant_metrics if m.cache_hit) / len(relevant_metrics)
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive performance report"""
        if not self.metrics:
            return "No performance data collected"
        
        report = ["OSMnx Performance Report", "=" * 50]
        
        # Overall statistics
        overall_stats = self.get_operation_stats()
        report.extend([
            f"Total Operations: {overall_stats['count']}",
            f"Total Time: {overall_stats['total_duration']:.2f}s",
            f"Average Duration: {overall_stats['avg_duration']:.2f}s",
            f"Cache Hit Rate: {overall_stats['cache_hit_rate']:.1%}",
            f"Peak Memory Usage: {overall_stats['max_memory_peak']:.1f}MB",
            ""
        ])
        
        # Per-operation breakdown
        operations = set(m.operation for m in self.metrics)
        for operation in sorted(operations):
            stats = self.get_operation_stats(operation)
            report.extend([
                f"{operation}:",
                f"  Count: {stats['count']}",
                f"  Avg Duration: {stats['avg_duration']:.2f}s",
                f"  Range: {stats['min_duration']:.2f}s - {stats['max_duration']:.2f}s",
                f"  Avg Memory Peak: {stats['avg_memory_peak']:.1f}MB",
                f"  Cache Hit Rate: {stats['cache_hit_rate']:.1%}",
                ""
            ])
        
        # Performance recommendations
        report.extend(self._generate_recommendations())
        
        return "\n".join(report)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations based on metrics"""
        recommendations = ["Performance Recommendations:", "-" * 30]
        
        # Analyze cache performance
        cache_stats = self.get_operation_stats()
        if cache_stats.get('cache_hit_rate', 0) < 0.5:
            recommendations.append("• Low cache hit rate - consider pre-generating common networks")
        
        # Analyze memory usage
        if cache_stats.get('max_memory_peak', 0) > 4000:  # 4GB
            recommendations.append("• High memory usage detected - consider processing smaller areas")
        
        # Analyze duration patterns
        download_metrics = [m for m in self.metrics if 'download' in m.operation.lower()]
        if download_metrics:
            avg_download_time = sum(m.duration for m in download_metrics) / len(download_metrics)
            if avg_download_time > 60:  # 1 minute
                recommendations.append("• Slow network downloads - consider using smaller radius or server-side filtering")
        
        # Analyze elevation processing
        elevation_metrics = [m for m in self.metrics if 'elevation' in m.operation.lower()]
        if elevation_metrics:
            avg_elevation_time = sum(m.duration for m in elevation_metrics) / len(elevation_metrics)
            total_nodes = sum(m.network_nodes for m in elevation_metrics)
            if total_nodes > 0:
                time_per_node = avg_elevation_time * 1000 / (total_nodes / len(elevation_metrics))
                if time_per_node > 10:  # 10ms per node
                    recommendations.append("• Slow elevation processing - consider batching or using raster data")
        
        if len(recommendations) == 2:  # Only headers
            recommendations.append("• Performance looks good! No specific recommendations.")
        
        return recommendations
    
    def export_metrics(self, filename: str):
        """Export metrics to JSON file"""
        import json
        
        metrics_data = []
        for metric in self.metrics:
            metrics_data.append({
                'operation': metric.operation,
                'duration': metric.duration,
                'memory_peak_mb': metric.memory_peak_mb,
                'memory_delta_mb': metric.memory_delta_mb,
                'network_nodes': metric.network_nodes,
                'network_edges': metric.network_edges,
                'cache_hit': metric.cache_hit,
                'additional_info': metric.additional_info
            })
        
        with open(filename, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Exported {len(metrics_data)} metrics to {filename}")


# Global profiler instance
profiler = OSMnxProfiler()

# Convenience decorators
def profile_download(func):
    """Profile network download operations"""
    return profiler.profile_function("network_download")(func)

def profile_elevation(func):
    """Profile elevation processing operations"""
    return profiler.profile_function("elevation_processing")(func)

def profile_routing(func):
    """Profile routing operations"""
    return profiler.profile_function("routing_calculation")(func)

def profile_caching(func):
    """Profile caching operations"""
    return profiler.profile_function("cache_operation")(func)


# Example integration with existing code
class ProfiledNetworkManager:
    """Example of how to integrate profiling with existing NetworkManager"""
    
    def __init__(self):
        self.profiler = OSMnxProfiler()
    
    @profile_download
    def download_network(self, center_point, radius_m, network_type):
        """Profiled network download"""
        import osmnx as ox
        return ox.graph_from_point(center_point, dist=radius_m, network_type=network_type)
    
    @profile_elevation
    def add_elevation_data(self, graph):
        """Profiled elevation processing"""
        import osmnx as ox
        return ox.elevation.add_node_elevations_raster(graph, 'srtm_20_05.tif')
    
    @profile_caching
    def save_to_cache(self, graph, cache_path):
        """Profiled cache save"""
        import pickle
        with open(cache_path, 'wb') as f:
            pickle.dump(graph, f)
    
    @profile_caching  
    def load_from_cache(self, cache_path):
        """Profiled cache load"""
        import pickle
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    def generate_performance_report(self):
        """Generate performance report"""
        return self.profiler.generate_report()


if __name__ == "__main__":
    # Example usage
    manager = ProfiledNetworkManager()
    
    # Simulate some operations
    print("Simulating network operations...")
    
    # This would profile actual operations
    # graph = manager.download_network((37.1299, -80.4094), 1000, 'all')
    # graph = manager.add_elevation_data(graph)
    
    # Generate report
    print(manager.generate_performance_report())