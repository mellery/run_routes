#!/usr/bin/env python3
"""
GA Memory Usage Optimizer
Memory optimization and monitoring for genetic algorithm operations
"""

import gc
import os
import sys
import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import weakref
import pickle

from ga_chromosome import RouteChromosome


@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: float
    total_mb: float
    available_mb: float
    used_mb: float
    percent_used: float
    process_memory_mb: float
    gc_objects: int
    
    
@dataclass
class MemoryStats:
    """Memory usage statistics"""
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    current_memory_mb: float = 0.0
    total_gc_collections: int = 0
    memory_warnings: int = 0
    snapshots: List[MemorySnapshot] = field(default_factory=list)


class MemoryPool:
    """Object pool for reusing chromosome and segment objects"""
    
    def __init__(self, max_size: int = 1000):
        """Initialize memory pool
        
        Args:
            max_size: Maximum number of objects to pool
        """
        self.max_size = max_size
        self.chromosome_pool = deque()
        self.segment_pool = deque()
        self.lock = threading.RLock()
        
        # Statistics
        self.pool_hits = 0
        self.pool_misses = 0
        self.objects_created = 0
        self.objects_recycled = 0
    
    def get_chromosome(self) -> Optional[RouteChromosome]:
        """Get a chromosome from the pool
        
        Returns:
            Reusable chromosome or None if pool is empty
        """
        with self.lock:
            if self.chromosome_pool:
                chromosome = self.chromosome_pool.popleft()
                # Reset chromosome state
                chromosome.segments.clear()
                chromosome.fitness = None
                chromosome.is_valid = True
                chromosome._invalidate_cache()
                
                self.pool_hits += 1
                return chromosome
            else:
                self.pool_misses += 1
                return None
    
    def return_chromosome(self, chromosome: RouteChromosome) -> None:
        """Return a chromosome to the pool
        
        Args:
            chromosome: Chromosome to return to pool
        """
        with self.lock:
            if len(self.chromosome_pool) < self.max_size:
                # Clear sensitive references
                chromosome.segments.clear()
                chromosome.fitness = None
                chromosome.parent_ids = None
                chromosome.creation_method = None
                
                self.chromosome_pool.append(chromosome)
                self.objects_recycled += 1
    
    def get_pool_stats(self) -> Dict[str, int]:
        """Get pool statistics
        
        Returns:
            Dictionary with pool performance metrics
        """
        with self.lock:
            hit_rate = self.pool_hits / max(self.pool_hits + self.pool_misses, 1)
            
            return {
                'chromosome_pool_size': len(self.chromosome_pool),
                'segment_pool_size': len(self.segment_pool),
                'pool_hits': self.pool_hits,
                'pool_misses': self.pool_misses,
                'hit_rate': hit_rate,
                'objects_created': self.objects_created,
                'objects_recycled': self.objects_recycled
            }
    
    def clear_pools(self):
        """Clear all object pools"""
        with self.lock:
            self.chromosome_pool.clear()
            self.segment_pool.clear()


class GAMemoryOptimizer:
    """Memory optimization and monitoring system for GA operations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize memory optimizer
        
        Args:
            config: Configuration options
        """
        default_config = {
            'memory_limit_mb': 2048,          # Memory limit in MB
            'warning_threshold': 0.85,        # Warning at 85% of limit
            'gc_threshold': 0.90,            # Force GC at 90% of limit
            'monitoring_interval': 5.0,       # Monitor every 5 seconds
            'enable_object_pool': True,       # Enable object pooling
            'pool_size': 1000,               # Object pool size
            'enable_weak_refs': True,        # Use weak references where possible
            'snapshot_history': 100          # Keep 100 memory snapshots
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Memory statistics
        self.stats = MemoryStats()
        
        # Object pool
        if self.config['enable_object_pool']:
            self.object_pool = MemoryPool(self.config['pool_size'])
        else:
            self.object_pool = None
        
        # Memory monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Callbacks for memory events
        self.warning_callback = None
        self.gc_callback = None
        
        # Weak reference tracking
        if self.config['enable_weak_refs']:
            self.tracked_objects = weakref.WeakSet()
        
        # Get initial memory reading
        self._update_memory_stats()
    
    def start_monitoring(self):
        """Start memory monitoring thread"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self.monitor_thread.start()
        print(f"🔍 Memory monitoring started (limit: {self.config['memory_limit_mb']}MB)")
    
    def stop_monitoring(self):
        """Stop memory monitoring thread"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        print("🔍 Memory monitoring stopped")
    
    def _monitor_memory(self):
        """Memory monitoring loop (runs in separate thread)"""
        while self.monitoring_active:
            try:
                self._update_memory_stats()
                
                current_mb = self.stats.current_memory_mb
                limit_mb = self.config['memory_limit_mb']
                usage_ratio = current_mb / limit_mb
                
                # Check for warnings
                if usage_ratio >= self.config['warning_threshold']:
                    self.stats.memory_warnings += 1
                    
                    if self.warning_callback:
                        self.warning_callback(current_mb, limit_mb, usage_ratio)
                    
                    print(f"⚠️ Memory warning: {current_mb:.1f}MB / {limit_mb}MB ({usage_ratio:.1%})")
                
                # Check for forced garbage collection
                if usage_ratio >= self.config['gc_threshold']:
                    self._force_garbage_collection()
                    
                    if self.gc_callback:
                        self.gc_callback(current_mb, limit_mb)
                
                time.sleep(self.config['monitoring_interval'])
                
            except Exception as e:
                print(f"Memory monitoring error: {e}")
                time.sleep(self.config['monitoring_interval'])
    
    def _update_memory_stats(self):
        """Update memory statistics"""
        try:
            # System memory
            memory = psutil.virtual_memory()
            
            # Process memory
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info()
            
            # Garbage collection info
            gc_objects = len(gc.get_objects())
            
            # Create snapshot
            snapshot = MemorySnapshot(
                timestamp=time.time(),
                total_mb=memory.total / (1024 * 1024),
                available_mb=memory.available / (1024 * 1024),
                used_mb=memory.used / (1024 * 1024),
                percent_used=memory.percent,
                process_memory_mb=process_memory.rss / (1024 * 1024),
                gc_objects=gc_objects
            )
            
            # Update statistics
            self.stats.current_memory_mb = snapshot.process_memory_mb
            
            if snapshot.process_memory_mb > self.stats.peak_memory_mb:
                self.stats.peak_memory_mb = snapshot.process_memory_mb
            
            # Add to snapshots (with size limit)
            self.stats.snapshots.append(snapshot)
            if len(self.stats.snapshots) > self.config['snapshot_history']:
                self.stats.snapshots.pop(0)
            
            # Calculate average memory
            if self.stats.snapshots:
                total_memory = sum(s.process_memory_mb for s in self.stats.snapshots)
                self.stats.avg_memory_mb = total_memory / len(self.stats.snapshots)
            
        except Exception as e:
            print(f"Error updating memory stats: {e}")
    
    def _force_garbage_collection(self):
        """Force garbage collection and cleanup"""
        print("🗑️ Forcing garbage collection...")
        
        # Standard garbage collection
        collected = gc.collect()
        self.stats.total_gc_collections += 1
        
        # Additional cleanup for cycles
        gc.collect()
        
        print(f"🗑️ Garbage collection completed: {collected} objects collected")
    
    def optimize_population_memory(self, population: List[RouteChromosome]) -> List[RouteChromosome]:
        """Optimize memory usage for a population
        
        Args:
            population: Population to optimize
            
        Returns:
            Memory-optimized population
        """
        optimized_population = []
        
        for chromosome in population:
            # Create optimized copy
            optimized_chromosome = self._optimize_chromosome_memory(chromosome)
            optimized_population.append(optimized_chromosome)
        
        return optimized_population
    
    def _optimize_chromosome_memory(self, chromosome: RouteChromosome) -> RouteChromosome:
        """Optimize memory usage for a single chromosome
        
        Args:
            chromosome: Chromosome to optimize
            
        Returns:
            Memory-optimized chromosome
        """
        # Try to get from object pool
        if self.object_pool:
            optimized = self.object_pool.get_chromosome()
            if optimized:
                # Copy essential data only
                optimized.segments = [seg.copy() for seg in chromosome.segments]
                optimized.fitness = chromosome.fitness
                optimized.is_valid = chromosome.is_valid
                return optimized
        
        # Create new optimized chromosome
        optimized = RouteChromosome()
        optimized.segments = [seg.copy() for seg in chromosome.segments]
        optimized.fitness = chromosome.fitness
        optimized.is_valid = chromosome.is_valid
        
        # Track with weak reference if enabled
        if self.config['enable_weak_refs']:
            self.tracked_objects.add(optimized)
        
        return optimized
    
    def cleanup_population(self, population: List[RouteChromosome]):
        """Clean up population memory (return to pool, clear references)
        
        Args:
            population: Population to clean up
        """
        if self.object_pool:
            for chromosome in population:
                self.object_pool.return_chromosome(chromosome)
        
        # Clear the population list
        population.clear()
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory usage report
        
        Returns:
            Dictionary with detailed memory information
        """
        self._update_memory_stats()
        
        report = {
            'current_memory_mb': self.stats.current_memory_mb,
            'peak_memory_mb': self.stats.peak_memory_mb,
            'avg_memory_mb': self.stats.avg_memory_mb,
            'memory_limit_mb': self.config['memory_limit_mb'],
            'memory_usage_percent': (self.stats.current_memory_mb / self.config['memory_limit_mb']) * 100,
            'total_gc_collections': self.stats.total_gc_collections,
            'memory_warnings': self.stats.memory_warnings,
            'gc_objects': len(gc.get_objects()) if self.stats.snapshots else 0
        }
        
        # Add object pool stats if available
        if self.object_pool:
            report['object_pool'] = self.object_pool.get_pool_stats()
        
        # Add recent memory trend
        if len(self.stats.snapshots) >= 2:
            recent_snapshots = self.stats.snapshots[-10:]  # Last 10 snapshots
            memory_trend = []
            
            for i in range(1, len(recent_snapshots)):
                prev_mem = recent_snapshots[i-1].process_memory_mb
                curr_mem = recent_snapshots[i].process_memory_mb
                trend = curr_mem - prev_mem
                memory_trend.append(trend)
            
            if memory_trend:
                report['memory_trend_mb'] = sum(memory_trend) / len(memory_trend)
                report['trend_direction'] = 'increasing' if report['memory_trend_mb'] > 0 else 'decreasing'
        
        # Add tracked objects count if using weak references
        if self.config['enable_weak_refs']:
            report['tracked_objects'] = len(self.tracked_objects)
        
        return report
    
    def optimize_for_large_populations(self, population_size: int) -> Dict[str, Any]:
        """Optimize settings for large population sizes
        
        Args:
            population_size: Expected population size
            
        Returns:
            Dictionary with optimization recommendations
        """
        # Estimate memory usage
        estimated_memory_per_chromosome = 10  # KB per chromosome (rough estimate)
        estimated_total_mb = (population_size * estimated_memory_per_chromosome) / 1024
        
        recommendations = {
            'estimated_memory_mb': estimated_total_mb,
            'recommended_limit_mb': max(self.config['memory_limit_mb'], estimated_total_mb * 2),
            'enable_object_pool': population_size > 100,
            'pool_size': min(population_size * 2, 5000),
            'enable_gc_optimization': population_size > 500,
            'recommended_batch_size': max(10, population_size // 20)
        }
        
        # Auto-apply recommendations if memory usage is high
        if estimated_total_mb > self.config['memory_limit_mb'] * 0.5:
            print(f"🔧 Auto-optimizing for large population ({population_size} chromosomes)")
            
            if not self.object_pool and recommendations['enable_object_pool']:
                self.object_pool = MemoryPool(recommendations['pool_size'])
                print(f"🔧 Enabled object pool (size: {recommendations['pool_size']})")
            
            if recommendations['enable_gc_optimization']:
                # Adjust GC thresholds for better performance
                import gc
                gc.set_threshold(700, 10, 10)  # More aggressive GC for large populations
                print("🔧 Optimized garbage collection thresholds")
        
        return recommendations
    
    def set_warning_callback(self, callback: Callable[[float, float, float], None]):
        """Set callback for memory warnings
        
        Args:
            callback: Function called with (current_mb, limit_mb, usage_ratio)
        """
        self.warning_callback = callback
    
    def set_gc_callback(self, callback: Callable[[float, float], None]):
        """Set callback for garbage collection events
        
        Args:
            callback: Function called with (current_mb, limit_mb)
        """
        self.gc_callback = callback
    
    def save_memory_profile(self, filename: str) -> str:
        """Save memory profile to file
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        profile_data = {
            'config': self.config,
            'stats': self.stats,
            'memory_report': self.get_memory_report(),
            'timestamp': time.time()
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(profile_data, f)
        
        return filename
    
    def cleanup(self):
        """Clean up memory optimizer resources"""
        self.stop_monitoring()
        
        if self.object_pool:
            self.object_pool.clear_pools()
        
        if self.config['enable_weak_refs']:
            self.tracked_objects.clear()
        
        # Force final garbage collection
        gc.collect()


def test_memory_optimizer():
    """Test function for memory optimizer"""
    print("Testing GA Memory Optimizer...")
    
    # Test configuration
    config = {
        'memory_limit_mb': 1024,
        'enable_object_pool': True,
        'pool_size': 100,
        'monitoring_interval': 1.0
    }
    
    optimizer = GAMemoryOptimizer(config)
    
    # Test memory report
    report = optimizer.get_memory_report()
    print(f"✅ Memory report: {report['current_memory_mb']:.1f}MB current")
    
    # Test population optimization
    from ga_chromosome import RouteSegment, RouteChromosome
    
    test_population = []
    for i in range(50):
        segment = RouteSegment(1, 2, [1, 2])
        chromosome = RouteChromosome([segment])
        test_population.append(chromosome)
    
    optimized_population = optimizer.optimize_population_memory(test_population)
    print(f"✅ Population optimization: {len(optimized_population)} chromosomes optimized")
    
    # Test object pool stats
    if optimizer.object_pool:
        pool_stats = optimizer.object_pool.get_pool_stats()
        print(f"✅ Object pool stats: {pool_stats['hit_rate']:.2f} hit rate")
    
    # Test large population optimization
    recommendations = optimizer.optimize_for_large_populations(1000)
    print(f"✅ Large population recommendations: {recommendations['estimated_memory_mb']:.1f}MB estimated")
    
    # Test cleanup
    optimizer.cleanup_population(optimized_population)
    print(f"✅ Population cleanup completed")
    
    # Cleanup
    optimizer.cleanup()
    
    print("✅ All memory optimizer tests completed")


if __name__ == "__main__":
    test_memory_optimizer()