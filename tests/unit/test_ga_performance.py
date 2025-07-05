#!/usr/bin/env python3
"""
Unit tests for GA Performance Optimizations
Tests for caching, parallel evaluation, distance optimization, and memory management
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import time
import threading
import numpy as np
import networkx as nx

# Performance optimization modules
from ga_performance_cache import GAPerformanceCache, LRUCache, CacheStats
from ga_parallel_evaluator import GAParallelEvaluator, ParallelConfig, WorkerProcess
from ga_distance_optimizer import GADistanceOptimizer, OptimizedDistanceMatrix, FastHaversine
from ga_memory_optimizer import GAMemoryOptimizer, MemoryPool
from ga_performance_benchmark import GAPerformanceBenchmark, BenchmarkResult

# Core GA modules
from ga_chromosome import RouteChromosome, RouteSegment


class TestGAPerformanceCache(unittest.TestCase):
    """Test GA performance caching system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cache = GAPerformanceCache({
            'enable_disk_cache': False,  # Disable for testing
            'segment_cache_size': 100,
            'distance_cache_size': 100
        })
        
        # Create test graph
        self.test_graph = nx.Graph()
        nodes = [(1, -80.4094, 37.1299, 100), (2, -80.4000, 37.1300, 110), (3, -80.4050, 37.1350, 105)]
        for node_id, x, y, elev in nodes:
            self.test_graph.add_node(node_id, x=x, y=y, elevation=elev)
        
        edges = [(1, 2, 100), (2, 3, 150), (3, 1, 200)]
        for n1, n2, length in edges:
            self.test_graph.add_edge(n1, n2, length=length)
    
    def test_lru_cache_basic_operations(self):
        """Test basic LRU cache operations"""
        cache = LRUCache(max_size=3)
        
        # Test put and get
        cache.put("key1", "value1")
        self.assertEqual(cache.get("key1"), "value1")
        
        # Test cache miss
        self.assertIsNone(cache.get("key2"))
        
        # Test size limit
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        cache.put("key4", "value4")  # Should evict key1
        
        self.assertIsNone(cache.get("key1"))  # Evicted
        self.assertEqual(cache.get("key4"), "value4")  # Most recent
        
        # Test statistics
        stats = cache.get_stats()
        self.assertEqual(stats.size, 3)
        self.assertGreater(stats.total_requests, 0)
    
    def test_lru_cache_thread_safety(self):
        """Test LRU cache thread safety"""
        cache = LRUCache(max_size=100)
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(50):
                    key = f"worker_{worker_id}_key_{i}"
                    value = f"worker_{worker_id}_value_{i}"
                    cache.put(key, value)
                    retrieved = cache.get(key)
                    results.append((key, retrieved == value))
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        self.assertGreater(len(results), 0)
        # Most results should be True (successful cache operations)
        success_rate = sum(1 for _, success in results if success) / len(results)
        self.assertGreater(success_rate, 0.8)
    
    def test_segment_caching(self):
        """Test segment caching functionality"""
        # Test segment retrieval and caching
        segment1 = self.cache.get_segment(1, 2, self.test_graph)
        self.assertIsNotNone(segment1)
        self.assertEqual(segment1.start_node, 1)
        self.assertEqual(segment1.end_node, 2)
        
        # Test cache hit
        segment2 = self.cache.get_segment(1, 2, self.test_graph)
        self.assertIsNotNone(segment2)
        
        # Test non-existent path
        self.test_graph.remove_edge(1, 2)
        segment3 = self.cache.get_segment(1, 2, self.test_graph)
        self.assertIsNone(segment3)
    
    def test_distance_caching(self):
        """Test distance caching functionality"""
        # Test distance retrieval and caching
        distance1 = self.cache.get_distance(1, 2, self.test_graph)
        self.assertIsNotNone(distance1)
        self.assertEqual(distance1, 100)
        
        # Test cache hit (should be faster)
        start_time = time.time()
        distance2 = self.cache.get_distance(1, 2, self.test_graph)
        cache_time = time.time() - start_time
        
        self.assertEqual(distance1, distance2)
        self.assertLess(cache_time, 0.01)  # Should be very fast
        
        # Test symmetric caching
        distance3 = self.cache.get_distance(2, 1, self.test_graph)
        self.assertEqual(distance1, distance3)
    
    def test_path_caching(self):
        """Test path caching functionality"""
        # Test path retrieval and caching
        path1 = self.cache.get_path(1, 3, self.test_graph)
        self.assertIsNotNone(path1)
        self.assertIn(1, path1)
        self.assertIn(3, path1)
        
        # Test cache hit
        path2 = self.cache.get_path(1, 3, self.test_graph)
        self.assertEqual(path1, path2)
        
        # Test invalid path
        isolated_graph = nx.Graph()
        isolated_graph.add_node(10)
        isolated_graph.add_node(11)
        path3 = self.cache.get_path(10, 11, isolated_graph)
        self.assertIsNone(path3)
    
    def test_fitness_component_caching(self):
        """Test fitness component caching"""
        # Test fitness caching
        components = {'distance': 0.8, 'elevation': 0.6, 'diversity': 0.7}
        self.cache.put_fitness_components("test_chromosome", components)
        
        retrieved = self.cache.get_fitness_components("test_chromosome")
        self.assertEqual(retrieved, components)
        
        # Test cache miss
        missing = self.cache.get_fitness_components("nonexistent")
        self.assertIsNone(missing)
    
    def test_distance_matrix_building(self):
        """Test distance matrix building"""
        nodes = [1, 2, 3]
        matrix = self.cache.build_distance_matrix(nodes, self.test_graph)
        
        self.assertEqual(matrix.shape, (3, 3))
        self.assertEqual(matrix[0, 0], 0.0)  # Diagonal should be 0
        self.assertGreater(matrix[0, 1], 0.0)  # Should have distances
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking"""
        # Add some data to caches
        for i in range(10):
            self.cache.get_distance(1, 2, self.test_graph)
            self.cache.get_segment(1, 2, self.test_graph)
        
        memory_usage = self.cache.get_memory_usage()
        self.assertIsInstance(memory_usage, dict)
        self.assertIn('total_mb', memory_usage)
        self.assertGreaterEqual(memory_usage['total_mb'], 0.0)
    
    def test_performance_statistics(self):
        """Test performance statistics tracking"""
        # Generate some cache activity
        for i in range(5):
            self.cache.get_distance(1, 2, self.test_graph)
            self.cache.get_segment(1, 2, self.test_graph)
        
        stats = self.cache.get_performance_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('uptime_seconds', stats)
        self.assertIn('segment_cache', stats)
        self.assertIn('distance_cache', stats)
        self.assertGreater(stats['uptime_seconds'], 0)


class TestGAParallelEvaluator(unittest.TestCase):
    """Test GA parallel evaluation system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = ParallelConfig(
            max_workers=2,
            chunk_size=5,
            use_processes=False,  # Use threads for testing
            timeout_seconds=5.0
        )
        self.evaluator = GAParallelEvaluator(self.config)
        
        # Create test population
        self.test_population = []
        for i in range(10):
            segment = RouteSegment(1, 2, [1, 2])
            segment.length = 1000.0 + i * 100
            segment.elevation_gain = 50.0
            chromosome = RouteChromosome([segment])
            self.test_population.append(chromosome)
    
    def test_evaluator_initialization(self):
        """Test parallel evaluator initialization"""
        self.assertEqual(self.evaluator.config.max_workers, 2)
        self.assertFalse(self.evaluator.config.use_processes)
        self.assertEqual(self.evaluator.total_evaluations, 0)
    
    def test_worker_process_functions(self):
        """Test worker process functions"""
        # Test single chromosome evaluation
        args = (self.test_population[0], "elevation", 3.0, 0)
        result = WorkerProcess.evaluate_chromosome_worker(args)
        
        self.assertIsInstance(result.fitness, float)
        self.assertGreaterEqual(result.fitness, 0.0)
        self.assertLessEqual(result.fitness, 1.0)
        self.assertTrue(result.success)
        self.assertEqual(result.task_id, 0)
        
        # Test batch evaluation
        batch_args = (self.test_population[:3], "elevation", 3.0, [0, 1, 2])
        batch_results = WorkerProcess.evaluate_batch_worker(batch_args)
        
        self.assertEqual(len(batch_results), 3)
        for result in batch_results:
            self.assertIsInstance(result.fitness, float)
            self.assertTrue(result.success)
    
    def test_sequential_evaluation_fallback(self):
        """Test sequential evaluation for small populations"""
        small_population = self.test_population[:3]
        fitness_scores = self.evaluator.evaluate_population_parallel(
            small_population, "elevation", 3.0
        )
        
        self.assertEqual(len(fitness_scores), 3)
        for fitness in fitness_scores:
            self.assertIsInstance(fitness, float)
            self.assertGreaterEqual(fitness, 0.0)
        
        # Should use sequential evaluation
        self.assertGreater(self.evaluator.sequential_evaluations, 0)
    
    def test_parallel_evaluation(self):
        """Test parallel evaluation with larger population"""
        fitness_scores = self.evaluator.evaluate_population_parallel(
            self.test_population, "elevation", 3.0
        )
        
        self.assertEqual(len(fitness_scores), len(self.test_population))
        for fitness in fitness_scores:
            self.assertIsInstance(fitness, float)
            self.assertGreaterEqual(fitness, 0.0)
            self.assertLessEqual(fitness, 1.0)
    
    def test_performance_statistics(self):
        """Test performance statistics tracking"""
        # Run some evaluations
        self.evaluator.evaluate_population_parallel(
            self.test_population, "elevation", 3.0
        )
        
        stats = self.evaluator.get_performance_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_evaluations', stats)
        self.assertIn('total_evaluation_time', stats)
        self.assertIn('parallel_ratio', stats)
        self.assertGreater(stats['total_evaluations'], 0)
    
    def test_error_handling(self):
        """Test error handling in parallel evaluation"""
        # Create invalid chromosome that might cause errors
        invalid_chromosome = RouteChromosome([])
        invalid_chromosome.segments = None  # This might cause errors
        
        population_with_errors = self.test_population + [invalid_chromosome]
        
        # Should handle errors gracefully
        fitness_scores = self.evaluator.evaluate_population_parallel(
            population_with_errors, "elevation", 3.0
        )
        
        self.assertEqual(len(fitness_scores), len(population_with_errors))
        # Last fitness should be 0.0 due to error
        self.assertEqual(fitness_scores[-1], 0.0)
    
    def test_benchmark_evaluation_methods(self):
        """Test benchmarking of different evaluation methods"""
        benchmark_results = self.evaluator.benchmark_evaluation_methods(
            self.test_population, "elevation", 3.0
        )
        
        self.assertIsInstance(benchmark_results, dict)
        self.assertIn('sequential_time', benchmark_results)
        self.assertIn('individual_parallel_time', benchmark_results)
        self.assertIn('individual_speedup', benchmark_results)
        self.assertGreater(benchmark_results['sequential_time'], 0)


class TestGADistanceOptimizer(unittest.TestCase):
    """Test GA distance calculation optimizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test graph with coordinates
        self.test_graph = nx.Graph()
        nodes = [
            (1, -80.4094, 37.1299, 100), (2, -80.4000, 37.1300, 110),
            (3, -80.4050, 37.1350, 105), (4, -80.4100, 37.1250, 120)
        ]
        
        for node_id, x, y, elev in nodes:
            self.test_graph.add_node(node_id, x=x, y=y, elevation=elev)
        
        edges = [(1, 2, 100), (2, 3, 150), (3, 4, 200), (4, 1, 180)]
        for n1, n2, length in edges:
            self.test_graph.add_edge(n1, n2, length=length)
        
        self.optimizer = GADistanceOptimizer(self.test_graph)
    
    def test_haversine_calculations(self):
        """Test Haversine distance calculations"""
        # Test single calculation
        lat1, lon1 = 37.1299, -80.4094
        lat2, lon2 = 37.1300, -80.4000
        
        distance = FastHaversine.single_haversine(lat1, lon1, lat2, lon2)
        self.assertIsInstance(distance, float)
        self.assertGreater(distance, 0)
        
        # Test vectorized calculation
        lat1_arr = np.array([lat1, lat1])
        lon1_arr = np.array([lon1, lon1])
        lat2_arr = np.array([lat2, lat2])
        lon2_arr = np.array([lon2, lon2])
        
        distances = FastHaversine.vectorized_haversine(lat1_arr, lon1_arr, lat2_arr, lon2_arr)
        self.assertEqual(len(distances), 2)
        self.assertAlmostEqual(distances[0], distance, places=3)
    
    def test_distance_matrix_operations(self):
        """Test optimized distance matrix operations"""
        matrix = self.optimizer.distance_matrix
        
        # Test Haversine distance caching
        dist1 = matrix.get_haversine_distance(1, 2)
        dist2 = matrix.get_haversine_distance(1, 2)  # Should be cached
        self.assertEqual(dist1, dist2)
        self.assertGreater(dist1, 0)
        
        # Test network distance caching
        net_dist1 = matrix.get_network_distance(1, 2)
        net_dist2 = matrix.get_network_distance(1, 2)  # Should be cached
        self.assertEqual(net_dist1, net_dist2)
        self.assertEqual(net_dist1, 100)  # Known edge weight
        
        # Test cache statistics
        self.assertGreater(matrix.stats.total_calculations, 0)
        self.assertGreater(matrix.stats.cache_hits, 0)
    
    def test_vectorized_distance_matrix(self):
        """Test vectorized distance matrix building"""
        nodes = [1, 2, 3, 4]
        matrix = self.optimizer.distance_matrix.build_distance_matrix_vectorized(nodes)
        
        self.assertEqual(matrix.shape, (4, 4))
        # Diagonal should be 0
        for i in range(4):
            self.assertEqual(matrix[i, i], 0.0)
        
        # Matrix should be symmetric
        for i in range(4):
            for j in range(4):
                self.assertEqual(matrix[i, j], matrix[j, i])
    
    def test_nearest_neighbors_search(self):
        """Test vectorized nearest neighbors search"""
        neighbors = self.optimizer.distance_matrix.get_nearest_neighbors_vectorized(
            1, [2, 3, 4], k=2
        )
        
        self.assertLessEqual(len(neighbors), 2)  # Requested k=2
        for node, distance in neighbors:
            self.assertIn(node, [2, 3, 4])
            self.assertGreater(distance, 0)
        
        # Results should be sorted by distance
        distances = [dist for _, dist in neighbors]
        self.assertEqual(distances, sorted(distances))
    
    def test_chromosome_distance_calculation(self):
        """Test optimized chromosome distance calculation"""
        # Create test chromosome
        segment1 = RouteSegment(1, 2, [1, 2])
        segment2 = RouteSegment(2, 3, [2, 3])
        chromosome = RouteChromosome([segment1, segment2])
        
        # Test distance calculation
        total_distance = self.optimizer.calculate_chromosome_distance_optimized(chromosome)
        self.assertGreater(total_distance, 0)
        self.assertIsInstance(total_distance, float)
        
        # Test caching
        cached_distance = self.optimizer.calculate_chromosome_distance_optimized(chromosome)
        self.assertEqual(total_distance, cached_distance)
    
    def test_optimization_statistics(self):
        """Test distance optimization statistics"""
        # Perform some operations to generate stats
        self.optimizer.distance_matrix.get_haversine_distance(1, 2)
        self.optimizer.distance_matrix.get_network_distance(1, 3)
        
        stats = self.optimizer.get_optimization_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_calculations', stats)
        self.assertIn('cache_hit_rate', stats)
        self.assertIn('haversine_cache_size', stats)
        self.assertGreater(stats['total_calculations'], 0)


class TestGAMemoryOptimizer(unittest.TestCase):
    """Test GA memory optimization system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'memory_limit_mb': 512,
            'enable_object_pool': True,
            'pool_size': 50,
            'monitoring_interval': 0.1  # Fast for testing
        }
        self.optimizer = GAMemoryOptimizer(self.config)
    
    def test_memory_optimizer_initialization(self):
        """Test memory optimizer initialization"""
        self.assertEqual(self.optimizer.config['memory_limit_mb'], 512)
        self.assertTrue(self.optimizer.config['enable_object_pool'])
        self.assertIsNotNone(self.optimizer.object_pool)
        self.assertFalse(self.optimizer.monitoring_active)
    
    def test_object_pool_operations(self):
        """Test object pool functionality"""
        pool = self.optimizer.object_pool
        
        # Test getting from empty pool
        chromosome = pool.get_chromosome()
        self.assertIsNone(chromosome)  # Pool is empty initially
        
        # Test returning and getting
        test_segment = RouteSegment(1, 2, [1, 2])
        test_chromosome = RouteChromosome([test_segment])
        
        pool.return_chromosome(test_chromosome)
        retrieved = pool.get_chromosome()
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(len(retrieved.segments), 0)  # Should be cleared
        self.assertIsNone(retrieved.fitness)  # Should be reset
        
        # Test pool statistics
        stats = pool.get_pool_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('hit_rate', stats)
        self.assertIn('chromosome_pool_size', stats)
    
    def test_memory_monitoring(self):
        """Test memory monitoring functionality"""
        # Start monitoring
        self.optimizer.start_monitoring()
        self.assertTrue(self.optimizer.monitoring_active)
        
        # Wait a short time for monitoring
        time.sleep(0.2)
        
        # Stop monitoring
        self.optimizer.stop_monitoring()
        self.assertFalse(self.optimizer.monitoring_active)
        
        # Should have some snapshots
        self.assertGreater(len(self.optimizer.stats.snapshots), 0)
    
    def test_population_memory_optimization(self):
        """Test population memory optimization"""
        # Create test population
        test_population = []
        for i in range(10):
            segment = RouteSegment(1, 2, [1, 2])
            chromosome = RouteChromosome([segment])
            test_population.append(chromosome)
        
        # Optimize population
        optimized_population = self.optimizer.optimize_population_memory(test_population)
        
        self.assertEqual(len(optimized_population), len(test_population))
        for chromosome in optimized_population:
            self.assertIsInstance(chromosome, RouteChromosome)
            self.assertGreater(len(chromosome.segments), 0)
        
        # Cleanup
        self.optimizer.cleanup_population(optimized_population)
        self.assertEqual(len(optimized_population), 0)
    
    def test_memory_report_generation(self):
        """Test memory report generation"""
        report = self.optimizer.get_memory_report()
        
        self.assertIsInstance(report, dict)
        self.assertIn('current_memory_mb', report)
        self.assertIn('peak_memory_mb', report)
        self.assertIn('memory_limit_mb', report)
        self.assertIn('memory_usage_percent', report)
        
        self.assertGreaterEqual(report['current_memory_mb'], 0.0)
        self.assertEqual(report['memory_limit_mb'], 512)
    
    def test_large_population_optimization(self):
        """Test optimization recommendations for large populations"""
        recommendations = self.optimizer.optimize_for_large_populations(1000)
        
        self.assertIsInstance(recommendations, dict)
        self.assertIn('estimated_memory_mb', recommendations)
        self.assertIn('recommended_limit_mb', recommendations)
        self.assertIn('enable_object_pool', recommendations)
        
        self.assertGreater(recommendations['estimated_memory_mb'], 0)
        self.assertTrue(recommendations['enable_object_pool'])
    
    def test_memory_callbacks(self):
        """Test memory warning and GC callbacks"""
        warning_called = []
        gc_called = []
        
        def warning_callback(current_mb, limit_mb, usage_ratio):
            warning_called.append((current_mb, limit_mb, usage_ratio))
        
        def gc_callback(current_mb, limit_mb):
            gc_called.append((current_mb, limit_mb))
        
        self.optimizer.set_warning_callback(warning_callback)
        self.optimizer.set_gc_callback(gc_callback)
        
        # Callbacks should be set
        self.assertEqual(self.optimizer.warning_callback, warning_callback)
        self.assertEqual(self.optimizer.gc_callback, gc_callback)


class TestGAPerformanceBenchmark(unittest.TestCase):
    """Test GA performance benchmarking system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.benchmark = GAPerformanceBenchmark("test_benchmark_output")
    
    def test_benchmark_initialization(self):
        """Test benchmark system initialization"""
        self.assertEqual(self.benchmark.output_dir, "test_benchmark_output")
        self.assertIn('small', self.benchmark.test_configs)
        self.assertIn('medium', self.benchmark.test_configs)
        self.assertIn('large', self.benchmark.test_configs)
    
    def test_benchmark_result_creation(self):
        """Test benchmark result data structure"""
        result = BenchmarkResult(
            test_name="test_benchmark",
            execution_time=1.5,
            memory_usage_mb=128.0,
            operations_per_second=100.0,
            error_rate=0.0,
            additional_metrics={'test_metric': 42}
        )
        
        self.assertEqual(result.test_name, "test_benchmark")
        self.assertEqual(result.execution_time, 1.5)
        self.assertEqual(result.memory_usage_mb, 128.0)
        self.assertEqual(result.operations_per_second, 100.0)
        self.assertEqual(result.error_rate, 0.0)
        self.assertEqual(result.additional_metrics['test_metric'], 42)
    
    def test_system_info_collection(self):
        """Test system information collection"""
        system_info = self.benchmark._get_system_info()
        
        self.assertIsInstance(system_info, dict)
        self.assertIn('platform', system_info)
        self.assertIn('cpu_count', system_info)
        self.assertIn('python_version', system_info)
        
        self.assertIsInstance(system_info['cpu_count'], int)
        self.assertGreater(system_info['cpu_count'], 0)
    
    def test_speedup_calculation(self):
        """Test speedup factor calculation"""
        # Create mock baseline and optimized results
        baseline_results = [
            BenchmarkResult("baseline_small", 2.0, 100.0, 50.0, 0.0),
            BenchmarkResult("baseline_medium", 4.0, 200.0, 25.0, 0.0)
        ]
        
        optimized_results = [
            BenchmarkResult("optimized_small", 1.0, 50.0, 100.0, 0.0),
            BenchmarkResult("optimized_medium", 2.0, 100.0, 50.0, 0.0)
        ]
        
        speedup_factors = self.benchmark._calculate_speedup_factors(
            baseline_results, optimized_results
        )
        
        self.assertIsInstance(speedup_factors, dict)
        # Should have 2x speedup for both execution and memory
        if 'small_execution' in speedup_factors:
            self.assertAlmostEqual(speedup_factors['small_execution'], 2.0, places=1)
        if 'small_memory' in speedup_factors:
            self.assertAlmostEqual(speedup_factors['small_memory'], 2.0, places=1)
    
    def test_recommendation_generation(self):
        """Test performance recommendation generation"""
        # Create mock results showing good performance
        baseline_results = [
            BenchmarkResult("baseline_test", 4.0, 200.0, 25.0, 0.0),
            BenchmarkResult("baseline_memory_test", 2.0, 150.0, 50.0, 0.0)
        ]
        
        optimized_results = [
            BenchmarkResult("optimized_test", 1.0, 100.0, 100.0, 0.0),
            BenchmarkResult("optimized_memory_test", 1.0, 75.0, 100.0, 0.0),
            BenchmarkResult("parallel_test", 0.5, 50.0, 200.0, 0.0)
        ]
        
        recommendations = self.benchmark._generate_recommendations(
            baseline_results, optimized_results
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Should mention speedup
        recommendation_text = ' '.join(recommendations)
        self.assertIn('speedup', recommendation_text.lower())
    
    def test_result_serialization(self):
        """Test benchmark result serialization"""
        result = BenchmarkResult(
            test_name="test_serialization",
            execution_time=1.0,
            memory_usage_mb=64.0,
            operations_per_second=100.0,
            error_rate=0.0
        )
        
        result_dict = self.benchmark._result_to_dict(result)
        
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['test_name'], "test_serialization")
        self.assertEqual(result_dict['execution_time'], 1.0)
        self.assertEqual(result_dict['memory_usage_mb'], 64.0)
        self.assertEqual(result_dict['operations_per_second'], 100.0)
        self.assertEqual(result_dict['error_rate'], 0.0)


if __name__ == '__main__':
    unittest.main()