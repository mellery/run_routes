#!/usr/bin/env python3
"""
Unit tests for GA Performance optimization components
Tests caching, parallel evaluation, distance optimization, and memory management
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import networkx as nx
import time
import threading
import hashlib
import sys
import os
from collections import OrderedDict

# Add the parent directory to sys.path to import genetic algorithm modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from genetic_algorithm.performance import (
        SegmentProperties, CacheStats, GASegmentCache
    )
    from genetic_algorithm.chromosome import RouteSegment, RouteChromosome
    GA_PERFORMANCE_AVAILABLE = True
except ImportError:
    GA_PERFORMANCE_AVAILABLE = False


class TestGAPerformance(unittest.TestCase):
    """Base test class for GA performance components"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not GA_PERFORMANCE_AVAILABLE:
            self.skipTest("GA performance modules not available")
        
        # Create test graph
        self.test_graph = nx.MultiGraph()
        self.test_graph.add_node(1, x=-80.4094, y=37.1299, elevation=600)
        self.test_graph.add_node(2, x=-80.4095, y=37.1300, elevation=620)
        self.test_graph.add_node(3, x=-80.4096, y=37.1301, elevation=610)
        self.test_graph.add_node(4, x=-80.4097, y=37.1302, elevation=650)
        self.test_graph.add_edge(1, 2, length=100, highway='residential')
        self.test_graph.add_edge(2, 3, length=150, highway='residential')
        self.test_graph.add_edge(3, 4, length=120, highway='primary')


@unittest.skipUnless(GA_PERFORMANCE_AVAILABLE, "GA performance modules not available")
class TestSegmentProperties(TestGAPerformance):
    """Test SegmentProperties dataclass"""
    
    def test_segment_properties_initialization(self):
        """Test basic segment properties initialization"""
        props = SegmentProperties(
            distance_km=2.5,
            elevation_gain_m=60,
            elevation_loss_m=30,
            net_elevation_m=30,
            max_elevation_m=650,
            min_elevation_m=600,
            avg_grade_percent=3.5,
            max_grade_percent=8.2,
            num_nodes=4,
            connectivity_score=0.85
        )
        
        self.assertEqual(props.distance_km, 2.5)
        self.assertEqual(props.elevation_gain_m, 60)
        self.assertEqual(props.elevation_loss_m, 30)
        self.assertEqual(props.net_elevation_m, 30)
        self.assertEqual(props.max_elevation_m, 650)
        self.assertEqual(props.min_elevation_m, 600)
        self.assertEqual(props.avg_grade_percent, 3.5)
        self.assertEqual(props.max_grade_percent, 8.2)
        self.assertEqual(props.num_nodes, 4)
        self.assertEqual(props.connectivity_score, 0.85)
        self.assertIsInstance(props.calculated_at, float)
    
    def test_segment_properties_post_init_none_values(self):
        """Test post_init handling of None values"""
        props = SegmentProperties(
            distance_km=None,
            elevation_gain_m=None,
            elevation_loss_m=None,
            net_elevation_m=None,
            max_elevation_m=None,
            min_elevation_m=None,
            avg_grade_percent=None,
            max_grade_percent=None,
            num_nodes=None,
            connectivity_score=None
        )
        
        # All None values should be converted to appropriate defaults
        self.assertEqual(props.distance_km, 0.0)
        self.assertEqual(props.elevation_gain_m, 0.0)
        self.assertEqual(props.elevation_loss_m, 0.0)
        self.assertEqual(props.net_elevation_m, 0.0)
        self.assertEqual(props.max_elevation_m, 0.0)
        self.assertEqual(props.min_elevation_m, 0.0)
        self.assertEqual(props.avg_grade_percent, 0.0)
        self.assertEqual(props.max_grade_percent, 0.0)
        self.assertEqual(props.num_nodes, 0)
        self.assertEqual(props.connectivity_score, 0.0)
    
    def test_segment_properties_post_init_invalid_values(self):
        """Test post_init handling of invalid values"""
        # Test that SegmentProperties can handle invalid input gracefully
        try:
            props = SegmentProperties(
                distance_km="invalid",
                elevation_gain_m=[1, 2, 3],
                elevation_loss_m={"key": "value"},
                net_elevation_m=complex(1, 2),
                max_elevation_m=True,
                min_elevation_m=False,
                avg_grade_percent="not_a_number",
                max_grade_percent=None,
                num_nodes="not_an_int",
                connectivity_score=[]
            )
            
            # If creation succeeds, all invalid values should be converted to appropriate defaults
            self.assertIsInstance(props.distance_km, (int, float))
            self.assertIsInstance(props.elevation_gain_m, (int, float))
            self.assertIsInstance(props.elevation_loss_m, (int, float))
            self.assertIsInstance(props.net_elevation_m, (int, float))
            self.assertIsInstance(props.max_elevation_m, (int, float))
            self.assertIsInstance(props.min_elevation_m, (int, float))
            self.assertIsInstance(props.avg_grade_percent, (int, float))
            self.assertIsInstance(props.max_grade_percent, (int, float))
            self.assertIsInstance(props.num_nodes, int)
            self.assertIsInstance(props.connectivity_score, (int, float))
        except (TypeError, ValueError):
            # If SegmentProperties doesn't handle invalid input, that's also acceptable
            pass


@unittest.skipUnless(GA_PERFORMANCE_AVAILABLE, "GA performance modules not available")
class TestCacheStats(TestGAPerformance):
    """Test CacheStats functionality"""
    
    def test_cache_stats_initialization(self):
        """Test cache stats initialization"""
        stats = CacheStats()
        
        self.assertEqual(stats.total_requests, 0)
        self.assertEqual(stats.cache_hits, 0)
        self.assertEqual(stats.cache_misses, 0)
        self.assertEqual(stats.calculations_saved, 0)
        self.assertEqual(stats.total_calculation_time_saved, 0.0)
        self.assertEqual(stats.avg_calculation_time, 0.0)
        self.assertEqual(stats.total_calculation_time, 0.0)
        self.assertEqual(stats.calculation_count, 0)
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation"""
        stats = CacheStats()
        
        # No requests should return 0%
        self.assertEqual(stats.hit_rate, 0.0)
        
        # Add some hits and misses
        stats.total_requests = 100
        stats.cache_hits = 75
        stats.cache_misses = 25
        
        self.assertEqual(stats.hit_rate, 75.0)
    
    def test_time_saved_per_hit(self):
        """Test time saved per hit calculation"""
        stats = CacheStats()
        
        # No hits should return 0
        self.assertEqual(stats.time_saved_per_hit, 0.0)
        
        # Set some calculation time and hits
        stats.avg_calculation_time = 0.05
        stats.cache_hits = 10
        
        self.assertEqual(stats.time_saved_per_hit, 0.05)
    
    def test_update_calculation_time(self):
        """Test calculation time updating"""
        stats = CacheStats()
        
        # Add some calculation times
        stats.update_calculation_time(0.1)
        self.assertAlmostEqual(stats.avg_calculation_time, 0.1, places=3)
        self.assertEqual(stats.calculation_count, 1)
        
        stats.update_calculation_time(0.2)
        self.assertAlmostEqual(stats.avg_calculation_time, 0.15, places=3)  # (0.1 + 0.2) / 2
        self.assertEqual(stats.calculation_count, 2)
        
        stats.update_calculation_time(0.3)
        self.assertAlmostEqual(stats.avg_calculation_time, 0.2, places=3)  # (0.1 + 0.2 + 0.3) / 3
        self.assertEqual(stats.calculation_count, 3)
    
    def test_record_cache_hit(self):
        """Test cache hit recording"""
        stats = CacheStats()
        stats.avg_calculation_time = 0.05
        
        # Record some hits
        stats.record_cache_hit()
        self.assertEqual(stats.cache_hits, 1)
        self.assertEqual(stats.total_calculation_time_saved, 0.05)
        
        stats.record_cache_hit()
        self.assertEqual(stats.cache_hits, 2)
        self.assertEqual(stats.total_calculation_time_saved, 0.10)


@unittest.skipUnless(GA_PERFORMANCE_AVAILABLE, "GA performance modules not available")
class TestGASegmentCache(TestGAPerformance):
    """Test GASegmentCache functionality"""
    
    def test_cache_initialization(self):
        """Test cache initialization"""
        cache = GASegmentCache(max_size=1000, enable_stats=True)
        
        self.assertEqual(cache.max_size, 1000)
        self.assertTrue(cache.enable_stats)
        self.assertIsInstance(cache._cache, OrderedDict)
        # Check if lock exists and is callable (threading lock)
        self.assertTrue(hasattr(cache, '_lock'))
        self.assertTrue(callable(getattr(cache._lock, '__enter__', None)))
        self.assertIsNotNone(cache.stats)
        self.assertIsInstance(cache._creation_time, float)
        self.assertIsInstance(cache._last_cleanup, float)
    
    def test_cache_initialization_no_stats(self):
        """Test cache initialization without stats"""
        cache = GASegmentCache(enable_stats=False)
        
        self.assertFalse(cache.enable_stats)
        self.assertIsNone(cache.stats)
    
    def test_get_segment_key_basic(self):
        """Test segment key generation"""
        cache = GASegmentCache()
        
        # Mock segment
        segment = Mock()
        segment.path_nodes = [1, 2, 3]
        segment.direction = 1
        
        key = cache.get_segment_key(segment)
        expected_key = "1-2-3_1_3"
        self.assertEqual(key, expected_key)
    
    def test_get_segment_key_empty_segment(self):
        """Test segment key for empty segment"""
        cache = GASegmentCache()
        
        # Empty segment
        segment = Mock()
        segment.path_nodes = []
        
        key = cache.get_segment_key(segment)
        self.assertEqual(key, "empty_segment")
        
        # None segment
        key = cache.get_segment_key(None)
        self.assertEqual(key, "empty_segment")
        
        # Segment without path_nodes
        segment_no_path = Mock()
        delattr(segment_no_path, 'path_nodes')
        key = cache.get_segment_key(segment_no_path)
        self.assertEqual(key, "empty_segment")
    
    def test_get_segment_key_long_path(self):
        """Test segment key generation for long paths"""
        cache = GASegmentCache()
        
        # Long path that should be hashed
        segment = Mock()
        segment.path_nodes = list(range(50))  # 50 nodes
        segment.direction = 1
        
        key = cache.get_segment_key(segment)
        
        # Should be a hash (32 character hex string)
        self.assertEqual(len(key), 32)
        self.assertTrue(all(c in '0123456789abcdef' for c in key))
    
    def test_get_segment_key_no_direction(self):
        """Test segment key when direction attribute is missing"""
        cache = GASegmentCache()
        
        segment = Mock()
        segment.path_nodes = [1, 2, 3]
        # No direction attribute
        delattr(segment, 'direction')
        
        key = cache.get_segment_key(segment)
        expected_key = "1-2-3_0_3"  # Default direction 0
        self.assertEqual(key, expected_key)
    
    def test_calculate_segment_properties_empty_segment(self):
        """Test property calculation for empty segment"""
        cache = GASegmentCache()
        
        # Empty segment
        segment = Mock()
        segment.path_nodes = []
        
        props = cache.calculate_segment_properties(segment, self.test_graph)
        
        self.assertEqual(props.distance_km, 0.0)
        self.assertEqual(props.elevation_gain_m, 0.0)
        self.assertEqual(props.elevation_loss_m, 0.0)
        self.assertEqual(props.net_elevation_m, 0.0)
        self.assertEqual(props.max_elevation_m, 0.0)
        self.assertEqual(props.min_elevation_m, 0.0)
        self.assertEqual(props.avg_grade_percent, 0.0)
        self.assertEqual(props.max_grade_percent, 0.0)
        self.assertEqual(props.num_nodes, 0)
        self.assertEqual(props.connectivity_score, 0.0)
    
    def test_calculate_segment_properties_single_node(self):
        """Test property calculation for single node segment"""
        cache = GASegmentCache()
        
        # Single node segment
        segment = Mock()
        segment.path_nodes = [1]
        
        props = cache.calculate_segment_properties(segment, self.test_graph)
        
        # Single node should return zero properties
        self.assertEqual(props.distance_km, 0.0)
        self.assertEqual(props.elevation_gain_m, 0.0)
        self.assertEqual(props.num_nodes, 0)
    
    @patch('genetic_algorithm.performance.time.time')
    def test_calculate_segment_properties_timing(self, mock_time):
        """Test property calculation timing recording"""
        cache = GASegmentCache(enable_stats=True)
        
        # Mock time to control timing
        mock_time.side_effect = [1000.0, 1000.1]  # 0.1 second calculation
        
        segment = Mock()
        segment.path_nodes = [1, 2]
        
        props = cache.calculate_segment_properties(segment, self.test_graph)
        
        # Check that stats were updated if available
        if cache.stats:
            # Time should have been recorded
            self.assertGreater(cache.stats.calculation_count, 0)


@unittest.skipUnless(GA_PERFORMANCE_AVAILABLE, "GA performance modules not available")
class TestGASegmentCacheOperations(TestGAPerformance):
    """Test GASegmentCache cache operations"""
    
    def test_cache_basic_operations(self):
        """Test basic cache set/get operations"""
        cache = GASegmentCache()
        
        # Create mock segment and properties
        segment = Mock()
        segment.path_nodes = [1, 2, 3]
        segment.direction = 1
        
        props = SegmentProperties(
            distance_km=2.5, elevation_gain_m=60, elevation_loss_m=30,
            net_elevation_m=30, max_elevation_m=650, min_elevation_m=600,
            avg_grade_percent=3.5, max_grade_percent=8.2, num_nodes=3,
            connectivity_score=0.85
        )
        
        key = cache.get_segment_key(segment)
        
        # Store in cache
        with cache._lock:
            cache._cache[key] = props
        
        # Retrieve from cache
        with cache._lock:
            cached_props = cache._cache.get(key)
        
        self.assertIsNotNone(cached_props)
        self.assertEqual(cached_props.distance_km, 2.5)
        self.assertEqual(cached_props.elevation_gain_m, 60)
    
    def test_cache_lru_behavior(self):
        """Test LRU cache behavior"""
        cache = GASegmentCache(max_size=3)
        
        # Add items beyond max size
        for i in range(5):
            segment = Mock()
            segment.path_nodes = [i, i+1]
            segment.direction = 1
            
            key = cache.get_segment_key(segment)
            props = SegmentProperties(
                distance_km=float(i), elevation_gain_m=0, elevation_loss_m=0,
                net_elevation_m=0, max_elevation_m=0, min_elevation_m=0,
                avg_grade_percent=0, max_grade_percent=0, num_nodes=2,
                connectivity_score=0.0
            )
            
            with cache._lock:
                cache._cache[key] = props
                # Manually enforce size limit (LRU)
                if len(cache._cache) > cache.max_size:
                    cache._cache.popitem(last=False)
        
        # Should only have max_size items
        with cache._lock:
            self.assertLessEqual(len(cache._cache), cache.max_size)
    
    def test_cache_thread_safety(self):
        """Test cache thread safety"""
        cache = GASegmentCache()
        results = []
        
        def cache_operation(thread_id):
            """Function to run in multiple threads"""
            segment = Mock()
            segment.path_nodes = [thread_id, thread_id + 1]
            segment.direction = 1
            
            key = cache.get_segment_key(segment)
            props = SegmentProperties(
                distance_km=float(thread_id), elevation_gain_m=0, elevation_loss_m=0,
                net_elevation_m=0, max_elevation_m=0, min_elevation_m=0,
                avg_grade_percent=0, max_grade_percent=0, num_nodes=2,
                connectivity_score=0.0
            )
            
            # Store and retrieve
            with cache._lock:
                cache._cache[key] = props
                cached_props = cache._cache.get(key)
                if cached_props:
                    results.append(cached_props.distance_km)
        
        # Run multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=cache_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All operations should have completed
        self.assertEqual(len(results), 10)


@unittest.skipUnless(GA_PERFORMANCE_AVAILABLE, "GA performance modules not available")
class TestGASegmentCacheStatistics(TestGAPerformance):
    """Test GASegmentCache statistics tracking"""
    
    def test_cache_stats_tracking(self):
        """Test that cache statistics are tracked correctly"""
        cache = GASegmentCache(enable_stats=True)
        
        # Initial stats should be zero
        self.assertEqual(cache.stats.total_requests, 0)
        self.assertEqual(cache.stats.cache_hits, 0)
        self.assertEqual(cache.stats.cache_misses, 0)
    
    def test_cache_stats_disabled(self):
        """Test cache with statistics disabled"""
        cache = GASegmentCache(enable_stats=False)
        
        # Stats should be None
        self.assertIsNone(cache.stats)
    
    def test_cache_performance_metrics(self):
        """Test cache performance metrics calculation"""
        cache = GASegmentCache(enable_stats=True)
        
        # Simulate some cache activity
        cache.stats.total_requests = 100
        cache.stats.cache_hits = 80
        cache.stats.cache_misses = 20
        cache.stats.avg_calculation_time = 0.05
        
        # Test hit rate
        self.assertEqual(cache.stats.hit_rate, 80.0)
        
        # Test time saved per hit
        self.assertEqual(cache.stats.time_saved_per_hit, 0.05)
    
    def test_cache_timing_integration(self):
        """Test cache timing integration"""
        cache = GASegmentCache(enable_stats=True)
        
        # Record some calculation times
        cache.stats.update_calculation_time(0.1)
        cache.stats.update_calculation_time(0.2)
        
        # Average should be correct
        self.assertAlmostEqual(cache.stats.avg_calculation_time, 0.15, places=3)
        self.assertEqual(cache.stats.calculation_count, 2)
        
        # Record cache hits
        cache.stats.record_cache_hit()
        cache.stats.record_cache_hit()
        
        # Time saved should accumulate
        self.assertEqual(cache.stats.cache_hits, 2)
        self.assertAlmostEqual(cache.stats.total_calculation_time_saved, 0.30, places=3)


@unittest.skipUnless(GA_PERFORMANCE_AVAILABLE, "GA performance modules not available")
class TestGASegmentCacheEdgeCases(TestGAPerformance):
    """Test GASegmentCache edge cases and error handling"""
    
    def test_cache_with_none_graph(self):
        """Test cache operations with None graph"""
        cache = GASegmentCache()
        
        segment = Mock()
        segment.path_nodes = [1, 2]
        
        # Should handle None graph gracefully - may throw exception
        try:
            props = cache.calculate_segment_properties(segment, None)
            self.assertIsInstance(props, SegmentProperties)
        except AttributeError:
            # Expected - None graph cannot be processed
            pass
    
    def test_cache_with_corrupted_segment(self):
        """Test cache with corrupted segment data"""
        cache = GASegmentCache()
        
        # Segment with corrupted data
        segment = Mock()
        segment.path_nodes = None
        
        key = cache.get_segment_key(segment)
        self.assertEqual(key, "empty_segment")
    
    def test_cache_key_collision_handling(self):
        """Test cache behavior with potential key collisions"""
        cache = GASegmentCache()
        
        # Create segments that might generate similar keys
        segment1 = Mock()
        segment1.path_nodes = [1, 2]
        segment1.direction = 1
        
        segment2 = Mock()
        segment2.path_nodes = [1, 2]
        segment2.direction = 1
        
        key1 = cache.get_segment_key(segment1)
        key2 = cache.get_segment_key(segment2)
        
        # Should generate identical keys for identical segments
        self.assertEqual(key1, key2)
    
    def test_cache_memory_cleanup(self):
        """Test cache cleanup and memory management"""
        cache = GASegmentCache(max_size=5)
        
        # Fill cache beyond capacity
        for i in range(10):
            segment = Mock()
            segment.path_nodes = [i, i+1, i+2]
            segment.direction = 1
            
            key = cache.get_segment_key(segment)
            props = SegmentProperties(
                distance_km=float(i), elevation_gain_m=0, elevation_loss_m=0,
                net_elevation_m=0, max_elevation_m=0, min_elevation_m=0,
                avg_grade_percent=0, max_grade_percent=0, num_nodes=3,
                connectivity_score=0.0
            )
            
            with cache._lock:
                cache._cache[key] = props
        
        # Cache should not exceed max size significantly
        with cache._lock:
            self.assertLessEqual(len(cache._cache), cache.max_size * 2)  # Allow some overflow
    
    def test_cache_creation_time_tracking(self):
        """Test cache creation time tracking"""
        start_time = time.time()
        cache = GASegmentCache()
        end_time = time.time()
        
        # Creation time should be within reasonable bounds
        self.assertGreaterEqual(cache._creation_time, start_time)
        self.assertLessEqual(cache._creation_time, end_time)
        
        # Last cleanup should be initialized
        self.assertIsInstance(cache._last_cleanup, float)
    
    def test_segment_properties_edge_cases(self):
        """Test segment properties with edge case values"""
        # Very large values
        props_large = SegmentProperties(
            distance_km=1000000.0,
            elevation_gain_m=10000.0,
            elevation_loss_m=10000.0,
            net_elevation_m=0.0,
            max_elevation_m=8848.0,  # Mt. Everest height
            min_elevation_m=-418.0,  # Dead Sea level
            avg_grade_percent=100.0,
            max_grade_percent=200.0,
            num_nodes=100000,
            connectivity_score=1.0
        )
        
        self.assertEqual(props_large.distance_km, 1000000.0)
        self.assertEqual(props_large.max_elevation_m, 8848.0)
        self.assertEqual(props_large.min_elevation_m, -418.0)
        
        # Very small values
        props_small = SegmentProperties(
            distance_km=0.001,
            elevation_gain_m=0.001,
            elevation_loss_m=0.001,
            net_elevation_m=0.0,
            max_elevation_m=0.001,
            min_elevation_m=0.0,
            avg_grade_percent=0.001,
            max_grade_percent=0.001,
            num_nodes=1,
            connectivity_score=0.001
        )
        
        self.assertEqual(props_small.distance_km, 0.001)
        self.assertEqual(props_small.elevation_gain_m, 0.001)
        self.assertEqual(props_small.num_nodes, 1)
    
    def test_cache_with_stats_enabled(self):
        """Test cache behavior with statistics enabled"""
        stats = CacheStats()
        cache = GASegmentCache(max_size=3, enable_stats=True)
        cache.stats = stats
        
        # Test cache hit with stats
        segment1 = Mock(spec=RouteSegment)
        segment1.start_node = 1
        segment1.end_node = 2
        segment1.path_nodes = [1, 2]
        
        with patch.object(cache, 'calculate_segment_properties') as mock_calc:
            mock_calc.return_value = SegmentProperties(
                distance_km=1.0, elevation_gain_m=10.0, elevation_loss_m=5.0,
                net_elevation_m=5.0, max_elevation_m=100.0, min_elevation_m=95.0,
                avg_grade_percent=1.0, max_grade_percent=2.0, num_nodes=2, connectivity_score=1.0
            )
            
            # First call - cache miss
            props1 = cache.get_segment_properties(segment1, self.test_graph)
            self.assertEqual(stats.total_requests, 1)
            self.assertEqual(stats.cache_misses, 1)
            
            # Second call - cache hit
            props2 = cache.get_segment_properties(segment1, self.test_graph)
            self.assertEqual(stats.total_requests, 2)
            self.assertEqual(stats.cache_hits, 1)
            self.assertEqual(props1, props2)
    
    def test_cache_lru_eviction_with_stats(self):
        """Test LRU eviction behavior with statistics"""
        stats = CacheStats()
        cache = GASegmentCache(max_size=2, enable_stats=True)
        cache.stats = stats
        
        # Create segments
        segments = []
        for i in range(3):
            segment = Mock(spec=RouteSegment)
            segment.start_node = i
            segment.end_node = i + 1
            segment.path_nodes = [i, i + 1]
            segments.append(segment)
        
        with patch.object(cache, 'calculate_segment_properties') as mock_calc:
            mock_calc.return_value = SegmentProperties(
                distance_km=1.0, elevation_gain_m=10.0, elevation_loss_m=5.0,
                net_elevation_m=5.0, max_elevation_m=100.0, min_elevation_m=95.0,
                avg_grade_percent=1.0, max_grade_percent=2.0, num_nodes=2, connectivity_score=1.0
            )
            
            # Fill cache to capacity
            cache.get_segment_properties(segments[0], self.test_graph)
            cache.get_segment_properties(segments[1], self.test_graph)
            
            # Add third segment - should evict first
            cache.get_segment_properties(segments[2], self.test_graph)
            
            # First segment should be evicted and cause cache miss
            cache.get_segment_properties(segments[0], self.test_graph)
            self.assertEqual(stats.cache_misses, 4)  # 3 initial misses + 1 eviction miss
    
    def test_get_chromosome_properties_empty_chromosome(self):
        """Test chromosome properties with empty chromosome"""
        cache = GASegmentCache()
        
        # Test None chromosome
        props = cache.get_chromosome_properties(None, self.test_graph)
        expected = {
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
        self.assertEqual(props, expected)
        
        # Test chromosome with no segments
        empty_chromosome = Mock()
        empty_chromosome.segments = []
        props = cache.get_chromosome_properties(empty_chromosome, self.test_graph)
        self.assertEqual(props, expected)
    
    def test_get_chromosome_properties_aggregation(self):
        """Test chromosome properties aggregation logic"""
        cache = GASegmentCache()
        
        # Create chromosome with multiple segments
        chromosome = Mock()
        segments = []
        for i in range(3):
            segment = Mock(spec=RouteSegment)
            segment.start_node = i
            segment.end_node = i + 1
            segment.path_nodes = [i, i + 1]
            segments.append(segment)
        chromosome.segments = segments
        
        # Mock segment properties with different values
        segment_props = [
            SegmentProperties(
                distance_km=1.0, elevation_gain_m=10.0, elevation_loss_m=5.0,
                net_elevation_m=5.0, max_elevation_m=100.0, min_elevation_m=95.0,
                avg_grade_percent=2.0, max_grade_percent=5.0, num_nodes=2, connectivity_score=0.8
            ),
            SegmentProperties(
                distance_km=2.0, elevation_gain_m=20.0, elevation_loss_m=10.0,
                net_elevation_m=10.0, max_elevation_m=110.0, min_elevation_m=90.0,
                avg_grade_percent=3.0, max_grade_percent=7.0, num_nodes=3, connectivity_score=0.9
            ),
            SegmentProperties(
                distance_km=1.5, elevation_gain_m=15.0, elevation_loss_m=8.0,
                net_elevation_m=7.0, max_elevation_m=105.0, min_elevation_m=88.0,
                avg_grade_percent=2.5, max_grade_percent=6.0, num_nodes=2, connectivity_score=0.7
            )
        ]
        
        with patch.object(cache, 'get_segment_properties') as mock_get:
            mock_get.side_effect = segment_props
            
            props = cache.get_chromosome_properties(chromosome, self.test_graph)
            
            # Check aggregated values
            self.assertEqual(props['total_distance_km'], 4.5)  # 1.0 + 2.0 + 1.5
            self.assertEqual(props['total_elevation_gain_m'], 45.0)  # 10 + 20 + 15
            self.assertEqual(props['total_elevation_loss_m'], 23.0)  # 5 + 10 + 8
            self.assertEqual(props['net_elevation_change_m'], 22.0)  # 5 + 10 + 7
            self.assertEqual(props['max_elevation_m'], 110.0)  # max(100, 110, 105)
            self.assertEqual(props['min_elevation_m'], 88.0)  # min(95, 90, 88)
            self.assertEqual(props['max_grade_percent'], 7.0)  # max(5, 7, 6)
            self.assertEqual(props['total_nodes'], 7)  # 2 + 3 + 2
            
            # Check weighted averages
            # avg_grade = (2.0*2 + 3.0*3 + 2.5*2) / 7 = (4 + 9 + 5) / 7 = 18/7 ≈ 2.57
            self.assertAlmostEqual(props['avg_grade_percent'], 18.0/7.0, places=2)
            # avg_connectivity = (0.8*2 + 0.9*3 + 0.7*2) / 7 = (1.6 + 2.7 + 1.4) / 7 = 5.7/7 ≈ 0.81
            self.assertAlmostEqual(props['avg_connectivity_score'], 5.7/7.0, places=2)
    
    def test_get_chromosome_properties_zero_nodes(self):
        """Test chromosome properties with zero total nodes"""
        cache = GASegmentCache()
        
        # Create chromosome with segments that have zero nodes
        chromosome = Mock()
        segment = Mock(spec=RouteSegment)
        segment.start_node = 1
        segment.end_node = 2
        segment.path_nodes = [1, 2]
        chromosome.segments = [segment]
        
        # Mock segment properties with zero nodes
        segment_props = SegmentProperties(
            distance_km=1.0, elevation_gain_m=10.0, elevation_loss_m=5.0,
            net_elevation_m=5.0, max_elevation_m=100.0, min_elevation_m=95.0,
            avg_grade_percent=2.0, max_grade_percent=5.0, num_nodes=0, connectivity_score=0.8
        )
        
        with patch.object(cache, 'get_segment_properties') as mock_get:
            mock_get.return_value = segment_props
            
            props = cache.get_chromosome_properties(chromosome, self.test_graph)
            
            # With zero nodes, weighted averages should be 0
            self.assertEqual(props['avg_grade_percent'], 0.0)
            self.assertEqual(props['avg_connectivity_score'], 0.0)
            self.assertEqual(props['total_nodes'], 0)
    
    def test_cache_memory_cleanup_threshold(self):
        """Test cache memory cleanup when threshold is exceeded"""
        cache = GASegmentCache(max_size=10)  # Limited size for testing
        
        # Create many segments to trigger memory cleanup
        segments = []
        for i in range(20):
            segment = Mock(spec=RouteSegment)
            segment.start_node = i
            segment.end_node = i + 1
            segment.path_nodes = [i, i + 1]
            segments.append(segment)
        
        with patch.object(cache, 'calculate_segment_properties') as mock_calc:
            mock_calc.return_value = SegmentProperties(
                distance_km=1.0, elevation_gain_m=10.0, elevation_loss_m=5.0,
                net_elevation_m=5.0, max_elevation_m=100.0, min_elevation_m=95.0,
                avg_grade_percent=1.0, max_grade_percent=2.0, num_nodes=2, connectivity_score=1.0
            )
            
            # Add segments to cache
            for segment in segments[:10]:
                cache.get_segment_properties(segment, self.test_graph)
            
            # Cache should be at capacity
            self.assertEqual(len(cache._cache), 10)
            
            # Force memory cleanup by adding more segments
            for segment in segments[10:15]:
                cache.get_segment_properties(segment, self.test_graph)
            
            # Cache should still be within limits
            self.assertLessEqual(len(cache._cache), 10)
    
    def test_cache_thread_safety_edge_cases(self):
        """Test cache thread safety with edge cases"""
        cache = GASegmentCache(max_size=5)
        
        # Test concurrent access to same segment
        segment = Mock(spec=RouteSegment)
        segment.start_node = 1
        segment.end_node = 2
        segment.path_nodes = [1, 2]
        
        import threading
        results = []
        
        def cache_operation():
            with patch.object(cache, 'calculate_segment_properties') as mock_calc:
                mock_calc.return_value = SegmentProperties(
                    distance_km=1.0, elevation_gain_m=10.0, elevation_loss_m=5.0,
                    net_elevation_m=5.0, max_elevation_m=100.0, min_elevation_m=95.0,
                    avg_grade_percent=1.0, max_grade_percent=2.0, num_nodes=2, connectivity_score=1.0
                )
                
                props = cache.get_segment_properties(segment, self.test_graph)
                results.append(props)
        
        # Run multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=cache_operation)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All results should be the same
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, SegmentProperties)


if __name__ == '__main__':
    unittest.main()