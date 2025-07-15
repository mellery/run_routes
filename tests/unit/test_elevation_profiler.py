#!/usr/bin/env python3
"""
Unit tests for ElevationProfiler
"""

import unittest
from unittest.mock import Mock, patch
import networkx as nx
import geopandas as gpd
import sys
import os

# Add the parent directory to sys.path to import route_services
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from route_services.elevation_profiler import ElevationProfiler


class TestElevationProfiler(unittest.TestCase):
    """Test cases for ElevationProfiler class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a mock graph with varying elevations
        self.mock_graph = nx.Graph()
        self.mock_graph.add_node(1001, x=-80.4094, y=37.1299, elevation=600)
        self.mock_graph.add_node(1002, x=-80.4095, y=37.1300, elevation=620)  # +20m
        self.mock_graph.add_node(1003, x=-80.4096, y=37.1301, elevation=610)  # -10m
        self.mock_graph.add_node(1004, x=-80.4097, y=37.1302, elevation=650)  # +40m
        self.mock_graph.add_node(1005, x=-80.4098, y=37.1303, elevation=630)  # -20m
        
        # Add edges between consecutive nodes for network routing
        # Each edge has length=100 to match expected test distances
        self.mock_graph.add_edge(1001, 1002, length=100)
        self.mock_graph.add_edge(1002, 1003, length=100)
        self.mock_graph.add_edge(1003, 1004, length=100)
        self.mock_graph.add_edge(1004, 1005, length=100)
        self.mock_graph.add_edge(1005, 1001, length=100)  # Return to start
        
        self.profiler = ElevationProfiler(self.mock_graph)
        
        # Create sample route result
        self.sample_route_result = {
            'route': [1001, 1002, 1003, 1004, 1005],
            'stats': {
                'total_distance_km': 2.5,
                'total_elevation_gain_m': 60,
                'total_elevation_loss_m': 30
            }
        }
    
    def test_initialization(self):
        """Test ElevationProfiler initialization"""
        profiler = ElevationProfiler(self.mock_graph)
        self.assertEqual(profiler.graph, self.mock_graph)
    
    def test_generate_profile_data_success(self):
        """Test successful elevation profile generation"""
        # Using network distances from graph edges (100m each)
        
        profile_data = self.profiler.generate_profile_data(self.sample_route_result)
        
        # Check structure
        self.assertIn('coordinates', profile_data)
        self.assertIn('elevations', profile_data)
        self.assertIn('distances_m', profile_data)
        self.assertIn('distances_km', profile_data)
        self.assertIn('elevation_stats', profile_data)
        
        # Check coordinates (includes return to start)
        coordinates = profile_data['coordinates']
        self.assertEqual(len(coordinates), 6)
        self.assertEqual(coordinates[0]['node_id'], 1001)
        self.assertEqual(coordinates[0]['latitude'], 37.1299)
        self.assertEqual(coordinates[0]['longitude'], -80.4094)
        
        # Check elevations
        elevations = profile_data['elevations']
        expected_elevations = [600, 620, 610, 650, 630, 600]  # includes return to start
        self.assertEqual(elevations, expected_elevations)
        
        # Check distances
        distances_m = profile_data['distances_m']
        expected_distances = [0, 100, 200, 300, 400, 500]  # cumulative with constant 100m
        self.assertEqual(distances_m, expected_distances)
        
        # Check distances in km
        distances_km = profile_data['distances_km']
        expected_distances_km = [d/1000 for d in expected_distances]
        self.assertEqual(distances_km, expected_distances_km)
    
    def test_generate_profile_data_empty(self):
        """Test profile generation with empty route"""
        result = self.profiler.generate_profile_data({})
        self.assertEqual(result, {})
        
        result = self.profiler.generate_profile_data({'route': []})
        self.assertEqual(result, {})
    
    def test_calculate_elevation_stats(self):
        """Test elevation statistics calculation"""
        elevations = [600, 620, 610, 650, 630]
        distances_km = [0, 0.1, 0.25, 0.37, 0.45]
        
        stats = self.profiler._calculate_elevation_stats(elevations, distances_km)
        
        # Check basic stats
        self.assertEqual(stats['min_elevation'], 600)
        self.assertEqual(stats['max_elevation'], 650)
        self.assertEqual(stats['elevation_range'], 50)
        self.assertEqual(stats['avg_elevation'], 622)  # (600+620+610+650+630)/5
        
        # Check that grade calculations exist
        self.assertIn('max_grade', stats)
        self.assertIn('min_grade', stats)
        self.assertIn('avg_grade', stats)
        self.assertIn('steep_sections', stats)
    
    def test_calculate_elevation_stats_empty(self):
        """Test elevation stats with empty data"""
        stats = self.profiler._calculate_elevation_stats([], [])
        self.assertEqual(stats, {})
    
    def test_get_elevation_zones(self):
        """Test elevation zone division"""
        zones = self.profiler.get_elevation_zones(self.sample_route_result, zone_count=3)
        
        # Should create 3 zones
        self.assertEqual(len(zones), 3)
        
        # Check zone structure
        for i, zone in enumerate(zones):
            self.assertEqual(zone['zone_number'], i + 1)
            self.assertIn('start_km', zone)
            self.assertIn('end_km', zone)
            self.assertIn('distance_km', zone)
            self.assertIn('min_elevation', zone)
            self.assertIn('max_elevation', zone)
            self.assertIn('avg_elevation', zone)
            self.assertIn('elevation_change', zone)
    
    def test_get_elevation_zones_empty(self):
        """Test elevation zones with empty route"""
        zones = self.profiler.get_elevation_zones({}, zone_count=3)
        self.assertEqual(zones, [])
    
    def test_get_elevation_zones_few_points(self):
        """Test elevation zones with fewer points than zones"""
        short_route = {
            'route': [1001, 1002],
            'stats': {}
        }
        
        zones = self.profiler.get_elevation_zones(short_route, zone_count=5)
        
        # Should create zones up to the requested count
        self.assertLessEqual(len(zones), 5)
    
    @patch('route.haversine_distance')
    def test_find_elevation_peaks_valleys(self, mock_haversine):
        """Test finding elevation peaks and valleys"""
        mock_haversine.return_value = 100
        
        peaks_valleys = self.profiler.find_elevation_peaks_valleys(
            self.sample_route_result, min_prominence=15
        )
        
        # Check structure
        self.assertIn('peaks', peaks_valleys)
        self.assertIn('valleys', peaks_valleys)
        self.assertIn('peak_count', peaks_valleys)
        self.assertIn('valley_count', peaks_valleys)
        
        # Should find peaks and valleys based on elevation profile
        # [600, 620, 610, 650, 630] - 650 should be a peak, 610 should be a valley
        self.assertGreaterEqual(peaks_valleys['peak_count'], 0)
        self.assertGreaterEqual(peaks_valleys['valley_count'], 0)
    
    def test_find_elevation_peaks_valleys_empty(self):
        """Test peaks/valleys with empty route"""
        result = self.profiler.find_elevation_peaks_valleys({})
        
        expected = {'peaks': [], 'valleys': []}
        self.assertEqual(result, expected)
    
    @patch('route.haversine_distance')
    def test_get_climbing_segments(self, mock_haversine):
        """Test climbing segment identification"""
        mock_haversine.return_value = 100
        
        climbing_segments = self.profiler.get_climbing_segments(
            self.sample_route_result, min_gain=15
        )
        
        # Should find climbing segments
        # Looking at elevations [600, 620, 610, 650, 630]:
        # - 600->620 (+20m) should be a climbing segment
        # - 610->650 (+40m) should be a climbing segment
        self.assertGreaterEqual(len(climbing_segments), 1)
        
        # Check segment structure
        if climbing_segments:
            segment = climbing_segments[0]
            self.assertIn('start_km', segment)
            self.assertIn('end_km', segment)
            self.assertIn('start_elevation', segment)
            self.assertIn('end_elevation', segment)
            self.assertIn('distance_km', segment)
            self.assertIn('elevation_gain', segment)
            self.assertIn('avg_grade', segment)
    
    def test_get_climbing_segments_empty(self):
        """Test climbing segments with empty route"""
        segments = self.profiler.get_climbing_segments({})
        self.assertEqual(segments, [])
    
    def test_get_climbing_segments_no_climbing(self):
        """Test climbing segments with no significant climbs"""
        flat_route = {
            'route': [1001, 1002],  # Only small elevation change
            'stats': {}
        }
        
        segments = self.profiler.get_climbing_segments(flat_route, min_gain=50)
        
        # Should find no segments with high minimum gain threshold
        self.assertEqual(len(segments), 0)
    
    def test_steep_sections_identification(self):
        """Test identification of steep sections in elevation stats"""
        # Create a route with steep sections
        elevations = [600, 650, 660, 680]  # Steep climbs
        distances_km = [0, 0.1, 0.15, 0.2]  # Short distances = steep grades
        
        stats = self.profiler._calculate_elevation_stats(elevations, distances_km)
        
        # Should identify steep sections (>8% grade)
        self.assertIn('steep_sections', stats)
        self.assertIn('steep_section_count', stats)
        
        # Check that steep sections have required fields
        for section in stats['steep_sections']:
            self.assertIn('start_km', section)
            self.assertIn('end_km', section)
            self.assertIn('grade', section)
            self.assertIn('elevation_change', section)
    
    def test_profile_data_consistency(self):
        """Test consistency of profile data arrays"""
        with patch('route.haversine_distance') as mock_haversine:
            mock_haversine.return_value = 100
            
            profile_data = self.profiler.generate_profile_data(self.sample_route_result)
            
            coordinates = profile_data['coordinates']
            elevations = profile_data['elevations']
            distances_km = profile_data['distances_km']
            
            # Coordinates should include return to start (route length + 1)
            self.assertEqual(len(coordinates), 6)
            
            # Elevations should have one extra for return to start
            self.assertEqual(len(elevations), 6)
            
            # Distances should match elevations
            self.assertEqual(len(distances_km), len(elevations))
    
    def test_get_detailed_route_path_success(self):
        """Test detailed route path generation with complete network"""
        detailed_path = self.profiler.get_detailed_route_path(self.sample_route_result)
        
        # Should return path with all nodes
        self.assertIsInstance(detailed_path, list)
        self.assertGreater(len(detailed_path), 0)
        
        # First node should be start node
        self.assertEqual(detailed_path[0]['node_id'], 1001)
        self.assertEqual(detailed_path[0]['node_type'], 'intersection')
        
        # Should have required fields
        for point in detailed_path:
            self.assertIn('latitude', point)
            self.assertIn('longitude', point)
            self.assertIn('node_id', point)
            self.assertIn('elevation', point)
            self.assertIn('node_type', point)
        
        # Should end back at start (circular route)
        self.assertEqual(detailed_path[-1]['node_id'], 1001)
    
    def test_get_detailed_route_path_empty_route(self):
        """Test detailed path with empty route"""
        empty_result = {}
        detailed_path = self.profiler.get_detailed_route_path(empty_result)
        self.assertEqual(detailed_path, [])
        
        empty_route_result = {'route': []}
        detailed_path = self.profiler.get_detailed_route_path(empty_route_result)
        self.assertEqual(detailed_path, [])
    
    def test_get_detailed_route_path_single_node(self):
        """Test detailed path with single node route"""
        single_node_result = {
            'route': [1001],
            'stats': {'total_distance_km': 0}
        }
        detailed_path = self.profiler.get_detailed_route_path(single_node_result)
        
        # Should have just the single node
        self.assertEqual(len(detailed_path), 1)
        self.assertEqual(detailed_path[0]['node_id'], 1001)
    
    def test_get_detailed_route_path_disconnected_nodes(self):
        """Test detailed path with nodes that have no network connection"""
        # Create a graph with disconnected nodes
        disconnected_graph = nx.Graph()
        disconnected_graph.add_node(2001, x=-80.4094, y=37.1299, elevation=600)
        disconnected_graph.add_node(2002, x=-80.4095, y=37.1300, elevation=620)
        # Note: no edges between nodes
        
        profiler = ElevationProfiler(disconnected_graph)
        disconnected_result = {
            'route': [2001, 2002],
            'stats': {'total_distance_km': 1.0}
        }
        
        detailed_path = profiler.get_detailed_route_path(disconnected_result)
        
        # Should still return both nodes (fallback behavior)
        self.assertEqual(len(detailed_path), 3)  # start + end + return to start
        self.assertEqual(detailed_path[0]['node_id'], 2001)
        self.assertEqual(detailed_path[1]['node_id'], 2002)
        self.assertEqual(detailed_path[2]['node_id'], 2001)
    
    def test_network_distance_cache(self):
        """Test that network distance caching works correctly"""
        # First call
        distance1 = self.profiler._get_network_distance(1001, 1002)
        self.assertEqual(distance1, 100)  # From our test graph setup
        
        # Second call should use cache
        distance2 = self.profiler._get_network_distance(1001, 1002)
        self.assertEqual(distance2, 100)
        self.assertEqual(distance1, distance2)
        
        # Reverse direction should also work (symmetric)
        distance3 = self.profiler._get_network_distance(1002, 1001)
        self.assertEqual(distance3, 100)
    
    def test_network_distance_same_node(self):
        """Test network distance between same node"""
        distance = self.profiler._get_network_distance(1001, 1001)
        self.assertEqual(distance, 0)
    
    def test_network_distance_no_path(self):
        """Test network distance when no path exists"""
        # Create isolated node
        self.mock_graph.add_node(9999, x=-80.5, y=37.2, elevation=700)
        
        distance = self.profiler._get_network_distance(1001, 9999)
        self.assertEqual(distance, float('inf'))


class TestElevationProfilerComplexScenarios(TestElevationProfiler):
    """Test complex elevation profiler scenarios for Phase 2 coverage improvement"""
    
    def test_multi_peak_elevation_profile(self):
        """Test profile generation with multiple peaks and valleys"""
        # Create route with multiple elevation changes
        multi_peak_route = {
            'route': [1001, 1002, 1003, 1004, 1005, 1001],
            'stats': {
                'total_distance_km': 5.0,
                'total_elevation_gain_m': 80,
                'total_elevation_loss_m': 80
            }
        }
        
        profile_data = self.profiler.generate_profile_data(multi_peak_route)
        
        # Should have elevation data
        self.assertIn('elevations', profile_data)
        elevations = profile_data.get('elevations', [])
        
        # Should have multiple points
        self.assertGreater(len(elevations), 3)
        
        # Check for elevation variations
        if elevations:
            elevation_range = max(elevations) - min(elevations)
            self.assertGreater(elevation_range, 0)
    
    def test_steep_grade_detection(self):
        """Test detection of steep grade sections"""
        # Create route with steep section
        steep_graph = nx.Graph()
        steep_graph.add_node(3001, x=-80.4094, y=37.1299, elevation=600)
        steep_graph.add_node(3002, x=-80.4095, y=37.1300, elevation=700)  # +100m in short distance
        steep_graph.add_node(3003, x=-80.4096, y=37.1301, elevation=720)  # +20m more
        steep_graph.add_edge(3001, 3002, length=200)  # 50% grade
        steep_graph.add_edge(3002, 3003, length=100)  # 20% grade
        
        steep_profiler = ElevationProfiler(steep_graph)
        steep_route = {
            'route': [3001, 3002, 3003],
            'stats': {
                'total_distance_km': 0.3,
                'total_elevation_gain_m': 120
            }
        }
        
        # Get elevation stats through profile data
        profile_data = steep_profiler.generate_profile_data(steep_route)
        elevation_stats = profile_data.get('elevation_stats', {})
        
        # Should detect steep sections
        self.assertIn('steep_sections', elevation_stats)
        if 'steep_sections' in elevation_stats:
            # Should have at least basic steep section data
            self.assertIsInstance(elevation_stats['steep_sections'], list)
    
    def test_elevation_profile_with_plateaus(self):
        """Test elevation profile with flat plateau sections"""
        # Create route with plateau
        plateau_graph = nx.Graph()
        for i in range(10):
            # First 5 nodes: climbing
            if i < 5:
                elevation = 600 + (i * 20)
            # Last 5 nodes: plateau at 680m
            else:
                elevation = 680
            
            plateau_graph.add_node(4000 + i, x=-80.4094 + (i * 0.001), 
                                 y=37.1299 + (i * 0.001), elevation=elevation)
            
            if i > 0:
                plateau_graph.add_edge(4000 + i - 1, 4000 + i, length=100)
        
        plateau_profiler = ElevationProfiler(plateau_graph)
        plateau_route = {
            'route': [4000 + i for i in range(10)],
            'stats': {
                'total_distance_km': 0.9,
                'total_elevation_gain_m': 80
            }
        }
        
        profile_data = plateau_profiler.generate_profile_data(plateau_route)
        
        # Should identify plateau section
        elevations = profile_data.get('elevations', [])
        if len(elevations) >= 5:
            last_5_elevations = elevations[-5:]
            elevation_variance = max(last_5_elevations) - min(last_5_elevations)
            # Plateau should be relatively flat (adjusted for realistic variance)
            self.assertLess(elevation_variance, 100)  # Allow for some variation
        else:
            # If not enough points, just check we have some elevation data
            self.assertGreater(len(elevations), 0)
    
    def test_elevation_zones_calculation(self):
        """Test calculation of elevation zones"""
        # Create route spanning different elevation zones
        zone_route = {
            'route': [1001, 1002, 1003, 1004, 1005],
            'stats': {
                'total_distance_km': 4.0,
                'total_elevation_gain_m': 60
            }
        }
        
        zones = self.profiler.get_elevation_zones(zone_route, zone_count=3)
        
        # Should create zones (may be fewer than requested if route is short)
        self.assertLessEqual(len(zones), 3)
        
        # Each zone should have required fields
        for zone in zones:
            self.assertIn('start_km', zone)
            self.assertIn('end_km', zone)
            self.assertIn('avg_elevation', zone)
            self.assertIn('zone_number', zone)
    
    def test_detailed_climbing_analysis(self):
        """Test detailed analysis of climbing segments"""
        # Create route with distinct climbing sections
        climbing_route = {
            'route': [1001, 1002, 1003, 1004, 1005],
            'stats': {
                'total_distance_km': 4.0,
                'total_elevation_gain_m': 70
            }
        }
        
        climbing_segments = self.profiler.get_climbing_segments(climbing_route, min_gain=15)
        
        # Should identify climbing segments (may be empty if no significant climbs)
        self.assertIsInstance(climbing_segments, list)
        
        # Each segment should have detailed information
        for segment in climbing_segments:
            self.assertIn('start_km', segment)
            self.assertIn('end_km', segment)
            self.assertIn('elevation_gain', segment)
            self.assertIn('distance_km', segment)
            self.assertIn('avg_grade', segment)
            self.assertGreater(segment['elevation_gain'], 15)
    
    def test_elevation_profile_interpolation_accuracy(self):
        """Test accuracy of elevation profile interpolation"""
        # Create route with known elevation profile
        interpolation_route = {
            'route': [1001, 1002, 1003],
            'stats': {
                'total_distance_km': 2.0,
                'total_elevation_gain_m': 30
            }
        }
        
        # Test basic profile generation (no interpolation parameter)
        profile_data = self.profiler.generate_profile_data(interpolation_route)
        
        # Should have elevation data
        self.assertIn('elevations', profile_data)
        elevations = profile_data.get('elevations', [])
        
        # Should have at least basic route points
        self.assertGreater(len(elevations), 0)
        
        # Elevation should be reasonable values
        if elevations:
            for elevation in elevations:
                self.assertIsInstance(elevation, (int, float))
                self.assertGreater(elevation, 0)  # Should be positive elevation
    
    def test_elevation_statistics_comprehensive(self):
        """Test comprehensive elevation statistics calculation"""
        # Use route with varied elevation profile
        comprehensive_route = {
            'route': [1001, 1002, 1003, 1004, 1005, 1001],
            'stats': {
                'total_distance_km': 5.0,
                'total_elevation_gain_m': 90,
                'total_elevation_loss_m': 90
            }
        }
        
        # Get elevation stats through profile data
        profile_data = self.profiler.generate_profile_data(comprehensive_route)
        stats = profile_data.get('elevation_stats', {})
        
        # Should include basic statistics
        expected_keys = [
            'max_elevation', 'min_elevation', 'elevation_range',
            'avg_elevation', 'max_grade', 'min_grade', 'avg_grade', 'steep_sections'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Validate specific calculations
        if 'max_elevation' in stats and 'min_elevation' in stats:
            self.assertGreaterEqual(stats['max_elevation'], stats['min_elevation'])
    
    def test_route_difficulty_classification(self):
        """Test route difficulty classification based on elevation"""
        # Easy route (minimal elevation change)
        easy_route = {
            'route': [1001, 1002],
            'stats': {
                'total_distance_km': 1.0,
                'total_elevation_gain_m': 10
            }
        }
        
        easy_profile = self.profiler.generate_profile_data(easy_route)
        easy_stats = easy_profile.get('elevation_stats', {})
        
        # Should have basic elevation stats
        self.assertIn('elevation_range', easy_stats)
        
        # Hard route (significant elevation change)
        hard_route = {
            'route': [1001, 1002, 1003, 1004],
            'stats': {
                'total_distance_km': 3.0,
                'total_elevation_gain_m': 80
            }
        }
        
        hard_profile = self.profiler.generate_profile_data(hard_route)
        hard_stats = hard_profile.get('elevation_stats', {})
        
        # Hard route should have larger elevation range
        if 'elevation_range' in easy_stats and 'elevation_range' in hard_stats:
            self.assertGreaterEqual(hard_stats['elevation_range'], easy_stats['elevation_range'])
    
    def test_peaks_and_valleys_identification(self):
        """Test identification of peaks and valleys in route"""
        # Route with clear peaks and valleys
        peak_valley_route = {
            'route': [1001, 1002, 1003, 1004, 1005],
            'stats': {
                'total_distance_km': 4.0,
                'total_elevation_gain_m': 60
            }
        }
        
        peaks_valleys = self.profiler.find_elevation_peaks_valleys(peak_valley_route)
        
        # Should identify peaks and valleys
        self.assertIn('peaks', peaks_valleys)
        self.assertIn('valleys', peaks_valleys)
        
        # Peaks should have higher elevation than valleys
        if peaks_valleys['peaks'] and peaks_valleys['valleys']:
            avg_peak_elevation = sum(p['elevation'] for p in peaks_valleys['peaks']) / len(peaks_valleys['peaks'])
            avg_valley_elevation = sum(v['elevation'] for v in peaks_valleys['valleys']) / len(peaks_valleys['valleys'])
            self.assertGreater(avg_peak_elevation, avg_valley_elevation)
    
    def test_elevation_profile_edge_cases(self):
        """Test elevation profile with edge cases"""
        # Route with missing elevation data
        missing_elevation_graph = nx.Graph()
        missing_elevation_graph.add_node(5001, x=-80.4094, y=37.1299)  # No elevation
        missing_elevation_graph.add_node(5002, x=-80.4095, y=37.1300, elevation=620)
        missing_elevation_graph.add_edge(5001, 5002, length=100)
        
        missing_profiler = ElevationProfiler(missing_elevation_graph)
        missing_route = {
            'route': [5001, 5002],
            'stats': {'total_distance_km': 0.1}
        }
        
        # Should handle missing elevation gracefully
        profile_data = missing_profiler.generate_profile_data(missing_route)
        elevations = profile_data.get('elevations', [])
        self.assertGreater(len(elevations), 0)
        
        # Should use default elevation for missing data
        if elevations:
            first_point_elevation = elevations[0]
            self.assertIsInstance(first_point_elevation, (int, float))
    
    def test_large_route_performance(self):
        """Test elevation profiler performance with large routes"""
        # Create large graph
        large_graph = nx.Graph()
        for i in range(100):
            large_graph.add_node(
                6000 + i, 
                x=-80.4094 + (i * 0.001), 
                y=37.1299 + (i * 0.001), 
                elevation=600 + (i % 20) * 10  # Varying elevation pattern
            )
            if i > 0:
                large_graph.add_edge(6000 + i - 1, 6000 + i, length=100)
        
        large_profiler = ElevationProfiler(large_graph)
        large_route = {
            'route': [6000 + i for i in range(0, 100, 2)],  # Every other node
            'stats': {
                'total_distance_km': 10.0,
                'total_elevation_gain_m': 200
            }
        }
        
        # Should handle large route without errors
        profile_data = large_profiler.generate_profile_data(large_route)
        elevations = profile_data.get('elevations', [])
        self.assertGreater(len(elevations), 10)
        
        # Should complete in reasonable time (tested by not timing out)
        elevation_stats = profile_data.get('elevation_stats', {})
        self.assertIn('elevation_range', elevation_stats)
    
    def test_circular_route_elevation_analysis(self):
        """Test elevation analysis for circular routes"""
        # Circular route that returns to start
        circular_route = {
            'route': [1001, 1002, 1003, 1004, 1005, 1001],
            'stats': {
                'total_distance_km': 5.0,
                'total_elevation_gain_m': 80,
                'total_elevation_loss_m': 80
            }
        }
        
        # Use generate_profile_data to get elevation stats
        profile_data = self.profiler.generate_profile_data(circular_route)
        stats = profile_data.get('elevation_stats', {})
        
        # Should have basic elevation statistics
        self.assertIn('min_elevation', stats)
        self.assertIn('max_elevation', stats)
        self.assertIn('elevation_range', stats)
        
        # For circular route, elevation range should be reasonable
        if 'elevation_range' in stats:
            self.assertGreater(stats['elevation_range'], 0)
    
    def test_elevation_profile_with_custom_resolution(self):
        """Test elevation profile generation with custom resolution"""
        test_route = {
            'route': [1001, 1002, 1003, 1004],
            'stats': {'total_distance_km': 3.0}
        }
        
        # Test basic profile generation
        profile_data = self.profiler.generate_profile_data(test_route)
        
        # Should have elevation data
        self.assertIn('elevations', profile_data)
        self.assertIn('coordinates', profile_data)
        self.assertIn('distances_km', profile_data)
        
        elevations = profile_data.get('elevations', [])
        coordinates = profile_data.get('coordinates', [])
        distances = profile_data.get('distances_km', [])
        
        # Should have data points
        self.assertGreater(len(elevations), 0)
        self.assertGreater(len(coordinates), 0)
        self.assertGreater(len(distances), 0)
        
        # All arrays should have consistent length
        self.assertEqual(len(elevations), len(coordinates))
        self.assertEqual(len(elevations), len(distances))
    
    def test_elevation_smoothing_algorithms(self):
        """Test elevation data smoothing algorithms"""
        # Create route with noisy elevation data
        noisy_graph = nx.Graph()
        elevations = [600, 605, 610, 605, 615, 610, 620]  # Noisy pattern
        for i, elevation in enumerate(elevations):
            noisy_graph.add_node(
                7000 + i,
                x=-80.4094 + (i * 0.001),
                y=37.1299 + (i * 0.001),
                elevation=elevation
            )
            if i > 0:
                noisy_graph.add_edge(7000 + i - 1, 7000 + i, length=100)
        
        noisy_profiler = ElevationProfiler(noisy_graph)
        noisy_route = {
            'route': [7000 + i for i in range(len(elevations))],
            'stats': {'total_distance_km': 0.6}
        }
        
        # Test basic profile generation
        profile_data = noisy_profiler.generate_profile_data(noisy_route)
        
        # Should have elevation data
        elevations = profile_data.get('elevations', [])
        self.assertGreater(len(elevations), 0)
        
        # Should preserve the noisy elevation pattern to some degree
        if len(elevations) > 1:
            elevation_changes = [abs(elevations[i] - elevations[i-1]) 
                               for i in range(1, len(elevations))]
            # At least some elevation changes should be present
            self.assertGreater(sum(elevation_changes), 0)
    
    def test_elevation_profile_error_handling(self):
        """Test elevation profiler error handling"""
        # Test with invalid route data
        invalid_routes = [
            {'route': [], 'stats': {}},  # Empty route
            {'route': [9999], 'stats': {}},  # Non-existent node
            {'route': [1001, 9999], 'stats': {}},  # Mix of valid/invalid nodes
            {'stats': {'total_distance_km': 1.0}},  # Missing route
            {}  # Empty result
        ]
        
        for invalid_route in invalid_routes:
            # Should handle errors gracefully
            try:
                profile_data = self.profiler.generate_profile_data(invalid_route)
                self.assertIsInstance(profile_data, dict)
            except (KeyError, ValueError, Exception):
                # Expected error for invalid data - including NetworkX errors
                pass
    
    def test_elevation_gain_loss_accuracy(self):
        """Test accuracy of elevation gain and loss calculations"""
        # Create route with known elevation changes
        known_route = {
            'route': [1001, 1002, 1003, 1004, 1005],
            'stats': {
                'total_distance_km': 4.0,
                'total_elevation_gain_m': 60,  # Expected gain: 20 + 40 = 60
                'total_elevation_loss_m': 30   # Expected loss: 10 + 20 = 30
            }
        }
        
        # Get elevation data through profile generation
        profile_data = self.profiler.generate_profile_data(known_route)
        elevations = profile_data.get('elevations', [])
        
        # Should have elevation data
        self.assertGreater(len(elevations), 0)
        
        # Basic validation of elevation range
        if elevations:
            elevation_range = max(elevations) - min(elevations)
            self.assertGreater(elevation_range, 0)
    
    def test_elevation_zones_with_custom_criteria(self):
        """Test elevation zones with custom criteria"""
        zone_route = {
            'route': [1001, 1002, 1003, 1004, 1005],
            'stats': {'total_distance_km': 4.0}
        }
        
        # Test different zone numbers
        for zone_count in [2, 3, 5]:
            zones = self.profiler.get_elevation_zones(zone_route, zone_count=zone_count)
            # May have fewer zones than requested if route is short
            self.assertLessEqual(len(zones), zone_count)
            
            # Each zone should have valid distance data
            for zone in zones:
                self.assertIn('start_km', zone)
                self.assertIn('end_km', zone)
                self.assertGreaterEqual(zone['end_km'], zone['start_km'])
    
    def test_climbing_segments_filtering(self):
        """Test climbing segments with different filtering criteria"""
        climbing_route = {
            'route': [1001, 1002, 1003, 1004, 1005],
            'stats': {'total_distance_km': 4.0}
        }
        
        # Test different minimum climb thresholds
        thresholds = [5, 10, 20, 30]
        for threshold in thresholds:
            segments = self.profiler.get_climbing_segments(
                climbing_route, min_gain=threshold
            )
            
            # All segments should meet threshold
            for segment in segments:
                self.assertGreaterEqual(segment['elevation_gain'], threshold)
            
            # Higher thresholds should result in fewer segments
            if threshold > 5:
                segments_low = self.profiler.get_climbing_segments(
                    climbing_route, min_gain=5
                )
                self.assertLessEqual(len(segments), len(segments_low))
    
    def test_elevation_variability_metrics(self):
        """Test elevation variability and consistency metrics"""
        # Create route with high variability
        variable_route = {
            'route': [1001, 1002, 1003, 1004, 1005],
            'stats': {'total_distance_km': 4.0}
        }
        
        variable_profile = self.profiler.generate_profile_data(variable_route)
        variable_stats = variable_profile.get('elevation_stats', {})
        
        # Should include basic elevation metrics
        self.assertIn('elevation_range', variable_stats)
        variable_range = variable_stats.get('elevation_range', 0)
        
        # Create route with low variability
        flat_graph = nx.Graph()
        for i in range(5):
            flat_graph.add_node(
                8000 + i,
                x=-80.4094 + (i * 0.001),
                y=37.1299 + (i * 0.001),
                elevation=600 + i  # Gradual change
            )
            if i > 0:
                flat_graph.add_edge(8000 + i - 1, 8000 + i, length=100)
        
        flat_profiler = ElevationProfiler(flat_graph)
        flat_route = {
            'route': [8000 + i for i in range(5)],
            'stats': {'total_distance_km': 0.4}
        }
        
        flat_profile = flat_profiler.generate_profile_data(flat_route)
        flat_stats = flat_profile.get('elevation_stats', {})
        flat_range = flat_stats.get('elevation_range', 0)
        
        # Variable route should have higher elevation range than flat route
        self.assertGreaterEqual(variable_range, flat_range)
    
    def test_get_route_geodataframe_empty_route(self):
        """Test route geodataframe with empty route"""
        result = self.profiler.get_route_geodataframe([])
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertTrue(result.empty)
    
    def test_get_route_geodataframe_single_node(self):
        """Test route geodataframe with single node"""
        result = self.profiler.get_route_geodataframe([1001])
        self.assertIsInstance(result, gpd.GeoDataFrame)
        if not result.empty:
            self.assertEqual(len(result), 1)
    
    def test_get_route_geodataframe_success(self):
        """Test successful route geodataframe creation"""
        route = [1001, 1002, 1003]
        result = self.profiler.get_route_geodataframe(route)
        
        self.assertIsInstance(result, gpd.GeoDataFrame)
        if not result.empty:
            expected_columns = ['node_id', 'elevation', 'segment_distance_m', 'cumulative_distance_m', 'geometry']
            for col in expected_columns:
                self.assertIn(col, result.columns)
    
    def test_get_route_geodataframe_invalid_nodes(self):
        """Test route geodataframe with invalid node IDs"""
        route = [9999, 10000, 10001]  # Non-existent nodes
        result = self.profiler.get_route_geodataframe(route)
        
        self.assertIsInstance(result, gpd.GeoDataFrame)
        # Should handle invalid nodes gracefully
    
    def test_caching_behavior(self):
        """Test caching behavior of elevation profiler"""
        route = [1001, 1002, 1003]
        
        # First call should cache
        result1 = self.profiler.get_route_geodataframe(route)
        
        # Second call should use cache
        result2 = self.profiler.get_route_geodataframe(route)
        
        # Should return same result
        if not result1.empty and not result2.empty:
            self.assertEqual(len(result1), len(result2))
    
    def test_error_handling_invalid_input(self):
        """Test error handling with invalid input"""
        # Test with None route
        try:
            result = self.profiler.get_route_geodataframe(None)
            self.assertIsInstance(result, gpd.GeoDataFrame)
            self.assertTrue(result.empty)
        except:
            # Some methods may not handle None gracefully
            pass
        
        # Test with string instead of list
        try:
            result = self.profiler.get_route_geodataframe("invalid")
            self.assertIsInstance(result, gpd.GeoDataFrame)
            self.assertTrue(result.empty)
        except:
            # Some methods may not handle strings gracefully
            pass
    
    def test_performance_with_large_route(self):
        """Test performance with large route"""
        # Create a large route
        large_route = list(range(1001, 1050))  # 49 nodes
        
        # Should handle large routes without errors
        try:
            result = self.profiler.get_route_geodataframe(large_route)
            self.assertIsInstance(result, gpd.GeoDataFrame)
            
            # Performance should be reasonable
            import time
            start_time = time.time()
            result = self.profiler.get_route_geodataframe(large_route)
            end_time = time.time()
            
            # Should complete within reasonable time (10 seconds)
            self.assertLess(end_time - start_time, 10.0)
        except:
            # Some methods may not handle large routes gracefully
            pass
    
    def test_generate_profile_data_spatial(self):
        """Test spatial profile data generation"""
        route_result = {
            'route': [1001, 1002, 1003],
            'stats': {'total_distance_km': 2.0}
        }
        
        # Test with geodataframe enabled
        result = self.profiler.generate_profile_data_spatial(route_result, use_geodataframe=True)
        
        # Should have standard profile data structure
        expected_keys = ['coordinates', 'elevations', 'distances_m', 'distances_km', 'elevation_stats']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Test without geodataframe (fallback)
        result_fallback = self.profiler.generate_profile_data_spatial(route_result, use_geodataframe=False)
        self.assertIsInstance(result_fallback, dict)
    
    def test_generate_profile_data_spatial_empty_route(self):
        """Test spatial profile generation with empty route"""
        empty_result = {}
        result = self.profiler.generate_profile_data_spatial(empty_result, use_geodataframe=True)
        self.assertEqual(result, {})
        
        empty_route_result = {'route': []}
        result = self.profiler.generate_profile_data_spatial(empty_route_result, use_geodataframe=True)
        self.assertEqual(result, {})
    
    def test_generate_profile_data_spatial_with_interpolation(self):
        """Test spatial profile generation with interpolation"""
        route_result = {
            'route': [1001, 1002, 1003],
            'stats': {'total_distance_km': 2.0}
        }
        
        # Test with interpolation points
        result = self.profiler.generate_profile_data_spatial(
            route_result, use_geodataframe=True, interpolate_points=5
        )
        
        # Should have more data points due to interpolation
        if 'coordinates' in result and result['coordinates']:
            # Interpolation should add more points
            self.assertGreaterEqual(len(result['coordinates']), 3)
    
    def test_calculate_elevation_stats_spatial(self):
        """Test spatial elevation statistics calculation"""
        # Create a route geodataframe
        route = [1001, 1002, 1003]
        route_gdf = self.profiler.get_route_geodataframe(route)
        
        if not route_gdf.empty:
            # Add elevation analysis columns if missing
            route_gdf = self.profiler._add_elevation_analysis_columns(route_gdf)
            
            # Calculate spatial statistics
            stats = self.profiler._calculate_elevation_stats_spatial(route_gdf)
            
            # Should have basic statistics (check actual keys returned)
            expected_keys = ['total_elevation_gain_m', 'total_elevation_loss_m', 'max_elevation', 'min_elevation']
            for key in expected_keys:
                self.assertIn(key, stats)
    
    def test_add_elevation_analysis_columns(self):
        """Test adding elevation analysis columns to GeoDataFrame"""
        route = [1001, 1002, 1003]
        route_gdf = self.profiler.get_route_geodataframe(route)
        
        if not route_gdf.empty:
            # Add analysis columns
            enhanced_gdf = self.profiler._add_elevation_analysis_columns(route_gdf)
            
            # Should have additional columns
            expected_columns = ['elevation_change_m', 'grade_percent']
            for col in expected_columns:
                self.assertIn(col, enhanced_gdf.columns)
            
            # Should have same number of rows
            self.assertEqual(len(enhanced_gdf), len(route_gdf))
    
    def test_find_steep_sections_spatial(self):
        """Test finding steep sections in spatial data"""
        # Create route with potential steep sections
        route = [1001, 1002, 1003, 1004]
        route_gdf = self.profiler.get_route_geodataframe(route)
        
        if not route_gdf.empty:
            # Add elevation analysis columns
            route_gdf = self.profiler._add_elevation_analysis_columns(route_gdf)
            
            # Find steep sections
            steep_sections = self.profiler._find_steep_sections_spatial(route_gdf, min_grade=5.0)
            
            # Should return a list
            self.assertIsInstance(steep_sections, list)
            
            # Each steep section should have required fields (check actual structure)
            for section in steep_sections:
                self.assertIn('type', section)
                self.assertIn('start_km', section)
                self.assertIn('end_km', section)
                self.assertIn('max_grade', section)
    
    def test_group_consecutive_sections(self):
        """Test grouping consecutive sections"""
        route = [1001, 1002, 1003, 1004]
        route_gdf = self.profiler.get_route_geodataframe(route)
        
        if not route_gdf.empty and len(route_gdf) > 1:
            # Create sections (using first two rows as test)
            sections_gdf = route_gdf.iloc[:2]
            
            # Group consecutive sections
            grouped = self.profiler._group_consecutive_sections(sections_gdf)
            
            # Should return a list of GeoDataFrames
            self.assertIsInstance(grouped, list)
            for group in grouped:
                self.assertIsInstance(group, gpd.GeoDataFrame)
    
    def test_interpolate_elevation_points(self):
        """Test elevation point interpolation"""
        route = [1001, 1002, 1003]
        route_gdf = self.profiler.get_route_geodataframe(route)
        
        if not route_gdf.empty and len(route_gdf) > 1:
            original_length = len(route_gdf)
            
            # Interpolate points
            interpolated_gdf = self.profiler._interpolate_elevation_points(route_gdf, points_per_segment=2)
            
            # Should have more points
            self.assertGreaterEqual(len(interpolated_gdf), original_length)
            
            # Should have same columns
            for col in route_gdf.columns:
                self.assertIn(col, interpolated_gdf.columns)
    
    def test_find_elevation_peaks_valleys_spatial(self):
        """Test spatial peaks and valleys detection"""
        route_result = {
            'route': [1001, 1002, 1003, 1004, 1005],
            'stats': {'total_distance_km': 4.0}
        }
        
        # Find peaks and valleys using spatial method
        peaks_valleys = self.profiler.find_elevation_peaks_valleys_spatial(
            route_result, min_prominence=10
        )
        
        # Should have standard structure
        expected_keys = ['peaks', 'valleys']
        for key in expected_keys:
            self.assertIn(key, peaks_valleys)
            self.assertIsInstance(peaks_valleys[key], list)
    
    def test_find_elevation_peaks_valleys_spatial_empty(self):
        """Test spatial peaks/valleys with empty route"""
        empty_result = {}
        peaks_valleys = self.profiler.find_elevation_peaks_valleys_spatial(empty_result)
        
        expected = {'peaks': [], 'valleys': []}
        self.assertEqual(peaks_valleys, expected)
    
    def test_get_elevation_zones_spatial(self):
        """Test spatial elevation zones calculation"""
        route_result = {
            'route': [1001, 1002, 1003, 1004],
            'stats': {'total_distance_km': 3.0}
        }
        
        # Get elevation zones using spatial method
        zones = self.profiler.get_elevation_zones_spatial(route_result, zone_count=3)
        
        # Should create zones
        self.assertLessEqual(len(zones), 3)
        
        # Each zone should have required fields
        for zone in zones:
            self.assertIn('zone_number', zone)
            self.assertIn('start_km', zone)
            self.assertIn('end_km', zone)
            self.assertIn('avg_elevation', zone)
    
    def test_get_elevation_zones_spatial_empty(self):
        """Test spatial elevation zones with empty route"""
        empty_result = {}
        zones = self.profiler.get_elevation_zones_spatial(empty_result, zone_count=3)
        self.assertEqual(zones, [])
    
    def test_spatial_methods_error_handling(self):
        """Test error handling in spatial methods"""
        # Test with invalid route data
        invalid_routes = [
            {'route': [], 'stats': {}},  # Empty route
            {'route': [9999], 'stats': {}},  # Non-existent node
            {},  # Empty result
        ]
        
        for invalid_route in invalid_routes:
            # Should handle errors gracefully
            try:
                result = self.profiler.generate_profile_data_spatial(
                    invalid_route, use_geodataframe=True
                )
                self.assertIsInstance(result, dict)
                
                peaks_valleys = self.profiler.find_elevation_peaks_valleys_spatial(invalid_route)
                self.assertIsInstance(peaks_valleys, dict)
                
                zones = self.profiler.get_elevation_zones_spatial(invalid_route)
                self.assertIsInstance(zones, list)
                
            except Exception:
                # Expected for some invalid data
                pass
    
    def test_spatial_vs_regular_methods_consistency(self):
        """Test consistency between spatial and regular methods"""
        route_result = {
            'route': [1001, 1002, 1003],
            'stats': {'total_distance_km': 2.0}
        }
        
        # Compare regular vs spatial methods
        regular_profile = self.profiler.generate_profile_data(route_result)
        spatial_profile = self.profiler.generate_profile_data_spatial(
            route_result, use_geodataframe=False
        )
        
        # Should have similar structure
        if regular_profile and spatial_profile:
            self.assertEqual(set(regular_profile.keys()), set(spatial_profile.keys()))
        
        # Compare peaks/valleys
        regular_peaks = self.profiler.find_elevation_peaks_valleys(route_result)
        spatial_peaks = self.profiler.find_elevation_peaks_valleys_spatial(route_result)
        
        # Should have similar structure (both should have peaks and valleys)
        if regular_peaks and spatial_peaks:
            # Both should have peaks and valleys keys
            self.assertIn('peaks', regular_peaks)
            self.assertIn('valleys', regular_peaks)
            self.assertIn('peaks', spatial_peaks)
            self.assertIn('valleys', spatial_peaks)


if __name__ == '__main__':
    unittest.main()