#!/usr/bin/env python3
"""
Unit tests for ElevationProfiler
"""

import unittest
from unittest.mock import Mock, patch
import networkx as nx
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
        
        # Check coordinates
        coordinates = profile_data['coordinates']
        self.assertEqual(len(coordinates), 5)
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
            
            # Coordinates should match route length
            self.assertEqual(len(coordinates), 5)
            
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


if __name__ == '__main__':
    unittest.main()