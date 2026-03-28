#!/usr/bin/env python3
"""
Unit tests for consolidated elevation services.

Tests both utils.elevation (data sources) and route_services.elevation_service (profiler).
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import networkx as nx
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.elevation import (
    ElevationService as UtilsElevationService,
    SRTMElevationSource,
    LocalThreeDEPSource,
    HybridElevationSource,
    get_elevation_service
)
from route_services.elevation_service import ElevationProfiler


class TestSRTMElevationSource(unittest.TestCase):
    """Test cases for SRTM elevation source"""

    @patch('utils.elevation.RASTERIO_AVAILABLE', True)
    @patch('utils.elevation.rasterio')
    def test_initialization_success(self, mock_rasterio):
        """Test SRTM source initialization"""
        mock_dataset = MagicMock()
        mock_dataset.bounds = MagicMock(left=-81, right=-80, bottom=37, top=38)
        mock_dataset.nodata = -9999
        mock_rasterio.open.return_value = mock_dataset

        with patch('os.path.exists', return_value=True):
            source = SRTMElevationSource('/fake/path.tif')
            self.assertEqual(source.resolution, 90.0)
            self.assertIsNotNone(source._dataset)

    @patch('utils.elevation.RASTERIO_AVAILABLE', True)
    @patch('utils.elevation.rasterio')
    def test_get_elevation_success(self, mock_rasterio):
        """Test successful elevation retrieval"""
        mock_dataset = MagicMock()
        mock_dataset.bounds = MagicMock(left=-81, right=-80, bottom=37, top=38)
        mock_dataset.nodata = -9999
        mock_dataset.sample.return_value = [[650.0]]
        mock_rasterio.open.return_value = mock_dataset

        with patch('os.path.exists', return_value=True):
            source = SRTMElevationSource('/fake/path.tif')
            elevation = source.get_elevation(37.13, -80.41)
            self.assertEqual(elevation, 650.0)

    @patch('utils.elevation.RASTERIO_AVAILABLE', True)
    @patch('utils.elevation.rasterio')
    def test_get_elevation_nodata(self, mock_rasterio):
        """Test elevation retrieval with nodata value"""
        mock_dataset = MagicMock()
        mock_dataset.bounds = MagicMock(left=-81, right=-80, bottom=37, top=38)
        mock_dataset.nodata = -9999
        mock_dataset.sample.return_value = [[-9999]]  # nodata value
        mock_rasterio.open.return_value = mock_dataset

        with patch('os.path.exists', return_value=True):
            source = SRTMElevationSource('/fake/path.tif')
            elevation = source.get_elevation(37.13, -80.41)
            self.assertIsNone(elevation)

    @patch('utils.elevation.RASTERIO_AVAILABLE', True)
    @patch('utils.elevation.rasterio')
    def test_is_available(self, mock_rasterio):
        """Test availability check"""
        mock_dataset = MagicMock()
        mock_dataset.bounds = MagicMock(left=-81, right=-80, bottom=37, top=38)
        mock_rasterio.open.return_value = mock_dataset

        with patch('os.path.exists', return_value=True):
            source = SRTMElevationSource('/fake/path.tif')
            self.assertTrue(source.is_available(37.5, -80.5))
            self.assertFalse(source.is_available(40.0, -80.5))  # Outside bounds

    @patch('utils.elevation.RASTERIO_AVAILABLE', True)
    @patch('utils.elevation.rasterio')
    def test_get_elevation_profile(self, mock_rasterio):
        """Test elevation profile retrieval"""
        mock_dataset = MagicMock()
        mock_dataset.bounds = MagicMock(left=-81, right=-80, bottom=37, top=38)
        mock_dataset.nodata = -9999
        mock_dataset.sample.side_effect = [[[600.0]], [[620.0]], [[640.0]]]
        mock_rasterio.open.return_value = mock_dataset

        with patch('os.path.exists', return_value=True):
            source = SRTMElevationSource('/fake/path.tif')
            coordinates = [(37.13, -80.41), (37.14, -80.42), (37.15, -80.43)]
            profile = source.get_elevation_profile(coordinates)
            self.assertEqual(len(profile), 3)
            self.assertEqual(profile, [600.0, 620.0, 640.0])


class TestHybridElevationSource(unittest.TestCase):
    """Test cases for hybrid elevation source"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_primary = Mock(spec=SRTMElevationSource)
        self.mock_fallback = Mock(spec=SRTMElevationSource)
        self.mock_primary.get_resolution.return_value = 1.0
        self.mock_fallback.get_resolution.return_value = 90.0

    def test_initialization(self):
        """Test hybrid source initialization"""
        hybrid = HybridElevationSource(self.mock_primary, self.mock_fallback)
        self.assertEqual(hybrid.resolution, 1.0)  # Minimum of two
        self.assertEqual(hybrid.stats['primary_queries'], 0)

    def test_get_elevation_primary_success(self):
        """Test elevation retrieval from primary source"""
        self.mock_primary.is_available.return_value = True
        self.mock_primary.get_elevation.return_value = 650.0

        hybrid = HybridElevationSource(self.mock_primary, self.mock_fallback)
        elevation = hybrid.get_elevation(37.13, -80.41)

        self.assertEqual(elevation, 650.0)
        self.assertEqual(hybrid.stats['primary_queries'], 1)
        self.assertEqual(hybrid.stats['fallback_queries'], 0)

    def test_get_elevation_fallback(self):
        """Test elevation retrieval falling back to secondary source"""
        self.mock_primary.is_available.return_value = False
        self.mock_fallback.is_available.return_value = True
        self.mock_fallback.get_elevation.return_value = 640.0

        hybrid = HybridElevationSource(self.mock_primary, self.mock_fallback)
        elevation = hybrid.get_elevation(37.13, -80.41)

        self.assertEqual(elevation, 640.0)
        self.assertEqual(hybrid.stats['primary_queries'], 0)
        self.assertEqual(hybrid.stats['fallback_queries'], 1)

    def test_get_elevation_both_unavailable(self):
        """Test elevation retrieval when both sources unavailable"""
        self.mock_primary.is_available.return_value = False
        self.mock_fallback.is_available.return_value = False

        hybrid = HybridElevationSource(self.mock_primary, self.mock_fallback)
        elevation = hybrid.get_elevation(37.13, -80.41)

        self.assertIsNone(elevation)
        self.assertEqual(hybrid.stats['failed_queries'], 1)

    def test_is_available_either_source(self):
        """Test availability check for hybrid source"""
        hybrid = HybridElevationSource(self.mock_primary, self.mock_fallback)

        # Primary available
        self.mock_primary.is_available.return_value = True
        self.mock_fallback.is_available.return_value = False
        self.assertTrue(hybrid.is_available(37.13, -80.41))

        # Fallback available
        self.mock_primary.is_available.return_value = False
        self.mock_fallback.is_available.return_value = True
        self.assertTrue(hybrid.is_available(37.13, -80.41))

        # Neither available
        self.mock_primary.is_available.return_value = False
        self.mock_fallback.is_available.return_value = False
        self.assertFalse(hybrid.is_available(37.13, -80.41))

    def test_get_stats(self):
        """Test statistics retrieval"""
        self.mock_primary.is_available.return_value = True
        self.mock_primary.get_elevation.return_value = 650.0

        hybrid = HybridElevationSource(self.mock_primary, self.mock_fallback)

        # Make some queries
        hybrid.get_elevation(37.13, -80.41)
        hybrid.get_elevation(37.14, -80.42)

        stats = hybrid.get_stats()
        self.assertEqual(stats['primary_queries'], 2)
        self.assertEqual(stats['primary_percentage'], 100.0)


class TestUtilsElevationService(unittest.TestCase):
    """Test cases for utils.elevation.ElevationService"""

    @patch('utils.elevation.os.path.exists')
    @patch('utils.elevation.RASTERIO_AVAILABLE', True)
    def test_initialization_srtm_only(self, mock_exists):
        """Test service initialization with SRTM only"""
        mock_exists.side_effect = lambda path: 'srtm' in path

        with patch('utils.elevation.SRTMElevationSource') as mock_srtm:
            service = UtilsElevationService(use_3dep=False, use_srtm=True)
            self.assertIsNotNone(service.active_source)

    @patch('utils.elevation.os.path.exists', return_value=True)
    @patch('utils.elevation.LocalThreeDEPSource')
    @patch('utils.elevation.SRTMElevationSource')
    def test_initialization_hybrid(self, mock_srtm, mock_3dep, mock_exists):
        """Test service initialization with hybrid sources"""
        # Mock 3DEP with tiles
        mock_3dep_instance = Mock()
        mock_3dep_instance.tile_index = {'tile1': {}}
        mock_3dep_instance.get_resolution.return_value = 1.0
        mock_3dep.return_value = mock_3dep_instance

        # Mock SRTM
        mock_srtm_instance = Mock()
        mock_srtm_instance.get_resolution.return_value = 90.0
        mock_srtm.return_value = mock_srtm_instance

        service = UtilsElevationService(use_3dep=True, use_srtm=True)
        self.assertIsNotNone(service.active_source)

    def test_get_elevation_no_source(self):
        """Test elevation retrieval with no sources available"""
        with patch('utils.elevation.os.path.exists', return_value=False):
            service = UtilsElevationService()
            elevation = service.get_elevation(37.13, -80.41)
            self.assertIsNone(elevation)

    @patch('utils.elevation.os.path.exists', return_value=True)
    @patch('utils.elevation.SRTMElevationSource')
    def test_add_elevation_to_graph(self, mock_srtm, mock_exists):
        """Test adding elevation to graph"""
        # Mock SRTM source
        mock_srtm_instance = Mock()
        mock_srtm_instance.get_elevation.return_value = 650.0
        mock_srtm.return_value = mock_srtm_instance

        service = UtilsElevationService(use_3dep=False, use_srtm=True)
        service.active_source = mock_srtm_instance

        # Create test graph
        graph = nx.Graph()
        graph.add_node(1, x=-80.41, y=37.13)
        graph.add_node(2, x=-80.42, y=37.14)

        service.add_elevation_to_graph(graph)

        # Check elevations were added
        self.assertEqual(graph.nodes[1]['elevation'], 650.0)
        self.assertEqual(graph.nodes[2]['elevation'], 650.0)


class TestElevationProfiler(unittest.TestCase):
    """Test cases for route_services.elevation_service.ElevationProfiler"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock graph with varying elevations
        self.mock_graph = nx.Graph()
        self.mock_graph.add_node(1001, x=-80.4094, y=37.1299, elevation=600)
        self.mock_graph.add_node(1002, x=-80.4095, y=37.1300, elevation=620)
        self.mock_graph.add_node(1003, x=-80.4096, y=37.1301, elevation=610)
        self.mock_graph.add_node(1004, x=-80.4097, y=37.1302, elevation=650)
        self.mock_graph.add_node(1005, x=-80.4098, y=37.1303, elevation=630)

        # Add edges with length
        self.mock_graph.add_edge(1001, 1002, length=100)
        self.mock_graph.add_edge(1002, 1003, length=100)
        self.mock_graph.add_edge(1003, 1004, length=100)
        self.mock_graph.add_edge(1004, 1005, length=100)
        self.mock_graph.add_edge(1005, 1001, length=100)

        # Create profiler with mock elevation service
        self.mock_elevation_service = Mock()
        self.mock_elevation_service.is_available.return_value = False  # Don't use enhanced elevation in tests
        self.profiler = ElevationProfiler(self.mock_graph, self.mock_elevation_service)

        # Sample route result
        self.sample_route_result = {
            'route': [1001, 1002, 1003, 1004, 1005],
            'stats': {
                'total_distance_km': 0.5,
                'total_elevation_gain_m': 60
            }
        }

    def test_initialization(self):
        """Test profiler initialization"""
        profiler = ElevationProfiler(self.mock_graph)
        self.assertEqual(profiler.graph, self.mock_graph)
        self.assertIsNotNone(profiler.elevation_service)

    def test_generate_profile_data_success(self):
        """Test successful profile generation"""
        profile_data = self.profiler.generate_profile_data(
            self.sample_route_result,
            use_enhanced_elevation=False
        )

        # Check structure
        self.assertIn('coordinates', profile_data)
        self.assertIn('elevations', profile_data)
        self.assertIn('distances_m', profile_data)
        self.assertIn('distances_km', profile_data)
        self.assertIn('elevation_stats', profile_data)

        # Check elevations (includes return to start)
        elevations = profile_data['elevations']
        expected_elevations = [600, 620, 610, 650, 630, 600]
        self.assertEqual(elevations, expected_elevations)

        # Check distances
        distances_m = profile_data['distances_m']
        expected_distances = [0, 100, 200, 300, 400, 500]
        self.assertEqual(distances_m, expected_distances)

    def test_generate_profile_data_empty(self):
        """Test profile generation with empty route"""
        result = self.profiler.generate_profile_data({})
        self.assertEqual(result, {})

        result = self.profiler.generate_profile_data({'route': []})
        self.assertEqual(result, {})

    def test_calculate_elevation_stats(self):
        """Test elevation statistics calculation"""
        elevations = [600, 620, 610, 650, 630]
        distances_km = [0, 0.1, 0.2, 0.3, 0.4]

        stats = self.profiler._calculate_elevation_stats(elevations, distances_km)

        # Check basic stats
        self.assertEqual(stats['min_elevation'], 600)
        self.assertEqual(stats['max_elevation'], 650)
        self.assertEqual(stats['elevation_range'], 50)

        # Check that stats exist
        self.assertIn('total_elevation_gain_m', stats)
        self.assertIn('total_elevation_loss_m', stats)
        self.assertIn('difficulty_score', stats)
        self.assertIn('terrain_type', stats)

    def test_calculate_difficulty_score(self):
        """Test difficulty score calculation"""
        score = self.profiler._calculate_difficulty_score(
            gain=100, loss=50, max_grade=15, distance_km=5.0
        )

        self.assertIn('score', score)
        self.assertIn('category', score)
        self.assertIn('gain_per_km', score)
        self.assertGreater(score['score'], 0)
        self.assertIn(score['category'], ['easy', 'moderate', 'hard', 'very_hard', 'extreme'])

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
            self.assertIn('min_elevation', zone)
            self.assertIn('max_elevation', zone)

    def test_get_elevation_zones_empty(self):
        """Test elevation zones with empty route"""
        zones = self.profiler.get_elevation_zones({})
        self.assertEqual(zones, [])

    def test_find_elevation_peaks_valleys(self):
        """Test peak and valley detection"""
        result = self.profiler.find_elevation_peaks_valleys(
            self.sample_route_result,
            min_prominence=10
        )

        self.assertIn('peaks', result)
        self.assertIn('valleys', result)
        self.assertIn('peak_count', result)
        self.assertIn('valley_count', result)

        # Should detect peak at 650m and valley at 610m
        self.assertGreater(result['peak_count'], 0)

    def test_get_climbing_segments(self):
        """Test climbing segment identification"""
        segments = self.profiler.get_climbing_segments(
            self.sample_route_result,
            min_gain=15
        )

        # Should identify climbing segments
        self.assertIsInstance(segments, list)
        if segments:
            segment = segments[0]
            self.assertIn('start_km', segment)
            self.assertIn('end_km', segment)
            self.assertIn('elevation_gain', segment)
            self.assertIn('avg_grade', segment)

    def test_get_detailed_route_path(self):
        """Test detailed route path generation"""
        path = self.profiler.get_detailed_route_path(self.sample_route_result)

        # Should include all nodes plus intermediate nodes
        self.assertIsInstance(path, list)
        self.assertGreater(len(path), 0)

        # Check first node structure
        first_node = path[0]
        self.assertIn('latitude', first_node)
        self.assertIn('longitude', first_node)
        self.assertIn('elevation', first_node)
        self.assertIn('node_type', first_node)

    def test_get_network_distance(self):
        """Test network distance calculation"""
        distance = self.profiler._get_network_distance(1001, 1002)
        self.assertEqual(distance, 100)

        # Test caching
        distance2 = self.profiler._get_network_distance(1001, 1002)
        self.assertEqual(distance2, 100)

        # Test same node
        distance_same = self.profiler._get_network_distance(1001, 1001)
        self.assertEqual(distance_same, 0)


class TestElevationServiceIntegration(unittest.TestCase):
    """Integration tests for elevation services"""

    def test_end_to_end_profile_generation(self):
        """Test complete profile generation workflow"""
        # Create test graph
        graph = nx.Graph()
        graph.add_node(1, x=-80.41, y=37.13, elevation=600)
        graph.add_node(2, x=-80.42, y=37.14, elevation=620)
        graph.add_node(3, x=-80.43, y=37.15, elevation=610)
        graph.add_edge(1, 2, length=100)
        graph.add_edge(2, 3, length=100)
        graph.add_edge(3, 1, length=100)

        # Create profiler
        profiler = ElevationProfiler(graph)

        # Generate profile
        route_result = {'route': [1, 2, 3]}
        profile = profiler.generate_profile_data(
            route_result,
            use_enhanced_elevation=False
        )

        # Validate profile
        self.assertIn('elevations', profile)
        self.assertIn('distances_km', profile)
        self.assertIn('elevation_stats', profile)
        self.assertGreater(len(profile['elevations']), 0)


if __name__ == '__main__':
    unittest.main()
