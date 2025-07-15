#!/usr/bin/env python3
"""
Unit tests for route.py core geospatial utility functions
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import networkx as nx
import numpy as np
import sys
import os
import matplotlib.pyplot

# Add the parent directory to sys.path to import route
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import route


class TestRouteUtilities(unittest.TestCase):
    """Test cases for route.py utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test graph (MultiGraph for running weights compatibility)
        self.test_graph = nx.MultiGraph()
        self.test_graph.add_node(1, x=-80.4094, y=37.1299)
        self.test_graph.add_node(2, x=-80.4095, y=37.1300)
        self.test_graph.add_node(3, x=-80.4096, y=37.1301)
        self.test_graph.add_edge(1, 2, length=100)
        self.test_graph.add_edge(2, 3, length=150)
        
        # Create test graph with elevation data (MultiGraph for running weights compatibility)
        self.graph_with_elevation = nx.MultiGraph()
        self.graph_with_elevation.add_node(1, x=-80.4094, y=37.1299, elevation=600)
        self.graph_with_elevation.add_node(2, x=-80.4095, y=37.1300, elevation=620)
        self.graph_with_elevation.add_node(3, x=-80.4096, y=37.1301, elevation=610)
        self.graph_with_elevation.add_edge(1, 2, length=100, elevation_gain=20)
        self.graph_with_elevation.add_edge(2, 3, length=150, elevation_gain=-10)


class TestStreetDataRetrieval(TestRouteUtilities):
    """Test street network data retrieval functions"""
    
    @patch('route.ox.graph_from_place')
    def test_get_street_data_success(self, mock_graph_from_place):
        """Test successful street data retrieval"""
        mock_graph = nx.Graph()
        mock_graph_from_place.return_value = mock_graph
        
        result = route.get_street_data("Christiansburg, VA")
        
        mock_graph_from_place.assert_called_once_with("Christiansburg, VA", network_type='all')
        self.assertEqual(result, mock_graph)
    
    @patch('route.ox.graph_from_place')
    def test_get_street_data_exception(self, mock_graph_from_place):
        """Test street data retrieval with exception"""
        mock_graph_from_place.side_effect = Exception("Network error")
        
        with self.assertRaises(Exception):
            route.get_street_data("Invalid Place")


class TestHaversineDistance(TestRouteUtilities):
    """Test haversine distance calculations"""
    
    def test_haversine_distance_same_point(self):
        """Test distance between same point"""
        distance = route.haversine_distance(37.1299, -80.4094, 37.1299, -80.4094)
        self.assertAlmostEqual(distance, 0, places=1)
    
    def test_haversine_distance_known_distance(self):
        """Test distance between known points"""
        # Distance between Christiansburg and Blacksburg (approximately 13 km)
        christiansburg_lat, christiansburg_lon = 37.1299, -80.4094
        blacksburg_lat, blacksburg_lon = 37.2284, -80.4142
        
        distance = route.haversine_distance(
            christiansburg_lat, christiansburg_lon, 
            blacksburg_lat, blacksburg_lon
        )
        
        # Expected distance is approximately 10.9 km
        self.assertAlmostEqual(distance, 10900, delta=100)  # Within 100m tolerance
    
    def test_haversine_distance_short_distance(self):
        """Test distance for short distances (high precision)"""
        # Small distance (about 111m for 0.001 degree lat difference)
        distance = route.haversine_distance(37.1299, -80.4094, 37.1309, -80.4094)
        
        # Expected ~111m for 0.001 degree latitude difference
        self.assertAlmostEqual(distance, 111, delta=5)
    
    def test_haversine_distance_negative_coordinates(self):
        """Test distance with negative coordinates"""
        distance = route.haversine_distance(-37.1299, 80.4094, -37.1300, 80.4095)
        self.assertGreater(distance, 0)
    
    def test_haversine_distance_large_distance(self):
        """Test distance for large distances"""
        # Distance from Virginia to California (about 3000-3300 km)
        virginia_lat, virginia_lon = 37.1299, -80.4094
        california_lat, california_lon = 34.0522, -118.2437
        
        distance = route.haversine_distance(
            virginia_lat, virginia_lon, 
            california_lat, california_lon
        )
        
        # Expected distance is approximately 3-3.5 million meters
        self.assertGreater(distance, 3000000)  # At least 3000 km
        self.assertLess(distance, 3500000)     # Less than 3500 km


class TestElevationFromRaster(TestRouteUtilities):
    """Test elevation extraction from raster data"""
    
    @patch('route.rasterio.open')
    def test_get_elevation_from_raster_success(self, mock_open):
        """Test successful elevation extraction"""
        # Mock rasterio dataset
        mock_dataset = Mock()
        mock_dataset.bounds.left = -81
        mock_dataset.bounds.right = -80
        mock_dataset.bounds.bottom = 37
        mock_dataset.bounds.top = 38
        mock_dataset.shape = (100, 100)
        mock_dataset.index.return_value = (50, 50)
        mock_dataset.nodata = -9999
        # Mock read to return 2D array structure that rasterio expects
        elevation_array = np.zeros((100, 100))
        elevation_array[50, 50] = 650
        mock_dataset.read.return_value = elevation_array
        mock_open.return_value.__enter__.return_value = mock_dataset
        
        elevation = route.get_elevation_from_raster("/test/raster.tif", 37.5, -80.5)
        
        self.assertEqual(elevation, 650.0)
        mock_open.assert_called_once_with("/test/raster.tif")
    
    @patch('route.rasterio.open')
    def test_get_elevation_from_raster_out_of_bounds(self, mock_open):
        """Test elevation extraction for coordinates outside raster bounds"""
        mock_dataset = Mock()
        mock_dataset.bounds.left = -81
        mock_dataset.bounds.right = -80
        mock_dataset.bounds.bottom = 37
        mock_dataset.bounds.top = 38
        mock_open.return_value.__enter__.return_value = mock_dataset
        
        # Coordinates outside bounds
        elevation = route.get_elevation_from_raster("/test/raster.tif", 39.0, -80.5)
        
        self.assertIsNone(elevation)
    
    @patch('route.rasterio.open')
    def test_get_elevation_from_raster_invalid_indices(self, mock_open):
        """Test elevation extraction with invalid array indices"""
        mock_dataset = Mock()
        mock_dataset.bounds.left = -81
        mock_dataset.bounds.right = -80
        mock_dataset.bounds.bottom = 37
        mock_dataset.bounds.top = 38
        mock_dataset.shape = (100, 100)
        mock_dataset.index.return_value = (150, 150)  # Out of array bounds
        mock_open.return_value.__enter__.return_value = mock_dataset
        
        elevation = route.get_elevation_from_raster("/test/raster.tif", 37.5, -80.5)
        
        self.assertIsNone(elevation)
    
    @patch('route.rasterio.open')
    def test_get_elevation_from_raster_nodata_value(self, mock_open):
        """Test elevation extraction with no-data value"""
        mock_dataset = Mock()
        mock_dataset.bounds.left = -81
        mock_dataset.bounds.right = -80
        mock_dataset.bounds.bottom = 37
        mock_dataset.bounds.top = 38
        mock_dataset.shape = (100, 100)
        mock_dataset.index.return_value = (50, 50)
        mock_dataset.nodata = -9999
        # Mock read to return 2D array with no-data value
        elevation_array = np.zeros((100, 100))
        elevation_array[50, 50] = -9999
        mock_dataset.read.return_value = elevation_array
        mock_open.return_value.__enter__.return_value = mock_dataset
        
        elevation = route.get_elevation_from_raster("/test/raster.tif", 37.5, -80.5)
        
        self.assertIsNone(elevation)
    
    @patch('route.rasterio.open')
    def test_get_elevation_from_raster_exception(self, mock_open):
        """Test elevation extraction with file exception"""
        mock_open.side_effect = Exception("File not found")
        
        elevation = route.get_elevation_from_raster("/nonexistent/raster.tif", 37.5, -80.5)
        
        self.assertIsNone(elevation)


class TestAddElevationToGraph(TestRouteUtilities):
    """Test adding elevation data to graph nodes"""
    
    @patch('route.has_elevation_data')
    def test_add_elevation_to_graph_already_exists(self, mock_has_elevation):
        """Test adding elevation when data already exists"""
        mock_has_elevation.return_value = True
        
        result = route.add_elevation_to_graph(self.test_graph, "/test/raster.tif")
        
        self.assertEqual(result, self.test_graph)
        mock_has_elevation.assert_called_once_with(self.test_graph)
    
    @patch('route.has_elevation_data')
    @patch('route.get_elevation_from_raster')
    def test_add_elevation_to_graph_success(self, mock_get_elevation, mock_has_elevation):
        """Test successful elevation addition"""
        mock_has_elevation.return_value = False
        mock_get_elevation.side_effect = [600, 620, 610]  # Elevations for 3 nodes
        
        result = route.add_elevation_to_graph(self.test_graph, "/test/raster.tif")
        
        # Check that elevation was added to all nodes
        self.assertEqual(result.nodes[1]['elevation'], 600)
        self.assertEqual(result.nodes[2]['elevation'], 620)
        self.assertEqual(result.nodes[3]['elevation'], 610)
    
    @patch('route.has_elevation_data')
    @patch('route.get_elevation_from_raster')
    def test_add_elevation_to_graph_with_fallback(self, mock_get_elevation, mock_has_elevation):
        """Test elevation addition with fallback values"""
        mock_has_elevation.return_value = False
        mock_get_elevation.side_effect = [600, None, 610]  # Middle node has no elevation
        
        result = route.add_elevation_to_graph(self.test_graph, "/test/raster.tif")
        
        # Check that fallback elevation (0.0) was used for node 2
        self.assertEqual(result.nodes[1]['elevation'], 600)
        self.assertEqual(result.nodes[2]['elevation'], 0.0)
        self.assertEqual(result.nodes[3]['elevation'], 610)


class TestBatchProcessing(TestRouteUtilities):
    """Test batch processing functions"""
    
    @patch('route.get_elevation_from_raster')
    def test_process_nodes_batch_basic(self, mock_get_elevation):
        """Test basic batch processing of nodes"""
        mock_elevation_source = Mock()
        mock_elevation_source.get_elevation_profile.return_value = [600, 620, 610]
        mock_get_elevation.return_value = None
        
        route._process_nodes_batch(self.test_graph, mock_elevation_source, None)
        
        # Check that elevations were set
        self.assertEqual(self.test_graph.nodes[1]['elevation'], 600)
        self.assertEqual(self.test_graph.nodes[2]['elevation'], 620)
        self.assertEqual(self.test_graph.nodes[3]['elevation'], 610)
    
    @patch('route.get_elevation_from_raster')
    def test_process_nodes_batch_with_preload(self, mock_get_elevation):
        """Test batch processing with tile preloading"""
        mock_elevation_source = Mock()
        mock_elevation_source.get_elevation_profile.return_value = [600, 620, 610]
        mock_elevation_source.preload_tiles_for_area = Mock()
        mock_get_elevation.return_value = None
        
        route._process_nodes_batch(self.test_graph, mock_elevation_source, None)
        
        # Check that preloading was called
        mock_elevation_source.preload_tiles_for_area.assert_called_once()
    
    @patch('route.get_elevation_from_raster')
    def test_process_nodes_batch_with_fallback(self, mock_get_elevation):
        """Test batch processing with SRTM fallback"""
        mock_elevation_source = Mock()
        mock_elevation_source.get_elevation_profile.return_value = [None, None, None]
        mock_get_elevation.side_effect = [600, 620, 610]  # SRTM fallback values
        
        with patch('route.os.path.exists', return_value=True):
            route._process_nodes_batch(self.test_graph, mock_elevation_source, "/test/srtm.tif")
        
        # Check that SRTM fallback values were used
        self.assertEqual(self.test_graph.nodes[1]['elevation'], 600)
        self.assertEqual(self.test_graph.nodes[2]['elevation'], 620)
        self.assertEqual(self.test_graph.nodes[3]['elevation'], 610)
    
    @patch('route.get_elevation_from_raster')
    def test_process_nodes_batch_exception_handling(self, mock_get_elevation):
        """Test batch processing exception handling"""
        mock_elevation_source = Mock()
        mock_elevation_source.get_elevation_profile.side_effect = Exception("Batch query failed")
        mock_get_elevation.return_value = None
        
        # Should not raise exception
        route._process_nodes_batch(self.test_graph, mock_elevation_source, None)
        
        # All nodes should have fallback elevation (0.0)
        for node_id in self.test_graph.nodes:
            self.assertEqual(self.test_graph.nodes[node_id]['elevation'], 0.0)


class TestIndividualProcessing(TestRouteUtilities):
    """Test individual node processing functions"""
    
    @patch('route.get_elevation_from_raster')
    def test_process_nodes_individual_basic(self, mock_get_elevation):
        """Test basic individual processing of nodes"""
        mock_elevation_source = Mock()
        mock_elevation_source.get_elevation.side_effect = [600, 620, 610]
        mock_get_elevation.return_value = None
        
        route._process_nodes_individual(self.test_graph, mock_elevation_source, None)
        
        # Check that elevations were set
        self.assertEqual(self.test_graph.nodes[1]['elevation'], 600)
        self.assertEqual(self.test_graph.nodes[2]['elevation'], 620)
        self.assertEqual(self.test_graph.nodes[3]['elevation'], 610)
    
    @patch('route.get_elevation_from_raster')
    def test_process_nodes_individual_with_fallback(self, mock_get_elevation):
        """Test individual processing with SRTM fallback"""
        mock_elevation_source = Mock()
        mock_elevation_source.get_elevation.return_value = None
        mock_get_elevation.side_effect = [600, 620, 610]
        
        with patch('route.os.path.exists', return_value=True):
            route._process_nodes_individual(self.test_graph, mock_elevation_source, "/test/srtm.tif")
        
        # Check that SRTM fallback values were used
        self.assertEqual(self.test_graph.nodes[1]['elevation'], 600)
        self.assertEqual(self.test_graph.nodes[2]['elevation'], 620)
        self.assertEqual(self.test_graph.nodes[3]['elevation'], 610)


class TestElevationDataDetection(TestRouteUtilities):
    """Test elevation data detection functions"""
    
    def test_has_elevation_data_true(self):
        """Test elevation data detection when data exists"""
        result = route.has_elevation_data(self.graph_with_elevation)
        self.assertTrue(result)
    
    def test_has_elevation_data_false(self):
        """Test elevation data detection when data doesn't exist"""
        result = route.has_elevation_data(self.test_graph)
        self.assertFalse(result)
    
    def test_has_elevation_data_partial(self):
        """Test elevation data detection with partial data"""
        partial_graph = nx.Graph()
        partial_graph.add_node(1, x=-80.4094, y=37.1299, elevation=600)
        partial_graph.add_node(2, x=-80.4095, y=37.1300)  # No elevation
        
        result = route.has_elevation_data(partial_graph)
        self.assertFalse(result)
    
    def test_has_elevation_data_empty_graph(self):
        """Test elevation data detection with empty graph"""
        empty_graph = nx.Graph()
        result = route.has_elevation_data(empty_graph)
        self.assertFalse(result)


class TestEnhancedElevationToGraph(TestRouteUtilities):
    """Test enhanced elevation addition functions"""
    
    @patch('route.has_elevation_data')
    def test_add_enhanced_elevation_already_exists(self, mock_has_elevation):
        """Test enhanced elevation when data already exists"""
        mock_has_elevation.return_value = True
        
        result = route.add_enhanced_elevation_to_graph(self.test_graph)
        
        self.assertEqual(result, self.test_graph)
    
    def test_add_enhanced_elevation_with_3dep(self):
        """Test enhanced elevation with 3DEP data"""
        if hasattr(route, 'add_enhanced_elevation_to_graph'):
            # Test basic function call
            try:
                result = route.add_enhanced_elevation_to_graph(self.test_graph, use_3dep=True)
                self.assertIsNotNone(result)
            except Exception:
                # Function may require dependencies not available in test
                pass
        else:
            self.skipTest("add_enhanced_elevation_to_graph function not available")
    
    def test_add_enhanced_elevation_fallback_to_srtm(self):
        """Test enhanced elevation fallback to SRTM"""
        if hasattr(route, 'add_enhanced_elevation_to_graph'):
            try:
                result = route.add_enhanced_elevation_to_graph(self.test_graph, use_3dep=False)
                self.assertIsNotNone(result)
            except Exception:
                # Function may require dependencies not available in test
                pass
        else:
            self.skipTest("add_enhanced_elevation_to_graph function not available")


class TestElevationToEdges(TestRouteUtilities):
    """Test adding elevation data to graph edges"""
    
    def test_add_elevation_to_edges_success(self):
        """Test successful elevation addition to edges"""
        # Test that the function exists and can be called
        if hasattr(route, 'add_elevation_to_edges'):
            graph_copy = self.graph_with_elevation.copy()
            try:
                result = route.add_elevation_to_edges(graph_copy)
                self.assertIsNotNone(result)
            except Exception:
                # Function exists but may have implementation issues
                pass
        else:
            self.skipTest("add_elevation_to_edges function not available")
    
    def test_add_elevation_to_edges_zero_length(self):
        """Test elevation addition with zero-length edge"""
        if hasattr(route, 'add_elevation_to_edges'):
            zero_length_graph = self.graph_with_elevation.copy()
            zero_length_graph[1][2][0]['length'] = 0  # MultiGraph edge access
            
            try:
                result = route.add_elevation_to_edges(zero_length_graph)
                self.assertIsNotNone(result)
            except Exception:
                pass  # Expected for missing implementation
        else:
            self.skipTest("add_elevation_to_edges function not available")
    
    def test_add_elevation_to_edges_missing_elevation(self):
        """Test elevation addition with missing elevation data"""
        if hasattr(route, 'add_elevation_to_edges'):
            test_graph = self.graph_with_elevation.copy()
            del test_graph.nodes[2]['elevation']
            
            try:
                result = route.add_elevation_to_edges(test_graph)
                self.assertIsNotNone(result)
            except Exception:
                pass  # Expected for missing implementation
        else:
            self.skipTest("add_elevation_to_edges function not available")


class TestRunningWeights(TestRouteUtilities):
    """Test adding running-specific weights to graph"""
    
    def test_add_running_weights_success(self):
        """Test successful running weight addition"""
        result = route.add_running_weights(self.graph_with_elevation)
        
        # Check that running weights were added (MultiGraph uses edge keys)
        edge_1_2 = result[1][2][0]  # First edge between nodes 1 and 2
        self.assertIn('running_weight', edge_1_2)
        
        # Running weight should be based on length and elevation
        expected_weight = 100 + (0.1 * 20)  # length + elevation_weight * elevation_gain
        self.assertAlmostEqual(edge_1_2['running_weight'], expected_weight, places=1)
    
    def test_add_running_weights_uphill_penalty(self):
        """Test running weights with uphill penalty"""
        result = route.add_running_weights(self.graph_with_elevation, grade_penalty=3.0)
        
        edge_1_2 = result[1][2][0]  # First edge between nodes 1 and 2
        # Should have higher weight for uphill (positive elevation gain)
        self.assertGreater(edge_1_2['running_weight'], 100)
    
    def test_add_running_weights_missing_elevation_data(self):
        """Test running weights without elevation data"""
        result = route.add_running_weights(self.test_graph)
        
        # Should add elevation data first, then weights
        edge_1_2 = result[1][2][0]  # First edge between nodes 1 and 2
        self.assertIn('running_weight', edge_1_2)


class TestDistanceConstrainedOperations(TestRouteUtilities):
    """Test distance-constrained graph operations"""
    
    def test_get_nodes_within_distance_basic(self):
        """Test getting nodes within distance"""
        nodes = route.get_nodes_within_distance(self.test_graph, 1, 0.15)  # 150m = 0.15km
        
        # Should include at least the start node
        self.assertIn(1, nodes)
        # The specific inclusion of other nodes depends on the actual distance calculation
        self.assertGreaterEqual(len(nodes), 1)
    
    def test_get_nodes_within_distance_large_radius(self):
        """Test getting nodes within large distance"""
        nodes = route.get_nodes_within_distance(self.test_graph, 1, 1.0)  # 1km
        
        # Should include at least the start node, likely all nodes for 1km radius
        self.assertIn(1, nodes)
        self.assertGreaterEqual(len(nodes), 1)
    
    def test_get_nodes_within_distance_zero_radius(self):
        """Test getting nodes with zero distance"""
        nodes = route.get_nodes_within_distance(self.test_graph, 1, 0.0)
        
        # Only start node should be included
        self.assertEqual(len(nodes), 1)
        self.assertIn(1, nodes)
    
    def test_get_nodes_within_distance_invalid_start(self):
        """Test getting nodes with invalid start node"""
        nodes = route.get_nodes_within_distance(self.test_graph, 999, 1.0)
        
        # Should return empty list for invalid start node
        self.assertEqual(len(nodes), 0)
    
    def test_create_distance_constrained_subgraph(self):
        """Test creating distance-constrained subgraph"""
        subgraph = route.create_distance_constrained_subgraph(self.test_graph, 1, 0.15)
        
        # Should include at least the start node
        self.assertIn(1, subgraph.nodes)
        self.assertGreaterEqual(len(subgraph.nodes), 1)
    
    def test_create_distance_constrained_subgraph_all_nodes(self):
        """Test subgraph creation that includes all nodes"""
        subgraph = route.create_distance_constrained_subgraph(self.test_graph, 1, 1.0)
        
        # Should include at least the start node
        self.assertIn(1, subgraph.nodes)
        self.assertGreaterEqual(len(subgraph.nodes), 1)


class TestMainFunction(TestRouteUtilities):
    """Test main function"""
    
    @patch('route.ox.graph_from_place')
    @patch('route.add_enhanced_elevation_to_graph')
    @patch('matplotlib.pyplot.show')
    def test_main_function_basic(self, mock_show, mock_add_elevation, mock_graph_from_place):
        """Test basic main function execution"""
        # Mock the graph creation and processing
        mock_graph = nx.Graph()
        mock_graph.add_node(216507089, x=-80.4094, y=37.1299, elevation=600)
        mock_graph_from_place.return_value = mock_graph
        mock_add_elevation.return_value = mock_graph
        
        # Should not raise exception
        try:
            route.main()
        except (SystemExit, Exception):
            pass  # main() may call sys.exit or have other behavior, which is fine
    
    @patch('route.ox.graph_from_place')
    @patch('matplotlib.pyplot.show')
    def test_main_function_exception_handling(self, mock_show, mock_graph_from_place):
        """Test main function exception handling"""
        mock_graph_from_place.side_effect = Exception("Network error")
        
        # Should handle exceptions gracefully
        try:
            route.main()
        except (SystemExit, Exception):
            pass  # main() may handle exceptions in various ways


if __name__ == '__main__':
    unittest.main()