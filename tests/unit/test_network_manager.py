#!/usr/bin/env python3
"""
Unit tests for NetworkManager
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import networkx as nx
import sys
import os

# Add the parent directory to sys.path to import route_services
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from route_services.network_manager import NetworkManager


class TestNetworkManager(unittest.TestCase):
    """Test cases for NetworkManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = NetworkManager()
        
        # Create a mock graph for testing
        self.mock_graph = nx.Graph()
        self.mock_graph.add_node(1001, x=-80.4094, y=37.1299, elevation=610)
        self.mock_graph.add_node(1002, x=-80.4095, y=37.1300, elevation=615)
        self.mock_graph.add_node(1003, x=-80.4096, y=37.1301, elevation=620)
        self.mock_graph.add_edge(1001, 1002, length=100)
        self.mock_graph.add_edge(1002, 1003, length=150)
    
    def test_initialization(self):
        """Test NetworkManager initialization"""
        # Test default initialization
        manager = NetworkManager()
        self.assertEqual(manager.center_point, (37.1299, -80.4094))
        self.assertEqual(manager._graph_cache, {})
        
        # Test custom center point
        custom_center = (40.7128, -74.0060)  # NYC
        manager = NetworkManager(center_point=custom_center)
        self.assertEqual(manager.center_point, custom_center)
    
    @patch('graph_cache.load_or_generate_graph')
    def test_load_network_success(self, mock_load_graph):
        """Test successful network loading"""
        mock_load_graph.return_value = self.mock_graph
        
        result = self.manager.load_network(radius_km=1.0, network_type='all')
        
        self.assertEqual(result, self.mock_graph)
        mock_load_graph.assert_called_once_with(
            center_point=(37.1299, -80.4094),
            radius_m=1000,
            network_type='all'
        )
    
    @patch('graph_cache.load_or_generate_graph')
    def test_load_network_failure(self, mock_load_graph):
        """Test network loading failure"""
        mock_load_graph.return_value = None
        
        result = self.manager.load_network()
        
        self.assertIsNone(result)
    
    @patch('graph_cache.load_or_generate_graph')
    def test_load_network_exception(self, mock_load_graph):
        """Test network loading with exception"""
        mock_load_graph.side_effect = Exception("Network error")
        
        result = self.manager.load_network()
        
        self.assertIsNone(result)
    
    @patch('graph_cache.load_or_generate_graph')
    def test_network_caching(self, mock_load_graph):
        """Test that networks are cached properly"""
        mock_load_graph.return_value = self.mock_graph
        
        # First call
        result1 = self.manager.load_network(radius_km=1.0)
        # Second call with same parameters
        result2 = self.manager.load_network(radius_km=1.0)
        
        # Should only call load_or_generate_graph once
        self.assertEqual(mock_load_graph.call_count, 1)
        self.assertEqual(result1, result2)
        self.assertEqual(result1, self.mock_graph)
    
    def test_get_network_stats(self):
        """Test network statistics calculation"""
        stats = self.manager.get_network_stats(self.mock_graph)
        
        expected_stats = {
            'nodes': 3,
            'edges': 2,
            'center_point': (37.1299, -80.4094),
            'has_elevation': True
        }
        
        self.assertEqual(stats, expected_stats)
    
    def test_get_network_stats_empty(self):
        """Test network statistics with empty graph"""
        stats = self.manager.get_network_stats(None)
        self.assertEqual(stats, {})
    
    def test_validate_node_exists(self):
        """Test node existence validation"""
        # Existing node
        self.assertTrue(self.manager.validate_node_exists(self.mock_graph, 1001))
        # Non-existing node
        self.assertFalse(self.manager.validate_node_exists(self.mock_graph, 9999))
        # None graph
        self.assertFalse(self.manager.validate_node_exists(None, 1001))
    
    def test_get_node_info(self):
        """Test node information retrieval"""
        info = self.manager.get_node_info(self.mock_graph, 1001)
        
        expected_info = {
            'node_id': 1001,
            'latitude': 37.1299,
            'longitude': -80.4094,
            'elevation': 610,
            'degree': 1
        }
        
        self.assertEqual(info, expected_info)
    
    def test_get_node_info_nonexistent(self):
        """Test node info for non-existent node"""
        info = self.manager.get_node_info(self.mock_graph, 9999)
        self.assertEqual(info, {})
    
    @patch('route.haversine_distance')
    def test_get_nearby_nodes(self, mock_haversine):
        """Test nearby nodes search"""
        # Mock haversine distance to return predictable values
        mock_haversine.side_effect = [100, 200, 600]  # distances in meters
        
        nearby = self.manager.get_nearby_nodes(
            self.mock_graph, 37.1299, -80.4094, radius_km=0.5, max_nodes=10
        )
        
        # Should return nodes within 500m, sorted by distance
        self.assertEqual(len(nearby), 2)  # 100m and 200m are within 500m
        self.assertEqual(nearby[0][0], 1001)  # Closest node first
        self.assertEqual(nearby[1][0], 1002)  # Second closest
    
    def test_get_nearby_nodes_empty_graph(self):
        """Test nearby nodes with empty graph"""
        nearby = self.manager.get_nearby_nodes(None, 37.1299, -80.4094)
        self.assertEqual(nearby, [])
    
    def test_clear_cache(self):
        """Test cache clearing"""
        # Add something to cache
        self.manager._graph_cache['test'] = self.mock_graph
        
        # Clear cache
        self.manager.clear_cache()
        
        self.assertEqual(self.manager._graph_cache, {})
    
    def test_default_constants(self):
        """Test default constants"""
        self.assertEqual(NetworkManager.DEFAULT_CENTER_POINT, (37.1299, -80.4094))
        self.assertEqual(NetworkManager.DEFAULT_RADIUS_KM, 0.8)
        self.assertEqual(NetworkManager.DEFAULT_NETWORK_TYPE, 'all')


if __name__ == '__main__':
    unittest.main()