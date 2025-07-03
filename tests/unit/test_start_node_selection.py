#!/usr/bin/env python3
"""
Unit tests for start node selection in NetworkManager
"""

import unittest
import sys
import os
import networkx as nx

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from route_services import NetworkManager


class TestStartNodeSelection(unittest.TestCase):
    """Test start node selection logic"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.network_manager = NetworkManager()
        
        # Create a simple test graph with known nodes
        self.test_graph = nx.Graph()
        
        # Add nodes with coordinates around Christiansburg center (37.1299, -80.4094)
        test_nodes = [
            (1529188403, {'y': 37.130950, 'x': -80.407501, 'elevation': 633.0}),  # Default node (205m from center)
            (216367653, {'y': 37.167446, 'x': -80.397347, 'elevation': 584.0}),   # Far node (4.3km from center)
            (216434573, {'y': 37.129844, 'x': -80.408813, 'elevation': 650.0}),   # Very close node (52m from center)
            (999999999, {'y': 37.135000, 'x': -80.405000, 'elevation': 625.0}),   # Another test node
        ]
        
        for node_id, data in test_nodes:
            self.test_graph.add_node(node_id, **data)
        
        # Add some edges to make it a connected graph
        self.test_graph.add_edge(1529188403, 216434573)
        self.test_graph.add_edge(216434573, 999999999)
        self.test_graph.add_edge(999999999, 216367653)
    
    def test_default_start_node_exists(self):
        """Test that default start node is used when it exists in graph"""
        start_node = self.network_manager.get_start_node(self.test_graph)
        
        self.assertEqual(start_node, NetworkManager.DEFAULT_START_NODE)
        self.assertEqual(start_node, 1529188403)
        self.assertIn(start_node, self.test_graph.nodes)
    
    def test_user_specified_start_node(self):
        """Test that user-specified start node takes priority"""
        user_node = 216434573
        start_node = self.network_manager.get_start_node(self.test_graph, user_start_node=user_node)
        
        self.assertEqual(start_node, user_node)
        self.assertIn(start_node, self.test_graph.nodes)
    
    def test_user_specified_invalid_node_raises_error(self):
        """Test that invalid user-specified node raises ValueError"""
        invalid_node = 123456789  # Node that doesn't exist
        
        with self.assertRaises(ValueError) as context:
            self.network_manager.get_start_node(self.test_graph, user_start_node=invalid_node)
        
        self.assertIn("not found in graph", str(context.exception))
    
    def test_fallback_to_closest_node(self):
        """Test fallback to closest node when default doesn't exist"""
        # Create graph without default node
        fallback_graph = nx.Graph()
        fallback_graph.add_node(216434573, y=37.129844, x=-80.408813, elevation=650.0)  # 52m from center
        fallback_graph.add_node(216367653, y=37.167446, x=-80.397347, elevation=584.0)  # 4.3km from center
        fallback_graph.add_edge(216434573, 216367653)
        
        start_node = self.network_manager.get_start_node(fallback_graph)
        
        # Should select the node closest to center (216434573 at 52m vs 216367653 at 4.3km)
        self.assertEqual(start_node, 216434573)
    
    def test_empty_graph_raises_error(self):
        """Test that empty graph raises ValueError"""
        empty_graph = nx.Graph()
        
        with self.assertRaises(ValueError) as context:
            self.network_manager.get_start_node(empty_graph)
        
        self.assertIn("No valid start node found", str(context.exception))
    
    def test_none_graph_raises_error(self):
        """Test that None graph raises ValueError"""
        with self.assertRaises(ValueError) as context:
            self.network_manager.get_start_node(None)
        
        self.assertIn("Graph is required", str(context.exception))
    
    def test_start_node_coordinates(self):
        """Test that default start node has expected coordinates"""
        start_node = self.network_manager.get_start_node(self.test_graph)
        node_data = self.test_graph.nodes[start_node]
        
        # Verify coordinates are in downtown Christiansburg
        self.assertAlmostEqual(node_data['y'], 37.130950, places=5)
        self.assertAlmostEqual(node_data['x'], -80.407501, places=5)
        
        # Verify it's close to center point
        center_lat, center_lon = self.network_manager.center_point
        self.assertAlmostEqual(node_data['y'], center_lat, delta=0.01)  # Within ~1km
        self.assertAlmostEqual(node_data['x'], center_lon, delta=0.01)
    
    def test_default_start_node_constant(self):
        """Test that DEFAULT_START_NODE constant is correctly defined"""
        self.assertEqual(NetworkManager.DEFAULT_START_NODE, 1529188403)
        self.assertIsInstance(NetworkManager.DEFAULT_START_NODE, int)


class TestStartNodeIntegration(unittest.TestCase):
    """Integration tests with real NetworkManager"""
    
    def setUp(self):
        """Set up real network manager"""
        self.network_manager = NetworkManager()
    
    def test_default_start_node_in_real_graph(self):
        """Test that default start node exists in real Christiansburg graph"""
        # Load the actual Christiansburg network
        graph = self.network_manager.load_network(radius_km=5.0)
        self.assertIsNotNone(graph, "Failed to load network")
        
        # Verify default start node exists
        self.assertIn(NetworkManager.DEFAULT_START_NODE, graph.nodes, 
                     f"Default start node {NetworkManager.DEFAULT_START_NODE} not found in real graph")
        
        # Test start node selection
        start_node = self.network_manager.get_start_node(graph)
        self.assertEqual(start_node, NetworkManager.DEFAULT_START_NODE)
    
    def test_start_node_location_in_real_graph(self):
        """Test that start node is in reasonable location"""
        graph = self.network_manager.load_network(radius_km=5.0)
        self.assertIsNotNone(graph, "Failed to load network")
        
        start_node = self.network_manager.get_start_node(graph)
        node_data = graph.nodes[start_node]
        
        # Calculate distance from center
        import math
        center_lat, center_lon = self.network_manager.center_point
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371000  # Earth radius in meters
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            return R * c
        
        distance = haversine_distance(center_lat, center_lon, node_data['y'], node_data['x'])
        
        # Start node should be reasonably close to center (within 1km for downtown area)
        self.assertLess(distance, 1000, 
                       f"Start node is {distance:.0f}m from center - too far for downtown start")
        
        print(f"✅ Start node {start_node} is {distance:.0f}m from center")


if __name__ == '__main__':
    unittest.main()