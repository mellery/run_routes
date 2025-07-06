#!/usr/bin/env python3
"""
Network Manager
Handles graph loading and caching for route planning
"""

import time
from typing import Optional, Tuple
import networkx as nx


class NetworkManager:
    """Manages street network loading and caching"""
    
    DEFAULT_CENTER_POINT = (37.1299, -80.4094)  # Christiansburg, VA
    DEFAULT_RADIUS_KM = 5.0  # Increased to cover all of Christiansburg
    DEFAULT_NETWORK_TYPE = 'all'
    DEFAULT_START_NODE = 1529188403  # Downtown Christiansburg intersection
    
    def __init__(self, center_point: Optional[Tuple[float, float]] = None, verbose: bool = True):
        """Initialize network manager
        
        Args:
            center_point: (lat, lon) tuple for network center
            verbose: Whether to show loading messages
        """
        self.center_point = center_point or self.DEFAULT_CENTER_POINT
        self.verbose = verbose
        self._graph_cache = {}
        
    def load_network(self, radius_km: float = None, network_type: str = None) -> Optional[nx.Graph]:
        """Load street network with elevation data
        
        Args:
            radius_km: Network radius in kilometers
            network_type: OSMnx network type ('all', 'drive', etc.)
            
        Returns:
            NetworkX graph or None if loading fails
        """
        radius_km = radius_km or self.DEFAULT_RADIUS_KM
        network_type = network_type or self.DEFAULT_NETWORK_TYPE
        
        # Create cache key
        cache_key = (self.center_point, radius_km, network_type)
        
        # Return cached graph if available
        if cache_key in self._graph_cache:
            return self._graph_cache[cache_key]
        
        print(f"🌐 Loading street network and elevation data...")
        print(f"   Area: {radius_km:.1f}km radius around {self.center_point}")
        
        try:
            # Use cached graph loader
            from graph_cache import load_or_generate_graph
            
            graph = load_or_generate_graph(
                center_point=self.center_point,
                radius_m=int(radius_km * 1000),
                network_type=network_type
            )
            
            if graph:
                if self.verbose:
                    print(f"✅ Loaded {len(graph.nodes)} intersections and {len(graph.edges)} road segments")
                # Cache the loaded graph
                self._graph_cache[cache_key] = graph
                return graph
            else:
                if self.verbose:
                    print("❌ Failed to load network")
                return None
                
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to load network: {e}")
            return None
    
    def get_network_stats(self, graph: nx.Graph) -> dict:
        """Get basic network statistics
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary with network statistics
        """
        if not graph:
            return {}
        
        return {
            'nodes': len(graph.nodes),
            'edges': len(graph.edges),
            'center_point': self.center_point,
            'has_elevation': any('elevation' in data for _, data in graph.nodes(data=True))
        }
    
    def validate_node_exists(self, graph: nx.Graph, node_id: int) -> bool:
        """Check if a node exists in the graph
        
        Args:
            graph: NetworkX graph
            node_id: Node ID to check
            
        Returns:
            True if node exists
        """
        return graph is not None and node_id in graph.nodes
    
    def get_node_info(self, graph: nx.Graph, node_id: int) -> dict:
        """Get information about a specific node
        
        Args:
            graph: NetworkX graph
            node_id: Node ID
            
        Returns:
            Dictionary with node information
        """
        if not self.validate_node_exists(graph, node_id):
            return {}
        
        data = graph.nodes[node_id]
        return {
            'node_id': node_id,
            'latitude': data.get('y', 0),
            'longitude': data.get('x', 0),
            'elevation': data.get('elevation', 0),
            'degree': graph.degree(node_id)
        }
    
    def get_nearby_nodes(self, graph: nx.Graph, lat: float, lon: float, 
                        radius_km: float = 2.0, max_nodes: int = 50) -> list:
        """Get nodes near a given location
        
        Args:
            graph: NetworkX graph
            lat: Latitude
            lon: Longitude
            radius_km: Search radius in kilometers (default 2.0 for city-wide coverage)
            max_nodes: Maximum number of nodes to return (default 50 for more options)
            
        Returns:
            List of (node_id, distance_m, node_data) tuples
        """
        if not graph:
            return []
        
        from route import haversine_distance
        
        nearby = []
        radius_m = radius_km * 1000
        
        for node_id, data in graph.nodes(data=True):
            distance = haversine_distance(lat, lon, data['y'], data['x'])
            if distance <= radius_m:
                nearby.append((node_id, distance, data))
        
        # Sort by distance and return closest nodes
        nearby.sort(key=lambda x: x[1])
        return nearby[:max_nodes]
    
    def get_all_intersections(self, graph: nx.Graph, max_nodes: int = 200) -> list:
        """Get all intersection nodes (degree != 2) in the network
        
        Args:
            graph: NetworkX graph
            max_nodes: Maximum number of intersections to return
            
        Returns:
            List of (node_id, node_data) tuples for intersections only
        """
        if not graph:
            return []
        
        intersections = []
        for node_id, data in graph.nodes(data=True):
            # Only include intersection nodes (not geometry nodes)
            if graph.degree(node_id) != 2:
                intersections.append((node_id, data))
        
        # Sort by node_id for consistent ordering
        intersections.sort(key=lambda x: x[0])
        return intersections[:max_nodes]
    
    def get_start_node(self, graph: nx.Graph, user_start_node: Optional[int] = None) -> int:
        """Get the starting node for route planning
        
        Args:
            graph: NetworkX graph
            user_start_node: User-specified start node (takes priority)
            
        Returns:
            Node ID for starting point
            
        Raises:
            ValueError: If specified start node doesn't exist in graph
        """
        if graph is None:
            raise ValueError("Graph is required")
        
        if len(graph.nodes) == 0:
            raise ValueError("No valid start node found in graph")
        
        # Use user-specified start node if provided
        if user_start_node is not None:
            if user_start_node in graph.nodes:
                print(f"🎯 Using user-specified start node: {user_start_node}")
                return user_start_node
            else:
                raise ValueError(f"User-specified start node {user_start_node} not found in graph")
        
        # Use default start node if it exists in graph
        if self.DEFAULT_START_NODE in graph.nodes:
            print(f"🎯 Using default start node: {self.DEFAULT_START_NODE}")
            return self.DEFAULT_START_NODE
        
        # Fallback: find node closest to center point
        print(f"⚠️ Default start node {self.DEFAULT_START_NODE} not found in graph")
        print(f"🔍 Finding node closest to center point as fallback...")
        
        center_lat, center_lon = self.center_point
        closest_node = None
        min_distance = float('inf')
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            import math
            R = 6371000  # Earth radius in meters
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            return R * c
        
        for node_id, node_data in graph.nodes(data=True):
            distance = haversine_distance(center_lat, center_lon, node_data['y'], node_data['x'])
            if distance < min_distance:
                min_distance = distance
                closest_node = node_id
        
        if closest_node is None:
            raise ValueError("No valid start node found in graph")
        
        print(f"📍 Using fallback start node: {closest_node} ({min_distance:.0f}m from center)")
        return closest_node
    
    def clear_cache(self):
        """Clear the graph cache"""
        self._graph_cache.clear()
        print("🗑️ Network cache cleared")