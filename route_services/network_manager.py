#!/usr/bin/env python3
"""
Network Manager
Handles graph loading and caching for route planning
"""

import time
from typing import Optional, Tuple, List, Dict, Any
import networkx as nx
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import numpy as np


class NetworkManager:
    """Manages street network loading and caching"""
    
    DEFAULT_CENTER_POINT = (37.1299, -80.4094)  # Christiansburg, VA
    DEFAULT_RADIUS_KM = 5.0  # Increased to cover all of Christiansburg
    DEFAULT_NETWORK_TYPE = 'all'
    DEFAULT_START_COORDINATES = (37.13095, -80.40749)  # Dynamic node lookup based on these coordinates
    
    def __init__(self, center_point: Optional[Tuple[float, float]] = None, verbose: bool = True):
        """Initialize network manager
        
        Args:
            center_point: (lat, lon) tuple for network center
            verbose: Whether to show loading messages
        """
        self.center_point = center_point or self.DEFAULT_CENTER_POINT
        self.verbose = verbose
        self._graph_cache = {}
        self._nodes_gdf_cache = {}  # Cache for GeoDataFrame nodes
        self._edges_gdf_cache = {}  # Cache for GeoDataFrame edges
        
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
        
        # Find node closest to default start coordinates
        print(f"🔍 Finding node closest to default start coordinates {self.DEFAULT_START_COORDINATES}...")
        
        center_lat, center_lon = self.DEFAULT_START_COORDINATES
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
        
        print(f"📍 Using dynamically found start node: {closest_node} ({min_distance:.0f}m from target)")
        return closest_node
    
    def clear_cache(self):
        """Clear the graph cache"""
        self._graph_cache.clear()
        self._nodes_gdf_cache.clear()
        self._edges_gdf_cache.clear()
        print("🗑️ Network cache cleared")
    
    def get_nodes_geodataframe(self, graph: nx.Graph = None) -> gpd.GeoDataFrame:
        """Get graph nodes as GeoDataFrame with spatial indexing
        
        Args:
            graph: NetworkX graph (if None, uses current loaded graph)
            
        Returns:
            GeoDataFrame with all graph nodes
        """
        if graph is None:
            graph = self.load_network()
        
        if graph is None:
            return gpd.GeoDataFrame()
        
        # Check cache first
        graph_id = id(graph)
        if graph_id in self._nodes_gdf_cache:
            return self._nodes_gdf_cache[graph_id]
        
        # Create GeoDataFrame
        nodes_data = []
        for node_id, data in graph.nodes(data=True):
            nodes_data.append({
                'node_id': node_id,
                'elevation': data.get('elevation', 0),
                'highway': data.get('highway', ''),
                'degree': graph.degree(node_id),
                'x': data.get('x', 0),
                'y': data.get('y', 0),
                'geometry': Point(data.get('x', 0), data.get('y', 0))
            })
        
        nodes_gdf = gpd.GeoDataFrame(nodes_data, crs='EPSG:4326')
        
        # Build spatial index for faster queries
        nodes_gdf.sindex
        
        # Cache the result
        self._nodes_gdf_cache[graph_id] = nodes_gdf
        
        return nodes_gdf
    
    def get_edges_geodataframe(self, graph: nx.Graph = None) -> gpd.GeoDataFrame:
        """Get graph edges as GeoDataFrame with spatial indexing
        
        Args:
            graph: NetworkX graph (if None, uses current loaded graph)
            
        Returns:
            GeoDataFrame with all graph edges
        """
        if graph is None:
            graph = self.load_network()
        
        if graph is None:
            return gpd.GeoDataFrame()
        
        # Check cache first
        graph_id = id(graph)
        if graph_id in self._edges_gdf_cache:
            return self._edges_gdf_cache[graph_id]
        
        # Create GeoDataFrame
        edges_data = []
        for u, v, data in graph.edges(data=True):
            if u in graph.nodes and v in graph.nodes:
                u_data = graph.nodes[u]
                v_data = graph.nodes[v]
                
                # Create LineString geometry
                from shapely.geometry import LineString
                line_geom = LineString([
                    (u_data.get('x', 0), u_data.get('y', 0)),
                    (v_data.get('x', 0), v_data.get('y', 0))
                ])
                
                edges_data.append({
                    'u': u,
                    'v': v,
                    'length': data.get('length', 0),
                    'highway': data.get('highway', ''),
                    'geometry': line_geom
                })
        
        edges_gdf = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        # Build spatial index for faster queries
        edges_gdf.sindex
        
        # Cache the result
        self._edges_gdf_cache[graph_id] = edges_gdf
        
        return edges_gdf
    
    def get_nearby_nodes_spatial(self, graph: nx.Graph, lat: float, lon: float, 
                               radius_km: float = 2.0, max_nodes: int = 50) -> gpd.GeoDataFrame:
        """Get nodes near a given location using spatial indexing
        
        Args:
            graph: NetworkX graph
            lat: Latitude
            lon: Longitude
            radius_km: Search radius in kilometers
            max_nodes: Maximum number of nodes to return
            
        Returns:
            GeoDataFrame with nearby nodes and distances
        """
        if not graph:
            return gpd.GeoDataFrame()
        
        nodes_gdf = self.get_nodes_geodataframe(graph)
        
        # Create query point
        query_point = Point(lon, lat)
        
        # Create buffer for spatial query (rough degrees conversion)
        buffer_degrees = radius_km / 111  # Approximate degrees per km
        query_buffer = query_point.buffer(buffer_degrees)
        
        # Use spatial index for efficient query
        possible_matches_index = list(nodes_gdf.sindex.intersection(query_buffer.bounds))
        possible_matches = nodes_gdf.iloc[possible_matches_index]
        
        # Filter by actual buffer intersection
        nearby_nodes = possible_matches[possible_matches.geometry.within(query_buffer)].copy()
        
        if nearby_nodes.empty:
            return gpd.GeoDataFrame()
        
        # Calculate precise distances
        nearby_nodes['distance_m'] = nearby_nodes.apply(
            lambda row: self._calculate_geo_distance(query_point, row['geometry']), 
            axis=1
        )
        
        # Sort by distance and limit results
        nearby_nodes = nearby_nodes.sort_values('distance_m').head(max_nodes)
        
        return nearby_nodes
    
    def get_intersections_geodataframe(self, graph: nx.Graph = None, 
                                     max_nodes: int = 200) -> gpd.GeoDataFrame:
        """Get intersection nodes (degree != 2) as GeoDataFrame
        
        Args:
            graph: NetworkX graph (if None, uses current loaded graph)
            max_nodes: Maximum number of intersections to return
            
        Returns:
            GeoDataFrame with intersection nodes only
        """
        if graph is None:
            graph = self.load_network()
        
        if graph is None:
            return gpd.GeoDataFrame()
        
        nodes_gdf = self.get_nodes_geodataframe(graph)
        
        # Filter for intersection nodes (degree != 2)
        intersections_gdf = nodes_gdf[nodes_gdf['degree'] != 2].copy()
        
        # Sort by node_id and limit results
        intersections_gdf = intersections_gdf.sort_values('node_id').head(max_nodes)
        
        return intersections_gdf
    
    def find_optimal_start_node_spatial(self, graph: nx.Graph, 
                                      target_lat: float, target_lon: float,
                                      prefer_intersections: bool = True) -> int:
        """Find optimal start node using spatial operations
        
        Args:
            graph: NetworkX graph
            target_lat: Target latitude
            target_lon: Target longitude
            prefer_intersections: Whether to prefer intersection nodes
            
        Returns:
            Optimal start node ID
        """
        if not graph or len(graph.nodes) == 0:
            raise ValueError("No valid start node found in graph")
        
        # Get nodes as GeoDataFrame
        if prefer_intersections:
            candidates_gdf = self.get_intersections_geodataframe(graph)
            if candidates_gdf.empty:
                # Fall back to all nodes if no intersections
                candidates_gdf = self.get_nodes_geodataframe(graph)
        else:
            candidates_gdf = self.get_nodes_geodataframe(graph)
        
        if candidates_gdf.empty:
            raise ValueError("No candidate nodes found")
        
        # Find closest node using spatial operations
        query_point = Point(target_lon, target_lat)
        
        # Calculate distances to all candidates
        candidates_gdf['distance_to_target'] = candidates_gdf.apply(
            lambda row: self._calculate_geo_distance(query_point, row['geometry']), 
            axis=1
        )
        
        # Get the closest node
        closest_node_row = candidates_gdf.loc[candidates_gdf['distance_to_target'].idxmin()]
        
        if self.verbose:
            print(f"📍 Found optimal start node: {closest_node_row['node_id']} "
                  f"({closest_node_row['distance_to_target']:.0f}m from target)")
        
        return int(closest_node_row['node_id'])
    
    def analyze_network_connectivity(self, graph: nx.Graph) -> Dict[str, Any]:
        """Analyze network connectivity using spatial operations
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary with connectivity analysis
        """
        if not graph:
            return {}
        
        nodes_gdf = self.get_nodes_geodataframe(graph)
        edges_gdf = self.get_edges_geodataframe(graph)
        
        # Calculate network statistics
        total_nodes = len(nodes_gdf)
        total_edges = len(edges_gdf)
        
        # Degree distribution
        degree_stats = {
            'min_degree': nodes_gdf['degree'].min(),
            'max_degree': nodes_gdf['degree'].max(),
            'avg_degree': nodes_gdf['degree'].mean(),
            'degree_std': nodes_gdf['degree'].std()
        }
        
        # Intersection analysis
        intersections = nodes_gdf[nodes_gdf['degree'] > 2]
        dead_ends = nodes_gdf[nodes_gdf['degree'] == 1]
        
        # Edge length statistics
        edge_length_stats = {
            'min_edge_length': edges_gdf['length'].min(),
            'max_edge_length': edges_gdf['length'].max(),
            'avg_edge_length': edges_gdf['length'].mean(),
            'total_network_length': edges_gdf['length'].sum()
        }
        
        # Spatial bounds
        bounds = nodes_gdf.bounds
        network_bounds = {
            'min_lat': bounds['miny'].min(),
            'max_lat': bounds['maxy'].max(),
            'min_lon': bounds['minx'].min(),
            'max_lon': bounds['maxx'].max()
        }
        
        return {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'intersection_count': len(intersections),
            'dead_end_count': len(dead_ends),
            'degree_statistics': degree_stats,
            'edge_length_statistics': edge_length_stats,
            'network_bounds': network_bounds,
            'connectivity_ratio': total_edges / total_nodes if total_nodes > 0 else 0
        }
    
    def _calculate_geo_distance(self, point1: Point, point2: Point) -> float:
        """Calculate geographic distance between two points
        
        Args:
            point1: First point geometry
            point2: Second point geometry
            
        Returns:
            Distance in meters
        """
        def haversine_distance(lat1, lon1, lat2, lon2):
            import math
            R = 6371000  # Earth radius in meters
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            return R * c
        
        return haversine_distance(point1.y, point1.x, point2.y, point2.x)
    
    def get_network_within_bounds(self, graph: nx.Graph, 
                                min_lat: float, max_lat: float,
                                min_lon: float, max_lon: float) -> gpd.GeoDataFrame:
        """Get network nodes within specified bounds
        
        Args:
            graph: NetworkX graph
            min_lat: Minimum latitude
            max_lat: Maximum latitude
            min_lon: Minimum longitude
            max_lon: Maximum longitude
            
        Returns:
            GeoDataFrame with nodes within bounds
        """
        if not graph:
            return gpd.GeoDataFrame()
        
        nodes_gdf = self.get_nodes_geodataframe(graph)
        
        # Filter by bounds
        bounds_mask = (
            (nodes_gdf['y'] >= min_lat) & 
            (nodes_gdf['y'] <= max_lat) &
            (nodes_gdf['x'] >= min_lon) & 
            (nodes_gdf['x'] <= max_lon)
        )
        
        return nodes_gdf[bounds_mask].copy()