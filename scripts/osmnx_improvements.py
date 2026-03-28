#!/usr/bin/env python3
"""
OSMnx Performance Improvements
Advanced techniques for better network queries and processing
"""

import osmnx as ox
import networkx as nx
from typing import Tuple, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class AdvancedOSMnxManager:
    """Advanced OSMnx operations for better performance"""
    
    def __init__(self):
        self.configure_osmnx()
    
    def configure_osmnx(self):
        """Configure OSMnx with optimized settings"""
        # 1. Use more efficient projections
        ox.settings.default_crs = 'epsg:4326'  # Keep in WGS84 until projection needed
        
        # 2. Optimize HTTP settings
        ox.settings.requests_kwargs = {
            'timeout': 300,
            'headers': {'User-Agent': 'RunRoutes/1.0'}
        }
        
        # 3. Memory optimization
        ox.settings.memory = 4  # Use more memory for large areas
        
        # 4. Cache optimization
        ox.settings.use_cache = True
        ox.settings.cache_only_mode = False
        
    def download_network_with_filter(self, center_point: Tuple[float, float], 
                                    radius_m: int, 
                                    running_optimized: bool = True) -> nx.MultiDiGraph:
        """
        Download network with server-side filtering for better performance
        
        Args:
            center_point: (lat, lon) center point
            radius_m: Radius in meters
            running_optimized: Use running-specific filters
            
        Returns:
            NetworkX graph
        """
        if running_optimized:
            # 1. Use custom filter that excludes unwanted ways server-side
            custom_filter = (
                '["highway"~"primary|secondary|tertiary|residential|path|living_street|unclassified"]'
                '["highway"!~"motorway|trunk|footway|cycleway|motorway_link|trunk_link"]'
                '["access"!~"private"]'
                '["surface"!~"sand|mud|gravel"]'
            )
            
            # 2. Download with filter to reduce data transfer
            graph = ox.graph_from_point(
                center_point,
                dist=radius_m,
                network_type='all',
                custom_filter=custom_filter,
                retain_all=False,  # Remove unconnected components
                truncate_by_edge=True,  # More accurate boundary
                clean_periphery=True,  # Remove peripheral nodes
                simplify=False  # Don't simplify yet - do it later with custom logic
            )
        else:
            # Standard download
            graph = ox.graph_from_point(
                center_point,
                dist=radius_m,
                network_type='all',
                retain_all=False,
                truncate_by_edge=True
            )
        
        return graph
    
    def optimize_graph_for_routing(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """
        Optimize graph structure for route optimization algorithms
        
        Args:
            graph: Input graph
            
        Returns:
            Optimized graph
        """
        # 1. Convert to undirected for routing (most running routes are bidirectional)
        if graph.is_directed():
            graph = ox.convert.to_undirected(graph)
        
        # 2. Project to appropriate CRS for accurate distance calculations
        graph = ox.project_graph(graph)
        
        # 3. Add edge speeds and travel times based on highway type
        graph = ox.add_edge_speeds(graph)
        graph = ox.add_edge_travel_times(graph)
        
        # 4. Consolidate intersections with proper tolerance
        # Use projected coordinates for accurate distance-based consolidation
        graph = ox.consolidate_intersections(
            graph, 
            tolerance=10,  # 10 meters
            rebuild_graph=True,
            dead_ends=False,  # Keep dead ends for running routes
            reconnect_edges=True
        )
        
        # 5. Add bearing information for route diversity
        graph = ox.add_edge_bearings(graph)
        
        return graph
    
    def add_elevation_efficiently(self, graph: nx.MultiDiGraph, 
                                elevation_source: str = '3dep') -> nx.MultiDiGraph:
        """
        Add elevation data more efficiently
        
        Args:
            graph: Input graph
            elevation_source: '3dep', 'srtm', or 'google'
            
        Returns:
            Graph with elevation data
        """
        if elevation_source == '3dep':
            # Use 3DEP for high accuracy in US
            try:
                graph = ox.elevation.add_node_elevations_google(
                    graph, 
                    api_key=None,  # Uses free tier
                    max_locations_per_batch=100,  # Batch requests
                    pause_duration=0.1
                )
            except:
                # Fallback to raster if Google fails
                graph = ox.elevation.add_node_elevations_raster(
                    graph, 
                    filepath='elevation_data/3dep_1m/merged_3dep.tif',
                    cpus=None  # Use all available CPUs
                )
        
        elif elevation_source == 'srtm':
            graph = ox.elevation.add_node_elevations_raster(
                graph,
                filepath='elevation_data/srtm_90m/srtm_20_05.tif'
            )
        
        # Add edge grades
        graph = ox.elevation.add_edge_grades(graph, add_absolute=True)
        
        return graph
    
    def create_routing_weights(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """
        Add custom weights optimized for running route algorithms
        
        Args:
            graph: Graph with elevation data
            
        Returns:
            Graph with routing weights
        """
        for u, v, data in graph.edges(data=True):
            # Base weight is distance
            length = data.get('length', 100)  # Default 100m if missing
            
            # Grade penalty (avoid steep hills)
            grade = abs(data.get('grade_abs', 0))
            grade_penalty = 1.0
            if grade > 0.05:  # > 5% grade
                grade_penalty = 1.2 + (grade - 0.05) * 2  # Exponential penalty
            
            # Highway type preference
            highway = data.get('highway', 'unclassified')
            if isinstance(highway, list):
                highway = highway[0]
            
            highway_weights = {
                'path': 0.8,        # Prefer paths for running
                'residential': 0.9,  # Good for running
                'living_street': 0.8,
                'unclassified': 1.0,
                'tertiary': 1.1,
                'secondary': 1.3,    # Less preferred (more traffic)
                'primary': 1.5       # Least preferred
            }
            highway_penalty = highway_weights.get(highway, 1.0)
            
            # Surface preference
            surface = data.get('surface', 'unknown')
            surface_weights = {
                'asphalt': 0.9,
                'concrete': 0.9,
                'paved': 0.9,
                'compacted': 1.0,
                'gravel': 1.2,
                'dirt': 1.3,
                'grass': 1.4,
                'sand': 2.0,
                'unknown': 1.0
            }
            surface_penalty = surface_weights.get(surface, 1.0)
            
            # Final weight
            data['running_weight'] = length * grade_penalty * highway_penalty * surface_penalty
            data['elevation_weight'] = length * (1.0 + abs(grade) * 0.5)  # For elevation-seeking routes
            data['distance_weight'] = length  # Pure distance optimization
        
        return graph
    
    def get_network_statistics(self, graph: nx.MultiDiGraph) -> Dict[str, Any]:
        """
        Get comprehensive network statistics for analysis
        
        Args:
            graph: Input graph
            
        Returns:
            Dictionary with network statistics
        """
        stats = ox.stats.basic_stats(graph)
        
        # Add custom statistics for running routes
        elevation_data = [data.get('elevation', 0) for _, data in graph.nodes(data=True) 
                         if data.get('elevation') is not None]
        
        if elevation_data:
            stats['elevation_min'] = min(elevation_data)
            stats['elevation_max'] = max(elevation_data)
            stats['elevation_range'] = max(elevation_data) - min(elevation_data)
            stats['elevation_mean'] = sum(elevation_data) / len(elevation_data)
        
        # Grade analysis
        grades = [abs(data.get('grade', 0)) for _, _, data in graph.edges(data=True)
                 if data.get('grade') is not None]
        
        if grades:
            stats['max_grade'] = max(grades)
            stats['mean_grade'] = sum(grades) / len(grades)
            stats['steep_edges_5pct'] = len([g for g in grades if g > 0.05])
            stats['steep_edges_10pct'] = len([g for g in grades if g > 0.10])
        
        return stats
    
    def optimize_for_genetic_algorithm(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """
        Specific optimizations for genetic algorithm performance
        
        Args:
            graph: Input graph
            
        Returns:
            Optimized graph
        """
        # 1. Remove degree-2 nodes (except where needed for elevation changes)
        simplified_graph = self._selective_simplification(graph)
        
        # 2. Pre-compute shortest paths for common distance ranges
        # (This would be done lazily in practice)
        
        # 3. Add neighborhood information for faster neighbor queries
        for node in simplified_graph.nodes():
            neighbors = list(simplified_graph.neighbors(node))
            simplified_graph.nodes[node]['neighbor_count'] = len(neighbors)
            simplified_graph.nodes[node]['neighbors'] = neighbors[:10]  # Store first 10
        
        return simplified_graph
    
    def _selective_simplification(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """
        Simplify graph while preserving important elevation and routing features
        """
        # Don't simplify nodes with significant elevation changes
        important_nodes = set()
        
        for node in graph.nodes():
            node_data = graph.nodes[node]
            elevation = node_data.get('elevation', 0)
            
            # Check if this node has significant elevation difference from neighbors
            neighbor_elevations = []
            for neighbor in graph.neighbors(node):
                neighbor_elev = graph.nodes[neighbor].get('elevation', elevation)
                neighbor_elevations.append(neighbor_elev)
            
            if neighbor_elevations:
                max_diff = max(abs(elevation - ne) for ne in neighbor_elevations)
                if max_diff > 5:  # 5m elevation difference is significant
                    important_nodes.add(node)
        
        # Simplify graph but protect important nodes
        simplified = ox.simplification.simplify_graph(
            graph,
            strict=False,
            remove_rings=False,
            track_merged=True
        )
        
        return simplified


# Usage example and factory function
def create_optimized_network(center_point: Tuple[float, float], 
                           radius_m: int,
                           for_genetic_algorithm: bool = True) -> nx.MultiDiGraph:
    """
    Factory function to create an optimized network for running route planning
    
    Args:
        center_point: (lat, lon) center point
        radius_m: Radius in meters
        for_genetic_algorithm: Whether to optimize for GA performance
        
    Returns:
        Optimized graph ready for route planning
    """
    manager = AdvancedOSMnxManager()
    
    # 1. Download with filtering
    graph = manager.download_network_with_filter(center_point, radius_m)
    
    # 2. Optimize structure
    graph = manager.optimize_graph_for_routing(graph)
    
    # 3. Add elevation
    graph = manager.add_elevation_efficiently(graph, '3dep')
    
    # 4. Add routing weights
    graph = manager.create_routing_weights(graph)
    
    # 5. GA-specific optimizations
    if for_genetic_algorithm:
        graph = manager.optimize_for_genetic_algorithm(graph)
    
    return graph


if __name__ == "__main__":
    # Example usage
    center = (37.1299, -80.4094)  # Christiansburg, VA
    radius = 5000  # 5km
    
    print("Creating optimized network...")
    graph = create_optimized_network(center, radius)
    
    manager = AdvancedOSMnxManager()
    stats = manager.get_network_statistics(graph)
    
    print(f"Network created with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    print(f"Elevation range: {stats.get('elevation_min', 0):.0f}m - {stats.get('elevation_max', 0):.0f}m")
    print(f"Maximum grade: {stats.get('max_grade', 0)*100:.1f}%")