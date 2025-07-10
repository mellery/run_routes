#!/usr/bin/env python3
"""
Elevation Profiler
Generates elevation profile data from routes
"""

from typing import List, Dict, Any, Tuple, Optional
import networkx as nx
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import numpy as np


class ElevationProfiler:
    """Generates elevation profile data and analysis"""
    
    def __init__(self, graph: nx.Graph):
        """Initialize elevation profiler
        
        Args:
            graph: NetworkX graph with elevation data
        """
        self.graph = graph
        self._distance_cache = {}  # Cache for network distances
        self._route_gdf_cache = {}  # Cache for route GeoDataFrames
    
    def generate_profile_data(self, route_result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate elevation profile data for a route
        
        Args:
            route_result: Route result from optimizer
            **kwargs: Additional parameters (for compatibility with enhanced version)
            
        Returns:
            Dictionary with elevation profile data
        """
        if not route_result or not route_result.get('route'):
            return {}
        
        route = route_result['route']
        
        # Extract coordinates and elevations
        coordinates = []
        elevations = []
        distances = [0]  # Start at 0 distance
        cumulative_distance = 0
        
        from route import haversine_distance
        
        # Process each node in the route
        for i, node in enumerate(route):
            if node in self.graph.nodes:
                data = self.graph.nodes[node]
                
                # Store coordinate and elevation
                coordinate = {
                    'latitude': data['y'],
                    'longitude': data['x'],
                    'node_id': node
                }
                coordinates.append(coordinate)
                elevations.append(data.get('elevation', 0))
                
                # Calculate cumulative distance using road network paths (same as TSP solver)
                if i > 0:
                    prev_node = route[i-1]
                    segment_dist = self._get_network_distance(prev_node, node)
                    cumulative_distance += segment_dist
                    distances.append(cumulative_distance)
        
        # Add return to start for complete loop
        if len(route) > 1:
            return_dist = self._get_network_distance(route[-1], route[0])
            cumulative_distance += return_dist
            distances.append(cumulative_distance)
            elevations.append(elevations[0])  # Back to start elevation
        
        # Convert distances to kilometers
        distances_km = [d / 1000 for d in distances]
        
        return {
            'coordinates': coordinates,
            'elevations': elevations,
            'distances_m': distances,
            'distances_km': distances_km,
            'total_distance_km': distances_km[-1] if distances_km else 0,
            'elevation_stats': self._calculate_elevation_stats(elevations, distances_km)
        }
    
    def _calculate_elevation_stats(self, elevations: List[float], distances_km: List[float]) -> Dict[str, Any]:
        """Calculate elevation statistics
        
        Args:
            elevations: List of elevation values
            distances_km: List of distance values in kilometers
            
        Returns:
            Dictionary with elevation statistics
        """
        if not elevations:
            return {}
        
        # Basic elevation statistics
        min_elevation = min(elevations)
        max_elevation = max(elevations)
        elevation_range = max_elevation - min_elevation
        avg_elevation = sum(elevations) / len(elevations)
        
        # Calculate grade information
        grades = []
        steep_sections = []
        
        for i in range(1, len(elevations)):
            if i < len(distances_km):
                elevation_change = elevations[i] - elevations[i-1]
                distance_change = (distances_km[i] - distances_km[i-1]) * 1000  # Convert to meters
                
                if distance_change > 0:
                    grade = (elevation_change / distance_change) * 100
                    grades.append(grade)
                    
                    # Track steep sections (>8% grade)
                    if abs(grade) > 8:
                        steep_sections.append({
                            'start_km': distances_km[i-1],
                            'end_km': distances_km[i],
                            'grade': grade,
                            'elevation_change': elevation_change
                        })
        
        # Grade statistics
        max_grade = max(grades) if grades else 0
        min_grade = min(grades) if grades else 0
        avg_grade = sum(grades) / len(grades) if grades else 0
        
        return {
            'min_elevation': min_elevation,
            'max_elevation': max_elevation,
            'elevation_range': elevation_range,
            'avg_elevation': avg_elevation,
            'max_grade': max_grade,
            'min_grade': min_grade,
            'avg_grade': avg_grade,
            'steep_sections': steep_sections,
            'steep_section_count': len(steep_sections)
        }
    
    def get_elevation_zones(self, route_result: Dict[str, Any], zone_count: int = 5) -> List[Dict[str, Any]]:
        """Divide route into elevation zones
        
        Args:
            route_result: Route result from optimizer
            zone_count: Number of zones to create
            
        Returns:
            List of zone dictionaries
        """
        profile_data = self.generate_profile_data(route_result)
        
        if not profile_data or not profile_data.get('elevations'):
            return []
        
        elevations = profile_data['elevations']
        distances_km = profile_data['distances_km']
        
        if len(elevations) < zone_count:
            zone_count = len(elevations)
        
        zones = []
        points_per_zone = len(elevations) // zone_count
        
        for i in range(zone_count):
            start_idx = i * points_per_zone
            if i == zone_count - 1:
                # Last zone gets remaining points
                end_idx = len(elevations)
            else:
                end_idx = (i + 1) * points_per_zone
            
            zone_elevations = elevations[start_idx:end_idx]
            zone_distances = distances_km[start_idx:end_idx+1] if end_idx < len(distances_km) else distances_km[start_idx:]
            
            if zone_elevations and zone_distances:
                zones.append({
                    'zone_number': i + 1,
                    'start_km': zone_distances[0],
                    'end_km': zone_distances[-1],
                    'distance_km': zone_distances[-1] - zone_distances[0],
                    'min_elevation': min(zone_elevations),
                    'max_elevation': max(zone_elevations),
                    'avg_elevation': sum(zone_elevations) / len(zone_elevations),
                    'elevation_change': zone_elevations[-1] - zone_elevations[0],
                    'point_count': len(zone_elevations)
                })
        
        return zones
    
    def find_elevation_peaks_valleys(self, route_result: Dict[str, Any], 
                                   min_prominence: float = 10) -> Dict[str, List]:
        """Find elevation peaks and valleys in the route
        
        Args:
            route_result: Route result from optimizer
            min_prominence: Minimum elevation change to be considered a peak/valley
            
        Returns:
            Dictionary with peaks and valleys lists
        """
        profile_data = self.generate_profile_data(route_result)
        
        if not profile_data or not profile_data.get('elevations'):
            return {'peaks': [], 'valleys': []}
        
        elevations = profile_data['elevations']
        distances_km = profile_data['distances_km']
        coordinates = profile_data['coordinates']
        
        peaks = []
        valleys = []
        
        # Find local maxima and minima
        for i in range(1, len(elevations) - 1):
            current_elev = elevations[i]
            prev_elev = elevations[i - 1]
            next_elev = elevations[i + 1]
            
            # Check for peak (higher than both neighbors)
            if current_elev > prev_elev and current_elev > next_elev:
                # Check prominence
                prominence = min(current_elev - prev_elev, current_elev - next_elev)
                if prominence >= min_prominence:
                    peaks.append({
                        'distance_km': distances_km[i],
                        'elevation': current_elev,
                        'prominence': prominence,
                        'coordinate': coordinates[i] if i < len(coordinates) else None
                    })
            
            # Check for valley (lower than both neighbors)
            elif current_elev < prev_elev and current_elev < next_elev:
                # Check prominence (depth)
                prominence = min(prev_elev - current_elev, next_elev - current_elev)
                if prominence >= min_prominence:
                    valleys.append({
                        'distance_km': distances_km[i],
                        'elevation': current_elev,
                        'prominence': prominence,
                        'coordinate': coordinates[i] if i < len(coordinates) else None
                    })
        
        return {
            'peaks': peaks,
            'valleys': valleys,
            'peak_count': len(peaks),
            'valley_count': len(valleys)
        }
    
    def get_climbing_segments(self, route_result: Dict[str, Any], 
                            min_gain: float = 20) -> List[Dict[str, Any]]:
        """Identify continuous climbing segments
        
        Args:
            route_result: Route result from optimizer
            min_gain: Minimum elevation gain to be considered a climbing segment
            
        Returns:
            List of climbing segment dictionaries
        """
        profile_data = self.generate_profile_data(route_result)
        
        if not profile_data or not profile_data.get('elevations'):
            return []
        
        elevations = profile_data['elevations']
        distances_km = profile_data['distances_km']
        
        climbing_segments = []
        current_segment = None
        
        for i in range(1, len(elevations)):
            elevation_change = elevations[i] - elevations[i - 1]
            
            if elevation_change > 0:  # Climbing
                if current_segment is None:
                    # Start new climbing segment
                    current_segment = {
                        'start_km': distances_km[i - 1],
                        'start_elevation': elevations[i - 1],
                        'end_km': distances_km[i],
                        'end_elevation': elevations[i]
                    }
                else:
                    # Continue current segment
                    current_segment['end_km'] = distances_km[i]
                    current_segment['end_elevation'] = elevations[i]
            else:
                # Not climbing, finish current segment if exists
                if current_segment is not None:
                    elevation_gain = current_segment['end_elevation'] - current_segment['start_elevation']
                    if elevation_gain >= min_gain:
                        current_segment['distance_km'] = current_segment['end_km'] - current_segment['start_km']
                        current_segment['elevation_gain'] = elevation_gain
                        current_segment['avg_grade'] = (elevation_gain / (current_segment['distance_km'] * 1000) * 100) if current_segment['distance_km'] > 0 else 0
                        climbing_segments.append(current_segment)
                    current_segment = None
        
        # Finish last segment if exists
        if current_segment is not None:
            elevation_gain = current_segment['end_elevation'] - current_segment['start_elevation']
            if elevation_gain >= min_gain:
                current_segment['distance_km'] = current_segment['end_km'] - current_segment['start_km']
                current_segment['elevation_gain'] = elevation_gain
                current_segment['avg_grade'] = (elevation_gain / (current_segment['distance_km'] * 1000) * 100) if current_segment['distance_km'] > 0 else 0
                climbing_segments.append(current_segment)
        
        return climbing_segments
    
    def _get_network_distance(self, u: int, v: int) -> float:
        """Get distance between two nodes using road network paths (same as TSP solver)"""
        if u == v:
            return 0
        
        # Check cache first
        cache_key = (min(u, v), max(u, v))
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]
        
        # Compute distance using shortest path (same method as TSP solver)
        try:
            path = nx.shortest_path(self.graph, u, v, weight='length')
            total_distance = 0
            
            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i+1])
                if edge_data:
                    if isinstance(edge_data, dict) and 0 in edge_data:
                        edge_data = edge_data[0]
                    total_distance += edge_data.get('length', float('inf'))
            
            # Cache the result
            self._distance_cache[cache_key] = total_distance
            return total_distance
            
        except nx.NetworkXNoPath:
            return float('inf')
    
    def get_detailed_route_path(self, route_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get detailed route path including all intermediate nodes along roads
        
        Args:
            route_result: Route result from optimizer
            
        Returns:
            List of coordinate dictionaries for complete route path
        """
        if not route_result or not route_result.get('route'):
            return []
        
        route = route_result['route']
        detailed_path = []
        
        # Add starting node
        if route[0] in self.graph.nodes:
            start_data = self.graph.nodes[route[0]]
            detailed_path.append({
                'latitude': start_data['y'],
                'longitude': start_data['x'],
                'node_id': route[0],
                'elevation': start_data.get('elevation', 0),
                'node_type': 'intersection'
            })
        
        # Add all intermediate nodes for each segment
        for i in range(len(route) - 1):
            current_node = route[i]
            next_node = route[i + 1]
            
            # Get shortest path between intersections
            try:
                path = nx.shortest_path(self.graph, current_node, next_node, weight='length')
                
                # Add all intermediate nodes (skip first node as it's already added)
                for j in range(1, len(path)):
                    node_id = path[j]
                    if node_id in self.graph.nodes:
                        node_data = self.graph.nodes[node_id]
                        detailed_path.append({
                            'latitude': node_data['y'],
                            'longitude': node_data['x'],
                            'node_id': node_id,
                            'elevation': node_data.get('elevation', 0),
                            'node_type': 'intersection' if self.graph.degree(node_id) != 2 else 'geometry'
                        })
            except nx.NetworkXNoPath:
                # If no path found, just connect with straight line (fallback)
                if next_node in self.graph.nodes:
                    next_data = self.graph.nodes[next_node]
                    detailed_path.append({
                        'latitude': next_data['y'],
                        'longitude': next_data['x'],
                        'node_id': next_node,
                        'elevation': next_data.get('elevation', 0),
                        'node_type': 'intersection'
                    })
        
        # Add return path to start
        if len(route) > 1:
            last_node = route[-1]
            start_node = route[0]
            
            try:
                path = nx.shortest_path(self.graph, last_node, start_node, weight='length')
                
                # Add intermediate nodes for return path (skip first node)
                for j in range(1, len(path)):
                    node_id = path[j]
                    if node_id in self.graph.nodes:
                        node_data = self.graph.nodes[node_id]
                        detailed_path.append({
                            'latitude': node_data['y'],
                            'longitude': node_data['x'],
                            'node_id': node_id,
                            'elevation': node_data.get('elevation', 0),
                            'node_type': 'intersection' if self.graph.degree(node_id) != 2 else 'geometry'
                        })
            except nx.NetworkXNoPath:
                # Fallback: return to start
                if start_node in self.graph.nodes:
                    start_data = self.graph.nodes[start_node]
                    detailed_path.append({
                        'latitude': start_data['y'],
                        'longitude': start_data['x'],
                        'node_id': start_node,
                        'elevation': start_data.get('elevation', 0),
                        'node_type': 'intersection'
                    })
        
        return detailed_path
    
    def get_route_geodataframe(self, route: List[int]) -> gpd.GeoDataFrame:
        """Convert route to GeoDataFrame with elevation and spatial data
        
        Args:
            route: List of node IDs
            
        Returns:
            GeoDataFrame with route elevation profile data
        """
        if not route:
            return gpd.GeoDataFrame()
        
        # Check cache first
        route_key = tuple(route)
        if route_key in self._route_gdf_cache:
            return self._route_gdf_cache[route_key]
        
        # Create route data
        route_data = []
        cumulative_distance = 0
        
        for i, node_id in enumerate(route):
            if node_id in self.graph.nodes:
                node_data = self.graph.nodes[node_id]
                
                # Calculate segment distance
                segment_distance = 0
                if i > 0:
                    prev_node = route[i-1]
                    segment_distance = self._get_network_distance(prev_node, node_id)
                    cumulative_distance += segment_distance
                
                route_data.append({
                    'node_id': node_id,
                    'route_index': i,
                    'elevation': node_data.get('elevation', 0),
                    'segment_distance_m': segment_distance,
                    'cumulative_distance_m': cumulative_distance,
                    'geometry': Point(node_data['x'], node_data['y'])
                })
        
        # Create GeoDataFrame
        route_gdf = gpd.GeoDataFrame(route_data, crs='EPSG:4326')
        
        if not route_gdf.empty:
            # Add elevation analysis columns
            route_gdf = self._add_elevation_analysis_columns(route_gdf)
        
        # Cache the result
        self._route_gdf_cache[route_key] = route_gdf
        
        return route_gdf
    
    def _add_elevation_analysis_columns(self, route_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add elevation analysis columns to route GeoDataFrame
        
        Args:
            route_gdf: GeoDataFrame with route data
            
        Returns:
            GeoDataFrame with additional elevation analysis columns
        """
        if route_gdf.empty:
            return route_gdf
        
        route_gdf = route_gdf.copy()
        
        # Calculate elevation changes
        route_gdf['elevation_change_m'] = route_gdf['elevation'].diff().fillna(0)
        
        # Calculate grades (percentage)
        route_gdf['grade_percent'] = np.where(
            route_gdf['segment_distance_m'] > 0,
            (route_gdf['elevation_change_m'] / route_gdf['segment_distance_m']) * 100,
            0
        )
        
        # Classify terrain
        route_gdf['terrain_type'] = np.select([
            route_gdf['grade_percent'] > 15,
            route_gdf['grade_percent'] > 8,
            route_gdf['grade_percent'] > 2,
            route_gdf['grade_percent'] > -2,
            route_gdf['grade_percent'] > -8,
            route_gdf['grade_percent'] > -15
        ], [
            'very_steep_uphill',
            'steep_uphill',
            'moderate_uphill',
            'level',
            'moderate_downhill',
            'steep_downhill'
        ], default='very_steep_downhill')
        
        # Add smoothed elevation (rolling average)
        window_size = min(5, len(route_gdf))
        if window_size > 1:
            route_gdf['elevation_smoothed'] = route_gdf['elevation'].rolling(
                window=window_size, center=True, min_periods=1
            ).mean()
        else:
            route_gdf['elevation_smoothed'] = route_gdf['elevation']
        
        return route_gdf
    
    def generate_profile_data_spatial(self, route_result: Dict[str, Any], 
                                    use_geodataframe: bool = True,
                                    interpolate_points: int = 0) -> Dict[str, Any]:
        """Generate elevation profile data using spatial operations
        
        Args:
            route_result: Route result from optimizer
            use_geodataframe: Whether to use GeoDataFrame-based analysis
            interpolate_points: Number of points to interpolate between nodes
            
        Returns:
            Dictionary with enhanced elevation profile data
        """
        if not use_geodataframe:
            return self.generate_profile_data(route_result)
        
        if not route_result or not route_result.get('route'):
            return {}
        
        route = route_result['route']
        route_gdf = self.get_route_geodataframe(route)
        
        if route_gdf.empty:
            return {}
        
        # Add return segment to start
        if len(route) > 1:
            start_node = route[0]
            end_node = route[-1]
            return_distance = self._get_network_distance(end_node, start_node)
            
            # Add return segment
            start_row = route_gdf.iloc[0].copy()
            start_row['route_index'] = len(route_gdf)
            start_row['segment_distance_m'] = return_distance
            start_row['cumulative_distance_m'] = route_gdf.iloc[-1]['cumulative_distance_m'] + return_distance
            start_row['elevation_change_m'] = start_row['elevation'] - route_gdf.iloc[-1]['elevation']
            start_row['grade_percent'] = (start_row['elevation_change_m'] / return_distance * 100) if return_distance > 0 else 0
            
            # Add to GeoDataFrame
            route_gdf = pd.concat([route_gdf, start_row.to_frame().T], ignore_index=True)
        
        # Interpolate additional points if requested
        if interpolate_points > 0:
            route_gdf = self._interpolate_elevation_points(route_gdf, interpolate_points)
        
        # Calculate enhanced statistics
        elevation_stats = self._calculate_elevation_stats_spatial(route_gdf)
        
        # Convert to standard format
        coordinates = []
        for _, row in route_gdf.iterrows():
            coordinates.append({
                'latitude': row['geometry'].y,
                'longitude': row['geometry'].x,
                'node_id': row['node_id'] if 'node_id' in row else None
            })
        
        return {
            'coordinates': coordinates,
            'elevations': route_gdf['elevation'].tolist(),
            'distances_m': route_gdf['cumulative_distance_m'].tolist(),
            'distances_km': (route_gdf['cumulative_distance_m'] / 1000).tolist(),
            'total_distance_km': route_gdf['cumulative_distance_m'].iloc[-1] / 1000,
            'elevation_stats': elevation_stats,
            'grade_profile': route_gdf['grade_percent'].tolist(),
            'terrain_profile': route_gdf['terrain_type'].tolist(),
            'geodataframe_used': True
        }
    
    def _calculate_elevation_stats_spatial(self, route_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Calculate elevation statistics using spatial operations
        
        Args:
            route_gdf: GeoDataFrame with route elevation data
            
        Returns:
            Dictionary with elevation statistics
        """
        if route_gdf.empty:
            return {}
        
        # Basic elevation statistics
        elevation_stats = {
            'min_elevation': route_gdf['elevation'].min(),
            'max_elevation': route_gdf['elevation'].max(),
            'elevation_range': route_gdf['elevation'].max() - route_gdf['elevation'].min(),
            'avg_elevation': route_gdf['elevation'].mean(),
            'elevation_std': route_gdf['elevation'].std(),
            'median_elevation': route_gdf['elevation'].median()
        }
        
        # Elevation gain/loss analysis
        elevation_gain = route_gdf[route_gdf['elevation_change_m'] > 0]['elevation_change_m'].sum()
        elevation_loss = abs(route_gdf[route_gdf['elevation_change_m'] < 0]['elevation_change_m'].sum())
        
        elevation_stats.update({
            'total_elevation_gain_m': elevation_gain,
            'total_elevation_loss_m': elevation_loss,
            'net_elevation_change_m': elevation_gain - elevation_loss
        })
        
        # Grade analysis
        grade_stats = {
            'max_grade': route_gdf['grade_percent'].max(),
            'min_grade': route_gdf['grade_percent'].min(),
            'avg_grade': route_gdf['grade_percent'].mean(),
            'grade_std': route_gdf['grade_percent'].std()
        }
        
        # Terrain distribution
        terrain_counts = route_gdf['terrain_type'].value_counts()
        terrain_distribution = {}
        total_points = len(route_gdf)
        
        for terrain_type in terrain_counts.index:
            count = terrain_counts[terrain_type]
            terrain_distribution[terrain_type] = {
                'count': count,
                'percentage': (count / total_points * 100) if total_points > 0 else 0
            }
        
        # Steep sections analysis
        steep_sections = self._find_steep_sections_spatial(route_gdf)
        
        elevation_stats.update({
            'grade_statistics': grade_stats,
            'terrain_distribution': terrain_distribution,
            'steep_sections': steep_sections,
            'steep_section_count': len(steep_sections)
        })
        
        return elevation_stats
    
    def _find_steep_sections_spatial(self, route_gdf: gpd.GeoDataFrame, 
                                   min_grade: float = 8.0) -> List[Dict[str, Any]]:
        """Find steep sections using spatial operations
        
        Args:
            route_gdf: GeoDataFrame with route data
            min_grade: Minimum grade to consider steep
            
        Returns:
            List of steep section dictionaries
        """
        if route_gdf.empty:
            return []
        
        steep_sections = []
        
        # Find steep uphill sections
        steep_uphill = route_gdf[route_gdf['grade_percent'] > min_grade]
        if not steep_uphill.empty:
            # Group consecutive steep sections
            steep_groups = self._group_consecutive_sections(steep_uphill)
            for group in steep_groups:
                steep_sections.append({
                    'type': 'uphill',
                    'start_km': group['cumulative_distance_m'].iloc[0] / 1000,
                    'end_km': group['cumulative_distance_m'].iloc[-1] / 1000,
                    'distance_km': (group['cumulative_distance_m'].iloc[-1] - group['cumulative_distance_m'].iloc[0]) / 1000,
                    'max_grade': group['grade_percent'].max(),
                    'avg_grade': group['grade_percent'].mean(),
                    'elevation_change': group['elevation_change_m'].sum()
                })
        
        # Find steep downhill sections
        steep_downhill = route_gdf[route_gdf['grade_percent'] < -min_grade]
        if not steep_downhill.empty:
            steep_groups = self._group_consecutive_sections(steep_downhill)
            for group in steep_groups:
                steep_sections.append({
                    'type': 'downhill',
                    'start_km': group['cumulative_distance_m'].iloc[0] / 1000,
                    'end_km': group['cumulative_distance_m'].iloc[-1] / 1000,
                    'distance_km': (group['cumulative_distance_m'].iloc[-1] - group['cumulative_distance_m'].iloc[0]) / 1000,
                    'max_grade': abs(group['grade_percent'].min()),
                    'avg_grade': abs(group['grade_percent'].mean()),
                    'elevation_change': abs(group['elevation_change_m'].sum())
                })
        
        return steep_sections
    
    def _group_consecutive_sections(self, sections_gdf: gpd.GeoDataFrame) -> List[gpd.GeoDataFrame]:
        """Group consecutive sections
        
        Args:
            sections_gdf: GeoDataFrame with sections
            
        Returns:
            List of GeoDataFrames with grouped sections
        """
        if sections_gdf.empty:
            return []
        
        groups = []
        current_group = []
        
        for i, (idx, row) in enumerate(sections_gdf.iterrows()):
            if i == 0:
                current_group = [idx]
            else:
                # Check if this section is consecutive to the previous
                prev_idx = current_group[-1]
                if idx == prev_idx + 1:
                    current_group.append(idx)
                else:
                    # Start new group
                    if current_group:
                        groups.append(sections_gdf.loc[current_group])
                    current_group = [idx]
        
        # Add last group
        if current_group:
            groups.append(sections_gdf.loc[current_group])
        
        return groups
    
    def _interpolate_elevation_points(self, route_gdf: gpd.GeoDataFrame, 
                                    points_per_segment: int) -> gpd.GeoDataFrame:
        """Interpolate additional elevation points between nodes
        
        Args:
            route_gdf: GeoDataFrame with route data
            points_per_segment: Number of points to interpolate per segment
            
        Returns:
            GeoDataFrame with interpolated points
        """
        if route_gdf.empty or points_per_segment <= 0:
            return route_gdf
        
        interpolated_data = []
        
        for i in range(len(route_gdf) - 1):
            current_row = route_gdf.iloc[i]
            next_row = route_gdf.iloc[i + 1]
            
            # Add current point
            interpolated_data.append(current_row)
            
            # Interpolate between current and next point
            for j in range(1, points_per_segment + 1):
                fraction = j / (points_per_segment + 1)
                
                # Interpolate coordinates
                x_interp = current_row['geometry'].x + fraction * (next_row['geometry'].x - current_row['geometry'].x)
                y_interp = current_row['geometry'].y + fraction * (next_row['geometry'].y - current_row['geometry'].y)
                
                # Interpolate elevation
                elev_interp = current_row['elevation'] + fraction * (next_row['elevation'] - current_row['elevation'])
                
                # Interpolate distance
                dist_interp = current_row['cumulative_distance_m'] + fraction * (next_row['cumulative_distance_m'] - current_row['cumulative_distance_m'])
                
                interpolated_data.append({
                    'node_id': None,  # Interpolated point
                    'route_index': current_row['route_index'] + fraction,
                    'elevation': elev_interp,
                    'segment_distance_m': 0,  # Will be recalculated
                    'cumulative_distance_m': dist_interp,
                    'geometry': Point(x_interp, y_interp)
                })
        
        # Add last point
        interpolated_data.append(route_gdf.iloc[-1])
        
        # Create new GeoDataFrame
        interpolated_gdf = gpd.GeoDataFrame(interpolated_data, crs='EPSG:4326')
        
        # Recalculate segment distances and add analysis columns
        if not interpolated_gdf.empty:
            interpolated_gdf = self._add_elevation_analysis_columns(interpolated_gdf)
        
        return interpolated_gdf
    
    def find_elevation_peaks_valleys_spatial(self, route_result: Dict[str, Any], 
                                           min_prominence: float = 10) -> Dict[str, List]:
        """Find elevation peaks and valleys using spatial operations
        
        Args:
            route_result: Route result from optimizer
            min_prominence: Minimum elevation change to be considered a peak/valley
            
        Returns:
            Dictionary with peaks and valleys with enhanced spatial data
        """
        if not route_result or not route_result.get('route'):
            return {'peaks': [], 'valleys': []}
        
        route = route_result['route']
        route_gdf = self.get_route_geodataframe(route)
        
        if route_gdf.empty:
            return {'peaks': [], 'valleys': []}
        
        peaks = []
        valleys = []
        
        # Use smoothed elevation for better peak/valley detection
        elevations = route_gdf['elevation_smoothed'].values
        distances_km = (route_gdf['cumulative_distance_m'] / 1000).values
        
        # Find local maxima and minima
        for i in range(1, len(elevations) - 1):
            current_elev = elevations[i]
            prev_elev = elevations[i - 1]
            next_elev = elevations[i + 1]
            
            # Check for peak
            if current_elev > prev_elev and current_elev > next_elev:
                prominence = min(current_elev - prev_elev, current_elev - next_elev)
                if prominence >= min_prominence:
                    peaks.append({
                        'distance_km': distances_km[i],
                        'elevation': current_elev,
                        'prominence': prominence,
                        'coordinate': {
                            'latitude': route_gdf.iloc[i]['geometry'].y,
                            'longitude': route_gdf.iloc[i]['geometry'].x
                        },
                        'terrain_type': route_gdf.iloc[i]['terrain_type']
                    })
            
            # Check for valley
            elif current_elev < prev_elev and current_elev < next_elev:
                prominence = min(prev_elev - current_elev, next_elev - current_elev)
                if prominence >= min_prominence:
                    valleys.append({
                        'distance_km': distances_km[i],
                        'elevation': current_elev,
                        'prominence': prominence,
                        'coordinate': {
                            'latitude': route_gdf.iloc[i]['geometry'].y,
                            'longitude': route_gdf.iloc[i]['geometry'].x
                        },
                        'terrain_type': route_gdf.iloc[i]['terrain_type']
                    })
        
        return {
            'peaks': peaks,
            'valleys': valleys,
            'peak_count': len(peaks),
            'valley_count': len(valleys),
            'total_prominence': sum(p['prominence'] for p in peaks) + sum(v['prominence'] for v in valleys)
        }
    
    def get_elevation_zones_spatial(self, route_result: Dict[str, Any], 
                                  zone_count: int = 5) -> List[Dict[str, Any]]:
        """Divide route into elevation zones using spatial operations
        
        Args:
            route_result: Route result from optimizer
            zone_count: Number of zones to create
            
        Returns:
            List of zone dictionaries with enhanced spatial data
        """
        if not route_result or not route_result.get('route'):
            return []
        
        route = route_result['route']
        route_gdf = self.get_route_geodataframe(route)
        
        if route_gdf.empty:
            return []
        
        if len(route_gdf) < zone_count:
            zone_count = len(route_gdf)
        
        zones = []
        points_per_zone = len(route_gdf) // zone_count
        
        for i in range(zone_count):
            start_idx = i * points_per_zone
            if i == zone_count - 1:
                end_idx = len(route_gdf)
            else:
                end_idx = (i + 1) * points_per_zone
            
            zone_gdf = route_gdf.iloc[start_idx:end_idx]
            
            if not zone_gdf.empty:
                # Calculate zone statistics
                zone_stats = {
                    'zone_number': i + 1,
                    'start_km': zone_gdf['cumulative_distance_m'].iloc[0] / 1000,
                    'end_km': zone_gdf['cumulative_distance_m'].iloc[-1] / 1000,
                    'distance_km': (zone_gdf['cumulative_distance_m'].iloc[-1] - zone_gdf['cumulative_distance_m'].iloc[0]) / 1000,
                    'min_elevation': zone_gdf['elevation'].min(),
                    'max_elevation': zone_gdf['elevation'].max(),
                    'avg_elevation': zone_gdf['elevation'].mean(),
                    'elevation_change': zone_gdf['elevation'].iloc[-1] - zone_gdf['elevation'].iloc[0],
                    'elevation_gain': zone_gdf[zone_gdf['elevation_change_m'] > 0]['elevation_change_m'].sum(),
                    'elevation_loss': abs(zone_gdf[zone_gdf['elevation_change_m'] < 0]['elevation_change_m'].sum()),
                    'avg_grade': zone_gdf['grade_percent'].mean(),
                    'max_grade': zone_gdf['grade_percent'].max(),
                    'min_grade': zone_gdf['grade_percent'].min(),
                    'dominant_terrain': zone_gdf['terrain_type'].mode().iloc[0] if not zone_gdf['terrain_type'].mode().empty else 'unknown',
                    'point_count': len(zone_gdf)
                }
                
                zones.append(zone_stats)
        
        return zones