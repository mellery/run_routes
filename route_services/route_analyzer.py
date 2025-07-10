#!/usr/bin/env python3
"""
Route Analyzer
Analyzes routes and generates statistics and directions
"""

from typing import Dict, List, Any, Optional, Union
import networkx as nx
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import numpy as np


class RouteAnalyzer:
    """Analyzes routes and generates statistics and directions"""
    
    def __init__(self, graph: nx.Graph):
        """Initialize route analyzer
        
        Args:
            graph: NetworkX graph
        """
        self.graph = graph
        self._nodes_gdf = None  # Cached GeoDataFrame of nodes
        self._edges_gdf = None  # Cached GeoDataFrame of edges
    
    def analyze_route(self, route_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a route and return comprehensive statistics
        
        Args:
            route_result: Route result from optimizer
            
        Returns:
            Dictionary with detailed route analysis
        """
        if not route_result or not route_result.get('route'):
            return {}
        
        route = route_result['route']
        
        # Get basic statistics from route result
        stats = route_result.get('stats', {})
        
        # Calculate additional statistics
        additional_stats = self._calculate_additional_stats(route)
        
        # Merge statistics
        analysis = {
            'basic_stats': stats,
            'additional_stats': additional_stats,
            'route_info': {
                'route_length': len(route),
                'start_node': route[0] if route else None,
                'end_node': route[-1] if len(route) > 1 else None,
                'is_loop': route[0] == route[-1] if len(route) > 1 else False
            }
        }
        
        return analysis
    
    def _calculate_additional_stats(self, route: List[int]) -> Dict[str, Any]:
        """Calculate additional route statistics
        
        Args:
            route: List of node IDs
            
        Returns:
            Dictionary with additional statistics
        """
        if not route or len(route) < 2:
            return {}
        
        from route import haversine_distance
        
        # Initialize counters
        total_segments = 0
        uphill_segments = 0
        downhill_segments = 0
        level_segments = 0
        steepest_uphill = 0
        steepest_downhill = 0
        elevation_changes = []
        
        # Analyze each segment
        for i in range(len(route)):
            current_node = route[i]
            if i < len(route) - 1:
                next_node = route[i + 1]
            else:
                # Return to start
                next_node = route[0]
            
            if current_node in self.graph.nodes and next_node in self.graph.nodes:
                current_elev = self.graph.nodes[current_node].get('elevation', 0)
                next_elev = self.graph.nodes[next_node].get('elevation', 0)
                
                elevation_change = next_elev - current_elev
                elevation_changes.append(elevation_change)
                
                # Calculate distance for grade
                current_data = self.graph.nodes[current_node]
                next_data = self.graph.nodes[next_node]
                distance = haversine_distance(
                    current_data['y'], current_data['x'],
                    next_data['y'], next_data['x']
                )
                
                # Calculate grade percentage
                if distance > 0:
                    grade = (elevation_change / distance) * 100
                    
                    if grade > 2:
                        uphill_segments += 1
                        steepest_uphill = max(steepest_uphill, grade)
                    elif grade < -2:
                        downhill_segments += 1
                        steepest_downhill = min(steepest_downhill, grade)
                    else:
                        level_segments += 1
                
                total_segments += 1
        
        # Calculate summary statistics
        avg_elevation_change = sum(elevation_changes) / len(elevation_changes) if elevation_changes else 0
        
        return {
            'total_segments': total_segments,
            'uphill_segments': uphill_segments,
            'downhill_segments': downhill_segments,
            'level_segments': level_segments,
            'uphill_percentage': (uphill_segments / total_segments * 100) if total_segments > 0 else 0,
            'downhill_percentage': (downhill_segments / total_segments * 100) if total_segments > 0 else 0,
            'steepest_uphill_grade': steepest_uphill,
            'steepest_downhill_grade': steepest_downhill,
            'avg_elevation_change': avg_elevation_change,
            'elevation_changes': elevation_changes
        }
    
    def generate_directions(self, route_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate turn-by-turn directions
        
        Args:
            route_result: Route result from optimizer
            
        Returns:
            List of direction dictionaries
        """
        if not route_result or not route_result.get('route'):
            return []
        
        route = route_result['route']
        directions = []
        
        from route import haversine_distance
        
        # Start instruction
        if route and route[0] in self.graph.nodes:
            start_data = self.graph.nodes[route[0]]
            directions.append({
                'step': 1,
                'type': 'start',
                'instruction': f"Start at intersection (Node {route[0]})",
                'node_id': route[0],
                'elevation': start_data.get('elevation', 0),
                'elevation_change': 0,
                'distance_km': 0.0,
                'cumulative_distance_km': 0.0,
                'terrain': 'start'
            })
        
        # Route segments
        cumulative_distance = 0
        for i in range(1, len(route)):
            if route[i] in self.graph.nodes and route[i-1] in self.graph.nodes:
                curr_data = self.graph.nodes[route[i]]
                prev_data = self.graph.nodes[route[i-1]]
                
                # Calculate segment distance
                segment_dist = haversine_distance(
                    prev_data['y'], prev_data['x'],
                    curr_data['y'], curr_data['x']
                )
                cumulative_distance += segment_dist
                
                # Determine terrain
                elevation_change = curr_data.get('elevation', 0) - prev_data.get('elevation', 0)
                if elevation_change > 5:
                    terrain = "uphill"
                elif elevation_change < -5:
                    terrain = "downhill"
                else:
                    terrain = "level"
                
                directions.append({
                    'step': i + 1,
                    'type': 'continue',
                    'instruction': f"Continue to intersection (Node {route[i]}) - {terrain}",
                    'node_id': route[i],
                    'elevation': curr_data.get('elevation', 0),
                    'elevation_change': elevation_change,
                    'distance_km': segment_dist / 1000,
                    'cumulative_distance_km': cumulative_distance / 1000,
                    'terrain': terrain
                })
        
        # Return to start
        if len(route) > 1:
            # Calculate distance from last node back to start
            last_data = self.graph.nodes[route[-1]]
            start_data = self.graph.nodes[route[0]]
            
            final_segment_dist = haversine_distance(
                last_data['y'], last_data['x'],
                start_data['y'], start_data['x']
            )
            cumulative_distance += final_segment_dist
            
            directions.append({
                'step': len(route) + 1,
                'type': 'finish',
                'instruction': "Return to starting point to complete the loop",
                'node_id': route[0],
                'elevation': directions[0]['elevation'],
                'elevation_change': directions[0]['elevation'] - directions[-1]['elevation'],
                'distance_km': final_segment_dist / 1000,
                'cumulative_distance_km': cumulative_distance / 1000,
                'terrain': 'finish'
            })
        
        return directions
    
    def get_route_difficulty_rating(self, route_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate route difficulty rating
        
        Args:
            route_result: Route result from optimizer
            
        Returns:
            Dictionary with difficulty rating and explanation
        """
        if not route_result:
            return {'rating': 'unknown', 'score': 0, 'factors': []}
        
        stats = route_result.get('stats', {})
        analysis = self.analyze_route(route_result)
        additional_stats = analysis.get('additional_stats', {})
        
        # Difficulty factors
        distance_km = stats.get('total_distance_km', 0)
        elevation_gain = stats.get('total_elevation_gain_m', 0)
        uphill_percentage = additional_stats.get('uphill_percentage', 0)
        steepest_grade = additional_stats.get('steepest_uphill_grade', 0)
        
        # Calculate difficulty score (0-100)
        score = 0
        factors = []
        
        # Distance factor (0-25 points)
        if distance_km > 10:
            score += 25
            factors.append("Very long distance")
        elif distance_km > 5:
            score += 15
            factors.append("Long distance")
        elif distance_km > 2:
            score += 5
            factors.append("Moderate distance")
        
        # Elevation gain factor (0-30 points)
        elevation_per_km = elevation_gain / distance_km if distance_km > 0 else 0
        if elevation_per_km > 100:
            score += 30
            factors.append("Very high elevation gain")
        elif elevation_per_km > 50:
            score += 20
            factors.append("High elevation gain")
        elif elevation_per_km > 20:
            score += 10
            factors.append("Moderate elevation gain")
        
        # Uphill percentage factor (0-25 points)
        if uphill_percentage > 50:
            score += 25
            factors.append("Mostly uphill")
        elif uphill_percentage > 30:
            score += 15
            factors.append("Significant uphill")
        elif uphill_percentage > 15:
            score += 5
            factors.append("Some uphill")
        
        # Steepest grade factor (0-20 points)
        if steepest_grade > 15:
            score += 20
            factors.append("Very steep sections")
        elif steepest_grade > 10:
            score += 15
            factors.append("Steep sections")
        elif steepest_grade > 5:
            score += 5
            factors.append("Moderate inclines")
        
        # Determine rating
        if score >= 70:
            rating = "Very Hard"
        elif score >= 50:
            rating = "Hard"
        elif score >= 30:
            rating = "Moderate"
        elif score >= 15:
            rating = "Easy"
        else:
            rating = "Very Easy"
        
        return {
            'rating': rating,
            'score': score,
            'factors': factors,
            'distance_km': distance_km,
            'elevation_gain': elevation_gain,
            'elevation_per_km': elevation_per_km,
            'uphill_percentage': uphill_percentage,
            'steepest_grade': steepest_grade
        }
    
    def get_nodes_geodataframe(self) -> gpd.GeoDataFrame:
        """Get graph nodes as GeoDataFrame with spatial indexing
        
        Returns:
            GeoDataFrame with all graph nodes
        """
        if self._nodes_gdf is None:
            nodes_data = []
            for node_id, data in self.graph.nodes(data=True):
                nodes_data.append({
                    'node_id': node_id,
                    'elevation': data.get('elevation', 0),
                    'highway': data.get('highway', ''),
                    'degree': self.graph.degree(node_id),
                    'geometry': Point(data['x'], data['y'])
                })
            self._nodes_gdf = gpd.GeoDataFrame(nodes_data, crs='EPSG:4326')
            # Build spatial index for faster queries
            self._nodes_gdf.sindex
        
        return self._nodes_gdf
    
    def get_route_geodataframe(self, route: List[int]) -> gpd.GeoDataFrame:
        """Convert route to GeoDataFrame with spatial operations
        
        Args:
            route: List of node IDs
            
        Returns:
            GeoDataFrame with route nodes and spatial data
        """
        if not route:
            return gpd.GeoDataFrame()
        
        nodes_gdf = self.get_nodes_geodataframe()
        
        # Filter nodes that are in the route
        route_nodes_gdf = nodes_gdf[nodes_gdf['node_id'].isin(route)].copy()
        
        # Maintain route order
        route_nodes_gdf['route_order'] = route_nodes_gdf['node_id'].map(
            {node_id: i for i, node_id in enumerate(route)}
        )
        route_nodes_gdf = route_nodes_gdf.sort_values('route_order').reset_index(drop=True)
        
        # Add route-specific calculations
        route_nodes_gdf['segment_distance'] = 0.0
        route_nodes_gdf['cumulative_distance'] = 0.0
        route_nodes_gdf['elevation_change'] = 0.0
        route_nodes_gdf['grade_percent'] = 0.0
        
        # Calculate distances and grades using vectorized operations
        for i in range(1, len(route_nodes_gdf)):
            prev_geom = route_nodes_gdf.iloc[i-1]['geometry']
            curr_geom = route_nodes_gdf.iloc[i]['geometry']
            
            # Calculate distance (in meters) using geographic distance
            distance_m = self._calculate_geo_distance(prev_geom, curr_geom)
            route_nodes_gdf.loc[i, 'segment_distance'] = distance_m
            route_nodes_gdf.loc[i, 'cumulative_distance'] = (
                route_nodes_gdf.loc[i-1, 'cumulative_distance'] + distance_m
            )
            
            # Calculate elevation change and grade
            elev_change = (route_nodes_gdf.iloc[i]['elevation'] - 
                          route_nodes_gdf.iloc[i-1]['elevation'])
            route_nodes_gdf.loc[i, 'elevation_change'] = elev_change
            
            if distance_m > 0:
                grade = (elev_change / distance_m) * 100
                route_nodes_gdf.loc[i, 'grade_percent'] = grade
        
        return route_nodes_gdf
    
    def analyze_route_spatial(self, route_result: Dict[str, Any], 
                            use_geodataframe: bool = True) -> Dict[str, Any]:
        """Analyze route using spatial operations with GeoDataFrame
        
        Args:
            route_result: Route result from optimizer
            use_geodataframe: Whether to use GeoDataFrame-based analysis
            
        Returns:
            Dictionary with enhanced spatial analysis
        """
        if not use_geodataframe:
            return self.analyze_route(route_result)
        
        if not route_result or not route_result.get('route'):
            return {}
        
        route = route_result['route']
        route_gdf = self.get_route_geodataframe(route)
        
        if route_gdf.empty:
            return {}
        
        # Get basic statistics from route result
        stats = route_result.get('stats', {})
        
        # Calculate enhanced spatial statistics
        spatial_stats = self._calculate_spatial_stats(route_gdf)
        
        # Merge statistics
        analysis = {
            'basic_stats': stats,
            'spatial_stats': spatial_stats,
            'route_info': {
                'route_length': len(route),
                'start_node': route[0] if route else None,
                'end_node': route[-1] if len(route) > 1 else None,
                'is_loop': route[0] == route[-1] if len(route) > 1 else False
            },
            'geodataframe_used': True
        }
        
        return analysis
    
    def _calculate_spatial_stats(self, route_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Calculate spatial statistics using GeoDataFrame operations
        
        Args:
            route_gdf: GeoDataFrame with route data
            
        Returns:
            Dictionary with spatial statistics
        """
        if route_gdf.empty:
            return {}
        
        # Vectorized calculations using pandas/numpy
        total_distance = route_gdf['segment_distance'].sum()
        total_elevation_gain = route_gdf[route_gdf['elevation_change'] > 0]['elevation_change'].sum()
        total_elevation_loss = abs(route_gdf[route_gdf['elevation_change'] < 0]['elevation_change'].sum())
        
        # Grade analysis
        uphill_mask = route_gdf['grade_percent'] > 2
        downhill_mask = route_gdf['grade_percent'] < -2
        level_mask = (route_gdf['grade_percent'] >= -2) & (route_gdf['grade_percent'] <= 2)
        
        uphill_segments = uphill_mask.sum()
        downhill_segments = downhill_mask.sum()
        level_segments = level_mask.sum()
        total_segments = len(route_gdf) - 1  # Exclude first point
        
        # Steep grade analysis
        steep_uphill = route_gdf[route_gdf['grade_percent'] > 8]
        steep_downhill = route_gdf[route_gdf['grade_percent'] < -8]
        
        # Elevation statistics
        elevation_stats = {
            'min_elevation': route_gdf['elevation'].min(),
            'max_elevation': route_gdf['elevation'].max(),
            'elevation_range': route_gdf['elevation'].max() - route_gdf['elevation'].min(),
            'avg_elevation': route_gdf['elevation'].mean(),
            'elevation_std': route_gdf['elevation'].std()
        }
        
        # Terrain classification
        terrain_analysis = self._classify_terrain_vectorized(route_gdf)
        
        return {
            'total_distance_m': total_distance,
            'total_elevation_gain_m': total_elevation_gain,
            'total_elevation_loss_m': total_elevation_loss,
            'net_elevation_change_m': total_elevation_gain - total_elevation_loss,
            'total_segments': total_segments,
            'uphill_segments': uphill_segments,
            'downhill_segments': downhill_segments,
            'level_segments': level_segments,
            'uphill_percentage': (uphill_segments / total_segments * 100) if total_segments > 0 else 0,
            'downhill_percentage': (downhill_segments / total_segments * 100) if total_segments > 0 else 0,
            'level_percentage': (level_segments / total_segments * 100) if total_segments > 0 else 0,
            'steepest_uphill_grade': route_gdf['grade_percent'].max(),
            'steepest_downhill_grade': route_gdf['grade_percent'].min(),
            'avg_grade': route_gdf['grade_percent'].mean(),
            'steep_uphill_count': len(steep_uphill),
            'steep_downhill_count': len(steep_downhill),
            'elevation_stats': elevation_stats,
            'terrain_analysis': terrain_analysis
        }
    
    def _classify_terrain_vectorized(self, route_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Classify terrain using vectorized operations
        
        Args:
            route_gdf: GeoDataFrame with route data
            
        Returns:
            Dictionary with terrain classification
        """
        if route_gdf.empty:
            return {}
        
        # Define terrain categories based on grade
        conditions = [
            route_gdf['grade_percent'] > 15,   # Very steep uphill
            route_gdf['grade_percent'] > 8,    # Steep uphill
            route_gdf['grade_percent'] > 2,    # Moderate uphill
            route_gdf['grade_percent'] > -2,   # Level
            route_gdf['grade_percent'] > -8,   # Moderate downhill
            route_gdf['grade_percent'] > -15,  # Steep downhill
        ]
        
        choices = [
            'very_steep_uphill',
            'steep_uphill', 
            'moderate_uphill',
            'level',
            'moderate_downhill',
            'steep_downhill'
        ]
        
        route_gdf['terrain_type'] = np.select(conditions, choices, default='very_steep_downhill')
        
        # Calculate terrain distribution
        terrain_counts = route_gdf['terrain_type'].value_counts()
        total_points = len(route_gdf)
        
        terrain_distribution = {}
        for terrain_type in choices + ['very_steep_downhill']:
            count = terrain_counts.get(terrain_type, 0)
            terrain_distribution[terrain_type] = {
                'count': count,
                'percentage': (count / total_points * 100) if total_points > 0 else 0
            }
        
        return {
            'terrain_distribution': terrain_distribution,
            'dominant_terrain': terrain_counts.index[0] if not terrain_counts.empty else 'unknown',
            'terrain_diversity': len(terrain_counts)
        }
    
    def _calculate_geo_distance(self, geom1: Point, geom2: Point) -> float:
        """Calculate geographic distance between two points
        
        Args:
            geom1: First point geometry
            geom2: Second point geometry
            
        Returns:
            Distance in meters
        """
        from route import haversine_distance
        return haversine_distance(geom1.y, geom1.x, geom2.y, geom2.x)
    
    def get_route_linestring(self, route: List[int]) -> LineString:
        """Convert route to LineString geometry
        
        Args:
            route: List of node IDs
            
        Returns:
            LineString geometry for the route
        """
        if not route:
            return LineString()
        
        coordinates = []
        for node_id in route:
            if node_id in self.graph.nodes:
                data = self.graph.nodes[node_id]
                coordinates.append((data['x'], data['y']))
        
        # Close the loop by returning to start
        if len(coordinates) > 1:
            coordinates.append(coordinates[0])
        
        return LineString(coordinates) if len(coordinates) > 1 else LineString()
    
    def find_nearby_points_of_interest(self, route_gdf: gpd.GeoDataFrame, 
                                     buffer_distance_m: float = 500) -> Dict[str, Any]:
        """Find points of interest near the route using spatial operations
        
        Args:
            route_gdf: GeoDataFrame with route data
            buffer_distance_m: Buffer distance in meters
            
        Returns:
            Dictionary with nearby POIs analysis
        """
        if route_gdf.empty:
            return {}
        
        # Create buffer around route
        # Note: This is a simplified approach - in production, you'd want to 
        # transform to a projected CRS for accurate buffer calculations
        route_line = LineString([(geom.x, geom.y) for geom in route_gdf['geometry']])
        route_buffer = route_line.buffer(buffer_distance_m / 111000)  # Rough degrees conversion
        
        # Get all nodes within buffer
        all_nodes_gdf = self.get_nodes_geodataframe()
        nearby_nodes = all_nodes_gdf[all_nodes_gdf.geometry.within(route_buffer)]
        
        # Analyze nearby features
        poi_analysis = {
            'total_nearby_nodes': len(nearby_nodes),
            'nearby_intersections': len(nearby_nodes[nearby_nodes['degree'] > 2]),
            'route_accessibility': len(nearby_nodes) / len(route_gdf) if len(route_gdf) > 0 else 0,
            'buffer_distance_m': buffer_distance_m
        }
        
        return poi_analysis