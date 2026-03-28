#!/usr/bin/env python3
"""
Ultra High Resolution Profiler - Generate terrain profiles with 1-meter precision
"""

import math
from typing import List, Dict, Tuple, Any
import networkx as nx


class UltraHighResolutionProfiler:
    """Generate ultra-high resolution terrain profiles matching 3DEP 1m elevation data"""
    
    def __init__(self, graph: nx.Graph):
        """Initialize with graph containing 1m elevation data"""
        self.graph = graph
        
    def generate_ultra_high_resolution_profile(self, route_result: Dict[str, Any], 
                                             target_spacing_m: float = 1.0) -> Dict[str, Any]:
        """Generate terrain profile with ultra-high resolution
        
        Args:
            route_result: Route result from optimizer
            target_spacing_m: Target spacing between points in meters (1.0 = 1 meter)
            
        Returns:
            Dictionary with ultra-high resolution profile data
        """
        if not route_result or not route_result.get('route'):
            return {}
        
        route = route_result['route']
        
        # Get detailed path with all intermediate nodes
        detailed_path = self._get_complete_route_path(route)
        
        if len(detailed_path) < 2:
            return self._fallback_profile(route)
        
        # Generate points at target spacing along the path
        high_res_points = self._interpolate_along_path(detailed_path, target_spacing_m)
        
        # Extract data for terrain profile
        coordinates = []
        elevations = []
        distances_m = []
        
        for point in high_res_points:
            coordinates.append({
                'latitude': point['lat'],
                'longitude': point['lon'],
                'node_id': point.get('node_id', 'interpolated')
            })
            elevations.append(point['elevation'])
            distances_m.append(point['distance'])
        
        # Convert to kilometers
        distances_km = [d / 1000 for d in distances_m]
        
        # Calculate enhanced statistics
        elevation_stats = self._calculate_detailed_stats(elevations, distances_km)
        
        return {
            'coordinates': coordinates,
            'elevations': elevations,
            'distances_m': distances_m,
            'distances_km': distances_km,
            'total_distance_km': distances_km[-1] if distances_km else 0,
            'elevation_stats': elevation_stats,
            'ultra_high_resolution': True,
            'resolution_meters': target_spacing_m,
            'points_generated': len(elevations)
        }
    
    def _get_complete_route_path(self, route: List[int]) -> List[Dict[str, Any]]:
        """Get complete path with all intermediate nodes"""
        complete_path = []
        
        for i in range(len(route)):
            # Add current route node
            node = route[i]
            if node in self.graph.nodes:
                node_data = self.graph.nodes[node]
                complete_path.append({
                    'node_id': node,
                    'lat': node_data['y'],
                    'lon': node_data['x'],
                    'elevation': node_data.get('elevation', 0),
                    'distance': 0  # Will be calculated later
                })
            
            # Add path to next node
            if i < len(route) - 1:
                next_node = route[i + 1]
                try:
                    # Get shortest path between nodes
                    path = nx.shortest_path(self.graph, node, next_node, weight='length')
                    
                    # Add intermediate nodes (skip first as it's already added)
                    for j in range(1, len(path)):
                        intermediate_node = path[j]
                        if intermediate_node in self.graph.nodes:
                            node_data = self.graph.nodes[intermediate_node]
                            complete_path.append({
                                'node_id': intermediate_node,
                                'lat': node_data['y'],
                                'lon': node_data['x'],
                                'elevation': node_data.get('elevation', 0),
                                'distance': 0
                            })
                            
                except nx.NetworkXNoPath:
                    # Skip if no path found
                    continue
        
        # Add return to start
        if len(route) > 1:
            start_node = route[0]
            last_node = route[-1]
            try:
                return_path = nx.shortest_path(self.graph, last_node, start_node, weight='length')
                for j in range(1, len(return_path)):
                    node = return_path[j]
                    if node in self.graph.nodes:
                        node_data = self.graph.nodes[node]
                        complete_path.append({
                            'node_id': node,
                            'lat': node_data['y'],
                            'lon': node_data['x'],
                            'elevation': node_data.get('elevation', 0),
                            'distance': 0
                        })
            except nx.NetworkXNoPath:
                # Fallback: just add start node again
                if start_node in self.graph.nodes:
                    node_data = self.graph.nodes[start_node]
                    complete_path.append({
                        'node_id': start_node,
                        'lat': node_data['y'],
                        'lon': node_data['x'],
                        'elevation': node_data.get('elevation', 0),
                        'distance': 0
                    })
        
        # Calculate cumulative distances
        cumulative_distance = 0
        for i in range(len(complete_path)):
            complete_path[i]['distance'] = cumulative_distance
            
            if i < len(complete_path) - 1:
                # Calculate distance to next point
                current = complete_path[i]
                next_point = complete_path[i + 1]
                
                segment_distance = self._haversine_distance(
                    current['lat'], current['lon'],
                    next_point['lat'], next_point['lon']
                )
                cumulative_distance += segment_distance
        
        return complete_path
    
    def _interpolate_along_path(self, path: List[Dict[str, Any]], 
                               target_spacing_m: float) -> List[Dict[str, Any]]:
        """Interpolate points along path at target spacing"""
        if len(path) < 2:
            return path
        
        interpolated_points = []
        
        for i in range(len(path) - 1):
            current = path[i]
            next_point = path[i + 1]
            
            # Add current point
            interpolated_points.append(current)
            
            # Calculate segment distance
            segment_distance = self._haversine_distance(
                current['lat'], current['lon'],
                next_point['lat'], next_point['lon']
            )
            
            # Add interpolated points if segment is long enough
            if segment_distance > target_spacing_m:
                num_interpolated = int(segment_distance / target_spacing_m)
                
                for j in range(1, num_interpolated):
                    ratio = j / num_interpolated
                    
                    # Linear interpolation of coordinates
                    interp_lat = current['lat'] + (next_point['lat'] - current['lat']) * ratio
                    interp_lon = current['lon'] + (next_point['lon'] - current['lon']) * ratio
                    interp_elevation = current['elevation'] + (next_point['elevation'] - current['elevation']) * ratio
                    interp_distance = current['distance'] + segment_distance * ratio
                    
                    interpolated_points.append({
                        'node_id': f"interp_{i}_{j}",
                        'lat': interp_lat,
                        'lon': interp_lon,
                        'elevation': interp_elevation,
                        'distance': interp_distance
                    })
        
        # Add final point
        interpolated_points.append(path[-1])
        
        return interpolated_points
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in meters"""
        R = 6371000  # Earth radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def _calculate_detailed_stats(self, elevations: List[float], distances_km: List[float]) -> Dict[str, Any]:
        """Calculate detailed elevation statistics"""
        if not elevations:
            return {}
        
        min_elevation = min(elevations)
        max_elevation = max(elevations)
        avg_elevation = sum(elevations) / len(elevations)
        
        # Calculate elevation gain/loss
        total_gain = 0
        total_loss = 0
        max_grade = 0
        
        for i in range(1, len(elevations)):
            elevation_change = elevations[i] - elevations[i-1]
            distance_change = (distances_km[i] - distances_km[i-1]) * 1000
            
            if elevation_change > 0:
                total_gain += elevation_change
            else:
                total_loss += abs(elevation_change)
            
            if distance_change > 0:
                grade = abs(elevation_change / distance_change) * 100
                max_grade = max(max_grade, grade)
        
        return {
            'min_elevation_m': round(min_elevation, 1),
            'max_elevation_m': round(max_elevation, 1),
            'elevation_range_m': round(max_elevation - min_elevation, 1),
            'avg_elevation_m': round(avg_elevation, 1),
            'total_elevation_gain_m': round(total_gain, 1),
            'total_elevation_loss_m': round(total_loss, 1),
            'max_grade_percent': round(max_grade, 1)
        }
    
    def _fallback_profile(self, route: List[int]) -> Dict[str, Any]:
        """Fallback profile for simple routes"""
        coordinates = []
        elevations = []
        distances_m = []
        
        cumulative_distance = 0
        
        for i, node in enumerate(route):
            if node in self.graph.nodes:
                data = self.graph.nodes[node]
                coordinates.append({
                    'latitude': data['y'],
                    'longitude': data['x'],
                    'node_id': node
                })
                elevations.append(data.get('elevation', 0))
                distances_m.append(cumulative_distance)
                
                if i < len(route) - 1:
                    # Simple distance calculation
                    next_node = route[i + 1]
                    if next_node in self.graph.nodes:
                        next_data = self.graph.nodes[next_node]
                        distance = self._haversine_distance(
                            data['y'], data['x'],
                            next_data['y'], next_data['x']
                        )
                        cumulative_distance += distance
        
        distances_km = [d / 1000 for d in distances_m]
        
        return {
            'coordinates': coordinates,
            'elevations': elevations,
            'distances_m': distances_m,
            'distances_km': distances_km,
            'total_distance_km': distances_km[-1] if distances_km else 0,
            'elevation_stats': self._calculate_detailed_stats(elevations, distances_km),
            'ultra_high_resolution': False,
            'resolution_meters': 'variable',
            'points_generated': len(elevations)
        }