#!/usr/bin/env python3
"""
Elevation Profiler
Generates elevation profile data from routes
"""

from typing import List, Dict, Any, Tuple
import networkx as nx


class ElevationProfiler:
    """Generates elevation profile data and analysis"""
    
    def __init__(self, graph: nx.Graph):
        """Initialize elevation profiler
        
        Args:
            graph: NetworkX graph with elevation data
        """
        self.graph = graph
        self._distance_cache = {}  # Cache for network distances
    
    def generate_profile_data(self, route_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate elevation profile data for a route
        
        Args:
            route_result: Route result from optimizer
            
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