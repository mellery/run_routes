"""
Consolidated Elevation Service

Provides unified elevation profile generation and analysis with 3DEP 1m and SRTM 90m support.

This module consolidates elevation_profiler.py and elevation_profiler_enhanced.py
into a single, streamlined service using the utils.elevation backend.

Key features:
    - High-precision elevation profiles with 3DEP 1m (when available) + SRTM 90m fallback
    - Elevation statistics and difficulty scoring
    - Peak/valley detection
    - Climbing segment identification
    - Elevation zone analysis
    - Detailed route path generation with all intermediate nodes
"""

from typing import List, Dict, Any, Tuple, Optional
import networkx as nx
from utils.elevation import ElevationService as UtilsElevationService, get_elevation_service


class ElevationProfiler:
    """
    Unified elevation profile service with high-precision elevation support.

    Combines elevation profile generation, analysis, and terrain characterization
    using 3DEP 1m elevation data (when available) with SRTM 90m fallback.

    Usage:
        profiler = ElevationProfiler(graph)
        profile = profiler.generate_profile_data(route_result)
        peaks = profiler.find_elevation_peaks_valleys(route_result)
    """

    def __init__(self, graph: nx.Graph, elevation_service: Optional[UtilsElevationService] = None):
        """
        Initialize elevation service.

        Args:
            graph: NetworkX graph with elevation data
            elevation_service: Optional custom elevation service (uses global singleton if None)
        """
        self.graph = graph
        self.elevation_service = elevation_service or get_elevation_service()
        self._distance_cache = {}  # Cache for network distances

    def generate_profile_data(self,
                            route_result: Dict[str, Any],
                            use_enhanced_elevation: bool = True,
                            interpolate_points: bool = False) -> Dict[str, Any]:
        """
        Generate elevation profile data for a route.

        Uses 3DEP 1m elevation data when available, with automatic SRTM 90m fallback.

        Args:
            route_result: Route result from optimizer containing 'route' key
            use_enhanced_elevation: Whether to query elevation service for better accuracy
            interpolate_points: Whether to add interpolated points for smooth profiles

        Returns:
            Dictionary with elevation profile data:
                - coordinates: List of (lat, lon, node_id) dictionaries
                - elevations: List of elevation values in meters
                - distances_m: List of cumulative distances in meters
                - distances_km: List of cumulative distances in kilometers
                - total_distance_km: Total route distance in kilometers
                - elevation_stats: Comprehensive elevation statistics
                - enhanced_profile: Whether enhanced elevation was used
        """
        if not route_result or not route_result.get('route'):
            return {}

        route = route_result['route']

        # Extract coordinates and elevations
        coordinates = []
        elevations = []
        distances = [0]  # Start at 0 distance
        cumulative_distance = 0

        from utils.geometry import haversine_distance

        # Process each node in the route
        for i, node in enumerate(route):
            if node in self.graph.nodes:
                data = self.graph.nodes[node]

                # Store coordinate
                coordinate = {
                    'latitude': data['y'],
                    'longitude': data['x'],
                    'node_id': node
                }
                coordinates.append(coordinate)

                # Get elevation using enhanced sources if requested
                if use_enhanced_elevation and self.elevation_service and self.elevation_service.is_available():
                    try:
                        enhanced_elevation = self.elevation_service.get_elevation(data['y'], data['x'])
                        if enhanced_elevation is not None:
                            elevations.append(enhanced_elevation)
                        else:
                            # Fallback to graph elevation
                            elevations.append(data.get('elevation', 0))
                    except Exception:
                        # Fallback to graph elevation on error
                        elevations.append(data.get('elevation', 0))
                else:
                    # Use graph elevation
                    elevations.append(data.get('elevation', 0))

                # Calculate cumulative distance using road network paths
                if i > 0:
                    prev_node = route[i-1]
                    segment_dist = self._get_network_distance(prev_node, node)
                    cumulative_distance += segment_dist
                    distances.append(cumulative_distance)

        # Add interpolated points for smoother profiles if requested
        if interpolate_points and len(coordinates) > 1:
            try:
                coordinates, elevations, distances = self._interpolate_route_points(
                    coordinates, elevations, distances, use_enhanced_elevation
                )
            except Exception:
                # Skip interpolation on error
                pass

        # Add return to start for complete loop
        if len(route) > 1:
            return_dist = self._get_network_distance(route[-1], route[0])
            cumulative_distance += return_dist
            distances.append(cumulative_distance)
            elevations.append(elevations[0])  # Back to start elevation
            coordinates.append(coordinates[0])  # Return to start coordinate

        # Convert distances to kilometers
        distances_km = [d / 1000 for d in distances]

        # Calculate enhanced statistics
        elevation_stats = self._calculate_elevation_stats(elevations, distances_km)

        return {
            'coordinates': coordinates,
            'elevations': elevations,
            'distances_m': distances,
            'distances_km': distances_km,
            'total_distance_km': distances_km[-1] if distances_km else 0,
            'elevation_stats': elevation_stats,
            'enhanced_profile': use_enhanced_elevation and self.elevation_service.is_available()
        }

    def _interpolate_route_points(self,
                                 coordinates: List[Dict],
                                 elevations: List[float],
                                 distances: List[float],
                                 use_enhanced_elevation: bool) -> Tuple[List[Dict], List[float], List[float]]:
        """
        Add interpolated points between route nodes for smoother elevation profiles.

        Uses elevation service to query real elevation at interpolated points when available.

        Args:
            coordinates: Original coordinate points
            elevations: Original elevation points
            distances: Original distance points
            use_enhanced_elevation: Whether to query elevation service for interpolated points

        Returns:
            Tuple of (interpolated_coordinates, interpolated_elevations, interpolated_distances)
        """
        if len(coordinates) < 2:
            return coordinates, elevations, distances

        # Determine target spacing based on route length
        total_route_distance = distances[-1] if distances else 0

        if total_route_distance <= 2000:  # Routes ≤ 2km: 1m spacing
            target_spacing = 1
        elif total_route_distance <= 5000:  # Routes ≤ 5km: 2m spacing
            target_spacing = 2
        elif total_route_distance <= 10000:  # Routes ≤ 10km: 5m spacing
            target_spacing = 5
        else:  # Long routes: 10m spacing
            target_spacing = 10

        new_coordinates = []
        new_elevations = []
        new_distances = []

        for i in range(len(coordinates) - 1):
            # Add current point
            new_coordinates.append(coordinates[i])
            new_elevations.append(elevations[i])
            new_distances.append(distances[i])

            # Calculate segment distance
            segment_distance = distances[i + 1] - distances[i]

            # Add interpolated points if segment is long enough
            if segment_distance > target_spacing * 2:
                num_interpolated = int(segment_distance // target_spacing)

                for j in range(1, num_interpolated):
                    # Linear interpolation of coordinates
                    ratio = j / num_interpolated

                    lat1, lon1 = coordinates[i]['latitude'], coordinates[i]['longitude']
                    lat2, lon2 = coordinates[i + 1]['latitude'], coordinates[i + 1]['longitude']

                    interp_lat = lat1 + (lat2 - lat1) * ratio
                    interp_lon = lon1 + (lon2 - lon1) * ratio
                    interp_dist = distances[i] + segment_distance * ratio

                    # Get elevation for interpolated point
                    if use_enhanced_elevation and self.elevation_service and self.elevation_service.is_available():
                        try:
                            interp_elevation = self.elevation_service.get_elevation(interp_lat, interp_lon)
                            if interp_elevation is None:
                                # Linear interpolation fallback
                                interp_elevation = elevations[i] + (elevations[i + 1] - elevations[i]) * ratio
                        except:
                            # Linear interpolation fallback
                            interp_elevation = elevations[i] + (elevations[i + 1] - elevations[i]) * ratio
                    else:
                        # Linear interpolation
                        interp_elevation = elevations[i] + (elevations[i + 1] - elevations[i]) * ratio

                    new_coordinates.append({
                        'latitude': interp_lat,
                        'longitude': interp_lon,
                        'node_id': f"interp_{i}_{j}"
                    })
                    new_elevations.append(interp_elevation)
                    new_distances.append(interp_dist)

        # Add final point
        new_coordinates.append(coordinates[-1])
        new_elevations.append(elevations[-1])
        new_distances.append(distances[-1])

        return new_coordinates, new_elevations, new_distances

    def _calculate_elevation_stats(self,
                                  elevations: List[float],
                                  distances_km: List[float]) -> Dict[str, Any]:
        """
        Calculate comprehensive elevation statistics.

        Args:
            elevations: List of elevation values in meters
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

        # Calculate elevation gain/loss and grade information
        total_gain = 0
        total_loss = 0
        grades = []
        steep_sections = []

        for i in range(1, len(elevations)):
            if i < len(distances_km):
                elevation_change = elevations[i] - elevations[i-1]
                distance_change = (distances_km[i] - distances_km[i-1]) * 1000  # Convert to meters

                # Track gain/loss
                if elevation_change > 0:
                    total_gain += elevation_change
                else:
                    total_loss += abs(elevation_change)

                # Calculate grade
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

        # Calculate difficulty score
        distance_km = distances_km[-1] if distances_km else 0
        difficulty_score = self._calculate_difficulty_score(
            total_gain, total_loss, max_grade, distance_km
        )

        # Terrain classification
        elevation_std = (sum((e - avg_elevation)**2 for e in elevations) / len(elevations))**0.5
        if elevation_std < 5:
            terrain_type = 'flat'
        elif elevation_std < 15:
            terrain_type = 'rolling'
        elif elevation_std < 30:
            terrain_type = 'hilly'
        else:
            terrain_type = 'mountainous'

        return {
            'min_elevation': round(min_elevation, 1),
            'max_elevation': round(max_elevation, 1),
            'elevation_range': round(elevation_range, 1),
            'avg_elevation': round(avg_elevation, 1),
            'total_elevation_gain_m': round(total_gain, 1),
            'total_elevation_loss_m': round(total_loss, 1),
            'max_grade': round(max_grade, 1),
            'min_grade': round(min_grade, 1),
            'avg_grade': round(avg_grade, 1),
            'steep_sections': steep_sections,
            'steep_section_count': len(steep_sections),
            'difficulty_score': difficulty_score,
            'terrain_type': terrain_type,
            'elevation_variability': round(elevation_std, 1)
        }

    def _calculate_difficulty_score(self,
                                   gain: float,
                                   loss: float,
                                   max_grade: float,
                                   distance_km: float) -> Dict[str, Any]:
        """
        Calculate route difficulty score based on elevation characteristics.

        Args:
            gain: Total elevation gain in meters
            loss: Total elevation loss in meters
            max_grade: Maximum grade percentage
            distance_km: Total distance in kilometers

        Returns:
            Dictionary with difficulty metrics
        """
        if distance_km == 0:
            return {'score': 0, 'category': 'flat'}

        # Normalize metrics per kilometer
        gain_per_km = gain / distance_km
        loss_per_km = loss / distance_km

        # Calculate composite difficulty score (0-100 scale)
        grade_factor = min(max_grade / 15 * 30, 30)  # Max 30 points for grade
        gain_factor = min(gain_per_km / 50 * 40, 40)  # Max 40 points for gain
        loss_factor = min(loss_per_km / 50 * 20, 20)  # Max 20 points for loss
        terrain_factor = min((gain + loss) / distance_km / 100 * 10, 10)  # Max 10 points for terrain

        score = grade_factor + gain_factor + loss_factor + terrain_factor

        # Categorize difficulty
        if score < 20:
            category = 'easy'
        elif score < 40:
            category = 'moderate'
        elif score < 60:
            category = 'hard'
        elif score < 80:
            category = 'very_hard'
        else:
            category = 'extreme'

        return {
            'score': round(score, 1),
            'category': category,
            'gain_per_km': round(gain_per_km, 1),
            'loss_per_km': round(loss_per_km, 1)
        }

    def get_elevation_zones(self,
                          route_result: Dict[str, Any],
                          zone_count: int = 5) -> List[Dict[str, Any]]:
        """
        Divide route into elevation zones for analysis.

        Args:
            route_result: Route result from optimizer
            zone_count: Number of zones to create

        Returns:
            List of zone dictionaries with elevation statistics
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

    def find_elevation_peaks_valleys(self,
                                   route_result: Dict[str, Any],
                                   min_prominence: float = 10) -> Dict[str, List]:
        """
        Find elevation peaks and valleys in the route.

        Args:
            route_result: Route result from optimizer
            min_prominence: Minimum elevation change to be considered a peak/valley (meters)

        Returns:
            Dictionary with 'peaks' and 'valleys' lists
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

    def get_climbing_segments(self,
                            route_result: Dict[str, Any],
                            min_gain: float = 20) -> List[Dict[str, Any]]:
        """
        Identify continuous climbing segments in the route.

        Args:
            route_result: Route result from optimizer
            min_gain: Minimum elevation gain to be considered a climbing segment (meters)

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

    def get_detailed_route_path(self, route_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get detailed route path including all intermediate nodes along roads.

        Useful for detailed route visualization with complete road geometry.

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
                # If no path found, just connect with direct line (fallback)
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

    def _get_network_distance(self, u: int, v: int) -> float:
        """
        Get distance between two nodes using road network paths.

        Uses the same distance calculation method as the TSP solver for consistency.

        Args:
            u: Source node ID
            v: Target node ID

        Returns:
            Distance in meters
        """
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
