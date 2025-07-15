#!/usr/bin/env python3
"""
Enhanced Elevation Profiler with 3DEP Integration
Generates high-precision elevation profile data using 1m 3DEP data when available
"""

from typing import List, Dict, Any, Tuple, Optional
import networkx as nx
import os
import sys

# Add project root to path for elevation_data_sources import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elevation_data_sources import get_elevation_manager, ElevationDataManager


class EnhancedElevationProfiler:
    """Enhanced elevation profiler with 3DEP 1m resolution support"""
    
    def __init__(self, graph: nx.Graph, elevation_config_path: Optional[str] = None, verbose: bool = True):
        """Initialize enhanced elevation profiler
        
        Args:
            graph: NetworkX graph with elevation data
            elevation_config_path: Optional path to elevation configuration file
            verbose: Whether to show initialization messages
        """
        self.graph = graph
        self.verbose = verbose
        self._distance_cache = {}  # Cache for network distances
        
        # Disable runtime elevation data manager to avoid recursion issues
        # 3DEP elevation data is now pre-computed and stored in graph nodes during cache generation
        self.elevation_manager = None
        self.elevation_source = None
        
        if self.verbose:
            print("📊 Enhanced ElevationProfiler initialized: using pre-computed graph elevation (includes 3DEP when available)")
    
    def generate_profile_data(self, route_result: Dict[str, Any], 
                            use_enhanced_elevation: bool = True,
                            interpolate_points: bool = False) -> Dict[str, Any]:
        """Generate enhanced elevation profile data for a route
        
        Args:
            route_result: Route result from optimizer
            use_enhanced_elevation: Whether to use 3DEP/SRTM elevation data
            interpolate_points: Whether to add interpolated points for smooth profiles
            
        Returns:
            Dictionary with elevation profile data
        """
        if not route_result or not route_result.get('route'):
            return {}
        
        route = route_result['route']
        
        # Extract coordinates and elevations
        coordinates = []
        elevations = []
        distances = []  # Will add 0 when we add first coordinate
        cumulative_distance = 0
        
        from route import haversine_distance
        
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
                
                # Get elevation using enhanced sources if available
                if use_enhanced_elevation and self.elevation_source:
                    try:
                        enhanced_elevation = self.elevation_source.get_elevation(data['y'], data['x'])
                        if enhanced_elevation is not None:
                            elevations.append(enhanced_elevation)
                        else:
                            # Fallback to graph elevation
                            elevations.append(data.get('elevation', 0))
                    except Exception as e:
                        if self.verbose:
                            print(f"⚠️ Enhanced elevation lookup failed for node {node}: {e}")
                        elevations.append(data.get('elevation', 0))
                else:
                    # Use original graph elevation
                    elevations.append(data.get('elevation', 0))
                
                # Calculate cumulative distance using road network paths
                if i == 0:
                    # First node starts at distance 0
                    distances.append(0)
                else:
                    prev_node = route[i-1]
                    segment_dist = self._get_network_distance(prev_node, node)
                    cumulative_distance += segment_dist
                    distances.append(cumulative_distance)
        
        # Add interpolated points for smoother profiles if requested
        if interpolate_points and len(coordinates) > 1:
            try:
                result = self._interpolate_route_points(
                    coordinates, elevations, distances, use_enhanced_elevation
                )
                if isinstance(result, tuple) and len(result) == 3:
                    coordinates, elevations, distances = result
                else:
                    # Skip interpolation if method doesn't return expected format
                    pass
            except Exception as e:
                # Skip interpolation on error
                if self.verbose:
                    print(f"⚠️ Interpolation failed, skipping: {e}")
                pass
        
        # Add return to start for complete loop
        if len(route) > 1:
            return_dist = self._get_network_distance(route[-1], route[0])
            cumulative_distance += return_dist
            distances.append(cumulative_distance)
            elevations.append(elevations[0])  # Back to start elevation
            # Add coordinate for return point (same as start)
            coordinates.append(coordinates[0])
        
        # Convert distances to kilometers
        distances_km = [d / 1000 for d in distances]
        
        # Calculate enhanced statistics
        elevation_stats = self._calculate_enhanced_elevation_stats(elevations, distances_km)
        
        # Add data source information
        data_source_info = self._get_data_source_info()
        
        # Ensure all arrays have the same length for safe unpacking
        min_length = min(len(coordinates), len(elevations), len(distances_km))
        if min_length == 0:
            return {}
        
        # Truncate arrays to the same length to prevent unpacking errors
        coordinates = coordinates[:min_length]
        elevations = elevations[:min_length]
        distances_km = distances_km[:min_length]
        
        # Ensure distances_m array also matches the length
        distances_m = distances[:min_length] if len(distances) > 0 else []
        
        return {
            'coordinates': coordinates,
            'elevations': elevations,
            'distances_m': distances_m,
            'distances_km': distances_km,
            'total_distance_km': distances_km[-1] if distances_km else 0,
            'elevation_stats': elevation_stats,
            'data_source_info': data_source_info,
            'enhanced_profile': use_enhanced_elevation and self.elevation_source is not None
        }
    
    def _interpolate_route_points(self, coordinates: List[Dict], elevations: List[float], 
                                distances: List[float], use_enhanced_elevation: bool) -> Tuple[List[Dict], List[float], List[float]]:
        """Add interpolated points between route nodes for smoother elevation profiles
        
        Args:
            coordinates: Original coordinate points
            elevations: Original elevation points  
            distances: Original distance points
            use_enhanced_elevation: Whether to use enhanced elevation for interpolated points
            
        Returns:
            Tuple of (interpolated_coordinates, interpolated_elevations, interpolated_distances)
        """
        if len(coordinates) < 2:
            return coordinates, elevations, distances
        
        # Target spacing for interpolated points (meters) - match 1m elevation resolution
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
                    if use_enhanced_elevation and self.elevation_source:
                        try:
                            interp_elevation = self.elevation_source.get_elevation(interp_lat, interp_lon)
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
    
    def _calculate_enhanced_elevation_stats(self, elevations: List[float], distances_km: List[float]) -> Dict[str, Any]:
        """Calculate enhanced elevation statistics with 3DEP precision
        
        Args:
            elevations: List of elevation values
            distances_km: List of distance values in kilometers
            
        Returns:
            Dictionary with enhanced elevation statistics
        """
        if not elevations:
            return {}
        
        # Basic elevation statistics
        min_elevation = min(elevations)
        max_elevation = max(elevations)
        elevation_range = max_elevation - min_elevation
        avg_elevation = sum(elevations) / len(elevations)
        
        # Calculate elevation gain/loss
        total_gain = 0
        total_loss = 0
        max_grade = 0
        
        for i in range(1, len(elevations)):
            elevation_change = elevations[i] - elevations[i-1]
            distance_change = (distances_km[i] - distances_km[i-1]) * 1000  # Convert to meters
            
            if elevation_change > 0:
                total_gain += elevation_change
            else:
                total_loss += abs(elevation_change)
            
            # Calculate grade (percentage)
            if distance_change > 0:
                grade = abs(elevation_change / distance_change) * 100
                max_grade = max(max_grade, grade)
        
        # Calculate difficulty metrics
        difficulty_score = self._calculate_difficulty_score(
            total_gain, total_loss, max_grade, distances_km[-1] if distances_km else 0
        )
        
        # Terrain analysis
        terrain_analysis = self._analyze_terrain_characteristics(elevations, distances_km)
        
        return {
            'min_elevation_m': round(min_elevation, 1),
            'max_elevation_m': round(max_elevation, 1),
            'elevation_range_m': round(elevation_range, 1),
            'avg_elevation_m': round(avg_elevation, 1),
            'total_elevation_gain_m': round(total_gain, 1),
            'total_elevation_loss_m': round(total_loss, 1),
            'max_grade_percent': round(max_grade, 1),
            'difficulty_score': difficulty_score,
            'terrain_analysis': terrain_analysis,
            'data_quality': {
                'resolution_m': self.elevation_source.get_resolution() if self.elevation_source else 90,
                'vertical_accuracy_m': 0.3 if (self.elevation_source and 
                                              self.elevation_source.get_resolution() <= 1) else 16,
                'points_analyzed': len(elevations)
            }
        }
    
    def _calculate_difficulty_score(self, gain: float, loss: float, max_grade: float, distance_km: float) -> Dict[str, Any]:
        """Calculate route difficulty score based on elevation characteristics
        
        Args:
            gain: Total elevation gain in meters
            loss: Total elevation loss in meters
            max_grade: Maximum grade percentage
            distance_km: Total distance in kilometers
            
        Returns:
            Dictionary with difficulty metrics
        """
        if distance_km == 0:
            return {'score': 0, 'category': 'flat', 'factors': {}}
        
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
            'factors': {
                'grade_factor': round(grade_factor, 1),
                'gain_factor': round(gain_factor, 1),
                'loss_factor': round(loss_factor, 1),
                'terrain_factor': round(terrain_factor, 1)
            },
            'normalized_metrics': {
                'gain_per_km': round(gain_per_km, 1),
                'loss_per_km': round(loss_per_km, 1),
                'max_grade_percent': round(max_grade, 1)
            }
        }
    
    def _analyze_terrain_characteristics(self, elevations: List[float], distances_km: List[float]) -> Dict[str, Any]:
        """Analyze terrain characteristics using high-resolution elevation data
        
        Args:
            elevations: List of elevation values
            distances_km: List of distance values in kilometers
            
        Returns:
            Dictionary with terrain analysis
        """
        if len(elevations) < 3:
            return {}
        
        # Calculate elevation variation metrics
        elevation_std = (sum((e - sum(elevations)/len(elevations))**2 for e in elevations) / len(elevations))**0.5
        
        # Identify terrain features
        peaks = []
        valleys = []
        flat_sections = []
        
        for i in range(1, len(elevations) - 1):
            prev_elev = elevations[i-1]
            curr_elev = elevations[i]
            next_elev = elevations[i+1]
            
            # Peak detection (local maximum)
            if curr_elev > prev_elev and curr_elev > next_elev and curr_elev - min(prev_elev, next_elev) > 5:
                peaks.append({
                    'distance_km': distances_km[i],
                    'elevation_m': curr_elev,
                    'prominence_m': curr_elev - min(prev_elev, next_elev)
                })
            
            # Valley detection (local minimum)
            elif curr_elev < prev_elev and curr_elev < next_elev and max(prev_elev, next_elev) - curr_elev > 5:
                valleys.append({
                    'distance_km': distances_km[i],
                    'elevation_m': curr_elev,
                    'depth_m': max(prev_elev, next_elev) - curr_elev
                })
        
        # Classify terrain type
        if elevation_std < 5:
            terrain_type = 'flat'
        elif elevation_std < 15:
            terrain_type = 'rolling'
        elif elevation_std < 30:
            terrain_type = 'hilly'
        else:
            terrain_type = 'mountainous'
        
        return {
            'terrain_type': terrain_type,
            'elevation_variability': round(elevation_std, 1),
            'peaks_count': len(peaks),
            'valleys_count': len(valleys),
            'major_peaks': sorted(peaks, key=lambda x: x['prominence_m'], reverse=True)[:3],
            'major_valleys': sorted(valleys, key=lambda x: x['depth_m'], reverse=True)[:3]
        }
    
    def _get_data_source_info(self) -> Dict[str, Any]:
        """Get information about elevation data sources used
        
        Returns:
            Dictionary with data source information
        """
        if not self.elevation_manager:
            return {'source': 'graph_only', 'resolution_m': 'unknown'}
        
        try:
            source_info = self.elevation_manager.get_source_info()
            active_source = source_info.get('active', {})
            
            result = {
                'available_sources': list(self.elevation_manager.get_available_sources()),
                'active_source': active_source.get('type', 'unknown'),
                'resolution_m': active_source.get('resolution', 'unknown')
            }
            
            # Add hybrid source statistics if available
            if hasattr(self.elevation_source, 'get_stats'):
                stats = self.elevation_source.get_stats()
                result['usage_stats'] = stats
            
            return result
            
        except Exception as e:
            return {'source': 'error', 'error': str(e)}
    
    def _get_network_distance(self, node1: int, node2: int) -> float:
        """Get shortest path distance between nodes using cached results
        
        Args:
            node1: First node ID
            node2: Second node ID
            
        Returns:
            Distance in meters
        """
        # Use cache key that works for both directions
        cache_key = tuple(sorted([node1, node2]))
        
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]
        
        try:
            # Calculate shortest path
            path = nx.shortest_path(self.graph, node1, node2, weight='length')
            
            # Sum edge lengths
            total_distance = 0
            for i in range(len(path) - 1):
                # Handle both directed and undirected graphs
                try:
                    # Handle MultiDiGraph edge access properly
                    edge_view = self.graph[path[i]][path[i + 1]]
                    
                    # For MultiDiGraph, edge_view is an AtlasView containing edge keys
                    if hasattr(edge_view, 'keys') and list(edge_view.keys()):
                        # Get the first edge (there may be multiple parallel edges)
                        first_edge_key = list(edge_view.keys())[0]
                        edge_data = edge_view[first_edge_key]
                        edge_length = edge_data.get('length', 0)
                        total_distance += edge_length
                    else:
                        # No edges found
                        total_distance += 0
                        
                except (KeyError, IndexError, AttributeError):
                    # Fallback: use 0 length
                    total_distance += 0
            
            # Cache result
            self._distance_cache[cache_key] = total_distance
            
            return total_distance
            
        except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
            # Fallback to haversine distance if no path found
            try:
                node1_data = self.graph.nodes[node1]
                node2_data = self.graph.nodes[node2]
                
                from route import haversine_distance
                distance = haversine_distance(
                    node1_data['y'], node1_data['x'],
                    node2_data['y'], node2_data['x']
                )
                
                # Cache fallback result
                self._distance_cache[cache_key] = distance
                
                return distance
                
            except KeyError:
                return 0
    
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
            elevation = start_data.get('elevation', 0)
            
            # Use graph elevation for detailed path to avoid slowdown
            # Enhanced elevation lookup is disabled here for performance
            
            detailed_path.append({
                'latitude': start_data['y'],
                'longitude': start_data['x'],
                'node_id': route[0],
                'elevation': elevation,
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
                        elevation = node_data.get('elevation', 0)
                        
                        # Use graph elevation for performance (enhanced elevation disabled for detailed path)
                        
                        detailed_path.append({
                            'latitude': node_data['y'],
                            'longitude': node_data['x'],
                            'node_id': node_id,
                            'elevation': elevation,
                            'node_type': 'intersection' if self.graph.degree(node_id) != 2 else 'geometry'
                        })
            except nx.NetworkXNoPath:
                # If no path found, just connect with straight line (fallback)
                if next_node in self.graph.nodes:
                    next_data = self.graph.nodes[next_node]
                    elevation = next_data.get('elevation', 0)
                    
                    # Try to get enhanced elevation if available
                    if self.elevation_source:
                        try:
                            enhanced_elevation = self.elevation_source.get_elevation(next_data['y'], next_data['x'])
                            if enhanced_elevation is not None:
                                elevation = enhanced_elevation
                        except Exception:
                            pass  # Fall back to graph elevation
                    
                    detailed_path.append({
                        'latitude': next_data['y'],
                        'longitude': next_data['x'],
                        'node_id': next_node,
                        'elevation': elevation,
                        'node_type': 'intersection'
                    })
        
        # Add return path to start
        if len(route) > 1:
            last_node = route[-1]
            start_node = route[0]
            
            # Get shortest path back to start
            try:
                return_path = nx.shortest_path(self.graph, last_node, start_node, weight='length')
                
                # Add nodes from return path (skip first node as it's already added)
                for j in range(1, len(return_path)):
                    node_id = return_path[j]
                    if node_id in self.graph.nodes:
                        node_data = self.graph.nodes[node_id]
                        elevation = node_data.get('elevation', 0)
                        
                        # Use graph elevation for performance (enhanced elevation disabled for detailed path)
                        
                        detailed_path.append({
                            'latitude': node_data['y'],
                            'longitude': node_data['x'],
                            'node_id': node_id,
                            'elevation': elevation,
                            'node_type': 'intersection' if self.graph.degree(node_id) != 2 else 'geometry'
                        })
            except nx.NetworkXNoPath:
                # Fallback: just add start node again
                if start_node in self.graph.nodes:
                    start_data = self.graph.nodes[start_node]
                    elevation = start_data.get('elevation', 0)
                    
                    # Try to get enhanced elevation if available
                    if self.elevation_source:
                        try:
                            enhanced_elevation = self.elevation_source.get_elevation(start_data['y'], start_data['x'])
                            if enhanced_elevation is not None:
                                elevation = enhanced_elevation
                        except Exception:
                            pass  # Fall back to graph elevation
                    
                    detailed_path.append({
                        'latitude': start_data['y'],
                        'longitude': start_data['x'],
                        'node_id': start_node,
                        'elevation': elevation,
                        'node_type': 'intersection'
                    })
        
        return detailed_path
    
    def close(self):
        """Clean up resources"""
        if self.elevation_manager:
            self.elevation_manager.close_all()


# Backwards compatibility wrapper
class ElevationProfiler(EnhancedElevationProfiler):
    """Backwards compatible ElevationProfiler that uses enhanced features automatically"""
    
    def __init__(self, graph: nx.Graph):
        """Initialize with automatic 3DEP enhancement"""
        super().__init__(graph)
    
    def generate_profile_data(self, route_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate profile data with automatic enhancement"""
        return super().generate_profile_data(route_result, use_enhanced_elevation=True, interpolate_points=False)