#!/usr/bin/env python3
"""
Precision-Aware Genetic Operators for 1m Elevation Data
Enhanced crossover and mutation operators that leverage micro-terrain features
"""

import numpy as np
import networkx as nx
import random
from typing import List, Tuple, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from ga_precision_fitness import PrecisionElevationAnalyzer, EnhancedGAFitnessEvaluator
    PRECISION_FITNESS_AVAILABLE = True
except ImportError:
    PRECISION_FITNESS_AVAILABLE = False


class PrecisionAwareCrossover:
    """Crossover operators that leverage 1m elevation precision for better offspring"""
    
    def __init__(self, graph: nx.Graph, elevation_analyzer: Optional['PrecisionElevationAnalyzer'] = None):
        """Initialize precision-aware crossover
        
        Args:
            graph: NetworkX graph with route network
            elevation_analyzer: Optional elevation analyzer for micro-terrain detection
        """
        self.graph = graph
        self.elevation_analyzer = elevation_analyzer
        
        # Crossover parameters
        self.micro_terrain_preference = 0.3  # Probability of preferring micro-terrain features
        self.elevation_similarity_threshold = 5.0  # meters
        
    def terrain_guided_crossover(self, parent1_route: List[int], parent2_route: List[int], 
                                target_distance_km: float = 5.0) -> Tuple[List[int], List[int]]:
        """Crossover that preserves micro-terrain features from both parents
        
        Args:
            parent1_route: First parent route (list of node IDs)
            parent2_route: Second parent route (list of node IDs)
            target_distance_km: Target distance for offspring
            
        Returns:
            Tuple of two offspring routes
        """
        if not self.elevation_analyzer or not PRECISION_FITNESS_AVAILABLE:
            return self._fallback_crossover(parent1_route, parent2_route)
        
        try:
            # Analyze micro-terrain features in both parents
            parent1_coords = self._route_to_coordinates(parent1_route)
            parent2_coords = self._route_to_coordinates(parent2_route)
            
            parent1_profile = self.elevation_analyzer.get_high_resolution_elevation_profile(parent1_coords)
            parent2_profile = self.elevation_analyzer.get_high_resolution_elevation_profile(parent2_coords)
            
            # Extract interesting terrain features
            parent1_features = self._extract_terrain_features(parent1_route, parent1_profile)
            parent2_features = self._extract_terrain_features(parent2_route, parent2_profile)
            
            # Create offspring by combining terrain features
            offspring1 = self._combine_terrain_features(parent1_features, parent2_features, target_distance_km)
            offspring2 = self._combine_terrain_features(parent2_features, parent1_features, target_distance_km)
            
            # Validate and repair offspring
            offspring1 = self._repair_route(offspring1, target_distance_km)
            offspring2 = self._repair_route(offspring2, target_distance_km)
            
            return offspring1, offspring2
            
        except Exception as e:
            logger.warning(f"Terrain-guided crossover failed: {e}")
            return self._fallback_crossover(parent1_route, parent2_route)
    
    def elevation_segment_crossover(self, parent1_route: List[int], parent2_route: List[int]) -> Tuple[List[int], List[int]]:
        """Crossover that exchanges elevation-similar segments between parents
        
        Args:
            parent1_route: First parent route
            parent2_route: Second parent route
            
        Returns:
            Tuple of two offspring routes
        """
        if len(parent1_route) < 3 or len(parent2_route) < 3:
            return self._fallback_crossover(parent1_route, parent2_route)
        
        try:
            # Find elevation-similar segments in both parents
            p1_segments = self._identify_elevation_segments(parent1_route)
            p2_segments = self._identify_elevation_segments(parent2_route)
            
            # Match segments with similar elevation characteristics
            segment_matches = self._match_elevation_segments(p1_segments, p2_segments)
            
            if not segment_matches:
                return self._fallback_crossover(parent1_route, parent2_route)
            
            # Exchange matched segments
            offspring1 = parent1_route.copy()
            offspring2 = parent2_route.copy()
            
            # Randomly select segments to exchange
            exchange_pairs = random.sample(segment_matches, 
                                         min(len(segment_matches), 
                                             max(1, len(segment_matches) // 2)))
            
            for (p1_seg, p2_seg) in exchange_pairs:
                # Exchange segments between offspring
                p1_start, p1_end = p1_seg['range']
                p2_start, p2_end = p2_seg['range']
                
                # Extract segments
                p1_segment = parent1_route[p1_start:p1_end+1]
                p2_segment = parent2_route[p2_start:p2_end+1]
                
                # Replace segments (with validation)
                if self._can_replace_segment(offspring1, p1_start, p1_end, p2_segment):
                    offspring1[p1_start:p1_end+1] = p2_segment
                
                if self._can_replace_segment(offspring2, p2_start, p2_end, p1_segment):
                    offspring2[p2_start:p2_end+1] = p1_segment
            
            # Ensure routes return to start
            if offspring1 and offspring1[-1] != offspring1[0]:
                offspring1.append(offspring1[0])
            if offspring2 and offspring2[-1] != offspring2[0]:
                offspring2.append(offspring2[0])
            
            return offspring1, offspring2
            
        except Exception as e:
            logger.warning(f"Elevation segment crossover failed: {e}")
            return self._fallback_crossover(parent1_route, parent2_route)
    
    def _extract_terrain_features(self, route: List[int], elevation_profile: Dict) -> Dict[str, Any]:
        """Extract interesting terrain features from a route
        
        Args:
            route: Route as list of node IDs
            elevation_profile: High-resolution elevation profile
            
        Returns:
            Dictionary with terrain features and corresponding route segments
        """
        features = elevation_profile.get('micro_terrain_features', {})
        
        # Map features back to route nodes
        feature_nodes = {
            'peak_nodes': [],
            'valley_nodes': [],
            'steep_segment_nodes': [],
            'interesting_segments': []
        }
        
        # Extract nodes near peaks
        for peak in features.get('peaks', []):
            peak_coord = peak.get('coordinate')
            if peak_coord:
                nearest_node = self._find_nearest_route_node(route, peak_coord)
                if nearest_node is not None:
                    feature_nodes['peak_nodes'].append({
                        'node': nearest_node,
                        'prominence': peak.get('prominence_m', 0),
                        'coordinate': peak_coord
                    })
        
        # Extract nodes near valleys
        for valley in features.get('valleys', []):
            valley_coord = valley.get('coordinate')
            if valley_coord:
                nearest_node = self._find_nearest_route_node(route, valley_coord)
                if nearest_node is not None:
                    feature_nodes['valley_nodes'].append({
                        'node': nearest_node,
                        'depth': valley.get('depth_m', 0),
                        'coordinate': valley_coord
                    })
        
        # Extract steep segment nodes
        for steep in features.get('steep_sections', []):
            start_coord = steep.get('start_coordinate')
            end_coord = steep.get('end_coordinate')
            if start_coord and end_coord:
                start_node = self._find_nearest_route_node(route, start_coord)
                end_node = self._find_nearest_route_node(route, end_coord)
                if start_node is not None and end_node is not None:
                    feature_nodes['steep_segment_nodes'].append({
                        'start_node': start_node,
                        'end_node': end_node,
                        'grade_percent': steep.get('grade_percent', 0)
                    })
        
        return feature_nodes
    
    def _combine_terrain_features(self, primary_features: Dict, secondary_features: Dict, 
                                target_distance_km: float) -> List[int]:
        """Combine terrain features from two parents to create offspring
        
        Args:
            primary_features: Primary parent's terrain features
            secondary_features: Secondary parent's terrain features
            target_distance_km: Target distance for offspring
            
        Returns:
            Offspring route combining best terrain features
        """
        # Start with primary parent's most interesting features
        offspring_nodes = []
        
        # Add peak nodes from primary parent
        peak_nodes = primary_features.get('peak_nodes', [])
        if peak_nodes:
            # Sort by prominence and take best ones
            best_peaks = sorted(peak_nodes, key=lambda x: x.get('prominence', 0), reverse=True)
            offspring_nodes.extend([p['node'] for p in best_peaks[:3]])  # Top 3 peaks
        
        # Add interesting peaks from secondary parent
        secondary_peaks = secondary_features.get('peak_nodes', [])
        if secondary_peaks:
            best_secondary = sorted(secondary_peaks, key=lambda x: x.get('prominence', 0), reverse=True)
            # Add if not already included
            for peak in best_secondary[:2]:  # Top 2 from secondary
                if peak['node'] not in offspring_nodes:
                    offspring_nodes.append(peak['node'])
        
        # Add steep segments for variety
        steep_segments = primary_features.get('steep_segment_nodes', [])
        if steep_segments:
            best_steep = max(steep_segments, key=lambda x: abs(x.get('grade_percent', 0)))
            if best_steep['start_node'] not in offspring_nodes:
                offspring_nodes.append(best_steep['start_node'])
            if best_steep['end_node'] not in offspring_nodes:
                offspring_nodes.append(best_steep['end_node'])
        
        # If not enough nodes, add some random ones
        if len(offspring_nodes) < 5:
            all_nodes = list(self.graph.nodes())
            additional_nodes = random.sample(all_nodes, min(3, len(all_nodes)))
            offspring_nodes.extend([n for n in additional_nodes if n not in offspring_nodes])
        
        # Create a valid route connecting these nodes
        if len(offspring_nodes) >= 2:
            return self._create_connecting_route(offspring_nodes, target_distance_km)
        else:
            # Fallback to random route
            return self._generate_random_route(target_distance_km)
    
    def _identify_elevation_segments(self, route: List[int]) -> List[Dict[str, Any]]:
        """Identify elevation-characterized segments in a route
        
        Args:
            route: Route as list of node IDs
            
        Returns:
            List of elevation segments with characteristics
        """
        if len(route) < 3:
            return []
        
        segments = []
        segment_size = 3  # Minimum segment size
        
        for i in range(0, len(route) - segment_size + 1, segment_size):
            end_idx = min(i + segment_size, len(route) - 1)
            segment_nodes = route[i:end_idx+1]
            
            # Calculate segment elevation characteristics
            elevations = []
            for node in segment_nodes:
                if node in self.graph.nodes:
                    elevation = self.graph.nodes[node].get('elevation', 0)
                    elevations.append(elevation)
            
            if elevations:
                segment_info = {
                    'range': (i, end_idx),
                    'nodes': segment_nodes,
                    'avg_elevation': np.mean(elevations),
                    'elevation_range': max(elevations) - min(elevations),
                    'start_elevation': elevations[0],
                    'end_elevation': elevations[-1],
                    'elevation_gain': elevations[-1] - elevations[0]
                }
                segments.append(segment_info)
        
        return segments
    
    def _match_elevation_segments(self, segments1: List[Dict], segments2: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """Match segments with similar elevation characteristics
        
        Args:
            segments1: Segments from first parent
            segments2: Segments from second parent
            
        Returns:
            List of matched segment pairs
        """
        matches = []
        
        for seg1 in segments1:
            for seg2 in segments2:
                # Calculate elevation similarity
                avg_diff = abs(seg1['avg_elevation'] - seg2['avg_elevation'])
                range_diff = abs(seg1['elevation_range'] - seg2['elevation_range'])
                gain_diff = abs(seg1['elevation_gain'] - seg2['elevation_gain'])
                
                # Check if segments are similar
                if (avg_diff <= self.elevation_similarity_threshold and
                    range_diff <= self.elevation_similarity_threshold and
                    gain_diff <= self.elevation_similarity_threshold):
                    matches.append((seg1, seg2))
        
        return matches
    
    def _can_replace_segment(self, route: List[int], start_idx: int, end_idx: int, 
                           new_segment: List[int]) -> bool:
        """Check if a segment can be safely replaced in a route
        
        Args:
            route: Current route
            start_idx: Start index of segment to replace
            end_idx: End index of segment to replace
            new_segment: New segment to insert
            
        Returns:
            True if replacement is valid
        """
        # Basic validity checks
        if start_idx < 0 or end_idx >= len(route) or start_idx > end_idx:
            return False
        
        if not new_segment:
            return False
        
        # Check if new segment nodes exist in graph
        for node in new_segment:
            if node not in self.graph.nodes:
                return False
        
        # Check connectivity (simplified)
        if start_idx > 0:
            prev_node = route[start_idx - 1]
            if not self.graph.has_edge(prev_node, new_segment[0]):
                return False
        
        if end_idx < len(route) - 1:
            next_node = route[end_idx + 1]
            if not self.graph.has_edge(new_segment[-1], next_node):
                return False
        
        return True
    
    def _find_nearest_route_node(self, route: List[int], coordinate: Tuple[float, float]) -> Optional[int]:
        """Find the nearest route node to a coordinate
        
        Args:
            route: Route as list of node IDs
            coordinate: (lat, lon) coordinate
            
        Returns:
            Nearest node ID or None
        """
        if not coordinate or len(coordinate) != 2:
            return None
        
        target_lat, target_lon = coordinate
        min_distance = float('inf')
        nearest_node = None
        
        for node in route:
            if node in self.graph.nodes:
                node_data = self.graph.nodes[node]
                node_lat = node_data.get('y', 0)
                node_lon = node_data.get('x', 0)
                
                # Calculate simple distance
                distance = ((node_lat - target_lat)**2 + (node_lon - target_lon)**2)**0.5
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_node = node
        
        return nearest_node
    
    def _create_connecting_route(self, nodes: List[int], target_distance_km: float) -> List[int]:
        """Create a route connecting specified nodes
        
        Args:
            nodes: List of nodes to connect
            target_distance_km: Target distance
            
        Returns:
            Connected route
        """
        if len(nodes) < 2:
            return nodes
        
        # Simple TSP-like connection
        route = [nodes[0]]  # Start with first node
        remaining = nodes[1:].copy()
        
        current_node = nodes[0]
        while remaining:
            # Find nearest remaining node
            nearest_node = min(remaining, 
                             key=lambda n: self._graph_distance(current_node, n))
            route.append(nearest_node)
            remaining.remove(nearest_node)
            current_node = nearest_node
        
        # Return to start
        route.append(nodes[0])
        
        return route
    
    def _graph_distance(self, node1: int, node2: int) -> float:
        """Calculate distance between two nodes"""
        if node1 not in self.graph.nodes or node2 not in self.graph.nodes:
            return float('inf')
        
        try:
            # Try shortest path
            path_length = nx.shortest_path_length(self.graph, node1, node2, weight='length')
            return path_length
        except:
            # Fallback to Euclidean distance
            data1 = self.graph.nodes[node1]
            data2 = self.graph.nodes[node2]
            lat1, lon1 = data1.get('y', 0), data1.get('x', 0)
            lat2, lon2 = data2.get('y', 0), data2.get('x', 0)
            return ((lat1 - lat2)**2 + (lon1 - lon2)**2)**0.5
    
    def _route_to_coordinates(self, route: List[int]) -> List[Tuple[float, float]]:
        """Convert route nodes to coordinates
        
        Args:
            route: Route as list of node IDs
            
        Returns:
            List of (lat, lon) coordinates
        """
        coordinates = []
        for node in route:
            if node in self.graph.nodes:
                data = self.graph.nodes[node]
                lat = data.get('y', 0)
                lon = data.get('x', 0)
                coordinates.append((lat, lon))
        return coordinates
    
    def _repair_route(self, route: List[int], target_distance_km: float) -> List[int]:
        """Repair and validate a route
        
        Args:
            route: Route to repair
            target_distance_km: Target distance
            
        Returns:
            Repaired route
        """
        if not route:
            return self._generate_random_route(target_distance_km)
        
        # Remove invalid nodes
        valid_route = [node for node in route if node in self.graph.nodes]
        
        if len(valid_route) < 2:
            return self._generate_random_route(target_distance_km)
        
        # Ensure route forms a loop
        if valid_route[0] != valid_route[-1]:
            valid_route.append(valid_route[0])
        
        return valid_route
    
    def _generate_random_route(self, target_distance_km: float) -> List[int]:
        """Generate a random route as fallback
        
        Args:
            target_distance_km: Target distance
            
        Returns:
            Random route
        """
        all_nodes = list(self.graph.nodes())
        if not all_nodes:
            return []
        
        # Estimate number of nodes needed
        num_nodes = max(3, min(10, int(target_distance_km * 2)))
        
        selected_nodes = random.sample(all_nodes, min(num_nodes, len(all_nodes)))
        return selected_nodes + [selected_nodes[0]]  # Close the loop
    
    def _fallback_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Simple fallback crossover
        
        Args:
            parent1: First parent route
            parent2: Second parent route
            
        Returns:
            Two offspring routes
        """
        # Simple single-point crossover
        if len(parent1) < 2 or len(parent2) < 2:
            return parent1.copy(), parent2.copy()
        
        # Find crossover point
        max_len = min(len(parent1), len(parent2))
        crossover_point = random.randint(1, max_len - 1)
        
        # Create offspring
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        # Ensure routes end at start
        if offspring1 and offspring1[-1] != offspring1[0]:
            offspring1.append(offspring1[0])
        if offspring2 and offspring2[-1] != offspring2[0]:
            offspring2.append(offspring2[0])
        
        return offspring1, offspring2


class PrecisionAwareMutation:
    """Mutation operators that leverage 1m elevation data for better route exploration"""
    
    def __init__(self, graph: nx.Graph, elevation_analyzer: Optional['PrecisionElevationAnalyzer'] = None):
        """Initialize precision-aware mutation
        
        Args:
            graph: NetworkX graph with route network
            elevation_analyzer: Optional elevation analyzer
        """
        self.graph = graph
        self.elevation_analyzer = elevation_analyzer
        
        # Mutation parameters
        self.elevation_bias_strength = 0.4  # Strength of elevation-based bias
        self.micro_feature_attraction = 0.3  # Attraction to micro-terrain features
        
    def elevation_guided_mutation(self, route: List[int], mutation_rate: float = 0.1, 
                                target_distance_km: float = 5.0) -> List[int]:
        """Mutation that seeks elevation gains and micro-terrain features
        
        Args:
            route: Route to mutate
            mutation_rate: Probability of mutating each position
            target_distance_km: Target distance
            
        Returns:
            Mutated route
        """
        if not route or len(route) < 3:
            return route.copy()
        
        mutated_route = route.copy()
        
        try:
            # Identify mutation candidates
            for i in range(1, len(route) - 1):  # Skip start/end
                if random.random() < mutation_rate:
                    current_node = route[i]
                    
                    # Find elevation-biased replacement
                    replacement_node = self._find_elevation_biased_node(
                        current_node, route, target_distance_km
                    )
                    
                    if replacement_node and replacement_node != current_node:
                        mutated_route[i] = replacement_node
            
            # Ensure route validity
            return self._validate_mutated_route(mutated_route)
            
        except Exception as e:
            logger.warning(f"Elevation-guided mutation failed: {e}")
            return route.copy()
    
    def micro_terrain_insertion_mutation(self, route: List[int], 
                                       target_distance_km: float = 5.0) -> List[int]:
        """Mutation that inserts nodes near interesting micro-terrain features
        
        Args:
            route: Route to mutate
            target_distance_km: Target distance
            
        Returns:
            Mutated route with micro-terrain insertions
        """
        if not self.elevation_analyzer or not PRECISION_FITNESS_AVAILABLE:
            return self._fallback_mutation(route)
        
        try:
            # Get route coordinates and analyze terrain
            route_coords = self._route_to_coordinates(route)
            elevation_profile = self.elevation_analyzer.get_high_resolution_elevation_profile(route_coords)
            
            # Find interesting micro-features near the route
            nearby_features = self._find_nearby_micro_features(route, elevation_profile)
            
            if not nearby_features:
                return route.copy()
            
            # Insert nodes near interesting features
            mutated_route = route.copy()
            
            for feature in nearby_features[:2]:  # Limit to 2 insertions
                feature_coord = feature.get('coordinate')
                if feature_coord:
                    # Find nearest graph node to this feature
                    feature_node = self._find_nearest_graph_node(feature_coord)
                    
                    if feature_node and feature_node not in mutated_route:
                        # Find best insertion point
                        insertion_point = self._find_best_insertion_point(
                            mutated_route, feature_node
                        )
                        
                        if insertion_point is not None:
                            mutated_route.insert(insertion_point, feature_node)
            
            return self._validate_mutated_route(mutated_route)
            
        except Exception as e:
            logger.warning(f"Micro-terrain insertion mutation failed: {e}")
            return route.copy()
    
    def grade_optimization_mutation(self, route: List[int], 
                                  prefer_steep: bool = True) -> List[int]:
        """Mutation that optimizes for specific grade characteristics
        
        Args:
            route: Route to mutate
            prefer_steep: Whether to prefer steeper grades
            
        Returns:
            Mutated route optimized for grade preferences
        """
        if len(route) < 3:
            return route.copy()
        
        try:
            mutated_route = route.copy()
            
            # Analyze current route grades
            route_grades = self._calculate_route_grades(route)
            
            # Find segments that don't meet grade preferences
            target_segments = []
            for i, grade in enumerate(route_grades):
                grade_percent = abs(grade * 100)
                
                if prefer_steep and grade_percent < 3.0:  # Too flat
                    target_segments.append(i)
                elif not prefer_steep and grade_percent > 8.0:  # Too steep
                    target_segments.append(i)
            
            # Mutate target segments
            for segment_idx in target_segments[:2]:  # Limit mutations
                if segment_idx < len(route) - 1:
                    current_node = route[segment_idx + 1]  # Node at end of segment
                    
                    # Find replacement with better grade characteristics
                    replacement = self._find_grade_optimized_node(
                        route[segment_idx], current_node, prefer_steep
                    )
                    
                    if replacement and replacement != current_node:
                        mutated_route[segment_idx + 1] = replacement
            
            return self._validate_mutated_route(mutated_route)
            
        except Exception as e:
            logger.warning(f"Grade optimization mutation failed: {e}")
            return route.copy()
    
    def _find_elevation_biased_node(self, current_node: int, route: List[int], 
                                  target_distance_km: float) -> Optional[int]:
        """Find a replacement node biased towards higher elevation
        
        Args:
            current_node: Current node to replace
            route: Full route context
            target_distance_km: Target distance
            
        Returns:
            Replacement node or None
        """
        if current_node not in self.graph.nodes:
            return None
        
        current_data = self.graph.nodes[current_node]
        current_elevation = current_data.get('elevation', 0)
        
        # Find nearby nodes
        try:
            # Get neighbors within reasonable distance
            neighbors = []
            for node in self.graph.nodes():
                if node != current_node and node not in route:
                    distance = self._graph_distance(current_node, node)
                    if distance < 500:  # 500m max distance
                        neighbors.append((node, distance))
            
            if not neighbors:
                return None
            
            # Sort by elevation (prefer higher)
            elevation_candidates = []
            for node, distance in neighbors:
                node_data = self.graph.nodes[node]
                node_elevation = node_data.get('elevation', 0)
                elevation_gain = node_elevation - current_elevation
                
                # Bias towards elevation gain
                score = elevation_gain + random.uniform(-10, 10)  # Add some randomness
                elevation_candidates.append((node, score))
            
            # Select based on elevation bias
            elevation_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Choose from top candidates with some randomness
            top_candidates = elevation_candidates[:min(5, len(elevation_candidates))]
            return random.choice(top_candidates)[0]
            
        except Exception as e:
            logger.warning(f"Elevation-biased node search failed: {e}")
            return None
    
    def _find_nearby_micro_features(self, route: List[int], 
                                  elevation_profile: Dict) -> List[Dict[str, Any]]:
        """Find micro-terrain features near the route
        
        Args:
            route: Current route
            elevation_profile: Elevation profile with micro-features
            
        Returns:
            List of nearby interesting features
        """
        features = elevation_profile.get('micro_terrain_features', {})
        nearby_features = []
        
        # Get route bounds
        route_coords = self._route_to_coordinates(route)
        if not route_coords:
            return []
        
        route_lats = [coord[0] for coord in route_coords]
        route_lons = [coord[1] for coord in route_coords]
        route_bounds = {
            'min_lat': min(route_lats),
            'max_lat': max(route_lats),
            'min_lon': min(route_lons),
            'max_lon': max(route_lons)
        }
        
        # Expand bounds slightly to find nearby features
        lat_margin = (route_bounds['max_lat'] - route_bounds['min_lat']) * 0.2
        lon_margin = (route_bounds['max_lon'] - route_bounds['min_lon']) * 0.2
        
        # Check peaks
        for peak in features.get('peaks', []):
            coord = peak.get('coordinate')
            if coord and self._is_near_route_bounds(coord, route_bounds, lat_margin, lon_margin):
                peak['feature_type'] = 'peak'
                nearby_features.append(peak)
        
        # Check valleys (might be interesting for variety)
        for valley in features.get('valleys', []):
            coord = valley.get('coordinate')
            if coord and self._is_near_route_bounds(coord, route_bounds, lat_margin, lon_margin):
                valley['feature_type'] = 'valley'
                nearby_features.append(valley)
        
        # Sort by interest (prominence for peaks, depth for valleys)
        nearby_features.sort(key=lambda f: f.get('prominence_m', f.get('depth_m', 0)), reverse=True)
        
        return nearby_features
    
    def _is_near_route_bounds(self, coord: Tuple[float, float], bounds: Dict, 
                            lat_margin: float, lon_margin: float) -> bool:
        """Check if coordinate is near route bounds
        
        Args:
            coord: (lat, lon) coordinate
            bounds: Route bounds dictionary
            lat_margin: Latitude margin
            lon_margin: Longitude margin
            
        Returns:
            True if coordinate is near route
        """
        lat, lon = coord
        return (bounds['min_lat'] - lat_margin <= lat <= bounds['max_lat'] + lat_margin and
                bounds['min_lon'] - lon_margin <= lon <= bounds['max_lon'] + lon_margin)
    
    def _find_nearest_graph_node(self, coordinate: Tuple[float, float]) -> Optional[int]:
        """Find nearest graph node to a coordinate
        
        Args:
            coordinate: (lat, lon) coordinate
            
        Returns:
            Nearest node ID or None
        """
        if not coordinate:
            return None
        
        target_lat, target_lon = coordinate
        min_distance = float('inf')
        nearest_node = None
        
        # Search within reasonable bounds
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            node_lat = node_data.get('y', 0)
            node_lon = node_data.get('x', 0)
            
            distance = ((node_lat - target_lat)**2 + (node_lon - target_lon)**2)**0.5
            
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        
        return nearest_node
    
    def _find_best_insertion_point(self, route: List[int], new_node: int) -> Optional[int]:
        """Find best point to insert a new node in the route
        
        Args:
            route: Current route
            new_node: Node to insert
            
        Returns:
            Best insertion index or None
        """
        if len(route) < 2:
            return None
        
        best_insertion = None
        min_detour = float('inf')
        
        # Try inserting between each pair of consecutive nodes
        for i in range(len(route) - 1):
            current_node = route[i]
            next_node = route[i + 1]
            
            # Calculate detour cost
            original_distance = self._graph_distance(current_node, next_node)
            detour_distance = (self._graph_distance(current_node, new_node) + 
                             self._graph_distance(new_node, next_node))
            
            detour_cost = detour_distance - original_distance
            
            if detour_cost < min_detour:
                min_detour = detour_cost
                best_insertion = i + 1
        
        return best_insertion
    
    def _calculate_route_grades(self, route: List[int]) -> List[float]:
        """Calculate grades between consecutive nodes in route
        
        Args:
            route: Route as list of node IDs
            
        Returns:
            List of grades (elevation_change / distance)
        """
        grades = []
        
        for i in range(len(route) - 1):
            node1, node2 = route[i], route[i + 1]
            
            if node1 in self.graph.nodes and node2 in self.graph.nodes:
                data1 = self.graph.nodes[node1]
                data2 = self.graph.nodes[node2]
                
                elev1 = data1.get('elevation', 0)
                elev2 = data2.get('elevation', 0)
                
                distance = self._graph_distance(node1, node2)
                
                if distance > 0:
                    grade = (elev2 - elev1) / distance
                    grades.append(grade)
                else:
                    grades.append(0)
            else:
                grades.append(0)
        
        return grades
    
    def _find_grade_optimized_node(self, prev_node: int, current_node: int, 
                                 prefer_steep: bool) -> Optional[int]:
        """Find a node that optimizes grade characteristics
        
        Args:
            prev_node: Previous node in route
            current_node: Current node to replace
            prefer_steep: Whether to prefer steeper grades
            
        Returns:
            Optimized replacement node or None
        """
        if prev_node not in self.graph.nodes:
            return None
        
        prev_data = self.graph.nodes[prev_node]
        prev_elevation = prev_data.get('elevation', 0)
        
        # Find candidates
        candidates = []
        for node in self.graph.nodes():
            if node != current_node and node != prev_node:
                distance = self._graph_distance(prev_node, node)
                if distance < 300 and distance > 0:  # Reasonable distance
                    node_data = self.graph.nodes[node]
                    node_elevation = node_data.get('elevation', 0)
                    grade = abs((node_elevation - prev_elevation) / distance)
                    
                    candidates.append((node, grade))
        
        if not candidates:
            return None
        
        # Sort by grade preference
        if prefer_steep:
            candidates.sort(key=lambda x: x[1], reverse=True)  # Higher grades first
        else:
            candidates.sort(key=lambda x: x[1])  # Lower grades first
        
        # Return top candidate
        return candidates[0][0] if candidates else None
    
    def _graph_distance(self, node1: int, node2: int) -> float:
        """Calculate distance between two nodes"""
        if node1 not in self.graph.nodes or node2 not in self.graph.nodes:
            return float('inf')
        
        try:
            return nx.shortest_path_length(self.graph, node1, node2, weight='length')
        except:
            # Fallback to Euclidean
            data1 = self.graph.nodes[node1]
            data2 = self.graph.nodes[node2]
            lat1, lon1 = data1.get('y', 0), data1.get('x', 0)
            lat2, lon2 = data2.get('y', 0), data2.get('x', 0)
            return ((lat1 - lat2)**2 + (lon1 - lon2)**2)**0.5 * 111000  # Rough m conversion
    
    def _route_to_coordinates(self, route: List[int]) -> List[Tuple[float, float]]:
        """Convert route to coordinates"""
        coords = []
        for node in route:
            if node in self.graph.nodes:
                data = self.graph.nodes[node]
                coords.append((data.get('y', 0), data.get('x', 0)))
        return coords
    
    def _validate_mutated_route(self, route: List[int]) -> List[int]:
        """Validate and repair a mutated route"""
        if not route:
            return []
        
        # Remove invalid nodes
        valid_route = [node for node in route if node in self.graph.nodes]
        
        if len(valid_route) < 2:
            return route  # Return original if too damaged
        
        # Ensure route forms a loop
        if valid_route[0] != valid_route[-1]:
            valid_route.append(valid_route[0])
        
        return valid_route
    
    def _fallback_mutation(self, route: List[int]) -> List[int]:
        """Simple fallback mutation"""
        if len(route) < 3:
            return route.copy()
        
        mutated = route.copy()
        
        # Simple random replacement
        mutation_point = random.randint(1, len(route) - 2)
        all_nodes = list(self.graph.nodes())
        
        if all_nodes:
            replacement = random.choice(all_nodes)
            mutated[mutation_point] = replacement
        
        return mutated


if __name__ == "__main__":
    # Test precision-aware genetic operators
    print("🧬 Testing Precision-Aware Genetic Operators")
    print("=" * 50)
    
    # Create test graph
    import networkx as nx
    test_graph = nx.Graph()
    
    # Add test nodes with elevation data
    test_nodes = [
        (1, {'x': -80.4094, 'y': 37.1299, 'elevation': 635}),
        (2, {'x': -80.4080, 'y': 37.1310, 'elevation': 640}),
        (3, {'x': -80.4070, 'y': 37.1320, 'elevation': 650}),
        (4, {'x': -80.4060, 'y': 37.1310, 'elevation': 645}),
        (5, {'x': -80.4070, 'y': 37.1300, 'elevation': 638}),
    ]
    
    for node_id, data in test_nodes:
        test_graph.add_node(node_id, **data)
    
    # Add edges
    test_graph.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)], length=100)
    
    # Test crossover
    print("\n1. Testing Precision-Aware Crossover")
    if PRECISION_FITNESS_AVAILABLE:
        analyzer = PrecisionElevationAnalyzer()
        crossover = PrecisionAwareCrossover(test_graph, analyzer)
        
        parent1 = [1, 2, 3, 1]
        parent2 = [1, 5, 4, 1]
        
        offspring1, offspring2 = crossover.terrain_guided_crossover(parent1, parent2)
        print(f"   Parent 1: {parent1}")
        print(f"   Parent 2: {parent2}")
        print(f"   Offspring 1: {offspring1}")
        print(f"   Offspring 2: {offspring2}")
        
        # Test elevation segment crossover
        offspring3, offspring4 = crossover.elevation_segment_crossover(parent1, parent2)
        print(f"   Elevation crossover - Offspring 1: {offspring3}")
        print(f"   Elevation crossover - Offspring 2: {offspring4}")
        
    else:
        print("   ⚠️ Precision fitness not available")
    
    # Test mutation
    print("\n2. Testing Precision-Aware Mutation")
    if PRECISION_FITNESS_AVAILABLE:
        mutation = PrecisionAwareMutation(test_graph, analyzer)
        
        test_route = [1, 2, 3, 4, 1]
        
        mutated1 = mutation.elevation_guided_mutation(test_route)
        print(f"   Original: {test_route}")
        print(f"   Elevation-guided mutation: {mutated1}")
        
        mutated2 = mutation.micro_terrain_insertion_mutation(test_route)
        print(f"   Micro-terrain insertion: {mutated2}")
        
        mutated3 = mutation.grade_optimization_mutation(test_route, prefer_steep=True)
        print(f"   Grade optimization (steep): {mutated3}")
        
    else:
        print("   ⚠️ Precision fitness not available")
    
    print("\n✅ Precision-aware genetic operators testing completed")