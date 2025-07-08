#!/usr/bin/env python3
"""
Genetic Algorithm Fitness Evaluation System
Evaluates chromosome fitness based on multiple objectives
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np
import networkx as nx

from ga_chromosome import RouteChromosome
from ga_segment_cache import GASegmentCache, get_global_segment_cache


class FitnessObjective(Enum):
    """Supported fitness objectives"""
    DISTANCE = "distance"
    ELEVATION = "elevation"
    BALANCED = "balanced"
    SCENIC = "scenic"
    EFFICIENCY = "efficiency"


class GAFitnessEvaluator:
    """Fitness evaluation system for genetic algorithm route optimization"""
    
    def __init__(self, objective: str = "elevation", target_distance_km: float = 5.0, 
                 segment_cache: Optional[GASegmentCache] = None, enable_micro_terrain: bool = True,
                 allow_bidirectional_segments: bool = True):
        """Initialize fitness evaluator
        
        Args:
            objective: Primary optimization objective
            target_distance_km: Target route distance in kilometers
            segment_cache: Optional segment cache for performance optimization
            enable_micro_terrain: Whether to enable micro-terrain analysis using pre-computed elevation data
            allow_bidirectional_segments: Whether to allow segments to be used in both directions
        """
        self.objective = FitnessObjective(objective.lower())
        self.target_distance_km = target_distance_km
        self.target_distance_m = target_distance_km * 1000
        self.allow_bidirectional_segments = allow_bidirectional_segments
        
        # Segment cache for performance optimization
        self.segment_cache = segment_cache or get_global_segment_cache()
        
        # Micro-terrain analysis settings
        self.enable_micro_terrain = enable_micro_terrain
        self.interpolation_distance_m = 10.0  # Distance between interpolated points for micro-terrain analysis
        self.grade_threshold = 0.03  # 3% grade threshold for significant features
        self.elevation_gain_threshold = 2.0  # 2m minimum for meaningful elevation changes
        
        # Fitness weights for different objectives
        self.weights = self._get_objective_weights()
        
        # Performance tracking
        self.evaluations = 0
        self.best_fitness = 0.0
        self.fitness_history = []
        
    def _get_objective_weights(self) -> Dict[str, float]:
        """Get fitness weights based on objective"""
        if self.objective == FitnessObjective.DISTANCE:
            return {
                'distance_penalty': 0.6,
                'elevation_reward': 0.1,
                'connectivity_bonus': 0.2,
                'diversity_bonus': 0.1,
                'micro_terrain_bonus': 0.0
            }
        elif self.objective == FitnessObjective.ELEVATION:
            return {
                'distance_penalty': 0.15,
                'elevation_reward': 0.4,
                'connectivity_bonus': 0.2,
                'diversity_bonus': 0.1,
                'micro_terrain_bonus': 0.15
            }
        elif self.objective == FitnessObjective.BALANCED:
            return {
                'distance_penalty': 0.25,
                'elevation_reward': 0.25,
                'connectivity_bonus': 0.2,
                'diversity_bonus': 0.2,
                'micro_terrain_bonus': 0.1
            }
        elif self.objective == FitnessObjective.SCENIC:
            return {
                'distance_penalty': 0.1,
                'elevation_reward': 0.3,
                'connectivity_bonus': 0.15,
                'diversity_bonus': 0.25,
                'micro_terrain_bonus': 0.2
            }
        elif self.objective == FitnessObjective.EFFICIENCY:
            return {
                'distance_penalty': 0.5,
                'elevation_reward': 0.2,
                'connectivity_bonus': 0.2,
                'diversity_bonus': 0.1,
                'micro_terrain_bonus': 0.0
            }
        else:
            # Default to elevation
            return {
                'distance_penalty': 0.15,
                'elevation_reward': 0.4,
                'connectivity_bonus': 0.2,
                'diversity_bonus': 0.1,
                'micro_terrain_bonus': 0.15
            }
    
    def evaluate_chromosome(self, chromosome: RouteChromosome, graph: Optional[nx.Graph] = None) -> float:
        """Evaluate fitness of a single chromosome
        
        Args:
            chromosome: Chromosome to evaluate
            graph: Network graph for cached calculations (optional)
            
        Returns:
            Fitness score (0.0 - 1.0, higher is better)
        """
        # Validate chromosome with bidirectional constraint before evaluation
        if not chromosome.validate_connectivity(self.allow_bidirectional_segments) or not chromosome.segments:
            chromosome.fitness = 0.0
            return 0.0
        
        # Calculate base metrics using cached segment properties
        if graph and self.segment_cache:
            stats = self.segment_cache.get_chromosome_properties(chromosome, graph)
        else:
            # Fallback to chromosome's built-in calculation
            stats = chromosome.get_route_stats()
        
        # Distance component
        distance_score = self._calculate_distance_score(stats['total_distance_km'])
        
        # Elevation component
        elevation_score = self._calculate_elevation_score(
            stats['total_elevation_gain_m'],
            stats['max_grade_percent']
        )
        
        # Connectivity component
        connectivity_score = self._calculate_connectivity_score(chromosome)
        
        # Diversity component (geographic spread + loop efficiency)
        diversity_score = self._calculate_diversity_score(chromosome)
        
        # Micro-terrain component (using pre-computed elevation data)
        micro_terrain_score = 0.0
        if self.enable_micro_terrain and graph:
            micro_terrain_score = self._calculate_micro_terrain_score(chromosome, graph)
        
        # Apply hard distance constraint BEFORE combining other scores
        stats = chromosome.get_route_stats() if hasattr(chromosome, 'get_route_stats') else {'total_distance_km': 0}
        route_distance_km = stats.get('total_distance_km', 0)
        
        # Hard constraint: routes outside 85%-115% range get maximum 0.05 fitness
        if (route_distance_km > self.target_distance_km * 1.15 or 
            route_distance_km < self.target_distance_km * 0.85):
            # Severely limit fitness for distance violations
            max_fitness = 0.05  # Even lower than the 0.1 filtering threshold
            
            # Still calculate other components but scale them down dramatically
            other_fitness = (
                elevation_score * self.weights['elevation_reward'] +
                connectivity_score * self.weights['connectivity_bonus'] +
                diversity_score * self.weights['diversity_bonus'] +
                micro_terrain_score * self.weights['micro_terrain_bonus']
            )
            fitness = min(max_fitness, other_fitness * 0.1)  # Scale down by 90%
        else:
            # Normal fitness calculation for distance-compliant routes
            fitness = (
                distance_score * self.weights['distance_penalty'] +
                elevation_score * self.weights['elevation_reward'] +
                connectivity_score * self.weights['connectivity_bonus'] +
                diversity_score * self.weights['diversity_bonus'] +
                micro_terrain_score * self.weights['micro_terrain_bonus']
            )
        
        # Normalize to 0-1 range
        fitness = max(0.0, min(1.0, fitness))
        
        # Cache fitness value
        chromosome.fitness = fitness
        
        # Update tracking
        self.evaluations += 1
        if fitness > self.best_fitness:
            self.best_fitness = fitness
        self.fitness_history.append(fitness)
        
        return fitness
    
    def evaluate_population(self, population: List[RouteChromosome], graph: Optional[nx.Graph] = None) -> List[float]:
        """Evaluate fitness of entire population
        
        Args:
            population: Population to evaluate
            graph: Network graph for cached calculations (optional)
            
        Returns:
            List of fitness scores
        """
        fitness_scores = []
        
        for chromosome in population:
            fitness = self.evaluate_chromosome(chromosome, graph)
            fitness_scores.append(fitness)
        
        return fitness_scores
    
    def _calculate_distance_score(self, distance_km: float) -> float:
        """Calculate distance-based score component with hard constraints"""
        if distance_km <= 0:
            return 0.0
        
        # Hard constraint: severely penalize routes > 115% of target or < 85% of target
        if distance_km > self.target_distance_km * 1.15 or distance_km < self.target_distance_km * 0.85:
            return 0.01  # Nearly zero fitness for extreme deviations
        
        # Penalty for distance deviation from target (tighter constraints)
        distance_error = abs(distance_km - self.target_distance_km)
        distance_tolerance = self.target_distance_km * 0.05  # 5% tolerance (reduced from 10%)
        
        if distance_error <= distance_tolerance:
            # Within tolerance - high score
            return 1.0 - (distance_error / distance_tolerance) * 0.2
        else:
            # Outside tolerance - severe exponential penalty
            excess_ratio = (distance_error - distance_tolerance) / self.target_distance_km
            penalty = min(0.95, excess_ratio * 4.0)  # Increased penalty multiplier
            return max(0.0, 0.8 - penalty)
    
    def _calculate_elevation_score(self, elevation_gain_m: float, max_grade_percent: float) -> float:
        """Calculate elevation-based score component"""
        if elevation_gain_m <= 0:
            return 0.0
        
        # Base elevation score (logarithmic for diminishing returns)
        elevation_score = math.log(1 + elevation_gain_m / 10) / math.log(1 + 500)  # Normalized to ~500m max
        
        # Penalty for excessive grades
        grade_penalty = 0.0
        if max_grade_percent > 15:  # Steep grades
            grade_penalty = min(0.3, (max_grade_percent - 15) / 100)
        
        return max(0.0, elevation_score - grade_penalty)
    
    def _calculate_connectivity_score(self, chromosome: RouteChromosome) -> float:
        """Calculate connectivity-based score component"""
        if not chromosome.segments:
            return 0.0
        
        # Base connectivity score
        connectivity_score = 1.0 if chromosome.is_valid else 0.0
        
        # Bonus for smooth connections (fewer sharp turns)
        smooth_connections = 0
        total_connections = len(chromosome.segments) - 1
        
        if total_connections > 0:
            for i in range(total_connections):
                current_segment = chromosome.segments[i]
                next_segment = chromosome.segments[i + 1]
                
                # Check if segments connect smoothly
                if current_segment.end_node == next_segment.start_node:
                    smooth_connections += 1
            
            smoothness_ratio = smooth_connections / total_connections
            connectivity_score *= (0.5 + 0.5 * smoothness_ratio)
        
        return connectivity_score
    
    def _calculate_diversity_score(self, chromosome: RouteChromosome) -> float:
        """Calculate diversity-based score component"""
        if not chromosome.segments:
            return 0.0
        
        # Geographic diversity - spread of route across different areas
        # Measure the geographic spread of nodes visited
        visited_nodes = set()
        for segment in chromosome.segments:
            visited_nodes.add(segment.start_node)
            visited_nodes.add(segment.end_node)
        
        if len(visited_nodes) <= 1:
            geographic_diversity = 0.0
        else:
            # Calculate the geographic spread using coordinate variance
            # This rewards routes that explore different areas rather than zigzagging
            try:
                import networkx as nx
                graph = chromosome.segments[0].graph if chromosome.segments else None
                if graph:
                    latitudes = []
                    longitudes = []
                    for node in visited_nodes:
                        if node in graph.nodes:
                            node_data = graph.nodes[node]
                            latitudes.append(node_data.get('y', 0))
                            longitudes.append(node_data.get('x', 0))
                    
                    if len(latitudes) > 1:
                        lat_std = np.std(latitudes)
                        lon_std = np.std(longitudes)
                        # Normalize by typical coordinate ranges (rough approximation)
                        geographic_diversity = min(1.0, (lat_std + lon_std) * 1000)
                    else:
                        geographic_diversity = 0.0
                else:
                    geographic_diversity = 0.0
            except:
                # Fallback: use number of unique nodes as proxy for geographic spread
                geographic_diversity = min(1.0, len(visited_nodes) / 20.0)
        
        # Loop efficiency - penalty for excessive backtracking
        # Reward routes that don't repeat the same roads
        if len(chromosome.segments) > 1:
            edge_visits = {}
            for segment in chromosome.segments:
                # Count how many times we use each edge (in either direction)
                edge_key = tuple(sorted([segment.start_node, segment.end_node]))
                edge_visits[edge_key] = edge_visits.get(edge_key, 0) + 1
            
            # Calculate efficiency: penalize repeated edges
            total_segments = len(chromosome.segments)
            unique_edges = len(edge_visits)
            repeat_penalty = sum(max(0, visits - 1) for visits in edge_visits.values())
            
            if total_segments > 0:
                loop_efficiency = max(0.0, 1.0 - (repeat_penalty / total_segments))
            else:
                loop_efficiency = 1.0
        else:
            loop_efficiency = 1.0
        
        # Combine diversity components: geographic spread + loop efficiency
        diversity_score = (geographic_diversity + loop_efficiency) / 2.0
        
        return diversity_score
    
    def get_fitness_stats(self) -> Dict[str, Any]:
        """Get fitness evaluation statistics"""
        if not self.fitness_history:
            return {
                'evaluations': 0,
                'best_fitness': 0.0,
                'average_fitness': 0.0,
                'worst_fitness': 0.0,
                'fitness_std': 0.0
            }
        
        return {
            'evaluations': self.evaluations,
            'best_fitness': self.best_fitness,
            'average_fitness': np.mean(self.fitness_history),
            'worst_fitness': min(self.fitness_history),
            'fitness_std': np.std(self.fitness_history),
            'recent_improvement': self._calculate_recent_improvement()
        }
    
    def _calculate_recent_improvement(self) -> float:
        """Calculate recent fitness improvement trend"""
        if len(self.fitness_history) < 10:
            return 0.0
        
        # Compare last 10 evaluations to previous 10
        recent_avg = np.mean(self.fitness_history[-10:])
        previous_avg = np.mean(self.fitness_history[-20:-10]) if len(self.fitness_history) >= 20 else 0.0
        
        return recent_avg - previous_avg
    
    def reset_tracking(self):
        """Reset fitness tracking statistics"""
        self.evaluations = 0
        self.best_fitness = 0.0
        self.fitness_history = []
    
    def is_fitness_plateau(self, generations: int = 10, threshold: float = 0.001) -> bool:
        """Check if fitness has plateaued (convergence detection)
        
        Args:
            generations: Number of recent generations to check
            threshold: Minimum improvement threshold
            
        Returns:
            True if fitness has plateaued
        """
        if len(self.fitness_history) < generations:
            return False
        
        recent_improvement = self._calculate_recent_improvement()
        return abs(recent_improvement) < threshold
    
    def _calculate_micro_terrain_score(self, chromosome: RouteChromosome, graph: nx.Graph) -> float:
        """Calculate micro-terrain fitness using pre-computed elevation data
        
        Args:
            chromosome: Route chromosome to analyze
            graph: Network graph with pre-computed elevation data
            
        Returns:
            Micro-terrain fitness score (0.0 - 1.0)
        """
        if not chromosome.segments:
            return 0.0
        
        try:
            # Extract route elevation profile from pre-computed graph data
            elevation_profile = self._extract_elevation_profile_from_graph(chromosome, graph)
            
            # Detect micro-terrain features
            terrain_features = self._detect_micro_terrain_features(elevation_profile)
            
            # Calculate component scores
            peaks_score = self._score_micro_peaks(terrain_features)
            grade_variety_score = self._score_grade_variety(terrain_features)
            terrain_complexity_score = self._score_terrain_complexity(elevation_profile)
            
            # Combine micro-terrain components
            micro_terrain_score = (peaks_score + grade_variety_score + terrain_complexity_score) / 3.0
            
            return max(0.0, min(1.0, micro_terrain_score))
            
        except Exception as e:
            # Fallback gracefully if micro-terrain analysis fails
            return 0.0
    
    def _extract_elevation_profile_from_graph(self, chromosome: RouteChromosome, graph: nx.Graph) -> Dict[str, Any]:
        """Extract elevation profile from route using pre-computed graph data
        
        Args:
            chromosome: Route chromosome
            graph: Network graph with elevation data
            
        Returns:
            Dictionary with elevation profile data
        """
        elevations = []
        distances = []
        coordinates = []
        cumulative_distance = 0.0
        
        # Process each segment in the chromosome
        for segment in chromosome.segments:
            # Get elevation data for path nodes using pre-computed data
            for i, node_id in enumerate(segment.path_nodes):
                if node_id in graph.nodes:
                    node_data = graph.nodes[node_id]
                    elevation = node_data.get('elevation', 0.0)
                    lat = node_data.get('y', 0.0)
                    lon = node_data.get('x', 0.0)
                    
                    elevations.append(elevation)
                    coordinates.append((lat, lon))
                    distances.append(cumulative_distance)
                    
                    # Calculate distance to next node
                    if i < len(segment.path_nodes) - 1:
                        next_node_id = segment.path_nodes[i + 1]
                        if next_node_id in graph.nodes:
                            next_node_data = graph.nodes[next_node_id]
                            next_lat = next_node_data.get('y', 0.0)
                            next_lon = next_node_data.get('x', 0.0)
                            
                            # Use haversine distance
                            segment_distance = self._haversine_distance(lat, lon, next_lat, next_lon)
                            cumulative_distance += segment_distance
        
        # Interpolate for higher resolution analysis if needed
        if self.interpolation_distance_m < 50.0 and len(elevations) > 1:
            elevations, distances, coordinates = self._interpolate_elevation_profile(
                elevations, distances, coordinates
            )
        
        return {
            'elevations': elevations,
            'distances': distances,
            'coordinates': coordinates,
            'total_distance_m': distances[-1] if distances else 0.0,
            'sample_count': len(elevations)
        }
    
    def _interpolate_elevation_profile(self, elevations: List[float], distances: List[float], 
                                     coordinates: List[Tuple[float, float]]) -> Tuple[List[float], List[float], List[Tuple[float, float]]]:
        """Interpolate elevation profile for higher resolution analysis
        
        Args:
            elevations: Original elevation values
            distances: Original distance values  
            coordinates: Original coordinate values
            
        Returns:
            Tuple of (interpolated_elevations, interpolated_distances, interpolated_coordinates)
        """
        if len(elevations) < 2:
            return elevations, distances, coordinates
        
        interpolated_elevations = [elevations[0]]
        interpolated_distances = [distances[0]]
        interpolated_coordinates = [coordinates[0]]
        
        for i in range(1, len(elevations)):
            prev_elev, curr_elev = elevations[i-1], elevations[i]
            prev_dist, curr_dist = distances[i-1], distances[i]
            prev_coord, curr_coord = coordinates[i-1], coordinates[i]
            
            segment_distance = curr_dist - prev_dist
            
            # Add interpolated points if segment is longer than interpolation distance
            if segment_distance > self.interpolation_distance_m:
                num_points = int(segment_distance // self.interpolation_distance_m)
                
                for j in range(1, num_points + 1):
                    ratio = (j * self.interpolation_distance_m) / segment_distance
                    
                    # Linear interpolation
                    interp_elev = prev_elev + (curr_elev - prev_elev) * ratio
                    interp_dist = prev_dist + (curr_dist - prev_dist) * ratio
                    interp_lat = prev_coord[0] + (curr_coord[0] - prev_coord[0]) * ratio
                    interp_lon = prev_coord[1] + (curr_coord[1] - prev_coord[1]) * ratio
                    
                    interpolated_elevations.append(interp_elev)
                    interpolated_distances.append(interp_dist)
                    interpolated_coordinates.append((interp_lat, interp_lon))
            
            # Add the current point
            interpolated_elevations.append(curr_elev)
            interpolated_distances.append(curr_dist)
            interpolated_coordinates.append(curr_coord)
        
        return interpolated_elevations, interpolated_distances, interpolated_coordinates
    
    def _detect_micro_terrain_features(self, elevation_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Detect micro-terrain features from elevation profile
        
        Args:
            elevation_profile: Elevation profile data
            
        Returns:
            Dictionary with detected terrain features
        """
        elevations = np.array(elevation_profile['elevations'])
        distances = np.array(elevation_profile['distances'])
        
        if len(elevations) < 3:
            return {'peaks': [], 'valleys': [], 'steep_sections': [], 'grade_changes': []}
        
        # Calculate grades between points
        grades = []
        for i in range(1, len(elevations)):
            elevation_change = elevations[i] - elevations[i-1]
            distance_change = distances[i] - distances[i-1]
            if distance_change > 0:
                grade = elevation_change / distance_change
                grades.append(grade)
            else:
                grades.append(0)
        
        grades = np.array(grades)
        
        # Detect peaks (local maxima)
        peaks = []
        for i in range(1, len(elevations) - 1):
            if (elevations[i] > elevations[i-1] and 
                elevations[i] > elevations[i+1] and
                elevations[i] - min(elevations[i-1], elevations[i+1]) >= self.elevation_gain_threshold):
                peaks.append({
                    'index': i,
                    'elevation_m': float(elevations[i]),
                    'distance_m': float(distances[i]),
                    'prominence_m': float(elevations[i] - min(elevations[i-1], elevations[i+1]))
                })
        
        # Detect valleys (local minima)
        valleys = []
        for i in range(1, len(elevations) - 1):
            if (elevations[i] < elevations[i-1] and 
                elevations[i] < elevations[i+1] and
                max(elevations[i-1], elevations[i+1]) - elevations[i] >= self.elevation_gain_threshold):
                valleys.append({
                    'index': i,
                    'elevation_m': float(elevations[i]),
                    'distance_m': float(distances[i]),
                    'depth_m': float(max(elevations[i-1], elevations[i+1]) - elevations[i])
                })
        
        # Detect steep sections
        steep_sections = []
        for i, grade in enumerate(grades):
            if abs(grade) >= self.grade_threshold:
                steep_sections.append({
                    'start_index': i,
                    'end_index': i + 1,
                    'grade_percent': float(grade * 100),
                    'distance_m': float(distances[i + 1] - distances[i]),
                    'elevation_change_m': float(elevations[i + 1] - elevations[i])
                })
        
        # Detect significant grade changes
        grade_changes = []
        if len(grades) > 1:
            for i in range(1, len(grades)):
                grade_change = abs(grades[i] - grades[i-1])
                if grade_change >= self.grade_threshold:
                    grade_changes.append({
                        'index': i,
                        'grade_change_percent': float(grade_change * 100),
                        'previous_grade_percent': float(grades[i-1] * 100),
                        'current_grade_percent': float(grades[i] * 100)
                    })
        
        return {
            'peaks': peaks,
            'valleys': valleys,
            'steep_sections': steep_sections,
            'grade_changes': grade_changes,
            'max_grade_percent': float(np.max(np.abs(grades)) * 100) if len(grades) > 0 else 0,
            'avg_grade_percent': float(np.mean(np.abs(grades)) * 100) if len(grades) > 0 else 0,
            'total_elevation_gain_m': float(np.sum(grades[grades > 0]) * np.mean(np.diff(distances))) if len(grades) > 0 else 0,
            'total_elevation_loss_m': float(np.sum(np.abs(grades[grades < 0])) * np.mean(np.diff(distances))) if len(grades) > 0 else 0
        }
    
    def _score_micro_peaks(self, terrain_features: Dict[str, Any]) -> float:
        """Score micro-terrain peaks for fitness evaluation
        
        Args:
            terrain_features: Detected terrain features
            
        Returns:
            Peaks score (0.0 - 1.0)
        """
        peaks = terrain_features.get('peaks', [])
        
        if not peaks:
            return 0.0
        
        # Score based on number and prominence of peaks
        peak_score = 0.0
        for peak in peaks:
            prominence = peak.get('prominence_m', 0)
            # Reward significant peaks (>2m prominence)
            if prominence >= self.elevation_gain_threshold:
                peak_score += min(prominence / 10.0, 1.0)  # Max 1.0 per peak
        
        # Normalize by expected maximum (5 significant peaks for a good route)
        return min(peak_score / 5.0, 1.0)
    
    def _score_grade_variety(self, terrain_features: Dict[str, Any]) -> float:
        """Score grade variety for fitness evaluation
        
        Args:
            terrain_features: Detected terrain features
            
        Returns:
            Grade variety score (0.0 - 1.0)
        """
        steep_sections = terrain_features.get('steep_sections', [])
        grade_changes = terrain_features.get('grade_changes', [])
        
        # Score based on variety of grades
        grade_variety_score = 0.0
        
        # Reward steep sections (challenging terrain)
        for section in steep_sections:
            grade_percent = abs(section.get('grade_percent', 0))
            if 3 <= grade_percent <= 15:  # Sweet spot for running
                grade_variety_score += min(grade_percent / 15.0, 1.0)
        
        # Reward grade changes (varied terrain)
        significant_changes = [gc for gc in grade_changes 
                             if gc.get('grade_change_percent', 0) >= 3.0]
        change_score = min(len(significant_changes) / 10.0, 1.0)
        
        return (grade_variety_score + change_score) / 2.0
    
    def _score_terrain_complexity(self, elevation_profile: Dict[str, Any]) -> float:
        """Score terrain complexity for fitness evaluation
        
        Args:
            elevation_profile: Elevation profile data
            
        Returns:
            Terrain complexity score (0.0 - 1.0)
        """
        elevations = elevation_profile.get('elevations', [])
        
        if len(elevations) < 2:
            return 0.0
        
        elevations = np.array(elevations)
        
        # Calculate terrain complexity indicators
        elevation_variability = float(np.std(elevations))
        elevation_range = float(np.max(elevations) - np.min(elevations))
        
        # Calculate gradient variance
        if len(elevations) > 1:
            elevation_gradients = np.diff(elevations)
            gradient_variance = float(np.var(elevation_gradients))
        else:
            gradient_variance = 0
        
        # Normalize complexity metrics
        variability_score = min(elevation_variability / 20.0, 1.0)  # 20m std = complex
        range_score = min(elevation_range / 100.0, 1.0)  # 100m range = complex
        gradient_score = min(gradient_variance / 5.0, 1.0)  # Normalized gradient variance
        
        return (variability_score + range_score + gradient_score) / 3.0
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in meters"""
        R = 6371000  # Earth's radius in meters
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat / 2) ** 2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c


def test_fitness_evaluator():
    """Test function for fitness evaluator"""
    print("Testing GA Fitness Evaluator...")
    
    # Create test chromosome
    from ga_chromosome import RouteSegment, RouteChromosome
    
    segment1 = RouteSegment(1, 2, [1, 2])
    segment1.length = 1000.0
    segment1.elevation_gain = 50.0
    segment1.direction = "N"
    
    segment2 = RouteSegment(2, 3, [2, 3])
    segment2.length = 1500.0
    segment2.elevation_gain = 30.0
    segment2.direction = "E"
    
    chromosome = RouteChromosome([segment1, segment2])
    chromosome.is_valid = True
    
    # Test different objectives
    objectives = ["distance", "elevation", "balanced"]
    
    for objective in objectives:
        evaluator = GAFitnessEvaluator(objective, target_distance_km=2.5)
        fitness = evaluator.evaluate_chromosome(chromosome)
        print(f"✅ {objective.title()} objective fitness: {fitness:.3f}")
    
    # Test population evaluation
    population = [chromosome] * 5
    evaluator = GAFitnessEvaluator("elevation", 2.5)
    fitness_scores = evaluator.evaluate_population(population)
    print(f"✅ Population evaluation: {len(fitness_scores)} scores")
    
    # Test fitness statistics
    stats = evaluator.get_fitness_stats()
    print(f"✅ Fitness stats: {stats['evaluations']} evaluations, best: {stats['best_fitness']:.3f}")
    
    # Test micro-terrain functionality
    print("\n🏔️ Testing micro-terrain analysis...")
    import networkx as nx
    
    # Create test graph with elevation data
    test_graph = nx.Graph()
    test_graph.add_node(1, x=-80.4094, y=37.1299, elevation=635.0)
    test_graph.add_node(2, x=-80.4080, y=37.1310, elevation=640.0)
    test_graph.add_node(3, x=-80.4090, y=37.1320, elevation=645.0)
    
    # Update segment path_nodes for testing
    segment1.path_nodes = [1, 2]
    segment2.path_nodes = [2, 3]
    
    # Test micro-terrain evaluation
    micro_evaluator = GAFitnessEvaluator("elevation", 2.5, enable_micro_terrain=True)
    micro_fitness = micro_evaluator.evaluate_chromosome(chromosome, test_graph)
    print(f"✅ Micro-terrain fitness: {micro_fitness:.3f}")
    
    # Test without micro-terrain
    basic_evaluator = GAFitnessEvaluator("elevation", 2.5, enable_micro_terrain=False)
    basic_fitness = basic_evaluator.evaluate_chromosome(chromosome, test_graph)
    print(f"✅ Basic fitness: {basic_fitness:.3f}")
    
    improvement = micro_fitness - basic_fitness
    print(f"✅ Micro-terrain improvement: {improvement:.3f}")
    
    print("✅ All fitness evaluator tests completed")


if __name__ == "__main__":
    test_fitness_evaluator()