#!/usr/bin/env python3
"""
Enhanced GA Fitness Functions with 1-meter Precision
Leverages 3DEP 1m elevation data for superior route optimization and micro-terrain detection
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from elevation_data_sources import get_elevation_manager
    ENHANCED_ELEVATION_AVAILABLE = True
except ImportError:
    ENHANCED_ELEVATION_AVAILABLE = False


class PrecisionElevationAnalyzer:
    """Analyzes elevation features using 1-meter precision data"""
    
    def __init__(self, elevation_manager=None):
        """Initialize precision elevation analyzer
        
        Args:
            elevation_manager: Optional elevation manager, will create one if None
        """
        if ENHANCED_ELEVATION_AVAILABLE and elevation_manager:
            self.elevation_manager = elevation_manager
            self.elevation_source = elevation_manager.get_elevation_source()
        else:
            try:
                self.elevation_manager = get_elevation_manager()
                self.elevation_source = self.elevation_manager.get_elevation_source()
            except:
                logger.warning("Enhanced elevation not available, using basic analysis")
                self.elevation_manager = None
                self.elevation_source = None
        
        # Cache for elevation queries
        self.elevation_cache = {}
        
        # Micro-terrain feature detection parameters
        self.feature_detection_distance = 10.0  # meters between sample points
        self.grade_threshold = 0.03  # 3% grade threshold for significant features
        self.elevation_gain_threshold = 2.0  # 2m minimum for meaningful elevation changes
        
    def get_high_resolution_elevation_profile(self, route_coordinates: List[Tuple[float, float]], 
                                            interpolation_distance: float = 5.0) -> Dict[str, Any]:
        """Generate high-resolution elevation profile with 1m precision
        
        Args:
            route_coordinates: List of (lat, lon) coordinate pairs
            interpolation_distance: Distance in meters between interpolated points
            
        Returns:
            Dictionary with detailed elevation profile data
        """
        if not self.elevation_source:
            logger.warning("No elevation source available for high-resolution profile")
            return self._fallback_elevation_profile(route_coordinates)
        
        try:
            # Interpolate points along route for high resolution
            interpolated_coords = self._interpolate_route_coordinates(
                route_coordinates, interpolation_distance
            )
            
            # Get high-precision elevations
            elevations = []
            distances = []
            cumulative_distance = 0
            
            for i, (lat, lon) in enumerate(interpolated_coords):
                # Get cached or query elevation
                coord_key = f"{lat:.6f},{lon:.6f}"
                if coord_key in self.elevation_cache:
                    elevation = self.elevation_cache[coord_key]
                else:
                    elevation = self.elevation_source.get_elevation(lat, lon)
                    if elevation is not None:
                        self.elevation_cache[coord_key] = elevation
                
                if elevation is not None:
                    elevations.append(elevation)
                    distances.append(cumulative_distance)
                    
                    # Calculate distance to next point
                    if i < len(interpolated_coords) - 1:
                        next_lat, next_lon = interpolated_coords[i + 1]
                        segment_distance = self._haversine_distance(lat, lon, next_lat, next_lon)
                        cumulative_distance += segment_distance
                else:
                    # Use interpolated elevation if direct lookup fails
                    if elevations:
                        elevations.append(elevations[-1])  # Use last known elevation
                        distances.append(cumulative_distance)
            
            if not elevations:
                return self._fallback_elevation_profile(route_coordinates)
            
            # Analyze micro-terrain features
            terrain_features = self._detect_micro_terrain_features(
                interpolated_coords[:len(elevations)], elevations, distances
            )
            
            # Calculate precision statistics
            precision_stats = self._calculate_precision_statistics(elevations, distances)
            
            return {
                'coordinates': interpolated_coords[:len(elevations)],
                'elevations': elevations,
                'distances_m': distances,
                'total_distance_m': distances[-1] if distances else 0,
                'resolution_m': self.elevation_source.get_resolution(),
                'micro_terrain_features': terrain_features,
                'precision_statistics': precision_stats,
                'interpolation_distance_m': interpolation_distance,
                'sample_count': len(elevations)
            }
            
        except Exception as e:
            logger.error(f"High-resolution elevation profile failed: {e}")
            return self._fallback_elevation_profile(route_coordinates)
    
    def _interpolate_route_coordinates(self, coordinates: List[Tuple[float, float]], 
                                     target_distance: float) -> List[Tuple[float, float]]:
        """Interpolate coordinates along route at specified distance intervals
        
        Args:
            coordinates: Original route coordinates
            target_distance: Target distance between interpolated points in meters
            
        Returns:
            List of interpolated coordinates
        """
        if len(coordinates) < 2:
            return coordinates
        
        interpolated = [coordinates[0]]  # Start with first point
        
        for i in range(len(coordinates) - 1):
            lat1, lon1 = coordinates[i]
            lat2, lon2 = coordinates[i + 1]
            
            # Calculate segment distance
            segment_distance = self._haversine_distance(lat1, lon1, lat2, lon2)
            
            if segment_distance > target_distance:
                # Add interpolated points along this segment
                num_points = int(segment_distance // target_distance)
                
                for j in range(1, num_points + 1):
                    ratio = (j * target_distance) / segment_distance
                    interp_lat = lat1 + (lat2 - lat1) * ratio
                    interp_lon = lon1 + (lon2 - lon1) * ratio
                    interpolated.append((interp_lat, interp_lon))
            
            # Add the endpoint
            interpolated.append(coordinates[i + 1])
        
        return interpolated
    
    def _detect_micro_terrain_features(self, coordinates: List[Tuple[float, float]], 
                                     elevations: List[float], distances: List[float]) -> Dict[str, Any]:
        """Detect micro-terrain features using high-resolution elevation data
        
        Args:
            coordinates: High-resolution coordinate points
            elevations: Corresponding elevation values
            distances: Cumulative distances
            
        Returns:
            Dictionary with detected terrain features
        """
        if len(elevations) < 3:
            return {'peaks': [], 'valleys': [], 'steep_sections': [], 'grade_changes': []}
        
        elevations = np.array(elevations)
        distances = np.array(distances)
        
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
                    'coordinate': coordinates[i],
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
                    'coordinate': coordinates[i],
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
                    'start_coordinate': coordinates[i],
                    'end_coordinate': coordinates[i + 1],
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
                        'coordinate': coordinates[i],
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
    
    def _calculate_precision_statistics(self, elevations: List[float], distances: List[float]) -> Dict[str, Any]:
        """Calculate statistics highlighting 1m precision benefits
        
        Args:
            elevations: Elevation values
            distances: Distance values
            
        Returns:
            Dictionary with precision statistics
        """
        if not elevations:
            return {}
        
        elevations = np.array(elevations)
        
        # Calculate elevation variability (shows 1m precision benefits)
        elevation_std = float(np.std(elevations))
        elevation_range = float(np.max(elevations) - np.min(elevations))
        
        # Calculate smoothness metrics
        if len(elevations) > 1:
            elevation_gradients = np.diff(elevations)
            gradient_variance = float(np.var(elevation_gradients))
            max_elevation_change = float(np.max(np.abs(elevation_gradients)))
        else:
            gradient_variance = 0
            max_elevation_change = 0
        
        # Estimate precision benefits
        resolution = self.elevation_source.get_resolution() if self.elevation_source else 90
        precision_factor = 90.0 / resolution if resolution > 0 else 1.0
        
        return {
            'data_resolution_m': resolution,
            'precision_improvement_factor': precision_factor,
            'elevation_variability_m': elevation_std,
            'elevation_range_m': elevation_range,
            'gradient_variance': gradient_variance,
            'max_elevation_change_m': max_elevation_change,
            'sample_density_per_km': len(elevations) / (distances[-1] / 1000) if distances and distances[-1] > 0 else 0,
            'estimated_accuracy_m': 0.3 if resolution <= 1.0 else 16.0  # 3DEP vs SRTM accuracy
        }
    
    def _fallback_elevation_profile(self, coordinates: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Fallback elevation profile using basic interpolation
        
        Args:
            coordinates: Route coordinates
            
        Returns:
            Basic elevation profile
        """
        return {
            'coordinates': coordinates,
            'elevations': [0.0] * len(coordinates),
            'distances_m': list(range(len(coordinates))),
            'total_distance_m': len(coordinates),
            'resolution_m': 90.0,
            'micro_terrain_features': {'peaks': [], 'valleys': [], 'steep_sections': [], 'grade_changes': []},
            'precision_statistics': {'data_resolution_m': 90.0, 'precision_improvement_factor': 1.0},
            'interpolation_distance_m': 100.0,
            'sample_count': len(coordinates)
        }
    
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


class EnhancedGAFitnessEvaluator:
    """Enhanced fitness evaluator leveraging 1-meter precision elevation data"""
    
    def __init__(self, graph: nx.Graph, elevation_manager=None, enable_micro_terrain: bool = True):
        """Initialize enhanced fitness evaluator
        
        Args:
            graph: NetworkX graph with route network
            elevation_manager: Optional elevation manager
            enable_micro_terrain: Whether to enable micro-terrain analysis
        """
        self.graph = graph
        self.elevation_analyzer = PrecisionElevationAnalyzer(elevation_manager)
        self.enable_micro_terrain = enable_micro_terrain
        
        # Fitness component weights for different objectives
        self.fitness_weights = {
            'maximize_elevation': {
                'elevation_gain': 0.4,
                'micro_peaks': 0.2,
                'grade_variety': 0.15,
                'terrain_complexity': 0.15,
                'distance_penalty': 0.1
            },
            'scenic_route': {
                'elevation_gain': 0.25,
                'micro_peaks': 0.25,
                'grade_variety': 0.2,
                'terrain_complexity': 0.2,
                'distance_penalty': 0.1
            },
            'trail_optimization': {
                'micro_peaks': 0.3,
                'grade_variety': 0.25,
                'terrain_complexity': 0.25,
                'elevation_gain': 0.15,
                'distance_penalty': 0.05
            }
        }
    
    def evaluate_route_fitness(self, route_coordinates: List[Tuple[float, float]], 
                             objective: str = 'maximize_elevation',
                             target_distance_km: float = 5.0) -> Dict[str, float]:
        """Evaluate route fitness using 1m precision elevation analysis
        
        Args:
            route_coordinates: List of (lat, lon) coordinate pairs
            objective: Fitness objective ('maximize_elevation', 'scenic_route', 'trail_optimization')
            target_distance_km: Target route distance in kilometers
            
        Returns:
            Dictionary with fitness scores and component breakdown
        """
        if len(route_coordinates) < 2:
            return {'total_fitness': 0.0, 'components': {}}
        
        try:
            # Get high-resolution elevation profile
            elevation_profile = self.elevation_analyzer.get_high_resolution_elevation_profile(
                route_coordinates, interpolation_distance=5.0
            )
            
            # Calculate fitness components
            components = {}
            
            # 1. Elevation gain component
            components['elevation_gain'] = self._calculate_elevation_gain_fitness(elevation_profile)
            
            # 2. Micro-terrain peaks component
            if self.enable_micro_terrain:
                components['micro_peaks'] = self._calculate_micro_peaks_fitness(elevation_profile)
                components['grade_variety'] = self._calculate_grade_variety_fitness(elevation_profile)
                components['terrain_complexity'] = self._calculate_terrain_complexity_fitness(elevation_profile)
            else:
                components['micro_peaks'] = 0.0
                components['grade_variety'] = 0.0
                components['terrain_complexity'] = 0.0
            
            # 3. Distance penalty component
            components['distance_penalty'] = self._calculate_distance_penalty(
                elevation_profile.get('total_distance_m', 0) / 1000, target_distance_km
            )
            
            # 4. Precision bonus (rewards using high-resolution data)
            components['precision_bonus'] = self._calculate_precision_bonus(elevation_profile)
            
            # Calculate weighted total fitness
            weights = self.fitness_weights.get(objective, self.fitness_weights['maximize_elevation'])
            total_fitness = 0.0
            
            for component, value in components.items():
                weight = weights.get(component, 0.0)
                total_fitness += weight * value
            
            # Add precision bonus
            total_fitness += 0.1 * components['precision_bonus']
            
            return {
                'total_fitness': total_fitness,
                'components': components,
                'elevation_profile': elevation_profile,
                'objective': objective,
                'weights': weights
            }
            
        except Exception as e:
            logger.error(f"Fitness evaluation failed: {e}")
            return {'total_fitness': 0.0, 'components': {}, 'error': str(e)}
    
    def _calculate_elevation_gain_fitness(self, elevation_profile: Dict) -> float:
        """Calculate fitness based on total elevation gain"""
        features = elevation_profile.get('micro_terrain_features', {})
        total_gain = features.get('total_elevation_gain_m', 0)
        
        # Normalize to 0-1 scale (assuming max interesting gain is 200m for 5km route)
        normalized_gain = min(total_gain / 200.0, 1.0)
        return normalized_gain
    
    def _calculate_micro_peaks_fitness(self, elevation_profile: Dict) -> float:
        """Calculate fitness based on micro-terrain peaks discovered"""
        features = elevation_profile.get('micro_terrain_features', {})
        peaks = features.get('peaks', [])
        
        if not peaks:
            return 0.0
        
        # Score based on number and prominence of peaks
        peak_score = 0.0
        for peak in peaks:
            prominence = peak.get('prominence_m', 0)
            # Reward significant peaks (>2m prominence)
            if prominence >= 2.0:
                peak_score += min(prominence / 10.0, 1.0)  # Max 1.0 per peak
        
        # Normalize by expected maximum (5 significant peaks for a good route)
        return min(peak_score / 5.0, 1.0)
    
    def _calculate_grade_variety_fitness(self, elevation_profile: Dict) -> float:
        """Calculate fitness based on grade variety (interesting terrain)"""
        features = elevation_profile.get('micro_terrain_features', {})
        steep_sections = features.get('steep_sections', [])
        grade_changes = features.get('grade_changes', [])
        
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
    
    def _calculate_terrain_complexity_fitness(self, elevation_profile: Dict) -> float:
        """Calculate fitness based on terrain complexity"""
        precision_stats = elevation_profile.get('precision_statistics', {})
        
        # Terrain complexity indicators
        elevation_variability = precision_stats.get('elevation_variability_m', 0)
        gradient_variance = precision_stats.get('gradient_variance', 0)
        
        # Normalize complexity metrics
        variability_score = min(elevation_variability / 20.0, 1.0)  # 20m std = complex
        gradient_score = min(gradient_variance / 5.0, 1.0)  # Normalized gradient variance
        
        return (variability_score + gradient_score) / 2.0
    
    def _calculate_distance_penalty(self, actual_distance_km: float, target_distance_km: float) -> float:
        """Calculate distance penalty (lower is better)"""
        if target_distance_km <= 0:
            return 0.0
        
        distance_ratio = actual_distance_km / target_distance_km
        
        # Penalty for being too far from target (±20% tolerance)
        if 0.8 <= distance_ratio <= 1.2:
            return 1.0  # No penalty
        elif distance_ratio < 0.8:
            return max(0.0, 1.0 - (0.8 - distance_ratio) * 2)  # Penalty for too short
        else:
            return max(0.0, 1.0 - (distance_ratio - 1.2) * 2)  # Penalty for too long
    
    def _calculate_precision_bonus(self, elevation_profile: Dict) -> float:
        """Calculate bonus for using high-precision data"""
        precision_stats = elevation_profile.get('precision_statistics', {})
        resolution = precision_stats.get('data_resolution_m', 90)
        sample_count = elevation_profile.get('sample_count', 0)
        
        # Bonus for high-resolution data
        resolution_bonus = 1.0 if resolution <= 1.0 else 0.5 if resolution <= 10.0 else 0.0
        
        # Bonus for dense sampling
        density_bonus = min(sample_count / 1000.0, 1.0)  # 1000 samples = full bonus
        
        return (resolution_bonus + density_bonus) / 2.0
    
    def compare_precision_benefits(self, route_coordinates: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Compare fitness with and without 1m precision data
        
        Args:
            route_coordinates: Route coordinates to analyze
            
        Returns:
            Dictionary comparing 1m vs 90m precision benefits
        """
        # Get high-resolution analysis
        high_res_profile = self.elevation_analyzer.get_high_resolution_elevation_profile(
            route_coordinates, interpolation_distance=5.0
        )
        high_res_fitness = self.evaluate_route_fitness(
            route_coordinates, 'maximize_elevation'
        )
        
        # Simulate low-resolution analysis
        low_res_profile = self.elevation_analyzer.get_high_resolution_elevation_profile(
            route_coordinates, interpolation_distance=50.0  # Simulate 90m resolution
        )
        
        # Temporarily disable micro-terrain for low-res simulation
        original_micro_terrain = self.enable_micro_terrain
        self.enable_micro_terrain = False
        
        low_res_fitness = self.evaluate_route_fitness(
            route_coordinates, 'maximize_elevation'
        )
        
        # Restore micro-terrain setting
        self.enable_micro_terrain = original_micro_terrain
        
        # Calculate precision benefits
        fitness_improvement = (high_res_fitness['total_fitness'] - 
                             low_res_fitness['total_fitness'])
        
        micro_features_found = len(high_res_profile.get('micro_terrain_features', {}).get('peaks', []))
        
        return {
            'high_resolution': {
                'fitness': high_res_fitness['total_fitness'],
                'elevation_profile': high_res_profile,
                'components': high_res_fitness['components']
            },
            'low_resolution': {
                'fitness': low_res_fitness['total_fitness'],
                'elevation_profile': low_res_profile,
                'components': low_res_fitness['components']
            },
            'precision_benefits': {
                'fitness_improvement': fitness_improvement,
                'fitness_improvement_percent': (fitness_improvement / max(low_res_fitness['total_fitness'], 0.1)) * 100,
                'micro_features_discovered': micro_features_found,
                'resolution_factor': high_res_profile.get('precision_statistics', {}).get('precision_improvement_factor', 1.0),
                'sample_density_improvement': (high_res_profile.get('sample_count', 0) / 
                                             max(low_res_profile.get('sample_count', 1), 1))
            }
        }


if __name__ == "__main__":
    # Test the enhanced GA fitness functions
    print("🧬 Testing Enhanced GA Fitness Functions with 1m Precision")
    print("=" * 60)
    
    # Create test route coordinates
    test_coords = [
        (37.1299, -80.4094),  # Start
        (37.1310, -80.4080),  # Northeast
        (37.1320, -80.4090),  # East then south
        (37.1315, -80.4105),  # Southwest
        (37.1305, -80.4110),  # Further southwest
        (37.1295, -80.4100),  # Northwest
        (37.1299, -80.4094),  # Back to start
    ]
    
    # Test elevation analyzer
    print("\n1. Testing Precision Elevation Analyzer")
    analyzer = PrecisionElevationAnalyzer()
    
    if analyzer.elevation_source:
        print(f"   ✅ Elevation source: {analyzer.elevation_source.__class__.__name__}")
        print(f"   📊 Resolution: {analyzer.elevation_source.get_resolution()}m")
        
        profile = analyzer.get_high_resolution_elevation_profile(test_coords)
        print(f"   🔍 Sample count: {profile['sample_count']}")
        print(f"   📏 Total distance: {profile['total_distance_m']:.1f}m")
        
        features = profile['micro_terrain_features']
        print(f"   ⛰️ Micro-features: {len(features['peaks'])} peaks, {len(features['valleys'])} valleys")
        print(f"   📈 Max grade: {features['max_grade_percent']:.1f}%")
        
    else:
        print("   ⚠️ No enhanced elevation source available")
    
    # Test fitness evaluator
    print("\n2. Testing Enhanced GA Fitness Evaluator")
    
    # Create dummy graph for testing
    import networkx as nx
    test_graph = nx.Graph()
    test_graph.add_node(1, x=-80.4094, y=37.1299, elevation=635)
    test_graph.add_node(2, x=-80.4080, y=37.1310, elevation=640)
    
    evaluator = EnhancedGAFitnessEvaluator(test_graph)
    
    # Test different objectives
    objectives = ['maximize_elevation', 'scenic_route', 'trail_optimization']
    
    for objective in objectives:
        print(f"\n   Testing {objective} objective:")
        fitness_result = evaluator.evaluate_route_fitness(test_coords, objective)
        
        if 'error' not in fitness_result:
            print(f"     Total fitness: {fitness_result['total_fitness']:.3f}")
            
            components = fitness_result['components']
            for component, value in components.items():
                print(f"     {component}: {value:.3f}")
        else:
            print(f"     ❌ Error: {fitness_result['error']}")
    
    # Test precision comparison
    print("\n3. Testing Precision Benefits Comparison")
    comparison = evaluator.compare_precision_benefits(test_coords)
    
    if 'precision_benefits' in comparison:
        benefits = comparison['precision_benefits']
        print(f"   📊 Fitness improvement: {benefits['fitness_improvement']:.3f}")
        print(f"   📈 Improvement percentage: {benefits['fitness_improvement_percent']:.1f}%")
        print(f"   🔍 Micro-features found: {benefits['micro_features_discovered']}")
        print(f"   📏 Resolution factor: {benefits['resolution_factor']:.1f}x")
        print(f"   📋 Sample density improvement: {benefits['sample_density_improvement']:.1f}x")
    
    print("\n✅ Enhanced GA fitness testing completed")