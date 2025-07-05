#!/usr/bin/env python3
"""
Enhanced GA Fitness Function System
Advanced fitness evaluation with multi-objective optimization and adaptive components
"""

import math
import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from abc import ABC, abstractmethod

from tsp_solver_fast import RouteObjective
from ga_chromosome import RouteChromosome


class FitnessComponent(Enum):
    """Available fitness components"""
    DISTANCE_ACCURACY = "distance_accuracy"
    ELEVATION_GAIN = "elevation_gain"
    ROUTE_DIVERSITY = "route_diversity"
    CONNECTIVITY_QUALITY = "connectivity_quality"
    GRADIENT_SMOOTHNESS = "gradient_smoothness"
    TURN_EFFICIENCY = "turn_efficiency"
    SCENIC_VALUE = "scenic_value"
    SAFETY_SCORE = "safety_score"
    COMPLEXITY_PENALTY = "complexity_penalty"
    NOVELTY_BONUS = "novelty_bonus"


class AggregationMethod(Enum):
    """Methods for aggregating multiple objectives"""
    WEIGHTED_SUM = "weighted_sum"
    PRODUCT = "product"
    TCHEBYCHEFF = "tchebycheff"
    ACHIEVEMENT_FUNCTION = "achievement_function"
    PARETO_RANKING = "pareto_ranking"
    NSGA2 = "nsga2"
    LEXICOGRAPHIC = "lexicographic"


@dataclass
class FitnessWeights:
    """Weights for different fitness components"""
    distance_accuracy: float = 0.4
    elevation_gain: float = 0.3
    route_diversity: float = 0.1
    connectivity_quality: float = 0.1
    gradient_smoothness: float = 0.05
    turn_efficiency: float = 0.03
    scenic_value: float = 0.02
    safety_score: float = 0.0
    complexity_penalty: float = 0.0
    novelty_bonus: float = 0.0
    
    def normalize(self):
        """Normalize weights to sum to 1.0"""
        total = sum(vars(self).values())
        if total > 0:
            for key in vars(self):
                setattr(self, key, getattr(self, key) / total)


@dataclass
class FitnessProfile:
    """Fitness evaluation profile for different scenarios"""
    name: str
    objective: RouteObjective
    weights: FitnessWeights
    aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_SUM
    adaptive_weights: bool = False
    reference_point: Optional[Dict[str, float]] = None
    constraints: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    preferences: Dict[str, float] = field(default_factory=dict)


@dataclass
class MultiObjectiveResult:
    """Result of multi-objective fitness evaluation"""
    chromosome: RouteChromosome
    objective_values: Dict[str, float]
    aggregated_fitness: float
    rank: int = 0
    crowding_distance: float = 0.0
    dominates: List[int] = field(default_factory=list)
    dominated_by: int = 0
    pareto_front: int = 0


class FitnessComponentEvaluator(ABC):
    """Abstract base class for fitness component evaluators"""
    
    @abstractmethod
    def evaluate(self, chromosome: RouteChromosome, context: Dict[str, Any]) -> float:
        """Evaluate fitness component for chromosome"""
        pass
    
    @abstractmethod
    def get_ideal_value(self) -> float:
        """Get ideal value for this component (for normalization)"""
        pass
    
    @abstractmethod
    def get_nadir_value(self) -> float:
        """Get worst value for this component (for normalization)"""
        pass


class DistanceAccuracyEvaluator(FitnessComponentEvaluator):
    """Evaluates how well route matches target distance"""
    
    def evaluate(self, chromosome: RouteChromosome, context: Dict[str, Any]) -> float:
        target_distance = context.get('target_distance_km', 5.0)
        tolerance = context.get('distance_tolerance', 0.2)
        
        if not chromosome.segments:
            return 0.0
        
        total_distance = sum(segment.length for segment in chromosome.segments) / 1000.0
        distance_error = abs(total_distance - target_distance) / target_distance
        
        if distance_error <= tolerance:
            return 1.0 - (distance_error / tolerance) * 0.1
        else:
            return max(0.0, 1.0 - distance_error)
    
    def get_ideal_value(self) -> float:
        return 1.0
    
    def get_nadir_value(self) -> float:
        return 0.0


class ElevationGainEvaluator(FitnessComponentEvaluator):
    """Evaluates elevation gain characteristics"""
    
    def evaluate(self, chromosome: RouteChromosome, context: Dict[str, Any]) -> float:
        objective = context.get('objective', RouteObjective.BALANCED_ROUTE)
        
        if not chromosome.segments:
            return 0.0
        
        total_elevation_gain = sum(max(0, segment.elevation_gain) 
                                 for segment in chromosome.segments)
        total_distance = sum(segment.length for segment in chromosome.segments)
        
        if total_distance == 0:
            return 0.0
        
        # Elevation gain per kilometer
        elevation_rate = total_elevation_gain / (total_distance / 1000.0)
        
        if objective == RouteObjective.MAXIMIZE_ELEVATION:
            # Favor higher elevation gain, with diminishing returns
            normalized_rate = min(1.0, elevation_rate / 100.0)  # 100m/km as reference
            return math.sqrt(normalized_rate)
        
        elif objective == RouteObjective.MINIMIZE_DISTANCE:
            # Penalize excessive elevation gain
            penalty = min(1.0, elevation_rate / 50.0)
            return 1.0 - penalty * 0.5
        
        else:  # BALANCED_ROUTE
            # Moderate elevation gain is preferred
            optimal_rate = 30.0  # 30m/km
            deviation = abs(elevation_rate - optimal_rate) / optimal_rate
            return max(0.0, 1.0 - deviation)
    
    def get_ideal_value(self) -> float:
        return 1.0
    
    def get_nadir_value(self) -> float:
        return 0.0


class RouteDiversityEvaluator(FitnessComponentEvaluator):
    """Evaluates route diversity and exploration"""
    
    def __init__(self):
        self.previous_routes = []
        self.max_history = 100
    
    def evaluate(self, chromosome: RouteChromosome, context: Dict[str, Any]) -> float:
        if not chromosome.segments:
            return 0.0
        
        # Extract route nodes
        route_nodes = set()
        for segment in chromosome.segments:
            route_nodes.update(segment.path_nodes)
        
        # Calculate diversity score based on novel nodes
        if not self.previous_routes:
            diversity = 1.0
        else:
            # Compare with recent routes
            novel_nodes = route_nodes.copy()
            for prev_route in self.previous_routes[-10:]:  # Last 10 routes
                novel_nodes -= prev_route
            
            diversity = len(novel_nodes) / max(len(route_nodes), 1)
        
        # Store current route for future comparisons
        self.previous_routes.append(route_nodes)
        if len(self.previous_routes) > self.max_history:
            self.previous_routes.pop(0)
        
        return diversity
    
    def get_ideal_value(self) -> float:
        return 1.0
    
    def get_nadir_value(self) -> float:
        return 0.0


class ConnectivityQualityEvaluator(FitnessComponentEvaluator):
    """Evaluates route connectivity and validity"""
    
    def evaluate(self, chromosome: RouteChromosome, context: Dict[str, Any]) -> float:
        if not chromosome.segments:
            return 0.0
        
        connectivity_score = 0.0
        total_connections = len(chromosome.segments) - 1
        
        if total_connections <= 0:
            return 1.0 if len(chromosome.segments) == 1 else 0.0
        
        # Check segment connectivity
        for i in range(len(chromosome.segments) - 1):
            current_segment = chromosome.segments[i]
            next_segment = chromosome.segments[i + 1]
            
            # Check if segments connect properly
            if current_segment.end_node == next_segment.start_node:
                connectivity_score += 1.0
            elif current_segment.end_node in next_segment.path_nodes:
                connectivity_score += 0.8  # Partial connection
            else:
                connectivity_score += 0.0  # No connection
        
        return connectivity_score / total_connections
    
    def get_ideal_value(self) -> float:
        return 1.0
    
    def get_nadir_value(self) -> float:
        return 0.0


class GradientSmoothnessEvaluator(FitnessComponentEvaluator):
    """Evaluates smoothness of elevation changes"""
    
    def evaluate(self, chromosome: RouteChromosome, context: Dict[str, Any]) -> float:
        if len(chromosome.segments) < 2:
            return 1.0
        
        gradient_changes = []
        
        for i in range(len(chromosome.segments) - 1):
            current_segment = chromosome.segments[i]
            next_segment = chromosome.segments[i + 1]
            
            # Calculate gradients
            current_gradient = (current_segment.elevation_gain / 
                              max(current_segment.length, 1.0)) * 100
            next_gradient = (next_segment.elevation_gain / 
                           max(next_segment.length, 1.0)) * 100
            
            gradient_change = abs(next_gradient - current_gradient)
            gradient_changes.append(gradient_change)
        
        if not gradient_changes:
            return 1.0
        
        # Calculate smoothness score
        avg_change = sum(gradient_changes) / len(gradient_changes)
        max_acceptable_change = 5.0  # 5% grade change
        
        smoothness = max(0.0, 1.0 - avg_change / max_acceptable_change)
        return smoothness
    
    def get_ideal_value(self) -> float:
        return 1.0
    
    def get_nadir_value(self) -> float:
        return 0.0


class TurnEfficiencyEvaluator(FitnessComponentEvaluator):
    """Evaluates turn efficiency and directional changes"""
    
    def evaluate(self, chromosome: RouteChromosome, context: Dict[str, Any]) -> float:
        if len(chromosome.segments) < 2:
            return 1.0
        
        direction_changes = 0
        total_segments = len(chromosome.segments)
        
        for i in range(1, len(chromosome.segments)):
            prev_segment = chromosome.segments[i - 1]
            current_segment = chromosome.segments[i]
            
            # Compare directions if available
            if (hasattr(prev_segment, 'direction') and hasattr(current_segment, 'direction') and
                prev_segment.direction and current_segment.direction):
                
                if prev_segment.direction != current_segment.direction:
                    direction_changes += 1
        
        # Favor routes with moderate number of turns
        if total_segments <= 1:
            return 1.0
        
        turn_ratio = direction_changes / total_segments
        
        # Optimal turn ratio is around 0.3-0.5 (not too straight, not too winding)
        if 0.2 <= turn_ratio <= 0.6:
            return 1.0
        elif turn_ratio < 0.2:
            return 0.5 + turn_ratio * 2.5  # Penalty for too straight
        else:
            return max(0.0, 1.5 - turn_ratio)  # Penalty for too winding
    
    def get_ideal_value(self) -> float:
        return 1.0
    
    def get_nadir_value(self) -> float:
        return 0.0


class NoveltyBonusEvaluator(FitnessComponentEvaluator):
    """Evaluates novelty and exploration bonus"""
    
    def __init__(self):
        self.population_archive = []
        self.novelty_threshold = 0.1
    
    def evaluate(self, chromosome: RouteChromosome, context: Dict[str, Any]) -> float:
        if not chromosome.segments:
            return 0.0
        
        current_population = context.get('current_population', [])
        
        # Calculate novelty based on distance to other solutions
        novelty_score = self._calculate_novelty(chromosome, current_population)
        
        # Bonus for highly novel solutions
        if novelty_score > self.novelty_threshold:
            return min(1.0, novelty_score * 2.0)
        else:
            return novelty_score
    
    def _calculate_novelty(self, chromosome: RouteChromosome, 
                          population: List[RouteChromosome]) -> float:
        """Calculate novelty score based on behavioral distance"""
        if not population:
            return 1.0
        
        # Extract behavioral characteristics
        behavior = self._extract_behavior(chromosome)
        
        distances = []
        for other in population:
            if other != chromosome:
                other_behavior = self._extract_behavior(other)
                distance = self._behavioral_distance(behavior, other_behavior)
                distances.append(distance)
        
        if not distances:
            return 1.0
        
        # Average distance to k-nearest neighbors
        k = min(5, len(distances))
        distances.sort()
        avg_distance = sum(distances[:k]) / k
        
        return min(1.0, avg_distance)
    
    def _extract_behavior(self, chromosome: RouteChromosome) -> Dict[str, float]:
        """Extract behavioral characteristics of chromosome"""
        if not chromosome.segments:
            return {}
        
        total_distance = sum(s.length for s in chromosome.segments) / 1000.0
        total_elevation = sum(max(0, s.elevation_gain) for s in chromosome.segments)
        num_segments = len(chromosome.segments)
        
        # Calculate spatial distribution (center of mass)
        if chromosome.segments and hasattr(chromosome.segments[0], 'path_nodes'):
            all_nodes = []
            for segment in chromosome.segments:
                all_nodes.extend(segment.path_nodes)
            
            if all_nodes:
                center_x = sum(node for node in all_nodes) / len(all_nodes)
                center_y = center_x  # Simplified for testing
            else:
                center_x = center_y = 0.0
        else:
            center_x = center_y = 0.0
        
        return {
            'distance': total_distance,
            'elevation': total_elevation,
            'segments': num_segments,
            'center_x': center_x,
            'center_y': center_y
        }
    
    def _behavioral_distance(self, behavior1: Dict[str, float], 
                           behavior2: Dict[str, float]) -> float:
        """Calculate distance between behavioral vectors"""
        if not behavior1 or not behavior2:
            return 1.0
        
        distance = 0.0
        count = 0
        
        for key in behavior1:
            if key in behavior2:
                # Normalize by typical ranges
                if key == 'distance':
                    norm_factor = 10.0  # 10km typical range
                elif key == 'elevation':
                    norm_factor = 500.0  # 500m typical range
                elif key == 'segments':
                    norm_factor = 20.0  # 20 segments typical
                else:
                    norm_factor = 1000.0  # For node coordinates
                
                diff = abs(behavior1[key] - behavior2[key]) / norm_factor
                distance += diff * diff
                count += 1
        
        return math.sqrt(distance / max(count, 1))
    
    def get_ideal_value(self) -> float:
        return 1.0
    
    def get_nadir_value(self) -> float:
        return 0.0


class EnhancedFitnessEvaluator:
    """Enhanced multi-objective fitness evaluator"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced fitness evaluator
        
        Args:
            config: Configuration options
        """
        default_config = {
            'enable_multi_objective': True,
            'adaptive_weights': False,
            'normalization_method': 'minmax',
            'aggregation_method': AggregationMethod.WEIGHTED_SUM,
            'pareto_ranking': False,
            'crowding_distance': False,
            'archive_size': 100,
            'diversity_preservation': True
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Initialize component evaluators
        self.evaluators = {
            FitnessComponent.DISTANCE_ACCURACY: DistanceAccuracyEvaluator(),
            FitnessComponent.ELEVATION_GAIN: ElevationGainEvaluator(),
            FitnessComponent.ROUTE_DIVERSITY: RouteDiversityEvaluator(),
            FitnessComponent.CONNECTIVITY_QUALITY: ConnectivityQualityEvaluator(),
            FitnessComponent.GRADIENT_SMOOTHNESS: GradientSmoothnessEvaluator(),
            FitnessComponent.TURN_EFFICIENCY: TurnEfficiencyEvaluator(),
            FitnessComponent.NOVELTY_BONUS: NoveltyBonusEvaluator()
        }
        
        # Fitness profiles for different scenarios
        self.profiles = self._create_fitness_profiles()
        self.active_profile = "balanced"
        
        # Multi-objective state
        self.objective_ranges = {}
        self.pareto_archive = []
        
        print("🎯 Enhanced Fitness Evaluator initialized with multi-objective support")
    
    def _create_fitness_profiles(self) -> Dict[str, FitnessProfile]:
        """Create predefined fitness profiles"""
        profiles = {}
        
        # Distance-focused profile
        distance_weights = FitnessWeights(
            distance_accuracy=0.7,
            elevation_gain=0.1,
            connectivity_quality=0.15,
            turn_efficiency=0.05
        )
        profiles["distance"] = FitnessProfile(
            name="distance",
            objective=RouteObjective.MINIMIZE_DISTANCE,
            weights=distance_weights
        )
        
        # Elevation-focused profile
        elevation_weights = FitnessWeights(
            distance_accuracy=0.3,
            elevation_gain=0.5,
            route_diversity=0.1,
            gradient_smoothness=0.1
        )
        profiles["elevation"] = FitnessProfile(
            name="elevation",
            objective=RouteObjective.MAXIMIZE_ELEVATION,
            weights=elevation_weights
        )
        
        # Balanced profile
        balanced_weights = FitnessWeights(
            distance_accuracy=0.4,
            elevation_gain=0.3,
            route_diversity=0.1,
            connectivity_quality=0.1,
            gradient_smoothness=0.05,
            turn_efficiency=0.05
        )
        profiles["balanced"] = FitnessProfile(
            name="balanced",
            objective=RouteObjective.BALANCED_ROUTE,
            weights=balanced_weights
        )
        
        # Exploration profile
        exploration_weights = FitnessWeights(
            distance_accuracy=0.25,
            elevation_gain=0.25,
            route_diversity=0.3,
            novelty_bonus=0.2
        )
        profiles["exploration"] = FitnessProfile(
            name="exploration",
            objective=RouteObjective.BALANCED_ROUTE,
            weights=exploration_weights,
            adaptive_weights=True
        )
        
        return profiles
    
    def set_fitness_profile(self, profile_name: str) -> bool:
        """Set active fitness profile
        
        Args:
            profile_name: Name of profile to activate
            
        Returns:
            True if successful, False otherwise
        """
        if profile_name in self.profiles:
            self.active_profile = profile_name
            print(f"🎯 Activated fitness profile: {profile_name}")
            return True
        else:
            print(f"❌ Unknown fitness profile: {profile_name}")
            return False
    
    def evaluate_population(self, population: List[RouteChromosome],
                          context: Dict[str, Any]) -> List[MultiObjectiveResult]:
        """Evaluate entire population with multi-objective fitness
        
        Args:
            population: Population of chromosomes
            context: Evaluation context
            
        Returns:
            List of multi-objective results
        """
        results = []
        
        # Add current population to context for novelty evaluation
        context['current_population'] = population
        
        # Evaluate each chromosome
        for i, chromosome in enumerate(population):
            result = self.evaluate_chromosome_multi_objective(chromosome, context)
            result.chromosome = chromosome
            results.append(result)
        
        # Apply multi-objective ranking if enabled
        if self.config['pareto_ranking']:
            results = self._apply_pareto_ranking(results)
        
        # Calculate crowding distance if enabled
        if self.config['crowding_distance']:
            results = self._calculate_crowding_distance(results)
        
        return results
    
    def evaluate_chromosome_multi_objective(self, chromosome: RouteChromosome,
                                          context: Dict[str, Any]) -> MultiObjectiveResult:
        """Evaluate chromosome with multiple objectives
        
        Args:
            chromosome: Chromosome to evaluate
            context: Evaluation context
            
        Returns:
            Multi-objective evaluation result
        """
        profile = self.profiles[self.active_profile]
        objective_values = {}
        
        # Evaluate each fitness component
        for component, evaluator in self.evaluators.items():
            try:
                value = evaluator.evaluate(chromosome, context)
                objective_values[component.value] = value
            except Exception as e:
                print(f"⚠️ Error evaluating {component.value}: {e}")
                objective_values[component.value] = 0.0
        
        # Apply aggregation method
        aggregated_fitness = self._aggregate_objectives(objective_values, profile)
        
        return MultiObjectiveResult(
            chromosome=chromosome,
            objective_values=objective_values,
            aggregated_fitness=aggregated_fitness
        )
    
    def _aggregate_objectives(self, objective_values: Dict[str, float],
                            profile: FitnessProfile) -> float:
        """Aggregate multiple objectives into single fitness value"""
        weights = profile.weights
        
        if profile.aggregation_method == AggregationMethod.WEIGHTED_SUM:
            total_fitness = 0.0
            total_weight = 0.0
            
            for component, value in objective_values.items():
                weight = getattr(weights, component, 0.0)
                total_fitness += value * weight
                total_weight += weight
            
            return total_fitness / max(total_weight, 0.001)
        
        elif profile.aggregation_method == AggregationMethod.PRODUCT:
            product = 1.0
            for component, value in objective_values.items():
                weight = getattr(weights, component, 0.0)
                if weight > 0:
                    product *= value ** weight
            return product
        
        elif profile.aggregation_method == AggregationMethod.TCHEBYCHEFF:
            # Tchebycheff scalarization
            reference_point = profile.reference_point or {}
            max_diff = 0.0
            
            for component, value in objective_values.items():
                weight = getattr(weights, component, 0.0)
                if weight > 0:
                    reference = reference_point.get(component, 1.0)
                    diff = weight * abs(reference - value)
                    max_diff = max(max_diff, diff)
            
            return 1.0 - max_diff  # Convert to maximization
        
        else:  # Default to weighted sum
            return self._aggregate_objectives(objective_values, 
                FitnessProfile("temp", profile.objective, weights, AggregationMethod.WEIGHTED_SUM))
    
    def _apply_pareto_ranking(self, results: List[MultiObjectiveResult]) -> List[MultiObjectiveResult]:
        """Apply Pareto ranking to population"""
        n = len(results)
        
        # Calculate dominance relationships
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(results[i], results[j]):
                        results[i].dominates.append(j)
                        results[j].dominated_by += 1
        
        # Assign Pareto fronts
        current_front = []
        front_number = 0
        
        # Find first front (non-dominated solutions)
        for i, result in enumerate(results):
            if result.dominated_by == 0:
                result.pareto_front = front_number
                current_front.append(i)
        
        # Find subsequent fronts
        while current_front:
            next_front = []
            
            for i in current_front:
                for j in results[i].dominates:
                    results[j].dominated_by -= 1
                    if results[j].dominated_by == 0:
                        results[j].pareto_front = front_number + 1
                        next_front.append(j)
            
            front_number += 1
            current_front = next_front
        
        # Assign ranks based on fronts
        for result in results:
            result.rank = result.pareto_front
        
        return results
    
    def _dominates(self, result1: MultiObjectiveResult, result2: MultiObjectiveResult) -> bool:
        """Check if result1 dominates result2 in Pareto sense"""
        better_in_any = False
        
        for objective in result1.objective_values:
            val1 = result1.objective_values[objective]
            val2 = result2.objective_values[objective]
            
            if val1 < val2:  # Assuming maximization
                return False
            elif val1 > val2:
                better_in_any = True
        
        return better_in_any
    
    def _calculate_crowding_distance(self, results: List[MultiObjectiveResult]) -> List[MultiObjectiveResult]:
        """Calculate crowding distance for diversity preservation"""
        if len(results) <= 2:
            for result in results:
                result.crowding_distance = float('inf')
            return results
        
        # Group by Pareto front
        fronts = {}
        for i, result in enumerate(results):
            front = result.pareto_front
            if front not in fronts:
                fronts[front] = []
            fronts[front].append(i)
        
        # Calculate crowding distance for each front
        for front_indices in fronts.values():
            if len(front_indices) <= 2:
                for idx in front_indices:
                    results[idx].crowding_distance = float('inf')
                continue
            
            # Initialize distances
            for idx in front_indices:
                results[idx].crowding_distance = 0.0
            
            # Calculate for each objective
            for objective in results[0].objective_values:
                # Sort by objective value
                front_indices.sort(key=lambda x: results[x].objective_values[objective])
                
                # Set boundary solutions to infinity
                results[front_indices[0]].crowding_distance = float('inf')
                results[front_indices[-1]].crowding_distance = float('inf')
                
                # Calculate range
                obj_range = (results[front_indices[-1]].objective_values[objective] -
                           results[front_indices[0]].objective_values[objective])
                
                if obj_range == 0:
                    continue
                
                # Calculate distances for intermediate solutions
                for i in range(1, len(front_indices) - 1):
                    idx = front_indices[i]
                    if results[idx].crowding_distance != float('inf'):
                        distance = (results[front_indices[i + 1]].objective_values[objective] -
                                  results[front_indices[i - 1]].objective_values[objective]) / obj_range
                        results[idx].crowding_distance += distance
        
        return results
    
    def get_fitness_statistics(self, results: List[MultiObjectiveResult]) -> Dict[str, Any]:
        """Get fitness evaluation statistics"""
        if not results:
            return {}
        
        stats = {
            'population_size': len(results),
            'objective_statistics': {},
            'aggregated_fitness': {
                'mean': np.mean([r.aggregated_fitness for r in results]),
                'std': np.std([r.aggregated_fitness for r in results]),
                'min': min(r.aggregated_fitness for r in results),
                'max': max(r.aggregated_fitness for r in results)
            }
        }
        
        # Calculate statistics for each objective
        for objective in results[0].objective_values:
            values = [r.objective_values[objective] for r in results]
            stats['objective_statistics'][objective] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': min(values),
                'max': max(values)
            }
        
        # Pareto front statistics
        if self.config['pareto_ranking']:
            fronts = {}
            for result in results:
                front = result.pareto_front
                fronts[front] = fronts.get(front, 0) + 1
            
            stats['pareto_fronts'] = fronts
            stats['first_front_size'] = fronts.get(0, 0)
        
        return stats


def test_enhanced_fitness():
    """Test function for enhanced fitness evaluator"""
    print("Testing Enhanced Fitness Evaluator...")
    
    # Create evaluator
    evaluator = EnhancedFitnessEvaluator({
        'enable_multi_objective': True,
        'pareto_ranking': True,
        'crowding_distance': True
    })
    
    # Create test chromosomes
    from ga_chromosome import RouteSegment
    
    test_population = []
    for i in range(5):
        segment = RouteSegment(1, 2, [1, 2])
        segment.length = 1000.0 + i * 200
        segment.elevation_gain = 20.0 + i * 10
        chromosome = RouteChromosome([segment])
        test_population.append(chromosome)
    
    # Test profile selection
    success = evaluator.set_fitness_profile("elevation")
    print(f"✅ Profile selection: {success}")
    
    # Test multi-objective evaluation
    context = {
        'target_distance_km': 5.0,
        'objective': RouteObjective.MAXIMIZE_ELEVATION,
        'distance_tolerance': 0.2
    }
    
    results = evaluator.evaluate_population(test_population, context)
    print(f"✅ Multi-objective evaluation: {len(results)} results")
    
    # Test individual chromosome evaluation
    single_result = evaluator.evaluate_chromosome_multi_objective(test_population[0], context)
    print(f"✅ Individual evaluation: {len(single_result.objective_values)} objectives")
    
    # Test fitness statistics
    stats = evaluator.get_fitness_statistics(results)
    print(f"✅ Fitness statistics: {len(stats)} categories")
    
    # Test component evaluators
    for component, eval_func in evaluator.evaluators.items():
        value = eval_func.evaluate(test_population[0], context)
        print(f"✅ {component.value}: {value:.3f}")
    
    print("✅ All enhanced fitness tests completed")


if __name__ == "__main__":
    test_enhanced_fitness()