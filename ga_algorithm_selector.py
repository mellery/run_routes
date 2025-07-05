#!/usr/bin/env python3
"""
GA Algorithm Performance Comparison and Selection System
Intelligent algorithm selection based on problem characteristics and performance analysis
"""

import time
import math
import numpy as np
import json
import os
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque, defaultdict

from tsp_solver_fast import RouteObjective
from ga_hyperparameter_optimizer import OptimizationMethod


class AlgorithmType(Enum):
    """Available algorithm types"""
    TSP_FAST = "tsp_fast"
    TSP_STANDARD = "tsp_standard"
    GENETIC_ALGORITHM = "genetic_algorithm"
    HYBRID_TSP_GA = "hybrid_tsp_ga"
    MULTI_OBJECTIVE_GA = "multi_objective_ga"


class ProblemCharacteristics(Enum):
    """Problem characteristic categories"""
    DISTANCE_FOCUSED = "distance_focused"
    ELEVATION_FOCUSED = "elevation_focused"
    BALANCED_OPTIMIZATION = "balanced_optimization"
    EXPLORATION_HEAVY = "exploration_heavy"
    SMALL_PROBLEM = "small_problem"
    LARGE_PROBLEM = "large_problem"
    TIME_CRITICAL = "time_critical"
    QUALITY_CRITICAL = "quality_critical"


@dataclass
class AlgorithmPerformance:
    """Performance metrics for an algorithm"""
    algorithm: AlgorithmType
    execution_time: float
    solution_quality: float
    convergence_speed: float
    memory_usage: float
    stability: float
    objective_satisfaction: float
    success_rate: float
    evaluation_count: int = 0
    last_updated: float = field(default_factory=time.time)


@dataclass
class ProblemInstance:
    """Instance of optimization problem"""
    objective: RouteObjective
    target_distance_km: float
    network_size: int
    time_constraint: Optional[float]
    quality_requirement: float
    characteristics: List[ProblemCharacteristics] = field(default_factory=list)
    complexity_score: float = 0.0


@dataclass
class SelectionDecision:
    """Algorithm selection decision with reasoning"""
    selected_algorithm: AlgorithmType
    confidence_score: float
    reasoning: List[str]
    fallback_algorithms: List[AlgorithmType]
    estimated_performance: AlgorithmPerformance
    selection_time: float = field(default_factory=time.time)


@dataclass
class AlgorithmComparison:
    """Comparison results between algorithms"""
    algorithms_compared: List[AlgorithmType]
    problem_instances: List[ProblemInstance]
    performance_matrix: Dict[str, Dict[str, float]]
    rankings: Dict[str, List[AlgorithmType]]
    statistical_significance: Dict[str, bool]
    comparison_time: float


class GAAlgorithmSelector:
    """Intelligent algorithm selection system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize algorithm selector
        
        Args:
            config: Configuration options for selector
        """
        default_config = {
            'performance_history_size': 100,    # Number of recent performances to track
            'min_evaluations_for_confidence': 5, # Minimum evaluations for reliable selection
            'confidence_threshold': 0.7,        # Minimum confidence for selection
            'performance_weight_time': 0.3,     # Weight for execution time
            'performance_weight_quality': 0.4,  # Weight for solution quality
            'performance_weight_stability': 0.2, # Weight for stability
            'performance_weight_convergence': 0.1, # Weight for convergence speed
            'auto_fallback_enabled': True,      # Enable automatic fallback
            'learning_rate': 0.1,              # Rate for updating algorithm preferences
            'exploration_factor': 0.2,         # Factor for exploring underused algorithms
            'save_selection_history': True,    # Save selection decisions
            'enable_statistical_analysis': True # Enable statistical performance analysis
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Performance tracking
        self.algorithm_performances = {alg: deque(maxlen=self.config['performance_history_size']) 
                                     for alg in AlgorithmType}
        
        # Algorithm preferences based on problem characteristics
        self.preference_matrix = self._initialize_preference_matrix()
        
        # Selection history
        self.selection_history = deque(maxlen=200)
        self.algorithm_usage_count = defaultdict(int)
        
        # Performance statistics
        self.performance_stats = {}
        self.comparison_cache = {}
        
        # Thread safety
        self.selection_lock = threading.RLock()
        
        print("🤖 GA Algorithm Selector initialized with intelligent selection")
    
    def _initialize_preference_matrix(self) -> Dict[ProblemCharacteristics, Dict[AlgorithmType, float]]:
        """Initialize algorithm preference matrix based on problem characteristics"""
        preferences = {}
        
        # Distance-focused problems
        preferences[ProblemCharacteristics.DISTANCE_FOCUSED] = {
            AlgorithmType.TSP_FAST: 0.9,
            AlgorithmType.TSP_STANDARD: 0.8,
            AlgorithmType.GENETIC_ALGORITHM: 0.4,
            AlgorithmType.HYBRID_TSP_GA: 0.7,
            AlgorithmType.MULTI_OBJECTIVE_GA: 0.3
        }
        
        # Elevation-focused problems
        preferences[ProblemCharacteristics.ELEVATION_FOCUSED] = {
            AlgorithmType.TSP_FAST: 0.3,
            AlgorithmType.TSP_STANDARD: 0.4,
            AlgorithmType.GENETIC_ALGORITHM: 0.9,
            AlgorithmType.HYBRID_TSP_GA: 0.8,
            AlgorithmType.MULTI_OBJECTIVE_GA: 0.9
        }
        
        # Balanced optimization
        preferences[ProblemCharacteristics.BALANCED_OPTIMIZATION] = {
            AlgorithmType.TSP_FAST: 0.6,
            AlgorithmType.TSP_STANDARD: 0.7,
            AlgorithmType.GENETIC_ALGORITHM: 0.8,
            AlgorithmType.HYBRID_TSP_GA: 0.9,
            AlgorithmType.MULTI_OBJECTIVE_GA: 0.8
        }
        
        # Exploration-heavy problems
        preferences[ProblemCharacteristics.EXPLORATION_HEAVY] = {
            AlgorithmType.TSP_FAST: 0.2,
            AlgorithmType.TSP_STANDARD: 0.3,
            AlgorithmType.GENETIC_ALGORITHM: 0.9,
            AlgorithmType.HYBRID_TSP_GA: 0.7,
            AlgorithmType.MULTI_OBJECTIVE_GA: 0.8
        }
        
        # Small problems
        preferences[ProblemCharacteristics.SMALL_PROBLEM] = {
            AlgorithmType.TSP_FAST: 0.9,
            AlgorithmType.TSP_STANDARD: 0.7,
            AlgorithmType.GENETIC_ALGORITHM: 0.5,
            AlgorithmType.HYBRID_TSP_GA: 0.6,
            AlgorithmType.MULTI_OBJECTIVE_GA: 0.4
        }
        
        # Large problems
        preferences[ProblemCharacteristics.LARGE_PROBLEM] = {
            AlgorithmType.TSP_FAST: 0.7,
            AlgorithmType.TSP_STANDARD: 0.5,
            AlgorithmType.GENETIC_ALGORITHM: 0.8,
            AlgorithmType.HYBRID_TSP_GA: 0.8,
            AlgorithmType.MULTI_OBJECTIVE_GA: 0.7
        }
        
        # Time-critical problems
        preferences[ProblemCharacteristics.TIME_CRITICAL] = {
            AlgorithmType.TSP_FAST: 0.9,
            AlgorithmType.TSP_STANDARD: 0.6,
            AlgorithmType.GENETIC_ALGORITHM: 0.3,
            AlgorithmType.HYBRID_TSP_GA: 0.5,
            AlgorithmType.MULTI_OBJECTIVE_GA: 0.2
        }
        
        # Quality-critical problems
        preferences[ProblemCharacteristics.QUALITY_CRITICAL] = {
            AlgorithmType.TSP_FAST: 0.4,
            AlgorithmType.TSP_STANDARD: 0.7,
            AlgorithmType.GENETIC_ALGORITHM: 0.9,
            AlgorithmType.HYBRID_TSP_GA: 0.9,
            AlgorithmType.MULTI_OBJECTIVE_GA: 0.8
        }
        
        return preferences
    
    def select_algorithm(self, problem: ProblemInstance) -> SelectionDecision:
        """Select best algorithm for given problem
        
        Args:
            problem: Problem instance with characteristics
            
        Returns:
            Selection decision with reasoning
        """
        with self.selection_lock:
            start_time = time.time()
            
            # Analyze problem characteristics
            problem.characteristics = self._analyze_problem_characteristics(problem)
            problem.complexity_score = self._calculate_complexity_score(problem)
            
            # Calculate algorithm scores
            algorithm_scores = self._calculate_algorithm_scores(problem)
            
            # Apply performance history
            performance_adjusted_scores = self._apply_performance_history(algorithm_scores, problem)
            
            # Apply exploration factor
            exploration_adjusted_scores = self._apply_exploration_factor(performance_adjusted_scores)
            
            # Select best algorithm
            selected_algorithm = max(exploration_adjusted_scores.items(), key=lambda x: x[1])[0]
            confidence_score = exploration_adjusted_scores[selected_algorithm]
            
            # Generate reasoning
            reasoning = self._generate_selection_reasoning(
                problem, algorithm_scores, performance_adjusted_scores, selected_algorithm
            )
            
            # Determine fallback algorithms
            fallback_algorithms = self._determine_fallback_algorithms(
                exploration_adjusted_scores, selected_algorithm
            )
            
            # Estimate performance
            estimated_performance = self._estimate_algorithm_performance(selected_algorithm, problem)
            
            # Create decision
            decision = SelectionDecision(
                selected_algorithm=selected_algorithm,
                confidence_score=confidence_score,
                reasoning=reasoning,
                fallback_algorithms=fallback_algorithms,
                estimated_performance=estimated_performance
            )
            
            # Record selection
            self._record_selection_decision(decision, problem)
            
            selection_time = time.time() - start_time
            print(f"🤖 Algorithm selected: {selected_algorithm.value} (confidence: {confidence_score:.2f}, "
                  f"time: {selection_time:.3f}s)")
            
            return decision
    
    def _analyze_problem_characteristics(self, problem: ProblemInstance) -> List[ProblemCharacteristics]:
        """Analyze problem to determine characteristics"""
        characteristics = []
        
        # Objective-based characteristics
        if problem.objective == RouteObjective.MINIMIZE_DISTANCE:
            characteristics.append(ProblemCharacteristics.DISTANCE_FOCUSED)
        elif problem.objective == RouteObjective.MAXIMIZE_ELEVATION:
            characteristics.append(ProblemCharacteristics.ELEVATION_FOCUSED)
            characteristics.append(ProblemCharacteristics.EXPLORATION_HEAVY)
        elif problem.objective == RouteObjective.BALANCED_ROUTE:
            characteristics.append(ProblemCharacteristics.BALANCED_OPTIMIZATION)
        
        # Size-based characteristics
        if problem.target_distance_km <= 3.0:
            characteristics.append(ProblemCharacteristics.SMALL_PROBLEM)
        elif problem.target_distance_km >= 8.0:
            characteristics.append(ProblemCharacteristics.LARGE_PROBLEM)
        
        # Constraint-based characteristics
        if problem.time_constraint and problem.time_constraint < 30.0:
            characteristics.append(ProblemCharacteristics.TIME_CRITICAL)
        
        if problem.quality_requirement >= 0.8:
            characteristics.append(ProblemCharacteristics.QUALITY_CRITICAL)
        
        # Network size characteristics
        if problem.network_size < 100:
            characteristics.append(ProblemCharacteristics.SMALL_PROBLEM)
        elif problem.network_size > 1000:
            characteristics.append(ProblemCharacteristics.LARGE_PROBLEM)
        
        return characteristics
    
    def _calculate_complexity_score(self, problem: ProblemInstance) -> float:
        """Calculate problem complexity score"""
        complexity = 0.0
        
        # Distance complexity
        distance_factor = min(1.0, problem.target_distance_km / 10.0)
        complexity += distance_factor * 0.3
        
        # Network size complexity
        network_factor = min(1.0, problem.network_size / 1000.0)
        complexity += network_factor * 0.3
        
        # Objective complexity
        if problem.objective == RouteObjective.MAXIMIZE_ELEVATION:
            complexity += 0.3  # Elevation optimization is more complex
        elif problem.objective == RouteObjective.BALANCED_ROUTE:
            complexity += 0.2
        
        # Quality requirement complexity
        quality_factor = problem.quality_requirement
        complexity += quality_factor * 0.1
        
        return min(1.0, complexity)
    
    def _calculate_algorithm_scores(self, problem: ProblemInstance) -> Dict[AlgorithmType, float]:
        """Calculate base algorithm scores based on problem characteristics"""
        scores = {alg: 0.0 for alg in AlgorithmType}
        
        # Apply preference matrix
        for characteristic in problem.characteristics:
            if characteristic in self.preference_matrix:
                char_preferences = self.preference_matrix[characteristic]
                weight = 1.0 / len(problem.characteristics)  # Equal weight for each characteristic
                
                for algorithm, preference in char_preferences.items():
                    scores[algorithm] += preference * weight
        
        # Apply complexity adjustments
        complexity_factor = problem.complexity_score
        
        # TSP algorithms perform better on less complex problems
        scores[AlgorithmType.TSP_FAST] *= (1.2 - complexity_factor * 0.4)
        scores[AlgorithmType.TSP_STANDARD] *= (1.1 - complexity_factor * 0.3)
        
        # GA algorithms perform better on more complex problems
        scores[AlgorithmType.GENETIC_ALGORITHM] *= (0.8 + complexity_factor * 0.4)
        scores[AlgorithmType.MULTI_OBJECTIVE_GA] *= (0.7 + complexity_factor * 0.5)
        scores[AlgorithmType.HYBRID_TSP_GA] *= (0.9 + complexity_factor * 0.2)
        
        # Normalize scores
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {alg: score / max_score for alg, score in scores.items()}
        
        return scores
    
    def _apply_performance_history(self, base_scores: Dict[AlgorithmType, float],
                                 problem: ProblemInstance) -> Dict[AlgorithmType, float]:
        """Apply performance history to adjust algorithm scores"""
        adjusted_scores = base_scores.copy()
        
        for algorithm in AlgorithmType:
            performance_history = self.algorithm_performances[algorithm]
            
            if len(performance_history) >= self.config['min_evaluations_for_confidence']:
                # Calculate average performance metrics
                avg_quality = np.mean([p.solution_quality for p in performance_history])
                avg_time = np.mean([p.execution_time for p in performance_history])
                avg_stability = np.mean([p.stability for p in performance_history])
                avg_convergence = np.mean([p.convergence_speed for p in performance_history])
                
                # Create performance score
                performance_score = (
                    avg_quality * self.config['performance_weight_quality'] +
                    (1.0 - min(1.0, avg_time / 60.0)) * self.config['performance_weight_time'] +
                    avg_stability * self.config['performance_weight_stability'] +
                    avg_convergence * self.config['performance_weight_convergence']
                )
                
                # Apply learning rate to adjust base score
                learning_rate = self.config['learning_rate']
                adjusted_scores[algorithm] = (
                    (1 - learning_rate) * adjusted_scores[algorithm] +
                    learning_rate * performance_score
                )
        
        return adjusted_scores
    
    def _apply_exploration_factor(self, scores: Dict[AlgorithmType, float]) -> Dict[AlgorithmType, float]:
        """Apply exploration factor to encourage trying underused algorithms"""
        exploration_scores = scores.copy()
        
        total_usage = sum(self.algorithm_usage_count.values())
        if total_usage == 0:
            return exploration_scores
        
        exploration_factor = self.config['exploration_factor']
        
        for algorithm in AlgorithmType:
            usage_ratio = self.algorithm_usage_count[algorithm] / total_usage
            
            # Boost score for underused algorithms
            exploration_bonus = exploration_factor * (1.0 - usage_ratio)
            exploration_scores[algorithm] += exploration_bonus
        
        # Normalize scores
        max_score = max(exploration_scores.values()) if exploration_scores else 1.0
        if max_score > 0:
            exploration_scores = {alg: score / max_score for alg, score in exploration_scores.items()}
        
        return exploration_scores
    
    def _generate_selection_reasoning(self, problem: ProblemInstance,
                                    base_scores: Dict[AlgorithmType, float],
                                    final_scores: Dict[AlgorithmType, float],
                                    selected_algorithm: AlgorithmType) -> List[str]:
        """Generate human-readable reasoning for algorithm selection"""
        reasoning = []
        
        # Problem characteristics reasoning
        if ProblemCharacteristics.DISTANCE_FOCUSED in problem.characteristics:
            reasoning.append("Problem is distance-focused, favoring TSP algorithms")
        
        if ProblemCharacteristics.ELEVATION_FOCUSED in problem.characteristics:
            reasoning.append("Problem seeks elevation optimization, favoring genetic algorithms")
        
        if ProblemCharacteristics.TIME_CRITICAL in problem.characteristics:
            reasoning.append("Time constraints favor fast algorithms")
        
        if ProblemCharacteristics.QUALITY_CRITICAL in problem.characteristics:
            reasoning.append("Quality requirements favor thorough search algorithms")
        
        if problem.complexity_score > 0.7:
            reasoning.append("High problem complexity favors evolutionary approaches")
        elif problem.complexity_score < 0.3:
            reasoning.append("Low problem complexity allows efficient exact algorithms")
        
        # Performance history reasoning
        selected_performance = self.algorithm_performances[selected_algorithm]
        if len(selected_performance) >= 3:
            avg_quality = np.mean([p.solution_quality for p in selected_performance])
            if avg_quality > 0.8:
                reasoning.append(f"{selected_algorithm.value} has strong historical performance")
        
        # Score-based reasoning
        base_score = base_scores[selected_algorithm]
        final_score = final_scores[selected_algorithm]
        
        if final_score > base_score + 0.1:
            reasoning.append("Performance history boosted algorithm preference")
        elif final_score < base_score - 0.1:
            reasoning.append("Performance history reduced algorithm preference")
        
        # Confidence reasoning
        if final_score > 0.8:
            reasoning.append("High confidence in algorithm selection")
        elif final_score < 0.6:
            reasoning.append("Moderate confidence - consider fallback options")
        
        return reasoning
    
    def _determine_fallback_algorithms(self, scores: Dict[AlgorithmType, float],
                                     selected_algorithm: AlgorithmType) -> List[AlgorithmType]:
        """Determine fallback algorithms in case selected algorithm fails"""
        # Sort algorithms by score, excluding selected
        sorted_algorithms = sorted(
            [(alg, score) for alg, score in scores.items() if alg != selected_algorithm],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top 2 fallback algorithms
        return [alg for alg, _ in sorted_algorithms[:2]]
    
    def _estimate_algorithm_performance(self, algorithm: AlgorithmType,
                                      problem: ProblemInstance) -> AlgorithmPerformance:
        """Estimate algorithm performance for given problem"""
        performance_history = self.algorithm_performances[algorithm]
        
        if len(performance_history) >= 3:
            # Use historical data
            return AlgorithmPerformance(
                algorithm=algorithm,
                execution_time=np.mean([p.execution_time for p in performance_history]),
                solution_quality=np.mean([p.solution_quality for p in performance_history]),
                convergence_speed=np.mean([p.convergence_speed for p in performance_history]),
                memory_usage=np.mean([p.memory_usage for p in performance_history]),
                stability=np.mean([p.stability for p in performance_history]),
                objective_satisfaction=np.mean([p.objective_satisfaction for p in performance_history]),
                success_rate=np.mean([p.success_rate for p in performance_history]),
                evaluation_count=len(performance_history)
            )
        else:
            # Use default estimates based on algorithm type and problem characteristics
            return self._get_default_performance_estimate(algorithm, problem)
    
    def _get_default_performance_estimate(self, algorithm: AlgorithmType,
                                        problem: ProblemInstance) -> AlgorithmPerformance:
        """Get default performance estimates for algorithms"""
        complexity_factor = problem.complexity_score
        
        if algorithm == AlgorithmType.TSP_FAST:
            return AlgorithmPerformance(
                algorithm=algorithm,
                execution_time=5.0 + complexity_factor * 10.0,
                solution_quality=0.8 - complexity_factor * 0.2,
                convergence_speed=0.9,
                memory_usage=0.3,
                stability=0.9,
                objective_satisfaction=0.7 if problem.objective == RouteObjective.MINIMIZE_DISTANCE else 0.5,
                success_rate=0.95
            )
        
        elif algorithm == AlgorithmType.TSP_STANDARD:
            return AlgorithmPerformance(
                algorithm=algorithm,
                execution_time=15.0 + complexity_factor * 30.0,
                solution_quality=0.85 - complexity_factor * 0.1,
                convergence_speed=0.7,
                memory_usage=0.5,
                stability=0.9,
                objective_satisfaction=0.8 if problem.objective == RouteObjective.MINIMIZE_DISTANCE else 0.6,
                success_rate=0.9
            )
        
        elif algorithm == AlgorithmType.GENETIC_ALGORITHM:
            return AlgorithmPerformance(
                algorithm=algorithm,
                execution_time=45.0 + complexity_factor * 60.0,
                solution_quality=0.7 + complexity_factor * 0.2,
                convergence_speed=0.6,
                memory_usage=0.7,
                stability=0.7,
                objective_satisfaction=0.9 if problem.objective == RouteObjective.MAXIMIZE_ELEVATION else 0.7,
                success_rate=0.85
            )
        
        else:  # Default for other algorithms
            return AlgorithmPerformance(
                algorithm=algorithm,
                execution_time=30.0,
                solution_quality=0.75,
                convergence_speed=0.7,
                memory_usage=0.6,
                stability=0.8,
                objective_satisfaction=0.75,
                success_rate=0.8
            )
    
    def _record_selection_decision(self, decision: SelectionDecision, problem: ProblemInstance):
        """Record selection decision for learning"""
        self.selection_history.append({
            'decision': decision,
            'problem': problem,
            'timestamp': time.time()
        })
        
        self.algorithm_usage_count[decision.selected_algorithm] += 1
    
    def record_algorithm_performance(self, algorithm: AlgorithmType, performance: AlgorithmPerformance):
        """Record actual algorithm performance for learning
        
        Args:
            algorithm: Algorithm that was executed
            performance: Actual performance metrics
        """
        with self.selection_lock:
            self.algorithm_performances[algorithm].append(performance)
            
            # Update performance statistics
            if algorithm not in self.performance_stats:
                self.performance_stats[algorithm] = {
                    'total_evaluations': 0,
                    'avg_quality': 0.0,
                    'avg_time': 0.0,
                    'success_rate': 0.0
                }
            
            stats = self.performance_stats[algorithm]
            stats['total_evaluations'] += 1
            
            # Update running averages
            n = stats['total_evaluations']
            stats['avg_quality'] = ((n-1) * stats['avg_quality'] + performance.solution_quality) / n
            stats['avg_time'] = ((n-1) * stats['avg_time'] + performance.execution_time) / n
            stats['success_rate'] = ((n-1) * stats['success_rate'] + performance.success_rate) / n
            
            print(f"📊 Performance recorded for {algorithm.value}: "
                  f"quality={performance.solution_quality:.3f}, time={performance.execution_time:.1f}s")
    
    def compare_algorithms(self, algorithms: List[AlgorithmType],
                          test_problems: List[ProblemInstance]) -> AlgorithmComparison:
        """Compare algorithms on test problems
        
        Args:
            algorithms: Algorithms to compare
            test_problems: Test problems for comparison
            
        Returns:
            Comparison results with rankings
        """
        print(f"🔬 Comparing {len(algorithms)} algorithms on {len(test_problems)} test problems")
        
        start_time = time.time()
        performance_matrix = {}
        
        # Initialize performance matrix
        for algorithm in algorithms:
            performance_matrix[algorithm.value] = {
                'avg_quality': 0.0,
                'avg_time': 0.0,
                'avg_stability': 0.0,
                'success_rate': 0.0,
                'objective_satisfaction': 0.0
            }
        
        # Simulate algorithm performance on test problems
        for problem in test_problems:
            for algorithm in algorithms:
                # Get performance estimate or historical data
                performance = self._estimate_algorithm_performance(algorithm, problem)
                
                # Add to performance matrix
                perf_data = performance_matrix[algorithm.value]
                perf_data['avg_quality'] += performance.solution_quality / len(test_problems)
                perf_data['avg_time'] += performance.execution_time / len(test_problems)
                perf_data['avg_stability'] += performance.stability / len(test_problems)
                perf_data['success_rate'] += performance.success_rate / len(test_problems)
                perf_data['objective_satisfaction'] += performance.objective_satisfaction / len(test_problems)
        
        # Calculate rankings
        rankings = {}
        metrics = ['avg_quality', 'avg_stability', 'success_rate', 'objective_satisfaction']
        
        for metric in metrics:
            algorithm_scores = [(alg, performance_matrix[alg.value][metric]) for alg in algorithms]
            algorithm_scores.sort(key=lambda x: x[1], reverse=True)
            rankings[metric] = [alg for alg, _ in algorithm_scores]
        
        # Time ranking (lower is better)
        time_scores = [(alg, performance_matrix[alg.value]['avg_time']) for alg in algorithms]
        time_scores.sort(key=lambda x: x[1])
        rankings['avg_time'] = [alg for alg, _ in time_scores]
        
        # Statistical significance (simplified)
        statistical_significance = {}
        for metric in metrics + ['avg_time']:
            # Simple significance test based on performance differences
            scores = [performance_matrix[alg.value][metric] for alg in algorithms]
            score_std = np.std(scores) if len(scores) > 1 else 0
            score_range = max(scores) - min(scores) if scores else 0
            
            # Consider significant if range > 2 * std
            statistical_significance[metric] = score_range > 2 * score_std
        
        comparison_time = time.time() - start_time
        
        comparison = AlgorithmComparison(
            algorithms_compared=algorithms,
            problem_instances=test_problems,
            performance_matrix=performance_matrix,
            rankings=rankings,
            statistical_significance=statistical_significance,
            comparison_time=comparison_time
        )
        
        print(f"🔬 Algorithm comparison completed in {comparison_time:.2f}s")
        
        return comparison
    
    def get_selection_recommendations(self) -> Dict[str, Any]:
        """Get algorithm selection recommendations based on learning
        
        Returns:
            Dictionary with recommendations and insights
        """
        recommendations = {
            'algorithm_preferences': {},
            'performance_insights': {},
            'usage_statistics': {},
            'optimization_suggestions': []
        }
        
        # Algorithm preferences by problem type
        for char in ProblemCharacteristics:
            if char in self.preference_matrix:
                sorted_prefs = sorted(
                    self.preference_matrix[char].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                recommendations['algorithm_preferences'][char.value] = [
                    {'algorithm': alg.value, 'preference': pref}
                    for alg, pref in sorted_prefs[:3]
                ]
        
        # Performance insights
        for algorithm in AlgorithmType:
            if algorithm in self.performance_stats:
                stats = self.performance_stats[algorithm]
                recommendations['performance_insights'][algorithm.value] = stats
        
        # Usage statistics
        total_usage = sum(self.algorithm_usage_count.values())
        if total_usage > 0:
            for algorithm, count in self.algorithm_usage_count.items():
                recommendations['usage_statistics'][algorithm.value] = {
                    'usage_count': count,
                    'usage_percentage': (count / total_usage) * 100
                }
        
        # Optimization suggestions
        suggestions = []
        
        # Underused algorithms
        if total_usage > 10:
            avg_usage = total_usage / len(AlgorithmType)
            for algorithm, count in self.algorithm_usage_count.items():
                if count < avg_usage * 0.5:
                    suggestions.append(f"Consider exploring {algorithm.value} - it may be underutilized")
        
        # Performance-based suggestions
        for algorithm, stats in self.performance_stats.items():
            if stats['total_evaluations'] >= 5:
                if stats['success_rate'] < 0.7:
                    suggestions.append(f"{algorithm.value} has low success rate - check implementation")
                if stats['avg_quality'] > 0.9:
                    suggestions.append(f"{algorithm.value} shows excellent quality - consider using more often")
        
        recommendations['optimization_suggestions'] = suggestions
        
        return recommendations
    
    def save_selector_state(self, filename: str) -> str:
        """Save selector state to file
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        state_data = {
            'config': self.config,
            'preference_matrix': {
                char.value: {alg.value: pref for alg, pref in prefs.items()}
                for char, prefs in self.preference_matrix.items()
            },
            'performance_stats': {
                alg.value: stats for alg, stats in self.performance_stats.items()
            },
            'algorithm_usage_count': {
                alg.value: count for alg, count in self.algorithm_usage_count.items()
            },
            'selection_history_summary': {
                'total_selections': len(self.selection_history),
                'recent_selections': [
                    {
                        'algorithm': entry['decision'].selected_algorithm.value,
                        'confidence': entry['decision'].confidence_score,
                        'timestamp': entry['timestamp']
                    }
                    for entry in list(self.selection_history)[-10:]
                ]
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        return filename
    
    def load_selector_state(self, filename: str):
        """Load selector state from file
        
        Args:
            filename: Input filename
        """
        if not os.path.exists(filename):
            print(f"⚠️ Selector state file not found: {filename}")
            return
        
        try:
            with open(filename, 'r') as f:
                state_data = json.load(f)
            
            # Load performance stats
            if 'performance_stats' in state_data:
                for alg_name, stats in state_data['performance_stats'].items():
                    try:
                        algorithm = AlgorithmType(alg_name)
                        self.performance_stats[algorithm] = stats
                    except ValueError:
                        pass
            
            # Load usage counts
            if 'algorithm_usage_count' in state_data:
                for alg_name, count in state_data['algorithm_usage_count'].items():
                    try:
                        algorithm = AlgorithmType(alg_name)
                        self.algorithm_usage_count[algorithm] = count
                    except ValueError:
                        pass
            
            print(f"✅ Selector state loaded from {filename}")
            
        except Exception as e:
            print(f"⚠️ Error loading selector state: {e}")
    
    def reset_learning_state(self):
        """Reset learning state for fresh start"""
        for algorithm in AlgorithmType:
            self.algorithm_performances[algorithm].clear()
        
        self.selection_history.clear()
        self.algorithm_usage_count.clear()
        self.performance_stats.clear()
        self.comparison_cache.clear()
        
        print("🤖 Algorithm selector learning state reset")


def test_algorithm_selector():
    """Test function for algorithm selector"""
    print("Testing GA Algorithm Selector...")
    
    # Create selector
    selector = GAAlgorithmSelector()
    
    # Create test problem
    test_problem = ProblemInstance(
        objective=RouteObjective.MAXIMIZE_ELEVATION,
        target_distance_km=5.0,
        network_size=500,
        time_constraint=60.0,
        quality_requirement=0.8
    )
    
    # Test algorithm selection
    decision = selector.select_algorithm(test_problem)
    print(f"✅ Algorithm selection: {decision.selected_algorithm.value} "
          f"(confidence: {decision.confidence_score:.2f})")
    print(f"   Reasoning: {decision.reasoning}")
    
    # Test performance recording
    performance = AlgorithmPerformance(
        algorithm=decision.selected_algorithm,
        execution_time=45.0,
        solution_quality=0.85,
        convergence_speed=0.7,
        memory_usage=0.6,
        stability=0.8,
        objective_satisfaction=0.9,
        success_rate=1.0
    )
    
    selector.record_algorithm_performance(decision.selected_algorithm, performance)
    print(f"✅ Performance recorded for {decision.selected_algorithm.value}")
    
    # Test algorithm comparison
    test_algorithms = [AlgorithmType.TSP_FAST, AlgorithmType.GENETIC_ALGORITHM]
    test_problems = [test_problem]
    
    comparison = selector.compare_algorithms(test_algorithms, test_problems)
    print(f"✅ Algorithm comparison completed: {len(comparison.rankings)} metrics")
    
    # Test recommendations
    recommendations = selector.get_selection_recommendations()
    print(f"✅ Selection recommendations: {len(recommendations)} categories")
    
    print("✅ All algorithm selector tests completed")


if __name__ == "__main__":
    test_algorithm_selector()