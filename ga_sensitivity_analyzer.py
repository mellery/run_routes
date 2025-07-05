#!/usr/bin/env python3
"""
GA Parameter Sensitivity Analysis and Tuning Recommendations
Advanced analysis of parameter sensitivity and automated tuning recommendations
"""

import numpy as np
import time
import json
import itertools
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict
import math
from concurrent.futures import ThreadPoolExecutor

from ga_parameter_tuner import GAParameterTuner, ParameterRange, PopulationStats
from tsp_solver_fast import RouteObjective


class SensitivityMethod(Enum):
    """Methods for sensitivity analysis"""
    ONE_AT_A_TIME = "one_at_a_time"      # Vary one parameter at a time
    MORRIS = "morris"                     # Morris method (global sensitivity)
    SOBOL = "sobol"                       # Sobol indices
    FACTORIAL = "factorial"               # Factorial design
    LATIN_HYPERCUBE = "latin_hypercube"   # Latin hypercube sampling
    VARIANCE_BASED = "variance_based"     # Variance-based methods


class AnalysisScope(Enum):
    """Scope of sensitivity analysis"""
    INDIVIDUAL_PARAMETERS = "individual"  # Individual parameter effects
    PAIRWISE_INTERACTIONS = "pairwise"   # Two-parameter interactions
    HIGHER_ORDER = "higher_order"        # Higher-order interactions
    GLOBAL_SENSITIVITY = "global"        # Global sensitivity indices


@dataclass
class SensitivityResult:
    """Result of parameter sensitivity analysis"""
    parameter_name: str
    sensitivity_index: float
    confidence_interval: Tuple[float, float]
    variance_contribution: float
    interaction_effects: Dict[str, float] = field(default_factory=dict)
    statistical_significance: bool = False
    rank: int = 0


@dataclass
class ParameterSample:
    """Sample point for sensitivity analysis"""
    parameters: Dict[str, float]
    performance_metrics: Dict[str, float]
    execution_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class TuningRecommendation:
    """Parameter tuning recommendation"""
    parameter_name: str
    current_value: float
    recommended_value: float
    confidence: float
    rationale: str
    expected_improvement: float
    priority: str  # "high", "medium", "low"
    sensitivity_rank: int


@dataclass
class OptimizationInsight:
    """Insight about optimization behavior"""
    insight_type: str
    description: str
    affected_parameters: List[str]
    confidence: float
    actionable_recommendation: str


class GASensitivityAnalyzer:
    """Advanced parameter sensitivity analysis and tuning recommendations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize sensitivity analyzer
        
        Args:
            config: Configuration options for analysis
        """
        default_config = {
            'sampling_budget': 1000,          # Total number of parameter samples
            'sensitivity_method': SensitivityMethod.MORRIS,  # Default method
            'analysis_scope': AnalysisScope.INDIVIDUAL_PARAMETERS,  # Analysis scope
            'confidence_level': 0.95,         # Statistical confidence level
            'min_samples_per_parameter': 20,  # Minimum samples per parameter
            'parallel_evaluation': True,      # Enable parallel sample evaluation
            'max_workers': 4,                 # Maximum parallel workers
            'bootstrap_samples': 1000,        # Bootstrap samples for confidence intervals
            'interaction_threshold': 0.1,     # Threshold for interaction effects
            'significance_threshold': 0.05,   # Statistical significance threshold
            'performance_metrics': ['fitness', 'convergence_speed', 'diversity'],
            'enable_caching': True,           # Cache evaluation results
            'save_analysis_results': True     # Save results to files
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Analysis state
        self.parameter_ranges = {}
        self.sample_history = []
        self.sensitivity_results = {}
        self.recommendations = []
        self.insights = []
        
        # Caching and performance
        self.evaluation_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Thread safety
        self.analysis_lock = threading.RLock()
        
        print(f"📊 GA Sensitivity Analyzer initialized with {self.config['sensitivity_method'].value} method")
    
    def set_parameter_ranges(self, parameter_ranges: Dict[str, ParameterRange]):
        """Set parameter ranges for analysis
        
        Args:
            parameter_ranges: Dictionary of parameter ranges
        """
        self.parameter_ranges = parameter_ranges.copy()
        print(f"📊 Set parameter ranges: {len(self.parameter_ranges)} parameters")
    
    def analyze_parameter_sensitivity(self, 
                                    performance_evaluator: Callable[[Dict[str, float]], Dict[str, float]],
                                    focus_parameters: Optional[List[str]] = None) -> Dict[str, SensitivityResult]:
        """Perform comprehensive parameter sensitivity analysis
        
        Args:
            performance_evaluator: Function to evaluate parameter configurations
            focus_parameters: Specific parameters to analyze (optional)
            
        Returns:
            Dictionary of sensitivity results by parameter
        """
        with self.analysis_lock:
            print(f"📊 Starting sensitivity analysis with {self.config['sampling_budget']} samples")
            
            start_time = time.time()
            
            # Determine parameters to analyze
            if focus_parameters:
                analysis_params = {k: v for k, v in self.parameter_ranges.items() 
                                 if k in focus_parameters}
            else:
                analysis_params = self.parameter_ranges.copy()
            
            # Generate parameter samples
            samples = self._generate_parameter_samples(analysis_params)
            
            # Evaluate samples
            evaluated_samples = self._evaluate_samples(samples, performance_evaluator)
            
            # Calculate sensitivity indices
            sensitivity_results = self._calculate_sensitivity_indices(evaluated_samples, analysis_params)
            
            # Calculate interaction effects if enabled
            if self.config['analysis_scope'] in [AnalysisScope.PAIRWISE_INTERACTIONS, 
                                                AnalysisScope.GLOBAL_SENSITIVITY]:
                sensitivity_results = self._calculate_interaction_effects(
                    sensitivity_results, evaluated_samples, analysis_params
                )
            
            # Assign statistical significance and rankings
            sensitivity_results = self._assign_significance_and_ranking(sensitivity_results)
            
            self.sensitivity_results = sensitivity_results
            analysis_time = time.time() - start_time
            
            print(f"📊 Sensitivity analysis completed in {analysis_time:.2f}s")
            print(f"   Cache efficiency: {self.cache_hits}/{self.cache_hits + self.cache_misses} hits")
            
            return sensitivity_results
    
    def _generate_parameter_samples(self, parameters: Dict[str, ParameterRange]) -> List[Dict[str, float]]:
        """Generate parameter samples using specified method"""
        method = self.config['sensitivity_method']
        budget = self.config['sampling_budget']
        
        if method == SensitivityMethod.ONE_AT_A_TIME:
            return self._generate_oat_samples(parameters, budget)
        elif method == SensitivityMethod.MORRIS:
            return self._generate_morris_samples(parameters, budget)
        elif method == SensitivityMethod.LATIN_HYPERCUBE:
            return self._generate_lhs_samples(parameters, budget)
        elif method == SensitivityMethod.FACTORIAL:
            return self._generate_factorial_samples(parameters, budget)
        else:
            # Default to Latin Hypercube
            return self._generate_lhs_samples(parameters, budget)
    
    def _generate_oat_samples(self, parameters: Dict[str, ParameterRange], 
                             budget: int) -> List[Dict[str, float]]:
        """Generate One-At-a-Time samples"""
        samples = []
        samples_per_param = max(self.config['min_samples_per_parameter'], 
                               budget // (len(parameters) + 1))
        
        # Base configuration (center point)
        base_config = {}
        for name, param_range in parameters.items():
            base_config[name] = (param_range.min_value + param_range.max_value) / 2
        
        samples.append(base_config.copy())
        
        # Vary each parameter individually
        for param_name, param_range in parameters.items():
            for i in range(samples_per_param):
                config = base_config.copy()
                # Linear sampling across parameter range
                alpha = i / max(samples_per_param - 1, 1)
                config[param_name] = param_range.min_value + alpha * (param_range.max_value - param_range.min_value)
                samples.append(config)
        
        return samples[:budget]
    
    def _generate_morris_samples(self, parameters: Dict[str, ParameterRange], 
                                budget: int) -> List[Dict[str, float]]:
        """Generate Morris method samples"""
        samples = []
        param_names = list(parameters.keys())
        num_params = len(param_names)
        
        # Number of trajectories
        num_trajectories = budget // (num_params + 1)
        
        for trajectory in range(num_trajectories):
            # Random starting point
            base_point = {}
            for name, param_range in parameters.items():
                base_point[name] = np.random.uniform(param_range.min_value, param_range.max_value)
            
            samples.append(base_point.copy())
            
            # Generate trajectory by changing one parameter at a time
            current_point = base_point.copy()
            param_order = np.random.permutation(param_names)
            
            for param_name in param_order:
                param_range = parameters[param_name]
                # Delta step (typically 10% of range)
                delta = (param_range.max_value - param_range.min_value) * 0.1
                
                # Ensure we stay within bounds
                if current_point[param_name] + delta <= param_range.max_value:
                    current_point[param_name] += delta
                else:
                    current_point[param_name] -= delta
                
                samples.append(current_point.copy())
        
        return samples[:budget]
    
    def _generate_lhs_samples(self, parameters: Dict[str, ParameterRange], 
                             budget: int) -> List[Dict[str, float]]:
        """Generate Latin Hypercube Sampling samples"""
        param_names = list(parameters.keys())
        num_params = len(param_names)
        
        # Generate LHS design
        lhs_design = np.zeros((budget, num_params))
        
        for i in range(num_params):
            # Generate random permutation of intervals
            intervals = np.random.permutation(budget)
            # Add random uniform within each interval
            uniform_random = np.random.uniform(0, 1, budget)
            lhs_design[:, i] = (intervals + uniform_random) / budget
        
        # Convert to actual parameter values
        samples = []
        for row in lhs_design:
            sample = {}
            for i, param_name in enumerate(param_names):
                param_range = parameters[param_name]
                value = param_range.min_value + row[i] * (param_range.max_value - param_range.min_value)
                
                # Handle integer constraints
                if param_range.constraint_type == 'integer':
                    value = round(value)
                
                sample[param_name] = param_range.clamp(value)
            
            samples.append(sample)
        
        return samples
    
    def _generate_factorial_samples(self, parameters: Dict[str, ParameterRange], 
                                   budget: int) -> List[Dict[str, float]]:
        """Generate factorial design samples"""
        param_names = list(parameters.keys())
        num_params = len(param_names)
        
        # Determine levels per parameter to fit budget
        levels_per_param = max(2, int(budget ** (1/num_params)))
        
        # Generate level values for each parameter
        param_levels = {}
        for param_name, param_range in parameters.items():
            levels = []
            for i in range(levels_per_param):
                alpha = i / max(levels_per_param - 1, 1)
                value = param_range.min_value + alpha * (param_range.max_value - param_range.min_value)
                
                if param_range.constraint_type == 'integer':
                    value = round(value)
                
                levels.append(param_range.clamp(value))
            
            param_levels[param_name] = levels
        
        # Generate all combinations
        samples = []
        level_combinations = itertools.product(*param_levels.values())
        
        for combination in level_combinations:
            sample = dict(zip(param_names, combination))
            samples.append(sample)
            
            if len(samples) >= budget:
                break
        
        return samples
    
    def _evaluate_samples(self, samples: List[Dict[str, float]], 
                         performance_evaluator: Callable[[Dict[str, float]], Dict[str, float]]) -> List[ParameterSample]:
        """Evaluate parameter samples"""
        evaluated_samples = []
        
        if self.config['parallel_evaluation']:
            # Parallel evaluation
            with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
                futures = []
                
                for sample in samples:
                    future = executor.submit(self._evaluate_single_sample, sample, performance_evaluator)
                    futures.append(future)
                
                for future in futures:
                    try:
                        result = future.result(timeout=60)  # 60 second timeout
                        evaluated_samples.append(result)
                    except Exception as e:
                        print(f"⚠️ Sample evaluation failed: {e}")
        else:
            # Sequential evaluation
            for sample in samples:
                result = self._evaluate_single_sample(sample, performance_evaluator)
                evaluated_samples.append(result)
        
        print(f"📊 Evaluated {len(evaluated_samples)} parameter samples")
        return evaluated_samples
    
    def _evaluate_single_sample(self, sample: Dict[str, float], 
                               performance_evaluator: Callable[[Dict[str, float]], Dict[str, float]]) -> ParameterSample:
        """Evaluate single parameter sample"""
        start_time = time.time()
        
        # Check cache
        if self.config['enable_caching']:
            cache_key = str(sorted(sample.items()))
            if cache_key in self.evaluation_cache:
                self.cache_hits += 1
                cached_result = self.evaluation_cache[cache_key]
                return ParameterSample(
                    parameters=sample,
                    performance_metrics=cached_result['metrics'],
                    execution_time=cached_result['time'],
                    success=cached_result['success']
                )
            else:
                self.cache_misses += 1
        
        # Evaluate sample
        try:
            performance_metrics = performance_evaluator(sample)
            execution_time = time.time() - start_time
            success = True
            error_message = None
            
            # Cache result
            if self.config['enable_caching']:
                self.evaluation_cache[cache_key] = {
                    'metrics': performance_metrics,
                    'time': execution_time,
                    'success': success
                }
            
        except Exception as e:
            performance_metrics = {metric: 0.0 for metric in self.config['performance_metrics']}
            execution_time = time.time() - start_time
            success = False
            error_message = str(e)
        
        return ParameterSample(
            parameters=sample,
            performance_metrics=performance_metrics,
            execution_time=execution_time,
            success=success,
            error_message=error_message
        )
    
    def _calculate_sensitivity_indices(self, samples: List[ParameterSample], 
                                     parameters: Dict[str, ParameterRange]) -> Dict[str, SensitivityResult]:
        """Calculate sensitivity indices for each parameter"""
        sensitivity_results = {}
        
        for metric in self.config['performance_metrics']:
            # Extract successful samples
            valid_samples = [s for s in samples if s.success]
            if len(valid_samples) < 10:
                print(f"⚠️ Insufficient valid samples for {metric}: {len(valid_samples)}")
                continue
            
            # Calculate sensitivity for each parameter
            for param_name in parameters:
                try:
                    sensitivity_index, confidence_interval, variance_contribution = \
                        self._calculate_parameter_sensitivity(valid_samples, param_name, metric)
                    
                    result_key = f"{param_name}_{metric}"
                    sensitivity_results[result_key] = SensitivityResult(
                        parameter_name=param_name,
                        sensitivity_index=sensitivity_index,
                        confidence_interval=confidence_interval,
                        variance_contribution=variance_contribution
                    )
                    
                except Exception as e:
                    print(f"⚠️ Error calculating sensitivity for {param_name}-{metric}: {e}")
        
        return sensitivity_results
    
    def _calculate_parameter_sensitivity(self, samples: List[ParameterSample], 
                                       param_name: str, metric: str) -> Tuple[float, Tuple[float, float], float]:
        """Calculate sensitivity index for specific parameter and metric"""
        # Extract parameter values and metric values
        param_values = [s.parameters[param_name] for s in samples]
        metric_values = [s.performance_metrics[metric] for s in samples]
        
        # Calculate correlation-based sensitivity
        correlation = np.corrcoef(param_values, metric_values)[0, 1]
        sensitivity_index = abs(correlation) if not np.isnan(correlation) else 0.0
        
        # Calculate variance contribution using ANOVA-like approach
        # Bin parameter values and calculate between-group variance
        num_bins = min(10, len(set(param_values)))
        if num_bins > 1:
            # Create bins
            param_range = max(param_values) - min(param_values)
            if param_range > 0:
                bin_size = param_range / num_bins
                bins = defaultdict(list)
                
                for param_val, metric_val in zip(param_values, metric_values):
                    bin_idx = min(int((param_val - min(param_values)) / bin_size), num_bins - 1)
                    bins[bin_idx].append(metric_val)
                
                # Calculate between-group and within-group variance
                total_variance = np.var(metric_values)
                bin_means = [np.mean(values) for values in bins.values() if values]
                bin_sizes = [len(values) for values in bins.values() if values]
                
                if bin_means and total_variance > 0:
                    overall_mean = np.mean(metric_values)
                    between_variance = sum(size * (mean - overall_mean) ** 2 
                                         for size, mean in zip(bin_sizes, bin_means)) / len(metric_values)
                    variance_contribution = between_variance / total_variance
                else:
                    variance_contribution = 0.0
            else:
                variance_contribution = 0.0
        else:
            variance_contribution = 0.0
        
        # Bootstrap confidence interval
        confidence_interval = self._calculate_confidence_interval(
            param_values, metric_values, sensitivity_index
        )
        
        return sensitivity_index, confidence_interval, variance_contribution
    
    def _calculate_confidence_interval(self, param_values: List[float], 
                                     metric_values: List[float], 
                                     sensitivity_index: float) -> Tuple[float, float]:
        """Calculate confidence interval using bootstrap"""
        bootstrap_sensitivities = []
        n_samples = len(param_values)
        
        for _ in range(self.config['bootstrap_samples']):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_param_values = [param_values[i] for i in indices]
            boot_metric_values = [metric_values[i] for i in indices]
            
            # Calculate sensitivity for bootstrap sample
            correlation = np.corrcoef(boot_param_values, boot_metric_values)[0, 1]
            boot_sensitivity = abs(correlation) if not np.isnan(correlation) else 0.0
            bootstrap_sensitivities.append(boot_sensitivity)
        
        # Calculate confidence interval
        alpha = 1 - self.config['confidence_level']
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_sensitivities, lower_percentile)
        upper_bound = np.percentile(bootstrap_sensitivities, upper_percentile)
        
        return (lower_bound, upper_bound)
    
    def _calculate_interaction_effects(self, sensitivity_results: Dict[str, SensitivityResult],
                                     samples: List[ParameterSample],
                                     parameters: Dict[str, ParameterRange]) -> Dict[str, SensitivityResult]:
        """Calculate pairwise interaction effects"""
        param_names = list(parameters.keys())
        
        for metric in self.config['performance_metrics']:
            valid_samples = [s for s in samples if s.success]
            
            # Calculate pairwise interactions
            for i, param1 in enumerate(param_names):
                for j, param2 in enumerate(param_names[i+1:], i+1):
                    interaction_effect = self._calculate_pairwise_interaction(
                        valid_samples, param1, param2, metric
                    )
                    
                    # Store interaction effect in both parameters' results
                    key1 = f"{param1}_{metric}"
                    key2 = f"{param2}_{metric}"
                    
                    if key1 in sensitivity_results:
                        sensitivity_results[key1].interaction_effects[param2] = interaction_effect
                    
                    if key2 in sensitivity_results:
                        sensitivity_results[key2].interaction_effects[param1] = interaction_effect
        
        return sensitivity_results
    
    def _calculate_pairwise_interaction(self, samples: List[ParameterSample],
                                      param1: str, param2: str, metric: str) -> float:
        """Calculate interaction effect between two parameters"""
        # Extract parameter and metric values
        param1_values = [s.parameters[param1] for s in samples]
        param2_values = [s.parameters[param2] for s in samples]
        metric_values = [s.performance_metrics[metric] for s in samples]
        
        # Discretize parameters into high/low bins
        param1_median = np.median(param1_values)
        param2_median = np.median(param2_values)
        
        # Create interaction groups
        groups = {
            'low_low': [],
            'low_high': [],
            'high_low': [],
            'high_high': []
        }
        
        for p1, p2, m in zip(param1_values, param2_values, metric_values):
            if p1 <= param1_median and p2 <= param2_median:
                groups['low_low'].append(m)
            elif p1 <= param1_median and p2 > param2_median:
                groups['low_high'].append(m)
            elif p1 > param1_median and p2 <= param2_median:
                groups['high_low'].append(m)
            else:
                groups['high_high'].append(m)
        
        # Calculate interaction effect
        try:
            mean_ll = np.mean(groups['low_low']) if groups['low_low'] else 0
            mean_lh = np.mean(groups['low_high']) if groups['low_high'] else 0
            mean_hl = np.mean(groups['high_low']) if groups['high_low'] else 0
            mean_hh = np.mean(groups['high_high']) if groups['high_high'] else 0
            
            # Interaction effect: (high_high - high_low) - (low_high - low_low)
            interaction_effect = (mean_hh - mean_hl) - (mean_lh - mean_ll)
            
            # Normalize by overall variance
            overall_variance = np.var(metric_values)
            if overall_variance > 0:
                interaction_effect = abs(interaction_effect) / math.sqrt(overall_variance)
            else:
                interaction_effect = 0.0
            
        except Exception:
            interaction_effect = 0.0
        
        return interaction_effect
    
    def _assign_significance_and_ranking(self, sensitivity_results: Dict[str, SensitivityResult]) -> Dict[str, SensitivityResult]:
        """Assign statistical significance and rankings"""
        # Group by metric
        metric_groups = defaultdict(list)
        for key, result in sensitivity_results.items():
            metric = key.split('_')[-1]
            metric_groups[metric].append((key, result))
        
        # Assign rankings within each metric group
        for metric, group in metric_groups.items():
            # Sort by sensitivity index
            sorted_group = sorted(group, key=lambda x: x[1].sensitivity_index, reverse=True)
            
            for rank, (key, result) in enumerate(sorted_group):
                result.rank = rank + 1
                
                # Statistical significance based on confidence interval
                lower_bound, upper_bound = result.confidence_interval
                result.statistical_significance = lower_bound > self.config['significance_threshold']
                
                sensitivity_results[key] = result
        
        return sensitivity_results
    
    def generate_tuning_recommendations(self, 
                                      current_parameters: Dict[str, float],
                                      objective: RouteObjective) -> List[TuningRecommendation]:
        """Generate parameter tuning recommendations based on sensitivity analysis
        
        Args:
            current_parameters: Current parameter configuration
            objective: Optimization objective
            
        Returns:
            List of tuning recommendations
        """
        recommendations = []
        
        if not self.sensitivity_results:
            return [TuningRecommendation(
                parameter_name="analysis_needed",
                current_value=0.0,
                recommended_value=0.0,
                confidence=0.0,
                rationale="No sensitivity analysis results available",
                expected_improvement=0.0,
                priority="high",
                sensitivity_rank=0
            )]
        
        # Determine primary metric based on objective
        if objective == RouteObjective.MAXIMIZE_ELEVATION:
            primary_metric = 'fitness'  # Assuming fitness captures elevation
        elif objective == RouteObjective.MINIMIZE_DISTANCE:
            primary_metric = 'fitness'
        else:
            primary_metric = 'fitness'
        
        # Analyze each parameter
        for param_name, current_value in current_parameters.items():
            result_key = f"{param_name}_{primary_metric}"
            
            if result_key in self.sensitivity_results:
                result = self.sensitivity_results[result_key]
                
                # Generate recommendation based on sensitivity
                recommendation = self._generate_parameter_recommendation(
                    param_name, current_value, result, objective
                )
                
                if recommendation:
                    recommendations.append(recommendation)
        
        # Sort by priority and expected improvement
        recommendations.sort(key=lambda x: (
            {'high': 3, 'medium': 2, 'low': 1}[x.priority],
            x.expected_improvement
        ), reverse=True)
        
        self.recommendations = recommendations
        return recommendations
    
    def _generate_parameter_recommendation(self, param_name: str, current_value: float,
                                         sensitivity_result: SensitivityResult,
                                         objective: RouteObjective) -> Optional[TuningRecommendation]:
        """Generate recommendation for specific parameter"""
        if param_name not in self.parameter_ranges:
            return None
        
        param_range = self.parameter_ranges[param_name]
        sensitivity = sensitivity_result.sensitivity_index
        
        # Skip parameters with low sensitivity
        if sensitivity < 0.1:
            return None
        
        # Determine recommendation direction and magnitude
        if objective == RouteObjective.MAXIMIZE_ELEVATION:
            # For elevation maximization
            if param_name == 'mutation_rate':
                # Higher mutation for more exploration
                direction = 1 if current_value < param_range.max_value * 0.7 else 0
            elif param_name == 'population_size':
                # Larger population for better exploration
                direction = 1 if current_value < param_range.max_value * 0.8 else 0
            elif param_name == 'elevation_weight':
                # Higher elevation weight
                direction = 1
            else:
                # Default: move toward center if at extremes
                center = (param_range.min_value + param_range.max_value) / 2
                direction = 1 if current_value < center else -1
        
        elif objective == RouteObjective.MINIMIZE_DISTANCE:
            # For distance minimization
            if param_name == 'crossover_rate':
                # Higher crossover for exploitation
                direction = 1 if current_value < param_range.max_value * 0.9 else 0
            elif param_name == 'elite_size_ratio':
                # More elites for convergence
                direction = 1 if current_value < param_range.max_value * 0.8 else 0
            elif param_name == 'elevation_weight':
                # Lower elevation weight
                direction = -1
            else:
                # Default recommendation
                center = (param_range.min_value + param_range.max_value) / 2
                direction = 1 if current_value < center else -1
        
        else:  # BALANCED_ROUTE
            # Move toward optimal ranges
            if param_name == 'mutation_rate':
                optimal = 0.1
                direction = 1 if current_value < optimal else -1
            elif param_name == 'crossover_rate':
                optimal = 0.8
                direction = 1 if current_value < optimal else -1
            else:
                center = (param_range.min_value + param_range.max_value) / 2
                direction = 1 if current_value < center else -1
        
        if direction == 0:
            return None  # No change recommended
        
        # Calculate recommended value
        adjustment_factor = sensitivity * 0.2  # Adjust based on sensitivity
        range_size = param_range.max_value - param_range.min_value
        adjustment = direction * adjustment_factor * range_size
        
        recommended_value = param_range.clamp(current_value + adjustment)
        
        # Skip if change is too small
        if abs(recommended_value - current_value) < param_range.step_size:
            return None
        
        # Determine priority
        if sensitivity > 0.7:
            priority = "high"
        elif sensitivity > 0.3:
            priority = "medium"
        else:
            priority = "low"
        
        # Generate rationale
        rationale = f"High sensitivity ({sensitivity:.3f}) suggests this parameter significantly affects performance. "
        if sensitivity_result.statistical_significance:
            rationale += "Change is statistically significant. "
        
        if sensitivity_result.interaction_effects:
            top_interaction = max(sensitivity_result.interaction_effects.items(), 
                                key=lambda x: abs(x[1]))
            rationale += f"Strong interaction with {top_interaction[0]} ({top_interaction[1]:.3f}). "
        
        # Expected improvement (heuristic)
        expected_improvement = sensitivity * abs(recommended_value - current_value) / range_size
        
        return TuningRecommendation(
            parameter_name=param_name,
            current_value=current_value,
            recommended_value=recommended_value,
            confidence=sensitivity * 0.8 + (0.2 if sensitivity_result.statistical_significance else 0.0),
            rationale=rationale.strip(),
            expected_improvement=expected_improvement,
            priority=priority,
            sensitivity_rank=sensitivity_result.rank
        )
    
    def generate_optimization_insights(self) -> List[OptimizationInsight]:
        """Generate high-level optimization insights"""
        insights = []
        
        if not self.sensitivity_results:
            return insights
        
        # Group results by parameter
        param_sensitivities = defaultdict(list)
        for key, result in self.sensitivity_results.items():
            param_name = '_'.join(key.split('_')[:-1])
            param_sensitivities[param_name].append(result)
        
        # Insight 1: Most influential parameters
        avg_sensitivities = {}
        for param_name, results in param_sensitivities.items():
            avg_sensitivity = np.mean([r.sensitivity_index for r in results])
            avg_sensitivities[param_name] = avg_sensitivity
        
        top_params = sorted(avg_sensitivities.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_params:
            insights.append(OptimizationInsight(
                insight_type="high_impact_parameters",
                description=f"Parameters with highest impact: {', '.join([p[0] for p in top_params])}",
                affected_parameters=[p[0] for p in top_params],
                confidence=np.mean([p[1] for p in top_params]),
                actionable_recommendation="Focus tuning efforts on these high-impact parameters first"
            ))
        
        # Insight 2: Parameter interactions
        interaction_strengths = []
        for param_name, results in param_sensitivities.items():
            for result in results:
                for other_param, interaction in result.interaction_effects.items():
                    if abs(interaction) > self.config['interaction_threshold']:
                        interaction_strengths.append((param_name, other_param, interaction))
        
        if interaction_strengths:
            strongest_interaction = max(interaction_strengths, key=lambda x: abs(x[2]))
            insights.append(OptimizationInsight(
                insight_type="parameter_interactions",
                description=f"Strong interaction between {strongest_interaction[0]} and {strongest_interaction[1]}",
                affected_parameters=[strongest_interaction[0], strongest_interaction[1]],
                confidence=abs(strongest_interaction[2]),
                actionable_recommendation="Tune these parameters together rather than independently"
            ))
        
        # Insight 3: Variance contribution analysis
        high_variance_params = []
        for param_name, results in param_sensitivities.items():
            avg_variance_contrib = np.mean([r.variance_contribution for r in results])
            if avg_variance_contrib > 0.3:  # 30% threshold
                high_variance_params.append((param_name, avg_variance_contrib))
        
        if high_variance_params:
            insights.append(OptimizationInsight(
                insight_type="variance_drivers",
                description="Parameters that drive most performance variance",
                affected_parameters=[p[0] for p in high_variance_params],
                confidence=np.mean([p[1] for p in high_variance_params]),
                actionable_recommendation="These parameters control performance variability - ensure they are properly set"
            ))
        
        self.insights = insights
        return insights
    
    def save_analysis_results(self, filename: str) -> str:
        """Save complete analysis results to file
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if not self.config['save_analysis_results']:
            return ""
        
        analysis_data = {
            'config': self.config,
            'parameter_ranges': {
                name: {
                    'min_value': pr.min_value,
                    'max_value': pr.max_value,
                    'default_value': pr.default_value,
                    'constraint_type': pr.constraint_type
                } for name, pr in self.parameter_ranges.items()
            },
            'sensitivity_results': {
                key: {
                    'parameter_name': result.parameter_name,
                    'sensitivity_index': result.sensitivity_index,
                    'confidence_interval': result.confidence_interval,
                    'variance_contribution': result.variance_contribution,
                    'interaction_effects': result.interaction_effects,
                    'statistical_significance': result.statistical_significance,
                    'rank': result.rank
                } for key, result in self.sensitivity_results.items()
            },
            'recommendations': [
                {
                    'parameter_name': rec.parameter_name,
                    'current_value': rec.current_value,
                    'recommended_value': rec.recommended_value,
                    'confidence': rec.confidence,
                    'rationale': rec.rationale,
                    'expected_improvement': rec.expected_improvement,
                    'priority': rec.priority,
                    'sensitivity_rank': rec.sensitivity_rank
                } for rec in self.recommendations
            ],
            'insights': [
                {
                    'insight_type': insight.insight_type,
                    'description': insight.description,
                    'affected_parameters': insight.affected_parameters,
                    'confidence': insight.confidence,
                    'actionable_recommendation': insight.actionable_recommendation
                } for insight in self.insights
            ],
            'analysis_timestamp': time.time()
        }
        
        with open(filename, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        return filename


def test_sensitivity_analyzer():
    """Test function for sensitivity analyzer"""
    print("Testing GA Sensitivity Analyzer...")
    
    # Create analyzer
    analyzer = GASensitivityAnalyzer({
        'sampling_budget': 100,  # Small budget for testing
        'sensitivity_method': SensitivityMethod.LATIN_HYPERCUBE
    })
    
    # Set up parameter ranges
    from ga_parameter_tuner import ParameterRange
    
    param_ranges = {
        'population_size': ParameterRange(50, 200, 100, constraint_type='integer'),
        'mutation_rate': ParameterRange(0.01, 0.3, 0.1),
        'crossover_rate': ParameterRange(0.5, 0.95, 0.8),
        'elite_size_ratio': ParameterRange(0.05, 0.3, 0.1)
    }
    
    analyzer.set_parameter_ranges(param_ranges)
    
    # Mock performance evaluator
    def mock_evaluator(params):
        # Simple mock function with some parameter relationships
        fitness = (
            0.5 + 0.3 * (params['mutation_rate'] / 0.3) +
            0.2 * (params['crossover_rate'] / 0.95) +
            0.1 * np.random.normal(0, 0.1)  # Add noise
        )
        return {
            'fitness': max(0.0, min(1.0, fitness)),
            'convergence_speed': 0.5 + 0.3 * np.random.random(),
            'diversity': 0.3 + 0.4 * (params['population_size'] / 200)
        }
    
    # Test sensitivity analysis
    sensitivity_results = analyzer.analyze_parameter_sensitivity(mock_evaluator)
    print(f"✅ Sensitivity analysis: {len(sensitivity_results)} results")
    
    # Test tuning recommendations
    current_params = {
        'population_size': 100,
        'mutation_rate': 0.1,
        'crossover_rate': 0.8,
        'elite_size_ratio': 0.1
    }
    
    recommendations = analyzer.generate_tuning_recommendations(
        current_params, RouteObjective.MAXIMIZE_ELEVATION
    )
    print(f"✅ Tuning recommendations: {len(recommendations)} recommendations")
    
    # Test optimization insights
    insights = analyzer.generate_optimization_insights()
    print(f"✅ Optimization insights: {len(insights)} insights")
    
    # Test results saving
    saved_file = analyzer.save_analysis_results("test_sensitivity_results.json")
    print(f"✅ Results saved: {saved_file}")
    
    print("✅ All sensitivity analyzer tests completed")


if __name__ == "__main__":
    test_sensitivity_analyzer()