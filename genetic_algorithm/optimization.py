#!/usr/bin/env python3
"""
Genetic Algorithm Optimization Components
Consolidated hyperparameter optimization, parameter tuning, and algorithm selection
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import math
import numpy as np
import random
import json
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from ga_common_imports import (
    GAConfiguration, GAStatistics, GAPerformanceMonitor,
    get_logger, validate_graph, normalize_fitness
)


class OptimizationMethod(Enum):
    """Hyperparameter optimization methods"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    PARTICLE_SWARM = "particle_swarm"
    SIMULATED_ANNEALING = "simulated_annealing"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"


class AdaptationStrategy(Enum):
    """Parameter adaptation strategies"""
    INCREASE_ON_STAGNATION = "increase_on_stagnation"
    DECREASE_ON_STAGNATION = "decrease_on_stagnation"
    OSCILLATE = "oscillate"
    PERFORMANCE_BASED = "performance_based"
    DIVERSITY_BASED = "diversity_based"
    CONVERGENCE_BASED = "convergence_based"
    HYBRID = "hybrid"


@dataclass
class ParameterRange:
    """Parameter range definition"""
    min_value: float
    max_value: float
    step_size: Optional[float] = None
    values: Optional[List[float]] = None
    parameter_type: str = "continuous"  # continuous, discrete, categorical
    
    def sample(self) -> float:
        """Sample a value from the range"""
        if self.values:
            return random.choice(self.values)
        elif self.parameter_type == "discrete":
            return random.randint(int(self.min_value), int(self.max_value))
        else:
            return random.uniform(self.min_value, self.max_value)


@dataclass
class HyperparameterSpace:
    """Definition of hyperparameter search space"""
    parameter_ranges: Dict[str, ParameterRange]
    constraints: List[Callable[[Dict[str, float]], bool]] = field(default_factory=list)
    objectives: List[str] = field(default_factory=lambda: ['fitness', 'convergence_speed'])
    weights: Dict[str, float] = field(default_factory=lambda: {'fitness': 0.7, 'convergence_speed': 0.3})


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization"""
    best_parameters: Dict[str, float]
    best_score: float
    evaluation_history: List[Dict[str, Any]]
    optimization_time: float
    total_evaluations: int
    convergence_generation: int
    method_used: OptimizationMethod
    objective_values: Dict[str, float]


@dataclass
class AdaptationRule:
    """Parameter adaptation rule"""
    parameter_name: str
    strategy: AdaptationStrategy
    trigger_condition: Callable[[GAStatistics], bool]
    adaptation_amount: float
    cooldown_generations: int = 5
    last_adjustment_generation: int = 0
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class GAHyperparameterOptimizer:
    """Automated hyperparameter optimization for genetic algorithms"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize hyperparameter optimizer
        
        Args:
            config: Configuration options for optimization
        """
        default_config = {
            'max_evaluations': 100,
            'evaluation_timeout': 300,
            'parallel_evaluations': 4,
            'early_stopping_patience': 20,
            'early_stopping_threshold': 0.01,
            'validation_splits': 3,
            'optimization_budget_minutes': 60,
            'save_intermediate_results': True,
            'random_seed': 42,
            'objective_weights': {
                'fitness': 0.7,
                'convergence_speed': 0.2,
                'diversity': 0.1
            }
        }
        
        self.config = {**default_config, **(config or {})}
        self.logger = get_logger(__name__)
        
        # Set random seed
        random.seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])
        
        # Optimization state
        self.best_score = float('-inf')
        self.best_parameters = {}
        self.evaluation_history = []
        self.optimization_start_time = None
        
    def optimize(self, hyperparameter_space: HyperparameterSpace,
                method: OptimizationMethod = OptimizationMethod.RANDOM_SEARCH,
                objective_function: Optional[Callable] = None) -> OptimizationResult:
        """Optimize hyperparameters using specified method
        
        Args:
            hyperparameter_space: Search space definition
            method: Optimization method to use
            objective_function: Custom objective function
            
        Returns:
            Optimization result
        """
        self.optimization_start_time = time.time()
        
        if method == OptimizationMethod.GRID_SEARCH:
            result = self._grid_search(hyperparameter_space, objective_function)
        elif method == OptimizationMethod.RANDOM_SEARCH:
            result = self._random_search(hyperparameter_space, objective_function)
        elif method == OptimizationMethod.BAYESIAN:
            result = self._bayesian_optimization(hyperparameter_space, objective_function)
        else:
            raise ValueError(f"Optimization method {method} not implemented")
        
        return result
    
    def _grid_search(self, space: HyperparameterSpace, 
                    objective_function: Optional[Callable]) -> OptimizationResult:
        """Grid search optimization"""
        # Generate parameter combinations
        parameter_combinations = []
        
        # Create parameter grids
        parameter_grids = {}
        for param_name, param_range in space.parameter_ranges.items():
            if param_range.values:
                parameter_grids[param_name] = param_range.values
            else:
                # Generate grid points
                if param_range.step_size:
                    steps = np.arange(param_range.min_value, 
                                    param_range.max_value + param_range.step_size,
                                    param_range.step_size)
                else:
                    steps = np.linspace(param_range.min_value, param_range.max_value, 10)
                parameter_grids[param_name] = steps.tolist()
        
        # Generate all combinations
        param_names = list(parameter_grids.keys())
        param_values = list(parameter_grids.values())
        
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            
            # Check constraints
            if all(constraint(params) for constraint in space.constraints):
                parameter_combinations.append(params)
        
        # Limit combinations if too many
        if len(parameter_combinations) > self.config['max_evaluations']:
            parameter_combinations = random.sample(
                parameter_combinations, self.config['max_evaluations'])
        
        # Evaluate combinations
        best_score = float('-inf')
        best_params = {}
        
        for i, params in enumerate(parameter_combinations):
            if self._should_stop():
                break
            
            score = self._evaluate_parameters(params, objective_function)
            
            if score > best_score:
                best_score = score
                best_params = params
            
            self.logger.info(f"Grid search iteration {i+1}/{len(parameter_combinations)}: "
                           f"score={score:.3f}, best={best_score:.3f}")
        
        return OptimizationResult(
            best_parameters=best_params,
            best_score=best_score,
            evaluation_history=self.evaluation_history,
            optimization_time=time.time() - self.optimization_start_time,
            total_evaluations=len(self.evaluation_history),
            convergence_generation=len(self.evaluation_history),
            method_used=OptimizationMethod.GRID_SEARCH,
            objective_values={'fitness': best_score}
        )
    
    def _random_search(self, space: HyperparameterSpace,
                      objective_function: Optional[Callable]) -> OptimizationResult:
        """Random search optimization"""
        best_score = float('-inf')
        best_params = {}
        
        for i in range(self.config['max_evaluations']):
            if self._should_stop():
                break
            
            # Sample parameters
            params = {}
            for param_name, param_range in space.parameter_ranges.items():
                params[param_name] = param_range.sample()
            
            # Check constraints
            if not all(constraint(params) for constraint in space.constraints):
                continue
            
            # Evaluate parameters
            score = self._evaluate_parameters(params, objective_function)
            
            if score > best_score:
                best_score = score
                best_params = params
            
            self.logger.info(f"Random search iteration {i+1}: "
                           f"score={score:.3f}, best={best_score:.3f}")
        
        return OptimizationResult(
            best_parameters=best_params,
            best_score=best_score,
            evaluation_history=self.evaluation_history,
            optimization_time=time.time() - self.optimization_start_time,
            total_evaluations=len(self.evaluation_history),
            convergence_generation=len(self.evaluation_history),
            method_used=OptimizationMethod.RANDOM_SEARCH,
            objective_values={'fitness': best_score}
        )
    
    def _bayesian_optimization(self, space: HyperparameterSpace,
                             objective_function: Optional[Callable]) -> OptimizationResult:
        """Bayesian optimization (simplified implementation)"""
        # For now, fall back to random search
        # In a full implementation, this would use libraries like scikit-optimize
        self.logger.warning("Bayesian optimization not fully implemented, using random search")
        return self._random_search(space, objective_function)
    
    def _evaluate_parameters(self, params: Dict[str, float], 
                           objective_function: Optional[Callable]) -> float:
        """Evaluate a set of parameters"""
        start_time = time.time()
        
        try:
            if objective_function:
                score = objective_function(params)
            else:
                # Default evaluation: simulate GA run
                score = self._simulate_ga_run(params)
            
            evaluation_time = time.time() - start_time
            
            # Record evaluation
            self.evaluation_history.append({
                'parameters': params.copy(),
                'score': score,
                'evaluation_time': evaluation_time,
                'timestamp': time.time()
            })
            
            return score
            
        except Exception as e:
            self.logger.error(f"Parameter evaluation failed: {e}")
            return float('-inf')
    
    def _simulate_ga_run(self, params: Dict[str, float]) -> float:
        """Simulate GA run with given parameters"""
        # Simplified simulation - in real implementation, would run actual GA
        # For now, return a score based on parameter values
        
        # Simulate some parameter interactions
        population_size = params.get('population_size', 100)
        mutation_rate = params.get('mutation_rate', 0.1)
        crossover_rate = params.get('crossover_rate', 0.8)
        
        # Simple heuristic scoring
        score = 0.0
        
        # Population size: optimal around 100-200
        if 50 <= population_size <= 200:
            score += 0.3
        
        # Mutation rate: optimal around 0.05-0.15
        if 0.05 <= mutation_rate <= 0.15:
            score += 0.3
        
        # Crossover rate: optimal around 0.7-0.9
        if 0.7 <= crossover_rate <= 0.9:
            score += 0.4
        
        # Add random noise
        score += random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, score))
    
    def _should_stop(self) -> bool:
        """Check if optimization should stop"""
        if self.optimization_start_time is None:
            return False
        
        elapsed_time = time.time() - self.optimization_start_time
        elapsed_minutes = elapsed_time / 60
        
        return elapsed_minutes >= self.config['optimization_budget_minutes']


class GAParameterTuner:
    """Dynamic parameter tuning for genetic algorithms"""
    
    def __init__(self, initial_parameters: Optional[Dict[str, float]] = None):
        """Initialize parameter tuner
        
        Args:
            initial_parameters: Initial parameter values
        """
        self.initial_parameters = initial_parameters or {
            'population_size': 100,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'elite_size': 2,
            'tournament_size': 3
        }
        
        self.current_parameters = self.initial_parameters.copy()
        self.adaptation_rules = []
        self.adaptation_history = []
        self.logger = get_logger(__name__)
        
        # Setup default adaptation rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default adaptation rules"""
        # Increase mutation rate when fitness stagnates
        self.adaptation_rules.append(AdaptationRule(
            parameter_name='mutation_rate',
            strategy=AdaptationStrategy.INCREASE_ON_STAGNATION,
            trigger_condition=lambda stats: self._is_fitness_stagnant(stats),
            adaptation_amount=0.02,
            min_value=0.05,
            max_value=0.3
        ))
        
        # Decrease population size if convergence is too slow
        self.adaptation_rules.append(AdaptationRule(
            parameter_name='population_size',
            strategy=AdaptationStrategy.DECREASE_ON_STAGNATION,
            trigger_condition=lambda stats: self._is_convergence_slow(stats),
            adaptation_amount=10,
            min_value=50,
            max_value=200
        ))
    
    def adapt_parameters(self, population_stats: GAStatistics) -> Dict[str, float]:
        """Adapt parameters based on population statistics
        
        Args:
            population_stats: Current population statistics
            
        Returns:
            Updated parameters
        """
        parameter_changes = {}
        adaptation_reasons = []
        
        for rule in self.adaptation_rules:
            if (rule.trigger_condition(population_stats) and 
                population_stats.generation - rule.last_adjustment_generation >= rule.cooldown_generations):
                
                current_value = self.current_parameters.get(rule.parameter_name, 0)
                
                if rule.strategy == AdaptationStrategy.INCREASE_ON_STAGNATION:
                    new_value = current_value + rule.adaptation_amount
                elif rule.strategy == AdaptationStrategy.DECREASE_ON_STAGNATION:
                    new_value = current_value - rule.adaptation_amount
                else:
                    new_value = current_value
                
                # Apply bounds
                if rule.min_value is not None:
                    new_value = max(new_value, rule.min_value)
                if rule.max_value is not None:
                    new_value = min(new_value, rule.max_value)
                
                if new_value != current_value:
                    parameter_changes[rule.parameter_name] = new_value
                    adaptation_reasons.append(f"{rule.parameter_name}: {rule.strategy.value}")
                    rule.last_adjustment_generation = population_stats.generation
        
        # Update parameters
        if parameter_changes:
            self.current_parameters.update(parameter_changes)
            self._record_adaptation(population_stats.generation, parameter_changes, 
                                  {}, ', '.join(adaptation_reasons))
        
        return self.current_parameters.copy()
    
    def _is_fitness_stagnant(self, stats: GAStatistics) -> bool:
        """Check if fitness has stagnated"""
        if len(self.adaptation_history) < 5:
            return False
        
        # Check if fitness improvement is below threshold
        recent_fitness = [h['best_fitness'] for h in self.adaptation_history[-5:]]
        fitness_improvement = max(recent_fitness) - min(recent_fitness)
        
        return fitness_improvement < 0.01
    
    def _is_convergence_slow(self, stats: GAStatistics) -> bool:
        """Check if convergence is slow"""
        return stats.generation > 50 and stats.diversity_score > 0.8
    
    def _record_adaptation(self, generation: int, changes: Dict[str, float],
                          metrics: Dict[str, float], reason: str):
        """Record parameter adaptation"""
        self.adaptation_history.append({
            'generation': generation,
            'parameter_changes': changes.copy(),
            'metrics': metrics.copy(),
            'reason': reason,
            'timestamp': time.time()
        })
        
        if changes:
            self.logger.info(f"Generation {generation}: Adapted parameters - {reason}")


class GAAlgorithmSelector:
    """Intelligent algorithm selection for genetic algorithms"""
    
    def __init__(self):
        """Initialize algorithm selector"""
        self.performance_history = {}
        self.algorithm_preferences = {
            'small_population': ['elitism', 'tournament'],
            'large_population': ['roulette', 'rank'],
            'high_diversity': ['tournament', 'diversity'],
            'low_diversity': ['elitism', 'uniform']
        }
        self.logger = get_logger(__name__)
    
    def select_algorithm(self, problem_characteristics: Dict[str, Any]) -> str:
        """Select best algorithm based on problem characteristics
        
        Args:
            problem_characteristics: Problem characteristics
            
        Returns:
            Selected algorithm name
        """
        # Analyze problem characteristics
        population_size = problem_characteristics.get('population_size', 100)
        diversity_score = problem_characteristics.get('diversity_score', 0.5)
        
        # Select based on characteristics
        if population_size < 50:
            category = 'small_population'
        elif population_size > 200:
            category = 'large_population'
        elif diversity_score > 0.7:
            category = 'high_diversity'
        else:
            category = 'low_diversity'
        
        # Get preferred algorithms
        preferred_algorithms = self.algorithm_preferences.get(category, ['tournament'])
        
        # Select best performing algorithm from preferred list
        best_algorithm = preferred_algorithms[0]
        best_score = self.performance_history.get(best_algorithm, 0.0)
        
        for algorithm in preferred_algorithms:
            score = self.performance_history.get(algorithm, 0.0)
            if score > best_score:
                best_score = score
                best_algorithm = algorithm
        
        self.logger.info(f"Selected algorithm: {best_algorithm} (category: {category})")
        return best_algorithm
    
    def update_performance(self, algorithm: str, performance_score: float):
        """Update algorithm performance history
        
        Args:
            algorithm: Algorithm name
            performance_score: Performance score
        """
        if algorithm not in self.performance_history:
            self.performance_history[algorithm] = performance_score
        else:
            # Exponential moving average
            alpha = 0.1
            self.performance_history[algorithm] = (
                alpha * performance_score + 
                (1 - alpha) * self.performance_history[algorithm]
            )


# Export main classes
__all__ = [
    'GAHyperparameterOptimizer',
    'GAParameterTuner',
    'GAAlgorithmSelector',
    'OptimizationMethod',
    'AdaptationStrategy',
    'HyperparameterSpace',
    'OptimizationResult',
    'ParameterRange',
    'AdaptationRule'
]