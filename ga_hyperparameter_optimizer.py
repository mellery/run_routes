#!/usr/bin/env python3
"""
GA Hyperparameter Optimization Framework
Automated hyperparameter tuning using optimization algorithms
"""

import time
import math
import numpy as np
import random
import json
import os
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from ga_parameter_tuner import GAParameterTuner, ParameterRange
from route_objective import RouteObjective


class OptimizationMethod(Enum):
    """Hyperparameter optimization methods"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    PARTICLE_SWARM = "particle_swarm"
    SIMULATED_ANNEALING = "simulated_annealing"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"


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
class EvaluationTask:
    """Individual parameter evaluation task"""
    task_id: int
    parameters: Dict[str, float]
    test_config: Dict[str, Any]
    objective: RouteObjective
    target_distance: float


@dataclass
class EvaluationResult:
    """Result of parameter evaluation"""
    task_id: int
    parameters: Dict[str, float]
    performance_score: float
    objective_scores: Dict[str, float]
    execution_time: float
    convergence_generation: int
    final_fitness: float
    success: bool
    error_message: Optional[str] = None


class GAHyperparameterOptimizer:
    """Automated hyperparameter optimization for genetic algorithms"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize hyperparameter optimizer
        
        Args:
            config: Configuration options for optimization
        """
        default_config = {
            'max_evaluations': 100,          # Maximum parameter evaluations
            'evaluation_timeout': 300,       # Timeout per evaluation (seconds)
            'parallel_evaluations': 4,       # Number of parallel evaluations
            'early_stopping_patience': 20,   # Early stopping patience
            'early_stopping_threshold': 0.01, # Minimum improvement threshold
            'validation_splits': 3,          # Cross-validation splits
            'optimization_budget_minutes': 60, # Total optimization time budget
            'save_intermediate_results': True, # Save results during optimization
            'random_seed': 42,               # Random seed for reproducibility
            'objective_weights': {           # Weights for multi-objective optimization
                'fitness': 0.4,
                'convergence_speed': 0.3,
                'stability': 0.2,
                'diversity': 0.1
            }
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Set random seed
        random.seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])
        
        # Initialize hyperparameter space
        self.hyperparameter_space = self._define_hyperparameter_space()
        
        # Optimization state
        self.evaluation_history = []
        self.best_result = None
        self.optimization_start_time = None
        self.total_evaluations = 0
        
        # Statistics tracking
        self.method_performance = {}
        self.convergence_tracking = []
        
        print(f"🔬 GA Hyperparameter Optimizer initialized with {self.config['max_evaluations']} max evaluations")
    
    def _define_hyperparameter_space(self) -> HyperparameterSpace:
        """Define hyperparameter search space"""
        parameter_ranges = {
            'population_size': ParameterRange(50, 300, 100, constraint_type='integer'),
            'mutation_rate': ParameterRange(0.01, 0.3, 0.1),
            'crossover_rate': ParameterRange(0.5, 0.95, 0.8),
            'elite_size_ratio': ParameterRange(0.05, 0.25, 0.1),
            'tournament_size': ParameterRange(3, 15, 5, constraint_type='integer'),
            'max_generations': ParameterRange(100, 500, 200, constraint_type='integer'),
            'diversity_threshold': ParameterRange(0.1, 0.7, 0.3),
            'selection_pressure': ParameterRange(1.5, 4.0, 2.0),
            'fitness_scaling_factor': ParameterRange(1.0, 5.0, 2.0),
            'early_stopping_patience': ParameterRange(20, 80, 50, constraint_type='integer'),
            'distance_tolerance': ParameterRange(0.1, 0.4, 0.2),
            'elevation_weight': ParameterRange(0.5, 2.5, 1.0)
        }
        
        # Define parameter constraints
        constraints = [
            # Elite size should be reasonable fraction of population
            lambda params: params['elite_size_ratio'] * params['population_size'] >= 2,
            # Tournament size should be reasonable fraction of population  
            lambda params: params['tournament_size'] <= params['population_size'] * 0.3,
            # Mutation and crossover rates should sum to reasonable range
            lambda params: 0.6 <= params['mutation_rate'] + params['crossover_rate'] <= 1.2,
            # Early stopping patience should be reasonable fraction of max generations
            lambda params: params['early_stopping_patience'] <= params['max_generations'] * 0.5
        ]
        
        return HyperparameterSpace(
            parameter_ranges=parameter_ranges,
            constraints=constraints,
            objectives=['fitness', 'convergence_speed', 'stability', 'diversity'],
            weights=self.config['objective_weights']
        )
    
    def optimize_hyperparameters(self, method: OptimizationMethod,
                                test_scenarios: List[Dict[str, Any]]) -> OptimizationResult:
        """Optimize hyperparameters using specified method
        
        Args:
            method: Optimization method to use
            test_scenarios: List of test scenarios for evaluation
            
        Returns:
            Optimization result with best parameters
        """
        print(f"🔬 Starting hyperparameter optimization using {method.value}")
        self.optimization_start_time = time.time()
        
        # Reset state
        self.evaluation_history = []
        self.best_result = None
        self.total_evaluations = 0
        
        # Run optimization based on method
        if method == OptimizationMethod.GRID_SEARCH:
            result = self._grid_search_optimization(test_scenarios)
        elif method == OptimizationMethod.RANDOM_SEARCH:
            result = self._random_search_optimization(test_scenarios)
        elif method == OptimizationMethod.BAYESIAN:
            result = self._bayesian_optimization(test_scenarios)
        elif method == OptimizationMethod.GENETIC:
            result = self._genetic_optimization(test_scenarios)
        elif method == OptimizationMethod.PARTICLE_SWARM:
            result = self._particle_swarm_optimization(test_scenarios)
        elif method == OptimizationMethod.SIMULATED_ANNEALING:
            result = self._simulated_annealing_optimization(test_scenarios)
        elif method == OptimizationMethod.DIFFERENTIAL_EVOLUTION:
            result = self._differential_evolution_optimization(test_scenarios)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        optimization_time = time.time() - self.optimization_start_time
        result.optimization_time = optimization_time
        
        print(f"🔬 Hyperparameter optimization completed in {optimization_time:.1f}s")
        print(f"   Best score: {result.best_score:.4f} with {result.total_evaluations} evaluations")
        
        return result
    
    def _grid_search_optimization(self, test_scenarios: List[Dict[str, Any]]) -> OptimizationResult:
        """Grid search hyperparameter optimization"""
        print("🔬 Running grid search optimization...")
        
        # Define grid points for each parameter
        grid_points = {}
        for param_name, param_range in self.hyperparameter_space.parameter_ranges.items():
            if param_range.constraint_type == 'integer':
                # Integer parameters - use fewer points
                points = np.linspace(param_range.min_value, param_range.max_value, 5)
                grid_points[param_name] = [int(p) for p in points]
            else:
                # Continuous parameters
                grid_points[param_name] = np.linspace(param_range.min_value, param_range.max_value, 7)
        
        # Generate all parameter combinations
        param_names = list(grid_points.keys())
        param_values = list(grid_points.values())
        
        all_combinations = list(itertools.product(*param_values))
        
        # Limit combinations to budget
        if len(all_combinations) > self.config['max_evaluations']:
            # Randomly sample combinations
            selected_combinations = random.sample(all_combinations, self.config['max_evaluations'])
        else:
            selected_combinations = all_combinations
        
        # Evaluate combinations
        best_params = None
        best_score = -float('inf')
        evaluation_results = []
        
        for i, combination in enumerate(selected_combinations):
            params = dict(zip(param_names, combination))
            
            # Check constraints
            if not self._check_constraints(params):
                continue
            
            # Evaluate parameters
            score, objectives = self._evaluate_parameters(params, test_scenarios)
            
            evaluation_results.append({
                'parameters': params.copy(),
                'score': score,
                'objectives': objectives,
                'evaluation_id': i
            })
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
            
            self.total_evaluations += 1
            
            # Check budget constraints
            if self._should_stop_optimization():
                break
        
        return OptimizationResult(
            best_parameters=best_params,
            best_score=best_score,
            evaluation_history=evaluation_results,
            optimization_time=0.0,  # Will be set by caller
            total_evaluations=self.total_evaluations,
            convergence_generation=len(evaluation_results),
            method_used=OptimizationMethod.GRID_SEARCH,
            objective_values=evaluation_results[-1]['objectives'] if evaluation_results else {}
        )
    
    def _random_search_optimization(self, test_scenarios: List[Dict[str, Any]]) -> OptimizationResult:
        """Random search hyperparameter optimization"""
        print("🔬 Running random search optimization...")
        
        best_params = None
        best_score = -float('inf')
        evaluation_results = []
        no_improvement_count = 0
        
        for i in range(self.config['max_evaluations']):
            # Generate random parameters
            params = self._generate_random_parameters()
            
            # Check constraints
            if not self._check_constraints(params):
                continue
            
            # Evaluate parameters
            score, objectives = self._evaluate_parameters(params, test_scenarios)
            
            evaluation_results.append({
                'parameters': params.copy(),
                'score': score,
                'objectives': objectives,
                'evaluation_id': i
            })
            
            if score > best_score + self.config['early_stopping_threshold']:
                best_score = score
                best_params = params.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            self.total_evaluations += 1
            
            # Early stopping
            if no_improvement_count >= self.config['early_stopping_patience']:
                print(f"🔬 Early stopping after {no_improvement_count} iterations without improvement")
                break
            
            # Check budget constraints
            if self._should_stop_optimization():
                break
        
        return OptimizationResult(
            best_parameters=best_params,
            best_score=best_score,
            evaluation_history=evaluation_results,
            optimization_time=0.0,
            total_evaluations=self.total_evaluations,
            convergence_generation=len(evaluation_results),
            method_used=OptimizationMethod.RANDOM_SEARCH,
            objective_values=evaluation_results[-1]['objectives'] if evaluation_results else {}
        )
    
    def _genetic_optimization(self, test_scenarios: List[Dict[str, Any]]) -> OptimizationResult:
        """Genetic algorithm for hyperparameter optimization"""
        print("🔬 Running genetic hyperparameter optimization...")
        
        # GA parameters for hyperparameter optimization
        population_size = 20
        mutation_rate = 0.1
        crossover_rate = 0.8
        elite_size = 3
        
        # Initialize population
        population = []
        for _ in range(population_size):
            params = self._generate_random_parameters()
            if self._check_constraints(params):
                population.append(params)
        
        # Fill population if constraints filtered too many
        while len(population) < population_size:
            params = self._generate_random_parameters()
            if self._check_constraints(params):
                population.append(params)
        
        best_params = None
        best_score = -float('inf')
        evaluation_results = []
        
        generation = 0
        max_generations = self.config['max_evaluations'] // population_size
        
        for gen in range(max_generations):
            # Evaluate population
            population_scores = []
            
            for params in population:
                score, objectives = self._evaluate_parameters(params, test_scenarios)
                population_scores.append((params, score, objectives))
                
                evaluation_results.append({
                    'parameters': params.copy(),
                    'score': score,
                    'objectives': objectives,
                    'generation': gen,
                    'evaluation_id': self.total_evaluations
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                self.total_evaluations += 1
                
                if self._should_stop_optimization():
                    break
            
            if self._should_stop_optimization():
                break
            
            # Sort by fitness
            population_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Create next generation
            new_population = []
            
            # Elitism
            for i in range(elite_size):
                new_population.append(population_scores[i][0].copy())
            
            # Crossover and mutation
            while len(new_population) < population_size:
                if random.random() < crossover_rate:
                    # Crossover
                    parent1 = self._tournament_selection(population_scores, 3)
                    parent2 = self._tournament_selection(population_scores, 3)
                    child = self._crossover_parameters(parent1, parent2)
                else:
                    # Copy parent
                    child = self._tournament_selection(population_scores, 3).copy()
                
                # Mutation
                if random.random() < mutation_rate:
                    child = self._mutate_parameters(child)
                
                # Check constraints
                if self._check_constraints(child):
                    new_population.append(child)
                else:
                    # Add random individual if constraint violation
                    new_population.append(self._generate_random_parameters())
            
            population = new_population
            generation += 1
        
        return OptimizationResult(
            best_parameters=best_params,
            best_score=best_score,
            evaluation_history=evaluation_results,
            optimization_time=0.0,
            total_evaluations=self.total_evaluations,
            convergence_generation=generation,
            method_used=OptimizationMethod.GENETIC,
            objective_values=evaluation_results[-1]['objectives'] if evaluation_results else {}
        )
    
    def _bayesian_optimization(self, test_scenarios: List[Dict[str, Any]]) -> OptimizationResult:
        """Simplified Bayesian optimization (acquisition function based)"""
        print("🔬 Running Bayesian optimization...")
        
        # For simplicity, implement as guided random search with exploitation/exploration
        best_params = None
        best_score = -float('inf')
        evaluation_results = []
        
        # Initial random evaluations
        initial_evaluations = min(10, self.config['max_evaluations'] // 4)
        
        for i in range(initial_evaluations):
            params = self._generate_random_parameters()
            
            if not self._check_constraints(params):
                continue
            
            score, objectives = self._evaluate_parameters(params, test_scenarios)
            
            evaluation_results.append({
                'parameters': params.copy(),
                'score': score,
                'objectives': objectives,
                'evaluation_id': i,
                'phase': 'exploration'
            })
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
            
            self.total_evaluations += 1
        
        # Guided search phase
        for i in range(initial_evaluations, self.config['max_evaluations']):
            # Balance exploration vs exploitation
            exploration_prob = max(0.1, 1.0 - (i / self.config['max_evaluations']))
            
            if random.random() < exploration_prob:
                # Exploration - random search
                params = self._generate_random_parameters()
                phase = 'exploration'
            else:
                # Exploitation - search near best parameters
                params = self._generate_near_best_parameters(best_params)
                phase = 'exploitation'
            
            if not self._check_constraints(params):
                continue
            
            score, objectives = self._evaluate_parameters(params, test_scenarios)
            
            evaluation_results.append({
                'parameters': params.copy(),
                'score': score,
                'objectives': objectives,
                'evaluation_id': i,
                'phase': phase
            })
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
            
            self.total_evaluations += 1
            
            if self._should_stop_optimization():
                break
        
        return OptimizationResult(
            best_parameters=best_params,
            best_score=best_score,
            evaluation_history=evaluation_results,
            optimization_time=0.0,
            total_evaluations=self.total_evaluations,
            convergence_generation=len(evaluation_results),
            method_used=OptimizationMethod.BAYESIAN,
            objective_values=evaluation_results[-1]['objectives'] if evaluation_results else {}
        )
    
    def _particle_swarm_optimization(self, test_scenarios: List[Dict[str, Any]]) -> OptimizationResult:
        """Particle Swarm Optimization for hyperparameters"""
        print("🔬 Running Particle Swarm Optimization...")
        
        # PSO parameters
        swarm_size = 15
        w = 0.5  # Inertia weight
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient
        
        # Initialize swarm
        particles = []
        velocities = []
        personal_best = []
        personal_best_scores = []
        
        for _ in range(swarm_size):
            params = self._generate_random_parameters()
            particles.append(params)
            
            # Initialize velocity
            velocity = {}
            for param_name in params:
                param_range = self.hyperparameter_space.parameter_ranges[param_name]
                max_velocity = (param_range.max_value - param_range.min_value) * 0.1
                velocity[param_name] = random.uniform(-max_velocity, max_velocity)
            velocities.append(velocity)
            
            personal_best.append(params.copy())
            personal_best_scores.append(-float('inf'))
        
        global_best = None
        global_best_score = -float('inf')
        evaluation_results = []
        
        iteration = 0
        max_iterations = self.config['max_evaluations'] // swarm_size
        
        for iter_num in range(max_iterations):
            for i, params in enumerate(particles):
                if not self._check_constraints(params):
                    # Re-initialize if constraints violated
                    particles[i] = self._generate_random_parameters()
                    continue
                
                # Evaluate particle
                score, objectives = self._evaluate_parameters(params, test_scenarios)
                
                evaluation_results.append({
                    'parameters': params.copy(),
                    'score': score,
                    'objectives': objectives,
                    'iteration': iter_num,
                    'particle_id': i,
                    'evaluation_id': self.total_evaluations
                })
                
                # Update personal best
                if score > personal_best_scores[i]:
                    personal_best[i] = params.copy()
                    personal_best_scores[i] = score
                
                # Update global best
                if score > global_best_score:
                    global_best = params.copy()
                    global_best_score = score
                
                self.total_evaluations += 1
                
                if self._should_stop_optimization():
                    break
            
            if self._should_stop_optimization():
                break
            
            # Update velocities and positions
            for i in range(swarm_size):
                for param_name in particles[i]:
                    param_range = self.hyperparameter_space.parameter_ranges[param_name]
                    
                    # Update velocity
                    r1, r2 = random.random(), random.random()
                    cognitive = c1 * r1 * (personal_best[i][param_name] - particles[i][param_name])
                    social = c2 * r2 * (global_best[param_name] - particles[i][param_name])
                    
                    velocities[i][param_name] = (w * velocities[i][param_name] + 
                                               cognitive + social)
                    
                    # Limit velocity
                    max_velocity = (param_range.max_value - param_range.min_value) * 0.2
                    velocities[i][param_name] = max(-max_velocity, 
                                                  min(max_velocity, velocities[i][param_name]))
                    
                    # Update position
                    particles[i][param_name] += velocities[i][param_name]
                    particles[i][param_name] = param_range.clamp(particles[i][param_name])
                    
                    # Handle integer constraints
                    if param_range.constraint_type == 'integer':
                        particles[i][param_name] = round(particles[i][param_name])
            
            iteration += 1
        
        return OptimizationResult(
            best_parameters=global_best,
            best_score=global_best_score,
            evaluation_history=evaluation_results,
            optimization_time=0.0,
            total_evaluations=self.total_evaluations,
            convergence_generation=iteration,
            method_used=OptimizationMethod.PARTICLE_SWARM,
            objective_values=evaluation_results[-1]['objectives'] if evaluation_results else {}
        )
    
    def _simulated_annealing_optimization(self, test_scenarios: List[Dict[str, Any]]) -> OptimizationResult:
        """Simulated Annealing optimization"""
        print("🔬 Running Simulated Annealing optimization...")
        
        # SA parameters
        initial_temperature = 1.0
        final_temperature = 0.01
        cooling_rate = 0.95
        
        # Initialize with random solution
        current_params = self._generate_random_parameters()
        current_score, current_objectives = self._evaluate_parameters(current_params, test_scenarios)
        
        best_params = current_params.copy()
        best_score = current_score
        
        evaluation_results = [{
            'parameters': current_params.copy(),
            'score': current_score,
            'objectives': current_objectives,
            'evaluation_id': 0,
            'temperature': initial_temperature,
            'accepted': True
        }]
        
        temperature = initial_temperature
        self.total_evaluations = 1
        
        while (temperature > final_temperature and 
               self.total_evaluations < self.config['max_evaluations'] and
               not self._should_stop_optimization()):
            
            # Generate neighbor solution
            neighbor_params = self._generate_neighbor_parameters(current_params, temperature)
            
            if not self._check_constraints(neighbor_params):
                continue
            
            # Evaluate neighbor
            neighbor_score, neighbor_objectives = self._evaluate_parameters(neighbor_params, test_scenarios)
            
            # Accept/reject decision
            score_diff = neighbor_score - current_score
            accepted = False
            
            if score_diff > 0:
                # Better solution - always accept
                accepted = True
            else:
                # Worse solution - accept with probability
                probability = math.exp(score_diff / temperature)
                if random.random() < probability:
                    accepted = True
            
            evaluation_results.append({
                'parameters': neighbor_params.copy(),
                'score': neighbor_score,
                'objectives': neighbor_objectives,
                'evaluation_id': self.total_evaluations,
                'temperature': temperature,
                'accepted': accepted,
                'score_difference': score_diff
            })
            
            if accepted:
                current_params = neighbor_params
                current_score = neighbor_score
                
                if neighbor_score > best_score:
                    best_params = neighbor_params.copy()
                    best_score = neighbor_score
            
            # Cool down
            temperature *= cooling_rate
            self.total_evaluations += 1
        
        return OptimizationResult(
            best_parameters=best_params,
            best_score=best_score,
            evaluation_history=evaluation_results,
            optimization_time=0.0,
            total_evaluations=self.total_evaluations,
            convergence_generation=len(evaluation_results),
            method_used=OptimizationMethod.SIMULATED_ANNEALING,
            objective_values=evaluation_results[-1]['objectives'] if evaluation_results else {}
        )
    
    def _differential_evolution_optimization(self, test_scenarios: List[Dict[str, Any]]) -> OptimizationResult:
        """Differential Evolution optimization"""
        print("🔬 Running Differential Evolution optimization...")
        
        # DE parameters
        population_size = 20
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability
        
        # Initialize population
        population = []
        for _ in range(population_size):
            params = self._generate_random_parameters()
            population.append(params)
        
        evaluation_results = []
        generation = 0
        max_generations = self.config['max_evaluations'] // population_size
        
        # Evaluate initial population
        population_scores = []
        for i, params in enumerate(population):
            score, objectives = self._evaluate_parameters(params, test_scenarios)
            population_scores.append(score)
            
            evaluation_results.append({
                'parameters': params.copy(),
                'score': score,
                'objectives': objectives,
                'generation': 0,
                'individual_id': i,
                'evaluation_id': self.total_evaluations
            })
            
            self.total_evaluations += 1
        
        best_idx = np.argmax(population_scores)
        best_params = population[best_idx].copy()
        best_score = population_scores[best_idx]
        
        for gen in range(1, max_generations):
            for i in range(population_size):
                if self._should_stop_optimization():
                    break
                
                # Select three random individuals (different from current)
                candidates = list(range(population_size))
                candidates.remove(i)
                a, b, c = random.sample(candidates, 3)
                
                # Create mutant vector
                mutant = {}
                for param_name in population[i]:
                    param_range = self.hyperparameter_space.parameter_ranges[param_name]
                    
                    # DE mutation: Vi = Xa + F * (Xb - Xc)
                    mutant_value = (population[a][param_name] + 
                                   F * (population[b][param_name] - population[c][param_name]))
                    
                    # Ensure bounds
                    mutant[param_name] = param_range.clamp(mutant_value)
                    
                    # Handle integer constraints
                    if param_range.constraint_type == 'integer':
                        mutant[param_name] = round(mutant[param_name])
                
                # Crossover
                trial = {}
                j_rand = random.randint(0, len(population[i]) - 1)  # Ensure at least one parameter is taken from mutant
                
                for j, param_name in enumerate(population[i]):
                    if random.random() < CR or j == j_rand:
                        trial[param_name] = mutant[param_name]
                    else:
                        trial[param_name] = population[i][param_name]
                
                # Check constraints
                if not self._check_constraints(trial):
                    continue
                
                # Evaluate trial
                trial_score, trial_objectives = self._evaluate_parameters(trial, test_scenarios)
                
                evaluation_results.append({
                    'parameters': trial.copy(),
                    'score': trial_score,
                    'objectives': trial_objectives,
                    'generation': gen,
                    'individual_id': i,
                    'evaluation_id': self.total_evaluations,
                    'mutation_type': 'DE_rand_1'
                })
                
                # Selection
                if trial_score > population_scores[i]:
                    population[i] = trial
                    population_scores[i] = trial_score
                    
                    if trial_score > best_score:
                        best_params = trial.copy()
                        best_score = trial_score
                
                self.total_evaluations += 1
            
            if self._should_stop_optimization():
                break
            
            generation += 1
        
        return OptimizationResult(
            best_parameters=best_params,
            best_score=best_score,
            evaluation_history=evaluation_results,
            optimization_time=0.0,
            total_evaluations=self.total_evaluations,
            convergence_generation=generation,
            method_used=OptimizationMethod.DIFFERENTIAL_EVOLUTION,
            objective_values=evaluation_results[-1]['objectives'] if evaluation_results else {}
        )
    
    def _evaluate_parameters(self, parameters: Dict[str, float], 
                           test_scenarios: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
        """Evaluate hyperparameters on test scenarios
        
        Args:
            parameters: Parameter dictionary to evaluate
            test_scenarios: Test scenarios for evaluation
            
        Returns:
            Tuple of (overall_score, objective_scores)
        """
        # Mock evaluation for now - in real implementation, this would run GA with parameters
        # and measure performance on test scenarios
        
        objective_scores = {}
        
        # Simulate fitness objective (higher mutation rate might help exploration)
        mutation_factor = 1.0 + (parameters['mutation_rate'] - 0.1) * 0.5
        population_factor = 1.0 + (parameters['population_size'] - 100) / 1000
        fitness_score = min(1.0, 0.7 + mutation_factor * 0.1 + population_factor * 0.1)
        objective_scores['fitness'] = fitness_score
        
        # Simulate convergence speed (larger population might converge slower)
        convergence_penalty = (parameters['population_size'] - 100) / 1000
        max_gen_penalty = (parameters['max_generations'] - 200) / 1000
        convergence_score = max(0.1, 0.8 - convergence_penalty * 0.3 - max_gen_penalty * 0.2)
        objective_scores['convergence_speed'] = convergence_score
        
        # Simulate stability (balanced parameters are more stable)
        mutation_stability = 1.0 - abs(parameters['mutation_rate'] - 0.1) * 2
        crossover_stability = 1.0 - abs(parameters['crossover_rate'] - 0.8) * 1
        stability_score = max(0.1, (mutation_stability + crossover_stability) / 2)
        objective_scores['stability'] = stability_score
        
        # Simulate diversity (higher mutation and larger tournament promote diversity)
        diversity_score = min(1.0, parameters['mutation_rate'] * 5 + 
                             parameters['tournament_size'] / 20)
        objective_scores['diversity'] = diversity_score
        
        # Calculate weighted overall score
        overall_score = sum(score * self.hyperparameter_space.weights.get(obj, 0.25) 
                           for obj, score in objective_scores.items())
        
        return overall_score, objective_scores
    
    def _generate_random_parameters(self) -> Dict[str, float]:
        """Generate random parameter values within valid ranges"""
        parameters = {}
        
        for param_name, param_range in self.hyperparameter_space.parameter_ranges.items():
            if param_range.constraint_type == 'integer':
                value = random.randint(int(param_range.min_value), int(param_range.max_value))
            else:
                value = random.uniform(param_range.min_value, param_range.max_value)
            
            parameters[param_name] = value
        
        return parameters
    
    def _generate_near_best_parameters(self, best_params: Dict[str, float]) -> Dict[str, float]:
        """Generate parameters near the current best"""
        parameters = best_params.copy()
        
        # Add noise to parameters
        for param_name in parameters:
            param_range = self.hyperparameter_space.parameter_ranges[param_name]
            
            # Add Gaussian noise scaled by parameter range
            noise_scale = (param_range.max_value - param_range.min_value) * 0.1
            noise = np.random.normal(0, noise_scale)
            
            parameters[param_name] = param_range.clamp(parameters[param_name] + noise)
            
            if param_range.constraint_type == 'integer':
                parameters[param_name] = round(parameters[param_name])
        
        return parameters
    
    def _generate_neighbor_parameters(self, current_params: Dict[str, float], 
                                    temperature: float) -> Dict[str, float]:
        """Generate neighbor parameters for simulated annealing"""
        parameters = current_params.copy()
        
        # Choose random parameter to modify
        param_name = random.choice(list(parameters.keys()))
        param_range = self.hyperparameter_space.parameter_ranges[param_name]
        
        # Scale modification by temperature
        max_change = (param_range.max_value - param_range.min_value) * temperature * 0.3
        change = random.uniform(-max_change, max_change)
        
        parameters[param_name] = param_range.clamp(parameters[param_name] + change)
        
        if param_range.constraint_type == 'integer':
            parameters[param_name] = round(parameters[param_name])
        
        return parameters
    
    def _tournament_selection(self, population_scores: List[Tuple], tournament_size: int) -> Dict[str, float]:
        """Tournament selection for genetic optimization"""
        tournament = random.sample(population_scores, min(tournament_size, len(population_scores)))
        winner = max(tournament, key=lambda x: x[1])
        return winner[0]
    
    def _crossover_parameters(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Dict[str, float]:
        """Crossover two parameter sets"""
        child = {}
        
        for param_name in parent1:
            if random.random() < 0.5:
                child[param_name] = parent1[param_name]
            else:
                child[param_name] = parent2[param_name]
        
        return child
    
    def _mutate_parameters(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Mutate parameter set"""
        mutated = parameters.copy()
        
        for param_name in mutated:
            if random.random() < 0.3:  # 30% mutation probability per parameter
                param_range = self.hyperparameter_space.parameter_ranges[param_name]
                
                # Add random noise
                noise_scale = (param_range.max_value - param_range.min_value) * 0.1
                noise = random.uniform(-noise_scale, noise_scale)
                
                mutated[param_name] = param_range.clamp(mutated[param_name] + noise)
                
                if param_range.constraint_type == 'integer':
                    mutated[param_name] = round(mutated[param_name])
        
        return mutated
    
    def _check_constraints(self, parameters: Dict[str, float]) -> bool:
        """Check if parameters satisfy constraints"""
        for constraint in self.hyperparameter_space.constraints:
            if not constraint(parameters):
                return False
        return True
    
    def _should_stop_optimization(self) -> bool:
        """Check if optimization should stop due to budget constraints"""
        if self.optimization_start_time is None:
            return False
        
        elapsed_time = time.time() - self.optimization_start_time
        budget_minutes = self.config['optimization_budget_minutes']
        
        return elapsed_time > budget_minutes * 60
    
    def compare_optimization_methods(self, test_scenarios: List[Dict[str, Any]],
                                   methods: Optional[List[OptimizationMethod]] = None) -> Dict[str, OptimizationResult]:
        """Compare different optimization methods
        
        Args:
            test_scenarios: Test scenarios for evaluation
            methods: List of methods to compare (default: all)
            
        Returns:
            Dictionary mapping method names to optimization results
        """
        if methods is None:
            methods = [OptimizationMethod.RANDOM_SEARCH, OptimizationMethod.GENETIC,
                      OptimizationMethod.PARTICLE_SWARM, OptimizationMethod.SIMULATED_ANNEALING]
        
        results = {}
        
        for method in methods:
            print(f"\n🔬 Comparing optimization method: {method.value}")
            
            # Reset state for fair comparison
            self.evaluation_history = []
            self.best_result = None
            self.total_evaluations = 0
            
            # Reduce evaluation budget for comparison
            original_max_eval = self.config['max_evaluations']
            self.config['max_evaluations'] = min(50, original_max_eval // len(methods))
            
            try:
                result = self.optimize_hyperparameters(method, test_scenarios)
                results[method.value] = result
                
                print(f"   {method.value}: Best score {result.best_score:.4f} "
                      f"({result.total_evaluations} evaluations)")
                
            except Exception as e:
                print(f"   {method.value}: Failed with error {e}")
                results[method.value] = None
            
            # Restore original budget
            self.config['max_evaluations'] = original_max_eval
        
        return results
    
    def save_optimization_results(self, result: OptimizationResult, filename: str) -> str:
        """Save optimization results to file
        
        Args:
            result: Optimization result to save
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        result_data = {
            'best_parameters': result.best_parameters,
            'best_score': result.best_score,
            'optimization_time': result.optimization_time,
            'total_evaluations': result.total_evaluations,
            'convergence_generation': result.convergence_generation,
            'method_used': result.method_used.value,
            'objective_values': result.objective_values,
            'evaluation_history': result.evaluation_history,
            'hyperparameter_space': {
                'parameter_ranges': {name: {
                    'min_value': pr.min_value,
                    'max_value': pr.max_value,
                    'default_value': pr.default_value,
                    'constraint_type': pr.constraint_type
                } for name, pr in self.hyperparameter_space.parameter_ranges.items()},
                'objectives': self.hyperparameter_space.objectives,
                'weights': self.hyperparameter_space.weights
            },
            'config': self.config
        }
        
        with open(filename, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        return filename


def test_hyperparameter_optimizer():
    """Test function for hyperparameter optimizer"""
    print("Testing GA Hyperparameter Optimizer...")
    
    # Create optimizer
    optimizer = GAHyperparameterOptimizer()
    
    # Test parameter generation
    random_params = optimizer._generate_random_parameters()
    print(f"✅ Random parameters generated: {len(random_params)} parameters")
    
    # Test constraint checking
    constraints_ok = optimizer._check_constraints(random_params)
    print(f"✅ Constraint checking: {constraints_ok}")
    
    # Test parameter evaluation
    test_scenarios = [{'objective': 'elevation', 'distance': 5.0}]
    score, objectives = optimizer._evaluate_parameters(random_params, test_scenarios)
    print(f"✅ Parameter evaluation: score {score:.3f}, objectives {objectives}")
    
    # Test random search optimization (small scale)
    optimizer.config['max_evaluations'] = 10
    result = optimizer._random_search_optimization(test_scenarios)
    print(f"✅ Random search optimization: best score {result.best_score:.3f}")
    
    print("✅ All hyperparameter optimizer tests completed")


if __name__ == "__main__":
    test_hyperparameter_optimizer()