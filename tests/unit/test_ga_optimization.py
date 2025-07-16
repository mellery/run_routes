#!/usr/bin/env python3
"""
Unit tests for genetic_algorithm/optimization.py
Tests comprehensive functionality of GA optimization components
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import tempfile
import shutil
import time
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from genetic_algorithm.optimization import (
    OptimizationMethod, AdaptationStrategy, ParameterRange, HyperparameterSpace,
    OptimizationResult, AdaptationRule, GAHyperparameterOptimizer, GAParameterTuner,
    GAAlgorithmSelector
)
from ga_common_imports import GAStatistics


class TestOptimizationEnums(unittest.TestCase):
    """Test optimization enums"""
    
    def test_optimization_method_enum(self):
        """Test OptimizationMethod enum values"""
        self.assertEqual(OptimizationMethod.GRID_SEARCH.value, "grid_search")
        self.assertEqual(OptimizationMethod.RANDOM_SEARCH.value, "random_search")
        self.assertEqual(OptimizationMethod.BAYESIAN.value, "bayesian")
        self.assertEqual(OptimizationMethod.GENETIC.value, "genetic")
        self.assertEqual(OptimizationMethod.PARTICLE_SWARM.value, "particle_swarm")
        self.assertEqual(OptimizationMethod.SIMULATED_ANNEALING.value, "simulated_annealing")
        self.assertEqual(OptimizationMethod.DIFFERENTIAL_EVOLUTION.value, "differential_evolution")
    
    def test_adaptation_strategy_enum(self):
        """Test AdaptationStrategy enum values"""
        self.assertEqual(AdaptationStrategy.INCREASE_ON_STAGNATION.value, "increase_on_stagnation")
        self.assertEqual(AdaptationStrategy.DECREASE_ON_STAGNATION.value, "decrease_on_stagnation")
        self.assertEqual(AdaptationStrategy.OSCILLATE.value, "oscillate")
        self.assertEqual(AdaptationStrategy.PERFORMANCE_BASED.value, "performance_based")
        self.assertEqual(AdaptationStrategy.DIVERSITY_BASED.value, "diversity_based")
        self.assertEqual(AdaptationStrategy.CONVERGENCE_BASED.value, "convergence_based")
        self.assertEqual(AdaptationStrategy.HYBRID.value, "hybrid")


class TestParameterRange(unittest.TestCase):
    """Test ParameterRange dataclass"""
    
    def test_parameter_range_creation(self):
        """Test ParameterRange creation"""
        param_range = ParameterRange(
            min_value=0.1,
            max_value=0.9,
            step_size=0.1,
            parameter_type="continuous"
        )
        
        self.assertEqual(param_range.min_value, 0.1)
        self.assertEqual(param_range.max_value, 0.9)
        self.assertEqual(param_range.step_size, 0.1)
        self.assertEqual(param_range.parameter_type, "continuous")
        self.assertIsNone(param_range.values)
    
    def test_parameter_range_with_values(self):
        """Test ParameterRange with predefined values"""
        param_range = ParameterRange(
            min_value=0.0,
            max_value=1.0,
            values=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        
        self.assertEqual(param_range.values, [0.1, 0.2, 0.3, 0.4, 0.5])
    
    @patch('random.choice')
    def test_parameter_range_sample_with_values(self, mock_choice):
        """Test ParameterRange sampling with predefined values"""
        mock_choice.return_value = 0.3
        
        param_range = ParameterRange(
            min_value=0.0,
            max_value=1.0,
            values=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        
        result = param_range.sample()
        self.assertEqual(result, 0.3)
        mock_choice.assert_called_once_with([0.1, 0.2, 0.3, 0.4, 0.5])
    
    @patch('random.randint')
    def test_parameter_range_sample_discrete(self, mock_randint):
        """Test ParameterRange sampling for discrete type"""
        mock_randint.return_value = 5
        
        param_range = ParameterRange(
            min_value=1,
            max_value=10,
            parameter_type="discrete"
        )
        
        result = param_range.sample()
        self.assertEqual(result, 5)
        mock_randint.assert_called_once_with(1, 10)
    
    @patch('random.uniform')
    def test_parameter_range_sample_continuous(self, mock_uniform):
        """Test ParameterRange sampling for continuous type"""
        mock_uniform.return_value = 0.5
        
        param_range = ParameterRange(
            min_value=0.0,
            max_value=1.0,
            parameter_type="continuous"
        )
        
        result = param_range.sample()
        self.assertEqual(result, 0.5)
        mock_uniform.assert_called_once_with(0.0, 1.0)


class TestHyperparameterSpace(unittest.TestCase):
    """Test HyperparameterSpace dataclass"""
    
    def test_hyperparameter_space_creation(self):
        """Test HyperparameterSpace creation"""
        param_ranges = {
            'mutation_rate': ParameterRange(0.01, 0.3),
            'population_size': ParameterRange(50, 200, parameter_type="discrete")
        }
        
        space = HyperparameterSpace(parameter_ranges=param_ranges)
        
        self.assertEqual(space.parameter_ranges, param_ranges)
        self.assertEqual(space.constraints, [])
        self.assertEqual(space.objectives, ['fitness', 'convergence_speed'])
        self.assertEqual(space.weights, {'fitness': 0.7, 'convergence_speed': 0.3})
    
    def test_hyperparameter_space_with_constraints(self):
        """Test HyperparameterSpace with constraints"""
        param_ranges = {
            'mutation_rate': ParameterRange(0.01, 0.3),
            'population_size': ParameterRange(50, 200, parameter_type="discrete")
        }
        
        def constraint_func(params):
            return params['mutation_rate'] < 0.2
        
        space = HyperparameterSpace(
            parameter_ranges=param_ranges,
            constraints=[constraint_func],
            objectives=['fitness', 'speed'],
            weights={'fitness': 0.8, 'speed': 0.2}
        )
        
        self.assertEqual(len(space.constraints), 1)
        self.assertEqual(space.objectives, ['fitness', 'speed'])
        self.assertEqual(space.weights, {'fitness': 0.8, 'speed': 0.2})


class TestOptimizationResult(unittest.TestCase):
    """Test OptimizationResult dataclass"""
    
    def test_optimization_result_creation(self):
        """Test OptimizationResult creation"""
        result = OptimizationResult(
            best_parameters={'mutation_rate': 0.1, 'population_size': 100},
            best_score=0.95,
            evaluation_history=[],
            optimization_time=60.0,
            total_evaluations=50,
            convergence_generation=25,
            method_used=OptimizationMethod.RANDOM_SEARCH,
            objective_values={'fitness': 0.95}
        )
        
        self.assertEqual(result.best_parameters, {'mutation_rate': 0.1, 'population_size': 100})
        self.assertEqual(result.best_score, 0.95)
        self.assertEqual(result.evaluation_history, [])
        self.assertEqual(result.optimization_time, 60.0)
        self.assertEqual(result.total_evaluations, 50)
        self.assertEqual(result.convergence_generation, 25)
        self.assertEqual(result.method_used, OptimizationMethod.RANDOM_SEARCH)
        self.assertEqual(result.objective_values, {'fitness': 0.95})


class TestAdaptationRule(unittest.TestCase):
    """Test AdaptationRule dataclass"""
    
    def test_adaptation_rule_creation(self):
        """Test AdaptationRule creation"""
        def trigger_condition(stats):
            return stats.generation > 10
        
        rule = AdaptationRule(
            parameter_name='mutation_rate',
            strategy=AdaptationStrategy.INCREASE_ON_STAGNATION,
            trigger_condition=trigger_condition,
            adaptation_amount=0.02,
            cooldown_generations=5,
            min_value=0.01,
            max_value=0.5
        )
        
        self.assertEqual(rule.parameter_name, 'mutation_rate')
        self.assertEqual(rule.strategy, AdaptationStrategy.INCREASE_ON_STAGNATION)
        self.assertEqual(rule.trigger_condition, trigger_condition)
        self.assertEqual(rule.adaptation_amount, 0.02)
        self.assertEqual(rule.cooldown_generations, 5)
        self.assertEqual(rule.last_adjustment_generation, 0)
        self.assertEqual(rule.min_value, 0.01)
        self.assertEqual(rule.max_value, 0.5)
    
    def test_adaptation_rule_defaults(self):
        """Test AdaptationRule with default values"""
        def trigger_condition(stats):
            return True
        
        rule = AdaptationRule(
            parameter_name='population_size',
            strategy=AdaptationStrategy.DECREASE_ON_STAGNATION,
            trigger_condition=trigger_condition,
            adaptation_amount=10
        )
        
        self.assertEqual(rule.cooldown_generations, 5)
        self.assertEqual(rule.last_adjustment_generation, 0)
        self.assertIsNone(rule.min_value)
        self.assertIsNone(rule.max_value)


class TestGAHyperparameterOptimizer(unittest.TestCase):
    """Test GAHyperparameterOptimizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = GAHyperparameterOptimizer()
        
        # Create test hyperparameter space
        self.param_ranges = {
            'mutation_rate': ParameterRange(0.01, 0.3),
            'population_size': ParameterRange(50, 200, parameter_type="discrete"),
            'crossover_rate': ParameterRange(0.5, 1.0)
        }
        self.hyperparameter_space = HyperparameterSpace(parameter_ranges=self.param_ranges)
    
    def test_optimizer_initialization_default(self):
        """Test optimizer initialization with default config"""
        optimizer = GAHyperparameterOptimizer()
        
        self.assertEqual(optimizer.config['max_evaluations'], 100)
        self.assertEqual(optimizer.config['evaluation_timeout'], 300)
        self.assertEqual(optimizer.config['parallel_evaluations'], 4)
        self.assertEqual(optimizer.config['random_seed'], 42)
        self.assertEqual(optimizer.best_score, float('-inf'))
        self.assertEqual(optimizer.best_parameters, {})
        self.assertEqual(optimizer.evaluation_history, [])
    
    def test_optimizer_initialization_custom_config(self):
        """Test optimizer initialization with custom config"""
        custom_config = {
            'max_evaluations': 50,
            'evaluation_timeout': 600,
            'random_seed': 123
        }
        
        optimizer = GAHyperparameterOptimizer(config=custom_config)
        
        self.assertEqual(optimizer.config['max_evaluations'], 50)
        self.assertEqual(optimizer.config['evaluation_timeout'], 600)
        self.assertEqual(optimizer.config['random_seed'], 123)
        # Should keep defaults for unspecified options
        self.assertEqual(optimizer.config['parallel_evaluations'], 4)
    
    @patch('time.time')
    def test_optimize_grid_search(self, mock_time):
        """Test grid search optimization"""
        mock_time.side_effect = [i * 0.1 for i in range(100)]  # Provide enough time values
        
        # Create smaller parameter space for testing
        param_ranges = {
            'mutation_rate': ParameterRange(0.1, 0.3, values=[0.1, 0.2, 0.3]),
            'population_size': ParameterRange(50, 100, values=[50, 100])
        }
        space = HyperparameterSpace(parameter_ranges=param_ranges)
        
        # Mock the evaluation function
        def mock_objective(params):
            return params['mutation_rate'] + params['population_size'] / 1000
        
        result = self.optimizer.optimize(space, OptimizationMethod.GRID_SEARCH, mock_objective)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.method_used, OptimizationMethod.GRID_SEARCH)
        self.assertGreater(result.best_score, 0)
        self.assertIn('mutation_rate', result.best_parameters)
        self.assertIn('population_size', result.best_parameters)
        self.assertGreater(result.optimization_time, 0)
    
    @patch('time.time')
    def test_optimize_random_search(self, mock_time):
        """Test random search optimization"""
        mock_time.side_effect = [i * 0.1 for i in range(100)]  # Provide enough time values
        
        # Mock the evaluation function
        def mock_objective(params):
            return 0.5
        
        # Use small max_evaluations for testing
        self.optimizer.config['max_evaluations'] = 5
        
        result = self.optimizer.optimize(self.hyperparameter_space, OptimizationMethod.RANDOM_SEARCH, mock_objective)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.method_used, OptimizationMethod.RANDOM_SEARCH)
        self.assertEqual(result.best_score, 0.5)
        self.assertLessEqual(result.total_evaluations, 5)
        self.assertGreater(result.optimization_time, 0)
    
    @patch('time.time')
    def test_optimize_bayesian_fallback(self, mock_time):
        """Test Bayesian optimization fallback to random search"""
        mock_time.side_effect = [i * 0.1 for i in range(100)]  # Provide enough time values
        
        # Mock the evaluation function
        def mock_objective(params):
            return 0.7
        
        # Use small max_evaluations for testing
        self.optimizer.config['max_evaluations'] = 3
        
        result = self.optimizer.optimize(self.hyperparameter_space, OptimizationMethod.BAYESIAN, mock_objective)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.method_used, OptimizationMethod.RANDOM_SEARCH)  # Should fallback
        self.assertEqual(result.best_score, 0.7)
    
    def test_optimize_unsupported_method(self):
        """Test optimization with unsupported method"""
        with self.assertRaises(ValueError) as context:
            self.optimizer.optimize(self.hyperparameter_space, OptimizationMethod.GENETIC)
        
        self.assertIn("not implemented", str(context.exception))
    
    @patch('time.time')
    def test_grid_search_with_constraints(self, mock_time):
        """Test grid search with constraints"""
        mock_time.side_effect = [i * 0.1 for i in range(100)]
        
        # Add constraint
        def constraint_func(params):
            return params['mutation_rate'] < 0.2
        
        param_ranges = {
            'mutation_rate': ParameterRange(0.1, 0.3, values=[0.1, 0.15, 0.2, 0.25, 0.3])
        }
        space = HyperparameterSpace(
            parameter_ranges=param_ranges,
            constraints=[constraint_func]
        )
        
        def mock_objective(params):
            return params['mutation_rate']
        
        result = self.optimizer.optimize(space, OptimizationMethod.GRID_SEARCH, mock_objective)
        
        # Should only find parameters that satisfy constraints
        self.assertLess(result.best_parameters['mutation_rate'], 0.2)
    
    @patch('time.time')
    def test_random_search_with_constraints(self, mock_time):
        """Test random search with constraints"""
        mock_time.side_effect = [i * 0.1 for i in range(100)]
        
        # Add constraint
        def constraint_func(params):
            return params['mutation_rate'] > 0.15
        
        space = HyperparameterSpace(
            parameter_ranges=self.param_ranges,
            constraints=[constraint_func]
        )
        
        def mock_objective(params):
            return params['mutation_rate']
        
        # Use small max_evaluations for testing
        self.optimizer.config['max_evaluations'] = 10
        
        # Mock parameter sampling to ensure we get valid samples
        with patch.object(self.param_ranges['mutation_rate'], 'sample', return_value=0.2):
            with patch.object(self.param_ranges['population_size'], 'sample', return_value=100):
                with patch.object(self.param_ranges['crossover_rate'], 'sample', return_value=0.8):
                    result = self.optimizer.optimize(space, OptimizationMethod.RANDOM_SEARCH, mock_objective)
        
        self.assertGreater(result.best_parameters['mutation_rate'], 0.15)
    
    @patch('time.time')
    def test_evaluate_parameters_with_objective(self, mock_time):
        """Test parameter evaluation with custom objective function"""
        mock_time.side_effect = [i * 0.1 for i in range(50)]
        
        def mock_objective(params):
            return params['mutation_rate'] * 2
        
        params = {'mutation_rate': 0.1, 'population_size': 100}
        score = self.optimizer._evaluate_parameters(params, mock_objective)
        
        self.assertEqual(score, 0.2)
        self.assertEqual(len(self.optimizer.evaluation_history), 1)
        self.assertEqual(self.optimizer.evaluation_history[0]['parameters'], params)
        self.assertEqual(self.optimizer.evaluation_history[0]['score'], 0.2)
    
    @patch('time.time')
    def test_evaluate_parameters_with_exception(self, mock_time):
        """Test parameter evaluation with exception"""
        mock_time.side_effect = [i * 0.1 for i in range(50)]
        
        def failing_objective(params):
            raise ValueError("Test exception")
        
        params = {'mutation_rate': 0.1}
        score = self.optimizer._evaluate_parameters(params, failing_objective)
        
        self.assertEqual(score, float('-inf'))
    
    def test_simulate_ga_run(self):
        """Test GA simulation for parameter evaluation"""
        # Test optimal parameters
        params = {
            'population_size': 150,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8
        }
        
        score = self.optimizer._simulate_ga_run(params)
        
        self.assertGreater(score, 0.5)  # Should score well
        self.assertLessEqual(score, 1.0)
        
        # Test suboptimal parameters
        params = {
            'population_size': 20,
            'mutation_rate': 0.5,
            'crossover_rate': 0.3
        }
        
        score = self.optimizer._simulate_ga_run(params)
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    @patch('time.time')
    def test_should_stop_time_limit(self, mock_time):
        """Test stopping condition based on time limit"""
        # Set short time limit
        self.optimizer.config['optimization_budget_minutes'] = 1
        
        # Simulate optimization start
        self.optimizer.optimization_start_time = 0.0
        
        # Test within time limit
        mock_time.return_value = 30.0  # 30 seconds
        self.assertFalse(self.optimizer._should_stop())
        
        # Test beyond time limit
        mock_time.return_value = 70.0  # 70 seconds (> 1 minute)
        self.assertTrue(self.optimizer._should_stop())
    
    def test_should_stop_no_start_time(self):
        """Test stopping condition when no start time set"""
        self.optimizer.optimization_start_time = None
        self.assertFalse(self.optimizer._should_stop())


class TestGAParameterTuner(unittest.TestCase):
    """Test GAParameterTuner class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tuner = GAParameterTuner()
        
        # Create mock statistics
        self.mock_stats = Mock(spec=GAStatistics)
        self.mock_stats.generation = 10
        self.mock_stats.diversity_score = 0.5
        self.mock_stats.best_fitness = 0.8
        self.mock_stats.convergence_rate = 0.1
    
    def test_tuner_initialization_default(self):
        """Test tuner initialization with default parameters"""
        tuner = GAParameterTuner()
        
        expected_params = {
            'population_size': 100,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'elite_size': 2,
            'tournament_size': 3
        }
        
        self.assertEqual(tuner.initial_parameters, expected_params)
        self.assertEqual(tuner.current_parameters, expected_params)
        self.assertEqual(len(tuner.adaptation_rules), 2)  # Default rules
        self.assertEqual(tuner.adaptation_history, [])
    
    def test_tuner_initialization_custom_parameters(self):
        """Test tuner initialization with custom parameters"""
        custom_params = {
            'population_size': 150,
            'mutation_rate': 0.05,
            'crossover_rate': 0.9
        }
        
        tuner = GAParameterTuner(initial_parameters=custom_params)
        
        self.assertEqual(tuner.initial_parameters, custom_params)
        self.assertEqual(tuner.current_parameters, custom_params)
    
    def test_setup_default_rules(self):
        """Test default adaptation rules setup"""
        tuner = GAParameterTuner()
        
        # Should have 2 default rules
        self.assertEqual(len(tuner.adaptation_rules), 2)
        
        # Check mutation rate rule
        mutation_rule = tuner.adaptation_rules[0]
        self.assertEqual(mutation_rule.parameter_name, 'mutation_rate')
        self.assertEqual(mutation_rule.strategy, AdaptationStrategy.INCREASE_ON_STAGNATION)
        self.assertEqual(mutation_rule.adaptation_amount, 0.02)
        self.assertEqual(mutation_rule.min_value, 0.05)
        self.assertEqual(mutation_rule.max_value, 0.3)
        
        # Check population size rule
        population_rule = tuner.adaptation_rules[1]
        self.assertEqual(population_rule.parameter_name, 'population_size')
        self.assertEqual(population_rule.strategy, AdaptationStrategy.DECREASE_ON_STAGNATION)
        self.assertEqual(population_rule.adaptation_amount, 10)
        self.assertEqual(population_rule.min_value, 50)
        self.assertEqual(population_rule.max_value, 200)
    
    def test_adapt_parameters_increase_mutation_rate(self):
        """Test parameter adaptation - increase mutation rate"""
        # Mock fitness stagnation
        with patch.object(self.tuner, '_is_fitness_stagnant', return_value=True):
            with patch.object(self.tuner, '_is_convergence_slow', return_value=False):
                
                initial_mutation_rate = self.tuner.current_parameters['mutation_rate']
                
                result = self.tuner.adapt_parameters(self.mock_stats)
                
                # Should increase mutation rate
                self.assertGreater(result['mutation_rate'], initial_mutation_rate)
                self.assertEqual(result['mutation_rate'], initial_mutation_rate + 0.02)
    
    def test_adapt_parameters_decrease_population_size(self):
        """Test parameter adaptation - decrease population size"""
        # Mock slow convergence
        with patch.object(self.tuner, '_is_fitness_stagnant', return_value=False):
            with patch.object(self.tuner, '_is_convergence_slow', return_value=True):
                
                initial_population_size = self.tuner.current_parameters['population_size']
                
                result = self.tuner.adapt_parameters(self.mock_stats)
                
                # Should decrease population size
                self.assertLess(result['population_size'], initial_population_size)
                self.assertEqual(result['population_size'], initial_population_size - 10)
    
    def test_adapt_parameters_with_bounds(self):
        """Test parameter adaptation respects bounds"""
        # Set mutation rate to maximum
        self.tuner.current_parameters['mutation_rate'] = 0.3
        
        with patch.object(self.tuner, '_is_fitness_stagnant', return_value=True):
            with patch.object(self.tuner, '_is_convergence_slow', return_value=False):
                
                result = self.tuner.adapt_parameters(self.mock_stats)
                
                # Should not exceed maximum
                self.assertEqual(result['mutation_rate'], 0.3)
    
    def test_adapt_parameters_cooldown(self):
        """Test parameter adaptation cooldown"""
        # Set last adjustment to recent generation
        self.tuner.adaptation_rules[0].last_adjustment_generation = 8
        self.mock_stats.generation = 10  # Within cooldown period
        
        with patch.object(self.tuner, '_is_fitness_stagnant', return_value=True):
            with patch.object(self.tuner, '_is_convergence_slow', return_value=False):
                
                initial_mutation_rate = self.tuner.current_parameters['mutation_rate']
                
                result = self.tuner.adapt_parameters(self.mock_stats)
                
                # Should not adapt due to cooldown
                self.assertEqual(result['mutation_rate'], initial_mutation_rate)
    
    def test_adapt_parameters_no_changes(self):
        """Test parameter adaptation when no conditions are met"""
        with patch.object(self.tuner, '_is_fitness_stagnant', return_value=False):
            with patch.object(self.tuner, '_is_convergence_slow', return_value=False):
                
                initial_params = self.tuner.current_parameters.copy()
                
                result = self.tuner.adapt_parameters(self.mock_stats)
                
                # Should not change any parameters
                self.assertEqual(result, initial_params)
    
    def test_is_fitness_stagnant_insufficient_history(self):
        """Test fitness stagnation detection with insufficient history"""
        self.tuner.adaptation_history = [{'best_fitness': 0.8}]  # Too few entries
        
        result = self.tuner._is_fitness_stagnant(self.mock_stats)
        
        self.assertFalse(result)
    
    def test_is_fitness_stagnant_with_history(self):
        """Test fitness stagnation detection with sufficient history"""
        # Create history with stagnant fitness - improvement = 0.81 - 0.80 = 0.01
        # Since 0.01 is NOT less than 0.01, this should return False
        self.tuner.adaptation_history = [
            {'best_fitness': 0.800},
            {'best_fitness': 0.801},
            {'best_fitness': 0.800},
            {'best_fitness': 0.801},
            {'best_fitness': 0.800}
        ]
        
        result = self.tuner._is_fitness_stagnant(self.mock_stats)
        
        self.assertTrue(result)  # Improvement is very small (0.001 < 0.01)
    
    def test_is_fitness_stagnant_with_improvement(self):
        """Test fitness stagnation detection with significant improvement"""
        # Create history with good improvement
        self.tuner.adaptation_history = [
            {'best_fitness': 0.70},
            {'best_fitness': 0.75},
            {'best_fitness': 0.80},
            {'best_fitness': 0.85},
            {'best_fitness': 0.90}
        ]
        
        result = self.tuner._is_fitness_stagnant(self.mock_stats)
        
        self.assertFalse(result)  # Improvement is significant (> 0.01)
    
    def test_is_convergence_slow(self):
        """Test convergence speed detection"""
        # Test slow convergence
        self.mock_stats.generation = 60
        self.mock_stats.diversity_score = 0.9
        
        result = self.tuner._is_convergence_slow(self.mock_stats)
        
        self.assertTrue(result)
        
        # Test normal convergence
        self.mock_stats.generation = 30
        self.mock_stats.diversity_score = 0.5
        
        result = self.tuner._is_convergence_slow(self.mock_stats)
        
        self.assertFalse(result)
    
    @patch('time.time')
    def test_record_adaptation(self, mock_time):
        """Test adaptation recording"""
        mock_time.return_value = 123456.789
        
        changes = {'mutation_rate': 0.12}
        metrics = {'diversity': 0.6}
        reason = "fitness stagnation"
        
        self.tuner._record_adaptation(10, changes, metrics, reason)
        
        self.assertEqual(len(self.tuner.adaptation_history), 1)
        record = self.tuner.adaptation_history[0]
        
        self.assertEqual(record['generation'], 10)
        self.assertEqual(record['parameter_changes'], changes)
        self.assertEqual(record['metrics'], metrics)
        self.assertEqual(record['reason'], reason)
        self.assertEqual(record['timestamp'], 123456.789)


class TestGAAlgorithmSelector(unittest.TestCase):
    """Test GAAlgorithmSelector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.selector = GAAlgorithmSelector()
    
    def test_selector_initialization(self):
        """Test selector initialization"""
        selector = GAAlgorithmSelector()
        
        self.assertEqual(selector.performance_history, {})
        self.assertIn('small_population', selector.algorithm_preferences)
        self.assertIn('large_population', selector.algorithm_preferences)
        self.assertIn('high_diversity', selector.algorithm_preferences)
        self.assertIn('low_diversity', selector.algorithm_preferences)
    
    def test_select_algorithm_small_population(self):
        """Test algorithm selection for small population"""
        characteristics = {
            'population_size': 30,
            'diversity_score': 0.5
        }
        
        algorithm = self.selector.select_algorithm(characteristics)
        
        self.assertIn(algorithm, ['elitism', 'tournament'])
    
    def test_select_algorithm_large_population(self):
        """Test algorithm selection for large population"""
        characteristics = {
            'population_size': 250,
            'diversity_score': 0.5
        }
        
        algorithm = self.selector.select_algorithm(characteristics)
        
        self.assertIn(algorithm, ['roulette', 'rank'])
    
    def test_select_algorithm_high_diversity(self):
        """Test algorithm selection for high diversity"""
        characteristics = {
            'population_size': 100,
            'diversity_score': 0.8
        }
        
        algorithm = self.selector.select_algorithm(characteristics)
        
        self.assertIn(algorithm, ['tournament', 'diversity'])
    
    def test_select_algorithm_low_diversity(self):
        """Test algorithm selection for low diversity"""
        characteristics = {
            'population_size': 100,
            'diversity_score': 0.3
        }
        
        algorithm = self.selector.select_algorithm(characteristics)
        
        self.assertIn(algorithm, ['elitism', 'uniform'])
    
    def test_select_algorithm_with_performance_history(self):
        """Test algorithm selection with performance history"""
        # Set up performance history
        self.selector.performance_history = {
            'elitism': 0.9,
            'tournament': 0.7,
            'roulette': 0.8
        }
        
        characteristics = {
            'population_size': 30,  # Small population -> elitism, tournament
            'diversity_score': 0.5
        }
        
        algorithm = self.selector.select_algorithm(characteristics)
        
        # Should select elitism (highest performance in small_population category)
        self.assertEqual(algorithm, 'elitism')
    
    def test_select_algorithm_missing_characteristics(self):
        """Test algorithm selection with missing characteristics"""
        characteristics = {}
        
        algorithm = self.selector.select_algorithm(characteristics)
        
        # Should use defaults and select from low_diversity category
        self.assertIn(algorithm, ['elitism', 'uniform'])
    
    def test_update_performance_new_algorithm(self):
        """Test updating performance for new algorithm"""
        self.selector.update_performance('new_algorithm', 0.85)
        
        self.assertEqual(self.selector.performance_history['new_algorithm'], 0.85)
    
    def test_update_performance_existing_algorithm(self):
        """Test updating performance for existing algorithm"""
        # Set initial performance
        self.selector.performance_history['tournament'] = 0.8
        
        # Update with new performance
        self.selector.update_performance('tournament', 0.9)
        
        # Should use exponential moving average
        # Expected: 0.1 * 0.9 + 0.9 * 0.8 = 0.81
        expected = 0.1 * 0.9 + 0.9 * 0.8
        self.assertAlmostEqual(self.selector.performance_history['tournament'], expected, places=5)


class TestOptimizationIntegration(unittest.TestCase):
    """Test integration between optimization components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = GAHyperparameterOptimizer({'max_evaluations': 5})
        self.tuner = GAParameterTuner()
        self.selector = GAAlgorithmSelector()
    
    @patch('time.time')
    def test_optimizer_tuner_integration(self, mock_time):
        """Test integration between optimizer and tuner"""
        mock_time.side_effect = [i * 0.1 for i in range(100)]
        
        # Create hyperparameter space
        param_ranges = {
            'mutation_rate': ParameterRange(0.05, 0.2, values=[0.05, 0.1, 0.15, 0.2]),
            'population_size': ParameterRange(50, 150, values=[50, 100, 150])
        }
        space = HyperparameterSpace(parameter_ranges=param_ranges)
        
        # Optimize parameters
        def mock_objective(params):
            return params['mutation_rate'] + params['population_size'] / 1000
        
        result = self.optimizer.optimize(space, OptimizationMethod.RANDOM_SEARCH, mock_objective)
        
        # Use optimized parameters in tuner
        self.tuner.current_parameters.update(result.best_parameters)
        
        # Test that tuner can work with optimized parameters
        mock_stats = Mock(spec=GAStatistics)
        mock_stats.generation = 10
        mock_stats.diversity_score = 0.5
        
        adapted_params = self.tuner.adapt_parameters(mock_stats)
        
        self.assertIsInstance(adapted_params, dict)
        self.assertIn('mutation_rate', adapted_params)
        self.assertIn('population_size', adapted_params)
    
    def test_selector_tuner_integration(self):
        """Test integration between selector and tuner"""
        # Select algorithm based on current parameters
        characteristics = {
            'population_size': self.tuner.current_parameters['population_size'],
            'diversity_score': 0.6
        }
        
        selected_algorithm = self.selector.select_algorithm(characteristics)
        
        # Update algorithm performance
        self.selector.update_performance(selected_algorithm, 0.85)
        
        # Verify algorithm was selected and performance updated
        self.assertIn(selected_algorithm, self.selector.performance_history)
        self.assertEqual(self.selector.performance_history[selected_algorithm], 0.85)


if __name__ == '__main__':
    unittest.main()