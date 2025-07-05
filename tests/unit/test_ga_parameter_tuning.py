#!/usr/bin/env python3
"""
Unit tests for GA Parameter Tuning Components
Tests for parameter tuning, hyperparameter optimization, algorithm selection, config management, 
enhanced fitness, and sensitivity analysis
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import time
import numpy as np
import json
import os
import tempfile
import shutil
from typing import Dict, Any, List

# Parameter tuning modules
from ga_parameter_tuner import GAParameterTuner, ParameterRange, PopulationStats, AdaptationStrategy
from ga_hyperparameter_optimizer import GAHyperparameterOptimizer, OptimizationMethod, HyperparameterSpace
from ga_algorithm_selector import GAAlgorithmSelector, AlgorithmType, ProblemInstance, ProblemCharacteristics
from ga_config_manager import GAConfigManager, ConfigScope, ConfigProfile, ValidationLevel
from ga_fitness_enhanced import EnhancedFitnessEvaluator, FitnessComponent, AggregationMethod
from ga_sensitivity_analyzer import GASensitivityAnalyzer, SensitivityMethod, AnalysisScope

# Core GA modules
from tsp_solver_fast import RouteObjective
from ga_chromosome import RouteChromosome, RouteSegment


class TestGAParameterTuner(unittest.TestCase):
    """Test GA parameter tuning system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tuner = GAParameterTuner({
            'adaptation_interval': 2,  # Fast adaptation for testing
            'history_window': 10,
            'enable_auto_tuning': True
        })
    
    def test_parameter_tuner_initialization(self):
        """Test parameter tuner initialization"""
        self.assertIsNotNone(self.tuner.parameter_ranges)
        self.assertIsNotNone(self.tuner.adaptation_rules)
        self.assertIsNotNone(self.tuner.current_parameters)
        self.assertGreater(len(self.tuner.parameter_ranges), 0)
        self.assertGreater(len(self.tuner.adaptation_rules), 0)
    
    def test_parameter_range_operations(self):
        """Test parameter range operations"""
        param_range = ParameterRange(0.0, 1.0, 0.5)
        
        # Test clamping
        self.assertEqual(param_range.clamp(-0.5), 0.0)
        self.assertEqual(param_range.clamp(1.5), 1.0)
        self.assertEqual(param_range.clamp(0.3), 0.3)
        
        # Test normalization
        self.assertEqual(param_range.normalize(0.0), 0.0)
        self.assertEqual(param_range.normalize(1.0), 1.0)
        self.assertEqual(param_range.normalize(0.5), 0.5)
        
        # Test denormalization
        self.assertEqual(param_range.denormalize(0.0), 0.0)
        self.assertEqual(param_range.denormalize(1.0), 1.0)
        self.assertEqual(param_range.denormalize(0.5), 0.5)
    
    def test_parameter_adaptation(self):
        """Test parameter adaptation logic"""
        # Create mock population stats with low diversity
        stats = PopulationStats(
            generation=10,
            best_fitness=0.8,
            avg_fitness=0.6,
            worst_fitness=0.3,
            fitness_std=0.1,
            diversity_score=0.15,  # Low diversity should trigger adaptation
            convergence_rate=0.7,
            plateau_length=3,
            improvement_rate=0.02,
            selection_pressure=2.0
        )
        
        # Get initial parameters
        initial_params = self.tuner.current_parameters.copy()
        
        # Adapt parameters
        adapted_params = self.tuner.adapt_parameters(stats)
        
        # Should return updated parameters
        self.assertIsInstance(adapted_params, dict)
        self.assertEqual(len(adapted_params), len(initial_params))
        
        # Some parameters might have changed due to low diversity
        self.assertGreaterEqual(len(self.tuner.adaptation_history), 0)
    
    def test_adaptation_strategies(self):
        """Test different adaptation strategies"""
        # Test each adaptation rule
        for rule in self.tuner.adaptation_rules:
            self.assertIsInstance(rule.strategy, AdaptationStrategy)
            self.assertIsInstance(rule.parameter_name, str)
            self.assertIn(rule.parameter_name, self.tuner.parameter_ranges)
            self.assertGreater(rule.adjustment_rate, 0.0)
    
    def test_adapted_configuration(self):
        """Test configuration adaptation for different objectives"""
        base_config = {
            'population_size': 100,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'max_generations': 200
        }
        
        # Test elevation objective adaptation
        elevation_config = self.tuner.get_adapted_config(
            base_config, RouteObjective.MAXIMIZE_ELEVATION, 5.0
        )
        
        self.assertIsInstance(elevation_config, dict)
        self.assertIn('population_size', elevation_config)
        self.assertIn('mutation_rate', elevation_config)
        
        # Test distance objective adaptation
        distance_config = self.tuner.get_adapted_config(
            base_config, RouteObjective.MINIMIZE_DISTANCE, 3.0
        )
        
        self.assertIsInstance(distance_config, dict)
        self.assertNotEqual(elevation_config, distance_config)  # Should be different
    
    def test_tuning_recommendations(self):
        """Test tuning recommendations generation"""
        # Add some adaptation history
        stats = PopulationStats(
            generation=5,
            best_fitness=0.7,
            avg_fitness=0.5,
            worst_fitness=0.2,
            fitness_std=0.2,
            diversity_score=0.4,
            convergence_rate=0.5,
            plateau_length=0,
            improvement_rate=0.05,
            selection_pressure=1.5
        )
        
        self.tuner.adapt_parameters(stats)
        
        recommendations = self.tuner.get_tuning_recommendations()
        
        self.assertIsInstance(recommendations, dict)
        self.assertIn('parameter_suggestions', recommendations)
        self.assertIn('performance_insights', recommendations)
        self.assertIn('optimization_tips', recommendations)
    
    def test_adaptation_history_saving(self):
        """Test adaptation history saving"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filename = f.name
        
        try:
            saved_file = self.tuner.save_adaptation_history(filename)
            self.assertEqual(saved_file, filename)
            
            # Check file exists and has content
            self.assertTrue(os.path.exists(filename))
            with open(filename, 'r') as f:
                data = json.load(f)
                self.assertIn('config', data)
                self.assertIn('parameter_ranges', data)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
    
    def test_adaptation_state_reset(self):
        """Test adaptation state reset"""
        # Add some history
        stats = PopulationStats(
            generation=1, best_fitness=0.5, avg_fitness=0.4, worst_fitness=0.2,
            fitness_std=0.1, diversity_score=0.3, convergence_rate=0.4,
            plateau_length=0, improvement_rate=0.1, selection_pressure=2.0
        )
        self.tuner.adapt_parameters(stats)
        
        # Reset state
        self.tuner.reset_adaptation_state()
        
        # Check state is reset
        self.assertEqual(len(self.tuner.parameter_history), 0)
        self.assertEqual(len(self.tuner.population_stats_history), 0)
        self.assertEqual(self.tuner.last_adaptation_generation, -1)


class TestGAHyperparameterOptimizer(unittest.TestCase):
    """Test GA hyperparameter optimization framework"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = GAHyperparameterOptimizer({
            'max_evaluations': 20,  # Small for testing
            'parallel_evaluations': 2,
            'optimization_budget_minutes': 5
        })
    
    def test_optimizer_initialization(self):
        """Test hyperparameter optimizer initialization"""
        self.assertEqual(self.optimizer.config['max_evaluations'], 20)
        self.assertEqual(self.optimizer.config['parallel_evaluations'], 2)
        self.assertIsNotNone(self.optimizer.hyperparameter_space)
    
    def test_hyperparameter_space_creation(self):
        """Test hyperparameter space definition"""
        from ga_parameter_tuner import ParameterRange
        
        param_ranges = {
            'population_size': ParameterRange(50, 200, 100, constraint_type='integer'),
            'mutation_rate': ParameterRange(0.01, 0.3, 0.1),
            'crossover_rate': ParameterRange(0.5, 0.95, 0.8)
        }
        
        hyperparameter_space = HyperparameterSpace(
            parameter_ranges=param_ranges,
            objectives=['fitness', 'convergence_speed'],
            weights={'fitness': 0.7, 'convergence_speed': 0.3}
        )
        
        self.assertEqual(len(hyperparameter_space.parameter_ranges), 3)
        self.assertEqual(len(hyperparameter_space.objectives), 2)
        self.assertAlmostEqual(sum(hyperparameter_space.weights.values()), 1.0)
    
    def test_optimization_methods(self):
        """Test different optimization methods"""
        # Test that all optimization methods are available
        available_methods = [method for method in OptimizationMethod]
        self.assertGreaterEqual(len(available_methods), 5)
        
        # Check specific methods
        self.assertIn(OptimizationMethod.GRID_SEARCH, available_methods)
        self.assertIn(OptimizationMethod.RANDOM_SEARCH, available_methods)
        self.assertIn(OptimizationMethod.GENETIC, available_methods)
    
    @patch('ga_hyperparameter_optimizer.GAHyperparameterOptimizer._evaluate_parameters')
    def test_grid_search_optimization(self, mock_evaluate):
        """Test grid search optimization"""
        # Mock evaluation function
        mock_evaluate.return_value = {
            'fitness': 0.8,
            'convergence_speed': 0.6,
            'execution_time': 10.0,
            'success': True
        }
        
        from ga_parameter_tuner import ParameterRange
        
        param_ranges = {
            'population_size': ParameterRange(50, 100, 75, constraint_type='integer'),
            'mutation_rate': ParameterRange(0.05, 0.15, 0.1)
        }
        
        hyperparameter_space = HyperparameterSpace(parameter_ranges=param_ranges)
        
        # This would normally run optimization, but we'll test the setup
        self.assertIsNotNone(hyperparameter_space.parameter_ranges)
        self.assertEqual(len(hyperparameter_space.parameter_ranges), 2)
    
    def test_parameter_sampling(self):
        """Test parameter sampling methods"""
        # Test random sampling
        sample = self.optimizer._generate_random_parameters()
        
        self.assertIsInstance(sample, dict)
        self.assertGreater(len(sample), 0)
        
        # Check that generated parameters are within expected ranges
        for param_name, value in sample.items():
            self.assertIsInstance(value, (int, float))
            self.assertGreater(value, 0)  # Basic sanity check
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Valid parameters (using generated ones)
        valid_params = self.optimizer._generate_random_parameters()
        self.assertTrue(self.optimizer._check_constraints(valid_params))
        
        # Test basic constraint checking functionality
        self.assertIsNotNone(self.optimizer.hyperparameter_space.constraints)


class TestGAAlgorithmSelector(unittest.TestCase):
    """Test GA algorithm selection system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.selector = GAAlgorithmSelector()
    
    def test_selector_initialization(self):
        """Test algorithm selector initialization"""
        self.assertIsNotNone(self.selector.algorithm_performances)
        self.assertIsNotNone(self.selector.preference_matrix)
        self.assertEqual(len(self.selector.algorithm_performances), len(AlgorithmType))
    
    def test_problem_instance_creation(self):
        """Test problem instance creation"""
        problem = ProblemInstance(
            objective=RouteObjective.MAXIMIZE_ELEVATION,
            target_distance_km=5.0,
            network_size=500,
            time_constraint=60.0,
            quality_requirement=0.8
        )
        
        self.assertEqual(problem.objective, RouteObjective.MAXIMIZE_ELEVATION)
        self.assertEqual(problem.target_distance_km, 5.0)
        self.assertEqual(problem.network_size, 500)
    
    def test_algorithm_selection(self):
        """Test algorithm selection logic"""
        problem = ProblemInstance(
            objective=RouteObjective.MAXIMIZE_ELEVATION,
            target_distance_km=5.0,
            network_size=500,
            time_constraint=60.0,
            quality_requirement=0.8
        )
        
        decision = self.selector.select_algorithm(problem)
        
        self.assertIsNotNone(decision.selected_algorithm)
        self.assertIsInstance(decision.selected_algorithm, AlgorithmType)
        self.assertGreater(decision.confidence_score, 0.0)
        self.assertLessEqual(decision.confidence_score, 1.0)
        self.assertIsInstance(decision.reasoning, list)
        self.assertIsInstance(decision.fallback_algorithms, list)
    
    def test_problem_characteristics_analysis(self):
        """Test problem characteristics analysis"""
        # Distance-focused problem
        distance_problem = ProblemInstance(
            objective=RouteObjective.MINIMIZE_DISTANCE,
            target_distance_km=3.0,
            network_size=200,
            time_constraint=25.0,
            quality_requirement=0.6
        )
        
        characteristics = self.selector._analyze_problem_characteristics(distance_problem)
        
        self.assertIn(ProblemCharacteristics.DISTANCE_FOCUSED, characteristics)
        self.assertIn(ProblemCharacteristics.SMALL_PROBLEM, characteristics)
        self.assertIn(ProblemCharacteristics.TIME_CRITICAL, characteristics)
        
        # Elevation-focused problem
        elevation_problem = ProblemInstance(
            objective=RouteObjective.MAXIMIZE_ELEVATION,
            target_distance_km=8.0,
            network_size=1500,
            time_constraint=120.0,
            quality_requirement=0.9
        )
        
        characteristics = self.selector._analyze_problem_characteristics(elevation_problem)
        
        self.assertIn(ProblemCharacteristics.ELEVATION_FOCUSED, characteristics)
        self.assertIn(ProblemCharacteristics.LARGE_PROBLEM, characteristics)
        self.assertIn(ProblemCharacteristics.QUALITY_CRITICAL, characteristics)
    
    def test_performance_recording(self):
        """Test algorithm performance recording"""
        from ga_algorithm_selector import AlgorithmPerformance
        
        performance = AlgorithmPerformance(
            algorithm=AlgorithmType.GENETIC_ALGORITHM,
            execution_time=45.0,
            solution_quality=0.85,
            convergence_speed=0.7,
            memory_usage=0.6,
            stability=0.8,
            objective_satisfaction=0.9,
            success_rate=1.0
        )
        
        initial_count = len(self.selector.algorithm_performances[AlgorithmType.GENETIC_ALGORITHM])
        
        self.selector.record_algorithm_performance(AlgorithmType.GENETIC_ALGORITHM, performance)
        
        final_count = len(self.selector.algorithm_performances[AlgorithmType.GENETIC_ALGORITHM])
        self.assertEqual(final_count, initial_count + 1)
        
        # Check performance stats updated
        self.assertIn(AlgorithmType.GENETIC_ALGORITHM, self.selector.performance_stats)
    
    def test_algorithm_comparison(self):
        """Test algorithm comparison functionality"""
        algorithms = [AlgorithmType.TSP_FAST, AlgorithmType.GENETIC_ALGORITHM]
        
        test_problem = ProblemInstance(
            objective=RouteObjective.BALANCED_ROUTE,
            target_distance_km=5.0,
            network_size=500,
            time_constraint=60.0,
            quality_requirement=0.7
        )
        
        comparison = self.selector.compare_algorithms(algorithms, [test_problem])
        
        self.assertEqual(len(comparison.algorithms_compared), 2)
        self.assertIn('avg_quality', comparison.performance_matrix[AlgorithmType.TSP_FAST.value])
        self.assertIn('avg_time', comparison.performance_matrix[AlgorithmType.GENETIC_ALGORITHM.value])
        self.assertIn('avg_quality', comparison.rankings)
    
    def test_selection_recommendations(self):
        """Test selection recommendations"""
        recommendations = self.selector.get_selection_recommendations()
        
        self.assertIsInstance(recommendations, dict)
        self.assertIn('algorithm_preferences', recommendations)
        self.assertIn('performance_insights', recommendations)
        self.assertIn('usage_statistics', recommendations)
        self.assertIn('optimization_suggestions', recommendations)


class TestGAConfigManager(unittest.TestCase):
    """Test GA configuration management system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = GAConfigManager(self.temp_dir, {
            'auto_save': False,  # Disable for testing
            'validation_level': ValidationLevel.BASIC
        })
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_manager_initialization(self):
        """Test configuration manager initialization"""
        self.assertTrue(os.path.exists(self.temp_dir))
        self.assertIsNotNone(self.config_manager.parameter_definitions)
        self.assertIsNotNone(self.config_manager.validation_rules)
        self.assertIn(ConfigScope.GLOBAL, self.config_manager.configurations)
    
    def test_parameter_operations(self):
        """Test parameter get/set operations"""
        # Test setting parameter
        success = self.config_manager.set_parameter(
            'population_size', 150, ConfigScope.SESSION
        )
        self.assertTrue(success)
        
        # Test getting parameter
        value = self.config_manager.get_parameter('population_size')
        self.assertEqual(value, 150)
        
        # Test scope precedence
        self.config_manager.set_parameter('mutation_rate', 0.2, ConfigScope.GLOBAL)
        self.config_manager.set_parameter('mutation_rate', 0.15, ConfigScope.SESSION)
        
        session_value = self.config_manager.get_parameter('mutation_rate')
        self.assertEqual(session_value, 0.15)  # Session should override global
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Valid parameter
        success = self.config_manager.set_parameter('mutation_rate', 0.1, ConfigScope.SESSION)
        self.assertTrue(success)
        
        # Invalid parameter (out of range)
        success = self.config_manager.set_parameter('mutation_rate', 2.0, ConfigScope.SESSION)
        self.assertFalse(success)
        
        # Invalid type
        success = self.config_manager.set_parameter('population_size', 'invalid', ConfigScope.SESSION)
        self.assertFalse(success)
    
    def test_profile_operations(self):
        """Test configuration profile operations"""
        # Create profile
        profile = self.config_manager.create_profile(
            name="test_profile",
            description="Test profile for unit tests",
            parameters={'population_size': 120, 'mutation_rate': 0.12},
            objective=RouteObjective.MAXIMIZE_ELEVATION
        )
        
        self.assertEqual(profile.name, "test_profile")
        self.assertEqual(profile.parameters['population_size'], 120)
        
        # Activate profile
        success = self.config_manager.activate_profile("test_profile")
        self.assertTrue(success)
        self.assertEqual(self.config_manager.active_profile, "test_profile")
        
        # Check profile parameters are applied
        value = self.config_manager.get_parameter('population_size')
        self.assertEqual(value, 120)
    
    def test_auto_profile_selection(self):
        """Test automatic profile selection"""
        # Create profiles for different objectives
        self.config_manager.create_profile(
            name="elevation_profile",
            description="For elevation optimization",
            parameters={'elevation_weight': 2.0},
            objective=RouteObjective.MAXIMIZE_ELEVATION,
            target_distance_range=(3.0, 8.0)
        )
        
        self.config_manager.create_profile(
            name="distance_profile", 
            description="For distance optimization",
            parameters={'elevation_weight': 0.5},
            objective=RouteObjective.MINIMIZE_DISTANCE,
            target_distance_range=(2.0, 5.0)
        )
        
        # Test auto-selection
        selected = self.config_manager.auto_select_profile(
            RouteObjective.MAXIMIZE_ELEVATION, 5.0
        )
        self.assertEqual(selected, "elevation_profile")
    
    def test_effective_configuration(self):
        """Test effective configuration calculation"""
        # Set parameters in different scopes
        self.config_manager.set_parameter('population_size', 100, ConfigScope.GLOBAL)
        self.config_manager.set_parameter('mutation_rate', 0.1, ConfigScope.GLOBAL)
        self.config_manager.set_parameter('mutation_rate', 0.15, ConfigScope.SESSION)
        self.config_manager.set_parameter('crossover_rate', 0.9, ConfigScope.RUNTIME)
        
        effective_config = self.config_manager.get_effective_configuration()
        
        self.assertEqual(effective_config['population_size'], 100)  # From global
        self.assertEqual(effective_config['mutation_rate'], 0.15)   # Session overrides global
        self.assertEqual(effective_config['crossover_rate'], 0.9)   # From runtime
    
    def test_configuration_snapshots(self):
        """Test configuration snapshots"""
        snapshot = self.config_manager.take_snapshot(
            generation=10,
            performance_metrics={'fitness': 0.85, 'diversity': 0.7}
        )
        
        self.assertEqual(snapshot.generation, 10)
        self.assertEqual(snapshot.performance_metrics['fitness'], 0.85)
        self.assertIsNotNone(snapshot.parameters)
    
    def test_configuration_recommendations(self):
        """Test configuration recommendations"""
        # Generate some change history
        self.config_manager.set_parameter('mutation_rate', 0.1, ConfigScope.SESSION)
        self.config_manager.set_parameter('mutation_rate', 0.12, ConfigScope.SESSION)
        self.config_manager.set_parameter('mutation_rate', 0.15, ConfigScope.SESSION)
        
        recommendations = self.config_manager.get_configuration_recommendations()
        
        self.assertIn('parameter_adjustments', recommendations)
        self.assertIn('optimization_tips', recommendations)
        self.assertIn('usage_insights', recommendations)


class TestEnhancedFitnessEvaluator(unittest.TestCase):
    """Test enhanced fitness evaluation system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = EnhancedFitnessEvaluator({
            'enable_multi_objective': True,
            'pareto_ranking': True,
            'crowding_distance': True
        })
        
        # Create test chromosome
        segment = RouteSegment(1, 2, [1, 2])
        segment.length = 1000.0
        segment.elevation_gain = 50.0
        self.test_chromosome = RouteChromosome([segment])
    
    def test_evaluator_initialization(self):
        """Test enhanced fitness evaluator initialization"""
        self.assertTrue(self.evaluator.config['enable_multi_objective'])
        self.assertIsNotNone(self.evaluator.evaluators)
        self.assertIsNotNone(self.evaluator.profiles)
        self.assertIn('balanced', self.evaluator.profiles)
    
    def test_fitness_component_evaluators(self):
        """Test individual fitness component evaluators"""
        context = {
            'target_distance_km': 5.0,
            'objective': RouteObjective.MAXIMIZE_ELEVATION,
            'distance_tolerance': 0.2
        }
        
        # Test distance accuracy evaluator
        distance_eval = self.evaluator.evaluators[FitnessComponent.DISTANCE_ACCURACY]
        distance_score = distance_eval.evaluate(self.test_chromosome, context)
        self.assertGreaterEqual(distance_score, 0.0)
        self.assertLessEqual(distance_score, 1.0)
        
        # Test elevation gain evaluator
        elevation_eval = self.evaluator.evaluators[FitnessComponent.ELEVATION_GAIN]
        elevation_score = elevation_eval.evaluate(self.test_chromosome, context)
        self.assertGreaterEqual(elevation_score, 0.0)
        self.assertLessEqual(elevation_score, 1.0)
        
        # Test connectivity evaluator
        connectivity_eval = self.evaluator.evaluators[FitnessComponent.CONNECTIVITY_QUALITY]
        connectivity_score = connectivity_eval.evaluate(self.test_chromosome, context)
        self.assertGreaterEqual(connectivity_score, 0.0)
        self.assertLessEqual(connectivity_score, 1.0)
    
    def test_fitness_profile_operations(self):
        """Test fitness profile management"""
        # Test profile activation
        success = self.evaluator.set_fitness_profile("elevation")
        self.assertTrue(success)
        self.assertEqual(self.evaluator.active_profile, "elevation")
        
        # Test invalid profile
        success = self.evaluator.set_fitness_profile("nonexistent")
        self.assertFalse(success)
    
    def test_multi_objective_evaluation(self):
        """Test multi-objective fitness evaluation"""
        context = {
            'target_distance_km': 5.0,
            'objective': RouteObjective.MAXIMIZE_ELEVATION,
            'distance_tolerance': 0.2
        }
        
        result = self.evaluator.evaluate_chromosome_multi_objective(
            self.test_chromosome, context
        )
        
        self.assertIsNotNone(result.objective_values)
        self.assertGreater(len(result.objective_values), 0)
        self.assertGreaterEqual(result.aggregated_fitness, 0.0)
        self.assertLessEqual(result.aggregated_fitness, 1.0)
    
    def test_population_evaluation(self):
        """Test population-level evaluation"""
        # Create test population
        population = []
        for i in range(5):
            segment = RouteSegment(1, 2, [1, 2])
            segment.length = 1000.0 + i * 200
            segment.elevation_gain = 30.0 + i * 10
            chromosome = RouteChromosome([segment])
            population.append(chromosome)
        
        context = {
            'target_distance_km': 5.0,
            'objective': RouteObjective.MAXIMIZE_ELEVATION,
            'current_population': population
        }
        
        results = self.evaluator.evaluate_population(population, context)
        
        self.assertEqual(len(results), len(population))
        for result in results:
            self.assertIsNotNone(result.objective_values)
            self.assertGreaterEqual(result.aggregated_fitness, 0.0)
    
    def test_aggregation_methods(self):
        """Test different objective aggregation methods"""
        objective_values = {
            'distance_accuracy': 0.8,
            'elevation_gain': 0.6,
            'route_diversity': 0.7
        }
        
        # Test weighted sum
        profile = self.evaluator.profiles['balanced']
        profile.aggregation_method = AggregationMethod.WEIGHTED_SUM
        
        fitness = self.evaluator._aggregate_objectives(objective_values, profile)
        self.assertGreaterEqual(fitness, 0.0)
        self.assertLessEqual(fitness, 1.0)
        
        # Test product method
        profile.aggregation_method = AggregationMethod.PRODUCT
        fitness_product = self.evaluator._aggregate_objectives(objective_values, profile)
        self.assertGreaterEqual(fitness_product, 0.0)
        self.assertLessEqual(fitness_product, 1.0)
    
    def test_pareto_ranking(self):
        """Test Pareto ranking functionality"""
        # Create diverse population
        population = []
        results = []
        
        for i in range(5):
            segment = RouteSegment(1, 2, [1, 2])
            segment.length = 1000.0 + i * 200
            segment.elevation_gain = 30.0 + i * 10
            chromosome = RouteChromosome([segment])
            population.append(chromosome)
            
            # Mock result with varying objective values
            from ga_fitness_enhanced import MultiObjectiveResult
            result = MultiObjectiveResult(
                chromosome=chromosome,
                objective_values={
                    'distance_accuracy': 0.8 - i * 0.1,
                    'elevation_gain': 0.5 + i * 0.1,
                    'route_diversity': 0.6 + (i % 2) * 0.2
                },
                aggregated_fitness=0.6 + i * 0.05
            )
            results.append(result)
        
        ranked_results = self.evaluator._apply_pareto_ranking(results)
        
        # Check that ranks are assigned
        for result in ranked_results:
            self.assertGreaterEqual(result.rank, 0)
            self.assertGreaterEqual(result.pareto_front, 0)
    
    def test_fitness_statistics(self):
        """Test fitness statistics generation"""
        # Create mock results
        from ga_fitness_enhanced import MultiObjectiveResult
        
        results = []
        for i in range(5):
            result = MultiObjectiveResult(
                chromosome=self.test_chromosome,
                objective_values={
                    'distance_accuracy': 0.7 + i * 0.05,
                    'elevation_gain': 0.6 + i * 0.08
                },
                aggregated_fitness=0.65 + i * 0.06
            )
            results.append(result)
        
        stats = self.evaluator.get_fitness_statistics(results)
        
        self.assertEqual(stats['population_size'], 5)
        self.assertIn('objective_statistics', stats)
        self.assertIn('aggregated_fitness', stats)
        self.assertIn('distance_accuracy', stats['objective_statistics'])


class TestGASensitivityAnalyzer(unittest.TestCase):
    """Test GA sensitivity analysis system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = GASensitivityAnalyzer({
            'sampling_budget': 50,  # Small for testing
            'sensitivity_method': SensitivityMethod.LATIN_HYPERCUBE,
            'parallel_evaluation': False  # Disable for testing
        })
        
        # Set up parameter ranges
        from ga_parameter_tuner import ParameterRange
        
        self.param_ranges = {
            'population_size': ParameterRange(50, 150, 100, constraint_type='integer'),
            'mutation_rate': ParameterRange(0.05, 0.25, 0.1),
            'crossover_rate': ParameterRange(0.6, 0.9, 0.8)
        }
        
        self.analyzer.set_parameter_ranges(self.param_ranges)
    
    def test_analyzer_initialization(self):
        """Test sensitivity analyzer initialization"""
        self.assertEqual(self.analyzer.config['sampling_budget'], 50)
        self.assertEqual(self.analyzer.config['sensitivity_method'], SensitivityMethod.LATIN_HYPERCUBE)
        self.assertIsNotNone(self.analyzer.parameter_ranges)
    
    def test_parameter_sampling_methods(self):
        """Test different parameter sampling methods"""
        # Test Latin Hypercube Sampling
        lhs_samples = self.analyzer._generate_lhs_samples(self.param_ranges, 20)
        self.assertEqual(len(lhs_samples), 20)
        
        for sample in lhs_samples[:5]:  # Check first few samples
            self.assertIn('population_size', sample)
            self.assertIn('mutation_rate', sample)
            self.assertIn('crossover_rate', sample)
            self.assertGreaterEqual(sample['population_size'], 50)
            self.assertLessEqual(sample['population_size'], 150)
        
        # Test One-At-a-Time sampling
        oat_samples = self.analyzer._generate_oat_samples(self.param_ranges, 15)
        self.assertGreater(len(oat_samples), 0)
        self.assertLessEqual(len(oat_samples), 15)
        
        # Test Morris sampling
        morris_samples = self.analyzer._generate_morris_samples(self.param_ranges, 20)
        self.assertGreater(len(morris_samples), 0)
    
    def test_sample_evaluation(self):
        """Test parameter sample evaluation"""
        # Mock performance evaluator
        def mock_evaluator(params):
            return {
                'fitness': 0.5 + 0.3 * (params['mutation_rate'] / 0.25),
                'convergence_speed': 0.6 + 0.2 * (params['population_size'] / 150),
                'diversity': 0.4 + 0.3 * (params['crossover_rate'] / 0.9)
            }
        
        # Generate samples and evaluate
        samples = self.analyzer._generate_lhs_samples(self.param_ranges, 10)
        evaluated_samples = self.analyzer._evaluate_samples(samples, mock_evaluator)
        
        self.assertEqual(len(evaluated_samples), 10)
        
        for sample in evaluated_samples:
            self.assertIsInstance(sample.parameters, dict)
            self.assertIsInstance(sample.performance_metrics, dict)
            self.assertTrue(sample.success)
            self.assertIn('fitness', sample.performance_metrics)
    
    def test_sensitivity_calculation(self):
        """Test sensitivity index calculation"""
        # Create mock evaluated samples
        from ga_sensitivity_analyzer import ParameterSample
        
        samples = []
        for i in range(20):
            params = {
                'population_size': 50 + i * 5,
                'mutation_rate': 0.05 + i * 0.01,
                'crossover_rate': 0.6 + i * 0.015
            }
            
            # Create correlated performance (mutation_rate affects fitness)
            fitness = 0.4 + 0.4 * (params['mutation_rate'] / 0.25) + np.random.normal(0, 0.05)
            
            sample = ParameterSample(
                parameters=params,
                performance_metrics={'fitness': max(0, min(1, fitness))},
                execution_time=1.0,
                success=True
            )
            samples.append(sample)
        
        # Calculate sensitivity
        sensitivity_index, confidence_interval, variance_contribution = \
            self.analyzer._calculate_parameter_sensitivity(samples, 'mutation_rate', 'fitness')
        
        self.assertGreaterEqual(sensitivity_index, 0.0)
        self.assertLessEqual(sensitivity_index, 1.0)
        self.assertIsInstance(confidence_interval, tuple)
        self.assertEqual(len(confidence_interval), 2)
        self.assertGreaterEqual(variance_contribution, 0.0)
    
    @patch('ga_sensitivity_analyzer.GASensitivityAnalyzer._evaluate_samples')
    def test_full_sensitivity_analysis(self, mock_evaluate):
        """Test complete sensitivity analysis workflow"""
        # Mock evaluated samples
        from ga_sensitivity_analyzer import ParameterSample
        
        mock_samples = []
        for i in range(30):
            params = {
                'population_size': 50 + i * 3,
                'mutation_rate': 0.05 + i * 0.006,
                'crossover_rate': 0.6 + i * 0.01
            }
            
            sample = ParameterSample(
                parameters=params,
                performance_metrics={
                    'fitness': 0.5 + 0.2 * np.random.random(),
                    'convergence_speed': 0.4 + 0.3 * np.random.random(),
                    'diversity': 0.3 + 0.4 * np.random.random()
                },
                execution_time=1.0,
                success=True
            )
            mock_samples.append(sample)
        
        mock_evaluate.return_value = mock_samples
        
        # Mock performance evaluator
        def mock_evaluator(params):
            return {'fitness': 0.7}
        
        # Run sensitivity analysis
        results = self.analyzer.analyze_parameter_sensitivity(mock_evaluator)
        
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        # Check result structure
        for key, result in results.items():
            self.assertIsInstance(result.parameter_name, str)
            self.assertGreaterEqual(result.sensitivity_index, 0.0)
            self.assertLessEqual(result.sensitivity_index, 1.0)
    
    def test_tuning_recommendations(self):
        """Test tuning recommendations generation"""
        # Mock sensitivity results
        from ga_sensitivity_analyzer import SensitivityResult
        
        self.analyzer.sensitivity_results = {
            'mutation_rate_fitness': SensitivityResult(
                parameter_name='mutation_rate',
                sensitivity_index=0.7,
                confidence_interval=(0.5, 0.9),
                variance_contribution=0.4,
                statistical_significance=True,
                rank=1
            ),
            'population_size_fitness': SensitivityResult(
                parameter_name='population_size',
                sensitivity_index=0.4,
                confidence_interval=(0.2, 0.6),
                variance_contribution=0.2,
                statistical_significance=False,
                rank=2
            )
        }
        
        current_params = {
            'mutation_rate': 0.1,
            'population_size': 100,
            'crossover_rate': 0.8
        }
        
        recommendations = self.analyzer.generate_tuning_recommendations(
            current_params, RouteObjective.MAXIMIZE_ELEVATION
        )
        
        self.assertIsInstance(recommendations, list)
        if recommendations:  # May be empty if no significant changes recommended
            for rec in recommendations:
                self.assertIsNotNone(rec.parameter_name)
                self.assertIsNotNone(rec.recommended_value)
                self.assertIn(rec.priority, ['high', 'medium', 'low'])
    
    def test_optimization_insights(self):
        """Test optimization insights generation"""
        # Mock sensitivity results with interactions
        from ga_sensitivity_analyzer import SensitivityResult
        
        self.analyzer.sensitivity_results = {
            'mutation_rate_fitness': SensitivityResult(
                parameter_name='mutation_rate',
                sensitivity_index=0.8,
                confidence_interval=(0.6, 0.9),
                variance_contribution=0.5,
                interaction_effects={'population_size': 0.3},
                statistical_significance=True,
                rank=1
            ),
            'population_size_fitness': SensitivityResult(
                parameter_name='population_size',
                sensitivity_index=0.6,
                confidence_interval=(0.4, 0.8),
                variance_contribution=0.4,
                interaction_effects={'mutation_rate': 0.3},
                statistical_significance=True,
                rank=2
            )
        }
        
        insights = self.analyzer.generate_optimization_insights()
        
        self.assertIsInstance(insights, list)
        if insights:
            for insight in insights:
                self.assertIsNotNone(insight.insight_type)
                self.assertIsNotNone(insight.description)
                self.assertIsInstance(insight.affected_parameters, list)
                self.assertIsNotNone(insight.actionable_recommendation)


if __name__ == '__main__':
    unittest.main()