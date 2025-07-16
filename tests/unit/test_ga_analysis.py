#!/usr/bin/env python3
"""
Unit tests for genetic_algorithm/analysis.py
Tests comprehensive functionality of GA analysis components
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys
import os
import tempfile
import shutil
import json
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from genetic_algorithm.analysis import (
    GASensitivityAnalyzer, GAConfigManager, SensitivityResult
)
from ga_common_imports import GAConfiguration


class TestSensitivityResult(unittest.TestCase):
    """Test SensitivityResult dataclass"""
    
    def test_sensitivity_result_creation(self):
        """Test SensitivityResult creation"""
        result = SensitivityResult(
            parameter_name="mutation_rate",
            parameter_values=[0.1, 0.2, 0.3],
            fitness_scores=[100, 120, 110],
            convergence_times=[10.5, 12.3, 11.8],
            sensitivity_coefficient=0.75,
            optimal_value=0.2,
            confidence_interval=(0.15, 0.25)
        )
        
        self.assertEqual(result.parameter_name, "mutation_rate")
        self.assertEqual(result.parameter_values, [0.1, 0.2, 0.3])
        self.assertEqual(result.fitness_scores, [100, 120, 110])
        self.assertEqual(result.convergence_times, [10.5, 12.3, 11.8])
        self.assertEqual(result.sensitivity_coefficient, 0.75)
        self.assertEqual(result.optimal_value, 0.2)
        self.assertEqual(result.confidence_interval, (0.15, 0.25))
    
    def test_sensitivity_result_equality(self):
        """Test SensitivityResult equality"""
        result1 = SensitivityResult(
            parameter_name="mutation_rate",
            parameter_values=[0.1, 0.2],
            fitness_scores=[100, 120],
            convergence_times=[10.5, 12.3],
            sensitivity_coefficient=0.75,
            optimal_value=0.2,
            confidence_interval=(0.15, 0.25)
        )
        
        result2 = SensitivityResult(
            parameter_name="mutation_rate",
            parameter_values=[0.1, 0.2],
            fitness_scores=[100, 120],
            convergence_times=[10.5, 12.3],
            sensitivity_coefficient=0.75,
            optimal_value=0.2,
            confidence_interval=(0.15, 0.25)
        )
        
        self.assertEqual(result1, result2)
    
    def test_sensitivity_result_inequality(self):
        """Test SensitivityResult inequality"""
        result1 = SensitivityResult(
            parameter_name="mutation_rate",
            parameter_values=[0.1, 0.2],
            fitness_scores=[100, 120],
            convergence_times=[10.5, 12.3],
            sensitivity_coefficient=0.75,
            optimal_value=0.2,
            confidence_interval=(0.15, 0.25)
        )
        
        result2 = SensitivityResult(
            parameter_name="crossover_rate",
            parameter_values=[0.1, 0.2],
            fitness_scores=[100, 120],
            convergence_times=[10.5, 12.3],
            sensitivity_coefficient=0.75,
            optimal_value=0.2,
            confidence_interval=(0.15, 0.25)
        )
        
        self.assertNotEqual(result1, result2)


class TestGASensitivityAnalyzer(unittest.TestCase):
    """Test GASensitivityAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = GASensitivityAnalyzer()
    
    def test_analyzer_initialization_default(self):
        """Test analyzer initialization with default config"""
        analyzer = GASensitivityAnalyzer()
        
        self.assertEqual(analyzer.config['num_samples'], 20)
        self.assertEqual(analyzer.config['num_trials'], 5)
        self.assertEqual(analyzer.config['confidence_level'], 0.95)
        self.assertEqual(analyzer.config['sensitivity_threshold'], 0.1)
        self.assertIn('parameter_ranges', analyzer.config)
        self.assertIsInstance(analyzer.sensitivity_results, dict)
        self.assertIsInstance(analyzer.analysis_history, list)
    
    def test_analyzer_initialization_custom_config(self):
        """Test analyzer initialization with custom config"""
        custom_config = {
            'num_samples': 15,
            'num_trials': 3,
            'confidence_level': 0.90,
            'sensitivity_threshold': 0.05
        }
        
        analyzer = GASensitivityAnalyzer(custom_config)
        
        self.assertEqual(analyzer.config['num_samples'], 15)
        self.assertEqual(analyzer.config['num_trials'], 3)
        self.assertEqual(analyzer.config['confidence_level'], 0.90)
        self.assertEqual(analyzer.config['sensitivity_threshold'], 0.05)
        
        # Should still have default parameter ranges
        self.assertIn('parameter_ranges', analyzer.config)
    
    def test_analyzer_default_parameter_ranges(self):
        """Test analyzer default parameter ranges"""
        analyzer = GASensitivityAnalyzer()
        
        ranges = analyzer.config['parameter_ranges']
        
        self.assertIn('population_size', ranges)
        self.assertIn('mutation_rate', ranges)
        self.assertIn('crossover_rate', ranges)
        self.assertIn('elite_size', ranges)
        self.assertIn('tournament_size', ranges)
        
        # Check range formats
        self.assertIsInstance(ranges['population_size'], tuple)
        self.assertEqual(len(ranges['population_size']), 2)
        self.assertLess(ranges['population_size'][0], ranges['population_size'][1])
    
    @patch('time.time')
    def test_analyze_parameter_sensitivity_basic(self, mock_time):
        """Test basic parameter sensitivity analysis"""
        # Mock time to return predictable values - provide enough values
        mock_time.side_effect = [i * 0.1 for i in range(100)]
        
        # Mock objective function
        def mock_objective(param_name, param_value):
            # Return higher fitness for higher parameter values
            return param_value * 100
        
        # Mock numpy.linspace to return predictable values
        with patch('numpy.linspace') as mock_linspace:
            mock_linspace.return_value = np.array([0.1, 0.2, 0.3])
            
            result = self.analyzer.analyze_parameter_sensitivity(
                'mutation_rate', mock_objective, (0.1, 0.3)
            )
        
        self.assertIsInstance(result, SensitivityResult)
        self.assertEqual(result.parameter_name, 'mutation_rate')
        self.assertEqual(result.parameter_values, [0.1, 0.2, 0.3])
        self.assertEqual(len(result.fitness_scores), 3)
        self.assertEqual(len(result.convergence_times), 3)
        self.assertGreater(result.sensitivity_coefficient, 0)
        self.assertIn('mutation_rate', self.analyzer.sensitivity_results)
    
    @patch('time.time')
    def test_analyze_parameter_sensitivity_with_default_range(self, mock_time):
        """Test parameter sensitivity analysis with default range"""
        mock_time.side_effect = [i * 0.1 for i in range(100)]
        
        def mock_objective(param_name, param_value):
            return param_value * 50
        
        with patch('numpy.linspace') as mock_linspace:
            mock_linspace.return_value = np.array([0.01, 0.5, 1.0])
            
            result = self.analyzer.analyze_parameter_sensitivity(
                'unknown_parameter', mock_objective
            )
        
        self.assertIsInstance(result, SensitivityResult)
        self.assertEqual(result.parameter_name, 'unknown_parameter')
        
        # Should use default range (0.01, 1.0)
        mock_linspace.assert_called_once_with(0.01, 1.0, 20)
    
    @patch('time.time')
    def test_analyze_parameter_sensitivity_with_exceptions(self, mock_time):
        """Test parameter sensitivity analysis with objective function exceptions"""
        mock_time.side_effect = [i * 0.1 for i in range(100)]
        
        def mock_objective(param_name, param_value):
            if param_value > 0.15:
                raise ValueError("Test error")
            return param_value * 100
        
        with patch('numpy.linspace') as mock_linspace:
            mock_linspace.return_value = np.array([0.1, 0.2, 0.3])
            
            with patch.object(self.analyzer, 'logger') as mock_logger:
                result = self.analyzer.analyze_parameter_sensitivity(
                    'mutation_rate', mock_objective, (0.1, 0.3)
                )
        
        self.assertIsInstance(result, SensitivityResult)
        
        # Should have logged warnings for failed trials
        self.assertTrue(mock_logger.warning.called)
        
        # Should have some zero fitness scores for failed trials
        self.assertIn(0.0, result.fitness_scores)
    
    @patch('time.time')
    def test_analyze_all_parameters(self, mock_time):
        """Test analyzing all parameters"""
        mock_time.side_effect = [i * 0.1 for i in range(500)]  # Provide enough time values for all parameters
        
        def mock_objective(param_name, param_value):
            return param_value * 10
        
        with patch('numpy.linspace') as mock_linspace:
            mock_linspace.return_value = np.array([0.1, 0.2, 0.3])
            
            results = self.analyzer.analyze_all_parameters(mock_objective)
        
        self.assertIsInstance(results, dict)
        
        # Should have results for all configured parameters
        expected_params = self.analyzer.config['parameter_ranges'].keys()
        self.assertEqual(set(results.keys()), set(expected_params))
        
        # Each result should be a SensitivityResult
        for param_name, result in results.items():
            self.assertIsInstance(result, SensitivityResult)
            self.assertEqual(result.parameter_name, param_name)
    
    def test_calculate_sensitivity_coefficient_basic(self):
        """Test basic sensitivity coefficient calculation"""
        parameter_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        fitness_scores = [10, 20, 30, 40, 50]
        
        coeff = self.analyzer._calculate_sensitivity_coefficient(
            parameter_values, fitness_scores
        )
        
        # Should be high correlation (close to 1.0)
        self.assertGreater(coeff, 0.9)
        self.assertLessEqual(coeff, 1.0)
    
    def test_calculate_sensitivity_coefficient_no_correlation(self):
        """Test sensitivity coefficient calculation with no correlation"""
        parameter_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        fitness_scores = [30, 30, 30, 30, 30]  # Constant fitness
        
        coeff = self.analyzer._calculate_sensitivity_coefficient(
            parameter_values, fitness_scores
        )
        
        # Should be zero or very close to zero
        self.assertLess(coeff, 0.1)
    
    def test_calculate_sensitivity_coefficient_negative_correlation(self):
        """Test sensitivity coefficient calculation with negative correlation"""
        parameter_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        fitness_scores = [50, 40, 30, 20, 10]  # Inverse relationship
        
        coeff = self.analyzer._calculate_sensitivity_coefficient(
            parameter_values, fitness_scores
        )
        
        # Should be high absolute correlation
        self.assertGreater(coeff, 0.9)
        self.assertLessEqual(coeff, 1.0)
    
    def test_calculate_sensitivity_coefficient_insufficient_data(self):
        """Test sensitivity coefficient calculation with insufficient data"""
        parameter_values = np.array([0.1])
        fitness_scores = [10]
        
        coeff = self.analyzer._calculate_sensitivity_coefficient(
            parameter_values, fitness_scores
        )
        
        self.assertEqual(coeff, 0.0)
    
    def test_calculate_confidence_interval_basic(self):
        """Test basic confidence interval calculation"""
        parameter_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        fitness_scores = [10, 50, 30, 20, 40]
        
        ci = self.analyzer._calculate_confidence_interval(
            parameter_values, fitness_scores
        )
        
        self.assertIsInstance(ci, tuple)
        self.assertEqual(len(ci), 2)
        self.assertLessEqual(ci[0], ci[1])
    
    def test_calculate_confidence_interval_single_value(self):
        """Test confidence interval calculation with single value"""
        parameter_values = np.array([0.2])
        fitness_scores = [30]
        
        ci = self.analyzer._calculate_confidence_interval(
            parameter_values, fitness_scores
        )
        
        self.assertEqual(ci, (0.2, 0.2))
    
    def test_calculate_confidence_interval_insufficient_data(self):
        """Test confidence interval calculation with insufficient data"""
        parameter_values = np.array([0.1])
        fitness_scores = [10]
        
        ci = self.analyzer._calculate_confidence_interval(
            parameter_values, fitness_scores
        )
        
        self.assertEqual(ci, (0.1, 0.1))
    
    def test_get_sensitivity_ranking_basic(self):
        """Test basic sensitivity ranking"""
        # Add some mock results
        self.analyzer.sensitivity_results = {
            'mutation_rate': SensitivityResult(
                'mutation_rate', [0.1, 0.2], [10, 20], [1.0, 2.0], 0.8, 0.2, (0.15, 0.25)
            ),
            'crossover_rate': SensitivityResult(
                'crossover_rate', [0.5, 0.8], [30, 35], [1.5, 2.5], 0.6, 0.8, (0.7, 0.9)
            ),
            'population_size': SensitivityResult(
                'population_size', [50, 100], [25, 28], [3.0, 4.0], 0.9, 100, (80, 120)
            )
        }
        
        ranking = self.analyzer.get_sensitivity_ranking()
        
        self.assertIsInstance(ranking, list)
        self.assertEqual(len(ranking), 3)
        
        # Should be sorted by sensitivity coefficient (descending)
        self.assertEqual(ranking[0][0], 'population_size')  # 0.9
        self.assertEqual(ranking[1][0], 'mutation_rate')    # 0.8
        self.assertEqual(ranking[2][0], 'crossover_rate')   # 0.6
        
        # Check values
        self.assertEqual(ranking[0][1], 0.9)
        self.assertEqual(ranking[1][1], 0.8)
        self.assertEqual(ranking[2][1], 0.6)
    
    def test_get_sensitivity_ranking_empty(self):
        """Test sensitivity ranking with empty results"""
        ranking = self.analyzer.get_sensitivity_ranking()
        
        self.assertIsInstance(ranking, list)
        self.assertEqual(len(ranking), 0)
    
    def test_get_recommendations_basic(self):
        """Test basic recommendations generation"""
        # Add some mock results
        self.analyzer.sensitivity_results = {
            'mutation_rate': SensitivityResult(
                'mutation_rate', [0.1, 0.2], [10, 20], [1.0, 2.0], 0.8, 0.2, (0.15, 0.25)
            ),
            'crossover_rate': SensitivityResult(
                'crossover_rate', [0.5, 0.8], [30, 35], [1.5, 2.5], 0.2, 0.8, (0.7, 0.9)
            )
        }
        
        recommendations = self.analyzer.get_recommendations()
        
        self.assertIsInstance(recommendations, dict)
        self.assertEqual(len(recommendations), 2)
        
        # Check mutation_rate recommendation
        mut_rec = recommendations['mutation_rate']
        self.assertEqual(mut_rec['optimal_value'], 0.2)
        self.assertEqual(mut_rec['sensitivity'], 0.8)
        self.assertEqual(mut_rec['confidence_interval'], (0.15, 0.25))
        self.assertEqual(mut_rec['recommendation'], "Highly sensitive - tune carefully")
        
        # Check crossover_rate recommendation
        cross_rec = recommendations['crossover_rate']
        self.assertEqual(cross_rec['optimal_value'], 0.8)
        self.assertEqual(cross_rec['sensitivity'], 0.2)
        self.assertEqual(cross_rec['confidence_interval'], (0.7, 0.9))
        self.assertEqual(cross_rec['recommendation'], "Low sensitivity - use default value")
    
    def test_get_recommendations_empty(self):
        """Test recommendations generation with empty results"""
        recommendations = self.analyzer.get_recommendations()
        
        self.assertIsInstance(recommendations, dict)
        self.assertEqual(len(recommendations), 0)
    
    def test_get_parameter_recommendation_high_sensitivity(self):
        """Test parameter recommendation for high sensitivity"""
        result = SensitivityResult(
            'test_param', [0.1, 0.2], [10, 20], [1.0, 2.0], 0.8, 0.2, (0.15, 0.25)
        )
        
        recommendation = self.analyzer._get_parameter_recommendation(result)
        
        self.assertEqual(recommendation, "Highly sensitive - tune carefully")
    
    def test_get_parameter_recommendation_moderate_sensitivity(self):
        """Test parameter recommendation for moderate sensitivity"""
        result = SensitivityResult(
            'test_param', [0.1, 0.2], [10, 20], [1.0, 2.0], 0.5, 0.2, (0.15, 0.25)
        )
        
        recommendation = self.analyzer._get_parameter_recommendation(result)
        
        self.assertEqual(recommendation, "Moderately sensitive - consider tuning")
    
    def test_get_parameter_recommendation_low_sensitivity(self):
        """Test parameter recommendation for low sensitivity"""
        result = SensitivityResult(
            'test_param', [0.1, 0.2], [10, 20], [1.0, 2.0], 0.1, 0.2, (0.15, 0.25)
        )
        
        recommendation = self.analyzer._get_parameter_recommendation(result)
        
        self.assertEqual(recommendation, "Low sensitivity - use default value")


class TestGAConfigManager(unittest.TestCase):
    """Test GAConfigManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = GAConfigManager(config_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_config_manager_initialization(self):
        """Test GAConfigManager initialization"""
        manager = GAConfigManager(config_dir=self.temp_dir)
        
        self.assertEqual(manager.config_dir, Path(self.temp_dir))
        self.assertTrue(manager.config_dir.exists())
        self.assertIsInstance(manager.configurations, dict)
        self.assertIsNotNone(manager.active_config)
        
        # Should have default configurations
        self.assertIn('standard', manager.configurations)
        self.assertIn('fast', manager.configurations)
        self.assertIn('thorough', manager.configurations)
        self.assertIn('distance_focused', manager.configurations)
        self.assertIn('balanced', manager.configurations)
    
    def test_config_manager_default_configurations(self):
        """Test default configuration setup"""
        manager = GAConfigManager(config_dir=self.temp_dir)
        
        # Check standard configuration
        standard_config = manager.configurations['standard']
        self.assertEqual(standard_config.population_size, 100)
        self.assertEqual(standard_config.max_generations, 200)
        self.assertEqual(standard_config.mutation_rate, 0.1)
        self.assertEqual(standard_config.crossover_rate, 0.8)
        self.assertEqual(standard_config.elite_size, 2)
        self.assertEqual(standard_config.objective, "elevation")
        
        # Check fast configuration
        fast_config = manager.configurations['fast']
        self.assertEqual(fast_config.population_size, 50)
        self.assertEqual(fast_config.max_generations, 100)
        self.assertEqual(fast_config.mutation_rate, 0.15)
        
        # Check thorough configuration
        thorough_config = manager.configurations['thorough']
        self.assertEqual(thorough_config.population_size, 200)
        self.assertEqual(thorough_config.max_generations, 500)
        self.assertEqual(thorough_config.mutation_rate, 0.05)
        
        # Check distance_focused configuration
        distance_config = manager.configurations['distance_focused']
        self.assertEqual(distance_config.objective, "distance")
        
        # Check balanced configuration
        balanced_config = manager.configurations['balanced']
        self.assertEqual(balanced_config.objective, "balanced")
    
    def test_save_configuration_basic(self):
        """Test basic configuration saving"""
        config = GAConfiguration(
            population_size=150,
            max_generations=250,
            mutation_rate=0.12,
            crossover_rate=0.85,
            elite_size=3,
            objective="elevation"
        )
        
        self.config_manager.save_configuration("test_config", config)
        
        # Should be in configurations dict
        self.assertIn("test_config", self.config_manager.configurations)
        self.assertEqual(self.config_manager.configurations["test_config"], config)
        
        # Should have created file
        config_file = Path(self.temp_dir) / "test_config.json"
        self.assertTrue(config_file.exists())
        
        # Check file content
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        self.assertEqual(config_dict['population_size'], 150)
        self.assertEqual(config_dict['max_generations'], 250)
        self.assertEqual(config_dict['mutation_rate'], 0.12)
        self.assertEqual(config_dict['crossover_rate'], 0.85)
        self.assertEqual(config_dict['elite_size'], 3)
        self.assertEqual(config_dict['objective'], "elevation")
    
    def test_save_configuration_file_error(self):
        """Test configuration saving with file error"""
        config = GAConfiguration(
            population_size=100,
            max_generations=200,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elite_size=2,
            objective="elevation"
        )
        
        # Mock file operations to raise exception
        with patch('builtins.open', side_effect=OSError("Permission denied")):
            with patch.object(self.config_manager, 'logger') as mock_logger:
                self.config_manager.save_configuration("test_config", config)
        
        # Should have logged error
        mock_logger.error.assert_called_once()
        
        # Should still be in configurations dict
        self.assertIn("test_config", self.config_manager.configurations)
    
    def test_load_configuration_from_memory(self):
        """Test loading configuration from memory"""
        config = GAConfiguration(
            population_size=120,
            max_generations=180,
            mutation_rate=0.08,
            crossover_rate=0.9,
            elite_size=4,
            objective="distance"
        )
        
        # Add to configurations
        self.config_manager.configurations["memory_config"] = config
        
        loaded_config = self.config_manager.load_configuration("memory_config")
        
        self.assertEqual(loaded_config, config)
    
    def test_load_configuration_from_file(self):
        """Test loading configuration from file"""
        # Create a config file
        config_dict = {
            'population_size': 80,
            'max_generations': 150,
            'mutation_rate': 0.15,
            'crossover_rate': 0.75,
            'elite_size': 1,
            'target_distance_km': 3.0,
            'objective': "balanced"
        }
        
        config_file = Path(self.temp_dir) / "file_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_dict, f)
        
        # Load configuration
        loaded_config = self.config_manager.load_configuration("file_config")
        
        self.assertIsInstance(loaded_config, GAConfiguration)
        self.assertEqual(loaded_config.population_size, 80)
        self.assertEqual(loaded_config.max_generations, 150)
        self.assertEqual(loaded_config.mutation_rate, 0.15)
        self.assertEqual(loaded_config.crossover_rate, 0.75)
        self.assertEqual(loaded_config.elite_size, 1)
        self.assertEqual(loaded_config.target_distance_km, 3.0)
        self.assertEqual(loaded_config.objective, "balanced")
        
        # Should now be in configurations dict
        self.assertIn("file_config", self.config_manager.configurations)
    
    def test_load_configuration_file_error(self):
        """Test loading configuration with file error"""
        # Create a config file with invalid JSON
        config_file = Path(self.temp_dir) / "invalid_config.json"
        with open(config_file, 'w') as f:
            f.write("invalid json content")
        
        with patch.object(self.config_manager, 'logger') as mock_logger:
            loaded_config = self.config_manager.load_configuration("invalid_config")
        
        self.assertIsNone(loaded_config)
        mock_logger.error.assert_called_once()
    
    def test_load_configuration_not_found(self):
        """Test loading configuration that doesn't exist"""
        loaded_config = self.config_manager.load_configuration("nonexistent_config")
        
        self.assertIsNone(loaded_config)
    
    def test_load_configurations_on_init(self):
        """Test loading configurations during initialization"""
        # Create some config files
        config1_dict = {
            'population_size': 60,
            'max_generations': 120,
            'mutation_rate': 0.2,
            'crossover_rate': 0.6,
            'elite_size': 2,
            'objective': "elevation"
        }
        
        config2_dict = {
            'population_size': 140,
            'max_generations': 280,
            'mutation_rate': 0.05,
            'crossover_rate': 0.95,
            'elite_size': 6,
            'objective': "distance"
        }
        
        config1_file = Path(self.temp_dir) / "init_config1.json"
        config2_file = Path(self.temp_dir) / "init_config2.json"
        
        with open(config1_file, 'w') as f:
            json.dump(config1_dict, f)
        
        with open(config2_file, 'w') as f:
            json.dump(config2_dict, f)
        
        # Create new manager (should load existing files)
        manager = GAConfigManager(config_dir=self.temp_dir)
        
        self.assertIn("init_config1", manager.configurations)
        self.assertIn("init_config2", manager.configurations)
        
        # Check loaded values
        config1 = manager.configurations["init_config1"]
        self.assertEqual(config1.population_size, 60)
        self.assertEqual(config1.max_generations, 120)
        
        config2 = manager.configurations["init_config2"]
        self.assertEqual(config2.population_size, 140)
        self.assertEqual(config2.objective, "distance")
    
    def test_config_to_dict_conversion(self):
        """Test configuration to dictionary conversion"""
        config = GAConfiguration(
            population_size=100,
            max_generations=200,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elite_size=2,
            target_distance_km=5.0,
            objective="elevation"
        )
        
        config_dict = self.config_manager._config_to_dict(config)
        
        expected_dict = {
            'population_size': 100,
            'max_generations': 200,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'elite_size': 2,
            'target_distance_km': 5.0,
            'objective': "elevation"
        }
        
        self.assertEqual(config_dict, expected_dict)
    
    def test_dict_to_config_conversion(self):
        """Test dictionary to configuration conversion"""
        config_dict = {
            'population_size': 120,
            'max_generations': 180,
            'mutation_rate': 0.12,
            'crossover_rate': 0.85,
            'elite_size': 3,
            'target_distance_km': 4.0,
            'objective': "balanced"
        }
        
        config = self.config_manager._dict_to_config(config_dict)
        
        self.assertIsInstance(config, GAConfiguration)
        self.assertEqual(config.population_size, 120)
        self.assertEqual(config.max_generations, 180)
        self.assertEqual(config.mutation_rate, 0.12)
        self.assertEqual(config.crossover_rate, 0.85)
        self.assertEqual(config.elite_size, 3)
        self.assertEqual(config.target_distance_km, 4.0)
        self.assertEqual(config.objective, "balanced")
    
    def test_dict_to_config_with_defaults(self):
        """Test dictionary to configuration conversion with missing values"""
        config_dict = {
            'population_size': 80,
            'objective': "elevation"
        }
        
        config = self.config_manager._dict_to_config(config_dict)
        
        self.assertIsInstance(config, GAConfiguration)
        self.assertEqual(config.population_size, 80)
        self.assertEqual(config.objective, "elevation")
        
        # Should have default values for missing fields
        self.assertEqual(config.max_generations, 200)  # DEFAULT_MAX_GENERATIONS
        self.assertEqual(config.mutation_rate, 0.1)    # DEFAULT_MUTATION_RATE
        self.assertEqual(config.crossover_rate, 0.8)   # DEFAULT_CROSSOVER_RATE
        self.assertEqual(config.elite_size, 2)
        self.assertEqual(config.target_distance_km, 5.0)
    
    def test_get_configuration_existing(self):
        """Test getting existing configuration"""
        config = GAConfiguration(
            population_size=90,
            max_generations=160,
            mutation_rate=0.14,
            crossover_rate=0.78,
            elite_size=3,
            objective="elevation"
        )
        
        self.config_manager.configurations["test_get"] = config
        
        retrieved_config = self.config_manager.get_configuration("test_get")
        
        self.assertEqual(retrieved_config, config)
    
    def test_get_configuration_nonexistent(self):
        """Test getting nonexistent configuration"""
        retrieved_config = self.config_manager.get_configuration("nonexistent")
        
        self.assertIsNone(retrieved_config)
    
    def test_list_configurations(self):
        """Test listing all configurations"""
        # Add some test configurations
        config1 = GAConfiguration(population_size=50, max_generations=100)
        config2 = GAConfiguration(population_size=150, max_generations=300)
        
        self.config_manager.configurations["test1"] = config1
        self.config_manager.configurations["test2"] = config2
        
        config_list = self.config_manager.list_configurations()
        
        self.assertIsInstance(config_list, list)
        self.assertIn("test1", config_list)
        self.assertIn("test2", config_list)
        
        # Should include default configurations
        self.assertIn("standard", config_list)
        self.assertIn("fast", config_list)
        self.assertIn("thorough", config_list)
    
    def test_set_active_configuration_existing(self):
        """Test setting active configuration that exists"""
        result = self.config_manager.set_active_configuration("fast")
        
        self.assertTrue(result)
        self.assertEqual(self.config_manager.active_config, "fast")
    
    def test_set_active_configuration_nonexistent(self):
        """Test setting active configuration that doesn't exist"""
        result = self.config_manager.set_active_configuration("nonexistent")
        
        self.assertFalse(result)
        self.assertNotEqual(self.config_manager.active_config, "nonexistent")
    
    def test_get_active_configuration(self):
        """Test getting active configuration"""
        self.config_manager.set_active_configuration("thorough")
        
        active_config = self.config_manager.get_active_configuration()
        
        self.assertIsNotNone(active_config)
        self.assertEqual(active_config, self.config_manager.configurations["thorough"])
    
    def test_get_active_configuration_none_set(self):
        """Test getting active configuration when none is set"""
        self.config_manager.active_config = None
        
        active_config = self.config_manager.get_active_configuration()
        
        self.assertIsNone(active_config)
    
    def test_delete_configuration_existing(self):
        """Test deleting existing configuration"""
        config = GAConfiguration(population_size=75, max_generations=125)
        
        self.config_manager.configurations["test_delete"] = config
        
        # Create corresponding file
        config_file = Path(self.temp_dir) / "test_delete.json"
        with open(config_file, 'w') as f:
            json.dump(self.config_manager._config_to_dict(config), f)
        
        self.assertTrue(config_file.exists())
        
        result = self.config_manager.delete_configuration("test_delete")
        
        self.assertTrue(result)
        self.assertNotIn("test_delete", self.config_manager.configurations)
        self.assertFalse(config_file.exists())
    
    def test_delete_configuration_nonexistent(self):
        """Test deleting nonexistent configuration"""
        result = self.config_manager.delete_configuration("nonexistent")
        
        self.assertFalse(result)
    
    def test_delete_configuration_file_only(self):
        """Test deleting configuration that exists only in memory"""
        config = GAConfiguration(population_size=65, max_generations=135)
        
        self.config_manager.configurations["memory_only"] = config
        
        result = self.config_manager.delete_configuration("memory_only")
        
        self.assertTrue(result)
        self.assertNotIn("memory_only", self.config_manager.configurations)


class TestGAAnalysisIntegration(unittest.TestCase):
    """Integration tests for GA analysis components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = GASensitivityAnalyzer()
        self.config_manager = GAConfigManager(config_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_analyzer_config_manager_integration(self):
        """Test integration between analyzer and config manager"""
        # Create a configuration based on analyzer recommendations
        def mock_objective(param_name, param_value):
            # Mock objective that prefers certain parameter values
            if param_name == 'mutation_rate':
                return 100 - abs(param_value - 0.15) * 100
            elif param_name == 'crossover_rate':
                return 100 - abs(param_value - 0.8) * 50
            else:
                return param_value * 50
        
        # Analyze sensitivity
        with patch('numpy.linspace') as mock_linspace:
            mock_linspace.side_effect = [
                np.array([0.05, 0.1, 0.15, 0.2, 0.25]),  # mutation_rate
                np.array([0.6, 0.7, 0.8, 0.9, 1.0])     # crossover_rate
            ]
            
            with patch('time.time', side_effect=[i * 0.1 for i in range(200)]):
                mutation_result = self.analyzer.analyze_parameter_sensitivity(
                    'mutation_rate', mock_objective, (0.05, 0.25)
                )
                crossover_result = self.analyzer.analyze_parameter_sensitivity(
                    'crossover_rate', mock_objective, (0.6, 1.0)
                )
        
        # Get recommendations
        recommendations = self.analyzer.get_recommendations()
        
        # Create configuration based on recommendations
        optimized_config = GAConfiguration(
            population_size=100,
            max_generations=200,
            mutation_rate=recommendations['mutation_rate']['optimal_value'],
            crossover_rate=recommendations['crossover_rate']['optimal_value'],
            elite_size=2,
            objective="elevation"
        )
        
        # Save configuration
        self.config_manager.save_configuration("optimized", optimized_config)
        
        # Verify configuration was saved and can be loaded
        loaded_config = self.config_manager.load_configuration("optimized")
        
        self.assertIsNotNone(loaded_config)
        self.assertEqual(loaded_config.mutation_rate, mutation_result.optimal_value)
        self.assertEqual(loaded_config.crossover_rate, crossover_result.optimal_value)
    
    def test_config_manager_persistence(self):
        """Test configuration manager persistence across instances"""
        # Create and save configuration
        config = GAConfiguration(
            population_size=130,
            max_generations=220,
            mutation_rate=0.09,
            crossover_rate=0.82,
            elite_size=3,
            objective="elevation"
        )
        
        self.config_manager.save_configuration("persistent_test", config)
        
        # Create new config manager instance
        new_manager = GAConfigManager(config_dir=self.temp_dir)
        
        # Should load the previously saved configuration
        loaded_config = new_manager.load_configuration("persistent_test")
        
        self.assertIsNotNone(loaded_config)
        self.assertEqual(loaded_config.population_size, 130)
        self.assertEqual(loaded_config.max_generations, 220)
        self.assertEqual(loaded_config.mutation_rate, 0.09)
        self.assertEqual(loaded_config.crossover_rate, 0.82)
        self.assertEqual(loaded_config.elite_size, 3)
        self.assertEqual(loaded_config.objective, "elevation")
    
    def test_error_handling_workflow(self):
        """Test error handling in complete workflow"""
        # Test analyzer with failing objective function
        def failing_objective(param_name, param_value):
            if param_value > 0.2:
                raise RuntimeError("Objective function failed")
            return param_value * 100
        
        with patch('numpy.linspace') as mock_linspace:
            mock_linspace.return_value = np.array([0.1, 0.3, 0.5])
            
            with patch('time.time', side_effect=[i * 0.1 for i in range(20)]):
                with patch.object(self.analyzer, 'logger') as mock_logger:
                    result = self.analyzer.analyze_parameter_sensitivity(
                        'test_param', failing_objective, (0.1, 0.5)
                    )
        
        # Should have completed despite errors
        self.assertIsNotNone(result)
        self.assertEqual(result.parameter_name, 'test_param')
        
        # Should have logged warnings
        self.assertTrue(mock_logger.warning.called)
        
        # Test config manager with file system errors
        with patch('builtins.open', side_effect=OSError("File system error")):
            with patch.object(self.config_manager, 'logger') as mock_logger:
                config = GAConfiguration(population_size=100, max_generations=200)
                self.config_manager.save_configuration("error_test", config)
        
        # Should have logged error but not crashed
        self.assertTrue(mock_logger.error.called)
        
        # Configuration should still be in memory
        self.assertIn("error_test", self.config_manager.configurations)


if __name__ == '__main__':
    unittest.main()