#!/usr/bin/env python3
"""
Genetic Algorithm Analysis Components
Consolidated sensitivity analysis and configuration management
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from ga_common_imports import (
    GAConfiguration, GAStatistics, GAPerformanceMonitor,
    get_logger, DEFAULT_POPULATION_SIZE, DEFAULT_MAX_GENERATIONS,
    DEFAULT_MUTATION_RATE, DEFAULT_CROSSOVER_RATE
)


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis"""
    parameter_name: str
    parameter_values: List[float]
    fitness_scores: List[float]
    convergence_times: List[float]
    sensitivity_coefficient: float
    optimal_value: float
    confidence_interval: Tuple[float, float]


class GASensitivityAnalyzer:
    """Analyzes parameter sensitivity for genetic algorithms"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize sensitivity analyzer
        
        Args:
            config: Configuration options
        """
        default_config = {
            'num_samples': 20,
            'num_trials': 5,
            'confidence_level': 0.95,
            'sensitivity_threshold': 0.1,
            'parameter_ranges': {
                'population_size': (20, 200),
                'mutation_rate': (0.01, 0.5),
                'crossover_rate': (0.3, 1.0),
                'elite_size': (1, 10),
                'tournament_size': (2, 8)
            }
        }
        
        self.config = {**default_config, **(config or {})}
        self.logger = get_logger(__name__)
        
        # Analysis results
        self.sensitivity_results = {}
        self.analysis_history = []
    
    def analyze_parameter_sensitivity(self, parameter_name: str, 
                                    objective_function: callable,
                                    parameter_range: Optional[Tuple[float, float]] = None) -> SensitivityResult:
        """Analyze sensitivity of a single parameter
        
        Args:
            parameter_name: Name of parameter to analyze
            objective_function: Function to evaluate parameter performance
            parameter_range: Range of values to test
            
        Returns:
            Sensitivity analysis result
        """
        if parameter_range is None:
            parameter_range = self.config['parameter_ranges'].get(
                parameter_name, (0.01, 1.0))
        
        # Generate parameter values to test
        min_val, max_val = parameter_range
        parameter_values = np.linspace(min_val, max_val, self.config['num_samples'])
        
        # Test each parameter value
        fitness_scores = []
        convergence_times = []
        
        for param_value in parameter_values:
            # Run multiple trials for each parameter value
            trial_fitness = []
            trial_times = []
            
            for trial in range(self.config['num_trials']):
                try:
                    start_time = time.time()
                    fitness = objective_function(parameter_name, param_value)
                    convergence_time = time.time() - start_time
                    
                    trial_fitness.append(fitness)
                    trial_times.append(convergence_time)
                    
                except Exception as e:
                    self.logger.warning(f"Trial failed for {parameter_name}={param_value}: {e}")
                    trial_fitness.append(0.0)
                    trial_times.append(float('inf'))
            
            # Average results across trials
            avg_fitness = np.mean(trial_fitness)
            avg_time = np.mean(trial_times)
            
            fitness_scores.append(avg_fitness)
            convergence_times.append(avg_time)
            
            self.logger.info(f"Parameter {parameter_name}={param_value:.3f}: "
                           f"fitness={avg_fitness:.3f}, time={avg_time:.1f}s")
        
        # Calculate sensitivity coefficient
        sensitivity_coeff = self._calculate_sensitivity_coefficient(
            parameter_values, fitness_scores)
        
        # Find optimal value
        optimal_idx = np.argmax(fitness_scores)
        optimal_value = parameter_values[optimal_idx]
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            parameter_values, fitness_scores)
        
        result = SensitivityResult(
            parameter_name=parameter_name,
            parameter_values=parameter_values.tolist(),
            fitness_scores=fitness_scores,
            convergence_times=convergence_times,
            sensitivity_coefficient=sensitivity_coeff,
            optimal_value=optimal_value,
            confidence_interval=confidence_interval
        )
        
        # Store result
        self.sensitivity_results[parameter_name] = result
        
        return result
    
    def analyze_all_parameters(self, objective_function: callable) -> Dict[str, SensitivityResult]:
        """Analyze sensitivity of all configured parameters
        
        Args:
            objective_function: Function to evaluate parameter performance
            
        Returns:
            Dictionary of sensitivity results
        """
        results = {}
        
        for param_name, param_range in self.config['parameter_ranges'].items():
            self.logger.info(f"Analyzing parameter: {param_name}")
            
            result = self.analyze_parameter_sensitivity(
                param_name, objective_function, param_range)
            results[param_name] = result
        
        return results
    
    def _calculate_sensitivity_coefficient(self, parameter_values: np.ndarray, 
                                         fitness_scores: List[float]) -> float:
        """Calculate sensitivity coefficient"""
        if len(parameter_values) < 2:
            return 0.0
        
        # Calculate correlation coefficient
        param_normalized = (parameter_values - np.mean(parameter_values)) / np.std(parameter_values)
        fitness_normalized = (np.array(fitness_scores) - np.mean(fitness_scores)) / np.std(fitness_scores)
        
        correlation = np.corrcoef(param_normalized, fitness_normalized)[0, 1]
        
        # Return absolute correlation as sensitivity coefficient
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    def _calculate_confidence_interval(self, parameter_values: np.ndarray,
                                     fitness_scores: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for optimal parameter value"""
        if len(fitness_scores) < 2:
            return (parameter_values[0], parameter_values[-1])
        
        # Simple confidence interval based on top performers
        sorted_indices = np.argsort(fitness_scores)[::-1]
        top_10_percent = max(1, len(sorted_indices) // 10)
        top_indices = sorted_indices[:top_10_percent]
        
        top_parameters = parameter_values[top_indices]
        
        return (float(np.min(top_parameters)), float(np.max(top_parameters)))
    
    def get_sensitivity_ranking(self) -> List[Tuple[str, float]]:
        """Get parameters ranked by sensitivity
        
        Returns:
            List of (parameter_name, sensitivity_coefficient) tuples
        """
        ranking = []
        
        for param_name, result in self.sensitivity_results.items():
            ranking.append((param_name, result.sensitivity_coefficient))
        
        # Sort by sensitivity coefficient (descending)
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        return ranking
    
    def get_recommendations(self) -> Dict[str, Any]:
        """Get parameter recommendations based on sensitivity analysis
        
        Returns:
            Dictionary of parameter recommendations
        """
        recommendations = {}
        
        for param_name, result in self.sensitivity_results.items():
            recommendations[param_name] = {
                'optimal_value': result.optimal_value,
                'sensitivity': result.sensitivity_coefficient,
                'confidence_interval': result.confidence_interval,
                'recommendation': self._get_parameter_recommendation(result)
            }
        
        return recommendations
    
    def _get_parameter_recommendation(self, result: SensitivityResult) -> str:
        """Get recommendation text for a parameter"""
        if result.sensitivity_coefficient > 0.7:
            return "Highly sensitive - tune carefully"
        elif result.sensitivity_coefficient > 0.3:
            return "Moderately sensitive - consider tuning"
        else:
            return "Low sensitivity - use default value"


class GAConfigManager:
    """Manages configuration profiles for genetic algorithms"""
    
    def __init__(self, config_dir: str = "ga_configs"):
        """Initialize configuration manager
        
        Args:
            config_dir: Directory to store configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.logger = get_logger(__name__)
        
        # Current configurations
        self.configurations = {}
        self.active_config = None
        
        # Load existing configurations
        self._load_configurations()
        
        # Setup default configurations
        self._setup_default_configurations()
    
    def _setup_default_configurations(self):
        """Setup default configuration profiles"""
        defaults = {
            'standard': GAConfiguration(
                population_size=100,
                max_generations=200,
                mutation_rate=0.1,
                crossover_rate=0.8,
                elite_size=2,
                objective="elevation"
            ),
            'fast': GAConfiguration(
                population_size=50,
                max_generations=100,
                mutation_rate=0.15,
                crossover_rate=0.9,
                elite_size=1,
                objective="elevation"
            ),
            'thorough': GAConfiguration(
                population_size=200,
                max_generations=500,
                mutation_rate=0.05,
                crossover_rate=0.7,
                elite_size=5,
                objective="elevation"
            ),
            'distance_focused': GAConfiguration(
                population_size=100,
                max_generations=150,
                mutation_rate=0.08,
                crossover_rate=0.85,
                elite_size=3,
                objective="distance"
            ),
            'balanced': GAConfiguration(
                population_size=150,
                max_generations=300,
                mutation_rate=0.12,
                crossover_rate=0.75,
                elite_size=4,
                objective="balanced"
            )
        }
        
        # Add defaults if they don't exist
        for name, config in defaults.items():
            if name not in self.configurations:
                self.configurations[name] = config
        
        # Set default active configuration
        if self.active_config is None:
            self.active_config = 'standard'
    
    def save_configuration(self, name: str, config: GAConfiguration):
        """Save a configuration profile
        
        Args:
            name: Configuration name
            config: Configuration object
        """
        self.configurations[name] = config
        
        # Save to file
        config_file = self.config_dir / f"{name}.json"
        try:
            with open(config_file, 'w') as f:
                json.dump(self._config_to_dict(config), f, indent=2)
            
            self.logger.info(f"Saved configuration: {name}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration {name}: {e}")
    
    def load_configuration(self, name: str) -> Optional[GAConfiguration]:
        """Load a configuration profile
        
        Args:
            name: Configuration name
            
        Returns:
            Configuration object or None if not found
        """
        if name in self.configurations:
            return self.configurations[name]
        
        # Try to load from file
        config_file = self.config_dir / f"{name}.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                
                config = self._dict_to_config(config_dict)
                self.configurations[name] = config
                
                return config
                
            except Exception as e:
                self.logger.error(f"Failed to load configuration {name}: {e}")
        
        return None
    
    def _load_configurations(self):
        """Load all configurations from files"""
        if not self.config_dir.exists():
            return
        
        for config_file in self.config_dir.glob("*.json"):
            config_name = config_file.stem
            
            try:
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                
                config = self._dict_to_config(config_dict)
                self.configurations[config_name] = config
                
            except Exception as e:
                self.logger.error(f"Failed to load configuration {config_name}: {e}")
    
    def _config_to_dict(self, config: GAConfiguration) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'population_size': config.population_size,
            'max_generations': config.max_generations,
            'mutation_rate': config.mutation_rate,
            'crossover_rate': config.crossover_rate,
            'elite_size': config.elite_size,
            'target_distance_km': config.target_distance_km,
            'objective': config.objective
        }
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> GAConfiguration:
        """Convert dictionary to configuration"""
        return GAConfiguration(
            population_size=config_dict.get('population_size', DEFAULT_POPULATION_SIZE),
            max_generations=config_dict.get('max_generations', DEFAULT_MAX_GENERATIONS),
            mutation_rate=config_dict.get('mutation_rate', DEFAULT_MUTATION_RATE),
            crossover_rate=config_dict.get('crossover_rate', DEFAULT_CROSSOVER_RATE),
            elite_size=config_dict.get('elite_size', 2),
            target_distance_km=config_dict.get('target_distance_km', 5.0),
            objective=config_dict.get('objective', "elevation")
        )
    
    def get_configuration(self, name: str) -> Optional[GAConfiguration]:
        """Get a configuration by name
        
        Args:
            name: Configuration name
            
        Returns:
            Configuration object or None
        """
        return self.configurations.get(name)
    
    def list_configurations(self) -> List[str]:
        """List all available configurations
        
        Returns:
            List of configuration names
        """
        return list(self.configurations.keys())
    
    def set_active_configuration(self, name: str) -> bool:
        """Set active configuration
        
        Args:
            name: Configuration name
            
        Returns:
            True if successful, False otherwise
        """
        if name in self.configurations:
            self.active_config = name
            self.logger.info(f"Set active configuration: {name}")
            return True
        
        return False
    
    def get_active_configuration(self) -> Optional[GAConfiguration]:
        """Get active configuration
        
        Returns:
            Active configuration or None
        """
        if self.active_config:
            return self.configurations.get(self.active_config)
        
        return None
    
    def delete_configuration(self, name: str) -> bool:
        """Delete a configuration
        
        Args:
            name: Configuration name
            
        Returns:
            True if successful, False otherwise
        """
        if name in self.configurations:
            del self.configurations[name]
            
            # Delete file
            config_file = self.config_dir / f"{name}.json"
            if config_file.exists():
                config_file.unlink()
            
            self.logger.info(f"Deleted configuration: {name}")
            return True
        
        return False


# Export main classes
__all__ = [
    'GASensitivityAnalyzer',
    'GAConfigManager',
    'SensitivityResult'
]