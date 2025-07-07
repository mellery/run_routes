#!/usr/bin/env python3
"""
GA Parameter Tuning and Adaptive Adjustment System
Dynamic parameter optimization for genetic algorithm performance
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
from collections import deque

from route_objective import RouteObjective


class AdaptationStrategy(Enum):
    """Strategies for parameter adaptation"""
    STATIC = "static"                    # Fixed parameters
    LINEAR = "linear"                    # Linear adaptation over time
    EXPONENTIAL = "exponential"          # Exponential decay/growth
    FEEDBACK = "feedback"                # Performance-based feedback
    DIVERSITY = "diversity"              # Diversity-based adaptation
    CONVERGENCE = "convergence"          # Convergence-based adaptation
    HYBRID = "hybrid"                    # Multiple strategies combined


@dataclass
class ParameterRange:
    """Parameter value range and constraints"""
    min_value: float
    max_value: float
    default_value: float
    step_size: float = 0.01
    constraint_type: str = "continuous"  # continuous, discrete, integer
    valid_values: Optional[List] = None  # For discrete parameters
    
    def clamp(self, value: float) -> float:
        """Clamp value to valid range"""
        return max(self.min_value, min(self.max_value, value))
    
    def normalize(self, value: float) -> float:
        """Normalize value to [0, 1] range"""
        if self.max_value == self.min_value:
            return 0.0
        return (value - self.min_value) / (self.max_value - self.min_value)
    
    def denormalize(self, normalized_value: float) -> float:
        """Convert normalized value back to actual range"""
        return self.min_value + normalized_value * (self.max_value - self.min_value)


@dataclass
class AdaptationRule:
    """Rule for parameter adaptation"""
    parameter_name: str
    strategy: AdaptationStrategy
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    adjustment_rate: float = 0.1
    target_metric: str = "fitness_improvement"
    threshold: float = 0.01
    cooldown_generations: int = 5
    
    def __post_init__(self):
        self.last_adjustment_generation = -1


@dataclass
class PopulationStats:
    """Statistics about current population"""
    generation: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    fitness_std: float
    diversity_score: float
    convergence_rate: float
    plateau_length: int
    improvement_rate: float
    selection_pressure: float


@dataclass
class ParameterHistory:
    """History of parameter values and performance"""
    generation: int
    parameters: Dict[str, float]
    performance_metrics: Dict[str, float]
    adaptation_reason: str
    timestamp: float


class GAParameterTuner:
    """Adaptive parameter tuning system for genetic algorithms"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize parameter tuner
        
        Args:
            config: Configuration options for tuning
        """
        default_config = {
            'adaptation_interval': 5,         # Generations between adaptations
            'history_window': 20,            # Generations to track for trends
            'min_improvement_threshold': 0.01, # Minimum improvement to consider significant
            'diversity_weight': 0.3,         # Weight for diversity in adaptation
            'convergence_weight': 0.4,       # Weight for convergence in adaptation
            'performance_weight': 0.3,       # Weight for performance in adaptation
            'enable_auto_tuning': True,      # Enable automatic parameter tuning
            'save_adaptation_history': True, # Save parameter adaptation history
            'adaptation_aggressiveness': 0.5 # How aggressively to adapt (0-1)
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Parameter definitions
        self.parameter_ranges = self._define_parameter_ranges()
        self.adaptation_rules = self._define_adaptation_rules()
        
        # State tracking
        self.current_parameters = self._get_default_parameters()
        self.parameter_history = deque(maxlen=self.config['history_window'])
        self.population_stats_history = deque(maxlen=self.config['history_window'])
        self.adaptation_history = []
        
        # Performance tracking
        self.last_adaptation_generation = -1
        self.best_performance_seen = 0.0
        self.plateau_counter = 0
        self.adaptation_stats = {
            'total_adaptations': 0,
            'successful_adaptations': 0,
            'parameter_changes': {},
            'performance_improvements': []
        }
        
        print("🎛️ GA Parameter Tuner initialized with adaptive strategies")
    
    def _define_parameter_ranges(self) -> Dict[str, ParameterRange]:
        """Define valid ranges for GA parameters"""
        return {
            'population_size': ParameterRange(20, 500, 100, constraint_type='integer'),
            'mutation_rate': ParameterRange(0.001, 0.5, 0.1),
            'crossover_rate': ParameterRange(0.3, 0.95, 0.8),
            'elite_size_ratio': ParameterRange(0.02, 0.3, 0.1),
            'tournament_size': ParameterRange(2, 20, 5, constraint_type='integer'),
            'max_generations': ParameterRange(50, 1000, 200, constraint_type='integer'),
            'diversity_threshold': ParameterRange(0.1, 0.9, 0.3),
            'convergence_threshold': ParameterRange(0.001, 0.1, 0.01),
            'selection_pressure': ParameterRange(1.0, 5.0, 2.0),
            'fitness_scaling_factor': ParameterRange(1.0, 10.0, 2.0),
            'early_stopping_patience': ParameterRange(10, 100, 50, constraint_type='integer'),
            'distance_tolerance': ParameterRange(0.05, 0.5, 0.2),
            'elevation_weight': ParameterRange(0.1, 3.0, 1.0),
            'diversity_weight': ParameterRange(0.0, 1.0, 0.1)
        }
    
    def _define_adaptation_rules(self) -> List[AdaptationRule]:
        """Define rules for parameter adaptation"""
        rules = []
        
        # Mutation rate adaptation
        rules.append(AdaptationRule(
            parameter_name='mutation_rate',
            strategy=AdaptationStrategy.DIVERSITY,
            condition=lambda stats: stats.get('diversity_score', 0.5) < 0.2,
            adjustment_rate=0.05,
            target_metric='diversity_improvement',
            threshold=0.05
        ))
        
        # Population size adaptation
        rules.append(AdaptationRule(
            parameter_name='population_size',
            strategy=AdaptationStrategy.CONVERGENCE,
            condition=lambda stats: stats.get('plateau_length', 0) > 15,
            adjustment_rate=0.2,
            target_metric='fitness_improvement',
            threshold=0.01,
            cooldown_generations=10
        ))
        
        # Crossover rate adaptation
        rules.append(AdaptationRule(
            parameter_name='crossover_rate',
            strategy=AdaptationStrategy.FEEDBACK,
            condition=lambda stats: stats.get('improvement_rate', 0) < 0.01,
            adjustment_rate=0.1,
            target_metric='fitness_improvement',
            threshold=0.02
        ))
        
        # Tournament size adaptation
        rules.append(AdaptationRule(
            parameter_name='tournament_size',
            strategy=AdaptationStrategy.CONVERGENCE,
            condition=lambda stats: stats.get('convergence_rate', 0) > 0.9,
            adjustment_rate=0.3,
            target_metric='diversity_preservation',
            threshold=0.1
        ))
        
        # Elite size adaptation
        rules.append(AdaptationRule(
            parameter_name='elite_size_ratio',
            strategy=AdaptationStrategy.HYBRID,
            condition=lambda stats: stats.get('fitness_std', 1.0) < 0.1,
            adjustment_rate=0.05,
            target_metric='population_diversity',
            threshold=0.15
        ))
        
        return rules
    
    def _get_default_parameters(self) -> Dict[str, float]:
        """Get default parameter values"""
        return {name: param_range.default_value 
                for name, param_range in self.parameter_ranges.items()}
    
    def adapt_parameters(self, population_stats: PopulationStats) -> Dict[str, float]:
        """Adapt parameters based on population statistics
        
        Args:
            population_stats: Current population statistics
            
        Returns:
            Updated parameter dictionary
        """
        # Store current stats
        self.population_stats_history.append(population_stats)
        
        # Check if adaptation should occur
        if not self._should_adapt(population_stats):
            return self.current_parameters.copy()
        
        # Calculate adaptation metrics
        adaptation_metrics = self._calculate_adaptation_metrics(population_stats)
        
        # Apply adaptation rules
        parameter_changes = {}
        adaptation_reasons = []
        
        for rule in self.adaptation_rules:
            if self._should_apply_rule(rule, population_stats, adaptation_metrics):
                old_value = self.current_parameters[rule.parameter_name]
                new_value = self._apply_adaptation_rule(rule, old_value, adaptation_metrics)
                
                if abs(new_value - old_value) > self.parameter_ranges[rule.parameter_name].step_size:
                    parameter_changes[rule.parameter_name] = new_value
                    adaptation_reasons.append(f"{rule.parameter_name}: {rule.strategy.value}")
                    rule.last_adjustment_generation = population_stats.generation
        
        # Update parameters
        if parameter_changes:
            self.current_parameters.update(parameter_changes)
            self._record_adaptation(population_stats.generation, parameter_changes, 
                                  adaptation_metrics, ', '.join(adaptation_reasons))
            
            print(f"🎛️ Parameters adapted at generation {population_stats.generation}: "
                  f"{', '.join(f'{k}={v:.3f}' for k, v in parameter_changes.items())}")
        
        return self.current_parameters.copy()
    
    def _should_adapt(self, population_stats: PopulationStats) -> bool:
        """Determine if parameters should be adapted"""
        # Check adaptation interval
        if (population_stats.generation - self.last_adaptation_generation < 
            self.config['adaptation_interval']):
            return False
        
        # Check if auto-tuning is enabled
        if not self.config['enable_auto_tuning']:
            return False
        
        return True
    
    def _calculate_adaptation_metrics(self, current_stats: PopulationStats) -> Dict[str, float]:
        """Calculate metrics for parameter adaptation"""
        metrics = {}
        
        if len(self.population_stats_history) < 2:
            return {'fitness_improvement': 0.0, 'diversity_trend': 0.0, 'convergence_trend': 0.0}
        
        # Calculate fitness improvement trend
        recent_fitness = [stats.best_fitness for stats in list(self.population_stats_history)[-5:]]
        if len(recent_fitness) >= 2:
            fitness_trend = (recent_fitness[-1] - recent_fitness[0]) / max(len(recent_fitness) - 1, 1)
            metrics['fitness_improvement'] = fitness_trend
        else:
            metrics['fitness_improvement'] = 0.0
        
        # Calculate diversity trend
        recent_diversity = [stats.diversity_score for stats in list(self.population_stats_history)[-5:]]
        if len(recent_diversity) >= 2:
            diversity_trend = (recent_diversity[-1] - recent_diversity[0]) / max(len(recent_diversity) - 1, 1)
            metrics['diversity_trend'] = diversity_trend
        else:
            metrics['diversity_trend'] = 0.0
        
        # Calculate convergence metrics
        metrics['convergence_trend'] = current_stats.convergence_rate
        metrics['plateau_length'] = current_stats.plateau_length
        metrics['selection_pressure'] = current_stats.selection_pressure
        
        # Calculate performance metrics
        metrics['fitness_variance'] = current_stats.fitness_std
        metrics['improvement_rate'] = current_stats.improvement_rate
        
        return metrics
    
    def _should_apply_rule(self, rule: AdaptationRule, stats: PopulationStats, 
                          metrics: Dict[str, float]) -> bool:
        """Check if adaptation rule should be applied"""
        # Check cooldown period
        if (stats.generation - rule.last_adjustment_generation < rule.cooldown_generations):
            return False
        
        # Check rule condition
        if rule.condition and not rule.condition(metrics):
            return False
        
        # Check metric threshold
        target_value = metrics.get(rule.target_metric, 0.0)
        if abs(target_value) < rule.threshold:
            return False
        
        return True
    
    def _apply_adaptation_rule(self, rule: AdaptationRule, current_value: float, 
                              metrics: Dict[str, float]) -> float:
        """Apply adaptation rule to parameter"""
        param_range = self.parameter_ranges[rule.parameter_name]
        adjustment_factor = self.config['adaptation_aggressiveness']
        
        if rule.strategy == AdaptationStrategy.LINEAR:
            # Linear adjustment based on generation progress
            total_generations = self.current_parameters.get('max_generations', 200)
            current_generation = len(self.population_stats_history)
            progress = current_generation / total_generations
            
            if rule.parameter_name == 'mutation_rate':
                # Decrease mutation rate over time
                adjustment = -rule.adjustment_rate * progress * adjustment_factor
            else:
                adjustment = rule.adjustment_rate * (1 - progress) * adjustment_factor
        
        elif rule.strategy == AdaptationStrategy.EXPONENTIAL:
            # Exponential decay/growth
            decay_rate = rule.adjustment_rate * adjustment_factor
            if rule.parameter_name in ['mutation_rate', 'crossover_rate']:
                adjustment = -current_value * decay_rate
            else:
                adjustment = current_value * decay_rate
        
        elif rule.strategy == AdaptationStrategy.FEEDBACK:
            # Performance-based feedback
            performance_metric = metrics.get(rule.target_metric, 0.0)
            if performance_metric < rule.threshold:
                # Poor performance - increase exploration
                if rule.parameter_name in ['mutation_rate', 'population_size']:
                    adjustment = rule.adjustment_rate * adjustment_factor
                else:
                    adjustment = -rule.adjustment_rate * adjustment_factor
            else:
                # Good performance - maintain or slightly decrease exploration
                adjustment = -rule.adjustment_rate * 0.5 * adjustment_factor
        
        elif rule.strategy == AdaptationStrategy.DIVERSITY:
            # Diversity-based adaptation
            diversity_score = metrics.get('diversity_trend', 0.0)
            if diversity_score < 0:  # Decreasing diversity
                if rule.parameter_name == 'mutation_rate':
                    adjustment = rule.adjustment_rate * adjustment_factor
                elif rule.parameter_name == 'tournament_size':
                    adjustment = -rule.adjustment_rate * adjustment_factor
                else:
                    adjustment = rule.adjustment_rate * adjustment_factor
            else:
                adjustment = -rule.adjustment_rate * 0.3 * adjustment_factor
        
        elif rule.strategy == AdaptationStrategy.CONVERGENCE:
            # Convergence-based adaptation
            convergence_rate = metrics.get('convergence_trend', 0.0)
            plateau_length = metrics.get('plateau_length', 0)
            
            if convergence_rate > 0.8 or plateau_length > 10:
                # High convergence or plateau - increase exploration
                if rule.parameter_name in ['mutation_rate', 'population_size']:
                    adjustment = rule.adjustment_rate * adjustment_factor
                else:
                    adjustment = -rule.adjustment_rate * adjustment_factor
            else:
                adjustment = 0.0
        
        elif rule.strategy == AdaptationStrategy.HYBRID:
            # Combine multiple strategies
            diversity_factor = metrics.get('diversity_trend', 0.0)
            performance_factor = metrics.get('fitness_improvement', 0.0)
            
            # Weight different factors
            diversity_weight = self.config['diversity_weight']
            performance_weight = self.config['performance_weight']
            
            combined_signal = (diversity_factor * diversity_weight + 
                             performance_factor * performance_weight)
            
            if combined_signal < -rule.threshold:
                adjustment = rule.adjustment_rate * adjustment_factor
            elif combined_signal > rule.threshold:
                adjustment = -rule.adjustment_rate * adjustment_factor
            else:
                adjustment = 0.0
        
        else:  # STATIC
            adjustment = 0.0
        
        # Apply adjustment and clamp to valid range
        new_value = current_value + adjustment
        new_value = param_range.clamp(new_value)
        
        # Handle integer constraints
        if param_range.constraint_type == 'integer':
            new_value = round(new_value)
        
        return new_value
    
    def _record_adaptation(self, generation: int, parameter_changes: Dict[str, float],
                          metrics: Dict[str, float], reason: str):
        """Record parameter adaptation for analysis"""
        history_entry = ParameterHistory(
            generation=generation,
            parameters=parameter_changes.copy(),
            performance_metrics=metrics.copy(),
            adaptation_reason=reason,
            timestamp=time.time()
        )
        
        self.adaptation_history.append(history_entry)
        self.adaptation_stats['total_adaptations'] += 1
        
        # Track parameter changes
        for param_name in parameter_changes:
            if param_name not in self.adaptation_stats['parameter_changes']:
                self.adaptation_stats['parameter_changes'][param_name] = 0
            self.adaptation_stats['parameter_changes'][param_name] += 1
        
        self.last_adaptation_generation = generation
    
    def get_adapted_config(self, base_config: Dict[str, Any], 
                          objective: RouteObjective, target_distance_km: float) -> Dict[str, Any]:
        """Get adapted configuration for specific optimization scenario
        
        Args:
            base_config: Base GA configuration
            objective: Route optimization objective  
            target_distance_km: Target route distance
            
        Returns:
            Adapted configuration dictionary
        """
        adapted_config = base_config.copy()
        
        # Apply current parameter values
        for param_name, param_value in self.current_parameters.items():
            if param_name in adapted_config:
                adapted_config[param_name] = param_value
        
        # Objective-specific adaptations
        if objective == RouteObjective.MAXIMIZE_ELEVATION:
            adapted_config['mutation_rate'] = min(0.15, adapted_config.get('mutation_rate', 0.1) * 1.5)
            adapted_config['population_size'] = max(80, adapted_config.get('population_size', 100))
            adapted_config['elevation_weight'] = 2.0
        
        elif objective == RouteObjective.MINIMIZE_DISTANCE:
            adapted_config['crossover_rate'] = min(0.9, adapted_config.get('crossover_rate', 0.8) * 1.2)
            adapted_config['elite_size_ratio'] = min(0.2, adapted_config.get('elite_size_ratio', 0.1) * 1.5)
            adapted_config['elevation_weight'] = 0.5
        
        elif objective == RouteObjective.BALANCED_ROUTE:
            # Balanced approach - use default parameter ranges
            pass
        
        # Distance-based adaptations
        if target_distance_km <= 3.0:
            # Short routes - smaller population, fewer generations
            adapted_config['population_size'] = int(adapted_config.get('population_size', 100) * 0.8)
            adapted_config['max_generations'] = int(adapted_config.get('max_generations', 200) * 0.7)
        
        elif target_distance_km >= 10.0:
            # Long routes - larger population, more generations
            adapted_config['population_size'] = int(adapted_config.get('population_size', 100) * 1.3)
            adapted_config['max_generations'] = int(adapted_config.get('max_generations', 200) * 1.5)
        
        return adapted_config
    
    def get_tuning_recommendations(self) -> Dict[str, Any]:
        """Get parameter tuning recommendations based on adaptation history
        
        Returns:
            Dictionary with tuning recommendations and insights
        """
        recommendations = {
            'parameter_suggestions': {},
            'performance_insights': {},
            'adaptation_summary': {},
            'optimization_tips': []
        }
        
        if not self.adaptation_history:
            recommendations['optimization_tips'].append("No adaptation history available yet")
            return recommendations
        
        # Analyze parameter change patterns
        param_changes = self.adaptation_stats['parameter_changes']
        most_changed_params = sorted(param_changes.items(), key=lambda x: x[1], reverse=True)
        
        recommendations['adaptation_summary'] = {
            'total_adaptations': self.adaptation_stats['total_adaptations'],
            'most_changed_parameters': most_changed_params[:3],
            'adaptation_frequency': len(self.adaptation_history) / max(len(self.population_stats_history), 1)
        }
        
        # Generate parameter suggestions
        for param_name, change_count in most_changed_params[:5]:
            if change_count > 2:  # Parameter changed frequently
                current_value = self.current_parameters[param_name]
                param_range = self.parameter_ranges[param_name]
                
                # Suggest range adjustment
                if current_value > param_range.default_value * 1.2:
                    recommendations['parameter_suggestions'][param_name] = {
                        'current': current_value,
                        'suggestion': 'Consider increasing default value',
                        'reason': f'Parameter adapted upward {change_count} times'
                    }
                elif current_value < param_range.default_value * 0.8:
                    recommendations['parameter_suggestions'][param_name] = {
                        'current': current_value,
                        'suggestion': 'Consider decreasing default value',
                        'reason': f'Parameter adapted downward {change_count} times'
                    }
        
        # Performance insights
        if len(self.population_stats_history) > 5:
            recent_stats = list(self.population_stats_history)[-5:]
            
            avg_diversity = sum(s.diversity_score for s in recent_stats) / len(recent_stats)
            avg_improvement = sum(s.improvement_rate for s in recent_stats) / len(recent_stats)
            
            recommendations['performance_insights'] = {
                'average_diversity': avg_diversity,
                'average_improvement_rate': avg_improvement,
                'plateau_tendency': sum(s.plateau_length for s in recent_stats) / len(recent_stats)
            }
        
        # Generate optimization tips
        tips = []
        
        if self.adaptation_stats['total_adaptations'] == 0:
            tips.append("Enable adaptive parameter tuning for better performance")
        
        if 'mutation_rate' in param_changes and param_changes['mutation_rate'] > 3:
            tips.append("Mutation rate is adapting frequently - consider population diversity issues")
        
        if 'population_size' in param_changes and param_changes['population_size'] > 2:
            tips.append("Population size adaptations suggest convergence issues")
        
        avg_diversity = (sum(s.diversity_score for s in self.population_stats_history) / 
                        max(len(self.population_stats_history), 1))
        if avg_diversity < 0.3:
            tips.append("Consider increasing mutation rate or population size for better diversity")
        
        recommendations['optimization_tips'] = tips
        
        return recommendations
    
    def save_adaptation_history(self, filename: str) -> str:
        """Save parameter adaptation history to file
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if not self.config['save_adaptation_history']:
            return ""
        
        history_data = {
            'config': self.config,
            'parameter_ranges': {name: {
                'min_value': pr.min_value,
                'max_value': pr.max_value,
                'default_value': pr.default_value,
                'constraint_type': pr.constraint_type
            } for name, pr in self.parameter_ranges.items()},
            'adaptation_history': [{
                'generation': h.generation,
                'parameters': h.parameters,
                'performance_metrics': h.performance_metrics,
                'adaptation_reason': h.adaptation_reason,
                'timestamp': h.timestamp
            } for h in self.adaptation_history],
            'adaptation_stats': self.adaptation_stats,
            'final_parameters': self.current_parameters
        }
        
        with open(filename, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        return filename
    
    def reset_adaptation_state(self):
        """Reset adaptation state for new optimization run"""
        self.current_parameters = self._get_default_parameters()
        self.parameter_history.clear()
        self.population_stats_history.clear()
        self.last_adaptation_generation = -1
        self.plateau_counter = 0
        
        # Reset rule states
        for rule in self.adaptation_rules:
            rule.last_adjustment_generation = -1
        
        print("🎛️ Parameter tuner state reset for new optimization run")


def test_parameter_tuner():
    """Test function for parameter tuner"""
    print("Testing GA Parameter Tuner...")
    
    # Create tuner
    tuner = GAParameterTuner()
    
    # Test default parameters
    default_params = tuner._get_default_parameters()
    print(f"✅ Default parameters: {len(default_params)} parameters defined")
    
    # Test parameter ranges
    for name, param_range in tuner.parameter_ranges.items():
        test_value = (param_range.min_value + param_range.max_value) / 2
        clamped = param_range.clamp(test_value)
        print(f"✅ Parameter {name}: range [{param_range.min_value}, {param_range.max_value}], test value {clamped}")
    
    # Test adaptation with mock population stats
    mock_stats = PopulationStats(
        generation=10,
        best_fitness=0.8,
        avg_fitness=0.6,
        worst_fitness=0.3,
        fitness_std=0.15,
        diversity_score=0.25,  # Low diversity
        convergence_rate=0.7,
        plateau_length=8,
        improvement_rate=0.02,
        selection_pressure=2.5
    )
    
    # Test adaptation
    adapted_params = tuner.adapt_parameters(mock_stats)
    print(f"✅ Parameter adaptation completed: {len(adapted_params)} parameters")
    
    # Test configuration adaptation
    base_config = {'population_size': 100, 'mutation_rate': 0.1, 'max_generations': 200}
    adapted_config = tuner.get_adapted_config(base_config, RouteObjective.MAXIMIZE_ELEVATION, 5.0)
    print(f"✅ Configuration adaptation: {adapted_config}")
    
    # Test recommendations
    recommendations = tuner.get_tuning_recommendations()
    print(f"✅ Tuning recommendations: {len(recommendations)} categories")
    
    print("✅ All parameter tuner tests completed")


if __name__ == "__main__":
    test_parameter_tuner()