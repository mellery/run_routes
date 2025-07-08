#!/usr/bin/env python3
"""
Genetic Algorithm Package
Consolidated genetic algorithm implementation for route optimization
"""

# Core components
from .chromosome import RouteSegment, RouteChromosome
from .population import PopulationInitializer
from .optimizer import GeneticRouteOptimizer

# Fitness evaluation
from .fitness import GAFitnessEvaluator, FitnessObjective

# Performance optimization
from .performance import GASegmentCache, get_global_segment_cache

# Visualization
from .visualization import GAVisualizer, GATuningVisualizer, PrecisionComparisonVisualizer

# Genetic operators
from .operators import GAOperators, PrecisionAwareCrossover, PrecisionAwareMutation

# Analysis components
from .analysis import GAConfigManager, GASensitivityAnalyzer, SensitivityResult

# Optimization components
from .optimization import (
    GAHyperparameterOptimizer, GAParameterTuner, GAAlgorithmSelector,
    OptimizationMethod, AdaptationStrategy, HyperparameterSpace, OptimizationResult
)

# Common utilities
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ga_common_imports import GAConfiguration, GAStatistics, GAPerformanceMonitor

__version__ = "1.0.0"
__author__ = "GA Route Optimization Team"

__all__ = [
    # Core
    'RouteSegment',
    'RouteChromosome', 
    'GeneticRouteOptimizer',
    'PopulationInitializer',
    
    # Fitness
    'GAFitnessEvaluator',
    'FitnessObjective',
    
    # Performance
    'GASegmentCache',
    'get_global_segment_cache',
    
    # Visualization
    'GAVisualizer',
    'GATuningVisualizer',
    'PrecisionComparisonVisualizer',
    
    # Operators
    'GAOperators',
    'PrecisionAwareCrossover',
    'PrecisionAwareMutation',
    
    # Optimization
    'GAHyperparameterOptimizer',
    'GAParameterTuner',
    'GAAlgorithmSelector',
    'OptimizationMethod',
    'AdaptationStrategy',
    'HyperparameterSpace',
    'OptimizationResult',
    
    # Analysis
    'GAConfigManager',
    'GASensitivityAnalyzer',
    'SensitivityResult',
    
    # Configuration
    'GAConfiguration',
    'GAStatistics',
    'GAPerformanceMonitor',
]