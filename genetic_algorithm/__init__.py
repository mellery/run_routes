#!/usr/bin/env python3
"""
Genetic Algorithm Package
Consolidated genetic algorithm implementation for route optimization

This package provides a complete genetic algorithm implementation for optimizing
running routes. The main entry point is GeneticRouteOptimizer.

Basic usage:
    from genetic_algorithm import GeneticRouteOptimizer, FitnessObjective

    optimizer = GeneticRouteOptimizer(graph)
    result = optimizer.optimize(
        start_node=1529188403,
        distance_km=5.0,
        objective=FitnessObjective.MAXIMIZE_ELEVATION
    )
"""

# Core components - These are the main public API
from .chromosome import RouteSegment, RouteChromosome
from .population import PopulationInitializer
from .optimizer import GeneticRouteOptimizer
from .fitness import GAFitnessEvaluator, FitnessObjective
from .operators import GAOperators

# Optional components - Import these explicitly if needed
# from .performance import GASegmentCache
# from .visualization import GAVisualizer
# from .analysis import GAConfigManager
# from .optimization import GAHyperparameterOptimizer

__version__ = "2.0.0"  # Incremented for Phase 3 refactoring
__author__ = "Route Optimization Team"

# Export only core classes for clean API
__all__ = [
    # Core chromosome and optimization
    'RouteSegment',
    'RouteChromosome',
    'GeneticRouteOptimizer',

    # Population and operators
    'PopulationInitializer',
    'GAOperators',

    # Fitness evaluation
    'GAFitnessEvaluator',
    'FitnessObjective',
]
