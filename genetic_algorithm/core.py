#!/usr/bin/env python3
"""
Genetic Algorithm Core Components
Consolidated core GA functionality including chromosomes, population initialization, and main optimizer
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ga_common_imports import (
    math, time, random, List, Optional, Tuple, Dict, Any, nx, np,
    calculate_distance, validate_route_connectivity, calculate_route_statistics,
    GAError, InvalidChromosomeError, GAConfiguration, GAStatistics,
    DEFAULT_POPULATION_SIZE, DEFAULT_MAX_GENERATIONS, DEFAULT_MUTATION_RATE,
    DEFAULT_CROSSOVER_RATE, DEFAULT_TARGET_DISTANCE, DEFAULT_ELITE_SIZE,
    get_logger, normalize_fitness, clamp, safe_divide
)

# Import components that will be used
from ga_fitness import GAFitnessEvaluator, FitnessObjective
from ga_operators import GAOperators
from ga_segment_cache import GASegmentCache, get_global_segment_cache

logger = get_logger(__name__)
