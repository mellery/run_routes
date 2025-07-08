#!/usr/bin/env python3
"""
Common Imports for GA Components
Consolidates frequently used imports across GA files
"""

# Standard library imports
import os
import sys
import time
import math
import random
import logging
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Set, Callable, Union

# Third-party imports
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# Common GA imports will be imported dynamically to avoid circular dependencies

# Common constants
DEFAULT_POPULATION_SIZE = 100
DEFAULT_MAX_GENERATIONS = 200
DEFAULT_MUTATION_RATE = 0.1
DEFAULT_CROSSOVER_RATE = 0.8
DEFAULT_TARGET_DISTANCE = 5.0
DEFAULT_ELITE_SIZE = 2

# Common utility functions
def setup_logging(level=logging.INFO):
    """Set up logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_logger(name: str) -> logging.Logger:
    """Get logger with consistent configuration"""
    return logging.getLogger(name)

def validate_graph(graph: nx.Graph) -> bool:
    """Validate that graph has required attributes"""
    if not graph.nodes:
        return False
    
    # Check for required node attributes
    for node, data in graph.nodes(data=True):
        if 'x' not in data or 'y' not in data or 'elevation' not in data:
            return False
    
    # Check for required edge attributes
    for u, v, data in graph.edges(data=True):
        if 'length' not in data:
            return False
    
    return True

def calculate_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Calculate distance between two coordinates"""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    # Haversine formula
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon/2)**2)
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth radius in meters
    r = 6371000
    return r * c

def normalize_fitness(fitness: float) -> float:
    """Normalize fitness value to [0, 1] range"""
    return max(0.0, min(1.0, fitness))

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range"""
    return max(min_val, min(max_val, value))

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide with default for division by zero"""
    return numerator / denominator if denominator != 0 else default

def random_choice_weighted(choices: List[Any], weights: List[float]) -> Any:
    """Random choice with weights"""
    if not choices or not weights or len(choices) != len(weights):
        return random.choice(choices) if choices else None
    
    total_weight = sum(weights)
    if total_weight == 0:
        return random.choice(choices)
    
    r = random.random() * total_weight
    cumulative = 0
    for choice, weight in zip(choices, weights):
        cumulative += weight
        if r <= cumulative:
            return choice
    
    return choices[-1]

def get_route_bounds(route_nodes: List[int], graph: nx.Graph) -> Dict[str, float]:
    """Get geographic bounds of route"""
    if not route_nodes:
        return {'min_lat': 0, 'max_lat': 0, 'min_lon': 0, 'max_lon': 0}
    
    lats = []
    lons = []
    
    for node in route_nodes:
        if node in graph.nodes:
            data = graph.nodes[node]
            lats.append(data.get('y', 0))
            lons.append(data.get('x', 0))
    
    if not lats:
        return {'min_lat': 0, 'max_lat': 0, 'min_lon': 0, 'max_lon': 0}
    
    return {
        'min_lat': min(lats),
        'max_lat': max(lats),
        'min_lon': min(lons),
        'max_lon': max(lons)
    }

def validate_route_connectivity(route_nodes: List[int], graph: nx.Graph) -> bool:
    """Validate that route nodes are connected"""
    if len(route_nodes) < 2:
        return len(route_nodes) == 1
    
    for i in range(len(route_nodes) - 1):
        if not graph.has_edge(route_nodes[i], route_nodes[i + 1]):
            return False
    
    return True

def calculate_route_statistics(route_nodes: List[int], graph: nx.Graph) -> Dict[str, float]:
    """Calculate basic route statistics"""
    if not route_nodes:
        return {
            'total_distance': 0.0,
            'elevation_gain': 0.0,
            'elevation_loss': 0.0,
            'net_elevation': 0.0,
            'max_elevation': 0.0,
            'min_elevation': 0.0,
            'num_nodes': 0
        }
    
    total_distance = 0.0
    elevation_gain = 0.0
    elevation_loss = 0.0
    elevations = []
    
    # Collect elevations
    for node in route_nodes:
        if node in graph.nodes:
            elevations.append(graph.nodes[node].get('elevation', 0))
    
    # Calculate distance and elevation changes
    for i in range(len(route_nodes) - 1):
        node1, node2 = route_nodes[i], route_nodes[i + 1]
        
        # Distance
        if graph.has_edge(node1, node2):
            edge_data = graph[node1][node2]
            total_distance += edge_data.get('length', 0)
        
        # Elevation change
        if i < len(elevations) - 1:
            elev_change = elevations[i + 1] - elevations[i]
            if elev_change > 0:
                elevation_gain += elev_change
            else:
                elevation_loss += abs(elev_change)
    
    return {
        'total_distance': total_distance,
        'elevation_gain': elevation_gain,
        'elevation_loss': elevation_loss,
        'net_elevation': elevations[-1] - elevations[0] if elevations else 0,
        'max_elevation': max(elevations) if elevations else 0,
        'min_elevation': min(elevations) if elevations else 0,
        'num_nodes': len(route_nodes)
    }

# Common exception classes
class GAError(Exception):
    """Base GA exception"""
    pass

class InvalidChromosomeError(GAError):
    """Invalid chromosome error"""
    pass

class InvalidGraphError(GAError):
    """Invalid graph error"""
    pass

class OptimizationError(GAError):
    """Optimization error"""
    pass

# Common data structures
@dataclass
class GAConfiguration:
    """GA configuration parameters"""
    population_size: int = DEFAULT_POPULATION_SIZE
    max_generations: int = DEFAULT_MAX_GENERATIONS
    mutation_rate: float = DEFAULT_MUTATION_RATE
    crossover_rate: float = DEFAULT_CROSSOVER_RATE
    elite_size: int = DEFAULT_ELITE_SIZE
    target_distance_km: float = DEFAULT_TARGET_DISTANCE
    objective: str = "elevation"
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        return (
            self.population_size > 0 and
            self.max_generations > 0 and
            0 <= self.mutation_rate <= 1 and
            0 <= self.crossover_rate <= 1 and
            self.elite_size >= 0 and
            self.target_distance_km > 0
        )

@dataclass
class GAStatistics:
    """GA run statistics"""
    generation: int = 0
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    worst_fitness: float = 0.0
    fitness_std: float = 0.0
    diversity_score: float = 0.0
    convergence_rate: float = 0.0
    elapsed_time: float = 0.0
    
    def update(self, population: List[Any]):
        """Update statistics from population"""
        if not population:
            return
        
        fitnesses = [c.fitness for c in population if hasattr(c, 'fitness') and c.fitness is not None]
        if not fitnesses:
            return
        
        self.best_fitness = max(fitnesses)
        self.avg_fitness = sum(fitnesses) / len(fitnesses)
        self.worst_fitness = min(fitnesses)
        self.fitness_std = float(np.std(fitnesses))
        
        # Simple diversity measure
        unique_routes = set()
        for chromosome in population:
            if hasattr(chromosome, 'segments') and chromosome.segments:
                route_key = tuple(s.start_node for s in chromosome.segments if hasattr(s, 'start_node'))
                unique_routes.add(route_key)
        
        self.diversity_score = len(unique_routes) / len(population) if population else 0.0

# Performance monitoring
class GAPerformanceMonitor:
    """Monitor GA performance metrics"""
    
    def __init__(self):
        self.start_time = None
        self.generation_times = []
        self.fitness_history = []
        self.diversity_history = []
    
    def start_timing(self):
        """Start timing a GA run"""
        self.start_time = time.time()
    
    def record_generation(self, generation: int, statistics: GAStatistics):
        """Record generation statistics"""
        if self.start_time:
            statistics.elapsed_time = time.time() - self.start_time
        
        self.fitness_history.append(statistics.best_fitness)
        self.diversity_history.append(statistics.diversity_score)
        
        if len(self.fitness_history) > 1:
            self.generation_times.append(statistics.elapsed_time)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.fitness_history:
            return {}
        
        return {
            'total_time': sum(self.generation_times),
            'avg_generation_time': np.mean(self.generation_times) if self.generation_times else 0,
            'best_fitness_achieved': max(self.fitness_history),
            'final_fitness': self.fitness_history[-1],
            'fitness_improvement': self.fitness_history[-1] - self.fitness_history[0],
            'avg_diversity': np.mean(self.diversity_history),
            'generations_run': len(self.fitness_history)
        }

# Export commonly used items
__all__ = [
    # Standard imports
    'os', 'sys', 'time', 'math', 'random', 'logging', 'datetime', 'Enum', 'dataclass',
    'List', 'Tuple', 'Optional', 'Dict', 'Any', 'Set', 'Callable', 'Union',
    'np', 'nx', 'plt', 'patches', 'LinearSegmentedColormap',
    
    # GA imports (to be imported dynamically)
    
    # Constants
    'DEFAULT_POPULATION_SIZE', 'DEFAULT_MAX_GENERATIONS', 'DEFAULT_MUTATION_RATE',
    'DEFAULT_CROSSOVER_RATE', 'DEFAULT_TARGET_DISTANCE', 'DEFAULT_ELITE_SIZE',
    
    # Utility functions
    'setup_logging', 'get_logger', 'validate_graph', 'calculate_distance',
    'normalize_fitness', 'clamp', 'safe_divide', 'random_choice_weighted',
    'get_route_bounds', 'validate_route_connectivity', 'calculate_route_statistics',
    
    # Exception classes
    'GAError', 'InvalidChromosomeError', 'InvalidGraphError', 'OptimizationError',
    
    # Data structures
    'GAConfiguration', 'GAStatistics', 'GAPerformanceMonitor'
]