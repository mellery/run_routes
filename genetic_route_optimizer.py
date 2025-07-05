#!/usr/bin/env python3
"""
Genetic Algorithm Route Optimizer
Main genetic algorithm implementation for route optimization
"""

import time
import random
import math
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import networkx as nx

from ga_chromosome import RouteChromosome
from ga_population import PopulationInitializer
from ga_operators import GAOperators
from ga_fitness import GAFitnessEvaluator, FitnessObjective


@dataclass
class GAConfig:
    """Configuration for genetic algorithm"""
    population_size: int = 100
    max_generations: int = 200
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    elite_size: int = 10
    tournament_size: int = 5
    convergence_threshold: float = 0.001
    convergence_generations: int = 20
    adaptive_sizing: bool = True
    verbose: bool = True


@dataclass
class GAResults:
    """Results from genetic algorithm optimization"""
    best_chromosome: RouteChromosome
    best_fitness: float
    generation_found: int
    total_generations: int
    total_time: float
    convergence_reason: str
    population_history: List[List[RouteChromosome]]
    fitness_history: List[List[float]]
    stats: Dict[str, Any]


class GeneticRouteOptimizer:
    """Main genetic algorithm optimizer for route optimization"""
    
    def __init__(self, graph: nx.Graph, config: Optional[GAConfig] = None):
        """Initialize genetic optimizer
        
        Args:
            graph: NetworkX graph with elevation and distance data
            config: GA configuration parameters
        """
        self.graph = graph
        self.config = config or GAConfig()
        
        # Initialize components
        self.population_initializer = None
        self.operators = GAOperators(graph)
        self.fitness_evaluator = None
        
        # Evolution tracking
        self.generation = 0
        self.population_history = []
        self.fitness_history = []
        self.best_chromosome = None
        self.best_fitness = 0.0
        self.best_generation = 0
        
        # Performance tracking
        self.start_time = None
        self.evaluation_times = []
        
        # Callbacks
        self.generation_callback = None
        self.progress_callback = None
        
    def optimize_route(self, start_node: int, distance_km: float, 
                      objective: str = "elevation",
                      visualizer: Optional[Any] = None) -> GAResults:
        """Optimize route using genetic algorithm
        
        Args:
            start_node: Starting node ID
            distance_km: Target route distance in kilometers
            objective: Optimization objective
            visualizer: Optional visualizer for progress tracking
            
        Returns:
            GAResults with optimization results
        """
        # Setup optimization
        self._setup_optimization(start_node, distance_km, objective)
        
        # Adaptive configuration
        if self.config.adaptive_sizing:
            self._adapt_configuration(distance_km)
        
        if self.config.verbose:
            print(f"🧬 Starting GA optimization:")
            print(f"   Target: {distance_km}km route from node {start_node}")
            print(f"   Objective: {objective}")
            print(f"   Population: {self.config.population_size}")
            print(f"   Max generations: {self.config.max_generations}")
        
        # Initialize population
        if self.config.verbose:
            print(f"👥 Initializing population...")
        
        population = self.population_initializer.create_population(
            self.config.population_size, distance_km
        )
        
        if not population:
            raise ValueError("Failed to initialize population")
        
        # Evaluate initial population
        fitness_scores = self.fitness_evaluator.evaluate_population(population)
        
        # Track initial state
        self.population_history.append(population.copy())
        self.fitness_history.append(fitness_scores.copy())
        
        # Find initial best
        best_idx = fitness_scores.index(max(fitness_scores))
        self.best_chromosome = population[best_idx].copy()
        self.best_fitness = fitness_scores[best_idx]
        self.best_generation = 0
        
        if self.config.verbose:
            print(f"✅ Initial population created")
            print(f"   Best fitness: {self.best_fitness:.4f}")
            print(f"   Average fitness: {sum(fitness_scores)/len(fitness_scores):.4f}")
        
        # Generate initial visualization
        if visualizer:
            self._generate_visualization(population, fitness_scores, 0, visualizer)
        
        # Main evolution loop
        self.start_time = time.time()
        convergence_reason = "max_generations"
        
        for generation in range(1, self.config.max_generations + 1):
            self.generation = generation
            
            gen_start_time = time.time()
            
            # Evolve population
            population, fitness_scores = self._evolve_generation(population, fitness_scores)
            
            # Track evolution time
            gen_time = time.time() - gen_start_time
            self.evaluation_times.append(gen_time)
            
            # Update best chromosome
            current_best_idx = fitness_scores.index(max(fitness_scores))
            current_best_fitness = fitness_scores[current_best_idx]
            
            if current_best_fitness > self.best_fitness:
                self.best_chromosome = population[current_best_idx].copy()
                self.best_fitness = current_best_fitness
                self.best_generation = generation
            
            # Track generation
            self.population_history.append(population.copy())
            self.fitness_history.append(fitness_scores.copy())
            
            # Progress reporting
            if self.config.verbose and generation % 10 == 0:
                avg_fitness = sum(fitness_scores) / len(fitness_scores)
                print(f"   Gen {generation:3d}: Best={self.best_fitness:.4f}, "
                      f"Avg={avg_fitness:.4f}, Time={gen_time:.2f}s")
            
            # Generation callback
            if self.generation_callback:
                self.generation_callback(generation, population, fitness_scores)
            
            # Generate visualization
            if visualizer and generation % 25 == 0:
                self._generate_visualization(population, fitness_scores, generation, visualizer)
            
            # Check convergence
            if self._check_convergence(fitness_scores):
                convergence_reason = "convergence"
                if self.config.verbose:
                    print(f"🎯 Converged after {generation} generations")
                break
            
            # Progress callback
            if self.progress_callback:
                progress = generation / self.config.max_generations
                self.progress_callback(progress, self.best_fitness)
        
        total_time = time.time() - self.start_time
        
        # Final results
        results = GAResults(
            best_chromosome=self.best_chromosome,
            best_fitness=self.best_fitness,
            generation_found=self.best_generation,
            total_generations=self.generation,
            total_time=total_time,
            convergence_reason=convergence_reason,
            population_history=self.population_history,
            fitness_history=self.fitness_history,
            stats=self._get_optimization_stats()
        )
        
        if self.config.verbose:
            print(f"🏁 Optimization completed:")
            print(f"   Best fitness: {self.best_fitness:.4f}")
            print(f"   Found at generation: {self.best_generation}")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Convergence: {convergence_reason}")
        
        return results
    
    def _setup_optimization(self, start_node: int, distance_km: float, objective: str):
        """Setup optimization components"""
        # Initialize population initializer
        self.population_initializer = PopulationInitializer(self.graph, start_node)
        
        # Initialize fitness evaluator
        self.fitness_evaluator = GAFitnessEvaluator(objective, distance_km)
        
        # Reset tracking
        self.generation = 0
        self.population_history = []
        self.fitness_history = []
        self.best_chromosome = None
        self.best_fitness = 0.0
        self.best_generation = 0
        self.evaluation_times = []
    
    def _adapt_configuration(self, distance_km: float):
        """Adapt GA configuration based on problem size"""
        # Adjust population size based on distance
        if distance_km < 3.0:
            self.config.population_size = max(50, self.config.population_size // 2)
            self.config.max_generations = max(100, self.config.max_generations // 2)
        elif distance_km > 8.0:
            self.config.population_size = min(200, int(self.config.population_size * 1.5))
            self.config.max_generations = min(500, int(self.config.max_generations * 1.5))
        
        # Adjust selection pressure
        self.config.elite_size = max(5, self.config.population_size // 10)
        self.config.tournament_size = max(3, self.config.population_size // 20)
    
    def _evolve_generation(self, population: List[RouteChromosome], 
                         fitness_scores: List[float]) -> Tuple[List[RouteChromosome], List[float]]:
        """Evolve population for one generation"""
        new_population = []
        
        # Elitism - preserve best individuals
        elite_indices = sorted(range(len(fitness_scores)), 
                             key=lambda i: fitness_scores[i], reverse=True)[:self.config.elite_size]
        elite = [population[i].copy() for i in elite_indices]
        new_population.extend(elite)
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Selection
            parent1 = self.operators.tournament_selection(population, self.config.tournament_size)
            parent2 = self.operators.tournament_selection(population, self.config.tournament_size)
            
            # Crossover
            offspring1, offspring2 = self.operators.segment_exchange_crossover(
                parent1, parent2, self.config.crossover_rate
            )
            
            # Mutation
            offspring1 = self.operators.segment_replacement_mutation(
                offspring1, self.config.mutation_rate
            )
            offspring2 = self.operators.route_extension_mutation(
                offspring2, self.fitness_evaluator.target_distance_km, self.config.mutation_rate
            )
            
            # Add to population
            new_population.extend([offspring1, offspring2])
        
        # Trim to exact population size
        new_population = new_population[:self.config.population_size]
        
        # Evaluate new population
        new_fitness_scores = self.fitness_evaluator.evaluate_population(new_population)
        
        return new_population, new_fitness_scores
    
    def _check_convergence(self, fitness_scores: List[float]) -> bool:
        """Check if population has converged"""
        if len(self.fitness_history) < self.config.convergence_generations:
            return False
        
        # Check fitness plateau
        recent_best = []
        for i in range(self.config.convergence_generations):
            gen_fitness = self.fitness_history[-(i+1)]
            recent_best.append(max(gen_fitness))
        
        # Check if improvement is below threshold
        max_recent = max(recent_best)
        min_recent = min(recent_best)
        improvement = max_recent - min_recent
        
        return improvement < self.config.convergence_threshold
    
    def _generate_visualization(self, population: List[RouteChromosome], 
                              fitness_scores: List[float], generation: int, visualizer: Any):
        """Generate visualization for current generation"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ga_evolution_gen_{generation:03d}_{timestamp}.png"
            
            visualizer.save_population_map(
                population, generation, filename,
                show_fitness=True, show_elevation=True,
                title=f"Generation {generation} - Best Fitness: {max(fitness_scores):.4f}"
            )
            
            if self.config.verbose and generation % 50 == 0:
                print(f"📸 Saved visualization: {filename}")
                
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️ Visualization failed: {e}")
    
    def _get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        if not self.fitness_history:
            return {}
        
        # Fitness statistics
        all_fitness = [f for gen_fitness in self.fitness_history for f in gen_fitness]
        best_per_generation = [max(gen_fitness) for gen_fitness in self.fitness_history]
        avg_per_generation = [sum(gen_fitness)/len(gen_fitness) for gen_fitness in self.fitness_history]
        
        # Performance statistics
        total_evaluations = sum(len(gen_fitness) for gen_fitness in self.fitness_history)
        avg_gen_time = sum(self.evaluation_times) / max(len(self.evaluation_times), 1)
        
        return {
            'total_evaluations': total_evaluations,
            'avg_generation_time': avg_gen_time,
            'best_fitness_progression': best_per_generation,
            'avg_fitness_progression': avg_per_generation,
            'fitness_improvement': self.best_fitness - (best_per_generation[0] if best_per_generation else 0),
            'convergence_generation': self.best_generation,
            'population_diversity': self._calculate_population_diversity(),
            'objective': self.fitness_evaluator.objective.value,
            'target_distance': self.fitness_evaluator.target_distance_km
        }
    
    def _calculate_population_diversity(self) -> float:
        """Calculate current population diversity"""
        if not self.population_history:
            return 0.0
        
        current_population = self.population_history[-1]
        distances = [chromo.get_total_distance() for chromo in current_population]
        
        if not distances:
            return 0.0
        
        # Coefficient of variation as diversity measure
        import numpy as np
        return np.std(distances) / max(np.mean(distances), 1.0)
    
    def set_generation_callback(self, callback: Callable[[int, List[RouteChromosome], List[float]], None]):
        """Set callback for each generation"""
        self.generation_callback = callback
    
    def set_progress_callback(self, callback: Callable[[float, float], None]):
        """Set callback for progress updates"""
        self.progress_callback = callback


def test_genetic_optimizer():
    """Test function for genetic optimizer"""
    print("Testing Genetic Route Optimizer...")
    
    # Create minimal test graph
    test_graph = nx.Graph()
    nodes = [(1, -80.4094, 37.1299, 100), (2, -80.4000, 37.1300, 110),
             (3, -80.4050, 37.1350, 105), (4, -80.4100, 37.1250, 120)]
    
    for node_id, x, y, elev in nodes:
        test_graph.add_node(node_id, x=x, y=y, elevation=elev)
    
    edges = [(1, 2, 100), (2, 3, 150), (3, 4, 200), (4, 1, 180), (1, 3, 250), (2, 4, 220)]
    for n1, n2, length in edges:
        test_graph.add_edge(n1, n2, length=length)
    
    # Test configuration
    config = GAConfig(
        population_size=10,
        max_generations=20,
        verbose=True
    )
    
    # Test optimization
    optimizer = GeneticRouteOptimizer(test_graph, config)
    results = optimizer.optimize_route(1, 2.0, "elevation")
    
    print(f"✅ Optimization completed:")
    print(f"   Best fitness: {results.best_fitness:.4f}")
    print(f"   Total generations: {results.total_generations}")
    print(f"   Total time: {results.total_time:.2f}s")
    print(f"   Convergence: {results.convergence_reason}")
    
    # Test results
    assert results.best_chromosome is not None
    assert results.best_fitness > 0
    assert results.total_generations > 0
    assert len(results.population_history) > 0
    assert len(results.fitness_history) > 0
    
    print("✅ All genetic optimizer tests completed")


if __name__ == "__main__":
    test_genetic_optimizer()