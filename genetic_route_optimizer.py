#!/usr/bin/env python3
"""
Genetic Algorithm Route Optimizer
Main genetic algorithm implementation for route optimization
"""

import os
import time
import random
import math
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import networkx as nx

# Configure logging
logger = logging.getLogger(__name__)

from ga_chromosome import RouteChromosome
from ga_population import PopulationInitializer
from ga_operators import GAOperators
from ga_fitness import GAFitnessEvaluator, FitnessObjective
from ga_segment_cache import GASegmentCache

# Enhanced 1m precision components
try:
    from ga_precision_fitness import PrecisionElevationAnalyzer, EnhancedGAFitnessEvaluator
    from ga_precision_operators import PrecisionAwareCrossover, PrecisionAwareMutation
    from ga_precision_visualizer import PrecisionComparisonVisualizer
    PRECISION_ENHANCEMENT_AVAILABLE = True
except ImportError:
    PRECISION_ENHANCEMENT_AVAILABLE = False


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
    
    # Enhanced 1m precision settings
    enable_precision_enhancement: bool = False  # Disabled by default for stability
    precision_fitness_weight: float = 0.3
    micro_terrain_preference: float = 0.4
    elevation_bias_strength: float = 0.5
    generate_precision_visualizations: bool = False
    precision_comparison_interval: int = 25  # Generate comparison every N generations


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
    
    # Enhanced 1m precision results
    precision_benefits: Optional[Dict[str, Any]] = None
    micro_terrain_features: Optional[Dict[str, Any]] = None
    precision_visualizations: Optional[List[str]] = None
    elevation_profile_comparison: Optional[Dict[str, Any]] = None


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
        
        # Initialize segment cache for performance optimization
        self.segment_cache = GASegmentCache(max_size=5000)
        
        # Initialize components
        self.population_initializer = None
        self.operators = GAOperators(graph)
        self.fitness_evaluator = None
        
        # Enhanced 1m precision components - disabled by default for testing
        self.precision_components_enabled = (PRECISION_ENHANCEMENT_AVAILABLE and 
                                           self.config.enable_precision_enhancement and
                                           not os.environ.get('DISABLE_PRECISION_ENHANCEMENT', False))
        
        if self.precision_components_enabled:
            self.precision_analyzer = PrecisionElevationAnalyzer()
            self.enhanced_fitness_evaluator = EnhancedGAFitnessEvaluator(graph, enable_micro_terrain=True)
            self.precision_crossover = PrecisionAwareCrossover(graph, self.precision_analyzer)
            self.precision_mutation = PrecisionAwareMutation(graph, self.precision_analyzer)
            
            if self.config.generate_precision_visualizations:
                self.precision_visualizer = PrecisionComparisonVisualizer()
                self.precision_visualizations = []
            else:
                self.precision_visualizer = None
                self.precision_visualizations = []
        else:
            self.precision_analyzer = None
            self.enhanced_fitness_evaluator = None
            self.precision_crossover = None
            self.precision_mutation = None
            self.precision_visualizer = None
            self.precision_visualizations = []
        
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
        
        # Evaluate initial population with precision enhancement
        fitness_scores = self._evaluate_population_with_precision(population, objective, distance_km)
        
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
            if self.config.verbose:
                avg_fitness = sum(fitness_scores) / len(fitness_scores)
                print(f"   Gen {generation:3d}: Best={self.best_fitness:.4f}, "
                      f"Avg={avg_fitness:.4f}, Time={gen_time:.2f}s")
            
            # Generation callback
            if self.generation_callback:
                self.generation_callback(generation, population, fitness_scores)
            
            # Generate visualization
            if visualizer and generation % 25 == 0:
                self._generate_visualization(population, fitness_scores, generation, visualizer)
            
            # Generate precision comparison visualization
            if (self.precision_components_enabled and self.precision_visualizer and 
                self.config.generate_precision_visualizations and 
                generation % self.config.precision_comparison_interval == 0):
                self._generate_precision_visualization(population, fitness_scores, generation)
            
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
        
        # Add precision enhancement results
        if self.precision_components_enabled:
            # Generate final precision analysis
            best_route_coords = self._chromosome_to_coordinates(self.best_chromosome)
            
            if self.enhanced_fitness_evaluator:
                try:
                    precision_comparison = self.enhanced_fitness_evaluator.compare_precision_benefits(
                        best_route_coords
                    )
                    results.precision_benefits = precision_comparison.get('precision_benefits', {})
                    results.micro_terrain_features = precision_comparison.get('high_resolution', {}).get('elevation_profile', {}).get('micro_terrain_features', {})
                    results.elevation_profile_comparison = precision_comparison
                except Exception as e:
                    logger.warning(f"Failed to generate precision comparison: {e}")
            
            # Add precision visualizations
            if self.precision_visualizations:
                results.precision_visualizations = self.precision_visualizations.copy()
            
            # Generate final precision comparison visualization
            if self.precision_visualizer:
                try:
                    final_viz = self.precision_visualizer.create_precision_comparison_visualization(
                        best_route_coords, self.graph, " - Final Best Route"
                    )
                    if final_viz:
                        results.precision_visualizations = results.precision_visualizations or []
                        results.precision_visualizations.append(final_viz)
                except Exception as e:
                    logger.warning(f"Failed to generate final precision visualization: {e}")
        
        if self.config.verbose:
            print(f"🏁 Optimization completed:")
            print(f"   Best fitness: {self.best_fitness:.4f}")
            print(f"   Found at generation: {self.best_generation}")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Convergence: {convergence_reason}")
            
            # Report segment cache performance
            cache_info = self.segment_cache.get_cache_info()
            if cache_info['total_requests'] > 0:
                print(f"⚡ Segment Cache Performance:")
                print(f"   Cache hits: {cache_info['cache_hits']:,} / {cache_info['total_requests']:,} ({cache_info['hit_rate_percent']:.1f}%)")
                print(f"   Time saved: {cache_info.get('total_time_saved_seconds', 0):.2f}s")
                print(f"   Segments cached: {cache_info['cache_size']:,}")
            
            # Report precision benefits if available
            if (self.precision_components_enabled and results.precision_benefits):
                benefits = results.precision_benefits
                print(f"🔬 Precision Enhancement Benefits:")
                print(f"   Micro-features discovered: {benefits.get('micro_features_discovered', 0)}")
                print(f"   Fitness improvement: {benefits.get('fitness_improvement', 0):.3f}")
                print(f"   Resolution factor: {benefits.get('resolution_factor', 1):.1f}x")
                
                if results.precision_visualizations:
                    print(f"   Visualizations saved: {len(results.precision_visualizations)}")
        
        return results
    
    def _setup_optimization(self, start_node: int, distance_km: float, objective: str):
        """Setup optimization components"""
        # Initialize population initializer
        self.population_initializer = PopulationInitializer(self.graph, start_node)
        
        # Initialize fitness evaluator with segment cache
        self.fitness_evaluator = GAFitnessEvaluator(objective, distance_km, self.segment_cache)
        
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
    
    def _evaluate_population_with_precision(self, population: List[RouteChromosome], 
                                           objective: str, distance_km: float) -> List[float]:
        """Evaluate population using precision-enhanced fitness if available
        
        Args:
            population: Population to evaluate
            objective: Fitness objective
            distance_km: Target distance
            
        Returns:
            List of fitness scores
        """
        if self.precision_components_enabled and self.enhanced_fitness_evaluator:
            # Use precision-enhanced fitness evaluation
            fitness_scores = []
            
            for chromosome in population:
                try:
                    # Convert chromosome to coordinates
                    route_coords = self._chromosome_to_coordinates(chromosome)
                    
                    # Evaluate with precision enhancement
                    fitness_result = self.enhanced_fitness_evaluator.evaluate_route_fitness(
                        route_coords, objective, distance_km
                    )
                    
                    # Apply precision fitness weight
                    base_fitness = fitness_result.get('total_fitness', 0.0)
                    precision_bonus = fitness_result.get('components', {}).get('precision_bonus', 0.0)
                    
                    # Weighted combination of base fitness and precision enhancement
                    enhanced_fitness = (
                        base_fitness * (1 - self.config.precision_fitness_weight) +
                        precision_bonus * self.config.precision_fitness_weight
                    )
                    
                    fitness_scores.append(enhanced_fitness)
                    
                    # Store precision data in chromosome for later analysis
                    if hasattr(chromosome, 'precision_data'):
                        chromosome.precision_data = {
                            'micro_terrain_features': fitness_result.get('elevation_profile', {}).get('micro_terrain_features', {}),
                            'precision_benefits': fitness_result.get('components', {}),
                            'fitness_components': fitness_result.get('components', {})
                        }
                
                except Exception as e:
                    logger.warning(f"Precision fitness evaluation failed for chromosome: {e}")
                    # Fallback to standard fitness evaluation
                    fallback_fitness = self.fitness_evaluator.evaluate_chromosome(chromosome)
                    fitness_scores.append(fallback_fitness)
            
            return fitness_scores
        else:
            # Use standard fitness evaluation
            return self.fitness_evaluator.evaluate_population(population, self.graph)
    
    def _chromosome_to_coordinates(self, chromosome: RouteChromosome) -> List[Tuple[float, float]]:
        """Convert chromosome route to coordinate list
        
        Args:
            chromosome: Route chromosome
            
        Returns:
            List of (lat, lon) coordinate pairs
        """
        coordinates = []
        
        # Use get_route_nodes() to get the route nodes
        route_nodes = chromosome.get_route_nodes()
        
        for node_id in route_nodes:
            if node_id in self.graph:
                node_data = self.graph.nodes[node_id]
                lat = node_data.get('y', 0.0)  # y is latitude
                lon = node_data.get('x', 0.0)  # x is longitude
                coordinates.append((lat, lon))
        
        return coordinates
    
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
            
            # Use standard crossover (precision operators need more integration work)
            offspring1, offspring2 = self.operators.segment_exchange_crossover(
                parent1, parent2, self.config.crossover_rate
            )
            
            # Use standard mutation (precision operators need more integration work)
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
        
        # Evaluate new population with precision enhancement
        new_fitness_scores = self._evaluate_population_with_precision(
            new_population, self.fitness_evaluator.objective.value, self.fitness_evaluator.target_distance_km
        )
        
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
    
    def _generate_precision_visualization(self, population: List[RouteChromosome], 
                                        fitness_scores: List[float], generation: int):
        """Generate precision comparison visualization for current generation"""
        try:
            # Get best chromosome from current generation
            best_idx = fitness_scores.index(max(fitness_scores))
            best_chromosome = population[best_idx]
            
            # Convert to coordinates
            route_coords = self._chromosome_to_coordinates(best_chromosome)
            
            # Generate precision comparison visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            title_suffix = f" - Generation {generation}"
            
            visualization_path = self.precision_visualizer.create_precision_comparison_visualization(
                route_coords, self.graph, title_suffix
            )
            
            if visualization_path:
                self.precision_visualizations.append(visualization_path)
                
                if self.config.verbose and generation % 50 == 0:
                    print(f"📊 Saved precision visualization: {visualization_path}")
                    
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️ Precision visualization failed: {e}")
    
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