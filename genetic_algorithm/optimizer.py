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

from .chromosome import RouteChromosome
from .population import PopulationInitializer
from .distance_compliant_population import DistanceCompliantPopulationInitializer
from .terrain_aware_initialization import TerrainAwarePopulationInitializer, TerrainAwareConfig
from .operators import GAOperators
from .constraint_preserving_operators import ConstraintPreservingOperators, RouteConstraints
from .fitness import GAFitnessEvaluator, FitnessObjective
from .performance import GASegmentCache
from .adaptive_mutation import AdaptiveMutationController, AdaptiveMutationConfig
from .restart_mechanisms import RestartMechanisms, RestartConfig
from .diversity_selection import DiversityPreservingSelector, DiversitySelectionConfig

# Enhanced 1m precision components (removed ga_precision_fitness - now integrated into ga_fitness)
try:
    from .operators import PrecisionAwareCrossover, PrecisionAwareMutation
    from .visualization import PrecisionComparisonVisualizer
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
    
    # Population initialization method
    use_distance_compliant_initialization: bool = True  # Use improved distance-compliant initializer
    
    # Segment usage constraints
    allow_bidirectional_segments: bool = True  # Allow using segments in both directions
    
    # Constraint-preserving operators
    use_constraint_preserving_operators: bool = True  # Use constraint-preserving crossover/mutation
    
    # Enhanced 1m precision settings
    enable_precision_enhancement: bool = False  # Disabled by default for stability
    precision_fitness_weight: float = 0.3
    micro_terrain_preference: float = 0.4
    elevation_bias_strength: float = 0.5
    generate_precision_visualizations: bool = False
    precision_comparison_interval: int = 25  # Generate comparison every N generations
    
    # Adaptive mutation settings
    enable_adaptive_mutation: bool = True  # Enable adaptive mutation rates
    adaptive_mutation_min_rate: float = 0.05  # Minimum mutation rate
    adaptive_mutation_max_rate: float = 0.50  # Maximum mutation rate
    adaptive_mutation_stagnation_threshold: int = 10  # Generations before considering stagnation
    adaptive_mutation_diversity_threshold: float = 0.3  # Diversity threshold for boosting
    
    # Terrain-aware initialization settings
    enable_terrain_aware_initialization: bool = True  # Enable terrain-aware population initialization
    terrain_elevation_gain_threshold: float = 30.0  # Minimum elevation gain for "high" nodes
    terrain_max_elevation_gain_threshold: float = 100.0  # Maximum elevation gain for "very high" nodes
    terrain_high_elevation_percentage: float = 0.4  # Percentage targeting high elevation
    terrain_very_high_elevation_percentage: float = 0.2  # Percentage targeting very high elevation
    
    # Restart mechanisms settings
    enable_restart_mechanisms: bool = True  # Enable restart mechanisms for escaping local optima
    restart_convergence_threshold: float = 0.01  # Fitness improvement threshold for convergence
    restart_stagnation_generations: int = 15  # Generations without improvement before restart
    restart_diversity_threshold: float = 0.2  # Minimum population diversity before restart
    restart_elite_retention_percentage: float = 0.2  # Percentage of elite to retain during restart
    restart_max_restarts: int = 2  # Maximum number of restarts per optimization
    
    # Diversity-preserving selection settings
    enable_diversity_selection: bool = True  # Enable diversity-preserving selection
    diversity_threshold: float = 0.3  # Minimum diversity between selected individuals
    diversity_weight: float = 0.3  # Weight of diversity in selection decisions
    diversity_fitness_weight: float = 0.7  # Weight of fitness in selection decisions
    diversity_elite_percentage: float = 0.1  # Percentage of best individuals to always select
    diversity_adaptive_threshold: bool = True  # Whether to adapt diversity threshold dynamically


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
    
    # Restart and diversity results
    restart_stats: Optional[Dict[str, Any]] = None
    diversity_stats: Optional[Dict[str, Any]] = None


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
        self.operators = GAOperators(graph, self.config.allow_bidirectional_segments)
        self.constraint_operators = None  # Will be initialized in _setup_optimization
        self.fitness_evaluator = None
        
        # Enhanced 1m precision components - disabled by default for testing
        self.precision_components_enabled = (PRECISION_ENHANCEMENT_AVAILABLE and 
                                           self.config.enable_precision_enhancement and
                                           not os.environ.get('DISABLE_PRECISION_ENHANCEMENT', False))
        
        if self.precision_components_enabled:
            self.precision_crossover = PrecisionAwareCrossover(graph)
            self.precision_mutation = PrecisionAwareMutation(graph)
            
            if self.config.generate_precision_visualizations:
                self.precision_visualizer = PrecisionComparisonVisualizer()
                self.precision_visualizations = []
            else:
                self.precision_visualizer = None
                self.precision_visualizations = []
        else:
            self.precision_crossover = None
            self.precision_mutation = None
            self.precision_visualizer = None
            self.precision_visualizations = []
        
        # Initialize adaptive mutation controller
        if self.config.enable_adaptive_mutation:
            adaptive_config = AdaptiveMutationConfig(
                base_mutation_rate=self.config.mutation_rate,
                min_mutation_rate=self.config.adaptive_mutation_min_rate,
                max_mutation_rate=self.config.adaptive_mutation_max_rate,
                stagnation_threshold=self.config.adaptive_mutation_stagnation_threshold,
                diversity_threshold=self.config.adaptive_mutation_diversity_threshold
            )
            self.adaptive_mutation_controller = AdaptiveMutationController(adaptive_config)
        else:
            self.adaptive_mutation_controller = None
        
        # Initialize restart mechanisms (will be set up in _setup_optimization)
        self.restart_mechanisms = None
        
        # Initialize diversity-preserving selector
        if self.config.enable_diversity_selection:
            diversity_config = DiversitySelectionConfig(
                diversity_threshold=self.config.diversity_threshold,
                diversity_weight=self.config.diversity_weight,
                fitness_weight=self.config.diversity_fitness_weight,
                elite_percentage=self.config.diversity_elite_percentage,
                adaptive_threshold=self.config.diversity_adaptive_threshold
            )
            self.diversity_selector = DiversityPreservingSelector(diversity_config)
        else:
            self.diversity_selector = None
        
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
            initial_stats = self.best_chromosome.get_route_stats()
            initial_distance = initial_stats['total_distance_km']
            initial_elevation = initial_stats['total_elevation_gain_m']
            print(f"✅ Initial population created")
            print(f"   Best fitness: {self.best_fitness:.4f}")
            print(f"   Average fitness: {sum(fitness_scores)/len(fitness_scores):.4f}")
            print(f"   Best distance: {initial_distance:.2f}km, elevation: {initial_elevation:.0f}m")
        
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
                best_stats = self.best_chromosome.get_route_stats()
                best_distance = best_stats['total_distance_km']
                best_elevation = best_stats['total_elevation_gain_m']
                print(f"   Gen {generation:3d}: Best={self.best_fitness:.4f}, "
                      f"Avg={avg_fitness:.4f}, Dist={best_distance:.2f}km, "
                      f"Elev={best_elevation:.0f}m, Time={gen_time:.2f}s")
            
            # Check for restart needed
            if self.restart_mechanisms and self.restart_mechanisms.check_restart_needed(generation, population, fitness_scores):
                population, restart_info = self.restart_mechanisms.execute_restart(population, fitness_scores, generation)
                
                # Re-evaluate population after restart
                fitness_scores = self._evaluate_population_with_precision(population, objective, distance_km)
                
                # Update best if restart found better solution
                current_best_idx = fitness_scores.index(max(fitness_scores))
                current_best_fitness = fitness_scores[current_best_idx]
                
                if current_best_fitness > self.best_fitness:
                    self.best_chromosome = population[current_best_idx].copy()
                    self.best_fitness = current_best_fitness
                    self.best_generation = generation
                
                if self.config.verbose:
                    print(f"   🔄 Restart completed: new best fitness = {self.best_fitness:.4f}")
            
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
            
            # Precision comparison functionality removed - micro-terrain analysis now integrated into main fitness evaluator
            
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
        
        # Add restart and diversity statistics
        if self.restart_mechanisms:
            results.restart_stats = self.restart_mechanisms.get_restart_stats()
        
        if self.diversity_selector:
            results.diversity_stats = self.diversity_selector.get_diversity_stats()
        
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
            
            # Report restart mechanisms performance
            if self.restart_mechanisms:
                restart_stats = self.restart_mechanisms.get_restart_stats()
                print(f"🔄 Restart Mechanisms:")
                print(f"   Restarts executed: {restart_stats['restart_count']}/{restart_stats['max_restarts']}")
                if restart_stats['restart_count'] > 0:
                    conv_stats = restart_stats['convergence_stats']
                    print(f"   Final stagnation: {conv_stats['stagnation_count']} generations")
                    print(f"   Final diversity: {conv_stats['recent_diversity']:.3f}")
            
            # Report diversity selection performance
            if self.diversity_selector:
                diversity_stats = self.diversity_selector.get_diversity_stats()
                print(f"🎯 Diversity Selection:")
                print(f"   Current diversity: {diversity_stats['current_avg_diversity']:.3f}")
                print(f"   Diversity trend: {diversity_stats['diversity_trend']:.4f}")
                print(f"   Diversity stability: {diversity_stats['diversity_stability']:.3f}")
            
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
        # Initialize population initializer (choose between terrain-aware, distance-compliant, and traditional)
        if self.config.enable_terrain_aware_initialization:
            if self.config.verbose:
                print("🏔️  Using terrain-aware population initialization")
            
            # Create terrain-aware config
            terrain_config = TerrainAwareConfig(
                elevation_gain_threshold=self.config.terrain_elevation_gain_threshold,
                max_elevation_gain_threshold=self.config.terrain_max_elevation_gain_threshold,
                high_elevation_percentage=self.config.terrain_high_elevation_percentage,
                very_high_elevation_percentage=self.config.terrain_very_high_elevation_percentage
            )
            
            self.population_initializer = TerrainAwarePopulationInitializer(
                self.graph, start_node, distance_km, terrain_config
            )
            
        elif self.config.use_distance_compliant_initialization:
            if self.config.verbose:
                print("🎯 Using distance-compliant population initialization")
            self.population_initializer = DistanceCompliantPopulationInitializer(self.graph, start_node, distance_km * 1.2)  # 20% buffer
        else:
            if self.config.verbose:
                print("🎲 Using traditional population initialization")
            self.population_initializer = PopulationInitializer(self.graph, start_node, self.config.allow_bidirectional_segments)
        
        # Initialize fitness evaluator with segment cache
        self.fitness_evaluator = GAFitnessEvaluator(
            objective, distance_km, self.segment_cache, 
            enable_micro_terrain=True, 
            allow_bidirectional_segments=self.config.allow_bidirectional_segments
        )
        
        # Initialize constraint-preserving operators if enabled
        if self.config.use_constraint_preserving_operators:
            if self.config.verbose:
                print("🔧 Using constraint-preserving genetic operators")
            
            constraints = RouteConstraints(
                min_distance_km=distance_km * 0.85,
                max_distance_km=distance_km * 1.15,
                start_node=start_node,
                must_return_to_start=True,
                must_be_connected=True,
                allow_bidirectional=self.config.allow_bidirectional_segments
            )
            
            self.constraint_operators = ConstraintPreservingOperators(self.graph, constraints)
        
        # Initialize restart mechanisms
        if self.config.enable_restart_mechanisms:
            restart_config = RestartConfig(
                convergence_threshold=self.config.restart_convergence_threshold,
                stagnation_generations=self.config.restart_stagnation_generations,
                diversity_threshold=self.config.restart_diversity_threshold,
                elite_retention_percentage=self.config.restart_elite_retention_percentage,
                max_restarts=self.config.restart_max_restarts
            )
            self.restart_mechanisms = RestartMechanisms(
                self.graph, start_node, distance_km, restart_config
            )
        else:
            if self.config.verbose:
                print("⚙️ Using standard genetic operators")
            self.constraint_operators = None
        
        # Reset tracking
        self.generation = 0
        self.population_history = []
        self.fitness_history = []
        self.best_chromosome = None
        self.best_fitness = 0.0
        self.best_generation = 0
        self.evaluation_times = []
        
        # Reset adaptive mutation controller
        if self.adaptive_mutation_controller:
            self.adaptive_mutation_controller.reset()
            if self.config.verbose:
                print("🧬 Adaptive mutation controller initialized")
    
    def _adapt_configuration(self, distance_km: float):
        """Adapt GA configuration based on problem size"""
        if not self.config.adaptive_sizing:
            return

        # Adjust population size based on distance
        if distance_km < 3.0:
            self.config.population_size = max(50, self.config.population_size // 2)
            self.config.max_generations = max(100, self.config.max_generations // 2)
        elif distance_km > 20.0:
            # Very long routes: smaller population to avoid timeouts during population creation
            self.config.population_size = max(30, self.config.population_size // 3)
            self.config.max_generations = max(50, self.config.max_generations // 2)
            if self.config.verbose:
                print(f"🏃‍♂️ Adapted for very long route: population={self.config.population_size}, generations={self.config.max_generations}")
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
        # Use standard fitness evaluation with integrated micro-terrain analysis
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
        
        # Update adaptive mutation rate if enabled
        current_mutation_rate = self.config.mutation_rate
        if self.adaptive_mutation_controller:
            current_best_fitness = max(fitness_scores) if fitness_scores else 0.0
            current_mutation_rate = self.adaptive_mutation_controller.update(
                self.generation, population, fitness_scores, current_best_fitness
            )
            
            # Report adaptive mutation diagnostics
            if self.config.verbose and self.generation % 10 == 0:
                diagnostics = self.adaptive_mutation_controller.get_diagnostics()
                print(f"   🧬 Adaptive mutation: rate={current_mutation_rate:.3f} "
                      f"({diagnostics['mutation_intensity']}), "
                      f"diversity={diagnostics['current_diversity']:.3f}, "
                      f"stagnation={diagnostics['stagnation_count']}")
        
        # Elitism - preserve best individuals
        elite_indices = sorted(range(len(fitness_scores)), 
                             key=lambda i: fitness_scores[i], reverse=True)[:self.config.elite_size]
        elite = [population[i].copy() for i in elite_indices]
        new_population.extend(elite)
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Selection with diversity consideration
            if self.diversity_selector:
                parent1 = self.diversity_selector.tournament_selection_with_diversity(
                    population, fitness_scores, self.config.tournament_size
                )
                parent2 = self.diversity_selector.tournament_selection_with_diversity(
                    population, fitness_scores, self.config.tournament_size
                )
            else:
                parent1 = self.operators.tournament_selection(population, self.config.tournament_size)
                parent2 = self.operators.tournament_selection(population, self.config.tournament_size)
            
            # Use constraint-preserving operators if available
            if self.constraint_operators:
                # Use constraint-preserving crossover
                offspring1, offspring2 = self.constraint_operators.connection_point_crossover(
                    parent1, parent2, self.config.crossover_rate
                )
                
                # Use constraint-preserving mutation with adaptive rate
                offspring1 = self.constraint_operators.distance_neutral_mutation(
                    offspring1, current_mutation_rate
                )
                offspring2 = self.constraint_operators.distance_neutral_mutation(
                    offspring2, current_mutation_rate
                )
            else:
                # Use standard operators
                offspring1, offspring2 = self.operators.segment_exchange_crossover(
                    parent1, parent2, self.config.crossover_rate
                )
                
                # Apply adaptive mutation with strategy selection
                offspring1 = self._apply_adaptive_mutation(offspring1, current_mutation_rate)
                offspring2 = self._apply_adaptive_mutation(offspring2, current_mutation_rate)
            
            # Add to population
            new_population.extend([offspring1, offspring2])
        
        # Trim to exact population size
        new_population = new_population[:self.config.population_size]
        
        # Evaluate new population with precision enhancement
        new_fitness_scores = self._evaluate_population_with_precision(
            new_population, self.fitness_evaluator.objective.value, self.fitness_evaluator.target_distance_km
        )
        
        # Apply population filtering to remove poor performers
        filtered_population, filtered_fitness_scores = self._apply_population_filtering(
            new_population, new_fitness_scores
        )
        
        # Apply diversity-preserving selection if enabled
        if self.diversity_selector:
            filtered_population = self.diversity_selector.select_population(
                filtered_population, filtered_fitness_scores, self.generation
            )
            
            # Re-evaluate if population changed
            if len(filtered_population) != len(filtered_fitness_scores):
                filtered_fitness_scores = self._evaluate_population_with_precision(
                    filtered_population, self.fitness_evaluator.objective.value, self.fitness_evaluator.target_distance_km
                )
        
        return filtered_population, filtered_fitness_scores
    
    def _apply_adaptive_mutation(self, chromosome: RouteChromosome, mutation_rate: float) -> RouteChromosome:
        """Apply adaptive mutation with strategy selection
        
        Args:
            chromosome: Chromosome to mutate
            mutation_rate: Current adaptive mutation rate
            
        Returns:
            Mutated chromosome
        """
        if not self.adaptive_mutation_controller:
            # Fall back to standard mutation
            return self.operators.segment_replacement_mutation(chromosome, mutation_rate)
        
        # Get mutation strategy weights
        strategies = self.adaptive_mutation_controller.get_mutation_strategies()
        
        # Select mutation strategy based on weights
        strategy_choice = random.random()
        cumulative_weight = 0.0
        
        for strategy, weight in strategies.items():
            cumulative_weight += weight
            if strategy_choice <= cumulative_weight:
                break
        
        # Apply selected mutation strategy
        if strategy == 'segment_replacement':
            return self.operators.segment_replacement_mutation(chromosome, mutation_rate)
        elif strategy == 'route_extension':
            return self.operators.route_extension_mutation(
                chromosome, self.fitness_evaluator.target_distance_km, mutation_rate
            )
        elif strategy == 'elevation_bias':
            return self.operators.elevation_bias_mutation(chromosome, mutation_rate)
        elif strategy == 'long_range_exploration':
            return self._apply_long_range_mutation(chromosome, mutation_rate)
        else:
            # Default to segment replacement
            return self.operators.segment_replacement_mutation(chromosome, mutation_rate)
    
    def _apply_long_range_mutation(self, chromosome: RouteChromosome, mutation_rate: float) -> RouteChromosome:
        """Apply long-range exploration mutation
        
        Args:
            chromosome: Chromosome to mutate
            mutation_rate: Current mutation rate
            
        Returns:
            Mutated chromosome with long-range exploration
        """
        if random.random() > mutation_rate or not chromosome.segments:
            return chromosome
        
        mutated = chromosome.copy()
        mutated.creation_method = "long_range_exploration_mutation"
        
        # Find high-elevation nodes that are not currently in the route
        route_nodes = set()
        for segment in chromosome.segments:
            route_nodes.add(segment.start_node)
            route_nodes.add(segment.end_node)
        
        # Find high-elevation nodes not in current route
        high_elevation_nodes = []
        for node_id, node_data in self.graph.nodes(data=True):
            if (node_id not in route_nodes and 
                node_data.get('elevation', 0) > 650):  # Above 650m elevation
                high_elevation_nodes.append(node_id)
        
        if not high_elevation_nodes:
            # Fall back to standard mutation
            return self.operators.segment_replacement_mutation(chromosome, mutation_rate)
        
        # Select a random high-elevation target
        target_node = random.choice(high_elevation_nodes)
        
        # Try to modify a segment to go toward the target
        if mutated.segments:
            # Replace a random segment with a path toward the target
            segment_index = random.randint(0, len(mutated.segments) - 1)
            old_segment = mutated.segments[segment_index]
            
            # Try to create a path from segment start to target
            try:
                if nx.has_path(self.graph, old_segment.start_node, target_node):
                    # Create path to target
                    path = nx.shortest_path(self.graph, old_segment.start_node, target_node)
                    
                    # Only use if path is reasonable length (not too long)
                    if len(path) <= 5:
                        from .chromosome import RouteSegment
                        new_segment = RouteSegment(old_segment.start_node, target_node, path)
                        new_segment.calculate_properties(self.graph)
                        mutated.segments[segment_index] = new_segment
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # Fall back to standard mutation
                return self.operators.segment_replacement_mutation(chromosome, mutation_rate)
        
        return mutated
    
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
    
    def _apply_population_filtering(self, population: List[RouteChromosome], 
                                   fitness_scores: List[float]) -> Tuple[List[RouteChromosome], List[float]]:
        """Apply population filtering to remove poor performers
        
        Args:
            population: Population to filter
            fitness_scores: Corresponding fitness scores
            
        Returns:
            Tuple of (filtered_population, filtered_fitness_scores)
        """
        if not population or not fitness_scores:
            return population, fitness_scores
        
        original_size = len(population)
        
        # Apply survival selection with adaptive fitness threshold
        avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0
        adaptive_threshold = max(0.02, avg_fitness * 0.25)  # 25% of average, minimum 0.02
        
        if self.config.verbose and original_size > 10:  # Only show for substantial populations
            print(f"   📊 Survival filtering: avg_fitness={avg_fitness:.3f}, threshold={adaptive_threshold:.3f}")
        
        filtered_population, filtered_fitness_scores = self.operators.survival_selection(
            population, fitness_scores,
            survival_rate=0.8,  # Keep 80% of population
            min_fitness_threshold=adaptive_threshold  # Adaptive threshold based on population quality
        )
        
        # If population became too small, replenish with new random chromosomes
        if len(filtered_population) < self.config.population_size * 0.5:
            needed = self.config.population_size - len(filtered_population)
            
            if self.config.verbose:
                print(f"   ⚠️ Population heavily filtered ({original_size} → {len(filtered_population)})")
                print(f"   🔄 Generating {needed} new chromosomes to maintain diversity")
            
            # Generate new chromosomes to maintain population size
            try:
                new_chromosomes = self.population_initializer.create_population(
                    needed, self.fitness_evaluator.target_distance_km
                )
                
                if new_chromosomes:
                    # Evaluate new chromosomes
                    new_fitness_scores = self.fitness_evaluator.evaluate_population(new_chromosomes, self.graph)
                    
                    # Add to filtered population
                    filtered_population.extend(new_chromosomes)
                    filtered_fitness_scores.extend(new_fitness_scores)
                    
            except Exception as e:
                if self.config.verbose:
                    print(f"   ⚠️ Failed to generate new chromosomes: {e}")
        
        # Ensure we don't exceed target population size
        if len(filtered_population) > self.config.population_size:
            # Keep best chromosomes if we have too many
            combined = list(zip(filtered_population, filtered_fitness_scores))
            combined.sort(key=lambda x: x[1], reverse=True)  # Sort by fitness descending
            combined = combined[:self.config.population_size]
            
            filtered_population = [x[0] for x in combined]
            filtered_fitness_scores = [x[1] for x in combined]
        
        return filtered_population, filtered_fitness_scores
    
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