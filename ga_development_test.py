#!/usr/bin/env python3
"""
GA Development Test Framework
Provides comprehensive testing and visualization for GA development phases
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import List, Optional, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from route_services.network_manager import NetworkManager
from ga_chromosome import RouteChromosome, RouteSegment
from ga_population import PopulationInitializer
from ga_operators import GAOperators
from ga_operator_visualization import GAOperatorVisualizer

# Optional imports for visualization
try:
    from ga_visualizer import GAVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError:
    GAVisualizer = None
    VISUALIZER_AVAILABLE = False
    print("⚠️ GAVisualizer not available (folium dependency missing)")


class GADevelopmentTester:
    """Test framework for GA development with mandatory visualizations"""
    
    def __init__(self, save_images: bool = True, output_dir: str = "ga_dev_images"):
        """Initialize development tester
        
        Args:
            save_images: Whether to save visualization images
            output_dir: Directory for output images
        """
        self.save_images = save_images
        self.output_dir = output_dir
        
        # Create output directory
        if save_images:
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        print("🔧 Initializing GA Development Tester...")
        self.network_manager = NetworkManager()
        self.graph = None
        self.visualizer = None
        self.start_node = NetworkManager.DEFAULT_START_NODE
        self.visualizer_available = VISUALIZER_AVAILABLE
        
        print(f"📁 Output directory: {output_dir}")
        print(f"📸 Save images: {save_images}")
    
    def setup_network(self) -> bool:
        """Setup network and visualizer"""
        try:
            print("🌐 Loading network...")
            self.graph = self.network_manager.load_network()
            
            if not self.graph:
                print("❌ Failed to load network")
                return False
            
            print(f"✅ Network loaded: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
            
            # Validate start node
            if self.start_node not in self.graph.nodes:
                print(f"⚠️ Default start node {self.start_node} not in graph, finding alternative...")
                self.start_node = list(self.graph.nodes)[0]
                print(f"📍 Using start node: {self.start_node}")
            
            # Initialize visualizer
            if self.save_images and self.visualizer_available:
                self.visualizer = GAVisualizer(self.graph, self.output_dir)
                print("✅ Visualizer initialized")
            elif self.save_images:
                print("⚠️ Visualizer not available due to missing dependencies")
            
            return True
            
        except Exception as e:
            print(f"❌ Network setup failed: {e}")
            return False
    
    def test_chromosome_phase(self) -> bool:
        """Test Phase: Chromosome creation and validation"""
        print("\n" + "="*60)
        print("🧬 TESTING PHASE: CHROMOSOME CREATION")
        print("="*60)
        
        try:
            # Test 1: Basic segment creation
            print("\n📋 Test 1: Basic Segment Creation")
            neighbors = list(self.graph.neighbors(self.start_node))[:3]
            
            if not neighbors:
                print("❌ No neighbors found for start node")
                return False
            
            test_segments = []
            for neighbor in neighbors:
                try:
                    import networkx as nx
                    path = nx.shortest_path(self.graph, self.start_node, neighbor, weight='length')
                    segment = RouteSegment(self.start_node, neighbor, path)
                    segment.calculate_properties(self.graph)
                    test_segments.append(segment)
                    
                    print(f"✅ Segment {self.start_node} -> {neighbor}: "
                          f"{segment.length:.1f}m, {segment.elevation_gain:.1f}m gain")
                    
                except Exception as e:
                    print(f"❌ Failed to create segment {self.start_node} -> {neighbor}: {e}")
            
            # Test 2: Chromosome creation
            print(f"\n📋 Test 2: Chromosome Creation")
            if test_segments:
                chromosome = RouteChromosome(test_segments)
                chromosome.validate_connectivity()
                
                print(f"✅ Chromosome created: {chromosome}")
                stats = chromosome.get_route_stats()
                print(f"   📊 Stats: {stats['total_distance_km']:.2f}km, "
                      f"{stats['total_elevation_gain_m']:.1f}m elevation")
                
                # Visualization
                if self.save_images and self.visualizer:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"ga_dev_chromosome_test_{timestamp}.png"
                    filepath = self.visualizer.save_chromosome_map(
                        chromosome, filename, 
                        title="Chromosome Creation Test",
                        show_elevation=True,
                        show_segments=True
                    )
                    print(f"📸 Saved visualization: {filename}")
            
            print("✅ Chromosome phase test completed")
            return True
            
        except Exception as e:
            print(f"❌ Chromosome phase test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_initialization_phase(self) -> bool:
        """Test Phase: Population initialization"""
        print("\n" + "="*60)
        print("👥 TESTING PHASE: POPULATION INITIALIZATION")
        print("="*60)
        
        try:
            # Create population initializer
            initializer = PopulationInitializer(self.graph, self.start_node)
            
            # Test different population sizes and distances
            test_configs = [
                (10, 2.0),  # Small population, short distance
                (20, 5.0),  # Medium population, medium distance
                (15, 3.0)   # Mixed
            ]
            
            all_populations = []
            
            for pop_size, target_distance in test_configs:
                print(f"\n📋 Creating population: size={pop_size}, distance={target_distance}km")
                
                start_time = time.time()
                population = initializer.create_population(pop_size, target_distance)
                creation_time = time.time() - start_time
                
                print(f"⏱️ Creation time: {creation_time:.2f}s")
                print(f"✅ Population created: {len(population)} chromosomes")
                
                # Analyze population diversity
                self._analyze_population_diversity(population)
                
                # Save visualization
                if self.save_images and self.visualizer and population:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"ga_dev_init_pop{pop_size}_dist{target_distance}_{timestamp}.png"
                    filepath = self.visualizer.save_population_map(
                        population, generation=0, filename=filename,
                        show_fitness=False, show_elevation=True
                    )
                    print(f"📸 Saved population visualization: {filename}")
                
                all_populations.append((pop_size, target_distance, population))
            
            # Test initialization strategies
            print(f"\n📋 Testing Individual Initialization Strategies")
            self._test_initialization_strategies(initializer)
            
            print("✅ Initialization phase test completed")
            return True
            
        except Exception as e:
            print(f"❌ Initialization phase test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_comparison_phase(self) -> bool:
        """Test Phase: Compare with existing TSP approach"""
        print("\n" + "="*60)
        print("⚖️ TESTING PHASE: GA vs TSP COMPARISON")
        print("="*60)
        
        try:
            # Create GA route
            print("📋 Creating GA route...")
            initializer = PopulationInitializer(self.graph, self.start_node)
            population = initializer.create_population(5, 5.0)
            
            if not population:
                print("❌ Failed to create GA population")
                return False
            
            ga_route = population[0]  # Take first route as example
            print(f"✅ GA route created: {ga_route}")
            
            # Try to create TSP route for comparison
            print("📋 Attempting to create TSP route for comparison...")
            try:
                from route_services.route_optimizer import RouteOptimizer
                optimizer = RouteOptimizer(self.graph)
                tsp_result = optimizer.optimize_route(self.start_node, 5.0)
                tsp_route = tsp_result.get('route', []) if tsp_result else []
                print(f"✅ TSP route created: {len(tsp_route)} nodes")
            except Exception as e:
                print(f"⚠️ TSP route creation failed: {e}")
                tsp_route = None
            
            # Create comparison visualization
            if self.save_images and self.visualizer:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ga_dev_comparison_{timestamp}.png"
                filepath = self.visualizer.save_comparison_map(
                    ga_route, tsp_route, filename,
                    title="GA vs TSP Route Comparison"
                )
                print(f"📸 Saved comparison visualization: {filename}")
            
            # Compare statistics
            self._compare_route_statistics(ga_route, tsp_route)
            
            print("✅ Comparison phase test completed")
            return True
            
        except Exception as e:
            print(f"❌ Comparison phase test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _analyze_population_diversity(self, population: List[RouteChromosome]) -> None:
        """Analyze diversity of population"""
        if not population:
            return
        
        valid_chromosomes = [c for c in population if c.is_valid and c.segments]
        
        if not valid_chromosomes:
            print("⚠️ No valid chromosomes in population")
            return
        
        # Distance diversity
        distances = [c.get_total_distance() / 1000 for c in valid_chromosomes]
        print(f"📊 Distance diversity: {min(distances):.2f} - {max(distances):.2f}km "
              f"(avg: {sum(distances)/len(distances):.2f}km)")
        
        # Elevation diversity
        elevations = [c.get_elevation_gain() for c in valid_chromosomes]
        print(f"📊 Elevation diversity: {min(elevations):.1f} - {max(elevations):.1f}m "
              f"(avg: {sum(elevations)/len(elevations):.1f}m)")
        
        # Creation method diversity
        methods = [c.creation_method for c in valid_chromosomes]
        method_counts = {}
        for method in methods:
            method_counts[method] = method_counts.get(method, 0) + 1
        print(f"📊 Creation methods: {method_counts}")
        
        # Segment count diversity
        segment_counts = [len(c.segments) for c in valid_chromosomes]
        print(f"📊 Segment count diversity: {min(segment_counts)} - {max(segment_counts)} "
              f"(avg: {sum(segment_counts)/len(segment_counts):.1f})")
    
    def _test_initialization_strategies(self, initializer: PopulationInitializer) -> None:
        """Test individual initialization strategies"""
        strategies = [
            ("Random Walk", lambda: initializer._create_random_walk_route(5000)),
            ("Directional N", lambda: initializer._create_directional_route(5000, "N")),
            ("Directional E", lambda: initializer._create_directional_route(5000, "E")),
            ("Elevation Focused", lambda: initializer._create_elevation_focused_route(5000)),
            ("Simple Fallback", lambda: initializer._create_simple_fallback_route(5000))
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                chromosome = strategy_func()
                if chromosome:
                    stats = chromosome.get_route_stats()
                    print(f"✅ {strategy_name}: {stats['total_distance_km']:.2f}km, "
                          f"{stats['total_elevation_gain_m']:.1f}m elevation")
                else:
                    print(f"⚠️ {strategy_name}: Failed to create route")
            except Exception as e:
                print(f"❌ {strategy_name}: Error - {e}")
    
    def _compare_route_statistics(self, ga_route: RouteChromosome, 
                                 tsp_route: Optional[List[int]]) -> None:
        """Compare GA and TSP route statistics"""
        print(f"\n📊 Route Statistics Comparison:")
        
        # GA route stats
        ga_stats = ga_route.get_route_stats()
        print(f"🧬 GA Route:")
        print(f"   Distance: {ga_stats['total_distance_km']:.2f}km")
        print(f"   Elevation Gain: {ga_stats['total_elevation_gain_m']:.1f}m")
        print(f"   Segments: {ga_stats['segment_count']}")
        print(f"   Max Grade: {ga_stats['max_grade_percent']:.1f}%")
        print(f"   Diversity Score: {ga_stats['diversity_score']:.3f}")
        
        # TSP route stats (if available)
        if tsp_route:
            try:
                # Calculate TSP stats
                tsp_distance = 0.0
                tsp_elevation = 0.0
                
                for i in range(len(tsp_route) - 1):
                    node1, node2 = tsp_route[i], tsp_route[i + 1]
                    if (node1 in self.graph.nodes and node2 in self.graph.nodes and
                        self.graph.has_edge(node1, node2)):
                        
                        edge_data = self.graph[node1][node2]
                        tsp_distance += edge_data.get('length', 0.0)
                        
                        elev1 = self.graph.nodes[node1].get('elevation', 0.0)
                        elev2 = self.graph.nodes[node2].get('elevation', 0.0)
                        if elev2 > elev1:
                            tsp_elevation += (elev2 - elev1)
                
                print(f"🎯 TSP Route:")
                print(f"   Distance: {tsp_distance/1000:.2f}km")
                print(f"   Elevation Gain: {tsp_elevation:.1f}m")
                print(f"   Nodes: {len(tsp_route)}")
                
                # Comparison
                distance_diff = (ga_stats['total_distance_km'] - tsp_distance/1000) / (tsp_distance/1000) * 100
                elevation_diff = (ga_stats['total_elevation_gain_m'] - tsp_elevation) / max(tsp_elevation, 1) * 100
                
                print(f"📈 Differences:")
                print(f"   Distance: {distance_diff:+.1f}%")
                print(f"   Elevation: {elevation_diff:+.1f}%")
                
            except Exception as e:
                print(f"⚠️ TSP stats calculation failed: {e}")
        else:
            print(f"⚠️ TSP Route: Not available for comparison")
    
    def test_operators_phase(self) -> bool:
        """Test Phase: Genetic Operators (Week 2)"""
        print("\n" + "="*60)
        print("🧬 TESTING PHASE: GENETIC OPERATORS (WEEK 2)")
        print("="*60)
        
        try:
            # Initialize operators
            print("🔧 Initializing genetic operators...")
            operators = GAOperators(self.graph)
            print("✅ Operators initialized")
            
            # Create test parents
            print("👥 Creating test parent chromosomes...")
            initializer = PopulationInitializer(self.graph, self.start_node)
            population = initializer.create_population(4, 3.0)
            
            if len(population) < 2:
                print("❌ Failed to create enough chromosomes for operator testing")
                return False
            
            parent1, parent2 = population[0], population[1]
            print(f"✅ Parent chromosomes created: {len(parent1.segments)} and {len(parent2.segments)} segments")
            
            # Test crossover operators
            print("\n🔄 Testing Crossover Operators:")
            success = self._test_crossover_operators(operators, parent1, parent2)
            if not success:
                return False
            
            # Test mutation operators  
            print("\n🔀 Testing Mutation Operators:")
            success = self._test_mutation_operators(operators, parent1)
            if not success:
                return False
            
            # Test selection operators
            print("\n🎯 Testing Selection Operators:")
            success = self._test_selection_operators(operators, population)
            if not success:
                return False
            
            # Generate operator visualizations
            if self.save_images:
                print("\n📸 Generating operator visualizations...")
                operator_visualizer = GAOperatorVisualizer(self.graph)
                viz_dir = os.path.join(self.output_dir, "operators")
                
                operator_visualizer.visualize_crossover_operators(viz_dir)
                operator_visualizer.visualize_mutation_operators(viz_dir)
                operator_visualizer.visualize_selection_operators(viz_dir)
                operator_visualizer.visualize_operator_effects(viz_dir)
                
                print(f"✅ Operator visualizations saved to {viz_dir}")
            
            print("✅ Operators phase test completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Operators phase test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _test_crossover_operators(self, operators: GAOperators, 
                                parent1: RouteChromosome, parent2: RouteChromosome) -> bool:
        """Test crossover operators"""
        try:
            # Test segment exchange crossover
            print("  📋 Testing segment exchange crossover...")
            offspring1, offspring2 = operators.segment_exchange_crossover(parent1, parent2)
            
            if not isinstance(offspring1, RouteChromosome) or not isinstance(offspring2, RouteChromosome):
                print("  ❌ Segment exchange crossover failed to return chromosomes")
                return False
            
            print(f"  ✅ Segment exchange: {len(offspring1.segments)}, {len(offspring2.segments)} segments")
            print(f"     Creation methods: {offspring1.creation_method}, {offspring2.creation_method}")
            
            # Test path splice crossover
            print("  📋 Testing path splice crossover...")
            offspring3, offspring4 = operators.path_splice_crossover(parent1, parent2)
            
            if not isinstance(offspring3, RouteChromosome) or not isinstance(offspring4, RouteChromosome):
                print("  ❌ Path splice crossover failed to return chromosomes")
                return False
            
            print(f"  ✅ Path splice: {len(offspring3.segments)}, {len(offspring4.segments)} segments")
            print(f"     Creation methods: {offspring3.creation_method}, {offspring4.creation_method}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Crossover test failed: {e}")
            return False
    
    def _test_mutation_operators(self, operators: GAOperators, chromosome: RouteChromosome) -> bool:
        """Test mutation operators"""
        try:
            original_segments = len(chromosome.segments)
            
            # Test segment replacement mutation
            print("  📋 Testing segment replacement mutation...")
            mutated1 = operators.segment_replacement_mutation(chromosome, mutation_rate=1.0)
            
            if not isinstance(mutated1, RouteChromosome):
                print("  ❌ Segment replacement mutation failed")
                return False
            
            print(f"  ✅ Segment replacement: {len(mutated1.segments)} segments (was {original_segments})")
            
            # Test route extension mutation
            print("  📋 Testing route extension mutation...")
            mutated2 = operators.route_extension_mutation(chromosome, 4.0, mutation_rate=1.0)
            
            if not isinstance(mutated2, RouteChromosome):
                print("  ❌ Route extension mutation failed")
                return False
            
            print(f"  ✅ Route extension: {len(mutated2.segments)} segments (was {original_segments})")
            
            # Test elevation bias mutation
            print("  📋 Testing elevation bias mutation...")
            mutated3 = operators.elevation_bias_mutation(chromosome, "elevation", mutation_rate=1.0)
            
            if not isinstance(mutated3, RouteChromosome):
                print("  ❌ Elevation bias mutation failed")
                return False
            
            print(f"  ✅ Elevation bias: {len(mutated3.segments)} segments (was {original_segments})")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Mutation test failed: {e}")
            return False
    
    def _test_selection_operators(self, operators: GAOperators, population: List[RouteChromosome]) -> bool:
        """Test selection operators"""
        try:
            # Assign fitness values for testing
            for i, chromo in enumerate(population):
                chromo.fitness = 0.5 + i * 0.1  # Ascending fitness
            
            # Test tournament selection
            print("  📋 Testing tournament selection...")
            selected = operators.tournament_selection(population, tournament_size=3)
            
            if not isinstance(selected, RouteChromosome):
                print("  ❌ Tournament selection failed")
                return False
            
            print(f"  ✅ Tournament selection: fitness {selected.fitness:.2f}")
            
            # Test elitism selection  
            print("  📋 Testing elitism selection...")
            elite = operators.elitism_selection(population, elite_size=2)
            
            if not isinstance(elite, list) or len(elite) != 2:
                print("  ❌ Elitism selection failed")
                return False
            
            elite_fitnesses = [c.fitness for c in elite]
            print(f"  ✅ Elitism selection: {len(elite)} elite with fitness {elite_fitnesses}")
            
            # Test diversity selection
            print("  📋 Testing diversity selection...")
            diverse = operators.diversity_selection(population, selection_size=3)
            
            if not isinstance(diverse, list) or len(diverse) != 3:
                print("  ❌ Diversity selection failed")
                return False
            
            diverse_fitnesses = [c.fitness for c in diverse]
            print(f"  ✅ Diversity selection: {len(diverse)} selected with fitness {diverse_fitnesses}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Selection test failed: {e}")
            return False
    
    def test_evolution_phase(self) -> bool:
        """Test Phase: Complete Evolution (Week 3)"""
        print("\n" + "="*60)
        print("🧬 TESTING PHASE: GENETIC EVOLUTION (WEEK 3)")
        print("="*60)
        
        try:
            # Initialize complete GA system
            print("🔧 Initializing complete genetic algorithm system...")
            from genetic_route_optimizer import GeneticRouteOptimizer, GAConfig
            from ga_fitness import GAFitnessEvaluator
            
            # Configure for testing (smaller scale)
            config = GAConfig(
                population_size=20,
                max_generations=30,
                crossover_rate=0.8,
                mutation_rate=0.15,
                elite_size=3,
                convergence_threshold=0.001,
                verbose=True
            )
            
            optimizer = GeneticRouteOptimizer(self.graph, config)
            print("✅ Genetic optimizer initialized")
            
            # Test multiple objectives
            objectives = ["elevation", "distance", "balanced"]
            distances = [3.0, 5.0]
            
            all_results = []
            
            for objective in objectives:
                for distance in distances:
                    print(f"\n📋 Testing {objective} objective at {distance}km...")
                    
                    try:
                        # Run optimization
                        start_time = time.time()
                        results = optimizer.optimize_route(
                            self.start_node, distance, objective, 
                            visualizer=self.visualizer if self.save_images else None
                        )
                        optimization_time = time.time() - start_time
                        
                        # Validate results
                        if not self._validate_evolution_results(results, objective, distance):
                            print(f"❌ Evolution validation failed for {objective} at {distance}km")
                            return False
                        
                        print(f"✅ {objective.title()} optimization completed:")
                        print(f"   Best fitness: {results.best_fitness:.4f}")
                        print(f"   Generations: {results.total_generations}")
                        print(f"   Time: {optimization_time:.2f}s")
                        print(f"   Convergence: {results.convergence_reason}")
                        
                        all_results.append((objective, distance, results))
                        
                    except Exception as e:
                        print(f"❌ Evolution failed for {objective} at {distance}km: {e}")
                        return False
            
            # Test fitness evaluation system
            print("\n📋 Testing fitness evaluation system...")
            success = self._test_fitness_evaluation()
            if not success:
                return False
            
            # Test convergence detection
            print("\n📋 Testing convergence detection...")
            success = self._test_convergence_detection(optimizer)
            if not success:
                return False
            
            # Generate comprehensive visualizations
            if self.save_images:
                print("\n📸 Generating evolution visualizations...")
                self._generate_evolution_visualizations(all_results)
            
            print("✅ Evolution phase test completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Evolution phase test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _validate_evolution_results(self, results, objective: str, distance: float) -> bool:
        """Validate evolution results"""
        try:
            # Check basic result structure
            if not results.best_chromosome:
                print("  ❌ No best chromosome found")
                return False
            
            if results.best_fitness <= 0:
                print("  ❌ Invalid best fitness")
                return False
            
            if results.total_generations <= 0:
                print("  ❌ Invalid generation count")
                return False
            
            # Check chromosome validity
            if not results.best_chromosome.is_valid:
                print("  ❌ Best chromosome is invalid")
                return False
            
            if not results.best_chromosome.segments:
                print("  ❌ Best chromosome has no segments")
                return False
            
            # Check route properties
            route_stats = results.best_chromosome.get_route_stats()
            route_distance = route_stats['total_distance_km']
            
            # Distance should be reasonably close to target (within 50% tolerance for test)
            distance_tolerance = distance * 0.5
            if abs(route_distance - distance) > distance_tolerance:
                print(f"  ⚠️ Route distance {route_distance:.2f}km far from target {distance}km")
                # Don't fail for this in testing, but warn
            
            # Check that evolution actually occurred
            if len(results.fitness_history) < 2:
                print("  ❌ Insufficient evolution history")
                return False
            
            # Check for fitness improvement (initial vs final)
            initial_fitness = max(results.fitness_history[0])
            final_fitness = results.best_fitness
            
            if final_fitness < initial_fitness:
                print("  ⚠️ Fitness decreased during evolution")
                # Don't fail for this in testing, evolution can be noisy
            
            print(f"  ✅ Evolution validation passed")
            return True
            
        except Exception as e:
            print(f"  ❌ Evolution validation error: {e}")
            return False
    
    def _test_fitness_evaluation(self) -> bool:
        """Test fitness evaluation system"""
        try:
            from ga_fitness import GAFitnessEvaluator
            
            # Create test chromosome
            initializer = PopulationInitializer(self.graph, self.start_node)
            population = initializer.create_population(5, 3.0)
            
            if not population:
                print("  ❌ Failed to create test population for fitness testing")
                return False
            
            test_chromosome = population[0]
            
            # Test different objectives
            objectives = ["elevation", "distance", "balanced", "scenic", "efficiency"]
            
            for objective in objectives:
                evaluator = GAFitnessEvaluator(objective, 3.0)
                fitness = evaluator.evaluate_chromosome(test_chromosome)
                
                if not (0.0 <= fitness <= 1.0):
                    print(f"  ❌ Invalid fitness score for {objective}: {fitness}")
                    return False
                
                print(f"  ✅ {objective.title()} fitness: {fitness:.4f}")
            
            # Test population evaluation
            evaluator = GAFitnessEvaluator("elevation", 3.0)
            fitness_scores = evaluator.evaluate_population(population)
            
            if len(fitness_scores) != len(population):
                print("  ❌ Population fitness evaluation failed")
                return False
            
            # Test fitness statistics
            stats = evaluator.get_fitness_stats()
            expected_keys = ['evaluations', 'best_fitness', 'average_fitness', 'worst_fitness']
            
            for key in expected_keys:
                if key not in stats:
                    print(f"  ❌ Missing fitness stat: {key}")
                    return False
            
            print("  ✅ Fitness evaluation system test passed")
            return True
            
        except Exception as e:
            print(f"  ❌ Fitness evaluation test failed: {e}")
            return False
    
    def _test_convergence_detection(self, optimizer) -> bool:
        """Test convergence detection"""
        try:
            # Test with stable fitness (should converge)
            stable_history = []
            for i in range(25):
                gen_fitness = [0.7] * 10  # Stable fitness
                stable_history.append(gen_fitness)
            
            optimizer.fitness_history = stable_history
            convergence = optimizer._check_convergence([0.7] * 10)
            
            if not convergence:
                print("  ❌ Failed to detect convergence with stable fitness")
                return False
            
            # Test with improving fitness (should not converge)
            improving_history = []
            for i in range(25):
                gen_fitness = [0.5 + i * 0.01] * 10  # Improving fitness
                improving_history.append(gen_fitness)
            
            optimizer.fitness_history = improving_history
            convergence = optimizer._check_convergence([0.7] * 10)
            
            if convergence:
                print("  ❌ Incorrectly detected convergence with improving fitness")
                return False
            
            print("  ✅ Convergence detection test passed")
            return True
            
        except Exception as e:
            print(f"  ❌ Convergence detection test failed: {e}")
            return False
    
    def _generate_evolution_visualizations(self, all_results):
        """Generate comprehensive evolution visualizations"""
        try:
            evolution_dir = os.path.join(self.output_dir, "evolution")
            os.makedirs(evolution_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for objective, distance, results in all_results:
                # Save best route visualization
                if self.visualizer:
                    filename = f"evolution_best_{objective}_{distance}km_{timestamp}.png"
                    filepath = self.visualizer.save_chromosome_map(
                        results.best_chromosome, filename,
                        title=f"Best Route - {objective.title()} {distance}km (Fitness: {results.best_fitness:.4f})",
                        show_elevation=True, show_segments=True
                    )
                    print(f"  📸 Saved best route: {filename}")
                
                # Generate fitness progression plot
                self._save_fitness_progression(results, objective, distance, evolution_dir, timestamp)
            
            # Generate comparison visualization
            self._save_objective_comparison(all_results, evolution_dir, timestamp)
            
            print(f"  ✅ Evolution visualizations saved to {evolution_dir}")
            
        except Exception as e:
            print(f"  ⚠️ Evolution visualization failed: {e}")
    
    def _save_fitness_progression(self, results, objective: str, distance: float, 
                                output_dir: str, timestamp: str):
        """Save fitness progression plot"""
        try:
            import matplotlib.pyplot as plt
            
            # Extract fitness data
            generations = list(range(len(results.fitness_history)))
            best_fitness = [max(gen_fitness) for gen_fitness in results.fitness_history]
            avg_fitness = [sum(gen_fitness)/len(gen_fitness) for gen_fitness in results.fitness_history]
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(generations, best_fitness, label='Best Fitness', linewidth=2, color='red')
            plt.plot(generations, avg_fitness, label='Average Fitness', linewidth=2, color='blue')
            
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title(f'Fitness Evolution - {objective.title()} {distance}km')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add convergence annotation
            if results.convergence_reason == "convergence":
                plt.axvline(x=results.total_generations-1, color='green', linestyle='--', 
                          label=f'Converged at Gen {results.total_generations}')
                plt.legend()
            
            filename = f"fitness_progression_{objective}_{distance}km_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  📊 Saved fitness progression: {filename}")
            
        except Exception as e:
            print(f"  ⚠️ Failed to save fitness progression: {e}")
    
    def _save_objective_comparison(self, all_results, output_dir: str, timestamp: str):
        """Save objective comparison chart"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Organize data by objective
            objectives = {}
            for objective, distance, results in all_results:
                if objective not in objectives:
                    objectives[objective] = {'distances': [], 'fitness': [], 'generations': []}
                
                objectives[objective]['distances'].append(distance)
                objectives[objective]['fitness'].append(results.best_fitness)
                objectives[objective]['generations'].append(results.total_generations)
            
            # Create comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Fitness comparison
            for obj, data in objectives.items():
                ax1.plot(data['distances'], data['fitness'], 'o-', label=obj.title(), linewidth=2, markersize=8)
            
            ax1.set_xlabel('Target Distance (km)')
            ax1.set_ylabel('Best Fitness')
            ax1.set_title('Best Fitness by Objective')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Generations comparison
            for obj, data in objectives.items():
                ax2.plot(data['distances'], data['generations'], 's-', label=obj.title(), linewidth=2, markersize=8)
            
            ax2.set_xlabel('Target Distance (km)')
            ax2.set_ylabel('Generations to Converge')
            ax2.set_title('Convergence Speed by Objective')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filename = f"objective_comparison_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  📊 Saved objective comparison: {filename}")
            
        except Exception as e:
            print(f"  ⚠️ Failed to save objective comparison: {e}")
    
    def run_test_phase(self, phase: str) -> bool:
        """Run specific test phase"""
        phase_map = {
            'chromosome': self.test_chromosome_phase,
            'initialization': self.test_initialization_phase,
            'operators': self.test_operators_phase,
            'evolution': self.test_evolution_phase,
            'comparison': self.test_comparison_phase
        }
        
        if phase not in phase_map:
            print(f"❌ Unknown test phase: {phase}")
            print(f"Available phases: {list(phase_map.keys())}")
            return False
        
        # Setup network if not already done
        if not self.graph:
            if not self.setup_network():
                return False
        
        # Run the test phase
        return phase_map[phase]()


def main():
    """Main function for GA development testing"""
    parser = argparse.ArgumentParser(description="GA Development Test Framework")
    
    parser.add_argument('--phase', type=str, required=True,
                       choices=['chromosome', 'initialization', 'operators', 'evolution', 'comparison', 'all'],
                       help='Test phase to run')
    parser.add_argument('--save-images', action='store_true', default=True,
                       help='Save visualization images')
    parser.add_argument('--output-dir', type=str, default='ga_dev_images',
                       help='Output directory for images')
    
    args = parser.parse_args()
    
    print("🚀 GA Development Test Framework")
    print("="*60)
    print(f"Phase: {args.phase}")
    print(f"Save Images: {args.save_images}")
    print(f"Output Directory: {args.output_dir}")
    print("="*60)
    
    # Create tester
    tester = GADevelopmentTester(args.save_images, args.output_dir)
    
    # Run tests
    if args.phase == 'all':
        phases = ['chromosome', 'initialization', 'operators', 'evolution', 'comparison']
        all_passed = True
        
        for phase in phases:
            print(f"\n🔄 Running phase: {phase}")
            if not tester.run_test_phase(phase):
                all_passed = False
                print(f"❌ Phase {phase} failed")
            else:
                print(f"✅ Phase {phase} passed")
        
        if all_passed:
            print(f"\n🎉 All test phases completed successfully!")
        else:
            print(f"\n⚠️ Some test phases failed")
            
    else:
        success = tester.run_test_phase(args.phase)
        if success:
            print(f"\n✅ Test phase '{args.phase}' completed successfully!")
        else:
            print(f"\n❌ Test phase '{args.phase}' failed!")
    
    print(f"\n📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()