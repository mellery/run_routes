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
from ga_visualizer import GAVisualizer


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
            if self.save_images:
                self.visualizer = GAVisualizer(self.graph, self.output_dir)
                print("✅ Visualizer initialized")
            
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
    
    def run_test_phase(self, phase: str) -> bool:
        """Run specific test phase"""
        phase_map = {
            'chromosome': self.test_chromosome_phase,
            'initialization': self.test_initialization_phase,
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
                       choices=['chromosome', 'initialization', 'comparison', 'all'],
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
        phases = ['chromosome', 'initialization', 'comparison']
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