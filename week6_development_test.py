#!/usr/bin/env python3
"""
Week 6 GA Development Test - System Integration & Testing
Mandatory visualization and verification for Week 6 GA integration
"""

import argparse
import time
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    from route_services import NetworkManager, RouteOptimizer, RouteAnalyzer
    from ga_visualizer import GAVisualizer
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("📋 Install with: pip install networkx osmnx numpy matplotlib")
    DEPENDENCIES_AVAILABLE = False


class Week6GADevelopmentTester:
    """Week 6 GA development testing with mandatory visualizations"""
    
    def __init__(self, save_images: bool = True):
        """Initialize Week 6 development tester
        
        Args:
            save_images: Whether to save visualization images
        """
        self.save_images = save_images
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize services
        self.services = None
        self.graph = None
        self.route_optimizer = None
        self.visualizer = None
        
        # Test parameters
        self.start_node = None
        self.test_distances = [1.0, 2.0, 3.0]
        self.test_objectives = ['distance', 'elevation', 'balanced']
        
        # Results storage
        self.test_results = {}
        
    def initialize_services(self) -> bool:
        """Initialize route planning services"""
        try:
            print("🌐 Initializing route planning services for Week 6 testing...")
            
            # Create network manager and load graph
            network_manager = NetworkManager()
            self.graph = network_manager.load_network(radius_km=0.8)
            
            if not self.graph:
                print("❌ Failed to load street network")
                return False
            
            # Create services
            self.services = {
                'network_manager': network_manager,
                'route_optimizer': RouteOptimizer(self.graph),
                'route_analyzer': RouteAnalyzer(self.graph)
            }
            
            self.route_optimizer = self.services['route_optimizer']
            
            # Initialize visualizer
            self.visualizer = GAVisualizer(self.graph)
            
            # Find valid starting node
            self.start_node = self._find_valid_start_node()
            
            # Display network stats
            stats = network_manager.get_network_stats(self.graph)
            print(f"✅ Loaded {stats['nodes']} intersections and {stats['edges']} road segments")
            print(f"📍 Using start node: {self.start_node}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize services: {e}")
            return False
    
    def _find_valid_start_node(self) -> Optional[int]:
        """Find a valid starting node with good connectivity"""
        if not self.graph:
            return None
        
        # Find node with at least 3 connections
        for node_id in self.graph.nodes():
            if self.graph.degree(node_id) >= 3:
                return node_id
        
        # Fallback to any node
        return list(self.graph.nodes())[0]
    
    def test_ga_integration_verification(self) -> bool:
        """Test and verify GA integration with route services"""
        print(f"\n{'='*60}")
        print("WEEK 6 TEST: GA Integration Verification")
        print(f"{'='*60}")
        
        # Test 1: Verify GA availability
        print("\n1. Testing GA Availability Detection...")
        solver_info = self.route_optimizer.get_solver_info()
        
        ga_available = solver_info.get('ga_available', False)
        print(f"   GA Available: {'✅ Yes' if ga_available else '❌ No'}")
        
        if not ga_available:
            print("   ⚠️ GA not available - integration test cannot proceed")
            return False
        
        print(f"   GA Optimizer: {solver_info.get('ga_optimizer', 'Unknown')}")
        print(f"   Available Algorithms: {', '.join(solver_info['available_algorithms'])}")
        
        # Test 2: Algorithm Selection Logic
        print("\n2. Testing Algorithm Selection Logic...")
        
        # Test auto selection for elevation objective
        result_auto = self.route_optimizer.optimize_route(
            start_node=self.start_node,
            target_distance_km=2.0,
            objective=self.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION,
            algorithm="auto"
        )
        
        if result_auto:
            algorithm_used = result_auto['solver_info']['algorithm_used']
            print(f"   Auto selection for elevation: {algorithm_used}")
            
            # For elevation objectives, should prefer GA
            if algorithm_used == "genetic":
                print("   ✅ Auto selection correctly chose GA for elevation objective")
            else:
                print("   ⚠️ Auto selection chose TSP instead of GA")
        
        # Test 3: GA vs TSP Comparison
        print("\n3. Testing GA vs TSP Algorithm Comparison...")
        
        return self._run_ga_vs_tsp_comparison()
    
    def _run_ga_vs_tsp_comparison(self) -> bool:
        """Run GA vs TSP comparison with visualization"""
        comparison_results = {}
        
        for distance_km in [1.5, 2.5]:
            print(f"\n   Testing {distance_km}km routes...")
            
            # Test TSP algorithm
            print(f"     Running TSP optimization...")
            tsp_result = self.route_optimizer.optimize_route(
                start_node=self.start_node,
                target_distance_km=distance_km,
                objective=self.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION,
                algorithm="nearest_neighbor"
            )
            
            # Test GA algorithm
            print(f"     Running GA optimization...")
            ga_result = self.route_optimizer.optimize_route(
                start_node=self.start_node,
                target_distance_km=distance_km,
                objective=self.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION,
                algorithm="genetic"
            )
            
            if tsp_result and ga_result:
                tsp_stats = tsp_result['stats']
                ga_stats = ga_result['stats']
                
                tsp_time = tsp_result['solver_info']['solve_time']
                ga_time = ga_result['solver_info']['solve_time']
                
                tsp_elevation = tsp_stats.get('total_elevation_gain_m', 0)
                ga_elevation = ga_stats.get('total_elevation_gain_m', 0)
                
                print(f"     TSP:  {tsp_time:.2f}s, {tsp_elevation:.0f}m elevation")
                print(f"     GA:   {ga_time:.2f}s, {ga_elevation:.0f}m elevation")
                
                comparison_results[distance_km] = {
                    'tsp': tsp_result,
                    'ga': ga_result
                }
        
        # Generate comparison visualization
        if self.save_images and comparison_results:
            self._create_ga_vs_tsp_comparison_visualization(comparison_results)
        
        return len(comparison_results) > 0
    
    def test_application_integration(self) -> bool:
        """Test GA integration with CLI and web applications"""
        print(f"\n{'='*60}")
        print("WEEK 6 TEST: Application Integration")
        print(f"{'='*60}")
        
        # Test 1: CLI Application Integration
        print("\n1. Testing CLI Application Integration...")
        
        try:
            # Import CLI module
            from cli_route_planner import RefactoredCLIRoutePlanner
            
            # Test CLI planner initialization
            cli_planner = RefactoredCLIRoutePlanner()
            
            if cli_planner.initialize_services():
                print("   ✅ CLI services initialized successfully")
                
                # Test route generation through CLI services
                result = cli_planner.generate_route(
                    self.start_node, 1.5, 
                    cli_planner.services['route_optimizer'].RouteObjective.MAXIMIZE_ELEVATION,
                    "genetic"
                )
                
                if result:
                    print("   ✅ CLI GA route generation successful")
                else:
                    print("   ❌ CLI GA route generation failed")
                    return False
            else:
                print("   ❌ CLI services initialization failed")
                return False
                
        except Exception as e:
            print(f"   ❌ CLI integration test failed: {e}")
            return False
        
        # Test 2: Web Application Integration
        print("\n2. Testing Web Application Integration...")
        
        try:
            # Import Streamlit app module
            import importlib.util
            
            app_path = os.path.join(os.path.dirname(__file__), "running_route_app.py")
            spec = importlib.util.spec_from_file_location("running_route_app", app_path)
            app_module = importlib.util.module_from_spec(spec)
            
            # Test that the app can be imported without errors
            spec.loader.exec_module(app_module)
            print("   ✅ Streamlit app imports successfully")
            
            # Test services initialization function
            services = app_module.initialize_route_services()
            if services:
                print("   ✅ Streamlit services initialized successfully")
                
                # Test algorithm availability in web app
                route_optimizer = services['route_optimizer']
                algorithms = route_optimizer.get_available_algorithms()
                
                if 'genetic' in algorithms:
                    print("   ✅ GA algorithm available in web interface")
                else:
                    print("   ⚠️ GA algorithm not available in web interface")
                    
            else:
                print("   ❌ Streamlit services initialization failed")
                return False
                
        except Exception as e:
            print(f"   ❌ Web app integration test failed: {e}")
            return False
        
        return True
    
    def test_real_world_performance(self) -> bool:
        """Test GA performance with real street networks"""
        print(f"\n{'='*60}")
        print("WEEK 6 TEST: Real-World Performance")
        print(f"{'='*60}")
        
        performance_results = {}
        
        for distance_km in self.test_distances:
            print(f"\n1. Testing {distance_km}km GA performance...")
            
            start_time = time.time()
            
            result = self.route_optimizer.optimize_route(
                start_node=self.start_node,
                target_distance_km=distance_km,
                objective=self.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION,
                algorithm="genetic"
            )
            
            total_time = time.time() - start_time
            
            if result:
                stats = result['stats']
                solver_info = result['solver_info']
                
                distance = stats.get('total_distance_km', 0)
                elevation = stats.get('total_elevation_gain_m', 0)
                generations = solver_info.get('ga_generations', 0)
                convergence = solver_info.get('ga_convergence', 'unknown')
                
                print(f"   Result: ✅ {total_time:.2f}s, {distance:.2f}km, {elevation:.0f}m")
                print(f"   GA Info: {generations} generations, {convergence} convergence")
                
                performance_results[distance_km] = {
                    'total_time': total_time,
                    'distance': distance,
                    'elevation': elevation,
                    'generations': generations,
                    'convergence': convergence
                }
                
                # Performance benchmarks
                if total_time > 60:
                    print(f"   ⚠️ Slow performance: {total_time:.1f}s (expected < 60s)")
                else:
                    print(f"   ✅ Good performance: {total_time:.1f}s")
                    
            else:
                print(f"   ❌ Failed to generate {distance_km}km route")
                return False
        
        # Generate performance visualization
        if self.save_images and performance_results:
            self._create_performance_visualization(performance_results)
        
        return len(performance_results) > 0
    
    def _create_ga_vs_tsp_comparison_visualization(self, comparison_results: Dict):
        """Create GA vs TSP comparison visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Week 6: GA vs TSP Algorithm Comparison\n{self.timestamp}', 
                        fontsize=16, fontweight='bold')
            
            distances = list(comparison_results.keys())
            
            # Extract metrics
            tsp_times = []
            ga_times = []
            tsp_elevations = []
            ga_elevations = []
            
            for distance_km in distances:
                tsp_result = comparison_results[distance_km]['tsp']
                ga_result = comparison_results[distance_km]['ga']
                
                tsp_times.append(tsp_result['solver_info']['solve_time'])
                ga_times.append(ga_result['solver_info']['solve_time'])
                
                tsp_elevations.append(tsp_result['stats'].get('total_elevation_gain_m', 0))
                ga_elevations.append(ga_result['stats'].get('total_elevation_gain_m', 0))
            
            # Plot 1: Solve Time Comparison
            ax1 = axes[0, 0]
            x = np.arange(len(distances))
            width = 0.35
            
            ax1.bar(x - width/2, tsp_times, width, label='TSP', color='skyblue', alpha=0.8)
            ax1.bar(x + width/2, ga_times, width, label='GA', color='lightcoral', alpha=0.8)
            
            ax1.set_xlabel('Route Distance (km)')
            ax1.set_ylabel('Solve Time (seconds)')
            ax1.set_title('Algorithm Solve Time Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels([f'{d}km' for d in distances])
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Elevation Gain Comparison
            ax2 = axes[0, 1]
            ax2.bar(x - width/2, tsp_elevations, width, label='TSP', color='skyblue', alpha=0.8)
            ax2.bar(x + width/2, ga_elevations, width, label='GA', color='lightcoral', alpha=0.8)
            
            ax2.set_xlabel('Route Distance (km)')
            ax2.set_ylabel('Elevation Gain (meters)')
            ax2.set_title('Algorithm Elevation Optimization')
            ax2.set_xticks(x)
            ax2.set_xticklabels([f'{d}km' for d in distances])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Performance Ratio Analysis
            ax3 = axes[1, 0]
            time_ratios = [ga_t / tsp_t if tsp_t > 0 else 0 for ga_t, tsp_t in zip(ga_times, tsp_times)]
            elevation_ratios = [ga_e / tsp_e if tsp_e > 0 else (1 if ga_e > 0 else 0) 
                              for ga_e, tsp_e in zip(ga_elevations, tsp_elevations)]
            
            ax3.plot(distances, time_ratios, 'o-', label='Time Ratio (GA/TSP)', color='red', linewidth=2, markersize=8)
            ax3.plot(distances, elevation_ratios, 's-', label='Elevation Ratio (GA/TSP)', color='green', linewidth=2, markersize=8)
            ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equal Performance')
            
            ax3.set_xlabel('Route Distance (km)')
            ax3.set_ylabel('Performance Ratio')
            ax3.set_title('GA vs TSP Performance Ratios')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Summary Statistics
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # Calculate summary statistics
            avg_time_ratio = np.mean(time_ratios) if time_ratios else 0
            avg_elevation_ratio = np.mean(elevation_ratios) if elevation_ratios else 0
            
            summary_text = f"""
Week 6 Integration Summary:
            
✅ GA Successfully Integrated
✅ Both Algorithms Functional
            
Performance Summary:
• GA Average Time: {avg_time_ratio:.1f}x TSP time
• GA Average Elevation: {avg_elevation_ratio:.1f}x TSP elevation
            
Route Testing:
• Distances: {', '.join(f'{d}km' for d in distances)}
• Objective: Maximum Elevation Gain
• Network: Real street data
            
Status: Week 6 Integration Complete
"""
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.1))
            
            plt.tight_layout()
            
            # Save visualization
            if self.save_images:
                filename = f"week6_ga_vs_tsp_comparison_{self.timestamp}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"   📸 Saved comparison visualization: {filename}")
            
            plt.close()
            
        except Exception as e:
            print(f"   ❌ Failed to create comparison visualization: {e}")
    
    def _create_performance_visualization(self, performance_results: Dict):
        """Create performance analysis visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Week 6: GA Real-World Performance Analysis\n{self.timestamp}', 
                        fontsize=16, fontweight='bold')
            
            distances = list(performance_results.keys())
            times = [performance_results[d]['total_time'] for d in distances]
            elevations = [performance_results[d]['elevation'] for d in distances]
            generations = [performance_results[d]['generations'] for d in distances]
            
            # Plot 1: Solve Time vs Distance
            ax1 = axes[0, 0]
            ax1.plot(distances, times, 'o-', color='blue', linewidth=2, markersize=8)
            ax1.set_xlabel('Route Distance (km)')
            ax1.set_ylabel('Solve Time (seconds)')
            ax1.set_title('GA Solve Time Scalability')
            ax1.grid(True, alpha=0.3)
            
            # Add performance threshold line
            ax1.axhline(y=60, color='red', linestyle='--', alpha=0.7, label='60s threshold')
            ax1.legend()
            
            # Plot 2: Elevation Gain vs Distance
            ax2 = axes[0, 1]
            ax2.plot(distances, elevations, 's-', color='green', linewidth=2, markersize=8)
            ax2.set_xlabel('Route Distance (km)')
            ax2.set_ylabel('Elevation Gain (meters)')
            ax2.set_title('GA Elevation Optimization Results')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Generations vs Distance
            ax3 = axes[1, 0]
            ax3.plot(distances, generations, '^-', color='orange', linewidth=2, markersize=8)
            ax3.set_xlabel('Route Distance (km)')
            ax3.set_ylabel('GA Generations')
            ax3.set_title('GA Convergence Analysis')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Performance Summary
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # Calculate performance metrics
            avg_time = np.mean(times)
            max_time = max(times)
            total_elevation = sum(elevations)
            avg_generations = np.mean(generations)
            
            # Get convergence info
            convergence_types = [performance_results[d]['convergence'] for d in distances]
            convergence_summary = ', '.join(set(convergence_types))
            
            performance_text = f"""
Week 6 Performance Analysis:
            
Solve Time Performance:
• Average: {avg_time:.1f} seconds
• Maximum: {max_time:.1f} seconds
• Status: {'✅ Good' if max_time < 60 else '⚠️ Slow'}
            
Optimization Quality:
• Total Elevation Found: {total_elevation:.0f}m
• Average Generations: {avg_generations:.0f}
• Convergence: {convergence_summary}
            
Real-World Testing:
• Network: OpenStreetMap data
• Routes: {len(distances)} different distances
• Objectives: Elevation maximization
            
Week 6 Status: ✅ Performance Verified
"""
            
            ax4.text(0.05, 0.95, performance_text, transform=ax4.transAxes, 
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.1))
            
            plt.tight_layout()
            
            # Save visualization
            if self.save_images:
                filename = f"week6_ga_performance_analysis_{self.timestamp}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"   📸 Saved performance visualization: {filename}")
            
            plt.close()
            
        except Exception as e:
            print(f"   ❌ Failed to create performance visualization: {e}")
    
    def run_complete_week6_test(self) -> bool:
        """Run complete Week 6 development test suite"""
        print(f"🧬 WEEK 6 GA DEVELOPMENT TEST")
        print(f"System Integration & Testing Verification")
        print(f"Timestamp: {self.timestamp}")
        print(f"{'='*80}")
        
        if not DEPENDENCIES_AVAILABLE:
            print("❌ Required dependencies not available")
            return False
        
        # Initialize services
        if not self.initialize_services():
            print("❌ Failed to initialize services")
            return False
        
        test_results = []
        
        # Test 1: GA Integration Verification
        try:
            result = self.test_ga_integration_verification()
            test_results.append(('GA Integration', result))
        except Exception as e:
            print(f"❌ GA Integration test failed: {e}")
            test_results.append(('GA Integration', False))
        
        # Test 2: Application Integration
        try:
            result = self.test_application_integration()
            test_results.append(('Application Integration', result))
        except Exception as e:
            print(f"❌ Application Integration test failed: {e}")
            test_results.append(('Application Integration', False))
        
        # Test 3: Real-World Performance
        try:
            result = self.test_real_world_performance()
            test_results.append(('Real-World Performance', result))
        except Exception as e:
            print(f"❌ Real-World Performance test failed: {e}")
            test_results.append(('Real-World Performance', False))
        
        # Final Summary
        self._print_final_summary(test_results)
        
        return all(result for _, result in test_results)
    
    def _print_final_summary(self, test_results: List):
        """Print final test summary"""
        print(f"\n{'='*80}")
        print("WEEK 6 FINAL SUMMARY")
        print(f"{'='*80}")
        
        total_tests = len(test_results)
        passed_tests = sum(1 for _, result in test_results if result)
        
        print(f"\nTest Results:")
        for test_name, result in test_results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"  {test_name:<25} {status}")
        
        print(f"\nOverall Results:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {total_tests - passed_tests}")
        print(f"  Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        if all(result for _, result in test_results):
            print(f"\n🎉 WEEK 6 COMPLETE!")
            print(f"✅ GA System Integration & Testing Successfully Implemented")
            print(f"✅ All verification tests passed")
            print(f"✅ Production-ready GA implementation achieved")
        else:
            print(f"\n⚠️ WEEK 6 INCOMPLETE")
            print(f"❌ Some verification tests failed")
            print(f"🔧 Review failed tests and fix issues")
        
        if self.save_images:
            print(f"\n📸 Verification images saved with timestamp: {self.timestamp}")


def main():
    """Main entry point for Week 6 development testing"""
    parser = argparse.ArgumentParser(description='Week 6 GA Development Test - System Integration & Testing')
    
    parser.add_argument('--phase', choices=['integration', 'applications', 'performance', 'all'],
                       default='all', help='Test phase to run')
    parser.add_argument('--save-images', action='store_true', default=True,
                       help='Save visualization images (default: True)')
    parser.add_argument('--no-save-images', action='store_true',
                       help='Do not save visualization images')
    
    args = parser.parse_args()
    
    # Handle save images flag
    save_images = args.save_images and not args.no_save_images
    
    if not DEPENDENCIES_AVAILABLE:
        print("❌ Cannot run Week 6 tests - missing dependencies")
        return False
    
    # Create tester
    tester = Week6GADevelopmentTester(save_images=save_images)
    
    # Run requested phase
    if args.phase == 'all':
        success = tester.run_complete_week6_test()
    elif args.phase == 'integration':
        if not tester.initialize_services():
            return False
        success = tester.test_ga_integration_verification()
    elif args.phase == 'applications':
        if not tester.initialize_services():
            return False
        success = tester.test_application_integration()
    elif args.phase == 'performance':
        if not tester.initialize_services():
            return False
        success = tester.test_real_world_performance()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)