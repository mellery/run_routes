#!/usr/bin/env python3
"""
GA vs TSP Performance Benchmark
Comprehensive benchmarking of genetic algorithm vs TSP solvers
"""

import unittest
import time
import statistics
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import networkx as nx

# Import route services
from route_services import NetworkManager, RouteOptimizer


class GABenchmarkResult:
    """Container for benchmark results"""
    
    def __init__(self, algorithm: str, objective: str):
        self.algorithm = algorithm
        self.objective = objective
        self.runs = []
        
    def add_run(self, result: Dict[str, Any], solve_time: float):
        """Add a benchmark run result"""
        self.runs.append({
            'result': result,
            'solve_time': solve_time,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get benchmark statistics"""
        if not self.runs:
            return {}
        
        # Extract metrics
        solve_times = [run['solve_time'] for run in self.runs]
        successful_runs = [run for run in self.runs if run['result'] is not None]
        
        stats = {
            'algorithm': self.algorithm,
            'objective': self.objective,
            'total_runs': len(self.runs),
            'successful_runs': len(successful_runs),
            'success_rate': len(successful_runs) / len(self.runs) if self.runs else 0.0,
            'solve_time_stats': {
                'mean': statistics.mean(solve_times) if solve_times else 0.0,
                'median': statistics.median(solve_times) if solve_times else 0.0,
                'min': min(solve_times) if solve_times else 0.0,
                'max': max(solve_times) if solve_times else 0.0,
                'stdev': statistics.stdev(solve_times) if len(solve_times) > 1 else 0.0
            }
        }
        
        # Extract route quality metrics from successful runs
        if successful_runs:
            distances = []
            elevations = []
            fitness_scores = []
            
            for run in successful_runs:
                result = run['result']
                stats_data = result.get('stats', {})
                
                if 'total_distance_km' in stats_data:
                    distances.append(stats_data['total_distance_km'])
                
                if 'total_elevation_gain_m' in stats_data:
                    elevations.append(stats_data['total_elevation_gain_m'])
                
                if 'fitness_score' in result:
                    fitness_scores.append(result['fitness_score'])
            
            if distances:
                stats['distance_stats'] = {
                    'mean': statistics.mean(distances),
                    'median': statistics.median(distances),
                    'min': min(distances),
                    'max': max(distances),
                    'stdev': statistics.stdev(distances) if len(distances) > 1 else 0.0
                }
            
            if elevations:
                stats['elevation_stats'] = {
                    'mean': statistics.mean(elevations),
                    'median': statistics.median(elevations),
                    'min': min(elevations),
                    'max': max(elevations),
                    'stdev': statistics.stdev(elevations) if len(elevations) > 1 else 0.0
                }
            
            if fitness_scores:
                stats['fitness_stats'] = {
                    'mean': statistics.mean(fitness_scores),
                    'median': statistics.median(fitness_scores),
                    'min': min(fitness_scores),
                    'max': max(fitness_scores),
                    'stdev': statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0.0
                }
        
        return stats


class TestGAVsTSPBenchmark(unittest.TestCase):
    """Comprehensive GA vs TSP benchmarking"""
    
    @classmethod
    def setUpClass(cls):
        """Set up benchmark environment"""
        try:
            # Initialize network
            cls.network_manager = NetworkManager()
            cls.graph = cls.network_manager.load_network(radius_km=0.6)
            
            if not cls.graph:
                cls.skipTest("Could not load network for benchmarking")
            
            # Initialize optimizer
            cls.route_optimizer = RouteOptimizer(cls.graph)
            cls.solver_info = cls.route_optimizer.get_solver_info()
            
            # Find valid start node
            cls.start_node = cls._find_valid_start_node()
            
            # Benchmark configuration
            cls.benchmark_config = {
                'distances': [1.0, 2.0, 3.0],
                'objectives': [
                    cls.route_optimizer.RouteObjective.MINIMIZE_DISTANCE,
                    cls.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION,
                    cls.route_optimizer.RouteObjective.BALANCED_ROUTE
                ],
                'algorithms': ['nearest_neighbor'],
                'runs_per_test': 3  # Limited for CI/CD
            }
            
            # Add GA if available
            if cls.solver_info.get('ga_available', False):
                cls.benchmark_config['algorithms'].append('genetic')
            
            # Results storage
            cls.benchmark_results = {}
            
        except Exception as e:
            cls.skipTest(f"Could not initialize benchmark: {e}")
    
    @classmethod
    def _find_valid_start_node(cls) -> Optional[int]:
        """Find a valid starting node with good connectivity"""
        if not cls.graph:
            return None
        
        # Find node with at least 3 connections
        for node_id in cls.graph.nodes():
            if cls.graph.degree(node_id) >= 3:
                return node_id
        
        # Fallback to any node
        return list(cls.graph.nodes())[0]
    
    def test_benchmark_distance_objective(self):
        """Benchmark algorithms for distance objective"""
        objective = self.route_optimizer.RouteObjective.MINIMIZE_DISTANCE
        self._run_benchmark_suite("distance_objective", objective)
    
    def test_benchmark_elevation_objective(self):
        """Benchmark algorithms for elevation objective"""
        objective = self.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION
        self._run_benchmark_suite("elevation_objective", objective)
    
    def test_benchmark_balanced_objective(self):
        """Benchmark algorithms for balanced objective"""
        objective = self.route_optimizer.RouteObjective.BALANCED_ROUTE
        self._run_benchmark_suite("balanced_objective", objective)
    
    def _run_benchmark_suite(self, suite_name: str, objective):
        """Run complete benchmark suite for an objective"""
        print(f"\n{'='*60}")
        print(f"BENCHMARK SUITE: {suite_name}")
        print(f"Objective: {objective}")
        print(f"{'='*60}")
        
        suite_results = {}
        
        for distance_km in self.benchmark_config['distances']:
            for algorithm in self.benchmark_config['algorithms']:
                
                # Skip GA tests if not available
                if algorithm == 'genetic' and not self.solver_info.get('ga_available', False):
                    continue
                
                test_key = f"{algorithm}_{distance_km}km"
                print(f"\nRunning: {test_key}")
                
                benchmark_result = GABenchmarkResult(algorithm, str(objective))
                
                for run_num in range(self.benchmark_config['runs_per_test']):
                    print(f"  Run {run_num + 1}/{self.benchmark_config['runs_per_test']}...", end='')
                    
                    start_time = time.time()
                    
                    result = self.route_optimizer.optimize_route(
                        start_node=self.start_node,
                        target_distance_km=distance_km,
                        objective=objective,
                        algorithm=algorithm
                    )
                    
                    solve_time = time.time() - start_time
                    benchmark_result.add_run(result, solve_time)
                    
                    if result:
                        stats = result.get('stats', {})
                        distance = stats.get('total_distance_km', 0)
                        elevation = stats.get('total_elevation_gain_m', 0)
                        print(f" ✅ {solve_time:.2f}s, {distance:.2f}km, {elevation:.0f}m")
                    else:
                        print(f" ❌ Failed in {solve_time:.2f}s")
                
                # Store results
                suite_results[test_key] = benchmark_result
        
        # Store and analyze results
        self.benchmark_results[suite_name] = suite_results
        self._analyze_benchmark_results(suite_name, suite_results)
    
    def _analyze_benchmark_results(self, suite_name: str, results: Dict[str, GABenchmarkResult]):
        """Analyze and print benchmark results"""
        print(f"\n{'='*60}")
        print(f"BENCHMARK ANALYSIS: {suite_name}")
        print(f"{'='*60}")
        
        # Group by distance
        distances = self.benchmark_config['distances']
        algorithms = [alg for alg in self.benchmark_config['algorithms'] 
                     if alg == 'nearest_neighbor' or self.solver_info.get('ga_available', False)]
        
        for distance_km in distances:
            print(f"\n--- {distance_km}km Routes ---")
            print(f"{'Algorithm':<15} {'Success%':<8} {'Avg Time':<10} {'Avg Dist':<10} {'Avg Elev':<10}")
            print("-" * 65)
            
            for algorithm in algorithms:
                test_key = f"{algorithm}_{distance_km}km"
                if test_key in results:
                    stats = results[test_key].get_statistics()
                    
                    success_rate = stats.get('success_rate', 0) * 100
                    avg_time = stats.get('solve_time_stats', {}).get('mean', 0)
                    avg_distance = stats.get('distance_stats', {}).get('mean', 0)
                    avg_elevation = stats.get('elevation_stats', {}).get('mean', 0)
                    
                    print(f"{algorithm:<15} {success_rate:>6.1f}% {avg_time:>8.2f}s "
                          f"{avg_distance:>8.2f}km {avg_elevation:>8.0f}m")
        
        # Compare algorithms
        if len(algorithms) > 1:
            self._compare_algorithms(results, distances)
    
    def _compare_algorithms(self, results: Dict[str, GABenchmarkResult], distances: List[float]):
        """Compare algorithm performance"""
        print(f"\n--- Algorithm Comparison ---")
        
        algorithms = [alg for alg in self.benchmark_config['algorithms'] 
                     if alg == 'nearest_neighbor' or self.solver_info.get('ga_available', False)]
        
        if 'genetic' in algorithms and 'nearest_neighbor' in algorithms:
            for distance_km in distances:
                ga_key = f"genetic_{distance_km}km"
                tsp_key = f"nearest_neighbor_{distance_km}km"
                
                if ga_key in results and tsp_key in results:
                    ga_stats = results[ga_key].get_statistics()
                    tsp_stats = results[tsp_key].get_statistics()
                    
                    print(f"\n{distance_km}km Route Comparison:")
                    
                    # Time comparison
                    ga_time = ga_stats.get('solve_time_stats', {}).get('mean', 0)
                    tsp_time = tsp_stats.get('solve_time_stats', {}).get('mean', 0)
                    
                    if tsp_time > 0:
                        time_ratio = ga_time / tsp_time
                        print(f"  Time: GA is {time_ratio:.1f}x {'slower' if time_ratio > 1 else 'faster'} than TSP")
                    
                    # Quality comparison (elevation for elevation-focused objectives)
                    ga_elevation = ga_stats.get('elevation_stats', {}).get('mean', 0)
                    tsp_elevation = tsp_stats.get('elevation_stats', {}).get('mean', 0)
                    
                    if tsp_elevation > 0:
                        elevation_ratio = ga_elevation / tsp_elevation
                        print(f"  Elevation: GA finds {elevation_ratio:.1f}x "
                              f"{'more' if elevation_ratio > 1 else 'less'} elevation gain")
                    elif ga_elevation > 0:
                        print(f"  Elevation: GA finds {ga_elevation:.0f}m, TSP finds {tsp_elevation:.0f}m")
    
    def test_benchmark_scalability(self):
        """Test algorithm scalability with different route distances"""
        if not self.solver_info.get('ga_available', False):
            self.skipTest("GA not available for scalability test")
        
        print(f"\n{'='*60}")
        print("SCALABILITY BENCHMARK")
        print(f"{'='*60}")
        
        # Test increasingly larger routes
        test_distances = [0.5, 1.0, 2.0, 3.0]
        objective = self.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION
        
        scalability_results = {}
        
        for algorithm in ['nearest_neighbor', 'genetic']:
            algorithm_results = []
            
            for distance_km in test_distances:
                print(f"\nTesting {algorithm} with {distance_km}km route...")
                
                start_time = time.time()
                result = self.route_optimizer.optimize_route(
                    start_node=self.start_node,
                    target_distance_km=distance_km,
                    objective=objective,
                    algorithm=algorithm
                )
                solve_time = time.time() - start_time
                
                algorithm_results.append({
                    'distance_km': distance_km,
                    'solve_time': solve_time,
                    'success': result is not None
                })
                
                print(f"  Result: {'✅' if result else '❌'} in {solve_time:.2f}s")
            
            scalability_results[algorithm] = algorithm_results
        
        # Analyze scalability
        print(f"\n--- Scalability Analysis ---")
        print(f"{'Distance':<10} {'TSP Time':<10} {'GA Time':<10} {'Ratio':<8}")
        print("-" * 40)
        
        for i, distance_km in enumerate(test_distances):
            tsp_time = scalability_results['nearest_neighbor'][i]['solve_time']
            ga_time = scalability_results['genetic'][i]['solve_time']
            ratio = ga_time / tsp_time if tsp_time > 0 else float('inf')
            
            print(f"{distance_km:<10.1f} {tsp_time:<10.2f} {ga_time:<10.2f} {ratio:<8.1f}")
    
    def test_save_benchmark_results(self):
        """Save benchmark results to file"""
        if not hasattr(self, 'benchmark_results') or not self.benchmark_results:
            self.skipTest("No benchmark results to save")
        
        # Create results directory
        results_dir = "benchmark_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Prepare results for JSON serialization
        json_results = {}
        
        for suite_name, suite_results in self.benchmark_results.items():
            json_results[suite_name] = {}
            
            for test_key, benchmark_result in suite_results.items():
                json_results[suite_name][test_key] = benchmark_result.get_statistics()
        
        # Add metadata
        json_results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'graph_stats': {
                'nodes': len(self.graph.nodes),
                'edges': len(self.graph.edges)
            },
            'solver_info': self.solver_info,
            'benchmark_config': self.benchmark_config
        }
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{results_dir}/ga_vs_tsp_benchmark_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n✅ Benchmark results saved to: {filename}")
        
        # Verify file was created
        self.assertTrue(os.path.exists(filename))


class TestGAPerformanceProfiler(unittest.TestCase):
    """Profile GA performance characteristics"""
    
    def setUp(self):
        """Set up profiler test"""
        try:
            self.network_manager = NetworkManager()
            self.graph = self.network_manager.load_network(radius_km=0.4)
            
            if not self.graph:
                self.skipTest("Could not load network for profiling")
            
            self.route_optimizer = RouteOptimizer(self.graph)
            solver_info = self.route_optimizer.get_solver_info()
            
            if not solver_info.get('ga_available', False):
                self.skipTest("GA not available for profiling")
            
            self.start_node = list(self.graph.nodes())[0]
            
        except Exception as e:
            self.skipTest(f"Could not initialize profiler: {e}")
    
    def test_ga_convergence_profiling(self):
        """Profile GA convergence behavior"""
        result = self.route_optimizer.optimize_route(
            start_node=self.start_node,
            target_distance_km=1.5,
            objective=self.route_optimizer.RouteOptimizer.MAXIMIZE_ELEVATION,
            algorithm="genetic"
        )
        
        if result:
            solver_info = result.get('solver_info', {})
            ga_stats = result.get('ga_stats', {})
            
            # Analyze convergence
            if 'ga_generations' in solver_info:
                generations = solver_info['ga_generations']
                convergence = solver_info.get('ga_convergence', 'unknown')
                
                print(f"GA Convergence Profile:")
                print(f"  Generations: {generations}")
                print(f"  Convergence reason: {convergence}")
                print(f"  Final fitness: {result.get('fitness_score', 'unknown')}")
                
                # Verify reasonable convergence
                self.assertGreater(generations, 0)
                self.assertLess(generations, 1000)  # Should converge reasonably
    
    def test_ga_memory_profiling(self):
        """Profile GA memory usage"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run GA
            result = self.route_optimizer.optimize_route(
                start_node=self.start_node,
                target_distance_km=2.0,
                objective=self.route_optimizer.RouteObjective.MAXIMIZE_ELEVATION,
                algorithm="genetic"
            )
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            print(f"GA Memory Profile:")
            print(f"  Initial memory: {initial_memory:.1f}MB")
            print(f"  Final memory: {final_memory:.1f}MB")
            print(f"  Memory increase: {memory_increase:.1f}MB")
            
            # Memory increase should be reasonable
            self.assertLess(memory_increase, 1000)  # Less than 1GB increase
            
        except ImportError:
            self.skipTest("psutil not available for memory profiling")


if __name__ == '__main__':
    # Run with high verbosity for detailed output
    unittest.main(verbosity=2)