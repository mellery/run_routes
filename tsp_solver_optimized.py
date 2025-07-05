#!/usr/bin/env python3
"""
Optimized TSP Solver with Progress Indicators and Performance Improvements
Enhanced version of the original TSP solver with better performance and user feedback
"""

import time
import threading
from typing import List, Tuple, Dict, Optional
from tsp_solver import *

class OptimizedGeneticAlgorithmTSP(GeneticAlgorithmTSP):
    """Genetic Algorithm TSP solver with progress indicators and timeouts"""
    
    def __init__(self, graph: nx.Graph, start_node: int, objective: str = RouteObjective.MINIMIZE_DISTANCE):
        super().__init__(graph, start_node, objective)
        # Reduced parameters for better performance
        self.population_size = 20
        self.generations = 30
        self.mutation_rate = 0.02
        self.elite_size = 5
        self.timeout_seconds = 30  # 30 second timeout
        self.progress_callback = None
        
    def set_progress_callback(self, callback):
        """Set callback function for progress updates"""
        self.progress_callback = callback
        
    def solve(self, max_nodes: Optional[int] = None) -> Tuple[List[int], float]:
        """Solve TSP using genetic algorithm with progress feedback"""
        start_time = time.time()
        
        if max_nodes:
            candidate_nodes = self._get_closest_nodes(max_nodes)
        else:
            candidate_nodes = [n for n in self.nodes if n != self.start_node]
        
        if len(candidate_nodes) < 2:
            return [self.start_node], 0
        
        # Initialize population
        if self.progress_callback:
            self.progress_callback("Initializing population...")
        
        population = self._create_initial_population(candidate_nodes)
        
        best_route = None
        best_cost = float('inf')
        stagnation_count = 0
        
        for generation in range(self.generations):
            # Check timeout
            if time.time() - start_time > self.timeout_seconds:
                if self.progress_callback:
                    self.progress_callback(f"Timeout reached, returning best solution...")
                break
            
            # Progress update
            if self.progress_callback and generation % 5 == 0:
                progress = (generation / self.generations) * 100
                self.progress_callback(f"Generation {generation}/{self.generations} ({progress:.0f}%)")
            
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                cost = self.get_route_cost(individual)
                fitness_scores.append((individual, cost))
            
            # Sort by fitness (lower cost = better)
            fitness_scores.sort(key=lambda x: x[1])
            
            # Update best solution
            previous_best = best_cost
            if fitness_scores[0][1] < best_cost:
                best_route = fitness_scores[0][0][:]
                best_cost = fitness_scores[0][1]
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Early stopping if no improvement for several generations
            if stagnation_count >= 10:
                if self.progress_callback:
                    self.progress_callback(f"Early stopping at generation {generation}")
                break
            
            # Create next generation
            population = self._create_next_generation(fitness_scores)
        
        if self.progress_callback:
            self.progress_callback("Optimization complete!")
        
        return best_route, best_cost

class OptimizedDistanceConstrainedTSP(DistanceConstrainedTSP):
    """Distance-constrained TSP solver with progress indicators"""
    
    def __init__(self, graph: nx.Graph, start_node: int, target_distance_km: float, 
                 tolerance: float = 0.15, objective: str = RouteObjective.MINIMIZE_DISTANCE):
        super().__init__(graph, start_node, target_distance_km, tolerance, objective)
        self.progress_callback = None
        
    def set_progress_callback(self, callback):
        """Set callback function for progress updates"""
        self.progress_callback = callback
        
    def solve(self, algorithm: str = "nearest_neighbor") -> Tuple[List[int], float]:
        """Solve distance-constrained TSP with progress feedback"""
        
        if self.progress_callback:
            self.progress_callback("Finding candidate nodes...")
        
        # Get candidate nodes within reasonable distance
        max_radius_km = self.target_distance_m / 2000  # Conservative estimate
        candidate_nodes = self._get_nodes_in_radius(max_radius_km)
        
        if self.progress_callback:
            self.progress_callback(f"Found {len(candidate_nodes)} nodes within {max_radius_km:.1f}km")
        
        # Create solver with reduced parameters for performance
        if algorithm == "genetic":
            solver = OptimizedGeneticAlgorithmTSP(self.graph, self.start_node, self.objective)
            if self.progress_callback:
                solver.set_progress_callback(self.progress_callback)
        else:
            solver = NearestNeighborTSP(self.graph, self.start_node, self.objective)
        
        # Try different subset sizes to find a route within distance constraints
        best_route = None
        best_cost = float('inf')
        
        # Limit the range for better performance
        max_nodes_to_try = min(len(candidate_nodes) + 1, 15)  # Reduced from 20
        
        for num_nodes in range(3, max_nodes_to_try):
            if self.progress_callback:
                progress = ((num_nodes - 3) / (max_nodes_to_try - 3)) * 100
                self.progress_callback(f"Trying {num_nodes} nodes ({progress:.0f}% of search space)")
            
            try:
                route, cost = solver.solve(max_nodes=num_nodes)
                route_stats = self.get_route_stats(route)
                route_distance = route_stats.get('total_distance_m', 0)
                
                # Check if route meets distance constraints
                if self.min_distance <= route_distance <= self.max_distance:
                    if cost < best_cost:
                        best_route = route
                        best_cost = cost
                        if self.progress_callback:
                            self.progress_callback(f"✅ Found good route: {route_distance/1000:.2f}km with {num_nodes} nodes")
                    
                    # Early exit if we found a very good solution
                    if algorithm == "nearest_neighbor":
                        break
                        
            except Exception as e:
                if self.progress_callback:
                    self.progress_callback(f"⚠️ Error with {num_nodes} nodes: {str(e)[:50]}")
                continue
        
        if best_route is None:
            # Fallback: return the best available route even if it doesn't meet distance constraints
            if self.progress_callback:
                self.progress_callback("⚠️ No route found within distance constraints, using fallback...")
            
            solver = NearestNeighborTSP(self.graph, self.start_node, self.objective)
            best_route, best_cost = solver.solve(max_nodes=min(10, len(candidate_nodes)))
        
        return best_route, best_cost

class OptimizedRunningRouteOptimizer(RunningRouteOptimizer):
    """Running route optimizer with progress indicators and better performance"""
    
    def find_optimal_route(self, start_node: int, target_distance_km: float, 
                          objective: str = RouteObjective.MINIMIZE_DISTANCE,
                          algorithm: str = "nearest_neighbor") -> Dict:
        """Find optimal running route with progress feedback"""
        
        print(f"🔧 Using optimized TSP solver (with progress feedback)...")
        
        # Progress indicator in a separate thread
        progress_messages = []
        
        def progress_callback(message):
            progress_messages.append(message)
            print(f"   {message}")
        
        start_time = time.time()
        
        # Create optimized distance-constrained solver
        solver = OptimizedDistanceConstrainedTSP(
            self.graph, start_node, target_distance_km, 
            tolerance=0.2, objective=objective
        )
        solver.set_progress_callback(progress_callback)
        
        # Solve the problem
        route, cost = solver.solve(algorithm=algorithm)
        
        # Get detailed statistics
        stats = solver.get_route_stats(route)
        
        solve_time = time.time() - start_time
        
        result = {
            'route': route,
            'cost': cost,
            'stats': stats,
            'solve_time': solve_time,
            'objective': objective,
            'algorithm': algorithm,
            'target_distance_km': target_distance_km,
            'progress_log': progress_messages
        }
        
        print(f"✅ Solution found in {solve_time:.2f} seconds")
        print(f"   Route length: {len(route)} nodes")
        print(f"   Actual distance: {stats.get('total_distance_km', 0):.2f} km")
        
        return result

def create_fast_optimizer(graph):
    """Create a fast optimizer instance"""
    return OptimizedRunningRouteOptimizer(graph)

if __name__ == "__main__":
    print("Optimized TSP Solver module loaded successfully!")
    print("Key improvements:")
    print("• Progress indicators during optimization")
    print("• 30-second timeout for genetic algorithm")  
    print("• Reduced generations (30) and population (20) for speed")
    print("• Early stopping when no improvement")
    print("• Limited search space (15 max nodes instead of 20)")
    print("• Better error handling and fallback routes")