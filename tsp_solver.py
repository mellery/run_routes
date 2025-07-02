#!/usr/bin/env python3
"""
TSP Solver implementations for running route optimization
Implements various algorithms for solving Traveling Salesman Problem variants
optimized for running routes with elevation considerations
"""

import random
import math
import time
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
import networkx as nx


class RouteObjective:
    """Defines different optimization objectives for route planning"""
    
    MINIMIZE_DISTANCE = "minimize_distance"
    MAXIMIZE_ELEVATION = "maximize_elevation"
    BALANCED_ROUTE = "balanced_route"
    MINIMIZE_DIFFICULTY = "minimize_difficulty"


class TSPSolver:
    """Base class for TSP solvers with running-specific optimizations"""
    
    def __init__(self, graph: nx.Graph, start_node: int, objective: str = RouteObjective.MINIMIZE_DISTANCE):
        self.graph = graph
        self.start_node = start_node
        self.objective = objective
        self.nodes = list(graph.nodes())
        
        # Precompute distance/weight matrices for efficiency
        self._build_distance_matrix()
        
    def _build_distance_matrix(self):
        """Build distance matrices for different objectives"""
        print(f"  Building distance matrix for {len(self.nodes)} nodes...")
        print(f"  This may take a moment for large networks...")
        
        self.distance_matrix = {}
        self.elevation_matrix = {}
        self.running_weight_matrix = {}
        
        total_pairs = len(self.nodes)
        processed = 0
        
        for u in self.nodes:
            self.distance_matrix[u] = {}
            self.elevation_matrix[u] = {}
            self.running_weight_matrix[u] = {}
            
            # Progress indicator
            processed += 1
            if processed % 50 == 0 or processed == total_pairs:
                progress = (processed / total_pairs) * 100
                print(f"  Matrix calculation: {processed}/{total_pairs} nodes ({progress:.0f}%)")
            
            for v in self.nodes:
                if u == v:
                    self.distance_matrix[u][v] = 0
                    self.elevation_matrix[u][v] = 0
                    self.running_weight_matrix[u][v] = 0
                else:
                    # Use shortest path if not directly connected
                    try:
                        path = nx.shortest_path(self.graph, u, v, weight='length')
                        
                        # Calculate total distance
                        total_distance = 0
                        total_elevation_gain = 0
                        total_running_weight = 0
                        
                        for i in range(len(path) - 1):
                            edge_data = self.graph.get_edge_data(path[i], path[i+1])
                            if edge_data:
                                # Handle multi-edges by taking the first edge
                                if isinstance(edge_data, dict) and 0 in edge_data:
                                    edge_data = edge_data[0]
                                
                                total_distance += edge_data.get('length', float('inf'))
                                total_elevation_gain += max(0, edge_data.get('elevation_gain', 0))
                                total_running_weight += edge_data.get('running_weight', edge_data.get('length', float('inf')))
                        
                        self.distance_matrix[u][v] = total_distance
                        self.elevation_matrix[u][v] = total_elevation_gain
                        self.running_weight_matrix[u][v] = total_running_weight
                        
                    except nx.NetworkXNoPath:
                        # No path between nodes
                        self.distance_matrix[u][v] = float('inf')
                        self.elevation_matrix[u][v] = 0
                        self.running_weight_matrix[u][v] = float('inf')
    
    def get_route_cost(self, route: List[int]) -> float:
        """Calculate cost of a route based on the objective"""
        if len(route) < 2:
            return 0
        
        total_cost = 0
        total_distance = 0
        total_elevation = 0
        
        # Calculate route metrics
        for i in range(len(route)):
            current = route[i]
            next_node = route[(i + 1) % len(route)]  # Return to start
            
            total_distance += self.distance_matrix[current][next_node]
            total_elevation += self.elevation_matrix[current][next_node]
        
        # Apply objective function
        if self.objective == RouteObjective.MINIMIZE_DISTANCE:
            total_cost = total_distance
            
        elif self.objective == RouteObjective.MAXIMIZE_ELEVATION:
            # Negative elevation gain (we want to maximize)
            total_cost = -total_elevation + total_distance * 0.1  # Small distance penalty
            
        elif self.objective == RouteObjective.BALANCED_ROUTE:
            # Balance distance and elevation
            normalized_distance = total_distance / 1000  # Normalize to km
            normalized_elevation = total_elevation / 100  # Normalize to 100m units
            total_cost = normalized_distance + 0.5 * (-normalized_elevation)  # Favor some elevation
            
        elif self.objective == RouteObjective.MINIMIZE_DIFFICULTY:
            # Use running weights that account for grades
            for i in range(len(route)):
                current = route[i]
                next_node = route[(i + 1) % len(route)]
                total_cost += self.running_weight_matrix[current][next_node]
        
        return total_cost
    
    def get_route_stats(self, route: List[int]) -> Dict:
        """Get detailed statistics for a route"""
        if len(route) < 2:
            return {}
        
        total_distance = 0
        total_elevation_gain = 0
        total_elevation_loss = 0
        max_grade = 0
        
        for i in range(len(route)):
            current = route[i]
            next_node = route[(i + 1) % len(route)]
            
            total_distance += self.distance_matrix[current][next_node]
            elev_gain = self.elevation_matrix[current][next_node]
            total_elevation_gain += elev_gain
            
            # Get actual path for detailed stats
            try:
                path = nx.shortest_path(self.graph, current, next_node, weight='length')
                for j in range(len(path) - 1):
                    edge_data = self.graph.get_edge_data(path[j], path[j+1])
                    if edge_data:
                        if isinstance(edge_data, dict) and 0 in edge_data:
                            edge_data = edge_data[0]
                        
                        grade = edge_data.get('grade', 0)
                        max_grade = max(max_grade, grade)
                        
                        elev_change = edge_data.get('elevation_gain', 0)
                        if elev_change < 0:
                            total_elevation_loss += abs(elev_change)
            except:
                pass
        
        return {
            'total_distance_m': total_distance,
            'total_distance_km': total_distance / 1000,
            'total_elevation_gain_m': total_elevation_gain,
            'total_elevation_loss_m': total_elevation_loss,
            'net_elevation_gain_m': total_elevation_gain - total_elevation_loss,
            'max_grade_percent': max_grade,
            'estimated_time_min': total_distance / 166.67,  # Assume 10 km/h pace
            'difficulty_score': self.get_route_cost(route) if self.objective == RouteObjective.MINIMIZE_DIFFICULTY else None
        }


class NearestNeighborTSP(TSPSolver):
    """Nearest Neighbor TSP solver - fast but not optimal"""
    
    def solve(self, max_nodes: Optional[int] = None) -> Tuple[List[int], float]:
        """Solve TSP using nearest neighbor heuristic"""
        if max_nodes:
            # Limit to closest nodes for performance
            candidate_nodes = self._get_closest_nodes(max_nodes)
        else:
            candidate_nodes = [n for n in self.nodes if n != self.start_node]
        
        route = [self.start_node]
        remaining = set(candidate_nodes)
        current = self.start_node
        
        while remaining:
            # Find nearest unvisited node
            nearest = min(remaining, key=lambda n: self.distance_matrix[current][n])
            route.append(nearest)
            remaining.remove(nearest)
            current = nearest
        
        cost = self.get_route_cost(route)
        return route, cost
    
    def _get_closest_nodes(self, max_nodes: int) -> List[int]:
        """Get the closest N nodes to the start node"""
        distances = [(n, self.distance_matrix[self.start_node][n]) 
                    for n in self.nodes if n != self.start_node]
        distances.sort(key=lambda x: x[1])
        return [n for n, d in distances[:max_nodes]]


class GeneticAlgorithmTSP(TSPSolver):
    """Genetic Algorithm TSP solver - better optimization but slower"""
    
    def __init__(self, graph: nx.Graph, start_node: int, objective: str = RouteObjective.MINIMIZE_DISTANCE):
        super().__init__(graph, start_node, objective)
        # Reduced parameters for better performance
        self.population_size = 20  # Was 50
        self.generations = 30      # Was 100
        self.mutation_rate = 0.02
        self.elite_size = 5        # Was 10
    
    def solve(self, max_nodes: Optional[int] = None) -> Tuple[List[int], float]:
        """Solve TSP using genetic algorithm with progress feedback"""
        start_time = time.time()
        timeout_seconds = 30  # 30 second timeout
        
        if max_nodes:
            candidate_nodes = self._get_closest_nodes(max_nodes)
        else:
            candidate_nodes = [n for n in self.nodes if n != self.start_node]
        
        if len(candidate_nodes) < 2:
            return [self.start_node], 0
        
        # Initialize population
        print(f"  Initializing genetic algorithm ({self.generations} generations)...")
        population = self._create_initial_population(candidate_nodes)
        
        best_route = None
        best_cost = float('inf')
        stagnation_count = 0
        
        for generation in range(self.generations):
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                print(f"  Timeout reached at generation {generation}, returning best solution...")
                break
            
            # Progress update every 5 generations
            if generation % 5 == 0 or generation == self.generations - 1:
                progress = (generation / self.generations) * 100
                print(f"  Generation {generation}/{self.generations} ({progress:.0f}%)")
            
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
            
            # Early stopping if no improvement for 10 generations
            if stagnation_count >= 10:
                print(f"  Early stopping at generation {generation} (no improvement)")
                break
            
            # Create next generation
            population = self._create_next_generation(fitness_scores)
        
        print(f"  Genetic algorithm complete!")
        return best_route, best_cost
    
    def _create_initial_population(self, candidate_nodes: List[int]) -> List[List[int]]:
        """Create initial population of random routes"""
        population = []
        
        for _ in range(self.population_size):
            route = [self.start_node] + random.sample(candidate_nodes, len(candidate_nodes))
            population.append(route)
        
        return population
    
    def _create_next_generation(self, fitness_scores: List[Tuple[List[int], float]]) -> List[List[int]]:
        """Create next generation using selection, crossover, and mutation"""
        next_generation = []
        
        # Keep elite individuals
        for i in range(self.elite_size):
            next_generation.append(fitness_scores[i][0][:])
        
        # Create offspring
        while len(next_generation) < self.population_size:
            parent1 = self._tournament_selection(fitness_scores)
            parent2 = self._tournament_selection(fitness_scores)
            
            child1, child2 = self._crossover(parent1, parent2)
            
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            next_generation.extend([child1, child2])
        
        return next_generation[:self.population_size]
    
    def _tournament_selection(self, fitness_scores: List[Tuple[List[int], float]], tournament_size: int = 3) -> List[int]:
        """Select parent using tournament selection"""
        tournament = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
        return min(tournament, key=lambda x: x[1])[0]
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Order crossover (OX) for TSP"""
        size = len(parent1)
        start_node = parent1[0]  # Keep start node fixed
        
        # Work with the non-start nodes
        p1_route = parent1[1:]
        p2_route = parent2[1:]
        
        if len(p1_route) < 2:
            return parent1[:], parent2[:]
        
        # Select crossover points
        start_pos = random.randint(0, len(p1_route) - 1)
        end_pos = random.randint(start_pos, len(p1_route) - 1)
        
        # Create children
        child1_route = [None] * len(p1_route)
        child2_route = [None] * len(p2_route)
        
        # Copy crossover sections
        child1_route[start_pos:end_pos+1] = p1_route[start_pos:end_pos+1]
        child2_route[start_pos:end_pos+1] = p2_route[start_pos:end_pos+1]
        
        # Fill remaining positions
        self._fill_crossover_gaps(child1_route, p2_route, start_pos, end_pos)
        self._fill_crossover_gaps(child2_route, p1_route, start_pos, end_pos)
        
        # Add start node back
        child1 = [start_node] + child1_route
        child2 = [start_node] + child2_route
        
        return child1, child2
    
    def _fill_crossover_gaps(self, child: List, other_parent: List, start_pos: int, end_pos: int):
        """Fill gaps in crossover child"""
        used = set(child[start_pos:end_pos+1])
        fill_values = [x for x in other_parent if x not in used]
        
        fill_idx = 0
        for i in range(len(child)):
            if child[i] is None and fill_idx < len(fill_values):
                child[i] = fill_values[fill_idx]
                fill_idx += 1
    
    def _mutate(self, individual: List[int]) -> List[int]:
        """Mutate individual using 2-opt swap"""
        if random.random() > self.mutation_rate or len(individual) < 4:
            return individual
        
        mutated = individual[:]
        route_part = mutated[1:]  # Don't mutate start node
        
        if len(route_part) >= 2:
            i, j = random.sample(range(len(route_part)), 2)
            route_part[i], route_part[j] = route_part[j], route_part[i]
            mutated[1:] = route_part
        
        return mutated
    
    def _get_closest_nodes(self, max_nodes: int) -> List[int]:
        """Get the closest N nodes to the start node"""
        distances = [(n, self.distance_matrix[self.start_node][n]) 
                    for n in self.nodes if n != self.start_node]
        distances.sort(key=lambda x: x[1])
        return [n for n, d in distances[:max_nodes]]


class DistanceConstrainedTSP(TSPSolver):
    """TSP solver with distance constraints for running routes"""
    
    def __init__(self, graph: nx.Graph, start_node: int, target_distance_km: float, 
                 tolerance: float = 0.15, objective: str = RouteObjective.MINIMIZE_DISTANCE):
        super().__init__(graph, start_node, objective)
        self.target_distance_m = target_distance_km * 1000
        self.tolerance = tolerance
        self.min_distance = self.target_distance_m * (1 - tolerance)
        self.max_distance = self.target_distance_m * (1 + tolerance)
    
    def solve(self, algorithm: str = "nearest_neighbor") -> Tuple[List[int], float]:
        """Solve distance-constrained TSP"""
        
        # Get candidate nodes within reasonable distance
        max_radius_km = self.target_distance_m / 2000  # Conservative estimate
        print(f"  Finding nodes within {max_radius_km:.1f}km radius...")
        candidate_nodes = self._get_nodes_in_radius(max_radius_km)
        print(f"  Found {len(candidate_nodes)} candidate nodes")
        
        if algorithm == "genetic":
            solver = GeneticAlgorithmTSP(self.graph, self.start_node, self.objective)
            # Already reduced in the class init
        else:
            solver = NearestNeighborTSP(self.graph, self.start_node, self.objective)
        
        # Try different subset sizes to find a route within distance constraints
        best_route = None
        best_cost = float('inf')
        
        max_nodes_to_try = min(len(candidate_nodes) + 1, 15)  # Reduced from 20 for performance
        print(f"  Trying different route sizes (3 to {max_nodes_to_try-1} nodes)...")
        
        for num_nodes in range(3, max_nodes_to_try):
            progress = ((num_nodes - 3) / (max_nodes_to_try - 3)) * 100
            print(f"  Trying {num_nodes} nodes ({progress:.0f}% of search space)...")
            
            try:
                route, cost = solver.solve(max_nodes=num_nodes)
                route_stats = self.get_route_stats(route)
                route_distance = route_stats.get('total_distance_m', 0)
                
                # Check if route meets distance constraints
                if self.min_distance <= route_distance <= self.max_distance:
                    if cost < best_cost:
                        best_route = route
                        best_cost = cost
                        print(f"  ✅ Found good route: {route_distance/1000:.2f}km with {num_nodes} nodes")
                    
                    # Early exit for nearest neighbor if we found a good solution
                    if algorithm == "nearest_neighbor":
                        print(f"  Early exit with nearest neighbor solution")
                        break
                
                # If we're getting too far from target, stop
                if route_distance > self.max_distance * 1.5:
                    break
                    
            except Exception as e:
                print(f"  ⚠️ Error with {num_nodes} nodes: {str(e)[:50]}")
                continue
        
        if best_route is None:
            # Fallback: return a simple route even if it doesn't meet constraints
            print(f"  ⚠️ No route found within distance constraints, using fallback...")
            solver = NearestNeighborTSP(self.graph, self.start_node, self.objective)
            best_route, best_cost = solver.solve(max_nodes=5)
            print(f"  📍 Fallback route generated with 5 nodes")
        
        return best_route, best_cost
    
    def _get_nodes_in_radius(self, radius_km: float) -> List[int]:
        """Get nodes within specified radius of start node"""
        from route import get_nodes_within_distance
        return get_nodes_within_distance(self.graph, self.start_node, radius_km)


class RunningRouteOptimizer:
    """Main class for running route optimization"""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
    
    def find_optimal_route(self, start_node: int, target_distance_km: float, 
                          objective: str = RouteObjective.MINIMIZE_DISTANCE,
                          algorithm: str = "nearest_neighbor") -> Dict:
        """Find optimal running route with specified parameters"""
        
        print(f"Finding optimal route...")
        print(f"  Start node: {start_node}")
        print(f"  Target distance: {target_distance_km:.1f} km")
        print(f"  Objective: {objective}")
        print(f"  Algorithm: {algorithm}")
        
        start_time = time.time()
        
        # Create distance-constrained solver
        solver = DistanceConstrainedTSP(
            self.graph, start_node, target_distance_km, 
            tolerance=0.2, objective=objective
        )
        
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
            'target_distance_km': target_distance_km
        }
        
        print(f"  Solution found in {solve_time:.2f} seconds")
        print(f"  Route length: {len(route)} nodes")
        print(f"  Actual distance: {stats.get('total_distance_km', 0):.2f} km")
        
        return result


if __name__ == "__main__":
    # Basic test
    print("TSP Solver module loaded successfully!")
    print("Available objectives:", [
        RouteObjective.MINIMIZE_DISTANCE,
        RouteObjective.MAXIMIZE_ELEVATION,
        RouteObjective.BALANCED_ROUTE,
        RouteObjective.MINIMIZE_DIFFICULTY
    ])