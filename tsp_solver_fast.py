#!/usr/bin/env python3
"""
Fast TSP Solver - No Precomputed Distance Matrix
Efficient TSP solver that computes distances on-demand instead of precomputing all pairs
"""

import random
import math
import time
import networkx as nx
from typing import List, Tuple, Dict, Optional
from enum import Enum

class RouteObjective(Enum):
    MINIMIZE_DISTANCE = "minimize_distance"
    MAXIMIZE_ELEVATION = "maximize_elevation"
    BALANCED_ROUTE = "balanced_route"
    MINIMIZE_DIFFICULTY = "minimize_difficulty"

class FastTSPSolver:
    """Fast TSP solver base class without precomputed distance matrix"""
    
    def __init__(self, graph: nx.Graph, start_node: int, objective: str = RouteObjective.MINIMIZE_DISTANCE):
        self.graph = graph
        self.start_node = start_node
        self.objective = objective
        self.nodes = list(graph.nodes())
        
        # Use a cache for computed distances to avoid recalculation
        self._distance_cache = {}
        self._elevation_cache = {}
        self._running_weight_cache = {}
        
        print(f"  Fast TSP solver initialized for {len(self.nodes)} nodes")
    
    def get_distance(self, u: int, v: int) -> float:
        """Get distance between two nodes with caching"""
        if u == v:
            return 0
        
        # Check cache first
        cache_key = (min(u, v), max(u, v))
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]
        
        # Compute distance using shortest path
        try:
            path = nx.shortest_path(self.graph, u, v, weight='length')
            total_distance = 0
            
            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i+1])
                if edge_data:
                    if isinstance(edge_data, dict) and 0 in edge_data:
                        edge_data = edge_data[0]
                    total_distance += edge_data.get('length', float('inf'))
            
            # Cache the result
            self._distance_cache[cache_key] = total_distance
            return total_distance
            
        except nx.NetworkXNoPath:
            return float('inf')
    
    def get_route_cost(self, route: List[int]) -> float:
        """Calculate route cost based on objective"""
        if len(route) < 2:
            return 0
        
        total_cost = 0
        
        # Add edges between consecutive nodes
        for i in range(len(route) - 1):
            if self.objective == RouteObjective.MINIMIZE_DISTANCE:
                total_cost += self.get_distance(route[i], route[i+1])
            elif self.objective == RouteObjective.MAXIMIZE_ELEVATION:
                # For maximizing elevation, use negative elevation gain as cost
                total_cost -= self.get_elevation_gain(route[i], route[i+1])
            else:  # BALANCED_ROUTE, MINIMIZE_DIFFICULTY
                total_cost += self.get_running_weight(route[i], route[i+1])
        
        # Add return to start
        if len(route) > 1:
            if self.objective == RouteObjective.MINIMIZE_DISTANCE:
                total_cost += self.get_distance(route[-1], route[0])
            elif self.objective == RouteObjective.MAXIMIZE_ELEVATION:
                total_cost -= self.get_elevation_gain(route[-1], route[0])
            else:
                total_cost += self.get_running_weight(route[-1], route[0])
        
        return total_cost
    
    def get_elevation_gain(self, u: int, v: int) -> float:
        """Get elevation gain between two nodes"""
        # Simplified - just use node elevation difference
        try:
            u_elev = self.graph.nodes[u].get('elevation', 0)
            v_elev = self.graph.nodes[v].get('elevation', 0)
            return max(0, v_elev - u_elev)  # Only positive gains
        except:
            return 0
    
    def get_running_weight(self, u: int, v: int) -> float:
        """Get running-specific weight between nodes"""
        distance = self.get_distance(u, v)
        elevation_gain = self.get_elevation_gain(u, v)
        
        # Simple running weight: distance + elevation penalty
        return distance + (elevation_gain * 2)  # 2x penalty for elevation gain

class FastNearestNeighborTSP(FastTSPSolver):
    """Fast Nearest Neighbor TSP solver"""
    
    def solve(self, max_nodes: Optional[int] = None) -> Tuple[List[int], float]:
        """Solve TSP using nearest neighbor heuristic"""
        print(f"  Starting nearest neighbor algorithm...")
        
        if max_nodes:
            # Use provided candidate nodes if available to avoid recalculation
            if hasattr(self, 'candidate_nodes') and self.candidate_nodes:
                candidate_nodes = self._get_closest_nodes(max_nodes, self.candidate_nodes)
            else:
                candidate_nodes = self._get_closest_nodes(max_nodes)
            print(f"  Using {len(candidate_nodes)} closest nodes")
        else:
            candidate_nodes = [n for n in self.nodes if n != self.start_node]
        
        if len(candidate_nodes) < 1:
            return [self.start_node], 0
        
        route = [self.start_node]
        remaining = set(candidate_nodes)
        current = self.start_node
        
        step = 0
        total_steps = len(remaining)
        
        while remaining:
            step += 1
            if step % 10 == 0 or step == total_steps:
                progress = (step / total_steps) * 100
                print(f"  Building route: {step}/{total_steps} nodes ({progress:.0f}%)")
            
            # Find nearest unvisited node
            best_node = None
            best_distance = float('inf')
            
            for node in remaining:
                distance = self.get_distance(current, node)
                if distance < best_distance:
                    best_distance = distance
                    best_node = node
            
            if best_node is not None:
                route.append(best_node)
                remaining.remove(best_node)
                current = best_node
            else:
                break
        
        cost = self.get_route_cost(route)
        print(f"  ✅ Nearest neighbor complete: {len(route)} nodes")
        return route, cost
    
    def _get_closest_nodes(self, max_nodes: int, candidate_nodes=None) -> List[int]:
        """Get the closest N nodes to the start node"""
        print(f"  Finding {max_nodes} closest nodes to start...")
        
        # Use provided candidate nodes if available, otherwise use all nodes
        if candidate_nodes:
            nodes_to_check = [n for n in candidate_nodes if n != self.start_node]
            print(f"  Using pre-filtered {len(nodes_to_check)} candidate nodes")
        else:
            nodes_to_check = [n for n in self.nodes if n != self.start_node]
        
        distances = []
        
        for i, node in enumerate(nodes_to_check):
            if i % 100 == 0 and len(nodes_to_check) > 200:
                progress = (i / len(nodes_to_check)) * 100
                print(f"    Checking distances: {i}/{len(nodes_to_check)} ({progress:.0f}%)")
            
            distance = self.get_distance(self.start_node, node)
            distances.append((node, distance))
        
        distances.sort(key=lambda x: x[1])
        result = [n for n, d in distances[:max_nodes]]
        print(f"  ✅ Found {len(result)} closest nodes")
        return result

class FastDistanceConstrainedTSP(FastTSPSolver):
    """Fast distance-constrained TSP solver"""
    
    def __init__(self, graph: nx.Graph, start_node: int, target_distance_km: float, 
                 tolerance: float = 0.15, objective: str = RouteObjective.MINIMIZE_DISTANCE):
        super().__init__(graph, start_node, objective)
        self.target_distance_m = target_distance_km * 1000
        self.tolerance = tolerance
        self.min_distance = self.target_distance_m * (1 - tolerance)
        self.max_distance = self.target_distance_m * (1 + tolerance)
    
    def solve(self, algorithm: str = "nearest_neighbor") -> Tuple[List[int], float]:
        """Solve distance-constrained TSP"""
        print(f"  Target distance: {self.target_distance_m/1000:.2f}km (±{self.tolerance*100:.0f}%)")
        
        # Get candidate nodes within reasonable distance
        max_radius_km = self.target_distance_m / 2000  # Conservative estimate
        print(f"  Finding nodes within {max_radius_km:.1f}km radius...")
        candidate_nodes = self._get_nodes_in_radius(max_radius_km)
        print(f"  Found {len(candidate_nodes)} candidate nodes")
        
        solver = FastNearestNeighborTSP(self.graph, self.start_node, self.objective)
        
        # Try different subset sizes to find a route within distance constraints
        best_route = None
        best_cost = float('inf')
        
        max_nodes_to_try = min(len(candidate_nodes) + 1, 12)  # Reduced for performance
        print(f"  Trying different route sizes (3 to {max_nodes_to_try-1} nodes)...")
        
        for num_nodes in range(3, max_nodes_to_try):
            progress = ((num_nodes - 3) / (max_nodes_to_try - 3)) * 100
            print(f"  Trying {num_nodes} nodes ({progress:.0f}% of search space)...")
            
            try:
                # Pass candidate nodes to avoid recalculating distances to all nodes
                solver.candidate_nodes = candidate_nodes
                route, cost = solver.solve(max_nodes=num_nodes)
                route_stats = self.get_route_stats(route)
                route_distance = route_stats.get('total_distance_m', 0)
                
                # Check if route meets distance constraints
                if self.min_distance <= route_distance <= self.max_distance:
                    if cost < best_cost:
                        best_route = route
                        best_cost = cost
                        print(f"  ✅ Found good route: {route_distance/1000:.2f}km with {num_nodes} nodes")
                    
                    # Early exit if we found a good solution
                    print(f"  Early exit with good solution")
                    break
                else:
                    distance_km = route_distance / 1000
                    target_km = self.target_distance_m / 1000
                    print(f"    Route {distance_km:.2f}km doesn't match target {target_km:.2f}km")
                    
            except Exception as e:
                print(f"  ⚠️ Error with {num_nodes} nodes: {str(e)[:50]}")
                continue
        
        if best_route is None:
            # Fallback: return a simple route
            print(f"  ⚠️ No route found within distance constraints, using fallback...")
            solver = FastNearestNeighborTSP(self.graph, self.start_node, self.objective)
            best_route, best_cost = solver.solve(max_nodes=5)
            print(f"  📍 Fallback route generated with 5 nodes")
        
        return best_route, best_cost
    
    def _get_nodes_in_radius(self, radius_km: float) -> List[int]:
        """Get nodes within specified radius of start node"""
        from route import get_nodes_within_distance
        return get_nodes_within_distance(self.graph, self.start_node, radius_km)
    
    def get_route_stats(self, route: List[int]) -> Dict:
        """Get route statistics"""
        if not route:
            return {}
        
        total_distance = 0
        total_elevation_gain = 0
        total_elevation_loss = 0
        
        # Calculate route segments
        for i in range(len(route)):
            if i < len(route) - 1:
                # Segment to next node
                next_node = route[i + 1]
            else:
                # Return to start
                next_node = route[0]
            
            current_node = route[i]
            distance = self.get_distance(current_node, next_node)
            total_distance += distance
            
            # Elevation calculations
            current_elev = self.graph.nodes[current_node].get('elevation', 0)
            next_elev = self.graph.nodes[next_node].get('elevation', 0)
            elev_change = next_elev - current_elev
            
            if elev_change > 0:
                total_elevation_gain += elev_change
            else:
                total_elevation_loss += abs(elev_change)
        
        return {
            'total_distance_m': total_distance,
            'total_distance_km': total_distance / 1000,
            'total_elevation_gain_m': total_elevation_gain,
            'total_elevation_loss_m': total_elevation_loss,
            'net_elevation_gain_m': total_elevation_gain - total_elevation_loss,
            'estimated_time_min': (total_distance / 1000) * 6,  # 6 min/km estimate
        }

class FastRunningRouteOptimizer:
    """Fast running route optimizer without distance matrix precomputation"""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
    
    def find_optimal_route(self, start_node: int, target_distance_km: float, 
                          objective: str = RouteObjective.MINIMIZE_DISTANCE,
                          algorithm: str = "nearest_neighbor") -> Dict:
        """Find optimal running route with fast algorithm"""
        
        print(f"🚀 Finding optimal route (fast algorithm)...")
        print(f"   Start node: {start_node}")
        print(f"   Target distance: {target_distance_km:.1f} km")
        print(f"   Objective: {objective}")
        print(f"   Algorithm: {algorithm}")
        
        start_time = time.time()
        
        # Create fast distance-constrained solver
        solver = FastDistanceConstrainedTSP(
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
        
        print(f"✅ Solution found in {solve_time:.2f} seconds")
        print(f"   Route length: {len(route)} nodes")
        print(f"   Actual distance: {stats.get('total_distance_km', 0):.2f} km")
        
        return result

if __name__ == "__main__":
    print("Fast TSP Solver module loaded successfully!")
    print("Key improvements:")
    print("• No precomputed distance matrix (avoids 1M+ shortest path calculations)")
    print("• On-demand distance calculation with caching")
    print("• Progress indicators throughout optimization")
    print("• Reduced search space for better performance")
    print("• Should be much faster for large networks!")