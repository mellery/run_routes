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
        """Get nodes distributed at various distances from start node (not just closest)"""
        print(f"  Finding {max_nodes} distributed nodes around start...")
        
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
        
        # Filter out nodes that are too close for meaningful routes
        # For target distances, we want nodes at least 200m away to create substantial routes
        min_distance_m = 200.0  # Minimum 200m from start for meaningful routes
        filtered_distances = [(node, dist) for node, dist in distances if dist >= min_distance_m]
        
        if len(filtered_distances) < max_nodes:
            print(f"    ⚠️ Only {len(filtered_distances)} nodes > {min_distance_m}m, using closest available")
            filtered_distances = distances  # Fall back to all nodes if not enough distant ones
        else:
            print(f"    ✅ Using {len(filtered_distances)} nodes > {min_distance_m}m from start")
        
        # Distribute selection across distance ranges to create longer, more circular routes
        if len(filtered_distances) > max_nodes * 3:
            # Divide nodes into distance bands and select strategically from each band
            total_nodes = len(filtered_distances)
            band_size = total_nodes // max_nodes
            result = []
            
            for i in range(max_nodes):
                band_start = i * band_size
                band_end = min((i + 1) * band_size, total_nodes)
                
                if band_start < len(filtered_distances):
                    # Take a node from the middle of each band (not the closest in the band)
                    # This creates better distribution and longer routes
                    mid_band_idx = band_start + (band_end - band_start) // 3  # Take from first third of band
                    if mid_band_idx < len(filtered_distances):
                        result.append(filtered_distances[mid_band_idx][0])
                    else:
                        result.append(filtered_distances[band_start][0])
            
            print(f"  ✅ Found {len(result)} distributed nodes across distance bands")
            
            # Show distance range for debugging
            if result:
                result_distances = [self.get_distance(self.start_node, node)/1000 for node in result]
                print(f"    Distance range: {min(result_distances):.3f}km - {max(result_distances):.3f}km")
        else:
            # Not enough nodes for banding, take evenly spaced nodes from the filtered list
            spacing = len(filtered_distances) // max_nodes if max_nodes > 0 else 1
            result = [filtered_distances[i * spacing][0] for i in range(max_nodes) 
                     if i * spacing < len(filtered_distances)]
            print(f"  ✅ Found {len(result)} evenly spaced nodes")
        
        return result

class FastDistanceConstrainedTSP(FastTSPSolver):
    """Fast distance-constrained TSP solver"""
    
    def __init__(self, graph: nx.Graph, start_node: int, target_distance_km: float, 
                 tolerance: float = 0.15, objective: str = RouteObjective.MINIMIZE_DISTANCE,
                 filtered_candidates: list = None):
        super().__init__(graph, start_node, objective)
        self.target_distance_m = target_distance_km * 1000
        self.target_distance_km = target_distance_km
        self.filtered_candidates = filtered_candidates
        
        # Use progressive tolerance for longer routes (more lenient for longer distances)
        adjusted_tolerance = min(0.3, tolerance + (target_distance_km - 1) * 0.02)
        self.tolerance = adjusted_tolerance
        self.min_distance = self.target_distance_m * (1 - adjusted_tolerance)
        self.max_distance = self.target_distance_m * (1 + adjusted_tolerance)
    
    def solve(self, algorithm: str = "nearest_neighbor") -> Tuple[List[int], float]:
        """Solve distance-constrained TSP"""
        print(f"  Target distance: {self.target_distance_m/1000:.2f}km (±{self.tolerance*100:.0f}%)")
        print(f"  Algorithm: {algorithm}")
        
        # Use filtered candidates if provided, otherwise get all intersection nodes
        if self.filtered_candidates:
            candidate_nodes = self.filtered_candidates
            print(f"  Using filtered candidates: {len(candidate_nodes)} candidate nodes")
        else:
            # Get ALL intersection nodes (not geometry nodes) in the network
            candidate_nodes = self._get_all_intersection_nodes()
            print(f"  Using ALL intersections in network: {len(candidate_nodes)} candidate nodes")
        
        # Check if genetic algorithm is requested
        if algorithm == "genetic":
            print("  ⚠️  Fast TSP solver doesn't support genetic algorithm")
            print("  ⚠️  Falling back to standard TSP solver with genetic algorithm...")
            
            # Import and use standard TSP solver for genetic algorithm
            try:
                from tsp_solver import GeneticAlgorithmTSP
                # Use filtered candidates if provided, otherwise use intersection filtering
                genetic_candidates = self.filtered_candidates if self.filtered_candidates else candidate_nodes
                genetic_solver = GeneticAlgorithmTSP(self.graph, self.start_node, self.objective, genetic_candidates)
                
                # Try different subset sizes to find a route within distance constraints
                best_route = None
                best_cost = float('inf')
                
                # Adaptive max nodes based on target distance
                adaptive_max = max(20, min(50, int(self.target_distance_km * 15)))  # Smaller for genetic
                max_nodes_to_try = min(len(candidate_nodes) + 1, adaptive_max)
                print(f"    Will try up to {adaptive_max} nodes from {len(candidate_nodes)} total intersections")
                
                for num_nodes in range(3, max_nodes_to_try):
                    progress = ((num_nodes - 3) / (max_nodes_to_try - 3)) * 100
                    print(f"  Trying {num_nodes} nodes ({progress:.0f}% of search space)...")
                    
                    try:
                        route, cost = genetic_solver.solve(max_nodes=num_nodes)
                        route_stats = self.get_route_stats(route)
                        route_distance = route_stats.get('total_distance_m', 0)
                        
                        # Check if route meets distance constraints
                        distance_km = route_distance / 1000
                        target_km = self.target_distance_m / 1000
                        min_km = self.min_distance / 1000
                        max_km = self.max_distance / 1000
                        
                        print(f"    Generated route: {distance_km:.2f}km (target: {target_km:.2f}km, range: {min_km:.2f}-{max_km:.2f}km)")
                        
                        if self.min_distance <= route_distance <= self.max_distance:
                            if cost < best_cost:
                                best_route = route
                                best_cost = cost
                                print(f"  ✅ Found VALID route: {distance_km:.2f}km with {num_nodes} nodes")
                            
                            # Early exit if we found a good solution
                            print(f"  Early exit with good solution")
                            break
                        else:
                            if distance_km < min_km:
                                print(f"    ❌ Route too SHORT: {distance_km:.2f}km < {min_km:.2f}km")
                            else:
                                print(f"    ❌ Route too LONG: {distance_km:.2f}km > {max_km:.2f}km")
                        
                    except Exception as e:
                        print(f"  ⚠️ Error with {num_nodes} nodes: {str(e)[:50]}")
                        continue
                
                if best_route is not None:
                    return best_route, best_cost
                else:
                    print("  ⚠️ Genetic algorithm didn't find valid route, falling back to nearest neighbor")
                    
            except ImportError:
                print("  ⚠️ Could not import genetic algorithm, falling back to nearest neighbor")
        
        # Use nearest neighbor algorithm (default or fallback)
        solver = FastNearestNeighborTSP(self.graph, self.start_node, self.objective)
        # Pass the filtered candidates to avoid using all nodes
        solver.candidate_nodes = candidate_nodes
        
        # Try different subset sizes to find a route within distance constraints
        best_route = None
        best_cost = float('inf')
        
        # Adaptive max nodes based on target distance (more nodes for longer routes)
        # With aggressive filtering, we have clean intersections and can use many more nodes
        adaptive_max = max(20, min(200, int(self.target_distance_km * 25)))
        max_nodes_to_try = min(len(candidate_nodes) + 1, adaptive_max)
        print(f"    Will try up to {adaptive_max} nodes from {len(candidate_nodes)} total intersections")
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
                distance_km = route_distance / 1000
                target_km = self.target_distance_m / 1000
                min_km = self.min_distance / 1000
                max_km = self.max_distance / 1000
                
                print(f"    Generated route: {distance_km:.2f}km (target: {target_km:.2f}km, range: {min_km:.2f}-{max_km:.2f}km)")
                
                if self.min_distance <= route_distance <= self.max_distance:
                    if cost < best_cost:
                        best_route = route
                        best_cost = cost
                        print(f"  ✅ Found VALID route: {distance_km:.2f}km with {num_nodes} nodes")
                    
                    # Early exit if we found a good solution
                    print(f"  Early exit with good solution")
                    break
                else:
                    if distance_km < min_km:
                        print(f"    ❌ Route too SHORT: {distance_km:.2f}km < {min_km:.2f}km")
                    else:
                        print(f"    ❌ Route too LONG: {distance_km:.2f}km > {max_km:.2f}km")
                    
            except Exception as e:
                print(f"  ⚠️ Error with {num_nodes} nodes: {str(e)[:50]}")
                continue
        
        if best_route is None:
            # Fallback: Use progressively more lenient constraints instead of unconstrained solver
            print(f"  ⚠️ No route found within distance constraints, trying with relaxed tolerance...")
            
            # Try with double tolerance (2x more lenient)
            relaxed_tolerance = min(0.5, self.tolerance * 2)
            relaxed_min = self.target_distance_m * (1 - relaxed_tolerance)
            relaxed_max = self.target_distance_m * (1 + relaxed_tolerance)
            print(f"    Relaxed range: {relaxed_min/1000:.2f}km - {relaxed_max/1000:.2f}km")
            
            # Try with fewer nodes but relaxed constraints
            # With full intersection set and higher limits, we can try more combinations
            fallback_max_nodes = min(40, max(10, int(self.target_distance_km * 6)))
            solver = FastNearestNeighborTSP(self.graph, self.start_node, self.objective)
            # Pass the filtered candidates to avoid using all nodes in fallback
            solver.candidate_nodes = candidate_nodes
            
            for num_nodes in range(5, fallback_max_nodes):
                try:
                    fallback_route, fallback_cost = solver.solve(max_nodes=num_nodes)
                    fallback_stats = self.get_route_stats(fallback_route)
                    fallback_distance = fallback_stats.get('total_distance_m', 0)
                    
                    # Accept if within relaxed constraints  
                    fallback_km = fallback_distance / 1000
                    relaxed_min_km = relaxed_min / 1000
                    relaxed_max_km = relaxed_max / 1000
                    
                    print(f"    Fallback route: {fallback_km:.2f}km (relaxed range: {relaxed_min_km:.2f}-{relaxed_max_km:.2f}km)")
                    
                    if relaxed_min <= fallback_distance <= relaxed_max:
                        best_route = fallback_route
                        best_cost = fallback_cost
                        print(f"  ✅ ACCEPTED relaxed fallback route: {fallback_km:.2f}km with {num_nodes} nodes")
                        break
                    elif fallback_distance < relaxed_min and num_nodes < fallback_max_nodes - 1:
                        print(f"    ❌ Fallback too SHORT, trying more nodes...")
                        continue  # Try more nodes
                    else:
                        print(f"    ❌ Fallback {fallback_km:.2f}km REJECTED (outside relaxed range)")
                except Exception as e:
                    continue
            
            # Final emergency fallback: build a constrained minimal route
            if best_route is None:
                print(f"  🚨 EMERGENCY FALLBACK: All constrained attempts failed!")
                print(f"     Building minimal route with hard distance limit...")
                
                # Hard limit: never return routes longer than 2x target
                absolute_max_distance = self.target_distance_m * 2
                print(f"     Absolute max allowed: {absolute_max_distance/1000:.2f}km")
                
                # Try very small routes with progressively fewer nodes
                for emergency_nodes in range(3, 8):
                    try:
                        emergency_route, emergency_cost = solver.solve(max_nodes=emergency_nodes)
                        emergency_stats = self.get_route_stats(emergency_route)
                        emergency_distance = emergency_stats.get('total_distance_m', 0)
                        emergency_km = emergency_distance / 1000
                        
                        print(f"     Emergency attempt {emergency_nodes} nodes: {emergency_km:.2f}km")
                        
                        # Accept if under absolute maximum
                        if emergency_distance <= absolute_max_distance:
                            best_route = emergency_route
                            best_cost = emergency_cost
                            print(f"  ✅ Emergency route ACCEPTED: {emergency_km:.2f}km with {emergency_nodes} nodes")
                            break
                        else:
                            print(f"     ❌ Emergency route {emergency_km:.2f}km > {absolute_max_distance/1000:.2f}km (REJECTED)")
                    except Exception as e:
                        print(f"     ❌ Error with {emergency_nodes} nodes: {str(e)[:30]}")
                        continue
                
                # If even emergency fallback fails, return a trivial 2-node route
                if best_route is None:
                    print(f"  💀 FINAL FALLBACK: Returning minimal 2-node route")
                    closest_node = min(candidate_nodes, key=lambda n: self.get_distance(self.start_node, n))
                    best_route = [self.start_node, closest_node]
                    best_cost = self.get_distance(self.start_node, closest_node) * 2  # out and back
                    final_stats = self.get_route_stats(best_route)
                    print(f"  💀 Final route: {final_stats.get('total_distance_km', 0):.2f}km")
            
        
        return best_route, best_cost
    
    def _get_nodes_in_radius(self, radius_km: float) -> List[int]:
        """Get nodes within specified radius of start node"""
        from route import get_nodes_within_distance
        return get_nodes_within_distance(self.graph, self.start_node, radius_km)
    
    def _get_all_intersection_nodes(self) -> List[int]:
        """Get intersections using 20m aggressive proximity filtering"""
        import math
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate distance between two points using haversine formula"""
            R = 6371000  # Earth radius in meters
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            return R * c
        
        # Get all potential intersections (non-degree-2 nodes)
        all_intersections = []
        for node_id, node_data in self.graph.nodes(data=True):
            if self.graph.degree(node_id) != 2:
                all_intersections.append({
                    'node_id': node_id,
                    'lat': node_data['y'],
                    'lon': node_data['x'],
                    'highway': node_data.get('highway', 'none')
                })
        
        print(f"    Processing {len(all_intersections):,} potential intersections...")
        
        # Step 1: Find all highway-tagged (real) intersections
        real_intersections = []
        artifacts = []
        
        for node in all_intersections:
            if node['highway'] in ['crossing', 'traffic_signals', 'stop', 'mini_roundabout']:
                real_intersections.append(node)
            else:
                artifacts.append(node)
        
        # Step 2: Remove artifacts within 20m of real intersections
        proximity_threshold_m = 20.0
        artifacts_after_real_filtering = []
        removed_near_real = 0
        
        for artifact in artifacts:
            # Check if this artifact is too close to any real intersection
            too_close_to_real = False
            for real_node in real_intersections:
                distance = haversine_distance(
                    artifact['lat'], artifact['lon'],
                    real_node['lat'], real_node['lon']
                )
                if distance < proximity_threshold_m:
                    too_close_to_real = True
                    break
            
            if too_close_to_real:
                removed_near_real += 1
            else:
                artifacts_after_real_filtering.append(artifact)
        
        # Step 3: Remove artifacts within 20m of other artifacts (clustering removal)
        final_kept_artifacts = []
        removed_by_clustering = 0
        
        for artifact in artifacts_after_real_filtering:
            # Check if this artifact is too close to any already-kept artifact
            too_close_to_kept = False
            for kept_artifact in final_kept_artifacts:
                distance = haversine_distance(
                    artifact['lat'], artifact['lon'],
                    kept_artifact['lat'], kept_artifact['lon']
                )
                if distance < proximity_threshold_m:
                    too_close_to_kept = True
                    break
            
            if too_close_to_kept:
                removed_by_clustering += 1
            else:
                final_kept_artifacts.append(artifact)
        
        # Combine final intersections
        final_intersection_nodes = [node['node_id'] for node in real_intersections] + \
                                  [node['node_id'] for node in final_kept_artifacts]
        
        total_removed = removed_near_real + removed_by_clustering
        
        print(f"    20m aggressive filtering results:")
        print(f"      Real intersections: {len(real_intersections)}")
        print(f"      Kept artifacts: {len(final_kept_artifacts)} (>20m apart)")
        print(f"      Total kept: {len(final_intersection_nodes)}")
        print(f"      Removed: {total_removed} ({(total_removed/len(all_intersections)*100):.1f}%)")
        
        return final_intersection_nodes
    
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
    
    def __init__(self, graph: nx.Graph, filtered_candidates: list = None):
        self.graph = graph
        self.filtered_candidates = filtered_candidates
    
    def find_optimal_route(self, start_node: int, target_distance_km: float, 
                          objective: str = RouteObjective.MINIMIZE_DISTANCE,
                          algorithm: str = "nearest_neighbor") -> Dict:
        """Find optimal running route with fast algorithm"""
        
        print(f"🔧 Using fast TSP solver (no distance matrix precomputation)...")
        
        start_time = time.time()
        
        # Create fast distance-constrained solver
        solver = FastDistanceConstrainedTSP(
            self.graph, start_node, target_distance_km, 
            tolerance=0.2, objective=objective, filtered_candidates=self.filtered_candidates
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