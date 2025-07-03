#!/usr/bin/env python3
"""
Route Optimizer
Handles route optimization with automatic solver selection
"""

import time
from typing import Dict, Any, Optional
import networkx as nx


class RouteOptimizer:
    """Manages route optimization with solver fallbacks"""
    
    def __init__(self, graph: nx.Graph):
        """Initialize route optimizer
        
        Args:
            graph: NetworkX graph for route planning
        """
        self.graph = graph
        self._optimizer_instance = None
        self._solver_type = None
        self._initialize_solver()
    
    def _initialize_solver(self):
        """Initialize the best available TSP solver"""
        try:
            from tsp_solver_fast import FastRunningRouteOptimizer
            from tsp_solver_fast import RouteObjective
            self._optimizer_class = FastRunningRouteOptimizer
            self._route_objective = RouteObjective
            self._solver_type = "fast"
            print("✅ Using fast TSP solver (no distance matrix precomputation)")
        except ImportError:
            try:
                from tsp_solver import RunningRouteOptimizer
                from tsp_solver import RouteObjective
                self._optimizer_class = RunningRouteOptimizer
                self._route_objective = RouteObjective
                self._solver_type = "standard"
                print("⚠️ Using standard TSP solver (with distance matrix)")
            except ImportError:
                raise ImportError("No TSP solver available. Please check tsp_solver modules.")
    
    @property
    def RouteObjective(self):
        """Get the RouteObjective enum for the current solver"""
        return self._route_objective
    
    @property
    def solver_type(self) -> str:
        """Get the type of solver being used"""
        return self._solver_type
    
    def optimize_route(self, start_node: int, target_distance_km: float,
                      objective: str = None, algorithm: str = "nearest_neighbor") -> Optional[Dict[str, Any]]:
        """Generate optimized route
        
        Args:
            start_node: Starting node ID
            target_distance_km: Target route distance in kilometers
            objective: Route objective (from RouteObjective enum)
            algorithm: Algorithm to use ('nearest_neighbor' or 'genetic')
            
        Returns:
            Route result dictionary or None if optimization fails
        """
        if not self.graph:
            print("❌ No graph loaded")
            return None
        
        if start_node not in self.graph.nodes:
            print(f"❌ Invalid start node: {start_node}")
            return None
        
        # Set default objective
        if objective is None:
            objective = self._route_objective.MINIMIZE_DISTANCE
        
        print(f"🚀 Generating optimized route...")
        print(f"   Start node: {start_node}")
        print(f"   Target distance: {target_distance_km:.1f} km")
        print(f"   Objective: {objective}")
        print(f"   Algorithm: {algorithm}")
        print(f"   Solver: {self._solver_type}")
        
        try:
            # Apply intelligent candidate filtering for all algorithms when using standard solver
            # (standard solver builds expensive distance matrix, fast solver computes on-demand)
            if self._solver_type == "standard":
                candidate_nodes = self._get_intersection_nodes()
                
                # Two-stage filtering: straight-line distance first (fast), then road distance (precise)
                # Stage 1: Fast straight-line filtering to reduce candidate pool
                max_straight_line_km = (target_distance_km / 2.0) + 1.5  # Generous straight-line filter
                straight_line_filtered = self._filter_nodes_by_distance(candidate_nodes, start_node, max_straight_line_km)
                
                # Stage 2: Precise road distance filtering on reduced set
                max_road_distance_km = (target_distance_km / 2.0) + 1.0  # +1km tolerance
                filtered_candidates = self._filter_nodes_by_road_distance(straight_line_filtered, start_node, max_road_distance_km)
                
                if start_node not in filtered_candidates:
                    filtered_candidates.append(start_node)
                
                print(f"   Candidate nodes: {len(filtered_candidates)} nodes")
                print(f"   Filtering stages: {len(candidate_nodes)} → {len(straight_line_filtered)} → {len(filtered_candidates)}")
                print(f"   Max road distance: {max_road_distance_km:.1f}km for {target_distance_km:.1f}km target route")
                print(f"   Distance matrix size: {len(filtered_candidates)}² = {len(filtered_candidates)**2:,} calculations")
                
                # Pass filtered candidates to standard solver
                optimizer = self._optimizer_class(self.graph, candidate_nodes=filtered_candidates)
            else:
                # Fast solver: Apply same filtering to reduce search space for all algorithms
                candidate_nodes = self._get_intersection_nodes()
                
                # Two-stage filtering: straight-line distance first (fast), then road distance (precise)
                # Stage 1: Fast straight-line filtering to reduce candidate pool
                max_straight_line_km = (target_distance_km / 2.0) + 1.5  # Generous straight-line filter
                straight_line_filtered = self._filter_nodes_by_distance(candidate_nodes, start_node, max_straight_line_km)
                
                # Stage 2: Precise road distance filtering on reduced set
                max_road_distance_km = (target_distance_km / 2.0) + 1.0  # +1km tolerance
                filtered_candidates = self._filter_nodes_by_road_distance(straight_line_filtered, start_node, max_road_distance_km)
                
                if start_node not in filtered_candidates:
                    filtered_candidates.append(start_node)
                
                print(f"   Candidate nodes: {len(filtered_candidates)} nodes")
                print(f"   Filtering stages: {len(candidate_nodes)} → {len(straight_line_filtered)} → {len(filtered_candidates)}")
                print(f"   Max road distance: {max_road_distance_km:.1f}km for {target_distance_km:.1f}km target route")
                print(f"   Search space reduction: {len(candidate_nodes)/len(filtered_candidates):.1f}x")
                
                # Pass filtered candidates to fast solver for all algorithms
                optimizer = self._optimizer_class(self.graph, filtered_candidates=filtered_candidates)
            
            # Record timing
            start_time = time.time()
            
            # Run optimization
            result = optimizer.find_optimal_route(
                start_node=start_node,
                target_distance_km=target_distance_km,
                objective=objective,
                algorithm=algorithm
            )
            
            solve_time = time.time() - start_time
            
            if result:
                print(f"✅ Route generated in {solve_time:.2f} seconds")
                
                # Add solver metadata
                result['solver_info'] = {
                    'solver_type': self._solver_type,
                    'solve_time': solve_time,
                    'algorithm_used': algorithm,
                    'objective_used': str(objective)
                }
                
                return result
            else:
                print("❌ Route optimization returned no result")
                return None
                
        except Exception as e:
            print(f"❌ Route generation failed: {e}")
            return None
    
    def get_available_objectives(self) -> Dict[str, Any]:
        """Get available route objectives
        
        Returns:
            Dictionary mapping objective names to enum values
        """
        return {
            "Shortest Route": self._route_objective.MINIMIZE_DISTANCE,
            "Maximum Elevation Gain": self._route_objective.MAXIMIZE_ELEVATION,
            "Balanced Route": self._route_objective.BALANCED_ROUTE,
            "Easiest Route": self._route_objective.MINIMIZE_DIFFICULTY
        }
    
    def get_available_algorithms(self) -> list:
        """Get available optimization algorithms
        
        Returns:
            List of algorithm names
        """
        return ["nearest_neighbor", "genetic"]
    
    def validate_parameters(self, start_node: int, target_distance_km: float,
                           objective: str = None, algorithm: str = None) -> Dict[str, Any]:
        """Validate optimization parameters
        
        Args:
            start_node: Starting node ID
            target_distance_km: Target distance
            objective: Route objective
            algorithm: Algorithm name
            
        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []
        
        # Validate start node
        if start_node not in self.graph.nodes:
            errors.append(f"Start node {start_node} not found in graph")
        
        # Validate distance
        if target_distance_km <= 0:
            errors.append("Target distance must be positive")
        elif target_distance_km > 20:
            warnings.append("Target distance > 20km may take very long to optimize")
        
        # Validate algorithm
        if algorithm and algorithm not in self.get_available_algorithms():
            errors.append(f"Unknown algorithm: {algorithm}")
        
        # Validate objective
        if objective and objective not in self.get_available_objectives().values():
            warnings.append(f"Unknown objective: {objective}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _create_filtered_graph(self, start_node: int) -> nx.Graph:
        """Create a filtered graph that preserves connectivity but limits TSP solver nodes
        
        Instead of creating a subgraph (which breaks connectivity), we'll pass the full graph
        but the TSP solver will only consider intersection nodes as candidates.
        
        Args:
            start_node: Starting node ID
            
        Returns:
            Full NetworkX graph (filtering handled in TSP solver)
        """
        # Return full graph - filtering will be handled in TSP solver
        # This preserves connectivity for shortest path calculations
        return self.graph
    
    def _get_intersection_nodes(self) -> list:
        """Get intersection nodes using 20m aggressive proximity filtering
        
        Returns:
            List of intersection node IDs
        """
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
            
            if not too_close_to_real:
                artifacts_after_real_filtering.append(artifact)
        
        # Step 3: Remove artifacts within 20m of other artifacts (clustering removal)
        final_kept_artifacts = []
        
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
            
            if not too_close_to_kept:
                final_kept_artifacts.append(artifact)
        
        # Combine final intersections
        final_intersection_nodes = [node['node_id'] for node in real_intersections] + \
                                  [node['node_id'] for node in final_kept_artifacts]
        
        return final_intersection_nodes
    
    def _filter_nodes_by_distance(self, candidate_nodes: list, start_node: int, max_radius_km: float) -> list:
        """Filter candidate nodes by straight-line distance from start node
        
        Args:
            candidate_nodes: List of candidate node IDs
            start_node: Starting node ID
            max_radius_km: Maximum radius in kilometers
            
        Returns:
            List of filtered node IDs within radius
        """
        import math
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate distance between two points using haversine formula"""
            R = 6371000  # Earth radius in meters
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            return R * c
        
        # Get start node coordinates
        start_data = self.graph.nodes[start_node]
        start_lat, start_lon = start_data['y'], start_data['x']
        max_radius_m = max_radius_km * 1000
        
        # Filter nodes by distance
        filtered_nodes = []
        for node in candidate_nodes:
            node_data = self.graph.nodes[node]
            node_lat, node_lon = node_data['y'], node_data['x']
            distance = haversine_distance(start_lat, start_lon, node_lat, node_lon)
            
            if distance <= max_radius_m:
                filtered_nodes.append(node)
        
        return filtered_nodes
    
    def _filter_nodes_by_road_distance(self, candidate_nodes: list, start_node: int, max_distance_km: float) -> list:
        """Filter candidate nodes by actual road distance from start node
        
        Args:
            candidate_nodes: List of candidate node IDs
            start_node: Starting node ID
            max_distance_km: Maximum road distance in kilometers
            
        Returns:
            List of filtered node IDs within road distance
        """
        import networkx as nx
        
        max_distance_m = max_distance_km * 1000
        filtered_nodes = []
        total_candidates = len(candidate_nodes)
        
        print(f"     Filtering {total_candidates} candidates by road distance...")
        
        # Batch process in chunks to show progress and avoid timeout
        chunk_size = 100
        processed = 0
        
        for i in range(0, len(candidate_nodes), chunk_size):
            chunk = candidate_nodes[i:i + chunk_size]
            
            for node in chunk:
                if node == start_node:
                    filtered_nodes.append(node)
                    continue
                
                try:
                    # Calculate shortest path distance
                    path_length = nx.shortest_path_length(
                        self.graph, start_node, node, weight='length'
                    )
                    
                    if path_length <= max_distance_m:
                        filtered_nodes.append(node)
                        
                except nx.NetworkXNoPath:
                    # Node is not reachable, skip it
                    pass
                except Exception:
                    # Any other error, skip this node
                    pass
            
            processed += len(chunk)
            if processed % 200 == 0 or processed >= total_candidates:
                progress = (processed / total_candidates) * 100
                print(f"     Progress: {processed}/{total_candidates} ({progress:.0f}%) - {len(filtered_nodes)} within range")
        
        print(f"     Road distance filtering: kept {len(filtered_nodes)}/{total_candidates} nodes")
        return filtered_nodes
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get information about the current solver
        
        Returns:
            Dictionary with solver information
        """
        return {
            'solver_type': self._solver_type,
            'solver_class': self._optimizer_class.__name__,
            'available_objectives': list(self.get_available_objectives().keys()),
            'available_algorithms': self.get_available_algorithms(),
            'graph_nodes': len(self.graph.nodes) if self.graph else 0,
            'graph_edges': len(self.graph.edges) if self.graph else 0
        }