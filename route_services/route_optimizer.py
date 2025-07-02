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
            # Create optimizer instance
            optimizer = self._optimizer_class(self.graph)
            
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