#!/usr/bin/env python3
"""
Route Optimizer with Enhanced 3DEP Elevation Integration
Handles route optimization with automatic solver selection and 3DEP precision elevation
"""

import time
import os
import sys
from typing import Dict, Any, Optional

# Add project root to path for elevation imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx

# GA imports
try:
    from genetic_algorithm import GeneticRouteOptimizer, FitnessObjective
    from genetic_algorithm.optimizer import GAConfig
    GA_AVAILABLE = True
except ImportError:
    GA_AVAILABLE = False

# Enhanced elevation imports
try:
    from elevation_data_sources import get_elevation_manager, ElevationDataManager
    ENHANCED_ELEVATION_AVAILABLE = True
except ImportError:
    ENHANCED_ELEVATION_AVAILABLE = False


class RouteOptimizer:
    """Manages route optimization with solver fallbacks"""
    
    def __init__(self, graph: nx.Graph, elevation_config_path: Optional[str] = None, verbose: bool = True):
        """Initialize route optimizer with enhanced elevation support
        
        Args:
            graph: NetworkX graph for route planning
            elevation_config_path: Optional path to elevation configuration file
            verbose: Whether to show initialization messages
        """
        self.graph = graph
        self.verbose = verbose
        self._optimizer_instance = None
        self._solver_type = None
        self._ga_optimizer = None
        self._elevation_manager = None
        self._elevation_source = None
        
        # Initialize enhanced elevation support
        self._initialize_elevation(elevation_config_path)
        self._initialize_solver()
    
    def _initialize_elevation(self, elevation_config_path: Optional[str] = None):
        """Initialize enhanced elevation data sources
        
        Args:
            elevation_config_path: Optional path to elevation configuration file
        """
        if ENHANCED_ELEVATION_AVAILABLE:
            try:
                self._elevation_manager = get_elevation_manager(elevation_config_path)
                self._elevation_source = self._elevation_manager.get_elevation_source()
                
                if self._elevation_source and self.verbose:
                    source_info = self._elevation_source.get_source_info()
                    print(f"📊 Enhanced elevation system initialized:")
                    print(f"   Source: {source_info.get('type', 'Unknown')}")
                    print(f"   Resolution: {self._elevation_source.get_resolution()}m")
                    
                    # Show hybrid source usage if applicable
                    if hasattr(self._elevation_source, 'get_stats'):
                        stats = self._elevation_source.get_stats()
                        if stats and 'primary_percentage' in stats:
                            print(f"   High-resolution coverage: Available for precision optimization")
                elif not self._elevation_source and self.verbose:
                    print("⚠️ No elevation sources configured")
                    
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Enhanced elevation initialization failed: {e}")
                self._elevation_manager = None
                self._elevation_source = None
        else:
            if self.verbose:
                print("⚠️ Enhanced elevation system not available")
    
    def _initialize_solver(self):
        """Initialize the genetic algorithm optimizer"""
        # Import RouteObjective from new module
        from route_objective import RouteObjective
        self._route_objective = RouteObjective
        self._solver_type = "genetic"
        
        # Initialize GA optimizer
        if GA_AVAILABLE:
            self._ga_optimizer = GeneticRouteOptimizer(self.graph)
            if self.verbose:
                print("✅ Using Genetic Algorithm optimizer")
        else:
            raise ImportError("Genetic Algorithm optimizer not available. Please check GA modules.")
    
    @property
    def RouteObjective(self):
        """Get the RouteObjective enum for the current solver"""
        return self._route_objective
    
    @property
    def solver_type(self) -> str:
        """Get the type of solver being used"""
        return self._solver_type
    
    def optimize_route(self, start_node: int, target_distance_km: float,
                      objective: str = None, algorithm: str = "genetic", 
                      exclude_footways: bool = True, 
                      allow_bidirectional_segments: bool = True) -> Optional[Dict[str, Any]]:
        """Generate optimized route
        
        Args:
            start_node: Starting node ID
            target_distance_km: Target route distance in kilometers
            objective: Route objective (from RouteObjective enum)
            algorithm: Algorithm to use ('genetic')
            exclude_footways: Whether to exclude footway segments (default True)
            allow_bidirectional_segments: Whether to allow segments to be used in both directions (default True)
            
        Returns:
            Route result dictionary or None if optimization fails
        """
        if not self.graph:
            print("❌ No graph loaded")
            return None
        
        if start_node not in self.graph.nodes:
            print(f"❌ Invalid start node: {start_node}")
            return None
        
        if target_distance_km <= 0:
            print(f"❌ Invalid target distance: {target_distance_km}")
            return None
        
        # Set default objective
        if objective is None:
            objective = self._route_objective.MINIMIZE_DISTANCE
        
        # Algorithm selection logic
        selected_algorithm = self._select_algorithm(algorithm, objective)
        
        print(f"🚀 Generating optimized route...")
        print(f"   Start node: {start_node}")
        print(f"   Target distance: {target_distance_km:.1f} km")
        print(f"   Objective: {objective}")
        print(f"   Algorithm: {selected_algorithm}")
        print(f"   Solver: {self._solver_type}")
        print(f"   Exclude footways: {exclude_footways}")
        print(f"   Allow bidirectional segments: {allow_bidirectional_segments}")
        
        try:
            # Use genetic algorithm optimization
            return self._optimize_genetic(start_node, target_distance_km, objective, exclude_footways, allow_bidirectional_segments)
                
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
        algorithms = []
        if GA_AVAILABLE:
            algorithms.append("genetic")
        return algorithms
    
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
    
    def _select_algorithm(self, algorithm: str, objective: str) -> str:
        """Select optimal algorithm based on parameters
        
        Args:
            algorithm: Requested algorithm ('genetic')
            objective: Route objective
            
        Returns:
            Selected algorithm name (always 'genetic')
        """
        if not GA_AVAILABLE:
            raise ImportError("Genetic Algorithm optimizer not available")
        
        # Always use genetic algorithm
        return "genetic"
    
    def _optimize_genetic(self, start_node: int, target_distance_km: float, objective: str, exclude_footways: bool = True, allow_bidirectional_segments: bool = True) -> Optional[Dict[str, Any]]:
        """Optimize route using genetic algorithm
        
        Args:
            start_node: Starting node ID
            target_distance_km: Target route distance in kilometers
            objective: Route objective
            exclude_footways: Whether to exclude footway segments
            allow_bidirectional_segments: Whether to allow segments to be used in both directions
            
        Returns:
            Route result dictionary or None if optimization fails
        """
        if not self._ga_optimizer:
            if self.verbose:
                print("❌ GA optimizer not available")
            return None
        
        # Filter graph if needed
        working_graph = self._filter_graph_for_routing(exclude_footways) if exclude_footways else self.graph
        
        # Create GA optimizer with appropriate configuration
        from genetic_route_optimizer import GeneticRouteOptimizer, GAConfig
        config = GAConfig(allow_bidirectional_segments=allow_bidirectional_segments)
        
        # Create filtered GA optimizer if needed
        if exclude_footways and working_graph != self.graph:
            ga_optimizer = GeneticRouteOptimizer(working_graph, config)
        else:
            # Create new optimizer with updated config for this request
            ga_optimizer = GeneticRouteOptimizer(self.graph, config)
        
        # Convert TSP objective to GA objective
        ga_objective = self._convert_tsp_to_ga_objective(objective)
        
        # Record timing
        start_time = time.time()
        
        # Run GA optimization
        ga_results = ga_optimizer.optimize_route(
            start_node=start_node,
            distance_km=target_distance_km,
            objective=ga_objective
        )
        
        solve_time = time.time() - start_time
        
        if ga_results and ga_results.best_chromosome:
            if self.verbose:
                print(f"✅ GA route generated in {solve_time:.2f} seconds")
            
            # Convert GA results to standard format
            result = self._convert_ga_results_to_standard(ga_results, objective)
            
            # Add solver metadata
            result['solver_info'] = {
                'solver_type': 'genetic',
                'solve_time': solve_time,
                'algorithm_used': 'genetic',
                'objective_used': str(objective),
                'ga_generations': ga_results.total_generations,
                'ga_convergence': ga_results.convergence_reason
            }
            
            return result
        else:
            if self.verbose:
                print("❌ GA optimization returned no result")
            return None
    
    
    def _convert_tsp_to_ga_objective(self, tsp_objective: str) -> str:
        """Convert TSP objective to GA objective string
        
        Args:
            tsp_objective: TSP RouteObjective enum value
            
        Returns:
            GA objective string
        """
        if tsp_objective == self._route_objective.MAXIMIZE_ELEVATION:
            return "elevation"
        elif tsp_objective == self._route_objective.MINIMIZE_DISTANCE:
            return "distance"
        elif tsp_objective == self._route_objective.BALANCED_ROUTE:
            return "balanced"
        elif tsp_objective == self._route_objective.MINIMIZE_DIFFICULTY:
            return "efficiency"
        else:
            return "elevation"  # default to elevation for GA
    
    def _convert_ga_results_to_standard(self, ga_results, objective: str) -> Dict[str, Any]:
        """Convert GA results to standard route format
        
        Args:
            ga_results: GAResults object
            objective: Original objective string
            
        Returns:
            Standard route result dictionary
        """
        best_chromosome = ga_results.best_chromosome
        
        # Get basic route information (use route nodes instead of complete path for better connectivity)
        route_nodes = best_chromosome.get_route_nodes()
        total_distance = best_chromosome.get_total_distance()
        total_elevation_gain = best_chromosome.get_total_elevation_gain()
        
        # Ensure route forms a complete loop by returning to start
        if route_nodes and len(best_chromosome.segments) > 0:
            start_node = best_chromosome.segments[0].start_node
            if route_nodes[-1] != start_node:
                # Only add start node if there's a direct edge, otherwise leave incomplete
                if self.graph.has_edge(route_nodes[-1], start_node):
                    route_nodes.append(start_node)
                # Note: If no direct edge exists, the route remains incomplete
                # This is better than adding invalid edges that break test validation
        
        # Create standard result format
        result = {
            'route': route_nodes,
            'total_distance_km': total_distance / 1000.0,
            'total_distance_m': total_distance,
            'total_elevation_gain_m': total_elevation_gain,
            'fitness_score': ga_results.best_fitness,
            'stats': {
                'total_distance_km': total_distance / 1000.0,
                'total_distance_m': total_distance,
                'total_elevation_gain_m': total_elevation_gain,
                'fitness_score': ga_results.best_fitness,
                'num_nodes': len(route_nodes),
                'route_type': 'genetic_algorithm',
                'objective': objective
            },
            'ga_stats': ga_results.stats
        }
        
        return result
    
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
    
    def _get_intersection_nodes(self, graph: nx.Graph = None) -> list:
        """Get intersection nodes using 20m aggressive proximity filtering
        
        Args:
            graph: Graph to analyze (uses self.graph if None)
            
        Returns:
            List of intersection node IDs
        """
        import math
        
        if graph is None:
            graph = self.graph
        
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
        for node_id, node_data in graph.nodes(data=True):
            if graph.degree(node_id) != 2:
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
    
    def _filter_nodes_by_distance(self, candidate_nodes: list, start_node: int, max_radius_km: float, graph: nx.Graph = None) -> list:
        """Filter candidate nodes by straight-line distance from start node
        
        Args:
            candidate_nodes: List of candidate node IDs
            start_node: Starting node ID
            max_radius_km: Maximum radius in kilometers
            graph: Graph to use (uses self.graph if None)
            
        Returns:
            List of filtered node IDs within radius
        """
        import math
        
        if graph is None:
            graph = self.graph
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate distance between two points using haversine formula"""
            R = 6371000  # Earth radius in meters
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            return R * c
        
        # Get start node coordinates
        start_data = graph.nodes[start_node]
        start_lat, start_lon = start_data['y'], start_data['x']
        max_radius_m = max_radius_km * 1000
        
        # Filter nodes by distance
        filtered_nodes = []
        for node in candidate_nodes:
            node_data = graph.nodes[node]
            node_lat, node_lon = node_data['y'], node_data['x']
            distance = haversine_distance(start_lat, start_lon, node_lat, node_lon)
            
            if distance <= max_radius_m:
                filtered_nodes.append(node)
        
        return filtered_nodes
    
    def _filter_nodes_by_road_distance(self, candidate_nodes: list, start_node: int, max_distance_km: float, graph: nx.Graph = None) -> list:
        """Filter candidate nodes by actual road distance from start node
        
        Args:
            candidate_nodes: List of candidate node IDs
            start_node: Starting node ID
            max_distance_km: Maximum road distance in kilometers
            graph: Graph to use (uses self.graph if None)
            
        Returns:
            List of filtered node IDs within road distance
        """
        import networkx as nx
        
        if graph is None:
            graph = self.graph
        
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
                        graph, start_node, node, weight='length'
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
        info = {
            'solver_type': self._solver_type,
            'solver_class': self._optimizer_instance.__class__.__name__ if self._optimizer_instance else 'Unknown',
            'available_objectives': list(self.get_available_objectives().keys()),
            'available_algorithms': self.get_available_algorithms(),
            'graph_nodes': len(self.graph.nodes) if self.graph else 0,
            'graph_edges': len(self.graph.edges) if self.graph else 0,
            'ga_available': GA_AVAILABLE
        }
        
        if GA_AVAILABLE:
            info['ga_optimizer'] = self._ga_optimizer.__class__.__name__
        
        return info
    
    def _filter_graph_for_routing(self, exclude_footways: bool) -> nx.Graph:
        """Create filtered graph for routing optimization
        
        Args:
            exclude_footways: Whether to exclude footway segments
            
        Returns:
            Filtered NetworkX graph
        """
        if not exclude_footways:
            return self.graph
        
        # Create a copy of the graph
        filtered_graph = self.graph.copy()
        
        # Count removals for reporting
        edges_removed = 0
        total_footways = 0
        
        # Remove footway edges
        edges_to_remove = []
        for u, v, data in filtered_graph.edges(data=True):
            highway = data.get('highway', '')
            if highway == 'footway':
                total_footways += 1
                edges_to_remove.append((u, v))
                edges_removed += 1
        
        # Remove the edges
        filtered_graph.remove_edges_from(edges_to_remove)
        
        # Remove isolated nodes (nodes with no connections after edge removal)
        isolated_nodes = [node for node in filtered_graph.nodes() if filtered_graph.degree(node) == 0]
        filtered_graph.remove_nodes_from(isolated_nodes)
        
        if edges_removed > 0:
            print(f"   Footway filtering: removed {edges_removed}/{total_footways} footway edges, {len(isolated_nodes)} isolated nodes")
        
        return filtered_graph