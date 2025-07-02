#!/usr/bin/env python3
"""
Refactored Command Line Route Planner
Uses shared route services for consistent functionality
"""

import argparse
import sys
import time
from typing import List, Tuple

from route_services import (
    NetworkManager, RouteOptimizer, RouteAnalyzer, 
    ElevationProfiler, RouteFormatter
)


class RefactoredCLIRoutePlanner:
    """Refactored command line interface using shared route services"""
    
    def __init__(self):
        """Initialize the CLI route planner"""
        self.services = None
        self.selected_start_node = 1529188403  # Default starting point
        
    def initialize_services(self, center_point=None, radius_km=0.8):
        """Initialize all route services
        
        Args:
            center_point: (lat, lon) tuple for network center
            radius_km: Network radius in kilometers
            
        Returns:
            True if services initialized successfully
        """
        try:
            print("🌐 Initializing route planning services...")
            
            # Create network manager and load graph
            network_manager = NetworkManager(center_point)
            graph = network_manager.load_network(radius_km)
            
            if not graph:
                print("❌ Failed to load street network")
                return False
            
            # Create all services
            self.services = {
                'network_manager': network_manager,
                'route_optimizer': RouteOptimizer(graph),
                'route_analyzer': RouteAnalyzer(graph),
                'elevation_profiler': ElevationProfiler(graph),
                'route_formatter': RouteFormatter(),
                'graph': graph
            }
            
            # Display network stats
            stats = network_manager.get_network_stats(graph)
            print(f"✅ Loaded {stats['nodes']} intersections and {stats['edges']} road segments")
            
            # Validate default starting node
            if not network_manager.validate_node_exists(graph, self.selected_start_node):
                print(f"⚠️ Default starting node {self.selected_start_node} not found, will need to select one")
                self.selected_start_node = None
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize services: {e}")
            return False
    
    def list_starting_points(self, num_points=10):
        """List potential starting points
        
        Args:
            num_points: Number of points to display
            
        Returns:
            List of node IDs
        """
        if not self.services:
            print("❌ Services not initialized")
            return []
        
        network_manager = self.services['network_manager']
        graph = self.services['graph']
        center_point = network_manager.center_point
        
        # Get nearby nodes around center point
        nearby_nodes = network_manager.get_nearby_nodes(
            graph, center_point[0], center_point[1], 
            radius_km=0.5, max_nodes=num_points
        )
        
        print(f"\\n📍 Available Starting Points (showing {len(nearby_nodes)}):")
        print("-" * 70)
        print(f"{'#':<3} {'Node ID':<12} {'Latitude':<12} {'Longitude':<12} {'Elevation':<10}")
        print("-" * 70)
        
        node_list = []
        for i, (node_id, distance, data) in enumerate(nearby_nodes):
            lat, lon = data['y'], data['x']
            elevation = data.get('elevation', 0)
            
            print(f"{i+1:>2}. {node_id:<12} {lat:<12.6f} {lon:<12.6f} {elevation:<10.0f}")
            node_list.append(node_id)
        
        print("-" * 70)
        print("💡 Tip: You can enter the option number (1-10) or the full node ID")
        
        return node_list
    
    def select_starting_point(self, num_points=10):
        """Interactive starting point selection
        
        Args:
            num_points: Number of points to display
            
        Returns:
            Selected node ID or None
        """
        if not self.services:
            print("❌ Services not initialized")
            return None
        
        available_nodes = self.list_starting_points(num_points)
        network_manager = self.services['network_manager']
        graph = self.services['graph']
        
        while True:
            try:
                user_input = input(f"\\nSelect starting point (1-{len(available_nodes)} or node ID, 'back' to return): ").strip()
                
                if user_input.lower() in ['back', '']:
                    return None
                
                # Try to parse as integer
                try:
                    input_num = int(user_input)
                except ValueError:
                    print(f"❌ Invalid input: '{user_input}' is not a valid integer")
                    continue
                
                # Check if it's an option number
                if 1 <= input_num <= len(available_nodes):
                    selected_node = available_nodes[input_num - 1]
                    print(f"✅ Selected option {input_num}: Node {selected_node}")
                else:
                    # Treat as direct node ID
                    selected_node = input_num
                    print(f"✅ Selected node ID: {selected_node}")
                
                # Validate the node exists
                if not network_manager.validate_node_exists(graph, selected_node):
                    print(f"❌ Invalid node ID: {selected_node}")
                    continue
                
                # Store selection and show details
                self.selected_start_node = selected_node
                node_info = network_manager.get_node_info(graph, selected_node)
                print(f"📍 Starting point confirmed:")
                print(f"   Node ID: {node_info['node_id']}")
                print(f"   Location: {node_info['latitude']:.6f}, {node_info['longitude']:.6f}")
                print(f"   Elevation: {node_info['elevation']:.0f}m")
                
                return selected_node
                
            except KeyboardInterrupt:
                print("\\n⏹️ Selection cancelled")
                return None
    
    def generate_route(self, start_node, target_distance, objective=None, algorithm="nearest_neighbor"):
        """Generate optimized route using route services
        
        Args:
            start_node: Starting node ID
            target_distance: Target distance in km
            objective: Route objective
            algorithm: Algorithm to use
            
        Returns:
            Route result dictionary or None
        """
        if not self.services:
            print("❌ Services not initialized")
            return None
        
        route_optimizer = self.services['route_optimizer']
        
        # Use default objective if not provided
        if objective is None:
            objective = route_optimizer.RouteObjective.MINIMIZE_DISTANCE
        
        print(f"\\n🚀 Generating optimized route...")
        print(f"   Start: Node {start_node}")
        print(f"   Target distance: {target_distance:.1f}km")
        print(f"   Algorithm: {algorithm}")
        print(f"   Solver: {route_optimizer.solver_type}")
        
        # Generate route
        result = route_optimizer.optimize_route(
            start_node=start_node,
            target_distance_km=target_distance,
            objective=objective,
            algorithm=algorithm
        )
        
        if result:
            solve_time = result.get('solver_info', {}).get('solve_time', 0)
            print(f"✅ Route generated in {solve_time:.2f} seconds")
        
        return result
    
    def display_route_stats(self, route_result):
        """Display route statistics using formatter
        
        Args:
            route_result: Route result from optimizer
        """
        if not self.services or not route_result:
            return
        
        route_analyzer = self.services['route_analyzer']
        route_formatter = self.services['route_formatter']
        
        # Analyze route for difficulty rating
        analysis = route_analyzer.analyze_route(route_result)
        difficulty = route_analyzer.get_route_difficulty_rating(route_result)
        analysis['difficulty'] = difficulty
        
        # Format and display stats
        stats_output = route_formatter.format_route_stats_cli(route_result, analysis)
        print(stats_output)
    
    def generate_directions(self, route_result):
        """Generate and display turn-by-turn directions
        
        Args:
            route_result: Route result from optimizer
        """
        if not self.services or not route_result:
            return
        
        route_analyzer = self.services['route_analyzer']
        route_formatter = self.services['route_formatter']
        
        # Generate directions
        directions = route_analyzer.generate_directions(route_result)
        
        # Format and display
        directions_output = route_formatter.format_directions_cli(directions)
        print(directions_output)
    
    def create_route_visualization(self, route_result, save_file=None):
        """Create route visualization
        
        Args:
            route_result: Route result from optimizer
            save_file: Optional file to save visualization
        """
        if not self.services or not route_result:
            return
        
        elevation_profiler = self.services['elevation_profiler']
        
        print(f"\\n📈 Creating route visualization...")
        
        try:
            # Generate elevation profile data
            profile_data = elevation_profiler.generate_profile_data(route_result)
            
            if not profile_data:
                print("❌ No profile data available")
                return
            
            print(f"   Route has {len(route_result['route'])} nodes")
            print(f"   Total distance: {profile_data.get('total_distance_km', 0):.2f}km")
            
            # Create matplotlib visualization (simplified version)
            try:
                import matplotlib.pyplot as plt
                
                elevations = profile_data['elevations']
                distances_km = profile_data['distances_km']
                coordinates = profile_data['coordinates']
                
                # Create figure with elevation profile
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                
                # Elevation profile
                ax.plot(distances_km, elevations, 'g-', linewidth=2, marker='o', markersize=4)
                ax.fill_between(distances_km, elevations, alpha=0.3, color='green')
                ax.set_xlabel('Distance (km)')
                ax.set_ylabel('Elevation (m)')
                ax.set_title(f'Elevation Profile - {profile_data.get("total_distance_km", 0):.2f}km Route')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                if save_file:
                    plt.savefig(save_file, dpi=150, bbox_inches='tight')
                    print(f"   ✅ Saved visualization to: {save_file}")
                
                plt.show()
                
            except ImportError:
                print("⚠️ Matplotlib not available, cannot create visualization")
            except Exception as viz_error:
                print(f"❌ Visualization failed: {viz_error}")
                
        except Exception as e:
            print(f"❌ Profile generation failed: {e}")


def interactive_mode():
    """Interactive CLI mode using refactored services"""
    planner = RefactoredCLIRoutePlanner()
    
    print("🏃 Welcome to the Refactored Running Route Optimizer CLI!")
    print("=" * 55)
    print("✨ Powered by shared route services")
    
    # Initialize services
    if not planner.initialize_services():
        return
    
    route_optimizer = planner.services['route_optimizer']
    
    while True:
        # Show current starting point if selected
        if planner.selected_start_node:
            network_manager = planner.services['network_manager']
            graph = planner.services['graph']
            
            if network_manager.validate_node_exists(graph, planner.selected_start_node):
                node_info = network_manager.get_node_info(graph, planner.selected_start_node)
                print(f"\\n📍 Current starting point: Node {node_info['node_id']}")
                print(f"   Location: {node_info['latitude']:.6f}, {node_info['longitude']:.6f}")
                print(f"   Elevation: {node_info['elevation']:.0f}m")
        
        print(f"\\n📋 Main Menu:")
        print("1. Select starting point")
        print("2. Generate route" + (" (with selected point)" if planner.selected_start_node else " (manual entry)"))
        print("3. Show solver information")
        print("4. Quit")
        
        try:
            choice = input("\\nSelect option (1-4): ").strip()
            
            if choice == '1':
                planner.select_starting_point()
                
            elif choice == '2':
                # Route generation
                try:
                    # Use pre-selected starting point or manual entry
                    if planner.selected_start_node:
                        start_node = planner.selected_start_node
                        print(f"✅ Using selected starting point: Node {start_node}")
                    else:
                        available_nodes = planner.list_starting_points(10)
                        start_input = input("\\nEnter starting node (option number or node ID): ").strip()
                        
                        try:
                            input_num = int(start_input)
                            if 1 <= input_num <= len(available_nodes):
                                start_node = available_nodes[input_num - 1]
                            else:
                                start_node = input_num
                        except ValueError:
                            print("❌ Invalid input")
                            continue
                        
                        # Validate node
                        network_manager = planner.services['network_manager']
                        if not network_manager.validate_node_exists(planner.services['graph'], start_node):
                            print(f"❌ Invalid node: {start_node}")
                            continue
                    
                    # Get target distance
                    distance_input = input("Enter target distance (km) [5.0]: ").strip()
                    try:
                        target_distance = float(distance_input) if distance_input else 5.0
                        if target_distance <= 0:
                            print("❌ Distance must be positive")
                            continue
                    except ValueError:
                        print(f"❌ Invalid distance: '{distance_input}'")
                        continue
                    
                    # Get route objective
                    print("\\nObjective options:")
                    objectives = route_optimizer.get_available_objectives()
                    obj_list = list(objectives.items())
                    
                    for i, (name, _) in enumerate(obj_list, 1):
                        print(f"{i}. {name}")
                    
                    obj_input = input("Select objective (1-4) [1]: ").strip()
                    try:
                        obj_choice = int(obj_input) if obj_input else 1
                        if 1 <= obj_choice <= len(obj_list):
                            objective = obj_list[obj_choice - 1][1]
                        else:
                            objective = obj_list[0][1]  # Default
                    except ValueError:
                        objective = obj_list[0][1]  # Default
                    
                    # Get algorithm
                    algorithms = route_optimizer.get_available_algorithms()
                    algorithm_input = input(f"Algorithm ({'/'.join(algorithms)}) [nearest_neighbor]: ").strip()
                    algorithm = algorithm_input if algorithm_input in algorithms else "nearest_neighbor"
                    
                    # Generate route
                    result = planner.generate_route(start_node, target_distance, objective, algorithm)
                    
                    if result:
                        planner.display_route_stats(result)
                        
                        # Ask for directions
                        directions_input = input("\\nShow turn-by-turn directions? (y/n): ").strip().lower()
                        if directions_input in ['y', 'yes']:
                            planner.generate_directions(result)
                        
                        # Ask for visualization
                        viz_input = input("\\nCreate route visualization? (y/n): ").strip().lower()
                        if viz_input in ['y', 'yes']:
                            save_file = f"route_{start_node}_{target_distance}km.png"
                            planner.create_route_visualization(result, save_file)
                
                except KeyboardInterrupt:
                    print("\\n⏹️ Operation cancelled")
                
            elif choice == '3':
                # Show solver information
                solver_info = route_optimizer.get_solver_info()
                print("\\n🔧 Solver Information:")
                print(f"   Type: {solver_info['solver_type']}")
                print(f"   Class: {solver_info['solver_class']}")
                print(f"   Graph nodes: {solver_info['graph_nodes']}")
                print(f"   Graph edges: {solver_info['graph_edges']}")
                print(f"   Available algorithms: {', '.join(solver_info['available_algorithms'])}")
                
            elif choice == '4':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice")
                
        except KeyboardInterrupt:
            print("\\n👋 Goodbye!")
            break
        except EOFError:
            print("\\n👋 Goodbye!")
            break


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Refactored Running Route Optimizer - Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Start interactive mode'
    )
    
    parser.add_argument(
        '--start-node', '-s',
        type=int,
        help='Starting node ID'
    )
    
    parser.add_argument(
        '--distance', '-d',
        type=float,
        help='Target distance in km'
    )
    
    parser.add_argument(
        '--objective', '-o',
        choices=['distance', 'elevation', 'balanced', 'difficulty'],
        default='distance',
        help='Route objective'
    )
    
    args = parser.parse_args()
    
    if args.interactive or (not args.start_node and not args.distance):
        interactive_mode()
    else:
        # Command line mode
        planner = RefactoredCLIRoutePlanner()
        
        if not planner.initialize_services():
            sys.exit(1)
        
        route_optimizer = planner.services['route_optimizer']
        
        # Map objective
        obj_map = {
            'distance': route_optimizer.RouteObjective.MINIMIZE_DISTANCE,
            'elevation': route_optimizer.RouteObjective.MAXIMIZE_ELEVATION,
            'balanced': route_optimizer.RouteObjective.BALANCED_ROUTE,
            'difficulty': route_optimizer.RouteObjective.MINIMIZE_DIFFICULTY
        }
        
        result = planner.generate_route(
            args.start_node, args.distance, 
            obj_map[args.objective]
        )
        
        if result:
            planner.display_route_stats(result)
            planner.generate_directions(result)


if __name__ == "__main__":
    main()