#!/usr/bin/env python3
"""
Command Line Route Planner
Interactive CLI for generating optimized running routes
"""

import argparse
import sys
import time
from typing import List, Tuple
import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt
from route import add_elevation_to_graph, add_elevation_to_edges, add_running_weights, haversine_distance
from tsp_solver import RouteObjective
try:
    from tsp_solver_fast import FastRunningRouteOptimizer as RunningRouteOptimizer
    print("✅ Using fast TSP solver (no distance matrix precomputation)")
except ImportError:
    from tsp_solver import RunningRouteOptimizer
    print("⚠️ Using standard TSP solver (with distance matrix)")

class CLIRoutePlanner:
    """Command line interface for route planning"""
    
    def __init__(self):
        self.graph = None
        self.center_point = (37.1299, -80.4094)  # Christiansburg, VA
        self.selected_start_node = None  # Store selected starting point
        
    def load_network(self, radius_km=0.8):
        """Load street network with elevation data (cached)"""
        print("🌐 Loading street network and elevation data...")
        print(f"   Area: {radius_km:.1f}km radius around Christiansburg, VA")
        
        try:
            # Use cached graph loader
            from graph_cache import load_or_generate_graph
            
            self.graph = load_or_generate_graph(
                center_point=self.center_point,
                radius_m=int(radius_km * 1000),
                network_type='all'
            )
            
            if self.graph:
                print(f"✅ Loaded {len(self.graph.nodes)} intersections and {len(self.graph.edges)} road segments")
                return True
            else:
                print("❌ Failed to load network")
                return False
            
        except Exception as e:
            print(f"❌ Failed to load network: {e}")
            return False
    
    def list_starting_points(self, num_points=10):
        """List potential starting points with elevations"""
        if not self.graph:
            print("❌ Network not loaded")
            return []
        
        print(f"\n📍 Available Starting Points (showing {num_points}):")
        print("-" * 70)
        print(f"{'Node ID':<12} {'Latitude':<12} {'Longitude':<12} {'Elevation':<10}")
        print("-" * 70)
        
        # Get a sample of nodes with good connectivity
        good_nodes = []
        for node_id, data in self.graph.nodes(data=True):
            degree = self.graph.degree(node_id)
            if degree >= 2:  # At least 2 connections
                good_nodes.append((node_id, data))
        
        # Sort by proximity to center and take first num_points
        good_nodes.sort(key=lambda x: haversine_distance(
            self.center_point[0], self.center_point[1],
            x[1]['y'], x[1]['x']
        ))
        
        selected_nodes = []
        for i, (node_id, data) in enumerate(good_nodes[:num_points]):
            lat, lon = data['y'], data['x']
            elevation = data.get('elevation', 0)
            
            print(f"{i+1:>2}. {node_id:<12} {lat:<12.6f} {lon:<12.6f} {elevation:<10.0f}")
            selected_nodes.append(node_id)
        
        print("-" * 70)
        print("💡 Tip: You can enter the node ID directly, or use option numbers (1-10)")
        return selected_nodes
    
    def select_starting_point(self, num_points=10):
        """Interactive starting point selection"""
        if not self.graph:
            print("❌ Network not loaded")
            return None
        
        # Display available starting points
        available_nodes = self.list_starting_points(num_points)
        
        while True:
            try:
                start_node_input = input(f"\nSelect starting point (1-{len(available_nodes)} or node ID, or 'back' to return): ").strip().lower()
                
                if start_node_input == 'back' or start_node_input == '':
                    return None
                
                # Try to parse as integer
                try:
                    input_num = int(start_node_input)
                except ValueError:
                    print(f"❌ Invalid input: '{start_node_input}' is not a valid integer")
                    print("   Please enter a number, node ID, or 'back'")
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
                if selected_node not in self.graph.nodes:
                    print(f"❌ Invalid node ID: {selected_node}")
                    print(f"   Graph has {len(self.graph.nodes)} nodes")
                    continue
                
                # Store the selection and show details
                self.selected_start_node = selected_node
                node_data = self.graph.nodes[selected_node]
                print(f"📍 Starting point confirmed:")
                print(f"   Node ID: {selected_node}")
                print(f"   Location: {node_data['y']:.6f}, {node_data['x']:.6f}")
                print(f"   Elevation: {node_data.get('elevation', 0):.0f}m")
                
                return selected_node
                
            except KeyboardInterrupt:
                print("\n⏹️ Selection cancelled")
                return None
    
    def find_start_by_location(self, lat=None, lon=None):
        """Find starting point near given coordinates"""
        if not self.graph:
            print("❌ Network not loaded")
            return None
        
        if lat is None or lon is None:
            # Use center point
            lat, lon = self.center_point
            print(f"Using center point: {lat:.6f}, {lon:.6f}")
        
        # Find nearest node
        nearest_node = None
        min_distance = float('inf')
        
        for node_id, data in self.graph.nodes(data=True):
            distance = haversine_distance(lat, lon, data['y'], data['x'])
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id
        
        if nearest_node:
            node_data = self.graph.nodes[nearest_node]
            print(f"📍 Found nearest intersection:")
            print(f"   Node ID: {nearest_node}")
            print(f"   Location: {node_data['y']:.6f}, {node_data['x']:.6f}")
            print(f"   Elevation: {node_data.get('elevation', 0):.0f}m")
            print(f"   Distance from target: {min_distance:.0f}m")
            
            # Ask if user wants to use this as starting point
            use_point = input("\nUse this as starting point? (y/n): ").strip().lower()
            if use_point in ['y', 'yes']:
                self.selected_start_node = nearest_node
                print(f"✅ Starting point set to Node {nearest_node}")
            
        return nearest_node
    
    def generate_route(self, start_node, target_distance, objective, algorithm="nearest_neighbor"):
        """Generate optimized route"""
        if not self.graph:
            print("❌ Network not loaded")
            return None
        
        print(f"\n🚀 Generating optimized route...")
        print(f"   Start: Node {start_node}")
        print(f"   Target distance: {target_distance:.1f}km")
        print(f"   Objective: {objective}")
        print(f"   Algorithm: {algorithm}")
        
        try:
            optimizer = RunningRouteOptimizer(self.graph)
            
            start_time = time.time()
            result = optimizer.find_optimal_route(
                start_node=start_node,
                target_distance_km=target_distance,
                objective=objective,
                algorithm=algorithm
            )
            solve_time = time.time() - start_time
            
            print(f"✅ Route generated in {solve_time:.2f} seconds")
            return result
            
        except Exception as e:
            print(f"❌ Route generation failed: {e}")
            return None
    
    def display_route_stats(self, result):
        """Display detailed route statistics"""
        if not result:
            return
        
        stats = result['stats']
        route = result['route']
        
        print(f"\n📊 Route Statistics:")
        print("=" * 50)
        print(f"Distance:        {stats.get('total_distance_km', 0):.2f} km")
        print(f"Elevation Gain:  {stats.get('total_elevation_gain_m', 0):.0f} m")
        print(f"Elevation Loss:  {stats.get('total_elevation_loss_m', 0):.0f} m")
        print(f"Net Elevation:   {stats.get('net_elevation_gain_m', 0):+.0f} m")
        print(f"Max Grade:       {stats.get('max_grade_percent', 0):.1f}%")
        print(f"Est. Time:       {stats.get('estimated_time_min', 0):.0f} minutes")
        print(f"Route Points:    {len(route)} intersections")
        print(f"Algorithm:       {result.get('algorithm', 'Unknown')}")
        print(f"Objective:       {result.get('objective', 'Unknown')}")
        print("=" * 50)
    
    def generate_directions(self, result):
        """Generate and display turn-by-turn directions"""
        if not result or not result.get('route'):
            return
        
        route = result['route']
        
        print(f"\n📋 Turn-by-Turn Directions:")
        print("=" * 60)
        
        cumulative_distance = 0
        
        # Start
        if route and route[0] in self.graph.nodes:
            start_data = self.graph.nodes[route[0]]
            print(f"1. Start at intersection (Node {route[0]})")
            print(f"   Elevation: {start_data.get('elevation', 0):.0f}m")
            print(f"   Distance: 0.0 km")
            print()
        
        # Route segments
        for i in range(1, len(route)):
            if route[i] in self.graph.nodes and route[i-1] in self.graph.nodes:
                curr_data = self.graph.nodes[route[i]]
                prev_data = self.graph.nodes[route[i-1]]
                
                # Calculate segment distance
                segment_dist = haversine_distance(
                    prev_data['y'], prev_data['x'],
                    curr_data['y'], curr_data['x']
                )
                cumulative_distance += segment_dist
                
                # Elevation change
                elevation_change = curr_data.get('elevation', 0) - prev_data.get('elevation', 0)
                if elevation_change > 5:
                    terrain = "⬆️ uphill"
                elif elevation_change < -5:
                    terrain = "⬇️ downhill"
                else:
                    terrain = "➡️ level"
                
                print(f"{i+1}. Continue to intersection (Node {route[i]}) - {terrain}")
                print(f"   Elevation: {curr_data.get('elevation', 0):.0f}m ({elevation_change:+.0f}m)")
                print(f"   Distance: {cumulative_distance/1000:.2f} km")
                print()
        
        # Return to start
        if len(route) > 1:
            print(f"{len(route)+1}. Return to starting point to complete the loop")
            print(f"   Total distance: {result['stats'].get('total_distance_km', 0):.2f} km")
        
        print("=" * 60)
    
    def create_route_visualization(self, result, save_file=None):
        """Create route visualization"""
        if not result or not result.get('route'):
            return
        
        route = result['route']
        
        print(f"\n📈 Creating route visualization...")
        print(f"   Route has {len(route)} nodes")
        
        # Get route coordinates and elevations
        lats, lons, elevations = [], [], []
        distances = [0]
        cumulative_distance = 0
        
        for i, node in enumerate(route):
            if node in self.graph.nodes:
                data = self.graph.nodes[node]
                lats.append(data['y'])
                lons.append(data['x'])
                elevations.append(data.get('elevation', 0))
                
                if i > 0:
                    prev_data = self.graph.nodes[route[i-1]]
                    segment_dist = haversine_distance(
                        prev_data['y'], prev_data['x'],
                        data['y'], data['x']
                    )
                    cumulative_distance += segment_dist
                    distances.append(cumulative_distance)
        
        # Add return to start for distance calculation only
        if len(route) > 1:
            start_data = self.graph.nodes[route[0]]
            end_data = self.graph.nodes[route[-1]]
            return_dist = haversine_distance(
                end_data['y'], end_data['x'],
                start_data['y'], start_data['x']
            )
            cumulative_distance += return_dist
            distances.append(cumulative_distance)
            # Don't append elevation here - it will create array length mismatch
        
        # Debug array lengths
        print(f"   Coordinates: {len(lats)} lats, {len(lons)} lons, {len(elevations)} elevations")
        print(f"   Distances: {len(distances)} points")
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Map view - ensure all arrays have same length
        if len(lons) == len(lats) == len(elevations):
            ax1.scatter(lons, lats, c=elevations, cmap='terrain', s=50, alpha=0.8)
            print("   ✅ Using elevation coloring for route points")
        else:
            # Fallback without elevation coloring if arrays don't match
            ax1.scatter(lons, lats, c='blue', s=50, alpha=0.8)
            print(f"   ⚠️ Array mismatch - using fallback coloring")
        ax1.plot(lons + [lons[0]], lats + [lats[0]], 'r-', linewidth=2, alpha=0.7)
        ax1.scatter([lons[0]], [lats[0]], c='green', s=100, marker='*', label='Start/Finish')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Route Map with Elevation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Elevation profile - handle array length mismatch
        distances_km = [d / 1000 for d in distances]
        
        # Add start elevation again for the return segment to close the loop
        elevations_with_return = elevations + [elevations[0]] if elevations else []
        
        # Ensure distances and elevations have same length
        print(f"   Elevation profile: {len(distances_km)} distances, {len(elevations_with_return)} elevations")
        
        if len(distances_km) == len(elevations_with_return):
            ax2.plot(distances_km, elevations_with_return, 'g-', linewidth=2, marker='o')
            ax2.fill_between(distances_km, elevations_with_return, alpha=0.3, color='green')
            ax2.set_xlabel('Distance (km)')
            print("   ✅ Using distance-based elevation profile")
        else:
            # Fallback: just plot elevations vs route points
            route_points = list(range(len(elevations)))
            ax2.plot(route_points, elevations, 'g-', linewidth=2, marker='o')
            ax2.fill_between(route_points, elevations, alpha=0.3, color='green')
            ax2.set_xlabel('Route Point')
            print("   ⚠️ Using fallback route-point-based profile")
        
        ax2.set_ylabel('Elevation (m)')
        ax2.set_title('Elevation Profile')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_file:
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            print(f"   Saved visualization to: {save_file}")
        
        plt.show()

def interactive_mode():
    """Interactive CLI mode"""
    planner = CLIRoutePlanner()
    
    print("🏃 Welcome to the Running Route Optimizer CLI!")
    print("=" * 50)
    
    # Load network
    if not planner.load_network():
        return
    
    while True:
        # Show current starting point if selected
        if planner.selected_start_node:
            node_data = planner.graph.nodes[planner.selected_start_node]
            print(f"\n📍 Current starting point: Node {planner.selected_start_node}")
            print(f"   Location: {node_data['y']:.6f}, {node_data['x']:.6f}")
            print(f"   Elevation: {node_data.get('elevation', 0):.0f}m")
        
        print(f"\n📋 Main Menu:")
        print("1. Select starting point")
        print("2. Find starting point by location")
        print("3. Generate route" + (" (with selected point)" if planner.selected_start_node else " (manual entry)"))
        print("4. Quit")
        
        try:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                planner.select_starting_point()
                
            elif choice == '2':
                try:
                    lat = float(input("Enter latitude (or press Enter for center): ").strip() or planner.center_point[0])
                    lon = float(input("Enter longitude (or press Enter for center): ").strip() or planner.center_point[1])
                    planner.find_start_by_location(lat, lon)
                except ValueError:
                    print("❌ Invalid coordinates")
                
            elif choice == '3':
                # Get route parameters
                try:
                    # Use pre-selected starting point if available
                    if planner.selected_start_node:
                        start_node = planner.selected_start_node
                        print(f"✅ Using selected starting point: Node {start_node}")
                    else:
                        # Manual starting point selection
                        print("\n📍 Available starting points:")
                        available_nodes = planner.list_starting_points(10)
                        
                        start_node_input = input("\nEnter starting node ID (or option number 1-10): ").strip()
                        
                        # Validate input is not empty
                        if not start_node_input:
                            print("❌ Empty input, please enter a node ID or option number")
                            continue
                        
                        # Try to parse as integer
                        try:
                            input_num = int(start_node_input)
                        except ValueError:
                            print(f"❌ Invalid input: '{start_node_input}' is not a valid integer")
                            print("   Please enter a numeric node ID or option number (1-10)")
                            continue
                        
                        # Check if it's an option number (1-10)
                        if 1 <= input_num <= len(available_nodes):
                            start_node = available_nodes[input_num - 1]  # Convert to 0-based index
                            print(f"✅ Selected option {input_num}: Node {start_node}")
                        else:
                            # Treat as direct node ID
                            start_node = input_num
                        
                        # Validate the final node ID
                        if start_node not in planner.graph.nodes:
                            print(f"❌ Invalid node ID: {start_node}")
                            print(f"   Input received: '{start_node_input}'")
                            print(f"   Interpreted as: {start_node} (type: {type(start_node).__name__})")
                            print(f"   Graph has {len(planner.graph.nodes)} nodes")
                            
                            # Show some nearby valid nodes
                            sample_nodes = list(planner.graph.nodes)[:5]
                            print(f"   Example valid nodes: {sample_nodes}")
                            
                            # Check if it's close to any valid nodes
                            similar_nodes = [n for n in planner.graph.nodes if abs(n - start_node) < 10]
                            if similar_nodes:
                                print(f"   Similar nodes found: {similar_nodes[:3]}")
                            
                            continue
                    
                    distance_input = input("Enter target distance (km): ").strip()
                    try:
                        target_distance = float(distance_input)
                        if target_distance <= 0:
                            print("❌ Distance must be positive")
                            continue
                    except ValueError:
                        print(f"❌ Invalid distance: '{distance_input}' is not a valid number")
                        continue
                    
                    print("\nObjective options:")
                    objectives = {
                        '1': RouteObjective.MINIMIZE_DISTANCE,
                        '2': RouteObjective.MAXIMIZE_ELEVATION,
                        '3': RouteObjective.BALANCED_ROUTE,
                        '4': RouteObjective.MINIMIZE_DIFFICULTY
                    }
                    
                    print("1. Shortest route")
                    print("2. Maximum elevation gain")
                    print("3. Balanced route")
                    print("4. Easiest route")
                    
                    obj_choice = input("Select objective (1-4): ").strip()
                    if obj_choice not in objectives:
                        print("❌ Invalid objective")
                        continue
                    
                    algorithm_input = input("Algorithm (nearest_neighbor/genetic) [nearest_neighbor]: ").strip() or "nearest_neighbor"
                    if algorithm_input not in ["nearest_neighbor", "genetic"]:
                        print(f"❌ Invalid algorithm: '{algorithm_input}'. Using nearest_neighbor.")
                        algorithm = "nearest_neighbor"
                    else:
                        algorithm = algorithm_input
                    
                    # Generate route
                    result = planner.generate_route(
                        start_node, target_distance, 
                        objectives[obj_choice], algorithm
                    )
                    
                    if result:
                        planner.display_route_stats(result)
                        
                        # Ask for directions
                        try:
                            directions_input = input("\nShow turn-by-turn directions? (y/n): ").strip().lower()
                            if directions_input in ['y', 'yes']:
                                planner.generate_directions(result)
                        except (ValueError, EOFError, KeyboardInterrupt):
                            print("\n⏹️ Skipping directions")
                        
                        # Ask for visualization
                        try:
                            viz_input = input("\nCreate route visualization? (y/n): ").strip().lower()
                            if viz_input in ['y', 'yes']:
                                save_file = f"route_{start_node}_{target_distance}km.png"
                                try:
                                    planner.create_route_visualization(result, save_file)
                                except Exception as viz_error:
                                    print(f"❌ Visualization failed: {viz_error}")
                                    print("   This might be due to missing matplotlib or display issues")
                            elif viz_input in ['n', 'no']:
                                print("⏹️ Skipping visualization")
                            else:
                                print(f"❌ Invalid input: '{viz_input}'. Please enter 'y' or 'n'")
                        except (EOFError, KeyboardInterrupt):
                            print("\n⏹️ Skipping visualization")
                    
                except ValueError:
                    print("❌ Invalid input")
                except KeyboardInterrupt:
                    print("\n⏹️ Operation cancelled")
                
            elif choice == '4':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Running Route Optimizer - Command Line Interface',
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
        planner = CLIRoutePlanner()
        if not planner.load_network():
            sys.exit(1)
        
        # Map objective
        obj_map = {
            'distance': RouteObjective.MINIMIZE_DISTANCE,
            'elevation': RouteObjective.MAXIMIZE_ELEVATION,
            'balanced': RouteObjective.BALANCED_ROUTE,
            'difficulty': RouteObjective.MINIMIZE_DIFFICULTY
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