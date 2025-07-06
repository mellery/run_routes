#!/usr/bin/env python3
"""
Enhanced Command Line Route Planner with 3DEP Support
Uses shared route services with enhanced 3DEP elevation integration
"""

import argparse
import sys
import time
import os
from typing import List, Tuple

# Add project root for elevation imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from route_services import (
    NetworkManager, RouteOptimizer, RouteAnalyzer, 
    RouteFormatter
)

# Import enhanced elevation profiler
try:
    from route_services.elevation_profiler_enhanced import EnhancedElevationProfiler
    ENHANCED_ELEVATION_AVAILABLE = True
except ImportError:
    from route_services import ElevationProfiler
    ENHANCED_ELEVATION_AVAILABLE = False

# Import elevation data sources for configuration
try:
    from elevation_data_sources import get_elevation_manager, ElevationConfig
    ELEVATION_SOURCES_AVAILABLE = True
except ImportError:
    ELEVATION_SOURCES_AVAILABLE = False


class RefactoredCLIRoutePlanner:
    """Enhanced command line interface with 3DEP elevation support"""
    
    def __init__(self, elevation_config_path=None):
        """Initialize the CLI route planner with enhanced elevation support
        
        Args:
            elevation_config_path: Optional path to elevation configuration file
        """
        self.services = None
        self.selected_start_node = 1529188403  # Default starting point
        self.elevation_config_path = elevation_config_path
        self.elevation_manager = None
        
    def initialize_services(self, center_point=None, radius_km=5.0):
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
            
            # Initialize elevation management
            if ELEVATION_SOURCES_AVAILABLE:
                try:
                    self.elevation_manager = get_elevation_manager(self.elevation_config_path)
                    elevation_source = self.elevation_manager.get_elevation_source()
                    if elevation_source:
                        print(f"📊 Enhanced elevation: {elevation_source.get_resolution()}m resolution available")
                except Exception as e:
                    print(f"⚠️ Enhanced elevation initialization failed: {e}")
            
            # Create all services with enhanced elevation support
            route_optimizer = RouteOptimizer(graph, self.elevation_config_path)
            
            if ENHANCED_ELEVATION_AVAILABLE:
                elevation_profiler = EnhancedElevationProfiler(graph, self.elevation_config_path)
            else:
                elevation_profiler = ElevationProfiler(graph)
            
            self.services = {
                'network_manager': network_manager,
                'route_optimizer': route_optimizer,
                'route_analyzer': RouteAnalyzer(graph),
                'elevation_profiler': elevation_profiler,
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
    
    def list_starting_points(self, num_points=50):
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
        
        # Get all intersections in the network (city-wide)
        all_intersections = network_manager.get_all_intersections(graph, max_nodes=num_points)
        
        # Convert to same format as nearby_nodes for compatibility
        nearby_nodes = []
        from route import haversine_distance
        for node_id, data in all_intersections:
            distance = haversine_distance(center_point[0], center_point[1], data['y'], data['x'])
            nearby_nodes.append((node_id, distance, data))
        
        print(f"\n📍 Available Starting Points (showing {len(nearby_nodes)}):")
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
        print(f"💡 Tip: You can enter the option number (1-{num_points}) or the full node ID")
        
        return node_list
    
    def select_starting_point(self, num_points=50):
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
                user_input = input(f"\nSelect starting point (1-{len(available_nodes)} or node ID, 'back' to return): ").strip()
                
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
                print("\n⏹️ Selection cancelled")
                return None
    
    def generate_route(self, start_node, target_distance, objective=None, algorithm="genetic", exclude_footways=True):
        """Generate optimized route using route services
        
        Args:
            start_node: Starting node ID
            target_distance: Target distance in km
            objective: Route objective
            algorithm: Algorithm to use ('nearest_neighbor', 'genetic', 'unconstrained')
            exclude_footways: Whether to exclude footway/sidewalk segments (default True)
            
        Returns:
            Route result dictionary or None
        """
        if not self.services:
            print("❌ Services not initialized")
            return None
        
        # Handle unconstrained algorithm separately
        if algorithm == "unconstrained":
            return self._generate_unconstrained_route(start_node, target_distance)
        
        route_optimizer = self.services['route_optimizer']
        
        # Use default objective if not provided
        if objective is None:
            objective = route_optimizer.RouteObjective.MAXIMIZE_ELEVATION
        
        # Generate route
        result = route_optimizer.optimize_route(
            start_node=start_node,
            target_distance_km=target_distance,
            objective=objective,
            algorithm=algorithm,
            exclude_footways=exclude_footways
        )
        
        if result:
            solve_time = result.get('solver_info', {}).get('solve_time', 0)
            print(f"✅ Route generated in {solve_time:.2f} seconds")
        
        return result
    
    def _generate_unconstrained_route(self, start_node, target_distance):
        """Generate route using unconstrained TSP approach
        
        Args:
            start_node: Starting node ID
            target_distance: Target distance in km
            
        Returns:
            Route result dictionary or None
        """
        print(f"\n🚀 Generating unconstrained TSP route...")
        print(f"   Start node: {start_node}")
        print(f"   Target distance: {target_distance:.1f}km")
        print(f"   Algorithm: unconstrained (shortest-path distances)")
        
        try:
            # Import the unconstrained solver
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from quick_unconstrained_tsp import QuickUnconstrainedTSP
            
            # Get filtered candidates using existing logic
            route_optimizer = self.services['route_optimizer']
            candidate_nodes = route_optimizer._get_intersection_nodes()
            
            # Apply distance filtering
            max_straight_line_km = (target_distance / 2.0) + 1.5
            straight_line_filtered = route_optimizer._filter_nodes_by_distance(
                candidate_nodes, start_node, max_straight_line_km
            )
            max_road_distance_km = (target_distance / 2.0) + 1.0
            filtered_candidates = route_optimizer._filter_nodes_by_road_distance(
                straight_line_filtered, start_node, max_road_distance_km
            )
            
            if start_node not in filtered_candidates:
                filtered_candidates.append(start_node)
            
            print(f"   Candidates: {len(filtered_candidates)} filtered nodes")
            
            # Create and run unconstrained solver
            graph = self.services['route_optimizer'].graph
            solver = QuickUnconstrainedTSP(graph, filtered_candidates)
            
            # Try both greedy methods
            tolerance = 0.12  # ±12% tolerance
            result = solver.greedy_distance_aware(start_node, target_distance, tolerance=tolerance)
            
            if not result or not result.get('valid'):
                print(f"   Trying enhanced method...")
                result = solver.enhanced_greedy(start_node, target_distance, tolerance=tolerance)
            
            if result:
                # Convert to format expected by CLI
                solve_time = result.get('search_time', 0)
                print(f"✅ Unconstrained route generated in {solve_time:.2f} seconds")
                
                # Convert to CLI format
                cli_result = {
                    'route': result['route'],
                    'distance_km': result['distance_km'],
                    'total_distance_km': result['distance_km'],
                    'solver_info': {
                        'solve_time': solve_time,
                        'algorithm': 'unconstrained',
                        'method': result.get('method', 'unconstrained'),
                        'routes_tested': result.get('routes_tested', 0)
                    },
                    'valid': result.get('valid', False),
                    'stats': {
                        'total_distance_km': result['distance_km'],
                        'distance_km': result['distance_km'],
                        'num_nodes': len(result['route']),
                        'total_elevation_gain_m': 0,  # Will be calculated by analyzer
                        'total_elevation_loss_m': 0,  # Will be calculated by analyzer
                        'net_elevation_gain_m': 0,   # Will be calculated by analyzer
                        'max_grade_percent': 0,      # Will be calculated by analyzer
                        'estimated_time_min': 0      # Will be calculated by analyzer
                    }
                }
                
                # Calculate proper elevation statistics using route analyzer
                print(f"   Calculating elevation profile...")
                try:
                    route_analyzer = self.services['route_analyzer']
                    analysis = route_analyzer.analyze_route(cli_result)
                    
                    # Calculate elevation manually since analyzer format differs
                    route_nodes = result['route']
                    elevations = []
                    for node in route_nodes:
                        if node in graph.nodes:
                            elevation = graph.nodes[node].get('elevation', 0)
                            elevations.append(elevation)
                    
                    if elevations:
                        # Calculate gains and losses
                        total_gain = 0
                        total_loss = 0
                        
                        # Go through route segments
                        for i in range(len(elevations) - 1):
                            diff = elevations[i+1] - elevations[i]
                            if diff > 0:
                                total_gain += diff
                            else:
                                total_loss += abs(diff)
                        
                        # Add return to start
                        return_diff = elevations[0] - elevations[-1]
                        if return_diff > 0:
                            total_gain += return_diff
                        else:
                            total_loss += abs(return_diff)
                        
                        # Estimate time (assume 6 min/km base + 30 sec per 10m elevation gain)
                        base_time = result['distance_km'] * 6  # 6 min/km
                        elevation_penalty = total_gain * 0.5  # 30 sec per 10m = 0.5 min per 10m
                        estimated_time = base_time + elevation_penalty
                        
                        # Calculate max grade (simplified)
                        max_grade = 0
                        if len(elevations) > 1:
                            for i in range(len(elevations) - 1):
                                # This is simplified - real grade needs distance between nodes
                                elev_diff = abs(elevations[i+1] - elevations[i])
                                # Assume ~500m between major intersections for grade calc
                                grade = (elev_diff / 500) * 100
                                max_grade = max(max_grade, grade)
                        
                        # Update stats with calculated elevation data
                        cli_result['stats'].update({
                            'total_elevation_gain_m': total_gain,
                            'total_elevation_loss_m': total_loss,
                            'net_elevation_gain_m': total_gain - total_loss,
                            'max_grade_percent': max_grade,
                            'estimated_time_min': estimated_time
                        })
                        
                        print(f"   ✅ Elevation analysis: +{total_gain:.0f}m gain, -{total_loss:.0f}m loss")
                    else:
                        print(f"   ⚠️ No elevation data available")
                    
                except Exception as e:
                    print(f"   ⚠️ Elevation analysis failed: {e}")
                
                return cli_result
            else:
                print(f"❌ Unconstrained solver failed to find route")
                return None
                
        except ImportError as e:
            print(f"❌ Unconstrained solver not available: {e}")
            return None
        except Exception as e:
            print(f"❌ Error in unconstrained solver: {e}")
            return None
    
    def show_elevation_status(self):
        """Show elevation data source status"""
        if not ELEVATION_SOURCES_AVAILABLE:
            print("❌ Enhanced elevation system not available")
            return
        
        try:
            if not self.elevation_manager:
                self.elevation_manager = get_elevation_manager(self.elevation_config_path)
            
            print("\n📊 Elevation Data Source Status")
            print("=" * 40)
            
            # Show available sources
            available_sources = self.elevation_manager.get_available_sources()
            print(f"Available sources: {available_sources}")
            
            # Show active source
            active_source = self.elevation_manager.get_elevation_source()
            if active_source:
                source_info = active_source.get_source_info()
                print(f"Active source: {source_info.get('type', 'Unknown')}")
                print(f"Resolution: {active_source.get_resolution()}m")
                
                # Show coverage
                bounds = active_source.get_coverage_bounds()
                print(f"Coverage: {bounds}")
                
                # Show hybrid source statistics if available
                if hasattr(active_source, 'get_stats'):
                    stats = active_source.get_stats()
                    if stats and 'primary_queries' in stats:
                        total_queries = sum(stats[k] for k in ['primary_queries', 'fallback_queries', 'failed_queries'])
                        if total_queries > 0:
                            print(f"\nUsage Statistics:")
                            print(f"  High-resolution queries: {stats['primary_percentage']:.1f}%")
                            print(f"  Fallback queries: {stats['fallback_percentage']:.1f}%")
                            print(f"  Failed queries: {stats['failure_percentage']:.1f}%")
            else:
                print("❌ No active elevation source")
            
            # Test elevation access
            test_lat, test_lon = 37.1299, -80.4094  # Christiansburg, VA
            test_results = self.elevation_manager.test_sources(test_lat, test_lon)
            
            print(f"\nElevation Test at ({test_lat}, {test_lon}):")
            for source_name, result in test_results.items():
                status = "✅" if result['available'] and result['elevation'] else "❌"
                elevation = f"{result['elevation']:.1f}m" if result['elevation'] else "N/A"
                resolution = f"{result['resolution']:.1f}m" if 'resolution' in result else "N/A"
                print(f"  {status} {source_name}: {elevation} (res: {resolution})")
            
        except Exception as e:
            print(f"❌ Failed to get elevation status: {e}")
    
    def configure_elevation_source(self, preferred_source=None):
        """Configure elevation data source preferences
        
        Args:
            preferred_source: Preferred source name ('3dep_local', 'srtm', 'hybrid')
        """
        if not ELEVATION_SOURCES_AVAILABLE:
            print("❌ Enhanced elevation system not available")
            return
        
        try:
            # Create or load configuration
            config = ElevationConfig()
            if self.elevation_config_path and os.path.exists(self.elevation_config_path):
                config = ElevationConfig.from_file(self.elevation_config_path)
            
            if preferred_source:
                config.preferred_source = preferred_source
                print(f"✅ Set preferred elevation source to: {preferred_source}")
            
            # Save configuration
            config_path = self.elevation_config_path or "elevation_config.json"
            config.to_file(config_path)
            print(f"✅ Saved elevation configuration to: {config_path}")
            
            # Reinitialize elevation manager
            self.elevation_manager = get_elevation_manager(config_path)
            print("✅ Elevation configuration updated")
            
        except Exception as e:
            print(f"❌ Failed to configure elevation source: {e}")
    
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
        
        # Show enhanced elevation information if available
        if hasattr(route_result, 'enhanced_profile') and route_result.get('enhanced_profile'):
            print("\n📊 Enhanced Elevation Analysis:")
            data_source = route_result.get('data_source_info', {})
            if data_source:
                print(f"   Data source: {data_source.get('active_source', 'Unknown')}")
                print(f"   Resolution: {data_source.get('resolution_m', 'Unknown')}m")
                
                if 'usage_stats' in data_source:
                    stats = data_source['usage_stats']
                    if 'primary_percentage' in stats:
                        print(f"   High-res coverage: {stats['primary_percentage']:.1f}%")
    
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
        
        print(f"\n📈 Creating route visualization...")
        
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
                
                # Elevation profile with improved Y-axis scaling
                ax.plot(distances_km, elevations, 'g-', linewidth=2, marker='o', markersize=4)
                ax.fill_between(distances_km, elevations, alpha=0.3, color='green')
                ax.set_xlabel('Distance (km)')
                ax.set_ylabel('Elevation (m)')
                ax.set_title(f'Elevation Profile - {profile_data.get("total_distance_km", 0):.2f}km Route')
                ax.grid(True, alpha=0.3)
                
                # Set Y-axis to start from lowest elevation with some padding
                if elevations:
                    min_elev = min(elevations)
                    max_elev = max(elevations)
                    elev_range = max_elev - min_elev
                    padding = max(5, elev_range * 0.1)  # 10% padding or 5m minimum
                    ax.set_ylim(min_elev - padding, max_elev + padding)
                
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
    
    def export_route_for_mapping(self, route_result, start_node, target_distance):
        """Export route with detailed path for mapping applications
        
        Args:
            route_result: Route result from optimizer
            start_node: Starting node ID
            target_distance: Target distance in km
        """
        if not self.services or not route_result:
            return
        
        elevation_profiler = self.services['elevation_profiler']
        route_formatter = self.services['route_formatter']
        
        print(f"\n🗺️ Exporting route for mapping...")
        
        try:
            # Get detailed path with all intermediate nodes
            detailed_path = elevation_profiler.get_detailed_route_path(route_result)
            
            if not detailed_path:
                print("❌ No detailed path data available")
                return
            
            # Count node types
            intersections = len([p for p in detailed_path if p.get('node_type') == 'intersection'])
            geometry_nodes = len([p for p in detailed_path if p.get('node_type') == 'geometry'])
            
            print(f"   Detailed path: {len(detailed_path)} total nodes")
            print(f"   - {intersections} intersections")
            print(f"   - {geometry_nodes} geometry nodes")
            
            # Export GeoJSON
            geojson_file = f"route_{start_node}_{target_distance}km.geojson"
            geojson_data = route_formatter.export_route_geojson(route_result, detailed_path)
            
            with open(geojson_file, 'w') as f:
                f.write(geojson_data)
            print(f"   ✅ Saved GeoJSON: {geojson_file}")
            
            # Export GPX
            gpx_file = f"route_{start_node}_{target_distance}km.gpx"
            gpx_data = route_formatter.export_route_gpx(route_result, detailed_path)
            
            with open(gpx_file, 'w') as f:
                f.write(gpx_data)
            print(f"   ✅ Saved GPX: {gpx_file}")
            
            print(f"\n📍 Map visualization files created:")
            print(f"   • {geojson_file} - Import into web mapping tools (Leaflet, Mapbox, etc.)")
            print(f"   • {gpx_file} - Import into GPS devices or apps like Strava/Garmin")
            print(f"   These files follow the actual road paths, not straight lines!")
            
        except Exception as e:
            print(f"❌ Route export failed: {e}")
    
    def create_route_map_png(self, route_result, start_node, target_distance):
        """Create static PNG map with route overlay
        
        Args:
            route_result: Route result from optimizer
            start_node: Starting node ID  
            target_distance: Target distance in km
        """
        if not self.services or not route_result:
            return
        
        elevation_profiler = self.services['elevation_profiler']
        
        print(f"\n🗺️ Creating route map...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Get detailed path
            detailed_path = elevation_profiler.get_detailed_route_path(route_result)
            
            if not detailed_path:
                print("❌ No detailed path data available")
                return
            
            # Extract coordinates
            lats = [p['latitude'] for p in detailed_path]
            lons = [p['longitude'] for p in detailed_path]
            elevations = [p['elevation'] for p in detailed_path]
            
            # Calculate bounds with some padding
            lat_margin = (max(lats) - min(lats)) * 0.1
            lon_margin = (max(lons) - min(lons)) * 0.1
            bounds = [
                min(lons) - lon_margin,  # west
                max(lons) + lon_margin,  # east  
                min(lats) - lat_margin,  # south
                max(lats) + lat_margin   # north
            ]
            
            # Create figure with map
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            # Try to add OpenStreetMap background
            try:
                import contextily as ctx
                print("   📍 Adding OpenStreetMap background...")
                
                # Set the bounds first
                ax.set_xlim(bounds[0], bounds[1])
                ax.set_ylim(bounds[2], bounds[3])
                
                # Convert to Web Mercator for contextily
                import pyproj
                transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
                
                # Transform route coordinates
                lons_merc, lats_merc = transformer.transform(lons, lats)
                
                # Transform bounds
                west_merc, south_merc = transformer.transform(bounds[0], bounds[2])
                east_merc, north_merc = transformer.transform(bounds[1], bounds[3])
                
                # Set mercator bounds
                ax.set_xlim(west_merc, east_merc)
                ax.set_ylim(south_merc, north_merc)
                
                # Add OpenStreetMap basemap
                ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.8)
                
                # Plot route path with elevation-based coloring (in mercator coordinates)
                scatter = ax.scatter(lons_merc, lats_merc, c=elevations, cmap='plasma', 
                                   s=12, alpha=0.9, edgecolors='white', linewidths=0.5, zorder=5)
                
                # Add route line (in mercator coordinates)
                ax.plot(lons_merc, lats_merc, 'r-', linewidth=3, alpha=0.8, 
                       label=f'Route ({route_result["stats"]["total_distance_km"]:.2f}km)', zorder=4)
                
                # Mark start/finish point
                if detailed_path:
                    start_point = detailed_path[0]
                    start_lon_merc, start_lat_merc = transformer.transform(start_point['longitude'], start_point['latitude'])
                    ax.plot(start_lon_merc, start_lat_merc, 'go', markersize=15, 
                           markeredgecolor='darkgreen', markeredgewidth=3, 
                           label='Start/Finish', zorder=6)
                
                # Mark key intersections (original TSP waypoints)
                original_route = route_result['route']
                key_intersections = [p for p in detailed_path if p['node_id'] in original_route]
                if len(key_intersections) > 1:  # Exclude start point
                    key_lons = [p['longitude'] for p in key_intersections[1:]]
                    key_lats = [p['latitude'] for p in key_intersections[1:]]
                    key_lons_merc, key_lats_merc = transformer.transform(key_lons, key_lats)
                    ax.plot(key_lons_merc, key_lats_merc, 'bo', markersize=10, 
                           markeredgecolor='darkblue', markeredgewidth=2,
                           label=f'Key Waypoints ({len(key_intersections)-1})', zorder=6)
                
                # Remove lat/lon labels since we're in mercator
                ax.set_xlabel('Easting (m)')
                ax.set_ylabel('Northing (m)')
                
                use_osm = True
                
            except ImportError:
                print("   ⚠️ Contextily not available, using coordinate plot...")
                use_osm = False
            except Exception as e:
                print(f"   ⚠️ OSM basemap failed ({str(e)[:50]}), using coordinate plot...")
                use_osm = False
            
            # Fallback to coordinate plot without OSM
            if not use_osm:
                # Plot route path with elevation-based coloring
                scatter = ax.scatter(lons, lats, c=elevations, cmap='terrain', 
                                   s=8, alpha=0.8, edgecolors='none')
                
                # Add route line
                ax.plot(lons, lats, 'r-', linewidth=2, alpha=0.7, 
                       label=f'Route ({route_result["stats"]["total_distance_km"]:.2f}km)')
                
                # Mark start/finish point
                if detailed_path:
                    start_point = detailed_path[0]
                    ax.plot(start_point['longitude'], start_point['latitude'], 
                           'go', markersize=12, markeredgecolor='darkgreen', 
                           markeredgewidth=2, label='Start/Finish')
                
                # Mark key intersections (original TSP waypoints)
                original_route = route_result['route']
                key_intersections = [p for p in detailed_path if p['node_id'] in original_route]
                if len(key_intersections) > 1:  # Exclude start point
                    key_lons = [p['longitude'] for p in key_intersections[1:]]
                    key_lats = [p['latitude'] for p in key_intersections[1:]]
                    ax.plot(key_lons, key_lats, 'bo', markersize=8, 
                           markeredgecolor='darkblue', markeredgewidth=1,
                           label=f'Key Waypoints ({len(key_intersections)-1})')
                
                # Set bounds and styling for coordinate plot
                ax.set_xlim(bounds[0], bounds[1])
                ax.set_ylim(bounds[2], bounds[3])
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.grid(True, alpha=0.3)
            
            # Common styling for both versions
            ax.set_aspect('equal')
            
            # Title
            ax.set_title(f'Running Route Map - {route_result["stats"]["total_distance_km"]:.2f}km\n'
                        f'Elevation: {min(elevations):.0f}m - {max(elevations):.0f}m '
                        f'(+{route_result["stats"].get("total_elevation_gain_m", 0):.0f}m gain)')
            
            # Add colorbar for elevation
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Elevation (m)')
            
            # Add legend
            ax.legend(loc='upper right', facecolor='white', framealpha=0.9)
            
            # Add route statistics text box
            stats_text = f"""Route Statistics:
Distance: {route_result["stats"]["total_distance_km"]:.2f} km
Elevation Gain: {route_result["stats"].get("total_elevation_gain_m", 0):.0f} m
Est. Time: {route_result["stats"].get("estimated_time_min", 0):.0f} min
Path Detail: {len(detailed_path)} nodes
Key Waypoints: {len(original_route)}"""
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            plt.tight_layout()
            
            # Save map
            map_file = f"route_{start_node}_{target_distance}km_map.png"
            plt.savefig(map_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            osm_status = "with OpenStreetMap background" if use_osm else "with coordinate grid"
            print(f"   ✅ Route map saved: {map_file}")
            print(f"   📍 Map shows ({osm_status}):")
            print(f"   - Route path colored by elevation ({len(detailed_path)} nodes)")
            print(f"   - Start/finish point (green circle)")
            print(f"   - Key waypoints (blue circles)")
            print(f"   - Route statistics and elevation range")
            if use_osm:
                print(f"   - OpenStreetMap background with streets, buildings, and labels")
            
        except ImportError as e:
            print(f"⚠️ Missing dependencies for map creation: {e}")
            print("   Install with: pip install matplotlib contextily pyproj")
        except Exception as e:
            print(f"❌ Map creation failed: {e}")


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
                print(f"\n📍 Current starting point: Node {node_info['node_id']}")
                print(f"   Location: {node_info['latitude']:.6f}, {node_info['longitude']:.6f}")
                print(f"   Elevation: {node_info['elevation']:.0f}m")
        
        print(f"\n📋 Main Menu:")
        print("1. Select starting point")
        print("2. Generate route" + (" (with selected point)" if planner.selected_start_node else " (manual entry)"))
        print("3. Show solver information")
        print("4. Quit")
        
        try:
            choice = input("\nSelect option (1-4): ").strip()
            
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
                        start_input = input("\nEnter starting node (option number or node ID): ").strip()
                        
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
                    print("\nObjective options:")
                    objectives = route_optimizer.get_available_objectives()
                    obj_list = list(objectives.items())
                    
                    for i, (name, _) in enumerate(obj_list, 1):
                        print(f"{i}. {name}")
                    
                    obj_input = input("Select objective (1-4) [2]: ").strip()
                    try:
                        obj_choice = int(obj_input) if obj_input else 2
                        if 1 <= obj_choice <= len(obj_list):
                            objective = obj_list[obj_choice - 1][1]
                        else:
                            objective = obj_list[1][1]  # Default to Maximum Elevation Gain
                    except ValueError:
                        objective = obj_list[1][1]  # Default to Maximum Elevation Gain
                    
                    # Get algorithm
                    print("\nAlgorithm options:")
                    available_algorithms = route_optimizer.get_available_algorithms()
                    algo_options = []
                    
                    # Build algorithm menu based on availability
                    option_num = 1
                    algo_map = {}
                    
                    if "auto" in available_algorithms:
                        print(f"{option_num}. auto (automatic selection based on objective)")
                        algo_map[option_num] = "auto"
                        algo_options.append("auto")
                        option_num += 1
                    
                    if "nearest_neighbor" in available_algorithms:
                        print(f"{option_num}. nearest_neighbor (standard TSP)")
                        algo_map[option_num] = "nearest_neighbor"
                        algo_options.append("nearest_neighbor")
                        option_num += 1
                    
                    if "genetic" in available_algorithms:
                        print(f"{option_num}. genetic (genetic algorithm)")
                        algo_map[option_num] = "genetic"
                        algo_options.append("genetic")
                        option_num += 1
                    
                    # Add unconstrained option
                    print(f"{option_num}. unconstrained (shortest-path distances, no road-adjacent constraints)")
                    algo_map[option_num] = "unconstrained"
                    algo_options.append("unconstrained")
                    
                    # Find genetic algorithm option number for default
                    genetic_option = None
                    for option_num, algo in algo_map.items():
                        if algo == "genetic":
                            genetic_option = option_num
                            break
                    
                    default_option = genetic_option if genetic_option else 1
                    algo_input = input(f"Select algorithm (1-{len(algo_map)}) [{default_option}]: ").strip()
                    try:
                        algo_choice = int(algo_input) if algo_input else default_option
                        algorithm = algo_map.get(algo_choice, "genetic" if "genetic" in available_algorithms else "auto" if "auto" in available_algorithms else "nearest_neighbor")
                    except ValueError:
                        algorithm = "genetic" if "genetic" in available_algorithms else "auto" if "auto" in available_algorithms else "nearest_neighbor"
                    
                    # Ask about footway inclusion
                    footway_input = input("\nInclude footways/sidewalks? (can cause redundant back-and-forth routes) (y/n) [n]: ").strip().lower()
                    exclude_footways = footway_input not in ['y', 'yes']
                    
                    # Check if we need to expand network for larger routes
                    if target_distance > 8.0:  # For routes > 8km
                        required_radius = min(target_distance * 0.8, 25.0)  # 80% of distance, max 25km
                        print(f"🌐 Large route detected ({target_distance}km), expanding network to {required_radius:.1f}km radius...")
                        
                        if not planner.initialize_services(radius_km=required_radius):
                            print("❌ Failed to expand network")
                            continue
                    
                    # Generate route
                    result = planner.generate_route(start_node, target_distance, objective, algorithm, exclude_footways)
                    
                    if result:
                        planner.display_route_stats(result)
                        
                        # Ask for directions
                        directions_input = input("\nShow turn-by-turn directions? (y/n): ").strip().lower()
                        if directions_input in ['y', 'yes']:
                            planner.generate_directions(result)
                        
                        # Ask for visualizations
                        viz_input = input("\nCreate route visualizations? (y/n): ").strip().lower()
                        if viz_input in ['y', 'yes']:
                            # Elevation profile
                            elevation_file = f"route_{start_node}_{target_distance}km_elevation.png"
                            planner.create_route_visualization(result, elevation_file)
                            
                            # Route map
                            planner.create_route_map_png(result, start_node, target_distance)
                            
                            # Ask for file exports
                            export_input = input("\nExport route files (GeoJSON/GPX)? (y/n): ").strip().lower()
                            if export_input in ['y', 'yes']:
                                planner.export_route_for_mapping(result, start_node, target_distance)
                
                except KeyboardInterrupt:
                    print("\n⏹️ Operation cancelled")
                
            elif choice == '3':
                # Show solver information
                solver_info = route_optimizer.get_solver_info()
                print("\n🔧 Solver Information:")
                print(f"   TSP Solver Type: {solver_info['solver_type']}")
                print(f"   TSP Solver Class: {solver_info['solver_class']}")
                print(f"   GA Available: {'✅ Yes' if solver_info.get('ga_available', False) else '❌ No'}")
                if solver_info.get('ga_available', False):
                    print(f"   GA Optimizer: {solver_info.get('ga_optimizer', 'Unknown')}")
                print(f"   Graph nodes: {solver_info['graph_nodes']:,}")
                print(f"   Graph edges: {solver_info['graph_edges']:,}")
                print(f"   Available algorithms: {', '.join(solver_info['available_algorithms'])}")
                print(f"   Available objectives: {', '.join(solver_info['available_objectives'])}")
                
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
    
    parser.add_argument(
        '--algorithm', '-a',
        choices=['auto', 'nearest_neighbor', 'genetic', 'unconstrained'],
        default='auto',
        help='Algorithm to use (auto = automatic selection, unconstrained = shortest-path distances)'
    )
    
    parser.add_argument(
        '--include-footways',
        action='store_true',
        help='Include footway/sidewalk segments (default is to exclude them to avoid redundant paths)'
    )
    
    args = parser.parse_args()
    
    if args.interactive or (not args.start_node and not args.distance):
        interactive_mode()
    else:
        # Command line mode
        planner = RefactoredCLIRoutePlanner()
        
        # Determine required network radius based on target distance
        required_radius = 5.0  # Default
        if args.distance and args.distance > 8.0:
            required_radius = min(args.distance * 0.8, 25.0)  # 80% of distance, max 25km
            print(f"🌐 Large route detected ({args.distance}km), using {required_radius:.1f}km network radius")
        
        if not planner.initialize_services(radius_km=required_radius):
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
            obj_map[args.objective], args.algorithm,
            exclude_footways=not args.include_footways
        )
        
        if result:
            planner.display_route_stats(result)
            planner.generate_directions(result)


if __name__ == "__main__":
    main()