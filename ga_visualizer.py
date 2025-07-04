#!/usr/bin/env python3
"""
GA Visualizer for Development Verification
Creates OpenStreetMap-based visualizations for GA development and debugging
"""

import os
import time
import math
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import folium
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import random

from ga_chromosome import RouteChromosome, RouteSegment


class GAVisualizer:
    """Visualizer for GA development with OpenStreetMap backgrounds"""
    
    def __init__(self, graph: nx.Graph, output_dir: str = "ga_visualizations"):
        """Initialize visualizer
        
        Args:
            graph: NetworkX graph with elevation data
            output_dir: Directory to save visualization images
        """
        self.graph = graph
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get graph bounds for visualization
        self.bounds = self._calculate_graph_bounds()
        
        # Color schemes
        self.elevation_colormap = LinearSegmentedColormap.from_list(
            'elevation', ['blue', 'green', 'yellow', 'orange', 'red']
        )
        
        # Route colors for population visualization
        self.route_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]
        
        print(f"GAVisualizer initialized. Output directory: {output_dir}")
    
    def _calculate_graph_bounds(self) -> Dict[str, float]:
        """Calculate bounding box of the graph"""
        if not self.graph.nodes:
            return {'min_lat': 0, 'max_lat': 0, 'min_lon': 0, 'max_lon': 0}
        
        lats = [data['y'] for _, data in self.graph.nodes(data=True)]
        lons = [data['x'] for _, data in self.graph.nodes(data=True)]
        
        return {
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons),
            'max_lon': max(lons),
            'center_lat': (min(lats) + max(lats)) / 2,
            'center_lon': (min(lons) + max(lons)) / 2
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for filenames"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_chromosome_map(self, chromosome: RouteChromosome, 
                           filename: Optional[str] = None,
                           title: str = "Route Chromosome",
                           show_elevation: bool = True,
                           show_segments: bool = True,
                           show_stats: bool = True) -> str:
        """Save chromosome visualization to PNG
        
        Args:
            chromosome: RouteChromosome to visualize
            filename: Output filename (auto-generated if None)
            title: Plot title
            show_elevation: Color segments by elevation
            show_segments: Show individual segments
            show_stats: Include statistics in title
            
        Returns:
            Path to saved image
        """
        if filename is None:
            timestamp = self._get_timestamp()
            filename = f"ga_dev_chromosome_{timestamp}.png"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot network background with optional OSM
        routes = [chromosome] if chromosome.segments else None
        print(f"   📍 Creating visualization with OpenStreetMap background...")
        use_mercator = self._plot_network_background(ax, routes=routes, use_osm=True)
        
        # Plot chromosome route (with coordinate system matching background)
        self._plot_chromosome_route(ax, chromosome, show_elevation, show_segments, 
                                   use_mercator=use_mercator)
        
        # Set title with statistics if requested
        if show_stats and chromosome.segments:
            stats = chromosome.get_route_stats()
            title_text = (f"{title}\n"
                         f"Distance: {stats['total_distance_km']:.2f}km, "
                         f"Elevation: {stats['total_elevation_gain_m']:.1f}m, "
                         f"Segments: {stats['segment_count']}, "
                         f"Valid: {stats['is_valid']}")
        else:
            title_text = title
        
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
        
        # Set bounds and labels (already set by _plot_network_background if using OSM)
        if not use_mercator:
            self._set_map_bounds(ax, [chromosome] if chromosome.segments else None)
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
        
        # Add legend if showing elevation
        if show_elevation and chromosome.segments:
            self._add_elevation_legend(ax)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved chromosome visualization: {filepath}")
        return filepath
    
    def save_population_map(self, population: List[RouteChromosome],
                           generation: int = 0,
                           filename: Optional[str] = None,
                           show_fitness: bool = True,
                           show_elevation: bool = True,
                           max_routes: int = 10) -> str:
        """Save population visualization to PNG
        
        Args:
            population: List of RouteChromosome objects
            generation: Generation number
            filename: Output filename (auto-generated if None)
            show_fitness: Show fitness values in legend
            show_elevation: Color routes by elevation gain
            max_routes: Maximum number of routes to show
            
        Returns:
            Path to saved image
        """
        if filename is None:
            timestamp = self._get_timestamp()
            filename = f"ga_dev_population_gen{generation:03d}_{timestamp}.png"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # Main plot
        ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        
        # Fitness histogram
        ax_fitness = plt.subplot2grid((3, 3), (0, 2))
        
        # Distance histogram  
        ax_distance = plt.subplot2grid((3, 3), (1, 2))
        
        # Statistics table
        ax_stats = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        
        # Plot network background with optional OSM
        routes_to_plot = population[:max_routes] if len(population) > max_routes else population
        print(f"   📍 Creating population visualization with OpenStreetMap background...")
        use_mercator = self._plot_network_background(ax_main, routes=routes_to_plot, use_osm=True)
        
        # Plot population routes
        for i, chromosome in enumerate(routes_to_plot):
            if chromosome.segments:
                color = self.route_colors[i % len(self.route_colors)]
                alpha = 0.8 if i == 0 else 0.6  # Highlight best route
                
                self._plot_chromosome_route(
                    ax_main, chromosome, 
                    show_elevation=False,  # Use fixed colors for clarity
                    show_segments=False,
                    route_color=color,
                    alpha=alpha,
                    linewidth=3 if i == 0 else 2,
                    use_mercator=use_mercator
                )
        
        # Plot fitness histogram
        fitness_values = [c.fitness for c in population if c and c.fitness is not None]
        if fitness_values:
            ax_fitness.hist(fitness_values, bins=min(10, len(fitness_values)), alpha=0.7, color='skyblue', edgecolor='black')
            ax_fitness.set_title('Fitness Distribution', fontsize=10)
            ax_fitness.set_xlabel('Fitness')
            ax_fitness.set_ylabel('Count')
        else:
            ax_fitness.text(0.5, 0.5, 'No Fitness\nData Yet', transform=ax_fitness.transAxes, 
                           ha='center', va='center', fontsize=10)
            ax_fitness.set_title('Fitness Distribution', fontsize=10)
        
        # Plot distance histogram
        if population:
            distances = [c.get_total_distance() / 1000 for c in population if c]
            if distances and max(distances) > 0:  # Only plot if we have real distances
                ax_distance.hist(distances, bins=min(10, len(distances)), alpha=0.7, color='lightgreen', edgecolor='black')
                ax_distance.set_title('Distance Distribution', fontsize=10)
                ax_distance.set_xlabel('Distance (km)')
                ax_distance.set_ylabel('Count')
            else:
                ax_distance.text(0.5, 0.5, 'Distance\nCalculation\nIssue', transform=ax_distance.transAxes, 
                               ha='center', va='center', fontsize=10)
                ax_distance.set_title('Distance Distribution', fontsize=10)
        else:
            ax_distance.text(0.5, 0.5, 'No Routes', transform=ax_distance.transAxes, 
                           ha='center', va='center', fontsize=10)
            ax_distance.set_title('Distance Distribution', fontsize=10)
        
        # Population statistics table
        self._add_population_stats_table(ax_stats, population, generation)
        
        # Set main plot properties
        ax_main.set_title(f'Population Visualization - Generation {generation}', 
                         fontsize=16, fontweight='bold', pad=20)
        
        # Set bounds and labels (already set by _plot_network_background if using OSM)
        if not use_mercator:
            self._set_map_bounds(ax_main, routes_to_plot)
            ax_main.set_xlabel('Longitude', fontsize=12)
            ax_main.set_ylabel('Latitude', fontsize=12)
        
        ax_main.grid(True, alpha=0.3)
        
        # Save
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved population visualization: {filepath}")
        return filepath
    
    def save_comparison_map(self, ga_route: RouteChromosome, 
                           tsp_route: Optional[List[int]] = None,
                           filename: Optional[str] = None,
                           title: str = "GA vs TSP Comparison") -> str:
        """Save GA vs TSP comparison visualization
        
        Args:
            ga_route: GA-generated route
            tsp_route: TSP-generated route (list of node IDs)
            filename: Output filename
            title: Plot title
            
        Returns:
            Path to saved image
        """
        if filename is None:
            timestamp = self._get_timestamp()
            filename = f"ga_dev_comparison_{timestamp}.png"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot GA route
        self._plot_network_background(ax1)
        self._plot_chromosome_route(ax1, ga_route, show_elevation=True)
        ga_stats = ga_route.get_route_stats()
        ax1.set_title(f'GA Route\n'
                     f'Distance: {ga_stats["total_distance_km"]:.2f}km, '
                     f'Elevation: {ga_stats["total_elevation_gain_m"]:.1f}m',
                     fontsize=14, fontweight='bold')
        self._set_map_bounds(ax1, [ga_route])
        ax1.grid(True, alpha=0.3)
        
        # Plot TSP route if provided
        if tsp_route:
            self._plot_network_background(ax2)
            self._plot_tsp_route(ax2, tsp_route)
            tsp_stats = self._calculate_tsp_stats(tsp_route)
            ax2.set_title(f'TSP Route\n'
                         f'Distance: {tsp_stats["distance_km"]:.2f}km, '
                         f'Elevation: {tsp_stats["elevation_gain_m"]:.1f}m',
                         fontsize=14, fontweight='bold')
            self._set_map_bounds(ax2, [tsp_route])
        else:
            ax2.text(0.5, 0.5, 'TSP Route\nNot Available', 
                    transform=ax2.transAxes, ha='center', va='center',
                    fontsize=16, fontweight='bold')
            self._set_map_bounds(ax2)
        ax2.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle(title, fontsize=18, fontweight='bold')
        
        # Save
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison visualization: {filepath}")
        return filepath
    
    def _plot_network_background(self, ax, routes=None, use_osm=True) -> bool:
        """Plot network background with optional OpenStreetMap basemap
        
        Args:
            ax: Matplotlib axis
            routes: Routes to determine bounds
            use_osm: Whether to try using OpenStreetMap background
            
        Returns:
            True if OSM background was used, False if fallback was used
        """
        # Try to use OpenStreetMap background
        if use_osm:
            try:
                import contextily as ctx
                import pyproj
                
                # Calculate bounds based on routes or graph
                if routes:
                    all_lats, all_lons = self._get_route_coordinates(routes)
                else:
                    all_lats = [data['y'] for _, data in self.graph.nodes(data=True)]
                    all_lons = [data['x'] for _, data in self.graph.nodes(data=True)]
                
                if not all_lats or not all_lons:
                    raise ValueError("No coordinate data available")
                
                # Calculate bounds with proper aspect ratio
                min_lat, max_lat = min(all_lats), max(all_lats)
                min_lon, max_lon = min(all_lons), max(all_lons)
                
                # Add base margin
                lat_range = max_lat - min_lat
                lon_range = max_lon - min_lon
                
                # Ensure minimum range to avoid tiny bounds
                min_range = 0.002  # ~200m at this latitude
                if lat_range < min_range:
                    lat_center = (min_lat + max_lat) / 2
                    min_lat = lat_center - min_range / 2
                    max_lat = lat_center + min_range / 2
                    lat_range = min_range
                
                if lon_range < min_range:
                    lon_center = (min_lon + max_lon) / 2
                    min_lon = lon_center - min_range / 2
                    max_lon = lon_center + min_range / 2
                    lon_range = min_range
                
                # Calculate proper aspect ratio for this latitude
                # 1 degree longitude ≈ cos(latitude) * 111 km
                lat_center = (min_lat + max_lat) / 2
                cos_lat = abs(math.cos(math.radians(lat_center)))
                
                # Adjust ranges to maintain square aspect ratio in projected coordinates
                if lat_range * cos_lat > lon_range:
                    # Latitude range is larger, expand longitude
                    target_lon_range = lat_range * cos_lat
                    lon_expansion = (target_lon_range - lon_range) / 2
                    min_lon -= lon_expansion
                    max_lon += lon_expansion
                else:
                    # Longitude range is larger, expand latitude
                    target_lat_range = lon_range / cos_lat
                    lat_expansion = (target_lat_range - lat_range) / 2
                    min_lat -= lat_expansion
                    max_lat += lat_expansion
                
                # Add final margin
                margin_factor = 0.1
                lat_margin = (max_lat - min_lat) * margin_factor
                lon_margin = (max_lon - min_lon) * margin_factor
                
                bounds = [
                    min_lon - lon_margin,  # west
                    max_lon + lon_margin,  # east
                    min_lat - lat_margin,  # south
                    max_lat + lat_margin   # north
                ]
                
                # Transform to Web Mercator
                transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
                west_merc, south_merc = transformer.transform(bounds[0], bounds[2])
                east_merc, north_merc = transformer.transform(bounds[1], bounds[3])
                
                # Set mercator bounds
                ax.set_xlim(west_merc, east_merc)
                ax.set_ylim(south_merc, north_merc)
                
                # Add OpenStreetMap basemap
                ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.7)
                
                # Update axis labels and maintain aspect ratio
                ax.set_xlabel('Easting (m)')
                ax.set_ylabel('Northing (m)')
                ax.set_aspect('equal')
                
                return True
                
            except (ImportError, Exception) as e:
                print(f"   ⚠️ OSM basemap failed ({str(e)[:50]}), using network background...")
        
        # Fallback to network plot
        self._plot_network_background_fallback(ax)
        return False
    
    def _plot_network_background_fallback(self, ax) -> None:
        """Plot network as light background (fallback when OSM unavailable)"""
        # Plot edges
        for edge in self.graph.edges():
            node1, node2 = edge
            x1, y1 = self.graph.nodes[node1]['x'], self.graph.nodes[node1]['y']
            x2, y2 = self.graph.nodes[node2]['x'], self.graph.nodes[node2]['y']
            ax.plot([x1, x2], [y1, y2], 'lightgray', alpha=0.3, linewidth=0.5)
        
        # Plot nodes
        node_x = [data['x'] for _, data in self.graph.nodes(data=True)]
        node_y = [data['y'] for _, data in self.graph.nodes(data=True)]
        ax.scatter(node_x, node_y, c='lightgray', s=1, alpha=0.5)
    
    def _get_route_coordinates(self, routes) -> Tuple[List[float], List[float]]:
        """Extract all coordinates from routes for bounds calculation"""
        all_lats = []
        all_lons = []
        
        for route in routes:
            if hasattr(route, 'segments'):
                # RouteChromosome
                for segment in route.segments:
                    for node in segment.path_nodes:
                        if node in self.graph.nodes:
                            all_lats.append(self.graph.nodes[node]['y'])
                            all_lons.append(self.graph.nodes[node]['x'])
            elif isinstance(route, list):
                # Node list (TSP route)
                for node in route:
                    if node in self.graph.nodes:
                        all_lats.append(self.graph.nodes[node]['y'])
                        all_lons.append(self.graph.nodes[node]['x'])
        
        return all_lats, all_lons
    
    def _plot_chromosome_route(self, ax, chromosome: RouteChromosome,
                             show_elevation: bool = True,
                             show_segments: bool = True,
                             route_color: Optional[str] = None,
                             alpha: float = 0.8,
                             linewidth: float = 3,
                             use_mercator: bool = False) -> None:
        """Plot chromosome route on axis"""
        if not chromosome.segments:
            return
        
        # Initialize coordinate transformer if needed
        transformer = None
        if use_mercator:
            try:
                import pyproj
                transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            except ImportError:
                use_mercator = False
        
        # Plot each segment
        for i, segment in enumerate(chromosome.segments):
            if not segment.path_nodes or len(segment.path_nodes) < 2:
                continue
            
            # Get coordinates for segment path
            x_coords = []
            y_coords = []
            elevations = []
            
            for node in segment.path_nodes:
                if node in self.graph.nodes:
                    lon = self.graph.nodes[node]['x']
                    lat = self.graph.nodes[node]['y']
                    
                    if use_mercator and transformer:
                        x_merc, y_merc = transformer.transform(lon, lat)
                        x_coords.append(x_merc)
                        y_coords.append(y_merc)
                    else:
                        x_coords.append(lon)
                        y_coords.append(lat)
                    
                    elevations.append(self.graph.nodes[node].get('elevation', 0))
            
            if len(x_coords) < 2:
                continue
            
            # Determine color
            if route_color:
                color = route_color
            elif show_elevation:
                # Color by elevation gain
                if segment.elevation_gain > 20:
                    color = 'red'
                elif segment.elevation_gain > 10:
                    color = 'orange'
                elif segment.elevation_gain > 0:
                    color = 'green'
                else:
                    color = 'blue'
            else:
                color = 'blue'
            
            # Plot segment
            ax.plot(x_coords, y_coords, color=color, alpha=alpha, 
                   linewidth=linewidth, zorder=10)
            
            # Mark segment boundaries if requested
            if show_segments and len(chromosome.segments) > 1:
                start_lon = self.graph.nodes[segment.start_node]['x']
                start_lat = self.graph.nodes[segment.start_node]['y']
                
                if use_mercator and transformer:
                    start_x, start_y = transformer.transform(start_lon, start_lat)
                else:
                    start_x, start_y = start_lon, start_lat
                
                ax.scatter(start_x, start_y, c='yellow', s=30, 
                          zorder=15, edgecolors='black', linewidth=1)
        
        # Mark start/end points
        if chromosome.segments:
            start_node = chromosome.segments[0].start_node
            end_node = chromosome.segments[-1].end_node
            
            start_lon = self.graph.nodes[start_node]['x']
            start_lat = self.graph.nodes[start_node]['y']
            
            if use_mercator and transformer:
                start_x, start_y = transformer.transform(start_lon, start_lat)
            else:
                start_x, start_y = start_lon, start_lat
            
            ax.scatter(start_x, start_y, c='green', s=100, marker='o',
                      zorder=20, edgecolors='black', linewidth=2, label='Start')
            
            if start_node != end_node:
                end_lon = self.graph.nodes[end_node]['x']
                end_lat = self.graph.nodes[end_node]['y']
                
                if use_mercator and transformer:
                    end_x, end_y = transformer.transform(end_lon, end_lat)
                else:
                    end_x, end_y = end_lon, end_lat
                
                ax.scatter(end_x, end_y, c='red', s=100, marker='s',
                          zorder=20, edgecolors='black', linewidth=2, label='End')
    
    def _plot_tsp_route(self, ax, tsp_route: List[int]) -> None:
        """Plot TSP route on axis"""
        if len(tsp_route) < 2:
            return
        
        # Plot route segments
        for i in range(len(tsp_route) - 1):
            node1, node2 = tsp_route[i], tsp_route[i + 1]
            if node1 in self.graph.nodes and node2 in self.graph.nodes:
                x1, y1 = self.graph.nodes[node1]['x'], self.graph.nodes[node1]['y']
                x2, y2 = self.graph.nodes[node2]['x'], self.graph.nodes[node2]['y']
                ax.plot([x1, x2], [y1, y2], 'purple', alpha=0.8, linewidth=3, zorder=10)
        
        # Mark start point
        if tsp_route:
            start_x = self.graph.nodes[tsp_route[0]]['x']
            start_y = self.graph.nodes[tsp_route[0]]['y']
            ax.scatter(start_x, start_y, c='green', s=100, marker='o',
                      zorder=20, edgecolors='black', linewidth=2)
    
    def _calculate_tsp_stats(self, tsp_route: List[int]) -> Dict[str, float]:
        """Calculate statistics for TSP route"""
        if len(tsp_route) < 2:
            return {'distance_km': 0.0, 'elevation_gain_m': 0.0}
        
        total_distance = 0.0
        total_elevation_gain = 0.0
        
        for i in range(len(tsp_route) - 1):
            node1, node2 = tsp_route[i], tsp_route[i + 1]
            if (node1 in self.graph.nodes and node2 in self.graph.nodes and
                self.graph.has_edge(node1, node2)):
                
                # Distance (handle MultiGraph format)
                edge_data = self.graph[node1][node2]
                if 0 in edge_data:
                    total_distance += edge_data[0].get('length', 0.0)
                else:
                    total_distance += edge_data.get('length', 0.0)
                
                # Elevation
                elev1 = self.graph.nodes[node1].get('elevation', 0.0)
                elev2 = self.graph.nodes[node2].get('elevation', 0.0)
                if elev2 > elev1:
                    total_elevation_gain += (elev2 - elev1)
        
        return {
            'distance_km': total_distance / 1000,
            'elevation_gain_m': total_elevation_gain
        }
    
    def _set_map_bounds(self, ax, routes=None) -> None:
        """Set axis bounds to route extent or graph extent"""
        if routes:
            # Calculate bounds based on actual routes
            all_lats = []
            all_lons = []
            
            for route in routes:
                if hasattr(route, 'segments'):
                    for segment in route.segments:
                        for node in segment.path_nodes:
                            if node in self.graph.nodes:
                                all_lats.append(self.graph.nodes[node]['y'])
                                all_lons.append(self.graph.nodes[node]['x'])
                elif isinstance(route, list):  # TSP route as node list
                    for node in route:
                        if node in self.graph.nodes:
                            all_lats.append(self.graph.nodes[node]['y'])
                            all_lons.append(self.graph.nodes[node]['x'])
            
            if all_lats and all_lons:
                min_lat, max_lat = min(all_lats), max(all_lats)
                min_lon, max_lon = min(all_lons), max(all_lons)
                
                # Add margin based on route extent
                lat_range = max_lat - min_lat
                lon_range = max_lon - min_lon
                margin_lat = max(0.0005, lat_range * 0.1)  # 10% margin, minimum 0.0005
                margin_lon = max(0.0005, lon_range * 0.1)
                
                ax.set_xlim(min_lon - margin_lon, max_lon + margin_lon)
                ax.set_ylim(min_lat - margin_lat, max_lat + margin_lat)
                ax.set_aspect('equal')
                return
        
        # Fallback to graph bounds
        margin = 0.001  # Small margin around bounds
        ax.set_xlim(self.bounds['min_lon'] - margin, self.bounds['max_lon'] + margin)
        ax.set_ylim(self.bounds['min_lat'] - margin, self.bounds['max_lat'] + margin)
        ax.set_aspect('equal')
    
    def _add_elevation_legend(self, ax) -> None:
        """Add elevation color legend"""
        legend_elements = [
            plt.Line2D([0], [0], color='red', lw=3, label='High Gain (>20m)'),
            plt.Line2D([0], [0], color='orange', lw=3, label='Medium Gain (10-20m)'),
            plt.Line2D([0], [0], color='green', lw=3, label='Low Gain (0-10m)'),
            plt.Line2D([0], [0], color='blue', lw=3, label='No Gain/Loss')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    def _add_population_stats_table(self, ax, population: List[RouteChromosome], 
                                   generation: int) -> None:
        """Add population statistics table"""
        ax.axis('off')
        
        if not population:
            ax.text(0.5, 0.5, 'No population data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=14)
            return
        
        # Calculate statistics
        valid_chromosomes = [c for c in population if c.is_valid and c.segments]
        
        if not valid_chromosomes:
            ax.text(0.5, 0.5, 'No valid chromosomes', transform=ax.transAxes,
                   ha='center', va='center', fontsize=14)
            return
        
        distances = [c.get_total_distance() / 1000 for c in valid_chromosomes]
        elevations = [c.get_elevation_gain() for c in valid_chromosomes]
        fitnesses = [c.fitness for c in valid_chromosomes if c.fitness is not None]
        
        # Create statistics table
        stats_data = [
            ['Generation', str(generation)],
            ['Population Size', str(len(population))],
            ['Valid Routes', str(len(valid_chromosomes))],
            ['Avg Distance (km)', f'{np.mean(distances):.2f}' if distances else 'N/A'],
            ['Avg Elevation (m)', f'{np.mean(elevations):.1f}' if elevations else 'N/A'],
            ['Best Fitness', f'{max(fitnesses):.3f}' if fitnesses else 'N/A'],
            ['Avg Fitness', f'{np.mean(fitnesses):.3f}' if fitnesses else 'N/A']
        ]
        
        # Create table
        table = ax.table(cellText=stats_data, cellLoc='left', loc='center',
                        colWidths=[0.3, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style table
        for i in range(len(stats_data)):
            table[(i, 0)].set_facecolor('#E6E6E6')
            table[(i, 0)].set_text_props(weight='bold')
        
        ax.set_title('Population Statistics', fontsize=12, fontweight='bold')


def test_visualizer():
    """Test function to verify visualizer functionality"""
    print("Testing GAVisualizer...")
    
    # This would normally use a real graph
    # For now, just test the class instantiation
    try:
        # Create a minimal test graph
        test_graph = nx.Graph()
        test_graph.add_node(1, x=-80.4094, y=37.1299, elevation=100)
        test_graph.add_node(2, x=-80.4000, y=37.1300, elevation=110)
        test_graph.add_edge(1, 2, length=100)
        
        visualizer = GAVisualizer(test_graph)
        print("✅ GAVisualizer created successfully")
        
        # Test chromosome creation (minimal)
        from ga_chromosome import RouteSegment, RouteChromosome
        segment = RouteSegment(1, 2, [1, 2])
        segment.calculate_properties(test_graph)
        chromosome = RouteChromosome([segment])
        
        print(f"✅ Test chromosome created: {chromosome}")
        
        # Test visualization (would save file)
        # visualizer.save_chromosome_map(chromosome, "test_chromosome.png")
        print("✅ Visualizer test completed")
        
    except Exception as e:
        print(f"❌ Visualizer test failed: {e}")


if __name__ == "__main__":
    test_visualizer()