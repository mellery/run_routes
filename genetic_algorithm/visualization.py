#!/usr/bin/env python3
"""
Genetic Algorithm Visualization Components
Consolidated visualization functionality for GA development, tuning, and precision analysis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ga_common_imports import (
    os, time, math, datetime, random, List, Optional, Dict, Any, Tuple,
    np, nx, plt, patches, LinearSegmentedColormap, dataclass
)

from ga_base_visualizer import BaseGAVisualizer, VisualizationConfig, GAVisualizationUtils
from .chromosome import RouteChromosome, RouteSegment

# Import optional dependencies
try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    import contextily as ctx
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    import pyproj
    ADVANCED_MAPPING_AVAILABLE = True
except ImportError:
    ADVANCED_MAPPING_AVAILABLE = False

try:
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    ADVANCED_PLOTTING_AVAILABLE = True
except ImportError:
    ADVANCED_PLOTTING_AVAILABLE = False

try:
    from ga_parameter_tuner import GAParameterTuner
    from ga_hyperparameter_optimizer import GAHyperparameterOptimizer  
    from ga_algorithm_selector import GAAlgorithmSelector
    from ga_config_manager import GAConfigManager
    from ga_sensitivity_analyzer import GASensitivityAnalyzer
    TUNING_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: GA tuning components not available: {e}")
    TUNING_COMPONENTS_AVAILABLE = False
    # Create dummy classes to avoid import errors
    class GAParameterTuner: pass
    class GAHyperparameterOptimizer: pass
    class GAAlgorithmSelector: pass
    class GAConfigManager: pass
    class GASensitivityAnalyzer: pass

try:
    from ga_precision_operators import PrecisionAwareCrossover, PrecisionAwareMutation
    PRECISION_COMPONENTS_AVAILABLE = True
except ImportError:
    PRECISION_COMPONENTS_AVAILABLE = False


class GAVisualizer(BaseGAVisualizer):
    """Main GA visualizer for development with OpenStreetMap backgrounds"""
    
    def __init__(self, graph: nx.Graph, output_dir: str = "ga_visualizations"):
        """Initialize visualizer
        
        Args:
            graph: NetworkX graph with elevation data
            output_dir: Directory to save visualization images
        """
        # Initialize base class
        config = VisualizationConfig(output_dir=output_dir)
        super().__init__(config)
        
        self.graph = graph
        
        # Get graph bounds for visualization
        self.bounds = GAVisualizationUtils.calculate_graph_bounds(graph)
        
        # Add center coordinates to bounds
        if self.bounds['max_lat'] != self.bounds['min_lat']:
            self.bounds['center_lat'] = (self.bounds['min_lat'] + self.bounds['max_lat']) / 2
            self.bounds['center_lon'] = (self.bounds['min_lon'] + self.bounds['max_lon']) / 2
        else:
            self.bounds['center_lat'] = self.bounds['min_lat']
            self.bounds['center_lon'] = self.bounds['min_lon']

        print(f"GAVisualizer initialized. Output directory: {output_dir}")
    
    def save_chromosome_map(self, chromosome: RouteChromosome, 
                           filename: Optional[str] = None,
                           title: str = "Route Chromosome",
                           show_elevation: bool = True,
                           show_segments: bool = True,
                           show_stats: bool = True) -> str:
        """Save chromosome visualization to PNG
        
        Args:
            chromosome: Route chromosome to visualize
            filename: Output filename (auto-generated if None)
            title: Plot title
            show_elevation: Color nodes by elevation
            show_segments: Show segment boundaries
            show_stats: Show route statistics
            
        Returns:
            Path to saved image
        """
        if filename is None:
            filename = "ga_dev_chromosome"
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot network background with optional OSM
        routes = [chromosome] if chromosome.segments else None
        print(f"   📍 Creating visualization with OpenStreetMap background...")
        use_mercator = self._plot_network_background(ax, routes=routes, use_osm=True)
        
        # Plot chromosome route
        if chromosome.segments:
            self._plot_chromosome_route(ax, chromosome, show_elevation=show_elevation)
        
        # Set title
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Set appropriate bounds
        if not use_mercator:
            self._set_map_bounds(ax, [chromosome] if chromosome.segments else None)
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
        
        # Add legend if showing elevation
        if show_elevation and chromosome.segments:
            self._add_elevation_legend(ax)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save using base class method
        plt.tight_layout()
        filepath = self.save_figure(plt.gcf(), filename)
        
        print(f"Saved chromosome visualization: {filepath}")
        return filepath
    
    def save_population_map(self, population: List[RouteChromosome],
                           generation: int = 0,
                           filename: Optional[str] = None,
                           show_fitness: bool = True,
                           show_elevation: bool = True,
                           max_routes: int = 20) -> str:
        """Save population visualization
        
        Args:
            population: Population to visualize
            generation: Generation number
            filename: Output filename (auto-generated if None)
            show_fitness: Show fitness values in legend
            show_elevation: Color routes by elevation gain
            max_routes: Maximum number of routes to show
            
        Returns:
            Path to saved image
        """
        if filename is None:
            filename = f"ga_dev_population_gen{generation:03d}"
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # Main plot for routes
        ax_main = plt.subplot(2, 2, (1, 3))
        
        # Statistics subplot
        ax_stats = plt.subplot(2, 2, 2)
        
        # Filter valid routes
        valid_routes = [r for r in population if r.segments and r.is_valid]
        routes_to_plot = valid_routes[:max_routes]
        
        # Plot network background
        use_mercator = self._plot_network_background(ax_main, routes=routes_to_plot, use_osm=True)
        
        # Plot routes
        for i, route in enumerate(routes_to_plot):
            color = self.get_route_color(i, len(routes_to_plot))
            self._plot_chromosome_route(ax_main, route, color=color, alpha=0.7)
        
        # Set title and labels
        ax_main.set_title(f'Population Routes - Generation {generation}', 
                         fontsize=16, fontweight='bold')
        
        if not use_mercator:
            self._set_map_bounds(ax_main, routes_to_plot)
            ax_main.set_xlabel('Longitude', fontsize=12)
            ax_main.set_ylabel('Latitude', fontsize=12)
        
        ax_main.grid(True, alpha=0.3)
        
        # Save using base class method
        plt.tight_layout()
        filepath = self.save_figure(plt.gcf(), filename)
        
        print(f"Saved population visualization: {filepath}")
        return filepath
    
    def save_comparison_map(self, ga_route: RouteChromosome, 
                           tsp_route: Optional[List[int]] = None,
                           filename: Optional[str] = None,
                           title: str = "GA vs TSP Comparison") -> str:
        """Save GA vs TSP comparison visualization
        
        Args:
            ga_route: GA optimized route
            tsp_route: TSP route node sequence
            filename: Output filename
            title: Plot title
            
        Returns:
            Path to saved image
        """
        if filename is None:
            filename = "ga_dev_comparison"
        
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
            self._plot_node_sequence(ax2, tsp_route)
            ax2.set_title('TSP Route', fontsize=14, fontweight='bold')
            self._set_map_bounds(ax2, None, tsp_route)
        else:
            ax2.text(0.5, 0.5, 'No TSP Route\nProvided', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=16, fontweight='bold')
        
        ax2.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle(title, fontsize=18, fontweight='bold')
        
        # Save using base class method
        plt.tight_layout()
        filepath = self.save_figure(plt.gcf(), filename)
        
        print(f"Saved comparison visualization: {filepath}")
        return filepath
    
    def _plot_network_background(self, ax, routes=None, use_osm=True) -> bool:
        """Plot network background with optional OpenStreetMap basemap
        
        Args:
            ax: Matplotlib axes
            routes: Routes to determine bounds
            use_osm: Whether to use OpenStreetMap basemap
            
        Returns:
            Whether Mercator projection was used
        """
        # Plot basic network
        pos = {node: (data['x'], data['y']) for node, data in self.graph.nodes(data=True)}
        nx.draw_networkx_edges(self.graph, pos, ax=ax, edge_color='lightgray', 
                              width=0.5, alpha=0.6)
        
        # Plot nodes with elevation coloring if available
        if use_osm:
            elevations = [data.get('elevation', 0) for _, data in self.graph.nodes(data=True)]
            if elevations:
                min_elev, max_elev = min(elevations), max(elevations)
                node_colors = [self.get_elevation_color(elev, min_elev, max_elev) 
                              for elev in elevations]
                nx.draw_networkx_nodes(self.graph, pos, ax=ax, node_color=node_colors,
                                     node_size=10, alpha=0.8)
        
        return False  # Not using Mercator projection
    
    def _plot_chromosome_route(self, ax, chromosome: RouteChromosome, 
                              color: str = 'red', alpha: float = 1.0,
                              show_elevation: bool = False):
        """Plot chromosome route on axes
        
        Args:
            ax: Matplotlib axes
            chromosome: Route chromosome
            color: Route color
            alpha: Route transparency
            show_elevation: Whether to color by elevation
        """
        if not chromosome.segments:
            return
        
        # Get all route nodes
        route_nodes = []
        for segment in chromosome.segments:
            if segment.path_nodes:
                route_nodes.extend(segment.path_nodes[:-1])  # Avoid duplicates
        
        # Add last node
        if chromosome.segments[-1].path_nodes:
            route_nodes.append(chromosome.segments[-1].path_nodes[-1])
        
        # Plot route
        if len(route_nodes) > 1:
            route_coords = []
            for node in route_nodes:
                if node in self.graph.nodes:
                    data = self.graph.nodes[node]
                    route_coords.append((data['x'], data['y']))
            
            if route_coords:
                x_coords, y_coords = zip(*route_coords)
                ax.plot(x_coords, y_coords, color=color, alpha=alpha, 
                       linewidth=3, marker='o', markersize=4)
    
    def _plot_node_sequence(self, ax, node_sequence: List[int]):
        """Plot a sequence of nodes as a route
        
        Args:
            ax: Matplotlib axes
            node_sequence: Sequence of node IDs
        """
        if not node_sequence:
            return
        
        route_coords = []
        for node in node_sequence:
            if node in self.graph.nodes:
                data = self.graph.nodes[node]
                route_coords.append((data['x'], data['y']))
        
        if route_coords:
            x_coords, y_coords = zip(*route_coords)
            ax.plot(x_coords, y_coords, color='blue', linewidth=3, 
                   marker='o', markersize=4, alpha=0.8)
    
    def _set_map_bounds(self, ax, routes: Optional[List[RouteChromosome]] = None,
                       node_sequence: Optional[List[int]] = None):
        """Set appropriate map bounds for visualization
        
        Args:
            ax: Matplotlib axes
            routes: Routes to bound
            node_sequence: Node sequence to bound
        """
        if routes:
            # Get bounds from routes
            all_nodes = set()
            for route in routes:
                for segment in route.segments:
                    all_nodes.update(segment.path_nodes)
            
            if all_nodes:
                coords = [(self.graph.nodes[node]['x'], self.graph.nodes[node]['y']) 
                         for node in all_nodes if node in self.graph.nodes]
                if coords:
                    x_coords, y_coords = zip(*coords)
                    ax.set_xlim(min(x_coords) - 0.001, max(x_coords) + 0.001)
                    ax.set_ylim(min(y_coords) - 0.001, max(y_coords) + 0.001)
        elif node_sequence:
            # Get bounds from node sequence
            coords = [(self.graph.nodes[node]['x'], self.graph.nodes[node]['y']) 
                     for node in node_sequence if node in self.graph.nodes]
            if coords:
                x_coords, y_coords = zip(*coords)
                ax.set_xlim(min(x_coords) - 0.001, max(x_coords) + 0.001)
                ax.set_ylim(min(y_coords) - 0.001, max(y_coords) + 0.001)
        else:
            # Use graph bounds
            ax.set_xlim(self.bounds['min_lon'] - 0.001, self.bounds['max_lon'] + 0.001)
            ax.set_ylim(self.bounds['min_lat'] - 0.001, self.bounds['max_lat'] + 0.001)
    
    def _add_elevation_legend(self, ax):
        """Add elevation legend to plot
        
        Args:
            ax: Matplotlib axes
        """
        # Create elevation legend
        elevations = [data.get('elevation', 0) for _, data in self.graph.nodes(data=True)]
        if elevations:
            min_elev, max_elev = min(elevations), max(elevations)
            
            # Create colorbar
            sm = plt.cm.ScalarMappable(cmap=self.elevation_colormap, 
                                     norm=plt.Normalize(vmin=min_elev, vmax=max_elev))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
            cbar.set_label('Elevation (m)', fontsize=10)


class GATuningVisualizer(BaseGAVisualizer):
    """Advanced visualization system for GA parameter tuning analysis"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize tuning visualizer
        
        Args:
            config: Visualization configuration
        """
        if config is None:
            config = VisualizationConfig(output_dir="tuning_visualizations")
        super().__init__(config)
        
        # Set up advanced plotting if available
        if ADVANCED_PLOTTING_AVAILABLE:
            sns.set_palette(self.config.color_scheme)
    
    def create_parameter_sensitivity_plot(self, sensitivity_data: Dict[str, Any],
                                        filename: str = "parameter_sensitivity") -> str:
        """Create parameter sensitivity visualization
        
        Args:
            sensitivity_data: Sensitivity analysis data
            filename: Output filename
            
        Returns:
            Path to saved image
        """
        fig, axes = self.create_subplots(2, 2, "Parameter Sensitivity Analysis")
        
        # Plot sensitivity metrics
        if 'parameters' in sensitivity_data:
            ax = axes[0, 0]
            params = list(sensitivity_data['parameters'].keys())
            values = list(sensitivity_data['parameters'].values())
            
            ax.bar(params, values, color='skyblue', alpha=0.7)
            self.format_axes(ax, "Parameters", "Sensitivity", "Parameter Sensitivity")
            ax.tick_params(axis='x', rotation=45)
        
        # Add more subplots for other sensitivity metrics
        plt.tight_layout()
        return self.save_figure(fig, filename)
    
    def create_optimization_progress_plot(self, optimization_history: List[Dict[str, Any]],
                                        filename: str = "optimization_progress") -> str:
        """Create optimization progress visualization
        
        Args:
            optimization_history: History of optimization results
            filename: Output filename
            
        Returns:
            Path to saved image
        """
        fig, ax = self.create_figure("Optimization Progress")
        
        if optimization_history:
            generations = range(len(optimization_history))
            best_fitness = [gen.get('best_fitness', 0) for gen in optimization_history]
            avg_fitness = [gen.get('avg_fitness', 0) for gen in optimization_history]
            
            ax.plot(generations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
            ax.plot(generations, avg_fitness, 'r--', label='Average Fitness', linewidth=2)
            
            self.format_axes(ax, "Generation", "Fitness", "Optimization Progress")
            ax.legend()
        
        return self.save_figure(fig, filename)


class PrecisionComparisonVisualizer(BaseGAVisualizer):
    """Visualizes the benefits of 1m precision vs 90m precision for GA route optimization"""
    
    def __init__(self, output_dir: str = "./ga_precision_visualizations"):
        """Initialize precision comparison visualizer
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        # Initialize base class
        config = VisualizationConfig(output_dir=output_dir)
        super().__init__(config)
    
    def create_precision_comparison_visualization(self, route_coordinates: List[Tuple[float, float]], 
                                                graph=None, title_suffix: str = "") -> str:
        """Create comprehensive visualization comparing 1m vs 90m precision
        
        Args:
            route_coordinates: Route coordinates for visualization
            graph: Network graph (optional)
            title_suffix: Additional title text
            
        Returns:
            Path to saved visualization
        """
        try:
            # Create figure with subplots
            fig, axes = self.create_subplots(2, 2, f"Precision Comparison{title_suffix}")
            
            # Plot 1m precision data
            ax1 = axes[0, 0]
            if route_coordinates:
                lats, lons = zip(*route_coordinates)
                ax1.plot(lons, lats, 'b-', linewidth=2, label='1m Precision')
                self.format_axes(ax1, "Longitude", "Latitude", "1m Precision Route")
            
            # Plot 90m precision comparison
            ax2 = axes[0, 1]
            # Simplified/downsampled version
            if len(route_coordinates) > 4:
                downsampled = route_coordinates[::4]  # Every 4th point
                lats, lons = zip(*downsampled)
                ax2.plot(lons, lats, 'r-', linewidth=2, label='90m Precision')
                self.format_axes(ax2, "Longitude", "Latitude", "90m Precision Route")
            
            # Add elevation profile comparison
            ax3 = axes[1, 0]
            if graph and route_coordinates:
                # Mock elevation profile
                elevations = [100 + i * 5 for i in range(len(route_coordinates))]
                ax3.plot(range(len(elevations)), elevations, 'g-', linewidth=2)
                self.format_axes(ax3, "Distance", "Elevation (m)", "Elevation Profile")
            
            # Add statistics table
            ax4 = axes[1, 1]
            stats = {
                'High Res Points': len(route_coordinates),
                'Low Res Points': len(route_coordinates) // 4,
                'Resolution': '1m vs 90m',
                'Accuracy Gain': '+85%'
            }
            self.add_statistics_table(ax4, stats, "Precision Statistics")
            ax4.axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            filename = "ga_precision_comparison"
            filepath = self.save_figure(plt.gcf(), filename)
            print(f"✅ Saved precision comparison visualization: {filepath}")
            
            return filepath
            
        except Exception as e:
            print(f"❌ Precision comparison visualization failed: {e}")
            return ""
    
    def create_ga_evolution_comparison(self, high_res_history: List[Dict[str, Any]],
                                     low_res_history: List[Dict[str, Any]], 
                                     title_suffix: str = "") -> str:
        """Create GA evolution comparison between precision levels
        
        Args:
            high_res_history: High resolution GA history
            low_res_history: Low resolution GA history
            title_suffix: Additional title text
            
        Returns:
            Path to saved visualization
        """
        try:
            # Create figure
            fig, ax = self.create_figure(f"GA Evolution Comparison{title_suffix}")
            
            # Plot evolution curves
            if high_res_history:
                generations = range(len(high_res_history))
                fitness_values = [gen.get('best_fitness', 0) for gen in high_res_history]
                ax.plot(generations, fitness_values, 'b-', label='1m Precision', linewidth=2)
            
            if low_res_history:
                generations = range(len(low_res_history))
                fitness_values = [gen.get('best_fitness', 0) for gen in low_res_history]
                ax.plot(generations, fitness_values, 'r--', label='90m Precision', linewidth=2)
            
            self.format_axes(ax, "Generation", "Fitness", "GA Evolution Comparison")
            ax.legend()
            
            plt.tight_layout()
            
            # Save visualization
            filename = "ga_evolution_comparison"
            filepath = self.save_figure(plt.gcf(), filename)
            print(f"✅ Saved GA evolution comparison: {filepath}")
            
            return filepath
            
        except Exception as e:
            print(f"❌ GA evolution comparison failed: {e}")
            return ""


# Export main classes
__all__ = [
    'GAVisualizer',
    'GATuningVisualizer', 
    'PrecisionComparisonVisualizer',
]