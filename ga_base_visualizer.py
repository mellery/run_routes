#!/usr/bin/env python3
"""
Base GA Visualization Class
Provides common functionality for all GA visualization components
"""

import os
import time
import math
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VisualizationConfig:
    """Configuration for visualization generation"""
    output_dir: str = "ga_visualizations"
    figure_format: str = "png"  # png, pdf, svg
    figure_size: Tuple[int, int] = (16, 12)
    dpi: int = 150
    color_scheme: str = "viridis"
    save_data: bool = True
    show_plots: bool = False
    timestamp_files: bool = True


class BaseGAVisualizer:
    """Base class for all GA visualization components"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize base visualizer
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('default')
        
        # Common color schemes
        self.elevation_colormap = LinearSegmentedColormap.from_list(
            'elevation', ['blue', 'green', 'yellow', 'orange', 'red']
        )
        
        # Route colors for population visualization
        self.route_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]
        
        # Color schemes for different precision levels
        self.precision_colors = {
            'high_res': '#1f77b4',      # Blue for 1m data
            'low_res': '#ff7f0e',       # Orange for 90m data
            'micro_features': '#2ca02c', # Green for micro-features
            'elevation_gain': '#d62728'  # Red for elevation highlights
        }
    
    def create_figure(self, title: str = "", figsize: Optional[Tuple[int, int]] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Create a matplotlib figure with consistent styling
        
        Args:
            title: Figure title
            figsize: Figure size (width, height)
            
        Returns:
            Tuple of (figure, axes)
        """
        figsize = figsize or self.config.figure_size
        
        fig, ax = plt.subplots(figsize=figsize, dpi=self.config.dpi)
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold')
        
        return fig, ax
    
    def create_subplots(self, nrows: int, ncols: int, title: str = "", 
                       figsize: Optional[Tuple[int, int]] = None) -> Tuple[plt.Figure, np.ndarray]:
        """Create matplotlib subplots with consistent styling
        
        Args:
            nrows: Number of rows
            ncols: Number of columns
            title: Figure title
            figsize: Figure size (width, height)
            
        Returns:
            Tuple of (figure, axes array)
        """
        figsize = figsize or self.config.figure_size
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=self.config.dpi)
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold')
        
        return fig, axes
    
    def save_figure(self, fig: plt.Figure, filename: str, **kwargs) -> str:
        """Save figure with consistent formatting
        
        Args:
            fig: Matplotlib figure to save
            filename: Base filename (without extension)
            **kwargs: Additional arguments for plt.savefig
            
        Returns:
            Full path to saved file
        """
        # Add timestamp if configured
        if self.config.timestamp_files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp}"
        
        # Create full path
        filepath = self.output_dir / f"{filename}.{self.config.figure_format}"
        
        # Default save parameters
        save_kwargs = {
            'dpi': self.config.dpi,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none'
        }
        save_kwargs.update(kwargs)
        
        # Save figure
        fig.savefig(filepath, **save_kwargs)
        
        if not self.config.show_plots:
            plt.close(fig)
        
        return str(filepath)
    
    def get_route_color(self, index: int, total_routes: int) -> str:
        """Get color for route visualization
        
        Args:
            index: Route index
            total_routes: Total number of routes
            
        Returns:
            Color string
        """
        if total_routes <= len(self.route_colors):
            return self.route_colors[index % len(self.route_colors)]
        else:
            # Generate color based on index for large populations
            hue = (index / total_routes) * 360
            return f"hsl({hue:.0f}, 70%, 60%)"
    
    def get_elevation_color(self, elevation: float, min_elev: float, max_elev: float) -> str:
        """Get color for elevation visualization
        
        Args:
            elevation: Elevation value
            min_elev: Minimum elevation in dataset
            max_elev: Maximum elevation in dataset
            
        Returns:
            Color string
        """
        if max_elev == min_elev:
            return '#888888'
        
        normalized = (elevation - min_elev) / (max_elev - min_elev)
        return self.elevation_colormap(normalized)
    
    def add_statistics_table(self, ax: plt.Axes, stats: Dict[str, Any], 
                           title: str = "Statistics", position: str = "upper right"):
        """Add statistics table to plot
        
        Args:
            ax: Matplotlib axes
            stats: Dictionary of statistics
            title: Table title
            position: Table position
        """
        # Format statistics
        table_data = []
        for key, value in stats.items():
            if isinstance(value, float):
                if abs(value) < 0.01:
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            table_data.append([key.replace('_', ' ').title(), formatted_value])
        
        # Create table
        table = ax.table(cellText=table_data, cellLoc='left', loc=position,
                        colWidths=[0.3, 0.2], cellColours=None)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Style table
        for i in range(len(table_data)):
            table[(i, 0)].set_facecolor('#E8E8E8')
            table[(i, 1)].set_facecolor('#F8F8F8')
    
    def add_legend(self, ax: plt.Axes, items: List[Tuple[str, str]], 
                  title: str = "Legend", position: str = "upper left"):
        """Add legend to plot
        
        Args:
            ax: Matplotlib axes
            items: List of (label, color) tuples
            title: Legend title
            position: Legend position
        """
        handles = []
        labels = []
        
        for label, color in items:
            handles.append(patches.Patch(color=color))
            labels.append(label)
        
        ax.legend(handles, labels, title=title, loc=position, 
                 frameon=True, fancybox=True, shadow=True)
    
    def format_axes(self, ax: plt.Axes, xlabel: str = "", ylabel: str = "", 
                   title: str = "", grid: bool = True):
        """Format axes with consistent styling
        
        Args:
            ax: Matplotlib axes
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Axes title
            grid: Whether to show grid
        """
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        if grid:
            ax.grid(True, alpha=0.3)
        
        # Style ticks
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    def cleanup(self):
        """Clean up resources"""
        plt.close('all')
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


class GAVisualizationUtils:
    """Utility functions for GA visualizations"""
    
    @staticmethod
    def calculate_graph_bounds(graph) -> Dict[str, float]:
        """Calculate geographic bounds of a graph
        
        Args:
            graph: NetworkX graph with node positions
            
        Returns:
            Dictionary with min/max lat/lon bounds
        """
        if not graph.nodes:
            return {'min_lat': 0, 'max_lat': 0, 'min_lon': 0, 'max_lon': 0}
        
        lats = [data.get('y', 0) for _, data in graph.nodes(data=True)]
        lons = [data.get('x', 0) for _, data in graph.nodes(data=True)]
        
        return {
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons),
            'max_lon': max(lons)
        }
    
    @staticmethod
    def calculate_route_statistics(route_nodes: List[int], graph) -> Dict[str, Any]:
        """Calculate statistics for a route
        
        Args:
            route_nodes: List of route node IDs
            graph: NetworkX graph
            
        Returns:
            Dictionary of route statistics
        """
        if not route_nodes or len(route_nodes) < 2:
            return {'distance_km': 0, 'elevation_gain_m': 0, 'elevation_loss_m': 0}
        
        total_distance = 0
        elevation_gain = 0
        elevation_loss = 0
        elevations = []
        
        for i in range(len(route_nodes) - 1):
            node1, node2 = route_nodes[i], route_nodes[i + 1]
            
            # Calculate distance
            if graph.has_edge(node1, node2):
                edge_data = graph[node1][node2]
                total_distance += edge_data.get('length', 0)
            
            # Collect elevations
            if node1 in graph.nodes:
                elevations.append(graph.nodes[node1].get('elevation', 0))
        
        # Add final node elevation
        if route_nodes[-1] in graph.nodes:
            elevations.append(graph.nodes[route_nodes[-1]].get('elevation', 0))
        
        # Calculate elevation changes
        for i in range(len(elevations) - 1):
            elev_change = elevations[i + 1] - elevations[i]
            if elev_change > 0:
                elevation_gain += elev_change
            else:
                elevation_loss += abs(elev_change)
        
        return {
            'distance_km': total_distance / 1000.0,
            'elevation_gain_m': elevation_gain,
            'elevation_loss_m': elevation_loss,
            'net_elevation_m': elevations[-1] - elevations[0] if elevations else 0,
            'max_elevation_m': max(elevations) if elevations else 0,
            'min_elevation_m': min(elevations) if elevations else 0,
            'num_nodes': len(route_nodes)
        }
    
    @staticmethod
    def generate_filename(prefix: str, suffix: str = "", timestamp: bool = True) -> str:
        """Generate filename with optional timestamp
        
        Args:
            prefix: Filename prefix
            suffix: Filename suffix
            timestamp: Whether to include timestamp
            
        Returns:
            Generated filename
        """
        parts = [prefix]
        if suffix:
            parts.append(suffix)
        if timestamp:
            parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        return "_".join(parts)