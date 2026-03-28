#!/usr/bin/env python3
"""
Terrain Profile Plotter
Modern terrain profile visualization for route elevation analysis
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import os
from datetime import datetime


class TerrainProfilePlotter:
    """Modern terrain profile visualization with enhanced features"""
    
    def __init__(self, output_dir: str = "output"):
        """Initialize terrain profile plotter
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot styling
        self.style = {
            'figure_size': (14, 8),
            'dpi': 150,
            'title_fontsize': 16,
            'axis_fontsize': 12,
            'legend_fontsize': 10,
            'grid_alpha': 0.3,
            'line_width': 2.5,
            'fill_alpha': 0.4
        }
        
        # Color scheme
        self.colors = {
            'elevation_line': '#2E86AB',
            'elevation_fill': '#A23B72',
            'elevation_gradient': ['#E8F4FD', '#2E86AB'],
            'climb_marker': '#F18F01',
            'descent_marker': '#C73E1D',
            'waypoint_marker': '#6A994E',
            'background': '#FAFAFA'
        }
    
    def plot_terrain_profile(self, route_result: Dict[str, Any], 
                           profile_data: Dict[str, Any],
                           title: Optional[str] = None,
                           save_filename: Optional[str] = None,
                           show_waypoints: bool = True,
                           show_climb_analysis: bool = True,
                           enhanced_resolution: bool = True) -> str:
        """Create comprehensive terrain profile visualization
        
        Args:
            route_result: Route result from optimizer
            profile_data: Profile data from elevation profiler
            title: Custom title for the plot
            save_filename: Custom filename for saving
            show_waypoints: Whether to show waypoint markers
            show_climb_analysis: Whether to show climb analysis
            enhanced_resolution: Whether to enhance resolution with interpolation
            
        Returns:
            Path to saved plot file
        """
        # Validate input data
        if not profile_data or not profile_data.get('elevations'):
            raise ValueError("No elevation data available for plotting")
        
        # Extract data
        elevations = profile_data['elevations']
        distances_km = profile_data['distances_km']
        coordinates = profile_data.get('coordinates', [])
        
        # Ensure data consistency
        min_length = min(len(elevations), len(distances_km))
        elevations = elevations[:min_length]
        distances_km = distances_km[:min_length]
        
        if min_length < 3:
            raise ValueError("Insufficient data points for terrain profile")
        
        # Enhance resolution if requested
        if enhanced_resolution and min_length < 100:
            elevations, distances_km = self._enhance_resolution(elevations, distances_km)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.style['figure_size'], dpi=self.style['dpi'])
        fig.patch.set_facecolor(self.colors['background'])
        
        # Plot elevation profile with gradient fill
        ax.plot(distances_km, elevations, 
               color=self.colors['elevation_line'], 
               linewidth=self.style['line_width'],
               label='Elevation Profile')
        
        # Add gradient fill
        ax.fill_between(distances_km, elevations, 
                       alpha=self.style['fill_alpha'],
                       color=self.colors['elevation_fill'])
        
        # Add climb analysis if requested
        if show_climb_analysis:
            self._add_climb_analysis(ax, distances_km, elevations)
        
        # Add waypoint markers if requested
        if show_waypoints and coordinates:
            self._add_waypoint_markers(ax, distances_km, elevations, coordinates)
        
        # Customize plot appearance
        self._customize_plot(ax, elevations, distances_km, profile_data, title)
        
        # Add statistics box
        self._add_statistics_box(ax, elevations, distances_km, profile_data)
        
        # Save plot
        if not save_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = f"terrain_profile_{timestamp}.png"
        
        output_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight', 
                   facecolor=self.colors['background'])
        plt.close()
        
        return output_path
    
    def _enhance_resolution(self, elevations: List[float], 
                          distances_km: List[float]) -> Tuple[List[float], List[float]]:
        """Enhance resolution to match 1-meter elevation data precision
        
        Args:
            elevations: Original elevation data
            distances_km: Original distance data
            
        Returns:
            Tuple of enhanced (elevations, distances_km)
        """
        if len(elevations) < 3:
            return elevations, distances_km
        
        total_distance = distances_km[-1]
        
        # Use 1-meter resolution to match our 3DEP elevation data
        # But limit maximum points for performance and visual clarity
        if total_distance <= 2.0:  # Routes ≤ 2km: 1m resolution
            target_spacing_km = 0.001  # 1 meter
            max_points = 2000
        elif total_distance <= 5.0:  # Routes ≤ 5km: 2m resolution  
            target_spacing_km = 0.002  # 2 meters
            max_points = 2500
        elif total_distance <= 10.0:  # Routes ≤ 10km: 5m resolution
            target_spacing_km = 0.005  # 5 meters
            max_points = 2000
        else:  # Long routes: 10m resolution
            target_spacing_km = 0.010  # 10 meters
            max_points = 1000
        
        target_points = int(total_distance / target_spacing_km)
        target_points = min(target_points, max_points)
        
        # Ensure minimum reasonable resolution
        target_points = max(target_points, 100)
        
        # Interpolate to high resolution
        new_distances = np.linspace(0, total_distance, target_points)
        new_elevations = np.interp(new_distances, distances_km, elevations)
        
        return new_elevations.tolist(), new_distances.tolist()
    
    def _add_climb_analysis(self, ax, distances_km: List[float], elevations: List[float]):
        """Add climb analysis markers to the plot
        
        Args:
            ax: Matplotlib axis
            distances_km: Distance data
            elevations: Elevation data
        """
        # Calculate gradients
        gradients = []
        for i in range(1, len(elevations)):
            elev_change = elevations[i] - elevations[i-1]
            dist_change = (distances_km[i] - distances_km[i-1]) * 1000  # Convert to meters
            
            if dist_change > 0:
                gradient = (elev_change / dist_change) * 100  # Percentage
                gradients.append(gradient)
            else:
                gradients.append(0)
        
        # Find significant climbs and descents
        steep_threshold = 5.0  # 5% grade
        
        climb_segments = []
        descent_segments = []
        
        for i, gradient in enumerate(gradients):
            if gradient > steep_threshold:
                climb_segments.append(i + 1)  # +1 because gradients are offset
            elif gradient < -steep_threshold:
                descent_segments.append(i + 1)
        
        # Mark steep climbs
        if climb_segments:
            climb_distances = [distances_km[i] for i in climb_segments]
            climb_elevations = [elevations[i] for i in climb_segments]
            
            ax.scatter(climb_distances, climb_elevations, 
                      color=self.colors['climb_marker'], 
                      s=30, alpha=0.8, marker='^', 
                      label='Steep Climbs (>5%)')
        
        # Mark steep descents
        if descent_segments:
            descent_distances = [distances_km[i] for i in descent_segments]
            descent_elevations = [elevations[i] for i in descent_segments]
            
            ax.scatter(descent_distances, descent_elevations, 
                      color=self.colors['descent_marker'], 
                      s=30, alpha=0.8, marker='v', 
                      label='Steep Descents (<-5%)')
    
    def _add_waypoint_markers(self, ax, distances_km: List[float], 
                            elevations: List[float], coordinates: List[Dict]):
        """Add waypoint markers for key locations
        
        Args:
            ax: Matplotlib axis
            distances_km: Distance data
            elevations: Elevation data
            coordinates: Coordinate data with node information
        """
        # Mark start and end points
        if len(distances_km) > 0:
            # Start point
            ax.scatter(distances_km[0], elevations[0], 
                      color=self.colors['waypoint_marker'], 
                      s=100, marker='o', 
                      edgecolors='white', linewidth=2,
                      label='Start/End', zorder=5)
            
            # End point (if different from start)
            if len(distances_km) > 1 and distances_km[-1] != distances_km[0]:
                ax.scatter(distances_km[-1], elevations[-1], 
                          color=self.colors['waypoint_marker'], 
                          s=100, marker='s', 
                          edgecolors='white', linewidth=2,
                          zorder=5)
        
        # Mark highest and lowest points
        if len(elevations) > 2:
            max_idx = elevations.index(max(elevations))
            min_idx = elevations.index(min(elevations))
            
            # Highest point
            ax.scatter(distances_km[max_idx], elevations[max_idx], 
                      color='gold', s=80, marker='*', 
                      edgecolors='darkgoldenrod', linewidth=1,
                      label='Highest Point', zorder=5)
            
            # Lowest point (if different from highest)
            if min_idx != max_idx:
                ax.scatter(distances_km[min_idx], elevations[min_idx], 
                          color='lightblue', s=80, marker='*', 
                          edgecolors='darkblue', linewidth=1,
                          label='Lowest Point', zorder=5)
    
    def _customize_plot(self, ax, elevations: List[float], 
                       distances_km: List[float], profile_data: Dict[str, Any],
                       title: Optional[str]):
        """Customize plot appearance
        
        Args:
            ax: Matplotlib axis
            elevations: Elevation data
            distances_km: Distance data
            profile_data: Profile data dictionary
            title: Plot title
        """
        # Set labels and title
        ax.set_xlabel('Distance (km)', fontsize=self.style['axis_fontsize'])
        ax.set_ylabel('Elevation (m)', fontsize=self.style['axis_fontsize'])
        
        if title:
            ax.set_title(title, fontsize=self.style['title_fontsize'], pad=20)
        else:
            total_distance = profile_data.get('total_distance_km', distances_km[-1] if distances_km else 0)
            ax.set_title(f'Terrain Profile - {total_distance:.2f}km Route', 
                        fontsize=self.style['title_fontsize'], pad=20)
        
        # Set axis limits with padding
        if elevations:
            min_elev = min(elevations)
            max_elev = max(elevations)
            elev_range = max_elev - min_elev
            padding = max(5, elev_range * 0.1)  # 10% padding or 5m minimum
            ax.set_ylim(min_elev - padding, max_elev + padding)
        
        if distances_km:
            ax.set_xlim(0, max(distances_km) * 1.02)  # 2% padding
        
        # Grid and styling
        ax.grid(True, alpha=self.style['grid_alpha'], linestyle='-', linewidth=0.5)
        ax.set_facecolor('white')
        
        # Legend
        ax.legend(loc='upper left', fontsize=self.style['legend_fontsize'], 
                 framealpha=0.9, fancybox=True, shadow=True)
        
        # Spines styling
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(0.8)
    
    def _add_statistics_box(self, ax, elevations: List[float], 
                          distances_km: List[float], profile_data: Dict[str, Any]):
        """Add statistics box to the plot
        
        Args:
            ax: Matplotlib axis
            elevations: Elevation data
            distances_km: Distance data
            profile_data: Profile data dictionary
        """
        if not elevations or len(elevations) < 2:
            return
        
        # Calculate statistics
        min_elev = min(elevations)
        max_elev = max(elevations)
        elev_range = max_elev - min_elev
        
        # Calculate elevation gain/loss
        elevation_gain = 0
        elevation_loss = 0
        
        for i in range(1, len(elevations)):
            diff = elevations[i] - elevations[i-1]
            if diff > 0:
                elevation_gain += diff
            else:
                elevation_loss += abs(diff)
        
        # Calculate average gradient
        total_distance_m = distances_km[-1] * 1000 if distances_km else 0
        avg_gradient = ((elevations[-1] - elevations[0]) / total_distance_m * 100) if total_distance_m > 0 else 0
        
        # Create statistics text
        stats_text = f"""Route Statistics:
• Distance: {distances_km[-1]:.2f} km
• Elevation Range: {elev_range:.0f} m
• Min/Max: {min_elev:.0f}m / {max_elev:.0f}m
• Total Climb: {elevation_gain:.0f} m
• Total Descent: {elevation_loss:.0f} m
• Avg Gradient: {avg_gradient:+.1f}%
• Data Points: {len(elevations):,}"""
        
        # Add text box
        ax.text(0.02, 0.98, stats_text, 
               transform=ax.transAxes,
               verticalalignment='top',
               fontsize=9,
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='white', 
                        alpha=0.9,
                        edgecolor='#CCCCCC'))
    
    def plot_elevation_comparison(self, route_results: List[Dict[str, Any]], 
                                profile_data_list: List[Dict[str, Any]], 
                                labels: List[str],
                                title: str = "Route Elevation Comparison",
                                save_filename: Optional[str] = None) -> str:
        """Create comparison plot of multiple route elevations
        
        Args:
            route_results: List of route results
            profile_data_list: List of profile data dictionaries
            labels: Labels for each route
            title: Plot title
            save_filename: Custom filename for saving
            
        Returns:
            Path to saved plot file
        """
        if not route_results or not profile_data_list:
            raise ValueError("No route data provided for comparison")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10), dpi=self.style['dpi'])
        fig.patch.set_facecolor(self.colors['background'])
        
        # Color palette for multiple routes
        colors = ['#2E86AB', '#F18F01', '#C73E1D', '#6A994E', '#A23B72']
        
        # Plot each route
        for i, (profile_data, label) in enumerate(zip(profile_data_list, labels)):
            if not profile_data.get('elevations'):
                continue
            
            elevations = profile_data['elevations']
            distances_km = profile_data['distances_km']
            
            # Ensure data consistency
            min_length = min(len(elevations), len(distances_km))
            elevations = elevations[:min_length]
            distances_km = distances_km[:min_length]
            
            color = colors[i % len(colors)]
            
            # Plot line
            ax.plot(distances_km, elevations, 
                   color=color, linewidth=2.5, label=label)
            
            # Add fill with alpha
            ax.fill_between(distances_km, elevations, 
                           alpha=0.2, color=color)
        
        # Customize plot
        ax.set_xlabel('Distance (km)', fontsize=self.style['axis_fontsize'])
        ax.set_ylabel('Elevation (m)', fontsize=self.style['axis_fontsize'])
        ax.set_title(title, fontsize=self.style['title_fontsize'], pad=20)
        
        # Grid and styling
        ax.grid(True, alpha=self.style['grid_alpha'])
        ax.set_facecolor('white')
        
        # Legend
        ax.legend(loc='upper left', fontsize=self.style['legend_fontsize'], 
                 framealpha=0.9, fancybox=True, shadow=True)
        
        # Save plot
        if not save_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = f"elevation_comparison_{timestamp}.png"
        
        output_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight',
                   facecolor=self.colors['background'])
        plt.close()
        
        return output_path


def create_terrain_profile_plot(route_result: Dict[str, Any], 
                               profile_data: Dict[str, Any],
                               output_dir: str = "output",
                               title: Optional[str] = None) -> str:
    """Convenience function to create terrain profile plot
    
    Args:
        route_result: Route result from optimizer
        profile_data: Profile data from elevation profiler
        output_dir: Directory to save plot
        title: Custom title for plot
        
    Returns:
        Path to saved plot file
    """
    plotter = TerrainProfilePlotter(output_dir)
    return plotter.plot_terrain_profile(route_result, profile_data, title=title)