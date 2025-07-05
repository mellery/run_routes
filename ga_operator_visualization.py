#!/usr/bin/env python3
"""
Genetic Algorithm Operator Visualization
Demonstrates and visualizes genetic operators in action
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import networkx as nx
import math
from typing import List, Tuple, Optional
import random

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from ga_operators import GAOperators
from ga_chromosome import RouteChromosome, RouteSegment
from ga_population import PopulationInitializer


class GAOperatorVisualizer:
    """Visualize genetic algorithm operators in action"""
    
    def __init__(self, graph: nx.Graph):
        """Initialize operator visualizer
        
        Args:
            graph: NetworkX graph with elevation and distance data
        """
        self.graph = graph
        self.operators = GAOperators(graph)
        # Use first node as default start node
        start_node = list(graph.nodes())[0] if graph.nodes() else 0
        self.population_init = PopulationInitializer(graph, start_node)
        
    def visualize_crossover_operators(self, save_dir: str = "ga_operator_visualizations"):
        """Visualize crossover operators with before/after comparison"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("🧬 Visualizing Crossover Operators...")
        
        # Create test parents with some overlap
        parent1, parent2 = self._create_test_parents_with_overlap()
        
        # Test segment exchange crossover
        offspring1, offspring2 = self.operators.segment_exchange_crossover(parent1, parent2)
        self._plot_crossover_comparison(
            parent1, parent2, offspring1, offspring2,
            "Segment Exchange Crossover",
            os.path.join(save_dir, "segment_exchange_crossover.png")
        )
        
        # Test path splice crossover
        offspring3, offspring4 = self.operators.path_splice_crossover(parent1, parent2)
        self._plot_crossover_comparison(
            parent1, parent2, offspring3, offspring4,
            "Path Splice Crossover",
            os.path.join(save_dir, "path_splice_crossover.png")
        )
        
        print("✅ Crossover visualization completed")
    
    def visualize_mutation_operators(self, save_dir: str = "ga_operator_visualizations"):
        """Visualize mutation operators with before/after comparison"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("🧬 Visualizing Mutation Operators...")
        
        # Create test chromosome
        original = self._create_test_chromosome()
        
        # Test segment replacement mutation
        mutated1 = self.operators.segment_replacement_mutation(original, mutation_rate=1.0)
        self._plot_mutation_comparison(
            original, mutated1,
            "Segment Replacement Mutation",
            os.path.join(save_dir, "segment_replacement_mutation.png")
        )
        
        # Test route extension mutation
        target_distance = 2.5  # km
        mutated2 = self.operators.route_extension_mutation(original, target_distance, mutation_rate=1.0)
        self._plot_mutation_comparison(
            original, mutated2,
            f"Route Extension Mutation (Target: {target_distance}km)",
            os.path.join(save_dir, "route_extension_mutation.png")
        )
        
        # Test elevation bias mutation
        mutated3 = self.operators.elevation_bias_mutation(original, "elevation", mutation_rate=1.0)
        self._plot_mutation_comparison(
            original, mutated3,
            "Elevation Bias Mutation",
            os.path.join(save_dir, "elevation_bias_mutation.png")
        )
        
        print("✅ Mutation visualization completed")
    
    def visualize_selection_operators(self, save_dir: str = "ga_operator_visualizations"):
        """Visualize selection operators with population examples"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("🧬 Visualizing Selection Operators...")
        
        # Create test population
        population = self._create_test_population(size=20)
        
        # Assign fitness values (simulate evaluation)
        for i, chromo in enumerate(population):
            chromo.fitness = random.uniform(0.1, 1.0)
        
        # Test tournament selection
        tournament_winners = []
        for _ in range(10):
            winner = self.operators.tournament_selection(population, tournament_size=5)
            tournament_winners.append(winner)
        
        # Test elitism selection
        elite = self.operators.elitism_selection(population, elite_size=5)
        
        # Test diversity selection
        diverse = self.operators.diversity_selection(population, selection_size=8)
        
        self._plot_selection_comparison(
            population, tournament_winners, elite, diverse,
            os.path.join(save_dir, "selection_operators.png")
        )
        
        print("✅ Selection visualization completed")
    
    def visualize_operator_effects(self, save_dir: str = "ga_operator_visualizations"):
        """Visualize the cumulative effects of operators over generations"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("🧬 Visualizing Operator Effects Over Generations...")
        
        # Create initial population
        population = self._create_test_population(size=10)
        
        generations_data = []
        for generation in range(5):
            # Assign fitness values
            for chromo in population:
                chromo.fitness = self._calculate_test_fitness(chromo)
                chromo.generation = generation
            
            generations_data.append({
                'generation': generation,
                'population': [chromo.copy() for chromo in population],
                'avg_fitness': np.mean([c.fitness for c in population]),
                'avg_distance': np.mean([c.get_total_distance() / 1000 for c in population]),
                'avg_elevation': np.mean([c.get_elevation_gain() for c in population])
            })
            
            if generation < 4:  # Don't evolve the last generation
                # Evolve population
                new_population = []
                
                # Elitism
                elite = self.operators.elitism_selection(population, elite_size=2)
                new_population.extend([chromo.copy() for chromo in elite])
                
                # Crossover and mutation
                while len(new_population) < len(population):
                    parent1 = self.operators.tournament_selection(population)
                    parent2 = self.operators.tournament_selection(population)
                    
                    offspring1, offspring2 = self.operators.segment_exchange_crossover(parent1, parent2)
                    offspring1 = self.operators.segment_replacement_mutation(offspring1, mutation_rate=0.3)
                    offspring2 = self.operators.segment_replacement_mutation(offspring2, mutation_rate=0.3)
                    
                    new_population.extend([offspring1, offspring2])
                
                population = new_population[:len(population)]
        
        self._plot_generation_evolution(
            generations_data,
            os.path.join(save_dir, "operator_effects_over_generations.png")
        )
        
        print("✅ Operator effects visualization completed")
    
    def _create_test_parents_with_overlap(self) -> Tuple[RouteChromosome, RouteChromosome]:
        """Create test parents with some overlapping nodes"""
        # Get nodes from graph
        nodes = list(self.graph.nodes())
        if len(nodes) < 8:
            raise ValueError("Graph needs at least 8 nodes for crossover testing")
        
        # Create parent 1: path through nodes 0,1,2,3
        segments1 = []
        for i in range(min(3, len(nodes) - 1)):
            segment = self._create_test_segment(nodes[i], nodes[i + 1])
            if segment:
                segments1.append(segment)
        
        # Create parent 2: path through nodes 1,2,4,5 (overlaps at 1,2)
        segments2 = []
        node_indices = [1, 2, min(4, len(nodes) - 1), min(5, len(nodes) - 1)]
        for i in range(len(node_indices) - 1):
            if node_indices[i + 1] < len(nodes):
                segment = self._create_test_segment(nodes[node_indices[i]], nodes[node_indices[i + 1]])
                if segment:
                    segments2.append(segment)
        
        parent1 = RouteChromosome(segments1)
        parent1.fitness = 0.8
        parent2 = RouteChromosome(segments2)
        parent2.fitness = 0.6
        
        return parent1, parent2
    
    def _create_test_chromosome(self) -> RouteChromosome:
        """Create a test chromosome for mutation testing"""
        # Use population initializer to create a realistic chromosome
        chromosomes = self.population_init.create_population(
            size=1,
            target_distance_km=2.0
        )
        return chromosomes[0] if chromosomes else RouteChromosome([])
    
    def _create_test_population(self, size: int) -> List[RouteChromosome]:
        """Create a test population"""
        return self.population_init.create_population(
            size=size,
            target_distance_km=2.0
        )
    
    def _create_test_segment(self, start_node: int, end_node: int) -> Optional[RouteSegment]:
        """Create a test segment between nodes"""
        return self.operators._create_segment(start_node, end_node)
    
    def _calculate_test_fitness(self, chromosome: RouteChromosome) -> float:
        """Calculate test fitness for visualization purposes"""
        if not chromosome.segments:
            return 0.0
        
        distance_km = chromosome.get_total_distance() / 1000
        elevation_gain = chromosome.get_elevation_gain()
        
        # Simple fitness function for testing
        distance_score = max(0, 1.0 - abs(distance_km - 2.0) / 2.0)  # Prefer ~2km routes
        elevation_score = min(elevation_gain / 100.0, 1.0)  # Prefer some elevation
        
        return (distance_score + elevation_score) / 2.0
    
    def _plot_crossover_comparison(self, parent1: RouteChromosome, parent2: RouteChromosome,
                                  offspring1: RouteChromosome, offspring2: RouteChromosome,
                                  title: str, filename: str):
        """Plot crossover before/after comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Plot parents
        self._plot_route_on_axis(axes[0, 0], parent1, "Parent 1", color='blue')
        self._plot_route_on_axis(axes[0, 1], parent2, "Parent 2", color='red')
        
        # Plot offspring
        self._plot_route_on_axis(axes[1, 0], offspring1, "Offspring 1", color='green')
        self._plot_route_on_axis(axes[1, 1], offspring2, "Offspring 2", color='purple')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  💾 Saved: {filename}")
    
    def _plot_mutation_comparison(self, original: RouteChromosome, mutated: RouteChromosome,
                                 title: str, filename: str):
        """Plot mutation before/after comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=16)
        
        # Plot original
        self._plot_route_on_axis(axes[0], original, "Original", color='blue')
        
        # Plot mutated
        self._plot_route_on_axis(axes[1], mutated, "Mutated", color='red')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  💾 Saved: {filename}")
    
    def _plot_selection_comparison(self, population: List[RouteChromosome],
                                  tournament_winners: List[RouteChromosome],
                                  elite: List[RouteChromosome],
                                  diverse: List[RouteChromosome],
                                  filename: str):
        """Plot selection results comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Selection Operators Comparison", fontsize=16)
        
        # Plot population fitness distribution
        fitnesses = [c.fitness for c in population]
        axes[0, 0].hist(fitnesses, bins=10, alpha=0.7, color='gray')
        axes[0, 0].set_title("Population Fitness Distribution")
        axes[0, 0].set_xlabel("Fitness")
        axes[0, 0].set_ylabel("Count")
        
        # Plot tournament winners
        tournament_fitnesses = [c.fitness for c in tournament_winners]
        axes[0, 1].hist(tournament_fitnesses, bins=10, alpha=0.7, color='green')
        axes[0, 1].set_title("Tournament Selection Results")
        axes[0, 1].set_xlabel("Fitness")
        axes[0, 1].set_ylabel("Count")
        
        # Plot elite
        elite_fitnesses = [c.fitness for c in elite]
        distances = [c.get_total_distance() / 1000 for c in population]
        elite_distances = [c.get_total_distance() / 1000 for c in elite]
        
        axes[1, 0].scatter(distances, fitnesses, alpha=0.5, color='gray', label='Population')
        axes[1, 0].scatter(elite_distances, elite_fitnesses, color='red', s=100, label='Elite')
        axes[1, 0].set_title("Elite Selection Results")
        axes[1, 0].set_xlabel("Distance (km)")
        axes[1, 0].set_ylabel("Fitness")
        axes[1, 0].legend()
        
        # Plot diversity selection
        diverse_fitnesses = [c.fitness for c in diverse]
        diverse_distances = [c.get_total_distance() / 1000 for c in diverse]
        
        axes[1, 1].scatter(distances, fitnesses, alpha=0.5, color='gray', label='Population')
        axes[1, 1].scatter(diverse_distances, diverse_fitnesses, color='blue', s=100, label='Diverse')
        axes[1, 1].set_title("Diversity Selection Results")
        axes[1, 1].set_xlabel("Distance (km)")
        axes[1, 1].set_ylabel("Fitness")
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  💾 Saved: {filename}")
    
    def _plot_generation_evolution(self, generations_data: List[dict], filename: str):
        """Plot evolution over generations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Genetic Operators Effects Over Generations", fontsize=16)
        
        generations = [data['generation'] for data in generations_data]
        avg_fitness = [data['avg_fitness'] for data in generations_data]
        avg_distance = [data['avg_distance'] for data in generations_data]
        avg_elevation = [data['avg_elevation'] for data in generations_data]
        
        # Average fitness over generations
        axes[0, 0].plot(generations, avg_fitness, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_title("Average Fitness Evolution")
        axes[0, 0].set_xlabel("Generation")
        axes[0, 0].set_ylabel("Average Fitness")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Average distance over generations
        axes[0, 1].plot(generations, avg_distance, 'ro-', linewidth=2, markersize=8)
        axes[0, 1].set_title("Average Distance Evolution")
        axes[0, 1].set_xlabel("Generation")
        axes[0, 1].set_ylabel("Average Distance (km)")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Average elevation over generations
        axes[1, 0].plot(generations, avg_elevation, 'go-', linewidth=2, markersize=8)
        axes[1, 0].set_title("Average Elevation Gain Evolution")
        axes[1, 0].set_xlabel("Generation")
        axes[1, 0].set_ylabel("Average Elevation Gain (m)")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Population diversity over generations
        diversities = []
        for data in generations_data:
            population = data['population']
            if population:
                distances = [c.get_total_distance() / 1000 for c in population]
                diversity = np.std(distances)
                diversities.append(diversity)
            else:
                diversities.append(0)
        
        axes[1, 1].plot(generations, diversities, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_title("Population Diversity Evolution")
        axes[1, 1].set_xlabel("Generation")
        axes[1, 1].set_ylabel("Distance Std Dev (km)")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  💾 Saved: {filename}")
    
    def _plot_route_on_axis(self, ax, chromosome: RouteChromosome, title: str, color: str = 'blue'):
        """Plot a route on a given axis with OpenStreetMap background"""
        if not chromosome.segments:
            ax.text(0.5, 0.5, "Empty Route", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        # Plot OpenStreetMap background
        use_mercator = self._plot_network_background(ax, [chromosome])
        
        # Plot chromosome route with proper coordinate transformation
        self._plot_chromosome_route(ax, chromosome, color=color, use_mercator=use_mercator)
        
        # Add route info
        stats = chromosome.get_route_stats()
        info_text = (f"Distance: {stats['total_distance_km']:.2f}km\\n"
                    f"Elevation: {stats['total_elevation_gain_m']:.1f}m\\n"
                    f"Segments: {len(chromosome.segments)}\\n"
                    f"Method: {chromosome.creation_method}")
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=8,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    def _plot_network_background(self, ax, routes=None, use_osm=True) -> bool:
        """Plot network background with optional OpenStreetMap basemap"""
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
    
    def _plot_chromosome_route(self, ax, chromosome: RouteChromosome, color: str = 'blue', 
                              use_mercator: bool = False) -> None:
        """Plot chromosome route on axis with proper coordinate transformation"""
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
            
            if len(x_coords) < 2:
                continue
            
            # Plot segment
            ax.plot(x_coords, y_coords, color=color, alpha=0.8, linewidth=3, zorder=10)
        
        # Mark start/end points
        if chromosome.segments:
            start_node = chromosome.segments[0].start_node
            end_node = chromosome.segments[-1].end_node
            
            if start_node in self.graph.nodes and end_node in self.graph.nodes:
                start_lon = self.graph.nodes[start_node]['x']
                start_lat = self.graph.nodes[start_node]['y']
                end_lon = self.graph.nodes[end_node]['x']
                end_lat = self.graph.nodes[end_node]['y']
                
                if use_mercator and transformer:
                    start_x, start_y = transformer.transform(start_lon, start_lat)
                    end_x, end_y = transformer.transform(end_lon, end_lat)
                else:
                    start_x, start_y = start_lon, start_lat
                    end_x, end_y = end_lon, end_lat
                
                ax.scatter(start_x, start_y, c='green', s=100, zorder=20, 
                          edgecolors='white', linewidth=2, label='Start')
                ax.scatter(end_x, end_y, c='red', s=100, zorder=20, marker='s',
                          edgecolors='white', linewidth=2, label='End')
                
                # Only add legend to the first plot to avoid duplication
                if not ax.get_legend():
                    ax.legend(loc='upper right')


def create_test_graph() -> nx.Graph:
    """Create a test graph for operator visualization"""
    graph = nx.Graph()
    
    # Create a grid of nodes with realistic coordinates and elevations
    base_lat, base_lon = 37.1299, -80.4094
    
    nodes = []
    for i in range(5):
        for j in range(5):
            node_id = i * 5 + j
            lat = base_lat + i * 0.002
            lon = base_lon + j * 0.002
            elevation = 100 + (i + j) * 10 + np.random.normal(0, 5)
            
            graph.add_node(node_id, x=lon, y=lat, elevation=elevation)
            nodes.append(node_id)
    
    # Add edges to create a connected graph
    for i in range(5):
        for j in range(5):
            node_id = i * 5 + j
            
            # Connect to right neighbor
            if j < 4:
                right_neighbor = i * 5 + (j + 1)
                distance = 150 + np.random.normal(0, 20)
                graph.add_edge(node_id, right_neighbor, length=distance)
            
            # Connect to bottom neighbor
            if i < 4:
                bottom_neighbor = (i + 1) * 5 + j
                distance = 150 + np.random.normal(0, 20)
                graph.add_edge(node_id, bottom_neighbor, length=distance)
            
            # Add some diagonal connections for variety
            if i < 4 and j < 4 and np.random.random() < 0.3:
                diagonal_neighbor = (i + 1) * 5 + (j + 1)
                distance = 200 + np.random.normal(0, 30)
                graph.add_edge(node_id, diagonal_neighbor, length=distance)
    
    return graph


def main():
    """Main visualization function"""
    print("🎨 GA Operator Visualization Tool")
    print("=" * 50)
    
    # Try to use real network data first, fallback to test graph
    try:
        print("📊 Loading real network data...")
        from route_services.network_manager import NetworkManager
        
        network_manager = NetworkManager()
        graph = network_manager.load_network()
        
        if graph and len(graph.nodes) > 100:
            print(f"✅ Loaded real network: {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        else:
            raise Exception("Network too small or failed to load")
            
    except Exception as e:
        print(f"⚠️ Real network loading failed ({e}), using test graph...")
        graph = create_test_graph()
        print(f"✅ Created test graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    # Create visualizer
    visualizer = GAOperatorVisualizer(graph)
    
    # Create output directory
    output_dir = "ga_operator_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Generate all visualizations
        visualizer.visualize_crossover_operators(output_dir)
        visualizer.visualize_mutation_operators(output_dir)
        visualizer.visualize_selection_operators(output_dir)
        visualizer.visualize_operator_effects(output_dir)
        
        print("\\n🎉 All operator visualizations completed!")
        print(f"📁 Files saved to: {output_dir}/")
        print("\\n📋 Generated files:")
        for filename in os.listdir(output_dir):
            if filename.endswith('.png'):
                print(f"  • {filename}")
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()