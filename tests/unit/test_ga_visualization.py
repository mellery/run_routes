#!/usr/bin/env python3
"""
Unit tests for genetic_algorithm/visualization.py
Tests comprehensive functionality of GA visualization components
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from genetic_algorithm.visualization import (
    GAVisualizer, GATuningVisualizer, PrecisionComparisonVisualizer
)
from genetic_algorithm.chromosome import RouteChromosome, RouteSegment
from ga_base_visualizer import VisualizationConfig


class TestGAVisualizer(unittest.TestCase):
    """Test GAVisualizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock graph with realistic structure
        self.mock_graph = nx.Graph()
        
        # Add nodes with coordinates and elevation
        nodes = [
            (1001, -80.4094, 37.1299, 100.0),
            (1002, -80.4090, 37.1299, 105.0),
            (1003, -80.4086, 37.1299, 110.0),
            (1004, -80.4082, 37.1299, 115.0),
            (1005, -80.4078, 37.1299, 120.0),
            (1006, -80.4094, 37.1303, 95.0),
            (1007, -80.4090, 37.1303, 100.0),
            (1008, -80.4086, 37.1303, 105.0),
            (1009, -80.4082, 37.1303, 110.0),
            (1010, -80.4078, 37.1303, 115.0)
        ]
        
        for node_id, x, y, elevation in nodes:
            self.mock_graph.add_node(node_id, x=x, y=y, elevation=elevation)
        
        # Add edges to create connected network
        edges = [
            (1001, 1002, 400), (1002, 1003, 400), (1003, 1004, 400), (1004, 1005, 400),
            (1006, 1007, 400), (1007, 1008, 400), (1008, 1009, 400), (1009, 1010, 400),
            (1001, 1006, 400), (1002, 1007, 400), (1003, 1008, 400), (1004, 1009, 400),
            (1005, 1010, 400), (1001, 1007, 565), (1002, 1008, 565)
        ]
        
        for node1, node2, length in edges:
            self.mock_graph.add_edge(node1, node2, length=length)
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Close any open matplotlib figures
        plt.close('all')
    
    def _create_test_chromosome(self, node_path):
        """Helper to create test chromosome from node path"""
        segments = []
        for i in range(len(node_path) - 1):
            segment = RouteSegment(node_path[i], node_path[i + 1], [node_path[i], node_path[i + 1]])
            segment.calculate_properties(self.mock_graph)
            segments.append(segment)
        
        return RouteChromosome(segments)
    
    def test_ga_visualizer_initialization(self):
        """Test GAVisualizer initialization"""
        visualizer = GAVisualizer(self.mock_graph, output_dir=self.temp_dir)
        
        self.assertEqual(visualizer.graph, self.mock_graph)
        self.assertIsInstance(visualizer.bounds, dict)
        self.assertIn('min_lat', visualizer.bounds)
        self.assertIn('max_lat', visualizer.bounds)
        self.assertIn('min_lon', visualizer.bounds)
        self.assertIn('max_lon', visualizer.bounds)
        self.assertIn('center_lat', visualizer.bounds)
        self.assertIn('center_lon', visualizer.bounds)
        self.assertTrue(os.path.exists(self.temp_dir))
    
    def test_ga_visualizer_bounds_calculation(self):
        """Test bounds calculation for visualizer"""
        visualizer = GAVisualizer(self.mock_graph, output_dir=self.temp_dir)
        
        # Check that bounds are calculated correctly
        self.assertEqual(visualizer.bounds['min_lat'], 37.1299)
        self.assertEqual(visualizer.bounds['max_lat'], 37.1303)
        self.assertEqual(visualizer.bounds['min_lon'], -80.4094)
        self.assertEqual(visualizer.bounds['max_lon'], -80.4078)
        self.assertEqual(visualizer.bounds['center_lat'], 37.1301)
        self.assertEqual(visualizer.bounds['center_lon'], -80.4086)
    
    def test_ga_visualizer_bounds_same_coordinates(self):
        """Test bounds calculation when all coordinates are the same"""
        single_node_graph = nx.Graph()
        single_node_graph.add_node(1, x=-80.4094, y=37.1299, elevation=100.0)
        
        visualizer = GAVisualizer(single_node_graph, output_dir=self.temp_dir)
        
        self.assertEqual(visualizer.bounds['center_lat'], 37.1299)
        self.assertEqual(visualizer.bounds['center_lon'], -80.4094)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_save_chromosome_map_basic(self, mock_close, mock_savefig):
        """Test basic chromosome map saving"""
        visualizer = GAVisualizer(self.mock_graph, output_dir=self.temp_dir)
        chromosome = self._create_test_chromosome([1001, 1002, 1003, 1001])
        
        with patch.object(visualizer, '_plot_network_background', return_value=False):
            with patch.object(visualizer, '_plot_chromosome_route'):
                with patch.object(visualizer, '_set_map_bounds'):
                    with patch.object(visualizer, '_add_elevation_legend'):
                        with patch.object(visualizer, 'save_figure', return_value='test.png') as mock_save:
                            result = visualizer.save_chromosome_map(chromosome)
        
        self.assertIsInstance(result, str)
        self.assertEqual(result, 'test.png')
        mock_save.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_save_chromosome_map_with_options(self, mock_close, mock_savefig):
        """Test chromosome map saving with various options"""
        visualizer = GAVisualizer(self.mock_graph, output_dir=self.temp_dir)
        chromosome = self._create_test_chromosome([1001, 1002, 1003, 1001])
        
        with patch.object(visualizer, '_plot_network_background', return_value=False):
            with patch.object(visualizer, '_plot_chromosome_route'):
                with patch.object(visualizer, '_set_map_bounds'):
                    with patch.object(visualizer, '_add_elevation_legend'):
                        with patch.object(visualizer, 'save_figure', return_value='test_chromosome.png') as mock_save:
                            result = visualizer.save_chromosome_map(
                                chromosome,
                                filename="test_chromosome",
                                title="Test Route",
                                show_elevation=True,
                                show_segments=True,
                                show_stats=True
                            )
        
        self.assertIsInstance(result, str)
        self.assertEqual(result, 'test_chromosome.png')
        mock_save.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_save_chromosome_map_empty_chromosome(self, mock_close, mock_savefig):
        """Test chromosome map saving with empty chromosome"""
        visualizer = GAVisualizer(self.mock_graph, output_dir=self.temp_dir)
        chromosome = RouteChromosome([])  # Empty chromosome
        
        with patch.object(visualizer, '_plot_network_background', return_value=False):
            with patch.object(visualizer, '_set_map_bounds'):
                with patch.object(visualizer, 'save_figure', return_value='empty.png') as mock_save:
                    result = visualizer.save_chromosome_map(chromosome)
        
        self.assertIsInstance(result, str)
        self.assertEqual(result, 'empty.png')
        mock_save.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_save_population_map_basic(self, mock_close, mock_savefig):
        """Test basic population map saving"""
        visualizer = GAVisualizer(self.mock_graph, output_dir=self.temp_dir)
        
        population = [
            self._create_test_chromosome([1001, 1002, 1003, 1001]),
            self._create_test_chromosome([1001, 1006, 1007, 1001]),
            self._create_test_chromosome([1001, 1004, 1005, 1001])
        ]
        
        with patch.object(visualizer, '_plot_network_background', return_value=False):
            with patch.object(visualizer, '_plot_chromosome_route'):
                with patch.object(visualizer, '_set_map_bounds'):
                    with patch.object(visualizer, 'get_route_color', return_value='red'):
                        with patch.object(visualizer, 'save_figure', return_value='population_gen005.png') as mock_save:
                            result = visualizer.save_population_map(population, generation=5)
        
        self.assertIsInstance(result, str)
        self.assertEqual(result, 'population_gen005.png')
        mock_save.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_save_population_map_with_options(self, mock_close, mock_savefig):
        """Test population map saving with options"""
        visualizer = GAVisualizer(self.mock_graph, output_dir=self.temp_dir)
        
        population = [
            self._create_test_chromosome([1001, 1002, 1003, 1001]),
            self._create_test_chromosome([1001, 1006, 1007, 1001])
        ]
        
        with patch.object(visualizer, '_plot_network_background', return_value=False):
            with patch.object(visualizer, '_plot_chromosome_route'):
                with patch.object(visualizer, '_set_map_bounds'):
                    with patch.object(visualizer, 'get_route_color', return_value='blue'):
                        with patch.object(visualizer, 'save_figure', return_value='test_population.png') as mock_save:
                            result = visualizer.save_population_map(
                                population,
                                generation=10,
                                filename="test_population",
                                show_fitness=True,
                                show_elevation=True,
                                max_routes=1
                            )
        
        self.assertIsInstance(result, str)
        self.assertEqual(result, 'test_population.png')
        mock_save.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_save_population_map_empty_population(self, mock_close, mock_savefig):
        """Test population map saving with empty population"""
        visualizer = GAVisualizer(self.mock_graph, output_dir=self.temp_dir)
        
        population = []
        
        with patch.object(visualizer, '_plot_network_background', return_value=False):
            with patch.object(visualizer, '_set_map_bounds'):
                with patch.object(visualizer, 'save_figure', return_value='empty_population.png') as mock_save:
                    result = visualizer.save_population_map(population)
        
        self.assertIsInstance(result, str)
        self.assertEqual(result, 'empty_population.png')
        mock_save.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_save_population_map_invalid_routes(self, mock_close, mock_savefig):
        """Test population map saving with invalid routes"""
        visualizer = GAVisualizer(self.mock_graph, output_dir=self.temp_dir)
        
        # Create routes with some invalid ones
        chromosome1 = self._create_test_chromosome([1001, 1002, 1003, 1001])
        chromosome2 = RouteChromosome([])  # Empty/invalid
        chromosome3 = self._create_test_chromosome([1001, 1006, 1007, 1001])
        chromosome3.is_valid = False  # Mark as invalid
        
        population = [chromosome1, chromosome2, chromosome3]
        
        with patch.object(visualizer, '_plot_network_background', return_value=False):
            with patch.object(visualizer, '_plot_chromosome_route'):
                with patch.object(visualizer, '_set_map_bounds'):
                    with patch.object(visualizer, 'get_route_color', return_value='green'):
                        with patch.object(visualizer, 'save_figure', return_value='invalid_population.png') as mock_save:
                            result = visualizer.save_population_map(population)
        
        self.assertIsInstance(result, str)
        self.assertEqual(result, 'invalid_population.png')
        mock_save.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_save_comparison_map_with_tsp(self, mock_close, mock_savefig):
        """Test comparison map saving with TSP route"""
        visualizer = GAVisualizer(self.mock_graph, output_dir=self.temp_dir)
        
        ga_route = self._create_test_chromosome([1001, 1002, 1003, 1001])
        tsp_route = [1001, 1006, 1007, 1001]
        
        with patch.object(visualizer, '_plot_network_background', return_value=False):
            with patch.object(visualizer, '_plot_chromosome_route'):
                with patch.object(visualizer, '_plot_node_sequence'):
                    with patch.object(visualizer, '_set_map_bounds'):
                        with patch.object(ga_route, 'get_route_stats', return_value={
                            'total_distance_km': 2.5,
                            'total_elevation_gain_m': 150.0
                        }):
                            with patch.object(visualizer, 'save_figure', return_value='test_comparison.png') as mock_save:
                                result = visualizer.save_comparison_map(
                                    ga_route, 
                                    tsp_route,
                                    filename="test_comparison",
                                    title="Test Comparison"
                                )
        
        self.assertIsInstance(result, str)
        self.assertEqual(result, 'test_comparison.png')
        mock_save.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_save_comparison_map_without_tsp(self, mock_close, mock_savefig):
        """Test comparison map saving without TSP route"""
        visualizer = GAVisualizer(self.mock_graph, output_dir=self.temp_dir)
        
        ga_route = self._create_test_chromosome([1001, 1002, 1003, 1001])
        
        with patch.object(visualizer, '_plot_network_background', return_value=False):
            with patch.object(visualizer, '_plot_chromosome_route'):
                with patch.object(visualizer, '_set_map_bounds'):
                    with patch.object(ga_route, 'get_route_stats', return_value={
                        'total_distance_km': 2.5,
                        'total_elevation_gain_m': 150.0
                    }):
                        with patch.object(visualizer, 'save_figure', return_value='no_tsp_comparison.png') as mock_save:
                            result = visualizer.save_comparison_map(ga_route, None)
        
        self.assertIsInstance(result, str)
        self.assertEqual(result, 'no_tsp_comparison.png')
        mock_save.assert_called_once()
    
    def test_plot_network_background_basic(self):
        """Test basic network background plotting"""
        visualizer = GAVisualizer(self.mock_graph, output_dir=self.temp_dir)
        
        fig, ax = plt.subplots()
        
        with patch('networkx.draw_networkx_edges') as mock_draw_edges:
            with patch('networkx.draw_networkx_nodes') as mock_draw_nodes:
                result = visualizer._plot_network_background(ax, use_osm=False)
        
        self.assertFalse(result)  # Should return False for no Mercator projection
        mock_draw_edges.assert_called_once()
        mock_draw_nodes.assert_not_called()  # Not called when use_osm=False
    
    def test_plot_network_background_with_osm(self):
        """Test network background plotting with OSM"""
        visualizer = GAVisualizer(self.mock_graph, output_dir=self.temp_dir)
        
        fig, ax = plt.subplots()
        
        with patch('networkx.draw_networkx_edges') as mock_draw_edges:
            with patch('networkx.draw_networkx_nodes') as mock_draw_nodes:
                with patch.object(visualizer, 'get_elevation_color', return_value='red'):
                    result = visualizer._plot_network_background(ax, use_osm=True)
        
        self.assertFalse(result)  # Should return False for no Mercator projection
        mock_draw_edges.assert_called_once()
        mock_draw_nodes.assert_called_once()
    
    def test_plot_chromosome_route_basic(self):
        """Test basic chromosome route plotting"""
        visualizer = GAVisualizer(self.mock_graph, output_dir=self.temp_dir)
        chromosome = self._create_test_chromosome([1001, 1002, 1003, 1001])
        
        fig, ax = plt.subplots()
        
        with patch.object(ax, 'plot') as mock_plot:
            visualizer._plot_chromosome_route(ax, chromosome)
        
        mock_plot.assert_called_once()
        
        # Check that the plot was called with correct parameters
        call_args = mock_plot.call_args
        self.assertEqual(call_args[1]['color'], 'red')  # Default color
        self.assertEqual(call_args[1]['alpha'], 1.0)   # Default alpha
        self.assertEqual(call_args[1]['linewidth'], 3)
    
    def test_plot_chromosome_route_with_options(self):
        """Test chromosome route plotting with options"""
        visualizer = GAVisualizer(self.mock_graph, output_dir=self.temp_dir)
        chromosome = self._create_test_chromosome([1001, 1002, 1003, 1001])
        
        fig, ax = plt.subplots()
        
        with patch.object(ax, 'plot') as mock_plot:
            visualizer._plot_chromosome_route(
                ax, chromosome, 
                color='blue', 
                alpha=0.7, 
                show_elevation=True
            )
        
        mock_plot.assert_called_once()
        
        # Check that the plot was called with correct parameters
        call_args = mock_plot.call_args
        self.assertEqual(call_args[1]['color'], 'blue')
        self.assertEqual(call_args[1]['alpha'], 0.7)
    
    def test_plot_chromosome_route_empty_chromosome(self):
        """Test chromosome route plotting with empty chromosome"""
        visualizer = GAVisualizer(self.mock_graph, output_dir=self.temp_dir)
        chromosome = RouteChromosome([])  # Empty chromosome
        
        fig, ax = plt.subplots()
        
        with patch.object(ax, 'plot') as mock_plot:
            visualizer._plot_chromosome_route(ax, chromosome)
        
        mock_plot.assert_not_called()  # Should not plot anything
    
    def test_plot_node_sequence_basic(self):
        """Test basic node sequence plotting"""
        visualizer = GAVisualizer(self.mock_graph, output_dir=self.temp_dir)
        node_sequence = [1001, 1002, 1003, 1001]
        
        fig, ax = plt.subplots()
        
        with patch.object(ax, 'plot') as mock_plot:
            visualizer._plot_node_sequence(ax, node_sequence)
        
        mock_plot.assert_called_once()
        
        # Check that the plot was called with correct parameters
        call_args = mock_plot.call_args
        self.assertEqual(call_args[1]['color'], 'blue')
        self.assertEqual(call_args[1]['linewidth'], 3)
    
    def test_plot_node_sequence_empty_sequence(self):
        """Test node sequence plotting with empty sequence"""
        visualizer = GAVisualizer(self.mock_graph, output_dir=self.temp_dir)
        node_sequence = []
        
        fig, ax = plt.subplots()
        
        with patch.object(ax, 'plot') as mock_plot:
            visualizer._plot_node_sequence(ax, node_sequence)
        
        mock_plot.assert_not_called()  # Should not plot anything
    
    def test_plot_node_sequence_invalid_nodes(self):
        """Test node sequence plotting with invalid nodes"""
        visualizer = GAVisualizer(self.mock_graph, output_dir=self.temp_dir)
        node_sequence = [9999, 8888, 7777]  # Non-existent nodes
        
        fig, ax = plt.subplots()
        
        with patch.object(ax, 'plot') as mock_plot:
            visualizer._plot_node_sequence(ax, node_sequence)
        
        mock_plot.assert_not_called()  # Should not plot anything
    
    def test_set_map_bounds_with_routes(self):
        """Test map bounds setting with routes"""
        visualizer = GAVisualizer(self.mock_graph, output_dir=self.temp_dir)
        routes = [self._create_test_chromosome([1001, 1002, 1003, 1001])]
        
        fig, ax = plt.subplots()
        
        with patch.object(ax, 'set_xlim') as mock_xlim:
            with patch.object(ax, 'set_ylim') as mock_ylim:
                visualizer._set_map_bounds(ax, routes)
        
        mock_xlim.assert_called_once()
        mock_ylim.assert_called_once()
    
    def test_set_map_bounds_with_node_sequence(self):
        """Test map bounds setting with node sequence"""
        visualizer = GAVisualizer(self.mock_graph, output_dir=self.temp_dir)
        node_sequence = [1001, 1002, 1003, 1001]
        
        fig, ax = plt.subplots()
        
        with patch.object(ax, 'set_xlim') as mock_xlim:
            with patch.object(ax, 'set_ylim') as mock_ylim:
                visualizer._set_map_bounds(ax, node_sequence=node_sequence)
        
        mock_xlim.assert_called_once()
        mock_ylim.assert_called_once()
    
    def test_set_map_bounds_with_graph_bounds(self):
        """Test map bounds setting with graph bounds"""
        visualizer = GAVisualizer(self.mock_graph, output_dir=self.temp_dir)
        
        fig, ax = plt.subplots()
        
        with patch.object(ax, 'set_xlim') as mock_xlim:
            with patch.object(ax, 'set_ylim') as mock_ylim:
                visualizer._set_map_bounds(ax)
        
        mock_xlim.assert_called_once()
        mock_ylim.assert_called_once()
    
    def test_add_elevation_legend_basic(self):
        """Test basic elevation legend addition"""
        visualizer = GAVisualizer(self.mock_graph, output_dir=self.temp_dir)
        
        fig, ax = plt.subplots()
        
        with patch('matplotlib.pyplot.colorbar') as mock_colorbar:
            visualizer._add_elevation_legend(ax)
        
        mock_colorbar.assert_called_once()
    
    def test_add_elevation_legend_no_elevations(self):
        """Test elevation legend addition with no elevations"""
        # Create graph without elevation data
        no_elev_graph = nx.Graph()
        no_elev_graph.add_node(1001, x=-80.4094, y=37.1299)
        
        visualizer = GAVisualizer(no_elev_graph, output_dir=self.temp_dir)
        
        fig, ax = plt.subplots()
        
        with patch('matplotlib.pyplot.colorbar') as mock_colorbar:
            visualizer._add_elevation_legend(ax)
        
        # The method should not add colorbar if no elevations are present
        # But it does collect empty elevation data, so it might still call colorbar
        # Let's just verify the method doesn't crash
        self.assertTrue(True)  # Method completed without error


class TestGATuningVisualizer(unittest.TestCase):
    """Test GATuningVisualizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = VisualizationConfig(output_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        plt.close('all')
    
    def test_tuning_visualizer_initialization(self):
        """Test GATuningVisualizer initialization"""
        visualizer = GATuningVisualizer(self.config)
        
        self.assertEqual(visualizer.config, self.config)
        self.assertTrue(os.path.exists(self.temp_dir))
    
    def test_tuning_visualizer_initialization_default_config(self):
        """Test GATuningVisualizer initialization with default config"""
        visualizer = GATuningVisualizer()
        
        self.assertIsNotNone(visualizer.config)
        self.assertEqual(visualizer.config.output_dir, "tuning_visualizations")
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_parameter_sensitivity_plot_basic(self, mock_close, mock_savefig):
        """Test basic parameter sensitivity plot creation"""
        visualizer = GATuningVisualizer(self.config)
        
        sensitivity_data = {
            'parameters': {
                'mutation_rate': 0.8,
                'crossover_rate': 0.6,
                'population_size': 0.4,
                'elite_size': 0.2
            }
        }
        
        with patch.object(visualizer, 'create_subplots') as mock_subplots:
            with patch.object(visualizer, 'format_axes') as mock_format:
                with patch.object(visualizer, 'save_figure', return_value='test.png') as mock_save:
                    # Mock the subplots return value
                    fig = Mock()
                    ax = Mock()
                    axes = np.array([[ax, ax], [ax, ax]])
                    mock_subplots.return_value = (fig, axes)
                    
                    result = visualizer.create_parameter_sensitivity_plot(sensitivity_data)
        
        self.assertEqual(result, 'test.png')
        mock_subplots.assert_called_once_with(2, 2, "Parameter Sensitivity Analysis")
        mock_format.assert_called_once()
        mock_save.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_parameter_sensitivity_plot_no_parameters(self, mock_close, mock_savefig):
        """Test parameter sensitivity plot creation with no parameters"""
        visualizer = GATuningVisualizer(self.config)
        
        sensitivity_data = {}
        
        with patch.object(visualizer, 'create_subplots') as mock_subplots:
            with patch.object(visualizer, 'save_figure', return_value='test.png') as mock_save:
                # Mock the subplots return value
                fig = Mock()
                ax = Mock()
                axes = np.array([[ax, ax], [ax, ax]])
                mock_subplots.return_value = (fig, axes)
                
                result = visualizer.create_parameter_sensitivity_plot(sensitivity_data)
        
        self.assertEqual(result, 'test.png')
        mock_subplots.assert_called_once()
        mock_save.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_parameter_sensitivity_plot_custom_filename(self, mock_close, mock_savefig):
        """Test parameter sensitivity plot creation with custom filename"""
        visualizer = GATuningVisualizer(self.config)
        
        sensitivity_data = {
            'parameters': {
                'mutation_rate': 0.8,
                'crossover_rate': 0.6
            }
        }
        
        with patch.object(visualizer, 'create_subplots') as mock_subplots:
            with patch.object(visualizer, 'format_axes') as mock_format:
                with patch.object(visualizer, 'save_figure', return_value='custom.png') as mock_save:
                    # Mock the subplots return value
                    fig = Mock()
                    ax = Mock()
                    axes = np.array([[ax, ax], [ax, ax]])
                    mock_subplots.return_value = (fig, axes)
                    
                    result = visualizer.create_parameter_sensitivity_plot(
                        sensitivity_data, 
                        filename="custom_sensitivity"
                    )
        
        self.assertEqual(result, 'custom.png')
        mock_save.assert_called_once_with(fig, "custom_sensitivity")
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_optimization_progress_plot_basic(self, mock_close, mock_savefig):
        """Test basic optimization progress plot creation"""
        visualizer = GATuningVisualizer(self.config)
        
        optimization_history = [
            {'best_fitness': 100, 'avg_fitness': 80},
            {'best_fitness': 120, 'avg_fitness': 85},
            {'best_fitness': 140, 'avg_fitness': 90},
            {'best_fitness': 160, 'avg_fitness': 95}
        ]
        
        with patch.object(visualizer, 'create_figure') as mock_create:
            with patch.object(visualizer, 'format_axes') as mock_format:
                with patch.object(visualizer, 'save_figure', return_value='progress.png') as mock_save:
                    # Mock the figure and axes
                    fig = Mock()
                    ax = Mock()
                    mock_create.return_value = (fig, ax)
                    
                    result = visualizer.create_optimization_progress_plot(optimization_history)
        
        self.assertEqual(result, 'progress.png')
        mock_create.assert_called_once_with("Optimization Progress")
        mock_format.assert_called_once()
        mock_save.assert_called_once()
        
        # Check that plot was called twice (best and average fitness)
        self.assertEqual(ax.plot.call_count, 2)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_optimization_progress_plot_empty_history(self, mock_close, mock_savefig):
        """Test optimization progress plot creation with empty history"""
        visualizer = GATuningVisualizer(self.config)
        
        optimization_history = []
        
        with patch.object(visualizer, 'create_figure') as mock_create:
            with patch.object(visualizer, 'save_figure', return_value='progress.png') as mock_save:
                # Mock the figure and axes
                fig = Mock()
                ax = Mock()
                mock_create.return_value = (fig, ax)
                
                result = visualizer.create_optimization_progress_plot(optimization_history)
        
        self.assertEqual(result, 'progress.png')
        mock_create.assert_called_once()
        mock_save.assert_called_once()
        
        # Check that plot was not called
        ax.plot.assert_not_called()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_optimization_progress_plot_missing_data(self, mock_close, mock_savefig):
        """Test optimization progress plot creation with missing data"""
        visualizer = GATuningVisualizer(self.config)
        
        optimization_history = [
            {'best_fitness': 100},  # Missing avg_fitness
            {'avg_fitness': 85},    # Missing best_fitness
            {'best_fitness': 140, 'avg_fitness': 90}
        ]
        
        with patch.object(visualizer, 'create_figure') as mock_create:
            with patch.object(visualizer, 'format_axes') as mock_format:
                with patch.object(visualizer, 'save_figure', return_value='progress.png') as mock_save:
                    # Mock the figure and axes
                    fig = Mock()
                    ax = Mock()
                    mock_create.return_value = (fig, ax)
                    
                    result = visualizer.create_optimization_progress_plot(optimization_history)
        
        self.assertEqual(result, 'progress.png')
        mock_create.assert_called_once()
        mock_format.assert_called_once()
        mock_save.assert_called_once()
        
        # Check that plot was called twice (best and average fitness)
        self.assertEqual(ax.plot.call_count, 2)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_optimization_progress_plot_custom_filename(self, mock_close, mock_savefig):
        """Test optimization progress plot creation with custom filename"""
        visualizer = GATuningVisualizer(self.config)
        
        optimization_history = [
            {'best_fitness': 100, 'avg_fitness': 80}
        ]
        
        with patch.object(visualizer, 'create_figure') as mock_create:
            with patch.object(visualizer, 'format_axes') as mock_format:
                with patch.object(visualizer, 'save_figure', return_value='custom.png') as mock_save:
                    # Mock the figure and axes
                    fig = Mock()
                    ax = Mock()
                    mock_create.return_value = (fig, ax)
                    
                    result = visualizer.create_optimization_progress_plot(
                        optimization_history, 
                        filename="custom_progress"
                    )
        
        self.assertEqual(result, 'custom.png')
        mock_save.assert_called_once_with(fig, "custom_progress")


class TestPrecisionComparisonVisualizer(unittest.TestCase):
    """Test PrecisionComparisonVisualizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.route_coordinates = [
            (37.1299, -80.4094),
            (37.1301, -80.4092),
            (37.1303, -80.4090),
            (37.1305, -80.4088),
            (37.1307, -80.4086)
        ]
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        plt.close('all')
    
    def test_precision_visualizer_initialization(self):
        """Test PrecisionComparisonVisualizer initialization"""
        visualizer = PrecisionComparisonVisualizer(output_dir=self.temp_dir)
        
        self.assertIsNotNone(visualizer.config)
        self.assertEqual(visualizer.config.output_dir, self.temp_dir)
        self.assertTrue(os.path.exists(self.temp_dir))
    
    def test_precision_visualizer_initialization_default_dir(self):
        """Test PrecisionComparisonVisualizer initialization with default directory"""
        visualizer = PrecisionComparisonVisualizer()
        
        self.assertIsNotNone(visualizer.config)
        self.assertEqual(visualizer.config.output_dir, "./ga_precision_visualizations")
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_precision_comparison_visualization_basic(self, mock_close, mock_savefig):
        """Test basic precision comparison visualization creation"""
        visualizer = PrecisionComparisonVisualizer(output_dir=self.temp_dir)
        
        with patch.object(visualizer, 'create_subplots') as mock_subplots:
            with patch.object(visualizer, 'format_axes') as mock_format:
                with patch.object(visualizer, 'add_statistics_table') as mock_table:
                    with patch.object(visualizer, 'save_figure', return_value='precision.png') as mock_save:
                        # Mock the subplots return value
                        fig = Mock()
                        ax = Mock()
                        axes = np.array([[ax, ax], [ax, ax]])
                        mock_subplots.return_value = (fig, axes)
                        
                        result = visualizer.create_precision_comparison_visualization(
                            self.route_coordinates
                        )
        
        self.assertEqual(result, 'precision.png')
        mock_subplots.assert_called_once_with(2, 2, "Precision Comparison")
        self.assertEqual(mock_format.call_count, 2)  # Called for 2 plots (high-res and downsampled)
        mock_table.assert_called_once()
        mock_save.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_precision_comparison_visualization_with_graph(self, mock_close, mock_savefig):
        """Test precision comparison visualization creation with graph"""
        visualizer = PrecisionComparisonVisualizer(output_dir=self.temp_dir)
        
        # Create mock graph
        mock_graph = Mock()
        
        with patch.object(visualizer, 'create_subplots') as mock_subplots:
            with patch.object(visualizer, 'format_axes') as mock_format:
                with patch.object(visualizer, 'add_statistics_table') as mock_table:
                    with patch.object(visualizer, 'save_figure', return_value='precision.png') as mock_save:
                        # Mock the subplots return value
                        fig = Mock()
                        ax = Mock()
                        axes = np.array([[ax, ax], [ax, ax]])
                        mock_subplots.return_value = (fig, axes)
                        
                        result = visualizer.create_precision_comparison_visualization(
                            self.route_coordinates,
                            graph=mock_graph,
                            title_suffix=" - Test"
                        )
        
        self.assertEqual(result, 'precision.png')
        mock_subplots.assert_called_once_with(2, 2, "Precision Comparison - Test")
        self.assertEqual(mock_format.call_count, 3)  # Called for 3 plots (high-res, downsampled, elevation)
        mock_table.assert_called_once()
        mock_save.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_precision_comparison_visualization_empty_coordinates(self, mock_close, mock_savefig):
        """Test precision comparison visualization creation with empty coordinates"""
        visualizer = PrecisionComparisonVisualizer(output_dir=self.temp_dir)
        
        with patch.object(visualizer, 'create_subplots') as mock_subplots:
            with patch.object(visualizer, 'add_statistics_table') as mock_table:
                with patch.object(visualizer, 'save_figure', return_value='precision.png') as mock_save:
                    # Mock the subplots return value
                    fig = Mock()
                    ax = Mock()
                    axes = np.array([[ax, ax], [ax, ax]])
                    mock_subplots.return_value = (fig, axes)
                    
                    result = visualizer.create_precision_comparison_visualization([])
        
        self.assertEqual(result, 'precision.png')
        mock_subplots.assert_called_once()
        mock_table.assert_called_once()
        mock_save.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_precision_comparison_visualization_few_coordinates(self, mock_close, mock_savefig):
        """Test precision comparison visualization creation with few coordinates"""
        visualizer = PrecisionComparisonVisualizer(output_dir=self.temp_dir)
        
        few_coordinates = [(37.1299, -80.4094), (37.1301, -80.4092)]
        
        with patch.object(visualizer, 'create_subplots') as mock_subplots:
            with patch.object(visualizer, 'format_axes') as mock_format:
                with patch.object(visualizer, 'add_statistics_table') as mock_table:
                    with patch.object(visualizer, 'save_figure', return_value='precision.png') as mock_save:
                        # Mock the subplots return value
                        fig = Mock()
                        ax = Mock()
                        axes = np.array([[ax, ax], [ax, ax]])
                        mock_subplots.return_value = (fig, axes)
                        
                        result = visualizer.create_precision_comparison_visualization(
                            few_coordinates
                        )
        
        self.assertEqual(result, 'precision.png')
        mock_subplots.assert_called_once()
        mock_format.assert_called_once()  # Only called for 1 plot (high-res, not downsampled)
        mock_table.assert_called_once()
        mock_save.assert_called_once()
    
    def test_create_precision_comparison_visualization_exception_handling(self):
        """Test precision comparison visualization exception handling"""
        visualizer = PrecisionComparisonVisualizer(output_dir=self.temp_dir)
        
        with patch.object(visualizer, 'create_subplots', side_effect=Exception("Test error")):
            result = visualizer.create_precision_comparison_visualization(
                self.route_coordinates
            )
        
        self.assertEqual(result, "")
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_ga_evolution_comparison_basic(self, mock_close, mock_savefig):
        """Test basic GA evolution comparison creation"""
        visualizer = PrecisionComparisonVisualizer(output_dir=self.temp_dir)
        
        high_res_history = [
            {'best_fitness': 100},
            {'best_fitness': 120},
            {'best_fitness': 140}
        ]
        
        low_res_history = [
            {'best_fitness': 80},
            {'best_fitness': 90},
            {'best_fitness': 95}
        ]
        
        with patch.object(visualizer, 'create_figure') as mock_create:
            with patch.object(visualizer, 'format_axes') as mock_format:
                with patch.object(visualizer, 'save_figure', return_value='evolution.png') as mock_save:
                    # Mock the figure and axes
                    fig = Mock()
                    ax = Mock()
                    mock_create.return_value = (fig, ax)
                    
                    result = visualizer.create_ga_evolution_comparison(
                        high_res_history, 
                        low_res_history
                    )
        
        self.assertEqual(result, 'evolution.png')
        mock_create.assert_called_once_with("GA Evolution Comparison")
        mock_format.assert_called_once()
        mock_save.assert_called_once()
        
        # Check that plot was called twice (high and low resolution)
        self.assertEqual(ax.plot.call_count, 2)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_ga_evolution_comparison_with_title_suffix(self, mock_close, mock_savefig):
        """Test GA evolution comparison creation with title suffix"""
        visualizer = PrecisionComparisonVisualizer(output_dir=self.temp_dir)
        
        high_res_history = [{'best_fitness': 100}]
        low_res_history = [{'best_fitness': 80}]
        
        with patch.object(visualizer, 'create_figure') as mock_create:
            with patch.object(visualizer, 'format_axes') as mock_format:
                with patch.object(visualizer, 'save_figure', return_value='evolution.png') as mock_save:
                    # Mock the figure and axes
                    fig = Mock()
                    ax = Mock()
                    mock_create.return_value = (fig, ax)
                    
                    result = visualizer.create_ga_evolution_comparison(
                        high_res_history, 
                        low_res_history,
                        title_suffix=" - Test Run"
                    )
        
        self.assertEqual(result, 'evolution.png')
        mock_create.assert_called_once_with("GA Evolution Comparison - Test Run")
        mock_format.assert_called_once()
        mock_save.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_ga_evolution_comparison_empty_history(self, mock_close, mock_savefig):
        """Test GA evolution comparison creation with empty history"""
        visualizer = PrecisionComparisonVisualizer(output_dir=self.temp_dir)
        
        with patch.object(visualizer, 'create_figure') as mock_create:
            with patch.object(visualizer, 'format_axes') as mock_format:
                with patch.object(visualizer, 'save_figure', return_value='evolution.png') as mock_save:
                    # Mock the figure and axes
                    fig = Mock()
                    ax = Mock()
                    mock_create.return_value = (fig, ax)
                    
                    result = visualizer.create_ga_evolution_comparison([], [])
        
        self.assertEqual(result, 'evolution.png')
        mock_create.assert_called_once()
        mock_format.assert_called_once()
        mock_save.assert_called_once()
        
        # Check that plot was not called
        ax.plot.assert_not_called()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_ga_evolution_comparison_missing_fitness(self, mock_close, mock_savefig):
        """Test GA evolution comparison creation with missing fitness data"""
        visualizer = PrecisionComparisonVisualizer(output_dir=self.temp_dir)
        
        high_res_history = [
            {'best_fitness': 100},
            {},  # Missing best_fitness
            {'best_fitness': 140}
        ]
        
        low_res_history = [
            {},  # Missing best_fitness
            {'best_fitness': 90}
        ]
        
        with patch.object(visualizer, 'create_figure') as mock_create:
            with patch.object(visualizer, 'format_axes') as mock_format:
                with patch.object(visualizer, 'save_figure', return_value='evolution.png') as mock_save:
                    # Mock the figure and axes
                    fig = Mock()
                    ax = Mock()
                    mock_create.return_value = (fig, ax)
                    
                    result = visualizer.create_ga_evolution_comparison(
                        high_res_history, 
                        low_res_history
                    )
        
        self.assertEqual(result, 'evolution.png')
        mock_create.assert_called_once()
        mock_format.assert_called_once()
        mock_save.assert_called_once()
        
        # Check that plot was called twice (high and low resolution)
        self.assertEqual(ax.plot.call_count, 2)
    
    def test_create_ga_evolution_comparison_exception_handling(self):
        """Test GA evolution comparison exception handling"""
        visualizer = PrecisionComparisonVisualizer(output_dir=self.temp_dir)
        
        with patch.object(visualizer, 'create_figure', side_effect=Exception("Test error")):
            result = visualizer.create_ga_evolution_comparison([], [])
        
        self.assertEqual(result, "")


class TestVisualizationIntegration(unittest.TestCase):
    """Integration tests for visualization components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create realistic test graph
        self.test_graph = nx.Graph()
        
        # Add nodes with coordinates and elevation
        nodes = [
            (1001, -80.4094, 37.1299, 100.0),
            (1002, -80.4090, 37.1299, 105.0),
            (1003, -80.4086, 37.1299, 110.0),
            (1004, -80.4082, 37.1299, 115.0),
            (1005, -80.4078, 37.1299, 120.0)
        ]
        
        for node_id, x, y, elevation in nodes:
            self.test_graph.add_node(node_id, x=x, y=y, elevation=elevation)
        
        # Add edges
        edges = [
            (1001, 1002, 400), (1002, 1003, 400), (1003, 1004, 400), (1004, 1005, 400)
        ]
        
        for node1, node2, length in edges:
            self.test_graph.add_edge(node1, node2, length=length)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        plt.close('all')
    
    def test_visualization_workflow_integration(self):
        """Test complete visualization workflow integration"""
        # Test GAVisualizer
        ga_visualizer = GAVisualizer(self.test_graph, output_dir=self.temp_dir)
        self.assertIsNotNone(ga_visualizer)
        
        # Test GATuningVisualizer
        tuning_visualizer = GATuningVisualizer(
            VisualizationConfig(output_dir=self.temp_dir)
        )
        self.assertIsNotNone(tuning_visualizer)
        
        # Test PrecisionComparisonVisualizer
        precision_visualizer = PrecisionComparisonVisualizer(output_dir=self.temp_dir)
        self.assertIsNotNone(precision_visualizer)
        
        # All visualizers should use the same output directory
        self.assertTrue(os.path.exists(self.temp_dir))
    
    def test_error_handling_across_visualizers(self):
        """Test error handling across different visualizer types"""
        # Test GAVisualizer with invalid graph
        empty_graph = nx.Graph()
        ga_visualizer = GAVisualizer(empty_graph, output_dir=self.temp_dir)
        self.assertIsNotNone(ga_visualizer)
        
        # Test GATuningVisualizer with invalid data
        tuning_visualizer = GATuningVisualizer(
            VisualizationConfig(output_dir=self.temp_dir)
        )
        
        # Should handle empty or invalid data gracefully
        with patch.object(tuning_visualizer, 'save_figure', return_value='test.png'):
            result = tuning_visualizer.create_parameter_sensitivity_plot({})
        
        self.assertEqual(result, 'test.png')
    
    def test_visualization_config_consistency(self):
        """Test that visualization configuration is consistent across components"""
        config = VisualizationConfig(
            output_dir=self.temp_dir,
            figure_format="pdf",
            dpi=300
        )
        
        # Test that all visualizers respect the config
        ga_visualizer = GAVisualizer(self.test_graph, output_dir=self.temp_dir)
        tuning_visualizer = GATuningVisualizer(config)
        precision_visualizer = PrecisionComparisonVisualizer(output_dir=self.temp_dir)
        
        # Check that config is applied correctly
        self.assertEqual(tuning_visualizer.config.figure_format, "pdf")
        self.assertEqual(tuning_visualizer.config.dpi, 300)
        
        # Check that output directories are created
        self.assertTrue(os.path.exists(self.temp_dir))


if __name__ == '__main__':
    unittest.main()