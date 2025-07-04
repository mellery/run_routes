#!/usr/bin/env python3
"""
Unit tests for GA Visualizer
Tests GAVisualizer functionality (mocked to avoid file I/O in unit tests)
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import networkx as nx
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Skip tests if folium is not available
try:
    from ga_visualizer import GAVisualizer
    from ga_chromosome import RouteChromosome, RouteSegment
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False


@unittest.skipIf(not FOLIUM_AVAILABLE, "folium not available")
class TestGAVisualizer(unittest.TestCase):
    """Test GAVisualizer class with mocked dependencies"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock graph
        self.mock_graph = nx.Graph()
        self.mock_graph.add_node(1, x=-80.4094, y=37.1299, elevation=100.0)
        self.mock_graph.add_node(2, x=-80.4090, y=37.1300, elevation=110.0)
        self.mock_graph.add_node(3, x=-80.4086, y=37.1301, elevation=105.0)
        self.mock_graph.add_edge(1, 2, length=100.0)
        self.mock_graph.add_edge(2, 3, length=150.0)
        self.mock_graph.add_edge(3, 1, length=200.0)
        
        # Create test segments and chromosome
        self.segment1 = RouteSegment(1, 2, [1, 2])
        self.segment1.calculate_properties(self.mock_graph)
        
        self.segment2 = RouteSegment(2, 3, [2, 3])
        self.segment2.calculate_properties(self.mock_graph)
        
        self.chromosome = RouteChromosome([self.segment1, self.segment2])
        self.chromosome.fitness = 0.75
    
    @patch('os.makedirs')
    def test_visualizer_initialization(self, mock_makedirs):
        """Test GAVisualizer initialization"""
        output_dir = "test_output"
        visualizer = GAVisualizer(self.mock_graph, output_dir)
        
        self.assertEqual(visualizer.graph, self.mock_graph)
        self.assertEqual(visualizer.output_dir, output_dir)
        self.assertIsInstance(visualizer.bounds, dict)
        
        # Check that output directory creation was attempted
        mock_makedirs.assert_called_once_with(output_dir, exist_ok=True)
    
    def test_calculate_graph_bounds(self):
        """Test graph bounds calculation"""
        visualizer = GAVisualizer(self.mock_graph)
        bounds = visualizer._calculate_graph_bounds()
        
        required_keys = ['min_lat', 'max_lat', 'min_lon', 'max_lon', 'center_lat', 'center_lon']
        for key in required_keys:
            self.assertIn(key, bounds)
        
        # Verify bounds are reasonable
        self.assertLess(bounds['min_lat'], bounds['max_lat'])
        self.assertLess(bounds['min_lon'], bounds['max_lon'])
        
        # Verify center calculation
        expected_center_lat = (bounds['min_lat'] + bounds['max_lat']) / 2
        expected_center_lon = (bounds['min_lon'] + bounds['max_lon']) / 2
        self.assertAlmostEqual(bounds['center_lat'], expected_center_lat, places=6)
        self.assertAlmostEqual(bounds['center_lon'], expected_center_lon, places=6)
    
    def test_calculate_graph_bounds_empty_graph(self):
        """Test graph bounds calculation with empty graph"""
        empty_graph = nx.Graph()
        visualizer = GAVisualizer(empty_graph)
        bounds = visualizer._calculate_graph_bounds()
        
        self.assertEqual(bounds['min_lat'], 0)
        self.assertEqual(bounds['max_lat'], 0)
        self.assertEqual(bounds['min_lon'], 0)
        self.assertEqual(bounds['max_lon'], 0)
    
    def test_get_timestamp(self):
        """Test timestamp generation"""
        visualizer = GAVisualizer(self.mock_graph)
        timestamp = visualizer._get_timestamp()
        
        self.assertIsInstance(timestamp, str)
        self.assertEqual(len(timestamp), 15)  # YYYYMMDD_HHMMSS format
        self.assertRegex(timestamp, r'\d{8}_\d{6}')
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.subplots')
    def test_save_chromosome_map(self, mock_subplots, mock_tight_layout, mock_close, mock_savefig):
        """Test chromosome map saving (mocked)"""
        # Setup mocks
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        visualizer = GAVisualizer(self.mock_graph, "test_output")
        
        # Test chromosome map saving
        result_path = visualizer.save_chromosome_map(
            self.chromosome,
            filename="test_chromosome.png",
            title="Test Chromosome"
        )
        
        # Verify file path
        expected_path = os.path.join("test_output", "test_chromosome.png")
        self.assertEqual(result_path, expected_path)
        
        # Verify matplotlib methods were called
        mock_subplots.assert_called_once()
        mock_tight_layout.assert_called_once()
        mock_savefig.assert_called_once_with(expected_path, dpi=150, bbox_inches='tight')
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.subplot2grid')
    def test_save_population_map(self, mock_subplot2grid, mock_figure, mock_tight_layout, mock_close, mock_savefig):
        """Test population map saving (mocked)"""
        # Setup mocks
        mock_fig = Mock()
        mock_figure.return_value = mock_fig
        mock_ax = Mock()
        
        # Create a mock table that supports subscript access
        mock_table = Mock()
        mock_table.__getitem__ = Mock(return_value=Mock())
        mock_ax.table.return_value = mock_table
        
        mock_subplot2grid.return_value = mock_ax
        
        visualizer = GAVisualizer(self.mock_graph, "test_output")
        
        # Create test population
        population = [self.chromosome]
        
        # Test population map saving
        result_path = visualizer.save_population_map(
            population,
            generation=5,
            filename="test_population.png"
        )
        
        # Verify file path
        expected_path = os.path.join("test_output", "test_population.png")
        self.assertEqual(result_path, expected_path)
        
        # Verify matplotlib methods were called
        mock_figure.assert_called_once()
        mock_tight_layout.assert_called_once()
        mock_savefig.assert_called_once_with(expected_path, dpi=150, bbox_inches='tight')
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.subplots')
    def test_save_comparison_map(self, mock_subplots, mock_tight_layout, mock_close, mock_savefig):
        """Test comparison map saving (mocked)"""
        # Setup mocks
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        
        visualizer = GAVisualizer(self.mock_graph, "test_output")
        
        # Test comparison map saving
        tsp_route = [1, 2, 3, 1]
        result_path = visualizer.save_comparison_map(
            self.chromosome,
            tsp_route,
            filename="test_comparison.png"
        )
        
        # Verify file path
        expected_path = os.path.join("test_output", "test_comparison.png")
        self.assertEqual(result_path, expected_path)
        
        # Verify matplotlib methods were called
        mock_subplots.assert_called_once()
        mock_tight_layout.assert_called_once()
        mock_savefig.assert_called_once_with(expected_path, dpi=150, bbox_inches='tight')
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.subplots')
    def test_save_comparison_map_no_tsp(self, mock_subplots, mock_tight_layout, mock_close, mock_savefig):
        """Test comparison map saving without TSP route (mocked)"""
        # Setup mocks
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        
        visualizer = GAVisualizer(self.mock_graph, "test_output")
        
        # Test comparison map saving without TSP route
        result_path = visualizer.save_comparison_map(
            self.chromosome,
            None,  # No TSP route
            filename="test_comparison_no_tsp.png"
        )
        
        # Should still work
        expected_path = os.path.join("test_output", "test_comparison_no_tsp.png")
        self.assertEqual(result_path, expected_path)
        mock_savefig.assert_called_once()
    
    def test_plot_network_background(self):
        """Test network background plotting"""
        visualizer = GAVisualizer(self.mock_graph)
        mock_ax = Mock()
        
        # Should not raise exception
        visualizer._plot_network_background(mock_ax)
        
        # Verify plot and scatter methods were called
        self.assertTrue(mock_ax.plot.called)
        self.assertTrue(mock_ax.scatter.called)
    
    def test_plot_chromosome_route(self):
        """Test chromosome route plotting"""
        visualizer = GAVisualizer(self.mock_graph)
        mock_ax = Mock()
        
        # Should not raise exception
        visualizer._plot_chromosome_route(mock_ax, self.chromosome)
        
        # Verify plot method was called
        self.assertTrue(mock_ax.plot.called)
    
    def test_plot_chromosome_route_empty(self):
        """Test chromosome route plotting with empty chromosome"""
        visualizer = GAVisualizer(self.mock_graph)
        mock_ax = Mock()
        
        empty_chromosome = RouteChromosome([])
        
        # Should handle empty chromosome gracefully
        visualizer._plot_chromosome_route(mock_ax, empty_chromosome)
        
        # Plot should not have been called for empty chromosome
        self.assertFalse(mock_ax.plot.called)
    
    def test_plot_tsp_route(self):
        """Test TSP route plotting"""
        visualizer = GAVisualizer(self.mock_graph)
        mock_ax = Mock()
        
        tsp_route = [1, 2, 3, 1]
        
        # Should not raise exception
        visualizer._plot_tsp_route(mock_ax, tsp_route)
        
        # Verify plot and scatter methods were called
        self.assertTrue(mock_ax.plot.called)
        self.assertTrue(mock_ax.scatter.called)
    
    def test_plot_tsp_route_empty(self):
        """Test TSP route plotting with empty route"""
        visualizer = GAVisualizer(self.mock_graph)
        mock_ax = Mock()
        
        empty_route = []
        
        # Should handle empty route gracefully
        visualizer._plot_tsp_route(mock_ax, empty_route)
        
        # Plot should not have been called for empty route
        self.assertFalse(mock_ax.plot.called)
    
    def test_calculate_tsp_stats(self):
        """Test TSP statistics calculation"""
        visualizer = GAVisualizer(self.mock_graph)
        
        tsp_route = [1, 2, 3]
        stats = visualizer._calculate_tsp_stats(tsp_route)
        
        self.assertIn('distance_km', stats)
        self.assertIn('elevation_gain_m', stats)
        self.assertIsInstance(stats['distance_km'], float)
        self.assertIsInstance(stats['elevation_gain_m'], float)
        self.assertGreaterEqual(stats['distance_km'], 0.0)
        self.assertGreaterEqual(stats['elevation_gain_m'], 0.0)
    
    def test_calculate_tsp_stats_empty(self):
        """Test TSP statistics calculation with empty route"""
        visualizer = GAVisualizer(self.mock_graph)
        
        empty_route = []
        stats = visualizer._calculate_tsp_stats(empty_route)
        
        self.assertEqual(stats['distance_km'], 0.0)
        self.assertEqual(stats['elevation_gain_m'], 0.0)
    
    def test_calculate_tsp_stats_single_node(self):
        """Test TSP statistics calculation with single node"""
        visualizer = GAVisualizer(self.mock_graph)
        
        single_route = [1]
        stats = visualizer._calculate_tsp_stats(single_route)
        
        self.assertEqual(stats['distance_km'], 0.0)
        self.assertEqual(stats['elevation_gain_m'], 0.0)
    
    def test_set_map_bounds(self):
        """Test map bounds setting"""
        visualizer = GAVisualizer(self.mock_graph)
        mock_ax = Mock()
        
        # Should not raise exception
        visualizer._set_map_bounds(mock_ax)
        
        # Verify axis methods were called
        self.assertTrue(mock_ax.set_xlim.called)
        self.assertTrue(mock_ax.set_ylim.called)
        self.assertTrue(mock_ax.set_aspect.called)
    
    def test_add_elevation_legend(self):
        """Test elevation legend addition"""
        visualizer = GAVisualizer(self.mock_graph)
        mock_ax = Mock()
        
        # Should not raise exception
        visualizer._add_elevation_legend(mock_ax)
        
        # Verify legend method was called
        self.assertTrue(mock_ax.legend.called)
    
    def test_add_population_stats_table_empty(self):
        """Test population stats table with empty population"""
        visualizer = GAVisualizer(self.mock_graph)
        mock_ax = Mock()
        
        empty_population = []
        
        # Should handle empty population gracefully
        visualizer._add_population_stats_table(mock_ax, empty_population, 0)
        
        # Should turn off axis
        mock_ax.axis.assert_called_with('off')
    
    def test_add_population_stats_table_valid(self):
        """Test population stats table with valid population"""
        visualizer = GAVisualizer(self.mock_graph)
        mock_ax = Mock()
        
        # Create a mock table that supports subscript access
        mock_table = Mock()
        mock_table.__getitem__ = Mock(return_value=Mock())
        mock_ax.table.return_value = mock_table
        
        population = [self.chromosome]
        
        # Should not raise exception
        visualizer._add_population_stats_table(mock_ax, population, 5)
        
        # Should turn off axis and create table
        mock_ax.axis.assert_called_with('off')
        self.assertTrue(mock_ax.table.called)
    
    def test_filename_generation(self):
        """Test automatic filename generation"""
        visualizer = GAVisualizer(self.mock_graph, "test_output")
        
        mock_fig = Mock()
        mock_ax = Mock()
        
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.subplots', return_value=(mock_fig, mock_ax)):
            
            # Test with no filename provided
            result_path = visualizer.save_chromosome_map(self.chromosome)
            
            # Should generate a filename
            self.assertTrue(result_path.startswith("test_output/ga_dev_chromosome_"))
            self.assertTrue(result_path.endswith(".png"))
    
    def test_visualization_with_different_options(self):
        """Test visualization with different options"""
        visualizer = GAVisualizer(self.mock_graph, "test_output")
        
        mock_fig = Mock()
        mock_ax = Mock()
        
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.subplots', return_value=(mock_fig, mock_ax)):
            
            # Test with different combinations of options
            visualizer.save_chromosome_map(
                self.chromosome,
                show_elevation=True,
                show_segments=True,
                show_stats=True
            )
            
            visualizer.save_chromosome_map(
                self.chromosome,
                show_elevation=False,
                show_segments=False,
                show_stats=False
            )
            
            # Should complete without errors
    
    def test_error_handling_invalid_chromosome(self):
        """Test error handling with invalid chromosome data"""
        visualizer = GAVisualizer(self.mock_graph)
        mock_ax = Mock()
        
        # Create chromosome with empty segments (valid case for error handling)
        empty_chromosome = RouteChromosome([])
        
        # Should handle gracefully without crashing
        visualizer._plot_chromosome_route(mock_ax, empty_chromosome)
        
        # Create chromosome with invalid connectivity (but valid nodes in graph)
        valid_segment = RouteSegment(1, 2, [1, 2])
        valid_segment.is_valid = False  # Mark as invalid
        invalid_chromosome = RouteChromosome([valid_segment])
        
        # Should handle gracefully without crashing
        visualizer._plot_chromosome_route(mock_ax, invalid_chromosome)
    
    def test_color_scheme_consistency(self):
        """Test that color schemes are properly defined"""
        visualizer = GAVisualizer(self.mock_graph)
        
        # Check that route colors are defined
        self.assertIsInstance(visualizer.route_colors, list)
        self.assertGreater(len(visualizer.route_colors), 0)
        
        # Check that elevation colormap is defined
        self.assertIsNotNone(visualizer.elevation_colormap)


@unittest.skipIf(not FOLIUM_AVAILABLE, "folium not available")
class TestGAVisualizerIntegration(unittest.TestCase):
    """Integration tests for GAVisualizer (testing actual functionality without file I/O)"""
    
    def setUp(self):
        """Set up more complex test scenario"""
        # Create larger mock graph
        self.mock_graph = nx.Graph()
        
        # Add multiple nodes
        for i in range(10):
            self.mock_graph.add_node(
                i + 1,
                x=-80.4094 + i * 0.001,
                y=37.1299 + i * 0.0005,
                elevation=100 + i * 10
            )
        
        # Connect nodes in a path
        for i in range(9):
            self.mock_graph.add_edge(i + 1, i + 2, length=100)
        
        # Create test population
        self.population = []
        for i in range(3):
            segment = RouteSegment(1, i + 2, [1, i + 2])
            segment.calculate_properties(self.mock_graph)
            chromosome = RouteChromosome([segment])
            chromosome.fitness = 0.5 + i * 0.1
            self.population.append(chromosome)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    def test_full_visualization_workflow(self, mock_tight_layout, mock_close, mock_savefig):
        """Test complete visualization workflow"""
        mock_fig = Mock()
        mock_ax = Mock()
        
        # Setup table mock for population visualization
        mock_table = Mock()
        mock_table.__getitem__ = Mock(return_value=Mock())
        mock_ax.table.return_value = mock_table
        
        # Mock subplots to handle different call signatures
        def mock_subplots_side_effect(*args, **kwargs):
            if len(args) >= 2 and args[0] == 1 and args[1] == 2:
                # save_comparison_map calls plt.subplots(1, 2, figsize=(20, 10))
                return (mock_fig, (mock_ax, mock_ax))
            else:
                # save_chromosome_map calls plt.subplots(figsize=(12, 10))
                return (mock_fig, mock_ax)
        
        with patch('matplotlib.pyplot.subplots', side_effect=mock_subplots_side_effect), \
             patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.subplot2grid', return_value=mock_ax):
            
            visualizer = GAVisualizer(self.mock_graph, "test_output")
            
            # Test chromosome visualization
            chrome_path = visualizer.save_chromosome_map(self.population[0])
            self.assertIsInstance(chrome_path, str)
            
            # Test population visualization
            pop_path = visualizer.save_population_map(self.population, generation=10)
            self.assertIsInstance(pop_path, str)
            
            # Test comparison visualization
            comp_path = visualizer.save_comparison_map(self.population[0], [1, 2, 3])
            self.assertIsInstance(comp_path, str)
            
            # Verify all saves were called
            self.assertEqual(mock_savefig.call_count, 3)


if __name__ == '__main__':
    unittest.main()