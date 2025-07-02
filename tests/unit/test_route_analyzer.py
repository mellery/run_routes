#!/usr/bin/env python3
"""
Unit tests for RouteAnalyzer
"""

import unittest
from unittest.mock import Mock, patch
import networkx as nx
import sys
import os

# Add the parent directory to sys.path to import route_services
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from route_services.route_analyzer import RouteAnalyzer


class TestRouteAnalyzer(unittest.TestCase):
    """Test cases for RouteAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a mock graph with elevation data
        self.mock_graph = nx.Graph()
        self.mock_graph.add_node(1001, x=-80.4094, y=37.1299, elevation=610)
        self.mock_graph.add_node(1002, x=-80.4095, y=37.1300, elevation=620)  # +10m
        self.mock_graph.add_node(1003, x=-80.4096, y=37.1301, elevation=615)  # -5m
        self.mock_graph.add_node(1004, x=-80.4097, y=37.1302, elevation=625)  # +10m
        self.mock_graph.add_edge(1001, 1002, length=100)
        self.mock_graph.add_edge(1002, 1003, length=150)
        self.mock_graph.add_edge(1003, 1004, length=120)
        
        self.analyzer = RouteAnalyzer(self.mock_graph)
        
        # Create sample route result
        self.sample_route_result = {
            'route': [1001, 1002, 1003, 1004],
            'stats': {
                'total_distance_km': 2.5,
                'total_elevation_gain_m': 20,
                'total_elevation_loss_m': 5,
                'net_elevation_gain_m': 15,
                'estimated_time_min': 15
            },
            'algorithm': 'nearest_neighbor',
            'objective': 'maximize_elevation'
        }
    
    def test_initialization(self):
        """Test RouteAnalyzer initialization"""
        analyzer = RouteAnalyzer(self.mock_graph)
        self.assertEqual(analyzer.graph, self.mock_graph)
    
    def test_analyze_route_success(self):
        """Test successful route analysis"""
        analysis = self.analyzer.analyze_route(self.sample_route_result)
        
        # Check structure
        self.assertIn('basic_stats', analysis)
        self.assertIn('additional_stats', analysis)
        self.assertIn('route_info', analysis)
        
        # Check basic stats passthrough
        self.assertEqual(analysis['basic_stats'], self.sample_route_result['stats'])
        
        # Check route info
        route_info = analysis['route_info']
        self.assertEqual(route_info['route_length'], 4)
        self.assertEqual(route_info['start_node'], 1001)
        self.assertEqual(route_info['end_node'], 1004)
        self.assertFalse(route_info['is_loop'])
        
        # Check additional stats exist
        additional_stats = analysis['additional_stats']
        self.assertIn('total_segments', additional_stats)
        self.assertIn('uphill_segments', additional_stats)
        self.assertIn('downhill_segments', additional_stats)
    
    def test_analyze_route_empty(self):
        """Test analysis with empty route result"""
        result = self.analyzer.analyze_route({})
        self.assertEqual(result, {})
        
        result = self.analyzer.analyze_route(None)
        self.assertEqual(result, {})
    
    @patch('route.haversine_distance')
    def test_calculate_additional_stats(self, mock_haversine):
        """Test additional statistics calculation"""
        # Mock haversine distance to return predictable values
        mock_haversine.side_effect = [100, 150, 120, 200]  # distances in meters
        
        route = [1001, 1002, 1003, 1004]
        additional_stats = self.analyzer._calculate_additional_stats(route)
        
        # Should analyze 4 segments (including return to start)
        self.assertEqual(additional_stats['total_segments'], 4)
        
        # Check that we have uphill/downhill/level counts
        self.assertIn('uphill_segments', additional_stats)
        self.assertIn('downhill_segments', additional_stats)
        self.assertIn('level_segments', additional_stats)
        self.assertIn('uphill_percentage', additional_stats)
        self.assertIn('downhill_percentage', additional_stats)
    
    def test_calculate_additional_stats_empty_route(self):
        """Test additional stats with empty route"""
        result = self.analyzer._calculate_additional_stats([])
        self.assertEqual(result, {})
        
        result = self.analyzer._calculate_additional_stats([1001])  # Single node
        self.assertEqual(result, {})
    
    @patch('route.haversine_distance')
    def test_generate_directions(self, mock_haversine):
        """Test turn-by-turn directions generation"""
        mock_haversine.return_value = 100  # constant distance
        
        directions = self.analyzer.generate_directions(self.sample_route_result)
        
        # Should have directions for each route segment 
        self.assertEqual(len(directions), 5)
        
        # Check start direction
        start_dir = directions[0]
        self.assertEqual(start_dir['step'], 1)
        self.assertEqual(start_dir['type'], 'start')
        self.assertEqual(start_dir['node_id'], 1001)
        self.assertEqual(start_dir['elevation'], 610)
        
        # Check intermediate directions
        dir_2 = directions[1]
        self.assertEqual(dir_2['step'], 2)
        self.assertEqual(dir_2['type'], 'continue')
        self.assertEqual(dir_2['node_id'], 1002)
        self.assertEqual(dir_2['elevation'], 620)
        self.assertEqual(dir_2['elevation_change'], 10)
        self.assertEqual(dir_2['terrain'], 'uphill')
        
        # Check return direction
        return_dir = directions[-1]
        self.assertEqual(return_dir['type'], 'finish')
        self.assertIn('Return to starting point', return_dir['instruction'])
    
    def test_generate_directions_empty(self):
        """Test directions generation with empty route"""
        result = self.analyzer.generate_directions({})
        self.assertEqual(result, [])
        
        result = self.analyzer.generate_directions({'route': []})
        self.assertEqual(result, [])
    
    def test_get_route_difficulty_rating_easy(self):
        """Test difficulty rating for easy route"""
        easy_route = {
            'route': [1001, 1002],
            'stats': {
                'total_distance_km': 1.0,
                'total_elevation_gain_m': 10,
                'estimated_time_min': 6
            }
        }
        
        difficulty = self.analyzer.get_route_difficulty_rating(easy_route)
        
        self.assertIn('rating', difficulty)
        self.assertIn('score', difficulty)
        self.assertIn('factors', difficulty)
        self.assertLessEqual(difficulty['score'], 40)  # Should be easy/moderate
        self.assertIn(difficulty['rating'], ['Very Easy', 'Easy', 'Moderate'])
    
    def test_get_route_difficulty_rating_hard(self):
        """Test difficulty rating for hard route"""
        hard_route = {
            'route': [1001, 1002, 1003, 1004],
            'stats': {
                'total_distance_km': 15.0,  # Very long
                'total_elevation_gain_m': 500,  # Lots of climbing
                'estimated_time_min': 90
            }
        }
        
        # Mock the additional stats to show mostly uphill
        with patch.object(self.analyzer, 'analyze_route') as mock_analyze:
            mock_analyze.return_value = {
                'additional_stats': {
                    'uphill_percentage': 60,
                    'steepest_uphill_grade': 18
                }
            }
            
            difficulty = self.analyzer.get_route_difficulty_rating(hard_route)
            
            self.assertGreaterEqual(difficulty['score'], 50)  # Should be hard
            self.assertIn(difficulty['rating'], ['Hard', 'Very Hard'])
            self.assertGreater(len(difficulty['factors']), 0)
    
    def test_get_route_difficulty_rating_empty(self):
        """Test difficulty rating with empty route"""
        difficulty = self.analyzer.get_route_difficulty_rating(None)
        
        self.assertEqual(difficulty['rating'], 'unknown')
        self.assertEqual(difficulty['score'], 0)
        self.assertEqual(difficulty['factors'], [])
    
    def test_terrain_classification(self):
        """Test terrain classification in directions"""
        # Test uphill (>5m elevation change)
        directions = self.analyzer.generate_directions(self.sample_route_result)
        
        # Find direction from 1001 to 1002 (+10m elevation)
        uphill_dir = next(d for d in directions if d.get('node_id') == 1002)
        self.assertEqual(uphill_dir['terrain'], 'uphill')
        
        # Find direction from 1002 to 1003 (-5m elevation)
        level_dir = next(d for d in directions if d.get('node_id') == 1003)
        self.assertEqual(level_dir['terrain'], 'level')  # -5m is not steep enough for downhill
    
    def test_cumulative_distance_calculation(self):
        """Test cumulative distance calculation in directions"""
        with patch('route.haversine_distance') as mock_haversine:
            mock_haversine.return_value = 100  # constant distance in meters
            
            directions = self.analyzer.generate_directions(self.sample_route_result)
            
            # Check cumulative distances (with constant 100m segments)
            self.assertEqual(directions[0]['cumulative_distance_km'], 0.0)  # Start
            self.assertEqual(directions[1]['cumulative_distance_km'], 0.1)  # 100m
            self.assertEqual(directions[2]['cumulative_distance_km'], 0.2)  # 200m
            self.assertEqual(directions[3]['cumulative_distance_km'], 0.3)  # 300m


if __name__ == '__main__':
    unittest.main()