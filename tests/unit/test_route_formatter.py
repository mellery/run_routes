#!/usr/bin/env python3
"""
Unit tests for RouteFormatter
"""

import unittest
from unittest.mock import patch
import json
import sys
import os

# Add the parent directory to sys.path to import route_services
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from route_services.route_formatter import RouteFormatter


class TestRouteFormatter(unittest.TestCase):
    """Test cases for RouteFormatter class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.formatter = RouteFormatter()
        
        # Sample route result
        self.sample_route_result = {
            'route': [1001, 1002, 1003, 1004],
            'stats': {
                'total_distance_km': 2.5,
                'total_elevation_gain_m': 120,
                'total_elevation_loss_m': 80,
                'net_elevation_gain_m': 40,
                'max_grade_percent': 8.5,
                'estimated_time_min': 15
            },
            'algorithm': 'nearest_neighbor',
            'objective': 'maximize_elevation',
            'solver_info': {
                'solver_type': 'fast',
                'solve_time': 3.2
            }
        }
        
        # Sample analysis data
        self.sample_analysis = {
            'difficulty': {
                'rating': 'Moderate',
                'score': 45,
                'factors': ['Moderate distance', 'Some elevation gain']
            }
        }
        
        # Sample directions
        self.sample_directions = [
            {
                'step': 1,
                'type': 'start',
                'instruction': 'Start at intersection (Node 1001)',
                'node_id': 1001,
                'elevation': 610,
                'elevation_change': 0,
                'cumulative_distance_km': 0.0,
                'terrain': 'start'
            },
            {
                'step': 2,
                'type': 'continue',
                'instruction': 'Continue to intersection (Node 1002) - uphill',
                'node_id': 1002,
                'elevation': 625,
                'elevation_change': 15,
                'cumulative_distance_km': 0.1,
                'terrain': 'uphill'
            }
        ]
    
    def test_initialization(self):
        """Test RouteFormatter initialization"""
        formatter = RouteFormatter()
        self.assertIsInstance(formatter, RouteFormatter)
    
    def test_format_route_stats_cli(self):
        """Test CLI route statistics formatting"""
        result = self.formatter.format_route_stats_cli(self.sample_route_result, self.sample_analysis)
        
        # Check that result is a string
        self.assertIsInstance(result, str)
        
        # Check that key information is included
        self.assertIn('Route Statistics:', result)
        self.assertIn('Distance:        2.50 km', result)
        self.assertIn('Elevation Gain:  120 m', result)
        self.assertIn('Net Elevation:   +40 m', result)
        self.assertIn('Algorithm:       nearest_neighbor', result)
        self.assertIn('Solver:          fast', result)
        self.assertIn('Difficulty:      Moderate (45/100)', result)
    
    def test_format_route_stats_cli_empty(self):
        """Test CLI formatting with empty route result"""
        result = self.formatter.format_route_stats_cli(None)
        self.assertEqual(result, "❌ No route data available")
    
    def test_format_route_stats_web(self):
        """Test web route statistics formatting"""
        result = self.formatter.format_route_stats_web(self.sample_route_result, self.sample_analysis)
        
        # Check structure
        self.assertIsInstance(result, dict)
        
        # Check required metrics
        expected_metrics = [
            'distance', 'elevation_gain', 'elevation_loss', 'net_elevation',
            'max_grade', 'estimated_time', 'route_points', 'solve_time', 'difficulty'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, result)
            self.assertIn('value', result[metric])
            self.assertIn('raw_value', result[metric])
            self.assertIn('unit', result[metric])
        
        # Check specific values
        self.assertEqual(result['distance']['value'], '2.50 km')
        self.assertEqual(result['distance']['raw_value'], 2.5)
        self.assertEqual(result['difficulty']['value'], 'Moderate')
    
    def test_format_route_stats_web_empty(self):
        """Test web formatting with empty route result"""
        result = self.formatter.format_route_stats_web(None)
        self.assertEqual(result, {})
    
    def test_format_directions_cli(self):
        """Test CLI directions formatting"""
        result = self.formatter.format_directions_cli(self.sample_directions)
        
        # Check that result is a string
        self.assertIsInstance(result, str)
        
        # Check content
        self.assertIn('Turn-by-Turn Directions:', result)
        self.assertIn('1. Start at intersection (Node 1001)', result)
        self.assertIn('Elevation: 610m (+0m)', result)
        self.assertIn('2. Continue to intersection (Node 1002) - uphill', result)
        self.assertIn('Elevation: 625m (+15m)', result)
    
    def test_format_directions_cli_empty(self):
        """Test CLI directions formatting with empty list"""
        result = self.formatter.format_directions_cli([])
        self.assertEqual(result, "❌ No directions available")
    
    def test_format_directions_web(self):
        """Test web directions formatting"""
        result = self.formatter.format_directions_web(self.sample_directions)
        
        # Check structure
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        
        # Check first direction
        first_dir = result[0]
        expected_keys = ['step', 'instruction', 'elevation', 'elevation_change', 'distance', 'terrain', 'type']
        for key in expected_keys:
            self.assertIn(key, first_dir)
        
        self.assertEqual(first_dir['step'], 1)
        self.assertEqual(first_dir['elevation'], '610m')
        self.assertEqual(first_dir['elevation_change'], '+0m')
        self.assertEqual(first_dir['terrain'], 'start')
    
    def test_format_elevation_profile_data_web(self):
        """Test elevation profile formatting for web"""
        profile_data = {
            'distances_km': [0, 0.1, 0.25, 0.4],
            'elevations': [610, 625, 620, 635],
            'coordinates': [{'lat': 37.1299, 'lon': -80.4094}],
            'elevation_stats': {'min_elevation': 610, 'max_elevation': 635},
            'total_distance_km': 0.4
        }
        
        result = self.formatter.format_elevation_profile_data(profile_data, 'web')
        
        expected_keys = ['distances', 'elevations', 'coordinates', 'stats', 'total_distance']
        for key in expected_keys:
            self.assertIn(key, result)
        
        self.assertEqual(result['distances'], [0, 0.1, 0.25, 0.4])
        self.assertEqual(result['elevations'], [610, 625, 620, 635])
    
    def test_format_elevation_profile_data_cli(self):
        """Test elevation profile formatting for CLI"""
        profile_data = {
            'elevation_stats': {
                'min_elevation': 610,
                'max_elevation': 635,
                'elevation_range': 25,
                'max_grade': 8.5,
                'steep_section_count': 2
            }
        }
        
        result = self.formatter.format_elevation_profile_data(profile_data, 'cli')
        
        self.assertIsInstance(result, str)
        self.assertIn('Elevation Profile:', result)
        self.assertIn('Min Elevation: 610m', result)
        self.assertIn('Max Elevation: 635m', result)
        self.assertIn('Max Grade: 8.5%', result)
    
    def test_format_elevation_profile_data_json(self):
        """Test elevation profile formatting for JSON"""
        profile_data = {'test': 'data'}
        
        result = self.formatter.format_elevation_profile_data(profile_data, 'json')
        
        self.assertEqual(result, profile_data)
    
    def test_format_elevation_profile_data_empty(self):
        """Test elevation profile formatting with empty data"""
        result = self.formatter.format_elevation_profile_data({}, 'web')
        self.assertEqual(result, {})
    
    @patch('route_services.route_formatter.RouteFormatter._get_timestamp')
    def test_export_route_json(self, mock_timestamp):
        """Test JSON export functionality"""
        mock_timestamp.return_value = '2023-12-01T12:00:00'
        
        result = self.formatter.export_route_json(
            self.sample_route_result,
            self.sample_analysis,
            self.sample_directions,
            {'test': 'profile_data'}
        )
        
        # Should return valid JSON string
        self.assertIsInstance(result, str)
        
        # Parse and check structure
        data = json.loads(result)
        expected_keys = ['route_result', 'analysis', 'directions', 'elevation_profile', 'export_timestamp', 'format_version']
        for key in expected_keys:
            self.assertIn(key, data)
        
        self.assertEqual(data['route_result'], self.sample_route_result)
        self.assertEqual(data['export_timestamp'], '2023-12-01T12:00:00')
        self.assertEqual(data['format_version'], '1.0')
    
    def test_format_route_summary_cli(self):
        """Test CLI route summary formatting"""
        result = self.formatter.format_route_summary(self.sample_route_result, 'cli')
        
        expected = "🏃 2.5km route • 120m elevation gain • ~15min"
        self.assertEqual(result, expected)
    
    def test_format_route_summary_web(self):
        """Test web route summary formatting"""
        result = self.formatter.format_route_summary(self.sample_route_result, 'web')
        
        expected = "2.5km • 120m gain • ~15min"
        self.assertEqual(result, expected)
    
    def test_format_route_summary_empty(self):
        """Test route summary with empty data"""
        result = self.formatter.format_route_summary(None)
        self.assertEqual(result, "No route data")
    
    def test_create_difficulty_badge(self):
        """Test difficulty badge creation"""
        badge = self.formatter.create_difficulty_badge(self.sample_analysis)
        
        expected_keys = ['text', 'score', 'color', 'description']
        for key in expected_keys:
            self.assertIn(key, badge)
        
        self.assertEqual(badge['text'], 'Moderate')
        self.assertEqual(badge['score'], '45/100')
        self.assertEqual(badge['color'], '#FF9800')  # Orange for moderate
        self.assertEqual(badge['description'], 'Difficulty: Moderate (45/100)')
    
    def test_create_difficulty_badge_all_levels(self):
        """Test difficulty badge for all difficulty levels"""
        difficulty_levels = [
            ('Very Easy', 10, '#4CAF50'),
            ('Easy', 20, '#8BC34A'),
            ('Moderate', 40, '#FF9800'),
            ('Hard', 60, '#FF5722'),
            ('Very Hard', 80, '#F44336'),
            ('Unknown', 0, '#9E9E9E')
        ]
        
        for rating, score, expected_color in difficulty_levels:
            analysis = {
                'difficulty': {
                    'rating': rating,
                    'score': score
                }
            }
            
            badge = self.formatter.create_difficulty_badge(analysis)
            self.assertEqual(badge['text'], rating)
            self.assertEqual(badge['color'], expected_color)
    
    def test_create_difficulty_badge_empty(self):
        """Test difficulty badge with empty analysis"""
        badge = self.formatter.create_difficulty_badge({})
        
        self.assertEqual(badge['text'], 'Unknown')
        self.assertEqual(badge['score'], '0/100')
        self.assertEqual(badge['color'], '#9E9E9E')
    
    @patch('datetime.datetime')
    def test_get_timestamp(self, mock_datetime):
        """Test timestamp generation"""
        mock_datetime.now.return_value.isoformat.return_value = '2023-12-01T12:00:00'
        
        timestamp = self.formatter._get_timestamp()
        self.assertEqual(timestamp, '2023-12-01T12:00:00')


if __name__ == '__main__':
    unittest.main()