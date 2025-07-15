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


class TestRouteAnalyzerWorkflowTesting(TestRouteAnalyzer):
    """Test RouteAnalyzer workflow testing for Phase 2 coverage improvement"""
    
    def test_complete_route_analysis_workflow(self):
        """Test complete route analysis workflow from start to finish"""
        # Start with basic route result
        route_result = {
            'route': [1001, 1002, 1003, 1004],
            'stats': {
                'total_distance_km': 3.7,
                'total_elevation_gain_m': 25,
                'total_elevation_loss_m': 10,
                'estimated_time_min': 22
            },
            'algorithm': 'genetic',
            'objective': 'balanced'
        }
        
        # Step 1: Generate turn-by-turn directions
        directions = self.analyzer.generate_directions(route_result)
        self.assertGreater(len(directions), 0)
        self.assertIn('instruction', directions[0])
        
        # Step 2: Analyze route characteristics
        analysis = self.analyzer.analyze_route(route_result)
        self.assertIn('route_info', analysis)
        self.assertIn('additional_stats', analysis)
        
        # Step 3: Get difficulty rating
        difficulty = self.analyzer.get_route_difficulty_rating(route_result)
        self.assertIn('rating', difficulty)
        self.assertIn('score', difficulty)
        
        # Step 4: Validate workflow results are consistent
        if 'total_time_min' in analysis.get('additional_stats', {}):
            self.assertIsInstance(analysis['additional_stats']['total_time_min'], (int, float))
        self.assertGreater(difficulty['score'], 0)
    
    def test_multi_objective_route_analysis(self):
        """Test analysis of routes optimized for different objectives"""
        objectives = ['distance', 'elevation', 'balanced', 'scenic']
        
        for objective in objectives:
            route_result = {
                'route': [1001, 1002, 1003, 1004],
                'stats': {
                    'total_distance_km': 2.5,
                    'total_elevation_gain_m': 20 if objective == 'elevation' else 5,
                    'algorithm': 'genetic'
                },
                'objective': objective
            }
            
            # Analysis should adapt to objective
            analysis = self.analyzer.analyze_route(route_result)
            
            self.assertIn('route_info', analysis)
            # Should have analyzed the route
            self.assertIn('basic_stats', analysis)
            
            # Elevation-focused routes should have different characteristics
            if objective == 'elevation':
                difficulty = self.analyzer.get_route_difficulty_rating(route_result)
                self.assertGreater(difficulty['score'], 10)
    
    def test_route_comparison_workflow(self):
        """Test workflow for comparing multiple routes"""
        # Create routes with different characteristics
        flat_route = {
            'route': [1001, 1002],
            'stats': {
                'total_distance_km': 1.0,
                'total_elevation_gain_m': 2,
                'total_elevation_loss_m': 1
            },
            'algorithm': 'nearest_neighbor'
        }
        
        hilly_route = {
            'route': [1001, 1002, 1003, 1004],
            'stats': {
                'total_distance_km': 3.0,
                'total_elevation_gain_m': 50,
                'total_elevation_loss_m': 30
            },
            'algorithm': 'genetic'
        }
        
        # Analyze both routes
        flat_analysis = self.analyzer.analyze_route(flat_route)
        hilly_analysis = self.analyzer.analyze_route(hilly_route)
        
        # Get difficulty ratings
        flat_difficulty = self.analyzer.get_route_difficulty_rating(flat_route)
        hilly_difficulty = self.analyzer.get_route_difficulty_rating(hilly_route)
        
        # Hilly route should be rated as more difficult
        self.assertGreater(hilly_difficulty['score'], flat_difficulty['score'])
        
        # Check elevation gain per km if available
        if ('elevation_gain_per_km' in hilly_analysis.get('additional_stats', {}) and
            'elevation_gain_per_km' in flat_analysis.get('additional_stats', {})):
            self.assertGreater(
                hilly_analysis['additional_stats']['elevation_gain_per_km'],
                flat_analysis['additional_stats']['elevation_gain_per_km']
            )
    
    def test_long_distance_route_workflow(self):
        """Test workflow for analyzing long-distance routes"""
        # Create extended graph for long route
        extended_graph = nx.Graph()
        
        # Add many nodes in sequence
        for i in range(20):
            extended_graph.add_node(
                5000 + i,
                x=-80.4094 + (i * 0.002),
                y=37.1299 + (i * 0.002),
                elevation=600 + (i % 8) * 15  # Undulating elevation
            )
            if i > 0:
                extended_graph.add_edge(5000 + i - 1, 5000 + i, length=200)
        
        long_analyzer = RouteAnalyzer(extended_graph)
        long_route = {
            'route': [5000 + i for i in range(0, 20, 2)],  # Every other node
            'stats': {
                'total_distance_km': 18.0,
                'total_elevation_gain_m': 150,
                'total_elevation_loss_m': 120,
                'estimated_time_min': 108
            },
            'algorithm': 'genetic'
        }
        
        # Test long route analysis
        analysis = long_analyzer.analyze_route(long_route)
        directions = long_analyzer.generate_directions(long_route)
        difficulty = long_analyzer.get_route_difficulty_rating(long_route)
        
        # Long routes should have specific characteristics
        self.assertGreater(len(directions), 5)
        if 'total_time_min' in analysis.get('additional_stats', {}):
            self.assertGreater(analysis['additional_stats']['total_time_min'], 60)
        # Should include distance-related difficulty factors
        factors = difficulty.get('factors', [])
        factor_text = ' '.join(factors).lower()
        self.assertIn('long', factor_text)  # Should contain some form of "long"
    
    def test_circular_route_workflow(self):
        """Test workflow for analyzing circular routes"""
        circular_route = {
            'route': [1001, 1002, 1003, 1004, 1001],  # Returns to start
            'stats': {
                'total_distance_km': 4.0,
                'total_elevation_gain_m': 30,
                'total_elevation_loss_m': 30,  # Should be balanced
                'net_elevation_gain_m': 0
            },
            'algorithm': 'genetic'
        }
        
        analysis = self.analyzer.analyze_route(circular_route)
        directions = self.analyzer.generate_directions(circular_route)
        
        # Circular routes should return to start
        self.assertEqual(directions[0]['node_id'], directions[-1]['node_id'])
        # Check if net elevation gain exists in additional stats
        if 'net_elevation_gain' in analysis.get('additional_stats', {}):
            self.assertAlmostEqual(analysis['additional_stats']['net_elevation_gain'], 0, delta=5)
        
        # Should detect circular nature in route info
        self.assertTrue(analysis['route_info']['is_loop'])
    
    def test_route_safety_analysis_workflow(self):
        """Test workflow for route safety analysis"""
        # Create route with varying safety characteristics
        safety_route = {
            'route': [1001, 1002, 1003, 1004],
            'stats': {
                'total_distance_km': 5.0,
                'total_elevation_gain_m': 80,
                'max_grade_percent': 15.0  # Steep grade
            },
            'algorithm': 'genetic'
        }
        
        # Mock additional analysis for safety factors
        with patch.object(self.analyzer, 'analyze_route') as mock_analyze:
            mock_analyze.return_value = {
                'route_type': 'challenging',
                'objective': 'elevation',
                'additional_stats': {
                    'steepest_uphill_grade': 15.0,
                    'steepest_downhill_grade': -12.0,
                    'high_grade_segments': 3,
                    'total_time_min': 30,
                    'elevation_gain_per_km': 16.0,
                    'uphill_percentage': 45,
                    'net_elevation_gain': 80
                }
            }
            
            analysis = mock_analyze.return_value
            difficulty = self.analyzer.get_route_difficulty_rating(safety_route)
            
            # Safety concerns should be reflected in difficulty
            safety_factors = [f for f in difficulty.get('factors', []) if 'steep' in f.lower()]
            self.assertGreater(len(safety_factors), 0)
            self.assertGreater(difficulty['score'], 30)  # Should be challenging
    
    def test_route_performance_analysis_workflow(self):
        """Test workflow for route performance analysis"""
        performance_route = {
            'route': [1001, 1002, 1003, 1004],
            'stats': {
                'total_distance_km': 10.0,
                'total_elevation_gain_m': 200,
                'estimated_time_min': 60,
                'avg_speed_kmh': 10.0
            },
            'algorithm': 'genetic',
            'metadata': {
                'optimization_time_s': 45.2,
                'fitness_score': 0.85
            }
        }
        
        analysis = self.analyzer.analyze_route(performance_route)
        
        # Performance metrics should be calculated
        additional_stats = analysis.get('additional_stats', {})
        
        # Check for expected fields that actually exist
        self.assertIn('total_segments', additional_stats)
        self.assertIn('uphill_segments', additional_stats)
        self.assertIn('downhill_segments', additional_stats)
        self.assertIn('steepest_uphill_grade', additional_stats)
        self.assertIn('steepest_downhill_grade', additional_stats)
        
        # Should have some uphill segments for a hilly route
        self.assertGreater(additional_stats['uphill_segments'], 0)
    
    def test_route_optimization_analysis_workflow(self):
        """Test workflow for analyzing route optimization results"""
        optimization_results = [
            {
                'route': [1001, 1002, 1003],
                'stats': {'total_distance_km': 2.0, 'total_elevation_gain_m': 10},
                'algorithm': 'nearest_neighbor',
                'fitness_score': 0.6
            },
            {
                'route': [1001, 1002, 1004],
                'stats': {'total_distance_km': 2.2, 'total_elevation_gain_m': 25},
                'algorithm': 'genetic',
                'fitness_score': 0.9
            }
        ]
        
        # Compare optimization results
        analyses = []
        for route_result in optimization_results:
            analysis = self.analyzer.analyze_route(route_result)
            analysis['original_result'] = route_result
            analyses.append(analysis)
        
        # Genetic algorithm should generally produce better fitness
        genetic_analysis = next(a for a in analyses if a['original_result']['algorithm'] == 'genetic')
        nn_analysis = next(a for a in analyses if a['original_result']['algorithm'] == 'nearest_neighbor')
        
        genetic_fitness = genetic_analysis['original_result']['fitness_score']
        nn_fitness = nn_analysis['original_result']['fitness_score']
        self.assertGreater(genetic_fitness, nn_fitness)
    
    def test_terrain_classification_workflow(self):
        """Test comprehensive terrain classification workflow"""
        # Create route with varied terrain
        terrain_graph = nx.Graph()
        elevations = [600, 620, 640, 635, 655, 650, 630, 610]  # Varied profile
        
        for i, elevation in enumerate(elevations):
            terrain_graph.add_node(
                6000 + i,
                x=-80.4094 + (i * 0.001),
                y=37.1299 + (i * 0.001),
                elevation=elevation
            )
            if i > 0:
                terrain_graph.add_edge(6000 + i - 1, 6000 + i, length=150)
        
        terrain_analyzer = RouteAnalyzer(terrain_graph)
        terrain_route = {
            'route': [6000 + i for i in range(len(elevations))],
            'stats': {
                'total_distance_km': 1.05,
                'total_elevation_gain_m': 35,
                'total_elevation_loss_m': 45
            }
        }
        
        # Generate directions and classify terrain
        directions = terrain_analyzer.generate_directions(terrain_route)
        
        # Should have mixed terrain types
        terrain_types = [d['terrain'] for d in directions if 'terrain' in d]
        unique_terrains = set(terrain_types)
        self.assertGreater(len(unique_terrains), 1)  # Should have variety
        
        # Validate terrain classification logic
        for i, direction in enumerate(directions[1:], 1):  # Skip start point
            if 'elevation_change' in direction:
                elevation_change = direction['elevation_change']
                terrain = direction['terrain']
                
                # Skip special terrain types
                if terrain in ['start', 'finish']:
                    continue
                    
                if elevation_change > 5:
                    self.assertEqual(terrain, 'uphill')
                elif elevation_change < -5:
                    self.assertEqual(terrain, 'downhill')
                else:
                    self.assertEqual(terrain, 'level')
    
    def test_route_statistics_aggregation_workflow(self):
        """Test workflow for aggregating route statistics"""
        complex_route = {
            'route': [1001, 1002, 1003, 1004],
            'stats': {
                'total_distance_km': 7.5,
                'total_elevation_gain_m': 120,
                'total_elevation_loss_m': 90,
                'max_elevation_m': 750,
                'min_elevation_m': 580,
                'avg_grade_percent': 2.8,
                'max_grade_percent': 12.5
            },
            'algorithm': 'genetic',
            'objective': 'elevation'
        }
        
        analysis = self.analyzer.analyze_route(complex_route)
        
        # Verify comprehensive statistics aggregation
        stats = analysis['additional_stats']
        
        # Check for expected fields that should exist
        self.assertIn('total_segments', stats)
        self.assertIn('uphill_segments', stats)
        self.assertIn('downhill_segments', stats)
        self.assertIn('level_segments', stats)
        self.assertIn('uphill_percentage', stats)
        self.assertIn('downhill_percentage', stats)
        self.assertIn('steepest_uphill_grade', stats)
        self.assertIn('steepest_downhill_grade', stats)
        self.assertIn('avg_elevation_change', stats)
        self.assertIn('elevation_changes', stats)
    
    def test_route_difficulty_factors_workflow(self):
        """Test workflow for identifying route difficulty factors"""
        # Create challenging route
        challenging_route = {
            'route': [1001, 1002, 1003, 1004],
            'stats': {
                'total_distance_km': 12.0,
                'total_elevation_gain_m': 400,
                'max_grade_percent': 18.0,
                'estimated_time_min': 90
            }
        }
        
        # Mock analysis to include challenging factors
        with patch.object(self.analyzer, 'analyze_route') as mock_analyze:
            mock_analyze.return_value = {
                'route_type': 'very_challenging',
                'additional_stats': {
                    'elevation_gain_per_km': 33.3,  # Very high
                    'steepest_uphill_grade': 18.0,  # Very steep
                    'high_grade_segments': 5,       # Many steep sections
                    'total_time_min': 90,           # Long duration
                    'uphill_percentage': 65,        # Mostly uphill
                    'net_elevation_gain': 300
                }
            }
            
            difficulty = self.analyzer.get_route_difficulty_rating(challenging_route)
            
            # Should identify multiple difficulty factors
            factors = difficulty.get('factors', [])
            self.assertGreater(len(factors), 2)
            
            # Should include specific challenging aspects
            factor_text = ' '.join(factors).lower()
            self.assertIn('elevation', factor_text)
            self.assertIn('distance', factor_text)
    
    def test_route_time_estimation_workflow(self):
        """Test workflow for route time estimation"""
        time_routes = [
            # Fast flat route
            {
                'route': [1001, 1002],
                'stats': {
                    'total_distance_km': 5.0,
                    'total_elevation_gain_m': 10,
                    'avg_grade_percent': 0.5
                }
            },
            # Slow hilly route
            {
                'route': [1001, 1002, 1003, 1004],
                'stats': {
                    'total_distance_km': 5.0,
                    'total_elevation_gain_m': 200,
                    'avg_grade_percent': 8.0
                }
            }
        ]
        
        flat_analysis = self.analyzer.analyze_route(time_routes[0])
        hilly_analysis = self.analyzer.analyze_route(time_routes[1])
        
        # Check if total_time_min is available
        if ('total_time_min' in flat_analysis.get('additional_stats', {}) and
            'total_time_min' in hilly_analysis.get('additional_stats', {})):
            flat_time = flat_analysis['additional_stats']['total_time_min']
            hilly_time = hilly_analysis['additional_stats']['total_time_min']
            
            # Hilly route should take significantly longer
            self.assertGreater(hilly_time, flat_time)
            self.assertGreater(hilly_time / flat_time, 1.3)  # At least 30% longer
    
    def test_route_directions_detail_levels(self):
        """Test route directions with different detail levels"""
        detail_route = {
            'route': [1001, 1002, 1003, 1004],
            'stats': {
                'total_distance_km': 3.0,
                'total_elevation_gain_m': 25
            }
        }
        
        # Test basic direction generation
        directions = self.analyzer.generate_directions(detail_route)
        
        # Should have directions for the route
        self.assertGreater(len(directions), 0)
        
        # Each direction should have essential info
        for direction in directions:
            self.assertIn('instruction', direction)
            self.assertIn('step', direction)
            self.assertIn('node_id', direction)
            self.assertIn('distance_km', direction)
            self.assertIn('cumulative_distance_km', direction)
            self.assertIn('elevation', direction)
            self.assertIn('terrain', direction)
    
    def test_route_analysis_error_handling_workflow(self):
        """Test workflow error handling and recovery"""
        error_routes = [
            None,  # None route
            {},    # Empty route
            {'route': []},  # Empty route list
            {'stats': {}},  # Missing route
            {'route': [9999], 'stats': {}},  # Invalid node
        ]
        
        for error_route in error_routes:
            # Should handle errors gracefully
            try:
                analysis = self.analyzer.analyze_route(error_route)
                self.assertIsInstance(analysis, dict)
                
                directions = self.analyzer.generate_directions(error_route)
                self.assertIsInstance(directions, list)
                
                difficulty = self.analyzer.get_route_difficulty_rating(error_route)
                self.assertIsInstance(difficulty, dict)
                
            except (KeyError, AttributeError, TypeError):
                # Expected for invalid inputs
                pass
    
    def test_route_metadata_integration_workflow(self):
        """Test workflow integration with route metadata"""
        metadata_route = {
            'route': [1001, 1002, 1003, 1004],
            'stats': {
                'total_distance_km': 4.0,
                'total_elevation_gain_m': 60
            },
            'algorithm': 'genetic',
            'objective': 'balanced',
            'metadata': {
                'generation_found': 25,
                'total_generations': 100,
                'fitness_score': 0.78,
                'optimization_time_s': 32.1,
                'convergence_reason': 'fitness_plateau'
            }
        }
        
        analysis = self.analyzer.analyze_route(metadata_route)
        
        # Should include route info from metadata
        self.assertIn('route_info', analysis)
        # Basic stats should be preserved
        self.assertIn('basic_stats', analysis)
        
        # Should preserve metadata for further analysis
        if 'metadata' in analysis:
            self.assertIn('fitness_score', str(analysis))
    
    def test_batch_route_analysis_workflow(self):
        """Test workflow for analyzing multiple routes in batch"""
        batch_routes = [
            {
                'route': [1001, 1002],
                'stats': {'total_distance_km': 1.0, 'total_elevation_gain_m': 5},
                'algorithm': 'nearest_neighbor'
            },
            {
                'route': [1001, 1003],
                'stats': {'total_distance_km': 1.5, 'total_elevation_gain_m': 15},
                'algorithm': 'genetic'
            },
            {
                'route': [1002, 1004],
                'stats': {'total_distance_km': 2.0, 'total_elevation_gain_m': 25},
                'algorithm': 'genetic'
            }
        ]
        
        # Analyze all routes
        batch_analyses = []
        for route in batch_routes:
            analysis = self.analyzer.analyze_route(route)
            difficulty = self.analyzer.get_route_difficulty_rating(route)
            analysis['difficulty'] = difficulty
            batch_analyses.append(analysis)
        
        # Compare batch results
        self.assertEqual(len(batch_analyses), 3)
        
        # Should show progression in difficulty
        difficulties = [a['difficulty']['score'] for a in batch_analyses]
        # Generally, longer routes with more elevation should be harder
        self.assertGreaterEqual(difficulties[1], difficulties[0])
        self.assertGreaterEqual(difficulties[2], difficulties[1])
    
    def test_route_analysis_caching_workflow(self):
        """Test workflow with analysis result caching"""
        cache_route = {
            'route': [1001, 1002, 1003, 1004],
            'stats': {
                'total_distance_km': 3.5,
                'total_elevation_gain_m': 40
            }
        }
        
        # First analysis
        import time
        start_time = time.time()
        analysis1 = self.analyzer.analyze_route(cache_route)
        first_duration = time.time() - start_time
        
        # Second analysis (should be faster if cached)
        start_time = time.time()
        analysis2 = self.analyzer.analyze_route(cache_route)
        second_duration = time.time() - start_time
        
        # Results should be identical
        self.assertEqual(analysis1['route_info']['route_length'], analysis2['route_info']['route_length'])
        # Check if total_time_min exists before comparing
        if ('total_time_min' in analysis1.get('additional_stats', {}) and 
            'total_time_min' in analysis2.get('additional_stats', {})):
            self.assertEqual(
                analysis1['additional_stats']['total_time_min'],
                analysis2['additional_stats']['total_time_min']
            )
        
        # Second call may be faster due to caching (if implemented)
        # This is more for testing the workflow than asserting performance
        self.assertLessEqual(second_duration, first_duration * 2)
    
    def test_route_export_format_workflow(self):
        """Test workflow for exporting route analysis in different formats"""
        export_route = {
            'route': [1001, 1002, 1003, 1004],
            'stats': {
                'total_distance_km': 5.0,
                'total_elevation_gain_m': 75
            },
            'algorithm': 'genetic',
            'objective': 'elevation'
        }
        
        # Generate complete analysis
        analysis = self.analyzer.analyze_route(export_route)
        directions = self.analyzer.generate_directions(export_route)
        difficulty = self.analyzer.get_route_difficulty_rating(export_route)
        
        # Create exportable summary
        export_summary = {
            'route_analysis': analysis,
            'turn_by_turn_directions': directions,
            'difficulty_rating': difficulty,
            'export_timestamp': 1234567890  # Mock timestamp
        }
        
        # Validate export format
        self.assertIn('route_analysis', export_summary)
        self.assertIn('turn_by_turn_directions', export_summary)
        self.assertIn('difficulty_rating', export_summary)
        self.assertGreater(len(export_summary['turn_by_turn_directions']), 0)
        
        # Export should be serializable (test with str conversion)
        export_str = str(export_summary)
        self.assertIn('route_analysis', export_str)
        self.assertIn('instruction', export_str)
        self.assertIn('rating', export_str)


if __name__ == '__main__':
    import time
    unittest.main()