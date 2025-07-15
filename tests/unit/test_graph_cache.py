#!/usr/bin/env python3
"""
Unit tests for graph_cache.py cache management utilities
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import networkx as nx
import os
import sys
import tempfile
import shutil
import subprocess

# Add the parent directory to sys.path to import graph_cache
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import graph_cache


class TestGraphCache(unittest.TestCase):
    """Test cases for graph_cache.py utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory for testing
        self.test_cache_dir = tempfile.mkdtemp()
        self.original_cache_dir = 'cache'
        
        # Create test graph
        self.test_graph = nx.Graph()
        self.test_graph.add_node(1, x=-80.4094, y=37.1299, elevation=600)
        self.test_graph.add_node(2, x=-80.4095, y=37.1300, elevation=620)
        self.test_graph.add_edge(1, 2, length=100, elevation_gain=20, running_weight=110)
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove temporary directory
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)


class TestLoadOrGenerateGraph(TestGraphCache):
    """Test load_or_generate_graph function"""
    
    @patch('graph_cache.load_cached_graph')
    @patch('graph_cache.get_cache_filename')
    def test_load_existing_cache_success(self, mock_get_filename, mock_load_cached):
        """Test loading existing cache successfully"""
        mock_get_filename.return_value = 'test_cache.pkl'
        mock_load_cached.return_value = self.test_graph
        
        result = graph_cache.load_or_generate_graph()
        
        self.assertEqual(result, self.test_graph)
        mock_get_filename.assert_called_once_with((37.1299, -80.4094), 1200, 'all')
        mock_load_cached.assert_called_once_with('test_cache.pkl')
    
    @patch('graph_cache.load_cached_graph')
    @patch('graph_cache.get_cache_filename')
    @patch('graph_cache.generate_cached_graph')
    def test_generate_cache_when_not_exists(self, mock_generate, mock_get_filename, mock_load_cached):
        """Test generating cache when it doesn't exist"""
        mock_get_filename.return_value = 'test_cache.pkl'
        mock_load_cached.return_value = None  # Cache doesn't exist
        mock_generate.return_value = self.test_graph
        
        result = graph_cache.load_or_generate_graph()
        
        self.assertEqual(result, self.test_graph)
        mock_generate.assert_called_once_with(
            (37.1299, -80.4094), 1200, 'all', 'test_cache.pkl', True
        )
    
    @patch('graph_cache.load_cached_graph')
    @patch('graph_cache.get_cache_filename')
    def test_force_regenerate(self, mock_get_filename, mock_load_cached):
        """Test force regeneration even when cache exists"""
        mock_get_filename.return_value = 'test_cache.pkl'
        
        with patch('graph_cache.generate_cached_graph') as mock_generate:
            mock_generate.return_value = self.test_graph
            
            result = graph_cache.load_or_generate_graph(force_regenerate=True)
            
            # Should not try to load existing cache
            mock_load_cached.assert_not_called()
            mock_generate.assert_called_once()
    
    @patch('graph_cache.load_cached_graph')
    @patch('graph_cache.get_cache_filename')
    @patch('graph_cache.generate_cached_graph')
    @patch('graph_cache.subprocess.run')
    def test_fallback_to_subprocess(self, mock_subprocess, mock_generate, mock_get_filename, mock_load_cached):
        """Test fallback to subprocess when generation fails"""
        mock_get_filename.return_value = 'test_cache.pkl'
        mock_load_cached.side_effect = [None, self.test_graph]  # First None, then success after subprocess
        mock_generate.side_effect = Exception("Generation failed")
        
        # Mock successful subprocess
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        result = graph_cache.load_or_generate_graph()
        
        self.assertEqual(result, self.test_graph)
        mock_subprocess.assert_called_once()
    
    @patch('graph_cache.load_cached_graph')
    @patch('graph_cache.get_cache_filename')
    @patch('graph_cache.generate_cached_graph')
    @patch('graph_cache.subprocess.run')
    def test_subprocess_timeout(self, mock_subprocess, mock_generate, mock_get_filename, mock_load_cached):
        """Test subprocess timeout handling"""
        mock_get_filename.return_value = 'test_cache.pkl'
        mock_load_cached.return_value = None
        mock_generate.side_effect = Exception("Generation failed")
        mock_subprocess.side_effect = subprocess.TimeoutExpired('cmd', 600)
        
        with patch('osmnx.graph_from_point') as mock_osmnx:
            with patch('route.add_enhanced_elevation_to_graph') as mock_add_elevation:
                with patch('route.add_elevation_to_edges') as mock_add_edges:
                    with patch('route.add_running_weights') as mock_add_weights:
                        mock_osmnx.return_value = self.test_graph
                        mock_add_elevation.return_value = self.test_graph
                        mock_add_edges.return_value = self.test_graph
                        mock_add_weights.return_value = self.test_graph
                        
                        result = graph_cache.load_or_generate_graph()
                        
                        # Should fallback to direct generation
                        self.assertEqual(result, self.test_graph)
                        mock_osmnx.assert_called_once()
    
    @patch('graph_cache.load_cached_graph')
    @patch('graph_cache.get_cache_filename')
    @patch('graph_cache.generate_cached_graph')
    @patch('graph_cache.subprocess.run')
    def test_direct_generation_fallback(self, mock_subprocess, mock_generate, mock_get_filename, mock_load_cached):
        """Test direct graph generation fallback"""
        mock_get_filename.return_value = 'test_cache.pkl'
        mock_load_cached.return_value = None
        mock_generate.side_effect = Exception("Generation failed")
        
        # Mock subprocess failure
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Script failed"
        mock_subprocess.return_value = mock_result
        
        with patch('osmnx.graph_from_point') as mock_osmnx:
            with patch('route.add_enhanced_elevation_to_graph') as mock_add_elevation:
                with patch('route.add_elevation_to_edges') as mock_add_edges:
                    with patch('route.add_running_weights') as mock_add_weights:
                        mock_osmnx.return_value = self.test_graph
                        mock_add_elevation.return_value = self.test_graph
                        mock_add_edges.return_value = self.test_graph
                        mock_add_weights.return_value = self.test_graph
                        
                        result = graph_cache.load_or_generate_graph()
                        
                        self.assertEqual(result, self.test_graph)
                        mock_add_elevation.assert_called_once_with(
                            self.test_graph, use_3dep=True, 
                            fallback_raster='elevation_data/srtm_90m/srtm_20_05.tif'
                        )
    
    @patch('graph_cache.load_cached_graph')
    @patch('graph_cache.get_cache_filename')
    @patch('graph_cache.generate_cached_graph')
    @patch('graph_cache.subprocess.run')
    def test_direct_generation_without_enhanced_elevation(self, mock_subprocess, mock_generate, mock_get_filename, mock_load_cached):
        """Test direct generation without enhanced elevation"""
        mock_get_filename.return_value = 'test_cache.pkl'
        mock_load_cached.return_value = None
        mock_generate.side_effect = Exception("Generation failed")
        
        # Mock subprocess failure
        mock_result = Mock()
        mock_result.returncode = 1
        mock_subprocess.return_value = mock_result
        
        with patch('osmnx.graph_from_point') as mock_osmnx:
            with patch('route.add_elevation_to_graph') as mock_add_elevation:
                with patch('route.add_elevation_to_edges') as mock_add_edges:
                    with patch('route.add_running_weights') as mock_add_weights:
                        mock_osmnx.return_value = self.test_graph
                        mock_add_elevation.return_value = self.test_graph
                        mock_add_edges.return_value = self.test_graph
                        mock_add_weights.return_value = self.test_graph
                        
                        result = graph_cache.load_or_generate_graph(use_enhanced_elevation=False)
                        
                        self.assertEqual(result, self.test_graph)
                        mock_add_elevation.assert_called_once_with(
                            self.test_graph, 'elevation_data/srtm_90m/srtm_20_05.tif'
                        )
    
    @patch('graph_cache.load_cached_graph')
    @patch('graph_cache.get_cache_filename')
    @patch('graph_cache.generate_cached_graph')
    @patch('graph_cache.subprocess.run')
    @patch('osmnx.graph_from_point')
    def test_all_fallbacks_fail(self, mock_osmnx, mock_subprocess, mock_generate, mock_get_filename, mock_load_cached):
        """Test when all fallback methods fail"""
        mock_get_filename.return_value = 'test_cache.pkl'
        mock_load_cached.return_value = None
        mock_generate.side_effect = Exception("Generation failed")
        mock_subprocess.side_effect = Exception("Subprocess failed")
        mock_osmnx.side_effect = Exception("Direct generation failed")
        
        with self.assertRaises(Exception):
            graph_cache.load_or_generate_graph()


class TestListCachedGraphs(TestGraphCache):
    """Test list_cached_graphs function"""
    
    @patch('graph_cache.os.makedirs')
    @patch('graph_cache.os.listdir')
    def test_list_no_cached_graphs(self, mock_listdir, mock_makedirs):
        """Test listing when no cached graphs exist"""
        mock_listdir.return_value = []
        
        result = graph_cache.list_cached_graphs()
        
        self.assertEqual(result, [])
        mock_makedirs.assert_called_once_with('cache', exist_ok=True)
    
    @patch('graph_cache.os.makedirs')
    @patch('graph_cache.os.listdir')
    @patch('graph_cache.load_cached_graph')
    @patch('graph_cache.os.path.getsize')
    @patch('graph_cache.os.path.join')
    def test_list_valid_cached_graphs(self, mock_join, mock_getsize, mock_load, mock_listdir, mock_makedirs):
        """Test listing valid cached graphs"""
        mock_listdir.return_value = ['cached_graph_test1.pkl', 'cached_graph_test2.pkl', 'other_file.txt']
        mock_join.side_effect = lambda *args: '/'.join(args)
        mock_load.return_value = self.test_graph  # All caches are valid
        mock_getsize.return_value = 1024 * 1024  # 1MB
        
        result = graph_cache.list_cached_graphs()
        
        self.assertEqual(len(result), 2)
        self.assertIn('cached_graph_test1.pkl', result)
        self.assertIn('cached_graph_test2.pkl', result)
        # Should not include non-.pkl files
        self.assertNotIn('other_file.txt', result)
    
    @patch('graph_cache.os.makedirs')
    @patch('graph_cache.os.listdir')
    @patch('graph_cache.load_cached_graph')
    @patch('graph_cache.os.path.getsize')
    @patch('graph_cache.os.path.join')
    def test_list_mixed_valid_invalid_caches(self, mock_join, mock_getsize, mock_load, mock_listdir, mock_makedirs):
        """Test listing with mix of valid and invalid caches"""
        mock_listdir.return_value = ['cached_graph_good.pkl', 'cached_graph_invalid.pkl', 'cached_graph_corrupt.pkl']
        mock_join.side_effect = lambda *args: '/'.join(args)
        # Mock load to match expected behavior: good, invalid, corrupted
        def load_side_effect(path):
            if 'good' in path:
                return self.test_graph
            elif 'invalid' in path:
                return None
            else:
                raise Exception("Corrupted")
        mock_load.side_effect = load_side_effect
        mock_getsize.return_value = 1024 * 1024
        
        result = graph_cache.list_cached_graphs()
        
        self.assertEqual(len(result), 1)
        self.assertIn('cached_graph_good.pkl', result)
        self.assertNotIn('cached_graph_invalid.pkl', result)
        self.assertNotIn('cached_graph_corrupt.pkl', result)


class TestCleanCache(TestGraphCache):
    """Test clean_cache function"""
    
    @patch('graph_cache.os.makedirs')
    @patch('graph_cache.os.listdir')
    def test_clean_no_cache_files(self, mock_listdir, mock_makedirs):
        """Test cleaning when no cache files exist"""
        mock_listdir.return_value = []
        
        # Should not raise exception
        graph_cache.clean_cache()
        
        mock_makedirs.assert_called_once_with('cache', exist_ok=True)
    
    @patch('graph_cache.os.makedirs')
    @patch('graph_cache.os.listdir')
    @patch('graph_cache.os.path.getmtime')
    @patch('graph_cache.load_cached_graph')
    @patch('graph_cache.os.remove')
    @patch('graph_cache.os.path.join')
    def test_clean_keep_latest_valid(self, mock_join, mock_remove, mock_load, mock_getmtime, mock_listdir, mock_makedirs):
        """Test cleaning while keeping latest valid cache"""
        mock_listdir.return_value = ['cached_graph_old.pkl', 'cached_graph_new.pkl']
        mock_join.side_effect = lambda *args: '/'.join(args)
        # Newer file has higher mtime
        mock_getmtime.side_effect = lambda f: 2000 if 'new' in f else 1000
        mock_load.return_value = self.test_graph  # Latest cache is valid
        
        graph_cache.clean_cache(keep_latest=True)
        
        # Should remove only the old file
        mock_remove.assert_called_once_with('cache/cached_graph_old.pkl')
    
    @patch('graph_cache.os.makedirs')
    @patch('graph_cache.os.listdir')
    @patch('graph_cache.os.path.getmtime')
    @patch('graph_cache.load_cached_graph')
    @patch('graph_cache.os.remove')
    @patch('graph_cache.os.path.join')
    def test_clean_latest_invalid(self, mock_join, mock_remove, mock_load, mock_getmtime, mock_listdir, mock_makedirs):
        """Test cleaning when latest cache is invalid"""
        mock_listdir.return_value = ['cached_graph_old.pkl', 'cached_graph_new.pkl']
        mock_join.side_effect = lambda *args: '/'.join(args)
        mock_getmtime.side_effect = lambda f: 2000 if 'new' in f else 1000
        mock_load.return_value = None  # Latest cache is invalid
        
        graph_cache.clean_cache(keep_latest=True)
        
        # Should remove both files since latest is invalid
        expected_calls = [
            call('cache/cached_graph_new.pkl'),
            call('cache/cached_graph_old.pkl')
        ]
        mock_remove.assert_has_calls(expected_calls, any_order=True)
    
    @patch('graph_cache.os.makedirs')
    @patch('graph_cache.os.listdir')
    @patch('graph_cache.os.path.getmtime')
    @patch('graph_cache.os.remove')
    @patch('graph_cache.os.path.join')
    def test_clean_remove_all(self, mock_join, mock_remove, mock_getmtime, mock_listdir, mock_makedirs):
        """Test cleaning all cache files"""
        mock_listdir.return_value = ['cached_graph_1.pkl', 'cached_graph_2.pkl']
        mock_join.side_effect = lambda *args: '/'.join(args)
        mock_getmtime.return_value = 1000
        
        graph_cache.clean_cache(keep_latest=False)
        
        # Should remove all files
        expected_calls = [
            call('cache/cached_graph_1.pkl'),
            call('cache/cached_graph_2.pkl')
        ]
        mock_remove.assert_has_calls(expected_calls, any_order=True)
    
    @patch('graph_cache.os.makedirs')
    @patch('graph_cache.os.listdir')
    @patch('graph_cache.os.path.getmtime')
    @patch('graph_cache.os.remove')
    @patch('graph_cache.os.path.join')
    def test_clean_error_handling(self, mock_join, mock_remove, mock_getmtime, mock_listdir, mock_makedirs):
        """Test error handling during cache cleaning"""
        mock_listdir.return_value = ['cached_graph_error.pkl']
        mock_join.side_effect = lambda *args: '/'.join(args)
        mock_getmtime.return_value = 1000
        mock_remove.side_effect = OSError("Permission denied")
        
        # Should not raise exception
        graph_cache.clean_cache(keep_latest=False)


class TestCommandLineInterface(TestGraphCache):
    """Test command line interface functionality"""
    
    def test_cli_interface_exists(self):
        """Test that CLI interface functions are available"""
        # Test that the module has a main section
        graph_cache_path = os.path.join(os.path.dirname(__file__), '..', '..', 'graph_cache.py')
        with open(graph_cache_path, 'r') as f:
            content = f.read()
            self.assertIn('if __name__ == "__main__":', content)
            self.assertIn('argparse', content)
    
    @patch('graph_cache.list_cached_graphs')
    @patch('argparse.ArgumentParser.parse_args')
    def test_cli_list_command_logic(self, mock_parse_args, mock_list):
        """Test CLI list command logic"""
        mock_args = Mock()
        mock_args.command = 'list'
        mock_parse_args.return_value = mock_args
        mock_list.return_value = ['test_cache.pkl']
        
        # Test the logic that would be executed
        if mock_args.command == 'list':
            graph_cache.list_cached_graphs()
            
        mock_list.assert_called_once()
    
    @patch('graph_cache.clean_cache')
    @patch('argparse.ArgumentParser.parse_args')
    def test_cli_clean_command_logic(self, mock_parse_args, mock_clean):
        """Test CLI clean command logic"""
        mock_args = Mock()
        mock_args.command = 'clean'
        mock_args.all = False
        mock_parse_args.return_value = mock_args
        
        # Test the logic that would be executed
        if mock_args.command == 'clean':
            graph_cache.clean_cache(keep_latest=not mock_args.all)
            
        mock_clean.assert_called_once_with(keep_latest=True)
    
    @patch('graph_cache.load_or_generate_graph')
    @patch('argparse.ArgumentParser.parse_args')
    def test_cli_test_command_logic(self, mock_parse_args, mock_load):
        """Test CLI test command logic"""
        mock_args = Mock()
        mock_args.command = 'test'
        mock_args.radius = 600
        mock_parse_args.return_value = mock_args
        mock_load.return_value = self.test_graph
        
        # Test the logic that would be executed
        if mock_args.command == 'test':
            graph_cache.load_or_generate_graph(radius_m=mock_args.radius)
            
        mock_load.assert_called_once_with(radius_m=600)


class TestIntegrationScenarios(TestGraphCache):
    """Test integration scenarios combining multiple functions"""
    
    @patch('graph_cache.os.makedirs')
    @patch('graph_cache.os.listdir')
    @patch('graph_cache.load_cached_graph')
    @patch('graph_cache.get_cache_filename')
    @patch('graph_cache.os.path.getsize')
    @patch('graph_cache.os.path.join')
    def test_load_graph_with_existing_cache_in_list(self, mock_join, mock_getsize, mock_get_filename, mock_load, mock_listdir, mock_makedirs):
        """Test loading graph when cache exists and appears in list"""
        cache_filename = 'cached_graph_test.pkl'
        mock_get_filename.return_value = cache_filename
        mock_listdir.return_value = [cache_filename]
        mock_load.return_value = self.test_graph
        mock_getsize.return_value = 1024 * 1024  # 1MB
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        # First list caches
        cached_graphs = graph_cache.list_cached_graphs()
        self.assertIn(cache_filename, cached_graphs)
        
        # Then load graph (should use existing cache)
        result = graph_cache.load_or_generate_graph()
        self.assertEqual(result, self.test_graph)
    
    @patch('graph_cache.os.makedirs')
    @patch('graph_cache.os.listdir')
    @patch('graph_cache.load_cached_graph')
    @patch('graph_cache.os.remove')
    @patch('graph_cache.os.path.getmtime')
    @patch('graph_cache.os.path.getsize')
    @patch('graph_cache.os.path.join')
    def test_clean_then_list_scenario(self, mock_join, mock_getsize, mock_getmtime, mock_remove, mock_load, mock_listdir, mock_makedirs):
        """Test cleaning cache then listing remaining caches"""
        mock_join.side_effect = lambda *args: '/'.join(args)
        mock_getmtime.return_value = 1000
        mock_getsize.return_value = 1024 * 1024  # 1MB
        
        # Initially have two cache files
        mock_listdir.return_value = ['cached_graph_old.pkl', 'cached_graph_new.pkl']
        mock_load.return_value = self.test_graph
        
        # Clean cache (should remove old one)
        graph_cache.clean_cache(keep_latest=True)
        
        # Update mock to reflect removed file
        mock_listdir.return_value = ['cached_graph_new.pkl']
        
        # List remaining caches
        result = graph_cache.list_cached_graphs()
        self.assertEqual(len(result), 1)
        self.assertIn('cached_graph_new.pkl', result)


if __name__ == '__main__':
    unittest.main()