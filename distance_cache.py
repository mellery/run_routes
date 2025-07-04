#!/usr/bin/env python3
"""
Distance Matrix Caching System
Persistent cache for computed distance matrices to avoid recalculation
"""

import os
import pickle
import hashlib
import time
from typing import Dict, List, Optional, Tuple
import networkx as nx


class DistanceMatrixCache:
    """Manages persistent caching of distance matrices"""
    
    def __init__(self, cache_dir: str = "cache/distance_matrices"):
        """Initialize distance matrix cache
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        self.ensure_cache_directory()
        
    def ensure_cache_directory(self):
        """Create cache directory if it doesn't exist"""
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def generate_cache_key(self, nodes: List[int], objective: str, graph_hash: str) -> str:
        """Generate unique cache key for distance matrix
        
        Args:
            nodes: List of node IDs in the matrix
            objective: Route objective (minimize_distance, maximize_elevation, etc.)
            graph_hash: Hash of the graph structure/data
            
        Returns:
            Unique cache key string
        """
        # Create deterministic hash from nodes, objective, and graph
        nodes_str = ",".join(map(str, sorted(nodes)))
        cache_content = f"{nodes_str}|{objective}|{graph_hash}"
        
        # Generate SHA-256 hash for cache key
        cache_key = hashlib.sha256(cache_content.encode()).hexdigest()[:16]
        return cache_key
        
    def get_graph_hash(self, graph: nx.Graph, nodes: List[int]) -> str:
        """Generate hash of relevant graph data
        
        Args:
            graph: NetworkX graph
            nodes: Nodes to include in hash
            
        Returns:
            Hash representing graph structure and edge data
        """
        # Create hash from graph structure and edge data for relevant nodes
        graph_data = []
        
        # Include node coordinates (affects haversine distances)
        for node in sorted(nodes):
            if node in graph.nodes:
                node_data = graph.nodes[node]
                graph_data.append(f"{node}:{node_data.get('x', 0):.6f},{node_data.get('y', 0):.6f}")
        
        # Include edge data for paths between nodes (sample to avoid huge hash)
        sample_nodes = sorted(nodes)[:min(50, len(nodes))]  # Sample first 50 nodes
        for i, u in enumerate(sample_nodes):
            for v in sample_nodes[i+1:i+6]:  # Check edges to next 5 nodes
                if graph.has_edge(u, v):
                    edge_data = graph.get_edge_data(u, v)
                    if isinstance(edge_data, dict) and 0 in edge_data:
                        edge_data = edge_data[0]
                    length = edge_data.get('length', 0)
                    elev_gain = edge_data.get('elevation_gain', 0)
                    graph_data.append(f"{u}-{v}:{length:.1f},{elev_gain:.1f}")
        
        # Generate hash
        graph_content = "|".join(graph_data)
        graph_hash = hashlib.sha256(graph_content.encode()).hexdigest()[:16]
        return graph_hash
        
    def get_cache_filename(self, cache_key: str) -> str:
        """Get cache filename for given key
        
        Args:
            cache_key: Cache key
            
        Returns:
            Full path to cache file
        """
        return os.path.join(self.cache_dir, f"matrix_{cache_key}.pkl")
        
    def save_distance_matrix(self, nodes: List[int], objective: str, graph: nx.Graph,
                           distance_matrix: Dict, elevation_matrix: Dict, 
                           running_weight_matrix: Dict) -> str:
        """Save distance matrices to cache
        
        Args:
            nodes: List of node IDs
            objective: Route objective
            graph: NetworkX graph
            distance_matrix: Distance matrix dictionary
            elevation_matrix: Elevation gain matrix dictionary  
            running_weight_matrix: Running weight matrix dictionary
            
        Returns:
            Cache key used for storage
        """
        try:
            # Generate cache key
            graph_hash = self.get_graph_hash(graph, nodes)
            cache_key = self.generate_cache_key(nodes, objective, graph_hash)
            cache_file = self.get_cache_filename(cache_key)
            
            # Prepare cache data
            cache_data = {
                'nodes': nodes,
                'objective': objective,
                'graph_hash': graph_hash,
                'distance_matrix': distance_matrix,
                'elevation_matrix': elevation_matrix,
                'running_weight_matrix': running_weight_matrix,
                'creation_time': time.time(),
                'node_count': len(nodes),
                'matrix_size': len(nodes) ** 2
            }
            
            # Save to cache file
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            print(f"  💾 Distance matrix cached: {cache_key} ({len(nodes)} nodes)")
            return cache_key
            
        except Exception as e:
            print(f"  ⚠️ Cache save failed: {e}")
            return None
            
    def load_distance_matrix(self, nodes: List[int], objective: str, 
                           graph: nx.Graph) -> Optional[Tuple[Dict, Dict, Dict]]:
        """Load distance matrices from cache if available
        
        Args:
            nodes: List of node IDs
            objective: Route objective
            graph: NetworkX graph
            
        Returns:
            Tuple of (distance_matrix, elevation_matrix, running_weight_matrix) or None
        """
        try:
            # Generate cache key
            graph_hash = self.get_graph_hash(graph, nodes)
            cache_key = self.generate_cache_key(nodes, objective, graph_hash)
            cache_file = self.get_cache_filename(cache_key)
            
            # Check if cache file exists
            if not os.path.exists(cache_file):
                return None
                
            # Load cache data
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Validate cache data
            if (cache_data.get('nodes') == nodes and 
                cache_data.get('objective') == objective and
                cache_data.get('graph_hash') == graph_hash):
                
                # Cache hit!
                creation_time = cache_data.get('creation_time', 0)
                age_hours = (time.time() - creation_time) / 3600
                
                print(f"  🚀 Cache HIT: {cache_key} ({len(nodes)} nodes, {age_hours:.1f}h old)")
                
                return (
                    cache_data['distance_matrix'],
                    cache_data['elevation_matrix'], 
                    cache_data['running_weight_matrix']
                )
            else:
                # Cache key collision or outdated
                print(f"  ⚠️ Cache key collision or outdated: {cache_key}")
                return None
                
        except Exception as e:
            print(f"  ⚠️ Cache load failed: {e}")
            return None
            
    def get_cache_info(self) -> Dict:
        """Get information about cached distance matrices
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.startswith('matrix_') and f.endswith('.pkl')]
            
            total_size = 0
            cache_entries = []
            
            for cache_file in cache_files:
                cache_path = os.path.join(self.cache_dir, cache_file)
                file_size = os.path.getsize(cache_path)
                total_size += file_size
                
                try:
                    with open(cache_path, 'rb') as f:
                        cache_data = pickle.load(f)
                        
                    cache_entries.append({
                        'key': cache_file.replace('matrix_', '').replace('.pkl', ''),
                        'nodes': cache_data.get('node_count', 0),
                        'objective': cache_data.get('objective', 'unknown'),
                        'age_hours': (time.time() - cache_data.get('creation_time', 0)) / 3600,
                        'size_mb': file_size / (1024 * 1024)
                    })
                except:
                    # Skip corrupted cache files
                    pass
                    
            return {
                'cache_dir': self.cache_dir,
                'total_files': len(cache_files),
                'total_size_mb': total_size / (1024 * 1024),
                'entries': cache_entries
            }
            
        except Exception as e:
            return {'error': str(e)}
            
    def clean_old_cache(self, max_age_hours: float = 168):  # 7 days default
        """Clean old cache entries
        
        Args:
            max_age_hours: Maximum age in hours before cache entry is removed
        """
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.startswith('matrix_') and f.endswith('.pkl')]
            
            removed_count = 0
            current_time = time.time()
            
            for cache_file in cache_files:
                cache_path = os.path.join(self.cache_dir, cache_file)
                
                try:
                    with open(cache_path, 'rb') as f:
                        cache_data = pickle.load(f)
                        
                    creation_time = cache_data.get('creation_time', 0)
                    age_hours = (current_time - creation_time) / 3600
                    
                    if age_hours > max_age_hours:
                        os.remove(cache_path)
                        removed_count += 1
                        print(f"  🗑️ Removed old cache: {cache_file} ({age_hours:.1f}h old)")
                        
                except:
                    # Remove corrupted cache files
                    os.remove(cache_path)
                    removed_count += 1
                    print(f"  🗑️ Removed corrupted cache: {cache_file}")
                    
            if removed_count > 0:
                print(f"  ✅ Cache cleanup: removed {removed_count} old entries")
            else:
                print(f"  ✅ Cache cleanup: no old entries found")
                
        except Exception as e:
            print(f"  ⚠️ Cache cleanup failed: {e}")


def test_distance_cache():
    """Test the distance cache functionality"""
    print("=== DISTANCE CACHE TEST ===")
    
    cache = DistanceMatrixCache()
    
    # Test cache info
    info = cache.get_cache_info()
    print(f"Cache directory: {info.get('cache_dir')}")
    print(f"Existing cache files: {info.get('total_files', 0)}")
    print(f"Total cache size: {info.get('total_size_mb', 0):.2f} MB")
    
    # Test cache key generation
    test_nodes = [1, 2, 3, 4, 5]
    test_objective = "minimize_distance"
    test_graph_hash = "abc123"
    
    cache_key = cache.generate_cache_key(test_nodes, test_objective, test_graph_hash)
    print(f"Generated cache key: {cache_key}")
    
    print("✅ Distance cache system ready")


if __name__ == "__main__":
    test_distance_cache()