#!/usr/bin/env python3
"""
Advanced Caching Strategy for OSMnx Networks
Multi-level caching with intelligent invalidation and compression
"""

import os
import pickle
import gzip
import hashlib
import json
import time
from typing import Dict, Any, Optional, Tuple
import networkx as nx
import logging

logger = logging.getLogger(__name__)

class AdvancedNetworkCache:
    """Advanced caching system for OSMnx networks"""
    
    def __init__(self, cache_dir: str = "cache/advanced"):
        self.cache_dir = cache_dir
        self.metadata_file = os.path.join(cache_dir, "cache_metadata.json")
        self.ensure_cache_dir()
        self.metadata = self.load_metadata()
    
    def ensure_cache_dir(self):
        """Ensure cache directory exists"""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"caches": {}, "last_cleanup": 0}
    
    def save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def generate_cache_key(self, center_point: Tuple[float, float], 
                          radius_m: int, 
                          network_type: str,
                          **kwargs) -> str:
        """
        Generate a unique cache key for network parameters
        
        Args:
            center_point: (lat, lon) tuple
            radius_m: Radius in meters
            network_type: Network type
            **kwargs: Additional parameters that affect the network
            
        Returns:
            Unique cache key
        """
        # Include all parameters that affect the network
        params = {
            'center': center_point,
            'radius': radius_m,
            'network_type': network_type,
            'running_filter': kwargs.get('running_filter', True),
            'elevation_source': kwargs.get('elevation_source', '3dep'),
            'consolidation_tolerance': kwargs.get('consolidation_tolerance', 10),
            'version': '2.0'  # Increment when cache format changes
        }
        
        # Create hash of parameters
        param_str = json.dumps(params, sort_keys=True)
        cache_key = hashlib.md5(param_str.encode()).hexdigest()
        return cache_key
    
    def get_cache_path(self, cache_key: str, compressed: bool = True) -> str:
        """Get file path for cache key"""
        ext = '.pkl.gz' if compressed else '.pkl'
        return os.path.join(self.cache_dir, f"network_{cache_key}{ext}")
    
    def cache_exists(self, cache_key: str) -> bool:
        """Check if cache exists and is valid"""
        cache_path = self.get_cache_path(cache_key)
        if not os.path.exists(cache_path):
            return False
        
        # Check if cache is in metadata and not expired
        if cache_key in self.metadata["caches"]:
            cache_info = self.metadata["caches"][cache_key]
            # Cache expires after 30 days (OSM data changes)
            if time.time() - cache_info["created"] > 30 * 24 * 3600:
                logger.info(f"Cache {cache_key} expired, will regenerate")
                return False
            return True
        
        return False
    
    def save_network(self, graph: nx.Graph, cache_key: str, 
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save network to cache with compression and metadata
        
        Args:
            graph: NetworkX graph
            cache_key: Cache key
            metadata: Additional metadata to store
            
        Returns:
            True if successful
        """
        try:
            cache_path = self.get_cache_path(cache_key)
            
            # Save compressed
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update metadata
            file_size = os.path.getsize(cache_path)
            self.metadata["caches"][cache_key] = {
                "created": time.time(),
                "file_size": file_size,
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
                "metadata": metadata or {}
            }
            self.save_metadata()
            
            logger.info(f"Cached network {cache_key}: {len(graph.nodes)} nodes, {file_size/1024/1024:.1f}MB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache network {cache_key}: {e}")
            return False
    
    def load_network(self, cache_key: str) -> Optional[nx.Graph]:
        """
        Load network from cache
        
        Args:
            cache_key: Cache key
            
        Returns:
            NetworkX graph or None if not found
        """
        if not self.cache_exists(cache_key):
            return None
        
        try:
            cache_path = self.get_cache_path(cache_key)
            
            with gzip.open(cache_path, 'rb') as f:
                graph = pickle.load(f)
            
            # Update access time
            self.metadata["caches"][cache_key]["last_accessed"] = time.time()
            self.save_metadata()
            
            logger.info(f"Loaded cached network {cache_key}: {len(graph.nodes)} nodes")
            return graph
            
        except Exception as e:
            logger.error(f"Failed to load cached network {cache_key}: {e}")
            # Remove corrupted cache
            self.remove_cache(cache_key)
            return None
    
    def remove_cache(self, cache_key: str):
        """Remove cache entry"""
        cache_path = self.get_cache_path(cache_key)
        if os.path.exists(cache_path):
            os.remove(cache_path)
        
        if cache_key in self.metadata["caches"]:
            del self.metadata["caches"][cache_key]
            self.save_metadata()
    
    def cleanup_old_caches(self, max_age_days: int = 30, max_total_size_gb: float = 10.0):
        """
        Clean up old and large caches
        
        Args:
            max_age_days: Maximum age in days
            max_total_size_gb: Maximum total cache size in GB
        """
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        max_size_bytes = max_total_size_gb * 1024 * 1024 * 1024
        
        # Find old caches
        old_caches = []
        total_size = 0
        
        for cache_key, info in self.metadata["caches"].items():
            age = current_time - info["created"]
            size = info["file_size"]
            total_size += size
            
            if age > max_age_seconds:
                old_caches.append((cache_key, age))
        
        # Remove old caches
        for cache_key, age in old_caches:
            logger.info(f"Removing old cache {cache_key} (age: {age/86400:.1f} days)")
            self.remove_cache(cache_key)
            total_size -= self.metadata["caches"].get(cache_key, {}).get("file_size", 0)
        
        # If still too large, remove least recently used
        if total_size > max_size_bytes:
            cache_by_access = [(k, v.get("last_accessed", v["created"])) 
                              for k, v in self.metadata["caches"].items()]
            cache_by_access.sort(key=lambda x: x[1])  # Oldest first
            
            for cache_key, _ in cache_by_access:
                if total_size <= max_size_bytes:
                    break
                
                size = self.metadata["caches"][cache_key]["file_size"]
                logger.info(f"Removing cache {cache_key} to free space ({size/1024/1024:.1f}MB)")
                self.remove_cache(cache_key)
                total_size -= size
        
        self.metadata["last_cleanup"] = current_time
        self.save_metadata()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_size = sum(info["file_size"] for info in self.metadata["caches"].values())
        total_nodes = sum(info["nodes"] for info in self.metadata["caches"].values())
        total_edges = sum(info["edges"] for info in self.metadata["caches"].values())
        
        return {
            "cache_count": len(self.metadata["caches"]),
            "total_size_mb": total_size / 1024 / 1024,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "oldest_cache": min((info["created"] for info in self.metadata["caches"].values()), 
                               default=time.time()),
            "newest_cache": max((info["created"] for info in self.metadata["caches"].values()), 
                               default=time.time())
        }
    
    def list_caches(self) -> List[Dict[str, Any]]:
        """List all cached networks with details"""
        caches = []
        current_time = time.time()
        
        for cache_key, info in self.metadata["caches"].items():
            age_days = (current_time - info["created"]) / 86400
            caches.append({
                "cache_key": cache_key,
                "nodes": info["nodes"],
                "edges": info["edges"],
                "size_mb": info["file_size"] / 1024 / 1024,
                "age_days": age_days,
                "metadata": info.get("metadata", {})
            })
        
        return sorted(caches, key=lambda x: x["age_days"])


# Integration with existing cache system
class ImprovedCacheManager:
    """Improved cache manager that integrates with existing system"""
    
    def __init__(self):
        self.advanced_cache = AdvancedNetworkCache()
        self.legacy_cache_dir = "cache"
    
    def migrate_legacy_caches(self):
        """Migrate old cache files to new system"""
        if not os.path.exists(self.legacy_cache_dir):
            return
        
        legacy_files = [f for f in os.listdir(self.legacy_cache_dir) 
                       if f.startswith('cached_graph_') and f.endswith('.pkl')]
        
        logger.info(f"Found {len(legacy_files)} legacy cache files to migrate")
        
        for filename in legacy_files:
            try:
                # Parse filename to extract parameters
                # Format: cached_graph_LAT_LON_RADIUSm_TYPE.pkl
                parts = filename.replace('cached_graph_', '').replace('.pkl', '').split('_')
                if len(parts) >= 4:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    radius = int(parts[2].replace('m', ''))
                    network_type = parts[3]
                    
                    # Load legacy cache
                    legacy_path = os.path.join(self.legacy_cache_dir, filename)
                    with open(legacy_path, 'rb') as f:
                        graph = pickle.load(f)
                    
                    # Generate new cache key
                    cache_key = self.advanced_cache.generate_cache_key(
                        (lat, lon), radius, network_type
                    )
                    
                    # Save to new cache system
                    metadata = {
                        "migrated_from": filename,
                        "migration_date": time.time()
                    }
                    
                    if self.advanced_cache.save_network(graph, cache_key, metadata):
                        logger.info(f"Migrated {filename} -> {cache_key}")
                        # Optionally remove legacy file
                        # os.remove(legacy_path)
                    
            except Exception as e:
                logger.error(f"Failed to migrate {filename}: {e}")
    
    def get_network(self, center_point: Tuple[float, float], 
                   radius_m: int, 
                   network_type: str = 'all',
                   **kwargs) -> Optional[nx.Graph]:
        """
        Get network using improved caching
        
        Args:
            center_point: (lat, lon) tuple
            radius_m: Radius in meters
            network_type: Network type
            **kwargs: Additional parameters
            
        Returns:
            NetworkX graph or None
        """
        # Generate cache key
        cache_key = self.advanced_cache.generate_cache_key(
            center_point, radius_m, network_type, **kwargs
        )
        
        # Try to load from cache
        graph = self.advanced_cache.load_network(cache_key)
        if graph is not None:
            return graph
        
        # Cache miss - would generate new network here
        logger.info(f"Cache miss for {cache_key}, would generate new network")
        return None
    
    def cleanup_caches(self):
        """Clean up old caches"""
        self.advanced_cache.cleanup_old_caches()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return self.advanced_cache.get_cache_stats()


if __name__ == "__main__":
    # Example usage
    cache_manager = ImprovedCacheManager()
    
    # Migrate legacy caches
    cache_manager.migrate_legacy_caches()
    
    # Get cache statistics
    stats = cache_manager.get_stats()
    print(f"Cache statistics: {stats}")
    
    # Cleanup old caches
    cache_manager.cleanup_caches()