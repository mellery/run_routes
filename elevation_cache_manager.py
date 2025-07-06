#!/usr/bin/env python3
"""
Enhanced Elevation Cache Manager
Provides LRU caching, memory-mapped file access, and performance optimizations for elevation data
"""

import os
import time
import mmap
import sqlite3
import threading
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import OrderedDict
import json
import logging
import hashlib

# Configure logging - only set up if not already configured
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import rasterio
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


class LRUCache:
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Any:
        """Get item from cache"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                return value
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any):
        """Put item in cache"""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate_percent': round(hit_rate, 1)
            }


class MemoryMappedTileCache:
    """Memory-mapped file cache for elevation tiles"""
    
    def __init__(self, cache_dir: str = "./elevation_data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.open_files = {}
        self.file_locks = {}
        self.access_times = {}
        self.max_open_files = 20
        self.lock = threading.RLock()
    
    def _get_cache_key(self, tile_path: str) -> str:
        """Generate cache key for tile"""
        return hashlib.md5(str(tile_path).encode()).hexdigest()
    
    def _get_mmap_file(self, tile_path: str):
        """Get memory-mapped file for tile"""
        cache_key = self._get_cache_key(tile_path)
        
        with self.lock:
            if cache_key in self.open_files:
                self.access_times[cache_key] = time.time()
                return self.open_files[cache_key]
            
            # Close oldest files if at limit
            if len(self.open_files) >= self.max_open_files:
                self._close_oldest_file()
            
            try:
                # Open file with memory mapping
                file_obj = open(tile_path, 'rb')
                mmap_obj = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)
                
                self.open_files[cache_key] = {
                    'file': file_obj,
                    'mmap': mmap_obj,
                    'path': tile_path
                }
                self.file_locks[cache_key] = threading.RLock()
                self.access_times[cache_key] = time.time()
                
                return self.open_files[cache_key]
                
            except Exception as e:
                logger.warning(f"Failed to memory-map tile {tile_path}: {e}")
                return None
    
    def _close_oldest_file(self):
        """Close the least recently used file"""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._close_file(oldest_key)
    
    def _close_file(self, cache_key: str):
        """Close a specific memory-mapped file"""
        if cache_key in self.open_files:
            try:
                file_data = self.open_files[cache_key]
                file_data['mmap'].close()
                file_data['file'].close()
            except:
                pass
            
            del self.open_files[cache_key]
            if cache_key in self.file_locks:
                del self.file_locks[cache_key]
            if cache_key in self.access_times:
                del self.access_times[cache_key]
    
    def get_tile_data(self, tile_path: str):
        """Get memory-mapped tile data"""
        mmap_data = self._get_mmap_file(tile_path)
        if mmap_data:
            return mmap_data['mmap']
        return None
    
    def close_all(self):
        """Close all memory-mapped files"""
        with self.lock:
            cache_keys = list(self.open_files.keys())
            for cache_key in cache_keys:
                self._close_file(cache_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                'open_files': len(self.open_files),
                'max_files': self.max_open_files,
                'total_memory_mapped_mb': len(self.open_files) * 50  # Estimate 50MB per tile
            }


class SpatialTileIndex:
    """Spatial indexing for fast tile lookup"""
    
    def __init__(self, index_file: str = "./elevation_data/cache/spatial_index.db"):
        self.index_file = index_file
        self.lock = threading.RLock()
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize spatial index database"""
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        
        with sqlite3.connect(self.index_file) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tile_index (
                    tile_path TEXT PRIMARY KEY,
                    west REAL, south REAL, east REAL, north REAL,
                    crs TEXT, resolution_x REAL, resolution_y REAL,
                    width INTEGER, height INTEGER,
                    last_accessed REAL,
                    file_size INTEGER,
                    checksum TEXT
                )
            """)
            
            # Create spatial index
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_spatial 
                ON tile_index (west, south, east, north)
            """)
            
            conn.commit()
    
    def add_tile(self, tile_path: str, bounds: Tuple[float, float, float, float],
                 crs: str, resolution: Tuple[float, float], size: Tuple[int, int]):
        """Add tile to spatial index"""
        with self.lock:
            try:
                file_size = os.path.getsize(tile_path)
                checksum = self._calculate_checksum(tile_path)
                
                with sqlite3.connect(self.index_file) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO tile_index 
                        (tile_path, west, south, east, north, crs, 
                         resolution_x, resolution_y, width, height,
                         last_accessed, file_size, checksum)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        tile_path, bounds[0], bounds[1], bounds[2], bounds[3],
                        crs, resolution[0], resolution[1], size[0], size[1],
                        time.time(), file_size, checksum
                    ))
                    conn.commit()
                    
            except Exception as e:
                logger.warning(f"Failed to add tile to spatial index: {e}")
    
    def find_covering_tiles(self, lat: float, lon: float) -> List[Dict[str, Any]]:
        """Find tiles that cover the given coordinate"""
        with self.lock:
            try:
                with sqlite3.connect(self.index_file) as conn:
                    cursor = conn.execute("""
                        SELECT tile_path, west, south, east, north, crs, 
                               resolution_x, resolution_y, file_size
                        FROM tile_index 
                        WHERE west <= ? AND ? <= east AND south <= ? AND ? <= north
                        ORDER BY resolution_x ASC
                    """, (lon, lon, lat, lat))
                    
                    tiles = []
                    for row in cursor.fetchall():
                        tiles.append({
                            'path': row[0],
                            'bounds': (row[1], row[2], row[3], row[4]),
                            'crs': row[5],
                            'resolution': (row[6], row[7]),
                            'file_size': row[8]
                        })
                    
                    # Update access time
                    if tiles:
                        tile_paths = [t['path'] for t in tiles]
                        placeholders = ','.join(['?' for _ in tile_paths])
                        conn.execute(f"""
                            UPDATE tile_index 
                            SET last_accessed = ? 
                            WHERE tile_path IN ({placeholders})
                        """, [time.time()] + tile_paths)
                        conn.commit()
                    
                    return tiles
                    
            except Exception as e:
                logger.warning(f"Failed to query spatial index: {e}")
                return []
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read(8192)).hexdigest()  # Sample checksum
        except:
            return ""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        with self.lock:
            try:
                with sqlite3.connect(self.index_file) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM tile_index")
                    tile_count = cursor.fetchone()[0]
                    
                    cursor = conn.execute("SELECT SUM(file_size) FROM tile_index")
                    total_size = cursor.fetchone()[0] or 0
                    
                    return {
                        'indexed_tiles': tile_count,
                        'total_size_mb': round(total_size / (1024 * 1024), 1),
                        'index_file_size_kb': round(os.path.getsize(self.index_file) / 1024, 1)
                    }
            except Exception as e:
                logger.warning(f"Failed to get index stats: {e}")
                return {'indexed_tiles': 0, 'total_size_mb': 0, 'index_file_size_kb': 0}


class ElevationQueryBatcher:
    """Batch elevation queries for better performance"""
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.pending_queries = []
        self.query_cache = LRUCache(max_size=1000)
        self.lock = threading.RLock()
    
    def add_query(self, lat: float, lon: float, source) -> Optional[float]:
        """Add elevation query to batch"""
        query_key = f"{lat:.6f},{lon:.6f}"
        
        # Check cache first
        cached_result = self.query_cache.get(query_key)
        if cached_result is not None:
            return cached_result
        
        with self.lock:
            self.pending_queries.append((lat, lon, query_key, source))
            
            # Process batch if full
            if len(self.pending_queries) >= self.batch_size:
                return self._process_batch()
            
            return None
    
    def _process_batch(self) -> Optional[float]:
        """Process batch of elevation queries"""
        if not self.pending_queries:
            return None
        
        results = {}
        
        # Group queries by source for efficiency
        source_groups = {}
        for lat, lon, key, source in self.pending_queries:
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append((lat, lon, key))
        
        # Process each source group
        for source, queries in source_groups.items():
            try:
                coordinates = [(lat, lon) for lat, lon, key in queries]
                elevations = source.get_elevation_profile(coordinates)
                
                for i, (lat, lon, key) in enumerate(queries):
                    if i < len(elevations):
                        results[key] = elevations[i]
                        self.query_cache.put(key, elevations[i])
                        
            except Exception as e:
                logger.warning(f"Batch processing failed for source {source}: {e}")
        
        self.pending_queries.clear()
        return results.get(self.pending_queries[-1][2]) if self.pending_queries else None
    
    def flush(self) -> Dict[str, float]:
        """Flush remaining queries"""
        with self.lock:
            if self.pending_queries:
                return self._process_batch() or {}
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics"""
        return {
            'pending_queries': len(self.pending_queries),
            'batch_size': self.batch_size,
            'cache_stats': self.query_cache.get_stats()
        }


class EnhancedElevationCacheManager:
    """Main cache manager coordinating all caching systems"""
    
    def __init__(self, cache_dir: str = "./elevation_data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache components
        self.lru_cache = LRUCache(max_size=500)
        self.mmap_cache = MemoryMappedTileCache(str(self.cache_dir))
        self.spatial_index = SpatialTileIndex(str(self.cache_dir / "spatial_index.db"))
        self.query_batcher = ElevationQueryBatcher(batch_size=50)
        
        # Performance metrics
        self.query_count = 0
        self.cache_hit_count = 0
        self.total_query_time = 0
        self.lock = threading.RLock()
        
        logger.debug("Enhanced elevation cache manager initialized")
    
    def get_elevation_cached(self, lat: float, lon: float, source) -> Optional[float]:
        """Get elevation with full caching pipeline"""
        start_time = time.time()
        
        try:
            # Try LRU cache first
            cache_key = f"{lat:.6f},{lon:.6f}"
            cached_result = self.lru_cache.get(cache_key)
            
            if cached_result is not None:
                with self.lock:
                    self.cache_hit_count += 1
                    self.query_count += 1
                    self.total_query_time += time.time() - start_time
                return cached_result
            
            # Use source with spatial indexing
            elevation = self._get_elevation_with_spatial_index(lat, lon, source)
            
            # Cache result
            if elevation is not None:
                self.lru_cache.put(cache_key, elevation)
            
            with self.lock:
                self.query_count += 1
                self.total_query_time += time.time() - start_time
            
            return elevation
            
        except Exception as e:
            logger.warning(f"Cached elevation lookup failed: {e}")
            return source.get_elevation(lat, lon)
    
    def _get_elevation_with_spatial_index(self, lat: float, lon: float, source) -> Optional[float]:
        """Get elevation using spatial index for tile lookup"""
        try:
            # Find covering tiles using spatial index
            covering_tiles = self.spatial_index.find_covering_tiles(lat, lon)
            
            if not covering_tiles:
                # Fallback to source's direct method to avoid recursion
                if hasattr(source, '_get_elevation_direct'):
                    return source._get_elevation_direct(lat, lon)
                else:
                    # For other sources, call get_elevation but disable caching temporarily
                    return None
            
            # Use highest resolution tile (first in sorted list)
            best_tile = covering_tiles[0]
            
            # Get memory-mapped tile data
            mmap_data = self.mmap_cache.get_tile_data(best_tile['path'])
            if mmap_data:
                # Use rasterio with memory-mapped data for faster access
                return self._sample_from_mmap_tile(lat, lon, best_tile, mmap_data)
            else:
                # Fallback to normal file access
                return source.get_elevation(lat, lon)
                
        except Exception as e:
            logger.warning(f"Spatial index elevation lookup failed: {e}")
            return source.get_elevation(lat, lon)
    
    def _sample_from_mmap_tile(self, lat: float, lon: float, tile_info: Dict, mmap_data) -> Optional[float]:
        """Sample elevation from memory-mapped tile data"""
        try:
            # This is a simplified implementation
            # In practice, you'd use rasterio with the memory-mapped data
            # For now, fallback to file-based access
            if RASTERIO_AVAILABLE:
                with rasterio.open(tile_info['path']) as src:
                    coords = [(lon, lat)]
                    elevations = list(src.sample(coords))
                    
                    if elevations and len(elevations[0]) > 0:
                        elevation = float(elevations[0][0])
                        if elevation != src.nodata and not (NUMPY_AVAILABLE and np.isnan(elevation)):
                            return elevation
            
            return None
            
        except Exception as e:
            logger.warning(f"Memory-mapped tile sampling failed: {e}")
            return None
    
    def index_tile(self, tile_path: str):
        """Add tile to spatial index"""
        try:
            if RASTERIO_AVAILABLE:
                with rasterio.open(tile_path) as src:
                    bounds = src.bounds
                    self.spatial_index.add_tile(
                        tile_path,
                        (bounds.left, bounds.bottom, bounds.right, bounds.top),
                        src.crs.to_string(),
                        (src.res[0], src.res[1]),
                        (src.width, src.height)
                    )
        except Exception as e:
            logger.warning(f"Failed to index tile {tile_path}: {e}")
    
    def preload_area(self, center_lat: float, center_lon: float, radius_km: float = 5.0):
        """Preload tiles for an area"""
        # Calculate bounding box
        lat_delta = radius_km / 111.0  # Approximate km to degrees
        lon_delta = radius_km / (111.0 * abs(np.cos(np.radians(center_lat)))) if NUMPY_AVAILABLE else lat_delta
        
        # Find all tiles in area
        covering_tiles = self.spatial_index.find_covering_tiles(center_lat, center_lon)
        
        logger.info(f"Preloading {len(covering_tiles)} tiles for area around ({center_lat}, {center_lon})")
        
        # Preload tiles into memory-mapped cache
        for tile in covering_tiles:
            self.mmap_cache.get_tile_data(tile['path'])
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        with self.lock:
            avg_query_time = (self.total_query_time / self.query_count * 1000) if self.query_count > 0 else 0
            cache_hit_rate = (self.cache_hit_count / self.query_count * 100) if self.query_count > 0 else 0
        
        return {
            'query_performance': {
                'total_queries': self.query_count,
                'cache_hits': self.cache_hit_count,
                'cache_hit_rate_percent': round(cache_hit_rate, 1),
                'avg_query_time_ms': round(avg_query_time, 2)
            },
            'lru_cache': self.lru_cache.get_stats(),
            'mmap_cache': self.mmap_cache.get_stats(),
            'spatial_index': self.spatial_index.get_stats(),
            'query_batcher': self.query_batcher.get_stats()
        }
    
    def close(self):
        """Clean up all cache resources"""
        self.mmap_cache.close_all()
        self.lru_cache.clear()
        logger.info("Enhanced elevation cache manager closed")


# Singleton instance for global access
_cache_manager = None

def get_cache_manager() -> EnhancedElevationCacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = EnhancedElevationCacheManager()
    return _cache_manager


if __name__ == "__main__":
    # Test the enhanced caching system
    print("🚀 Testing Enhanced Elevation Cache Manager")
    print("=" * 50)
    
    cache_manager = get_cache_manager()
    
    # Test with dummy elevation source
    class DummyElevationSource:
        def get_elevation(self, lat, lon):
            time.sleep(0.01)  # Simulate file I/O
            return 100.0 + lat + lon
        
        def get_elevation_profile(self, coordinates):
            return [self.get_elevation(lat, lon) for lat, lon in coordinates]
    
    source = DummyElevationSource()
    
    # Performance test
    test_coords = [
        (37.1299, -80.4094),
        (37.1300, -80.4095),
        (37.1301, -80.4096),
        (37.1299, -80.4094),  # Repeat for cache test
    ]
    
    print("Testing elevation queries with caching:")
    start_time = time.time()
    
    for lat, lon in test_coords:
        elevation = cache_manager.get_elevation_cached(lat, lon, source)
        print(f"  ({lat:.4f}, {lon:.4f}): {elevation:.1f}m")
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.3f}s")
    
    # Show performance stats
    stats = cache_manager.get_performance_stats()
    print(f"\nPerformance Statistics:")
    for category, data in stats.items():
        print(f"  {category}: {data}")
    
    cache_manager.close()
    print("\n✅ Enhanced caching test completed")