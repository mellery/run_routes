#!/usr/bin/env python3
"""
Elevation Data Sources - Abstraction Layer for Multiple Elevation Data Sources
Supports SRTM, local 3DEP files, and hybrid configurations
"""

import os
import json
import math
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

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
    logger.warning("NumPy not available - some features will be limited")

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.transform import from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    logger.warning("Rasterio not available - 3DEP support will be disabled")

try:
    from shapely.geometry import Point, box
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    logger.warning("Shapely not available - some geometry operations will be limited")


class ElevationDataSource(ABC):
    """Abstract base class for elevation data sources"""
    
    @abstractmethod
    def get_elevation(self, lat: float, lon: float) -> Optional[float]:
        """Get elevation at a specific coordinate
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            
        Returns:
            Elevation in meters, or None if not available
        """
        pass
    
    @abstractmethod
    def get_elevation_profile(self, coordinates: List[Tuple[float, float]]) -> List[float]:
        """Get elevation profile for a list of coordinates
        
        Args:
            coordinates: List of (lat, lon) tuples
            
        Returns:
            List of elevations in meters
        """
        pass
    
    @abstractmethod
    def get_resolution(self) -> float:
        """Get data resolution in meters
        
        Returns:
            Resolution in meters
        """
        pass
    
    @abstractmethod
    def get_coverage_bounds(self) -> Tuple[float, float, float, float]:
        """Get data coverage bounds
        
        Returns:
            Tuple of (west, south, east, north) in degrees
        """
        pass
    
    @abstractmethod
    def is_available(self, lat: float, lon: float) -> bool:
        """Check if data is available at coordinate
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            
        Returns:
            True if data is available
        """
        pass
    
    def get_source_info(self) -> Dict:
        """Get information about this data source
        
        Returns:
            Dictionary with source information
        """
        return {
            'type': self.__class__.__name__,
            'resolution': self.get_resolution(),
            'coverage_bounds': self.get_coverage_bounds()
        }


class SRTMElevationSource(ElevationDataSource):
    """SRTM 90m elevation data source (existing implementation)"""
    
    def __init__(self, srtm_file_path: str):
        self.srtm_file_path = srtm_file_path
        self.resolution = 90.0  # meters
        self._dataset = None
        self._bounds = None
        
        self._initialize_dataset()
    
    def _initialize_dataset(self):
        """Initialize SRTM dataset"""
        if not os.path.exists(self.srtm_file_path):
            logger.error(f"SRTM file not found: {self.srtm_file_path}")
            return
        
        if not RASTERIO_AVAILABLE:
            logger.error("Rasterio not available - SRTM support disabled")
            return
        
        try:
            self._dataset = rasterio.open(self.srtm_file_path)
            self._bounds = self._dataset.bounds
            logger.debug(f"SRTM dataset initialized: {self.srtm_file_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SRTM dataset: {e}")
    
    def get_elevation(self, lat: float, lon: float) -> Optional[float]:
        """Get elevation from SRTM data"""
        if not self._dataset:
            return None
        
        try:
            # Sample elevation at coordinate
            coords = [(lon, lat)]
            elevations = list(self._dataset.sample(coords))
            
            if elevations and len(elevations[0]) > 0:
                elevation = float(elevations[0][0])
                # Filter out nodata values
                if elevation != self._dataset.nodata and not (NUMPY_AVAILABLE and np.isnan(elevation)):
                    return elevation
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get SRTM elevation at ({lat}, {lon}): {e}")
            return None
    
    def get_elevation_profile(self, coordinates: List[Tuple[float, float]]) -> List[float]:
        """Get elevation profile from SRTM data"""
        elevations = []
        
        for lat, lon in coordinates:
            elevation = self.get_elevation(lat, lon)
            elevations.append(elevation if elevation is not None else 0.0)
        
        return elevations
    
    def get_resolution(self) -> float:
        """Get SRTM resolution in meters"""
        return self.resolution
    
    def get_coverage_bounds(self) -> Tuple[float, float, float, float]:
        """Get SRTM coverage bounds"""
        if self._bounds:
            return (self._bounds.left, self._bounds.bottom, 
                   self._bounds.right, self._bounds.top)
        return (0, 0, 0, 0)
    
    def is_available(self, lat: float, lon: float) -> bool:
        """Check if SRTM data is available at coordinate"""
        if not self._bounds:
            return False
        
        return (self._bounds.left <= lon <= self._bounds.right and 
                self._bounds.bottom <= lat <= self._bounds.top)
    
    def close(self):
        """Close SRTM dataset"""
        if self._dataset:
            self._dataset.close()
            self._dataset = None


class LocalThreeDEPSource(ElevationDataSource):
    """Local file-based 3DEP 1-meter elevation data source with enhanced caching"""
    
    def __init__(self, data_directory: str = "./elevation_data/3dep_1m", enable_enhanced_caching: bool = True):
        self.data_dir = Path(data_directory)
        self.tiles_dir = self.data_dir / "tiles"
        self.index_dir = self.data_dir / "index"
        self.cache_dir = self.data_dir / "cache"
        
        # Create directories if they don't exist
        for directory in [self.tiles_dir, self.index_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.resolution = 1.0  # meters
        self.tile_index = {}
        self.spatial_index = {}  # Grid-based spatial index for fast tile lookup
        self.grid_size = 0.01  # Grid cell size in degrees (~1km)
        self.open_files = {}  # Cache for opened rasterio files
        self.file_access_order = []  # LRU tracking for file cache
        self.max_open_files = 100  # Increased limit for better performance
        self.preloaded_tiles = set()  # Track which tiles are kept in memory
        self.transformer_cache = {}  # Cache for pyproj transformers
        
        # Enhanced caching support
        self.enhanced_caching = enable_enhanced_caching
        self.cache_manager = None
        
        if not RASTERIO_AVAILABLE:
            logger.error("Rasterio not available - 3DEP support disabled")
            return
        
        self._initialize_tile_index()
        self._initialize_enhanced_caching()
    
    def _initialize_enhanced_caching(self):
        """Initialize enhanced caching system"""
        if not self.enhanced_caching:
            return
        
        try:
            from elevation_cache_manager import get_cache_manager
            self.cache_manager = get_cache_manager()
            
            # Index all tiles in the spatial cache
            for tile_path in self.tile_index.keys():
                if os.path.exists(tile_path):
                    self.cache_manager.index_tile(tile_path)
            
            logger.debug(f"Enhanced caching initialized for {len(self.tile_index)} tiles")
            
        except ImportError:
            logger.warning("Enhanced caching not available - elevation_cache_manager not found")
            self.enhanced_caching = False
        except Exception as e:
            logger.warning(f"Failed to initialize enhanced caching: {e}")
            self.enhanced_caching = False
    
    def _initialize_tile_index(self):
        """Initialize tile index from available files"""
        index_file = self.index_dir / "tile_index.json"
        
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    self.tile_index = json.load(f)
                logger.debug(f"Loaded tile index with {len(self.tile_index)} tiles")
            except Exception as e:
                logger.warning(f"Failed to load tile index: {e}")
                self._rebuild_tile_index()
        else:
            self._rebuild_tile_index()
    
    def _rebuild_tile_index(self):
        """Rebuild tile index by scanning available files"""
        logger.info("Rebuilding 3DEP tile index...")
        
        self.tile_index = {}
        tile_files = list(self.tiles_dir.glob("*.tif"))
        
        if not tile_files:
            logger.warning(f"No .tif files found in {self.tiles_dir}")
            return
        
        for tile_file in tile_files:
            try:
                with rasterio.open(tile_file) as src:
                    bounds = src.bounds
                    self.tile_index[str(tile_file)] = {
                        'bounds': [bounds.left, bounds.bottom, bounds.right, bounds.top],
                        'crs': src.crs.to_string(),
                        'resolution': [src.res[0], src.res[1]],
                        'size': [src.width, src.height],
                        'nodata': src.nodata
                    }
            except Exception as e:
                logger.warning(f"Failed to index tile {tile_file}: {e}")
        
        # Save index
        index_file = self.index_dir / "tile_index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self.tile_index, f, indent=2)
            logger.info(f"Saved tile index with {len(self.tile_index)} tiles")
        except Exception as e:
            logger.error(f"Failed to save tile index: {e}")
        
        # Build spatial index for fast lookups
        self._build_spatial_index()
    
    def _build_spatial_index(self):
        """Build spatial grid index for fast tile lookups"""
        logger.info("Building spatial index for tile lookups...")
        self.spatial_index = {}
        
        for tile_path, tile_info in self.tile_index.items():
            bounds = tile_info['bounds']  # [west, south, east, north]
            
            # Calculate grid cells covered by this tile
            west, south, east, north = bounds
            
            # Find grid bounds
            min_col = int(west / self.grid_size)
            max_col = int(east / self.grid_size) + 1
            min_row = int(south / self.grid_size)
            max_row = int(north / self.grid_size) + 1
            
            # Add tile to all covering grid cells
            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    grid_key = (row, col)
                    if grid_key not in self.spatial_index:
                        self.spatial_index[grid_key] = []
                    self.spatial_index[grid_key].append(tile_path)
        
        logger.info(f"Spatial index built with {len(self.spatial_index)} grid cells")
    
    def _get_grid_key(self, lat: float, lon: float) -> Tuple[int, int]:
        """Get grid key for coordinate"""
        row = int(lat / self.grid_size)
        col = int(lon / self.grid_size)
        return (row, col)
    
    def _find_covering_tiles_fast(self, lat: float, lon: float) -> List[str]:
        """Fast tile lookup using spatial index"""
        grid_key = self._get_grid_key(lat, lon)
        candidate_tiles = self.spatial_index.get(grid_key, [])
        
        covering_tiles = []
        for tile_path in candidate_tiles:
            tile_info = self.tile_index.get(tile_path)
            if not tile_info:
                continue
            
            try:
                # Get tile CRS and transform lat/lon to tile coordinates
                tile_crs = tile_info.get('crs', 'EPSG:4326')
                
                if tile_crs != 'EPSG:4326':
                    # Transform lat/lon to tile CRS
                    transformer = self._get_transformer('EPSG:4326', tile_crs)
                    if transformer:
                        x, y = transformer.transform(lon, lat)
                    else:
                        continue
                else:
                    x, y = lon, lat
                
                bounds = tile_info['bounds']  # [west, south, east, north] in tile CRS
                
                if (bounds[0] <= x <= bounds[2] and 
                    bounds[1] <= y <= bounds[3]):
                    covering_tiles.append(tile_path)
                    
            except Exception as e:
                logger.warning(f"Failed to check coverage for tile {tile_path}: {e}")
                continue
        
        return covering_tiles
    
    def _find_covering_tiles(self, lat: float, lon: float) -> List[str]:
        """Find tiles that cover the given coordinate"""
        covering_tiles = []
        
        for tile_path, tile_info in self.tile_index.items():
            try:
                # Get tile CRS and transform lat/lon to tile coordinates
                tile_crs = tile_info.get('crs', 'EPSG:4326')
                
                if tile_crs != 'EPSG:4326':
                    # Transform lat/lon to tile CRS
                    transformer = self._get_transformer('EPSG:4326', tile_crs)
                    if transformer:
                        x, y = transformer.transform(lon, lat)
                    else:
                        continue
                else:
                    x, y = lon, lat
                
                bounds = tile_info['bounds']  # [west, south, east, north] in tile CRS
                
                if (bounds[0] <= x <= bounds[2] and 
                    bounds[1] <= y <= bounds[3]):
                    covering_tiles.append(tile_path)
                    
            except Exception as e:
                logger.warning(f"Failed to check coverage for tile {tile_path}: {e}")
                continue
        
        return covering_tiles
    
    def _get_tile_dataset(self, tile_path: str):
        """Get rasterio dataset for tile, with LRU caching"""
        # Check if already open
        if tile_path in self.open_files:
            # Move to end of LRU list
            if tile_path in self.file_access_order:
                self.file_access_order.remove(tile_path)
            self.file_access_order.append(tile_path)
            return self.open_files[tile_path]
        
        # Close oldest files if we're at the limit
        while len(self.open_files) >= self.max_open_files:
            if not self.file_access_order:
                break
            oldest_file = self.file_access_order.pop(0)
            if oldest_file in self.open_files:
                # Don't close preloaded tiles unless absolutely necessary
                if oldest_file not in self.preloaded_tiles or len(self.open_files) >= self.max_open_files + 20:
                    try:
                        self.open_files[oldest_file].close()
                    except Exception:
                        pass
                    del self.open_files[oldest_file]
        
        try:
            dataset = rasterio.open(tile_path)
            self.open_files[tile_path] = dataset
            self.file_access_order.append(tile_path)
            return dataset
        except Exception as e:
            logger.error(f"Failed to open tile {tile_path}: {e}")
            return None
    
    def preload_tiles_for_area(self, center_lat: float, center_lon: float, radius_km: float = 2.0):
        """Preload tiles for a specific area to improve batch processing performance"""
        logger.info(f"Preloading tiles for area ({center_lat:.4f}, {center_lon:.4f}) with {radius_km}km radius...")
        
        # Calculate bounding box
        lat_delta = radius_km / 111.0  # Rough conversion km to degrees
        lon_delta = radius_km / (111.0 * abs(math.cos(math.radians(center_lat))))
        
        bounds = [
            center_lon - lon_delta,  # west
            center_lat - lat_delta,  # south
            center_lon + lon_delta,  # east
            center_lat + lat_delta   # north
        ]
        
        # Find all tiles that intersect with this area
        preload_count = 0
        for tile_path, tile_info in self.tile_index.items():
            tile_bounds = tile_info['bounds']
            
            # Check if tile intersects with area bounds
            if (tile_bounds[2] >= bounds[0] and tile_bounds[0] <= bounds[2] and
                tile_bounds[3] >= bounds[1] and tile_bounds[1] <= bounds[3]):
                
                try:
                    dataset = self._get_tile_dataset(tile_path)
                    if dataset:
                        self.preloaded_tiles.add(tile_path)
                        preload_count += 1
                except Exception as e:
                    logger.warning(f"Failed to preload tile {tile_path}: {e}")
        
        logger.info(f"Preloaded {preload_count} tiles for area")
    
    def _get_transformer(self, from_crs: str, to_crs: str):
        """Get cached coordinate transformer"""
        cache_key = (from_crs, to_crs)
        if cache_key not in self.transformer_cache:
            try:
                import pyproj
                self.transformer_cache[cache_key] = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True)
            except Exception as e:
                logger.error(f"Failed to create transformer {from_crs} -> {to_crs}: {e}")
                return None
        return self.transformer_cache[cache_key]
    
    def get_elevation(self, lat: float, lon: float) -> Optional[float]:
        """Get elevation at a specific coordinate with enhanced caching"""
        if not RASTERIO_AVAILABLE:
            return None
        
        # Use enhanced caching if available
        if self.enhanced_caching and self.cache_manager:
            try:
                return self.cache_manager.get_elevation_cached(lat, lon, self)
            except RecursionError:
                # Disable enhanced caching permanently to prevent recursion
                logger.warning("Enhanced caching recursion detected, disabling enhanced caching")
                self.enhanced_caching = False
                return self._get_elevation_direct(lat, lon)
            except Exception as e:
                logger.warning(f"Enhanced caching failed, falling back to direct access: {e}")
        
        # Original implementation as fallback
        return self._get_elevation_direct(lat, lon)
    
    def _get_elevation_direct(self, lat: float, lon: float) -> Optional[float]:
        """Direct elevation lookup without caching (used as fallback)"""
        # Use fast spatial index if available, fallback to linear search
        if self.spatial_index:
            covering_tiles = self._find_covering_tiles_fast(lat, lon)
        else:
            covering_tiles = self._find_covering_tiles(lat, lon)
        
        if not covering_tiles:
            return None
        
        # Use first covering tile (they should be consistent)
        tile_path = covering_tiles[0]
        
        try:
            src = self._get_tile_dataset(tile_path)
            if not src:
                return None
            
            # Transform coordinates to tile CRS if needed
            tile_crs = src.crs.to_string()
            if tile_crs != 'EPSG:4326':
                transformer = self._get_transformer('EPSG:4326', tile_crs)
                if not transformer:
                    return None
                x, y = transformer.transform(lon, lat)
                coords = [(x, y)]
            else:
                coords = [(lon, lat)]
            
            elevations = list(src.sample(coords))
            
            if elevations and len(elevations[0]) > 0:
                elevation = float(elevations[0][0])
                # Filter out nodata values
                if elevation != src.nodata and not (NUMPY_AVAILABLE and np.isnan(elevation)):
                    return elevation
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to read elevation from {tile_path}: {e}")
            return None
    
    def preload_area(self, center_lat: float, center_lon: float, radius_km: float = 5.0):
        """Preload tiles for an area to improve performance"""
        if self.enhanced_caching and self.cache_manager:
            self.cache_manager.preload_area(center_lat, center_lon, radius_km)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching performance statistics"""
        if self.enhanced_caching and self.cache_manager:
            return self.cache_manager.get_performance_stats()
        return {"enhanced_caching": False}
    
    def get_elevation_profile(self, coordinates: List[Tuple[float, float]]) -> List[float]:
        """Get elevation profile for a list of coordinates with batch optimization"""
        if not coordinates:
            return []
        
        # Try batch processing if available
        try:
            return self._get_elevation_profile_batch(coordinates)
        except Exception as e:
            logger.warning(f"Batch processing failed, falling back to individual queries: {e}")
            # Fallback to individual queries
            elevations = []
            for lat, lon in coordinates:
                elevation = self.get_elevation(lat, lon)
                elevations.append(elevation if elevation is not None else 0.0)
            return elevations
    
    def _get_elevation_profile_batch(self, coordinates: List[Tuple[float, float]]) -> List[float]:
        """Optimized batch elevation lookup"""
        if not RASTERIO_AVAILABLE or not coordinates:
            return []
        
        # Group coordinates by covering tiles to minimize file I/O
        tile_groups = {}
        coord_to_tiles = {}
        
        # First pass: find covering tiles for each coordinate
        for i, (lat, lon) in enumerate(coordinates):
            if self.spatial_index:
                covering_tiles = self._find_covering_tiles_fast(lat, lon)
            else:
                covering_tiles = self._find_covering_tiles(lat, lon)
            
            coord_to_tiles[i] = covering_tiles
            
            # Group coordinates by tiles
            for tile_path in covering_tiles:
                if tile_path not in tile_groups:
                    tile_groups[tile_path] = []
                tile_groups[tile_path].append((i, lat, lon))
        
        # Initialize results array
        elevations = [0.0] * len(coordinates)
        
        # Second pass: process coordinates by tile to maximize cache efficiency
        for tile_path, coord_list in tile_groups.items():
            try:
                src = self._get_tile_dataset(tile_path)
                if not src:
                    continue
                
                # Prepare coordinate transformations
                tile_crs = src.crs.to_string()
                if tile_crs != 'EPSG:4326':
                    transformer = self._get_transformer('EPSG:4326', tile_crs)
                    if not transformer:
                        continue
                    
                    # Batch transform coordinates
                    lons = [coord[2] for coord in coord_list]
                    lats = [coord[1] for coord in coord_list]
                    xs, ys = transformer.transform(lons, lats)
                    transformed_coords = [(xs[j], ys[j]) for j in range(len(coord_list))]
                else:
                    transformed_coords = [(coord[2], coord[1]) for coord in coord_list]  # (lon, lat)
                
                # Batch sample elevations
                try:
                    # Sample all coordinates at once
                    sampled_elevations = list(src.sample(transformed_coords))
                    
                    for j, (coord_idx, lat, lon) in enumerate(coord_list):
                        if j < len(sampled_elevations):
                            elevation_array = sampled_elevations[j]
                            if len(elevation_array) > 0:
                                elevation = float(elevation_array[0])
                                # Filter out nodata values
                                if elevation != src.nodata and not (NUMPY_AVAILABLE and np.isnan(elevation)):
                                    elevations[coord_idx] = elevation
                
                except Exception as e:
                    logger.warning(f"Batch sampling failed for {tile_path}, falling back to individual: {e}")
                    # Fallback to individual sampling for this tile
                    for coord_idx, lat, lon in coord_list:
                        try:
                            if tile_crs != 'EPSG:4326':
                                if transformer:
                                    x, y = transformer.transform(lon, lat)
                                    coords = [(x, y)]
                                else:
                                    continue
                            else:
                                coords = [(lon, lat)]
                            
                            elevation_array = list(src.sample(coords))[0]
                            if len(elevation_array) > 0:
                                elevation = float(elevation_array[0])
                                if elevation != src.nodata and not (NUMPY_AVAILABLE and np.isnan(elevation)):
                                    elevations[coord_idx] = elevation
                        except Exception:
                            continue
                            
            except Exception as e:
                logger.warning(f"Failed to process tile {tile_path}: {e}")
                continue
        
        return elevations
    
    def get_resolution(self) -> float:
        """Get data resolution in meters"""
        return self.resolution
    
    def get_coverage_bounds(self) -> Tuple[float, float, float, float]:
        """Get data coverage bounds (west, south, east, north)"""
        if not self.tile_index:
            return (0, 0, 0, 0)
        
        all_bounds = [info['bounds'] for info in self.tile_index.values()]
        
        if not all_bounds:
            return (0, 0, 0, 0)
        
        west = min(bounds[0] for bounds in all_bounds)
        south = min(bounds[1] for bounds in all_bounds)
        east = max(bounds[2] for bounds in all_bounds)
        north = max(bounds[3] for bounds in all_bounds)
        
        return (west, south, east, north)
    
    def is_available(self, lat: float, lon: float) -> bool:
        """Check if data is available at coordinate"""
        return len(self._find_covering_tiles(lat, lon)) > 0
    
    def get_tile_info(self) -> Dict:
        """Get information about available tiles"""
        return {
            'tile_count': len(self.tile_index),
            'total_coverage_area': self.get_coverage_bounds(),
            'tiles': list(self.tile_index.keys())
        }
    
    def refresh_index(self):
        """Refresh tile index (useful after downloading new tiles)"""
        self._rebuild_tile_index()
    
    def close(self):
        """Close all open rasterio files and cache resources"""
        for src in self.open_files.values():
            try:
                src.close()
            except:
                pass
        self.open_files.clear()
        
        # Close enhanced caching resources
        if self.enhanced_caching and self.cache_manager:
            try:
                self.cache_manager.close()
            except:
                pass


class HybridElevationSource(ElevationDataSource):
    """Hybrid source that prefers primary but falls back to secondary"""
    
    def __init__(self, primary_source: ElevationDataSource, 
                 fallback_source: ElevationDataSource):
        self.primary = primary_source
        self.fallback = fallback_source
        self.resolution = min(primary_source.get_resolution(), 
                             fallback_source.get_resolution())
        
        # Statistics tracking
        self.stats = {
            'primary_queries': 0,
            'fallback_queries': 0,
            'failed_queries': 0
        }
    
    def get_elevation(self, lat: float, lon: float) -> Optional[float]:
        """Get elevation with fallback logic"""
        # Try primary source first
        if self.primary.is_available(lat, lon):
            elevation = self.primary.get_elevation(lat, lon)
            if elevation is not None:
                self.stats['primary_queries'] += 1
                return elevation
        
        # Fall back to secondary source
        if self.fallback.is_available(lat, lon):
            elevation = self.fallback.get_elevation(lat, lon)
            if elevation is not None:
                self.stats['fallback_queries'] += 1
                return elevation
        
        self.stats['failed_queries'] += 1
        return None
    
    def get_elevation_profile(self, coordinates: List[Tuple[float, float]]) -> List[float]:
        """Get elevation profile with mixed sources"""
        elevations = []
        
        for lat, lon in coordinates:
            elevation = self.get_elevation(lat, lon)
            elevations.append(elevation if elevation is not None else 0.0)
        
        return elevations
    
    def get_resolution(self) -> float:
        """Get effective resolution (highest of the two sources)"""
        return self.resolution
    
    def get_coverage_bounds(self) -> Tuple[float, float, float, float]:
        """Get combined coverage bounds"""
        primary_bounds = self.primary.get_coverage_bounds()
        fallback_bounds = self.fallback.get_coverage_bounds()
        
        # Combine bounds (union)
        west = min(primary_bounds[0], fallback_bounds[0])
        south = min(primary_bounds[1], fallback_bounds[1])
        east = max(primary_bounds[2], fallback_bounds[2])
        north = max(primary_bounds[3], fallback_bounds[3])
        
        return (west, south, east, north)
    
    def is_available(self, lat: float, lon: float) -> bool:
        """Check if data is available from either source"""
        return (self.primary.is_available(lat, lon) or 
                self.fallback.is_available(lat, lon))
    
    def get_source_info(self) -> Dict:
        """Get information about both sources"""
        return {
            'type': 'HybridElevationSource',
            'primary': self.primary.get_source_info(),
            'fallback': self.fallback.get_source_info(),
            'stats': self.stats.copy()
        }
    
    def get_stats(self) -> Dict:
        """Get usage statistics"""
        total_queries = sum(self.stats.values())
        
        if total_queries == 0:
            return self.stats.copy()
        
        return {
            **self.stats,
            'primary_percentage': (self.stats['primary_queries'] / total_queries) * 100,
            'fallback_percentage': (self.stats['fallback_queries'] / total_queries) * 100,
            'failure_percentage': (self.stats['failed_queries'] / total_queries) * 100
        }


class ElevationConfig:
    """Configuration for elevation data sources"""
    
    def __init__(self):
        self.preferred_source = "hybrid"  # "3dep_local", "srtm", "hybrid"
        self.fallback_enabled = True
        self.cache_enabled = True
        
        # Local 3DEP settings
        self.threedep_data_directory = "./elevation_data/3dep_1m"
        self.auto_rebuild_index = True
        
        # SRTM settings
        self.srtm_file_path = "elevation_data/srtm_90m/srtm_20_05.tif"  # Use available SRTM file
        
        # Performance settings
        self.max_open_files = 10
        self.enable_statistics = True
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ElevationConfig':
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            config = cls()
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return cls()
    
    def to_file(self, config_path: str):
        """Save configuration to JSON file"""
        config_data = {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"Saved config to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")


class ElevationDataManager:
    """Manager for elevation data sources"""
    
    def __init__(self, config: Optional[ElevationConfig] = None, verbose: bool = True):
        self.config = config or ElevationConfig()
        self.sources = {}
        self.active_source = None
        self.verbose = verbose
        
        self._initialize_sources()
    
    def _initialize_sources(self):
        """Initialize available elevation data sources"""
        
        # Initialize SRTM source (always try to make available)
        try:
            if os.path.exists(self.config.srtm_file_path):
                self.sources['srtm'] = SRTMElevationSource(self.config.srtm_file_path)
                logger.debug("SRTM elevation source initialized")
            else:
                logger.warning(f"SRTM file not found: {self.config.srtm_file_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SRTM source: {e}")
        
        # Initialize Local 3DEP source
        if RASTERIO_AVAILABLE:
            try:
                local_3dep = LocalThreeDEPSource(self.config.threedep_data_directory)
                tile_info = local_3dep.get_tile_info()
                
                if tile_info['tile_count'] > 0:
                    self.sources['3dep_local'] = local_3dep
                    logger.debug(f"Local 3DEP source initialized with {tile_info['tile_count']} tiles")
                else:
                    logger.warning("No 3DEP tiles found. Run setup_3dep_data.py to download tiles.")
                    
            except Exception as e:
                logger.error(f"Failed to initialize Local 3DEP source: {e}")
        else:
            logger.warning("Rasterio not available - 3DEP support disabled")
    
    def get_elevation_source(self) -> Optional[ElevationDataSource]:
        """Get the configured elevation source"""
        
        if self.active_source:
            return self.active_source
        
        if self.config.preferred_source == "3dep_local" and '3dep_local' in self.sources:
            if self.config.fallback_enabled and 'srtm' in self.sources:
                self.active_source = HybridElevationSource(
                    self.sources['3dep_local'], 
                    self.sources['srtm']
                )
            else:
                self.active_source = self.sources['3dep_local']
                
        elif self.config.preferred_source == "hybrid" and all(s in self.sources for s in ['3dep_local', 'srtm']):
            self.active_source = HybridElevationSource(
                self.sources['3dep_local'], 
                self.sources['srtm']
            )
            
        elif self.config.preferred_source == "srtm" and 'srtm' in self.sources:
            self.active_source = self.sources['srtm']
            
        else:
            # Default fallback
            if 'srtm' in self.sources:
                self.active_source = self.sources['srtm']
                logger.info("Using SRTM as default elevation source")
            else:
                logger.error("No elevation sources available")
                return None
        
        return self.active_source
    
    def get_available_sources(self) -> List[str]:
        """Get list of available elevation sources"""
        return list(self.sources.keys())
    
    def get_source_info(self) -> Dict:
        """Get information about all sources"""
        info = {}
        for name, source in self.sources.items():
            info[name] = source.get_source_info()
        
        active_source = self.get_elevation_source()
        if active_source:
            info['active'] = active_source.get_source_info()
        
        return info
    
    def test_sources(self, test_lat: float = 37.1299, test_lon: float = -80.4094) -> Dict:
        """Test all available sources at a coordinate"""
        results = {}
        
        for name, source in self.sources.items():
            try:
                available = source.is_available(test_lat, test_lon)
                elevation = source.get_elevation(test_lat, test_lon) if available else None
                
                results[name] = {
                    'available': available,
                    'elevation': elevation,
                    'resolution': source.get_resolution()
                }
            except Exception as e:
                results[name] = {
                    'available': False,
                    'elevation': None,
                    'error': str(e)
                }
        
        return results
    
    def close_all(self):
        """Close all elevation data sources"""
        for source in self.sources.values():
            if hasattr(source, 'close'):
                source.close()
        
        if self.active_source and hasattr(self.active_source, 'close'):
            self.active_source.close()


# Global elevation manager instance for singleton pattern
_global_elevation_manager = None
_global_config_path = None
_global_manager_initialized = False

# Convenience function for easy access
def get_elevation_manager(config_path: Optional[str] = None) -> ElevationDataManager:
    """Get configured elevation data manager (singleton pattern)
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configured ElevationDataManager instance
    """
    global _global_elevation_manager, _global_config_path, _global_manager_initialized
    
    # Return existing instance if config hasn't changed
    if _global_elevation_manager is not None and _global_config_path == config_path:
        return _global_elevation_manager
    
    # Create new instance (only show initialization messages on first creation)
    if config_path and os.path.exists(config_path):
        config = ElevationConfig.from_file(config_path)
    else:
        config = ElevationConfig()
    
    verbose = not _global_manager_initialized
    _global_elevation_manager = ElevationDataManager(config, verbose=verbose)
    _global_config_path = config_path
    _global_manager_initialized = True
    
    return _global_elevation_manager


if __name__ == "__main__":
    # Test the elevation data sources
    print("🧪 Testing Elevation Data Sources")
    print("=" * 50)
    
    # Create manager
    manager = ElevationDataManager()
    
    # Show available sources
    available = manager.get_available_sources()
    print(f"Available sources: {available}")
    
    # Get active source
    source = manager.get_elevation_source()
    if source:
        print(f"Active source: {source.__class__.__name__}")
        print(f"Resolution: {source.get_resolution()}m")
        
        # Test elevation lookup
        test_lat, test_lon = 37.1299, -80.4094  # Christiansburg, VA
        elevation = source.get_elevation(test_lat, test_lon)
        print(f"Test elevation at ({test_lat}, {test_lon}): {elevation}m")
        
        # Test all sources
        test_results = manager.test_sources(test_lat, test_lon)
        print("\nSource test results:")
        for name, result in test_results.items():
            print(f"  {name}: {result}")
    else:
        print("❌ No elevation source available")
    
    # Clean up
    manager.close_all()