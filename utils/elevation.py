"""
Unified Elevation Data Access

Provides consolidated elevation data access with automatic fallback between
3DEP 1m (high-resolution) and SRTM 90m (global coverage) data sources.

This module consolidates elevation_data_sources.py by removing over-engineering
and providing a clean, simple API for elevation queries.

Architecture:
    - ElevationDataSource: Abstract base class
    - SRTMElevationSource: SRTM 90m data source
    - LocalThreeDEPSource: 3DEP 1m data source (simplified)
    - HybridElevationSource: Automatic 3DEP → SRTM fallback
    - ElevationService: Main public API (singleton)
"""

import os
import json
import math
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
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
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    logger.warning("PyProj not available - coordinate transformations limited")


class ElevationDataSource(ABC):
    """Abstract base class for elevation data sources"""

    @abstractmethod
    def get_elevation(self, lat: float, lon: float) -> Optional[float]:
        """Get elevation at a specific coordinate (lat, lon) in meters"""
        pass

    @abstractmethod
    def get_elevation_profile(self, coordinates: List[Tuple[float, float]]) -> List[float]:
        """Get elevation profile for a list of (lat, lon) tuples in meters"""
        pass

    @abstractmethod
    def get_resolution(self) -> float:
        """Get data resolution in meters"""
        pass

    @abstractmethod
    def is_available(self, lat: float, lon: float) -> bool:
        """Check if data is available at coordinate"""
        pass


class SRTMElevationSource(ElevationDataSource):
    """SRTM 90m elevation data source"""

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
    """Local file-based 3DEP 1-meter elevation data source"""

    def __init__(self, data_directory: str = "./elevation_data/3dep_1m"):
        self.data_dir = Path(data_directory)
        self.tiles_dir = self.data_dir / "tiles"
        self.index_dir = self.data_dir / "index"

        # Create directories if they don't exist
        for directory in [self.tiles_dir, self.index_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.resolution = 1.0  # meters
        self.tile_index = {}
        self.spatial_index = {}  # Grid-based spatial index for fast tile lookup
        self.grid_size = 0.01  # Grid cell size in degrees (~1km)
        self.open_files = {}  # Cache for opened rasterio files
        self.file_access_order = []  # LRU tracking for file cache
        self.max_open_files = 50  # LRU cache limit
        self.transformer_cache = {}  # Cache for pyproj transformers

        if not RASTERIO_AVAILABLE:
            logger.error("Rasterio not available - 3DEP support disabled")
            return

        self._initialize_tile_index()

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

    def _find_covering_tiles(self, lat: float, lon: float) -> List[str]:
        """Fast tile lookup using spatial index"""
        grid_key = self._get_grid_key(lat, lon)
        candidate_tiles = self.spatial_index.get(grid_key, [])

        covering_tiles = []
        for tile_path in candidate_tiles:
            tile_info = self.tile_index.get(tile_path)
            if not tile_info:
                continue

            try:
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

                bounds = tile_info['bounds']  # [west, south, east, north]

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

    def _get_transformer(self, from_crs: str, to_crs: str):
        """Get cached coordinate transformer"""
        cache_key = (from_crs, to_crs)
        if cache_key not in self.transformer_cache:
            if not PYPROJ_AVAILABLE:
                return None
            try:
                self.transformer_cache[cache_key] = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True)
            except Exception as e:
                logger.error(f"Failed to create transformer {from_crs} -> {to_crs}: {e}")
                return None
        return self.transformer_cache[cache_key]

    def get_elevation(self, lat: float, lon: float) -> Optional[float]:
        """Get elevation at a specific coordinate"""
        if not RASTERIO_AVAILABLE:
            return None

        covering_tiles = self._find_covering_tiles(lat, lon)

        if not covering_tiles:
            return None

        # Use first covering tile
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

    def get_elevation_profile(self, coordinates: List[Tuple[float, float]]) -> List[float]:
        """Get elevation profile for a list of coordinates"""
        elevations = []
        for lat, lon in coordinates:
            elevation = self.get_elevation(lat, lon)
            elevations.append(elevation if elevation is not None else 0.0)
        return elevations

    def get_resolution(self) -> float:
        """Get data resolution in meters"""
        return self.resolution

    def is_available(self, lat: float, lon: float) -> bool:
        """Check if data is available at coordinate"""
        return len(self._find_covering_tiles(lat, lon)) > 0

    def refresh_index(self):
        """Refresh tile index (useful after downloading new tiles)"""
        self._rebuild_tile_index()

    def close(self):
        """Close all open rasterio files"""
        for src in self.open_files.values():
            try:
                src.close()
            except:
                pass
        self.open_files.clear()


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

    def is_available(self, lat: float, lon: float) -> bool:
        """Check if data is available from either source"""
        return (self.primary.is_available(lat, lon) or
                self.fallback.is_available(lat, lon))

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


class ElevationService:
    """
    Unified elevation service with automatic 3DEP → SRTM fallback.

    This is the main public API for elevation queries. It automatically
    initializes the best available data sources and provides a simple interface.

    Usage:
        elevation_svc = ElevationService()
        elevation = elevation_svc.get_elevation(37.1299, -80.4094)
        profile = elevation_svc.get_elevation_profile([(37.12, -80.40), (37.13, -80.41)])
    """

    def __init__(self,
                 threedep_dir: str = "./elevation_data/3dep_1m",
                 srtm_file: str = "elevation_data/srtm_90m/srtm_20_05.tif",
                 use_3dep: bool = True,
                 use_srtm: bool = True):
        """
        Initialize elevation service.

        Args:
            threedep_dir: Directory containing 3DEP 1m tiles
            srtm_file: Path to SRTM 90m file
            use_3dep: Enable 3DEP 1m source
            use_srtm: Enable SRTM 90m source
        """
        self.sources = {}
        self.active_source = None

        # Initialize SRTM source
        if use_srtm and os.path.exists(srtm_file):
            try:
                self.sources['srtm'] = SRTMElevationSource(srtm_file)
                logger.debug("SRTM elevation source initialized")
            except Exception as e:
                logger.error(f"Failed to initialize SRTM source: {e}")

        # Initialize 3DEP source
        if use_3dep and RASTERIO_AVAILABLE:
            try:
                threedep_source = LocalThreeDEPSource(threedep_dir)
                if len(threedep_source.tile_index) > 0:
                    self.sources['3dep'] = threedep_source
                    logger.debug(f"3DEP source initialized with {len(threedep_source.tile_index)} tiles")
                else:
                    logger.warning("No 3DEP tiles found")
            except Exception as e:
                logger.error(f"Failed to initialize 3DEP source: {e}")

        # Configure active source with fallback
        if '3dep' in self.sources and 'srtm' in self.sources:
            self.active_source = HybridElevationSource(
                self.sources['3dep'],
                self.sources['srtm']
            )
            logger.info("Using hybrid elevation (3DEP 1m → SRTM 90m fallback)")
        elif '3dep' in self.sources:
            self.active_source = self.sources['3dep']
            logger.info("Using 3DEP 1m elevation only")
        elif 'srtm' in self.sources:
            self.active_source = self.sources['srtm']
            logger.info("Using SRTM 90m elevation only")
        else:
            logger.error("No elevation sources available")

    def get_elevation(self, lat: float, lon: float) -> Optional[float]:
        """Get elevation at a specific coordinate in meters"""
        if not self.active_source:
            return None
        return self.active_source.get_elevation(lat, lon)

    def get_elevation_profile(self, coordinates: List[Tuple[float, float]]) -> List[float]:
        """Get elevation profile for a list of (lat, lon) tuples in meters"""
        if not self.active_source:
            return [0.0] * len(coordinates)
        return self.active_source.get_elevation_profile(coordinates)

    def add_elevation_to_graph(self, graph):
        """Add elevation attribute to all nodes in a NetworkX graph"""
        if not self.active_source:
            logger.warning("No elevation source available for graph")
            return

        for node, data in graph.nodes(data=True):
            if 'y' in data and 'x' in data:
                lat, lon = data['y'], data['x']
                elevation = self.get_elevation(lat, lon)
                data['elevation'] = elevation if elevation is not None else 0.0

    def is_available(self) -> bool:
        """Check if any elevation source is available"""
        return self.active_source is not None

    def get_stats(self) -> Dict:
        """Get usage statistics (if using hybrid source)"""
        if isinstance(self.active_source, HybridElevationSource):
            return self.active_source.get_stats()
        return {}

    def close(self):
        """Close all elevation data sources"""
        for source in self.sources.values():
            if hasattr(source, 'close'):
                source.close()


# Global elevation service instance (singleton pattern)
_global_elevation_service = None


def get_elevation_service() -> ElevationService:
    """
    Get global elevation service instance (singleton).

    Returns:
        ElevationService instance
    """
    global _global_elevation_service

    if _global_elevation_service is None:
        _global_elevation_service = ElevationService()

    return _global_elevation_service


# Update __all__ exports
__all__ = [
    'ElevationDataSource',
    'SRTMElevationSource',
    'LocalThreeDEPSource',
    'HybridElevationSource',
    'ElevationService',
    'get_elevation_service'
]
