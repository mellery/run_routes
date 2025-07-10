#!/usr/bin/env python3
"""
OSMnx Configuration Module
Optimized settings for running route optimization with 3DEP elevation data
"""

import os
import osmnx as ox
from typing import List, Optional
import glob
import logging

# Configure logging
logger = logging.getLogger(__name__)

class OSMnxConfig:
    """OSMnx configuration management for route optimization"""
    
    # Running-specific network filters - optimized for genetic algorithm
    RUNNING_FILTER = (
        '["highway"~"primary|secondary|tertiary|residential|path|living_street|unclassified"]'
        '["access"!~"private"]'
        '["surface"!~"sand|mud|gravel"]'
        '["highway"!~"motorway|trunk|motorway_link|trunk_link|footway|cycleway"]'
    )
    
    # Legacy filter with footways/cycleways (for comparison)
    RUNNING_FILTER_LEGACY = (
        '["highway"~"primary|secondary|tertiary|residential|path|footway|cycleway|living_street|unclassified"]'
        '["access"!~"private"]'
        '["surface"!~"sand|mud|gravel"]'
        '["highway"!~"motorway|trunk|motorway_link|trunk_link"]'
    )
    
    # Performance settings
    DEFAULT_TIMEOUT = 300  # 5 minutes
    DEFAULT_MEMORY_GB = 2
    DEFAULT_CACHE_FOLDER = "./cache/osmnx"
    
    def __init__(self, cache_folder: Optional[str] = None):
        """Initialize OSMnx configuration
        
        Args:
            cache_folder: Custom cache folder path
        """
        self.cache_folder = cache_folder or self.DEFAULT_CACHE_FOLDER
        self.configured = False
        
    def configure_osmnx(self, 
                       timeout: int = DEFAULT_TIMEOUT,
                       memory_gb: int = DEFAULT_MEMORY_GB,
                       use_cache: bool = True) -> None:
        """Configure OSMnx for optimal performance
        
        Args:
            timeout: Request timeout in seconds
            memory_gb: Memory limit in GB
            use_cache: Whether to use HTTP caching
        """
        if self.configured:
            return
            
        # Cache settings
        ox.settings.use_cache = use_cache
        ox.settings.cache_folder = self.cache_folder
        
        # Performance settings
        ox.settings.timeout = timeout
        ox.settings.memory = memory_gb
        # Remove conflicting timeout from requests_kwargs since ox.settings.timeout handles this
        ox.settings.requests_kwargs = {}
        
        # Create cache directory
        os.makedirs(self.cache_folder, exist_ok=True)
        
        # Log configuration
        logger.info(f"OSMnx configured: cache={use_cache}, timeout={timeout}s, memory={memory_gb}GB")
        logger.info(f"Cache folder: {self.cache_folder}")
        
        self.configured = True
        
    def get_running_network_filter(self) -> str:
        """Get network filter optimized for running routes"""
        return self.RUNNING_FILTER
        
    def get_3dep_tiles(self, tile_directory: str = "elevation_data/3dep_1m/tiles") -> List[str]:
        """Get all available 3DEP tiles
        
        Args:
            tile_directory: Directory containing 3DEP tiles
            
        Returns:
            List of tile file paths
        """
        tile_pattern = os.path.join(tile_directory, "*.tif")
        tiles = glob.glob(tile_pattern)
        
        if not tiles:
            logger.warning(f"No 3DEP tiles found in {tile_directory}")
            return []
            
        logger.info(f"Found {len(tiles)} 3DEP tiles")
        return sorted(tiles)
        
    def get_network_config(self, 
                          network_type: str = 'all',
                          simplify: bool = True,
                          retain_all: bool = False) -> dict:
        """Get network download configuration
        
        Args:
            network_type: OSMnx network type, 'running' for custom filter, or 'all' for running-optimized
            simplify: Whether to simplify the graph
            retain_all: Whether to retain all connected components
            
        Returns:
            Configuration dictionary for ox.graph_from_point()
        """
        config = {
            'simplify': simplify,
            'retain_all': retain_all
        }
        
        # Use running-optimized filter for 'all' network type (default behavior)
        # This eliminates footways and provides better GA performance
        if network_type == 'running' or network_type == 'all':
            config['custom_filter'] = self.get_running_network_filter()
        elif network_type == 'all_legacy':
            # Use original OSMnx 'all' for backwards compatibility
            config['network_type'] = 'all'
        else:
            config['network_type'] = network_type
            
        return config
        
    def optimize_graph(self, graph, 
                      simplify: bool = True,
                      consolidate_intersections: bool = True,
                      tolerance: float = 10.0,
                      preserve_nodes: bool = False):
        """Optimize graph for better performance
        
        Args:
            graph: NetworkX graph
            simplify: Whether to simplify the graph
            consolidate_intersections: Whether to consolidate nearby intersections
            tolerance: Tolerance for intersection consolidation in meters
            preserve_nodes: If True, skip aggressive optimization for route planning
            
        Returns:
            Optimized graph
        """
        original_nodes = len(graph.nodes)
        
        simplified_nodes = original_nodes  # Initialize
        
        if simplify and not graph.graph.get('simplified', False):
            logger.info("Simplifying graph...")
            graph = ox.simplify_graph(graph, strict=False)
            simplified_nodes = len(graph.nodes)
            logger.info(f"Simplified: {original_nodes} -> {simplified_nodes} nodes")
            
        if consolidate_intersections and not preserve_nodes:
            logger.info(f"Consolidating intersections (tolerance={tolerance}m)...")
            
            # CRITICAL: Ensure graph is projected for accurate consolidation
            original_crs = graph.graph.get('crs', 'epsg:4326')
            is_geographic = str(original_crs).lower().startswith('epsg:4326') or 'geographic' in str(original_crs).lower()
            
            if is_geographic:
                logger.info("Graph is in geographic CRS, projecting for consolidation...")
                # Project to appropriate UTM zone for accurate meter-based operations
                graph_projected = ox.project_graph(graph)
                
                # Perform consolidation on projected graph
                graph_consolidated = ox.consolidate_intersections(graph_projected, tolerance=tolerance)
                
                # Project back to original CRS
                logger.info("Projecting consolidated graph back to original CRS...")
                graph = ox.project_graph(graph_consolidated, to_crs=original_crs)
            else:
                logger.info("Graph already projected, consolidating directly...")
                graph = ox.consolidate_intersections(graph, tolerance=tolerance)
            
            final_nodes = len(graph.nodes)
            logger.info(f"Consolidated: {simplified_nodes} -> {final_nodes} nodes")
            
            # Safety check - if we've reduced nodes too much, warn but continue
            reduction_ratio = final_nodes / original_nodes
            if reduction_ratio < 0.1:  # Less than 10% of original nodes remaining
                logger.warning(f"Consolidation very aggressive ({reduction_ratio:.1%} nodes remaining)")
                logger.warning(f"Consider increasing tolerance above {tolerance}m for this area")
            elif reduction_ratio < 0.3:  # Less than 30% remaining
                logger.info(f"Consolidation effective ({reduction_ratio:.1%} nodes remaining)")
            else:
                logger.info(f"Consolidation moderate ({reduction_ratio:.1%} nodes remaining)")
                
        elif consolidate_intersections and preserve_nodes:
            logger.info("Skipping intersection consolidation (preserve_nodes=True)")
            
        return graph


# Global configuration instance
_config = OSMnxConfig()

def configure_osmnx(**kwargs):
    """Configure OSMnx globally"""
    _config.configure_osmnx(**kwargs)
    
def get_config() -> OSMnxConfig:
    """Get global OSMnx configuration"""
    return _config

def ensure_configured():
    """Ensure OSMnx is configured"""
    if not _config.configured:
        _config.configure_osmnx()