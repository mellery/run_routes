#!/usr/bin/env python3
"""
OSMnx + 3DEP Integration Module
Enhanced elevation processing using OSMnx with 3DEP 1m data
"""

import os
import time
import logging
from typing import List, Optional, Tuple, Union
import networkx as nx
import osmnx as ox
from osmnx_config import get_config, ensure_configured
from route import get_elevation_from_raster, add_enhanced_elevation_to_graph

# Configure logging
logger = logging.getLogger(__name__)

class OSMnx3DEPIntegration:
    """Integration class for OSMnx with 3DEP elevation data"""
    
    def __init__(self, tile_directory: str = "elevation_data/3dep_1m/tiles"):
        """Initialize integration
        
        Args:
            tile_directory: Directory containing 3DEP tiles
        """
        self.tile_directory = tile_directory
        self.config = get_config()
        ensure_configured()
        
    def get_3dep_tiles(self, filter_crs: bool = True) -> List[str]:
        """Get all available 3DEP tiles
        
        Args:
            filter_crs: Whether to filter tiles by consistent CRS (recommended)
        """
        all_tiles = self.config.get_3dep_tiles(self.tile_directory)
        
        if not filter_crs or not all_tiles:
            return all_tiles
        
        # Filter tiles to use only consistent CRS (UTM Zone 17N for Christiansburg)
        # Look for tiles containing "17_x" which indicates UTM Zone 17N
        filtered_tiles = [tile for tile in all_tiles if "17_x" in tile]
        
        logger.info(f"Filtered {len(all_tiles)} tiles to {len(filtered_tiles)} with consistent CRS")
        
        return filtered_tiles if filtered_tiles else all_tiles
        
    def get_covering_3dep_tiles(self) -> List[str]:
        """Get 3DEP tiles that actually cover Christiansburg area
        
        Returns:
            List of tile paths that cover the target coordinates
        """
        # Target coordinates for Christiansburg, VA
        target_lat, target_lon = 37.1299, -80.4094
        
        # Get all filtered tiles
        all_tiles = self.get_3dep_tiles()
        
        if not all_tiles:
            return []
        
        covering_tiles = []
        
        # Import required modules
        import rasterio
        import pyproj
        
        for tile_path in all_tiles:
            try:
                with rasterio.open(tile_path) as src:
                    # Convert target coordinates to tile CRS
                    transformer = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                    utm_x, utm_y = transformer.transform(target_lon, target_lat)
                    
                    # Check if coordinates are within tile bounds
                    bounds = src.bounds
                    if (bounds.left <= utm_x <= bounds.right and 
                        bounds.bottom <= utm_y <= bounds.top):
                        
                        # Verify we can extract valid elevation
                        row, col = src.index(utm_x, utm_y)
                        if 0 <= row < src.height and 0 <= col < src.width:
                            elevation = src.read(1)[row, col]
                            if elevation != src.nodata:
                                covering_tiles.append(tile_path)
                                logger.debug(f"Found covering tile: {os.path.basename(tile_path)}")
                                
            except Exception as e:
                logger.debug(f"Error checking tile {os.path.basename(tile_path)}: {e}")
        
        if covering_tiles:
            logger.info(f"Found {len(covering_tiles)} tiles covering target area")
        else:
            logger.warning("No tiles found covering target area")
            
        return covering_tiles
        
    def add_3dep_elevation_osmnx(self, 
                                graph: nx.Graph,
                                add_grades: bool = True,
                                cpus: Optional[int] = None,
                                auto_project: bool = True) -> nx.Graph:
        """Add 3DEP elevation using OSMnx for optimal accuracy
        
        Args:
            graph: NetworkX graph to add elevation to
            add_grades: Whether to add edge grades
            cpus: Number of CPU cores to use (None for all)
            auto_project: Whether to automatically project graph to match raster CRS
            
        Returns:
            Graph with elevation data and optionally edge grades
        """
        start_time = time.time()
        
        # Get 3DEP tiles that actually cover the area
        tile_paths = self.get_covering_3dep_tiles()
        
        if not tile_paths:
            # Fall back to all filtered tiles if no covering tiles found
            tile_paths = self.get_3dep_tiles()
            logger.warning("No covering tiles found, using all filtered tiles")
        
        if not tile_paths:
            raise ValueError(f"No 3DEP tiles found in {self.tile_directory}")
        
        logger.info(f"Adding 3DEP elevation using OSMnx with {len(tile_paths)} tiles")
        
        try:
            # CRITICAL FIX: Check minimum graph size for OSMnx compatibility  
            if len(graph.nodes) < 10:
                logger.warning(f"Graph too small ({len(graph.nodes)} nodes) - OSMnx may fail, will use manual elevation extraction")
                # For very small graphs, use manual elevation extraction instead of OSMnx
                return self._add_elevation_manual_small_graph(graph, tile_paths[0], add_grades)
            
            if len(graph.nodes) < 50:
                logger.info(f"Small graph ({len(graph.nodes)} nodes) - using single CPU to avoid multiprocessing issues")
                cpus = 1  # Force single-threaded processing for small graphs
            
            # CRITICAL FIX: Project graph to match raster CRS for accurate elevation extraction
            original_graph = graph
            if auto_project:
                # Get CRS from first tile
                import rasterio
                with rasterio.open(tile_paths[0]) as src:
                    target_crs = src.crs
                
                # Check if projection is needed
                graph_crs = graph.graph.get('crs', 'epsg:4326')
                if str(graph_crs).lower() != str(target_crs).lower():
                    logger.info(f"Projecting graph from {graph_crs} to {target_crs}")
                    graph = ox.project_graph(graph, to_crs=target_crs)
            
            # Handle single vs multiple tiles
            if len(tile_paths) == 1:
                raster_input = tile_paths[0]
            else:
                raster_input = tile_paths
                
            # Use OSMnx to add elevation from raster tiles
            # Force single CPU for small graphs to avoid multiprocessing issues
            graph = ox.elevation.add_node_elevations_raster(
                graph, 
                raster_input,
                band=1,
                cpus=cpus
            )
            
            # Project back to original CRS if needed
            if auto_project and original_graph.graph.get('crs') != graph.graph.get('crs'):
                logger.info(f"Projecting graph back to original CRS")
                graph = ox.project_graph(graph, to_crs=original_graph.graph.get('crs', 'epsg:4326'))
            
            # Add edge grades if requested
            if add_grades:
                logger.info("Adding edge grades...")
                graph = ox.elevation.add_edge_grades(graph, add_absolute=True)
            
            # Analyze results
            elevation_stats = self._analyze_elevation_data(graph)
            
            processing_time = time.time() - start_time
            logger.info(f"OSMnx 3DEP processing completed in {processing_time:.1f}s")
            logger.info(f"Elevation stats: {elevation_stats}")
            
            return graph
            
        except Exception as e:
            logger.error(f"OSMnx 3DEP processing failed: {e}")
            raise
            
    def add_elevation_hybrid_osmnx(self,
                                  graph: nx.Graph,
                                  fallback_raster: str = "elevation_data/srtm_90m/srtm_20_05.tif",
                                  min_coverage: float = 0.9) -> nx.Graph:
        """Hybrid approach: OSMnx for 3DEP coverage, fallback for gaps
        
        Args:
            graph: NetworkX graph to add elevation to
            fallback_raster: SRTM raster for gap filling
            min_coverage: Minimum coverage threshold for 3DEP
            
        Returns:
            Graph with elevation data
        """
        start_time = time.time()
        
        try:
            # Primary: OSMnx with 3DEP tiles
            graph = self.add_3dep_elevation_osmnx(graph, add_grades=False)
            
            # Check coverage
            coverage = self._calculate_coverage(graph)
            logger.info(f"3DEP coverage: {coverage:.1f}%")
            
            if coverage < min_coverage:
                logger.info(f"Coverage {coverage:.1f}% below threshold {min_coverage:.1f}%, filling gaps...")
                graph = self._fill_elevation_gaps(graph, fallback_raster)
                
                # Recalculate coverage
                final_coverage = self._calculate_coverage(graph)
                logger.info(f"Final coverage: {final_coverage:.1f}%")
            
            # Add edge grades
            logger.info("Adding edge grades...")
            graph = ox.elevation.add_edge_grades(graph, add_absolute=True)
            
            processing_time = time.time() - start_time
            logger.info(f"Hybrid elevation processing completed in {processing_time:.1f}s")
            
            return graph
            
        except Exception as e:
            logger.error(f"Hybrid elevation processing failed: {e}")
            # Fallback to current implementation
            logger.info("Falling back to current elevation implementation...")
            return add_enhanced_elevation_to_graph(graph, use_3dep=True, fallback_raster=fallback_raster)
    
    def compare_elevation_methods(self, 
                                 graph: nx.Graph,
                                 fallback_raster: str = "elevation_data/srtm_90m/srtm_20_05.tif") -> dict:
        """Compare current vs OSMnx elevation methods
        
        Args:
            graph: NetworkX graph for comparison
            fallback_raster: SRTM raster for fallback
            
        Returns:
            Comparison results dictionary
        """
        results = {
            'current_method': {},
            'osmnx_method': {},
            'comparison': {}
        }
        
        # Test current method
        logger.info("Testing current elevation method...")
        start_time = time.time()
        graph_current = add_enhanced_elevation_to_graph(
            graph.copy(), 
            use_3dep=True, 
            fallback_raster=fallback_raster
        )
        current_time = time.time() - start_time
        
        results['current_method'] = {
            'processing_time': current_time,
            'stats': self._analyze_elevation_data(graph_current)
        }
        
        # Test OSMnx method
        logger.info("Testing OSMnx elevation method...")
        start_time = time.time()
        try:
            graph_osmnx = self.add_elevation_hybrid_osmnx(graph.copy(), fallback_raster)
            osmnx_time = time.time() - start_time
            
            results['osmnx_method'] = {
                'processing_time': osmnx_time,
                'stats': self._analyze_elevation_data(graph_osmnx)
            }
            
            # Compare results
            results['comparison'] = self._compare_elevation_values(graph_current, graph_osmnx)
            
        except Exception as e:
            logger.error(f"OSMnx method failed: {e}")
            results['osmnx_method'] = {'error': str(e)}
            
        return results
    
    def _analyze_elevation_data(self, graph: nx.Graph) -> dict:
        """Analyze elevation data quality in graph"""
        elevations = []
        nodes_with_elevation = 0
        
        for node_id, data in graph.nodes(data=True):
            elevation = data.get('elevation', 0)
            if elevation and elevation != 0:
                elevations.append(elevation)
                nodes_with_elevation += 1
        
        if not elevations:
            return {'nodes_with_elevation': 0, 'coverage': 0.0}
            
        return {
            'nodes_with_elevation': nodes_with_elevation,
            'total_nodes': len(graph.nodes),
            'coverage': nodes_with_elevation / len(graph.nodes) * 100,
            'elevation_range': (min(elevations), max(elevations)),
            'mean_elevation': sum(elevations) / len(elevations)
        }
    
    def _calculate_coverage(self, graph: nx.Graph) -> float:
        """Calculate elevation data coverage percentage"""
        nodes_with_elevation = sum(1 for _, data in graph.nodes(data=True) 
                                 if data.get('elevation', 0) > 0)
        return nodes_with_elevation / len(graph.nodes) * 100
    
    def _add_elevation_manual_small_graph(self, graph: nx.Graph, tile_path: str, add_grades: bool = True) -> nx.Graph:
        """Manual elevation extraction for very small graphs where OSMnx fails
        
        Args:
            graph: Small NetworkX graph
            tile_path: Path to 3DEP tile
            add_grades: Whether to add edge grades
            
        Returns:
            Graph with elevation data
        """
        import rasterio
        import pyproj
        import time
        
        start_time = time.time()
        logger.info(f"Using manual elevation extraction for small graph ({len(graph.nodes)} nodes)")
        
        try:
            with rasterio.open(tile_path) as src:
                # Set up coordinate transformation
                transformer = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                
                nodes_with_elevation = 0
                
                # Extract elevation for each node
                for node_id, data in graph.nodes(data=True):
                    lat, lon = data['y'], data['x']
                    
                    # Transform to tile CRS
                    tile_x, tile_y = transformer.transform(lon, lat)
                    
                    # Check if in bounds
                    if (src.bounds.left <= tile_x <= src.bounds.right and 
                        src.bounds.bottom <= tile_y <= src.bounds.top):
                        
                        try:
                            # Extract elevation
                            row, col = src.index(tile_x, tile_y)
                            if 0 <= row < src.height and 0 <= col < src.width:
                                elevation = src.read(1)[row, col]
                                if elevation != src.nodata:
                                    graph.nodes[node_id]['elevation'] = float(elevation)
                                    nodes_with_elevation += 1
                                else:
                                    graph.nodes[node_id]['elevation'] = 0.0
                            else:
                                graph.nodes[node_id]['elevation'] = 0.0
                        except Exception:
                            graph.nodes[node_id]['elevation'] = 0.0
                    else:
                        graph.nodes[node_id]['elevation'] = 0.0
                
                # Add edge grades if requested
                if add_grades and len(graph.edges) > 0:
                    logger.info("Adding edge grades manually...")
                    
                    # Calculate grades manually
                    for u, v, data in graph.edges(data=True):
                        u_elevation = graph.nodes[u].get('elevation', 0)
                        v_elevation = graph.nodes[v].get('elevation', 0)
                        
                        # Calculate grade (rise over run)
                        if 'length' in data and data['length'] > 0:
                            grade = (v_elevation - u_elevation) / data['length']
                            data['grade'] = grade
                            data['grade_abs'] = abs(grade)
                        else:
                            data['grade'] = 0.0
                            data['grade_abs'] = 0.0
                
                processing_time = time.time() - start_time
                logger.info(f"Manual elevation extraction completed in {processing_time:.1f}s")
                logger.info(f"Added elevation to {nodes_with_elevation}/{len(graph.nodes)} nodes")
                
                return graph
                
        except Exception as e:
            logger.error(f"Manual elevation extraction failed: {e}")
            raise
    
    def _fill_elevation_gaps(self, graph: nx.Graph, fallback_raster: str) -> nx.Graph:
        """Fill elevation gaps using fallback raster"""
        if not os.path.exists(fallback_raster):
            logger.warning(f"Fallback raster {fallback_raster} not found")
            return graph
        
        gaps_filled = 0
        
        for node_id, data in graph.nodes(data=True):
            if data.get('elevation', 0) <= 0:
                # Try to get elevation from fallback raster
                elevation = get_elevation_from_raster(fallback_raster, data['y'], data['x'])
                if elevation is not None:
                    graph.nodes[node_id]['elevation'] = elevation
                    gaps_filled += 1
        
        logger.info(f"Filled {gaps_filled} elevation gaps using fallback raster")
        return graph
    
    def _compare_elevation_values(self, graph1: nx.Graph, graph2: nx.Graph) -> dict:
        """Compare elevation values between two graphs"""
        differences = []
        identical_count = 0
        
        for node_id in graph1.nodes():
            if node_id in graph2.nodes():
                elev1 = graph1.nodes[node_id].get('elevation', 0)
                elev2 = graph2.nodes[node_id].get('elevation', 0)
                
                if elev1 == elev2:
                    identical_count += 1
                else:
                    differences.append(abs(elev1 - elev2))
        
        if not differences:
            return {'identical_nodes': identical_count, 'mean_difference': 0.0}
            
        return {
            'identical_nodes': identical_count,
            'nodes_with_differences': len(differences),
            'mean_difference': sum(differences) / len(differences),
            'max_difference': max(differences),
            'min_difference': min(differences)
        }


# Convenience functions for backward compatibility
def add_3dep_elevation_osmnx(graph: nx.Graph, **kwargs) -> nx.Graph:
    """Add 3DEP elevation using OSMnx - convenience function"""
    integration = OSMnx3DEPIntegration()
    return integration.add_3dep_elevation_osmnx(graph, **kwargs)

def add_elevation_hybrid_osmnx(graph: nx.Graph, **kwargs) -> nx.Graph:
    """Add elevation using hybrid OSMnx approach - convenience function"""
    integration = OSMnx3DEPIntegration()
    return integration.add_elevation_hybrid_osmnx(graph, **kwargs)

def compare_elevation_methods(graph: nx.Graph, **kwargs) -> dict:
    """Compare elevation methods - convenience function"""
    integration = OSMnx3DEPIntegration()
    return integration.compare_elevation_methods(graph, **kwargs)