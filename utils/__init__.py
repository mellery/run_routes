"""
Shared utilities for the run_routes project.

This package provides the bottom layer of the architecture hierarchy:
- elevation: Unified elevation data access (3DEP 1m + SRTM 90m)
- geometry: Geographic calculations (haversine distance, bearing, etc.)
- graph_utils: Graph operations and manipulations
- cache: Graph caching utilities

Architecture hierarchy:
    utils/ (bottom layer)
       ↓
    genetic_algorithm/
       ↓
    route_services/
       ↓
    applications (CLI, web)
"""

# Export elevation utilities
from .elevation import (
    ElevationService,
    get_elevation_service,
    ElevationDataSource,
    SRTMElevationSource,
    LocalThreeDEPSource,
    HybridElevationSource
)

# Export geometry utilities
from .geometry import (
    haversine_distance,
    calculate_bearing,
    destination_point
)

# Export graph utilities
from .graph_utils import (
    get_street_data,
    has_elevation_data,
    add_elevation_to_edges,
    add_running_weights,
    get_nodes_within_distance,
    create_distance_constrained_subgraph
)

# Export cache utilities
from .cache import (
    get_cache_filename,
    load_cached_graph,
    save_cached_graph,
    list_cached_graphs,
    clean_cache,
    load_or_generate_graph
)

__all__ = [
    # Elevation
    'ElevationService',
    'get_elevation_service',
    'ElevationDataSource',
    'SRTMElevationSource',
    'LocalThreeDEPSource',
    'HybridElevationSource',
    # Geometry
    'haversine_distance',
    'calculate_bearing',
    'destination_point',
    # Graph
    'get_street_data',
    'has_elevation_data',
    'add_elevation_to_edges',
    'add_running_weights',
    'get_nodes_within_distance',
    'create_distance_constrained_subgraph',
    # Cache
    'get_cache_filename',
    'load_cached_graph',
    'save_cached_graph',
    'list_cached_graphs',
    'clean_cache',
    'load_or_generate_graph'
]
