#!/usr/bin/env python3
"""
Graph Cache Utilities

Provides graph caching and loading functionality.
Consolidated from graph_cache.py and advanced_caching.py.
"""

import os
import pickle
from typing import Optional, Tuple
import networkx as nx


def get_cache_filename(center_point: Tuple[float, float],
                      radius_m: int,
                      network_type: str = 'all') -> str:
    """
    Generate standardized cache filename.

    Args:
        center_point: (lat, lon) tuple for network center
        radius_m: Network radius in meters
        network_type: OSMnx network type ('all', 'drive', etc.)

    Returns:
        Cache filename path

    Example:
        >>> filename = get_cache_filename((37.1299, -80.4094), 5000, 'all')
    """
    lat, lon = center_point
    return f"cache/cached_graph_{lat:.4f}_{lon:.4f}_{radius_m}m_{network_type}.pkl"


def load_cached_graph(cache_file: str) -> Optional[nx.Graph]:
    """
    Load a cached graph from pickle file.

    Args:
        cache_file: Path to cache file

    Returns:
        NetworkX graph or None if loading fails

    Example:
        >>> graph = load_cached_graph("cache/cached_graph_37.1299_-80.4094_5000m_all.pkl")
    """
    if not os.path.exists(cache_file):
        return None

    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        # Handle both old and new cache formats
        if isinstance(cache_data, dict):
            graph = cache_data.get('graph')
        else:
            graph = cache_data

        # Validate graph
        if graph is None or not isinstance(graph, nx.Graph):
            return None

        # Check for essential data
        if not graph.nodes or not graph.edges:
            return None

        return graph

    except Exception as e:
        print(f"⚠️ Failed to load cache file {cache_file}: {e}")
        return None


def save_cached_graph(graph: nx.Graph,
                     cache_file: str,
                     metadata: Optional[dict] = None) -> bool:
    """
    Save a graph to cache file.

    Args:
        graph: NetworkX graph to cache
        cache_file: Path to cache file
        metadata: Optional metadata dictionary

    Returns:
        True if successful, False otherwise

    Example:
        >>> success = save_cached_graph(graph, cache_file, {'radius_m': 5000})
    """
    try:
        # Ensure cache directory exists
        os.makedirs('cache', exist_ok=True)

        # Prepare cache data
        cache_data = {
            'graph': graph,
            'metadata': metadata or {}
        }

        # Save to file
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        return True

    except Exception as e:
        print(f"⚠️ Failed to save cache file {cache_file}: {e}")
        return False


def list_cached_graphs() -> list:
    """
    List all available cached graphs.

    Returns:
        List of cache filenames

    Example:
        >>> caches = list_cached_graphs()
        >>> print(f"Found {len(caches)} cached graphs")
    """
    # Ensure cache directory exists
    os.makedirs('cache', exist_ok=True)

    cache_files = [
        f for f in os.listdir('cache')
        if f.startswith('cached_graph_') and f.endswith('.pkl')
    ]

    if not cache_files:
        return []

    valid_caches = []
    for cache_file in sorted(cache_files):
        try:
            cache_path = os.path.join('cache', cache_file)
            graph = load_cached_graph(cache_path)
            if graph is not None:
                valid_caches.append(cache_file)
        except:
            pass

    return valid_caches


def clean_cache(keep_latest: bool = True) -> int:
    """
    Clean up old or invalid cache files.

    Args:
        keep_latest: If True, keep the most recent valid cache

    Returns:
        Number of files removed

    Example:
        >>> removed = clean_cache(keep_latest=True)
        >>> print(f"Removed {removed} cache files")
    """
    # Ensure cache directory exists
    os.makedirs('cache', exist_ok=True)

    cache_files = [
        f for f in os.listdir('cache')
        if f.startswith('cached_graph_') and f.endswith('.pkl')
    ]

    if not cache_files:
        return 0

    removed_count = 0

    # Sort by modification time (newest first)
    cache_files.sort(
        key=lambda f: os.path.getmtime(os.path.join('cache', f)),
        reverse=True
    )

    for i, cache_file in enumerate(cache_files):
        cache_path = os.path.join('cache', cache_file)

        # Keep the first (newest) file if keep_latest is True
        if keep_latest and i == 0:
            # Validate the newest file
            graph = load_cached_graph(cache_path)
            if graph is not None:
                continue

        # Remove invalid or old cache files
        try:
            os.remove(cache_path)
            removed_count += 1
        except Exception as e:
            print(f"⚠️ Failed to remove {cache_file}: {e}")

    return removed_count


def load_or_generate_graph(center_point: Tuple[float, float] = (37.1299, -80.4094),
                          radius_m: int = 5000,
                          network_type: str = 'all',
                          force_regenerate: bool = False) -> Optional[nx.Graph]:
    """
    Load cached graph or generate if not available.

    This is a convenience function that attempts to load a cached graph
    and falls back to generation if needed.

    Args:
        center_point: (lat, lon) tuple for network center
        radius_m: Network radius in meters
        network_type: OSMnx network type ('all', 'drive', etc.)
        force_regenerate: Force regeneration even if cache exists

    Returns:
        NetworkX graph or None if loading/generation fails

    Example:
        >>> graph = load_or_generate_graph(radius_m=5000)
    """
    cache_file = get_cache_filename(center_point, radius_m, network_type)

    # Try to load existing cache first
    if not force_regenerate:
        graph = load_cached_graph(cache_file)
        if graph is not None:
            return graph

    # Cache doesn't exist, need to generate
    print(f"🔄 No valid cache found, please run: python generate_cached_graph.py --radius {radius_m/1000:.1f}")
    print(f"   Or use: python setup_cache.py")

    return None


__all__ = [
    'get_cache_filename',
    'load_cached_graph',
    'save_cached_graph',
    'list_cached_graphs',
    'clean_cache',
    'load_or_generate_graph'
]
