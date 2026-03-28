#!/usr/bin/env python3
"""
Graph Utility Functions

Provides graph manipulation and analysis utilities for NetworkX graphs.
Consolidated from route.py to create a cleaner architecture.
"""

import networkx as nx
import osmnx as ox
from typing import List, Set, Optional


def get_street_data(place_name: str) -> nx.MultiDiGraph:
    """
    Download street network data for a place using OSMnx.

    Args:
        place_name: Name of place (e.g., "Christiansburg, Virginia, USA")

    Returns:
        NetworkX MultiDiGraph with street network

    Example:
        >>> graph = get_street_data("Christiansburg, Virginia, USA")
    """
    return ox.graph_from_place(place_name, network_type='all')


def has_elevation_data(graph: nx.Graph) -> bool:
    """
    Check if graph nodes have elevation data.

    Args:
        graph: NetworkX graph

    Returns:
        True if elevation data exists in at least 10% of nodes

    Example:
        >>> has_elev = has_elevation_data(graph)
    """
    if not graph.nodes:
        return False

    nodes_with_elevation = sum(
        1 for _, data in graph.nodes(data=True)
        if 'elevation' in data and data['elevation'] != 0
    )

    # Consider elevation data present if at least 10% of nodes have it
    threshold = max(1, len(graph.nodes) * 0.1)
    return nodes_with_elevation >= threshold


def add_elevation_to_edges(graph: nx.Graph) -> nx.Graph:
    """
    Calculate and add elevation gain/loss to graph edges.

    Uses node elevation data to compute:
    - grade: Percentage grade of edge
    - elevation_gain: Positive elevation change
    - elevation_loss: Negative elevation change (as positive value)

    Args:
        graph: NetworkX graph with node elevation data

    Returns:
        Modified graph with edge elevation attributes

    Example:
        >>> graph = add_elevation_to_edges(graph)
    """
    for u, v, data in graph.edges(data=True):
        if u in graph.nodes and v in graph.nodes:
            u_elev = graph.nodes[u].get('elevation', 0)
            v_elev = graph.nodes[v].get('elevation', 0)

            elevation_change = v_elev - u_elev
            length = data.get('length', 0)

            if length > 0:
                grade = (elevation_change / length) * 100
                data['grade'] = grade

                if elevation_change > 0:
                    data['elevation_gain'] = elevation_change
                    data['elevation_loss'] = 0
                else:
                    data['elevation_gain'] = 0
                    data['elevation_loss'] = abs(elevation_change)
            else:
                data['grade'] = 0
                data['elevation_gain'] = 0
                data['elevation_loss'] = 0

    return graph


def add_running_weights(graph: nx.Graph,
                       elevation_weight: float = 0.1,
                       grade_penalty: float = 2.0) -> nx.Graph:
    """
    Add running-specific weights to graph edges.

    Adjusts edge weights based on elevation gain to better represent
    running effort. Uphill segments get higher weights.

    Args:
        graph: NetworkX graph with elevation data
        elevation_weight: Weight factor for elevation (default 0.1)
        grade_penalty: Additional penalty for steep grades (default 2.0)

    Returns:
        Modified graph with running weights

    Example:
        >>> graph = add_running_weights(graph, elevation_weight=0.15)
    """
    for u, v, data in graph.edges(data=True):
        base_length = data.get('length', 0)
        elevation_gain = data.get('elevation_gain', 0)
        grade = data.get('grade', 0)

        # Base running weight is the distance
        running_weight = base_length

        # Add penalty for elevation gain
        if elevation_gain > 0:
            running_weight += elevation_gain * elevation_weight

        # Additional penalty for steep grades (>5%)
        if abs(grade) > 5:
            grade_factor = (abs(grade) - 5) / 5  # Scale factor
            running_weight += base_length * grade_factor * grade_penalty

        data['running_weight'] = running_weight

    return graph


def get_nodes_within_distance(graph: nx.Graph,
                             start_node: int,
                             max_distance_km: float) -> List[int]:
    """
    Get all nodes reachable within a maximum distance from start node.

    Uses Dijkstra's algorithm to find all reachable nodes.

    Args:
        graph: NetworkX graph
        start_node: Starting node ID
        max_distance_km: Maximum distance in kilometers

    Returns:
        List of node IDs within distance

    Example:
        >>> nodes = get_nodes_within_distance(graph, 1529188403, 5.0)
    """
    max_distance_m = max_distance_km * 1000

    try:
        # Use Dijkstra to find all nodes within distance
        distances = nx.single_source_dijkstra_path_length(
            graph, start_node, weight='length', cutoff=max_distance_m
        )
        return list(distances.keys())
    except Exception:
        return [start_node]


def create_distance_constrained_subgraph(graph: nx.Graph,
                                        start_node: int,
                                        max_distance_km: float) -> nx.Graph:
    """
    Create a subgraph containing only nodes within max distance from start.

    Args:
        graph: NetworkX graph
        start_node: Starting node ID
        max_distance_km: Maximum distance in kilometers

    Returns:
        Subgraph containing only reachable nodes

    Example:
        >>> subgraph = create_distance_constrained_subgraph(graph, 1529188403, 5.0)
    """
    nodes_within_distance = get_nodes_within_distance(graph, start_node, max_distance_km)
    return graph.subgraph(nodes_within_distance).copy()


__all__ = [
    'get_street_data',
    'has_elevation_data',
    'add_elevation_to_edges',
    'add_running_weights',
    'get_nodes_within_distance',
    'create_distance_constrained_subgraph'
]
