#!/usr/bin/env python3
"""
Route Objective Enum
Defines optimization objectives for genetic algorithm route planning
"""

from enum import Enum


class RouteObjective(Enum):
    """Route optimization objectives for genetic algorithm"""
    
    # Distance-based objectives
    MINIMIZE_DISTANCE = "minimize_distance"
    
    # Elevation-based objectives  
    MAXIMIZE_ELEVATION = "maximize_elevation"
    MINIMIZE_ELEVATION = "minimize_elevation"
    
    # Difficulty-based objectives
    MINIMIZE_DIFFICULTY = "minimize_difficulty"
    
    # Balanced objectives
    BALANCED_ROUTE = "balanced_route"
    
    # Scenic objectives
    SCENIC_ROUTE = "scenic_route"
    
    # Efficiency objectives
    EFFICIENCY_ROUTE = "efficiency_route"


# Alias for backward compatibility
MINIMIZE_DISTANCE = RouteObjective.MINIMIZE_DISTANCE
MAXIMIZE_ELEVATION = RouteObjective.MAXIMIZE_ELEVATION
MINIMIZE_ELEVATION = RouteObjective.MINIMIZE_ELEVATION
MINIMIZE_DIFFICULTY = RouteObjective.MINIMIZE_DIFFICULTY
BALANCED_ROUTE = RouteObjective.BALANCED_ROUTE
SCENIC_ROUTE = RouteObjective.SCENIC_ROUTE
EFFICIENCY_ROUTE = RouteObjective.EFFICIENCY_ROUTE