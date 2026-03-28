#!/usr/bin/env python3
"""
Geometric Utility Functions

Provides geographic distance calculations and coordinate transformations.
Consolidated from route.py to create a cleaner architecture.
"""

from math import radians, cos, sin, asin, sqrt, atan2, degrees


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth in meters.

    Uses the haversine formula for calculating distances on a sphere.

    Args:
        lat1: Latitude of first point in decimal degrees
        lon1: Longitude of first point in decimal degrees
        lat2: Latitude of second point in decimal degrees
        lon2: Longitude of second point in decimal degrees

    Returns:
        Distance in meters

    Example:
        >>> distance = haversine_distance(37.1299, -80.4094, 37.1300, -80.4095)
        >>> print(f"{distance:.2f}m")
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    # Radius of earth in meters
    r = 6371000
    return c * r


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the bearing (direction) from point 1 to point 2.

    Args:
        lat1: Latitude of first point in decimal degrees
        lon1: Longitude of first point in decimal degrees
        lat2: Latitude of second point in decimal degrees
        lon2: Longitude of second point in decimal degrees

    Returns:
        Bearing in degrees (0-360, where 0 is North)
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1

    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)

    bearing = atan2(x, y)
    bearing = degrees(bearing)
    bearing = (bearing + 360) % 360

    return bearing


def destination_point(lat: float, lon: float, bearing: float, distance: float) -> tuple:
    """
    Calculate destination point given start point, bearing, and distance.

    Args:
        lat: Latitude of start point in decimal degrees
        lon: Longitude of start point in decimal degrees
        bearing: Bearing in degrees (0-360)
        distance: Distance in meters

    Returns:
        Tuple of (destination_lat, destination_lon) in decimal degrees
    """
    # Earth radius in meters
    R = 6371000

    # Convert to radians
    lat1 = radians(lat)
    lon1 = radians(lon)
    bearing_rad = radians(bearing)

    # Calculate destination
    lat2 = asin(sin(lat1) * cos(distance / R) +
                cos(lat1) * sin(distance / R) * cos(bearing_rad))

    lon2 = lon1 + atan2(sin(bearing_rad) * sin(distance / R) * cos(lat1),
                        cos(distance / R) - sin(lat1) * sin(lat2))

    # Convert back to degrees
    lat2 = degrees(lat2)
    lon2 = degrees(lon2)

    return lat2, lon2


__all__ = [
    'haversine_distance',
    'calculate_bearing',
    'destination_point'
]
