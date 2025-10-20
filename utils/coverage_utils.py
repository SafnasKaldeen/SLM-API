"""
Core utility functions for coverage area calculations
"""

import math
from typing import Tuple
from functools import lru_cache


@lru_cache(maxsize=10000)
def calculate_distance_cached(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Cached distance calculation using Haversine formula
    
    Args:
        lat1, lon1: Starting coordinates
        lat2, lon2: Ending coordinates
    
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * 
         math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def parse_location(location_str: str) -> Tuple[float, float]:
    """
    Parse a location string into (lat, lon) tuple
    
    Args:
        location_str: String in 'lat,lon' format
    
    Returns:
        Tuple of (latitude, longitude)
    
    Raises:
        ValueError: If location format is invalid
    """
    try:
        if ',' in location_str:
            lat_str, lon_str = location_str.split(',')
            lat = float(lat_str.strip())
            lon = float(lon_str.strip())
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("Latitude must be between -90 and 90 and longitude between -180 and 180.")
            return (lat, lon)
        else:
            raise ValueError("Location must be in 'lat,lon' format")
    except Exception as e:
        raise ValueError(f"Invalid location format: {location_str}. Error: {e}")


def create_circular_boundary(lat: float, lon: float, radius_km: float, 
                            num_points: int = 32) -> list:
    """
    Create circular boundary points around a center location
    
    Args:
        lat, lon: Center coordinates
        radius_km: Radius in kilometers
        num_points: Number of points to generate
    
    Returns:
        List of [lat, lon] coordinate pairs
    """
    points = []
    angle_step = 2 * math.pi / num_points
    
    # Pre-calculate constants
    lat_offset_per_km = 1 / 111.32
    lon_offset_per_km = 1 / (111.32 * math.cos(math.radians(lat)))
    
    for i in range(num_points):
        angle = angle_step * i
        
        lat_offset = radius_km * lat_offset_per_km * math.cos(angle)
        lon_offset = radius_km * lon_offset_per_km * math.sin(angle)
        
        new_lat = lat + lat_offset
        new_lon = lon + lon_offset
        points.append([new_lat, new_lon])
    
    return points


def get_sri_lanka_bounds() -> dict:
    """
    Get bounding box coordinates for Sri Lanka
    
    Returns:
        Dictionary with min/max lat/lon values
    """
    return {
        'min_lat': 5.9,
        'max_lat': 9.9,
        'min_lon': 79.5,
        'max_lon': 81.9
    }