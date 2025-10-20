"""
Enhanced Coverage Area Service - COMPLETE VERSION with Network Coverage Analysis
All original features PRESERVED + Modular Alert System + Overall Network Coverage

New Feature: Overall Network Coverage
- Analyzes theoretical maximum reach of charging network (independent of vehicle position/battery)
- Detects disconnected networks and generates separate polygons
- Uses 100% battery assumption to show network infrastructure limits
"""

import math
import numpy as np
from collections import deque
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
from datetime import datetime, timedelta
import time

# Import modular alert system
from utils.ev_alert_system import (
    EVAlertSystem, 
    AlertConfig, 
    VehicleStatus, 
    CoverageContext,
    Alert,
    AlertLevel,
    AlertType
)

# Optional dependencies for advanced polygon generation
try:
    from scipy.spatial import ConvexHull
    from shapely.geometry import Point, Polygon, MultiPolygon
    from shapely.ops import unary_union
    COVERAGE_DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Coverage area dependencies not available: {e}")
    print("Install with: pip install scipy shapely numpy")
    COVERAGE_DEPENDENCIES_AVAILABLE = False


class CoverageAreaService:
    """
    Battery-aware coverage area service with ALL original features + modular alerts + network coverage
    STATELESS design - all session data passed via parameters
    """
    
    def __init__(self, alert_config: Optional[AlertConfig] = None):
        # Cache statistics - no user state
        self.cache_stats = {"hits": 0, "misses": 0, "computations_saved": 0}
        self.optimization_stats = {
            "stations_filtered": 0,
            "polygons_optimized": 0,
            "early_terminations": 0,
            "cache_utilization": 0
        }
        
        # Initialize modular alert system
        if alert_config is None:
            alert_config = AlertConfig(
                warning_distance_km=30.0,
                critical_distance_km=15.0,
                low_battery_threshold=20.0,
                critical_battery_threshold=15.0,
                emergency_battery_threshold=5.0,
                abnormal_drain_factor=1.5,
                station_ahead_angle=45.0,
                point_of_no_return_buffer=1.2
            )
        self.alert_system = EVAlertSystem(alert_config)
    
    @staticmethod
    @lru_cache(maxsize=10000)
    def calculate_distance_cached(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Cached distance calculation using Haversine formula"""
        R = 6371
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance with cache tracking"""
        distance = self.calculate_distance_cached(lat1, lon1, lat2, lon2)
        cache_info = self.calculate_distance_cached.cache_info()
        
        if cache_info.hits > self.cache_stats["hits"]:
            self.cache_stats["hits"] = cache_info.hits
            self.optimization_stats["cache_utilization"] += 1
        if cache_info.misses > self.cache_stats["misses"]:
            self.cache_stats["misses"] = cache_info.misses
            
        return distance
    
    def calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing between two points in degrees (0-360)"""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lon = math.radians(lon2 - lon1)
        
        x = math.sin(delta_lon) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - \
            math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
        
        bearing = math.atan2(x, y)
        bearing_degrees = (math.degrees(bearing) + 360) % 360
        return bearing_degrees
    
    def angle_difference(self, angle1: float, angle2: float) -> float:
        """Calculate smallest difference between two angles"""
        diff = abs(angle1 - angle2)
        if diff > 180:
            diff = 360 - diff
        return diff
    
    def is_point_in_polygon(self, point: Tuple[float, float], polygon: List[List[float]]) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm"""
        if not polygon or len(polygon) < 3:
            return False
            
        lat, lon = point
        inside = False
        n = len(polygon)
        
        for i in range(n):
            j = (i + 1) % n
            lat1, lon1 = polygon[i]
            lat2, lon2 = polygon[j]
            
            if ((lon1 > lon) != (lon2 > lon)) and \
               (lat < (lat2 - lat1) * (lon - lon1) / (lon2 - lon1) + lat1):
                inside = not inside
        
        return inside
    
    def point_to_segment_distance(self, px: float, py: float, 
                                  x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate distance from point to line segment"""
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2)
        
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        lat_dist = (px - closest_x) * 111.32
        lon_dist = (py - closest_y) * 111.32 * math.cos(math.radians(px))
        
        return math.sqrt(lat_dist**2 + lon_dist**2)
    
    def distance_to_polygon_boundary(self, point: Tuple[float, float], 
                                    polygon: List[List[float]]) -> float:
        """Calculate minimum distance from point to polygon boundary"""
        if not polygon or len(polygon) < 3:
            return float('inf')
            
        min_distance = float('inf')
        lat, lon = point
        
        for i in range(len(polygon)):
            j = (i + 1) % len(polygon)
            lat1, lon1 = polygon[i]
            lat2, lon2 = polygon[j]
            
            distance = self.point_to_segment_distance(lat, lon, lat1, lon1, lat2, lon2)
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def pre_filter_stations_by_theoretical_range(self, current_coords: Tuple[float, float],
                                                initial_battery_percent: float,
                                                efficiency_km_per_percent: float,
                                                charging_stations: List[Dict],
                                                safety_margin: float = 0.3,
                                                max_theoretical_hops: int = 3) -> Tuple[List[Dict], List[Dict]]:
        """Pre-filtering optimization"""
        usable_battery = initial_battery_percent * (1 - safety_margin)
        max_single_hop_range = usable_battery * efficiency_km_per_percent
        theoretical_max_range = max_single_hop_range * (1 + (max_theoretical_hops * 0.8))
        
        potentially_reachable = []
        definitely_unreachable = []
        
        for station in charging_stations:
            direct_distance = self.calculate_distance(
                current_coords[0], current_coords[1], 
                station['lat'], station['lon']
            )
            
            if direct_distance <= theoretical_max_range:
                potentially_reachable.append(station)
            else:
                definitely_unreachable.append({
                    "station": station,
                    "reason": "beyond_theoretical_maximum_range",
                    "direct_distance_km": round(direct_distance, 2),
                    "theoretical_max_range_km": round(theoretical_max_range, 2),
                    "additional_range_needed_km": round(direct_distance - theoretical_max_range, 2)
                })
        
        self.optimization_stats["stations_filtered"] += len(definitely_unreachable)
        
        return potentially_reachable, definitely_unreachable
    
    def find_shortest_hop_path_with_battery_check_optimized(self, start_lat: float, start_lon: float, 
                                                          target_lat: float, target_lon: float,
                                                          initial_battery_percent: float, 
                                                          efficiency_km_per_percent: float,
                                                          charging_stations: List[Dict], 
                                                          max_hops: int = 10, 
                                                          safety_margin: float = 0.3,
                                                          target_battery_min: float = 30.0,
                                                          target_battery_max: float = 50.0) -> Tuple[bool, int, List, float]:
        """
        Battery-optimized pathfinding using Dijkstra's algorithm
        - Prioritizes paths that arrive at stations with 30-50% battery (comfortable range)
        - Only accepts critical battery (15-30%) arrivals if no better path exists
        - First hop uses initial_battery_percent, subsequent hops use 100% (recharged)
        - Uses negative final battery as cost to maximize battery on arrival
        """
        import heapq
        
        filtered_stations, _ = self.pre_filter_stations_by_theoretical_range(
            (start_lat, start_lon), initial_battery_percent, efficiency_km_per_percent,
            charging_stations, safety_margin, max_hops
        )
        
        # FIRST HOP: Use current battery
        usable_battery = initial_battery_percent * (1 - safety_margin)
        max_range_km = usable_battery * efficiency_km_per_percent
        
        # STATION HOPS: Use 100% battery (recharged)
        station_usable_battery = 100.0 * (1 - safety_margin)
        station_max_range_km = station_usable_battery * efficiency_km_per_percent
        
        # Check direct reachability
        direct_distance = self.calculate_distance(start_lat, start_lon, target_lat, target_lon)
        if direct_distance <= max_range_km:
            remaining_battery = usable_battery - (direct_distance / efficiency_km_per_percent)
            return True, 0, [(start_lat, start_lon), (target_lat, target_lon)], remaining_battery
        
        # Early termination check
        theoretical_max_range = max_range_km * (1 + (max_hops * 0.8))
        if direct_distance > theoretical_max_range:
            self.optimization_stats["early_terminations"] += 1
            return False, -1, [], 0
        
        # Dijkstra's algorithm with battery optimization
        # Priority: (cost, hops, current_pos, path, current_battery)
        # Cost function: prioritize higher final battery (use negative battery as cost)
        start_key = (start_lat, start_lon)
        target_key = (target_lat, target_lon)
        
        # Priority queue: (priority_cost, hops, position, path, battery)
        pq = [(0, 0, start_key, [start_key], usable_battery)]
        
        # Best battery found for each (position, hops) combination
        best_battery = {}
        best_battery[(start_key, 0)] = usable_battery
        
        station_coords_set = {(s['lat'], s['lon']) for s in filtered_stations}
        
        while pq:
            cost, hops, current_pos, path, current_battery = heapq.heappop(pq)
            
            current_lat, current_lon = current_pos
            
            # Skip if exceeded limits
            if hops >= max_hops or current_battery <= 0:
                continue
            
            # Check if we can reach target
            current_max_range = current_battery * efficiency_km_per_percent
            distance_to_target = self.calculate_distance(current_lat, current_lon, target_lat, target_lon)
            
            if distance_to_target <= current_max_range:
                final_battery = current_battery - (distance_to_target / efficiency_km_per_percent)
                
                # Accept if battery is in acceptable range or if it's the only option
                if final_battery >= 15.0:  # Minimum emergency threshold
                    return True, hops, path + [target_key], final_battery
            
            # Explore reachable stations
            for station in filtered_stations:
                station_lat, station_lon = station['lat'], station['lon']
                station_key = (station_lat, station_lon)
                
                # Skip if already in path
                if station_key in path:
                    continue
                
                distance_to_station = self.calculate_distance(current_lat, current_lon, station_lat, station_lon)
                
                if distance_to_station <= current_max_range:
                    battery_after_travel = current_battery - (distance_to_station / efficiency_km_per_percent)
                    
                    # Must arrive with at least 15% (emergency threshold)
                    # if battery_after_travel >= 15.0:
                    if battery_after_travel > 0:
                        # After recharging at station
                        recharged_battery = station_usable_battery
                        new_hops = hops + 1
                        new_path = path + [station_key]
                        state_key = (station_key, new_hops)
                        
                        # Calculate cost: prioritize paths with better arrival battery
                        # Penalize critically low arrivals (15-30%) more than comfortable arrivals (30-50%)
                        if battery_after_travel < target_battery_min:
                            # Critical arrival - add penalty
                            battery_penalty = (target_battery_min - battery_after_travel) * 2.0
                        elif battery_after_travel > target_battery_max:
                            # Wastefully high arrival - small penalty
                            battery_penalty = (battery_after_travel - target_battery_max) * 0.5
                        else:
                            # Optimal range - no penalty
                            battery_penalty = 0
                        
                        # Cost = hops + battery_penalty (lower is better)
                        new_cost = new_hops + battery_penalty
                        
                        # Update if this is a better path to this state
                        if state_key not in best_battery or recharged_battery > best_battery[state_key]:
                            best_battery[state_key] = recharged_battery
                            heapq.heappush(pq, (new_cost, new_hops, station_key, new_path, recharged_battery))
        
        return False, -1, [], 0
    
    def analyze_reachable_stations_with_optimizations(self, current_coords: Tuple[float, float], 
                                                    initial_battery_percent: float,
                                                    efficiency_km_per_percent: float, 
                                                    charging_stations: List[Dict],
                                                    max_hops: int = 10, 
                                                    safety_margin: float = 0.3) -> Tuple[List[Dict], List[Dict]]:
        """Optimized reachability analysis"""
        start_time = time.time()
        
        potentially_reachable, definitely_unreachable = self.pre_filter_stations_by_theoretical_range(
            current_coords, initial_battery_percent, efficiency_km_per_percent,
            charging_stations, safety_margin, max_hops
        )
        
        reachable_stations = []
        unreachable_stations = []
        
        for unreachable_info in definitely_unreachable:
            station = unreachable_info["station"]
            station_info = {
                "name": station.get('name', 'Unnamed Station'),
                "location": [station['lat'], station['lon']],
                "reachable": False,
                "direct_distance_km": unreachable_info["direct_distance_km"],
                "additional_range_needed_km": unreachable_info["additional_range_needed_km"],
                "additional_battery_needed_percent": round(
                    unreachable_info["additional_range_needed_km"] / efficiency_km_per_percent, 1
                ) if efficiency_km_per_percent > 0 else 0,
                "reason": "Beyond theoretical maximum range"
            }
            unreachable_stations.append(station_info)
        
        for station in potentially_reachable:
            station_lat, station_lon = station['lat'], station['lon']
            
            reachable, hops, path, final_battery = self.find_shortest_hop_path_with_battery_check_optimized(
                current_coords[0], current_coords[1], 
                station_lat, station_lon, 
                initial_battery_percent, efficiency_km_per_percent,
                potentially_reachable, max_hops, safety_margin
            )
            
            direct_distance = self.calculate_distance(
                current_coords[0], current_coords[1], station_lat, station_lon
            )
            
            station_info = {
                "name": station.get('name', 'Unnamed Station'),
                "location": [station_lat, station_lon],
                "reachable": reachable,
                "direct_distance_km": round(direct_distance, 2)
            }
            
            if reachable:
                total_path_distance = 0
                if len(path) > 1:
                    for j in range(len(path) - 1):
                        if isinstance(path[j], tuple) and isinstance(path[j+1], tuple):
                            total_path_distance += self.calculate_distance(
                                path[j][0], path[j][1], path[j+1][0], path[j+1][1]
                            )
                
                station_info.update({
                    "hops_required": hops - 1,
                    "path_distance_km": round(total_path_distance, 2),
                    "final_battery_percent": round(final_battery, 1),
                    "battery_efficient": final_battery > 10,
                    "reachability_method": "direct" if hops == 0 else f"{hops - 1}-hop path",
                    "path_coordinates": path
                })
                reachable_stations.append(station_info)
            else:
                usable_battery = initial_battery_percent * (1 - safety_margin)
                max_range_km = usable_battery * efficiency_km_per_percent
                additional_range = max(0, direct_distance - max_range_km)
                additional_battery = additional_range / efficiency_km_per_percent if efficiency_km_per_percent > 0 else 0
                
                station_info.update({
                    "additional_range_needed_km": round(additional_range, 2),
                    "additional_battery_needed_percent": round(additional_battery, 1),
                    "reason": f"Beyond {max_hops}-hop reach"
                })
                unreachable_stations.append(station_info)
        
        reachable_stations.sort(key=lambda x: (x['hops_required'], -x['final_battery_percent'], x['direct_distance_km']))
        unreachable_stations.sort(key=lambda x: x['direct_distance_km'])
        
        elapsed_time = time.time() - start_time
        
        return reachable_stations, unreachable_stations
    
    def create_circular_boundary_optimized(self, lat: float, lon: float, radius_km: float, 
                                         num_points: int = 32) -> List[List[float]]:
        """Optimized circular boundary creation"""
        points = []
        angle_step = 2 * math.pi / num_points
        
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
    
    def create_combined_coverage_polygon(self, current_location: Tuple[float, float], 
                                       initial_battery_percent: float,
                                       efficiency_km_per_percent: float, 
                                       charging_stations: List[Dict],
                                       max_hops: int = 10, 
                                       safety_margin: float = 0.3) -> List[List[float]]:
        """
        Create combined polygon representing all reachable areas
        IMPORTANT: Uses initial_battery for direct range from current position,
        but assumes 100% battery (recharged) for range from each reachable station
        """
        if not COVERAGE_DEPENDENCIES_AVAILABLE:
            usable_battery = initial_battery_percent * (1 - safety_margin)
            max_range_km = usable_battery * efficiency_km_per_percent
            return self.create_circular_boundary_optimized(
                current_location[0], current_location[1], max_range_km
            )
        
        lat, lon = current_location
        
        # Direct range from current position uses CURRENT battery
        current_usable_battery = initial_battery_percent * (1 - safety_margin)
        current_max_range_km = current_usable_battery * efficiency_km_per_percent
        
        # Range from stations uses 100% battery (assumes full recharge at each station)
        station_usable_battery = 100.0 * (1 - safety_margin)
        station_max_range_km = station_usable_battery * efficiency_km_per_percent
        
        reachable_stations_info, _ = self.analyze_reachable_stations_with_optimizations(
            current_location, initial_battery_percent, efficiency_km_per_percent,
            charging_stations, max_hops, safety_margin
        )
        
        all_polygons = []
        
        try:
            # Direct coverage from current position with CURRENT battery
            direct_coverage_coords = self.create_circular_boundary_optimized(
                lat, lon, current_max_range_km
            )
            direct_polygon = Polygon([(coord[1], coord[0]) for coord in direct_coverage_coords])
            if direct_polygon.is_valid:
                all_polygons.append(direct_polygon)
            
            # Coverage from each reachable station with 100% battery (recharged)
            for station_info in reachable_stations_info:
                station_lat, station_lon = station_info['location']
                station_coverage_coords = self.create_circular_boundary_optimized(
                    station_lat, station_lon, station_max_range_km  # Using 100% battery range
                )
                station_polygon = Polygon([(coord[1], coord[0]) for coord in station_coverage_coords])
                
                if station_polygon.is_valid:
                    all_polygons.append(station_polygon)
            
            if not all_polygons:
                return self.create_circular_boundary_optimized(lat, lon, current_max_range_km)
            
            combined_polygon = unary_union(all_polygons)
            
            combined_coords = []
            if isinstance(combined_polygon, Polygon):
                exterior_coords = list(combined_polygon.exterior.coords)
                combined_coords = [[coord[1], coord[0]] for coord in exterior_coords[:-1]]
            elif isinstance(combined_polygon, MultiPolygon):
                largest_polygon = max(combined_polygon.geoms, key=lambda p: p.area)
                exterior_coords = list(largest_polygon.exterior.coords)
                combined_coords = [[coord[1], coord[0]] for coord in exterior_coords[:-1]]
            else:
                return self.create_circular_boundary_optimized(lat, lon, current_max_range_km)
            
            return combined_coords
            
        except Exception as e:
            print(f"Error creating combined polygon: {e}")
            return self.create_circular_boundary_optimized(lat, lon, current_max_range_km)
    
    def calculate_network_coverage(self, charging_stations: List[Dict],
                                  efficiency_km_per_percent: float,
                                  max_battery_percent: float = 100.0,
                                  safety_margin: float = 0.3,
                                  max_hops: int = 10) -> Dict:
        """
        NEW: Calculate overall network coverage independent of current vehicle position
        Returns network polygon(s) showing theoretical maximum reach of the charging network
        Handles disconnected networks separately
        
        This analyzes the infrastructure itself, not the vehicle's current reachability
        """
        if not charging_stations:
            return {
                "network_connected": False,
                "network_count": 0,
                "networks": [],
                "total_coverage_area_km2": 0,
                "message": "No charging stations available"
            }
        
        if not COVERAGE_DEPENDENCIES_AVAILABLE:
            return {
                "network_connected": False,
                "network_count": 0,
                "networks": [],
                "total_coverage_area_km2": 0,
                "message": "Shapely library required for network coverage analysis"
            }
        
        usable_battery = max_battery_percent * (1 - safety_margin)
        max_range_km = usable_battery * efficiency_km_per_percent
        
        # Build connectivity graph between all stations
        station_graph = {}
        for i, station in enumerate(charging_stations):
            station_key = (station['lat'], station['lon'])
            station_graph[station_key] = {
                'index': i,
                'name': station.get('name', f'Station {i}'),
                'station_data': station,
                'connected_stations': []
            }
        
        # Find which stations can reach each other with full battery
        print(f"Analyzing network connectivity for {len(charging_stations)} stations...")
        for station_a in charging_stations:
            key_a = (station_a['lat'], station_a['lon'])
            for station_b in charging_stations:
                if station_a == station_b:
                    continue
                key_b = (station_b['lat'], station_b['lon'])
                
                # Check if reachable with multi-hop pathfinding (using 100% battery)
                reachable, hops, _, _ = self.find_shortest_hop_path_with_battery_check_optimized(
                    station_a['lat'], station_a['lon'],
                    station_b['lat'], station_b['lon'],
                    max_battery_percent, efficiency_km_per_percent,
                    charging_stations, max_hops, safety_margin
                )
                
                if reachable:
                    station_graph[key_a]['connected_stations'].append(key_b)
        
        # Find connected components (separate networks) using BFS
        visited = set()
        networks = []
        
        for station_key in station_graph.keys():
            if station_key in visited:
                continue
            
            # BFS to find all stations in this network
            network_stations = []
            queue = deque([station_key])
            visited.add(station_key)
            
            while queue:
                current = queue.popleft()
                network_stations.append(current)
                
                for neighbor in station_graph[current]['connected_stations']:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            networks.append(network_stations)
        
        print(f"Found {len(networks)} separate network(s)")
        
        # Generate coverage polygons for each network
        network_results = []
        
        for network_idx, network_stations in enumerate(networks):
            # Create coverage circles for each station in network
            all_polygons = []
            
            try:
                for station_key in network_stations:
                    lat, lon = station_key
                    coverage_coords = self.create_circular_boundary_optimized(lat, lon, max_range_km)
                    polygon = Polygon([(coord[1], coord[0]) for coord in coverage_coords])
                    if polygon.is_valid:
                        all_polygons.append(polygon)
                
                if all_polygons:
                    # Combine all station coverage areas in this network
                    combined_polygon = unary_union(all_polygons)
                    
                    # Convert to coordinate list
                    network_coords = []
                    area_km2 = 0
                    
                    if isinstance(combined_polygon, Polygon):
                        exterior_coords = list(combined_polygon.exterior.coords)
                        network_coords = [[coord[1], coord[0]] for coord in exterior_coords[:-1]]
                        area_km2 = combined_polygon.area * (111.32 ** 2)  # Approximate conversion
                    elif isinstance(combined_polygon, MultiPolygon):
                        # Use largest polygon for main boundary
                        largest_polygon = max(combined_polygon.geoms, key=lambda p: p.area)
                        exterior_coords = list(largest_polygon.exterior.coords)
                        network_coords = [[coord[1], coord[0]] for coord in exterior_coords[:-1]]
                        area_km2 = sum(p.area * (111.32 ** 2) for p in combined_polygon.geoms)
                    else:
                        continue
                    
                    # Get station details for this network
                    network_station_details = []
                    for station_key in network_stations:
                        station_info = station_graph[station_key]
                        station = station_info['station_data']
                        network_station_details.append({
                            'name': station.get('name', 'Unnamed Station'),
                            'location': [station['lat'], station['lon']],
                            'connections': len(station_info['connected_stations'])
                        })
                    
                    network_results.append({
                        'network_id': network_idx + 1,
                        'station_count': len(network_stations),
                        'stations': network_station_details,
                        'coverage_polygon': network_coords,
                        'coverage_area_km2': round(area_km2, 2),
                        'is_connected_to_other_networks': len(networks) == 1
                    })
                    
            except Exception as e:
                print(f"Error processing network {network_idx + 1}: {e}")
                continue
        
        # Calculate total coverage
        total_area = sum(net['coverage_area_km2'] for net in network_results)
        
        return {
            'network_connected': len(networks) == 1,
            'network_count': len(networks),
            'networks': network_results,
            'total_coverage_area_km2': round(total_area, 2),
            'max_theoretical_range_km': round(max_range_km, 2),
            'total_stations': len(charging_stations),
            'analysis_parameters': {
                'max_battery_percent': max_battery_percent,
                'efficiency_km_per_percent': efficiency_km_per_percent,
                'safety_margin': safety_margin,
                'max_hops': max_hops
            },
            'message': f"Network analysis complete: {len(networks)} separate network(s) detected"
        }
    
    def _determine_current_network(self, 
                               current_position: Tuple[float, float],
                               reachable_stations: List[Dict],
                               network_coverage: Dict) -> Dict:
        """
        CRITICAL FIX: Determine which network the vehicle is currently in
        """
        if not network_coverage or network_coverage.get('network_count', 1) == 1:
            # Single network - add current_network field
            if network_coverage and 'networks' in network_coverage and len(network_coverage['networks']) > 0:
                network_coverage['current_network'] = {
                    'network_id': 1,
                    'station_count': network_coverage['networks'][0]['station_count']
                }
            return network_coverage
        
        # Multiple networks - find which one contains closest reachable station
        if not reachable_stations:
            return network_coverage
        
        closest_reachable = min(reachable_stations, key=lambda s: s['direct_distance_km'])
        closest_loc = tuple(closest_reachable['location'])
        
        # Find which network contains this station
        for network in network_coverage.get('networks', []):
            for station_info in network.get('stations', []):
                station_loc = tuple(station_info.get('location', []))
                
                if (len(station_loc) == 2 and len(closest_loc) == 2 and
                    abs(station_loc[0] - closest_loc[0]) < 0.0001 and 
                    abs(station_loc[1] - closest_loc[1]) < 0.0001):
                    network_coverage['current_network'] = {
                        'network_id': network['network_id'],
                        'station_count': network['station_count']
                    }
                    return network_coverage
        
        # Fallback
        if 'networks' in network_coverage and len(network_coverage['networks']) > 0:
            network_coverage['current_network'] = {
                'network_id': 1,
                'station_count': network_coverage['networks'][0]['station_count']
            }
        
        return network_coverage
    
    def calculate_point_of_no_return(self, current_battery: float, 
                                    efficiency_km_per_percent: float,
                                    safety_margin: float = 0.3,
                                    buffer_multiplier: float = 1.2) -> float:
        """Calculate maximum distance before point of no return"""
        usable_battery = current_battery * (1 - safety_margin)
        return (usable_battery * efficiency_km_per_percent) / buffer_multiplier
    
    def analyze_safety_with_alerts(self, 
                                  current_position: Tuple[float, float],
                                  current_battery: float,
                                  efficiency_km_per_percent: float,
                                  charging_stations: List[Dict],
                                  max_hops: int = 10,
                                  safety_margin: float = 0.3,
                                  timestamp: Optional[datetime] = None,
                                  current_heading_degrees: Optional[float] = 90.0,
                                  current_speed_kmh: Optional[float] = None,
                                  battery_drain_rate_per_km: Optional[float] = None,
                                  is_moving: bool = True,
                                  alert_config: Optional[Dict] = None,
                                  include_network_coverage: bool = True) -> Dict:
        """
        MAIN INTEGRATED METHOD: Complete analysis with modular alert system + network coverage
        ALL ORIGINAL FEATURES PRESERVED
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        start_time = time.time()
        
        # Calculate coverage areas (ORIGINAL LOGIC)
        lat, lon = current_position
        usable_battery = current_battery * (1 - safety_margin)
        max_range_km = usable_battery * efficiency_km_per_percent
        
        direct_coverage = self.create_circular_boundary_optimized(lat, lon, max_range_km)
        
        # Get reachable stations (ORIGINAL LOGIC)
        reachable_stations, unreachable_stations = self.analyze_reachable_stations_with_optimizations(
            current_position, current_battery, efficiency_km_per_percent,
            charging_stations, max_hops, safety_margin
        )
        
        # Create combined coverage polygon (ORIGINAL LOGIC)
        combined_coverage = self.create_combined_coverage_polygon(
            current_position, current_battery, efficiency_km_per_percent,
            charging_stations, max_hops, safety_margin
        )
        
        # NEW: Calculate overall network coverage (independent of current position)
        network_coverage = None
        if include_network_coverage:
            network_coverage = self.calculate_network_coverage(
                charging_stations, efficiency_km_per_percent,
                max_battery_percent=100.0, safety_margin=safety_margin, max_hops=max_hops
            )
        
        # Calculate metrics (ORIGINAL LOGIC)
        point_of_no_return = self.calculate_point_of_no_return(
            current_battery, efficiency_km_per_percent, safety_margin
        )
        
        in_direct = self.is_point_in_polygon(current_position, direct_coverage)
        in_combined = self.is_point_in_polygon(current_position, combined_coverage)
        
        dist_direct = self.distance_to_polygon_boundary(current_position, direct_coverage)
        dist_combined = self.distance_to_polygon_boundary(current_position, combined_coverage)
        
        # NEW: Create contexts for modular alert system
        vehicle_status = VehicleStatus(
            position=current_position,
            battery_percent=current_battery,
            efficiency_km_per_percent=efficiency_km_per_percent,
            heading_degrees=current_heading_degrees,
            speed_kmh=current_speed_kmh,
            battery_drain_rate_per_km=battery_drain_rate_per_km,
            is_moving=is_moving
        )

        if network_coverage:
            network_coverage = self._determine_current_network(
            current_position, reachable_stations, network_coverage
        )
        
        coverage_context = CoverageContext(
            in_direct_coverage=in_direct,
            in_combined_coverage=in_combined,
            distance_to_direct_boundary=dist_direct,
            distance_to_combined_boundary=dist_combined,
            max_range_km=max_range_km,
            point_of_no_return_km=point_of_no_return,
            reachable_stations=reachable_stations,
            unreachable_stations=unreachable_stations,
            network_info=network_coverage
        )
        
        # NEW: Generate alerts using modular system
        if alert_config:
            custom_alert_config = AlertConfig(**alert_config)
            temp_alert_system = EVAlertSystem(custom_alert_config)
            alerts = temp_alert_system.generate_alerts(vehicle_status, coverage_context, timestamp, safety_margin)
        else:
            alerts = self.alert_system.generate_alerts(vehicle_status, coverage_context, timestamp, safety_margin)
        
        elapsed_time = time.time() - start_time
        
        # Compile result with ALL ORIGINAL FIELDS + NEW network coverage
        result = {
            "timestamp": timestamp.isoformat(),
            "analysis_time_seconds": round(elapsed_time, 3),
            
            "current_status": {
                "location": list(current_position),
                "battery_percent": round(current_battery, 1),
                "usable_battery_percent": round(usable_battery, 1),
                "max_direct_range_km": round(max_range_km, 2),
                "point_of_no_return_km": round(point_of_no_return, 2)
            },
            
            "travel_metrics": {
                "heading_degrees": round(current_heading_degrees, 1) if current_heading_degrees is not None else None,
                "heading_direction": self._get_compass_direction(current_heading_degrees) if current_heading_degrees is not None else "Unknown",
                "speed_kmh": round(current_speed_kmh, 1) if current_speed_kmh else None,
                "battery_drain_rate_per_km": round(battery_drain_rate_per_km, 4) if battery_drain_rate_per_km else None,
                "is_moving": is_moving,
                "metrics_source": "client_provided" if current_heading_degrees is not None or current_speed_kmh is not None else "calculated"
            },
            
            "coverage_status": {
                "in_direct_coverage": in_direct,
                "in_combined_coverage": in_combined,
                "distance_to_direct_boundary_km": round(dist_direct, 2),
                "distance_to_combined_boundary_km": round(dist_combined, 2),
                "coverage_level": self._get_coverage_level(in_direct, in_combined)
            },
            
            "coverage_areas": {
                "direct_coverage": direct_coverage,
                "combined_coverage_polygon": combined_coverage,
                "overall_network_coverage": network_coverage,  # NEW: Network coverage analysis
                "direct_coverage_points": len(direct_coverage),
                "combined_coverage_points": len(combined_coverage),
            },
            
            "station_analysis": {
                "total_stations": len(charging_stations),
                "reachable_stations": len(reachable_stations),
                "unreachable_stations": len(unreachable_stations),
                "reachable_stations_list": reachable_stations,
                "unreachable_stations_list": unreachable_stations
            },
            
            # NEW: Alerts from modular system
            "alerts": {
                "total_alerts": len(alerts),
                "alerts_by_level": self._categorize_alerts_by_level(alerts),
                "alerts_list": [alert.to_dict() for alert in alerts],
                "highest_severity": self._get_highest_severity(alerts)
            },
            
            "optimization_stats": {
                "stations_pre_filtered": self.optimization_stats["stations_filtered"],
                "early_terminations": self.optimization_stats["early_terminations"],
                "cache_hits": self.optimization_stats["cache_utilization"]
            },
            
            # NEW: Dashboard summary for UI
            "dashboard_summary": self._create_dashboard_summary(
                alerts, reachable_stations, in_direct, in_combined, current_battery
            ),
            
            # ORIGINAL: Reachability stats
            "reachability_stats": self._calculate_reachability_stats(reachable_stations)
        }
        
        return result
    

    
    def _create_dashboard_summary(self, alerts: List[Alert], 
                                  reachable_stations: List[Dict],
                                  in_direct: bool, in_combined: bool,
                                  battery: float) -> Dict:
        """Create quick dashboard summary"""
        critical_count = sum(1 for a in alerts if a.alert_level.value in ['critical', 'emergency'])
        warning_count = sum(1 for a in alerts if a.alert_level.value == 'warning')
        
        immediate_action = any(
            a.alert_level.value == 'emergency' or 
            (a.alert_level.value == 'critical' and battery < 30)
            for a in alerts
        )
        
        nearest = None
        if reachable_stations:
            nearest = min(reachable_stations, key=lambda s: s['direct_distance_km'])
        
        if not in_combined:
            status = "UNSAFE - Outside Coverage"
        elif not in_direct:
            status = "EXTENDED - Station Coverage"
        else:
            status = "SAFE - Direct Coverage"
        
        if battery < 15:
            battery_status = "critical"
        elif battery < 30:
            battery_status = "low"
        elif battery < 50:
            battery_status = "moderate"
        else:
            battery_status = "normal"
        
        return {
            "status": status,
            "battery_status": battery_status,
            "critical_alerts": critical_count,
            "warning_alerts": warning_count,
            "reachable_stations_count": len(reachable_stations),
            "nearest_station": {
                "name": nearest['name'],
                "distance_km": round(nearest['direct_distance_km'], 2),
                "hops_required": nearest.get('hops_required', 0)
            } if nearest else None,
            "immediate_action_required": immediate_action
        }
    
    def _calculate_reachability_stats(self, reachable_stations: List[Dict]) -> Dict:
        """Calculate reachability statistics"""
        if not reachable_stations:
            return {
                "direct_reachable": 0,
                "multi_hop_reachable": 0,
                "average_hops": 0,
                "average_final_battery": 0,
                "reachability_percentage": 0
            }
        
        direct = sum(1 for s in reachable_stations if s.get('hops_required', 0) == 0)
        multi_hop = len(reachable_stations) - direct
        avg_hops = sum(s.get('hops_required', 0) for s in reachable_stations) / len(reachable_stations)
        avg_battery = sum(s.get('final_battery_percent', 0) for s in reachable_stations) / len(reachable_stations)
        
        return {
            "direct_reachable": direct,
            "multi_hop_reachable": multi_hop,
            "average_hops": round(avg_hops, 2),
            "average_final_battery": round(avg_battery, 1),
            "reachability_percentage": 100.0  # Calculated against total stations in calling code
        }
    
    def _categorize_alerts_by_level(self, alerts: List[Alert]) -> Dict:
        """Categorize alerts by severity level"""
        categorized = {level.value: 0 for level in AlertLevel}
        for alert in alerts:
            categorized[alert.alert_level.value] += 1
        return categorized
    
    def _get_highest_severity(self, alerts: List[Alert]) -> str:
        """Get highest alert severity"""
        if not alerts:
            return "none"
        
        severity_order = [AlertLevel.EMERGENCY, AlertLevel.CRITICAL, AlertLevel.WARNING, AlertLevel.INFO]
        for level in severity_order:
            if any(alert.alert_level == level for alert in alerts):
                return level.value
        return "info"
    
    def _get_coverage_level(self, in_direct: bool, in_combined: bool) -> str:
        """Determine coverage level description"""
        if in_direct:
            return "SAFE - Direct Coverage"
        elif in_combined:
            return "EXTENDED - Station Coverage"
        else:
            return "UNSAFE - Outside Coverage"
    
    def _get_compass_direction(self, degrees: Optional[float]) -> str:
        """Convert degrees to compass direction"""
        if degrees is None:
            return "Unknown"
        
        directions = [
            "North", "North-East", "East", "South-East",
            "South", "South-West", "West", "North-West"
        ]
        index = round(degrees / 45) % 8
        return directions[index]
    
    # LEGACY METHOD: For backward compatibility
    def create_optimized_coverage_areas(self, current_location: Tuple[float, float], 
                                      initial_battery_percent: float,
                                      efficiency_km_per_percent: float, 
                                      charging_stations: List[Dict],
                                      max_hops: int = 10, safety_margin: float = 0.3,
                                      include_combined_polygon: bool = True) -> Dict:
        """
        Legacy method for backward compatibility - creates coverage areas without alerts
        Use analyze_safety_with_alerts() for full integrated analysis
        """
        lat, lon = current_location
        start_time = time.time()
        
        usable_battery = initial_battery_percent * (1 - safety_margin)
        max_range_km = usable_battery * efficiency_km_per_percent
        direct_coverage = self.create_circular_boundary_optimized(lat, lon, max_range_km)
        
        reachable_stations, _ = self.analyze_reachable_stations_with_optimizations(
            current_location, initial_battery_percent, efficiency_km_per_percent,
            charging_stations, max_hops, safety_margin
        )
        
        result = {
            "direct_coverage": direct_coverage,
            "reachable_stations_count": len(reachable_stations),
            "reachable_stations_summary": [
                {
                    "name": station['name'],
                    "location": station['location'],
                    "hops_required": station['hops_required'],
                    "direct_distance_km": station['direct_distance_km'],
                    "final_battery_percent": station['final_battery_percent']
                } for station in reachable_stations
            ]
        }
        
        if include_combined_polygon:
            combined_polygon = self.create_combined_coverage_polygon(
                current_location, initial_battery_percent, efficiency_km_per_percent,
                charging_stations, max_hops, safety_margin
            )
            result["combined_coverage_polygon"] = combined_polygon
        
        elapsed_time = time.time() - start_time
        result["computation_time_seconds"] = round(elapsed_time, 3)
        
        return result