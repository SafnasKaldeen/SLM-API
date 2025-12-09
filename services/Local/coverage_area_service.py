"""
Enhanced Coverage Area Service - COMPLETE VERSION
All original features PRESERVED + Modular Alert System integrated

This maintains 100% backward compatibility while adding the modular alert system
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
    Battery-aware coverage area service with ALL original features + modular alerts
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
                low_battery_threshold=50.0,
                critical_battery_threshold=30.0,
                emergency_battery_threshold=15.0,
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
                                                          safety_margin: float = 0.3) -> Tuple[bool, int, List, float]:
        """Optimized pathfinding"""
        filtered_stations, _ = self.pre_filter_stations_by_theoretical_range(
            (start_lat, start_lon), initial_battery_percent, efficiency_km_per_percent,
            charging_stations, safety_margin, max_hops
        )
        
        usable_battery = initial_battery_percent * (1 - safety_margin)
        max_range_km = usable_battery * efficiency_km_per_percent
        
        direct_distance = self.calculate_distance(start_lat, start_lon, target_lat, target_lon)
        if direct_distance <= max_range_km:
            remaining_battery = usable_battery - (direct_distance / efficiency_km_per_percent)
            return True, 0, [(start_lat, start_lon), (target_lat, target_lon)], remaining_battery
        
        theoretical_max_range = max_range_km * (1 + (max_hops * 0.8))
        if direct_distance > theoretical_max_range:
            self.optimization_stats["early_terminations"] += 1
            return False, -1, [], 0
        
        queue = deque([(start_lat, start_lon, 0, [(start_lat, start_lon)], usable_battery)])
        visited_stations = set()
        station_coords = [(s['lat'], s['lon']) for s in filtered_stations]
        
        while queue:
            current_lat, current_lon, hops, path, current_battery = queue.popleft()
            
            if hops >= max_hops or current_battery <= 0:
                continue
            
            current_max_range = current_battery * efficiency_km_per_percent
            distance_to_target = self.calculate_distance(current_lat, current_lon, target_lat, target_lon)
            
            if distance_to_target <= current_max_range:
                final_battery = current_battery - (distance_to_target / efficiency_km_per_percent)
                return True, hops, path + [(target_lat, target_lon)], final_battery
            
            reachable_stations = []
            for station in filtered_stations:
                station_lat, station_lon = station['lat'], station['lon']
                station_key = (station_lat, station_lon)
                
                if station_key in visited_stations or station_key in path:
                    continue
                    
                distance_to_station = self.calculate_distance(current_lat, current_lon, station_lat, station_lon)
                
                if distance_to_station <= current_max_range:
                    battery_after_travel = current_battery - (distance_to_station / efficiency_km_per_percent)
                    if battery_after_travel > 0:
                        reachable_stations.append((distance_to_station, station, station_key, battery_after_travel))
            
            reachable_stations.sort(key=lambda x: x[0])
            
            for distance_to_station, station, station_key, battery_after_travel in reachable_stations:
                recharged_battery = initial_battery_percent * (1 - safety_margin)
                new_path = path + [station_key]
                queue.append((station['lat'], station['lon'], hops + 1, new_path, recharged_battery))
            
            current_key = (current_lat, current_lon)
            if current_key in station_coords:
                visited_stations.add(current_key)
        
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
                    "hops_required": hops,
                    "path_distance_km": round(total_path_distance, 2),
                    "final_battery_percent": round(final_battery, 1),
                    "battery_efficient": final_battery > 10,
                    "reachability_method": "direct" if hops == 0 else f"{hops}-hop path",
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
        """Create combined polygon representing all reachable areas"""
        if not COVERAGE_DEPENDENCIES_AVAILABLE:
            usable_battery = initial_battery_percent * (1 - safety_margin)
            max_range_km = usable_battery * efficiency_km_per_percent
            return self.create_circular_boundary_optimized(
                current_location[0], current_location[1], max_range_km
            )
        
        lat, lon = current_location
        usable_battery = initial_battery_percent * (1 - safety_margin)
        max_range_km = usable_battery * efficiency_km_per_percent
        
        reachable_stations_info, _ = self.analyze_reachable_stations_with_optimizations(
            current_location, initial_battery_percent, efficiency_km_per_percent,
            charging_stations, max_hops, safety_margin
        )
        
        all_polygons = []
        
        try:
            direct_coverage_coords = self.create_circular_boundary_optimized(lat, lon, max_range_km)
            direct_polygon = Polygon([(coord[1], coord[0]) for coord in direct_coverage_coords])
            if direct_polygon.is_valid:
                all_polygons.append(direct_polygon)
            
            for station_info in reachable_stations_info:
                station_lat, station_lon = station_info['location']
                station_coverage_coords = self.create_circular_boundary_optimized(
                    station_lat, station_lon, max_range_km
                )
                station_polygon = Polygon([(coord[1], coord[0]) for coord in station_coverage_coords])
                
                if station_polygon.is_valid:
                    all_polygons.append(station_polygon)
            
            if not all_polygons:
                return self.create_circular_boundary_optimized(lat, lon, max_range_km)
            
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
                return self.create_circular_boundary_optimized(lat, lon, max_range_km)
            
            return combined_coords
            
        except Exception as e:
            print(f"Error creating combined polygon: {e}")
            return self.create_circular_boundary_optimized(lat, lon, max_range_km)
    
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
                                  alert_config: Optional[Dict] = None) -> Dict:
        """
        MAIN INTEGRATED METHOD: Complete analysis with modular alert system
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
        
        coverage_context = CoverageContext(
            in_direct_coverage=in_direct,
            in_combined_coverage=in_combined,
            distance_to_direct_boundary=dist_direct,
            distance_to_combined_boundary=dist_combined,
            max_range_km=max_range_km,
            point_of_no_return_km=point_of_no_return,
            reachable_stations=reachable_stations,
            unreachable_stations=unreachable_stations
        )
        
        # NEW: Generate alerts using modular system
        if alert_config:
            custom_alert_config = AlertConfig(**alert_config)
            temp_alert_system = EVAlertSystem(custom_alert_config)
            alerts = temp_alert_system.generate_alerts(vehicle_status, coverage_context, timestamp, safety_margin)
        else:
            alerts = self.alert_system.generate_alerts(vehicle_status, coverage_context, timestamp, safety_margin)
        
        elapsed_time = time.time() - start_time
        
        # Compile result with ALL ORIGINAL FIELDS
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
                # "network_coverage_polygon": network_coverage,
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


# EXAMPLE USAGE
if __name__ == "__main__":
    print("="*80)
    print("COMPLETE COVERAGE SERVICE WITH MODULAR ALERT SYSTEM")
    print("All Original Features + Enhanced Alert Generation")
    print("="*80)
    
    # Initialize service with custom alert configuration
    custom_config = AlertConfig(
        warning_distance_km=30.0,
        critical_distance_km=15.0,
        low_battery_threshold=50.0,
        critical_battery_threshold=30.0,
        emergency_battery_threshold=15.0
    )
    
    service = CoverageAreaService(alert_config=custom_config)
    
    # Sample stations (your actual data)
    sample_stations = [
        {"lat": 6.84179341, "lon": 80.10289527, "name": "Padukka Station"},
        {"lat": 6.53511873, "lon": 80.15020067, "name": "Agalawaththa"},
        {"lat": 7.07287165, "lon": 80.01556976, "name": "Miriswatta Station"},
        {"lat": 7.48035822, "lon": 80.35044315, "name": "Kurunagala Station"},
    ]
    
    # Test Case 1: CHARGING DESERT SCENARIO (your actual issue)
    print("\n" + "="*80)
    print("TEST CASE 1: CHARGING DESERT - Your Actual Scenario")
    print("="*80)
    
    result = service.analyze_safety_with_alerts(
        current_position=(6.9032, 80.5966),  # Your location
        current_battery=80.0,
        efficiency_km_per_percent=0.7,  # 70% efficiency
        charging_stations=sample_stations,
        max_hops=10,
        safety_margin=0.35,
        current_heading_degrees=90.0,  # Heading East
        current_speed_kmh=50.0,
        is_moving=True
    )
    
    print(f"\nLocation: {result['current_status']['location']}")
    print(f"Battery: {result['current_status']['battery_percent']}%")
    print(f"Max Range: {result['current_status']['max_direct_range_km']} km")
    print(f"Status: {result['coverage_status']['coverage_level']}")
    print(f"\nStations:")
    print(f"  Reachable: {result['station_analysis']['reachable_stations']}")
    print(f"  Unreachable: {result['station_analysis']['unreachable_stations']}")
    
    print(f"\n{'='*80}")
    print(f"ALERTS GENERATED: {result['alerts']['total_alerts']}")
    print(f"{'='*80}")
    print(f"Emergency: {result['alerts']['alerts_by_level']['emergency']}")
    print(f"Critical: {result['alerts']['alerts_by_level']['critical']}")
    print(f"Warning: {result['alerts']['alerts_by_level']['warning']}")
    print(f"Info: {result['alerts']['alerts_by_level']['info']}")
    print(f"Highest Severity: {result['alerts']['highest_severity'].upper()}")
    
    print(f"\n{'='*80}")
    print("ALERT DETAILS:")
    print(f"{'='*80}")
    for alert_dict in result['alerts']['alerts_list']:
        print(f"\n[{alert_dict['alert_level'].upper()}] {alert_dict['title']}")
        print(f"Message: {alert_dict['message']}")
        print(f"Action: {alert_dict['recommended_action']}")
        if 'metadata' in alert_dict and alert_dict['metadata']:
            print(f"Metadata: {alert_dict['metadata']}")
    
    print(f"\n{'='*80}")
    print("DASHBOARD SUMMARY (for UI):")
    print(f"{'='*80}")
    summary = result['dashboard_summary']
    print(f"Status: {summary['status']}")
    print(f"Battery Status: {summary['battery_status']}")
    print(f"Critical Alerts: {summary['critical_alerts']}")
    print(f"Warning Alerts: {summary['warning_alerts']}")
    print(f"Immediate Action Required: {summary['immediate_action_required']}")
    print(f"Reachable Stations: {summary['reachable_stations_count']}")
    print(f"Nearest Station: {summary['nearest_station']}")
    
    # Test Case 2: NORMAL SCENARIO (should show minimal alerts)
    print("\n\n" + "="*80)
    print("TEST CASE 2: NORMAL SCENARIO - Good Battery, Stations Available")
    print("="*80)
    
    result2 = service.analyze_safety_with_alerts(
        current_position=(6.9271, 79.8612),  # Colombo
        current_battery=75.0,
        efficiency_km_per_percent=0.7,
        charging_stations=sample_stations,
        max_hops=10,
        safety_margin=0.35,
        current_heading_degrees=45.0,
        current_speed_kmh=60.0,
        is_moving=True
    )
    
    print(f"\nLocation: {result2['current_status']['location']}")
    print(f"Battery: {result2['current_status']['battery_percent']}%")
    print(f"Status: {result2['coverage_status']['coverage_level']}")
    print(f"Reachable Stations: {result2['station_analysis']['reachable_stations']}")
    print(f"\nAlerts: {result2['alerts']['total_alerts']} (Severity: {result2['alerts']['highest_severity']})")
    
    # Test Case 3: LOW BATTERY SCENARIO
    print("\n\n" + "="*80)
    print("TEST CASE 3: LOW BATTERY - Should trigger battery warnings")
    print("="*80)
    
    result3 = service.analyze_safety_with_alerts(
        current_position=(6.9271, 79.8612),
        current_battery=25.0,  # Low battery
        efficiency_km_per_percent=0.7,
        charging_stations=sample_stations,
        max_hops=10,
        safety_margin=0.35,
        current_heading_degrees=90.0,
        is_moving=True
    )
    
    print(f"\nBattery: {result3['current_status']['battery_percent']}%")
    print(f"Alerts: {result3['alerts']['total_alerts']} (Severity: {result3['alerts']['highest_severity']})")
    print(f"Immediate Action: {result3['dashboard_summary']['immediate_action_required']}")
    
    print("\n" + "="*80)
    print("COMPLETE SERVICE READY - ALL FEATURES PRESERVED + MODULAR ALERTS")
    print("="*80)