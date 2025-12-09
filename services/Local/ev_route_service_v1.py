# services/Local/ev_route_service_v1.py
"""
EV Route Service V1 - Strategic Battery Utilization Strategy
- First station: Maximize distance towards destination + battery utilization
- Final station: Closest to destination for maximum arrival battery
- Middle stations: Maximum battery utilization
"""

import heapq
import json
import os
import time
import hashlib
import pickle
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass
import requests
from geopy.distance import geodesic

# --- Configuration and Constants ---
STATION_MAPPING = {
    "(7.123456,80.123456)": "Miriswaththa_Station",
    "(7.182689,79.961171)": "Minuwangoda_Station", 
    "(7.148497,79.873276)": "Seeduwa_Station",
    "(7.222404,80.017613)": "Divulapitiya_Station",
    "(7.222445,80.017625)": "Katunayake_Station",
    "(7.120498,79.983923)": "Udugampola_Station",
    "(7.006685,79.958184)": "Kadawatha_Station",
    "(7.274298,79.862597)": "Kochchikade_Station",
    "(6.960975,79.880949)": "Paliyagoda_Station",
    "(6.837024,79.903572)": "Boralesgamuwa_Station",
    "(6.877865,79.939505)": "Thalawathugoda_Station",
    "(6.787022,79.884759)": "Moratuwa_Station",
    "(6.915059,79.881394)": "Borella_Station",
    "(6.847305,80.102153)": "Padukka_Station",
    "(7.222348,80.017553)": "Beruwala_Station",
    "(6.714853,79.989208)": "Bandaragama_Station",
    "(7.222444,80.017606)": "Maggona_Station",
    "(6.713372,79.906452)": "Panadura_Station",
    "(7.8715,80.011)": "Anamaduwa_Station",
    "(7.2845,80.6375)": "Kandy_Station",
    "(6.9847,81.0564)": "Badulla_Station",
    "(6.1528,80.2239)": "Matara_Station",
    "(8.4947,80.1739)": "Pemaduwa_Station",
    "(7.5742,79.8482)": "Chilaw_Station",
    "(7.0094,81.0565)": "Mahiyangana_Station",
    "(7.2531,80.3453)": "Kegalle_Station",
}

DEFAULT_CHARGING_STATIONS_COORDS = [
    [7.123456, 80.123456], [7.148497, 79.873276], [7.182689, 79.961171],
    [7.222404, 80.017613], [7.222445, 80.017625], [7.120498, 79.983923],
    [7.006685, 79.958184], [7.274298, 79.862597], [6.960975, 79.880949],
    [6.837024, 79.903572], [6.877865, 79.939505], [6.787022, 79.884759],
    [6.915059, 79.881394], [6.847305, 80.102153], [7.222348, 80.017553],
    [6.714853, 79.989208], [7.222444, 80.017606], [6.713372, 79.906452],
    [7.8715, 80.011], [7.2845, 80.6375], [6.9847, 81.0564],
    [6.1528, 80.2239], [8.4947, 80.1739], [7.5742, 79.8482],
    [7.0094, 81.0565], [7.2531, 80.3453]
]

# Cache configuration
CACHE_DIR = "cache"
DISTANCE_CACHE_FILE = os.path.join(CACHE_DIR, "google_distances_V1.pkl")
CACHE_EXPIRY_DAYS = 30

# Google Maps API settings
GOOGLE_MAPS_BASE_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"
MAX_ELEMENTS_PER_REQUEST = 25
API_RATE_LIMIT_DELAY = 0.1

# --- Distance Cache Manager (Same as before) ---
class GoogleDistanceCache:
    """Manages caching of Google Maps distance calculations with O(1) lookup"""
    
    def __init__(self, api_key: str, cache_file: str = DISTANCE_CACHE_FILE):
        self.api_key = api_key
        self.cache_file = cache_file
        self.cache_data = {}
        self.api_calls_made = 0
        self.cache_hits = 0
        self._load_cache()
    
    def _load_cache(self):
        """Load existing cache from disk"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    cache_with_metadata = pickle.load(f)
                    if isinstance(cache_with_metadata, dict) and 'data' in cache_with_metadata:
                        cache_age_days = (time.time() - cache_with_metadata.get('timestamp', 0)) / (24 * 3600)
                        if cache_age_days < CACHE_EXPIRY_DAYS:
                            self.cache_data = cache_with_metadata['data']
                            print(f"Loaded {len(self.cache_data)} cached distances (age: {cache_age_days:.1f} days)")
                        else:
                            print(f"Cache expired (age: {cache_age_days:.1f} days), starting fresh")
        except Exception as e:
            print(f"Warning: Could not load distance cache: {e}")
    
    def _save_cache(self):
        """Save cache to disk with metadata"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            cache_with_metadata = {
                'data': self.cache_data,
                'timestamp': time.time(),
                'version': '3.0'
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_with_metadata, f)
        except Exception as e:
            print(f"Warning: Could not save distance cache: {e}")
    
    def _create_cache_key(self, origin: Tuple[float, float], destination: Tuple[float, float]) -> str:
        """Create a unique cache key for origin-destination pair"""
        orig_key = f"{origin[0]:.6f},{origin[1]:.6f}"
        dest_key = f"{destination[0]:.6f},{destination[1]:.6f}"
        key_pair = tuple(sorted([orig_key, dest_key]))
        return f"{key_pair[0]}|{key_pair[1]}"
    
    def get_distance(self, origin: Tuple[float, float], destination: Tuple[float, float]) -> Optional[float]:
        """Get cached distance or None if not cached"""
        cache_key = self._create_cache_key(origin, destination)
        if cache_key in self.cache_data:
            self.cache_hits += 1
            return self.cache_data[cache_key]['distance']
        return None
    
    def _query_google_api(self, origins: List[Tuple[float, float]], 
                         destinations: List[Tuple[float, float]]) -> Dict:
        """Query Google Maps Distance Matrix API"""
        origins_str = "|".join([f"{lat},{lon}" for lat, lon in origins])
        destinations_str = "|".join([f"{lat},{lon}" for lat, lon in destinations])
        
        params = {
            'origins': origins_str,
            'destinations': destinations_str,
            'units': 'metric',
            'mode': 'driving',
            'avoid': 'tolls',
            'key': self.api_key
        }
        
        try:
            response = requests.get(GOOGLE_MAPS_BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            
            self.api_calls_made += 1
            time.sleep(API_RATE_LIMIT_DELAY)
            
            return response.json()
        except Exception as e:
            print(f"Google API request failed: {e}")
            return None
    
    def batch_cache_distances(self, points: List[Tuple[float, float]]):
        """Pre-cache distances between all point pairs using batch requests"""
        print(f"Pre-caching distances for {len(points)} points...")
        
        uncached_pairs = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if self.get_distance(points[i], points[j]) is None:
                    uncached_pairs.append((points[i], points[j]))
        
        if not uncached_pairs:
            print("All distances already cached!")
            return
        
        print(f"Need to cache {len(uncached_pairs)} distance pairs")
        
        batch_size = min(MAX_ELEMENTS_PER_REQUEST, int(MAX_ELEMENTS_PER_REQUEST**0.5))
        cached_count = 0
        
        for i in range(0, len(uncached_pairs), batch_size):
            batch_pairs = uncached_pairs[i:i + batch_size]
            origins = [pair[0] for pair in batch_pairs]
            destinations = [pair[1] for pair in batch_pairs]
            
            api_response = self._query_google_api(origins, destinations)
            
            if api_response and api_response.get('status') == 'OK':
                rows = api_response.get('rows', [])
                
                for row_idx, row in enumerate(rows):
                    elements = row.get('elements', [])
                    for col_idx, element in enumerate(elements):
                        if element.get('status') == 'OK':
                            distance_m = element['distance']['value']
                            duration_s = element['duration']['value']
                            distance_km = distance_m / 1000.0
                            
                            origin = origins[row_idx]
                            destination = destinations[col_idx]
                            cache_key = self._create_cache_key(origin, destination)
                            
                            self.cache_data[cache_key] = {
                                'distance': distance_km,
                                'duration': duration_s,
                                'timestamp': time.time()
                            }
                            cached_count += 1
                
                if cached_count % 50 == 0:
                    self._save_cache()
                    print(f"Cached {cached_count}/{len(uncached_pairs)} distances...")
        
        self._save_cache()
        print(f"Completed caching {cached_count} distances. API calls made: {self.api_calls_made}")
    
    def get_distance_with_fallback(self, origin: Tuple[float, float], 
                                  destination: Tuple[float, float]) -> float:
        """Get distance with fallback to geodesic if not cached and no API key"""
        cached_distance = self.get_distance(origin, destination)
        if cached_distance is not None:
            return cached_distance
        
        if self.api_key:
            api_response = self._query_google_api([origin], [destination])
            if api_response and api_response.get('status') == 'OK':
                rows = api_response.get('rows', [])
                if rows and rows[0].get('elements'):
                    element = rows[0]['elements'][0]
                    if element.get('status') == 'OK':
                        distance_km = element['distance']['value'] / 1000.0
                        cache_key = self._create_cache_key(origin, destination)
                        self.cache_data[cache_key] = {
                            'distance': distance_km,
                            'duration': element['duration']['value'],
                            'timestamp': time.time()
                        }
                        self._save_cache()
                        return distance_km
        
        geodesic_dist = geodesic(origin, destination).km
        print(f"Using geodesic fallback for {origin} -> {destination}: {geodesic_dist:.2f}km")
        return geodesic_dist
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'cached_distances': len(self.cache_data),
            'api_calls_made': self.api_calls_made,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': f"{(self.cache_hits / max(1, self.cache_hits + self.api_calls_made) * 100):.1f}%"
        }

# --- Path Node Definition ---
@dataclass
class PathNode:
    """Represents a node in the route path"""
    station_idx: int  # -1 for source, -2 for destination, >=0 for stations
    coordinates: Tuple[float, float]
    visit_count: int = 1
    
    def __lt__(self, other):
        if not isinstance(other, PathNode):
            return NotImplemented
        if self.station_idx != other.station_idx:
            return self.station_idx < other.station_idx
        return self.visit_count < other.visit_count
    
    def __eq__(self, other):
        if not isinstance(other, PathNode):
            return False
        return (self.station_idx == other.station_idx and 
                self.coordinates == other.coordinates and
                self.visit_count == other.visit_count)

# --- Strategic Station Selection ---
class StrategicBatteryStrategy:
    """Strategy for strategic station selection based on route position"""
    
    @staticmethod
    def find_first_station(current_location: Tuple[float, float],
                          destination: Tuple[float, float],
                          current_battery: float,
                          km_per_percent: float,
                          available_stations: List[Tuple[int, Tuple[float, float]]],
                          distance_cache: GoogleDistanceCache,
                          min_battery_reserve: float = 5.0) -> Optional[Tuple[int, Tuple[float, float], float]]:
        """
        Find the first station that maximizes progress towards destination while utilizing battery
        Priority: Distance towards destination + battery utilization
        """
        
        viable_stations = []
        current_to_dest_direct = distance_cache.get_distance_with_fallback(current_location, destination)
        
        for station_idx, station_coords in available_stations:
            # Calculate distance to station
            dist_to_station = distance_cache.get_distance_with_fallback(current_location, station_coords)
            battery_to_reach_station = dist_to_station / km_per_percent + min_battery_reserve
            
            # Check if we can reach this station
            if current_battery < battery_to_reach_station:
                continue
            
            # Calculate distance from station to destination
            dist_station_to_dest = distance_cache.get_distance_with_fallback(station_coords, destination)
            battery_needed_from_station = dist_station_to_dest / km_per_percent + min_battery_reserve
            
            # Check if we can reach destination from this station with full charge
            if battery_needed_from_station > 100.0:
                continue
            
            # Calculate progress towards destination
            progress_made = max(0, current_to_dest_direct - dist_station_to_dest)
            progress_percentage = (progress_made / current_to_dest_direct) * 100 if current_to_dest_direct > 0 else 0
            
            # Calculate battery utilization
            battery_used_to_station = dist_to_station / km_per_percent
            battery_utilization = (battery_used_to_station / current_battery) * 100
            
            # For first station: Prioritize progress (60%) + battery utilization (40%)
            combined_score = progress_percentage * 0.7 + battery_utilization * 0.3
            
            viable_stations.append((station_idx, station_coords, combined_score, progress_percentage, 
                                  battery_utilization, dist_to_station, progress_made))
        
        if not viable_stations:
            return None
        
        # Sort by combined score (descending) - highest progress + utilization first
        viable_stations.sort(key=lambda x: x[2], reverse=True)
        
        best_station = viable_stations[0]
        print(f"First station selected - Progress: {best_station[3]:.1f}% towards destination, "
              f"Battery utilization: {best_station[4]:.1f}%, Distance: {best_station[5]:.1f}km, "
              f"Progress made: {best_station[6]:.1f}km")
        
        return (best_station[0], best_station[1], best_station[2])
    
    @staticmethod
    def find_final_station(current_location: Tuple[float, float],
                          destination: Tuple[float, float],
                          current_battery: float,
                          km_per_percent: float,
                          available_stations: List[Tuple[int, Tuple[float, float]]],
                          distance_cache: GoogleDistanceCache,
                          min_battery_reserve: float = 15.0) -> Optional[Tuple[int, Tuple[float, float], float]]:
        """
        Find the final station closest to destination to maximize arrival battery
        Priority: Minimize distance to destination
        """
        
        viable_stations = []
        
        for station_idx, station_coords in available_stations:
            # Calculate distance to station
            dist_to_station = distance_cache.get_distance_with_fallback(current_location, station_coords)
            battery_to_reach_station = dist_to_station / km_per_percent + min_battery_reserve
            
            # Check if we can reach this station
            if current_battery < battery_to_reach_station:
                continue
            
            # Calculate distance from station to destination
            dist_station_to_dest = distance_cache.get_distance_with_fallback(station_coords, destination)
            battery_needed_from_station = dist_station_to_dest / km_per_percent + min_battery_reserve
            
            # Check if we can reach destination from this station with full charge
            if battery_needed_from_station > 100.0:
                continue
            
            # Calculate arrival battery at destination
            arrival_battery = 100.0 - (dist_station_to_dest / km_per_percent)
            
            # For final station: Prioritize closest to destination (maximize arrival battery)
            # Score is inverse of distance to destination (closer = higher score)
            closeness_score = 1000.0 / (dist_station_to_dest + 1.0)  # +1 to avoid division by zero
            
            viable_stations.append((station_idx, station_coords, closeness_score, dist_station_to_dest, 
                                  arrival_battery, dist_to_station))
        
        if not viable_stations:
            return None
        
        # Sort by closeness score (descending) - closest to destination first
        viable_stations.sort(key=lambda x: x[2], reverse=True)
        
        best_station = viable_stations[0]
        print(f"Final station selected - Distance to destination: {best_station[3]:.1f}km, "
              f"Expected arrival battery: {best_station[4]:.1f}%, Distance to station: {best_station[5]:.1f}km")
        
        return (best_station[0], best_station[1], best_station[2])
    
    @staticmethod
    def find_middle_station(current_location: Tuple[float, float],
                           destination: Tuple[float, float],
                           current_battery: float,
                           km_per_percent: float,
                           available_stations: List[Tuple[int, Tuple[float, float]]],
                           distance_cache: GoogleDistanceCache,
                           min_battery_reserve: float = 5.0) -> Optional[Tuple[int, Tuple[float, float], float]]:
        """
        Find middle station that maximizes battery utilization
        Priority: Battery utilization + reasonable progress
        """
        
        viable_stations = []
        
        for station_idx, station_coords in available_stations:
            # Calculate distance to station
            dist_to_station = distance_cache.get_distance_with_fallback(current_location, station_coords)
            battery_to_reach_station = dist_to_station / km_per_percent + min_battery_reserve
            
            # Check if we can reach this station
            if current_battery < battery_to_reach_station:
                continue
            
            # Calculate distance from station to destination
            dist_station_to_dest = distance_cache.get_distance_with_fallback(station_coords, destination)
            battery_needed_from_station = dist_station_to_dest / km_per_percent + min_battery_reserve
            
            # Check if we can reach destination from this station with full charge
            if battery_needed_from_station > 100.0:
                continue
            
            # Calculate battery utilization score
            battery_used_to_station = dist_to_station / km_per_percent
            battery_utilization = battery_used_to_station / current_battery * 100
            
            # Calculate progress score
            current_to_dest = distance_cache.get_distance_with_fallback(current_location, destination)
            progress_score = max(0, (current_to_dest - dist_station_to_dest) / current_to_dest * 100)
            
            # For middle stations: Prioritize battery utilization (70%) + progress (30%)
            combined_score = battery_utilization * 0.7 + progress_score * 0.3
            
            viable_stations.append((station_idx, station_coords, combined_score, battery_utilization, dist_to_station))
        
        if not viable_stations:
            return None
        
        # Sort by combined score (descending) - highest utilization + progress first
        viable_stations.sort(key=lambda x: x[2], reverse=True)
        
        best_station = viable_stations[0]
        print(f"Middle station selected - Battery utilization: {best_station[3]:.1f}%, "
              f"Distance: {best_station[4]:.1f}km")
        
        return (best_station[0], best_station[1], best_station[2])

# --- Enhanced EV Route Planner with Strategic Selection ---
class EVRoutePlannerV1:
    """EV Route Planner with strategic first/final station selection"""
    
    def __init__(self,
                 source: Tuple[float, float],
                 destination: Tuple[float, float],
                 initial_battery_percent: float,
                 km_per_percent: float,
                 charging_stations: List[Tuple[float, float]],
                 max_charging_stops: int = 10,
                 min_battery_reserve: float = 5.0,
                 google_api_key: str = None):
        self.source = source
        self.destination = destination
        self.initial_battery_percent = initial_battery_percent
        self.km_per_percent = km_per_percent
        self.charging_stations = charging_stations
        self.max_charging_stops = max_charging_stops
        self.min_battery_reserve = min_battery_reserve
        
        # Initialize distance cache
        self.distance_cache = GoogleDistanceCache(google_api_key or "")
        
        # Pre-cache all station-to-station distances
        self._initialize_distance_cache()
    
    def _initialize_distance_cache(self):
        """Pre-cache distances between all charging stations for O(1) lookup"""
        print("Initializing distance cache for strategic battery utilization...")
        all_points = [self.source, self.destination] + self.charging_stations
        self.distance_cache.batch_cache_distances(all_points)
        stats = self.distance_cache.get_stats()
        print(f"Distance cache ready: {stats['cached_distances']} cached distances")
    
    def find_strategic_route(self):
        """
        Find route using strategic station selection:
        - First station: Maximize progress + battery utilization
        - Final station: Closest to destination
        - Middle stations: Maximum battery utilization
        """
        
        route_path = [PathNode(-1, self.source)]
        current_location = self.source
        current_battery = self.initial_battery_percent
        total_distance = 0.0
        charging_stops = 0
        
        print(f"Starting strategic route planning")
        print(f"From {self.source} to {self.destination}")
        print(f"Initial battery: {current_battery}%, Range: {current_battery * self.km_per_percent:.1f}km")
        
        while charging_stops < self.max_charging_stops:
            # First, check if we can reach destination directly
            dist_to_dest = self.distance_cache.get_distance_with_fallback(current_location, self.destination)
            battery_needed_for_dest = dist_to_dest / self.km_per_percent + self.min_battery_reserve
            
            if current_battery >= battery_needed_for_dest:
                # We can reach destination!
                total_distance += dist_to_dest
                route_path.append(PathNode(-2, self.destination))
                arrival_battery = current_battery - (dist_to_dest / self.km_per_percent)
                print(f"✓ Reached destination directly. Distance: {dist_to_dest:.1f}km, "
                      f"Arrival battery: {arrival_battery:.1f}%, Total distance: {total_distance:.2f}km, Stops: {charging_stops}")
                return (total_distance, route_path)
            
            # Find stations not yet visited in current path
            visited_stations = set(node.station_idx for node in route_path if node.station_idx >= 0)
            available_stations = [
                (idx, tuple(coords)) for idx, coords in enumerate(self.charging_stations)
                if idx not in visited_stations
            ]
            
            if not available_stations:
                print("✗ No more available stations and cannot reach destination")
                return None
            
            # Determine which strategy to use based on route progress
            selected_station = None
            
            if charging_stops == 0:
                # First station: Maximize progress towards destination + battery utilization
                print("Selecting FIRST station (maximize progress + battery utilization)...")
                selected_station = StrategicBatteryStrategy.find_first_station(
                    current_location=current_location,
                    destination=self.destination,
                    current_battery=current_battery,
                    km_per_percent=self.km_per_percent,
                    available_stations=available_stations,
                    distance_cache=self.distance_cache,
                    min_battery_reserve=self.min_battery_reserve
                )
            else:
                # Check if this might be the final station needed
                # Try to find a station that can reach destination and see if it's close
                potential_final_stations = []
                for station_idx, station_coords in available_stations:
                    dist_to_station = self.distance_cache.get_distance_with_fallback(current_location, station_coords)
                    battery_to_reach_station = dist_to_station / self.km_per_percent + self.min_battery_reserve
                    
                    if current_battery >= battery_to_reach_station:
                        dist_station_to_dest = self.distance_cache.get_distance_with_fallback(station_coords, self.destination)
                        battery_needed_from_station = dist_station_to_dest / self.km_per_percent + self.min_battery_reserve
                        
                        if battery_needed_from_station <= 100.0:
                            potential_final_stations.append((station_idx, station_coords, dist_station_to_dest))
                
                # If we have stations that can reach destination, and the closest is reasonably close,
                # use final station strategy
                if potential_final_stations:
                    potential_final_stations.sort(key=lambda x: x[2])  # Sort by distance to destination
                    closest_to_dest = potential_final_stations[0][2]
                    
                    # If closest station is within reasonable distance to destination (say, 1/3 of total original distance)
                    total_original_distance = self.distance_cache.get_distance_with_fallback(self.source, self.destination)
                    if closest_to_dest <= total_original_distance * 0.4:  # Within 40% of original distance
                        print("Selecting FINAL station (closest to destination)...")
                        selected_station = StrategicBatteryStrategy.find_final_station(
                            current_location=current_location,
                            destination=self.destination,
                            current_battery=current_battery,
                            km_per_percent=self.km_per_percent,
                            available_stations=available_stations,
                            distance_cache=self.distance_cache,
                            min_battery_reserve=self.min_battery_reserve
                        )
                
                # If not using final station strategy, use middle station strategy
                if selected_station is None:
                    print("Selecting MIDDLE station (maximize battery utilization)...")
                    selected_station = StrategicBatteryStrategy.find_middle_station(
                        current_location=current_location,
                        destination=self.destination,
                        current_battery=current_battery,
                        km_per_percent=self.km_per_percent,
                        available_stations=available_stations,
                        distance_cache=self.distance_cache,
                        min_battery_reserve=self.min_battery_reserve
                    )
            
            if not selected_station:
                print("✗ No viable stations found")
                return None
            
            station_idx, station_coords, score = selected_station
            
            # Move to selected station
            dist_to_station = self.distance_cache.get_distance_with_fallback(current_location, station_coords)
            battery_used = dist_to_station / self.km_per_percent
            
            total_distance += dist_to_station
            current_battery -= battery_used
            current_location = station_coords
            charging_stops += 1
            
            # Add station to route
            route_path.append(PathNode(station_idx, station_coords))
            
            # Charge to full
            current_battery = 100.0
            
            coord_key = f"({station_coords[0]},{station_coords[1]})"
            station_name = STATION_MAPPING.get(coord_key, f"Station_{station_idx}")
            print(f"Stop {charging_stops}: {station_name}, Distance: {dist_to_station:.1f}km, "
                  f"Battery used: {battery_used:.1f}%, Score: {score:.1f}")
        
        print("✗ Exceeded maximum charging stops")
        return None
    
    def find_optimal_route(self):
        """
        Main route finding method - uses strategic station selection
        """
        
        # Try strategic approach
        strategic_result = self.find_strategic_route()
        if strategic_result:
            return strategic_result
        
        print("Strategic approach failed, falling back to A* search...")
        
        # Fallback to A* with battery utilization bias
        return self.find_astar_route_with_battery_bias()
    
    def find_astar_route_with_battery_bias(self):
        """A* search with bias toward maximum battery utilization"""
        
        def heuristic(point):
            return self.distance_cache.get_distance_with_fallback(point, self.destination)

        start_point = self.source
        start_f_score = heuristic(start_point)
        initial_path = [PathNode(-1, start_point)]
        
        # Modified scoring to prefer high battery utilization
        queue = [(start_f_score, 0, start_point, self.initial_battery_percent, initial_path, 0)]
        visited_states = {}
        best_route = None
        nodes_explored = 0
        max_nodes = 20000

        while queue and nodes_explored < max_nodes:
            f_score, g_score, current_loc, current_battery, path, total_utilization = heapq.heappop(queue)
            nodes_explored += 1

            battery_rounded = round(current_battery, 1)
            state_key = (current_loc, battery_rounded, len(path))
            
            if state_key in visited_states and visited_states[state_key] <= g_score:
                continue
            visited_states[state_key] = g_score
            
            if best_route and g_score >= best_route[0] * 1.1:
                continue

            if len([n for n in path if n.station_idx >= 0]) >= self.max_charging_stops:
                continue

            # Check destination reachability
            dist_to_dest = self.distance_cache.get_distance_with_fallback(current_loc, self.destination)
            required_battery = dist_to_dest / self.km_per_percent + self.min_battery_reserve
            
            if current_battery >= required_battery:
                total_distance = g_score + dist_to_dest
                if best_route is None or total_distance < best_route[0]:
                    destination_node = PathNode(-2, self.destination)
                    best_route = (total_distance, path + [destination_node])

            # Explore stations with battery utilization preference
            visited_stations = set(node.station_idx for node in path if node.station_idx >= 0)
            
            for idx, station_coords in enumerate(self.charging_stations):
                if idx in visited_stations:
                    continue
                    
                station_tuple = tuple(station_coords)
                dist_to_station = self.distance_cache.get_distance_with_fallback(current_loc, station_tuple)
                required_battery_to_station = dist_to_station / self.km_per_percent + self.min_battery_reserve

                if current_battery < required_battery_to_station:
                    continue
                
                # Calculate battery utilization for this choice
                battery_utilization = (dist_to_station / self.km_per_percent) / current_battery
                utilization_bonus = battery_utilization * 20  # Reward high utilization
                
                new_g_score = g_score + dist_to_station - utilization_bonus  # Subtract to prefer higher utilization
                
                if best_route and new_g_score >= best_route[0]:
                    continue
                
                station_node = PathNode(idx, station_tuple)
                new_path = path + [station_node]
                
                battery_after_charge = 100.0
                h_val = heuristic(station_tuple)
                new_f_score = new_g_score + h_val
                new_total_utilization = total_utilization + battery_utilization
                
                heapq.heappush(queue, (new_f_score, new_g_score, station_tuple, 
                                     battery_after_charge, new_path, new_total_utilization))
        
        print(f"A* completed. Explored {nodes_explored} nodes")
        return best_route

    def generate_route_summary(self, route_tuple: Optional[Tuple[float, List[PathNode]]]) -> List[Dict]:
        """Generate detailed route summary with strategic selection info"""
        if not route_tuple:
            return []

        total_distance, path_nodes = route_tuple
        route_summary = []
        current_battery = self.initial_battery_percent
        station_visit_counts = {}

        for i, node in enumerate(path_nodes):
            current_coords = node.coordinates
            
            if node.station_idx == -1:
                category = "Source"
                station_name = None
                selection_strategy = "Start Point"
            elif node.station_idx == -2:
                category = "Destination"
                station_name = None
                selection_strategy = "End Point"
            else:
                category = "Visiting_Charging_Station"
                coord_key = f"({current_coords[0]},{current_coords[1]})"
                base_name = STATION_MAPPING.get(coord_key, f"Station_{node.station_idx}")
                
                if node.station_idx not in station_visit_counts:
                    station_visit_counts[node.station_idx] = 0
                station_visit_counts[node.station_idx] += 1
                visit_number = station_visit_counts[node.station_idx]
                
                if visit_number > 1:
                    station_name = f"{base_name}_Visit_{visit_number}"
                else:
                    station_name = base_name
                
                # Determine selection strategy based on position
                charging_stations_in_route = [n for n in path_nodes if n.station_idx >= 0]
                current_station_position = next(j for j, n in enumerate(charging_stations_in_route) if n == node)
                
                if current_station_position == 0:
                    selection_strategy = "First Station (Progress + Battery Utilization)"
                elif current_station_position == len(charging_stations_in_route) - 1:
                    selection_strategy = "Final Station (Closest to Destination)"
                else:
                    selection_strategy = "Middle Station (Maximum Battery Utilization)"

            battery_arrival = current_battery
            battery_departure = 100.0 if category == "Visiting_Charging_Station" else current_battery
            
            # Calculate real distance to next stop
            next_stop_distance = 0.0
            battery_utilization = 0.0
            progress_made = 0.0
            
            if i < len(path_nodes) - 1:
                next_coords = path_nodes[i+1].coordinates
                next_stop_distance = self.distance_cache.get_distance_with_fallback(current_coords, next_coords)
                if battery_departure > 0:
                    battery_utilization = (next_stop_distance / self.km_per_percent) / battery_departure * 100
                
                # Calculate progress made towards destination
                if i == 0:
                    # From source
                    direct_to_dest = self.distance_cache.get_distance_with_fallback(current_coords, self.destination)
                    next_to_dest = self.distance_cache.get_distance_with_fallback(next_coords, self.destination)
                    progress_made = max(0, direct_to_dest - next_to_dest)
            
            stop_info = {
                "location": f"({current_coords[0]},{current_coords[1]})",
                "category": category,
                "battery_on_arrival_percent": round(battery_arrival, 2),
                "battery_on_departure_percent": round(battery_departure, 2),
                "next_stop_distance_km": round(next_stop_distance, 2),
                "battery_utilization_percent": round(battery_utilization, 1),
                "selection_strategy": selection_strategy
            }
            
            if station_name:
                stop_info["station_name"] = station_name
            
            if progress_made > 0:
                stop_info["progress_towards_destination_km"] = round(progress_made, 2)
            
            route_summary.append(stop_info)
            
            # Update battery for next iteration
            if next_stop_distance > 0:
                current_battery = battery_departure - (next_stop_distance / self.km_per_percent)

        return route_summary

# --- Main API Function ---
def plan_optimal_ev_route(
    source: str,
    destination: str,
    battery: float,
    efficiency: float,
    stations_json: str,
    max_charging_stops: int = 10,
    google_api_key: str = None,
    output_path: str = None,
    format_: str = 'json'
) -> str:
    """
    Strategic EV route planning with optimized first/final station selection
    """
    
    print(f"EV Route Service V1 - Strategic Battery Utilization")
    print(f"  Source: {source}")
    print(f"  Destination: {destination}")
    print(f"  Battery: {battery}%")
    print(f"  Efficiency: {efficiency} km/%")
    print(f"  Max stops: {max_charging_stops}")
    print(f"  Strategy: First=Progress+Battery, Final=Closest, Middle=Max Battery")
    
    try:
        source_lat, source_lon = map(float, source.split(','))
        dest_lat, dest_lon = map(float, destination.split(','))

        charging_stations = []
        if stations_json:
            try:
                parsed_stations = json.loads(stations_json)
                charging_stations = [
                    (float(station[0]), float(station[1]))
                    for station in parsed_stations 
                    if isinstance(station, list) and len(station) == 2
                ]
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                print(f"Error parsing stations_json: {e}. Using default stations.")
                charging_stations = [(coords[0], coords[1]) for coords in DEFAULT_CHARGING_STATIONS_COORDS]
        else:
            charging_stations = [(coords[0], coords[1]) for coords in DEFAULT_CHARGING_STATIONS_COORDS]

        if not charging_stations:
            return json.dumps({
                "success": False,
                "distance_km": None,
                "message": "No charging stations available",
                "planned_charging_stops_count": 0,
                "unique_stations_visited": 0,
                "route_summary": [],
                "algorithm_used": "EV Route Planner V1 (Strategic Battery Utilization)",
                "google_api_calls_used": 0
            }, indent=2)

        # Create route planner with strategic battery utilization
        planner = EVRoutePlannerV1(
            source=(source_lat, source_lon),
            destination=(dest_lat, dest_lon),
            initial_battery_percent=battery,
            km_per_percent=efficiency,
            charging_stations=charging_stations,
            max_charging_stops=max_charging_stops,
            google_api_key=google_api_key
        )
        
        optimal_route = planner.find_optimal_route()
        
        # Get cache statistics
        cache_stats = planner.distance_cache.get_stats()
        
        if optimal_route:
            route_summary = planner.generate_route_summary(optimal_route)
            
            charging_stops = sum(1 for stop in route_summary 
                               if stop["category"] == "Visiting_Charging_Station")
            
            visited_stations = set()
            total_battery_efficiency = 0
            utilization_count = 0
            total_progress = 0
            
            strategy_counts = {"First Station": 0, "Final Station": 0, "Middle Station": 0}
            
            for stop in route_summary:
                if stop["category"] == "Visiting_Charging_Station" and "station_name" in stop:
                    base_name = stop["station_name"].split("_Visit_")[0]
                    visited_stations.add(base_name)
                
                # Calculate average battery utilization
                if stop.get("battery_utilization_percent", 0) > 0:
                    total_battery_efficiency += stop["battery_utilization_percent"]
                    utilization_count += 1
                
                # Track progress made
                if stop.get("progress_towards_destination_km", 0) > 0:
                    total_progress += stop["progress_towards_destination_km"]
                
                # Count strategies used
                strategy = stop.get("selection_strategy", "")
                for key in strategy_counts:
                    if key in strategy:
                        strategy_counts[key] += 1
            
            unique_stations = len(visited_stations)
            avg_battery_utilization = total_battery_efficiency / max(1, utilization_count)
            
            # Calculate final arrival battery
            final_stop = route_summary[-1] if route_summary else {}
            arrival_battery = final_stop.get("battery_on_arrival_percent", 0)
            
            result = {
                "success": True,
                "distance_km": round(optimal_route[0], 2),
                "message": f"Strategic route optimized - Avg battery utilization: {avg_battery_utilization:.1f}%, Arrival battery: {arrival_battery:.1f}%",
                "planned_charging_stops_count": charging_stops,
                "unique_stations_visited": unique_stations,
                "route_summary": route_summary,
                "algorithm_used": "Strategic Battery Utilization (First=Progress, Final=Closest, Middle=MaxBattery)",
                "google_api_calls_used": cache_stats['api_calls_made'],
                "cache_hit_rate": cache_stats['cache_hit_rate'],
                "distance_calculation_source": "Google Maps Distance Matrix API (cached)",
                "average_battery_utilization_percent": round(avg_battery_utilization, 1),
                "estimated_arrival_battery_percent": round(arrival_battery, 1),
                "total_progress_first_leg_km": round(total_progress, 2),
                "strategy_summary": {
                    "first_station_strategy": "Maximize progress towards destination + battery utilization",
                    "final_station_strategy": "Closest to destination for maximum arrival battery",
                    "middle_station_strategy": "Maximum battery utilization",
                    "stations_by_strategy": strategy_counts
                },
                "optimization_goals": [
                    "First leg: Maximum progress toward destination",
                    "Final leg: Maximum arrival battery",
                    "Middle legs: Maximum battery utilization"
                ]
            }
        else:
            result = {
                "success": False,
                "distance_km": None,
                "message": "Could not find feasible route with strategic station selection. Try increasing max_charging_stops.",
                "planned_charging_stops_count": 0,
                "unique_stations_visited": 0,
                "route_summary": [],
                "algorithm_used": "Strategic Battery Utilization (First=Progress, Final=Closest, Middle=MaxBattery)",
                "google_api_calls_used": cache_stats['api_calls_made'],
                "cache_hit_rate": cache_stats['cache_hit_rate']
            }

        # Save output if requested
        if output_path and result["success"]:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                result["export_message"] = f"Route data saved to: {output_path}"
            except Exception as e:
                print(f"Warning: Could not save output to {output_path}: {e}")

        return json.dumps(result, indent=2)

    except Exception as e:
        print(f"Error in route planning: {e}")
        import traceback
        traceback.print_exc()
        
        error_result = {
            "success": False,
            "distance_km": None,
            "message": f"Route planning failed: {str(e)}",
            "planned_charging_stops_count": 0,
            "unique_stations_visited": 0,
            "route_summary": [],
            "algorithm_used": "EV Route Planner V1 (Strategic Battery Utilization)",
            "google_api_calls_used": 0
        }
        return json.dumps(error_result, indent=2)

# --- Service Information ---
def get_service_info():
    """Get information about this enhanced service"""
    return {
        "service_name": "EV Route Service V1",
        "algorithm": "Strategic Battery Utilization Strategy with Google Maps",
        "features": [
            "Strategic first station selection (progress + battery utilization)",
            "Strategic final station selection (closest to destination)",
            "Middle stations use maximum battery utilization",
            "Google Maps real distance integration with O(1) caching",
            "Minimizes charging stops while optimizing battery arrival",
            "A* fallback for complex scenarios",
            "Detailed strategy tracking and reporting"
        ],
        "version": "3.0.0",
        "optimization_goals": [
            "Primary: Strategic station selection based on route position",
            "First station: Maximum progress toward destination + battery usage",
            "Final station: Minimize distance to destination (maximize arrival battery)",
            "Middle stations: Maximum battery utilization per segment"
        ],
        "strategy_details": {
            "first_station": "60% progress toward destination + 40% battery utilization",
            "final_station": "Closest to destination when within 40% of original trip distance",
            "middle_stations": "70% battery usage + 30% progress toward destination",
            "destination_validation": "Ensure each selected station allows reaching destination",
            "fallback_mechanism": "A* with battery utilization bias if strategic approach fails"
        },
        "cache_system": {
            "type": "Persistent disk-based cache with expiry",
            "lookup_complexity": "O(1)",
            "cache_file": DISTANCE_CACHE_FILE,
            "expiry_days": CACHE_EXPIRY_DAYS
        }
    }

# Compatibility aliases
optimize_ev_route = plan_optimal_ev_route
plan_route = plan_optimal_ev_route