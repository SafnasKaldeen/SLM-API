# integrated_hybrid_ev_route_service_with_logging.py
"""
Integrated Hybrid EV Route Planning Service with Detailed Decision Logging
- Shows reasoning behind each routing decision
- Explains battery calculations and station selections
- Tracks Google API usage optimization
"""

import heapq
import json
import os
import time
import pickle
import googlemaps
from typing import List, Tuple, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
from geopy.distance import geodesic
import pandas as pd
from dataclasses import dataclass

# --- Existing Pydantic Models (unchanged) ---
class RouteRequest(BaseModel):
    source_lat: float = Field(..., description="Source latitude")
    source_long: float = Field(..., description="Source longitude")
    destination_lat: float = Field(..., description="Destination latitude")
    destination_long: float = Field(..., description="Destination longitude")
    initial_battery_percent: float = Field(..., description="Initial battery percentage")
    km_per_percent: float = Field(1.4, description="Distance in km covered per percentage of battery")
    charging_stations: List[Tuple[float, float]] = Field(..., description="List of charging stations as (lat, long) tuples")
    max_charging_stops: int = Field(10, description="Maximum number of charging stops allowed")
    google_api_key: Optional[str] = Field(None, description="Google API key for enhanced routing")

class RouteStop(BaseModel):
    latitude: float
    longitude: float
    category: str  # Source, Destination, Visiting_Charging_Station, Charging_Station
    battery_arrival: float
    range_arrival: float
    battery_departure: float
    range_departure: float
    next_stop_distance: float
    visiting_flag: str  # Visit, Not Visit
    station_name: Optional[str] = None
    visit_number: Optional[int] = None
    actual_driving_distance: Optional[float] = None  # Real road distance from Google

class RouteResponse(BaseModel):
    success: bool = Field(..., description="Whether a route was found")
    total_distance: Optional[float] = Field(None, description="Total distance of the route in km")
    route_details: Optional[List[RouteStop]] = Field(None, description="Detailed information about each stop")
    error_message: Optional[str] = Field(None, description="Error message if route planning failed")
    charging_stops_count: Optional[int] = Field(None, description="Number of charging stops")
    unique_stations_visited: Optional[int] = Field(None, description="Number of unique stations visited")
    google_api_calls_used: Optional[int] = Field(None, description="Number of Google API calls made")
    distance_source: Optional[str] = Field(None, description="Source of distance calculations")

# --- Enhanced Hybrid Distance Calculator with Logging ---
class HybridDistanceCalculator:
    """Minimal Google API usage with smart caching and detailed logging"""
    
    def __init__(self, api_key: Optional[str], company_stations: List[Tuple[float, float]], cache_dir: str = "cache"):
        self.google_enabled = api_key is not None
        if self.google_enabled:
            self.client = googlemaps.Client(key=api_key)
            print(f"🔌 Google API initialized successfully")
        else:
            print(f"📍 No Google API key provided - using geodesic distance calculations with road factor")
        
        self.company_stations = company_stations
        self.cache_dir = cache_dir
        self.api_call_count = 0
        self.station_matrix = {}
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        print(f"📁 Cache directory: {cache_dir}")
        self._load_station_matrix()
    
    def _load_station_matrix(self):
        """Load pre-computed station distance matrix"""
        cache_file = os.path.join(self.cache_dir, "station_distance_matrix.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.station_matrix = cache_data.get('matrix', {})
                    created_with_google = cache_data.get('created_with_google', False)
                    station_count = cache_data.get('station_count', 0)
                    
                print(f"💾 Loaded cached station matrix:")
                print(f"   - {len(self.station_matrix)} distance pairs cached")
                print(f"   - Created with Google API: {'Yes' if created_with_google else 'No'}")
                print(f"   - Station count: {station_count}")
                
                if len(self.company_stations) != station_count:
                    print(f"⚠️  Station count mismatch - cache has {station_count}, current has {len(self.company_stations)}")
                    
            except Exception as e:
                print(f"❌ Error loading cache: {e}")
        else:
            print(f"📂 No existing cache file found - will create new matrix if needed")
    
    def setup_station_matrix(self):
        """ONE-TIME SETUP: Pre-compute station-to-station distances"""
        print(f"\n🔧 Setting up station distance matrix...")
        print(f"   - Number of stations: {len(self.company_stations)}")
        print(f"   - Total distance pairs to calculate: {len(self.company_stations) * (len(self.company_stations) - 1)}")
        
        if not self.google_enabled:
            print(f"📐 Using geodesic calculations (no Google API)")
            return self._setup_geodesic_matrix()
        
        print(f"🌐 Using Google Distance Matrix API for accurate road distances...")
        matrix = {}
        
        try:
            origins = [f"{lat},{lng}" for lat, lng in self.company_stations]
            print(f"   - Preparing batch API call for {len(origins)} stations")
            
            # Batch API call for all stations
            print(f"🔄 Making Google API call...")
            distance_result = self.client.distance_matrix(
                origins=origins,
                destinations=origins,
                mode="driving",
                units="metric"
            )
            
            self.api_call_count += 1
            print(f"✅ Google API call completed (Total calls: {self.api_call_count})")
            
            # Parse and store results
            successful_pairs = 0
            failed_pairs = 0
            
            for i, origin_row in enumerate(distance_result['rows']):
                for j, element in enumerate(origin_row['elements']):
                    if element['status'] == 'OK' and i != j:
                        distance_km = element['distance']['value'] / 1000
                        duration_sec = element['duration']['value']
                        matrix[(i, j)] = {
                            'distance_km': distance_km,
                            'duration_sec': duration_sec
                        }
                        successful_pairs += 1
                    elif i != j:
                        failed_pairs += 1
            
            print(f"📊 Matrix creation results:")
            print(f"   - Successful pairs: {successful_pairs}")
            print(f"   - Failed pairs: {failed_pairs}")
            
            # Save to cache
            cache_file = os.path.join(self.cache_dir, "station_distance_matrix.pkl")
            cache_data = {
                'matrix': matrix,
                'created_with_google': True,
                'station_count': len(self.company_stations),
                'created_timestamp': time.time()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            self.station_matrix = matrix
            print(f"💾 Matrix saved to cache: {cache_file}")
            print(f"✅ Station matrix setup complete! API calls used: {self.api_call_count}")
            return matrix
            
        except Exception as e:
            print(f"❌ Error with Google API: {e}")
            print(f"🔄 Falling back to geodesic calculations...")
            return self._setup_geodesic_matrix()
    
    def _setup_geodesic_matrix(self):
        """Fallback: Create matrix using geodesic distances with road factor"""
        print(f"📐 Creating geodesic-based distance matrix...")
        matrix = {}
        total_pairs = 0
        
        for i, station_i in enumerate(self.company_stations):
            for j, station_j in enumerate(self.company_stations):
                if i != j:
                    geodesic_dist = geodesic(station_i, station_j).km
                    road_distance = geodesic_dist * 1.3  # Road factor
                    matrix[(i, j)] = {
                        'distance_km': road_distance,
                        'duration_sec': road_distance * 60  # Rough time estimate
                    }
                    total_pairs += 1
        
        # Save to cache
        cache_file = os.path.join(self.cache_dir, "station_distance_matrix.pkl")
        cache_data = {
            'matrix': matrix,
            'created_with_google': False,
            'station_count': len(self.company_stations),
            'created_timestamp': time.time()
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        self.station_matrix = matrix
        print(f"✅ Geodesic matrix created: {total_pairs} pairs calculated")
        return matrix
    
    def get_reachable_stations_from_source(self, source: Tuple[float, float], 
                                         current_battery: float, 
                                         km_per_percent: float,
                                         min_battery_reserve: float = 5.0) -> List[Tuple[int, float]]:
        """
        Find stations reachable from source, prioritizing the farthest station
        """
        print(f"\n🔍 Finding reachable stations from source...")
        max_range = (current_battery - min_battery_reserve) * km_per_percent
        print(f"   - Current battery: {current_battery}%")
        print(f"   - Battery reserve: {min_battery_reserve}%")
        print(f"   - Usable battery: {current_battery - min_battery_reserve}%")
        print(f"   - Maximum range: {max_range:.1f} km")
        
        reachable_stations = []
        
        # First, find reachable stations using geodesic distance (quick filter)
        print(f"📐 Quick geodesic distance check for all {len(self.company_stations)} stations...")
        
        for i, station in enumerate(self.company_stations):
            geodesic_dist = geodesic(source, station).km
            # Use conservative estimate (geodesic * 1.4 for road factor)
            estimated_road_distance = geodesic_dist * 1.4
            
            if estimated_road_distance <= max_range:
                reachable_stations.append((i, geodesic_dist))
                print(f"   ✅ Station_{i}: {geodesic_dist:.1f}km geodesic, ~{estimated_road_distance:.1f}km road (REACHABLE)")
            else:
                print(f"   ❌ Station_{i}: {geodesic_dist:.1f}km geodesic, ~{estimated_road_distance:.1f}km road (TOO FAR)")
        
        if not reachable_stations:
            print(f"⚠️  No stations reachable with current battery level!")
            return []
        
        print(f"📊 Found {len(reachable_stations)} potentially reachable stations")
        
        # If Google API available, get exact distances for top candidates
        if self.google_enabled and len(reachable_stations) > 1:
            # Sort by geodesic distance and take top candidates
            reachable_stations.sort(key=lambda x: x[1], reverse=True)  # Farthest first
            top_candidates = reachable_stations[:min(5, len(reachable_stations))]
            
            print(f"🌐 Getting exact Google distances for top {len(top_candidates)} candidates...")
            print(f"   Strategy: Prioritize FARTHEST reachable stations for maximum progress")
            
            # Get exact Google distances
            exact_distances = self.get_user_to_stations_distance(source, [i for i, _ in top_candidates])
            
            # Filter by actual reachability and return farthest reachable
            final_reachable = []
            for station_idx, geodesic_dist in top_candidates:
                if station_idx in exact_distances:
                    actual_distance = exact_distances[station_idx]
                    if actual_distance <= max_range:
                        final_reachable.append((station_idx, actual_distance))
                        print(f"   ✅ Station_{station_idx}: {actual_distance:.1f}km actual road distance (CONFIRMED REACHABLE)")
                    else:
                        print(f"   ❌ Station_{station_idx}: {actual_distance:.1f}km actual road distance (TOO FAR - geodesic underestimated)")
                else:
                    print(f"   ⚠️  Station_{station_idx}: Google API failed, skipping")
            
            # Return sorted by actual distance (farthest first)
            final_sorted = sorted(final_reachable, key=lambda x: x[1], reverse=True)
            
            if final_sorted:
                print(f"🎯 Final selection: Station_{final_sorted[0][0]} at {final_sorted[0][1]:.1f}km (FARTHEST REACHABLE)")
            
            return final_sorted
        
        # Fallback: return sorted by geodesic distance (farthest first)
        print(f"📐 Using geodesic distances (no Google API verification)")
        fallback_sorted = sorted(reachable_stations, key=lambda x: x[1], reverse=True)
        
        if fallback_sorted:
            print(f"🎯 Fallback selection: Station_{fallback_sorted[0][0]} at ~{fallback_sorted[0][1] * 1.3:.1f}km estimated (FARTHEST REACHABLE)")
        
        return fallback_sorted
    
    def get_stations_for_destination(self, destination: Tuple[float, float], 
                                   top_k: int = 4) -> List[Tuple[int, float]]:
        """Find nearest stations to destination"""
        print(f"\n🎯 Finding stations near destination...")
        distances = []
        
        for i, station in enumerate(self.company_stations):
            dist = geodesic(destination, station).km
            distances.append((i, dist))
            print(f"   Station_{i}: {dist:.1f}km from destination")
        
        # Sort by distance (nearest first for destination)
        distances.sort(key=lambda x: x[1])
        selected = distances[:top_k]
        
        print(f"📊 Selected {len(selected)} nearest stations to destination:")
        for i, (station_idx, dist) in enumerate(selected):
            print(f"   {i+1}. Station_{station_idx}: {dist:.1f}km")
        
        return selected
    
    def get_user_to_stations_distance(self, user_location: Tuple[float, float], 
                                    station_indices: List[int]) -> Dict[int, float]:
        """Get actual driving distances from user to stations (1 API call)"""
        print(f"\n🌐 Getting Google API distances from user to {len(station_indices)} stations...")
        
        if not self.google_enabled or not station_indices:
            print(f"📐 Falling back to geodesic calculations")
            fallback_distances = {i: geodesic(user_location, self.company_stations[i]).km * 1.3 
                                for i in station_indices}
            
            for station_idx, dist in fallback_distances.items():
                print(f"   Station_{station_idx}: ~{dist:.1f}km (geodesic * 1.3)")
            
            return fallback_distances
        
        origins = [f"{user_location[0]},{user_location[1]}"]
        destinations = [f"{self.company_stations[i][0]},{self.company_stations[i][1]}" 
                       for i in station_indices]
        
        print(f"   Making API call for user-to-stations distances...")
        
        try:
            distance_result = self.client.distance_matrix(
                origins=origins,
                destinations=destinations,
                mode="driving",
                units="metric"
            )
            
            self.api_call_count += 1
            print(f"✅ API call successful (Total calls: {self.api_call_count})")
            
            distances = {}
            if distance_result['rows']:
                row = distance_result['rows'][0]
                for i, element in enumerate(row['elements']):
                    if element['status'] == 'OK':
                        station_idx = station_indices[i]
                        distance_km = element['distance']['value'] / 1000
                        distances[station_idx] = distance_km
                        print(f"   ✅ Station_{station_idx}: {distance_km:.1f}km actual road distance")
                    else:
                        station_idx = station_indices[i]
                        print(f"   ❌ Station_{station_idx}: Google API failed for this route")
            
            return distances
            
        except Exception as e:
            print(f"❌ Google API error: {e}")
            print(f"🔄 Using geodesic fallback...")
            return {i: geodesic(user_location, self.company_stations[i]).km * 1.3 
                   for i in station_indices}
    
    def get_stations_to_user_distance(self, station_indices: List[int], 
                                    user_location: Tuple[float, float]) -> Dict[int, float]:
        """Get actual driving distances from stations to user (1 API call)"""
        print(f"\n🌐 Getting Google API distances from {len(station_indices)} stations to destination...")
        
        if not self.google_enabled or not station_indices:
            print(f"📐 Using geodesic fallback")
            return {i: geodesic(self.company_stations[i], user_location).km * 1.3 
                   for i in station_indices}
        
        origins = [f"{self.company_stations[i][0]},{self.company_stations[i][1]}" 
                   for i in station_indices]
        destinations = [f"{user_location[0]},{user_location[1]}"]
        
        print(f"   Making API call for stations-to-destination distances...")
        
        try:
            distance_result = self.client.distance_matrix(
                origins=origins,
                destinations=destinations,
                mode="driving",
                units="metric"
            )
            
            self.api_call_count += 1
            print(f"✅ API call successful (Total calls: {self.api_call_count})")
            
            distances = {}
            for i, row in enumerate(distance_result['rows']):
                if row['elements'] and row['elements'][0]['status'] == 'OK':
                    station_idx = station_indices[i]
                    distance_km = row['elements'][0]['distance']['value'] / 1000
                    distances[station_idx] = distance_km
                    print(f"   ✅ Station_{station_idx}: {distance_km:.1f}km actual road distance")
                else:
                    station_idx = station_indices[i]
                    print(f"   ❌ Station_{station_idx}: Google API failed for this route")
            
            return distances
            
        except Exception as e:
            print(f"❌ Google API error: {e}")
            print(f"🔄 Using geodesic fallback...")
            return {i: geodesic(self.company_stations[i], user_location).km * 1.3 
                   for i in station_indices}
    
    def get_distance_between_stations(self, station_i: int, station_j: int) -> float:
        """Get cached distance between stations (NO API call)"""
        if (station_i, station_j) in self.station_matrix:
            distance = self.station_matrix[(station_i, station_j)]['distance_km']
            print(f"💾 Using cached distance Station_{station_i} → Station_{station_j}: {distance:.1f}km")
            return distance
        elif (station_j, station_i) in self.station_matrix:
            distance = self.station_matrix[(station_j, station_i)]['distance_km']
            print(f"💾 Using cached distance Station_{station_j} → Station_{station_i}: {distance:.1f}km (reversed)")
            return distance
        else:
            # Fallback to geodesic
            dist = geodesic(self.company_stations[station_i], 
                          self.company_stations[station_j]).km
            road_distance = dist * 1.3
            print(f"📐 No cache, using geodesic Station_{station_i} → Station_{station_j}: {road_distance:.1f}km")
            return road_distance

# --- Enhanced PathNode ---
@dataclass
class PathNode:
    station_idx: int
    coordinates: Tuple[float, float]
    visit_count: int = 1
    actual_distance_to_here: Optional[float] = None  # Google API distance
    
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

# --- Enhanced EV Route Planner with Decision Logging ---
class IntegratedHybridEVRoutePlanner:
    """Enhanced EV Route Planner with detailed decision logging"""
    
    def __init__(self,
                 source: Tuple[float, float],
                 destination: Tuple[float, float],
                 initial_battery_percent: float,
                 km_per_percent: float,
                 charging_stations: List[Tuple[float, float]],
                 max_charging_stops: int = 10,
                 min_battery_reserve: float = 5.0,
                 google_api_key: Optional[str] = None):
        
        print(f"\n🚗 Initializing EV Route Planner...")
        print(f"   📍 Source: {source}")
        print(f"   🎯 Destination: {destination}")
        print(f"   🔋 Initial battery: {initial_battery_percent}%")
        print(f"   ⚡ Efficiency: {km_per_percent} km per 1% battery")
        print(f"   🛣️  Total range with current battery: {initial_battery_percent * km_per_percent:.1f} km")
        print(f"   🚏 Available charging stations: {len(charging_stations)}")
        print(f"   🔢 Max charging stops allowed: {max_charging_stops}")
        print(f"   🛡️ Battery reserve: {min_battery_reserve}%")
        
        self.source = source
        self.destination = destination
        self.initial_battery_percent = initial_battery_percent
        self.km_per_percent = km_per_percent
        self.charging_stations = charging_stations
        self.max_charging_stops = max_charging_stops
        self.min_battery_reserve = min_battery_reserve
        self.max_range = initial_battery_percent * km_per_percent
        
        # Initialize hybrid distance calculator
        self.distance_calculator = HybridDistanceCalculator(
            google_api_key, charging_stations
        )
        
        # Smart station selection
        self._select_candidate_stations()
    
    def _select_candidate_stations(self):
        """Smart station selection based on battery range and destination proximity"""
        print(f"\n🧠 Smart station selection process...")
        
        # Get reachable stations from source (prioritize farthest)
        self.source_candidates = self.distance_calculator.get_reachable_stations_from_source(
            self.source, self.initial_battery_percent, self.km_per_percent, self.min_battery_reserve
        )
        
        # Get stations near destination (any station can be final)
        self.dest_candidates = self.distance_calculator.get_stations_for_destination(
            self.destination, top_k=4
        )
        
        print(f"\n📊 Station Selection Summary:")
        print(f"   🎯 Source candidates (farthest first): {len(self.source_candidates)} stations")
        print(f"   🏁 Destination candidates: {len(self.dest_candidates)} stations")
        
        if self.source_candidates:
            farthest_idx, farthest_dist = self.source_candidates[0]
            print(f"   🥇 Best source candidate: Station_{farthest_idx} at {farthest_dist:.2f} km")
            print(f"      Reasoning: Maximizes progress toward destination with current battery")
        else:
            print(f"   ⚠️  No reachable stations from source - may need to adjust parameters")

    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate distance - uses Google API results when available, otherwise geodesic"""
        return geodesic(point1, point2).km
    
    def get_station_visit_count(self, path: List[PathNode], station_idx: int) -> int:
        """Count visits to a station in current path"""
        count = sum(1 for node in path if node.station_idx == station_idx)
        if count > 0:
            print(f"   📊 Station_{station_idx} already visited {count} times in current path")
        return count
    
    def is_path_too_long(self, path: List[PathNode]) -> bool:
        """Check if path exceeds maximum charging stops"""
        charging_stops = sum(1 for node in path if node.station_idx >= 0)
        too_long = charging_stops >= self.max_charging_stops
        if too_long:
            print(f"   🚫 Path rejected: {charging_stops} charging stops >= {self.max_charging_stops} limit")
        return too_long
    
    def find_optimal_route(self):
        """Enhanced A* with hybrid distance calculation and detailed logging"""
        
        print(f"\n🔍 Starting optimal route search using A* algorithm...")
        
        def heuristic(point):
            return self.calculate_distance(point, self.destination)
        
        # Initialize with source
        start_point = self.source
        start_f_score = heuristic(start_point)
        initial_path = [PathNode(-1, start_point)]
        
        print(f"   🎯 Heuristic distance to destination: {start_f_score:.1f} km")
        
        queue = [(start_f_score, 0, start_point, self.initial_battery_percent, initial_path)]
        visited_states = {}
        best_route = None
        nodes_explored = 0
        max_nodes = 30000  # Reduced for efficiency
        
        print(f"   🔧 Search parameters:")
        print(f"      - Maximum nodes to explore: {max_nodes}")
        print(f"      - Google API enabled: {self.distance_calculator.google_enabled}")
        
        while queue and nodes_explored < max_nodes:
            f_score, g_score, current_loc, current_battery, path = heapq.heappop(queue)
            nodes_explored += 1
            
            if nodes_explored % 1000 == 0:
                print(f"   🔄 Explored {nodes_explored} nodes, API calls: {self.distance_calculator.api_call_count}")
            
            # State pruning
            battery_rounded = round(current_battery, 1)
            state_key = (current_loc, battery_rounded, len(path))
            
            if state_key in visited_states and visited_states[state_key] <= g_score:
                continue
            visited_states[state_key] = g_score
            
            # Early termination
            if best_route and g_score >= best_route[0] * 1.1:
                continue
            
            if self.is_path_too_long(path):
                continue
            
            # Check if can reach destination
            current_node = path[-1]
            current_location_name = "Source" if current_node.station_idx == -1 else f"Station_{current_node.station_idx}"
            
            print(f"\n🧭 Evaluating route from {current_location_name}:")
            print(f"   📍 Current location: {current_loc}")
            print(f"   🔋 Current battery: {current_battery:.1f}%")
            print(f"   📏 Distance traveled so far: {g_score:.1f} km")
            
            dist_to_dest = self.calculate_distance(current_loc, self.destination)
            print(f"   🎯 Direct distance to destination: {dist_to_dest:.1f} km")
            
            # Use actual Google distance for final leg if at a station
            if (current_node.station_idx >= 0 and 
                self.distance_calculator.google_enabled and 
                current_node.station_idx in [i for i, _ in self.dest_candidates]):
                
                print(f"   🌐 Getting exact Google distance to destination...")
                # Get exact distance to destination
                station_indices = [current_node.station_idx]
                actual_distances = self.distance_calculator.get_stations_to_user_distance(
                    station_indices, self.destination
                )
                if current_node.station_idx in actual_distances:
                    old_dist = dist_to_dest
                    dist_to_dest = actual_distances[current_node.station_idx]
                    print(f"   ✅ Google API distance: {dist_to_dest:.1f} km (vs {old_dist:.1f} km direct)")
            
            required_battery = dist_to_dest / self.km_per_percent + self.min_battery_reserve
            print(f"   ⚡ Battery needed for destination: {required_battery:.1f}% ({dist_to_dest:.1f} km + {self.min_battery_reserve}% reserve)")
            
            if current_battery >= required_battery:
                total_distance = g_score + dist_to_dest
                battery_at_dest = current_battery - (dist_to_dest / self.km_per_percent)
                
                print(f"   ✅ CAN REACH DESTINATION!")
                print(f"      - Total route distance: {total_distance:.1f} km")
                print(f"      - Battery remaining at destination: {battery_at_dest:.1f}%")
                
                if best_route is None or total_distance < best_route[0]:
                    charging_stops = sum(1 for n in path if n.station_idx >= 0)
                    print(f"   🏆 NEW BEST ROUTE FOUND!")
                    print(f"      - Previous best: {best_route[0]:.1f} km" if best_route else "No previous route")
                    print(f"      - New best: {total_distance:.1f} km")
                    print(f"      - Charging stops: {charging_stops}")
                    print(f"      - API calls used: {self.distance_calculator.api_call_count}")
                    
                    destination_node = PathNode(-2, self.destination)
                    destination_node.actual_distance_to_here = dist_to_dest
                    best_route = (total_distance, path + [destination_node])
                else:
                    print(f"   ⚖️ Route found but not better than current best ({best_route[0]:.1f} km)")
            else:
                battery_shortage = required_battery - current_battery
                print(f"   ❌ CANNOT REACH DESTINATION")
                print(f"      - Need {required_battery:.1f}% battery, have {current_battery:.1f}%")
                print(f"      - Short by {battery_shortage:.1f}% ({battery_shortage * self.km_per_percent:.1f} km)")
                print(f"      - Must visit charging station first")
            
            # Explore charging stations with smart selection
            print(f"\n🔍 Exploring charging station options from {current_location_name}...")
            candidate_stations = []
            
            # If at source, prioritize farthest reachable stations
            if current_node.station_idx == -1:  # At source
                print(f"   📍 At source - using pre-selected farthest reachable stations")
                candidate_stations = [(idx, dist) for idx, dist in self.source_candidates[:3]]  # Top 3 farthest
                print(f"   🎯 Considering top 3 farthest reachable stations:")
                for i, (idx, dist) in enumerate(candidate_stations):
                    print(f"      {i+1}. Station_{idx}: {dist:.1f} km")
            else:
                # At a charging station, consider all stations
                print(f"   🚏 At charging station - evaluating all other stations")
                all_candidates = []
                for idx, station_coords in enumerate(self.charging_stations):
                    if idx != current_node.station_idx:  # Don't revisit same station immediately
                        geodesic_dist = geodesic(current_loc, tuple(station_coords)).km
                        all_candidates.append((idx, geodesic_dist))
                
                # Sort by distance and take reasonable candidates
                all_candidates.sort(key=lambda x: x[1])
                candidate_stations = all_candidates[:8]  # Top 8 nearest
                
                print(f"   📊 Considering 8 nearest stations:")
                for i, (idx, dist) in enumerate(candidate_stations[:5]):  # Show first 5
                    print(f"      {i+1}. Station_{idx}: {dist:.1f} km")
                if len(candidate_stations) > 5:
                    print(f"      ... and {len(candidate_stations)-5} more")
            
            stations_explored = 0
            stations_reachable = 0
            stations_skipped_multivisit = 0
            stations_skipped_battery = 0
            
            for idx, estimated_dist in candidate_stations:
                stations_explored += 1
                station_tuple = tuple(self.charging_stations[idx])
                
                print(f"\n   🔍 Evaluating Station_{idx}:")
                
                # Use cached distance between stations if available
                if current_node.station_idx >= 0:  # Currently at a station
                    dist_to_station = self.distance_calculator.get_distance_between_stations(
                        current_node.station_idx, idx
                    )
                else:
                    # From source to station - will be calculated with Google if available
                    dist_to_station = estimated_dist * 1.3  # Road factor
                    print(f"      📏 Estimated road distance: {dist_to_station:.1f} km")
                
                required_battery_to_station = dist_to_station / self.km_per_percent + self.min_battery_reserve
                print(f"      ⚡ Battery needed: {required_battery_to_station:.1f}% ({dist_to_station:.1f} km + {self.min_battery_reserve}% reserve)")
                print(f"      🔋 Battery available: {current_battery:.1f}%")
                
                if current_battery < required_battery_to_station:
                    stations_skipped_battery += 1
                    battery_shortage = required_battery_to_station - current_battery
                    print(f"      ❌ UNREACHABLE - short by {battery_shortage:.1f}% ({battery_shortage * self.km_per_percent:.1f} km)")
                    continue
                
                print(f"      ✅ REACHABLE - battery after arrival: {current_battery - (dist_to_station / self.km_per_percent):.1f}%")
                
                # Multi-visit logic
                current_visit_count = self.get_station_visit_count(path, idx)
                max_visits_per_station = 3
                if current_visit_count >= max_visits_per_station:
                    stations_skipped_multivisit += 1
                    print(f"      🚫 SKIPPING - already visited {current_visit_count} times (limit: {max_visits_per_station})")
                    continue
                
                # Penalties
                revisit_penalty = current_visit_count * 15
                new_g_score = g_score + dist_to_station + revisit_penalty
                
                if revisit_penalty > 0:
                    print(f"      ⚠️ Revisit penalty applied: +{revisit_penalty} km (visit #{current_visit_count + 1})")
                
                print(f"      📊 New route distance: {new_g_score:.1f} km")
                
                if best_route and new_g_score >= best_route[0]:
                    print(f"      🚫 PRUNING - route distance {new_g_score:.1f} km >= best route {best_route[0]:.1f} km")
                    continue
                
                stations_reachable += 1
                
                # Create new path node
                station_node = PathNode(idx, station_tuple, current_visit_count + 1)
                station_node.actual_distance_to_here = dist_to_station
                new_path = path + [station_node]
                
                battery_after_charge = 100.0
                h_val = heuristic(station_tuple)
                new_f_score = new_g_score + h_val
                
                print(f"      ✅ ADDING TO QUEUE")
                print(f"         - Battery after charging: {battery_after_charge}%")
                print(f"         - Heuristic to destination: {h_val:.1f} km")
                print(f"         - Total estimated distance (f_score): {new_f_score:.1f} km")
                
                heapq.heappush(queue, (new_f_score, new_g_score, station_tuple, battery_after_charge, new_path))
            
            # Summary of station exploration
            print(f"\n   📊 Station exploration summary:")
            print(f"      - Stations evaluated: {stations_explored}")
            print(f"      - Reachable and added to queue: {stations_reachable}")
            print(f"      - Skipped (insufficient battery): {stations_skipped_battery}")
            print(f"      - Skipped (too many visits): {stations_skipped_multivisit}")
            print(f"      - Queue size: {len(queue)}")
        
        print(f"\n🏁 Route search completed!")
        print(f"   📊 Search statistics:")
        print(f"      - Nodes explored: {nodes_explored}")
        print(f"      - Google API calls made: {self.distance_calculator.api_call_count}")
        print(f"      - Best route found: {'Yes' if best_route else 'No'}")
        
        if best_route:
            charging_stops = sum(1 for n in best_route[1] if n.station_idx >= 0)
            print(f"   🏆 Optimal route details:")
            print(f"      - Total distance: {best_route[0]:.1f} km")
            print(f"      - Charging stops: {charging_stops}")
            print(f"      - Unique stations: {len(set(n.station_idx for n in best_route[1] if n.station_idx >= 0))}")
        else:
            print(f"   ❌ No feasible route found")
            print(f"      - Try increasing max_charging_stops")
            print(f"      - Check if destination is reachable")
            print(f"      - Verify charging station locations")
        
        return best_route
    
    def generate_route_log(self, route_tuple):
        """Generate detailed route log with actual Google distances when available"""
        if not route_tuple:
            print(f"\n❌ Cannot generate route log - no route provided")
            return []

        total_distance, path_nodes = route_tuple
        logged_stops = []
        current_batt = self.initial_battery_percent
        station_visit_counts = {}

        print(f"\n📋 Generating detailed route log...")
        print(f"   🛣️ Total route distance: {total_distance:.1f} km")
        print(f"   🚏 Route stops: {len(path_nodes)}")

        for i in range(len(path_nodes)):
            node = path_nodes[i]
            current_coords = node.coordinates
            
            print(f"\n   📍 Stop {i+1}/{len(path_nodes)}:")
            
            # Determine category and station name
            if node.station_idx == -1:
                category_str = "Source"
                station_name = None
                visit_number = None
                print(f"      🚗 Type: Starting point")
            elif node.station_idx == -2:
                category_str = "Destination" 
                station_name = None
                visit_number = None
                print(f"      🏁 Type: Final destination")
            else:
                category_str = "Visiting_Charging_Station"
                coord_key = f"({current_coords[0]},{current_coords[1]})"
                
                # Use existing station mapping
                STATION_MAPPING = {
                    "(7.123456,80.123456)": "Miriswaththa_Station",
                    "(7.182689,79.961171)": "Minuwangoda_Station", 
                    "(7.148497,79.873276)": "Seeduwa_Station",
                }
                
                base_name = STATION_MAPPING.get(coord_key, f"Station_{node.station_idx}")
                
                if node.station_idx not in station_visit_counts:
                    station_visit_counts[node.station_idx] = 0
                station_visit_counts[node.station_idx] += 1
                visit_number = station_visit_counts[node.station_idx]
                
                if visit_number > 1:
                    station_name = f"{base_name}_Visit_{visit_number}"
                    print(f"      🔌 Type: Charging station (revisit #{visit_number})")
                else:
                    station_name = base_name
                    print(f"      🔌 Type: Charging station (first visit)")

            batt_arrival = current_batt
            range_arrival = batt_arrival * self.km_per_percent
            
            print(f"      🔋 Battery on arrival: {batt_arrival:.1f}% ({range_arrival:.1f} km range)")
            
            # Determine departure battery
            if category_str == "Visiting_Charging_Station":
                batt_departure = 100.0
                print(f"      ⚡ Charging to 100% battery")
            else:
                batt_departure = current_batt
            
            range_departure = batt_departure * self.km_per_percent
            print(f"      🔋 Battery on departure: {batt_departure:.1f}% ({range_departure:.1f} km range)")
            
            # Calculate distance to next stop
            dist_to_next = 0.0
            actual_distance = None
            
            if i < len(path_nodes) - 1:
                next_coords = path_nodes[i+1].coordinates
                next_node = path_nodes[i+1]
                next_name = "Destination" if next_node.station_idx == -2 else f"Station_{next_node.station_idx}"
                
                print(f"      🎯 Next stop: {next_name}")
                
                # Use actual distance if available from PathNode
                if hasattr(path_nodes[i+1], 'actual_distance_to_here') and path_nodes[i+1].actual_distance_to_here:
                    dist_to_next = path_nodes[i+1].actual_distance_to_here
                    actual_distance = dist_to_next
                    print(f"      📏 Distance to next stop: {dist_to_next:.1f} km (Google API)")
                else:
                    # Fallback to calculated distance
                    if node.station_idx >= 0 and path_nodes[i+1].station_idx >= 0:
                        # Station to station - use cached
                        dist_to_next = self.distance_calculator.get_distance_between_stations(
                            node.station_idx, path_nodes[i+1].station_idx
                        )
                        print(f"      📏 Distance to next stop: {dist_to_next:.1f} km (cached)")
                    else:
                        # Use geodesic with road factor
                        dist_to_next = self.calculate_distance(current_coords, next_coords) * 1.3
                        print(f"      📏 Distance to next stop: {dist_to_next:.1f} km (geodesic + road factor)")
                
                # Check if there's enough battery for next leg
                battery_needed = dist_to_next / self.km_per_percent + self.min_battery_reserve
                if batt_departure < battery_needed:
                    print(f"      ⚠️ WARNING: May not have enough battery for next leg!")
                    print(f"         Need: {battery_needed:.1f}%, Have: {batt_departure:.1f}%")
                else:
                    print(f"      ✅ Sufficient battery for next leg")
            else:
                print(f"      🏁 Final destination - no next stop")
            
            logged_stops.append(RouteStop(
                latitude=current_coords[0],
                longitude=current_coords[1],
                category=category_str,
                battery_arrival=round(batt_arrival, 2),
                range_arrival=round(range_arrival, 2),
                battery_departure=round(batt_departure, 2),
                range_departure=round(range_departure, 2),
                next_stop_distance=round(dist_to_next, 2),
                visiting_flag="Visit",
                station_name=station_name,
                visit_number=visit_number,
                actual_driving_distance=actual_distance
            ))
            
            # Update battery for next iteration
            if dist_to_next > 0:
                current_batt = batt_departure - (dist_to_next / self.km_per_percent)
                print(f"      ⚡ Battery after next leg: {current_batt:.1f}%")

        # Add unused stations
        visited_station_indices = set(node.station_idx for node in path_nodes if node.station_idx >= 0)
        unused_stations = len(self.charging_stations) - len(visited_station_indices)
        
        print(f"\n📊 Adding {unused_stations} unused stations to log...")
        
        STATION_MAPPING = {
            "(7.123456,80.123456)": "Miriswaththa_Station",
            "(7.182689,79.961171)": "Minuwangoda_Station", 
        }
        
        for i, station_coords_tuple in enumerate(self.charging_stations):
            if i not in visited_station_indices:
                coord_key = f"({station_coords_tuple[0]},{station_coords_tuple[1]})"
                station_name = STATION_MAPPING.get(coord_key, f"Station_{i}")
                
                logged_stops.append(RouteStop(
                    latitude=station_coords_tuple[0],
                    longitude=station_coords_tuple[1],
                    category="Charging_Station",
                    battery_arrival=0,
                    range_arrival=0,
                    battery_departure=0,
                    range_departure=0,
                    next_stop_distance=0,
                    visiting_flag="Not Visit",
                    station_name=station_name,
                    visit_number=None,
                    actual_driving_distance=None
                ))
        
        print(f"✅ Route log generation complete - {len(logged_stops)} total entries")
        return logged_stops
    
    def plan_route(self):
        """Main route planning method with Google API integration"""
        try:
            print(f"\n🚀 Starting comprehensive EV route planning...")
            print(f"   📍 Route: {self.source} → {self.destination}")
            print(f"   ⚡ Vehicle specs: {self.initial_battery_percent}% battery, {self.km_per_percent} km/% efficiency")
            print(f"   🌐 Google API available: {self.distance_calculator.google_enabled}")
            
            # Check if station matrix exists, create if needed
            if not self.distance_calculator.station_matrix:
                print(f"\n⚙️ Station matrix not found - creating new matrix...")
                self.distance_calculator.setup_station_matrix()
            else:
                print(f"✅ Station matrix loaded - {len(self.distance_calculator.station_matrix)} cached distances")
            
            optimal_route_tuple = self.find_optimal_route()
            
            if optimal_route_tuple:
                print(f"\n🎉 Route planning successful!")
                route_log_details = self.generate_route_log(optimal_route_tuple)
                charging_stops = sum(1 for stop in route_log_details 
                                   if stop.category == "Visiting_Charging_Station" and stop.visiting_flag == "Visit")
                unique_stations = len(set(stop.station_name for stop in route_log_details 
                                        if stop.category == "Visiting_Charging_Station" and stop.visiting_flag == "Visit"))
                
                distance_source = "Google API + Cached" if self.distance_calculator.google_enabled else "Geodesic + Road Factor"
                
                print(f"\n📊 Final route summary:")
                print(f"   🛣️ Total distance: {optimal_route_tuple[0]:.1f} km")
                print(f"   ⚡ Charging stops: {charging_stops}")
                print(f"   🚏 Unique stations visited: {unique_stations}")
                print(f"   🌐 Google API calls used: {self.distance_calculator.api_call_count}")
                print(f"   📏 Distance calculation method: {distance_source}")
                
                return RouteResponse(
                    success=True,
                    total_distance=round(optimal_route_tuple[0], 2),
                    route_details=route_log_details,
                    charging_stops_count=charging_stops,
                    unique_stations_visited=unique_stations,
                    google_api_calls_used=self.distance_calculator.api_call_count,
                    distance_source=distance_source
                )
            else:
                print(f"\n❌ Route planning failed!")
                print(f"   Possible reasons:")
                print(f"   - Destination unreachable with available charging stations")
                print(f"   - Max charging stops limit too restrictive")
                print(f"   - Battery level too low for any meaningful progress")
                print(f"   💡 Suggestions:")
                print(f"   - Increase max_charging_stops parameter")
                print(f"   - Check charging station locations")
                print(f"   - Verify vehicle efficiency settings")
                
                return RouteResponse(
                    success=False,
                    error_message="Could not find a feasible route. Try increasing max_charging_stops or check if destination is reachable.",
                    google_api_calls_used=self.distance_calculator.api_call_count,
                    distance_source="Failed"
                )
        except Exception as e:
            print(f"\n💥 Error in hybrid route planning: {e}")
            print(f"   🔍 Debug information:")
            print(f"   - API calls made: {getattr(self, 'distance_calculator', {}).get('api_call_count', 0) if hasattr(self, 'distance_calculator') else 0}")
            
# --- Enhanced API Functions with Detailed Logging ---
def integrated_plan_route_from_params(
    source: str,
    destination: str,
    battery: float,
    efficiency: float,
    stations_json: str,
    max_charging_stops: int = 10,
    google_api_key: Optional[str] = None,
    output_path: str = None,
    format_: str = 'json',
    enhance: bool = True,
    query: str = None
) -> str:
    """
    Enhanced route planning with Google API integration, smart station selection, and detailed logging
    """
    
    try:
        print(f"\n🎬 Starting integrated route planning with enhanced logging...")
        
        # Parse coordinates
        source_lat, source_long = map(float, source.split(','))
        dest_lat, dest_long = map(float, destination.split(','))
        
        print(f"📍 Parsed coordinates:")
        print(f"   Source: ({source_lat}, {source_long})")
        print(f"   Destination: ({dest_lat}, {dest_long})")

        # Load charging stations
        actual_charging_stations_coords = []
        if stations_json:
            try:
                parsed_stations = json.loads(stations_json)
                actual_charging_stations_coords = [
                    (float(station[0]), float(station[1]))
                    for station in parsed_stations if isinstance(station, list) and len(station) == 2
                ]
                print(f"✅ Loaded {len(actual_charging_stations_coords)} charging stations from JSON")
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                print(f"❌ Error parsing stations_json: {e}. Using default stations.")
                # Use default stations
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
                actual_charging_stations_coords = [tuple(coords) for coords in DEFAULT_CHARGING_STATIONS_COORDS]

        if not actual_charging_stations_coords:
            error_msg = "No charging stations available."
            print(f"❌ {error_msg}")
            return json.dumps({
                "success": False,
                "message": error_msg,
                "route_summary": [],
                "google_api_calls_used": 0,
                "distance_source": "None"
            }, indent=2)

        print(f"⚡ Planning parameters:")
        print(f"   Battery: {battery}%")
        print(f"   Efficiency: {efficiency} km/%")
        print(f"   Max stops: {max_charging_stops}")
        print(f"   Google API: {'Enabled' if google_api_key else 'Disabled'}")

        # Create integrated hybrid route planner
        planner_instance = IntegratedHybridEVRoutePlanner(
            source=(source_lat, source_long),
            destination=(dest_lat, dest_long),
            initial_battery_percent=battery,
            km_per_percent=efficiency,
            charging_stations=actual_charging_stations_coords,
            max_charging_stops=max_charging_stops,
            google_api_key=google_api_key
        )
        
        core_route_response = planner_instance.plan_route()
        
        # Format response with enhanced logging
        if core_route_response.success and core_route_response.route_details:
            visited_stops = [s for s in core_route_response.route_details if s.visiting_flag == "Visit"]
            
            print(f"\n📋 Formatting response for {len(visited_stops)} visited stops...")
            
            detailed_stops_output = []
            for stop in visited_stops:
                stop_output = {
                    "location": f"({stop.latitude}, {stop.longitude})",
                    "category": stop.category,
                    "battery_on_arrival_percent": stop.battery_arrival,
                    "battery_on_departure_percent": stop.battery_departure,
                    "next_stop_distance_km": stop.next_stop_distance
                }
                
                if stop.station_name:
                    stop_output["station_name"] = stop.station_name
                if stop.visit_number:
                    stop_output["visit_number"] = stop.visit_number
                if stop.actual_driving_distance:
                    stop_output["actual_driving_distance_km"] = stop.actual_driving_distance
                
                detailed_stops_output.append(stop_output)
            
            # Smart station selection summary
            source_candidates_info = []
            if hasattr(planner_instance, 'source_candidates') and planner_instance.source_candidates:
                print(f"📊 Including source candidates information...")
                for i, (station_idx, distance) in enumerate(planner_instance.source_candidates[:3]):
                    source_candidates_info.append({
                        "station_index": station_idx,
                        "distance_km": round(distance, 2),
                        "rank": i + 1,
                        "reason": "Farthest reachable with current battery"
                    })
            
            final_result_dict = {
                "success": True,
                "distance_km": core_route_response.total_distance,
                "message": "Integrated hybrid route planned successfully with enhanced decision logging.",
                "planned_charging_stops_count": core_route_response.charging_stops_count,
                "unique_stations_visited": core_route_response.unique_stations_visited,
                "google_api_calls_used": core_route_response.google_api_calls_used,
                "distance_calculation_source": core_route_response.distance_source,
                "route_summary": detailed_stops_output,
                "source_station_candidates": source_candidates_info,
                "optimization_features": {
                    "google_api_integration": google_api_key is not None,
                    "smart_station_selection": True,
                    "battery_based_routing": True,
                    "multi_visit_capability": True,
                    "cached_distances": True,
                    "detailed_decision_logging": True
                }
            }
            
            print(f"✅ Route planning completed successfully!")
            
        else:
            print(f"❌ Route planning failed")
            final_result_dict = {
                "success": False,
                "distance_km": None,
                "message": core_route_response.error_message or "Hybrid route planning failed.",
                "planned_charging_stops_count": 0,
                "unique_stations_visited": 0,
                "google_api_calls_used": core_route_response.google_api_calls_used or 0,
                "distance_calculation_source": core_route_response.distance_source or "Failed",
                "route_summary": [],
                "source_station_candidates": [],
                "optimization_features": {
                    "google_api_integration": google_api_key is not None,
                    "smart_station_selection": True,
                    "battery_based_routing": True,
                    "multi_visit_capability": True,
                    "cached_distances": True,
                    "detailed_decision_logging": True
                }
            }

        # Export data if requested
        if output_path and core_route_response.success:
            print(f"💾 Saving results to {output_path}...")
            file_content_str = ""
            if format_ == 'json':
                file_content_str = json.dumps(final_result_dict, indent=2)
            
            if file_content_str:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(file_content_str)
                final_result_dict["export_message"] = f"Data saved to: {output_path}"
                print(f"✅ Results saved successfully")

        return json.dumps(final_result_dict, indent=2)

    except Exception as e:
        error_msg = f"Integrated hybrid route planning failed: {str(e)}"
        print(f"💥 {error_msg}")
        error_response = {
            "success": False,
            "distance_km": None,
            "message": error_msg,
            "planned_charging_stops_count": 0,
            "unique_stations_visited": 0,
            "google_api_calls_used": 0,
            "distance_calculation_source": "Error",
            "route_summary": [],
            "source_station_candidates": [],
            "optimization_features": {
                "google_api_integration": google_api_key is not None,
                "smart_station_selection": True,
                "battery_based_routing": True,
                "multi_visit_capability": True,
                "cached_distances": True,
                "detailed_decision_logging": True
            }
        }
        return json.dumps(error_response, indent=2)

# --- Utility Functions for Setup ---
def setup_station_matrix_utility(stations_json: str, google_api_key: str, cache_dir: str = "cache"):
    """
    Utility function to set up the station distance matrix (run once)
    """
    try:
        print(f"🔧 Setting up station distance matrix utility...")
        
        # Parse stations
        parsed_stations = json.loads(stations_json)
        station_coords = [
            (float(station[0]), float(station[1]))
            for station in parsed_stations if isinstance(station, list) and len(station) == 2
        ]
        
        if not station_coords:
            return "Error: No valid stations found in JSON"
        
        print(f"📊 Found {len(station_coords)} valid stations")
        
        # Initialize distance calculator
        distance_calc = HybridDistanceCalculator(google_api_key, station_coords, cache_dir)
        
        # Set up matrix
        matrix = distance_calc.setup_station_matrix()
        
        if matrix:
            result_msg = f"Station matrix setup complete! API calls used: {distance_calc.api_call_count}, Matrix size: {len(matrix)} pairs"
            print(f"✅ {result_msg}")
            return result_msg
        else:
            return "Failed to set up station matrix"
            
    except Exception as e:
        error_msg = f"Error setting up station matrix: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

# --- Comprehensive Main Function with Test Scenarios ---
def main():
    """
    Main function to demonstrate the EV route planner with detailed logging
    Tests multiple scenarios with real Sri Lankan charging stations
    """
    
    print("=" * 80)
    print("🚗 INTEGRATED HYBRID EV ROUTE PLANNER DEMONSTRATION")
    print("=" * 80)
    
    # Define charging stations with proper names
    DEFAULT_CHARGING_STATIONS = [
        [7.123456, 80.123456],   # Miriswaththa_Station
        [7.148497, 79.873276],   # Seeduwa_Station  
        [7.182689, 79.961171],   # Minuwangoda_Station
        [7.222404, 80.017613],   # Divulapitiya_Station
        [7.222445, 80.017625],   # Katunayake_Station
        [7.120498, 79.983923],   # Udugampola_Station
        [7.006685, 79.958184],   # Kadawatha_Station
        [7.274298, 79.862597],   # Kochchikade_Station
        [6.960975, 79.880949],   # Paliyagoda_Station
        [6.837024, 79.903572],   # Boralesgamuwa_Station
        [6.877865, 79.939505],   # Thalawathugoda_Station
        [6.787022, 79.884759],   # Moratuwa_Station
        [6.915059, 79.881394],   # Borella_Station
        [6.847305, 80.102153],   # Padukka_Station
        [7.222348, 80.017553],   # Beruwala_Station
        [6.714853, 79.989208],   # Bandaragama_Station
        [7.222444, 80.017606],   # Maggona_Station
        [6.713372, 79.906452],   # Panadura_Station
        [7.8715, 80.011],        # Anamaduwa_Station
        [7.2845, 80.6375],       # Kandy_Station
        [6.9847, 81.0564],       # Badulla_Station
        [6.1528, 80.2239],       # Matara_Station (fixed coordinates)
        [8.4947, 80.1739],       # Pemaduwa_Station
        [7.5742, 79.8482],       # Chilaw_Station (fixed coordinates)
        [7.0094, 81.0565],       # Mahiyangana_Station
        [7.2531, 80.3453],       # Kegalle_Station
    ]
    
    stations_json = json.dumps(DEFAULT_CHARGING_STATIONS)
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Colombo to Kandy (Optimal Route)",
            "source": "6.9271,79.8612",      # Colombo
            "destination": "7.2906,80.6337",   # Kandy
            "battery": 75.0,
            "efficiency": 1.4,
            "description": "Classic long-distance route requiring strategic charging"
        },
        {
            "name": "Colombo to Galle (Low Battery Challenge)",
            "source": "6.9271,79.8612",      # Colombo
            "destination": "6.0535,80.2210",   # Galle
            "battery": 45.0,                   # Low battery
            "efficiency": 1.2,                 # Less efficient vehicle
            "description": "Low battery scenario requiring careful planning"
        },
        {
            "name": "Negombo to Badulla (Cross-Country)",
            "source": "7.2088,79.8358",      # Negombo
            "destination": "6.9847,81.0564",   # Badulla
            "battery": 85.0,
            "efficiency": 1.5,                 # More efficient vehicle
            "description": "Long cross-country route testing multi-stop capability"
        },
        {
            "name": "Short Local Trip (Direct Route)",
            "source": "6.9271,79.8612",      # Colombo
            "destination": "6.8378,79.9036",   # Panadura area
            "battery": 60.0,
            "efficiency": 1.3,
            "description": "Short trip that might not need charging"
        }
    ]
    
    # Run test scenarios
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n" + "=" * 80)
        print(f"🧪 TEST SCENARIO {i}: {scenario['name']}")
        print(f"📝 Description: {scenario['description']}")
        print(f"📍 Route: {scenario['source']} → {scenario['destination']}")
        print(f"🔋 Vehicle: {scenario['battery']}% battery, {scenario['efficiency']} km/% efficiency")
        print(f"🛣️  Theoretical range: {scenario['battery'] * scenario['efficiency']:.1f} km")
        print("=" * 80)
        
        try:
            # Run route planning (without Google API for demo)
            result = integrated_plan_route_from_params(
                source=scenario['source'],
                destination=scenario['destination'],
                battery=scenario['battery'],
                efficiency=scenario['efficiency'],
                stations_json=stations_json,
                max_charging_stops=6,
                google_api_key=None,  # Set to your API key for real testing
                output_path=None,
                format_='json',
                enhance=True
            )
            
            # Parse and display results
            result_data = json.loads(result)
            
            print(f"\n📊 SCENARIO {i} RESULTS:")
            print(f"   ✅ Success: {result_data['success']}")
            
            if result_data['success']:
                print(f"   🛣️  Total Distance: {result_data['distance_km']} km")
                print(f"   ⚡ Charging Stops: {result_data['planned_charging_stops_count']}")
                print(f"   🚏 Unique Stations: {result_data['unique_stations_visited']}")
                print(f"   🌐 API Calls: {result_data['google_api_calls_used']}")
                print(f"   📏 Distance Source: {result_data['distance_calculation_source']}")
                
                # Show route summary
                print(f"\n🗺️  Route Summary:")
                for j, stop in enumerate(result_data['route_summary']):
                    stop_num = j + 1
                    location = stop['location']
                    category = stop['category']
                    
                    if category == "Source":
                        print(f"   {stop_num}. 🚗 START: {location}")
                        print(f"        Battery: {stop['battery_on_departure_percent']}%")
                    elif category == "Destination":
                        print(f"   {stop_num}. 🏁 END: {location}")
                        print(f"        Battery on arrival: {stop['battery_on_arrival_percent']}%")
                    else:  # Charging station
                        station_name = stop.get('station_name', 'Unknown Station')
                        print(f"   {stop_num}. 🔌 CHARGE: {station_name}")
                        print(f"        Arrive: {stop['battery_on_arrival_percent']}% → Depart: {stop['battery_on_departure_percent']}%")
                        print(f"        Next leg: {stop['next_stop_distance_km']} km")
                
                # Show optimization insights
                if 'source_station_candidates' in result_data and result_data['source_station_candidates']:
                    print(f"\n🎯 Smart Station Selection:")
                    for candidate in result_data['source_station_candidates'][:3]:
                        print(f"   Rank {candidate['rank']}: Station_{candidate['station_index']} "
                              f"({candidate['distance_km']} km) - {candidate['reason']}")
            else:
                print(f"   ❌ Error: {result_data['message']}")
                print(f"   💡 This scenario demonstrates planning limitations")
            
        except Exception as e:
            print(f"   💥 Scenario failed with error: {str(e)}")
        
        # Add separator between scenarios
        if i < len(test_scenarios):
            print(f"\n{'⏭️  ' * 20}")
            input("Press Enter to continue to next scenario...")
    
    print(f"\n" + "=" * 80)
    print("🏆 DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\n📋 Key Insights from Testing:")
    print("• The system prioritizes farthest reachable stations for maximum progress")
    print("• Battery calculations include safety reserves to prevent stranding")
    print("• Multi-visit capability handles complex long-distance routes")
    print("• Google API integration provides accurate road distances (when enabled)")
    print("• Smart pruning eliminates inefficient route options early")
    print("• Detailed logging shows reasoning behind every decision")
    
    print(f"\n🔧 To enable Google API integration:")
    print("1. Get a Google Maps API key with Distance Matrix API enabled")
    print("2. Set google_api_key parameter in the function calls")
    print("3. Run setup_station_matrix_utility() once to cache station distances")
    
    print(f"\n💡 The system is designed for:")
    print("• Production EV route planning services")
    print("• Fleet management systems")
    print("• Navigation apps with charging optimization")
    print("• Research into EV routing algorithms")

# --- One-time setup utility function ---
def setup_demo_station_matrix():
    """
    One-time setup function to create station distance matrix with Google API
    Run this once if you have a Google API key
    """
    
    DEFAULT_CHARGING_STATIONS = [
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
    
    stations_json = json.dumps(DEFAULT_CHARGING_STATIONS)
    
    # Replace with your actual Google API key
    google_api_key = "AIzaSyALG5CigQX1KFqVkYxAD_2E6BvtNYcHQVY"
    
    print("Setting up station distance matrix...")
    result = setup_station_matrix_utility(stations_json, google_api_key)
    print(result)
    
    return result

if __name__ == "__main__":
    # Run the comprehensive demonstration
    main()
    
    # Uncomment to set up Google API matrix (requires valid API key)
    setup_demo_station_matrix()