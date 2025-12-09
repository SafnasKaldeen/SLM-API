# integrated_hybrid_ev_route_service.py

"""
Integrated Hybrid EV Route Planning Service with Google API Integration

- Minimal Google API usage (2 calls per route)
- Smart station selection based on battery range
- Multi-visit capability maintained
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
category: str # Source, Destination, Visiting_Charging_Station, Charging_Station
battery_arrival: float
range_arrival: float
battery_departure: float
range_departure: float
next_stop_distance: float
visiting_flag: str # Visit, Not Visit
station_name: Optional[str] = None
visit_number: Optional[int] = None
actual_driving_distance: Optional[float] = None # Real road distance from Google

class RouteResponse(BaseModel):
success: bool = Field(..., description="Whether a route was found")
total_distance: Optional[float] = Field(None, description="Total distance of the route in km")
route_details: Optional[List[RouteStop]] = Field(None, description="Detailed information about each stop")
error_message: Optional[str] = Field(None, description="Error message if route planning failed")
charging_stops_count: Optional[int] = Field(None, description="Number of charging stops")
unique_stations_visited: Optional[int] = Field(None, description="Number of unique stations visited")
google_api_calls_used: Optional[int] = Field(None, description="Number of Google API calls made")
distance_source: Optional[str] = Field(None, description="Source of distance calculations")

# --- Hybrid Distance Calculator ---

class HybridDistanceCalculator:
"""Minimal Google API usage with smart caching"""

    def __init__(self, api_key: Optional[str], company_stations: List[Tuple[float, float]], cache_dir: str = "cache"):
        self.google_enabled = api_key is not None
        if self.google_enabled:
            self.client = googlemaps.Client(key=api_key)
        self.company_stations = company_stations
        self.cache_dir = cache_dir
        self.api_call_count = 0
        self.station_matrix = {}

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        self._load_station_matrix()

    def _load_station_matrix(self):
        """Load pre-computed station distance matrix"""
        cache_file = os.path.join(self.cache_dir, "station_distance_matrix.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.station_matrix = cache_data.get('matrix', {})
                    print(f"Loaded {len(self.station_matrix)} cached station distance pairs")
            except Exception as e:
                print(f"Error loading cache: {e}")

    def setup_station_matrix(self):
        """ONE-TIME SETUP: Pre-compute station-to-station distances"""
        if not self.google_enabled:
            print("Google API not available. Using geodesic distances with road factor.")
            return self._setup_geodesic_matrix()

        print("Setting up station distance matrix with Google API...")
        matrix = {}

        try:
            origins = [f"{lat},{lng}" for lat, lng in self.company_stations]

            # Batch API call for all stations
            distance_result = self.client.distance_matrix(
                origins=origins,
                destinations=origins,
                mode="driving",
                units="metric"
            )

            self.api_call_count += 1

            # Parse and store results
            for i, origin_row in enumerate(distance_result['rows']):
                for j, element in enumerate(origin_row['elements']):
                    if element['status'] == 'OK' and i != j:
                        distance_km = element['distance']['value'] / 1000
                        duration_sec = element['duration']['value']
                        matrix[(i, j)] = {
                            'distance_km': distance_km,
                            'duration_sec': duration_sec
                        }

            # Save to cache
            cache_file = os.path.join(self.cache_dir, "station_distance_matrix.pkl")
            cache_data = {
                'matrix': matrix,
                'created_with_google': True,
                'station_count': len(self.company_stations)
            }

            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

            self.station_matrix = matrix
            print(f"Station matrix setup complete. API calls used: {self.api_call_count}")
            return matrix

        except Exception as e:
            print(f"Error with Google API, falling back to geodesic: {e}")
            return self._setup_geodesic_matrix()

    def _setup_geodesic_matrix(self):
        """Fallback: Create matrix using geodesic distances with road factor"""
        matrix = {}
        for i, station_i in enumerate(self.company_stations):
            for j, station_j in enumerate(self.company_stations):
                if i != j:
                    geodesic_dist = geodesic(station_i, station_j).km
                    road_distance = geodesic_dist * 1.3  # Road factor
                    matrix[(i, j)] = {
                        'distance_km': road_distance,
                        'duration_sec': road_distance * 60  # Rough time estimate
                    }

        self.station_matrix = matrix
        print(f"Geodesic matrix created for {len(self.company_stations)} stations")
        return matrix

    def get_reachable_stations_from_source(self, source: Tuple[float, float],
                                         current_battery: float,
                                         km_per_percent: float,
                                         min_battery_reserve: float = 5.0) -> List[Tuple[int, float]]:
        """
        Find stations reachable from source, prioritizing the farthest station
        that can be reached with current battery
        """
        max_range = (current_battery - min_battery_reserve) * km_per_percent
        reachable_stations = []

        # First, find reachable stations using geodesic distance (quick filter)
        for i, station in enumerate(self.company_stations):
            geodesic_dist = geodesic(source, station).km
            # Use conservative estimate (geodesic * 1.4 for road factor)
            estimated_road_distance = geodesic_dist * 1.4

            if estimated_road_distance <= max_range:
                reachable_stations.append((i, geodesic_dist))

        if not reachable_stations:
            return []

        # If Google API available, get exact distances for top candidates
        if self.google_enabled and len(reachable_stations) > 1:
            # Sort by geodesic distance and take top candidates
            reachable_stations.sort(key=lambda x: x[1], reverse=True)  # Farthest first
            top_candidates = reachable_stations[:min(5, len(reachable_stations))]

            # Get exact Google distances
            exact_distances = self.get_user_to_stations_distance(source, [i for i, _ in top_candidates])

            # Filter by actual reachability and return farthest reachable
            final_reachable = []
            for station_idx, geodesic_dist in top_candidates:
                if station_idx in exact_distances:
                    actual_distance = exact_distances[station_idx]
                    if actual_distance <= max_range:
                        final_reachable.append((station_idx, actual_distance))

            # Return sorted by actual distance (farthest first)
            return sorted(final_reachable, key=lambda x: x[1], reverse=True)

        # Fallback: return sorted by geodesic distance (farthest first)
        return sorted(reachable_stations, key=lambda x: x[1], reverse=True)

    def get_stations_for_destination(self, destination: Tuple[float, float],
                                   top_k: int = 4) -> List[Tuple[int, float]]:
        """Find nearest stations to destination (any station can be final)"""
        distances = []
        for i, station in enumerate(self.company_stations):
            dist = geodesic(destination, station).km
            distances.append((i, dist))

        # Sort by distance (nearest first for destination)
        distances.sort(key=lambda x: x[1])
        return distances[:top_k]

    def get_user_to_stations_distance(self, user_location: Tuple[float, float],
                                    station_indices: List[int]) -> Dict[int, float]:
        """Get actual driving distances from user to stations (1 API call)"""
        if not self.google_enabled or not station_indices:
            # Fallback to geodesic with road factor
            return {i: geodesic(user_location, self.company_stations[i]).km * 1.3
                   for i in station_indices}

        origins = [f"{user_location[0]},{user_location[1]}"]
        destinations = [f"{self.company_stations[i][0]},{self.company_stations[i][1]}"
                       for i in station_indices]

        try:
            distance_result = self.client.distance_matrix(
                origins=origins,
                destinations=destinations,
                mode="driving",
                units="metric"
            )

            self.api_call_count += 1

            distances = {}
            if distance_result['rows']:
                row = distance_result['rows'][0]
                for i, element in enumerate(row['elements']):
                    if element['status'] == 'OK':
                        station_idx = station_indices[i]
                        distances[station_idx] = element['distance']['value'] / 1000

            return distances

        except Exception as e:
            print(f"Google API error, using geodesic fallback: {e}")
            return {i: geodesic(user_location, self.company_stations[i]).km * 1.3
                   for i in station_indices}

    def get_stations_to_user_distance(self, station_indices: List[int],
                                    user_location: Tuple[float, float]) -> Dict[int, float]:
        """Get actual driving distances from stations to user (1 API call)"""
        if not self.google_enabled or not station_indices:
            return {i: geodesic(self.company_stations[i], user_location).km * 1.3
                   for i in station_indices}

        origins = [f"{self.company_stations[i][0]},{self.company_stations[i][1]}"
                   for i in station_indices]
        destinations = [f"{user_location[0]},{user_location[1]}"]

        try:
            distance_result = self.client.distance_matrix(
                origins=origins,
                destinations=destinations,
                mode="driving",
                units="metric"
            )

            self.api_call_count += 1

            distances = {}
            for i, row in enumerate(distance_result['rows']):
                if row['elements'] and row['elements'][0]['status'] == 'OK':
                    station_idx = station_indices[i]
                    distances[station_idx] = row['elements'][0]['distance']['value'] / 1000

            return distances

        except Exception as e:
            print(f"Google API error, using geodesic fallback: {e}")
            return {i: geodesic(self.company_stations[i], user_location).km * 1.3
                   for i in station_indices}

    def get_distance_between_stations(self, station_i: int, station_j: int) -> float:
        """Get cached distance between stations (NO API call)"""
        if (station_i, station_j) in self.station_matrix:
            return self.station_matrix[(station_i, station_j)]['distance_km']
        elif (station_j, station_i) in self.station_matrix:
            return self.station_matrix[(station_j, station_i)]['distance_km']
        else:
            # Fallback to geodesic
            dist = geodesic(self.company_stations[station_i],
                          self.company_stations[station_j]).km
            return dist * 1.3

# --- Enhanced PathNode (unchanged) ---

@dataclass
class PathNode:
station_idx: int
coordinates: Tuple[float, float]
visit_count: int = 1
actual_distance_to_here: Optional[float] = None # Google API distance

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

# --- Enhanced EV Route Planner ---

class IntegratedHybridEVRoutePlanner:
"""Enhanced EV Route Planner with Google API integration and smart station selection"""

    def __init__(self,
                 source: Tuple[float, float],
                 destination: Tuple[float, float],
                 initial_battery_percent: float,
                 km_per_percent: float,
                 charging_stations: List[Tuple[float, float]],
                 max_charging_stops: int = 10,
                 min_battery_reserve: float = 5.0,
                 google_api_key: Optional[str] = None):

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
        print("Selecting candidate stations...")

        # Get reachable stations from source (prioritize farthest)
        self.source_candidates = self.distance_calculator.get_reachable_stations_from_source(
            self.source, self.initial_battery_percent, self.km_per_percent, self.min_battery_reserve
        )

        # Get stations near destination (any station can be final)
        self.dest_candidates = self.distance_calculator.get_stations_for_destination(
            self.destination, top_k=4
        )

        print(f"Source candidates (farthest first): {len(self.source_candidates)} stations")
        print(f"Destination candidates: {len(self.dest_candidates)} stations")

        if self.source_candidates:
            print(f"Farthest reachable station from source: Station_{self.source_candidates[0][0]} at {self.source_candidates[0][1]:.2f} km")

    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate distance - uses Google API results when available, otherwise geodesic"""
        return geodesic(point1, point2).km

    def get_station_visit_count(self, path: List[PathNode], station_idx: int) -> int:
        """Count visits to a station in current path"""
        return sum(1 for node in path if node.station_idx == station_idx)

    def is_path_too_long(self, path: List[PathNode]) -> bool:
        """Check if path exceeds maximum charging stops"""
        charging_stops = sum(1 for node in path if node.station_idx >= 0)
        return charging_stops >= self.max_charging_stops

    def find_optimal_route(self):
        """Enhanced A* with hybrid distance calculation and smart station selection"""

        def heuristic(point):
            return self.calculate_distance(point, self.destination)

        # Initialize with source
        start_point = self.source
        start_f_score = heuristic(start_point)
        initial_path = [PathNode(-1, start_point)]

        queue = [(start_f_score, 0, start_point, self.initial_battery_percent, initial_path)]
        visited_states = {}
        best_route = None
        nodes_explored = 0
        max_nodes = 30000  # Reduced for efficiency

        print(f"Starting route search with Google API integration...")
        print(f"Google API enabled: {self.distance_calculator.google_enabled}")

        while queue and nodes_explored < max_nodes:
            f_score, g_score, current_loc, current_battery, path = heapq.heappop(queue)
            nodes_explored += 1

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
            dist_to_dest = self.calculate_distance(current_loc, self.destination)

            # Use actual Google distance for final leg if at a station
            current_node = path[-1]
            if (current_node.station_idx >= 0 and
                self.distance_calculator.google_enabled and
                current_node.station_idx in [i for i, _ in self.dest_candidates]):

                # Get exact distance to destination
                station_indices = [current_node.station_idx]
                actual_distances = self.distance_calculator.get_stations_to_user_distance(
                    station_indices, self.destination
                )
                if current_node.station_idx in actual_distances:
                    dist_to_dest = actual_distances[current_node.station_idx]

            required_battery = dist_to_dest / self.km_per_percent + self.min_battery_reserve

            if current_battery >= required_battery:
                total_distance = g_score + dist_to_dest
                if best_route is None or total_distance < best_route[0]:
                    destination_node = PathNode(-2, self.destination)
                    destination_node.actual_distance_to_here = dist_to_dest
                    best_route = (total_distance, path + [destination_node])
                    charging_stops = sum(1 for n in path if n.station_idx >= 0)
                    print(f"Found route: {total_distance:.2f} km, {charging_stops} stops, API calls: {self.distance_calculator.api_call_count}")

            # Explore charging stations with smart selection
            candidate_stations = []

            # If at source, prioritize farthest reachable stations
            if current_node.station_idx == -1:  # At source
                candidate_stations = [(idx, dist) for idx, dist in self.source_candidates[:3]]  # Top 3 farthest
            else:
                # At a charging station, consider all stations
                for idx, station_coords in enumerate(self.charging_stations):
                    if idx != current_node.station_idx:  # Don't revisit same station immediately
                        geodesic_dist = geodesic(current_loc, tuple(station_coords)).km
                        candidate_stations.append((idx, geodesic_dist))

                # Sort by distance and take reasonable candidates
                candidate_stations.sort(key=lambda x: x[1])
                candidate_stations = candidate_stations[:8]  # Top 8 nearest

            for idx, estimated_dist in candidate_stations:
                station_tuple = tuple(self.charging_stations[idx])

                # Use cached distance between stations if available
                if current_node.station_idx >= 0:  # Currently at a station
                    dist_to_station = self.distance_calculator.get_distance_between_stations(
                        current_node.station_idx, idx
                    )
                else:
                    # From source to station - will be calculated with Google if available
                    dist_to_station = estimated_dist * 1.3  # Road factor

                required_battery_to_station = dist_to_station / self.km_per_percent + self.min_battery_reserve

                if current_battery < required_battery_to_station:
                    continue

                # Multi-visit logic
                current_visit_count = self.get_station_visit_count(path, idx)
                max_visits_per_station = 3
                if current_visit_count >= max_visits_per_station:
                    continue

                # Penalties
                revisit_penalty = current_visit_count * 15
                new_g_score = g_score + dist_to_station + revisit_penalty

                if best_route and new_g_score >= best_route[0]:
                    continue

                # Create new path node
                station_node = PathNode(idx, station_tuple, current_visit_count + 1)
                station_node.actual_distance_to_here = dist_to_station
                new_path = path + [station_node]

                battery_after_charge = 100.0
                h_val = heuristic(station_tuple)
                new_f_score = new_g_score + h_val

                heapq.heappush(queue, (new_f_score, new_g_score, station_tuple, battery_after_charge, new_path))

        print(f"Search complete. Explored {nodes_explored} nodes, API calls: {self.distance_calculator.api_call_count}")
        return best_route

    def generate_route_log(self, route_tuple: Optional[Tuple[float, List[PathNode]]]) -> List[RouteStop]:
        """Generate detailed route log with actual Google distances when available"""
        if not route_tuple:
            return []

        total_distance, path_nodes = route_tuple
        logged_stops = []
        current_batt = self.initial_battery_percent
        station_visit_counts = {}

        for i in range(len(path_nodes)):
            node = path_nodes[i]
            current_coords = node.coordinates

            # Determine category and station name
            if node.station_idx == -1:
                category_str = "Source"
                station_name = None
                visit_number = None
            elif node.station_idx == -2:
                category_str = "Destination"
                station_name = None
                visit_number = None
            else:
                category_str = "Visiting_Charging_Station"
                coord_key = f"({current_coords[0]},{current_coords[1]})"

                # Use existing station mapping
                STATION_MAPPING = {
                    "(7.123456,80.123456)": "Miriswaththa_Station",
                    "(7.182689,79.961171)": "Minuwangoda_Station",
                    "(7.148497,79.873276)": "Seeduwa_Station",
                    # ... (use your existing mapping)
                }

                base_name = STATION_MAPPING.get(coord_key, f"Station_{node.station_idx}")

                if node.station_idx not in station_visit_counts:
                    station_visit_counts[node.station_idx] = 0
                station_visit_counts[node.station_idx] += 1
                visit_number = station_visit_counts[node.station_idx]

                if visit_number > 1:
                    station_name = f"{base_name}_Visit_{visit_number}"
                else:
                    station_name = base_name

            batt_arrival = current_batt
            range_arrival = batt_arrival * self.km_per_percent

            # Determine departure battery
            if category_str == "Visiting_Charging_Station":
                batt_departure = 100.0
            else:
                batt_departure = current_batt

            range_departure = batt_departure * self.km_per_percent

            # Calculate distance to next stop
            dist_to_next = 0.0
            actual_distance = None

            if i < len(path_nodes) - 1:
                next_coords = path_nodes[i+1].coordinates

                # Use actual distance if available from PathNode
                if hasattr(path_nodes[i+1], 'actual_distance_to_here') and path_nodes[i+1].actual_distance_to_here:
                    dist_to_next = path_nodes[i+1].actual_distance_to_here
                    actual_distance = dist_to_next
                else:
                    # Fallback to calculated distance
                    if node.station_idx >= 0 and path_nodes[i+1].station_idx >= 0:
                        # Station to station - use cached
                        dist_to_next = self.distance_calculator.get_distance_between_stations(
                            node.station_idx, path_nodes[i+1].station_idx
                        )
                    else:
                        # Use geodesic with road factor
                        dist_to_next = self.calculate_distance(current_coords, next_coords) * 1.3

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

        # Add unused stations
        visited_station_indices = set(node.station_idx for node in path_nodes if node.station_idx >= 0)

        STATION_MAPPING = {
            "(7.123456,80.123456)": "Miriswaththa_Station",
            "(7.182689,79.961171)": "Minuwangoda_Station",
            # ... (add all your mappings)
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

        return logged_stops

    def plan_route(self) -> RouteResponse:
        """Main route planning method with Google API integration"""
        try:
            print(f"Planning hybrid route from {self.source} to {self.destination}")
            print(f"Initial battery: {self.initial_battery_percent}%, Efficiency: {self.km_per_percent} km/%")
            print(f"Google API available: {self.distance_calculator.google_enabled}")

            # Check if station matrix exists, create if needed
            if not self.distance_calculator.station_matrix:
                print("No station matrix found. Setting up...")
                self.distance_calculator.setup_station_matrix()

            optimal_route_tuple = self.find_optimal_route()

            if optimal_route_tuple:
                route_log_details = self.generate_route_log(optimal_route_tuple)
                charging_stops = sum(1 for stop in route_log_details
                                   if stop.category == "Visiting_Charging_Station" and stop.visiting_flag == "Visit")
                unique_stations = len(set(stop.station_name for stop in route_log_details
                                        if stop.category == "Visiting_Charging_Station" and stop.visiting_flag == "Visit"))

                distance_source = "Google API + Cached" if self.distance_calculator.google_enabled else "Geodesic + Road Factor"

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
                return RouteResponse(
                    success=False,
                    error_message="Could not find a feasible route. Try increasing max_charging_stops or check if destination is reachable.",
                    google_api_calls_used=self.distance_calculator.api_call_count,
                    distance_source="Failed"
                )
        except Exception as e:
            print(f"Error in hybrid route planning: {e}")
            return RouteResponse(
                success=False,
                error_message=f"Error planning route: {str(e)}",
                google_api_calls_used=self.distance_calculator.api_call_count if hasattr(self, 'distance_calculator') else 0,
                distance_source="Error"
            )

# --- Enhanced API Functions ---

def integrated*plan_route_from_params(
source: str,
destination: str,
battery: float,
efficiency: float,
stations_json: str,
max_charging_stops: int = 10,
google_api_key: Optional[str] = None,
output_path: str = None,
format*: str = 'json',
enhance: bool = True,
query: str = None
) -> str:
"""
Enhanced route planning with Google API integration and smart station selection

    Args:
        source: "lat,lng" string for source location
        destination: "lat,lng" string for destination location
        battery: Initial battery percentage
        efficiency: km per battery percentage
        stations_json: JSON string of charging stations
        max_charging_stops: Maximum charging stops allowed
        google_api_key: Google API key for enhanced routing
        output_path: Path to save results
        format_: Output format (json)
        enhance: Enable enhanced features
        query: Additional query parameters

    Returns:
        JSON string with route planning results
    """

    try:
        # Parse coordinates
        source_lat, source_long = map(float, source.split(','))
        dest_lat, dest_long = map(float, destination.split(','))

        # Load charging stations
        actual_charging_stations_coords = []
        if stations_json:
            try:
                parsed_stations = json.loads(stations_json)
                actual_charging_stations_coords = [
                    (float(station[0]), float(station[1]))
                    for station in parsed_stations if isinstance(station, list) and len(station) == 2
                ]
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                print(f"Error parsing stations_json: {e}. Using default stations.")
                # Use default stations from your original code
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
        else:
            # Use default if no stations provided
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
            return json.dumps({
                "success": False,
                "message": "No charging stations available.",
                "route_summary": [],
                "google_api_calls_used": 0,
                "distance_source": "None"
            }, indent=2)

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

        # Format response
        if core_route_response.success and core_route_response.route_details:
            visited_stops = [s for s in core_route_response.route_details if s.visiting_flag == "Visit"]

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
                "message": "Integrated hybrid route planned successfully with Google API optimization.",
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
                    "cached_distances": True
                }
            }
        else:
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
                    "cached_distances": True
                }
            }

        # Export data if requested
        if output_path and core_route_response.success:
            file_content_str = ""
            if format_ == 'json':
                file_content_str = json.dumps(final_result_dict, indent=2)

            if file_content_str:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(file_content_str)
                final_result_dict["export_message"] = f"Data saved to: {output_path}"

        return json.dumps(final_result_dict, indent=2)

    except Exception as e:
        error_response = {
            "success": False,
            "distance_km": None,
            "message": f"Integrated hybrid route planning failed: {str(e)}",
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
                "cached_distances": True
            }
        }
        return json.dumps(error_response, indent=2)

# --- Utility Functions for Setup ---

def setup_station_matrix_utility(stations_json: str, google_api_key: str, cache_dir: str = "cache"):
"""
Utility function to set up the station distance matrix (run once)

    Args:
        stations_json: JSON string of charging stations
        google_api_key: Google API key
        cache_dir: Directory to store cache files

    Returns:
        Success message with API usage statistics
    """
    try:
        # Parse stations
        parsed_stations = json.loads(stations_json)
        station_coords = [
            (float(station[0]), float(station[1]))
            for station in parsed_stations if isinstance(station, list) and len(station) == 2
        ]

        if not station_coords:
            return "Error: No valid stations found in JSON"

        # Initialize distance calculator
        distance_calc = HybridDistanceCalculator(google_api_key, station_coords, cache_dir)

        # Set up matrix
        matrix = distance_calc.setup_station_matrix()

        if matrix:
            return f"Station matrix setup complete! API calls used: {distance_calc.api_call_count}, Matrix size: {len(matrix)} pairs"
        else:
            return "Failed to set up station matrix"

    except Exception as e:
        return f"Error setting up station matrix: {str(e)}"

# --- Example Usage ---

def example_usage():
"""Example of how to use the integrated hybrid planner"""

    # Your charging stations
    stations = [
        [7.123456, 80.123456], [7.148497, 79.873276], [7.182689, 79.961171],
        [7.222404, 80.017613], [7.222445, 80.017625], [7.120498, 79.983923]
        # ... add all your stations
    ]

    stations_json = json.dumps(stations)

    # Route planning
    result = integrated_plan_route_from_params(
        source="6.9271,79.8612",  # Colombo
        destination="7.2906,80.6337",  # Kandy
        battery=75.0,
        efficiency=1.4,
        stations_json=stations_json,
        max_charging_stops=8,
        google_api_key="YOUR_GOOGLE_API_KEY_HERE"  # Replace with actual key
    )

    print("Route Planning Result:")
    print(result)

    return result

if **name** == "**main**": # Uncomment to run example # example_usage()

    # Uncomment to set up station matrix (run once)
    # setup_result = setup_station_matrix_utility(
    #     stations_json='[[7.123456, 80.123456], [7.148497, 79.873276]]',  # Your stations
    #     google_api_key="YOUR_GOOGLE_API_KEY_HERE"
    # )
    # print(setup_result)
    pass
