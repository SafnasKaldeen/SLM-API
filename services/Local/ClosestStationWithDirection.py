# services/closest_station_service.py
"""
Local Closest Station Service
Adapted from Snowflake version to work with local data
"""

import pandas as pd
import numpy as np
from geopy.distance import geodesic
import json
import os
from datetime import datetime
from math import degrees, acos, sqrt
from typing import Dict, List, Tuple, Optional
import csv

# Sample GPS data structure for testing
SAMPLE_GPS_DATA = [
    {"TBOX_ID": "EV001", "LAT": 6.9271, "LONG": 79.8612, "TIMESTAMP": "2024-01-01 10:00:00"},
    {"TBOX_ID": "EV001", "LAT": 6.9275, "LONG": 79.8615, "TIMESTAMP": "2024-01-01 10:01:00"},
    {"TBOX_ID": "EV001", "LAT": 6.9278, "LONG": 79.8618, "TIMESTAMP": "2024-01-01 10:02:00"},
    {"TBOX_ID": "EV002", "LAT": 7.2906, "LONG": 80.6337, "TIMESTAMP": "2024-01-01 10:00:00"},
    {"TBOX_ID": "EV002", "LAT": 7.2908, "LONG": 80.6340, "TIMESTAMP": "2024-01-01 10:01:00"},
    {"TBOX_ID": "EV002", "LAT": 7.2910, "LONG": 80.6342, "TIMESTAMP": "2024-01-01 10:02:00"},
]

# Predefined charging/swap stations in Sri Lanka
PREDEFINED_STATIONS = [
    {
        "station_id": "ST001",
        "station_name": "Colombo Fort Station",
        "lat": 6.9319444,
        "long": 79.8477778,
        "district": "Colombo",
        "address": "Fort, Colombo"
    },
    {
        "station_id": "ST002", 
        "station_name": "Galle Face Station",
        "lat": 6.9147222,
        "long": 79.8441667,
        "district": "Colombo",
        "address": "Galle Face, Colombo"
    },
    {
        "station_id": "ST003",
        "station_name": "Kandy Station",
        "lat": 7.2906,
        "long": 80.6337,
        "district": "Kandy", 
        "address": "Kandy City Center"
    },
    {
        "station_id": "ST004",
        "station_name": "Negombo Station",
        "lat": 7.2084,
        "long": 79.8358,
        "district": "Gampaha",
        "address": "Negombo Town"
    },
    {
        "station_id": "ST005",
        "station_name": "Gampaha Station", 
        "lat": 7.0873,
        "long": 79.9990,
        "district": "Gampaha",
        "address": "Gampaha Center"
    },
    {
        "station_id": "ST006",
        "station_name": "Kalutara Station",
        "lat": 6.5854,
        "long": 79.9607,
        "district": "Kalutara", 
        "address": "Kalutara South"
    },
    {
        "station_id": "ST007",
        "station_name": "Matara Station",
        "lat": 5.9549,
        "long": 80.5550,
        "district": "Matara",
        "address": "Matara City"
    },
    {
        "station_id": "ST008",
        "station_name": "Jaffna Station",
        "lat": 9.6615,
        "long": 80.0255,
        "district": "Jaffna",
        "address": "Jaffna Town"
    }
]

class LocalClosestStationService:
    """Service to find closest stations with directional analysis"""
    
    def __init__(self, gps_data_file: Optional[str] = None, stations_file: Optional[str] = None):
        """
        Initialize with optional custom data files
        Args:
            gps_data_file: Path to CSV file with GPS data
            stations_file: Path to JSON file with station data
        """
        self.gps_data = self._load_gps_data(gps_data_file)
        self.stations = self._load_stations_data(stations_file)
    
    def _load_gps_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load GPS data from file or use sample data"""
        if file_path and os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                # Ensure required columns exist
                required_cols = ['TBOX_ID', 'LAT', 'LONG', 'TIMESTAMP']
                if all(col in df.columns for col in required_cols):
                    return df
                else:
                    print(f"CSV missing required columns: {required_cols}")
            except Exception as e:
                print(f"Error loading GPS data: {e}")
        
        # Return sample data as DataFrame
        return pd.DataFrame(SAMPLE_GPS_DATA)
    
    def _load_stations_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load stations data from file or use predefined data"""
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    stations_data = json.load(f)
                return pd.DataFrame(stations_data)
            except Exception as e:
                print(f"Error loading stations data: {e}")
        
        # Return predefined stations as DataFrame
        return pd.DataFrame(PREDEFINED_STATIONS)
    
    def calculate_direction_vector(self, coords_sequence: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """Calculate normalized direction vector from coordinate sequence"""
        if len(coords_sequence) < 2:
            return None
        
        # Use last two points to determine direction
        x0, y0 = coords_sequence[-2]
        x1, y1 = coords_sequence[-1]
        
        dx, dy = x1 - x0, y1 - y0
        if dx == 0 and dy == 0:
            return None
        
        # Normalize
        length = sqrt(dx**2 + dy**2)
        return (dx / length, dy / length)
    
    def find_best_station(
        self, 
        current_pos: Tuple[float, float], 
        direction_vec: Optional[Tuple[float, float]] = None,
        max_radius_km: float = 5.0
    ) -> Dict:
        """
        Find the best charging station based on distance and direction
        
        Args:
            current_pos: (lat, lon) of current position
            direction_vec: (dx, dy) normalized direction vector
            max_radius_km: Maximum search radius in kilometers
        
        Returns:
            Dict with selected station info and analysis
        """
        y1, x1 = current_pos  # lat, lon
        
        # Find stations within radius
        within_radius = []
        closest_station = None
        closest_distance = float("inf")
        
        for _, station in self.stations.iterrows():
            station_coords = (station["lat"], station["long"])
            scooter_coords = (y1, x1)
            dist = geodesic(scooter_coords, station_coords).km
            
            if dist <= max_radius_km:
                within_radius.append((dist, station))
            
            if dist < closest_distance:
                closest_distance = dist
                closest_station = station
        
        selected_station = None
        selection_reason = ""
        
        if within_radius and direction_vec:
            # Choose directionally aligned among close stations
            dx, dy = direction_vec
            best_score = float("inf")
            max_dist = max([x[0] for x in within_radius])
            
            for dist, station in within_radius:
                # Calculate angle between direction and station vector
                station_vec = np.array([station["long"] - x1, station["lat"] - y1])
                dot = dx * station_vec[0] + dy * station_vec[1]
                norm_product = np.linalg.norm([dx, dy]) * np.linalg.norm(station_vec)
                
                if norm_product > 0:
                    cos_angle = max(min(dot / norm_product, 1.0), -1.0)
                    angle = degrees(acos(cos_angle))
                    
                    # Combined score: 50% angle alignment, 50% distance
                    score = 0.5 * (angle / 180.0) + 0.5 * (dist / max_dist)
                    
                    if score < best_score:
                        best_score = score
                        selected_station = station
                        selection_reason = f"Best directional alignment (angle: {angle:.1f}°, distance: {dist:.2f}km)"
        
        elif within_radius:
            # No direction info, choose closest within radius
            within_radius.sort(key=lambda x: x[0])
            selected_station = within_radius[0][1]
            selection_reason = f"Closest within {max_radius_km}km radius"
            
        elif direction_vec:
            # No stations in radius, find best directional match globally
            dx, dy = direction_vec
            best_station = None
            best_score = float("inf")
            distances = []
            
            for _, s in self.stations.iterrows():
                station_vec = np.array([s["long"] - x1, s["lat"] - y1])
                distances.append(np.linalg.norm(station_vec))
            
            max_dist = max(distances) if distances else 1
            
            for _, station in self.stations.iterrows():
                station_vec = np.array([station["long"] - x1, station["lat"] - y1])
                if np.linalg.norm(station_vec) < 1e-6:
                    continue
                
                dot = dx * station_vec[0] + dy * station_vec[1]
                norm_product = np.linalg.norm([dx, dy]) * np.linalg.norm(station_vec)
                
                if norm_product > 0:
                    cos_angle = max(min(dot / norm_product, 1.0), -1.0)
                    angle = degrees(acos(cos_angle))
                    
                    distance = geodesic((y1, x1), (station["lat"], station["long"])).km
                    norm_angle = angle / 180.0
                    norm_distance = distance / max_dist if max_dist else 0
                    score = 0.5 * norm_angle + 0.5 * norm_distance
                    
                    if score < best_score:
                        best_score = score
                        best_station = station
                        best_distance = distance
            
            # Compare directional vs closest
            if best_distance > 4 * closest_distance:
                selected_station = closest_station
                selection_reason = f"Closest station (directional too far: {best_distance:.2f}km vs {closest_distance:.2f}km)"
            else:
                selected_station = best_station
                selection_reason = f"Best directional match (distance: {best_distance:.2f}km)"
        else:
            # Fallback to closest station
            selected_station = closest_station
            selection_reason = f"Closest available station"
        
        if selected_station is not None:
            final_distance = geodesic((y1, x1), (selected_station["lat"], selected_station["long"])).km
            return {
                "success": True,
                "station": {
                    "station_id": selected_station["station_id"],
                    "station_name": selected_station["station_name"],
                    "lat": selected_station["lat"],
                    "long": selected_station["long"],
                    "district": selected_station["district"],
                    "address": selected_station["address"],
                    "distance_km": round(final_distance, 2)
                },
                "selection_reason": selection_reason,
                "current_position": {"lat": y1, "long": x1},
                "stations_in_radius": len(within_radius),
                "closest_distance_km": round(closest_distance, 2)
            }
        else:
            return {
                "success": False,
                "error": "No suitable station found",
                "current_position": {"lat": y1, "long": x1}
            }
    
    def process_vehicle_data(self, output_file: Optional[str] = None) -> List[Dict]:
        """
        Process all vehicle GPS data to find recommended stations
        
        Args:
            output_file: Optional path to save results as JSON
            
        Returns:
            List of results for each vehicle
        """
        results = []
        path_traces = []
        direction_vectors = []
        selected_stations = []
        
        for tbox_id in self.gps_data["TBOX_ID"].unique():
            vehicle_data = self.gps_data[self.gps_data["TBOX_ID"] == tbox_id].sort_values("TIMESTAMP")
            
            if len(vehicle_data) < 2:
                continue
            
            # Use last 30 points or all available data
            last_n = vehicle_data.tail(30)
            
            # Extract coordinates
            coords_sequence = list(zip(last_n["LONG"].values, last_n["LAT"].values))
            current_pos = (last_n["LAT"].iloc[-1], last_n["LONG"].iloc[-1])
            
            # Calculate direction vector
            direction_vec = self.calculate_direction_vector(coords_sequence)
            
            # Find best station
            station_result = self.find_best_station(current_pos, direction_vec)
            
            # Store path trace for visualization
            path_traces.append({
                "tbox_id": tbox_id,
                "lat": last_n["LAT"].values.tolist(),
                "long": last_n["LONG"].values.tolist(),
                "color": "blue",
                "name": f"{tbox_id} Path"
            })
            
            # Store direction vector for visualization
            if direction_vec:
                dx, dy = direction_vec
                x1, y1 = last_n["LONG"].iloc[-1], last_n["LAT"].iloc[-1]
                extension_factor = 0.01  # Adjust based on coordinate system
                
                direction_vectors.append({
                    "tbox_id": tbox_id,
                    "lat": [y1, y1 + extension_factor * dy],
                    "long": [x1, x1 + extension_factor * dx],
                    "color": "yellow",
                    "name": f"{tbox_id} Direction"
                })
            
            # Store selected station connection
            if station_result["success"]:
                station = station_result["station"]
                selected_stations.append({
                    "tbox_id": tbox_id,
                    "lat": [current_pos[0], station["lat"]],
                    "long": [current_pos[1], station["long"]],
                    "color": "red",
                    "name": f"{tbox_id} to {station['station_name']}"
                })
            
            # Add to results
            vehicle_result = {
                "tbox_id": tbox_id,
                "current_position": {"lat": current_pos[0], "long": current_pos[1]},
                "station_recommendation": station_result,
                "path_length": len(last_n)
            }
            results.append(vehicle_result)
        
        # Combine visualization data
        final_result = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_vehicles": len(results),
            "vehicle_recommendations": results,
            "visualization_data": {
                "path_traces": path_traces,
                "direction_vectors": direction_vectors,
                "selected_stations": selected_stations
            }
        }
        
        # Save to file if requested
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(final_result, f, indent=2)
        
        return final_result
    
    def find_closest_station_simple(self, lat: float, lon: float, max_distance_km: float = 50) -> Dict:
        """
        Simple closest station finder without directional analysis
        
        Args:
            lat: Latitude of current position
            lon: Longitude of current position 
            max_distance_km: Maximum search distance
            
        Returns:
            Dict with closest station information
        """
        current_pos = (lat, lon)
        station_distances = []
        
        for _, station in self.stations.iterrows():
            station_pos = (station["lat"], station["long"])
            distance = geodesic(current_pos, station_pos).km
            
            if distance <= max_distance_km:
                station_distances.append({
                    "station_id": station["station_id"],
                    "station_name": station["station_name"],
                    "lat": station["lat"],
                    "long": station["long"],
                    "district": station["district"],
                    "address": station["address"],
                    "distance_km": round(distance, 2)
                })
        
        if station_distances:
            # Sort by distance
            station_distances.sort(key=lambda x: x["distance_km"])
            
            return {
                "success": True,
                "current_position": {"lat": lat, "long": lon},
                "closest_stations": station_distances,
                "total_found": len(station_distances)
            }
        else:
            return {
                "success": False,
                "current_position": {"lat": lat, "long": lon},
                "message": f"No stations found within {max_distance_km}km radius",
                "total_found": 0
            }


# Convenience functions for API integration
def find_closest_station(lat: float, lon: float, max_distance_km: float = 50) -> Dict:
    """Find closest charging stations to given coordinates"""
    service = LocalClosestStationService()
    return service.find_closest_station_simple(lat, lon, max_distance_km)

def find_closest_station_with_direction(
    gps_data_file: Optional[str] = None,
    stations_file: Optional[str] = None,
    output_file: Optional[str] = None
) -> Dict:
    """Find closest stations with directional analysis for fleet of vehicles"""
    service = LocalClosestStationService(gps_data_file, stations_file)
    return service.process_vehicle_data(output_file)

# For backward compatibility with original Snowflake function signature
def main(session=None, stage_name: str = "@CLOSEST_STATION") -> List[Dict]:
    """
    Main function compatible with original Snowflake version
    Now works with local data instead of Snowflake session
    """
    service = LocalClosestStationService()
    result = service.process_vehicle_data()
    
    # If stage_name provided, save to local file instead of Snowflake
    if stage_name.startswith('@'):
        # Convert stage name to local path
        local_path = stage_name.replace('@', 'output/')
        os.makedirs(local_path, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"geo_clustered_data_{timestamp}.json"
        full_path = os.path.join(local_path, filename)
        
        with open(full_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Results saved to: {full_path}")
    
    return result["visualization_data"]  # Return format compatible with original