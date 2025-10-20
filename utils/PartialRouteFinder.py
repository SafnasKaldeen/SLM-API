"""
Partial Route Finder Service
Finds the best possible partial route when destination is unreachable
"""

from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
import json


@dataclass
class PartialRouteNode:
    """Node in partial route"""
    station_idx: int
    coordinates: Tuple[float, float]
    category: str
    station_name: Optional[str] = None


class PartialRouteFinder:
    """Finds best partial route when full route to destination is impossible"""
    
    def __init__(self, 
                 planner,  # EVRoutePlannerV3 instance
                 station_mapping: Dict[str, str]):
        self.planner = planner
        self.station_mapping = station_mapping
    
    def find_best_partial_route(self) -> Optional[Dict]:
        """
        Find the best partial route that gets as close as possible to destination
        Returns detailed analysis including why destination is unreachable
        """
        print("\n=== PARTIAL ROUTE FINDER ACTIVATED ===")
        print("No feasible complete route found. Searching for best partial route...")
        
        best_partial = {
            'distance_covered': 0,
            'path': [],
            'final_location': self.planner.source,
            'distance_to_destination': float('inf'),
            'charging_stops': 0,
            'final_battery': self.planner.initial_battery_percent,
            'closest_approach_km': float('inf')
        }
        
        # Try to find the farthest reachable point
        current_location = self.planner.source
        current_battery = self.planner.initial_battery_percent
        path = [PartialRouteNode(-1, self.planner.source, "Source")]
        total_distance = 0
        charging_stops = 0
        visited_stations = set()
        
        # Calculate initial distance to destination
        initial_distance_to_dest = self.planner.distance_cache.get_distance_with_fallback(
            current_location, self.planner.destination
        )
        closest_approach = initial_distance_to_dest
        closest_approach_location = current_location
        
        print(f"Starting from source. Initial distance to destination: {initial_distance_to_dest:.1f} km")
        
        while charging_stops < self.planner.max_charging_stops:
            # Find all reachable stations from current location
            reachable_stations = []
            
            for idx, station_coords in enumerate(self.planner.charging_stations):
                if idx in visited_stations:
                    continue
                
                station_tuple = tuple(station_coords)
                dist_to_station = self.planner.distance_cache.get_distance_with_fallback(
                    current_location, station_tuple
                )
                
                segment_efficiency = self.planner.efficiency_manager.get_dynamic_efficiency(
                    current_location, station_tuple
                )
                
                battery_needed = dist_to_station / segment_efficiency + self.planner.min_battery_reserve
                
                if current_battery >= battery_needed:
                    # Calculate how close this station is to destination
                    dist_station_to_dest = self.planner.distance_cache.get_distance_with_fallback(
                        station_tuple, self.planner.destination
                    )
                    
                    reachable_stations.append({
                        'idx': idx,
                        'coords': station_tuple,
                        'distance_to_station': dist_to_station,
                        'distance_to_destination': dist_station_to_dest,
                        'battery_needed': battery_needed,
                        'segment_efficiency': segment_efficiency,
                        'progress_score': initial_distance_to_dest - dist_station_to_dest
                    })
            
            if not reachable_stations:
                print(f"No more reachable stations from current location. Stopping at charging stop {charging_stops}")
                break
            
            # Select station that makes most progress toward destination
            best_station = max(reachable_stations, key=lambda s: s['progress_score'])
            
            # Move to selected station
            station_idx = best_station['idx']
            station_coords = best_station['coords']
            dist_to_station = best_station['distance_to_station']
            battery_used = dist_to_station / best_station['segment_efficiency']
            
            total_distance += dist_to_station
            current_battery -= battery_used
            current_location = station_coords
            visited_stations.add(station_idx)
            charging_stops += 1
            
            # Add to path
            coord_key = f"({station_coords[0]},{station_coords[1]})"
            station_name = self.station_mapping.get(coord_key, f"Station_{station_idx}")
            path.append(PartialRouteNode(station_idx, station_coords, "Visiting_Charging_Station", station_name))
            
            # Update closest approach
            if best_station['distance_to_destination'] < closest_approach:
                closest_approach = best_station['distance_to_destination']
                closest_approach_location = station_coords
            
            # Charge to full
            current_battery = 100.0
            
            print(f"Stop {charging_stops}: {station_name}")
            print(f"  Distance traveled: {dist_to_station:.1f} km")
            print(f"  Distance to destination: {best_station['distance_to_destination']:.1f} km")
            print(f"  Progress made: {best_station['progress_score']:.1f} km")
            print(f"  Efficiency: {best_station['segment_efficiency']:.2f} km/%")
        
        # Check if we can reach destination directly from final location
        final_dist_to_dest = self.planner.distance_cache.get_distance_with_fallback(
            current_location, self.planner.destination
        )
        final_efficiency = self.planner.efficiency_manager.get_dynamic_efficiency(
            current_location, self.planner.destination
        )
        battery_needed_for_dest = final_dist_to_dest / final_efficiency + self.planner.min_battery_reserve
        
        can_reach_destination = current_battery >= battery_needed_for_dest
        
        if can_reach_destination:
            # We can actually reach destination! (shouldn't happen, but handle it)
            print("✓ Destination is reachable from final station!")
            return None  # Let main route finder handle this
        
        # Build partial route analysis
        return {
            'partial_route_available': True,
            'reason': self._analyze_unreachable_reason(
                current_location, 
                current_battery, 
                final_dist_to_dest, 
                battery_needed_for_dest,
                final_efficiency
            ),
            'partial_route_summary': self._generate_partial_summary(
                path, 
                total_distance, 
                charging_stops, 
                current_location, 
                current_battery,
                closest_approach,
                initial_distance_to_dest
            ),
            'gap_analysis': {
                'final_location': f"({current_location[0]},{current_location[1]})",
                'final_battery_percent': round(current_battery, 1),
                'remaining_distance_to_destination_km': round(final_dist_to_dest, 1),
                'battery_needed_percent': round(battery_needed_for_dest, 1),
                'battery_shortage_percent': round(battery_needed_for_dest - current_battery, 1),
                'efficiency_at_final_segment': round(final_efficiency, 2),
                'closest_approach_km': round(closest_approach, 1),
                'total_progress_km': round(initial_distance_to_dest - final_dist_to_dest, 1),
                'progress_percentage': round((initial_distance_to_dest - final_dist_to_dest) / initial_distance_to_dest * 100, 1)
            },
            'recommendations': self._generate_recommendations(
                final_dist_to_dest,
                battery_needed_for_dest,
                current_battery,
                final_efficiency,
                charging_stops
            )
        }
    
    def _analyze_unreachable_reason(self, 
                                   final_location: Tuple[float, float],
                                   final_battery: float,
                                   distance_to_dest: float,
                                   battery_needed: float,
                                   efficiency: float) -> str:
        """Analyze why destination cannot be reached"""
        
        battery_shortage = battery_needed - final_battery
        
        reasons = []
        
        if distance_to_dest > 200:
            reasons.append(f"Destination is very far ({distance_to_dest:.1f} km)")
        
        if battery_shortage > 50:
            reasons.append(f"Significant battery shortage ({battery_shortage:.1f}% needed)")
        
        if efficiency < 1.5:
            reasons.append(f"Low efficiency conditions ({efficiency:.2f} km/% due to elevation/weather/traffic)")
        
        if not reasons:
            reasons.append(f"Battery shortage of {battery_shortage:.1f}% from final reachable station")
        
        return "Destination unreachable: " + ", ".join(reasons)
    
    def _generate_partial_summary(self,
                                 path: List[PartialRouteNode],
                                 total_distance: float,
                                 charging_stops: int,
                                 final_location: Tuple[float, float],
                                 final_battery: float,
                                 closest_approach: float,
                                 initial_distance: float) -> List[Dict]:
        """Generate route summary for partial route"""
        
        route_summary = []
        current_battery = self.planner.initial_battery_percent
        
        for i, node in enumerate(path):
            if i < len(path) - 1:
                next_node = path[i + 1]
                next_coords = next_node.coordinates
                
                dist_to_next = self.planner.distance_cache.get_distance_with_fallback(
                    node.coordinates, next_coords
                )
                
                segment_efficiency = self.planner.efficiency_manager.get_dynamic_efficiency(
                    node.coordinates, next_coords
                )
                
                battery_used = dist_to_next / segment_efficiency
                
                # Get efficiency breakdown
                elevation_factor = self.planner.efficiency_manager._get_elevation_factor(
                    node.coordinates, next_coords
                )
                midpoint = (
                    (node.coordinates[0] + next_coords[0]) / 2,
                    (node.coordinates[1] + next_coords[1]) / 2
                )
                weather_factor = self.planner.efficiency_manager._get_weather_factor(midpoint)
                traffic_factor = self.planner.efficiency_manager._get_traffic_factor()
                
                efficiency_breakdown = {
                    "base_efficiency_km_per_percent": round(self.planner.base_efficiency, 3),
                    "elevation_factor": round(elevation_factor, 3),
                    "weather_factor": round(weather_factor, 3),
                    "traffic_factor": round(traffic_factor, 3),
                    "combined_efficiency_km_per_percent": round(segment_efficiency, 3),
                    "efficiency_change_percent": round(
                        ((segment_efficiency - self.planner.base_efficiency) / self.planner.base_efficiency) * 100, 1
                    )
                }
                
                battery_on_departure = 100.0 if node.category == "Visiting_Charging_Station" else current_battery
                
                stop_info = {
                    "location": f"({node.coordinates[0]},{node.coordinates[1]})",
                    "category": node.category,
                    "battery_on_arrival_percent": round(current_battery, 2),
                    "battery_on_departure_percent": round(battery_on_departure, 2),
                    "next_stop_distance_km": round(dist_to_next, 2),
                    "segment_efficiency_km_per_percent": round(segment_efficiency, 3),
                    "battery_utilization_percent": round((battery_used / battery_on_departure) * 100, 1),
                    "station_name": node.station_name,
                    "efficiency_breakdown": efficiency_breakdown
                }
                
                route_summary.append(stop_info)
                current_battery = battery_on_departure - battery_used
            else:
                # Final stop
                stop_info = {
                    "location": f"({node.coordinates[0]},{node.coordinates[1]})",
                    "category": node.category,
                    "battery_on_arrival_percent": round(current_battery, 2),
                    "battery_on_departure_percent": round(current_battery, 2),
                    "next_stop_distance_km": 0,
                    "station_name": node.station_name,
                    "note": "Final reachable station before destination becomes unreachable"
                }
                route_summary.append(stop_info)
        
        return route_summary
    
    def _generate_recommendations(self,
                                 distance_to_dest: float,
                                 battery_needed: float,
                                 current_battery: float,
                                 efficiency: float,
                                 charging_stops: int) -> List[str]:
        """Generate recommendations for reaching destination"""
        
        recommendations = []
        battery_shortage = battery_needed - current_battery
        
        # Calculate required efficiency improvement
        required_efficiency = distance_to_dest / (current_battery - self.planner.min_battery_reserve)
        efficiency_improvement_needed = (required_efficiency / efficiency - 1) * 100
        
        if efficiency_improvement_needed > 0:
            recommendations.append(
                f"Need {efficiency_improvement_needed:.1f}% better efficiency "
                f"(wait for better weather/traffic conditions)"
            )
        
        # Battery upgrade recommendation
        battery_increase_needed = battery_shortage / self.planner.initial_battery_percent * 100
        recommendations.append(
            f"Increase initial battery by {battery_increase_needed:.1f}% "
            f"(charge to at least {self.planner.initial_battery_percent + battery_shortage:.1f}%)"
        )
        
        # Additional stations needed
        if distance_to_dest > 100:
            recommendations.append(
                f"Additional charging stations needed in the {distance_to_dest:.0f} km gap "
                f"between final reachable station and destination"
            )
        
        # Route alternatives
        recommendations.append(
            "Consider alternative destination or waypoints within reachable range"
        )
        
        return recommendations


def integrate_partial_route_finder(planner, station_mapping: Dict[str, str]) -> Optional[Dict]:
    """
    Integrate partial route finder into main planning flow
    Returns partial route analysis if destination is unreachable
    """
    
    finder = PartialRouteFinder(planner, station_mapping)
    return finder.find_best_partial_route()