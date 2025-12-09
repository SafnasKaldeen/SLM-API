"""
EV Safety Alert System - Fixed with Intelligent Rerouting
- Detects unreachable stations (negative battery)
- Finds alternative routes within 20% extra distance
- Vehicle-type aware recommendations
- No more negative battery displays
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from enum import Enum

from coverage import coverage


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Types of alerts"""
    STATION_AHEAD = "station_ahead"
    STATION_DETOUR = "station_detour"
    LEAVING_SAFE_ZONE = "leaving_safe_zone"
    POINT_OF_NO_RETURN = "point_of_no_return"
    LOW_BATTERY_CRITICAL = "low_battery_critical"
    NEAREST_STATION_UNREACHABLE = "nearest_station_unreachable"
    ROUTE_DEVIATION = "route_deviation"
    BATTERY_DRAIN_ABNORMAL = "battery_drain_abnormal"
    SAFE_TO_CONTINUE = "safe_to_continue"
    RETURN_TO_COVERAGE = "return_to_coverage"
    STATIONARY_LOW_BATTERY = "stationary_low_battery"
    COVERAGE_ZONE_EXIT = "coverage_zone_exit"
    DIRECT_ZONE_EXIT = "direct_zone_exit"
    APPROACHING_BOUNDARY = "approaching_boundary"
    OUTSIDE_ALL_COVERAGE = "outside_all_coverage"
    STATION_IN_DIRECTION = "station_in_direction"
    NO_STATION_IN_DIRECTION = "no_station_in_direction"
    MULTI_HOP_REQUIRED = "multi_hop_required"
    CHARGING_DESERT = "charging_desert"
    STRANDED_IMMINENT = "stranded_imminent"
    DRIVING_AWAY_FROM_STATIONS = "driving_away_from_stations"
    STATIONS_BEHIND_ONLY = "stations_behind_only"
    TURN_AROUND_REQUIRED = "turn_around_required"
    ISOLATED_NETWORK = "isolated_network"
    HEADING_TO_DEAD_ZONE = "heading_to_dead_zone"
    STATION_UNREACHABLE_REROUTE = "station_unreachable_reroute"


class VehicleType(Enum):
    """Vehicle type for context-aware recommendations"""
    CAR = "car"
    SCOOTER = "scooter"
    BIKE = "bike"
    MOTORCYCLE = "motorcycle"
    TRUCK = "truck"


@dataclass
class Alert:
    """Alert data structure"""
    timestamp: datetime
    alert_type: AlertType
    alert_level: AlertLevel
    title: str
    message: str
    recommended_action: str
    station_info: Optional[Dict] = None
    alternative_station: Optional[Dict] = None
    distance_to_boundary: Optional[float] = None
    time_to_boundary: Optional[float] = None
    battery_at_boundary: Optional[float] = None
    point_of_no_return_distance: Optional[float] = None
    metadata: Optional[Dict] = None
    priority: int = 0
    
    def to_dict(self):
        """Convert alert to dictionary for serialization"""
        result = {
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type.value,
            'alert_level': self.alert_level.value,
            'priority': self.priority,
            'title': self.title,
            'message': self.message,
            'recommended_action': self.recommended_action,
        }
        
        if self.station_info:
            result['station_info'] = self.station_info
        if self.alternative_station:
            result['alternative_station'] = self.alternative_station
        if self.distance_to_boundary is not None:
            result['distance_to_boundary_km'] = round(self.distance_to_boundary, 2)
        if self.time_to_boundary is not None:
            result['time_to_boundary_minutes'] = round(self.time_to_boundary * 60, 1)
        if self.battery_at_boundary is not None:
            result['battery_at_boundary_percent'] = round(self.battery_at_boundary, 1)
        if self.point_of_no_return_distance is not None:
            result['point_of_no_return_km'] = round(self.point_of_no_return_distance, 2)
        if self.metadata:
            result['metadata'] = self.metadata
            
        return result


@dataclass
class VehicleStatus:
    """Current vehicle status"""
    position: Tuple[float, float]
    battery_percent: float
    efficiency_km_per_percent: float
    heading_degrees: Optional[float] = None
    speed_kmh: Optional[float] = None
    battery_drain_rate_per_km: Optional[float] = None
    is_moving: bool = True
    vehicle_type: VehicleType = VehicleType.CAR


@dataclass
class CoverageContext:
    """Coverage area context"""
    in_direct_coverage: bool
    in_combined_coverage: bool
    distance_to_direct_boundary: float
    distance_to_combined_boundary: float
    max_range_km: float
    point_of_no_return_km: float
    reachable_stations: List[Dict]
    unreachable_stations: List[Dict]
    network_info: Optional[Dict] = None


@dataclass
class AlertConfig:
    """Configurable alert thresholds"""
    warning_distance_km: float = 30.0
    critical_distance_km: float = 15.0
    low_battery_threshold: float = 20.0
    critical_battery_threshold: float = 15.0
    emergency_battery_threshold: float = 5.0
    abnormal_drain_factor: float = 1.5
    station_ahead_angle: float = 45.0
    point_of_no_return_buffer: float = 1.2
    significant_gap_km: float = 10.0
    significant_battery_deficit_percent: float = 15.0
    rear_cone_angle: float = 135.0
    side_cone_min_angle: float = 60.0
    max_detour_percent: float = 20.0  # Maximum 20% extra distance for rerouting
    safety_buffer_percent: float = 35.0  # Keep 30% battery as safety margin


class EVAlertSystem:
    """
    Production-Ready EV Safety Alert System with Intelligent Rerouting
    - Detects unreachable stations
    - Finds optimal alternatives within detour threshold
    - Vehicle-type aware recommendations
    """
    
    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
    
    def generate_alerts(
        self,
        vehicle: VehicleStatus,
        coverage: CoverageContext,
        timestamp: Optional[datetime] = None,
        safety_margin: float = 0.3,
        debug: bool = False
    ) -> List[Alert]:
        """
        Main entry point: Generate all applicable alerts with rerouting
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        alerts = []
        
        # DEBUG: Log critical information
        if debug:
            print("\n=== ALERT SYSTEM DEBUG ===")
            print(f"Battery: {vehicle.battery_percent}%")
            print(f"Heading: {vehicle.heading_degrees}°")
            print(f"Vehicle Type: {vehicle.vehicle_type.value}")
            print(f"Position: {vehicle.position}")
            print(f"Reachable stations: {len(coverage.reachable_stations)}")
            print("========================\n")
        
        # CRITICAL PRIORITY 1: Check for isolated network with directional analysis
        if coverage.network_info and vehicle.heading_degrees is not None:
            isolation_alert = self._check_network_isolation(vehicle, coverage, timestamp)
            if isolation_alert:
                alerts.append(isolation_alert)
        
        # CRITICAL PRIORITY 2: Verify heading safety
        heading_to_disconnected = False
        if vehicle.heading_degrees is not None and coverage.network_info:
            heading_to_disconnected = self._is_heading_to_disconnected_only(vehicle, coverage)
            
            has_isolation_alert = any(
                a.alert_type in [AlertType.ISOLATED_NETWORK, AlertType.HEADING_TO_DEAD_ZONE] 
                for a in alerts
            )
            
            if heading_to_disconnected and not has_isolation_alert:
                failsafe_alert = Alert(
                    timestamp=timestamp,
                    alert_type=AlertType.HEADING_TO_DEAD_ZONE,
                    alert_level=AlertLevel.CRITICAL,
                    priority=96,
                    title="CRITICAL: Heading to Unreachable Network",
                    message=f"You are heading toward stations in disconnected networks. These stations are UNREACHABLE with current battery ({vehicle.battery_percent:.1f}%). No accessible stations in this direction.",
                    recommended_action="TURN AROUND NOW\nAll stations ahead are in isolated networks\nReturn to accessible charging stations",
                    metadata={
                        'heading_to_disconnected_only': True,
                        'failsafe_alert': True,
                        'battery_percent': vehicle.battery_percent
                    }
                )
                alerts.insert(0, failsafe_alert)
        
        # PRIORITY 3: Check for charging desert
        if len(coverage.reachable_stations) == 0:
            if vehicle.heading_degrees is not None:
                desert_alert = self._check_charging_desert_directional(vehicle, coverage, timestamp)
            else:
                desert_alert = self._check_charging_desert(vehicle, coverage, timestamp)
            
            if desert_alert:
                alerts.append(desert_alert)
                if desert_alert.alert_level == AlertLevel.EMERGENCY:
                    return [desert_alert]
        
        # PRIORITY 4: Stationary low battery
        if not vehicle.is_moving:
            alert = self._check_stationary_battery(vehicle, coverage, timestamp)
            if alert:
                alerts.append(alert)
        
        # PRIORITY 5: Battery level checks
        battery_alert = self._check_battery_level(vehicle, coverage, timestamp)
        if battery_alert:
            alerts.append(battery_alert)
        
        # PRIORITY 6: Abnormal battery drain
        if vehicle.battery_drain_rate_per_km:
            alert = self._check_abnormal_drain(vehicle, timestamp)
            if alert:
                alerts.append(alert)
        
        # PRIORITY 7: Coverage-based alerts
        if not coverage.in_combined_coverage:
            alert = self._check_outside_coverage(vehicle, coverage, timestamp)
            if alert:
                alerts.append(alert)
        elif not coverage.in_direct_coverage and coverage.in_combined_coverage:
            alert = self._check_extended_coverage(vehicle, coverage, timestamp)
            if alert:
                alerts.append(alert)
        elif coverage.in_direct_coverage:
            alert = self._check_direct_boundary(vehicle, coverage, timestamp)
            if alert:
                alerts.append(alert)
        
        # PRIORITY 8: Direction-based station alerts WITH REROUTING
        if vehicle.heading_degrees is not None and coverage.reachable_stations:
            direction_alerts = self._check_directional_stations_with_rerouting(
                vehicle, coverage, timestamp
            )
            alerts.extend(direction_alerts)
        elif coverage.reachable_stations:
            direction_alerts = self._check_stations_no_heading(vehicle, coverage, timestamp)
            alerts.extend(direction_alerts)
        
        # PRIORITY 9: Time-based boundary approach
        if vehicle.speed_kmh and vehicle.speed_kmh > 0:
            alert = self._check_time_to_boundary(vehicle, coverage, timestamp)
            if alert:
                alerts.append(alert)
        
        # PRIORITY 10: Multi-hop info
        if coverage.reachable_stations:
            alert = self._check_multi_hop(vehicle, coverage, timestamp)
            if alert:
                alerts.append(alert)
        
        # SAFE STATUS LOGIC
        has_danger_alerts = any(
            a.alert_level in [AlertLevel.WARNING, AlertLevel.CRITICAL, AlertLevel.EMERGENCY] 
            for a in alerts
        )
        
        has_isolation_or_heading_danger = any(
            a.alert_type in [AlertType.ISOLATED_NETWORK, AlertType.HEADING_TO_DEAD_ZONE] 
            for a in alerts
        )
        
        should_show_safe = (
            not has_danger_alerts and
            not has_isolation_or_heading_danger and
            not heading_to_disconnected and
            coverage.in_direct_coverage and 
            vehicle.battery_percent > 30 and
            len(coverage.reachable_stations) > 0
        )
        
        if should_show_safe:
            alerts.append(self._create_safe_alert(vehicle, coverage, timestamp))
        
        # Sort by priority
        severity_order = {
            AlertLevel.EMERGENCY: 4,
            AlertLevel.CRITICAL: 3,
            AlertLevel.WARNING: 2,
            AlertLevel.INFO: 1
        }
        
        alerts.sort(
            key=lambda a: (severity_order[a.alert_level], a.priority),
            reverse=True
        )
        
        return alerts

    def _calculate_network_density_score(
        self,
        station_location: Tuple[float, float],
        coverage: CoverageContext
    ) -> Dict:
        """
        Calculate density based on which NETWORK the station belongs to
        Networks with more stations = better infrastructure = higher score
        """
        network_info = coverage.network_info if coverage else None
        
        if not network_info or network_info.get('network_count', 1) == 1:
            # Single network or no network info - use total reachable stations as density
            total_stations = len(coverage.reachable_stations) if coverage else 0
            return {
                'network_id': 1,
                'network_station_count': total_stations,
                'density_score': total_stations * 10,
                'is_main_network': True
            }
        
        # Find which network this station belongs to
        station_loc = tuple(station_location)
        
        for network in network_info.get('networks', []):
            network_id = network.get('network_id')
            network_station_count = network.get('station_count', 0)
            
            # Check if station is in this network
            for station_info in network.get('stations', []):
                station_info_loc = tuple(station_info.get('location', []))
                if len(station_info_loc) == 2 and len(station_loc) == 2:
                    if (abs(station_info_loc[0] - station_loc[0]) < 0.0001 and 
                        abs(station_info_loc[1] - station_loc[1]) < 0.0001):
                        # Found the network this station belongs to
                        return {
                            'network_id': network_id,
                            'network_station_count': network_station_count,
                            'density_score': network_station_count * 10,
                            'is_main_network': network_id == network_info.get('current_network', {}).get('network_id')
                        }
        
        # Fallback - station not found in any network
        return {
            'network_id': None,
            'network_station_count': 0,
            'density_score': 0,
            'is_main_network': False
        }

    def _find_alternative_station(
        self,
        vehicle: VehicleStatus,
        target_station: Dict,
        all_stations: List[Dict],
        coverage: CoverageContext,
        max_detour_percent: float = None
    ) -> Optional[Dict]:
        """
        Find alternative station within detour threshold
        PRIORITIZES:
        1. Stations in DENSE NETWORKS (networks with many stations)
        2. Stations in current network (avoid crossing if possible)
        3. Stations in direction of travel
        4. Stations closer to target station
        """
        if max_detour_percent is None:
            max_detour_percent = self.config.max_detour_percent
        
        target_distance = target_station.get('distance_km', target_station.get('direct_distance_km', 0))
        max_allowed_distance = target_distance * (1 + max_detour_percent / 100)
        
        battery_drain_per_km = vehicle.battery_drain_rate_per_km or (1 / vehicle.efficiency_km_per_percent)
        target_location = target_station.get('location')
        
        # Get current network info
        network_info = coverage.network_info if coverage else None
        current_network_id = None
        if network_info:
            current_network = network_info.get('current_network')
            if current_network:
                current_network_id = current_network.get('network_id')
        
        alternatives = []
        
        for station in all_stations:
            if station.get('name') == target_station.get('name'):
                continue
            
            station_distance = station.get('distance_km', station.get('direct_distance_km', 0))

            if station_distance > max_allowed_distance:
                continue
            
            safety_margin = 0.35  # Should match the safety_margin passed to coverage analysis
            usable_battery = vehicle.battery_percent * (1 - safety_margin)
                
            # Calculate battery needed and arrival battery using USABLE battery
            battery_needed = (station_distance * battery_drain_per_km)
            battery_on_arrival = usable_battery - battery_needed  # Changed from vehicle.battery_percent
            
            if battery_on_arrival <= 0:
                print(f"Alternative '{station.get('name')}': Insufficient battery on arrival")
                continue
            
            detour_percent = ((station_distance - target_distance) / target_distance) * 100
            
            station_location = station.get('location')
            distance_to_target = self._calculate_distance(
                station_location[0], station_location[1],
                target_location[0], target_location[1]
            )
            
            # NEW: Calculate NETWORK density (not nearby stations)
            network_density = self._calculate_network_density_score(
                tuple(station_location),
                coverage
            )
            
            density_score = network_density['density_score']
            network_station_count = network_density['network_station_count']
            station_network_id = network_density['network_id']
            in_current_network = network_density['is_main_network']
            
            # Direction scoring
            direction_category = "unknown"
            angle_from_heading = None
            direction_score = 0

            if vehicle.heading_degrees is not None:
                bearing = self._calculate_bearing(
                    vehicle.position[0], vehicle.position[1],
                    station['location'][0], station['location'][1]
                )
                angle_from_heading = self._angle_difference(vehicle.heading_degrees, bearing)
                
                # CRITICAL: Heavy penalty for going backward
                if angle_from_heading <= self.config.station_ahead_angle:
                    direction_category = "ahead"
                    direction_score = 200  # HIGHEST - forward progress
                elif angle_from_heading >= self.config.rear_cone_angle:
                    direction_category = "behind"
                    direction_score = -100  # NEGATIVE - going backward is BAD
                else:
                    direction_category = "side"
                    direction_score = 50  # Moderate - requires course change

            proximity_score = max(0, 100 - distance_to_target)
            
            # Network preference bonus
            network_bonus = 50 if in_current_network else 0

            # Update the comprehensive score calculation:
            score = (
                density_score * 20 +            # Network infrastructure (20x)
                direction_score * 10 +          # CRITICAL: Direction (10x) - forward progress
                network_bonus +                 # Stay in current network (+50)
                proximity_score * 3 +           # Proximity to target
                (-distance_to_target * 5)       # Prefer closer to target
            )

            print(f"Alternative '{station.get('name')}': Distance {station_distance:.1f}km, Detour {detour_percent:+.1f}%, Battery on arrival {battery_on_arrival:.1f}%, Direction {direction_category}, Angle {angle_from_heading}, Score {score:.1f}, Network Stations {network_station_count}, In Current Network {in_current_network}")
            
            alternatives.append({
                'station': station,
                'distance_km': station_distance,
                'detour_percent': detour_percent,
                'battery_on_arrival': battery_on_arrival,
                'direction_category': direction_category,
                'angle_from_heading': angle_from_heading,
                'direction_score': direction_score,
                'distance_to_target': distance_to_target,
                'proximity_score': proximity_score,
                'density_score': density_score,
                'network_station_count': network_station_count,
                'network_id': station_network_id,
                'in_current_network': in_current_network,
                'score': score
            })
        
        if not alternatives:
            return None
        
        alternatives.sort(key=lambda x: -x['score'])
        best = alternatives[0]
        
        # Build recommendation message emphasizing network density
        if best['network_station_count'] >= 10:
            density_message = f"Network {best['network_id']} has {best['network_station_count']} stations (EXCELLENT infrastructure)"
        elif best['network_station_count'] >= 5:
            density_message = f"Network {best['network_id']} has {best['network_station_count']} stations (good coverage)"
        elif best['network_station_count'] >= 3:
            density_message = f"Network {best['network_id']} has {best['network_station_count']} stations (moderate coverage)"
        else:
            density_message = f"Network {best['network_id']} has only {best['network_station_count']} station(s) (limited options)"
        
        return {
            **best['station'],
            'alternative_distance_km': best['distance_km'],
            'alternative_detour_percent': best['detour_percent'],
            'alternative_battery_arrival': best['battery_on_arrival'],
            'direction_category': best['direction_category'],
            'angle_from_heading': best['angle_from_heading'],
            'distance_to_target_station': best['distance_to_target'],
            'density_score': best['density_score'],
            'network_station_count': best['network_station_count'],
            'network_id': best['network_id'],
            'density_message': density_message,
            'in_current_network': best['in_current_network'],
            'network_safety_warning': None if best['in_current_network'] else f"NOTE: Crossing to Network #{best['network_id']}"
        }

    def _check_directional_stations_with_rerouting(
            self,
            vehicle: VehicleStatus,
            coverage: CoverageContext,
            timestamp: datetime
        ) -> List[Alert]:
            """
            Direction-aware station analysis WITH intelligent rerouting
            """
            alerts = []
            
            stations_ahead = []
            stations_behind = []
            stations_side = []
            
            for station in coverage.reachable_stations:
                bearing = self._calculate_bearing(
                    vehicle.position[0], vehicle.position[1],
                    station['location'][0], station['location'][1]
                )
                angle_diff = self._angle_difference(vehicle.heading_degrees, bearing)
                
                station_with_dir = {
                    **station,
                    'bearing': bearing,
                    'angle_from_heading': angle_diff
                }
                
                if angle_diff <= self.config.station_ahead_angle:
                    stations_ahead.append(station_with_dir)
                elif angle_diff >= self.config.rear_cone_angle:
                    stations_behind.append(station_with_dir)
                else:
                    stations_side.append(station_with_dir)
            
            stations_ahead.sort(key=lambda x: x.get('distance_km', x.get('direct_distance_km', 0)))
            stations_behind.sort(key=lambda x: x.get('distance_km', x.get('direct_distance_km', 0)))
            stations_side.sort(key=lambda x: x.get('distance_km', x.get('direct_distance_km', 0)))
            
            # All stations behind
            if stations_behind and not stations_ahead and not stations_side:
                closest_behind = stations_behind[0]
                
                if vehicle.battery_percent < self.config.critical_battery_threshold:
                    level = AlertLevel.EMERGENCY
                    priority = 95
                    title = "EMERGENCY: All Stations Behind You"
                elif vehicle.battery_percent < self.config.low_battery_threshold:
                    level = AlertLevel.CRITICAL
                    priority = 78
                    title = "CRITICAL: All Stations Behind You"
                else:
                    level = AlertLevel.WARNING
                    priority = 52
                    title = "WARNING: Driving Away From All Stations"
                
                alerts.append(Alert(
                    timestamp=timestamp,
                    alert_type=AlertType.STATIONS_BEHIND_ONLY,
                    alert_level=level,
                    priority=priority,
                    title=title,
                    message=f"Heading away from ALL {len(stations_behind)} reachable station(s). Battery: {vehicle.battery_percent:.1f}%. Nearest '{closest_behind['name']}' is {closest_behind['angle_from_heading']:.0f}° behind.",
                    recommended_action=f"{'TURN AROUND NOW!' if level != AlertLevel.WARNING else 'Consider turning around.'} Closest: '{closest_behind['name']}' at {closest_behind.get('distance_km', closest_behind.get('direct_distance_km', 0)):.1f}km ({closest_behind['angle_from_heading']:.0f}° reversal needed).",
                    station_info=closest_behind,
                    metadata={'stations_behind': len(stations_behind), 'direction_status': 'ALL_BEHIND'}
                ))
                return alerts
            
            # Stations only to side
            if stations_side and not stations_ahead and not stations_behind:
                closest_side = stations_side[0]
                level = AlertLevel.WARNING if vehicle.battery_percent < self.config.low_battery_threshold else AlertLevel.INFO
                priority = 50 if level == AlertLevel.WARNING else 25
                
                alerts.append(Alert(
                    timestamp=timestamp,
                    alert_type=AlertType.NO_STATION_IN_DIRECTION,
                    alert_level=level,
                    priority=priority,
                    title=f"{'WARNING' if level == AlertLevel.WARNING else 'Info'}: No Station in Current Direction with safety buffer",
                    message=f"No stations ahead or behind. All {len(stations_side)} reachable station(s) require course change to arrive with safety buffer. Battery: {vehicle.battery_percent:.1f}%.",
                    recommended_action=f"Consider detouring to '{closest_side['name']}' ({closest_side.get('distance_km', closest_side.get('direct_distance_km', 0)):.1f}km, {closest_side['angle_from_heading']:.0f}° off heading).",
                    station_info=closest_side,
                    metadata={'stations_side': len(stations_side), 'direction_status': 'ALL_SIDE'}
                ))
            
            # CRITICAL FIX: Check stations ahead for reachability
            if stations_ahead:
                best = stations_ahead[0]
                distance = best.get('distance_km', best.get('direct_distance_km', 0))
                battery_drain_per_km = vehicle.battery_drain_rate_per_km or (1 / vehicle.efficiency_km_per_percent)
                
                # FIX: Calculate battery_on_arrival with safety buffer

                safety_margin = 0.35  # Should match the safety_margin passed to coverage analysis
                usable_battery = vehicle.battery_percent * (1 - safety_margin)
                
                # Calculate battery needed and arrival battery using USABLE battery
                battery_needed = (distance * battery_drain_per_km)
                battery_on_arrival = usable_battery - battery_needed  # Changed from vehicle.battery_percent
                
                
                # UNREACHABLE STATION - Find alternative
                if battery_on_arrival < 0:
                    # CORRECT - pass coverage, then max_detour_percent
                    alternative = self._find_alternative_station(
                        vehicle, best, coverage.reachable_stations, coverage, self.config.max_detour_percent
                    )
                    
                    # In _check_directional_stations_with_rerouting:
                    if alternative:
                        alt_distance = alternative['alternative_distance_km']
                        detour_percent = alternative['alternative_detour_percent']
                        alt_battery = alternative['alternative_battery_arrival']
                        density_msg = alternative.get('density_message', '')
                        
                        alerts.append(Alert(
                            timestamp=timestamp,
                            alert_type=AlertType.STATION_UNREACHABLE_REROUTE,
                            alert_level=AlertLevel.CRITICAL,
                            priority=85,
                            title="CRITICAL: Direct Route Unreachable - Reroute Available",
                            message=f"'{best['name']}' ({distance:.1f}km) is UNREACHABLE. Need {abs(battery_on_arrival):.1f}% MORE battery.",
                            recommended_action=f"REROUTE to '{alternative['name']}'\n"
                                            f"Distance: {alt_distance:.1f}km ({detour_percent:+.1f}% detour)\n"
                                            f"Arrival battery: ~{alt_battery:.1f}%\n"
                                            f"Area coverage: {density_msg}",  # NEW
                            station_info=best,
                            alternative_station=alternative,
                            metadata={
                                'unreachable_distance': distance,
                                'battery_deficit': abs(battery_on_arrival),
                                'alternative_available': True,
                                'detour_percent': detour_percent,
                                'density_score': alternative.get('density_score', 0),
                                'nearby_stations': alternative.get('nearby_stations_count', 0)
                            }
                        ))
                    else:
                        # No viable alternative within threshold
                        alerts.append(Alert(
                            timestamp=timestamp,
                            alert_type=AlertType.NEAREST_STATION_UNREACHABLE,
                            alert_level=AlertLevel.EMERGENCY,
                            priority=98,
                            title="EMERGENCY: Nearest Station Unreachable - No Alternative",
                            message=f"'{best['name']}' ({distance:.1f}km ahead) is UNREACHABLE. Need {abs(battery_on_arrival):.1f}% MORE battery. No alternative stations within {self.config.max_detour_percent:.0f}% detour threshold.",
                            recommended_action=f"CRITICAL ACTION REQUIRED:\n"
                                            f"1. TURN AROUND if possible\n"
                                            f"2. Check behind/side stations\n"
                                            f"3. Call roadside assistance if no options\n"
                                            f"Battery: {vehicle.battery_percent:.1f}%",
                            station_info=best,
                            metadata={
                                'unreachable_distance': distance,
                                'battery_deficit': abs(battery_on_arrival),
                                'alternative_available': False
                            }
                        ))
                
                # REACHABLE but low arrival battery (below critical threshold)
                elif battery_on_arrival < self.config.critical_battery_threshold:
                    # Get vehicle-appropriate recommendations
                    efficiency_tips = self._get_efficiency_tips(vehicle.vehicle_type)
                    
                    alerts.append(Alert(
                        timestamp=timestamp,
                        alert_type=AlertType.STATION_AHEAD,
                        alert_level=AlertLevel.WARNING,
                        priority=48,
                        title="WARNING: Station Ahead - Low Arrival Battery",
                        message=f"'{best['name']}' is {distance:.1f}km ahead ({best['angle_from_heading']:.0f}° from heading). Arrival battery will be low (~{battery_on_arrival:.1f}%) with safety buffer.",
                        recommended_action=f"Plan to charge at '{best['name']}'. {efficiency_tips}",
                        station_info=best,
                        metadata={'battery_on_arrival': round(battery_on_arrival, 1)}
                    ))
                
                # REACHABLE with comfortable margin - show info
                elif (distance < 20 or vehicle.battery_percent < 50) and battery_on_arrival < 30:
                    alerts.append(Alert(
                        timestamp=timestamp,
                        alert_type=AlertType.STATION_AHEAD,
                        alert_level=AlertLevel.INFO,
                        priority=15,
                        title="Info: Charging Station Ahead",
                        message=f"'{best['name']}' is {distance:.1f}km ahead. Battery: {vehicle.battery_percent:.1f}%.",
                        recommended_action=f"Consider charging. Estimated arrival: ~{battery_on_arrival:.1f}%",
                        station_info=best,
                        metadata={'battery_on_arrival': round(battery_on_arrival, 1)}
                    ))
            
            return alerts

    def _get_efficiency_tips(self, vehicle_type: VehicleType) -> str:
        """Get vehicle-specific efficiency recommendations"""
        if vehicle_type in [VehicleType.SCOOTER, VehicleType.BIKE, VehicleType.MOTORCYCLE]:
            return "Reduce speed, avoid aggressive acceleration, coast when possible."
        elif vehicle_type == VehicleType.TRUCK:
            return "Maintain steady speed, minimize stops, reduce cargo weight if possible."
        else:  # CAR
            return "Minimize AC/heating, reduce speed, drive efficiently."

    def _is_heading_to_disconnected_only(
        self,
        vehicle: VehicleStatus,
        coverage: CoverageContext
    ) -> bool:
        """Check if heading towards disconnected networks only"""
        network_info = coverage.network_info
        if not network_info or vehicle.heading_degrees is None:
            return False
        
        is_isolated = network_info.get('network_count', 1) > 1
        if not is_isolated:
            return False
        
        current_network = network_info.get('current_network')
        if not current_network:
            return False
        
        current_network_id = current_network.get('network_id')
        
        network_stations = {}
        if 'networks' in network_info:
            for network in network_info.get('networks', []):
                network_id = network.get('network_id')
                network_stations[network_id] = set()
                for station_info in network.get('stations', []):
                    station_loc = tuple(station_info.get('location', []))
                    if len(station_loc) == 2:
                        network_stations[network_id].add(station_loc)
        
        current_network_stations_ahead = 0
        other_network_stations_ahead = 0
        
        all_stations = coverage.reachable_stations + coverage.unreachable_stations
        
        for station in all_stations:
            station_loc = tuple(station.get('location', []))
            bearing = self._calculate_bearing(
                vehicle.position[0], vehicle.position[1],
                station['location'][0], station['location'][1]
            )
            angle_diff = self._angle_difference(vehicle.heading_degrees, bearing)
            
            if angle_diff <= self.config.station_ahead_angle:
                in_current_network = any(
                    station_loc in station_set and net_id == current_network_id
                    for net_id, station_set in network_stations.items()
                )
                
                if in_current_network:
                    current_network_stations_ahead += 1
                else:
                    other_network_stations_ahead += 1
        
        return other_network_stations_ahead > 0 and current_network_stations_ahead == 0
        
    def _is_directly_reachable(self, station: Dict, coverage: CoverageContext) -> bool:
        """Check if station is in the directly reachable list (0 hops)"""
        return any(
            station.get('name') == s.get('name') and s.get('hops_required', 0) == 0
            for s in coverage.reachable_stations
        )

    def _check_network_isolation(
        self,
        vehicle: VehicleStatus,
        coverage: CoverageContext,
        timestamp: datetime
    ) -> Optional[Alert]:
        """Comprehensive network isolation detection with directional analysis"""
        network_info = coverage.network_info
        
        if not network_info or vehicle.heading_degrees is None:
            return None
        
        is_isolated = network_info.get('network_count', 1) > 1
        if not is_isolated:
            return None
            
        current_network = network_info.get('current_network')
        if not current_network:
            return None
        
        current_network_id = current_network.get('network_id')
        total_networks = network_info.get('network_count')
        
        network_stations = {}
        if 'networks' in network_info:
            for network in network_info.get('networks', []):
                network_id = network.get('network_id')
                network_stations[network_id] = set()
                for station_info in network.get('stations', []):
                    station_loc = tuple(station_info.get('location', []))
                    if len(station_loc) == 2:
                        network_stations[network_id].add(station_loc)
        
        current_ahead = []
        current_behind = []
        current_side = []
        other_ahead = []
        other_behind = []
        other_side = []
        
        all_stations = coverage.reachable_stations + coverage.unreachable_stations
        
        for station in all_stations:
            station_loc = tuple(station.get('location', []))
            
            bearing = self._calculate_bearing(
                vehicle.position[0], vehicle.position[1],
                station['location'][0], station['location'][1]
            )
            angle_diff = self._angle_difference(vehicle.heading_degrees, bearing)
            
            is_ahead = angle_diff <= self.config.station_ahead_angle
            is_behind = angle_diff >= self.config.rear_cone_angle
            
            in_current_network = any(
                station_loc in station_set and net_id == current_network_id
                for net_id, station_set in network_stations.items()
            )
            
            if in_current_network:
                if is_ahead:
                    current_ahead.append(station)
                elif is_behind:
                    current_behind.append(station)
                else:
                    current_side.append(station)
            else:
                if is_ahead:
                    other_ahead.append(station)
                elif is_behind:
                    other_behind.append(station)
                else:
                    other_side.append(station)
        
        # CRITICAL SCENARIO: Heading to disconnected networks only
        only_disconnected_ahead = (len(other_ahead) > 0 and len(current_ahead) == 0)
        
    def _check_network_isolation(
        self,
        vehicle: VehicleStatus,
        coverage: CoverageContext,
        timestamp: datetime
        ) -> Optional[Alert]:
        """Comprehensive network isolation detection with directional analysis"""
        network_info = coverage.network_info
    
        if not network_info or vehicle.heading_degrees is None:
            return None
        
        is_isolated = network_info.get('network_count', 1) > 1
        if not is_isolated:
            return None
            
        current_network = network_info.get('current_network')
        if not current_network:
            return None
        
        current_network_id = current_network.get('network_id')
        total_networks = network_info.get('network_count')
        
        # Build network station sets
        network_stations = {}
        if 'networks' in network_info:
            for network in network_info.get('networks', []):
                network_id = network.get('network_id')
                network_stations[network_id] = set()
                for station_info in network.get('stations', []):
                    station_loc = tuple(station_info.get('location', []))
                    if len(station_loc) == 2:
                        network_stations[network_id].add(station_loc)
        
        current_ahead, current_behind, current_side = [], [], []
        other_ahead, other_behind, other_side = [], [], []
        
        all_stations = coverage.reachable_stations + coverage.unreachable_stations
        
        for station in all_stations:
            station_loc = tuple(station.get('location', []))
            bearing = self._calculate_bearing(
                vehicle.position[0], vehicle.position[1],
                station['location'][0], station['location'][1]
            )
            angle_diff = self._angle_difference(vehicle.heading_degrees, bearing)
            
            is_ahead = angle_diff <= self.config.station_ahead_angle
            is_behind = angle_diff >= self.config.rear_cone_angle
            
            in_current_network = any(
                station_loc in station_set and net_id == current_network_id
                for net_id, station_set in network_stations.items()
            )
            
            if in_current_network:
                if is_ahead:
                    current_ahead.append(station)
                elif is_behind:
                    current_behind.append(station)
                else:
                    current_side.append(station)
            else:
                if is_ahead:
                    other_ahead.append(station)
                elif is_behind:
                    other_behind.append(station)
                else:
                    other_side.append(station)
        
        only_disconnected_ahead = (len(other_ahead) > 0 and len(current_ahead) == 0)
        
        if only_disconnected_ahead:
            has_current_behind_or_side = (len(current_behind) > 0 or len(current_side) > 0)
            
            reachable_ahead = [s for s in other_ahead if self._is_directly_reachable(s, coverage)]
            unreachable_ahead = [s for s in other_ahead if not self._is_directly_reachable(s, coverage)]
            
            num_reachable = len(reachable_ahead)
            num_unreachable = len(unreachable_ahead)
            
            print(f"Reachable ahead: {num_reachable}, Unreachable ahead: {num_unreachable}")
            
            if has_current_behind_or_side:
                # Determine alert level based on battery and reachability
                if num_unreachable > 0:
                    if vehicle.battery_percent < self.config.critical_battery_threshold:
                        level = AlertLevel.EMERGENCY
                        priority = 99
                        title = "EMERGENCY: Heading to Disconnected Network - Low Battery"
                    else:
                        level = AlertLevel.CRITICAL
                        priority = 89
                        title = "CRITICAL: Heading to Disconnected Network"
                else:
                    level = AlertLevel.WARNING
                    priority = 75
                    title = "WARNING: Heading to Isolated Network"
                
                message_parts = []
                if num_reachable > 0:
                    message_parts.append(
                        f"out of {len(reachable_ahead + unreachable_ahead)} only {num_reachable} station(s) ahead are PHYSICALLY reachable, "
                        f"but it's outside your current network ⚠️ ONE-WAY TRIP risk."
                    )
                if num_unreachable > 0:
                    message_parts.append(
                        f"If you continue, you won't be able to return to Network #{current_network_id} "
                    )
                
                message = " ".join(message_parts)
                
                if level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
                    action = (
                        f"TURN BACK NOW - Heading to disconnected networks only\n"
                        f"Stations ahead belong to isolated networks you cannot access.\n"
                        f"Return to Network #{current_network_id} stations behind/side ({len(current_behind) + len(current_side)}).\n"
                        "Charge to 100% from before attempting other networks."
                    )
                else:
                    action = (
                        "Consider carefully: Reachable stations ahead exist, but you may not return to your current network.\n"
                        f"Reachable stations: {[s.get('name') for s in reachable_ahead]}"
                    )
                
                return Alert(
                    timestamp=timestamp,
                    alert_type=AlertType.HEADING_TO_DEAD_ZONE,
                    alert_level=level,
                    priority=priority,
                    title=title,
                    message=message,
                    recommended_action=action,
                    metadata={
                        'network_id': current_network_id,
                        'total_networks': total_networks,
                        'current_stations_behind': len(current_behind),
                        'current_stations_side': len(current_side),
                        'reachable_stations_ahead': num_reachable,
                        'unreachable_stations_ahead': num_unreachable,
                        'heading_to_disconnected_only': True,
                        'isolation_risk': 'CRITICAL' if num_unreachable > 0 else 'WARNING'
                    }
                )
        
        elif len(current_ahead) == 0 and len(current_behind) > 0:
            # Leaving network but some stations behind
            if vehicle.battery_percent < self.config.critical_battery_threshold:
                level = AlertLevel.EMERGENCY
                priority = 92
            elif vehicle.battery_percent < self.config.low_battery_threshold:
                level = AlertLevel.CRITICAL
                priority = 82
            else:
                level = AlertLevel.WARNING
                priority = 58
            
            return Alert(
                timestamp=timestamp,
                alert_type=AlertType.HEADING_TO_DEAD_ZONE,
                alert_level=level,
                priority=priority,
                title=f"{'EMERGENCY' if level == AlertLevel.EMERGENCY else 'WARNING'}: Leaving Isolated Network",
                message=f"Heading away from Network #{current_network_id}. All {len(current_behind)} station(s) behind you. Battery: {vehicle.battery_percent:.1f}%.",
                recommended_action=f"{'TURN AROUND IMMEDIATELY' if level == AlertLevel.EMERGENCY else 'Consider turning back'} - Stay within Network #{current_network_id}",
                metadata={'network_id': current_network_id, 'isolation_risk': 'HIGH'}
            )
        
        return None

    
    def _check_charging_desert_directional(
        self,
        vehicle: VehicleStatus,
        coverage: CoverageContext,
        timestamp: datetime
    ) -> Optional[Alert]:
        """Direction-aware charging desert check"""
        closest = self._find_closest_unreachable(vehicle.position, coverage.unreachable_stations)
        
        if not closest:
            return None
        
        distance_gap = closest['distance_km'] - coverage.max_range_km
        battery_deficit = distance_gap / vehicle.efficiency_km_per_percent
        cannot_return = coverage.distance_to_combined_boundary > coverage.point_of_no_return_km
        
        moving_away = False
        if vehicle.heading_degrees is not None:
            bearing_to_station = self._calculate_bearing(
                vehicle.position[0], vehicle.position[1],
                closest['location'][0], closest['location'][1]
            )
            angle_diff = self._angle_difference(vehicle.heading_degrees, bearing_to_station)
            moving_away = angle_diff > 90
        
        time_until_stranded = None
        if vehicle.is_moving and vehicle.speed_kmh and vehicle.speed_kmh > 0:
            battery_drain_per_km = vehicle.battery_drain_rate_per_km or (1 / vehicle.efficiency_km_per_percent)
            remaining_range = vehicle.battery_percent * vehicle.efficiency_km_per_percent
            time_until_stranded = remaining_range / vehicle.speed_kmh
        
        emergency_conditions = [
            cannot_return,
            vehicle.battery_percent < self.config.emergency_battery_threshold,
            (time_until_stranded is not None and time_until_stranded < 0.5),
            (moving_away and vehicle.battery_percent < self.config.critical_battery_threshold)
        ]
        
        if any(emergency_conditions):
            reasons = []
            if cannot_return:
                reasons.append(f"past point of no return ({coverage.distance_to_combined_boundary:.1f}km from coverage)")
            if vehicle.battery_percent < self.config.emergency_battery_threshold:
                reasons.append(f"critical battery ({vehicle.battery_percent:.1f}%)")
            if time_until_stranded is not None and time_until_stranded < 0.5:
                reasons.append(f"stranding in ~{time_until_stranded * 60:.0f} minutes")
            if moving_away:
                reasons.append("HEADING AWAY FROM ALL STATIONS")
            
            reason_text = "; ".join(reasons)
            
            return Alert(
                timestamp=timestamp,
                alert_type=AlertType.STRANDED_IMMINENT,
                alert_level=AlertLevel.EMERGENCY,
                priority=100,
                title="EMERGENCY: STRANDING IMMINENT",
                message=f"CRITICAL: {reason_text}. NO reachable charging stations. Nearest '{closest['name']}' is {closest['distance_km']:.1f}km (need {battery_deficit:.1f}% MORE battery).",
                recommended_action=f"IMMEDIATE ACTION:\n1. {'TURN AROUND NOW' if moving_away else 'STOP immediately'}\n2. Call roadside assistance NOW\n3. Location: {vehicle.position[0]:.6f}, {vehicle.position[1]:.6f}",
                station_info=closest,
                metadata={
                    'distance_gap_km': round(distance_gap, 2),
                    'battery_deficit_percent': round(battery_deficit, 1),
                    'cannot_return': cannot_return,
                    'moving_away': moving_away,
                    'stranded_risk': 'CRITICAL'
                }
            )
        
        if distance_gap > self.config.significant_gap_km or battery_deficit > self.config.significant_battery_deficit_percent:
            return Alert(
                timestamp=timestamp,
                alert_type=AlertType.CHARGING_DESERT,
                alert_level=AlertLevel.CRITICAL,
                priority=85 if moving_away else 80,
                title=f"CRITICAL: Charging Desert{' - Wrong Direction' if moving_away else ''}",
                message=f"{distance_gap:.1f}km beyond infrastructure. Battery: {vehicle.battery_percent:.1f}%. ALL {len(coverage.unreachable_stations)} stations unreachable. Need {battery_deficit:.1f}% more.{' DRIVING AWAY.' if moving_away else ''}",
                recommended_action=f"HIGH PRIORITY:\n1. {'TURN AROUND IMMEDIATELY' if moving_away else 'Return to network IMMEDIATELY'}\n2. Do NOT drop below 30%\n3. Nearest: '{closest['name']}' at {closest['distance_km']:.1f}km",
                station_info=closest,
                metadata={'distance_gap_km': round(distance_gap, 2), 'moving_away': moving_away}
            )
        
        return Alert(
            timestamp=timestamp,
            alert_type=AlertType.OUTSIDE_ALL_COVERAGE,
            alert_level=AlertLevel.WARNING,
            priority=40,
            title="WARNING: Beyond Charging Network",
            message=f"{distance_gap:.1f}km beyond network. Battery adequate ({vehicle.battery_percent:.1f}%) but {len(coverage.unreachable_stations)} stations unreachable.",
            recommended_action=f"Return before battery drops below 50%\nNearest: '{closest['name']}' at {closest['distance_km']:.1f}km",
            station_info=closest,
            metadata={'distance_gap_km': round(distance_gap, 2)}
        )
    
    def _check_stations_no_heading(
        self,
        vehicle: VehicleStatus,
        coverage: CoverageContext,
        timestamp: datetime
    ) -> List[Alert]:
        """Fallback station checks when heading not available"""
        alerts = []
        
        closest = self._find_closest_reachable(vehicle.position, coverage.reachable_stations)
        if not closest:
            return alerts
        
        distance = closest.get('direct_distance_km', 0)
        battery_drain_per_km = vehicle.battery_drain_rate_per_km or (1 / vehicle.efficiency_km_per_percent)

        safety_margin = 0.35  # Should match the safety_margin passed to coverage analysis
        usable_battery = vehicle.battery_percent * (1 - safety_margin)
                
                # Calculate battery needed and arrival battery using USABLE battery
        battery_needed = (distance * battery_drain_per_km)
        battery_on_arrival = usable_battery - battery_needed  # Changed from vehicle.battery_percent
        

        # Check if unreachable
        if battery_on_arrival < 0:
            alternative = self._find_alternative_station(
                vehicle, closest, coverage.reachable_stations, coverage, self.config.max_detour_percent
            )
            
            if alternative:
                alt_distance = alternative['alternative_distance_km']
                detour_percent = alternative['alternative_detour_percent']
                alt_battery = alternative['alternative_battery_arrival']
                direction = alternative.get('direction_category', 'unknown')
                angle = alternative.get('angle_from_heading')
                
                # Build direction description
                if direction == "ahead":
                    direction_text = f"AHEAD of you ({angle:.0f}° from heading)" if angle else "ahead"
                elif direction == "side":
                    direction_text = f"to your SIDE ({angle:.0f}° from heading)" if angle else "to the side"
                elif direction == "behind":
                    direction_text = f"BEHIND you ({angle:.0f}° turn required)" if angle else "behind"
                else:
                    direction_text = "nearby"
                
                alerts.append(Alert(
                    timestamp=timestamp,
                    alert_type=AlertType.STATION_UNREACHABLE_REROUTE,
                    alert_level=AlertLevel.CRITICAL,
                    priority=86,
                    title="CRITICAL: Nearest Station Unreachable - Reroute Available",
                    message=f"'{closest['name']}' ({distance:.1f}km) is UNREACHABLE. Need {abs(battery_on_arrival):.1f}% MORE battery.",
                    recommended_action=f"REROUTE to '{alternative['name']}' ({direction_text})\n"
                                     f"Distance: {alt_distance:.1f}km ({detour_percent:+.1f}% detour)\n"
                                     f"Arrival battery: ~{alt_battery:.1f}%",
                    station_info=closest,
                    alternative_station=alternative,
                    metadata={'battery_deficit': abs(battery_on_arrival), 'detour_percent': detour_percent, 'alternative_direction': direction}
                ))
            else:
                alerts.append(Alert(
                    timestamp=timestamp,
                    alert_type=AlertType.NEAREST_STATION_UNREACHABLE,
                    alert_level=AlertLevel.EMERGENCY,
                    priority=97,
                    title="EMERGENCY: Nearest Station Unreachable",
                    message=f"'{closest['name']}' is UNREACHABLE. Need {abs(battery_on_arrival):.1f}% MORE battery. No alternatives within {self.config.max_detour_percent:.0f}% detour.",
                    recommended_action="EMERGENCY: Call roadside assistance immediately.",
                    station_info=closest,
                    metadata={'battery_deficit': abs(battery_on_arrival)}
                ))
        
        elif battery_on_arrival < self.config.critical_battery_threshold and distance < 30:
            efficiency_tips = self._get_efficiency_tips(vehicle.vehicle_type)
            alerts.append(Alert(
                timestamp=timestamp,
                alert_type=AlertType.STATION_AHEAD,
                alert_level=AlertLevel.WARNING,
                priority=46,
                title="WARNING: Nearest Station - Low Arrival Battery",
                message=f"Nearest '{closest['name']}' is {distance:.1f}km away. Battery low on arrival (~{battery_on_arrival:.1f}%).",
                recommended_action=f"Plan to charge at '{closest['name']}'. {efficiency_tips}",
                station_info=closest,
                metadata={'battery_on_arrival': round(battery_on_arrival, 1)}
            ))
        elif distance < 15 and vehicle.battery_percent < 50:
            alerts.append(Alert(
                timestamp=timestamp,
                alert_type=AlertType.STATION_AHEAD,
                alert_level=AlertLevel.INFO,
                priority=12,
                title="Info: Charging Station Nearby",
                message=f"'{closest['name']}' is {distance:.1f}km away. Battery: {vehicle.battery_percent:.1f}%.",
                recommended_action=f"Consider charging soon. Estimated arrival: ~{battery_on_arrival:.1f}%",
                station_info=closest,
                metadata={'battery_on_arrival': round(battery_on_arrival, 1)}
            ))
        
        return alerts
    
    def _check_stationary_battery(
        self,
        vehicle: VehicleStatus,
        coverage: CoverageContext,
        timestamp: datetime
    ) -> Optional[Alert]:
        """Check if vehicle is stationary with low battery"""
        if vehicle.battery_percent >= self.config.low_battery_threshold:
            return None
        
        closest = self._find_closest_reachable(vehicle.position, coverage.reachable_stations)
        
        if vehicle.battery_percent < self.config.emergency_battery_threshold:
            level = AlertLevel.EMERGENCY
            priority = 90
            title_prefix = "EMERGENCY"
        elif vehicle.battery_percent < self.config.critical_battery_threshold:
            level = AlertLevel.CRITICAL
            priority = 70
            title_prefix = "CRITICAL"
        else:
            level = AlertLevel.WARNING
            priority = 50
            title_prefix = "WARNING"
        
        action = f"Start driving to '{closest['name']}' ({closest['direct_distance_km']:.1f}km)" if closest else "No reachable stations. Contact roadside assistance."
        
        return Alert(
            timestamp=timestamp,
            alert_type=AlertType.STATIONARY_LOW_BATTERY,
            alert_level=level,
            priority=priority,
            title=f"{title_prefix}: Stationary with Low Battery",
            message=f"Vehicle stopped with {vehicle.battery_percent:.1f}% battery.",
            recommended_action=action,
            station_info=closest
        )
    
    def _check_battery_level(
        self,
        vehicle: VehicleStatus,
        coverage: CoverageContext,
        timestamp: datetime
    ) -> Optional[Alert]:
        """Check critical battery levels"""
        if vehicle.battery_percent >= self.config.low_battery_threshold:
            return None
        
        closest = self._find_closest_reachable(vehicle.position, coverage.reachable_stations)
        
        if vehicle.battery_percent < self.config.emergency_battery_threshold:
            if closest and closest['direct_distance_km'] <= coverage.max_range_km:
                return Alert(
                    timestamp=timestamp,
                    alert_type=AlertType.LOW_BATTERY_CRITICAL,
                    alert_level=AlertLevel.EMERGENCY,
                    priority=95,
                    title="EMERGENCY: Battery Critical",
                    message=f"Only {vehicle.battery_percent:.1f}% battery remaining!",
                    recommended_action=f"URGENT: Proceed to '{closest['name']}' ({closest['direct_distance_km']:.1f}km). {self._get_efficiency_tips(vehicle.vehicle_type)}",
                    station_info=closest
                )
            else:
                return Alert(
                    timestamp=timestamp,
                    alert_type=AlertType.NEAREST_STATION_UNREACHABLE,
                    alert_level=AlertLevel.EMERGENCY,
                    priority=100,
                    title="EMERGENCY: Critical Battery - No Reachable Station",
                    message=f"Only {vehicle.battery_percent:.1f}% battery. Nearest station unreachable.",
                    recommended_action="EMERGENCY: Stop NOW. Contact roadside assistance. Do not drive.",
                    station_info=closest
                )
        elif vehicle.battery_percent < self.config.critical_battery_threshold:
            if closest:
                return Alert(
                    timestamp=timestamp,
                    alert_type=AlertType.LOW_BATTERY_CRITICAL,
                    alert_level=AlertLevel.CRITICAL,
                    priority=75,
                    title="CRITICAL: Battery Level Low",
                    message=f"Battery at {vehicle.battery_percent:.1f}%. Charging needed urgently.",
                    recommended_action=f"Head to '{closest['name']}' ({closest['direct_distance_km']:.1f}km) immediately. No detours.",
                    station_info=closest
                )
        else:
            if closest:
                return Alert(
                    timestamp=timestamp,
                    alert_type=AlertType.LOW_BATTERY_CRITICAL,
                    alert_level=AlertLevel.WARNING,
                    priority=45,
                    title="Low Battery Warning",
                    message=f"Battery at {vehicle.battery_percent:.1f}%. Plan charging soon.",
                    recommended_action=f"Consider charging at '{closest['name']}' ({closest['direct_distance_km']:.1f}km)",
                    station_info=closest
                )
        
        return None
    
    def _check_abnormal_drain(self, vehicle: VehicleStatus, timestamp: datetime) -> Optional[Alert]:
        """Check for abnormal battery drain rate"""
        expected_drain = 1 / vehicle.efficiency_km_per_percent
        
        if vehicle.battery_drain_rate_per_km <= expected_drain * self.config.abnormal_drain_factor:
            return None
        
        drain_multiplier = vehicle.battery_drain_rate_per_km / expected_drain
        
        if drain_multiplier > 2.5:
            level = AlertLevel.CRITICAL
            priority = 65
            title = "CRITICAL: Severe Battery Drain"
        elif drain_multiplier > 2.0:
            level = AlertLevel.WARNING
            priority = 55
            title = "WARNING: High Battery Drain"
        else:
            level = AlertLevel.WARNING
            priority = 35
            title = "Abnormal Battery Drain"
        
        # Vehicle-specific recommendations
        if vehicle.vehicle_type in [VehicleType.SCOOTER, VehicleType.BIKE, VehicleType.MOTORCYCLE]:
            tips = "Check: aggressive acceleration, terrain, tire pressure."
        else:
            tips = "Check: AC/heating, acceleration, terrain, tire pressure."
        
        return Alert(
            timestamp=timestamp,
            alert_type=AlertType.BATTERY_DRAIN_ABNORMAL,
            alert_level=level,
            priority=priority,
            title=title,
            message=f"Battery draining {drain_multiplier:.1f}x faster than expected ({vehicle.battery_drain_rate_per_km:.2f}%/km vs {expected_drain:.2f}%/km).",
            recommended_action=f"{tips} Consider charging soon.",
            metadata={'drain_multiplier': round(drain_multiplier, 2)}
        )
    
    def _check_outside_coverage(
        self,
        vehicle: VehicleStatus,
        coverage: CoverageContext,
        timestamp: datetime
    ) -> Optional[Alert]:
        """Check if outside all coverage zones"""
        closest = self._find_closest_reachable(vehicle.position, coverage.reachable_stations)
        cannot_return = coverage.distance_to_combined_boundary > coverage.point_of_no_return_km
        
        if cannot_return:
            return Alert(
                timestamp=timestamp,
                alert_type=AlertType.POINT_OF_NO_RETURN,
                alert_level=AlertLevel.EMERGENCY,
                priority=98,
                title="EMERGENCY: Point of No Return Passed",
                message=f"Outside coverage. Cannot return ({coverage.distance_to_combined_boundary:.1f}km to coverage, {coverage.point_of_no_return_km:.1f}km safe range).",
                recommended_action=f"CRITICAL: {'Head to ' + closest['name'] if closest else 'Contact emergency support'}. Minimize power.",
                distance_to_boundary=coverage.distance_to_combined_boundary,
                station_info=closest
            )
        
        severity = AlertLevel.CRITICAL if coverage.distance_to_combined_boundary > 20 else AlertLevel.WARNING
        priority = 60 if severity == AlertLevel.CRITICAL else 42
        
        return Alert(
            timestamp=timestamp,
            alert_type=AlertType.OUTSIDE_ALL_COVERAGE,
            alert_level=severity,
            priority=priority,
            title=f"{'CRITICAL' if severity == AlertLevel.CRITICAL else 'WARNING'}: Outside Safe Coverage",
            message=f"You are {coverage.distance_to_combined_boundary:.1f}km outside coverage zone.",
            recommended_action=f"{'URGENT' if severity == AlertLevel.CRITICAL else 'Plan'}: Return to coverage or head to {closest['name'] if closest else 'nearest station'}",
            distance_to_boundary=coverage.distance_to_combined_boundary,
            station_info=closest
        )
    
    def _check_extended_coverage(
        self,
        vehicle: VehicleStatus,
        coverage: CoverageContext,
        timestamp: datetime
    ) -> Optional[Alert]:
        """Check if in extended coverage but not direct"""
        if coverage.distance_to_combined_boundary >= self.config.warning_distance_km:
            return None
        
        closest = self._find_closest_reachable(vehicle.position, coverage.reachable_stations)
        
        if coverage.distance_to_combined_boundary < self.config.critical_distance_km:
            return Alert(
                timestamp=timestamp,
                alert_type=AlertType.COVERAGE_ZONE_EXIT,
                alert_level=AlertLevel.CRITICAL,
                priority=68,
                title="CRITICAL: Approaching Coverage Boundary",
                message=f"Only {coverage.distance_to_combined_boundary:.1f}km from edge of coverage.",
                recommended_action=f"Change course or charge at '{closest['name'] if closest else 'nearest station'}' immediately.",
                distance_to_boundary=coverage.distance_to_combined_boundary,
                station_info=closest
            )
        
        return Alert(
            timestamp=timestamp,
            alert_type=AlertType.COVERAGE_ZONE_EXIT,
            alert_level=AlertLevel.WARNING,
            priority=38,
            title="Nearing Coverage Edge",
            message=f"{coverage.distance_to_combined_boundary:.1f}km from boundary. In extended coverage (multi-hop required).",
            recommended_action=f"Monitor battery. Nearest: '{closest['name'] if closest else 'station'}'",
            distance_to_boundary=coverage.distance_to_combined_boundary,
            station_info=closest
        )
    
    def _check_direct_boundary(
        self,
        vehicle: VehicleStatus,
        coverage: CoverageContext,
        timestamp: datetime
    ) -> Optional[Alert]:
        """Check if approaching direct coverage boundary"""
        if coverage.distance_to_direct_boundary >= self.config.critical_distance_km:
            return None
        
        return Alert(
            timestamp=timestamp,
            alert_type=AlertType.DIRECT_ZONE_EXIT,
            alert_level=AlertLevel.WARNING,
            priority=30,
            title="Approaching Direct Range Limit",
            message=f"Approaching edge of direct range ({coverage.distance_to_direct_boundary:.1f}km to boundary).",
            recommended_action="Consider charging at next station or turning back soon.",
            distance_to_boundary=coverage.distance_to_direct_boundary
        )
    
    def _check_time_to_boundary(
        self,
        vehicle: VehicleStatus,
        coverage: CoverageContext,
        timestamp: datetime
    ) -> Optional[Alert]:
        """Check time until reaching coverage boundary"""
        if not coverage.in_combined_coverage:
            return None
        
        time_to_boundary = coverage.distance_to_combined_boundary / vehicle.speed_kmh
        
        if time_to_boundary >= 0.5 or coverage.distance_to_combined_boundary >= self.config.warning_distance_km:
            return None
        
        battery_drain_per_km = vehicle.battery_drain_rate_per_km or (1 / vehicle.efficiency_km_per_percent)
        battery_loss = coverage.distance_to_combined_boundary * battery_drain_per_km
        battery_at_boundary = vehicle.battery_percent - battery_loss
        
        level = AlertLevel.CRITICAL if time_to_boundary < 0.17 else AlertLevel.WARNING
        priority = 72 if level == AlertLevel.CRITICAL else 36
        
        return Alert(
            timestamp=timestamp,
            alert_type=AlertType.APPROACHING_BOUNDARY,
            alert_level=level,
            priority=priority,
            title=f"{'CRITICAL' if level == AlertLevel.CRITICAL else 'WARNING'}: Approaching Boundary Soon",
            message=f"Will reach boundary in ~{time_to_boundary * 60:.0f} minutes at {vehicle.speed_kmh:.0f} km/h.",
            recommended_action="Plan charging stop or route change immediately." if level == AlertLevel.CRITICAL else "Plan charging stop or route change soon.",
            distance_to_boundary=coverage.distance_to_combined_boundary,
            time_to_boundary=time_to_boundary,
            battery_at_boundary=battery_at_boundary
        )
    
    def _check_multi_hop(
        self,
        vehicle: VehicleStatus,
        coverage: CoverageContext,
        timestamp: datetime
    ) -> Optional[Alert]:
        """Check if multi-hop charging is required"""
        multi_hop_stations = [s for s in coverage.reachable_stations if s.get('hops_required', 0) > 0]
        
        if not multi_hop_stations or vehicle.battery_percent >= 40:
            return None
        
        closest_multi = min(multi_hop_stations, key=lambda x: x['direct_distance_km'])
        
        return Alert(
            timestamp=timestamp,
            alert_type=AlertType.MULTI_HOP_REQUIRED,
            alert_level=AlertLevel.INFO,
            priority=10,
            title="Info: Multi-Hop Charging Required",
            message=f"'{closest_multi['name']}' requires {closest_multi['hops_required']} intermediate charging stop(s).",
            recommended_action="Plan route with intermediate stops. Check station availability.",
            station_info=closest_multi,
            metadata={'hops_required': closest_multi['hops_required']}
        )
    
    def _create_safe_alert(
        self,
        vehicle: VehicleStatus,
        coverage: CoverageContext,
        timestamp: datetime
    ) -> Alert:
        """Create safe status alert"""
        return Alert(
            timestamp=timestamp,
            alert_type=AlertType.SAFE_TO_CONTINUE,
            alert_level=AlertLevel.INFO,
            priority=5,
            title="Safe to Continue",
            message=f"Within safe coverage. Battery: {vehicle.battery_percent:.1f}%. Range: {coverage.max_range_km:.1f}km.",
            recommended_action="Continue as planned. Monitor battery level.",
            distance_to_boundary=coverage.distance_to_direct_boundary,
            metadata={
                'heading': f"{vehicle.heading_degrees:.0f}°" if vehicle.heading_degrees else "Unknown",
                'distance_to_boundary_km': round(coverage.distance_to_direct_boundary, 2)
            }
        )
    
    def _check_charging_desert(
        self,
        vehicle: VehicleStatus,
        coverage: CoverageContext,
        timestamp: datetime
    ) -> Optional[Alert]:
        """Fallback charging desert check (no direction info)"""
        closest = self._find_closest_unreachable(vehicle.position, coverage.unreachable_stations)
        
        if not closest:
            return None
        
        distance_gap = closest['distance_km'] - coverage.max_range_km
        battery_deficit = distance_gap / vehicle.efficiency_km_per_percent
        cannot_return = coverage.distance_to_combined_boundary > coverage.point_of_no_return_km
        
        if cannot_return or vehicle.battery_percent < self.config.emergency_battery_threshold:
            return Alert(
                timestamp=timestamp,
                alert_type=AlertType.STRANDED_IMMINENT,
                alert_level=AlertLevel.EMERGENCY,
                priority=100,
                title="EMERGENCY: STRANDING IMMINENT",
                message=f"CRITICAL: {'Past point of no return' if cannot_return else 'Critical battery'}. NO reachable stations. Nearest '{closest['name']}' is {closest['distance_km']:.1f}km (need {battery_deficit:.1f}% MORE).",
                recommended_action=f"STOP NOW\n1. Stop vehicle\n2. Call roadside assistance\n3. Location: {vehicle.position[0]:.6f}, {vehicle.position[1]:.6f}",
                station_info=closest,
                metadata={'cannot_return': cannot_return, 'battery_deficit': round(battery_deficit, 1)}
            )
        
        if distance_gap > self.config.significant_gap_km:
            return Alert(
                timestamp=timestamp,
                alert_type=AlertType.CHARGING_DESERT,
                alert_level=AlertLevel.CRITICAL,
                priority=80,
                title="CRITICAL: In Charging Desert",
                message=f"Beyond infrastructure by {distance_gap:.1f}km. Need {battery_deficit:.1f}% more battery.",
                recommended_action=f"Return to network immediately. Nearest: '{closest['name']}' at {closest['distance_km']:.1f}km.",
                station_info=closest,
                metadata={'distance_gap_km': round(distance_gap, 2)}
            )
        
        return Alert(
            timestamp=timestamp,
            alert_type=AlertType.OUTSIDE_ALL_COVERAGE,
            alert_level=AlertLevel.WARNING,
            priority=40,
            title="WARNING: Beyond Charging Network",
            message=f"Beyond network by {distance_gap:.1f}km. Battery: {vehicle.battery_percent:.1f}%.",
            recommended_action=f"Return before battery drops below 50%. Nearest: '{closest['name']}' at {closest['distance_km']:.1f}km",
            station_info=closest,
            metadata={'distance_gap_km': round(distance_gap, 2)}
        )
    
    # Helper methods
    
    def _find_closest_reachable(self, position: Tuple[float, float], stations: List[Dict]) -> Optional[Dict]:
        """Find closest reachable station"""
        if not stations:
            return None
        return min(stations, key=lambda s: s.get('direct_distance_km', float('inf')))
    
    def _find_closest_unreachable(self, position: Tuple[float, float], stations: List[Dict]) -> Optional[Dict]:
        """Find closest unreachable station"""
        if not stations:
            return None
        
        for station in stations:
            if 'distance_km' not in station:
                station['distance_km'] = self._calculate_distance(
                    position[0], position[1],
                    station['location'][0], station['location'][1]
                )
        
        return min(stations, key=lambda s: s.get('distance_km', float('inf')))
    
    @staticmethod
    def _calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine distance calculation"""
        import math
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
    
    @staticmethod
    def _calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing between two points"""
        import math
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lon = math.radians(lon2 - lon1)
        
        x = math.sin(delta_lon) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - \
            math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
        
        bearing = math.atan2(x, y)
        return (math.degrees(bearing) + 360) % 360
    
    @staticmethod
    def _angle_difference(angle1: float, angle2: float) -> float:
        """Calculate smallest difference between two angles"""
        diff = abs(angle1 - angle2)
        if diff > 180:
            diff = 360 - diff
        return diff


# Example usage demonstrating the fixed system
if __name__ == "__main__":
    from datetime import datetime
    
    # Initialize alert system
    alert_system = EVAlertSystem()
    
    # Example 1: Scooter with unreachable direct station but alternative available
    print("=" * 80)
    print("EXAMPLE 1: Scooter - Direct station unreachable, alternative available")
    print("=" * 80)
    
    vehicle_scooter = VehicleStatus(
        position=(7.2906, 80.6337),  # Current position
        battery_percent=45.0,
        efficiency_km_per_percent=1.5,  # 1.5 km per 1% battery
        heading_degrees=45.0,
        speed_kmh=40.0,
        vehicle_type=VehicleType.SCOOTER,
        is_moving=True
    )
    
    coverage_scooter = CoverageContext(
        in_direct_coverage=True,
        in_combined_coverage=True,
        distance_to_direct_boundary=50.0,
        distance_to_combined_boundary=80.0,
        max_range_km=67.5,  # 45% * 1.5 km/% = 67.5km max range
        point_of_no_return_km=56.25,  # With safety buffer
        reachable_stations=[
            {
                'name': 'Kurunagala Station',
                'location': (7.4863, 80.3623),
                'direct_distance_km': 86.1,  # Too far - unreachable
                'distance_km': 86.1,
                'hops_required': 0
            },
            {
                'name': 'Dambulla Station',
                'location': (7.3500, 80.5500),  # Closer alternative
                'direct_distance_km': 72.0,  # Within 20% detour, reachable
                'distance_km': 72.0,
                'hops_required': 0
            },
            {
                'name': 'Matale Station',
                'location': (7.4500, 80.6200),
                'direct_distance_km': 65.0,  # Also reachable
                'distance_km': 65.0,
                'hops_required': 0
            }
        ],
        unreachable_stations=[]
    )
    
    alerts_scooter = alert_system.generate_alerts(
        vehicle=vehicle_scooter,
        coverage=coverage_scooter,
        timestamp=datetime.now()
    )
    
    print(f"\nGenerated {len(alerts_scooter)} alert(s):\n")
    for i, alert in enumerate(alerts_scooter, 1):
        print(f"Alert {i}:")
        print(f"  Level: {alert.alert_level.value.upper()}")
        print(f"  Title: {alert.title}")
        print(f"  Message: {alert.message}")
        print(f"  Action: {alert.recommended_action}")
        if alert.station_info:
            print(f"  Station: {alert.station_info['name']} ({alert.station_info.get('direct_distance_km', 'N/A')} km)")
        if alert.alternative_station:
            print(f"  Alternative: {alert.alternative_station['name']} "
                  f"({alert.alternative_station.get('alternative_distance_km', 'N/A')} km, "
                  f"{alert.alternative_station.get('alternative_detour_percent', 0):+.1f}% detour)")
        print()
    
    # Example 2: Car with negative battery - emergency situation
    print("=" * 80)
    print("EXAMPLE 2: Car - All stations unreachable, emergency")
    print("=" * 80)
    
    vehicle_car = VehicleStatus(
        position=(7.2906, 80.6337),
        battery_percent=15.0,  # Very low battery
        efficiency_km_per_percent=2.0,
        heading_degrees=90.0,
        speed_kmh=60.0,
        vehicle_type=VehicleType.CAR,
        is_moving=True
    )
    
    coverage_car = CoverageContext(
        in_direct_coverage=False,
        in_combined_coverage=False,
        distance_to_direct_boundary=80.0,
        distance_to_combined_boundary=100.0,
        max_range_km=30.0,  # Only 30km range left
        point_of_no_return_km=25.0,
        reachable_stations=[],  # No reachable stations!
        unreachable_stations=[
            {
                'name': 'Distant Station',
                'location': (7.5000, 81.0000),
                'direct_distance_km': 85.0
            }
        ]
    )
    
    alerts_car = alert_system.generate_alerts(
        vehicle=vehicle_car,
        coverage=coverage_car,
        timestamp=datetime.now()
    )
    
    print(f"\nGenerated {len(alerts_car)} alert(s):\n")
    for i, alert in enumerate(alerts_car, 1):
        print(f"Alert {i}:")
        print(f"  Level: {alert.alert_level.value.upper()}")
        print(f"  Title: {alert.title}")
        print(f"  Message: {alert.message}")
        print(f"  Action: {alert.recommended_action}")
        print()
    
    print("=" * 80)
    print("Key improvements in this version:")
    print("- ✓ Detects unreachable stations (negative battery)")
    print("- ✓ Finds alternatives within 20% detour threshold")
    print("- ✓ Vehicle-type aware recommendations (no AC tips for scooters)")
    print("- ✓ Never displays negative battery percentages")
    print("- ✓ Intelligent rerouting with detour calculations")
    print("=" * 80)