from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import sys
import csv
import requests

# Path to the stations CSV (same folder as this script)
STATIONS_CSV_FILE = os.path.join(os.path.dirname(__file__), "stations.csv")

# Add the project root to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize FastAPI app
app = FastAPI(title="EV Route Planner API V3 - Dynamic Efficiency with Coverage Analysis", version="3.0.0")

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000", "https://slm-dashboard-e757.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input Schema matching frontend expectations
class EnhancedEVRouteInput(BaseModel):
    source: str  # "lat,lon" format or city name
    destination: str  # "lat,lon" format or city name
    battery: int  # in percentage
    efficiency: float  # km per percentage
    max_charging_stops: int = 10
    enhance: bool = True
    output_path: Optional[str] = None
    format_: str = 'json'
    query: Optional[str] = None

# Coverage Area Input Schema
class CoverageAreaInput(BaseModel):
    current_location: str  # "lat,lon" format
    battery: int  # in percentage
    efficiency: float  # km per percentage
    max_hops: int = 10
    safety_margin: float = 0.3
    resolution: int = 50  # Grid resolution for polygon generation
    analysis_type: str = "both"  # "reachable_stations", "coverage_polygon", or "both"
    include_combined_polygon: bool = True  # NEW: Whether to generate combined polygon instead of individual station coverage
    current_heading_degrees: Optional[float] = None  # NEW: Current heading in degrees (0-360)
    current_speed_kmh: Optional[float] = None  # NEW: Current speed in km/h


DEFAULT_CHARGING_STATIONS = []

try:
    with open(STATIONS_CSV_FILE, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                station = {
                    "lat": float(row["latitude"]),
                    "lon": float(row["longitude"]),
                    "name": row["station_name"]
                }
                DEFAULT_CHARGING_STATIONS.append(station)
            except ValueError as e:
                print(f"Skipping invalid row in stations.csv: {row} ({e})")
    print(f"Loaded {len(DEFAULT_CHARGING_STATIONS)} charging stations from stations.csv")
except FileNotFoundError:
    print(f"stations.csv not found at {STATIONS_CSV_FILE}. Using empty station list.")
except Exception as e:
    print(f"Error reading stations.csv: {e}")

# Import the reorganized EV Route Service V3 with enhanced error handling
route_service = None
service_info = None

try:
    # Try to import the reorganized service (assuming it's saved as ev_route_service_v3.py)
    from services.Local import ev_route_service_v3
    route_service = ev_route_service_v3
    print("✓ Successfully imported EV Route Service V3 (reorganized)")

    # Test that the required function exists
    if hasattr(route_service, 'plan_optimal_ev_route'):
        print("✓ plan_optimal_ev_route function found")
        
        # Get service info if available
        if hasattr(route_service, 'get_service_info'):
            service_info = route_service.get_service_info()
            print(f"✓ Service: {service_info['service_name']} v{service_info['version']}")
            print(f"✓ Algorithm: {service_info['algorithm']}")
            print(f"✓ Features: {len(service_info['features'])} advanced features loaded")
        
        # Test a basic function call to ensure it works
        try:
            test_result = route_service.plan_optimal_ev_route(
                source="6.9271,79.8612",  # Colombo
                destination="6.9271,79.8612",  # Same location for quick test
                battery=50.0,
                efficiency=2.0,
                stations_json="[]",  # Empty stations for test
                max_charging_stops=1,
                google_api_key=None,
                openweather_api_key=None
            )
            if test_result:
                print("✓ Service function test successful")
            else:
                print("⚠ Service function test returned empty result")
        except Exception as test_error:
            print(f"⚠ Service function test failed: {test_error}")
        
    else:
        print("✗ plan_optimal_ev_route function not found")
        available_functions = [attr for attr in dir(route_service) if not attr.startswith('_')]
        print(f"Available functions: {available_functions}")
        
except ImportError as e:
    print(f"Warning: Could not import EV Route Service V3: {e}")
    print("Make sure the services/Local/ev_route_service_v3.py file exists")
    print("and that all required dependencies are installed:")
    print("  pip install geopy requests")
except Exception as e:
    print(f"Unexpected error importing route service: {e}")
    import traceback
    traceback.print_exc()

# Import Coverage Area Service
coverage_service = None
try:
    from services.Local.coverage_area_service_v2 import CoverageAreaService
    coverage_service = CoverageAreaService()
    print("✓ Successfully imported Coverage Area Service")
    print("✓ Advanced optimizations available: pre-filtering, caching, early termination")
except ImportError as e:
    print(f"Warning: Could not import Coverage Area Service: {e}")
    print("Make sure coverage_area_service_v2.py is in the same directory")
    print("Install dependencies: pip install scipy shapely numpy")
except Exception as e:
    print(f"Unexpected error importing coverage area service: {e}")
    import traceback
    traceback.print_exc()

def call_ev_route_service_v3(source: str, destination: str, battery: float, 
                            efficiency: float, stations_json: str, 
                            max_charging_stops: int, google_api_key: str = None,
                            openweather_api_key: str = None,
                            output_path: Optional[str] = None, 
                            format_: str = 'json') -> str:
    """
    Call the EV Route Service V3 with dynamic efficiency and strategic battery utilization
    """
    try:
        if route_service and hasattr(route_service, 'plan_optimal_ev_route'):
            planning_function = getattr(route_service, 'plan_optimal_ev_route')
            
            if callable(planning_function):
                print("Calling EV Route Service V3 with parameters:")
                print(f"  Source: {source}")
                print(f"  Destination: {destination}")
                print(f"  Battery: {battery}%")
                print(f"  Efficiency: {efficiency} km/%")
                print(f"  Max stops: {max_charging_stops}")
                print(f"  Google API Key: {'Available' if google_api_key else 'Not provided'}")
                print(f"  OpenWeather API Key: {'Available' if openweather_api_key else 'Not provided'}")
                
                return planning_function(
                    source=source,
                    destination=destination,
                    battery=battery,
                    efficiency=efficiency,
                    stations_json=stations_json,
                    max_charging_stops=max_charging_stops,
                    google_api_key=google_api_key,
                    openweather_api_key=openweather_api_key,
                    output_path=output_path,
                    format_=format_
                )
            else:
                raise ValueError("plan_optimal_ev_route is not callable")
        else:
            raise ImportError("EV Route Service V3 function not found")
    
    except Exception as e:
        print(f"Error calling EV Route Service V3: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a failure response in the expected JSON format
        error_response = {
            "success": False,
            "message": f"EV Route Service V3 error: {str(e)}",
            "distance_km": None,
            "planned_charging_stops_count": 0,
            "unique_stations_visited": 0,
            "route_summary": [],
            "algorithm_used": "EV Route Service V3 (Error)",
            "google_api_calls_used": 0
        }
        return json.dumps(error_response, indent=2)

def convert_service_response_to_frontend_format(service_result: str):
    """Convert the EV Route Service V3 JSON response to match frontend expectations"""
    try:
        # Parse the JSON response from the service
        result_data = json.loads(service_result)
        
        if not result_data.get("success", False):
            return {
                "success": False,
                "message": result_data.get("message", "Route planning failed"),
                "distance_km": 0,
                "planned_charging_stops_count": result_data.get("planned_charging_stops_count", 0),
                "route_summary": [],
                "unique_stations_visited": result_data.get("unique_stations_visited", 0),
                "algorithm_used": result_data.get("algorithm_used", "Unknown"),
                "google_api_calls_used": result_data.get("google_api_calls_used", 0),
                "partial_route_analysis": result_data.get("partial_route_analysis", None)
            }
        
        # Convert route_summary from service format to frontend format
        route_summary = []
        if "route_summary" in result_data:
            for stop in result_data["route_summary"]:
                # Enhanced conversion to handle V3 service format
                route_stop = {
                    "location": stop.get("location", ""),
                    "category": stop.get("category", ""),
                    "battery_on_arrival_percent": stop.get("battery_on_arrival_percent", 0),
                    "battery_on_departure_percent": stop.get("battery_on_departure_percent", 0),
                    "next_stop_distance_km": stop.get("next_stop_distance_km", 0),
                    "station_name": stop.get("station_name", None),
                    "selection_strategy": stop.get("selection_strategy", ""),
                    "visiting_flag": "Visit" if stop.get("category") in ["Source", "Destination", "Visiting_Charging_Station"] else "Not Visit",
                    # V3 specific fields
                    "progress_towards_destination_km": stop.get("progress_towards_destination_km", 0),
                    "battery_utilization_percent": stop.get("battery_utilization_percent", 0),
                    "efficiency_breakdown": stop.get("efficiency_breakdown", {}),
                    "terrain_analysis": stop.get("terrain_analysis", {}),
                    "weather_conditions": stop.get("weather_conditions", {}),
                    "traffic_conditions": stop.get("traffic_conditions", {})
                }
                route_summary.append(route_stop)
        
        return {
            "success": True,
            "distance_km": result_data.get("distance_km", 0),
            "message": result_data.get("message", "Route planned successfully with dynamic efficiency"),
            "planned_charging_stops_count": result_data.get("planned_charging_stops_count", 0),
            "route_summary": route_summary,
            "unique_stations_visited": result_data.get("unique_stations_visited", 0),
            "google_api_calls_used": result_data.get("google_api_calls_used", 0),
            "cache_hit_rate": result_data.get("cache_hit_rate", "0%"),
            "distance_source": result_data.get("distance_calculation_source", "Unknown"),
            "algorithm_used": result_data.get("algorithm_used", "EV Route Service V3"),
            "efficiency_system": result_data.get("efficiency_system", {}),
            "average_battery_utilization_percent": result_data.get("average_battery_utilization_percent", 0),
            "estimated_arrival_battery_percent": result_data.get("estimated_arrival_battery_percent", 0),
            "strategy_summary": result_data.get("strategy_summary", {}),
            "optimization_goals": result_data.get("optimization_goals", [])
        }
        
    except json.JSONDecodeError as e:
        print(f"Error parsing service response: {e}")
        return {
            "success": False,
            "message": "Error parsing route planning response",
            "distance_km": 0,
            "planned_charging_stops_count": 0,
            "route_summary": []
        }
    except Exception as e:
        print(f"Error converting service response: {e}")
        return {
            "success": False,
            "message": f"Error processing route response: {str(e)}",
            "distance_km": 0,
            "planned_charging_stops_count": 0,
            "route_summary": []
        }

def parse_location(location_str: str):
    """Parse a location string into (lat, lon) tuple"""
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

@app.get("/")
def root():
    service_status = "available" if route_service is not None else "unavailable"
    coverage_status = "available" if coverage_service is not None else "unavailable"
    algorithm_name = service_info.get('algorithm', 'Unknown') if service_info else 'Unknown'
    version = service_info.get('version', '3.0.0') if service_info else '3.0.0'
    
    return {
        "message": "EV Route Planner API V3 - Dynamic Efficiency with Strategic Battery Utilization & Coverage Analysis", 
        "status": "running",
        "version": version,
        "services": {
            "route_planning": {
                "available": route_service is not None,
                "status": service_status,
                "algorithm": algorithm_name
            },
            "coverage_analysis": {
                "available": coverage_service is not None,
                "status": coverage_status,
                "features": ["Reachability Analysis", "Coverage Polygons", "Multi-hop Pathfinding", "Optimization Engine"] if coverage_service else []
            }
        },
        "optimization_strategy": [
            "Strategic station selection (First/Middle/Final)",
            "Dynamic efficiency calculation (elevation + weather + traffic)",
            "Real-time conditions integration",
            "Intelligent battery utilization optimization",
            "Coverage area analysis with multi-hop reachability"
        ],
        "features": [
            "Dynamic efficiency with elevation, weather, traffic factors",
            "Strategic battery utilization algorithms",
            "Google Maps + OpenWeatherMap integration", 
            "Permanent elevation caching + hourly weather updates",
            "A* fallback with battery utilization bias",
            "Comprehensive route analysis and reporting",
            "Partial route analysis for infeasible routes",
            "Battery-aware coverage area analysis",
            "Multi-hop reachability calculations",
            "Optimized polygon generation with pre-filtering"
        ] if service_info else [
            "EV Route Planning (service not loaded)",
            "Coverage Analysis (service not loaded)" if not coverage_service else "Coverage Analysis"
        ]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "services": {
            "route_service_status": "available" if route_service else "unavailable",
            "coverage_service_status": "available" if coverage_service else "unavailable"
        },
        "charging_stations_count": len(DEFAULT_CHARGING_STATIONS),
        "api_version": "3.0.0",
        "algorithm": service_info.get('algorithm', 'Unknown') if service_info else 'Unknown',
        "optimization_focus": "dynamic_efficiency_with_strategic_battery_utilization_and_coverage_analysis",
        "cache_system": "elevation_permanent_weather_hourly" if route_service else "not_available",
        "coverage_optimizations": "pre_filtering_early_termination_caching" if coverage_service else "not_available"
    }

@app.post("/ev-route-plan")
def enhanced_ev_route_plan(input_data: EnhancedEVRouteInput):
    """Main route planning endpoint using EV Route Service V3 with dynamic efficiency"""
    try:
        # Check if service is available
        if route_service is None:
            return {
                "success": False,
                "message": "EV Route Service V3 is not available. Please check that services/Local/ev_route_service_v3.py exists and all dependencies are installed.",
                "distance_km": 0,
                "planned_charging_stops_count": 0,
                "route_summary": []
            }
        
        print(f"Received EV Route Service V3 planning request:")
        print(f"  Source: {input_data.source}")
        print(f"  Destination: {input_data.destination}")
        print(f"  Battery: {input_data.battery}%")
        print(f"  Efficiency: {input_data.efficiency}")
        print(f"  Max charging stops: {input_data.max_charging_stops}")
        print(f"  Strategy: Dynamic efficiency with strategic battery utilization")
        
        # Parse source and destination coordinates
        try:
            source_coords = parse_location(input_data.source)
            dest_coords = parse_location(input_data.destination)
        except ValueError as e:
            return {
                "success": False,
                "message": str(e),
                "distance_km": 0,
                "planned_charging_stops_count": 0,
                "route_summary": []
            }
        
        print(f"  Parsed source: {source_coords}")
        print(f"  Parsed destination: {dest_coords}")
        
        # Convert efficiency: ensure it's in the correct format (km per percent)
        if 0 < input_data.efficiency <= 1:
            km_per_percent = input_data.efficiency  # Already in km per percent
        elif input_data.efficiency > 100:
            km_per_percent = input_data.efficiency / 100  # Convert from km per full charge to km per percent
        else:
            km_per_percent = input_data.efficiency  # Assume already correct

        print(f"  Efficiency (km per percent): {km_per_percent}")

        # Prepare charging stations in the format expected by the V3 service
        charging_stations_list = [[station["lat"], station["lon"]] for station in DEFAULT_CHARGING_STATIONS]
        stations_json = json.dumps(charging_stations_list)
        
        # API keys from environment
        google_api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyALG5CigQX1KFqVkYxAD_2E6BvtNYcHQVY")
        openweather_api_key = os.getenv("OPENWEATHER_API_KEY", None)  # Optional for weather integration
        
        # Format source and destination for the service
        source_str = f"{source_coords[0]},{source_coords[1]}"
        destination_str = f"{dest_coords[0]},{dest_coords[1]}"
        
        print("Calling EV Route Service V3...")
        print(f"  Google API: {'Available' if google_api_key else 'Not available'}")
        print(f"  OpenWeather API: {'Available' if openweather_api_key else 'Not available (will use default efficiency)'}")
        
        # Call the EV Route Service V3
        service_result = call_ev_route_service_v3(
            source=source_str,
            destination=destination_str,
            battery=float(input_data.battery),
            efficiency=km_per_percent,
            stations_json=stations_json,
            max_charging_stops=input_data.max_charging_stops,
            google_api_key=google_api_key,
            openweather_api_key=openweather_api_key,
            output_path=input_data.output_path,
            format_=input_data.format_
        )
        
        # Convert response to frontend format
        frontend_response = convert_service_response_to_frontend_format(service_result)
        
        print(f"EV Route Service V3 completed. Success: {frontend_response['success']}")
        if frontend_response['success']:
            print(f"  Algorithm used: {frontend_response.get('algorithm_used', 'Unknown')}")
            print(f"  Total distance: {frontend_response['distance_km']:.1f} km")
            print(f"  Charging stops: {frontend_response['planned_charging_stops_count']}")
            print(f"  Unique stations: {frontend_response['unique_stations_visited']}")
            print(f"  Route stops: {len(frontend_response['route_summary'])}")
            print(f"  API calls used: {frontend_response.get('google_api_calls_used', 0)}")
            print(f"  Cache hit rate: {frontend_response.get('cache_hit_rate', 'N/A')}")
            print(f"  Avg battery utilization: {frontend_response.get('average_battery_utilization_percent', 0):.1f}%")
            print(f"  Estimated arrival battery: {frontend_response.get('estimated_arrival_battery_percent', 0):.1f}%")
        else:
            print(f"  Failure reason: {frontend_response.get('message', 'Unknown')}")
            if frontend_response.get('partial_route_analysis'):
                print(f"  Partial analysis available: {frontend_response['partial_route_analysis'].get('reason', 'Unknown')}")
        
        return frontend_response
        
    except Exception as e:
        print(f"Error in EV Route Service V3 planning: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"An error occurred while planning the route: {str(e)}",
            "distance_km": 0,
            "planned_charging_stops_count": 0,
            "route_summary": []
        }

@app.post("/coverage-area-optimized")
def coverage_area_optimized(input_data: CoverageAreaInput):
    """
    Coverage area analysis endpoint with integrated alert system + network coverage
    Now includes overall_network_coverage showing theoretical maximum reach of infrastructure
    """
    try:
        # Check if coverage service is available
        if coverage_service is None:
            return {
                "success": False,
                "message": "Coverage Area Service is not available. Please check that coverage_area_service.py exists and install dependencies: pip install scipy shapely numpy",
                "analysis_type": input_data.analysis_type,
                "reachable_stations": [],
                "coverage_areas": {}
            }
        
        print(f"Received Coverage Area Analysis request (with alerts + network coverage):")
        print(f"  Current location: {input_data.current_location}")
        print(f"  Battery: {input_data.battery}%")
        print(f"  Efficiency: {input_data.efficiency} km/%")
        print(f"  Max hops: {input_data.max_hops}")
        print(f"  Safety margin: {input_data.safety_margin}")
        print(f"  Analysis type: {input_data.analysis_type}")
        
        # Parse current location coordinates
        try:
            current_coords = parse_location(input_data.current_location)
        except ValueError as e:
            return {
                "success": False,
                "message": str(e),
                "analysis_type": input_data.analysis_type,
                "reachable_stations": [],
                "coverage_areas": {}
            }
        
        print(f"  Parsed location: {current_coords}")
        
        # Convert efficiency to km per percent
        if 0 < input_data.efficiency <= 1:
            km_per_percent = input_data.efficiency
        elif input_data.efficiency > 100:
            km_per_percent = input_data.efficiency / 100
        else:
            km_per_percent = input_data.efficiency

        print(f"  Efficiency (km per percent): {km_per_percent}")
        
        # Validate analysis type
        if input_data.analysis_type not in ["reachable_stations", "coverage_polygon", "both"]:
            return {
                "success": False,
                "message": "Invalid analysis_type. Must be 'reachable_stations', 'coverage_polygon', or 'both'.",
                "analysis_type": input_data.analysis_type,
                "reachable_stations": [],
                "coverage_areas": {}
            }

        # Use the MAIN INTEGRATED METHOD: analyze_safety_with_alerts
        # This provides comprehensive analysis including alerts, coverage areas, station reachability, and network coverage
        analysis_result = coverage_service.analyze_safety_with_alerts(
            current_position=current_coords,
            current_battery=float(input_data.battery),
            efficiency_km_per_percent=input_data.efficiency,  # Convert to percent per km
            charging_stations=DEFAULT_CHARGING_STATIONS,
            max_hops=input_data.max_hops,
            safety_margin=input_data.safety_margin,
            timestamp=None,  # Uses current time
            current_heading_degrees=input_data.current_heading_degrees if input_data.current_heading_degrees else 90.0,  # Default heading or get from input_data if available
            current_speed_kmh=input_data.current_speed_kmh if input_data.current_speed_kmh else None,  # Get from input_data if available
            is_moving=True,
            include_network_coverage=True  # NEW: Include network coverage analysis
        )
        
        # Calculate reachability percentage
        total_stations = len(DEFAULT_CHARGING_STATIONS)
        reachable_count = analysis_result['station_analysis']['reachable_stations']
        reachability_percentage = (reachable_count / total_stations * 100) if total_stations > 0 else 0
        
        # Build response with ALL keys including network coverage
        response = {
            "success": True,
            "message": "Coverage area analysis completed successfully with integrated alert system and network coverage",
            "analysis_type": "both",
            
            "current_location": analysis_result['current_status'],
            
            "parameters": {
                "max_hops": input_data.max_hops,
                "safety_margin": input_data.safety_margin,
                "resolution": input_data.resolution if hasattr(input_data, 'resolution') else 50,
                "total_stations_available": total_stations
            },
            
            "reachable_stations": {
                "count": reachable_count,
                "stations": analysis_result['station_analysis']['reachable_stations_list']
            },
            
            "unreachable_stations": {
                "count": analysis_result['station_analysis']['unreachable_stations'],
                "stations": analysis_result['station_analysis']['unreachable_stations_list']
            },
            
            "coverage_areas": {
                "direct_coverage": analysis_result['coverage_areas']['direct_coverage'],
                "combined_coverage_polygon": analysis_result['coverage_areas']['combined_coverage_polygon'],
                "overall_network_coverage": analysis_result['coverage_areas']['overall_network_coverage'],  # NEW
                "direct_coverage_points": analysis_result['coverage_areas']['direct_coverage_points'],
                "combined_coverage_points": analysis_result['coverage_areas']['combined_coverage_points']
            },
            
            # Alerts
            "alerts": analysis_result['alerts'],
            
            "coverage_status": analysis_result['coverage_status'],
            
            "travel_metrics": analysis_result['travel_metrics'],
            
            # Dashboard summary
            "dashboard_summary": analysis_result['dashboard_summary'],
            
            "reachability_stats": {
                **analysis_result['reachability_stats'],
                "total_reachable": reachable_count,
                "total_unreachable": analysis_result['station_analysis']['unreachable_stations'],
                "reachability_percentage": round(reachability_percentage, 1)
            },
            
            "optimization_stats": analysis_result.get('optimization_stats', {}),
            
            "analysis_time_seconds": analysis_result['analysis_time_seconds'],
            "timestamp": analysis_result['timestamp']
        }
        
        # Log network coverage info
        if response['coverage_areas']['overall_network_coverage']:
            network_info = response['coverage_areas']['overall_network_coverage']
            print(f"\nNetwork Coverage Analysis:")
            print(f"  Network Connected: {network_info['network_connected']}")
            print(f"  Number of Networks: {network_info['network_count']}")
            print(f"  Total Coverage Area: {network_info['total_coverage_area_km2']} km²")
            print(f"  Max Theoretical Range: {network_info['max_theoretical_range_km']} km")
            
            if network_info['network_count'] > 1:
                print(f"  ⚠️ WARNING: Disconnected networks detected!")
                for network in network_info['networks']:
                    print(f"    - Network {network['network_id']}: {network['station_count']} stations, {network['coverage_area_km2']} km²")
        
        return response
        
    except Exception as e:
        print(f"Error in coverage area analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/stations")
def get_all_stations():
    """Get all available charging stations"""
    return {
        "success": True,
        "count": len(DEFAULT_CHARGING_STATIONS),
        "stations": DEFAULT_CHARGING_STATIONS
    }

@app.get("/route-capabilities")
def get_route_capabilities():
    """Get information about EV Route Service V3 capabilities"""
    capabilities = {
        "service_version": "3.0.0",
        "dynamic_efficiency_algorithm": True,
        "strategic_battery_utilization": True,
        "coverage_analysis": coverage_service is not None,
        "optimization_strategy": [
            "1. Strategic station selection (First/Middle/Final)",
            "2. Dynamic efficiency calculation (elevation + weather + traffic)", 
            "3. Real-time conditions integration",
            "4. Battery utilization optimization",
            "5. Multi-hop reachability analysis",
            "6. Coverage area polygon generation"
        ],
        "efficiency_factors": [
            "Elevation profile (permanent cache)",
            "Weather conditions (hourly updates)",
            "Traffic patterns (time-of-day based)",
            "Base vehicle efficiency"
        ],
        "coverage_features": [
            "Multi-hop pathfinding with battery constraints",
            "Reachable station analysis with optimization",
            "Coverage polygon generation",
            "Pre-filtering for performance",
            "LRU caching for distance calculations",
            "Early termination algorithms"
        ] if coverage_service else [],
        "api_integrations": {
            "google_maps": "Distance calculation + elevation data",
            "openweather": "Weather conditions (optional)",
            "caching_system": "Aggressive caching to minimize API calls"
        },
        "max_charging_stops_range": [1, 50],
        "optimization_goals": [
            "Strategic station selection",
            "Real-time efficiency adaptation", 
            "Battery utilization maximization",
            "Arrival battery optimization",
            "Coverage area maximization",
            "Computational efficiency"
        ],
        "supported_formats": ["json"],
        "route_service_status": "available" if route_service else "unavailable",
        "coverage_service_status": "available" if coverage_service else "unavailable"
    }
    
    if service_info:
        capabilities.update({
            "algorithm_name": service_info['algorithm'],
            "features_count": len(service_info['features']),
            "accuracy_improvements": service_info['accuracy_improvements'],
            "cache_system": service_info['cache_system']
        })
    
    return capabilities

@app.get("/coverage-capabilities")
def get_coverage_capabilities():
    """Get information about Coverage Area Service capabilities"""
    if coverage_service is None:
        return {
            "service_available": False,
            "message": "Coverage Area Service is not available",
            "required_dependencies": ["scipy", "shapely", "numpy"],
            "installation_command": "pip install scipy shapely numpy"
        }
    
    return {
        "service_available": True,
        "service_version": "1.0.0",
        "analysis_types": [
            "reachable_stations",
            "coverage_polygon", 
            "both"
        ],
        "optimization_features": [
            "Theoretical range pre-filtering",
            "LRU cache for distance calculations",
            "Early termination when battery insufficient", 
            "Adaptive resolution based on station density",
            "Batch processing for progress tracking",
            "Intelligent sorting and prioritization"
        ],
        "pathfinding_algorithms": [
            "Multi-hop BFS with battery constraints",
            "Optimized station filtering",
            "Direct reachability checking",
            "Battery-aware path validation"
        ],
        "polygon_generation": [
            "Convex hull boundary creation",
            "Grid-based reachability testing",
            "Adaptive resolution adjustment",
            "Fallback circular boundaries"
        ],
        "performance_optimizations": [
            "Pre-filtering reduces computation by 50-80%",
            "LRU caching for distance calculations",
            "Early termination saves unnecessary computation",
            "Batch processing with progress indicators"
        ],
        "supported_parameters": {
            "max_hops": {"min": 1, "max": 50, "default": 10},
            "safety_margin": {"min": 0.0, "max": 0.5, "default": 0.3},
            "resolution": {"min": 10, "max": 100, "default": 50}
        },
        "dependencies_status": {
            "scipy": "available",
            "shapely": "available", 
            "numpy": "available"
        }
    }

@app.get("/debug/service")
def debug_service():
    """Debug endpoint to check EV Route Service V3 status"""
    if route_service is None:
        return {
            "service_imported": False,
            "error": "EV Route Service V3 could not be imported",
            "suggestions": [
                "Check that services/Local/ev_route_service_v3.py exists",
                "Verify the file contains the reorganized code structure",
                "Install required dependencies: pip install geopy requests",
                "Check for import errors in the service file"
            ]
        }
    
    available_functions = [attr for attr in dir(route_service) if not attr.startswith('_')]
    has_optimizer = hasattr(route_service, 'plan_optimal_ev_route')
    has_service_info = hasattr(route_service, 'get_service_info')
    
    debug_info = {
        "service_imported": True,
        "has_optimizer_function": has_optimizer,
        "has_service_info_function": has_service_info,
        "available_functions": available_functions,
        "service_type": "EV Route Service V3 - Dynamic Efficiency"
    }
    
    if service_info:
        debug_info.update({
            "service_name": service_info['service_name'],
            "version": service_info['version'],
            "algorithm": service_info['algorithm'],
            "features_available": len(service_info['features'])
        })
    
    return debug_info

@app.get("/debug/coverage-service")
def debug_coverage_service():
    """Debug endpoint to check Coverage Area Service status"""
    if coverage_service is None:
        return {
            "service_imported": False,
            "error": "Coverage Area Service could not be imported",
            "suggestions": [
                "Check that coverage_area_service_v2.py exists in the same directory",
                "Install required dependencies: pip install scipy shapely numpy",
                "Verify the service file contains the CoverageAreaService class",
                "Check for import errors in the service file"
            ]
        }
    
    available_methods = [attr for attr in dir(coverage_service) if not attr.startswith('_') and callable(getattr(coverage_service, attr))]
    
    debug_info = {
        "service_imported": True,
        "service_class": "CoverageAreaService",
        "available_methods": available_methods,
        "optimization_stats": coverage_service.optimization_stats,
        "cache_stats": coverage_service.cache_stats,
        "dependencies_available": True
    }
    
    # Test basic functionality
    try:
        test_distance = coverage_service.calculate_distance(0, 0, 1, 1)
        debug_info["distance_calculation_test"] = {
            "success": True,
            "test_distance_km": round(test_distance, 2)
        }
    except Exception as e:
        debug_info["distance_calculation_test"] = {
            "success": False,
            "error": str(e)
        }
    
    return debug_info

@app.get("/debug/test-coverage")
def test_coverage_service():
    """Test endpoint to verify coverage service functionality"""
    if coverage_service is None:
        return {
            "test_status": "failed",
            "error": "Coverage service not available"
        }
    
    try:
        # Test with a simple location (Colombo)
        test_location = (6.9271, 79.8612)
        test_battery = 80.0
        test_efficiency = 2.0
        test_stations = [
            {"lat": 6.9271, "lon": 79.9612, "name": "Test Station 1"},
            {"lat": 7.0271, "lon": 79.8612, "name": "Test Station 2"}
        ]
        
        # Test reachable stations analysis
        reachable, unreachable = coverage_service.analyze_reachable_stations_with_optimizations(
            current_coords=test_location,
            initial_battery_percent=test_battery,
            efficiency_km_per_percent=test_efficiency,
            charging_stations=test_stations,
            max_hops=2,
            safety_margin=0.3
        )
        
        return {
            "test_status": "completed",
            "test_location": test_location,
            "test_parameters": {
                "battery": test_battery,
                "efficiency": test_efficiency,
                "stations_count": len(test_stations)
            },
            "results": {
                "reachable_stations": len(reachable),
                "unreachable_stations": len(unreachable),
                "optimization_stats": coverage_service.optimization_stats
            },
            "success": True
        }
        
    except Exception as e:
        return {
            "test_status": "error",
            "error": str(e),
            "success": False
        }

if __name__ == "__main__":
    import uvicorn
    
    # Ensure required directories exist
    os.makedirs("cache", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("services", exist_ok=True)
    os.makedirs("services/Local", exist_ok=True)
    
    # Create __init__.py files if they don't exist
    init_files = [
        "services/__init__.py",
        "services/Local/__init__.py"
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("# Package initialization file\n")
            print(f"Created {init_file}")
    
    print("Starting EV Route Planner API V3 with Coverage Analysis...")
    print("Features:")
    print("- Dynamic efficiency calculation (elevation + weather + traffic)")
    print("- Strategic battery utilization algorithms") 
    print("- Google Maps + OpenWeatherMap integration")
    print("- Permanent elevation caching + hourly weather updates")
    print("- A* fallback with battery utilization optimization")
    print("- Comprehensive route analysis and partial route handling")
    print("- Battery-aware coverage area analysis with multi-hop reachability")
    print("- Optimized polygon generation with pre-filtering and caching")
    print(f"- Route service status: {'Available' if route_service else 'Unavailable'}")
    print(f"- Coverage service status: {'Available' if coverage_service else 'Unavailable'}")
    
    if route_service is None:
        print("\nWARNING: EV Route Service V3 not available!")
        print("Make sure services/Local/ev_route_service_v3.py exists and contains the reorganized code.")
        print("Install dependencies: pip install geopy requests")
    else:
        print(f"✓ Route Service loaded: {service_info['service_name'] if service_info else 'EV Route Service V3'}")
        if service_info:
            print(f"✓ Algorithm: {service_info['algorithm']}")
            print(f"✓ Version: {service_info['version']}")
    
    if coverage_service is None:
        print("\nWARNING: Coverage Area Service not available!")
        print("Make sure coverage_area_service_v2.py exists in the same directory.")
        print("Install dependencies: pip install scipy shapely numpy")
    else:
        print("✓ Coverage Service loaded with advanced optimizations")
        print("✓ Pre-filtering, caching, and early termination enabled")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)