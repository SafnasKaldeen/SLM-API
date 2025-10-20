from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import sys
import csv

# Path to the stations CSV (same folder as this script)
STATIONS_CSV_FILE = os.path.join(os.path.dirname(__file__), "stations.csv")

# Add the project root to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize FastAPI app
app = FastAPI(title="EV Route Planner API V3 - Dynamic Efficiency", version="3.0.0")

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

@app.get("/")
def root():
    service_status = "available" if route_service is not None else "unavailable"
    algorithm_name = service_info.get('algorithm', 'Unknown') if service_info else 'Unknown'
    version = service_info.get('version', '3.0.0') if service_info else '3.0.0'
    
    return {
        "message": "EV Route Planner API V3 - Dynamic Efficiency with Strategic Battery Utilization", 
        "status": "running",
        "version": version,
        "service_available": route_service is not None,
        "service_status": service_status,
        "algorithm": algorithm_name,
        "optimization_strategy": [
            "Strategic station selection (First/Middle/Final)",
            "Dynamic efficiency calculation (elevation + weather + traffic)",
            "Real-time conditions integration",
            "Intelligent battery utilization optimization"
        ],
        "features": [
            "Dynamic efficiency with elevation, weather, traffic factors",
            "Strategic battery utilization algorithms",
            "Google Maps + OpenWeatherMap integration", 
            "Permanent elevation caching + hourly weather updates",
            "A* fallback with battery utilization bias",
            "Comprehensive route analysis and reporting",
            "Partial route analysis for infeasible routes"
        ] if service_info else [
            "EV Route Planning (service not loaded)"
        ]
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

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service_status": "available" if route_service else "unavailable",
        "charging_stations_count": len(DEFAULT_CHARGING_STATIONS),
        "api_version": "3.0.0",
        "algorithm": service_info.get('algorithm', 'Unknown') if service_info else 'Unknown',
        "optimization_focus": "dynamic_efficiency_with_strategic_battery_utilization",
        "cache_system": "elevation_permanent_weather_hourly" if route_service else "not_available"
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
        "optimization_strategy": [
            "1. Strategic station selection (First/Middle/Final)",
            "2. Dynamic efficiency calculation (elevation + weather + traffic)", 
            "3. Real-time conditions integration",
            "4. Battery utilization optimization"
        ],
        "efficiency_factors": [
            "Elevation profile (permanent cache)",
            "Weather conditions (hourly updates)",
            "Traffic patterns (time-of-day based)",
            "Base vehicle efficiency"
        ],
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
            "Arrival battery optimization"
        ],
        "supported_formats": ["json"],
        "service_status": "available" if route_service else "unavailable"
    }
    
    if service_info:
        capabilities.update({
            "algorithm_name": service_info['algorithm'],
            "features_count": len(service_info['features']),
            "accuracy_improvements": service_info['accuracy_improvements'],
            "cache_system": service_info['cache_system']
        })
    
    return capabilities

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

@app.get("/debug/test-service")
def test_service():
    """Test endpoint to verify service functionality"""
    if route_service is None:
        return {
            "test_status": "failed",
            "error": "Service not available"
        }
    
    try:
        # Test with a simple route (Colombo to nearby location)
        test_result = call_ev_route_service_v3(
            source="6.9271,79.8612",  # Colombo
            destination="6.9271,79.9612",  # Slightly east of Colombo
            battery=80.0,
            efficiency=2.0,
            stations_json=json.dumps([[7.148497, 79.873276], [7.182689, 79.961171]]),  # 2 test stations
            max_charging_stops=2,
            google_api_key=None,  # Test without API
            openweather_api_key=None
        )
        
        result_data = json.loads(test_result)
        
        return {
            "test_status": "completed",
            "success": result_data.get("success", False),
            "message": result_data.get("message", "No message"),
            "algorithm_used": result_data.get("algorithm_used", "Unknown"),
            "test_distance": result_data.get("distance_km", 0),
            "test_api_calls": result_data.get("google_api_calls_used", 0)
        }
        
    except Exception as e:
        return {
            "test_status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    
    # Ensure required directories exist
    os.makedirs("cache", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("services", exist_ok=True)
    os.makedirs("services/Local", exist_ok=True)
    
    # Create _init_.py files if they don't exist
    init_files = [
        "services/_init_.py",
        "services/Local/_init_.py"
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("# Package initialization file\n")
            print(f"Created {init_file}")
    
    print("Starting EV Route Planner API V3...")
    print("Features:")
    print("- Dynamic efficiency calculation (elevation + weather + traffic)")
    print("- Strategic battery utilization algorithms") 
    print("- Google Maps + OpenWeatherMap integration")
    print("- Permanent elevation caching + hourly weather updates")
    print("- A* fallback with battery utilization optimization")
    print("- Comprehensive route analysis and partial route handling")
    print(f"- Route service status: {'Available' if route_service else 'Unavailable'}")
    
    if route_service is None:
        print("\nWARNING: EV Route Service V3 not available!")
        print("Make sure services/Local/ev_route_service_v3.py exists and contains the reorganized code.")
        print("Install dependencies: pip install geopy requests")
    else:
        print(f"✓ Service loaded: {service_info['service_name'] if service_info else 'EV Route Service V3'}")
        if service_info:
            print(f"✓ Algorithm: {service_info['algorithm']}")
            print(f"✓ Version: {service_info['version']}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)