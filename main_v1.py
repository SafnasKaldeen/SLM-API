from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import sys

# Add the project root to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize FastAPI app
app = FastAPI(title="Distance-Optimized EV Route Planner API", version="2.1.0")

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input Schema matching frontend expectations
class EnhancedEVRouteInput(BaseModel):
    source: str  # "lat,lon" format or city name
    destination: str  # "lat,lon" format or city name
    battery: int  # in percentage
    efficiency: float  # km per percentage (frontend sends as decimal like 0.7)
    max_charging_stops: int = 10
    enhance: bool = True
    output_path: Optional[str] = None
    format_: str = 'json'
    query: Optional[str] = None

# Enhanced charging stations data
DEFAULT_CHARGING_STATIONS = [
    {"lat": 7.123456, "lon": 80.123456, "name": "Miriswaththa_Station"},
    {"lat": 7.148497, "lon": 79.873276, "name": "Seeduwa_Station"},
    {"lat": 7.182689, "lon": 79.961171, "name": "Minuwangoda_Station"},
    {"lat": 7.222404, "lon": 80.017613, "name": "Divulapitiya_Station"},
    {"lat": 7.222445, "lon": 80.017625, "name": "Katunayake_Station"},
    {"lat": 7.120498, "lon": 79.983923, "name": "Udugampola_Station"},
    {"lat": 7.006685, "lon": 79.958184, "name": "Kadawatha_Station"},
    {"lat": 7.274298, "lon": 79.862597, "name": "Kochchikade_Station"},
    {"lat": 6.960975, "lon": 79.880949, "name": "Paliyagoda_Station"},
    {"lat": 6.837024, "lon": 79.903572, "name": "Boralesgamuwa_Station"},
    {"lat": 6.877865, "lon": 79.939505, "name": "Thalawathugoda_Station"},
    {"lat": 6.787022, "lon": 79.884759, "name": "Moratuwa_Station"},
    {"lat": 6.915059, "lon": 79.881394, "name": "Borella_Station"},
    {"lat": 6.847305, "lon": 80.102153, "name": "Padukka_Station"},
    {"lat": 7.222348, "lon": 80.017553, "name": "Beruwala_Station"},
    {"lat": 6.714853, "lon": 79.989208, "name": "Bandaragama_Station"},
    {"lat": 7.222444, "lon": 80.017606, "name": "Maggona_Station"},
    {"lat": 6.713372, "lon": 79.906452, "name": "Panadura_Station"},
    {"lat": 7.581641, "lon": 79.799323, "name": "Chilaw_Station"},
    {"lat": 7.8715, "lon": 80.011, "name": "Anamaduwa_Station"},
    {"lat": 7.2845, "lon": 80.6375, "name": "Kandy_Station"},
    {"lat": 6.9847, "lon": 81.0564, "name": "Badulla_Station"},
    {"lat": 6.1528, "lon": 80.2239, "name": "Matara_Station"},
    {"lat": 8.4947, "lon": 80.1739, "name": "Pemaduwa_Station"},
    {"lat": 7.5742, "lon": 79.8482, "name": "Chilaw_Station_2"},
    {"lat": 7.0094, "lon": 81.0565, "name": "Mahiyangana_Station"},
    {"lat": 7.2531, "lon": 80.3453, "name": "Kegalle_Station"},
]

# Import route service with error handling
# Replace the import section in your main.py (lines 34-41) with this:

# Import route service with error handling
route_service = None

try:
    from services.Local import ev_route_service_v1
    route_service = ev_route_service_v1
    print("Successfully imported EV Route Service V1")
    
    # Test that the required function exists
    if hasattr(route_service, 'plan_optimal_ev_route'):
        print("✓ plan_optimal_ev_route function found")
        
        # Get service info if available
        if hasattr(route_service, 'get_service_info'):
            service_info = route_service.get_service_info()
            print(f"✓ Service: {service_info['service_name']} v{service_info['version']}")
            print(f"✓ Algorithm: {service_info['algorithm']}")
        
    else:
        print("✗ plan_optimal_ev_route function not found")
        available_functions = [attr for attr in dir(route_service) if not attr.startswith('_')]
        print(f"Available functions: {available_functions}")
        
except ImportError as e:
    print(f"Warning: Could not import EV Route Service V1: {e}")
    print("Make sure the services/Local/ev_route_service_v1.py file exists")
    print("and that all required dependencies are installed (geopy)")
except Exception as e:
    print(f"Unexpected error importing route service: {e}")
    import traceback
    traceback.print_exc()

    
def call_distance_optimized_service(source: str, destination: str, battery: float, 
                                   efficiency: float, stations_json: str, 
                                   max_charging_stops: int, google_api_key: str,
                                   output_path: Optional[str] = None, format_: str = 'json') -> str:
    """
    Call the distance-optimized route planning service function.
    Uses the renamed API function to avoid naming conflicts.
    """
    try:
        # Access the API function by its new name
        if route_service and hasattr(route_service, 'plan_optimal_ev_route'):
            planning_function = getattr(route_service, 'plan_optimal_ev_route')
            
            if callable(planning_function):
                return planning_function(
                    source=source,
                    destination=destination,
                    battery=battery,
                    efficiency=efficiency,
                    stations_json=stations_json,
                    max_charging_stops=max_charging_stops,
                    google_api_key=google_api_key,
                    output_path=output_path,
                    format_=format_
                )
            else:
                raise ValueError("plan_optimal_ev_route is not callable")
        else:
            raise ImportError("Distance optimization service function not found")
    
    except Exception as e:
        print(f"Error calling distance optimization service: {e}")
        # Return a failure response in the expected JSON format
        error_response = {
            "success": False,
            "message": f"Distance optimization service error: {str(e)}",
            "distance_km": None,
            "planned_charging_stops_count": 0,
            "route_summary": []
        }
        return json.dumps(error_response, indent=2)

def convert_service_response_to_frontend_format(service_result: str):
    """Convert the service JSON response to match frontend expectations"""
    try:
        # Parse the JSON response from the service
        result_data = json.loads(service_result)
        
        if not result_data.get("success", False):
            return {
                "success": False,
                "message": result_data.get("message", "Route planning failed"),
                "distance_km": 0,
                "planned_charging_stops_count": 0,
                "route_summary": []
            }
        
        # Convert route_summary from service format to frontend format
        route_summary = []
        if "route_summary" in result_data:
            for stop in result_data["route_summary"]:
                # The service already provides the correct format, just ensure all fields are present
                route_stop = {
                    "location": stop.get("location", ""),
                    "category": stop.get("category", ""),
                    "battery_on_arrival_percent": stop.get("battery_on_arrival_percent", 0),
                    "battery_on_departure_percent": stop.get("battery_on_departure_percent", 0),
                    "next_stop_distance_km": stop.get("next_stop_distance_km", 0),
                    "station_name": stop.get("station_name", None),
                    "visiting_flag": "Visit" if stop.get("category") in ["Source", "Destination", "Visiting_Charging_Station"] else "Not Visit"
                }
                route_summary.append(route_stop)
        
        return {
            "success": True,
            "distance_km": result_data.get("distance_km", 0),
            "message": result_data.get("message", "Route planned successfully"),
            "planned_charging_stops_count": result_data.get("planned_charging_stops_count", 0),
            "route_summary": route_summary,
            "unique_stations_visited": result_data.get("unique_stations_visited", 0),
            "google_api_calls_used": result_data.get("google_api_calls_used", 0),
            "distance_source": result_data.get("distance_calculation_source", "Unknown"),
            "algorithm_used": result_data.get("algorithm_used", "Distance Optimization")
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
    return {
        "message": "Distance-Optimized EV Route Planner API", 
        "status": "running",
        "version": "2.1.0",
        "service_available": route_service is not None,
        "optimization_strategy": "Direct -> Single Station -> Multi-Station (minimum distance)",
        "api_efficiency": "Google API only for source/destination distances",
        "features": [
            "Distance-optimized pathfinding",
            "Smart API usage (cached station-to-station)", 
            "Direct route priority",
            "Single station optimization",
            "Multi-station minimum distance"
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
        "api_version": "2.1.0",
        "optimization_focus": "minimum_total_distance"
    }

@app.post("/ev-route-plan")
def optimized_ev_route_plan(input_data: EnhancedEVRouteInput):
    """Main route planning endpoint using distance optimization strategy"""
    try:
        # Check if service is available
        if route_service is None:
            return {
                "success": False,
                "message": "Distance optimization service is not available. Please check that services/Local/ev_route_service.py exists.",
                "distance_km": 0,
                "planned_charging_stops_count": 0,
                "route_summary": []
            }
        
        print(f"Received distance-optimized route planning request:")
        print(f"  Source: {input_data.source}")
        print(f"  Destination: {input_data.destination}")
        print(f"  Battery: {input_data.battery}%")
        print(f"  Efficiency: {input_data.efficiency}")
        print(f"  Max charging stops: {input_data.max_charging_stops}")
        print(f"  Strategy: Direct -> Single -> Multi (min distance)")
        
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
        
        # Convert efficiency: ensure it's in the correct format
        if 0 < input_data.efficiency <= 1:
            km_per_percent = input_data.efficiency  # Already in km per percent
        else:
            km_per_percent = input_data.efficiency / 100 # Convert from km per full charge to km per percent

        print(f"  Efficiency (km per percent): {km_per_percent}")

        # Prepare charging stations in the format expected by the service
        charging_stations_list = [[station["lat"], station["lon"]] for station in DEFAULT_CHARGING_STATIONS]
        stations_json = json.dumps(charging_stations_list)
        
        # Google API key
        google_api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyALG5CigQX1KFqVkYxAD_2E6BvtNYcHQVY")
        
        # Format source and destination for the service
        source_str = f"{source_coords[0]},{source_coords[1]}"
        destination_str = f"{dest_coords[0]},{dest_coords[1]}"
        
        print("Calling distance-optimized route planning service...")
        
        # Call the distance optimization service through our wrapper function
        service_result = call_distance_optimized_service(
            source=source_str,
            destination=destination_str,
            battery=float(input_data.battery),
            efficiency=km_per_percent,
            stations_json=stations_json,
            max_charging_stops=input_data.max_charging_stops,
            google_api_key=google_api_key,
            output_path=input_data.output_path,
            format_=input_data.format_
        )
        
        # Convert response to frontend format
        frontend_response = convert_service_response_to_frontend_format(service_result)
        
        print(f"Distance optimization completed. Success: {frontend_response['success']}")
        if frontend_response['success']:
            print(f"  Strategy used: {frontend_response.get('algorithm_used', 'Unknown')}")
            print(f"  Total distance: {frontend_response['distance_km']:.1f} km")
            print(f"  Charging stops: {frontend_response['planned_charging_stops_count']}")
            print(f"  Route stops: {len(frontend_response['route_summary'])}")
            print(f"  API calls used: {frontend_response.get('google_api_calls_used', 0)}")
        
        return frontend_response
        
    except Exception as e:
        print(f"Error in distance-optimized route planning: {e}")
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
    """Get information about distance optimization capabilities"""
    return {
        "distance_optimization_algorithm": True,
        "optimization_strategy": [
            "1. Direct route (if possible)",
            "2. Single station route (minimum total distance)",
            "3. Multi-station route (minimum total distance)"
        ],
        "api_efficiency": {
            "google_api_usage": "Only for source/destination distances",
            "station_distances": "Pre-computed and cached",
            "caching": "Aggressive caching to minimize API calls"
        },
        "max_charging_stops_range": [1, 50],
        "optimization_goals": ["minimum_total_distance"],
        "supported_formats": ["json"],
        "service_status": "available" if route_service else "unavailable",
        "features": {
            "direct_route_priority": "Always try direct route first",
            "single_station_optimization": "Find single station with minimum total distance",
            "multi_station_fallback": "Multi-station route only if needed",
            "smart_api_usage": "Google API only for source/destination, cached for station-to-station",
            "distance_focus": "Primary optimization criterion is total distance"
        }
    }

@app.get("/debug/service")
def debug_service():
    """Debug endpoint to check service status"""
    if route_service is None:
        return {
            "service_imported": False,
            "error": "Distance optimization service could not be imported",
            "suggestions": [
                "Check that services/Local/ev_route_service.py exists",
                "Verify the file contains the plan_optimal_ev_route function",
                "Check for import errors in the service file"
            ]
        }
    
    available_functions = [attr for attr in dir(route_service) if not attr.startswith('_')]
    has_optimizer = hasattr(route_service, 'plan_optimal_ev_route')
    
    return {
        "service_imported": True,
        "has_optimizer_function": has_optimizer,
        "available_functions": available_functions,
        "optimizer_type": str(type(getattr(route_service, 'plan_optimal_ev_route', None))),
        "service_type": "Distance-Optimized Route Planning"
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
    
    print("Starting Enhanced EV Route Planner API...")
    print("Features:")
    print("- Vector projection pathfinding algorithm")
    print("- Backtracking for optimal route finding") 
    print("- Google Maps integration with smart caching")
    print("- Enhanced battery management with reserves")
    print("- Frontend compatibility maintained")
    print(f"- Route service status: {'Available' if route_service else 'Unavailable'}")
    
    if route_service is None:
        print("\nWARNING: Route service not available!")
        print("Make sure services/Local/ev_route_service.py exists and is properly formatted.")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)