from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from utils.snowflake_connector import get_snowflake_session
from fastapi.middleware.cors import CORSMiddleware
from services.Route_planner import call_ev_route_planner
from services.station_allocation import (
    call_density_based_station_allocation,
    call_geo_based_station_allocation,
)
from services.StationSuggester import (
    ClosestStation,
    ClosestStationWithDirection,
)

# Initialize FastAPI app
app = FastAPI()

# CORS settings (optional)
# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any origin, you can replace "*" with your frontend URL like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow any HTTP methods
    allow_headers=["*"],  # Allow any headers
)

# ------------------------- #
#        Input Schemas      #
# ------------------------- #

# Input for EV route planning
class EVRouteInput(BaseModel):
    source: str
    destination: str
    battery: int  # in percentage
    efficiency: float  # e.g., km/kWh

# Input for density-based clustering of stations
class StationAllocationInput(BaseModel):
    eps: float = 0.001  # DBSCAN epsilon parameter
    min_samples: int = 50  # DBSCAN minimum samples
    top_n: int = 5  # Top N stations to return
    zoom_level: int = 13  # Used for map rendering or clustering
    stage_name: str = '@STATION_ALLOCATION'  # Snowflake stage

# Input for geo-based station allocation
class GeoStationAllocationInput(BaseModel):
    max_radius_km: float = 5
    outlier_threshold_km: float = 2.5
    top_n: int = 5
    zoom_level: int = 13
    stage_name: str = '@STATION_ALLOCATION'

# Input for finding closest stations
class ClosestStationInput(BaseModel):
    stage_name: str = '@CLOSEST_STATION'

# Input for finding closest stations with direction info
class ClosestStationWithDirectionInput(BaseModel):
    stage_name: str = '@CLOSEST_STATION'

# ------------------------- #
#          Routes           #
# ------------------------- #

@app.post("/ev-route-plan")
def ev_route_plan(input: EVRouteInput):
    """Plan an EV route based on source, destination, battery, and efficiency."""
    try:
        session = get_snowflake_session()
        data = call_ev_route_planner(
            session,
            input.source,
            input.destination,
            input.battery,
            input.efficiency,
        )
        return {"status": "success", "data": data}
    except Exception as e:
        # Always log the error for debugging
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'session' in locals():
            session.close()  # Ensure session is closed properly

@app.post("/DensityBased-station-allocation")
def station_allocation(input: StationAllocationInput):
    """Cluster stations using density-based clustering (e.g., DBSCAN)."""
    try:
        session = get_snowflake_session()
        data = call_density_based_station_allocation(
            session=session,
            eps=input.eps,
            min_samples=input.min_samples,
            top_n=input.top_n,
            zoom_level=input.zoom_level,
            stage_name=input.stage_name
        )
        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'session' in locals():
            session.close()

@app.post("/GeoBased-station-allocation")
def geo_station_allocation(input: GeoStationAllocationInput):
    """Allocate stations using geospatial filtering and radius-based logic."""
    try:
        session = get_snowflake_session()
        data = call_geo_based_station_allocation(
            session=session,
            max_radius_km=input.max_radius_km,
            outlier_threshold_km=input.outlier_threshold_km,
            top_n=input.top_n,
            zoom_level=input.zoom_level,
            stage_name=input.stage_name
        )
        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'session' in locals():
            session.close()

@app.post("/closest-station")
def closest_station(input: ClosestStationInput):
    """Return the closest station based on current coordinates or context."""
    try:
        session = get_snowflake_session()
        data = ClosestStation(
            session=session,
            stage_name=input.stage_name
        )
        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'session' in locals():
            session.close()

@app.post("/closest-station-with-direction")
def closest_station_with_direction(input: ClosestStationWithDirectionInput):
    """Return closest station along with directional context (e.g., towards destination)."""
    try:
        session = get_snowflake_session()
        data = ClosestStationWithDirection(
            session=session,
            stage_name=input.stage_name
        )
        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'session' in locals():
            session.close()