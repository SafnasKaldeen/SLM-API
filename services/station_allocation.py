from utils.snowflake_connector import get_snowflake_session
import json
from snowflake.snowpark import Session

def _parse_sp_result(raw_value, proc_name: str):
    """
    Parses the result from a stored procedure call which may be a double-encoded JSON string or dict.
    """
    try:
        if isinstance(raw_value, dict):
            return raw_value
        elif isinstance(raw_value, str):
            parsed = json.loads(raw_value)
            if isinstance(parsed, str):  # Nested JSON string
                return json.loads(parsed)
            return parsed
        else:
            raise TypeError(f"Unexpected return type from {proc_name}: {type(raw_value)}")
    except Exception as e:
        print(f"❌ Error parsing result from {proc_name}: {e}")
        return None

def call_density_based_station_allocation(
    session: Session,
    eps: float = 0.001,
    min_samples: int = 50,
    top_n: int = 5,
    zoom_level: int = 13,
    stage_name: str = '@STATION_ALLOCATION'
):
    """
    Calls the DensityBasedStationAllocation stored procedure.
    """
    sp_call = f"""
        CALL DensityBasedStationAllocation(
            {eps},
            {min_samples},
            {top_n},
            {zoom_level},
            '{stage_name}'
        );
    """

    result = session.sql(sp_call).collect()
    raw_value = result[0]['DENSITYBASEDSTATIONALLOCATION']
    return _parse_sp_result(raw_value, "DensityBasedStationAllocation")


def call_geo_based_station_allocation(
    session: Session,
    max_radius_km: float = 5,
    outlier_threshold_km: float = 2.5,
    top_n: int = 5,
    zoom_level: int = 13,
    stage_name: str = '@STATION_ALLOCATION'
):
    """
    Calls the GeoBasedStationAllocation stored procedure.
    """
    sp_call = f"""
        CALL GeoBasedStationAllocation(
            {max_radius_km},
            {outlier_threshold_km},
            {top_n},
            {zoom_level},
            '{stage_name}'
        );
    """

    result = session.sql(sp_call).collect()
    raw_value = result[0]['GEOBASEDSTATIONALLOCATION']
    return _parse_sp_result(raw_value, "GeoBasedStationAllocation")
