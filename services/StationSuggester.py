import json
import math
import re
from snowflake.snowpark import Session

def json_serializable_converter(obj):
    """
    Convert non-JSON-serializable Python objects into JSON-friendly values.
    - NaN and Infinity are converted to `None`
    - Recursively processes nested structures
    """
    if isinstance(obj, (list, tuple)):
        return [json_serializable_converter(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: json_serializable_converter(value) for key, value in obj.items()}
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj

def _parse_json_response(raw_value, fallback_field: str):
    """
    Generic parser to handle Snowflake stored procedure JSON output.
    """
    try:
        if isinstance(raw_value, dict):
            return json_serializable_converter(raw_value)
        elif isinstance(raw_value, str):
            raw_value = raw_value.replace("'", '"')
            raw_value = raw_value.replace("True", "true").replace("False", "false")
            raw_value = re.sub(r'\b(NaN|nan)\b', 'null', raw_value)
            raw_value = re.sub(r'\b(-?Infinity|-?infinity)\b', '"\\1"', raw_value)

            parsed = json.loads(raw_value)
            if isinstance(parsed, str):
                return json_serializable_converter(json.loads(parsed))
            return json_serializable_converter(parsed)
        else:
            raise TypeError(f"Unexpected type for field '{fallback_field}': {type(raw_value)}")
    except Exception as e:
        print(f"❌ Error parsing value for '{fallback_field}': {e}")
        return None

def ClosestStation(session: Session, stage_name: str = '@CLOSEST_STATION'):
    """
    Calls the `ClosestStation` stored procedure.
    """
    sp_call = f"CALL ClosestStation('{stage_name}');"

    try:
        result = session.sql(sp_call).collect()
        if not result or 'CLOSESTSTATION' not in result[0]:
            raise ValueError("Expected field 'CLOSESTSTATION' not found in result.")

        return _parse_json_response(result[0]['CLOSESTSTATION'], 'CLOSESTSTATION')
    except Exception as e:
        print(f"❌ Error in ClosestStation: {e}")
        return None

def ClosestStationWithDirection(session: Session, stage_name: str = '@CLOSEST_STATION'):
    """
    Calls the `ClosestStationWithDirection` stored procedure.
    """
    sp_call = f"CALL ClosestStationWithDirection('{stage_name}');"

    try:
        result = session.sql(sp_call).collect()
        if not result or 'CLOSESTSTATIONWITHDIRECTION' not in result[0]:
            raise ValueError("Expected field 'CLOSESTSTATIONWITHDIRECTION' not found in result.")

        return _parse_json_response(result[0]['CLOSESTSTATIONWITHDIRECTION'], 'CLOSESTSTATIONWITHDIRECTION')
    except Exception as e:
        print(f"❌ Error in ClosestStationWithDirection: {e}")
        return None
    finally:
        # Always close the session
        session.close()
