# Correct import (adjust the path if your project structure is different)
from utils.snowflake_connector import get_snowflake_session
import json
from snowflake.snowpark import Session

def call_ev_route_planner(session: Session, source: str, destination: str, battery: int, efficiency: float):
    """
    Calls the EV_ROUTE_PLANNER stored procedure in Snowflake to get an EV route plan.

    Args:
        session (Session): Active Snowflake session.
        source (str): Starting point for the route.
        destination (str): Ending point for the route.
        battery (int): Current battery level of the EV.
        efficiency (float): Efficiency of the EV (e.g., km per kWh).

    Returns:
        dict: Parsed JSON response from the stored procedure.
    """
    
    # Construct the stored procedure call with formatted arguments.
    # - Note: Ensure that `source` and `destination` strings are properly escaped if they can contain special characters.
    sp_call = f"""
        CALL EV_ROUTE_PLANNER(
            '{source}',                          -- source location
            '{destination}',                     -- destination location
            {battery},                           -- EV battery level
            {efficiency},                        -- EV efficiency
            TO_VARIANT('{json.dumps([
                ["7.123456", "80.123456"], ["7.148497", "79.873276"], ["7.182689", "79.961171"],
                ["7.222404", "80.017613"], ["7.222445", "80.017625"], ["7.120498", "79.983923"],
                ["7.006685", "79.958184"], ["7.274298", "79.862597"], ["6.960975", "79.880949"],
                ["6.837024", "79.903572"], ["6.877865", "79.939505"], ["6.787022", "79.884759"],
                ["6.915059", "79.881394"], ["6.847305", "80.102153"], ["7.222348", "80.017553"],
                ["6.714853", "79.989208"], ["7.222444", "80.017606"], ["6.713372", "79.906452"]
            ])}'),                             -- stations_json as a variant
            '@ROUTE_PLANNER/plan.csv',          -- output file path in Snowflake stage
            'csv',                              -- output file format
            False,                              -- enhance flag (currently not used)
            ''                                  -- optional query
        );
    """

    # Execute the stored procedure
    result = session.sql(sp_call).collect()

    # Retrieve and parse the nested JSON result
    raw_json_str = result[0]['EV_ROUTE_PLANNER']  # This is a stringified JSON
    decoded_str = json.loads(raw_json_str)        # Decode outer JSON string
    parsed_json = json.loads(decoded_str)         # Parse inner JSON string

    return parsed_json
