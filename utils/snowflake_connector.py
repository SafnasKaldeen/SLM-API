import configparser
import os
from snowflake.snowpark import Session

def get_snowflake_session():
    # Path to the Snowflake configuration file
    config_path = 'D:/SL_Mobility/API/snowflake_config.ini'

    # Check if configuration file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError("❌ Configuration file 'snowflake_config.ini' not found.")

    # Parse the configuration file
    config = configparser.ConfigParser()
    config.read(config_path)

    # Ensure the 'snowflake' section exists in the configuration
    if 'snowflake' not in config:
        raise ValueError("❌ 'snowflake' section not found in the configuration file.")

    snowflake_config = config['snowflake']

    # Prepare Snowflake connection parameters
    snowflake_params = {
        'account': snowflake_config['account'],
        'user': snowflake_config['user'],
        'password': snowflake_config['password'],
        'role': snowflake_config['role'],
        'warehouse': snowflake_config['warehouse'],
        'database': snowflake_config['database'],
        'schema': snowflake_config['schema'],
        'ssl': False  # Adjust if necessary
    }

    # Establish and return a Snowflake session
    return Session.builder.configs(snowflake_params).create()
