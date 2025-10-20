import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.5"))

# Default values
DEFAULT_KM_PER_PERCENT = float(os.getenv("DEFAULT_KM_PER_PERCENT", "1.4"))
