# eliana_soul/config.py

from dotenv import load_dotenv
import os

# Load variables from the .env file into the environment
load_dotenv()

# Retrieve the OpenAI key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Optional: warn if the key is missing
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY is not set in the .env file.")