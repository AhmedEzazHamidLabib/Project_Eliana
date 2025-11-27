"""
config.py
---------
Configuration loader for the Eliana system.

This module is responsible for loading environment variables,
including API keys or other sensitive runtime settings.

Important:
- This repository intentionally does *not* include any `.env` file.
- Users must provide their own environment variables locally.
- Only template configuration structure is provided here.
- This prevents reproduction of the full system, preserving
  proprietary components and security boundaries.

Functions and Variables:
- Loads `.env` file using python-dotenv (if present).
- Exposes OPENAI_API_KEY as a runtime environment variable.

Usage:
Import this module wherever API access is required. Example:

    from eliana_soul.config import OPENAI_API_KEY
"""

from dotenv import load_dotenv
import os

# Load variables from the .env file into the environment
load_dotenv()

# Retrieve the OpenAI key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Optional: warn if the key is missing
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY is not set in the .env file.")
