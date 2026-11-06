import os
from pathlib import Path

from dotenv import load_dotenv

# Anchor to the project root so the .env is found regardless of the working directory.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
