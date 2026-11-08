import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The app's DATABASE_URL is relative, so tests and app must share a working dir.
os.chdir(ROOT)
