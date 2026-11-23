"""
Put the project root on the import path so `import app...` works when pytest is
run from anywhere.

This is the only path adjustment the test suite needs. The application packages
import each other absolutely (`from app.documents.models import record`), so
nothing else has to be arranged.
"""

import os
import sys
from pathlib import Path

# Tests build services without a Qdrant; the CAG tests inject their own core.
os.environ.setdefault("CAG_ENABLED", "false")

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
