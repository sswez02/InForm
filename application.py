import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))

from src.api.main import app

application = app
