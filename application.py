import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))

try:
    from src.api.main import app

    application = app
except Exception as e:
    import traceback

    print("=" * 50, flush=True)
    print("STARTUP ERROR:", flush=True)
    print(traceback.format_exc(), flush=True)
    print("=" * 50, flush=True)
    raise
