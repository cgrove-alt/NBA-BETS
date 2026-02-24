"""
Environment loader — ensures .env variables are available in os.environ.

Import this module at the top of any entry point script:
    import load_env  # noqa: F401

On Railway/production, env vars are injected by the platform and this is a no-op.
Locally, this loads from the .env file at the project root.

python-dotenv's load_dotenv() never overwrites existing env vars, so it's
always safe to call — production values always take precedence.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve .env relative to this file (project root)
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)
