"""Shared logging setup for the ideas/ package."""

import logging
import os
import sys
from pathlib import Path


def load_dotenv():
    """Load .env from the repo root (two levels up from this file) into os.environ.

    Only sets keys that are missing or empty in the current environment, so
    shell-exported values always take precedence.  Uses bare key=value parsing
    (no quoting, no comments within values) — sufficient for API key files.
    """
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip().removeprefix("export").strip()
            if key:
                os.environ[key] = value.strip()

_LOG_FILE = Path(__file__).parent / "results" / "godel.log"


def setup(name: str = "godel") -> logging.Logger:
    """Configure and return a logger writing to console (INFO) and file (DEBUG)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    # Console — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File — DEBUG and above (full detail)
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(_LOG_FILE)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
