"""Allow running as: python -m src.openclaw"""
import sys
import logging

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

from .server import mcp

mcp.run()