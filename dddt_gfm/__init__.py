"""ddDT-GFM: data-driven Digital Twin Graph Foundation Models for fleet-level
system performance prediction using spatiotemporal graph neural networks.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ddDT-GFM")
except PackageNotFoundError:  # package is not installed
    __version__ = "0.0.0"
