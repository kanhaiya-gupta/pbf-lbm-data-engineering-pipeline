"""
External Software Dependencies Module

This module contains external software packages integrated into the PBF-LB/M Data Pipeline.
These are specialized libraries for handling PBF-LB/M machine build files and data processing.

External Packages:
- libSLM: C++ library with Python bindings for PBF-LB/M build file formats
- PySLM: High-level Python library for PBF-LB/M data analysis and visualization

See README.md for detailed information about each package.
"""

import os
import sys
from pathlib import Path

# Get the external directory path
EXTERNAL_DIR = Path(__file__).parent

# libSLM paths
LIBSLM_DIR = EXTERNAL_DIR / "libSLM"
LIBSLM_PYTHON_DIR = LIBSLM_DIR / "python" / "libSLM"

# PySLM paths  
PYSLM_DIR = EXTERNAL_DIR / "pyslm"
PYSLM_PACKAGE_DIR = PYSLM_DIR / "pyslm"

# Check availability of external packages
LIBSLM_AVAILABLE = False
PYSLM_AVAILABLE = False

# Check libSLM availability
try:
    if LIBSLM_PYTHON_DIR.exists():
        # Add libSLM to Python path if not already there
        libslm_path = str(LIBSLM_PYTHON_DIR)
        if libslm_path not in sys.path:
            sys.path.insert(0, libslm_path)
        
        # Try to import libSLM modules
        try:
            import slm
            import translators
            # Check if translators has the expected modules
            if hasattr(translators, 'mtt') and hasattr(translators, 'eos') and hasattr(translators, 'realizer') and hasattr(translators, 'slmsol'):
                LIBSLM_AVAILABLE = True
        except ImportError:
            # libSLM Python bindings not available
            pass
except Exception:
    pass

# Check PySLM availability
try:
    if PYSLM_PACKAGE_DIR.exists():
        # Add PySLM to Python path if not already there
        pyslm_path = str(PYSLM_DIR)
        if pyslm_path not in sys.path:
            sys.path.insert(0, pyslm_path)
        
        # Try to import PySLM
        try:
            import pyslm
            PYSLM_AVAILABLE = True
        except ImportError:
            # PySLM not available
            pass
except Exception:
    pass

# Package information
__version__ = "1.0.0"
__author__ = "PBF-LB/M Research Team"
__description__ = "External software dependencies for PBF-LB/M data pipeline"

# Export availability flags
__all__ = [
    "LIBSLM_AVAILABLE",
    "PYSLM_AVAILABLE", 
    "LIBSLM_DIR",
    "PYSLM_DIR",
    "EXTERNAL_DIR"
]

# Log availability status
import logging
logger = logging.getLogger(__name__)

if LIBSLM_AVAILABLE:
    logger.info("libSLM external package is available")
else:
    logger.warning("libSLM external package is not available")

if PYSLM_AVAILABLE:
    logger.info("PySLM external package is available")
else:
    logger.warning("PySLM external package is not available")
