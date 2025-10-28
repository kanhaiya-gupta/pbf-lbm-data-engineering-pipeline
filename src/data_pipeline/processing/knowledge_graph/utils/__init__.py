"""
Knowledge Graph Utilities

This module contains utility functions for knowledge graph processing.
"""

from .json_parser import safe_json_loads, safe_json_loads_with_fallback

__all__ = ['safe_json_loads', 'safe_json_loads_with_fallback']
