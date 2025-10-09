"""
ML Pipelines Module for PBF-LB/M Data Pipeline

This module provides comprehensive ML pipeline capabilities for:
- Training pipelines for model development
- Inference pipelines for real-time and batch prediction
- Evaluation pipelines for model validation and testing

All pipelines integrate with the YAML configuration system and support
the complete ML lifecycle from data ingestion to model deployment.
"""

from .training import *
from .inference import *
from .evaluation import *

__version__ = "1.0.0"
__author__ = "PBF-LB/M Data Pipeline Team"
