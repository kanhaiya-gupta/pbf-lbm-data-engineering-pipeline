"""
ML/AI Integration Module for PBF-LB/M Data Pipeline

This module provides comprehensive Machine Learning and Artificial Intelligence capabilities
for the Powder Bed Fusion - Laser Beam/Metal manufacturing process. It includes:

- Process Optimization: Laser parameter prediction, build strategy optimization
- Defect Detection: Real-time defect prediction and classification
- Quality Assessment: Quality scoring and dimensional accuracy prediction
- Predictive Maintenance: Equipment health monitoring and failure prediction
- Material Analysis: Material property prediction and microstructure analysis

The module is organized into:
- config/: Configuration management for YAML-based ML configurations
- models/: ML model implementations for different use cases
- pipelines/: Training, inference, and evaluation pipelines
- features/: Feature engineering for various data sources
- serving/: Model serving and API implementations
- monitoring/: Model monitoring, drift detection, and explainability
- utils/: ML utilities for data loading, preprocessing, and evaluation
"""

from .config import ConfigManager, ConfigLoader

__version__ = "1.0.0"
__author__ = "ML Team"

__all__ = [
    'ConfigManager',
    'ConfigLoader',
]