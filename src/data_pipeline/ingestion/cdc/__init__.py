"""
Change Data Capture (CDC) Module

This module contains CDC components for real-time data change detection.
"""

from .postgres_cdc import PostgresCDC
from .kafka_cdc_connector import KafkaCDCConnector
from .change_event_processor import ChangeEventProcessor, ChangeEventType
from .conflict_resolver import ConflictResolver, ConflictResolutionStrategy

__all__ = [
    "PostgresCDC",
    "KafkaCDCConnector",
    "ChangeEventProcessor",
    "ChangeEventType",
    "ConflictResolver",
    "ConflictResolutionStrategy"
]