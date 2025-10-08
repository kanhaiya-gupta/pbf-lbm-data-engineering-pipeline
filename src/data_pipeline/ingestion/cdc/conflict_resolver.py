"""
Conflict Resolver for PBF-LB/M Data Pipeline

This module provides conflict resolution capabilities for change data capture events.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ConflictResolutionStrategy(Enum):
    """Conflict resolution strategies."""
    LAST_WRITE_WINS = "last_write_wins"
    FIRST_WRITE_WINS = "first_write_wins"
    MERGE = "merge"
    CUSTOM = "custom"
    REJECT = "reject"


class ConflictResolver:
    """
    Resolver for conflicts in change data capture events.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize conflict resolver.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.resolution_strategies = {}
        self.custom_resolvers = {}
        self.conflict_stats = {
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "conflicts_failed": 0,
            "start_time": datetime.now()
        }
    
    def set_resolution_strategy(self, table_name: str, strategy: ConflictResolutionStrategy) -> None:
        """
        Set conflict resolution strategy for a table.
        
        Args:
            table_name: Name of the table
            strategy: Resolution strategy
        """
        self.resolution_strategies[table_name] = strategy
        logger.info(f"Set resolution strategy for table {table_name}: {strategy.value}")
    
    def add_custom_resolver(self, table_name: str, resolver: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]) -> None:
        """
        Add custom conflict resolver for a table.
        
        Args:
            table_name: Name of the table
            resolver: Custom resolver function
        """
        self.custom_resolvers[table_name] = resolver
        logger.info(f"Added custom resolver for table {table_name}")
    
    def resolve_conflict(self, table_name: str, existing_record: Dict[str, Any], new_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Resolve conflict between existing and new records.
        
        Args:
            table_name: Name of the table
            existing_record: Existing record data
            new_record: New record data
            
        Returns:
            Optional[Dict[str, Any]]: Resolved record or None if conflict cannot be resolved
        """
        try:
            self.conflict_stats["conflicts_detected"] += 1
            
            # Get resolution strategy for table
            strategy = self.resolution_strategies.get(table_name, ConflictResolutionStrategy.LAST_WRITE_WINS)
            
            # Resolve conflict based on strategy
            resolved_record = None
            
            if strategy == ConflictResolutionStrategy.LAST_WRITE_WINS:
                resolved_record = self._resolve_last_write_wins(existing_record, new_record)
            elif strategy == ConflictResolutionStrategy.FIRST_WRITE_WINS:
                resolved_record = self._resolve_first_write_wins(existing_record, new_record)
            elif strategy == ConflictResolutionStrategy.MERGE:
                resolved_record = self._resolve_merge(existing_record, new_record)
            elif strategy == ConflictResolutionStrategy.CUSTOM:
                resolved_record = self._resolve_custom(table_name, existing_record, new_record)
            elif strategy == ConflictResolutionStrategy.REJECT:
                resolved_record = None  # Reject the new record
            
            if resolved_record:
                self.conflict_stats["conflicts_resolved"] += 1
                logger.debug(f"Conflict resolved for table {table_name}")
            else:
                self.conflict_stats["conflicts_failed"] += 1
                logger.warning(f"Conflict resolution failed for table {table_name}")
            
            return resolved_record
            
        except Exception as e:
            self.conflict_stats["conflicts_failed"] += 1
            logger.error(f"Error resolving conflict for table {table_name}: {e}")
            return None
    
    def _resolve_last_write_wins(self, existing_record: Dict[str, Any], new_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve conflict using last write wins strategy.
        
        Args:
            existing_record: Existing record data
            new_record: New record data
            
        Returns:
            Dict[str, Any]: Resolved record
        """
        # Compare timestamps if available
        existing_timestamp = self._extract_timestamp(existing_record)
        new_timestamp = self._extract_timestamp(new_record)
        
        if existing_timestamp and new_timestamp:
            if new_timestamp > existing_timestamp:
                return new_record
            else:
                return existing_record
        
        # If no timestamps, prefer new record
        return new_record
    
    def _resolve_first_write_wins(self, existing_record: Dict[str, Any], new_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve conflict using first write wins strategy.
        
        Args:
            existing_record: Existing record data
            new_record: New record data
            
        Returns:
            Dict[str, Any]: Resolved record
        """
        # Compare timestamps if available
        existing_timestamp = self._extract_timestamp(existing_record)
        new_timestamp = self._extract_timestamp(new_record)
        
        if existing_timestamp and new_timestamp:
            if existing_timestamp < new_timestamp:
                return existing_record
            else:
                return new_record
        
        # If no timestamps, prefer existing record
        return existing_record
    
    def _resolve_merge(self, existing_record: Dict[str, Any], new_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve conflict using merge strategy.
        
        Args:
            existing_record: Existing record data
            new_record: New record data
            
        Returns:
            Dict[str, Any]: Merged record
        """
        # Start with existing record
        merged_record = existing_record.copy()
        
        # Merge in new record data
        for key, value in new_record.items():
            if key not in merged_record or merged_record[key] is None:
                merged_record[key] = value
            elif isinstance(value, dict) and isinstance(merged_record[key], dict):
                # Recursively merge nested objects
                merged_record[key] = self._resolve_merge(merged_record[key], value)
            elif isinstance(value, list) and isinstance(merged_record[key], list):
                # Merge lists
                merged_record[key] = list(set(merged_record[key] + value))
            else:
                # Use new value for non-null fields
                if value is not None:
                    merged_record[key] = value
        
        # Add merge metadata
        merged_record["_merged_at"] = datetime.now().isoformat()
        merged_record["_merge_source"] = "conflict_resolver"
        
        return merged_record
    
    def _resolve_custom(self, table_name: str, existing_record: Dict[str, Any], new_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Resolve conflict using custom resolver.
        
        Args:
            table_name: Name of the table
            existing_record: Existing record data
            new_record: New record data
            
        Returns:
            Optional[Dict[str, Any]]: Resolved record or None if custom resolver not found
        """
        if table_name not in self.custom_resolvers:
            logger.warning(f"No custom resolver found for table {table_name}")
            return None
        
        try:
            custom_resolver = self.custom_resolvers[table_name]
            resolved_record = custom_resolver(existing_record, new_record)
            return resolved_record
        except Exception as e:
            logger.error(f"Error in custom resolver for table {table_name}: {e}")
            return None
    
    def _extract_timestamp(self, record: Dict[str, Any]) -> Optional[datetime]:
        """
        Extract timestamp from record.
        
        Args:
            record: Record data
            
        Returns:
            Optional[datetime]: Extracted timestamp or None if not found
        """
        timestamp_fields = ['timestamp', 'created_at', 'updated_at', 'modified_at', 'last_modified']
        
        for field in timestamp_fields:
            if field in record:
                try:
                    timestamp_value = record[field]
                    if isinstance(timestamp_value, str):
                        return datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                    elif isinstance(timestamp_value, datetime):
                        return timestamp_value
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def detect_conflict(self, existing_record: Dict[str, Any], new_record: Dict[str, Any], key_fields: List[str]) -> bool:
        """
        Detect if there's a conflict between existing and new records.
        
        Args:
            existing_record: Existing record data
            new_record: New record data
            key_fields: List of key fields to compare
            
        Returns:
            bool: True if conflict detected, False otherwise
        """
        try:
            for field in key_fields:
                if field in existing_record and field in new_record:
                    if existing_record[field] != new_record[field]:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting conflict: {e}")
            return False
    
    def resolve_batch_conflicts(self, table_name: str, conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve a batch of conflicts.
        
        Args:
            table_name: Name of the table
            conflicts: List of conflict data
            
        Returns:
            List[Dict[str, Any]]: List of resolved records
        """
        logger.info(f"Resolving batch of {len(conflicts)} conflicts for table {table_name}")
        
        resolved_records = []
        
        for conflict in conflicts:
            try:
                existing_record = conflict.get("existing_record", {})
                new_record = conflict.get("new_record", {})
                
                resolved_record = self.resolve_conflict(table_name, existing_record, new_record)
                if resolved_record:
                    resolved_records.append(resolved_record)
                    
            except Exception as e:
                logger.error(f"Error resolving batch conflict: {e}")
                continue
        
        logger.info(f"Resolved {len(resolved_records)} out of {len(conflicts)} conflicts")
        return resolved_records
    
    def get_conflict_stats(self) -> Dict[str, Any]:
        """
        Get conflict resolution statistics.
        
        Returns:
            Dict[str, Any]: Conflict resolution statistics
        """
        current_time = datetime.now()
        processing_time = (current_time - self.conflict_stats["start_time"]).total_seconds()
        
        stats = self.conflict_stats.copy()
        stats["processing_time_seconds"] = processing_time
        stats["conflicts_per_second"] = stats["conflicts_resolved"] / processing_time if processing_time > 0 else 0
        stats["resolution_rate"] = stats["conflicts_resolved"] / stats["conflicts_detected"] if stats["conflicts_detected"] > 0 else 0
        stats["current_time"] = current_time.isoformat()
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset conflict resolution statistics."""
        self.conflict_stats = {
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "conflicts_failed": 0,
            "start_time": datetime.now()
        }
        logger.info("Conflict resolution statistics reset")
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.
        
        Returns:
            Dict[str, Any]: Ingestion statistics
        """
        return {
            "resolution_strategies_count": len(self.resolution_strategies),
            "custom_resolvers_count": len(self.custom_resolvers),
            "conflict_stats": self.get_conflict_stats(),
            "ingestion_timestamp": datetime.now().isoformat()
        }
